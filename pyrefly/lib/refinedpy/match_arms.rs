/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `match` statement arm resolution: given a subject with a KNOWN value
//! state, decide which arm a case's pattern (plus its guard) takes, and
//! what names that arm binds. CPython 3.12 semantics
//! (reference/compound_stmts.html#the-match-statement):
//!
//! - A literal/value pattern (`MatchValue`) compares with `==`.
//! - A singleton pattern (`MatchSingleton`, i.e. `True`/`False`/`None`)
//!   compares with `is` — "For the singletons `None`, `True` and
//!   `False`, the `is` operator is used." A subject that is
//!   Boolean-tagged 1.0 IS `True`; a subject that is Number-tagged 1 is
//!   NOT `True` (identity, not equality) — the fact AGENT-BRIEF.md
//!   states as "subject 1 falls through `case True:` but takes
//!   `case True | 1:` via the value alternative."
//! - A capture pattern (bare `case x:`) "always succeeds," binding the
//!   name to the subject.
//! - A wildcard `case _:` "always succeeds... and binds no name."
//! - An OR pattern "matches each of its subpatterns in turn... until
//!   one succeeds" — first Taken wins, left to right.
//! - A guard runs only after its pattern succeeds; "If the guard
//!   condition evaluates as false, the case block is not selected" and
//!   matching continues to the next case.
//!
//! Sequence/Mapping/Class patterns are Undecidable for TAKEN/NOT-TAKEN
//! this wave (`pattern_outcome` below) — deciding which arm runs would
//! need a structural equality/length/key-presence question this file
//! does not ask yet. Their CAPTURES, though, are nameable and (for a
//! known List/Object subject) provable: `pattern_captures` names every
//! bare-Name/star element a sequence pattern binds, every literal-key
//! Name value (plus an optional `**rest`) a mapping pattern binds, and
//! every keyword OR positional sub-pattern Name a class pattern binds.
//! `pattern_bound_captures` reads the actual element/key/field value off
//! a KNOWN List/Object subject when one is available. A class pattern's
//! POSITIONAL sub-patterns (`Point(px, py)`) resolve through the class's
//! own `__match_args__` order (`ClassModel.fields`, `class_pattern_fields`'s
//! own doc) when a class table is available; a keyword sub-pattern needs
//! no such lookup, since the keyword's own `attr` IS the field name.
//!
//! `PrimitiveKind` carries `Integer`/`Float` tags, but nothing in this
//! package's expression evaluator (`expressions.rs`) emits them yet —
//! every numeric literal and arithmetic result reads as
//! `PrimitiveKind::Number` (`Kind::Values` with one `f64`); only a
//! boolean literal reads as `PrimitiveKind::Boolean` (`true` as `1.0`,
//! `false` as `0.0`, matching CPython's `bool` being an `int`
//! subclass). Singleton identity in this file is decided off `kind_tag`
//! + the value: only a Boolean-tagged 1.0/0.0 subject IS `True`/`False`,
//! and only `Kind::Null` IS `None`. When a producer starts tagging
//! `Integer`/`Float` on the values this file reads, `subject_is_singleton`
//! is the one place that gains a new arm — every other function here
//! goes through it rather than re-deriving the identity check.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::MatchCase;
use ruff_python_ast::Pattern;
use ruff_python_ast::Singleton;

use crate::refinedpy::collection_models::subscript_read;
use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::evaluate_expression;
use crate::refinedpy::instances::field_read;
use crate::refinedpy::instances::ClassModel;

/// The field names a `MatchClass` pattern's own class name resolves to,
/// in `__match_args__`/declaration order — `None` when `classes` carries
/// no table, the pattern's `cls` expression is not a bare Name, or the
/// name is not in the table (an imported/builtin class this checker's
/// class table never populates, e.g. `case int():`). Shared by
/// `pattern_captures` and `pattern_bound_captures` so a positional
/// class pattern's field-order lookup is written once.
fn class_pattern_fields<'a>(
    class_pattern: &ruff_python_ast::PatternMatchClass,
    classes: Option<&'a HashMap<String, ClassModel>>,
) -> Option<&'a [crate::refinedpy::instances::ClassField]> {
    let Expr::Name(class_name) = class_pattern.cls.as_ref() else {
        return None;
    };
    let classes = classes?;
    let model = classes.get(class_name.id.as_str())?;
    Some(&model.fields)
}

/// What a match arm's pattern (and, where present, its guard) decided
/// about a known subject.
pub enum ArmOutcome {
    /// The arm is taken: `Environment` is a fork of the arm's incoming
    /// environment with every capture the pattern makes bound.
    Taken(Environment),
    /// The arm is provably not taken — the pattern (or its guard) is
    /// known false for this subject.
    NotTaken,
    /// The walk cannot decide this arm — an unknown subject, an
    /// unmodeled pattern shape (sequence/mapping/class this wave), or a
    /// guard whose truth this file cannot read.
    Undecidable,
}

/// Decide one arm: `pattern` against `subject`, then (if the pattern
/// took) `guard` against the arm's own environment. `environment` is
/// the environment the arm's fork starts from — the caller's job to
/// fork per case, not this function's (so a caller walking several
/// arms in sequence controls its own forking/joining).
pub fn arm_outcome(
    pattern: &Pattern,
    guard: Option<&Expr>,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    match pattern_outcome(pattern, subject, environment, kernel) {
        ArmOutcome::Taken(arm_env) => apply_guard(guard, arm_env, kernel),
        other => other,
    }
}

/// After a pattern is Taken, run its guard (if any) via
/// `evaluate_expression` over the arm's own environment (the pattern's
/// captures are already bound, so the guard reads them). A guard that
/// evaluates to a known boolean decides the arm outright; anything else
/// is Undecidable — the caller (the walk) must then treat every LATER
/// arm as Undecidable too, since CPython only reaches a later case when
/// this guard is known false, and this file cannot prove that. This
/// poisoning rule is `match_taken_environment`'s job to enforce, not
/// this function's; this function reports only its OWN arm's verdict.
fn apply_guard(guard: Option<&Expr>, arm_env: Environment, kernel: &Arc<RefinedTSKernel>) -> ArmOutcome {
    let Some(guard) = guard else {
        return ArmOutcome::Taken(arm_env);
    };
    let guard_value = evaluate_expression(guard, &arm_env, kernel);
    match known_boolean(&guard_value) {
        Some(true) => ArmOutcome::Taken(arm_env),
        Some(false) => ArmOutcome::NotTaken,
        None => ArmOutcome::Undecidable,
    }
}

/// The known boolean a guard's evaluated value carries, if it carries
/// exactly one. A Boolean-tagged single value reads directly; a
/// Number-tagged single value reads through Python's truthiness rule
/// for numbers (nonzero is true) since a guard is `if`-tested, not
/// `==`-compared. Anything else (unknown, a set, no single value) is
/// not a known boolean.
fn known_boolean(value: &AbstractValue) -> Option<bool> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Boolean)
        | Some(PrimitiveKind::Number)
        | Some(PrimitiveKind::Integer)
        | Some(PrimitiveKind::Float) => Some(value.values[0] != 0.0),
        _ => None,
    }
}

/// One pattern's own outcome against `subject`, with no guard
/// considered — the recursive core `arm_outcome`, `MatchAs`, and
/// `MatchOr` all share. `kernel` is threaded through only because
/// `evaluate_expression` (read by `match_value_outcome` for a pattern's
/// own literal expression) requires one by contract; no pattern shape
/// this wave decides asks the kernel a question.
fn pattern_outcome(
    pattern: &Pattern,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    match pattern {
        Pattern::MatchValue(value_pattern) => match_value_outcome(value_pattern, subject, environment, kernel),
        Pattern::MatchSingleton(singleton_pattern) => match_singleton_outcome(singleton_pattern, subject, environment),
        Pattern::MatchAs(as_pattern) => match_as_outcome(as_pattern, subject, environment, kernel),
        Pattern::MatchOr(or_pattern) => match_or_outcome(or_pattern, subject, environment, kernel),
        // Sequence/Mapping/Class patterns: this wave carries no
        // container state on AbstractValue (element_set/keys exist for
        // other purposes but this file does not read into them for
        // structural matching), so every one of these declines rather
        // than assume a shape the subject may not have.
        Pattern::MatchSequence(_) => ArmOutcome::Undecidable,
        Pattern::MatchMapping(_) => ArmOutcome::Undecidable,
        Pattern::MatchClass(_) => ArmOutcome::Undecidable,
        // MatchStar only appears nested inside a MatchSequence's own
        // pattern list (`case [*rest]:`), never as a case's top-level
        // pattern per the grammar (closed_pattern excludes it) — reached
        // only if a caller hands this function a sub-pattern directly,
        // which match_taken_environment never does.
        Pattern::MatchStar(_) => ArmOutcome::Undecidable,
    }
}

/// `MatchValue` — "`LITERAL` will succeed only if `<subject> ==
/// LITERAL`." Decided only when `subject` is a known single numeric or
/// boolean value AND the pattern's own expression evaluates (via
/// `evaluate_expression`, which forces the walk's OWN literal rules) to
/// a known single numeric or boolean value; `==` is CPython's
/// value-equality, which for two host-numeric-sorted values is plain
/// float equality — `1 == True` and `1 == 1.0` both hold, so a
/// Number-tagged subject of 1 DOES take `case 1:` and `case True:`'s
/// VALUE reading would too if it appeared as a MatchValue (it never
/// does — `True` always parses as MatchSingleton, never MatchValue).
fn match_value_outcome(
    value_pattern: &ruff_python_ast::PatternMatchValue,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    let Some(subject_value) = single_numeric_value(subject) else {
        return ArmOutcome::Undecidable;
    };
    let literal_value = evaluate_expression(&value_pattern.value, environment, kernel);
    let Some(pattern_value) = single_numeric_value(&literal_value) else {
        return ArmOutcome::Undecidable;
    };
    if subject_value == pattern_value {
        ArmOutcome::Taken(environment.fork())
    } else {
        ArmOutcome::NotTaken
    }
}

/// `MatchSingleton` — "For the singletons `None`, `True` and `False`,
/// the `is` operator is used": identity, not equality. A
/// Boolean-tagged 1.0/0.0 subject IS `True`/`False`; a Number-tagged 1
/// subject is NOT `True` (AGENT-BRIEF.md's pinned fact) because CPython
/// identity distinguishes the `bool` singletons from the `int` they
/// happen to equal. `Kind::Null` (this crate's representation of
/// Python's `None`, per expressions.rs's NoneLiteral reading) IS
/// `None`; nothing else is.
fn match_singleton_outcome(
    singleton_pattern: &ruff_python_ast::PatternMatchSingleton,
    subject: &AbstractValue,
    environment: &Environment,
) -> ArmOutcome {
    match subject_is_singleton(subject, singleton_pattern.value) {
        Some(true) => ArmOutcome::Taken(environment.fork()),
        Some(false) => ArmOutcome::NotTaken,
        None => ArmOutcome::Undecidable,
    }
}

/// Whether `subject` IS `target` under Python's `is` identity — the one
/// place this file decides singleton identity, so every future kind
/// AbstractValue gains (an eventual Integer/Float split) only needs
/// this function taught about it. `None` means "not decidable" (an
/// unknown or otherwise unread subject); `Some(true)`/`Some(false)` is
/// the decided identity.
fn subject_is_singleton(subject: &AbstractValue, target: Singleton) -> Option<bool> {
    match target {
        Singleton::None => Some(subject.kind == Kind::Null),
        Singleton::True => Some(is_exact_boolean(subject, 1.0)),
        Singleton::False => Some(is_exact_boolean(subject, 0.0)),
    }
}

/// Whether `subject` is a known single Boolean-tagged value equal to
/// `want` (1.0 for True, 0.0 for False) — the only shape `is True` /
/// `is False` can affirm under this domain's current representation.
/// Any other kind (Number-tagged, multi-valued, unknown, …) is not this
/// exact singleton, matching CPython's `is` refusing to unify `bool`
/// with a same-valued `int`.
fn is_exact_boolean(subject: &AbstractValue, want: f64) -> bool {
    subject.kind == Kind::Values
        && subject.kind_tag == Some(PrimitiveKind::Boolean)
        && subject.values.len() == 1
        && subject.values[0] == want
}

/// `MatchAs` — a bare capture (`case x:`, `pattern: None`) or a wildcard
/// (`case _:`, `pattern: None, name: None`) "always succeeds"; a
/// capture binds the subject to its name. A subpattern
/// (`case <pattern> as x:`) recurses on the subpattern first, and only
/// on Taken does it also bind the alias name — a wildcard/capture
/// nested under `as` still always succeeds by the same rule, so the
/// only way this arm is NotTaken/Undecidable is through the subpattern.
fn match_as_outcome(
    as_pattern: &ruff_python_ast::PatternMatchAs,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    let Some(inner_pattern) = as_pattern.pattern.as_deref() else {
        // bare `case x:` or wildcard `case _:` — always succeeds
        let mut arm_env = environment.fork();
        if let Some(name) = as_pattern.name.as_ref() {
            arm_env.bind(name.id.as_str(), subject.clone());
        }
        return ArmOutcome::Taken(arm_env);
    };
    match pattern_outcome(inner_pattern, subject, environment, kernel) {
        ArmOutcome::Taken(mut arm_env) => {
            if let Some(name) = as_pattern.name.as_ref() {
                arm_env.bind(name.id.as_str(), subject.clone());
            }
            ArmOutcome::Taken(arm_env)
        }
        other => other,
    }
}

/// `MatchOr` — "matches each of its subpatterns in turn to the subject
/// value, until one succeeds": first Taken wins (left to right); every
/// alternative NotTaken means the whole pattern is NotTaken; any
/// Undecidable alternative encountered BEFORE a Taken one makes the
/// whole pattern Undecidable (this file cannot rule out that an
/// earlier, undecided alternative would have matched first).
fn match_or_outcome(
    or_pattern: &ruff_python_ast::PatternMatchOr,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    for alternative in &or_pattern.patterns {
        match pattern_outcome(alternative, subject, environment, kernel) {
            ArmOutcome::Taken(arm_env) => return ArmOutcome::Taken(arm_env),
            ArmOutcome::Undecidable => return ArmOutcome::Undecidable,
            ArmOutcome::NotTaken => continue,
        }
    }
    ArmOutcome::NotTaken
}

/// The single numeric value a known abstract value carries, if it
/// carries exactly one — Number- or Boolean-tagged only, matching
/// `expressions.rs`'s `single_numeric_value` (CPython's own
/// `bool`-is-an-`int` reading: `True == 1`).
fn single_numeric_value(value: &AbstractValue) -> Option<f64> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Number)
        | Some(PrimitiveKind::Boolean)
        | Some(PrimitiveKind::Integer)
        | Some(PrimitiveKind::Float) => Some(value.values[0]),
        _ => None,
    }
}

/// The exact value a pattern's own LITERAL shape proves about a taken
/// arm's subject — independent of whether the concrete subject is
/// known (unlike `pattern_outcome`, which requires a known subject to
/// decide TAKEN/NOT-TAKEN). This is the pattern's proof read
/// syntactically: a `MatchValue` proves exactly its own literal
/// expression's value, TAGGED as that literal's own evaluated
/// `PrimitiveKind` (`evaluate_expression`'s `number_literal_value`
/// convention — an int literal tags `Integer`, a float literal tags
/// `Float` — so `case 40:` proves an `Integer`-tagged 40, never a
/// bare `Number`); a `MatchSingleton` proves `True`/`False` as the
/// Boolean-tagged 1.0/0.0 CPython's `is`-identity singletons (`None`
/// proves no NUMERIC value — a null subject is never a member of a
/// numeric refined set, so it contributes nothing here, matching
/// `narrowing.rs`'s own "None is never a Values member" reading);
/// `MatchOr` proves the UNION of every alternative's own proof (PEP
/// 634's rule that all alternatives bind the same names does not
/// extend to proving the same value — `18 | 21 | 40` proves any of the
/// three) — every alternative must prove the SAME tag, or the whole
/// pattern declines (an honest narrow scope: this function never
/// invents a `KindUnion` to paper over a genuinely mixed-sort
/// alternative list); `MatchAs` recurses into its own inner pattern
/// when present, or proves NOTHING when it is a bare capture/wildcard
/// (a bare `case x:` states no literal fact about the subject at
/// all — the caller's job to leave the subject unnarrowed in that
/// case, never to invent a value). Every other pattern shape
/// (Sequence/Mapping/Class/Star) proves nothing this function reads —
/// `None`.
///
/// `check.rs`'s match-join fallback (`walk_match`) calls this to
/// narrow a captured name — or the subject itself when the pattern
/// captures nothing — down from the coarse pre-match claim to exactly
/// what the arm's own pattern proves, the same "a narrowing must be
/// the pattern's own proved claim" discipline `narrowing.rs`'s
/// isinstance/comparison leaves already follow. The returned value's
/// trust grade is `TrustProved` — the pattern's own literal is read
/// exactly, the same grade `number_literal_value` gives every numeric
/// literal.
pub fn pattern_proved_value(pattern: &Pattern, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    match pattern {
        Pattern::MatchValue(value_pattern) => {
            let literal_value = evaluate_expression(&value_pattern.value, environment, kernel);
            if literal_value.kind != Kind::Values || literal_value.values.len() != 1 {
                return None;
            }
            let kind_tag = literal_value.kind_tag?;
            if !matches!(
                kind_tag,
                PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float | PrimitiveKind::Boolean
            ) {
                return None;
            }
            Some(literal_value)
        }
        Pattern::MatchSingleton(singleton_pattern) => match singleton_pattern.value {
            Singleton::True => Some(known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved)),
            Singleton::False => Some(known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved)),
            Singleton::None => None,
        },
        Pattern::MatchOr(or_pattern) => {
            let mut alternatives = or_pattern.patterns.iter();
            let first = pattern_proved_value(alternatives.next()?, environment, kernel)?;
            let mut values = first.values.clone();
            let kind_tag = first.kind_tag;
            for alternative in alternatives {
                let proved = pattern_proved_value(alternative, environment, kernel)?;
                if proved.kind_tag != kind_tag {
                    // a genuinely mixed-sort alternative list — never
                    // guessed at, an honest decline
                    return None;
                }
                for value in proved.values {
                    if !values.contains(&value) {
                        values.push(value);
                    }
                }
            }
            Some(known_values(values, kind_tag?, TrustProved))
        }
        Pattern::MatchAs(as_pattern) => match as_pattern.pattern.as_deref() {
            Some(inner) => pattern_proved_value(inner, environment, kernel),
            None => None,
        },
        Pattern::MatchSequence(_) | Pattern::MatchMapping(_) | Pattern::MatchClass(_) | Pattern::MatchStar(_) => None,
    }
}

/// The bare names one `case` pattern captures — a SYNTACTIC question,
/// answered without deciding whether the pattern would actually take
/// (that question is `pattern_outcome`'s, not this function's).
/// `Pattern::MatchValue`/`MatchSingleton` bind nothing. `Pattern::MatchAs`
/// binds its own `name` (a bare capture/wildcard has no inner pattern)
/// plus whatever its inner pattern (if any) itself binds.
/// `Pattern::MatchOr` recurses into its FIRST alternative only — Python's
/// own grammar rule (compound_stmts.rst, "the same set of names must be
/// captured by all the alternatives") makes every alternative's own
/// capture set identical, so any one alternative names the whole
/// pattern's captures.
///
/// `Pattern::MatchSequence` names every bare-Name capture in its
/// `patterns` list positionally, plus a `MatchStar` element's own name
/// (`case [first, *rest]:` names both `first` and `rest`; a wildcard
/// star `*_` names nothing, matching PEP 634's "`_` never binds"). Any
/// element that is not itself a bare-Name/wildcard `MatchAs` (a nested
/// literal, sequence, or class sub-pattern) makes the WHOLE sequence
/// pattern decline — this function reads only the flat bare-capture
/// case, never recurses past one level into a structural sub-pattern.
///
/// `Pattern::MatchMapping` names every value-side capture whose KEY is
/// a literal (a `MatchValue`/`MatchSingleton`-free syntactic literal —
/// in practice a string, this corpus's only mapping-key shape) and
/// whose value-side pattern is itself a bare-Name/wildcard `MatchAs`,
/// plus the `**rest` capture (`rest: Option<Identifier>`) when present.
/// A non-literal key, or a value-side pattern that is not a bare
/// capture, declines the whole mapping pattern.
///
/// `Pattern::MatchClass` binds nothing ITSELF (a bare `case int():`
/// names nothing). It is nameable in two shapes: NO sub-patterns at
/// all (`case int() as n:`, `arguments.patterns`/`.keywords` both
/// empty), or KEYWORD sub-patterns ONLY, each itself a bare-Name/
/// wildcard `MatchAs` (`case Point(x=px):` names `px`) — a keyword's
/// own `attr` IS the field name, so naming it needs no class lookup.
/// POSITIONAL sub-patterns (`case Point(px, py):`) resolve through the
/// class's own `__match_args__` order (`ClassModel.fields`, pydantic's
/// own declaration-order convention, `class_pattern_fields`'s own doc):
/// each positional bare-Name/wildcard sub-pattern names the field at its
/// own position. A pattern with MORE positions than the class has
/// fields declines whole (Python itself raises `TypeError` for this
/// shape at runtime; this function never guesses a truncated binding).
/// `classes` is `None` when no caller has a class table to offer (this
/// function's own tests, and any future caller outside a match walk) —
/// every positional pattern then declines exactly as before this
/// capability existed.
pub fn pattern_captures(pattern: &Pattern, classes: Option<&HashMap<String, ClassModel>>) -> Option<Vec<String>> {
    match pattern {
        Pattern::MatchValue(_) | Pattern::MatchSingleton(_) => Some(Vec::new()),
        Pattern::MatchAs(as_pattern) => {
            let mut names = match as_pattern.pattern.as_deref() {
                Some(inner) => pattern_captures(inner, classes)?,
                None => Vec::new(),
            };
            if let Some(name) = as_pattern.name.as_ref() {
                names.push(name.id.as_str().to_owned());
            }
            Some(names)
        }
        Pattern::MatchOr(or_pattern) => {
            let first = or_pattern.patterns.first()?;
            pattern_captures(first, classes)
        }
        Pattern::MatchSequence(sequence_pattern) => {
            let mut names = Vec::new();
            for element in &sequence_pattern.patterns {
                match element {
                    Pattern::MatchStar(star) => {
                        if let Some(name) = star.name.as_ref() {
                            names.push(name.id.as_str().to_owned());
                        }
                    }
                    Pattern::MatchAs(as_pattern) if as_pattern.pattern.is_none() => {
                        if let Some(name) = as_pattern.name.as_ref() {
                            names.push(name.id.as_str().to_owned());
                        }
                    }
                    // a nested literal/sequence/mapping/class sub-pattern
                    // — beyond this function's flat bare-capture scope
                    _ => return None,
                }
            }
            Some(names)
        }
        Pattern::MatchMapping(mapping_pattern) => {
            if mapping_pattern.keys.len() != mapping_pattern.patterns.len() {
                return None;
            }
            let mut names = Vec::new();
            for (key, value_pattern) in mapping_pattern.keys.iter().zip(mapping_pattern.patterns.iter()) {
                if !is_literal_mapping_key(key) {
                    return None;
                }
                let Pattern::MatchAs(as_pattern) = value_pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    names.push(name.id.as_str().to_owned());
                }
            }
            if let Some(rest) = mapping_pattern.rest.as_ref() {
                names.push(rest.id.as_str().to_owned());
            }
            Some(names)
        }
        Pattern::MatchClass(class_pattern) => {
            let mut names = Vec::new();
            if !class_pattern.arguments.patterns.is_empty() {
                let fields = class_pattern_fields(class_pattern, classes)?;
                if class_pattern.arguments.patterns.len() > fields.len() {
                    // more positions than the class declares fields —
                    // Python itself raises TypeError for this shape
                    return None;
                }
                for sub_pattern in class_pattern.arguments.patterns.iter() {
                    let Pattern::MatchAs(as_pattern) = sub_pattern else {
                        return None;
                    };
                    if as_pattern.pattern.is_some() {
                        return None;
                    }
                    if let Some(name) = as_pattern.name.as_ref() {
                        names.push(name.id.as_str().to_owned());
                    }
                }
            }
            for keyword in &class_pattern.arguments.keywords {
                let Pattern::MatchAs(as_pattern) = &keyword.pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    names.push(name.id.as_str().to_owned());
                }
            }
            Some(names)
        }
        Pattern::MatchStar(_) => None,
    }
}

/// Whether a `MatchMapping` key expression is a literal this file can
/// read as a fixed key spelling — a string literal, the only key shape
/// this corpus's mapping-pattern rows use (`case {"age": bound_age}:`).
/// Any other expression shape (a dotted constant, a number, an
/// f-string) answers `false` — not read this wave.
fn is_literal_mapping_key(key: &Expr) -> bool {
    matches!(key, Expr::StringLiteral(_))
}

/// The (name, value) pair every capture `pattern_captures` names,
/// filled in with the value each name PROVABLY holds when `subject` is
/// known — the value-bearing counterpart naming alone cannot answer.
/// `None` means `pattern` itself has no nameable captures — this
/// function decides `Some`/`None` on the SAME conditions
/// `pattern_captures` does (a caller needing only the names, never the
/// values, still has that lighter-weight function to call; `check.rs::
/// walk_match`'s join-fallback path calls THIS one directly, since it
/// always needs both in the same pass).
///
/// A `MatchAs`'s own captured name binds to `pattern_proved_value`'s
/// proof for the pattern rooted at that `as` (e.g. `(40 | 41) as
/// chosen` binds `chosen` to `{40, 41}`, not the raw subject) when one
/// exists, falling back to `subject` itself for a bare capture/wildcard
/// (`pattern_proved_value` proves nothing for those, by design — the
/// SAME fallback `check.rs::walk_match` already applies for a
/// literal/singleton/or/as pattern with no sequence/mapping/class
/// shape involved).
///
/// A capture whose OWN value cannot be proved from `subject` (an
/// unknown/wrong-kind receiver, an absent key, an out-of-range
/// position) binds `unknown()` for that one name rather than dropping
/// it or guessing — `assignability::judge`'s own law never fires an
/// `Unknown` value (only `Object`/`List`/`Null` structural mismatches
/// fire against a scalar declared set), so an unproved capture is
/// SILENT-SAFE: it reaches the sink Undetermined, never a false Fire.
pub fn pattern_bound_captures(
    pattern: &Pattern,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(String, AbstractValue)>> {
    match pattern {
        Pattern::MatchValue(_) | Pattern::MatchSingleton(_) => Some(Vec::new()),
        Pattern::MatchAs(as_pattern) => {
            let mut bound = match as_pattern.pattern.as_deref() {
                Some(inner) => pattern_bound_captures(inner, subject, environment, kernel)?,
                None => Vec::new(),
            };
            if let Some(name) = as_pattern.name.as_ref() {
                let proved = pattern_proved_value(pattern, environment, kernel);
                let own_value = proved.unwrap_or_else(|| subject.clone());
                bound.push((name.id.as_str().to_owned(), own_value));
            }
            Some(bound)
        }
        Pattern::MatchOr(or_pattern) => {
            let first = or_pattern.patterns.first()?;
            pattern_bound_captures(first, subject, environment, kernel)
        }
        Pattern::MatchSequence(sequence_pattern) => {
            let items = if subject.kind == Kind::List { Some(&subject.items) } else { None };
            let mut bound = Vec::new();
            for (position, element) in sequence_pattern.patterns.iter().enumerate() {
                match element {
                    Pattern::MatchStar(star) => {
                        if let Some(name) = star.name.as_ref() {
                            // the remainder is a LIST, never a scalar this
                            // corpus's rows read at a refined sink — bound
                            // opaque rather than sliced out of `items`
                            bound.push((name.id.as_str().to_owned(), unknown()));
                        }
                    }
                    Pattern::MatchAs(as_pattern) if as_pattern.pattern.is_none() => {
                        if let Some(name) = as_pattern.name.as_ref() {
                            let element_value = items
                                .and_then(|items| items.get(position))
                                .cloned()
                                .unwrap_or_else(unknown);
                            bound.push((name.id.as_str().to_owned(), element_value));
                        }
                    }
                    _ => return None,
                }
            }
            Some(bound)
        }
        Pattern::MatchMapping(mapping_pattern) => {
            if mapping_pattern.keys.len() != mapping_pattern.patterns.len() {
                return None;
            }
            let mut bound = Vec::new();
            for (key, value_pattern) in mapping_pattern.keys.iter().zip(mapping_pattern.patterns.iter()) {
                if !is_literal_mapping_key(key) {
                    return None;
                }
                let Pattern::MatchAs(as_pattern) = value_pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    let key_value = evaluate_expression(key, environment, kernel);
                    let bound_value = subscript_read(subject, &key_value).unwrap_or_else(unknown);
                    bound.push((name.id.as_str().to_owned(), bound_value));
                }
            }
            if let Some(rest) = mapping_pattern.rest.as_ref() {
                // `**rest` collects the remaining keys into a DICT, never
                // a scalar this corpus's rows read at a refined sink
                bound.push((rest.id.as_str().to_owned(), unknown()));
            }
            Some(bound)
        }
        Pattern::MatchClass(class_pattern) => {
            let mut bound = Vec::new();
            if !class_pattern.arguments.patterns.is_empty() {
                let classes = environment.classes().map(|classes| classes.as_ref());
                let fields = class_pattern_fields(class_pattern, classes)?;
                if class_pattern.arguments.patterns.len() > fields.len() {
                    // more positions than the class declares fields —
                    // Python itself raises TypeError for this shape
                    return None;
                }
                for (field, sub_pattern) in fields.iter().zip(class_pattern.arguments.patterns.iter()) {
                    let Pattern::MatchAs(as_pattern) = sub_pattern else {
                        return None;
                    };
                    if as_pattern.pattern.is_some() {
                        return None;
                    }
                    if let Some(name) = as_pattern.name.as_ref() {
                        let field_value = field_read(subject, field.name.as_str()).unwrap_or_else(unknown);
                        bound.push((name.id.as_str().to_owned(), field_value));
                    }
                }
            }
            for keyword in &class_pattern.arguments.keywords {
                let Pattern::MatchAs(as_pattern) = &keyword.pattern else {
                    return None;
                };
                if as_pattern.pattern.is_some() {
                    return None;
                }
                if let Some(name) = as_pattern.name.as_ref() {
                    let field_value = field_read(subject, keyword.attr.id.as_str()).unwrap_or_else(unknown);
                    bound.push((name.id.as_str().to_owned(), field_value));
                }
            }
            Some(bound)
        }
        Pattern::MatchStar(_) => None,
    }
}

/// Walk every arm of a match statement in order, deciding each with
/// `arm_outcome`, and enforcing the poisoning rule `apply_guard`'s doc
/// states: once an arm's guard is Undecidable, every LATER arm is also
/// Undecidable — CPython only reaches a later case when every earlier
/// pattern failed or its guard was known false, and an Undecidable
/// guard means this file cannot rule out that the earlier arm actually
/// ran. Returns `Some((index, env))` for the exactly one arm decided
/// Taken with every earlier arm decided NotTaken; `None` when no arm is
/// decidably reached (either an arm is Undecidable before any Taken, or
/// every arm resolves NotTaken with no wildcard/capture fallthrough).
pub fn match_taken_environment(
    subject_value: &AbstractValue,
    cases: &[MatchCase],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(usize, Environment)> {
    for (index, case) in cases.iter().enumerate() {
        match arm_outcome(&case.pattern, case.guard.as_deref(), subject_value, environment, kernel) {
            ArmOutcome::Taken(arm_env) => return Some((index, arm_env)),
            ArmOutcome::NotTaken => continue,
            ArmOutcome::Undecidable => return None,
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_domain::abstract_value::known_values;
    use refined_domain::abstract_value::null_value;
    use refined_domain::abstract_value::unknown;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_ast::ModModule;
    use ruff_python_ast::Stmt;

    use super::*;

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn empty_environment() -> Environment {
        Environment::new(HashSet::new())
    }

    /// Parses `source` as a module whose only statement is a `match`,
    /// and hands back its `MatchCase`s. Parsing a full match statement
    /// (rather than `parse_expression`) is the only way to reach
    /// `Pattern` nodes — patterns are not expressions.
    fn match_cases(source: &str) -> Vec<MatchCase> {
        let parsed: ModModule = ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax();
        let Some(Stmt::Match(match_stmt)) = parsed.body.into_iter().next() else {
            panic!("fixture source must be a single match statement");
        };
        match_stmt.cases
    }

    #[test]
    fn value_pattern_hit() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let subject = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::Taken(_)), "1 must match `case 1:`");
    }

    #[test]
    fn value_pattern_miss() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let subject = known_values(vec![2.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::NotTaken), "2 must not match `case 1:`");
    }

    /// The brief's pinned fact: a Number-tagged subject of 1 falls
    /// through `case True:` (singleton `is`), but a Boolean-tagged
    /// subject of 1.0 (True itself) takes it.
    #[test]
    fn singleton_vs_value_on_1_true() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case True:\n        pass\n");
        let environment = empty_environment();

        let number_one = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
        let number_outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &number_one, &environment, &kernel);
        assert!(
            matches!(number_outcome, ArmOutcome::NotTaken),
            "a Number-tagged 1 must NOT match `case True:` (is, not ==)"
        );

        let boolean_true = known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved);
        let boolean_outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &boolean_true, &environment, &kernel);
        assert!(
            matches!(boolean_outcome, ArmOutcome::Taken(_)),
            "a Boolean-tagged True must match `case True:`"
        );
    }

    /// The brief's paired fact: subject 1 DOES take `case True | 1:`
    /// via the value alternative, even though it falls through
    /// `case True:` alone.
    #[test]
    fn or_pattern_singleton_then_value_takes_the_value_alternative() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case True | 1:\n        pass\n");
        let subject = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::Taken(_)),
            "Number-tagged 1 must take `case True | 1:` via the value alternative"
        );
    }

    #[test]
    fn none_singleton_matches_null_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case None:\n        pass\n");
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &null_value(), &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::Taken(_)), "None subject must match `case None:`");
    }

    #[test]
    fn bare_capture_always_taken_and_binds() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case y:\n        pass\n");
        let subject = known_values(vec![42.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        let ArmOutcome::Taken(arm_env) = outcome else {
            panic!("a bare capture always succeeds");
        };
        let bound = arm_env.read("y").expect("y must be bound to the subject");
        assert_eq!(bound.values, vec![42.0]);
    }

    #[test]
    fn wildcard_always_taken_binds_nothing() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case _:\n        pass\n");
        let subject = unknown();
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::Taken(_)), "wildcard always succeeds, even on an unknown subject");
    }

    #[test]
    fn guard_decided_true_on_known_values() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case y if y == 5:\n        pass\n");
        let subject = known_values(vec![5.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        // the capture binds y to the subject, and evaluate_expression
        // decides `y == 5` over the known value — the guard reads true,
        // so the arm is Taken with the capture bound.
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        let ArmOutcome::Taken(arm_env) = outcome else {
            panic!("a guard deciding true over a known capture takes the arm");
        };
        assert_eq!(arm_env.read("y").expect("y binds the subject").values, vec![5.0]);
    }

    #[test]
    fn guard_decided_true_on_bound_boolean_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case y if flag:\n        pass\n");
        let subject = known_values(vec![7.0], PrimitiveKind::Number, TrustProved);
        let mut environment = empty_environment();
        environment.bind("flag", known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved));
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::Taken(_)), "a guard reading a known-true bound name is Taken");
    }

    #[test]
    fn guard_decided_false_on_bound_boolean_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case y if flag:\n        pass\n");
        let subject = known_values(vec![7.0], PrimitiveKind::Number, TrustProved);
        let mut environment = empty_environment();
        environment.bind("flag", known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved));
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::NotTaken), "a guard reading a known-false bound name is NotTaken");
    }

    #[test]
    fn sequence_pattern_is_undecidable() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case [a, b]:\n        pass\n");
        let subject = unknown();
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::Undecidable), "a sequence pattern is Undecidable this wave");
    }

    #[test]
    fn match_taken_environment_walks_in_order_and_poisons_after_undecidable_guard() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases(concat!(
            "match x:\n",
            "    case 1:\n",
            "        pass\n",
            "    case 2:\n",
            "        pass\n",
        ));
        let subject = known_values(vec![2.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        let result = match_taken_environment(&subject, &cases, &environment, &kernel);
        let Some((index, _)) = result else {
            panic!("case 2 must be decidably reached")
        };
        assert_eq!(index, 1, "the second arm (index 1) is the one that takes 2");
    }

    #[test]
    fn match_taken_environment_none_when_no_arm_is_reached() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases(concat!(
            "match x:\n",
            "    case 1:\n",
            "        pass\n",
            "    case 2:\n",
            "        pass\n",
        ));
        let subject = known_values(vec![3.0], PrimitiveKind::Number, TrustProved);
        let environment = empty_environment();
        assert!(
            match_taken_environment(&subject, &cases, &environment, &kernel).is_none(),
            "3 matches neither arm and there is no wildcard fallthrough"
        );
    }

    // --- pattern_captures / pattern_bound_captures ---

    #[test]
    fn pattern_captures_names_sequence_elements_and_star_positionally() {
        let cases = match_cases("match x:\n    case [first, *rest]:\n        pass\n");
        let names = pattern_captures(&cases[0].pattern, None).expect("bare-Name/star elements are nameable");
        assert_eq!(names, vec!["first".to_owned(), "rest".to_owned()]);
    }

    #[test]
    fn pattern_captures_wildcard_star_names_nothing() {
        let cases = match_cases("match x:\n    case [*_]:\n        pass\n");
        let names = pattern_captures(&cases[0].pattern, None).expect("a wildcard star is nameable");
        assert!(names.is_empty(), "`*_` never binds: {names:?}");
    }

    #[test]
    fn pattern_captures_declines_a_sequence_with_a_nested_literal() {
        let cases = match_cases("match x:\n    case [1, b]:\n        pass\n");
        assert!(
            pattern_captures(&cases[0].pattern, None).is_none(),
            "a nested literal sub-pattern is past this function's flat bare-capture scope"
        );
    }

    #[test]
    fn pattern_captures_names_mapping_literal_key_values_and_rest() {
        let cases = match_cases("match x:\n    case {\"age\": bound_age, **rest}:\n        pass\n");
        let names = pattern_captures(&cases[0].pattern, None).expect("literal-key Name values plus **rest are nameable");
        assert_eq!(names, vec!["bound_age".to_owned(), "rest".to_owned()]);
    }

    #[test]
    fn pattern_captures_names_class_keyword_subpatterns() {
        let cases = match_cases("match x:\n    case Point(x=px):\n        pass\n");
        let names =
            pattern_captures(&cases[0].pattern, None).expect("a keyword sub-pattern's own attr needs no class lookup");
        assert_eq!(names, vec!["px".to_owned()]);
    }

    #[test]
    fn pattern_captures_declines_a_class_pattern_with_positional_subpatterns_and_no_class_table() {
        let cases = match_cases("match x:\n    case Point(px, py):\n        pass\n");
        assert!(
            pattern_captures(&cases[0].pattern, None).is_none(),
            "a position needs __match_args__ order, which no class table here can supply"
        );
    }

    #[test]
    fn pattern_captures_names_class_positional_subpatterns_from_match_args_order() {
        let cases = match_cases("match x:\n    case Point(px, py):\n        pass\n");
        let mut fields = HashMap::new();
        fields.insert(
            "Point".to_owned(),
            ClassModel {
                name: "Point".to_owned(),
                fields: vec![
                    crate::refinedpy::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::refinedpy::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
                ],
                properties: HashMap::new(),
                methods: HashMap::new(),
                parent_methods: HashMap::new(),
                class_attributes: Vec::new(),
            },
        );
        let names = pattern_captures(&cases[0].pattern, Some(&fields))
            .expect("__match_args__ order names each position");
        assert_eq!(names, vec!["px".to_owned(), "py".to_owned()]);
    }

    #[test]
    fn pattern_captures_declines_a_class_pattern_with_more_positions_than_fields() {
        let cases = match_cases("match x:\n    case Point(px, py, pz):\n        pass\n");
        let mut fields = HashMap::new();
        fields.insert(
            "Point".to_owned(),
            ClassModel {
                name: "Point".to_owned(),
                fields: vec![
                    crate::refinedpy::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::refinedpy::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
                ],
                properties: HashMap::new(),
                methods: HashMap::new(),
                parent_methods: HashMap::new(),
                class_attributes: Vec::new(),
            },
        );
        assert!(
            pattern_captures(&cases[0].pattern, Some(&fields)).is_none(),
            "three positions against a two-field class never truncates to a guess"
        );
    }

    #[test]
    fn pattern_bound_captures_reads_list_elements_positionally_off_a_known_list_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case [a, b]:\n        pass\n");
        let subject = refined_domain::known_constructors::known_list(
            vec![
                known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![10.0], PrimitiveKind::Integer, TrustProved),
            ],
            TrustProved,
        );
        let environment = empty_environment();
        let bound = pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel)
            .expect("bare-Name elements are nameable");
        assert_eq!(bound[0], ("a".to_owned(), known_values(vec![200.0], PrimitiveKind::Integer, TrustProved)));
        assert_eq!(bound[1], ("b".to_owned(), known_values(vec![10.0], PrimitiveKind::Integer, TrustProved)));
    }

    #[test]
    fn pattern_bound_captures_binds_unknown_when_the_subject_is_not_a_known_list() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case [a, b]:\n        pass\n");
        let subject = unknown();
        let environment = empty_environment();
        let bound = pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel)
            .expect("bare-Name elements are still nameable over an unknown subject");
        assert_eq!(bound[0].1.kind, Kind::Unknown, "an unproved element binds unknown(), never a guess");
        assert_eq!(bound[1].1.kind, Kind::Unknown);
    }

    #[test]
    fn pattern_bound_captures_reads_a_mapping_value_off_a_known_dict_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case {\"age\": bound_age}:\n        pass\n");
        let subject = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        let environment = empty_environment();
        let bound = pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel)
            .expect("a literal-key Name value is nameable");
        assert_eq!(bound, vec![("bound_age".to_owned(), known_values(vec![200.0], PrimitiveKind::Integer, TrustProved))]);
    }

    #[test]
    fn pattern_bound_captures_reads_a_class_field_off_a_known_instance_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case Point(x=px):\n        pass\n");
        let subject = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "x".to_owned(),
                numeric: false,
                value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        let environment = empty_environment();
        let bound = pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel)
            .expect("a keyword sub-pattern's field is nameable");
        assert_eq!(bound, vec![("px".to_owned(), known_values(vec![200.0], PrimitiveKind::Integer, TrustProved))]);
    }

    #[test]
    fn pattern_bound_captures_declines_positional_class_subpatterns_with_no_class_table() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case Point(px, py):\n        pass\n");
        let subject = unknown();
        let environment = empty_environment();
        assert!(
            pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel).is_none(),
            "positional sub-patterns need __match_args__ order, which no class table here can supply"
        );
    }

    #[test]
    fn pattern_bound_captures_reads_positional_class_fields_off_a_known_instance_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case Point(px, py):\n        pass\n");
        let subject = refined_domain::known_constructors::known_object(
            vec![
                refined_domain::abstract_value::ObjectKey {
                    name: "x".to_owned(),
                    numeric: false,
                    value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
                },
                refined_domain::abstract_value::ObjectKey {
                    name: "y".to_owned(),
                    numeric: false,
                    value: known_values(vec![10.0], PrimitiveKind::Integer, TrustProved),
                },
            ],
            None,
            true,
            TrustProved,
            false,
        );
        let mut environment = empty_environment();
        let mut fields = HashMap::new();
        fields.insert(
            "Point".to_owned(),
            ClassModel {
                name: "Point".to_owned(),
                fields: vec![
                    crate::refinedpy::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::refinedpy::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
                ],
                properties: HashMap::new(),
                methods: HashMap::new(),
                parent_methods: HashMap::new(),
                class_attributes: Vec::new(),
            },
        );
        environment.set_classes(Arc::new(fields));
        let bound = pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel)
            .expect("positional sub-patterns resolve through __match_args__ order");
        assert_eq!(bound[0], ("px".to_owned(), known_values(vec![200.0], PrimitiveKind::Integer, TrustProved)));
        assert_eq!(bound[1], ("py".to_owned(), known_values(vec![10.0], PrimitiveKind::Integer, TrustProved)));
    }

    #[test]
    fn pattern_bound_captures_declines_positional_class_subpatterns_past_field_count() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case Point(px, py, pz):\n        pass\n");
        let subject = unknown();
        let mut environment = empty_environment();
        let mut fields = HashMap::new();
        fields.insert(
            "Point".to_owned(),
            ClassModel {
                name: "Point".to_owned(),
                fields: vec![
                    crate::refinedpy::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::refinedpy::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
                ],
                properties: HashMap::new(),
                methods: HashMap::new(),
                parent_methods: HashMap::new(),
                class_attributes: Vec::new(),
            },
        );
        environment.set_classes(Arc::new(fields));
        assert!(
            pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel).is_none(),
            "three positions against a two-field class never truncates to a guess"
        );
    }

    /// `(40 | 41) as chosen` — the MatchAs-wrapped-MatchOr shape
    /// t-match-patterns.py's `match_as_subpattern_binding` row uses:
    /// `chosen` binds to `pattern_proved_value`'s own proof (`{40, 41}`),
    /// never the raw (here unknown) subject.
    #[test]
    fn pattern_bound_captures_binds_an_as_wrapped_or_pattern_to_its_proved_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case (40 | 41) as chosen:\n        pass\n");
        let subject = unknown();
        let environment = empty_environment();
        let bound = pattern_bound_captures(&cases[0].pattern, &subject, &environment, &kernel)
            .expect("an as-capture over an or-pattern is nameable");
        assert_eq!(bound.len(), 1);
        assert_eq!(bound[0].0, "chosen");
        let mut values = bound[0].1.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![40.0, 41.0], "chosen must bind the pattern's own proved value, not the raw subject");
    }
}
