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
//!
//! A `MatchValue`/`MatchOr` subject is not always one known scalar.
//! `enumerable_subject_members` reads the admitted numeric members off
//! THREE subject shapes: a multi-valued `Kind::Values` (`{1, 2, 4}`
//! read directly off `subject.values`); a `Kind::Set` that enumerates a
//! union-of-singleton-scalars form (`scalars_of_union_of_singletons`,
//! `collection_models.rs`'s own reader for exactly this shape, reused
//! rather than re-parsed); and, per arm, a `Kind::KindUnion`'s own
//! Values-kind arms. `match_value_outcome` then asks MEMBERSHIP rather
//! than the single-value equality it used to: a pattern literal that IS
//! a member is Taken, one that is NOT is NotTaken (a dead arm — the
//! same NotTaken every other unreachable arm answers, never a new
//! label), and a subject this reading cannot enumerate stays
//! Undecidable exactly as before. `pattern_outcome`'s own `Kind::KindUnion`
//! arm judges the pattern against EACH arm through this same recursive
//! core (mirroring `assignability.rs`'s KindUnion judge: a Fire/Taken
//! arm decides, an Undetermined/Undecidable arm poisons the whole
//! union, and the union is NotTaken only when every arm is) — the same
//! "apply per arm, keep what the pattern admits" reading
//! `narrow_isinstance_call`'s own KindUnion filter (`narrowing.rs`)
//! already uses for `isinstance`, applied here to `match`.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::min_trust_level;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::MatchCase;
use ruff_python_ast::Pattern;
use ruff_python_ast::Singleton;
use ruff_python_ast::Stmt;

use crate::collection_models::subscript_read;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances::field_read;
use crate::instances::ClassModel;

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
) -> Option<&'a [crate::instances::ClassField]> {
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
    if subject.kind == Kind::KindUnion {
        return kind_union_pattern_outcome(pattern, subject, environment, kernel);
    }
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

/// A `Kind::KindUnion` subject (`json.loads`'s own honest return-space
/// union, `expressions.rs::json_loads_value_space`, is the one producer
/// today) — the pattern is judged against EACH arm through this same
/// recursive `pattern_outcome` core, the per-arm reading
/// `assignability.rs`'s KindUnion judge and `narrowing.rs`'s
/// `narrow_isinstance_call` KindUnion filter both already use for their
/// own questions, applied here to a match pattern's TAKEN/NOT-TAKEN
/// question instead of an assignability Fire or an isinstance filter.
/// The union claims the runtime subject is SOME arm, never which one:
/// an arm the pattern proves Undecidable makes the whole union
/// Undecidable (this file cannot rule out that arm being the real
/// runtime value); with no Undecidable arm, ANY arm the pattern takes
/// makes the union Taken (that arm's own environment is a sound fork —
/// CPython runs the arm whenever the runtime value happens to be that
/// arm, so the union subject can genuinely reach this case); every arm
/// NotTaken is the whole union NotTaken (no possible runtime shape ever
/// reaches this pattern).
fn kind_union_pattern_outcome(
    pattern: &Pattern,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    for arm in &subject.arms {
        if let ArmOutcome::Undecidable = pattern_outcome(pattern, arm, environment, kernel) {
            return ArmOutcome::Undecidable;
        }
    }
    for arm in &subject.arms {
        if let ArmOutcome::Taken(arm_env) = pattern_outcome(pattern, arm, environment, kernel) {
            return ArmOutcome::Taken(arm_env);
        }
    }
    ArmOutcome::NotTaken
}

/// `MatchValue` — "`LITERAL` will succeed only if `<subject> ==
/// LITERAL`." Decided when `subject` ENUMERATES its admitted numeric
/// members (`enumerable_numeric_members` — a single known scalar, a
/// multi-valued `Kind::Values` set such as `{1, 2, 4}`, or a `Kind::Set`
/// that enumerates a union-of-singletons form) and the pattern's own
/// evaluated expression (via `evaluate_expression`, which forces the
/// walk's OWN literal rules) is a known single numeric/boolean value —
/// or BOTH sides are known exact strings — the two `==` rows CPython's
/// own equality actually reaches for a `MatchValue` pattern
/// (expressions.rst, "Comparisons": numeric types compare by
/// mathematical value, strings compare by their code-point sequence).
/// Numeric: `1 == True` and `1 == 1.0` both hold, so a Number-tagged
/// subject admitting 1 DOES take `case 1:` and `case True:`'s VALUE
/// reading would too if it appeared as a MatchValue (it never does —
/// `True` always parses as MatchSingleton, never MatchValue). The
/// pattern's own literal being a MEMBER of the subject's admitted set
/// is Taken; not a member is NotTaken (a dead arm no runtime value of
/// this subject can ever reach — the same NotTaken every other
/// unreachable arm answers). String: `case "left":` against a
/// String-tagged subject compares the code-point vectors directly, the
/// same reading `expressions.rs::exact_string_values` gives an ordinary
/// `==` comparison — this is what lets `anchor_of`'s own `match o: case
/// "left": ...` decide its arm for a concrete `Literal["left", ...]`
/// argument instead of falling through to the undecided join over every
/// arm. A subject/pattern pair that is neither both-numeric nor
/// both-string (or one side unknown, or the subject's set does not
/// enumerate) is Undecidable.
fn match_value_outcome(
    value_pattern: &ruff_python_ast::PatternMatchValue,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    let literal_value = evaluate_expression(&value_pattern.value, environment, kernel);
    if let (Some(subject_members), Some(pattern_value)) =
        (enumerable_numeric_members(subject), single_numeric_value(&literal_value))
    {
        return if subject_members.iter().any(|member| *member == pattern_value) {
            ArmOutcome::Taken(environment.fork())
        } else {
            ArmOutcome::NotTaken
        };
    }
    if let (Some(subject_text), Some(pattern_text)) =
        (exact_string_values(subject), exact_string_values(&literal_value))
    {
        return if subject_text == pattern_text {
            ArmOutcome::Taken(environment.fork())
        } else {
            ArmOutcome::NotTaken
        };
    }
    // `None`/a dict/a list is a STRUCTURAL SORT MISMATCH against any
    // scalar (numeric or string) `MatchValue` literal — neither is ever
    // `==` a number or a string, the same "never a member of a scalar
    // set" reading `assignability.rs::judge`'s own Null/Object/List
    // rows give a declared scalar refinement. This is a definite
    // NotTaken, not Undecidable: it is what lets a `Kind::KindUnion`'s
    // own None/list/dict arms (`kind_union_pattern_outcome`) drop out
    // of a numeric pattern's union judgment instead of poisoning it.
    if is_structural_scalar_mismatch(subject) && (single_numeric_value(&literal_value).is_some() || exact_string_values(&literal_value).is_some())
    {
        return ArmOutcome::NotTaken;
    }
    ArmOutcome::Undecidable
}

/// Whether `subject` is a KNOWN kind that can never be `==` a scalar
/// (a number or a string) — `Kind::Null` (`None`), `Kind::Object` (a
/// dict), and `Kind::List` (a list/tuple). The same three kinds
/// `assignability.rs::judge` fires outright against a declared scalar
/// refinement, read here for a `MatchValue` pattern's own `==` question
/// instead of an assignability question.
fn is_structural_scalar_mismatch(subject: &AbstractValue) -> bool {
    matches!(subject.kind, Kind::Null | Kind::Object | Kind::List)
}

/// The admitted numeric members a subject enumerates, if it enumerates
/// any this file can read — the membership-question counterpart
/// `match_value_outcome` asks instead of the plain single-value
/// equality `single_numeric_value` alone can answer. Three shapes:
///
/// - `Kind::Values` (Number/Boolean/Integer/Float-tagged): its own
///   `values` directly — a single known scalar is the `len() == 1`
///   case already handled before this function existed; a
///   multi-valued binding (`{1, 2, 4}`, an ordinary join of several
///   known values — `lattice_operations.rs::join_known`'s same-sort
///   arm) enumerates every value it carries.
/// - `Kind::Set` that enumerates a union-of-singleton-scalars form —
///   `collection_models.rs::scalars_of_union_of_singletons`, reused
///   here rather than re-parsed, the same reader
///   `known_value_of_state` uses to read a kernel-joined dict value
///   back to exact values. A set that does NOT enumerate (a range, a
///   star, a multi-codepoint string tuple) answers `None` — this
///   function never guesses at values that are not actually
///   enumerated.
/// - `Kind::KindUnion` is read one level up, in
///   `kind_union_pattern_outcome` — a union asks per-arm, not through
///   this flat membership list, since an Undecidable arm must poison
///   the whole judgment rather than silently drop out of a merged
///   member list.
fn enumerable_numeric_members(subject: &AbstractValue) -> Option<Vec<f64>> {
    if subject.kind == Kind::Values {
        return match subject.kind_tag {
            Some(PrimitiveKind::Number)
            | Some(PrimitiveKind::Boolean)
            | Some(PrimitiveKind::Integer)
            | Some(PrimitiveKind::Float) => Some(subject.values.clone()),
            _ => None,
        };
    }
    if subject.kind == Kind::Set {
        return crate::collection_models::scalars_of_union_of_singletons(&subject.set);
    }
    None
}

/// One `case` pattern's own flat list of numeric literals — every value
/// a `MatchValue`/`MatchOr`-of-numerics/`MatchAs`-wrapping-one names,
/// read via `pattern_proved_value` and unwrapped back to its bare
/// `Vec<f64>` (this function drops the tag/grade `pattern_proved_value`
/// carries, since the two callers below fold the result against a
/// SUBJECT's own tag, never the pattern's). `None` for a pattern
/// `pattern_proved_value` itself does not prove a value for (a bare
/// capture/wildcard, a singleton `None`, a sequence/mapping/class
/// pattern) — the same declines, read through the one existing proof
/// function rather than re-deriving them.
fn pattern_literal_members(pattern: &Pattern, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<Vec<f64>> {
    pattern_proved_value(pattern, environment, kernel).map(|proved| proved.values)
}

/// A decidable scalar subject's own narrowed value after ONE arm's
/// pattern decides TAKEN or NOT-TAKEN against it — the intersection/
/// difference pair `narrowing.rs`'s own isinstance/comparison leaves
/// already spell for a Values binding (`narrow_name_against_literal`'s
/// `filter` by a kept predicate, `narrow_isinstance_call`'s KindUnion
/// `filter` by tag match), applied here to a match arm's own admitted
/// members instead of a comparison/isinstance test:
///
/// - TAKEN (`keep_matched` true): the arm's own environment sees the
///   INTERSECTION — exactly the subject's admitted members that the
///   pattern's own literals also name (`case 1:` over `{1, 2, 4}`
///   narrows to `{1}`; `case 2 | 4:` narrows to `{2, 4}`, the union of
///   admitted alternatives, which IS the intersection of `{1, 2, 4}`
///   with the pattern's own `{2, 4}`).
/// - NOT-TAKEN (`keep_matched` false): the remainder every LATER arm
///   and the wildcard must see is the DIFFERENCE — the subject's
///   admitted members with the pattern's own literals removed.
///
/// `None` when the subject does not enumerate (`enumerable_numeric_
/// members`) or the pattern proves no literal (`pattern_literal_
/// members`) — the caller's own job to fall back to the unnarrowed
/// subject in that case, never to guess. The narrowed result keeps the
/// subject's own `kind_tag` (a pattern's literal tag is never trusted
/// over the subject's, matching `narrow_name_against_literal`'s own
/// "the binding's own tag survives" reading) and the WEAKER of the two
/// trust grades (`min_trust_level` — a narrowing is never claimed
/// stronger than either input that fed it).
fn narrow_scalar_subject(
    subject: &AbstractValue,
    pattern: &Pattern,
    keep_matched: bool,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let subject_members = enumerable_numeric_members(subject)?;
    let kind_tag = subject.kind_tag?;
    let pattern_members = pattern_literal_members(pattern, environment, kernel)?;
    let kept: Vec<f64> = subject_members
        .into_iter()
        .filter(|member| pattern_members.contains(member) == keep_matched)
        .collect();
    let grade = min_trust_level(trust_level_of(subject), TrustProved);
    Some(known_values(kept, kind_tag, grade))
}

/// The code-point vector an AbstractValue carries, if it is a known
/// exact string (`Kind::Values` tagged `PrimitiveKind::String`) —
/// `expressions.rs::exact_string_values`'s own twin, reimplemented
/// locally rather than imported (this file's own "no importing
/// loops.rs" precedent, `generator_yields`'s own doc, applied to
/// expressions.rs's private helper the same way).
fn exact_string_values(value: &AbstractValue) -> Option<&[f64]> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(&value.values)
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

/// Whether `narrowed` is the SAME admitted set as `remaining` — both
/// read through `enumerable_numeric_members` and compared as sets
/// (order-independent: a join can enumerate its members in either
/// order). `narrowed` is always a subset of `remaining` by construction
/// (`narrow_scalar_subject`'s own intersection), so equal LENGTH with
/// every member of one present in the other is exactly set equality
/// here — this is the FULL-OVERLAP test: an arm whose intersection is
/// the whole remaining subject consumes it entirely, the same
/// unconditional Taken this file gave every arm before the per-arm
/// split existed (no later arm can ever be reached, so no join is
/// needed for this arm).
fn is_full_overlap(narrowed: &AbstractValue, remaining: &AbstractValue) -> bool {
    let (Some(narrowed_members), Some(remaining_members)) =
        (enumerable_numeric_members(narrowed), enumerable_numeric_members(remaining))
    else {
        return false;
    };
    narrowed_members.len() == remaining_members.len()
        && remaining_members.iter().all(|member| narrowed_members.contains(member))
}

/// Rebinds `subject_name` (the match subject's own name, when the
/// subject expression is a bare `Name`) and every name `pattern`
/// captures (`pattern_captures`) to `intersected` inside `arm_env` — the
/// two slots the PARTIAL-OVERLAP split (`match_taken_environment`'s own
/// doc) must narrow before the arm's body ever walks: a bare `MatchAs`
/// binds its own name to the raw `remaining_subject` when the pattern
/// is Taken (`match_as_outcome`'s own doc), which is correct only for
/// the FULL-overlap case; a split arm's body must see the INTERSECTION
/// instead, on every name that would otherwise still read the coarser
/// pre-split claim. A pattern with no nameable captures
/// (`pattern_captures` answers `None` for a shape past this file's flat
/// scope — never reached here, since only a literal/or pattern that
/// itself proved a value reaches a split) simply rebinds the subject
/// name alone.
fn rebind_split_subject(
    arm_env: &mut Environment,
    subject_name: Option<&str>,
    pattern: &Pattern,
    intersected: &AbstractValue,
) {
    if let Some(name) = subject_name {
        arm_env.bind(name, intersected.clone());
    }
    if let Some(captures) = pattern_captures(pattern, None) {
        for name in captures {
            arm_env.bind(&name, intersected.clone());
        }
    }
}

/// Walk every arm of a match statement in order, deciding each with
/// `arm_outcome`, joining the arms that survive exactly the way
/// `walk_if` joins its own branch environments (`Environment::join`,
/// left-folded over every surviving arm — 1 survivor is that arm alone,
/// 2+ actually joins; see `finalize_survivors`). `walk_arm_body(body,
/// &mut arm_env)` runs the caller's OWN statement walker over one arm's
/// body (`check.rs`'s `walk_statement` loop, `summaries.rs`'s
/// `interpret_body` — this file stays walker-agnostic on purpose, so it
/// never depends on either caller's own types) and answers `Some(true)`
/// when the arm survives (falls through, matching `arm_terminates`'s
/// own reading of "does not end in `return`/`raise`"), `Some(false)`
/// when it terminates, or `None` when the walk itself must decline the
/// WHOLE match (an unsupported statement inside the body) — propagated
/// here by `?`, the same short-circuit `interpret_body`'s own callers
/// already rely on.
///
/// Returns `Some((environment, falls_through))`: the post-match
/// environment (the one taken arm's own, or every surviving split arm's
/// joined together), and whether the match AS A WHOLE falls through —
/// `true` unless every reached arm terminated. `None` means no arm is
/// decidably reached at all (the whole match is undecided from its own
/// start, no body has walked, and the caller's own join-fallback is
/// free to re-walk every case for itself).
///
/// A subject that DOES NOT enumerate (`enumerable_numeric_members`
/// answers `None` — an unbounded `Kind::Set` ray, `Kind::Unknown`, a
/// non-numeric subject, …) never splits: every arm is judged, walked,
/// and joined exactly as before this rule existed — one arm's own
/// `Taken` outcome is unconditional (the same "later arms are dead,
/// stop scanning" this function always gave), and the answer is that
/// one arm's own body's environment.
///
/// A DECIDABLE SCALAR SUBJECT splits per arm — the abstractly-precise
/// reading of what a multi-valued subject actually means: SOME admitted
/// values take this arm, the rest fall through to later ones, the same
/// two-branch shape an `if`/`else` over an unknown boolean already
/// walks. Before an arm is even judged, an ALREADY-EMPTY
/// `remaining_subject` (every earlier arm together consumed every
/// admitted value) makes it — and every arm after it — dead: no body
/// walks, nothing joins, the loop simply runs out (this is the
/// wildcard-sees-emptiness case: `case 1: / case 2 | 4: / case _:` over
/// `{1, 2, 4}` leaves the wildcard's own `remaining_subject` empty, and
/// its body never walks). Otherwise, a Taken arm asks `narrow_scalar_
/// subject`'s own INTERSECTION (`keep_matched: true`) against
/// `remaining_subject`:
///
/// - the intersection is the WHOLE remaining subject (`is_full_overlap`)
///   — this arm alone consumes every value still live, so it behaves as
///   an unconditional Taken: no later arm is ever reached, and this
///   function returns as soon as this one arm's own body is walked
///   (survives or not; a survivor is the sole answer, no join needed).
/// - the intersection is a PROPER, NONEMPTY subset — a genuine split:
///   `rebind_split_subject` narrows the arm's own subject binding and
///   capture(s) down to the intersection, the body walks under THAT
///   narrower claim, and (if it survives) its environment joins the
///   later arms' the same way two `if`/`else` branches join. The
///   DIFFERENCE (`keep_matched: false`) becomes the new
///   `remaining_subject` for every arm still to come, and the walk
///   continues scanning rather than stopping.
/// - `narrow_scalar_subject` answers `None` (the pattern proves no
///   scalar literal — a bare capture/wildcard, a guard-only arm, a
///   class/sequence/mapping pattern) — the pattern is not itself
///   scalar-decidable even though the subject is, so this arm keeps
///   TODAY'S binary semantics: unconditional Taken, no split.
///
/// An `Undecidable` arm still poisons every LATER arm exactly as
/// before — this file cannot rule out that arm having actually run.
/// When no EARLIER arm's body has walked yet, the whole match declines
/// (`None`), the same as before this rule existed. Once an earlier
/// split arm's body already walked for real, declining here would let
/// the caller's join-fallback re-walk it a second time (duplicating any
/// side effects that walk already recorded) — so this answers instead
/// with whatever already survived (see the `any_arm_walked` guard on
/// the `Undecidable` arm below).
pub fn match_taken_environment(
    subject_value: &AbstractValue,
    subject_name: Option<&str>,
    cases: &[MatchCase],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    walk_arm_body: &mut dyn FnMut(&[Stmt], &mut Environment) -> Option<bool>,
) -> Option<(Environment, bool)> {
    let mut remaining_subject = subject_value.clone();
    let mut survivors: Vec<Environment> = Vec::new();
    // Whether ANY arm's body was actually walked (an unconditional Taken,
    // or a split arm) — distinguishes "every arm bottomed out at
    // NotTaken/dead, the match is genuinely undecided" (still `None`,
    // the caller's own join-fallback engages, exactly as before this
    // rule existed) from "arms were decided and walked, but every
    // surviving one terminated" (a real answer, falls-through `false` —
    // the same "0 survivors" reading `walk_if`'s own join gives, never
    // a fallback re-walk that would run those bodies' side effects a
    // second time).
    let mut any_arm_walked = false;
    for case in cases {
        // An already-exhausted remaining subject makes this arm (and
        // every arm after it) dead: no runtime value is left for it to
        // ever see, so its body never walks and it never joins.
        if enumerable_numeric_members(&remaining_subject).is_some_and(|members| members.is_empty()) {
            continue;
        }
        match arm_outcome(&case.pattern, case.guard.as_deref(), &remaining_subject, environment, kernel) {
            ArmOutcome::Taken(mut arm_env) => {
                let intersection = narrow_scalar_subject(&remaining_subject, &case.pattern, true, environment, kernel);
                let split = intersection
                    .as_ref()
                    .filter(|intersected| !is_full_overlap(intersected, &remaining_subject));
                let Some(intersected) = split else {
                    // no scalar split for this arm — unconditional Taken,
                    // exactly today's behavior: walk its body once and
                    // commit ITS OWN environment regardless of whether
                    // the body survives (`check.rs`'s pre-existing
                    // decided-arm branch committed `arm_env`
                    // unconditionally too — a body ending in
                    // `return`/`raise` is still the honest post-match
                    // state, the same way a single surviving `if` arm
                    // needs no join). No later arm is ever reached; the
                    // whole match falls through iff this one body does.
                    let survives = walk_arm_body(&case.body, &mut arm_env)?;
                    return Some((arm_env, survives));
                };
                // a genuine partial split: narrow this arm's own subject
                // binding/capture(s) to the intersection, walk under
                // that narrower claim, and thread the difference onward
                // to every arm still to come instead of stopping here.
                any_arm_walked = true;
                rebind_split_subject(&mut arm_env, subject_name, &case.pattern, intersected);
                let survives = walk_arm_body(&case.body, &mut arm_env)?;
                if survives {
                    survivors.push(arm_env);
                }
                if let Some(narrowed) =
                    narrow_scalar_subject(&remaining_subject, &case.pattern, false, environment, kernel)
                {
                    remaining_subject = narrowed;
                }
                continue;
            }
            ArmOutcome::NotTaken => {
                if let Some(narrowed) =
                    narrow_scalar_subject(&remaining_subject, &case.pattern, false, environment, kernel)
                {
                    remaining_subject = narrowed;
                }
                continue;
            }
            // An Undecidable arm poisons every LATER arm exactly as
            // before — this file cannot rule out that arm having
            // actually run. When NO earlier arm's body was walked yet,
            // that is still `None`: the whole match is undecided from
            // its own start, and the caller's join-fallback is free to
            // re-walk every case from scratch. Once an earlier SPLIT
            // arm's body already walked for real (side effects: findings
            // recorded, returns collected), falling back would re-walk
            // it a second time — so this answers with whatever already
            // survived instead (falls-through `true` only when at least
            // one split arm actually survived; `false`, same as the
            // "no survivors" tail below, otherwise).
            ArmOutcome::Undecidable => {
                if !any_arm_walked {
                    return None;
                }
                return Some(if survivors.is_empty() {
                    (environment.fork(), false)
                } else {
                    (finalize_survivors(survivors), true)
                });
            }
        }
    }
    if survivors.is_empty() {
        return if any_arm_walked { Some((environment.fork(), false)) } else { None };
    }
    Some((finalize_survivors(survivors), true))
}

/// The join every split-arm walk in `match_taken_environment` funnels
/// through — the SAME shape `walk_if`'s own tail takes
/// (`check.rs::walk_if`): 1 survivor is that arm's environment alone,
/// 2+ left-folds through `Environment::join` exactly as `walk_if`/
/// today's `walk_match` join their own branch environments — this
/// function never invents a different join. The caller never calls
/// this with an empty `survivors` (it reads that case itself, before
/// ever reaching here — see `match_taken_environment`'s own tail).
fn finalize_survivors(mut survivors: Vec<Environment>) -> Environment {
    match survivors.len() {
        1 => survivors.into_iter().next().expect("len checked above"),
        _ => {
            let mut joined = survivors.remove(0);
            for arm in survivors {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_domain::abstract_value::known_set;
    use refined_domain::abstract_value::known_values;
    use refined_domain::abstract_value::null_value;
    use refined_domain::abstract_value::unknown;
    use refined_domain::abstract_value::SetKindTag;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_ast::ModModule;

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

    /// `case 1:` over a multi-valued `{1, 2, 4}` subject (an ordinary
    /// `Kind::Values` join, not a single known scalar) is Taken — 1 is
    /// a member of the subject's admitted set, the membership question
    /// `enumerable_numeric_members` answers instead of the old
    /// single-value equality.
    #[test]
    fn value_pattern_hit_on_multi_valued_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::Taken(_)),
            "1 is a member of {{1, 2, 4}} and must match `case 1:`"
        );
    }

    /// The miss half: a literal that is NOT a member of the subject's
    /// multi-valued set is a dead arm — NotTaken, the same outcome
    /// every other unreachable arm answers, never a new label.
    #[test]
    fn value_pattern_miss_on_multi_valued_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 8:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::NotTaken),
            "8 is not a member of {{1, 2, 4}}: `case 8:` must be a dead arm"
        );
    }

    /// `case 2 | 4:` over the same `{1, 2, 4}` subject is Taken — both
    /// alternatives are members (`match_or_outcome` takes the first
    /// Taken alternative, `case 2:`, without needing to try `case 4:`).
    #[test]
    fn or_pattern_hit_on_multi_valued_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 2 | 4:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::Taken(_)),
            "2 and 4 are both members of {{1, 2, 4}}: `case 2 | 4:` must be Taken"
        );
        let proved = pattern_proved_value(&cases[0].pattern, &environment, &kernel)
            .expect("a MatchOr of two numeric literals proves their union");
        let mut values = proved.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![2.0, 4.0], "`case 2 | 4:` proves exactly {{2, 4}}");
    }

    /// The set-subject narrowing rule, pinned directly against
    /// `narrow_scalar_subject`: a multi-valued `{1, 2, 4}` subject
    /// through `case 1:` keeps the INTERSECTION — exactly `{1}`, the one
    /// admitted member the literal names, never the pattern's own
    /// literal alone and never the untouched subject.
    #[test]
    fn narrow_scalar_subject_keeps_the_intersection_for_a_literal_arm() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let narrowed = narrow_scalar_subject(&subject, &cases[0].pattern, true, &environment, &kernel)
            .expect("a multi-valued {1, 2, 4} subject enumerates and `case 1:` proves a literal");
        assert_eq!(narrowed.values, vec![1.0], "`case 1:` over {{1, 2, 4}} narrows to exactly {{1}}");
    }

    /// The or-pattern half: `case 2 | 4:` over the same subject keeps
    /// the UNION of admitted alternatives — `{2, 4}` — which IS the
    /// intersection of the subject with the pattern's own `{2, 4}`.
    #[test]
    fn narrow_scalar_subject_keeps_the_union_of_admitted_alternatives_for_an_or_arm() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 2 | 4:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let narrowed = narrow_scalar_subject(&subject, &cases[0].pattern, true, &environment, &kernel)
            .expect("a multi-valued {1, 2, 4} subject enumerates and `case 2 | 4:` proves a union of literals");
        let mut values = narrowed.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![2.0, 4.0], "`case 2 | 4:` over {{1, 2, 4}} narrows to exactly {{2, 4}}");
    }

    /// A literal not admitted by the subject at all makes the arm dead:
    /// the intersection is empty, matching `match_value_outcome`'s own
    /// NotTaken verdict for the same pair (`value_pattern_miss_on_
    /// multi_valued_subject` above).
    #[test]
    fn narrow_scalar_subject_intersection_is_empty_for_a_literal_the_subject_never_admits() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 8:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let narrowed = narrow_scalar_subject(&subject, &cases[0].pattern, true, &environment, &kernel)
            .expect("a multi-valued {1, 2, 4} subject enumerates and `case 8:` proves a literal");
        assert!(narrowed.values.is_empty(), "8 is not admitted by {{1, 2, 4}}: the arm's own intersection is empty");
    }

    /// The DIFFERENCE half — the remainder a NotTaken arm leaves for
    /// every later arm and the eventual wildcard: a multi-valued
    /// `{1, 2, 4}` subject minus `case 1:`'s own literal is exactly
    /// `{2, 4}`.
    #[test]
    fn narrow_scalar_subject_keeps_the_difference_for_a_not_taken_arm() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let remainder = narrow_scalar_subject(&subject, &cases[0].pattern, false, &environment, &kernel)
            .expect("a multi-valued {1, 2, 4} subject enumerates and `case 1:` proves a literal");
        let mut values = remainder.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![2.0, 4.0], "{{1, 2, 4}} minus {{1}} is exactly {{2, 4}}");
    }

    /// The visible per-arm split the ledger construct names: a
    /// multi-valued `{1, 2, 4}` subject through `case 1 as a: / case 2 |
    /// 4 as b: / case _ as c:` — arm one's body walks with `a` bound to
    /// exactly `{1}` (the intersection), arm two's walks with `b` bound
    /// to exactly `{2, 4}` (the intersection of the DIFFERENCE `{2, 4}`
    /// left after arm one with arm two's own `{2, 4}`), and the
    /// wildcard's own remaining subject is exhausted to empty by the
    /// first two arms together — so its body never walks at all. Each
    /// walked arm behaves as a genuine `if`/`elif` branch: both arm one
    /// and arm two survive (return `Some(true)`), so their environments
    /// JOIN (`Environment::join`, the same call `walk_if` makes) rather
    /// than either one alone winning outright.
    #[test]
    fn match_taken_environment_splits_a_multi_valued_subject_across_its_arms() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases(concat!(
            "match x:\n",
            "    case 1 as a:\n",
            "        pass\n",
            "    case 2 | 4 as b:\n",
            "        pass\n",
            "    case _ as c:\n",
            "        pass\n",
        ));
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let environment = empty_environment();
        let mut walked_bodies: Vec<Environment> = Vec::new();
        let (joined, falls_through) = match_taken_environment(&subject, None, &cases, &environment, &kernel, &mut |_body, arm_env| {
            walked_bodies.push(arm_env.fork());
            Some(true)
        })
        .expect("a multi-valued {1, 2, 4} subject through three arms that together cover it is decided");
        assert!(falls_through, "both walked arms survive, so the whole match falls through");
        assert_eq!(walked_bodies.len(), 2, "only arm one and arm two walk; the wildcard's own remainder is empty");

        let mut first_arm_a = walked_bodies[0].read("a").expect("arm one's own capture binds `a`").values.clone();
        first_arm_a.sort_by(f64::total_cmp);
        assert_eq!(first_arm_a, vec![1.0], "arm one's body sees `a` narrowed to exactly {{1}}, the intersection");

        let mut second_arm_b = walked_bodies[1].read("b").expect("arm two's own capture binds `b`").values.clone();
        second_arm_b.sort_by(f64::total_cmp);
        assert_eq!(second_arm_b, vec![2.0, 4.0], "arm two's body sees `b` narrowed to exactly {{2, 4}}, the intersection");

        assert!(joined.read("c").is_none(), "the wildcard's own capture `c` never binds — its body never walked");
    }

    /// The dead-arm half of the same construct, pinned directly:
    /// `narrow_scalar_subject`'s own membership question over the
    /// wildcard's remaining subject — once `case 1:` and `case 2 | 4:`
    /// have both been subtracted from `{1, 2, 4}`, what remains is the
    /// EMPTY set `enumerable_numeric_members` reads back, the exact
    /// signal `match_taken_environment`'s own dead-arm skip
    /// (`is_some_and(|members| members.is_empty())`) tests before ever
    /// calling `arm_outcome` on a later arm.
    #[test]
    fn wildcard_remaining_subject_is_empty_after_earlier_arms_consume_every_admitted_value() {
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let environment = empty_environment();
        let Some(kernel) = loaded_kernel() else { return };
        let after_first_arm = narrow_scalar_subject(&subject, &cases[0].pattern, false, &environment, &kernel)
            .expect("{1, 2, 4} enumerates and `case 1:` proves a literal");
        let or_cases = match_cases("match x:\n    case 2 | 4:\n        pass\n");
        let after_second_arm =
            narrow_scalar_subject(&after_first_arm, &or_cases[0].pattern, false, &environment, &kernel)
                .expect("{2, 4} enumerates and `case 2 | 4:` proves a union of literals");
        assert!(
            after_second_arm.values.is_empty(),
            "case 1: then case 2 | 4: together consume every admitted value: {{1, 2, 4}} minus {{1}} minus {{2, 4}} is empty"
        );
    }

    /// A `Kind::Set` subject that enumerates a union-of-singletons form
    /// (`{1, 2, 4}` built as a right-fold `Union` tree of one-element
    /// `OneOf` leaves — the exact shape
    /// `collection_models.rs::scalars_of_union_of_singletons` reads,
    /// and the shape the kernel's own `join_state` answers for several
    /// distinct scalar values, per that function's own doc) is
    /// readable the same way a multi-valued `Kind::Values` subject is —
    /// `case 1:` is Taken.
    #[test]
    fn value_pattern_hit_on_enumerable_set_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::refinement_forms::make_refined_set;
        use refined_sets::refinement_forms::one_of;
        use refined_sets::refinement_forms::union;
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let singleton = |v: f64| make_refined_set(vec![one_of(&[v])]);
        let set = make_refined_set(vec![union(singleton(1.0), make_refined_set(vec![union(singleton(2.0), singleton(4.0))]))]);
        let subject = known_set(set, None, TrustProved, SetKindTag::None);
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::Taken(_)),
            "1 is a member of the enumerable set {{1, 2, 4}} and must match `case 1:`"
        );
    }

    /// The KindUnion axis: a subject union with a numeric-tagged arm
    /// admitting `{1, 2, 4}` alongside a non-numeric arm (a null value,
    /// the same "some arm, never which one" shape `json.loads`'s own
    /// return-space union carries, `expressions.rs::
    /// json_loads_value_space`) narrows under `case 1:` to the numeric
    /// arm's own intersection — `kind_union_pattern_outcome` judges the
    /// pattern against each arm and takes the first arm the pattern
    /// admits, mirroring `assignability.rs`'s per-arm KindUnion judge.
    #[test]
    fn value_pattern_hit_on_kind_union_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 1:\n        pass\n");
        let numeric_arm = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let subject = refined_domain::abstract_value::kind_union_of(vec![null_value(), numeric_arm]);
        assert_eq!(subject.kind, Kind::KindUnion, "the two distinct-kind arms must not collapse");
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::Taken(_)),
            "the numeric arm admits 1, so the KindUnion subject must match `case 1:`"
        );
    }

    /// The miss half: every arm of the union rules the pattern out (no
    /// arm's value could ever be `8`) — the whole union is NotTaken.
    #[test]
    fn value_pattern_miss_on_kind_union_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case 8:\n        pass\n");
        let numeric_arm = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let subject = refined_domain::abstract_value::kind_union_of(vec![null_value(), numeric_arm]);
        assert_eq!(subject.kind, Kind::KindUnion, "the two distinct-kind arms must not collapse");
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(
            matches!(outcome, ArmOutcome::NotTaken),
            "no arm of the union can ever be 8: `case 8:` must be NotTaken"
        );
    }

    /// `anchor_of`'s own row: a String-tagged subject against a
    /// string-literal `MatchValue` pattern (`case "left":`) is DECIDED,
    /// not Undecidable — the fix this file's own doc now names.
    #[test]
    fn string_value_pattern_hit() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case \"left\":\n        pass\n");
        let subject = crate::string_models::string_literal_value("left");
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::Taken(_)), "\"left\" must match `case \"left\":`");
    }

    /// The miss half of the same shape: a different known string subject
    /// against the same literal pattern decides NotTaken, not
    /// Undecidable.
    #[test]
    fn string_value_pattern_miss() {
        let Some(kernel) = loaded_kernel() else { return };
        let cases = match_cases("match x:\n    case \"left\":\n        pass\n");
        let subject = crate::string_models::string_literal_value("right");
        let environment = empty_environment();
        let outcome = arm_outcome(&cases[0].pattern, cases[0].guard.as_deref(), &subject, &environment, &kernel);
        assert!(matches!(outcome, ArmOutcome::NotTaken), "\"right\" must not match `case \"left\":`");
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
    fn match_taken_environment_walks_the_one_arm_a_single_valued_subject_takes() {
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
        let mut walked: Vec<usize> = Vec::new();
        let (_, falls_through) = match_taken_environment(&subject, None, &cases, &environment, &kernel, &mut |_body, _arm_env| {
            walked.push(walked.len());
            Some(true)
        })
        .expect("case 2 must be decidably reached");
        assert!(falls_through, "an ordinary `pass` body survives");
        assert_eq!(walked, vec![0], "a single-valued subject of 2 is a full-overlap arm: only `case 2:` walks, unconditionally");
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
            match_taken_environment(&subject, None, &cases, &environment, &kernel, &mut |_body, _arm_env| Some(true)).is_none(),
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
                    crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
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
                    crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
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
                    crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
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
                    crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None },
                    crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None },
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
