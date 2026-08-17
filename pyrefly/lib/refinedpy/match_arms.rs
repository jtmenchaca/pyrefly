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
//! Sequence/Mapping/Class patterns are Undecidable this wave — no
//! container state is carried by `AbstractValue` yet (`element_set`,
//! `keys` exist but this file does not read into them for match
//! purposes), so a sequence/mapping/class arm always declines rather
//! than guess a shape.
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

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::MatchCase;
use ruff_python_ast::Pattern;
use ruff_python_ast::Singleton;

use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::evaluate_expression;

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
}
