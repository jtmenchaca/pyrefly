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

/// `guarded_bare_capture_narrowed`'s own direct pin — the guarded
/// twin of `narrow_scalar_subject_keeps_the_intersection_for_a_
/// literal_arm` above: `case x if x == 1:` over `{1, 2, 4}` narrows
/// `x`'s own intersection to exactly `{1}`, through
/// `narrowing::guard_narrowed_values`'s comparison reader rather than
/// a pattern's own literal proof.
#[test]
fn guarded_bare_capture_narrowed_keeps_the_intersection_for_a_comparison_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases("match x:\n    case a if a == 1:\n        pass\n");
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let narrowed =
        guarded_bare_capture_narrowed(&subject, &cases[0].pattern, cases[0].guard.as_deref(), true, &kernel)
            .expect("`a == 1` is a comparison narrowing.rs's reader proves both directions of");
    assert_eq!(narrowed.values, vec![1.0], "`case a if a == 1:` over {{1, 2, 4}} narrows to exactly {{1}}");
}

/// The DIFFERENCE half of the same guarded pair: `case x if x == 1:`
/// being NOT-TAKEN leaves exactly `{2, 4}` — the same remainder
/// `narrow_scalar_subject_keeps_the_difference_for_a_not_taken_arm`
/// gives the literal spelling `case 1:`.
#[test]
fn guarded_bare_capture_narrowed_keeps_the_difference_for_a_comparison_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases("match x:\n    case a if a == 1:\n        pass\n");
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let remainder =
        guarded_bare_capture_narrowed(&subject, &cases[0].pattern, cases[0].guard.as_deref(), false, &kernel)
            .expect("`a == 1` is a comparison narrowing.rs's reader proves both directions of");
    let mut values = remainder.values.clone();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![2.0, 4.0], "{{1, 2, 4}} minus {{1}} (the guard's own excluded value) is exactly {{2, 4}}");
}

/// A pattern with no bare capture at all (`case 1 if a > 0:` — the
/// pattern is a LITERAL, not a capture) declines: this file's own
/// scope is a bare capture's own guard, never a literal pattern's.
#[test]
fn guarded_bare_capture_narrowed_declines_for_a_non_capture_pattern() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases("match x:\n    case 1 if x > 0:\n        pass\n");
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    assert!(
        guarded_bare_capture_narrowed(&subject, &cases[0].pattern, cases[0].guard.as_deref(), true, &kernel)
            .is_none(),
        "a literal pattern's own guard is out of this function's scope"
    );
}

/// `narrowing::narrow_name_against_membership`'s own direct pin, read
/// through this function: `case a if a in (2, 4):` over `{1, 2, 4}`
/// narrows the Taken side's own intersection to exactly `{2, 4}` — the
/// same membership-over-`Kind::Values` shape the match-guard lane's own
/// scope note named as declining before this reader existed.
#[test]
fn guarded_bare_capture_narrowed_keeps_the_intersection_for_a_membership_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases("match x:\n    case a if a in (2, 4):\n        pass\n");
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let narrowed =
        guarded_bare_capture_narrowed(&subject, &cases[0].pattern, cases[0].guard.as_deref(), true, &kernel)
            .expect("`a in (2, 4)` is a membership narrowing narrowing.rs's reader now proves both directions of");
    let mut values = narrowed.values.clone();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![2.0, 4.0], "`case a if a in (2, 4):` over {{1, 2, 4}} narrows to exactly {{2, 4}}");
}

/// The guard's own negation arm: everything `{1, 2, 4}` holds that is
/// NOT one of `(2, 4)` — exactly `{1}` — the difference every LATER arm
/// and the wildcard must still see, the same NEGATION role
/// `guarded_bare_capture_narrowed_keeps_the_difference_for_a_comparison_
/// guard` plays for a single-comparison guard.
#[test]
fn guarded_bare_capture_narrowed_keeps_the_difference_for_a_membership_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases("match x:\n    case a if a in (2, 4):\n        pass\n");
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let remainder =
        guarded_bare_capture_narrowed(&subject, &cases[0].pattern, cases[0].guard.as_deref(), false, &kernel)
            .expect("`a in (2, 4)` is a membership narrowing narrowing.rs's reader now proves both directions of");
    assert_eq!(remainder.values, vec![1.0], "{{1, 2, 4}} minus {{2, 4}} (the guard's own excluded values) is exactly {{1}}");
}

/// A guard shape neither narrowing channel reads at all (a CALL other
/// than `isinstance`/a recognized `TypeGuard` — `x.bit_length() > 0` —
/// names its own subject through an attribute call this file's leaf
/// vocabulary does not recognize) still declines both directions: this
/// arm keeps today's binary `arm_outcome` semantics rather than a
/// guessed split.
#[test]
fn guarded_bare_capture_narrowed_declines_for_an_unproved_guard_shape() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases("match x:\n    case a if a.bit_length() > 0:\n        pass\n");
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    assert!(
        guarded_bare_capture_narrowed(&subject, &cases[0].pattern, cases[0].guard.as_deref(), true, &kernel)
            .is_none(),
        "a method call on the guard's own subject is not a shape this reader proves"
    );
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

/// MATCH-GUARD INTEGRATION's own pin: the GUARDED twin of
/// `match_taken_environment_splits_a_multi_valued_subject_across_its_
/// arms` above — `case x if x == 1: / case x if x == 2: / case _:`
/// over the same `{1, 2, 4}` subject splits IDENTICALLY to that
/// test's literal spelling. Each guard is a single comparison
/// `narrowing::guard_narrowed_values` proves both directions of
/// (`narrow_compare`'s single-op path, reused rather than
/// reimplemented): arm one's own `x` narrows to exactly `{1}` (the
/// guard's admitted value, intersected with the remaining subject),
/// arm two's own `x` narrows to exactly `{2}` (the guard's admitted
/// value intersected with the DIFFERENCE `{2, 4}` arm one leaves
/// behind), and the wildcard's own remaining subject is `{4}` — NOT
/// empty, since neither guard excludes 4 the way `case 2 | 4:` would
/// have — so the wildcard's body DOES walk, unlike the literal
/// pattern's twin above where the wildcard's remainder was exhausted.
/// This is the guarded arm behaving exactly like an `if`/`elif`
/// chain over the same conditions: every reached arm's own capture
/// sees only the slice of the subject its own guard actually proves.
#[test]
fn match_taken_environment_splits_a_multi_valued_subject_across_guarded_bare_capture_arms() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases(concat!(
        "match x:\n",
        "    case a if a == 1:\n",
        "        pass\n",
        "    case b if b == 2:\n",
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
    .expect("a multi-valued {1, 2, 4} subject through three guarded arms is decided");
    assert!(falls_through, "every walked arm survives, so the whole match falls through");
    assert_eq!(walked_bodies.len(), 3, "all three arms walk; the wildcard's own remainder {{4}} is not empty");

    let mut first_arm_a = walked_bodies[0].read("a").expect("arm one's own capture binds `a`").values.clone();
    first_arm_a.sort_by(f64::total_cmp);
    assert_eq!(first_arm_a, vec![1.0], "arm one's body sees `a` narrowed to exactly {{1}}, the guard's own intersection");

    let mut second_arm_b = walked_bodies[1].read("b").expect("arm two's own capture binds `b`").values.clone();
    second_arm_b.sort_by(f64::total_cmp);
    assert_eq!(
        second_arm_b,
        vec![2.0],
        "arm two's body sees `b` narrowed to exactly {{2}}, the guard's own intersection with the {{2, 4}} difference"
    );

    let mut wildcard_c = walked_bodies[2].read("c").expect("the wildcard's own capture binds `c`").values.clone();
    wildcard_c.sort_by(f64::total_cmp);
    assert_eq!(
        wildcard_c,
        vec![4.0],
        "the wildcard's own remaining subject is exactly {{4}} — neither guard excludes it"
    );
    // three survivors join left-to-right (`finalize_survivors`); the
    // joined environment carries whichever binding each contributing
    // arm's own name held, so `a`/`b`/`c` all read back from `joined`
    // too, not only from each arm's own pre-join fork.
    assert_eq!(joined.read("c").map(|v| v.values.clone()), Some(vec![4.0]));
}

/// FULL PARITY WITH `case 2 | 4:` — the construct's own closing pin: a
/// membership guard over the SAME `{1, 2, 4}` subject as
/// `match_taken_environment_splits_a_multi_valued_subject_across_its_arms`
/// splits IDENTICALLY to that test's literal-OR spelling
/// (`case 1 as a: / case 2 | 4 as b: / case _ as c:`): arm one's own `a`
/// narrows to exactly `{1}`, arm two's own `b` narrows to exactly `{2,
/// 4}` (the guard's admitted values `{2, 4}` intersected with the
/// `{2, 4}` difference arm one leaves behind — here the FULL
/// difference, since the guard admits both remaining members), and the
/// wildcard's own remaining subject is exhausted to empty — its body
/// never walks, exactly as the literal-OR twin's wildcard never walks.
/// `x in (2, 4)` now reaches the same split `2 | 4` reaches, closing the
/// match-guard lane's own scope note.
#[test]
fn match_taken_environment_splits_a_multi_valued_subject_across_a_membership_guard_matching_the_or_pattern() {
    let Some(kernel) = loaded_kernel() else { return };
    let cases = match_cases(concat!(
        "match x:\n",
        "    case 1 as a:\n",
        "        pass\n",
        "    case b if b in (2, 4):\n",
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
    .expect("a multi-valued {1, 2, 4} subject through a literal arm and a membership-guarded arm is decided");
    assert!(falls_through, "both walked arms survive, so the whole match falls through");
    assert_eq!(walked_bodies.len(), 2, "only arm one and the membership-guarded arm walk; the wildcard's own remainder is empty");

    let mut first_arm_a = walked_bodies[0].read("a").expect("arm one's own capture binds `a`").values.clone();
    first_arm_a.sort_by(f64::total_cmp);
    assert_eq!(first_arm_a, vec![1.0], "arm one's body sees `a` narrowed to exactly {{1}}, the intersection");

    let mut second_arm_b = walked_bodies[1].read("b").expect("the guarded arm's own capture binds `b`").values.clone();
    second_arm_b.sort_by(f64::total_cmp);
    assert_eq!(
        second_arm_b,
        vec![2.0, 4.0],
        "the guarded arm's body sees `b` narrowed to exactly {{2, 4}}, matching `case 2 | 4:`'s own intersection"
    );

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
                crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None, base_sort: None },
                crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None, base_sort: None },
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
                crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None, base_sort: None },
                crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None, base_sort: None },
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
                crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None, base_sort: None },
                crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None, base_sort: None },
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
                crate::instances::ClassField { name: "x".to_owned(), declared: None, default: None, base_sort: None },
                crate::instances::ClassField { name: "y".to_owned(), declared: None, default: None, base_sort: None },
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
