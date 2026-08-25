//! While-loop shape: the concrete counter step, the cap/decline on
//! non-convergence, break/else_runs, the kernel-bounded counter over
//! a seeded declared set, and a judged write that widens the counter
//! past a concrete value.

use super::*;

#[test]
fn while_counter_loop_runs_to_its_own_halt() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("while n < 5:\n    n += 1\n    total += n\n");
    let environment = environment_with(&[("n", 0.0), ("total", 0.0)]);
    let result = run(&stmt, &environment, &kernel).expect("bounded counter");
    // n: 0->1->2->3->4->5, loop stops once n == 5; total sums 1+2+3+4+5
    assert_eq!(result.read("n").unwrap().values, vec![5.0]);
    assert_eq!(result.read("total").unwrap().values, vec![15.0]);
}

#[test]
fn while_that_never_resolves_within_the_cap_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    // n never changes, so the condition holds forever — must not
    // guess convergence; must decline once the cap is hit
    let stmt = parsed_loop("while n < 5:\n    total += 1\n");
    let environment = environment_with(&[("n", 0.0), ("total", 0.0)]);
    assert!(run(&stmt, &environment, &kernel).is_none());
}

#[test]
fn while_break_stops_immediately_and_reports_else_runs_false() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("while n < 5:\n    if n == 2:\n        break\n    n += 1\nelse:\n    n = 200\n");
    let environment = environment_with(&[("n", 0.0)]);
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("while break is concrete");
    assert_eq!(answer.environment.read("n").unwrap().values, vec![2.0]);
    assert!(!answer.else_runs, "a break must report else_runs: false");
}

#[test]
fn a_while_with_no_break_reports_else_runs_true() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("while n < 3:\n    n += 1\nelse:\n    done = 1\n");
    let environment = environment_with(&[("n", 0.0)]);
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("while with no break is concrete");
    assert!(answer.else_runs, "no break ever fires — the else clause runs");
}

// --- while over a kernel-bounded counter (a Kind::Set start, UNIT 3) ---

#[test]
fn while_counter_over_a_seeded_known_set_asks_the_kernel_and_binds_a_set() {
    let Some(kernel) = loaded_kernel() else { return };
    // `n` starts as a Kind::Set (a seeded parameter's declared
    // range, e.g. `def f(n: Age): while n < 121: n += 1`) rather
    // than one known number — the concrete stepping path above
    // cannot step a set one value at a time, so this falls to
    // kernel_bounded_counter_environment.
    let stmt = parsed_loop("while n < 121:\n    n += 1\n");
    let mut environment = Environment::new(HashSet::from(["n".to_owned()]));
    environment.bind("n", known_set(age_set(), None, TrustProved, SetKindTag::None));
    let result = run(&stmt, &environment, &kernel).expect("kernel bounds the counter");
    let bound = result.read("n").expect("n stays bound");
    assert_eq!(bound.kind, Kind::Set);
}

#[test]
fn while_counter_over_a_known_set_with_an_unsupported_step_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    // `n *= 2` is not the Add/Sub step shape this file trusts to
    // lower into the kernel's LoopEffect grammar — must decline
    // rather than approximate.
    let stmt = parsed_loop("while n < 121:\n    n *= 2\n");
    let mut environment = Environment::new(HashSet::from(["n".to_owned()]));
    environment.bind("n", known_set(age_set(), None, TrustProved, SetKindTag::None));
    assert!(run(&stmt, &environment, &kernel).is_none());
}

#[test]
fn while_counter_over_a_known_set_with_a_nonempty_else_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    // a non-empty else after a kernel-certified (not concretely
    // run) loop is outside kernel_bounded_counter_environment's
    // scoped shape
    let stmt = parsed_loop("while n < 121:\n    n += 1\nelse:\n    done = 1\n");
    let mut environment = Environment::new(HashSet::from(["n".to_owned(), "done".to_owned()]));
    environment.bind("n", known_set(age_set(), None, TrustProved, SetKindTag::None));
    assert!(run(&stmt, &environment, &kernel).is_none());
}

// --- while body write widens the counter past Kind::Values (UNIT 3) ---

#[test]
fn a_refused_write_that_widens_the_counter_fires_and_still_answers_some() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:494's own shape (loop_body_over_ceiling): the
    // single-statement body's own `age = age + 121` fires on
    // iteration 1 against Age's [0, 120] ceiling, and the
    // refused-write law rebinds `age` to the DECLARED set
    // (Kind::Set) — the next condition check (`age < 3`) can no
    // longer read a single known number, so this run must stop
    // WITHOUT declining the whole loop: the fire already proved is
    // a real fact, and check.rs must not ALSO record its own "while
    // statement is not yet walked" blocker on top of it.
    let stmt = parsed_loop("while age < 3:\n    age = age + 121\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned()]));
    environment.bind("age", integer(0.0));
    let declared = declared_age("age");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("a widened counter after a judged fire is an honest stop, not a decline");
    assert_eq!(out.len(), 1, "the +121 step must fire exactly once: {out:?}");
    let age = answer.environment.read("age").expect("age stays bound to the declared set");
    assert_eq!(age.kind, Kind::Set);
}

#[test]
fn an_unreadable_condition_on_the_first_check_still_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    // `age` starts already unbound (not a single known number, and
    // not a Kind::Set the kernel path could pick up either) — the
    // FIRST condition check itself is unreadable, so this is a
    // shape this module never recognized at all, not a widened
    // counter after a judged run. Must still decline.
    let stmt = parsed_loop("while age < 3:\n    age = age + 1\n");
    let environment = Environment::new(HashSet::from(["age".to_owned()]));
    assert!(run(&stmt, &environment, &kernel).is_none());
}
