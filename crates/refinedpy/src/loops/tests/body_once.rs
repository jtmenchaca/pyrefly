//! Body-once analysis: `run_body_once`'s own single-pass walk, and
//! the `known_number` test helper's own contract.

use super::*;

/// `run_body_once` over the simplest self-referencing rebind —
/// `total = total * 2.0` against an exact binding — completes and
/// binds the doubled exact value: two known operands are the most
/// determinable arithmetic this module reads, and a decline here is
/// what turns a non-stabilizing accumulation body into the coarser
/// "not yet walked" blocker instead of the fixed-point one.
#[test]
fn run_body_once_completes_an_exact_self_referencing_rebind() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("total = total * 2.0\n");
    let mut environment = environment_with(&[("total", 1.0)]);
    let declared = no_declared();
    let mut judge_context = JudgeContext {
        declared: &declared,
        newly_declared: HashMap::new(),
        already_fired: std::collections::HashSet::new(),
        fires: Vec::new(),
    };
    let body = [stmt];
    let outcome = run_body_once(&body, &mut environment, &kernel, &mut judge_context);
    assert!(outcome.is_some(), "an exact rebind of two known operands is walkable");
    let total = environment.read("total").expect("total stays bound");
    assert_eq!(total.values, vec![2.0], "1.0 * 2.0 binds exactly 2.0: {total:?}");
}

#[test]
fn known_number_helper_carries_proved_number_values() {
    let value = known_number(3.0);
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Number));
    // TrustProved renders as no grade at all — see known_values
    assert_eq!(value.grade, None);
}
