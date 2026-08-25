use std::sync::Arc;

use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::make_refined_set;

use super::math_call_result;
use super::math_constant_value;
use super::sqrt_argument_is_known_negative;
use crate::expressions::binary_arithmetic_value_with_kernel;

fn int_operand(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustProved)
}

fn float_operand(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

/// A kernel handle for tests that ask a `floor`-over-a-set question.
/// `None` when the native dylib artifact has not been built (the
/// same skip `expressions.rs`'s own tests use), so this file's tests
/// run without requiring `pnpm kernel:native` first.
fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

fn math_call(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let Some(kernel) = loaded_kernel() else { return None };
    math_call_result(function, arguments, &kernel)
}

#[test]
fn test_floor_known_float() {
    let Some(result) = math_call("floor", &[float_operand(200.9)]) else { return };
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.values, vec![200.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_ceil_known_float() {
    let Some(result) = math_call("ceil", &[float_operand(200.1)]) else { return };
    assert_eq!(result.values, vec![201.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_trunc_known_float_positive() {
    let Some(result) = math_call("trunc", &[float_operand(200.9)]) else { return };
    assert_eq!(result.values, vec![200.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_trunc_known_float_negative() {
    // trunc rounds toward zero, not floor — -200.9 truncates to -200
    let Some(result) = math_call("trunc", &[float_operand(-200.9)]) else { return };
    assert_eq!(result.values, vec![-200.0]);
}

#[test]
fn test_isqrt_perfect_square() {
    let Some(result) = math_call("isqrt", &[int_operand(16.0)]) else { return };
    assert_eq!(result.values, vec![4.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_isqrt_non_perfect_square_floors() {
    // isqrt(17) == 4 (floor of the exact square root, not a perfect square)
    let Some(result) = math_call("isqrt", &[int_operand(17.0)]) else { return };
    assert_eq!(result.values, vec![4.0]);
}

#[test]
fn test_isqrt_negative_declines() {
    // math.isqrt raises ValueError for a negative operand; no exception
    // channel exists yet, so this declines rather than answer a value
    // the real call never produces
    if loaded_kernel().is_none() {
        return;
    }
    let result = math_call("isqrt", &[int_operand(-1.0)]);
    assert_eq!(result, None);
}

/// `floor`/`ceil`/`trunc` return an `Integral`, and no Python `int`
/// is infinite or NaN — CPython raises there, so no value is
/// answered. The same shape as `test_isqrt_negative_declines`.
#[test]
fn test_rounding_of_non_finite_arguments_declines() {
    if loaded_kernel().is_none() {
        return;
    }
    for name in ["floor", "ceil", "trunc"] {
        for input in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(
                math_call(name, &[float_operand(input)]),
                None,
                "math.{name}({input:?}) must answer no value — CPython raises there"
            );
        }
    }
}

/// The raise predicate names CPython's own exception and message for
/// each non-finite argument, and stays silent on a finite one — the
/// pairing `provable_raise` reads.
#[test]
fn test_rounding_argument_raises_names_the_exception() {
    use super::rounding_argument_raises;
    assert_eq!(
        rounding_argument_raises("floor", &[float_operand(f64::INFINITY)]),
        Some(("OverflowError", "cannot convert float infinity to integer"))
    );
    assert_eq!(
        rounding_argument_raises("ceil", &[float_operand(f64::NEG_INFINITY)]),
        Some(("OverflowError", "cannot convert float infinity to integer"))
    );
    assert_eq!(
        rounding_argument_raises("trunc", &[float_operand(f64::NAN)]),
        Some(("ValueError", "cannot convert float NaN to integer"))
    );
    assert_eq!(rounding_argument_raises("floor", &[float_operand(40.9)]), None);
    // `fabs` returns a float, so `inf` is an ordinary answer there —
    // not a rounding row, and not a raise
    assert_eq!(rounding_argument_raises("fabs", &[float_operand(f64::INFINITY)]), None);
}

/// `math.fabs` needs no finiteness gate: it returns a FLOAT, and
/// `inf`/`nan` are ordinary Python floats, so the call returns
/// normally where the rounding family raises.
#[test]
fn test_fabs_of_an_infinity_still_answers() {
    let Some(result) = math_call("fabs", &[float_operand(f64::NEG_INFINITY)]) else { return };
    assert_eq!(result.values, vec![f64::INFINITY]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn test_fabs_negative_int_widens_to_float() {
    let Some(result) = math_call("fabs", &[int_operand(-3.0)]) else { return };
    assert_eq!(result.values, vec![3.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn test_copysign_exact() {
    let Some(result) = math_call("copysign", &[float_operand(3.0), float_operand(-1.0)]) else { return };
    assert_eq!(result.values, vec![-3.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `math.copysign(1, x)` with a KNOWN magnitude but a fully
/// UNRESOLVED sign source `x` (`copysign_call_over_unresolved_sign`'s
/// own doc): the answer is the two-signed-branch set `{-1.0, 1.0}` —
/// both magnitude-preserving outcomes IEEE 754's copysign can
/// produce when the sign bit is unknown, A2.xfer.sign's own row.
#[test]
fn test_copysign_of_a_known_magnitude_over_an_unresolved_sign_answers_both_branches() {
    let Some(kernel) = loaded_kernel() else { return };
    let unresolved_sign = float_sorted_unknown();
    let result = math_call_result("copysign", &[int_operand(1.0), unresolved_sign], &kernel)
        .expect("copysign(1, x) over an unresolved sign should answer");
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    let mut values = result.values.clone();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![-1.0, 1.0]);
}

/// `math.copysign(1, x)` where `x` is provably NONNEGATIVE
/// (`x` narrowed to `x >= 0.0` by an upstream guard): the sign
/// source's own window excludes every negative sign bit, so the
/// answer narrows to the single positive branch, not both.
#[test]
fn test_copysign_of_a_known_magnitude_over_a_provably_nonnegative_sign_answers_one_branch() {
    let Some(kernel) = loaded_kernel() else { return };
    let nonnegative_sign = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustSpec, SetKindTag::None)
    };
    let result = math_call_result("copysign", &[int_operand(1.0), nonnegative_sign], &kernel)
        .expect("copysign(1, x) over a provably nonnegative sign should answer");
    assert_eq!(result.values, vec![1.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `math.sqrt` on a KNOWN PERFECT SQUARE answers the exact Float
/// value — IEEE 754 correct rounding pins `sqrt(40000.0) == 200.0`
/// exactly, not merely approximately.
#[test]
fn test_sqrt_perfect_square_answers_the_exact_value() {
    let Some(result) = math_call("sqrt", &[float_operand(40000.0)]) else { return };
    assert_eq!(result.values, vec![200.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `math.sqrt` on a NON-perfect-square known non-negative argument
/// answers the sort-only Float set, never a specific value.
#[test]
fn test_sqrt_non_perfect_square_answers_float_sorted_unknown() {
    let Some(result) = math_call("sqrt", &[float_operand(2.0)]) else { return };
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.set_kind_tag, SetKindTag::None);
}

/// `math.sqrt` on a KNOWN NEGATIVE argument answers no value at
/// all — the real call raises `ValueError`, which is
/// `provable_raise`'s row, not this dispatcher's.
#[test]
fn test_sqrt_known_negative_answers_none() {
    if loaded_kernel().is_none() {
        return;
    }
    let result = math_call("sqrt", &[float_operand(-2.0)]);
    assert_eq!(result, None);
}

#[test]
fn test_sqrt_argument_is_known_negative_reads_true_for_a_negative_operand() {
    assert!(sqrt_argument_is_known_negative(&[float_operand(-2.0)]));
    assert!(!sqrt_argument_is_known_negative(&[float_operand(4.0)]));
}

/// `math.sin(0.0)` is one of `jsSin`'s pinned corners (`trigSingleton`'s
/// own `zero` case): the KERNEL-BACKED row now answers the exact
/// value `0.0`, not the sort-only unconstrained set — `sin` is one
/// of the trig.1-13 rows this wave wires.
#[test]
fn test_sin_known_argument_answers_the_kernel_backed_exact_zero() {
    let Some(result) = math_call("sin", &[float_operand(0.0)]) else { return };
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.values, vec![0.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `math.sin(1.0)` is NOT one of `jsSin`'s pinned corners — the
/// kernel answers a certified bracketing WINDOW around the true
/// value (sin(1) ≈ 0.8414709848), never a bare sort-only set: the
/// window must be narrow enough to exclude 0.9 and 0.8 while
/// containing the true value's own neighborhood.
#[test]
fn test_sin_of_one_answers_a_narrow_kernel_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let result = math_call_result("sin", &[float_operand(1.0)], &kernel).expect("sin(1.0) should answer");
    assert_eq!(result.kind, Kind::Set, "sin(1.0) is not a pinned corner — expect a window: {result:?}");
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!(
        (kernel.member)(&result.set, &[0.841_470_984_807_896_5]),
        "the true value must be inside the window: {:?}",
        result.set
    );
    assert!(
        !(kernel.member)(&result.set, &[0.9]),
        "0.9 is outside sin(1.0)'s true window: {:?}",
        result.set
    );
}

/// `math.hypot(3, 4)` is the textbook 3-4-5 right triangle: `sqrt(3**2
/// + 4**2) == sqrt(25) == 5.0` exactly, no rounding error to
/// introduce — `hypot_exact_perfect_square`'s own row, answered
/// directly without a kernel round trip (`math_call`, not
/// `math_call_result`). c-reads-and-values.py's own `math_hypot`
/// fixture row (`int(math.hypot(3, 4))`) is this shape.
#[test]
fn test_hypot_of_a_known_perfect_square_answers_the_exact_value() {
    let Some(result) = math_call("hypot", &[float_operand(3.0), float_operand(4.0)]) else { return };
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert_eq!(result.values, vec![5.0]);
}

/// `math.hypot(1, 1)` — `sqrt(1**2 + 1**2) == sqrt(2)`, irrational: no
/// perfect-square shortcut applies, so this falls through to the
/// KERNEL-BACKED two-argument row (`kernel_backed_hypot_call`,
/// `math_call_result`'s own `"hypot" if arguments.len() == 2` arm) —
/// a bracketing window around the true value, not the sort-only
/// `float_sorted_unknown()` this row used to answer before the
/// kernel's own `js.hypot` transfer was wired in.
#[test]
fn test_hypot_of_a_non_perfect_square_still_answers_a_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let result = math_call_result("hypot", &[float_operand(1.0), float_operand(1.0)], &kernel)
        .expect("hypot(1.0, 1.0) should answer");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!((kernel.member)(&result.set, &[std::f64::consts::SQRT_2]), "sqrt(2) must be inside hypot(1,1)'s window");
    assert!(!(kernel.member)(&result.set, &[2.0]), "2.0 is outside hypot(1,1)'s true window");
}

/// `math.hypot(0.3, 0.4)` — A2.xfer.roots.py's own `hypot_inside`
/// row: the true value is 0.5 (binary64 does not represent `0.3` or
/// `0.4` exactly, so the sum-of-squares and its root carry a few ulp
/// of rounding — the kernel widens by 3 ulp, per its own `hypot_
/// sound.lean` proof), so this asserts a BRACKETING window
/// containing 0.5, never exactness. Before `kernel_backed_hypot_
/// call`, this operand pair (not a perfect square in the Rust-side
/// shortcut's own integer-root sense) answered the fully unbounded
/// `float_sorted_unknown()` — the false positive A2.xfer.roots.py's
/// `hypot_inside` row named (`'number'`, no window, RTS7001 against
/// `Unit`'s `[0, 1]`).
#[test]
fn test_hypot_of_zero_three_zero_four_answers_a_window_containing_one_half() {
    let Some(kernel) = loaded_kernel() else { return };
    let result = math_call_result("hypot", &[float_operand(0.3), float_operand(0.4)], &kernel)
        .expect("hypot(0.3, 0.4) should answer");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!((kernel.member)(&result.set, &[0.5]), "0.5 must be inside hypot(0.3, 0.4)'s window");
    assert!(!(kernel.member)(&result.set, &[2.0]), "2.0 is far outside hypot(0.3, 0.4)'s window");
}

#[test]
fn test_sin_of_unknown_argument_declines() {
    if loaded_kernel().is_none() {
        return;
    }
    let unknown_argument = AbstractValue::default();
    let result = math_call("sin", &[unknown_argument]);
    assert_eq!(result, None);
}

#[test]
fn test_unmodeled_function_declines() {
    if loaded_kernel().is_none() {
        return;
    }
    let result = math_call("frexp", &[float_operand(1.0)]);
    assert_eq!(result, None);
}

/// `math.log2(1024.0)`/`math.log10(1000.0)` are interior points, not
/// `jsLog2`/`jsLog10`'s pinned corners (`0`, `1`, `posInf`) — the
/// KERNEL-BACKED row answers a certified window containing the
/// exact value (10.0 / 3.0 respectively), never the sort-only
/// unconstrained set.
#[test]
fn test_log2_and_log10_answer_kernel_windows_containing_their_exact_values() {
    let Some(kernel) = loaded_kernel() else { return };
    let log2 = math_call_result("log2", &[float_operand(1024.0)], &kernel).expect("log2(1024.0) should answer");
    assert_eq!(log2.kind, Kind::Set);
    assert_eq!(log2.kind_tag, Some(PrimitiveKind::Float));
    assert!((kernel.member)(&log2.set, &[10.0]), "log2(1024) = 10 exactly, must be inside the window");
    assert!(!(kernel.member)(&log2.set, &[11.0]), "11 is outside log2(1024)'s true window");

    let log10 = math_call_result("log10", &[float_operand(1000.0)], &kernel).expect("log10(1000.0) should answer");
    assert_eq!(log10.kind, Kind::Set);
    assert_eq!(log10.kind_tag, Some(PrimitiveKind::Float));
    assert!((kernel.member)(&log10.set, &[3.0]), "log10(1000) = 3 exactly, must be inside the window");
    assert!(!(kernel.member)(&log10.set, &[4.0]), "4 is outside log10(1000)'s true window");
}

/// `math.cbrt(x)` over a KNOWN INTERVAL operand `[0.0, 1.0]`
/// (`cbrt_call_over_set`'s own doc): cbrt is monotone increasing, and
/// the cube root of `[0, 1]` is `[0, 1]` again (cbrt(0) = 0,
/// cbrt(1) = 1), so the answer's window must enclose both endpoints
/// and exclude a value clearly outside them.
#[test]
fn test_cbrt_over_a_known_interval_answers_a_window_containing_zero_to_one() {
    let Some(kernel) = loaded_kernel() else { return };
    let interval = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0), at_most(1.0)]), None, TrustSpec, SetKindTag::None)
    };
    let result =
        math_call_result("cbrt", std::slice::from_ref(&interval), &kernel).expect("cbrt([0,1]) should answer");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!((kernel.member)(&result.set, &[0.0]), "cbrt(0) = 0 must be inside the window");
    assert!((kernel.member)(&result.set, &[1.0]), "cbrt(1) = 1 must be inside the window");
    assert!(!(kernel.member)(&result.set, &[2.0]), "cbrt([0,1]) stays within [0,1] — 2 must be outside");
}

/// `math.floor(random.random() * 121)` — the kernel's own `Mult`
/// transfer carries the half-open `[0.0, 1.0)` window through
/// multiplication by 121 to `[0.0, 121.0)`, and this file's
/// `rounding_call_over_set` asks the kernel's `Floor` transfer on that
/// set — the row this file was built to close (c-reads-and-values.py
/// c:546). Floor of `[0.0, 121.0)` is the integer set `[0, 120]`, so
/// a value of exactly 121 must never be reachable through this row.
#[test]
fn test_floor_over_a_bounded_float_set_from_multiplication() {
    let Some(kernel) = loaded_kernel() else { return };
    let random_window = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0), below(1.0)]), None, TrustSpec, SetKindTag::None)
    };
    let scale = int_operand(121.0);
    let scaled =
        binary_arithmetic_value_with_kernel(ruff_python_ast::Operator::Mult, &random_window, &scale, &kernel);
    assert_eq!(scaled.kind, Kind::Set, "random() * 121 should stay a set: {scaled:?}");
    let result = math_call_result("floor", std::slice::from_ref(&scaled), &kernel)
        .expect("floor of a bounded float set should answer");
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    // the set must decide 121 as NOT a member — the upper bound is
    // strictly below 121, so floor never reaches it
    assert!(!(kernel.member)(&result.set, &[121.0]), "121 must not be a member of floor([0, 121))");
    assert!((kernel.member)(&result.set, &[120.0]), "120 must be a member of floor([0, 121))");
    assert!((kernel.member)(&result.set, &[0.0]), "0 must be a member of floor([0, 121))");
}

/// `math.exp(x)` over a KNOWN INTERVAL operand `[0.0, 1.0]` (not a
/// single known value): `float_transferable_operand`'s `Kind::Set`
/// branch poses the interval directly to the kernel's `js.exp`
/// transfer, which answers a window enclosing `[exp(0), exp(1)] =
/// [1, e]` — exp is monotone increasing, so the true image of a
/// closed interval is exactly that bracketing window.
#[test]
fn test_exp_over_a_known_interval_answers_a_window_containing_one_to_e() {
    let Some(kernel) = loaded_kernel() else { return };
    let interval = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0), at_most(1.0)]), None, TrustSpec, SetKindTag::None)
    };
    let result =
        math_call_result("exp", std::slice::from_ref(&interval), &kernel).expect("exp([0,1]) should answer");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!((kernel.member)(&result.set, &[1.0]), "exp(0) = 1 must be inside the window");
    assert!(
        (kernel.member)(&result.set, &[std::f64::consts::E]),
        "exp(1) = e must be inside the window"
    );
    assert!(!(kernel.member)(&result.set, &[3.0]), "e < 3, so 3 must be outside exp([0,1])'s window");
    assert!(!(kernel.member)(&result.set, &[-0.5]), "exp is always positive — -0.5 must be outside");
}

/// `math.exp(1.0)` on a KNOWN SINGLE VALUE: the kernel still answers
/// a bracketing WINDOW (exp is implementation-approximated at an
/// interior point, per the pins table's own note), containing e but
/// not a bare sort-only unconstrained set.
#[test]
fn test_exp_of_one_answers_a_narrow_kernel_window_containing_e() {
    let Some(kernel) = loaded_kernel() else { return };
    let result = math_call_result("exp", &[float_operand(1.0)], &kernel).expect("exp(1.0) should answer");
    assert_eq!(result.kind, Kind::Set);
    assert!((kernel.member)(&result.set, &[std::f64::consts::E]), "e must be inside the window");
    assert!(!(kernel.member)(&result.set, &[3.0]), "3 is outside exp(1)'s true window");
    assert!(!(kernel.member)(&result.set, &[2.5]), "2.5 is outside exp(1)'s true window");
}

/// `math.log(x)` on a KNOWN NEGATIVE operand: the kernel's own
/// `logCorners` answers `NaN` for that operand (JavaScript's
/// `Math.log(-1)` returns `NaN`), but CPython's `math.log(-1)`
/// RAISES `ValueError` instead of returning a value —
/// `kernel_backed_unary_family_call`'s own doc. This row must
/// decline rather than answer a value the real call never
/// produces.
#[test]
fn test_log_of_a_known_negative_declines_rather_than_answer_nan() {
    if loaded_kernel().is_none() {
        return;
    }
    let result = math_call("log", &[float_operand(-1.0)]);
    assert_eq!(result, None, "math.log(-1) raises in CPython — must answer no value");
}

/// `math.asin(x)` outside `[-1, 1]`: the same NaN-vs-raise gate as
/// `log`'s negative row, for the inverse-trig domain instead of the
/// logarithm's sign domain.
#[test]
fn test_asin_outside_domain_declines_rather_than_answer_nan() {
    if loaded_kernel().is_none() {
        return;
    }
    let result = math_call("asin", &[float_operand(2.0)]);
    assert_eq!(result, None, "math.asin(2.0) raises in CPython — must answer no value");
}

/// `math.log(1.0)` is one of `logCorners`' pinned corners — the
/// exact value `0.0`, not a window.
#[test]
fn test_log_of_one_answers_the_kernel_backed_exact_zero() {
    let Some(result) = math_call("log", &[float_operand(1.0)]) else { return };
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.values, vec![0.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `math.atan2(1.0, 1.0)` (trig.10): `y = x = 1.0` lands in
/// `jsAtan2`'s served quadrant (`x > 0, y != 0`) — the true value
/// is `pi/4 ≈ 0.7853981634`, and the kernel answers a bracketing
/// window around it.
#[test]
fn test_atan2_of_one_one_answers_a_window_containing_pi_over_four() {
    let Some(kernel) = loaded_kernel() else { return };
    let result = math_call_result("atan2", &[float_operand(1.0), float_operand(1.0)], &kernel)
        .expect("atan2(1.0, 1.0) should answer");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!(
        (kernel.member)(&result.set, &[std::f64::consts::FRAC_PI_4]),
        "pi/4 must be inside atan2(1,1)'s window"
    );
    assert!(!(kernel.member)(&result.set, &[1.0]), "1.0 > pi/4 by enough to be outside the window");
}

/// `math.atan2(1.0, -1.0)`: `y = 1.0, x = -1.0` is quadrant II
/// (`x < 0, y > 0`), true value `3*pi/4 ≈ 2.356194490192345` — the
/// kernel's own atan2 extension now serves the left half-plane (the
/// axis and left-half-plane corners this test used to name as
/// unserved are pinned now), so the call answers a bracketing
/// window around that value rather than declining.
#[test]
fn test_atan2_outside_served_quadrant_now_answers_a_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let result = math_call_result("atan2", &[float_operand(1.0), float_operand(-1.0)], &kernel)
        .expect("atan2(1.0, -1.0) should answer now that the kernel serves x <= 0");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert!(
        (kernel.member)(&result.set, &[3.0 * std::f64::consts::FRAC_PI_4]),
        "3*pi/4 must be inside atan2(1, -1)'s window"
    );
    assert!(!(kernel.member)(&result.set, &[0.0]), "0.0 is far outside atan2(1, -1)'s window");
}

/// `math.pi` answers the exact `std::f64::consts::PI` value — the
/// nearest binary64 double to the mathematical constant, and
/// CPython's own value (library/math.rst's "Constants" section).
#[test]
fn test_math_pi_is_exact_value() {
    let result = math_constant_value("pi").expect("math.pi should answer a value");
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.values, vec![std::f64::consts::PI]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `math.e`/`math.tau`/`math.inf` each answer their exact concrete
/// value, not a sort-only approximation.
#[test]
fn test_math_e_tau_inf_are_exact_values() {
    let expectations = [("e", std::f64::consts::E), ("tau", std::f64::consts::TAU), ("inf", f64::INFINITY)];
    for (name, expected) in expectations {
        let result = math_constant_value(name).unwrap_or_else(|| panic!("math.{name} should answer a value"));
        assert_eq!(result.kind, Kind::Values);
        assert_eq!(result.values, vec![expected]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }
}

/// `math.nan` answers the domain's own NaN carrier (`Kind::NaN`),
/// never a value inside `known_values` — `element`'s construction-time
/// panic refuses NaN for every refined-set form.
#[test]
fn test_math_nan_is_nan_kind() {
    let result = math_constant_value("nan").expect("math.nan should answer a value");
    assert_eq!(result.kind, Kind::NaN);
}

#[test]
fn test_math_unmodeled_attribute_declines() {
    assert_eq!(math_constant_value("floor"), None);
}
