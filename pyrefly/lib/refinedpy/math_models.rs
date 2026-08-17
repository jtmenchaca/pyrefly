/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `math.*` call transfers: the exactly-decidable slice of the `math`
//! module (PYREFLY-NUMERIC-B3-B4.md's "do first" list — `floor`,
//! `ceil`, `trunc`, `isqrt`, `fabs`, `copysign`), and nothing past it.
//! `math` is CPython's thin libm wrapper (library/math.html,
//! implementation detail note); PYREFLY-NUMERIC-B3-B4.md grades the
//! general float/transcendental family B4 (engine-approximated, same
//! honesty as RefinedTS's approximated `Math`), so those rows answer
//! `None` here rather than an approximated value.

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;

/// The single numeric value and its sort (int vs float), read off a
/// known single-valued AbstractValue — the same reading
/// `expressions.rs`'s `single_numeric_value`/`NumericOperand` pair uses
/// for arithmetic transfers. Boolean-sorted values read as int (`True`
/// is a subclass of `int`, AGENT-BRIEF.md), matching CPython's own
/// `math.floor(True) == 1` behavior.
fn single_numeric_operand(value: &AbstractValue) -> Option<(f64, bool)> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Integer) => Some((value.values[0], true)),
        Some(PrimitiveKind::Float) => Some((value.values[0], false)),
        Some(PrimitiveKind::Boolean) => Some((value.values[0], true)),
        _ => None,
    }
}

/// Wraps an exact Integer-sort result. `floor`/`ceil`/`trunc`/`isqrt`
/// all return an `Integral` per their doc entries (library/math.html:
/// "delegates to `x.__floor__`/`__ceil__`/`__trunc__`, which should
/// return an Integral value"; `isqrt`'s own return is stated as the
/// integer square root directly) — Integer sort, not Float, regardless
/// of the operand's own sort.
fn integer_result(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustProved)
}

/// Wraps an exact Float-sort result. `fabs`/`copysign` are covered by
/// the module's blanket rule (library/math.html intro: "Except when
/// explicitly noted otherwise, all return values are floats") — Float
/// sort always, regardless of the operand's own sort.
fn float_result(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

/// `math.floor(x)` on a known single numeric value: the exact
/// mathematical floor, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.floor —
/// "Return the floor of x, the largest integer less than or equal to
/// x... delegates to x.__floor__, which should return an Integral
/// value").
fn floor_call(value: f64) -> Option<AbstractValue> {
    Some(integer_result(value.floor()))
}

/// `math.ceil(x)` on a known single numeric value: the exact
/// mathematical ceiling, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.ceil —
/// "Return the ceiling of x, the smallest integer greater than or
/// equal to x... delegates to x.__ceil__, which should return an
/// Integral value").
fn ceil_call(value: f64) -> Option<AbstractValue> {
    Some(integer_result(value.ceil()))
}

/// `math.trunc(x)` on a known single numeric value: truncation toward
/// zero, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.trunc —
/// "Return x with the fractional part removed, leaving the integer
/// part. This rounds toward 0... delegates to x.__trunc__, which
/// should return an Integral value"). `f64::trunc` is exactly this
/// toward-zero truncation.
fn trunc_call(value: f64) -> Option<AbstractValue> {
    Some(integer_result(value.trunc()))
}

/// `math.isqrt(n)` on a known non-negative integer: the exact integer
/// square root, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.isqrt —
/// "Return the integer square root of the nonnegative integer n. This
/// is the floor of the exact square root of n, or equivalently the
/// greatest integer a such that a² ≤ n"). A negative n RAISES
/// `ValueError` in CPython; this file has no exception channel yet
/// (AGENT-BRIEF.md's raising-case note), so a negative operand
/// declines to `None` rather than answering a value the real call
/// would never produce. A non-integer operand also declines — isqrt's
/// domain is integers only.
fn isqrt_call(value: f64, is_int: bool) -> Option<AbstractValue> {
    if !is_int || value.fract() != 0.0 || value < 0.0 {
        return None;
    }
    Some(integer_result(value.sqrt().floor()))
}

/// `math.fabs(x)` on a known single numeric value: the absolute value,
/// Float sort ALWAYS
/// (https://docs.python.org/3.12/library/math.html#math.fabs —
/// "Return the absolute value of x"; the module's own blanket rule,
/// library/math.html's introduction, "Except when explicitly noted
/// otherwise, all return values are floats," applies here). Unlike the
/// builtin `abs()`, which preserves an int operand's sort, `fabs`
/// always widens to float — the same distinction AGENT-BRIEF.md and
/// PYREFLY-NUMERIC-B3-B4.md's `Math.abs` row call out.
fn fabs_call(value: f64) -> Option<AbstractValue> {
    Some(float_result(value.abs()))
}

/// `math.copysign(x, y)` on two known values: a float with the
/// magnitude of `x` and the sign of `y`, Float sort ALWAYS
/// (https://docs.python.org/3.12/library/math.html#math.copysign —
/// "Return a float with the magnitude (absolute value) of x but the
/// sign of y. On platforms that support signed zeros, copysign(1.0,
/// -0.0) returns -1.0"). Exact per IEEE 754's copysign operation: no
/// rounding or approximation is involved, only a magnitude/sign
/// recombination, so this is answered unconditionally rather than
/// gated behind an IEEE dial the way general float arithmetic is.
/// `f64::copysign` is exactly this operation, including the signed-zero
/// case the doc calls out by name.
fn copysign_call(magnitude: f64, sign_source: f64) -> Option<AbstractValue> {
    Some(float_result(magnitude.copysign(sign_source)))
}

/// `math.hypot(*coordinates)` — DEFERRED, not modeled, even for
/// all-known arguments. library/math.html grades `hypot` as
/// **implementation** (PYREFLY-NUMERIC-B3-B4.md's honesty table): the
/// 3.10+ accuracy note only promises "maximum error is under 1 ulp...
/// more typically... correctly rounded to within 1/2 ulp" — a soft
/// accuracy claim, not the exact-value guarantee `floor`/`ceil`/
/// `isqrt`/`fabs`/`copysign` carry. Without a clause pinning one
/// correctly-rounded bit pattern the way IEEE `copysign` does, this
/// file cannot compute an exact representable result and declines
/// rather than approximate.
#[allow(dead_code)]
fn hypot_call() -> Option<AbstractValue> {
    None
}

/// `math_call_result` is the FROZEN entry point: `function` is the
/// attribute name after `math.` ("floor", "sqrt", …); `arguments` are
/// the already-evaluated operands in call order. `None` means "not
/// modeled" — the caller declines, same honesty as every other B4 row
/// in PYREFLY-NUMERIC-B3-B4.md.
///
/// Modeled (each an exactly-decidable row cited above): `floor`,
/// `ceil`, `trunc`, `isqrt`, `fabs`, `copysign`.
///
/// Deferred (PYREFLY-NUMERIC-B3-B4.md's "Defer or B4-shape" list —
/// engine-approximated, no pinned exact result): `sqrt`, `pow`, `exp`,
/// `log`, `log2`, `log10`, `log1p`, `expm1`, every trig/hyperbolic
/// function (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`,
/// `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`), `cbrt`, `hypot`,
/// `fsum`, `remainder`, `fmod`, `gcd`, `lcm`, `factorial`, `comb`,
/// `perm`, `degrees`, `radians`, `isnan`, `isinf`, `isfinite`, `nextafter`,
/// `ulp`, `frexp`, `ldexp`, `modf`, `dist`, `prod` — none of these
/// carries a cited exact-value clause in this wave, so every one of
/// them falls through the wildcard arm below to `None`. Constants
/// (`math.pi`, `math.e`, `math.tau`, `math.inf`, `math.nan`) are
/// attribute reads, not calls — out of scope for this function
/// entirely.
pub fn math_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "floor" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            floor_call(value)
        }
        "ceil" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            ceil_call(value)
        }
        "trunc" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            trunc_call(value)
        }
        "isqrt" => {
            let (value, is_int) = single_numeric_operand(arguments.first()?)?;
            isqrt_call(value, is_int)
        }
        "fabs" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            fabs_call(value)
        }
        "copysign" => {
            let (magnitude, _) = single_numeric_operand(arguments.first()?)?;
            let (sign_source, _) = single_numeric_operand(arguments.get(1)?)?;
            copysign_call(magnitude, sign_source)
        }
        // general float pow, sqrt, trig, log, exp, hypot, and every
        // other libm-graded row: PYREFLY-NUMERIC-B3-B4.md's "Defer or
        // B4-shape" list — engine-approximated, no pinned exact
        // result, so this file declines rather than approximate
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_operand(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    fn float_operand(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Float, TrustProved)
    }

    #[test]
    fn test_floor_known_float() {
        let result = math_call_result("floor", &[float_operand(200.9)]).expect("floor should answer");
        assert_eq!(result.kind, Kind::Values);
        assert_eq!(result.values, vec![200.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_ceil_known_float() {
        let result = math_call_result("ceil", &[float_operand(200.1)]).expect("ceil should answer");
        assert_eq!(result.values, vec![201.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_trunc_known_float_positive() {
        let result = math_call_result("trunc", &[float_operand(200.9)]).expect("trunc should answer");
        assert_eq!(result.values, vec![200.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_trunc_known_float_negative() {
        // trunc rounds toward zero, not floor — -200.9 truncates to -200
        let result = math_call_result("trunc", &[float_operand(-200.9)]).expect("trunc should answer");
        assert_eq!(result.values, vec![-200.0]);
    }

    #[test]
    fn test_isqrt_perfect_square() {
        let result = math_call_result("isqrt", &[int_operand(16.0)]).expect("isqrt should answer");
        assert_eq!(result.values, vec![4.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_isqrt_non_perfect_square_floors() {
        // isqrt(17) == 4 (floor of the exact square root, not a perfect square)
        let result = math_call_result("isqrt", &[int_operand(17.0)]).expect("isqrt should answer");
        assert_eq!(result.values, vec![4.0]);
    }

    #[test]
    fn test_isqrt_negative_declines() {
        // math.isqrt raises ValueError for a negative operand; no exception
        // channel exists yet, so this declines rather than answer a value
        // the real call never produces
        let result = math_call_result("isqrt", &[int_operand(-1.0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_fabs_negative_int_widens_to_float() {
        let result = math_call_result("fabs", &[int_operand(-3.0)]).expect("fabs should answer");
        assert_eq!(result.values, vec![3.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_copysign_exact() {
        let result =
            math_call_result("copysign", &[float_operand(3.0), float_operand(-1.0)]).expect("copysign should answer");
        assert_eq!(result.values, vec![-3.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_sqrt_declines() {
        // general float sqrt is engine-approximated (B4) — never answered
        let result = math_call_result("sqrt", &[float_operand(4.0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_sin_declines() {
        let result = math_call_result("sin", &[float_operand(0.0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_hypot_declines_even_with_all_known_arguments() {
        // hypot is graded implementation/B4 in PYREFLY-NUMERIC-B3-B4.md —
        // declines even though every argument here is known exactly
        let result = math_call_result("hypot", &[float_operand(3.0), float_operand(4.0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_unmodeled_function_declines() {
        let result = math_call_result("frexp", &[float_operand(1.0)]);
        assert_eq!(result, None);
    }
}
