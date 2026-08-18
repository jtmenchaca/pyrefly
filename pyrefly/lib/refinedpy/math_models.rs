/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `math.*` call transfers: the exactly-decidable slice of the `math`
//! module (`floor`, `ceil`, `trunc`, `isqrt`, `fabs`, `copysign`, and
//! `sqrt` on a known perfect square), PLUS the sort-only approximated
//! family (`sqrt` on any other operand, every trig/hyperbolic function,
//! `cbrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `hypot`),
//! which answers `float_sorted_unknown()` — a Float-tagged all-numbers
//! SET, never a specific value — once every argument is known, so
//! assignability's sort-fire law can still refuse an int-sorted sink.
//! `math` is CPython's thin libm wrapper (library/math.html,
//! implementation detail note: "the current implementation... will
//! raise ValueError for invalid operations... and OverflowError for
//! results that overflow" — an IMPLEMENTATION-graded accuracy promise,
//! never a pinned exact bit pattern). `math_constant_value` answers the
//! module's own ATTRIBUTE constants (`pi`, `e`, `tau`, `inf`) —
//! separate from `math_call_result` since a constant read is never a
//! call.

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
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
/// recombination, so this is answered from the `math.copysign` clause
/// (`tmp/cpython/Doc/library/math.rst`), not from Lean `TransferOpAdd`.
/// `f64::copysign` is exactly this operation, including the signed-zero
/// case the doc calls out by name.
fn copysign_call(magnitude: f64, sign_source: f64) -> Option<AbstractValue> {
    Some(float_result(magnitude.copysign(sign_source)))
}

/// `math.sqrt(x)` on a KNOWN NEGATIVE operand provably raises
/// `ValueError` rather than answering a value — library/math.html's own
/// module-introduction note: "The current implementation will raise
/// `ValueError` for invalid operations like `sqrt(-1.0)`..." A
/// negative operand is `provable_raise`'s own business (expressions.rs
/// calls this row through its own dispatch), not this function's — this
/// helper only reports WHETHER the operand is a known negative,
/// leaving the raise message's own wording to the caller that owns
/// `provable_raise`'s one voice.
pub fn sqrt_argument_is_known_negative(arguments: &[AbstractValue]) -> bool {
    let Some(first) = arguments.first() else {
        return false;
    };
    match single_numeric_operand(first) {
        Some((value, _)) => value < 0.0,
        None => false,
    }
}

/// `math.sqrt(x)` on a KNOWN NON-NEGATIVE operand that is an EXACT
/// PERFECT SQUARE: the exact Float result, not sort-only. IEEE 754
/// (the standard C99's `sqrt` — and therefore CPython's libm wrapper —
/// implements, library/math.html's own module intro: "This module
/// provides access to the mathematical functions defined by the C
/// standard") requires `sqrt` to be CORRECTLY ROUNDED: the returned
/// double is the closest representable value to the true mathematical
/// square root. When the true square root is itself an integer that
/// fits exactly in an f64 (`arithmetic_result`'s own 2^53 exactness
/// bound, `expressions.rs`), "closest representable value" IS that
/// exact integer — there is no rounding error to introduce, so
/// `math.sqrt(40000) == 200.0` is a provable fact of the standard, not
/// an approximation this file merely observes. A non-perfect-square
/// operand (whose true root is irrational or non-terminating in
/// binary) has no such exactness guarantee and falls through to the
/// sort-only row below.
fn sqrt_exact_perfect_square(value: f64) -> Option<AbstractValue> {
    if value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    let root = value.sqrt();
    if root.fract() != 0.0 {
        return None;
    }
    if root * root != value {
        return None;
    }
    Some(float_result(root))
}

/// The approximated float family this wave promotes from `None` (plain
/// unknown) to `float_sorted_unknown()` (a Float-tagged, all-numbers
/// SET) once every argument is known: `sqrt`, `sin`, `cos`, `tan`,
/// `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`,
/// `acosh`, `atanh`, `cbrt`, `exp`, `expm1`, `log`, `log1p`, `log2`,
/// `log10`, `hypot` — `log2(x)`/`log10(x)` ("Return the base-2/base-10
/// logarithm of x," library/math.rst) are the same float-returning,
/// no-pinned-exact-value shape `log`/`log1p` already carry.
/// None of these carries a pinned exact-value clause (library/math.html's
/// module intro: "the current implementation... Behavior in exceptional
/// cases follows Annex F... will raise ValueError for invalid
/// operations... and OverflowError for results that overflow" — an
/// IMPLEMENTATION-graded accuracy promise, not an exact bit pattern),
/// so this row never answers a specific VALUE; it answers only the
/// SORT — Float, unconstrained — so `assignability`'s own sort-fire law
/// can still refuse an int-sorted sink without this file pretending to
/// know which float the real call would produce. Every argument must
/// still be a known single numeric value (an unknown argument answers
/// plain `unknown()` instead — see the dispatcher below), matching
/// every other row in this file's "known operands only" discipline.
fn approximated_family_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    const APPROXIMATED_NAMES: &[&str] = &[
        "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh",
        "atanh", "cbrt", "exp", "expm1", "log", "log1p", "log2", "log10", "hypot",
    ];
    if !APPROXIMATED_NAMES.contains(&function) {
        return None;
    }
    // sqrt(negative) is `provable_raise`'s row, not this one's — a
    // negative sqrt argument answers no VALUE here (the real call never
    // returns; it raises), matching every other raising row's decline
    if function == "sqrt" && sqrt_argument_is_known_negative(arguments) {
        return None;
    }
    for argument in arguments {
        single_numeric_operand(argument)?;
    }
    Some(float_sorted_unknown())
}

/// `math.pi` / `math.e` / `math.tau` / `math.inf` — ATTRIBUTE READS, not
/// calls (library/math.rst, "Constants" section: `data:: pi`/`data:: e`/
/// `data:: tau`/`data:: inf`, each "to available precision" or, for
/// `inf`, "Equivalent to the output of `float('inf')`"). None of the
/// four is a whole number, so a Float-sorted sort-only answer
/// (`float_sorted_unknown()`) is enough for `assignability`'s int-sort
/// fire law to refuse an int-sorted sink — the exact digit sequence is
/// never claimed. `math.nan` is deliberately excluded: NaN fails every
/// ordering comparison, which would make the sort-only Float set answer
/// UNSOUND for a sink that compares by value (a NaN is never `<=` any
/// bound), so this row stays undecided rather than answer a set that
/// does not actually contain the value. `None` for any other attribute
/// name.
pub fn math_constant_value(name: &str) -> Option<AbstractValue> {
    match name {
        "pi" | "e" | "tau" | "inf" => Some(float_sorted_unknown()),
        _ => None,
    }
}

/// `math_call_result` is the FROZEN entry point: `function` is the
/// attribute name after `math.` ("floor", "sqrt", …); `arguments` are
/// the already-evaluated operands in call order. `None` means "not
/// modeled" — the caller declines, same honesty as every other B4 row
/// in PYREFLY-NUMERIC-B3-B4.md.
///
/// Modeled EXACTLY (each an exactly-decidable row cited above):
/// `floor`, `ceil`, `trunc`, `isqrt`, `fabs`, `copysign`, and `sqrt` on
/// a known non-negative PERFECT-SQUARE operand (`sqrt_exact_perfect_square`'s
/// own doc — IEEE 754 correct rounding, not an approximation).
///
/// Modeled at SORT-ONLY precision (`approximated_family_result`'s own
/// doc): `sqrt` on a non-perfect-square operand, `sin`, `cos`, `tan`,
/// `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`,
/// `acosh`, `atanh`, `cbrt`, `exp`, `expm1`, `log`, `log1p`, `log2`,
/// `log10`, `hypot` — every argument known answers
/// `float_sorted_unknown()` (a Float-tagged all-numbers set), never a
/// specific value; `math.sqrt` on a known negative argument answers
/// `None` here because it provably RAISES instead (see
/// `sqrt_argument_is_known_negative`, read by `provable_raise`).
///
/// Still declined (no cited row this wave, and not sort-only-graded
/// either): `pow`, `fsum`, `remainder`, `fmod`, `gcd`, `lcm`,
/// `factorial`, `comb`, `perm`, `degrees`, `radians`, `isnan`, `isinf`,
/// `isfinite`, `nextafter`, `ulp`, `frexp`, `ldexp`, `modf`, `dist`,
/// `prod` — every one of them falls through the wildcard arm below to
/// `None`. Constants (`math.pi`, `math.e`, `math.tau`, `math.inf`,
/// `math.nan`) are attribute reads, not calls — out of scope for this
/// function entirely; see `math_constant_value` for those (`math.nan`
/// still excluded there, see its own doc).
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
        // `sqrt` on an exact perfect square answers the exact Float
        // result (IEEE 754 correct rounding — see
        // sqrt_exact_perfect_square's own doc); any other sqrt argument,
        // and every other approximated-family function, falls through
        // to the sort-only row below
        "sqrt" => {
            let [only] = arguments else { return None };
            let (value, _) = single_numeric_operand(only)?;
            sqrt_exact_perfect_square(value).or_else(|| approximated_family_result(function, arguments))
        }
        // the sort-only approximated family (trig, log, exp, hypot,
        // and sqrt on a non-perfect-square): float_sorted_unknown()
        // once every argument is known, per
        // approximated_family_result's own doc
        _ => approximated_family_result(function, arguments),
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

    /// `math.sqrt` on a KNOWN PERFECT SQUARE answers the exact Float
    /// value — IEEE 754 correct rounding pins `sqrt(40000.0) == 200.0`
    /// exactly, not merely approximately.
    #[test]
    fn test_sqrt_perfect_square_answers_the_exact_value() {
        let result = math_call_result("sqrt", &[float_operand(40000.0)]).expect("sqrt(40000) should answer exactly");
        assert_eq!(result.values, vec![200.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }

    /// `math.sqrt` on a NON-perfect-square known non-negative argument
    /// answers the sort-only Float set, never a specific value.
    #[test]
    fn test_sqrt_non_perfect_square_answers_float_sorted_unknown() {
        let result = math_call_result("sqrt", &[float_operand(2.0)]).expect("sqrt should answer sort-only");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.set_kind_tag, SetKindTag::None);
    }

    /// `math.sqrt` on a KNOWN NEGATIVE argument answers no value at
    /// all — the real call raises `ValueError`, which is
    /// `provable_raise`'s row, not this dispatcher's.
    #[test]
    fn test_sqrt_known_negative_answers_none() {
        let result = math_call_result("sqrt", &[float_operand(-2.0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_sqrt_argument_is_known_negative_reads_true_for_a_negative_operand() {
        assert!(sqrt_argument_is_known_negative(&[float_operand(-2.0)]));
        assert!(!sqrt_argument_is_known_negative(&[float_operand(4.0)]));
    }

    #[test]
    fn test_sin_known_argument_answers_sort_only() {
        let result = math_call_result("sin", &[float_operand(0.0)]).expect("sin should answer sort-only");
        assert_eq!(result.kind, Kind::Set);
    }

    #[test]
    fn test_hypot_known_arguments_answer_sort_only() {
        let result =
            math_call_result("hypot", &[float_operand(3.0), float_operand(4.0)]).expect("hypot should answer sort-only");
        assert_eq!(result.kind, Kind::Set);
    }

    #[test]
    fn test_sin_of_unknown_argument_declines() {
        let unknown_argument = AbstractValue::default();
        let result = math_call_result("sin", &[unknown_argument]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_unmodeled_function_declines() {
        let result = math_call_result("frexp", &[float_operand(1.0)]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_log2_and_log10_answer_sort_only() {
        let log2 = math_call_result("log2", &[float_operand(1024.0)]).expect("log2 should answer sort-only");
        assert_eq!(log2.kind, Kind::Set);
        let log10 = math_call_result("log10", &[float_operand(1000.0)]).expect("log10 should answer sort-only");
        assert_eq!(log10.kind, Kind::Set);
    }

    /// `math.pi` is a sort-only Float set — never an exact digit
    /// sequence, and never a whole number (so an int-sorted sink still
    /// fires against it).
    #[test]
    fn test_math_pi_is_sort_only_float() {
        let result = math_constant_value("pi").expect("math.pi should answer sort-only");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_math_e_tau_inf_are_sort_only_float() {
        for name in ["e", "tau", "inf"] {
            let result = math_constant_value(name).unwrap_or_else(|| panic!("math.{name} should answer sort-only"));
            assert_eq!(result.kind, Kind::Set);
        }
    }

    /// `math.nan` is excluded: a NaN value would make the sort-only
    /// Float set claim unsound for a value-comparing sink.
    #[test]
    fn test_math_nan_declines() {
        assert_eq!(math_constant_value("nan"), None);
    }

    #[test]
    fn test_math_unmodeled_attribute_declines() {
        assert_eq!(math_constant_value("floor"), None);
    }
}
