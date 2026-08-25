use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::nan_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::make_refined_set;

use super::float_transferable_operand;

/// The single numeric value and its sort (int vs float), read off a
/// known single-valued AbstractValue — the same reading
/// `expressions.rs`'s `single_numeric_value`/`NumericOperand` pair uses
/// for arithmetic transfers. Boolean-sorted values read as int (`True`
/// is a subclass of `int`, AGENT-BRIEF.md), matching CPython's own
/// `math.floor(True) == 1` behavior.
pub(super) fn single_numeric_operand(value: &AbstractValue) -> Option<(f64, bool)> {
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

/// `math.isnan(x)` / `math.isinf(x)` / `math.isfinite(x)` — the three
/// float predicates, each stating a `True`/`False` return outright
/// (math.rst: `isnan` "Return ``True`` if *x* is a NaN (not a number),
/// and ``False`` otherwise"; `isinf` "Return ``True`` if *x* is a
/// positive or negative infinity, and ``False`` otherwise"; `isfinite`
/// "Return ``True`` if *x* is neither an infinity nor a NaN, and
/// ``False`` otherwise. (Note that ``0.0`` *is* considered finite.)").
///
/// A KNOWN single operand decides which of the two values the call
/// answers. The domain carries NaN as its own `Kind::NaN` state rather
/// than an f64 inside `Kind::Values` (`single_numeric_operand` refuses
/// it, and `refinement_forms::element` will not admit one), so the NaN
/// operand is read from that state directly.
///
/// Any other operand shape still answers the exact two-member boolean
/// domain rather than declining: the return is a `bool` whatever the
/// argument is, and stating that keeps a downstream `int(...)` at
/// `{0, 1}` instead of `int_image`'s unbounded ray.
pub(super) fn float_predicate_call(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let decided = if only.kind == Kind::NaN {
        Some(function == "isnan")
    } else {
        single_numeric_operand(only).map(|(value, _)| match function {
            "isnan" => value.is_nan(),
            "isinf" => value.is_infinite(),
            _ => value.is_finite(),
        })
    };
    match decided {
        Some(answer) => Some(known_values(
            vec![if answer { 1.0 } else { 0.0 }],
            PrimitiveKind::Boolean,
            TrustProved,
        )),
        None => Some(known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec)),
    }
}

/// Wraps an exact Integer-sort result. `floor`/`ceil`/`trunc`/`isqrt`
/// all return an `Integral` per their doc entries (library/math.html:
/// "delegates to `x.__floor__`/`__ceil__`/`__trunc__`, which should
/// return an Integral value"; `isqrt`'s own return is stated as the
/// integer square root directly) — Integer sort, not Float, regardless
/// of the operand's own sort.
pub(super) fn integer_result(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustProved)
}

/// Wraps an exact Float-sort result. `fabs`/`copysign` are covered by
/// the module's blanket rule (library/math.html intro: "Except when
/// explicitly noted otherwise, all return values are floats") — Float
/// sort always, regardless of the operand's own sort. `fabs_call`'s own
/// doc states `math.fabs(nan)` returns `nan` normally (a real value,
/// not a raise), so this wrapper answers `nan_value()` — the domain's
/// own NaN state (`refined_domain::abstract_value::nan_value`) — rather
/// than let a bare NaN enter `known_values`, which no refined set
/// admits (`refinement_forms::element`'s own construction-time refusal).
pub(super) fn float_result(value: f64) -> AbstractValue {
    if value.is_nan() {
        return nan_value();
    }
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

/// The DOMAIN GATE the whole `floor`/`ceil`/`trunc` family shares:
/// their result is an `Integral` (library/math.html — "delegates to
/// `x.__floor__`, which should return an Integral value"), and NO Python
/// `int` is infinite or NaN. `math.floor(float('inf'))` raises
/// `OverflowError` and `math.floor(float('nan'))` raises `ValueError`;
/// neither call ever returns a value.
///
/// The IEEE answer is not wrong — `f64::floor(inf)` IS `inf`, the same
/// clause the kernel's own `binary64.floor` proves. What is wrong is
/// reading that float answer back as a Python `int` without the check
/// `math.floor` itself performs. So a non-finite operand declines here
/// rather than claim an Integer-sorted infinity or NaN, the same
/// discipline `isqrt_call` keeps for its negative operand ("rather than
/// answering a value the real call would never produce") and
/// `binary_arithmetic_value`'s zero-divisor rows keep for theirs.
///
/// The raise itself is `provable_raise`'s row, not this one's — see
/// `rounding_argument_raises` below, which `expressions.rs` reads the
/// same way it reads `sqrt_argument_is_known_negative`.
pub(super) fn integral_domain_admits(value: f64) -> bool {
    value.is_finite()
}

/// `math.floor(x)` on a known single numeric value: the exact
/// mathematical floor, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.floor —
/// "Return the floor of x, the largest integer less than or equal to
/// x... delegates to x.__floor__, which should return an Integral
/// value"). A non-finite operand declines — `integral_domain_admits`'s
/// own doc.
pub(super) fn floor_call(value: f64) -> Option<AbstractValue> {
    if !integral_domain_admits(value) {
        return None;
    }
    Some(integer_result(value.floor()))
}

/// `math.ceil(x)` on a known single numeric value: the exact
/// mathematical ceiling, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.ceil —
/// "Return the ceiling of x, the smallest integer greater than or
/// equal to x... delegates to x.__ceil__, which should return an
/// Integral value"). A non-finite operand declines —
/// `integral_domain_admits`'s own doc.
pub(super) fn ceil_call(value: f64) -> Option<AbstractValue> {
    if !integral_domain_admits(value) {
        return None;
    }
    Some(integer_result(value.ceil()))
}

/// `math.trunc(x)` on a known single numeric value: truncation toward
/// zero, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.trunc —
/// "Return x with the fractional part removed, leaving the integer
/// part. This rounds toward 0... delegates to x.__trunc__, which
/// should return an Integral value"). `f64::trunc` is exactly this
/// toward-zero truncation. A non-finite operand declines —
/// `integral_domain_admits`'s own doc.
pub(super) fn trunc_call(value: f64) -> Option<AbstractValue> {
    if !integral_domain_admits(value) {
        return None;
    }
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
pub(super) fn isqrt_call(value: f64, is_int: bool) -> Option<AbstractValue> {
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
///
/// This row needs NO finiteness gate, unlike the `floor`/`ceil`/`trunc`
/// family above (`integral_domain_admits`): those return an `Integral`,
/// and no Python `int` is infinite or NaN, but `fabs` returns a FLOAT,
/// and `inf`/`nan` are ordinary Python floats. `math.fabs(float('inf'))`
/// is `inf` and `math.fabs(float('nan'))` is `nan` — both return
/// normally, so answering them is right rather than a missing check.
pub(super) fn fabs_call(value: f64) -> Option<AbstractValue> {
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
pub(super) fn copysign_call(magnitude: f64, sign_source: f64) -> Option<AbstractValue> {
    Some(float_result(magnitude.copysign(sign_source)))
}

/// `math.copysign(x, y)` on a KNOWN magnitude `x` but an UNRESOLVED sign
/// source `y` (a seeded range, a bare `float` parameter's own unbounded
/// set, or any operand `single_numeric_operand` cannot read as one
/// value): library/math.rst's clause states the result is "a float with
/// the magnitude (absolute value) of x but the sign of y" — the
/// magnitude is FIXED (`x` is known), so the only freedom left is the
/// TWO possible outcomes IEEE 754's copysign can produce, `{-|x|, +|x|}`
/// (`f64::copysign`'s own two-branch definition — the sign bit of `y`
/// is copied onto `|x|` verbatim, and a sign bit has exactly two
/// states). `y`'s own set is checked against the kernel's
/// `scalar_subset` first, the same "prove, never guess" discipline
/// `isqrt_as_sqrt_floor_composition` keeps for its own sign exclusion:
/// a `y` provably `>= 0.0` answers the single positive branch `{+|x|}`
/// outright (`copysign(x, +0.0)` is `+|x|` too — signed-zero's positive
/// case, the same clause's own "on platforms that support signed
/// zeros" note), a `y` provably `<= 0.0` (`< 0.0` union the zero point,
/// so `-0.0` — the clause's OWN named example, `copysign(1.0, -0.0) ==
/// -1.0` — reads as this branch) answers `{-|x|}`, and a `y` that
/// straddles (or a kernel refusal to decide) answers both signed
/// branches — sound over the full sign uncertainty either way.
pub(super) fn copysign_call_over_unresolved_sign(
    magnitude: f64,
    sign_source: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let sign_operand = float_transferable_operand(sign_source)?;
    let positive_magnitude = magnitude.abs();
    let nonneg = make_refined_set(vec![at_least(0.0)]);
    if matches!(crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&sign_operand, &nonneg)), Ok(true)) {
        return Some(float_result(positive_magnitude));
    }
    let nonpos = make_refined_set(vec![at_most(0.0)]);
    if matches!(crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&sign_operand, &nonpos)), Ok(true)) {
        return Some(float_result(-positive_magnitude));
    }
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(sign_source));
    Some(known_values(vec![-positive_magnitude, positive_magnitude], PrimitiveKind::Float, grade))
}

/// Which exception `math.floor`/`ceil`/`trunc` provably raises for a
/// KNOWN NON-FINITE argument, or `None` when the call returns a value
/// normally. The value dispatch and the raise dispatch read the SAME
/// operand through the same `integral_domain_admits` gate, so they agree
/// on exactly which rounding calls raise — the same pairing
/// `sqrt_argument_is_known_negative` keeps with `sqrt`'s own value row.
///
/// The two outcomes are CPython's own, confirmed against the running
/// interpreter and stated by library/math.html's implementation note
/// ("will raise ValueError for invalid operations... and OverflowError
/// for results that overflow"):
///
/// - an infinite argument raises `OverflowError: cannot convert float
///   infinity to integer`
/// - a NaN argument raises `ValueError: cannot convert float NaN to
///   integer`
///
/// Both messages are returned verbatim so `provable_raise`'s one voice
/// speaks CPython's own wording. A non-`math` rounding call, an unknown
/// operand, or a finite one answers `None` — this predicate never
/// guesses at a raise, the same way the file never guesses at a value.
pub fn rounding_argument_raises(function: &str, arguments: &[AbstractValue]) -> Option<(&'static str, &'static str)> {
    if !matches!(function, "floor" | "ceil" | "trunc") {
        return None;
    }
    let [only] = arguments else { return None };
    let (value, _) = single_numeric_operand(only)?;
    if value.is_nan() {
        return Some(("ValueError", "cannot convert float NaN to integer"));
    }
    if value.is_infinite() {
        return Some(("OverflowError", "cannot convert float infinity to integer"));
    }
    None
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
pub(super) fn sqrt_exact_perfect_square(value: f64) -> Option<AbstractValue> {
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
