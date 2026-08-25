use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::AbstractValue;

use super::float_result;
use super::single_numeric_operand;
use super::sqrt_argument_is_known_negative;

/// The approximated float family still riding sort-only precision on a
/// KNOWN SINGLE-VALUE operand: `sqrt` on a non-perfect-square operand,
/// `cbrt` (a Set-shaped `cbrt` operand is `cbrt_call_over_set`'s own
/// row, tried first — this function is `math_call_result`'s `"cbrt"`
/// fallback only), and `hypot`'s own general VARIADIC form (three or
/// more coordinates — the TWO-argument form poses `TransferQuestionOp::
/// Hypot` directly through `kernel_backed_hypot_call`, `math_call_
/// result`'s own `"hypot" if arguments.len() == 2` arm; a Set-shaped
/// operand in the variadic form still declines outright here — no
/// kernel row exists for the N-ary shape, `kernel_backed_unary_family_
/// op`'s own doc) — `float_sorted_unknown()` (a Float-tagged, all-numbers
/// SET) once every argument is known, never a specific value. None of these
/// carries a pinned exact-value clause (library/math.html's module
/// intro: "the current implementation... Behavior in exceptional
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
///
/// The 18 kernel-backed unary names plus `atan2` are DELIBERATELY
/// ABSENT from `APPROXIMATED_NAMES` below, even though
/// `kernel_backed_unary_family_call`/`kernel_backed_atan2_call` can
/// themselves decline: `math_call_result` does NOT fall through from
/// those functions into this one. A decline from the kernel-backed row
/// means either a provable Python raise (the kernel answered `NaN`) or
/// the kernel arm's own served-shape gap (`Unknown`) — in both cases
/// `float_sorted_unknown()` would be a FALSE claim ("some float value
/// exists") layered on top of a call that provably does not return
/// one, or a claim stronger than the kernel itself was willing to
/// make. Answering nothing is the correct decline there; this function
/// is never consulted for those 19 names. `hypot`'s own TWO-argument
/// form joins that same no-fallback list (`math_call_result`'s own
/// `"hypot" if arguments.len() == 2` arm, `kernel_backed_hypot_call`'s
/// doc) — `hypot` stays IN `APPROXIMATED_NAMES` below only because its
/// general VARIADIC form (three or more coordinates, no kernel
/// election) still reaches this function through the ordinary `_` arm.
pub(super) fn approximated_family_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    const APPROXIMATED_NAMES: &[&str] = &["sqrt", "cbrt", "hypot"];
    if !APPROXIMATED_NAMES.contains(&function) {
        return None;
    }
    // sqrt(negative) is `provable_raise`'s row, not this one's — a
    // negative sqrt argument answers no VALUE here (the real call never
    // returns; it raises), matching every other raising row's decline
    if function == "sqrt" && sqrt_argument_is_known_negative(arguments) {
        return None;
    }
    // the TWO-argument form is served by `kernel_backed_hypot_call`
    // through `math_call_result`'s own dedicated arm and never reaches
    // this function at all — only `hypot`'s VARIADIC form (three or
    // more coordinates) arrives here, so this reads the perfect-square
    // shortcut over that N-ary shape only.
    if function == "hypot" {
        if let Some(result) = hypot_exact_perfect_square(arguments) {
            return Some(result);
        }
    }
    for argument in arguments {
        single_numeric_operand(argument)?;
    }
    Some(float_sorted_unknown())
}

/// `math.hypot(*coordinates)` on KNOWN operands whose Euclidean norm is
/// an EXACT PERFECT SQUARE: the exact Float result, not sort-only — the
/// same IEEE-754-correctly-rounded reasoning `sqrt_exact_perfect_square`
/// states for `math.sqrt`, applied to pow.8's own formula
/// (`python-pins.md`: `math.hypot(*coordinates)` is `sqrt(sum(x**2 for x
/// in coordinates))`, the Euclidean norm over ANY number of
/// coordinates). `3, 4` is the textbook case: `sum(x**2) = 9 + 16 = 25`,
/// `sqrt(25) = 5.0` exactly, no rounding error to introduce. Every
/// coordinate must be a known single numeric value
/// (`single_numeric_operand`) — a Set-shaped or unread argument declines
/// here and falls through to `approximated_family_result`'s own
/// sort-only row. The sum of squares must itself be a NONNEGATIVE
/// PERFECT SQUARE that lands inside the f64-exact 2^53 window, mirroring
/// `sqrt_exact_perfect_square`'s own three-part check (finite whole
/// operand, an integral root, and the root squaring back to the exact
/// operand — ruling out an f64 rounding coincidence).
pub(super) fn hypot_exact_perfect_square(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let mut sum_of_squares = 0.0;
    for argument in arguments {
        let (value, _) = single_numeric_operand(argument)?;
        if !value.is_finite() {
            return None;
        }
        sum_of_squares += value * value;
    }
    if sum_of_squares.abs() >= 2f64.powi(53) {
        return None;
    }
    let root = sum_of_squares.sqrt();
    if root.fract() != 0.0 || root * root != sum_of_squares {
        return None;
    }
    Some(float_result(root))
}
