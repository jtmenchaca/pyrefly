#![allow(unused_imports)]
//! `math.*` call transfers: the exactly-decidable slice of the `math`
//! module (`floor`, `ceil`, `trunc`, `isqrt`, `fabs`, `copysign`, and
//! `sqrt` on a known perfect square), the exact `int`-theory slice the
//! kernel serves (`factorial`, `gcd`, `lcm`, `comb`, `perm`, and
//! `isqrt` over a set — `int_theory_call`), the KERNEL-BACKED
//! transcendental family (`exp`, `expm1`, `log`, `log1p`, `log2`,
//! `log10`, every trig/hyperbolic function, and `atan2` —
//! `kernel_backed_unary_family_call`/`kernel_backed_atan2_call` —
//! python-pins.md's explog.1–6 and trig.1–13 rows, each posed to the
//! SAME `boundary/javascript.lean` transfer arm the JS adapter asks,
//! answering a certified window rather than a bare sort), PLUS `cbrt`
//! and `sqrt` on a known numeric SET operand (`cbrt_call_over_set`,
//! `sqrt_call_over_set` — the kernel's own `Cbrt`/`Sqrt` transfers over
//! a bounded window), PLUS the sort-only approximated family that
//! remains (`cbrt`/`hypot` on a known SINGLE value, and `sqrt` on a
//! non-perfect-square SINGLE value — pow.6/pow.8/pow.4's own pins rows
//! name why each stays local), which answers `float_sorted_unknown()`
//! — a Float-tagged all-numbers SET, never a specific value — once
//! every argument is known, so assignability's sort-fire law can still
//! refuse an int-sorted sink.
//! `math` is CPython's thin libm wrapper (library/math.html,
//! implementation detail note: "the current implementation... will
//! raise ValueError for invalid operations... and OverflowError for
//! results that overflow" — an IMPLEMENTATION-graded accuracy promise,
//! never a pinned exact bit pattern). That same note is a DOMAIN fact,
//! not only an accuracy one: a row whose real call raises answers no
//! value here. `isqrt` gates its negative operand for that reason, and
//! `floor`/`ceil`/`trunc` gate their non-finite ones
//! (`integral_domain_admits`) — each returns an `Integral`, and no
//! Python `int` is infinite or NaN. `math_constant_value` answers the
//! module's own ATTRIBUTE constants (`pi`, `e`, `tau`, `inf`) —
//! separate from `math_call_result` since a constant read is never a
//! call.

mod approximated;
mod constants;
mod int_theory;
mod kernel_transcendental;
mod set_family;
mod single_value_family;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::TransferQuestionOp;

use approximated::approximated_family_result;
use approximated::hypot_exact_perfect_square;
use int_theory::int_theory_call;
use kernel_transcendental::float_transferable_operand;
use kernel_transcendental::kernel_backed_atan2_call;
use kernel_transcendental::kernel_backed_hypot_call;
use kernel_transcendental::kernel_backed_unary_family_call;
use kernel_transcendental::kernel_backed_unary_family_op;
use set_family::cbrt_call_over_set;
use set_family::enclosure_is_provably_finite;
use set_family::rounding_call_over_set;
use set_family::sqrt_call_over_set;
use single_value_family::ceil_call;
use single_value_family::copysign_call;
use single_value_family::copysign_call_over_unresolved_sign;
use single_value_family::fabs_call;
use single_value_family::float_predicate_call;
use single_value_family::float_result;
use single_value_family::floor_call;
use single_value_family::integer_result;
use single_value_family::integral_domain_admits;
use single_value_family::isqrt_call;
use single_value_family::single_numeric_operand;
use single_value_family::sqrt_exact_perfect_square;
use single_value_family::trunc_call;

pub use constants::math_constant_value;
pub use constants::random_call_result;
pub use kernel_transcendental::domain_raise_classification;
pub use kernel_transcendental::domain_raise_served_half_value;
pub use kernel_transcendental::pow_arguments_provably_raise;
pub use kernel_transcendental::sqrt_argument_is_known_negative;
pub use kernel_transcendental::trig_argument_is_known_infinite;
pub use kernel_transcendental::DomainLimitedFamily;
pub use kernel_transcendental::DomainRaiseClassification;
pub use single_value_family::rounding_argument_raises;

/// `math_call_result` is the FROZEN entry point: `function` is the
/// attribute name after `math.` ("floor", "sqrt", …); `arguments` are
/// the already-evaluated operands in call order; `kernel` answers
/// `floor`'s own set-valued row (`floor_call_over_set`'s own doc) and
/// `sqrt`'s own set-valued row (`sqrt_call_over_set`'s own doc) when the
/// operand is a bounded numeric set rather than one known value.
/// `None` means "not modeled" — the caller declines, same honesty as
/// every other B4 row in PYREFLY-NUMERIC-B3-B4.md.
///
/// Modeled EXACTLY (each an exactly-decidable row cited above):
/// `floor`, `ceil`, `trunc` — each on a FINITE argument only
/// (`integral_domain_admits`: they return an `Integral`, and CPython
/// raises `OverflowError`/`ValueError` for ±inf/NaN, which is
/// `rounding_argument_raises`' row) — `isqrt`, `fabs`, `copysign`, and `sqrt` on
/// a known non-negative PERFECT-SQUARE operand (`sqrt_exact_perfect_square`'s
/// own doc — IEEE 754 correct rounding, not an approximation). `floor`,
/// `ceil`, `trunc`, `sqrt`, and `cbrt` additionally answer a bounded
/// numeric SET operand through the kernel's own
/// `Floor`/`Ceil`/`Trunc`/`Sqrt`/`Cbrt` transfers (`rounding_call_over_set`,
/// `sqrt_call_over_set`, `cbrt_call_over_set`).
///
/// Modeled EXACTLY through the `int` theory (`int_theory_call`'s own
/// doc, each row's pins clause named there): `factorial`, `gcd`, `lcm`,
/// `comb`, `perm`, and `isqrt` on a SET operand its concrete row above
/// cannot read. These ask `boundary/python.lean`'s own `int.*` arms and
/// answer Integer-sorted exact values, never the float image.
///
/// Modeled through the KERNEL-BACKED transcendental family
/// (`kernel_backed_unary_family_call`/`kernel_backed_atan2_call`'s own
/// doc, each row's pins clause named there): `exp`, `expm1`, `log`,
/// `log1p`, `log2`, `log10`, `sin`, `cos`, `tan`, `sinh`, `cosh`,
/// `tanh`, `asin`, `acos`, `atan`, `atan2`, `asinh`, `acosh`, `atanh` —
/// python-pins.md's explog.1–6 and trig.1–13 rows. Each poses the
/// operand's window to the same `boundary/javascript.lean` transfer
/// arm the JS adapter asks and answers a certified enclosure, never a
/// bare sort; a domain-violating known operand (where CPython raises
/// `ValueError` rather than returning a value) answers `None` here
/// instead of the kernel's own `NaN` reading — see
/// `kernel_backed_unary_family_call`'s own doc. A decline from ANY of
/// these 19 names — an unread operand shape, a provable Python raise,
/// or the kernel arm's own served-shape gap — is FINAL: it does NOT
/// fall through to the sort-only row below, because that row's
/// `float_sorted_unknown()` claim ("some float value exists") would be
/// exactly as false on a raise, or stronger than the kernel itself was
/// willing to claim.
///
/// Modeled at SORT-ONLY precision (`approximated_family_result`'s own
/// doc) still: `sqrt` on a non-perfect-square operand, `cbrt`, `hypot`
/// — every argument known answers `float_sorted_unknown()` (a
/// Float-tagged all-numbers set), never a specific value;
/// `math.sqrt` on a known negative argument answers `None` here
/// because it provably RAISES instead (see
/// `sqrt_argument_is_known_negative`, read by `provable_raise`). None
/// of the 19 kernel-backed names above reach this row — see
/// `approximated_family_result`'s own doc for why.
///
/// Still declined for a VALUE here (no cited row this wave, and not
/// sort-only-graded either): `pow`, `fsum`, `remainder`, `fmod`,
/// `degrees`, `radians`, `nextafter`,
/// `ulp`, `frexp`, `ldexp`, `modf`, `dist`, `prod` — every one of them
/// falls through to `None` here. `pow`'s own PROVABLE-RAISE half is
/// modeled separately: `pow_arguments_provably_raise`'s own doc,
/// read by `expressions.rs::call_provable_raise` — a known negative
/// finite base under a known finite non-integer exponent provably
/// raises `ValueError` before this function is ever asked for a
/// value, the same "raise channel, not a value row" split
/// `sqrt_argument_is_known_negative` already keeps for `math.sqrt`.
/// Constants (`math.pi`, `math.e`, `math.tau`, `math.inf`,
/// `math.nan`) are attribute reads, not calls — out of scope for this
/// function entirely; see `math_constant_value` for those (each answers
/// its exact CPython value, see its own doc).
pub fn math_call_result(
    function: &str,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    match function {
        "floor" => {
            let first = arguments.first()?;
            match single_numeric_operand(first) {
                Some((value, _)) => floor_call(value),
                // no single known value — try the bounded-set row
                // (e.g. `math.floor(random.random() * 121)`) before
                // declining
                None => rounding_call_over_set(TransferQuestionOp::Floor, first, kernel),
            }
        }
        "ceil" => {
            let first = arguments.first()?;
            match single_numeric_operand(first) {
                Some((value, _)) => ceil_call(value),
                // no single known value — try the bounded-set row
                // (e.g. `math.ceil(x)` over a declared `0.1 <= x <=
                // 0.9` window) before declining
                None => rounding_call_over_set(TransferQuestionOp::Ceil, first, kernel),
            }
        }
        "trunc" => {
            let first = arguments.first()?;
            match single_numeric_operand(first) {
                Some((value, _)) => trunc_call(value),
                // no single known value — try the bounded-set row
                // the same way `ceil`/`floor` do
                None => rounding_call_over_set(TransferQuestionOp::Trunc, first, kernel),
            }
        }
        // the concrete row first (no kernel round trip for a known
        // nonnegative int), then the kernel's own `int.isqrt` for a SET
        // operand — pow.4's row
        "isqrt" => match single_numeric_operand(arguments.first()?) {
            Some((value, is_int)) => isqrt_call(value, is_int),
            None => int_theory_call(function, arguments, kernel),
        },
        // the exact `int` theory serves these outright — no pure-Rust
        // row above computes them, so every operand shape they answer
        // comes from the kernel (`int_theory_call`'s own doc names each
        // row's pins clause)
        "factorial" | "gcd" | "lcm" | "comb" | "perm" => int_theory_call(function, arguments, kernel),
        // the three float PREDICATES, each stating a `True`/`False`
        // return outright: math.rst — `isnan(x)` "Return ``True`` if *x*
        // is a NaN (not a number), and ``False`` otherwise"; `isinf(x)`
        // "Return ``True`` if *x* is a positive or negative infinity, and
        // ``False`` otherwise"; `isfinite(x)` "Return ``True`` if *x* is
        // neither an infinity nor a NaN, and ``False`` otherwise." A
        // single known operand decides which of the two; any other
        // operand shape still answers the exact two-member boolean
        // domain, since the return is a `bool` whatever the argument is.
        "isnan" | "isinf" | "isfinite" => float_predicate_call(function, arguments),
        "fabs" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            fabs_call(value)
        }
        // both arguments known answers the exact IEEE 754 recombination
        // (`copysign_call`'s own doc); a known magnitude with an
        // UNRESOLVED sign source still answers the two-signed-branch
        // set, or the single branch the sign source's own window
        // provably fixes (`copysign_call_over_unresolved_sign`'s own
        // doc) — an unresolved MAGNITUDE has no row here, since neither
        // function reads one.
        "copysign" => {
            let (magnitude, _) = single_numeric_operand(arguments.first()?)?;
            let sign_source = arguments.get(1)?;
            match single_numeric_operand(sign_source) {
                Some((value, _)) => copysign_call(magnitude, value),
                None => copysign_call_over_unresolved_sign(magnitude, sign_source, kernel),
            }
        }
        // `sqrt` on an exact perfect square answers the exact Float
        // result (IEEE 754 correct rounding — see
        // sqrt_exact_perfect_square's own doc); a KNOWN SET operand
        // (no single known value to read) asks the kernel's own `Sqrt`
        // transfer next (sqrt_call_over_set's own doc); any other sqrt
        // argument, and every other approximated-family function, falls
        // through to the sort-only row below
        "sqrt" => {
            let [only] = arguments else { return None };
            match single_numeric_operand(only) {
                Some((value, _)) => {
                    sqrt_exact_perfect_square(value).or_else(|| approximated_family_result(function, arguments))
                }
                None => sqrt_call_over_set(only, kernel).or_else(|| approximated_family_result(function, arguments)),
            }
        }
        // a KNOWN SET operand asks the kernel's own `Cbrt` transfer
        // directly (`cbrt_call_over_set`'s own doc); a known single
        // value still falls through to the sort-only row below — no
        // exact-perfect-cube shortcut is modeled here, the same posture
        // `sqrt`'s single-value row keeps for a non-perfect-square
        "cbrt" => {
            let [only] = arguments else { return None };
            match single_numeric_operand(only) {
                Some(_) => approximated_family_result(function, arguments),
                None => cbrt_call_over_set(only, kernel).or_else(|| approximated_family_result(function, arguments)),
            }
        }
        // trig.10 — the one two-argument member of the kernel-backed
        // family. A decline here (an unread operand shape, a provable
        // Python raise the kernel answered NaN for, or the kernel arm's
        // own served-quadrant gap answering Unknown —
        // `kernel_backed_atan2_call`'s own doc) is FINAL: it does NOT
        // fall through to `approximated_family_result`'s sort-only
        // `float_sorted_unknown()`, because that claim ("some float
        // value exists") is equally false on a raise, and no weaker
        // than the kernel's own Unknown on its serving gap — falling
        // through there would answer a value-bearing claim the kernel
        // itself just declined to make.
        "atan2" => {
            let [y, x] = arguments else { return None };
            kernel_backed_atan2_call(y, x, kernel)
        }
        // pow.8's own TWO-ARGUMENT `math.hypot(a, b)`. The exact-perfect-
        // square shortcut (`hypot_exact_perfect_square`, e.g. `hypot(3,
        // 4) == 5.0` exactly) is tried FIRST, the same "exact value
        // before a kernel window" order `sqrt`'s own arm above keeps for
        // `sqrt_exact_perfect_square` before `sqrt_call_over_set` — a
        // kernel-served bracketing window is a WEAKER, still-sound
        // claim than the exact value this file can already prove
        // outright, and answering it FIRST here would silently drop
        // that stronger claim. Only once the perfect-square shortcut
        // declines does this pose `TransferQuestionOp::Hypot`
        // (`kernel_backed_hypot_call`'s own doc: `transferHypot` serves
        // every finite operand window, no served-quadrant gap the way
        // `atan2`'s can have), and THAT decline is FINAL — it does not
        // fall through to `approximated_family_result`'s sort-only
        // `float_sorted_unknown()`, the same no-fallback reasoning
        // `atan2` above keeps (a sort-only claim is no weaker than the
        // kernel's own declined answer). The general VARIADIC form
        // (three or more coordinates) is a DIFFERENT arity this arm
        // does not match at all — it falls through to the `_` arm
        // below, which never reads "hypot" (`kernel_backed_unary_
        // family_op` excludes it), so it reaches `approximated_family_
        // result`'s own `hypot_exact_perfect_square`/sort-only rows
        // through `math_call_result`'s caller unchanged.
        "hypot" if arguments.len() == 2 => {
            let [a, b] = arguments else { unreachable!("len() == 2 checked above") };
            hypot_exact_perfect_square(arguments).or_else(|| kernel_backed_hypot_call(a, b, kernel))
        }
        // explog.1–explog.6, trig.1–trig.9, trig.11–trig.13: the
        // kernel's own transfer answers the window, or declines FINALLY
        // — same no-fallback reasoning as `atan2` above, and the same
        // reason these 18 names are no longer in
        // `approximated_family_result`'s own `APPROXIMATED_NAMES` list.
        // The sort-only approximated family still rides local Rust for
        // the names that stay there: `cbrt` (pow.6) — `kernel_backed_
        // unary_family_op`'s own doc names why it is excluded — plus
        // `sqrt` on a non-perfect-square falls through from its own
        // arm, and `hypot`'s own VARIADIC (three-or-more-argument) form
        // falls through from the `_` arm below (the two-argument form
        // is served above, one match arm up): float_sorted_unknown()
        // once every argument is known, per approximated_family_
        // result's own doc.
        _ => match kernel_backed_unary_family_op(function) {
            Some(op) => {
                let [only] = arguments else { return None };
                kernel_backed_unary_family_call(function, op, only, kernel)
            }
            None => approximated_family_result(function, arguments),
        },
    }
}
