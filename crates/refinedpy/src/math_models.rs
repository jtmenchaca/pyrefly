/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `math.*` call transfers: the exactly-decidable slice of the `math`
//! module (`floor`, `ceil`, `trunc`, `isqrt`, `fabs`, `copysign`, and
//! `sqrt` on a known perfect square), the exact `int`-theory slice the
//! kernel serves (`factorial`, `gcd`, `lcm`, `comb`, `perm`, and
//! `isqrt` over a set — `int_theory_call`), PLUS the sort-only approximated
//! family (`sqrt` on any other operand, every trig/hyperbolic function,
//! `cbrt`, `exp`, `expm1`, `log`, `log1p`, `log2`, `log10`, `hypot`),
//! which answers `float_sorted_unknown()` — a Float-tagged all-numbers
//! SET, never a specific value — once every argument is known, so
//! assignability's sort-fire law can still refuse an int-sorted sink.
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

use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustLevel;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::PowOperandKind;
use refined_kernel::transfer_questions::PowOperandWire;
use refined_kernel::transfer_questions::TransferAnswerKind;
use refined_kernel::transfer_questions::TransferQuestion;
use refined_kernel::transfer_questions::TransferQuestionOp;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::RefinedSet;

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
fn integral_domain_admits(value: f64) -> bool {
    value.is_finite()
}

/// Whether a set the kernel answered describes only FINITE values — the
/// set-shaped twin of `integral_domain_admits`, for the arm that reads a
/// kernel enclosure back as a Python `int` result.
///
/// `±inf` ARE elements of the grammar (`refinement_forms`'s own module
/// note: "+-infinity are elements of R-bar and are admitted"), so a
/// bound or an admitted value can be infinite and the set is still
/// well-formed — it just describes a result no Python `int` can hold.
/// NaN cannot appear at all (`element` panics on it at construction), so
/// there is nothing to check for it here.
///
/// This reads the set's OWN top-level forms, the same shallow reading
/// `requires_integer` performs, looking through `Union`/`Difference` the
/// same way. A form this recognizer does not understand answers `false`
/// — an unread shape declines rather than being assumed finite, which is
/// the direction that keeps the gate honest.
fn enclosure_is_provably_finite(set: &RefinedSet) -> bool {
    if set.forms.is_empty() {
        // the unconstrained set — every real AND both infinities
        return false;
    }
    let mut bounded_below = false;
    let mut bounded_above = false;
    for form in &set.forms {
        match form.form {
            Form::AtLeast | Form::Above => {
                if !form.a.is_finite() {
                    // `atLeast(-inf)` constrains nothing; `atLeast(+inf)`
                    // admits only +inf
                    return false;
                }
                bounded_below = true;
            }
            Form::AtMost | Form::Below => {
                if !form.a.is_finite() {
                    return false;
                }
                bounded_above = true;
            }
            // an explicit value list is finite exactly when every value is
            Form::OneOf => {
                if form.w.iter().all(|v| v.is_finite()) {
                    return true;
                }
                return false;
            }
            Form::Union => {
                let (Some(left), Some(right)) = (form.a_.as_ref(), form.b.as_ref()) else {
                    return false;
                };
                // a union is finite only if BOTH arms are
                if !enclosure_is_provably_finite(left) || !enclosure_is_provably_finite(right) {
                    return false;
                }
                return true;
            }
            // a difference is finite when its left arm is — removing
            // values never adds an infinity
            Form::Difference => {
                let Some(left) = form.a_.as_ref() else {
                    return false;
                };
                if !enclosure_is_provably_finite(left) {
                    return false;
                }
                return true;
            }
            // `Integer`/`MultipleOf` narrow but do not bound; the
            // sequence shapes are not scalar sets at all
            Form::Integer | Form::MultipleOf => {}
            _ => return false,
        }
    }
    bounded_below && bounded_above
}

/// `math.floor(x)` on a known single numeric value: the exact
/// mathematical floor, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.floor —
/// "Return the floor of x, the largest integer less than or equal to
/// x... delegates to x.__floor__, which should return an Integral
/// value"). A non-finite operand declines — `integral_domain_admits`'s
/// own doc.
fn floor_call(value: f64) -> Option<AbstractValue> {
    if !integral_domain_admits(value) {
        return None;
    }
    Some(integer_result(value.floor()))
}

/// `math.floor(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded
/// set another transfer already produced, e.g. `random.random() * 121`):
/// the kernel's own `Floor` transfer answers the floored enclosure
/// directly, so the half-open/closed distinction at the set's own
/// bounds is the kernel's proved arithmetic, never a bound this file
/// recomputes by hand
/// (https://docs.python.org/3.12/library/math.html#math.floor — same
/// clause `floor_call`'s single-value row cites; the set-valued
/// argument still floors to Integer sort). A non-numeric-sorted set, or
/// a kernel refusal on this set shape, declines to `None` — the same
/// honesty every other row in this file keeps.
///
/// The ANSWER must additionally be provably finite
/// (`enclosure_is_provably_finite`): the same `Integral` domain
/// `floor_call`'s single-value row gates on applies to a set-shaped
/// answer, and a kernel enclosure whose bound is `±inf` describes a
/// result `math.floor` would raise `OverflowError` on rather than
/// return. The kernel is answering its own question correctly there —
/// `binary64.floor(inf)` IS `inf` — so this is the adapter declining to
/// read that float answer as a Python `int`, not a kernel disagreement.
fn floor_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer)
            | Some(PrimitiveKind::Float)
            | Some(PrimitiveKind::Boolean)
            | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Floor,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    }))
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    // the same `Integral` domain the single-value row gates on: an
    // infinite result is one `math.floor` raises OverflowError for
    match asked.kind {
        TransferAnswerKind::Values => {
            if !asked.values.iter().all(|v| integral_domain_admits(*v)) {
                return None;
            }
            Some(known_values(asked.values, PrimitiveKind::Integer, grade))
        }
        TransferAnswerKind::Set => {
            if !enclosure_is_provably_finite(&asked.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(asked.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// Poses one `int.*` question — the exact unbounded-integer theory
/// (`boundary/python.lean`'s `pythonTransferOfOp1`/`pythonTransferOfOp2`)
/// — and reads the answer back as an INTEGER-SORTED value. The exact
/// mirror of `floor_call_over_set` above: same `TransferQuestion`
/// construction, same `catch_unwind` refusal discipline, same
/// `TransferAnswerKind` match. `b` is the empty set for the one-operand
/// members.
///
/// Two guards `floor_call_over_set` does not need, both about the
/// unboundedness python-pins.md arith.1 states ("integers have unlimited
/// precision"): a non-integral answer declines (no `int.*` member can
/// produce one), and an answer past the f64-exact 2^53 window declines
/// because `boundary/encode_sets.lean`'s `encodeNumber` puts every
/// result through `roundNE` before it crosses the wire — a bigger result
/// arrives ROUNDED, and this file's carrier is f64, so claiming it as
/// exact would claim a value CPython never computes.
fn int_transfer_call(
    op: TransferQuestionOp,
    a: RefinedSet,
    b: RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (kernel.transfer)(&TransferQuestion {
            op,
            a,
            b,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    }))
    .ok()?;
    match asked.kind {
        TransferAnswerKind::Values => {
            if asked.values.iter().any(|v| v.fract() != 0.0 || v.abs() >= 2f64.powi(53)) {
                return None;
            }
            Some(known_values(asked.values, PrimitiveKind::Integer, grade))
        }
        // a SET answer must carry its own integrality before it is
        // tagged Integer-sorted — tagging one without that mark would
        // claim an integrality the kernel did not state
        TransferAnswerKind::Set => {
            if !requires_integer(&asked.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(asked.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// An operand an `int.*` question can be posed over: an int-sorted
/// `Kind::Set` reads as its own set, and a known single int-sorted value
/// reads as the one-element set `{v}`, so a set-vs-known-value pair
/// poses the same question a set-vs-set pair does — the same reading
/// `expressions.rs::transferable_numeric_operand` performs, narrowed to
/// the INT sort because every `int.*` member's domain is the integers
/// (python-pins.md arith.1). A Float-sorted operand declines: `math.gcd`
/// and friends raise `TypeError` on one, so there is no value to answer.
fn int_transferable_operand(value: &AbstractValue) -> Option<RefinedSet> {
    if let Some((number, is_int)) = single_numeric_operand(value) {
        if !is_int || number.fract() != 0.0 {
            return None;
        }
        return Some(make_refined_set(vec![one_of(&[number])]));
    }
    if value.kind == Kind::Set
        && matches!(value.kind_tag, Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Boolean))
    {
        return Some(value.set.clone());
    }
    None
}

/// The `int.*` rows this file serves where its own concrete paths
/// decline — a SET operand, or a known-value pair no pure-Rust row above
/// computes. Each names the pins row that elects it:
///
/// - `isqrt` → `int.isqrt` (pow.4: "the integer square root of the
///   nonnegative int n... the floor of the exact square root"). Tried
///   only after `isqrt_call`'s concrete path declines, so a known
///   nonnegative int still answers without a kernel round trip.
/// - `factorial` → `int.factorial` (arith.21: "exact int factorial,
///   raises `ValueError` if n is not integral or negative").
/// - `gcd`/`lcm` → `int.gcd`/`int.lcm` (arith.20: "exact
///   greatest-common-divisor / least-common-multiple... on the unbounded
///   `int` theory"). CPython's own signature is variadic
///   (`math.gcd(*integers)`); the kernel members are binary, so this
///   folds the arguments left-to-right through repeated asks —
///   associativity of gcd/lcm is what makes the fold equal the variadic
///   call, and a fold step the kernel declines declines the whole call
///   rather than answering a partial product.
/// - `comb`/`perm` → `int.comb`/`int.perm` (arith.21: "exact
///   combinatorial counts, same int theory"). `math.perm(n)` with the
///   count omitted defaults to `k = n`, per the same clause's
///   `perm(n, k=None)` signature.
///
/// A negative operand is NOT filtered here: the kernel arms read their
/// `Nat`-domain operands through `exactNatOf` (`boundary/python.lean`),
/// which answers `none` on a negative exact integer rather than
/// extending the theory function silently — so the refusal that
/// corresponds to CPython's `ValueError` is the kernel's own, not a
/// condition this file restates.
fn int_theory_call(
    function: &str,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let grade = derived_trust_level(TrustProved, arguments);
    let empty = make_refined_set(vec![]);
    match function {
        "isqrt" => {
            let [only] = arguments else { return None };
            int_transfer_call(TransferQuestionOp::IntIsqrt, int_transferable_operand(only)?, empty, grade, kernel)
        }
        "factorial" => {
            let [only] = arguments else { return None };
            int_transfer_call(
                TransferQuestionOp::IntFactorial,
                int_transferable_operand(only)?,
                empty,
                grade,
                kernel,
            )
        }
        // the variadic fold — gcd/lcm are associative, so folding the
        // binary member left-to-right computes the same value the
        // variadic call does
        "gcd" | "lcm" => {
            let op = if function == "gcd" { TransferQuestionOp::IntGcd } else { TransferQuestionOp::IntLcm };
            let (first, rest) = arguments.split_first()?;
            if rest.is_empty() {
                return None;
            }
            let mut accumulated = int_transferable_operand(first)?;
            let mut answer = None;
            for argument in rest {
                let next = int_transferable_operand(argument)?;
                let step = int_transfer_call(op, accumulated, next, grade, kernel)?;
                accumulated = int_transferable_operand(&step)?;
                answer = Some(step);
            }
            answer
        }
        "comb" => {
            let [n, k] = arguments else { return None };
            int_transfer_call(
                TransferQuestionOp::IntComb,
                int_transferable_operand(n)?,
                int_transferable_operand(k)?,
                grade,
                kernel,
            )
        }
        // `math.perm(n)` defaults k to n (functions' own `perm(n,
        // k=None)` signature, arith.21's clause)
        "perm" => {
            let n = arguments.first()?;
            let n_set = int_transferable_operand(n)?;
            let k_set = match arguments.get(1) {
                Some(k) => int_transferable_operand(k)?,
                None if arguments.len() == 1 => n_set.clone(),
                None => return None,
            };
            int_transfer_call(TransferQuestionOp::IntPerm, n_set, k_set, grade, kernel)
        }
        _ => None,
    }
}

/// `math.ceil(x)` on a known single numeric value: the exact
/// mathematical ceiling, Integer sort
/// (https://docs.python.org/3.12/library/math.html#math.ceil —
/// "Return the ceiling of x, the smallest integer greater than or
/// equal to x... delegates to x.__ceil__, which should return an
/// Integral value"). A non-finite operand declines —
/// `integral_domain_admits`'s own doc.
fn ceil_call(value: f64) -> Option<AbstractValue> {
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
fn trunc_call(value: f64) -> Option<AbstractValue> {
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
///
/// This row needs NO finiteness gate, unlike the `floor`/`ceil`/`trunc`
/// family above (`integral_domain_admits`): those return an `Integral`,
/// and no Python `int` is infinite or NaN, but `fabs` returns a FLOAT,
/// and `inf`/`nan` are ordinary Python floats. `math.fabs(float('inf'))`
/// is `inf` and `math.fabs(float('nan'))` is `nan` — both return
/// normally, so answering them is right rather than a missing check.
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

/// `math.sqrt(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded
/// set another transfer already produced): the kernel's own `Sqrt`
/// transfer answers the square-rooted enclosure directly, the exact
/// mirror of `floor_call_over_set` above — same `TransferQuestion`
/// construction, same `catch_unwind` refusal discipline, same
/// `TransferAnswerKind` match. Unlike `Floor` (always Integer sort),
/// `sqrt` is Float sort ALWAYS (library/math.html's own module intro:
/// "Except when explicitly noted otherwise, all return values are
/// floats" — the same blanket rule `fabs_call`/`copysign_call` cite),
/// regardless of the operand set's own sort. A negative-admitting
/// operand set is NOT excluded here — `math.sqrt` on a known negative
/// SINGLE value is `sqrt_argument_is_known_negative`'s own provable-raise
/// row, but a SET that merely ADMITS a negative member alongside
/// nonnegative ones has no single known value to raise on, so the
/// kernel's own enclosure answer (or refusal) is this row's only
/// determination — the same "known operands only" discipline the rest
/// of this file keeps, deferring to the kernel rather than guessing. A
/// non-numeric-sorted set, or a kernel refusal on this set shape,
/// declines to `None`.
fn sqrt_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer)
            | Some(PrimitiveKind::Float)
            | Some(PrimitiveKind::Boolean)
            | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Sqrt,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    }))
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
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

/// `random.random()` — library/random.rst, `function:: random()`:
/// "Return the next random floating-point number in the range `0.0 <=
/// X < 1.0`." A Float-tagged Set bounded to that half-open window
/// (`at_least(0.0)` meets `below(1.0)`, the same ray-intersection shape
/// `float_sorted_unknown()` builds over the unbounded ray) — narrower
/// than the sort-only all-numbers answer other approximated `math`
/// calls carry, since this clause pins the interval exactly, only the
/// specific real drawn within it. Scoped to this one function of the
/// `random` module; no other `random.*` call is modeled here.
pub fn random_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "random" || !arguments.is_empty() {
        return None;
    }
    let window = make_refined_set(vec![at_least(0.0), below(1.0)]);
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(window, None, TrustSpec, SetKindTag::None)
    })
}

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
/// own doc — IEEE 754 correct rounding, not an approximation). `floor`
/// and `sqrt` additionally answer a bounded numeric SET operand through
/// the kernel's own `Floor`/`Sqrt` transfers (`floor_call_over_set`,
/// `sqrt_call_over_set`).
///
/// Modeled EXACTLY through the `int` theory (`int_theory_call`'s own
/// doc, each row's pins clause named there): `factorial`, `gcd`, `lcm`,
/// `comb`, `perm`, and `isqrt` on a SET operand its concrete row above
/// cannot read. These ask `boundary/python.lean`'s own `int.*` arms and
/// answer Integer-sorted exact values, never the float image.
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
/// either): `pow`, `fsum`, `remainder`, `fmod`, `degrees`, `radians`,
/// `isnan`, `isinf`, `isfinite`, `nextafter`, `ulp`, `frexp`, `ldexp`,
/// `modf`, `dist`, `prod` — every one of them falls through to
/// `None`. Constants (`math.pi`, `math.e`, `math.tau`, `math.inf`,
/// `math.nan`) are attribute reads, not calls — out of scope for this
/// function entirely; see `math_constant_value` for those (`math.nan`
/// still excluded there, see its own doc).
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
                None => floor_call_over_set(first, kernel),
            }
        }
        "ceil" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            ceil_call(value)
        }
        "trunc" => {
            let (value, _) = single_numeric_operand(arguments.first()?)?;
            trunc_call(value)
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
        // the sort-only approximated family (trig, log, exp, hypot,
        // and sqrt on a non-perfect-square): float_sorted_unknown()
        // once every argument is known, per
        // approximated_family_result's own doc
        _ => approximated_family_result(function, arguments),
    }
}

#[cfg(test)]
mod tests {
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;

    use super::*;
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

    #[test]
    fn test_sin_known_argument_answers_sort_only() {
        let Some(result) = math_call("sin", &[float_operand(0.0)]) else { return };
        assert_eq!(result.kind, Kind::Set);
    }

    #[test]
    fn test_hypot_known_arguments_answer_sort_only() {
        let Some(result) = math_call("hypot", &[float_operand(3.0), float_operand(4.0)]) else { return };
        assert_eq!(result.kind, Kind::Set);
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

    #[test]
    fn test_log2_and_log10_answer_sort_only() {
        let Some(log2) = math_call("log2", &[float_operand(1024.0)]) else { return };
        assert_eq!(log2.kind, Kind::Set);
        let Some(log10) = math_call("log10", &[float_operand(1000.0)]) else { return };
        assert_eq!(log10.kind, Kind::Set);
    }

    /// `math.floor(random.random() * 121)` — the kernel's own `Mult`
    /// transfer carries the half-open `[0.0, 1.0)` window through
    /// multiplication by 121 to `[0.0, 121.0)`, and this file's
    /// `floor_call_over_set` asks the kernel's `Floor` transfer on that
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
