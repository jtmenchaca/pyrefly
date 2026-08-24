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

use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::nan_value;
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
use refined_sets::refinement_forms::above;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::union;
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
/// sort always, regardless of the operand's own sort. `fabs_call`'s own
/// doc states `math.fabs(nan)` returns `nan` normally (a real value,
/// not a raise), so this wrapper answers `nan_value()` — the domain's
/// own NaN state (`refined_domain::abstract_value::nan_value`) — rather
/// than let a bare NaN enter `known_values`, which no refined set
/// admits (`refinement_forms::element`'s own construction-time refusal).
fn float_result(value: f64) -> AbstractValue {
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
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Floor,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
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
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op,
            a,
            b,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
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
            let operand = int_transferable_operand(only)?;
            if let Some(answer) = int_transfer_call(TransferQuestionOp::IntIsqrt, operand.clone(), empty, grade, kernel) {
                return Some(answer);
            }
            isqrt_as_sqrt_floor_composition(&operand, grade, kernel)
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

/// `math.isqrt(n)` over a NON-SINGLETON `RefinedSet` (a seeded
/// parameter range, or a bounded set another transfer already
/// produced) — the FALLBACK `int_theory_call`'s own `isqrt` row takes
/// once `int.isqrt` (`boundary/python.lean`) has already declined on
/// it. `int.isqrt` matches `exactNatOf` on its one operand only — it
/// carries no general-window arm the way `int.floorDiv` does
/// (`shift_as_int_composition`'s own doc names the identical gap for
/// `int.mul`), so a bounded, non-singleton `n` always reads back
/// `.unknown` from `int.isqrt` and would decline outright with no
/// further attempt.
///
/// pow.4 states `math.isqrt(n)` IS "the floor of the exact square
/// root" — the same composition CPython's own implementation performs
/// — so this asks `binary64.sqrt` (`sqrt_call_over_set`'s own kernel
/// row, which DOES carry a general-window decider) over `operand`,
/// then floors that Float-sorted enclosure through `binary64.floor`
/// (`floor_call_over_set`'s own row) back to Integer sort. Both rows
/// already exist and are already exercised by `math.sqrt`/`math.floor`
/// over a declared range elsewhere in this file; this composes them
/// rather than adding a third kernel-transfer implementation.
///
/// Gated on `operand` PROVABLY EXCLUDING every negative value first
/// (`kernel.scalar_subset(operand, [0, +inf))`, the same "prove, never
/// guess" discipline `divisor_provably_excludes_zero`
/// (`expressions.rs`) keeps for its own zero exclusion): `math.isqrt`
/// raises `ValueError` on a negative n (`isqrt_call`'s own doc), and
/// `math.sqrt` raises `ValueError` on one too (pow.3) — composing
/// sqrt+floor over a window that ADMITS a negative would either
/// silently answer a value the real call never produces (the negative
/// arm raises) or hand `binary64.sqrt` an operand its own domain
/// excludes. `int_transferable_operand`'s caller already only reaches
/// this fallback with an Integer-sorted `operand`, which `int.isqrt`
/// itself already requires — the composition adds no new sort
/// admission, only a wider VALUE reading of the same sort.
fn isqrt_as_sqrt_floor_composition(
    operand: &RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let nonneg = make_refined_set(vec![at_least(0.0)]);
    if !(kernel.scalar_subset)(operand, &nonneg) {
        return None;
    }
    let operand_value = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..known_set(operand.clone(), None, grade, SetKindTag::None) };
    let root = sqrt_call_over_set(&operand_value, kernel)?;
    floor_call_over_set(&root, kernel)
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
fn copysign_call_over_unresolved_sign(
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
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Sqrt,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
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

/// `math.cbrt(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded
/// set another transfer already produced): the kernel's own `Cbrt`
/// transfer (wire `"js.cbrt"`, `TransferQuestionOp::Cbrt`, already
/// posed as a UNARY operand — `transfer_op_is_unary`) answers the
/// cube-rooted enclosure directly, the exact mirror of
/// `sqrt_call_over_set` above — same `TransferQuestion` construction,
/// same `catch_unwind` refusal discipline, same `TransferAnswerKind`
/// match. library/math.rst's own clause: "Return the cube root of x" —
/// unlike `sqrt`, `cbrt` is total over every real (the cube root of a
/// negative is itself negative), so no negative-operand exclusion is
/// needed here the way `sqrt_argument_is_known_negative` gates the
/// single-value `sqrt` row. Float sort ALWAYS (the same blanket rule
/// `sqrt_call_over_set` cites), regardless of the operand set's own
/// sort. A non-numeric-sorted set, or a kernel refusal on this set
/// shape, declines to `None`.
fn cbrt_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
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
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Cbrt,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
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

/// The operand a one-argument float transcendental question can be
/// posed over: a known single numeric value reads as the one-element
/// set `{v}` (the same "known value → singleton set" reading
/// `int_transferable_operand` performs for the `int` theory, widened
/// to every numeric sort since these questions are not integer-only),
/// and an already-numeric-sorted `Kind::Set` reads as its own set —
/// the same operand shape `sqrt_call_over_set`/`floor_call_over_set`
/// pose, generalized to accept a known SINGLE value too (`transferExp`
/// and its siblings answer a bracketing window even for a singleton
/// operand, since none of this family is exactly computable at an
/// arbitrary interior point — the pins table's own "implementation-
/// approximated interior" note). A boolean-sorted or non-numeric
/// operand declines.
fn float_transferable_operand(value: &AbstractValue) -> Option<RefinedSet> {
    if let Some((number, _)) = single_numeric_operand(value) {
        return Some(make_refined_set(vec![one_of(&[number])]));
    }
    if value.kind == Kind::Set
        && matches!(
            value.kind_tag,
            Some(PrimitiveKind::Integer)
                | Some(PrimitiveKind::Float)
                | Some(PrimitiveKind::Boolean)
                | Some(PrimitiveKind::Number)
        )
    {
        return Some(value.set.clone());
    }
    None
}

/// The DOMAIN-LIMITED members of the kernel-backed family and the exact
/// window each one raises `ValueError` over in CPython — verified
/// against `tmp/cpython/Modules/mathmodule.c`, not against the kernel's
/// own JavaScript-facing `.nan` corner, because the two do NOT always
/// agree at the boundary:
///
/// - `log`/`log2`/`log10`: `loghelper` routes a float argument through
///   `math_1(arg, func, 0)` (`can_overflow = 0`). `m_log`/`m_log2`/
///   `m_log10` (mathmodule.c) return `-HUGE_VAL` (an INFINITE result)
///   at `x == 0.0` — a finite input — and `math_1`'s own rule ("an
///   infinite result from finite inputs causes... ValueError if
///   can_overflow is 0") fires there, so `math.log(0.0)` RAISES rather
///   than returning `-inf`. The kernel's `logCorners` answers the
///   value `-inf` at that same point (JavaScript's `Math.log(0) ===
///   -Infinity`) — a real JS/Python divergence at exactly one point.
///   The raise domain is therefore `x <= 0` (CLOSED at zero), one wider
///   than the kernel's own open `x < 0` NaN corner. Cited by
///   specifications/python/Doc/library/math.rst:696-698, whose own
///   worked example is `log(0.0)`.
/// - `log1p`: `FUNC1(log1p, m_log1p, 0, ...)` — same `can_overflow = 0`
///   rule. The platform `log1p(-1.0)` returns `-inf` (an infinite
///   result from a finite input), so `math.log1p(-1.0)` RAISES —
///   diverging from the kernel's `jsLog1p`, which serves the exact
///   value `-inf` there (`Eqv d ⟨-1,0⟩`). The raise domain is `x <=
///   -1` (closed), one wider than the kernel's own open `x < -1` NaN
///   corner.
/// - `asin`/`acos`: `FUNC1(asin, asin, 0, ...)` / `FUNC1(acos, acos, 0,
///   ...)`, the platform libm functions directly. `|x| = 1` is finite
///   (`asin(1) = pi/2`, `acos(-1) = pi`) — no infinite-result rule
///   fires there, so the raise domain is `|x| > 1` (OPEN), matching the
///   kernel's own boundary exactly: no divergence.
/// - `atanh`: `FUNC1(atanh, atanh, 0, ...)`. The platform `atanh(±1.0)`
///   returns `±inf` (an infinite result from a finite input), so
///   `math.atanh(±1.0)` RAISES — diverging from the kernel's
///   `jsAtanh`, which serves `±inf` there. The raise domain is `|x| >=
///   1` (closed), matching `atanh_sound.lean`'s own "`1 ± x <= 0`"
///   domain-error comment, one wider than a naive open reading.
/// - `acosh`: `FUNC1(acosh, acosh, 0, ...)`. `x = 1` is finite
///   (`acosh(1) = 0`) — the raise domain is `x < 1` (OPEN), matching
///   the kernel's own boundary exactly: no divergence.
///
/// Each row's SERVED half — the window's complement against the raise
/// domain — is what `served_half_window` intersects the operand
/// against for the straddling case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainLimitedFamily {
    Log,
    Log2,
    Log10,
    Log1p,
    Asin,
    Acos,
    Atanh,
    Acosh,
}

impl DomainLimitedFamily {
    /// The `math.*` attribute name this family answers, or `None` for
    /// every other function — the one place a name string is read into
    /// this enum, so every caller (the value dispatch, `expressions.rs`'s
    /// fire arms) shares one recognition.
    pub fn of_function(function: &str) -> Option<DomainLimitedFamily> {
        match function {
            "log" => Some(DomainLimitedFamily::Log),
            "log2" => Some(DomainLimitedFamily::Log2),
            "log10" => Some(DomainLimitedFamily::Log10),
            "log1p" => Some(DomainLimitedFamily::Log1p),
            "asin" => Some(DomainLimitedFamily::Asin),
            "acos" => Some(DomainLimitedFamily::Acos),
            "atanh" => Some(DomainLimitedFamily::Atanh),
            "acosh" => Some(DomainLimitedFamily::Acosh),
            _ => None,
        }
    }

    fn transfer_op(self) -> TransferQuestionOp {
        match self {
            DomainLimitedFamily::Log => TransferQuestionOp::Log,
            DomainLimitedFamily::Log2 => TransferQuestionOp::Log2,
            DomainLimitedFamily::Log10 => TransferQuestionOp::Log10,
            DomainLimitedFamily::Log1p => TransferQuestionOp::Log1p,
            DomainLimitedFamily::Asin => TransferQuestionOp::Asin,
            DomainLimitedFamily::Acos => TransferQuestionOp::Acos,
            DomainLimitedFamily::Atanh => TransferQuestionOp::Atanh,
            DomainLimitedFamily::Acosh => TransferQuestionOp::Acosh,
        }
    }

    /// The window CPython raises `ValueError` over — this enum's own
    /// doc names the exact `mathmodule.c` clause behind each row.
    fn raise_domain(self) -> RefinedSet {
        match self {
            DomainLimitedFamily::Log | DomainLimitedFamily::Log2 | DomainLimitedFamily::Log10 => {
                make_refined_set(vec![at_most(0.0)])
            }
            DomainLimitedFamily::Log1p => make_refined_set(vec![at_most(-1.0)]),
            DomainLimitedFamily::Asin | DomainLimitedFamily::Acos => {
                make_refined_set(vec![union(make_refined_set(vec![below(-1.0)]), make_refined_set(vec![above(1.0)]))])
            }
            DomainLimitedFamily::Atanh => make_refined_set(vec![union(
                make_refined_set(vec![at_most(-1.0)]),
                make_refined_set(vec![at_least(1.0)]),
            )]),
            DomainLimitedFamily::Acosh => make_refined_set(vec![below(1.0)]),
        }
    }

    /// The window's COMPLEMENT — the served half — spelled directly
    /// rather than through a generic set-difference form, the same way
    /// `split_divisor_transfer`'s own negative/positive halves are
    /// spelled directly rather than built from a `Difference` node.
    fn served_domain(self) -> RefinedSet {
        match self {
            DomainLimitedFamily::Log | DomainLimitedFamily::Log2 | DomainLimitedFamily::Log10 => {
                make_refined_set(vec![above(0.0)])
            }
            DomainLimitedFamily::Log1p => make_refined_set(vec![above(-1.0)]),
            DomainLimitedFamily::Asin | DomainLimitedFamily::Acos => {
                make_refined_set(vec![at_least(-1.0), at_most(1.0)])
            }
            DomainLimitedFamily::Atanh => make_refined_set(vec![above(-1.0), below(1.0)]),
            DomainLimitedFamily::Acosh => make_refined_set(vec![at_least(1.0)]),
        }
    }

    /// CPython's own runtime message for every row in this family —
    /// `is_error` (mathmodule.c): `if (errno == EDOM) PyErr_SetString
    /// (PyExc_ValueError, "math domain error")` — one shared string
    /// across the whole module, not a per-function wording, matching
    /// `expressions.rs`'s existing `math.sqrt` raise arm.
    pub fn raise_message(self) -> &'static str {
        "this expression provably raises ValueError: math domain error"
    }
}

/// Whether a KNOWN operand's window is ENTIRELY inside a family's raise
/// domain, STRADDLES the boundary (admits both raising and non-raising
/// values), or is ENTIRELY inside the served domain — the three-way
/// read `expressions.rs`'s `call_provable_raise` (entirely-raises) and
/// `possible_raise` (straddles) both ask, mirroring
/// `divisor_is_provably_always_zero`/`divisor_provably_excludes_zero`'s
/// own `scalar_subset`/`scalar_disjoint` pair exactly — the same two
/// kernel questions, posed against this family's own `raise_domain()`
/// rather than the fixed `{0.0}` divisor does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainRaiseClassification {
    EntirelyRaises,
    Straddles,
    EntirelyServed,
}

/// Classifies a KNOWN operand (a single value or a bounded set) against
/// `family`'s raise domain. `None` when the operand cannot be read as a
/// transferable window at all (an unknown argument, a non-numeric
/// sort) — the caller declines exactly as every other unread shape in
/// this file does.
pub fn domain_raise_classification(
    family: DomainLimitedFamily,
    argument: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<DomainRaiseClassification> {
    let operand = float_transferable_operand(argument)?;
    let raise_domain = family.raise_domain();
    let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(&operand));
    if matches!(empty, Ok(true)) || empty.is_err() {
        return None;
    }
    let entirely_raises = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&operand, &raise_domain));
    if matches!(entirely_raises, Ok(true)) {
        return Some(DomainRaiseClassification::EntirelyRaises);
    }
    let entirely_served = crate::kernel_ask::ask_kernel(|| (kernel.scalar_disjoint)(&operand, &raise_domain));
    if matches!(entirely_served, Ok(true)) {
        return Some(DomainRaiseClassification::EntirelyServed);
    }
    Some(DomainRaiseClassification::Straddles)
}

/// The served half's kernel window for a STRADDLING operand — the exact
/// mirror of `split_divisor_transfer`'s own split-and-re-ask pattern,
/// narrowed to one half (this family's `served_domain()`) rather than
/// two, since a domain-limited unary function has one raise-side ray
/// and one served-side ray/interval, not two symmetric halves around a
/// point. Poses the operand's window INTERSECTED with the served
/// domain — never the raw operand window, which would ask
/// `js.log`/`js.asin`/… a question a raising sub-window makes unsound
/// for Python. `None` on a kernel refusal, an empty intersection (the
/// operand does not actually straddle — the caller's own
/// `domain_raise_classification` should have already ruled this out),
/// or a `NaN`/`Unknown` answer on the served half (a decline, never a
/// mis-answer — the same discipline `kernel_backed_unary_family_call`
/// keeps for the non-straddling case).
pub fn domain_raise_served_half_value(
    family: DomainLimitedFamily,
    argument: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let operand = float_transferable_operand(argument)?;
    let served_half = make_refined_set({
        let mut forms = operand.forms.clone();
        forms.extend(family.served_domain().forms.clone());
        forms
    });
    let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(&served_half));
    if matches!(empty, Ok(true)) || empty.is_err() {
        return None;
    }
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: family.transfer_op(),
            a: served_half,
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(argument));
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// Poses one KERNEL-BACKED question for the explog/trig family's
/// one-argument members (`Exp`, `Expm1`, `Log`, `Log1p`, `Log2`,
/// `Log10`, `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh`, `Asin`,
/// `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`) and reads the answer back
/// Float-sorted — the exact mirror of `sqrt_call_over_set`'s own
/// construction and refusal discipline, generalized to any unary
/// `TransferQuestionOp` and to a known-single-value operand via
/// `float_transferable_operand`.
///
/// A `TransferAnswerKind::NaN` answer declines to `None` rather than
/// answering a value — the same reasoning `sqrt_argument_is_known_negative`
/// already keeps for `sqrt`, generalized to the rest of the family
/// rather than restated per function.
///
/// For the six DOMAIN-LIMITED members (`DomainLimitedFamily::of_function`),
/// this function additionally gates the VALUE side against CPython's
/// own raise domain — `DomainLimitedFamily::raise_domain`'s own doc —
/// which is WIDER than the kernel's `.nan` corner for `log`/`log2`/
/// `log10`/`log1p`/`atanh` at exactly one boundary point each (the
/// JS-vs-Python divergence that enum documents). Without this gate,
/// `math.log(0.0)` would read the kernel's served `-inf` value as a
/// Python return, when the real call raises there instead. A window
/// that STRADDLES the raise domain (some served values, some raising)
/// still declines HERE — `expressions.rs`'s `possible_raise` sibling
/// asks `domain_raise_served_half_value` directly for that case, since
/// this function's own "one call, one answer" shape has no room to
/// speak the served HALF only.
fn kernel_backed_unary_family_call(
    function: &str,
    op: TransferQuestionOp,
    value: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if let Some(family) = DomainLimitedFamily::of_function(function) {
        match domain_raise_classification(family, value, kernel) {
            Some(DomainRaiseClassification::EntirelyServed) => {}
            // EntirelyRaises: the real call never returns a value here —
            // `call_provable_raise`'s own row, not this function's to
            // answer. Straddles: only the served half determines, and
            // this function answers no partial value —
            // `possible_raise`'s own row reads `domain_raise_served_
            // half_value` directly. A classification refusal (`None`)
            // is the same unread-operand-shape decline every other row
            // in this file already gives.
            _ => return None,
        }
    }
    let operand = float_transferable_operand(value)?;
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op,
            a: operand,
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        // NaN: the real Python call raises rather than returning a
        // value — this function's own doc. Unknown: the kernel arm
        // itself declines this operand shape (e.g. `jsAtan2`'s
        // non-`x>0` quadrants — see `kernel_backed_atan2_call`).
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// `math.atan2(y, x)` — the one two-argument member of this family
/// (pins row trig.10). Poses `TransferQuestionOp::Atan2` over both
/// known operands; the exact two-operand mirror of
/// `kernel_backed_unary_family_call` above.
///
/// `jsAtan2` (`languages/javascript/trig/atan2.lean`) only serves the
/// `x > 0, y ≠ 0` quadrant today ("the axis and left-half-plane
/// corners wait on π pins," the file's own comment) and answers
/// `Unknown` — never `NaN` — everywhere else, so there is no raise-vs-
/// NaN divergence to gate here the way the log/asin/acos/atanh/acosh
/// family needs: `atan2` is total over the reals in Python
/// (library/math.rst's own clause states no domain restriction), and
/// an `Unknown` kernel answer is this arm's own current serving gap,
/// not a Python raise — it declines the same as every other unread
/// shape in this file.
fn kernel_backed_atan2_call(
    y: &AbstractValue,
    x: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let y_operand = float_transferable_operand(y)?;
    let x_operand = float_transferable_operand(x)?;
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Atan2,
            a: y_operand,
            b: x_operand,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, &[y.clone(), x.clone()]);
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// The explog/trig pins rows' own `TransferQuestionOp` election, one
/// per one-argument function name — the kernel operation column each
/// pins row (`explog.1`–`explog.6`, `trig.1`–`trig.9`, `trig.11`–
/// `trig.13`) now reads through `boundary/javascript.lean`'s shared
/// name-keyed transfer table (`"js.exp"`, `"js.sin"`, …), the same
/// table Python's own `int.*` arms register into
/// (`boundary/python.lean`'s own header: "Registered into the SAME
/// name-keyed transfer table... every wire op name is a flat string
/// key"). `atan2` (trig.10) is excluded — its own
/// `kernel_backed_atan2_call` above poses the two-operand question
/// directly. `hypot` (pow.8) is excluded: its own pins row states
/// plainly "no wire arm registered for the N-ary form" (Python's
/// variadic `math.hypot(*coordinates)` has no kernel election, only
/// JS's landed two-argument `js.hypot`). `cbrt` (pow.6) is excluded from
/// THIS table for the same reason `sqrt` is — its known-SET operand is
/// posed directly by its own dedicated row (`cbrt_call_over_set`),
/// mirroring `sqrt_call_over_set`, not folded into this shared-shape
/// dispatch table.
fn kernel_backed_unary_family_op(function: &str) -> Option<TransferQuestionOp> {
    match function {
        "exp" => Some(TransferQuestionOp::Exp),
        "expm1" => Some(TransferQuestionOp::Expm1),
        "log" => Some(TransferQuestionOp::Log),
        "log1p" => Some(TransferQuestionOp::Log1p),
        "log2" => Some(TransferQuestionOp::Log2),
        "log10" => Some(TransferQuestionOp::Log10),
        "sin" => Some(TransferQuestionOp::Sin),
        "cos" => Some(TransferQuestionOp::Cos),
        "tan" => Some(TransferQuestionOp::Tan),
        "sinh" => Some(TransferQuestionOp::Sinh),
        "cosh" => Some(TransferQuestionOp::Cosh),
        "tanh" => Some(TransferQuestionOp::Tanh),
        "asin" => Some(TransferQuestionOp::Asin),
        "acos" => Some(TransferQuestionOp::Acos),
        "atan" => Some(TransferQuestionOp::Atan),
        "asinh" => Some(TransferQuestionOp::Asinh),
        "acosh" => Some(TransferQuestionOp::Acosh),
        "atanh" => Some(TransferQuestionOp::Atanh),
        _ => None,
    }
}

/// The approximated float family still riding sort-only precision on a
/// KNOWN SINGLE-VALUE operand: `sqrt` on a non-perfect-square operand,
/// `cbrt` (a Set-shaped `cbrt` operand is `cbrt_call_over_set`'s own
/// row, tried first — this function is `math_call_result`'s `"cbrt"`
/// fallback only), `hypot` (a Set-shaped `hypot` operand still declines
/// outright — no kernel row exists for it, `kernel_backed_unary_family_op`'s
/// own doc) — `float_sorted_unknown()` (a Float-tagged, all-numbers SET)
/// once every argument is known, never a specific value. None of these
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
/// is never consulted for those 19 names.
fn approximated_family_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
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
fn hypot_exact_perfect_square(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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

/// `math.pi` / `math.e` / `math.tau` / `math.inf` / `math.nan` —
/// ATTRIBUTE READS, not calls (library/math.rst, "Constants" section:
/// `data:: pi`/`data:: e`/`data:: tau`/`data:: inf`/`data:: nan`, each
/// "to available precision" or, for `inf`/`nan`, "Equivalent to the
/// output of `float('inf')`" / `float('nan')`. Each answers the EXACT
/// CPython value, not a sort-only approximation:
///
/// - `pi`/`e`/`tau`: `std::f64::consts::PI`/`E`/`TAU` ARE CPython's own
///   values — both are the nearest binary64 double to the mathematical
///   constant, and IEEE 754 binary64 has exactly one nearest
///   representable value for a given real number, so Rust's constant
///   and CPython's libm-derived constant are the same bit pattern.
/// - `inf`: `f64::INFINITY`, library/math.rst's own "Equivalent to the
///   output of `float('inf')`." `+inf` is a legal `RefinedSet` element
///   (`refinement_forms.rs`'s `element` helper panics ONLY on NaN, never
///   on an infinite operand — `one_of`/`at_least`/`above` all route
///   through it unchanged), matching the Lean kernel's own admission of
///   `+-infinity` as elements of R-bar (`refinement_forms.go`'s twin
///   comment) — so `known_values` is the ordinary, unmodified route for
///   this constant, the same route every other exact numeric constant
///   in this file takes.
/// - `nan`: `nan_value()` — the domain's own `Kind::NaN` carrier, NEVER
///   a value inside `known_values`: `element`'s construction-time panic
///   refuses NaN for every refined-set form, so a `one_of`/singleton
///   containing NaN cannot be built at all. This is the same NaN
///   carrier `float_result` reaches for elsewhere in this file.
pub fn math_constant_value(name: &str) -> Option<AbstractValue> {
    match name {
        "pi" => Some(known_values(vec![std::f64::consts::PI], PrimitiveKind::Float, TrustProved)),
        "e" => Some(known_values(vec![std::f64::consts::E], PrimitiveKind::Float, TrustProved)),
        "tau" => Some(known_values(vec![std::f64::consts::TAU], PrimitiveKind::Float, TrustProved)),
        "inf" => Some(known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved)),
        "nan" => Some(nan_value()),
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
/// own doc — IEEE 754 correct rounding, not an approximation). `floor`,
/// `sqrt`, and `cbrt` additionally answer a bounded numeric SET operand
/// through the kernel's own `Floor`/`Sqrt`/`Cbrt` transfers
/// (`floor_call_over_set`, `sqrt_call_over_set`, `cbrt_call_over_set`).
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
/// Still declined (no cited row this wave, and not sort-only-graded
/// either): `pow`, `fsum`, `remainder`, `fmod`, `degrees`, `radians`,
/// `isnan`, `isinf`, `isfinite`, `nextafter`, `ulp`, `frexp`, `ldexp`,
/// `modf`, `dist`, `prod` — every one of them falls through to
/// `None`. Constants (`math.pi`, `math.e`, `math.tau`, `math.inf`,
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
        // explog.1–explog.6, trig.1–trig.9, trig.11–trig.13: the
        // kernel's own transfer answers the window, or declines FINALLY
        // — same no-fallback reasoning as `atan2` above, and the same
        // reason these 18 names are no longer in
        // `approximated_family_result`'s own `APPROXIMATED_NAMES` list.
        // The sort-only approximated family still rides local Rust for
        // the names that stay there: `cbrt` (pow.6) and `hypot`
        // (pow.8) — `kernel_backed_unary_family_op`'s own doc names why
        // both are excluded here — plus `sqrt` on a non-perfect-square
        // falls through from its own arm: float_sorted_unknown() once
        // every argument is known, per approximated_family_result's own
        // doc.
        _ => match kernel_backed_unary_family_op(function) {
            Some(op) => {
                let [only] = arguments else { return None };
                kernel_backed_unary_family_call(function, op, only, kernel)
            }
            None => approximated_family_result(function, arguments),
        },
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

    /// `math.hypot(1, 1)` — `sqrt(1**2 + 1**2) == sqrt(2)`, irrational:
    /// no perfect-square shortcut applies, so this still answers the
    /// SORT-ONLY set `hypot_exact_perfect_square`'s own doc names as the
    /// fallback (`approximated_family_result`'s `float_sorted_unknown()`),
    /// the row `test_hypot_known_arguments_answer_sort_only` used to pin
    /// on the perfect-square pair before this wave gave that pair its own
    /// exact row.
    #[test]
    fn test_hypot_of_a_non_perfect_square_still_answers_sort_only() {
        let Some(result) = math_call("hypot", &[float_operand(1.0), float_operand(1.0)]) else { return };
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

    /// `math.atan2(1.0, -1.0)`: `x < 0` is outside `jsAtan2`'s served
    /// quadrant ("the axis and left-half-plane corners wait on π
    /// pins") — the kernel answers `Unknown`, which is this arm's own
    /// serving gap (not a Python raise), so the call declines and
    /// falls through rather than mis-answering.
    #[test]
    fn test_atan2_outside_served_quadrant_declines() {
        if loaded_kernel().is_none() {
            return;
        }
        let result = math_call("atan2", &[float_operand(1.0), float_operand(-1.0)]);
        assert_eq!(result, None, "atan2's kernel arm does not yet serve x <= 0 — must decline, not guess");
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
}
