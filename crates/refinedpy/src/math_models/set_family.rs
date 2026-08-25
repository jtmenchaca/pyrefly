use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::PowOperandKind;
use refined_kernel::transfer_questions::PowOperandWire;
use refined_kernel::transfer_questions::TransferAnswerKind;
use refined_kernel::transfer_questions::TransferQuestion;
use refined_kernel::transfer_questions::TransferQuestionOp;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;

use super::integral_domain_admits;

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
pub(super) fn enclosure_is_provably_finite(set: &RefinedSet) -> bool {
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

/// `math.floor`/`math.ceil`/`math.trunc` on a KNOWN NUMERIC SET (a
/// seeded range, or a bounded set another transfer already produced,
/// e.g. `random.random() * 121`, or `0.1 <= x <= 0.9`): the kernel's
/// own `Floor`/`Ceil`/`Trunc` transfer answers the rounded enclosure
/// directly, so the half-open/closed distinction at the set's own
/// bounds is the kernel's proved arithmetic, never a bound this file
/// recomputes by hand (https://docs.python.org/3.12/library/math.html —
/// `math.floor`/`math.ceil`/`math.trunc`'s own entries, each "delegates
/// to `x.__floor__`/`__ceil__`/`__trunc__`, which should return an
/// Integral value"; the set-valued argument rounds to Integer sort the
/// same way the single-value rows above do). A non-numeric-sorted set,
/// or a kernel refusal on this set shape, declines to `None` — the same
/// honesty every other row in this file keeps. `op` must be one of
/// `Floor`/`Ceil`/`Trunc` — every caller of this function already fixes
/// one of those three at its own call site.
///
/// The ANSWER must additionally be provably finite
/// (`enclosure_is_provably_finite`): the same `Integral` domain each
/// single-value row gates on (`integral_domain_admits`) applies to a
/// set-shaped answer, and a kernel enclosure whose bound is `±inf`
/// describes a result the real call would raise `OverflowError` on
/// rather than return. The kernel is answering its own question
/// correctly there — `binary64.floor(inf)` IS `inf` — so this is the
/// adapter declining to read that float answer as a Python `int`, not a
/// kernel disagreement.
pub(super) fn rounding_call_over_set(
    op: TransferQuestionOp,
    value: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
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
            op,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    // the same `Integral` domain each single-value row gates on: an
    // infinite result is one the real call raises OverflowError for
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

/// `math.sqrt(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded
/// set another transfer already produced): the kernel's own `Sqrt`
/// transfer answers the square-rooted enclosure directly, the exact
/// mirror of `rounding_call_over_set` above — same `TransferQuestion`
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
pub(super) fn sqrt_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
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
pub(super) fn cbrt_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
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
