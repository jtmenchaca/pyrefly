use std::sync::Arc;

use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::above;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Operator;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::env::Environment;
use crate::expressions::evaluate_expression;

use super::kernel_transfer::divisor_provably_excludes_zero;
use super::known_values::single_numeric_value;
use super::sequence_row::datetime_difference_provable_raise;

/// `a / b` for a divisor window `b` that ADMITS zero (`0.0 ∈ b`) but is
/// not itself always zero (`divisor_is_provably_always_zero` already owns
/// the always-zero case as an unconditional raise, in `binop_provable_raise`
/// below). CPython's `/` still raises `ZeroDivisionError` on the zero arm
/// of such a window (arith.10), so this never asks the kernel with `b`
/// itself — that would ask `binary64.div` a question the divisor's own
/// zero member makes unsound for a Python `/`. Instead it splits `b`
/// around zero into its strictly-negative half (`b ∩ below(0)`) and its
/// strictly-positive half (`b ∩ above(0)`) — each half PROVABLY excludes
/// zero by construction — asks `binary64.div` on `a` against each half
/// separately, and unions whichever halves answer into one `RefinedSet`.
/// A half whose intersection with `b` is empty (`kernel.scalar_empty`,
/// e.g. `b` is entirely negative so its positive half is vacuous) is
/// skipped rather than asked, matching `divisor_is_provably_always_zero`'s
/// own empty-set guard.
///
/// This determines the VALUE question on every path that does not raise;
/// the zero arm itself is a MAY-RAISE this function does not speak to —
/// `binop_provable_raise` only fires an unconditional raise when the
/// entire divisor window is zero, so a window that merely ADMITS zero
/// alongside other values raises on SOME inputs and returns a value on
/// others. Reporting that raise arm as its own diagnostic (rather than
/// leaving it to CPython at runtime) is future work; no existing
/// possibly-raising expression in this file reports a partial-raise
/// arm alongside its value determination, so this function returns only
/// the sound value binding over the non-raising split, exactly as every
/// other admitted transfer answer already does.
pub(in crate::expressions) fn split_divisor_transfer(
    left_set: RefinedSet,
    right_set: &RefinedSet,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<refined_kernel::transfer_questions::TransferAnswer> {
    use refined_kernel::transfer_questions::TransferAnswerKind;
    use refined_kernel::transfer_questions::TransferQuestion;
    use refined_kernel::transfer_questions::TransferQuestionOp;

    let ask_half = |divisor_half: RefinedSet| -> Option<refined_kernel::transfer_questions::TransferAnswer> {
        let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(&divisor_half));
        if matches!(empty, Ok(true)) || empty.is_err() {
            return None;
        }
        let asked = crate::kernel_ask::ask_kernel(|| {
            (kernel.transfer)(&TransferQuestion {
                op: TransferQuestionOp::Div,
                a: left_set.clone(),
                b: divisor_half,
                c: 0.0,
                base: refined_kernel::transfer_questions::PowOperandWire {
                    kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
                    set: make_refined_set(vec![]),
                },
                exp: refined_kernel::transfer_questions::PowOperandWire {
                    kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
                    set: make_refined_set(vec![]),
                },
            })
        });
        asked.ok()
    };

    let negative_half = make_refined_set({
        let mut forms = right_set.forms.clone();
        forms.push(below(0.0));
        forms
    });
    let positive_half = make_refined_set({
        let mut forms = right_set.forms.clone();
        forms.push(above(0.0));
        forms
    });

    let negative_answer = ask_half(negative_half);
    let positive_answer = ask_half(positive_half);

    // A may-be-NaN answer on either half must never masquerade as a
    // NaN-free result — the whole split declines rather than silently
    // drop the NaN-carrying half's values.
    if matches!(negative_answer.as_ref().map(|a| a.kind), Some(TransferAnswerKind::NaN))
        || matches!(positive_answer.as_ref().map(|a| a.kind), Some(TransferAnswerKind::NaN))
    {
        return None;
    }

    match (negative_answer, positive_answer) {
        (None, None) => None,
        (Some(only), None) | (None, Some(only)) => Some(only),
        (Some(neg), Some(pos)) => Some(union_transfer_answers(neg, pos)),
    }
}

/// Unions two `TransferAnswer`s of the SAME kind family into one answer:
/// `Values` concatenates (both sides are exact singleton sets); `Set`
/// unions the two enclosures via the grammar's own `Union` form; either
/// side reading `Unknown` widens the whole union to `Unknown` (an
/// enclosure the kernel could not narrow on one half narrows nothing
/// once joined with the other). NaN is never passed here —
/// `split_divisor_transfer` already declines before this is called.
pub(in crate::expressions) fn union_transfer_answers(
    a: refined_kernel::transfer_questions::TransferAnswer,
    b: refined_kernel::transfer_questions::TransferAnswer,
) -> refined_kernel::transfer_questions::TransferAnswer {
    use refined_kernel::transfer_questions::TransferAnswer;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    match (a.kind, b.kind) {
        (TransferAnswerKind::Values, TransferAnswerKind::Values) => {
            let mut values = a.values;
            values.extend(b.values);
            TransferAnswer {
                kind: TransferAnswerKind::Values,
                values,
                set: make_refined_set(vec![]),
            }
        }
        (TransferAnswerKind::Unknown, _) | (_, TransferAnswerKind::Unknown) => TransferAnswer {
            kind: TransferAnswerKind::Unknown,
            values: vec![],
            set: make_refined_set(vec![]),
        },
        _ => {
            let a_set = match a.kind {
                TransferAnswerKind::Values => make_refined_set(vec![one_of(&a.values)]),
                _ => a.set,
            };
            let b_set = match b.kind {
                TransferAnswerKind::Values => make_refined_set(vec![one_of(&b.values)]),
                _ => b.set,
            };
            TransferAnswer {
                kind: TransferAnswerKind::Set,
                values: vec![],
                set: make_refined_set(vec![union(a_set, b_set)]),
            }
        }
    }
}

/// `x / 0`, `x // 0`, `x % 0` — a known ZERO divisor provably raises
/// `ZeroDivisionError: division by zero` (expressions.rst §6.7:
/// "raise[s] ZeroDivisionError" for `/`/`//`/`%` when the right operand
/// is zero). The evaluation path (`binary_arithmetic_value`/
/// `transfer_over_sets`) already declines these to `unknown()` for the
/// VALUE question; this is the same zero-divisor check speaking the
/// fact as a provable raise rather than a silent decline — the value
/// path is unchanged.
///
/// Two shapes prove the divisor is ALWAYS zero, never SOMETIMES zero
/// (`provable_raise`'s own contract — a fire here means every real
/// execution raises, `check.rs::sink_value`'s doc): a known scalar
/// `0.0`/`-0.0` (`single_numeric_value`), or a `Kind::Set` divisor whose
/// entire real range is the singleton `{0.0}` — a seeded window that
/// has narrowed to nothing but zero, not merely a window that ADMITS
/// zero alongside other values (`age - age`'s own `[0, 0]` window
/// against a `/`, for instance). A wider window that only ADMITS zero
/// (e.g. `[0.0, 2.0]`) is a SOMETIMES-raises divisor, which this
/// function must NOT fire on — `possible_raise`/`binop_possible_raise`
/// below is that window's own row; firing an unconditional raise for a
/// mostly-nonzero window here would be a false positive, the same
/// overreach `rounding_argument_raises`' finite-argument gate avoids
/// on the value side.
pub(in crate::expressions) fn binop_provable_raise(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    // The MIXED-AWARENESS DATETIME SUBTRACTION row — a `Sub` this
    // function's own zero-divisor rows never speak to, decided from the
    // two operands' `aware` tags rather than from a divisor window
    // (`datetime_difference_provable_raise`'s own doc carries note (3)'s
    // citation). It sits ahead of the operator gate below because that
    // gate admits only `/`, `//`, and `%`; only a `Sub` evaluates the
    // operands for it, so no other operator pays for this row.
    if binop.op == Operator::Sub {
        let left_value = evaluate_expression(&binop.left, environment, kernel);
        let right_value = evaluate_expression(&binop.right, environment, kernel);
        if let Some(message) = datetime_difference_provable_raise(binop.op, &left_value, &right_value) {
            return Some((binop.range(), message));
        }
    }
    if !matches!(binop.op, Operator::Div | Operator::FloorDiv | Operator::Mod) {
        return None;
    }
    let right = evaluate_expression(&binop.right, environment, kernel);
    if let Some((right_value, _)) = single_numeric_value(&right) {
        if right_value == 0.0 {
            return Some((
                binop.range(),
                "this expression provably raises ZeroDivisionError: division by zero".to_owned(),
            ));
        }
        return None;
    }
    if right.kind == Kind::Set && divisor_is_provably_always_zero(&right.set, kernel) {
        return Some((
            binop.range(),
            "this expression provably raises ZeroDivisionError: division by zero".to_owned(),
        ));
    }
    None
}

/// `x / d`, `x // d`, `x % d` where `d`'s set ADMITS zero without being
/// entirely zero (e.g. `[0.0, 2.0]`) — a SOMETIMES-raises divisor: most
/// real executions clear it, and CPython raises `ZeroDivisionError` on
/// the zero arm of the window for all three operators alike
/// (expressions.rst, "Binary arithmetic operations": "Division by zero
/// raises the ZeroDivisionError exception" for `/`/`//`, "A zero right
/// argument raises the ZeroDivisionError exception" for `%`).
/// `divisor_provably_excludes_zero` gates the value question's OWN
/// silence for `/`'s shape (`transfer_over_sets`); this row asks the
/// same membership question this file already asks there, so the two
/// never disagree about which windows admit zero. `binop_provable_
/// raise`'s own always-zero rows are excluded by construction: a
/// divisor this function reads as NOT provably excluding zero is
/// either always-zero (that row's own claim, made there) or
/// sometimes-zero (this row's claim) — the caller decides which
/// question it is asking by which function it calls.
///
/// The three operators diverge only on the VALUE side of this same
/// corner, never on the RAISE side this function speaks to:
/// `split_divisor_transfer` is `/`'s own fix (`transfer_over_sets`'s own
/// gate, `op == Operator::Div`) — it determines a value over the
/// divisor's zero-excluded halves, so `/`'s fire here rides alongside a
/// determined value. `//` and `%` still ask the kernel over the WHOLE
/// zero-admitting window, which the kernel declines for a non-singleton
/// divisor (`admitted_int_transfer_op`'s row only ever answers over two
/// exact singletons) — so their fire here rides alongside a silent
/// value question, the value side wholly unchanged by this row.
/// `diagnostic_sentences::division_by_a_set_that_admits_zero` already
/// speaks generically to "this expression's divisor set" without
/// naming `/` specifically, so the one sentence serves all three
/// operators without inventing a sibling.
pub(in crate::expressions) fn binop_possible_raise(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if !matches!(binop.op, Operator::Div | Operator::FloorDiv | Operator::Mod) {
        return None;
    }
    let right = evaluate_expression(&binop.right, environment, kernel);
    if right.kind != Kind::Set {
        return None;
    }
    if divisor_is_provably_always_zero(&right.set, kernel) {
        // `binop_provable_raise`'s own row already speaks this window
        // as an unconditional raise — not this function's claim to make.
        return None;
    }
    if divisor_provably_excludes_zero(&right.set, kernel) {
        return None;
    }
    Some((binop.range(), crate::diagnostic_sentences::division_by_a_set_that_admits_zero()))
}

/// Whether a divisor SET's entire real range is nothing but zero — a
/// nonempty subset of `{0.0}` (`kernel.scalar_subset`, guarded by
/// `kernel.scalar_empty` since the empty set is vacuously a subset of
/// everything but names no real divisor to raise on). Both closures are
/// total over the scalar shapes this file builds (the same discipline
/// `divisor_provably_excludes_zero` and `assignability.rs`'s own
/// containment ask keep), so there is no refusal to catch here.
pub(in crate::expressions) fn divisor_is_provably_always_zero(divisor: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    let zero = make_refined_set(vec![one_of(&[0.0])]);
    let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(divisor));
    if matches!(empty, Ok(true)) || empty.is_err() {
        return false;
    }
    let subset = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(divisor, &zero));
    matches!(subset, Ok(true))
}
