//! Python arithmetic and unary operations: the exact-value fast path
//! over two known numeric operands (`known_values`), the kernel-asked
//! SET transfer over a seeded range or a sort-only unknown
//! (`kernel_transfer`), the `**` wire question's own window rows
//! (`power`), the divide/floor-divide/modulo raise conditions
//! (`division`), string/list/set/date-timedelta operand rows
//! (`sequence_row`), unary negation over a set (`unary`), and the
//! provable/possible-raise family for subscripts, calls, and the
//! `math` domain-limited functions (`raise_conditions`).
//!
//! The re-export block below is this module's one door: every row its
//! children implement is named there, whether or not a caller outside
//! the module reads that particular row today. A row with no current
//! reader is still part of the stated interface, so the block carries
//! `allow(unused_imports)` rather than being trimmed to today's
//! callers and re-grown one line at a time as callers appear.
#![allow(unused_imports)]

mod division;
mod kernel_transfer;
mod known_values;
mod power;
mod raise_conditions;
mod sequence_row;
mod unary;

pub(super) use division::binop_possible_raise;
pub(super) use division::binop_provable_raise;
pub(super) use division::divisor_is_provably_always_zero;
pub(super) use division::split_divisor_transfer;
pub(super) use division::union_transfer_answers;
pub(super) use kernel_transfer::admitted_int_transfer_op;
pub(super) use kernel_transfer::admitted_transfer_op;
pub(super) use kernel_transfer::divisor_provably_excludes_zero;
pub(super) use kernel_transfer::exact_nonnegative_integer;
pub(super) use kernel_transfer::float_mul_as_shift_fallback;
pub(super) use kernel_transfer::int_transfer_answer;
pub(super) use kernel_transfer::int_transfer_over_sets;
pub(super) use kernel_transfer::shift_as_int_composition;
pub(super) use kernel_transfer::transfer_over_sets;
pub(super) use kernel_transfer::transferable_numeric_operand;
pub use known_values::binary_arithmetic_value;
pub(super) use known_values::binary_arithmetic_pair;
pub(super) use known_values::multi_value_binary_arithmetic;
pub(super) use known_values::numeric_values_with_sort;
pub(super) use known_values::single_numeric_value;
pub(super) use power::pow_over_sets;
pub(super) use raise_conditions::call_provable_raise;
pub(super) use raise_conditions::callee_display_name;
pub(super) use raise_conditions::absent_receiver_possible_raise;
pub(super) use raise_conditions::domain_limited_family_possible_raise;
pub(super) use raise_conditions::eval_literal_value;
pub(super) use raise_conditions::known_container_index_absent;
pub(super) use raise_conditions::known_string_index_out_of_range;
pub(super) use raise_conditions::one_voice_raise_message;
// Carries wider visibility than the rest of this module's surface —
// `check::control`'s try walk reads it to decide whether a handler
// catches a provable raise — so it needs its own `pub(crate)` re-export
// rather than riding the `pub(super)` block above.
pub(crate) use raise_conditions::raised_exception_class;
pub(super) use raise_conditions::subscript_provable_raise;
pub(super) use sequence_row::date_timedelta_binop_value;
pub(super) use sequence_row::datetime_difference_provable_raise;
pub(super) use sequence_row::sequence_binop_value;
pub(super) use unary::negate_over_set;

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::Form;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprBinOp;
use ruff_python_ast::Operator;
use ruff_text_size::TextRange;

use crate::env::Environment;
use crate::expressions::datetime::binary_arithmetic_value_with_kernel;
use crate::expressions::evaluate_expression;

/// `binary_arithmetic_value` already falls through to
/// `sequence_binop_value` for a non-numeric operand pair (that
/// function's own doc — the same fallthrough the AugAssign callers
/// share), so a plain BinOp reads through the one shared entry point
/// too rather than re-run the same numeric-then-sequence dispatch a
/// second time.
pub(in crate::expressions) fn evaluate_binop(binop: &ExprBinOp, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let left = evaluate_expression(&binop.left, environment, kernel);
    let right = evaluate_expression(&binop.right, environment, kernel);
    // `x ^ x` over ONE name is exactly {0}: any int XORed with itself
    // (stdtypes.rst's binary bitwise table). Gated on the operand
    // reading as an integer sort — a non-int operand raises TypeError
    // at runtime instead of producing a value.
    if binop.op == Operator::BitXor {
        if let (Expr::Name(l), Expr::Name(r)) = (binop.left.as_ref(), binop.right.as_ref()) {
            if l.id == r.id && matches!(transferable_numeric_operand(&left), Some((_, PrimitiveKind::Integer))) {
                return known_values(vec![0.0], PrimitiveKind::Integer, TrustSpec);
            }
        }
    }
    if let Some(value) = boolean_bitwise_value(binop.op, &left, &right) {
        return value;
    }
    if let Some(value) = date_timedelta_binop_value(binop.op, &left, &right, kernel) {
        return value;
    }
    binary_arithmetic_value_with_kernel(binop.op, &left, &right, kernel)
}

/// `&`, `|`, `^` over TWO boolean operands answer a `bool`, not an int:
/// stdtypes.rst, "Boolean Type" — "When applying the bitwise operators
/// ``&``, ``|``, ``^`` to two booleans, they return a bool equivalent to
/// the logical operations 'and', 'or', 'xor'." So the result is the
/// exact two-member boolean domain whenever both operands are booleans
/// whose values are not both pinned; when both ARE pinned to one value
/// each, the ordinary exact-value path below computes the one answer, so
/// this row leaves that case alone.
///
/// Without this row, two `bool` parameters (each seeded as the
/// `one_of{0, 1}` `Kind::Set` `typereading::base_sort` builds) reach the
/// kernel's int-bitwise transfer, which states no bound for them — and a
/// downstream `int(...)` then widens to `int_image`'s unbounded ray.
fn boolean_bitwise_value(op: Operator, left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    if !matches!(op, Operator::BitAnd | Operator::BitOr | Operator::BitXor) {
        return None;
    }
    if !(is_boolean_domained(left) && is_boolean_domained(right)) {
        return None;
    }
    // both operands pinned to ONE value each: the exact-value path below
    // computes the single answer, which is strictly more than this row
    // states, so this row stands aside for it
    if single_numeric_value(left).is_some() && single_numeric_value(right).is_some() {
        return None;
    }
    Some(known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec))
}

/// Whether a value is a Python `bool` — either the Boolean-tagged
/// `Kind::Values` a narrowed `isinstance(x, bool)` seeds, or the
/// `one_of{0, 1}` `Kind::Set` a bare `b: bool` parameter seeds
/// (`typereading::base_sort`'s own `"bool"` row).
fn is_boolean_domained(value: &AbstractValue) -> bool {
    if value.kind == Kind::Values {
        return value.kind_tag == Some(PrimitiveKind::Boolean);
    }
    if value.kind != Kind::Set {
        return false;
    }
    let [only] = &value.set.forms[..] else { return false };
    only.form == Form::OneOf && !only.w.is_empty() && only.w.iter().all(|member| *member == 0.0 || *member == 1.0)
}

/// Whether `expression` (or a sub-expression `provable_raise`'s own
/// pre-order walk already cleared) has a SOMETIMES-raising corner: some
/// admitted operand values raise, the rest still produce a value this
/// file determines. A DIFFERENT claim from `provable_raise`'s
/// all-or-nothing one, and a DIFFERENT sink discipline follows from it
/// — the finding and the value both stand at whatever sink this
/// expression flows into; the sink decides how to combine them
/// (`check.rs`'s own wiring, not this file's). `Some((range, message))`
/// names the escaping expression's own range and the sentence
/// `diagnostic_sentences.rs` builds for it; `None` when no recognized
/// sometimes-raising shape applies.
///
/// Recognized rows, each cited in the function that decides it: a `/`,
/// `//`, or `%` divisor set that ADMITS zero without being entirely
/// zero (`binop_possible_raise`); a domain-limited `math` call whose
/// operand window straddles its raise domain
/// (`domain_limited_family_possible_raise`); and a method call whose
/// receiver admits `None` (`absent_receiver_possible_raise`).
pub fn possible_raise(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    match expression {
        Expr::BinOp(binop) => binop_possible_raise(binop, environment, kernel),
        Expr::Call(call) => domain_limited_family_possible_raise(call, environment, kernel)
            .or_else(|| absent_receiver_possible_raise(call, environment, kernel)),
        _ => None,
    }
}
