use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;

use super::kernel_transfer::transfer_over_sets;

/// `-x` over an INT-SORTED SET operand (a seeded parameter range, or a
/// set another transfer already produced) — the row `evaluate_unary`'s
/// known-single-value path cannot reach. python-pins.md arith.11: "unary
/// `-` yields the numeric negation (`__neg__`)... on ints rides `int.*`
/// exactly," electing `int.neg`, whose kernel arm is
/// `boundary/python.lean`'s `pythonTransferOfOp1`. The answer is
/// Integer-sorted: negation of an integer is an integer, and arith.1's
/// unlimited precision means it never wraps.
///
/// Only `USub` has a row here. `UAdd` over a set is the operand itself
/// and needs no kernel question, but answering it would restate a value
/// this function was handed rather than transfer one, so it is left to
/// the caller's own decline; `Invert` (`~x`) is `-(x+1)`, a composition
/// no pins row states as an `int.*` member; `Not` is decided before the
/// numeric guard entirely.
///
/// A Float-sorted set declines: `binary64.neg` is that row's own
/// election (arith.11's own float branch), a different question this
/// function does not pose. A kernel refusal reads as `None` through the
/// same `catch_unwind` discipline `transfer_over_sets` keeps.
pub(in crate::expressions) fn negate_over_set(op: UnaryOp, operand: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    if op != UnaryOp::USub || operand.kind != Kind::Set {
        return None;
    }
    // `-x` IS `0 - x` on every numeric operand (arith.11's negation is
    // arith.1's exact subtraction from zero), so the same
    // int-theory-first, float-window-fallback ladder the binary `-`
    // rides (`transfer_over_sets`) answers the negation — including
    // the general WINDOW arm the one-operand `int.neg` row lacks.
    let zero = known_values(vec![0.0], PrimitiveKind::Integer, TrustProved);
    transfer_over_sets(Operator::Sub, &zero, operand, kernel)
}
