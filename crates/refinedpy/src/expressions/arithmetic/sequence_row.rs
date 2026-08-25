use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Operator;

use crate::collection_models;
use crate::expressions::compare::exact_string_values;
use crate::expressions::datetime::date_shifted_by_timedelta;
use crate::expressions::sequence_ops::sequence_repetition;
use crate::expressions::sequence_ops::set_operator_value;
use crate::expressions::sequence_ops::string_repetition_sort_only;
use crate::expressions::sequence_ops::string_set_concatenation;

/// String/list `+`/`*`, and the SET operator spelling of `|`/`&`/`-`/
/// `^` — stdtypes.rst, "Common Sequence Operations": `s + t` is "the
/// concatenation of s and t," and `s * n` (either operand order) is "n
/// shallow copies of s concatenated," with note 2 pinning "values of n
/// less than 0 are treated as 0." The set section states the operator
/// spellings directly beside their method names (`union(*others)`:
/// "set | other | ..."; `intersection`: "set & other & ..."; the
/// `difference`/`symmetric_difference` operator rows the same section
/// states as `-`/`^`) — `set_method_result` already carries every one
/// of those row's semantics, so `|`/`&`/`-`/`^` over two known
/// `Kind::List` operands (both operands, per this domain's shared
/// list/set representation — see `evaluate_set`'s own doc) call
/// through to it under the equivalent method name rather than
/// duplicate the four loops. Called from `binary_arithmetic_value`'s
/// OWN fallthrough the moment either operand is not a single known
/// numeric value — a numeric `+`/`*`/bitwise op never reaches here, since
/// `binary_arithmetic_value` answers those itself first.
pub(in crate::expressions) fn sequence_binop_value(op: Operator, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    match op {
        Operator::Add => {
            if let (Some(left_text), Some(right_text)) = (exact_string_values(left), exact_string_values(right)) {
                let mut joined = left_text.to_vec();
                joined.extend_from_slice(right_text);
                return known_values(joined, PrimitiveKind::String, TrustProved);
            }
            if left.kind == Kind::List && right.kind == Kind::List {
                let mut joined = left.items.clone();
                joined.extend(right.items.iter().cloned());
                return collection_models::list_literal_value(&joined);
            }
            if let Some(result) = string_set_concatenation(left, right) {
                return result;
            }
            unknown()
        }
        Operator::BitOr => set_operator_value("union", left, right),
        Operator::BitAnd => set_operator_value("intersection", left, right),
        Operator::Sub => set_operator_value("difference", left, right),
        Operator::BitXor => set_operator_value("symmetric_difference", left, right),
        Operator::Mult => {
            if let Some(result) = sequence_repetition(left, right) {
                return result;
            }
            if let Some(result) = sequence_repetition(right, left) {
                return result;
            }
            if let Some(result) = string_repetition_sort_only(left, right).or_else(|| string_repetition_sort_only(right, left))
            {
                return result;
            }
            unknown()
        }
        _ => unknown(),
    }
}

/// `date ± timedelta` (date.7's own operation-table row) — tried BEFORE
/// the ordinary numeric/sequence dispatch, since neither operand is a
/// single numeric value or a string/list (`binary_arithmetic_value`'s
/// own fallthrough would otherwise reach `sequence_binop_value` and
/// answer `unknown()` for a tagged-Object pair). `date + timedelta` and
/// `timedelta + date` both shift forward (`Operator::Add`, either
/// operand order — datetime.rst states the operation both ways);
/// `date - timedelta` shifts backward (`Operator::Sub`, `date` on the
/// LEFT only — `timedelta - date` is not a datetime.rst operation).
/// `date - date` (the OTHER `date.7` row, an exact `timedelta` result)
/// is NOT built here: no row in this file's construct list asks for it,
/// and `timedelta_construction_value`'s own single `days` field gives no
/// two-instance subtraction a shape to land in without inventing one.
/// `None` for every operand pair that is not exactly one tagged
/// `datetime_date` and one tagged `datetime_timedelta` — the caller
/// falls through to the ordinary dispatch unchanged.
pub(in crate::expressions) fn date_timedelta_binop_value(op: Operator, left: &AbstractValue, right: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let is_date = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_date";
    let is_timedelta = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_timedelta";
    match op {
        Operator::Add => {
            if is_date(left) && is_timedelta(right) {
                return date_shifted_by_timedelta(left, right, false, kernel);
            }
            if is_timedelta(left) && is_date(right) {
                return date_shifted_by_timedelta(right, left, false, kernel);
            }
            None
        }
        Operator::Sub => {
            if is_date(left) && is_timedelta(right) {
                return date_shifted_by_timedelta(left, right, true, kernel);
            }
            None
        }
        _ => None,
    }
}
