use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Operator;

use crate::collection_models;
use crate::expressions::compare::exact_string_values;
use crate::expressions::datetime::date_shifted_by_timedelta;
use crate::expressions::datetime::datetime_difference_value;
use crate::expressions::datetime::datetime_field;
use crate::expressions::datetime::datetime_shifted_by_timedelta;
use crate::expressions::datetime::datetime_window_difference_value;
use crate::expressions::datetime::timedelta_floordiv_value;
use crate::expressions::sequence_ops::dict_union_value;
use crate::expressions::sequence_ops::list_repetition_sort_only;
use crate::expressions::sequence_ops::sequence_repetition;
use crate::expressions::sequence_ops::sequence_window_concatenation;
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
            // The two-WINDOW row is tried before `string_set_
            // concatenation`, and declines every STRING-shaped pair
            // itself (`sequence_window_concatenation`'s own doc), so the
            // string row below still sees exactly the pairs it did
            // before. What reaches the window row instead is the
            // NON-string sequence pair — two `list[int]` parameters'
            // own seeds — which the string row would otherwise have
            // read as a codepoint grammar.
            if let Some(result) = sequence_window_concatenation(left, right) {
                return result;
            }
            if let Some(result) = string_set_concatenation(left, right) {
                return result;
            }
            unknown()
        }
        // `|` is BOTH the set-union and the dict-merge spelling
        // (stdtypes.rst states `d | other` in its own Mapping Types
        // section, separate from the set operations table) — the two
        // never collide, since a dict is `Kind::Object` and a set is
        // `Kind::List` in this domain, so the dict row is tried first
        // and declines every non-dict pair unchanged.
        Operator::BitOr => match dict_union_value(left, right) {
            Some(merged) => merged,
            None => set_operator_value("union", left, right),
        },
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
            if let Some(result) = list_repetition_sort_only(left, right).or_else(|| list_repetition_sort_only(right, left)) {
                return result;
            }
            unknown()
        }
        _ => unknown(),
    }
}

/// The temporal operand rows of datetime.rst's own operation tables —
/// tried BEFORE the ordinary numeric/sequence dispatch, since neither
/// operand is a single numeric value or a string/list
/// (`binary_arithmetic_value`'s own fallthrough would otherwise reach
/// `sequence_binop_value` and answer `unknown()` for a tagged-Object
/// pair). Four rows:
///
/// - `date + timedelta` and `timedelta + date` both shift forward
///   (`Operator::Add`, either operand order — date.7 states the
///   operation both ways).
/// - `date - timedelta` shifts backward (`Operator::Sub`, `date` on the
///   LEFT only — `timedelta - date` is not a datetime.rst operation).
/// - `datetime ± timedelta` shifts an instant the same two ways
///   (`.datetime`'s own table, notes (1) and (2) —
///   `datetime_shifted_by_timedelta`).
/// - `datetime1 - datetime2` answers the exact `timedelta` between two
///   instants (`.datetime`'s own table, note (3) —
///   `datetime_difference_value` carries the naive/aware premise), or,
///   when either side is a WINDOW rather than one instant, the range of
///   differences that window admits
///   (`datetime_window_difference_value`).
/// - `timedelta2 // timedelta3` answers the floor of the quotient as an
///   integer (`timedelta`'s own table —`timedelta_floordiv_value`).
///
/// `None` for every operand pair outside those rows — the caller falls
/// through to the ordinary dispatch unchanged.
pub(in crate::expressions) fn date_timedelta_binop_value(op: Operator, left: &AbstractValue, right: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let is_date = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_date";
    let is_timedelta = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_timedelta";
    let is_datetime = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_datetime";
    // a `temporal_flow`-tagged parameter on the Duration chart — the
    // window a bare `td: timedelta` parameter seeds
    // (`check::seed_parameters`' own temporal seed, `surface::
    // bare_temporal_annotation`'s Duration-chart row). Carries no
    // normalized days/seconds/microseconds triple, only an ISO window
    // (unbounded for a bare parameter, since `seed_parameters` states no
    // bound without an `Annotated[timedelta, Field(...)]` alias) — for
    // `//`, this reduces to the SAME no-keys `datetime_timedelta`
    // instance `datetime_window_difference_value`'s own unbounded branch
    // builds, which `timedelta_floordiv_value` already reads as the
    // unbounded integer sort (datetime.rst's own `t2 // t3` row: "an
    // integer is returned"), narrowed the ordinary way by a later guard.
    let is_temporal_flow_duration = |value: &AbstractValue| {
        value.kind == Kind::Object
            && value.source == "temporal_flow"
            && value
                .temporal
                .as_ref()
                .is_some_and(|window| window.chart == refined_sets::calendar_interpreter::TemporalChart::Duration)
    };
    let as_timedelta_operand = |value: &AbstractValue| -> Option<AbstractValue> {
        if is_timedelta(value) {
            return Some(value.clone());
        }
        if is_temporal_flow_duration(value) {
            let mut instance = known_object(Vec::new(), None, true, TrustSpec, false);
            instance.source = "datetime_timedelta".to_owned();
            return Some(instance);
        }
        None
    };
    // a `temporal_flow`-tagged parameter on the Instant chart — a WINDOW
    // of instants rather than one, `check::seed_parameters`' own seed for
    // a bare `d: datetime`
    let is_instant_window = |value: &AbstractValue| {
        value.kind == Kind::Object
            && value.source == "temporal_flow"
            && value
                .temporal
                .as_ref()
                .is_some_and(|window| window.chart == refined_sets::calendar_interpreter::TemporalChart::Instant)
    };
    match op {
        Operator::Add => {
            if is_date(left) && is_timedelta(right) {
                return date_shifted_by_timedelta(left, right, false, kernel);
            }
            if is_timedelta(left) && is_date(right) {
                return date_shifted_by_timedelta(right, left, false, kernel);
            }
            if is_datetime(left) && is_timedelta(right) {
                return datetime_shifted_by_timedelta(left, right, false, kernel);
            }
            if is_timedelta(left) && is_datetime(right) {
                return datetime_shifted_by_timedelta(right, left, false, kernel);
            }
            None
        }
        Operator::Sub => {
            if is_date(left) && is_timedelta(right) {
                return date_shifted_by_timedelta(left, right, true, kernel);
            }
            if is_datetime(left) && is_timedelta(right) {
                return datetime_shifted_by_timedelta(left, right, true, kernel);
            }
            if is_datetime(left) && is_datetime(right) {
                return datetime_difference_value(left, right, kernel);
            }
            // at least one side a WINDOW rather than one instant — the
            // shape a bare `d: datetime` parameter takes
            if (is_datetime(left) || is_instant_window(left)) && (is_datetime(right) || is_instant_window(right)) {
                return datetime_window_difference_value(left, right, kernel);
            }
            None
        }
        Operator::FloorDiv => {
            if let (Some(left), Some(right)) = (as_timedelta_operand(left), as_timedelta_operand(right)) {
                return timedelta_floordiv_value(&left, &right);
            }
            None
        }
        _ => None,
    }
}

/// `datetime1 - datetime2` where ONE operand is naive and the other is
/// aware — datetime.rst's `.datetime` operation table, note (3):
/// "Subtraction of a `.datetime` from a `.datetime` is defined only if
/// both operands are naive, or if both are aware. If one is aware and
/// the other is naive, `TypeError` is raised." Every run of such a
/// subtraction raises, so this is a `binop_provable_raise` row rather
/// than a `binop_possible_raise` one — the all-or-nothing discipline
/// that function's own doc states.
///
/// The awareness premise is read from the SAME `aware` tag
/// `datetime_difference_value` decides on (0 = naive, 1 = aware with an
/// exactly known offset, 2 = aware with an unresolved offset), so the
/// value side's decline and this raise row never disagree about the same
/// operand pair. An `aware = 2` operand on either side is NOT a row
/// here: note (3) makes an unresolved offset a value this file cannot
/// subtract, not a proven raise — tag 2 is aware, so a `2`-against-`1`
/// pair is an aware/aware subtraction CPython performs successfully, and
/// only its VALUE is out of reach. `None` for any operand pair that is
/// not two `datetime_datetime` instances, or whose two `aware` tags this
/// reader cannot both read — an instant WINDOW (`temporal_flow`) carries
/// no `aware` tag of its own, so it never reaches this row.
pub(in crate::expressions) fn datetime_difference_provable_raise(
    op: Operator,
    left: &AbstractValue,
    right: &AbstractValue,
) -> Option<String> {
    if op != Operator::Sub {
        return None;
    }
    let is_datetime = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_datetime";
    if !is_datetime(left) || !is_datetime(right) {
        return None;
    }
    let left_aware = datetime_field(left, "aware")?;
    let right_aware = datetime_field(right, "aware")?;
    let naive = |tag: f64| tag == 0.0;
    if naive(left_aware) == naive(right_aware) {
        return None;
    }
    let (naive_side, aware_side) = if naive(left_aware) { ("left", "right") } else { ("right", "left") };
    Some(format!(
        "this expression provably raises TypeError: can't subtract offset-naive and offset-aware datetimes — \
        the {naive_side} operand is naive and the {aware_side} operand is aware (datetime.rst, the datetime \
        operation table's note (3))"
    ))
}
