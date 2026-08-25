//! Arithmetic on temporal values: an aware `datetime`'s exact POSIX
//! timestamp, `date ± timedelta` shifts, and the kernel-aware binary
//! arithmetic dispatcher exported for `expressions.rs`'s own `BinOp`
//! evaluation.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::CalendarQuestion;
use refined_kernel::kernel_interface::CalendarQuestionOp;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Operator;

use super::components::datetime_field;
use super::components::epoch_days_and_day_of_week;
use super::components::epoch_days_of_civil_date;
use super::construction::integer_object_key;
use super::construction::python_year_in_range;

use super::super::arithmetic::*;

/// `<an aware-UTC datetime_datetime instance>.timestamp()` — the EXACT
/// POSIX timestamp: datetime.rst, `method:: datetime.timestamp()`, "For
/// aware datetime instances, the return value is computed as: `(dt -
/// datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()`." UTC has
/// no DST/leap-second adjustment, so that difference reduces to plain
/// calendar-day arithmetic (`epoch_days_of_civil_date`'s kernel ask)
/// times 86400, plus the wall-clock seconds-of-day. Modeled ONLY for a
/// `datetime_construction_value`-tagged instance whose own `aware_utc`
/// field is `true` — `None` for a NAIVE instance (datetime.rst's own
/// note: "Naive datetime instances are assumed to represent local time
/// and this method relies on the platform C mktime function," a
/// host-dependent conversion this file does not claim to reproduce).
pub(in crate::expressions) fn datetime_timestamp_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let aware = datetime_field(instance, "aware_utc")?;
    if aware != 1.0 {
        return None;
    }
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let hour = datetime_field(instance, "hour")? as i64;
    let minute = datetime_field(instance, "minute")? as i64;
    let second = datetime_field(instance, "second")? as i64;
    let days = epoch_days_of_civil_date(year, month, day, kernel)?;
    let seconds = days * 86400 + hour * 3600 + minute * 60 + second;
    Some(known_values(vec![seconds as f64], PrimitiveKind::Float, TrustProved))
}

/// `date1 ± timedelta` — datetime.rst's operation table (date.7's own
/// row): shifts by `timedelta.days` (the only field
/// `timedelta_construction_value` ever populates) and answers a NEW
/// tagged `datetime_date` instance, or declines (`None`) exactly where
/// CPython raises `OverflowError`. The kernel's `epochDays`/`isoDate`
/// pair (date.1's seam) computes the shifted day count and certifies it
/// lands back on a calendrically valid date (`isoDate`'s own
/// "self-certification" — `exports_calendar.lean`'s comment), but that
/// certification alone is NOT date.7's `OverflowError` bound: the
/// kernel's own PlainDate window is far wider than Python's
/// `[MINYEAR, MAXYEAR]`, so this function additionally poses the
/// `pyYearInRange` ask (`python_year_in_range`) on the shifted result —
/// a shift the kernel's `isoDate` arm would happily answer but Python
/// would reject (`date(9999, 12, 31) + timedelta(days=1)`, landing on
/// year 10000) still declines here. `negate` flips the shift for
/// `date - timedelta` (`date + timedelta` passes `false`).
pub(in crate::expressions) fn date_shifted_by_timedelta(date: &AbstractValue, timedelta: &AbstractValue, negate: bool, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let (days, _) = epoch_days_and_day_of_week(date, kernel)?;
    let shift = datetime_field(timedelta, "days")? as i64;
    let shifted_days = if negate { days - shift } else { days + shift };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoDate,
            year: 0,
            month: 0,
            day: 0,
            days: shifted_days,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let year = asked.get("year")?.as_i64()?;
    let month = asked.get("month")?.as_i64()?;
    let day = asked.get("day")?.as_i64()?;
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    let keys = vec![integer_object_key("year", year), integer_object_key("month", month), integer_object_key("day", day)];
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_date".to_owned();
    Some(instance)
}

/// `binary_arithmetic_value` WITH the kernel available: tries the SET
/// path first (`transfer_over_sets` — at least one operand a numeric
/// set, the admitted operators only), then falls through to
/// `binary_arithmetic_value` unchanged for everything else (two known
/// single values, or a non-numeric pair headed for
/// `sequence_binop_value`). Exported for `expressions.rs`'s OWN
/// BinOp evaluation (`evaluate_binop`) — the other call sites
/// (`loops.rs`, `summaries.rs`, `check.rs`'s AugAssign paths) still call
/// the plain `binary_arithmetic_value` today; wiring them onto this
/// function is a follow-on, not a behavior change this function's own
/// landing makes for them.
pub fn binary_arithmetic_value_with_kernel(
    op: Operator,
    left: &AbstractValue,
    right: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if let Some(result) = transfer_over_sets(op, left, right, kernel) {
        return result;
    }
    binary_arithmetic_value(op, left, right)
}
