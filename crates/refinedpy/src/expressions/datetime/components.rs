//! Component reads off a tagged `datetime_date`/`datetime_datetime`
//! instance: single fields, and the kernel-derived weekday/ordinal/
//! isocalendar/epoch-day answers.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::CalendarQuestion;
use refined_kernel::kernel_interface::CalendarQuestionOp;
use refined_kernel::kernel_interface::RefinedTSKernel;

use crate::collection_models;

use super::super::arithmetic::*;

/// The proleptic Gregorian day count from the civil (year, month, day)
/// triple to the POSIX epoch (1970-01-01 = day 0), asked of the kernel's
/// `calendar` seam (`refined_calendar`'s `"epochDays"` op,
/// `theories/calendar/epoch_days.lean`'s `isoDateToEpochDays`, the SAME
/// anchor `datetime_timestamp_value`'s own doc already cited: day 0 is
/// `date(1970, 1, 1).toordinal()`). The kernel validates the date
/// itself (`isValidISODate`) and the PlainDate day-range limit
/// (`epochDaysWithinLimits`) before answering, so an out-of-range or
/// invalid civil date is a caught refusal here (`ask_kernel`'s
/// `catch_unwind`), not a value this function returns — `None` in that
/// case, matching every other refused kernel ask in this crate.
pub(in crate::expressions) fn epoch_days_of_civil_date(year: i64, month: i64, day: i64, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::EpochDays,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("days")?.as_i64()
}

/// One numeric `ObjectKey` field's own value off a tagged instance — the
/// linear scan `datetime_timestamp_value` reads each calendar field
/// through (the same by-name `ObjectKey` shape `instances::field_read`
/// reads for an untagged instance, repeated here as a private single-
/// field helper since every caller already knows the exact field name
/// it wants).
pub(in crate::expressions) fn datetime_field(instance: &AbstractValue, name: &str) -> Option<f64> {
    let entry = instance.keys.iter().find(|key| key.name == name)?;
    let (value, _) = single_numeric_value(&entry.value)?;
    Some(value)
}

/// The kernel's `epochDays` answer for a tagged `datetime_date`
/// instance's own `year`/`month`/`day` fields — `.weekday()` and
/// `.toordinal()`'s shared first step, both riding the SAME kernel ask
/// `epoch_days_of_civil_date` already makes for `datetime_datetime`
/// (this function reads its own `dayOfWeek` field too, which that
/// function's caller never needed). `None` on a refused ask (an
/// out-of-range or invalid date, though a tagged `datetime_date`
/// instance was already validated at construction).
pub(in crate::expressions) fn epoch_days_and_day_of_week(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<(i64, i64)> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::EpochDays,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let days = asked.get("days")?.as_i64()?;
    let day_of_week = asked.get("dayOfWeek")?.as_i64()?;
    Some((days, day_of_week))
}

/// `date.weekday()` — datetime.rst:687, "Monday is 0 and Sunday is 6."
/// Asks the kernel's `"weekday"` op directly (`exports_calendar.lean`'s
/// `"weekday"` arm, `Refinements.pyWeekday`, `languages/python/
/// dates_durations/weekday.lean`) over the instance's own `year`/
/// `month`/`day` fields — the kernel answers Python's Monday-0 form
/// itself, so this function poses one ask and reads its `weekday` field
/// unchanged, no local arithmetic.
pub(in crate::expressions) fn date_weekday_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::Weekday,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let weekday = asked.get("weekday")?.as_i64()?;
    Some(known_values(vec![weekday as f64], PrimitiveKind::Integer, TrustProved))
}

/// `date.isoweekday()` — datetime.rst:694-695, "Monday is 1 and Sunday
/// is 7," ONE more than `.weekday()`'s Monday-0 form (both elections
/// walk the same seven days in the same order — the kernel's `"weekday"`
/// arm already IS the Monday-0 answer this method shifts by one).
/// Reuses `date_weekday_value`'s own ask rather than posing a second
/// one: the ISO-1 form has no dedicated kernel arm of its own, and
/// deriving it from the already-asked Monday-0 answer needs no further
/// kernel round trip.
pub(in crate::expressions) fn date_isoweekday_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let weekday = date_weekday_value(instance, kernel)?;
    let (monday_zero, _) = single_numeric_value(&weekday)?;
    Some(known_values(vec![monday_zero + 1.0], PrimitiveKind::Integer, TrustProved))
}

/// `date.toordinal()` — datetime.rst:525-526, "January 1 of year 1 has
/// ordinal 1." Asks the kernel's `"toordinal"` op directly
/// (`exports_calendar.lean`'s `"toordinal"` arm, `Refinements.
/// pyToOrdinal`, `languages/python/dates_durations/ordinal.lean`) over
/// the instance's own `year`/`month`/`day` fields — the kernel applies
/// the proved `719163` anchor shift itself, so this function poses one
/// ask and reads its `ordinal` field unchanged, no local arithmetic.
pub(in crate::expressions) fn date_toordinal_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::ToOrdinal,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let ordinal = asked.get("ordinal")?.as_i64()?;
    Some(known_values(vec![ordinal as f64], PrimitiveKind::Integer, TrustProved))
}

/// `date.isocalendar()` — datetime.rst:699-721, the (ISO year, ISO
/// week, ISO weekday) triple. Asks the kernel's `"isoCalendar"` op
/// directly (`exports_calendar.lean`'s `"isoCalendar"` arm,
/// `Refinements.pyIsoCalendar`, `languages/python/dates_durations/
/// iso_week_date.lean`) over the instance's own `year`/`month`/`day`
/// fields, then binds the three answered ints (`isoYear`, `week`,
/// `weekday`) as a known 3-element tuple through
/// `collection_models::tuple_literal_value` — the same constructor
/// `evaluate_tuple` uses for a literal `(a, b, c)` display, so the
/// answer type-checks identically to a real tuple.
pub(in crate::expressions) fn date_isocalendar_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoCalendar,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let iso_year = asked.get("isoYear")?.as_i64()?;
    let week = asked.get("week")?.as_i64()?;
    let weekday = asked.get("weekday")?.as_i64()?;
    let elements = [iso_year, week, weekday].map(|value| known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved));
    Some(collection_models::tuple_literal_value(&elements))
}
