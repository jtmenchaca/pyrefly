//! Rendering a tagged instance back to text: `date.strftime` and the
//! `date`/`datetime` ISO renderings.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::CalendarQuestion;
use refined_kernel::kernel_interface::CalendarQuestionOp;
use refined_kernel::kernel_interface::RefinedTSKernel;

use crate::string_models::string_literal_value;

use super::components::datetime_field;
use super::construction::offset_iso_suffix;

/// The kernel's `YYYY-MM-DD` render of a civil date —
/// `exports_calendar.lean`'s `"isoDateText"` arm,
/// `Refinements.Calendar.isoDateText` (theories/calendar/iso_render.lean),
/// which composes the digit and zero-fill helpers the decimal renderers
/// already share. The four-digit year bound is the kernel arm's own: a
/// year outside 0…9999 declines there rather than truncating, so this
/// wrapper answers `None` for it exactly as it does for a refused ask.
/// The render direction is asked of the kernel rather than spelled with
/// a local `format!` so the text a date prints is the kernel's claim.
pub(in crate::expressions) fn iso_date_text(year: i64, month: i64, day: i64, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoDateText,
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
    Some(asked.get("text")?.as_str()?.to_owned())
}

/// date.12 — `date.strftime(format)` (datetime.rst, `method::
/// date.strftime(format)`: "Return a string representing the date,
/// controlled by an explicit format string"). Modeled ONLY for the
/// exact literal format `"%Y-%m-%d"` on a tagged `datetime_date`
/// instance whose OWN `year`/`month`/`day` fields are already known —
/// the same text `date.isoformat()` produces (datetime.rst:725, "ISO
/// 8601 format, YYYY-MM-DD"), asked of the kernel's own `isoDateText`
/// arm through `iso_date_text`.
pub(in crate::expressions) fn strftime_iso_date_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    date_isoformat_value(instance, kernel)
}

/// `<a tagged datetime_date instance>.isoformat()` — datetime.rst,
/// `method:: date.isoformat()`, "Return a string representing the date
/// in ISO 8601 format, YYYY-MM-DD." The text is the kernel's
/// (`iso_date_text`); `None` where a field is not exactly known or the
/// kernel arm declines the year.
pub(in crate::expressions) fn date_isoformat_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let text = iso_date_text(year, month, day, kernel)?;
    Some(string_literal_value(&text))
}

/// `<a tagged datetime_datetime instance>.isoformat()` — datetime.rst,
/// `method:: datetime.isoformat(sep='T')`: "Return a string representing
/// the date and time in ISO 8601 format, `YYYY-MM-DDTHH:MM:SS.ffffff`,
/// or, if microsecond is 0, `YYYY-MM-DDTHH:MM:SS`... If `utcoffset()`
/// does not return `None`, a string is appended, giving the UTC offset:
/// `YYYY-MM-DDTHH:MM:SS.ffffff+HH:MM[:SS[.ffffff]]`."
///
/// The date half is the kernel's `isoDateText` render; the clock half is
/// the instance's own `hour`/`minute`/`second`/`microsecond` fields, each
/// already an exactly-known Integer on a tagged instance, spelled at the
/// fixed two-digit (six for microsecond) widths the clause quotes above.
/// The offset suffix comes from the `aware` tag
/// (`datetime_construction_value`'s own doc): `0` is naive and appends
/// nothing, `1` is aware with an EXACTLY known offset, `2` is aware with
/// an offset this crate cannot resolve — that last one declines, since
/// no text can be spelled without the offset.
///
/// CPython renders a UTC-aware datetime's offset as `+00:00`, never the
/// `Z` military designator, so the offset text is read off the
/// instance's own ISO spelling (`instance.temporal`, whose UTC arm
/// spells `Z` for the CHART's benefit) rather than copied from it.
pub(in crate::expressions) fn datetime_isoformat_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let hour = datetime_field(instance, "hour")? as i64;
    let minute = datetime_field(instance, "minute")? as i64;
    let second = datetime_field(instance, "second")? as i64;
    let microsecond = datetime_field(instance, "microsecond")? as i64;
    let aware = datetime_field(instance, "aware")? as i64;
    let offset_seconds = match aware {
        0 => None,
        1 => Some(datetime_offset_seconds(instance)?),
        // `TzinfoKind::OtherAware` — aware, but with no offset this
        // crate resolved, so no text can be spelled.
        _ => return None,
    };
    let date_text = iso_date_text(year, month, day, kernel)?;
    let mut text = format!("{date_text}T{hour:02}:{minute:02}:{second:02}");
    if microsecond != 0 {
        text.push_str(&format!(".{microsecond:06}"));
    }
    if let Some(seconds) = offset_seconds {
        text.push_str(&offset_iso_suffix(seconds));
    }
    Some(string_literal_value(&text))
}

/// The exactly-known UTC offset, in whole seconds, of a tagged
/// `datetime_datetime` instance whose `aware` tag is 1 — the `aware_utc`
/// marker means offset zero, and any other exactly-offset construction
/// spelled its own suffix onto `instance.temporal`, which
/// `offset_iso_suffix` built and this reader reads back. `None` where the
/// instance carries no ISO spelling at all.
fn datetime_offset_seconds(instance: &AbstractValue) -> Option<i64> {
    if datetime_field(instance, "aware_utc")? == 1.0 {
        return Some(0);
    }
    let temporal = instance.temporal.as_ref()?;
    let spelling = temporal.min.as_ref()?;
    // `offset_iso_suffix`'s own output: a sign, two hour digits, a
    // colon, two minute digits, always the last six characters of a
    // fixed-offset spelling.
    let suffix = spelling.get(spelling.len().checked_sub(6)?..)?;
    let sign = match suffix.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let hours: i64 = suffix.get(1..3)?.parse().ok()?;
    let minutes: i64 = suffix.get(4..6)?.parse().ok()?;
    Some(sign * (hours * 3600 + minutes * 60))
}
