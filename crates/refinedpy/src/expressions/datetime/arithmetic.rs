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
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;

use crate::env::Environment;

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

/// `datetime1 - datetime2` — datetime.rst's `.datetime` operation table
/// (row `timedelta = datetime1 - datetime2`, note (3)): "Subtraction of
/// a `.datetime` from a `.datetime` is defined only if both operands are
/// naive, or if both are aware. If one is aware and the other is naive,
/// `TypeError` is raised." Answers the exact `timedelta` that difference
/// names, built through `timedelta_instance_of_microseconds` so the
/// result is the SAME instance shape a literal `timedelta(...)`
/// construction carries.
///
/// The awareness premise is decided from each instance's own `aware` tag
/// (`datetime_construction_value`'s own marker: 0 = naive, 1 = aware
/// with an exactly known offset, 2 = aware with an unresolved offset).
/// A mixed naive/aware pair declines — no value flows where CPython
/// raises. An `aware = 2` operand declines too: note (3)'s second
/// paragraph makes the answer depend on each side's `utcoffset()`, and
/// an unresolved offset gives no number to subtract.
///
/// Both operands' instants reduce to a whole-second epoch count the same
/// way `datetime_timestamp_value` does — the kernel's `epochDays` ask
/// (`epoch_days_of_civil_date`, date.1's own seam) times 86400 plus the
/// wall clock — with the tzinfo offset applied, since note (3) states
/// the difference is taken over UTC-normalized instants when the two
/// carry different offsets. The microsecond fields ride the difference
/// unchanged. `None` on any refused kernel ask.
pub(in crate::expressions) fn datetime_difference_value(left: &AbstractValue, right: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let left_aware = datetime_field(left, "aware")?;
    let right_aware = datetime_field(right, "aware")?;
    // naive-vs-aware raises TypeError; an unresolved offset (tag 2) has
    // no number this function can subtract
    if left_aware != right_aware || left_aware == 2.0 {
        return None;
    }
    let left_microseconds = datetime_epoch_microseconds(left, kernel)?;
    let right_microseconds = datetime_epoch_microseconds(right, kernel)?;
    super::construction::timedelta_instance_of_microseconds(left_microseconds - right_microseconds, kernel)
}

/// `datetime ± timedelta` — datetime.rst's `.datetime` operation table,
/// notes (1) and (2): "`datetime2` is a duration of `timedelta` removed
/// from `datetime1`, moving forward in time if `timedelta.days > 0`, or
/// backward if `timedelta.days < 0`. The result has the same `tzinfo`
/// attribute as the input datetime... Note that no time zone adjustments
/// are done even if the input is an aware object."
///
/// Modeled for an EXACT instant and an EXACT duration: the instant
/// reduces to its microsecond count (`datetime_epoch_microseconds`, the
/// kernel's `epochDays` seam), the duration to its own
/// (`timedelta_total_microseconds`), and the shifted count splits back
/// into calendar and clock fields through the kernel's `isoDate` op —
/// the same self-certifying arm `datetime_fromtimestamp_value` uses. The
/// result carries the SAME tzinfo markers as the input, per the cited
/// note, so `.timestamp()` and the temporal admission law read it
/// exactly as they read a literal construction.
///
/// `None` for a shift landing outside Python's own `[MINYEAR, MAXYEAR]`
/// (`python_year_in_range` — note (1)'s `OverflowError`), for an
/// unresolved-offset instant (`aware == 2`, whose own ISO spelling this
/// crate never settled), or on any refused kernel ask. `negate` flips
/// the shift for `datetime - timedelta`.
pub(in crate::expressions) fn datetime_shifted_by_timedelta(instant: &AbstractValue, duration: &AbstractValue, negate: bool, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let aware = datetime_field(instant, "aware")?;
    if aware == 2.0 {
        return None;
    }
    let base = datetime_epoch_microseconds(instant, kernel)?;
    let shift = super::construction::timedelta_total_microseconds(duration)?;
    let shifted = if negate { base - shift } else { base + shift };
    let days = i64::try_from(shifted.div_euclid(86_400_000_000)).ok()?;
    let within_day = shifted.rem_euclid(86_400_000_000);
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoDate,
            year: 0,
            month: 0,
            day: 0,
            days,
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
    let microsecond = (within_day % 1_000_000) as i64;
    let seconds_of_day = (within_day / 1_000_000) as i64;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    // the input's own tzinfo markers ride through unchanged — note (1)'s
    // "the result has the same tzinfo attribute as the input datetime"
    let aware_utc = datetime_field(instant, "aware_utc")?;
    let mut keys = vec![
        integer_object_key("year", year),
        integer_object_key("month", month),
        integer_object_key("day", day),
        integer_object_key("hour", hour),
        integer_object_key("minute", minute),
        integer_object_key("second", second),
        integer_object_key("microsecond", microsecond),
    ];
    keys.push(refined_domain::abstract_value::ObjectKey {
        name: "aware_utc".to_owned(),
        numeric: false,
        value: known_values(vec![aware_utc], PrimitiveKind::Boolean, TrustProved),
    });
    keys.push(integer_object_key("aware", aware as i64));
    let mut result = known_object(keys, None, true, TrustProved, false);
    result.source = "datetime_datetime".to_owned();
    // The ISO spelling this crate writes for a UTC instant; a naive one
    // keeps the offset-free form, matching
    // `datetime_construction_value`'s own two spellings.
    let zone = if aware_utc == 1.0 { "Z" } else { "" };
    let point = if microsecond == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{zone}")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{microsecond:06}{zone}")
    };
    result.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
        chart: refined_sets::calendar_interpreter::TemporalChart::Instant,
        min: Some(point.clone()),
        max: Some(point),
    }));
    Some(result)
}

/// `datetime1 - datetime2` where at least one side is a WINDOW rather
/// than one instant — a `temporal_flow`-tagged parameter narrowed to a
/// calendar window, the shape `check::seed_parameters` binds a bare
/// `d: datetime` to. datetime.rst's note (3) applies pointwise across
/// the window, so the difference is the window of differences: its
/// smallest value is the left window's earliest instant minus the right
/// window's latest, and its largest the reverse.
///
/// Answers a tagged `datetime_timedelta` instance carrying the
/// microsecond range as its two `low_microseconds`/`high_microseconds`
/// Integer fields — a window has no single normalized triple to store,
/// so the range IS what the instance carries, and
/// `timedelta_floordiv_value` reads either shape. `None` when either
/// side's window is not bounded on both ends, or carries a spelling this
/// reader does not split — an unbounded difference states no range.
pub(in crate::expressions) fn datetime_window_difference_value(left: &AbstractValue, right: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let mut instance = known_object(Vec::new(), None, true, TrustProved, false);
    instance.source = "datetime_timedelta".to_owned();
    // An UNBOUNDED window on either side leaves the difference
    // unbounded too. The instance still carries the `datetime_timedelta`
    // tag with no range fields — `timedelta_floordiv_value` reads that
    // as the unbounded integer sort, which is the claim datetime.rst's
    // own `t2 // t3` row states for it ("an integer is returned"), and a
    // later comparison guard narrows that sort the ordinary way. This is
    // strictly weaker than a stated range and strictly stronger than no
    // value at all.
    let (Some((left_low, left_high)), Some((right_low, right_high))) =
        (instant_window_microseconds(left, kernel), instant_window_microseconds(right, kernel))
    else {
        return Some(instance);
    };
    let (Ok(low), Ok(high)) = (i64::try_from(left_low - right_high), i64::try_from(left_high - right_low)) else {
        return Some(instance);
    };
    instance.keys = vec![
        super::construction::integer_object_key("low_microseconds", low),
        super::construction::integer_object_key("high_microseconds", high),
    ];
    Some(instance)
}

/// One temporal value's own instant window as a microsecond pair — the
/// EXACT instant twice for a concrete `datetime_datetime` construction,
/// or the two ISO endpoints of a `temporal_flow` window. `None` for a
/// window missing either endpoint, a non-`Instant` chart, or an endpoint
/// spelling this reader does not split.
fn instant_window_microseconds(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<(i128, i128)> {
    if value.source == "datetime_datetime" {
        let exact = datetime_epoch_microseconds(value, kernel)?;
        return Some((exact, exact));
    }
    if value.source != "temporal_flow" {
        return None;
    }
    let window = value.temporal.as_ref()?;
    if window.chart != refined_sets::calendar_interpreter::TemporalChart::Instant {
        return None;
    }
    let low = utc_iso_microseconds(window.min.as_deref()?, kernel)?;
    let high = utc_iso_microseconds(window.max.as_deref()?, kernel)?;
    Some((low, high))
}

/// A `YYYY-MM-DDTHH:MM:SS[.ffffff]Z` UTC instant spelling as a
/// microsecond count from the POSIX epoch. This is the exact fixed-width
/// form this crate itself writes for every UTC instant it spells
/// (`datetime_construction_value`'s own `instance.temporal`, the
/// temporal surface's own bound spelling, and the temporal narrowing
/// channel's own year endpoints), so the reader splits that one form and
/// declines anything else — an offset-bearing or naive spelling has no
/// UTC instant this function could name without an offset to apply. The
/// calendar half rides the kernel's `epochDays` ask, the same seam every
/// other date reduction in this file uses.
pub fn utc_iso_microseconds(text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<i128> {
    let body = text.strip_suffix('Z')?;
    let (date_text, clock_text) = body.split_once('T')?;
    let mut date_parts = date_text.split('-');
    let year: i64 = date_parts.next()?.parse().ok()?;
    let month: i64 = date_parts.next()?.parse().ok()?;
    let day: i64 = date_parts.next()?.parse().ok()?;
    if date_parts.next().is_some() {
        return None;
    }
    let (clock_text, microsecond) = match clock_text.split_once('.') {
        Some((clock, fraction)) => {
            // the fixed six-digit microsecond tail this crate writes
            if fraction.len() != 6 {
                return None;
            }
            (clock, fraction.parse::<i64>().ok()?)
        }
        None => (clock_text, 0),
    };
    let mut clock_parts = clock_text.split(':');
    let hour: i64 = clock_parts.next()?.parse().ok()?;
    let minute: i64 = clock_parts.next()?.parse().ok()?;
    let second: i64 = clock_parts.next()?.parse().ok()?;
    if clock_parts.next().is_some() {
        return None;
    }
    let days = epoch_days_of_civil_date(year, month, day, kernel)?;
    let seconds = days as i128 * 86_400 + hour as i128 * 3_600 + minute as i128 * 60 + second as i128;
    Some(seconds * 1_000_000 + microsecond as i128)
}

/// One tagged `datetime_datetime` instance's instant as a whole count of
/// MICROSECONDS from the POSIX epoch — `datetime_difference_value`'s own
/// reduction, the same calendar-day-times-86400-plus-wall-clock form
/// `datetime_timestamp_value` uses (the kernel's `epochDays` ask carries
/// the calendar half), with the microsecond field added and the
/// instance's UTC offset subtracted so two instants in different zones
/// are differenced over the same origin. A NAIVE instance carries no
/// offset, and datetime.rst note (3)'s first paragraph says the tzinfo
/// attributes are ignored for a naive pair — so 0 is the right shift for
/// both sides of such a pair.
fn datetime_epoch_microseconds(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<i128> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let hour = datetime_field(instance, "hour")? as i64;
    let minute = datetime_field(instance, "minute")? as i64;
    let second = datetime_field(instance, "second")? as i64;
    let microsecond = datetime_field(instance, "microsecond")? as i64;
    let days = epoch_days_of_civil_date(year, month, day, kernel)?;
    let seconds = days as i128 * 86_400 + hour as i128 * 3_600 + minute as i128 * 60 + second as i128;
    Some(seconds * 1_000_000 + microsecond as i128)
}

/// `timedelta2 // timedelta3` — datetime.rst's `timedelta` operation
/// table (row `t1 = t2 // i` or `t1 = t2 // t3`): "The floor is computed
/// and the remainder (if any) is thrown away. In the second case, an
/// integer is returned." Both operands reduce to their whole microsecond
/// counts (`timedelta_total_microseconds`, the normalized triple
/// datetime.rst:221 stores), and the answer is the FLOOR of the
/// quotient, matching Python's own floor semantics for `//` on a
/// negative dividend.
///
/// A WINDOW dividend (a `datetime_window_difference_value` result, whose
/// duration is a microsecond RANGE rather than one triple) answers the
/// range of quotients instead — the floor is monotone in the dividend
/// for a positive divisor, so the range's own two endpoints bound it.
///
/// `None` for a zero divisor — note (3) on that same table: "Division by
/// zero raises `ZeroDivisionError`," so no value flows there.
pub(in crate::expressions) fn timedelta_floordiv_value(left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let divisor = super::construction::timedelta_total_microseconds(right)?;
    if divisor == 0 {
        return None;
    }
    if let Some(dividend) = super::construction::timedelta_total_microseconds(left) {
        return Some(known_values(vec![python_floordiv(dividend, divisor) as f64], PrimitiveKind::Integer, TrustProved));
    }
    // the window shape: two Integer fields naming the microsecond range,
    // or NO fields at all when the difference itself is unbounded (a
    // `datetime_window_difference_value` over an unbounded window) — in
    // which case the answer is the unbounded integer sort, the plain
    // claim datetime.rst's `t2 // t3` row states ("an integer is
    // returned"), leaving a later guard to narrow it.
    let mut forms = vec![refined_sets::refinement_forms::integer()];
    if let (Some(low), Some(high)) = (datetime_field(left, "low_microseconds"), datetime_field(left, "high_microseconds")) {
        let (low_quotient, high_quotient) = if divisor > 0 {
            (python_floordiv(low as i128, divisor), python_floordiv(high as i128, divisor))
        } else {
            (python_floordiv(high as i128, divisor), python_floordiv(low as i128, divisor))
        };
        forms.push(refined_sets::refinement_forms::at_least(low_quotient as f64));
        forms.push(refined_sets::refinement_forms::at_most(high_quotient as f64));
    } else {
        forms.push(refined_sets::refinement_forms::at_least(f64::NEG_INFINITY));
    }
    let set = refined_sets::refinement_forms::make_refined_set(forms);
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(set, None, TrustProved, refined_domain::abstract_value::SetKindTag::None)
    })
}

/// Python's `//` on two integers — floor division. Rust's `/` truncates
/// toward zero, so a quotient with a nonzero remainder and mismatched
/// operand signs steps one further down.
fn python_floordiv(dividend: i128, divisor: i128) -> i128 {
    let truncated = dividend / divisor;
    let remainder = dividend % divisor;
    if remainder != 0 && ((dividend < 0) != (divisor < 0)) { truncated - 1 } else { truncated }
}

/// A UTC instant spelling shifted by a whole number of MICROSECONDS,
/// answered in the same fixed-width UTC form. The shifted count splits
/// back into calendar and clock fields through the kernel's `isoDate` op
/// — the same self-certifying arm every other date reduction in this
/// file uses — so no date math happens here. `None` for a spelling this
/// reader does not split, or a refused kernel ask.
///
/// Used by the narrowing channel to spell a STRICT instant comparison:
/// the calendar window carries no open/closed bit, so `d < CUTOFF` is
/// spelled as the window ending one microsecond before `CUTOFF`.
pub fn instant_stepped_by_microseconds(text: &str, step: i128, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    let shifted = utc_iso_microseconds(text, kernel)? + step;
    let days = i64::try_from(shifted.div_euclid(86_400_000_000)).ok()?;
    let within_day = shifted.rem_euclid(86_400_000_000);
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoDate,
            year: 0,
            month: 0,
            day: 0,
            days,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let year = asked.get("year")?.as_i64()?;
    let month = asked.get("month")?.as_i64()?;
    let day = asked.get("day")?.as_i64()?;
    let microsecond = within_day % 1_000_000;
    let seconds_of_day = within_day / 1_000_000;
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
    Some(if microsecond == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{microsecond:06}Z")
    })
}

/// One expression's own duration in MICROSECONDS, when it evaluates to a
/// tagged `datetime_timedelta` instance carrying an exact normalized
/// triple — the divisor half of `check`'s temporal-offset derivation
/// reader. `None` for anything that is not such a duration.
pub fn timedelta_microseconds_of_expression(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i128> {
    let value = crate::expressions::evaluate_expression(expr, environment, kernel);
    if value.source != "datetime_timedelta" {
        return None;
    }
    super::construction::timedelta_total_microseconds(&value)
}

/// One expression's own instant as a MICROSECOND count from the POSIX
/// epoch, when it evaluates to a concrete `datetime_datetime`
/// construction — the origin half of `check`'s temporal-offset
/// derivation reader. `None` for a window, an unresolved offset, or
/// anything that is not a datetime construction at all.
pub fn exact_instant_microseconds_of_expression(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i128> {
    let value = crate::expressions::evaluate_expression(expr, environment, kernel);
    if value.source != "datetime_datetime" {
        return None;
    }
    datetime_epoch_microseconds(&value, kernel)
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
