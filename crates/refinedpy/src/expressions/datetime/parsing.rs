//! Parsing text into a tagged instance: `date.fromisoformat`, its
//! raise-side twin, and `datetime.strptime`'s directive-by-directive
//! grammar (STAGE 1's ISO shortcut and STAGE 2's full scanner/parser).

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;

use super::construction::integer_object_key;
use super::construction::python_year_in_range;
use super::construction::valid_civil_date;

/// `date.fromisoformat("YYYY-MM-DD")` — datetime.rst, `classmethod::
/// date.fromisoformat(date_string)`. Modeled ONLY for the strict
/// `YYYY-MM-DD` shape date.3's own row states as the committed
/// (non-reduced-precision, non-extended, non-ordinal) grammar — a known
/// exact string this file can split by its two ASCII hyphens into three
/// all-digit runs. The parsed year/month/day is then validated through
/// the SAME two kernel asks `date_construction_value` uses —
/// `python_year_in_range`'s `pyYearInRange` for date.2's window, then
/// `calendar.validDate` for calendar correctness — so a syntactically
/// well-shaped but calendrically invalid string (`"2023-02-30"`)
/// declines the same way a bad `datetime.date(...)` construction does.
/// Any other shape (a non-string argument, an unparseable string, a
/// string with the wrong hyphen count or non-digit runs) answers
/// `None`.
pub(in crate::expressions) fn date_fromisoformat_value(text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let mut parts = text.split('-');
    let year_text = parts.next()?;
    let month_text = parts.next()?;
    let day_text = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if year_text.len() != 4 || month_text.len() != 2 || day_text.len() != 2 {
        return None;
    }
    if !year_text.bytes().all(|b| b.is_ascii_digit())
        || !month_text.bytes().all(|b| b.is_ascii_digit())
        || !day_text.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let year: i64 = year_text.parse().ok()?;
    let month: i64 = month_text.parse().ok()?;
    let day: i64 = day_text.parse().ok()?;
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    if !valid_civil_date(year, month, day, kernel)? {
        return None;
    }
    let keys = vec![integer_object_key("year", year), integer_object_key("month", month), integer_object_key("day", day)];
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_date".to_owned();
    let point = format!("{year:04}-{month:02}-{day:02}");
    instance.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
        chart: refined_sets::calendar_interpreter::TemporalChart::PlainDate,
        min: Some(point.clone()),
        max: Some(point),
    }));
    Some(instance)
}

/// Whether `date.fromisoformat(text)` provably RAISES `ValueError`, for a
/// KNOWN exact `text` argument — the raise-dispatch twin of
/// `date_fromisoformat_value`'s own value dispatch, read the SAME way
/// `call_provable_raise`'s existing `math.sqrt`/`int(<string>)` rows pair
/// their own value-side reader with a raise-side classifier. datetime.rst
/// states `fromisoformat` "will raise a `ValueError`" on a string it does
/// not accept — CPython's own implementation note (`Changed in version
/// 3.11: ... Previously, this method only supported the format YYYY-MM-DD`)
/// backs the strict `YYYY-MM-DD` grammar date.3's own row already commits
/// this file to reading, so a string OUTSIDE that grammar (`"13:45"`, a
/// clock time with no date fields at all) is exactly as much a raise as a
/// syntactically-shaped but calendrically-invalid one (`"2023-02-29"`,
/// `"2023-04-31"`) — CPython's own `_parse_isoformat_date` raises
/// `ValueError: Invalid isoformat string` on the former and the
/// `datetime.date` CONSTRUCTOR raises on the latter, and neither
/// distinction changes what a caller can safely assume: the call raises
/// either way.
///
/// Returns `Some(true)` only where the string is PROVABLY malformed or
/// PROVABLY calendrically invalid (every kernel ask involved answered,
/// never refused) — `Some(false)` for a provably VALID date (mirrors
/// `date_fromisoformat_value`'s own `Some` case exactly, so the two
/// functions never disagree on the same string), and `None` wherever a
/// kernel ask this reads (`python_year_in_range`/`valid_civil_date`)
/// itself declines to answer — an honest "cannot tell" rather than a
/// guessed raise.
pub(in crate::expressions) fn date_fromisoformat_raises(text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let mut parts = text.split('-');
    let Some(year_text) = parts.next() else {
        return Some(true);
    };
    let Some(month_text) = parts.next() else {
        return Some(true);
    };
    let Some(day_text) = parts.next() else {
        return Some(true);
    };
    if parts.next().is_some() {
        return Some(true);
    }
    if year_text.len() != 4 || month_text.len() != 2 || day_text.len() != 2 {
        return Some(true);
    }
    if !year_text.bytes().all(|b| b.is_ascii_digit())
        || !month_text.bytes().all(|b| b.is_ascii_digit())
        || !day_text.bytes().all(|b| b.is_ascii_digit())
    {
        return Some(true);
    }
    let Ok(year) = year_text.parse::<i64>() else { return Some(true) };
    let Ok(month) = month_text.parse::<i64>() else { return Some(true) };
    let Ok(day) = day_text.parse::<i64>() else { return Some(true) };
    let year_ok = python_year_in_range(year, kernel)?;
    if !year_ok {
        return Some(true);
    }
    let date_ok = valid_civil_date(year, month, day, kernel)?;
    Some(!date_ok)
}

/// date.12 STAGE 1 — the ISO-equivalent directive subset of
/// `datetime.strptime(date_string, format)` (datetime.rst,
/// `classmethod:: datetime.strptime(date_string, format)`: "Return a
/// datetime corresponding to date_string, parsed according to format").
/// Modeled ONLY for the exact literal format `"%Y-%m-%d"` — the ISO
/// `YYYY-MM-DD` directive sequence date.3's grammar already commits to
/// (`%Y` datetime.rst:2413-2415, `%m` :2407-2409, `%d` :2394-2396, each
/// a zero-padded decimal field) — lowered to EXACTLY the same value
/// `date_fromisoformat_value` binds for the identical text: this
/// function reuses that function outright rather than re-deriving its
/// parse or its two kernel asks (`pyYearInRange` then `validDate`), so
/// `strptime(text, "%Y-%m-%d")` and `date.fromisoformat(text)` produce
/// the SAME `AbstractValue` for the same `text` — a `datetime_date`-
/// tagged instance, not a `datetime_datetime` one: the format carries
/// no time-of-day directive, so the honest value this file can prove is
/// calendar-date-shaped, even though CPython's real return type is
/// `datetime`. EXCLUDED from this stage: any `"%H:%M:%S"`-composite
/// format (`"%Y-%m-%d %H:%M:%S"` and similar) — datetime.rst's own
/// `strftime`/`strptime` directive table gives each of `%H`/`%M`/`%S`
/// (:2416-2430) no existing kernel-crossed bind this file's
/// `datetime_datetime` construction path reads FROM a string (only
/// FROM already-known Integer arguments,
/// `datetime_construction_value`'s own doc), so composing them here
/// would invent a new value shape this stage does not build; a non-ISO
/// literal format or a non-literal (computed) format both decline
/// through this function's own caller, never reaching here.
pub(in crate::expressions) fn strptime_iso_date_value(text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    date_fromisoformat_value(text, kernel)
}

/// `%Y` — datetime.rst:2413-2415, "Year with century as a decimal
/// number," note (2): "years < 1000 must be zero-filled to 4-digit
/// width." Reads EXACTLY 4 digits, the same width `date_fromisoformat_
/// value`'s own `year_text` split already commits to — note (9)'s
/// optional-leading-zero list does NOT name `%Y`, so this directive
/// alone keeps the fixed-width read rather than the variable-width rule
/// every OTHER numeric directive below takes.
pub(in crate::expressions) fn read_year_field(rest: &str) -> Option<(i64, &str)> {
    let (digits, tail) = take_fixed_digits(rest, 4)?;
    let value: i64 = digits.parse().ok()?;
    Some((value, tail))
}

/// `%y` — datetime.rst:2410-2412, "Year without century as a zero-padded
/// decimal number," note (9): "Format `%y` DOES require a leading zero"
/// — so this reads EXACTLY 2 digits, never 1. The two-digit value is
/// then pivoted to its full year per time.rst:45-48 ("values 69--99 are
/// mapped to 1969--1999, and values 0--68 are mapped to 2000--2068") —
/// the POSIX/ISO C rule datetime.strptime itself defers to (this table's
/// own `%y` row cites no separate pivot, so the platform-shared rule
/// applies).
pub(in crate::expressions) fn read_two_digit_year_field(rest: &str) -> Option<(i64, &str)> {
    let (digits, tail) = take_fixed_digits(rest, 2)?;
    let two_digit: i64 = digits.parse().ok()?;
    let full_year = if two_digit >= 69 { 1900 + two_digit } else { 2000 + two_digit };
    Some((full_year, tail))
}

/// `%m`/`%d`/`%H`/`%M`/`%S` — each a zero-padded two-digit field in the
/// table (`%m` :2407-2409, `%d` :2394-2396, `%H` :2416-2418, `%M`
/// :2425-2427, `%S` :2428-2430), but note (9) makes the leading zero
/// OPTIONAL for strptime across this exact directive set. Reads 1 or 2
/// digits, greedily preferring 2 — the same optional-width rule every
/// directive note (9) names shares, spelled once here rather than
/// per-directive since the read rule is identical for all five.
pub(in crate::expressions) fn read_one_or_two_digit_field(rest: &str) -> Option<(i64, &str)> {
    let (digits, tail) = take_variable_digits(rest, 1, 2)?;
    let value: i64 = digits.parse().ok()?;
    Some((value, tail))
}

/// `%j` — datetime.rst:2443-2445, "Day of the year as a zero-padded
/// decimal number," 001-366; note (9) makes the leading zero optional
/// for strptime, so this reads 1 to 3 digits.
pub(in crate::expressions) fn read_day_of_year_field(rest: &str) -> Option<(&str, &str)> {
    take_variable_digits(rest, 1, 3)
}

/// `%U`/`%W` — datetime.rst:2446-2461, week-of-year 00-53; note (9)
/// makes the leading zero optional. Reads 1 or 2 digits. Note (7): "`%U`
/// and `%W` are only used in calculations when the day of the week and
/// the calendar year (`%Y`) are specified" — this stage carries no
/// day-of-week directive (`%a`/`%w`/`%u` are all outside this round's
/// transcribed set), so a recognized `%U`/`%W` field is read and range-
/// checked but never folded into the constructed date, matching the
/// spec's own statement that it does nothing without a weekday alongside
/// it.
pub(in crate::expressions) fn read_week_number_field(rest: &str) -> Option<(&str, &str)> {
    take_variable_digits(rest, 1, 2)
}

/// `%f` — datetime.rst note (5): "the `%f` directive accepts from one to
/// six digits and zero pads on the right" (e.g. `"5"` reads as `500000`
/// microseconds, `"123"` as `123000`). Reads 1 to 6 digits and pads the
/// result to 6 digits before parsing, matching the note's own rule
/// exactly.
pub(in crate::expressions) fn read_microsecond_field(rest: &str) -> Option<(i64, &str)> {
    let (digits, tail) = take_variable_digits(rest, 1, 6)?;
    let mut padded = digits.to_owned();
    while padded.len() < 6 {
        padded.push('0');
    }
    let value: i64 = padded.parse().ok()?;
    Some((value, tail))
}

/// `%z` — datetime.rst note (6)'s own STRPTIME grammar (the aware-object
/// paragraph, ":strptime" versionchanged 3.7 note): `±HHMM[SS[.ffffff]]`,
/// where `HH`/`MM`/`SS` are each exactly 2 digits and `ffffff` is exactly
/// 6, OR the bare literal `'Z'` ("providing `'Z'` is identical to
/// `'+00:00'`"). The same versionchanged note adds that "the UTC offsets
/// can have a colon as a separator between hours, minutes and seconds"
/// (`'+01:00:00'` parses as one hour) — an optional `:` is accepted
/// before `MM`, before `SS`, and before `ffffff`, independently of
/// whether an earlier one was present, matching CPython's own `_strptime.
/// py` regex (`(?P<z>[+-]\d\d:?[0-5]\d(:?[0-5]\d(\.\d{1,6})?)?|Z)`).
/// Returns the offset's total signed magnitude in SECONDS (microseconds
/// dropped — this stage's own tagged instance carries no sub-second
/// field, matching `%f`'s own "read but not carried" convention above) —
/// exactly `0` for `'Z'` or any spelling of `+00:00[:00[.000000]]`, the
/// one value `strptime2_parse`'s own `aware_utc` marker can set `true`
/// for; every other value still parses (the offset is consumed and the
/// construction still succeeds), but the instance answers `aware_utc:
/// false`, the same "an aware-but-non-UTC construction is not modeled
/// past this field" boundary `datetime_construction_value`'s own
/// `tzinfo=` gate already draws for the forward direction.
pub(in crate::expressions) fn read_utc_offset_field(rest: &str) -> Option<(i64, &str)> {
    if let Some(tail) = rest.strip_prefix('Z') {
        return Some((0, tail));
    }
    let sign = match rest.as_bytes().first() {
        Some(b'+') => 1i64,
        Some(b'-') => -1i64,
        _ => return None,
    };
    let rest = &rest[1..];
    let (hours_text, rest) = take_fixed_digits(rest, 2)?;
    let hours: i64 = hours_text.parse().ok()?;
    let rest = rest.strip_prefix(':').unwrap_or(rest);
    let (minutes_text, rest) = take_fixed_digits(rest, 2)?;
    let minutes: i64 = minutes_text.parse().ok()?;
    let mut total_seconds = hours * 3600 + minutes * 60;
    let mut tail = rest;
    if let Some(after_colon) = rest.strip_prefix(':') {
        if let Some((seconds_text, seconds_tail)) = take_fixed_digits(after_colon, 2) {
            let seconds: i64 = seconds_text.parse().ok()?;
            total_seconds += seconds;
            tail = seconds_tail;
        }
    } else if let Some((seconds_text, seconds_tail)) = take_fixed_digits(rest, 2) {
        let seconds: i64 = seconds_text.parse().ok()?;
        total_seconds += seconds;
        tail = seconds_tail;
    }
    // the optional `.ffffff` microsecond tail, read and discarded — this
    // stage carries no sub-second field on the constructed instance
    // (matching `%f`'s own convention), and a fractional offset never
    // changes whether the WHOLE offset is zero (a nonzero `ffffff` can
    // only sit alongside an `SS` already read above)
    if let Some(after_dot) = tail.strip_prefix('.') {
        if let Some((_, fraction_tail)) = take_variable_digits(after_dot, 1, 6) {
            tail = fraction_tail;
        }
    }
    Some((sign * total_seconds, tail))
}

/// `%a` under the PORTABLE `'C'` LOCALE — datetime.rst note (1) marks
/// `%a` locale-dependent ("Weekday as locale's abbreviated name"), and
/// its own table row shows the `en_US` spelling `Sun, Mon, ..., Sat`. A
/// program that never calls `locale.setlocale` runs under the C locale
/// (locale.rst:326-327, "According to POSIX, a program which has not
/// called `setlocale(LC_ALL, '')` runs using the portable `'C'` locale"),
/// whose weekday abbreviations are this same fixed ASCII set (the C
/// locale IS the `en_US`/POSIX default this table's own example row
/// shows) — a closed, host-independent set this reader can therefore
/// name exactly, gated on that premise by this function's own caller
/// (`strptime2_module_never_calls_setlocale`) rather than assumed here.
/// Case-sensitive, matching CPython's own `_strptime.py` locale-time
/// table exactly (no case-folding on the input).
const C_LOCALE_WEEKDAY_ABBREVIATIONS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/// Reads one of `C_LOCALE_WEEKDAY_ABBREVIATIONS` off the front of `rest`,
/// or `None` when no member matches. The value itself is not folded into
/// the constructed instance — `%U`/`%W`'s own doc: a weekday name alone
/// (with no `%Y`-anchored week-number calculation alongside it) does
/// nothing to the constructed date either, matching this stage's existing
/// "read but not carried" convention for a directive whose value this
/// stage's tagged shape has no field for.
pub(in crate::expressions) fn read_weekday_abbreviation_field(rest: &str) -> Option<&str> {
    for candidate in C_LOCALE_WEEKDAY_ABBREVIATIONS {
        if let Some(tail) = rest.strip_prefix(candidate) {
            return Some(tail);
        }
    }
    None
}

/// Exactly `width` ASCII digits off the front of `rest`, or `None` if
/// fewer than `width` digits are available or a non-digit byte sits
/// inside that span. `date_fromisoformat_value`'s own `year_text`/
/// `month_text`/`day_text` split is this same rule specialized to `%Y`'s
/// fixed 4-digit width.
pub(in crate::expressions) fn take_fixed_digits(rest: &str, width: usize) -> Option<(&str, &str)> {
    if rest.len() < width || !rest.as_bytes()[..width].iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some((&rest[..width], &rest[width..]))
}

/// 1 to `max_width` ASCII digits off the front of `rest`, preferring the
/// LONGEST run up to `max_width` — the greedy read note (9)'s optional-
/// leading-zero rule needs (a written `"1"` before a literal `"/"` reads
/// as one digit; a written `"12"` reads as two, never leaving a spare
/// digit for the next directive to misread). `None` when fewer than
/// `min_width` digits are available.
pub(in crate::expressions) fn take_variable_digits(rest: &str, min_width: usize, max_width: usize) -> Option<(&str, &str)> {
    let available = rest.bytes().take_while(u8::is_ascii_digit).count();
    if available < min_width {
        return None;
    }
    let width = available.min(max_width);
    Some((&rest[..width], &rest[width..]))
}

/// The three decline reasons date.12 STAGE 2 names, per the AGENT-BRIEF's
/// own split: an UNREAD directive (`%Z %I %G %u %V`, this round's named
/// remainder — not yet transcribed against the spec, but a host-
/// independent set is buildable once it is); a LOCALE directive (`%A %b
/// %B %p %c %x %X`, datetime.rst note (1): "the format depends on the
/// current locale... Field orderings will vary... and the output may
/// contain non-ASCII characters") — a genuinely different construct,
/// since a locale directive has no host-independent value set to derive
/// AT ALL, not merely one this round left unread; and `WeekdayAbbreviation`
/// (`%a` alone, split out from the other six locale directives) — under
/// the PORTABLE `'C'` locale POSIX runs by default (locale.rst:326-327),
/// `%a`'s own value set IS host-independent (`read_weekday_abbreviation_
/// field`'s own doc), so this format is not a permanent decline the way
/// the other five locale directives are — the caller
/// (`evaluate_attribute_call`) reads the module's own C-locale premise
/// (`environment.locale_never_set()`, `module_never_calls_setlocale`'s
/// own doc) and passes it to `strptime2_scan_format` as
/// `accept_weekday_abbreviation`, so a `WeekdayAbbreviation` outcome
/// from THIS scanner only ever happens when the premise does not hold
/// — the scanner treats `%a` as transcribed and never returns this
/// variant at all when the caller passes `accept_weekday_abbreviation
/// = true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::expressions) enum Strptime2Decline {
    UnreadDirective(char),
    LocaleDirective(char),
    WeekdayAbbreviation,
}

/// Whether `letter` is one of the six PERMANENTLY locale-dependent
/// directives datetime.rst note (1) names, EXCLUDING `%a` — split out as
/// `Strptime2Decline::WeekdayAbbreviation`'s own case, since the C-locale
/// premise makes `%a` alone determinable while these five stay a genuine
/// decline (no host-independent full-weekday-name/month-name/AM-PM/
/// locale-composite set exists for any of `%A %b %B %p %c %x %X`).
pub(in crate::expressions) fn is_locale_directive(letter: char) -> bool {
    matches!(letter, 'A' | 'b' | 'B' | 'p' | 'c' | 'x' | 'X')
}

/// STAGE 2's directive transcription: every letter this round reads
/// against datetime.rst's format-codes table (`%Y %m %d %H %M %S %f %j
/// %U %W %y %z`, plus `%%`'s literal-percent escape, :2474-2475) answers
/// `Ok(())`; `Err` names the FIRST directive letter this round does NOT
/// transcribe outright, distinguishing the locale boundary, the weekday-
/// abbreviation boundary, and the plain not-yet-read one
/// (`Strptime2Decline`'s own doc) — the AGENT-BRIEF's own requirement
/// that an unread directive names ITSELF, never the whole format string.
/// Run as its own pre-pass before `strptime2_parse` so a format's decline
/// reason is named even when the TEXT does not match at all
/// (`strptime2_parse` has no reason channel of its own, only `None`).
///
/// `accept_weekday_abbreviation` is the caller's own C-locale premise
/// (`module_never_calls_setlocale`'s own doc): `true` treats `%a` as a
/// TRANSCRIBED directive, the same as `%Y`/`%m`/etc — the scan continues
/// past it rather than stopping, so a LATER genuinely-locale directive
/// in the same format (`%a %A`) still surfaces its own
/// `LocaleDirective` reason rather than being masked by `%a`'s earlier
/// position. `false` keeps the original behavior: `%a` itself is the
/// first-blocking directive, named `WeekdayAbbreviation`.
pub(in crate::expressions) fn strptime2_scan_format(format: &str, accept_weekday_abbreviation: bool) -> Result<(), Strptime2Decline> {
    let mut chars = format.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '%' {
            continue;
        }
        let Some(letter) = chars.next() else {
            continue; // a trailing lone '%' — no directive follows, nothing to name
        };
        match letter {
            '%' | 'Y' | 'y' | 'm' | 'd' | 'H' | 'M' | 'S' | 'f' | 'j' | 'U' | 'W' | 'z' => continue,
            'a' if accept_weekday_abbreviation => continue,
            'a' => return Err(Strptime2Decline::WeekdayAbbreviation),
            other if is_locale_directive(other) => return Err(Strptime2Decline::LocaleDirective(other)),
            other => return Err(Strptime2Decline::UnreadDirective(other)),
        }
    }
    Ok(())
}

/// STAGE 2's parse: walks `format` and `text` in lockstep, matching each
/// literal span byte-for-byte and each directive through its own reader
/// above, then folds the read fields into a tagged instance the SAME way
/// `datetime_construction_value` does (`year`/`month`/`day`/`hour`/
/// `minute`/`second` Integer `ObjectKey`s, `aware_utc` true only when a
/// recognized `%z` read an exactly-zero offset — `read_utc_offset_field`'s
/// own doc; every other case, including a naive format with no `%z` at
/// all, answers `aware_utc: false`) — the fields this stage's own
/// construct feeds are read FROM a string here, the mirror of
/// `datetime_construction_value` reading them FROM already-known Integer
/// arguments. Absent fields (an `%H:%M:%S`-less format) default to 0,
/// `datetime.strptime`'s own default-value rule (datetime.rst:2527-2529:
/// "the default value is `1900-01-01T00:00:00.000`: any components not
/// specified in the format string will be pulled from the default
/// value") restricted to the six components this file tracks —
/// `year`/`month`/`day` are NOT required by a directive in the format (a
/// bare `"%H:%M:%S"` format is modeled, defaulting to 1900-01-01,
/// matching `test_strptime_stage_2_derives_a_time_of_day_only_format`'s
/// own pin). Declines (`None`) on any text that does not match the
/// literal/directive sequence, a repeated field kind (two `%Y`s), or a
/// value that fails the SAME two kernel asks `date_fromisoformat_value`
/// poses (`pyYearInRange` then `validDate`). `%f`'s microsecond is read
/// but not carried on the constructed instance — `datetime_construction_
/// value`'s own tagged shape has no `microsecond` field (its own doc: "a
/// `microsecond`/`fold` argument... declines the WHOLE construction" for
/// the FORWARD constructor, so this reverse direction matches by simply
/// not adding one), so `%f` is recognized and its digits validated (1-6
/// digits, note 5) but its resolved value does not reach the returned
/// instance. `%a`'s weekday abbreviation is likewise read and range-
/// checked (against the caller's own C-locale premise gate — this
/// function itself poses no premise question, matching `%U`/`%W`'s own
/// "read but not carried" convention) but never folded into the
/// constructed date, the same as those two directives.
pub(in crate::expressions) fn strptime2_parse(format: &str, text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let mut year: Option<i64> = None;
    let mut month: Option<i64> = None;
    let mut day: Option<i64> = None;
    let mut hour: Option<i64> = None;
    let mut minute: Option<i64> = None;
    let mut second: Option<i64> = None;
    let mut utc_offset_seconds: Option<i64> = None;

    let mut format_chars = format.chars().peekable();
    let mut remaining_text = text;
    while let Some(ch) = format_chars.next() {
        if ch != '%' {
            remaining_text = remaining_text.strip_prefix(ch)?;
            continue;
        }
        let letter = format_chars.next()?;
        match letter {
            '%' => {
                remaining_text = remaining_text.strip_prefix('%')?;
            }
            'Y' => {
                let (value, tail) = read_year_field(remaining_text)?;
                set_once(&mut year, value)?;
                remaining_text = tail;
            }
            'y' => {
                let (value, tail) = read_two_digit_year_field(remaining_text)?;
                set_once(&mut year, value)?;
                remaining_text = tail;
            }
            'm' => {
                let (value, tail) = read_one_or_two_digit_field(remaining_text)?;
                if !(1..=12).contains(&value) {
                    return None;
                }
                set_once(&mut month, value)?;
                remaining_text = tail;
            }
            'd' => {
                let (value, tail) = read_one_or_two_digit_field(remaining_text)?;
                if !(1..=31).contains(&value) {
                    return None;
                }
                set_once(&mut day, value)?;
                remaining_text = tail;
            }
            'H' => {
                let (value, tail) = read_one_or_two_digit_field(remaining_text)?;
                if !(0..=23).contains(&value) {
                    return None;
                }
                set_once(&mut hour, value)?;
                remaining_text = tail;
            }
            'M' => {
                let (value, tail) = read_one_or_two_digit_field(remaining_text)?;
                if !(0..=59).contains(&value) {
                    return None;
                }
                set_once(&mut minute, value)?;
                remaining_text = tail;
            }
            'S' => {
                let (value, tail) = read_one_or_two_digit_field(remaining_text)?;
                if !(0..=59).contains(&value) {
                    return None;
                }
                set_once(&mut second, value)?;
                remaining_text = tail;
            }
            'f' => {
                let (_value, tail) = read_microsecond_field(remaining_text)?;
                remaining_text = tail;
            }
            'j' => {
                let (digits, tail) = read_day_of_year_field(remaining_text)?;
                let value: i64 = digits.parse().ok()?;
                if !(1..=366).contains(&value) {
                    return None;
                }
                remaining_text = tail;
            }
            'U' | 'W' => {
                let (digits, tail) = read_week_number_field(remaining_text)?;
                let value: i64 = digits.parse().ok()?;
                if !(0..=53).contains(&value) {
                    return None;
                }
                remaining_text = tail;
            }
            'z' => {
                let (offset_seconds, tail) = read_utc_offset_field(remaining_text)?;
                set_once(&mut utc_offset_seconds, offset_seconds)?;
                remaining_text = tail;
            }
            'a' => {
                let tail = read_weekday_abbreviation_field(remaining_text)?;
                remaining_text = tail;
            }
            // every other letter is caught by strptime2_scan_format's own
            // pre-pass before this function is ever called — unreachable
            // here by construction, matching this file's own convention
            // of proving unreachable states through the caller's gate
            // rather than re-checking them a second time
            _ => return None,
        }
    }
    if !remaining_text.is_empty() {
        return None; // trailing text the format did not account for
    }
    // datetime.rst:2527-2529's own default value, `1900-01-01T00:00:00.
    // 000`: any component the format did not name is pulled from this
    // default, restricted to the three date components (hour/minute/
    // second already default to 0 below). Since 1900 is not a leap year,
    // a partial format landing on default-year February 29 fails the
    // SAME `valid_civil_date` kernel ask any other invalid date fails
    // through — reproducing datetime.rst:2531-2546's own documented
    // "will raise when encountering February 29" note without a
    // separate special case.
    let year = year.unwrap_or(1900);
    let month = month.unwrap_or(1);
    let day = day.unwrap_or(1);
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    if !valid_civil_date(year, month, day, kernel)? {
        return None;
    }
    // `aware_utc` is true only when a recognized `%z` read EXACTLY the
    // zero offset (`'Z'`, `'+0000'`, `'+00:00'`, …) — the one shape
    // `read_utc_offset_field`'s own doc names as this stage's own
    // determinable aware case; every other offset still parses
    // (the text is consumed, the construction still succeeds) but the
    // instance answers naive, matching `datetime_construction_value`'s
    // own forward-direction boundary at a non-UTC `tzinfo=`.
    let aware_utc = utc_offset_seconds == Some(0);
    let keys = vec![
        integer_object_key("year", year),
        integer_object_key("month", month),
        integer_object_key("day", day),
        integer_object_key("hour", hour.unwrap_or(0)),
        integer_object_key("minute", minute.unwrap_or(0)),
        integer_object_key("second", second.unwrap_or(0)),
        ObjectKey {
            name: "aware_utc".to_owned(),
            numeric: false,
            value: known_values(vec![if aware_utc { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved),
        },
    ];
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_datetime".to_owned();
    Some(instance)
}

/// Sets an `Option<i64>` slot the FIRST time a directive fills it, and
/// declines (`None`) if the SAME calendar component is named twice by
/// the format (`"%Y-%Y"` or similar) — CPython's own `_strptime.py`
/// raises `ValueError` on a repeated directive naming the same group;
/// this file has no exception channel, so the construction simply
/// declines, matching every other malformed-input row in this stage.
pub(in crate::expressions) fn set_once(slot: &mut Option<i64>, value: i64) -> Option<()> {
    if slot.is_some() {
        return None;
    }
    *slot = Some(value);
    Some(())
}
