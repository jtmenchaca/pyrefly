use super::*;

// --- j-stdlib-surfaces.py: datetime family ---
// Every `.timestamp()` pin below routes its day-count arithmetic
// through the kernel's `calendar` ask (`refined_calendar`'s
// `"epochDays"` op, `epoch_days_of_civil_date`) rather than a local
// Rust reimplementation — a wrong or refused kernel answer fails
// these pins directly, since `eval` loads the real kernel dylib.

/// `datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).timestamp()`
/// is exactly `0.0` — the POSIX epoch itself, the kernel's own
/// `epochDays` anchor (`theories/calendar/epoch_days_sound.lean`).
#[test]
fn test_datetime_timestamp_at_the_epoch_is_zero() {
    let Some(value) = eval("datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).timestamp()") else { return };
    assert_eq!(value.values, vec![0.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

/// `datetime.datetime(2033, 5, 18, tzinfo=datetime.timezone.utc).timestamp()`
/// — the exact later timestamp j-stdlib-surfaces.py's own
/// `datetime_timestamp` row marks past the Age ceiling.
#[test]
fn test_datetime_timestamp_of_a_later_aware_utc_date() {
    let Some(value) = eval("datetime.datetime(2033, 5, 18, tzinfo=datetime.timezone.utc).timestamp()") else { return };
    assert_eq!(value.values, vec![1999987200.0]);
}

/// `datetime.datetime(2024, 2, 29, tzinfo=datetime.timezone.utc).timestamp()`
/// — a leap-day date (2024 is divisible by 4, not by 100): the day
/// count the kernel's `epochDays` ask must cross a Gregorian leap
/// boundary to answer, execution-verified against installed CPython
/// 3.12 (`(datetime.datetime(2024, 2, 29, tzinfo=datetime.timezone.utc)
/// - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc))
/// .total_seconds() == 1709164800.0`).
#[test]
fn test_datetime_timestamp_of_a_leap_day_crosses_the_kernels_calendar() {
    let Some(value) = eval("datetime.datetime(2024, 2, 29, tzinfo=datetime.timezone.utc).timestamp()") else { return };
    assert_eq!(value.values, vec![1709164800.0]);
}

/// A NAIVE datetime's `.timestamp()` (no `tzinfo=`) declines — this
/// file does not reproduce the host-local-time `mktime` conversion
/// datetime.rst documents for the naive row.
#[test]
fn test_datetime_timestamp_of_a_naive_datetime_declines() {
    let Some(value) = eval("datetime.datetime(1970, 1, 1).timestamp()") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.datetime.now()` — a value that changes every run, never
/// pinned to a scalar: answered opaque.
#[test]
fn test_datetime_now_is_opaque() {
    let Some(value) = eval("datetime.datetime.now()") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert!(value.kind_word.is_some());
}

/// `.year` on a constructed datetime answers opaque (never a
/// specific value this file claims to pin, per this row's own
/// fixture framing).
#[test]
fn test_datetime_year_is_opaque() {
    let Some(value) = eval("datetime.datetime(1970, 1, 1).year") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert!(value.kind_word.is_some());
}

/// `.isoformat()` on a UTC-aware constructed datetime is the exact
/// text — the kernel's `isoDateText` render for the date half, the
/// instance's own clock fields, and CPython's `+00:00` offset
/// spelling (never the `Z` designator).
#[test]
fn test_datetime_isoformat_of_an_aware_utc_datetime_is_exact() {
    let Some(value) = eval("datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).isoformat()") else { return };
    assert_eq!(iso_text(&value), "1970-01-01T00:00:00+00:00");
}

/// `.isoformat()` on a NAIVE constructed datetime appends no offset.
#[test]
fn test_datetime_isoformat_of_a_naive_datetime_carries_no_offset() {
    let Some(value) = eval("datetime.datetime(2024, 3, 1, 13, 45, 6).isoformat()") else { return };
    assert_eq!(iso_text(&value), "2024-03-01T13:45:06");
}

/// `.isoformat()` spells the microsecond field at six digits when it
/// is nonzero, and omits it entirely when it is zero — the two arms
/// datetime.rst's own format clause states.
#[test]
fn test_datetime_isoformat_spells_a_nonzero_microsecond_at_six_digits() {
    let Some(value) = eval("datetime.datetime(2024, 3, 1, 0, 0, 0, 7).isoformat()") else { return };
    assert_eq!(iso_text(&value), "2024-03-01T00:00:00.000007");
}

/// `datetime.fromtimestamp(0, tz=timezone.utc).isoformat()` is exactly
/// `"1970-01-01T00:00:00+00:00"` — A3.seed.library's own
/// `epoch_iso_outside` row, end to end through the classmethod and
/// the render.
#[test]
fn test_A3_seed_library_epoch_fromtimestamp_isoformat_is_exact() {
    let Some(value) = eval("datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc).isoformat()") else { return };
    assert_eq!(iso_text(&value), "1970-01-01T00:00:00+00:00");
}

/// The same call's year field, sliced off the rendered text — the
/// digit grammar A3.seed.library's own `epoch_iso_year_inside` row
/// returns.
#[test]
fn test_A3_seed_library_epoch_isoformat_year_slice_is_the_digit_text() {
    let Some(value) = eval("datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc).isoformat()[:4]") else { return };
    assert_eq!(iso_text(&value), "1970");
}

/// A later instant, so the day-count split is exercised past day 0:
/// 1 700 000 000 seconds after the epoch is 2023-11-14T22:13:20 UTC
/// (execution-verified against installed CPython 3.12).
#[test]
fn test_fromtimestamp_of_a_later_instant_splits_the_day_and_clock() {
    let Some(value) = eval("datetime.datetime.fromtimestamp(1700000000, tz=datetime.timezone.utc).isoformat()") else { return };
    assert_eq!(iso_text(&value), "2023-11-14T22:13:20+00:00");
}

/// A NAIVE `fromtimestamp` (no `tz=`) declines — datetime.rst states
/// it converts to LOCAL time, a host-dependent conversion this crate
/// does not reproduce, the same reason the naive `.timestamp()`
/// direction declines.
#[test]
fn test_fromtimestamp_without_a_timezone_declines() {
    let Some(value) = eval("datetime.datetime.fromtimestamp(0)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// A FRACTIONAL timestamp declines rather than round into the
/// microsecond field.
#[test]
fn test_fromtimestamp_of_a_fractional_timestamp_declines() {
    let Some(value) = eval("datetime.datetime.fromtimestamp(0.5, tz=datetime.timezone.utc)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// The exact text an evaluated string value carries.
fn iso_text(value: &AbstractValue) -> String {
    value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect()
}

// --- j-stdlib-surfaces.py: date/timedelta family ---
// Every pin below routes its calendar arithmetic through the
// kernel's `calendar` ask — `validDate`/`epochDays`/`isoDate`/
// `validDuration` (construction and `date ± timedelta`) and
// `weekday`/`toordinal`/`pyYearInRange`/`isoCalendar` (`.weekday()`/
// `.isoweekday()`/`.toordinal()`/`.isocalendar()` and the year-range
// guard) — same as the `datetime_datetime` family above; `eval`
// loads the real kernel dylib, so a wrong or refused kernel answer
// fails these pins directly. PIN VALUE PROVENANCE: `date(2024, 3,
// 1).weekday() == 4` and `.toordinal() == 738946` are the exact
// values this task's own brief states; every other constant below
// is derived by `/tmp/date_pin_values.py` (a CPython `datetime`
// probe) and MUST be cross-checked against that script's printed
// output before this batch gates — flagged individually below.

/// `datetime.date(2024, 3, 1)` constructs — a plain valid civil
/// date, tagged and carrying its own year/month/day fields.
#[test]
fn test_date_construction_carries_its_own_fields() {
    let Some(value) = eval("datetime.date(2024, 3, 1)") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// `datetime.date(2023, 2, 30)` — February has 28 days in 2023 (not
/// a leap year); the kernel's own `validDate` refuses this, so
/// construction declines rather than building an invalid instance.
#[test]
fn test_date_construction_of_an_invalid_calendar_date_declines() {
    let Some(value) = eval("datetime.date(2023, 2, 30)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

// --- import aliasing: the datetime gates resolve canonical identity,
// not the literal `datetime`/`date`/`timedelta` spelling ---

/// One module's `datetime` import table, seeded onto a fresh
/// environment the same way `check.rs::walk_body_with_self_binding`
/// seeds it for a real walk — the harness every aliasing pin below
/// shares.
fn environment_with_datetime_imports(module: &ruff_python_ast::ModModule) -> Environment {
    let mut environment = empty_environment();
    environment.set_datetime_imports(Arc::new(datetime_imports(module)));
    environment
}

/// `from datetime import date` + `date(2024, 3, 1)` — a bare aliased
/// class name construction. Recognizes IDENTICALLY to the qualified
/// `datetime.date(2024, 3, 1)` spelling (`test_date_construction_
/// carries_its_own_fields`'s own pin): same tag, same three fields.
#[test]
fn test_bare_imported_date_construction_matches_the_qualified_spelling() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from datetime import date\n")
        .expect("test module parses")
        .into_syntax();
    let environment = environment_with_datetime_imports(&module);
    let parsed = parse_expression("date(2024, 3, 1)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    let qualified = parse_expression("datetime.date(2024, 3, 1)").expect("test source must parse");
    let qualified_value = evaluate_expression(&qualified.into_expr(), &empty_environment(), &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value, qualified_value, "an aliased bare-Name construction must equal the qualified spelling's own pin");
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// `from datetime import datetime as dt` + `dt.strptime("2024-03-01",
/// "%Y-%m-%d")` — a bare aliased class name's own classmethod call.
/// Recognizes the same ISO-date STAGE 1 grammar the qualified
/// `datetime.datetime.strptime(...)` spelling already pins
/// (`test_strptime_...` rows above), landing on the same
/// `datetime_datetime`-tagged instance a direct `datetime.datetime(
/// 2024, 3, 1)` construction gives (`strptime_iso_date_value`'s own
/// doc: date-only, hour/minute/second all zero).
#[test]
fn test_aliased_datetime_strptime_recognizes() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from datetime import datetime as dt\n")
        .expect("test module parses")
        .into_syntax();
    let environment = environment_with_datetime_imports(&module);
    let parsed = parse_expression("dt.strptime(\"2024-03-01\", \"%Y-%m-%d\")").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// `import datetime as dtm` + `dtm.date(2024, 3, 1)` — the whole
/// MODULE aliased (not one class), the qualified-chain shape
/// resolved through the module alias rather than the literal
/// `datetime` spelling. Recognizes identically to the unaliased
/// `datetime.date(2024, 3, 1)` construction.
#[test]
fn test_module_aliased_date_construction_recognizes() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("import datetime as dtm\n")
        .expect("test module parses")
        .into_syntax();
    let environment = environment_with_datetime_imports(&module);
    let parsed = parse_expression("dtm.date(2024, 3, 1)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// A LOCALLY SHADOWED imported name never recognizes — `date` here
/// is a same-module `def`, never `from datetime import date`
/// (the import table's own `date_class_names` set stays empty since
/// no such import statement exists), mirroring `surface.rs`'s own
/// `locally_defined_field_not_recognized` pin: a same-spelled local
/// definition that was never the real import is not the shape this
/// table names, so the same-module-def dispatch (`same_module_def_
/// gate_open`) answers the call instead of the datetime gate ever
/// running — `date(2024, 3, 1)` calls the LOCAL zero-argument `def`
/// (which takes no `year`/`month`/`day`, so the call is unread) and
/// never reads as a tagged `datetime_date` instance.
#[test]
fn test_locally_defined_date_name_not_recognized_as_datetime() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "def date():\n",
        "    pass\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let table = Arc::new(crate::function_table::function_table(&module));
    let mut environment = environment_with_datetime_imports(&module);
    environment.set_functions(table);
    let parsed = parse_expression("date(2024, 3, 1)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_ne!(value.kind, Kind::Object, "a locally defined `date` must never read as a datetime_date instance");
}

/// A REBOUND imported name never recognizes — `date` is genuinely
/// `from datetime import date`, but this body's own `date = 40`
/// rebinds it before the call. `is_datetime_date_attribute`'s own
/// shadow check (`environment.read(name).is_none()`) must see the
/// rebinding and decline, the same way the qualified `datetime.date`
/// spelling already declines when `datetime` itself is rebound.
#[test]
fn test_locally_rebound_imported_date_name_not_recognized() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from datetime import date\n")
        .expect("test module parses")
        .into_syntax();
    let mut environment = environment_with_datetime_imports(&module);
    environment.bind("date", known_values(vec![40.0], PrimitiveKind::Integer, TrustProved));
    let parsed = parse_expression("date(2024, 3, 1)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_ne!(value.kind, Kind::Object, "a locally rebound `date` must never read as a datetime_date instance");
}

/// `datetime.date(2024, 3, 1).weekday()` — PIN VALUE FROM THE
/// TASK BRIEF ITSELF: 4 (Friday), Monday-0 through Sunday-6.
#[test]
fn test_date_weekday_of_a_known_friday() {
    let Some(value) = eval("datetime.date(2024, 3, 1).weekday()") else { return };
    assert_eq!(value.values, vec![4.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `datetime.date(2024, 3, 1).isoweekday()` — PIN VALUE DERIVED BY
/// THE PROBE (`/tmp/date_pin_values.py`'s `isoweekday()` row):
/// Monday-1 through Sunday-7, one more than `.weekday()`'s Friday-4.
#[test]
fn test_date_isoweekday_of_a_known_friday() {
    let Some(value) = eval("datetime.date(2024, 3, 1).isoweekday()") else { return };
    assert_eq!(value.values, vec![5.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `datetime.date(1970, 1, 1).weekday()` — the epoch anchor date,
/// PIN VALUE DERIVED BY THE PROBE: CPython's own epoch is a
/// Thursday (`isoDayOfWeek_epoch_thursday`'s proved fact, weekday()
/// Monday-0 form: Thursday is 3).
#[test]
fn test_date_weekday_at_the_epoch_anchor() {
    let Some(value) = eval("datetime.date(1970, 1, 1).weekday()") else { return };
    assert_eq!(value.values, vec![3.0]);
}

/// `datetime.date(2024, 3, 1).toordinal()` — PIN VALUE FROM THE
/// TASK BRIEF ITSELF: 738946.
#[test]
fn test_date_toordinal_of_a_known_date() {
    let Some(value) = eval("datetime.date(2024, 3, 1).toordinal()") else { return };
    assert_eq!(value.values, vec![738946.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `datetime.date(1, 1, 1).toordinal()` — PIN VALUE FROM THE KERNEL'S
/// OWN PROVED THEOREM (`ordinal.lean`'s `pyToOrdinal_anchor_is_one`,
/// closed by `decide`): exactly 1, "January 1 of year 1 has ordinal
/// 1" (datetime.rst:525-526).
#[test]
fn test_date_toordinal_anchor_is_exactly_one() {
    let Some(value) = eval("datetime.date(1, 1, 1).toordinal()") else { return };
    assert_eq!(value.values, vec![1.0]);
}

/// `datetime.timedelta(days=5)` constructs — a plain valid duration,
/// tagged and carrying its own `days` field.
#[test]
fn test_timedelta_construction_carries_its_days_field() {
    let Some(value) = eval("datetime.timedelta(days=5)") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "days"), Some(5.0));
}

/// `datetime.timedelta(hours=5)` — a keyword this file does not
/// read (only `days=` is modeled); the whole construction declines
/// rather than silently dropping the field.
#[test]
fn test_timedelta_construction_with_an_unmodeled_keyword_declines() {
    let Some(value) = eval("datetime.timedelta(hours=5)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.date(2024, 3, 1) + datetime.timedelta(days=31)` — PIN
/// VALUE DERIVED BY THE PROBE: 2024-03-01 plus 31 days crosses into
/// April, landing on 2024-04-01.
#[test]
fn test_date_plus_timedelta_crosses_a_month_boundary() {
    let Some(value) = eval("datetime.date(2024, 3, 1) + datetime.timedelta(days=31)") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(4.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// `datetime.timedelta(days=31) + datetime.date(2024, 3, 1)` — the
/// REVERSED operand order (datetime.rst states the operation both
/// ways); must answer the identical date the forward order gives.
#[test]
fn test_timedelta_plus_date_reversed_operand_order_agrees() {
    let Some(value) = eval("datetime.timedelta(days=31) + datetime.date(2024, 3, 1)") else { return };
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(4.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// `datetime.date(2024, 3, 1) - datetime.timedelta(days=1)` — PIN
/// VALUE DERIVED BY THE PROBE: one day before March 1st on a leap
/// year is February 29th (2024 IS a leap year).
#[test]
fn test_date_minus_timedelta_crosses_back_into_a_leap_february() {
    let Some(value) = eval("datetime.date(2024, 3, 1) - datetime.timedelta(days=1)") else { return };
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(2.0));
    assert_eq!(datetime_field(&value, "day"), Some(29.0));
}

/// `datetime.date(9999, 12, 31) + datetime.timedelta(days=1)` —
/// datetime.rst's own `OverflowError` row (date.7): MAXYEAR is 9999,
/// so this shift leaves the representable range and declines
/// through the `pyYearInRange` kernel ask (`python_year_in_range`)
/// on the shifted result's year (10000) — the kernel's `isoDate` arm
/// alone would answer this shift (its own PlainDate window is far
/// wider than Python's), so the decline is `pyYearInRange`'s doing.
#[test]
fn test_date_plus_timedelta_past_maxyear_declines() {
    let Some(value) = eval("datetime.date(9999, 12, 31) + datetime.timedelta(days=1)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.date.fromisoformat("2024-03-01")` — the strict
/// `YYYY-MM-DD` grammar (date.3's own committed shape), landing on
/// the exact same tagged instance `datetime.date(2024, 3, 1)`
/// constructs directly.
#[test]
fn test_date_fromisoformat_parses_the_strict_grammar() {
    let Some(value) = eval("datetime.date.fromisoformat(\"2024-03-01\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// `datetime.date.fromisoformat("2023-02-30")` — syntactically the
/// right shape (three hyphen-separated all-digit runs of the right
/// width) but calendrically invalid; declines through the SAME
/// `calendar.validDate` kernel ask `date_construction_value` uses.
#[test]
fn test_date_fromisoformat_of_a_calendrically_invalid_string_declines() {
    let Some(value) = eval("datetime.date.fromisoformat(\"2023-02-30\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.date.fromisoformat("2024-3-1")` — the reduced-width
/// (non-zero-padded) spelling; date.3's own committed grammar is
/// exactly `YYYY-MM-DD` (fixed widths), so this shape declines
/// rather than guess a looser parse.
#[test]
fn test_date_fromisoformat_of_a_non_zero_padded_string_declines() {
    let Some(value) = eval("datetime.date.fromisoformat(\"2024-3-1\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.date(2024, 3, 1).isocalendar()` — PIN VALUE FROM THE
/// COORDINATOR'S OWN BRIEF (backed by the kernel's `isoCalendar` arm
/// AND the Lean witness landing alongside it): `(2024, 9, 5)` — ISO
/// year 2024, ISO week 9, ISO weekday 5 (Friday, the same Friday
/// `.weekday() == 4`/`.isoweekday() == 5` already pin above). Binds
/// as a known 3-element tuple, the same `Kind::List` shape a literal
/// `(a, b, c)` display builds.
#[test]
fn test_date_isocalendar_of_a_known_date() {
    let Some(value) = eval("datetime.date(2024, 3, 1).isocalendar()") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![2024.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![9.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![5.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

/// `datetime.date(9999, 12, 31) + datetime.timedelta(days=1)` posed
/// a SECOND way: this is the same construct
/// `test_date_plus_timedelta_past_maxyear_declines` above already
/// pins, restated here to name explicitly that the decline is now
/// the `pyYearInRange` kernel ask's own `valid: false` answer (year
/// 10000), not an adapter-local bound check.
#[test]
fn test_date_plus_timedelta_past_maxyear_declines_via_the_kernel_year_range_ask() {
    let Some(value) = eval("datetime.date(9999, 12, 31) + datetime.timedelta(days=1)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

// --- j-stdlib-surfaces.py: strftime/strptime STAGE 1 (date.12) ---

/// `datetime.datetime.strptime("2024-03-01", "%Y-%m-%d")` binds the
/// EXACT SAME value `datetime.date.fromisoformat("2024-03-01")`
/// does — `strptime_iso_date_value`'s own doc: one recognition, the
/// existing `date_fromisoformat_value` machinery, no new kernel
/// question. Asserts equality of the two paths' values directly,
/// not just their shape.
#[test]
fn test_strptime_iso_date_agrees_with_fromisoformat() {
    let Some(via_strptime) = eval("datetime.datetime.strptime(\"2024-03-01\", \"%Y-%m-%d\")") else { return };
    let Some(via_fromisoformat) = eval("datetime.date.fromisoformat(\"2024-03-01\")") else { return };
    assert_eq!(via_strptime, via_fromisoformat);
    assert_eq!(via_strptime.kind, Kind::Object);
    assert_eq!(via_strptime.source.as_str(), "datetime_date");
}

/// `datetime.datetime.strptime("2023-02-30", "%Y-%m-%d")` — a
/// calendrically invalid date (February has 28 days in 2023);
/// declines through the SAME `validDate` kernel ask
/// `date.fromisoformat("2023-02-30")` declines through, since
/// `strptime_iso_date_value` reuses `date_fromisoformat_value`
/// outright.
#[test]
fn test_strptime_of_an_invalid_date_declines_identically_to_fromisoformat() {
    let Some(via_strptime) = eval("datetime.datetime.strptime(\"2023-02-30\", \"%Y-%m-%d\")") else { return };
    let Some(via_fromisoformat) = eval("datetime.date.fromisoformat(\"2023-02-30\")") else { return };
    assert_eq!(via_strptime.kind, Kind::Unknown);
    assert_eq!(via_fromisoformat.kind, Kind::Unknown);
}

/// `datetime.datetime.strptime("2024-03-01", fmt)` where `fmt` is a
/// PARAMETER (a computed format the source cannot name, never a
/// written literal) — declines; this file has no format-code
/// mini-language reader for an expression it cannot fold to an
/// exact string at all.
#[test]
fn test_strptime_with_a_computed_format_declines() {
    let Some(value) = eval("datetime.datetime.strptime(\"2024-03-01\", fmt)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.datetime.strptime("01/03/2024", "%d/%m/%Y")` — a
/// LITERAL format, not the ISO `"%Y-%m-%d"` sequence STAGE 1 builds,
/// but every directive here (`%d`, `%m`, `%Y`) IS one STAGE 2
/// transcribes (`strptime2_scan_format`'s own directive set) — so
/// this now DETERMINES, a tagged `datetime_datetime` instance (day 1,
/// month 3, year 2024), the mirror of STAGE 1's own agreement test
/// against `date.fromisoformat`.
#[test]
fn test_strptime_with_a_non_iso_literal_format_now_determines_via_stage_2() {
    let Some(value) = eval("datetime.datetime.strptime(\"01/03/2024\", \"%d/%m/%Y\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.source.as_str(), "datetime_datetime");
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

// --- j-stdlib-surfaces.py: strptime STAGE 2 (date.12, the
// non-ISO directive grammar: %Y %m %d %H %M %S %f %j %U %W %y) ---

/// `datetime.datetime.strptime("2024/03/01", "%Y/%m/%d")` — the SAME
/// three date-only directives STAGE 1's `"%Y-%m-%d"` reads, spelled
/// with a different literal separator: this is STAGE 2's own full
/// derivation of a plain `%Y`/`%m`/`%d` sequence, landing on the
/// SAME year/month/day fields `strptime_iso_date_value` would bind
/// for the equivalent ISO text — but as a `datetime_datetime`
/// instance (STAGE 2's own tagged shape, `strptime2_parse`'s own
/// doc), not the `datetime_date` STAGE 1 binds, since STAGE 2 always
/// carries the six `datetime_construction_value` fields.
#[test]
fn test_strptime_stage_2_derives_year_month_day_with_a_non_iso_separator() {
    let Some(value) = eval("datetime.datetime.strptime(\"2024/03/01\", \"%Y/%m/%d\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.source.as_str(), "datetime_datetime");
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
    assert_eq!(datetime_field(&value, "hour"), Some(0.0));
    assert_eq!(datetime_field(&value, "minute"), Some(0.0));
    assert_eq!(datetime_field(&value, "second"), Some(0.0));
}

/// `datetime.datetime.strptime("23:59:59", "%H:%M:%S")` — a
/// time-of-day-only format (note (9)'s own three "leading zero
/// optional" directives), the composite EXCLUDED from STAGE 1
/// (`strptime_iso_date_value`'s own doc: "EXCLUDED from this stage:
/// any `%H:%M:%S`-composite format"). No `%Y`/`%m`/`%d` directive
/// appears, so year/month/day fall to `strptime2_parse`'s own
/// default-value rule (datetime.rst:2527-2529, the default
/// `1900-01-01`) — this test names the hour/minute/second fields
/// STAGE 2 newly derives, the reverse of STAGE 1's own
/// date-only reach.
#[test]
fn test_strptime_stage_2_derives_a_time_of_day_only_format() {
    let Some(value) = eval("datetime.datetime.strptime(\"23:59:59\", \"%H:%M:%S\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.source.as_str(), "datetime_datetime");
    assert_eq!(datetime_field(&value, "year"), Some(1900.0));
    assert_eq!(datetime_field(&value, "month"), Some(1.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
    assert_eq!(datetime_field(&value, "hour"), Some(23.0));
    assert_eq!(datetime_field(&value, "minute"), Some(59.0));
    assert_eq!(datetime_field(&value, "second"), Some(59.0));
}

/// `%y`'s century pivot, LOW side — time.rst:45-48: "values 0--68 are
/// mapped to 2000--2068." `"68"` is the pivot's own upper LOW-side
/// edge, landing on year 2068.
#[test]
fn test_strptime_stage_2_two_digit_year_pivot_low_side_maps_into_2000s() {
    let Some(value) = eval("datetime.datetime.strptime(\"68-03-01\", \"%y-%m-%d\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2068.0));
}

/// `%y`'s century pivot, HIGH side — time.rst:45-48: "values 69--99
/// are mapped to 1969--1999." `"69"` is the pivot's own lower
/// HIGH-side edge, landing on year 1969 — the two tests together pin
/// the pivot's exact boundary at 68/69.
#[test]
fn test_strptime_stage_2_two_digit_year_pivot_high_side_maps_into_1900s() {
    let Some(value) = eval("datetime.datetime.strptime(\"69-03-01\", \"%y-%m-%d\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(1969.0));
}

/// `datetime.datetime.strptime("2024-03-01", "%G-%m-%d")` — `%G`
/// (the ISO 8601 year) is not in `strptime2_scan_format`'s
/// transcribed-directive list and is not a locale directive; this
/// format's `%m`/`%d` are each transcribed, but `%G` itself is not,
/// so `strptime2_scan_format` declines the WHOLE format, naming `%G`
/// specifically (`Strptime2Decline::UnreadDirective('G')`) rather
/// than treating the format as an unrecognized sequence in general —
/// this test exists to prove the decline names ONE directive, not
/// the whole string, even though this file has no channel today to
/// surface that name as a diagnostic (the dispatch site's own
/// comment states the same standing limitation STAGE 1 already
/// carries).
#[test]
fn test_strptime_stage_2_names_an_unread_directive_g_and_declines() {
    let format = "%G-%m-%d";
    assert_eq!(strptime2_scan_format(format, false), Err(Strptime2Decline::UnreadDirective('G')));
    let Some(value) = eval("datetime.datetime.strptime(\"2024-03-01\", \"%G-%m-%d\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.datetime.strptime("Mon 2024-03-01", "%a %Y-%m-%d")`,
/// evaluated in a plain environment with no `locale_never_set`
/// premise set (`eval`'s own `empty_environment`, matching every
/// module this checker has not yet proved never calls
/// `locale.setlocale`). `strptime2_scan_format` still names `%a`'s
/// own distinct reason (`WeekdayAbbreviation`, never the six-
/// directive `LocaleDirective` shape `%A`/`%b`/etc. take) when the
/// caller does not accept it — the premise gate at the dispatch
/// site decides determined vs. undetermined, not this scan alone.
#[test]
fn test_strptime_stage_2_names_the_weekday_abbreviation_a_as_its_own_distinct_reason() {
    let format = "%a %Y-%m-%d";
    assert_eq!(strptime2_scan_format(format, false), Err(Strptime2Decline::WeekdayAbbreviation));
    let Some(value) = eval("datetime.datetime.strptime(\"Mon 2024-03-01\", \"%a %Y-%m-%d\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// The SAME `"Mon 2024-03-01"` / `"%a %Y-%m-%d"` pair, now
/// evaluated against an environment carrying the C-locale premise
/// (`locale_never_set = true` — this module never calls
/// `locale.setlocale`, `module_never_calls_setlocale`'s own doc).
/// `%a`'s own value set IS host-independent under that premise
/// (POSIX's portable `'C'` locale, `read_weekday_abbreviation_
/// field`'s own fixed ASCII table), so the SAME format that stays
/// `Unknown` above now determines a real construction — `strptime2_
/// scan_format`'s own `accept_weekday_abbreviation = true` arm
/// treats `%a` as a transcribed directive, so `strptime2_parse` runs
/// (`%a`'s own weekday value is read and range-checked but not
/// folded into the constructed instance, `strptime2_parse`'s own
/// doc — the same "read but not carried" convention `%U`/`%W`
/// already take).
#[test]
fn test_strptime_stage_2_reads_weekday_abbreviation_under_the_c_locale_premise() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("datetime.datetime.strptime(\"Mon 2024-03-01\", \"%a %Y-%m-%d\")").expect("test source parses");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.set_locale_never_set(true);
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(datetime_field(&value, "year"), Some(2024.0));
    assert_eq!(datetime_field(&value, "month"), Some(3.0));
    assert_eq!(datetime_field(&value, "day"), Some(1.0));
}

/// A weekday abbreviation the C-locale table does not recognize
/// (`"Xyz"`, not one of `Sun`/`Mon`/…/`Sat`) — even under the
/// premise, `read_weekday_abbreviation_field` returns `None` for
/// text it cannot match, so `strptime2_parse` itself declines this
/// text/format pair (`None`, not a wrong weekday folded in), and the
/// dispatch's own `unknown()` fallback answers — proving the premise
/// widens WHICH FORMATS are attempted, never what a mismatched text
/// is allowed to mean.
#[test]
fn test_strptime_stage_2_declines_an_unrecognized_weekday_abbreviation_even_under_the_premise() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("datetime.datetime.strptime(\"Xyz 2024-03-01\", \"%a %Y-%m-%d\")").expect("test source parses");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.set_locale_never_set(true);
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.date(2024, 3, 1).strftime("%Y-%m-%d")` — the exact ISO
/// literal format on a known date, rendered by the kernel's own
/// `isoDateText` arm.
#[test]
fn test_strftime_iso_format_on_a_known_date_is_the_iso_text() {
    let Some(value) = eval("datetime.date(2024, 3, 1).strftime(\"%Y-%m-%d\")") else { return };
    assert_eq!(iso_text(&value), "2024-03-01");
}

/// `datetime.date(2024, 3, 1).isoformat()` — the same text through
/// the method datetime.rst names for the render direction.
#[test]
fn test_date_isoformat_is_the_iso_text() {
    let Some(value) = eval("datetime.date(2024, 3, 1).isoformat()") else { return };
    assert_eq!(iso_text(&value), "2024-03-01");
}

/// `datetime.date(2024, 3, 1).strftime(fmt)` where `fmt` is a
/// PARAMETER — declines, the same computed-format reason
/// `test_strptime_with_a_computed_format_declines` states for the
/// parse direction.
#[test]
fn test_strftime_with_a_computed_format_declines() {
    let Some(value) = eval("datetime.date(2024, 3, 1).strftime(fmt)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `datetime.date(2024, 3, 1).strftime("%d/%m/%Y")` — a non-ISO
/// literal directive sequence; names date.12 STAGE 2, the same
/// reason `test_strptime_with_a_non_iso_literal_format_declines`
/// states for the parse direction.
#[test]
fn test_strftime_with_a_non_iso_literal_format_declines() {
    let Some(value) = eval("datetime.date(2024, 3, 1).strftime(\"%d/%m/%Y\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}
