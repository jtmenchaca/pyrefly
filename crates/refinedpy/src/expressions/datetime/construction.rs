//! `datetime`/`date`/`timedelta` import identity, construction, and
//! their shared field/tzinfo/validity readers.

use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::CalendarQuestion;
use refined_kernel::kernel_interface::CalendarQuestionOp;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::Visitor;

use crate::env::Environment;

use super::super::evaluate_expression;
use super::super::arithmetic::*;
use super::super::call::*;

/// Which datetime construct a local name means, read once from the
/// module's own `import`/`from … import …` statements — the same
/// "one import table, read once" mechanism `surface::SurfaceImports`
/// already carries for the pydantic surface (`surface_imports`'s own
/// doc), scoped here to the `datetime` module family so this file's
/// gates answer by CANONICAL identity rather than the literal spelling
/// `datetime`/`date`/`timedelta`. Three shapes recognize:
/// `import datetime[ as x]` (`module_names`, `x` means the WHOLE
/// module — `x.datetime`/`x.date`/`x.timedelta` all still resolve
/// through it), `from datetime import datetime[ as x]`/`date[ as
/// x]`/`timedelta[ as x]` (each lands in its own class-name set, `x`
/// alone now means that ONE class, no further attribute needed), and
/// no import at all (every set stays empty, and `datetime_imports`'s
/// caller falls back to the literal `datetime.*` spelling unchanged —
/// datetime.rst's classes are named `datetime`/`date`/`timedelta`
/// either way, so a module with no explicit `datetime` import still
/// reads its bare `datetime.date(...)` calls the same as before this
/// table existed).
#[derive(Default)]
pub struct DatetimeImports {
    module_names: HashSet<String>,
    datetime_class_names: HashSet<String>,
    date_class_names: HashSet<String>,
    timedelta_class_names: HashSet<String>,
}

/// Reads `module`'s top-level `import`/`from … import …` statements
/// into a `DatetimeImports` table (see that struct's own doc for the
/// three recognized shapes). Anything else — a re-export, a
/// submodule import (`import datetime.date`, not a real Python
/// shape for this stdlib module anyway), a star import — is out of
/// scope and leaves the corresponding set empty, the same "recognize
/// only the shapes the mission names" discipline `surface_imports`
/// already keeps.
pub(crate) fn datetime_imports(module: &ModModule) -> DatetimeImports {
    let mut table = DatetimeImports::default();
    for stmt in module.body.iter() {
        match stmt {
            Stmt::Import(import) => {
                for alias in &import.names {
                    if alias.name.id.as_str() == "datetime" {
                        let local = alias.asname.as_ref().unwrap_or(&alias.name);
                        table.module_names.insert(local.id.as_str().to_owned());
                    }
                }
            }
            Stmt::ImportFrom(import) => {
                let Some(source) = import.module.as_ref() else {
                    continue;
                };
                if source.id.as_str() != "datetime" || import.level != 0 {
                    continue;
                }
                for alias in &import.names {
                    let local = alias.asname.as_ref().unwrap_or(&alias.name);
                    match alias.name.id.as_str() {
                        "datetime" => {
                            table.datetime_class_names.insert(local.id.as_str().to_owned());
                        }
                        "date" => {
                            table.date_class_names.insert(local.id.as_str().to_owned());
                        }
                        "timedelta" => {
                            table.timedelta_class_names.insert(local.id.as_str().to_owned());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    table
}

/// The `Visitor` `module_never_calls_setlocale` drives — records
/// whether ANY `locale.setlocale(...)` call appears anywhere in the
/// module (a top-level statement, or nested inside any function/
/// method/class body: `setlocale` can be called from anywhere, unlike
/// `datetime_imports`'s own top-level-only import statements, so this
/// walk must descend into every body the way `CallSiteCollector`
/// (function_table.rs) already does, rather than iterating
/// `module.body` directly). Recognizes both the qualified
/// `locale.setlocale(...)` call and a bare `setlocale(...)` call
/// reached through `from locale import setlocale` — the same
/// no-import-identity-table convention `is_utc_tzinfo_expression`
/// already takes (matched by literal callee spelling, since this
/// premise does not need a full import-identity table the way
/// `DatetimeImports` does: a module that merely SHADOWS the name
/// with an unrelated `setlocale` function would be a vanishingly
/// rare false decline, never a false "safe to assume C locale").
pub(in crate::expressions) struct SetlocaleCallFinder {
    found: bool,
}

impl<'a> Visitor<'a> for SetlocaleCallFinder {
    fn visit_expr(&mut self, expr: &'a Expr) {
        if self.found {
            return;
        }
        if let Expr::Call(call) = expr {
            let callee_name = match call.func.as_ref() {
                Expr::Name(name) => Some(name.id.as_str()),
                Expr::Attribute(attribute) => Some(attribute.attr.as_str()),
                _ => None,
            };
            if callee_name == Some("setlocale") {
                self.found = true;
                return;
            }
        }
        walk_expr(self, expr);
    }
}

/// Whether `module` never calls `locale.setlocale` anywhere in its own
/// source — the premise `%a`'s C-locale weekday-abbreviation reading
/// needs (locale.rst:326-327: "a program which has not called
/// `setlocale(LC_ALL, '')` runs using the portable `'C'` locale",
/// whose weekday abbreviations are the fixed ASCII set
/// `read_weekday_abbreviation_field` already reads). Built once per
/// module (`DatetimeImports`'s own "one table, read once" pattern),
/// riding `Environment` the same way (`env.rs`'s
/// `set_locale_never_set`/`locale_never_set`, `check.rs`'s
/// module-setup site). A module this walk cannot fully account for
/// (none today — every statement/expression shape `SetlocaleCallFinder`
/// does not specifically recognize still walks through the ordinary
/// `walk_expr` default) is not a concern here the way it is for a
/// value-carrying reader: a `visit_expr` override that returns `Some`
/// only for recognized call shapes still descends into every OTHER
/// expression through the default walk, so a `setlocale` call nested
/// arbitrarily deep (inside a lambda, a comprehension, a nested `def`)
/// is still found.
pub(crate) fn module_never_calls_setlocale(module: &ModModule) -> bool {
    let mut finder = SetlocaleCallFinder { found: false };
    for stmt in &module.body {
        finder.visit_stmt(stmt);
        if finder.found {
            break;
        }
    }
    !finder.found
}

/// Whether `callee` names the `datetime.datetime` class, NOT locally
/// shadowed — resolved by CANONICAL import identity through
/// `environment`'s own `DatetimeImports` table (`datetime_imports`'s
/// own doc) rather than the literal spelling. `callee` is the exact
/// expression a caller wants to prove IS the `datetime.datetime`
/// class — either the CONSTRUCTION call's own callee
/// (`datetime.datetime(...)`) or a classmethod call's own RECEIVER
/// (`datetime.datetime.now()`'s `datetime.datetime`). Two shapes
/// recognize: the qualified attribute chain `datetime.datetime`/
/// `dtm.datetime` (any local name the table's `module_names`
/// resolved to the whole module), and the bare aliased class name
/// (`dt`, from `from datetime import datetime as dt` — the table's
/// `datetime_class_names`). A module with no `DatetimeImports` table
/// at all (`environment.datetime_imports()` answers `None` — a test
/// environment, or a walk that never set one) falls back to the
/// literal `datetime.datetime` spelling only for the qualified shape,
/// and never recognizes a bare name — matching this function's own
/// behavior before the table existed. Shadowing is checked the same
/// way either shape already did: the resolved base name must read
/// `None` from `environment`'s own bindings — a body that locally
/// rebinds `datetime`/`dtm`/`dt` to some other value shadows the
/// import regardless of which spelling reached it.
pub(in crate::expressions) fn is_datetime_datetime_attribute(callee: &Expr, environment: &Environment) -> bool {
    if let Expr::Attribute(attribute) = callee {
        if attribute.attr.as_str() == "datetime" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if environment.read(module_name.id.as_str()).is_some() {
                    return false;
                }
                if let Some(imports) = environment.datetime_imports() {
                    return imports.module_names.contains(module_name.id.as_str());
                }
                return module_name.id.as_str() == "datetime";
            }
        }
        return false;
    }
    let Expr::Name(name) = callee else {
        return false;
    };
    let Some(imports) = environment.datetime_imports() else {
        return false;
    };
    imports.datetime_class_names.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none()
}

/// The four tzinfo shapes `datetime_construction_value` distinguishes
/// — datetime.rst, `class:: datetime(..., tzinfo=None, ...)`: `Naive`
/// (no `tzinfo=`, "a naive object does not contain enough information
/// to unambiguously locate itself"), `Utc` (`tzinfo=` reads exactly
/// `datetime.timezone.utc`/`datetime.UTC`, `is_utc_tzinfo_expression`'s
/// own recognition), `FixedOffset(seconds)` (`tzinfo=timezone(timedelta
/// (hours=…))` — datetime.rst, `class:: timezone(offset, name=None)`,
/// "offset ... representing the difference between the local time and
/// UTC" — read SYNTACTICALLY off the `timedelta(...)` argument's own
/// literal `hours=`/`minutes=` fields, the SAME no-import-identity
/// convention `is_utc_tzinfo_expression` already takes for `timezone`
/// itself), and `OtherAware` (`tzinfo=` reads a recognized tzinfo
/// CONSTRUCTOR this file cannot resolve to an exact offset — today only
/// `zoneinfo.ZoneInfo(...)`). `Utc` and `FixedOffset` both carry an
/// EXACTLY known offset, so the instant is provable to the microsecond
/// either way; `OtherAware`'s own "aware" definition needs only a
/// non-None `tzinfo`, not a resolvable offset, so `AwareDatetime`'s own
/// admission test (assignability.rs's temporal arm) reads it as aware
/// while `bounds_verdict_of`'s own exact-instant comparison still
/// cannot be proved for it — `Unprovable`, never a guess.
#[derive(Clone, PartialEq, Eq)]
pub(in crate::expressions) enum TzinfoKind {
    Naive,
    Utc,
    FixedOffset(i64),
    /// `zoneinfo.ZoneInfo("<Area>/<Location>")` with a string-literal
    /// zone name this file cannot resolve to an offset SYNTACTICALLY —
    /// `datetime_construction_value` resolves it once the wall-clock
    /// fields are known, reading the system's own tzdata
    /// (`tzif::utc_offset_seconds_for_wall_time`). Kept distinct from
    /// `OtherAware` so a `ZoneInfo(...)` call with a NON-literal or
    /// unrecognized argument (this variant's own construction never
    /// applies) still falls back to `OtherAware`'s unresolved reading.
    ZoneName(String),
    OtherAware,
}

/// `datetime.datetime(year, month, day, hour=0, minute=0, second=0,
/// microsecond=0, tzinfo=...)` — a tagged `Kind::Object` (`source =
/// "datetime_datetime"`) carrying `year`/`month`/`day`/`hour`/
/// `minute`/`second`/`microsecond` as Integer `ObjectKey`s, PLUS an
/// `aware_utc` marker (a Boolean `ObjectKey`, kept for
/// `datetime_timestamp_value`'s own existing reader) — datetime.rst,
/// `class:: datetime(year, month, day, hour=0, minute=0, second=0,
/// microsecond=0, tzinfo=None, *, fold=0)`. Modeled ONLY when every
/// positional/keyword argument this file reads is a known Integer
/// literal (year/month/day always positional in this corpus;
/// hour/minute/second/microsecond read from EITHER a positional slot
/// or a keyword, defaulting to 0 when absent, matching the
/// constructor's own defaults) — a `fold` argument, or ANY argument
/// this file cannot read as a known Integer, declines the WHOLE
/// construction (never a partially-built datetime). `tzinfo=` is read
/// SYNTACTICALLY (`TzinfoKind`'s own doc); the whole construction
/// declines for any OTHER `tzinfo=` expression this reader does not
/// recognize as one of the three kinds.
///
/// `instance.temporal` carries the construction's own ISO spelling on
/// the `Instant` chart — `"YYYY-MM-DDTHH:MM:SS[.ffffff]Z"` for a
/// UTC-aware construction (the exact offset lets the microsecond
/// fraction ride the ISO text `duration`/`compare_on_chart`'s own
/// fractional-second grammar already reads), the offset-free
/// `"YYYY-MM-DDTHH:MM:SS[.ffffff]"` for a NAIVE construction (still
/// spelled, so a naive-vs-AwareDatetime refusal is decided from the
/// construction's own fields rather than falling through undetermined)
/// — `None` for `OtherAware` (no exact offset this file can spell,
/// `chart_reading`'s own `Instant` arm would read it as `Unprovable`
/// regardless, so the wasted ask is skipped).
pub(in crate::expressions) fn datetime_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let positional_names = ["year", "month", "day", "hour", "minute", "second", "microsecond"];
    let mut fields: Vec<Option<i64>> = vec![None; positional_names.len()];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        let slot = fields.get_mut(index)?;
        *slot = Some(datetime_field_argument(arg, environment, kernel)?);
    }
    let mut tzinfo_kind = TzinfoKind::Naive;
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            return None;
        };
        if arg_name.as_str() == "tzinfo" {
            tzinfo_kind = classify_tzinfo_expression(&keyword.value, environment, kernel)?;
            continue;
        }
        let Some(position) = positional_names.iter().position(|name| *name == arg_name.as_str()) else {
            // `fold=` (or any other keyword) — not modeled, decline the
            // whole construction
            return None;
        };
        let slot = fields.get_mut(position)?;
        *slot = Some(datetime_field_argument(&keyword.value, environment, kernel)?);
    }
    // year/month/day have no default (positional-required per the
    // constructor's own signature); hour/minute/second/microsecond
    // default to 0
    let mut keys = Vec::with_capacity(positional_names.len() + 1);
    let mut resolved = [0i64; 7];
    for (index, name) in positional_names.iter().enumerate() {
        let value = match fields[index] {
            Some(value) => value,
            None if index < 3 => return None,
            None => 0,
        };
        resolved[index] = value;
        keys.push(integer_object_key(name, value));
    }
    // A `ZoneInfo("<zone>")` tzinfo resolves to an exact offset HERE,
    // now that the wall-clock fields (`resolved`) are known — a
    // literal instant in a literal zone name has a tzdata-determined
    // offset (`tzif::utc_offset_seconds_for_wall_time`, reading the
    // system's own compiled zoneinfo). Once resolved it behaves
    // exactly like `TzinfoKind::FixedOffset` below (an EXACTLY known
    // offset, `aware_tag = 1`, an ISO-suffixed `instance.temporal`);
    // an unresolvable zone name (unknown zone, a wall time tzdata's
    // own transition table cannot settle) falls back to `OtherAware`
    // — the same unresolved reading `ZoneInfo(...)` always gave before
    // this reader existed, never a guess.
    if let TzinfoKind::ZoneName(zone_name) = &tzinfo_kind {
        let [year, month, day, hour, minute, second, _microsecond] = resolved;
        let epoch_seconds_as_utc = crate::tzif::wall_clock_epoch_seconds(year, month, day, hour, minute, second);
        tzinfo_kind = match crate::tzif::utc_offset_seconds_for_wall_time(zone_name, epoch_seconds_as_utc) {
            Some(offset_seconds) => TzinfoKind::FixedOffset(offset_seconds),
            None => TzinfoKind::OtherAware,
        };
    }
    keys.push(ObjectKey {
        name: "aware_utc".to_owned(),
        numeric: false,
        value: known_values(vec![if tzinfo_kind == TzinfoKind::Utc { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved),
    });
    // `aware`: 0 = naive, 1 = aware with an EXACTLY known offset (UTC or
    // a fixed `timezone(timedelta(...))` offset), 2 = aware with an
    // UNRESOLVED exact offset (`TzinfoKind::OtherAware`) —
    // assignability.rs's own temporal admission law reads this to
    // decide `AwareDatetime`/`NaiveDatetime`'s designated-fire rule
    // without re-parsing `instance.temporal`'s ISO text (which is
    // `None` for `OtherAware` and offset-ambiguous between `Naive`/
    // `Utc`/`FixedOffset` on its own).
    let aware_tag = match &tzinfo_kind {
        TzinfoKind::Naive => 0,
        TzinfoKind::Utc | TzinfoKind::FixedOffset(_) => 1,
        TzinfoKind::OtherAware => 2,
        // resolved to `FixedOffset`/`OtherAware` above — never reaches
        // here still holding a zone name
        TzinfoKind::ZoneName(_) => unreachable!("ZoneName is resolved to FixedOffset/OtherAware before this match"),
    };
    keys.push(integer_object_key("aware", aware_tag));
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_datetime".to_owned();
    if tzinfo_kind != TzinfoKind::OtherAware {
        let [year, month, day, hour, minute, second, microsecond] = resolved;
        let zone = match &tzinfo_kind {
            TzinfoKind::Utc => "Z".to_owned(),
            TzinfoKind::FixedOffset(seconds) => offset_iso_suffix(*seconds),
            _ => String::new(),
        };
        let point = if microsecond == 0 {
            format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{zone}")
        } else {
            format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{microsecond:06}{zone}")
        };
        instance.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
            chart: refined_sets::calendar_interpreter::TemporalChart::Instant,
            min: Some(point.clone()),
            max: Some(point),
        }));
    }
    Some(instance)
}

/// `datetime.datetime.fromtimestamp(timestamp, tz=...)` — datetime.rst,
/// `classmethod:: datetime.fromtimestamp(timestamp, tz=None)`: "Return
/// the local date and time corresponding to the POSIX timestamp... If
/// optional argument `tz` is specified... the timestamp is converted to
/// `tz`'s time zone."
///
/// Modeled ONLY for a `tz=` naming an EXACTLY known offset — UTC
/// (`TzinfoKind::Utc`) or a fixed `timezone(timedelta(...))` offset
/// (`TzinfoKind::FixedOffset`). The NAIVE form (`tz` absent or `None`)
/// declines: datetime.rst states it converts to LOCAL time, a host- and
/// environment-dependent conversion this crate does not claim to
/// reproduce — the same reason `datetime_timestamp_value` declines the
/// naive direction. An `OtherAware`/`ZoneName` `tz` declines too: the
/// wall-clock fields cannot be settled without an offset, and
/// `ZoneName`'s own tzdata resolution runs the other way round (it needs
/// the wall clock to pick the offset).
///
/// `timestamp` must be an exactly known whole number of seconds. A
/// fractional timestamp declines rather than round: the microsecond
/// field would be a binary-float remainder, and CPython's own note says
/// `fromtimestamp` "may have microsecond... subject to the platform's
/// floating point rounding".
///
/// The wall clock is derived by shifting the instant into the target
/// offset and splitting: the DAY count goes to the kernel's `isoDate` op
/// (the same self-certifying arm `date_shifted_by_timedelta` uses) for
/// the calendar fields, and the within-day remainder is the clock. The
/// result is the SAME tagged instance `datetime_construction_value`
/// builds — same keys, same `aware`/`aware_utc` markers, same ISO
/// `instance.temporal` — so every downstream reader (`.isoformat()`,
/// `.timestamp()`, the temporal admission law) treats it identically to
/// a literal construction.
pub(in crate::expressions) fn datetime_fromtimestamp_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let [timestamp_arg] = call.arguments.args.as_ref() else {
        return None;
    };
    let timestamp_value = evaluate_expression(timestamp_arg, environment, kernel);
    let (timestamp, _) = single_numeric_value(&timestamp_value)?;
    if timestamp.fract() != 0.0 || !timestamp.is_finite() {
        return None;
    }
    let epoch_seconds = timestamp as i64;
    let mut tzinfo_kind = None;
    for keyword in &call.arguments.keywords {
        let arg_name = keyword.arg.as_ref()?;
        if arg_name.as_str() != "tz" {
            return None;
        }
        tzinfo_kind = Some(classify_tzinfo_expression(&keyword.value, environment, kernel)?);
    }
    let offset_seconds = match tzinfo_kind {
        Some(TzinfoKind::Utc) => 0,
        Some(TzinfoKind::FixedOffset(seconds)) => seconds,
        // naive (no `tz=`), or an offset this crate never resolved
        _ => return None,
    };
    let local_seconds = epoch_seconds.checked_add(offset_seconds)?;
    // Floor division, so an instant before the epoch still lands on the
    // day that CONTAINS it with a nonnegative within-day remainder.
    let days = local_seconds.div_euclid(86400);
    let seconds_of_day = local_seconds.rem_euclid(86400);
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
    let hour = seconds_of_day / 3600;
    let minute = (seconds_of_day % 3600) / 60;
    let second = seconds_of_day % 60;
    let is_utc = offset_seconds == 0 && tzinfo_kind == Some(TzinfoKind::Utc);
    let mut keys = vec![
        integer_object_key("year", year),
        integer_object_key("month", month),
        integer_object_key("day", day),
        integer_object_key("hour", hour),
        integer_object_key("minute", minute),
        integer_object_key("second", second),
        integer_object_key("microsecond", 0),
    ];
    keys.push(ObjectKey {
        name: "aware_utc".to_owned(),
        numeric: false,
        value: known_values(vec![if is_utc { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved),
    });
    keys.push(integer_object_key("aware", 1));
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_datetime".to_owned();
    let zone = if is_utc { "Z".to_owned() } else { offset_iso_suffix(offset_seconds) };
    let point = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{zone}");
    instance.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
        chart: refined_sets::calendar_interpreter::TemporalChart::Instant,
        min: Some(point.clone()),
        max: Some(point),
    }));
    Some(instance)
}

/// A whole-second UTC offset, spelled the ISO 8601 sign-hour-minute
/// suffix `calendar_interpreter.rs`'s own `read_offset` accepts
/// (`OFFSET_RE`: `^([+-])(\d{2})(?::?(\d{2})...)`) — `TzinfoKind::
/// FixedOffset`'s own ISO spelling. Seconds beyond a whole minute are
/// dropped (`timezone(timedelta(...))`'s own offset argument is always
/// a whole number of minutes in this crate's corpus; a sub-minute
/// remainder would need a third `:SS` segment this reader does not
/// build, since nothing in showcase.py or the pydantic surface needs
/// one).
pub(in crate::expressions) fn offset_iso_suffix(seconds: i64) -> String {
    let sign = if seconds < 0 { '-' } else { '+' };
    let magnitude = seconds.unsigned_abs();
    let hours = magnitude / 3600;
    let minutes = (magnitude % 3600) / 60;
    format!("{sign}{hours:02}:{minutes:02}")
}

/// `tzinfo=`'s own value expression, read SYNTACTICALLY as one of the
/// three `TzinfoKind`s — see that enum's own doc for the exact
/// recognized spellings. `None` for a `tzinfo=` expression this reader
/// recognizes as none of the three (a computed value, an unrecognized
/// constructor) — the whole construction declines rather than guess.
pub(in crate::expressions) fn classify_tzinfo_expression(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<TzinfoKind> {
    if is_utc_tzinfo_expression(expr) {
        return Some(TzinfoKind::Utc);
    }
    if let Expr::Call(call) = expr {
        let callee_name = match call.func.as_ref() {
            Expr::Name(name) => Some(name.id.as_str()),
            Expr::Attribute(attribute) => Some(attribute.attr.as_str()),
            _ => None,
        };
        // `timezone(timedelta(hours=…))` — read by bare callee name
        // only, the same no-import-identity convention this file's own
        // datetime gates take when no import table exists; the single
        // positional argument must itself be a recognized
        // `timedelta(...)` call this file can evaluate to a known
        // `datetime_timedelta`-tagged instance. Read here by
        // `offset_seconds_of_timedelta_call`, which takes the
        // `hours=`/`minutes=`/`seconds=` keywords straight off the AST:
        // a `timezone(...)` offset is a WHOLE-SECOND count, the one
        // shape `TzinfoKind::FixedOffset` carries, so this arm reads the
        // seconds directly rather than normalize through the duration
        // triple and read it back.
        if callee_name == Some("timezone") {
            if let [offset_arg] = call.arguments.args.as_ref() {
                if let Some(seconds) = offset_seconds_of_timedelta_call(offset_arg, environment, kernel) {
                    return Some(TzinfoKind::FixedOffset(seconds));
                }
            }
            return Some(TzinfoKind::OtherAware);
        }
        // `zoneinfo.ZoneInfo(...)` / a bare aliased `ZoneInfo(...)` —
        // read by bare callee name only, the same no-import-identity
        // convention already taken above; this file tracks no
        // `zoneinfo` import table (unlike `datetime`'s own
        // `DatetimeImports`), so the recognition is syntactic. A
        // single string-literal positional argument names the zone
        // (`ZoneInfo("Europe/Paris")`, the IANA key form the stdlib
        // documents) — `ZoneName` carries the key for
        // `datetime_construction_value` to resolve against tzdata
        // once the wall-clock fields are known; any other argument
        // shape (computed, keyword, multiple args) falls back to
        // `OtherAware`, unresolved.
        if callee_name == Some("ZoneInfo") {
            if let [Expr::StringLiteral(literal)] = call.arguments.args.as_ref() {
                return Some(TzinfoKind::ZoneName(literal.value.to_str().to_owned()));
            }
            return Some(TzinfoKind::OtherAware);
        }
    }
    None
}

/// `timedelta(hours=…)` / `timedelta(minutes=…)` (optionally combined),
/// read as its own exact SECOND count — `classify_tzinfo_expression`'s
/// own `timezone(timedelta(...))` arm. A `timezone`'s own offset is a
/// WHOLE-SECOND count, so this reader takes the `hours=`/`minutes=`/
/// `seconds=` keywords straight off the literal AST rather than
/// normalize through `timedelta_construction_value`'s duration triple
/// and read it back. `None` for any other keyword
/// (`days=`, `weeks=`, …), a positional argument, or a non-literal
/// value — this reader never guesses an offset.
pub(in crate::expressions) fn offset_seconds_of_timedelta_call(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "timedelta" || !call.arguments.args.is_empty() {
        return None;
    }
    let mut seconds: i64 = 0;
    for keyword in &call.arguments.keywords {
        let Some(name) = keyword.arg.as_ref() else {
            return None;
        };
        let value = datetime_field_argument(&keyword.value, environment, kernel)?;
        match name.as_str() {
            "hours" => seconds = seconds.checked_add(value.checked_mul(3600)?)?,
            "minutes" => seconds = seconds.checked_add(value.checked_mul(60)?)?,
            "seconds" => seconds = seconds.checked_add(value)?,
            _ => return None,
        }
    }
    Some(seconds)
}

/// One `ObjectKey` carrying a known Integer field — the small builder
/// `datetime_construction_value` repeats once per calendar field.
pub(in crate::expressions) fn integer_object_key(name: &str, value: i64) -> ObjectKey {
    ObjectKey {
        name: name.to_owned(),
        numeric: false,
        value: known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved),
    }
}

/// One `datetime.datetime(...)` constructor argument's known Integer
/// value — every positional/keyword calendar field this file reads
/// (`datetime_construction_value`'s own doc).
pub(in crate::expressions) fn datetime_field_argument(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let value = evaluate_expression(expr, environment, kernel);
    let (number, sort) = single_numeric_value(&value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    Some(number as i64)
}

/// Whether `callee` names the `datetime.date` class, NOT locally
/// shadowed — `date.1`'s own receiver shape, resolved by CANONICAL
/// import identity the same way `is_datetime_datetime_attribute` is
/// for the sibling `datetime` class (that function's own doc — the
/// qualified chain `datetime.date`/`dtm.date` OR the bare aliased
/// name `from datetime import date[ as x]` gave `x`). Gates both the
/// `datetime.date(...)` CONSTRUCTION call and the
/// `datetime.date.fromisoformat(...)` CLASSMETHOD call's own receiver
/// (datetime.rst, `class:: date(year, month, day)`).
pub(in crate::expressions) fn is_datetime_date_attribute(callee: &Expr, environment: &Environment) -> bool {
    if let Expr::Attribute(attribute) = callee {
        if attribute.attr.as_str() == "date" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if environment.read(module_name.id.as_str()).is_some() {
                    return false;
                }
                if let Some(imports) = environment.datetime_imports() {
                    return imports.module_names.contains(module_name.id.as_str());
                }
                return module_name.id.as_str() == "datetime";
            }
        }
        return false;
    }
    let Expr::Name(name) = callee else {
        return false;
    };
    let Some(imports) = environment.datetime_imports() else {
        return false;
    };
    imports.date_class_names.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none()
}

/// Whether `callee` names the `datetime.timedelta` class, NOT locally
/// shadowed — date.5's own receiver shape, resolved by CANONICAL
/// import identity the same way `is_datetime_datetime_attribute` is
/// for the sibling `datetime` class (that function's own doc — the
/// qualified chain `datetime.timedelta`/`dtm.timedelta` OR the bare
/// aliased name `from datetime import timedelta[ as x]` gave `x`).
/// Gates the `datetime.timedelta(days=n)` CONSTRUCTION call
/// (datetime.rst, `class:: timedelta(days=0, ...)`).
pub(in crate::expressions) fn is_datetime_timedelta_attribute(callee: &Expr, environment: &Environment) -> bool {
    if let Expr::Attribute(attribute) = callee {
        if attribute.attr.as_str() == "timedelta" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if environment.read(module_name.id.as_str()).is_some() {
                    return false;
                }
                if let Some(imports) = environment.datetime_imports() {
                    return imports.module_names.contains(module_name.id.as_str());
                }
                return module_name.id.as_str() == "datetime";
            }
        }
        return false;
    }
    let Expr::Name(name) = callee else {
        return false;
    };
    let Some(imports) = environment.datetime_imports() else {
        return false;
    };
    imports.timedelta_class_names.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none()
}

/// datetime.rst:88,94 — `MINYEAR` is 1, `MAXYEAR` is 9999 (date.2's own
/// row): "every `date`/`datetime` year satisfies `MINYEAR <= year <=
/// MAXYEAR`." The kernel's OWN range check (`epochDaysWithinLimits`,
/// Temporal's PlainDate window, roughly ±271821 years) is far WIDER
/// than Python's — date.2's row states this directly ("narrower than
/// Temporal's PlainDate day-range limit the JS kernel elects"), and the
/// kernel's `validDate`/`isoDate` ops enforce ONLY their own wider bound
/// (or, for `validDate`, no year bound at all — `isValidISODate` checks
/// month/day-of-month only). Every `datetime_date` construction path in
/// this file therefore asks the kernel's OWN `pyYearInRange` op
/// (`exports_calendar.lean`'s `"pyYearInRange"` arm, `Refinements.
/// pyYearInRange`, `languages/python/dates_durations/year_range.lean`)
/// — one wrapper, three call sites (`date_construction_value`,
/// `date_fromisoformat_value`, `date_shifted_by_timedelta`) unchanged.
/// `None` on a refused ask, matching every other kernel ask in this
/// crate.
pub(in crate::expressions) fn python_year_in_range(year: i64, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::PyYearInRange,
            year,
            month: 0,
            day: 0,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("valid")?.as_bool()
}

/// `datetime.date(year, month, day)` — a tagged `Kind::Object` (`source =
/// "datetime_date"`) carrying `year`/`month`/`day` Integer `ObjectKey`s.
/// datetime.rst, `class:: date(year, month, day)`: all three arguments
/// are REQUIRED, positional-or-keyword, no defaults — unlike
/// `datetime_construction_value`'s `hour`/`minute`/`second`, a missing
/// field here declines the whole construction rather than defaulting.
/// Validated through TWO kernel asks: `calendar.validDate` (date.1's own
/// seam) for calendar correctness (month/day-of-month), and
/// `python_year_in_range`'s own `pyYearInRange` ask for date.2's
/// `MINYEAR`/`MAXYEAR` window (see that function's own doc for why
/// `validDate` alone does not cover it) — a year/month/day combination
/// either ask refuses answers `None`.
pub(in crate::expressions) fn date_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let field_names = ["year", "month", "day"];
    let mut fields: Vec<Option<i64>> = vec![None; field_names.len()];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        let slot = fields.get_mut(index)?;
        *slot = Some(datetime_field_argument(arg, environment, kernel)?);
    }
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            return None;
        };
        let position = field_names.iter().position(|name| *name == arg_name.as_str())?;
        let slot = fields.get_mut(position)?;
        *slot = Some(datetime_field_argument(&keyword.value, environment, kernel)?);
    }
    let year = fields[0]?;
    let month = fields[1]?;
    let day = fields[2]?;
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    if !valid_civil_date(year, month, day, kernel)? {
        return None;
    }
    let keys = field_names.iter().zip([year, month, day]).map(|(name, value)| integer_object_key(name, value)).collect();
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

/// `calendar.validDate` — date.1's own kernel seam, asked directly
/// (rather than through `epoch_days_of_civil_date`'s `epochDays` op)
/// because construction only needs the `valid` verdict, not a day
/// count. `None` on a refused ask (the kernel panics on no answer;
/// `ask_kernel` catches that the same way `epoch_days_of_civil_date`
/// does), matching every other refused kernel ask in this crate.
pub(in crate::expressions) fn valid_civil_date(year: i64, month: i64, day: i64, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::ValidDate,
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
    asked.get("valid")?.as_bool()
}

/// `datetime.timedelta(days=…, seconds=…, microseconds=…,
/// milliseconds=…, minutes=…, hours=…, weeks=…)` — a tagged
/// `Kind::Object` (`source = "datetime_timedelta"`) carrying the
/// NORMALIZED `days`/`seconds`/`microseconds` triple as Integer
/// `ObjectKey`s. datetime.rst, `class:: timedelta(days=0, seconds=0,
/// microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)`:
/// "Only *days*, *seconds* and *microseconds* are stored internally,"
/// with the stated conversions (a millisecond is 1000 microseconds, a
/// minute 60 seconds, an hour 3600 seconds, a week 7 days) and the
/// normalization datetime.rst:221 pins — `0 <= microseconds < 1000000`,
/// `0 <= seconds < 3600*24`, `-999999999 <= days <= 999999999`. Every
/// one of the seven keywords is read; a positional argument or a
/// keyword outside the seven declines the whole construction, matching
/// this crate's `datetime_construction_value` convention of declining
/// rather than guessing at an argument shape it does not read. Each
/// argument must read as a known Integer (`datetime_field_argument`) —
/// the constructor also admits floats, and a float argument declines
/// here rather than round into `timedelta.resolution`.
///
/// Validated through the kernel's `calendar.validDuration` ask (date.5's
/// own seam): the ten-field vector is `(years, months, weeks, days,
/// hours, minutes, seconds, milliseconds, microseconds, nanoseconds)`
/// (`theories/calendar/duration.lean`'s own comment), posed over the
/// NORMALIZED triple so the kernel's magnitude/sign guards see the same
/// three fields Python stores.
pub(in crate::expressions) fn timedelta_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.args.is_empty() {
        return None;
    }
    // The whole duration in microseconds, accumulated from whichever of
    // the seven keywords the call spells, each converted by the factor
    // datetime.rst states for it.
    let mut microseconds: i128 = 0;
    for keyword in &call.arguments.keywords {
        let name = keyword.arg.as_ref()?;
        let count = datetime_field_argument(&keyword.value, environment, kernel)? as i128;
        let per_unit: i128 = match name.as_str() {
            "days" => 86_400_000_000,
            "seconds" => 1_000_000,
            "microseconds" => 1,
            "milliseconds" => 1_000,
            "minutes" => 60_000_000,
            "hours" => 3_600_000_000,
            "weeks" => 7 * 86_400_000_000,
            _ => return None,
        };
        microseconds = microseconds.checked_add(count.checked_mul(per_unit)?)?;
    }
    timedelta_instance_of_microseconds(microseconds, kernel)
}

/// A tagged `datetime_timedelta` instance built from a whole duration in
/// MICROSECONDS, normalized to the `days`/`seconds`/`microseconds`
/// triple datetime.rst:221 pins (`0 <= microseconds < 1000000`,
/// `0 <= seconds < 3600*24`, the day count carrying the sign). Shared by
/// `timedelta_construction_value` and `datetime_difference_value` — a
/// `datetime1 - datetime2` result is the SAME instance shape a literal
/// `timedelta(...)` construction builds, so every downstream reader
/// treats the two identically. `None` when the kernel's
/// `calendar.validDuration` ask refuses the normalized triple (the
/// `-999999999 <= days <= 999999999` bound), or when the microsecond
/// total exceeds what an `i64` field can carry.
pub(in crate::expressions) fn timedelta_instance_of_microseconds(microseconds: i128, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    const MICROS_PER_DAY: i128 = 86_400_000_000;
    let days = microseconds.div_euclid(MICROS_PER_DAY);
    let within_day = microseconds.rem_euclid(MICROS_PER_DAY);
    let seconds = within_day / 1_000_000;
    let residual_microseconds = within_day % 1_000_000;
    let days = i64::try_from(days).ok()?;
    let seconds = i64::try_from(seconds).ok()?;
    let residual_microseconds = i64::try_from(residual_microseconds).ok()?;
    // The kernel's `isValidDuration` (§7.5, theories/calendar/
    // duration.lean) requires ONE SIGN across all ten fields, which is
    // the ISO 8601 duration form. Python's stored triple is a different
    // normalization — datetime.rst:221 puts the whole sign on `days` and
    // keeps `seconds`/`microseconds` nonnegative, so `timedelta(
    // microseconds=-175000)` stores `days=-1, seconds=86399,
    // microseconds=825000`, a mixed-sign vector the ISO validator
    // rightly refuses. The ask therefore poses the SAME duration in the
    // single-sign form: the magnitude split into ISO days/seconds/
    // microseconds, with one sign carried across all three.
    let iso_magnitude = microseconds.unsigned_abs() as i128;
    let sign: i64 = if microseconds < 0 { -1 } else { 1 };
    let iso_days = i64::try_from(iso_magnitude / MICROS_PER_DAY).ok()? * sign;
    let iso_seconds = i64::try_from(iso_magnitude % MICROS_PER_DAY / 1_000_000).ok()? * sign;
    let iso_microseconds = i64::try_from(iso_magnitude % 1_000_000).ok()? * sign;
    if !valid_duration_triple(iso_days, iso_seconds, iso_microseconds, kernel)? {
        return None;
    }
    let instance_keys = vec![
        integer_object_key("days", days),
        integer_object_key("seconds", seconds),
        integer_object_key("microseconds", residual_microseconds),
    ];
    let mut instance = known_object(instance_keys, None, true, TrustProved, false);
    instance.source = "datetime_timedelta".to_owned();
    // The ISO 8601 duration spelling `calendar_interpreter`'s own
    // `duration_fields` grammar reads: whole days, then the within-day
    // seconds with the microsecond residue as a fractional second — the
    // SAME single-sign ISO fields the kernel ask above poses, with the
    // sign spelled once as a leading `-` over the whole duration.
    let magnitude = if sign < 0 { "-" } else { "" };
    let point = format!(
        "{magnitude}P{}DT{}.{:06}S",
        iso_days.abs(),
        iso_seconds.abs(),
        iso_microseconds.unsigned_abs()
    );
    instance.temporal = Some(Box::new(refined_sets::calendar_interpreter::TemporalAnnotation {
        chart: refined_sets::calendar_interpreter::TemporalChart::Duration,
        min: Some(point.clone()),
        max: Some(point),
    }));
    Some(instance)
}

/// The whole duration in MICROSECONDS carried by a tagged
/// `datetime_timedelta` instance — the inverse of
/// `timedelta_instance_of_microseconds`, reading back the normalized
/// `days`/`seconds`/`microseconds` triple datetime.rst stores. `None`
/// for an instance carrying no `days` field (the pydantic-surface
/// instance, which carries its duration only as ISO text).
pub(in crate::expressions) fn timedelta_total_microseconds(instance: &AbstractValue) -> Option<i128> {
    let days = super::components::datetime_field(instance, "days")? as i128;
    let seconds = super::components::datetime_field(instance, "seconds").unwrap_or(0.0) as i128;
    let microseconds = super::components::datetime_field(instance, "microseconds").unwrap_or(0.0) as i128;
    Some(days * 86_400_000_000 + seconds * 1_000_000 + microseconds)
}

/// `calendar.validDuration` asked over a SINGLE-SIGN ISO
/// `days`/`seconds`/`microseconds` triple —
/// `timedelta_instance_of_microseconds`'s own validity gate (date.5's
/// kernel seam), spelled as its own function so the field-order comment
/// lives beside the one call site that builds the vector. The caller
/// converts Python's own mixed-sign normalization into this form first;
/// see its comment for why.
pub(in crate::expressions) fn valid_duration_triple(days: i64, seconds: i64, microseconds: i64, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    // (years, months, weeks, days, hours, minutes, seconds,
    // milliseconds, microseconds, nanoseconds)
    let fields = vec![0.0, 0.0, 0.0, days as f64, 0.0, 0.0, seconds as f64, 0.0, microseconds as f64, 0.0];
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::ValidDuration,
            year: 0,
            month: 0,
            day: 0,
            days: 0,
            fields,
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("valid")?.as_bool()
}
