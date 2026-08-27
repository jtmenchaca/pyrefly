//! `Annotated[date|timedelta|datetime|AwareDatetime|NaiveDatetime,
//! Field(ge=…/le=…/gt=…/lt=…)]` read into the module's own temporal
//! calendar window — the pydantic temporal surface's own compiler.

use refined_sets::calendar_interpreter::{TemporalAnnotation, TemporalChart};
use ruff_python_ast::{Expr, ModModule, Stmt};

use super::aliases::TemporalAwareness;
use super::annotated_set::{names_field, INERT_FIELD_KWARGS};
use super::imports::SurfaceImports;
use super::literals::literal_length;

/// `Annotated[date|timedelta|datetime|AwareDatetime|NaiveDatetime,
/// Field(ge=…/le=…/gt=…/lt=…)]` → the stated calendar window, resolved
/// against the module's import identities. The base must be one of the
/// five recognized temporal names (by import identity, the same
/// discipline `annotated_expression_set` holds for `Annotated`/
/// `StrictInt`); every metadata element must be a recognized `Field(…)`
/// call whose `ge`/`le`/`gt`/`lt` value is itself a `date(...)`/
/// `timedelta(...)`/`datetime(...)` literal call OR a bare Name
/// resolving to a MODULE-LEVEL plain assignment of one (`_cutoff =
/// datetime(...)` then `Field(ge=_cutoff)`, showcase.py's own `Cutoff`/
/// `Stamp` rows) — any other kwarg, or a bound this reader cannot read
/// as a temporal literal, refuses the whole alias, the same
/// all-or-nothing discipline `annotated_expression_set` already holds
/// for its own numeric/string kwargs.
///
/// `gt`/`lt` are read as `ge`/`le` — this table's `TemporalAnnotation`
/// only carries `min`/`max` (calendar_interpreter.rs, no open/closed
/// bit), the same rounding-free identification refined-ts-go's own
/// temporal surface makes (`SPEC-boundaries.md`'s "the calendar
/// carries no strict/non-strict distinction, only the ISO endpoint
/// itself" convention); a strict bound stated at these positions and a
/// non-strict one at the same value are indistinguishable this table's
/// own kernel questions can decide (`compare_on_chart` compares two
/// ISO points, never a strictness bit), matching every showcase.py row
/// this table serves (`ge=`/`le=` only).
pub(super) fn temporal_alias_annotation(
    value: &Expr,
    imports: &SurfaceImports,
    module: &ModModule,
) -> Option<(TemporalAnnotation, TemporalAwareness)> {
    temporal_annotation_of(value, imports, Some(module))
}

/// `temporal_alias_annotation`'s own recognition, usable with no module
/// in hand — `typereading.rs::declared_refinement`'s own inline
/// `Annotated[date|timedelta|datetime|AwareDatetime|NaiveDatetime,
/// Field(…)]` arm (a PARAMETER annotation spelled directly, not
/// through a module-level alias) never carries a `&ModModule` — every
/// caller in that file passes only the annotation expression,
/// `aliases`, and `imports`. A bare-Name bound (`Field(ge=_cutoff)`)
/// is therefore unresolvable at that call site; a module-level
/// `Cutoff`/`Stamp`-style alias reads through `compile_aliases`'s own
/// `temporal_alias_annotation` instead, which DOES have the module.
pub fn temporal_inline_annotation(value: &Expr, imports: &SurfaceImports) -> Option<(TemporalAnnotation, TemporalAwareness)> {
    temporal_annotation_of(value, imports, None)
}

/// A BARE `date`/`timedelta`/`datetime`/`AwareDatetime`/`NaiveDatetime`
/// parameter annotation — no `Annotated[…, Field(…)]` wrapper, no stated
/// bound — read as the UNBOUNDED window on that name's own chart. This
/// is the temporal twin of `typereading::base_sort_return_refinement`'s
/// bare-`int` ray: `d: datetime` states "any instant," exactly as `n:
/// int` states "any whole number," and it is a claim, not an absence.
/// The window carries no `min` and no `max`, and `bounds_imply` reads
/// that as REFUTING any bounded target window (its own doc) — so `d:
/// datetime` reaching a `Year2021`-declared position is refused, which
/// is what an unconstrained instant against a stated year window owes.
///
/// Resolved by import identity through the module's own `SurfaceImports`
/// table, the same discipline every other recognition in this file
/// keeps. `AwareDatetime`/`NaiveDatetime` additionally carry their
/// awareness requirement, so a bare `d: AwareDatetime` still refuses a
/// naive construction through `temporal_admission_refusal`.
///
/// SCOPED TO PARAMETERS: `check::seed_parameters` is the one caller, for
/// the same reason the bare-sort reader is scoped there — reading bare
/// temporal names in the general annotation table would make every
/// `-> datetime` helper return a judged position and turn each unreadable
/// helper body into a fresh blocker.
pub fn bare_temporal_annotation(value: &Expr, imports: &SurfaceImports) -> Option<(TemporalAnnotation, TemporalAwareness)> {
    let Expr::Name(name) = value else {
        return None;
    };
    let (chart, awareness) = if imports.date_names.contains(name.id.as_str()) {
        (TemporalChart::PlainDate, TemporalAwareness::Any)
    } else if imports.timedelta_names.contains(name.id.as_str()) {
        (TemporalChart::Duration, TemporalAwareness::Any)
    } else if imports.datetime_names.contains(name.id.as_str()) {
        (TemporalChart::Instant, TemporalAwareness::Any)
    } else if imports.aware_datetime_names.contains(name.id.as_str()) {
        (TemporalChart::Instant, TemporalAwareness::RequireAware)
    } else if imports.naive_datetime_names.contains(name.id.as_str()) {
        (TemporalChart::Instant, TemporalAwareness::RequireNaive)
    } else {
        return None;
    };
    Some((TemporalAnnotation { chart, min: None, max: None }, awareness))
}

/// The shared recognition both `temporal_alias_annotation` (module-level
/// alias RHS) and `temporal_inline_annotation` (inline parameter
/// annotation) drive: see `temporal_alias_annotation`'s own doc for the
/// full rule. `module` is `None` at an inline call site — a bare-Name
/// bound simply fails to resolve there (`temporal_literal_iso`'s own
/// `Option<&ModModule>` threading), rather than the whole function
/// requiring a module it does not have.
fn temporal_annotation_of(
    value: &Expr,
    imports: &SurfaceImports,
    module: Option<&ModModule>,
) -> Option<(TemporalAnnotation, TemporalAwareness)> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !imports.annotated_names.contains(head.id.as_str()) {
        return None;
    }
    let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
        return None;
    };
    let (base, metadata) = arguments.elts.split_first()?;
    let Expr::Name(base_name) = base else {
        return None;
    };
    let (chart, awareness) = if imports.date_names.contains(base_name.id.as_str()) {
        (TemporalChart::PlainDate, TemporalAwareness::Any)
    } else if imports.timedelta_names.contains(base_name.id.as_str()) {
        (TemporalChart::Duration, TemporalAwareness::Any)
    } else if imports.datetime_names.contains(base_name.id.as_str()) {
        (TemporalChart::Instant, TemporalAwareness::Any)
    } else if imports.aware_datetime_names.contains(base_name.id.as_str()) {
        (TemporalChart::Instant, TemporalAwareness::RequireAware)
    } else if imports.naive_datetime_names.contains(base_name.id.as_str()) {
        (TemporalChart::Instant, TemporalAwareness::RequireNaive)
    } else {
        return None;
    };
    let mut min: Option<String> = None;
    let mut max: Option<String> = None;
    for meta in metadata {
        let Expr::Call(call) = meta else {
            return None;
        };
        if !names_field(call.func.as_ref(), imports) {
            return None;
        }
        for keyword in call.arguments.keywords.iter() {
            let name = keyword.arg.as_ref()?;
            match name.as_str() {
                "ge" | "gt" => min = Some(temporal_literal_iso(&keyword.value, chart, imports, module)?),
                "le" | "lt" => max = Some(temporal_literal_iso(&keyword.value, chart, imports, module)?),
                other if INERT_FIELD_KWARGS.contains(&other) => {}
                _ => return None,
            }
        }
    }
    if min.is_none() && max.is_none() {
        return None;
    }
    Some((TemporalAnnotation { chart, min, max }, awareness))
}

/// One `Field(ge=…)`-style bound expression, read as the ISO spelling
/// `calendar_interpreter.rs`'s own chart reader accepts for `chart`: a
/// `date(...)`/`timedelta(...)`/`datetime(...)` call read directly
/// (`temporal_construction_iso`), or — when a module is in hand — a
/// bare Name resolved against the module's own TOP-LEVEL plain
/// assignments (`_cutoff = datetime(...)`) first, falling back to the
/// SAME construction reading on the resolved RHS. `None` for anything
/// else: a computed expression, a bare Name with no module to resolve
/// it against (the inline call site, `module: None`), an unresolvable
/// name, or a construction this table cannot read.
fn temporal_literal_iso(expr: &Expr, chart: TemporalChart, imports: &SurfaceImports, module: Option<&ModModule>) -> Option<String> {
    if let Expr::Name(name) = expr {
        let bound_value = top_level_plain_assignment(module?, name.id.as_str())?;
        return temporal_construction_iso(bound_value, chart, imports);
    }
    temporal_construction_iso(expr, chart, imports)
}

/// The module's own top-level `name = <value>` plain assignment RHS
/// (`Stmt::Assign` with a single bare-Name target) — the shape
/// showcase.py's own `_cutoff = datetime(2024, 1, 1, 0, 0, 0,
/// tzinfo=timezone.utc)` takes. `None` when no such assignment exists,
/// or the target is not a single bare Name (a tuple-unpack assignment
/// states no single RHS this reader could point at).
fn top_level_plain_assignment<'a>(module: &'a ModModule, name: &str) -> Option<&'a Expr> {
    for stmt in module.body.iter() {
        let Stmt::Assign(assign) = stmt else {
            continue;
        };
        let [Expr::Name(target)] = assign.targets.as_slice() else {
            continue;
        };
        if target.id.as_str() == name {
            return Some(assign.value.as_ref());
        }
    }
    None
}

/// A `date(year, month, day)` / `timedelta(days=n)` /
/// `datetime(year, month, day, hour=0, minute=0, second=0,
/// tzinfo=...)` call, read as the ISO spelling its own chart expects:
/// `date` → `"YYYY-MM-DD"`; `timedelta` → the ISO 8601 duration
/// `"PnD"` (`duration_fields`'s own grammar, calendar_interpreter.rs —
/// every field but `days` is 0 here, the same single-field scope
/// `expressions.rs::timedelta_construction_value` already keeps);
/// `datetime` → `"YYYY-MM-DDTHH:MM:SSZ"` when `tzinfo=` spells the UTC
/// singleton (`datetime.timezone.utc`/`datetime.UTC`), or the offset-
/// free `"YYYY-MM-DDTHH:MM:SS"` when no `tzinfo=` is given (a NAIVE
/// spelling — `chart_reading`'s own `Instant` arm reads this as
/// `Unprovable`, which `compare_on_chart` then reports through
/// `BoundsVerdict::Alert` rather than a decided proof, matching a
/// naive datetime's own documented unprovability against an exact
/// instant). Every field is a literal int (or, for `days`, the same
/// literal-int reading `Field`'s own numeric kwargs already take) —
/// this reader never evaluates an expression, matching
/// `annotated_expression_set`'s own literal-only discipline. Any other
/// argument shape (a non-literal field, an unrecognized keyword, a
/// non-UTC `tzinfo=`) declines.
fn temporal_construction_iso(expr: &Expr, chart: TemporalChart, imports: &SurfaceImports) -> Option<String> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    match chart {
        TemporalChart::PlainDate => {
            if !imports.date_names.contains(callee.id.as_str()) {
                return None;
            }
            let [year, month, day] = date_fields(call, &["year", "month", "day"])?;
            Some(format!("{year:04}-{month:02}-{day:02}"))
        }
        TemporalChart::Duration => {
            if !imports.timedelta_names.contains(callee.id.as_str()) {
                return None;
            }
            if !call.arguments.args.is_empty() {
                return None;
            }
            let [keyword] = call.arguments.keywords.as_slice() else {
                return None;
            };
            if keyword.arg.as_ref().map(|name| name.as_str()) != Some("days") {
                return None;
            }
            let days = literal_length(&keyword.value)?;
            Some(format!("P{days}D"))
        }
        TemporalChart::Instant => {
            if !imports.datetime_names.contains(callee.id.as_str())
                && !imports.aware_datetime_names.contains(callee.id.as_str())
                && !imports.naive_datetime_names.contains(callee.id.as_str())
            {
                return None;
            }
            let field_names = ["year", "month", "day", "hour", "minute", "second"];
            let mut fields: Vec<Option<i64>> = vec![None; field_names.len()];
            for (index, arg) in call.arguments.args.iter().enumerate() {
                let slot = fields.get_mut(index)?;
                *slot = Some(literal_length(arg)?);
            }
            let mut is_utc = false;
            for keyword in &call.arguments.keywords {
                let Some(arg_name) = keyword.arg.as_ref() else {
                    return None;
                };
                if arg_name.as_str() == "tzinfo" {
                    is_utc = is_utc_tzinfo_expr(&keyword.value);
                    if !is_utc {
                        // a tzinfo this reader cannot prove is exactly UTC
                        // — decline rather than guess an offset, matching
                        // `expressions.rs::datetime_construction_value`'s
                        // own discipline.
                        return None;
                    }
                    continue;
                }
                let Some(position) = field_names.iter().position(|name| *name == arg_name.as_str()) else {
                    return None;
                };
                let slot = fields.get_mut(position)?;
                *slot = Some(literal_length(&keyword.value)?);
            }
            let mut values = [0i64; 6];
            for index in 0..field_names.len() {
                values[index] = match fields[index] {
                    Some(v) => v,
                    None if index < 3 => return None,
                    None => 0,
                };
            }
            let [year, month, day, hour, minute, second] = values;
            let zone = if is_utc { "Z" } else { "" };
            Some(format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}{zone}"))
        }
        _ => None,
    }
}

/// One `date(...)` constructor argument's own field triple, positional
/// or keyword, read as literal ints — `temporal_construction_iso`'s own
/// `PlainDate` arm, factored out so the field-order comment lives
/// beside its one call site.
fn date_fields(call: &ruff_python_ast::ExprCall, names: &[&str; 3]) -> Option<[i64; 3]> {
    let mut fields: Vec<Option<i64>> = vec![None; names.len()];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        let slot = fields.get_mut(index)?;
        *slot = Some(literal_length(arg)?);
    }
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            return None;
        };
        let position = names.iter().position(|name| *name == arg_name.as_str())?;
        let slot = fields.get_mut(position)?;
        *slot = Some(literal_length(&keyword.value)?);
    }
    Some([fields[0]?, fields[1]?, fields[2]?])
}

/// Whether `expr` is exactly `datetime.timezone.utc` or `datetime.UTC`
/// — `expressions.rs::is_utc_tzinfo_expression`'s own twin, mirrored
/// locally (this file cannot import `expressions.rs` without cycling:
/// `expressions.rs` itself does not depend on `surface.rs` today, but
/// `typereading.rs` — which `surface.rs` is imported BY — sits between
/// them, and a fresh `surface.rs -> expressions.rs` edge would still
/// need to route through the same module graph `literal_alias_set`'s
/// own doc already declines to cross).
fn is_utc_tzinfo_expr(expr: &Expr) -> bool {
    if let Expr::Attribute(outer) = expr {
        if outer.attr.as_str() == "UTC" {
            if let Expr::Name(name) = outer.value.as_ref() {
                if name.id.as_str() == "datetime" {
                    return true;
                }
            }
        }
        if outer.attr.as_str() == "utc" {
            // datetime.timezone.utc — a three-level chain.
            if let Expr::Attribute(middle) = outer.value.as_ref() {
                if middle.attr.as_str() == "timezone" {
                    if let Expr::Name(name) = middle.value.as_ref() {
                        if name.id.as_str() == "datetime" {
                            return true;
                        }
                    }
                }
            }
            // `timezone.utc` — a two-level chain, `Name("timezone").utc`
            // (`from datetime import timezone`, showcase.py's own
            // spelling) — the same recognition `expressions.rs::is_utc_
            // tzinfo_expression`'s own twin arm now takes.
            if let Expr::Name(name) = outer.value.as_ref() {
                if name.id.as_str() == "timezone" {
                    return true;
                }
            }
        }
    }
    false
}
