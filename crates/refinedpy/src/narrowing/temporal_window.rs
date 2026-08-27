//! Temporal channel: a calendar-component test over a window-flowing
//! temporal name (`d.year == 2021`) tightens that name's own temporal
//! WINDOW to the instants the test admits.

use std::sync::Arc;

use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::calendar_interpreter::TemporalAnnotation;
use refined_sets::calendar_interpreter::TemporalChart;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::literal_number;
use crate::expressions::instant_stepped_by_microseconds;
use crate::expressions::utc_iso_microseconds;

/// The TEMPORAL channel's own entry point: for every `<name>.year ==
/// <literal>` comparison `condition` folds through `and`, tighten
/// `<name>`'s own temporal window to that calendar year.
///
/// The window a year test admits is the year's own two ISO endpoints —
/// datetime.rst, `attribute:: datetime.year` ("Between MINYEAR and
/// MAXYEAR inclusive") read together with the calendar: an instant whose
/// calendar year is N sits at or after `N-01-01T00:00:00Z` and at or
/// before `N-12-31T23:59:59Z`. Those are the SAME two spellings a
/// `Year2021`-style declaration states for itself, so the narrowed
/// window and a declared year window compare on one chart with no
/// further conversion.
///
/// Scoped to the `Instant` chart and to `==`: a `date`'s `PlainDate`
/// chart has its own endpoints, and an inequality over a year states a
/// window this channel does not build. Both are the honest
/// "narrows nothing" default every other narrowing leaf keeps.
///
/// Runs on a WINDOW-FLOWING value only (`source == "temporal_flow"` —
/// `check::seed_parameters`' own temporal seed). A concrete construction
/// already carries its exact instant, and a year test over it decides
/// nothing this channel could tighten.
pub(super) fn narrow_temporal_windows(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    match condition {
        Expr::BoolOp(bool_op) if bool_op.op == BoolOp::And && truth => {
            for value in &bool_op.values {
                narrow_temporal_windows(value, environment, kernel, truth);
            }
        }
        Expr::Compare(compare) if truth => {
            let mut left = compare.left.as_ref();
            for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
                narrow_one_year_test(left, *op, right, environment, kernel);
                narrow_one_instant_ordering(left, *op, right, environment, kernel);
                left = right;
            }
        }
        _ => {}
    }
}

/// One `<name> < <instant>` ordering pair (any of the four ordering
/// operators, either operand order), applied to `<name>`'s own window.
/// datetime.rst's `.datetime` operation table, note (5): an order
/// comparison between two datetimes compares the instants they name — so
/// `d < CUTOFF` proving true puts every value `d` can hold at or before
/// `CUTOFF`, which is exactly a tightened upper bound on `d`'s window.
///
/// The bound instant is read from the OTHER side's own exact temporal
/// spelling — a concrete `datetime_datetime` construction, or a
/// module-level constant bound to one. A window-flowing other side states
/// no single instant to bound against and narrows nothing.
///
/// This crate's calendar window carries no open/closed bit
/// (`surface::temporal`'s own note, `SPEC-boundaries.md`'s convention),
/// so a STRICT comparison is spelled by stepping the bound one
/// MICROSECOND inward before the window is built. That is exact rather
/// than conservative: datetime.rst gives `timedelta.resolution` as one
/// microsecond and `datetime.microsecond` as an `int` "in
/// `range(1000000)`", so the instant one microsecond before `CUTOFF` IS
/// the largest instant strictly below it — there is nothing between.
fn narrow_one_instant_ordering(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>) {
    // Which side names the narrowed value, and whether the surviving
    // values sit BELOW the bound (an upper bound) or above it. The
    // narrowed name may sit on either side: `d < CUTOFF` bounds `d`
    // above, and the mirrored `CUTOFF > d` states the same thing.
    let narrows_upper_when_left = match op {
        CmpOp::Lt | CmpOp::LtE => true,
        CmpOp::Gt | CmpOp::GtE => false,
        _ => return,
    };
    let is_flowing_instant = |expr: &Expr| {
        let Expr::Name(name) = expr else { return false };
        let Some(value) = environment.read(name.id.as_str()) else { return false };
        value.kind == Kind::Object
            && value.source == "temporal_flow"
            && value.temporal.as_ref().is_some_and(|window| window.chart == TemporalChart::Instant)
    };
    let (name_side, bound_side, narrows_upper) = if is_flowing_instant(left) {
        (left, right, narrows_upper_when_left)
    } else if is_flowing_instant(right) {
        (right, left, !narrows_upper_when_left)
    } else {
        return;
    };
    let Expr::Name(name) = name_side else {
        return;
    };
    let Some(current) = environment.read(name.id.as_str()) else {
        return;
    };
    let Some(flowing) = &current.temporal else {
        return;
    };
    if flowing.chart != TemporalChart::Instant {
        return;
    }
    let Some(bound) = exact_instant_spelling(bound_side, environment) else {
        return;
    };
    // a STRICT comparison excludes the bound itself: step it one
    // microsecond inward, the finest instant Python's own resolution
    // distinguishes
    let strict = matches!(op, CmpOp::Lt | CmpOp::Gt);
    let bound = if strict {
        let step = if narrows_upper { -1 } else { 1 };
        let Some(stepped) = instant_stepped_by_microseconds(&bound, step, kernel) else {
            return;
        };
        stepped
    } else {
        bound
    };
    let narrowed = if narrows_upper {
        TemporalAnnotation {
            chart: TemporalChart::Instant,
            min: flowing.min.clone(),
            max: Some(min_iso_point(flowing.max.as_deref(), &bound, kernel)),
        }
    } else {
        TemporalAnnotation {
            chart: TemporalChart::Instant,
            min: Some(max_iso_point(flowing.min.as_deref(), &bound, kernel)),
            max: flowing.max.clone(),
        }
    };
    bind_tightened_window(name.id.as_str(), &current.clone(), narrowed, environment, kernel);
}

/// Binds `name` to its own value carrying the tightened window, and
/// RE-DERIVES every offset the environment records as derived from it
/// (`env::TemporalOffsetDerivation`'s own doc): `offset_ms = (d -
/// REFERENCE) // timedelta(milliseconds=1)` computed BEFORE a guard that
/// narrows `d` still owes the offset window the narrowed instant
/// implies, and no fact about `d` alone can carry it.
///
/// The re-derived offset is the derivation applied to the window's two
/// endpoints — the map is monotone in the instant for a positive unit,
/// so the endpoints bound it. The result MEETS whatever the offset
/// already held (its own earlier guards, `offset_ms >= 0` among them)
/// rather than replacing it: a narrowing never widens.
fn bind_tightened_window(
    name: &str,
    current: &refined_domain::abstract_value::AbstractValue,
    narrowed: TemporalAnnotation,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) {
    let derived = environment.temporal_offsets_of(name);
    let mut tightened = current.clone();
    tightened.temporal = Some(Box::new(narrowed.clone()));
    environment.bind(name, tightened);
    for (offset_name, derivation) in derived {
        if derivation.unit_microseconds <= 0 {
            continue;
        }
        // Each side of the window carries independently: a guard that
        // bounds the instant only ABOVE (`d < CUTOFF`) bounds the offset
        // only above too, and the offset keeps whatever lower bound its
        // own earlier guards gave it.
        let offset_of = |point: Option<&str>| {
            let instant = utc_iso_microseconds(point?, kernel)?;
            Some((instant - derivation.origin_microseconds).div_euclid(derivation.unit_microseconds))
        };
        let low = offset_of(narrowed.min.as_deref());
        let high = offset_of(narrowed.max.as_deref());
        if low.is_none() && high.is_none() {
            continue;
        }
        let Some(existing) = environment.read(&offset_name) else {
            continue;
        };
        if existing.kind != Kind::Set {
            continue;
        }
        // A `RefinedSet`'s forms CONJOIN (`condition_tree::
        // meet_set_answer`'s own reading), so meeting the re-derived
        // window into whatever the offset already held is appending its
        // bound forms.
        let mut combined = existing.set.forms.clone();
        if let Some(low) = low {
            combined.push(refined_sets::refinement_forms::at_least(low as f64));
        }
        if let Some(high) = high {
            combined.push(refined_sets::refinement_forms::at_most(high as f64));
        }
        let mut rebound = existing.clone();
        rebound.set = refined_sets::refinement_forms::make_refined_set(combined);
        environment.bind(&offset_name, rebound);
    }
}

/// The exact UTC ISO instant a bare name is bound to — a concrete
/// `datetime_datetime` construction's own `.temporal` point, where both
/// endpoints name the same instant. `None` for any other expression
/// shape, an unbound name, or a value whose window is not a single point.
fn exact_instant_spelling(expr: &Expr, environment: &Environment) -> Option<String> {
    let Expr::Name(name) = expr else {
        return None;
    };
    let value = environment.read(name.id.as_str())?;
    if value.source != "datetime_datetime" {
        return None;
    }
    let window = value.temporal.as_ref()?;
    if window.chart != TemporalChart::Instant {
        return None;
    }
    let min = window.min.as_deref()?;
    let max = window.max.as_deref()?;
    if min != max {
        return None;
    }
    Some(min.to_owned())
}

/// One `<name>.year == <literal>` pair (either operand order), applied to
/// `<name>`'s own binding. Anything else — a different attribute, a
/// non-`Eq` operator, a receiver that is not a bare name, a binding that
/// is not a window-flowing Instant — narrows nothing.
fn narrow_one_year_test(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>) {
    if op != CmpOp::Eq {
        return;
    }
    let (attribute, literal) = if let Some(literal) = literal_number(right) {
        (left, literal)
    } else if let Some(literal) = literal_number(left) {
        (right, literal)
    } else {
        return;
    };
    let Expr::Attribute(attribute) = attribute else {
        return;
    };
    if attribute.attr.as_str() != "year" {
        return;
    }
    let Expr::Name(name) = attribute.value.as_ref() else {
        return;
    };
    if literal.fract() != 0.0 || !literal.is_finite() {
        return;
    }
    let year = literal as i64;
    let Some(current) = environment.read(name.id.as_str()) else {
        return;
    };
    if current.kind != Kind::Object || current.source != "temporal_flow" {
        return;
    }
    let Some(flowing) = &current.temporal else {
        return;
    };
    if flowing.chart != TemporalChart::Instant {
        return;
    }
    // The tightened window is the INTERSECTION of what already flowed
    // and the year's own endpoints — a narrowing never widens, the same
    // rule every other channel in this module keeps.
    let year_min = format!("{year:04}-01-01T00:00:00Z");
    let year_max = format!("{year:04}-12-31T23:59:59Z");
    let narrowed = TemporalAnnotation {
        chart: TemporalChart::Instant,
        min: Some(max_iso_point(flowing.min.as_deref(), &year_min, kernel)),
        max: Some(min_iso_point(flowing.max.as_deref(), &year_max, kernel)),
    };
    bind_tightened_window(name.id.as_str(), &current.clone(), narrowed, environment, kernel);
}

/// The LATER of two ISO instant spellings, or `candidate` when the
/// flowing window states no bound on that side. Compared through each
/// spelling's own microsecond count (the kernel's `epochDays` seam
/// underneath), never lexicographically: a fractional-second tail and a
/// whole-second spelling of the same instant sort differently as text.
/// A flowing spelling this reader cannot split loses to `candidate`,
/// which is the tightening direction — a narrowing never widens.
fn max_iso_point(flowing: Option<&str>, candidate: &str, kernel: &Arc<RefinedTSKernel>) -> String {
    match later_of(flowing, candidate, kernel) {
        Some(true) => flowing.expect("later_of answers Some only for a readable flowing point").to_owned(),
        _ => candidate.to_owned(),
    }
}

/// The EARLIER of two ISO instant spellings — `max_iso_point`'s twin for
/// the upper bound, resting on the same exact comparison.
fn min_iso_point(flowing: Option<&str>, candidate: &str, kernel: &Arc<RefinedTSKernel>) -> String {
    match later_of(flowing, candidate, kernel) {
        Some(false) => flowing.expect("later_of answers Some only for a readable flowing point").to_owned(),
        _ => candidate.to_owned(),
    }
}

/// Whether `flowing` names a LATER instant than `candidate` — `None`
/// when either spelling does not split into a microsecond count.
fn later_of(flowing: Option<&str>, candidate: &str, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let flowing = utc_iso_microseconds(flowing?, kernel)?;
    let candidate = utc_iso_microseconds(candidate, kernel)?;
    Some(flowing > candidate)
}
