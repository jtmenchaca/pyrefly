//! Temporal admission, refutation, and alert sentences for assignability.

use refined_domain::abstract_value::AbstractValue;
use refined_sets::calendar_interpreter::TemporalAnnotation;

use crate::diagnostic_sentences::refutation;
use crate::diagnostic_sentences::SENTENCE;
use crate::typereading::DeclaredRefinement;

/// One numeric `ObjectKey` field's own value off a tagged temporal
/// instance — the same by-name linear scan `expressions.rs::
/// datetime_field` already keeps for the identical shape, mirrored
/// locally (assignability.rs cannot import `expressions.rs` without
/// cycling — `expressions.rs` itself already imports `assignability`).
pub(super) fn temporal_field(value: &AbstractValue, name: &str) -> Option<f64> {
    let entry = value.keys.iter().find(|key| key.name == name)?;
    entry.value.values.first().copied()
}

/// THE AWARE/NAIVE ADMISSION LAW: `pydantic.AwareDatetime` "will fail
/// validation if the datetime doesn't have timezone info," and
/// `pydantic.NaiveDatetime` "will fail validation if the datetime
/// provided has timezone info" (pydantic docs, `pydantic.types` module
/// reference, the `AwareDatetime`/`NaiveDatetime` entries — vendored at
/// `specifications/python/pydantic-types.md`). Decided from the
/// construction's own `aware` field (`expressions.rs::datetime_
/// construction_value`'s own doc: `0` naive, `1` UTC-aware, `2` aware
/// with an unresolved exact offset) — `RequireAware` fires on `aware ==
/// 0`; `RequireNaive` fires on `aware` one of `1`/`2`. `Any` (bare
/// `datetime`, or a non-`Instant`-chart declaration) never fires here.
/// `None` when the declared chart does not match the flowing
/// construction's own chart (`date`/`timedelta` values reaching an
/// `Instant`-declared position, or the reverse) — that mismatch is a
/// different, ORDINARY chart mismatch this function does not own;
/// `bounds_verdict_of`'s own `chart_reading` would refuse it, so the
/// caller's fallback (Undetermined, on a chart it cannot compare) is
/// the honest answer there, never a designated fire this law does not
/// state.
pub(super) fn temporal_admission_refusal(value: &AbstractValue, declared: &DeclaredRefinement) -> Option<String> {
    use crate::surface::TemporalAwareness;
    if declared.temporal_awareness == TemporalAwareness::Any {
        return None;
    }
    let declared_temporal = declared.temporal.as_ref()?;
    if declared_temporal.chart != refined_sets::calendar_interpreter::TemporalChart::Instant {
        return None;
    }
    if value.source != "datetime_datetime" {
        return None;
    }
    let aware = temporal_field(value, "aware")?;
    let is_naive = aware == 0.0;
    let fires = match declared.temporal_awareness {
        TemporalAwareness::RequireAware => is_naive,
        TemporalAwareness::RequireNaive => !is_naive,
        TemporalAwareness::Any => false,
    };
    if !fires {
        return None;
    }
    let (value_word, why) = if declared.temporal_awareness == TemporalAwareness::RequireAware {
        ("a naive datetime", "AwareDatetime requires timezone info; this construction carries no tzinfo")
    } else {
        ("an aware datetime", "NaiveDatetime requires no timezone info; this construction carries a tzinfo")
    };
    Some(format!("{} — {why}", refutation(value_word, &declared.spelling, &declared.set)))
}

/// The sentence a temporal bounds REFUTATION earns — `bounds_verdict_of`
/// proved the flowing instant sits outside the declared window.
/// Mirrors `containment_refutation`'s own shape for the scalar case,
/// spelled through `format_temporal` for both sides.
pub(super) fn temporal_refutation(value_temporal: &TemporalAnnotation, declared: &DeclaredRefinement, declared_temporal: &TemporalAnnotation) -> String {
    format!(
        "a value of type '{}' is not assignable to type '{}' ({})",
        refined_sets::calendar_interpreter::format_temporal(value_temporal),
        declared.spelling,
        refined_sets::calendar_interpreter::format_temporal(declared_temporal),
    )
}

/// The undetermined sentence for a temporal `BoundsVerdict::Alert` —
/// `bounds_verdict_of`'s own `why` (a plain per-position reason: "the
/// spelling did not split," "the exact time of a zone-named spelling
/// needs the zone's data," …) reported as-is, matching the "every
/// silent row speaks a plain per-position sentence" doctrine.
pub(super) fn temporal_alert_sentence(why: &str) -> String {
    if why.is_empty() {
        SENTENCE.temporal_unprovable_instant.to_owned()
    } else {
        format!("Type not yet determined. Narrow type for safe type inference. {why}.")
    }
}
