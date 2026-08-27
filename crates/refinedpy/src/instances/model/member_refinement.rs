//! A module-level class's own fields read as a `DeclaredRefinement`,
//! for the MEMBERS LAW to judge a constructed instance against.

use refined_sets::refinement_forms::make_refined_set;

use crate::typereading::{DeclaredRefinement, TypedDictMember};

use super::types::ClassModel;

/// A module-level class's own fields, wrapped as a `DeclaredRefinement`
/// with `members: Some(...)` — `typed_dict_return_refinement`'s exact
/// shape, built here instead from an already-built `ClassModel` rather
/// than a fresh scan, since a class MEMBER's own declared refinement
/// (`declared: Option<DeclaredRefinement>` per field) is already exactly
/// what `assignability::judge`'s MEMBERS LAW reads. A field whose own
/// annotation states no refinement (`declared: None` — an unrefined
/// `str`/`int`, or an annotation this table cannot read) is left out of
/// the member list entirely, matching `typed_dict_table`'s own
/// `let Some(declared) = ... else { continue }` convention: an absent
/// member states nothing the MEMBERS LAW judges, never a guessed set.
///
/// `pub`: `class_model_of`'s own field loop (below) is not the only
/// caller a bare CLASS-NAME annotation needs.
///
/// A STATEMENT-LEVEL construction (`return Person.model_validate({"age":
/// 200, ...})`, `m-pydantic-schema.py`'s own corpus shape) already fires
/// correctly with NO help from this function: `check.rs::sink_value`'s
/// law 3 (`construction_call_verdict`) surfaces `judge_construction`'s
/// own per-field fires directly at the statement sink, regardless of
/// whether `Person`'s bare-class-name RETURN annotation itself compiles
/// a `DeclaredRefinement` (`declared_refinement` never learns a class
/// name at all — only `typed_dict_return_refinement`'s narrower
/// `TypedDict`-only table, `typereading.rs`'s own doc). That is why the
/// corpus's key-by-key membership rows already measure clean without
/// this reader.
///
/// A construction NESTED inside a call ARGUMENT
/// (`record_vitals(Vitals(heart_rate=72, spo2=130))`, the showcase's own
/// row) is the one shape that still loses its fire: `check.rs::judge_
/// one_call_argument` evaluates each argument through plain
/// `evaluate_expression`, whose own same-module-construction arm
/// (`expressions.rs`) discards `judge_construction`'s fires by design —
/// "a construction's fires belong to whichever statement sink hosts
/// this call expression, not to this nested value read" — because
/// ordinarily SOME enclosing sink (a return, an assignment) already
/// re-fires them through `sink_value`. An argument position is not such
/// a sink today: it neither calls `sink_value` NOR re-derives a
/// `members`-carrying `DeclaredRefinement` for a bare class-name
/// parameter to judge the constructed instance against afterward. THIS
/// function is exactly the second route — the caller can build `v`'s
/// own `DeclaredRefinement` from `context.classes.get("Vitals")` and let
/// `assignability::judge`'s MEMBERS LAW re-judge the already-built
/// instance — but surfacing `construction_call_verdict`'s fires the same
/// way `sink_value` already does for its OWN argument-evaluating step is
/// the more direct fix, since it reuses a verdict already computed
/// correctly rather than re-deriving one. Either fix lands in
/// `check.rs`, not here; this function is exported so whichever route is
/// chosen has the member-table reader ready.
pub fn model_members_refinement(model: &ClassModel) -> DeclaredRefinement {
    let members: Vec<TypedDictMember> = model
        .fields
        .iter()
        .filter_map(|field| {
            field.declared.clone().map(|declared| TypedDictMember {
                name: field.name.clone(),
                // `required` is a TypedDict TOTALITY fact, and an ordinary
                // class declaration states no totality — a field's presence
                // on an instance is settled by construction, not by this
                // table. `false` keeps the MEMBERS LAW's absent-key arm
                // silent for a class-derived member table, exactly as it
                // was before requiredness was recorded at all.
                required: false,
                declared,
            })
        })
        .collect();
    DeclaredRefinement {
        set: make_refined_set(Vec::new()),
        spelling: model.name.clone(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: Some(members),
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    }
}
