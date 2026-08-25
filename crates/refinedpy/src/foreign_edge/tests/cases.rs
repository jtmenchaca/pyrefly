//! The cases-schema lowering: OBJECT return/entry cases, the
//! Result-shape multi-case return, float/integer return tagging,
//! and the OneOf-return sort read.

use super::*;

/// A single CLOSED, empty-member OBJECT return case binds through
/// `known_object` — pins the plainest object shape
/// (`foreign_case_value`'s own Object arm): `Kind::Object`, no keys,
/// `complete: true` straight from the case's own `closed`.
#[test]
fn a_closed_empty_object_return_case_binds_as_a_complete_object() {
    register_fixture_artifact("./audio_level.ts", audio_level_object_return_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Object);
            assert!(value.keys.is_empty());
            assert!(value.complete);
        }
        ForeignEdgeOutcome::Decline { message, .. } => {
            panic!("wanted an override binding the closed empty object, got a decline: {message}")
        }
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted an override binding the closed empty object, got a fire: {message}")
        }
    }
}

/// END-TO-END PIN: the Result-shape return (two OBJECT cases — `{ok,
/// value}` and `{ok, error}`) binds at `json.loads`, and a member
/// read (`parsed["value"]`, `collection_models.rs`'s `subscript_read`
/// own `Kind::Object` arm) reaches a judged verdict against a
/// declared window: the crossed `"value"` member's own number window
/// is `[0, 1]`, so asking the kernel whether it fits inside `[0, 2]`
/// answers true, and outside `[10, 20]` answers false — the
/// consumer-side judge running unchanged over a value this lane's
/// lowering produced.
#[test]
fn a_result_shape_return_binds_and_its_value_member_judges_against_a_declared_window() {
    register_fixture_artifact("./audio_level.ts", audio_level_result_shape_return_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let value = match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => value,
        ForeignEdgeOutcome::Decline { message, .. } => {
            panic!("wanted an override binding the Result-shape union, got a decline: {message}")
        }
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted an override binding the Result-shape union, got a fire: {message}")
        }
    };
    assert_eq!(value.kind, Kind::KindUnion);
    assert_eq!(value.arms.len(), 2);
    for arm in &value.arms {
        assert_eq!(arm.kind, Kind::Object);
        assert!(arm.complete);
    }
    let value_key = known_values(
        "value".chars().map(|c| c as u32 as f64).collect(),
        PrimitiveKind::String,
        TrustProved,
    );
    let value_member = subscript_read(&value.arms[0], &value_key).expect("the \"value\" member reads");
    assert_eq!(value_member.kind, Kind::Set);
    assert_eq!(value_member.kind_tag, Some(PrimitiveKind::Float));
    let inside_window = make_refined_set(vec![at_least(0.0), at_most(2.0)]);
    let outside_window = make_refined_set(vec![at_least(10.0), at_most(20.0)]);
    assert_eq!(foreign_scalar_subset(&kernel, &value_member.set, &inside_window), Some(true));
    assert_eq!(foreign_scalar_subset(&kernel, &value_member.set, &outside_window), Some(false));
}

/// An OBJECT case at the ENTRY (outbound) leg declines through the
/// existing "nothing says whether the value fits" sentence —
/// `admitted_set_of_cases` answers no-set for an Object case exactly
/// as it already does for Boolean/Null, so no new sentence is owed
/// at this leg.
#[test]
fn an_object_entry_case_declines_at_the_outbound_leg() {
    register_fixture_artifact("./audio_level.ts", audio_level_object_entry_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("audioLevel"), "{message}");
            assert!(message.contains("whether the value fits"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => {
            panic!("wanted the entry-leg decline, got an override binding an unlowered case")
        }
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted the entry-leg decline, got a fire: {message}")
        }
    }
}

/// DEFECT 2's fix: an unmarked, genuinely float-sorted return window
/// (no `Integer` form) still reads Float — the sibling row proving
/// the fix does not over-correct into tagging every crossed return
/// Integer.
#[test]
fn a_float_window_return_reads_float_tagged() {
    register_fixture_artifact("./audio_level.ts", audio_level_float_return_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// DEFECT 2's fix: an all-integer `OneOf` return (`{1, 2, 4}`, the
/// shape `union_levels.ts`'s derived Literal-set return carries,
/// f-value-unions.py's `louder_level_wider_window` pin) reads
/// Integer-tagged and passes an integer-window judge — the crossed
/// value's own sort read from the set, never a Float stamp.
#[test]
fn an_all_integer_one_of_return_reads_integer_and_fits_an_integer_window() {
    register_fixture_artifact("./audio_level.ts", audio_level_one_of_integer_return_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let value = match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => value,
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    // an integer-window judge: {1, 2, 4} subset-of [0, 10] ∧ integer
    let narrow_window = make_refined_set(vec![integer(), at_least(0.0), at_most(10.0)]);
    let fits = foreign_scalar_subset(&kernel, &value.set, &narrow_window);
    assert_eq!(fits, Some(true), "the all-integer OneOf return must fit an integer-window judge");
}
