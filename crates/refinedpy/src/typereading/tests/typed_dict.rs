use super::*;

#[test]
fn typed_dict_return_refinement_wraps_the_classs_own_member_table() {
    let mut typed_dicts = HashMap::new();
    let age_declared = DeclaredRefinement {
        set: make_refined_set(vec![at_least(0.0)]),
        spelling: "Age".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    };
    typed_dicts.insert("PersonDict".to_owned(), vec![("age".to_owned(), age_declared)]);

    let got = typed_dict_return_refinement(&name_expr("PersonDict"), &typed_dicts)
        .expect("a recorded TypedDict name resolves");
    assert_eq!(got.spelling, "PersonDict");
    let members = got.members.expect("members carries the per-field table");
    assert_eq!(members.len(), 1);
    assert_eq!(members[0].0, "age");
}

#[test]
fn typed_dict_return_refinement_declines_a_name_absent_from_the_table() {
    let typed_dicts: HashMap<String, Vec<(String, DeclaredRefinement)>> = HashMap::new();
    assert!(typed_dict_return_refinement(&name_expr("PersonDict"), &typed_dicts).is_none());
}
