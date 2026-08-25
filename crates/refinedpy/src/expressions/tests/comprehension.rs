use super::*;

#[test]
fn test_list_comp_over_known_list() {
    let Some(value) = eval("[x for x in [1, 2, 3]]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

#[test]
fn test_list_comp_with_a_condition_filters_elements() {
    let Some(value) = eval("[x for x in [1, 2, 3, 4] if x > 2]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![4.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

#[test]
fn test_set_comp_and_generator_share_the_list_shape() {
    let Some(set_value) = eval("{x for x in [1, 2]}") else { return };
    assert_eq!(set_value.kind, Kind::List);
    let Some(gen_value) = eval("(x for x in [1, 2])") else { return };
    assert_eq!(gen_value.kind, Kind::List);
}

#[test]
fn test_dict_comp_over_known_list_with_string_keys() {
    let Some(value) = eval("{str(x): x for x in [1]}") else { return };
    // str(x) IS a modeled builtin call for a known Integer argument
    // (builtin_models::str_call — CPython's plain decimal spelling),
    // so the key expression is the known string "1" and the whole
    // comprehension builds a known dict, matching CPython's own
    // `{str(x): x for x in [1]}` == `{'1': 1}`
    assert_eq!(value.kind, Kind::Object);
}

#[test]
fn test_multiple_generator_clauses_decline() {
    let Some(value) = eval("[x for x in [1, 2] for y in [3, 4]]") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `{name: age for name, age in d.items()}` — a two-name tuple
/// target unpacking each `.items()` pair-list; the whole
/// comprehension re-builds the same dict.
#[test]
fn test_dict_comp_two_name_tuple_target_over_items() {
    let Some(value) = eval("{name: age for name, age in {\"ann\": 40, \"bea\": 41}.items()}") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.keys.len(), 2);
}

/// A list comprehension can ALSO use a two-name tuple target —
/// `[age for name, age in d.items()]` reads only the value half.
#[test]
fn test_list_comp_two_name_tuple_target_over_items() {
    let Some(value) = eval("[age for name, age in {\"ann\": 40}.items()]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items, vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)]);
}

// --- sum over a generator (a-statements.py's own generator_expression row) ---

#[test]
fn test_sum_over_generator_expression() {
    let Some(value) = eval("sum(age for age in [10, 20, 30])") else { return };
    assert_eq!(value.values, vec![60.0]);
}
