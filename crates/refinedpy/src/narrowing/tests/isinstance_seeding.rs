use super::*;

/// `isinstance(value, int)` on a name the environment has bound
/// NOTHING for (the `object`-typed parameter shape,
/// b-body-expressions.py's `len_in_guard`/`guard_over_ceiling`)
/// seeds a fresh `Kind::Set` holding the unbounded integer ray —
/// the "(a seeded parameter, a sort-set)" case `assume`'s own
/// module doc names.
#[test]
fn test_isinstance_int_seeds_a_fresh_integer_set_from_unbound() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("isinstance seeded value");
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    assert_eq!(value.set, unbounded_integers());
}

/// A3.guard.sort's own `isinstance_str_outside` shape: `isinstance(x,
/// str)` on a name the environment has bound NOTHING for (an
/// `Any`/`object`-typed parameter) seeds a fresh `Kind::Set` holding
/// the whole-strings ground `strings()`, untagged — the string twin
/// of `test_isinstance_int_seeds_a_fresh_integer_set_from_unbound`.
/// Before `sort_seed`/`primitive_kind_of_type_name` grew a `String`
/// arm, `x` stayed unbound past this test entirely, so a later
/// `Code`-declared sink read `x` as never-narrowed and the checker
/// determined nothing (RTS7002) instead of refusing the unbounded
/// string ground against `Code`'s narrower [A-Z]{2} pattern.
#[test]
fn test_isinstance_str_seeds_a_fresh_string_set_from_unbound() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed("isinstance(value, str)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("isinstance seeded value");
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, None);
    assert_eq!(value.set, strings());
}

/// `isinstance(value, int)` on a name the environment HAS bound —
/// but to `Kind::Unknown` (a subscript into an unrecognized
/// container shape, `expressions.rs::evaluate_subscript`'s own
/// `unknown()` fallback for `parsed["value"]` over a `json.loads`
/// `Kind::KindUnion` result — `collection_models::subscript_read`
/// carries no `Kind::KindUnion` arm) — takes the SAME seeding path
/// the unbound case above does, not the "existing binding" arm: an
/// `Unknown` value states no information for the isinstance test to
/// filter, disagree with, or agree with, so a guard re-establishing
/// the sort over it is the honest reading, mirroring the e2e fixture
/// A10.edge.json's own `json_inside` row (`value = parsed["value"]`
/// guarded by `isinstance(value, int) and 0 <= value <= 150` before
/// `return value`).
#[test]
fn test_isinstance_int_seeds_a_fresh_integer_set_from_an_unknown_binding() {
    let mut locally_bound = HashSet::new();
    locally_bound.insert("value".to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind("value", refined_domain::abstract_value::unknown());
    let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("isinstance seeded value");
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    assert_eq!(value.set, unbounded_integers());
}

/// The same seeding applies to `Kind::Unknown` marked `opaque: true`
/// (`abstract_value::opaque`'s own "determined to be undeterminable"
/// shape, e.g. an external call's result) — both share `Kind::Unknown`,
/// so both carry zero information for this test to read.
#[test]
fn test_isinstance_int_seeds_a_fresh_integer_set_from_an_opaque_binding() {
    let mut locally_bound = HashSet::new();
    locally_bound.insert("value".to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind("value", refined_domain::abstract_value::opaque());
    let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("isinstance seeded value");
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    assert_eq!(value.set, unbounded_integers());
}

/// `isinstance(value, bool)` seeds `Kind::Values` over the two
/// exact booleans, never a `Kind::Set` — `bool`'s domain is exactly
/// `{0, 1}`.
#[test]
fn test_isinstance_bool_seeds_the_two_boolean_values_from_unbound() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed("isinstance(value, bool)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("isinstance seeded value");
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
    let mut values = value.values.clone();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![0.0, 1.0]);
}

/// `isinstance(value, float)` on a `Kind::KindUnion` binding (the
/// honest JSON-union `json.loads` answers over an opaque string,
/// `expressions.rs::json_loads_value_space`) keeps ONLY the
/// Float-tagged arm — the gain the ledger names: a downstream guard
/// must still narrow the union rather than reading it as
/// unnarrowable. Built inline here (rather than reaching into
/// `expressions.rs`'s private constructor) with the same seven arms
/// that function builds.
#[test]
fn test_isinstance_float_narrows_a_json_loads_union_to_its_float_arm() {
    use refined_domain::abstract_value::float_sorted_unknown;
    use refined_domain::abstract_value::null_value;
    use refined_domain::abstract_value::opaque_value;
    use refined_domain::abstract_value::AbstractValue;
    use refined_sets::codepoint_sets::strings;
    use refined_sets::refinement_forms::at_least;

    let integer_arm = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]), None, TrustProved, SetKindTag::None)
    };
    let float_arm = float_sorted_unknown();
    let union = kind_union_of(vec![
        null_value(),
        known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustProved),
        known_set(strings(), None, TrustProved, SetKindTag::None),
        integer_arm,
        float_arm.clone(),
        opaque_value("a list"),
        opaque_value("a dict"),
    ]);
    assert_eq!(union.kind, Kind::KindUnion, "the seven distinct-kind arms must not collapse");

    let mut locally_bound = HashSet::new();
    locally_bound.insert("value".to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind("value", union);

    let Some(narrowed) = assumed("isinstance(value, float)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("value still bound");
    assert_eq!(value.kind, Kind::Set, "only the float arm should survive, unwrapped from the union");
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    assert_eq!(value.set, float_arm.set);
}

/// `isinstance(value, int)` proving FALSE seeds nothing — a
/// falsified test says which sort `value` is NOT, never which sort
/// it IS.
#[test]
fn test_isinstance_proving_false_seeds_nothing() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed("isinstance(value, int)", environment, false) else {
        return;
    };
    assert!(narrowed.read("value").is_none());
}

/// The full `len_in_guard` guard
/// (`isinstance(value, int) and not isinstance(value, bool) and
/// 0 <= value <= 120`) run as ONE `assume` call: the isinstance
/// seed and the chained-comparison narrowing compose end to end,
/// landing the exact `[0, 120]` integer window `Age` admits.
#[test]
fn test_len_in_guard_shape_narrows_to_the_zero_to_120_integer_window() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed(
        "isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 120",
        environment,
        true,
    ) else {
        return;
    };
    let value = narrowed.read("value").expect("value bound by the guard");
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    let Some(kernel) = loaded_kernel() else { return };
    let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(
        (kernel.scalar_subset)(&value.set, &age) && (kernel.scalar_subset)(&age, &value.set),
        "value.set = {:?}, want the same set as {:?}",
        value.set,
        age
    );
}

/// The `guard_over_ceiling` shape — same guard, but only the
/// single-sided `value >= 0` bound: the narrowed set still admits
/// 200 (not a subset of `Age`), matching the fixture's own marked
/// "the guard does not bound the ceiling" fire.
#[test]
fn test_guard_over_ceiling_shape_still_admits_the_ceiling_violation() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed(
        "isinstance(value, int) and not isinstance(value, bool) and value >= 0",
        environment,
        true,
    ) else {
        return;
    };
    let value = narrowed.read("value").expect("value bound by the guard");
    let Some(kernel) = loaded_kernel() else { return };
    let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(
        !(kernel.scalar_subset)(&value.set, &age),
        "value.set = {:?} must still admit values above 120 (200, …)",
        value.set
    );
}
