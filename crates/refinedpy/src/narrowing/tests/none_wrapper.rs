use super::*;

/// `sample is not None` proving TRUE against a `Kind::PossiblyUndefined`
/// binding (an `Optional[X]`-declared parameter's own seed,
/// `check.rs::seed_parameters`) unwraps to the wrapper's own INNER
/// value — the annotated set, never the wrapper itself.
#[test]
fn test_is_not_none_true_unwraps_a_possibly_undefined_binding() {
    use refined_domain::abstract_value::possibly_absent;
    use refined_domain::abstract_value::AbsentFlavor;

    let mut locally_bound = HashSet::new();
    locally_bound.insert("sample".to_owned());
    let mut environment = Environment::new(locally_bound);
    let inner = known_set(
        make_refined_set(vec![at_least(-2.0), refined_sets::refinement_forms::at_most(2.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    environment.bind("sample", possibly_absent(inner.clone(), AbsentFlavor::NullOnly, None, false));

    let Some(narrowed) = assumed("sample is not None", environment, true) else {
        return;
    };
    let value = narrowed.read("sample").expect("sample still bound");
    assert_eq!(value.kind, Kind::Set, "the wrapper must unwrap to its inner Kind::Set, not stay a maybe carrier");
    assert_eq!(value.set, inner.set);
}

/// The mirror: `sample is None` proving TRUE rebinds to the exact
/// `null_value` — the wrapper's absent side, matching what
/// `assignability::judge` reads directly for a bare `None`.
#[test]
fn test_is_none_true_rebinds_a_possibly_undefined_binding_to_null() {
    use refined_domain::abstract_value::possibly_absent;
    use refined_domain::abstract_value::AbsentFlavor;

    let mut locally_bound = HashSet::new();
    locally_bound.insert("sample".to_owned());
    let mut environment = Environment::new(locally_bound);
    let inner = known_set(
        make_refined_set(vec![at_least(-2.0), refined_sets::refinement_forms::at_most(2.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    environment.bind("sample", possibly_absent(inner, AbsentFlavor::NullOnly, None, false));

    let Some(narrowed) = assumed("sample is None", environment, true) else {
        return;
    };
    let value = narrowed.read("sample").expect("sample still bound");
    assert_eq!(value.kind, Kind::Null, "the wrapper must rebind to the exact null_value on the is-None-true fork");
}
