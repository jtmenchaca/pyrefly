use super::*;

// ── the ACCESS-PATH channel ──────────────────────────────────────

/// A15.guard.eq/A15.guard.ne's own shape: `0 <= a.n <= 150` narrows
/// the PATH `a.n`, not the bare name `a` (which this environment
/// never even binds a Values/Set fact for — `a` is a class-instance
/// receiver, not a number). `env::tracked_place_of`'s own chain
/// reading finds `a.n`, and `narrow_path_window` tightens the SAME
/// `{lo, hi}` window shape a length comparison already tightens,
/// seeded fresh from the unbounded integer ray on first touch.
#[test]
fn test_path_chained_comparison_narrows_the_attribute_chain() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut locally_bound = HashSet::new();
    locally_bound.insert("a".to_owned());
    let environment = Environment::new(locally_bound);
    let parsed = parse_expression("0 <= a.n <= 150").expect("test source must parse");
    let narrowed = assume(&parsed.into_expr(), environment, &kernel, true);
    let place = crate::env::TrackedPlace::bare("a").extend("n");
    let a_n = narrowed.read_path(&place).expect("a.n's own path fact is bound");
    assert_eq!(a_n.kind, Kind::Set);
    let expected = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(150.0)]);
    assert!(
        (kernel.scalar_subset)(&a_n.set, &expected) && (kernel.scalar_subset)(&expected, &a_n.set),
        "a.n's set = {:?}, want the same set as {:?}",
        a_n.set,
        expected
    );
}

/// A write to the base name forgets every path fact rooted at it
/// (`env::Environment::forget`'s own doc) — the one forget resolver
/// this channel relies on to never leave a stale `a.n` fact standing
/// once `a` itself is reassigned to a DIFFERENT instance.
#[test]
fn test_forgetting_the_base_name_drops_its_own_path_facts() {
    let mut environment = Environment::new(HashSet::new());
    let place = crate::env::TrackedPlace::bare("a").extend("n");
    environment.bind_path(&place, known_values(vec![40.0], PrimitiveKind::Integer, TrustProved));
    assert!(environment.read_path(&place).is_some());
    environment.forget("a");
    assert!(environment.read_path(&place).is_none(), "a write to the base must drop its path facts");
}

/// A write to a PREFIX of a deeper path forgets every path
/// continuing it, but leaves an unrelated sibling untouched
/// (`TrackedPlace::extends`'s own doc) — `a.n` write drops `a.n.x`,
/// never `a.m`.
#[test]
fn test_forgetting_a_path_prefix_drops_continuations_but_not_siblings() {
    let mut environment = Environment::new(HashSet::new());
    let a_n = crate::env::TrackedPlace::bare("a").extend("n");
    let a_n_x = a_n.extend("x");
    let a_m = crate::env::TrackedPlace::bare("a").extend("m");
    environment.bind_path(&a_n_x, known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
    environment.bind_path(&a_m, known_values(vec![2.0], PrimitiveKind::Integer, TrustProved));
    environment.forget_path_base(&a_n);
    assert!(environment.read_path(&a_n_x).is_none(), "a.n.x continues the written prefix a.n");
    assert!(environment.read_path(&a_m).is_some(), "a.m is an unrelated sibling of a.n");
}
