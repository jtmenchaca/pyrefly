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

/// A8.seed.boundary's `query_param_inside` shape: a regex guard over
/// the SUBSCRIPT READ `v[0]` narrows that read's own place, so the
/// same read in the guarded branch answers the pattern's grammar
/// rather than the bare string ground. `v` is the star-of-strings
/// repetition a `parse_qs` value list seeds to, and `v[0]` draws `Σ*`
/// from it — the guard is what cuts that down to the two ASCII upper
/// letters the pattern spells.
#[test]
fn test_regex_guard_over_a_subscript_read_narrows_that_read_place() {
    let element = known_set(strings(), None, TrustProved, SetKindTag::None);
    let list = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element.set, 0, None)]);
    let environment = environment_with_set("v", list, PrimitiveKind::String);
    let Some(narrowed) = assumed("len(v) > 0 and re.fullmatch(r\"^[A-Z]{2}$\", v[0])", environment, true) else {
        return;
    };
    let place = crate::env::TrackedPlace::bare("v").extend_index("0");
    let read = narrowed.read_path(&place).expect("v[0]'s own place fact is bound");
    assert_eq!(read.kind, Kind::Set);
    let Some(kernel) = loaded_kernel() else { return };
    let code = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
        make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]),
        2,
        Some(2),
    )]);
    assert!(
        (kernel.seq_subset)(&read.set, &code),
        "v[0]'s set = {:?}, want it inside the guard's own grammar {:?}",
        read.set,
        code
    );
}

/// A COMPUTED index names no place (`env::tracked_place_of`'s own
/// doc): `v[i]` in the guard and `v[i]` in the branch can select
/// different elements, so a fact recorded at one would not be a fact
/// about the other. The guard narrows nothing rather than record one.
#[test]
fn test_regex_guard_over_a_computed_index_narrows_nothing() {
    let element = known_set(strings(), None, TrustProved, SetKindTag::None);
    let list = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element.set, 0, None)]);
    let mut environment = environment_with_set("v", list, PrimitiveKind::String);
    environment.bind("i", known_values(vec![0.0], PrimitiveKind::Integer, TrustProved));
    let Some(narrowed) = assumed("re.fullmatch(r\"^[A-Z]{2}$\", v[i])", environment, true) else {
        return;
    };
    let place = crate::env::TrackedPlace::bare("v").extend_index("0");
    assert!(narrowed.read_path(&place).is_none(), "a computed index names no place to record a fact at");
}

/// A write to the base drops the subscript read's own fact, the same
/// one forget resolver an attribute path already goes through — this
/// is what makes the narrowing hold only while `v` is genuinely
/// unwritten between the guard and the read.
#[test]
fn test_writing_the_base_drops_a_subscript_read_fact() {
    let mut environment = Environment::new(HashSet::new());
    let place = crate::env::TrackedPlace::bare("v").extend_index("0");
    environment.bind_path(&place, known_values(vec![65.0], PrimitiveKind::String, TrustProved));
    assert!(environment.read_path(&place).is_some());
    environment.forget("v");
    assert!(environment.read_path(&place).is_none(), "a write to v must drop v[0]'s own fact");
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
