use super::*;

// ── the SET channel ──────────────────────────────────────────────

/// `>` on a Set-kind binding intersects the kernel's claim into the
/// current set — `x > 0` on the unbounded integer ray narrows to
/// the open-above-zero integer ray, which the assignability law's
/// own containment ask would then judge against a declared window.
#[test]
fn test_set_kind_greater_than_literal_intersects_the_kernel_claim() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x > 0", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.kind, Kind::Set);
    assert_eq!(x.kind_tag, Some(PrimitiveKind::Integer));
    let Some(kernel) = loaded_kernel() else { return };
    let expected = make_refined_set(vec![integer(), refined_sets::refinement_forms::above(0.0)]);
    assert!(
        (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        expected
    );
}

/// `>=` mirrors the same leaf with the inclusive operator.
#[test]
fn test_set_kind_greater_than_or_equal_intersects() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x >= 0", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let expected = make_refined_set(vec![integer(), at_least(0.0)]);
    assert!(
        (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        expected
    );
}

/// `n - 1 >= 0` (B1.keep.join's own ternary guard, `n - 1 if n - 1 >=
/// 0 else 0`) narrows `n` ITSELF, not `n - 1` (which is not a place
/// this file's environment binds at all) — the affine-shift reading
/// `comparison_leaf_tree_of` folds before falling through to
/// `other_tree()`. `n - 1 >= 0` is exactly `n >= 1`, so this asks the
/// SAME question `test_set_kind_greater_than_or_equal_intersects`
/// asks with a literal `1` in place of `0`.
#[test]
fn test_set_kind_affine_shift_left_narrows_the_base_place() {
    let environment = environment_with_set("n", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("n - 1 >= 0", environment, true) else {
        return;
    };
    let n = narrowed.read("n").expect("n still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let expected = make_refined_set(vec![integer(), at_least(1.0)]);
    assert!(
        (kernel.scalar_subset)(&n.set, &expected) && (kernel.scalar_subset)(&expected, &n.set),
        "n.set = {:?}, want the same set as {:?}",
        n.set,
        expected
    );
}

/// The mirrored spelling, `0 <= n - 1` — the affine shift sits on the
/// RIGHT of the comparison, so the effective operator mirrors too
/// (the same `mirror_cmp_op` reading the bare-name arm already takes
/// for a literal-on-the-left comparison), landing the identical `n >=
/// 1` claim as the left-shifted spelling above.
#[test]
fn test_set_kind_affine_shift_right_narrows_the_base_place_with_mirrored_operator() {
    let environment = environment_with_set("n", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("0 <= n - 1", environment, true) else {
        return;
    };
    let n = narrowed.read("n").expect("n still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let expected = make_refined_set(vec![integer(), at_least(1.0)]);
    assert!(
        (kernel.scalar_subset)(&n.set, &expected) && (kernel.scalar_subset)(&expected, &n.set),
        "n.set = {:?}, want the same set as {:?}",
        n.set,
        expected
    );
}

/// `0 <= x <= 120` — b-body-expressions.py's `len_in_guard` shape
/// (~b:649) once `x` already carries the unbounded integer ray:
/// the chained comparison's `And` tree intersects BOTH bounds in
/// one kernel ask, landing the exact `[0, 120]` integer window
/// `Age` (the fixture's own declared alias) admits.
#[test]
fn test_set_kind_chained_comparison_intersects_both_bounds() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(
        (kernel.scalar_subset)(&x.set, &age) && (kernel.scalar_subset)(&age, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        age
    );
}

/// `x >= 0` alone (no upper bound) — b-body-expressions.py's
/// `guard_over_ceiling` shape (~b:656): the narrowed set is `[0,
/// ∞) ∩ ℤ`, which is NOT a subset of `Age`'s `[0, 120]` window (it
/// still admits 200) — this is the fixture's own marked fire,
/// proved here at the SET level (the assignability law's own
/// containment ask is what actually fires it at the sink; this
/// test pins that the narrowed set is exactly what admits the
/// ceiling violation, not something already tighter).
#[test]
fn test_set_kind_single_sided_bound_still_admits_the_ceiling() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x >= 0", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(
        !(kernel.scalar_subset)(&x.set, &age),
        "x.set = {:?} must still admit values above 120 (200, …)",
        x.set
    );
}

/// `and` composes two Set-kind leaves on the SAME name into one
/// tree, same as the chained-comparison test above but spelled as
/// an explicit `and`.
#[test]
fn test_set_kind_and_composes_both_leaves() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x >= 0 and x <= 120", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(
        (kernel.scalar_subset)(&x.set, &age) && (kernel.scalar_subset)(&age, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        age
    );
}

/// `not (x > 120)` on a Set-kind binding — De Morgan through `not`
/// folds to the kernel's own `whenFalse` claim for `x > 120`
/// (`¬(x > 120)` is `x <= 120`), landing the at-most-120 half of
/// the integer ray.
#[test]
fn test_set_kind_not_wrapped_comparison_uses_the_kernel_negation() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("not (x > 120)", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let expected = make_refined_set(vec![integer(), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(
        (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        expected
    );
}
