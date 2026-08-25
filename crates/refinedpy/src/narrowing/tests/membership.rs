use super::*;

// ── `in` / `not in` membership ───────────────────────────────────

/// `x in [1, 2, 3]` on a Set-kind binding narrows to exactly those
/// three values: the `Or`-fold of the members' own `Eq` leaves has
/// the kernel union their singletons into the one-of set.
#[test]
fn test_in_a_literal_list_narrows_to_the_member_set() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x in [1, 2, 3]", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.kind, Kind::Set);
    let Some(kernel) = loaded_kernel() else { return };
    let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
    assert!(
        (kernel.scalar_subset)(&x.set, &members) && (kernel.scalar_subset)(&members, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        members
    );
}

/// A tuple and a set display are the same membership question as a
/// list — `x in (1, 2, 3)` narrows identically.
#[test]
fn test_in_a_literal_tuple_narrows_the_same_way() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x in (1, 2, 3)", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
    assert!(
        (kernel.scalar_subset)(&x.set, &members) && (kernel.scalar_subset)(&members, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        members
    );
}

/// The COMPLEMENT: `x in [1, 2, 3]` proving FALSE (and its `not in`
/// spelling proving true) leaves a set that still admits values
/// outside the list — and no longer admits the listed ones. Pinned
/// as "200 survives, 2 does not," the two facts the claim states.
#[test]
fn test_not_in_a_literal_list_drops_the_members_and_keeps_the_rest() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x not in [1, 2, 3]", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
    assert!(
        !(kernel.scalar_subset)(&x.set, &members),
        "x.set = {:?} must not be inside the very set it excludes",
        x.set
    );
    let two = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[2.0])]);
    assert!(
        !(kernel.scalar_subset)(&two, &x.set),
        "x.set = {:?} must no longer admit 2",
        x.set
    );
}

/// `not in` proving FALSE is membership again — the kernel's own
/// `Not` swaps the sides, so this lands the same one-of set the
/// plain `in` truth arm does.
#[test]
fn test_not_in_proving_false_is_membership() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x not in [1, 2, 3]", environment, false) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
    assert!(
        (kernel.scalar_subset)(&x.set, &members) && (kernel.scalar_subset)(&members, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        members
    );
}

/// A MIXED collection (a number beside a word) states nothing this
/// file lowers: the boundary refuses a tree mixing the numeric and
/// string worlds outright, so the leaf declines before asking and
/// the binding is left exactly as it was.
#[test]
fn test_in_a_mixed_sort_collection_narrows_nothing() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x in [1, \"two\"]", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, unbounded_integers());
}

/// A collection with a member this file cannot read as a literal (a
/// name) declines the whole leaf — never a partial reading of the
/// members it happened to recognize.
#[test]
fn test_in_a_collection_holding_a_name_narrows_nothing() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x in [1, some_name]", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, unbounded_integers());
}

/// A DICT display tests the dict's KEYS, a different collection from
/// the members a list display names — declined, narrowing nothing.
#[test]
fn test_in_a_dict_display_narrows_nothing() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x in {1: 'a', 2: 'b'}", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, unbounded_integers());
}

/// Membership with the PLACE on the collection side (`1 in x`) is a
/// different question entirely — it tests the place's own contents,
/// not its value — and narrows nothing.
#[test]
fn test_place_on_the_collection_side_narrows_nothing() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("1 in x", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, unbounded_integers());
}

/// A leaf this file cannot read at all (`x in y` — membership
/// against a collection that is not a literal display this file
/// reads) lowers to `other_tree()`; the whole tree says nothing
/// (`says_anything` false), so the binding is left exactly as it
/// was — never narrowed, never refused.
#[test]
fn test_an_unreadable_leaf_shape_leaves_the_set_binding_untouched() {
    let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
    let Some(narrowed) = assumed("x in y", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, unbounded_integers());
}
