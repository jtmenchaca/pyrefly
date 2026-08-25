//! Bindings: a body-local `AnnAssign` reusing an already-declared
//! alias's own `DeclaredRefinement` by spelling — scoped to a
//! matching alias, and never shadowing an already-declared name.

use super::*;

// --- body-local AnnAssign reuses an already-declared alias's own
// DeclaredRefinement by SPELLING (UNIT 4) ---

/// g-binding-destructuring.py:191-193's own shape: the for-target is
/// a TUPLE UNPACK (`for _, over_value in over_items:`), and the
/// body's first statement is an `AnnAssign` (`bad: Age = over_value`)
/// whose target was never bound before this loop — `declared` (the
/// pre-loop snapshot) has no entry for `bad`, only for `total` (an
/// EARLIER `total: Age = 0` in the same enclosing function). The
/// alias-spelling reuse must still fire the out-of-range write.
#[test]
fn body_local_ann_assign_reuses_an_alias_already_declared_under_a_different_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(concat!(
        "for _, over_value in over_items:\n",
        "    bad: Age = over_value\n",
    ));
    let mut environment = Environment::new(HashSet::from([
        "over_items".to_owned(),
        "_".to_owned(),
        "over_value".to_owned(),
        "bad".to_owned(),
    ]));
    let pairs = known_list(
        vec![
            known_list(vec![known_string("a"), integer(200.0)], TrustProved),
            known_list(vec![known_string("b"), integer(201.0)], TrustProved),
        ],
        TrustProved,
    );
    environment.bind("over_items", pairs);
    // `declared` carries Age only under "total" — "bad" is not a key
    // here at all, matching the pre-loop snapshot's real shape.
    let declared = declared_age("total");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the tuple-unpack target binds and the loop runs concretely");
    assert_eq!(out.len(), 1, "the 200/201 writes into Age must fire, deduped to one syntactic row: {out:?}");
    assert!(out[0].1.contains("Age"), "{}", out[0].1);
    let bad = answer.environment.read("bad").expect("bad stays bound to the declared set after the refused write");
    assert_eq!(bad.kind, Kind::Set);
}

/// The reuse is scoped to a MATCHING alias spelling only: a
/// body-local AnnAssign under an annotation that names NO alias
/// already present in `declared` stays unjudged, exactly as before —
/// this is not a general "annotation reading" fallback.
#[test]
fn body_local_ann_assign_under_an_unmatched_alias_stays_unjudged() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(concat!(
        "for x in [200]:\n",
        "    bad: Unrelated = x\n",
    ));
    let environment = Environment::new(HashSet::from(["x".to_owned(), "bad".to_owned()]));
    let declared = declared_age("total");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop still runs concretely — an unmatched annotation never declines it");
    assert!(out.is_empty(), "no declared entry matches 'Unrelated' by spelling, so nothing fires: {out:?}");
    assert_eq!(answer.environment.read("bad").unwrap().values, vec![200.0], "bad binds unjudged, unchanged from before this fix");
}

/// A body-local AnnAssign target that IS already a key in `declared`
/// (a name the pre-loop snapshot already recorded, then rewritten
/// with a fresh `x: Age = …` inside the SAME loop body) keeps reading
/// `declared`'s own entry — `newly_declared` never shadows it, since
/// `bind_checked` tries `declared` first.
#[test]
fn a_redeclared_name_already_in_declared_is_not_overridden_by_the_reuse_table() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(concat!("for x in [200]:\n", "    total: Age = x\n",));
    let mut environment = Environment::new(HashSet::from(["x".to_owned(), "total".to_owned()]));
    environment.bind("total", integer(0.0));
    let declared = declared_age("total");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop runs concretely");
    assert_eq!(out.len(), 1, "the redeclared write still fires against Age's own declared entry: {out:?}");
    let total = answer.environment.read("total").expect("total stays bound to the declared set");
    assert_eq!(total.kind, Kind::Set);
}
