use super::*;

// ── ASCII case-conjunction alphabet narrowing ─────────────────────

/// F2.fixed's own `str_len_fixed_inside` shape: `len(x) == 2 and
/// x.isascii() and x.isupper()` narrows `x` to exactly the `Code`
/// alias's own set — two ASCII upper-case letters.
#[test]
fn test_isascii_and_isupper_conjunction_narrows_to_the_ascii_upper_alphabet() {
    let environment = environment_with_bare_string("x");
    let Some(narrowed) = assumed("len(x) == 2 and x.isascii() and x.isupper()", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.kind, Kind::Set);
    let Some(kernel) = loaded_kernel() else { return };
    let code = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
        make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]),
        2,
        Some(2),
    )]);
    assert!(
        (kernel.scalar_subset)(&x.set, &code) && (kernel.scalar_subset)(&code, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        code
    );
}

/// A2.guard.eq's Python twin (A3.guard.eq's own `eq_inside`): after
/// `s == "AA"` holds on a bare-`str` Set-kind binding, `s` is
/// provably the exact word `"AA"` — pinning the EqSeq narrowing leaf
/// (`narrow_set_kind_names`'s own `condition_tree_of` → `NarrowTreeKind::
/// EqSeq`) against a MULTI-character literal specifically, the shape
/// `refined_kernel::wire_decode::decode_wire_set` could not read
/// before it grew a `"word"` arm (the kernel's own `narrow.lean::
/// tupleSet` wires a two-or-more-character equality claim as a bare
/// `Word` leaf, matching the Go checker's `FormWord`, which this
/// crate's `Form` enum had never carried) — the multi-char length is
/// what distinguishes this from `test_equality_against_literal_keeps_
/// only_that_value`'s existing single-VALUE numeric case, and from
/// `narrow_set_kind_names`'s existing set-kind tests, none of which
/// exercise a word two codepoints or longer.
#[test]
fn test_eq_seq_narrows_bare_string_to_exact_multi_char_word() {
    let environment = environment_with_bare_string("s");
    let Some(narrowed) = assumed("s == \"AA\"", environment, true) else {
        return;
    };
    let s = narrowed.read("s").expect("s still bound");
    assert_eq!(s.kind, Kind::Set);
    let Some(kernel) = loaded_kernel() else { return };
    let word = make_refined_set(vec![refined_sets::refinement_forms::word(&[65.0, 65.0])]);
    assert!(
        (kernel.seq_subset)(&s.set, &word) && (kernel.seq_subset)(&word, &s.set),
        "s.set = {:?}, want the exact word {:?}",
        s.set,
        word
    );
}

/// The lower-case twin: `x.isascii() and x.islower()` narrows to
/// `[0x61, 0x7A]` instead of `[0x41, 0x5A]`.
#[test]
fn test_isascii_and_islower_conjunction_narrows_to_the_ascii_lower_alphabet() {
    let environment = environment_with_bare_string("x");
    let Some(narrowed) = assumed("len(x) == 2 and x.isascii() and x.islower()", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let Some(kernel) = loaded_kernel() else { return };
    let lower = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
        make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]),
        2,
        Some(2),
    )]);
    assert!(
        (kernel.scalar_subset)(&x.set, &lower) && (kernel.scalar_subset)(&lower, &x.set),
        "x.set = {:?}, want the same set as {:?}",
        x.set,
        lower
    );
}

/// `x.isupper()` ALONE (no `x.isascii()` in the same conjunction)
/// narrows nothing — the module doc's own reason: `isupper()` alone
/// is pinned only against the full Unicode cased-character
/// categories, which reach far outside ASCII, so bounding it to
/// `[0x41, 0x5A]` without the `isascii()` co-occurrence would
/// overclaim.
#[test]
fn test_isupper_alone_narrows_nothing() {
    let environment = environment_with_bare_string("x");
    let Some(narrowed) = assumed("x.isupper()", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, strings(), "isupper() alone must not narrow the alphabet");
}

/// `x.isascii()` ALONE (no `isupper()`/`islower()` in the same
/// conjunction) narrows nothing through this leaf — `isascii()`
/// alone states a `[0x00, 0x7F]` bound, a different (wider) claim
/// this leaf does not build, matching the "only the conjunction"
/// scope this leaf's own doc states.
#[test]
fn test_isascii_alone_narrows_nothing_through_this_leaf() {
    let environment = environment_with_bare_string("x");
    let Some(narrowed) = assumed("x.isascii()", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.set, strings(), "isascii() alone narrows nothing through the case-conjunction leaf");
}

/// `x.isascii() and y.isupper()` — the two calls on DIFFERENT
/// receivers — narrows neither: the conjunction must name the SAME
/// place from both calls.
#[test]
fn test_isascii_and_isupper_on_different_names_narrows_neither() {
    let mut locally_bound = HashSet::new();
    locally_bound.insert("x".to_owned());
    locally_bound.insert("y".to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind("x", known_set(strings(), None, TrustProved, SetKindTag::None));
    environment.bind("y", known_set(strings(), None, TrustProved, SetKindTag::None));
    let Some(narrowed) = assumed("x.isascii() and y.isupper()", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    let y = narrowed.read("y").expect("y still bound");
    assert_eq!(x.set, strings(), "x's own alphabet must stay unnarrowed");
    assert_eq!(y.set, strings(), "y's own alphabet must stay unnarrowed");
}
