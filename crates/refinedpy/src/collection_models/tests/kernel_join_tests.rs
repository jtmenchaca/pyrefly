// Kernel-joined scalar sets read back as Kind::Values, and the
// joined-string-key subscript reads (`"age" if flag else "years"`'s
// own shape) that route through the fold.

use super::*;

// test_known_value_of_state_reads_a_union_of_singletons_back_as_values
// pins the conversion itself: a kernel `join_state` answer shaped
// `{40} ∪ {41}` (a right-fold Union of singleton OneOf forms — the
// exact shape `KnownState.join` in `known_state.lean` builds, and
// `wire_decode.rs`'s `union` arm decodes back verbatim, `a_`/`b` in
// call order) reads back as `Kind::Values{[40, 41], Some(Integer)}`
// when the caller states a shared Integer tag — the same richer
// shape `join_known`'s own same-tag arm would have built locally —
// never the poorer untagged `Kind::Set` the kernel's bare wire
// would otherwise force.
#[test]
fn test_known_value_of_state_reads_a_union_of_singletons_back_as_values() {
    let union_of_singletons = make_refined_set(vec![refined_sets::refinement_forms::union(
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[40.0])]),
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[41.0])]),
    )]);
    let state = KnownStateWire {
        top: false,
        set: union_of_singletons,
        undef: false,
        null: false,
        nan: false,
        thrown: false,
    };
    let got = known_value_of_state(&state, TrustProved, Some(PrimitiveKind::Integer))
        .expect("a flag-free state must convert");
    assert_eq!(got, known_values(vec![40.0, 41.0], PrimitiveKind::Integer, TrustProved));
}

// test_known_value_of_state_a_non_singleton_arm_stays_a_set pins the
// refusal half: a union with ONE arm that is not a singleton scalar
// (here, an unbounded `atLeast` range) is not an enumerable set of
// exact values, so the conversion must decline and the caller keeps
// the plain `Kind::Set` shape — never guessing values that are not
// actually there.
#[test]
fn test_known_value_of_state_a_non_singleton_arm_stays_a_set() {
    let union_with_a_range = make_refined_set(vec![refined_sets::refinement_forms::union(
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[40.0])]),
        make_refined_set(vec![at_least(0.0)]),
    )]);
    let state = KnownStateWire {
        top: false,
        set: union_with_a_range.clone(),
        undef: false,
        null: false,
        nan: false,
        thrown: false,
    };
    let got = known_value_of_state(&state, TrustProved, Some(PrimitiveKind::Integer))
        .expect("a flag-free state must convert");
    assert_eq!(got, known_set(union_with_a_range, None, TrustProved, SetKindTag::None));
}

fn joined_string_key(a: &str, b: &str) -> AbstractValue {
    // `key = "age" if flag else "years"`'s own shape: join_known of
    // two distinct multi-codepoint exact strings builds a Kind::Set
    // over the union of their string_tuple forms (lattice_operations
    // ::join_known's own tests pin this exact join path).
    refined_domain::lattice_operations::join_known(string(a), string(b))
}

#[test]
fn subscript_read_joined_string_key_both_present_answers_the_shared_value() {
    // {"age": 40, "years": 40} — both candidate keys map to the SAME
    // value, so the join of the two entries reads exactly 40.
    let built = dict_literal_value(
        &[Some(key("age")), Some(key("years"))],
        &[integer(40.0), integer(40.0)],
    );
    let joined_key = joined_string_key("age", "years");
    assert_eq!(subscript_read(&built, &joined_key), Some(integer(40.0)));
}

#[test]
fn subscript_read_joined_string_key_different_values_answers_their_join() {
    // {"age": 40, "years": 41} — the two candidate keys map to
    // DIFFERENT values, so the read answers the join of both (the
    // value the real subscription reads depends on which branch ran).
    let built = dict_literal_value(
        &[Some(key("age")), Some(key("years"))],
        &[integer(40.0), integer(41.0)],
    );
    let joined_key = joined_string_key("age", "years");
    let got = subscript_read(&built, &joined_key).expect("both candidate keys are present");
    assert_eq!(got, refined_domain::lattice_operations::join_known(integer(40.0), integer(41.0)));
}

/// `loaded_kernel` mirrors `assignability.rs`/`builtin_models.rs`'s
/// own test helper: a missing dylib artifact prints to stderr and
/// the caller returns early, never failing the run.
fn loaded_kernel() -> Option<std::sync::Arc<refined_kernel::kernel_interface::RefinedTSKernel>> {
    let path = refined_kernel::kernel_bridge::dylib_path();
    if !refined_kernel::kernel_bridge::kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(refined_kernel::kernel_bridge::load_kernel(&path).expect("load_kernel"))
}

#[test]
fn kernel_joined_set_agrees_with_join_known_on_two_scalar_sets() {
    // The shape `dict_key_set_read`'s fold hands `kernel_joined_set`:
    // two DIFFERENT known-Integer scalar sets, exactly the shape a
    // `{"age": 40, "years": 41}` read against a joined string key
    // builds (subscript_read_joined_string_key_different_values_
    // answers_their_join's own scenario, isolated to the fold step
    // alone). `load_kernel` adopts a process-wide singleton
    // (`kernel_bridge.rs`'s own doc), so `kernel_if_loaded` inside
    // `kernel_joined_set` sees the same instance this test loads.
    //
    // Compared by mutual SET CONTENT, not `AbstractValue::eq`: the
    // kernel's own wire carries no Python sort tag at all
    // (`lattice_conformance.rs`'s module doc), so `kernel_joined_set`
    // always answers a bare `Kind::Set`, while `join_known`'s own
    // same-Integer-tag arm keeps the answer `Kind::Values` tagged
    // Integer — two different SHAPES for the identical set {40, 41},
    // the same reason `lattice_conformance.rs`'s own `same_state`
    // compares by mutual `scalar_subset` rather than `==`.
    let Some(kernel) = loaded_kernel() else { return };
    let via_kernel = kernel_joined_set(integer(40.0), integer(41.0));
    let via_local = join_known(integer(40.0), integer(41.0));
    let kernel_set = set_of_known(&via_kernel).expect("kernel_joined_set answers a set-shaped value");
    let local_set = set_of_known(&via_local).expect("join_known(40, 41) answers a set-shaped value");
    assert!(
        (kernel.scalar_subset)(&kernel_set, &local_set) && (kernel.scalar_subset)(&local_set, &kernel_set),
        "kernel_joined_set(40, 41) = {via_kernel:?}, want the same set content as join_known's {via_local:?}"
    );
}

#[test]
fn kernel_joined_set_falls_back_to_join_known_on_a_non_set_shaped_operand() {
    // An Object-shaped operand converts through neither
    // `known_state_of` gate — the fold must fall back to the local
    // `join_known` rather than misreading `set_of_known`'s own
    // refusal as a kernel answer.
    let object_side = known_object(vec![], None, true, TrustProved, false);
    let via_fallback = kernel_joined_set(object_side.clone(), integer(41.0));
    let via_local = join_known(object_side, integer(41.0));
    assert_eq!(via_fallback, via_local);
}

#[test]
fn subscript_read_joined_string_key_one_candidate_missing_declines() {
    // {"age": 40} only — "years" names no entry, so the whole read
    // declines rather than guessing at the missing branch's value.
    let built = dict_literal_value(&[Some(key("age"))], &[integer(40.0)]);
    let joined_key = joined_string_key("age", "years");
    assert_eq!(subscript_read(&built, &joined_key), None);
}
