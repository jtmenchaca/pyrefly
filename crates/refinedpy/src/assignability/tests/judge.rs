//! The verdict/judge core: the reasoned-sentence wording every fire
//! carries (sort crossing, structural mismatch, arity, vacuous
//! contents), and NaN freedom against a declared numeric refinement.

use super::*;

// --- the REASONED SENTENCE: what the value is, what the sink
// requires. Every fire below is already pinned above for its
// VERDICT; these pin the WORDING the reader gets.

/// A refutation states the sink's own REQUIREMENT, not just its
/// name: `Age`'s bounds ride beside it, so a reader never opens the
/// alias to learn what it admits.
#[test]
fn a_refutation_spells_what_the_sink_requires_beside_its_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_values(vec![200.0], PrimitiveKind::Integer, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'200'"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.contains("120"), "names Age's own ceiling: {message}");
}

/// A SORT crossing states the reason in plain words — the value's
/// sort, the position's sort, and that no run reconciles them. The
/// Go twin's "— <said> is not allowed here" clause.
#[test]
fn a_float_into_an_int_sorted_alias_states_both_sorts_and_the_reason() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("not assignable"), "{message}");
    assert!(message.contains("the value is a float"), "{message}");
    assert!(message.contains("states an integer"), "{message}");
    assert!(message.contains("not allowed here"), "{message}");
}

/// The mirror direction: a number arriving where a string is stated
/// says so as a sort crossing, never as a bare "not assignable".
#[test]
fn a_number_into_a_string_ground_alias_states_the_sort_crossing() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = any_string_refinement();
    let value = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("not assignable"), "{message}");
    assert!(message.contains("the value is a number"), "{message}");
    assert!(message.contains("states a string"), "{message}");
    assert!(message.contains("not allowed here"), "{message}");
}

/// A structural mismatch (a dict where a scalar is stated) names
/// what the position states and why the value cannot sit there.
#[test]
fn a_dict_into_a_numeric_ground_alias_states_what_the_position_requires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = refined_domain::known_constructors::known_object(
        Vec::new(),
        Default::default(),
        false,
        TrustProved,
        false,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("the position states"), "{message}");
    assert!(message.contains("not allowed here"), "{message}");
}

/// A container declaration carries an EMPTY outer set, so the
/// requirement clause must not append a vacuous "(any value)" — the
/// name stands alone.
#[test]
fn a_container_declaration_names_itself_without_a_vacuous_contents_clause() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = dict_of_age_refinement();
    let value = refined_domain::abstract_value::null_value();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'dict[str, Age]'"), "{message}");
    assert!(!message.contains("any value"), "{message}");
}

/// An arity mismatch says how many elements arrived AND how many the
/// position states — the reader sees both counts.
#[test]
fn an_arity_mismatch_states_both_counts() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_label_tuple_refinement();
    let value = refined_domain::known_constructors::known_list(
        vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)],
        TrustProved,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("1 element"), "{message}");
    assert!(message.contains("states 2 element"), "{message}");
}

// --- NaN against a declared numeric refinement: a RefinedSet denotes
// a subset of the reals, and NaN is a member of no refined set —
// `foreign_edge.rs::nan_freedom_obstacle` states the same boundary
// ruling for a cross-language crossing.

/// `x: Age = float("nan")` (or `inf - inf`, `inf * 0`, `inf / inf` —
/// every arithmetic-layer producer of `Kind::NaN`) fires against a
/// bounded float declaration: NaN escapes every declared refinement
/// unconditionally, never gated on `declared.admits_none` or any
/// other declared-side field, because no spelling admits NaN into a
/// set (`PYREFLY-PYDANTIC-SURFACE.md`'s `allow_inf_nan` row: "honesty:
/// do not admit NaN into sets" — `typereading.rs` never compiles that
/// knob into `DeclaredRefinement`).
#[test]
fn a_provably_nan_value_into_a_bounded_float_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = refined_domain::abstract_value::nan_value();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("NaN"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.contains("is a member of no refined set"), "{message}");
}

/// A `possibly_nan`-wrapped in-window value judges its PRESENT side
/// through this same seam, exactly the way `Kind::PossiblyUndefined`
/// judges its own present side: the wrapper is not PROVABLY NaN (the
/// runtime value may be the real, in-range inner value), so it is
/// neither fired as NaN nor waved through as silent on the wrapper
/// alone — an in-window inner value under the wrapper is silent,
/// because the inner value itself sits inside Age's bounds.
#[test]
fn a_possibly_nan_wrapped_in_window_value_judges_its_inner_value_and_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let inner = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
    let value = refined_domain::abstract_value::possibly_nan(inner);
    assert_eq!(value.kind, refined_domain::abstract_value::Kind::PossiblyNaN);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// The mirror: a `possibly_nan`-wrapped OUT-OF-WINDOW inner value
/// fires on the inner value's own escape, not on the NaN wrapper —
/// the recursion judges exactly what the present side states.
#[test]
fn a_possibly_nan_wrapped_out_of_window_value_fires_on_the_inner_values_own_escape() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let inner = known_values(vec![200.0], PrimitiveKind::Integer, TrustProved);
    let value = refined_domain::abstract_value::possibly_nan(inner);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'200'"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
}
