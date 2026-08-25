use super::*;

// --- return_sort_fallback: declined-call sort fallback ---
//
// A body `interpret_body` genuinely declines (a `while` loop, `**kwargs`/
// a keyword-only parameter, the depth cap, or an unbindable argument
// list — a `*args` parameter is NO LONGER one of these, see the
// `varargs_*` tests above) still states its return annotation's bare
// SORT rather than declining outright to `None` — item 1's own
// regression was never this fallback firing per se; it was the
// vararg/tuple-unpack/isinstance-narrowed bodies genuinely declining
// when they should have interpreted (or, for the vararg case,
// genuinely bound a known tuple). `for_over_unread_iterable`
// (a-statements.py) and `fstring_unread_substitution`
// (b-body-expressions.py) both lean on this fallback reaching a real
// sink and correctly FIRING there — see `loops.rs`'s own
// `iterable_values` doc and `expressions.rs`'s own `evaluate_fstring`
// doc for why a coarse sort-only claim is sound to flow all the way
// to a sink in those two cases (the checker's own admitted-coarse
// claim is what the row is testing, not a smuggled-in wrong answer).
#[test]
fn a_declined_while_loop_body_with_a_bare_int_return_annotation_answers_the_whole_number_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def counted(n) -> int:\n    while n > 0:\n        n -= 1\n    return n\n");
    let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0)
        .expect("the -> int annotation answers the whole-number set on a declined body");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// `-> float` reads through to the existing `float_sorted_unknown()`
/// shape — the same Float-tagged all-numbers set `math.sqrt` answers.
#[test]
fn a_declined_while_loop_body_with_a_bare_float_return_annotation_answers_float_sorted_unknown() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def counted(n) -> float:\n    while n > 0:\n        n -= 1\n    return n\n");
    let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0)
        .expect("the -> float annotation answers float_sorted_unknown on a declined body");
    assert_eq!(result, float_sorted_unknown());
}

/// A return annotation that is not a bare `int`/`float`/`str` name
/// (a compiled alias name, `Age`) still declines outright on a
/// genuinely-declining body when the CALLER's environment carries no
/// alias table (a plain `call_result` test, exactly like every test
/// above this one) — `declared_return_seed` requires `Environment::
/// declared_aliases`, which `fresh_body_environment` never populates
/// on its own; only `check.rs::walk_body_with_self_binding` does
/// (the alias-aware path is exercised below, through an environment
/// that DOES carry the table).
#[test]
fn a_declined_while_loop_body_with_a_non_base_sort_annotation_still_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def counted(n) -> Age:\n    while n > 0:\n        n -= 1\n    return n\n");
    assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
}

/// The depth cap's own decline point reaches the fallback too.
#[test]
fn the_depth_cap_decline_with_a_bare_int_return_annotation_answers_the_whole_number_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def double(x) -> int:\n    return x + x\n");
    let result = call_result(&def, &[known_int(3.0)], None, &kernel, CALL_DEPTH_CAP)
        .expect("the -> int annotation answers the whole-number set at the depth cap");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// The whole-number set genuinely admits a value the Age alias
/// refuses — this is the CONTAINMENT check `for_over_unread_iterable`
/// leans on: `whole_integers()` is not a subset of `Age`'s [0, 120]
/// window (it admits 200, 121, negative values, …), so `scalar_subset`
/// must answer false, matching `float_sorted_unknown`'s own sibling
/// test in refined_domain.
#[test]
fn whole_integers_is_not_a_subset_of_a_bounded_int_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let bounded = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(!(kernel.scalar_subset)(&whole_integers(), &bounded));
}
