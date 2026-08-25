use super::*;

// --- callable_return_refinement ---

/// `Callable[[int], Age] | None` — b-body-expressions.py:38's own
/// shape but with a refined return (`Age`, `int` in the row) rather
/// than the row's plain `int`: the return reads through the
/// ordinary `declared_refinement` path, and the `| None` wrapper is
/// dropped from the RETURN refinement (it describes the variable,
/// not `R`) — `admits_none` on the answer is false.
#[test]
fn callable_return_reads_a_declared_alias_return_dropping_the_variable_none() {
    let module = ruff_python_parser::parse_module(
        "from typing import Callable\n\
         f: Callable[[int], Age] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = callable_return_refinement(annotation, &aliases, &imports, &environment)
        .expect("Callable[[int], Age] | None resolves a return refinement");
    assert!(!got.admits_none);
    assert_eq!(got.spelling, "Age");
    assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
}

/// b-body-expressions.py:38's EXACT shape:
/// `Callable[[int], int] | None` — the return has no refined alias,
/// so it falls back to the bare `int` base sort
/// (`summaries.rs::return_sort_fallback`'s own unbounded
/// whole-number ray), matching `call_optional`'s marker ("the
/// guarded call still admits a whole number outside the set").
#[test]
fn callable_return_falls_back_to_the_bare_int_base_sort() {
    let module = ruff_python_parser::parse_module(
        "from typing import Callable\n\
         maybe_next_year: Callable[[int], int] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = callable_return_refinement(annotation, &aliases, &imports, &environment)
        .expect("Callable[[int], int] | None falls back to the int base sort");
    assert!(!got.admits_none);
    assert_eq!(
        got.set,
        make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            at_least(f64::NEG_INFINITY)
        ])
    );
}

/// No `| None` wrapper at all — `Callable[[int], int]` reads the
/// same return refinement directly.
#[test]
fn callable_return_reads_without_the_none_wrapper() {
    let module = ruff_python_parser::parse_module(
        "from typing import Callable\n\
         f: Callable[[int], int] = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = callable_return_refinement(annotation, &aliases, &imports, &environment)
        .expect("Callable[[int], int] resolves");
    assert_eq!(
        got.set,
        make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            at_least(f64::NEG_INFINITY)
        ])
    );
}

/// A non-Callable annotation (a plain alias name) declines — this
/// reader is specific to the `Callable[...]` subscript shape.
#[test]
fn a_non_callable_annotation_declines() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();

    let got = callable_return_refinement(&name_expr("Age"), &aliases, &imports, &environment);
    assert!(got.is_none());
}
