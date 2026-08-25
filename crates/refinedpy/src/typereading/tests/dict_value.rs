use super::*;

// --- dict[str, X]'s value-slot reading ---

/// `dict[str, Age]` — a-statements.py's `return_dict_members` own
/// shape: the outer declaration carries no set of its own (`element`
/// Some, `set` empty) and the element is `Age` read through the
/// ordinary alias recursion.
#[test]
fn dict_of_str_to_age_reads_age_as_the_element() {
    let module = ruff_python_parser::parse_module(
        "x: dict[str, Age] = {}\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("dict[str, Age] resolves");
    assert!(!got.admits_none);
    assert_eq!(got.spelling, "dict[str, Age]");
    let element = got.element.expect("dict[str, Age] carries an element refinement");
    assert_eq!(element.spelling, "Age");
    assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
}

/// `dict[str, Age] | None` — composes with the existing
/// `admits_none` machinery for free: the union arm recurses into
/// this same dict read, then marks `admits_none` true, without
/// touching `element`.
#[test]
fn dict_of_str_to_age_or_none_reads_the_element_with_admits_none_true() {
    let module = ruff_python_parser::parse_module(
        "x: dict[str, Age] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("dict[str, Age] | None resolves");
    assert!(got.admits_none);
    assert_eq!(got.spelling, "dict[str, Age]");
    let element = got.element.expect("dict[str, Age] | None still carries an element refinement");
    assert_eq!(element.spelling, "Age");
}

/// `dict[int, Age]` — a non-`str` key declines the whole subscript,
/// same as any other unrecognized dict shape.
#[test]
fn dict_of_int_to_age_declines() {
    let module = ruff_python_parser::parse_module(
        "x: dict[int, Age] = {}\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment);
    assert!(got.is_none());
}

/// `dict[str, Unreadable]` — a value type this table cannot read
/// (no alias by that name) declines the whole subscript.
#[test]
fn dict_of_str_to_an_unreadable_value_type_declines() {
    let module = ruff_python_parser::parse_module(
        "x: dict[str, Unreadable] = {}\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment);
    assert!(got.is_none());
}
