use super::*;

#[test]
fn a_plain_alias_name_reads_with_admits_none_false() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&name_expr("Age"), &aliases, &imports, &environment)
        .expect("Age resolves");
    assert!(!got.admits_none);
}

#[test]
fn age_or_none_reads_age_with_admits_none_true() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&none_union(name_expr("Age")), &aliases, &imports, &environment)
        .expect("Age | None resolves");
    assert_eq!(got.spelling, "Age");
    assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
    assert!(got.admits_none);
}

#[test]
fn none_or_age_reversed_reads_age_with_admits_none_true() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&union_none(name_expr("Age")), &aliases, &imports, &environment)
        .expect("None | Age resolves");
    assert_eq!(got.spelling, "Age");
    assert!(got.admits_none);
}

#[test]
fn optional_age_reads_age_with_admits_none_true() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&optional_of(name_expr("Age")), &aliases, &imports, &environment)
        .expect("Optional[Age] resolves");
    assert_eq!(got.spelling, "Age");
    assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
    assert!(got.admits_none);
}

#[test]
fn age_or_label_a_union_of_two_non_none_sets_still_declines_whole() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();

    let union = bin_or(name_expr("Age"), name_expr("Label"));
    let got = declared_refinement(&union, &aliases, &imports, &environment);
    assert!(got.is_none());
}

#[test]
fn none_or_none_declines_whole() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();

    let union = bin_or(none_literal_expr(), none_literal_expr());
    let got = declared_refinement(&union, &aliases, &imports, &environment);
    assert!(got.is_none());
}

/// `Annotated[int, Field(ge=0)] | None` — the recursion into the
/// non-None side of a `| None` union reaches an inline `Annotated`
/// form exactly as it would reach a bare alias name. The compiled
/// forms arrive in `surface::canonical_scalar_form_order`'s order
/// (rays, then `Integer`), not the source's own `int`-then-`ge`
/// reading order.
#[test]
fn annotated_or_none_reads_with_admits_none_true() {
    let module = ruff_python_parser::parse_module(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         x: Annotated[int, Field(ge=0)] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Annotated[int, Field(ge=0)] | None resolves");
    assert!(got.admits_none);
    assert_eq!(got.set, make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::integer()]));
}

/// `Optional[Annotated[int, Field(ge=0)]]` — the recursion into
/// `Optional[...]`'s inner expression reaches the same inline
/// `Annotated` form. The compiled forms arrive in
/// `surface::canonical_scalar_form_order`'s order (rays, then
/// `Integer`), not the source's own `int`-then-`ge` reading order.
#[test]
fn optional_of_annotated_reads_with_admits_none_true() {
    let module = ruff_python_parser::parse_module(
        "from pydantic import Field\n\
         from typing import Annotated, Optional\n\
         x: Optional[Annotated[int, Field(ge=0)]] = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Optional[Annotated[int, Field(ge=0)]] resolves");
    assert!(got.admits_none);
    assert_eq!(got.set, make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::integer()]));
}

/// `"Sequence[Age]"` — a quoted forward reference to
/// `collections.abc.Sequence`/`typing.Sequence`: the string re-parses
/// (the `Expr::StringLiteral` arm) to an ordinary `Sequence[Age]`
/// subscript, which reads the same one-element-slot shape `list[X]`/
/// `set[X]` already read, carrying `Age` as `element` rather than a
/// scalar `set`.
#[test]
fn quoted_sequence_of_age_reads_age_as_the_element() {
    let module = ruff_python_parser::parse_module("x: \"Sequence[Age]\" = None\n")
        .expect("test module parses")
        .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Sequence[Age] resolves");
    assert_eq!(got.spelling, "Sequence[Age]");
    let element = got.element.expect("Sequence carries an element refinement");
    assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
}
