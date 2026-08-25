use super::*;

/// `Literal[10, 20]` — a multi-member int Literal (ruff wraps the
/// slice in a `Tuple`) compiles to a `one_of` set over exactly those
/// two values, admitting neither.
#[test]
fn literal_of_two_ints_compiles_to_one_of_those_values() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[10, 20] = 10\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[10, 20] resolves");
    assert!(!got.admits_none);
    assert_eq!(
        got.set,
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0])])
    );
}

/// `Literal[40]` — a single-member Literal (no `Tuple` wrap) reads
/// the same way.
#[test]
fn literal_of_one_int_compiles_to_one_of_that_single_value() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[40] = 40\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[40] resolves");
    assert_eq!(
        got.set,
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[40.0])])
    );
}

/// `Literal[10, 20] | None` — composes with the existing
/// `admits_none` machinery for free: the union arm recurses into
/// this same Literal read, then marks `admits_none` true, exactly
/// as it does for an alias name or an inline `Annotated` form.
#[test]
fn literal_or_none_reads_the_literal_set_with_admits_none_true() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[10, 20] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[10, 20] | None resolves");
    assert!(got.admits_none);
    assert_eq!(
        got.set,
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0])])
    );
}

/// `Literal["horizontal", "vertical"]` — a multi-member STRING
/// Literal compiles to the UNION of each member's own singleton
/// string tuple (`string_literal_set`), the unambiguous form the
/// numeric `one_of` reader cannot share.
#[test]
fn literal_of_two_strings_compiles_to_the_union_of_their_tuples() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[\"horizontal\", \"vertical\"] = \"horizontal\"\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[\"horizontal\", \"vertical\"] resolves");
    assert!(!got.admits_none);
    assert_eq!(
        got.set,
        make_refined_set(vec![union(
            refined_sets::codepoint_sets::string_tuple("horizontal"),
            refined_sets::codepoint_sets::string_tuple("vertical"),
        )])
    );
}

/// `Literal["horizontal"]` — a single-member string Literal (no
/// `Tuple` wrap) reads as exactly that member's own tuple set, no
/// union node needed.
#[test]
fn literal_of_one_string_compiles_to_that_single_tuple() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[\"horizontal\"] = \"horizontal\"\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[\"horizontal\"] resolves");
    assert_eq!(got.set, refined_sets::codepoint_sets::string_tuple("horizontal"));
}

/// `Literal["horizontal", "vertical"] | None` — composes with the
/// existing `admits_none` machinery for free, the string Literal's
/// own twin of `literal_or_none_reads_the_literal_set_with_admits_none_true`.
#[test]
fn string_literal_or_none_reads_the_literal_set_with_admits_none_true() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[\"horizontal\", \"vertical\"] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[\"horizontal\", \"vertical\"] | None resolves");
    assert!(got.admits_none);
    assert_eq!(
        got.set,
        make_refined_set(vec![union(
            refined_sets::codepoint_sets::string_tuple("horizontal"),
            refined_sets::codepoint_sets::string_tuple("vertical"),
        )])
    );
}

/// A MIXED int/string `Literal[...]` member list declines whole:
/// neither `int_literal_members` (one member is a string) nor
/// `string_literal_members` (one member is an int) matches every
/// element, so no reading is built for either sort.
#[test]
fn a_mixed_int_and_string_literal_declines() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[40, \"horizontal\"] = 40\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment);
    assert!(got.is_none());
}

/// A negative int Literal member (`Literal[-1]`) reads through the
/// same unary-minus recognition `int_literal_value` shares with
/// `surface.rs::literal_number`.
#[test]
fn a_negative_int_literal_member_reads() {
    let module = ruff_python_parser::parse_module(
        "from typing import Literal\n\
         x: Literal[-1, 1] = -1\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Literal[-1, 1] resolves");
    assert_eq!(
        got.set,
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[-1.0, 1.0])])
    );
}
