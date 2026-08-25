use super::*;

// --- Aliased sequence carries the same window as the inline spelling ---

/// `boosted: Boosted` (`Boosted = Annotated[list[BoostedSample],
/// Field(min_length=1)]`, the exact shape
/// audio-level-reverse.py uses) seeds the IDENTICAL
/// `DeclaredRefinement` shape — same element set, same length
/// window, same `"list[…]"` spelling prefix — as the inline
/// `boosted: Annotated[list[BoostedSample], Field(min_length=1)]`
/// spelling. A BOUNDED element (`BoostedSample`, not bare `float`)
/// is deliberate: `check.rs::seed_parameters` only takes the
/// repetition-window branch when the element's own set is
/// non-empty, so this is the shape that actually exercises it. This
/// is the determination gap the reverse-crossing fixture surfaced
/// (ISSUES.md): before this fix, the alias table dropped the
/// container window and `element`/`element_length` came back `None`.
#[test]
fn an_aliased_sequence_parameter_seeds_the_same_shape_as_the_inline_spelling() {
    let module = ruff_python_parser::parse_module(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         BoostedSample = Annotated[float, Field(ge=-2.0, le=2.0)]\n\
         Boosted = Annotated[list[BoostedSample], Field(min_length=1)]\n\
         def boost_samples(boosted: Boosted) -> None: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let aliases = crate::surface::compile_aliases(&module);
    let environment = no_locals();

    let alias_annotation = name_expr("Boosted");
    let via_alias = declared_refinement(&alias_annotation, &aliases, &imports, &environment)
        .expect("Boosted resolves through the alias table");

    let inline_parsed = parse_expression("Annotated[list[BoostedSample], Field(min_length=1)]")
        .expect("inline annotation parses");
    let via_inline = declared_refinement(&inline_parsed.into_expr(), &aliases, &imports, &environment)
        .expect("the inline spelling resolves directly");

    assert_eq!(via_alias.spelling, via_inline.spelling);
    // The written element NAME, not its unpacked bounds — the alias
    // path must reconstruct "list[BoostedSample]", never
    // "list[>= -2 && <= 2]" (the gate finding this test caught).
    assert_eq!(via_alias.spelling, "list[BoostedSample]");
    assert_eq!(via_alias.element_length, via_inline.element_length);
    assert_eq!(via_alias.element_length, Some((1, None)));
    let alias_element = via_alias.element.expect("alias path carries an element");
    let inline_element = via_inline.element.expect("inline path carries an element");
    assert_eq!(alias_element.set, inline_element.set);
    assert_eq!(alias_element.spelling, inline_element.spelling);
    assert_eq!(alias_element.spelling, "BoostedSample");
    assert!(!alias_element.set.forms.is_empty(), "BoostedSample's element set carries its ge/le bound");
    assert!(via_alias.spelling.starts_with("list["));
}

/// All three alias spellings (`type X = ...`, `X = Annotated[...]`,
/// `X: TypeAlias = Annotated[...]`) seed the identical parameter
/// shape once read through `declared_refinement`.
#[test]
fn all_three_alias_spellings_seed_the_same_parameter_shape() {
    let sources = [
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Boosted = Annotated[list[float], Field(min_length=1)]\n",
        "from pydantic import Field\n\
         from typing import Annotated\n\
         Boosted = Annotated[list[float], Field(min_length=1)]\n",
        "from pydantic import Field\n\
         from typing import Annotated, TypeAlias\n\
         Boosted: TypeAlias = Annotated[list[float], Field(min_length=1)]\n",
    ];
    let environment = no_locals();
    let mut shapes = Vec::new();
    for source in sources {
        let module = ruff_python_parser::parse_module(source)
            .expect("test module parses")
            .into_syntax();
        let imports = crate::surface::surface_imports(&module);
        let aliases = crate::surface::compile_aliases(&module);
        let got = declared_refinement(&name_expr("Boosted"), &aliases, &imports, &environment)
            .expect("Boosted resolves for every spelling");
        shapes.push((got.spelling, got.element_length, got.element.map(|e| e.set)));
    }
    assert_eq!(shapes[0], shapes[1]);
    assert_eq!(shapes[1], shapes[2]);
}

/// A scalar alias (`Age`) sitting beside a sequence alias in the
/// same module is unaffected — it still resolves with no element/
/// length-window fields.
#[test]
fn a_scalar_alias_parameter_is_unaffected_by_the_sequence_carry() {
    let module = ruff_python_parser::parse_module(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Field(ge=0)]\n\
         Boosted = Annotated[list[float], Field(min_length=1)]\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let aliases = crate::surface::compile_aliases(&module);
    let environment = no_locals();

    let got = declared_refinement(&name_expr("Age"), &aliases, &imports, &environment)
        .expect("Age resolves");
    assert!(got.element.is_none());
    assert!(got.element_length.is_none());
    assert_eq!(got.spelling, "Age");
}
