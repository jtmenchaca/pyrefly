use super::*;

// --- Generator[Y, S, R] / AsyncGenerator[Y, S] / Iterator[Y] / Iterable[Y] ---

/// `Generator[Age, None, Age]` — i-more-expressions.py's own
/// `yield_expression` shape: both the yield and return positions
/// read `Age` through the ordinary alias recursion, and the outer
/// `set`/`element` fields stay empty/None the same way an
/// `element`-carrying container declaration does.
#[test]
fn generator_of_age_none_age_reads_both_positions_as_age() {
    let module = ruff_python_parser::parse_module(
        "from typing import Generator\n\
         def f() -> Generator[Age, None, Age]: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = def_return_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Generator[Age, None, Age] resolves");
    assert_eq!(got.spelling, "Generator[Age]");
    let generator = got.generator.expect("carries a generator refinement");
    assert_eq!(generator.yield_type.spelling, "Age");
    assert_eq!(generator.yield_type.set, make_refined_set(vec![at_least(0.0)]));
    let return_type = generator.return_type.expect("Generator's third argument states a return type");
    assert_eq!(return_type.spelling, "Age");
}

/// `Generator[int, None, None]` — a bare base-sort yield type falls
/// back to `base_sort_return_refinement`'s own unbounded whole-number
/// ray, matching `Callable[[...], R]`'s identical fallback: the
/// generator's own annotation is what makes a yield a checked
/// position, so a bare `int` argument must still state its ordinary
/// claim rather than silently declining the position.
#[test]
fn generator_of_bare_int_falls_back_to_the_int_base_sort() {
    let module = ruff_python_parser::parse_module(
        "from typing import Generator\n\
         def f() -> Generator[int, None, None]: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = def_return_annotation(&module);
    let aliases = HashMap::new();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Generator[int, None, None] resolves");
    let generator = got.generator.expect("carries a generator refinement");
    assert_eq!(
        generator.yield_type.set,
        make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            at_least(f64::NEG_INFINITY)
        ])
    );
    assert!(generator.return_type.is_none(), "a bare None third argument states no return type");
}

/// `AsyncGenerator[Age, None]` — the two-argument form: `yield_type`
/// reads `Age`, and `return_type` is always `None` (an async
/// generator cannot `return` a value).
#[test]
fn async_generator_of_age_none_reads_the_yield_position_only() {
    let module = ruff_python_parser::parse_module(
        "from typing import AsyncGenerator\n\
         async def f() -> AsyncGenerator[Age, None]: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = def_return_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("AsyncGenerator[Age, None] resolves");
    let generator = got.generator.expect("carries a generator refinement");
    assert_eq!(generator.yield_type.spelling, "Age");
    assert!(generator.return_type.is_none());
}

/// `Iterator[Age]` — the one-argument form: `yield_type` reads
/// `Age`, no `return_type` at all.
#[test]
fn iterator_of_age_reads_the_yield_position_only() {
    let module = ruff_python_parser::parse_module(
        "from typing import Iterator\n\
         def f() -> Iterator[Age]: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = def_return_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Iterator[Age] resolves");
    let generator = got.generator.expect("carries a generator refinement");
    assert_eq!(generator.yield_type.spelling, "Age");
    assert!(generator.return_type.is_none());
}

/// `Iterable[Age]` — `Iterator`'s twin, the same one-argument shape.
#[test]
fn iterable_of_age_reads_the_yield_position_only() {
    let module = ruff_python_parser::parse_module(
        "from typing import Iterable\n\
         def f() -> Iterable[Age]: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = def_return_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("Iterable[Age] resolves");
    let generator = got.generator.expect("carries a generator refinement");
    assert_eq!(generator.yield_type.spelling, "Age");
}

/// `Generator[Unreadable, None, Age]` — a yield type this table
/// cannot read declines the WHOLE subscript, the same all-or-nothing
/// rule `dict[str, Unreadable]` already takes for its own value slot.
#[test]
fn generator_with_an_unreadable_yield_type_declines() {
    let module = ruff_python_parser::parse_module(
        "from typing import Generator\n\
         def f() -> Generator[Unreadable, None, Age]: ...\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = def_return_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment);
    assert!(got.is_none());
}
