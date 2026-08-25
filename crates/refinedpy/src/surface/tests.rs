use std::collections::HashMap;

use refined_sets::codepoint_sets::string_tuple;
use refined_sets::refinement_forms::{
    above, at_least, at_most, below, integer, make_refined_set, multiple_of, union,
};
use refined_sets::regex_compiler::format_grammar;
use ruff_python_ast::{Expr, ModModule};

use super::*;

fn parsed(source: &str) -> ModModule {
    ruff_python_parser::parse_module(source)
        .expect("fixture source parses")
        .into_syntax()
}

/// `Field as F` — the import alias is still recognized as `Field`.
#[test]
fn field_import_alias_recognized() {
    let module = parsed(
        "from pydantic import Field as F\n\
         from typing import Annotated\n\
         type Age = Annotated[int, F(ge=0)]\n",
    );
    let out = compile_aliases(&module);
    assert!(out.contains_key("Age"));
}

/// `import pydantic as p` + `p.Field(...)` — the module alias is
/// still recognized as the pydantic module.
#[test]
fn pydantic_module_alias_recognized() {
    let module = parsed(
        "import pydantic as p\n\
         from typing import Annotated\n\
         type Age = Annotated[int, p.Field(ge=0)]\n",
    );
    let out = compile_aliases(&module);
    assert!(out.contains_key("Age"));
}

/// A locally defined `Field` shadowing the name is never a pydantic
/// `Field` — no alias compiles.
#[test]
fn locally_defined_field_not_recognized() {
    let module = parsed(
        "from typing import Annotated\n\
         def Field(**kwargs):\n\
         \x20\x20\x20\x20pass\n\
         type Age = Annotated[int, Field(ge=0)]\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Age"));
}

/// `from mylib import Field` — a same-named import from any other
/// module is never recognized as pydantic's `Field`.
#[test]
fn field_from_other_module_not_recognized() {
    let module = parsed(
        "from mylib import Field\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Field(ge=0)]\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Age"));
}

fn parsed_expression(source: &str) -> Expr {
    ruff_python_parser::parse_expression(source)
        .expect("test source parses")
        .into_expr()
}

/// `2**53 + 2` folds to the same value its literal spelling
/// 9007199254740994 reads — the computed spelling of a bound is not
/// a different construct.
#[test]
fn literal_number_folds_constant_integer_arithmetic() {
    assert_eq!(literal_number(&parsed_expression("2**53 + 2")), Some(9007199254740994.0));
    assert_eq!(literal_number(&parsed_expression("2**31 - 1")), Some(2147483647.0));
    assert_eq!(literal_number(&parsed_expression("60 * 60")), Some(3600.0));
}

/// `2**53 + 1` has no exact f64 spelling, an i64-overflowing fold
/// has no exact value at all, and a division is not an operator the
/// fold reads — each declines rather than approximating.
#[test]
fn literal_number_declines_inexact_and_unread_folds() {
    assert_eq!(literal_number(&parsed_expression("2**53 + 1")), None);
    assert_eq!(literal_number(&parsed_expression("2**63")), None);
    assert_eq!(literal_number(&parsed_expression("10 / 2")), None);
}

/// The construct the ledger named: `Field(le=2**53 + 2)` compiles
/// where the identical literal spelling already did, and the
/// inexact `2**53 + 1` spelling declines the whole row rather than
/// rounding the bound.
#[test]
fn field_bound_from_constant_arithmetic_compiles() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Big = Annotated[int, Field(le=2**53 + 2)]\n\
         type Odd = Annotated[int, Field(le=2**53 + 1)]\n",
    );
    let out = compile_aliases(&module);
    assert!(out.contains_key("Big"));
    assert!(!out.contains_key("Odd"));
}

/// `Annotated` used bare, with no import naming it, is never
/// recognized.
#[test]
fn annotated_without_import_not_recognized() {
    let module = parsed(
        "from pydantic import Field\n\
         type Age = Annotated[int, Field(ge=0)]\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Age"));
}

/// `type Adult = Age` — alias-of-alias compiles both names to the
/// same set.
#[test]
fn alias_of_alias_compiles_both_names() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Field(ge=0)]\n\
         type Adult = Age\n",
    );
    let out = compile_aliases(&module);
    assert!(out.contains_key("Age"));
    assert!(out.contains_key("Adult"));
    assert_eq!(out.get("Age"), out.get("Adult"));
}

/// A bare scalar alias carries no container fields — `head`,
/// `element`, and `length_window` are all `None`.
#[test]
fn a_scalar_alias_carries_no_container_fields() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Field(ge=0)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Age").expect("Age compiles");
    assert!(compiled.head.is_none());
    assert!(compiled.element.is_none());
    assert!(compiled.length_window.is_none());
}

/// An anchored `pattern=r"^[0-9a-f]+$"` compiles — the alias's set
/// is exactly what `format_grammar` gives the same pattern string
/// directly, so a matching literal ("1a2b", o-file's in-set row)
/// and a non-matching one ("zz", the o-file's out-of-set row) judge
/// against the identical compiled set the standalone grammar
/// reader would give either literal.
#[test]
fn anchored_pattern_compiles_to_the_grammar_reader_own_set() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Hex = Annotated[str, Field(min_length=1, max_length=6, pattern=r\"^[0-9a-f]+$\")]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Hex").expect("Hex compiles");
    let direct = format_grammar("^[0-9a-f]+$", "");
    assert!(direct.ok);
    // the pattern conjunct is present verbatim in the compiled
    // alias's forms (matching o-file's "1a2b" is a hex string, "zz"
    // is not — both judge against this same conjunct at check time)
    assert!(
        compiled.set.forms.iter().any(|f| direct.set.forms.contains(f)),
        "the anchored pattern's own compiled form must appear in Hex's forms"
    );
}

/// `Timestamp`'s own shape (g-strings-and-formats.py): `pattern`
/// ALONE, no `min_length`/`max_length` — the ONE path that used to
/// leave the bare `str` base's own C* ground (`strings()`) stacked
/// beside the pattern's own compiled forms, unlike the length-window
/// branch below, which already strips it. A stray ground conjunct
/// blinds the kernel's aligned-segment pattern prover
/// (`alignedSegSubsetB`, `boundary/exports_sets.lean`) exactly the way
/// TS's own `.regex()` compilation already documents and strips
/// (`chain_method.go`'s `WithoutStringGround` call) — the compiled
/// alias must carry ONLY the pattern's own forms, matching
/// `format_grammar`'s own direct output exactly.
#[test]
fn pattern_only_alias_drops_the_redundant_string_ground() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Timestamp = Annotated[str, Field(pattern=r\"^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$\")]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Timestamp").expect("Timestamp compiles");
    let direct = format_grammar(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", "");
    assert!(direct.ok);
    assert_eq!(
        compiled.set.forms, direct.set.forms,
        "a pattern-only alias's compiled forms must be EXACTLY the grammar's own forms, with no \
         redundant C* ground riding alongside them"
    );
}

/// An unanchored `pattern=r"^id-"` (anchored only at the start, the
/// o-file's `Anchored` row) compiles to a set whose top-level shape
/// is the padded concatenation `format_grammar` gives that same
/// pattern directly (prefix, then any suffix) — not the exact
/// two-sided anchored shape.
#[test]
fn unanchored_pattern_pads_the_open_side() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Anchored = Annotated[str, Field(min_length=3, max_length=10, pattern=r\"^id-\")]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Anchored").expect("Anchored compiles");
    let direct = format_grammar("^id-", "");
    assert!(direct.ok);
    assert!(
        compiled.set.forms.iter().any(|f| direct.set.forms.contains(f)),
        "the unanchored pattern's own padded form must appear in Anchored's forms"
    );
}

/// A pattern `format_grammar` refuses (a backreference, which does
/// not denote a regular language) declines the WHOLE alias — no
/// partial set missing just the pattern conjunct.
#[test]
fn a_pattern_the_grammar_refuses_declines_the_whole_alias() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Bad = Annotated[str, Field(min_length=1, pattern=r\"(a)\\1\")]\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Bad"));
}

/// `min_length`/`max_length` on a `str` alias (the o-file's
/// `Handle` row) compile to ONE repetition window over the
/// codepoint ground — `as_repetition` reads the compiled set back
/// with the exact [lo, hi] the two kwargs stated.
#[test]
fn string_length_window_compiles_to_one_repetition_form() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Handle = Annotated[str, Field(min_length=2, max_length=6)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Handle").expect("Handle compiles");
    let read_back = refined_sets::repetition_window_forms::as_repetition(&compiled.set)
        .expect("a length-window-only str alias reads back as one repetition");
    assert_eq!(read_back.lo, 2);
    assert_eq!(read_back.hi, Some(6));
}

/// `min_length` with no `max_length` (an open ceiling) reads back
/// unbounded on the high side.
#[test]
fn string_min_length_alone_is_an_open_upper_bound() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type AtLeastTwo = Annotated[str, Field(min_length=2)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("AtLeastTwo").expect("AtLeastTwo compiles");
    let read_back = refined_sets::repetition_window_forms::as_repetition(&compiled.set)
        .expect("a min_length-only str alias reads back as one repetition");
    assert_eq!(read_back.lo, 2);
    assert_eq!(read_back.hi, None);
}

/// An unrecognized kwarg on a `str` alias (`json_schema_extra`,
/// never on the inert list and never a bound) declines the whole
/// alias — the same discipline as the existing int-sort test
/// `an_alias_the_table_cannot_lower_declines_whole` in check.rs.
#[test]
fn an_unrecognized_string_kwarg_declines_the_whole_alias() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Odd = Annotated[str, Field(min_length=1, json_schema_extra={})]\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Odd"));
}

// --- Literal alias / Literal-union alias (m-pydantic-schema.py's Pick/PickUnion) ---

/// `type Pick = Literal[10, 20, 30]` compiles to a `one_of` set over
/// exactly those three members.
#[test]
fn a_bare_int_literal_alias_compiles_to_one_of_its_members() {
    let module = parsed("from typing import Literal\ntype Pick = Literal[10, 20, 30]\n");
    let out = compile_aliases(&module);
    let compiled = out.get("Pick").expect("Pick compiles");
    assert_eq!(
        compiled.set,
        make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0, 30.0])])
    );
}

/// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]`
/// compiles to the union of the int-Literal's `one_of` and the
/// string-Literal's own tuple union.
#[test]
fn a_literal_union_alias_compiles_to_the_union_of_both_arms() {
    let module = parsed(
        "from typing import Literal\n\
         type PickUnion = Literal[10, 20, 30] | Literal[\"ten\", \"twenty\"]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("PickUnion").expect("PickUnion compiles");
    let int_arm = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0, 30.0])]);
    let string_arm = make_refined_set(vec![union(
        string_tuple("ten"),
        string_tuple("twenty"),
    )]);
    assert_eq!(compiled.set, make_refined_set(vec![union(int_arm, string_arm)]));
}

/// A union of a Literal arm and a non-Literal arm declines whole —
/// `literal_union_alias_set` only reads a TWO-Literal-arm union.
#[test]
fn a_literal_union_with_a_non_literal_arm_declines() {
    let module = parsed(
        "from typing import Literal\n\
         type Bad = Literal[10, 20] | int\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Bad"));
}

// --- annotated_types constructors (Ge/Gt/Le/Lt/MultipleOf/MinLen/MaxLen) ---

/// `Annotated[int, Ge(0), Le(120)]` compiles the same set
/// `Annotated[int, Field(ge=0, le=120)]` would — m-pydantic-schema.py's
/// `AgeAT` shape. The compiled forms arrive in
/// `canonical_scalar_form_order`'s order (rays, then `Integer`)
/// rather than the source's own `int`-then-`Ge`-then-`Le` reading
/// order.
#[test]
fn ge_and_le_constructors_compile_the_same_set_field_kwargs_would() {
    let module = parsed(
        "from annotated_types import Ge, Le\n\
         from typing import Annotated\n\
         type AgeAT = Annotated[int, Ge(0), Le(120)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("AgeAT").expect("AgeAT compiles");
    assert_eq!(
        compiled.set,
        make_refined_set(vec![at_least(0.0), at_most(120.0), integer()])
    );
}

/// `Annotated[str, MinLen(1), Field(max_length=8)]` — a `MinLen`
/// constructor and a `Field(max_length=...)` kwarg on the SAME alias
/// fold into one repetition window, m-pydantic-schema.py's `LabelAT`
/// shape.
#[test]
fn min_len_constructor_and_field_max_length_fold_into_one_window() {
    let module = parsed(
        "from annotated_types import MinLen\n\
         from pydantic import Field\n\
         from typing import Annotated\n\
         type LabelAT = Annotated[str, MinLen(1), Field(max_length=8)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("LabelAT").expect("LabelAT compiles");
    let read_back = refined_sets::repetition_window_forms::as_repetition(&compiled.set)
        .expect("MinLen + Field(max_length) folds to one repetition window");
    assert_eq!(read_back.lo, 1);
    assert_eq!(read_back.hi, Some(8));
}

/// `Gt`/`Lt`/`MultipleOf` each recognized by their own import
/// identity, matching `Field`'s `gt`/`lt`/`multiple_of` kwargs. The
/// compiled forms arrive in `canonical_scalar_form_order`'s own
/// order (rays, then `Integer`, then `MultipleOf`) rather than the
/// source's own `int`-then-`Gt`-then-`Lt`-then-`MultipleOf` reading
/// order.
#[test]
fn gt_lt_and_multiple_of_constructors_compile_the_matching_forms() {
    let module = parsed(
        "from annotated_types import Gt, Lt, MultipleOf\n\
         from typing import Annotated\n\
         type EvenAge = Annotated[int, Gt(0), Lt(120), MultipleOf(2)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("EvenAge").expect("EvenAge compiles");
    assert_eq!(
        compiled.set,
        make_refined_set(vec![above(0.0), below(120.0), integer(), multiple_of(2.0)])
    );
}

/// The cross-adapter battery's `numeric-window-int-multiple-of` row:
/// `Annotated[int, Field(ge=0, le=100, multiple_of=5)]` compiles its
/// four forms in `canonical_scalar_form_order`'s priority — rays
/// first (`atLeast`/`above`, `atMost`/`below`), then `Integer`, then
/// `MultipleOf` — matching the Go adapter's golden order rather than
/// the source's own `int`-then-`ge`-then-`le`-then-`multiple_of`
/// reading order.
#[test]
fn a_numeric_window_with_multiple_of_compiles_in_canonical_form_order() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Bounded = Annotated[int, Field(ge=0, le=100, multiple_of=5)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Bounded").expect("Bounded compiles");
    assert_eq!(
        compiled.set,
        make_refined_set(vec![at_least(0.0), at_most(100.0), integer(), multiple_of(5.0)])
    );
}

/// An `annotated_types` constructor imported from any OTHER module
/// (never `annotated_types` itself) is not recognized — the same
/// import-identity discipline `names_field` already holds for `Field`.
#[test]
fn an_annotated_types_name_from_another_module_is_not_recognized() {
    let module = parsed(
        "from mylib import Ge\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Ge(0)]\n",
    );
    let out = compile_aliases(&module);
    assert!(!out.contains_key("Age"));
}

// --- StrictInt base sort / strict_int_alias_names ---

/// `type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]`
/// compiles the identical integer-ground set a plain `int` base would
/// — strictness is not a SET fact, it is check.rs's own coercion-gate
/// consult against `strict_int_alias_names`.
#[test]
fn strict_int_base_compiles_the_same_set_a_plain_int_base_would() {
    let module = parsed(
        "from pydantic import Field, StrictInt\n\
         from typing import Annotated\n\
         type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]\n\
         type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
    );
    let out = compile_aliases(&module);
    assert_eq!(out.get("StrictAge"), out.get("LaxAge"));
}

/// `strict_int_alias_names` names exactly the `StrictInt`-based alias,
/// never the plain `int`-based one.
#[test]
fn strict_int_alias_names_names_only_the_strict_int_based_alias() {
    let module = parsed(
        "from pydantic import Field, StrictInt\n\
         from typing import Annotated\n\
         type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]\n\
         type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
    );
    let strict_names = strict_int_alias_names(&module);
    assert!(strict_names.contains("StrictAge"));
    assert!(!strict_names.contains("LaxAge"));
}

// --- Sequence alias container window (Boosted-shaped) ---

/// `Boosted = Annotated[list[float], Field(min_length=1)]` carries
/// its OWN length window — the alias table no longer drops it (the
/// determination gap the reverse-crossing fixture surfaced). The
/// bare `float` element resolves to the UNBOUNDED real ray
/// (`numbers()`, `typereading.rs::base_sort_return_refinement`'s own
/// set for a bare `float` — never the empty set), spelled `"float"`.
#[test]
fn a_sequence_alias_carries_its_own_length_window() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         Boosted = Annotated[list[float], Field(min_length=1)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Boosted").expect("Boosted compiles");
    assert_eq!(compiled.head, Some("list"));
    assert_eq!(compiled.length_window, Some((1, None)));
    let (element_set, element_spelling) = compiled.element.as_deref().expect("Boosted carries an element set");
    assert_eq!(element_spelling.as_str(), "float");
    assert!(!element_set.forms.is_empty(), "a bare float element carries the unbounded real ray");
}

/// The alias's compiled element set and spelling are IDENTICAL to
/// what a bare `float` parameter's own `DeclaredRefinement` would be
/// (`numbers()`, spelled `"float"`), and the container's own scalar
/// `set` field stays empty (the container states nothing itself,
/// the same convention `annotated_expression_set` keeps for the
/// inline `Annotated[list[X], …]` case).
#[test]
fn a_sequence_alias_element_matches_the_bare_element_sort_exactly() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         Boosted = Annotated[list[float], Field(min_length=1)]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Boosted").expect("Boosted compiles");
    assert!(compiled.set.forms.is_empty(), "the container's own set states nothing");
    let (element_set, element_spelling) = compiled.element.as_deref().expect("Boosted carries an element set");
    assert_eq!(element_set, &refined_sets::refinement_forms::numbers());
    assert_eq!(element_spelling.as_str(), "float");
}

/// A `min_length`+`max_length` sequence alias element resolving
/// through a NESTED alias name (`Boosted = Annotated[list[Age],
/// Field(min_length=1, max_length=4)]`) reads `Age`'s own compiled
/// set as the element, spelled `"Age"` — the WRITTEN name, not
/// `Age`'s own unpacked bound — exactly like `declared_refinement`'s
/// inline `list[Age]` arm does.
#[test]
fn a_sequence_alias_element_resolves_through_a_nested_alias_name() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Field(ge=0)]\n\
         Boosted = Annotated[list[Age], Field(min_length=1, max_length=4)]\n",
    );
    let out = compile_aliases(&module);
    let age = out.get("Age").expect("Age compiles").set.clone();
    let compiled = out.get("Boosted").expect("Boosted compiles");
    assert_eq!(compiled.head, Some("list"));
    assert_eq!(compiled.length_window, Some((1, Some(4))));
    let (element_set, element_spelling) = compiled.element.as_deref().expect("Boosted carries an element");
    assert_eq!(element_set, &age);
    assert_eq!(element_spelling.as_str(), "Age");
}

// --- Fixed-arity tuple alias positions (Color-shaped) ---

/// showcase.py's own `Color = tuple[Channel, Channel, Channel]` row:
/// a bare-RHS fixed-arity tuple alias (no outer `Annotated[...]`
/// wrapper) compiles a per-position table, one entry per slot, each
/// resolving `Channel`'s own compiled set — never the container's
/// own `set` field, which stays empty (the same "the container
/// itself states nothing" convention every other container alias
/// keeps), and never `head`/`element`/`length_window`, which have
/// no meaning for a fixed-arity shape.
#[test]
fn a_fixed_arity_tuple_alias_compiles_one_position_per_slot() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Channel = Annotated[int, Field(ge=0, le=255)]\n\
         type Color = tuple[Channel, Channel, Channel]\n",
    );
    let out = compile_aliases(&module);
    let channel = out.get("Channel").expect("Channel compiles").set.clone();
    let compiled = out.get("Color").expect("Color compiles");
    assert!(compiled.set.forms.is_empty(), "a fixed-arity tuple's own set states nothing");
    assert_eq!(compiled.head, None);
    assert_eq!(compiled.element, None);
    assert_eq!(compiled.length_window, None);
    let positions = compiled.positions.as_ref().expect("Color carries a per-position table");
    assert_eq!(positions.len(), 3);
    for (slot_set, slot_spelling) in positions {
        assert_eq!(slot_set, &channel);
        assert_eq!(slot_spelling.as_str(), "Channel");
    }
}

/// All three alias spellings — the 3.12 `type X = ...` statement,
/// the plain `X = Annotated[...]` assignment, and the `X: TypeAlias
/// = Annotated[...]` form — carry the IDENTICAL container window for
/// the same `list[float]`/`min_length=1` shape.
#[test]
fn all_three_alias_spellings_carry_the_identical_sequence_window() {
    let type_stmt = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Boosted = Annotated[list[float], Field(min_length=1)]\n",
    );
    let plain_assign = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         Boosted = Annotated[list[float], Field(min_length=1)]\n",
    );
    let type_alias_assign = parsed(
        "from pydantic import Field\n\
         from typing import Annotated, TypeAlias\n\
         Boosted: TypeAlias = Annotated[list[float], Field(min_length=1)]\n",
    );
    let from_type_stmt = compile_aliases(&type_stmt).get("Boosted").cloned().expect("type-stmt spelling compiles");
    let from_plain_assign = compile_aliases(&plain_assign).get("Boosted").cloned().expect("plain-assign spelling compiles");
    let from_type_alias_assign = compile_aliases(&type_alias_assign)
        .get("Boosted")
        .cloned()
        .expect("TypeAlias-annotated spelling compiles");
    assert_eq!(from_type_stmt, from_plain_assign);
    assert_eq!(from_plain_assign, from_type_alias_assign);
}

// --- Bare container alias, no outer Annotated[...] wrapper (showcase.py's Amounts-shaped) ---

/// `type Amounts = list[Annotated[float, Field(ge=0)]]` — a BARE
/// `list[...]` RHS with no outer `Annotated[...]` wrapper. The
/// element itself is `Annotated[float, Field(ge=0)]`, so it resolves
/// through `annotated_expression_set` (the non-container case) to
/// the `ge=0` float ray, spelled through `format_for_diagnostics` —
/// the same element reading `Boosted`'s own `list[float]` element
/// gets, but reached from a bare container RHS instead of a wrapped
/// one.
#[test]
fn a_bare_container_alias_with_an_annotated_element_compiles() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Amounts = list[Annotated[float, Field(ge=0)]]\n",
    );
    let out = compile_aliases(&module);
    let compiled = out.get("Amounts").expect("Amounts compiles");
    assert_eq!(compiled.head, Some("list"));
    assert!(compiled.set.forms.is_empty(), "the container's own set states nothing");
    assert!(compiled.length_window.is_none(), "a bare container RHS carries no Field(min_length=…) slot");
    let (element_set, _element_spelling) = compiled.element.as_deref().expect("Amounts carries an element set");
    let direct = annotated_expression_set(
        &parsed_expression("Annotated[float, Field(ge=0)]"),
        &surface_imports(&parsed("from pydantic import Field\nfrom typing import Annotated\n")),
        &HashMap::new(),
    )
    .expect("the element's own Annotated[...] spelling compiles directly")
    .0;
    assert_eq!(element_set, &direct);
}

/// `type Ints = list[int]` — the bare container's element is a plain
/// ground sort, not `Annotated[...]`-wrapped at all. Resolves through
/// `element_set_and_spelling_for_alias`'s own bare `int` fallback,
/// the same set `base_sort_return_refinement` gives `int` everywhere
/// else it is read: the whole-number ray, never the empty set.
#[test]
fn a_bare_container_alias_with_a_plain_ground_element_compiles() {
    let module = parsed("type Ints = list[int]\n");
    let out = compile_aliases(&module);
    let compiled = out.get("Ints").expect("Ints compiles");
    assert_eq!(compiled.head, Some("list"));
    assert!(compiled.length_window.is_none());
    let (element_set, element_spelling) = compiled.element.as_deref().expect("Ints carries an element set");
    assert_eq!(element_spelling.as_str(), "int");
    assert_eq!(
        element_set,
        &make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
    );
}

/// The `Annotated[...]`-wrapped spelling of the SAME container still
/// compiles identically to before this arm existed — widening
/// `compile_aliases`'s recognition to a bare RHS must not change what
/// an already-wrapped alias compiles to. `type Wrapped =
/// Annotated[list[Annotated[float, Field(ge=0)]], Field()]` (an inert
/// no-op outer `Field()`, since a container needs SOME metadata tuple
/// to spell as `Annotated[...]` at all) carries the identical head,
/// element set, and element spelling `Amounts`'s bare spelling gives
/// the same inner shape.
#[test]
fn the_annotated_wrapped_spelling_of_the_same_container_is_unaffected() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Amounts = list[Annotated[float, Field(ge=0)]]\n\
         type Wrapped = Annotated[list[Annotated[float, Field(ge=0)]], Field()]\n",
    );
    let out = compile_aliases(&module);
    let bare = out.get("Amounts").expect("Amounts compiles");
    let wrapped = out.get("Wrapped").expect("Wrapped compiles");
    assert_eq!(bare.head, wrapped.head);
    assert_eq!(bare.element, wrapped.element);
    assert_eq!(bare.length_window, wrapped.length_window);
}

/// A scalar alias (`Age`) is unaffected by the container carry — it
/// still compiles to a bare `RefinedSet` with no container fields,
/// exercised earlier by `a_scalar_alias_carries_no_container_fields`;
/// this variant additionally checks a scalar alias sitting BESIDE a
/// sequence alias in the same module does not pick up the other's
/// container fields by accident.
#[test]
fn a_scalar_alias_beside_a_sequence_alias_stays_unaffected() {
    let module = parsed(
        "from pydantic import Field\n\
         from typing import Annotated\n\
         type Age = Annotated[int, Field(ge=0)]\n\
         Boosted = Annotated[list[float], Field(min_length=1)]\n",
    );
    let out = compile_aliases(&module);
    let age = out.get("Age").expect("Age compiles");
    assert!(age.head.is_none());
    assert!(age.element.is_none());
    assert!(age.length_window.is_none());
    let boosted = out.get("Boosted").expect("Boosted compiles");
    assert_eq!(boosted.head, Some("list"));
}
