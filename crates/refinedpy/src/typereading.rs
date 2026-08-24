/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Annotation expressions read into declared refinements: alias names
//! (only where visible), inline `Annotated[...]` forms, alias-of-alias,
//! `X | None` / `Optional[X]` (admits-None wrapping the inner read),
//! and string annotations. This file is the contract the walk calls;
//! the typereading unit fills it in behind these signatures.
//!
//! A None never approximates a set — it declines to state one, and the
//! walk decides what silence says at the call site. Nothing here widens
//! or guesses a set it cannot read exactly (the same discipline as
//! refined-ts-go's typereading package: refuse rather than approximate).

use std::collections::HashMap;

use refined_sets::calendar_interpreter::format_temporal;
use refined_sets::calendar_interpreter::TemporalAnnotation;
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;
use ruff_python_parser::parse_expression;

use crate::env::Environment;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;
use crate::surface::annotated_expression_set;
use crate::surface::temporal_inline_annotation;

/// A refinement an annotation states, with the spelling diagnostics
/// use for it (the alias name, or the formatted set for inline forms).
#[derive(Clone)]
pub struct DeclaredRefinement {
    pub set: RefinedSet,
    pub spelling: String,
    /// true when the annotation admits None alongside the set (`X |
    /// None`, `Optional[X]`) — Kind::Null judges Silent against such a
    /// declaration; the set describes the non-None values.
    pub admits_none: bool,
    /// The per-member refinement of a container declaration —
    /// `dict[str, X]`'s VALUE slot; the outer `set` field is unused
    /// (empty) when `element` is Some.
    pub element: Option<Box<DeclaredRefinement>>,
    /// A `list[X]`/`set[X]`/`Sequence[X]` declaration's own SEQUENCE
    /// length bounds — `{lo, hi}`, `hi` `None` for unbounded — read from
    /// `Annotated[list[X], Field(min_length=…, max_length=…)]` (or the
    /// `annotated_types.MinLen`/`MaxLen` constructor spelling,
    /// `annotated_expression_set`'s own doc). `None` when the
    /// declaration states no length bound, OR the declaration is not a
    /// sequence container at all — always `None` unless `element` is
    /// also `Some` (the same "belongs to the container arm only"
    /// convention `element` itself keeps against `set`).
    /// `check.rs::seed_parameters` reads this to seed a bounded
    /// `repeat_of` in place of the unbounded `star` when present.
    pub element_length: Option<(i64, Option<i64>)>,
    /// A GENERATOR declaration's two checked positions —
    /// `Generator[YieldType, SendType, ReturnType]` /
    /// `AsyncGenerator[YieldType, SendType]` / `Iterator[YieldType]` /
    /// `Iterable[YieldType]` (typing / collections.abc). `yield_type` is
    /// what every `yield <expr>` in the body is judged against;
    /// `return_type` is what `return <expr>` is judged against —
    /// `Generator`'s third parameter, `None` for `AsyncGenerator`/
    /// `Iterator`/`Iterable` (an async generator cannot `return` a
    /// value; a plain `Iterator`/`Iterable` states no return type at
    /// all). `set`/`element` are unused (empty/None) when this is Some,
    /// the same "one active field" convention `element` already keeps
    /// with `set`.
    pub generator: Option<Box<GeneratorRefinement>>,
    /// A TypedDict declaration's own member table: each declared field's
    /// name paired with the refinement ITS OWN annotation states, in
    /// declaration order — `PersonDict`'s `age: Age` becomes
    /// `[("age", <Age's own DeclaredRefinement>)]`. Unlike `element`
    /// (`dict[str, X]`'s one refinement shared by every member), a
    /// TypedDict's members are HETEROGENEOUS by name, so this carries one
    /// refinement per field rather than one shared refinement. `set`/
    /// `element`/`generator` are unused (empty/None) when this is Some,
    /// the same "one active field" convention the other container shapes
    /// already keep.
    pub members: Option<Vec<(String, DeclaredRefinement)>>,
    /// A FIXED-ARITY tuple declaration's own per-position table —
    /// `tuple[int, int]`'s slot 0 and slot 1, each read through the
    /// ordinary `declared_refinement` recursion, in declaration order.
    /// Unlike `element` (one refinement shared by every position of a
    /// `list[X]`), a fixed-arity tuple's positions are each checked
    /// separately, so this carries one refinement per slot rather than
    /// one shared refinement — the same "heterogeneous, keyed by
    /// position instead of by name" shape `members` already carries for a
    /// TypedDict, keyed by index instead of by field name. `set`/
    /// `element`/`generator`/`members` are unused (empty/None) when this
    /// is Some, the same "one active field" convention every other
    /// container shape already keeps. `tuple[X, ...]` (a variadic tuple,
    /// the slice ending in a bare `...`) is a DIFFERENT shape this field
    /// does not carry — that subscript is read elsewhere or not at all.
    pub positions: Option<Vec<DeclaredRefinement>>,
    /// A `date`/`timedelta`/`datetime`/`AwareDatetime`/`NaiveDatetime`
    /// declaration's own calendar window — `surface::AliasEntry::
    /// temporal`'s exact twin, the same "one active field" convention
    /// every other container shape here already keeps: `set` carries
    /// nothing for a temporal declaration (a `Temporal*` value is never
    /// a member of a numeric/string `RefinedSet`). `None` for every
    /// non-temporal declaration.
    pub temporal: Option<TemporalAnnotation>,
    /// `surface::AliasEntry::temporal_awareness`'s exact twin — which
    /// of pydantic's aware/naive `datetime` bases `temporal` was read
    /// from, `Any` for a non-temporal declaration.
    pub temporal_awareness: crate::surface::TemporalAwareness,
}

/// The two checked positions a generator-shaped return annotation
/// states — see `DeclaredRefinement::generator`'s own doc.
#[derive(Clone)]
pub struct GeneratorRefinement {
    pub yield_type: DeclaredRefinement,
    pub return_type: Option<DeclaredRefinement>,
}

/// The refinement this annotation states here, or None when it states
/// none this table can read. A None never approximates — it declines.
pub fn declared_refinement(
    annotation: &Expr,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    match annotation {
        // A bare name: only a visible module-level alias states anything.
        // A name the body rebinds locally means something else here, so
        // the alias table must not be consulted for it — the walk owns
        // saying why a rebound name reads as unrefined.
        Expr::Name(name) => {
            let spelling = name.id.as_str();
            if !environment.alias_is_visible(spelling) {
                return None;
            }
            let entry = aliases.get(spelling)?;
            // A container alias (`Boosted = Annotated[list[BoostedSample],
            // Field(min_length=1)]`, `entry.element` Some) seeds the
            // IDENTICAL shape the inline `Annotated[list[X],
            // Field(min_length=…)]` arm below builds: the element's own
            // set wrapped as a scalar `DeclaredRefinement`, the
            // container's own length window, and a spelling of the same
            // `"list[…]"` shape (`entry.head` carries the head word) —
            // so `check.rs::seed_parameters`' `spelling.starts_with
            // ("list[")` gate (and its `set[`/`Sequence[` siblings) fires
            // identically whether the parameter names the alias or
            // spells the container inline. The element's spelling is its
            // OWN WRITTEN spelling (`entry.element`'s second tuple slot,
            // `surface::element_set_and_spelling_for_alias`'s own answer)
            // — never a reformatting of its resolved set — so
            // `list[BoostedSample]` reconstructs as `"list[BoostedSample]"`,
            // matching the inline path's own nested `declared_refinement`
            // recursion exactly (a bare alias name's spelling IS the name).
            let element = entry.element.as_ref().map(|element_entry| {
                let (element_set, element_spelling) = element_entry.as_ref();
                Box::new(DeclaredRefinement {
                    set: element_set.clone(),
                    spelling: element_spelling.clone(),
                    admits_none: false,
                    element: None,
                    element_length: None,
                    generator: None,
                    members: None,
                    positions: None,
                    temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                })
            });
            let container_spelling = match (entry.head, &element) {
                (Some(head), Some(element_declared)) => Some(format!("{}[{}]", head, element_declared.spelling)),
                _ => None,
            };
            // A FIXED-ARITY `tuple[X, Y, Z]` alias (`entry.positions`
            // Some, `surface::AliasEntry::positions`'s own doc — the
            // same "one active field" convention `element` keeps
            // against `positions`) seeds the IDENTICAL shape the inline
            // `Expr::Subscript` tuple arm below builds: one scalar
            // `DeclaredRefinement` per slot, each carrying the slot's
            // own WRITTEN spelling (never a reformatting of its
            // resolved set, the same fidelity `element`'s own spelling
            // keeps), so `tuple[Channel, Channel, Channel]` reconstructs
            // as `"tuple[Channel, Channel, Channel]"` whether the
            // parameter spells the tuple inline or names an alias of it.
            let positions = entry.positions.as_ref().map(|slots| {
                slots
                    .iter()
                    .map(|(slot_set, slot_spelling)| DeclaredRefinement {
                        set: slot_set.clone(),
                        spelling: slot_spelling.clone(),
                        admits_none: false,
                        element: None,
                        element_length: None,
                        generator: None,
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    })
                    .collect::<Vec<_>>()
            });
            let tuple_spelling = positions.as_ref().map(|slots| {
                format!(
                    "tuple[{}]",
                    slots.iter().map(|position| position.spelling.as_str()).collect::<Vec<_>>().join(", ")
                )
            });
            Some(DeclaredRefinement {
                set: entry.set.clone(),
                spelling: container_spelling.or(tuple_spelling).unwrap_or_else(|| spelling.to_owned()),
                // The alias's OWN admission (`type OptionalAge =
                // Optional[Age]`, `surface::peel_alias_optional`) —
                // never the element's; a container alias's admits_none
                // is a fact about the ALIAS NAME itself admitting None,
                // orthogonal to whether its element slot does.
                admits_none: entry.admits_none,
                element,
                element_length: entry.length_window,
                generator: None,
                members: None,
                positions,
                temporal: entry.temporal.clone(),
                temporal_awareness: entry.temporal_awareness,
            })
        }
        // `Optional[X]` reads X through the ordinary path and marks the
        // result as admitting None — "Optional[X] is equivalent to X |
        // None" (tmp/cpython Doc/library/typing.rst, "Optional").
        // Recognized by bare name only: `SurfaceImports` carries no
        // `typing.Optional` import identity today (only `Field` and
        // `Annotated`), so this matches the same bare-name convention
        // `annotated_expression_set` uses for `int`/`float`/`str` sort
        // names rather than gating on an identity this table cannot
        // check. `subscript.slice` is the single argument directly (not
        // a `Tuple` — ruff wraps in `Tuple` only for multi-element
        // subscripts like `Annotated[a, b]`). An inline `Annotated[...]`
        // form (the OTHER subscript head this table reads) is
        // unaffected: the surface unit owns recognizing and lowering it
        // against the module's import identities, and the spelling
        // there is the set itself, formatted, since no alias name
        // stands for it.
        Expr::Subscript(subscript) => {
            let is_optional = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Optional");
            if is_optional {
                let mut declared =
                    declared_refinement(subscript.slice.as_ref(), aliases, imports, environment)?;
                declared.admits_none = true;
                return Some(declared);
            }
            // `Literal[...]` (tmp/cpython Doc/library/typing.rst,
            // "Literal"): recognized by bare name only, the same
            // no-import-identity convention `Optional` above takes —
            // `SurfaceImports` carries no `typing.Literal` identity
            // today either. `subscript.slice` is either one bare member
            // (`Literal[40]`, no `Tuple` wrap — ruff only wraps a
            // MULTI-element subscript, the same rule
            // `annotated_expression_set` documents for `Annotated`) or
            // an `Expr::Tuple` of them (`Literal[10, 20]`). INT members
            // build a numeric `one_of` (`int_literal_members`); STRING
            // members build the union of each member's own singleton
            // string tuple (`string_literal_members` /
            // `string_literal_set`); BOOL members build the numeric
            // `one_of` over the boolean domain's two values
            // (`bool_literal_members` — True is 1, False is 0). The
            // wire shapes cannot share one reading (a string member's
            // code points would collide with `one_of`'s numeric
            // encoding), so each sort gets its OWN member reader and
            // only one may recognize a given member list; a MIXED-sort
            // `Literal[...]` matches no reader (every element of each
            // reader's map must be its own sort) and declines whole.
            // Any other member (a name, an expression, a float, a
            // bytes literal) declines every reader too, same as
            // `annotated_expression_set`'s own metadata gate.
            let is_literal = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Literal");
            if is_literal {
                if let Some(members) = int_literal_members(subscript.slice.as_ref()) {
                    let set = make_refined_set(vec![one_of(&members)]);
                    let spelling = format_for_diagnostics(&set);
                    return Some(DeclaredRefinement {
                        set,
                        spelling,
                        admits_none: false,
                        element: None,
                        element_length: None,
                        generator: None,
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
                if let Some(members) = string_literal_members(subscript.slice.as_ref()) {
                    let set = string_literal_set(&members);
                    let spelling = format_for_diagnostics(&set);
                    return Some(DeclaredRefinement {
                        set,
                        spelling,
                        admits_none: false,
                        element: None,
                        element_length: None,
                        generator: None,
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
                // BOOL members (`Literal[True]`/`Literal[False]`/both):
                // `bool`'s domain is the two exact values 0 and 1
                // (`string_models.rs`'s `boolean_value` convention, the
                // same reading `narrow_isinstance_call` seeds for
                // `isinstance(x, bool)`), so the members build the same
                // numeric `one_of` an int `Literal` does. The spelling
                // keeps the annotation's own words — `format_for_
                // diagnostics` would print the encoded numbers.
                if let Some(members) = bool_literal_members(subscript.slice.as_ref()) {
                    let set = make_refined_set(vec![one_of(&members)]);
                    let spelling = format!(
                        "Literal[{}]",
                        members.iter().map(|member| if *member == 1.0 { "True" } else { "False" }).collect::<Vec<_>>().join(", ")
                    );
                    return Some(DeclaredRefinement {
                        set,
                        spelling,
                        admits_none: false,
                        element: None,
                        element_length: None,
                        generator: None,
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
                return None;
            }
            // `dict[str, X]`: the container itself carries no scalar
            // set — its VALUE SLOT does. Recognized by bare-Name head
            // `dict` with a two-element `Tuple` slice (ruff always
            // wraps a multi-element subscript, the same rule
            // `Callable[[...], R]` above documents) whose FIRST member
            // is the bare name `str` (the only key sort this reader
            // states a member law for) and whose SECOND member reads
            // through the ordinary `declared_refinement` recursion —
            // so `dict[str, Age]` reads `Age` exactly, including a
            // nested `X | None` value slot. The value position ALSO
            // falls back to the bare `int`/`float`/`str` sort reading
            // (`base_sort_return_refinement`) when the ordinary alias
            // table has nothing for it — `dict[str, int]`'s value is
            // `int`, which is not an alias name — the SAME narrow
            // exception the `list[X]`/`set[X]`/`Sequence[X]` element arm
            // below already takes, scoped to this one call site so a
            // bare sort reaching here never turns an unrelated `-> int`
            // return into a fresh blocker. Any other shape (a non-`str`
            // key, an unreadable value type, no `Tuple` at all) declines
            // this arm and falls through to `annotated_expression_set`
            // below, which also declines (its own head-identity gate
            // never matches `dict`) — so the whole subscript states
            // nothing, as it did before this arm existed.
            let is_dict = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "dict");
            if is_dict {
                if let Expr::Tuple(arguments) = subscript.slice.as_ref() {
                    if let [key, value] = arguments.elts.as_slice() {
                        let key_is_str = matches!(key, Expr::Name(sort) if sort.id.as_str() == "str");
                        if key_is_str {
                            if let Some(value_declared) = declared_refinement(value, aliases, imports, environment)
                                .or_else(|| base_sort_return_refinement(value))
                            {
                                let spelling = format!("dict[str, {}]", value_declared.spelling);
                                return Some(DeclaredRefinement {
                                    set: make_refined_set(Vec::new()),
                                    spelling,
                                    admits_none: false,
                                    element: Some(Box::new(value_declared)),
                                    element_length: None,
                                    generator: None,
                                    members: None,
                                    positions: None,
                                    temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                                });
                            }
                        }
                    }
                }
                return None;
            }
            // `list[X]` / `set[X]` / `Sequence[X]` — the same
            // one-element-slot shape `dict[str, X]` reads for its VALUE
            // slot: the container itself carries no scalar set, its
            // ELEMENT does. The slice is the single element annotation
            // directly (no Tuple wrap for a one-argument subscript, the
            // same ruff rule the Optional arm above documents).
            // `Sequence` is `collections.abc.Sequence`/`typing.Sequence`
            // (tmp/cpython Doc/library/typing.rst: "typing.Sequence ...
            // Deprecated alias to collections.abc.Sequence"), a
            // read-only container with the same one-element-slot shape
            // `list`/`set` already read — recognized by bare name only,
            // the same no-import-identity convention this table already
            // takes for `Optional`/`Literal`/`Callable`.
            let is_element_container = matches!(
                subscript.value.as_ref(),
                Expr::Name(head) if head.id.as_str() == "list" || head.id.as_str() == "set" || head.id.as_str() == "Sequence"
            );
            if is_element_container {
                let head = match subscript.value.as_ref() {
                    Expr::Name(head) => head.id.as_str(),
                    _ => unreachable!("matched Name above"),
                };
                // The element position ALSO falls back to the bare
                // `int`/`float`/`str` sort reading (`base_sort_return_
                // refinement`) when the ordinary alias table has nothing
                // for it — `list[int]`'s element is `int`, which is not
                // an alias name. Scoped to this one call site, the same
                // narrow exception `seed_parameters` already takes at the
                // top level: reading a base sort HERE only ever widens
                // what a container's OWN element states, never turns an
                // unrelated `-> int` return into a fresh blocker (the
                // general-table doc's own concern), because the sort
                // never reaches `declared_refinement`'s general recursion
                // except through this one element slot.
                if let Some(element_declared) = declared_refinement(subscript.slice.as_ref(), aliases, imports, environment)
                    .or_else(|| base_sort_return_refinement(subscript.slice.as_ref()))
                {
                    let spelling = format!("{}[{}]", head, element_declared.spelling);
                    return Some(DeclaredRefinement {
                        set: make_refined_set(Vec::new()),
                        spelling,
                        admits_none: false,
                        element: Some(Box::new(element_declared)),
                        element_length: None,
                        generator: None,
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
                return None;
            }
            // `tuple[X, Y, ...]` FIXED-ARITY (every slot a concrete type,
            // no trailing bare `...`) — a known-length positional shape,
            // unlike `list[X]`'s one-element-slot shape above: slot `i`
            // carries POSITION `i`'s own declared refinement, not one
            // refinement shared by every slot. Recognized by bare-Name
            // head `tuple` (no `SurfaceImports` identity for it either,
            // the same no-import-identity convention this table already
            // takes) with a `Tuple` slice (ruff wraps a multi-element
            // subscript, the same rule `dict[str, X]` above documents; a
            // one-element `tuple[X]` has no `Tuple` wrap and is read as a
            // single-position tuple below). `tuple[X, ...]` (a VARIADIC
            // tuple, the slice ending in a bare `Expr::EllipsisLiteral`)
            // is a different, unbounded-length shape this arm does not
            // recognize — it declines here and falls through to
            // `annotated_expression_set` below, which also declines,
            // leaving that shape's own reading (if any) to a different
            // unit. Every position must itself read through the ordinary
            // `declared_refinement` recursion, falling back to the same
            // bare `int`/`float`/`str` sort reading the element-slot arm
            // above already takes (`tuple[int, int]`'s own slots are not
            // alias names) — any position that does not read declines the
            // WHOLE tuple, the same all-or-nothing rule `dict[str,
            // Unreadable]` already takes for its own value slot.
            let is_tuple = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "tuple");
            if is_tuple {
                let slots: Option<Vec<&Expr>> = match subscript.slice.as_ref() {
                    Expr::Tuple(arguments) => {
                        if arguments.elts.iter().any(|element| matches!(element, Expr::EllipsisLiteral(_))) {
                            None
                        } else {
                            Some(arguments.elts.iter().collect())
                        }
                    }
                    Expr::EllipsisLiteral(_) => None,
                    other => Some(vec![other]),
                };
                if let Some(slots) = slots {
                    let mut positions = Vec::with_capacity(slots.len());
                    for slot in slots {
                        let Some(slot_declared) = declared_refinement(slot, aliases, imports, environment)
                            .or_else(|| base_sort_return_refinement(slot))
                        else {
                            return None;
                        };
                        positions.push(slot_declared);
                    }
                    let spelling = format!(
                        "tuple[{}]",
                        positions.iter().map(|position| position.spelling.as_str()).collect::<Vec<_>>().join(", ")
                    );
                    return Some(DeclaredRefinement {
                        set: make_refined_set(Vec::new()),
                        spelling,
                        admits_none: false,
                        element: None,
                        element_length: None,
                        generator: None,
                        members: None,
                        positions: Some(positions),
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
            }
            // `Generator[YieldType, SendType, ReturnType]` /
            // `AsyncGenerator[YieldType, SendType]` / `Iterator[YieldType]`
            // / `Iterable[YieldType]` — the container itself carries no
            // scalar set; its two checked positions (`yield`/`return`
            // inside a generator body, `check.rs`'s own judging) live in
            // `generator`. Recognized by bare-Name head under either
            // `typing` or `collections.abc` (both modules export all four
            // names identically — `tmp/cpython Doc/library/typing.rst`,
            // "Generator"/"AsyncGenerator" — and this reader has no
            // `SurfaceImports` identity for any of them yet, the same
            // no-import-identity convention `Optional`/`Literal`/
            // `Callable` already take). The FIRST slice member is always
            // the yield type; `Generator`'s is a 3-tuple (ruff wraps a
            // multi-element subscript in a `Tuple`), the other three take
            // a single bare element (no `Tuple` wrap, one-argument
            // subscript). Any slice shape this arm cannot destructure, or
            // whose yield-type member does not itself read, declines the
            // whole subscript — no partial generator reading.
            let generator_head = match subscript.value.as_ref() {
                Expr::Name(head) => Some(head.id.as_str()),
                _ => None,
            };
            if let Some(head) = generator_head {
                if let Some(generator) =
                    generator_refinement(head, subscript.slice.as_ref(), aliases, imports, environment)
                {
                    let spelling = format!("{}[{}]", head, generator.yield_type.spelling);
                    return Some(DeclaredRefinement {
                        set: make_refined_set(Vec::new()),
                        spelling,
                        admits_none: false,
                        element: None,
                        element_length: None,
                        generator: Some(Box::new(generator)),
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
            }
            // `Annotated[list[X]|set[X]|Sequence[X], Field(min_length=…,
            // max_length=…)]` — the OUTER wrapper is `Annotated`, unlike
            // the bare `list[X]` the `is_element_container` arm above
            // reads, so this shape only ever reaches this wildcard
            // fallthrough. `annotated_expression_set`'s own container
            // recognition answers the length window (its second tuple
            // slot); the element itself is read HERE, through the
            // ordinary `declared_refinement` recursion (full alias
            // resolution, the same the `is_element_container` arm above
            // already gives a bare `list[X]`'s element), so `Sample` in
            // `list[Sample]` resolves the same way whether or not an
            // outer `Annotated[...]`/`Field(min_length=…)` wraps it.
            if let Some((head, element_expr)) = annotated_sequence_container(annotation, imports) {
                // `annotated_expression_set` only reads its own `aliases`
                // parameter through `element_container_element`, which
                // never dereferences it (that function's own doc) — a
                // scalar-set view of this table is enough to answer the
                // container/length-window question here.
                let sets_by_name: HashMap<String, RefinedSet> =
                    aliases.iter().map(|(name, entry)| (name.clone(), entry.set.clone())).collect();
                let (_container_set, length_window) = annotated_expression_set(annotation, imports, &sets_by_name)?;
                if let Some(element_declared) = declared_refinement(element_expr, aliases, imports, environment)
                    .or_else(|| base_sort_return_refinement(element_expr))
                {
                    let spelling = format!("{}[{}]", head, element_declared.spelling);
                    return Some(DeclaredRefinement {
                        set: make_refined_set(Vec::new()),
                        spelling,
                        admits_none: false,
                        element: Some(Box::new(element_declared)),
                        element_length: length_window,
                        generator: None,
                        members: None,
                        positions: None,
                        temporal: None,
                    temporal_awareness: crate::surface::TemporalAwareness::Any,
                    });
                }
                return None;
            }
            // A temporal base (`date`/`timedelta`/`datetime`, or
            // pydantic's `AwareDatetime`/`NaiveDatetime`) spelled INLINE
            // at the parameter — `surface::temporal_alias_annotation`'s
            // own recognition, reused here for the unaliased shape
            // (`d: Annotated[date, Field(ge=date(1900, 1, 1), …)]`,
            // showcase.py's own `DateOfBirth`/`Period`/`Visit`/
            // `FollowUp`/`Cutoff`/`Stamp` rows, whichever of those a
            // caller spells inline rather than through a module-level
            // alias). `environment`'s own module is not reachable from
            // here (typereading.rs never carries a `&ModModule` — every
            // caller passes only the annotation expression, `aliases`,
            // and `imports`), so a bare-Name bound (`Field(ge=_cutoff)`)
            // is not resolved at this inline call site the way
            // `compile_aliases`' own module-level scan resolves it; a
            // module-level `type Cutoff = Annotated[AwareDatetime,
            // Field(ge=_cutoff)]` alias (showcase.py's own spelling)
            // reads through the `Expr::Name` arm above instead, which
            // already carries `entry.temporal` resolved by
            // `compile_aliases`.
            if let Some((temporal, awareness)) = temporal_inline_annotation(annotation, imports) {
                let spelling = format_temporal(&temporal);
                return Some(DeclaredRefinement {
                    set: make_refined_set(Vec::new()),
                    spelling,
                    admits_none: false,
                    element: None,
                    element_length: None,
                    generator: None,
                    members: None,
                    positions: None,
                    temporal: Some(temporal),
                    temporal_awareness: awareness,
                });
            }
            let sets_by_name: HashMap<String, RefinedSet> =
                aliases.iter().map(|(name, entry)| (name.clone(), entry.set.clone())).collect();
            let (set, _length_window) = annotated_expression_set(annotation, imports, &sets_by_name)?;
            let spelling = format_for_diagnostics(&set);
            Some(DeclaredRefinement {
                set,
                spelling,
                admits_none: false,
                element: None,
                element_length: None,
                generator: None,
                members: None,
                positions: None,
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
            })
        }
        // `X | None` / `None | X` (exactly one side a bare `None`
        // literal): read the OTHER side through the ordinary path and
        // mark the result as admitting None — the `X | None` union
        // syntax (tmp/cpython Doc/library/stdtypes.rst, "Union Type",
        // `types.UnionType`). `None | None` and a union of two non-None
        // shapes (`Age | Label`) are a different unit — a general union
        // of two sets — and decline here exactly as before.
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                // both None, or neither — not this table's one-sided form
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            // `int | None` / `float | None` / `bool | None` / `str |
            // None`: the non-None side is a BARE SORT the general table
            // deliberately does not read — inside the one-sided union
            // the bare-sort seed applies, so the value reads as "whole
            // sort, absence admitted" instead of nothing at all.
            let mut declared = declared_refinement(other, aliases, imports, environment)
                .or_else(|| base_sort_return_refinement(other))?;
            declared.admits_none = true;
            Some(declared)
        }
        // A string annotation (a forward reference or a quoted form):
        // parse the string's own contents as an expression and recurse
        // on what it parses to. A string that fails to parse as an
        // expression states nothing this table can read.
        Expr::StringLiteral(literal) => {
            let source = literal.value.to_str();
            let parsed = parse_expression(source).ok()?;
            let inner = parsed.into_syntax().body;
            declared_refinement(&inner, aliases, imports, environment)
        }
        // Every other shape (a general union of two non-None sets,
        // attribute paths, calls, …) is not a form this table reads.
        // None here declines judgment; it is never read as "no
        // refinement applies".
        _ => None,
    }
}

/// Whether `annotation` is `Annotated[list[X]|set[X]|Sequence[X],
/// Field(…)|MinLen(…)|MaxLen(…)…]` — the OUTER `Annotated` wrapping a
/// container base, the shape a `list[X]` PARAMETER states its own
/// SEQUENCE length bound through (`declared_refinement`'s own wildcard
/// fallthrough doc). Recognized by the same `Annotated` import-identity
/// check `annotated_expression_set` itself takes, and the same bare-Name
/// `list`/`set`/`Sequence` head check `declared_refinement`'s own
/// `is_element_container` arm takes for the UNWRAPPED shape — this
/// function answers the head spelling and the unread element EXPRESSION
/// only; the length window itself is `annotated_expression_set`'s own
/// answer (its second tuple slot), and the element's own
/// `DeclaredRefinement` is read by the caller through the ordinary
/// `declared_refinement` recursion, never duplicated here.
fn annotated_sequence_container<'a>(annotation: &'a Expr, imports: &SurfaceImports) -> Option<(&'static str, &'a Expr)> {
    let Expr::Subscript(subscript) = annotation else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !imports.annotated_names.contains(head.id.as_str()) {
        return None;
    }
    let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
        return None;
    };
    let base = arguments.elts.first()?;
    let Expr::Subscript(base_subscript) = base else {
        return None;
    };
    let Expr::Name(base_head) = base_subscript.value.as_ref() else {
        return None;
    };
    let head_spelling = match base_head.id.as_str() {
        "list" => "list",
        "set" => "set",
        "Sequence" => "Sequence",
        _ => return None,
    };
    Some((head_spelling, base_subscript.slice.as_ref()))
}

/// A CALLABLE-VARIABLE's own RETURN refinement: `Callable[[...], R]`
/// (typing's `Callable`, tmp/cpython Doc/library/typing.rst,
/// "Callable" — `Callable[[int], str]` is "a function of (int) ->
/// str"), read the same bare-name-subscript-head way `Literal`/
/// `Optional` are above (no `SurfaceImports` identity for `Callable`
/// exists yet either), plus its `| None` wrapper (`X | None`/
/// `Optional[X]`). `admits_none` on the RETURNED `DeclaredRefinement`
/// here is never set true by the `| None` wrapper: that `None` means
/// the CALLABLE VARIABLE itself may be `None` (a fact `env.rs`'s
/// caller judges at the call site — a call through a possibly-None
/// callable additionally RAISES if the variable actually holds `None`,
/// out of this function's scope), not that `R` admits `None` — `R`'s
/// own refinement is read through the ordinary `declared_refinement`
/// path (so `Callable[[int], Age]` reads `Age` exactly, including ITS
/// own `admits_none` if `Age` were `Optional`), falling back to the
/// same bare `int`/`float`/`str` base-sort reading
/// `summaries.rs::return_sort_fallback` gives a declined call's return
/// annotation — matched here to the identical sets (`int` → the
/// unbounded whole-number ray, `float` → the unbounded real ray,
/// `str` → the whole-strings ground) so a callable-typed slot and an
/// ordinary same-module `def`'s declined body agree on what a bare
/// base-sort return annotation states.
pub fn callable_return_refinement(
    annotation: &Expr,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    match annotation {
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            // the variable's OWN possible-None-ness is not carried onto
            // its return refinement — see the doc comment above.
            callable_return_refinement(other, aliases, imports, environment)
        }
        Expr::Subscript(subscript) => {
            let is_callable = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Callable");
            if !is_callable {
                return None;
            }
            let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
                return None;
            };
            // `Callable[[params...], R]` — ruff always wraps the
            // two-element (params-list, return) slice in a Tuple; the
            // params element itself must be a `List` (an ellipsis
            // `Callable[..., R]` is a different, unparameterized shape
            // this reader does not recognize).
            let [params, returns] = arguments.elts.as_slice() else {
                return None;
            };
            if !matches!(params, Expr::List(_)) {
                return None;
            }
            declared_refinement(returns, aliases, imports, environment)
                .or_else(|| base_sort_return_refinement(returns))
        }
        _ => None,
    }
}

/// The bare `int`/`float`/`str`/`bool` return-annotation fallback,
/// matched to `summaries.rs::return_sort_fallback`'s own sets exactly:
/// `int` is the unbounded whole-number ray (`integer()` conjoined with
/// the unbounded `at_least(NEG_INFINITY)` ray, the same "no
/// ceiling/floor" shape that fallback builds), `float` is the unbounded
/// real ray (`numbers()`, the same set `float_sorted_unknown()`
/// carries), `str` is the whole-strings ground
/// (`codepoint_sets::strings()`), `bool` is the exact two-member domain
/// (`oneOf{0, 1}`, the boolean-domain convention).
/// EXPORTED for check.rs's parameter seeding ONLY: a bare-`int`
/// parameter seeds the whole-int sort claim ("a whole int admits
/// values outside the set", the corpus's own reason). The general
/// declared_refinement table deliberately does NOT read base sorts —
/// doing so made every `-> int` helper return judge, turning each
/// unreadable helper body into a new undetermined blocker.
pub fn base_sort_return_refinement(returns: &Expr) -> Option<DeclaredRefinement> {
    let Expr::Name(sort) = returns else {
        return None;
    };
    let set = match sort.id.as_str() {
        "int" => make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
        ]),
        "float" => refined_sets::refinement_forms::numbers(),
        "str" => refined_sets::codepoint_sets::strings(),
        // `bool`'s whole domain is the two exact values 0 and 1 (the
        // boolean-domain convention `bool_literal_members` and
        // `narrow_isinstance_call` both read), so a bare `bool`
        // parameter seeds the exact two-member set rather than a ray.
        "bool" => make_refined_set(vec![refined_sets::refinement_forms::one_of(&[0.0, 1.0])]),
        _ => return None,
    };
    let spelling = sort.id.as_str().to_owned();
    Some(DeclaredRefinement {
        set,
        spelling,
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    })
}

/// A bare-Name return/AnnAssign annotation naming a module-level
/// TypedDict class (`instances::typed_dict_table`'s own keys) —
/// `PersonDict`'s own per-member table, wrapped as a `DeclaredRefinement`
/// with `members: Some(...)` so `assignability::judge`'s MEMBERS law can
/// judge a dict literal against it field-by-field. `None` for anything
/// else (an alias name, a class that is not a recognized TypedDict, a
/// non-bare-Name annotation) — the ordinary `declared_refinement` path
/// already owns every other shape, and this function is a narrow
/// addition alongside it, not a replacement.
pub fn typed_dict_return_refinement(
    annotation: &Expr,
    typed_dicts: &HashMap<String, Vec<(String, DeclaredRefinement)>>,
) -> Option<DeclaredRefinement> {
    let Expr::Name(name) = annotation else {
        return None;
    };
    let members = typed_dicts.get(name.id.as_str())?;
    Some(DeclaredRefinement {
        set: make_refined_set(Vec::new()),
        spelling: name.id.as_str().to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: Some(members.clone()),
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    })
}

/// One generator-family subscript's slice read into its yield/return
/// checked positions — `declared_refinement`'s own generator arm. `head`
/// is the bare subscript-head name (`Generator`/`AsyncGenerator`/
/// `Iterator`/`Iterable`), already matched as a Name by the caller.
/// `Generator[Y, S, R]` reads a 3-element `Tuple` slice, yield type
/// first, return type third (the SEND type, second, states nothing this
/// reader judges — a generator's `.send()` argument is outside the
/// checker's scope); `AsyncGenerator[Y, S]` reads a 2-element `Tuple`
/// the same way but never carries a return type (datamodel.rst's
/// asynchronous generator functions cannot use a value-carrying
/// `return`); `Iterator[Y]`/`Iterable[Y]` read the single element
/// directly (no `Tuple` wrap for a one-argument subscript) with no
/// return type at all. Any other head name, or a slice shape that does
/// not match the head's own arity, declines — `None`, never a partial
/// reading.
///
/// Each position falls back to `base_sort_return_refinement` when
/// `declared_refinement` itself declines (a bare `int`/`float`/`str`
/// argument, e.g. `Generator[int, None, None]`) — the SAME fallback
/// `callable_return_refinement`'s own `R` position already takes, and
/// for the identical reason: the generator's own annotation is what
/// MAKES a yield/return a checked position in the first place (this
/// file's own module doc), so a bare base-sort argument here must
/// still state its ordinary whole-sort claim rather than silently
/// declining the position — unlike `declared_refinement`'s own general
/// table, which deliberately does NOT read base sorts for an ordinary
/// (non-generator) return annotation, to avoid turning every unrelated
/// `-> int` helper into a new blocker.
fn generator_refinement(
    head: &str,
    slice: &Expr,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<GeneratorRefinement> {
    let read_position = |argument: &Expr| -> Option<DeclaredRefinement> {
        declared_refinement(argument, aliases, imports, environment).or_else(|| base_sort_return_refinement(argument))
    };
    match head {
        "Generator" => {
            let Expr::Tuple(members) = slice else {
                return None;
            };
            let [yield_type, _send_type, return_type] = members.elts.as_slice() else {
                return None;
            };
            let yield_type = read_position(yield_type)?;
            let return_type = read_position(return_type);
            Some(GeneratorRefinement { yield_type, return_type })
        }
        "AsyncGenerator" => {
            let Expr::Tuple(members) = slice else {
                return None;
            };
            let [yield_type, _send_type] = members.elts.as_slice() else {
                return None;
            };
            let yield_type = read_position(yield_type)?;
            Some(GeneratorRefinement { yield_type, return_type: None })
        }
        "Iterator" | "Iterable" => {
            let yield_type = read_position(slice)?;
            Some(GeneratorRefinement { yield_type, return_type: None })
        }
        _ => None,
    }
}

/// `Literal[...]`'s slice read as a list of int-literal members: one
/// bare (possibly negated) `NumberLiteral` for a single-member
/// `Literal[40]`, or every element of an `Expr::Tuple` for
/// `Literal[10, 20]`. `None` the moment any member is not a plain int
/// literal (a string, a bool, a float, a name) — the whole subscript
/// declines rather than reading a partial member list.
fn int_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(int_literal_value).collect();
    }
    Some(vec![int_literal_value(slice)?])
}

/// `Literal[...]`'s slice read as a list of BOOL-literal members —
/// `True` encodes 1 and `False` 0, the boolean-domain convention. `None`
/// the moment any member is not a bare bool literal, the same
/// all-or-nothing rule the int and string readers keep.
fn bool_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    let bool_literal_value = |expr: &Expr| -> Option<f64> {
        match expr {
            Expr::BooleanLiteral(literal) => Some(if literal.value { 1.0 } else { 0.0 }),
            _ => None,
        }
    };
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(bool_literal_value).collect();
    }
    Some(vec![bool_literal_value(slice)?])
}

/// One `Literal[...]` member read as an int, with unary minus
/// (`Literal[-1]`) — the same shape `surface.rs::literal_number` reads,
/// but INTEGER ONLY: a `Number::Float` member declines, since a float
/// value can never be a `Literal[...]` member in the typing grammar
/// (`tmp/cpython Doc/library/typing.rst`, "Literal" — "Literal[3.14]" is
/// not a valid Literal parameter in the first place; only int, str,
/// bytes, bool, and None literals are).
fn int_literal_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => Some(-int_literal_value(unary.operand.as_ref())?),
        _ => None,
    }
}

/// `Literal[...]`'s slice read as a list of STRING-literal members —
/// `int_literal_members`'s twin. `None` the moment any member is not a
/// plain `Expr::StringLiteral` (an int, a bool, a name, an f-string) —
/// a MIXED int/string `Literal[...]` declines whole, the same
/// all-or-nothing rule `int_literal_members` already applies to a
/// mixed int/name member list.
fn string_literal_members(slice: &Expr) -> Option<Vec<String>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(string_literal_value).collect();
    }
    Some(vec![string_literal_value(slice)?])
}

/// One `Literal[...]` member read as a plain string literal — no
/// f-string, no concatenation, the same bare shape `int_literal_value`
/// reads on the numeric side.
fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
        _ => None,
    }
}

/// The UNION of every member's own singleton string tuple
/// (`codepoint_sets::string_tuple`) — the unambiguous string-Literal
/// wire shape `int_literal_members` cannot share: a string member's
/// code points would collide with `one_of`'s numeric encoding, so each
/// member gets its own tuple set and the members fold together by
/// `union`, not by one shared `one_of`. A single member is exactly its
/// own tuple set (no union node needed); `members` is never empty —
/// `string_literal_members` always returns at least one element when it
/// returns at all.
fn string_literal_set(members: &[String]) -> RefinedSet {
    let mut set = refined_sets::codepoint_sets::string_tuple(&members[0]);
    for member in &members[1..] {
        set = make_refined_set(vec![union(set, refined_sets::codepoint_sets::string_tuple(member))]);
    }
    set
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::at_most;
    use refined_sets::refinement_forms::make_refined_set;
    use ruff_python_ast::ExprContext;
    use ruff_python_ast::ExprName;
    use ruff_python_ast::ExprStringLiteral;
    use ruff_python_ast::StringLiteralValue;
    use ruff_text_size::TextRange;

    use super::*;

    fn name_expr(id: &str) -> Expr {
        Expr::Name(ExprName {
            node_index: Default::default(),
            range: TextRange::default(),
            id: id.into(),
            ctx: ExprContext::Load,
        })
    }

    fn string_literal_expr(text: &str) -> Expr {
        Expr::StringLiteral(ExprStringLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
            value: StringLiteralValue::single(ruff_python_ast::StringLiteral {
                range: TextRange::default(),
                node_index: Default::default(),
                value: text.into(),
                flags: ruff_python_ast::StringLiteralFlags::empty(),
            }),
        })
    }

    fn no_locals() -> Environment {
        Environment::new(HashSet::new())
    }

    /// SurfaceImports has private fields — tests build it through the
    /// public constructor on a parsed module with no imports.
    fn no_imports() -> SurfaceImports {
        let module = ruff_python_parser::parse_module("x = 1")
            .expect("test module parses")
            .into_syntax();
        crate::surface::surface_imports(&module)
    }

    #[test]
    fn a_visible_alias_name_resolves_with_its_name_as_spelling() {
        let mut aliases = HashMap::new();
        aliases.insert(
            "PositiveInt".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(1.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment)
            .expect("a visible alias resolves");
        assert_eq!(got.spelling, "PositiveInt");
        assert_eq!(got.set, make_refined_set(vec![at_least(1.0)]));
    }

    /// `list[int]`'s element position resolves through the bare-sort
    /// fallback: `int` is not a module-level alias, so without the
    /// `base_sort_return_refinement` fallback at this one call site the
    /// whole `list[int]` subscript declines (`f-type-nodes.py`'s
    /// `list_annotation_parameter` row, undetermined before this fix).
    #[test]
    fn list_of_a_bare_int_resolves_its_element_through_the_base_sort_fallback() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("list[int]").expect("test source must parse");
        let annotation = parsed.into_expr();

        let got = declared_refinement(&annotation, &aliases, &imports, &environment)
            .expect("list[int]'s element must resolve through the base-sort fallback");
        assert_eq!(got.spelling, "list[int]");
        let element = got.element.expect("list[X] carries its element, not a scalar set");
        assert_eq!(element.spelling, "int");
        assert!(!element.set.forms.is_empty(), "int's own set must not be empty");
    }

    /// `set[str]` and `Sequence[float]` take the identical fallback path
    /// — the same `is_element_container` arm, keyed only on the head
    /// name.
    #[test]
    fn set_and_sequence_of_a_bare_base_sort_also_resolve_their_element() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let set_parsed = parse_expression("set[str]").expect("test source must parse");
        let set_got = declared_refinement(&set_parsed.into_expr(), &aliases, &imports, &environment)
            .expect("set[str]'s element must resolve");
        assert_eq!(set_got.element.expect("element present").spelling, "str");

        let sequence_parsed = parse_expression("Sequence[float]").expect("test source must parse");
        let sequence_got = declared_refinement(&sequence_parsed.into_expr(), &aliases, &imports, &environment)
            .expect("Sequence[float]'s element must resolve");
        assert_eq!(sequence_got.element.expect("element present").spelling, "float");
    }

    /// `tuple[int, int]` — a FIXED-ARITY tuple of two bare base sorts:
    /// each position reads through the same base-sort fallback the
    /// element-container arm above takes, kept SEPARATE per position
    /// (unlike `list[int]`'s one shared element) — `c-reads-and-values.py`'s
    /// `ternary_spread_copies_optional_list` own parameter shape.
    #[test]
    fn fixed_arity_tuple_of_two_bare_ints_resolves_each_position_through_the_base_sort_fallback() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("tuple[int, int]").expect("test source must parse");

        let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
            .expect("tuple[int, int]'s positions must resolve through the base-sort fallback");
        assert_eq!(got.spelling, "tuple[int, int]");
        let positions = got.positions.expect("a fixed-arity tuple carries its positions, not a scalar set");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].spelling, "int");
        assert_eq!(positions[1].spelling, "int");
    }

    /// `tuple[Age, Label]` — mixed alias positions each read through the
    /// ordinary alias recursion, keeping their own distinct sets.
    #[test]
    fn fixed_arity_tuple_of_two_aliases_resolves_each_positions_own_set() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("tuple[Age, Label]").expect("test source must parse");

        let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
            .expect("tuple[Age, Label]'s positions must resolve");
        let positions = got.positions.expect("positions present");
        assert_eq!(positions[0].spelling, "Age");
        assert_eq!(positions[0].set, make_refined_set(vec![at_least(0.0)]));
        assert_eq!(positions[1].spelling, "Label");
        assert_eq!(positions[1].set, make_refined_set(vec![at_least(1.0)]));
    }

    /// showcase.py's own `Color = tuple[Channel, Channel, Channel]` row:
    /// a bare ALIAS NAME whose `AliasEntry` carries `positions` Some
    /// (`surface::compile_aliases`'s own tuple arm) resolves through
    /// this SAME bare-Name arm that reads `element`/`head` for a
    /// `list[X]`-shaped alias — forwarding the alias's own per-position
    /// table onto the returned `DeclaredRefinement`, spelled `"tuple[
    /// Channel, Channel, Channel]"` (the alias's OWN slot spellings
    /// joined, the identical spelling an inline `c: tuple[Channel,
    /// Channel, Channel]` parameter would carry — `all_three_alias_
    /// spellings_carry_the_identical_sequence_window`'s own doc states
    /// the same equivalence for a `list[X]` alias). Before this
    /// forwarding, the hardcoded `positions: None` here made `Color`
    /// resolve as a scalar with an EMPTY set, so `paint((255, 300, 0))`
    /// never reached the POSITIONS LAW at all.
    #[test]
    fn a_bare_alias_name_forwards_its_compiled_tuple_positions() {
        let mut aliases = HashMap::new();
        aliases.insert(
            "Channel".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(0.0), at_most(255.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        let channel_set = aliases.get("Channel").expect("just inserted").set.clone();
        aliases.insert(
            "Color".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(Vec::new()),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: Some(vec![
                    (channel_set.clone(), "Channel".to_owned()),
                    (channel_set.clone(), "Channel".to_owned()),
                    (channel_set, "Channel".to_owned()),
                ]),
            },
        );
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&name_expr("Color"), &aliases, &imports, &environment)
            .expect("Color resolves through the alias table");
        assert_eq!(got.spelling, "tuple[Channel, Channel, Channel]");
        let positions = got.positions.expect("Color carries a per-position table, not a scalar set");
        assert_eq!(positions.len(), 3);
        assert_eq!(positions[1].spelling, "Channel");
        assert_eq!(positions[1].set, aliases.get("Channel").expect("still present").set);
    }

    /// `tuple[int, Unreadable]` — one position this table cannot read
    /// declines the WHOLE tuple, the same all-or-nothing rule
    /// `dict[str, Unreadable]` already takes for its own value slot.
    #[test]
    fn fixed_arity_tuple_with_one_unreadable_position_declines_whole() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("tuple[int, Unreadable]").expect("test source must parse");

        let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    /// `tuple[int, ...]` — a VARIADIC tuple (the slice ends in a bare
    /// `...`) is a different, unbounded-length shape this reader does not
    /// recognize; it declines rather than misreading the ellipsis as a
    /// second fixed position.
    #[test]
    fn variadic_tuple_declines_the_fixed_arity_reader() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("tuple[int, ...]").expect("test source must parse");

        let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    /// `tuple[int]` — a SINGLE-element tuple has no `Tuple`-wrapped slice
    /// (ruff only wraps a multi-element subscript), so this reads as a
    /// one-position tuple, not the element-container `list[X]` shape.
    #[test]
    fn single_element_tuple_resolves_one_position() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("tuple[int]").expect("test source must parse");

        let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
            .expect("tuple[int]'s one position must resolve");
        let positions = got.positions.expect("positions present");
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].spelling, "int");
    }

    /// `list[Age]` (an alias element, not a bare sort) is unaffected by
    /// the fallback — it still resolves through the ordinary alias path,
    /// the same as before this fix.
    #[test]
    fn list_of_an_alias_element_still_resolves_through_the_alias_path() {
        let mut aliases = HashMap::new();
        aliases.insert(
            "Age".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(0.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        let imports = no_imports();
        let environment = no_locals();
        let parsed = parse_expression("list[Age]").expect("test source must parse");

        let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
            .expect("list[Age]'s element must resolve through the alias table");
        let element = got.element.expect("element present");
        assert_eq!(element.spelling, "Age");
        assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
    }

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

    #[test]
    fn typed_dict_return_refinement_wraps_the_classs_own_member_table() {
        let mut typed_dicts = HashMap::new();
        let age_declared = DeclaredRefinement {
            set: make_refined_set(vec![at_least(0.0)]),
            spelling: "Age".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
        };
        typed_dicts.insert("PersonDict".to_owned(), vec![("age".to_owned(), age_declared)]);

        let got = typed_dict_return_refinement(&name_expr("PersonDict"), &typed_dicts)
            .expect("a recorded TypedDict name resolves");
        assert_eq!(got.spelling, "PersonDict");
        let members = got.members.expect("members carries the per-field table");
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "age");
    }

    #[test]
    fn typed_dict_return_refinement_declines_a_name_absent_from_the_table() {
        let typed_dicts: HashMap<String, Vec<(String, DeclaredRefinement)>> = HashMap::new();
        assert!(typed_dict_return_refinement(&name_expr("PersonDict"), &typed_dicts).is_none());
    }

    #[test]
    fn a_locally_rebound_alias_name_states_nothing() {
        let mut aliases = HashMap::new();
        aliases.insert(
            "PositiveInt".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(1.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        let imports = no_imports();
        let mut locally_bound = HashSet::new();
        locally_bound.insert("PositiveInt".to_owned());
        let environment = Environment::new(locally_bound);

        let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    #[test]
    fn a_string_annotation_naming_a_visible_alias_resolves() {
        let mut aliases = HashMap::new();
        aliases.insert(
            "PositiveInt".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(1.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(
            &string_literal_expr("PositiveInt"),
            &aliases,
            &imports,
            &environment,
        )
        .expect("a string annotation naming a visible alias resolves");
        assert_eq!(got.spelling, "PositiveInt");
    }

    #[test]
    fn an_alias_name_not_in_the_table_states_nothing_even_as_one_side_of_a_none_union() {
        // `NotAnAlias | None`: `NotAnAlias` is not a compiled alias in
        // this test's table, AND not one of the bare sorts
        // (`int`/`float`/`str`/`bool`) the union arm's own base-sort
        // fallback reads (`declared_refinement`'s `Expr::BinOp` arm doc:
        // inside a `X | None` union, an unresolved `X` falls back to
        // `base_sort_return_refinement` before declining) — so both the
        // alias lookup AND the base-sort fallback miss, and the whole
        // union states nothing, the same "alias lookup miss" reason a
        // bare `NotAnAlias` would give outside a union too.
        let union = none_union(name_expr("NotAnAlias"));
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    fn none_literal_expr() -> Expr {
        Expr::NoneLiteral(ruff_python_ast::ExprNoneLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
        })
    }

    fn bin_or(left: Expr, right: Expr) -> Expr {
        Expr::BinOp(ruff_python_ast::ExprBinOp {
            node_index: Default::default(),
            range: TextRange::default(),
            left: Box::new(left),
            op: ruff_python_ast::Operator::BitOr,
            right: Box::new(right),
        })
    }

    /// `<other> | None`.
    fn none_union(other: Expr) -> Expr {
        bin_or(other, none_literal_expr())
    }

    /// `None | <other>` — the reversed side.
    fn union_none(other: Expr) -> Expr {
        bin_or(none_literal_expr(), other)
    }

    /// `Optional[<other>]`.
    fn optional_of(other: Expr) -> Expr {
        Expr::Subscript(ruff_python_ast::ExprSubscript {
            node_index: Default::default(),
            range: TextRange::default(),
            value: Box::new(name_expr("Optional")),
            slice: Box::new(other),
            ctx: ExprContext::Load,
        })
    }

    fn age_aliases() -> HashMap<String, AliasEntry> {
        let mut aliases = HashMap::new();
        aliases.insert(
            "Age".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(0.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        aliases.insert(
            "Label".to_owned(),
            AliasEntry {
                temporal: None,
                temporal_awareness: crate::surface::TemporalAwareness::Any,
                set: make_refined_set(vec![at_least(1.0)]),
                head: None,
                element: None,
                length_window: None,
                admits_none: false,
                positions: None,
            },
        );
        aliases
    }

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

    /// The parsed module's one `AnnAssign`'s annotation — shared by the
    /// two nested-form tests above.
    fn annotated_or_none_annotation(module: &ruff_python_ast::ModModule) -> &Expr {
        for stmt in module.body.iter() {
            if let ruff_python_ast::Stmt::AnnAssign(ann_assign) = stmt {
                return ann_assign.annotation.as_ref();
            }
        }
        panic!("test module has one AnnAssign statement");
    }

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

    /// The parsed module's one top-level `def`'s own `-> Annotation` —
    /// this section's own twin of `annotated_or_none_annotation` for a
    /// return-typed function rather than an `AnnAssign`.
    fn def_return_annotation(module: &ruff_python_ast::ModModule) -> &Expr {
        for stmt in module.body.iter() {
            if let ruff_python_ast::Stmt::FunctionDef(def) = stmt {
                return def.returns.as_deref().expect("test def carries a return annotation");
            }
        }
        panic!("test module has one top-level def");
    }
}
