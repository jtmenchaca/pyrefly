//! The general `declared_refinement` table: what one annotation
//! expression states, across every recognized shape — alias names,
//! `Optional[X]`/`Literal[...]`/`dict[str, X]`/`list[X]`/`set[X]`/
//! `Sequence[X]`/`tuple[X, ...]`/generator family/`X | None`/string
//! forward references, and the plain `Annotated[...]` fallthrough.

use std::collections::HashMap;

use refined_sets::calendar_interpreter::format_temporal;
use refined_sets::calendar_interpreter::TemporalAnnotation;
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_parser::parse_expression;

use crate::env::Environment;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;
use crate::surface::annotated_expression_set;
use crate::surface::temporal_inline_annotation;

use super::base_sort::base_sort_return_refinement;
use super::generator::generator_refinement;
use super::literal_members::bool_literal_members;
use super::literal_members::int_literal_members;
use super::literal_members::string_literal_members;
use super::literal_members::string_literal_set;

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
    /// name, whether the declaration REQUIRES that key to be present, and
    /// the refinement ITS OWN annotation states, in declaration order —
    /// `PersonDict`'s `age: Age` becomes one `TypedDictMember` naming
    /// `"age"`, required, carrying `Age`'s own `DeclaredRefinement`.
    /// Unlike `element` (`dict[str, X]`'s one refinement shared by every
    /// member), a TypedDict's members are HETEROGENEOUS by name, so this
    /// carries one entry per field rather than one shared refinement.
    /// `set`/`element`/`generator` are unused (empty/None) when this is
    /// Some, the same "one active field" convention the other container
    /// shapes already keep.
    pub members: Option<Vec<TypedDictMember>>,
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
    /// container shape here already keeps. `tuple[X, ...]` (a variadic
    /// tuple, the slice ending in a bare `...`) is a DIFFERENT shape this
    /// field does not carry — that subscript is read elsewhere or not at
    /// all.
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

/// One declared member of a TypedDict (or of a module-level class read
/// the same way): the key's name, whether the declaration requires that
/// key to be PRESENT, and the refinement the key's own annotation
/// states.
///
/// `required` follows typing.rst's own totality rules (library/
/// typing.rst, `TypedDict`): "By default, all keys must be present in a
/// ``TypedDict``" and "``True`` is the default, and makes all items
/// defined in the class body required" — so a class with no `total=`
/// keyword, or `total=True`, marks every member required. `total=False`
/// makes every member not required ("a ``Point2D`` ``TypedDict`` can
/// have any of the keys omitted"). A per-key `Required[X]` /
/// `NotRequired[X]` marker overrides the class totality for that one key
/// in either direction ("It is possible to mark individual keys as
/// non-required using :data:`NotRequired`"; "Individual keys of a
/// ``total=False`` ``TypedDict`` can be marked as required using
/// :data:`Required`").
///
/// The MEMBERS LAW (`assignability::judge`) reads this to decide whether
/// a key ABSENT from a flowing value is a refusal or nothing to say.
#[derive(Clone)]
pub struct TypedDictMember {
    pub name: String,
    pub required: bool,
    pub declared: DeclaredRefinement,
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
            // `weakref.WeakKeyDictionary[K, V]` / `weakref.
            // WeakValueDictionary[K, V]` (library/weakref.rst: both are
            // "Mapping class that references keys weakly" /
            // "references values weakly") — the SAME one-value-slot shape
            // the plain `dict[K, X]` arm below reads, at the SAME
            // argument position (argument 2 of 2 for both classes: which
            // side of the pair holds the weak reference does not move
            // the value slot). The mapping's own lookup and membership
            // semantics are the ordinary ones stdtypes.rst's Mapping
            // Types section states once for any hashable key — the
            // "weak" half is a LIFETIME fact (an entry can vanish when a
            // key/value is collected, weakref.rst's own note), invisible
            // to a reader that only ever consumes a PRESENT key's value,
            // the same "collection is invisible to a containment/
            // subscript reader" note `attribute_call.rs`'s own
            // `WeakSet`/`WeakKeyDictionary` bare-constructor row already
            // takes for its zero-argument form. Recognized by the
            // ATTRIBUTE head `weakref.WeakKeyDictionary`/`weakref.
            // WeakValueDictionary` — `Expr::Attribute` whose own value is
            // the bare `Expr::Name` `weakref` — since neither class is a
            // `SurfaceImports` import identity this table tracks; guarded
            // by `environment.read("weakref").is_none()`, the same
            // module-not-shadowed check `attribute_call.rs`'s own
            // constructor row already takes, so a body that rebinds the
            // name `weakref` to something else does not fire this arm.
            // The KEY slot is read only for its own recursive declared
            // refinement to compose properly with a container VALUE
            // slot's key (irrelevant here); the WEAK MAP's key sort is
            // never one of the four `dict[K, X]` states a spelling for
            // (a weak-referenceable object is never `str`/`int`/`float`,
            // library/weakref.rst's own "cannot create weak references
            // to ... int, str" note) so the spelling always carries
            // `object`, the same "any hashable" spelling the plain dict
            // arm already gives its own `object` key sort.
            let is_weak_dict = match subscript.value.as_ref() {
                Expr::Attribute(attribute) => {
                    matches!(attribute.value.as_ref(), Expr::Name(module) if module.id.as_str() == "weakref")
                        && environment.read("weakref").is_none()
                        && matches!(attribute.attr.as_str(), "WeakKeyDictionary" | "WeakValueDictionary")
                }
                _ => false,
            };
            if is_weak_dict {
                if let Expr::Tuple(arguments) = subscript.slice.as_ref() {
                    if let [_key, value] = arguments.elts.as_slice() {
                        if let Some(value_declared) = declared_refinement(value, aliases, imports, environment)
                            .or_else(|| base_sort_return_refinement(value))
                        {
                            // The spelling reuses the plain `dict[K, X]`
                            // arm's OWN shape (`"dict[object, X]"`), never
                            // a `"WeakKeyDictionary[…]"` word of its own —
                            // so every reader keyed on the `"dict["`
                            // spelling prefix (`check::seed_parameters`'
                            // `is_dict_container`, `check::judge`'s
                            // `declared_container_slot`, `union_arm_seed`'s
                            // `"dict[str, "` gate) rides this shape with
                            // no separate WeakKeyDictionary case of its
                            // own to keep in step.
                            let spelling = format!("dict[object, {}]", value_declared.spelling);
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
                return None;
            }
            let is_dict = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "dict");
            if is_dict {
                if let Expr::Tuple(arguments) = subscript.slice.as_ref() {
                    if let [key, value] = arguments.elts.as_slice() {
                        // The KEY sort decides which keys the mapping
                        // admits; the VALUE law — what every present key
                        // reads back as — is the same whichever sort the
                        // keys are, since stdtypes.rst's Mapping Types
                        // section states `d[key]`'s own value rule once,
                        // for any :term:`hashable` key. So this reader
                        // admits the four key sorts the corpus declares
                        // (`str`, `int`, `float`, `object` — the last
                        // being stdtypes.rst's own "any hashable object"
                        // spelling) and carries the key sort in the
                        // SPELLING, so a later reader can still tell them
                        // apart. Every other key annotation still
                        // declines this arm unchanged.
                        let key_sort = match key {
                            Expr::Name(sort) if matches!(sort.id.as_str(), "str" | "int" | "float" | "object") => {
                                Some(sort.id.as_str())
                            }
                            _ => None,
                        };
                        if let Some(key_sort) = key_sort {
                            if let Some(value_declared) = declared_refinement(value, aliases, imports, environment)
                                .or_else(|| base_sort_return_refinement(value))
                            {
                                let spelling = format!("dict[{key_sort}, {}]", value_declared.spelling);
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
        // shapes (`Age | Label`, `list[Age] | int`) decline here: a
        // general union states two alternatives a narrowing test picks
        // BETWEEN, and this table's one-active-field shape has nowhere to
        // hold them apart. A PARAMETER of that shape is seeded as a
        // `Kind::KindUnion` directly instead
        // (`check::seed::union_parameter_seed`), which is the form
        // `isinstance` narrowing already filters arm by arm.
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
