//! `Annotated[int|float|str|list[X]|set[X]|Sequence[X], Field(…), …]`
//! read into its stated `RefinedSet` — the scalar/string/sequence
//! surface's own compiler, plus the container-element recognition its
//! callers (the aliases unit, typereading.rs) share.

use std::collections::HashMap;

use refined_sets::codepoint_sets::{codepoints, strings, without_string_ground};
use refined_sets::refinement_forms::{
    above, at_least, at_most, below, integer, make_refined_set, multiple_of, Form, Refinement,
    RefinedSet,
};
use refined_sets::regex_compiler::format_grammar;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::Expr;

use super::imports::SurfaceImports;
use super::literals::{literal_length, literal_number, literal_string};

/// Field kwargs that state nothing about the value set — safe to skip.
/// Any OTHER unrecognized kwarg refuses the whole alias: a constraint
/// this table cannot state must not silently widen or narrow the set.
pub(super) const INERT_FIELD_KWARGS: &[&str] = &[
    "alias",
    "default",
    "description",
    "examples",
    "title",
];

/// A stable priority for a scalar `Annotated[...]` alias's own compiled
/// forms, matching the Go adapter's diagnostic formatter convention
/// (`chain_numeric_method`'s own ordering, e.g. `>= 97 && <= 122 &&
/// integer`) so the two adapters' compiled sets agree on form order for
/// the same annotation — the cross-adapter battery's own
/// `numeric-window-int-multiple-of` row compares them positionally.
/// Rays first (`atLeast`/`above`, then `atMost`/`below`), then
/// `Integer`, then `MultipleOf`, then every other form in its existing
/// relative order (`sort_by_key` is stable, so ties never reorder).
///
/// Scoped to THIS function's own two `make_refined_set` call sites only
/// — never applied inside `refinement_forms::make_refined_set` itself,
/// which every other caller in the tree also uses to build a
/// `RefinedSet` from an already-ordered `forms` vec; sorting there would
/// reorder every compiled artifact's wire form, not just a scalar
/// `Annotated[...]` alias's.
fn canonical_scalar_form_order(form: &Refinement) -> u8 {
    match form.form {
        Form::AtLeast | Form::Above => 0,
        Form::AtMost | Form::Below => 1,
        Form::Integer => 2,
        Form::MultipleOf => 3,
        _ => 4,
    }
}

/// `Annotated[int|float|str|list[X]|set[X]|Sequence[X], Field(…), …]` →
/// the stated set, resolved against the module's import identities. The
/// `Annotated` head name must itself resolve to an import of
/// `typing.Annotated` (or `typing_extensions.Annotated`) — a bare
/// `Annotated` that was never imported is not recognized. The `int`
/// sort carries the integer form (int ≠ float is a product law); the
/// `str` sort carries the string ground (`C*`, codepoint_sets::strings)
/// so a bare `Annotated[str, Field(…)]` with no length/pattern kwarg
/// still names a set (every string). A `list[X]`/`set[X]`/`Sequence[X]`
/// base carries no scalar set of its own (the empty set, the same
/// "container states nothing itself" convention
/// `declared_refinement`'s own bare `list[X]` arm keeps) — this
/// function only recognizes WHETHER `base` has that shape
/// (`element_container_element`'s own doc); the element `X` itself is
/// resolved by the CALLER (`declared_refinement`'s own wildcard
/// fallthrough), through the ordinary `declared_refinement` recursion
/// against `aliases`, never duplicated here. Every metadata element
/// must be a recognized `Field(…)` call (by import identity, not
/// spelling) or the alias refuses.
///
/// `min_length`/`max_length` mean two DIFFERENT things depending on the
/// base's own sort: on a string base they fold into ONE repetition
/// window over the codepoint ground rather than stacking a form per
/// kwarg (pydantic itself reads them as one window's two edges —
/// `StringConstraints`/`Len`, PYREFLY-PYDANTIC-SURFACE.md §2.3, and
/// `tighten_repetition`'s own reading of chained `.min`/`.max` folds
/// the same way); on a `list[X]`/`set[X]`/`Sequence[X]` base they state
/// the SEQUENCE's own length bounds instead, returned as the second
/// tuple element (`None` when the base is not a container, or a
/// container base states no length kwarg) rather than folded into any
/// `RefinedSet` — `check.rs::seed_parameters` reads this to seed a
/// bounded repetition (`repeat_of`) in place of the unbounded star.
/// `pattern` intersects the compiled grammar set (`format_grammar`,
/// unanchored search semantics per AGENT-BRIEF.md's pydantic surface
/// facts) as its own conjoined form, string base only — a length window
/// and a pattern on the same alias both hold at once, exactly like
/// pydantic validates both constraints on the same field.
pub fn annotated_expression_set(
    value: &Expr,
    imports: &SurfaceImports,
    aliases: &HashMap<String, RefinedSet>,
) -> Option<(RefinedSet, Option<(i64, Option<i64>)>)> {
    let Expr::Subscript(subscript) = value else {
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
    let (base, metadata) = arguments.elts.split_first()?;
    let is_string_sort = matches!(base, Expr::Name(sort) if sort.id.as_str() == "str");
    let is_sequence_base = element_container_element(base, imports, aliases).is_some();
    let mut forms: Vec<Refinement> = match base {
        Expr::Name(sort) if sort.id.as_str() == "int" => vec![integer()],
        // `StrictInt` (pydantic's own strict-mode int type, imported by
        // identity via `imports.strict_int_names`) carries the same
        // integer form as bare `int` — strictness itself (no str-to-int
        // coercion) is not a SET fact this table's refinement carries;
        // check.rs's adapter route consults `imports.strict_int_names`
        // directly to decide whether a coercion is admitted.
        Expr::Name(sort) if imports.strict_int_names.contains(sort.id.as_str()) => vec![integer()],
        Expr::Name(sort) if sort.id.as_str() == "float" => vec![],
        // RefinedSet carries an iterative Drop, so `.forms` cannot move
        // out of a set — std::mem::take is the house pattern (AGENT-BRIEF)
        Expr::Name(sort) if sort.id.as_str() == "str" => std::mem::take(&mut strings().forms),
        // `list[X]`/`set[X]`/`Sequence[X]` — the container itself states
        // nothing (the empty set), the same convention
        // `declared_refinement`'s own bare `list[X]` arm keeps; its
        // element belongs to `element_container_element`, read once
        // more below rather than carried through this match arm's own
        // value (the element set is not returned here — it is not this
        // function's own `RefinedSet`, and `declared_refinement`'s own
        // container arm already knows how to read it independently).
        _ if is_sequence_base => Vec::new(),
        _ => return None,
    };
    let mut min_length: Option<i64> = None;
    let mut max_length: Option<i64> = None;
    for meta in metadata {
        let Expr::Call(call) = meta else {
            return None;
        };
        if names_field(call.func.as_ref(), imports) {
            for keyword in call.arguments.keywords.iter() {
                let name = keyword.arg.as_ref()?;
                match name.as_str() {
                    "ge" => forms.push(at_least(literal_number(&keyword.value)?)),
                    "gt" => forms.push(above(literal_number(&keyword.value)?)),
                    "le" => forms.push(at_most(literal_number(&keyword.value)?)),
                    "lt" => forms.push(below(literal_number(&keyword.value)?)),
                    "multiple_of" => forms.push(multiple_of(literal_number(&keyword.value)?)),
                    "min_length" if is_string_sort || is_sequence_base => {
                        min_length = Some(literal_length(&keyword.value)?);
                    }
                    "max_length" if is_string_sort || is_sequence_base => {
                        max_length = Some(literal_length(&keyword.value)?);
                    }
                    "pattern" if is_string_sort => {
                        let pattern = literal_string(&keyword.value)?;
                        let mut grammar = format_grammar(pattern, "");
                        if !grammar.ok {
                            // a pattern this table cannot compile refuses the
                            // WHOLE alias, the same decline the table gives
                            // any other unrecognized kwarg — never a partial
                            // set missing the pattern conjunct.
                            return None;
                        }
                        forms.extend(std::mem::take(&mut grammar.set.forms));
                        // `forms` up to here still carries the bare `str`
                        // base's own C* ground (`strings()`, seeded at this
                        // function's top) STACKED beside the pattern's own
                        // compiled concatenation/repeat forms — a language
                        // over C either way, so the ground conjunct adds
                        // nothing to the claim, but the kernel's aligned-
                        // segment pattern prover (alignedSegSubsetB) reads
                        // ONE shape, never a stack, and refuses the pair the
                        // moment a second, unrelated top-level form rides
                        // alongside the pattern's own chain (the redundant
                        // ground blinds it exactly the way TS's own
                        // `.regex()` compilation already documents and
                        // strips — chain_method.go's `WithoutStringGround`
                        // call, mirrored here). Dropping it is the same
                        // `without_string_ground` this file's `min_length`/
                        // `max_length` branch below already applies for the
                        // identical reason; unlike that branch this one does
                        // not need the "keep the ground when it is the only
                        // form" carve-out spelled out by hand, since `without_
                        // string_ground` already keeps it when nothing else
                        // remains.
                        forms = without_string_ground(&forms);
                    }
                    other if INERT_FIELD_KWARGS.contains(&other) => {}
                    _ => return None,
                }
            }
            continue;
        }
        // `annotated_types` constructors (`Ge`/`Gt`/`Le`/`Lt`/
        // `MultipleOf`/`MinLen`/`MaxLen`, each a one-positional-argument
        // dataclass — probe-verified against the installed
        // `annotated_types` package's own `inspect.signature`, 2026-08-17)
        // — pydantic recognizes these as `Field(...)`'s own kwarg
        // equivalents when they appear as `Annotated[...]` metadata
        // (execution-verified against pydantic 2.13.4: `Annotated[int,
        // Ge(0), Le(120)]` enforces the identical bound `Field(ge=0,
        // le=120)` would). `MinLen`/`MaxLen` route through the SAME
        // `min_length`/`max_length` window variables `Field`'s own kwargs
        // set, so a `MinLen(...)` and a later `Field(max_length=...)` on
        // the same alias (m-pydantic-schema.py's `LabelAT`) still fold
        // into one window.
        let Some((argument, at_kind)) = annotated_types_argument(call, imports) else {
            return None;
        };
        match at_kind {
            AnnotatedTypesCtor::Ge => forms.push(at_least(literal_number(argument)?)),
            AnnotatedTypesCtor::Gt => forms.push(above(literal_number(argument)?)),
            AnnotatedTypesCtor::Le => forms.push(at_most(literal_number(argument)?)),
            AnnotatedTypesCtor::Lt => forms.push(below(literal_number(argument)?)),
            AnnotatedTypesCtor::MultipleOf => forms.push(multiple_of(literal_number(argument)?)),
            AnnotatedTypesCtor::MinLen if is_string_sort || is_sequence_base => {
                min_length = Some(literal_length(argument)?);
            }
            AnnotatedTypesCtor::MaxLen if is_string_sort || is_sequence_base => {
                max_length = Some(literal_length(argument)?);
            }
            AnnotatedTypesCtor::MinLen | AnnotatedTypesCtor::MaxLen => return None,
        }
    }
    if is_sequence_base {
        // the container's own length bounds ride the SECOND tuple slot
        // — a length window here is a SEQUENCE fact, never a `RefinedSet`
        // conjunct on the (always empty) container set itself.
        let length_window = if min_length.is_some() || max_length.is_some() {
            Some((min_length.unwrap_or(0), max_length))
        } else {
            None
        };
        forms.sort_by_key(canonical_scalar_form_order);
        return Some((make_refined_set(forms), length_window));
    }
    if min_length.is_some() || max_length.is_some() {
        // the window REPLACES the plain C* ground rather than joining
        // it (a length window is strictly tighter), so the ground
        // conjunct is dropped before the window is added — leaving
        // exactly one repetition form when no pattern conjunct is
        // present, and the pattern conjunct plus the window when one
        // is (codepoint_sets::without_string_ground keeps the ground
        // when it is the ONLY form, the opposite of what a REPLACING
        // window needs, so this drops it unconditionally instead).
        let mut window = repetition(strings_codepoint_ground(), min_length.unwrap_or(0), max_length);
        let ground = strings();
        let plain_ground = &ground.forms[0];
        forms.retain(|f| f != plain_ground);
        forms.extend(std::mem::take(&mut window.forms));
    }
    forms.sort_by_key(canonical_scalar_form_order);
    Some((make_refined_set(forms), None))
}

/// An `Annotated[...]` expression's own first tuple slot (the `base` every
/// other reader in this file destructures from `value` directly) — the
/// same `Subscript` → `Annotated` name check → `Tuple` → first-element
/// walk `annotated_expression_set` does at its own top, factored out so
/// `compile_aliases` can hand `element_container_element` the actual
/// base rather than the outer `Annotated[...]` wrapper. `None` for
/// anything that is not this exact shape, matching every other reader
/// here.
pub(super) fn annotated_base_expr<'a>(value: &'a Expr, imports: &SurfaceImports) -> Option<&'a Expr> {
    let Expr::Subscript(subscript) = value else {
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
    arguments.elts.first()
}

/// Whether `base` (the first slot of an `Annotated[...]` subscript) is
/// itself a `list[X]`/`set[X]`/`Sequence[X]` container shape — recognized
/// by bare-Name head only, the same no-import-identity convention
/// `declared_refinement`'s own container arm takes. Returns the element
/// EXPRESSION (unread — `declared_refinement`'s own container arm, not
/// this function, is what actually resolves it against `aliases` and
/// builds the `DeclaredRefinement`); this function only answers WHETHER
/// `base` has this shape, so `annotated_expression_set` can gate its
/// `min_length`/`max_length` reading onto "a container base" without
/// duplicating the element-resolution work `declared_refinement`
/// already owns. `_imports`/`_aliases` are unused today (the element
/// itself is never read here) but kept in the signature so a future
/// caller that DOES need the resolved element does not have to thread
/// them in fresh.
pub(super) fn element_container_element<'a>(
    base: &'a Expr,
    _imports: &SurfaceImports,
    _aliases: &HashMap<String, RefinedSet>,
) -> Option<&'a Expr> {
    let Expr::Subscript(subscript) = base else {
        return None;
    };
    let is_container_head = matches!(
        subscript.value.as_ref(),
        Expr::Name(head) if head.id.as_str() == "list" || head.id.as_str() == "set" || head.id.as_str() == "Sequence"
    );
    if !is_container_head {
        return None;
    }
    Some(subscript.slice.as_ref())
}

/// `element_container_element`'s own recognition, plus the container's
/// head spelling (`"list"`/`"set"`/`"Sequence"`) alongside the element
/// expression — `compile_aliases`' own use, so a container alias's
/// `AliasEntry::head` matches the exact spelling
/// `declared_refinement`'s own inline container arm builds
/// (`typereading.rs::annotated_sequence_container`'s twin, mirrored
/// locally for the same import-direction reason `literal_alias_set`'s
/// doc already gives).
pub(super) fn container_head_and_element<'a>(
    base: &'a Expr,
    imports: &SurfaceImports,
    aliases: &HashMap<String, RefinedSet>,
) -> Option<(&'static str, &'a Expr)> {
    let Expr::Subscript(subscript) = base else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    let head_spelling = match head.id.as_str() {
        "list" => "list",
        "set" => "set",
        "Sequence" => "Sequence",
        _ => return None,
    };
    let element = element_container_element(base, imports, aliases)?;
    Some((head_spelling, element))
}

/// The codepoint ground (`C`, one scalar) `min_length`/`max_length`
/// repeat over — `repetition_window_forms::repetition` takes the
/// ELEMENT set, not the already-starred string ground.
fn strings_codepoint_ground() -> RefinedSet {
    codepoints()
}

/// A metadata call names pydantic's `Field` when its callee is either
/// a bare name that imports resolved to `Field`, or an attribute whose
/// base is a name that imports resolved to the pydantic module and
/// whose attribute is literally `Field`. A `Field` defined locally or
/// imported from any other module never matches either shape.
pub(super) fn names_field(func: &Expr, imports: &SurfaceImports) -> bool {
    match func {
        Expr::Name(n) => imports.field_names.contains(n.id.as_str()),
        Expr::Attribute(a) => {
            a.attr.as_str() == "Field"
                && matches!(a.value.as_ref(), Expr::Name(base) if imports.pydantic_modules.contains(base.id.as_str()))
        }
        _ => false,
    }
}

/// One `annotated_types` constructor this table lowers — each is a
/// one-positional-argument dataclass (probe-verified against the
/// installed package, 2026-08-17: `Ge(ge)`, `Gt(gt)`, `Le(le)`,
/// `Lt(lt)`, `MultipleOf(multiple_of)`, `MinLen(min_length)`,
/// `MaxLen(max_length)`), so the constructor identity alone decides
/// which `Field` kwarg it plays.
#[derive(Clone, Copy)]
enum AnnotatedTypesCtor {
    Ge,
    Gt,
    Le,
    Lt,
    MultipleOf,
    MinLen,
    MaxLen,
}

/// A metadata call names a recognized `annotated_types` constructor —
/// by IMPORT IDENTITY, the same discipline `names_field` holds for
/// `Field` — and reads its single positional argument. `None` for a
/// call with zero or more-than-one positional argument, any keyword
/// argument (`annotated_types`' own constructors take one positional
/// value; a keyword-spelled call is not the shape these rows use), or a
/// callee this table's import table does not name.
fn annotated_types_argument<'a>(call: &'a ruff_python_ast::ExprCall, imports: &SurfaceImports) -> Option<(&'a Expr, AnnotatedTypesCtor)> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    let kind = if imports.annotated_types_ge.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Ge
    } else if imports.annotated_types_gt.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Gt
    } else if imports.annotated_types_le.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Le
    } else if imports.annotated_types_lt.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Lt
    } else if imports.annotated_types_multiple_of.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::MultipleOf
    } else if imports.annotated_types_min_len.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::MinLen
    } else if imports.annotated_types_max_len.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::MaxLen
    } else {
        return None;
    };
    Some((argument, kind))
}
