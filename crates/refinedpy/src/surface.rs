/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The pydantic surface: `type X = Annotated[int, Field(ge=…, le=…)]`
//! aliases lowered to refined sets, one table (plan-v2 L7).
//!
//! The lowering walks the RAW annotation expression, never the host's
//! resolved `Type`: pydantic `Field`'s stub returns `Any`, so
//! `Type::Annotated`'s metadata slot holds the inferred return type
//! and the `ge`/`le` values are unrecoverable from it
//! (PYREFLY-API-NOTES.md §3).

use std::collections::HashMap;
use std::collections::HashSet;

use refined_sets::codepoint_sets::{string_tuple, strings, without_string_ground};
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::regex_compiler::format_grammar;
use refined_sets::refinement_forms::{
    Form, Refinement, RefinedSet, above, at_least, at_most, below, integer, make_refined_set,
    multiple_of, numbers, one_of, union,
};
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::{Expr, ModModule, Number, Operator, Stmt, StmtImport, StmtImportFrom, UnaryOp};

/// Field kwargs that state nothing about the value set — safe to skip.
/// Any OTHER unrecognized kwarg refuses the whole alias: a constraint
/// this table cannot state must not silently widen or narrow the set.
const INERT_FIELD_KWARGS: &[&str] = &[
    "alias",
    "default",
    "description",
    "examples",
    "title",
];

/// One compiled alias: its own scalar set (empty for a container alias,
/// the same "the container itself states nothing" convention
/// `annotated_expression_set` keeps), plus — when the alias names a
/// `list[X]`/`set[X]`/`Sequence[X]` container — the container's own
/// head spelling, the element's own resolved set AND written spelling,
/// and the container's own length window. A scalar alias (`Age =
/// Annotated[int, Field(ge=0)]`) carries `head: None, element: None,
/// length_window: None`; a container alias (`Boosted =
/// Annotated[list[BoostedSample], Field(min_length=1)]`) carries `head:
/// Some("list")`, `element: Some((<BoostedSample's set>,
/// "BoostedSample"))`, `length_window: Some((1, None))`.
///
/// `element`'s second slot is the element's WRITTEN spelling, not a
/// re-derived one: when the element is itself an alias name
/// (`BoostedSample`), the reconstructed container spelling must read
/// `"list[BoostedSample]"`, the name as written, never
/// `"list[<BoostedSample's unpacked bounds>]"` — the two spell
/// differently even though the compiled SETS are identical, and
/// `declared_refinement`'s own inline `Annotated[list[X],
/// Field(min_length=…)]` arm always preserves the element's own
/// spelling (a nested `declared_refinement` recursion, which reads a
/// bare alias name's spelling as the name itself). Carrying the written
/// spelling here — rather than reformatting the resolved set — is what
/// makes an alias-sourced parameter and an inline-spelled one produce
/// the IDENTICAL `DeclaredRefinement.spelling`
/// (`check.rs::seed_parameters`'s `spelling.starts_with("list[")` gate
/// reads either one the same way regardless).
#[derive(Clone, PartialEq, Debug)]
pub struct AliasEntry {
    pub set: RefinedSet,
    pub head: Option<&'static str>,
    pub element: Option<Box<(RefinedSet, String)>>,
    pub length_window: Option<(i64, Option<i64>)>,
    /// true when the alias's OWN right-hand side admits None alongside
    /// the set (`type X = Optional[Age]`, `type X = Age | None`) —
    /// mirrors `DeclaredRefinement::admits_none` (typereading.rs), read
    /// by `declared_refinement`'s bare-alias-name arm the same way an
    /// inline `Optional[X]`/`X | None` annotation sets it, so `sample:
    /// OptionalSample` narrows and judges identically whether the
    /// `Optional[...]` wrapper is spelled inline at the parameter or
    /// hoisted into a module-level alias name.
    pub admits_none: bool,
}

/// Every `type X = Annotated[int|float, Field(…)]` alias at the
/// module's top level, lowered to its refined set, plus alias-of-alias
/// (`type Adult = Age`, where `Age` already named a compiled set),
/// plus a bare `type Pick = Literal[…]` alias (`literal_alias_set`),
/// plus a union of two `Literal[…]` aliases (`type PickUnion =
/// Literal[10, 20, 30] | Literal["ten", "twenty"]`,
/// `literal_union_alias_set`), plus a `list[X]`/`set[X]`/`Sequence[X]`
/// container alias carrying its element set and length window
/// (`AliasEntry` doc). Statements walk in source order so a later
/// alias can point at an earlier one. Aliases the table cannot lower
/// faithfully are absent — absence declines judgment, it never
/// approximates.
pub fn compile_aliases(module: &ModModule) -> HashMap<String, AliasEntry> {
    let imports = surface_imports(module);
    let mut out: HashMap<String, AliasEntry> = HashMap::new();
    for stmt in module.body.iter() {
        // The three spellings of one module-level alias: the 3.12
        // `type X = ...` statement, the plain `X = Annotated[...]`
        // assignment, and the `X: TypeAlias = Annotated[...]` form —
        // the two assignment spellings are the ONLY ones a
        // cpython-3.11 runtime parses, and the exported runtime band
        // admits 3.11, so the reader must admit them too. A plain
        // assignment whose RHS is not an alias shape simply fails the
        // lowering below and is skipped, exactly like an unreadable
        // `type` RHS.
        let (name, value) = match stmt {
            Stmt::TypeAlias(alias) => {
                let Expr::Name(name) = alias.name.as_ref() else {
                    continue;
                };
                (name.id.as_str(), alias.value.as_ref())
            }
            Stmt::Assign(assign) => {
                let [Expr::Name(name)] = assign.targets.as_slice() else {
                    continue;
                };
                (name.id.as_str(), assign.value.as_ref())
            }
            Stmt::AnnAssign(annotated) => {
                let Expr::Name(name) = annotated.target.as_ref() else {
                    continue;
                };
                let Some(value) = annotated.value.as_deref() else {
                    continue;
                };
                (name.id.as_str(), value)
            }
            _ => continue,
        };
        let sets_by_name: HashMap<String, RefinedSet> =
            out.iter().map(|(name, entry)| (name.clone(), entry.set.clone())).collect();
        // `Optional[X]` / `X | None` (exactly one side a bare `None`
        // literal): peel to the inner `X` and lower THAT through the
        // ordinary chain below — the same peel
        // `typereading::declared_refinement`'s own `Optional`/`BinOp`
        // arms apply to an inline annotation, applied here so a
        // module-level alias spelled `type OptionalAge = Optional[Age]`
        // reads identically to a parameter spelled `age: Optional[Age]`
        // inline. `admits_none` rides onto the compiled `AliasEntry`
        // afterward, never into the RHS this chain lowers.
        let (value, admits_none) = peel_alias_optional(value);
        // A container base's length window rides `annotated_expression_set`'s
        // OWN second tuple slot now — the table's value carries it
        // (`AliasEntry::length_window`) instead of dropping it, and the
        // element itself is resolved by `element_set_and_spelling_for_alias`
        // below using the same fallback chain `declared_refinement`'s own
        // inline container arm applies. `annotated_base_expr` re-reads the
        // SAME `Annotated[...]` subscript's first tuple slot
        // `annotated_expression_set` destructures internally, so
        // `element_container_element` is asked about the actual base
        // (`list[X]`), never the outer `Annotated[...]` wrapper.
        let entry = annotated_expression_set(value, &imports, &sets_by_name)
            .and_then(|(set, length_window)| {
                let base = annotated_base_expr(value, &imports);
                let container = base.and_then(|base| container_head_and_element(base, &imports, &sets_by_name));
                let (head, element) = match container {
                    Some((head, element_expr)) => {
                        (Some(head), Some(element_set_and_spelling_for_alias(element_expr, &imports, &out)?))
                    }
                    None => (None, None),
                };
                Some(AliasEntry {
                    set,
                    head,
                    element: element.map(Box::new),
                    length_window,
                    admits_none: false,
                })
            })
            .or_else(|| {
                // A BARE container RHS with no outer `Annotated[...]`
                // wrapper (`type Amounts = list[Annotated[float,
                // Field(ge=0)]]`, or a plain `type Ints = list[int]`) —
                // `annotated_expression_set` above only recognizes an
                // OUTER `Annotated[...]` subscript, so this shape never
                // reaches it and falls through to here. Enters the SAME
                // `container_head_and_element` recognition the wrapped
                // arm above uses, on `value` directly rather than on
                // `annotated_base_expr(value, ...)`'s unwrap (there is no
                // outer wrapper to unwrap). The container itself states
                // no scalar set — the same empty-set convention the
                // wrapped arm's `set` carries when it has no `Field(…)`
                // kwarg of its own — and `length_window` is always `None`
                // here: a bare container RHS carries no `Field(min_length=
                // …)` slot to read (that kwarg only ever rides the OUTER
                // `Annotated[...]` metadata tuple the wrapped arm reads).
                let (head, element_expr) = container_head_and_element(value, &imports, &sets_by_name)?;
                let element = element_set_and_spelling_for_alias(element_expr, &imports, &out)?;
                Some(AliasEntry {
                    set: make_refined_set(Vec::new()),
                    head: Some(head),
                    element: Some(Box::new(element)),
                    length_window: None,
                    admits_none: false,
                })
            })
            .or_else(|| {
                literal_alias_set(value).map(|set| AliasEntry {
                    set,
                    head: None,
                    element: None,
                    length_window: None,
                    admits_none: false,
                })
            })
            .or_else(|| {
                literal_union_alias_set(value).map(|set| AliasEntry {
                    set,
                    head: None,
                    element: None,
                    length_window: None,
                    admits_none: false,
                })
            })
            .or_else(|| {
                // `type Adult = Age`: the RHS is a bare name that already
                // names a compiled set in this same table.
                let Expr::Name(rhs) = value else {
                    return None;
                };
                out.get(rhs.id.as_str()).cloned()
            })
            .map(|entry| AliasEntry { admits_none: admits_none || entry.admits_none, ..entry });
        if let Some(entry) = entry {
            out.insert(name.to_owned(), entry);
        }
    }
    out
}

/// Peels a bare `Optional[X]` (recognized by bare-Name head, the same
/// no-import-identity convention `typereading::declared_refinement`'s
/// own `Optional` arm takes) or `X | None`/`None | X` (exactly one side
/// a bare `None` literal) down to `(X, true)`. Every other shape is
/// `(value, false)` unchanged — an alias RHS that is neither of these
/// two forms lowers through the ordinary chain exactly as before this
/// peel existed.
fn peel_alias_optional(value: &Expr) -> (&Expr, bool) {
    if let Expr::Subscript(subscript) = value {
        if matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Optional") {
            return (subscript.slice.as_ref(), true);
        }
    }
    if let Expr::BinOp(binop) = value {
        if binop.op == Operator::BitOr {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none != right_is_none {
                let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
                return (other, true);
            }
        }
    }
    (value, false)
}

/// A container alias's own element expression resolved to `(the
/// resolved RefinedSet, the element's own WRITTEN spelling)` — the same
/// fallback chain `declared_refinement`'s inline container arm applies
/// to a `list[X]`/`set[X]`/`Sequence[X]` element (a bare alias name
/// already compiled earlier in `out`, spelled as the alias name itself;
/// a bare `int`/`float`/`str` base sort, spelled `"int"`/`"float"`/
/// `"str"` with the IDENTICAL unbounded sets
/// `typereading.rs::base_sort_return_refinement` gives that sort
/// everywhere else it is read — `int` is the whole-number ray, `float`
/// is the unbounded real ray `numbers()`, never the empty set; a
/// nested inline `Annotated[…]` via `annotated_expression_set`,
/// restricted to the non-container case, spelled through
/// `format_for_diagnostics`; or a `Literal[…]`/`Literal[…] |
/// Literal[…]`, spelled the same way), so `list[float]`,
/// `list[SomeAlias]`, and `list[Literal[1, 2]]` elements all resolve to
/// the SAME set AND the SAME spelling a bare parameter of that same
/// element type would — carrying the alias name's own spelling forward
/// (rather than reformatting its resolved set) is what makes
/// `list[BoostedSample]` reconstruct as `"list[BoostedSample]"`, never
/// `"list[<BoostedSample's unpacked bounds>]"`. `None` when the element
/// expression matches none of these — declines the whole container
/// alias rather than guessing an empty element set.
fn element_set_and_spelling_for_alias(
    element_expr: &Expr,
    imports: &SurfaceImports,
    out: &HashMap<String, AliasEntry>,
) -> Option<(RefinedSet, String)> {
    if let Expr::Name(name) = element_expr {
        let spelling = name.id.as_str();
        // A bare alias name first — `declared_refinement`'s own bare-Name
        // arm tries the alias table BEFORE `base_sort_return_refinement`'s
        // fallback, so an alias that happens to be named `int`/`float`/
        // `str` (impossible in practice, since those are keywords/builtins
        // no alias could shadow as a bare Name target — but the ORDER
        // matters for fidelity) takes the alias reading, never the base
        // sort's. The WRITTEN name is the spelling — never the alias's own
        // unpacked set, formatted — the exact distinction the
        // alias-vs-inline spelling equivalence in `declared_refinement`'s
        // bare-Name arm depends on.
        if let Some(entry) = out.get(spelling) {
            return Some((entry.set.clone(), spelling.to_owned()));
        }
        // The bare `int`/`float`/`str` base-sort fallback
        // (`base_sort_return_refinement`'s own three sets, exactly):
        // `int` is the whole-number ray, `float` is the unbounded real
        // ray `numbers()` (never the empty set), `str` is the whole-
        // strings ground. `StrictInt` is NOT a recognized element sort
        // here — `base_sort_return_refinement` itself does not match it
        // either, so a bare `list[StrictInt]` element declines through
        // this same fallback chain, matching the inline path exactly.
        match spelling {
            "int" => {
                return Some((make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]), spelling.to_owned()));
            }
            "float" => return Some((numbers(), spelling.to_owned())),
            "str" => return Some((strings(), spelling.to_owned())),
            _ => {}
        }
    }
    let sets_by_name: HashMap<String, RefinedSet> =
        out.iter().map(|(name, entry)| (name.clone(), entry.set.clone())).collect();
    if let Some((set, length_window)) = annotated_expression_set(element_expr, imports, &sets_by_name) {
        if length_window.is_none() {
            let spelling = format_for_diagnostics(&set);
            return Some((set, spelling));
        }
        return None;
    }
    if let Some(set) = literal_alias_set(element_expr).or_else(|| literal_union_alias_set(element_expr)) {
        let spelling = format_for_diagnostics(&set);
        return Some((set, spelling));
    }
    None
}

/// Every `type X = Annotated[StrictInt, …]` alias name at the module's
/// top level — check.rs's `TypeAdapter(<alias>).validate_python(...)`
/// adapter route consults this to decide whether a `str` argument may
/// COERCE (a lax `int` base) or must REFUSE outright (a `StrictInt`
/// base, execution-verified against pydantic 2.13.4: `StrictInt` never
/// attempts str-to-int coercion, unlike bare `int`). Scans the SAME
/// `Annotated[...]` subscript shape `annotated_expression_set` reads,
/// but only the base sort — `compile_aliases`' own `RefinedSet` answer
/// carries no strictness bit, since `int` and `StrictInt` compile to
/// the identical integer form.
pub fn strict_int_alias_names(module: &ModModule) -> HashSet<String> {
    let imports = surface_imports(module);
    let mut out = HashSet::new();
    for stmt in module.body.iter() {
        // the same three alias spellings compile_aliases admits
        let (name, value) = match stmt {
            Stmt::TypeAlias(alias) => {
                let Expr::Name(name) = alias.name.as_ref() else {
                    continue;
                };
                (name, alias.value.as_ref())
            }
            Stmt::Assign(assign) => {
                let [Expr::Name(name)] = assign.targets.as_slice() else {
                    continue;
                };
                (name, assign.value.as_ref())
            }
            Stmt::AnnAssign(annotated) => {
                let Expr::Name(name) = annotated.target.as_ref() else {
                    continue;
                };
                let Some(value) = annotated.value.as_deref() else {
                    continue;
                };
                (name, value)
            }
            _ => continue,
        };
        let Expr::Subscript(subscript) = value else {
            continue;
        };
        let Expr::Name(head) = subscript.value.as_ref() else {
            continue;
        };
        if !imports.annotated_names.contains(head.id.as_str()) {
            continue;
        }
        let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
            continue;
        };
        let Some(Expr::Name(base)) = arguments.elts.first() else {
            continue;
        };
        if imports.strict_int_names.contains(base.id.as_str()) {
            out.insert(name.id.as_str().to_owned());
        }
    }
    out
}

/// `type Pick = Literal[10, 20, 30]` (or a single-member/string-member
/// form) — the type-alias-RHS twin of `typereading.rs`'s
/// `declared_refinement`'s own `Literal[...]` arm (int members build a
/// numeric `one_of`, string members build the union of each member's
/// own singleton tuple, a mixed int/string member list declines whole).
/// Mirrored locally rather than imported: `surface.rs` is imported BY
/// `typereading.rs` (`annotated_expression_set`), so importing the
/// other direction would cycle.
fn literal_alias_set(value: &Expr) -> Option<RefinedSet> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let is_literal = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Literal");
    if !is_literal {
        return None;
    }
    if let Some(members) = int_literal_members(subscript.slice.as_ref()) {
        return Some(make_refined_set(vec![one_of(&members)]));
    }
    if let Some(members) = string_literal_members(subscript.slice.as_ref()) {
        return Some(string_literal_set(&members));
    }
    None
}

/// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]` —
/// exactly two `Literal[...]` arms joined by `|`, each read through
/// `literal_alias_set` and folded together by `refinement_forms::union`.
/// Any other union shape (a non-Literal arm, more than two arms — ruff
/// parses a chained `|` as nested `BinOp`s so a third arm would need a
/// second union node this reader does not build) declines.
fn literal_union_alias_set(value: &Expr) -> Option<RefinedSet> {
    let Expr::BinOp(binop) = value else {
        return None;
    };
    if binop.op != Operator::BitOr {
        return None;
    }
    let left = literal_alias_set(binop.left.as_ref())?;
    let right = literal_alias_set(binop.right.as_ref())?;
    Some(make_refined_set(vec![union(left, right)]))
}

/// `Literal[...]`'s slice read as a list of int-literal members —
/// `typereading.rs::int_literal_members`'s exact twin (see that
/// function's doc for the all-or-nothing member-list rule).
fn int_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(int_literal_value).collect();
    }
    Some(vec![int_literal_value(slice)?])
}

/// One `Literal[...]` member read as an int, with unary minus —
/// `typereading.rs::int_literal_value`'s exact twin.
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
/// `typereading.rs::string_literal_members`'s exact twin.
fn string_literal_members(slice: &Expr) -> Option<Vec<String>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(string_literal_value).collect();
    }
    Some(vec![string_literal_value(slice)?])
}

/// One `Literal[...]` member read as a plain string literal —
/// `typereading.rs::string_literal_value`'s exact twin.
fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
        _ => None,
    }
}

/// The UNION of every member's own singleton string tuple —
/// `typereading.rs::string_literal_set`'s exact twin.
fn string_literal_set(members: &[String]) -> RefinedSet {
    let mut set = string_tuple(&members[0]);
    for member in &members[1..] {
        set = make_refined_set(vec![union(set, string_tuple(member))]);
    }
    set
}

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
fn annotated_base_expr<'a>(value: &'a Expr, imports: &SurfaceImports) -> Option<&'a Expr> {
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
fn element_container_element<'a>(
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
fn container_head_and_element<'a>(
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
    refined_sets::codepoint_sets::codepoints()
}

/// A metadata call names pydantic's `Field` when its callee is either
/// a bare name that imports resolved to `Field`, or an attribute whose
/// base is a name that imports resolved to the pydantic module and
/// whose attribute is literally `Field`. A `Field` defined locally or
/// imported from any other module never matches either shape.
fn names_field(func: &Expr, imports: &SurfaceImports) -> bool {
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

/// The import identities the surface resolves names against: which
/// local names mean pydantic's `Field`, which local names mean the
/// pydantic module itself, which local names mean `Annotated` (from
/// `typing` or `typing_extensions`), which local name means pydantic's
/// `StrictInt`, and one set per recognized `annotated_types`
/// constructor (`Ge`/`Gt`/`Le`/`Lt`/`MultipleOf`/`MinLen`/`MaxLen`).
pub struct SurfaceImports {
    field_names: HashSet<String>,
    pydantic_modules: HashSet<String>,
    pub(crate) annotated_names: HashSet<String>,
    pub(crate) literal_names: HashSet<String>,
    strict_int_names: HashSet<String>,
    annotated_types_ge: HashSet<String>,
    annotated_types_gt: HashSet<String>,
    annotated_types_le: HashSet<String>,
    annotated_types_lt: HashSet<String>,
    annotated_types_multiple_of: HashSet<String>,
    annotated_types_min_len: HashSet<String>,
    annotated_types_max_len: HashSet<String>,
}

/// Reads the module's top-level `import`/`from … import …` statements
/// and records the local names that mean pydantic's `Field`, the
/// pydantic module, `Annotated`, pydantic's `StrictInt`, and each
/// recognized `annotated_types` constructor. Only the shapes named in
/// the mission are recognized: `import pydantic[ as x]`,
/// `from pydantic import Field[ as x]` (and `StrictInt[ as x]`), the
/// same two shapes for `Annotated` from `typing`/`typing_extensions`,
/// and `from annotated_types import Ge[ as x]` (and its six siblings).
/// Anything else (a `fields`-style submodule import, a re-export) is
/// out of scope and leaves the corresponding set empty.
pub fn surface_imports(module: &ModModule) -> SurfaceImports {
    let mut field_names = HashSet::new();
    let mut pydantic_modules = HashSet::new();
    let mut annotated_names = HashSet::new();
    let mut literal_names = HashSet::new();
    let mut strict_int_names = HashSet::new();
    let mut annotated_types_ge = HashSet::new();
    let mut annotated_types_gt = HashSet::new();
    let mut annotated_types_le = HashSet::new();
    let mut annotated_types_lt = HashSet::new();
    let mut annotated_types_multiple_of = HashSet::new();
    let mut annotated_types_min_len = HashSet::new();
    let mut annotated_types_max_len = HashSet::new();
    for stmt in module.body.iter() {
        match stmt {
            Stmt::Import(StmtImport { names, .. }) => {
                for alias in names {
                    if alias.name.id.as_str() == "pydantic" {
                        let local = alias.asname.as_ref().unwrap_or(&alias.name);
                        pydantic_modules.insert(local.id.as_str().to_owned());
                    }
                }
            }
            Stmt::ImportFrom(StmtImportFrom {
                module: Some(source),
                names,
                level: 0,
                ..
            }) => {
                for alias in names {
                    let local = alias.asname.as_ref().unwrap_or(&alias.name);
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "Field" {
                        field_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "StrictInt" {
                        strict_int_names.insert(local.id.as_str().to_owned());
                    }
                    if (source.id.as_str() == "typing" || source.id.as_str() == "typing_extensions")
                        && alias.name.id.as_str() == "Annotated"
                    {
                        annotated_names.insert(local.id.as_str().to_owned());
                    }
                    if (source.id.as_str() == "typing" || source.id.as_str() == "typing_extensions")
                        && alias.name.id.as_str() == "Literal"
                    {
                        // A `Literal[...]` annotation states an exact value
                        // set with no alias and no `Annotated` wrapper, so
                        // this import alone means the module can carry
                        // refinement vocabulary the checker reads.
                        literal_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "annotated_types" {
                        match alias.name.id.as_str() {
                            "Ge" => {
                                annotated_types_ge.insert(local.id.as_str().to_owned());
                            }
                            "Gt" => {
                                annotated_types_gt.insert(local.id.as_str().to_owned());
                            }
                            "Le" => {
                                annotated_types_le.insert(local.id.as_str().to_owned());
                            }
                            "Lt" => {
                                annotated_types_lt.insert(local.id.as_str().to_owned());
                            }
                            "MultipleOf" => {
                                annotated_types_multiple_of.insert(local.id.as_str().to_owned());
                            }
                            "MinLen" => {
                                annotated_types_min_len.insert(local.id.as_str().to_owned());
                            }
                            "MaxLen" => {
                                annotated_types_max_len.insert(local.id.as_str().to_owned());
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    SurfaceImports {
        field_names,
        pydantic_modules,
        annotated_names,
        literal_names,
        strict_int_names,
        annotated_types_ge,
        annotated_types_gt,
        annotated_types_le,
        annotated_types_lt,
        annotated_types_multiple_of,
        annotated_types_min_len,
        annotated_types_max_len,
    }
}

/// A numeric literal, with unary minus — the readable-RHS gate for
/// this slice; a constant integer expression over literals
/// (`2**53 + 2`, `2**31 - 1`) folds through `literal_integer_fold`
/// and is accepted only when the folded value converts to f64 without
/// rounding, so the computed spelling of a bound reads exactly as its
/// literal spelling would. None anywhere else (an unread value
/// declines, it never guesses).
pub fn literal_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(f) => Some(*f),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
            Some(-literal_number(unary.operand.as_ref())?)
        }
        Expr::BinOp(_) => {
            let folded = literal_integer_fold(expr)?;
            let as_float = folded as f64;
            if as_float as i64 != folded {
                return None;
            }
            Some(as_float)
        }
        _ => None,
    }
}

/// Constant integer arithmetic over literals, folded exactly in i64:
/// `2**53`, `2**31 - 1`, `60 * 60`. Overflow, a float operand, a
/// division, or any non-literal leaf declines — the fold never
/// approximates.
fn literal_integer_fold(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64(),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
            literal_integer_fold(unary.operand.as_ref())?.checked_neg()
        }
        Expr::BinOp(bin) => {
            let left = literal_integer_fold(bin.left.as_ref())?;
            let right = literal_integer_fold(bin.right.as_ref())?;
            match bin.op {
                Operator::Add => left.checked_add(right),
                Operator::Sub => left.checked_sub(right),
                Operator::Mult => left.checked_mul(right),
                Operator::Pow => left.checked_pow(u32::try_from(right).ok()?),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A plain (non-f-string) string literal — the readable-RHS gate for
/// `pattern=r"…"`. None anywhere else, matching `literal_number`'s
/// decline-don't-guess discipline.
fn literal_string(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str()),
        _ => None,
    }
}

/// `min_length`/`max_length`'s literal int argument — pydantic's own
/// `StringConstraints`/`Field` types these as `int`, never a float, so
/// a fractional or non-literal value declines rather than truncating.
fn literal_length(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64(),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
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
}
