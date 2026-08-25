//! Every `type X = Annotated[…]` module-level alias, compiled into its
//! own refined set (or container/tuple/temporal shape) — the surface
//! unit's own entry table.

use std::collections::HashMap;

use refined_sets::calendar_interpreter::TemporalAnnotation;
use refined_sets::codepoint_sets::strings;
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::{at_least, integer, make_refined_set, numbers, RefinedSet};
use ruff_python_ast::{Expr, ModModule, Operator, Stmt};

use super::annotated_set::{annotated_base_expr, annotated_expression_set, container_head_and_element};
use super::imports::{surface_imports, SurfaceImports};
use super::literal_alias::{literal_alias_set, literal_union_alias_set};
use super::temporal::temporal_alias_annotation;

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
#[derive(Clone, Debug)]
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
    /// A FIXED-ARITY `tuple[X, Y, Z]` alias's own per-position table —
    /// mirrors `DeclaredRefinement::positions` (typereading.rs), the
    /// same "one active field" convention `element` already keeps: a
    /// tuple-shaped alias carries `set` empty and `element`/`head`/
    /// `length_window` unset, since a fixed-arity tuple has no single
    /// shared element or length window, only per-slot refinements keyed
    /// by index. Each slot's own resolved set AND written spelling —
    /// the same `(RefinedSet, String)` pair `element` carries for its
    /// one shared slot — built through `element_set_and_spelling_for_
    /// alias`'s identical fallback chain (a bare alias name, a bare
    /// `int`/`float`/`str` base sort, a nested `Annotated[...]`, or a
    /// `Literal[...]`), applied once per position instead of once for a
    /// shared element. `None` for every other alias shape (scalar,
    /// `list[X]`/`set[X]`/`Sequence[X]`, `Literal[...]`) — populated
    /// ONLY when the alias's own RHS is a recognized fixed-arity tuple
    /// subscript (`compile_aliases`' own tuple arm).
    pub positions: Option<Vec<(RefinedSet, String)>>,
    /// A `date`/`timedelta`/`datetime`/`AwareDatetime`/`NaiveDatetime`
    /// base's own calendar window — the same "one active field"
    /// convention every other container shape keeps: `set` carries
    /// nothing for a temporal alias (a `Temporal*` value is never a
    /// member of a numeric/string `RefinedSet`), so the calendar bound
    /// lives here instead. `None` for every non-temporal alias.
    /// `AliasEntry::admits_none`'s own `bool` still applies on top —
    /// `type OptionalCutoff = Optional[Cutoff]` still means what it
    /// means for a temporal alias exactly as for a scalar one.
    pub temporal: Option<TemporalAnnotation>,
    /// Which of pydantic's two AWARENESS-typed `datetime` bases (if
    /// either) `temporal` was read from — `Any` for bare `datetime`
    /// (either awareness admitted), `RequireAware` for `AwareDatetime`,
    /// `RequireNaive` for `NaiveDatetime` (pydantic's own documented
    /// distinction — cited at `assignability.rs`'s own admission-law
    /// call site). Meaningless (and left `Any`) whenever `temporal` is
    /// `None` or names a non-`Instant` chart (`date`/`timedelta` carry
    /// no awareness concept at all).
    pub temporal_awareness: TemporalAwareness,
}

/// Which of pydantic's aware/naive `datetime` bases a temporal
/// declaration names — `AliasEntry::temporal_awareness`'s own doc.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TemporalAwareness {
    /// Bare `datetime` (or a non-`Instant`-chart declaration, where the
    /// distinction does not apply) — either an aware or a naive
    /// construction is admitted.
    #[default]
    Any,
    /// `pydantic.AwareDatetime` — a naive construction is a designated
    /// fire (assignability.rs's own admission law).
    RequireAware,
    /// `pydantic.NaiveDatetime` — an aware construction is a designated
    /// fire (the mirror).
    RequireNaive,
}

/// Hand-written: `TemporalAnnotation` (a sibling crate's type, out of
/// this crate's reach under the orphan rule) derives only Debug/Clone,
/// not PartialEq, so a blanket `#[derive(PartialEq)]` on `AliasEntry`
/// does not compile once `temporal` is populated — the same situation
/// `refined_domain::abstract_value::AbstractValue` already documents
/// for its own `temporal` field. Every other field compares by its own
/// derived/structural equality; `temporal` compares chart/min/max by
/// hand.
impl PartialEq for AliasEntry {
    fn eq(&self, other: &Self) -> bool {
        self.set == other.set
            && self.head == other.head
            && self.element == other.element
            && self.length_window == other.length_window
            && self.admits_none == other.admits_none
            && self.positions == other.positions
            && self.temporal_awareness == other.temporal_awareness
            && match (&self.temporal, &other.temporal) {
                (None, None) => true,
                (Some(a), Some(b)) => a.chart == b.chart && a.min == b.min && a.max == b.max,
                _ => false,
            }
    }
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
                    positions: None,
                    temporal: None,
                    temporal_awareness: TemporalAwareness::Any,
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
                    positions: None,
                    temporal: None,
                    temporal_awareness: TemporalAwareness::Any,
                })
            })
            .or_else(|| {
                // A bare-RHS FIXED-ARITY `tuple[X, Y, Z]` alias (`type
                // Color = tuple[Channel, Channel, Channel]`,
                // showcase.py's own row) — the same all-or-nothing,
                // per-slot resolution `typereading.rs`'s inline
                // `Expr::Subscript` tuple arm applies to a parameter
                // spelled `tuple[X, Y, Z]` directly, mirrored here so an
                // ALIASED fixed-arity tuple reads identically. `set`/
                // `head`/`element`/`length_window` stay unset (the same
                // "one active field" convention `positions` itself
                // documents): a fixed-arity tuple has no single shared
                // element or scalar set, only per-slot refinements.
                tuple_alias_positions(value, &imports, &out).map(|positions| AliasEntry {
                    set: make_refined_set(Vec::new()),
                    head: None,
                    element: None,
                    length_window: None,
                    admits_none: false,
                    positions: Some(positions),
                    temporal: None,
                    temporal_awareness: TemporalAwareness::Any,
                })
            })
            .or_else(|| {
                // A temporal base (`date`/`timedelta`/`datetime`, or
                // pydantic's `AwareDatetime`/`NaiveDatetime`) wrapped in
                // `Annotated[...]` with `Field(ge=…/le=…/gt=…/lt=…)`
                // bounds spelled as `date(...)`/`timedelta(...)`/
                // `datetime(...)` literal calls, or a bare Name that
                // resolves to a module-level `_cutoff = datetime(...)`
                // assignment (showcase.py's own `Cutoff`/`Stamp` rows).
                // `set`/`head`/`element`/`length_window`/`positions` stay
                // unset — the same "one active field" convention every
                // other container shape already keeps: a `Temporal*`
                // value is never a member of a numeric/string
                // `RefinedSet`.
                temporal_alias_annotation(value, &imports, module).map(|(temporal, awareness)| AliasEntry {
                    set: make_refined_set(Vec::new()),
                    head: None,
                    element: None,
                    length_window: None,
                    admits_none: false,
                    positions: None,
                    temporal: Some(temporal),
                    temporal_awareness: awareness,
                })
            })
            .or_else(|| {
                literal_alias_set(value).map(|set| AliasEntry {
                    set,
                    head: None,
                    element: None,
                    length_window: None,
                    admits_none: false,
                    positions: None,
                    temporal: None,
                    temporal_awareness: TemporalAwareness::Any,
                })
            })
            .or_else(|| {
                literal_union_alias_set(value).map(|set| AliasEntry {
                    set,
                    head: None,
                    element: None,
                    length_window: None,
                    admits_none: false,
                    positions: None,
                    temporal: None,
                    temporal_awareness: TemporalAwareness::Any,
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

/// A bare-RHS FIXED-ARITY `tuple[X, Y, Z]` alias's own per-position
/// table — `typereading.rs`'s inline `Expr::Subscript` tuple arm's own
/// recognition (bare-Name head `tuple`, a `Tuple`-wrapped multi-element
/// slice or a single unwrapped slot for `tuple[X]`, `tuple[X, ...]`'s
/// trailing bare `Expr::EllipsisLiteral` declining as a different,
/// unbounded-length shape this reader does not carry), mirrored here so
/// the ALIAS path resolves the identical shape an inline parameter
/// annotation already does. Each slot resolves through `element_set_
/// and_spelling_for_alias`'s own fallback chain — the SAME reader
/// `element` itself uses for a `list[X]`/`set[X]`/`Sequence[X]` alias's
/// one shared slot, applied once per position instead of once overall.
/// `None` when the RHS is not a `tuple[...]` subscript, the slice ends
/// in a bare `...`, or ANY position fails to resolve — the same all-
/// or-nothing rule the inline arm and `element`'s own container arm
/// both already take: a declined position declines the whole alias
/// rather than guessing a narrower table.
fn tuple_alias_positions(
    value: &Expr,
    imports: &SurfaceImports,
    out: &HashMap<String, AliasEntry>,
) -> Option<Vec<(RefinedSet, String)>> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let is_tuple = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "tuple");
    if !is_tuple {
        return None;
    }
    let slots: Vec<&Expr> = match subscript.slice.as_ref() {
        Expr::Tuple(arguments) => {
            if arguments.elts.iter().any(|element| matches!(element, Expr::EllipsisLiteral(_))) {
                return None;
            }
            arguments.elts.iter().collect()
        }
        Expr::EllipsisLiteral(_) => return None,
        other => vec![other],
    };
    let mut positions = Vec::with_capacity(slots.len());
    for slot in slots {
        positions.push(element_set_and_spelling_for_alias(slot, imports, out)?);
    }
    Some(positions)
}
