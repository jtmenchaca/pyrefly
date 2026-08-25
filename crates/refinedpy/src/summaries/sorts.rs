/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::on_one_tuple_layer;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::StmtFunctionDef;

use crate::assignability::states_sequence;
use crate::env::Environment;
use crate::typereading::declared_refinement;

/// The SORT SET a same-module call's return annotation states, for a
/// caller that explicitly wants a coarse "some value of this sort"
/// CLAIM rather than the call's own (possibly declined) VALUE — never
/// called from `call_result`/`call_result_with_enclosing`'s own decline
/// points (both answer `None` outright on a genuine decline; see that
/// function's own doc). The one caller today is `evaluate_fstring`'s
/// PATTERN tier: an f-string interpolation only ever COMPOSES this set
/// into a concatenated pattern (never checks it for exact containment
/// against a narrow declared sink), so a fabricated sort-only claim is
/// safe there in a way it is NOT safe as an ordinary call's return value
/// — flowing this set into `assignability.rs`'s CONTAINMENT-REFUTATION
/// law as if it were a checkable fact FIRES the checker's own admission
/// of ignorance against a narrow sink on an otherwise IN-SET call
/// (item 1's own regression: e-class-and-function.py's
/// `first_age(40, 41)`, i-more-expressions.py's
/// `rest_identifier_parameter(40, 41)`, and others — see
/// `call_result_with_enclosing`'s own doc for the full list). This is
/// why the fallback is no longer wired into that function's decline
/// points and is instead exposed here as its own named capability, for
/// `evaluate_fstring` to call directly on a bare same-module call whose
/// ordinary `evaluate_expression` reading already came back `unknown()`.
///
/// NOT reached by `a-statements.py`'s own `def unread_number() -> int:
/// ...`: an ellipsis-only body is NOT a decline in `interpret_body` — a
/// bare `...` is an ordinary `Stmt::Expr` (evaluated and discarded, like
/// `pass`), so the body falls off the end and contributes `null_value()`
/// instead, matching CPython itself (execution-verified: `def f() -> int:
/// ...` really returns `None` at runtime). That call already answers
/// `Kind::Null`, a DIFFERENT existing law's business (`assignability.rs`'s
/// Null-vs-scalar-ground fire) — `evaluate_fstring` only ever retries
/// THIS fallback when the plain reading answered `Kind::Unknown`, so an
/// ellipsis-bodied call's own `Kind::Null` answer never reaches it either.
/// Recognizes only a BARE `int`/`float`/`str` return annotation — the
/// same three base-sort names `surface.rs::annotated_expression_set`
/// matches on an `Annotated[...]` base (that function's own `Expr::Name`
/// arms), reused here by the identical convention rather than re-deriving
/// a different one. `int` answers the whole-number SET (every integer,
/// unbounded — `whole_integers()` below, the same "R-bar itself, no
/// range narrows it" shape `float_sorted_unknown` builds for the float
/// case, but Integer-tagged instead of Float-tagged) rather than one
/// exact value: CPython's own runtime enforces NOTHING about a return
/// annotation (`tmp/cpython/Doc/reference/compound_stmts.rst`'s `funcdef`
/// grammar states `-> expression` as a syntactic annotation only), so
/// this is a language/library-level CLAIM about the def's own contract —
/// graded `TrustSpec` for that reason, matching `float_sorted_unknown`'s
/// identical grading rationale for the `math` family. `float` answers
/// `float_sorted_unknown()` directly. `str` answers the whole-strings set
/// (`codepoint_sets::strings()`, `C*`) at the same Spec grade. Any other
/// return annotation shape (a compiled alias name, `None`, a missing
/// annotation, a `dict[...]`/`list[...]` subscript, …) declines — this
/// fallback states nothing beyond the three base sorts a bare name can
/// spell.
pub fn return_sort_fallback(def: &StmtFunctionDef) -> Option<AbstractValue> {
    let Expr::Name(sort) = def.returns.as_deref()? else {
        return None;
    };
    match sort.id.as_str() {
        "int" => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(whole_integers(), None, TrustSpec, SetKindTag::None)
        }),
        "float" => Some(float_sorted_unknown()),
        "str" => Some(known_set(strings(), None, TrustSpec, SetKindTag::None)),
        _ => None,
    }
}

/// `return_sort_fallback`'s own answer, widened to a declared ALIAS
/// return (`-> Age`, `Age = Annotated[int, Field(ge=0, le=150)]`) —
/// every `call_result_with_enclosing` decline point calls this instead
/// of `return_sort_fallback` directly, so a callee this checker cannot
/// interpret still answers its own declared window, not just the three
/// bare `int`/`float`/`str` sorts `return_sort_fallback` alone reads.
///
/// Tries `typereading::declared_refinement` first, through the alias
/// table `environment` carries (`Environment::declared_aliases`,
/// `check.rs::walk_body_with_self_binding`'s own seeding site) — the
/// SAME table `check.rs::walk_function_def` already reads a def's own
/// `-> Annotation` through, made reachable here too. `None` when this
/// environment carries no alias table (a bare test environment, or a
/// walk that never threaded one through), when the annotation resolves
/// to a container/generator/temporal/TypedDict shape (this reading
/// converts a SCALAR declared set only — the same scope
/// `return_sort_fallback` already keeps), or when the annotation names
/// nothing the alias table recognizes; every one of those falls back to
/// `return_sort_fallback`'s own bare-sort reading unchanged.
///
/// The declared set carries its own numeric sort onto the seeded
/// value's `kind_tag` under the exact gate `check.rs::seed_parameters`
/// already applies to a scalar parameter (`on_one_tuple_layer` true,
/// `states_sequence` false) — a string/sequence-shaped declared set is
/// left untagged, matching that function's own convention. Graded
/// TrustSpec: an annotation states the developer's claim, never an
/// execution-proved fact, the same grading `return_sort_fallback`
/// itself already carries for a bare `int`/`float`/`str` reading.
pub fn declared_return_seed(def: &StmtFunctionDef, environment: &Environment) -> Option<AbstractValue> {
    let annotation = def.returns.as_deref()?;
    let (aliases, imports) = environment.declared_aliases()?;
    let declared = declared_refinement(annotation, aliases, imports, environment)?;
    if declared.set.forms.is_empty() {
        // A container/generator/temporal/TypedDict declaration (typereading's
        // own "one active field" convention — `set` sits empty when
        // `element`/`positions`/`generator`/`temporal`/`members` carries the
        // answer instead) is out of this reading's scope; the caller's own
        // `return_sort_fallback` retry answers what it always answered for
        // that def (nothing, for any of those shapes — `return_sort_
        // fallback`'s own doc).
        return None;
    }
    let seeded = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
        let sort = if requires_integer(&declared.set) { PrimitiveKind::Integer } else { PrimitiveKind::Float };
        AbstractValue {
            kind_tag: Some(sort),
            ..known_set(declared.set, None, TrustSpec, SetKindTag::None)
        }
    } else {
        known_set(declared.set, None, TrustSpec, SetKindTag::None)
    };
    Some(seeded)
}

/// R-bar (`refinement_forms::numbers()`'s own unbounded ray) conjoined
/// with the `int` form — the unbounded whole-number set: every integer,
/// no ceiling/floor. The same shape `surface.rs::annotated_expression_set`
/// builds for a bare `Annotated[int, Field(…)]` with no `ge`/`le` kwarg
/// (`vec![integer()]`, which the kernel already reads as "integer, no
/// other bound" — `numbers()`'s own `at_least(NEG_INFINITY)` form states
/// the identical "unbounded" fact explicitly, so conjoining it changes
/// nothing about which values the set admits and only makes the
/// unbounded-ness textually visible here, mirroring `float_sorted_unknown`'s
/// own `numbers()` base).
pub(super) fn whole_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

/// The ELEMENT sort a same-module GENERATOR/STREAM def's return
/// annotation states, once the body's own straight-line interpretation
/// GENUINELY declines it — a-statements.py's own `stream() ->
/// AsyncIterator[int]: raise NotImplementedError; yield 0` (the `yield`
/// after the `raise` marks this def as an async generator syntactically,
/// datamodel.rst's generator-iterator protocol, but is never reached at
/// runtime; `interpret_body` has no `Stmt::Raise` row, so calling it on
/// this body already answers `None`, the same genuine-decline `loops.rs`'s
/// own for-loop reader hits). Unlike a same-module call's own declined
/// return value (`call_result`/`call_result_with_enclosing`, which answer
/// `None` outright on a genuine decline — a fabricated sort-only claim is
/// never safe to check for exact containment against a narrow sink, since
/// the checker never actually read the body it would be claiming a fact
/// about), a `for`/`async for` loop's own ITERATION count is bounded
/// separately by `loops.rs`'s own cap machinery, so stating the element's
/// bare SORT here (never a value) is a fact the loop reader can use
/// without that same soundness hazard — see `loops.rs` for how the
/// element sort composes with the loop's own iteration bound.
///
/// Recognizes `AsyncIterator[T]` / `Iterator[T]` / `Iterable[T]` — a
/// `Subscript` whose HEAD is one of those three bare names (no import-
/// identity check — this table has no `typing.AsyncIterator`/`Iterator`/
/// `Iterable` import identity to check against, matching `Optional`/
/// `Literal`'s own no-identity reading in `typereading.rs`) — and `T` is
/// itself one of three base-sort names (`int` → the unbounded whole-number
/// set, Integer-tagged; `float` → `float_sorted_unknown()`; `str` → the
/// whole-strings set — the same three base-sort names
/// `surface.rs::annotated_expression_set` matches on an `Annotated[...]`
/// base). Any other subscript head, a `T` that is not one of the three
/// base sorts, or a non-`Subscript` annotation (a missing annotation, a
/// bare name, `None`) declines — this fallback states nothing beyond the
/// three base sorts one level down.
pub fn iterable_element_sort(def: &StmtFunctionDef) -> Option<AbstractValue> {
    let Expr::Subscript(subscript) = def.returns.as_deref()? else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !matches!(head.id.as_str(), "AsyncIterator" | "Iterator" | "Iterable") {
        return None;
    }
    let Expr::Name(element_sort) = subscript.slice.as_ref() else {
        return None;
    };
    match element_sort.id.as_str() {
        "int" => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(whole_integers(), None, TrustSpec, SetKindTag::None)
        }),
        "float" => Some(float_sorted_unknown()),
        "str" => Some(known_set(strings(), None, TrustSpec, SetKindTag::None)),
        _ => None,
    }
}
