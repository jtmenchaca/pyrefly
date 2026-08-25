//! Condition narrowing: what a test being true (or false) says about
//! the names it mentions. The walk forks an environment per branch arm
//! and asks this module to tighten each fork before walking the arm.
//! This file is the contract the walk calls; the narrowing unit fills
//! it in behind these signatures.
//!
//! Two channels narrow a name, in order:
//!
//! - The SET channel (refined-ts-go's condition-tree design): every
//!   name the condition mentions that is CURRENTLY bound `Kind::Set`
//!   (a seeded parameter, or a sort-set `isinstance` just built — see
//!   below) has the whole condition lowered, relative to that one
//!   name, into the kernel's own `NarrowTree` grammar
//!   (`condition_tree_of`) — `!`/`and`/`or` fold exactly as
//!   `refined-ts-go/internal/refinedts/conditiontree` folds them, and
//!   every leaf this file cannot express (a call other than
//!   `isinstance`, two changing names, a string test, …) lowers to
//!   `NarrowTreeKind::Other`, which the kernel treats as "no claim." A
//!   tree that says nothing (`says_anything` false) or that the kernel
//!   REFUSES (panics — an unrefined set shape; caught, never read as a
//!   refutation) leaves the binding untouched. A tree the kernel
//!   answers narrows the CURRENT set by intersection (this file's own
//!   `meet_set_answer`), never widens it, and keeps the binding's own
//!   trust floor — the kernel's answer is never claimed stronger than
//!   what already flowed in.
//! - The VALUES channel (today's ad-hoc path, unchanged): a name
//!   currently bound `Kind::Values` keeps being narrowed leaf-by-leaf
//!   exactly as before this wave — comparisons, `is`/`is not None`,
//!   `isinstance`. The two channels never overlap on one binding: a
//!   name is Values-kind or Set-kind, never both, so a leaf reads
//!   whichever channel its current binding is in.
//!
//! `isinstance(x, int | float)` proving true ALSO seeds a fresh
//! `Kind::Set` binding for a name the environment has not bound at all
//! (an `object`-typed parameter, never seeded by `check.rs::
//! seed_parameters` — no alias states anything for the bare `object`
//! annotation) — this is the "(a seeded parameter, a sort-set)" case
//! the mission names: the SORT itself is what `isinstance` proves, and
//! a sort with no further bound IS a `Kind::Set` (the unbounded
//! integer/float ray), exactly the shape `summaries.rs`'s own
//! `return_sort_fallback` builds for a declined `-> int` return. Only
//! `int`/`float` seed a Set this way; `bool`'s domain is the two exact
//! values `{0, 1}`, so a proved `isinstance(x, bool)` seeds
//! `Kind::Values` instead, matching every other Boolean value this
//! domain carries.
//!
//! A binding of any other kind (`Kind::Null`, `Kind::Object`, …)
//! passes through unchanged on both channels: the honest default is to
//! narrow nothing.
//!
//! Chained comparisons lower to a conjunction of adjacent pairs
//! (`a op1 b op2 c` == `a op1 b and b op2 c`, CPython
//! tmp/cpython/Doc/reference/expressions.rst — Comparisons, "Comparisons
//! can be chained arbitrarily"), so `ExprCompare`'s multi-op form and
//! `and`'s multi-value form share one conjunction helper on both
//! channels.

mod bool_op;
mod compare;
mod condition_tree;
mod isinstance_guards;
mod none_truthiness;
mod path;
mod predicates;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;

use crate::env::Environment;

pub(crate) use isinstance_guards::isinstance_type_tags;

use bool_op::collect_names;
use bool_op::narrow;
use bool_op::narrow_set_kind_names;
use compare::literal_numeric_collection;
use path::narrow_path_comparisons;

/// Tighten `environment` by what `condition` being `truth` says.
/// Returns the narrowed environment for that arm. The honest default
/// narrows nothing — an arm walked with the unnarrowed fork is
/// conservative, never wrong.
pub fn assume(
    condition: &Expr,
    environment: Environment,
    kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) -> Environment {
    let mut environment = environment;
    // the VALUES channel runs FIRST: it is what SEEDS a fresh Set-kind
    // binding from an `isinstance` test on a name the environment held
    // nothing about (`narrow_isinstance_call`'s own doc) — the SAME
    // `and` chain that proves the sort (`isinstance(value, int) and …
    // and 0 <= value <= 120`) also states the comparison bound in this
    // one `assume` call, so the SET channel must see the seeded
    // binding, not run before it exists.
    narrow(condition, &mut environment, kernel, truth);
    narrow_set_kind_names(condition, &mut environment, kernel, truth);
    narrow_path_comparisons(condition, &mut environment, truth);
    environment
}

/// A match arm's GUARD (`case x if <condition>:`), read as a narrowing
/// of `name` — the bare capture the arm's own pattern bound to
/// `subject` — through the SAME comparison-narrowing reader `assume`
/// already runs, rather than a second implementation of the same leaf
/// vocabulary. Built the same sandbox way `narrow_type_guard_call`
/// proves a `TypeGuard` predicate's own body: `name` is bound to
/// `subject` in a fresh `Environment`, `assume(condition, sandbox,
/// kernel, truth)` runs, and whatever `name` ends up bound to is read
/// back — `None` when the guard's shape is not one `assume`'s narrowing
/// channels read at all (the binding never changed, or the name never
/// rebinds because the leaf declined), matching every other "narrows
/// nothing" default this file gives. `truth: true` is the guard's own
/// admitted values (the intersection `match_taken_environment`'s Taken
/// arm needs); `truth: false` is the values the guard rules OUT for this
/// arm (the difference every LATER arm and the wildcard must still see),
/// mirroring the two `narrow_scalar_subject` calls a literal pattern's
/// split already makes with `keep_matched: true`/`false`.
///
/// Only a Values-kind result that GENUINELY narrowed is read back — a
/// guard that seeds a Set-kind binding (an `isinstance` sort proof), or
/// whose own condition shape none of `assume`'s narrowing channels
/// recognize (an unrecognized leaf leaves the binding byte-for-byte the
/// SAME `subject` that went in, `assume`'s own "the honest default
/// narrows nothing" contract), declines to `None` here — an unchanged
/// binding is never read as a PROOF that every member survives; it is
/// the absence of a proof either way, and the caller's job is to keep
/// today's binary guard semantics for that arm rather than treat
/// "unproved" as "proved to admit everything." A guard that genuinely
/// proves every member survives answers with the full binding rather
/// than declining: `proves_its_own_shape` tells that decided full-width
/// answer apart from an untouched binding, which the value list alone
/// cannot distinguish.
pub fn guard_narrowed_values(
    condition: &Expr,
    name: &str,
    subject: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) -> Option<AbstractValue> {
    let mut sandbox = Environment::new(std::collections::HashSet::new());
    sandbox.bind(name, subject.clone());
    let narrowed = assume(condition, sandbox, kernel, truth);
    let bound = narrowed.read(name)?;
    if bound.kind != Kind::Values || bound.kind_tag != subject.kind_tag {
        return None;
    }
    if subject.kind == Kind::Values && same_members(&bound.values, &subject.values) && !proves_its_own_shape(condition, name) {
        // unchanged AND unrecognized: `assume` declined this condition's
        // own shape rather than proving anything about it — never read as
        // a proof. A recognized shape that happens to leave every member
        // standing is a different thing entirely: the reader DID decide
        // the condition and found it true of the whole binding, so the
        // answer is that binding, not a decline.
        return None;
    }
    Some(bound.clone())
}

/// Whether `condition` is a shape this file's own leaves read to a
/// decision about `name` — asked only to tell a PROVED full-width answer
/// ("every member satisfies this guard") apart from an unrecognized one
/// ("no leaf touched the binding"), which an unchanged value list alone
/// cannot distinguish. Membership against a literal numeric collection is
/// the shape that reaches full width in practice (`x in (2, 4)` over a
/// `{2, 4}` binding), so it is the shape recognized here; a comparison
/// leaf narrows strictly or empties, and never needs this question asked.
fn proves_its_own_shape(condition: &Expr, name: &str) -> bool {
    let Expr::Compare(compare) = condition else {
        return false;
    };
    if compare.ops.len() != 1 || compare.comparators.len() != 1 {
        return false;
    }
    if !matches!(compare.ops[0], CmpOp::In | CmpOp::NotIn) {
        return false;
    }
    if name_of(&compare.left) != Some(name) {
        return false;
    }
    literal_numeric_collection(&compare.comparators[0]).is_some()
}

/// Whether two Values bindings admit the SAME set of members, order- and
/// duplicate-insensitive — the identity test `guard_narrowed_values`
/// uses to tell "genuinely narrowed" apart from "passed through
/// untouched."
fn same_members(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().all(|value| b.contains(value)) && b.iter().all(|value| a.contains(value))
}

/// Whether `condition` being `truth`, under `environment` (already
/// narrowed by `assume`), is PROVEN IMPOSSIBLE for this call's own
/// concrete arguments — every name the condition names (`collect_names`)
/// that is bound `Kind::Values` with an EMPTY `values` list is a name
/// `narrow_isinstance_call`'s own "the whole binding disagrees with the
/// test" arm (or an equivalent comparison narrowing) already proved has
/// no member left standing under this branch. CPython never executes a
/// branch it cannot reach for the ACTUAL argument (`pick_years(200)`'s
/// own `isinstance(value, int)` false arm: `value` narrows to the empty
/// Integer-tagged set, since 200 genuinely is an int, so `return
/// len(value)` — unmodeled on a non-string `Kind::Values` — never runs
/// for this call at all); the caller uses this to skip interpreting a
/// dead arm's body rather than letting an unrelated construct inside it
/// decline the WHOLE call.
pub fn arm_is_infeasible(condition: &Expr, environment: &Environment) -> bool {
    let mut names = Vec::new();
    collect_names(condition, &mut names);
    names.iter().any(|name| {
        environment
            .read(name.as_str())
            .is_some_and(|value| value.kind == Kind::Values && value.values.is_empty())
    })
}

/// Whether `expression` is the bare name of a tracked place — the only
/// shape every narrowing leaf here reads on the tested side (mission's
/// "filter what is known, never invent," matching the Go reference's
/// `onPlace` restriction to a single identifier).
pub(super) fn name_of(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    }
}

pub(super) fn is_none_literal(expression: &Expr) -> bool {
    matches!(expression, Expr::NoneLiteral(_))
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression). Mirrors the sibling
/// private helpers of the same shape in loops.rs/expressions.rs/
/// surface.rs — each narrowing-adjacent file keeps its own copy rather
/// than sharing a cross-file dependency for one small leaf reader.
pub(super) fn literal_number(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            Number::Float(value) => Some(*value),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_number(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
}
