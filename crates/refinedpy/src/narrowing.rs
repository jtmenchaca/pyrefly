/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

use std::sync::Arc;

use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::NarrowTree;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::narrow_questions::NarrowCmpOp;
use refined_kernel::narrow_questions::NarrowTreeKind;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::regex_compiler::format_grammar;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Stmt;
use ruff_python_ast::UnaryOp;

use crate::env::Environment;

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

/// The ACCESS-PATH channel's own entry point (`env.rs`'s own
/// `TrackedPlace`/`bind_path`/`read_path` doc): for every numeric
/// comparison `condition` folds through `and` (chained comparisons and
/// `and`-conjunctions only — an `or`'s own operand alone could have made
/// the whole thing true, so it states nothing about any single operand,
/// the same rule the VALUES channel's `narrow_bool_op` already follows),
/// whose tested side is an ATTRIBUTE CHAIN rather than a bare name
/// (`a.n`, `env::tracked_place_of`'s own doc), tighten a WINDOW bound at
/// that path. Run after the VALUES and SET channels, for the identical
/// reason `narrow_set_kind_names` runs after the VALUES channel: nothing
/// in this wave SEEDS a path fact from an `isinstance` test the way a
/// bare name can, so there is no ordering dependency the other direction,
/// but keeping the SAME position after the two name-keyed channels keeps
/// the three channels' relative order stable and easy to reason about.
fn narrow_path_comparisons(condition: &Expr, environment: &mut Environment, truth: bool) {
    match condition {
        Expr::BoolOp(bool_op) if bool_op.op == BoolOp::And && truth => {
            for value in &bool_op.values {
                narrow_path_comparisons(value, environment, truth);
            }
        }
        Expr::Compare(compare) if truth => {
            let mut left = compare.left.as_ref();
            for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
                narrow_one_path_comparison(left, *op, right, environment);
                left = right;
            }
        }
        // `not`, `or`, a single-pair falsity read, a call, anything else:
        // no shape this channel narrows — the honest "narrows nothing"
        // default every leaf in this file keeps. Falsity is not read at
        // all here (unlike the VALUES channel's single-pair `is`/`is not
        // None` exception): a comparison's negation over an unenumerated
        // WINDOW fact has no single bound this channel can tighten to,
        // the same reason `narrow_name_length_against_literal`'s own
        // falsity path folds through `negate_numeric_cmp_op` instead of
        // being read leaf-by-leaf — the path channel keeps that
        // tightening scoped to the truth arm only, matching the mission's
        // narrower ask.
        _ => {}
    }
}

/// One comparison pair (`left op right`) as an ACCESS-PATH narrowing leaf:
/// a numeric literal on one side, an attribute chain on the other
/// (`env::tracked_place_of`), tightens that chain's own WINDOW bound —
/// the identical `{lo, hi}` tightening `narrow_name_length_against_literal`
/// already gives a length window, applied here to a path's own numeric
/// fact instead of a `len(...)` call's result. `is`/`is not`, `in`/`not
/// in`, and a non-numeric operator narrow nothing this leaf reads (the
/// VALUES/SET channels' own leaves already cover those over a BARE name;
/// a path is scoped to the numeric-comparison construct the mission
/// names). Two changing paths (`a.n < b.m`), or a side that is neither a
/// path nor a literal, narrow nothing — the honest default.
fn narrow_one_path_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment) {
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return;
    };
    let (place, on_place, literal) = if let (Some(place), Some(literal)) =
        (crate::env::tracked_place_of(left), literal_number(right))
    {
        (place, true, literal)
    } else if let (Some(literal), Some(place)) =
        (literal_number(left), crate::env::tracked_place_of(right))
    {
        (place, false, literal)
    } else {
        return;
    };
    // a bare name is already the VALUES/SET channels' own business —
    // this leaf is scoped to a GENUINE multi-segment path, the shape
    // those two channels cannot bind at all (`bindings` is keyed on one
    // name)
    if place.path.is_empty() {
        return;
    }
    let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
    narrow_path_window(&place, effective_op, literal, environment);
}

/// Tightens the WINDOW bound recorded for `place` by `place op literal`
/// holding — the path-keyed twin of `narrow_name_length_against_literal`'s
/// own `{lo, hi}` tightening, reused here rather than duplicated: a path
/// fact is read back (or seeded fresh, the unbounded integer ray — every
/// `Age`-annotated instance field this construct's own rows use is an
/// int, and this wave states no path fact for a non-integer field) then
/// tightened the identical way that function tightens a length window,
/// through the SAME `{lo, hi}` triple. `!=` tightens nothing (the same
/// "no shape for a single excluded point" decline that function's own
/// `NotEq` arm gives); a tightened-to-empty window is left UNBOUND rather
/// than rebound (an infeasible path fact is the walk's own dead-branch
/// business, never a narrowing claim this leaf makes).
fn narrow_path_window(place: &crate::env::TrackedPlace, op: NumericCmpOp, literal: f64, environment: &mut Environment) {
    let current = environment.read_path(place).cloned().unwrap_or_else(|| AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(unbounded_integers(), None, TrustSpec, SetKindTag::None)
    });
    if current.kind != Kind::Set {
        return;
    }
    let Some(repeated_or_window) = numeric_window_of(&current.set) else {
        return;
    };
    let (mut lo, mut hi) = repeated_or_window;
    // integer-only bounds (this channel seeds nothing but the unbounded
    // integer ray — `narrow_path_window`'s own doc), the same `± 1`
    // strict-inequality reading `narrow_name_length_against_literal`
    // already takes for a length window.
    match op {
        NumericCmpOp::GtE => lo = lo.max(literal),
        NumericCmpOp::Gt => lo = lo.max(literal + 1.0),
        NumericCmpOp::LtE => hi = Some(hi.map_or(literal, |current_hi| current_hi.min(literal))),
        NumericCmpOp::Lt => hi = Some(hi.map_or(literal - 1.0, |current_hi| current_hi.min(literal - 1.0))),
        NumericCmpOp::Eq => {
            lo = lo.max(literal);
            hi = Some(hi.map_or(literal, |current_hi| current_hi.min(literal)));
        }
        NumericCmpOp::NotEq => return,
    }
    if let Some(h) = hi {
        if h < lo {
            // the window is now provably empty — leave the path fact
            // unchanged rather than rebind to an empty claim; the walk's
            // own dead-branch handling is what skips an unreachable arm,
            // not a narrowed-to-empty path fact here
            return;
        }
    }
    let mut forms = vec![integer()];
    forms.push(at_least(lo));
    if let Some(h) = hi {
        forms.push(at_most(h));
    }
    environment.bind_path(
        place,
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(forms), None, TrustSpec, SetKindTag::None)
        },
    );
}

/// A `RefinedSet`'s own `{lo, hi}` numeric window, read from its
/// `AtLeast`/`AtMost`/`Integer` forms — the SAME shape
/// `narrow_name_length_against_literal` reads off a length window,
/// applied here to a path's own numeric fact. `None` for any set shape
/// other than exactly these three forms (a `oneOf`, a string ground) —
/// this wave's path channel only ever builds this one shape itself, so a
/// set built any other way is not one it can tighten.
fn numeric_window_of(set: &RefinedSet) -> Option<(f64, Option<f64>)> {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &set.forms {
        match form.form {
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost => hi = Some(form.a),
            Form::Integer => {}
            _ => return None,
        }
    }
    Some((lo.unwrap_or(f64::NEG_INFINITY), hi))
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

/// The SET channel's own entry point: for every name `condition`
/// mentions (`collect_names`) that is CURRENTLY bound `Kind::Set`,
/// lower the whole condition relative to that name and ask the kernel
/// once. Run AFTER the VALUES channel's leaf walk (`narrow`, `assume`'s
/// own doc explains why) so a name `narrow`'s own `narrow_isinstance_
/// call` just seeded is already `Kind::Set` by the time this runs — one
/// ask per name, by the WHOLE condition at once (`0 <= value <= 120`
/// intersects both bounds in one ask, matching an `and`'s conjunction
/// reading), rather than once per leaf.
fn narrow_set_kind_names(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    let mut names = Vec::new();
    collect_names(condition, &mut names);
    for name in names {
        let Some(current) = environment.read(name.as_str()) else {
            continue;
        };
        if current.kind != Kind::Set {
            continue;
        }
        let Some(tree) = condition_tree_of(condition, name.as_str()) else {
            continue;
        };
        if !says_anything(&tree) {
            continue;
        }
        // a REFUSED question (a set shape the kernel's narrowing decider
        // does not decide) panics inside the kernel closure rather than
        // returning a claim; caught here and treated as "narrows
        // nothing" — never read as a refutation, mirroring
        // assignability.rs's own containment-ask recover() (and
        // refined-ts-go's narrowRefusable).
        let asked = crate::kernel_ask::ask_kernel(|| (kernel.narrow)(&tree));
        let Ok(answer) = asked else {
            continue;
        };
        let claim = if truth { answer.when_true } else { answer.when_false };
        let Some(claim) = claim else {
            continue;
        };
        let current = environment.read(name.as_str()).expect("just read above").clone();
        let narrowed = meet_set_answer(&current, &claim.set);
        environment.bind(name.as_str(), narrowed);
    }
}

/// Every bare name a leaf this file recognizes tests — the SET
/// channel's own place collector, scoped to exactly the shapes
/// `condition_tree_of` folds (`!`/`and`/`or`, a `Compare`'s two sides,
/// an `isinstance` call's first argument): wide enough to find every
/// name a tree might be built for, never wider — a name mentioned only
/// inside a shape this file does not read (a call other than
/// `isinstance`, an attribute) is never added, matching
/// `condition_tree_of`'s own leaf vocabulary.
fn collect_names(condition: &Expr, out: &mut Vec<String>) {
    let add = |name: &str, out: &mut Vec<String>| {
        if !out.iter().any(|held| held == name) {
            out.push(name.to_owned());
        }
    };
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => collect_names(&unary.operand, out),
        Expr::BoolOp(bool_op) => {
            for value in &bool_op.values {
                collect_names(value, out);
            }
        }
        Expr::Compare(compare) => {
            if let Some(name) = name_of(&compare.left).or_else(|| affine_shifted_name_of(&compare.left)) {
                add(name, out);
            }
            for comparator in &compare.comparators {
                if let Some(name) = name_of(comparator).or_else(|| affine_shifted_name_of(comparator)) {
                    add(name, out);
                }
            }
        }
        Expr::Call(call) => {
            if let Expr::Name(func_name) = call.func.as_ref() {
                if func_name.id.as_str() == "isinstance" && call.arguments.args.len() == 2 {
                    if let Some(name) = name_of(&call.arguments.args[0]) {
                        add(name, out);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Tightens `environment` in place by what `condition` being `truth`
/// says, dispatching on the condition's shape. Every arm that cannot
/// narrow simply returns without touching `environment`. This is the
/// VALUES channel — see the module doc; `kernel` is threaded through
/// only so `narrow_isinstance_call` can seed a fresh Set-kind binding
/// (the "(a seeded parameter, a sort-set)" case), never asked a
/// per-comparison question here — that is the SET channel's own job
/// (`narrow_set_kind_names`), run by `assume` right after this.
fn narrow(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            narrow(&unary.operand, environment, kernel, !truth);
        }
        Expr::BoolOp(bool_op) => narrow_bool_op(bool_op, environment, kernel, truth),
        Expr::Compare(compare) => narrow_compare(compare, environment, truth),
        Expr::Call(call) => {
            if recognizes_type_guard_call(call, environment) {
                if truth {
                    narrow_type_guard_call(call, environment, kernel);
                }
                // The FALSE arm of a `TypeGuard`/`TypeIs` call states
                // nothing this file reads: the predicate's own body proves
                // a set of inputs that make it return True, and a body
                // returning False elsewhere says nothing about which
                // narrower set of inputs that leaves (`is_age`'s own
                // `and`-chain has no single leaf whose negation alone
                // characterizes every False-producing input). Declining
                // the False arm is conservative, never wrong.
                return;
            }
            narrow_isinstance_call(call, environment, truth);
            narrow_regex_module_call(call, environment, truth);
        }
        Expr::Name(_) => narrow_name_truthiness(condition, environment, truth),
        // Calls other than isinstance, attributes, walrus, string
        // comparisons, and everything else this wave does not read: no
        // narrowing, the honest default. (`Expr::Compare`'s own dispatch
        // covers `in`/`not in` over a Values binding — `narrow_one_
        // comparison`'s own membership leaf — and the SET channel narrows
        // the same operator over a Set binding independently.)
        _ => {}
    }
}

/// `and`/`or` narrowing (mission point 3): `and` under truth, and `or`
/// under falsity (De Morgan — `not (a or b)` == `not a and not b`),
/// both apply every operand in conjunction. `and` under falsity and
/// `or` under truth narrow nothing this wave — either arm alone could
/// have made the whole true/false, so no single operand's negation (or
/// affirmation) is forced, and this wave builds no set-union machinery
/// to hold the "at least one of" case.
fn narrow_bool_op(
    bool_op: &ruff_python_ast::ExprBoolOp,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) {
    // `x is None or <tests on x>` under TRUTH (either operand order):
    // the held disjunction admits the absent value OR a value the
    // other side proves — the present part narrows by the other side's
    // own truth while absence stays admitted. Read on a
    // possibly-absent binding only; every other or-under-truth still
    // narrows nothing (either arm alone could have made the whole
    // true).
    if bool_op.op == BoolOp::Or && truth && bool_op.values.len() == 2 {
        for (absence_side, other_side) in [
            (&bool_op.values[0], &bool_op.values[1]),
            (&bool_op.values[1], &bool_op.values[0]),
        ] {
            let Some(name) = is_none_test_name(absence_side) else {
                continue;
            };
            let Some(current) = environment.read(name).cloned() else {
                break;
            };
            if current.kind != Kind::PossiblyUndefined {
                break;
            }
            let inner = current
                .inner
                .as_deref()
                .expect("Kind::PossiblyUndefined always carries an inner value")
                .clone();
            let mut fork = environment.fork();
            fork.bind(name, inner);
            // both channels, the same pair `assume` itself runs: the
            // values channel first, then the set channel over the
            // seeded Set-kind binding
            narrow(other_side, &mut fork, kernel, true);
            narrow_set_kind_names(other_side, &mut fork, kernel, true);
            if let Some(narrowed) = fork.read(name) {
                let mut rewrapped = current.clone();
                rewrapped.inner = Some(Box::new(narrowed.clone()));
                environment.bind(name, rewrapped);
            }
            break;
        }
        return;
    }
    let conjunction = match (bool_op.op, truth) {
        (BoolOp::And, true) => true,
        (BoolOp::Or, false) => true,
        _ => false,
    };
    if !conjunction {
        return;
    }
    // `or` under falsity narrows each operand by its own negation
    // (De Morgan); `and` under truth narrows each by its own truth.
    let per_operand_truth = bool_op.op == BoolOp::And;
    // `x.isascii() and x.isupper()` (or `x.isascii() and x.islower()`),
    // co-occurring anywhere in this SAME conjunction's own operand list —
    // read together, never from either call alone (`narrow_ascii_case_
    // conjunction`'s own doc names why `isupper()`/`islower()` alone
    // cannot bound to ASCII). Asked under `per_operand_truth`, the truth
    // EACH operand is individually proven at (`And`-under-outer-true:
    // each operand proven true; `Or`-under-outer-false: each operand
    // proven FALSE, which this leaf's own `if !truth` guard correctly
    // declines — "not ascii"/"not uppercase" states no positive window).
    narrow_ascii_case_conjunction(&bool_op.values, environment, per_operand_truth);
    for value in &bool_op.values {
        narrow(value, environment, kernel, per_operand_truth);
    }
}

/// `x.isascii() and x.isupper()` / `x.isascii() and x.islower()`, found
/// TOGETHER anywhere among `operands` (an `and` chain's own flat operand
/// list — `len(x) == 2 and x.isascii() and x.isupper()` is one
/// three-value `BoolOp`, F2.fixed's own shape), narrows `x`'s codepoint
/// ALPHABET to exactly the ASCII cased-letter window: `[0x41, 0x5A]` for
/// `isupper()`, `[0x61, 0x7A]` for `islower()`.
///
/// Neither call alone states this bound: `str.isascii()` alone only
/// proves every code point sits in `[0x00, 0x7F]` (stdtypes.rst,
/// `str.isascii()` — "ASCII characters have code points in the range
/// U+0000-U+007F"), and `str.isupper()`/`str.islower()` alone are pinned
/// only against the full Unicode "cased character" categories
/// (stdtypes.rst's own `[4]` footnote), which include cased letters far
/// outside ASCII (e.g. 'É', 'ß') — bounding either call by itself to
/// `[0x41,0x5A]`/`[0x61,0x7A]` would overclaim. Restricted to ASCII BY
/// `isascii()` in the same conjunction, though, the codepoints
/// `isupper()`/`islower()` can additionally hold narrow to EXACTLY the
/// ASCII cased letters: within `[0x00, 0x7F]`, the only cased code
/// points at all are `A`-`Z` (`0x41`-`0x5A`) and `a`-`z` (`0x61`-`0x7A`)
/// — every other ASCII code point (control characters, digits,
/// punctuation, space) is uncased, so "every cased character is
/// uppercase, and there is at least one" restricted to that alphabet
/// collapses to "every code point is in `[0x41, 0x5A]`."
///
/// Reads and rebuilds through `as_repetition`/`repeat_of`, the same
/// element-preserving pattern `narrow_name_length_against_literal` uses
/// for the LENGTH half of this same guard — this leaf tightens the
/// ELEMENT instead, so the two compose regardless of which operand the
/// source lists first (each leaf reads whatever the OTHER already
/// narrowed, since both run against the same shared `environment`).
/// Only the TRUE arm narrows: "not ASCII" or "not uppercase" states no
/// single alphabet this window can name (the excluded codepoints are
/// scattered, not a window), matching `narrow_regex_module_call`'s own
/// "no complement" default for a state this grammar cannot express.
/// Every other shape (no `isascii()` call, no `isupper()`/`islower()`
/// call, receivers naming different places, a non-Set binding) narrows
/// nothing — the honest default every leaf in this file keeps.
fn narrow_ascii_case_conjunction(operands: &[Expr], environment: &mut Environment, truth: bool) {
    if !truth {
        return;
    }
    let Some(name) = operands.iter().find_map(is_isascii_call_name) else {
        return;
    };
    let Some((case_name, ascii_case)) = operands.iter().find_map(is_ascii_case_call) else {
        return;
    };
    if case_name != name {
        return;
    }
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Set {
        return;
    }
    let Some(repeated) = refined_sets::repetition_window_forms::as_repetition(&current.set) else {
        return;
    };
    let (lo, hi) = ascii_case.codepoint_window();
    let element = make_refined_set(vec![integer(), at_least(lo), at_most(hi)]);
    let grade = trust_level_of(&current);
    let narrowed_set = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element, repeated.lo, repeated.hi)]);
    environment.bind(
        name,
        AbstractValue {
            kind_tag: current.kind_tag,
            ..known_set(narrowed_set, None, grade, current.set_kind_tag)
        },
    );
}

/// `A`-`Z` or `a`-`z` — the two ASCII cased-letter windows
/// `narrow_ascii_case_conjunction` narrows to, told apart by which call
/// (`isupper`/`islower`) named them.
#[derive(Clone, Copy)]
enum AsciiCase {
    Upper,
    Lower,
}

impl AsciiCase {
    fn codepoint_window(self) -> (f64, f64) {
        match self {
            AsciiCase::Upper => (0x41 as f64, 0x5A as f64),
            AsciiCase::Lower => (0x61 as f64, 0x7A as f64),
        }
    }
}

/// Whether `expression` is `<bare name>.isascii()` — zero arguments, no
/// keywords, the receiver a bare tracked name. The tested place's own
/// name, or `None` for any other shape.
fn is_isascii_call_name(expression: &Expr) -> Option<&str> {
    is_bare_string_predicate_call(expression, "isascii")
}

/// Whether `expression` is `<bare name>.isupper()` / `<bare name>.
/// islower()` — the tested place's own name paired with which case the
/// call names, or `None` for any other shape.
fn is_ascii_case_call(expression: &Expr) -> Option<(&str, AsciiCase)> {
    if let Some(name) = is_bare_string_predicate_call(expression, "isupper") {
        return Some((name, AsciiCase::Upper));
    }
    if let Some(name) = is_bare_string_predicate_call(expression, "islower") {
        return Some((name, AsciiCase::Lower));
    }
    None
}

/// `<bare name>.<method>()` with zero arguments and no keywords — the
/// shape every `str` no-argument predicate call in this file reads,
/// shared by `isascii`/`isupper`/`islower` rather than duplicated three
/// times.
fn is_bare_string_predicate_call<'a>(expression: &'a Expr, method: &str) -> Option<&'a str> {
    let Expr::Call(call) = expression else { return None };
    let Expr::Attribute(attribute) = call.func.as_ref() else { return None };
    if attribute.attr.as_str() != method {
        return None;
    }
    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return None;
    }
    name_of(&attribute.value)
}

/// `ExprCompare` narrowing: chained comparisons lower to the
/// conjunction of adjacent pairs (see the module doc's CPython
/// citation). Under falsity, a conjunction narrows nothing (the same
/// rule `and`-under-falsity follows in `narrow_bool_op`) — the chain's
/// negation is a disjunction over which pair failed, and this wave
/// holds no union.
fn narrow_compare(compare: &ruff_python_ast::ExprCompare, environment: &mut Environment, truth: bool) {
    if !truth {
        // is/is not None still narrows under falsity for a single pair
        // (mission point 5) — handled directly here since it is not a
        // conjunction the same way numeric comparisons are.
        if compare.ops.len() == 1 {
            narrow_one_comparison(&compare.left, compare.ops[0], &compare.comparators[0], environment, false);
        }
        return;
    }
    let mut left = compare.left.as_ref();
    for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
        narrow_one_comparison(left, *op, right, environment, true);
        left = right;
    }
}

/// One comparison pair (`left op right`) as a narrowing leaf: is/is not
/// None (mission point 5), then numeric literal-side comparisons
/// (mission point 1), mirrored so the literal may sit on either side, then
/// membership against a literal collection (`in`/`not in` — closes the
/// match-guard lane's own scope note: `x in (2, 4)` now narrows a
/// `Kind::Values` binding the same pointwise way a numeric comparison
/// does). Anything else — a call, an attribute, a string, two changing
/// names — narrows nothing.
fn narrow_one_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    if matches!(op, CmpOp::Is | CmpOp::IsNot) {
        narrow_is_none(left, op, right, environment, truth);
        narrow_bool_literal_comparison(left, op, right, environment, truth);
        return;
    }
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        // `b == True` on a bool-domain binding — read by the same leaf as
        // `b is True` (CPython interns the two bools, so the pair
        // coincides there); the numeric paths below still read the same
        // op for every other operand shape.
        narrow_bool_literal_comparison(left, op, right, environment, truth);
    }
    if matches!(op, CmpOp::In | CmpOp::NotIn) {
        if let Some(name) = name_of(left) {
            narrow_name_against_membership(name, right, environment, op == CmpOp::In, truth);
        }
        return;
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return;
    };
    if let (Some(name), Some(literal)) = (len_call_name(left), literal_number(right)) {
        narrow_name_length_against_literal(name, numeric_op, literal, environment, truth);
        return;
    }
    if let (Some(literal), Some(name)) = (literal_number(left), len_call_name(right)) {
        narrow_name_length_against_literal(name, mirror_cmp_op(numeric_op), literal, environment, truth);
        return;
    }
    if let (Some(name), Some(literal)) = (name_of(left), literal_number(right)) {
        narrow_name_against_literal(name, numeric_op, literal, environment, truth);
        return;
    }
    if let (Some(literal), Some(name)) = (literal_number(left), name_of(right)) {
        narrow_name_against_literal(name, mirror_cmp_op(numeric_op), literal, environment, truth);
        return;
    }
}

/// Whether `expression` is `len(<bare name>)` — the one shape
/// `narrow_name_length_against_literal` reads on the tested side, the
/// `len(...)`-wrapped twin of `name_of`'s bare-identifier restriction.
/// `len` called on anything other than a single bare name (an
/// attribute, a call, a literal) is not this leaf's business —
/// `narrow_one_comparison` falls through unchanged for it, the same
/// "narrows nothing" default every unread leaf shape keeps.
fn len_call_name(expression: &Expr) -> Option<&str> {
    let Expr::Call(call) = expression else { return None };
    let Expr::Name(func_name) = call.func.as_ref() else { return None };
    if func_name.id.as_str() != "len" {
        return None;
    }
    let [only] = &*call.arguments.args else { return None };
    name_of(only)
}

/// Narrows a Set-kind binding named `name` by `len(name) op literal`
/// being `truth`: `ages: list[Age]` (no `min_length`/`max_length` in
/// its own surface) seeds `check.rs::seed_parameters`'s star-repetition
/// shape (`refined_sets::refinement_forms::repeat_of`, `lo` 0, `hi`
/// `None` — the bare unbounded window, `typereading.rs`'s own doc for
/// a length-unconstrained sequence parameter) — a window `min_max_over_
/// star` (`builtin_models.rs`) REFUSES to read for `min`/`max` while
/// `lo` could still be 0 (CPython's `ValueError` on an empty sequence).
/// `if len(ages) >= 1:` is exactly the guard that fixture's own doc
/// names as "what a real caller must write to make this call safe at
/// all" — this is the narrowing that makes the checker SEE that guard:
/// under `len(name) >= k` truth (or the mirrored `k <= len(name)`),
/// `lo` tightens to `max(lo, k)`; under `len(name) <= k`/`== k`, `hi`
/// tightens to `min(hi.unwrap_or(k), k)`; under `len(name) > k`, `lo`
/// tightens to `max(lo, k + 1)`; under `len(name) < k`, `hi` tightens
/// the same way one below `k`. Falsity mirrors through `satisfies`'
/// own negation the same way `narrow_name_against_literal` does — a
/// COMPARISON's false arm still states a fact (`not (len(ages) >= 1)`
/// is `len(ages) < 1`, i.e. `len(ages) == 0`, the empty-window case).
///
/// Reads and rebuilds through `as_repetition`/`repeat_of`
/// (`refined_sets::refinement_forms`, `repetition_window_forms`) — the
/// same {element, lo, hi} triple `check.rs`'s own seeding and
/// `min_max_over_star`'s own reading already agree on, so this adds no
/// new window shape, only a narrower `lo`/`hi` on the existing one. A
/// binding that is not `Kind::Set`, or a `Kind::Set` whose own top
/// layer is not this exact repetition shape (a plain numeric range —
/// `len` has no meaning there — or a fixed-arity `Kind::List` already
/// read through the Values-shaped element channel), narrows nothing.
fn narrow_name_length_against_literal(name: &str, op: NumericCmpOp, literal: f64, environment: &mut Environment, truth: bool) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Set {
        return;
    }
    let Some(repeated) = refined_sets::repetition_window_forms::as_repetition(&current.set) else {
        return;
    };
    if literal < 0.0 || literal.fract() != 0.0 {
        // a length is never negative or fractional; a comparison
        // against one is either vacuous or a construct this leaf does
        // not read — narrow nothing rather than guess
        return;
    }
    let k = literal as i64;
    // `op`/`truth` folds to the single EFFECTIVE operator this leaf
    // narrows under — `satisfies`'s own truth-table, applied once at
    // the operator level rather than per element (a length window has
    // no member list to filter, only two bounds to tighten)
    let effective = if truth { op } else { negate_numeric_cmp_op(op) };
    let (mut lo, mut hi) = (repeated.lo, repeated.hi);
    match effective {
        NumericCmpOp::GtE => lo = lo.max(k),
        NumericCmpOp::Gt => lo = lo.max(k + 1),
        NumericCmpOp::LtE => hi = Some(hi.map_or(k, |current_hi| current_hi.min(k))),
        NumericCmpOp::Lt => hi = Some(hi.map_or(k - 1, |current_hi| current_hi.min(k - 1))),
        NumericCmpOp::Eq => {
            lo = lo.max(k);
            hi = Some(hi.map_or(k, |current_hi| current_hi.min(k)));
        }
        // `!=` excludes one point from an interval, which the {lo, hi}
        // window vocabulary cannot state — narrows nothing, the same
        // "no shape for this" decline `narrow_name_against_literal`'s
        // own Values channel never needs (it filters pointwise instead)
        NumericCmpOp::NotEq => return,
    }
    if let Some(h) = hi {
        if h < lo {
            // the window is now provably empty — every leaf in this
            // file leaves an infeasible-branch binding UNCHANGED
            // (`narrow_name_against_literal`'s own "zero survivors"
            // comment states the twin case for a Values binding); the
            // walk's own dead-branch handling is what skips the body,
            // not a narrowed-to-empty rebind here
            return;
        }
    }
    let grade = trust_level_of(&current);
    let narrowed_set = refined_sets::refinement_forms::make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
        repeated.element,
        lo,
        hi,
    )]);
    environment.bind(
        name,
        AbstractValue {
            kind_tag: current.kind_tag,
            ..known_set(narrowed_set, None, grade, SetKindTag::None)
        },
    );
}

/// The strict negation of one `NumericCmpOp` — `not (x >= k)` is
/// `x < k`, etc. — the operator-level mirror of `satisfies`'s own
/// per-element negation (`satisfies(value, op, literal) == truth`),
/// needed here because a length window narrows by tightening a BOUND,
/// not by filtering a member list, so the falsity case folds to a
/// different effective operator up front rather than at each element.
fn negate_numeric_cmp_op(op: NumericCmpOp) -> NumericCmpOp {
    match op {
        NumericCmpOp::Lt => NumericCmpOp::GtE,
        NumericCmpOp::LtE => NumericCmpOp::Gt,
        NumericCmpOp::Gt => NumericCmpOp::LtE,
        NumericCmpOp::GtE => NumericCmpOp::Lt,
        NumericCmpOp::Eq => NumericCmpOp::NotEq,
        NumericCmpOp::NotEq => NumericCmpOp::Eq,
    }
}

/// The subset of `CmpOp` this wave's numeric side-bounds filter reads:
/// `< <= > >= == !=`. `is`/`is not` are handled by `narrow_is_none`;
/// `in`/`not in` by `narrow_name_against_membership` on a Values binding, and
/// by the SET channel's own `membership_leaf_tree_of` on a Set binding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumericCmpOp {
    Lt,
    LtE,
    Gt,
    GtE,
    Eq,
    NotEq,
}

fn numeric_cmp_op(op: CmpOp) -> Option<NumericCmpOp> {
    match op {
        CmpOp::Lt => Some(NumericCmpOp::Lt),
        CmpOp::LtE => Some(NumericCmpOp::LtE),
        CmpOp::Gt => Some(NumericCmpOp::Gt),
        CmpOp::GtE => Some(NumericCmpOp::GtE),
        CmpOp::Eq => Some(NumericCmpOp::Eq),
        CmpOp::NotEq => Some(NumericCmpOp::NotEq),
        CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => None,
    }
}

/// Mirrors the operator when the literal was on the left (`k >= x`
/// means `x <= k`).
fn mirror_cmp_op(op: NumericCmpOp) -> NumericCmpOp {
    match op {
        NumericCmpOp::Lt => NumericCmpOp::Gt,
        NumericCmpOp::LtE => NumericCmpOp::GtE,
        NumericCmpOp::Gt => NumericCmpOp::Lt,
        NumericCmpOp::GtE => NumericCmpOp::LtE,
        NumericCmpOp::Eq => NumericCmpOp::Eq,
        NumericCmpOp::NotEq => NumericCmpOp::NotEq,
    }
}

/// Whether a known single value `v` satisfies `v op literal` — applied
/// pointwise over a Values binding's exact members in
/// `narrow_name_against_literal`.
fn satisfies(value: f64, op: NumericCmpOp, literal: f64) -> bool {
    match op {
        NumericCmpOp::Lt => value < literal,
        NumericCmpOp::LtE => value <= literal,
        NumericCmpOp::Gt => value > literal,
        NumericCmpOp::GtE => value >= literal,
        NumericCmpOp::Eq => value == literal,
        NumericCmpOp::NotEq => value != literal,
    }
}

/// Narrows a Values-kind binding named `name` by `name op literal`
/// being `truth` (mission point 1): keep exactly the members
/// satisfying the (possibly negated) predicate. Zero survivors bind the
/// empty Values state — sound infeasibility (this branch arm's Values
/// set is empty, so any read of `name` inside it answers from no
/// members rather than answering unknown).
fn narrow_name_against_literal(
    name: &str,
    op: NumericCmpOp,
    literal: f64,
    environment: &mut Environment,
    truth: bool,
) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // a numeric-sorted binding (any of Number/Integer/Float — sort-
    // unknown or sort-known — or Boolean, which reads numerically the
    // same way Python's `True + True == 2` does); a String/Array-tagged
    // binding is not read as a number here
    if !is_numeric_or_boolean(kind_tag) {
        return;
    }
    let grade = trust_level_of(&current);
    let kept: Vec<f64> = current
        .values
        .iter()
        .copied()
        .filter(|&value| satisfies(value, op, literal) == truth)
        .collect();
    environment.bind(name, known_values(kept, kind_tag, grade));
}

/// Whether `kind_tag` reads numerically for a literal comparison:
/// Number (sort-unknown), Integer, Float, or Boolean (`True`/`False`
/// compare as `1`/`0`). String and Array are not numeric.
fn is_numeric_or_boolean(kind_tag: PrimitiveKind) -> bool {
    matches!(
        kind_tag,
        PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float | PrimitiveKind::Boolean
    )
}

/// Narrows a Values-kind binding named `name` by `name in <collection>` (or
/// `not in`, `is_in: false`) being `truth`: keep exactly the members that
/// are (`is_in`) or are not (`!is_in`) themselves a member of `collection`'s
/// own literal elements, mirroring `narrow_name_against_literal`'s pointwise
/// filter one-for-one — membership is read directly against the collection's
/// literal numbers here rather than through the kernel's `NarrowTree`/`assume`
/// ask that channel takes for a `Kind::Set` binding (`membership_leaf_tree_of`
/// builds that tree for the SET channel; this is the VALUES channel's own
/// leaf, over an already-enumerated binding, so the members are just read and
/// filtered, the same way every other Values leaf in this file narrows).
/// `collection` must be a literal list/tuple/set of plain number literals
/// (mirroring `membership_leaf_tree_of`'s own numeric half) — anything else
/// (a name, a comprehension, a mixed or string collection) narrows nothing,
/// the same "no shape this file reads" default every other declined leaf
/// gives. Zero survivors bind the empty Values state, the same sound
/// infeasibility `narrow_name_against_literal` gives.
fn narrow_name_against_membership(name: &str, collection: &Expr, environment: &mut Environment, is_in: bool, truth: bool) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    if !is_numeric_or_boolean(kind_tag) {
        return;
    }
    let Some(members) = literal_numeric_collection(collection) else {
        return;
    };
    // `name in <collection>` true, or `name not in <collection>` false,
    // both mean "keep the members present in the collection"; the other
    // two combinations keep the members ABSENT from it — the same
    // `is_in == truth` flip `narrow_name_against_literal`'s own
    // `satisfies(...) == truth` gives a single predicate.
    let keep_present = is_in == truth;
    let grade = trust_level_of(&current);
    let kept: Vec<f64> = current
        .values
        .iter()
        .copied()
        .filter(|value| members.contains(value) == keep_present)
        .collect();
    environment.bind(name, known_values(kept, kind_tag, grade));
}

/// A literal list/tuple/set of plain number literals, read as `f64`s — the
/// numeric half of `membership_leaf_tree_of`'s own element reading, kept as
/// a separate small reader here since this file's own convention
/// (`literal_number`'s doc) is a leaf reader per file rather than a shared
/// cross-file helper. An empty collection, a non-literal collection, or one
/// with any non-numeric member (a string, a name, a nested expression)
/// answers `None` — declined, never partially read.
fn literal_numeric_collection(collection: &Expr) -> Option<Vec<f64>> {
    let elements: &[Expr] = match collection {
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::Set(set) => &set.elts,
        _ => return None,
    };
    if elements.is_empty() {
        return None;
    }
    elements.iter().map(literal_number).collect()
}

/// `is None` / `is not None` (mission point 5): a Values-kind binding
/// narrows by emptying (see below); a `Kind::PossiblyUndefined` binding
/// — an `Optional[X]`/`X | None`-declared parameter's own seed
/// (`check.rs::seed_parameters`) — narrows by UNWRAPPING, the maybe
/// carrier's own reason for existing. A non-Values, non-wrapper binding
/// (including one already `Kind::Null`) passes through unchanged, per
/// the mission's instruction that non-Values states pass through
/// everywhere this wave.
/// `P is None` as a single-pair comparison (either operand order) —
/// the shape the cross-channel disjunction reader in `narrow_bool_op`
/// recognizes as its absence side. The tested bare name, or None for
/// any other shape.
fn is_none_test_name(e: &Expr) -> Option<&str> {
    let Expr::Compare(compare) = e else {
        return None;
    };
    if compare.ops.len() != 1 || compare.ops[0] != CmpOp::Is {
        return None;
    }
    if is_none_literal(&compare.comparators[0]) {
        return name_of(&compare.left);
    }
    if is_none_literal(&compare.left) {
        return name_of(&compare.comparators[0]);
    }
    None
}

fn narrow_is_none(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let is_not = op == CmpOp::IsNot;
    let name = if is_none_literal(right) {
        name_of(left)
    } else if is_none_literal(left) {
        name_of(right)
    } else {
        None
    };
    let Some(name) = name else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    // `name is None` true, or `name is not None` false, both mean
    // "None": a `Kind::PossiblyUndefined` wrapper's own absent side
    // proves this reachable, so the TRUE reading of "None" rebinds to
    // the exact null_value (matching what `assignability::judge` reads
    // directly for an admits_none declaration) — never the wrapper
    // itself, since the wrapper's present side is now proved
    // unreachable on this fork.
    // `name is None` false, or `name is not None` true, both mean "not
    // None": the wrapper's own INNER value is what remains — unwrapped,
    // so a later read sees the plain present-side value (the annotated
    // set, a plain scalar, …) rather than the maybe carrier.
    let means_is_none = truth != is_not;
    if current.kind == Kind::PossiblyUndefined {
        let inner = current.inner.as_deref().expect("Kind::PossiblyUndefined always carries an inner value");
        let narrowed = if means_is_none { null_value() } else { inner.clone() };
        environment.bind(name, narrowed);
        return;
    }
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // `name is None` true, or `name is not None` false, both mean
    // "this Values binding holds None" — impossible for a Values state,
    // so every member is infeasible: bind the empty set.
    // `name is None` false, or `name is not None` true, both mean
    // "not None" — a Values binding already satisfies that for every
    // member, so it is left as is (still narrows nothing further, which
    // is sound: no member is dropped).
    if means_is_none {
        let grade = trust_level_of(&current);
        environment.bind(name, known_values(Vec::new(), kind_tag, grade));
    }
}

/// `name is True` / `name is False` — and the `==`/`!=` spellings of
/// the same pair — against a binding already scoped to the BOOL domain
/// (every member 0 or 1: `bool` seeds `oneOf{0, 1}`, `Literal[...]`
/// bool members and `isinstance(x, bool)` build the same two-value
/// reading). CPython interns exactly two bool objects (datamodel.rst,
/// "Booleans": "The two objects representing the values False and
/// True"), so identity and equality coincide on this domain and one
/// leaf reads all four operators. A binding admitting ANY other member
/// declines whole: `1 is True` is False (distinct objects), so
/// identity against a literal keeps nothing pointwise for a general
/// int, and equality on a wider set is the numeric paths' own
/// business. A filter that would empty the members also declines — an
/// unreachable arm is the walk's provably-false business, not a
/// narrowing claim.
fn narrow_bool_literal_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let bool_literal_value = |expr: &Expr| -> Option<f64> {
        match expr {
            Expr::BooleanLiteral(literal) => Some(if literal.value { 1.0 } else { 0.0 }),
            _ => None,
        }
    };
    let (name, literal) = match (name_of(left), bool_literal_value(right)) {
        (Some(name), Some(literal)) => (name, literal),
        _ => match (bool_literal_value(left), name_of(right)) {
            (Some(literal), Some(name)) => (name, literal),
            _ => return,
        },
    };
    let keep_equal = matches!(op, CmpOp::Is | CmpOp::Eq) == truth;
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    let bool_domain = |members: &[f64]| members.iter().all(|member| *member == 0.0 || *member == 1.0);
    if current.kind == Kind::Values {
        if current.values.is_empty() || !bool_domain(&current.values) {
            return;
        }
        let kept: Vec<f64> = current.values.iter().copied().filter(|member| (*member == literal) == keep_equal).collect();
        if kept.is_empty() {
            return;
        }
        let Some(kind_tag) = current.kind_tag else {
            return;
        };
        environment.bind(name, known_values(kept, kind_tag, trust_level_of(&current)));
        return;
    }
    if current.kind == Kind::Set {
        let [form] = current.set.forms.as_slice() else {
            return;
        };
        if form.form != Form::OneOf || form.w.is_empty() || !bool_domain(&form.w) {
            return;
        }
        let kept: Vec<f64> = form.w.iter().copied().filter(|member| (*member == literal) == keep_equal).collect();
        if kept.is_empty() {
            return;
        }
        let narrowed = AbstractValue {
            kind_tag: current.kind_tag,
            ..known_set(make_refined_set(vec![one_of(&kept)]), None, trust_level_of(&current), current.set_kind_tag)
        };
        environment.bind(name, narrowed);
    }
}

/// `re.fullmatch(pattern, name)` / `re.match` / `re.search` as the
/// whole condition: a truthy match object proves `name`'s string is in
/// the pattern's own language (library/re.html: `fullmatch` — "the
/// whole string matches"; `match` — "at the beginning of the string";
/// `search` — "the first location where"). The pattern compiles through
/// the SAME `format_grammar` the pydantic `pattern=` kwarg uses
/// (surface.rs), anchored to each function's own semantics: `fullmatch`
/// pins both ends, `match` the start, `search` neither
/// (`format_grammar` itself pads an unanchored side with C*). The
/// narrowed binding meets the compiled set into the current one,
/// dropping the bare C* string ground first — the kernel's aligned-
/// segment pattern prover reads one chain, never a stack (surface.rs's
/// own `pattern` branch documents the identical strip). The FALSE arm
/// narrows nothing: "no match" has no complement this grammar states.
/// A non-literal pattern, keyword or flag arguments, a non-name
/// subject, a non-Set binding, or a pattern `format_grammar` refuses
/// all decline — the honest default.
fn narrow_regex_module_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, truth: bool) {
    if !truth {
        return;
    }
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };
    if name_of(attribute.value.as_ref()) != Some("re") {
        return;
    }
    let (anchor_start, anchor_end) = match attribute.attr.as_str() {
        "fullmatch" => (true, true),
        "match" => (true, false),
        "search" => (false, false),
        _ => return,
    };
    if !call.arguments.keywords.is_empty() || call.arguments.args.len() != 2 {
        return;
    }
    let Expr::StringLiteral(literal) = &call.arguments.args[0] else {
        return;
    };
    let Some(name) = name_of(&call.arguments.args[1]) else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Set {
        return;
    }
    let mut pattern = literal.value.to_str().to_owned();
    if anchor_start && !pattern.starts_with('^') {
        pattern.insert(0, '^');
    }
    if anchor_end && !(pattern.ends_with('$') && !pattern.ends_with("\\$")) {
        pattern.push('$');
    }
    let mut grammar = format_grammar(&pattern, "");
    if !grammar.ok {
        return;
    }
    let ground = strings();
    let plain_ground = &ground.forms[0];
    let mut combined: Vec<_> = current.set.forms.iter().filter(|form| *form != plain_ground).cloned().collect();
    combined.extend(std::mem::take(&mut grammar.set.forms));
    let narrowed = AbstractValue {
        kind_tag: current.kind_tag,
        ..known_set(make_refined_set(combined), None, trust_level_of(&current), current.set_kind_tag)
    };
    environment.bind(name, narrowed);
}

/// A bare name as the whole condition (`if x:`): Python truthiness.
/// `x` truthy proves `x is not None` AND `x != 0` (`bool(None)` and
/// `bool(0)` are both False; every other int/float, NaN included, is
/// truthy). A `Kind::PossiblyUndefined` binding — an `Optional[X]`/
/// `X | None` seed — unwraps to its inner value on the truthy arm,
/// exactly as `narrow_is_none`'s not-None side does; a Values inner
/// (or a bare Values binding) additionally keeps only its truthy
/// members. The falsy arm keeps a wrapper unchanged when its inner
/// could itself be falsy (0 in the annotated set) — None and a falsy
/// inner member are then both live; when the inner is a Values set
/// with no falsy member, falsity proves None exactly. A Set-kind or
/// otherwise unread binding narrows nothing — the inner set may hold
/// 0, and dropping nothing is conservative, never wrong.
fn narrow_name_truthiness(condition: &Expr, environment: &mut Environment, truth: bool) {
    let Some(name) = name_of(condition) else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    let truthy_members = |value: &AbstractValue| -> AbstractValue {
        if value.kind == Kind::Values {
            if let Some(kind_tag) = value.kind_tag {
                let kept: Vec<f64> = value.values.iter().copied().filter(|member| *member != 0.0).collect();
                return known_values(kept, kind_tag, trust_level_of(value));
            }
        }
        value.clone()
    };
    if current.kind == Kind::PossiblyUndefined {
        let inner = current.inner.as_deref().expect("Kind::PossiblyUndefined always carries an inner value");
        if truth {
            environment.bind(name, truthy_members(inner));
        } else if inner.kind == Kind::Values && inner.values.iter().all(|member| *member != 0.0) {
            environment.bind(name, null_value());
        }
        return;
    }
    if current.kind == Kind::Set {
        // Exact-member form first: a `oneOf` keeps the members whose
        // truthiness matches (`b: bool` seeds `oneOf{0, 1}`, so
        // `if not b:` proves `{0}`). A filter that would empty the
        // members proves nothing this arm states (the arm is then
        // unreachable — the walk's own provably-false business).
        if current.set.forms.iter().any(|form| form.form == Form::OneOf) {
            let mut forms = current.set.forms.clone();
            let mut rewrote = false;
            for form in &mut forms {
                if form.form == Form::OneOf {
                    let kept: Vec<f64> = form.w.iter().copied().filter(|member| (*member != 0.0) == truth).collect();
                    if kept.is_empty() {
                        return;
                    }
                    if kept.len() != form.w.len() {
                        rewrote = true;
                    }
                    form.w = kept;
                }
            }
            if rewrote {
                let narrowed = AbstractValue {
                    kind_tag: current.kind_tag,
                    ..known_set(make_refined_set(forms), None, trust_level_of(&current), current.set_kind_tag)
                };
                environment.bind(name, narrowed);
            }
            return;
        }
        // WINDOW form ([atLeast, atMost, integer]): truthiness on the
        // integer domain is exactly "≠ 0" (datamodel.rst, truth value
        // testing — a zero of any numeric type is false). The TRUE arm
        // trims a 0 edge off the window (an interior 0 is a hole one
        // window cannot state and trims nothing); the FALSE arm IS the
        // value 0 whenever the window admits it.
        if !current.set.forms.iter().any(|form| form.form == Form::Integer) {
            return;
        }
        let mut lo: Option<f64> = None;
        let mut hi: Option<f64> = None;
        for form in &current.set.forms {
            match form.form {
                Form::AtLeast => lo = Some(form.a),
                Form::AtMost => hi = Some(form.a),
                Form::Integer => {}
                _ => return,
            }
        }
        if truth {
            let mut forms = current.set.forms.clone();
            let mut rewrote = false;
            for form in &mut forms {
                if form.form == Form::AtLeast && form.a == 0.0 {
                    form.a = 1.0;
                    rewrote = true;
                }
                if form.form == Form::AtMost && form.a == 0.0 {
                    form.a = -1.0;
                    rewrote = true;
                }
            }
            if rewrote {
                let narrowed = AbstractValue {
                    kind_tag: current.kind_tag,
                    ..known_set(make_refined_set(forms), None, trust_level_of(&current), current.set_kind_tag)
                };
                environment.bind(name, narrowed);
            }
        } else {
            let admits_zero = lo.is_none_or(|floor| floor <= 0.0) && hi.is_none_or(|ceiling| ceiling >= 0.0);
            if admits_zero {
                environment.bind(
                    name,
                    known_values(vec![0.0], current.kind_tag.unwrap_or(PrimitiveKind::Integer), trust_level_of(&current)),
                );
            }
        }
        return;
    }
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    let kept: Vec<f64> = current.values.iter().copied().filter(|member| (*member != 0.0) == truth).collect();
    environment.bind(name, known_values(kept, kind_tag, trust_level_of(&current)));
}

/// Whether `call` is a same-module call to a function whose OWN return
/// annotation is `TypeGuard[X]`/`TypeIs[X]` (typing.rst's user-defined
/// type guard: "a special form... that can be used to annotate the
/// return type of a user-defined type guard function") — recognized
/// SYNTACTICALLY only. `typing.TypeGuard`/`TypeIs` state a CLAIM the
/// function's own signature makes; this recognizer's caller
/// (`narrow_type_guard_call`) never trusts that claim on its own —
/// f-type-nodes.py's own `dishonest_predicate` row is exactly why:
/// `claims_age`'s signature states `TypeGuard[Age]`, but its body only
/// proves `isinstance(v, int)`, strictly weaker than `Age`; trusting the
/// claim would wrongly narrow `value` all the way to `Age` and read the
/// row SILENT, when the row expects a fire. This function's ONLY job is
/// to recognize the shape so the `Expr::Call` dispatch knows to ATTEMPT
/// body-proof narrowing at all — the claimed `X` itself is never read
/// anywhere in this recognizer or its caller.
fn recognizes_type_guard_call(call: &ruff_python_ast::ExprCall, environment: &Environment) -> bool {
    let Expr::Name(callee) = call.func.as_ref() else {
        return false;
    };
    let Some(def) = environment.functions().and_then(|table| table.def(callee.id.as_str())) else {
        return false;
    };
    let Some(returns) = def.returns.as_deref() else {
        return false;
    };
    let Expr::Subscript(subscript) = returns else {
        return false;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return false;
    };
    matches!(head.id.as_str(), "TypeGuard" | "TypeIs")
}

/// Narrows `call`'s own first argument by what a `TypeGuard[X]`/`TypeIs[X]`-
/// annotated predicate's OWN BODY proves, never by the annotation's claimed
/// `X` — `recognizes_type_guard_call`'s own doc names why trusting the
/// claim alone is unsound. The proof: when the predicate's body is exactly
/// one statement, `return <condition>` (`is_age`/`claims_age`'s own shape —
/// a boolean expression naming the predicate's own first parameter), that
/// `<condition>` is handed to THIS SAME `assume` function, in a fresh
/// sandbox environment where the predicate's own parameter name starts
/// UNBOUND (mirroring a real call, where `check.rs::seed_parameters` states
/// nothing for `object`-typed parameters), asked under `truth = true` (the
/// question this narrowing site itself is asking: "given the call proved
/// True, what does that say"). Whatever the predicate's own parameter name
/// ends up bound to in that sandbox IS the proven set — read back and
/// copied onto the CALL's own first argument name in the real environment.
/// `is_age`'s `isinstance(v, int) and not isinstance(v, bool) and 0 <= v <=
/// 120` proves `v` down to exactly `Age`'s own set through this same
/// mechanism the ordinary top-level walk already uses for a seeded
/// parameter; `claims_age`'s bare `isinstance(v, int)` proves only the
/// unbounded `int` sort, which is NOT a subset of `Age` — so `return value`
/// against `-> Age` still fires, exactly as the row expects. A predicate
/// whose body is not this single-`return`-of-a-condition shape, or whose
/// own parameter never ends up bound in the sandbox (the condition proved
/// nothing this file's narrowing channels read), leaves the call's argument
/// untouched — the same "narrows nothing" default as any other declined
/// leaf.
fn narrow_type_guard_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>) {
    let Expr::Name(callee) = call.func.as_ref() else {
        return;
    };
    let Some(argument) = call.arguments.args.first() else {
        return;
    };
    let Some(argument_name) = name_of(argument) else {
        return;
    };
    if environment.read(argument_name).is_some() {
        return;
    }
    let Some(def) = environment.functions().and_then(|table| table.def(callee.id.as_str())) else {
        return;
    };
    // Skip every leading docstring (`is_age`'s own `"""honest TypeGuard...
    // """`) before requiring the SOLE remaining statement be a bare
    // `return` — the same docstring-shaped skip
    // `summaries::first_non_docstring_statement` applies to a callee body
    // elsewhere, inlined here since that function answers only the FIRST
    // such statement, not the remaining slice this needs.
    let non_docstring_body: Vec<&Stmt> = def
        .body
        .iter()
        .skip_while(|stmt| matches!(stmt, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::StringLiteral(_))))
        .collect();
    let [Stmt::Return(ret)] = non_docstring_body.as_slice() else {
        return;
    };
    let Some(condition) = ret.value.as_deref() else {
        return;
    };
    let Some(parameter) = def.parameters.posonlyargs.iter().chain(def.parameters.args.iter()).next() else {
        return;
    };
    let parameter_name = parameter.parameter.name.id.as_str();

    let sandbox = Environment::new(std::collections::HashSet::new());
    let sandbox = assume(condition, sandbox, kernel, true);
    let Some(proven) = sandbox.read(parameter_name) else {
        return;
    };
    environment.bind(argument_name, proven.clone());
}

/// `isinstance(name, int | float | bool)` (mission point 6): filters a
/// Values binding by `kind_tag`. `PrimitiveKind::Number` is the
/// sort-unknown numeric tag (AGENT-BRIEF.md, Wave-1 recognition facts —
/// int-vs-float is not yet distinguished at the value level except
/// where the syntax proves it), so a Number-tagged state passes
/// unfiltered both ways: this wave cannot prove which arm of an
/// int/float isinstance test it falls on.
///
/// A name the environment has NOT bound at all (an `object`-typed
/// parameter — `check.rs::seed_parameters` states nothing for the bare
/// `object` annotation, since no alias names it) is a SEPARATE case
/// from an existing Values binding: `environment.read` answers `None`,
/// not a Values state to filter. A name bound to `Kind::Unknown` — a
/// read this file genuinely determined NOTHING about (a subscript into
/// an unrecognized container shape, an unmodeled call's own result,
/// `abstract_value::unknown`'s own doc: "no fact reads through it at
/// all") — carries the identical absence of information, so it takes
/// the SAME seeding path rather than the "existing binding" arm below:
/// an `Unknown` value is not a state with members to filter, and
/// treating it as one that "already agrees" or "wholly disagrees"
/// with the test would be a claim this file never derived. Both cases
/// converge here as `no_information`. `isinstance(value, int)`/`float`
/// PROVING true (`truth` and no information) is itself the first fact
/// this environment learns about `value` — it seeds a fresh
/// `Kind::Set` binding holding the unbounded sort (the same set
/// `summaries.rs::return_sort_fallback`/`expressions.rs`'s `int(...)`
/// row build for a proved-but-unbounded `int`/`float`), grade
/// `TrustSpec` (the isinstance test is read, not executed — the same
/// grade `seed_parameters`'s own annotation-read seeding uses).
/// `isinstance(value, bool)` seeds `Kind::Values` instead: `bool`'s
/// domain is the two exact values `{0, 1}` (`string_models.rs`'s
/// `boolean_value` convention), not an unbounded ray, so it is not a
/// Set-kind sort seed. Proving FALSE, or a name already bound to a
/// READABLE value (however far from the sort being tested), never
/// seeds here — a falsified test says nothing positive about which
/// sort `value` DOES hold, and an existing readable binding is this
/// function's other, unchanged, arm below.
fn narrow_isinstance_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, truth: bool) {
    let Expr::Name(func_name) = call.func.as_ref() else {
        return;
    };
    if func_name.id.as_str() != "isinstance" {
        return;
    }
    if call.arguments.args.len() != 2 {
        return;
    }
    let Some(name) = name_of(&call.arguments.args[0]) else {
        return;
    };
    let Some(tags) = isinstance_type_tags(&call.arguments.args[1]) else {
        return;
    };
    let current = environment.read(name).cloned();
    let no_information = match &current {
        None => true,
        Some(value) => value.kind == Kind::Unknown,
    };
    if no_information {
        if truth {
            if let [tag] = tags.as_slice() {
                if let Some(seeded) = sort_seed(*tag) {
                    environment.bind(name, seeded);
                }
            }
        }
        return;
    }
    let current = current.expect("no_information false means Some was read above");
    // A KindUnion binding (json.loads's own honest return space,
    // `expressions.rs::json_loads_value_space`) narrows arm-by-arm: each
    // arm already carries the `kind_tag` an ordinary Values/Set binding
    // does, so `isinstance(x, float)` keeps only the arms whose tag
    // matches (`truth`) or excludes them (`!truth`) — the same filter
    // this function already runs on a single Values binding, applied
    // per arm instead of once. An arm with no `kind_tag` at all (the
    // list/dict arms, built via `opaque_value` on `Kind::Object`) never
    // matches a primitive tag either way, so it survives a `truth` test
    // only when the test is proving the union does NOT hold that sort
    // (`!truth` keeps it) and is dropped when `truth` asks for a sort it
    // cannot be. `kind_union_of` collapses the result: one surviving arm
    // answers bare, and no dropped arm decides the fold "no member
    // left standing" — that reading belongs to `arm_is_infeasible`
    // (Values-only today), not to this narrowing.
    if current.kind == Kind::KindUnion {
        let kept: Vec<AbstractValue> = current
            .arms
            .iter()
            .filter(|arm| {
                let matches_tag = arm.kind_tag.is_some_and(|tag| tags.contains(&tag));
                matches_tag == truth
            })
            .cloned()
            .collect();
        environment.bind(name, kind_union_of(kept));
        return;
    }
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // a sort-unknown Number state cannot be proved in, or proved out,
    // of an int/float isinstance test — pass through unfiltered
    if kind_tag == PrimitiveKind::Number {
        return;
    }
    let matches_tag = tags.contains(&kind_tag);
    if matches_tag == truth {
        // every member already agrees with the test — nothing to drop
        return;
    }
    // the whole binding disagrees with the test — every member is
    // infeasible under this arm
    let grade = trust_level_of(&current);
    environment.bind(name, known_values(Vec::new(), kind_tag, grade));
}

/// The fresh binding a PROVED `isinstance(x, tag)` seeds for a name the
/// environment held nothing about at all (see `narrow_isinstance_call`'s
/// doc): the unbounded `Kind::Set` ray for `int`/`float`, or `None` for
/// `bool` (and `Number`, which no `isinstance` argument ever names —
/// `isinstance_type_tags` only ever answers Integer/Float/Boolean) —
/// `bool`'s own two-value seed is built directly at the call site
/// instead, since it is `Kind::Values`, a different constructor
/// entirely.
fn sort_seed(tag: PrimitiveKind) -> Option<AbstractValue> {
    match tag {
        PrimitiveKind::Integer => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(unbounded_integers(), None, TrustSpec, SetKindTag::None)
        }),
        PrimitiveKind::Float => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(refined_sets::refinement_forms::numbers(), None, TrustSpec, SetKindTag::None)
        }),
        PrimitiveKind::Boolean => Some(known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec)),
        PrimitiveKind::Number | PrimitiveKind::String | PrimitiveKind::Array => None,
    }
}

/// The unbounded whole-number ray (every integer, no floor or ceiling)
/// — the same shape `summaries.rs`'s private `whole_integers` builds;
/// copied here rather than shared cross-file, matching this file's own
/// documented precedent (`literal_number`'s doc) for a small leaf
/// reader every narrowing-adjacent file keeps its own copy of.
fn unbounded_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

/// The `PrimitiveKind`s an `isinstance` second argument names, for
/// exactly the shapes mission point 6 covers: a bare type name
/// (`int`/`float`/`bool`), or a `|`-chain of them
/// (`isinstance(x, int | float)`). Any other shape (a tuple form, a
/// non-primitive type) answers `None` — not read this wave.
fn isinstance_type_tags(expression: &Expr) -> Option<Vec<PrimitiveKind>> {
    match expression {
        Expr::Name(name) => primitive_kind_of_type_name(name.id.as_str()).map(|tag| vec![tag]),
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let mut left = isinstance_type_tags(&binop.left)?;
            let right = isinstance_type_tags(&binop.right)?;
            left.extend(right);
            Some(left)
        }
        _ => None,
    }
}

fn primitive_kind_of_type_name(name: &str) -> Option<PrimitiveKind> {
    match name {
        "int" => Some(PrimitiveKind::Integer),
        "float" => Some(PrimitiveKind::Float),
        "bool" => Some(PrimitiveKind::Boolean),
        _ => None,
    }
}

/// Whether `expression` is the bare name of a tracked place — the only
/// shape every narrowing leaf here reads on the tested side (mission's
/// "filter what is known, never invent," matching the Go reference's
/// `onPlace` restriction to a single identifier).
fn name_of(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    }
}

fn is_none_literal(expression: &Expr) -> bool {
    matches!(expression, Expr::NoneLiteral(_))
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression). Mirrors the sibling
/// private helpers of the same shape in loops.rs/expressions.rs/
/// surface.rs — each narrowing-adjacent file keeps its own copy rather
/// than sharing a cross-file dependency for one small leaf reader.
fn literal_number(expression: &Expr) -> Option<f64> {
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

// ── the SET channel: condition → NarrowTree ─────────────────────────

/// A `NarrowTree` leaf that claims nothing — the kernel's own "no
/// reading" leaf (`gate_narrow`/`narrow_wire`'s `Other` arm never reads
/// its other fields), matching `refined-ts-go/internal/refinedts/
/// narrowing/type_guard_recognizers.go`'s package-level `Other` value.
fn other_tree() -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Other,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: None,
        b: None,
    }
}

/// One `Cmp` leaf (`name op literal`) — the bare-fields constructor
/// every other `NarrowTree` variant this file never builds also needs,
/// since the struct derives no `Default` (`refined_kernel::
/// narrow_questions` — every field explicit at every call site, the
/// same discipline that module's own tests follow).
fn cmp_tree(op: NarrowCmpOp, k: f64) -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Cmp,
        op: Some(op),
        k,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: None,
        b: None,
    }
}

fn not_tree(a: NarrowTree) -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Not,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: Some(Box::new(a)),
        b: None,
    }
}

fn and_or_tree(kind: NarrowTreeKind, a: NarrowTree, b: NarrowTree) -> NarrowTree {
    NarrowTree {
        kind,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: Some(Box::new(a)),
        b: Some(Box::new(b)),
    }
}

/// `NumericCmpOp` → the kernel's own `NarrowCmpOp` — `Eq`/`NotEq` are
/// NOT `Cmp` leaves (the kernel's `Cmp` carries only the four ORDER
/// operators; equality is its own `NarrowTreeKind::Eq`/its negation),
/// so this reads only the four this file's `Cmp` leaf can name; `None`
/// for `Eq`/`NotEq`, read directly by `condition_tree_of` instead.
fn narrow_cmp_op_of(op: NumericCmpOp) -> Option<NarrowCmpOp> {
    match op {
        NumericCmpOp::Lt => Some(NarrowCmpOp::Lt),
        NumericCmpOp::LtE => Some(NarrowCmpOp::Le),
        NumericCmpOp::Gt => Some(NarrowCmpOp::Gt),
        NumericCmpOp::GtE => Some(NarrowCmpOp::Ge),
        NumericCmpOp::Eq | NumericCmpOp::NotEq => None,
    }
}

/// Lowers `condition`, RELATIVE TO `place` (the one name the kernel's
/// narrowing question is always scoped to — `refined-ts-go`'s own
/// `TreeOf`/`leafTreeOf` convention), into the kernel's `NarrowTree`
/// grammar: `!`/`and`/`or` fold the same shape/structure the condition
/// itself has (a `not` node wraps its operand's own tree UNCHANGED —
/// the KERNEL's own `narrowQ` is what swaps a `not`'s when-true/
/// when-false pair, `set_functions/narrow.lean`'s `.not t => (p.2,
/// p.1)`, so this builder never needs to track which polarity it is
/// "under" the way the VALUES channel's `narrow`/`narrow_bool_op` does
/// — the caller (`narrow_set_kind_names`) reads whichever side of the
/// ONE resulting `NarrowAnswer` its own branch truth names).
///
/// Any leaf NOT on `place`, or NOT one of this file's recognized
/// shapes (a call other than `isinstance`, a string test, two changing
/// names, `is`/`is not None` — Python's `None` is never a member of a
/// `Kind::Set`'s own domain, so an absence test states nothing a set
/// claim could narrow), lowers to `other_tree()` — the honest "no
/// claim" leaf, never a guess. The WHOLE tree is `None` only when
/// `condition` itself has no shape this function reads at all (an
/// unreachable case today — every `Expr` arm below returns `Some`,
/// down to the catch-all `other_tree()` — kept `Option` so a future
/// leaf that must genuinely decline has the same "no tree at all" exit
/// `narrow_set_kind_names`'s `let Some(tree) = … else { continue }`
/// already expects).
fn condition_tree_of(condition: &Expr, place: &str) -> Option<NarrowTree> {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            condition_tree_of(&unary.operand, place).map(not_tree)
        }
        Expr::BoolOp(bool_op) => {
            let kind = match bool_op.op {
                BoolOp::And => NarrowTreeKind::And,
                BoolOp::Or => NarrowTreeKind::Or,
            };
            let mut trees = bool_op.values.iter().map(|value| condition_tree_of(value, place));
            let mut folded = trees.next()??;
            for next in trees {
                folded = and_or_tree(kind, folded, next?);
            }
            Some(folded)
        }
        Expr::Compare(compare) => Some(compare_tree_of(compare, place)),
        Expr::Call(call) => Some(call_leaf_tree_of(call, place)),
        _ => Some(other_tree()),
    }
}

/// `ExprCompare` → a `NarrowTree`: a chained comparison folds to the
/// `And` of its adjacent pairs (same CPython citation the VALUES
/// channel's `narrow_compare` follows) — this reading does not depend
/// on `truth` the way the VALUES channel's falsity short-circuit does,
/// since the kernel's own answer already carries BOTH the `whenTrue`
/// (the chain held) and `whenFalse` (the chain's negation — a
/// disjunction over which pair failed, which the kernel proves
/// directly rather than this file approximating it as "narrows
/// nothing").
fn compare_tree_of(compare: &ruff_python_ast::ExprCompare, place: &str) -> NarrowTree {
    let mut left = compare.left.as_ref();
    let mut folded: Option<NarrowTree> = None;
    for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
        let leaf = comparison_leaf_tree_of(left, *op, right, place);
        folded = Some(match folded {
            Some(existing) => and_or_tree(NarrowTreeKind::And, existing, leaf),
            None => leaf,
        });
        left = right;
    }
    folded.unwrap_or_else(other_tree)
}

/// One comparison pair (`left op right`) → a `NarrowTree` leaf, scoped
/// to `place`: a numeric literal on the other side lowers to `Cmp`
/// (`<`/`<=`/`>`/`>=`) or `Eq`/`not Eq` (`==`/`!=`); a literal
/// collection on the right of `in`/`not in` lowers to the membership
/// fold (`membership_leaf_tree_of`); anything else (`is`/`is not` —
/// read separately by the VALUES channel only, since `None` is never a
/// `Kind::Set` member; two changing names) is `other_tree()`.
fn comparison_leaf_tree_of(left: &Expr, op: CmpOp, right: &Expr, place: &str) -> NarrowTree {
    // `place in <collection>` / `place not in <collection>` — membership
    // against a literal collection of scalars, folded to the DISJUNCTION
    // of its members' own equality leaves (see `membership_leaf_tree_of`).
    if matches!(op, CmpOp::In | CmpOp::NotIn) {
        if name_of(left) != Some(place) {
            return other_tree();
        }
        let Some(leaf) = membership_leaf_tree_of(right) else {
            return other_tree();
        };
        return if op == CmpOp::In { leaf } else { not_tree(leaf) };
    }
    // A STRING-literal equality (`layout == "horizontal"`) lowers to the
    // kernel's own EqSeq leaf — the word's code points ride `points`
    // (set_functions/narrow.lean's `.eqSeq`), so a string-tuple-union
    // set (a `Literal["…", …]` alias) narrows to the named member on
    // the when-true side and its complement on the when-false side.
    // `!=` is the same leaf under Not. Ordering (`<` etc.) over strings
    // has no kernel leaf and stays `other_tree()` below.
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        let word = if name_of(left) == Some(place) {
            string_literal_points(right)
        } else if name_of(right) == Some(place) {
            string_literal_points(left)
        } else {
            None
        };
        if let Some(points) = word {
            let leaf = NarrowTree { kind: NarrowTreeKind::EqSeq, points, ..other_tree() };
            return if op == CmpOp::Eq { leaf } else { not_tree(leaf) };
        }
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return other_tree();
    };
    // `<place> ± k1 <op> k2` (`n - 1 >= 0`, B1.keep.join's own ternary
    // guard) — the tested side names no bare `place`, but an AFFINE SHIFT
    // of it (`affine_place_of`'s own doc). Read BEFORE the bare-name arms
    // below (which would otherwise silently fall through to `other_tree()`
    // for this shape, exactly the gap that left the ternary's own arms
    // unnarrowed): folding the shift's own literal into the comparison's
    // literal by the inverse operation turns "n - 1 >= 0" into "n >= 1",
    // a claim `place` itself, letting this leaf build the SAME `Cmp`/`Eq`
    // tree the bare-name arms below build. Checked on EITHER side (`n - 1
    // >= 0` or `0 <= n - 1`), mirroring `mirror_cmp_op`'s own two-sided
    // reading for a bare name.
    if let Some((on_place, literal)) = affine_comparison_literal(left, right, place) {
        let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
        return numeric_comparison_tree(effective_op, literal);
    }
    let (on_place, literal) = if name_of(left) == Some(place) {
        (true, literal_number(right))
    } else if name_of(right) == Some(place) {
        (false, literal_number(left))
    } else {
        return other_tree();
    };
    let Some(literal) = literal else {
        return other_tree();
    };
    let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
    numeric_comparison_tree(effective_op, literal)
}

/// The `NarrowTree` a numeric comparison's own effective operator and
/// literal build — `Eq`/`NotEq` fold to the kernel's own `Eq` leaf (never
/// `Cmp`, which carries only the four order operators), every other
/// operator folds to `Cmp`. Shared by `comparison_leaf_tree_of`'s bare-name
/// arm and its affine-shift arm so the two build the identical tree shape
/// once the effective operator and literal are known.
fn numeric_comparison_tree(effective_op: NumericCmpOp, literal: f64) -> NarrowTree {
    match effective_op {
        NumericCmpOp::Eq => NarrowTree { kind: NarrowTreeKind::Eq, k: literal, ..other_tree() },
        NumericCmpOp::NotEq => not_tree(NarrowTree { kind: NarrowTreeKind::Eq, k: literal, ..other_tree() }),
        _ => {
            let kernel_op = narrow_cmp_op_of(effective_op).expect("Eq/NotEq handled above");
            cmp_tree(kernel_op, literal)
        }
    }
}

/// Whether `expression` is `<place> + k` or `<place> - k` — a literal
/// AFFINE SHIFT of the tested place (`n - 1`, `n + 1`), for a literal `k`
/// this file already reads (`literal_number`). The shift amount `k`, or
/// `None` for any other shape (a bare name, a shift of a DIFFERENT name,
/// two changing names, a non-literal offset).
fn affine_shift_of_place(expression: &Expr, place: &str) -> Option<f64> {
    let Expr::BinOp(binop) = expression else {
        return None;
    };
    if name_of(&binop.left) != Some(place) {
        return None;
    }
    let offset = literal_number(&binop.right)?;
    match binop.op {
        ruff_python_ast::Operator::Add => Some(offset),
        ruff_python_ast::Operator::Sub => Some(-offset),
        _ => None,
    }
}

/// The BASE NAME inside an affine shift (`n - 1` names `n`), for whichever
/// name sits there — `collect_names`'s own place collector needs this
/// (unlike `affine_shift_of_place`, which is only asked once `place` is
/// already known) to discover that a comparison like `n - 1 >= 0` is
/// relevant to `n` at all, before the SET channel's per-name loop can ask
/// `condition_tree_of` to build a tree relative to it. `None` for any
/// other shape (a bare name — read separately by `name_of` — a
/// non-literal offset, an operator other than `+`/`-`).
fn affine_shifted_name_of(expression: &Expr) -> Option<&str> {
    let Expr::BinOp(binop) = expression else {
        return None;
    };
    if !matches!(binop.op, ruff_python_ast::Operator::Add | ruff_python_ast::Operator::Sub) {
        return None;
    }
    literal_number(&binop.right)?;
    name_of(&binop.left)
}

/// One comparison pair's own AFFINE-SHIFT reading: `<place> ± k1 <op> k2`
/// (`n - 1 >= 0`) or the mirrored `k2 <op> <place> ± k1` (`0 <= n - 1`) —
/// `place` sits inside an affine shift on one side, a plain literal on the
/// other. Answers `(on_place, effective_literal)`: `on_place` tells the
/// caller which side `place`'s own shift sits on (so it can still mirror
/// the comparison operator the same way the bare-name arm does), and
/// `effective_literal` is the comparison's own literal with the shift's
/// offset folded in — "n - 1 >= 0" is exactly "n >= 0 + 1", so the shift
/// (`-1`) is subtracted back out: `effective_literal = other_literal -
/// shift`. `None` when neither side is an affine shift of `place`, or the
/// OTHER side is not a plain literal (a shift compared to a second
/// changing expression states no single-literal claim this leaf can fold).
fn affine_comparison_literal(left: &Expr, right: &Expr, place: &str) -> Option<(bool, f64)> {
    if let Some(shift) = affine_shift_of_place(left, place) {
        let other = literal_number(right)?;
        return Some((true, other - shift));
    }
    if let Some(shift) = affine_shift_of_place(right, place) {
        let other = literal_number(left)?;
        return Some((false, other - shift));
    }
    None
}

/// A plain string literal's code points, one f64 per point — the word
/// an `EqSeq` leaf carries. Any other expression (an f-string, a
/// concatenation, a name) is `None`; only the literal's own spelling
/// is a proved word.
fn string_literal_points(expr: &Expr) -> Option<Vec<f64>> {
    let Expr::StringLiteral(literal) = expr else {
        return None;
    };
    Some(literal.value.to_str().chars().map(|c| c as u32 as f64).collect())
}

/// `place in <collection>` as a `NarrowTree`: a literal list/tuple/set
/// of scalars folds to the DISJUNCTION of its members' own equality
/// leaves — `x in [1, 2, 3]` becomes `Or(Eq 1, Or(Eq 2, Eq 3))`.
///
/// That fold IS membership under the kernel's own `narrowQ`
/// (`set_functions/narrow.lean`), on both sides at once, with no new
/// leaf needed. Truth: `orClaim` unions the members' singletons into
/// exactly the one-of set, STRONG (every disjunct's own truth claim is
/// strong, and `orClaim` keeps strength only when all are — each `Eq`
/// leaf's holding proves the value real, so the union does too).
/// Falsity: `andClaim` intersects the members' own real-difference
/// claims, WEAK — ℝ̄ minus every listed value, which is precisely what
/// `not in` proves for a value already known real. Both claims are the
/// ones `narrowQ_sound` already proves; the fold buys the whole `in`
/// vocabulary at the existing soundness theorem's price.
///
/// The kernel's `inSet` leaf is NOT the route here: it is a SEQUENCE-
/// world leaf (`leafEval`'s own `.inSet _, _, _ => False` gives it no
/// scalar runs, and its falsity claim is `diffSet stringsSet S` — C*
/// minus the set, a claim about strings). A numeric place tested for
/// membership belongs in the scalar world, and the `Eq`/`Or` fold puts
/// it there.
///
/// Members must share ONE sort — all numeric, or all string — matching
/// the boundary's own refusal to mix the worlds in one tree
/// (`exports_narrow.lean`'s `treeScalarClaim && treeSeqClaim` check,
/// which FAILS the whole question for a mixed tree). A mixed or empty
/// collection, a non-literal collection (a name, a comprehension, a
/// call), or any member this file cannot read as a literal, answers
/// `None` — the caller lowers `other_tree()`, narrowing nothing.
///
/// A DICT is never read: `x in {...}` tests the dict's KEYS, and a
/// `dict` display's keys are a different collection from the members a
/// list display names. Declining is conservative, never wrong.
fn membership_leaf_tree_of(collection: &Expr) -> Option<NarrowTree> {
    let elements: &[Expr] = match collection {
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::Set(set) => &set.elts,
        _ => return None,
    };
    if elements.is_empty() {
        return None;
    }
    // one sort per collection: read every member as a number, or every
    // member as a string, and decline the moment the two mix
    let leaves: Option<Vec<NarrowTree>> = elements
        .iter()
        .map(|element| {
            literal_number(element)
                .map(|k| NarrowTree { kind: NarrowTreeKind::Eq, k, ..other_tree() })
                .or_else(|| {
                    string_literal_points(element)
                        .map(|points| NarrowTree { kind: NarrowTreeKind::EqSeq, points, ..other_tree() })
                })
        })
        .collect();
    let leaves = leaves?;
    let all_numeric = leaves.iter().all(|leaf| leaf.kind == NarrowTreeKind::Eq);
    let all_words = leaves.iter().all(|leaf| leaf.kind == NarrowTreeKind::EqSeq);
    if !all_numeric && !all_words {
        return None;
    }
    let mut folded = None;
    for leaf in leaves {
        folded = Some(match folded {
            Some(existing) => and_or_tree(NarrowTreeKind::Or, existing, leaf),
            None => leaf,
        });
    }
    folded
}

/// The SET channel's own dispatcher for a bare `Expr::Call` test:
/// `<place>.is_integer()` reads as the `IsInt AND IsFinite` leaf
/// (`is_integer_leaf_tree_of`'s own doc); every other call, INCLUDING
/// `isinstance(...)`, is `other_tree()` (`isinstance_leaf_tree_of`'s own
/// doc — a sort claim the kernel's COMPARISON/MEMBERSHIP vocabulary
/// cannot further express about a Set already scoped to that sort).
fn call_leaf_tree_of(call: &ruff_python_ast::ExprCall, place: &str) -> NarrowTree {
    if let Some(leaf) = is_integer_leaf_tree_of(call, place) {
        return leaf;
    }
    isinstance_leaf_tree_of(call, place)
}

/// `<place>.is_integer()` as a `NarrowTree` leaf, `None` for any other
/// call shape (a different method name, a non-empty argument list, or a
/// receiver that is not the bare name `place`) — the SET-channel twin of
/// `expressions.rs`'s own single-known-value `is_integer` row
/// (`stdtypes.rst`, `float.is_integer()`: "Return True if the float
/// instance is finite with integral value, and False otherwise"). A
/// `place` currently bound `Kind::Set` has no one value to test that way,
/// so this states the SAME two-part claim as a kernel leaf instead:
/// `IsInt` (integral within ℝ̄) AND `IsFinite` (excludes both
/// infinities) — `is_integer()` on `float('inf')` is `False` precisely
/// because the finite half fails, so `IsInt` alone would overclaim on an
/// unbounded-above parameter like `to_page_size`'s own `x: float`
/// (showcase.py) the way this leaf's own construct citation names.
fn is_integer_leaf_tree_of(call: &ruff_python_ast::ExprCall, place: &str) -> Option<NarrowTree> {
    let Expr::Attribute(attribute) = call.func.as_ref() else { return None };
    if attribute.attr.as_str() != "is_integer" {
        return None;
    }
    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return None;
    }
    if name_of(&attribute.value) != Some(place) {
        return None;
    }
    let is_int = NarrowTree { kind: NarrowTreeKind::IsInt, ..other_tree() };
    let is_finite = NarrowTree { kind: NarrowTreeKind::IsFinite, ..other_tree() };
    Some(and_or_tree(NarrowTreeKind::And, is_int, is_finite))
}

/// `isinstance(place, ...)` as a `NarrowTree` leaf: the kernel's own
/// grammar has no "Python sort" leaf (`IsInt` tests integrality within
/// ℝ̄, not "is this Python `int`") — a sort claim is entirely this
/// file's own `narrow_isinstance_call`/`sort_seed` job, on the VALUES
/// channel or at seeding time, never the kernel's. `other_tree()`
/// always: an `isinstance` test says nothing further the kernel's
/// COMPARISON/MEMBERSHIP vocabulary can express about a Set already
/// scoped to that sort.
fn isinstance_leaf_tree_of(_call: &ruff_python_ast::ExprCall, _place: &str) -> NarrowTree {
    other_tree()
}

/// Whether `tree` states anything at all — an all-`Other` tree asks
/// the kernel a question with no answer worth having (`refined-ts-go`'s
/// own `SaysAnything`).
fn says_anything(tree: &NarrowTree) -> bool {
    match tree.kind {
        NarrowTreeKind::Other => false,
        NarrowTreeKind::Not => says_anything(tree.a.as_deref().expect("not carries A")),
        NarrowTreeKind::And | NarrowTreeKind::Or => {
            says_anything(tree.a.as_deref().expect("and/or carries A"))
                || says_anything(tree.b.as_deref().expect("and/or carries B"))
        }
        _ => true,
    }
}

/// Meets a kernel narrowing claim into `current`'s own set: the
/// INTERSECTION of `current.set`'s forms with `claim_set`'s forms
/// (`RefinedSet`'s forms conjoin — the same reading
/// `refined_domain::lattice_operations::meet_known`'s own Set×Set
/// branch takes), keeping `current`'s `kind_tag` (the kernel's claim
/// carries no Python sort tag of its own) and `current`'s own trust
/// grade (never claimed stronger by a narrowing than the value that
/// flowed in — `loops.rs`'s `kernel_bounded_counter_environment` binds
/// its own kernel answer at the SAME grade the entry binding carried,
/// the matching precedent).
fn meet_set_answer(current: &AbstractValue, claim_set: &RefinedSet) -> AbstractValue {
    let mut combined = current.set.forms.clone();
    combined.extend(claim_set.forms.clone());
    let grade = trust_level_of(current);
    AbstractValue {
        kind_tag: current.kind_tag,
        ..known_set(make_refined_set(combined), None, grade, current.set_kind_tag)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_domain::abstract_value::known_values;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_parser::parse_expression;

    use super::*;

    /// A kernel handle for tests that never ask it anything — `assume`
    /// takes the parameter for the frozen signature's sake, but no
    /// construct this wave asks a question of it. `None` when the
    /// native dylib artifact has not been built, so this file's tests
    /// run without requiring `pnpm kernel:native` first.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn environment_with(name: &str, values: Vec<f64>, kind_tag: PrimitiveKind) -> Environment {
        let mut locally_bound = HashSet::new();
        locally_bound.insert(name.to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind(name, known_values(values, kind_tag, TrustProved));
        environment
    }

    fn assumed(source: &str, environment: Environment, truth: bool) -> Option<Environment> {
        let kernel = loaded_kernel()?;
        let parsed = parse_expression(source).expect("test source must parse");
        let expression = parsed.into_expr();
        Some(assume(&expression, environment, &kernel, truth))
    }

    #[test]
    fn test_greater_than_literal_keeps_satisfying_drops_others() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![200.0]);
    }

    #[test]
    fn test_greater_than_literal_falsity_flips_the_kept_side() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x > 100", environment, false) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_chained_comparison_keeps_the_middle_window() {
        let environment = environment_with("x", vec![-5.0, 0.0, 60.0, 120.0, 200.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![0.0, 60.0, 120.0]);
    }

    #[test]
    fn test_equality_against_literal_keeps_only_that_value() {
        let environment = environment_with("x", vec![40.0, 41.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x == 40", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    /// `guard_narrowed_values`'s own pin — a match arm's guard read as a
    /// narrowing through the SAME `assume` machinery
    /// `test_equality_against_literal_keeps_only_that_value` above
    /// exercises directly, but through the sandbox-and-read-back path
    /// `match_arms.rs`'s guarded bare-capture split calls: `x == 1` over
    /// `{1, 2, 4}` narrows to exactly `{1}` on the admitted (`truth:
    /// true`) side.
    #[test]
    fn test_guard_narrowed_values_keeps_the_admitted_side() {
        let Some(kernel) = loaded_kernel() else { return };
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let parsed = parse_expression("x == 1").expect("test source must parse");
        let narrowed = guard_narrowed_values(&parsed.into_expr(), "x", &subject, &kernel, true)
            .expect("a single equality comparison is a guard shape this reader proves");
        assert_eq!(narrowed.values, vec![1.0]);
    }

    /// The excluded (`truth: false`) side of the same guard: `x == 1`
    /// being false over `{1, 2, 4}` leaves exactly `{2, 4}`.
    #[test]
    fn test_guard_narrowed_values_keeps_the_excluded_side() {
        let Some(kernel) = loaded_kernel() else { return };
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let parsed = parse_expression("x == 1").expect("test source must parse");
        let narrowed = guard_narrowed_values(&parsed.into_expr(), "x", &subject, &kernel, false)
            .expect("a single equality comparison is a guard shape this reader proves");
        let mut values = narrowed.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![2.0, 4.0]);
    }

    /// A guard shape `assume` narrows nothing for (`x.bit_length() > 0` —
    /// a method call on the guard's own subject, which none of this
    /// file's comparison, membership, or type-guard leaves recognize)
    /// leaves the binding UNCHANGED, so `guard_narrowed_values` declines
    /// outright — an unchanged binding is never read as a proof every
    /// member survives; it is the absence of a proof.
    #[test]
    fn test_guard_narrowed_values_declines_when_assume_narrows_nothing() {
        let Some(kernel) = loaded_kernel() else { return };
        let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
        let parsed = parse_expression("x.bit_length() > 0").expect("test source must parse");
        let narrowed = guard_narrowed_values(&parsed.into_expr(), "x", &subject, &kernel, true);
        assert!(
            narrowed.is_none(),
            "an unproved guard shape leaves the binding unchanged — never read as a genuine narrowing"
        );
    }

    #[test]
    fn test_not_wrapped_comparison_flips_truth() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("not (x > 100)", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_and_narrows_both_names() {
        let mut locally_bound = HashSet::new();
        locally_bound.insert("a".to_owned());
        locally_bound.insert("b".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("a", known_values(vec![-1.0, 5.0], PrimitiveKind::Number, TrustProved));
        environment.bind("b", known_values(vec![-2.0, 7.0], PrimitiveKind::Number, TrustProved));
        let Some(narrowed) = assumed("a > 0 and b > 0", environment, true) else {
            return;
        };
        let a = narrowed.read("a").expect("a still bound");
        let b = narrowed.read("b").expect("b still bound");
        assert_eq!(a.values, vec![5.0]);
        assert_eq!(b.values, vec![7.0]);
    }

    #[test]
    fn test_non_values_binding_untouched() {
        use refined_domain::abstract_value::null_value;
        let mut locally_bound = HashSet::new();
        locally_bound.insert("x".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("x", null_value());
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Null);
    }

    #[test]
    fn test_unbound_name_untouched() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        assert!(narrowed.read("x").is_none());
    }

    // ── TypeGuard/TypeIs: recognized, never trusted ───────────────────

    /// An environment carrying a same-module function table with one
    /// `def`, parsed from `source` — the shape `recognizes_type_guard_
    /// call` reads via `environment.functions()`.
    fn environment_with_function_table(source: &str) -> Environment {
        let module = ruff_python_parser::parse_module(source).expect("test source must parse").into_syntax();
        let table = crate::function_table::function_table(&module);
        let mut environment = Environment::new(HashSet::new());
        environment.set_functions(Arc::new(table));
        environment
    }

    /// A call to a `TypeGuard[X]`-annotated same-module predicate narrows
    /// an unbound name to what the predicate's OWN BODY proves, never to
    /// the annotation's claimed `X` (`recognizes_type_guard_call`'s own
    /// doc: trusting the claim unverified would read `dishonest_predicate`
    /// silent when the row expects a fire). This predicate's body only
    /// proves `isinstance(v, int)` — a weaker claim than `Age` — so
    /// `value` narrows to the unbounded `int` sort, not `Age`.
    #[test]
    fn test_type_guard_call_narrows_an_unbound_name_to_its_bodys_own_proof() {
        let environment = environment_with_function_table(concat!(
            "def is_age(v: object) -> TypeGuard[Age]:\n",
            "    return isinstance(v, int)\n",
        ));
        let Some(narrowed) = assumed("is_age(value)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("the body's own proof seeds a binding");
        assert_eq!(value.kind, Kind::Set, "the proof is a sort, not an exact value");
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// The same recognition for `TypeIs[X]` (typing.rst's narrower
    /// sibling of `TypeGuard`) — the same syntactic shape, same
    /// proof-not-claim narrowing.
    #[test]
    fn test_type_is_call_narrows_an_unbound_name_to_its_bodys_own_proof() {
        let environment = environment_with_function_table(concat!(
            "def is_age(v: object) -> TypeIs[Age]:\n",
            "    return isinstance(v, int)\n",
        ));
        let Some(narrowed) = assumed("is_age(value)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("the body's own proof seeds a binding");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// A call to a function with NO `TypeGuard`/`TypeIs` return
    /// annotation is not recognized by this reader at all — it falls
    /// through to `narrow_isinstance_call`'s own decline for a
    /// non-`isinstance` callee, the same untouched outcome, but through
    /// the ordinary path rather than this one.
    #[test]
    fn test_plain_predicate_call_is_not_recognized_as_a_type_guard() {
        let environment = environment_with_function_table(concat!(
            "def is_age(v: object) -> bool:\n",
            "    return isinstance(v, int)\n",
        ));
        let Some(narrowed) = assumed("is_age(value)", environment, true) else {
            return;
        };
        assert!(narrowed.read("value").is_none());
    }

    /// An EXISTING binding of a name a `TypeGuard` call names is also
    /// left untouched — the decline applies regardless of whether the
    /// name was previously bound.
    #[test]
    fn test_type_guard_call_does_not_narrow_an_existing_binding() {
        let mut environment = environment_with_function_table(concat!(
            "def is_age(v: object) -> TypeGuard[Age]:\n",
            "    return isinstance(v, int)\n",
        ));
        environment.bind("value", known_values(vec![200.0], PrimitiveKind::Number, TrustProved));
        let Some(narrowed) = assumed("is_age(value)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("value stays bound");
        assert_eq!(value.values, vec![200.0], "the pre-existing binding survives unchanged");
    }

    /// f-type-nodes.py's own honest/dishonest contrast, run end to end
    /// through `assume`: `is_age`'s body chains `isinstance(v, int) and
    /// not isinstance(v, bool) and 0 <= v <= 120` — the SAME shape the
    /// module doc names as the SET channel's own canonical example — so
    /// the proof narrows `value` all the way down to a bounded `[0, 120]`
    /// integer window, a strict subset of the unbounded `int` sort.
    /// Needs a live kernel: the bound comparison narrows through the SET
    /// channel's own kernel question, not the VALUES channel alone.
    #[test]
    fn test_an_honest_type_guard_narrows_to_a_bounded_window() {
        let environment = environment_with_function_table(concat!(
            "def is_age(v: object) -> TypeGuard[Age]:\n",
            "    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 120\n",
        ));
        let Some(narrowed) = assumed("is_age(value)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("the bounded proof seeds a binding");
        assert_eq!(value.kind, Kind::Set, "a bounded window is still a Set-kind proof, not an exact value");
    }

    // ── the SET channel ──────────────────────────────────────────────

    fn environment_with_set(name: &str, set: RefinedSet, kind_tag: PrimitiveKind) -> Environment {
        let mut locally_bound = HashSet::new();
        locally_bound.insert(name.to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind(name, AbstractValue { kind_tag: Some(kind_tag), ..known_set(set, None, TrustProved, SetKindTag::None) });
        environment
    }

    /// `>` on a Set-kind binding intersects the kernel's claim into the
    /// current set — `x > 0` on the unbounded integer ray narrows to
    /// the open-above-zero integer ray, which the assignability law's
    /// own containment ask would then judge against a declared window.
    #[test]
    fn test_set_kind_greater_than_literal_intersects_the_kernel_claim() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x > 0", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Set);
        assert_eq!(x.kind_tag, Some(PrimitiveKind::Integer));
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), refined_sets::refinement_forms::above(0.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            expected
        );
    }

    /// `>=` mirrors the same leaf with the inclusive operator.
    #[test]
    fn test_set_kind_greater_than_or_equal_intersects() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x >= 0", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), at_least(0.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            expected
        );
    }

    /// `n - 1 >= 0` (B1.keep.join's own ternary guard, `n - 1 if n - 1 >=
    /// 0 else 0`) narrows `n` ITSELF, not `n - 1` (which is not a place
    /// this file's environment binds at all) — the affine-shift reading
    /// `comparison_leaf_tree_of` folds before falling through to
    /// `other_tree()`. `n - 1 >= 0` is exactly `n >= 1`, so this asks the
    /// SAME question `test_set_kind_greater_than_or_equal_intersects`
    /// asks with a literal `1` in place of `0`.
    #[test]
    fn test_set_kind_affine_shift_left_narrows_the_base_place() {
        let environment = environment_with_set("n", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("n - 1 >= 0", environment, true) else {
            return;
        };
        let n = narrowed.read("n").expect("n still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), at_least(1.0)]);
        assert!(
            (kernel.scalar_subset)(&n.set, &expected) && (kernel.scalar_subset)(&expected, &n.set),
            "n.set = {:?}, want the same set as {:?}",
            n.set,
            expected
        );
    }

    /// The mirrored spelling, `0 <= n - 1` — the affine shift sits on the
    /// RIGHT of the comparison, so the effective operator mirrors too
    /// (the same `mirror_cmp_op` reading the bare-name arm already takes
    /// for a literal-on-the-left comparison), landing the identical `n >=
    /// 1` claim as the left-shifted spelling above.
    #[test]
    fn test_set_kind_affine_shift_right_narrows_the_base_place_with_mirrored_operator() {
        let environment = environment_with_set("n", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("0 <= n - 1", environment, true) else {
            return;
        };
        let n = narrowed.read("n").expect("n still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), at_least(1.0)]);
        assert!(
            (kernel.scalar_subset)(&n.set, &expected) && (kernel.scalar_subset)(&expected, &n.set),
            "n.set = {:?}, want the same set as {:?}",
            n.set,
            expected
        );
    }

    // ── the ACCESS-PATH channel ──────────────────────────────────────

    /// A15.guard.eq/A15.guard.ne's own shape: `0 <= a.n <= 150` narrows
    /// the PATH `a.n`, not the bare name `a` (which this environment
    /// never even binds a Values/Set fact for — `a` is a class-instance
    /// receiver, not a number). `env::tracked_place_of`'s own chain
    /// reading finds `a.n`, and `narrow_path_window` tightens the SAME
    /// `{lo, hi}` window shape a length comparison already tightens,
    /// seeded fresh from the unbounded integer ray on first touch.
    #[test]
    fn test_path_chained_comparison_narrows_the_attribute_chain() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut locally_bound = HashSet::new();
        locally_bound.insert("a".to_owned());
        let environment = Environment::new(locally_bound);
        let parsed = parse_expression("0 <= a.n <= 150").expect("test source must parse");
        let narrowed = assume(&parsed.into_expr(), environment, &kernel, true);
        let place = crate::env::TrackedPlace::bare("a").extend("n");
        let a_n = narrowed.read_path(&place).expect("a.n's own path fact is bound");
        assert_eq!(a_n.kind, Kind::Set);
        let expected = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(150.0)]);
        assert!(
            (kernel.scalar_subset)(&a_n.set, &expected) && (kernel.scalar_subset)(&expected, &a_n.set),
            "a.n's set = {:?}, want the same set as {:?}",
            a_n.set,
            expected
        );
    }

    /// A write to the base name forgets every path fact rooted at it
    /// (`env::Environment::forget`'s own doc) — the one forget resolver
    /// this channel relies on to never leave a stale `a.n` fact standing
    /// once `a` itself is reassigned to a DIFFERENT instance.
    #[test]
    fn test_forgetting_the_base_name_drops_its_own_path_facts() {
        let mut environment = Environment::new(HashSet::new());
        let place = crate::env::TrackedPlace::bare("a").extend("n");
        environment.bind_path(&place, known_values(vec![40.0], PrimitiveKind::Integer, TrustProved));
        assert!(environment.read_path(&place).is_some());
        environment.forget("a");
        assert!(environment.read_path(&place).is_none(), "a write to the base must drop its path facts");
    }

    /// A write to a PREFIX of a deeper path forgets every path
    /// continuing it, but leaves an unrelated sibling untouched
    /// (`TrackedPlace::extends`'s own doc) — `a.n` write drops `a.n.x`,
    /// never `a.m`.
    #[test]
    fn test_forgetting_a_path_prefix_drops_continuations_but_not_siblings() {
        let mut environment = Environment::new(HashSet::new());
        let a_n = crate::env::TrackedPlace::bare("a").extend("n");
        let a_n_x = a_n.extend("x");
        let a_m = crate::env::TrackedPlace::bare("a").extend("m");
        environment.bind_path(&a_n_x, known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
        environment.bind_path(&a_m, known_values(vec![2.0], PrimitiveKind::Integer, TrustProved));
        environment.forget_path_base(&a_n);
        assert!(environment.read_path(&a_n_x).is_none(), "a.n.x continues the written prefix a.n");
        assert!(environment.read_path(&a_m).is_some(), "a.m is an unrelated sibling of a.n");
    }

    /// `0 <= x <= 120` — b-body-expressions.py's `len_in_guard` shape
    /// (~b:649) once `x` already carries the unbounded integer ray:
    /// the chained comparison's `And` tree intersects BOTH bounds in
    /// one kernel ask, landing the exact `[0, 120]` integer window
    /// `Age` (the fixture's own declared alias) admits.
    #[test]
    fn test_set_kind_chained_comparison_intersects_both_bounds() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &age) && (kernel.scalar_subset)(&age, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            age
        );
    }

    /// `x >= 0` alone (no upper bound) — b-body-expressions.py's
    /// `guard_over_ceiling` shape (~b:656): the narrowed set is `[0,
    /// ∞) ∩ ℤ`, which is NOT a subset of `Age`'s `[0, 120]` window (it
    /// still admits 200) — this is the fixture's own marked fire,
    /// proved here at the SET level (the assignability law's own
    /// containment ask is what actually fires it at the sink; this
    /// test pins that the narrowed set is exactly what admits the
    /// ceiling violation, not something already tighter).
    #[test]
    fn test_set_kind_single_sided_bound_still_admits_the_ceiling() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x >= 0", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            !(kernel.scalar_subset)(&x.set, &age),
            "x.set = {:?} must still admit values above 120 (200, …)",
            x.set
        );
    }

    /// `and` composes two Set-kind leaves on the SAME name into one
    /// tree, same as the chained-comparison test above but spelled as
    /// an explicit `and`.
    #[test]
    fn test_set_kind_and_composes_both_leaves() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x >= 0 and x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &age) && (kernel.scalar_subset)(&age, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            age
        );
    }

    /// `not (x > 120)` on a Set-kind binding — De Morgan through `not`
    /// folds to the kernel's own `whenFalse` claim for `x > 120`
    /// (`¬(x > 120)` is `x <= 120`), landing the at-most-120 half of
    /// the integer ray.
    #[test]
    fn test_set_kind_not_wrapped_comparison_uses_the_kernel_negation() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("not (x > 120)", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            expected
        );
    }

    // ── `in` / `not in` membership ───────────────────────────────────

    /// `x in [1, 2, 3]` on a Set-kind binding narrows to exactly those
    /// three values: the `Or`-fold of the members' own `Eq` leaves has
    /// the kernel union their singletons into the one-of set.
    #[test]
    fn test_in_a_literal_list_narrows_to_the_member_set() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in [1, 2, 3]", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Set);
        let Some(kernel) = loaded_kernel() else { return };
        let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
        assert!(
            (kernel.scalar_subset)(&x.set, &members) && (kernel.scalar_subset)(&members, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            members
        );
    }

    /// A tuple and a set display are the same membership question as a
    /// list — `x in (1, 2, 3)` narrows identically.
    #[test]
    fn test_in_a_literal_tuple_narrows_the_same_way() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in (1, 2, 3)", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
        assert!(
            (kernel.scalar_subset)(&x.set, &members) && (kernel.scalar_subset)(&members, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            members
        );
    }

    /// The COMPLEMENT: `x in [1, 2, 3]` proving FALSE (and its `not in`
    /// spelling proving true) leaves a set that still admits values
    /// outside the list — and no longer admits the listed ones. Pinned
    /// as "200 survives, 2 does not," the two facts the claim states.
    #[test]
    fn test_not_in_a_literal_list_drops_the_members_and_keeps_the_rest() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x not in [1, 2, 3]", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
        assert!(
            !(kernel.scalar_subset)(&x.set, &members),
            "x.set = {:?} must not be inside the very set it excludes",
            x.set
        );
        let two = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[2.0])]);
        assert!(
            !(kernel.scalar_subset)(&two, &x.set),
            "x.set = {:?} must no longer admit 2",
            x.set
        );
    }

    /// `not in` proving FALSE is membership again — the kernel's own
    /// `Not` swaps the sides, so this lands the same one-of set the
    /// plain `in` truth arm does.
    #[test]
    fn test_not_in_proving_false_is_membership() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x not in [1, 2, 3]", environment, false) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let members = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[1.0, 2.0, 3.0])]);
        assert!(
            (kernel.scalar_subset)(&x.set, &members) && (kernel.scalar_subset)(&members, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            members
        );
    }

    /// A MIXED collection (a number beside a word) states nothing this
    /// file lowers: the boundary refuses a tree mixing the numeric and
    /// string worlds outright, so the leaf declines before asking and
    /// the binding is left exactly as it was.
    #[test]
    fn test_in_a_mixed_sort_collection_narrows_nothing() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in [1, \"two\"]", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, unbounded_integers());
    }

    /// A collection with a member this file cannot read as a literal (a
    /// name) declines the whole leaf — never a partial reading of the
    /// members it happened to recognize.
    #[test]
    fn test_in_a_collection_holding_a_name_narrows_nothing() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in [1, some_name]", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, unbounded_integers());
    }

    /// A DICT display tests the dict's KEYS, a different collection from
    /// the members a list display names — declined, narrowing nothing.
    #[test]
    fn test_in_a_dict_display_narrows_nothing() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in {1: 'a', 2: 'b'}", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, unbounded_integers());
    }

    /// Membership with the PLACE on the collection side (`1 in x`) is a
    /// different question entirely — it tests the place's own contents,
    /// not its value — and narrows nothing.
    #[test]
    fn test_place_on_the_collection_side_narrows_nothing() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("1 in x", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, unbounded_integers());
    }

    /// A leaf this file cannot read at all (`x in y` — membership
    /// against a collection that is not a literal display this file
    /// reads) lowers to `other_tree()`; the whole tree says nothing
    /// (`says_anything` false), so the binding is left exactly as it
    /// was — never narrowed, never refused.
    #[test]
    fn test_an_unreadable_leaf_shape_leaves_the_set_binding_untouched() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in y", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, unbounded_integers());
    }

    /// `isinstance(value, int)` on a name the environment has bound
    /// NOTHING for (the `object`-typed parameter shape,
    /// b-body-expressions.py's `len_in_guard`/`guard_over_ceiling`)
    /// seeds a fresh `Kind::Set` holding the unbounded integer ray —
    /// the "(a seeded parameter, a sort-set)" case `assume`'s own
    /// module doc names.
    #[test]
    fn test_isinstance_int_seeds_a_fresh_integer_set_from_unbound() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("isinstance seeded value");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        assert_eq!(value.set, unbounded_integers());
    }

    /// `isinstance(value, int)` on a name the environment HAS bound —
    /// but to `Kind::Unknown` (a subscript into an unrecognized
    /// container shape, `expressions.rs::evaluate_subscript`'s own
    /// `unknown()` fallback for `parsed["value"]` over a `json.loads`
    /// `Kind::KindUnion` result — `collection_models::subscript_read`
    /// carries no `Kind::KindUnion` arm) — takes the SAME seeding path
    /// the unbound case above does, not the "existing binding" arm: an
    /// `Unknown` value states no information for the isinstance test to
    /// filter, disagree with, or agree with, so a guard re-establishing
    /// the sort over it is the honest reading, mirroring the e2e fixture
    /// A10.edge.json's own `json_inside` row (`value = parsed["value"]`
    /// guarded by `isinstance(value, int) and 0 <= value <= 150` before
    /// `return value`).
    #[test]
    fn test_isinstance_int_seeds_a_fresh_integer_set_from_an_unknown_binding() {
        let mut locally_bound = HashSet::new();
        locally_bound.insert("value".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("value", refined_domain::abstract_value::unknown());
        let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("isinstance seeded value");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        assert_eq!(value.set, unbounded_integers());
    }

    /// The same seeding applies to `Kind::Unknown` marked `opaque: true`
    /// (`abstract_value::opaque`'s own "determined to be undeterminable"
    /// shape, e.g. an external call's result) — both share `Kind::Unknown`,
    /// so both carry zero information for this test to read.
    #[test]
    fn test_isinstance_int_seeds_a_fresh_integer_set_from_an_opaque_binding() {
        let mut locally_bound = HashSet::new();
        locally_bound.insert("value".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("value", refined_domain::abstract_value::opaque());
        let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("isinstance seeded value");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        assert_eq!(value.set, unbounded_integers());
    }

    /// `isinstance(value, bool)` seeds `Kind::Values` over the two
    /// exact booleans, never a `Kind::Set` — `bool`'s domain is exactly
    /// `{0, 1}`.
    #[test]
    fn test_isinstance_bool_seeds_the_two_boolean_values_from_unbound() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("isinstance(value, bool)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("isinstance seeded value");
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
        let mut values = value.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![0.0, 1.0]);
    }

    /// `isinstance(value, float)` on a `Kind::KindUnion` binding (the
    /// honest JSON-union `json.loads` answers over an opaque string,
    /// `expressions.rs::json_loads_value_space`) keeps ONLY the
    /// Float-tagged arm — the gain the ledger names: a downstream guard
    /// must still narrow the union rather than reading it as
    /// unnarrowable. Built inline here (rather than reaching into
    /// `expressions.rs`'s private constructor) with the same seven arms
    /// that function builds.
    #[test]
    fn test_isinstance_float_narrows_a_json_loads_union_to_its_float_arm() {
        use refined_domain::abstract_value::float_sorted_unknown;
        use refined_domain::abstract_value::null_value;
        use refined_domain::abstract_value::opaque_value;
        use refined_domain::abstract_value::AbstractValue;
        use refined_sets::codepoint_sets::strings;
        use refined_sets::refinement_forms::at_least;

        let integer_arm = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]), None, TrustProved, SetKindTag::None)
        };
        let float_arm = float_sorted_unknown();
        let union = kind_union_of(vec![
            null_value(),
            known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustProved),
            known_set(strings(), None, TrustProved, SetKindTag::None),
            integer_arm,
            float_arm.clone(),
            opaque_value("a list"),
            opaque_value("a dict"),
        ]);
        assert_eq!(union.kind, Kind::KindUnion, "the seven distinct-kind arms must not collapse");

        let mut locally_bound = HashSet::new();
        locally_bound.insert("value".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("value", union);

        let Some(narrowed) = assumed("isinstance(value, float)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("value still bound");
        assert_eq!(value.kind, Kind::Set, "only the float arm should survive, unwrapped from the union");
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
        assert_eq!(value.set, float_arm.set);
    }

    /// `isinstance(value, int)` proving FALSE seeds nothing — a
    /// falsified test says which sort `value` is NOT, never which sort
    /// it IS.
    #[test]
    fn test_isinstance_proving_false_seeds_nothing() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("isinstance(value, int)", environment, false) else {
            return;
        };
        assert!(narrowed.read("value").is_none());
    }

    /// The full `len_in_guard` guard
    /// (`isinstance(value, int) and not isinstance(value, bool) and
    /// 0 <= value <= 120`) run as ONE `assume` call: the isinstance
    /// seed and the chained-comparison narrowing compose end to end,
    /// landing the exact `[0, 120]` integer window `Age` admits.
    #[test]
    fn test_len_in_guard_shape_narrows_to_the_zero_to_120_integer_window() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed(
            "isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 120",
            environment,
            true,
        ) else {
            return;
        };
        let value = narrowed.read("value").expect("value bound by the guard");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&value.set, &age) && (kernel.scalar_subset)(&age, &value.set),
            "value.set = {:?}, want the same set as {:?}",
            value.set,
            age
        );
    }

    /// The `guard_over_ceiling` shape — same guard, but only the
    /// single-sided `value >= 0` bound: the narrowed set still admits
    /// 200 (not a subset of `Age`), matching the fixture's own marked
    /// "the guard does not bound the ceiling" fire.
    #[test]
    fn test_guard_over_ceiling_shape_still_admits_the_ceiling_violation() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed(
            "isinstance(value, int) and not isinstance(value, bool) and value >= 0",
            environment,
            true,
        ) else {
            return;
        };
        let value = narrowed.read("value").expect("value bound by the guard");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            !(kernel.scalar_subset)(&value.set, &age),
            "value.set = {:?} must still admit values above 120 (200, …)",
            value.set
        );
    }

    /// `sample is not None` proving TRUE against a `Kind::PossiblyUndefined`
    /// binding (an `Optional[X]`-declared parameter's own seed,
    /// `check.rs::seed_parameters`) unwraps to the wrapper's own INNER
    /// value — the annotated set, never the wrapper itself.
    #[test]
    fn test_is_not_none_true_unwraps_a_possibly_undefined_binding() {
        use refined_domain::abstract_value::possibly_absent;
        use refined_domain::abstract_value::AbsentFlavor;

        let mut locally_bound = HashSet::new();
        locally_bound.insert("sample".to_owned());
        let mut environment = Environment::new(locally_bound);
        let inner = known_set(
            make_refined_set(vec![at_least(-2.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        environment.bind("sample", possibly_absent(inner.clone(), AbsentFlavor::NullOnly, None, false));

        let Some(narrowed) = assumed("sample is not None", environment, true) else {
            return;
        };
        let value = narrowed.read("sample").expect("sample still bound");
        assert_eq!(value.kind, Kind::Set, "the wrapper must unwrap to its inner Kind::Set, not stay a maybe carrier");
        assert_eq!(value.set, inner.set);
    }

    /// The mirror: `sample is None` proving TRUE rebinds to the exact
    /// `null_value` — the wrapper's absent side, matching what
    /// `assignability::judge` reads directly for a bare `None`.
    #[test]
    fn test_is_none_true_rebinds_a_possibly_undefined_binding_to_null() {
        use refined_domain::abstract_value::possibly_absent;
        use refined_domain::abstract_value::AbsentFlavor;

        let mut locally_bound = HashSet::new();
        locally_bound.insert("sample".to_owned());
        let mut environment = Environment::new(locally_bound);
        let inner = known_set(
            make_refined_set(vec![at_least(-2.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        environment.bind("sample", possibly_absent(inner, AbsentFlavor::NullOnly, None, false));

        let Some(narrowed) = assumed("sample is None", environment, true) else {
            return;
        };
        let value = narrowed.read("sample").expect("sample still bound");
        assert_eq!(value.kind, Kind::Null, "the wrapper must rebind to the exact null_value on the is-None-true fork");
    }

    // ── ASCII case-conjunction alphabet narrowing ─────────────────────

    /// A bare `str` parameter's own seed: `Kind::Set` over the whole
    /// string ground (`codepoint_sets::strings()`), untagged
    /// (`kind_tag: None` — `check.rs::seed_parameters`'s own choice for
    /// a sequence-shaped declared set, `states_sequence` true), matching
    /// what `x: str` actually seeds to.
    fn environment_with_bare_string(name: &str) -> Environment {
        let mut locally_bound = HashSet::new();
        locally_bound.insert(name.to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind(name, known_set(strings(), None, TrustProved, SetKindTag::None));
        environment
    }

    /// F2.fixed's own `str_len_fixed_inside` shape: `len(x) == 2 and
    /// x.isascii() and x.isupper()` narrows `x` to exactly the `Code`
    /// alias's own set — two ASCII upper-case letters.
    #[test]
    fn test_isascii_and_isupper_conjunction_narrows_to_the_ascii_upper_alphabet() {
        let environment = environment_with_bare_string("x");
        let Some(narrowed) = assumed("len(x) == 2 and x.isascii() and x.isupper()", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Set);
        let Some(kernel) = loaded_kernel() else { return };
        let code = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
            make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]),
            2,
            Some(2),
        )]);
        assert!(
            (kernel.scalar_subset)(&x.set, &code) && (kernel.scalar_subset)(&code, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            code
        );
    }

    /// The lower-case twin: `x.isascii() and x.islower()` narrows to
    /// `[0x61, 0x7A]` instead of `[0x41, 0x5A]`.
    #[test]
    fn test_isascii_and_islower_conjunction_narrows_to_the_ascii_lower_alphabet() {
        let environment = environment_with_bare_string("x");
        let Some(narrowed) = assumed("len(x) == 2 and x.isascii() and x.islower()", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let lower = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
            make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]),
            2,
            Some(2),
        )]);
        assert!(
            (kernel.scalar_subset)(&x.set, &lower) && (kernel.scalar_subset)(&lower, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            lower
        );
    }

    /// `x.isupper()` ALONE (no `x.isascii()` in the same conjunction)
    /// narrows nothing — the module doc's own reason: `isupper()` alone
    /// is pinned only against the full Unicode cased-character
    /// categories, which reach far outside ASCII, so bounding it to
    /// `[0x41, 0x5A]` without the `isascii()` co-occurrence would
    /// overclaim.
    #[test]
    fn test_isupper_alone_narrows_nothing() {
        let environment = environment_with_bare_string("x");
        let Some(narrowed) = assumed("x.isupper()", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, strings(), "isupper() alone must not narrow the alphabet");
    }

    /// `x.isascii()` ALONE (no `isupper()`/`islower()` in the same
    /// conjunction) narrows nothing through this leaf — `isascii()`
    /// alone states a `[0x00, 0x7F]` bound, a different (wider) claim
    /// this leaf does not build, matching the "only the conjunction"
    /// scope this leaf's own doc states.
    #[test]
    fn test_isascii_alone_narrows_nothing_through_this_leaf() {
        let environment = environment_with_bare_string("x");
        let Some(narrowed) = assumed("x.isascii()", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, strings(), "isascii() alone narrows nothing through the case-conjunction leaf");
    }

    /// `x.isascii() and y.isupper()` — the two calls on DIFFERENT
    /// receivers — narrows neither: the conjunction must name the SAME
    /// place from both calls.
    #[test]
    fn test_isascii_and_isupper_on_different_names_narrows_neither() {
        let mut locally_bound = HashSet::new();
        locally_bound.insert("x".to_owned());
        locally_bound.insert("y".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("x", known_set(strings(), None, TrustProved, SetKindTag::None));
        environment.bind("y", known_set(strings(), None, TrustProved, SetKindTag::None));
        let Some(narrowed) = assumed("x.isascii() and y.isupper()", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let y = narrowed.read("y").expect("y still bound");
        assert_eq!(x.set, strings(), "x's own alphabet must stay unnarrowed");
        assert_eq!(y.set, strings(), "y's own alphabet must stay unnarrowed");
    }
}
