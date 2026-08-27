//! Bool-op / VALUES dispatcher helpers: `narrow`, `narrow_bool_op`,
//! set-kind name asks, and `collect_names`.

use std::sync::Arc;

use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::BoolOp;
use ruff_python_ast::Expr;
use ruff_python_ast::UnaryOp;

use crate::env::Environment;

use super::compare::narrow_compare;
use super::compare::narrow_name_against_dict_literal;
use super::condition_tree::affine_shifted_name_of;
use super::condition_tree::condition_tree_of;
use super::condition_tree::meet_set_answer;
use super::condition_tree::says_anything;
use super::isinstance_guards::narrow_isinstance_call;
use super::isinstance_guards::narrow_type_guard_call;
use super::isinstance_guards::recognizes_type_guard_call;
use super::name_of;
use super::none_truthiness::is_none_test_name;
use super::none_truthiness::narrow_name_truthiness;
use super::predicates::narrow_all_generator_call;
use super::predicates::narrow_ascii_case_conjunction;
use super::predicates::narrow_regex_module_call;

/// The SET channel's own entry point: for every name `condition`
/// mentions (`collect_names`) that is CURRENTLY bound `Kind::Set`,
/// lower the whole condition relative to that name and ask the kernel
/// once. Run AFTER the VALUES channel's leaf walk (`narrow`, `assume`'s
/// own doc explains why) so a name `narrow`'s own `narrow_isinstance_
/// call` just seeded is already `Kind::Set` by the time this runs — one
/// ask per name, by the WHOLE condition at once (`0 <= value <= 120`
/// intersects both bounds in one ask, matching an `and`'s conjunction
/// reading), rather than once per leaf.
pub(super) fn narrow_set_kind_names(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
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
pub(super) fn collect_names(condition: &Expr, out: &mut Vec<String>) {
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
pub(super) fn narrow(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            narrow(&unary.operand, environment, kernel, !truth);
        }
        Expr::BoolOp(bool_op) => narrow_bool_op(bool_op, environment, kernel, truth),
        Expr::Compare(compare) => {
            narrow_compare(compare, environment, truth);
            narrow_name_against_dict_literal(compare, environment, kernel, truth);
        }
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
            narrow_regex_module_call(call, environment, kernel, truth);
            narrow_all_generator_call(call, environment, kernel, truth);
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
pub(super) fn narrow_bool_op(
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
