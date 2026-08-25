/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::lattice_operations::truthiness;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;

use crate::collection_models::dict_with_item;
use crate::collection_models::list_with_item;
use crate::env::Environment;
use crate::expressions::binary_arithmetic_value;
use crate::expressions::evaluate_expression;
use crate::function_table::FunctionTable;
use crate::instances::field_write;

use super::call_result::call_result_with_enclosing;
use super::interpret::collect_bound_names;
use super::seed::bind_parameters;
use super::seed::fresh_body_environment;
use super::seed::seed_free_variables;

/// `call_result_with_enclosing`'s own answer, PLUS every ENCLOSING-SCOPE
/// write the body itself performs — the channel that `call_result_with_
/// enclosing`'s own doc names as out of its scope ("A WRITE to an
/// enclosing name from inside the callee... is not modeled"):
/// a-statements.py's `nonlocal_rebind` (`nonlocal age` then `age = 200`)
/// and `closure_mutates_flattened_capture` (`outlaw["age"] = 200`, a
/// mutation THROUGH a captured free name, no `nonlocal` needed since the
/// write never rebinds `outlaw` itself — CPython's own rule,
/// executionmodel.rst's "Naming and binding": "if a name is bound in a
/// block, it is a local variable of that block" applies to the NAME
/// `outlaw`, never to a subscript/attribute STORE through it, so no
/// `nonlocal` declaration is needed or read for that shape).
///
/// Two kinds of effect, both read against the SAME interpreted run
/// `call_result_with_enclosing` would produce (this function re-runs the
/// body rather than sharing state with that call, since the two answers
/// serve different callers — a value-only call site never needs the
/// effect list, and building it costs one extra interpretation of an
/// already-bounded, already depth-capped body):
///
/// 1. A `nonlocal <name>` declaration anywhere at this body's own
///    TOP LEVEL (`collect_nonlocal_names`, one level of `if`/elif/else
///    nesting included, matching `interpret_if`'s own reach) followed by
///    a plain `name = <expr>` / `name op= <expr>` assignment: the
///    ENCLOSING scope's own `age` is what CPython actually rebinds
///    (executionmodel.rst: "The nonlocal statement causes... names to
///    refer to previously bound variables in the nearest enclosing
///    scope"), so the effect is the assignment's own evaluated value —
///    judged by the CALLER (`check.rs`'s statement-level dispatch)
///    against the enclosing body's OWN declared table exactly as a
///    straight-line `age = 200` would be, which is what makes
///    `nonlocal_rebind`'s own row FIRE: the outer `age` is a declared
///    `Age` slot, and 200 is the effect value judged against it.
/// 2. A STORE THROUGH A FREE NAME: `<free-name>[<key>] = <value>` or
///    `<free-name>.<field> = <value>` where `<free-name>` is neither a
///    parameter nor a name this body's own statements bind (the same
///    `locally_bound` set `fresh_body_environment` builds) — composes
///    the receiver's NEW value via `collection_models::dict_with_item`/
///    `list_with_item` (subscript) or `instances::field_write`
///    (attribute), reading the free name's CURRENT value from
///    `enclosing` first (so two writes to the same captured name inside
///    one call compose, matching real execution order) — a store this
///    function cannot compose (a receiver shape neither helper answers,
///    or a free name `enclosing` never bound) answers that name
///    `unknown()` instead of dropping the effect silently: the caller
///    MUST forget a name this function could not account for, never
///    keep a stale pre-call value.
///
/// Returns `None` under the exact same conditions
/// `call_result_with_enclosing` would decline outright (the depth cap,
/// an unsupported parameter shape, or `interpret_body` declining the
/// body) — an effect list is only ever built alongside a value this
/// call genuinely answers, never as a consolation prize for an otherwise
/// declined call.
pub fn call_effects(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    enclosing: &Environment,
) -> Option<(AbstractValue, Vec<(String, AbstractValue)>)> {
    let value = call_result_with_enclosing(def, arguments, table, kernel, depth, Some(enclosing))?;

    let mut nonlocal_names = std::collections::HashSet::new();
    collect_nonlocal_names(&def.body, &mut nonlocal_names);

    // `collect_bound_names` reads any `name = ...` target as a LOCAL
    // binding — it has no `nonlocal` awareness of its own (a restricted
    // body never had one to read before this channel existed). A name
    // this body declares `nonlocal` is, by CPython's own scoping rule,
    // NEVER local (executionmodel.rst: "the nonlocal statement causes
    // the listed identifiers to refer to previously bound variables in
    // the nearest enclosing scope"), so it is removed here — this is
    // what lets `seed_free_variables` (below) copy its CURRENT value in
    // from `enclosing` for a shape like `nonlocal age; age = age + 1`
    // to read correctly, and what lets `record_write_effect`'s own
    // subscript/attribute arms treat it as a free base name too.
    let mut locally_bound = std::collections::HashSet::new();
    for parameter in def.parameters.posonlyargs.iter().chain(def.parameters.args.iter()) {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    collect_bound_names(&def.body, &mut locally_bound);
    for nonlocal_name in &nonlocal_names {
        locally_bound.remove(nonlocal_name);
    }

    let mut effect_environment = fresh_body_environment(def, table, depth);
    seed_free_variables(def, enclosing, &mut effect_environment);
    if bind_parameters(def, arguments, kernel, &mut effect_environment, Some(enclosing)).is_none() {
        return Some((value, Vec::new()));
    }
    let mut effects: Vec<(String, AbstractValue)> = Vec::new();
    collect_call_effects(&def.body, kernel, &mut effect_environment, &nonlocal_names, &locally_bound, &mut effects);
    Some((value, effects))
}

/// Every name declared `nonlocal` anywhere at `body`'s own top level or
/// one level inside an `if`/elif/else arm — the same reach
/// `interpret_if`/`interpret_undecided_arms` give an ordinary statement,
/// since a `nonlocal` declaration inside an untaken arm still applies to
/// this scope regardless of which arm executes (CPython resolves
/// `nonlocal` at COMPILE time, not at the declaring statement's own
/// runtime position — executionmodel.rst, "the nonlocal statement...
/// applies to the entire scope of a function or class body").
pub(super) fn collect_nonlocal_names(body: &[Stmt], names: &mut std::collections::HashSet<String>) {
    for stmt in body {
        match stmt {
            Stmt::Nonlocal(nonlocal) => {
                for name in &nonlocal.names {
                    names.insert(name.id.as_str().to_owned());
                }
            }
            Stmt::If(if_stmt) => {
                collect_nonlocal_names(&if_stmt.body, names);
                for clause in &if_stmt.elif_else_clauses {
                    collect_nonlocal_names(&clause.body, names);
                }
            }
            _ => {}
        }
    }
}

/// Walks `body`'s own top-level statements (plus one level of `if` arms)
/// evaluating each against `environment` IN PLACE — the same restricted
/// forms `interpret_body` reads, but this walk's OWN job is recording
/// `effects`, not answering a return value, so it never declines: a
/// statement shape it does not specifically recognize is simply skipped
/// (its own value-producing behavior is already accounted for by
/// `call_result_with_enclosing`'s own separate, complete interpretation;
/// this second pass only needs to notice WRITES that escape the callee's
/// own local scope). `declared` name resolution is not this function's
/// job — every effect is reported as a plain value, judged by the
/// CALLER against ITS OWN declared table, exactly as `bind_checked` in
/// `loops.rs` judges a loop body's declared-slot writes.
fn collect_call_effects(
    body: &[Stmt],
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
    nonlocal_names: &std::collections::HashSet<String>,
    locally_bound: &std::collections::HashSet<String>,
    effects: &mut Vec<(String, AbstractValue)>,
) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                let [target] = assign.targets.as_slice() else {
                    continue;
                };
                record_write_effect(target, assign.value.as_ref(), kernel, environment, nonlocal_names, locally_bound, effects);
            }
            Stmt::AugAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    if nonlocal_names.contains(name.id.as_str()) {
                        let current = environment.read(name.id.as_str()).cloned().unwrap_or_else(unknown);
                        let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
                        let updated = binary_arithmetic_value(assign.op, &current, &operand);
                        environment.bind(name.id.as_str(), updated.clone());
                        effects.push((name.id.as_str().to_owned(), updated));
                    }
                }
            }
            Stmt::If(if_stmt) => {
                let test_value = evaluate_expression(if_stmt.test.as_ref(), environment, kernel);
                let (truthy, known) = truthiness(&test_value);
                if known {
                    let body = if truthy {
                        Some(if_stmt.body.as_slice())
                    } else {
                        if_stmt
                            .elif_else_clauses
                            .iter()
                            .find(|clause| clause.test.is_none())
                            .map(|clause| clause.body.as_slice())
                    };
                    if let Some(body) = body {
                        collect_call_effects(body, kernel, environment, nonlocal_names, locally_bound, effects);
                    }
                    continue;
                }
                // an undecidable test: both arms may have run under real
                // execution, so both are scanned for effects (on a shared
                // fork each, never rejoined — this function reports every
                // POSSIBLE effect, and the caller's own judging handles an
                // over-approximated value the same honest way a loop's
                // Undetermined-declines-the-whole-run posture does not
                // need to apply here, since an effect is additive
                // information, not a replacement for the value answer).
                let mut arm_environment = environment.fork();
                collect_call_effects(&if_stmt.body, kernel, &mut arm_environment, nonlocal_names, locally_bound, effects);
                for clause in &if_stmt.elif_else_clauses {
                    let mut clause_environment = environment.fork();
                    collect_call_effects(&clause.body, kernel, &mut clause_environment, nonlocal_names, locally_bound, effects);
                }
            }
            _ => {}
        }
    }
}

/// One `Assign` target's own effect, when it is a shape this channel
/// tracks: a bare `nonlocal` name, or a subscript/attribute store whose
/// BASE is a free name (neither a parameter nor a name this body's own
/// statements bind). Every other target shape (a locally-bound plain
/// name, a tuple/list unpack, a store through a non-Name base) records
/// no effect — that write is either purely local (already answered by
/// `call_result_with_enclosing`'s own value) or outside this channel's
/// read shapes.
pub(super) fn record_write_effect(
    target: &Expr,
    value_expr: &Expr,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
    nonlocal_names: &std::collections::HashSet<String>,
    locally_bound: &std::collections::HashSet<String>,
    effects: &mut Vec<(String, AbstractValue)>,
) {
    match target {
        Expr::Name(name) if nonlocal_names.contains(name.id.as_str()) => {
            let value = evaluate_expression(value_expr, environment, kernel);
            environment.bind(name.id.as_str(), value.clone());
            effects.push((name.id.as_str().to_owned(), value));
        }
        Expr::Subscript(subscript) => {
            let Expr::Name(base) = subscript.value.as_ref() else {
                return;
            };
            if locally_bound.contains(base.id.as_str()) {
                return;
            }
            let value = evaluate_expression(value_expr, environment, kernel);
            let Some(receiver) = environment.read(base.id.as_str()).cloned() else {
                effects.push((base.id.as_str().to_owned(), unknown()));
                return;
            };
            let key = evaluate_expression(subscript.slice.as_ref(), environment, kernel);
            let composed = match receiver.kind {
                Kind::Object => dict_with_item(&receiver, &key, &value),
                Kind::List => list_with_item(&receiver, &key, &value),
                _ => None,
            };
            let new_receiver = composed.unwrap_or_else(unknown);
            environment.bind(base.id.as_str(), new_receiver.clone());
            effects.push((base.id.as_str().to_owned(), new_receiver));
        }
        Expr::Attribute(attribute) => {
            let Expr::Name(base) = attribute.value.as_ref() else {
                return;
            };
            if locally_bound.contains(base.id.as_str()) {
                return;
            }
            let value = evaluate_expression(value_expr, environment, kernel);
            let Some(receiver) = environment.read(base.id.as_str()).cloned() else {
                effects.push((base.id.as_str().to_owned(), unknown()));
                return;
            };
            let new_receiver = field_write(&receiver, attribute.attr.as_str(), value).unwrap_or_else(unknown);
            environment.bind(base.id.as_str(), new_receiver.clone());
            effects.push((base.id.as_str().to_owned(), new_receiver));
        }
        _ => {}
    }
}
