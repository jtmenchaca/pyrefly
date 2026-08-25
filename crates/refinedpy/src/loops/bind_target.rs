/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use crate::env::Environment;

/// The loop target's own bare-name spelling, if the target is one — a
/// tuple target's own sub-names are read the same way `bind_for_target`
/// binds them, but every one of these three passes only ever binds a
/// SINGLE element abstraction to the whole target, so a tuple target
/// here is out of scope for the same reason it already declines
/// `bind_for_target` widely elsewhere: this helper only needs the bare
/// case to build the exclusion set `stabilized_join` compares against.
pub(super) fn target_names(target: &Expr, names: &mut std::collections::HashSet<String>) {
    match target {
        Expr::Name(name) => {
            names.insert(name.id.to_string());
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                target_names(element, names);
            }
        }
        _ => {}
    }
}

/// Every bare name a loop body's own statements write to, collected
/// SYNTACTICALLY (never by reading bindings back) — `Assign`/`AnnAssign`
/// targets, `AugAssign` targets, a subscript-store's/mutating-method-
/// call's own receiver name (`run_subscript_assign_once`/`run_expr_
/// statement_once`'s own rebind), recursed into every `if`/`elif`/`else`
/// arm the same way `run_body_once`/`run_if_once` walk them. The set is
/// a superset of what any ONE concrete pass actually writes (an untaken
/// `if` arm's names are included too), which is the safe direction for
/// `stabilized_join`'s own use: a name this walk never actually wrote on
/// either pass reads identically from both (nothing rebinds it), so
/// including it in the comparison costs nothing — it is never found
/// unstable, just checked and confirmed stable.
pub(super) fn written_names(body: &[Stmt], names: &mut std::collections::HashSet<String>) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    match target {
                        Expr::Name(name) => {
                            names.insert(name.id.to_string());
                        }
                        Expr::Subscript(subscript) => {
                            if let Expr::Name(name) = subscript.value.as_ref() {
                                names.insert(name.id.to_string());
                            }
                        }
                        _ => {}
                    }
                }
            }
            Stmt::AnnAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    names.insert(name.id.to_string());
                }
            }
            Stmt::AugAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    names.insert(name.id.to_string());
                }
            }
            Stmt::If(if_stmt) => {
                written_names(&if_stmt.body, names);
                for clause in &if_stmt.elif_else_clauses {
                    written_names(&clause.body, names);
                }
            }
            Stmt::Expr(expr_stmt) => {
                if let Expr::Call(call) = expr_stmt.value.as_ref()
                    && let Expr::Attribute(attribute) = call.func.as_ref()
                {
                    // both the bare mutating-call shape and the chained
                    // `setdefault(...).append(...)` shape rebind the
                    // OUTERMOST receiver name — `run_expr_statement_once`/
                    // `run_setdefault_append_once`'s own `environment.bind`
                    // call — found by descending through `.value` past any
                    // number of chained attribute/call layers to the
                    // innermost bare Name.
                    let mut receiver = attribute.value.as_ref();
                    loop {
                        match receiver {
                            Expr::Name(name) => {
                                names.insert(name.id.to_string());
                                break;
                            }
                            Expr::Call(inner_call) => match inner_call.func.as_ref() {
                                Expr::Attribute(inner_attribute) => receiver = inner_attribute.value.as_ref(),
                                _ => break,
                            },
                            _ => break,
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Binds a `for` target to one iterate: a bare name binds directly; a
/// tuple target (`for k, v in d.items():`) unpacks an EXACT-arity
/// `Kind::List` element positionally — CPython raises `ValueError` on
/// an arity mismatch (simple_stmts.rst, "Assignment statements":
/// unpacking "requires the same number of items"), which this domain
/// has no exception channel for this wave, so a mismatch is `false`
/// (decline) rather than a partial bind. Any other target shape
/// (starred, attribute, subscript) is `false`.
pub(super) fn bind_for_target(target: &Expr, element: &AbstractValue, environment: &mut Environment) -> bool {
    match target {
        Expr::Name(name) => {
            environment.bind(name.id.as_str(), element.clone());
            true
        }
        Expr::Tuple(tuple) => {
            if element.kind != Kind::List || element.items.len() != tuple.elts.len() {
                return false;
            }
            for (sub_target, sub_value) in tuple.elts.iter().zip(element.items.iter()) {
                if !bind_for_target(sub_target, sub_value, environment) {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

/// Forgets a `del` target's name, restricted to a bare name or a
/// tuple/list of bare names — `false` for anything wider (a starred
/// target, an attribute/subscript target), which declines the whole
/// loop rather than silently skip an un-forgettable target.
pub(super) fn forget_bare_name_target(target: &Expr, environment: &mut Environment) -> bool {
    match target {
        Expr::Name(name) => {
            environment.forget(name.id.as_str());
            true
        }
        Expr::Tuple(tuple) => tuple.elts.iter().all(|element| forget_bare_name_target(element, environment)),
        Expr::List(list) => list.elts.iter().all(|element| forget_bare_name_target(element, environment)),
        _ => false,
    }
}
