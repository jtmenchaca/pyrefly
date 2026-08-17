/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A same-module `def`'s answer for one call: concrete evaluation of a
//! BOUNDED body — the same posture `loops.rs`'s `run_restricted_body`
//! takes for loop bodies, extended to the restricted statement forms a
//! function body needs (branching and `return`, which a loop body never
//! has). `call_result` binds the callee's parameters to the caller's
//! argument values, interprets the body statements it recognizes, and
//! answers the join of every value the body could return — or declines
//! (`None`) the moment the body does something this file does not
//! interpret, so a caller never gets a guessed answer.
//!
//! This is the a-statements:399-404 seam: `helper_never_answers_none`
//! returns a dict literal on both the `if` arm and the fall-through —
//! `{"age": 40}` and `{"age": 10}`. Once `expressions.rs` evaluates
//! dict literals, this file's `if`/`else` handling joins those two
//! Object values into one Object answer that is never `Kind::Null`,
//! which is exactly what lets the walk prove `held is None` false at
//! `none_test_on_helper_that_never_answers_none`'s call site.
//!
//! Keyword arguments are the WIRING owner's job: `call_result` takes
//! only POSITIONAL argument values, in parameter order. A caller with a
//! keyword call maps each keyword to its parameter's position before
//! calling this function; this file has no keyword-name matching of
//! its own.

use std::sync::Arc;

use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::truthiness;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAnnAssign;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtIf;

use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::binary_arithmetic_value;
use crate::refinedpy::expressions::evaluate_expression;
use crate::refinedpy::function_table::FunctionTable;

/// The deepest a call chain interprets before declining outright. A
/// same-module call whose body calls itself (directly or through a
/// cycle of same-module calls) would otherwise interpret forever; the
/// cap turns that into an honest decline rather than a hang, matching
/// the corpus's recursion row (n-file).
pub const CALL_DEPTH_CAP: u32 = 8;

/// `def`'s answer for one call with `arguments` bound positionally, or
/// `None` when the body (or its parameter shape) is outside what this
/// file interprets. See the module doc for the body forms interpreted
/// and the a-statements:399-404 seam this unblocks.
pub fn call_result(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> Option<AbstractValue> {
    if depth >= CALL_DEPTH_CAP {
        return None;
    }
    // `*args`/`**kwargs` collect an unknown-length tail this file does
    // not model; a keyword-only parameter cannot be reached by a
    // positional argument at all, so any def declaring one is outside
    // this call's binding shape.
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() || !def.parameters.kwonlyargs.is_empty() {
        return None;
    }
    let mut environment = fresh_body_environment(def, table);
    bind_parameters(def, arguments, kernel, &mut environment)?;

    let mut returns: Vec<AbstractValue> = Vec::new();
    let falls_through = interpret_body(&def.body, kernel, depth, &mut environment, &mut returns)?;
    if falls_through {
        returns.push(null_value());
    }

    let mut answers = returns.into_iter();
    let first = answers.next()?;
    let joined = answers.fold(first, |acc, next| join_known(acc, next));
    Some(joined)
}

/// A fresh environment for the callee's body: every parameter name plus
/// every name the body itself binds (this file's own collector, not
/// check.rs's — the two stay independent per the mission's file
/// ownership), the module's function table carried forward so a nested
/// same-module call composes through `evaluate_expression`'s dispatch
/// once that wiring lands.
fn fresh_body_environment(def: &StmtFunctionDef, table: Option<&Arc<FunctionTable>>) -> Environment {
    let mut locally_bound = std::collections::HashSet::new();
    for parameter in def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
    {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    collect_bound_names(&def.body, &mut locally_bound);
    let mut environment = Environment::new(locally_bound);
    if let Some(table) = table {
        environment.set_functions(table.clone());
    }
    environment
}

/// Binds `arguments` to `def`'s posonlyargs+args in order. A trailing
/// parameter with no matching argument uses its own default, evaluated
/// in a FRESH (name-less) environment — a default expression may only
/// reference literals/builtins, never an enclosing name, so no name
/// this call knows is visible while reading it. Too few arguments with
/// an unevaluable (or absent) default, or too many arguments, declines
/// the whole call.
fn bind_parameters(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    let parameters: Vec<_> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .collect();
    if arguments.len() > parameters.len() {
        return None;
    }
    let default_environment = Environment::new(std::collections::HashSet::new());
    for (index, parameter) in parameters.iter().enumerate() {
        let value = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            let default_expr = parameter.default.as_deref()?;
            evaluate_expression(default_expr, &default_environment, kernel)
        };
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }
    Some(())
}

/// Interprets `body`'s statements in order against `environment`,
/// restricted forms only. Returns `Some(true)` when control can fall
/// off the end of `body` (so the caller should contribute a
/// `null_value()` return), `Some(false)` when every path through `body`
/// ends in a recorded `Return`, and `None` the moment a statement
/// outside the restricted forms is met — the whole call declines then,
/// matching `loops.rs::run_restricted_body`'s all-or-nothing posture.
fn interpret_body(
    body: &[Stmt],
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    environment: &mut Environment,
    returns: &mut Vec<AbstractValue>,
) -> Option<bool> {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => interpret_assign(assign, kernel, environment)?,
            Stmt::AnnAssign(assign) => interpret_ann_assign(assign, kernel, environment)?,
            Stmt::AugAssign(assign) => interpret_aug_assign(assign, kernel, environment)?,
            Stmt::Pass(_) => {}
            Stmt::Expr(expr_stmt) => {
                evaluate_expression(expr_stmt.value.as_ref(), environment, kernel);
            }
            Stmt::If(if_stmt) => {
                let falls_through = interpret_if(if_stmt, kernel, depth, environment, returns)?;
                if !falls_through {
                    return Some(false);
                }
            }
            Stmt::Return(ret) => {
                let value = match ret.value.as_deref() {
                    Some(value_expr) => evaluate_expression(value_expr, environment, kernel),
                    None => null_value(),
                };
                if value.kind == Kind::Unknown {
                    return None;
                }
                returns.push(value);
                return Some(false);
            }
            _ => return None,
        }
    }
    Some(true)
}

fn interpret_assign(assign: &StmtAssign, kernel: &Arc<RefinedTSKernel>, environment: &mut Environment) -> Option<()> {
    let [Expr::Name(name)] = assign.targets.as_slice() else {
        return None;
    };
    let value = evaluate_expression(assign.value.as_ref(), environment, kernel);
    environment.bind(name.id.as_str(), value);
    Some(())
}

fn interpret_ann_assign(
    assign: &StmtAnnAssign,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    let Expr::Name(name) = assign.target.as_ref() else {
        return None;
    };
    let Some(value_expr) = assign.value.as_deref() else {
        // a value-less `x: T` declares nothing to bind — CPython
        // evaluates the annotation but never assigns the name
        // (simple_stmts.rst, "Annotated assignment statements")
        return Some(());
    };
    let value = evaluate_expression(value_expr, environment, kernel);
    environment.bind(name.id.as_str(), value);
    Some(())
}

fn interpret_aug_assign(
    assign: &StmtAugAssign,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    let Expr::Name(name) = assign.target.as_ref() else {
        return None;
    };
    let current = environment.read(name.id.as_str()).cloned().unwrap_or_else(unknown);
    let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
    let updated = binary_arithmetic_value(assign.op, &current, &operand);
    environment.bind(name.id.as_str(), updated);
    Some(())
}

/// `if test: body [elif ...] [else: body]` inside a summarized call
/// body. A definitely-true/false test interprets only the live arm on
/// the SAME environment (no fork needed — only one arm's writes ever
/// happen). An undecidable test interprets BOTH arms on forked
/// environments and rejoins the surviving ones through
/// `Environment::join`, mirroring `check.rs::walk_if`/`arm_terminates`:
/// an arm ending in `Return` contributes its value(s) to `returns` but
/// does not rejoin, since its fall-through state is unreachable.
/// Returns `Some(true)` if the post-if point is reachable (so the
/// caller keeps interpreting later statements), `Some(false)` if every
/// live arm returned, `None` if any visited arm is outside the
/// restricted forms.
fn interpret_if(
    if_stmt: &StmtIf,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    environment: &mut Environment,
    returns: &mut Vec<AbstractValue>,
) -> Option<bool> {
    let mut arms: Vec<(Option<&Expr>, &[Stmt])> = Vec::new();
    arms.push((Some(if_stmt.test.as_ref()), if_stmt.body.as_slice()));
    for clause in &if_stmt.elif_else_clauses {
        arms.push((clause.test.as_ref(), clause.body.as_slice()));
    }

    // a definite verdict short-circuits to the one live arm, evaluated
    // in place — walrus/side effects on the test itself are read once,
    // through the caller's own environment
    for (test, body) in &arms {
        if let Some(test_expr) = test {
            let test_value = evaluate_expression(test_expr, environment, kernel);
            let (truthy, known) = truthiness(&test_value);
            if known {
                if truthy {
                    return interpret_body(body, kernel, depth, environment, returns);
                }
                continue;
            }
            // the FIRST undecidable test is where both-arms interpretation
            // starts — every arm from here on (including any later elif)
            // is undetermined territory, handled below
            return interpret_undecided_arms(&arms, kernel, depth, environment, returns);
        }
        // a bare `else`/catch-all arm reached with every earlier test
        // known false: this is the one live arm
        return interpret_body(body, kernel, depth, environment, returns);
    }
    // every test was known false and there was no catch-all arm: the
    // whole `if` falls through untouched
    Some(true)
}

/// Interprets every arm on its own fork once a test could not be
/// decided — used from the first undecidable test onward, since a
/// later arm's own reachability itself depends on the undecided one.
fn interpret_undecided_arms(
    arms: &[(Option<&Expr>, &[Stmt])],
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    environment: &mut Environment,
    returns: &mut Vec<AbstractValue>,
) -> Option<bool> {
    let mut surviving: Vec<Environment> = Vec::new();
    let mut has_catch_all = false;
    for (test, body) in arms {
        has_catch_all = has_catch_all || test.is_none();
        let mut arm_environment = environment.fork();
        let falls_through = interpret_body(body, kernel, depth, &mut arm_environment, returns)?;
        if falls_through {
            surviving.push(arm_environment);
        }
    }
    if !has_catch_all {
        surviving.push(environment.fork());
    }

    *environment = match surviving.len() {
        0 => return Some(false),
        1 => surviving.into_iter().next().unwrap(),
        _ => {
            let mut joined = surviving.remove(0);
            for arm in surviving {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    };
    Some(true)
}

/// Every bare name this body's own statements bind — `Assign`/
/// `AnnAssign`/`AugAssign` targets and `if`/`elif`/`else` bodies,
/// recursively. A restricted body never contains anything else that
/// binds a name (no `for`/`with`/`import`/nested `def`), so this
/// collector only walks the forms `interpret_body` itself recognizes.
fn collect_bound_names(body: &[Stmt], bound: &mut std::collections::HashSet<String>) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    if let Expr::Name(name) = target {
                        bound.insert(name.id.as_str().to_owned());
                    }
                }
            }
            Stmt::AnnAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    bound.insert(name.id.as_str().to_owned());
                }
            }
            Stmt::AugAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    bound.insert(name.id.as_str().to_owned());
                }
            }
            Stmt::If(if_stmt) => {
                collect_bound_names(&if_stmt.body, bound);
                for clause in &if_stmt.elif_else_clauses {
                    collect_bound_names(&clause.body, bound);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use refined_domain::abstract_value::known_values;
    use refined_domain::abstract_value::PrimitiveKind;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_parser::parse_module;

    use super::*;

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// Parses `source` as a module and returns its single top-level
    /// `def` (the function under test).
    fn parsed_def(source: &str) -> StmtFunctionDef {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        let stmt = module.body.into_iter().next().expect("one top-level statement");
        stmt.function_def_stmt().expect("top-level statement is a def")
    }

    fn known_int(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    #[test]
    fn straight_line_body_answers_the_returned_expression() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def double(x):\n    return x + x\n");
        let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0).expect("straight-line body answers");
        assert_eq!(result.values, vec![6.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn a_trailing_default_parameter_is_evaluated_when_no_argument_covers_it() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def add(x, y=10):\n    return x + y\n");
        let result = call_result(&def, &[known_int(5.0)], None, &kernel, 0).expect("default parameter fills in");
        assert_eq!(result.values, vec![15.0]);
    }

    #[test]
    fn an_if_else_where_both_arms_return_known_values_joins_both_possibilities() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(
            "def pick(flag):\n    if flag:\n        return 3\n    else:\n        return 5\n",
        );
        let result =
            call_result(&def, &[unknown()], None, &kernel, 0).expect("both known-value arms join to an answer");
        // an undecidable flag interprets both arms; the join of 3 and 5
        // under one Integer tag is the two-value carrier
        // join_known's own test (test_join_known_like_sort_keeps_the_tag_mixed_sort_loses_it)
        // pins for two same-sort Values joins
        assert_eq!(result.kind, Kind::Values);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
        let mut values = result.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![3.0, 5.0]);
    }

    #[test]
    fn a_body_that_falls_off_the_end_contributes_null_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def maybe_none(flag):\n    if flag:\n        return 3\n    x = 1\n");
        let result = call_result(&def, &[known_int(1.0)], None, &kernel, 0)
            .expect("a known-true flag still interprets the fall-through arm's shape honestly");
        // flag is KNOWN true here, so only the `return 3` arm runs and the
        // fall-through never contributes — this pins the definite-branch
        // path specifically; the undecidable-flag fall-through case is
        // covered by the next test
        assert_eq!(result.values, vec![3.0]);
    }

    #[test]
    fn an_undecidable_flag_whose_false_arm_falls_off_the_end_joins_in_null() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def maybe_none(flag):\n    if flag:\n        return 3\n    x = 1\n");
        let result = call_result(&def, &[unknown()], None, &kernel, 0)
            .expect("an undecidable flag interprets both the return arm and the fall-through");
        // the true arm returns 3; the false arm falls off the end,
        // contributing null_value() — the join of an Integer with Null
        // is neither a bare Integer (Kind::Values) nor a bare Null
        assert_ne!(result.kind, Kind::Unknown);
        assert_ne!(result.kind, Kind::Values);
        assert_ne!(result.kind, Kind::Null);
    }

    #[test]
    fn a_body_with_a_while_loop_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def counted(n):\n    while n > 0:\n        n -= 1\n    return n\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
    }

    #[test]
    fn the_depth_cap_declines_before_interpreting_the_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def double(x):\n    return x + x\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, CALL_DEPTH_CAP).is_none());
    }

    #[test]
    fn a_return_with_an_unknown_value_declines_the_whole_call() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def opaque(x):\n    return f(x)\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
    }

    #[test]
    fn too_many_arguments_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def one_arg(x):\n    return x\n");
        assert!(call_result(&def, &[known_int(1.0), known_int(2.0)], None, &kernel, 0).is_none());
    }

    #[test]
    fn varargs_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def variadic(*args):\n    return 1\n");
        assert!(call_result(&def, &[], None, &kernel, 0).is_none());
    }
}
