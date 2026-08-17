/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Concrete execution of the corpus's bounded loop shapes: `for x in
//! [lit, ...]:`/`for x in range(...):` over known numeric iterables,
//! and `while name < literal:`-style counters with a provable
//! iteration bound. Every iterate in these shapes is known, so running
//! the loop body once per iterate is sound, not an approximation — the
//! walk still owns whether to call this or record its own blocker
//! (`Some` result replaces the blocker; `None` means the walk keeps
//! it). Only a restricted body qualifies: plain `Assign`/`AugAssign`
//! on plain names, arithmetic the expressions module can evaluate, no
//! break/continue, no nested statements. A loop the kernel would have
//! to bound (an unbounded `while`, a non-literal iterable) is this
//! module's `None` — the kernel's `SolveLoop` path (refined_kernel
//! loop_questions) supersedes this module for those shapes; that wire
//! is not built this wave.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Number;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFor;
use ruff_python_ast::StmtWhile;
use ruff_python_ast::UnaryOp;

use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::binary_arithmetic_value;
use crate::refinedpy::expressions::evaluate_expression;

/// A `while` loop is only concretely executed up to this many
/// iterations. Reaching the cap with the condition still true means
/// the bound was not proved (an unbounded or too-large loop) — this
/// function declines rather than guessing where it converges.
const WHILE_ITERATION_CAP: u32 = 1000;

/// The post-loop environment for a `for`/`while` statement matching
/// one of this module's concretely-executable shapes; `None` for
/// anything else (any other statement kind, an unrecognized iterable,
/// a body outside the restricted forms, or a `while` that does not
/// resolve within the iteration cap). The walk keeps its own blocker
/// on `None`.
pub fn loop_final_environment(
    stmt: &Stmt,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Environment> {
    match stmt {
        Stmt::For(for_stmt) => for_loop_final_environment(for_stmt, environment, kernel),
        Stmt::While(while_stmt) => while_loop_final_environment(while_stmt, environment, kernel),
        _ => None,
    }
}

/// `for target in <literal list/tuple, or range(...)>: <restricted body>
/// [else: <restricted body>]` — every element is a known number, so the
/// body runs once per element over a forked environment. Python leaves
/// the target bound to the last element after the loop ends (never
/// reset or deleted, compound_stmts.html "the for statement"); an empty
/// iterable runs the body zero times, so the target keeps whatever the
/// pre-loop environment already held for that name. The `else` clause
/// always runs for these shapes (a restricted body can never `break`,
/// so the iterator always exhausts) and sees the post-loop bindings.
fn for_loop_final_environment(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Environment> {
    if for_stmt.is_async {
        return None;
    }
    let Expr::Name(target) = for_stmt.target.as_ref() else {
        return None;
    };
    let elements = literal_iterable_values(for_stmt.iter.as_ref())?;
    if !body_is_restricted(&for_stmt.body) {
        return None;
    }
    if !body_is_restricted(&for_stmt.orelse) {
        return None;
    }
    let mut current = environment.fork();
    for element in elements {
        current.bind(target.id.as_str(), known_number(element));
        if !run_restricted_body(&for_stmt.body, &mut current, kernel) {
            return None;
        }
    }
    if !run_restricted_body(&for_stmt.orelse, &mut current, kernel) {
        return None;
    }
    Some(current)
}

/// `while <name> <op> <literal>: <restricted body> [else: <restricted
/// body>]`, where `<op>` is `<` or `<=` and the loop is a plain counter
/// this function can run out to its own halt. Each iteration
/// re-evaluates the condition against the CURRENT environment (a real
/// interpretation step, not a one-shot bound check) and stops the
/// moment the condition reads false or unknown. Reaching
/// `WHILE_ITERATION_CAP` with the condition still provably true is an
/// unproved bound — declines rather than guessing convergence. The
/// `else` clause runs once the condition is false (compound_stmts.html
/// "the while statement": exhaustion runs `else`; a restricted body
/// can never `break`, so that is the only way this loop ends).
fn while_loop_final_environment(
    while_stmt: &StmtWhile,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Environment> {
    if !body_is_restricted(&while_stmt.body) {
        return None;
    }
    if !body_is_restricted(&while_stmt.orelse) {
        return None;
    }
    let mut current = environment.fork();
    for _ in 0..WHILE_ITERATION_CAP {
        match counter_condition_value(while_stmt.test.as_ref(), &current, kernel) {
            Some(true) => {
                if !run_restricted_body(&while_stmt.body, &mut current, kernel) {
                    return None;
                }
            }
            Some(false) => {
                if !run_restricted_body(&while_stmt.orelse, &mut current, kernel) {
                    return None;
                }
                return Some(current);
            }
            None => return None,
        }
    }
    // the cap was reached with the condition still true (or unreadable
    // on the final check) — the bound was never proved
    None
}

/// The condition's truth value for a `name < literal` / `name <=
/// literal` counter test, or `None` when the shape or the operand
/// values are not this function's provable counter form. Any other
/// comparison shape (an `and`/`or`, `==`, a non-Name left side, a
/// non-literal right side) is `None` — this function only runs
/// counters it can prove terminate, never approximates one that might.
fn counter_condition_value(
    test: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<bool> {
    let Expr::Compare(compare) = test else {
        return None;
    };
    if compare.ops.len() != 1 || compare.comparators.len() != 1 {
        return None;
    }
    let op = compare.ops[0];
    if !matches!(op, CmpOp::Lt | CmpOp::LtE) {
        return None;
    }
    let left = evaluate_expression(compare.left.as_ref(), environment, kernel);
    let right = evaluate_expression(&compare.comparators[0], environment, kernel);
    let left_value = single_known_number(&left)?;
    let right_value = single_known_number(&right)?;
    Some(match op {
        CmpOp::Lt => left_value < right_value,
        CmpOp::LtE => left_value <= right_value,
        _ => unreachable!("guarded to Lt | LtE above"),
    })
}

/// The one number a known, single-valued numeric/boolean AbstractValue
/// carries, or `None` for anything unknown/multi-valued/non-numeric —
/// the same reading `single_numeric_value` in expressions.rs does, but
/// that helper is private to its module, so this module reads the
/// public `Kind`/`values`/`kind_tag` fields directly.
fn single_known_number(value: &AbstractValue) -> Option<f64> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Number)
        | Some(PrimitiveKind::Integer)
        | Some(PrimitiveKind::Float)
        | Some(PrimitiveKind::Boolean) => Some(value.values[0]),
        _ => None,
    }
}

fn known_number(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Number, TrustProved)
}

/// The known numeric elements a `for` loop's iterable expression names,
/// in iteration order — a literal list/tuple of number literals, or a
/// `range(...)` call whose arguments are all literal ints (`range(stop)`
/// / `range(start, stop)` / `range(start, stop, step)`,
/// library/stdtypes.html#range). Anything else (a name, a call other
/// than `range`, a non-literal element) is `None`: this function only
/// answers when every iterate is known without running any code.
fn literal_iterable_values(iterable: &Expr) -> Option<Vec<f64>> {
    match iterable {
        Expr::List(list) => elements_as_number_literals(&list.elts),
        Expr::Tuple(tuple) => elements_as_number_literals(&tuple.elts),
        Expr::Call(call) => range_call_values(call),
        _ => None,
    }
}

fn elements_as_number_literals(elements: &[Expr]) -> Option<Vec<f64>> {
    let mut values = Vec::with_capacity(elements.len());
    for element in elements {
        values.push(number_literal_value(element)?);
    }
    Some(values)
}

/// A `range(...)` call's produced values, or `None` when the callee
/// is not the bare name `range`, an argument is not a literal int, or
/// the argument count is not 1/2/3. `step == 0` is `None` — CPython
/// raises `ValueError` there rather than producing a sequence.
fn range_call_values(call: &ExprCall) -> Option<Vec<f64>> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "range" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let args = &call.arguments.args;
    let (start, stop, step) = match args.len() {
        1 => (0.0, int_literal_value(&args[0])?, 1.0),
        2 => (int_literal_value(&args[0])?, int_literal_value(&args[1])?, 1.0),
        3 => (
            int_literal_value(&args[0])?,
            int_literal_value(&args[1])?,
            int_literal_value(&args[2])?,
        ),
        _ => return None,
    };
    if step == 0.0 {
        return None;
    }
    let mut values = Vec::new();
    let mut current = start;
    // r[i] = start + step*i, while r[i] < stop (step > 0) or r[i] > stop
    // (step < 0) — library/stdtypes.html#range
    if step > 0.0 {
        while current < stop {
            values.push(current);
            current += step;
        }
    } else {
        while current > stop {
            values.push(current);
            current += step;
        }
    }
    Some(values)
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression).
fn number_literal_value(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            Number::Float(value) => Some(*value),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) => {
            let operand = number_literal_value(unary.operand.as_ref())?;
            match unary.op {
                UnaryOp::USub => Some(-operand),
                UnaryOp::UAdd => Some(operand),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A `range()` argument's value, restricted to an INT literal (`range`
/// rejects a float argument at call time — this function will not
/// treat `range(3.0, 5)` as known, staying honest about that CPython
/// restriction rather than silently truncating).
fn int_literal_value(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            _ => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = int_literal_value(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
}

/// Whether every statement in a loop body is one this module can run
/// concretely: a plain `Assign`/`AugAssign` whose target is a bare
/// name, or `Pass`. No `break`/`continue` (this module does not model
/// early exit — a body containing either declines whole, since a
/// restricted body's `for`/`while` else-clause reasoning above assumes
/// the loop always exhausts) and no nested compound statement (an
/// `if`/nested `for`/`while`/etc. is a body this module does not
/// interpret).
fn body_is_restricted(body: &[Stmt]) -> bool {
    body.iter().all(|stmt| {
        matches!(
            stmt,
            Stmt::Assign(assign) if matches!(assign.targets.as_slice(), [Expr::Name(_)])
        ) || matches!(
            stmt,
            Stmt::AugAssign(assign) if matches!(assign.target.as_ref(), Expr::Name(_))
        ) || matches!(stmt, Stmt::Pass(_))
    })
}

/// Runs a restricted body's statements against `environment` in order,
/// exactly as `body_is_restricted` verified it: plain-name
/// `Assign`/`AugAssign`/`Pass` only. `AugAssign` reads the target's
/// CURRENT value from `environment` and folds it with the RHS through
/// `binary_arithmetic_value` — the same arithmetic transfer the
/// expressions module exposes for ordinary binary operators, so a
/// `total += age` row and a `total = total + age` row agree exactly.
/// Returns false the moment any computed value is not a known exact
/// state — a loop is Some only when EVERY iterate is determinable, so
/// an unreadable right side (a call, an unbound name) fails the whole
/// loop rather than silently binding unknown.
fn run_restricted_body(
    body: &[Stmt],
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> bool {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                let value = evaluate_expression(assign.value.as_ref(), environment, kernel);
                if value.kind != Kind::Values {
                    return false;
                }
                if let Expr::Name(name) = &assign.targets[0] {
                    environment.bind(name.id.as_str(), value);
                }
            }
            Stmt::AugAssign(assign) => {
                let Expr::Name(name) = assign.target.as_ref() else {
                    continue;
                };
                let current = match environment.read(name.id.as_str()) {
                    Some(value) => value.clone(),
                    None => unknown(),
                };
                let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
                let updated = binary_arithmetic_value(assign.op, &current, &operand);
                if updated.kind != Kind::Values {
                    return false;
                }
                environment.bind(name.id.as_str(), updated);
            }
            Stmt::Pass(_) => {}
            _ => unreachable!("body_is_restricted only admits Assign/AugAssign/Pass"),
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

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

    /// Parses `source` as a module body and returns its single
    /// top-level statement (the loop under test).
    fn parsed_loop(source: &str) -> Stmt {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        module.body.into_iter().next().expect("one top-level statement")
    }

    fn environment_with(bindings: &[(&str, f64)]) -> Environment {
        let locally_bound: HashSet<String> = bindings.iter().map(|(name, _)| name.to_string()).collect();
        let mut environment = Environment::new(locally_bound);
        for (name, value) in bindings {
            environment.bind(name, known_number(*value));
        }
        environment
    }

    #[test]
    fn for_over_literal_list_sums_and_keeps_last_target_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [60, 61]:\n    total += age\n");
        let environment = environment_with(&[("total", 0.0), ("age", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("shape is concrete");
        assert_eq!(result.read("total").unwrap().values, vec![121.0]);
        // the target stays bound to the LAST element after the loop —
        // never reset or deleted (compound_stmts.html "the for statement")
        assert_eq!(result.read("age").unwrap().values, vec![61.0]);
    }

    #[test]
    fn for_over_range_three_sums_zero_one_two() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for i in range(3):\n    total += i\n");
        let environment = environment_with(&[("total", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("range(3) is concrete");
        assert_eq!(result.read("total").unwrap().values, vec![3.0]);
        assert_eq!(result.read("i").unwrap().values, vec![2.0]);
    }

    #[test]
    fn while_counter_loop_runs_to_its_own_halt() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("while n < 5:\n    n += 1\n    total += n\n");
        let environment = environment_with(&[("n", 0.0), ("total", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("bounded counter");
        // n: 0->1->2->3->4->5, loop stops once n == 5; total sums 1+2+3+4+5
        assert_eq!(result.read("n").unwrap().values, vec![5.0]);
        assert_eq!(result.read("total").unwrap().values, vec![15.0]);
    }

    #[test]
    fn body_with_a_call_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    total = f(x)\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn break_in_body_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    total += x\n    break\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn continue_in_body_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    continue\n");
        let environment = environment_with(&[("x", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn for_else_applies_its_body_after_exhaustion() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    total += x\nelse:\n    done = 1\n");
        let environment = environment_with(&[("total", 0.0), ("done", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("restricted else runs");
        assert_eq!(result.read("total").unwrap().values, vec![3.0]);
        assert_eq!(result.read("done").unwrap().values, vec![1.0]);
    }

    #[test]
    fn while_that_never_resolves_within_the_cap_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // n never changes, so the condition holds forever — must not
        // guess convergence; must decline once the cap is hit
        let stmt = parsed_loop("while n < 5:\n    total += 1\n");
        let environment = environment_with(&[("n", 0.0), ("total", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn empty_literal_list_leaves_target_unbound_when_it_was_never_bound() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in []:\n    total += x\n");
        let environment = environment_with(&[("total", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("empty literal list is concrete");
        // x was never assigned by the loop (compound_stmts.html): it
        // carries forward whatever the pre-loop environment held, which
        // here is nothing
        assert!(result.read("x").is_none());
        assert_eq!(result.read("total").unwrap().values, vec![0.0]);
    }

    #[test]
    fn nested_if_in_body_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    if x:\n        total += x\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn non_loop_statement_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("total = 1\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn known_number_helper_carries_proved_number_values() {
        let value = known_number(3.0);
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Number));
        // TrustProved renders as no grade at all — see known_values
        assert_eq!(value.grade, None);
    }
}
