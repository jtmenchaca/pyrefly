/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::trust_grades::trust_level_of;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::loop_questions::InvariantPremise;
use refined_kernel::loop_questions::InvariantPremiseKind;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use refined_kernel::loop_questions::LoopQuestion;
use refined_kernel::loop_questions::LoopVarAnswerKind;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtWhile;
use crate::env::Environment;
use crate::expressions::evaluate_expression;

use super::JudgeContext;
use super::LoopAnswer;
use super::body_once::BodyOutcome;
use super::body_once::run_body_once;
use super::iterable::number_literal_value;

/// A `while` loop is only concretely executed up to this many
/// iterations. Reaching the cap with the condition still true means
/// the bound was not proved (an unbounded or too-large loop) — this
/// function declines rather than guessing where it converges.
const WHILE_ITERATION_CAP: u32 = 1000;

/// `while <name> <op> <literal>: <body> [else: <body>]`, where `<op>`
/// is `<` or `<=` and the loop is a plain counter this function can run
/// out to its own halt. Each iteration re-evaluates the condition
/// against the CURRENT environment (a real interpretation step, not a
/// one-shot bound check) and stops the moment the condition reads
/// false. Reaching `WHILE_ITERATION_CAP` with the condition still
/// provably true is an unproved bound — declines. A counter whose
/// CURRENT value is a known SET rather than one known number
/// (`Kind::Set` — a seeded parameter's declared range) can never
/// resolve a single concrete step at all — `counter_condition_value`
/// reads `None` on the very first check, so this function tries
/// `kernel_bounded_counter_environment` FIRST for exactly that shape,
/// before the concrete stepping loop ever runs. A `break` stops the
/// loop immediately and reports `else_runs: false`; otherwise
/// (`else_runs: true`) once the condition reads false — this function
/// never runs `while_stmt.orelse` itself (`check.rs` walks it, fully
/// judged, when `else_runs`; `kernel_bounded_counter_environment`'s own
/// shape requires an empty `else`, so it always reports `else_runs:
/// true` trivially, and never runs a body that could return either). A
/// `return` stops the loop immediately, same as `break`, and reports
/// `returned: Some((value, range))`.
///
/// A condition that reads UNKNOWN after at least one iteration ran (the
/// counter's own `Kind::Values` widened to `Kind::Set` — the refused-
/// write law's own rebind, `bind_checked`'s doc: a body write judged
/// `Fire` against the counter's `declared` entry keeps the DECLARED set
/// afterward) is a genuinely reached, honest terminal state, not an
/// unrecognized shape: every statement up to and including the one that
/// widened the counter is a real, already-judged fact (`loop_body_over_
/// ceiling`, a-statements.py:494 — the single-statement body's own `age
/// = age + 121` fires against `Age`'s ceiling on iteration 1, and the
/// refused-write rebind then makes the counter's OWN condition test
/// unreadable on iteration 2's check). Reporting `Some` here (rather
/// than `None`) is what lets `check.rs`'s `walk_loop` adopt the judged
/// environment and stop recording its OWN "a while statement is not yet
/// walked" blocker on TOP of the fire this module already proved —
/// `check.rs`'s RTS7002 channel is for a shape this module never even
/// started running, not for a run that reached a real, judged stopping
/// point. `else_runs: false` here (never proven to reach exhaustion, so
/// the safe answer matches `break`'s own posture) — this is distinct
/// from the CAP case below, which never ran any further body statement
/// past the point the bound stopped being provable and stays `None`:
/// a full iteration-budget's worth of `Some(true)` reads is the
/// unbounded-loop shape this module must keep refusing to guess at.
pub(super) fn while_loop_final_environment(
    while_stmt: &StmtWhile,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    if let Some(kernel_result) = kernel_bounded_counter_environment(while_stmt, environment, kernel) {
        return Some(LoopAnswer { environment: kernel_result, else_runs: true, returned: None, widened_names: Vec::new() });
    }
    let mut current = environment.fork();
    let mut ran_an_iteration = false;
    for _ in 0..WHILE_ITERATION_CAP {
        match counter_condition_value(while_stmt.test.as_ref(), &current, kernel) {
            Some(true) => {
                match run_body_once(&while_stmt.body, &mut current, kernel, judge_context)? {
                    BodyOutcome::Fell | BodyOutcome::Continued => {}
                    BodyOutcome::Broke => {
                        return Some(LoopAnswer {
                            environment: current,
                            else_runs: false,
                            returned: None,
                            widened_names: Vec::new(),
                        });
                    }
                    BodyOutcome::Returned(value, range) => {
                        return Some(LoopAnswer {
                            environment: current,
                            else_runs: false,
                            returned: Some((value, range)),
                            widened_names: Vec::new(),
                        });
                    }
                }
                ran_an_iteration = true;
            }
            Some(false) => {
                return Some(LoopAnswer { environment: current, else_runs: true, returned: None, widened_names: Vec::new() });
            }
            // an UNREADABLE condition after at least one judged iteration
            // is the counter's own honest widening (see this function's
            // doc); an unreadable condition on the very FIRST check is a
            // shape this module never recognized at all and must decline,
            // same as before.
            None if ran_an_iteration => {
                return Some(LoopAnswer {
                    environment: current,
                    else_runs: false,
                    returned: None,
                    widened_names: Vec::new(),
                });
            }
            None => return None,
        }
    }
    // the cap was reached with the condition still true — the bound was
    // never proved
    None
}

/// `while <name> <op> <literal>:` where `<name>`'s CURRENT value is a
/// known SET (`Kind::Set` — a seeded parameter's declared range, or any
/// other set-valued binding) rather than one known number — the shape
/// `counter_condition_value` always reads `None` for, since
/// `single_known_number` requires `Kind::Values`. The concrete stepping
/// loop above cannot run this at all (there is no single value to step),
/// so the kernel's own `solve_loop` is asked instead: it iterates the
/// body's own arithmetic transfer, widens, and certifies a candidate set
/// that holds after every iterate — a proof, not a guess.
///
/// Scoped to exactly the shape `lower_counter_step_body` recognizes: a
/// SINGLE tracked name, a body that only ever adds/subtracts a known
/// literal to/from that same name (`n += 1`, `n = n + 1`, `n = n - 1`),
/// and an EMPTY `else` clause (a non-empty else after a kernel-certified,
/// not concretely-run, loop is not this pass's shape — the concrete path
/// above already covers every else-clause row the corpus states). `None`
/// for anything wider: a second written name, an operator this file does
/// not trust to agree with the kernel's own transfer, or a kernel answer
/// that is not `Kind::Set` (`Unknown` is an honest refusal, not a guess
/// to build a set from).
///
/// The bound `environment.bind`s the counter to is the kernel's
/// CERTIFIED INVARIANT — what holds at every body ENTRY, which is sound
/// but not the tightest possible claim (the true post-loop state also
/// carries the negated condition — `narrowing.rs`'s own doc states this
/// file's narrowing channel acts on `Kind::Values` only, no `Kind::Set`
/// machinery exists yet — so this function does not intersect the
/// invariant with the exit narrowing the way the loop's LAST entry
/// technically would let it. Never wrong, just not maximally tight.
pub(super) fn kernel_bounded_counter_environment(
    while_stmt: &StmtWhile,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Environment> {
    if !while_stmt.orelse.is_empty() {
        return None;
    }
    let Expr::Compare(compare) = while_stmt.test.as_ref() else {
        return None;
    };
    if compare.ops.len() != 1 || compare.comparators.len() != 1 {
        return None;
    }
    if !matches!(compare.ops[0], CmpOp::Lt | CmpOp::LtE) {
        return None;
    }
    let Expr::Name(counter) = compare.left.as_ref() else {
        return None;
    };
    let bound_value = number_literal_value(&compare.comparators[0])?;
    // the body runs only while the test held — the kernel's own
    // narrowing set for what the CONDITION admits at every body entry,
    // same shape counter_condition_value's Lt/LtE reads concretely
    let condition_set = make_refined_set(vec![match compare.ops[0] {
        CmpOp::Lt => below(bound_value),
        CmpOp::LtE => at_most(bound_value),
        _ => unreachable!("guarded to Lt | LtE above"),
    }]);
    let counter_name = counter.id.as_str();
    let current = environment.read(counter_name)?;
    if current.kind != Kind::Set {
        return None;
    }
    let entry_set = set_of_known(current)?;
    let entry_grade = trust_level_of(current);
    let step = lower_counter_step_body(&while_stmt.body, counter_name)?;

    let question = LoopQuestion {
        entry: vec![Some(InvariantPremise {
            kind: InvariantPremiseKind::Set,
            values: Vec::new(),
            set: entry_set,
        })],
        cond: vec![Some(condition_set)],
        body: vec![step],
        cond_cmp: None,
    };
    let answers = (kernel.solve_loop)(&question);
    let [answer] = answers.as_slice() else {
        return None;
    };
    if answer.kind != LoopVarAnswerKind::Set {
        return None;
    }
    let mut result = environment.fork();
    result.bind(counter_name, known_set(answer.set.clone(), None, entry_grade, SetKindTag::None));
    Some(result)
}

/// The body's step, lowered into the kernel's per-binding `LoopEffect`
/// grammar rather than run concretely — `set_functions/loop_solve.lean`
/// iterates this itself. Recognizes exactly `name += literal`,
/// `name -= literal`, `name = name + literal`, and `name = name -
/// literal`, one statement, `name` being `counter_name` — the only step
/// shape this pass trusts to mean the same thing under the kernel's
/// `LoopOpAdd`/`LoopOpSub` transfer as it does under CPython's own `+`/`-`
/// (both sort-agnostic — no Python/JS divergence the way `/`, `//`, `%`,
/// and `**` carry). Anything else (a second statement, a different
/// operator, a non-literal operand, a body touching another name) is
/// `None`: this function never approximates a step it cannot state
/// exactly.
fn lower_counter_step_body(body: &[Stmt], counter_name: &str) -> Option<LoopEffect> {
    let [stmt] = body else {
        return None;
    };
    let (op, operand_expr) = match stmt {
        Stmt::AugAssign(assign) => {
            let Expr::Name(target) = assign.target.as_ref() else {
                return None;
            };
            if target.id.as_str() != counter_name {
                return None;
            }
            let op = match assign.op {
                Operator::Add => LoopEffectOp::Add,
                Operator::Sub => LoopEffectOp::Sub,
                _ => return None,
            };
            (op, assign.value.as_ref())
        }
        Stmt::Assign(assign) => {
            let [Expr::Name(target)] = assign.targets.as_slice() else {
                return None;
            };
            if target.id.as_str() != counter_name {
                return None;
            }
            let Expr::BinOp(binop) = assign.value.as_ref() else {
                return None;
            };
            let Expr::Name(left) = binop.left.as_ref() else {
                return None;
            };
            if left.id.as_str() != counter_name {
                return None;
            }
            let op = match binop.op {
                Operator::Add => LoopEffectOp::Add,
                Operator::Sub => LoopEffectOp::Sub,
                _ => return None,
            };
            (op, binop.right.as_ref())
        }
        _ => return None,
    };
    let step_value = number_literal_value(operand_expr)?;
    let counter_leaf = LoopEffect { kind: LoopEffectKind::Var, index: 0, ..Default::default() };
    let step_leaf = LoopEffect {
        kind: LoopEffectKind::Const,
        set: make_refined_set(vec![one_of(&[step_value])]),
        ..Default::default()
    };
    Some(LoopEffect {
        kind: LoopEffectKind::Binary,
        op,
        a: Some(Box::new(counter_leaf)),
        b: Some(Box::new(step_leaf)),
        ..Default::default()
    })
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
