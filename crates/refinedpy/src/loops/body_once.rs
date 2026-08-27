/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtIf;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use crate::assignability::judge;
use crate::assignability::Verdict;
use crate::collection_models;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::narrowing::assume;
use crate::typereading::DeclaredRefinement;

use super::JudgeContext;
use super::bind_target::forget_bare_name_target;

/// What running one loop body ONCE (top level or nested inside an `if`
/// arm) says about the rest of the CURRENT iteration: `Fell` — ran every
/// statement, keep going; `Broke` — a `break` fired, the signal
/// `for_loop_final_environment`/`while_loop_final_environment` use to
/// skip the `else` clause and, for `for`, stop advancing the target
/// past the element the `break` fired on (compound_stmts.rst, "the
/// `for` statement"/"the `while` statement": "the `else` clause...
/// executes when the loop terminates through exhaustion... rather than
/// by `break`"); `Continued` — a `continue` fired, which must skip every
/// statement still left in EVERY enclosing body for this iteration (not
/// just the innermost `if` arm's own body) and land back at the
/// iteration boundary. `Continued` is a DISTINCT case from `Fell`
/// precisely so a `continue` inside a nested `if` arm does not get
/// mistaken, once folded back into the enclosing body's own outcome,
/// for an ordinary fall-through that should let the enclosing body's
/// LATER statements still run. `Returned(value, range)` — a
/// `Stmt::Return` fired (RETURN-THROUGH-LOOP CHANNEL): propagates
/// straight out through every enclosing body/if-arm/loop the same way
/// `Broke` does, ending the WHOLE loop (real CPython: a `return` exits
/// the function outright, so no later statement in this body, this
/// iteration, or any further iteration ever runs).
#[derive(Debug)]
pub(super) enum BodyOutcome {
    Fell,
    Broke,
    Continued,
    Returned(Option<AbstractValue>, TextRange),
}

/// Runs one loop body's statements against `environment` IN PLACE, in
/// order, honoring real control flow: `break` stops immediately
/// (`BodyOutcome::Broke`, propagated straight out — CPython never runs
/// statements after a `break` in the same body); `continue` stops THIS
/// body's statement loop early and reports `BodyOutcome::Continued` — a
/// distinct outcome from `Fell` precisely because this same function
/// also runs a NESTED `if`-arm's body (via `run_if_once`/
/// `outcome_of_body`): when the `continue` fired inside an if-arm, the
/// enclosing body still has statements left after the `if`, and those
/// must NOT run. Reporting `Continued` up through
/// `StatementOutcome::Continue` (see `outcome_of_body`) lets the
/// enclosing body's own statement loop, right here, also stop early
/// rather than mistake the if-statement's `Next` for an ordinary
/// fall-through. `None` is the same "this loop is not this module's
/// shape" honesty every other decline here uses — no statement here
/// EVER writes a value that might be wrong; an unrecognized shape
/// declines the WHOLE loop rather than skip or approximate.
pub(super) fn run_body_once(
    body: &[Stmt],
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<BodyOutcome> {
    for stmt in body {
        match run_statement_once(stmt, environment, kernel, judge_context)? {
            StatementOutcome::Next => {}
            StatementOutcome::Continue => return Some(BodyOutcome::Continued),
            StatementOutcome::Break => return Some(BodyOutcome::Broke),
            StatementOutcome::Returned(value, range) => return Some(BodyOutcome::Returned(value, range)),
        }
    }
    Some(BodyOutcome::Fell)
}

/// What one statement, run once against the current environment, says
/// about the rest of THIS iteration: keep going (`Next`), stop this
/// iteration early (`Continue`), stop the whole loop (`Break`), or stop
/// the whole loop AND carry a returned value out
/// (`Returned(value, range)` — RETURN-THROUGH-LOOP CHANNEL).
pub(super) enum StatementOutcome {
    Next,
    Continue,
    Break,
    Returned(Option<AbstractValue>, TextRange),
}

/// Runs exactly one loop-body statement, dispatched by syntactic form.
/// `None` for any statement shape this module does not interpret — the
/// caller (`run_body_once`) propagates that straight into a whole-loop
/// decline.
pub(super) fn run_statement_once(
    stmt: &Stmt,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<StatementOutcome> {
    match stmt {
        Stmt::Pass(_) => Some(StatementOutcome::Next),
        Stmt::Break(_) => Some(StatementOutcome::Break),
        Stmt::Continue(_) => Some(StatementOutcome::Continue),
        Stmt::Assign(assign) => {
            let [target] = assign.targets.as_slice() else {
                return None;
            };
            if let Expr::Subscript(subscript) = target {
                run_subscript_assign_once(subscript, assign.value.as_ref(), environment, kernel)?;
                return Some(StatementOutcome::Next);
            }
            if matches!(target, Expr::Tuple(_) | Expr::List(_)) {
                run_unpack_assign_once(target, assign.value.as_ref(), environment, kernel)?;
                return Some(StatementOutcome::Next);
            }
            run_assign_once(target, assign.value.as_ref(), stmt.range(), environment, kernel, judge_context)?;
            Some(StatementOutcome::Next)
        }
        Stmt::AnnAssign(assign) => {
            // A declared-slot target INSIDE the loop body (`bad: Age =
            // over_value` where `bad` is never bound before this
            // statement) carries no entry in `judge_context.declared` —
            // that table is `check.rs`'s own `aug_assign_refinements`
            // snapshot from BEFORE this loop started (`loop_final_
            // environment`'s own doc), and this module has no access to
            // `WalkContext`'s alias table to read a fresh annotation the
            // way `check.rs`'s own `walk_ann_assign` does. Reusing an
            // ALREADY-RECORDED entry's own `DeclaredRefinement` by ALIAS
            // SPELLING (rather than re-reading the annotation) is sound
            // without that table: a module-level type alias (`type Age =
            // …`) names exactly one set, so any two `declared` entries
            // that read the same bare-Name annotation carry an identical
            // `set`/`admits_none` — matching `declared`'s own existing
            // entry for a DIFFERENT name is the same fact, not a guess.
            // Scoped to a bare `Expr::Name` annotation only (never a
            // subscript/union/string form this module cannot parse
            // without the alias table); `None` from this lookup leaves
            // the target OUTSIDE `declared`, unjudged, same as before.
            if let Expr::Name(target_name) = assign.target.as_ref()
                && let Expr::Name(annotation_name) = assign.annotation.as_ref()
                && !judge_context.declared.contains_key(target_name.id.as_str())
            {
                let matched: Option<DeclaredRefinement> = judge_context
                    .declared
                    .values()
                    .find(|declared| declared.spelling == annotation_name.id.as_str())
                    .cloned();
                if let Some(matched) = matched {
                    judge_context.newly_declared.insert(target_name.id.as_str().to_owned(), matched);
                }
            }
            let Some(value_expr) = assign.value.as_deref() else {
                // `x: T` alone declares no value — nothing to bind or
                // judge, matching simple_stmts.rst's "the `=` clause is
                // optional" reading check.rs's own walk_ann_assign uses.
                return Some(StatementOutcome::Next);
            };
            run_assign_once(assign.target.as_ref(), value_expr, stmt.range(), environment, kernel, judge_context)?;
            Some(StatementOutcome::Next)
        }
        Stmt::AugAssign(assign) => {
            let Expr::Name(name) = assign.target.as_ref() else {
                return None;
            };
            let current = match environment.read(name.id.as_str()) {
                Some(value) => value.clone(),
                None => unknown(),
            };
            let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
            // an accumulator (`total += x`) folding a Set-shaped operand —
            // a for-loop element bound off an ABSTRACT pass
            // (`repetition_window_element_pass`, `windowed_range_element_
            // pass`), never one concrete number — has no answer through
            // the plain, kernel-less arithmetic path: `single_numeric_
            // value` needs one known scalar on both sides. `binary_
            // arithmetic_value_with_kernel` asks `transfer_over_sets`
            // first for exactly that shape (at least one operand
            // `Kind::Set`), falling through to the identical plain path
            // for the two-known-values case this function already served
            // — one arithmetic transfer, not two independently maintained
            // copies.
            let updated = crate::expressions::binary_arithmetic_value_with_kernel(assign.op, &current, &operand, kernel);
            if !matches!(updated.kind, Kind::Values | Kind::Set) {
                return None;
            }
            bind_checked(name.id.as_str(), updated, stmt.range(), environment, kernel, judge_context)?;
            Some(StatementOutcome::Next)
        }
        Stmt::If(if_stmt) => run_if_once(if_stmt, environment, kernel, judge_context),
        Stmt::Expr(expr_stmt) => run_expr_statement_once(expr_stmt.value.as_ref(), environment, kernel),
        // RETURN-THROUGH-LOOP CHANNEL: `return [expr]` inside a loop body
        // ends the whole loop right here (real CPython — a return exits
        // the function, so no later statement in this iteration or any
        // further iteration ever runs). A BARE `return` (no expression)
        // carries `None` — matching `check.rs`'s own `walk_return`
        // convention that a bare return "carries no value expression and
        // judges nothing either," so this channel must not invent a
        // Null value for check.rs to judge where the straight-line walk
        // never would; `return <expr>` evaluates the expression against
        // the CURRENT environment (the same plain read `check.rs`'s own
        // `sink_value` falls back to) and carries `Some(value)`. The
        // carried `TextRange` is the value expression's own range when
        // one exists, else the whole `return` statement's own range.
        Stmt::Return(ret) => {
            let (value, range) = match ret.value.as_deref() {
                Some(value_expr) => (Some(evaluate_expression(value_expr, environment, kernel)), value_expr.range()),
                None => (None, stmt.range()),
            };
            Some(StatementOutcome::Returned(value, range))
        }
        // `del a, b, ...` (simple_stmts.rst, "The `del` statement":
        // "Deletion of a target list recursively deletes each target,
        // from left to right") — every named target simply forgets
        // what this run knew; no judgment, so no cross-family check
        // applies (there is nothing left to compare against after a
        // forget). Matches check.rs's own `Stmt::Delete` handling for
        // the ordinary (non-loop) walk.
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                if !forget_bare_name_target(target, environment) {
                    return None;
                }
            }
            Some(StatementOutcome::Next)
        }
        _ => None,
    }
}

/// `name = value` / `name: T = value` on a plain-name target: evaluates
/// the RHS and binds it (through `bind_checked`'s own judging), `None`
/// unless the value comes back fully known (`Kind::Values`, `Kind::List`,
/// `Kind::Object`, `Kind::Null`, or `Kind::Set` — an unreadable right
/// side, a call, or an unbound name fails the whole loop rather than
/// silently binding unknown, and so does a write `bind_checked` judges
/// `Undetermined`). A non-name
/// target (attribute, subscript-outside-the-mutation-contract) is
/// `None`: this function only ever writes a name it can name.
/// `stmt_range` is the ENCLOSING statement's own range — the dedupe key
/// and fire anchor `bind_checked` uses, so `x = y` and `x: Age = y` both
/// fire (if they fire) at their own statement, never at a sub-expression.
pub(super) fn run_assign_once(
    target: &Expr,
    value_expr: &Expr,
    stmt_range: TextRange,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<()> {
    let Expr::Name(name) = target else {
        return None;
    };
    let value = evaluate_expression(value_expr, environment, kernel);
    // Kind::Null (Python's None) is a fully-known value — accepted
    // alongside Values/List/Object so a declared-slot write of None
    // (a-statements.py:541's own row: an iterate that evaluates to
    // None) reaches bind_checked's own judging rather than declining
    // the whole loop for a kind this guard used to treat as unknown.
    // Kind::Set is likewise a fully-known value — a for-loop iterate
    // bound off a display's own tuple/list element (elements_as_values'
    // own widened acceptance) can be a whole-number/whole-string SET
    // rather than one scalar (`for item in (unread_number(),):` — the
    // element is `-> int`'s own claimed whole-number set, not a single
    // value); `age = item` inside the loop body re-reads that same Set
    // value and must reach `bind_checked`'s own `assignability::judge`
    // CONTAINMENT law rather than decline the whole loop for a kind this
    // guard used to treat as unknown.
    if !matches!(value.kind, Kind::Values | Kind::List | Kind::Object | Kind::Null | Kind::Set) {
        return None;
    }
    bind_checked(name.id.as_str(), value, stmt_range, environment, kernel, judge_context)
}

/// `(a, b, ...) = value` / `[a, b, ...] = value` inside a loop body —
/// A8.edge.process's own `k, v = line.split("=", 1)`. simple_stmts.rst,
/// "Assignment statements", states the rule for a target list that is
/// not a single target: "The object must be an iterable with the same
/// number of items as there are targets in the target list, and the
/// items are assigned, from left to right, to the corresponding
/// targets."
///
/// Two right-side shapes carry that reading here:
///
/// - an EXACT `Kind::List` (`"a=1".split("=", 1)` over a known string —
///   `string_models::method_result`'s own `split` rows answer the exact
///   two-element list): the arity is known, so a count that matches
///   binds positionally and a count that does not is CPython's own
///   `ValueError`, which this domain has no exception channel for, so a
///   mismatch declines the whole loop rather than bind partially — the
///   same posture `bind_for_target` keeps for a `for k, v in ...`
///   target.
/// - a REPETITION WINDOW (`line.split("=", 1)` over an unread `line` —
///   `string_models::sort_only`'s own `split` row answers
///   `repetition(strings(), 1, None)`, an unbounded list of unread
///   strings): the window states no exact item count, but every
///   position of a repetition draws from the SAME element set
///   (`repetition_window_forms::as_repetition`, the reading
///   `star_element_read` already gives one indexed position), so on
///   every run whose arity DOES match the target list — the only runs
///   that do not raise — each target's item is somewhere in that one
///   element set. Binding every target to the element is therefore the
///   claim the window supports. A run whose arity does not match raises
///   `ValueError`, the same unmodeled raise the exact arm declines on;
///   this domain states nothing about it either way, exactly as
///   `bind_for_target`'s own tuple arm does not.
///
/// `None` for any other right-side value (nothing is known about the
/// items to bind), a nested tuple/list sub-target, or a starred target
/// (a different clause of the same paragraph, with its own item-count
/// rule — out of this function's scope). Every sub-target must be a
/// bare `Expr::Name`: this function only ever writes a name it can name,
/// the same restriction `run_assign_once` keeps.
///
/// Unlike `run_assign_once`, no sub-target is judged against
/// `judge_context.declared`: an unpack writes several names at once and
/// the corpus's own shapes (`k, v = ...`) name plain locals, so there is
/// no declared slot to judge. A declared name reached this way binds
/// unjudged, matching `run_subscript_assign_once`'s own reasoning that
/// the read side catches what the write side does not judge.
pub(super) fn run_unpack_assign_once(
    target: &Expr,
    value_expr: &Expr,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<()> {
    let targets: &[Expr] = match target {
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::List(list) => &list.elts,
        _ => return None,
    };
    let mut names: Vec<&str> = Vec::with_capacity(targets.len());
    for sub_target in targets {
        let Expr::Name(name) = sub_target else {
            return None;
        };
        names.push(name.id.as_str());
    }
    let value = evaluate_expression(value_expr, environment, kernel);
    if value.kind == Kind::List {
        if value.items.len() != names.len() {
            return None;
        }
        for (name, item) in names.iter().zip(value.items.iter()) {
            environment.bind(name, item.clone());
        }
        return Some(());
    }
    if value.kind == Kind::Set && value.set_kind_tag == SetKindTag::None {
        let repeated = as_repetition(&value.set)?;
        let element = AbstractValue {
            kind_tag: value.kind_tag,
            ..known_set(repeated.element, None, trust_level_of(&value), SetKindTag::None)
        };
        for name in &names {
            environment.bind(name, element.clone());
        }
        return Some(());
    }
    None
}

/// `name[k] = v` — the MUTATION CONTRACT's subscript-target shape.
/// `name` must be a bare name already bound to a known receiver;
/// `collection_models::dict_with_item`/`list_with_item` (dispatched by
/// the receiver's own `Kind`) answer the new receiver value, which
/// rebinds `name` directly — a subscript-store receiver is a
/// container (dict/list), never itself a scalar declared slot, so this
/// write is not a `declared`-table judging candidate the way a bare-name
/// Assign/AugAssign is; `bind_checked` is not called here; sound because
/// a container name reaching a scalar declared sink is caught at the
/// READ side (a later `x[i]` flowing into a declared sink), same as the
/// ordinary (non-loop) walk. `None` for anything the contract does not
/// resolve (an unknown receiver, a key/value shape the contract
/// declines, a receiver `Kind` neither function owns).
pub(super) fn run_subscript_assign_once(
    subscript: &ExprSubscript,
    value_expr: &Expr,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<()> {
    let Expr::Name(name) = subscript.value.as_ref() else {
        return None;
    };
    let receiver = environment.read(name.id.as_str())?.clone();
    let key = evaluate_expression(subscript.slice.as_ref(), environment, kernel);
    let value = evaluate_expression(value_expr, environment, kernel);
    let new_receiver = match receiver.kind {
        // `Kind::ObjectStar` is the UNBOUNDED-KEY dict — a `dict[K, X]`
        // parameter's own seed, and what a keyed dict becomes once this
        // same statement writes it at a key no spelling names
        // (`dict_write.rs::dict_widened_at_unread_key`). A loop's SECOND
        // judged pass (`stabilized_join`) reads back exactly that widened
        // receiver, so it must route to the same `dict_with_item` the
        // first pass took — that function's own star arm records the
        // written key or absorbs an unread one, and declining here left
        // every dict-accumulation loop over an unread key unwalked.
        Kind::Object | Kind::ObjectStar => collection_models::dict_with_item(&receiver, &key, &value)?,
        Kind::List => collection_models::list_with_item(&receiver, &key, &value)?,
        _ => return None,
    };
    environment.bind(name.id.as_str(), new_receiver);
    Some(())
}

/// Binds `name` to `value`, judging first when `name` carries a
/// recorded declaration in `judge_context.declared` (this body's own
/// `x: Age = …` table, threaded in from `check.rs`'s
/// `aug_assign_refinements`) — the REPLACEMENT for the old cross-family
/// decline guard: rather than declining the whole loop the moment a
/// write's sort family disagrees with the slot's prior value, this
/// function now judges the write through `assignability::judge` exactly
/// as `check.rs`'s own `judge_and_bind` does for a straight-line write.
///
/// `Verdict::Fire`: pushed to `judge_context.fires` ONCE PER SYNTACTIC
/// `stmt_range` (`judge_context.already_fired`'s dedupe — a loop that
/// iterates the same statement many times must not repeat the same
/// fire once per iteration), and the slot binds the DECLARED set
/// afterward (the refused-write law: the write is refused, so the slot
/// keeps its declaration, matching `judge_and_bind`'s own convention —
/// a later read in a further iteration or after the loop is silent
/// against the declaration, not a second fire for the same refusal).
/// `Verdict::Silent`: binds the evaluated value, unchanged from before.
/// `Verdict::Undetermined`: declines the WHOLE loop (`None`) — this
/// module cannot record a body-local blocker mid-run; `check.rs`'s own
/// outer blocker for the whole loop statement is the honest stand-in.
///
/// A name with NO recorded declaration (in EITHER `declared`, the
/// pre-loop snapshot, or `newly_declared`, this loop's own body-local
/// alias-reuse table — see `Stmt::AnnAssign`'s own doc) binds directly,
/// unjudged — every plain (undeclared) local this module already
/// tracked, unchanged.
pub(super) fn bind_checked(
    name: &str,
    value: AbstractValue,
    stmt_range: TextRange,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<()> {
    let Some(declared) = judge_context.declared.get(name).or_else(|| judge_context.newly_declared.get(name)) else {
        environment.bind(name, value);
        return Some(());
    };
    let declared = declared.clone();
    match judge(&value, &declared, kernel) {
        Verdict::Fire(message) => {
            if judge_context.already_fired.insert(stmt_range) {
                judge_context.fires.push((stmt_range, message));
            }
            let refused_slot = known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None);
            environment.bind(name, refused_slot);
            Some(())
        }
        Verdict::Silent => {
            environment.bind(name, value);
            Some(())
        }
        Verdict::Undetermined(_) => None,
    }
}

/// `if test: body [elif test: body ...] [else: body]` inside a loop —
/// the taken arm is decided PER ITERATION by evaluating `test` against
/// the CURRENT environment (`lattice_operations::truthiness`'s
/// `(value, known)` pair). Most of this module's callers step ONE
/// concrete element (a display's own literal, a dict's own key) — a
/// test over that element's own scalar value always reads a known
/// `(taken, true)`/`(false, true)` pair, so the single-branch execution
/// below (matching CPython's own `if` semantics, compound_stmts.rst)
/// covers the whole concrete-iterate story exactly.
///
/// The loop's own ABSTRACT passes (`repetition_window_element_pass`,
/// `windowed_range_element_pass`, `abstract_element_sort_pass`,
/// `custom_iterator_element_pass` — every one whose own doc names
/// itself "one JUDGED pass standing in for the whole run", never a
/// concrete per-element walk) bind the loop target to a Set-shaped
/// abstraction rather than one concrete value, so a test over that
/// target (`0 <= x <= 149`) never resolves to one known boolean —
/// `evaluate_expression`'s comparison reader has no single scalar to
/// compare. `run_if_once_over_unknown_test` is this case's own
/// fallback: EXACTLY the same "narrow each arm, walk it, join the
/// survivors" contract `check.rs::walk_if` already uses for a
/// module-level `if` whose test is not proved either way — sound here
/// for the identical reason it is sound there, since an abstract
/// pass's own fires already carry that pass's "some argument reaches
/// here" caveat, never the concrete path's stronger "this really
/// happened" one. Scoped to a plain `if: ... else: ...` (or a bare
/// `if: ...` with no `elif`/`else`) whose every taken arm falls
/// through (`BodyOutcome::Fell`) — an unknown test on any WIDER shape
/// (an `elif` chain, a `break`/`continue`/`return` inside either arm)
/// still declines the whole loop, the same honesty this function
/// always kept: this module never approximates a step it cannot state
/// exactly.
pub(super) fn run_if_once(
    if_stmt: &StmtIf,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<StatementOutcome> {
    let condition = evaluate_expression(if_stmt.test.as_ref(), environment, kernel);
    let (taken, known) = truthiness(&condition);
    if !known {
        return run_if_once_over_unknown_test(if_stmt, environment, kernel, judge_context);
    }
    if taken {
        return run_body_once(&if_stmt.body, environment, kernel, judge_context).map(outcome_of_body);
    }
    for clause in &if_stmt.elif_else_clauses {
        match clause.test.as_ref() {
            None => {
                // a bare `else:` — always taken once every prior
                // `elif`/`if` test read false
                return run_body_once(&clause.body, environment, kernel, judge_context).map(outcome_of_body);
            }
            Some(test) => {
                let clause_condition = evaluate_expression(test, environment, kernel);
                let (clause_taken, clause_known) = truthiness(&clause_condition);
                if !clause_known {
                    return None;
                }
                if clause_taken {
                    return run_body_once(&clause.body, environment, kernel, judge_context).map(outcome_of_body);
                }
            }
        }
    }
    // no arm's test held and there was no bare `else:` — the whole `if`
    // statement is a no-op this iteration
    Some(StatementOutcome::Next)
}

/// `run_if_once`'s own fallback for a test whose truth value this
/// abstract pass cannot read off the CURRENT (Set-shaped) binding —
/// mirrors `check.rs::walk_if`'s own narrow-each-arm-then-join
/// contract, restricted to the one shape this module's abstract passes
/// actually need: a bare `if: body` or `if: body else: body`, no
/// `elif` clause. `narrowing::assume` tightens each arm's own fork by
/// what the test being true (respectively false) says — the SAME
/// narrowing `walk_if` runs before walking a module-level arm whose
/// test is not itself proved — and each fork's body then runs through
/// the ordinary concrete `run_body_once`.
///
/// Both arms must report `BodyOutcome::Fell` — a `break`/`continue`/
/// `return` reachable on only ONE of the two hypothetical arms has no
/// single per-iteration outcome this function can state (the real
/// iterate takes exactly one arm, and this function does not know
/// which), so that shape still declines the WHOLE loop, `None`,
/// exactly as an unrecognized statement anywhere else in this module
/// does. Two `Fell` arms join through `Environment::join` (the same
/// per-name lattice join `walk_if`'s own `surviving` fold uses), and
/// the joined environment becomes this statement's own outcome.
///
/// An absent `else` arm folds through the SAME machine as a bare
/// `else: pass`: the untaken-when-false path is the test's own
/// `assume(..., false)` narrowing of the CURRENT environment, run
/// through no statements at all — matching CPython's own "no `else`
/// clause" semantics (compound_stmts.rst, "the `if` statement") without
/// a second `run_body_once([])` call.
///
/// Gated on the test naming AT LEAST ONE currently `Kind::Set`-bound
/// name (`test_mentions_a_set_bound_name`) — the one signal that
/// distinguishes "this test is unknown because it reads an ABSTRACT
/// per-pass element" (join-worthy) from "this test is unknown because
/// it calls something this module cannot evaluate at all" (`if f():`
/// over a CONCRETE per-element iterate, `unknown_if_test_on_any_
/// iteration_declines_the_whole_loop`'s own pin) — an opaque call
/// mentions no bound name this reader recognizes, so it still declines
/// exactly as before this fallback existed, rather than joining two
/// arms neither `assume` narrowed at all.
pub(super) fn run_if_once_over_unknown_test(
    if_stmt: &StmtIf,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<StatementOutcome> {
    let (else_body, has_wider_chain): (&[Stmt], bool) = match if_stmt.elif_else_clauses.as_slice() {
        [] => (&[], false),
        [clause] if clause.test.is_none() => (clause.body.as_slice(), false),
        _ => (&[], true),
    };
    if has_wider_chain {
        return None;
    }
    let test = if_stmt.test.as_ref();
    if !test_mentions_a_set_bound_name(test, environment) {
        return None;
    }

    let mut true_arm = environment.fork();
    true_arm = assume(test, true_arm, kernel, true);
    let true_outcome = run_body_once(&if_stmt.body, &mut true_arm, kernel, judge_context)?;
    if !matches!(true_outcome, BodyOutcome::Fell) {
        return None;
    }

    let mut false_arm = environment.fork();
    false_arm = assume(test, false_arm, kernel, false);
    let false_outcome = run_body_once(else_body, &mut false_arm, kernel, judge_context)?;
    if !matches!(false_outcome, BodyOutcome::Fell) {
        return None;
    }

    *environment = Environment::join(true_arm, &false_arm);
    Some(StatementOutcome::Next)
}

/// Whether `test` names at least one bare identifier CURRENTLY bound
/// `Kind::Set` in `environment` — walked over the same leaf vocabulary
/// `narrowing::condition_tree_of`/`collect_names` read (`not`, `and`/
/// `or`, a `Compare`'s two sides, an `isinstance` call's first
/// argument), wide enough to catch every name a real narrowing ask
/// might reach, never wider. A test with no such name (every operand a
/// literal, or an opaque call `narrowing::narrow`'s own `Call` arm does
/// not recognize) answers `false` — this function's own caller reads
/// that as "nothing here for `assume` to narrow," not as a shape to
/// guess at.
fn test_mentions_a_set_bound_name(test: &Expr, environment: &Environment) -> bool {
    match test {
        Expr::Name(name) => environment.read(name.id.as_str()).is_some_and(|value| value.kind == Kind::Set),
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => test_mentions_a_set_bound_name(&unary.operand, environment),
        Expr::BoolOp(bool_op) => bool_op.values.iter().any(|value| test_mentions_a_set_bound_name(value, environment)),
        Expr::Compare(compare) => {
            test_mentions_a_set_bound_name(&compare.left, environment)
                || compare.comparators.iter().any(|comparator| test_mentions_a_set_bound_name(comparator, environment))
        }
        Expr::Call(call) => {
            let Expr::Name(func_name) = call.func.as_ref() else {
                return false;
            };
            if func_name.id.as_str() != "isinstance" || call.arguments.args.len() != 2 {
                return false;
            }
            test_mentions_a_set_bound_name(&call.arguments.args[0], environment)
        }
        _ => false,
    }
}

/// Folds a nested `run_body_once` result (an `if` arm's own body, which
/// may itself `break`/`continue`/`return`) into this statement's own
/// outcome — `break`/`continue`/`return` inside an `if` arm propagates
/// exactly as if it had appeared directly in the enclosing loop body
/// (compound_stmts.rst places no restriction on `break`/`continue`
/// nesting inside `if`, and a `return` statement is legal anywhere a
/// function body reaches). `Continued` maps to `StatementOutcome::Continue`
/// (not `Next`) so the ENCLOSING body's own `run_body_once` statement
/// loop also stops at the `if` statement rather than running whatever
/// comes after it this iteration; `Returned` maps straight through the
/// same way.
pub(super) fn outcome_of_body(outcome: BodyOutcome) -> StatementOutcome {
    match outcome {
        BodyOutcome::Fell => StatementOutcome::Next,
        BodyOutcome::Broke => StatementOutcome::Break,
        BodyOutcome::Continued => StatementOutcome::Continue,
        BodyOutcome::Returned(value, range) => StatementOutcome::Returned(value, range),
    }
}

/// A bare expression-statement inside a loop body: only a mutating
/// method call on a bare-name receiver (`name.method(args)`) is
/// modeled, through the MUTATION CONTRACT
/// (`collection_models::mutated_receiver`) — `Some((new_receiver,
/// _call_result))` rebinds `name` to the new receiver (the call
/// result itself is discarded, same as every other statement-position
/// sink in this file: a loop body never reads a bare expression
/// statement's own value back) — OR the one chained shape
/// `run_setdefault_append_once` recognizes
/// (`name.setdefault(key, default).append(value)`, dict_groupby's own
/// group-by idiom, c-reads-and-values.py:1007). Any other expression
/// statement (a read with no effect, a call this module cannot
/// resolve) is `None`.
pub(super) fn run_expr_statement_once(
    expr: &Expr,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<StatementOutcome> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if let Some(outcome) = run_setdefault_append_once(call, attribute, environment, kernel) {
        return Some(outcome);
    }
    let Expr::Name(receiver_name) = attribute.value.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let receiver = environment.read(receiver_name.id.as_str())?.clone();
    let mut arguments = Vec::with_capacity(call.arguments.args.len());
    for argument in call.arguments.args.iter() {
        arguments.push(evaluate_expression(argument, environment, kernel));
    }
    let (new_receiver, _call_result) =
        collection_models::mutated_receiver(attribute.attr.as_str(), &receiver, &arguments)?;
    // a mutating-call receiver is a container (list/dict/set), never
    // itself a scalar declared slot — matches run_subscript_assign_once's
    // own reasoning: this rebind is not a `declared`-table judging
    // candidate, so it binds directly rather than through bind_checked.
    environment.bind(receiver_name.id.as_str(), new_receiver);
    Some(StatementOutcome::Next)
}

/// `name.setdefault(<key>, <default>).append(<value>)` — the manual
/// group-by idiom (`dict_groupby`, c-reads-and-values.py:1007's own
/// shape: `grouped.setdefault("old" if age > 100 else "young",
/// []).append(age)`): `name` must be a bare-name receiver already bound
/// to a known `Kind::Object`, and the outer call's OWN attribute must
/// be `append` with exactly one positional argument (`value`) and no
/// keywords. The chain's inner call — `attribute.value`, the `append`
/// receiver — must itself be exactly `name.setdefault(key[, default])`
/// (stdtypes.rst's own dict `setdefault(key, default=None)` row: "If
/// *key* is in the dictionary, return its value. If not, insert *key*
/// with a value of *default* and return *default*"), so its own answer
/// composes the two contracts already proved elsewhere in this crate
/// rather than re-deriving either: `collection_models::mutated_receiver`
/// answers `(dict-after-setdefault, entry-value)` for the inner call
/// exactly as `run_expr_statement_once`'s own bare-mutating-call arm
/// would if `setdefault` sat alone in statement position, and the entry
/// value it answers must itself be a `Kind::List` — `.append`'s own
/// receiver contract (`list.append`, stdtypes.rst) — for `append`'s own
/// row of `mutated_receiver` to answer the appended list. The final
/// write is `dict_with_item(dict-after-setdefault, key, appended-list)`
/// (`collection_models::dict_with_item`'s own `d[key] = value` contract)
/// rather than a second walk of `setdefault`'s own key-presence branch —
/// `setdefault`'s dict-after-answer already carries the right entry
/// whether the key was present (unchanged) or absent (freshly inserted
/// with the default), so overwriting that SAME key with the appended
/// list is correct either way. `key` is evaluated ONCE against the
/// current environment (matching CPython's own single left-to-right
/// evaluation of a chained call's every sub-expression) and reused for
/// both the `setdefault` receiver-answer and the final rebind — this
/// function never re-evaluates it. `None` for anything off this exact
/// shape (a non-Name inner receiver, a wrong argument count/keyword on
/// either call, a non-Object/non-List intermediate value, an
/// unresolved `setdefault`/`append` row) — the caller's own bare-call
/// arm, or an outer decline, is the fallback.
pub(super) fn run_setdefault_append_once(
    outer_call: &ExprCall,
    outer_attribute: &ExprAttribute,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<StatementOutcome> {
    if outer_attribute.attr.as_str() != "append" {
        return None;
    }
    if !outer_call.arguments.keywords.is_empty() {
        return None;
    }
    let [value_expr] = &*outer_call.arguments.args else {
        return None;
    };
    let Expr::Call(inner_call) = outer_attribute.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(inner_attribute) = inner_call.func.as_ref() else {
        return None;
    };
    if inner_attribute.attr.as_str() != "setdefault" {
        return None;
    }
    let Expr::Name(receiver_name) = inner_attribute.value.as_ref() else {
        return None;
    };
    if !inner_call.arguments.keywords.is_empty() {
        return None;
    }
    let (key_expr, default_expr) = match &*inner_call.arguments.args {
        [key] => (key, None),
        [key, default] => (key, Some(default)),
        _ => return None,
    };
    let receiver = environment.read(receiver_name.id.as_str())?.clone();
    let key = evaluate_expression(key_expr, environment, kernel);
    let mut setdefault_arguments = Vec::with_capacity(2);
    setdefault_arguments.push(key.clone());
    if let Some(default_expr) = default_expr {
        setdefault_arguments.push(evaluate_expression(default_expr, environment, kernel));
    }
    let (dict_after_setdefault, entry_value) =
        collection_models::mutated_receiver("setdefault", &receiver, &setdefault_arguments)?;
    let value = evaluate_expression(value_expr, environment, kernel);
    let (appended_list, _null_result) = collection_models::mutated_receiver("append", &entry_value, &[value])?;
    let written_receiver = collection_models::dict_with_item(&dict_after_setdefault, &key, &appended_list)?;
    environment.bind(receiver_name.id.as_str(), written_receiver);
    Some(StatementOutcome::Next)
}
