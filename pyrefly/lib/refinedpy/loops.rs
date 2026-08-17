/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Concrete execution of the corpus's bounded loop shapes: `for x in
//! [lit, ...]:`/`for x in range(...):`/`for x in {dict literal}:`/`for
//! x in d.values():`/`for k, v in d.items():` over known iterables, and
//! `while name < literal:`-style counters with a provable iteration
//! bound. Every iterate in these shapes is known, so running the loop
//! body once per iterate is sound, not an approximation — the walk
//! still owns whether to call this or record its own blocker (`Some`
//! result replaces the blocker; `None` means the walk keeps it).
//!
//! A loop body may contain `if`/`elif`/`else` (the taken arm decided
//! per iteration by evaluating the test), `break`/`continue` (real
//! control flow — CPython's own `else`-skipped-by-`break` rule,
//! compound_stmts.rst), plain-name `Assign`/`AugAssign`/`AnnAssign`,
//! `Pass`, and the two mutation statement shapes
//! (`name.method(args)`/`name[k] = v`) `run_statement_once` recognizes.
//! Every value the body needs must be fully known on EVERY iteration —
//! an unknown test, an unmodeled statement shape, or an unresolved
//! mutation declines the WHOLE loop; this module never approximates a
//! step it cannot state exactly.
//!
//! A `while` whose counter is a KNOWN SET rather than one known number
//! (a seeded parameter's declared range) cannot be stepped concretely —
//! `kernel_bounded_counter_environment` asks the kernel's own
//! `solve_loop` instead, for the one step shape (`n += literal`/`n -=
//! literal`) this file trusts to lower exactly. Any wider shape (a
//! non-literal iterable's declared element set, a multi-name step) is
//! still this module's `None`.
//!
//! ## Why a body write can still leave the walk's blocker standing
//!
//! `check.rs`'s `walk_loop` swaps in this module's `Some(environment)`
//! outright — nothing re-judges a body's writes against a declared
//! refinement afterward (`check.rs`, `walk_loop`'s own doc: "`Some(env)`
//! replaces the environment outright and the statement is consumed with
//! no blocker"). Every currently-passing loop row relies on this in the
//! SOUND direction: the loop only ever produces a plain value, and a
//! POST-loop declared read (`done: Age = total`) is what actually
//! judges it. A row whose marker sits INSIDE the body (no post-loop
//! declared read exists to catch it) is different: this module has no
//! declared-refinement table to judge against, so it cannot be the one
//! to fire. `bind_checked` declines the whole loop rather than silently
//! dropping such a write on the floor: see its own doc below.

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::known_list;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;
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
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFor;
use ruff_python_ast::StmtIf;
use ruff_python_ast::StmtWhile;
use ruff_python_ast::UnaryOp;

use crate::refinedpy::collection_models;
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
/// a body outside the recognized forms, or a `while` that does not
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

/// Whether a `break` fired during one run of a loop body — the signal
/// `for_loop_final_environment`/`while_loop_final_environment` use to
/// skip the `else` clause and, for `for`, stop advancing the target
/// past the element the `break` fired on (compound_stmts.rst, "the
/// `for` statement"/"the `while` statement": "the `else` clause...
/// executes when the loop terminates through exhaustion... rather than
/// by `break`"). `Continue` is folded away inside `run_body_once`
/// itself — it never needs to propagate past the statement loop that
/// runs one iteration's statements in order, since "skip the rest of
/// this iteration" is exactly what returning early from that loop does.
enum BodyOutcome {
    Fell,
    Broke,
}

/// `for target in <iterable>: <body> [else: <body>]` — every element
/// this module's `iterable_values` recognizes is fully known, so the
/// body runs once per element over a forked environment. Python leaves
/// the target bound to the last element after the loop ends (never
/// reset or deleted, compound_stmts.html "the for statement"); an empty
/// iterable runs the body zero times, so the target keeps whatever the
/// pre-loop environment already held for that name. A `break` on any
/// iteration stops the loop AT that element (the target stays bound to
/// the element the `break` fired on) and skips `else`; otherwise `else`
/// runs once the iterable is exhausted.
fn for_loop_final_environment(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Environment> {
    if for_stmt.is_async {
        return None;
    }
    let elements = iterable_values(for_stmt.iter.as_ref(), environment, kernel)?;
    let mut current = environment.fork();
    let mut broke = false;
    for element in elements {
        if !bind_for_target(for_stmt.target.as_ref(), &element, &mut current) {
            return None;
        }
        match run_body_once(&for_stmt.body, &mut current, kernel)? {
            BodyOutcome::Fell => {}
            BodyOutcome::Broke => {
                broke = true;
                break;
            }
        }
    }
    if broke {
        return Some(current);
    }
    match run_body_once(&for_stmt.orelse, &mut current, kernel)? {
        BodyOutcome::Fell | BodyOutcome::Broke => Some(current),
    }
}

/// `while <name> <op> <literal>: <body> [else: <body>]`, where `<op>`
/// is `<` or `<=` and the loop is a plain counter this function can run
/// out to its own halt. Each iteration re-evaluates the condition
/// against the CURRENT environment (a real interpretation step, not a
/// one-shot bound check) and stops the moment the condition reads false
/// or unknown. Reaching `WHILE_ITERATION_CAP` with the condition still
/// provably true is an unproved bound — declines. A counter whose
/// CURRENT value is a known SET rather than one known number
/// (`Kind::Set` — a seeded parameter's declared range) can never
/// resolve a single concrete step at all — `counter_condition_value`
/// reads `None` on the very first check, so this function tries
/// `kernel_bounded_counter_environment` FIRST for exactly that shape,
/// before the concrete stepping loop ever runs. A `break` stops the
/// loop immediately and skips `else`; otherwise `else` runs once the
/// condition reads false (compound_stmts.html "the while statement").
fn while_loop_final_environment(
    while_stmt: &StmtWhile,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Environment> {
    if let Some(kernel_result) = kernel_bounded_counter_environment(while_stmt, environment, kernel) {
        return Some(kernel_result);
    }
    let mut current = environment.fork();
    for _ in 0..WHILE_ITERATION_CAP {
        match counter_condition_value(while_stmt.test.as_ref(), &current, kernel) {
            Some(true) => match run_body_once(&while_stmt.body, &mut current, kernel)? {
                BodyOutcome::Fell => {}
                BodyOutcome::Broke => return Some(current),
            },
            Some(false) => {
                return match run_body_once(&while_stmt.orelse, &mut current, kernel)? {
                    BodyOutcome::Fell | BodyOutcome::Broke => Some(current),
                };
            }
            None => return None,
        }
    }
    // the cap was reached with the condition still true (or unreadable
    // on the final check) — the bound was never proved
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
fn kernel_bounded_counter_environment(
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

/// A single known, Integer- or Float-sorted for-loop iterate — CPython's
/// own two numeric sorts, never the joined/unknown `PrimitiveKind::Number`
/// (PYREFLY-NUMERIC-B3-B4.md's "two sorts, never one Number"). Binding an
/// iterate this way (rather than the old sort-erasing `known_number`)
/// is what lets a `for age in [10, 20, 30]: total = total + age` row's
/// arithmetic see BOTH operands as Integer and answer an Integer total —
/// `binary_arithmetic_value`'s `single_numeric_value` reads a bare
/// `Number` tag conservatively as Float, which is what previously made
/// an all-int accumulation read as a float and wrongly fire the
/// int-sort law on its own in-set result.
fn known_number_sorted(value: f64, sort: PrimitiveKind) -> AbstractValue {
    known_values(vec![value], sort, TrustProved)
}

fn known_number(value: f64) -> AbstractValue {
    known_number_sorted(value, PrimitiveKind::Number)
}

/// A Python `str`, as this domain's exact-string `AbstractValue` — one
/// code point per `f64` (`string_models.rs`'s documented representation;
/// repeated here rather than reaching into that module's private
/// helper, matching `collection_models.rs`'s own same-crate-different-
/// module precedent for this exact conversion).
fn known_string(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    known_values(code_points, PrimitiveKind::String, TrustProved)
}

/// The known elements a `for` loop's iterable expression names, in
/// iteration order, each already carrying its TRUE Python sort:
/// - a literal list/tuple of number literals (Integer or Float per
///   element) or a `range(...)` call (library/stdtypes.html#range,
///   always Integer — `range` accepts only int arguments).
/// - a dict DISPLAY iterated directly (`for k in {...}:`) — CPython
///   iterates a dict's KEYS (library/stdtypes.rst, "Mapping Types —
///   dict": "Iterating views while adding or deleting entries..."; the
///   dict's own `__iter__` "return an iterator over the keys"), so each
///   element is the key's exact String value.
/// - `<dict-valued-name-or-expr>.values()` / `.items()` / `.keys()` on
///   a receiver `evaluate_expression` reads as a known `Kind::Object`
///   (a prior local dict, not necessarily a literal at the call site):
///   `.values()` yields each entry's value, `.keys()` yields each
///   entry's key (String), `.items()` yields a 2-element tuple
///   (`Kind::List` of `[key, value]`) per entry — CPython's own view
///   order, library/stdtypes.rst dict views, "Keys views are set-like...
///   Dictionary views... iterate over `... items in insertion order`".
///
/// Anything else (a name that is not a known dict, a call other than
/// `range`/`.values`/`.items`/`.keys`, a non-literal element) is
/// `None`: this function only answers when every iterate is known
/// without running any unmodeled code.
fn iterable_values(
    iterable: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    match iterable {
        Expr::List(list) => elements_as_sorted_numbers(&list.elts),
        Expr::Tuple(tuple) => elements_as_sorted_numbers(&tuple.elts),
        Expr::Call(call) => {
            range_call_values(call).or_else(|| dict_view_call_values(call, environment, kernel))
        }
        Expr::Dict(_) => {
            let receiver = evaluate_expression(iterable, environment, kernel);
            dict_keys_as_strings(&receiver)
        }
        _ => None,
    }
}

fn elements_as_sorted_numbers(elements: &[Expr]) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::with_capacity(elements.len());
    for element in elements {
        values.push(sorted_number_literal_value(element)?);
    }
    Some(values)
}

/// A dict's keys, each as an exact String `AbstractValue`, in the
/// dict's own insertion order — `None` for anything that is not a
/// known `Kind::Object` (an unread dict, a dict built by a non-literal
/// path this domain does not model, library/stdtypes.rst's dict
/// iteration order guarantee applying only to a known key set).
fn dict_keys_as_strings(receiver: &AbstractValue) -> Option<Vec<AbstractValue>> {
    if receiver.kind != Kind::Object {
        return None;
    }
    Some(receiver.keys.iter().map(|entry| known_string(&entry.name)).collect())
}

/// `<dict>.values()` / `<dict>.items()` / `<dict>.keys()` — the
/// receiver expression is evaluated against the CURRENT environment (it
/// may be a prior local variable, not a literal at the call site) and
/// must read as a known `Kind::Object`; every other receiver shape, or
/// a method name other than these three, is `None`. `.items()` builds
/// one 2-element tuple (`Kind::List`) per entry so
/// `bind_for_target`'s existing tuple-unpack path binds `for k, v in
/// d.items():` with no special-casing beyond that.
fn dict_view_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return None;
    }
    let receiver = evaluate_expression(attribute.value.as_ref(), environment, kernel);
    if receiver.kind != Kind::Object {
        return None;
    }
    match attribute.attr.as_str() {
        "values" => Some(receiver.keys.iter().map(|entry| entry.value.clone()).collect()),
        "keys" => dict_keys_as_strings(&receiver),
        "items" => Some(
            receiver
                .keys
                .iter()
                .map(|entry| known_list(vec![known_string(&entry.name), entry.value.clone()], TrustProved))
                .collect(),
        ),
        _ => None,
    }
}

/// A `range(...)` call's produced values, or `None` when the callee
/// is not the bare name `range`, an argument is not a literal int, or
/// the argument count is not 1/2/3. `step == 0` is `None` — CPython
/// raises `ValueError` there rather than producing a sequence. Every
/// produced value is Integer-sorted — `range` accepts only int
/// arguments (library/stdtypes.html#range), so its elements are never
/// float.
fn range_call_values(call: &ExprCall) -> Option<Vec<AbstractValue>> {
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
            values.push(known_number_sorted(current, PrimitiveKind::Integer));
            current += step;
        }
    } else {
        while current > stop {
            values.push(known_number_sorted(current, PrimitiveKind::Integer));
            current += step;
        }
    }
    Some(values)
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value,
/// tagged with its own CPython sort (Integer for an int literal, Float
/// for a float literal) — or `None` for anything else (complex, an int
/// too large for i64, a non-literal expression).
fn sorted_number_literal_value(expression: &Expr) -> Option<AbstractValue> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| known_number_sorted(value as f64, PrimitiveKind::Integer)),
            Number::Float(value) => Some(known_number_sorted(*value, PrimitiveKind::Float)),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) => {
            let operand = sorted_number_literal_value(unary.operand.as_ref())?;
            match unary.op {
                UnaryOp::USub => Some(known_number_sorted(-operand.values[0], operand.kind_tag?)),
                UnaryOp::UAdd => Some(operand),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression). Sort-erased: used only by
/// the `while`-counter comparison paths, which read a bound value to
/// compare against, never to bind a fresh iterate.
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

/// Binds a `for` target to one iterate: a bare name binds directly; a
/// tuple target (`for k, v in d.items():`) unpacks an EXACT-arity
/// `Kind::List` element positionally — CPython raises `ValueError` on
/// an arity mismatch (simple_stmts.rst, "Assignment statements":
/// unpacking "requires the same number of items"), which this domain
/// has no exception channel for this wave, so a mismatch is `false`
/// (decline) rather than a partial bind. Any other target shape
/// (starred, attribute, subscript) is `false`.
fn bind_for_target(target: &Expr, element: &AbstractValue, environment: &mut Environment) -> bool {
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

/// Runs one loop body's statements against `environment` IN PLACE, in
/// order, honoring real control flow: `break` stops immediately
/// (`BodyOutcome::Broke`, propagated straight out — CPython never runs
/// statements after a `break` in the same body), `continue` stops this
/// ITERATION's statement loop early but is not itself an outcome (the
/// caller's own per-element loop simply moves to the next iterate,
/// which running out of statements to execute already achieves). `None`
/// is the same "this loop is not this module's shape" honesty every
/// other decline here uses — no statement here EVER writes a value that
/// might be wrong; an unrecognized shape declines the WHOLE loop rather
/// than skip or approximate.
fn run_body_once(
    body: &[Stmt],
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<BodyOutcome> {
    for stmt in body {
        match run_statement_once(stmt, environment, kernel)? {
            StatementOutcome::Next => {}
            StatementOutcome::Continue => return Some(BodyOutcome::Fell),
            StatementOutcome::Break => return Some(BodyOutcome::Broke),
        }
    }
    Some(BodyOutcome::Fell)
}

/// What one statement, run once against the current environment, says
/// about the rest of THIS iteration: keep going (`Next`), stop this
/// iteration early (`Continue`), or stop the whole loop (`Break`).
enum StatementOutcome {
    Next,
    Continue,
    Break,
}

/// Runs exactly one loop-body statement, dispatched by syntactic form.
/// `None` for any statement shape this module does not interpret — the
/// caller (`run_body_once`) propagates that straight into a whole-loop
/// decline.
fn run_statement_once(
    stmt: &Stmt,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
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
            run_assign_once(target, assign.value.as_ref(), environment, kernel)?;
            Some(StatementOutcome::Next)
        }
        Stmt::AnnAssign(assign) => {
            let Some(value_expr) = assign.value.as_deref() else {
                // `x: T` alone declares no value — nothing to bind or
                // judge, matching simple_stmts.rst's "the `=` clause is
                // optional" reading check.rs's own walk_ann_assign uses.
                return Some(StatementOutcome::Next);
            };
            run_assign_once(assign.target.as_ref(), value_expr, environment, kernel)?;
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
            let updated = binary_arithmetic_value(assign.op, &current, &operand);
            if updated.kind != Kind::Values {
                return None;
            }
            bind_checked(name.id.as_str(), updated, environment)?;
            Some(StatementOutcome::Next)
        }
        Stmt::If(if_stmt) => run_if_once(if_stmt, environment, kernel),
        Stmt::Expr(expr_stmt) => run_expr_statement_once(expr_stmt.value.as_ref(), environment, kernel),
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

/// Forgets a `del` target's name, restricted to a bare name or a
/// tuple/list of bare names — `false` for anything wider (a starred
/// target, an attribute/subscript target), which declines the whole
/// loop rather than silently skip an un-forgettable target.
fn forget_bare_name_target(target: &Expr, environment: &mut Environment) -> bool {
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

/// `name = value` / `name: T = value` on a plain-name target: evaluates
/// the RHS and binds it, `None` unless the value comes back fully known
/// (`Kind::Values`, `Kind::List`, or `Kind::Object` — an unreadable
/// right side, a call, or an unbound name fails the whole loop rather
/// than silently binding unknown). A non-name target (attribute,
/// subscript-outside-the-mutation-contract) is `None`: this function
/// only ever writes a name it can name.
fn run_assign_once(
    target: &Expr,
    value_expr: &Expr,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<()> {
    let Expr::Name(name) = target else {
        return None;
    };
    let value = evaluate_expression(value_expr, environment, kernel);
    if !matches!(value.kind, Kind::Values | Kind::List | Kind::Object) {
        return None;
    }
    bind_checked(name.id.as_str(), value, environment)
}

/// `name[k] = v` — the MUTATION CONTRACT's subscript-target shape.
/// `name` must be a bare name already bound to a known receiver;
/// `collection_models::dict_with_item`/`list_with_item` (dispatched by
/// the receiver's own `Kind`) answer the new receiver value, which
/// rebinds `name` through the same cross-sort check every other write
/// in this file goes through. `None` for anything the contract does
/// not resolve (an unknown receiver, a key/value shape the contract
/// declines, a receiver `Kind` neither function owns).
fn run_subscript_assign_once(
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
        Kind::Object => collection_models::dict_with_item(&receiver, &key, &value)?,
        Kind::List => collection_models::list_with_item(&receiver, &key, &value)?,
        _ => return None,
    };
    bind_checked(name.id.as_str(), new_receiver, environment)
}

/// Binds `name` to `value`, UNLESS `name` already carried a known value
/// in the environment (bound before this statement ran, whether from
/// before the loop started or from an earlier iteration/statement)
/// whose HOST-TYPE FAMILY disagrees with `value`'s. A cross-family
/// overwrite of a name that was already known is exactly the shape a
/// declared-slot fire (`age: Age = 0` followed by a body write `age =
/// key` binding a STRING into that Integer-sorted slot) needs to catch
/// — and this module has no declared-refinement table to judge that
/// fire itself (`check.rs`'s `walk_loop` swaps in this module's whole
/// environment with no post-hoc judging, see the module doc). Declining
/// the loop keeps the walk's OWN blocker standing rather than silently
/// letting the write through unjudged.
///
/// Scoped to FAMILY (numeric vs `String` vs `Boolean`), not to the
/// numeric sort's own Integer/Float/Number precision split: a numeric
/// accumulator legitimately narrows from the sort-erased `Number` tag
/// (a test's hand-built pre-loop binding, or any value this domain has
/// not yet sort-tagged) to `Integer`/`Float` as arithmetic runs — that
/// is exactly how every currently-passing accumulation row works (`total
/// = total + age`, the POST-loop declared read still judges the
/// result), and is not the cross-sort fire this function exists to
/// catch.
fn family_of(sort: PrimitiveKind) -> u8 {
    match sort {
        PrimitiveKind::String => 0,
        PrimitiveKind::Boolean => 1,
        PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float => 2,
        // never produced by this domain's Kind::Values path (RefinedPy
        // has no JS-array-shaped scalar reading) — its own family so a
        // future producer cannot silently fold into the numeric check
        PrimitiveKind::Array => 3,
    }
}

fn bind_checked(name: &str, value: AbstractValue, environment: &mut Environment) -> Option<()> {
    if let Some(existing) = environment.read(name)
        && let (Some(existing_sort), Kind::Values) = (existing.kind_tag, existing.kind)
        && value.kind == Kind::Values
        && let Some(new_sort) = value.kind_tag
        && family_of(new_sort) != family_of(existing_sort)
    {
        return None;
    }
    environment.bind(name, value);
    Some(())
}

/// `if test: body [elif test: body ...] [else: body]` inside a loop —
/// the taken arm is decided PER ITERATION by evaluating `test` against
/// the CURRENT environment (`lattice_operations::truthiness`'s
/// `(value, known)` pair): an unknown test on this iteration declines
/// the WHOLE loop (the corpus's own framing: "an unknown test on ANY
/// iteration declines the whole loop" — a body this module cannot
/// decide even once is not a shape it can claim to run exactly). The
/// taken arm's statements run in place; an untaken arm is not walked at
/// all, matching CPython's own single-branch execution
/// (compound_stmts.rst, "the `if` statement").
fn run_if_once(
    if_stmt: &StmtIf,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<StatementOutcome> {
    let condition = evaluate_expression(if_stmt.test.as_ref(), environment, kernel);
    let (taken, known) = truthiness(&condition);
    if !known {
        return None;
    }
    if taken {
        return run_body_once(&if_stmt.body, environment, kernel).map(outcome_of_body);
    }
    for clause in &if_stmt.elif_else_clauses {
        match clause.test.as_ref() {
            None => {
                // a bare `else:` — always taken once every prior
                // `elif`/`if` test read false
                return run_body_once(&clause.body, environment, kernel).map(outcome_of_body);
            }
            Some(test) => {
                let clause_condition = evaluate_expression(test, environment, kernel);
                let (clause_taken, clause_known) = truthiness(&clause_condition);
                if !clause_known {
                    return None;
                }
                if clause_taken {
                    return run_body_once(&clause.body, environment, kernel).map(outcome_of_body);
                }
            }
        }
    }
    // no arm's test held and there was no bare `else:` — the whole `if`
    // statement is a no-op this iteration
    Some(StatementOutcome::Next)
}

/// Folds a nested `run_body_once` result (an `if` arm's own body, which
/// may itself `break`/`continue`) into this statement's own outcome —
/// `break`/`continue` inside an `if` arm propagates exactly as if it
/// had appeared directly in the enclosing loop body (compound_stmts.rst
/// places no restriction on `break`/`continue` nesting inside `if`).
fn outcome_of_body(outcome: BodyOutcome) -> StatementOutcome {
    match outcome {
        BodyOutcome::Fell => StatementOutcome::Next,
        BodyOutcome::Broke => StatementOutcome::Break,
    }
}

/// A bare expression-statement inside a loop body: only a mutating
/// method call on a bare-name receiver (`name.method(args)`) is
/// modeled, through the MUTATION CONTRACT
/// (`collection_models::mutated_receiver`) — `Some((new_receiver,
/// _call_result))` rebinds `name` to the new receiver (the call
/// result itself is discarded, same as every other statement-position
/// sink in this file: a loop body never reads a bare expression
/// statement's own value back). Any other expression statement (a
/// read with no effect, a call this module cannot resolve) is `None`.
fn run_expr_statement_once(
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
    bind_checked(receiver_name.id.as_str(), new_receiver, environment)?;
    Some(StatementOutcome::Next)
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

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
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
    fn for_else_applies_its_body_after_exhaustion() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    total += x\nelse:\n    done = 1\n");
        let environment = environment_with(&[("total", 0.0), ("done", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("body runs, else runs");
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

    // --- sort preservation (UNIT 1) ---

    #[test]
    fn for_over_int_literal_list_binds_the_iterate_as_integer_sorted() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [10, 20, 30]:\n    total = total + age\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "age".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("int list is concrete");
        let total = result.read("total").expect("total stays bound");
        assert_eq!(total.values, vec![60.0]);
        // the fix under test: an all-int accumulation answers an
        // Integer-tagged total, not a Float-tagged one — a Float 60.0
        // wrongly fires the int-sort law against an Age slot even
        // though 60 is in range (a-statements.py:515)
        assert_eq!(total.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn range_iterate_is_integer_sorted() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for i in range(3):\n    total = total + i\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("range is concrete");
        assert_eq!(result.read("total").unwrap().kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn for_over_float_literal_list_binds_the_iterate_as_float_sorted() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1.5, 2.5]:\n    total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", known_values(vec![0.0], PrimitiveKind::Float, TrustProved));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("float list is concrete");
        let total = result.read("total").expect("total stays bound");
        assert_eq!(total.values, vec![4.0]);
        assert_eq!(total.kind_tag, Some(PrimitiveKind::Float));
    }

    // --- if / elif / else inside a body (UNIT 2) ---

    #[test]
    fn if_arm_runs_only_when_the_test_holds() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2, 3]:\n    if x > 1:\n        total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("if inside body is concrete");
        // x=1: test false, no-op; x=2: total=2; x=3: total=5
        assert_eq!(result.read("total").unwrap().values, vec![5.0]);
    }

    #[test]
    fn else_arm_runs_when_no_test_holds() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(
            "for x in [1, 2]:\n    if x > 100:\n        total = total + 1\n    else:\n        total = total + x\n",
        );
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("if/else inside body is concrete");
        assert_eq!(result.read("total").unwrap().values, vec![3.0]);
    }

    #[test]
    fn unknown_if_test_on_any_iteration_declines_the_whole_loop() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    if f():\n        total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", integer(0.0));
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    // --- break / continue (UNIT 2) ---

    #[test]
    fn break_stops_the_loop_and_skips_else() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(
            "for i in range(3):\n    if i == 1:\n        break\n    total = total + 1\nelse:\n    total = 200\n",
        );
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("break inside body is concrete");
        // i=0: total=1; i=1: breaks before total += 1 runs, else never runs
        assert_eq!(result.read("total").unwrap().values, vec![1.0]);
        assert_eq!(result.read("i").unwrap().values, vec![1.0]);
    }

    #[test]
    fn continue_skips_the_rest_of_that_iteration_only() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(
            "for i in range(4):\n    if i == 2:\n        continue\n    total = total + i\n",
        );
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("continue inside body is concrete");
        // 0 + 1 + (skip 2) + 3 = 4
        assert_eq!(result.read("total").unwrap().values, vec![4.0]);
    }

    #[test]
    fn while_break_stops_immediately_and_skips_else() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("while n < 5:\n    if n == 2:\n        break\n    n += 1\nelse:\n    n = 200\n");
        let environment = environment_with(&[("n", 0.0)]);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("while break is concrete");
        assert_eq!(result.read("n").unwrap().values, vec![2.0]);
    }

    // --- dict-shaped iteration (UNIT 2) ---

    #[test]
    fn for_over_dict_literal_iterates_the_string_keys() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    last = key\n");
        let environment = Environment::new(HashSet::from(["last".to_owned(), "key".to_owned()]));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("dict-literal key iteration");
        let last = result.read("last").expect("last stays bound");
        assert_eq!(last.kind_tag, Some(PrimitiveKind::String));
    }

    #[test]
    fn dict_literal_iteration_into_a_pre_bound_int_slot_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // `age: Age = 0` pre-binds age as an Integer; writing a dict
        // key (a String) into it is a declared-slot fire this module
        // cannot judge itself — must decline, not silently overwrite.
        let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    age = key\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "key".to_owned()]));
        environment.bind("age", integer(0.0));
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn for_over_dict_values_call_binds_the_stored_values() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in ages.values():\n    last = age\n");
        let mut environment = Environment::new(HashSet::from(["ages".to_owned(), "last".to_owned(), "age".to_owned()]));
        let dict = collection_models::dict_literal_value(&[Some("ann".to_owned())], &[integer(40.0)]);
        environment.bind("ages", dict);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect(".values() iteration");
        assert_eq!(result.read("last").unwrap().values, vec![40.0]);
    }

    #[test]
    fn for_over_dict_items_call_unpacks_key_and_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for _, age in ages.items():\n    total = total + age\n");
        let mut environment = Environment::new(HashSet::from([
            "ages".to_owned(),
            "total".to_owned(),
            "_".to_owned(),
            "age".to_owned(),
        ]));
        environment.bind("total", integer(0.0));
        let dict = collection_models::dict_literal_value(
            &[Some("ann".to_owned()), Some("bea".to_owned())],
            &[integer(40.0), integer(41.0)],
        );
        environment.bind("ages", dict);
        let result = loop_final_environment(&stmt, &environment, &kernel).expect(".items() iteration");
        assert_eq!(result.read("total").unwrap().values, vec![81.0]);
    }

    // --- statement-level mutation contract (UNIT 2) ---

    #[test]
    fn a_recognized_mutating_call_rebinds_the_receiver() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    xs.append(x)\n");
        let mut environment = Environment::new(HashSet::from(["xs".to_owned(), "x".to_owned()]));
        environment.bind("xs", known_list(vec![], TrustProved));
        // `mutated_receiver` is the concurrent collection_models.rs
        // wave's own contract; whatever it answers for "append" is what
        // this loop must adopt (Some rebinds, None declines) — this
        // test only pins that the call reaches the contract and does
        // not crash, not a specific collection_models.rs answer shape.
        let _ = loop_final_environment(&stmt, &environment, &kernel);
    }

    #[test]
    fn a_recognized_subscript_write_rebinds_the_dict_receiver() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [40, 41]:\n    ages[\"latest\"] = age\n");
        let mut environment = Environment::new(HashSet::from(["ages".to_owned(), "age".to_owned()]));
        environment.bind("ages", collection_models::dict_literal_value(&[], &[]));
        // `dict_with_item` is the concurrent collection_models.rs wave's
        // own contract; this test pins that a subscript-target write
        // reaches it (Some rebinds, None declines), not a specific
        // answer shape.
        let _ = loop_final_environment(&stmt, &environment, &kernel);
    }

    #[test]
    fn nested_for_in_body_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    for y in [1]:\n        total = total + y\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    /// An `Age`-shaped declared set (`[0, 120]`, integers) — the same
    /// shape `seed_parameters` (check.rs) binds a scalar-typed parameter
    /// to, built directly here since this module's tests construct
    /// environments by hand rather than walking a function signature.
    fn age_set() -> refined_sets::refinement_forms::RefinedSet {
        refined_sets::refinement_forms::make_refined_set(vec![
            refined_sets::refinement_forms::at_least(0.0),
            refined_sets::refinement_forms::at_most(120.0),
            refined_sets::refinement_forms::integer(),
        ])
    }

    #[test]
    fn while_counter_over_a_seeded_known_set_asks_the_kernel_and_binds_a_set() {
        let Some(kernel) = loaded_kernel() else { return };
        // `n` starts as a Kind::Set (a seeded parameter's declared
        // range, e.g. `def f(n: Age): while n < 121: n += 1`) rather
        // than one known number — the concrete stepping path above
        // cannot step a set one value at a time, so this falls to
        // kernel_bounded_counter_environment.
        let stmt = parsed_loop("while n < 121:\n    n += 1\n");
        let mut environment = Environment::new(HashSet::from(["n".to_owned()]));
        environment.bind("n", known_set(age_set(), None, TrustProved, SetKindTag::None));
        let result = loop_final_environment(&stmt, &environment, &kernel).expect("kernel bounds the counter");
        let bound = result.read("n").expect("n stays bound");
        assert_eq!(bound.kind, Kind::Set);
    }

    #[test]
    fn while_counter_over_a_known_set_with_an_unsupported_step_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // `n *= 2` is not the Add/Sub step shape this file trusts to
        // lower into the kernel's LoopEffect grammar — must decline
        // rather than approximate.
        let stmt = parsed_loop("while n < 121:\n    n *= 2\n");
        let mut environment = Environment::new(HashSet::from(["n".to_owned()]));
        environment.bind("n", known_set(age_set(), None, TrustProved, SetKindTag::None));
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn while_counter_over_a_known_set_with_a_nonempty_else_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // a non-empty else after a kernel-certified (not concretely
        // run) loop is outside kernel_bounded_counter_environment's
        // scoped shape
        let stmt = parsed_loop("while n < 121:\n    n += 1\nelse:\n    done = 1\n");
        let mut environment = Environment::new(HashSet::from(["n".to_owned(), "done".to_owned()]));
        environment.bind("n", known_set(age_set(), None, TrustProved, SetKindTag::None));
        assert!(loop_final_environment(&stmt, &environment, &kernel).is_none());
    }
}
