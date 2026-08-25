use std::collections::{HashMap, HashSet};

use refined_domain::abstract_value::{AbstractValue, Kind, ObjectKey};
use ruff_python_ast::{
    ExceptHandler, ExceptHandlerExceptHandler, Expr, Stmt, StmtMatch, StmtRaise, StmtTry, StmtWith,
};
use ruff_text_size::{Ranged, TextRange};

use crate::assignability::{judge, Verdict};
use crate::diagnostic_sentences::loop_accumulation_did_not_stabilize;
use crate::env::Environment;
use crate::expressions::{evaluate_expression, fieldless_exception_value};
use crate::instances;
use crate::loops::{loop_final_environment, LoopAnswer};
use crate::match_arms;
use crate::match_arms::match_taken_environment;
use crate::narrowing::assume;
use crate::typereading::DeclaredRefinement;

use super::*;

/// `for`/`while`: `loops::loop_final_environment` concretely executes the
/// bounded shapes it recognizes (literal list/tuple/range iterables,
/// bounded counter `while`s), judging every declared-slot write inside
/// the body against `aug_assign_refinements` as it runs (loops.rs's own
/// contract — a fire lands in `judged_fires`, deduped by statement
/// range) and reporting whether the loop's `else` clause RUNS
/// (`else_runs`). `Some((env, else_runs))` replaces the environment
/// outright and the statement is consumed with no blocker for the
/// `for`/`while` itself; this function then owns the `orelse` body
/// itself — loops.rs never runs it: DEAD-BRANCH LAW's own sibling, the
/// LOOP ELSE + DEAD-ELSE LAW (serves a-statements:446/472/486): when
/// `else_runs`, `orelse` walks through `walk_statement` exactly like any
/// other body (fully judged — this is what makes an else-arm's own
/// out-of-set write fire); when `!else_runs` (every execution provably
/// `break`s), `orelse` never runs, and this function fires RTS7001 at
/// the orelse body's own first statement instead, naming why. `None`
/// from `loop_final_environment` means the shape is outside what that
/// module can run; the walk keeps its own blocker AND forgets every name
/// the loop statement binds anywhere (its target plus every name its
/// body/orelse bind, PLUS every attribute-call/subscript-store receiver
/// the unmodeled body only ever MUTATES — `forget_mutated_receivers_in_stmt`),
/// so a stale pre-loop fact never survives an unmodeled loop that may
/// have rebound it. The blocker is recorded regardless of whether THIS
/// body has already recorded a declared return/yield position or a
/// declared slot (`aug_assign_refinements`) at the loop's OWN position —
/// a slot declared LATER in the same straight-line body (an AnnAssign
/// after the loop, e.g. `for_loop_over_an_unknown_iterable_blocks_and_
/// forgets_its_stale_binding`'s own `check: Age = total`) still has a
/// real sink waiting on this loop's answer, and `aug_assign_refinements`
/// only ever holds what has been WALKED SO FAR — checking it at the
/// loop's own position can never see a later declaration. Recording the
/// blocker unconditionally is sound either way: a body with truly no
/// checked position anywhere still records one blocker (first-blocker-
/// wins, same as every other construct this walk cannot yet handle),
/// never a second one.
///
/// A `Some` answer is not always blocker-free: a `for` loop's own
/// abstract pass (`loops.rs`'s `stabilized_join`) may reach a real
/// stopping point while still havocing one or more written names to
/// `unknown()`, because their value never settled to a fixed point
/// across its two judged passes — `LoopAnswer.widened_names`, sorted for
/// a reproducible first name. This function records THAT as the body's
/// blocker before ever looking at `return_refinement`, so a bare
/// `-> float`/`-> int`/`-> str` return (unreadable to `declared_
/// refinement`, `typereading.rs`'s own doc) never leaves this loop's own
/// genuine undetermined value unnamed.
pub(super) fn walk_loop(
    stmt: &Stmt,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> bool {
    let mut judged_fires: Vec<(TextRange, String)> = Vec::new();
    let result = loop_final_environment(stmt, environment, context.kernel, aug_assign_refinements, &mut judged_fires);
    // A fire recorded ALONGSIDE a decline is `loops.rs`'s own family of
    // checked-before-any-element-runs proofs that no statement after
    // this loop is ever reached on a real execution — either the FIRST
    // iteration unconditionally raises (the dict iterator-invalidation
    // check, `for_loop_final_environment`'s own doc) or the loop itself
    // provably never terminates (the list self-append check,
    // `repetition_window_element_pass`'s own doc, `diagnostic_sentences::
    // list_never_terminates_self_append`) — both read identically here:
    // nothing past this statement in the current body is UNREACHABLE —
    // the same "no fall-through path exists" fact
    // `arm_terminates_or_provably_raises` already states for a `return`/
    // `raise`-terminated arm, just proved by a different construct. This
    // is checked BEFORE the blocker record below so a proved-unreachable
    // loop never reports a spurious "not yet walked" — the fire IS the
    // full account of every real execution, not an unread remainder.
    let raise_terminates = result.is_none() && !judged_fires.is_empty();
    for (range, message) in judged_fires {
        out.push(Finding {
            range,
            code: "RTS7001",
            message,
        });
    }
    if raise_terminates {
        // Nothing after this statement in the current body ever runs —
        // the same bottom environment `walk_try` forks when its own
        // `nothing_survives`. Names this statement bound/mutated are
        // forgotten too: a later read past unreachable code must not
        // resolve to a stale pre-loop value.
        forget_names_bound_by_stmt(stmt, environment);
        forget_mutated_receivers_in_stmt(stmt, environment);
        *environment = environment.fork();
        return true;
    }
    if let Some(LoopAnswer { environment: final_env, else_runs, returned, widened_names }) = result {
        *environment = final_env;
        // A `for` loop's own abstract pass reached a real stopping
        // point, but one or more of its OWN written names never settled
        // to a fixed point across its two judged passes
        // (`loops.rs::stabilized_join`'s own doc) and reads as
        // `unknown()` from here on — a genuine blocker this loop itself
        // is the first construct to name, independent of whether the
        // enclosing function's own `-> Annotation` is readable (a bare
        // `-> float` leaves `return_refinement` `None`, which is
        // `walk_return`'s own signal to skip JUDGING a value — it must
        // not also silence THIS body's own record of why that value is
        // unreadable in the first place). `widened_names` is sorted
        // (`stabilized_join`'s own doc), so the first entry is a stable,
        // reproducible choice across runs.
        if let Some(first_widened) = widened_names.first() {
            record_blocker(blocked, stmt.range(), loop_accumulation_did_not_stabilize(first_widened), out);
        }
        // RETURN-THROUGH-LOOP CHANNEL: a value SOME concrete iteration
        // returned judges against the enclosing function's own
        // `-> Annotation`, exactly as `walk_return` judges a
        // straight-line `return` — the same Fire/Silent/Undetermined
        // law, anchored at the range loops.rs carried (the returned
        // expression's own range, or the bare `return` statement's own
        // range). A BARE return (`value: None`, loops.rs's own
        // `walk_return`-matching convention) judges nothing, same as a
        // straight-line bare `return`. Additive: the walk still proceeds
        // to the `else`/dead-else law below on the SAME environment,
        // since loops.rs never tries to prove the statement after the
        // loop unreachable (see `LoopAnswer`'s own doc).
        if let Some((Some(value), range)) = returned {
            if let Some(declared) = return_refinement {
                match judge(&value, declared, context.kernel) {
                    Verdict::Fire(message) => out.push(Finding { range, code: "RTS7001", message }),
                    Verdict::Silent => {}
                    Verdict::Undetermined(sentence) => {
                        record_blocker(blocked, range, sentence, out);
                    }
                }
            }
        }
        let orelse: &[Stmt] = match stmt {
            Stmt::For(for_stmt) => for_stmt.orelse.as_slice(),
            Stmt::While(while_stmt) => while_stmt.orelse.as_slice(),
            _ => &[],
        };
        if orelse.is_empty() {
            return false;
        }
        if !else_runs {
            out.push(Finding {
                range: orelse[0].range(),
                code: "RTS7001",
                message: "this else arm provably never runs: the loop above always breaks".to_owned(),
            });
            return false;
        }
        let mut orelse_provably_unbound: HashSet<String> = HashSet::new();
        for orelse_stmt in orelse {
            walk_statement(
                orelse_stmt,
                return_refinement,
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                &mut orelse_provably_unbound,
                blocked,
                out,
            );
        }
        return false;
    }
    record_blocker(
        blocked,
        stmt.range(),
        format!("{} is not yet walked", statement_kind_name(stmt)),
        out,
    );
    forget_names_bound_by_stmt(stmt, environment);
    forget_mutated_receivers_in_stmt(stmt, environment);
    false
}

/// `match subject: case ... case ...` (compound_stmts.rst, "The `match`
/// statement"): `match_arms::match_taken_environment` decides, for a
/// KNOWN subject, which single arm CPython would take. `Some((index,
/// arm_env))` adopts that arm's environment (already carrying any
/// capture-pattern bindings match_arms.rs made) and walks ONLY that
/// case's body statements in order — the other arms are not taken and
/// are never walked, matching CPython's own first-match semantics.
///
/// MATCH JOIN FALLBACK (serves b-body-expressions.py:889/900 — a class
/// pattern like `case int() as n:`, which `match_arms.rs` cannot decide
/// TAKEN/NOT-TAKEN this wave, `Pattern::MatchClass` being Undecidable
/// there regardless of the subject): when `match_taken_environment`
/// answers `None`, this function no longer records a blocker outright.
/// Instead it walks EVERY case on its OWN fork of the incoming
/// environment — `match_arms::pattern_bound_captures` names what a
/// pattern captures SYNTACTICALLY (a question `pattern_outcome` never
/// has to answer, unlike TAKEN/NOT-TAKEN) AND binds each captured name
/// to the most specific value it can prove (an exact literal proof, a
/// sequence element/mapping value/class field read off a KNOWN
/// container subject, or — when none of those apply — `unknown()`,
/// never a guess) — a guarded case still walks under the same fork (the
/// guard's own truth is not decided either; over-approximating past it
/// is sound, never wrong). Every surviving fork (one whose last
/// statement is not `return`/`raise`, `arm_terminates`) joins exactly
/// as `walk_if` joins its own arms. The blocker is kept ONLY when
/// `pattern_bound_captures` itself cannot even name a case's own
/// captures (a `MatchSequence`/`MatchMapping`/`MatchClass` shape past
/// its own flat bare-capture scope, or a `MatchClass` with POSITIONAL
/// sub-patterns — `match_arms.rs`'s own doc names exactly which shapes
/// those are) — that one case, and every case after it (CPython could
/// have reached any of them), forgets its own bound names instead of
/// joining, and the match statement's own blocker still records once.
pub(super) fn walk_match(
    match_stmt: &StmtMatch,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let subject_value = evaluate_expression(match_stmt.subject.as_ref(), environment, context.kernel);
    // The subject's own bare name, when it has one, so a split arm can
    // rebind it to the arm's intersection — the same name the fallback
    // path below already reads off `Expr::Name`.
    let subject_name = match match_stmt.subject.as_ref() {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    };
    {
        // The arm-body walker match_taken_environment calls per decided
        // arm — the identical per-arm walk `walk_if` runs for a fork:
        // a fresh provably-unbound set, every statement through
        // walk_statement, and the arm's own termination read off its
        // last statement. This walker can always walk, so it never
        // answers `None` (the decline arm is for callers whose own
        // interpreters can genuinely refuse, `summaries.rs`).
        let mut walk_arm = |body: &[Stmt], arm_environment: &mut Environment| -> Option<bool> {
            let mut arm_provably_unbound: HashSet<String> = HashSet::new();
            for stmt in body {
                walk_statement(
                    stmt,
                    return_refinement,
                    yield_refinement,
                    context,
                    arm_environment,
                    aug_assign_refinements,
                    &mut arm_provably_unbound,
                    blocked,
                    out,
                );
            }
            Some(!arm_terminates(body))
        };
        if let Some((post_environment, _falls_through)) = match_taken_environment(
            &subject_value,
            subject_name,
            &match_stmt.cases,
            environment,
            context.kernel,
            &mut walk_arm,
        ) {
            *environment = post_environment;
            return;
        }
    }

    let mut surviving: Vec<Environment> = Vec::new();
    let mut every_case_nameable = true;
    // The literals every GUARDLESS earlier arm's pattern proved — a
    // later arm runs only when every earlier pattern failed
    // (compound_stmts.rst, the match statement), so its subject cannot
    // hold any of these. A guarded literal arm sheds nothing: it can
    // fail on its guard with the literal still live.
    let mut shed_literals: Vec<f64> = Vec::new();
    for case in &match_stmt.cases {
        if !every_case_nameable {
            // an earlier case's own captures could not be named — CPython
            // might have reached THIS case too (or any later one), so its
            // bound names are equally unknown from here on
            forget_names_bound_in_body(&case.body, environment);
            forget_mutated_receivers_in_body(&case.body, environment);
            continue;
        }
        let Some(bound_captures) =
            match_arms::pattern_bound_captures(&case.pattern, &subject_value, environment, context.kernel)
        else {
            every_case_nameable = false;
            record_blocker(
                blocked,
                case.pattern.range(),
                "this match arm's own pattern does not yet name its captures".to_owned(),
                out,
            );
            forget_names_bound_in_body(&case.body, environment);
            forget_mutated_receivers_in_body(&case.body, environment);
            continue;
        };
        let mut arm_env = environment.fork();
        // SHED-LITERAL RESIDUAL: this arm runs only when every earlier
        // guardless literal arm failed, so a Set-kind subject's integer
        // window trims those literals off its EDGES here — `case 150:`
        // over `[150, 151]` leaves the wildcard arm `[151, 151]`. An
        // interior literal is a hole one window cannot state and trims
        // nothing (the over-approximating sound side — the same law the
        // C++ adapter's flowingWindow default arm keeps). A literal
        // arm's own pattern-proved narrowing below overwrites this
        // residual with its tighter intersection.
        if let Some(subject_name) = subject_name {
            if !shed_literals.is_empty() {
                if let Some(current) = arm_env.read(subject_name).cloned() {
                    if let Some(trimmed) = integer_window_minus_edge_literals(&current, &shed_literals) {
                        arm_env.bind(subject_name, trimmed);
                    }
                }
            }
        }
        // PATTERN-PROVED NARROWING: every capture `pattern_bound_
        // captures` names binds to what its own position/key/field (or,
        // for a literal/singleton/or/as pattern with none of those, the
        // pattern's own PROVED value — `match_arms::pattern_bound_
        // captures`'s own doc) states about a taken arm, tighter than
        // the coarse pre-match claim `subject_value` carries alone. The
        // SUBJECT NAME ITSELF (when the match subject is a bare Name,
        // e.g. `match pick:`) rebinds to `pattern_proved_value`'s own
        // whole-pattern proof, since a body that reads the subject name
        // directly inside a literal arm (`case 18 | 21 | 40: return
        // pick`) is reading exactly what the pattern just proved. A
        // pattern proving nothing whole (a bare capture, a sequence/
        // mapping/class pattern) leaves the subject name unchanged —
        // the honest, unnarrowed subject claim.
        for (name, value) in &bound_captures {
            arm_env.bind(name, value.clone());
        }
        let narrowed = match_arms::pattern_proved_value(&case.pattern, &arm_env, context.kernel);
        if let Expr::Name(subject_name) = match_stmt.subject.as_ref() {
            if let Some(narrowed) = &narrowed {
                arm_env.bind(subject_name.id.as_str(), narrowed.clone());
            }
        }
        // GUARD NARROWING: `case x if x >= 0 and x <= 120:` — the guard
        // is a boolean expression over names the pattern already bound
        // (captures, or the subject itself above); it narrows exactly
        // the way `walk_if`'s own test narrows an `if` arm
        // (`narrowing::assume`, mission point: "a guard narrows"). The
        // guard's own TRUTH is not decided here (unlike `arm_outcome`'s
        // TAKEN/NOT-TAKEN reading) — this walk already treats every
        // case as reachable in the join fallback, so the guard is
        // assumed TRUE for the body it gates, sound because CPython
        // only runs this body when the guard is in fact true.
        if let Some(guard) = case.guard.as_deref() {
            arm_env = assume(guard, arm_env, context.kernel, true);
        }
        let mut arm_provably_unbound: HashSet<String> = HashSet::new();
        for stmt in &case.body {
            walk_statement(
                stmt,
                return_refinement,
                yield_refinement,
                context,
                &mut arm_env,
                aug_assign_refinements,
                &mut arm_provably_unbound,
                blocked,
                out,
            );
        }
        if !arm_terminates(&case.body) {
            surviving.push(arm_env);
        }
        if case.guard.is_none() {
            if let Some(proved) = match_arms::pattern_proved_value(&case.pattern, environment, context.kernel) {
                if proved.kind == Kind::Values {
                    shed_literals.extend(proved.values.iter().copied());
                }
            }
        }
    }

    if !every_case_nameable {
        return;
    }
    *environment = match surviving.len() {
        0 => environment.fork(),
        1 => surviving.into_iter().next().unwrap(),
        _ => {
            let mut joined = surviving.remove(0);
            for arm in surviving {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    };
}

/// A Set-kind binding's integer window with `shed` literals trimmed off
/// its EDGES — the residual a match's later arms see once earlier
/// guardless literal arms are behind them. A literal equal to the
/// window's own floor (`atLeast`) raises it by one; one equal to the
/// ceiling (`atMost`) lowers it by one, to a fixpoint. An interior
/// literal is a hole one window cannot state and trims nothing — the
/// over-approximating sound side. `None` when nothing trimmed, when the
/// binding is not a Set of exactly atLeast/atMost/integer forms this
/// reader states, or when trimming empties the window (the arm is then
/// unreachable — no residual claim is owed, and the unnarrowed subject
/// stands).
pub(super) fn integer_window_minus_edge_literals(current: &AbstractValue, shed: &[f64]) -> Option<AbstractValue> {
    use refined_sets::refinement_forms::Form;
    if current.kind != Kind::Set {
        return None;
    }
    if !current.set.forms.iter().any(|form| form.form == Form::Integer) {
        return None;
    }
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &current.set.forms {
        match form.form {
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost => hi = Some(form.a),
            Form::Integer => {}
            _ => return None,
        }
    }
    let (Some(mut lo), Some(mut hi)) = (lo, hi) else {
        return None;
    };
    let mut changed = true;
    let mut trimmed_any = false;
    while changed && lo <= hi {
        changed = false;
        for literal in shed {
            if *literal == lo {
                lo += 1.0;
                changed = true;
                trimmed_any = true;
            }
            if *literal == hi {
                hi -= 1.0;
                changed = true;
                trimmed_any = true;
            }
        }
    }
    if !trimmed_any || lo > hi {
        return None;
    }
    let mut narrowed = current.clone();
    for form in &mut narrowed.set.forms {
        match form.form {
            Form::AtLeast => form.a = lo,
            Form::AtMost => form.a = hi,
            _ => {}
        }
    }
    Some(narrowed)
}

/// `with EXPRESSION as TARGET: SUITE` (compound_stmts.rst, "The `with`
/// statement"): step 5 of the with-statement's own execution order binds
/// TARGET to `__enter__`'s (or, for `async with` — `with_stmt.is_async`,
/// the collapsed sync/async node ruff's own generated.rs doc states —
/// `__aenter__`'s) RETURN VALUE. When the context expression reads as a
/// known INSTANCE (`Kind::Object` with a non-empty `source` naming a
/// `ClassModel` — `instances::judge_construction`'s own tag, the same
/// shape `instance_method_call_result` already reads for a statement-side
/// method call) and that class declares the matching enter method,
/// `instances::method_call_result` interprets its body the same way any
/// other zero-argument method call on a known instance does: `Some`
/// REBINDS the receiver (when it is a bare Name) to the returned working
/// instance and binds TARGET to the method's own return value —
/// `with device() as handle:` (`_Device.__enter__` returns `self`) is
/// exactly this shape. `None` (an unmodeled receiver, a class with no
/// matching enter method, a method body/parameter shape outside the
/// restricted interpreter) forgets TARGET instead — the honest answer
/// when this walk cannot know `__enter__`'s own return value, unchanged
/// from before this law. The body then walks inline, on the SAME
/// environment, with no blocker for the with statement itself.
pub(super) fn walk_with(
    with_stmt: &StmtWith,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    for item in &with_stmt.items {
        let receiver = evaluate_expression(&item.context_expr, environment, context.kernel);
        let Some(target) = item.optional_vars.as_deref() else {
            continue;
        };
        match enter_method_result(&receiver, with_stmt.is_async, &item.context_expr, context, environment) {
            Some(entered) => bind_with_target(target, entered, environment),
            None => forget_target_names(target, environment),
        }
    }
    let mut with_provably_unbound: HashSet<String> = HashSet::new();
    // FOREIGN EDGE: a temp-file leg (or any other recognized crossing
    // call) nested inside this with-block's own body is offered to
    // `serve_foreign_edge_at` the same way the top-level body loop offers
    // it — `with_stmt.body` is itself a statement list, so nesting one
    // level inside another `with` (`level_via_nested_tempdir`) no longer
    // removes it from every scan `foreign_edge_at` ever runs.
    let mut foreign_edge_overrides: HashMap<usize, Vec<(TextRange, AbstractValue)>> = HashMap::new();
    for (position, stmt) in with_stmt.body.iter().enumerate() {
        serve_foreign_edge_at(&with_stmt.body, position, environment, context, blocked, out, &mut foreign_edge_overrides);
        if let Some(published) = foreign_edge_overrides.get(&position) {
            environment.set_evaluated_node(published.clone());
        }
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
            context,
            environment,
            aug_assign_refinements,
            &mut with_provably_unbound,
            blocked,
            out,
        );
        environment.set_evaluated_node(Vec::new());
        foreign_edge_overrides.remove(&position);
    }
}

/// `EXPRESSION.__enter__()` (or `.__aenter__()` when `is_async`) on a
/// known instance — the same receiver-tag/method-lookup/interpretation
/// chain `instance_method_call_result` already runs for a statement-side
/// `name.method(...)` call, reused here with a FIXED, zero-argument
/// method name instead of reading one off the call syntax (`with`'s own
/// context expression is not itself a method call — `device()` names a
/// constructor or a same-module def, never `.__enter__` directly; the
/// PROTOCOL supplies that name, per compound_stmts.rst's with-statement
/// execution steps). `receiver_expr` is threaded through so a bare-Name
/// context expression (`with handle_holder as handle:`, the receiver
/// already a local) can have its OWN binding rebound to the entered
/// working instance, the same "the receiver survives a self-mutating
/// method call" law `instance_method_call_result` keeps; a context
/// expression that is not itself a bare Name (`with device() as handle:`)
/// has no environment slot to rebind, so only the returned value is
/// reported. `None` for a non-instance receiver, a class with no matching
/// enter method, or a method body/parameter shape `method_call_result`
/// itself declines.
pub(super) fn enter_method_result(
    receiver: &AbstractValue,
    is_async: bool,
    receiver_expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
) -> Option<AbstractValue> {
    if receiver.kind != Kind::Object || receiver.source.is_empty() {
        return None;
    }
    let model = context.classes.get(receiver.source.as_str())?;
    let method_name = if is_async { "__aenter__" } else { "__enter__" };
    let method = instances::method_def_of(model, method_name)?;
    let (new_instance, result) = instances::method_call_result(
        receiver,
        model,
        method,
        &[],
        Some(&context.functions),
        Some(&context.classes),
        Some(&context.datetime_imports),
        context.kernel,
        environment.call_depth(),
    )?;
    if let Expr::Name(receiver_name) = receiver_expr {
        environment.bind(receiver_name.id.as_str(), new_instance);
    }
    Some(result)
}

/// Binds a `with ... as TARGET:` target to `__enter__`'s own return
/// value: a bare Name binds directly. Every other target shape (a
/// tuple/list/starred unpack — `with cm() as (a, b):`, out of this
/// corpus's own rows) FORGETS every name it names instead — this
/// function has no positional unpack rule for `__enter__`'s single
/// return value the way `bind_for_target`'s tuple arm has for a
/// known-arity iterate, so the honest answer is the same "no fact
/// survives an unmodeled target" rule `forget_target_names` already
/// states everywhere else in this file, never a silent stale-value
/// pass-through.
pub(super) fn bind_with_target(target: &Expr, value: AbstractValue, environment: &mut Environment) {
    match target {
        Expr::Name(name) => environment.bind(name.id.as_str(), value),
        _ => forget_target_names(target, environment),
    }
}

/// `try: BODY (except ... )+ [else: BODY] [finally: BODY]`
/// (compound_stmts.rst, "The `try` statement"): the try body walks on
/// its own fork; each handler starts from a fork of the INCOMING
/// environment with every name the try body binds forgotten — an
/// exception may interrupt the body at any point, so no write in it is
/// guaranteed to have happened, and any write that DID happen
/// invalidates the pre-try fact either way, so forgetting is the one
/// sound answer for both. A handler's `as`-name (if present) is bound by
/// forgetting it (exception objects are not modeled) and forgotten AGAIN
/// after the handler body — "when an exception has been assigned using
/// `as target`, it is cleared at the end of the except clause... as if
/// `except E as N: foo` was translated to... `finally: del N`". `orelse`
/// runs only once the try body completes without raising, continuing
/// that same fork ("the else clause is executed if the control flow
/// leaves the try suite, no exception was raised"). The post-try
/// environment joins every surviving path (the try+orelse fork and each
/// handler fork, `arm_terminates` deciding survival exactly as `if`
/// does); zero survivors keeps the incoming environment, the same
/// convention `walk_if` uses. `finalbody` then always runs, after the
/// join, on the joined environment — "the finally clause is executed...
/// if the try clause is executed, including any except and else
/// clauses."
///
/// Returns whether the try statement itself provably never falls
/// through: zero surviving arms means every path either raises past an
/// uncaught exception or terminates inside its own handler — the exact
/// fact `surviving.is_empty()` already computes for the join below,
/// surfaced so the caller's own body loop can stop walking statements
/// that follow, the same way it already stops after a bare `return`/
/// `raise` (`arm_terminates`). A `finally` clause still always runs
/// first regardless — CPython runs `finally` even when nothing survives
/// the try/except — so this only governs statements AFTER the whole
/// try/except/else/finally construct, never `finalbody` itself.
pub(super) fn walk_try(
    try_stmt: &StmtTry,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> bool {
    let mut surviving: Vec<Environment> = Vec::new();

    let mut try_env = environment.fork();
    let mut try_provably_unbound: HashSet<String> = HashSet::new();
    let try_body_findings_before = out.len();
    for stmt in &try_stmt.body {
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
            context,
            &mut try_env,
            aug_assign_refinements,
            &mut try_provably_unbound,
            blocked,
            out,
        );
    }
    // orelse ("the try statement", `else` clause) runs only once the try
    // body completes without raising — walking it here is this analysis's
    // static approximation of that runtime condition, same as every other
    // body this walk always visits regardless of the branch actually
    // taken at runtime. The combined path's survival is decided by the
    // LAST body actually executed along it: orelse's own last statement
    // when orelse is present, otherwise the try body's.
    let orelse_findings_before = out.len();
    for stmt in &try_stmt.orelse {
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
            context,
            &mut try_env,
            aug_assign_refinements,
            &mut try_provably_unbound,
            blocked,
            out,
        );
    }
    let (try_path_terminal_body, terminal_findings_before) = if try_stmt.orelse.is_empty() {
        (try_stmt.body.as_slice(), try_body_findings_before)
    } else {
        (try_stmt.orelse.as_slice(), orelse_findings_before)
    };
    if !arm_terminates_or_provably_raises(try_path_terminal_body, out, terminal_findings_before) {
        surviving.push(try_env);
    }

    for handler in &try_stmt.handlers {
        let ExceptHandler::ExceptHandler(handler) = handler;
        let mut handler_env = environment.fork();
        for stmt in &try_stmt.body {
            forget_names_bound_by_stmt(stmt, &mut handler_env);
        }
        // HANDLER AS-NAME: `except Exception as error:` binds `error` to
        // a caught-exception value at handler entry — not a forget — so
        // a read inside the handler body (e.g. `try_except_binding`'s
        // `age = error`) has something to judge rather than reading
        // Undetermined. `caught_exception_value` gives the MOST SPECIFIC
        // value this walk can prove: a tagged, `args`-carrying (and,
        // where a sole matching `raise ... from` names it, `__cause__`-
        // carrying) exception when the try body's own raise is
        // findable, else the plain opaque marker every other unmodeled
        // instance already reads as Unknown.
        if let Some(name) = handler.name.as_ref() {
            let caught = caught_exception_value(handler, &try_stmt.body, environment, context);
            handler_env.bind(name.id.as_str(), caught);
        }
        let mut handler_provably_unbound: HashSet<String> = HashSet::new();
        let handler_findings_before = out.len();
        for stmt in &handler.body {
            walk_statement(
                stmt,
                return_refinement,
                yield_refinement,
                context,
                &mut handler_env,
                aug_assign_refinements,
                &mut handler_provably_unbound,
                blocked,
                out,
            );
        }
        if let Some(name) = handler.name.as_ref() {
            handler_env.forget(name.id.as_str());
        }
        if !arm_terminates_or_provably_raises(&handler.body, out, handler_findings_before) {
            surviving.push(handler_env);
        }
    }

    let nothing_survives = surviving.is_empty();
    *environment = match surviving.len() {
        0 => environment.fork(),
        1 => surviving.into_iter().next().unwrap(),
        _ => {
            let mut joined = surviving.remove(0);
            for arm in surviving {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    };

    let mut finally_provably_unbound: HashSet<String> = HashSet::new();
    for stmt in &try_stmt.finalbody {
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
            context,
            environment,
            aug_assign_refinements,
            &mut finally_provably_unbound,
            blocked,
            out,
        );
    }
    nothing_survives
}

/// The value one `except <type> as <name>:` handler's own `<name>`
/// binds — the most specific value this walk can prove about the
/// exception CPython would actually deliver there. `handler`'s own
/// `type_` (a bare `Name`, the only shape this reader matches — a
/// tuple-of-types `except (A, B):` or an attribute `except mod.Err:`
/// falls through to the fieldless answer) names the exception CLASS the
/// search below looks for: `find_raise_from`, walking `try_body` in
/// source order (recursing into a nested `Stmt::Try`'s own body, since
/// `raise ... from` can sit arbitrarily deep — j-stdlib-surfaces.py's
/// own `exception_cause` nests one level), returns the SOLE `Stmt::Raise`
/// whose `exc` is a call to that exact class name, or `None` when there
/// is none or more than one (an ambiguous try body proves no ONE raise
/// reaches this handler over another). A found raise's `exc` evaluates
/// through the ordinary `evaluate_expression` path — already routed
/// through `exception_construction_value` for a recognized builtin
/// exception constructor (`expressions.rs`'s own `is_builtin_exception_
/// constructor` gate) — and its own `cause`, when present, resolves
/// through `resolve_cause_name`: a bare Name matching an ENCLOSING
/// handler's own `as`-name recurses into THAT handler's nested try via
/// this same function (j-stdlib-surfaces.py's `inner` naming the INNER
/// try's own caught `ValueError`); any other cause shape (a fresh
/// construction, an unrecognized name) evaluates plainly against
/// `environment`. No raise found, no cause resolvable, or `exc` itself
/// not a recognized exception constructor: `fieldless_exception_value`
/// — the honest "an exception, but this walk cannot prove its fields"
/// answer, never the fully opaque `opaque_value` (which cannot even be
/// recognized as an exception downstream).
pub(super) fn caught_exception_value(
    handler: &ExceptHandlerExceptHandler,
    try_body: &[Stmt],
    environment: &Environment,
    context: &WalkContext,
) -> AbstractValue {
    let Some(Expr::Name(caught_type)) = handler.type_.as_deref() else {
        return fieldless_exception_value();
    };
    let Some(raise) = find_raise_from(try_body, caught_type.id.as_str()) else {
        return fieldless_exception_value();
    };
    let Some(exc) = raise.exc.as_deref() else {
        return fieldless_exception_value();
    };
    let mut value = evaluate_expression(exc, environment, context.kernel);
    if value.kind != Kind::Object || value.source != "exception" {
        return fieldless_exception_value();
    }
    if let Some(cause) = raise.cause.as_deref() {
        let cause_value = resolve_cause_name(cause, try_body, environment, context);
        value.keys.push(ObjectKey {
            name: "__cause__".to_owned(),
            numeric: false,
            value: cause_value,
        });
    }
    value
}

/// The sole `raise <exc> from <cause>` (or bare `raise <exc>`) whose
/// `exc` is a call to `class_name`, searched in source order through
/// `body`'s own statements — recursing into a nested `Stmt::Try`'s
/// `body`/`handlers`/`orelse`/`finalbody` (every place CPython itself
/// could reach a raise from within a try construct) and an `if`
/// statement's own arms (the same shape `walk_if` itself walks
/// unconditionally). `None` when no raise names that class, or more
/// than one does — this function proves nothing about WHICH raise
/// reaches the handler when the body's own control flow leaves that
/// ambiguous, so it answers only the unambiguous one-match case.
pub(super) fn find_raise_from<'a>(body: &'a [Stmt], class_name: &str) -> Option<&'a StmtRaise> {
    let mut found: Option<&'a StmtRaise> = None;
    for stmt in body {
        for raise in raises_in_stmt(stmt) {
            let Some(Expr::Call(call)) = raise.exc.as_deref() else {
                continue;
            };
            let Expr::Name(callee) = call.func.as_ref() else {
                continue;
            };
            if callee.id.as_str() != class_name {
                continue;
            }
            if found.is_some() {
                // more than one raise names this class — ambiguous
                return None;
            }
            found = Some(raise);
        }
    }
    found
}

/// Every `Stmt::Raise` reachable inside one statement, recursing into
/// the compound shapes a raise can sit under — `Stmt::Try` (its own
/// `body`/`handlers`/`orelse`/`finalbody`, in that order) and `Stmt::If`
/// (its own `body` plus every `elif`/`else` clause's own body) — the two
/// shapes this corpus's own rows nest a raise inside. Any other compound
/// statement (a `for`/`while`/`with`/`match`) is not walked into: this
/// reader is scoped to the try/if nesting `caught_exception_value`'s own
/// rows use, not a general statement-tree walk.
pub(super) fn raises_in_stmt(stmt: &Stmt) -> Vec<&StmtRaise> {
    match stmt {
        Stmt::Raise(raise) => vec![raise],
        Stmt::Try(try_stmt) => {
            let mut found = Vec::new();
            for inner in &try_stmt.body {
                found.extend(raises_in_stmt(inner));
            }
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    found.extend(raises_in_stmt(inner));
                }
            }
            for inner in &try_stmt.orelse {
                found.extend(raises_in_stmt(inner));
            }
            for inner in &try_stmt.finalbody {
                found.extend(raises_in_stmt(inner));
            }
            found
        }
        Stmt::If(if_stmt) => {
            let mut found = Vec::new();
            for inner in &if_stmt.body {
                found.extend(raises_in_stmt(inner));
            }
            for clause in &if_stmt.elif_else_clauses {
                for inner in &clause.body {
                    found.extend(raises_in_stmt(inner));
                }
            }
            found
        }
        _ => Vec::new(),
    }
}

/// The value a `raise ... from <cause>` expression's own `cause` names —
/// `caught_exception_value`'s own cause-resolution step. A bare Name
/// matching an ENCLOSING `except ... as <name>:` handler nested inside
/// `try_body` recurses into that handler's own nested try (its `type_`
/// naming which class to search for, `caught_exception_value` again),
/// so a chain of `raise ... from` across nested try/except blocks
/// resolves end to end (j-stdlib-surfaces.py's `inner` naming the INNER
/// try's own caught `ValueError`). Any other cause shape (a fresh
/// construction, a name this search cannot trace to an enclosing
/// handler) evaluates plainly through `evaluate_expression` — sound
/// either way, since an unrecognized cause still reads SOME value, just
/// not necessarily the exact tagged exception shape.
pub(super) fn resolve_cause_name(cause: &Expr, try_body: &[Stmt], environment: &Environment, context: &WalkContext) -> AbstractValue {
    if let Expr::Name(cause_name) = cause {
        if let Some((nested_try, nested_handler)) = enclosing_handler_named(try_body, cause_name.id.as_str()) {
            return caught_exception_value(nested_handler, &nested_try.body, environment, context);
        }
    }
    evaluate_expression(cause, environment, context.kernel)
}

/// The nested `Stmt::Try` and its own `except ... as <name>:` handler,
/// searched inside `body` — `resolve_cause_name`'s own lookup for a
/// `raise ... from <name>` whose `name` was bound by an ENCLOSING
/// handler rather than by ordinary code. Recurses into every nested
/// `Stmt::Try`'s own `body` (the same one-level-deep nesting
/// `find_raise_from`'s search already walks) so a chain of nested
/// try/except resolves at any depth. `None` when no enclosing handler
/// binds that name.
pub(super) fn enclosing_handler_named<'a>(body: &'a [Stmt], name: &str) -> Option<(&'a StmtTry, &'a ExceptHandlerExceptHandler)> {
    for stmt in body {
        let Stmt::Try(try_stmt) = stmt else {
            continue;
        };
        for handler in &try_stmt.handlers {
            let ExceptHandler::ExceptHandler(handler) = handler;
            if handler.name.as_ref().is_some_and(|handler_name| handler_name.id.as_str() == name) {
                return Some((try_stmt, handler));
            }
        }
        if let Some(found) = enclosing_handler_named(&try_stmt.body, name) {
            return Some(found);
        }
    }
    None
}
