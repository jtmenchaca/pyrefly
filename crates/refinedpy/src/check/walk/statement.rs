//! The per-statement dispatcher: one statement, walked by syntactic
//! form. Determinable forms judge or bind/forget as they can; the first
//! form this walk cannot handle names itself as the body's blocker and
//! the walk moves on conservatively.

use std::collections::{HashMap, HashSet};

use ruff_python_ast::{Expr, Stmt};
use ruff_text_size::{Ranged, TextRange};

use crate::diagnostic_sentences::empty_set;
use crate::env::Environment;
use crate::narrowing::assume;
use crate::typereading::DeclaredRefinement;

use crate::check::{
    bind_or_forget_imported_name, declared_set_is_empty, instance_method_call_result, sink_value,
    walk_aug_assign, walk_ann_assign, walk_assign, walk_class_def, walk_del_subscript_target,
    walk_function_def, walk_if, walk_loop, walk_match, walk_mutating_call_statement, walk_return,
    walk_try, walk_with, walk_yield, Finding, WalkContext,
};

use super::{bind_walrus_targets, forget_target_names};

/// Record this body's one blocker, if it has not already recorded one.
/// Every later call in the same body is a no-op — the FIRST blocker
/// wins, and the walk still keeps going conservatively afterward.
///
/// The blocker also opens its OWN reader span over the blocked range
/// and records the decline into it, so a trace of a position this
/// blocker owns names the construct rather than carrying an empty leaf.
/// Without the span there is nothing for `record_decline` to attach to
/// — a decline with no open reader span is refused (`trace::collector`'s
/// own root rule) — and the blocker's own words never reach the trace.
/// The span is opened and closed here, around the record, because the
/// constructs that reach this function are exactly the ones with no
/// dispatch of their own to nest under: a statement form the walk does
/// not read, a loop whose accumulation never settled, a judgment that
/// came back Undetermined.
pub(in crate::check) fn record_blocker(blocked: &mut bool, range: TextRange, sentence: String, out: &mut Vec<Finding>) {
    if *blocked {
        return;
    }
    *blocked = true;
    if crate::trace::is_tracing() {
        let _span = crate::trace::span_scope(
            "blocked_construct",
            usize::from(range.start()),
            usize::from(range.end()),
        );
        crate::trace::record_decline(
            &sentence,
            Some((usize::from(range.start()), usize::from(range.end()))),
            None,
        );
    }
    out.push(Finding {
        range,
        code: "RTS7002",
        message: sentence,
    });
}

/// One statement, dispatched by syntactic form. Determinable forms
/// judge or bind/forget as they can; the first form this walk cannot
/// handle names itself as the body's blocker and the walk moves on
/// conservatively. `aug_assign_refinements` is this body's own
/// AnnAssign-recorded refinement table (`x: Age = …` records `x ↦ Age`
/// here) — read back when a later `x += …` needs a set to judge
/// against. `provably_unbound` is this body's own PROVABLY-UNBOUND-READS
/// tracking set (`walk_body`'s own doc): every straight-line statement
/// form below that can OBSERVE a name being bound removes it here, and
/// every form whose execution could bind a name on SOME path this walk
/// does not track directly (a branch, a loop, a match, a with, a try, or
/// this body's own first unwalkable construct) clears the set wholesale
/// before dispatching — the conservative "any branch/loop/blocker
/// between declaration and read → no fire" rule the mission states.
///
/// Returns whether this statement provably never falls through to
/// whatever follows it, and — when it does not — whether the statement
/// that follows is DEAD CODE the caller owes a report for. This is not
/// a general reachability signal; it carries only the two facts the
/// walk already computes: a `try` whose own arms all terminate
/// (`walk_try`'s return, `FallsThrough::No`), and an `if` whose test
/// proved true and whose selected arm terminates (`walk_if`'s return,
/// `FallsThrough::NoAndFollowingIsDead`). Every other form answers
/// `FallsThrough::Yes`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(in crate::check) enum FallsThrough {
    /// Ordinary: whatever follows this statement can run.
    Yes,
    /// Nothing reaches past this statement, and that is the ordinary
    /// shape of the construct — a `return`/`raise`, or a `try` whose
    /// every arm terminates. The caller stops walking and reports
    /// nothing: code after a `return` at a body's end is not a defect
    /// this walk speaks to.
    No,
    /// Nothing reaches past this statement BECAUSE a condition proved
    /// true and the arm it selected terminates — so whatever follows is
    /// provably unreachable code. The caller reports that statement and
    /// stops walking.
    NoAndFollowingIsDead,
}

#[allow(clippy::too_many_arguments)]
pub(in crate::check) fn walk_statement(
    stmt: &Stmt,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    provably_unbound: &mut HashSet<String>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> FallsThrough {
    match stmt {
        Stmt::AnnAssign(assign) => {
            walk_ann_assign(
                assign,
                context,
                environment,
                aug_assign_refinements,
                provably_unbound,
                blocked,
                out,
            );
        }
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                super::forget_target_from_provably_unbound(target, provably_unbound);
            }
            walk_assign(assign, context, environment, aug_assign_refinements, out);
        }
        Stmt::AugAssign(assign) => {
            super::forget_target_from_provably_unbound(assign.target.as_ref(), provably_unbound);
            walk_aug_assign(assign, context, environment, aug_assign_refinements, blocked, out);
        }
        // `yield`/`yield from` — tried before the ordinary Stmt::Expr
        // handling below: a yield is not a call and carries no receiver,
        // method, or mutation shape any of those channels recognize, so
        // routing it there would only ever reach `sink_value`'s plain
        // `evaluate_expression` fallback with no yield-position judging
        // at all. `walk_yield` owns judging it against the enclosing
        // generator's own declared yield set.
        Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::Yield(_) | Expr::YieldFrom(_)) => {
            walk_yield(
                expr_stmt.value.as_ref(),
                yield_refinement,
                context,
                aug_assign_refinements,
                environment,
                blocked,
                out,
            );
        }
        Stmt::Expr(expr_stmt) => {
            bind_walrus_targets(expr_stmt.value.as_ref(), context, aug_assign_refinements, environment, out);
            // STATEMENT-SIDE METHOD CALLS tries first: a bare-Name receiver
            // bound to a known instance rebinds through
            // instances::method_call_result, discarding the returned
            // value (an expression-statement's own value is never read,
            // matching sink_value's other callers' discard convention at
            // Stmt::Expr). Declining (None) falls to the collection
            // `mutated_receiver` path, then to `sink_value` — which owns
            // the CALLEE-EFFECTS CHANNEL itself (`apply_call_effects`, a
            // bare-Name same-module call whose body writes an enclosing
            // name) alongside every other call-shaped check `sink_value`
            // makes (provable/possible raises, construction, callable-
            // variable calls, manifest-argument judging, and same-module
            // call-argument judging against the callee's declared
            // parameters). `apply_call_effects` used to be tried HERE as
            // its own gate, `Option`-short-circuiting `sink_value`
            // entirely on `Some(())` — which that function's own doc
            // states it answers for EVERY same-module def call that
            // resolves, whether or not it reports any actual effect ("a
            // same-module def with an empty effect list still matched").
            // That meant `sink_value` — and every fire only it can
            // produce, `same_module_call_argument_fires` chief among
            // them — never ran for the ordinary, effect-free case, which
            // is most same-module calls. `sink_value` is the one place
            // `apply_call_effects` runs now, so every check it makes
            // applies uniformly regardless of whether the callee also
            // happens to write an enclosing name.
            if instance_method_call_result(expr_stmt.value.as_ref(), context, environment).is_none()
                && !walk_mutating_call_statement(
                    expr_stmt.value.as_ref(),
                    context,
                    environment,
                    aug_assign_refinements,
                    out,
                )
            {
                sink_value(expr_stmt.value.as_ref(), context, environment, aug_assign_refinements, out);
            }
        }
        Stmt::Pass(_) => {}
        Stmt::Return(ret) => {
            walk_return(
                ret,
                return_refinement,
                context,
                aug_assign_refinements,
                provably_unbound,
                environment,
                blocked,
                out,
            );
        }
        Stmt::FunctionDef(def) => {
            walk_function_def(def, context, out);
        }
        Stmt::ClassDef(def) => {
            walk_class_def(def, environment, context, out);
        }
        // `del a, b, …` (simple_stmts.rst, "The `del` statement":
        // "Deletion of a target list recursively deletes each target,
        // from left to right") — every named target forgets what the
        // walk knew; no judgment and no blocker either way. A deleted
        // name is UNBOUND again afterward (the same state a valueless
        // AnnAssign leaves), but this table only tracks names a valueless
        // AnnAssign itself declared — `del` on an ordinary name states
        // nothing this law reads, so `provably_unbound` is untouched.
        //
        // `del d[k]` (a `Subscript` target, bare-Name receiver): the
        // MUTATION CONTRACT's own delete-shaped write —
        // `collection_models::dict_without_item` answers the receiver
        // WITHOUT that key, which rebinds `name` through
        // `walk_del_subscript_target` so a later read sees the key's
        // absence; an unresolved receiver/key FORGETS `name` (the stale
        // pre-delete value must not survive), the same honesty every
        // other unresolved write in this file keeps. Every other target
        // shape (bare name, tuple/list/starred, a non-Name-receiver
        // subscript) still forgets through `forget_target_names`,
        // unchanged.
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                if let Expr::Subscript(subscript) = target {
                    walk_del_subscript_target(subscript, context, environment);
                } else {
                    forget_target_names(target, environment);
                }
            }
        }
        // `assert test[, msg]`: narrows the environment by the test
        // being true and keeps walking — the same `assume` seam an
        // `if` arm's truthy fork uses. A failing assert raises at
        // runtime and this walk has no exception channel, so only the
        // success continuation is modeled.
        Stmt::Assert(assert) => {
            bind_walrus_targets(assert.test.as_ref(), context, aug_assign_refinements, environment, out);
            *environment = assume(assert.test.as_ref(), environment.fork(), context.kernel, true);
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                bind_walrus_targets(exc, context, aug_assign_refinements, environment, out);
                crate::expressions::evaluate_expression(exc, environment, context.kernel);
            }
        }
        Stmt::Global(_) | Stmt::Nonlocal(_) => {}
        Stmt::If(if_stmt) => {
            provably_unbound.clear();
            let nothing_falls_through = walk_if(
                if_stmt,
                return_refinement,
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
            if nothing_falls_through {
                return FallsThrough::NoAndFollowingIsDead;
            }
        }
        // Imports and type aliases are consumed statements, not
        // blockers: the surface reads them (import identities, the
        // alias table, and — this unit — the cross-module surface) before
        // the walk starts. At MODULE TOP LEVEL, the import statement is
        // where the imported name first becomes live: `context.
        // module_bindings` already carries whatever `cross_module::
        // module_surface` resolved for it, so binding from that table
        // here (rather than merely forgetting) is what makes an imported
        // name readable at its own import site and every statement after
        // it. A name the surface did not resolve (an unresolved module,
        // a name that resolves to a function/class rather than a plain
        // value, a star import's own literal `"*"` alias) still forgets,
        // exactly as before, so no stale fact survives regardless.
        Stmt::Import(import) => {
            for alias in &import.names {
                let local = alias.asname.as_ref().unwrap_or(&alias.name);
                bind_or_forget_imported_name(local.id.as_str(), context, environment);
            }
        }
        Stmt::ImportFrom(import) => {
            for alias in &import.names {
                let local = alias.asname.as_ref().unwrap_or(&alias.name);
                bind_or_forget_imported_name(local.id.as_str(), context, environment);
            }
        }
        Stmt::TypeAlias(alias) => {
            if let Expr::Name(name) = alias.name.as_ref() {
                // The alias's compiled set is judged for emptiness at its own
                // declaration, exactly as the AnnAssign arm judges an inline
                // annotation: a set no value can satisfy is the declaration's
                // own defect and fires before any value flows. A kernel
                // decline leaves the alias unjudged.
                if let Some(entry) = context.aliases.get(name.id.as_str()) {
                    if let Some(true) = declared_set_is_empty(&entry.set, context.kernel) {
                        out.push(Finding {
                            range: alias.value.range(),
                            code: "RTS7003",
                            message: empty_set(&entry.set),
                        });
                    }
                }
                environment.forget(name.id.as_str());
            }
        }
        Stmt::For(_) | Stmt::While(_) => {
            provably_unbound.clear();
            let terminates = walk_loop(stmt, return_refinement, yield_refinement, context, environment, aug_assign_refinements, blocked, out);
            return if terminates { FallsThrough::No } else { FallsThrough::Yes };
        }
        Stmt::Match(match_stmt) => {
            provably_unbound.clear();
            walk_match(
                match_stmt,
                return_refinement,
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        Stmt::With(with_stmt) => {
            provably_unbound.clear();
            walk_with(
                with_stmt,
                return_refinement,
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        Stmt::Try(try_stmt) => {
            provably_unbound.clear();
            let terminates = walk_try(
                try_stmt,
                return_refinement,
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
            return if terminates { FallsThrough::No } else { FallsThrough::Yes };
        }
        _ => {
            provably_unbound.clear();
            record_blocker(
                blocked,
                stmt.range(),
                format!("{} is not yet walked", super::statement_kind_name(stmt)),
                out,
            );
        }
    }
    FallsThrough::Yes
}
