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
//! ## Judging a body's declared-slot writes
//!
//! `check.rs`'s `walk_loop` swaps in this module's post-iteration
//! environment outright, so a body write that is never re-read at a
//! declared sink after the loop needs to be judged HERE, during
//! execution, or not at all. `loop_final_environment` takes the body's
//! own `declared` table (`check.rs`'s `aug_assign_refinements` — every
//! name a preceding `x: Age = …` recorded in this same body) and an
//! `out` sink for judged fires: every bare-name `Assign`/`AugAssign`
//! write inside the body is judged against `declared` through
//! `assignability::judge`, exactly as `check.rs`'s own `judge_and_bind`
//! judges a straight-line write. A `Fire` is pushed to `out` ONCE PER
//! SYNTACTIC ROW (deduped by the statement's own `TextRange` — a loop
//! that iterates many times must not repeat the same fire once per
//! iteration) and the write BINDS the declared set afterward (the same
//! refused-write law `judge_and_bind` uses — the slot keeps its
//! DECLARED set, so a later read in a further iteration or after the
//! loop is silent against it rather than firing again). A name with no
//! recorded declaration in `declared` binds its evaluated value
//! directly, unjudged, matching every other plain local this module
//! already tracks. An `Undetermined` verdict declines the WHOLE loop —
//! this module cannot itself record a body's own blocker in the middle
//! of a run it does not complete, and check.rs's outer blocker for the
//! whole loop statement is the honest stand-in.
//!
//! `Finding` (check.rs's own struct) is not imported here to avoid a
//! cycle (check.rs already imports this module) — judged fires are
//! handed back as plain `(TextRange, String)` rows in `out`, and
//! `check.rs` wraps each into its own `Finding` at the call site.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Stmt;
use ruff_text_size::TextRange;
use crate::env::Environment;
use crate::typereading::DeclaredRefinement;

mod bind_target;
mod body_once;
mod for_loop;
mod iterable;
mod while_loop;
mod widen;

use for_loop::for_loop_final_environment;
use while_loop::while_loop_final_environment;

/// The judging context threaded through every body-execution helper:
/// the body's own declared-refinement table (bare name → its recorded
/// `x: Age = …` annotation, `check.rs`'s own PRE-LOOP snapshot) to judge
/// a write against, `newly_declared` — the SAME shape table for a name
/// this loop's OWN body declares for the first time INSIDE the body
/// (`Stmt::AnnAssign`'s own alias-spelling reuse, see its doc) — checked
/// second so a body-local declaration never shadows the enclosing body's
/// own snapshot, the dedupe set of statement ranges already fired on
/// this run (one fire per SYNTACTIC row, however many iterations
/// actually execute it), and the fires collected so far — moved out into
/// the caller's `out` parameter once the whole run completes.
pub(super) struct JudgeContext<'a> {
    pub(super) declared: &'a HashMap<String, DeclaredRefinement>,
    pub(super) newly_declared: HashMap<String, DeclaredRefinement>,
    pub(super) already_fired: std::collections::HashSet<TextRange>,
    pub(super) fires: Vec<(TextRange, String)>,
}

/// A `for`/`while` statement's own answer: the post-loop environment
/// (whatever the concrete run left, matching `else_runs`'s own
/// documented shape below regardless of `returned`), whether the
/// loop's `else` clause RUNS, and `returned` — `Some((value, range))`
/// when SOME concrete iteration hit a `Stmt::Return` and the loop ended
/// right there (CPython's own semantics: a `return` inside a loop body
/// exits the function, so no further iteration ever runs — RETURN-
/// THROUGH-LOOP CHANNEL, serving c-reads-and-values.py:927/928's own
/// `for age in overs.values(): return age` shape). The inner
/// `value: Option<AbstractValue>` is `None` for a BARE `return` (no
/// expression) — matching `check.rs`'s own `walk_return` convention
/// that a bare return "carries no value expression and judges nothing
/// either"; `Some(value)` for `return <expr>`. `check.rs`'s `walk_loop`
/// judges a `Some` value against the enclosing function's own
/// `-> Annotation` at the carried range, exactly as `walk_return` would
/// for a straight-line return, and ALSO keeps walking the rest of the
/// body with `environment`/`else_runs` — this module never tries to
/// prove the statements after the loop are unreachable (a return that
/// fires on one concrete run states nothing about every OTHER call
/// site's own arguments), so `returned` is purely ADDITIVE information
/// layered on top of the ordinary environment/else_runs answer, never a
/// replacement for it. A return that never fires across every
/// concretely-run iteration reports `returned: None`, unchanged from
/// before this law.
///
/// `widened_names` names every bare name `stabilized_join` had to rebind
/// to `unknown()` because its two-pass join never reached a fixed point
/// (`stabilized_join`'s own doc) — empty for every OTHER answer shape
/// (a concrete per-element run over a known iterable has nothing to
/// widen; `while_loop_final_environment`'s own widening is already a
/// judged fire, not a silent one). `check.rs`'s `walk_loop` records the
/// FIRST name here as this body's own blocker: the loop reached a real
/// stopping point, but that one name's true accumulated value is
/// unreadable past it, and nothing downstream would otherwise say so.
pub struct LoopAnswer {
    pub environment: Environment,
    pub else_runs: bool,
    pub returned: Option<(Option<AbstractValue>, TextRange)>,
    pub widened_names: Vec<String>,
}

/// The post-loop answer for a `for`/`while` statement matching one of
/// this module's concretely-executable shapes (see `LoopAnswer`'s own
/// doc for the full contract). `None` for anything else (any other
/// statement kind, an unrecognized iterable, a body outside the
/// recognized forms, a `while` that does not resolve within the
/// iteration cap, or a body write judged `Undetermined` against
/// `declared`). The walk keeps its own blocker on `None`; this module
/// never runs the `orelse` body itself — `check.rs` walks it (fully
/// judged) when `else_runs`, or fires the dead-else law when not.
pub fn loop_final_environment(
    stmt: &Stmt,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    declared: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<(TextRange, String)>,
) -> Option<LoopAnswer> {
    let mut judge_context = JudgeContext {
        declared,
        newly_declared: HashMap::new(),
        already_fired: std::collections::HashSet::new(),
        fires: Vec::new(),
    };
    let result = match stmt {
        Stmt::For(for_stmt) => for_loop_final_environment(for_stmt, environment, kernel, &mut judge_context),
        Stmt::While(while_stmt) => while_loop_final_environment(while_stmt, environment, kernel, &mut judge_context),
        _ => None,
    };
    // A fire recorded during a run that LATER declines (e.g. iteration 1
    // provably refuses a write, and a later iteration's condition then
    // reads unknown because that same write also widened the counter to
    // a Kind::Set) is still a genuine, already-proven fact: CPython
    // really did execute that statement with that value at least once.
    // Surfacing it — even though the loop as a whole is this module's
    // blocker — is strictly more determined than dropping it silently,
    // so fires propagate unconditionally, before the `?` on the run's
    // own success.
    out.append(&mut judge_context.fires);
    result
}

#[cfg(test)]
pub(self) use body_once::bind_checked;
#[cfg(test)]
pub(self) use body_once::run_assign_once;
#[cfg(test)]
pub(self) use body_once::run_body_once;
#[cfg(test)]
pub(self) use body_once::run_if_once;
#[cfg(test)]
pub(self) use body_once::run_if_once_over_unknown_test;
#[cfg(test)]
pub(self) use for_loop::abstract_element_sort_pass;
#[cfg(test)]
pub(self) use for_loop::repetition_window_element_pass;
#[cfg(test)]
pub(self) use iterable::generator_call_values;
#[cfg(test)]
pub(self) use iterable::is_dict_size_changing_method_call;
#[cfg(test)]
pub(self) use iterable::iterable_values;
#[cfg(test)]
pub(self) use iterable::known_number_sorted;
#[cfg(test)]
pub(self) use iterable::known_string;
#[cfg(test)]
pub(self) use iterable::list_size_changing_mutation_range;
#[cfg(test)]
pub(self) use while_loop::kernel_bounded_counter_environment;
#[cfg(test)]
pub(self) use widen::hull_window;
#[cfg(test)]
pub(self) use widen::stabilized_join;
#[cfg(test)]
pub(self) use widen::stable_by_containment;
#[cfg(test)]
pub(self) use widen::widened_set_candidate;

#[cfg(test)]
mod tests;
