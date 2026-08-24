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
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::loop_questions::InvariantPremise;
use refined_kernel::loop_questions::InvariantPremiseKind;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use refined_kernel::loop_questions::LoopQuestion;
use refined_kernel::loop_questions::LoopVarAnswerKind;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprAttribute;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprSubscript;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFor;
use ruff_python_ast::StmtIf;
use ruff_python_ast::StmtWhile;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::assignability::judge;
use crate::assignability::Verdict;
use crate::collection_models;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances;
use crate::narrowing::assume;
use crate::summaries::iterable_element_sort;
use crate::typereading::DeclaredRefinement;

/// A `while` loop is only concretely executed up to this many
/// iterations. Reaching the cap with the condition still true means
/// the bound was not proved (an unbounded or too-large loop) — this
/// function declines rather than guessing where it converges.
const WHILE_ITERATION_CAP: u32 = 1000;

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
struct JudgeContext<'a> {
    declared: &'a HashMap<String, DeclaredRefinement>,
    newly_declared: HashMap<String, DeclaredRefinement>,
    already_fired: std::collections::HashSet<TextRange>,
    fires: Vec<(TextRange, String)>,
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
enum BodyOutcome {
    Fell,
    Broke,
    Continued,
    Returned(Option<AbstractValue>, TextRange),
}

/// `for target in <iterable>: <body> [else: <body>]` — every element
/// this module's `iterable_values` recognizes is fully known, so the
/// body runs once per element over a forked environment. Python leaves
/// the target bound to the last element after the loop ends (never
/// reset or deleted, compound_stmts.html "the for statement"); an empty
/// iterable runs the body zero times, so the target keeps whatever the
/// pre-loop environment already held for that name. A `break` on any
/// iteration stops the loop AT that element (the target stays bound to
/// the element the `break` fired on) and reports `else_runs: false`;
/// otherwise (the iterable is exhausted with no `break`) `else_runs:
/// true` — this function never runs `for_stmt.orelse` itself
/// (`check.rs` walks it, fully judged, when `else_runs`). A `return`
/// stops the loop immediately (no further elements bind, no `else`
/// clause runs — `else_runs: false`, matching `break`'s own posture,
/// though `check.rs` never reads `else_runs` once `returned` is `Some`)
/// and reports `returned: Some((value, range))`.
///
/// `for_stmt.is_async` (an `async for`) runs through the SAME
/// `iterable_values` path as a plain `for` — compound_stmts.rst, "The
/// `async for` statement": an `async for` desugars to a `while` binding
/// `TARGET = await type(iter).__anext__(iter)` each pass, and `await`
/// only ever suspends/resumes scheduling around whatever value the
/// awaited call eventually produces; it never changes WHICH elements
/// come out of a receiver whose element sequence this module already
/// reads concretely (a literal tuple/list, `range(...)`, a dict view —
/// `iterable_values`'s own recognized shapes). There is no such literal/
/// range/dict-view shape that is also asynchronous in the corpus or in
/// CPython at all (those builtins have no `__aiter__`), so this arm is
/// reachable only in principle; the honest boundary is that `is_async`
/// itself is NEVER the reason to decline — an unrecognized receiver
/// (an async generator call, a custom `__aiter__`/`__anext__` class
/// instance — a-statements.py:555's `stream()`, b-body-expressions.py:
/// 877's `Stream()`) still declines through `iterable_values`'s own
/// `None`, exactly as an equivalent unmodeled SYNC receiver would.
/// Concretely stepping a genuine async source is out of this function's
/// scope regardless of `is_async`: an async iterator's `__anext__` is
/// arbitrary code this module never executes, sync or async.
fn for_loop_final_environment(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    if let Some(elements) = iterable_values(for_stmt.iter.as_ref(), environment, kernel) {
        // ITERATOR INVALIDATION: a dict iterated DIRECTLY (`iterated_dict_
        // name`) whose own body provably changes that same dict's size on
        // every reachable pass (`dict_size_changing_mutation_range`) raises
        // `RuntimeError` on CPython's first such pass — checked here, before
        // any element runs, so an EMPTY dict (no elements, the `for` never
        // executes its body at all) never raises, matching real CPython
        // exactly. A non-empty dict's raise is provable regardless of what
        // ELSE the body does: this fire is recorded and the whole loop
        // still declines (`None`, below) — the loop's own decline is
        // secondary to the raise this checker can now state exactly, the
        // "fires propagate unconditionally" convention `loop_final_
        // environment`'s own doc already keeps for a fire discovered on a
        // run that later declines.
        if !elements.is_empty() {
            if let Some(dict_name) = iterated_dict_name(for_stmt.iter.as_ref(), environment) {
                if let Some(range) = dict_size_changing_mutation_range(&for_stmt.body, dict_name) {
                    judge_context
                        .fires
                        .push((range, crate::diagnostic_sentences::dict_changed_size_during_iteration(dict_name)));
                    return None;
                }
            }
        }
        let mut current = environment.fork();
        let mut broke = false;
        for element in elements {
            if !bind_for_target(for_stmt.target.as_ref(), &element, &mut current) {
                return None;
            }
            match run_body_once(&for_stmt.body, &mut current, kernel, judge_context)? {
                BodyOutcome::Fell | BodyOutcome::Continued => {}
                BodyOutcome::Broke => {
                    broke = true;
                    break;
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
        }
        return Some(LoopAnswer { environment: current, else_runs: !broke, returned: None, widened_names: Vec::new() });
    }
    abstract_element_sort_pass(for_stmt, environment, kernel, judge_context)
        .or_else(|| custom_iterator_element_pass(for_stmt, environment, kernel, judge_context))
        .or_else(|| repetition_window_element_pass(for_stmt, environment, kernel, judge_context))
        .or_else(|| windowed_range_element_pass(for_stmt, environment, kernel, judge_context))
}

/// `for`/`async for` over a CUSTOM ITERATOR — a class instance whose own
/// model declares the iterator protocol (`__aiter__`/`__anext__` for
/// `async for`, `__iter__`/`__next__` for a plain `for`,
/// b-body-expressions.py's own `Stream`/a-statements.py:555's `stream()`
/// twin). The element is `__anext__`/`__next__`'s own declared return
/// SORT (`typereading::base_sort_return_refinement`, the same bare-
/// `int`/`float`/`str` reading `iterable_element_sort` gives a
/// generator's own `-> AsyncIterator[int]` one subscript level up) —
/// never the method body's own concretely INTERPRETED value, even when
/// that body is a plain `return <literal>` a restricted interpreter
/// could read exactly: the real iteration protocol has no length this
/// checker can observe (the loop ends only when `__anext__`/`__next__`
/// itself raises `StopIteration`, arbitrary code this module never
/// executes), so trusting one concrete call's answer as "the" element
/// would overstate what a single read of `__anext__` proves about every
/// call the real loop could make. This mirrors `abstract_element_sort_
/// pass`'s own doctrine exactly, applied to a class-based iterator
/// instead of a same-module generator def: state the SORT, run the
/// body once through the same judged executor, then JOIN the pre-loop
/// and one-pass environments for the same zero-or-more honesty.
///
/// `None` when the iterable is not a bare call (`f(...)`), the callee
/// does not resolve to a known class, the class's own model declares
/// neither iterator-protocol pair, or the resolved `__anext__`/
/// `__next__` method states no bare `int`/`float`/`str` return
/// annotation this reader recognizes.
fn custom_iterator_element_pass(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    let receiver = evaluate_expression(for_stmt.iter.as_ref(), environment, kernel);
    if receiver.kind != Kind::Object || receiver.source.is_empty() {
        return None;
    }
    let classes = environment.classes()?;
    let model = classes.get(receiver.source.as_str())?;
    let next_method_name = if instances::method_def_of(model, "__aiter__").is_some()
        && instances::method_def_of(model, "__anext__").is_some()
    {
        "__anext__"
    } else if instances::method_def_of(model, "__iter__").is_some()
        && instances::method_def_of(model, "__next__").is_some()
    {
        "__next__"
    } else {
        return None;
    };
    let next_method = instances::method_def_of(model, next_method_name)?;
    let declared = crate::typereading::base_sort_return_refinement(next_method.returns.as_deref()?)?;
    let element = known_set(declared.set, None, TrustSpec, SetKindTag::None);

    let mut one_pass = environment.fork();
    if !bind_for_target(for_stmt.target.as_ref(), &element, &mut one_pass) {
        return None;
    }
    match run_body_once(&for_stmt.body, &mut one_pass, kernel, judge_context)? {
        BodyOutcome::Fell | BodyOutcome::Continued | BodyOutcome::Broke => {}
        BodyOutcome::Returned(value, range) => {
            return Some(LoopAnswer {
                environment: one_pass,
                else_runs: false,
                returned: Some((value, range)),
                widened_names: Vec::new(),
            });
        }
    }
    let (joined, widened_names) = stabilized_join(
        environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        kernel,
        judge_context,
    )?;
    Some(LoopAnswer { environment: joined, else_runs: true, returned: None, widened_names })
}

/// `for`/`async for` over a KNOWN-LENGTH-UNKNOWN, known-element-set
/// receiver — `Kind::Set` whose only form is the repetition window
/// `check.rs::seed_parameters` builds for a declared `list[X]`/`set[X]`/
/// `Sequence[X]` parameter (the element set repeated rather than nested
/// into exact positional slots, `collection_models::star_element_read`'s
/// own doc — the same window shape, read the same way, never a second
/// reader). Every position of the window draws from the SAME element
/// set, so there is exactly one abstraction to bind the target against:
/// `as_repetition` reads the window back to its element
/// (`refined_sets::repetition_window_forms`), the target binds to
/// `known_set(element, ...)` carrying the SEQUENCE's own trust grade
/// (`trust_level_of(receiver)` — the same grade `star_element_read`
/// assigns a read element), and the body runs ONE judged pass over that
/// single binding, joined with the pre-loop environment for the same
/// zero-or-more honesty `abstract_element_sort_pass` states in its own
/// doc.
///
/// `None` when the iterable's evaluated value is not a bare repetition
/// window (any other `Kind::Set` shape, or a different `Kind` — the
/// concrete `iterable_values` path and `custom_iterator_element_pass`
/// both already declined by the time this pass runs), when `as_
/// repetition` cannot read the window back, or when the one abstract
/// pass hits a statement shape `run_body_once` does not recognize.
fn repetition_window_element_pass(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    let receiver = evaluate_expression(for_stmt.iter.as_ref(), environment, kernel);
    if receiver.kind != Kind::Set || receiver.set_kind_tag != SetKindTag::None {
        return None;
    }
    // ITERATOR NON-TERMINATION: `for x in lst: lst.append(x)` — a list
    // iterated DIRECTLY (`iterated_list_name`) whose own body provably
    // appends to that SAME list on every reachable pass
    // (`list_size_changing_mutation_range`) never reaches its own end
    // (a list's iterator, unlike a dict's, carries no size-changed
    // guard — `diagnostic_sentences::list_never_terminates_self_append`'s
    // own citation) — checked here, before the one abstract pass runs,
    // mirroring `for_loop_final_environment`'s own dict-iterator-
    // invalidation check one level up: this fire is recorded and the
    // whole loop still declines (`None`, below), the same "fires
    // propagate unconditionally" convention this file already keeps.
    if let Some(list_name) = iterated_list_name(for_stmt.iter.as_ref()) {
        if let Some(range) = list_size_changing_mutation_range(&for_stmt.body, list_name) {
            judge_context.fires.push((range, crate::diagnostic_sentences::list_never_terminates_self_append(list_name)));
            return None;
        }
    }
    let repeated = as_repetition(&receiver.set)?;
    // The element inherits the sequence's own numeric sort — the same
    // threading the comprehension's element bind and star_element_read
    // perform, so a body term like `s * s` reaches the sort-gated
    // transfer models.
    let element = AbstractValue {
        kind_tag: receiver.kind_tag,
        ..known_set(repeated.element, None, trust_level_of(&receiver), SetKindTag::None)
    };

    let mut one_pass = environment.fork();
    if !bind_for_target(for_stmt.target.as_ref(), &element, &mut one_pass) {
        return None;
    }
    match run_body_once(&for_stmt.body, &mut one_pass, kernel, judge_context)? {
        BodyOutcome::Fell | BodyOutcome::Continued | BodyOutcome::Broke => {}
        BodyOutcome::Returned(value, range) => {
            return Some(LoopAnswer {
                environment: one_pass,
                else_runs: false,
                returned: Some((value, range)),
                widened_names: Vec::new(),
            });
        }
    }
    let (joined, widened_names) = stabilized_join(
        environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        kernel,
        judge_context,
    )?;
    Some(LoopAnswer { environment: joined, else_runs: true, returned: None, widened_names })
}

/// WINDOWED-RANGE PASS: `for i in range(<expr>)` where the stop
/// expression evaluates to a SET or a multi-member Values binding
/// rather than one known number (`iterable_values`' concrete path
/// already declined — A1.xfer.loop's own `for i in range(n)` under
/// `n: Wide`). `range`'s elements are the integers `0 <= i < stop`
/// (library/stdtypes.html#range, "range(stop)"), so across every stop
/// the binding admits, the target's element set is
/// `integer ∧ [0, max(stop) - 1]` — bounded below by 0 always, and
/// above by the stop set's own upper form when it carries one (an
/// unbounded stop leaves the element set unbounded above, still sound).
/// The element binds once and the body runs the SAME one judged pass +
/// pre-loop join every abstract pass here uses, so a body return of the
/// counter reaches `check.rs`'s return judgment carrying the window —
/// the row's own fire or silence.
///
/// `None` when the iterable is not a one-positional-argument `range`
/// call, when the evaluated stop is neither a Set nor an all-integer
/// Values binding, or when a Set stop does not prove integer sort
/// (`range` accepts only int arguments — a float-sorted stop raises,
/// which this pass never vouches for). A stop provably at most 0 runs
/// the body ZERO times: the answer is the pre-loop environment
/// unchanged, `else_runs: true` — real CPython, an empty range.
fn windowed_range_element_pass(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    let Expr::Call(call) = for_stmt.iter.as_ref() else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "range" {
        return None;
    }
    if !call.arguments.keywords.is_empty() || call.arguments.args.len() != 1 {
        return None;
    }
    let stop = evaluate_expression(&call.arguments.args[0], environment, kernel);
    let element_upper = range_stop_element_upper(&stop)?;
    if let Some(upper) = element_upper {
        if upper < 0.0 {
            // every admitted stop is <= 0: the body never runs.
            return Some(LoopAnswer { environment: environment.fork(), else_runs: true, returned: None, widened_names: Vec::new() });
        }
    }
    let mut forms = vec![at_least(0.0), integer()];
    if let Some(upper) = element_upper {
        forms.push(at_most(upper));
    }
    let element = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(forms), None, trust_level_of(&stop), SetKindTag::None)
    };

    let mut one_pass = environment.fork();
    if !bind_for_target(for_stmt.target.as_ref(), &element, &mut one_pass) {
        return None;
    }
    match run_body_once(&for_stmt.body, &mut one_pass, kernel, judge_context)? {
        BodyOutcome::Fell | BodyOutcome::Continued | BodyOutcome::Broke => {}
        BodyOutcome::Returned(value, range) => {
            return Some(LoopAnswer {
                environment: one_pass,
                else_runs: false,
                returned: Some((value, range)),
                widened_names: Vec::new(),
            });
        }
    }
    let (joined, widened_names) = stabilized_join(
        environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        kernel,
        judge_context,
    )?;
    Some(LoopAnswer { environment: joined, else_runs: true, returned: None, widened_names })
}

/// The largest element `range(stop)` can yield across every stop a
/// binding admits, read off the binding itself:
/// - a Values binding of known integers answers `max - 1` exactly;
/// - a Set binding must prove integer sort (an `Integer` form, or an
///   Integer kind tag) and answers the largest admitted stop minus 1:
///   `atMost a` admits stops up to `⌊a⌋`, `below a` up to `a - 1` when
///   `a` is itself an integer (`⌊a⌋` otherwise), `oneOf w` up to
///   `max(w)`; several forms intersect, so the tightest wins;
/// - `Ok(None)` (outer `Some(None)`) when the set proves integer sort
///   but carries no upper form — unbounded above, still walkable.
/// `None` declines: not a Set/Values shape, a non-integer member, or a
/// Set with no integer proof.
fn range_stop_element_upper(stop: &AbstractValue) -> Option<Option<f64>> {
    match stop.kind {
        Kind::Values => {
            if stop.values.is_empty() {
                return None;
            }
            if stop.values.iter().any(|member| !member.is_finite() || member.fract() != 0.0) {
                return None;
            }
            let max = stop.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            Some(Some(max - 1.0))
        }
        Kind::Set => {
            let integer_sorted = stop.kind_tag == Some(PrimitiveKind::Integer)
                || stop.set.forms.iter().any(|form| form.form == Form::Integer);
            if !integer_sorted {
                return None;
            }
            let mut upper: Option<f64> = None;
            let mut tighten = |candidate: f64| {
                upper = Some(match upper {
                    Some(held) => held.min(candidate),
                    None => candidate,
                });
            };
            for form in &stop.set.forms {
                match form.form {
                    // stop <= a, stop an integer: stop <= ⌊a⌋.
                    Form::AtMost => tighten(form.a.floor() - 1.0),
                    // stop < a, stop an integer: stop <= a - 1 when a is
                    // itself an integer, ⌊a⌋ otherwise.
                    Form::Below => tighten(if form.a.fract() == 0.0 { form.a - 2.0 } else { form.a.floor() - 1.0 }),
                    Form::OneOf => {
                        if form.w.is_empty() || form.w.iter().any(|member| !member.is_finite() || member.fract() != 0.0) {
                            return None;
                        }
                        tighten(form.w.iter().copied().fold(f64::NEG_INFINITY, f64::max) - 1.0);
                    }
                    _ => {}
                }
            }
            Some(upper)
        }
        _ => None,
    }
}

/// ABSTRACT SORT-ELEMENT PASS: `for`/`async for` over a same-module
/// generator/stream `def` whose OWN element sort is readable
/// (`summaries::iterable_element_sort`) but whose concrete elements are
/// not (`iterable_values` already declined — a-statements.py's own
/// `async_for_over_stream`: `stream() -> AsyncIterator[int]` declines
/// the body-level `raise` this checker never executes concretely, so
/// there is no LIST of known iterates to walk one at a time the way
/// `for_loop_final_environment`'s own concrete path does). Mirrors
/// refined-ts-go's own abstract loop walk (one JUDGED pass standing in
/// for the whole unknown-length run, `tmp/cpython/Doc/reference/
/// compound_stmts.rst`'s `for` statement: the body may run zero or more
/// times over an iterable whose LENGTH this checker cannot observe, so
/// no CONCRETE per-element walk is honest here — a single pass over the
/// CLAIMED element sort is the coarser, sound stand-in): the target
/// binds to the element sort-set (never one concrete value — every
/// element the real stream could produce is somewhere in that set), the
/// body runs ONCE through the same judged executor (`run_body_once`) a
/// concrete pass already uses, so a declared-slot write inside the body
/// (`age = chunk` under `age: Age`) reaches `bind_checked`'s own
/// `assignability::judge` CONTAINMENT law and fires exactly as it would
/// on a concrete iterate — the row's own fire. The answer JOINS the
/// PRE-LOOP environment (the zero-iterations possibility — an empty
/// stream runs the body not at all) with the ONE-PASS environment (the
/// at-least-one-iteration possibility) through `Environment::join`,
/// stating the loop's own zero-or-more semantics honestly rather than
/// assuming the body ran. `else_runs: true` (the `for`/`else` clause
/// runs whenever no `break` stops the loop — an abstract pass never
/// observes a `break`, so `else_runs` cannot be proven false here; a
/// `break`/`continue`/`return` inside the one abstract pass still
/// propagates through `outcome_of_body`, and a `Returned` outcome still
/// reports `returned: Some(...)`, the same RETURN-THROUGH-LOOP CHANNEL
/// every concrete pass uses).
///
/// `None` when the iterable is not a bare-Name call to a SAME-MODULE def
/// (any keyword/starred argument declines, matching `generator_call_
/// values`'s own no-keyword-guessing posture), the def's element sort is
/// itself unreadable, or the one abstract pass hits a statement shape
/// `run_body_once` does not recognize — the ordinary "this shape is not
/// this module's business" decline, same honesty as every other row.
fn abstract_element_sort_pass(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    let Expr::Call(call) = for_stmt.iter.as_ref() else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() || !call.arguments.args.is_empty() {
        return None;
    }
    let table = environment.functions()?;
    let def = table.def(callee.id.as_str())?;
    let element_sort = iterable_element_sort(def)?;

    let mut one_pass = environment.fork();
    if !bind_for_target(for_stmt.target.as_ref(), &element_sort, &mut one_pass) {
        return None;
    }
    match run_body_once(&for_stmt.body, &mut one_pass, kernel, judge_context)? {
        BodyOutcome::Fell | BodyOutcome::Continued | BodyOutcome::Broke => {}
        BodyOutcome::Returned(value, range) => {
            return Some(LoopAnswer {
                environment: one_pass,
                else_runs: false,
                returned: Some((value, range)),
                widened_names: Vec::new(),
            });
        }
    }
    let (joined, widened_names) = stabilized_join(
        environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element_sort,
        kernel,
        judge_context,
    )?;
    Some(LoopAnswer { environment: joined, else_runs: true, returned: None, widened_names })
}

/// The loop target's own bare-name spelling, if the target is one — a
/// tuple target's own sub-names are read the same way `bind_for_target`
/// binds them, but every one of these three passes only ever binds a
/// SINGLE element abstraction to the whole target, so a tuple target
/// here is out of scope for the same reason it already declines
/// `bind_for_target` widely elsewhere: this helper only needs the bare
/// case to build the exclusion set `stabilized_join` compares against.
fn target_names(target: &Expr, names: &mut std::collections::HashSet<String>) {
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
fn written_names(body: &[Stmt], names: &mut std::collections::HashSet<String>) {
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

/// Whether `narrower.set` is provably contained in `wider.set` — the
/// question `stabilized_join` asks when the structural rejoin test
/// cannot answer stability for a `Kind::Set` pair, because `join_known`'s
/// general set-combining path has NO STRUCTURAL FIXPOINT: it always
/// wraps both operand sets in a fresh, unreduced `union(...)` node
/// (`lattice_operations.rs`'s fallback), so `join(J, second_pass)`
/// re-wraps rather than converging back to `J`'s own shape even when the
/// second pass denotes nothing new — a raw element set and that same set
/// folded one layer deeper through a prior `union` never compare equal
/// under `RefinedSet`'s derived `PartialEq`, no matter how many times the
/// rejoin runs. Stability under repetition means exactly that the second
/// pass's set is already covered by the first join's set (join only
/// grows, so "the rejoin adds nothing" and "the second pass's set ⊆ the
/// first join's set" are the same claim) — a question the KERNEL decides
/// on the actual admitted values, not on either side's syntactic form.
///
/// `kernel.scalar_subset` is tried first — the ordinary 1-tuple-layer
/// question, covering the two-passes-of-a-numeric-set case both
/// `g_iter_bind`/`g_iter_mul` are — then `kernel.seq_subset` when
/// `scalar_subset` refuses (a sequence-shaped set the scalar decider
/// cannot read; `assignability.rs`'s own containment law tries the same
/// two asks, ordered by which shape is more likely, with the same
/// fallback-on-refusal posture). Both asks panic inside the kernel
/// closure on a refusal — the crate's established `catch_unwind`/
/// `AssertUnwindSafe` idiom (`assignability.rs`, `lattice_conformance.rs`)
/// catches that and reads it as "no proof," never a crash. `true` from
/// either ask is a theorem; `false`, or a refusal from both, is not a
/// disproof — it is simply no proof of stability, so the caller havocs,
/// the same posture every other refused containment ask in this crate
/// already takes.
fn stable_by_containment(narrower: &RefinedSet, wider: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    let scalar_asked = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(narrower, wider));
    if let Ok(subset) = scalar_asked {
        return subset;
    }
    let seq_asked = crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(narrower, wider));
    matches!(seq_asked, Ok(true))
}

/// The stability check every one-pass-plus-join abstract loop pass
/// shares: a body that only REBINDS its written names (`last = s`) sees
/// the same value on a second pass as the first, so joining the pre-loop
/// state with one pass is already a fixpoint. A body that ACCUMULATES
/// (`total += s * s`) does not — a second pass adds another term on top
/// of the first pass's own joined value, so the name a single join would
/// report is a bound the real, unboundedly-many-iteration run can
/// exceed. This function tells the two apart by running the body a
/// SECOND time, from a fork of the join of `environment` (pre-loop) and
/// `one_pass` (the first pass's own environment) — call that join `J` —
/// and testing, for every name the body writes, whether joining the
/// SECOND pass's own value into `J` changes `J` at all: a name is stable
/// when `join(J, second_pass) == J`, since `join_known` is idempotent
/// exactly where the second pass adds no new information beyond what `J`
/// already states. `PartialEq` alone answers this correctly for a
/// `Kind::Values` pair (the same-tag join arms only append values not
/// already present, so an already-covered join reproduces the identical
/// `Vec<f64>`), but `join_known` HAS NO STRUCTURAL FIXPOINT for a
/// `Kind::Set` pair — its general fallback always wraps both sides in a
/// fresh `union(...)` node, so a rejoin that denotes nothing new still
/// produces a NEW, differently-shaped `RefinedSet` that `PartialEq` calls
/// unequal. For that case (both sides `Kind::Set`, `SetKindTag::None`)
/// the structural mismatch is not read as instability outright — the
/// kernel is asked the real question instead, `stable_by_containment`'s
/// own containment verdict: the second pass's set is stable exactly when
/// it is CONTAINED in `J`'s set, which is what "the rejoin adds nothing"
/// actually means once the join no longer has a structural fixed point
/// to compare against. A name whose value is still `PartialEq`-unequal
/// AND (for a Set pair) not kernel-proved contained is REBOUND to
/// `unknown()` in the final environment, since it holds no claim this
/// walk can make; every other name — including one the body never
/// actually touches on this concrete run — keeps its `J` value. The loop
/// target itself is excluded from this comparison and this havoc: it is
/// rebound to a fresh element abstraction every iteration by construction
/// (`bind_for_target`'s own call at each pass), never accumulated, so
/// comparing it across passes would only ever measure two different
/// intentional bindings and never a genuine instability.
///
/// The names compared are every bare name `written_names` finds
/// SYNTACTICALLY in `body` (a superset of what one concrete pass
/// actually writes is safe here — see that function's own doc). For each
/// one, `J`'s own value and the second pass's own value are re-joined
/// through the same `lattice_operations::join_known` every ordinary
/// branch join already uses (via `Environment::join` on two single-
/// binding forks). A value that happens to be `PartialEq`-different from
/// `J` after the re-join, and is not a `Kind::Set` pair the kernel proves
/// contained, is at worst treated as unstable and havoced to unknown,
/// which is always a weaker, still-undetermined answer, never a wrong
/// one — a question the kernel refuses leaves the name havoced, the same
/// as a structural mismatch it never had the chance to ask about.
///
/// Returns `None` when the second pass hits a statement shape `run_body_
/// once` cannot run (the same "this loop is not this module's shape"
/// decline the first pass already uses) — an unwalkable second pass
/// gives no stability answer to trust, so the whole loop declines rather
/// than publish the first pass's own join unchecked. The second pass's
/// own control-flow outcome (`Broke`/`Continued`/`Returned`) is read only
/// as this success/failure signal; its `Returned` value is not itself
/// used to build the answer since the second pass's whole purpose here
/// is the stability comparison, not a fresh answer to return through.
///
/// `Some((environment, widened))` — `widened` names every bare name this
/// pass rebound to `unknown()` because it never reached a fixed point,
/// SORTED (`HashSet` iteration order is not stable, and a body writing
/// more than one such name still needs a single, reproducible FIRST name
/// for the caller's own blocker) — empty when every written name
/// stabilized. This function itself records no finding: `check.rs`'s
/// `walk_loop` owns turning a non-empty `widened` into this body's own
/// blocker, the same way it already owns every other loop-shaped
/// blocker.
fn stabilized_join(
    environment: &Environment,
    one_pass: &Environment,
    body: &[Stmt],
    target: &Expr,
    element: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<(Environment, Vec<String>)> {
    let joined = Environment::join(environment.fork(), one_pass);

    let mut second_pass = joined.fork();
    if !bind_for_target(target, element, &mut second_pass) {
        return None;
    }
    run_body_once(body, &mut second_pass, kernel, judge_context)?;

    let mut excluded = std::collections::HashSet::new();
    target_names(target, &mut excluded);
    let mut candidates = std::collections::HashSet::new();
    written_names(body, &mut candidates);

    let mut result = joined.fork();
    let mut widened: Vec<String> = Vec::new();
    for name in candidates {
        if excluded.contains(&name) {
            continue;
        }
        let Some(joined_value) = joined.read(&name) else {
            continue;
        };
        let Some(second_value) = second_pass.read(&name) else {
            continue;
        };
        // a single-name fork carrying just this one binding, joined
        // against the same single-name binding off `joined` — the
        // per-name reading of `join(J, second_pass) == J`, built out of
        // the same two-environment `Environment::join` every call site
        // already uses rather than a new per-value join entry point.
        let mut left = joined.fork();
        left.bind(&name, joined_value.clone());
        let mut right = joined.fork();
        right.bind(&name, second_value.clone());
        let rejoined = Environment::join(left, &right);
        let rejoined_value = rejoined.read(&name);
        let mut stable = rejoined_value == joined.read(&name);
        // the structural rejoin has no fixpoint for a Set pair — ask the
        // kernel whether the second pass's set is genuinely covered by
        // `J`'s set before havocing what may be a real determination.
        if !stable
            && joined_value.kind == Kind::Set
            && second_value.kind == Kind::Set
            && joined_value.set_kind_tag == SetKindTag::None
            && second_value.set_kind_tag == SetKindTag::None
        {
            stable = stable_by_containment(&second_value.set, &joined_value.set, kernel);
        }
        if !stable {
            result.bind(&name, unknown());
            widened.push(name);
        }
    }
    widened.sort();
    Some((result, widened))
}

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
fn while_loop_final_environment(
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
/// - a same-module (sync or async) generator `def`'s own call
///   (`generator_call_values`, `instances::generator_yields`) — a
///   bare-Name call whose def's body is straight-line `yield`
///   statements; each yielded value becomes one iterate, in yield
///   order.
///
/// Anything else (a name that is not a known dict, a call other than
/// `range`/`.values`/`.items`/`.keys`/a readable same-module generator,
/// a non-literal element whose EVALUATED value is not itself known) is
/// `None`: this function only answers when every iterate is known
/// without running any unmodeled code.
///
/// EVALUATED ELEMENTS: a `List`/`Tuple` display's own elements are read
/// SYNTACTICALLY first (`sorted_number_literal_value` — the exact
/// literal-number path, which also carries the element's true Integer/
/// Float sort); an element that is not a bare number literal falls back
/// to `evaluate_expression`. a-statements.py's `for_over_unread_iterable`:
/// `(unread_number(),)`'s single element is a CALL, and `unread_number`'s
/// own body (`raise NotImplementedError`) is a genuine decline in
/// `summaries::interpret_body` (no `Stmt::Raise` row there) — its call
/// answers `return_sort_fallback`'s `-> int` claim instead, the
/// whole-number SET (`Kind::Set`, Integer-tagged), never `Kind::Null`.
/// Accepted evaluated shapes: ANY known AbstractValue whose `kind` is not
/// `Kind::Unknown` — a known single scalar, `Kind::Null`, or a known SET
/// (Integer/Float/String-sorted, `Kind::Set`) all accepted alike, because
/// the DISPLAY's own element COUNT is syntactic (this is a fixed-arity
/// tuple/list literal, not an iterable whose length depends on a value),
/// so binding the `for` target to each element's own value — whatever
/// shape that value is — and running the body once per element is sound
/// regardless of what sort of value that element turns out to be. Only a
/// truly UNKNOWN element (`Kind::Unknown` — nothing at all is known about
/// it) declines the WHOLE display, same as every other honest refusal in
/// this file. This acceptance is scoped to a DISPLAY's own elements only:
/// a non-display iterable (a bare Name bound to a set-VALUED expression,
/// for instance) has no syntactic element count to fall back on and is
/// not read through this function at all.
fn iterable_values(
    iterable: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    match iterable {
        Expr::List(list) => elements_as_values(&list.elts, environment, kernel),
        Expr::Tuple(tuple) => elements_as_values(&tuple.elts, environment, kernel),
        Expr::Call(call) => range_call_values(call)
            .or_else(|| dict_view_call_values(call, environment, kernel))
            .or_else(|| generator_call_values(call, environment, kernel)),
        Expr::Dict(_) => {
            let receiver = evaluate_expression(iterable, environment, kernel);
            dict_keys_as_strings(&receiver)
        }
        // Any other iterable expression (a bare Name most commonly)
        // whose EVALUATED value is a known List of fully-known items:
        // the element count is carried by the value itself, so
        // iterating its items is exactly as sound as a display's — the
        // same acceptance rule elements_as_values applies per element.
        // A known dict value iterates its keys, the same reading the
        // Dict-display arm gives. Anything else stays None.
        other => {
            let receiver = evaluate_expression(other, environment, kernel);
            if receiver.kind == Kind::List
                && receiver.items.iter().all(|item| item.kind != Kind::Unknown)
            {
                return Some(receiver.items.clone());
            }
            dict_keys_as_strings(&receiver)
        }
    }
}

fn elements_as_values(
    elements: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::with_capacity(elements.len());
    for element in elements {
        if let Some(literal) = sorted_number_literal_value(element) {
            values.push(literal);
            continue;
        }
        let evaluated = evaluate_expression(element, environment, kernel);
        if evaluated.kind == Kind::Unknown {
            return None;
        }
        values.push(evaluated);
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

/// The dict name a `for` loop iterates DIRECTLY over its own entries —
/// `for k in d:`/`for k in d.keys():`/`for v in d.values():`/`for k, v
/// in d.items():` — bound to a known `Kind::Object` in `environment`.
/// `Some(name)` only for a bare-Name receiver (a fresh dict literal or a
/// computed expression has no single WRITABLE name a body statement
/// could mutate through, so `dict_size_changing_mutation_range` has
/// nothing to match against); every other iterable shape (a list/tuple
/// display, `range(...)`, a generator call, a dict LITERAL display) is
/// `None` — this reader exists only to feed the iterator-invalidation
/// check below, never `iterable_values`'s own element-reading contract.
fn iterated_dict_name<'a>(iterable: &'a Expr, environment: &Environment) -> Option<&'a str> {
    let receiver_expr = match iterable {
        Expr::Name(name) => name.id.as_str(),
        Expr::Call(call) => {
            let Expr::Attribute(attribute) = call.func.as_ref() else {
                return None;
            };
            if !matches!(attribute.attr.as_str(), "keys" | "values" | "items") {
                return None;
            }
            let Expr::Name(name) = attribute.value.as_ref() else {
                return None;
            };
            name.id.as_str()
        }
        _ => return None,
    };
    let receiver = environment.read(receiver_expr)?;
    if receiver.kind != Kind::Object {
        return None;
    }
    Some(receiver_expr)
}

/// Whether `expr` is one of the four dict methods that provably change a
/// dict's own SIZE — `.pop(...)`/`.popitem()`/`.clear()` — called on a
/// bare Name equal to `dict_name`, or a `del <dict_name>[...]` subscript
/// target reads the identical shape one level up in
/// `dict_size_changing_mutation_range`. `d[key] = value` and `.update(...)`
/// are deliberately EXCLUDED: an existing-key assignment never changes
/// size at all (library/stdtypes.rst never raises there), and `.update`'s
/// own size delta is not staticaly provable from its argument alone — this
/// function only ever names a mutation CPython's own dict-views note
/// states unconditionally changes size ("don't add or remove entries").
fn is_dict_size_changing_method_call(expr: &Expr, dict_name: &str) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return false;
    };
    if receiver.id.as_str() != dict_name {
        return false;
    }
    matches!(attribute.attr.as_str(), "pop" | "popitem" | "clear")
}

/// Scans `body`'s own TOP-LEVEL statements (mirroring `run_statement_once`'s
/// own straight-line scope — a mutation nested inside an `if`/`for`/`try`
/// one level down is not proved to run on EVERY reachable pass, so it is
/// outside this function's provable claim) for a statement that provably
/// changes `dict_name`'s own size: `del dict_name[...]`,
/// `dict_name.pop(...)`, `dict_name.popitem()`, `dict_name.clear()`
/// (`is_dict_size_changing_method_call`'s own set, as an expression
/// statement). `Some(range)` names the FIRST such statement's own range —
/// the first-blocker-wins convention this file's own `already_fired`
/// dedupe and `check.rs`'s `record_blocker` both keep; `None` when no
/// top-level statement in this body provably changes the dict's size.
fn dict_size_changing_mutation_range(body: &[Stmt], dict_name: &str) -> Option<TextRange> {
    for stmt in body {
        match stmt {
            Stmt::Delete(delete) => {
                for target in &delete.targets {
                    if let Expr::Subscript(subscript) = target {
                        if let Expr::Name(receiver) = subscript.value.as_ref() {
                            if receiver.id.as_str() == dict_name {
                                return Some(stmt.range());
                            }
                        }
                    }
                }
            }
            Stmt::Expr(expr_stmt) if is_dict_size_changing_method_call(expr_stmt.value.as_ref(), dict_name) => {
                return Some(stmt.range());
            }
            _ => {}
        }
    }
    None
}

/// The bare Name a `for` loop iterates DIRECTLY over — `for x in lst:`
/// — when `lst` is itself the loop's own iterable expression, no
/// `.keys()`/`.values()`/`.items()` view or other wrapping call
/// involved. `Some(name)` only for this exact bare-Name shape (a
/// computed expression, a literal display, or a view call has no
/// single WRITABLE name a body statement could mutate through, so
/// `list_size_changing_mutation_range` has nothing to match against);
/// mirrors `iterated_dict_name`'s own scoping, one level simpler since
/// a list carries no `.keys()`-style view methods to see through.
fn iterated_list_name(iterable: &Expr) -> Option<&str> {
    let Expr::Name(name) = iterable else {
        return None;
    };
    Some(name.id.as_str())
}

/// Whether `expr` is `<list_name>.append(...)` — the one list mutation
/// that unconditionally GROWS the receiver on every call (stdtypes.rst,
/// "list.append(x): Add an item to the end of the list. Equivalent to
/// a[len(a):] = [x]"), called on a bare Name equal to `list_name`.
/// `insert`/`extend`/`+=` also grow a list, but are not read here: this
/// function's own caller (`list_size_changing_mutation_range`) only
/// needs the ONE shape the corpus states as non-terminating —
/// `for x in lst: lst.append(x)`, a self-feeding append that runs the
/// iterator's own internal index into elements the SAME pass just
/// added (`tmp/cpython/Doc/library/stdtypes.rst`'s list iterator has
/// no length snapshot the way a `range(len(...))` counter would) —
/// extending the recognized method set to the wider non-terminating
/// family is a follow-on, not a behavior this one row needs.
fn is_list_growing_append_call(expr: &Expr, list_name: &str) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    if attribute.attr.as_str() != "append" {
        return false;
    }
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return false;
    };
    receiver.id.as_str() == list_name
}

/// Scans `body`'s own TOP-LEVEL statements (the same straight-line
/// scope `dict_size_changing_mutation_range` reads — a nested `.append`
/// one level inside an `if`/`for`/`try` is not proved to run on EVERY
/// reachable pass) for a statement that provably grows `list_name` on
/// every pass: `list_name.append(...)` as an expression statement
/// (`is_list_growing_append_call`). `Some(range)` names the FIRST such
/// statement's own range; `None` when no top-level statement in this
/// body provably appends to the iterated list.
fn list_size_changing_mutation_range(body: &[Stmt], list_name: &str) -> Option<TextRange> {
    for stmt in body {
        if let Stmt::Expr(expr_stmt) = stmt {
            if is_list_growing_append_call(expr_stmt.value.as_ref(), list_name) {
                return Some(stmt.range());
            }
        }
    }
    None
}

/// `some_generator(args...)` — a bare-Name call to a SAME-MODULE `def`
/// (sync or async: `async def stream(): ...` still parses as
/// `StmtFunctionDef`, ruff carries `is_async` as a flag on the def, not
/// a distinct node type) whose body `instances::generator_yields` can
/// read straight-line — `for value in gen(): ...`/`async for value in
/// gen(): ...` both iterate the SAME element sequence a plain call's
/// yields name: compound_stmts.rst, "The `async for` statement" desugars
/// to `TARGET = await type(iter).__anext__(iter)` each pass, and
/// `await` only ever suspends/resumes scheduling — it does not change
/// which values `__anext__` (itself backed by the same generator body's
/// `yield` statements, datamodel.rst's generator-iterator protocol)
/// hands back. `is_async` on `def` is therefore not read here at all:
/// an async generator's yielded elements are the same values a sync
/// generator's would be, only reached through a different awaited
/// protocol. `None` for a non-Name callee, a name with no same-module
/// `def`, any keyword/starred argument (this file does not guess
/// keyword-to-position mapping the way `expressions.rs`'s own
/// `positional_arguments_for_def` does — that helper is private to its
/// module), or a def `generator_yields` itself declines (no top-level
/// `yield`, a conditional yield, a `yield` reached only through a loop
/// or other nested control flow, `yield from` — see that function's own
/// doc for its exact straight-line-body contract).
fn generator_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    if call.arguments.args.iter().any(|argument| matches!(argument, Expr::Starred(_))) {
        return None;
    }
    let table = environment.functions()?;
    let def = table.def(callee.id.as_str())?;
    let mut arguments = Vec::with_capacity(call.arguments.args.len());
    for argument in &call.arguments.args {
        arguments.push(evaluate_expression(argument, environment, kernel));
    }
    let yields = instances::generator_yields(def, &arguments, Some(table), kernel, environment.call_depth())?;
    let mut values = Vec::with_capacity(yields.len());
    for yielded in yields {
        // NOT the same widened acceptance `elements_as_values` now takes
        // for a DISPLAY's own elements: a generator's own yield COUNT is
        // not syntactic the way a tuple/list literal's element count is
        // (`generator_yields` itself already declines any body shape
        // wider than its own two recognized forms before this point is
        // ever reached), so this guard stays at the narrower "a known
        // single scalar or Kind::Null" acceptance — anything wider
        // declines the WHOLE generator's contribution rather than
        // silently narrow it.
        if yielded.kind == Kind::Null || (yielded.kind == Kind::Values && yielded.values.len() == 1) {
            values.push(yielded);
            continue;
        }
        return None;
    }
    Some(values)
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
fn run_body_once(
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
enum StatementOutcome {
    Next,
    Continue,
    Break,
    Returned(Option<AbstractValue>, TextRange),
}

/// Runs exactly one loop-body statement, dispatched by syntactic form.
/// `None` for any statement shape this module does not interpret — the
/// caller (`run_body_once`) propagates that straight into a whole-loop
/// decline.
fn run_statement_once(
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
fn run_assign_once(
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
fn bind_checked(
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
fn run_if_once(
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
fn run_if_once_over_unknown_test(
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
fn outcome_of_body(outcome: BodyOutcome) -> StatementOutcome {
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
fn run_setdefault_append_once(
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_domain::abstract_value::ObjectKey;
    use refined_domain::known_constructors::known_object;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::at_most;
    use refined_sets::refinement_forms::integer as integer_form;
    use refined_sets::refinement_forms::make_refined_set;
    use ruff_python_ast::StmtFunctionDef;
    use ruff_python_parser::parse_module;

    use super::*;

    /// Test-only convenience: a Number-sorted (unsplit-int/float) known
    /// value — `known_number_sorted`'s own doc explains why production
    /// code now always states the true CPython sort instead (`for age
    /// in [10, 20, 30]` binds Integer, not this joined `Number` tag).
    fn known_number(value: f64) -> AbstractValue {
        known_number_sorted(value, PrimitiveKind::Number)
    }

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

    /// Parses `source` as a module body and returns its single top-level
    /// `def` — `iterable_element_sort`'s own test fixture shape, which
    /// needs a `&StmtFunctionDef` directly rather than a loop statement.
    fn parsed_def(source: &str) -> StmtFunctionDef {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        let stmt = module.body.into_iter().next().expect("one top-level statement");
        stmt.function_def_stmt().expect("top-level statement is a def")
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

    fn no_declared() -> HashMap<String, DeclaredRefinement> {
        HashMap::new()
    }

    /// `type Age = Annotated[int, Field(ge=0, le=120)]` — the one
    /// declared refinement this module's judged-write tests need,
    /// built directly (this module's tests construct environments and
    /// declared tables by hand rather than walking a function
    /// signature — matching `check.rs`'s own `age_refinement` test
    /// fixture in spirit).
    fn age_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![integer_form(), at_least(0.0), at_most(120.0)]),
            spelling: "Age".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    fn declared_age(name: &str) -> HashMap<String, DeclaredRefinement> {
        let mut declared = HashMap::new();
        declared.insert(name.to_owned(), age_refinement());
        declared
    }

    /// Runs `loop_final_environment` with no declared table and
    /// discards its judged-fires/else_runs/returned — the shape every
    /// UNIT 1/2 test above cares about is just the post-loop environment.
    fn run(stmt: &Stmt, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<Environment> {
        let declared = no_declared();
        let mut out = Vec::new();
        loop_final_environment(stmt, environment, kernel, &declared, &mut out).map(|answer| answer.environment)
    }

    /// Parses `source` as a module with MULTIPLE top-level statements
    /// (a generator `def` plus the loop under test) and returns the
    /// LAST statement (the loop) alongside the module's own function
    /// table — the generator-call tests need `environment.functions()`
    /// to resolve the callee, which `parsed_loop`'s single-statement
    /// module cannot carry.
    fn parsed_loop_with_functions(source: &str) -> (Stmt, Arc<crate::function_table::FunctionTable>) {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        let table = Arc::new(crate::function_table::function_table(&module));
        let loop_stmt = module.body.into_iter().last().expect("at least one top-level statement");
        (loop_stmt, table)
    }

    /// `run_body_once` over the simplest self-referencing rebind —
    /// `total = total * 2.0` against an exact binding — completes and
    /// binds the doubled exact value: two known operands are the most
    /// determinable arithmetic this module reads, and a decline here is
    /// what turns a non-stabilizing accumulation body into the coarser
    /// "not yet walked" blocker instead of the fixed-point one.
    #[test]
    fn run_body_once_completes_an_exact_self_referencing_rebind() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("total = total * 2.0\n");
        let mut environment = environment_with(&[("total", 1.0)]);
        let declared = no_declared();
        let mut judge_context = JudgeContext {
            declared: &declared,
            newly_declared: HashMap::new(),
            already_fired: std::collections::HashSet::new(),
            fires: Vec::new(),
        };
        let body = [stmt];
        let outcome = run_body_once(&body, &mut environment, &kernel, &mut judge_context);
        assert!(outcome.is_some(), "an exact rebind of two known operands is walkable");
        let total = environment.read("total").expect("total stays bound");
        assert_eq!(total.values, vec![2.0], "1.0 * 2.0 binds exactly 2.0: {total:?}");
    }

    /// `stabilized_join`'s widening, pinned at its own layer: a second
    /// pass that binds a DIFFERENT exact value than the first proves the
    /// name never reached a fixed point, so the join rebinds it to
    /// unknown and names it in `widened` — the list `check.rs`'s
    /// `walk_loop` turns into the body's fixed-point blocker.
    #[test]
    fn stabilized_join_names_the_name_that_never_reaches_a_fixed_point() {
        let Some(kernel) = loaded_kernel() else { return };
        let for_stmt = parsed_loop("for s in samples:\n    total = total * 2.0\n");
        let Stmt::For(for_stmt) = for_stmt else {
            panic!("fixture is a for statement");
        };
        let environment = environment_with(&[("total", 1.0)]);
        let one_pass = environment_with(&[("total", 1.0)]);
        let declared = no_declared();
        let mut judge_context = JudgeContext {
            declared: &declared,
            newly_declared: HashMap::new(),
            already_fired: std::collections::HashSet::new(),
            fires: Vec::new(),
        };
        let element = known_number(0.0);
        let (result, widened) = stabilized_join(
            &environment,
            &one_pass,
            &for_stmt.body,
            for_stmt.target.as_ref(),
            &element,
            &kernel,
            &mut judge_context,
        )
        .expect("both judged passes complete for an exact rebind");
        assert_eq!(widened, vec!["total".to_owned()], "the non-stabilizing name is named");
        assert_eq!(
            result.read("total").map(|v| v.kind),
            Some(Kind::Unknown),
            "the unstable name holds no claim past the loop"
        );
    }

    // --- STEPWISE DIAGNOSTIC CHAIN for showcase.py's own `total = total +
    // amount` shape (invoice_total/refund_everything) — the two pins at
    // check.rs's own test module (`a_plain_rebind_accumulation_over_a_
    // float_list_parameter_walks_the_loop` and its subtracting twin) still
    // fail with the coarser "a for statement is not yet walked" blocker
    // after the join's own numeric-fallback union was fixed to thread a
    // shared `kind_tag` (lattice_operations.rs) — these three tests
    // measure each link of the chain in isolation rather than inferring
    // which one still declines from the pins' own outer failure.

    /// STEP 1: `join_known` directly, on the EXACT pair `stabilized_join`
    /// builds for `total` after the loop's first pass — the pre-loop
    /// binding (`total = 0.0`, `Kind::Values`, Float-tagged) against a
    /// pass-one Set (`total + amount` — the same `[0, +inf)`-shaped
    /// Float-tagged window `transfer_over_sets`'s `TransferAnswerKind::Set`
    /// row answers for a non-negative Float set added to `{0.0}`, built by
    /// hand here rather than round-tripped through the kernel, matching
    /// this test's own narrow question: what the JOIN does with this
    /// shape, not what the transfer computes). Asserts the joined value's
    /// kind, kind_tag, and set forms — the fixed `shared_kind_tag` should
    /// carry `Some(Float)` through onto the union.
    #[test]
    fn join_known_of_preloop_total_and_pass_one_set_keeps_the_float_tag() {
        let preloop_total = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
        let pass_one_total = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
        };
        let joined = refined_domain::lattice_operations::join_known(preloop_total, pass_one_total);
        assert_eq!(joined.kind, Kind::Set, "a Values/Set numeric pair joins to a Set: {joined:?}");
        assert_eq!(
            joined.kind_tag,
            Some(PrimitiveKind::Float),
            "the join must thread the shared Float tag onto the union rather than drop it: {joined:?}"
        );
        assert_eq!(joined.set_kind_tag, SetKindTag::None, "a plain numeric set carries no worn tag: {joined:?}");
    }

    /// STEP 2: feeds STEP 1's joined value as the LEFT operand of the same
    /// `total + amount` binary op the loop's SECOND pass evaluates,
    /// through `binary_arithmetic_value_with_kernel` — the exact function
    /// `evaluate_binop` calls (expressions.rs) — asking what the kernel's
    /// own `transfer_over_sets` path does with a UNION-shaped operand
    /// (`union({0.0}, [0, +inf))`) now that it carries a tag. Two
    /// possibilities distinguish the remaining failing link: if this
    /// answers `Kind::Unknown`, the tag fix alone was not enough — the
    /// kernel's own `transfer` closure declines a Union-form operand
    /// outright (unfolded), and the fix is a fold at the ask site; if this
    /// answers `Kind::Set`/`Kind::Values`, the join/transfer chain itself
    /// is clean and the remaining defect is downstream (`run_assign_once`/
    /// `stabilized_join`'s own comparison, or `bind_checked`).
    #[test]
    fn transfer_over_the_joined_union_set_plus_amount_measures_the_kernel_answer() {
        let Some(kernel) = loaded_kernel() else { return };
        let preloop_total = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
        let pass_one_total = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
        };
        let joined_total = refined_domain::lattice_operations::join_known(preloop_total, pass_one_total);
        let amount = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
        };
        let result = crate::expressions::binary_arithmetic_value_with_kernel(Operator::Add, &joined_total, &amount, &kernel);
        eprintln!("STEP 2 measured answer: {result:?}");
        assert_ne!(
            result.kind,
            Kind::Unknown,
            "if this fails, the kernel's transfer declines the joined UNION operand outright — the \
            remaining fix is a fold (fold_ray_forms) of the joined set before the ask, either at \
            transfer_over_sets' own call site or at join_known's union-building arm: {result:?}"
        );
    }

    /// STEP 3: the same union-shaped joined value, folded through
    /// `refined_sets::refinement_forms::fold_ray_forms` BEFORE the ask —
    /// the Rust twin of the Go adapter's `FoldRayForms`/`CanonicalScalarForms`
    /// hygiene (refinement_forms.go's own doc: "posing the folded question
    /// saves the kernel the redundant forms... while asking for the same
    /// set"). `{0.0} ∪ [0, +inf)` folds to the single ray `[0, +inf)` —
    /// `at_least(0.0)` dominates the singleton, so the fold both simplifies
    /// AND stays semantically identical. If STEP 2 shows the kernel
    /// declining the unfolded union but this step's folded ask determines,
    /// the fix site is confirmed: fold the joined set's forms before
    /// `transfer_over_sets` asks the kernel (or fold at `join_known`'s own
    /// union-building arms directly, so every caller of `join_known`
    /// inherits the same hygiene without a second call site).
    #[test]
    fn transfer_over_the_folded_joined_set_plus_amount_measures_the_kernel_answer() {
        let Some(kernel) = loaded_kernel() else { return };
        let preloop_total = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
        let pass_one_total = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
        };
        let joined_total = refined_domain::lattice_operations::join_known(preloop_total, pass_one_total);
        let folded_forms = refined_sets::refinement_forms::fold_ray_forms(&joined_total.set.forms);
        let folded_total = AbstractValue {
            set: make_refined_set(folded_forms),
            ..joined_total
        };
        let amount = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
        };
        let result = crate::expressions::binary_arithmetic_value_with_kernel(Operator::Add, &folded_total, &amount, &kernel);
        eprintln!("STEP 3 measured answer (folded operand): {result:?}");
        assert_ne!(
            result.kind,
            Kind::Unknown,
            "the folded ray form still declines — the remaining defect is not the union's redundant \
            forms: {result:?}"
        );
    }

    #[test]
    fn for_over_literal_list_sums_and_keeps_last_target_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [60, 61]:\n    total += age\n");
        let environment = environment_with(&[("total", 0.0), ("age", 0.0)]);
        let result = run(&stmt, &environment, &kernel).expect("shape is concrete");
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
        let result = run(&stmt, &environment, &kernel).expect("range(3) is concrete");
        assert_eq!(result.read("total").unwrap().values, vec![3.0]);
        assert_eq!(result.read("i").unwrap().values, vec![2.0]);
    }

    #[test]
    fn while_counter_loop_runs_to_its_own_halt() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("while n < 5:\n    n += 1\n    total += n\n");
        let environment = environment_with(&[("n", 0.0), ("total", 0.0)]);
        let result = run(&stmt, &environment, &kernel).expect("bounded counter");
        // n: 0->1->2->3->4->5, loop stops once n == 5; total sums 1+2+3+4+5
        assert_eq!(result.read("n").unwrap().values, vec![5.0]);
        assert_eq!(result.read("total").unwrap().values, vec![15.0]);
    }

    #[test]
    fn body_with_a_call_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    total = f(x)\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(run(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn for_else_reports_else_runs_true_after_exhaustion() {
        let Some(kernel) = loaded_kernel() else { return };
        // this module no longer runs the else body itself (check.rs
        // owns that, fully judged) — it only reports else_runs: true,
        // since the loop is exhausted with no break.
        let stmt = parsed_loop("for x in [1, 2]:\n    total += x\nelse:\n    done = 1\n");
        let environment = environment_with(&[("total", 0.0), ("done", 0.0)]);
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("body runs, else_runs reported");
        assert_eq!(answer.environment.read("total").unwrap().values, vec![3.0]);
        assert!(answer.else_runs, "the loop exhausts with no break — the else clause runs");
        assert!(answer.returned.is_none(), "no return fires in this row");
        // the orelse body (`done = 1`) never runs HERE — this module
        // only reports else_runs; check.rs walks the orelse. `done`
        // therefore still carries its PRE-loop binding (0.0), proving
        // the executor did not run the else itself.
        assert_eq!(answer.environment.read("done").unwrap().values, vec![0.0]);
    }

    #[test]
    fn while_that_never_resolves_within_the_cap_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // n never changes, so the condition holds forever — must not
        // guess convergence; must decline once the cap is hit
        let stmt = parsed_loop("while n < 5:\n    total += 1\n");
        let environment = environment_with(&[("n", 0.0), ("total", 0.0)]);
        assert!(run(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn empty_literal_list_leaves_target_unbound_when_it_was_never_bound() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in []:\n    total += x\n");
        let environment = environment_with(&[("total", 0.0)]);
        let result = run(&stmt, &environment, &kernel).expect("empty literal list is concrete");
        // x was never assigned by the loop (compound_stmts.html): it
        // carries forward whatever the pre-loop environment held, which
        // here is nothing
        assert!(result.read("x").is_none());
        assert_eq!(result.read("total").unwrap().values, vec![0.0]);
    }

    /// A `list[Wide]`-shaped parameter — the repetition-window seed
    /// `check.rs::seed_parameters` builds for a bare `list[X]` annotation
    /// (`AbstractValue { kind_tag: Some(sort), ..known_set(repeat_of(...))
    /// }`, that function's own doc) — the SAME shape this test builds by
    /// hand so `repetition_window_element_pass` sees exactly what a real
    /// `xs: list[Wide]` parameter would.
    fn wide_list_parameter() -> AbstractValue {
        let element = make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]);
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element, 0, None)]), None, TrustProved, SetKindTag::None)
        }
    }

    /// UNIT: `run_if_once_over_unknown_test`'s own join path — `for x in
    /// xs: if 0 <= x <= 149: out.append(x + 1) else: out.append(0)`
    /// against `xs: list[Wide]` ([0, 200]). Before `run_if_once`'s own
    /// Set-narrowing fallback, this whole loop declined with the coarse
    /// "not yet walked" blocker, because `0 <= x <= 149` never resolves
    /// to one known boolean against a Set-bound `x`. This pins the fixed
    /// mechanism the wave's A10.seed.library/A15.xfer.dedupe/A15.xfer.inject
    /// rows share: the loop now runs, joining both narrowed arms.
    #[test]
    fn if_else_over_a_set_bound_loop_element_joins_both_narrowed_arms() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(
            "for x in xs:\n    if 0 <= x <= 149:\n        out.append(x + 1)\n    else:\n        out.append(0)\n",
        );
        let mut environment = Environment::new(HashSet::from(["xs".to_owned(), "x".to_owned(), "out".to_owned()]));
        environment.bind("xs", wide_list_parameter());
        environment.bind("out", collection_models::list_literal_value(&[]));
        let result = run(&stmt, &environment, &kernel).expect("the if/else over a Set-bound element now runs");
        // `out` stays a known List value (never widened to unknown) —
        // the join produced a real answer, not a decline-shaped stand-in.
        assert_eq!(result.read("out").unwrap().kind, Kind::List);
    }

    /// UNIT: `run_if_once`'s own EXISTING contract is unchanged for a
    /// test this file's narrowing channels do not recognize at all —
    /// `if f():` over a CONCRETE per-element iterate never reaches the
    /// new join fallback (no name in the test is Set-bound), so the
    /// whole loop still declines exactly as before this wave's fix.
    #[test]
    fn unknown_if_test_on_a_concrete_iterate_still_declines_the_whole_loop() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    if f():\n        total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", integer(0.0));
        assert!(run(&stmt, &environment, &kernel).is_none(), "an opaque call still declines — nothing here for assume to narrow");
    }

    /// UNIT: the AugAssign kernel-aware fix — `total += x` where `x` is
    /// the abstract pass's own Set-bound element (never one concrete
    /// number). Before wiring `binary_arithmetic_value_with_kernel` in,
    /// this AugAssign's `updated.kind != Kind::Values` guard declined
    /// the whole loop the moment the operand was Set-shaped — the
    /// mechanism E1.loop/E2.loop/B2.est.loop/B3.est.loop share.
    #[test]
    fn aug_assign_folds_a_set_shaped_operand_through_the_kernel_aware_transfer() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in xs:\n    total += x\n");
        let mut environment = Environment::new(HashSet::from(["xs".to_owned(), "x".to_owned(), "total".to_owned()]));
        environment.bind("xs", wide_list_parameter());
        environment.bind("total", integer(0.0));
        let result = run(&stmt, &environment, &kernel).expect("the Set-shaped accumulation now runs");
        // total widens to a known Set/Values answer, never unknown()
        assert_ne!(result.read("total").unwrap().kind, Kind::Unknown);
    }

    /// UNIT: `list_size_changing_mutation_range`'s own fire —
    /// `for x in lst: lst.append(x)` on a `list[int]`-shaped (repetition-
    /// window) parameter provably never terminates. Pins C5.rangefor's
    /// own mechanism: the fire is recorded and the loop declines, rather
    /// than silently running the abstract pass over a receiver the body
    /// itself keeps growing.
    #[test]
    fn list_appended_to_inside_its_own_for_loop_fires_and_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in lst:\n    lst.append(x)\n");
        let mut environment = Environment::new(HashSet::from(["lst".to_owned(), "x".to_owned()]));
        environment.bind("lst", wide_list_parameter());
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(answer.is_none(), "a self-feeding append never terminates — the loop must decline");
        assert_eq!(out.len(), 1, "exactly one fire names the non-termination: {out:?}");
        assert!(out[0].1.contains("never terminates"), "the fire names non-termination: {:?}", out[0].1);
    }

    #[test]
    fn non_loop_statement_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("total = 1\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(run(&stmt, &environment, &kernel).is_none());
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
        let result = run(&stmt, &environment, &kernel).expect("int list is concrete");
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
        let result = run(&stmt, &environment, &kernel).expect("range is concrete");
        assert_eq!(result.read("total").unwrap().kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn for_over_float_literal_list_binds_the_iterate_as_float_sorted() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1.5, 2.5]:\n    total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", known_values(vec![0.0], PrimitiveKind::Float, TrustProved));
        let result = run(&stmt, &environment, &kernel).expect("float list is concrete");
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
        let result = run(&stmt, &environment, &kernel).expect("if inside body is concrete");
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
        let result = run(&stmt, &environment, &kernel).expect("if/else inside body is concrete");
        assert_eq!(result.read("total").unwrap().values, vec![3.0]);
    }

    #[test]
    fn unknown_if_test_on_any_iteration_declines_the_whole_loop() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    if f():\n        total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", integer(0.0));
        assert!(run(&stmt, &environment, &kernel).is_none());
    }

    // --- break / continue / else_runs (UNIT 2, extended for the LOOP
    // ELSE + DEAD-ELSE LAW) ---

    #[test]
    fn break_stops_the_loop_and_reports_else_runs_false() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(
            "for i in range(3):\n    if i == 1:\n        break\n    total = total + 1\nelse:\n    total = 200\n",
        );
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
        environment.bind("total", integer(0.0));
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("break inside body is concrete");
        // i=0: total=1; i=1: breaks before total += 1 runs
        assert_eq!(answer.environment.read("total").unwrap().values, vec![1.0]);
        assert_eq!(answer.environment.read("i").unwrap().values, vec![1.0]);
        assert!(!answer.else_runs, "a break must report else_runs: false");
    }

    #[test]
    fn continue_skips_the_rest_of_that_iteration_only() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(
            "for i in range(4):\n    if i == 2:\n        continue\n    total = total + i\n",
        );
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = run(&stmt, &environment, &kernel).expect("continue inside body is concrete");
        // 0 + 1 + (skip 2) + 3 = 4
        assert_eq!(result.read("total").unwrap().values, vec![4.0]);
    }

    #[test]
    fn while_break_stops_immediately_and_reports_else_runs_false() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("while n < 5:\n    if n == 2:\n        break\n    n += 1\nelse:\n    n = 200\n");
        let environment = environment_with(&[("n", 0.0)]);
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("while break is concrete");
        assert_eq!(answer.environment.read("n").unwrap().values, vec![2.0]);
        assert!(!answer.else_runs, "a break must report else_runs: false");
    }

    #[test]
    fn a_while_with_no_break_reports_else_runs_true() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("while n < 3:\n    n += 1\nelse:\n    done = 1\n");
        let environment = environment_with(&[("n", 0.0)]);
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("while with no break is concrete");
        assert!(answer.else_runs, "no break ever fires — the else clause runs");
    }

    // --- dict-shaped iteration (UNIT 2) ---

    #[test]
    fn for_over_dict_literal_iterates_the_string_keys() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    last = key\n");
        let environment = Environment::new(HashSet::from(["last".to_owned(), "key".to_owned()]));
        let result = run(&stmt, &environment, &kernel).expect("dict-literal key iteration");
        let last = result.read("last").expect("last stays bound");
        assert_eq!(last.kind_tag, Some(PrimitiveKind::String));
    }

    #[test]
    fn dict_literal_iteration_into_a_declared_int_slot_fires_through_judge() {
        let Some(kernel) = loaded_kernel() else { return };
        // `age: Age = 0` pre-binds age as an Integer; writing a dict key
        // (a String) into it is now JUDGED through assignability::judge
        // — a-statements.py:508's own row — rather than declining the
        // whole loop the way the old cross-family guard did.
        let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    age = key\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "key".to_owned()]));
        environment.bind("age", integer(0.0));
        let declared = declared_age("age");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop still runs concretely — the write fires, it does not decline");
        assert!(!out.is_empty(), "a String into a declared int-sorted Age slot must fire");
        // the refused write keeps the declared set afterward (refused-
        // write law) — a later read of `age` is silent against Age
        let age = answer.environment.read("age").expect("age stays bound to the declared set");
        assert_eq!(age.kind, Kind::Set);
    }

    #[test]
    fn dedupe_by_range_fires_once_per_syntactic_row_across_many_iterations() {
        let Some(kernel) = loaded_kernel() else { return };
        // the loop iterates twice; both keys are strings, so the SAME
        // syntactic write (`age = key`) would fire twice without the
        // dedupe-by-range rule. Only ONE fire must land.
        let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    age = key\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "key".to_owned()]));
        environment.bind("age", integer(0.0));
        let declared = declared_age("age");
        let mut out = Vec::new();
        loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop runs concretely");
        assert_eq!(out.len(), 1, "one syntactic row fires once, however many iterations run: {out:?}");
    }

    #[test]
    fn a_declared_slot_write_that_stays_in_set_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [10, 20]:\n    age = x\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "x".to_owned()]));
        environment.bind("age", integer(0.0));
        let declared = declared_age("age");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop runs concretely");
        assert!(out.is_empty(), "every in-set write must stay silent: {out:?}");
        assert_eq!(answer.environment.read("age").unwrap().values, vec![20.0]);
    }

    #[test]
    fn a_declared_slot_write_of_none_fires_rather_than_declining_the_loop() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:541's own shape: an evaluated (non-literal)
        // iterate that is Kind::Null must still reach bind_checked's own
        // judging — run_assign_once's kind guard used to reject
        // Kind::Null outright (only Values/List/Object were accepted),
        // which declined the WHOLE loop before any judging ever ran.
        let stmt = parsed_loop("for item in [x]:\n    age = item\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "item".to_owned(), "x".to_owned()]));
        environment.bind("age", integer(0.0));
        environment.bind("x", refined_domain::abstract_value::null_value());
        let declared = declared_age("age");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("a Kind::Null iterate must still run the loop concretely, not decline it");
        assert_eq!(out.len(), 1, "None into a non-Optional declared Age slot must fire: {out:?}");
        let age = answer.environment.read("age").expect("age stays bound to the declared set after the refused write");
        assert_eq!(age.kind, Kind::Set);
    }

    // --- RETURN-THROUGH-LOOP CHANNEL ---

    #[test]
    fn a_return_on_the_first_iteration_ends_the_loop_and_carries_the_value_out() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [40, 200]:\n    return age\n");
        let environment = Environment::new(HashSet::from(["age".to_owned()]));
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("a return inside the body is still a concretely-executable shape");
        let (value, _range) = answer.returned.expect("the first iteration's return must be carried out");
        assert_eq!(
            value.expect("return age carries a value, not a bare return").values,
            vec![40.0],
            "only the FIRST iterate's return fires — the loop ends right there"
        );
        assert!(!answer.else_runs, "a return, like a break, never lets the else clause run");
    }

    #[test]
    fn a_return_under_an_if_that_never_triggers_reports_no_return() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [10, 20]:\n    if age == 999:\n        return age\n    total = total + age\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "total".to_owned()]));
        environment.bind("total", integer(0.0));
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop runs concretely — the guarded return never fires");
        assert!(answer.returned.is_none(), "the guard is false on every concrete iterate, so no return fires");
        assert_eq!(answer.environment.read("total").unwrap().values, vec![30.0]);
    }

    #[test]
    fn a_return_under_an_if_that_triggers_on_a_later_iterate_ends_the_loop_there() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [10, 200]:\n    if age > 100:\n        return age\n    total = total + age\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "total".to_owned()]));
        environment.bind("total", integer(0.0));
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop runs concretely up to the returning iterate");
        let (value, _range) = answer.returned.expect("age=200 triggers the guard and returns");
        assert_eq!(value.expect("return age carries a value").values, vec![200.0]);
        // the first iterate (age=10) ran total = total + age BEFORE the
        // second iterate's return fired — the environment still reflects
        // that, even though the returned value is what check.rs judges
        assert_eq!(answer.environment.read("total").unwrap().values, vec![10.0]);
    }

    #[test]
    fn a_bare_return_inside_a_loop_carries_no_value_to_judge() {
        let Some(kernel) = loaded_kernel() else { return };
        // matches check.rs's own walk_return convention: a bare `return`
        // (no expression) judges nothing — this channel must not invent
        // a Null value the way a straight-line bare return never would.
        let stmt = parsed_loop("for age in [40]:\n    return\n");
        let environment = Environment::new(HashSet::from(["age".to_owned()]));
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("a bare return inside the body is still concretely executable");
        let (value, _range) = answer.returned.expect("the bare return must still end the loop and be carried out");
        assert!(value.is_none(), "a bare `return` carries no value to judge, matching walk_return's own convention");
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
        let _ = run(&stmt, &environment, &kernel);
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
        let _ = run(&stmt, &environment, &kernel);
    }

    #[test]
    fn nested_for_in_body_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for x in [1, 2]:\n    for y in [1]:\n        total = total + y\n");
        let environment = environment_with(&[("total", 0.0)]);
        assert!(run(&stmt, &environment, &kernel).is_none());
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
        let result = run(&stmt, &environment, &kernel).expect("kernel bounds the counter");
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
        assert!(run(&stmt, &environment, &kernel).is_none());
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
        assert!(run(&stmt, &environment, &kernel).is_none());
    }

    // --- while body write widens the counter past Kind::Values (UNIT 3) ---

    #[test]
    fn a_refused_write_that_widens_the_counter_fires_and_still_answers_some() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:494's own shape (loop_body_over_ceiling): the
        // single-statement body's own `age = age + 121` fires on
        // iteration 1 against Age's [0, 120] ceiling, and the
        // refused-write law rebinds `age` to the DECLARED set
        // (Kind::Set) — the next condition check (`age < 3`) can no
        // longer read a single known number, so this run must stop
        // WITHOUT declining the whole loop: the fire already proved is
        // a real fact, and check.rs must not ALSO record its own "while
        // statement is not yet walked" blocker on top of it.
        let stmt = parsed_loop("while age < 3:\n    age = age + 121\n");
        let mut environment = Environment::new(HashSet::from(["age".to_owned()]));
        environment.bind("age", integer(0.0));
        let declared = declared_age("age");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("a widened counter after a judged fire is an honest stop, not a decline");
        assert_eq!(out.len(), 1, "the +121 step must fire exactly once: {out:?}");
        let age = answer.environment.read("age").expect("age stays bound to the declared set");
        assert_eq!(age.kind, Kind::Set);
    }

    #[test]
    fn an_unreadable_condition_on_the_first_check_still_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // `age` starts already unbound (not a single known number, and
        // not a Kind::Set the kernel path could pick up either) — the
        // FIRST condition check itself is unreadable, so this is a
        // shape this module never recognized at all, not a widened
        // counter after a judged run. Must still decline.
        let stmt = parsed_loop("while age < 3:\n    age = age + 1\n");
        let environment = Environment::new(HashSet::from(["age".to_owned()]));
        assert!(run(&stmt, &environment, &kernel).is_none());
    }

    // --- async for (UNIT 3) ---

    #[test]
    fn async_for_over_a_known_literal_tuple_runs_concretely() {
        let Some(kernel) = loaded_kernel() else { return };
        // `is_async` alone must never decline — the same literal-tuple
        // shape a plain `for` already runs concretely.
        let stmt = parsed_loop("async for x in (10, 20, 30):\n    total = total + x\n");
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.bind("total", integer(0.0));
        let result = run(&stmt, &environment, &kernel).expect("a known literal tuple runs under async for too");
        assert_eq!(result.read("total").unwrap().values, vec![60.0]);
    }

    #[test]
    fn async_for_over_an_unmodeled_call_receiver_still_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:555's own shape: `stream()` is neither `range`
        // nor a `.values()`/`.items()`/`.keys()` dict-view call —
        // iterable_values cannot read it regardless of is_async, so this
        // must still decline, exactly as an equivalent sync receiver
        // would (body_with_a_call_declines, above).
        let stmt = parsed_loop("async for chunk in stream():\n    age = chunk\n");
        let environment = Environment::new(HashSet::from(["age".to_owned(), "chunk".to_owned()]));
        assert!(run(&stmt, &environment, &kernel).is_none());
    }

    #[test]
    fn for_over_a_same_module_generator_call_iterates_its_straight_line_yields() {
        let Some(kernel) = loaded_kernel() else { return };
        // a same-module `def` whose body is straight-line `yield`
        // statements (no loop, no conditional — the shape
        // `instances::generator_yields` itself reads) is a recognized
        // `for` iterable through `generator_call_values`.
        let (stmt, table) = parsed_loop_with_functions(concat!(
            "def gen():\n",
            "    yield 10\n",
            "    yield 20\n",
            "    yield 30\n",
            "for x in gen():\n",
            "    total = total + x\n",
        ));
        let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
        environment.set_functions(table);
        environment.bind("total", integer(0.0));
        let result = run(&stmt, &environment, &kernel).expect("a straight-line generator's yields are known iterates");
        assert_eq!(result.read("total").unwrap().values, vec![60.0]);
        assert_eq!(result.read("x").unwrap().values, vec![30.0], "the target stays bound to the last yield");
    }

    #[test]
    fn for_over_a_loop_bodied_generator_iterates_its_yields() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:547's own `stream` shape: the `yield` is
        // nested inside a single `for` loop over a literal iterable —
        // `generator_yields` reads exactly this shape, so the consuming
        // loop iterates the yields concretely.
        let (stmt, table) = parsed_loop_with_functions(concat!(
            "def stream():\n",
            "    for value in (10, 20, 30):\n",
            "        yield value\n",
            "for chunk in stream():\n",
            "    age = chunk\n",
        ));
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "chunk".to_owned()]));
        environment.set_functions(table);
        let answer = run(&stmt, &environment, &kernel).expect("the yields iterate concretely");
        assert_eq!(answer.read("age").unwrap().values, vec![30.0]);
    }

    // --- setdefault(...).append(...) composition (UNIT 3) ---

    #[test]
    fn setdefault_append_extends_an_absent_key_with_the_default_and_the_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [40]:\n    grouped.setdefault(\"young\", []).append(age)\n");
        let mut environment = Environment::new(HashSet::from(["grouped".to_owned(), "age".to_owned()]));
        environment.bind("grouped", collection_models::dict_literal_value(&[], &[]));
        let result = run(&stmt, &environment, &kernel).expect("the chained mutation is a recognized statement shape");
        let grouped = result.read("grouped").expect("grouped stays bound");
        assert_eq!(grouped.kind, Kind::Object);
        assert_eq!(grouped.keys.len(), 1);
        assert_eq!(grouped.keys[0].name, "young");
        assert_eq!(grouped.keys[0].value.items.len(), 1);
        assert_eq!(grouped.keys[0].value.items[0].values, vec![40.0]);
    }

    #[test]
    fn setdefault_append_appends_to_a_present_key_without_losing_earlier_entries() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for age in [40, 200]:\n    grouped.setdefault(\"young\", []).append(age)\n");
        let mut environment = Environment::new(HashSet::from(["grouped".to_owned(), "age".to_owned()]));
        environment.bind("grouped", collection_models::dict_literal_value(&[], &[]));
        let result = run(&stmt, &environment, &kernel).expect("two iterates over the same key both compose");
        let grouped = result.read("grouped").expect("grouped stays bound");
        assert_eq!(grouped.keys.len(), 1, "one key, both appends land on it");
        assert_eq!(
            grouped.keys[0].value.items.iter().map(|v| v.values[0]).collect::<Vec<_>>(),
            vec![40.0, 200.0]
        );
    }

    #[test]
    fn setdefault_append_over_a_ternary_key_groups_by_the_per_iterate_branch() {
        let Some(kernel) = loaded_kernel() else { return };
        // c-reads-and-values.py:1007's own dict_groupby shape: the key
        // expression is a ternary that reads differently PER ITERATE.
        let stmt = parsed_loop(
            "for age in [40, 200]:\n    grouped.setdefault(\"old\" if age > 100 else \"young\", []).append(age)\n",
        );
        let mut environment = Environment::new(HashSet::from(["grouped".to_owned(), "age".to_owned()]));
        environment.bind("grouped", collection_models::dict_literal_value(&[], &[]));
        let result = run(&stmt, &environment, &kernel).expect("the ternary key resolves per iterate");
        let grouped = result.read("grouped").expect("grouped stays bound");
        assert_eq!(grouped.keys.len(), 2, "40 groups under young, 200 groups under old");
        let young = grouped.keys.iter().find(|k| k.name == "young").expect("young key exists");
        assert_eq!(young.value.items[0].values, vec![40.0]);
        let old = grouped.keys.iter().find(|k| k.name == "old").expect("old key exists");
        assert_eq!(old.value.items[0].values, vec![200.0]);
    }

    // --- abstract_element_sort_pass: ABSTRACT SORT-ELEMENT LOOP PASS ---

    /// a-statements.py's own `async_for_over_stream`/`stream` shape:
    /// `stream() -> AsyncIterator[int]` is opaque (`raise
    /// NotImplementedError` — `iterable_values` cannot read any concrete
    /// element), but the return annotation still states the element's own
    /// sort. `age = chunk` under a DECLARED `age: Age` slot must fire —
    /// the one-pass judged write, proof the abstract pass runs the body
    /// through the same `bind_checked`/`assignability::judge` seam a
    /// concrete pass uses, not merely binding the target and stopping.
    #[test]
    fn abstract_element_sort_pass_fires_a_judged_write_inside_the_one_pass_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let (stmt, table) = parsed_loop_with_functions(concat!(
            "async def stream() -> AsyncIterator[int]:\n",
            "    raise NotImplementedError\n",
            "    yield 0\n",
            "async for chunk in stream():\n",
            "    age = chunk\n",
        ));
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "chunk".to_owned()]));
        environment.set_functions(table);
        environment.bind("age", integer(0.0));
        let declared = declared_age("age");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the AsyncIterator[int] annotation carries an abstract element sort even though the body declines concretely");
        assert_eq!(out.len(), 1, "{:?}", out.iter().map(|(_, message)| message).collect::<Vec<_>>());
        assert!(out[0].1.contains("Age"), "{}", out[0].1);
        // the refused write keeps the DECLARED set afterward (the same
        // refused-write law every other judged sink uses) — never the
        // whole-number element sort itself.
        assert_eq!(answer.environment.read("age").unwrap().kind, Kind::Set);
    }

    /// The abstract pass's own JOIN semantics: the answer is
    /// `join(pre-loop environment, one-pass environment)`, stating the
    /// loop's real zero-or-more possibility rather than assuming the body
    /// definitely ran — a name the body does NOT touch (`untouched`)
    /// still reads its PRE-LOOP value afterward, since both sides of the
    /// join agree on it.
    #[test]
    fn abstract_element_sort_pass_joins_the_pre_loop_and_one_pass_environments() {
        let Some(kernel) = loaded_kernel() else { return };
        let (stmt, table) = parsed_loop_with_functions(concat!(
            "async def stream() -> AsyncIterator[int]:\n",
            "    raise NotImplementedError\n",
            "    yield 0\n",
            "async for chunk in stream():\n",
            "    age = chunk\n",
        ));
        let mut environment = Environment::new(HashSet::from(["age".to_owned(), "chunk".to_owned(), "untouched".to_owned()]));
        environment.set_functions(table);
        environment.bind("age", integer(0.0));
        environment.bind("untouched", integer(7.0));
        let result = run(&stmt, &environment, &kernel).expect("the abstract pass answers instead of declining");
        assert_eq!(result.read("untouched").unwrap().values, vec![7.0], "a name neither side's own pass touches survives the join unchanged");
    }

    /// `iterable_element_sort` itself: `AsyncIterator[int]` reads as the
    /// Integer-tagged whole-number set — the same `whole_integers()` shape
    /// `return_sort_fallback` builds for a bare `-> int`, one subscript
    /// level up.
    #[test]
    fn iterable_element_sort_reads_asynciterator_int_as_the_whole_number_set() {
        let def = parsed_def("async def stream() -> AsyncIterator[int]:\n    raise NotImplementedError\n    yield 0\n");
        let element_sort = iterable_element_sort(&def).expect("AsyncIterator[int] states an element sort");
        assert_eq!(element_sort.kind, Kind::Set);
        assert_eq!(element_sort.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// A return annotation that is not one of `AsyncIterator`/`Iterator`/
    /// `Iterable` (a bare `-> int`, the RETURN value's own sort, never an
    /// element sort) reads as `None` — this fallback never confuses the
    /// two claims.
    #[test]
    fn iterable_element_sort_declines_a_bare_return_annotation() {
        let def = parsed_def("def counted() -> int:\n    return 3\n");
        assert!(iterable_element_sort(&def).is_none());
    }

    // --- body-local AnnAssign reuses an already-declared alias's own
    // DeclaredRefinement by SPELLING (UNIT 4) ---

    /// g-binding-destructuring.py:191-193's own shape: the for-target is
    /// a TUPLE UNPACK (`for _, over_value in over_items:`), and the
    /// body's first statement is an `AnnAssign` (`bad: Age = over_value`)
    /// whose target was never bound before this loop — `declared` (the
    /// pre-loop snapshot) has no entry for `bad`, only for `total` (an
    /// EARLIER `total: Age = 0` in the same enclosing function). The
    /// alias-spelling reuse must still fire the out-of-range write.
    #[test]
    fn body_local_ann_assign_reuses_an_alias_already_declared_under_a_different_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(concat!(
            "for _, over_value in over_items:\n",
            "    bad: Age = over_value\n",
        ));
        let mut environment = Environment::new(HashSet::from([
            "over_items".to_owned(),
            "_".to_owned(),
            "over_value".to_owned(),
            "bad".to_owned(),
        ]));
        let pairs = known_list(
            vec![
                known_list(vec![known_string("a"), integer(200.0)], TrustProved),
                known_list(vec![known_string("b"), integer(201.0)], TrustProved),
            ],
            TrustProved,
        );
        environment.bind("over_items", pairs);
        // `declared` carries Age only under "total" — "bad" is not a key
        // here at all, matching the pre-loop snapshot's real shape.
        let declared = declared_age("total");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the tuple-unpack target binds and the loop runs concretely");
        assert_eq!(out.len(), 1, "the 200/201 writes into Age must fire, deduped to one syntactic row: {out:?}");
        assert!(out[0].1.contains("Age"), "{}", out[0].1);
        let bad = answer.environment.read("bad").expect("bad stays bound to the declared set after the refused write");
        assert_eq!(bad.kind, Kind::Set);
    }

    /// The reuse is scoped to a MATCHING alias spelling only: a
    /// body-local AnnAssign under an annotation that names NO alias
    /// already present in `declared` stays unjudged, exactly as before —
    /// this is not a general "annotation reading" fallback.
    #[test]
    fn body_local_ann_assign_under_an_unmatched_alias_stays_unjudged() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(concat!(
            "for x in [200]:\n",
            "    bad: Unrelated = x\n",
        ));
        let environment = Environment::new(HashSet::from(["x".to_owned(), "bad".to_owned()]));
        let declared = declared_age("total");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop still runs concretely — an unmatched annotation never declines it");
        assert!(out.is_empty(), "no declared entry matches 'Unrelated' by spelling, so nothing fires: {out:?}");
        assert_eq!(answer.environment.read("bad").unwrap().values, vec![200.0], "bad binds unjudged, unchanged from before this fix");
    }

    /// A body-local AnnAssign target that IS already a key in `declared`
    /// (a name the pre-loop snapshot already recorded, then rewritten
    /// with a fresh `x: Age = …` inside the SAME loop body) keeps reading
    /// `declared`'s own entry — `newly_declared` never shadows it, since
    /// `bind_checked` tries `declared` first.
    #[test]
    fn a_redeclared_name_already_in_declared_is_not_overridden_by_the_reuse_table() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop(concat!("for x in [200]:\n", "    total: Age = x\n",));
        let mut environment = Environment::new(HashSet::from(["x".to_owned(), "total".to_owned()]));
        environment.bind("total", integer(0.0));
        let declared = declared_age("total");
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
            .expect("the loop runs concretely");
        assert_eq!(out.len(), 1, "the redeclared write still fires against Age's own declared entry: {out:?}");
        let total = answer.environment.read("total").expect("total stays bound to the declared set");
        assert_eq!(total.kind, Kind::Set);
    }

    // --- iterator invalidation: dict-changed-size-during-iteration ---

    /// A known two-entry dict, `{"a": 10, "b": 20}` — the fixture every
    /// iterator-invalidation test below iterates over.
    fn two_entry_dict() -> AbstractValue {
        known_object(
            vec![
                ObjectKey { name: "a".to_owned(), numeric: false, value: integer(10.0) },
                ObjectKey { name: "b".to_owned(), numeric: false, value: integer(20.0) },
            ],
            None,
            true,
            TrustProved,
            false,
        )
    }

    /// `for k in counts: del counts[k]` — CPython's own canonical
    /// iterator-invalidation shape (library/stdtypes.rst's dict-views
    /// note) provably raises `RuntimeError` on the first pass, never
    /// runs the body's own `del`, and never returns a post-loop
    /// environment — `loop_final_environment` answers `None`, with the
    /// raise itself recorded in `out`.
    #[test]
    fn deleting_the_iterated_dicts_own_key_inside_the_loop_provably_raises() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for k in counts:\n    del counts[k]\n");
        let mut environment = Environment::new(HashSet::from(["k".to_owned(), "counts".to_owned()]));
        environment.bind("counts", two_entry_dict());
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(answer.is_none(), "the loop itself declines once the raise is proved");
        assert_eq!(out.len(), 1, "exactly one raise is recorded: {out:?}");
        assert!(out[0].1.contains("RuntimeError"), "{}", out[0].1);
        assert!(out[0].1.contains("'counts'"), "{}", out[0].1);
        assert!(out[0].1.contains("changed size during"), "{}", out[0].1);
    }

    /// The identical shape over `.keys()`/`.values()`/`.items()` view
    /// calls — the raise is proved from the dict's OWN size change, not
    /// from which view the loop happens to iterate.
    #[test]
    fn deleting_the_iterated_dicts_own_key_through_a_keys_view_provably_raises() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for k in counts.keys():\n    del counts[k]\n");
        let mut environment = Environment::new(HashSet::from(["k".to_owned(), "counts".to_owned()]));
        environment.bind("counts", two_entry_dict());
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(answer.is_none());
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(out[0].1.contains("RuntimeError"), "{}", out[0].1);
    }

    /// `.pop(k)` inside the loop body is the SAME provable raise as an
    /// explicit `del` — both provably change the dict's own size.
    #[test]
    fn popping_the_iterated_dicts_own_key_inside_the_loop_provably_raises() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for k in counts:\n    counts.pop(k)\n");
        let mut environment = Environment::new(HashSet::from(["k".to_owned(), "counts".to_owned()]));
        environment.bind("counts", two_entry_dict());
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(answer.is_none());
        assert_eq!(out.len(), 1, "{out:?}");
        assert!(out[0].1.contains("RuntimeError"), "{}", out[0].1);
    }

    /// `counts[k] = v` — reassigning an EXISTING key inside the loop
    /// never changes the dict's own size, so CPython never raises here;
    /// this shape stays outside the provable-raise scope on purpose
    /// (`is_dict_size_changing_method_call`'s own doc: only `pop`/
    /// `popitem`/`clear` are unconditionally size-changing). The loop
    /// still runs concretely to completion — no raise, no decline.
    #[test]
    fn reassigning_an_existing_key_inside_the_loop_does_not_raise() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for k in counts:\n    counts[k] = 0\n");
        let mut environment = Environment::new(HashSet::from(["k".to_owned(), "counts".to_owned()]));
        environment.bind("counts", two_entry_dict());
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(out.is_empty(), "reassigning an existing key never changes size, so no raise fires: {out:?}");
        let _ = answer;
    }

    /// An EMPTY dict never runs the loop body at all, so a `del` inside
    /// it never executes and never raises — matching real CPython: `for
    /// k in {}: del counts[k]` completes with zero iterations.
    #[test]
    fn an_empty_dict_never_raises_even_with_a_size_changing_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for k in counts:\n    del counts[k]\n");
        let mut environment = Environment::new(HashSet::from(["k".to_owned(), "counts".to_owned()]));
        environment.bind("counts", known_object(vec![], None, true, TrustProved, false));
        let declared = no_declared();
        let mut out = Vec::new();
        let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(out.is_empty(), "an empty dict runs zero iterations, so nothing raises: {out:?}");
        assert!(answer.is_some(), "an empty-dict loop still completes concretely");
    }

    /// A `del`/`.pop` on a DIFFERENT name than the one iterated never
    /// raises this construct — the mutation must target the SAME dict
    /// the loop reads from.
    #[test]
    fn mutating_a_different_dict_inside_the_loop_does_not_raise() {
        let Some(kernel) = loaded_kernel() else { return };
        let stmt = parsed_loop("for k in counts:\n    del other[k]\n");
        let mut environment =
            Environment::new(HashSet::from(["k".to_owned(), "counts".to_owned(), "other".to_owned()]));
        environment.bind("counts", two_entry_dict());
        environment.bind("other", two_entry_dict());
        let declared = no_declared();
        let mut out = Vec::new();
        let _ = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
        assert!(out.is_empty(), "a different dict's own mutation is not this construct: {out:?}");
    }
}
