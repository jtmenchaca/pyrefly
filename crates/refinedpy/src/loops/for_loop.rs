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
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::StmtFor;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances;
use crate::summaries::iterable_element_sort;

use super::JudgeContext;
use super::LoopAnswer;
use super::bind_target::bind_for_target;
use super::body_once::BodyOutcome;
use super::body_once::run_body_once;
use super::iterable::dict_size_changing_mutation_range;
use super::iterable::iterable_values;
use super::iterable::iterated_dict_name;
use super::iterable::iterated_list_name;
use super::iterable::list_size_changing_mutation_range;
use super::widen::stabilized_join;

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
pub(super) fn for_loop_final_environment(
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
pub(super) fn custom_iterator_element_pass(
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
pub(super) fn repetition_window_element_pass(
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
pub(super) fn windowed_range_element_pass(
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
pub(super) fn abstract_element_sort_pass(
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
