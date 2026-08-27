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
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::StmtFor;
use crate::env::Environment;
use crate::expressions::call_one_argument_expression;
use crate::expressions::evaluate_expression;
use crate::instances;
use crate::summaries::iterable_element_sort;

use super::JudgeContext;
use super::LoopAnswer;
use super::bind_target::bind_for_target;
use super::body_once::BodyOutcome;
use super::body_once::run_body_once;
use super::iterable::body_can_resize_iterated_list;
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
        // STALE SNAPSHOT: `elements` was read from the iterable BEFORE
        // the first pass, but a list's iterator holds the live list and
        // re-reads its length on every `__next__` (stdtypes.rst,
        // "Iterator Types"). A body that can append to or remove from
        // the very list being iterated therefore visits a different
        // sequence than this snapshot describes, and stepping the
        // snapshot would state an exact element count CPython does not
        // produce — so this SNAPSHOT walk stands aside for the LIVE one
        // (`live_list_element_walk`), which re-reads the list each pass
        // and marches the index the way stdtypes.rst's own mutable-
        // sequence-iterator paragraph states; that function declines to
        // the abstract passes below whenever a pass stops being exactly
        // readable. Narrower than `list_size_changing_mutation_range`,
        // which proves a growth on EVERY pass to name non-termination: a
        // resize reached on ANY pass, however deeply guarded, already
        // invalidates the snapshot's count.
        if let Some(list_name) = iterated_list_name(for_stmt.iter.as_ref()) {
            if body_can_resize_iterated_list(&for_stmt.body, list_name) {
                // The snapshot is stale, but the LIVE list may still be
                // exactly readable at every step — `live_list_element_walk`
                // re-reads the name each pass and steps the index forward
                // the way stdtypes.rst says the iterator does, answering
                // exactly when every pass stays concrete.
                return live_list_element_walk(for_stmt, list_name, environment, kernel, judge_context);
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
    // `windowed_range_element_pass` runs BEFORE
    // `repetition_window_element_pass`: both can read a `for i in
    // range(n)` whose stop is not one known scalar, but they answer
    // different element sets from the same iterable. The range pass
    // reads the STOP binding itself and answers `[0, max(stop) - 1]`;
    // the repetition pass reads the value `range(n)` evaluates to,
    // which `range_expression_value`'s one-argument fallback states as
    // the sort-only window `integer ∧ [0, +inf)` — every element the
    // range pass admits and more. Consulting the general window reader
    // first would discard the stop's own upper bound on every counted
    // loop over a bounded parameter, so the specific reader is asked
    // first and the general one keeps every iterable it alone reads.
    abstract_element_sort_pass(for_stmt, environment, kernel, judge_context)
        .or_else(|| custom_iterator_element_pass(for_stmt, environment, kernel, judge_context))
        .or_else(|| groupby_element_pass(for_stmt, environment, kernel, judge_context))
        .or_else(|| windowed_range_element_pass(for_stmt, environment, kernel, judge_context))
        .or_else(|| repetition_window_element_pass(for_stmt, environment, kernel, judge_context))
}

/// `for key, group in groupby(<iterable>[, key=<callable>]):` —
/// A8.seed.library's own `group_by_parity`. library/itertools.rst,
/// `groupby(iterable, key=None)`: "Make an iterator that returns
/// consecutive keys and groups from the *iterable*. The *key* is a
/// function computing a key value for each element. If not specified or
/// is ``None``, *key* defaults to an identity function and returns the
/// element unchanged." The same entry states what a group holds: "The
/// returned group is itself an iterator that shares the underlying
/// iterable," and its own equivalent-code block yields values drawn
/// straight from that iterable — `[list(g) for k, g in
/// groupby('AAAABBBCCD')] → AAAA BBB CC D`.
///
/// Over an iterable this domain reads only as a REPETITION WINDOW (an
/// unread `list[X]` parameter, possibly through `sorted(...)`, which
/// leaves the window exactly as it was — `sorted_over_star_with_
/// keywords`' own doc), no exact grouping exists to walk: the element
/// values decide both how many groups there are and where the breaks
/// fall, and none of that is read. What the entry DOES pin, given the
/// element set:
///
/// - the KEY is `key(element)` for some element of the window, so the
///   key set is the key function's IMAGE over that element set — read
///   here by calling the key function's own raw expression once against
///   the element (`call_one_argument_expression`, the same seam
///   `map`/`filter` fold with). The fixture's key is `lambda x: "even"
///   if x % 2 == 0 else "odd"`, whose ternary joins both arms into the
///   closed two-member set {"even", "odd"} — an exact image, not a
///   sort. An ABSENT `key=` is the entry's own identity default, so the
///   image is the element set itself.
/// - the GROUP is a sequence of elements of the SAME iterable, so it is
///   a repetition window over that same element set. Its own item count
///   is unstated (a group holds at least one element — every group
///   `groupby` emits is non-empty, since a group is created by an
///   element — so the window starts at 1).
///
/// The target must be the two-name tuple the entry's own signature
/// produces (`for k, g in ...`); the body then runs ONE judged pass over
/// that binding and joins with the pre-loop environment, the same
/// zero-or-more honesty every other abstract pass in this file states.
///
/// `None` when the iterable is not a `groupby(...)` call, when the
/// receiver is not a bare repetition window, when the target is not a
/// two-name tuple, when a `key=` is present but its image cannot be
/// read, or when the one abstract pass hits a statement shape
/// `run_body_once` does not recognize.
pub(super) fn groupby_element_pass(
    for_stmt: &StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    let Expr::Call(call) = for_stmt.iter.as_ref() else {
        return None;
    };
    // `groupby(...)` reaches a module either bare (`from itertools import
    // groupby`) or qualified (`itertools.groupby(...)`); neither spelling
    // may be shadowed by a local binding, the same gate every other
    // module-call row in this checker keeps.
    let bare = matches!(call.func.as_ref(), Expr::Name(name) if name.id.as_str() == "groupby")
        && environment.read("groupby").is_none();
    let qualified = match call.func.as_ref() {
        Expr::Attribute(attribute) if attribute.attr.as_str() == "groupby" => {
            matches!(attribute.value.as_ref(), Expr::Name(module) if module.id.as_str() == "itertools")
                && environment.read("itertools").is_none()
        }
        _ => false,
    };
    if !bare && !qualified {
        return None;
    }
    // The element set the whole reading rests on: the iterable's own
    // repetition window, read back to one element.
    let [iterable_expr] = &*call.arguments.args else {
        return None;
    };
    let iterable = evaluate_expression(iterable_expr, environment, kernel);
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    let repeated = as_repetition(&iterable.set)?;
    let grade = trust_level_of(&iterable);
    let element = AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(repeated.element.clone(), None, grade, SetKindTag::None)
    };

    // The KEY: the key function's image over one element, or — with no
    // `key=` at all — the entry's own identity default, the element set
    // itself. `key=None` is spelled explicitly in the signature and means
    // the same identity function, so it reads the same way.
    let mut key_expression: Option<&Expr> = None;
    for keyword in &call.arguments.keywords {
        let name = keyword.arg.as_ref()?;
        if name.id.as_str() != "key" {
            return None;
        }
        key_expression = Some(&keyword.value);
    }
    let key_value = match key_expression {
        None => element.clone(),
        Some(Expr::NoneLiteral(_)) => element.clone(),
        Some(expression) => call_one_argument_expression(expression, &element, environment, kernel)?,
    };
    if key_value.kind == Kind::Unknown {
        return None;
    }

    // The GROUP: elements of the same iterable, at least one of them.
    let group_value = AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(repetition(repeated.element, 1, None), None, grade, SetKindTag::None)
    };

    let pair = known_list(vec![key_value, group_value], grade);
    let mut one_pass = environment.fork();
    if !bind_for_target(for_stmt.target.as_ref(), &pair, &mut one_pass) {
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
        &pair,
        kernel,
        judge_context,
    )?;
    Some(LoopAnswer { environment: joined, else_runs: true, returned: None, widened_names })
}

/// `for x in lst:` where the body MUTATES `lst`'s own length — the case
/// the snapshot walk above declines because the pre-loop element list no
/// longer describes what the loop visits. stdtypes.rst, "Common Sequence
/// Operations," states the mechanism exactly: "Forward and reversed
/// iterators over mutable sequences access values using an index. That
/// index will continue to march forward (or backward) even if the
/// underlying sequence is mutated. The iterator terminates only when an
/// :exc:`IndexError` or a :exc:`StopIteration` is encountered (or when
/// the index drops below zero)."
///
/// So the loop is not a walk over a snapshot at all — it is an index
/// starting at 0, incremented after each pass, reading the LIVE list and
/// ending when the index runs past the live length. This function runs
/// exactly that: each pass re-reads `list_name` from the CURRENT
/// environment, stops when the index is at or past the live item count
/// (the `IndexError` the paragraph names), binds position `index`, runs
/// the body, and steps. An element appended by pass `i` is therefore
/// visited by a later pass, and an element removed before the index
/// reaches it is never visited — both the paragraph's own consequence,
/// not an extra rule.
///
/// Every step must stay EXACT for the answer to stand. The live receiver
/// must read back as a `Kind::List` with known items on every pass (an
/// abstract window carries no position to index and no count to stop at),
/// and each pass's body must run through the same judged `run_body_once`
/// every concrete walk uses. Anything else declines to `None` and the
/// caller's abstract passes take the loop.
///
/// NON-TERMINATION is not a cap here and is not this function's to
/// invent: a body that appends on EVERY reachable pass never lets the
/// index catch the length, and `repetition_window_element_pass`'s own
/// `list_size_changing_mutation_range` already names that shape and
/// fires `list_never_terminates_self_append` for it. This function walks
/// a body whose appends are CONDITIONAL, so the live length stops
/// growing once the condition stops holding and the index reaches it —
/// A7.xfer.iterate's own `if len(lst) < 2: lst.append(2)`, which grows
/// the list once and then leaves the index to run off the end at 2. The
/// unconditional shape is already fired on before this runs, so a body
/// that keeps growing past the point where this function can still read
/// each step exactly falls out through the ordinary decline paths below
/// rather than being counted.
fn live_list_element_walk(
    for_stmt: &StmtFor,
    list_name: &str,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<LoopAnswer> {
    // The unconditional self-append is the non-terminating shape, named
    // and fired by `repetition_window_element_pass`'s own row. Stepping
    // it here would run the index behind a length that always outruns
    // it, so it is refused before the first pass rather than stepped.
    if list_size_changing_mutation_range(&for_stmt.body, list_name).is_some() {
        return None;
    }
    let mut current = environment.fork();
    let mut broke = false;
    let mut index: usize = 0;
    loop {
        // re-read the LIVE list, as the iterator's own `__next__` does
        let live = current.read(list_name)?;
        if live.kind != Kind::List {
            return None;
        }
        // the index has marched past the live length — the `IndexError`
        // the paragraph names, which is where the iterator stops
        if index >= live.items.len() {
            break;
        }
        let element = live.items[index].clone();
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
        index += 1;
    }
    Some(LoopAnswer { environment: current, else_runs: !broke, returned: None, widened_names: Vec::new() })
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
