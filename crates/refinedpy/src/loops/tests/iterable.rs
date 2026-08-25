//! Iterables and bindings: what a loop's own iterable expression
//! resolves to — re.finditer, dict-literal key iteration, same-
//! module generator calls, the abstract element-sort pass over an
//! opaque async iterable, `iterable_element_sort` itself, and the
//! dict-changed-size-during-iteration raise.

use super::*;

/// A3.xfer.matchall.py's own `finditer_outside`: `for m in
/// re.finditer(r"\d+", s): return m.group(0)`. Pins `finditer_call_
/// values`: the loop binds `m` to ONE representative match-object
/// value (`string_models::match_object_value(r"\d+")`), and the loop
/// body's own `m.group(0)` read — through the `.group` dispatch —
/// resolves to the `\d+` grammar the match SPANS rather than
/// declining the whole loop.
#[test]
fn for_over_re_finditer_binds_the_target_to_a_representative_match_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for m in re.finditer(r\"\\d+\", s):\n    matched = m.group(0)\n");
    let mut environment = Environment::new(HashSet::from(["m".to_owned(), "matched".to_owned()]));
    environment.bind("s", known_string("abc123"));
    let result = run(&stmt, &environment, &kernel).expect("a literal-pattern finditer loop is concrete");
    let matched = result.read("matched").expect("matched is bound after the loop runs");
    assert_eq!(matched.kind, Kind::Set, "m.group(0) reads the \\d+ grammar as a String-sorted set: {matched:?}");
    assert_eq!(matched.kind_tag, Some(PrimitiveKind::String));
}

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

// --- iterator invalidation: dict-changed-size-during-iteration ---

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
