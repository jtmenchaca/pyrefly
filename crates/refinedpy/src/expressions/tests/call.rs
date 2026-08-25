use super::*;

#[test]
fn test_len_call() {
    let Some(value) = eval("len([1, 2, 3])") else { return };
    assert_eq!(value.values, vec![3.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_range_one_argument_materializes_stop_exclusive() {
    let Some(value) = eval("range(3)") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![0.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

#[test]
fn test_range_two_arguments_start_stop() {
    let Some(value) = eval("range(2, 5)") else { return };
    assert_eq!(
        value.items,
        vec![
            known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![4.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

#[test]
fn test_range_len_over_200() {
    // c-reads-and-values.py's dict_size row: {str(i): i for i in
    // range(200)} — the length is exactly 200
    let Some(value) = eval("len(range(200))") else { return };
    assert_eq!(value.values, vec![200.0]);
}

#[test]
fn test_range_zero_step_declines() {
    let Some(value) = eval("range(0, 10, 0)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `reduce(lambda acc, age: acc + age, [100, 101], 0)` folds
/// concretely: 0 + 100 + 101 == 201.
#[test]
fn test_reduce_with_lambda_and_seed_folds_concretely() {
    let Some(value) = eval("reduce(lambda acc, age: acc + age, [100, 101], 0)") else { return };
    assert_eq!(value.values, vec![201.0]);
}

/// `reduce` with no `initializer` on a NON-empty iterable seeds the
/// accumulator with the FIRST element (functools.rst's own row).
#[test]
fn test_reduce_without_initializer_seeds_from_the_first_element() {
    let Some(value) = eval("reduce(lambda acc, age: acc + age, [10, 20, 30])") else { return };
    assert_eq!(value.values, vec![60.0]);
}

/// `reduce`'s `function` argument resolving to a same-module `def`
/// (not only a lambda) folds through `summaries::call_result`.
#[test]
fn test_reduce_with_same_module_def_function() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def add(acc, age):\n    return acc + age\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("reduce(add, [10, 20], 0)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![30.0]);
}

/// `reduce` over a non-List iterable declines.
#[test]
fn test_reduce_non_list_iterable_declines() {
    let Some(value) = eval("reduce(lambda acc, age: acc + age, 5, 0)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `eval("40")` is execution-verified to answer the exact int 40
/// (`eval("40") == 40`, `type(eval("40")) is int`), but `eval` is a
/// host boundary this file never interprets: the answer is the
/// whole-number SET (sort-only), never the exact value — the same
/// posture `math.sqrt`'s approximated family and a declined
/// same-module call's return-annotation fallback both take.
#[test]
fn test_eval_of_a_plain_int_literal_string_answers_the_whole_number_set() {
    let Some(value) = eval("eval(\"40\")") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `eval("3.5")` answers `float_sorted_unknown()`, never the exact
/// float — the same sort-only posture as the int-literal row above.
#[test]
fn test_eval_of_a_plain_float_literal_string_answers_float_sorted_unknown() {
    let Some(value) = eval("eval(\"3.5\")") else { return };
    assert_eq!(value, float_sorted_unknown());
}

/// `eval("-7")` still recognizes the leading-sign int spelling and
/// answers the whole-number set (never the exact -7).
#[test]
fn test_eval_of_a_negative_int_literal_string_answers_the_whole_number_set() {
    let Some(value) = eval("eval(\"-7\")") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// The whole-number set `eval` answers genuinely admits a value the
/// Age alias refuses (200, 121, negatives, …) — the CONTAINMENT
/// question the corpus's `call_eval_bare` row leans on.
#[test]
fn test_eval_whole_number_set_is_not_a_subset_of_a_bounded_int_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let Some(value) = eval("eval(\"40\")") else { return };
    let age_window = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
    assert!(!(kernel.scalar_subset)(&value.set, &age_window));
}

/// `eval` on anything past a plain int/float literal string
/// declines — general expression evaluation is never modeled.
#[test]
fn test_eval_of_a_non_literal_expression_declines() {
    let Some(value) = eval("eval(\"1 + 1\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

#[test]
fn test_abs_call() {
    let Some(value) = eval("abs(-7)") else { return };
    assert_eq!(value.values, vec![7.0]);
}

/// `max(*[200, 201])` — a starred CALL argument over a known list
/// splices its elements into the positional arguments before
/// dispatch, the same way a starred list-display element does.
#[test]
fn test_starred_call_argument_splices_a_known_list() {
    let Some(value) = eval("max(*[200, 201])") else { return };
    assert_eq!(value.values, vec![201.0]);
}

/// A starred call argument over an UNBOUND name (no proven element
/// count) declines the whole call rather than guess how many
/// positional slots it fills.
#[test]
fn test_starred_call_argument_unknown_iterable_declines() {
    let Some(value) = eval("max(*values)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// round(40.5) == 40 — round-half-to-even, the AGENT-BRIEF
/// row-inverting fact against a naive round-half-up reading.
#[test]
fn test_round_half_to_even() {
    let Some(value) = eval("round(40.5)") else { return };
    assert_eq!(value.values, vec![40.0]);
}

#[test]
fn test_math_floor_call() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("math.floor(x)").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("x", known_values(vec![7.9], PrimitiveKind::Float, TrustProved));
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.values, vec![7.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_string_upper_method() {
    let Some(value) = eval("\"ab\".upper()") else { return };
    let text: String = value
        .values
        .iter()
        .filter_map(|c| char::from_u32(*c as i64 as u32))
        .collect();
    assert_eq!(text, "AB");
}

#[test]
fn test_string_repetition() {
    let Some(value) = eval("\"ab\" * 3") else { return };
    let text: String = value
        .values
        .iter()
        .filter_map(|c| char::from_u32(*c as i64 as u32))
        .collect();
    assert_eq!(text, "ababab");
}

#[test]
fn test_list_concatenation() {
    let Some(value) = eval("[1, 2] + [3, 4]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items.len(), 4);
}

// --- item 1: same-module def calls ---

/// A bare unbound name naming a same-module `def` summarizes through
/// `summaries::call_result`, before the builtin path — `double(3)`
/// answers 6 via the module's own function table, not a builtin.
#[test]
fn test_same_module_function_call() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def double(x):\n    return x + x\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("double(3)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![6.0]);
}

/// A name bound to an opaque LAMBDA value still reaches a
/// same-module `def` of the same name — the gate widening
/// `same_module_def_gate_open` states: a lambda binding carries no
/// scalar/collection value of its own to shadow the def dispatch
/// with.
#[test]
fn test_lambda_bound_name_still_reaches_a_same_module_def_of_the_same_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def double(x):\n    return x + x\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    environment.bind("double", opaque_value("a function value"));
    let parsed = parse_expression("double(3)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![6.0]);
}

/// An ORDINARY value binding (not a lambda) still blocks the
/// same-module-def dispatch — the gate only widens for the opaque
/// lambda shape, matching the def-shadowing-a-builtin test's own
/// "a real bound value wins" posture.
#[test]
fn test_ordinary_bound_value_still_blocks_the_same_module_def_dispatch() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def double(x):\n    return x + x\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    environment.bind("double", known_values(vec![9.0], PrimitiveKind::Integer, TrustProved));
    let parsed = parse_expression("double(3)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Unknown, "a bound Integer is not callable, and shadows the def dispatch");
}

/// A module-level `def` named `len` shadows the builtin `len` —
/// dispatch checks `environment.functions()` before the builtin path.
#[test]
fn test_same_module_def_shadows_a_builtin_of_the_same_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def len(x):\n    return 999\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("len([1, 2, 3])").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    // the shadowing def always answers 999, never the real length 3
    assert_eq!(value.values, vec![999.0]);
}

// --- generators: a same-module generator def's call ---

/// `over_ages()` where `over_ages`'s body is straight-line
/// `yield`s — the CALL answers the ordered List of yields, tagged
/// `source == "generator"`, never routing through
/// `summaries::call_result` (which has no `yield` row).
#[test]
fn test_generator_def_call_answers_the_ordered_yield_list() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "def over_ages():\n",
        "    yield 200\n",
        "    yield 40\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("over_ages()").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.source.as_str(), "generator");
    assert_eq!(
        value.items,
        vec![
            known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

/// `is_generator_def` routing test — a-statements.py's own `stream()`
/// shape: a generator whose body is a single `for` loop with the
/// yield ONE LEVEL inside it (`for value in (10, 20, 30): yield
/// value`), no top-level `yield` statement at all. Before this
/// wave's recursion fix, `is_generator_def` never saw the nested
/// yield and the call would have fallen through to the ORDINARY
/// `summaries::call_result` path instead (which has no `yield` row
/// and would decline outright the same way). This test proves the
/// call now reaches the GENERATOR dispatch: `instances::
/// generator_yields` does not yet read a `Stmt::For` body (that
/// extension is a separate owner's work, tracked in this file's own
/// report), so the call still answers `unknown()` today — but it
/// answers `unknown()` via `generator_yields`'s own honest decline,
/// not via `summaries::call_result`'s. Once `generator_yields` gains
/// the `Stmt::For` reading, this same call site starts answering the
/// ordered yield list with no further change here.
#[test]
fn test_loop_bodied_generator_is_recognized_as_generator_shaped() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "def stream():\n",
        "    for value in (10, 20, 30):\n",
        "        yield value\n",
    ))
    .expect("test module parses")
    .into_syntax();
    assert!(
        is_generator_def(
            module
                .body
                .first()
                .expect("one top-level statement")
                .as_function_def_stmt()
                .expect("is a def")
        ),
        "a yield one level inside a for-loop body is generator-shaped"
    );
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("stream()").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    // generator_yields reads the single-for-loop yield shape, so
    // the call answers the ordered yields as the generator's own
    // list-shaped value
    assert_eq!(value.kind, Kind::List);
    let elements: Vec<f64> = value.items.iter().map(|item| item.values[0]).collect();
    assert_eq!(elements, vec![10.0, 20.0, 30.0]);
}

/// `next(over_ages())` — the first yielded value, per `next_call`'s
/// own generator row.
#[test]
fn test_next_of_a_generator_call_answers_the_first_yield() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "def over_ages():\n",
        "    yield 200\n",
        "    yield 40\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("next(over_ages())").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![200.0]);
}

// --- item 2: construction is a value, not a statement-level fire ---

/// A same-module class construction call evaluates to its instance
/// value; any fire the construction would raise is discarded here
/// (statement-level fires are check.rs's own business).
#[test]
fn test_same_module_construction_is_a_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "class Person:\n",
        "    def __init__(self, age):\n",
        "        self.age = age\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let classes = std::sync::Arc::new(crate::instances::class_table(
        &module, &aliases, &imports, &kernel,
    ));
    let mut environment = empty_environment();
    environment.set_classes(classes);
    let parsed = parse_expression("Person(40)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(
        crate::instances::field_read(&value, "age"),
        Some(known_values(vec![40.0], PrimitiveKind::Integer, TrustProved))
    );
}

/// b-body-expressions.py's own `binary_chained_builder_call` shape:
/// TWO same-module defs each declare their own `class Builder`, with
/// DIFFERENT `size` method bodies. `environment.classes()` is set to
/// the COLLAPSED table `check.rs::local_class_table`'s own
/// first-scanned-wins merge would build for the enclosing body
/// (`make_ok_builder`'s Builder, the one whose `size` returns
/// `"ab"`) — the exact stale, shared guess a chained call must NOT
/// trust. `make_over_builder().type("x").size(1)` still answers
/// `"too-long-str"`, `make_over_builder`'s OWN `size`, proving
/// `receiver_def_local_classes` traces the chain back to the right
/// def instead of reading the collapsed table.
#[test]
fn test_chained_call_on_a_same_named_sibling_local_class_reads_its_own_def() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "def make_ok_builder():\n",
        "    class Builder:\n",
        "        def type(self, _t):\n",
        "            return self\n",
        "        def size(self, _n):\n",
        "            return \"ab\"\n",
        "    return Builder()\n",
        "def make_over_builder():\n",
        "    class Builder:\n",
        "        def type(self, _t):\n",
        "            return self\n",
        "        def size(self, _n):\n",
        "            return \"too-long-str\"\n",
        "    return Builder()\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let ruff_python_ast::Stmt::FunctionDef(make_ok_builder) = &module.body[0] else {
        panic!("module's first statement is def make_ok_builder")
    };
    // the STALE, collapsed table: only `make_ok_builder`'s own
    // `Builder` (whose `size` answers "ab") — the first-scanned-wins
    // shape `local_class_table`'s recursive merge would leave behind
    // for a body enclosing both nested defs.
    let stale_classes = std::sync::Arc::new(crate::instances::class_table(
        &ruff_python_ast::ModModule {
            node_index: ruff_python_ast::AtomicNodeIndex::NONE,
            range: TextRange::default(),
            body: make_ok_builder
                .body
                .iter()
                .filter(|stmt| matches!(stmt, ruff_python_ast::Stmt::ClassDef(_)))
                .cloned()
                .collect(),
        },
        &aliases,
        &imports,
        &kernel,
    ));
    let mut environment = empty_environment();
    environment.set_functions(table);
    environment.set_classes(stale_classes);
    let parsed = parse_expression("make_over_builder().type(\"x\").size(1)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, string_models::string_literal_value("too-long-str").values);
}

// --- method dispatch (value side) ---

/// `person.next_year(40)`-shaped positional call — resolves through
/// `method_def_of`/`method_call_result`, answering the RESULT value.
#[test]
fn test_method_call_positional_answers_the_result_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = environment_with_person_classes(&kernel);
    let constructed = parse_expression("Person(40)").expect("test source must parse");
    let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
    environment.bind("person", instance);
    let call = parse_expression("person.next_year(1)").expect("test source must parse");
    let value = evaluate_expression(&call.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![41.0]);
}

/// A method call with a KEYWORD argument maps to position the same
/// way a plain `def` call does.
#[test]
fn test_method_call_keyword_argument_maps_to_position() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = environment_with_person_classes(&kernel);
    let constructed = parse_expression("Person(40)").expect("test source must parse");
    let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
    environment.bind("person", instance);
    let call = parse_expression("person.next_year(bump=2)").expect("test source must parse");
    let value = evaluate_expression(&call.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![42.0]);
}

/// A method call's own receiver MUTATION is not threaded back into
/// the environment from a nested expression read — only the result
/// is answered here (the mutation half is check.rs's statement-sink
/// business).
#[test]
fn test_method_call_does_not_thread_the_mutated_receiver_back() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "class Counter:\n",
        "    def __init__(self):\n",
        "        self.count = 0\n",
        "    def bump(self):\n",
        "        self.count = self.count + 1\n",
        "        return self.count\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let classes =
        std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, &kernel));
    let mut environment = empty_environment();
    environment.set_classes(classes);
    let constructed = parse_expression("Counter()").expect("test source must parse");
    let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
    environment.bind("counter", instance);
    let call = parse_expression("counter.bump()").expect("test source must parse");
    let value = evaluate_expression(&call.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![1.0], "the call answers the result value");
    // the environment's own `counter` binding is UNCHANGED — a
    // nested expression read never writes the mutated instance back
    let still_bound = environment.read("counter").expect("counter remains bound");
    assert_eq!(
        crate::instances::field_read(still_bound, "count"),
        Some(known_values(vec![0.0], PrimitiveKind::Integer, TrustProved))
    );
}

// --- item 7: await ---

#[test]
fn test_await_evaluates_the_inner_expression() {
    let Some(value) = eval("await x") else { return };
    // `x` is unbound in the empty test environment, so the await of
    // it is unknown — this pins that await passes THROUGH to the
    // inner expression's own value rather than always answering
    // unknown regardless of the inner expression
    assert_eq!(value.kind, Kind::Unknown);
}

#[test]
fn test_await_of_a_known_value_passes_it_through() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("await x").expect("test source must parse");
    let mut environment = empty_environment();
    environment.bind("x", known_values(vec![7.0], PrimitiveKind::Integer, TrustProved));
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![7.0]);
}

/// `await asyncio.gather(a, b)` answers the aggregate List of the
/// already-evaluated argument values, in call order.
#[test]
fn test_asyncio_gather_awaited_answers_the_aggregate_list() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("await asyncio.gather(1, 2)").expect("test source must parse");
    let environment = empty_environment();
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

// --- d-module-surface.py: importlib.import_module ---

/// `importlib.import_module("d_helper")` — d-module-surface.py's own
/// `dynamic_import` row: this domain has no module-object Kind, so
/// the answer is the opaque "a module object" sort.
#[test]
fn test_importlib_import_module_answers_opaque() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("import importlib\n").expect("test module parses").into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("importlib.import_module(\"d_helper\")").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert!(value.kind_word.is_some(), "importlib.import_module(...) must answer opaque, not unknown: {value:?}");
}

// --- e-class-and-function.py: generator METHODS via next()/anext() ---

/// e-class-and-function.py's own `generator_method`: `next(GenAges()
/// .ages())` where `ages(self)` is a generator METHOD, not a bare
/// def — the method-call dispatch now routes a generator-shaped
/// method through `instances::generator_yields` (with `self`
/// prepended to the positional arguments) instead of declining
/// through `method_call_result`'s own no-`yield`-row interpreter.
#[test]
fn test_generator_method_call_answers_the_first_yielded_value_via_next() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "class GenAges:\n",
        "    def ages(self):\n",
        "        yield 40\n",
        "        yield 41\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let empty_aliases = std::collections::HashMap::new();
    let empty_imports = crate::surface::surface_imports(&ruff_python_ast::ModModule {
        node_index: ruff_python_ast::AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: Vec::new().into(),
    });
    let classes = crate::instances::class_table(&module, &empty_aliases, &empty_imports, &kernel);
    let mut environment = empty_environment();
    environment.set_classes(std::sync::Arc::new(classes));
    let parsed = parse_expression("next(GenAges().ages())").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![40.0], "the generator method's first yield must read through next(): {value:?}");
}

/// `anext` dispatches identically to `next` once `await` transparently
/// unwraps — e-class-and-function.py's own `async_generator_first_
/// value`/`generator_first_value` pair.
#[test]
fn test_anext_of_a_generator_call_answers_the_first_yielded_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "async def async_yield_ages():\n",
        "    yield 40\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("await anext(async_yield_ages())").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![40.0]);
}

// --- e-class-and-function.py: keyword-only and **kwargs calls ---

/// e-class-and-function.py's own `keyword_only_call`: a keyword-only
/// parameter the CALLER covers by keyword (`only_keyword(age=200)`)
/// now interprets the body's own exact value, rather than declining
/// outright.
#[test]
fn test_keyword_only_call_binds_and_interprets() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def only_keyword(*, age):\n    return age\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("only_keyword(age=200)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![200.0]);
}

/// e-class-and-function.py's own `kwargs_parameter`: `**fields`
/// collects every keyword the call site passes into a dict, and the
/// body's own `fields["age"]` reads it back exactly.
#[test]
fn test_kwargs_call_collects_keywords_into_a_dict() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def gather_kwargs(**fields):\n    return fields[\"age\"]\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("gather_kwargs(age=200)").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![200.0]);
}
