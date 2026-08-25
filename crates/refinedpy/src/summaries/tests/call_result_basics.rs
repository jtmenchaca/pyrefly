use super::*;

/// A10.xfer.recursion.py's own `fact`: a self-recursive `def` with a
/// decreasing measure (`n - 1`, strictly below `n`, bounded below by
/// the `n <= 1` base case), called with a WINDOW argument `n ∈ [0, 5]`
/// rather than a single value. Pins `enumerated_recursive_call`: the
/// symbolic path alone (the ordinary depth-capped interpreter, `n <=
/// 1` never deciding) cannot resolve this at all, so the call-site
/// enumeration must run once per admitted `n` (0 through 5) and join —
/// factorial of `[0, 5]` is `{1, 1, 2, 6, 24, 120}`, DEDUPLICATED by
/// `join_known`'s own Integer-tagged `Kind::Values` merge (`fact(0)`
/// and `fact(1)` both answer the exact value 1) to `{1, 2, 6, 24,
/// 120}`.
#[test]
fn a_self_recursive_call_over_a_bounded_window_unrolls_by_enumeration() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parse_module(concat!(
        "def fact(n: int) -> int:\n",
        "    if n <= 1:\n",
        "        return 1\n",
        "    return n * fact(n - 1)\n",
    ))
    .expect("fixture source parses")
    .into_syntax();
    let table = Arc::new(crate::function_table::function_table(&module));
    let fact = table.def("fact").expect("fact is a top-level def").clone();
    let window = known_integer_window(0.0, 5.0);
    let result = call_result(&fact, &[window], Some(&table), &kernel, 0)
        .expect("a self-recursive call over a small bounded window must enumerate, never decline");
    let mut values = result.values.clone();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(
        result.kind,
        Kind::Values,
        "factorial of [0, 5] is the small exact, deduplicated set {{1, 2, 6, 24, 120}}: {result:?}"
    );
    assert_eq!(values, vec![1.0, 2.0, 6.0, 24.0, 120.0]);
}

#[test]
fn straight_line_body_answers_the_returned_expression() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def double(x):\n    return x + x\n");
    let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0).expect("straight-line body answers");
    assert_eq!(result.values, vec![6.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// A nested `def` returned out of its own enclosing function
/// (e-class-and-function.py's `make_counter`, r-ast-census.py's
/// `with_paramspec_presence`): `interpret_body`'s `Stmt::FunctionDef`
/// arm retains the def's own body and binds its name to a
/// retained-callable value, which `return inner` then answers as an
/// ordinary `Expr::Name` read — no special-casing needed there.
#[test]
fn a_nested_def_returned_out_of_its_enclosing_function_is_retained() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def make_adder(step):\n    def inner(x):\n        return x + step\n    return inner\n");
    let result = call_result(&def, &[known_int(1.0)], None, &kernel, 0)
        .expect("a body ending in a bare-name return of its own nested def answers");
    assert_eq!(result.kind, Kind::Object);
    assert_eq!(result.kind_word, Some("a function value"));
    assert!(
        crate::env::retained_callable_key(&result).is_some(),
        "a retained callable's source must parse as its table key: {result:?}"
    );
    // the retained body was recorded against `call_result`'s own
    // (disposable) interpretation environment — `call_result` itself
    // exposes no handle to it, so this test only pins that the VALUE
    // carries a real key; `expressions.rs`'s own retained-callable
    // tests pin the full call-and-answer round trip through
    // `evaluate_call`.
}

/// `A10.xfer.closure.py`'s own `counter`/`closure_inside` shape: a
/// factory `def` binds a LOCAL (`n = 0`), returns a NESTED `def`
/// declared `nonlocal n; n += 1; return n`, and a SEPARATE top-level
/// `def` calls the factory then calls the returned closure. Pins
/// `locally_bound_names`' own `nonlocal`-name exclusion (folded from
/// `call_effects`'s own local copy into the shared free-name
/// question `free_variable_snapshot`/`needs_enclosing_scope` both
/// read) together with `collect_write_target_base_name`'s new bare-
/// Name-target read: without either half, `n += 1` reads `n` as an
/// ordinary local, finds nothing seeded, and the whole call declines
/// — the gap this fix closes. With both, the closure's snapshot
/// carries `n: {0}`, the aug-assign folds it to `{1}`, and the outer
/// call's return value is the exact set `{1}`, never a decline.
#[test]
fn a_retained_closures_nonlocal_aug_assign_reads_its_own_captured_snapshot() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parse_module(concat!(
        "def counter():\n",
        "    n = 0\n",
        "    def next_value():\n",
        "        nonlocal n\n",
        "        n += 1\n",
        "        return n\n",
        "    return next_value\n",
        "\n",
        "def closure_inside():\n",
        "    next_value = counter()\n",
        "    return next_value()\n",
    ))
    .expect("fixture source parses")
    .into_syntax();
    let table = Arc::new(crate::function_table::function_table(&module));
    let closure_inside = table.def("closure_inside").expect("closure_inside is a top-level def").clone();
    let result = call_result(&closure_inside, &[], Some(&table), &kernel, 0)
        .expect("the closure's nonlocal aug-assign must read its own captured n, never decline");
    assert_eq!(result.kind, Kind::Values, "the first call of a fresh counter answers the exact value {{1}}: {result:?}");
    assert_eq!(result.values, vec![1.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn a_trailing_default_parameter_is_evaluated_when_no_argument_covers_it() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def add(x, y=10):\n    return x + y\n");
    let result = call_result(&def, &[known_int(5.0)], None, &kernel, 0).expect("default parameter fills in");
    assert_eq!(result.values, vec![15.0]);
}

/// e-class-and-function.py's own `grow_into_bucket`: a default
/// parameter's value (read from `enclosing`, since the default
/// expression names a module-level list) is MUTATED inside the body
/// (`bucket.append(age)`) before a later statement reads it back
/// (`return bucket[0]`). Before `write_mutating_call_expr` existed,
/// the append call was evaluated and discarded, leaving `bucket`
/// bound to its stale pre-append value — the read then saw an empty
/// list and declined. `arguments` is empty here (`bucket` fills from
/// its own default), so this pins the mutation-carries-forward
/// behavior in isolation from the enclosing-environment default read
/// (that seam already has its own test above).
#[test]
fn a_mutating_call_on_a_parameter_carries_its_write_into_a_later_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(
        "def grow_into_bucket(age, bucket=[40]):\n    bucket.append(age)\n    return bucket[0]\n",
    );
    let result = call_result(&def, &[known_int(41.0)], None, &kernel, 0)
        .expect("the append must carry forward so bucket[0] still reads the first element, 40");
    assert_eq!(result, known_int(40.0));
}

#[test]
fn an_if_else_where_both_arms_return_known_values_joins_both_possibilities() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(
        "def pick(flag):\n    if flag:\n        return 3\n    else:\n        return 5\n",
    );
    let result =
        call_result(&def, &[unknown()], None, &kernel, 0).expect("both known-value arms join to an answer");
    // an undecidable flag interprets both arms; the join of 3 and 5
    // under one Integer tag is the two-value carrier
    // join_known's own test (test_join_known_like_sort_keeps_the_tag_mixed_sort_loses_it)
    // pins for two same-sort Values joins
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    let mut values = result.values.clone();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![3.0, 5.0]);
}

#[test]
fn a_body_that_falls_off_the_end_contributes_null_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def maybe_none(flag):\n    if flag:\n        return 3\n    x = 1\n");
    let result = call_result(&def, &[known_int(1.0)], None, &kernel, 0)
        .expect("a known-true flag still interprets the fall-through arm's shape honestly");
    // flag is KNOWN true here, so only the `return 3` arm runs and the
    // fall-through never contributes — this pins the definite-branch
    // path specifically; the undecidable-flag fall-through case is
    // covered by the next test
    assert_eq!(result.values, vec![3.0]);
}

#[test]
fn an_undecidable_flag_whose_false_arm_falls_off_the_end_joins_in_null() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def maybe_none(flag):\n    if flag:\n        return 3\n    x = 1\n");
    let result = call_result(&def, &[unknown()], None, &kernel, 0)
        .expect("an undecidable flag interprets both the return arm and the fall-through");
    // the true arm returns 3; the false arm falls off the end,
    // contributing null_value() — the join of an Integer with Null
    // is neither a bare Integer (Kind::Values) nor a bare Null
    assert_ne!(result.kind, Kind::Unknown);
    assert_ne!(result.kind, Kind::Values);
    assert_ne!(result.kind, Kind::Null);
}

#[test]
fn a_body_with_a_while_loop_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def counted(n):\n    while n > 0:\n        n -= 1\n    return n\n");
    assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
}

#[test]
fn the_depth_cap_declines_before_interpreting_the_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def double(x):\n    return x + x\n");
    assert!(call_result(&def, &[known_int(3.0)], None, &kernel, CALL_DEPTH_CAP).is_none());
}

#[test]
fn a_return_with_an_unknown_value_declines_the_whole_call() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def opaque(x):\n    return f(x)\n");
    assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
}

#[test]
fn too_many_arguments_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def one_arg(x):\n    return x\n");
    assert!(call_result(&def, &[known_int(1.0), known_int(2.0)], None, &kernel, 0).is_none());
}

/// `*args` genuinely interprets — bound to the caller's own trailing
/// arguments as a known tuple (`bind_parameters`'s own vararg row) —
/// rather than declining outright. This body never reads `args` at
/// all, so the call answers the literal `1` regardless of what
/// arguments the caller passed.
#[test]
fn varargs_with_no_argument_reads_interprets_the_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def variadic(*args):\n    return 1\n");
    let result = call_result(&def, &[], None, &kernel, 0).expect("a *args parameter is no longer a decline");
    assert_eq!(result.values, vec![1.0]);
}

/// e-class-and-function.py's own `first_age` shape: `*ages: int`
/// bound to the caller's own trailing arguments as a tuple, then
/// `ages[0]` reads the first one through the ordinary subscript path
/// — the regression this pins: `first_age(40, 41)` (an IN-SET call
/// under `Age`) answers the exact value 40, never a coarse fallback
/// set the containment law would wrongly fire against a narrow sink.
#[test]
fn varargs_binds_a_known_tuple_of_the_trailing_arguments() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def first_age(*ages):\n    return ages[0]\n");
    let result = call_result(&def, &[known_int(40.0), known_int(41.0)], None, &kernel, 0)
        .expect("*ages binds to the known (40, 41) tuple, and ages[0] reads through it");
    assert_eq!(result, known_int(40.0));
}

/// q-decline-names.py's own `sum_rest` shape: `*rest: int` binds to a
/// known tuple (`bind_parameters`'s own vararg row), and a `for`
/// loop over that SAME name now interprets instead of declining the
/// whole call — `Stmt::For`'s own arm in `interpret_body`.
#[test]
fn call_result_sums_a_for_loop_over_the_vararg_tuple() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(
        "def sum_rest(first, *rest):\n    total = first\n    for value in rest:\n        total = total + value\n    return total\n",
    );
    let result = call_result(&def, &[known_int(40.0), known_int(0.0)], None, &kernel, 0)
        .expect("a for loop over the known vararg tuple must interpret");
    assert_eq!(result, known_int(40.0));
}

/// The same shape with more than one rest element, pinning the
/// left-to-right accumulation order (`bind_parameters`'s own tuple
/// order, `tuple_literal_value` producing `Kind::List` in source
/// argument order).
#[test]
fn call_result_for_loop_accumulates_every_vararg_element_in_order() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(
        "def sum_rest(first, *rest):\n    total = first\n    for value in rest:\n        total = total + value\n    return total\n",
    );
    let result = call_result(&def, &[known_int(1.0), known_int(2.0), known_int(3.0)], None, &kernel, 0)
        .expect("every vararg element must accumulate");
    assert_eq!(result, known_int(6.0));
}

/// A `for` loop over a receiver that is not a known `Kind::List` (a
/// bare, unmodeled parameter here) still declines the whole call —
/// the new `Stmt::For` arm never guesses at an unread iterable.
#[test]
fn call_result_for_loop_over_a_non_list_receiver_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def sum_values(values):\n    total = 0\n    for value in values:\n        total = total + value\n    return total\n");
    assert!(call_result(&def, &[unknown()], None, &kernel, 0).is_none());
}

/// A `return` inside a `for` body ends the loop immediately —
/// CPython's own semantics — so a LATER element never runs; this
/// pins that the loop stops at the first iteration's own return
/// rather than continuing to accumulate past it.
#[test]
fn call_result_for_loop_return_ends_the_loop_on_the_first_element() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def first_rest(first, *rest):\n    for value in rest:\n        return value\n    return first\n");
    let result = call_result(&def, &[known_int(1.0), known_int(2.0), known_int(3.0)], None, &kernel, 0)
        .expect("a return inside the for body must decide the call");
    assert_eq!(result, known_int(2.0), "the loop returns on its first element, 2, never reaching 3");
}

/// A def with both a plain parameter and a `*args` tail: the plain
/// parameter takes the first argument, `*args` collects the rest.
#[test]
fn varargs_after_a_plain_parameter_collects_only_the_remaining_arguments() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def first_and_rest(first, *rest):\n    return rest[0]\n");
    let result = call_result(&def, &[known_int(1.0), known_int(2.0), known_int(3.0)], None, &kernel, 0)
        .expect("rest binds to the known (2, 3) tuple");
    assert_eq!(result, known_int(2.0));
}
