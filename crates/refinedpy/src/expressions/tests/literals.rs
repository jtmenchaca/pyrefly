use super::*;

#[test]
fn test_int_literal() {
    let Some(value) = eval("7") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![7.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_float_literal() {
    let Some(value) = eval("3.5") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![3.5]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn test_negative_int_literal() {
    let Some(value) = eval("-7") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![-7.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_name_bound() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("x").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("x", known_values(vec![42.0], PrimitiveKind::Integer, TrustProved));
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.values, vec![42.0]);
}

/// A name bound to an Integer-sorted value keeps the Integer tag
/// through `a + 1` — the arithmetic transfer reads the BOUND
/// value's own sort (never re-derives it syntactically from the
/// name), so `both_int` sees Integer op Integer here.
#[test]
fn test_name_bound_int_keeps_integer_sort_through_addition() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("a + 1").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("a", known_values(vec![10.0], PrimitiveKind::Integer, TrustProved));
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.values, vec![11.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_name_unbound() {
    let Some(value) = eval("y") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `s.upper()` where `s` is a bare, unrefined `str` parameter (seeded
/// as the whole-strings ground, `typereading::base_sort_return_
/// refinement`'s own doc — never `Kind::Values`) now answers the
/// SORT-ONLY Σ* claim (`string_models::string_method_sort_only_
/// result`) rather than `unknown()` — `evaluate_attribute_call`'s own
/// fallback past the exact-string block, A3.xfer.case's own row.
#[test]
fn test_upper_over_an_unbounded_str_parameter_answers_the_sort_only_claim() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("s.upper()").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("s", known_set(strings(), None, TrustProved, SetKindTag::None));
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.kind, Kind::Set, "the sort-only fallback answers a Set, never unknown(): {value:?}");
}

/// `s * n` where BOTH `s: str` and `n: int` are bare, unrefined
/// parameters — `sequence_repetition`'s own exact row declines twice
/// over (no exact code points, no single known count), so
/// `string_repetition_sort_only` is what keeps this a real Σ*
/// answer rather than `unknown()` — A3.xfer.repeat's own row.
#[test]
fn test_string_times_unbounded_int_answers_the_sort_only_claim() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("s * n").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("s", known_set(strings(), None, TrustProved, SetKindTag::None));
    environment.bind(
        "n",
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]), None, TrustProved, SetKindTag::None)
        },
    );
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.kind, Kind::Set, "the sort-only fallback answers a Set, never unknown(): {value:?}");
}

/// A Float-sorted (never Integer) count still declines — `str * n`
/// where `n` provably is not an int is CPython's own `TypeError`,
/// not a row `string_repetition_sort_only` may answer a value for.
#[test]
fn test_string_times_a_float_sorted_count_still_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("s * n").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("s", known_set(strings(), None, TrustProved, SetKindTag::None));
    environment.bind("n", known_values(vec![1.5], PrimitiveKind::Float, TrustProved));
    let value = evaluate_expression(&expression, &environment, &kernel);
    assert_eq!(value.kind, Kind::Unknown, "a non-integer count is not this row's shape: {value:?}");
}

#[test]
fn test_add_int() {
    let Some(value) = eval("2 + 3") else { return };
    assert_eq!(value.values, vec![5.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_sub_int() {
    let Some(value) = eval("5 - 8") else { return };
    assert_eq!(value.values, vec![-3.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_mult_int() {
    let Some(value) = eval("4 * 6") else { return };
    assert_eq!(value.values, vec![24.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `/` is ALWAYS true division in Python — the result is Float-sorted
/// even when both operands are int-sorted and the quotient is whole
/// (6 / 3 == 2.0, not the int 2). This is the row the mission's
/// int-sort fire depends on: a Float-tagged `6 / 3` assigned into an
/// int-sorted alias must fire, not silently pass as if it were `int`.
#[test]
fn test_true_division_of_two_ints_is_float_tagged_even_on_a_whole_quotient() {
    let Some(value) = eval("6 / 3") else { return };
    assert_eq!(value.values, vec![2.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn test_true_division_int_gives_float() {
    // 7 / 2 == 3.5 — Python `/` is always true division
    let Some(value) = eval("7 / 2") else { return };
    assert_eq!(value.values, vec![3.5]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn test_floor_division_negative_floors_toward_negative_infinity() {
    // -7 // 2 == -4 (not -3, which truncation toward zero would give)
    let Some(value) = eval("-7 // 2") else { return };
    assert_eq!(value.values, vec![-4.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_mod_sign_follows_divisor_negative_divisor() {
    // -7 % 2 == 1 — sign of the result follows the divisor (2, positive)
    let Some(value) = eval("-7 % 2") else { return };
    assert_eq!(value.values, vec![1.0]);
}

#[test]
fn test_mod_sign_follows_divisor_negative_dividend_side() {
    // 7 % -2 == -1 — sign of the result follows the divisor (-2, negative)
    let Some(value) = eval("7 % -2") else { return };
    assert_eq!(value.values, vec![-1.0]);
}

#[test]
fn test_pow_int_exact() {
    let Some(value) = eval("2 ** 10") else { return };
    assert_eq!(value.values, vec![1024.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `int ** negative int` converts to float per §6.5 / stdtypes note
/// (5) — `10 ** -2 == 0.01`, Float-sorted even though both operands
/// were Integer-sorted.
#[test]
fn test_pow_negative_int_exponent_widens_to_float() {
    let Some(value) = eval("10 ** -2") else { return };
    assert!((value.values[0] - 0.01).abs() < 1e-12);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn test_division_by_zero_declines() {
    let Some(value) = eval("1 / 0") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

#[test]
fn test_boolean_literal_true() {
    let Some(value) = eval("True") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![1.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
}

/// `True + True == 2` — Python's `bool` is an `int` subclass, so
/// arithmetic on booleans reads them as Integer and yields an
/// ordinary Integer-sorted result (AGENT-BRIEF.md).
#[test]
fn test_boolean_arithmetic_yields_integer_sort() {
    let Some(value) = eval("True + True") else { return };
    assert_eq!(value.values, vec![2.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn test_none_literal() {
    let Some(value) = eval("None") else { return };
    assert_eq!(value.kind, Kind::Null);
}

#[test]
fn test_unsupported_construct_is_unknown() {
    // `f` is an unbound name and not a modeled builtin — the call
    // dispatch declines rather than guessing at an unmodeled callee
    let Some(value) = eval("f(1)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `lambda: 40` read as a VALUE answers opaque — "a function value,"
/// never a specific scalar.
#[test]
fn test_lambda_as_a_value_is_opaque() {
    let Some(value) = eval("lambda: 40") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("a function value"));
}

/// `register_retained_callables` scanning a bare `lambda: 40`
/// (the shape `summaries::interpret_body`'s `Stmt::Return` arm
/// hands it) makes a LATER read of that SAME `Expr::Lambda` node
/// answer a retained-callable value rather than the plain opaque
/// one — and calling that value through `evaluate_call`'s
/// retained-callable arm interprets the lambda's own body,
/// answering its exact return value.
#[test]
fn test_retained_lambda_call_answers_its_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let lambda_expr = parse_expression("lambda: 40").expect("test source must parse").into_expr();
    let mut environment = empty_environment();
    register_retained_callables(&lambda_expr, &mut environment);
    let retained = evaluate_expression(&lambda_expr, &environment, &kernel);
    assert_eq!(retained.kind, Kind::Object);
    assert_eq!(retained.kind_word, Some("a function value"));
    assert!(!retained.source.is_empty(), "a registered lambda's source carries its table key");

    let call_expr = parse_expression("f()").expect("test source must parse").into_expr();
    let Expr::Call(call) = call_expr else { panic!("expected a call expression") };
    environment.bind("f", retained);
    let result = evaluate_call(&call, &environment, &kernel);
    assert_eq!(result.values, vec![40.0]);
}

/// A bare reference to a SAME-MODULE `def` — `f = identity` — reads
/// as `env::same_module_def_alias_value`, never bare `unknown()`:
/// `identity` is never separately bound in `environment.bindings`
/// (only indexed in `environment.functions()`), so without the new
/// `Expr::Name` dispatch arm this would fall to the catch-all.
#[test]
fn test_a_bare_reference_to_a_same_module_def_reads_as_an_alias_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def identity(x):\n    return x\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let name_expr = parse_expression("identity").expect("test source must parse").into_expr();
    let value = evaluate_expression(&name_expr, &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("a function value"));
    assert_eq!(env::same_module_def_alias_name(&value), Some("identity"));
}

/// Calling through a same-module-def alias value (`f = identity;
/// f(x)`) reaches `identity`'s own body via `evaluate_call`'s new
/// alias-call arm — the same interpretation a direct `identity(x)`
/// call would answer, not a bare `unknown()`.
#[test]
fn test_calling_through_a_same_module_def_alias_answers_the_defs_own_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def identity(x):\n    return x\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let name_expr = parse_expression("identity").expect("test source must parse").into_expr();
    let aliased = evaluate_expression(&name_expr, &environment, &kernel);
    environment.bind("f", aliased);
    let call_expr = parse_expression("f(40)").expect("test source must parse").into_expr();
    let Expr::Call(call) = call_expr else { panic!("expected a call expression") };
    let result = evaluate_call(&call, &environment, &kernel);
    assert_eq!(result.values, vec![40.0], "f(40) through the alias must answer identity(40)'s own body");
}

/// A retained lambda that reads a FREE variable
/// (`e-class-and-function.py`'s own `make_adder` shape: `lambda
/// age: age + step` closes over `step`) carries that free name's
/// value in its own closure snapshot, taken at the moment
/// `register_retained_callables` runs — a later call answers using
/// THAT snapshot, not whatever the call site happens to bind the
/// free name to.
#[test]
fn test_retained_lambda_closure_reads_a_free_name_at_creation() {
    let Some(kernel) = loaded_kernel() else { return };
    let lambda_expr = parse_expression("lambda age: age + step").expect("test source must parse").into_expr();
    let mut environment = empty_environment();
    environment.bind("step", known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
    register_retained_callables(&lambda_expr, &mut environment);
    let retained = evaluate_expression(&lambda_expr, &environment, &kernel);

    // rebinding `step` AFTER registration must not affect the
    // already-taken closure snapshot — Python's own closure rule
    // pins the binding to the DEFINING scope, not the call site.
    environment.bind("step", known_values(vec![999.0], PrimitiveKind::Integer, TrustProved));
    environment.bind("f", retained);
    let call_expr = parse_expression("f(40)").expect("test source must parse").into_expr();
    let Expr::Call(call) = call_expr else { panic!("expected a call expression") };
    let result = evaluate_call(&call, &environment, &kernel);
    assert_eq!(result.values, vec![41.0], "must use step=1 from the closure, not step=999 from the call site");
}

/// Two creations of the textually SAME lambda (two calls to a
/// function returning `lambda x: x + step`, each closing over a
/// different `step`) never conflate: each registration mints its
/// own key, so the second's closure never overwrites the first's
/// still-live retained value (`conflation_probe.py`'s own row,
/// reproduced directly against `register_retained_callables`).
#[test]
fn test_two_creations_of_the_same_lambda_text_keep_separate_closures() {
    let Some(kernel) = loaded_kernel() else { return };
    let lambda_expr = parse_expression("lambda x: x + step").expect("test source must parse").into_expr();
    let mut environment = empty_environment();

    environment.bind("step", known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
    register_retained_callables(&lambda_expr, &mut environment);
    let first = evaluate_expression(&lambda_expr, &environment, &kernel);

    environment.bind("step", known_values(vec![100.0], PrimitiveKind::Integer, TrustProved));
    register_retained_callables(&lambda_expr, &mut environment);
    let second = evaluate_expression(&lambda_expr, &environment, &kernel);

    environment.bind("first", first);
    environment.bind("second", second);
    let call_first = parse_expression("first(40)").expect("test source must parse").into_expr();
    let Expr::Call(call_first) = call_first else { panic!("expected a call expression") };
    let call_second = parse_expression("second(40)").expect("test source must parse").into_expr();
    let Expr::Call(call_second) = call_second else { panic!("expected a call expression") };
    assert_eq!(evaluate_call(&call_first, &environment, &kernel).values, vec![41.0]);
    assert_eq!(evaluate_call(&call_second, &environment, &kernel).values, vec![140.0]);
}

// --- item 4: __name__ ---

#[test]
fn test_dunder_name_is_a_sort_only_string() {
    let Some(value) = eval("__name__") else { return };
    assert_eq!(value.kind, Kind::Set);
}

#[test]
fn test_dunder_name_shadowed_by_a_local_binding_reads_the_binding() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("__name__").expect("test source must parse");
    let mut environment = empty_environment();
    environment.bind("__name__", known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![1.0]);
}

// --- item 5: bytes literal ---

#[test]
fn test_bytes_literal() {
    let Some(value) = eval("b\"ab\"") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![97.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![98.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

#[test]
fn test_bytes_index_reads_an_int() {
    // b"ab"[0] is the int 97 — AGENT-BRIEF.md's own pinned fact
    let Some(value) = eval("b\"ab\"[0]") else { return };
    assert_eq!(value.values, vec![97.0]);
}
