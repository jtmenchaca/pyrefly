use super::*;

/// `and`/`or` return an OPERAND, not a coerced bool — `0 and 5`
/// answers `0` (the falsy left operand), `0 or 5` answers `5` (the
/// first truthy operand reached).
#[test]
fn test_and_or_return_operands() {
    let Some(and_result) = eval("0 and 5") else { return };
    assert_eq!(and_result.values, vec![0.0]);
    assert_eq!(and_result.kind_tag, Some(PrimitiveKind::Integer));

    let Some(or_result) = eval("0 or 5") else { return };
    assert_eq!(or_result.values, vec![5.0]);
}

#[test]
fn test_not_and_invert() {
    let Some(not_result) = eval("not 0") else { return };
    assert_eq!(not_result.values, vec![1.0]);
    assert_eq!(not_result.kind_tag, Some(PrimitiveKind::Boolean));

    // ~5 == -(5+1) == -6
    let Some(invert_result) = eval("~5") else { return };
    assert_eq!(invert_result.values, vec![-6.0]);
    assert_eq!(invert_result.kind_tag, Some(PrimitiveKind::Integer));
}

/// A ternary whose test is not decidable joins both arms' values —
/// the loosest sound answer once neither arm can be ruled out.
#[test]
fn test_ternary_both_arms_join() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("1 if flag else 2").expect("test source must parse");
    let expression = parsed.into_expr();
    let mut environment = empty_environment();
    environment.bind("flag", unknown());
    let value = evaluate_expression(&expression, &environment, &kernel);
    // an Integer 1 joined with an Integer 2 is not exactly-known —
    // the join is not equal to either arm alone
    assert_ne!(value, known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
    assert_ne!(value, known_values(vec![2.0], PrimitiveKind::Integer, TrustProved));
}

#[test]
fn test_ternary_decided_test_answers_one_arm() {
    let Some(value) = eval("1 if True else 2") else { return };
    assert_eq!(value.values, vec![1.0]);
}
