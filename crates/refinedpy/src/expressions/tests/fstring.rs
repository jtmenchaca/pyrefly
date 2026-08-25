use super::*;

#[test]
fn test_fstring_composition_int_and_str() {
    let Some(value) = eval("f\"n={1 + 1} s={'ab'}\"") else { return };
    let text: String = value
        .values
        .iter()
        .filter_map(|c| char::from_u32(*c as i64 as u32))
        .collect();
    assert_eq!(text, "n=2 s=ab");
}

// --- f-string float spelling ---

#[test]
fn test_fstring_float_spelling_keeps_the_decimal_point() {
    let Some(value) = eval("f\"{30.0}\"") else { return };
    let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
    assert_eq!(text, "30.0");
}

#[test]
fn test_fstring_float_spelling_non_whole_value() {
    let Some(value) = eval("f\"{3.5}\"") else { return };
    let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
    assert_eq!(text, "3.5");
}

// --- f-string composition over a known SET interpolation (item 2) ---

/// `f"n={counted(n)}"` where `counted`'s body is a `while` loop (a
/// genuine `interpret_body` decline, unlike an ellipsis-only stub —
/// see this unit's own report on why `a-statements.py`'s literal
/// `unread_number() -> int: ...` does NOT reach this fallback: an
/// ellipsis body falls through to `Kind::Null`, never a decline).
/// The declined call's `-> int` annotation answers the whole-number
/// set (`summaries::return_sort_fallback`, item 1), so the f-string
/// steps down to the PATTERN tier instead of `unknown()` — a known
/// `Kind::Set`, never `Kind::Unknown`.
#[test]
fn test_fstring_with_a_sort_only_set_interpolation_composes_a_pattern() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("def counted(n) -> int:\n    while n > 0:\n        n -= 1\n    return n\n")
        .expect("test module parses")
        .into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("f\"n={counted(3)}\"").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    // the pattern tier answers a SET (the concatenation of the "n="
    // tuple with the interpolation's spellings), never unknown().
    // Whether that pattern is contained in a bounded length window
    // is the kernel's containment question — its subset decider
    // REFUSES this concatenation-vs-window shape today (assignability
    // catches the refusal and answers Undetermined), so no raw
    // subset ask is made here; the composition itself is the claim
    // this test pins.
    assert_eq!(value.kind, Kind::Set);
    assert!(!value.set.forms.is_empty());
}

/// A plain literal-only f-string with no interpolation at all still
/// composes the exact string it always did — the pattern tier is
/// never reached when there is nothing to interpolate.
#[test]
fn test_fstring_plain_literal_still_answers_exact() {
    let Some(value) = eval("f\"hello\"") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
}

// --- zero_padded_decimal_spelling / zero_padded_decimal_width ---

/// `year: [1970, 9999]` formatted `:04d` — every member already
/// spells exactly 4 decimal digits, so the zero-fill is a no-op and
/// the exact digit-window `Repeat(digits, 4, 4)` is sound.
#[test]
fn test_zero_padded_decimal_spelling_exact_when_padding_is_a_no_op() {
    let year = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![at_least(1970.0), refined_sets::refinement_forms::at_most(9999.0)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let Some(kernel) = loaded_kernel() else { return };
    let source = "f\"{year:04d}\"";
    let parsed = parse_expression(source).expect("test source must parse");
    let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
    let mut environment = empty_environment();
    environment.bind("year", year);
    let result = evaluate_fstring(&fstring, &environment, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert!(
        assignability::states_sequence(&result.set),
        "a zero-padded bounded-range interpolation must answer a sequence-shaped set: {:?}",
        result.set
    );
}

/// A range that WOULD need real padding for some members but not
/// others (`8..12` against `02d`: "08".."12") declines rather than
/// approximate — `decimal_digit_count(8) == 1` while
/// `decimal_digit_count(12) == 2`, so the two ends disagree with the
/// stated width and the whole interpolation must answer `unknown()`.
#[test]
fn test_zero_padded_decimal_spelling_declines_when_padding_would_actually_fire() {
    let count = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![at_least(8.0), refined_sets::refinement_forms::at_most(12.0)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let Some(kernel) = loaded_kernel() else { return };
    let source = "f\"{count:02d}\"";
    let parsed = parse_expression(source).expect("test source must parse");
    let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
    let mut environment = empty_environment();
    environment.bind("count", count);
    let result = evaluate_fstring(&fstring, &environment, &kernel);
    assert_eq!(result.kind, Kind::Unknown);
}

/// `f"{value:.2f}"` — the fixed-precision decimal grammar
/// (`fixed_precision_decimal_spelling`'s own doc,
/// `string_models::fixed_precision_decimal_grammar`): a Float-sorted
/// `value` with a plain `.2f` spec answers a String-sorted Set, not a
/// decline — the digit-run/point/digit-run shape, disjoint from a
/// two-uppercase-letter grammar like `Code`'s own `/^[A-Z]{2}$/`
/// (A2.xfer.tostring's own row).
#[test]
fn test_fixed_precision_format_spec_answers_the_decimal_grammar() {
    let Some(kernel) = loaded_kernel() else { return };
    let value = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustSpec, SetKindTag::None)
    };
    let source = "f\"{value:.2f}\"";
    let parsed = parse_expression(source).expect("test source must parse");
    let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
    let mut environment = empty_environment();
    environment.bind("value", value);
    let result = evaluate_fstring(&fstring, &environment, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::String));
}

/// A format spec that is not the recognized `0{width}d` or `.{p}f`
/// spelling (here, a fill/align character, `:^10`) declines the
/// whole f-string — neither reader's single-literal-element check
/// admits it.
#[test]
fn test_unrecognized_format_spec_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let value = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustSpec, SetKindTag::None)
    };
    let source = "f\"{value:^10}\"";
    let parsed = parse_expression(source).expect("test source must parse");
    let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
    let mut environment = empty_environment();
    environment.bind("value", value);
    let result = evaluate_fstring(&fstring, &environment, &kernel);
    assert_eq!(result.kind, Kind::Unknown);
}
