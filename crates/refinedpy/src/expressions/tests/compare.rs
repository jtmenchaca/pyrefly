use super::*;

#[test]
fn test_chained_comparison_true() {
    let Some(value) = eval("1 < 2 <= 2") else { return };
    assert_eq!(value.values, vec![1.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
}

#[test]
fn test_chained_comparison_false() {
    // 1 < 2 <= 2 is True, but 1 < 2 <= 1 is False (second pair fails)
    let Some(value) = eval("1 < 2 <= 1") else { return };
    assert_eq!(value.values, vec![0.0]);
}

#[test]
fn test_string_comparison() {
    let Some(equal) = eval("\"ab\" == \"ab\"") else { return };
    assert_eq!(equal.values, vec![1.0]);

    let Some(less) = eval("\"ab\" < \"ac\"") else { return };
    assert_eq!(less.values, vec![1.0]);
}

#[test]
fn test_is_none() {
    let Some(is_none) = eval("None is None") else { return };
    assert_eq!(is_none.values, vec![1.0]);

    let Some(value_is_none) = eval("1 is None") else { return };
    assert_eq!(value_is_none.values, vec![0.0]);
}

#[test]
fn test_in_over_list_literal() {
    let Some(present) = eval("2 in [1, 2, 3]") else { return };
    assert_eq!(present.values, vec![1.0]);

    let Some(absent) = eval("5 in [1, 2, 3]") else { return };
    assert_eq!(absent.values, vec![0.0]);
}

// --- numeric_value_vs_window_compare ---

/// `len(padded) >= 3` where `padded` is a `[:3]` prefix window — the
/// exact construct `text_label.py`'s `return padded if len(padded)
/// >= 3 else "xxx"` compares. `len()` over a `Repeat(alphabet, 3, 3)`
/// window (`collection_models::len_result`'s own reading of a
/// DEGENERATE bound) answers a bounded Integer `Kind::Set`, `{AtLeast
/// 3, AtMost 3}` — never a single known value — so this decides only
/// through `numeric_value_vs_window_compare`'s own window arm, not
/// `compare_pair`'s exact-numeric row. Every admitted length (all of
/// them, since the window is degenerate) satisfies `>= 3`, so the
/// comparison decides `True`.
#[test]
fn test_compare_decides_over_a_degenerate_length_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let seed_window = make_refined_set(vec![repeat_of(
        refined_sets::codepoint_sets::codepoints(),
        1,
        Some(8),
    )]);
    let literal = refined_sets::codepoint_sets::string_tuple("xxxxxxxx");
    let concatenation_with_a_leading_window = make_refined_set(vec![
        refined_sets::refinement_forms::concatenation(seed_window, literal),
    ]);
    let receiver = AbstractValue {
        kind_tag: None,
        ..known_set(concatenation_with_a_leading_window, None, TrustProved, SetKindTag::None)
    };
    let mut environment = empty_environment();
    environment.bind("padded", receiver);
    let sliced_parsed = parse_expression("padded[:3]").expect("test source must parse");
    let Expr::Subscript(subscript) = sliced_parsed.into_expr() else { panic!("expected a Subscript") };
    let sliced = evaluate_subscript(&subscript, &environment, &kernel);
    assert_eq!(sliced.kind, Kind::Set, "the [:3] slice must admit now that the kernel recognizes the shape");
    environment.bind("sliced", sliced);

    let compare_parsed = parse_expression("len(sliced) >= 3").expect("test source must parse");
    let compare_value = evaluate_expression(&compare_parsed.into_expr(), &environment, &kernel);
    assert_eq!(compare_value.kind, Kind::Values, "the comparison must decide, not stay unknown: {compare_value:?}");
    assert_eq!(
        compare_value.values,
        vec![1.0],
        "len(a 3-length window) >= 3 must decide True: {compare_value:?}"
    );
}

/// A window that only SOMETIMES satisfies the comparison (`[0, 5]`
/// against `>= 3`) must stay undecided — some admitted lengths (0,
/// 1, 2) fail the bound while others (3, 4, 5) pass it, and this
/// function never guesses across a partial overlap.
#[test]
fn test_compare_stays_undecided_over_a_window_straddling_the_bound() {
    let straddling_window = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let Some(kernel) = loaded_kernel() else { return };
    let three = known_values(vec![3.0], PrimitiveKind::Integer, TrustProved);
    assert_eq!(
        compare_pair(CmpOp::GtE, &straddling_window, &three, &kernel),
        None,
        "a window straddling the bound must not decide >="
    );
    assert_eq!(
        compare_pair(CmpOp::GtE, &three, &straddling_window, &kernel),
        None,
        "the swapped operand order must not decide either"
    );
}

/// A window entirely BELOW the target decides `<`/`<=` true and
/// `>`/`>=` false — the mirror of the degenerate-window admit case,
/// pinning the non-degenerate ordering rows and the swapped operand
/// order together.
#[test]
fn test_compare_decides_over_a_window_entirely_below_the_target() {
    let low_window = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let Some(kernel) = loaded_kernel() else { return };
    let three = known_values(vec![3.0], PrimitiveKind::Integer, TrustProved);
    assert_eq!(compare_pair(CmpOp::Lt, &low_window, &three, &kernel), Some(1.0), "[0,2] < 3 must decide True");
    assert_eq!(compare_pair(CmpOp::GtE, &low_window, &three, &kernel), Some(0.0), "[0,2] >= 3 must decide False");
    // swapped: `3 > window` is the same claim as `window < 3`
    assert_eq!(compare_pair(CmpOp::Gt, &three, &low_window, &kernel), Some(1.0), "3 > [0,2] must decide True");
}

/// A3.sink.dead's own shape: after `re.fullmatch(r"[0-9]+", s)`, `s`
/// carries the digit grammar, so `s == "abc"` can never be true — the
/// arm it guards is dead, and the dead-branch law needs this pair
/// decided False to say so.
#[test]
fn test_compare_decides_an_exact_string_outside_a_narrowed_grammar() {
    let Some(kernel) = loaded_kernel() else { return };
    let digit = make_refined_set(vec![
        refined_sets::refinement_forms::integer(),
        at_least(0x30 as f64),
        refined_sets::refinement_forms::at_most(0x39 as f64),
    ]);
    let digits = AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(
            refined_sets::repetition_window_forms::repetition(digit, 1, None),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    let abc = known_values("abc".chars().map(|c| c as u32 as f64).collect(), PrimitiveKind::String, TrustProved);
    assert_eq!(compare_pair(CmpOp::Eq, &digits, &abc, &kernel), Some(0.0), "[0-9]+ == \"abc\" must decide False");
    assert_eq!(compare_pair(CmpOp::NotEq, &digits, &abc, &kernel), Some(1.0), "[0-9]+ != \"abc\" must decide True");
    // the swapped operand order states the same claim
    assert_eq!(compare_pair(CmpOp::Eq, &abc, &digits, &kernel), Some(0.0), "the swapped order must decide too");
    // a string the grammar DOES admit stays undecided — the set states
    // what is possible, never which value a run holds
    let twelve = known_values("12".chars().map(|c| c as u32 as f64).collect(), PrimitiveKind::String, TrustProved);
    assert_eq!(compare_pair(CmpOp::Eq, &digits, &twelve, &kernel), None, "an admitted string must not decide ==");
}

// --- `in`/`not in` over a `set[X]` parameter's REPETITION-WINDOW
// receiver (A15.xfer.add's own `s: set[int]` seed shape) ---

/// The bare unbounded-window `Kind::Set` a `set[int]` parameter seeds —
/// `check.rs::seed_parameters`'s own sequence-container arm, mirrored
/// here as `collection_models::tests::mutation::bounded_ints(0, None)`'s
/// twin for THIS test module (no shared private helper across the two
/// test files, matching this crate's own no-shared-private-helper
/// convention already stated elsewhere).
fn unbounded_int_set() -> AbstractValue {
    let whole_ints = make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(f64::NEG_INFINITY)]);
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(refined_sets::repetition_window_forms::repetition(whole_ints, 0, None), None, TrustProved, SetKindTag::None)
    }
}

/// `set.add(x)` then `x in s` answers exactly `True` — the recorded
/// entry `list_set_mutation::set_mutated_receiver`'s own `add` arm
/// writes into `.keys` is read back here the same way a dict-star's
/// written key already is (`compare_pair`'s `Kind::ObjectStar` arm, one
/// row up from the new `Kind::Set` arm this pins).
#[test]
fn test_in_over_a_repetition_window_answers_true_for_a_recorded_element() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut receiver = unbounded_int_set();
    receiver.keys.push(refined_domain::abstract_value::ObjectKey {
        name: "9".to_owned(),
        numeric: true,
        value: refined_domain::abstract_value::null_value(),
    });
    let nine = known_values(vec![9.0], PrimitiveKind::Integer, TrustProved);
    assert_eq!(compare_pair(CmpOp::In, &nine, &receiver, &kernel), Some(1.0), "a recorded element must decide `in` True");
    assert_eq!(compare_pair(CmpOp::NotIn, &nine, &receiver, &kernel), Some(0.0), "a recorded element must decide `not in` False");
}

/// An element with NO recorded entry over the same window declines —
/// the window states only what the set MIGHT hold, never that a named
/// value is absent, so `compare_pair` returns `None` and
/// `evaluate_compare`'s own caller-side fallback is what turns that
/// into the exact two-member boolean domain (pinned in
/// `test_in_over_a_repetition_window_with_no_recorded_entry_is_the_two_
/// valued_boolean_ground` below, through the full `evaluate_expression`
/// path rather than `compare_pair` alone, since the two-valued fallback
/// itself lives in `evaluate_compare`, not `compare_pair`).
#[test]
fn test_in_over_a_repetition_window_with_no_recorded_entry_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let receiver = unbounded_int_set();
    let twelve_thousand_three_forty_five = known_values(vec![12345.0], PrimitiveKind::Integer, TrustProved);
    assert_eq!(
        compare_pair(CmpOp::In, &twelve_thousand_three_forty_five, &receiver, &kernel),
        None,
        "an unrecorded element must not decide `in` at the compare_pair layer"
    );
}

/// The full expression path: `12345 in s` over a repetition-window `s`
/// with no recorded entry for `12345` answers the exact two-member
/// boolean domain `{0, 1}` — `evaluate_compare`'s own single-pair
/// fallback (expressions/subscript.rs), reached because `compare_pair`
/// declines. This is A15.xfer.add's `other_membership_unchanged` row:
/// `other = int(12345 in s)` must read `{0, 1}`, never `unknown()`.
#[test]
fn test_in_over_a_repetition_window_with_no_recorded_entry_is_the_two_valued_boolean_ground() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    environment.bind("s", unbounded_int_set());
    let parsed = parse_expression("12345 in s").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Values, "the comparison must decide the two-valued boolean, not stay unknown: {value:?}");
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
    assert_eq!(value.values, vec![0.0, 1.0], "an unrecorded element must answer the boolean ground, not a guessed True/False: {value:?}");
}
