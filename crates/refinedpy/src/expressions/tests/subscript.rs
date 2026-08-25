use super::*;

#[test]
fn test_string_literal() {
    let Some(value) = eval("\"ab\"") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
    assert_eq!(value.values, vec!['a' as u32 as f64, 'b' as u32 as f64]);
}

#[test]
fn test_list_tuple_literal_and_subscript_read() {
    let Some(list_value) = eval("[10, 20, 30]") else { return };
    assert_eq!(list_value.kind, Kind::List);
    assert_eq!(list_value.items.len(), 3);

    let Some(tuple_value) = eval("(1, 2)") else { return };
    assert_eq!(tuple_value.kind, Kind::List);
    assert_eq!(tuple_value.items.len(), 2);

    let Some(subscripted) = eval("[10, 20, 30][1]") else { return };
    assert_eq!(subscripted.values, vec![20.0]);
}

/// `[*xs, 30]` splices a known list's own elements in place, in
/// order (expressions.rst, "List displays").
#[test]
fn test_list_display_starred_element_splices_a_known_list() {
    let Some(value) = eval("[*[200, 201], 30]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(
        value.items,
        vec![
            known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![201.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![30.0], PrimitiveKind::Integer, TrustProved),
        ]
    );
}

#[test]
fn test_list_display_starred_unknown_element_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("[*xs, 30]").expect("test source must parse");
    let environment = empty_environment();
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Unknown);
}

#[test]
fn test_dict_literal_and_subscript_read() {
    let Some(value) = eval("{\"a\": 1, \"b\": 2}[\"b\"]") else { return };
    assert_eq!(value.values, vec![2.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `{**base, "age": 41}` splices a known dict's own entries, then a
/// later ordinary key overwrites the spread's same-named entry —
/// last-value-wins, matching `dict_literal_value`'s own overwrite
/// rule.
#[test]
fn test_dict_display_double_star_spread_merges_and_later_keys_win() {
    let Some(value) = eval("{**{\"age\": 40, \"name\": \"ann\"}, \"age\": 41}") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.keys.len(), 2);
    let age = value.keys.iter().find(|entry| entry.name == "age").expect("age present");
    assert_eq!(age.value.values, vec![41.0]);
}

/// `{**a, **b}` — a LATER spread's same-named key wins over an
/// earlier spread's.
#[test]
fn test_dict_display_two_spreads_later_wins() {
    let Some(value) = eval("{**{\"age\": 40}, **{\"age\": 200}}") else { return };
    assert_eq!(value.keys.len(), 1);
    assert_eq!(value.keys[0].value.values, vec![200.0]);
}

/// `dict.setdefault(key, default)` read as a VALUE: a PRESENT key
/// answers its own value, winning over the default argument.
#[test]
fn test_dict_setdefault_present_key_wins_over_the_default() {
    let Some(value) = eval("{\"bea\": 200}.setdefault(\"bea\", 40)") else { return };
    assert_eq!(value.values, vec![200.0]);
}

#[test]
fn test_dict_setdefault_absent_key_answers_the_default() {
    let Some(value) = eval("{\"ann\": 40}.setdefault(\"bea\", 0)") else { return };
    assert_eq!(value.values, vec![0.0]);
}

/// A subscript past the list's bounds declines: CPython raises
/// `IndexError`, which this file has no channel for
/// (collection_models.rs's own pinned decline).
#[test]
fn test_subscript_out_of_range_declines() {
    let Some(value) = eval("[1, 2][5]") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

// --- string slicing ---

#[test]
fn test_string_slice_basic_range() {
    let Some(value) = eval("\"abcdefgh\"[0:4]") else { return };
    let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
    assert_eq!(text, "abcd");
}

#[test]
fn test_string_slice_clamps_past_the_end_rather_than_raising() {
    let Some(value) = eval("\"abcdefghij\"[0:99]") else { return };
    let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
    assert_eq!(text, "abcdefghij");
}

#[test]
fn test_string_slice_missing_bounds_default_to_whole_string() {
    let Some(value) = eval("\"ab\"[:]") else { return };
    let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
    assert_eq!(text, "ab");
}

#[test]
fn test_string_slice_with_step_declines() {
    let Some(value) = eval("\"abcdef\"[::2]") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

// --- list slicing (item 6, c-reads-and-values.py's list_slice) ---

/// `xs[0:1][0]` — a slice re-subscripted, c-reads-and-values.py's
/// own `list_slice` shape: the slice answers a known one-element
/// list, and the following `[0]` reads its sole element back out.
#[test]
fn test_list_slice_then_subscript_reads_the_sliced_element() {
    let Some(value) = eval("[200, 201][0:1][0]") else { return };
    assert_eq!(value.values, vec![200.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// An out-of-order slice (`lower >= upper` after clamping) answers
/// the empty list, matching the string-slice sibling's same row.
#[test]
fn test_list_slice_empty_range_answers_the_empty_list() {
    let Some(value) = eval("[1, 2, 3][2:1]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items.len(), 0);
}

/// A negative slice bound adjusts by the list's own length first,
/// the same rule the plain-index and string-slice rows already
/// follow.
#[test]
fn test_list_slice_negative_bound_adjusts_by_length() {
    let Some(value) = eval("[10, 20, 30][-2:]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items, vec![known_values(vec![20.0], PrimitiveKind::Integer, TrustProved), known_values(vec![30.0], PrimitiveKind::Integer, TrustProved)]);
}

// --- list.pop() as an RHS value (item 5) ---

/// `overs.pop()` used directly as a value (not first bound to a
/// name) — c-reads-and-values.py's `list_pop` shape: `return
/// overs.pop()`. The RESULT half of `mutated_receiver`'s pair reads
/// through the value-call dispatch, answering the popped element.
#[test]
fn test_list_pop_as_a_value_expression_answers_the_popped_element() {
    let Some(value) = eval("[200, 201].pop()") else { return };
    assert_eq!(value.values, vec![201.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `xs.pop(0)` — the one-argument indexed form also reads through
/// the value path.
#[test]
fn test_list_pop_with_an_index_as_a_value_expression() {
    let Some(value) = eval("[200, 201].pop(0)") else { return };
    assert_eq!(value.values, vec![200.0]);
}

/// `[].pop()` on an empty receiver declines (there is nothing to
/// pop) — the same honesty `mutated_receiver`'s own statement-sink
/// row already carries.
#[test]
fn test_list_pop_on_an_empty_receiver_declines() {
    let Some(value) = eval("[].pop()") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

// --- kernel.seq_prefix / evaluate_slice's [:n] arm ---

/// The kernel ask itself: `seq_prefix` over an UNBOUNDED repetition
/// window (`Repeat(codepoints, 1, None)` — the shape
/// set_functions/subset_seq_shape.lean's `seqOf` recognizes directly
/// via its `.Repeat A lo none` arm) answers a set that itself states
/// a sequence shape, per `prefixReadOf`'s own over-approximation
/// (boundary/exports_sets.lean's `kernelSeqPrefix`).
#[test]
fn test_kernel_seq_prefix_answers_a_sequence_shaped_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let unbounded_window = make_refined_set(vec![repeat_of(
        refined_sets::codepoint_sets::codepoints(),
        1,
        None,
    )]);
    let Some(answered) = (kernel.seq_prefix)(&unbounded_window, 3) else {
        panic!("seqOf-recognized receiver must not decline");
    };
    assert!(
        assignability::states_sequence(&answered),
        "seq_prefix's answer must itself carry a sequence form: {answered:?}"
    );
}

/// The SAME receiver shape `evaluate_slice`'s regression test
/// exercises end to end, pinned here at the bare ask level: a
/// `Concatenation` whose leading operand is a `Repeat` window (the
/// shape `text_label.py`'s own `seed + "xxxxxxxx"` builds).
///
/// Pre-extension, the kernel's `seqOf` recognized a `Concatenation
/// A B` only when `A.scalarB` — a single fixed scalar, never a
/// `Repeat`/`Star` window — so this shape declined regardless of
/// the window's own bound; that was this test's original premise
/// (`test_kernel_seq_prefix_declines_a_concatenation_with_a_leading_
/// window`, now renamed). The kernel extension
/// (`seqWindowOf`/`prefix_read.lean`) now reads a `Concatenation`
/// with a leading `Repeat` window in either operand order, so this
/// now ANSWERS the proved window instead of declining.
#[test]
fn test_kernel_seq_prefix_admits_a_concatenation_with_a_leading_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let seed_window = make_refined_set(vec![repeat_of(
        refined_sets::codepoint_sets::codepoints(),
        1,
        Some(8),
    )]);
    let literal = refined_sets::codepoint_sets::string_tuple("xxxxxxxx");
    let joined = make_refined_set(vec![refined_sets::refinement_forms::concatenation(
        seed_window,
        literal,
    )]);
    let Some(answered) = (kernel.seq_prefix)(&joined, 3) else {
        panic!("a leading-window concatenation must now be seqOf-recognized, not decline");
    };
    assert!(
        assignability::states_sequence(&answered),
        "seq_prefix's answer must itself carry a sequence form: {answered:?}"
    );
}

/// `evaluate_slice`'s `[:n]` admit case: a receiver `Kind::Set` whose
/// own form is the UNBOUNDED repetition window `seqOf` recognizes,
/// sliced `[:3]`, asks `seq_prefix` and binds the answered set —
/// never `unknown()`.
#[test]
fn test_slice_prefix_admits_over_a_seq_of_recognized_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let receiver = AbstractValue {
        kind_tag: None,
        ..known_set(
            make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, None)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let mut environment = empty_environment();
    environment.bind("padded", receiver);
    let parsed = parse_expression("padded[:3]").expect("test source must parse");
    let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
    let result = evaluate_subscript(&subscript, &environment, &kernel);
    assert_eq!(result.kind, Kind::Set, "expected a bound prefix set, got {result:?}");
    assert!(
        assignability::states_sequence(&result.set),
        "the bound prefix must itself carry a sequence form: {:?}",
        result.set
    );
}

/// A `step` slice over the same set-shaped receiver keeps declining —
/// `evaluate_slice`'s own `slice.step.is_some()` gate fires before
/// `sequence_prefix_slice` ever runs, per the mission's own
/// unmodeled-step scope.
#[test]
fn test_slice_prefix_declines_a_step_slice() {
    let Some(kernel) = loaded_kernel() else { return };
    let receiver = AbstractValue {
        kind_tag: None,
        ..known_set(
            make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, None)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let mut environment = empty_environment();
    environment.bind("padded", receiver);
    let parsed = parse_expression("padded[:3:2]").expect("test source must parse");
    let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
    let result = evaluate_subscript(&subscript, &environment, &kernel);
    assert_eq!(result.kind, Kind::Unknown, "a step slice must still decline: {result:?}");
}

/// A NEGATIVE `upper` bound over the same set-shaped receiver
/// declines: `sequence_prefix_slice` refuses `n < 0` rather than
/// asking the kernel a nonsensical prefix length, and the length-based
/// fallback below it has no known length for a `Kind::Set` receiver
/// either, so the whole slice stays `unknown()`.
#[test]
fn test_slice_prefix_declines_a_negative_upper_bound() {
    let Some(kernel) = loaded_kernel() else { return };
    let receiver = AbstractValue {
        kind_tag: None,
        ..known_set(
            make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, None)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let mut environment = empty_environment();
    environment.bind("padded", receiver);
    let parsed = parse_expression("padded[:-1]").expect("test source must parse");
    let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
    let result = evaluate_subscript(&subscript, &environment, &kernel);
    assert_eq!(result.kind, Kind::Unknown, "a negative upper bound must decline: {result:?}");
}

/// The KERNEL's own decline — not a shape `sequence_prefix_slice`'s
/// own gate rejects up front, but a receiver that reaches
/// `kernel.seq_prefix` and gets `None` back from IT — completes
/// without panicking and keeps the length-based fallback exactly as
/// if the `[:n]` arm had never matched.
///
/// The declining shape is a `Difference` operand nested inside a
/// `Concatenation`: `seqWindowOf` (`prefix_read.lean`) reads scalar
/// sets, the empty tuple, `Star`/`Repeat` of a scalar set, and
/// folds `Concatenation`/`Union` of recognized operands — its own
/// doc names `Difference` as the permanent decline ("no window
/// claim is safe there since the removed piece can itself be
/// unbounded"), so the recursive `seqWindowOf A` call on the
/// `Difference` operand gets `none` back and the whole ask
/// declines.
#[test]
fn test_slice_prefix_completes_without_panic_when_the_kernel_itself_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let window_a = make_refined_set(vec![repeat_of(one_char_of("ab"), 1, Some(4))]);
    let window_b = make_refined_set(vec![repeat_of(one_char_of("cd"), 1, Some(4))]);
    let unrecognized_difference_operand =
        make_refined_set(vec![refined_sets::refinement_forms::difference(window_a, window_b)]);
    let literal = refined_sets::codepoint_sets::string_tuple("xxxxxxxx");
    let concatenation_with_a_difference_operand = make_refined_set(vec![
        refined_sets::refinement_forms::concatenation(unrecognized_difference_operand, literal),
    ]);
    // pin the ask-level premise directly: seqWindowOf must still
    // decline this shape, or the rest of the test would be testing
    // nothing
    assert_eq!(
        (kernel.seq_prefix)(&concatenation_with_a_difference_operand, 3),
        None,
        "a Concatenation over a Difference operand must still decline (seqWindowOf's own named permanent decline)"
    );
    let receiver = AbstractValue {
        kind_tag: None,
        ..known_set(concatenation_with_a_difference_operand, None, TrustProved, SetKindTag::None)
    };
    let mut environment = empty_environment();
    environment.bind("padded", receiver);
    let parsed = parse_expression("padded[:3]").expect("test source must parse");
    let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
    // the assertion itself is the regression: a prior version of this
    // arm panicked reaching this call ("kernel: the set is not a
    // recognized sequence shape") instead of returning a value
    let result = evaluate_subscript(&subscript, &environment, &kernel);
    assert_eq!(
        result.kind,
        Kind::Unknown,
        "a kernel-declined prefix must fall through to unknown(), not panic: {result:?}"
    );
}
