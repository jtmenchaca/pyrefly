use refined_domain::abstract_value::{known_set, known_values, Kind, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustProved;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set, repeat_of, Form};
use refined_sets::regex_compiler::format_grammar;

use super::*;

#[test]
fn test_string_literal_value_round_trips_ascii() {
    let value = string_literal_value("ab");
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
    assert_eq!(exact_string_text(&value).as_deref(), Some("ab"));
}

/// "héllo" is 5 Unicode code points ('h','é','l','l','o') — the
/// same count `len("héllo")` gives in CPython, and different from
/// Rust's `"héllo".len()` (6 UTF-8 bytes, because 'é' is two
/// bytes).
#[test]
fn test_string_literal_value_length_is_code_points_not_bytes() {
    let value = string_literal_value("héllo");
    assert_eq!(value.values.len(), 5);
    assert_ne!(value.values.len(), "héllo".len());
}

/// `fixed_precision_decimal_grammar(2)` composes as ONE top-level
/// `Concatenation` form (`concatenation`'s own construction — the
/// same "one Concatenation, nested" shape `codepoint_sets::string_tuple`
/// builds), and different precisions build DIFFERENT grammars — the
/// `precision` parameter must actually reach the fractional-digit
/// repeat bound, not be silently ignored.
#[test]
fn test_fixed_precision_decimal_grammar_varies_with_precision() {
    let two_digits = fixed_precision_decimal_grammar(2);
    assert_eq!(two_digits.forms.len(), 1);
    assert!(matches!(two_digits.forms[0].form, Form::Concatenation));
    let three_digits = fixed_precision_decimal_grammar(3);
    assert_ne!(two_digits, three_digits, "a different precision must build a different grammar");
}

/// `.2f` parses as precision `2`; `02d` (a DIFFERENT reader's own
/// spelling, `zero_padded_decimal_width`'s row) and a fill/align
/// spec (`^10`) are not this reader's grammar at all.
#[test]
fn test_fixed_precision_decimal_width_reads_the_plain_dot_f_spelling() {
    let source = "f\"{x:.2f}\"";
    let parsed = ruff_python_parser::parse_expression(source).expect("test source must parse");
    let ruff_python_ast::Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
    let single = fstring.as_single_part_fstring().expect("single-part f-string");
    let [ruff_python_ast::InterpolatedStringElement::Interpolation(interpolation)] = &*single.elements else {
        panic!("expected one interpolation")
    };
    let format_spec = interpolation.format_spec.as_ref().expect("format spec present");
    assert_eq!(fixed_precision_decimal_width(format_spec), Some(2));
}

#[test]
fn test_upper_no_arg() {
    let receiver = string_literal_value("ab");
    let result = string_method_result("upper", &receiver, &[]).expect("upper must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("AB"));
}

#[test]
fn test_lower_no_arg() {
    let receiver = string_literal_value("AB");
    let result = string_method_result("lower", &receiver, &[]).expect("lower must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
}

#[test]
fn test_strip_no_arg() {
    let receiver = string_literal_value("  ab  ");
    let result = string_method_result("strip", &receiver, &[]).expect("strip must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
}

#[test]
fn test_lstrip_no_arg() {
    let receiver = string_literal_value("  ab");
    let result = string_method_result("lstrip", &receiver, &[]).expect("lstrip must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
}

#[test]
fn test_rstrip_no_arg() {
    let receiver = string_literal_value("ab  ");
    let result = string_method_result("rstrip", &receiver, &[]).expect("rstrip must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
}

/// str.replace with no count replaces EVERY occurrence — the
/// brief's confirmed fact, distinct from a single-substitution
/// replace.
#[test]
fn test_replace_all_occurrences() {
    let receiver = string_literal_value("abXcdXef");
    let old = string_literal_value("X");
    let new = string_literal_value("Y");
    let result =
        string_method_result("replace", &receiver, &[old, new]).expect("replace must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("abYcdYef"));
}

#[test]
fn test_startswith_true() {
    let receiver = string_literal_value("banana");
    let prefix = string_literal_value("ban");
    let result =
        string_method_result("startswith", &receiver, &[prefix]).expect("startswith must decide");
    assert_eq!(result.values, vec![1.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Boolean));
}

#[test]
fn test_startswith_false() {
    let receiver = string_literal_value("banana");
    let prefix = string_literal_value("apple");
    let result =
        string_method_result("startswith", &receiver, &[prefix]).expect("startswith must decide");
    assert_eq!(result.values, vec![0.0]);
}

#[test]
fn test_endswith_true() {
    let receiver = string_literal_value("banana");
    let suffix = string_literal_value("ana");
    let result = string_method_result("endswith", &receiver, &[suffix]).expect("endswith must decide");
    assert_eq!(result.values, vec![1.0]);
}

#[test]
fn test_find_hit() {
    let receiver = string_literal_value("banana");
    let needle = string_literal_value("a");
    let result = string_method_result("find", &receiver, &[needle]).expect("find must decide");
    assert_eq!(result.values, vec![1.0]);
    // Integer, not bare Number: str.find always returns a Python int
    // (the found index or -1), so its result can feed a slice bound
    // (expressions.rs's slice_bound_index requires Integer sort).
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// str.find answers -1 on a missing needle — the twin of JS
/// `indexOf`, never a raised exception (that is str.index's row).
#[test]
fn test_find_miss_answers_negative_one() {
    let receiver = string_literal_value("banana");
    let needle = string_literal_value("z");
    let result = string_method_result("find", &receiver, &[needle]).expect("find must decide");
    assert_eq!(result.values, vec![-1.0]);
}

/// find's index counts CODE POINTS: "é" is one position, so the "l"
/// after it is at index 2, not 3 (which a byte-offset find would
/// give, since "é" is two UTF-8 bytes).
#[test]
fn test_find_counts_code_points_not_bytes() {
    let receiver = string_literal_value("héllo");
    let needle = string_literal_value("l");
    let result = string_method_result("find", &receiver, &[needle]).expect("find must decide");
    assert_eq!(result.values, vec![2.0]);
}

/// str.index on a present needle answers the same position find
/// would — the c-reads-and-values.py string_index row's in-set leg.
#[test]
fn test_index_hit_answers_the_found_position() {
    let receiver = string_literal_value("banana");
    let needle = string_literal_value("a");
    let result = string_method_result("index", &receiver, &[needle]).expect("index must decide");
    assert_eq!(result.values, vec![1.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// str.index on a missing needle declines — the miss is a raise
/// (ValueError), not a value this function answers.
#[test]
fn test_index_miss_declines() {
    let receiver = string_literal_value("banana");
    let needle = string_literal_value("z");
    assert_eq!(string_method_result("index", &receiver, &[needle]), None);
}

/// casefold on an ASCII-only receiver matches plain lowercasing
/// exactly — ASCII has no case mapping the two diverge on.
#[test]
fn test_casefold_ascii_matches_lowercase() {
    let receiver = string_literal_value("AbC");
    let result = string_method_result("casefold", &receiver, &[]).expect("casefold(ascii) must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("abc"));
}

/// casefold declines outside ASCII: German "ß" casefolds to "ss"
/// (length-changing), which plain `to_lowercase` does not produce —
/// stdtypes.rst's own worked example for why casefold and lower
/// diverge.
#[test]
fn test_casefold_non_ascii_declines() {
    let receiver = string_literal_value("stra\u{df}e");
    assert_eq!(string_method_result("casefold", &receiver, &[]), None);
}

/// A non-exact-string receiver (unknown) declines every row.
#[test]
fn test_non_string_receiver_declines() {
    let receiver = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
    assert_eq!(string_method_result("upper", &receiver, &[]), None);
}

/// The unbounded whole-strings ground — `s: str`'s own seed
/// (`typereading::base_sort_return_refinement`) — as this test
/// module's own Set-shaped receiver.
fn any_string_receiver() -> AbstractValue {
    known_set(strings(), None, TrustProved, SetKindTag::None)
}

/// `s.upper()` over an unbounded receiver answers Σ* — `string_
/// method_result`'s own exact row already declined (no exact text to
/// read), so this is the sort-only fallback `A3.xfer.case`'s own row
/// needs: the method still names a real `str`-sorted claim rather
/// than declining the whole call.
#[test]
fn test_sort_only_upper_over_an_unbounded_receiver_answers_any_string() {
    let receiver = any_string_receiver();
    let result = string_method_sort_only_result("upper", &receiver, &[]).expect("upper must decide the sort");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(exact_string_text(&result), None, "the answer states no exact content");
}

/// F2.fixed/F2.dead/F2.select's own shape: `len(x) == 2 and
/// x.isascii() and x.islower()` narrows `x` to a length-2 repetition
/// of the ASCII lowercase window (`narrowing.rs`'s
/// `narrow_ascii_case_conjunction`, `[0x61, 0x7A]`) — `x.upper()`
/// over THAT receiver must answer the mapped ASCII UPPERCASE window
/// at the SAME length, not the unbounded `Σ*` fallback: the mapped
/// answer is exactly `Code`'s own declared set
/// (`(>= 65 && <= 90 && integer) × exactly 2`), so this is the
/// difference between a determined pass and the RTS7001 mismatch
/// those three rows previously answered.
fn ascii_lowercase_pair_receiver() -> AbstractValue {
    let element = make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]);
    let set = make_refined_set(vec![repeat_of(element, 2, Some(2))]);
    known_set(set, None, TrustProved, SetKindTag::None)
}

#[test]
fn test_upper_over_a_narrowed_ascii_lowercase_pair_answers_the_mapped_uppercase_window() {
    let receiver = ascii_lowercase_pair_receiver();
    let result = string_method_sort_only_result("upper", &receiver, &[]).expect("upper must decide the mapped window");
    let expected_element = make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]);
    let expected = make_refined_set(vec![repeat_of(expected_element, 2, Some(2))]);
    assert_eq!(result.set, expected, "x.upper() must map the ASCII window, keeping the length-2 bound");
}

/// The lower-case twin: an ASCII UPPERCASE pair's `.lower()` maps to
/// the lowercase window, same length-2 bound preserved.
#[test]
fn test_lower_over_a_narrowed_ascii_uppercase_pair_answers_the_mapped_lowercase_window() {
    let element = make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]);
    let set = make_refined_set(vec![repeat_of(element, 2, Some(2))]);
    let receiver = known_set(set, None, TrustProved, SetKindTag::None);
    let result = string_method_sort_only_result("lower", &receiver, &[]).expect("lower must decide the mapped window");
    let expected_element = make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]);
    let expected = make_refined_set(vec![repeat_of(expected_element, 2, Some(2))]);
    assert_eq!(result.set, expected, "x.lower() must map the ASCII window, keeping the length-2 bound");
}

/// A receiver narrowed to a window OTHER than the two ASCII
/// cased-letter windows (e.g. digits) states no mapped image — this
/// row declines to the caller's own `Σ*` fallback rather than
/// guessing a case mapping for uncased code points.
#[test]
fn test_upper_over_a_non_cased_window_falls_back_to_any_string() {
    let element = make_refined_set(vec![integer(), at_least(0x30 as f64), at_most(0x39 as f64)]);
    let set = make_refined_set(vec![repeat_of(element, 2, Some(2))]);
    let receiver = known_set(set, None, TrustProved, SetKindTag::None);
    let result = string_method_sort_only_result("upper", &receiver, &[]).expect("upper must still decide the Σ* fallback");
    assert_eq!(result.set, strings(), "a non-cased window's own .upper() falls back to Σ*, not a guessed mapping");
}

/// `s.replace("a", "b")`/`s.strip()`/`s.zfill(4)` over an unbounded
/// receiver all answer the same Σ* sort-only claim — `A3.xfer.
/// replace`/`A3.xfer.trim`/`A3.xfer.pad`'s own rows.
#[test]
fn test_sort_only_replace_strip_zfill_all_answer_any_string() {
    let receiver = any_string_receiver();
    let replace = string_method_sort_only_result("replace", &receiver, &[string_literal_value("a"), string_literal_value("b")])
        .expect("replace must decide the sort");
    assert_eq!(replace.kind, Kind::Set);
    let strip = string_method_sort_only_result("strip", &receiver, &[]).expect("strip must decide the sort");
    assert_eq!(strip.kind, Kind::Set);
    let zfill = string_method_sort_only_result("zfill", &receiver, &[known_values(vec![4.0], PrimitiveKind::Integer, TrustProved)])
        .expect("zfill must decide the sort");
    assert_eq!(zfill.kind, Kind::Set);
}

/// A method this file states no sort-only claim for still declines,
/// matching `string_method_result`'s own "not modeled" honesty at this
/// precision — `startswith` answers a `bool`, a sort this file speaks
/// no unread-receiver claim for.
#[test]
fn test_sort_only_declines_a_method_with_no_string_sorted_claim() {
    let receiver = any_string_receiver();
    assert_eq!(string_method_sort_only_result("startswith", &receiver, &[string_literal_value("a")]), None);
}

/// A3.xfer.split's own `split_length_outside`: `s.split(",")` over an
/// unread receiver states no PIECES but does state the piece-count
/// floor — "Splitting an empty string with a specified separator
/// returns `['']`" (stdtypes.rst), so every split yields at least one
/// piece and `len(s.split(","))` reads `[1, +inf)`.
#[test]
fn test_sort_only_split_states_a_piece_count_floor_of_one() {
    let receiver = any_string_receiver();
    let result = string_method_sort_only_result("split", &receiver, &[string_literal_value(",")]).expect("split must state the window");
    let window = refined_sets::repetition_window_forms::as_repetition(&result.set).expect("the answer is a repetition");
    assert_eq!(window.lo, 1);
    assert_eq!(window.hi, None);
}

/// A3.xfer.join's own `join_outside`/`join_codes_outside`: `"-".join(
/// parts)` over an iterable this file cannot read element-wise still
/// answers a `str`-sorted `Σ*` rather than declining the whole call.
#[test]
fn test_sort_only_join_over_an_unread_iterable_answers_any_string() {
    let separator = string_literal_value("-");
    let parts = known_set(strings(), None, TrustProved, SetKindTag::None);
    let result = string_method_sort_only_result("join", &separator, &[parts]).expect("join must decide the sort");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::String));
    assert_eq!(result.set, strings());
}

/// A3.xfer.pad's own `zfill_inside`: `digits.zfill(4)` over a receiver
/// narrowed to `Digits`'s own `/^[0-9]+$/` left-fills ASCII `'0'`
/// digits onto a digit-only string with no sign to insert after, so
/// every code point stays inside `[0x30, 0x39]` and the length rises to
/// at least 4 — still inside `Digits`, rather than the `Σ*` fallback
/// that refused it.
#[test]
fn test_zfill_over_a_digit_repetition_keeps_the_digit_window() {
    let element = make_refined_set(vec![integer(), at_least(0x30 as f64), at_most(0x39 as f64)]);
    let receiver = known_set(make_refined_set(vec![repeat_of(element.clone(), 1, None)]), None, TrustProved, SetKindTag::None);
    let width = known_values(vec![4.0], PrimitiveKind::Integer, TrustProved);
    let result = string_method_sort_only_result("zfill", &receiver, &[width]).expect("zfill must decide the window");
    let expected = make_refined_set(vec![repeat_of(element, 4, None)]);
    assert_eq!(result.set, expected, "zfill raises the length floor to the width and keeps the digit window");
}

/// A3.xfer.replace's own `replace_first_outside`: the three-argument
/// row replaces only the first `count` occurrences — `"AAB".replace(
/// "A", "B", 1)` is exactly `"BAB"`, not `"BBB"`.
#[test]
fn test_replace_with_a_count_replaces_only_the_leading_occurrences() {
    let receiver = string_literal_value("AAB");
    let arguments = [
        string_literal_value("A"),
        string_literal_value("B"),
        known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
    ];
    let result = string_method_result("replace", &receiver, &arguments).expect("replace with a count must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("BAB"));
}

/// A3.xfer.split's own `split_limit_caps_splits_outside`: `maxsplit`
/// caps the SPLIT COUNT, so the remainder stays whole in the last
/// element — `"a,b,c".split(",", 1)` is `["a", "b,c"]`, two pieces.
#[test]
fn test_split_with_a_maxsplit_keeps_the_remainder_in_the_last_piece() {
    let receiver = string_literal_value("a,b,c");
    let arguments = [string_literal_value(","), known_values(vec![1.0], PrimitiveKind::Integer, TrustProved)];
    let result = string_method_result("split", &receiver, &arguments).expect("split with a maxsplit must decide");
    assert_eq!(result.items.len(), 2);
    assert_eq!(exact_string_text(&result.items[0]).as_deref(), Some("a"));
    assert_eq!(exact_string_text(&result.items[1]).as_deref(), Some("b,c"));
}

/// A3.xfer.pad's own `ljust_outside_digits3`: `"12".ljust(4, "0")` is
/// exactly `"1200"` — padding goes on the RIGHT for `ljust`.
#[test]
fn test_ljust_pads_on_the_right_to_the_given_width() {
    let receiver = string_literal_value("12");
    let arguments = [known_values(vec![4.0], PrimitiveKind::Integer, TrustProved), string_literal_value("0")];
    let result = string_method_result("ljust", &receiver, &arguments).expect("ljust must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("1200"));
}

/// "The original string is returned if *width* is less than or equal
/// to `len(s)`" (stdtypes.rst, str.ljust) — no truncation.
#[test]
fn test_ljust_with_a_width_below_the_length_returns_the_original() {
    let receiver = string_literal_value("12345");
    let arguments = [known_values(vec![2.0], PrimitiveKind::Integer, TrustProved), string_literal_value("0")];
    let result = string_method_result("ljust", &receiver, &arguments).expect("ljust must decide");
    assert_eq!(exact_string_text(&result).as_deref(), Some("12345"));
}

/// `s.find("z")` over an unbounded receiver answers an Integer-sorted
/// `[-1, +inf)` claim — `A3.xfer.search`'s own row: `find` never
/// raises, so this sound bound is the whole real answer.
#[test]
fn test_sort_only_find_over_an_unbounded_receiver_answers_an_integer_ray() {
    let result = string_method_int_sort_only_result("find", &[string_literal_value("z")]).expect("find must decide the sort");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// replace with a non-exact-string argument declines rather than
/// guessing.
#[test]
fn test_replace_with_unknown_argument_declines() {
    let receiver = string_literal_value("abXcd");
    let old = string_literal_value("X");
    let new = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
    assert_eq!(string_method_result("replace", &receiver, &[old, new]), None);
}

#[test]
fn test_split_by_string_separator() {
    let receiver = string_literal_value("ab,cd,ef");
    let sep = string_literal_value(",");
    let result = string_method_result("split", &receiver, &[sep]).expect("split must decide");
    assert_eq!(result.kind, Kind::List);
    assert_eq!(result.items.len(), 3);
    assert_eq!(exact_string_text(&result.items[0]).as_deref(), Some("ab"));
    assert_eq!(exact_string_text(&result.items[1]).as_deref(), Some("cd"));
    assert_eq!(exact_string_text(&result.items[2]).as_deref(), Some("ef"));
}

/// consecutive delimiters delimit an empty string, matching
/// stdtypes.rst's own worked example ("'1,,2'.split(',')" -> ['1',
/// '', '2']).
#[test]
fn test_split_consecutive_delimiters_yield_an_empty_element() {
    let receiver = string_literal_value("1,,2");
    let sep = string_literal_value(",");
    let result = string_method_result("split", &receiver, &[sep]).expect("split must decide");
    assert_eq!(result.items.len(), 3);
    assert_eq!(exact_string_text(&result.items[1]).as_deref(), Some(""));
}

#[test]
fn test_split_empty_separator_declines() {
    let receiver = string_literal_value("ab");
    let sep = string_literal_value("");
    assert_eq!(string_method_result("split", &receiver, &[sep]), None);
}

/// A3.xfer.encode's own `encoded_length_inside`: an EXACT receiver
/// answers its exact UTF-8 bytes, so `len(s.encode())` reads the BYTE
/// count — `"é"` is one code point but two bytes, `⟨0xC3, 0xA9⟩`.
#[test]
fn test_encode_on_an_exact_receiver_answers_the_exact_utf8_bytes() {
    let receiver = string_literal_value("ab");
    let got = string_method_result("encode", &receiver, &[]).expect("encode must decide");
    assert_eq!(got.kind, Kind::List);
    let bytes: Vec<f64> = got.items.iter().map(|item| item.values[0]).collect();
    assert_eq!(bytes, vec![0x61 as f64, 0x62 as f64]);

    let accented = string_literal_value("é");
    let got = string_method_result("encode", &accented, &[]).expect("encode must decide");
    let bytes: Vec<f64> = got.items.iter().map(|item| item.values[0]).collect();
    assert_eq!(bytes, vec![0xC3 as f64, 0xA9 as f64], "one code point, two UTF-8 bytes");
}

#[test]
fn test_encode_sort_only_on_an_unread_receiver_answers_the_opaque_bytes_state() {
    let receiver = known_set(strings(), None, TrustProved, SetKindTag::None);
    let got = string_method_sort_only_result("encode", &receiver, &[]).expect("encode must decide sort-only");
    assert_eq!(got.kind, Kind::Object);
    assert_eq!(got.kind_word, Some(crate::bytes_models::ENCODED_BYTES_WORD));
}

#[test]
fn test_encode_with_an_argument_declines() {
    let receiver = string_literal_value("ab");
    let encoding = string_literal_value("utf-8");
    assert_eq!(string_method_result("encode", &receiver, &[encoding]), None);
}

#[test]
fn capture_group_spans_reads_two_plain_groups_in_order() {
    let got = capture_group_spans(r"(\d+)-(\d+)").expect("two plain groups parse");
    let bodies: Vec<String> = got.iter().map(|g| g.body.clone()).collect();
    assert_eq!(bodies, vec![r"\d+".to_owned(), r"\d+".to_owned()]);
    assert!(got.iter().all(|g| g.name.is_none()));
}

#[test]
fn capture_group_spans_skips_a_non_capturing_group() {
    let got = capture_group_spans(r"(?:\d+)-([a-z]+)").expect("one capturing group parses");
    let bodies: Vec<String> = got.iter().map(|g| g.body.clone()).collect();
    assert_eq!(bodies, vec!["[a-z]+".to_owned()]);
}

/// A3.xfer.capture's own named-group pattern: each `(?P<name>...)` is
/// numbered exactly like a plain group AND carries its symbolic name.
#[test]
fn capture_group_spans_reads_named_groups_by_number_and_name() {
    let got = capture_group_spans(r"(?P<code>[A-Z]{2})-(?P<digits>\d+)").expect("named groups parse");
    let bodies: Vec<String> = got.iter().map(|g| g.body.clone()).collect();
    assert_eq!(bodies, vec!["[A-Z]{2}".to_owned(), r"\d+".to_owned()]);
    let names: Vec<Option<String>> = got.iter().map(|g| g.name.clone()).collect();
    assert_eq!(names, vec![Some("code".to_owned()), Some("digits".to_owned())]);
}

#[test]
fn capture_group_spans_answers_empty_for_a_group_free_pattern() {
    let got = capture_group_spans(r"\d+").expect("a group-free pattern parses");
    assert!(got.is_empty());
}

#[test]
fn capture_group_spans_declines_on_an_unmatched_paren() {
    assert!(capture_group_spans(r"(\d+").is_none());
}

#[test]
fn match_object_value_carries_group_0_and_every_numbered_group() {
    let got = match_object_value(r"(\d+)-(\d+)").expect("(\\d+)-(\\d+) compiles");
    assert_eq!(got.kind, Kind::Object);
    assert_eq!(got.kind_word, Some(MATCH_WITH_GROUPS_WORD));
    assert_eq!(got.keys.len(), 3);
    assert!(got.keys.iter().any(|k| k.name == "0"));
    assert!(got.keys.iter().any(|k| k.name == "1"));
    assert!(got.keys.iter().any(|k| k.name == "2"));
}

/// A3.xfer.matchall's own `finditer_inside`: every group of a match —
/// group 0 included — is the text the match SPANS, so `re.finditer(
/// r"[A-Z]{2}", ...)`'s own `m.group(0)` is exactly two upper-case
/// letters, never two letters padded with the surrounding context an
/// unanchored compile would admit. The padded reading refused an
/// admitted `Code` value at A3.xfer.matchall.py:19.
#[test]
fn match_object_value_group_0_is_the_matched_span_not_a_padded_substring() {
    let got = match_object_value(r"[A-Z]{2}").expect("[A-Z]{2} compiles");
    assert_eq!(got.keys.len(), 1);
    assert_eq!(got.keys[0].name, "0");
    let anchored_code = format_grammar("^[A-Z]{2}$", "");
    assert!(anchored_code.ok);
    assert_eq!(got.keys[0].value.set, anchored_code.set);
}

/// A3.xfer.capture's own `named_group_inside`: `m.group("code")` on
/// `(?P<code>[A-Z]{2})-(?P<digits>\d+)` reads the named group's own
/// grammar, exactly what the same group's number reads.
#[test]
fn matched_group_grammar_reads_a_named_group_by_string_argument() {
    let receiver = match_object_value(r"(?P<code>[A-Z]{2})-(?P<digits>\d+)").expect("compiles");
    let by_name = matched_group_grammar(&receiver, &[string_literal_value("code")]).expect("group(\"code\") must decide");
    let group_one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let by_number = matched_group_grammar(&receiver, &[group_one]).expect("group(1) must decide");
    assert_eq!(by_name.set, by_number.set);
    assert_eq!(by_name.kind_tag, Some(PrimitiveKind::String));
}

#[test]
fn matched_group_grammar_declines_an_undeclared_group_name() {
    let receiver = match_object_value(r"(?P<code>[A-Z]{2})").expect("compiles");
    let got = matched_group_grammar(&receiver, &[string_literal_value("missing")]);
    assert!(got.is_none(), "a name the pattern never declares should decline: {got:?}");
}

#[test]
fn matched_group_grammar_reads_the_numbered_group_by_known_integer_argument() {
    let receiver = match_object_value(r"(\d+)-(\d+)").expect("compiles");
    let group_one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let got = matched_group_grammar(&receiver, &[group_one]).expect("group(1) must decide");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
}

#[test]
fn matched_group_grammar_out_of_range_declines() {
    let receiver = match_object_value(r"\d+").expect("compiles");
    let group_five = known_values(vec![5.0], PrimitiveKind::Integer, TrustProved);
    let got = matched_group_grammar(&receiver, &[group_five]);
    assert!(got.is_none(), "group(5) on a pattern with no such group should decline: {got:?}");
}

#[test]
fn matched_group_grammar_on_a_non_match_receiver_declines() {
    let receiver = string_literal_value("not a match");
    let group_zero = known_values(vec![0.0], PrimitiveKind::Integer, TrustProved);
    assert!(matched_group_grammar(&receiver, &[group_zero]).is_none());
}
