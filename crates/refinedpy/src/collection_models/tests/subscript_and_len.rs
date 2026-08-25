// Positive/negative indexing (list, string, dict), unknown-length
// star-shaped set receivers, and len().

use super::*;

// --- positive and negative indexing ---

#[test]
fn subscript_read_positive_index_into_list() {
    let list = list_literal_value(&[integer(10.0), integer(20.0), integer(30.0)]);
    assert_eq!(subscript_read(&list, &integer(0.0)), Some(integer(10.0)));
    assert_eq!(subscript_read(&list, &integer(2.0)), Some(integer(30.0)));
}

#[test]
fn subscript_read_negative_index_into_list() {
    // x[-1] selects the last item — expressions.rst, "Subscriptions."
    let list = list_literal_value(&[integer(10.0), integer(20.0), integer(30.0)]);
    assert_eq!(subscript_read(&list, &integer(-1.0)), Some(integer(30.0)));
    assert_eq!(subscript_read(&list, &integer(-3.0)), Some(integer(10.0)));
}

#[test]
fn subscript_read_out_of_range_index_declines() {
    let list = list_literal_value(&[integer(10.0)]);
    assert_eq!(subscript_read(&list, &integer(1.0)), None);
    assert_eq!(subscript_read(&list, &integer(-2.0)), None);
}

#[test]
fn subscript_read_positive_index_into_exact_string() {
    // word[0] on "banana" — single-character indexing, the
    // c-reads-and-values.py string_index_access row's own shape.
    let word = string("banana");
    assert_eq!(subscript_read(&word, &integer(0.0)), Some(string("b")));
    assert_eq!(subscript_read(&word, &integer(5.0)), Some(string("a")));
}

#[test]
fn subscript_read_negative_index_into_exact_string() {
    // word[-1] selects the last character — the same negative-index
    // adjustment list_index_read already applies.
    let word = string("banana");
    assert_eq!(subscript_read(&word, &integer(-1.0)), Some(string("a")));
    assert_eq!(subscript_read(&word, &integer(-6.0)), Some(string("b")));
}

#[test]
fn subscript_read_out_of_range_string_index_declines() {
    // word[99] — past the end; IndexError at runtime, no value here.
    let word = string("banana");
    assert_eq!(subscript_read(&word, &integer(99.0)), None);
    assert_eq!(subscript_read(&word, &integer(-99.0)), None);
}

#[test]
fn subscript_read_string_key_into_dict() {
    let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
    assert_eq!(subscript_read(&dict, &string("k")), Some(integer(5.0)));
}

#[test]
fn subscript_read_missing_dict_key_declines() {
    let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
    assert_eq!(subscript_read(&dict, &string("missing")), None);
}

#[test]
fn subscript_read_int_key_does_not_match_a_string_index() {
    // an Object receiver keyed numerically stays a dict read — a
    // known-Integer index matches only a numeric ObjectKey, never a
    // string-spelled one, and vice versa
    let dict = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
    assert_eq!(subscript_read(&dict, &string("15")), None);
    assert_eq!(subscript_read(&dict, &integer(15.0)), Some(integer(115.0)));
}

// --- unknown-length, known-element-set receivers (the `list[int]`/
// `set[int]`/`Sequence[int]` parameter seed's own star shape) ---

/// The star-of-a-set receiver `check.rs::seed_parameters` builds for
/// a `list[int]` parameter: `Kind::Set` over one bare
/// `Form::Star(element)`. Any known Integer index reads "some member
/// of element" — the star's own definition, no bounds check possible
/// since the length is unstated.
fn star_of(element: refined_sets::refinement_forms::RefinedSet) -> AbstractValue {
    known_set(
        refined_sets::refinement_forms::make_refined_set(vec![refined_sets::refinement_forms::star(element)]),
        None,
        TrustProved,
        SetKindTag::None,
    )
}

#[test]
fn subscript_read_of_a_star_shaped_set_answers_the_element_set_at_any_index() {
    let whole_ints = refined_sets::refinement_forms::make_refined_set(vec![
        refined_sets::refinement_forms::integer(),
        refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
    ]);
    let ages = star_of(whole_ints.clone());
    let element_at_zero = subscript_read(&ages, &integer(0.0)).expect("index 0 reads the star's element");
    assert_eq!(element_at_zero.kind, Kind::Set);
    assert_eq!(element_at_zero.set, whole_ints.clone());
    // the length is unstated — a large index reads the SAME element
    // set, never a bounds refusal the way an exact Kind::List would
    let element_at_large_index =
        subscript_read(&ages, &integer(9000.0)).expect("a star has no length to bound against");
    assert_eq!(element_at_large_index.set, whole_ints);
}

#[test]
fn subscript_read_of_a_star_shaped_set_declines_a_non_integer_index() {
    let whole_ints = refined_sets::refinement_forms::make_refined_set(vec![
        refined_sets::refinement_forms::integer(),
        refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
    ]);
    let ages = star_of(whole_ints);
    assert_eq!(subscript_read(&ages, &string("0")), None);
}

#[test]
fn subscript_read_of_a_bounded_scalar_set_is_not_read_as_a_star() {
    // an ordinary bound scalar range (not a star) must not fall into
    // the star reader — it declines the same as before this feature
    let bound = known_set(
        refined_sets::refinement_forms::make_refined_set(vec![refined_sets::refinement_forms::at_least(0.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    assert_eq!(subscript_read(&bound, &integer(0.0)), None);
}

// --- len() ---

#[test]
fn len_of_list() {
    let list = list_literal_value(&[integer(1.0), integer(2.0), integer(3.0)]);
    let got = len_result(&list).expect("len(list) must decide");
    assert_eq!(got.values, vec![3.0]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn len_of_dict() {
    let dict = dict_literal_value(
        &[Some(key("a")), Some(key("b"))],
        &[integer(1.0), integer(2.0)],
    );
    let got = len_result(&dict).expect("len(dict) must decide");
    assert_eq!(got.values, vec![2.0]);
}

#[test]
fn len_of_string_counts_code_points_not_bytes() {
    let got = len_result(&string("héllo")).expect("len(str) must decide");
    assert_eq!(got.values, vec![5.0]);
}

#[test]
fn len_of_unknown_declines() {
    assert_eq!(len_result(&unknown()), None);
}
