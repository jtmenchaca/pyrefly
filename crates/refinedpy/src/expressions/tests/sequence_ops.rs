use super::*;

// --- set display and set operators/methods ---

#[test]
fn test_set_display_builds_the_shared_list_shape() {
    let Some(value) = eval("{1, 2, 3}") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items.len(), 3);
}

#[test]
fn test_set_union_operator_and_method_agree() {
    let Some(operator_result) = eval("{1, 2} | {2, 3}") else { return };
    assert_eq!(operator_result.items.len(), 3);
    let Some(method_result) = eval("{1, 2}.union({2, 3})") else { return };
    assert_eq!(method_result.items.len(), 3);
}

#[test]
fn test_set_intersection_operator() {
    let Some(value) = eval("{1, 2, 3} & {2, 3, 4}") else { return };
    assert_eq!(value.items.len(), 2);
}

#[test]
fn test_set_difference_operator() {
    let Some(value) = eval("{1, 2, 3} - {2}") else { return };
    assert_eq!(value.items.len(), 2);
}

#[test]
fn test_set_symmetric_difference_operator() {
    let Some(value) = eval("{1, 2} ^ {2, 3}") else { return };
    assert_eq!(value.items.len(), 2);
}

#[test]
fn test_set_issubset_true() {
    let Some(value) = eval("{1}.issubset({1, 2})") else { return };
    assert_eq!(value.values, vec![1.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
}

#[test]
fn test_set_issubset_false() {
    let Some(value) = eval("{1, 9}.issubset({1, 2})") else { return };
    assert_eq!(value.values, vec![0.0]);
}

#[test]
fn test_set_issuperset() {
    let Some(value) = eval("{1, 2}.issuperset({1})") else { return };
    assert_eq!(value.values, vec![1.0]);
}

#[test]
fn test_in_over_set_display() {
    let Some(present) = eval("2 in {1, 2, 3}") else { return };
    assert_eq!(present.values, vec![1.0]);
}

// --- dict view methods ---

#[test]
fn test_dict_keys_view() {
    let Some(value) = eval("list({\"a\": 1, \"b\": 2}.keys())") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items.len(), 2);
}

#[test]
fn test_dict_values_view() {
    let Some(value) = eval("list({\"a\": 1, \"b\": 2}.values())[0]") else { return };
    assert_eq!(value.values, vec![1.0]);
}

#[test]
fn test_dict_items_view() {
    let Some(value) = eval("list({\"a\": 1}.items())[0]") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items.len(), 2);
    assert_eq!(value.items[1].values, vec![1.0]);
}

// --- os / time / unicodedata / base64 families ---

/// `os.open(path, os.O_RDONLY)` answers the nonnegative Integer
/// ground — a fresh file descriptor carries no further identity
/// claim (A15.xfer.handle's own row).
#[test]
fn test_os_open_answers_a_nonnegative_integer_ground() {
    let Some(value) = eval("os.open(path, 0)") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `os.close(fd)` always answers `None`.
#[test]
fn test_os_close_answers_none() {
    let Some(value) = eval("os.close(fd)") else { return };
    assert_eq!(value.kind, Kind::Null);
}

/// `time.time()` answers the nonnegative Float ground.
#[test]
fn test_time_time_answers_a_nonnegative_float_ground() {
    let Some(value) = eval("time.time()") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

/// `unicodedata.normalize("NFC", s)` answers the whole-strings
/// ground — A3.xfer.normalize's own claim.
#[test]
fn test_unicodedata_normalize_answers_the_whole_strings_ground() {
    let Some(value) = eval("unicodedata.normalize(\"NFC\", s)") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
}

/// `base64.b64encode("ab".encode()).decode()` — A3.xfer.base64's own
/// row end to end: `.encode()` answers an opaque bytes value,
/// `base64.b64encode` tags it, `.decode()` reads the base64-alphabet
/// string grammar off that tag. An exact-string literal receiver
/// (rather than an unbound name) so `.encode()` reaches the exact-
/// receiver row (`string_models::string_method_result`) under this
/// test's own empty environment, which cannot bind a free name.
#[test]
fn test_base64_b64encode_then_decode_answers_the_base64_alphabet_grammar() {
    let Some(value) = eval("base64.b64encode(\"ab\".encode()).decode()") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
}

// --- j-stdlib-surfaces.py: dict/misc ---

/// `types.MappingProxyType(d)["age"]` reads through to the wrapped
/// dict's own value.
#[test]
fn test_mapping_proxy_type_reads_through_to_the_wrapped_dict() {
    let Some(value) = eval("types.MappingProxyType({\"age\": 40})[\"age\"]") else { return };
    assert_eq!(value.values, vec![40.0]);
}

/// `xs.sort()` used directly as a value expression — the RETURN
/// VALUE is always `None`, a sort mismatch against a refined Age.
#[test]
fn test_list_sort_as_a_value_expression_answers_none() {
    let Some(value) = eval("[41, 40].sort()") else { return };
    assert_eq!(value.kind, Kind::Null);
}

/// `list(map(lambda age: age + 1, [39]))[0]` — the materialized map.
#[test]
fn test_map_materialized_via_list_answers_the_mapped_elements() {
    let Some(value) = eval("list(map(lambda age: age + 1, [39]))") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items, vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)]);
}

/// `list(filter(lambda age: age > 100, [40, 200]))[0]` — the
/// materialized filter, keeping only the surviving element.
#[test]
fn test_filter_materialized_via_list_answers_the_kept_elements() {
    let Some(value) = eval("list(filter(lambda age: age > 100, [40, 200]))") else { return };
    assert_eq!(value.kind, Kind::List);
    assert_eq!(value.items, vec![known_values(vec![200.0], PrimitiveKind::Integer, TrustProved)]);
}

// --- j-stdlib-surfaces.py: str ---

/// `long.find("%")` feeding `long[:long_at]` — the fixed `find`
/// Integer-sort bug this wave closes (`string_models.rs`'s own
/// `find` row): a `Number`-tagged result used to decline the slice
/// bound outright.
#[test]
fn test_find_result_feeds_a_slice_bound() {
    let Some(value) = eval("\"123456789%\"[:\"123456789%\".find(\"%\")]") else { return };
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
    assert_eq!(exact_string_values(&value).and_then(code_points_to_string).as_deref(), Some("123456789"));
}

/// `key in bag` — a known List container whose elements are opaque
/// class instances (weakref.WeakSet's own `.add(key)` shape,
/// j-stdlib-surfaces.py's `weak_set_contains` row): element equality
/// cannot be decided, but the `in` expression's own SORT is still
/// provably `bool` — answered opaque rather than fully unknown.
#[test]
fn test_in_operator_over_opaque_elements_answers_an_opaque_boolean() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("key in bag").expect("test source must parse");
    let mut environment = empty_environment();
    environment.bind("key", opaque_value("a class instance"));
    environment.bind("bag", collection_models::list_literal_value(&[opaque_value("a class instance")]));
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert!(value.kind_word.is_some());
}

// --- p-typed-array.py: bytes/bytearray/memoryview/array.array construction ---

/// `bytes([10, 20, 30])[2]` — p-typed-array.py's own `bytes_from_
/// iterable` row: the constructor call answers the known list, and
/// element 2 reads through unchanged.
#[test]
fn test_bytes_constructor_from_a_known_list_reads_the_exact_element() {
    let Some(value) = eval("bytes([10, 20, 30])[2]") else { return };
    assert_eq!(value.values, vec![30.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `bytearray(4)[0]` — p-typed-array.py's own `bytearray_from_
/// length` row: a length-only construction zero-fills every slot.
#[test]
fn test_bytearray_constructor_from_a_length_zero_fills() {
    let Some(value) = eval("bytearray(4)[0]") else { return };
    assert_eq!(value.values, vec![0.0]);
}

/// `bytearray(b"\x0a\x14")[1]` — a bytes-literal argument to
/// `bytearray(...)` copies through the same known-list-of-known-
/// Integers shape a `bytes([...])` display builds.
#[test]
fn test_bytearray_constructor_from_a_bytes_literal_reads_the_exact_element() {
    let Some(value) = eval("bytearray(b\"\\x0a\\x14\")[1]") else { return };
    assert_eq!(value.values, vec![20.0]);
}

/// `memoryview(bytearray(b"..."))[3]` — p-typed-array.py's own
/// `memoryview_over_bytearray_reads` row: a view shares the SAME
/// element sequence as the underlying bytearray.
#[test]
fn test_memoryview_constructor_reads_through_the_shared_buffer() {
    let Some(value) = eval("memoryview(bytearray(b\"\\x00\\x01\\x02\\x03\"))[3]") else { return };
    assert_eq!(value.values, vec![3.0]);
}

/// `array.array("d", [10.0, 20.0, 30.0])[2]` — p-typed-array.py's
/// own `array_double_from_iterable` row: every element reads as a
/// FLOAT, never an int, whatever numeric literal built it.
#[test]
fn test_array_double_constructor_reads_a_float_tagged_element() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("import array\n").expect("test module parses").into_syntax();
    let table = std::sync::Arc::new(crate::function_table::function_table(&module));
    let mut environment = empty_environment();
    environment.set_functions(table);
    let parsed = parse_expression("array.array(\"d\", [10.0, 20.0, 30.0])[2]").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![30.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

/// `len(bytearray(10))` — the constructed value's own element count
/// composes through the ordinary `len()` dispatch once the
/// constructor answers a known `Kind::List`, with no bytes-specific
/// `len()` row needed (`collection_models::len_result`'s own generic
/// `Kind::List` row already covers it).
#[test]
fn test_len_of_a_bytearray_constructor_composes() {
    let Some(value) = eval("len(bytearray(10))") else { return };
    assert_eq!(value.values, vec![10.0]);
}

/// `bytearray(4)`/`bytearray(b"...")`/`bytes([...])`/
/// `memoryview(bytearray(...))` each carry their own species word
/// (`bytes_models::tagged`'s own doc) — `check.rs`'s write sink reads
/// this to decide which of the three write rules applies. A plain
/// list literal carries none of these words.
#[test]
fn test_bytearray_from_length_is_tagged_bytearray() {
    let Some(value) = eval("bytearray(4)") else { return };
    assert_eq!(value.kind_word, Some(bytes_models::BYTEARRAY_WORD));
}

#[test]
fn test_bytearray_from_a_bytes_literal_is_tagged_bytearray() {
    let Some(value) = eval("bytearray(b\"\\x0a\\x14\")") else { return };
    assert_eq!(value.kind_word, Some(bytes_models::BYTEARRAY_WORD));
}

#[test]
fn test_bytes_constructor_is_tagged_bytes() {
    let Some(value) = eval("bytes([10, 20, 30])") else { return };
    assert_eq!(value.kind_word, Some(bytes_models::BYTES_WORD));
}

#[test]
fn test_memoryview_over_bytearray_is_tagged_memoryview_not_bytearray() {
    // the view's OWN word must win — a write through the view raises
    // the memoryview-specific wording, not bytearray's, even though
    // the wrapped argument was itself tagged bytearray.
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("memoryview(bytearray(2))").expect("test source must parse");
    let environment = empty_environment();
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind_word, Some(bytes_models::MEMORYVIEW_WORD));
}

#[test]
fn test_plain_list_literal_carries_no_bytes_species_word() {
    let Some(value) = eval("[10, 20, 30]") else { return };
    assert_eq!(value.kind_word, None);
}

// --- h/c-file: computed dict key evaluating to a known string ---

/// h-object-literal-members.py's own `computed_key_other_expression`
/// / c-reads-and-values.py's own `read_type_member_computed_name`:
/// `key = "age"` then `{key: 200}` — a COMPUTED key (a bare Name,
/// never a string LITERAL) that reduces to a known exact string now
/// has a slot, the same `DictKey::string` entry a literal `{"age":
/// 200}` would build.
#[test]
fn test_dict_literal_with_a_computed_string_key_builds_and_reads_back() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("{key: 200}[key]").expect("test source must parse");
    let mut environment = empty_environment();
    environment.bind("key", string_models::string_literal_value("age"));
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![200.0]);
}

/// The SAME computed-key shape through a ternary — c-reads-and-
/// values.py's `read_computed_other_key`'s own `"age" if flag else
/// "years"` construction (proven here directly against a bound
/// String value, the ternary's own settled answer).
#[test]
fn test_dict_literal_with_a_ternary_computed_string_key_builds() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("{(\"age\" if flag else \"years\"): 40}[\"age\"]").expect("test source must parse");
    let mut environment = empty_environment();
    environment.bind("flag", known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved));
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![40.0]);
}

// --- string_set_concatenation / string_shaped_set ---

/// A length-windowed string parameter (`seed`, `Repeat(codepoints,
/// 1, 8)` — the shape `check.rs::seed_parameters` seeds for
/// `Annotated[str, Field(min_length=1, max_length=8)]`) concatenated
/// with a literal: `Add` must compose a `Concatenation` set rather
/// than falling through to `unknown()`, since neither operand is an
/// exact string (the literal side is exact; the parameter side is
/// not, which is what used to make `exact_string_values` refuse the
/// whole row).
#[test]
fn test_add_concatenates_a_string_window_with_a_literal() {
    let seed = AbstractValue {
        kind_tag: None,
        ..known_set(
            make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, Some(8))]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let literal = string_models::string_literal_value("xxxxxxxx");
    let result = sequence_binop_value(Operator::Add, &seed, &literal);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.set_kind_tag, SetKindTag::None);
    assert!(
        assignability::states_sequence(&result.set),
        "the concatenation must itself carry a sequence form: {:?}",
        result.set
    );
}

/// Two known EXACT strings still take the exact-value row above
/// `string_set_concatenation`'s own fallback (`sequence_binop_value`'s
/// first check) — this pins that the new fallback never fires for
/// the case the exact row already answers, so the two rows do not
/// double-handle the same input.
#[test]
fn test_add_two_exact_strings_stays_exact() {
    let a = string_models::string_literal_value("ab");
    let b = string_models::string_literal_value("c");
    let result = sequence_binop_value(Operator::Add, &a, &b);
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::String));
}

/// A NUMERIC set (never string-shaped) plus a string literal must
/// stay `unknown()` — `string_shaped_set` refuses the numeric side,
/// so the concatenation row never fires for a cross-sort operand
/// pair.
#[test]
fn test_add_numeric_set_and_string_literal_stays_unknown() {
    let number_set = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let literal = string_models::string_literal_value("x");
    let result = sequence_binop_value(Operator::Add, &number_set, &literal);
    assert_eq!(result.kind, Kind::Unknown);
}
