use super::*;

// --- j-stdlib-surfaces.py: re family ---

/// `re.search("z", "banana")` — the literal pattern "z" never
/// occurs in "banana," so the exact answer is `None`.
#[test]
fn test_re_search_absent_literal_pattern_answers_none() {
    let Some(value) = eval("re.search(\"z\", \"banana\")") else { return };
    assert_eq!(value.kind, Kind::Null);
}

/// `re.search("a", "banana")` — the literal pattern IS present, so
/// the answer is the match-object sort (opaque).
#[test]
fn test_re_search_present_literal_pattern_answers_a_match_object() {
    let Some(value) = eval("re.search(\"a\", \"banana\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert!(value.kind_word.is_some());
}

/// `re.sub("a", "b", "aaaaaaaaaa")` replaces EVERY match — ten "a"s
/// become ten "b"s.
#[test]
fn test_re_sub_literal_pattern_replaces_every_match() {
    let Some(value) = eval("re.sub(\"a\", \"b\", \"aaaaaaaaaa\")") else { return };
    assert_eq!(value.values.len(), 10);
}

/// A pattern carrying a regex metacharacter declines — this file
/// only reduces METACHARACTER-FREE patterns to a substring test.
#[test]
fn test_re_search_with_a_metacharacter_pattern_declines() {
    let Some(value) = eval("re.search(\"a.b\", \"axb\")") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

/// `re.fullmatch(r"(\d+)-(\d+)", s)` read as a VALUE — the whole
/// call answers a Match-object carrying group 0 and both numbered
/// groups; `.group(1)` on it reads the first group's own grammar.
#[test]
fn test_re_fullmatch_value_then_group_reads_the_numbered_groups_grammar() {
    let Some(value) = eval("re.fullmatch(r\"(\\d+)-(\\d+)\", s).group(1)") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
}

/// `re.fullmatch(r"[A-Z]{2}", s).group(0)` — the whole-match group,
/// group 0, is always present even with no capturing groups.
#[test]
fn test_re_fullmatch_group_zero_is_always_present() {
    let Some(value) = eval("re.fullmatch(r\"[A-Z]{2}\", s).group(0)") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
}

/// `re.finditer(r"\d+", s)` read as a value directly (not through a
/// `for` loop, which `loops.rs` does not yet route to this arm) —
/// `.group(0)` on the single match value this call answers reads
/// the unanchored `\d+` grammar.
#[test]
fn test_re_finditer_value_then_group_zero_reads_the_unanchored_grammar() {
    let Some(value) = eval("re.finditer(r\"\\d+\", s).group(0)") else { return };
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
}

/// `re.fullmatch` with a metacharacter-free but non-literal pattern
/// (an unbound name) declines — the same honest refusal
/// `test_re_search_with_a_metacharacter_pattern_declines` shows for
/// `re.search`.
#[test]
fn test_re_fullmatch_with_a_non_literal_pattern_declines() {
    let Some(value) = eval("re.fullmatch(pattern, s)") else { return };
    assert_eq!(value.kind, Kind::Unknown);
}

// --- j-stdlib-surfaces.py: json family ---

#[test]
fn test_json_loads_parses_an_integer_literal() {
    let Some(value) = eval("json.loads(\"200\")") else { return };
    assert_eq!(value.values, vec![200.0]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// `json.loads(x)` over an operand this file holds no fact about (an
/// unbound name, so `exact_string_values` reads nothing) answers the
/// full JSON-union — every arm of `json_loads_value_space` — rather
/// than bare `unknown()` (ISSUES.md, "generic json.loads of an
/// opaque string answers bare unknown"). All seven shapes
/// library/json.rst's conversion table admits ride as arms: None,
/// bool, an unbounded string set, an unbounded int-sorted set, a
/// value-unknown float-sorted set, and the two opaque list/dict
/// arms.
#[test]
fn test_json_loads_of_an_opaque_operand_answers_the_full_json_union() {
    let Some(value) = eval("json.loads(x)") else { return };
    assert_eq!(value.kind, Kind::KindUnion);
    assert!(value.arms.iter().any(|arm| arm.kind == Kind::Null), "missing the None arm: {value:?}");
    assert!(
        value.arms.iter().any(|arm| arm.kind == Kind::Values && arm.kind_tag == Some(PrimitiveKind::Boolean)),
        "missing the bool arm: {value:?}"
    );
    // the str arm is untagged (kind_tag: None), matching the same
    // convention `__name__`'s own read builds (assignability.rs's
    // doc: an untagged Set whose own set is sequence-shaped reads
    // as string-sorted) — its own set is the full codepoint ground.
    assert!(
        value.arms.iter().any(|arm| arm.kind == Kind::Set && arm.kind_tag.is_none() && arm.set == strings()),
        "missing the str arm: {value:?}"
    );
    assert!(
        value.arms.iter().any(|arm| arm.kind == Kind::Set && arm.kind_tag == Some(PrimitiveKind::Integer)),
        "missing the int arm: {value:?}"
    );
    assert!(
        value.arms.iter().any(|arm| arm.kind == Kind::Set && arm.kind_tag == Some(PrimitiveKind::Float)),
        "missing the float arm: {value:?}"
    );
    assert!(
        value.arms.iter().any(|arm| arm.kind == Kind::Object && arm.kind_word == Some("a list")),
        "missing the list arm: {value:?}"
    );
    assert!(
        value.arms.iter().any(|arm| arm.kind == Kind::Object && arm.kind_word == Some("a dict")),
        "missing the dict arm: {value:?}"
    );
}

#[test]
fn test_json_dumps_serializes_a_known_dict() {
    let Some(value) = eval("json.dumps({\"age\": 40})") else { return };
    assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
    assert_eq!(exact_string_values(&value).and_then(code_points_to_string).as_deref(), Some(r#"{"age": 40}"#));
}

// --- j-stdlib-surfaces.py: exceptions ---

/// `str(Exception("failure"))` answers the message unchanged.
#[test]
fn test_str_of_exception_answers_the_message() {
    let Some(value) = eval("str(Exception(\"failure\"))") else { return };
    assert_eq!(exact_string_values(&value).and_then(code_points_to_string).as_deref(), Some("failure"));
}

/// `ExceptionGroup(...)` answers opaque — its message/wrapped
/// exceptions are never decomposed by this file.
#[test]
fn test_exception_group_construction_is_opaque() {
    let Some(value) = eval("ExceptionGroup(\"many\", [ValueError(\"a\")])") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert!(value.kind_word.is_some());
}
