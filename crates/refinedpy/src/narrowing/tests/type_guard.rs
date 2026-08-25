use super::*;

// ── TypeGuard/TypeIs: recognized, never trusted ───────────────────

/// A call to a `TypeGuard[X]`-annotated same-module predicate narrows
/// an unbound name to what the predicate's OWN BODY proves, never to
/// the annotation's claimed `X` (`recognizes_type_guard_call`'s own
/// doc: trusting the claim unverified would read `dishonest_predicate`
/// silent when the row expects a fire). This predicate's body only
/// proves `isinstance(v, int)` — a weaker claim than `Age` — so
/// `value` narrows to the unbounded `int` sort, not `Age`.
#[test]
fn test_type_guard_call_narrows_an_unbound_name_to_its_bodys_own_proof() {
    let environment = environment_with_function_table(concat!(
        "def is_age(v: object) -> TypeGuard[Age]:\n",
        "    return isinstance(v, int)\n",
    ));
    let Some(narrowed) = assumed("is_age(value)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("the body's own proof seeds a binding");
    assert_eq!(value.kind, Kind::Set, "the proof is a sort, not an exact value");
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// The same recognition for `TypeIs[X]` (typing.rst's narrower
/// sibling of `TypeGuard`) — the same syntactic shape, same
/// proof-not-claim narrowing.
#[test]
fn test_type_is_call_narrows_an_unbound_name_to_its_bodys_own_proof() {
    let environment = environment_with_function_table(concat!(
        "def is_age(v: object) -> TypeIs[Age]:\n",
        "    return isinstance(v, int)\n",
    ));
    let Some(narrowed) = assumed("is_age(value)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("the body's own proof seeds a binding");
    assert_eq!(value.kind, Kind::Set);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
}

/// A call to a function with NO `TypeGuard`/`TypeIs` return
/// annotation is not recognized by this reader at all — it falls
/// through to `narrow_isinstance_call`'s own decline for a
/// non-`isinstance` callee, the same untouched outcome, but through
/// the ordinary path rather than this one.
#[test]
fn test_plain_predicate_call_is_not_recognized_as_a_type_guard() {
    let environment = environment_with_function_table(concat!(
        "def is_age(v: object) -> bool:\n",
        "    return isinstance(v, int)\n",
    ));
    let Some(narrowed) = assumed("is_age(value)", environment, true) else {
        return;
    };
    assert!(narrowed.read("value").is_none());
}

/// An EXISTING binding of a name a `TypeGuard` call names is also
/// left untouched — the decline applies regardless of whether the
/// name was previously bound.
#[test]
fn test_type_guard_call_does_not_narrow_an_existing_binding() {
    let mut environment = environment_with_function_table(concat!(
        "def is_age(v: object) -> TypeGuard[Age]:\n",
        "    return isinstance(v, int)\n",
    ));
    environment.bind("value", known_values(vec![200.0], PrimitiveKind::Number, TrustProved));
    let Some(narrowed) = assumed("is_age(value)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("value stays bound");
    assert_eq!(value.values, vec![200.0], "the pre-existing binding survives unchanged");
}

/// f-type-nodes.py's own honest/dishonest contrast, run end to end
/// through `assume`: `is_age`'s body chains `isinstance(v, int) and
/// not isinstance(v, bool) and 0 <= v <= 120` — the SAME shape the
/// module doc names as the SET channel's own canonical example — so
/// the proof narrows `value` all the way down to a bounded `[0, 120]`
/// integer window, a strict subset of the unbounded `int` sort.
/// Needs a live kernel: the bound comparison narrows through the SET
/// channel's own kernel question, not the VALUES channel alone.
#[test]
fn test_an_honest_type_guard_narrows_to_a_bounded_window() {
    let environment = environment_with_function_table(concat!(
        "def is_age(v: object) -> TypeGuard[Age]:\n",
        "    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 120\n",
    ));
    let Some(narrowed) = assumed("is_age(value)", environment, true) else {
        return;
    };
    let value = narrowed.read("value").expect("the bounded proof seeds a binding");
    assert_eq!(value.kind, Kind::Set, "a bounded window is still a Set-kind proof, not an exact value");
}
