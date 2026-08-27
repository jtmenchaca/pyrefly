// Literal round-trips (list/tuple/dict) and identity-keyed dict entries
// — an `object()` sentinel key and a class instance's own
// `instance_identity`.

use super::*;

// --- literal round-trips ---

#[test]
fn list_literal_round_trips_elements() {
    let built = list_literal_value(&[integer(1.0), integer(2.0)]);
    assert_eq!(built.kind, Kind::List);
    assert_eq!(built.items, vec![integer(1.0), integer(2.0)]);
}

#[test]
fn tuple_literal_round_trips_elements() {
    let built = tuple_literal_value(&[integer(1.0), string("a")]);
    assert_eq!(built.kind, Kind::List);
    assert_eq!(built.items, vec![integer(1.0), string("a")]);
}

#[test]
fn dict_literal_round_trips_string_keyed_entries() {
    let built = dict_literal_value(
        &[Some(key("a")), Some(key("b"))],
        &[integer(1.0), integer(2.0)],
    );
    assert_eq!(built.kind, Kind::Object);
    assert_eq!(subscript_read(&built, &string("a")), Some(integer(1.0)));
    assert_eq!(subscript_read(&built, &string("b")), Some(integer(2.0)));
}

#[test]
fn dict_literal_with_a_computed_key_answers_unknown() {
    let built = dict_literal_value(&[None, Some(key("b"))], &[integer(1.0), integer(2.0)]);
    assert_eq!(built.kind, Kind::Unknown);
}

#[test]
fn dict_literal_keeps_the_last_value_for_a_repeated_key() {
    let built = dict_literal_value(
        &[Some(key("a")), Some(key("a"))],
        &[integer(1.0), integer(2.0)],
    );
    assert_eq!(built.keys.len(), 1);
    assert_eq!(subscript_read(&built, &string("a")), Some(integer(2.0)));
}

#[test]
fn dict_literal_int_key_reads_by_int_subscript() {
    // {15: 115} — the a-statements.py dict_comprehension row's own
    // shape: a known Integer key builds a numeric ObjectKey, and a
    // matching Integer subscript reads it back.
    let built = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
    assert_eq!(built.keys.len(), 1);
    assert_eq!(built.keys[0].name, "15");
    assert!(built.keys[0].numeric);
    assert_eq!(subscript_read(&built, &integer(15.0)), Some(integer(115.0)));
}

#[test]
fn dict_literal_int_key_and_string_key_of_the_same_spelling_do_not_collide() {
    // {"15": 1, 15: 2} — CPython holds BOTH entries (1 == "15" is
    // False; only values that compare equal, like 1/1.0/True, share
    // one dict slot, stdtypes.rst's own Mapping Types note).
    let built = dict_literal_value(
        &[Some(key("15")), Some(DictKey::integer(15))],
        &[integer(1.0), integer(2.0)],
    );
    assert_eq!(built.keys.len(), 2);
    assert_eq!(subscript_read(&built, &string("15")), Some(integer(1.0)));
    assert_eq!(subscript_read(&built, &integer(15.0)), Some(integer(2.0)));
}

// --- identity-keyed dict entries (an object() sentinel key) ---

fn identity_sentinel(tag: &str) -> AbstractValue {
    let mut instance = refined_domain::abstract_value::opaque_value("a featureless object");
    instance.source = tag.to_owned();
    instance
}

#[test]
fn dict_key_spelling_names_each_key_sort_the_way_a_message_reads_it() {
    // the `KeyError: <spelling>` detail a provable-absence row writes:
    // a string quoted, a number bare, an identity key's own tag with
    // the reserved uncollidable prefix stripped
    assert_eq!(DictKey::string("missing").spelling(), "'missing'");
    assert_eq!(DictKey::integer(16).spelling(), "16");
    assert_eq!(DictKey::identity("object()").spelling(), "object()");
}

#[test]
fn dict_literal_identity_key_reads_back_by_the_same_sentinel() {
    let sentinel = identity_sentinel("object()");
    let built = dict_literal_value(&[Some(DictKey::identity("object()"))], &[integer(40.0)]);
    assert_eq!(subscript_read(&built, &sentinel), Some(integer(40.0)));
}

#[test]
fn dict_get_result_identity_key_present_answers_the_stored_value() {
    let sentinel = identity_sentinel("object()");
    let built = dict_literal_value(&[Some(DictKey::identity("object()"))], &[integer(40.0)]);
    assert_eq!(dict_get_result(&built, &sentinel, None), Some(integer(40.0)));
}

#[test]
fn dict_get_result_identity_key_absent_answers_none_value() {
    // a sentinel that was never inserted answers None, not the
    // stored entry for a DIFFERENT sentinel's own tag
    let stored = identity_sentinel("object()");
    let other = identity_sentinel("a different opaque value");
    let built = dict_literal_value(&[Some(DictKey::identity("object()"))], &[integer(40.0)]);
    assert_eq!(dict_get_result(&built, &stored, None), Some(integer(40.0)));
    assert_eq!(dict_get_result(&built, &other, None), Some(null_value()));
}

#[test]
fn dict_with_item_identity_key_round_trips_through_get() {
    let sentinel = identity_sentinel("object()");
    let empty = known_object(vec![], None, true, TrustProved, false);
    let written = dict_with_item(&empty, &sentinel, &integer(200.0)).expect("identity-keyed write must decide");
    assert_eq!(dict_get_result(&written, &sentinel, None), Some(integer(200.0)));
}

#[test]
fn known_dict_key_ignores_a_class_instances_source_tag_with_no_instance_identity() {
    // a constructed class instance with `source` set but no
    // `instance_identity` (a hand-built instance this test never ran
    // through `judge_construction`) is NOT an opaque value (no
    // kind_word) and carries no per-construction id — reading its
    // shared `source` as an identity tag would wrongly treat every
    // instance of the SAME class as one shared dict key, so
    // known_dict_key must decline here rather than build a
    // DictKey::identity from it.
    let mut class_instance = known_object(vec![], None, true, TrustProved, false);
    class_instance.source = "Holder".to_owned();
    assert_eq!(known_dict_key(&class_instance), None);
}

// --- identity-keyed dict entries (a class instance's own
// instance_identity, `instances::judge_construction`'s own tag) ---

/// Two class instances that share the SAME `source` (both `Holder`)
/// but carry DIFFERENT `instance_identity` ids — the shape
/// `judge_construction` builds for two separate `Holder()` calls.
fn class_instance(class_name: &str, identity: u32) -> AbstractValue {
    let mut instance = known_object(vec![], None, true, TrustProved, false);
    instance.source = class_name.to_owned();
    instance.instance_identity = Some(identity);
    instance
}

#[test]
fn known_dict_key_reads_a_class_instances_own_instance_identity() {
    let a = class_instance("Holder", 1);
    let b = class_instance("Holder", 2);
    assert_ne!(known_dict_key(&a), known_dict_key(&b), "two distinct constructions must not share a key");
}

#[test]
fn dict_get_result_finds_the_exact_instance_a_key_was_inserted_with() {
    // cache[key] = 40; cache.get(key) must read 40 back; a DIFFERENT
    // instance of the same class (missing_key) must miss — the
    // WeakKeyDictionary.get row this table exists to serve.
    let key = class_instance("Holder", 1);
    let missing_key = class_instance("Holder", 2);
    let dict_key = known_dict_key(&key).expect("a constructed instance is an identity key");
    let built = dict_literal_value(&[Some(dict_key)], &[integer(40.0)]);
    assert_eq!(dict_get_result(&built, &key, None), Some(integer(40.0)));
    assert_eq!(dict_get_result(&built, &missing_key, None), Some(null_value()));
}

#[test]
fn dict_with_item_class_instance_key_round_trips_through_get() {
    let key = class_instance("Holder", 7);
    let empty = known_object(vec![], None, true, TrustProved, false);
    let written = dict_with_item(&empty, &key, &integer(40.0)).expect("identity-keyed write must decide");
    assert_eq!(dict_get_result(&written, &key, None), Some(integer(40.0)));
}

// --- FLOAT keys: stdtypes.rst's own interchangeability rule ---

fn float_value(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

#[test]
fn a_negative_zero_and_a_positive_zero_are_one_key() {
    // "Values that compare equal ... can be used interchangeably to index
    // the same dictionary entry", and -0.0 == 0.0 is True in Python
    assert_eq!(known_dict_key(&float_value(-0.0)), known_dict_key(&float_value(0.0)));
}

#[test]
fn a_whole_float_key_indexes_the_same_entry_its_int_twin_does() {
    // the same clause names 1 and 1.0 as one key
    assert_eq!(known_dict_key(&float_value(1.0)), known_dict_key(&integer(1.0)));
}

#[test]
fn a_nan_key_is_no_value_key_at_all() {
    // float("nan") == float("nan") is False, so a NaN compares equal to
    // nothing and keys no entry by value — an identity hit is the
    // identity channel's business, not this one's
    assert_eq!(known_dict_key(&float_value(f64::NAN)), None);
}

#[test]
fn a_negative_zero_write_is_read_back_by_a_positive_zero() {
    // A8.xfer.identity's own `zero_is_one_key` row
    let empty = known_object(vec![], None, true, TrustProved, false);
    let written = dict_with_item(&empty, &float_value(-0.0), &integer(30.0)).expect("a float-keyed write must decide");
    assert_eq!(dict_get_result(&written, &float_value(0.0), None), Some(integer(30.0)));
}
