// dict.get (present/absent/default), and the written-through container
// pair dict_with_item/dict_without_item/list_with_item.

use super::*;

// --- dict.get present/absent/default ---

#[test]
fn dict_get_present_key_answers_its_value() {
    let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
    let got = dict_get_result(&dict, &string("k"), None).expect("get(present) must decide");
    assert_eq!(got, integer(5.0));
}

#[test]
fn dict_get_absent_key_with_no_default_answers_null() {
    let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
    let got = dict_get_result(&dict, &string("missing"), None).expect("get(absent) must decide");
    assert_eq!(got.kind, Kind::Null);
}

#[test]
fn dict_get_int_key_answers_its_value() {
    let dict = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
    let got = dict_get_result(&dict, &integer(15.0), None).expect("get(present int key) must decide");
    assert_eq!(got, integer(115.0));
}

#[test]
fn dict_get_absent_key_with_default_answers_the_default() {
    let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
    let fallback = integer(0.0);
    let got = dict_get_result(&dict, &string("missing"), Some(&fallback))
        .expect("get(absent, default) must decide");
    assert_eq!(got, fallback);
}

// --- dict_with_item / list_with_item (the written-through container) ---

#[test]
fn dict_with_item_overwrites_an_existing_key() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let written = dict_with_item(&dict, &string("a"), &integer(9.0)).expect("write must decide");
    assert_eq!(subscript_read(&written, &string("a")), Some(integer(9.0)));
}

#[test]
fn dict_with_item_appends_a_new_key() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let written = dict_with_item(&dict, &string("b"), &integer(2.0)).expect("write must decide");
    assert_eq!(written.keys.len(), 2);
    assert_eq!(subscript_read(&written, &string("b")), Some(integer(2.0)));
}

#[test]
fn dict_with_item_writes_an_int_key_without_colliding_a_string_key_of_the_same_spelling() {
    let dict = dict_literal_value(&[Some(key("15"))], &[integer(1.0)]);
    let written = dict_with_item(&dict, &integer(15.0), &integer(2.0)).expect("write must decide");
    assert_eq!(written.keys.len(), 2);
    assert_eq!(subscript_read(&written, &string("15")), Some(integer(1.0)));
    assert_eq!(subscript_read(&written, &integer(15.0)), Some(integer(2.0)));
}

#[test]
fn dict_without_item_removes_a_present_key() {
    let dict = dict_literal_value(
        &[Some(key("a")), Some(key("b"))],
        &[integer(1.0), integer(2.0)],
    );
    let written = dict_without_item(&dict, &string("a")).expect("del must decide");
    assert_eq!(written.keys.len(), 1);
    assert_eq!(subscript_read(&written, &string("b")), Some(integer(2.0)));
    assert_eq!(subscript_read(&written, &string("a")), None);
}

#[test]
fn dict_without_item_absent_key_declines() {
    // del on a missing key RAISES KeyError at runtime — this function
    // does not mutate on a raise, matching provable_raise's own
    // absent-key row for a plain subscript read
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    assert_eq!(dict_without_item(&dict, &string("missing")), None);
}

#[test]
fn dict_without_item_int_key_does_not_remove_a_string_key_of_the_same_spelling() {
    let dict = dict_literal_value(
        &[Some(key("15")), Some(DictKey::integer(15))],
        &[integer(1.0), integer(2.0)],
    );
    let written = dict_without_item(&dict, &integer(15.0)).expect("del must decide");
    assert_eq!(written.keys.len(), 1);
    assert_eq!(subscript_read(&written, &string("15")), Some(integer(1.0)));
    assert_eq!(subscript_read(&written, &integer(15.0)), None);
}

#[test]
fn dict_without_item_non_dict_receiver_declines() {
    let list = list_literal_value(&[integer(1.0)]);
    assert_eq!(dict_without_item(&list, &string("a")), None);
}

#[test]
fn list_with_item_writes_a_positive_index() {
    let list = list_literal_value(&[integer(1.0), integer(2.0)]);
    let written = list_with_item(&list, &integer(0.0), &integer(9.0)).expect("write must decide");
    assert_eq!(written.items, vec![integer(9.0), integer(2.0)]);
}

#[test]
fn list_with_item_out_of_range_declines() {
    let list = list_literal_value(&[integer(1.0)]);
    assert_eq!(list_with_item(&list, &integer(5.0), &integer(9.0)), None);
}

#[test]
fn list_with_item_carries_the_receivers_kind_word_forward() {
    // a bytes-like receiver's own species word (bytes_models::tagged)
    // must survive a write that mutates its contents — a SECOND write
    // to the same name still needs to read which write rule applies.
    let mut bytes_like = list_literal_value(&[integer(1.0), integer(2.0)]);
    bytes_like.kind_word = Some("a bytearray value");
    let written = list_with_item(&bytes_like, &integer(0.0), &integer(9.0)).expect("write must decide");
    assert_eq!(written.kind_word, Some("a bytearray value"));
}

#[test]
fn list_with_item_on_an_untagged_list_stays_untagged() {
    let list = list_literal_value(&[integer(1.0), integer(2.0)]);
    let written = list_with_item(&list, &integer(0.0), &integer(9.0)).expect("write must decide");
    assert_eq!(written.kind_word, None);
}
