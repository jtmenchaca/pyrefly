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

// --- the unbounded-key mapping (Kind::ObjectStar) write/read pair ---

/// A `dict[str, X]` parameter's own seed: the star wrapping `element`.
fn dict_star(element: AbstractValue) -> AbstractValue {
    let (star, built) = refined_domain::known_constructors::known_dict_star(element, TrustProved);
    assert!(built, "the star must build over a scalar-shaped element");
    star
}

/// The bare integer ray a `dict[str, int]` value slot seeds.
fn integer_ray() -> AbstractValue {
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    }
}

#[test]
fn dict_star_write_then_read_the_same_key_answers_exactly_what_was_written() {
    let star = dict_star(integer_ray());
    let written = dict_with_item(&star, &string("a"), &integer(20.0)).expect("a star write must decide");
    assert_eq!(written.kind, Kind::ObjectStar);
    assert_eq!(subscript_read(&written, &string("a")), Some(integer(20.0)));
}

#[test]
fn dict_star_write_then_read_another_key_answers_the_declarations_own_law() {
    let star = dict_star(integer_ray());
    let written = dict_with_item(&star, &string("a"), &integer(20.0)).expect("a star write must decide");
    assert_eq!(subscript_read(&written, &string("b")), Some(integer_ray()));
}

#[test]
fn dict_star_delete_drops_the_recorded_entry() {
    let star = dict_star(integer_ray());
    let written = dict_with_item(&star, &string("a"), &integer(20.0)).expect("a star write must decide");
    let deleted = dict_without_item(&written, &string("a")).expect("a star delete must decide");
    assert!(deleted.keys.is_empty());
    assert_eq!(subscript_read(&deleted, &string("a")), Some(integer_ray()));
}

#[test]
fn dict_star_len_counts_the_written_keys_as_a_floor() {
    let star = dict_star(integer_ray());
    let written = dict_with_item(&star, &string("a"), &integer(20.0)).expect("a star write must decide");
    let counted = len_result(&written).expect("len over a star must decide");
    assert_eq!(counted.kind_tag, Some(PrimitiveKind::Integer));
    let lower_bounds: Vec<f64> = counted
        .set
        .forms
        .iter()
        .filter(|form| form.form == refined_sets::refinement_forms::Form::AtLeast)
        .map(|form| form.a)
        .collect();
    assert_eq!(lower_bounds, vec![1.0]);
}

#[test]
fn dict_star_get_on_a_written_key_takes_the_present_branch_alone() {
    let star = dict_star(integer_ray());
    let written = dict_with_item(&star, &string("a"), &integer(20.0)).expect("a star write must decide");
    // no miss branch is folded in: the write proves the key present
    assert_eq!(dict_get_result(&written, &string("a"), None), Some(integer(20.0)));
}

/// The provenance twin of the row above: a GUARD-recorded entry
/// (`narrowing::compare::narrow_dict_membership_against_literal_key`'s
/// own `DictKey::guarded` wrapper — the shape a `"a" in d` membership
/// test records, never a write) must NOT take the written-key shortcut.
/// A guard proves presence AT THE GUARD, not that the key survives to
/// this read, so `.get` still folds the miss branch in — the honest
/// "element or the default" `dict_star_get_result` gives an unwritten
/// key. This is the unit-level pin for A8.guard.forget's own
/// `read_after_callee_write`: a receiver whose only recorded entry is a
/// stale guard fact must never be read as if it were a write's exact
/// value.
#[test]
fn dict_star_get_on_a_guard_recorded_key_still_folds_the_miss_branch() {
    let star = dict_star(integer_ray());
    let mut guarded = star.clone();
    guarded.keys.push(refined_domain::abstract_value::ObjectKey {
        name: DictKey::guarded(&key("a")).name,
        numeric: false,
        value: integer_ray(),
    });
    let answered = dict_get_result(&guarded, &string("a"), None).expect("a guarded key must still answer the join");
    assert_eq!(
        answered,
        join_known(integer_ray(), null_value()),
        "a guard entry answers element-or-default, never the exact recorded value alone"
    );
}

/// The write-provenance sibling of the same receiver shape: once the
/// SAME key is also WRITTEN (not merely guarded), the written-key
/// shortcut applies again — the guard entry and the write entry are
/// different `ObjectKey` spellings (`DictKey::guarded` vs a plain key),
/// so the write's own entry is what the shortcut finds.
#[test]
fn dict_star_get_on_a_written_key_takes_the_shortcut_even_beside_a_stale_guard_entry() {
    let star = dict_star(integer_ray());
    let mut written = dict_with_item(&star, &string("a"), &integer(20.0)).expect("a star write must decide");
    written.keys.push(refined_domain::abstract_value::ObjectKey {
        name: DictKey::guarded(&key("a")).name,
        numeric: false,
        value: integer_ray(),
    });
    assert_eq!(dict_get_result(&written, &string("a"), None), Some(integer(20.0)));
}

// --- writes at an UNREAD key (A8.edge.process's own `result[k] = v`) ---

/// A key with no exact spelling this domain can read (`known_dict_key`
/// declines it) has no entry to record — but the write still states that
/// SOME key now holds the written value. The receiver widens into the
/// unbounded-key shape rather than the write declining outright, which
/// is what left the built dict with no derived value at all.
#[test]
fn a_write_at_an_unread_key_widens_a_closed_dict_into_an_unbounded_key_dict() {
    let closed = dict_literal_value(&[Some(key("a"))], &[integer(10.0)]);
    let unread_key = integer_ray();
    let written = dict_with_item(&closed, &unread_key, &integer(20.0)).expect("an unread-key write must decide");
    assert_eq!(written.kind, Kind::ObjectStar, "no key list survives a write this domain cannot name a key for");
    let element = refined_domain::known_constructors::element_of_object_star(&written)
        .expect("the star wraps the joined value claim");
    // both the value already held (10) and the value just written (20)
    // are in the one claim — a read of any key must answer a set that
    // contains whatever that key really holds, so the element is the
    // JOIN of the two, never either one alone
    assert_eq!(
        element,
        join_known(integer(20.0), integer(10.0)),
        "the value already held joins with the value just written"
    );
}

/// The same write over a receiver that is ALREADY unbounded — the second
/// and later passes of a loop that widened on its first. The star's own
/// element absorbs the new value; nothing claims a key is present that
/// was not.
#[test]
fn a_write_at_an_unread_key_over_a_star_joins_into_the_stars_own_element() {
    let star = dict_star(integer_ray());
    let unread_key = integer_ray();
    let written = dict_with_item(&star, &unread_key, &integer(20.0)).expect("an unread-key star write must decide");
    assert_eq!(written.kind, Kind::ObjectStar);
    assert!(written.keys.is_empty(), "an unnamed key records no entry");
    refined_domain::known_constructors::element_of_object_star(&written).expect("the star still wraps one claim");
}

/// An exact key is unaffected: it still records its own entry and reads
/// back exactly. The widening arm is strictly weaker and never reached
/// in the exact row's place.
#[test]
fn an_exact_key_still_records_its_own_entry_beside_the_widening_arm() {
    let closed = dict_literal_value(&[Some(key("a"))], &[integer(10.0)]);
    let written = dict_with_item(&closed, &string("b"), &integer(20.0)).expect("an exact write must decide");
    assert_eq!(written.kind, Kind::Object, "an exact key keeps the closed key list");
    assert_eq!(subscript_read(&written, &string("b")), Some(integer(20.0)));
}
