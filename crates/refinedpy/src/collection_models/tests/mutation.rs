// mutated_receiver across every receiver shape: list, set (both the
// Kind::List shape and the repetition-shaped Kind::Set star seed),
// dict, and list.sort/list.reverse — plus list_bounded_range_read's
// own bounded-index subscript reads.

use super::*;

// --- mutated_receiver: list ---

#[test]
fn mutated_receiver_list_append() {
    let list = list_literal_value(&[integer(1.0)]);
    let (new_receiver, result) = mutated_receiver("append", &list, &[integer(2.0)]).expect("append must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0), integer(2.0)]);
    assert_eq!(result.kind, Kind::Null);
}

#[test]
fn mutated_receiver_list_extend() {
    let list = list_literal_value(&[integer(1.0)]);
    let other = list_literal_value(&[integer(2.0), integer(3.0)]);
    let (new_receiver, _) = mutated_receiver("extend", &list, &[other]).expect("extend must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0), integer(2.0), integer(3.0)]);
}

#[test]
fn mutated_receiver_list_insert() {
    let list = list_literal_value(&[integer(1.0), integer(3.0)]);
    let (new_receiver, _) =
        mutated_receiver("insert", &list, &[integer(1.0), integer(2.0)]).expect("insert must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0), integer(2.0), integer(3.0)]);
}

#[test]
fn mutated_receiver_list_pop_no_arg_removes_the_last_element() {
    let list = list_literal_value(&[integer(1.0), integer(2.0)]);
    let (new_receiver, popped) = mutated_receiver("pop", &list, &[]).expect("pop must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0)]);
    assert_eq!(popped, integer(2.0));
}

#[test]
fn mutated_receiver_list_pop_empty_receiver_declines() {
    let list = list_literal_value(&[]);
    assert_eq!(mutated_receiver("pop", &list, &[]), None);
}

#[test]
fn mutated_receiver_list_clear() {
    let list = list_literal_value(&[integer(1.0)]);
    let (new_receiver, _) = mutated_receiver("clear", &list, &[]).expect("clear must decide");
    assert_eq!(new_receiver.items.len(), 0);
}

// --- mutated_receiver: set (the same Kind::List shape as list) ---

#[test]
fn mutated_receiver_set_add_appends_a_new_element() {
    let set = list_literal_value(&[integer(1.0)]);
    let (new_receiver, _) = mutated_receiver("add", &set, &[integer(2.0)]).expect("add must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0), integer(2.0)]);
}

#[test]
fn mutated_receiver_set_add_a_duplicate_is_a_no_op() {
    let set = list_literal_value(&[integer(1.0)]);
    let (new_receiver, _) = mutated_receiver("add", &set, &[integer(1.0)]).expect("add must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0)]);
}

/// `bag.add(key)` on an EMPTY set with a non-`Kind::Values` element
/// (a class instance — weakref.WeakSet's own `.add()` shape,
/// j-stdlib-surfaces.py's `weak_set_contains` row) still succeeds:
/// an empty receiver trivially contains nothing, regardless of the
/// new element's own shape, so `element_contains`'s empty-receiver
/// short-circuit answers `false` without needing to compare the
/// opaque element's equality at all.
#[test]
fn mutated_receiver_set_add_an_opaque_element_to_an_empty_set_succeeds() {
    let empty_set = list_literal_value(&[]);
    let opaque_instance = refined_domain::abstract_value::opaque_value("a class instance");
    let (new_receiver, _) =
        mutated_receiver("add", &empty_set, &[opaque_instance]).expect("add of an opaque element to an empty set must decide");
    assert_eq!(new_receiver.items.len(), 1);
}

#[test]
fn mutated_receiver_set_discard_present_element_removes_it() {
    let set = list_literal_value(&[integer(1.0), integer(2.0)]);
    let (new_receiver, _) = mutated_receiver("discard", &set, &[integer(1.0)]).expect("discard must decide");
    assert_eq!(new_receiver.items, vec![integer(2.0)]);
}

#[test]
fn mutated_receiver_set_discard_absent_element_is_a_no_op() {
    let set = list_literal_value(&[integer(2.0)]);
    let (new_receiver, _) = mutated_receiver("discard", &set, &[integer(1.0)]).expect("discard must decide");
    assert_eq!(new_receiver.items, vec![integer(2.0)]);
}

#[test]
fn mutated_receiver_set_remove_present_element_removes_it() {
    let set = list_literal_value(&[integer(1.0), integer(2.0)]);
    let (new_receiver, _) = mutated_receiver("remove", &set, &[integer(1.0)]).expect("remove must decide");
    assert_eq!(new_receiver.items, vec![integer(2.0)]);
}

#[test]
fn mutated_receiver_set_remove_absent_element_declines() {
    // remove RAISES KeyError on a miss — this row does not mutate
    // on a raise, matching dict.pop's own no-default row
    let set = list_literal_value(&[integer(2.0)]);
    assert_eq!(mutated_receiver("remove", &set, &[integer(1.0)]), None);
}

#[test]
fn mutated_receiver_set_update_unions_in_place_skipping_duplicates() {
    let set = list_literal_value(&[integer(1.0)]);
    let other = list_literal_value(&[integer(1.0), integer(2.0)]);
    let (new_receiver, _) = mutated_receiver("update", &set, &[other]).expect("update must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0), integer(2.0)]);
}

// --- mutated_receiver: set (repetition-shaped Kind::Set receiver,
// the `list[int]`/`set[int]`/`Sequence[int]` parameter seed's own
// star shape — A10.seed.library / A15.xfer.inject) ---

/// A bounded integer window `[lo, hi]` — the whole-number element set
/// `star_of`'s own fixture uses, repeated through
/// `refined_sets::repetition_window_forms::repetition` rather than
/// `star`, so the window carries finite bounds a test can assert on.
fn bounded_ints(lo: i64, hi: Option<i64>) -> AbstractValue {
    let whole_ints = refined_sets::refinement_forms::make_refined_set(vec![
        refined_sets::refinement_forms::integer(),
        refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
    ]);
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            refined_sets::repetition_window_forms::repetition(whole_ints, lo, hi),
            None,
            TrustProved,
            SetKindTag::None,
        )
    }
}

#[test]
fn mutated_receiver_set_append_widens_the_window_by_one() {
    let set = bounded_ints(0, Some(3));
    let (new_receiver, result) = mutated_receiver("append", &set, &[integer(9.0)]).expect("append must decide");
    assert_eq!(new_receiver.kind, Kind::Set);
    let window = as_repetition(&new_receiver.set).expect("append must keep the repetition shape");
    assert_eq!(window.lo, 1);
    assert_eq!(window.hi, Some(4));
    assert_eq!(result.kind, Kind::Null);
}

#[test]
fn mutated_receiver_set_append_on_an_unbounded_window_stays_unbounded() {
    let set = bounded_ints(0, None);
    let (new_receiver, _) = mutated_receiver("append", &set, &[integer(9.0)]).expect("append must decide");
    let window = as_repetition(&new_receiver.set).expect("append must keep the repetition shape");
    assert_eq!(window.lo, 1);
    assert_eq!(window.hi, None);
}

#[test]
fn mutated_receiver_set_extend_adds_the_iterables_own_count_window() {
    let set = bounded_ints(0, Some(3));
    let other = list_literal_value(&[integer(1.0), integer(2.0)]);
    let (new_receiver, _) = mutated_receiver("extend", &set, &[other]).expect("extend must decide");
    let window = as_repetition(&new_receiver.set).expect("extend must keep the repetition shape");
    assert_eq!(window.lo, 2);
    assert_eq!(window.hi, Some(5));
}

#[test]
fn mutated_receiver_set_extend_with_an_unbounded_iterable_stays_unbounded() {
    let set = bounded_ints(0, Some(3));
    let other = bounded_ints(0, None);
    let (new_receiver, _) = mutated_receiver("extend", &set, &[other]).expect("extend must decide");
    let window = as_repetition(&new_receiver.set).expect("extend must keep the repetition shape");
    assert_eq!(window.lo, 0);
    assert_eq!(window.hi, None);
}

#[test]
fn mutated_receiver_set_extend_with_an_empty_list_is_a_no_op_on_the_window() {
    let set = bounded_ints(1, Some(3));
    let empty = list_literal_value(&[]);
    let (new_receiver, _) = mutated_receiver("extend", &set, &[empty]).expect("extend must decide");
    let window = as_repetition(&new_receiver.set).expect("extend must keep the repetition shape");
    assert_eq!(window.lo, 1);
    assert_eq!(window.hi, Some(3));
}

#[test]
fn mutated_receiver_set_other_methods_decline() {
    let set = bounded_ints(0, Some(3));
    assert_eq!(mutated_receiver("pop", &set, &[]), None);
    assert_eq!(mutated_receiver("clear", &set, &[]), None);
    assert_eq!(mutated_receiver("add", &set, &[integer(1.0)]), None);
}

// --- mutated_receiver: dict ---

#[test]
fn mutated_receiver_dict_update_merges_and_overwrites() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let other = dict_literal_value(
        &[Some(key("a")), Some(key("b"))],
        &[integer(9.0), integer(2.0)],
    );
    let (new_receiver, _) = mutated_receiver("update", &dict, &[other]).expect("update must decide");
    assert_eq!(subscript_read(&new_receiver, &string("a")), Some(integer(9.0)));
    assert_eq!(subscript_read(&new_receiver, &string("b")), Some(integer(2.0)));
}

#[test]
fn mutated_receiver_dict_clear() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let (new_receiver, _) = mutated_receiver("clear", &dict, &[]).expect("clear must decide");
    assert_eq!(new_receiver.keys.len(), 0);
}

#[test]
fn mutated_receiver_dict_setdefault_present_key_leaves_the_dict_unchanged() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let (new_receiver, result) =
        mutated_receiver("setdefault", &dict, &[string("a"), integer(0.0)]).expect("setdefault must decide");
    assert_eq!(new_receiver.keys.len(), 1);
    assert_eq!(result, integer(1.0));
}

#[test]
fn mutated_receiver_dict_setdefault_absent_key_extends_and_answers_the_default() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let (new_receiver, result) =
        mutated_receiver("setdefault", &dict, &[string("b"), integer(0.0)]).expect("setdefault must decide");
    assert_eq!(new_receiver.keys.len(), 2);
    assert_eq!(result, integer(0.0));
}

#[test]
fn mutated_receiver_dict_pop_present_key_removes_it() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    let (new_receiver, popped) = mutated_receiver("pop", &dict, &[string("a")]).expect("pop must decide");
    assert_eq!(new_receiver.keys.len(), 0);
    assert_eq!(popped, integer(1.0));
}

#[test]
fn mutated_receiver_dict_pop_absent_key_with_no_default_declines() {
    let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
    // an absent key with no default RAISES KeyError at runtime — this
    // function does not mutate on a raise, matching set.remove's row
    assert_eq!(mutated_receiver("pop", &dict, &[string("missing")]), None);
}

#[test]
fn mutated_receiver_dict_popitem_removes_the_last_inserted_entry() {
    let dict = dict_literal_value(
        &[Some(key("a")), Some(key("b"))],
        &[integer(1.0), integer(2.0)],
    );
    let (new_receiver, pair) = mutated_receiver("popitem", &dict, &[]).expect("popitem must decide");
    assert_eq!(new_receiver.keys.len(), 1);
    assert_eq!(pair.items, vec![string("b"), integer(2.0)]);
}

#[test]
fn mutated_receiver_dict_popitem_int_key_answers_an_int_pair() {
    let dict = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
    let (new_receiver, pair) = mutated_receiver("popitem", &dict, &[]).expect("popitem must decide");
    assert_eq!(new_receiver.keys.len(), 0);
    assert_eq!(pair.items, vec![integer(15.0), integer(115.0)]);
}

#[test]
fn mutated_receiver_dict_setdefault_int_key_does_not_match_a_string_key_of_the_same_spelling() {
    let dict = dict_literal_value(&[Some(key("15"))], &[integer(1.0)]);
    let (new_receiver, result) = mutated_receiver("setdefault", &dict, &[integer(15.0), integer(0.0)])
        .expect("setdefault must decide");
    // "15" (string) is present, but the call's key is the INT 15 — a
    // different entry, so setdefault inserts a second one and answers
    // the default, never the string entry's value
    assert_eq!(new_receiver.keys.len(), 2);
    assert_eq!(result, integer(0.0));
}

#[test]
fn mutated_receiver_unmodeled_method_declines() {
    let list = list_literal_value(&[integer(1.0)]);
    assert_eq!(mutated_receiver("count", &list, &[integer(1.0)]), None);
}

// --- mutated_receiver: list.sort / list.reverse ---

#[test]
fn mutated_receiver_list_sort_ascending() {
    let list = list_literal_value(&[integer(3.0), integer(1.0), integer(2.0)]);
    let (new_receiver, result) = mutated_receiver("sort", &list, &[]).expect("sort must decide");
    assert_eq!(new_receiver.items, vec![integer(1.0), integer(2.0), integer(3.0)]);
    assert_eq!(result.kind, Kind::Null);
}

#[test]
fn mutated_receiver_list_sort_non_numeric_element_declines() {
    let list = list_literal_value(&[string("b"), string("a")]);
    assert_eq!(mutated_receiver("sort", &list, &[]), None);
}

#[test]
fn mutated_receiver_list_reverse_reorders_in_place() {
    let list = list_literal_value(&[integer(1.0), integer(2.0), integer(3.0)]);
    let (new_receiver, result) = mutated_receiver("reverse", &list, &[]).expect("reverse must decide");
    assert_eq!(new_receiver.items, vec![integer(3.0), integer(2.0), integer(1.0)]);
    assert_eq!(result.kind, Kind::Null);
}

// --- list_bounded_range_read / integer_range_bounds ---

/// A bounded Integer-sorted index (`ge=0, le=2` — the seeded shape
/// `["ok", "warn", "error"][code]` reads) into a three-element list
/// of exact strings: every position is in range, so the read joins
/// all three — `["ok", "warn", "error"][code]`'s own shape.
fn bounded_index(lo: f64, hi: f64) -> AbstractValue {
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![at_least(lo), at_most(hi)]), None, TrustProved, SetKindTag::None)
    }
}

#[test]
fn subscript_read_bounded_index_into_full_length_list_joins_every_position() {
    let list = list_literal_value(&[string("ok"), string("warn"), string("error")]);
    let index = bounded_index(0.0, 2.0);
    let got = subscript_read(&list, &index).expect("every index in [0, 2] is in range");
    let want = join_known(join_known(string("ok"), string("warn")), string("error"));
    assert_eq!(got, want);
}

/// A bounded index narrower than the full list still joins only the
/// positions the range actually admits.
#[test]
fn subscript_read_bounded_index_into_a_sub_range_joins_only_those_positions() {
    let list = list_literal_value(&[string("ok"), string("warn"), string("error")]);
    let index = bounded_index(0.0, 1.0);
    let got = subscript_read(&list, &index).expect("[0, 1] is in range");
    let want = join_known(string("ok"), string("warn"));
    assert_eq!(got, want);
}

/// A bounded index whose ceiling reaches past the list's own length
/// declines rather than joining only the in-range prefix — a partial
/// read would misreport what the OUT-of-range positions could hold.
#[test]
fn subscript_read_bounded_index_past_list_length_declines() {
    let list = list_literal_value(&[string("ok"), string("warn"), string("error")]);
    let index = bounded_index(0.0, 5.0);
    assert_eq!(subscript_read(&list, &index), None);
}

/// An UNBOUNDED index (no ceiling at all — `integer_range_bounds`
/// answers `None` for a set with no `AtMost`/`Below` form) declines:
/// there is no enumerable window to join over.
#[test]
fn subscript_read_unbounded_index_declines() {
    let list = list_literal_value(&[string("ok"), string("warn"), string("error")]);
    let index = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    };
    assert_eq!(subscript_read(&list, &index), None);
}

/// A NEGATIVE-lo range declines — this reader models only the
/// nonnegative window (per its own doc), never CPython's per-index
/// negative adjustment applied across a mixed-sign range.
#[test]
fn subscript_read_negative_lo_index_declines() {
    let list = list_literal_value(&[string("ok"), string("warn"), string("error")]);
    let index = bounded_index(-1.0, 1.0);
    assert_eq!(subscript_read(&list, &index), None);
}

/// A plain EXACT index still takes the exact-value row (`Kind::
/// Values`, never reaching `list_bounded_range_read` at all) — pins
/// that the new bounded-range fallback never displaces the existing
/// exact read.
#[test]
fn subscript_read_exact_index_still_reads_one_position() {
    let list = list_literal_value(&[string("ok"), string("warn"), string("error")]);
    assert_eq!(subscript_read(&list, &integer(1.0)), Some(string("warn")));
}
