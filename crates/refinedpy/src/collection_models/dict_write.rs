//! The dict write channel: `dict[key] = value`, `del d[key]`, and the
//! list-slice-deletion statement `del lst[lower:]` — the written-through
//! receivers the walk's own mutation sink threads through.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::known_constructors::element_of_object_star;
use refined_domain::known_constructors::known_dict_star;
use refined_domain::known_constructors::known_object;
use refined_domain::lattice_operations::join_known;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;

use super::dict_key::{known_dict_key, DictKey};
use super::list_literal::list_literal_value;

/// `dict[key] = value` — the written-through dict, known shapes only:
/// a known `Kind::Object` receiver and a known String- or
/// Integer-sorted key (`known_dict_key`'s own doc). The new entry
/// overwrites a same-IDENTITY existing entry (matched by BOTH `name`
/// and `numeric`, an ordinary assignment, not the dict-DISPLAY's own
/// duplicate-literal-key rule, but the same last-value-wins effect);
/// an absent key appends a new entry in insertion order, matching
/// `dict.__setitem__`'s own behavior (library/stdtypes.rst, "Mapping
/// Types — dict": "`d[key] = value` — Set `d[key]` to *value*"). `None`
/// for any other receiver or an unsupported key sort — the write is
/// not modeled, so the caller must not assume the container is
/// unchanged.
pub fn dict_with_item(receiver: &AbstractValue, key: &AbstractValue, value: &AbstractValue) -> Option<AbstractValue> {
    // `d[key] = value` on an UNBOUNDED-KEY `dict[K, X]` receiver
    // (`Kind::ObjectStar` — `check.rs::seed_parameters`' `known_dict_star`
    // seed): the star carries one claim about every present key and no key
    // list of its own. `d[key] = value` "Set `d[key]` to *value*"
    // (library/stdtypes.rst, "Mapping Types — dict") changes exactly one
    // key, and this row records exactly that: the WRITTEN key gets its own
    // `ObjectKey` entry holding the exact value written, while `inner`
    // keeps stating what every OTHER present key holds, unchanged. The two
    // halves are read back by `dict_star_index_read`/`dict_star_get_result`,
    // which look in the recorded entries first and fall back to `inner`.
    //
    // Recording the key exactly, rather than joining the written value
    // into the star's element, is what keeps a read-back of the SAME key
    // precise: `d["a"] = 20` then `d["a"]` answers exactly `{20}` —
    // A8.xfer.set's own `replace_value_is_read` row — where the join
    // answered `Age ∪ {20}` and refused at an Age-declared sink even
    // though every run reads the value just written. The join was also
    // unsound in the other direction for the unwritten keys' own law,
    // which no write to one key can widen.
    //
    // The key must be one this domain can read at all (`known_dict_key` —
    // a string, an int, or an identity-comparable sentinel), but its SORT
    // does not gate the row: stdtypes.rst's Mapping Types section states
    // `d[key] = value` once for any hashable key. That is what lets
    // `d[object()] = 30` on a `dict[object, Age]` parameter write through
    // — A8.xfer.identity's own rows. A key this domain cannot read has no
    // entry to record, so the write declines rather than widen the star.
    if receiver.kind == Kind::ObjectStar {
        // An UNREAD key over a star receiver — the second and later
        // passes of A8.edge.process's own loop, once the first pass has
        // already widened the result dict. There is no spelling to record
        // an entry under, so the star's own element absorbs the written
        // value, the same join `dict_widened_at_unread_key` performs one
        // arm down. Every recorded entry survives: this write names no
        // key, so it cannot have overwritten one.
        let Some(key) = known_dict_key(key) else {
            return dict_widened_at_unread_key(receiver, value);
        };
        element_of_object_star(receiver)?;
        let mut written = receiver.clone();
        match written.keys.iter_mut().find(|entry| entry.name == key.name && entry.numeric == key.numeric) {
            Some(existing) => existing.value = value.clone(),
            None => written.keys.push(ObjectKey {
                name: key.name,
                numeric: key.numeric,
                value: value.clone(),
            }),
        }
        return Some(written);
    }
    if receiver.kind != Kind::Object {
        return None;
    }
    // `d[key] = value` at an UNREAD key — A8.edge.process's own
    // `result[k] = v`, where `k` is one piece of a line split whose
    // receiver is an unread `str`, so no exact spelling exists to record
    // an entry under. `known_dict_key` declines it, and declining the
    // whole write left the dict with no derived value at all.
    //
    // What the write DOES state is stdtypes.rst's own Mapping Types rule
    // for `d[key] = value` ("Set `d[key]` to *value*"), applied without a
    // key name: SOME key now holds `value`, and every other key holds
    // whatever it already held. A key list this domain can no longer
    // name is exactly `Kind::ObjectStar`'s own shape — one claim about
    // every present key, no key list — so the receiver widens into the
    // star whose element JOINS the written value with every value the
    // receiver already held. The join is sound in both directions: a
    // read of any key answers a set containing what that key really
    // holds, and no key is claimed present that was not.
    //
    // This is strictly weaker than the exact-key rows above and never
    // reached in their place: a key this domain CAN read still records
    // its own entry and reads back exactly, unchanged.
    let Some(key) = known_dict_key(key) else {
        return dict_widened_at_unread_key(receiver, value);
    };
    let mut entries = receiver.keys.clone();
    if let Some(existing) = entries.iter_mut().find(|entry| entry.name == key.name && entry.numeric == key.numeric) {
        existing.value = value.clone();
    } else {
        entries.push(ObjectKey {
            name: key.name,
            numeric: key.numeric,
            value: value.clone(),
        });
    }
    // The receiver's own `instance_identity` carries forward: a key
    // assignment MUTATES the dict in place (stdtypes.rst, "Mapping Types
    // — dict", `d[key] = value`: "Set `d[key]` to *value*" — it does not
    // build a new mapping), so it is the same object afterward and every
    // container holding a reference to it still does. That is what
    // `Environment::rebind_referents_of_item` relies on to find them.
    let mut written = known_object(entries, None, true, TrustProved, false);
    written.instance_identity = receiver.instance_identity;
    Some(written)
}

/// The receiver `d[<unread key>] = value` leaves behind: an
/// unbounded-key dict (`Kind::ObjectStar`) whose one element claim is
/// the JOIN of `value` with every value the receiver already stated —
/// each recorded entry's own value for a `Kind::Object` receiver, and
/// the star's existing element (plus its recorded entries' values) for a
/// `Kind::ObjectStar` one.
///
/// The join is what makes the answer sound with no key names left to
/// distinguish entries by: a later read of ANY key answers a set that
/// contains whatever that key really holds, because every value the dict
/// could hold at any key is in the join. The key SET is deliberately
/// dropped — `Kind::ObjectStar` states no key list, so nothing claims a
/// key is present that is not.
///
/// `None` when the join does not land on a shape a star can wrap
/// (`known_dict_star`'s own scalar-shaped gate — a `Kind::Set`,
/// `Kind::Values`, or a nested star, optionally under a maybe wrapper):
/// a dict of LISTS written at an unread key has no star element this
/// domain can state, and declining is the honest answer there.
fn dict_widened_at_unread_key(receiver: &AbstractValue, value: &AbstractValue) -> Option<AbstractValue> {
    let mut element = value.clone();
    if receiver.kind == Kind::ObjectStar {
        element = join_known(element, element_of_object_star(receiver)?);
    }
    for entry in &receiver.keys {
        element = join_known(element, entry.value.clone());
    }
    let grade = trust_level_of(receiver);
    let (star, built) = known_dict_star(element, grade);
    if !built {
        return None;
    }
    Some(star)
}

/// `del d[key]` — the written-through dict with `key`'s own entry
/// removed: a known `Kind::Object` receiver and a known String-sorted
/// key that IS present (library/simple_stmts.rst's own `del` entry:
/// "Deletion of a name removes the binding of that name... Deletion of
/// items... follows the semantics defined for `object.__delitem__()`" —
/// dict's `__delitem__` in turn is `d[key]`'s own removal counterpart,
/// stdtypes.rst's Mapping Types table). `None` for any other receiver
/// or a non-String key (the write is not modeled), AND for a key that
/// is ABSENT — CPython raises `KeyError` on `del` of a missing key
/// (the same raise `d[key]` itself raises, stdtypes.rst's `d[key]`
/// row), so an absent-key `del` is `provable_raise`'s own row to speak
/// (its existing `known_container_index_absent` check already reads
/// this exact container/key pair for the ordinary subscript-read raise)
/// rather than this function inventing a second decline message for
/// the identical fact.
pub fn dict_without_item(receiver: &AbstractValue, key: &AbstractValue) -> Option<AbstractValue> {
    // `del d[key]` on an UNBOUNDED-KEY `dict[K, X]` receiver
    // (`Kind::ObjectStar` — `check.rs::seed_parameters`' `known_dict_star`
    // seed): the star states which value every PRESENT key holds and
    // nothing about WHICH keys are present, and a delete only ever removes
    // a key — it never changes what a remaining key holds. So the star's
    // own claim survives the delete exactly, and the receiver carries
    // through unchanged. The key must still be one this domain can read
    // (`known_dict_key`), the same gate `dict_with_item`'s own star arm
    // keeps, but its sort does not gate the row for the same reason
    // stated there.
    //
    // Declining instead invalidated the receiver outright, leaving every
    // read after a `del` into a dict parameter with no derived value —
    // A8.xfer.delete's own rows read `d["a"]` after `del d["a"]` and after
    // a skipped `del d["z"]` through this arm.
    if receiver.kind == Kind::ObjectStar {
        let key = known_dict_key(key)?;
        element_of_object_star(receiver)?;
        // A key this receiver was WRITTEN at carries its own recorded
        // entry (`dict_with_item`'s own star arm); the delete removes that
        // key, so the entry goes with it. A GUARD entry for the same key
        // goes with it too: a membership test proved the key present at
        // the guard, and this statement removes exactly that entry, so
        // the fact no longer holds (`DictKey::guarded`'s own provenance
        // rule — a guard entry states "present at the guard", never
        // "present after every later statement"). Every other key's law
        // is untouched — `inner` carries through unchanged.
        let guarded = DictKey::guarded(&key);
        let mut written = receiver.clone();
        written.keys.retain(|entry| {
            !(entry.name == key.name && entry.numeric == key.numeric)
                && !(entry.name == guarded.name && entry.numeric == guarded.numeric)
        });
        return Some(written);
    }
    if receiver.kind != Kind::Object {
        return None;
    }
    let key = known_dict_key(key)?;
    if !receiver.keys.iter().any(|entry| entry.name == key.name && entry.numeric == key.numeric) {
        return None;
    }
    let entries: Vec<ObjectKey> = receiver
        .keys
        .iter()
        .filter(|entry| !(entry.name == key.name && entry.numeric == key.numeric))
        .cloned()
        .collect();
    Some(known_object(entries, None, true, TrustProved, false))
}

/// `del lst[lower:]` — a SLICE DELETION statement (simple_stmts.rst,
/// "The `del` statement": "Deletion of a target list recursively
/// deletes each target, from left to right... `del s[i:j]` is
/// equivalent to setting `s[i:j] = []`" over a Mutable Sequence
/// receiver). Modeled for the one shape the corpus needs a value for:
/// `lower` a known nonnegative Integer index, `upper` ABSENT (`lst[1:]`
/// — "delete everything from `lower` to the end"), no `step`. The same
/// clamp-not-raise bound Slicings states for an ordinary read applies
/// here too — an out-of-range `lower` just truncates to the whole list
/// (`lower >= len`) or leaves it unchanged (`lower <= 0`), never raises.
/// `None` for a non-`Kind::List` receiver, a negative `lower`, or any
/// other slice shape (`upper` present, a `step`, an unknown `lower`) —
/// this is the ONE contract this file states for `del`, mirroring
/// `mutated_receiver`'s own all-or-nothing decline discipline. Reached from
/// `check.rs::walk_del_subscript_target`'s own `Expr::Slice` arm, which
/// reads `lower` through `expressions::slice_bound_index` before calling
/// here.
pub fn sliced_delete_receiver(receiver: &AbstractValue, lower: i64) -> Option<AbstractValue> {
    if receiver.kind != Kind::List || lower < 0 {
        return None;
    }
    let length = receiver.items.len() as i64;
    let kept = lower.min(length) as usize;
    Some(list_literal_value(&receiver.items[..kept]))
}
