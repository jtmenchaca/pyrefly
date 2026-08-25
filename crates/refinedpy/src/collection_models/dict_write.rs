//! The dict write channel: `dict[key] = value`, `del d[key]`, and the
//! list-slice-deletion statement `del lst[lower:]` — the written-through
//! receivers the walk's own mutation sink threads through.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;

use super::dict_key::known_dict_key;
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
    if receiver.kind != Kind::Object {
        return None;
    }
    let key = known_dict_key(key)?;
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
    Some(known_object(entries, None, true, TrustProved, false))
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
