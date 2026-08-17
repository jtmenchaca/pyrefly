/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Container VALUE states: `list`/`tuple`/`dict` literals, subscript
//! reads (`s[i]`, `d[key]`), `len()`, and `dict.get`. Mutating methods
//! (`append`, `pop`, `add`, `update`, `clear`) decline — write effects
//! belong to the walk's World, not this pure-value layer (see
//! `mutating_method_result`'s own doc).
//!
//! ## How the domain carries a container
//!
//! `refined_domain::abstract_value::AbstractValue` has no dedicated
//! tuple variant, and Python's `list`/`tuple` both map to
//! `Kind::List` (`known_constructors::known_list`, "a nested exact
//! sequence") — the same "exact positional slots" shape, indexed the
//! same way, so this file's `tuple_literal_value` is `list_literal_value`
//! under a different name (the TS twin has no tuple either: JS has no
//! tuple type, so `known_constructors.rs` never split one out).
//!
//! `dict` maps to `Kind::Object` (`known_constructors::known_object`,
//! "rooted-keys record") — an ordered `Vec<ObjectKey>` of
//! `{name: String, value: AbstractValue}` pairs, never a JS-style
//! prototype-bearing map. This is a deliberate choice over
//! `Kind::Collection`/`Flavor::Map` (`abstract_value.rs`): the
//! `Collection`/`Flavor` pair is the TS twin's carry-over for a JS
//! `Map`/`Set` INSTANCE built via `new Map()` — the AGENT-BRIEF's
//! `AbstractValue` fields doc calls it "a built Map or Set" — not for
//! a `{...}` object literal read positionally by name, which is what a
//! Python `dict` LITERAL is. `known_object`'s ordered-`Vec` shape
//! already matches a dict literal exactly, and `pyrefly`'s translated
//! domain has no caller of either constructor yet, so this file is the
//! first to decide the mapping. A `dict` built by a non-literal path
//! (`dict(...)`, a comprehension) is out of this file's scope — only
//! `dict_literal_value` (a literal `{...}` display) is modeled.
//!
//! String-keyed entries only: a Python dict key that is not a string
//! literal (an int key, a computed key, a tuple key) has no slot in
//! `ObjectKey.name: String` to occupy. `dict_literal_value` takes
//! `keys: &[Option<String>]` — `None` at a position means "this key
//! is not a string literal" — and that entire literal answers
//! `unknown()` rather than silently dropping the non-string entry
//! (dropping would misreport the dict's key set to every later read).
//!
//! `len()` is modeled for known lists/tuples/dicts (their slot/key
//! count) and exact strings (`values.len()`, one code point per
//! `f64` — `string_models.rs`'s documented representation, cited
//! there against library/stdtypes.html's Text Sequence Type section:
//! "Strings are immutable sequences of Unicode code points").
//!
//! ## Coverage cited against the vendored CPython 3.12 docs
//!
//! - Subscription negative-index rule: `Doc/reference/expressions.rst`,
//!   section "Subscriptions" — "built-in sequences all provide a
//!   `__getitem__` method that interprets negative indices by adding
//!   the length of the sequence to the index... The resulting value
//!   must be a nonnegative integer less than the number of items in
//!   the sequence." An index that is still out of range after that
//!   adjustment has no row here: CPython raises `IndexError`, and this
//!   domain carries no exception channel this wave (per the brief) —
//!   `subscript_read` answers `None`, the same "not modeled" honesty
//!   every other decline in this file uses.
//! - Mapping subscription: same section — "the expression list must
//!   evaluate to an object whose value is one of the keys of the
//!   mapping, and the subscription selects the value in the mapping
//!   that corresponds to that key."
//! - `d[key]` on a missing key: `Doc/library/stdtypes.rst`, "Mapping
//!   Types — dict" — "Raises a `KeyError` if key is not in the map."
//!   Again no exception channel this wave, so a missing string key
//!   answers `None` from `subscript_read`, not a fabricated value.
//! - `len(d)`: same section, `describe:: len(d)` — "Return the number
//!   of items in the dictionary d."
//! - `dict.get`: same section, `method:: get(key, default=None, /)` —
//!   "Return the value for key if key is in the dictionary, else
//!   default. If default is not given, it defaults to None, so that
//!   this method never raises a KeyError."

use refined_domain::abstract_value::{
    known_values, null_value, unknown, AbstractValue, Kind, ObjectKey, PrimitiveKind,
};
use refined_domain::known_constructors::{known_list, known_object};
use refined_domain::trust_grades::TrustProved;

/// A Python `list` display (`[a, b, c]`): `Kind::List` with one exact
/// slot per element, in source order. `known_list`'s own floor logic
/// already carries a weaker-grade element's trust up to the whole
/// list — this constructor states nothing further about grade.
pub fn list_literal_value(elements: &[AbstractValue]) -> AbstractValue {
    known_list(elements.to_vec(), TrustProved)
}

/// A Python `tuple` display (`(a, b, c)`): the same exact-positional-
/// slots shape a `list` display carries — `Kind::List` is this
/// domain's one sequence kind (module doc: no dedicated tuple
/// variant exists, matching the TS twin, which has no tuple sort to
/// port from either). A one-element tuple `(a,)` and a zero-element
/// tuple `()` both pass through unchanged; the caller's own parse
/// already resolved the trailing-comma/parenthesized-expression
/// grammar before this function sees the element list.
pub fn tuple_literal_value(elements: &[AbstractValue]) -> AbstractValue {
    known_list(elements.to_vec(), TrustProved)
}

/// A Python `dict` display (`{k: v, ...}`) with STRING-LITERAL keys
/// only. `keys[i]` is the string a key expression displayed as a
/// literal; `None` at a position means that key expression was not a
/// string literal (an int key, a computed key, an f-string key, a
/// `**spread` entry) — this domain's `ObjectKey.name` has no slot for
/// a non-string key, so the presence of even one `None` makes the
/// WHOLE literal `unknown()` rather than silently omitting that one
/// entry (an omission would misreport the dict's key set to every
/// later `subscript_read`/`dict_get_result`/`len_result` call, which
/// is worse than declining outright).
///
/// `keys` and `values` are the same length, one key AbstractValue per
/// value at the same index — the caller's own walk of the dict
/// display's key/value expression pairs. A duplicate string key
/// follows CPython's own "if a key occurs more than once, the last
/// value... becomes the corresponding value" rule
/// (library/stdtypes.rst, `dict(...)` constructor doc, the same rule
/// a literal display honors): this function keeps the LAST ObjectKey
/// entry for a repeated name, matching that overwrite.
pub fn dict_literal_value(keys: &[Option<String>], values: &[AbstractValue]) -> AbstractValue {
    if keys.len() != values.len() {
        return unknown();
    }
    if keys.iter().any(|key| key.is_none()) {
        return unknown();
    }
    let mut entries: Vec<ObjectKey> = Vec::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let name = key.clone().expect("checked above: no None key remains");
        // last-value-wins on a repeated key, matching CPython's own
        // dict-display overwrite rule
        if let Some(existing) = entries.iter_mut().find(|entry| entry.name == name) {
            existing.value = value.clone();
        } else {
            entries.push(ObjectKey {
                name,
                value: value.clone(),
            });
        }
    }
    known_object(entries, None, true, TrustProved, false)
}

/// The 0-based (post negative-index-adjustment) integer index an
/// AbstractValue states, if it is a single known Integer-sorted
/// value. Boolean-sorted values are NOT accepted here: `s[True]` is
/// legal Python (`True` is an `int`), but no row in this file's
/// corpus band needs that cross-sort read, and accepting it here
/// would be an unasked-for widening of this function's contract.
fn known_integer_index(index: &AbstractValue) -> Option<i64> {
    if index.kind != Kind::Values || index.values.len() != 1 {
        return None;
    }
    if index.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    Some(index.values[0] as i64)
}

/// The string an AbstractValue states, if it is a single known
/// String-sorted value — the same code-point-vector shape
/// `string_models.rs`'s `exact_string_text` reads (this file is a
/// sibling in the same crate directory but a different Rust crate
/// from `refined_domain`, so the conversion is repeated here rather
/// than reaching into `string_models.rs`'s private helper or widening
/// its visibility for one caller).
fn known_string_key(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(
        value
            .values
            .iter()
            .filter_map(|code_point| char::from_u32(*code_point as i64 as u32))
            .collect(),
    )
}

/// `container[index]` on a known LIST/TUPLE receiver (`Kind::List`)
/// with a known Integer index: negative indexing adjusts by the
/// list's own length first (expressions.rst, "Subscriptions" —
/// "interprets negative indices by adding the length of the sequence
/// to the index"), and the adjusted index must land in
/// `0..items.len()` ("a nonnegative integer less than the number of
/// items"). An index still out of range after adjustment answers
/// `None`: CPython raises `IndexError`, which this domain has no
/// channel for this wave (AGENT-BRIEF: "the exception channel
/// doesn't exist").
fn list_index_read(items: &[AbstractValue], index: i64) -> Option<AbstractValue> {
    let length = items.len() as i64;
    let adjusted = if index < 0 { index + length } else { index };
    if adjusted < 0 || adjusted >= length {
        return None;
    }
    Some(items[adjusted as usize].clone())
}

/// `container[key]` on a known DICT receiver (`Kind::Object`) with a
/// known string key: the value at that key's `ObjectKey` entry, or
/// `None` if no entry carries that name — `d[key]` raises `KeyError`
/// on a miss (library/stdtypes.rst, dict's `d[key]` row), which this
/// domain has no channel for this wave, matching the list/tuple
/// out-of-range row's same honesty.
fn dict_key_read(keys: &[ObjectKey], key: &str) -> Option<AbstractValue> {
    keys.iter()
        .find(|entry| entry.name == key)
        .map(|entry| entry.value.clone())
}

/// `container[index]` — the subscription read (expressions.rst,
/// "Subscriptions"): a known list/tuple (`Kind::List`) with a known
/// Integer index, or a known dict (`Kind::Object`) with a known
/// String-sorted key. Every other receiver shape or index/key shape
/// answers `None` — an unknown receiver, a non-Integer index into a
/// list, a non-String key into a dict, or a slice — none of those are
/// modeled here and this function declines honestly rather than
/// guessing.
pub fn subscript_read(container: &AbstractValue, index: &AbstractValue) -> Option<AbstractValue> {
    match container.kind {
        Kind::List => {
            let position = known_integer_index(index)?;
            list_index_read(&container.items, position)
        }
        Kind::Object => {
            let key = known_string_key(index)?;
            dict_key_read(&container.keys, &key)
        }
        _ => None,
    }
}

/// `len(container)` — an Integer-tagged exact count:
/// - a known list/tuple (`Kind::List`): `items.len()`.
/// - a known dict (`Kind::Object`): `keys.len()` (library/stdtypes.rst,
///   dict's `describe:: len(d)` — "the number of items in the
///   dictionary d").
/// - an exact string (`Kind::Values` tagged `PrimitiveKind::String`):
///   `values.len()`, one code point per `f64` — the same count
///   `string_models.rs` already establishes `len()` reads as.
///
/// Every other shape (an unknown value, a non-string `Kind::Values`)
/// answers `None`.
pub fn len_result(container: &AbstractValue) -> Option<AbstractValue> {
    let count = match container.kind {
        Kind::List => container.items.len(),
        Kind::Object => container.keys.len(),
        Kind::Values if container.kind_tag == Some(PrimitiveKind::String) => container.values.len(),
        _ => return None,
    };
    Some(known_values(
        vec![count as f64],
        PrimitiveKind::Integer,
        TrustProved,
    ))
}

/// `dict.get(key, default=None, /)` — library/stdtypes.rst, dict's
/// `method:: get`: "Return the value for key if key is in the
/// dictionary, else default. If default is not given, it defaults to
/// None, so that this method never raises a KeyError." A present key
/// answers its value; an absent key answers the caller's `default`
/// argument if one was passed, else the null state (`null_value`,
/// `abstract_value.rs`) standing in for Python's `None` — the same
/// exactly-null admission the Lean kernel's AbsentMark split carries
/// (`null_value`'s own doc). Only a known-`Kind::Object` receiver
/// with a known-String key is modeled; every other shape answers
/// `None`.
pub fn dict_get_result(
    container: &AbstractValue,
    key: &AbstractValue,
    default: Option<&AbstractValue>,
) -> Option<AbstractValue> {
    if container.kind != Kind::Object {
        return None;
    }
    let key_text = known_string_key(key)?;
    if let Some(found) = dict_key_read(&container.keys, &key_text) {
        return Some(found);
    }
    Some(match default {
        Some(default_value) => default_value.clone(),
        None => null_value(),
    })
}

/// Mutating container methods (`list.append`/`list.pop`, `set.add`,
/// `dict.update`, `.clear`, and their kin) always answer `None` —
/// declined, not "not yet modeled." A mutating call's INTERESTING
/// effect is not its return value (several of these return `None`/the
/// popped element themselves) but the WRITE it performs on the
/// receiver's own state, and this file's functions are pure value
/// readers with no receiver to write back into — there is no `World`
/// or environment parameter here for a write to land in. Modeling a
/// mutating method's write effect is the walk's job (AGENT-BRIEF: "a
/// later unit"), not this file's; answering only the return value
/// while silently dropping the write would be unsound (a caller could
/// read `xs.append(1)`'s `None` return and never learn `xs` grew).
pub fn mutating_method_result(_method: &str, _receiver: &AbstractValue) -> Option<AbstractValue> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    fn string(text: &str) -> AbstractValue {
        let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
        known_values(code_points, PrimitiveKind::String, TrustProved)
    }

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
            &[Some("a".to_string()), Some("b".to_string())],
            &[integer(1.0), integer(2.0)],
        );
        assert_eq!(built.kind, Kind::Object);
        assert_eq!(subscript_read(&built, &string("a")), Some(integer(1.0)));
        assert_eq!(subscript_read(&built, &string("b")), Some(integer(2.0)));
    }

    #[test]
    fn dict_literal_with_a_computed_key_answers_unknown() {
        let built = dict_literal_value(&[None, Some("b".to_string())], &[integer(1.0), integer(2.0)]);
        assert_eq!(built.kind, Kind::Unknown);
    }

    #[test]
    fn dict_literal_keeps_the_last_value_for_a_repeated_key() {
        let built = dict_literal_value(
            &[Some("a".to_string()), Some("a".to_string())],
            &[integer(1.0), integer(2.0)],
        );
        assert_eq!(built.keys.len(), 1);
        assert_eq!(subscript_read(&built, &string("a")), Some(integer(2.0)));
    }

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
    fn subscript_read_string_key_into_dict() {
        let dict = dict_literal_value(&[Some("k".to_string())], &[integer(5.0)]);
        assert_eq!(subscript_read(&dict, &string("k")), Some(integer(5.0)));
    }

    #[test]
    fn subscript_read_missing_dict_key_declines() {
        let dict = dict_literal_value(&[Some("k".to_string())], &[integer(5.0)]);
        assert_eq!(subscript_read(&dict, &string("missing")), None);
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
            &[Some("a".to_string()), Some("b".to_string())],
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

    // --- dict.get present/absent/default ---

    #[test]
    fn dict_get_present_key_answers_its_value() {
        let dict = dict_literal_value(&[Some("k".to_string())], &[integer(5.0)]);
        let got = dict_get_result(&dict, &string("k"), None).expect("get(present) must decide");
        assert_eq!(got, integer(5.0));
    }

    #[test]
    fn dict_get_absent_key_with_no_default_answers_null() {
        let dict = dict_literal_value(&[Some("k".to_string())], &[integer(5.0)]);
        let got = dict_get_result(&dict, &string("missing"), None).expect("get(absent) must decide");
        assert_eq!(got.kind, Kind::Null);
    }

    #[test]
    fn dict_get_absent_key_with_default_answers_the_default() {
        let dict = dict_literal_value(&[Some("k".to_string())], &[integer(5.0)]);
        let fallback = integer(0.0);
        let got = dict_get_result(&dict, &string("missing"), Some(&fallback))
            .expect("get(absent, default) must decide");
        assert_eq!(got, fallback);
    }

    // --- mutating methods decline ---

    #[test]
    fn mutating_methods_decline() {
        let list = list_literal_value(&[integer(1.0)]);
        assert_eq!(mutating_method_result("append", &list), None);
        assert_eq!(mutating_method_result("pop", &list), None);
        assert_eq!(mutating_method_result("add", &list), None);
        assert_eq!(mutating_method_result("update", &list), None);
        assert_eq!(mutating_method_result("clear", &list), None);
    }
}
