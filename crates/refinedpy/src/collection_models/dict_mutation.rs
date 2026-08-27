//! `dict.update`/`clear`/`setdefault`/`pop`/`popitem` — the dict-only
//! mutating method calls, over a CLOSED receiver
//! (`dict_mutated_receiver`) and over an unbounded-key one
//! (`dict_star_mutated_receiver`, which answers the subset of those rows
//! a star's recorded entries decide). See `mutated_receiver`'s own doc
//! (in `collection_models/mod.rs`) for the cited row-by-row contract.

use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;

use super::dict_key::known_dict_key;
use super::list_literal::list_literal_value;
use super::subscript_read::dict_key_read;

/// `dict.update`/`clear`/`setdefault`/`pop`/`popitem` — see
/// `mutated_receiver`'s own doc for the cited row-by-row contract.
pub(super) fn dict_mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
    match method {
        "update" => {
            let [other] = arguments else { return None };
            if other.kind != Kind::Object {
                return None;
            }
            let mut entries = receiver.keys.clone();
            for incoming in &other.keys {
                if let Some(existing) = entries
                    .iter_mut()
                    .find(|entry| entry.name == incoming.name && entry.numeric == incoming.numeric)
                {
                    existing.value = incoming.value.clone();
                } else {
                    entries.push(incoming.clone());
                }
            }
            Some((known_object(entries, None, true, TrustProved, false), null_value()))
        }
        "clear" if arguments.is_empty() => Some((known_object(Vec::new(), None, true, TrustProved, false), null_value())),
        "setdefault" => {
            let (key_expr, default) = match arguments {
                [key] => (key, None),
                [key, default] => (key, Some(default)),
                _ => return None,
            };
            let key = known_dict_key(key_expr)?;
            if let Some(found) = dict_key_read(&receiver.keys, &key) {
                return Some((receiver.clone(), found));
            }
            let default_value = default.cloned().unwrap_or_else(null_value);
            let mut entries = receiver.keys.clone();
            entries.push(ObjectKey {
                name: key.name,
                numeric: key.numeric,
                value: default_value.clone(),
            });
            Some((known_object(entries, None, true, TrustProved, false), default_value))
        }
        "pop" => {
            let (key_expr, default) = match arguments {
                [key] => (key, None),
                [key, default] => (key, Some(default)),
                _ => return None,
            };
            let key = known_dict_key(key_expr)?;
            if let Some(found) = dict_key_read(&receiver.keys, &key) {
                let entries: Vec<ObjectKey> = receiver
                    .keys
                    .iter()
                    .filter(|entry| !(entry.name == key.name && entry.numeric == key.numeric))
                    .cloned()
                    .collect();
                return Some((known_object(entries, None, true, TrustProved, false), found));
            }
            // an absent key with no default RAISES KeyError — this row
            // declines the whole call rather than mutate on a raise
            // (provable_raise is the raise channel, not this function)
            let default_value = default?;
            Some((receiver.clone(), default_value.clone()))
        }
        "popitem" if arguments.is_empty() => {
            let last = receiver.keys.last()?.clone();
            let entries: Vec<ObjectKey> = receiver.keys[..receiver.keys.len() - 1].to_vec();
            let key_value = if last.numeric {
                integer_key_value(&last.name)?
            } else {
                string_key_value(&last.name)
            };
            let pair = list_literal_value(&[key_value, last.value]);
            Some((known_object(entries, None, true, TrustProved, false), pair))
        }
        _ => None,
    }
}

/// `setdefault`/`pop`/`clear` on an UNBOUNDED-KEY mapping
/// (`Kind::ObjectStar` — `check::seed_parameters`' `known_dict_star`
/// seed, whose `keys` hold the entries a write already recorded and whose
/// `inner` states what every OTHER present key holds,
/// `dict_write::dict_with_item`'s own star arm).
///
/// - `setdefault(key, default=None)` — stdtypes.rst: "If *key* is in the
///   dictionary, return its value. If not, insert *key* with a value of
///   *default* and return *default*." A key with a RECORDED entry is
///   provably present, so the recorded value is returned and the receiver
///   is unchanged. A key with no recorded entry is not provably absent
///   either — the declaration never said which keys the mapping arrived
///   holding — so the returned value is the JOIN of `default` (the
///   insert branch) and the star's own law (the present branch), and the
///   key IS recorded afterward with that same join, since either branch
///   leaves it present holding one of the two. That is what A8.xfer.
///   getorinsert's rows need: `k in d` is provably True afterward, and
///   the returned value lands where `default` and the declaration both
///   land. A key expression this file cannot spell exactly (`k: str`, a
///   sort-only string window) still answers that same join — both of
///   the doc's branches land inside it whichever key it names — and only
///   the key-set recording is skipped, since no key can be claimed
///   present without knowing which one it is.
/// - `pop(key)`/`pop(key, default)` — same section: a RECORDED key
///   answers its own value and loses its entry. A key with no recorded
///   entry cannot be proven present OR absent, so the call declines
///   rather than pick a branch.
/// - `clear()` — "Remove all items from the dictionary": every key is
///   gone, recorded or not, which the EMPTY closed dict states exactly,
///   so this row leaves the star shape behind entirely.
///
/// `update`/`popitem` decline: both need the receiver's full key set —
/// `update` to know which incoming keys overwrite, `popitem` to name the
/// last-inserted entry — and a star states no such set.
pub(super) fn dict_star_mutated_receiver(
    method: &str,
    receiver: &AbstractValue,
    arguments: &[AbstractValue],
) -> Option<(AbstractValue, AbstractValue)> {
    match method {
        "clear" if arguments.is_empty() => Some((known_object(Vec::new(), None, true, TrustProved, false), null_value())),
        "setdefault" => {
            let (key_expr, default) = match arguments {
                [key] => (key, None),
                [key, default] => (key, Some(default)),
                _ => return None,
            };
            // An EXACTLY KNOWN key with a recorded entry is provably
            // present, so that entry's own value is what the call
            // answers and the receiver is unchanged.
            let key = known_dict_key(key_expr);
            if let Some(key) = &key
                && let Some(found) = dict_key_read(&receiver.keys, key)
            {
                return Some((receiver.clone(), found));
            }
            let element = refined_domain::known_constructors::element_of_object_star(receiver)?;
            let default_value = default.cloned().unwrap_or_else(null_value);
            let answer = refined_domain::lattice_operations::join_known(element, default_value);
            let mut written = receiver.clone();
            // Which key was written is only knowable when the key
            // expression states one exactly (`d.setdefault("z", …)`).
            // A key this file cannot spell — `k: str`, a sort-only
            // string window with no exact characters — still leaves the
            // RETURNED value fully determined: both of stdtypes.rst's
            // branches (return the present value, or insert and return
            // the default) land inside the same join of the star's own
            // element law and `default`, whichever key it turns out to
            // be. So the value is answered either way, and only the
            // key-set recording is skipped — no key is claimed present,
            // which is exactly what the mapping already stated.
            if let Some(key) = key {
                written.keys.push(ObjectKey {
                    name: key.name,
                    numeric: key.numeric,
                    value: answer.clone(),
                });
            }
            Some((written, answer))
        }
        "pop" => {
            let (key_expr, _default) = match arguments {
                [key] => (key, None),
                [key, default] => (key, Some(default)),
                _ => return None,
            };
            let key = known_dict_key(key_expr)?;
            let found = dict_key_read(&receiver.keys, &key)?;
            let mut written = receiver.clone();
            written.keys.retain(|entry| !(entry.name == key.name && entry.numeric == key.numeric));
            Some((written, found))
        }
        _ => None,
    }
}

/// A String-sorted AbstractValue spelling `text` — the same code-point
/// encoding `string_literal_value` builds (this file is out-of-crate
/// from `string_models.rs`, so the conversion is repeated here rather
/// than reaching into that file's own constructor for one caller,
/// matching the existing `known_string_key` note above).
fn string_key_value(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    refined_domain::abstract_value::known_values(code_points, PrimitiveKind::String, TrustProved)
}

/// An Integer-sorted AbstractValue for `popitem`'s `(key, value)` pair
/// when the popped entry is a numeric-keyed dict slot (`ObjectKey.name`
/// is the key's own plain decimal spelling, `DictKey::integer`'s own
/// doc) — parses the digits back to the `f64` the domain's Integer
/// values carry. `None` only if `name` is not a valid decimal spelling,
/// which never happens for an entry this file itself built via
/// `DictKey::integer`.
fn integer_key_value(name: &str) -> Option<AbstractValue> {
    let parsed: i64 = name.parse().ok()?;
    Some(refined_domain::abstract_value::known_values(vec![parsed as f64], PrimitiveKind::Integer, TrustProved))
}
