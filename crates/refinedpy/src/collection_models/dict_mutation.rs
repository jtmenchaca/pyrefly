//! `dict.update`/`clear`/`setdefault`/`pop`/`popitem` — the dict-only
//! mutating method calls. See `mutated_receiver`'s own doc (in
//! `collection_models/mod.rs`) for the cited row-by-row contract.

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
