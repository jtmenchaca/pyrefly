//! The `dict` display constructor: `dict_literal_value` builds a
//! `Kind::Object` from a dict `{k: v, ...}`'s own key/value expression
//! pairs. See `collection_models`'s own module doc for why `dict` maps
//! to `Kind::Object` rather than `Kind::Collection`/`Flavor::Map`.

use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;

use super::dict_key::DictKey;

/// A Python `dict` display (`{k: v, ...}`) with STRING-LITERAL or known
/// single-Integer keys. `keys[i]` is the key expression's own
/// `DictKey` spelling; `None` at a position means that key expression
/// was not one of the two supported shapes (a computed key, an
/// f-string key, a float/bool/tuple key, a `**spread` entry) — this
/// domain's `ObjectKey.name`/`numeric` pair has no slot for any other
/// key shape, so the presence of even one `None` makes the WHOLE
/// literal `unknown()` rather than silently omitting that one entry
/// (an omission would misreport the dict's key set to every later
/// `subscript_read`/`dict_get_result`/`len_result` call, which is
/// worse than declining outright).
///
/// `keys` and `values` are the same length, one key AbstractValue per
/// value at the same index — the caller's own walk of the dict
/// display's key/value expression pairs. A duplicate key (same name
/// AND same numeric-ness) follows CPython's own "if a key occurs more
/// than once, the last value... becomes the corresponding value" rule
/// (library/stdtypes.rst, `dict(...)` constructor doc, the same rule
/// a literal display honors): this function keeps the LAST ObjectKey
/// entry for a repeated key. A string key and an int key of the same
/// spelling (`"15"` and `15`) are NOT a repeat — they hold two
/// separate entries, matching CPython's own `1 == "1"` being `False`
/// (`abstract_value.rs`'s own `ObjectKey` doc).
pub fn dict_literal_value(keys: &[Option<DictKey>], values: &[AbstractValue]) -> AbstractValue {
    if keys.len() != values.len() {
        return unknown();
    }
    if keys.iter().any(|key| key.is_none()) {
        return unknown();
    }
    let mut entries: Vec<ObjectKey> = Vec::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let key = key.clone().expect("checked above: no None key remains");
        // last-value-wins on a repeated key, matching CPython's own
        // dict-display overwrite rule — a string key and a numeric key
        // of the same spelling are DIFFERENT keys, so both `name` AND
        // `numeric` must match for this to be a repeat
        if let Some(existing) = entries.iter_mut().find(|entry| entry.name == key.name && entry.numeric == key.numeric) {
            existing.value = value.clone();
        } else {
            entries.push(ObjectKey {
                name: key.name,
                numeric: key.numeric,
                value: value.clone(),
            });
        }
    }
    known_object(entries, None, true, TrustProved, false)
}
