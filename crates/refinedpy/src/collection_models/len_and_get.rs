//! `len(container)` and `dict.get(key, default=None, /)` — the two
//! read-side queries that are neither a plain subscript nor a
//! mutation.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::element_of_object_star;
use refined_domain::lattice_operations::join_known;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::repetition_window_forms::as_repetition;

use super::dict_key::known_dict_key;
use super::subscript_read::dict_key_read;

/// `len(container)` — an Integer-tagged exact count:
/// - a known list/tuple (`Kind::List`): `items.len()`.
/// - a known dict (`Kind::Object`): `keys.len()` (library/stdtypes.rst,
///   dict's `describe:: len(d)` — "the number of items in the
///   dictionary d").
/// - an exact string (`Kind::Values` tagged `PrimitiveKind::String`):
///   `values.len()`, one code point per `f64` — the same count
///   `string_models.rs` already establishes `len()` reads as.
/// - an UNKNOWN-LENGTH star-shaped sequence (`Kind::Set`, the bare star
///   `as_repetition` reads back — `star_element_read`'s own doc, a
///   declared `list[X]`/`set[X]`/`Sequence[X]` parameter with no
///   concrete items): an Integer-tagged SET, `[window.lo, window.hi]`
///   (unbounded `hi` answers `[window.lo, +inf)`), never one exact
///   count — the real length is unstated, only its own declared bounds
///   are known.
///
/// Every other shape (an unknown value, a non-string `Kind::Values`, a
/// bounded-but-not-bare-star `Kind::Set`) answers `None`.
pub fn len_result(container: &AbstractValue) -> Option<AbstractValue> {
    if container.kind == Kind::Set && container.set_kind_tag == SetKindTag::None {
        let window = as_repetition(&container.set)?;
        let mut forms = vec![at_least(window.lo as f64)];
        if let Some(hi) = window.hi {
            forms.push(at_most(hi as f64));
        }
        let grade = trust_level_of(container);
        return Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(forms), None, grade, SetKindTag::None)
        });
    }
    // `len(<some bytes value>)` — a `str.encode()` result whose own byte
    // content is unread (`bytes_models::ENCODED_BYTES_WORD`). The byte
    // count is unstated, but it is a count: a non-negative integer.
    // `len()` "Return the length (the number of items) of an object"
    // (library/functions.rst), and no object has a negative one — so
    // `[0, +inf)` is the sound claim, which is what A3.xfer.encode's
    // own `encoded_length_outside` needs to refuse against `Wide`'s
    // `[0, 200]` rather than state nothing at all.
    if container.kind_word == Some(crate::bytes_models::ENCODED_BYTES_WORD) {
        let grade = trust_level_of(container);
        return Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, grade, SetKindTag::None)
        });
    }
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

/// `dict.get(key, default=None, /)` on an UNBOUNDED-KEY `dict[str, X]`
/// receiver (`Kind::ObjectStar` — `check.rs::seed_parameters`'
/// `known_dict_star` seed): unlike a closed dict, this receiver states
/// no fixed key set to prove the key present OR absent against, so
/// `.get(k)` reads BOTH branches stdtypes.rst's `get` describes at
/// once — the value if present (the star's own element,
/// `element_of_object_star`), OR the miss-branch value if absent (the
/// caller's own `default` argument when one was passed, else the null
/// state standing in for Python's `None`, the same "default defaults to
/// None" reading `dict_get_result`'s own closed-dict arm gives). Joined
/// with `join_known` rather than wrapped as a maybe carrier around the
/// element alone: a passed `default` is a REAL value the miss branch
/// answers, not merely "absence," so dropping it and always claiming
/// "element or None" would be UNSOUND the moment a caller passes a
/// non-None default — `join_known` folds it in as an ordinary second
/// arm the same way any other two-branch value join in this crate
/// already would. A non-string key sort answers `None`, the same "not
/// this dict's key sort" decline the closed-dict arm gives below.
fn dict_star_get_result(container: &AbstractValue, key: &AbstractValue, default: Option<&AbstractValue>) -> Option<AbstractValue> {
    let key = known_dict_key(key)?;
    if key.numeric {
        return None;
    }
    let element = element_of_object_star(container)?;
    let miss_branch = match default {
        Some(default_value) => default_value.clone(),
        None => null_value(),
    };
    Some(join_known(element, miss_branch))
}

/// `dict.get(key, default=None, /)` — library/stdtypes.rst, dict's
/// `method:: get`: "Return the value for key if key is in the
/// dictionary, else default. If default is not given, it defaults to
/// None, so that this method never raises a KeyError." A present key
/// answers its value; an absent key answers the caller's `default`
/// argument if one was passed, else the null state (`null_value`,
/// `abstract_value.rs`) standing in for Python's `None` — the same
/// exactly-null admission the Lean kernel's AbsentMark split carries
/// (`null_value`'s own doc). A known-`Kind::Object` receiver with a
/// known String- or Integer-sorted key (`known_dict_key`'s own doc),
/// or an unbounded-key `dict[str, X]` receiver with a known string key
/// (`dict_star_get_result`'s own doc), is modeled; every other shape
/// answers `None`.
pub fn dict_get_result(
    container: &AbstractValue,
    key: &AbstractValue,
    default: Option<&AbstractValue>,
) -> Option<AbstractValue> {
    if container.kind == Kind::ObjectStar {
        return dict_star_get_result(container, key, default);
    }
    if container.kind != Kind::Object {
        return None;
    }
    let key = known_dict_key(key)?;
    if let Some(found) = dict_key_read(&container.keys, &key) {
        return Some(found);
    }
    Some(match default {
        Some(default_value) => default_value.clone(),
        None => null_value(),
    })
}
