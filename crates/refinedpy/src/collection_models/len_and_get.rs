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
use super::subscript_read::dict_key_read_written;

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
    // `len(d)` on an UNBOUNDED-KEY `dict[str, X]` receiver
    // (`Kind::ObjectStar` — `check.rs::seed_parameters`' `known_dict_star`
    // seed): the declaration states which values every present key holds,
    // never how many keys are present, so no exact count exists. It is
    // still a count: `len()` "Return the length (the number of items) of
    // an object" (library/functions.rst, `len`), and no object has a
    // negative item count — `[0, +inf)` is the claim the receiver's own
    // shape supports, the same non-negative-count answer the bytes arm
    // above gives a value whose byte content is unread. Answering `None`
    // here instead left every `len()` of a dict parameter undetermined,
    // even where the sink it flows into refuses the whole ray.
    if container.kind == Kind::ObjectStar {
        let grade = trust_level_of(container);
        // Every key this receiver was WRITTEN at is present
        // (`dict_with_item`'s own star arm records one entry per written
        // key, distinct by `(name, numeric)`), so the count is at least
        // how many were recorded — a floor the bare `[0, +inf)` above
        // misses. It stays a floor, never an exact count: the declaration
        // states nothing about how many keys were already present before
        // the writes.
        let floor = container.keys.len() as f64;
        return Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(floor), refined_sets::refinement_forms::integer()]),
                None,
                grade,
                SetKindTag::None,
            )
        });
    }
    // `len(json.loads(text))` — the parsed value is the JSON conversion
    // table's own union (`expressions::json_re::json_loads_value_space`),
    // whose list and dict arms are `opaque_value`s: the kind of thing is
    // known, its contents are not, so there is no exact count and no
    // element set. The COUNT is still a count — `len()` "Return the
    // length (the number of items) of an object" (library/functions.rst),
    // and no object has a negative one — so `[0, +inf)` is the claim
    // every arm on which `len()` is defined at all supports, the same
    // non-negative-count answer the bytes and dict-star arms above give a
    // value whose contents are unread. That is what A7.edge.json's own
    // `parsed_length_is_sixteen` needs: `n == 16` narrows the ray to the
    // exact `{16}`, which Age admits, instead of leaving `n` with no
    // reading at all.
    if is_json_sized_union(container) {
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

/// Whether `container` is a `Kind::KindUnion` at least one of whose arms
/// is a SIZED thing — a container the parse could have produced whose
/// item count `len()` answers. `json.loads`'s own return space is the
/// one producer (`expressions::json_re::json_loads_value_space`), where
/// the sized arms are the `"a list"` / `"a dict"` opaque values and the
/// whole-strings set.
///
/// Requiring at least one sized arm, rather than every arm being sized,
/// is what the `len()` reading needs: on a run that reaches an unsized
/// arm the call raises `TypeError` and no value flows at all, so the
/// count claim is about exactly the runs where a count exists.
fn is_json_sized_union(container: &AbstractValue) -> bool {
    if container.kind != Kind::KindUnion {
        return false;
    }
    container.arms.iter().any(|arm| {
        matches!(arm.kind_word, Some("a list") | Some("a dict"))
            || (arm.kind == Kind::Set && arm.kind_tag.is_none())
    })
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
/// already would. The key's SORT does not gate the row —
/// `dict_star_index_read`'s own doc states why the star's value law
/// holds for any hashable key — and neither does the key being
/// SPELLABLE at all: an unread `k: str` parameter names no entry this
/// domain can match against the written set, but the star's own law
/// still describes every value the receiver can hold at every key, so
/// the two-branch join is exactly as true for an unknown key as for a
/// known-absent one. Only the written-key shortcut needs a spelling; an
/// unspellable key simply skips it and folds both branches, the same
/// answer `mutated_receiver`'s own star arm already gives the write
/// path for this shape.
fn dict_star_get_result(container: &AbstractValue, key: &AbstractValue, default: Option<&AbstractValue>) -> Option<AbstractValue> {
    // A key this receiver was WRITTEN at is PRESENT — the write put it
    // there (`dict_with_item`'s own star arm) — so `.get` on it takes the
    // present branch only, answering the recorded value exactly, with no
    // miss branch to fold in. `dict_key_read_written` (not the plain
    // `dict_key_read`) is the reader that matters here: a GUARD-recorded
    // entry (`narrowing::compare::narrow_dict_membership_against_
    // literal_key`, `DictKey::guarded`'s own doc) proves presence only AT
    // THE GUARD, and a mutation since — including one inside a callee
    // handed this receiver — can have removed the key, so a guard entry
    // must fall through to the two-branch join below rather than take
    // this shortcut.
    if let Some(key) = known_dict_key(key) {
        if let Some(written) = dict_key_read_written(&container.keys, &key) {
            return Some(written);
        }
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
