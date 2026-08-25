//! `container[index]` — the subscription read (expressions.rst,
//! "Subscriptions"), across every receiver shape this domain models:
//! known list/tuple, exact string, known dict (by exact key or a
//! finite union of keys), an unbounded-key `dict[str, X]`, and an
//! unknown-length repetition-shaped sequence.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::element_of_object_star;
use refined_domain::lattice_operations::join_known;
use refined_domain::trust_grades::trust_level_of;
use refined_sets::repetition_window_forms::as_repetition;

use super::dict_key::known_dict_key;
use super::dict_key::DictKey;
use super::kernel_join::dict_key_set_read;

/// The 0-based (post negative-index-adjustment) integer index an
/// AbstractValue states, if it is a single known Integer-sorted
/// value. Boolean-sorted values are NOT accepted here: `s[True]` is
/// legal Python (`True` is an `int`), but no row in this file's
/// corpus band needs that cross-sort read, and accepting it here
/// would be an unasked-for widening of this function's contract.
pub(super) fn known_integer_index(index: &AbstractValue) -> Option<i64> {
    if index.kind != Kind::Values || index.values.len() != 1 {
        return None;
    }
    if index.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    Some(index.values[0] as i64)
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
pub(super) fn list_index_read(items: &[AbstractValue], index: i64) -> Option<AbstractValue> {
    let length = items.len() as i64;
    let adjusted = if index < 0 { index + length } else { index };
    if adjusted < 0 || adjusted >= length {
        return None;
    }
    Some(items[adjusted as usize].clone())
}

/// `container[index]` on a known LIST/TUPLE receiver whose index is a
/// bounded Integer RANGE rather than one exact value — `["ok", "warn",
/// "error"][code]` where `code: Annotated[int, Field(ge=0, le=2)]` seeds
/// `Kind::Set` (`check.rs::seed_parameters`'s scalar-declared-set arm),
/// never `Kind::Values`, so `known_integer_index` (the exact-value
/// reader) answers `None` and this is the caller's fallback. Reads the
/// index's own closed bound (`integer_range_bounds`) and, ONLY when
/// every integer in `[lo, hi]` lands in range after negative-index
/// adjustment (never a partial range — a bound that could still fall
/// outside `items` after adjustment answers `None` rather than guessing
/// which positions are safe), joins every position `items[lo..=hi]` —
/// the loosest sound answer once the concrete index is unknown but its
/// possible positions are all known and all in-bounds. No kernel round
/// trip: `hi - lo` is always small enough to enumerate directly (a
/// range wide enough to be impractical to enumerate is also almost
/// certainly wider than the list itself, which the in-bounds check
/// already refuses).
fn list_bounded_range_read(items: &[AbstractValue], index: &AbstractValue) -> Option<AbstractValue> {
    if index.kind != Kind::Set || index.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    let (lo, hi) = integer_range_bounds(&index.set)?;
    let length = items.len() as i64;
    if lo < 0 || hi < lo {
        // negative bounds/indices are not modeled here — CPython's own
        // adjustment (`index + length`) would need to apply PER
        // CANDIDATE index, which a single [lo, hi] window cannot state
        // uniformly once negative values are mixed in with nonnegative
        // ones; a purely negative or purely nonnegative window still
        // wants an explicit brief before widening this reader
        return None;
    }
    if hi >= length {
        return None;
    }
    let mut joined: Option<AbstractValue> = None;
    for position in lo..=hi {
        let candidate = items[position as usize].clone();
        joined = Some(match joined {
            None => candidate,
            Some(so_far) => join_known(so_far, candidate),
        });
    }
    joined
}

/// The closed integer bound `[lo, hi]` a scalar `RefinedSet` states, read
/// from its own top-level `AtLeast`/`Above`/`AtMost`/`Below` forms — the
/// same kind of syntactic hull `foreign_edge.rs::hull_of` reads for its
/// own uncarriable-corner check, narrowed here to the CLOSED case only
/// (`None` the moment either side is unbounded, since an unbounded range
/// can never be enumerated). `Above`/`Below` are the strict-bound forms
/// (`x > a`/`x < a`) — `.ceil()`/`.floor()` step them to the nearest
/// INTEGER the strict bound still admits, which is exact for an
/// Integer-sorted set (a strict bound between two consecutive integers
/// admits the same integers a non-strict bound at the stepped value
/// would). A set carrying any OTHER form (`Union`, `MultipleOf`, `OneOf`,
/// a bare `Integer` marker with no numeric bound) answers `None` — this
/// reader is the plain closed-window case only, not a general hull.
fn integer_range_bounds(set: &refined_sets::refinement_forms::RefinedSet) -> Option<(i64, i64)> {
    use refined_sets::refinement_forms::Form;
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &set.forms {
        match form.form {
            Form::AtLeast => lo = Some(lo.map_or(form.a, |current: f64| current.max(form.a))),
            Form::Above => lo = Some(lo.map_or(form.a.floor() + 1.0, |current: f64| current.max(form.a.floor() + 1.0))),
            Form::AtMost => hi = Some(hi.map_or(form.a, |current: f64| current.min(form.a))),
            Form::Below => hi = Some(hi.map_or(form.a.ceil() - 1.0, |current: f64| current.min(form.a.ceil() - 1.0))),
            Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, hi) = (lo?, hi?);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// `container[index]` on a known EXACT STRING receiver (`Kind::Values`
/// tagged `PrimitiveKind::String`) with a known Integer index: the same
/// negative-index adjustment `list_index_read` applies (expressions.rst,
/// "Subscriptions" — the adjustment rule is stated once, for "built-in
/// sequences" generally, and a string is one of those sequences,
/// library/stdtypes.rst's Text Sequence Type section), landing on a
/// SINGLE code point that answers a one-character `Kind::Values` String
/// — the same shape `evaluate_slice`'s own sliced-string answer already
/// builds (`expressions.rs`). An index still out of range after
/// adjustment answers `None`: CPython raises `IndexError`
/// (`subscript_provable_raise`'s own row already proves this case
/// separately), which this domain has no read channel for.
fn string_index_read(values: &[f64], index: i64) -> Option<AbstractValue> {
    let length = values.len() as i64;
    let adjusted = if index < 0 { index + length } else { index };
    if adjusted < 0 || adjusted >= length {
        return None;
    }
    Some(known_values(vec![values[adjusted as usize]], PrimitiveKind::String, refined_domain::trust_grades::TrustProved))
}

/// `container[key]` on a known DICT receiver (`Kind::Object`) with a
/// known string OR int key: the value at that key's `ObjectKey` entry
/// — matched by BOTH `name` and `numeric` (a string key and an int key
/// of the same spelling are different entries, `ObjectKey`'s own doc)
/// — or `None` if no entry carries that identity. `d[key]` raises
/// `KeyError` on a miss (library/stdtypes.rst, dict's `d[key]` row),
/// which this domain has no channel for this wave, matching the
/// list/tuple out-of-range row's same honesty.
pub(super) fn dict_key_read(keys: &[ObjectKey], key: &DictKey) -> Option<AbstractValue> {
    keys.iter()
        .find(|entry| entry.name == key.name && entry.numeric == key.numeric)
        .map(|entry| entry.value.clone())
}

/// `container[index]` on a KNOWN-LENGTH-UNKNOWN, known-element-set
/// receiver: `Kind::Set` whose only form is `Form::Star(element)` — the
/// shape `check.rs::seed_parameters` builds for a `list[X]`/`set[X]`/
/// `Sequence[X]` parameter, `X`'s own set repeated rather than nested
/// into exact positional slots (unlike `Kind::List`, which states an
/// exact count `list_index_read` bounds-checks against). A repetition
/// window's own positions never hold anything outside its element set —
/// the grammar's definition, not a fact this function proves — so ANY
/// known Integer index reads the same answer: "some member of
/// `element`", regardless of the window's own `{lo, hi}`
/// (`check.rs::seed_parameters` seeds the bare unbounded window when the
/// declaration states no length bound, or a TIGHTER `{lo, hi}` window
/// when it does — `typereading.rs`'s own `DeclaredRefinement::
/// element_length` doc — but a symbolic index read can never bounds-
/// check against either shape host-side: the concrete integer VALUE at
/// the index is unknown either way, only its membership in the element
/// alphabet is known). `as_repetition` reads any repetition window back
/// to its element without a kernel round trip
/// (`refined_sets::repetition_window_forms`, the same reader
/// `format_for_hover.rs`/`format_string_shapes.rs` already use for the
/// string-domain's own `C*`). Any OTHER set shape (a union, a bare
/// scalar range with no repetition wrapper) answers `None`.
pub(super) fn star_element_read(container: &AbstractValue, index: &AbstractValue) -> Option<AbstractValue> {
    if container.kind != Kind::Set || container.set_kind_tag != SetKindTag::None {
        return None;
    }
    known_integer_index(index)?;
    let repeated = as_repetition(&container.set)?;
    Some(AbstractValue {
        kind_tag: container.kind_tag,
        ..known_set(repeated.element, None, trust_level_of(container), SetKindTag::None)
    })
}

/// `container[key]` on a `dict[str, X]` PARAMETER'S own unbounded-key
/// receiver (`Kind::ObjectStar` — `check.rs::seed_parameters`' own
/// `known_dict_star` seed, `element_of_object_star`'s doc): every
/// STRING key, if present, reads the star's wrapped element — the same
/// value `.get(k)` reads on its own present-key branch
/// (`dict_get_result`'s dict-star arm below). Gated on a STRING key
/// (`known_dict_key` proving a `numeric: false` identity) because the
/// declaration is `dict[str, X]`, never an int key — an Integer- or
/// unrecognized-sorted index answers `None`, the same "not this dict's
/// key sort" decline `dict_key_read` gives a closed dict.
///
/// This does not itself prove the key is PRESENT (an unbounded dict
/// states no fixed key set to check against) — it reads the value the
/// declaration states EVERY present key holds, the same way a closed
/// dict's own `dict_key_read` answers a value without separately
/// proving the runtime dict was literally built with that key. CPython
/// raises `KeyError` on a genuinely missing key either way
/// (`stdtypes.rst`'s `d[key]` row); this domain carries no exception
/// channel for that miss on a closed dict either, so an unbounded dict
/// keeps the identical honesty.
pub(super) fn dict_star_index_read(container: &AbstractValue, index: &AbstractValue) -> Option<AbstractValue> {
    let key = known_dict_key(index)?;
    if key.numeric {
        return None;
    }
    element_of_object_star(container)
}

/// `container[index]` — the subscription read (expressions.rst,
/// "Subscriptions"): a known list/tuple (`Kind::List`) with a known
/// Integer index, a known exact string (`Kind::Values` tagged
/// `PrimitiveKind::String`) with a known Integer index
/// (`string_index_read`'s own doc), a known dict (`Kind::Object`)
/// with a known String- or Integer-sorted key (`known_dict_key`'s own
/// doc — an Object receiver keyed numerically is still a DICT read,
/// never the list/tuple positional-index path above: the two receiver
/// kinds never share one dispatch arm), a dict keyed by a finite
/// UNION of known strings where every named entry is present
/// (`dict_key_set_read`'s own doc), an unbounded-key `dict[str, X]`
/// receiver with a known string key (`dict_star_index_read`'s own
/// doc), or an unknown-length sequence whose element set is known
/// (`star_element_read`'s own doc). Every other receiver shape or
/// index/key shape answers `None` — an unknown receiver, a
/// non-Integer index into a list or string, an unsupported key sort
/// into a dict, or a slice — none of those are modeled here and this
/// function declines honestly rather than guessing.
pub fn subscript_read(container: &AbstractValue, index: &AbstractValue) -> Option<AbstractValue> {
    match container.kind {
        Kind::List => {
            if let Some(position) = known_integer_index(index) {
                return list_index_read(&container.items, position);
            }
            list_bounded_range_read(&container.items, index)
        }
        Kind::Values if container.kind_tag == Some(PrimitiveKind::String) => {
            let position = known_integer_index(index)?;
            string_index_read(&container.values, position)
        }
        Kind::Object => match known_dict_key(index) {
            Some(key) => dict_key_read(&container.keys, &key),
            None => dict_key_set_read(&container.keys, index),
        },
        Kind::ObjectStar => dict_star_index_read(container, index),
        Kind::Set => star_element_read(container, index),
        _ => None,
    }
}
