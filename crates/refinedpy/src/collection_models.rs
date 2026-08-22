/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Container VALUE states: `list`/`tuple`/`dict` literals, subscript
//! reads (`s[i]`, `d[key]`), `len()`, `dict.get`, and the mutation
//! contract (`mutated_receiver`, `dict_with_item`, `list_with_item`)
//! the walk's World calls to thread a write's new receiver value
//! through. Every mutation row answers `None` the moment the receiver
//! or an argument is not fully known — an unknown write is silently
//! dropped only by returning no new state, never guessed at (see
//! `mutated_receiver`'s own doc).
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
//! `{name: String, numeric: bool, value: AbstractValue}` pairs, never
//! a JS-style prototype-bearing map. This is a deliberate choice over
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
//! String- and int-keyed entries: a Python dict key that is a string
//! literal OR a single known `Integer`-sorted value has a slot in
//! `ObjectKey` — `ObjectKey.name` carries the key's spelling (a
//! string's own text, or an int key's plain decimal digits) and
//! `ObjectKey.numeric` tells the two apart (`abstract_value.rs`'s own
//! `ObjectKey` doc: an int key and a string key of the same spelling
//! are DIFFERENT Python dict keys — `1 == "1"` is `False`). Any other
//! key shape (a computed key this file cannot reduce to one of those
//! two sorts, a tuple key, a float/bool key — this domain does not
//! yet fold `1.0`/`True` into the same slot `1` occupies, per
//! stdtypes.rst's "values that compare equal... can be used
//! interchangeably") has no slot to occupy: `dict_literal_value` takes
//! `keys: &[Option<DictKey>]` — `None` at a position means "this key
//! expression is not a supported literal" — and that entire literal
//! answers `unknown()` rather than silently dropping the unsupported
//! entry (dropping would misreport the dict's key set to every later
//! read).
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
    known_set, known_values, null_value, unknown, AbstractValue, Kind, ObjectKey, PrimitiveKind, SetKindTag,
};
use refined_domain::known_constructors::{known_list, known_object};
use refined_domain::lattice_operations::{join_known, set_of_known};
use refined_domain::trust_grades::{min_trust_level, trust_level_of, TrustLevel, TrustProved};
use refined_kernel::kernel_bridge::kernel_if_loaded;
use refined_kernel::kernel_interface::KnownStateWire;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::repetition_window_forms::as_repetition;

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

/// One dict-display key's spelling and sort: a plain string key
/// (`numeric: false`, `name` is the string's own text), an int key
/// (`numeric: true`, `name` is the key's plain decimal spelling, e.g.
/// `"15"` for the key `15`), or an IDENTITY key (`numeric: false`,
/// `name` carries the identity tag under a reserved prefix — see
/// `DictKey::identity`'s own doc) — the same (name, numeric) identity
/// pair `ObjectKey` carries (`abstract_value.rs`'s own doc), read here
/// before the value side of a dict-display/comprehension row is known.
#[derive(Debug, Clone, PartialEq)]
pub struct DictKey {
    pub name: String,
    pub numeric: bool,
}

/// The reserved prefix an identity key's `name` always carries — chosen
/// so it can never collide with a plain string key's own text: a
/// Python string literal used as a dict key spells its own characters
/// verbatim into `DictKey::string`'s `name`, and this prefix contains
/// `\0` (NUL), a code point no `Doc/library/stdtypes.rst` string LITERAL
/// row this file's callers build from source text can ever produce (a
/// key read off actual Python source is always a sequence of ordinary
/// printable/escape characters, never an embedded NUL from a
/// `StringLiteral` node).
const IDENTITY_KEY_PREFIX: &str = "\0identity:";

impl DictKey {
    /// A plain string key, `numeric: false` — the ordinary case every
    /// existing string-literal-keyed dict display/constructor call
    /// builds.
    pub fn string(text: &str) -> DictKey {
        DictKey {
            name: text.to_owned(),
            numeric: false,
        }
    }

    /// An int key's plain decimal spelling, `numeric: true` — built
    /// from the key expression's own known Integer value
    /// (`format!("{}", value as i64)`, the same bare-integer spelling
    /// `expressions.rs`'s own `format_integer_spelling` builds for an
    /// f-string interpolation: Python's `str()` of an int has no
    /// decimal point).
    pub fn integer(value: i64) -> DictKey {
        DictKey {
            name: format!("{value}"),
            numeric: true,
        }
    }

    /// An IDENTITY key, `numeric: false` — a dict key that is neither a
    /// string nor an int, matched by PROVENANCE rather than by any
    /// value comparison (stdtypes.rst's mapping rule only requires a
    /// key be :term:`hashable`, never a string or number — a bare
    /// `object()` sentinel, hashable by identity alone, is a legal dict
    /// key this way). `tag` is `identity_key_tag`'s own answer: either an
    /// opaque value's source text (today, only `object_call`'s fixed
    /// `"object()"` tag, `builtin_models.rs`), or a `#`-prefixed spelling
    /// of a constructed class instance's own `instance_identity`
    /// (`instances::judge_construction`'s own doc) — this constructor
    /// does not itself decide WHICH values are identity-comparable;
    /// `known_dict_key`'s own identity arm decides that, and this is
    /// just the spelling it wraps the tag in.
    fn identity(tag: &str) -> DictKey {
        DictKey {
            name: format!("{IDENTITY_KEY_PREFIX}{tag}"),
            numeric: false,
        }
    }
}

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

/// An IDENTITY-KEYED value's own tag, if `value` is a `Kind::Object`
/// value this file recognizes as identity-comparable rather than
/// value-comparable. Two shapes:
/// - an opaque object (`kind_word` is `Some`, the `opaque_value` shape —
///   a featureless `object()`, `builtin_models.rs`) carrying a non-empty
///   `source`, tagged with that source text.
/// - a constructed CLASS instance (`instances::judge_construction`'s own
///   `instance_identity`, `Some(id)`) — a PER-CONSTRUCTION id, distinct
///   from `source` (which carries the CLASS's name and is shared by
///   every instance of that class): reading `source` here would wrongly
///   treat every `Holder()` instance as the same dict key, so this reads
///   `instance_identity` instead, tagged under a `#`-prefixed spelling no
///   opaque value's own `source` text can produce (`object_call`'s fixed
///   `"object()"` tag has no leading `#`, and no other opaque tag in this
///   checker does either).
fn identity_key_tag(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Object {
        return None;
    }
    if let Some(id) = value.instance_identity {
        return Some(format!("#{id}"));
    }
    if value.kind_word.is_some() && !value.source.is_empty() {
        return Some(value.source.clone());
    }
    None
}

/// An already-evaluated subscript/read index, read as a dict key: a
/// known exact String reads as an ordinary string key (`numeric:
/// false`), a known single Integer-sorted value reads as an int key
/// (`numeric: true`, `DictKey::integer`'s own plain-decimal spelling),
/// or a recognized IDENTITY value (`identity_key_tag`'s own doc) reads
/// as an identity key (`DictKey::identity`) — matched by provenance, the
/// same way `stdtypes.rst`'s mapping rule admits any hashable value,
/// never a string/number requirement. These are the three key sorts
/// `dict_literal_value` accepts, so a `d[15]` subscript read matches the
/// exact entry `{15: ...}` built, and `d[sentinel]` matches
/// `{sentinel: ...}`. Boolean-sorted values are NOT accepted here,
/// matching `known_integer_index`'s own scope note (no row in this
/// file's corpus band needs `d[True]`). Any other shape (unknown, Float,
/// String not exact, an ordinary dict/list/class-instance object)
/// answers `None`. Public: `expressions.rs::evaluate_dict` reuses this
/// same key reading on the CONSTRUCTION side, so a dict literal's
/// identity/Integer/String keys are read the identical way whichever
/// side (build or subscript) reads them.
pub fn known_dict_key(value: &AbstractValue) -> Option<DictKey> {
    if let Some(text) = known_string_key(value) {
        return Some(DictKey::string(&text));
    }
    if value.kind == Kind::Values && value.values.len() == 1 && value.kind_tag == Some(PrimitiveKind::Integer) {
        return Some(DictKey::integer(value.values[0] as i64));
    }
    if let Some(tag) = identity_key_tag(value) {
        return Some(DictKey::identity(&tag));
    }
    None
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
    Some(known_values(vec![values[adjusted as usize]], PrimitiveKind::String, TrustProved))
}

/// `container[key]` on a known DICT receiver (`Kind::Object`) with a
/// known string OR int key: the value at that key's `ObjectKey` entry
/// — matched by BOTH `name` and `numeric` (a string key and an int key
/// of the same spelling are different entries, `ObjectKey`'s own doc)
/// — or `None` if no entry carries that identity. `d[key]` raises
/// `KeyError` on a miss (library/stdtypes.rst, dict's `d[key]` row),
/// which this domain has no channel for this wave, matching the
/// list/tuple out-of-range row's same honesty.
fn dict_key_read(keys: &[ObjectKey], key: &DictKey) -> Option<AbstractValue> {
    keys.iter()
        .find(|entry| entry.name == key.name && entry.numeric == key.numeric)
        .map(|entry| entry.value.clone())
}

/// The kernel state a SCALAR knowledge value denotes — narrowed to the
/// two shapes `dict_key_set_read`'s fold ever hands it: an untagged
/// numeric-or-other `Kind::Values` singleton set, or a plain `Kind::Set`
/// (`set_kind_tag == SetKindTag::None`). Mirrors
/// `lattice_conformance.rs`'s own `state_of_known`, cut down to the
/// scalar-set rows this call site can produce — no Undef/Null/NaN/
/// wrapper arm, since a dict entry's own value never reaches this fold
/// in one of those shapes without already having declined earlier
/// (`dict_key_read` hands back the entry's `AbstractValue` verbatim,
/// and `word_tuples_of`'s gate above only ever supplies exact-string
/// keys, never an absent/NaN VALUE). `set_of_known` is the existing
/// tuple-layer reader every other kernel-asking row in this crate
/// already uses to reach a `RefinedSet`.
fn known_state_of(value: &AbstractValue) -> Option<KnownStateWire> {
    if value.kind != Kind::Values && value.kind != Kind::Set {
        return None;
    }
    if value.kind == Kind::Set && value.set_kind_tag != SetKindTag::None {
        return None; // a worn set (bigint/symbol) carries no ℝ̄ member — set_of_known refuses it too
    }
    let set = set_of_known(value)?;
    Some(KnownStateWire { top: false, set, undef: false, null: false, nan: false, thrown: false })
}

/// A right-fold `Form::Union` tree whose every leaf is a singleton
/// scalar `OneOf` (never a multi-codepoint string tuple, never a bare
/// range/star/etc): the exact values it admits, in no particular
/// order. This is the shape the kernel's `join_state` answers for two
/// (or, folded further, more) distinct scalar values — `{40} ∪ {41}` —
/// the same set `join_known`'s own untagged-numeric arm spells
/// (`lattice_operations.rs`'s `known_set(make_refined_set(vec![union(left,
/// right)]), ...)` tail) before this file's fold hands it to the
/// kernel. A bare (non-union) singleton also qualifies — `word_of`
/// alone reads it — so a one-member fold (no join ever ran) still
/// converts. Any leaf that is not a length-one `word_of` result (a
/// string tuple, a window, a star) fails the whole recognition: the
/// caller must keep the `Kind::Set` form rather than guess at values
/// that are not actually enumerated.
///
/// `pub(crate)`: `match_arms.rs`'s `MatchValue`/`MatchOr` pattern
/// outcome reuses this exact reading for a `Kind::Set` match subject
/// (`case 1:` over a set that enumerates {1, 2, 4}) rather than writing
/// a second set-enumeration parser.
pub(crate) fn scalars_of_union_of_singletons(set: &refined_sets::refinement_forms::RefinedSet) -> Option<Vec<f64>> {
    if let Some(values) = refined_sets::refinement_forms::word_of(set) {
        if values.len() == 1 {
            return Some(values);
        }
        return None;
    }
    if set.forms.len() != 1 || set.forms[0].form != refined_sets::refinement_forms::Form::Union {
        return None;
    }
    let mut values = scalars_of_union_of_singletons(set.forms[0].a_.as_ref().unwrap())?;
    values.extend(scalars_of_union_of_singletons(set.forms[0].b.as_ref().unwrap())?);
    Some(values)
}

/// The reverse of `known_state_of`: a kernel-answered state back to an
/// `AbstractValue`, at the joined trust grade — only for the plain,
/// flag-free state the two scalar-set rows above ever produce or ask
/// about. `top`/`undef`/`null`/`nan`/`thrown` all being unset is the
/// gate: any flag means the answer left the scalar-set world this fold
/// lives in, and the caller falls back to `join_known` rather than
/// misreading a flagged wire as a plain set.
///
/// The kernel's own wire carries no Python sort tag at all
/// (`lattice_conformance.rs`'s module doc), so its answer is always a
/// bare set shape — but when that shape is a union of singleton
/// scalars (`scalars_of_union_of_singletons`), the ANSWER denotes the
/// same exact values `join_known`'s local numeric-tagged arms would
/// have kept as `Kind::Values`: reading it back that way, tagged with
/// `shared_tag` (the caller's own agreement on both operands' Python
/// sort — `Some(Integer)` when both sides were Integer, `Some(Float)`
/// when both were Float, `None` otherwise), recovers the exact-values
/// representation instead of losing it to a poorer `Kind::Set` — every
/// transfer/min/max/sort-law consumer downstream of a kernel-joined
/// dict read gets the richer shape either way. A leaf that is NOT a
/// union of singletons (a range, a star, a multi-codepoint string
/// tuple) stays `Kind::Set` — there are no enumerated values to lift.
fn known_value_of_state(
    state: &KnownStateWire,
    grade: TrustLevel,
    shared_tag: Option<PrimitiveKind>,
) -> Option<AbstractValue> {
    if state.top || state.undef || state.null || state.nan || state.thrown {
        return None;
    }
    if let Some(tag) = shared_tag {
        if let Some(values) = scalars_of_union_of_singletons(&state.set) {
            // The kernel's join keeps both operands' members even when
            // they repeat (`Union({40},{40})`); `join_known`'s own
            // same-sort arms merge with a membership check, so the
            // read-back applies the identical rule.
            let mut merged: Vec<f64> = Vec::with_capacity(values.len());
            for v in values {
                if !merged.iter().any(|kept| *kept == v) {
                    merged.push(v);
                }
            }
            return Some(known_values(merged, tag, grade));
        }
    }
    Some(known_set(state.set.clone(), None, grade, SetKindTag::None))
}

/// The delegate-first fold `dict_key_set_read` folds two dict-entry
/// values through: ask the kernel's proved `join_state`
/// (`kernel_interface.rs`'s `join_state` field, the same entry
/// `lattice_conformance.rs`'s own conformance suite holds
/// `refined_domain::lattice_operations::join_known` to) when both sides
/// convert to a scalar kernel state, and use ITS answer over the local
/// `join_known`. `catch_unwind` turns a kernel panic into a refusal
/// rather than a crash — the same discipline `assignability.rs`/
/// `builtin_models.rs` already hold every kernel ask to. On any
/// refusal — no loaded kernel, a non-convertible operand shape, a
/// flagged answer, or a caught panic — `join_known` runs unchanged as
/// the fallback; this function never weakens what `join_known` alone
/// already proves.
///
/// `shared_tag_of` is the same rule `lattice_operations.rs`'s own
/// same-sorted `join_known` arms follow (finding this fold must not
/// special-case): both operands `Kind::Values` tagged the SAME
/// Integer-or-Float sort keeps that tag; anything else (mixed sorts, a
/// non-Values operand, an already-Set operand) states no shared sort,
/// and `known_value_of_state` then keeps the untagged `Kind::Set`
/// form — Integer ⊔ Float (or Integer ⊔ an unrelated set) is the bare
/// "Number" reading, never one side's tag winning by omission.
fn shared_tag_of(a: &AbstractValue, b: &AbstractValue) -> Option<PrimitiveKind> {
    if a.kind != Kind::Values || b.kind != Kind::Values {
        return None;
    }
    match (a.kind_tag, b.kind_tag) {
        (Some(PrimitiveKind::Integer), Some(PrimitiveKind::Integer)) => Some(PrimitiveKind::Integer),
        (Some(PrimitiveKind::Float), Some(PrimitiveKind::Float)) => Some(PrimitiveKind::Float),
        _ => None,
    }
}

fn kernel_joined_set(so_far: AbstractValue, found: AbstractValue) -> AbstractValue {
    let fallback = || join_known(so_far.clone(), found.clone());
    let Some(kernel) = kernel_if_loaded() else {
        return fallback();
    };
    let Some(state_a) = known_state_of(&so_far) else {
        return fallback();
    };
    let Some(state_b) = known_state_of(&found) else {
        return fallback();
    };
    let asked = crate::kernel_ask::ask_kernel(|| (kernel.join_state)(&state_a, &state_b));
    let Ok(joined_state) = asked else {
        return fallback();
    };
    let grade = min_trust_level(trust_level_of(&so_far), trust_level_of(&found));
    let shared_tag = shared_tag_of(&so_far, &found);
    match known_value_of_state(&joined_state, grade, shared_tag) {
        Some(value) => value,
        None => fallback(),
    }
}

/// `container[key]` on a known DICT receiver with a key that is a
/// FINITE UNION of known exact strings (`key = "age" if flag else
/// "years"`'s own joined shape, `Kind::Set` — `lattice_operations
/// ::join_known` of two distinct multi-codepoint exact strings builds
/// exactly this union-of-`string_tuple` form, per that function's own
/// tests). `stdtypes.rst`'s mapping-subscription rule reads a single
/// key; this is the SOUND generalization when every branch's own key
/// names a PRESENT entry: `person[key]` with `key` known to be `"age"`
/// OR `"years"`, and both `person["age"]` and `person["years"]` present,
/// answers the join of the two entries' own values — exactly the value
/// the real subscription reads on whichever branch actually ran.  A key
/// naming any string not present in `keys` declines the whole read
/// (`None`, never a partial/guessed answer) — the same honesty a single
/// missing key already gives `dict_key_read`. `word_tuples_of` is the
/// existing exact-word enumerator `refined_sets::codepoint_sets` already
/// proves against a union-of-`string_tuple` set (the string-equality
/// narrowing rows use the identical reader); a set that is not this
/// union-of-known-words shape (an unbounded range, an unrelated form)
/// answers `None` from `word_tuples_of` itself, and this function
/// declines in step. The fold asks the kernel's `join_state` first
/// (`kernel_joined_set`) when both accumulated values are scalar-set
/// shaped, falling back to the local `join_known` otherwise — see that
/// function's own doc.
fn dict_key_set_read(keys: &[ObjectKey], index: &AbstractValue) -> Option<AbstractValue> {
    if index.kind != Kind::Set || index.kind_tag.is_some_and(|tag| tag != PrimitiveKind::String) {
        return None;
    }
    let words = refined_sets::codepoint_sets::word_tuples_of(&index.set)?;
    if words.is_empty() {
        return None;
    }
    let mut joined: Option<AbstractValue> = None;
    for points in words {
        let text: String = points.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        let found = dict_key_read(keys, &DictKey::string(&text))?;
        joined = Some(match joined {
            Some(so_far) => kernel_joined_set(so_far, found),
            None => found,
        });
    }
    joined
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
fn star_element_read(container: &AbstractValue, index: &AbstractValue) -> Option<AbstractValue> {
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
/// (`dict_key_set_read`'s own doc), or an unknown-length sequence whose
/// element set is known (`star_element_read`'s own doc). Every other
/// receiver shape or index/key shape answers `None` — an unknown
/// receiver, a non-Integer index into a list or string, an unsupported
/// key sort into a dict, or a slice — none of those are modeled here
/// and this function declines honestly rather than guessing.
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
        Kind::Set => star_element_read(container, index),
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
/// with a known String- or Integer-sorted key (`known_dict_key`'s own
/// doc) is modeled; every other shape answers `None`.
pub fn dict_get_result(
    container: &AbstractValue,
    key: &AbstractValue,
    default: Option<&AbstractValue>,
) -> Option<AbstractValue> {
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

/// `list[index] = value` — the written-through list, known shapes
/// only: a known `Kind::List` receiver and a known Integer index that
/// (after the same negative-index adjustment `list_index_read` reads
/// by) lands inside the list's current bounds (expressions.rst,
/// "Subscriptions" — item assignment on a sequence follows the same
/// negative-index rule as a read; an index past the end raises
/// `IndexError`, which this domain has no channel for, so it declines
/// rather than silently extending the list the way `append` would).
///
/// Carries `receiver`'s own `kind_word` forward onto the written-through
/// list — a bytes-like receiver (`bytes_models::tagged`'s own species
/// word) stays tagged after a write that mutated its contents, so a
/// SECOND write to the same name still reads which of the three
/// bytes-like write rules applies rather than losing the tag the moment
/// this function rebuilds the list.
pub fn list_with_item(receiver: &AbstractValue, index: &AbstractValue, value: &AbstractValue) -> Option<AbstractValue> {
    if receiver.kind != Kind::List {
        return None;
    }
    let position = known_integer_index(index)?;
    let length = receiver.items.len() as i64;
    let adjusted = if position < 0 { position + length } else { position };
    if adjusted < 0 || adjusted >= length {
        return None;
    }
    let mut items = receiver.items.clone();
    items[adjusted as usize] = value.clone();
    let mut written = list_literal_value(&items);
    written.kind_word = receiver.kind_word;
    Some(written)
}

/// A mutating container-method call's (new receiver, call result) pair
/// — the walk's own write channel: `check.rs`/`loops.rs` write the
/// returned receiver back into the environment binding the method was
/// called on, and use the call result the same way any other
/// expression value is used. `None` means "not modeled" (the call is
/// silently NOT threaded as a write — the caller must not assume the
/// receiver is unchanged, matching every other decline in this file);
/// every row below requires the receiver AND every argument fully
/// known, per the mission's own scope — a receiver or argument this
/// file cannot read never answers a guessed write.
///
/// Modeled, each cited against library/stdtypes.rst's own method
/// entry:
/// - list: `append(x)` ("appends *x* to the end of the sequence"),
///   `extend(t)` ("extends *s* with the contents of *t*"),
///   `insert(i, x)` ("inserts *x* into *s* at the index given by *i*"
///   — clamped to `[0, len]`, matching `list.insert`'s own
///   out-of-range-index clamping rather than `IndexError`), `pop()`/
///   `pop(i)` ("retrieves the item at *i* and also removes it from
///   *s*" — no-arg defaults to the LAST item), `clear()` ("removes all
///   items from *s*"), `remove(x)` ("removes the first item from *s*
///   where `s[i]` is equal to *x*" — an ABSENT element declines rather
///   than mutate on the real call's `ValueError`), `sort()` (ascending,
///   known single-numeric elements only), `reverse()` (in place).
/// - set: `add(elem)` ("Add element *elem* to the set"), `discard(elem)`
///   ("Remove element *elem* from the set if it is present" — silent
///   no-op on a miss), `remove(elem)` ("Remove element *elem* from the
///   set. Raises `KeyError` if *elem* is not contained in the set" —
///   an ABSENT element declines the whole call, since the real call
///   raises rather than mutates; `provable_raise` is the raise
///   channel), `update(other)` ("Update the set, adding elements from
///   all others" — the two-arg union-in-place, skipping a duplicate),
///   `clear()`.
/// - dict: `update(other)` ("Update the dictionary with the key/value
///   pairs from *other*, overwriting existing keys" — merges a known
///   dict argument entry by entry), `clear()`, `setdefault(key,
///   default=None)` ("If *key* is in the dictionary, return its
///   value. If not, insert *key* with a value of *default* and return
///   *default*" — the ONE row whose receiver AND call result both
///   change: an absent key both extends the receiver and answers
///   `default`), `pop(key)`/`pop(key, default)` ("If *key* is in the
///   dictionary, remove it and return its value, else return
///   *default*. If *default* is not given and *key* is not in the
///   dictionary, a `KeyError` is raised" — a missing key with no
///   default declines the whole call, matching `set.remove`'s same
///   raise-not-mutate honesty), `popitem()` ("Remove and return a
///   `(key, value)` pair... in LIFO order" — the LAST inserted entry).
///
/// `list.sort()` (no `key=`/`reverse=` keyword arguments) sorts a known
/// list of known single-numeric elements ascending, the same order
/// `builtin_models::sorted_call` already reads for the free function —
/// `list.sort(*, key=None, reverse=False)`: "This method sorts the list
/// in place, using only `<` comparisons between items" (stdtypes.rst).
/// `list.reverse()` reverses a known list's elements in place —
/// stdtypes.rst's Mutable-Sequence-Types table, `s.reverse()`:
/// "reverses the items of *s* in place." Both answer `null_value()` as
/// the call result (neither method returns a value). `list`/`set` share
/// the identical `Kind::List` receiver shape (this file's own module
/// doc), so `add`/`discard`/`remove`/`update` on a plain-list receiver
/// also answer through the same rows — this domain has no separate set
/// Kind to gate that on, and the method NAME is the only signal that a
/// call is set-shaped.
pub fn mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
    match receiver.kind {
        Kind::List => list_mutated_receiver(method, receiver, arguments),
        Kind::Object => dict_mutated_receiver(method, receiver, arguments),
        _ => None,
    }
}

/// `list.append`/`extend`/`insert`/`pop`/`clear`, PLUS the set-only
/// method names `add`/`discard`/`remove`/`update` — see
/// `mutated_receiver`'s own doc for the cited row-by-row contract. Both
/// families dispatch through this one function because a set and a
/// list share the identical `Kind::List` receiver shape in this domain
/// (this file's own module doc) — there is no separate set Kind to
/// route on, so the METHOD NAME alone tells a set call apart from a
/// list call, and both live in the same match.
fn list_mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
    match method {
        "append" => {
            let [element] = arguments else { return None };
            let mut items = receiver.items.clone();
            items.push(element.clone());
            Some((list_literal_value(&items), null_value()))
        }
        "extend" | "update" => {
            let [other] = arguments else { return None };
            if other.kind != Kind::List {
                return None;
            }
            let mut items = receiver.items.clone();
            for candidate in &other.items {
                // `update`'s own set-union-in-place semantics skip a
                // duplicate; `extend`'s own list semantics do not — the
                // method name itself decides which rule applies
                if method == "update" && element_contains(&items, candidate)? {
                    continue;
                }
                items.push(candidate.clone());
            }
            Some((list_literal_value(&items), null_value()))
        }
        "insert" => {
            let [index, element] = arguments else { return None };
            let position = known_integer_index(index)?;
            let length = receiver.items.len() as i64;
            // out-of-range clamps to the nearest end rather than raising
            // (stdtypes.rst's `s.insert(i, x)` row states no bounds check,
            // matching CPython's own clamp-not-raise behavior)
            let clamped = if position < 0 {
                (length + position).max(0)
            } else {
                position.min(length)
            } as usize;
            let mut items = receiver.items.clone();
            items.insert(clamped, element.clone());
            Some((list_literal_value(&items), null_value()))
        }
        "pop" if arguments.is_empty() => {
            let popped = receiver.items.last().cloned()?;
            let mut items = receiver.items.clone();
            items.pop();
            Some((list_literal_value(&items), popped))
        }
        "pop" => {
            let [index] = arguments else { return None };
            let position = known_integer_index(index)?;
            let popped = list_index_read(&receiver.items, position)?;
            let length = receiver.items.len() as i64;
            let adjusted = if position < 0 { position + length } else { position } as usize;
            let mut items = receiver.items.clone();
            items.remove(adjusted);
            Some((list_literal_value(&items), popped))
        }
        "clear" if arguments.is_empty() => Some((list_literal_value(&[]), null_value())),
        // set.add(elem) — "Add element *elem* to the set." A duplicate
        // (already-present) element is a silent no-op (set membership,
        // not list append).
        "add" => {
            let [element] = arguments else { return None };
            if element_contains(&receiver.items, element)? {
                return Some((receiver.clone(), null_value()));
            }
            let mut items = receiver.items.clone();
            items.push(element.clone());
            Some((list_literal_value(&items), null_value()))
        }
        // set.discard(elem) — "Remove element *elem* from the set if it
        // is present." A MISSING element is a silent no-op (unlike
        // `remove`, which raises on a miss). `remove_first_element`
        // removes only the FIRST match, which is exactly "the one
        // occurrence" for a set (no duplicates by construction) and
        // also the correct `list.remove`/`list.discard`-shaped
        // behavior if this receiver happens to be a plain list with
        // duplicate elements.
        "discard" => {
            let [element] = arguments else { return None };
            let items = remove_first_element(&receiver.items, element)?;
            Some((list_literal_value(&items), null_value()))
        }
        // `list.remove(x)`/`set.remove(elem)` — stdtypes.rst's
        // Mutable-Sequence-Types table: "removes the first item from
        // *s* where `s[i]` is equal to *x*"; the set section: "Remove
        // element *elem* from the set. Raises KeyError if *elem* is not
        // contained in the set." An ABSENT element declines the whole
        // call rather than mutate on a raise (`provable_raise` is the
        // raise channel, not this function) — sound for BOTH receiver
        // shapes, since a list `.remove` on a missing element raises
        // `ValueError` the same way a set `.remove` raises `KeyError`.
        "remove" => {
            let [element] = arguments else { return None };
            if !element_contains(&receiver.items, element)? {
                return None;
            }
            let items = remove_first_element(&receiver.items, element)?;
            Some((list_literal_value(&items), null_value()))
        }
        // `list.sort(*, key=None, reverse=False)` — the no-keyword-argument
        // default row: ascending order, only `<` comparisons over known
        // single-numeric elements (stdtypes.rst's own method entry).
        "sort" if arguments.is_empty() => {
            let sorted_items = sorted_numeric_items(&receiver.items)?;
            Some((list_literal_value(&sorted_items), null_value()))
        }
        // `list.reverse()` — "reverses the items of *s* in place"
        // (stdtypes.rst's Mutable-Sequence-Types table, `s.reverse()`).
        "reverse" if arguments.is_empty() => {
            let mut items = receiver.items.clone();
            items.reverse();
            Some((list_literal_value(&items), null_value()))
        }
        _ => None,
    }
}

/// `items` sorted ascending by numeric value, or `None` the moment one
/// element is not a single known Integer/Float/Boolean-sorted value —
/// the same "known numeric elements only" acceptance
/// `builtin_models::sorted_call` reads for the free `sorted()` function,
/// repeated here rather than reaching across the crate boundary for one
/// small helper (this file owns no dependency on `builtin_models.rs`).
fn sorted_numeric_items(items: &[AbstractValue]) -> Option<Vec<AbstractValue>> {
    let mut pairs: Vec<(f64, AbstractValue)> = Vec::with_capacity(items.len());
    for element in items {
        if element.kind != Kind::Values {
            return None;
        }
        if element.values.len() != 1 {
            return None;
        }
        if !matches!(
            element.kind_tag,
            Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean)
        ) {
            return None;
        }
        pairs.push((element.values[0], element.clone()));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("known numeric values are never NaN"));
    Some(pairs.into_iter().map(|(_, value)| value).collect())
}

/// Whether `needle` is a member of `items` by exact-value equality —
/// scalar values (`Kind::Values`) compare by their `values`/`kind_tag`
/// pair; every other shape declines (`None`) rather than guess at
/// equality for a shape this file has no comparison row for. This is
/// the SAME membership question `expressions.rs`'s own `set_contains`
/// answers for the read-side set methods, kept as a separate small copy
/// here rather than reaching across the module boundary for one helper
/// (this file owns no dependency on `expressions.rs`, and adding one
/// would invert the existing `expressions.rs -> collection_models.rs`
/// direction into a cycle).
fn element_contains(items: &[AbstractValue], needle: &AbstractValue) -> Option<bool> {
    // an EMPTY collection contains nothing, regardless of the needle's
    // own shape — this is trivially true by the definition of
    // membership, so a needle this file otherwise cannot compare
    // equality for (e.g. a class instance, `Kind::Object`) still
    // answers `false` against an empty receiver rather than declining
    // (weakref.WeakSet's own `bag.add(key)` on a freshly-built empty
    // set, `expressions.rs`'s corpus this function serves).
    if items.is_empty() {
        return Some(false);
    }
    if needle.kind != Kind::Values {
        return None;
    }
    for element in items {
        if element.kind != Kind::Values {
            return None;
        }
        if element.kind_tag != needle.kind_tag {
            continue;
        }
        if element.values == needle.values {
            return Some(true);
        }
    }
    Some(false)
}

/// `items` with the FIRST element EQUAL to `needle` removed — correct
/// for a set (there is at most one match, no duplicates by
/// construction) and for a plain list's own `.remove`/`.discard`
/// semantics ("removes the first item... where `s[i]` is equal to
/// *x*," stdtypes.rst's Mutable-Sequence-Types table). `None` the
/// moment `element_contains`'s own equality question cannot be decided
/// for some element scanned before the match.
fn remove_first_element(items: &[AbstractValue], needle: &AbstractValue) -> Option<Vec<AbstractValue>> {
    if needle.kind != Kind::Values {
        return None;
    }
    let mut kept = Vec::with_capacity(items.len());
    let mut removed_one = false;
    for element in items {
        if element.kind != Kind::Values {
            return None;
        }
        if !removed_one && element.kind_tag == needle.kind_tag && element.values == needle.values {
            removed_one = true;
            continue;
        }
        kept.push(element.clone());
    }
    Some(kept)
}

/// `dict.update`/`clear`/`setdefault`/`pop`/`popitem` — see
/// `mutated_receiver`'s own doc for the cited row-by-row contract.
fn dict_mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
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
    known_values(code_points, PrimitiveKind::String, TrustProved)
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
    Some(known_values(vec![parsed as f64], PrimitiveKind::Integer, TrustProved))
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

    fn key(text: &str) -> DictKey {
        DictKey::string(text)
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
            &[Some(key("a")), Some(key("b"))],
            &[integer(1.0), integer(2.0)],
        );
        assert_eq!(built.kind, Kind::Object);
        assert_eq!(subscript_read(&built, &string("a")), Some(integer(1.0)));
        assert_eq!(subscript_read(&built, &string("b")), Some(integer(2.0)));
    }

    #[test]
    fn dict_literal_with_a_computed_key_answers_unknown() {
        let built = dict_literal_value(&[None, Some(key("b"))], &[integer(1.0), integer(2.0)]);
        assert_eq!(built.kind, Kind::Unknown);
    }

    #[test]
    fn dict_literal_keeps_the_last_value_for_a_repeated_key() {
        let built = dict_literal_value(
            &[Some(key("a")), Some(key("a"))],
            &[integer(1.0), integer(2.0)],
        );
        assert_eq!(built.keys.len(), 1);
        assert_eq!(subscript_read(&built, &string("a")), Some(integer(2.0)));
    }

    #[test]
    fn dict_literal_int_key_reads_by_int_subscript() {
        // {15: 115} — the a-statements.py dict_comprehension row's own
        // shape: a known Integer key builds a numeric ObjectKey, and a
        // matching Integer subscript reads it back.
        let built = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
        assert_eq!(built.keys.len(), 1);
        assert_eq!(built.keys[0].name, "15");
        assert!(built.keys[0].numeric);
        assert_eq!(subscript_read(&built, &integer(15.0)), Some(integer(115.0)));
    }

    #[test]
    fn dict_literal_int_key_and_string_key_of_the_same_spelling_do_not_collide() {
        // {"15": 1, 15: 2} — CPython holds BOTH entries (1 == "15" is
        // False; only values that compare equal, like 1/1.0/True, share
        // one dict slot, stdtypes.rst's own Mapping Types note).
        let built = dict_literal_value(
            &[Some(key("15")), Some(DictKey::integer(15))],
            &[integer(1.0), integer(2.0)],
        );
        assert_eq!(built.keys.len(), 2);
        assert_eq!(subscript_read(&built, &string("15")), Some(integer(1.0)));
        assert_eq!(subscript_read(&built, &integer(15.0)), Some(integer(2.0)));
    }

    // --- identity-keyed dict entries (an object() sentinel key) ---

    fn identity_sentinel(tag: &str) -> AbstractValue {
        let mut instance = refined_domain::abstract_value::opaque_value("a featureless object");
        instance.source = tag.to_owned();
        instance
    }

    #[test]
    fn dict_literal_identity_key_reads_back_by_the_same_sentinel() {
        let sentinel = identity_sentinel("object()");
        let built = dict_literal_value(&[Some(DictKey::identity("object()"))], &[integer(40.0)]);
        assert_eq!(subscript_read(&built, &sentinel), Some(integer(40.0)));
    }

    #[test]
    fn dict_get_result_identity_key_present_answers_the_stored_value() {
        let sentinel = identity_sentinel("object()");
        let built = dict_literal_value(&[Some(DictKey::identity("object()"))], &[integer(40.0)]);
        assert_eq!(dict_get_result(&built, &sentinel, None), Some(integer(40.0)));
    }

    #[test]
    fn dict_get_result_identity_key_absent_answers_none_value() {
        // a sentinel that was never inserted answers None, not the
        // stored entry for a DIFFERENT sentinel's own tag
        let stored = identity_sentinel("object()");
        let other = identity_sentinel("a different opaque value");
        let built = dict_literal_value(&[Some(DictKey::identity("object()"))], &[integer(40.0)]);
        assert_eq!(dict_get_result(&built, &stored, None), Some(integer(40.0)));
        assert_eq!(dict_get_result(&built, &other, None), Some(null_value()));
    }

    #[test]
    fn dict_with_item_identity_key_round_trips_through_get() {
        let sentinel = identity_sentinel("object()");
        let empty = known_object(vec![], None, true, TrustProved, false);
        let written = dict_with_item(&empty, &sentinel, &integer(200.0)).expect("identity-keyed write must decide");
        assert_eq!(dict_get_result(&written, &sentinel, None), Some(integer(200.0)));
    }

    #[test]
    fn known_dict_key_ignores_a_class_instances_source_tag_with_no_instance_identity() {
        // a constructed class instance with `source` set but no
        // `instance_identity` (a hand-built instance this test never ran
        // through `judge_construction`) is NOT an opaque value (no
        // kind_word) and carries no per-construction id — reading its
        // shared `source` as an identity tag would wrongly treat every
        // instance of the SAME class as one shared dict key, so
        // known_dict_key must decline here rather than build a
        // DictKey::identity from it.
        let mut class_instance = known_object(vec![], None, true, TrustProved, false);
        class_instance.source = "Holder".to_owned();
        assert_eq!(known_dict_key(&class_instance), None);
    }

    // --- identity-keyed dict entries (a class instance's own
    // instance_identity, `instances::judge_construction`'s own tag) ---

    /// Two class instances that share the SAME `source` (both `Holder`)
    /// but carry DIFFERENT `instance_identity` ids — the shape
    /// `judge_construction` builds for two separate `Holder()` calls.
    fn class_instance(class_name: &str, identity: u32) -> AbstractValue {
        let mut instance = known_object(vec![], None, true, TrustProved, false);
        instance.source = class_name.to_owned();
        instance.instance_identity = Some(identity);
        instance
    }

    #[test]
    fn known_dict_key_reads_a_class_instances_own_instance_identity() {
        let a = class_instance("Holder", 1);
        let b = class_instance("Holder", 2);
        assert_ne!(known_dict_key(&a), known_dict_key(&b), "two distinct constructions must not share a key");
    }

    #[test]
    fn dict_get_result_finds_the_exact_instance_a_key_was_inserted_with() {
        // cache[key] = 40; cache.get(key) must read 40 back; a DIFFERENT
        // instance of the same class (missing_key) must miss — the
        // WeakKeyDictionary.get row this table exists to serve.
        let key = class_instance("Holder", 1);
        let missing_key = class_instance("Holder", 2);
        let dict_key = known_dict_key(&key).expect("a constructed instance is an identity key");
        let built = dict_literal_value(&[Some(dict_key)], &[integer(40.0)]);
        assert_eq!(dict_get_result(&built, &key, None), Some(integer(40.0)));
        assert_eq!(dict_get_result(&built, &missing_key, None), Some(null_value()));
    }

    #[test]
    fn dict_with_item_class_instance_key_round_trips_through_get() {
        let key = class_instance("Holder", 7);
        let empty = known_object(vec![], None, true, TrustProved, false);
        let written = dict_with_item(&empty, &key, &integer(40.0)).expect("identity-keyed write must decide");
        assert_eq!(dict_get_result(&written, &key, None), Some(integer(40.0)));
    }

    // --- kernel-joined scalar sets read back as Kind::Values ---

    // test_known_value_of_state_reads_a_union_of_singletons_back_as_values
    // pins the conversion itself: a kernel `join_state` answer shaped
    // `{40} ∪ {41}` (a right-fold Union of singleton OneOf forms — the
    // exact shape `KnownState.join` in `known_state.lean` builds, and
    // `wire_decode.rs`'s `union` arm decodes back verbatim, `a_`/`b` in
    // call order) reads back as `Kind::Values{[40, 41], Some(Integer)}`
    // when the caller states a shared Integer tag — the same richer
    // shape `join_known`'s own same-tag arm would have built locally —
    // never the poorer untagged `Kind::Set` the kernel's bare wire
    // would otherwise force.
    #[test]
    fn test_known_value_of_state_reads_a_union_of_singletons_back_as_values() {
        let union_of_singletons = make_refined_set(vec![refined_sets::refinement_forms::union(
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[40.0])]),
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[41.0])]),
        )]);
        let state = KnownStateWire {
            top: false,
            set: union_of_singletons,
            undef: false,
            null: false,
            nan: false,
            thrown: false,
        };
        let got = known_value_of_state(&state, TrustProved, Some(PrimitiveKind::Integer))
            .expect("a flag-free state must convert");
        assert_eq!(got, known_values(vec![40.0, 41.0], PrimitiveKind::Integer, TrustProved));
    }

    // test_known_value_of_state_a_non_singleton_arm_stays_a_set pins the
    // refusal half: a union with ONE arm that is not a singleton scalar
    // (here, an unbounded `atLeast` range) is not an enumerable set of
    // exact values, so the conversion must decline and the caller keeps
    // the plain `Kind::Set` shape — never guessing values that are not
    // actually there.
    #[test]
    fn test_known_value_of_state_a_non_singleton_arm_stays_a_set() {
        let union_with_a_range = make_refined_set(vec![refined_sets::refinement_forms::union(
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[40.0])]),
            make_refined_set(vec![at_least(0.0)]),
        )]);
        let state = KnownStateWire {
            top: false,
            set: union_with_a_range.clone(),
            undef: false,
            null: false,
            nan: false,
            thrown: false,
        };
        let got = known_value_of_state(&state, TrustProved, Some(PrimitiveKind::Integer))
            .expect("a flag-free state must convert");
        assert_eq!(got, known_set(union_with_a_range, None, TrustProved, SetKindTag::None));
    }

    fn joined_string_key(a: &str, b: &str) -> AbstractValue {
        // `key = "age" if flag else "years"`'s own shape: join_known of
        // two distinct multi-codepoint exact strings builds a Kind::Set
        // over the union of their string_tuple forms (lattice_operations
        // ::join_known's own tests pin this exact join path).
        refined_domain::lattice_operations::join_known(string(a), string(b))
    }

    #[test]
    fn subscript_read_joined_string_key_both_present_answers_the_shared_value() {
        // {"age": 40, "years": 40} — both candidate keys map to the SAME
        // value, so the join of the two entries reads exactly 40.
        let built = dict_literal_value(
            &[Some(key("age")), Some(key("years"))],
            &[integer(40.0), integer(40.0)],
        );
        let joined_key = joined_string_key("age", "years");
        assert_eq!(subscript_read(&built, &joined_key), Some(integer(40.0)));
    }

    #[test]
    fn subscript_read_joined_string_key_different_values_answers_their_join() {
        // {"age": 40, "years": 41} — the two candidate keys map to
        // DIFFERENT values, so the read answers the join of both (the
        // value the real subscription reads depends on which branch ran).
        let built = dict_literal_value(
            &[Some(key("age")), Some(key("years"))],
            &[integer(40.0), integer(41.0)],
        );
        let joined_key = joined_string_key("age", "years");
        let got = subscript_read(&built, &joined_key).expect("both candidate keys are present");
        assert_eq!(got, refined_domain::lattice_operations::join_known(integer(40.0), integer(41.0)));
    }

    /// `loaded_kernel` mirrors `assignability.rs`/`builtin_models.rs`'s
    /// own test helper: a missing dylib artifact prints to stderr and
    /// the caller returns early, never failing the run.
    fn loaded_kernel() -> Option<std::sync::Arc<refined_kernel::kernel_interface::RefinedTSKernel>> {
        let path = refined_kernel::kernel_bridge::dylib_path();
        if !refined_kernel::kernel_bridge::kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(refined_kernel::kernel_bridge::load_kernel(&path).expect("load_kernel"))
    }

    #[test]
    fn kernel_joined_set_agrees_with_join_known_on_two_scalar_sets() {
        // The shape `dict_key_set_read`'s fold hands `kernel_joined_set`:
        // two DIFFERENT known-Integer scalar sets, exactly the shape a
        // `{"age": 40, "years": 41}` read against a joined string key
        // builds (subscript_read_joined_string_key_different_values_
        // answers_their_join's own scenario, isolated to the fold step
        // alone). `load_kernel` adopts a process-wide singleton
        // (`kernel_bridge.rs`'s own doc), so `kernel_if_loaded` inside
        // `kernel_joined_set` sees the same instance this test loads.
        //
        // Compared by mutual SET CONTENT, not `AbstractValue::eq`: the
        // kernel's own wire carries no Python sort tag at all
        // (`lattice_conformance.rs`'s module doc), so `kernel_joined_set`
        // always answers a bare `Kind::Set`, while `join_known`'s own
        // same-Integer-tag arm keeps the answer `Kind::Values` tagged
        // Integer — two different SHAPES for the identical set {40, 41},
        // the same reason `lattice_conformance.rs`'s own `same_state`
        // compares by mutual `scalar_subset` rather than `==`.
        let Some(kernel) = loaded_kernel() else { return };
        let via_kernel = kernel_joined_set(integer(40.0), integer(41.0));
        let via_local = join_known(integer(40.0), integer(41.0));
        let kernel_set = set_of_known(&via_kernel).expect("kernel_joined_set answers a set-shaped value");
        let local_set = set_of_known(&via_local).expect("join_known(40, 41) answers a set-shaped value");
        assert!(
            (kernel.scalar_subset)(&kernel_set, &local_set) && (kernel.scalar_subset)(&local_set, &kernel_set),
            "kernel_joined_set(40, 41) = {via_kernel:?}, want the same set content as join_known's {via_local:?}"
        );
    }

    #[test]
    fn kernel_joined_set_falls_back_to_join_known_on_a_non_set_shaped_operand() {
        // An Object-shaped operand converts through neither
        // `known_state_of` gate — the fold must fall back to the local
        // `join_known` rather than misreading `set_of_known`'s own
        // refusal as a kernel answer.
        let object_side = known_object(vec![], None, true, TrustProved, false);
        let via_fallback = kernel_joined_set(object_side.clone(), integer(41.0));
        let via_local = join_known(object_side, integer(41.0));
        assert_eq!(via_fallback, via_local);
    }

    #[test]
    fn subscript_read_joined_string_key_one_candidate_missing_declines() {
        // {"age": 40} only — "years" names no entry, so the whole read
        // declines rather than guessing at the missing branch's value.
        let built = dict_literal_value(&[Some(key("age"))], &[integer(40.0)]);
        let joined_key = joined_string_key("age", "years");
        assert_eq!(subscript_read(&built, &joined_key), None);
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
    fn subscript_read_positive_index_into_exact_string() {
        // word[0] on "banana" — single-character indexing, the
        // c-reads-and-values.py string_index_access row's own shape.
        let word = string("banana");
        assert_eq!(subscript_read(&word, &integer(0.0)), Some(string("b")));
        assert_eq!(subscript_read(&word, &integer(5.0)), Some(string("a")));
    }

    #[test]
    fn subscript_read_negative_index_into_exact_string() {
        // word[-1] selects the last character — the same negative-index
        // adjustment list_index_read already applies.
        let word = string("banana");
        assert_eq!(subscript_read(&word, &integer(-1.0)), Some(string("a")));
        assert_eq!(subscript_read(&word, &integer(-6.0)), Some(string("b")));
    }

    #[test]
    fn subscript_read_out_of_range_string_index_declines() {
        // word[99] — past the end; IndexError at runtime, no value here.
        let word = string("banana");
        assert_eq!(subscript_read(&word, &integer(99.0)), None);
        assert_eq!(subscript_read(&word, &integer(-99.0)), None);
    }

    #[test]
    fn subscript_read_string_key_into_dict() {
        let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
        assert_eq!(subscript_read(&dict, &string("k")), Some(integer(5.0)));
    }

    #[test]
    fn subscript_read_missing_dict_key_declines() {
        let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
        assert_eq!(subscript_read(&dict, &string("missing")), None);
    }

    #[test]
    fn subscript_read_int_key_does_not_match_a_string_index() {
        // an Object receiver keyed numerically stays a dict read — a
        // known-Integer index matches only a numeric ObjectKey, never a
        // string-spelled one, and vice versa
        let dict = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
        assert_eq!(subscript_read(&dict, &string("15")), None);
        assert_eq!(subscript_read(&dict, &integer(15.0)), Some(integer(115.0)));
    }

    // --- unknown-length, known-element-set receivers (the `list[int]`/
    // `set[int]`/`Sequence[int]` parameter seed's own star shape) ---

    /// The star-of-a-set receiver `check.rs::seed_parameters` builds for
    /// a `list[int]` parameter: `Kind::Set` over one bare
    /// `Form::Star(element)`. Any known Integer index reads "some member
    /// of element" — the star's own definition, no bounds check possible
    /// since the length is unstated.
    fn star_of(element: refined_sets::refinement_forms::RefinedSet) -> AbstractValue {
        known_set(
            refined_sets::refinement_forms::make_refined_set(vec![refined_sets::refinement_forms::star(element)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    }

    #[test]
    fn subscript_read_of_a_star_shaped_set_answers_the_element_set_at_any_index() {
        let whole_ints = refined_sets::refinement_forms::make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
        ]);
        let ages = star_of(whole_ints.clone());
        let element_at_zero = subscript_read(&ages, &integer(0.0)).expect("index 0 reads the star's element");
        assert_eq!(element_at_zero.kind, Kind::Set);
        assert_eq!(element_at_zero.set, whole_ints.clone());
        // the length is unstated — a large index reads the SAME element
        // set, never a bounds refusal the way an exact Kind::List would
        let element_at_large_index =
            subscript_read(&ages, &integer(9000.0)).expect("a star has no length to bound against");
        assert_eq!(element_at_large_index.set, whole_ints);
    }

    #[test]
    fn subscript_read_of_a_star_shaped_set_declines_a_non_integer_index() {
        let whole_ints = refined_sets::refinement_forms::make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
        ]);
        let ages = star_of(whole_ints);
        assert_eq!(subscript_read(&ages, &string("0")), None);
    }

    #[test]
    fn subscript_read_of_a_bounded_scalar_set_is_not_read_as_a_star() {
        // an ordinary bound scalar range (not a star) must not fall into
        // the star reader — it declines the same as before this feature
        let bound = known_set(
            refined_sets::refinement_forms::make_refined_set(vec![refined_sets::refinement_forms::at_least(0.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        assert_eq!(subscript_read(&bound, &integer(0.0)), None);
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
            &[Some(key("a")), Some(key("b"))],
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
        let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
        let got = dict_get_result(&dict, &string("k"), None).expect("get(present) must decide");
        assert_eq!(got, integer(5.0));
    }

    #[test]
    fn dict_get_absent_key_with_no_default_answers_null() {
        let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
        let got = dict_get_result(&dict, &string("missing"), None).expect("get(absent) must decide");
        assert_eq!(got.kind, Kind::Null);
    }

    #[test]
    fn dict_get_int_key_answers_its_value() {
        let dict = dict_literal_value(&[Some(DictKey::integer(15))], &[integer(115.0)]);
        let got = dict_get_result(&dict, &integer(15.0), None).expect("get(present int key) must decide");
        assert_eq!(got, integer(115.0));
    }

    #[test]
    fn dict_get_absent_key_with_default_answers_the_default() {
        let dict = dict_literal_value(&[Some(key("k"))], &[integer(5.0)]);
        let fallback = integer(0.0);
        let got = dict_get_result(&dict, &string("missing"), Some(&fallback))
            .expect("get(absent, default) must decide");
        assert_eq!(got, fallback);
    }

    // --- dict_with_item / list_with_item (the written-through container) ---

    #[test]
    fn dict_with_item_overwrites_an_existing_key() {
        let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
        let written = dict_with_item(&dict, &string("a"), &integer(9.0)).expect("write must decide");
        assert_eq!(subscript_read(&written, &string("a")), Some(integer(9.0)));
    }

    #[test]
    fn dict_with_item_appends_a_new_key() {
        let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
        let written = dict_with_item(&dict, &string("b"), &integer(2.0)).expect("write must decide");
        assert_eq!(written.keys.len(), 2);
        assert_eq!(subscript_read(&written, &string("b")), Some(integer(2.0)));
    }

    #[test]
    fn dict_with_item_writes_an_int_key_without_colliding_a_string_key_of_the_same_spelling() {
        let dict = dict_literal_value(&[Some(key("15"))], &[integer(1.0)]);
        let written = dict_with_item(&dict, &integer(15.0), &integer(2.0)).expect("write must decide");
        assert_eq!(written.keys.len(), 2);
        assert_eq!(subscript_read(&written, &string("15")), Some(integer(1.0)));
        assert_eq!(subscript_read(&written, &integer(15.0)), Some(integer(2.0)));
    }

    #[test]
    fn dict_without_item_removes_a_present_key() {
        let dict = dict_literal_value(
            &[Some(key("a")), Some(key("b"))],
            &[integer(1.0), integer(2.0)],
        );
        let written = dict_without_item(&dict, &string("a")).expect("del must decide");
        assert_eq!(written.keys.len(), 1);
        assert_eq!(subscript_read(&written, &string("b")), Some(integer(2.0)));
        assert_eq!(subscript_read(&written, &string("a")), None);
    }

    #[test]
    fn dict_without_item_absent_key_declines() {
        // del on a missing key RAISES KeyError at runtime — this function
        // does not mutate on a raise, matching provable_raise's own
        // absent-key row for a plain subscript read
        let dict = dict_literal_value(&[Some(key("a"))], &[integer(1.0)]);
        assert_eq!(dict_without_item(&dict, &string("missing")), None);
    }

    #[test]
    fn dict_without_item_int_key_does_not_remove_a_string_key_of_the_same_spelling() {
        let dict = dict_literal_value(
            &[Some(key("15")), Some(DictKey::integer(15))],
            &[integer(1.0), integer(2.0)],
        );
        let written = dict_without_item(&dict, &integer(15.0)).expect("del must decide");
        assert_eq!(written.keys.len(), 1);
        assert_eq!(subscript_read(&written, &string("15")), Some(integer(1.0)));
        assert_eq!(subscript_read(&written, &integer(15.0)), None);
    }

    #[test]
    fn dict_without_item_non_dict_receiver_declines() {
        let list = list_literal_value(&[integer(1.0)]);
        assert_eq!(dict_without_item(&list, &string("a")), None);
    }

    #[test]
    fn list_with_item_writes_a_positive_index() {
        let list = list_literal_value(&[integer(1.0), integer(2.0)]);
        let written = list_with_item(&list, &integer(0.0), &integer(9.0)).expect("write must decide");
        assert_eq!(written.items, vec![integer(9.0), integer(2.0)]);
    }

    #[test]
    fn list_with_item_out_of_range_declines() {
        let list = list_literal_value(&[integer(1.0)]);
        assert_eq!(list_with_item(&list, &integer(5.0), &integer(9.0)), None);
    }

    #[test]
    fn list_with_item_carries_the_receivers_kind_word_forward() {
        // a bytes-like receiver's own species word (bytes_models::tagged)
        // must survive a write that mutates its contents — a SECOND write
        // to the same name still needs to read which write rule applies.
        let mut bytes_like = list_literal_value(&[integer(1.0), integer(2.0)]);
        bytes_like.kind_word = Some("a bytearray value");
        let written = list_with_item(&bytes_like, &integer(0.0), &integer(9.0)).expect("write must decide");
        assert_eq!(written.kind_word, Some("a bytearray value"));
    }

    #[test]
    fn list_with_item_on_an_untagged_list_stays_untagged() {
        let list = list_literal_value(&[integer(1.0), integer(2.0)]);
        let written = list_with_item(&list, &integer(0.0), &integer(9.0)).expect("write must decide");
        assert_eq!(written.kind_word, None);
    }

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
}
