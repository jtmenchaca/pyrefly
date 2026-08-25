//! The dict-key machinery: `DictKey`'s own spelling/sort, and the
//! readers that recognize a String, Integer, or IDENTITY key off an
//! already-evaluated `AbstractValue` (`known_dict_key`) or a String
//! literal (`known_string_key`). See `collection_models`'s own module
//! doc for the `ObjectKey.name`/`numeric` identity pair this file's
//! `DictKey` mirrors.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;

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
    pub(super) fn identity(tag: &str) -> DictKey {
        DictKey {
            name: format!("{IDENTITY_KEY_PREFIX}{tag}"),
            numeric: false,
        }
    }
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
