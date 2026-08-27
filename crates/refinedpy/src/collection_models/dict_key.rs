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

/// The reserved prefix a GUARD-RECORDED presence entry's `name` always
/// carries — the same NUL-based uncollidability `IDENTITY_KEY_PREFIX`
/// relies on, wrapped around an INNER key's own spelling rather than a
/// bare tag. A guard (`narrowing::compare::narrow_dict_membership_
/// against_literal_key`'s doc) proves a key present AT THE GUARD, never
/// that it stays present — an intervening mutation, including one
/// inside a callee handed the receiver, can remove it. A WRITE
/// (`d[k] = v`, `setdefault`) actually puts the value there, so a
/// written entry's `name` carries no such prefix. The two provenances
/// share `ObjectKey`'s `(name, numeric)` identity slot, so a reader
/// that must trust only the write's stronger claim (`dict_star_get_
/// result`'s own written-key shortcut) tells them apart by this prefix
/// alone, and a reader that may trust either (the SUBSCRIPT read,
/// which is sound for a guard exactly when nothing could have mutated
/// since) simply reads through it via `DictKey::guard_inner`.
const GUARD_KEY_PREFIX: &str = "\0guard:";

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

    /// A FLOAT key's own spelling, `numeric: true` — the same numeric
    /// slot an int key occupies, because stdtypes.rst's Mapping Types
    /// section states one key identity for both: "Values that compare
    /// equal (such as `1`, `1.0`, and `True`) can be used
    /// interchangeably to index the same dictionary entry." So a float
    /// key that carries a whole number spells the SAME `name` its int
    /// twin does (`DictKey::integer`'s plain decimal), and `d[1.0]`
    /// reads the entry `{1: ...}` built — which is the cited rule, not
    /// an approximation of it.
    ///
    /// Two float values need their own spelling rather than a decimal
    /// one:
    /// - `-0.0` and `0.0` are ONE key. The cited interchangeability rule
    ///   decides it: `-0.0 == 0.0` is `True` in Python, so the two index
    ///   the same entry. Rust's own `{}` of `-0.0` prints `-0`, a
    ///   different string from `0`, so this constructor normalizes a
    ///   zero of either sign to the single spelling `0` — A8.xfer.
    ///   identity's own `zero_is_one_key` row, where `d[-0.0] = 30` then
    ///   `d.get(0.0)` must HIT.
    /// - `NaN` is never a value-equal key to anything, itself included
    ///   (`float("nan") == float("nan")` is `False`), so no two NaN keys
    ///   compare equal and this constructor answers `None` for one —
    ///   there is no value spelling that would make two NaN keys match
    ///   without being wrong. A dict lookup CAN still hit a NaN key by
    ///   the identity fast path CPython's own `lookdict` takes (`is`
    ///   before `==`), but that is an IDENTITY match, and identity is
    ///   `DictKey::identity`'s business, not this constructor's.
    pub(super) fn float(value: f64) -> Option<DictKey> {
        if value.is_nan() {
            return None;
        }
        let normalized = if value == 0.0 { 0.0 } else { value };
        if normalized.fract() == 0.0 && normalized.abs() < 2f64.powi(53) {
            return Some(DictKey::integer(normalized as i64));
        }
        Some(DictKey {
            name: format!("{normalized}"),
            numeric: true,
        })
    }

    /// An IDENTITY key, `numeric: false` — a dict key that is neither a
    /// string nor an int, matched by PROVENANCE rather than by any
    /// value comparison (stdtypes.rst's mapping rule only requires a
    /// key be :term:`hashable`, never a string or number — a bare
    /// `object()` sentinel, hashable by identity alone, is a legal dict
    /// key this way). `tag` is one of three provenances: an opaque
    /// value's source text (today, only `object_call`'s fixed
    /// `"object()"` tag, `builtin_models.rs`), a `#`-prefixed spelling of
    /// a constructed class instance's own `instance_identity`
    /// (`instances::judge_construction`'s own doc, both read by
    /// `identity_key_tag`/`known_dict_key`'s own identity arm off an
    /// already-evaluated VALUE), or a `"binding:<name>"`-prefixed
    /// spelling of an UNWRITTEN PARAMETER/LOCAL binding's own name
    /// (`narrowing::compare::narrow_dict_membership_against_literal_key`'s
    /// own doc — a membership guard's fact when the key expression is a
    /// plain name rather than a literal or a value this domain can read
    /// an identity off, e.g. a weak-referenceable class-instance
    /// parameter). This constructor does not itself decide WHICH values
    /// or bindings are identity-comparable; this is just the spelling
    /// each caller wraps its own tag in.
    pub fn identity(tag: &str) -> DictKey {
        DictKey {
            name: format!("{IDENTITY_KEY_PREFIX}{tag}"),
            numeric: false,
        }
    }

    /// A GUARD-RECORDED presence entry, wrapping `inner`'s own spelling
    /// under the reserved guard prefix (this file's own doc for
    /// `GUARD_KEY_PREFIX`) — `numeric` carries through from `inner`
    /// unchanged, since the guard wraps an existing key identity rather
    /// than inventing a new sort. `narrowing::compare::narrow_dict_
    /// membership_against_literal_key` calls this instead of recording
    /// `inner` bare, so its entries never match the written-key
    /// shortcut a plain WRITE's entry is entitled to.
    pub fn guarded(inner: &DictKey) -> DictKey {
        DictKey {
            name: format!("{GUARD_KEY_PREFIX}{}", inner.name),
            numeric: inner.numeric,
        }
    }

    /// Whether `self` is a guard-recorded entry, and if so, the
    /// UNDERLYING key spelling it wraps (`inner.name`, before the guard
    /// prefix) paired with the same `numeric` the guarded entry carries
    /// — what a reader that MAY still trust a guard's fact (a subscript
    /// read, sound exactly when nothing could have mutated since the
    /// guard) needs to rebuild the plain `DictKey` it should compare
    /// against a freshly-evaluated index. `None` for a write-provenance
    /// or identity-provenance entry, which carry no guard wrapper to
    /// strip.
    pub fn guard_inner(&self) -> Option<DictKey> {
        let stripped = self.name.strip_prefix(GUARD_KEY_PREFIX)?;
        Some(DictKey {
            name: stripped.to_owned(),
            numeric: self.numeric,
        })
    }

    /// How this key reads in a diagnostic sentence naming it — the
    /// `KeyError: <spelling>` detail `known_container_index_absent`
    /// writes. A GUARD-wrapped key reads by its own inner spelling (the
    /// guard wrapper is a provenance marker, not part of what a person
    /// reading the message should see). A string key is quoted the way
    /// CPython's own `KeyError` repr quotes it; a numeric key prints
    /// bare (`KeyError: 15`); an identity key prints its own tag with
    /// the reserved prefix stripped (`object()`, or a constructed
    /// instance's `#<id>`), since the NUL that makes the prefix
    /// uncollidable has no place in a message a person reads.
    pub fn spelling(&self) -> String {
        if let Some(inner) = self.guard_inner() {
            return inner.spelling();
        }
        if let Some(tag) = self.name.strip_prefix(IDENTITY_KEY_PREFIX) {
            return tag.to_owned();
        }
        if self.numeric {
            return self.name.clone();
        }
        format!("'{}'", self.name)
    }
}

/// Whether a recorded entry's own `name` (an `ObjectKey.name` read
/// straight off the domain's `Vec<ObjectKey>`, not wrapped in this
/// file's own `DictKey`) carries the guard prefix — the same test
/// `DictKey::guard_inner` runs, exposed as a free function for a caller
/// that only has the bare `(name, numeric)` pair `ObjectKey` carries and
/// has no `DictKey` to call the method on. Used by `subscript_read::
/// dict_key_read_written` to exclude a guard-provenance entry from the
/// written-key shortcut without first rebuilding a `DictKey`.
pub fn name_is_guarded(name: &str) -> bool {
    name.starts_with(GUARD_KEY_PREFIX)
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
/// (`numeric: true`, `DictKey::integer`'s own plain-decimal spelling), a
/// known single Float-sorted value (`DictKey::float`'s own doc — the
/// same numeric slot, since stdtypes.rst states `1` and `1.0` index one
/// entry; a NaN answers `None` there, keying no entry by value), or a
/// recognized IDENTITY value (`identity_key_tag`'s own doc) reads as an
/// identity key (`DictKey::identity`) — matched by provenance, the same
/// way `stdtypes.rst`'s mapping rule admits any hashable value, never a
/// string/number requirement. These are the key sorts
/// `dict_literal_value` accepts, so a `d[15]` subscript read matches the
/// exact entry `{15: ...}` built, and `d[sentinel]` matches
/// `{sentinel: ...}`. Boolean-sorted values are NOT accepted here,
/// matching `known_integer_index`'s own scope note (no row in this
/// file's corpus band needs `d[True]`). Any other shape (unknown, a
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
    // A known single FLOAT-sorted value — `DictKey::float`'s own doc
    // states the two rules the spelling has to carry (a whole float keys
    // the same entry its int twin does; a zero of either sign is one
    // key), and answers `None` for a NaN, which compares equal to
    // nothing and so keys no entry by value at all.
    if value.kind == Kind::Values && value.values.len() == 1 && value.kind_tag == Some(PrimitiveKind::Float) {
        return DictKey::float(value.values[0]);
    }
    if let Some(tag) = identity_key_tag(value) {
        return Some(DictKey::identity(&tag));
    }
    None
}
