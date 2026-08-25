//! Binary-sequence VALUE states: `bytes`/`bytearray` literals and
//! element reads, `array.array('d', ...)` (the Float64Array twin), and
//! `memoryview` reads/writes over a `bytearray` buffer. A provable
//! runtime raise (an out-of-range index, an out-of-[0,255] write) is
//! spoken through `BytesAnswer::Raises` rather than a new finding
//! category — the walk turns a `Raises` answer into an RTS7001 at the
//! raising expression, the same finding code `assignability.rs`'s
//! `Verdict::Fire` already produces, per the mission's product
//! decision (no new code, no new category).
//!
//! ## How the domain carries a bytes-like value
//!
//! `refined_domain::abstract_value::AbstractValue` has no dedicated
//! bytes/array variant (ORIENTATION.md's `Kind` list has none, and
//! `PYREFLY-NUMERIC-B3-B4.md`'s two-numeric-sorts rule is the only
//! sort split this domain draws). A `bytes`/`bytearray` value is a
//! sequence of known ints in `0..=255`
//! (`Doc/library/stdtypes.rst`, "Bytes and Bytearray Objects" —
//! "Since bytes objects are sequences of integers (akin to a tuple)"),
//! which is exactly the shape `collection_models.rs`'s
//! `list_literal_value`/`tuple_literal_value` already build for a
//! Python sequence literal: `Kind::List` (`known_constructors::known_list`)
//! with one Integer-tagged `known_values` element per slot, in source
//! order. This file reuses that constructor rather than inventing a
//! second "sequence of known ints" shape — `bytes_literal_value` is
//! `list_literal_value` under a different name for the same reason
//! `collection_models.rs`'s own doc gives `tuple_literal_value`: no
//! new `Kind` variant is asked for or needed, and `len()`/indexed reads
//! on the result already work through `collection_models.rs`'s
//! existing `len_result`/`subscript_read` (`Kind::List` is `Kind::List`
//! regardless of which literal built it — this file does not
//! reimplement those two functions).
//!
//! `array.array('d', ...)` (the Float64Array twin) is the same
//! `Kind::List` shape with Float-tagged elements instead of
//! Integer-tagged ones — CPython's `array` module doc,
//! "array.array(typecode, initializer)" with typecode `'d'`, states
//! "double / 8" storage, and every element read back is a Python
//! `float`. Age (this corpus's int-sorted alias) rejects a Float-typed
//! element by the SORT law `assignability.rs`'s `judge` already
//! enforces (`requires_integer`) — reading `array.array('d', ...)`'s
//! elements needs no new judging path, only the Float-tagged
//! `known_values` construction this file supplies.
//!
//! ## Slicing is not indexing
//!
//! `b"ab"[0]` is the int 97; `b"ab"[0:1]` is the one-element bytes
//! object `b"a"` (execution-verified against installed CPython 3.12,
//! `tmp/cpython/` being absent from this checkout — see this file's
//! Blockers note in the owning report). A slice of a known bytes/array
//! `Kind::List` is itself a `Kind::List` built from the same sliced
//! sub-range of elements, never collapsed to a scalar — `bytes_slice`
//! answers that sub-sequence directly rather than routing through
//! `bytes_index`, which only ever answers one element or a raise.
//!
//! ## The provable-raise rows
//!
//! Every raise wording below is execution-verified against installed
//! CPython 3.12 (`tmp/cpython/` gitignored and absent from this
//! checkout, per AGENT-BRIEF's "execution is ground truth where docs
//! are silent" and this mission's own allowance):
//! - `bytes([1, 2])[10]` → `IndexError: index out of range` — the
//!   bytes/bytearray twin of `collection_models.rs`'s own list
//!   out-of-range read, which declines (`None`) rather than fires
//!   because that file's domain has no exception channel; this
//!   mission gives bytes reads that channel via `BytesAnswer::Raises`.
//! - `bytearray(1)[0] = 256` → `ValueError: byte must be in
//!   range(0, 256)` — confirmed for both a too-large value (256) and a
//!   negative value (-1); the same wording either side of the range.
//! - `memoryview(bytearray(2))[0] = 256` → `ValueError: memoryview:
//!   invalid value for format 'B'` — a different wording than
//!   bytearray's own out-of-range write, because the raise is raised
//!   by the `memoryview` C-buffer layer, not by `bytearray.__setitem__`
//!   directly (AGENT-BRIEF: "bytearray/memoryview writes RAISE outside
//!   [0,255] — no wrap or clamp").
//!
//! Fire-message voice matches `assignability.rs`'s `Verdict::Fire`:
//! plain, states the value, states the rule/range that was crossed.

use refined_domain::abstract_value::{known_set, known_values, opaque_value, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::{trust_level_of, TrustProved};
use refined_sets::refinement_forms::{make_refined_set, one_of, repeat_of};

/// What judging one bytes-like operation (a read or a write) against
/// its runtime semantics concluded — the twin of `assignability.rs`'s
/// `Verdict`, scoped to this file's own operations rather than the
/// declared-refinement judging seam (a bytes index/write raise is a
/// LANGUAGE-level fact, not a comparison against a declared set).
pub enum BytesAnswer {
    /// The operation provably produces this value.
    Value(AbstractValue),
    /// The operation provably raises — the full diagnostic text, in
    /// `assignability.rs`'s own voice (plain, names the value, names
    /// the rule).
    Raises(String),
}

/// The species word a bytes-like `Kind::List` value carries on its own
/// `kind_word` — the same "extra species tag riding a shared shape"
/// pattern `env.rs`'s `FUNCTION_VALUE_WORD` already uses on `Kind::Object`
/// (a retained lambda/def is still `Kind::Object`, distinguished only by
/// its word). No existing reader in this file's own package inspects
/// `kind_word` on a `Kind::List` value (every `kind_word` match elsewhere
/// gates on `Kind::Object` first — `assignability.rs`, `collection_
/// models.rs`, `env.rs`), so tagging a list this way is invisible to
/// every read/join/judge path that does not explicitly ask for it, and
/// `known_list`'s own zero value never sets it — an ordinary `list`/
/// `tuple` literal carries `kind_word: None` exactly as before.
///
/// Three words, not one: `bytes` (immutable — ANY element write raises
/// `TypeError`, `bytearray_write_answer`'s sibling below has no row for
/// it), `bytearray` (mutable, `ValueError` outside `0..=255`), and
/// `memoryview` (mutable over a shared buffer, the SAME `0..=255` range
/// but a DIFFERENT `ValueError` wording — `bytes_models.rs`'s own module
/// doc). `bytes_write_answer` below reads whichever of the three a
/// receiver carries to pick the right rule.
pub const BYTES_WORD: &str = "a bytes value";
pub const BYTEARRAY_WORD: &str = "a bytearray value";
pub const MEMORYVIEW_WORD: &str = "a memoryview value";

/// A `bytes`/`bytearray` display built from KNOWN element values
/// (`bytes([10, 20, 30])`, `bytearray(b"\x0a\x14")`'s literal form) —
/// `Kind::List` with one Integer-tagged `known_values` slot per byte,
/// the same shape `collection_models.rs`'s `list_literal_value` builds
/// for any Python sequence literal (module doc: no dedicated
/// bytes/array `Kind` exists or is needed). The caller supplies each
/// element already validated to be `0..=255` — a raw `bytearray(4)`
/// zero-fill or a `bytes([...])`/`bytearray(b"...")` literal walk,
/// never a value this function itself range-checks; `bytes(iterable)`
/// where an element is truly out of `range(0, 256)` raises at
/// CONSTRUCTION time (`ValueError: bytes must be in range(0, 256)`),
/// a fact the caller's own construction-site reading owns, not this
/// literal-encoding step.
pub fn bytes_literal_value(bytes: &[u8]) -> AbstractValue {
    let elements: Vec<AbstractValue> = bytes
        .iter()
        .map(|byte| known_values(vec![*byte as f64], PrimitiveKind::Integer, TrustProved))
        .collect();
    known_list(elements, TrustProved)
}

/// An `array.array('d', [...])` display (the Float64Array twin):
/// `Kind::List` with one Float-tagged `known_values` slot per element —
/// `array.rst`'s typecode `'d'` row, "double / 8", every element
/// always read back as a Python `float`. Reading one of these
/// elements into an int-sorted alias (Age) fires through
/// `assignability.rs`'s existing sort law
/// (`requires_integer`/Float-tagged mismatch); this constructor states
/// nothing about that judging, only the Float tag the sort law reads.
pub fn array_double_literal_value(elements: &[f64]) -> AbstractValue {
    let items: Vec<AbstractValue> = elements
        .iter()
        .map(|value| known_values(vec![*value], PrimitiveKind::Float, TrustProved))
        .collect();
    known_list(items, TrustProved)
}

/// The 0-based (post negative-index-adjustment) integer index an
/// AbstractValue states, if it is a single known Integer-sorted value
/// — the same reading `collection_models.rs`'s own
/// `known_integer_index` performs (repeated here rather than reaching
/// into that file's private helper: a different Rust module in the
/// same crate, but ORIENTATION.md/AGENT-BRIEF.md name this wave as
/// this file only).
fn known_integer_index(index: &AbstractValue) -> Option<i64> {
    if index.kind != Kind::Values || index.values.len() != 1 {
        return None;
    }
    if index.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    Some(index.values[0] as i64)
}

/// `receiver[index]` on a known `bytes`/`bytearray`/`array.array`
/// receiver (`Kind::List`, this file's own literal shape) with a known
/// Integer index: negative indexing adjusts by the sequence's own
/// length first, matching `collection_models.rs`'s list-read rule
/// (`expressions.rst`, "Subscriptions"). An index landing in range
/// answers `Value` with that element; an index still out of range
/// after adjustment RAISES `IndexError: index out of range`
/// (execution-verified against CPython 3.12 — see this file's module
/// doc) — this is where this file's reads diverge from
/// `collection_models.rs`'s own `subscript_read`, which declines
/// rather than fires because that file's domain carries no exception
/// channel; this mission gives bytes-like reads that channel.
///
/// An unknown receiver (not `Kind::List`) or an unknown/non-Integer
/// index answers `None` — not yet readable, never guessed.
pub fn bytes_index(receiver: &AbstractValue, index: &AbstractValue) -> Option<BytesAnswer> {
    if receiver.kind != Kind::List {
        return None;
    }
    let position = known_integer_index(index)?;
    let length = receiver.items.len() as i64;
    let adjusted = if position < 0 { position + length } else { position };
    if adjusted < 0 || adjusted >= length {
        return Some(BytesAnswer::Raises(format!(
            "this read provably raises IndexError: index out of range, the index is {position}"
        )));
    }
    Some(BytesAnswer::Value(receiver.items[adjusted as usize].clone()))
}

/// `receiver[start:stop]` on a known `bytes`/`bytearray`/`array.array`
/// receiver: a SLICE answers a sub-sequence (still `Kind::List`), never
/// a scalar element — `b"ab"[0:1]` is the bytes object `b"a"`, not the
/// int 97 `b"ab"[0]` is (execution-verified, module doc). Both bounds
/// are known, non-negative, and already CLAMPED to the sequence's own
/// length by the caller (Python slicing never raises for an
/// out-of-bounds bound — `expressions.rst`, "Subscriptions": a slice's
/// bounds are silently clamped into range) — this function performs
/// the clamp itself so a caller passing a raw out-of-bounds `stop`
/// still answers the correctly truncated slice rather than panicking
/// on an out-of-bounds `Vec` read.
pub fn bytes_slice(receiver: &AbstractValue, start: i64, stop: i64) -> Option<AbstractValue> {
    if receiver.kind != Kind::List {
        return None;
    }
    let length = receiver.items.len() as i64;
    let clamped_start = start.clamp(0, length) as usize;
    let clamped_stop = stop.clamp(0, length) as usize;
    if clamped_stop <= clamped_start {
        return Some(known_list(Vec::new(), TrustProved));
    }
    Some(known_list(
        receiver.items[clamped_start..clamped_stop].to_vec(),
        TrustProved,
    ))
}

/// The single known Integer value in `0..=255` an AbstractValue
/// states, or `None` if it is not a single known Integer at all
/// (distinct from "known but out of range," which the caller below
/// still needs to see in order to raise on it).
fn known_integer_value(value: &AbstractValue) -> Option<i64> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    if value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    Some(value.values[0] as i64)
}

/// `bytearray[i] = v` — a KNOWN single-Integer `v` in `0..=255`
/// answers `Value` with that same int (the write completes; the
/// element read back is exactly what was written, `stdtypes.rst`'s
/// bytearray `__setitem__` row). A known Integer `v` OUTSIDE
/// `0..=255` (either side — CPython raises identically for 256 and for
/// -1, execution-verified) RAISES `ValueError: byte must be in
/// range(0, 256)` — AGENT-BRIEF: "bytearray/memoryview writes RAISE
/// outside [0,255] — no wrap or clamp." An unknown or non-Integer `v`
/// answers `None`: not yet readable.
pub fn bytearray_write_answer(value: &AbstractValue) -> Option<BytesAnswer> {
    let candidate = known_integer_value(value)?;
    if (0..=255).contains(&candidate) {
        return Some(BytesAnswer::Value(known_values(
            vec![candidate as f64],
            PrimitiveKind::Integer,
            TrustProved,
        )));
    }
    Some(BytesAnswer::Raises(format!(
        "this write provably raises ValueError: byte must be in range(0, 256), the value is {candidate}"
    )))
}

/// `memoryview[i] = v` over a `bytearray`-backed buffer with format
/// `'B'` (unsigned byte, the implicit format a `memoryview(bytearray(...))`
/// always carries) — the same `0..=255` range `bytearray_write_answer`
/// enforces, but CPython raises a DIFFERENT message: `ValueError:
/// memoryview: invalid value for format 'B'`, not bytearray's own
/// "byte must be in range(0, 256)" (execution-verified — the raise
/// comes from the memoryview C-buffer layer's own bounds check, not
/// from `bytearray.__setitem__`, so the wording differs even though
/// the accepted range is identical). An in-range known Integer answers
/// `Value` with that int, matching a direct bytearray write's
/// read-back (the two share the same underlying buffer, module doc).
pub fn memoryview_write_answer(value: &AbstractValue) -> Option<BytesAnswer> {
    let candidate = known_integer_value(value)?;
    if (0..=255).contains(&candidate) {
        return Some(BytesAnswer::Value(known_values(
            vec![candidate as f64],
            PrimitiveKind::Integer,
            TrustProved,
        )));
    }
    Some(BytesAnswer::Raises(
        "this write provably raises ValueError: memoryview: invalid value for format 'B'"
            .to_owned(),
    ))
}

/// Stamps `word` (one of the three constants above) onto a `Kind::List`
/// value's own `kind_word` — a bytes-like construction call's own last
/// step, so the receiver a later write reads carries which of the three
/// write rules applies. A non-`Kind::List` value passes through
/// unchanged (never stamped): this tag means nothing outside the one
/// `Kind` it is read against.
pub fn tagged(value: AbstractValue, word: &'static str) -> AbstractValue {
    if value.kind != Kind::List {
        return value;
    }
    AbstractValue {
        kind_word: Some(word),
        ..value
    }
}

/// `receiver[index] = value` on a receiver carrying one of this file's
/// own three species words (`BYTES_WORD`/`BYTEARRAY_WORD`/`MEMORYVIEW_
/// WORD`, `tagged`'s own doc) — the one write-time dispatch every
/// bytes-like receiver's write goes through, so `check.rs`'s write sink
/// does not itself need to know which of the three rules applies.
///
/// `bytes` is IMMUTABLE: `stdtypes.rst`'s own "Bytes objects are
/// immutable sequences" states there is no `__setitem__` at all, so
/// EVERY write RAISES, regardless of the value (execution-verified:
/// `bytes([1,2])[0] = 1` raises `TypeError: 'bytes' object does not
/// support item assignment` even though 1 is a perfectly in-range
/// byte). `bytearray`/`memoryview` route to their own existing
/// range-checked answer. A receiver with no recognized word (a plain
/// `list`/`tuple`, `kind_word: None`) answers `None`: this function has
/// no rule for it, and the caller's own existing `list_with_item` path
/// is the honest one to keep using.
pub fn bytes_write_answer(receiver: &AbstractValue, value: &AbstractValue) -> Option<BytesAnswer> {
    match receiver.kind_word {
        Some(BYTES_WORD) => Some(BytesAnswer::Raises(
            "this write provably raises TypeError: 'bytes' object does not support item assignment".to_owned(),
        )),
        Some(BYTEARRAY_WORD) => bytearray_write_answer(value),
        Some(MEMORYVIEW_WORD) => memoryview_write_answer(value),
        _ => None,
    }
}

/// The word `base64.b64encode(...)`'s own result carries on `kind_word`
/// — a `Kind::Object` (`opaque_value`'s own shape), the SAME "kind of
/// thing known, contents not" posture `object_call`/`type_as_value`
/// keep in `builtin_models.rs` for a value this file has no concrete
/// bytes elements to enumerate. `.decode()` below reads this word to
/// answer the base64-alphabet STRING grammar exactly, rather than the
/// unbounded `Σ*` a generic bytes-decode would otherwise fall back to.
pub const BASE64_ENCODED_WORD: &str = "a base64-encoded bytes value";

/// The word a plain `str.encode()` result carries — a bytes value with
/// NO further claim than "some bytes," distinct from
/// `BASE64_ENCODED_WORD`'s own narrower alphabet claim so
/// `bytes_decode_call` can tell the two apart (an `.encode()` result's
/// OWN `.decode()` is not modeled — general UTF-8 decoding of arbitrary
/// unread bytes can raise, `bytes_decode_call`'s own doc — only the
/// base64-tagged shape answers a value there).
pub const ENCODED_BYTES_WORD: &str = "an encoded bytes value";

/// `str.encode()` — library/stdtypes.html#str.encode: "Return an
/// encoded version of the string as a bytes object... `encoding`
/// defaults to 'utf-8'." No per-byte encoding table is built; the
/// answer is the opaque "some bytes value" state, `ENCODED_BYTES_WORD`-
/// tagged. `string_models.rs`'s own `.encode()` rows (exact and
/// sort-only receiver) both call this rather than build the tag
/// locally, so the one tag has one definition.
pub fn encoded_bytes_value() -> AbstractValue {
    opaque_value(ENCODED_BYTES_WORD)
}

/// The base64 alphabet, unbounded repetition — `library/base64.rst`,
/// `base64.b64encode`: "Encode the bytes-like object *s* using Base64
/// and return the encoded bytes... the alphabet used... consists of the
/// letters A-Z, a-z, digits 0-9, and the characters + and /... the
/// encoded byte string is... padded with '=' so that its length is
/// always a multiple of 4." Every output byte is drawn from
/// `[A-Za-z0-9+/=]`, with no OTHER byte ever appearing (the pad
/// character `=` only ever appears at the end, but this row states the
/// sound over-approximation — any-position membership in the six-way
/// alphabet — rather than a tighter trailing-pad-only grammar, matching
/// the corpus's own claim, "result is [A-Za-z0-9+/=]*"). Modeled for
/// ANY single argument this file/its caller has already resolved to a
/// bytes-like receiver (a `Kind::List` this file's own literal shape
/// builds, or an opaque bytes-sorted value from `.encode()`'s own
/// sort-only row) — the alphabet claim holds regardless of the input
/// bytes' own content, so the argument's shape is not otherwise
/// inspected once the caller has confirmed it is bytes-like.
fn b64encode_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [_only] = arguments else { return None };
    // the claim holds for any bytes-like receiver; content is not
    // otherwise inspected — the same "no operand-derived grade to
    // thread" posture `object_call`/`type_as_value` keep in
    // builtin_models.rs for their own opaque answers
    Some(opaque_value(BASE64_ENCODED_WORD))
}

/// The dispatcher for a `base64.<function>(...)` module-qualified call
/// — the `base64` twin of `builtin_models::stdlib_call_result`, kept
/// here rather than there since this file already owns every other
/// bytes-shaped answer this crate builds. `b64decode` is NOT modeled:
/// the brief's own scope names `b64encode` only, and no corpus fixture
/// calls `b64decode` (verified against `packages/tests/e2e`).
pub fn base64_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "b64encode" => b64encode_call(arguments),
        _ => None,
    }
}

/// `bytesish.decode()` — library/stdtypes.html#bytes.decode: "Return a
/// string decoded from the given bytes... `encoding` defaults to
/// 'utf-8'." The zero-argument, default-encoding form only. Modeled for
/// ONE receiver shape: a `base64.b64encode(...)`-tagged opaque value
/// (`BASE64_ENCODED_WORD`, `b64encode_call`'s own doc) answers the
/// EXACT base64-alphabet string grammar — `[A-Za-z0-9+/=]*` — since
/// every byte `b64encode` ever produces is ASCII, and decoding an
/// ASCII byte string is the identity mapping onto that same alphabet
/// of `str` characters (no encoding error is possible over an
/// ASCII-only byte string, `Doc/library/codecs.rst`'s own "ascii" is a
/// strict subset of "utf-8"). Any OTHER bytes-like receiver (a known
/// `Kind::List` of concrete byte values, or an untagged unknown bytes
/// value) declines — a general UTF-8 decode of arbitrary bytes can
/// raise `UnicodeDecodeError` on invalid sequences, which this row does
/// not attempt to prove absent for the untagged case.
///
/// One further receiver shape IS decided: a KNOWN `Kind::List` of
/// concrete byte values, decoded with an explicit `"ascii"` encoding.
/// Every byte in `[0, 127]` is its own code point under ASCII
/// (`Doc/library/codecs.rst`'s own "ascii" codec), so a known
/// all-ASCII byte sequence decodes to an EXACT known string. A byte
/// outside `[0, 127]` makes `"ascii"` raise `UnicodeDecodeError`
/// instead — that raise is `expressions`'s own `provable_raise` channel
/// to speak, so this value row declines rather than fabricate a
/// substituted character.
pub fn bytes_decode_call(receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if let Some(value) = ascii_decoded_known_bytes(receiver, arguments) {
        return Some(value);
    }
    if !arguments.is_empty() {
        return None;
    }
    if receiver.kind != Kind::Object || receiver.kind_word != Some(BASE64_ENCODED_WORD) {
        return None;
    }
    let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
    let points: Vec<f64> = alphabet.chars().map(|c| c as u32 as f64).collect();
    let grammar = make_refined_set(vec![repeat_of(make_refined_set(vec![one_of(&points)]), 0, None)]);
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(grammar, None, trust_level_of(receiver), SetKindTag::None)
    })
}

/// `<known bytes>.decode("ascii")` — the exact string a known all-ASCII
/// byte sequence decodes to. `bytes_decode_call`'s own doc states the
/// clause and why a non-ASCII byte declines here.
fn ascii_decoded_known_bytes(receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [encoding] = arguments else { return None };
    if encoding.kind != Kind::Values || encoding.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    let encoding_text: String = encoding.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
    if encoding_text != "ascii" {
        return None;
    }
    if receiver.kind != Kind::List {
        return None;
    }
    let mut code_points: Vec<f64> = Vec::with_capacity(receiver.items.len());
    for element in &receiver.items {
        if element.kind != Kind::Values || element.values.len() != 1 {
            return None;
        }
        let byte = element.values[0];
        if !(0.0..=127.0).contains(&byte) {
            return None; // "ascii" raises UnicodeDecodeError past 0x7F
        }
        code_points.push(byte);
    }
    Some(known_values(code_points, PrimitiveKind::String, TrustProved))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    fn float(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Float, TrustProved)
    }

    fn unwrap_value(answer: BytesAnswer) -> AbstractValue {
        match answer {
            BytesAnswer::Value(value) => value,
            BytesAnswer::Raises(message) => panic!("expected Value, got Raises({message})"),
        }
    }

    fn unwrap_raises(answer: BytesAnswer) -> String {
        match answer {
            BytesAnswer::Value(value) => panic!("expected Raises, got Value({value:?})"),
            BytesAnswer::Raises(message) => message,
        }
    }

    // --- bytes_literal_value ---

    #[test]
    fn bytes_literal_value_round_trips_elements_as_known_integers() {
        // bytes([10, 20, 30]) — p-typed-array.py `bytes_from_iterable`
        let built = bytes_literal_value(&[10, 20, 30]);
        assert_eq!(built.kind, Kind::List);
        assert_eq!(built.items, vec![integer(10.0), integer(20.0), integer(30.0)]);
    }

    #[test]
    fn bytes_literal_value_of_empty_slice_is_an_empty_list() {
        let built = bytes_literal_value(&[]);
        assert_eq!(built.kind, Kind::List);
        assert!(built.items.is_empty());
    }

    // --- array_double_literal_value ---

    #[test]
    fn array_double_literal_value_round_trips_elements_as_known_floats() {
        // array.array('d', [10.0, 20.0, 30.0]) — p-typed-array.py `array_double_from_iterable`
        let built = array_double_literal_value(&[10.0, 20.0, 30.0]);
        assert_eq!(built.kind, Kind::List);
        assert_eq!(built.items, vec![float(10.0), float(20.0), float(30.0)]);
    }

    // --- bytes_index: Value path ---

    #[test]
    fn bytes_index_positive_in_range_answers_the_element() {
        // bytes([10, 20, 30])[2] is 30 — p-typed-array.py `bytes_from_iterable`'s `ok` row
        let receiver = bytes_literal_value(&[10, 20, 30]);
        let got = unwrap_value(bytes_index(&receiver, &integer(2.0)).expect("must decide"));
        assert_eq!(got, integer(30.0));
    }

    #[test]
    fn bytes_index_negative_answers_from_the_end() {
        let receiver = bytes_literal_value(&[10, 20, 30]);
        let got = unwrap_value(bytes_index(&receiver, &integer(-1.0)).expect("must decide"));
        assert_eq!(got, integer(30.0));
    }

    #[test]
    fn bytes_index_reads_an_int_not_a_length_one_bytes() {
        // b"ab"[0] is the int 97 — p-typed-array.py `bytes_slice_is_not_an_element`
        let receiver = bytes_literal_value(b"ab");
        let got = unwrap_value(bytes_index(&receiver, &integer(0.0)).expect("must decide"));
        assert_eq!(got, integer(97.0));
    }

    // --- bytes_index: Raises path ---

    #[test]
    fn bytes_index_out_of_range_raises_index_error() {
        // execution-verified: bytes([1, 2])[10] raises IndexError: index out of range
        let receiver = bytes_literal_value(&[1, 2]);
        let message = unwrap_raises(bytes_index(&receiver, &integer(10.0)).expect("must decide"));
        assert!(message.contains("IndexError"), "{message}");
        assert!(message.contains("index out of range"), "{message}");
        assert!(message.contains("10"), "{message}");
    }

    #[test]
    fn bytes_index_negative_out_of_range_raises_index_error() {
        let receiver = bytes_literal_value(&[1, 2]);
        let message = unwrap_raises(bytes_index(&receiver, &integer(-5.0)).expect("must decide"));
        assert!(message.contains("IndexError"), "{message}");
    }

    // --- bytes_index: None path ---

    #[test]
    fn bytes_index_unknown_receiver_declines() {
        let receiver = refined_domain::abstract_value::unknown();
        assert!(bytes_index(&receiver, &integer(0.0)).is_none());
    }

    #[test]
    fn bytes_index_unknown_index_declines() {
        let receiver = bytes_literal_value(&[1, 2]);
        assert!(bytes_index(&receiver, &refined_domain::abstract_value::unknown()).is_none());
    }

    // --- bytes_slice ---

    #[test]
    fn bytes_slice_answers_a_sub_sequence_not_a_scalar() {
        // b"ab"[0:1] is b"a" — p-typed-array.py `bytes_slice_is_not_an_element`
        let receiver = bytes_literal_value(b"ab");
        let got = bytes_slice(&receiver, 0, 1).expect("must decide");
        assert_eq!(got.kind, Kind::List);
        assert_eq!(got.items, vec![integer(97.0)]);
    }

    #[test]
    fn bytes_slice_clamps_an_out_of_bounds_stop() {
        let receiver = bytes_literal_value(&[1, 2, 3]);
        let got = bytes_slice(&receiver, 0, 100).expect("must decide");
        assert_eq!(got.items.len(), 3);
    }

    #[test]
    fn bytes_slice_unknown_receiver_declines() {
        let receiver = refined_domain::abstract_value::unknown();
        assert!(bytes_slice(&receiver, 0, 1).is_none());
    }

    // --- bytearray_write_answer: Value path ---

    #[test]
    fn bytearray_write_in_range_answers_the_same_int() {
        // p-typed-array.py `bytearray_index_write`'s `data[0] = 10` row
        let got = unwrap_value(bytearray_write_answer(&integer(10.0)).expect("must decide"));
        assert_eq!(got, integer(10.0));
    }

    #[test]
    fn bytearray_write_of_200_does_not_raise() {
        // p-typed-array.py: "bytearray write of 200 does not raise (200 is
        // within bytearray's [0,255])" — only Age's declared range rejects it
        let got = unwrap_value(bytearray_write_answer(&integer(200.0)).expect("must decide"));
        assert_eq!(got, integer(200.0));
    }

    #[test]
    fn bytearray_write_of_boundary_values_does_not_raise() {
        assert!(matches!(
            bytearray_write_answer(&integer(0.0)),
            Some(BytesAnswer::Value(_))
        ));
        assert!(matches!(
            bytearray_write_answer(&integer(255.0)),
            Some(BytesAnswer::Value(_))
        ));
    }

    // --- bytearray_write_answer: Raises path ---

    #[test]
    fn bytearray_write_of_256_raises_value_error() {
        // p-typed-array.py `bytearray_write_out_of_byte_range_raises`: data[0] = 256
        let message = unwrap_raises(bytearray_write_answer(&integer(256.0)).expect("must decide"));
        assert!(message.contains("ValueError"), "{message}");
        assert!(message.contains("byte must be in range(0, 256)"), "{message}");
        assert!(message.contains("256"), "{message}");
    }

    #[test]
    fn bytearray_write_of_negative_one_raises_value_error() {
        // execution-verified: bytearray(1)[0] = -1 raises the SAME message as 256
        let message = unwrap_raises(bytearray_write_answer(&integer(-1.0)).expect("must decide"));
        assert!(message.contains("ValueError"), "{message}");
        assert!(message.contains("byte must be in range(0, 256)"), "{message}");
    }

    // --- bytearray_write_answer: None path ---

    #[test]
    fn bytearray_write_unknown_value_declines() {
        assert!(bytearray_write_answer(&refined_domain::abstract_value::unknown()).is_none());
    }

    #[test]
    fn bytearray_write_float_tagged_value_declines() {
        // a Float-tagged value is not this function's "known single Integer"
        // shape — declines rather than guessing at a cross-sort read
        assert!(bytearray_write_answer(&float(10.0)).is_none());
    }

    // --- memoryview_write_answer: Value path ---

    #[test]
    fn memoryview_write_in_range_answers_the_same_int_as_bytearray_would() {
        let got = unwrap_value(memoryview_write_answer(&integer(200.0)).expect("must decide"));
        assert_eq!(got, integer(200.0));
    }

    // --- memoryview_write_answer: Raises path (different wording than bytearray) ---

    #[test]
    fn memoryview_write_of_256_raises_the_memoryview_specific_wording() {
        // p-typed-array.py `memoryview_write_out_of_byte_range_raises`: view[0] = 256
        // execution-verified: "memoryview: invalid value for format 'B'",
        // NOT bytearray's "byte must be in range(0, 256)" wording
        let message = unwrap_raises(memoryview_write_answer(&integer(256.0)).expect("must decide"));
        assert!(message.contains("ValueError"), "{message}");
        assert!(message.contains("memoryview: invalid value for format 'B'"), "{message}");
        assert!(!message.contains("byte must be in range"), "{message}");
    }

    // --- memoryview_write_answer: None path ---

    #[test]
    fn memoryview_write_unknown_value_declines() {
        assert!(memoryview_write_answer(&refined_domain::abstract_value::unknown()).is_none());
    }

    // --- tagged ---

    #[test]
    fn tagged_stamps_kind_word_on_a_list_value() {
        let list = bytes_literal_value(&[0]);
        let stamped = tagged(list, BYTEARRAY_WORD);
        assert_eq!(stamped.kind, Kind::List);
        assert_eq!(stamped.kind_word, Some(BYTEARRAY_WORD));
    }

    #[test]
    fn tagged_leaves_a_non_list_value_unstamped() {
        // a receiver this function never claims to know how to tag (an
        // unrecognized memoryview argument, for instance) passes through
        // exactly as `bytes_like_construction_value`'s own decline already
        // states — no accidental tag on a shape this file does not own.
        let scalar = integer(40.0);
        let stamped = tagged(scalar.clone(), BYTEARRAY_WORD);
        assert_eq!(stamped, scalar);
        assert_eq!(stamped.kind_word, None);
    }

    // --- bytes_write_answer: species dispatch ---

    #[test]
    fn bytes_write_answer_on_a_bytearray_receiver_uses_the_bytearray_rule() {
        let receiver = tagged(bytes_literal_value(&[0]), BYTEARRAY_WORD);
        let message = unwrap_raises(bytes_write_answer(&receiver, &integer(256.0)).expect("must decide"));
        assert!(message.contains("byte must be in range(0, 256)"), "{message}");
    }

    #[test]
    fn bytes_write_answer_on_a_memoryview_receiver_uses_the_memoryview_rule() {
        let receiver = tagged(bytes_literal_value(&[0]), MEMORYVIEW_WORD);
        let message = unwrap_raises(bytes_write_answer(&receiver, &integer(256.0)).expect("must decide"));
        assert!(message.contains("memoryview: invalid value for format 'B'"), "{message}");
    }

    #[test]
    fn bytes_write_answer_on_a_bytes_receiver_always_raises_type_error() {
        // p-typed-array.py `bytes_is_immutable`: `frozen[0] = 99` raises
        // even though 99 is a perfectly in-range byte — bytes has no
        // __setitem__ at all, so the VALUE never matters.
        let receiver = tagged(bytes_literal_value(&[10, 20]), BYTES_WORD);
        let message = unwrap_raises(bytes_write_answer(&receiver, &integer(99.0)).expect("must decide"));
        assert!(message.contains("TypeError"), "{message}");
        assert!(message.contains("'bytes' object does not support item assignment"), "{message}");
    }

    #[test]
    fn bytes_write_answer_on_an_untagged_list_declines() {
        // a plain list/tuple (kind_word: None) has no rule this function
        // owns — the caller's own list_with_item path is the honest one.
        let receiver = list_literal_value_for_test();
        assert!(bytes_write_answer(&receiver, &integer(200.0)).is_none());
    }

    fn list_literal_value_for_test() -> AbstractValue {
        known_list(vec![integer(1.0), integer(2.0)], TrustProved)
    }

    #[test]
    fn base64_call_result_b64encode_tags_the_result() {
        let got = base64_call_result("b64encode", &[list_literal_value_for_test()]).expect("base64.b64encode(...) models");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.kind_word, Some(BASE64_ENCODED_WORD));
    }

    #[test]
    fn base64_call_result_unmodeled_function_declines() {
        let got = base64_call_result("b64decode", &[list_literal_value_for_test()]);
        assert!(got.is_none(), "b64decode is out of this brief's scope: {got:?}");
    }

    #[test]
    fn bytes_decode_on_a_b64encode_tagged_receiver_answers_the_base64_alphabet_grammar() {
        let receiver = base64_call_result("b64encode", &[list_literal_value_for_test()]).expect("b64encode models");
        let got = bytes_decode_call(&receiver, &[]).expect(".decode() on a b64encode result models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
        let alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=";
        let points: Vec<f64> = alphabet.chars().map(|c| c as u32 as f64).collect();
        let want = make_refined_set(vec![repeat_of(make_refined_set(vec![one_of(&points)]), 0, None)]);
        assert_eq!(got.set, want);
    }

    #[test]
    fn bytes_decode_on_an_untagged_receiver_declines() {
        let receiver = tagged(bytes_literal_value(&[10, 20]), BYTES_WORD);
        let got = bytes_decode_call(&receiver, &[]);
        assert!(got.is_none(), "a plain bytes receiver's .decode() is not modeled: {got:?}");
    }

    #[test]
    fn bytes_decode_with_an_argument_declines() {
        let receiver = base64_call_result("b64encode", &[list_literal_value_for_test()]).expect("b64encode models");
        let got = bytes_decode_call(&receiver, &[string_value("utf-8")]);
        assert!(got.is_none());
    }

    fn string_value(text: &str) -> AbstractValue {
        let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
        known_values(code_points, PrimitiveKind::String, TrustProved)
    }
}
