//! The per-position refusal rewrites: `at_key`/`at_index` (the plain
//! dict/list element suffix), `at_member`/`at_slot` (the sharper
//! key-prefixed / ordinal-slot rewrites, both recognizing ONLY
//! `refutation`'s own plain composed shape via `plain_refutation_
//! value_words`), `slot_word` (the ordinal-slot name),
//! `missing_required_key` (a closed value's own absent-required-key
//! refusal), and `element_set_refutation` (a container-typed sink's own
//! whole-range refusal).

use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::RefinedSet;

/// A member's own refutation, keyed by the name it escaped under — the
/// dict/TypedDict element law's own suffix.
pub fn at_key(message: &str, key: &str) -> String {
    format!("{message} (at key '{key}')")
}

/// A member's own refutation, PREFIXED by which key escaped and the
/// value that did — "the key 'spo2' of value 130 is not assignable to
/// type '>= 0 && <= 100'" — the showcase's own `Vitals(heart_rate=72,
/// spo2=130)` row, where the value arrives through pydantic's
/// `BaseModel` CONSTRUCTOR rather than a bare dict literal (the shape
/// `at_key` already serves for `m-pydantic-schema.py`'s
/// `Person.model_validate({...})`/`{"age": 200, ...}` rows). Like
/// `at_slot`, this recognizes ONLY `refutation`'s own plain composed
/// shape ("a value of type '<value>' is not assignable to type ...", no
/// reason clause riding after it) and rewrites it; any other shape (a
/// sort crossing, a nested structural mismatch) falls back to the
/// ordinary `at_key` suffix unchanged, so a shape this function does
/// not precisely recognize is never corrupted by a blind rewrite.
pub fn at_member(message: &str, key: &str, member_set: &RefinedSet) -> String {
    if let Some(value_words) = plain_refutation_value_words(message) {
        return format!(
            "the key '{key}' of value {value_words} is not assignable to type '{}'",
            format_for_diagnostics(member_set)
        );
    }
    at_key(message, key)
}

/// The refusal a CLOSED value earns for a declared key it does not
/// carry — the MEMBERS LAW's own absent-required-key sentence, the twin
/// of the Go checker's `"<what> is missing the key '<name>'"`
/// (`walk/object_assignability.go`'s `CheckObjectTarget`). `judge` is
/// sink-agnostic by this module's own design, so the flowing thing is
/// "a value" — the same word every sibling law here uses — rather than a
/// sink-specific word (return/argument/key) `judge` cannot know.
///
/// Says WHY the absence is a refusal rather than nothing: the
/// declaration requires the key (library/typing.rst, `TypedDict`: "By
/// default, all keys must be present in a ``TypedDict``"), and the value
/// states its complete key set, so the key is proved absent rather than
/// merely unread.
pub fn missing_required_key(key: &str, spelling: &str) -> String {
    format!("a value is missing the required key '{key}' — '{spelling}' requires it, and this value states its complete key set")
}

/// A member's own refutation, keyed by the index it escaped at — the
/// list/tuple element law's own suffix.
pub fn at_index(message: &str, index: usize) -> String {
    format!("{message} (at index {index})")
}

/// The ordinal name a fixed-arity tuple's own slot earns in prose: "the
/// first slot"/"the second slot"/… for a short tuple, "the middle slot"
/// for the exact center position of an ODD-length tuple (never a
/// numbered ordinal there — a 3-tuple's own center reads as "the middle
/// slot," matching the fixed-arity POSITIONS LAW's own corpus wording),
/// and "the last slot" for the final position of a tuple longer than
/// two. `length` is the tuple's own arity; `index` is the zero-based
/// slot this word names.
pub fn slot_word(index: usize, length: usize) -> String {
    if length > 2 && length % 2 == 1 && index == length / 2 {
        return "the middle slot".to_owned();
    }
    if length > 2 && index == length - 1 {
        return "the last slot".to_owned();
    }
    const ORDINALS: [&str; 10] = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    ];
    match ORDINALS.get(index) {
        Some(word) => format!("the {word} slot"),
        None => format!("slot {index}"),
    }
}

/// A fixed-arity tuple's own PER-SLOT refutation suffix — the
/// POSITIONS LAW's own twin of `at_index`. The recursive
/// `judge(item, position_declared, kernel)` call that produced
/// `message` may have taken any of `judge`'s own laws (an ordinary
/// scalar refutation, a sort crossing, a nested member/element fire),
/// so this recognizes ONLY the one shape `refutation` itself composes
/// ("a value of type '<value>' is not assignable to type ..." — the
/// plain per-value membership fire, what a scalar-ground slot like
/// `Channel` actually produces) and rewrites it into the ordinal-slot
/// phrasing the showcase's own `paint((255, 300, 0))` row states: "300
/// in the middle slot is not assignable to type '>= 0 && <= 255 &&
/// integer'" — the value bare (no "a value of type" preamble, since the
/// slot word already carries the position), the ordinal slot name, and
/// the slot's own declared contents (never the outer tuple's alias
/// name — a slot's own set is what a reader needs at that exact
/// position). Any OTHER message shape (a sort-crossing reason clause, a
/// nested structural mismatch) falls back to `at_index`'s own plain
/// suffix, unchanged — this never rewrites a shape it does not
/// precisely recognize, so no other law's own wording is corrupted by
/// a shape-blind string edit.
pub fn at_slot(message: &str, index: usize, length: usize, slot_set: &RefinedSet) -> String {
    if let Some(value_words) = plain_refutation_value_words(message) {
        return format!(
            "{value_words} in {} is not assignable to type '{}'",
            slot_word(index, length),
            format_for_diagnostics(slot_set)
        );
    }
    at_index(message, index)
}

/// Recognizes `refutation`'s own exact composed shape — "a value of
/// type '<value>' is not assignable to type ..." — and answers `<value>`
/// bare, or `None` for a message any other law in this file composed
/// (a sort crossing appends its own reason clause after this same
/// prefix, so it is EXCLUDED here on purpose: a slot's sort-crossing
/// fire keeps `at_index`'s ordinary suffix, since the reason clause
/// names both sides' sorts in prose that already reads correctly
/// without an ordinal rewrite).
pub(super) fn plain_refutation_value_words(message: &str) -> Option<&str> {
    let rest = message.strip_prefix("a value of type '")?;
    let (value_words, rest) = rest.split_once("' is not assignable to type ")?;
    if rest.contains(" — ") {
        return None; // a reason clause rides after this prefix — not the plain shape
    }
    Some(value_words)
}

/// A container-typed sink's own ELEMENT-SET refutation: the flowing
/// value's own element set is not a SUBSET of the declared element
/// set — unlike `at_index`'s per-position naming (one escaping item at
/// one index), this names the whole admitted element range on both
/// sides, the shape a bare `list[X]` sink states when the flowing
/// value's own length is unknown (a comprehension over an
/// unbounded-length parameter, `check.rs::seed_parameters`'s own
/// `Kind::Set` sequence seed) so there is no single index to blame.
/// Spelled "list of (<contents>)" — Python's own container word, never
/// the shared `refined_sets` "array of" spelling
/// (`format_for_diagnostics`'s own `Form::Star` wording, which this
/// sentence does not call for the outer container itself). `judge` is
/// sink-agnostic by this file's own design (one seam every sink routes
/// through), so this names the flowing thing "a value" — the same word
/// every sibling law in this module uses — rather than a sink-specific
/// word (return/argument/key) `judge` has no way to know here.
pub fn element_set_refutation(value_element_set: &RefinedSet, declared_element_set: &RefinedSet) -> String {
    format!(
        "a value of type 'list of ({})' is not assignable to type 'list of ({})'",
        format_for_diagnostics(value_element_set),
        format_for_diagnostics(declared_element_set)
    )
}
