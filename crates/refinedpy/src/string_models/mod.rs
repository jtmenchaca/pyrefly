//! String VALUE states: a known string literal's abstract value, the
//! exactly-decidable `str` methods on an exact-string receiver
//! (`string_method_result`), and the SORT-ONLY answer the same methods
//! state over a receiver that is String-sorted but not exact
//! (`string_method_sort_only_result`/`string_method_int_sort_only_result`
//! — the whole-strings-ground or whole-int-ray claim a method's own
//! CPython contract still proves even when the receiver's own content
//! is unread).
//!
//! ## How the domain carries a string
//!
//! `refined_domain::abstract_value::AbstractValue` has no dedicated
//! string variant. An exact string is `Kind::Values` with
//! `kind_tag == Some(PrimitiveKind::String)`: `values: Vec<f64>` holds
//! one Unicode code point per element (`format_abstract_values.rs`'s
//! `string_of` reads it back with `char::from_u32`, and
//! `expressions.rs`'s `Expr::BooleanLiteral` arm shows the same
//! `known_values(vec![...], PrimitiveKind::<Sort>, TrustProved)`
//! shape this file reuses for strings and booleans alike). There is no
//! separate "string length" measure carried on `Kind::Values` — the
//! length is `values.len()`, read directly off the code-point vector.
//! `Measures` (abstract_value.rs) carries `sum`/`sorted`, which are
//! sequence-reduce facts for `Kind::Set`, not a string-length state;
//! nothing here needs it.
//!
//! `len(s)` on a Python string counts Unicode code points, not bytes or
//! grapheme clusters ("Strings are immutable sequences of Unicode code
//! points," library/stdtypes.html, Text Sequence Type — str). A
//! one-code-point-per-`f64` vector already counts code points by
//! construction, so `string_literal_value`'s length and every method
//! result below need no separate code-point count — `values.len()` IS
//! `len()`.
//!
//! ## Coverage cited against library/stdtypes.html (str methods)
//!
//! - `upper`: "Return a copy of the string with all the cased
//!   characters converted to uppercase."
//! - `lower`: "Return a copy of the string with all the cased
//!   characters converted to lowercase."
//! - `strip`/`lstrip`/`rstrip` (no `chars` argument): "the chars
//!   argument defaults to removing whitespace" — the no-arg row this
//!   file models trims Unicode whitespace (`str.isspace()`-defined)
//!   from both/leading/trailing ends.
//! - `replace(old, new)` (no `count` argument): "If count is not
//!   specified or -1, then all occurrences are replaced" — every
//!   occurrence of `old` is replaced, matching the brief's confirmed
//!   fact.
//! - `startswith`/`endswith` (no `start`/`end` slice arguments): "Return
//!   True if string starts with the prefix, otherwise return False" /
//!   "Return True if the string ends with the specified suffix,
//!   otherwise return False."
//! - `find`: "Return the lowest index in the string where substring sub
//!   is found... Return -1 if sub is not found."
//! - `index`: "Like `str.find`, but raise `ValueError` when the
//!   substring is not found" — this file answers only the FOUND case
//!   (the same position `find` would); a miss is `provable_raise`'s row
//!   (`expressions.rs`), never a fabricated value here.
//! - `casefold`: "Casefolding is similar to lowercasing but more
//!   aggressive" and follows Unicode's full case-folding table (Unicode
//!   Standard section 3.13) — that table diverges from plain
//!   lowercasing only OUTSIDE ASCII, so an ASCII-only receiver answers
//!   `to_lowercase()` exactly (see `string_method_result`'s own doc); a
//!   non-ASCII receiver declines.
//! - `join(iterable)`: "Return a string which is the concatenation of
//!   the strings in *iterable*... The separator between elements is
//!   the string providing this method." Modeled for a known
//!   `Kind::List` (this domain's shared list/generator/comprehension
//!   shape, `collection_models.rs`'s own module doc) of known exact-
//!   string elements only; a non-string element declines the whole
//!   call, matching str.join's own `TypeError` on a non-string member.
//! - `split(sep)` (one string-separator argument, no `maxsplit`):
//!   "Return a list of the words in the string, using *sep* as the
//!   delimiter string... consecutive delimiters are not grouped
//!   together and are deemed to delimit empty strings." The no-argument
//!   whitespace-splitting form (`sep=None`) is NOT modeled: that row
//!   collapses runs of whitespace and strips leading/trailing empty
//!   strings, a different splitting rule from the one-argument exact-
//!   separator row this file builds.
//!
//! Concatenation (`+`) is not a method call in Python's grammar — it is
//! `ast.BinOp` with `Operator::Add`, the same node `expressions.rs`
//! already dispatches numeric arithmetic through. Modeling it belongs
//! beside that dispatcher, not in this method-result function.
//!
//! ## Module layout
//!
//! `method_result` holds the exact-receiver dispatch
//! (`string_method_result`); `sort_only` holds the two SORT-ONLY
//! fallbacks and the ASCII case-mapping window reader; `regex_match`
//! holds the `re.fullmatch`/`re.finditer` Match-object value and
//! `.group(n)` reader; `formatting` holds the `f"{x:.{p}f}"` grammar
//! reader. This file keeps the three items every sibling shares:
//! `string_literal_value` (the literal builder), `exact_string_text`
//! (its reader), and `boolean_value` (the shared True/False encoding).

mod formatting;
mod method_result;
mod regex_match;
mod sort_only;

#[cfg(test)]
mod tests;

pub use formatting::fixed_precision_decimal_grammar;
pub use formatting::fixed_precision_decimal_width;
pub use method_result::string_method_result;
pub use regex_match::match_object_value;
pub use regex_match::matched_group_grammar;
pub use regex_match::MATCH_WITH_GROUPS_WORD;
pub use sort_only::string_method_int_sort_only_result;
pub use sort_only::string_method_sort_only_result;

// Test module is a sibling of the domain children, so re-export the
// private items it needs into this module's namespace for `tests`'s
// `use super::*`.
#[cfg(test)]
pub(self) use regex_match::capture_group_spans;
#[cfg(test)]
#[allow(unused_imports)]
pub(self) use regex_match::CaptureGroup;

use refined_domain::abstract_value::{known_values, AbstractValue, Kind, PrimitiveKind};
use refined_domain::trust_grades::TrustProved;

/// The state for a known string literal: an exact value, one f64 code
/// point per `char`, sorted `PrimitiveKind::String`. `len()` on the
/// result is `text.chars().count()` — Python's own code-point count —
/// never `text.len()` (Rust's UTF-8 byte length, which overcounts every
/// multibyte character).
pub fn string_literal_value(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    known_values(code_points, PrimitiveKind::String, TrustProved)
}

/// The exact text an AbstractValue carries, if it is a `Kind::Values`
/// state sorted `PrimitiveKind::String` — the receiver/argument shape
/// every row in this module requires. The code-point-to-`String`
/// conversion is the same one `refined_domain::format_abstract_values`'s
/// (private, `pub(crate)`) `string_of` and `lattice_operations`'s
/// (module-private) `string_of` both already carry; this file is
/// out-of-crate from `refined_domain` and owns no edit rights there
/// (AGENT-BRIEF.md: this wave touches only this file), so the same
/// one-line conversion is repeated here rather than widening another
/// crate's visibility.
pub(super) fn exact_string_text(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(
        value
            .values
            .iter()
            .filter_map(|c| char::from_u32(*c as i64 as u32))
            .collect(),
    )
}

/// A boolean AbstractValue — `PrimitiveKind::Boolean` with a single 0.0
/// (false) or 1.0 (true), the same encoding `expressions.rs`'s
/// `Expr::BooleanLiteral` arm builds for `True`/`False` literals.
pub(super) fn boolean_value(value: bool) -> AbstractValue {
    known_values(vec![if value { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved)
}
