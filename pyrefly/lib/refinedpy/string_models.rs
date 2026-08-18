/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! String VALUE states: a known string literal's abstract value, and
//! the exactly-decidable `str` methods on an exact-string receiver.
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

use refined_domain::abstract_value::{known_values, AbstractValue, Kind, PrimitiveKind};
use refined_domain::known_constructors::known_list;
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

/// The state a `str` method call answers, for the rows this file can
/// decide exactly: an exact-string receiver, and (where the method
/// takes one) exact-string arguments. Every other method, receiver
/// shape, or argument shape answers `None` — the caller's honest
/// "not modeled" rather than a guessed value.
///
/// `+` concatenation is not covered here: Python's `+` is a BinOp, not
/// a method call, and belongs beside `expressions.rs`'s arithmetic
/// dispatch on `Operator::Add` (out of this file's scope by the brief).
pub fn string_method_result(
    method: &str,
    receiver: &AbstractValue,
    arguments: &[AbstractValue],
) -> Option<AbstractValue> {
    let receiver_text = exact_string_text(receiver)?;
    match method {
        // "Return a copy of the string with all the cased characters
        // converted to uppercase." (library/stdtypes.html, str.upper)
        "upper" if arguments.is_empty() => Some(string_literal_value(&receiver_text.to_uppercase())),
        // "Return a copy of the string with all the cased characters
        // converted to lowercase." (str.lower)
        "lower" if arguments.is_empty() => Some(string_literal_value(&receiver_text.to_lowercase())),
        // "Return a copy of the string with the leading and trailing
        // characters removed... the chars argument defaults to removing
        // whitespace." (str.strip, no-arg row)
        "strip" if arguments.is_empty() => Some(string_literal_value(receiver_text.trim())),
        // "Return a copy of the string with leading characters
        // removed... defaults to removing whitespace." (str.lstrip,
        // no-arg row)
        "lstrip" if arguments.is_empty() => Some(string_literal_value(receiver_text.trim_start())),
        // "Return a copy of the string with trailing characters
        // removed... defaults to removing whitespace." (str.rstrip,
        // no-arg row)
        "rstrip" if arguments.is_empty() => Some(string_literal_value(receiver_text.trim_end())),
        // "Return a copy of the string with all occurrences of
        // substring old replaced by new. ... If count is not specified
        // or -1, then all occurrences are replaced." (str.replace,
        // two-arg row — every occurrence, per the brief's confirmed
        // fact)
        "replace" if arguments.len() == 2 => {
            let old = exact_string_text(&arguments[0])?;
            let new = exact_string_text(&arguments[1])?;
            Some(string_literal_value(&receiver_text.replace(&old, &new)))
        }
        // "Return True if string starts with the prefix, otherwise
        // return False." (str.startswith, one-arg exact-prefix row)
        "startswith" if arguments.len() == 1 => {
            let prefix = exact_string_text(&arguments[0])?;
            Some(boolean_value(receiver_text.starts_with(&prefix)))
        }
        // "Return True if the string ends with the specified suffix,
        // otherwise return False." (str.endswith, one-arg exact-suffix
        // row)
        "endswith" if arguments.len() == 1 => {
            let suffix = exact_string_text(&arguments[0])?;
            Some(boolean_value(receiver_text.ends_with(&suffix)))
        }
        // "Return the lowest index in the string where substring sub is
        // found... Return -1 if sub is not found." (str.find, one-arg
        // row). The index is a CODE-POINT index (chars().position),
        // matching len()'s own code-point count — never a byte offset.
        // Integer sort, not bare Number: `str.find` always returns a
        // Python `int` (the found index, or the literal `-1`), never a
        // float — so its result feeds a slice bound
        // (`expressions.rs`'s `slice_bound_index`, which accepts only
        // Integer-sorted bounds) the same way an ordinary `int` literal
        // index does.
        "find" if arguments.len() == 1 => {
            let needle = exact_string_text(&arguments[0])?;
            Some(known_values(
                vec![find_code_point_index(&receiver_text, &needle)],
                PrimitiveKind::Integer,
                TrustProved,
            ))
        }
        // "Like str.find, but raise ValueError when the substring is not
        // found." (str.index, one-arg row — no `start`/`end` bounds
        // modeled, matching find's own one-arg scope). Only the FOUND
        // case answers a value here: a miss raises at runtime, which
        // `expressions.rs::call_provable_raise` already proves separately
        // (this file's own `str.find` row already carries the shared
        // code-point-index computation both rows read).
        "index" if arguments.len() == 1 => {
            let needle = exact_string_text(&arguments[0])?;
            let position = find_code_point_index(&receiver_text, &needle);
            if position < 0.0 {
                return None;
            }
            Some(known_values(vec![position], PrimitiveKind::Integer, TrustProved))
        }
        // "Return a casefolded copy of the string... Casefolding is
        // similar to lowercasing but more aggressive because it is
        // intended to remove all case distinctions in a string"
        // (library/stdtypes.rst, str.casefold). The full Unicode
        // case-folding table (Unicode Standard section 3.13, cited by
        // the same doc) diverges from plain lowercasing only OUTSIDE
        // ASCII (its own worked example: German "ß" casefolds to "ss",
        // which "lower" leaves unchanged) — inside the ASCII range,
        // casefolding and lowercasing coincide exactly (ASCII has no
        // multi-character or non-1:1 case mapping at all), so an
        // ASCII-only receiver answers `to_lowercase()` exactly. A
        // receiver carrying any non-ASCII code point declines rather
        // than approximate with a mapping that is not exact there.
        "casefold" if arguments.is_empty() => {
            if !receiver_text.is_ascii() {
                return None;
            }
            Some(string_literal_value(&receiver_text.to_lowercase()))
        }
        // "Return a list of the words in the string, using *sep* as the
        // delimiter string... consecutive delimiters are not grouped
        // together and are deemed to delimit empty strings." (str.split,
        // one-arg exact-separator row; str.split's own `maxsplit`
        // argument is not modeled). An EMPTY separator raises
        // `ValueError` in CPython ("empty separator") — this row
        // declines rather than answer the whitespace-splitting fallback
        // an empty `sep` never actually falls back to.
        "split" if arguments.len() == 1 => {
            let sep = exact_string_text(&arguments[0])?;
            if sep.is_empty() {
                return None;
            }
            let parts: Vec<AbstractValue> = receiver_text.split(&sep).map(string_literal_value).collect();
            Some(known_list(parts, TrustProved))
        }
        // "Return a string which is the concatenation of the strings in
        // *iterable*... The separator between elements is the string
        // providing this method." (str.join, one-arg row — the
        // receiver IS the separator; the argument is a known Kind::List
        // of known exact-string elements, this domain's shared
        // list/generator shape)
        "join" if arguments.len() == 1 => {
            let iterable = &arguments[0];
            if iterable.kind != Kind::List {
                return None;
            }
            let mut parts: Vec<String> = Vec::with_capacity(iterable.items.len());
            for element in &iterable.items {
                parts.push(exact_string_text(element)?);
            }
            Some(string_literal_value(&parts.join(&receiver_text)))
        }
        _ => None,
    }
}

/// The exact text an AbstractValue carries, if it is a `Kind::Values`
/// state sorted `PrimitiveKind::String` — the receiver/argument shape
/// every row above requires. The code-point-to-`String` conversion is
/// the same one `refined_domain::format_abstract_values`'s (private,
/// `pub(crate)`) `string_of` and `lattice_operations`'s (module-private)
/// `string_of` both already carry; this file is out-of-crate from
/// `refined_domain` and owns no edit rights there (AGENT-BRIEF.md: this
/// wave touches only this file), so the same one-line conversion is
/// repeated here rather than widening another crate's visibility.
fn exact_string_text(value: &AbstractValue) -> Option<String> {
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

/// The lowest CODE-POINT index of `needle` in `haystack`, or -1 if
/// absent — `str.find`'s own contract, computed over `chars()` so a
/// multibyte character before the match counts as one position, the
/// same way Python's own indexing does.
fn find_code_point_index(haystack: &str, needle: &str) -> f64 {
    if needle.is_empty() {
        return 0.0;
    }
    let haystack_chars: Vec<char> = haystack.chars().collect();
    let needle_chars: Vec<char> = needle.chars().collect();
    if needle_chars.len() > haystack_chars.len() {
        return -1.0;
    }
    for start in 0..=(haystack_chars.len() - needle_chars.len()) {
        if haystack_chars[start..start + needle_chars.len()] == needle_chars[..] {
            return start as f64;
        }
    }
    -1.0
}

/// A boolean AbstractValue — `PrimitiveKind::Boolean` with a single 0.0
/// (false) or 1.0 (true), the same encoding `expressions.rs`'s
/// `Expr::BooleanLiteral` arm builds for `True`/`False` literals.
fn boolean_value(value: bool) -> AbstractValue {
    known_values(vec![if value { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_literal_value_round_trips_ascii() {
        let value = string_literal_value("ab");
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
        assert_eq!(exact_string_text(&value).as_deref(), Some("ab"));
    }

    /// "héllo" is 5 Unicode code points ('h','é','l','l','o') — the
    /// same count `len("héllo")` gives in CPython, and different from
    /// Rust's `"héllo".len()` (6 UTF-8 bytes, because 'é' is two
    /// bytes).
    #[test]
    fn test_string_literal_value_length_is_code_points_not_bytes() {
        let value = string_literal_value("héllo");
        assert_eq!(value.values.len(), 5);
        assert_ne!(value.values.len(), "héllo".len());
    }

    #[test]
    fn test_upper_no_arg() {
        let receiver = string_literal_value("ab");
        let result = string_method_result("upper", &receiver, &[]).expect("upper must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("AB"));
    }

    #[test]
    fn test_lower_no_arg() {
        let receiver = string_literal_value("AB");
        let result = string_method_result("lower", &receiver, &[]).expect("lower must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
    }

    #[test]
    fn test_strip_no_arg() {
        let receiver = string_literal_value("  ab  ");
        let result = string_method_result("strip", &receiver, &[]).expect("strip must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
    }

    #[test]
    fn test_lstrip_no_arg() {
        let receiver = string_literal_value("  ab");
        let result = string_method_result("lstrip", &receiver, &[]).expect("lstrip must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
    }

    #[test]
    fn test_rstrip_no_arg() {
        let receiver = string_literal_value("ab  ");
        let result = string_method_result("rstrip", &receiver, &[]).expect("rstrip must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("ab"));
    }

    /// str.replace with no count replaces EVERY occurrence — the
    /// brief's confirmed fact, distinct from a single-substitution
    /// replace.
    #[test]
    fn test_replace_all_occurrences() {
        let receiver = string_literal_value("abXcdXef");
        let old = string_literal_value("X");
        let new = string_literal_value("Y");
        let result =
            string_method_result("replace", &receiver, &[old, new]).expect("replace must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("abYcdYef"));
    }

    #[test]
    fn test_startswith_true() {
        let receiver = string_literal_value("banana");
        let prefix = string_literal_value("ban");
        let result =
            string_method_result("startswith", &receiver, &[prefix]).expect("startswith must decide");
        assert_eq!(result.values, vec![1.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Boolean));
    }

    #[test]
    fn test_startswith_false() {
        let receiver = string_literal_value("banana");
        let prefix = string_literal_value("apple");
        let result =
            string_method_result("startswith", &receiver, &[prefix]).expect("startswith must decide");
        assert_eq!(result.values, vec![0.0]);
    }

    #[test]
    fn test_endswith_true() {
        let receiver = string_literal_value("banana");
        let suffix = string_literal_value("ana");
        let result = string_method_result("endswith", &receiver, &[suffix]).expect("endswith must decide");
        assert_eq!(result.values, vec![1.0]);
    }

    #[test]
    fn test_find_hit() {
        let receiver = string_literal_value("banana");
        let needle = string_literal_value("a");
        let result = string_method_result("find", &receiver, &[needle]).expect("find must decide");
        assert_eq!(result.values, vec![1.0]);
        // Integer, not bare Number: str.find always returns a Python int
        // (the found index or -1), so its result can feed a slice bound
        // (expressions.rs's slice_bound_index requires Integer sort).
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// str.find answers -1 on a missing needle — the twin of JS
    /// `indexOf`, never a raised exception (that is str.index's row).
    #[test]
    fn test_find_miss_answers_negative_one() {
        let receiver = string_literal_value("banana");
        let needle = string_literal_value("z");
        let result = string_method_result("find", &receiver, &[needle]).expect("find must decide");
        assert_eq!(result.values, vec![-1.0]);
    }

    /// find's index counts CODE POINTS: "é" is one position, so the "l"
    /// after it is at index 2, not 3 (which a byte-offset find would
    /// give, since "é" is two UTF-8 bytes).
    #[test]
    fn test_find_counts_code_points_not_bytes() {
        let receiver = string_literal_value("héllo");
        let needle = string_literal_value("l");
        let result = string_method_result("find", &receiver, &[needle]).expect("find must decide");
        assert_eq!(result.values, vec![2.0]);
    }

    /// str.index on a present needle answers the same position find
    /// would — the c-reads-and-values.py string_index row's in-set leg.
    #[test]
    fn test_index_hit_answers_the_found_position() {
        let receiver = string_literal_value("banana");
        let needle = string_literal_value("a");
        let result = string_method_result("index", &receiver, &[needle]).expect("index must decide");
        assert_eq!(result.values, vec![1.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// str.index on a missing needle declines — the miss is a raise
    /// (ValueError), not a value this function answers.
    #[test]
    fn test_index_miss_declines() {
        let receiver = string_literal_value("banana");
        let needle = string_literal_value("z");
        assert_eq!(string_method_result("index", &receiver, &[needle]), None);
    }

    /// casefold on an ASCII-only receiver matches plain lowercasing
    /// exactly — ASCII has no case mapping the two diverge on.
    #[test]
    fn test_casefold_ascii_matches_lowercase() {
        let receiver = string_literal_value("AbC");
        let result = string_method_result("casefold", &receiver, &[]).expect("casefold(ascii) must decide");
        assert_eq!(exact_string_text(&result).as_deref(), Some("abc"));
    }

    /// casefold declines outside ASCII: German "ß" casefolds to "ss"
    /// (length-changing), which plain `to_lowercase` does not produce —
    /// stdtypes.rst's own worked example for why casefold and lower
    /// diverge.
    #[test]
    fn test_casefold_non_ascii_declines() {
        let receiver = string_literal_value("stra\u{df}e");
        assert_eq!(string_method_result("casefold", &receiver, &[]), None);
    }

    /// A non-exact-string receiver (unknown) declines every row.
    #[test]
    fn test_non_string_receiver_declines() {
        let receiver = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
        assert_eq!(string_method_result("upper", &receiver, &[]), None);
    }

    /// replace with a non-exact-string argument declines rather than
    /// guessing.
    #[test]
    fn test_replace_with_unknown_argument_declines() {
        let receiver = string_literal_value("abXcd");
        let old = string_literal_value("X");
        let new = known_values(vec![1.0], PrimitiveKind::Number, TrustProved);
        assert_eq!(string_method_result("replace", &receiver, &[old, new]), None);
    }

    #[test]
    fn test_split_by_string_separator() {
        let receiver = string_literal_value("ab,cd,ef");
        let sep = string_literal_value(",");
        let result = string_method_result("split", &receiver, &[sep]).expect("split must decide");
        assert_eq!(result.kind, Kind::List);
        assert_eq!(result.items.len(), 3);
        assert_eq!(exact_string_text(&result.items[0]).as_deref(), Some("ab"));
        assert_eq!(exact_string_text(&result.items[1]).as_deref(), Some("cd"));
        assert_eq!(exact_string_text(&result.items[2]).as_deref(), Some("ef"));
    }

    /// consecutive delimiters delimit an empty string, matching
    /// stdtypes.rst's own worked example ("'1,,2'.split(',')" -> ['1',
    /// '', '2']).
    #[test]
    fn test_split_consecutive_delimiters_yield_an_empty_element() {
        let receiver = string_literal_value("1,,2");
        let sep = string_literal_value(",");
        let result = string_method_result("split", &receiver, &[sep]).expect("split must decide");
        assert_eq!(result.items.len(), 3);
        assert_eq!(exact_string_text(&result.items[1]).as_deref(), Some(""));
    }

    #[test]
    fn test_split_empty_separator_declines() {
        let receiver = string_literal_value("ab");
        let sep = string_literal_value("");
        assert_eq!(string_method_result("split", &receiver, &[sep]), None);
    }
}
