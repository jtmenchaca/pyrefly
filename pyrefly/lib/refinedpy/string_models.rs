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
//! - `casefold`: declines (see `string_method_result`'s own doc) —
//!   "Casefolding is similar to lowercasing but more aggressive" and
//!   follows Unicode's full case-folding table (Unicode Standard
//!   section 3.13), a locale-and-table-dependent mapping this file does
//!   not carry.
//!
//! Concatenation (`+`) is not a method call in Python's grammar — it is
//! `ast.BinOp` with `Operator::Add`, the same node `expressions.rs`
//! already dispatches numeric arithmetic through. Modeling it belongs
//! beside that dispatcher, not in this method-result function.

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
        "find" if arguments.len() == 1 => {
            let needle = exact_string_text(&arguments[0])?;
            Some(known_values(
                vec![find_code_point_index(&receiver_text, &needle)],
                PrimitiveKind::Number,
                TrustProved,
            ))
        }
        // Casefolding follows the Unicode Standard's full case-folding
        // table (section 3.13, cited by str.casefold's own docs) rather
        // than a per-character mapping this file can compute exactly —
        // "strasse" for German "straße" is a length-changing,
        // locale-independent-but-table-driven transform Rust's std
        // library has no built-in equivalent for. Declining rather than
        // approximating with `to_lowercase` (which does NOT casefold:
        // "ß".to_lowercase() stays "ß", not "ss").
        "casefold" => None,
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
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Number));
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

    /// casefold declines: no built-in Rust equivalent of Unicode's full
    /// case-folding table exists to model it exactly.
    #[test]
    fn test_casefold_declines() {
        let receiver = string_literal_value("ab");
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
}
