//! The exact-receiver `str` method dispatch: `string_method_result`,
//! the state a `str` method call answers when the receiver (and, where
//! the method takes one, the argument) is a known exact string. See
//! `super`'s module doc for the per-method library/stdtypes.html
//! citations.

use refined_domain::abstract_value::{known_values, AbstractValue, Kind, PrimitiveKind};
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::TrustProved;

use super::{boolean_value, exact_string_text, string_literal_value};

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
        // The same clause's THREE-argument row: "If the optional
        // argument *count* is given, only the first *count* occurrences
        // are replaced." (str.replace.) A known NON-NEGATIVE Integer
        // count replaces exactly that many leading occurrences. A
        // negative count declines: `str.replace`'s own clause states no
        // meaning for one, and the "all occurrences" reading of `-1`
        // that `str.split`'s own `maxsplit` clause spells out
        // explicitly is not written here for `count`.
        "replace" if arguments.len() == 3 => {
            let old = exact_string_text(&arguments[0])?;
            let new = exact_string_text(&arguments[1])?;
            let count = &arguments[2];
            if count.kind != Kind::Values || count.kind_tag != Some(PrimitiveKind::Integer) || count.values.len() != 1 {
                return None;
            }
            let count = count.values[0] as i64;
            if count < 0 {
                return None;
            }
            Some(string_literal_value(&receiver_text.replacen(&old, &new, count as usize)))
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
        // The same clause's TWO-argument row: "If *maxsplit* is given,
        // at most *maxsplit* splits are done (thus, the list will have
        // at most `maxsplit+1` elements). If *maxsplit* is not
        // specified or `-1`, then there is no limit on the number of
        // splits (all possible splits are made)." (str.split.) The
        // limit caps the SPLIT COUNT, so the remainder after the last
        // split stays whole in the final element — `'1,2,3'.split(',',
        // maxsplit=1)` is `['1', '2,3']`, the doc's own worked example
        // and A3.xfer.split's own claim. A count below `-1` declines:
        // the clause states a meaning for `-1` only.
        "split" if arguments.len() == 2 => {
            let sep = exact_string_text(&arguments[0])?;
            if sep.is_empty() {
                return None;
            }
            let maxsplit = &arguments[1];
            if maxsplit.kind != Kind::Values || maxsplit.kind_tag != Some(PrimitiveKind::Integer) || maxsplit.values.len() != 1 {
                return None;
            }
            let maxsplit = maxsplit.values[0] as i64;
            if maxsplit < -1 {
                return None;
            }
            let parts: Vec<AbstractValue> = if maxsplit == -1 {
                receiver_text.split(&sep).map(string_literal_value).collect()
            } else {
                receiver_text.splitn(maxsplit as usize + 1, &sep).map(string_literal_value).collect()
            };
            Some(known_list(parts, TrustProved))
        }
        // "Return the string left justified in a string of length
        // *width*. Padding is done using the specified *fillchar*
        // (default is an ASCII space). The original string is returned
        // if *width* is less than or equal to `len(s)`." (str.ljust.)
        // Modeled for the two-argument form with a known
        // single-character `fillchar` and a known Integer `width`;
        // `width` counts CODE POINTS, the same measure `len(s)` uses.
        "ljust" if arguments.len() == 2 => {
            let width = &arguments[0];
            if width.kind != Kind::Values || width.kind_tag != Some(PrimitiveKind::Integer) || width.values.len() != 1 {
                return None;
            }
            let width = width.values[0] as i64;
            let fill = exact_string_text(&arguments[1])?;
            let mut fill_characters = fill.chars();
            let (Some(fill_character), None) = (fill_characters.next(), fill_characters.next()) else {
                return None; // str.ljust requires a single-character fillchar
            };
            let length = receiver_text.chars().count() as i64;
            if width <= length {
                return Some(string_literal_value(&receiver_text));
            }
            let mut padded = receiver_text.clone();
            for _ in 0..(width - length) {
                padded.push(fill_character);
            }
            Some(string_literal_value(&padded))
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
        // "Return an encoded version of the string as a bytes object...
        // encoding defaults to 'utf-8'." (library/stdtypes.html,
        // str.encode). The zero-argument, default-encoding form only.
        // An EXACT receiver answers the exact byte sequence: UTF-8 is a
        // total, deterministic encoding of every Unicode scalar value,
        // so the receiver's own code points determine each byte, and
        // the result is the same `Kind::List` of known byte ints a
        // `bytes` literal builds (`bytes_models.rs`'s own module doc).
        // That makes `len(s.encode())` — the BYTE count, distinct from
        // `len(s)`'s code-point count — readable, which is A3.xfer.
        // encode's own claim. `exact_string_text` reads back only code
        // points `char::from_u32` accepts, so every character reaching
        // this row is a Unicode scalar value that UTF-8 encodes.
        "encode" if arguments.is_empty() => {
            let mut bytes: Vec<AbstractValue> = Vec::new();
            for character in receiver_text.chars() {
                let mut buffer = [0u8; 4];
                for byte in character.encode_utf8(&mut buffer).as_bytes() {
                    bytes.push(known_values(vec![*byte as f64], PrimitiveKind::Integer, TrustProved));
                }
            }
            Some(known_list(bytes, TrustProved))
        }
        _ => None,
    }
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
