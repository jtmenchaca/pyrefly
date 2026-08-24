/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

use refined_domain::abstract_value::{known_set, known_values, AbstractValue, Kind, ObjectKey, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::{known_list, known_object};
use refined_domain::trust_grades::{trust_level_of, TrustProved};
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set, repeat_of, Form, RefinedSet};
use refined_sets::regex_compiler::format_grammar;
use refined_sets::repetition_window_forms::as_repetition;

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
        // "Return an encoded version of the string as a bytes object...
        // encoding defaults to 'utf-8'." (library/stdtypes.html,
        // str.encode). The zero-argument, default-encoding form only.
        // This domain has no per-byte UTF-8 encoding table to walk, so
        // the answer is the opaque "some bytes value" state
        // (`crate::bytes_models::encoded_bytes_value`'s own doc) rather
        // than the exact byte sequence — sufficient for every corpus row
        // that only ever routes an `.encode()` result onward through
        // `base64.b64encode`, which reads no content off its argument
        // either (`bytes_models::b64encode_call`'s own doc).
        "encode" if arguments.is_empty() => Some(crate::bytes_models::encoded_bytes_value()),
        _ => None,
    }
}

/// The SORT-ONLY answer a `str` method call states when the receiver is
/// known to be STRING-SORTED but not an EXACT string (`string_method_
/// result`'s own `exact_string_text` already declined) — the same
/// "state the sort, never a guessed value" posture
/// `math_models::approximated_family_result` keeps for a transcendental
/// call over a known numeric window rather than one known number.
/// Every row here answers `refined_sets::codepoint_sets::strings()`
/// (`Σ*`, the whole-strings ground) tagged `PrimitiveKind::String` —
/// sound regardless of what the receiver's own content actually is,
/// since every one of these methods' own CPython contract (library/
/// stdtypes.html) returns another Python `str`, and this file carries
/// no NARROWER-than-Σ* claim (a length bound, a case-only image) for an
/// unbounded receiver to state exactly.
///
/// Scoped to the methods a corpus row actually needs a value for over
/// an unbounded receiver: `upper`/`lower`/`casefold`/`strip`/`lstrip`/
/// `rstrip`/`replace`/`zfill` (`str.zfill` is not modeled at ALL in
/// `string_method_result` — no exact row exists for it either — so its
/// only citation is here: "Return a copy of the string left filled with
/// ASCII '0' digits to make a string of length width," library/
/// stdtypes.html, str.zfill). `find`/`index` are NOT here: their own
/// result is `int`-sorted, not `str`-sorted, so `find`/`index`'s own
/// sort-only answer is `string_method_int_sort_only_result` below, a
/// separate row rather than a shared one — the two return different
/// Python types and must not share one function's "always strings"
/// answer. Every other method name answers `None`: `startswith`/
/// `endswith` are `bool`-sorted, not `str`- or `int`-sorted, and this
/// file states no sort-only claim for a boolean predicate over an
/// unread receiver; `split` answers a LIST, a third sort again; `join`'s
/// receiver is the SEPARATOR, not the thing being transformed, and no
/// corpus row needs its own sort-only answer here — every one of these
/// is this function's own "not modeled at this precision" decline, same
/// honesty as `string_method_result`'s own catch-all.
///
/// `receiver`'s own trust grade carries onto the answer
/// (`trust_grades::trust_level_of`) — the same grade-preservation
/// `math_models::kernel_backed_unary_family_call` keeps for its own
/// sort-only-adjacent Set answers, so a WORN receiver's own weaker
/// claim never gets restated as `TrustSpec`-strength here.
pub fn string_method_sort_only_result(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    // "Return an encoded version of the string as a bytes object"
    // (str.encode, `string_method_result`'s own exact-receiver row
    // states the identical citation) — the opaque bytes answer needs no
    // receiver CONTENT, only that the receiver is string-sorted, so it
    // is answered here directly rather than folded into the `Σ*`-only
    // shaped-row match below (encode's own result is bytes-sorted, not
    // string-sorted, so it must not share that match's one grade-wrapped
    // `strings()` answer).
    if method == "encode" && arguments.is_empty() {
        return Some(crate::bytes_models::encoded_bytes_value());
    }
    // `upper`/`lower` over a receiver already narrowed to an ASCII
    // cased-letter WINDOW (`narrowing.rs`'s `narrow_ascii_case_
    // conjunction`, the `len(x) == 2 and x.isascii() and x.islower()`
    // shape F2.fixed/F2.dead/F2.select all guard with) answer the
    // MAPPED window, not the unbounded `Σ*` fallback below: `str.upper`/
    // `str.lower`'s own contract (stdtypes.html) maps every code point,
    // and inside the two ASCII cased-letter windows that mapping is the
    // exact, length-preserving, bijective shift `casefold`'s own doc
    // already cites (ASCII has no multi-character or non-1:1 case
    // mapping at all) — so a guard that narrowed `x` to two ASCII
    // lowercase letters lets `x.upper()` answer two ASCII UPPERCASE
    // letters exactly, matching `Code`'s own declared set, rather than
    // discarding the narrowing and falling back to `Σ*`.
    if let (Some(result), true) = (ascii_case_mapped_shaped_result(method, receiver), arguments.is_empty()) {
        return Some(result);
    }
    let is_shaped_row = match method {
        "upper" | "lower" | "strip" | "lstrip" | "rstrip" => arguments.is_empty(),
        "casefold" => arguments.is_empty(),
        "replace" => arguments.len() == 2,
        "zfill" => arguments.len() == 1,
        _ => false,
    };
    if !is_shaped_row {
        return None;
    }
    let grade = trust_level_of(receiver);
    Some(known_set(strings(), None, grade, SetKindTag::None))
}

/// `x.upper()`/`x.lower()` over a receiver `narrowing.rs`'s
/// `narrow_ascii_case_conjunction` already narrowed to a REPETITION of
/// exactly one ASCII cased-letter window (`[0x41, 0x5A]` for
/// `isupper()`, `[0x61, 0x7A]` for `islower()`) at some fixed length —
/// `x.upper()` maps the window to `[0x41, 0x5A]`, `x.lower()` maps it
/// to `[0x61, 0x7A]`, and the length stays exactly what it already was
/// (`str.upper`/`str.lower` never change a string's length: mapping is
/// one Unicode code point to one Unicode code point,
/// library/stdtypes.html — inside ASCII this is additionally BIJECTIVE,
/// so mapping the whole window is exact, not an over-approximation).
///
/// Declines (returns `None`) for every other shape: a receiver that is
/// not a repetition at all (`as_repetition` reads back only the shapes
/// `repetition`/`narrow_ascii_case_conjunction` build), a repetition
/// whose element is not EXACTLY one of the two ASCII cased-letter
/// windows (any wider or narrower element — this row states no image
/// for it, matching `casefold`'s own "ASCII cased letters only" scope
/// one level up: outside these two exact windows, ASCII still has
/// uncased code points this function does not attempt to fix
/// pointwise). The caller's own `Σ*` fallback (`string_method_sort_
/// only_result`'s tail) covers every declined case honestly.
fn ascii_case_mapped_shaped_result(method: &str, receiver: &AbstractValue) -> Option<AbstractValue> {
    if !matches!(method, "upper" | "lower") {
        return None;
    }
    if receiver.kind != Kind::Set {
        return None;
    }
    let repeated = as_repetition(&receiver.set)?;
    let element_window = ascii_case_window_bounds(&repeated.element)?;
    let ascii_upper = (0x41 as f64, 0x5A as f64);
    let ascii_lower = (0x61 as f64, 0x7A as f64);
    // the element must already be exactly one of the two ASCII
    // cased-letter windows — any other element (a wider alphabet, a
    // single code point, a non-cased window) states no image this row
    // answers exactly, and falls through to the caller's own `Σ*`
    // fallback instead.
    if element_window != ascii_upper && element_window != ascii_lower {
        return None;
    }
    let mapped = if method == "upper" { ascii_upper } else { ascii_lower };
    let mapped_element = make_refined_set(vec![integer(), at_least(mapped.0), at_most(mapped.1)]);
    let mapped_set = make_refined_set(vec![repeat_of(mapped_element, repeated.lo, repeated.hi)]);
    let grade = trust_level_of(receiver);
    Some(known_set(mapped_set, None, grade, SetKindTag::None))
}

/// The `(lo, hi)` bounds of `element` if it is EXACTLY the one-tuple
/// set `{integer(), at_least(lo), at_most(hi)}` — the shape
/// `narrow_ascii_case_conjunction` builds for its own ASCII
/// cased-letter window (`narrowing.rs`). Any other form composition
/// (extra forms, missing bounds, a non-Integer element) answers `None`
/// — this reader states no window for a shape it was not built to
/// read, matching `as_repetition`'s own "only the shapes I know" scope.
fn ascii_case_window_bounds(element: &RefinedSet) -> Option<(f64, f64)> {
    if element.forms.len() != 3 {
        return None;
    }
    let mut has_integer = false;
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &element.forms {
        match form.form {
            Form::Integer => has_integer = true,
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost => hi = Some(form.a),
            _ => return None,
        }
    }
    if !has_integer {
        return None;
    }
    Some((lo?, hi?))
}

/// The SORT-ONLY answer `str.find`/`str.index` state over a receiver
/// known to be STRING-SORTED but not exact: every real call answers an
/// Integer, never wider than `[-1, +inf)` (`str.find`'s own contract,
/// `string_method_result`'s own doc — "Return -1 if sub is not
/// found"). `str.index` never actually returns `-1` (a miss raises
/// instead), but `[-1, +inf)` still SOUNDLY bounds it — a superset of
/// the true `[0, +inf)` answer costs this file nothing it needs to
/// state exactly here, and keeping the two methods on one row (rather
/// than a tighter, `index`-only `[0, +inf)` claim) matches `find`'s own
/// exact row, which likewise never distinguishes the two beyond the
/// miss case `expressions.rs::call_provable_raise` already reads
/// separately.
pub fn string_method_int_sort_only_result(method: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if !matches!(method, "find" | "index") || arguments.len() != 1 {
        return None;
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![integer(), at_least(-1.0)]), None, TrustProved, SetKindTag::None)
    })
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

/// The word a Match-object value built by `match_object_value` carries
/// on `kind_word` — distinct from `evaluate_attribute_call`'s existing
/// bare `"a match object"` opaque tag (`re.match`/`re.search`'s own
/// contentless answer), since THIS value additionally carries readable
/// group grammars a caller's `.group(n)` needs to find. `expressions.rs`
/// reads this word to route a `.group(...)` call through
/// `matched_group_grammar` below rather than the opaque-value default.
pub const MATCH_WITH_GROUPS_WORD: &str = "a match object with readable groups";

/// The top-level PARENTHESIZED groups of a regex pattern, each group's
/// own inner text (no enclosing parens), in left-to-right opening order
/// — `re.fullmatch(pattern, s)`'s own capture-group numbering
/// (library/re.html, "Group 0 is the entire match... groups are
/// numbered from 1 in the order their opening parentheses appear").
/// Recognizes only the shapes the corpus's own patterns and
/// `format_grammar`'s own supported subset both need: plain capturing
/// groups `(...)`, with `\(`/`\)` escapes and NESTED parens read as
/// plain text ONE LEVEL DEEP (a nested group inside a captured group is
/// not itself extracted as a separate numbered group — this reader
/// finds no corpus row needing that). A non-capturing group `(?:...)`
/// is recognized and its own parens are consumed but it contributes NO
/// numbered group — `re.rst`'s own "the contents of a group ... `(?:...)`
/// ... cannot be retrieved". An unmatched paren, or a `(?...)` extension
/// other than `(?:`, makes the WHOLE read decline (`None`) — this is
/// not a general regex parser, only enough to find each `(...)`'s own
/// span in the corpus's own literal patterns.
fn capture_group_spans(pattern: &str) -> Option<Vec<String>> {
    let chars: Vec<char> = pattern.chars().collect();
    let mut groups = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\\' => {
                i += 2; // an escaped character is never a group boundary
            }
            '(' => {
                let is_non_capturing = chars.get(i + 1) == Some(&'?') && chars.get(i + 2) == Some(&':');
                if chars.get(i + 1) == Some(&'?') && !is_non_capturing {
                    return None; // an unsupported (?...) extension
                }
                let body_start = if is_non_capturing { i + 3 } else { i + 1 };
                let mut depth = 1;
                let mut j = body_start;
                while j < chars.len() && depth > 0 {
                    match chars[j] {
                        '\\' => j += 1,
                        '(' => depth += 1,
                        ')' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth != 0 {
                    return None; // an unmatched opening paren
                }
                let body_end = j - 1;
                if !is_non_capturing {
                    groups.push(chars[body_start..body_end].iter().collect());
                }
                i = j;
            }
            _ => i += 1,
        }
    }
    Some(groups)
}

/// The Match-object value `re.fullmatch(pattern, subject)` /
/// `re.finditer(pattern, subject)`'s own yielded match answer —
/// library/re.html: `fullmatch(pattern, string)` "If the whole string
/// matches this regular expression, return a corresponding match
/// object"; `finditer(pattern, string)` "Return an iterator yielding
/// match objects." A `Kind::Object` (`MATCH_WITH_GROUPS_WORD`-tagged)
/// whose keys are `"0"` (the whole match, ALWAYS present — "group()"
/// with no argument or `group(0)` "The entire match") through `"N"`
/// (each capturing group, in `capture_group_spans`'s own left-to-right
/// numbering), every key's value the group's OWN compiled grammar,
/// UNANCHORED (`format_grammar(text, "")` with no `^`/`$` inserted —
/// library/re.html, "group()... Returns one or more subgroups of the
/// match" is the group's OWN matched substring, which carries no
/// anchor semantics of its own; `fullmatch`'s whole-pattern anchoring
/// only pins the OUTER match, not what a captured GROUP's own text
/// looks like). `anchor_whole_match` anchors group `"0"` ONLY, matching
/// each caller's own semantics (`fullmatch`: both ends; `finditer`: an
/// unanchored single iteration's match, so `false` — a global match is
/// itself an arbitrary substring, `narrow_regex_module_call`'s own
/// `search`-neither-anchor row states the identical unanchored-both-
/// ends default `format_grammar` itself pads with `C*`). `None` on a
/// pattern `capture_group_spans` cannot read, or a compiled grammar
/// `format_grammar` refuses for group 0 or any numbered group — the
/// WHOLE match value declines rather than answer a partial object
/// missing some groups.
pub fn match_object_value(pattern: &str, anchor_whole_match: bool) -> Option<AbstractValue> {
    let mut whole = pattern.to_owned();
    if anchor_whole_match {
        if !whole.starts_with('^') {
            whole.insert(0, '^');
        }
        if !(whole.ends_with('$') && !whole.ends_with("\\$")) {
            whole.push('$');
        }
    }
    let whole_grammar = format_grammar(&whole, "");
    if !whole_grammar.ok {
        return None;
    }
    let groups = capture_group_spans(pattern)?;
    let mut keys = vec![ObjectKey {
        name: "0".to_owned(),
        numeric: true,
        value: AbstractValue {
            kind_tag: Some(PrimitiveKind::String),
            ..known_set(whole_grammar.set, None, TrustProved, SetKindTag::None)
        },
    }];
    for (index, group_text) in groups.iter().enumerate() {
        let compiled = format_grammar(group_text, "");
        if !compiled.ok {
            return None;
        }
        keys.push(ObjectKey {
            name: (index + 1).to_string(),
            numeric: true,
            value: AbstractValue {
                kind_tag: Some(PrimitiveKind::String),
                ..known_set(compiled.set, None, TrustProved, SetKindTag::None)
            },
        });
    }
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.kind_word = Some(MATCH_WITH_GROUPS_WORD);
    Some(instance)
}

/// `match.group(n)` (one-argument, known-Integer form) over a
/// `match_object_value`-built receiver — library/re.html#re.Match.group:
/// "If a single argument is used, result is a single string." Group `0`
/// (or the no-argument default, `group()`'s own zero-arg row — not this
/// function, which only ever reads the one-argument numeric form) is
/// the whole match; group `N` (`N >= 1`) is that numbered capturing
/// group's own compiled grammar, `match_object_value`'s own key
/// layout. A group number this match's own value carries no key for
/// (out of range for the pattern's own group count) declines — CPython
/// raises `IndexError: no such group` for it, a fact this row does not
/// itself speak the raise for (no exception channel in this file);
/// a non-Match receiver, or a non-Integer/unknown argument, declines
/// the same way.
pub fn matched_group_grammar(receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if receiver.kind != Kind::Object || receiver.kind_word != Some(MATCH_WITH_GROUPS_WORD) {
        return None;
    }
    let [group_number] = arguments else { return None };
    if group_number.kind != Kind::Values || group_number.kind_tag != Some(PrimitiveKind::Integer) || group_number.values.len() != 1 {
        return None;
    }
    let name = format!("{}", group_number.values[0] as i64);
    receiver.keys.iter().find(|key| key.numeric && key.name == name).map(|key| key.value.clone())
}

/// One codepoint drawn from the given ASCII characters — mirrors
/// `expressions.rs::one_char_of`/`json_grammar.rs::one_char_of`, kept as
/// a private copy per this crate's file-scope convention
/// (`json_grammar.rs`'s own doc on its own copy) rather than widening
/// either function's visibility for one caller outside its file.
fn one_char_of(chars: &str) -> RefinedSet {
    let points: Vec<f64> = chars.chars().map(|c| c as u32 as f64).collect();
    make_refined_set(vec![refined_sets::refinement_forms::one_of(&points)])
}

/// `f"{x:.{precision}f}"` — the fixed-precision decimal grammar
/// format_spec's own `'f'` presentation type states (library/string.rst,
/// "Format examples" table, type `'f'`: "Fixed-point notation. For a
/// given precision p, formats the number as a decimal number with
/// exactly p digits following the decimal point"). This is a SOUND
/// OVER-APPROXIMATION over every finite float, not a value-exact window
/// (the same posture `json_grammar.rs::integer_window_grammar` takes for
/// its own digit-count bound): an optional leading `-` sign (CPython
/// never emits a leading `+` here — `format_spec.rst`'s own `sign`
/// option defaults to `-`-only, and this row does not model an explicit
/// `+`/` ` sign flag), one-or-more integer-part digits (unbounded above,
/// since the fixed argument's own magnitude is not read here), a literal
/// `.`, then EXACTLY `precision` fractional digits — never fewer, never
/// more, the clause's own "exactly p digits" reading. Every digit drawn
/// from the plain `0-9` alphabet (`one_char_of`, mirroring
/// `json_grammar.rs`'s copy) — no grouping separator, since this row
/// does not model the `,`/`_` grouping option.
pub fn fixed_precision_decimal_grammar(precision: u32) -> RefinedSet {
    let sign = repeat_of(one_char_of("-"), 0, Some(1));
    let integer_part = repeat_of(one_char_of("0123456789"), 1, None);
    let point = refined_sets::codepoint_sets::string_tuple(".");
    let fractional_part = repeat_of(one_char_of("0123456789"), precision as i64, Some(precision as i64));
    let signed_integer =
        refined_sets::refinement_forms::concatenation(make_refined_set(vec![sign]), make_refined_set(vec![integer_part]));
    let with_point = refined_sets::refinement_forms::concatenation(make_refined_set(vec![signed_integer]), point);
    make_refined_set(vec![refined_sets::refinement_forms::concatenation(
        make_refined_set(vec![with_point]),
        make_refined_set(vec![fractional_part]),
    )])
}

/// The `precision` a format spec states, when the spec is EXACTLY the
/// plain `.{precision}f` spelling (no fill/align/sign/`#`/`0`/width/
/// grouping option, `type` exactly `f`) — the fixed-point counterpart of
/// `expressions.rs::zero_padded_decimal_width`'s own `0{width}d` reader,
/// same single-literal-element, no-nested-interpolation restriction.
pub fn fixed_precision_decimal_width(format_spec: &ruff_python_ast::InterpolatedStringFormatSpec) -> Option<u32> {
    let [ruff_python_ast::InterpolatedStringElement::Literal(literal)] = &*format_spec.elements else {
        return None;
    };
    let digits = literal.value.strip_prefix('.')?.strip_suffix('f')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
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

    /// `fixed_precision_decimal_grammar(2)` composes as ONE top-level
    /// `Concatenation` form (`concatenation`'s own construction — the
    /// same "one Concatenation, nested" shape `codepoint_sets::string_tuple`
    /// builds), and different precisions build DIFFERENT grammars — the
    /// `precision` parameter must actually reach the fractional-digit
    /// repeat bound, not be silently ignored.
    #[test]
    fn test_fixed_precision_decimal_grammar_varies_with_precision() {
        let two_digits = fixed_precision_decimal_grammar(2);
        assert_eq!(two_digits.forms.len(), 1);
        assert!(matches!(two_digits.forms[0].form, Form::Concatenation));
        let three_digits = fixed_precision_decimal_grammar(3);
        assert_ne!(two_digits, three_digits, "a different precision must build a different grammar");
    }

    /// `.2f` parses as precision `2`; `02d` (a DIFFERENT reader's own
    /// spelling, `zero_padded_decimal_width`'s row) and a fill/align
    /// spec (`^10`) are not this reader's grammar at all.
    #[test]
    fn test_fixed_precision_decimal_width_reads_the_plain_dot_f_spelling() {
        let source = "f\"{x:.2f}\"";
        let parsed = ruff_python_parser::parse_expression(source).expect("test source must parse");
        let ruff_python_ast::Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
        let single = fstring.as_single_part_fstring().expect("single-part f-string");
        let [ruff_python_ast::InterpolatedStringElement::Interpolation(interpolation)] = &*single.elements else {
            panic!("expected one interpolation")
        };
        let format_spec = interpolation.format_spec.as_ref().expect("format spec present");
        assert_eq!(fixed_precision_decimal_width(format_spec), Some(2));
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

    /// The unbounded whole-strings ground — `s: str`'s own seed
    /// (`typereading::base_sort_return_refinement`) — as this test
    /// module's own Set-shaped receiver.
    fn any_string_receiver() -> AbstractValue {
        known_set(strings(), None, TrustProved, SetKindTag::None)
    }

    /// `s.upper()` over an unbounded receiver answers Σ* — `string_
    /// method_result`'s own exact row already declined (no exact text to
    /// read), so this is the sort-only fallback `A3.xfer.case`'s own row
    /// needs: the method still names a real `str`-sorted claim rather
    /// than declining the whole call.
    #[test]
    fn test_sort_only_upper_over_an_unbounded_receiver_answers_any_string() {
        let receiver = any_string_receiver();
        let result = string_method_sort_only_result("upper", &receiver, &[]).expect("upper must decide the sort");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(exact_string_text(&result), None, "the answer states no exact content");
    }

    /// F2.fixed/F2.dead/F2.select's own shape: `len(x) == 2 and
    /// x.isascii() and x.islower()` narrows `x` to a length-2 repetition
    /// of the ASCII lowercase window (`narrowing.rs`'s
    /// `narrow_ascii_case_conjunction`, `[0x61, 0x7A]`) — `x.upper()`
    /// over THAT receiver must answer the mapped ASCII UPPERCASE window
    /// at the SAME length, not the unbounded `Σ*` fallback: the mapped
    /// answer is exactly `Code`'s own declared set
    /// (`(>= 65 && <= 90 && integer) × exactly 2`), so this is the
    /// difference between a determined pass and the RTS7001 mismatch
    /// those three rows previously answered.
    fn ascii_lowercase_pair_receiver() -> AbstractValue {
        let element = make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]);
        let set = make_refined_set(vec![repeat_of(element, 2, Some(2))]);
        known_set(set, None, TrustProved, SetKindTag::None)
    }

    #[test]
    fn test_upper_over_a_narrowed_ascii_lowercase_pair_answers_the_mapped_uppercase_window() {
        let receiver = ascii_lowercase_pair_receiver();
        let result = string_method_sort_only_result("upper", &receiver, &[]).expect("upper must decide the mapped window");
        let expected_element = make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]);
        let expected = make_refined_set(vec![repeat_of(expected_element, 2, Some(2))]);
        assert_eq!(result.set, expected, "x.upper() must map the ASCII window, keeping the length-2 bound");
    }

    /// The lower-case twin: an ASCII UPPERCASE pair's `.lower()` maps to
    /// the lowercase window, same length-2 bound preserved.
    #[test]
    fn test_lower_over_a_narrowed_ascii_uppercase_pair_answers_the_mapped_lowercase_window() {
        let element = make_refined_set(vec![integer(), at_least(0x41 as f64), at_most(0x5A as f64)]);
        let set = make_refined_set(vec![repeat_of(element, 2, Some(2))]);
        let receiver = known_set(set, None, TrustProved, SetKindTag::None);
        let result = string_method_sort_only_result("lower", &receiver, &[]).expect("lower must decide the mapped window");
        let expected_element = make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]);
        let expected = make_refined_set(vec![repeat_of(expected_element, 2, Some(2))]);
        assert_eq!(result.set, expected, "x.lower() must map the ASCII window, keeping the length-2 bound");
    }

    /// A receiver narrowed to a window OTHER than the two ASCII
    /// cased-letter windows (e.g. digits) states no mapped image — this
    /// row declines to the caller's own `Σ*` fallback rather than
    /// guessing a case mapping for uncased code points.
    #[test]
    fn test_upper_over_a_non_cased_window_falls_back_to_any_string() {
        let element = make_refined_set(vec![integer(), at_least(0x30 as f64), at_most(0x39 as f64)]);
        let set = make_refined_set(vec![repeat_of(element, 2, Some(2))]);
        let receiver = known_set(set, None, TrustProved, SetKindTag::None);
        let result = string_method_sort_only_result("upper", &receiver, &[]).expect("upper must still decide the Σ* fallback");
        assert_eq!(result.set, strings(), "a non-cased window's own .upper() falls back to Σ*, not a guessed mapping");
    }

    /// `s.replace("a", "b")`/`s.strip()`/`s.zfill(4)` over an unbounded
    /// receiver all answer the same Σ* sort-only claim — `A3.xfer.
    /// replace`/`A3.xfer.trim`/`A3.xfer.pad`'s own rows.
    #[test]
    fn test_sort_only_replace_strip_zfill_all_answer_any_string() {
        let receiver = any_string_receiver();
        let replace = string_method_sort_only_result("replace", &receiver, &[string_literal_value("a"), string_literal_value("b")])
            .expect("replace must decide the sort");
        assert_eq!(replace.kind, Kind::Set);
        let strip = string_method_sort_only_result("strip", &receiver, &[]).expect("strip must decide the sort");
        assert_eq!(strip.kind, Kind::Set);
        let zfill = string_method_sort_only_result("zfill", &receiver, &[known_values(vec![4.0], PrimitiveKind::Integer, TrustProved)])
            .expect("zfill must decide the sort");
        assert_eq!(zfill.kind, Kind::Set);
    }

    /// A method this file states no sort-only claim for (`split` answers
    /// a LIST, not a string — a different sort this function does not
    /// speak to) still declines, matching `string_method_result`'s own
    /// "not modeled" honesty at this precision.
    #[test]
    fn test_sort_only_declines_a_method_with_no_string_sorted_claim() {
        let receiver = any_string_receiver();
        assert_eq!(string_method_sort_only_result("split", &receiver, &[string_literal_value(",")]), None);
    }

    /// `s.find("z")` over an unbounded receiver answers an Integer-sorted
    /// `[-1, +inf)` claim — `A3.xfer.search`'s own row: `find` never
    /// raises, so this sound bound is the whole real answer.
    #[test]
    fn test_sort_only_find_over_an_unbounded_receiver_answers_an_integer_ray() {
        let result = string_method_int_sort_only_result("find", &[string_literal_value("z")]).expect("find must decide the sort");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
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

    #[test]
    fn test_encode_on_an_exact_receiver_answers_the_opaque_bytes_state() {
        let receiver = string_literal_value("ab");
        let got = string_method_result("encode", &receiver, &[]).expect("encode must decide");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.kind_word, Some(crate::bytes_models::ENCODED_BYTES_WORD));
    }

    #[test]
    fn test_encode_sort_only_on_an_unread_receiver_answers_the_opaque_bytes_state() {
        let receiver = known_set(strings(), None, TrustProved, SetKindTag::None);
        let got = string_method_sort_only_result("encode", &receiver, &[]).expect("encode must decide sort-only");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.kind_word, Some(crate::bytes_models::ENCODED_BYTES_WORD));
    }

    #[test]
    fn test_encode_with_an_argument_declines() {
        let receiver = string_literal_value("ab");
        let encoding = string_literal_value("utf-8");
        assert_eq!(string_method_result("encode", &receiver, &[encoding]), None);
    }

    #[test]
    fn capture_group_spans_reads_two_plain_groups_in_order() {
        let got = capture_group_spans(r"(\d+)-(\d+)").expect("two plain groups parse");
        assert_eq!(got, vec![r"\d+".to_owned(), r"\d+".to_owned()]);
    }

    #[test]
    fn capture_group_spans_skips_a_non_capturing_group() {
        let got = capture_group_spans(r"(?:\d+)-([a-z]+)").expect("one capturing group parses");
        assert_eq!(got, vec!["[a-z]+".to_owned()]);
    }

    #[test]
    fn capture_group_spans_answers_empty_for_a_group_free_pattern() {
        let got = capture_group_spans(r"\d+").expect("a group-free pattern parses");
        assert!(got.is_empty());
    }

    #[test]
    fn capture_group_spans_declines_on_an_unmatched_paren() {
        assert!(capture_group_spans(r"(\d+").is_none());
    }

    #[test]
    fn match_object_value_for_fullmatch_carries_group_0_and_every_numbered_group() {
        let got = match_object_value(r"(\d+)-(\d+)", true).expect("(\\d+)-(\\d+) compiles");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.kind_word, Some(MATCH_WITH_GROUPS_WORD));
        assert_eq!(got.keys.len(), 3);
        assert!(got.keys.iter().any(|k| k.name == "0"));
        assert!(got.keys.iter().any(|k| k.name == "1"));
        assert!(got.keys.iter().any(|k| k.name == "2"));
    }

    #[test]
    fn match_object_value_for_finditer_group_0_is_the_unanchored_whole_pattern() {
        // A3.xfer.matchall's own shape: re.finditer(r"\d+", s), reading
        // m.group(0) — no capturing groups at all, only group 0.
        let got = match_object_value(r"\d+", false).expect(r"\d+ compiles");
        assert_eq!(got.keys.len(), 1);
        assert_eq!(got.keys[0].name, "0");
    }

    #[test]
    fn matched_group_grammar_reads_the_numbered_group_by_known_integer_argument() {
        let receiver = match_object_value(r"(\d+)-(\d+)", true).expect("compiles");
        let group_one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
        let got = matched_group_grammar(&receiver, &[group_one]).expect("group(1) must decide");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
    }

    #[test]
    fn matched_group_grammar_out_of_range_declines() {
        let receiver = match_object_value(r"\d+", false).expect("compiles");
        let group_five = known_values(vec![5.0], PrimitiveKind::Integer, TrustProved);
        let got = matched_group_grammar(&receiver, &[group_five]);
        assert!(got.is_none(), "group(5) on a pattern with no such group should decline: {got:?}");
    }

    #[test]
    fn matched_group_grammar_on_a_non_match_receiver_declines() {
        let receiver = string_literal_value("not a match");
        let group_zero = known_values(vec![0.0], PrimitiveKind::Integer, TrustProved);
        assert!(matched_group_grammar(&receiver, &[group_zero]).is_none());
    }
}
