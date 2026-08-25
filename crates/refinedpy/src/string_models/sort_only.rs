//! The SORT-ONLY answers a `str` method call states over a receiver
//! that is STRING-SORTED but not exact: `string_method_sort_only_result`
//! (the `str`-sorted family) and `string_method_int_sort_only_result`
//! (`find`/`index`'s own `int`-sorted family). Includes the ASCII
//! case-mapping window reader `ascii_case_mapped_shaped_result` that
//! lets `upper`/`lower` answer a MAPPED window rather than the
//! unbounded `Σ*` fallback when the receiver is already narrowed to one
//! of the two ASCII cased-letter windows.

use refined_domain::abstract_value::{known_set, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::{trust_level_of, TrustProved};
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set, repeat_of, Form, RefinedSet};
use refined_sets::repetition_window_forms::{as_repetition, repetition};

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
    if method == "zfill" {
        if let Some(result) = zfilled_digit_repetition(receiver, arguments) {
            return Some(result);
        }
    }
    // `s.split(sep)` over an unread receiver — the PIECES are unread,
    // but the piece COUNT is bounded below by 1: "Splitting an empty
    // string with a specified separator returns `['']`"
    // (stdtypes.rst, str.split), and every longer string splits into at
    // least that one piece too. The answer is the unbounded repetition
    // of unread strings whose length window starts at 1, so
    // `collection_models::len_result` reads `len(s.split(","))` as
    // `[1, +inf)` — A3.xfer.split's own `split_length_outside` claim —
    // rather than declining the whole call.
    if method == "split" && (arguments.len() == 1 || arguments.len() == 2) {
        let separator = &arguments[0];
        // the no-argument whitespace form is a different splitting rule
        // (`string_method_result`'s own `split` doc) and a `None`
        // separator is not this row; require a string-sorted separator.
        if separator.kind_tag == Some(PrimitiveKind::String) || separator.kind == Kind::Set {
            let pieces = repetition(strings(), 1, None);
            let grade = trust_level_of(receiver);
            return Some(known_set(pieces, None, grade, SetKindTag::None));
        }
    }
    let is_shaped_row = match method {
        "upper" | "lower" | "strip" | "lstrip" | "rstrip" => arguments.is_empty(),
        "casefold" => arguments.is_empty(),
        // `replace`'s own two- and three-argument rows, and `split`'s
        // own one- and two-argument rows, all still answer a `str` when
        // their exact rows decline for an unread receiver or an unread
        // argument. `split` answers a LIST of them, so it is NOT here —
        // it stays this function's own decline, as its doc states.
        "replace" => arguments.len() == 2 || arguments.len() == 3,
        "zfill" => arguments.len() == 1,
        "ljust" | "rjust" => arguments.len() == 1 || arguments.len() == 2,
        // "Return a string which is the concatenation of the strings in
        // *iterable*. ... The separator between elements is the string
        // providing this method." (stdtypes.rst, str.join.) The result
        // is always a Python `str`, whatever the iterable holds, so the
        // whole-strings ground is the sound claim when
        // `string_method_result`'s own exact row declines — an
        // unread-element list, or a list whose own length is unproven.
        // A NARROWER claim (the alternation-of-elements grammar the
        // joined pieces actually form) needs a kernel decider this file
        // does not have; see `join`'s note in the module doc.
        "join" => arguments.len() == 1,
        _ => false,
    };
    if !is_shaped_row {
        return None;
    }
    let grade = trust_level_of(receiver);
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, grade, SetKindTag::None)
    })
}

/// `x.zfill(width)` over a receiver already narrowed to a REPETITION of
/// the ASCII DIGIT window (`[0x30, 0x39]`) — `Digits`'s own declared
/// grammar `/^[0-9]+$/` and every narrower fixed-length digit window.
///
/// "Return a copy of the string left filled with ASCII '0' digits to
/// make a string of length *width*. A leading sign prefix
/// (`'+'`/`'-'`) is handled by inserting the padding AFTER the sign
/// character rather than before. The original string is returned if
/// *width* is less than or equal to `len(s)`." (library/stdtypes.html,
/// str.zfill.) A digit-only receiver carries NO sign character, so the
/// sign clause never applies and the result is the receiver's own
/// digits with some number of ASCII `'0'` digits prepended — still
/// every code point inside `[0x30, 0x39]`. The LENGTH is the receiver's
/// own length raised to at least `width`: `max(lo, width)` at the low
/// end, and unchanged at the high end when the receiver's own high
/// bound already exceeds `width` (the "original string is returned"
/// clause).
///
/// Modeled for a known Integer `width` only; an unknown or non-Integer
/// width, or a receiver that is not a digit-window repetition, declines
/// to the caller's own `Σ*` fallback.
fn zfilled_digit_repetition(receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [width] = arguments else { return None };
    if width.kind != Kind::Values || width.kind_tag != Some(PrimitiveKind::Integer) || width.values.len() != 1 {
        return None;
    }
    let width = width.values[0] as i64;
    if receiver.kind != Kind::Set {
        return None;
    }
    let repeated = as_repetition(&receiver.set)?;
    let ascii_digits = (0x30 as f64, 0x39 as f64);
    if ascii_case_window_bounds(&repeated.element)? != ascii_digits {
        return None;
    }
    let element = make_refined_set(vec![integer(), at_least(ascii_digits.0), at_most(ascii_digits.1)]);
    let filled = repetition(element, repeated.lo.max(width), repeated.hi.map(|high| high.max(width)));
    let grade = trust_level_of(receiver);
    Some(known_set(filled, None, grade, SetKindTag::None))
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
