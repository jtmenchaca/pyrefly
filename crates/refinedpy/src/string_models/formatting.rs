//! `f"{x:.{precision}f}"` fixed-precision decimal formatting: the
//! grammar `format_spec`'s `'f'` presentation type states
//! (`fixed_precision_decimal_grammar`), and the `precision` reader for
//! the plain `.{precision}f` spelling (`fixed_precision_decimal_width`).

use refined_sets::refinement_forms::{concatenation, one_of, repeat_of, RefinedSet};
use refined_sets::codepoint_sets::string_tuple;
use refined_sets::refinement_forms::make_refined_set;

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
    let point = string_tuple(".");
    let fractional_part = repeat_of(one_char_of("0123456789"), precision as i64, Some(precision as i64));
    let signed_integer = concatenation(make_refined_set(vec![sign]), make_refined_set(vec![integer_part]));
    let with_point = concatenation(make_refined_set(vec![signed_integer]), point);
    make_refined_set(vec![concatenation(make_refined_set(vec![with_point]), make_refined_set(vec![fractional_part]))])
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

/// One codepoint drawn from the given ASCII characters — mirrors
/// `expressions.rs::one_char_of`/`json_grammar.rs::one_char_of`, kept as
/// a private copy per this crate's file-scope convention
/// (`json_grammar.rs`'s own doc on its own copy) rather than widening
/// either function's visibility for one caller outside its file.
fn one_char_of(chars: &str) -> RefinedSet {
    let points: Vec<f64> = chars.chars().map(|c| c as u32 as f64).collect();
    make_refined_set(vec![one_of(&points)])
}
