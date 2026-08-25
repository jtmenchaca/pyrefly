
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::repeat_of;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::ConversionFlag;
use ruff_python_ast::InterpolatedStringElement;

use crate::env::Environment;
use crate::narrowing;
use crate::string_models;

use super::evaluate_expression;
use super::arithmetic::*;
use super::compare::*;

/// `f"...{expr}..."` composes the literal text and each interpolation's
/// contribution, in source order (expressions.rst, "Formatted string
/// literals"). Only the plainest interpolation shape is modeled: no
/// conversion (`!s`/`!r`/`!a`) and no format spec (`:...`) — either one
/// changes the spelling in ways this file does not compute exactly, so
/// their presence declines the WHOLE f-string rather than composing a
/// partially-wrong string. Three tiers, mirroring refined-ts-go's
/// `evaluateTemplate` (walk/literal_values.go): when every interpolation
/// is EXACTLY readable (a known string, a single known Integer-sorted
/// value spelled bare, or a single known Float-sorted value spelled via
/// `format_py_number`), the whole f-string is one exact string, as
/// before this wave. The moment one interpolation is instead a known SET
/// — sort-only, no exact value (a same-module call's declined-body
/// `summaries::return_sort_fallback`, `float_sorted_unknown()`, or a
/// compiled `Label`-shaped string alias) — the f-string steps down to a
/// PATTERN: every part (literal text, an exact interpolation's spelling,
/// or a set interpolation's own admitted spellings) is a `RefinedSet`,
/// folded by `refinement_forms::concatenation` right to left into one
/// set the checker can still judge a declared max-length against. An
/// interpolation with NO readable value at all (`Kind::Null` — an unread
/// same-module call whose own summary answers nothing, e.g. an
/// ellipsis-only body, `summaries::return_sort_fallback`'s own doc — or
/// any other shape none of the readers above accept) still contributes
/// to the pattern rather than losing the whole f-string: CPython's own
/// `str()` of whatever the interpolation holds at runtime is SOME
/// string, so it folds in the unbounded top-string ground `strings()` —
/// the same encoding a bare `str` annotation seeds
/// (`check.rs::seed_parameters`) — exactly as a set interpolation's own
/// spelling folds. b-body-expressions.py's own
/// `fstring_unread_substitution` (`f"n={unread_number()}"` against
/// `Label`, max_length=8) is this row: `unread_number`'s ellipsis-only
/// body answers `Kind::Null`, the fold reaches `concatenation("n=",
/// strings())`, and `seq_subset` DECIDES that against `Label`'s
/// max-length window (assignability.rs's own
/// `an_unbounded_string_set_against_a_max_length_window_fires_containment`
/// pin) — a decided containment refutation, not a kernel refusal, so the
/// row fires. `unknown()` remains the answer only for the two declines
/// above this tier (a conversion or format-spec interpolation) and for
/// an implicitly concatenated f-string (`f"a" f"b"`, not modeled — only
/// the single-part form `as_single_part_fstring` is read).
pub(super) fn evaluate_fstring(fstring: &ruff_python_ast::ExprFString, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(single) = fstring.as_single_part_fstring() else {
        return unknown();
    };
    let mut composed = String::new();
    let mut has_exact = true;
    let mut parts: Vec<RefinedSet> = Vec::new();
    let mut grade = TrustProved;
    for element in &single.elements {
        match element {
            InterpolatedStringElement::Literal(literal) => {
                if has_exact {
                    composed.push_str(&literal.value);
                }
                if !literal.value.is_empty() {
                    parts.push(refined_sets::codepoint_sets::string_tuple(&literal.value));
                }
            }
            InterpolatedStringElement::Interpolation(interpolation) => {
                if interpolation.conversion != ConversionFlag::None {
                    return unknown();
                }
                if let Some(format_spec) = &interpolation.format_spec {
                    let value = evaluate_expression(&interpolation.expression, environment, kernel);
                    let part = zero_padded_decimal_spelling(format_spec, &value)
                        .or_else(|| fixed_precision_decimal_spelling(format_spec, &value));
                    let Some(part) = part else {
                        return unknown();
                    };
                    has_exact = false;
                    grade = refined_domain::trust_grades::min_trust_level(grade, TrustSpec);
                    parts.push(part);
                    continue;
                }
                let value = evaluate_expression(&interpolation.expression, environment, kernel);
                if let Some(text) = exact_string_values(&value) {
                    let Some(text) = code_points_to_string(text) else {
                        return unknown();
                    };
                    if has_exact {
                        composed.push_str(&text);
                    }
                    parts.push(refined_sets::codepoint_sets::string_tuple(&text));
                } else if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(&value) {
                    let spelling = format_integer_spelling(number);
                    if has_exact {
                        composed.push_str(&spelling);
                    }
                    parts.push(refined_sets::codepoint_sets::string_tuple(&spelling));
                } else if let Some((number, PrimitiveKind::Float)) = single_numeric_value(&value) {
                    let spelling = refined_sets::format_string_shapes::format_py_number(number, true);
                    if has_exact {
                        composed.push_str(&spelling);
                    }
                    parts.push(refined_sets::codepoint_sets::string_tuple(&spelling));
                } else if let Some(part) = spellings_of_known_set(&value) {
                    // a sort-only SET (no exact value): the exact-string
                    // composition can no longer track one spelling, so the
                    // f-string steps down to the pattern tier from here on
                    has_exact = false;
                    grade = refined_domain::trust_grades::min_trust_level(grade, TrustSpec);
                    parts.push(part);
                } else {
                    // NO readable value at all (`Kind::Null` — an unread
                    // same-module call whose own summary answers nothing,
                    // `summaries::return_sort_fallback`'s own doc — or any
                    // other shape none of the readers above accept):
                    // CPython's own `str()` of whatever this interpolation
                    // holds at runtime is SOME string, so this contributes
                    // the unbounded top-string ground `strings()` — the
                    // same encoding a bare `str` annotation seeds
                    // (`check.rs::seed_parameters`) and `__name__`'s own
                    // read already carries — rather than lose the whole
                    // f-string to `unknown()`. Untagged (`kind_tag: None`),
                    // the same String/None convention
                    // `spellings_of_known_set` folds through its own
                    // `set_kind_tag == SetKindTag::None` arm. `seq_subset`
                    // DECIDES an unbounded `strings()` part against a
                    // declared max-length window (assignability.rs's own
                    // `an_unbounded_string_set_against_a_max_length_window_
                    // fires_containment` pin: the kernel's sequence-
                    // containment decider proves `strings() ⊄
                    // repeat_of(codepoints(), 0, Some(n))`, firing the
                    // containment-refutation message) — this is not a
                    // shape the kernel refuses today, so composing it here
                    // reaches a real fire rather than staying silent.
                    has_exact = false;
                    grade = refined_domain::trust_grades::min_trust_level(grade, TrustSpec);
                    parts.push(strings());
                }
            }
        }
    }
    if has_exact {
        return string_models::string_literal_value(&composed);
    }
    let Some(mut folded) = parts.pop() else {
        return string_models::string_literal_value("");
    };
    while let Some(part) = parts.pop() {
        folded = make_refined_set(vec![refined_sets::refinement_forms::concatenation(part, folded)]);
    }
    AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(folded, None, grade, SetKindTag::None)
    }
}

/// `f"{year:04d}"` — an interpolation carrying a ZERO-PADDED DECIMAL
/// format spec (`format_spec.rst`, "Format Specification Mini-Language":
/// `[[fill]align][sign][z][#][0][width][grouping_option][.precision][type]`
/// — this reader recognizes only the plain `0{width}d` spelling: no
/// fill/align/sign/`#`/grouping/precision, `type` exactly `d`). `value`
/// need not be a single known integer — this is the row that fires for
/// a BOUNDED Integer-sorted set (`year: Annotated[int, Field(ge=1970,
/// le=9999)]` seeds `Kind::Set`, never `Kind::Values` —
/// `check.rs::seed_parameters`'s scalar-declared-set arm), which
/// `single_numeric_value`'s exact-value row above never reaches. Exact
/// only when EVERY integer in the set's own closed range needs no
/// padding at all: `min_digit_count`/`max_digit_count` (the decimal
/// digit count of the range's two ends — the monotone extremes, since a
/// wider magnitude never has FEWER digits) both equal `width` exactly,
/// so the zero-fill never actually adds a digit and the plain decimal
/// alphabet is the exact spelling set either way. A range that would
/// need real padding for some members but not others (`ge=8, le=12`
/// against `02d`: "08".."12", where padding does fire) declines rather
/// than approximate — this row states only the sub-case where padding
/// is provably a no-op. `RefinedSet` is a `Repeat` over the plain digit
/// alphabet at EXACTLY `width` positions — a stronger claim than
/// `int_spelling_set`'s own unbounded-length superset, and exact for
/// this admitted case since every member has exactly `width` digits and
/// carries no sign (the range's own `lo` is checked non-negative below).
pub(super) fn zero_padded_decimal_spelling(
    format_spec: &ruff_python_ast::InterpolatedStringFormatSpec,
    value: &AbstractValue,
) -> Option<RefinedSet> {
    let width = zero_padded_decimal_width(format_spec)?;
    let (lo, hi) = integer_set_bounds(value)?;
    if lo < 0 {
        return None;
    }
    if decimal_digit_count(lo) != width || decimal_digit_count(hi) != width {
        return None;
    }
    Some(make_refined_set(vec![repeat_of(one_char_of("0123456789"), width as i64, Some(width as i64))]))
}

/// `f"{x:.2f}"` — an interpolation carrying a FIXED-PRECISION format
/// spec (`format_spec.rst`'s `'f'` presentation type: "formats the
/// number as a decimal number with exactly p digits following the
/// decimal point"), read only when `format_spec` is EXACTLY the plain
/// `.{precision}f` spelling (`string_models::fixed_precision_decimal_width`'s
/// own doc — no fill/align/sign/`#`/`0`/width/grouping option) and
/// `value` is NUMERIC-sorted (Integer or Float — `single_numeric_value`'s
/// own sort read, widened to accept a Set-shaped numeric operand too,
/// never only a known single value: the grammar built
/// (`string_models::fixed_precision_decimal_grammar`) is value-
/// independent, sound over every finite float regardless of `value`'s
/// own bound). A non-numeric-sorted `value` declines — CPython's `'f'`
/// type raises `TypeError` on one, and this file has no exception
/// channel to speak that raise through, matching every other row's
/// "known operands only" discipline.
pub(super) fn fixed_precision_decimal_spelling(
    format_spec: &ruff_python_ast::InterpolatedStringFormatSpec,
    value: &AbstractValue,
) -> Option<RefinedSet> {
    let precision = string_models::fixed_precision_decimal_width(format_spec)?;
    if !matches!(value.kind_tag, Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean)) {
        return None;
    }
    Some(string_models::fixed_precision_decimal_grammar(precision))
}

/// The `width` a format spec states, when the spec is EXACTLY the plain
/// `0{width}d` spelling this reader recognizes — a single literal
/// element (no nested interpolation inside the spec itself, which
/// `format_spec.rst` allows but this reader does not model) whose text
/// is `0` followed by one or more digits followed by `d`. Any other
/// spelling (a fill/align/sign/`#`/grouping/precision character, a
/// different `type`, a spec with its own interpolation) answers `None`.
pub(super) fn zero_padded_decimal_width(format_spec: &ruff_python_ast::InterpolatedStringFormatSpec) -> Option<u32> {
    let [InterpolatedStringElement::Literal(literal)] = &*format_spec.elements else {
        return None;
    };
    let digits = literal.value.strip_prefix('0')?.strip_suffix('d')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The closed integer bound `[lo, hi]` a value states, when the value is
/// a BOUNDED Integer-sorted `Kind::Set` (`seed_parameters`'s scalar
/// arm — never `Kind::Values`, which `single_numeric_value` already
/// reads exactly). Reads the set's own top-level `AtLeast`/`Above`/
/// `AtMost`/`Below` forms, the same syntactic hull
/// `collection_models::integer_range_bounds` reads for its own bounded-
/// index subscript read — duplicated here rather than exported, since
/// the two files' own AGENT-BRIEF scope (`collection_models.rs`'s
/// container reads; this file's expression evaluation) keeps neither
/// reaching into the other's private helpers, the same convention
/// `string_models.rs`'s own `exact_string_text` doc states for this
/// exact situation.
pub(super) fn integer_set_bounds(value: &AbstractValue) -> Option<(i64, i64)> {
    if value.kind != Kind::Set || value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &value.set.forms {
        match form.form {
            refined_sets::refinement_forms::Form::AtLeast => {
                lo = Some(lo.map_or(form.a, |current: f64| current.max(form.a)))
            }
            refined_sets::refinement_forms::Form::Above => {
                lo = Some(lo.map_or(form.a.floor() + 1.0, |current: f64| current.max(form.a.floor() + 1.0)))
            }
            refined_sets::refinement_forms::Form::AtMost => {
                hi = Some(hi.map_or(form.a, |current: f64| current.min(form.a)))
            }
            refined_sets::refinement_forms::Form::Below => {
                hi = Some(hi.map_or(form.a.ceil() - 1.0, |current: f64| current.min(form.a.ceil() - 1.0)))
            }
            refined_sets::refinement_forms::Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, hi) = (lo?, hi?);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// The number of decimal digits a NONNEGATIVE integer's plain `str()`
/// spelling carries — `0` itself spells one digit ("0"), matching
/// `format_integer_spelling`'s own no-leading-zero convention.
pub(super) fn decimal_digit_count(value: i64) -> u32 {
    if value == 0 {
        return 1;
    }
    value.unsigned_abs().to_string().len() as u32
}

/// One codepoint drawn from the given ASCII characters — the digit and
/// sign alphabet `int_spelling_set`/`float_spelling_set` repeat.
pub(super) fn one_char_of(chars: &str) -> RefinedSet {
    let points: Vec<f64> = chars.chars().map(|c| c as u32 as f64).collect();
    make_refined_set(vec![one_of(&points)])
}

/// Every string `str()` can spell for an Integer-sorted value: one or
/// more characters drawn from the digits and `-` — `stdtypes.rst`
/// (`int.__repr__`) states no other characters and no length ceiling
/// (CPython `int` is arbitrary-precision, verified: `str(10**30) ==
/// "1000000000000000000000000000000"`, `str(-5) == "-5"`, `str(0) ==
/// "0"`). A single `Repeat` over the two-character alphabet, rather than
/// a union of a bare digit run and a `-`-prefixed one, admits a few
/// strings `str()` never produces (an interior or repeated `-`, e.g.
/// `"1-2"`) — still a SOUND superset of every real spelling, and the
/// shape the kernel's sequence reader
/// (`set_functions/subset_seq_shape.lean`'s `seqOf`) recognizes directly:
/// a lone `.Repeat A lo none` is read as a positional shape outright
/// (line `some (List.replicate lo A, some A)`), where a `Union` of two
/// concatenation shapes is not — the pattern union routes
/// (`set_functions/pattern_union_routes.lean`'s `leftRouteB`) only
/// distribute a union that is the FIRST piece of an outer concatenation,
/// and this set is always the TRAILING piece once the caller concatenates
/// it after the f-string's own literal text. The alphabet stays bounded
/// even though the length does not — `Repeat` over a finite `one_of`,
/// never `Star` over the whole codepoint ground — so the kernel's
/// counting-window decider (the same route
/// `temporal_string_grammars.rs`'s `TSG_DIGIT`/`tsg_rep` uses for a
/// bounded digit run) can refute containment in a length window instead
/// of falling through to the unresolved general pattern search.
pub(super) fn int_spelling_set() -> RefinedSet {
    make_refined_set(vec![repeat_of(one_char_of("0123456789-"), 1, None)])
}

/// Every string `str()` can spell for a Float-sorted value: CPython's
/// `repr(float)` alphabet is digits, `-`, `.`, and a lowercase `e`
/// exponent marker (verified: `str(3.5) == "3.5"`, `str(1e+300) ==
/// "1e+300"`, `str(1e-300) == "1e-300"`), or one of the three
/// non-numeric words `inf`, `-inf`, `nan` (verified: `str(float('inf'))
/// == "inf"`, `str(float('-inf')) == "-inf"`, `str(float('nan')) ==
/// "nan"`) — all three of which are themselves spelled only from
/// letters already admitted below (`i`, `n`, `f`, `a`), so folding their
/// three extra letters into the SAME repeated alphabet as the digit/sign
/// run covers every case with one `Repeat`, the shape
/// `int_spelling_set`'s own doc explains the kernel recognizes directly
/// (a bare `Union` embedded as this set's own trailing position, the way
/// a separate words-union would be, is not recognized the same way).
/// CPython never emits an uppercase `E` or a bare `+` outside an
/// exponent, but admitting `+`/`e`/`i`/`n`/`f`/`a` freely only widens the
/// claim, never narrows it past what `str()` can actually produce.
pub(super) fn float_spelling_set() -> RefinedSet {
    make_refined_set(vec![repeat_of(one_char_of("0123456789.+-einaf"), 1, None)])
}

/// The set of strings an f-string interpolation admits, once it is known
/// to be a `Kind::Set` but not readable as one exact value — the
/// spellings-of-a-known-set half of `evaluateTemplate`'s own concatenated
/// pattern (walk/literal_values.go's `case known.Kind == KindSet &&
/// stringy`). A STRING-sorted set (`set_kind_tag == SetKindTag::None`
/// with no numeric `kind_tag` — a compiled `Label`-shaped alias, or the
/// `strings()` set an `__name__` read or `str`-return sort fallback
/// already carries) contributes its OWN set verbatim: every spelling the
/// interpolation can hold IS a member of that set already. A NUMERIC-
/// sorted set (Integer or Float `kind_tag` — `summaries::
/// return_sort_fallback`'s int-sort fallback, or `float_sorted_unknown()`)
/// spells through `int_spelling_set`/`float_spelling_set` instead of the
/// unbounded `codepoint_sets::strings()` this used to fall back to: the
/// bare `strings()` claim is sound but its `Star` shape is one the
/// kernel's placement search cannot always decide against a length
/// window (verified: `refinedpy-check` on this file used to panic with
/// "no pattern inclusion proof — the placement search found none" rather
/// than fire), where a bounded-alphabet `Repeat` routes through the
/// proved counting-window decider instead (refined-lean's
/// `set_functions/subset_window.lean`, `RefinedSet.seqAskableB` on a
/// single `.Repeat` form). A bare `Number` tag (no Python sort proved,
/// `summaries.rs`'s int/float join) reads through the FLOAT alphabet —
/// the wider of the two, so a value that could be either sort still gets
/// a sound superset. Any other `Kind::Set` shape (a set carrying no sort
/// tag at all, or one this function does not recognize) declines — the
/// caller's own `unknown()` fallback stays honest for it.
pub(super) fn spellings_of_known_set(value: &AbstractValue) -> Option<RefinedSet> {
    if value.kind != Kind::Set {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Integer) => Some(int_spelling_set()),
        Some(PrimitiveKind::Float) | Some(PrimitiveKind::Number) => Some(float_spelling_set()),
        Some(PrimitiveKind::String) | None => {
            if value.set_kind_tag == SetKindTag::None {
                Some(value.set.clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// The `Vec<f64>` code points `string_models.rs` builds, converted back
/// to a Rust `String` — the same conversion `string_models.rs`'s own
/// (private) `exact_string_text` performs; repeated here because this
/// file is out-of-crate from `string_models.rs`'s module (AGENT-BRIEF.md:
/// this wave touches only `expressions.rs`, so no visibility is widened
/// there for this one caller).
pub(super) fn code_points_to_string(code_points: &[f64]) -> Option<String> {
    code_points
        .iter()
        .map(|point| char::from_u32(*point as i64 as u32))
        .collect()
}

/// A known Integer-sorted value's plain spelling: `"42"`, never `"42.0"`
/// — Python's f-string `str()` conversion of an int has no decimal
/// point (contrast `format_py_number`'s float spelling, which is a
/// different sort this row does not attempt).
pub(super) fn format_integer_spelling(value: f64) -> String {
    format!("{}", value as i64)
}

/// `body if test else orelse` — expressions.rst, "Conditional
/// expressions": "Only one of the expressions is evaluated" once `test`
/// is decided. A decided test evaluates and answers only the taken arm
/// (the other arm's side effects, if any, never happen — matching
/// CPython's own short-circuit read); an undecided test still evaluates
/// both arms (neither is skipped when it is not known which one runs)
/// and joins their values, the loosest sound answer once both cannot be
/// ruled out.
///
/// Each arm is evaluated under its OWN forked, narrowed environment —
/// exactly the fork/`narrowing::assume` pattern `walk_if` runs for an
/// `if`/`else` STATEMENT (check.rs's own `walk_if`), applied here to the
/// expression form instead of duplicating it. `sample if sample is not
/// None else 0.0` forks on `sample is not None`: the true fork narrows
/// `sample` (its possibly-absent tag drops) before `ternary.body` reads
/// it, and the false fork narrows it the other way before
/// `ternary.orelse` reads it. A decided test still narrows before
/// picking the one arm it evaluates, since a name the taken arm reads
/// may depend on that same narrowing (an `isinstance`-proved sort, a
/// walrus-bound comparison, …).
pub(super) fn evaluate_ternary(ternary: &ruff_python_ast::ExprIf, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let test = evaluate_expression(&ternary.test, environment, kernel);
    let (value, known) = truthiness(&test);
    if known {
        return if value {
            let body_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, true);
            evaluate_expression(&ternary.body, &body_environment, kernel)
        } else {
            let orelse_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, false);
            evaluate_expression(&ternary.orelse, &orelse_environment, kernel)
        };
    }
    let body_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, true);
    let orelse_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, false);
    let body = evaluate_expression(&ternary.body, &body_environment, kernel);
    let orelse = evaluate_expression(&ternary.orelse, &orelse_environment, kernel);
    join_known(body, orelse)
}
