//! Sort-conversion builtins: `int`, `float`, `str`, `chr`, `format` —
//! the single-value rows, the string-parsing rows, and the kernel-asked
//! `Kind::Set` rows. Every row cites its clause of docs.python.org/3.12/
//! library/functions.html or library/stdtypes.html; a row with no
//! citation is not written.

use std::sync::Arc;

use refined_domain::abstract_value::{
    float_sorted_unknown, known_set, known_values, nan_value, AbstractValue, Kind, ObjectKey, PrimitiveKind, SetKindTag,
};
use refined_domain::known_constructors::{known_list, known_object};
use refined_domain::trust_grades::{derived_trust_level, TrustProved, TrustSpec};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::{PowOperandKind, PowOperandWire, TransferAnswerKind, TransferQuestion, TransferQuestionOp};
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::{make_refined_set, Form, RefinedSet};
use unicode_normalization::UnicodeNormalization;

use super::numeric::single_known_numeric;
use crate::string_models::string_literal_value;

/// `int(x)` — library/functions.html#int: "For floating-point numbers,
/// this truncates towards zero." An already-Integer argument is the
/// identity read under this row (the same trunc-toward-zero rule with
/// no fractional part to discard). A known EXACT STRING parses through
/// `parse_base_ten_int_string` — the base-10 `int(string, base=10)`
/// row (functions.rst): j-stdlib-surfaces.py's own `int_parse`,
/// `int("40")`/`int("200")`, both exact parses this row now answers
/// precisely rather than declining. A string that does not parse as a
/// base-10 integer (`int("abc")`) still declines HERE — CPython raises
/// `ValueError` for it, which `expressions.rs`'s own `call_provable_
/// raise` speaks through the raise channel (its own `is_valid_base_
/// ten_int_string` gate, a parallel/duplicate validity check to this
/// row's own `parse_base_ten_int_string` — the two files stay
/// independent per the mission's own file-ownership split, so the
/// validity rule is written twice rather than shared across the
/// boundary). A KNOWN Boolean-tagged `Kind::Values` operand — the exact
/// shape a proved `isinstance(x, bool)` seeds
/// (`narrowing.rs::sort_seed`'s own `known_values(vec![0.0, 1.0],
/// PrimitiveKind::Boolean, ...)`, possibly narrowed further to a single
/// member) — reads through `boolean_operand_as_int_values` below rather
/// than `single_known_numeric`, since that helper only ever reads an
/// Integer/Float-tagged SINGLE value and a Boolean binding both carries
/// the wrong tag and, before any further narrowing, TWO members.
/// `bool.__int__` is the identity on each member (`True` is an `int`
/// subclass whose integer value is exactly `1`, `False` exactly `0`,
/// stdtypes.rst's own Boolean Type note), so every admitted member maps
/// straight across.
pub(super) fn int_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if let [text, base] = arguments {
        return int_call_in_radix(text, base);
    }
    let [only] = arguments else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        let text: String = only.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        let parsed = parse_base_ten_int_string(&text)?;
        let grade = derived_trust_level(TrustSpec, arguments);
        return Some(known_values(vec![parsed], PrimitiveKind::Integer, grade));
    }
    if let Some(result) = boolean_operand_as_int_values(only, arguments) {
        return Some(result);
    }
    let (value, _sort) = single_known_numeric(only)?;
    // `int(float('nan'))` RAISES `ValueError: cannot convert float NaN
    // to integer` in CPython (library/functions.html#int delegates to
    // `__trunc__`, and `float.__trunc__` raises on a non-finite operand
    // — the same domain gate `math_models.rs`'s `integral_domain_admits`
    // documents for `math.floor`/`ceil`/`trunc`). No value is returned,
    // so this declines outright rather than answer a value the real
    // call never produces.
    if !value.is_finite() {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.trunc()], PrimitiveKind::Integer, grade))
}

/// `bool(object=False, /)` — library/functions.rst: "Return a Boolean
/// value, i.e. one of ``True`` or ``False``. The argument is converted
/// using the standard truth testing procedure. If the argument is false
/// or omitted, this returns ``False``; otherwise, it returns ``True``."
///
/// The result is ALWAYS one of the two values, so this row never
/// declines: `python_truth_value` decides which one when it can, and
/// the exact two-member boolean domain is the answer when it cannot.
/// That two-member answer is what a downstream `int(...)` reads back as
/// `{0, 1}` (`boolean_operand_as_int_values`), instead of widening to
/// `int_image`'s unbounded ray.
///
/// The zero-argument form `bool()` is `False` by the same clause's own
/// "or omitted".
pub(super) fn bool_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if arguments.is_empty() {
        return Some(known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved));
    }
    let [only] = arguments else { return None };
    match python_truth_value(only) {
        Some(truthy) => Some(known_values(
            vec![if truthy { 1.0 } else { 0.0 }],
            PrimitiveKind::Boolean,
            TrustProved,
        )),
        None => Some(known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec)),
    }
}

/// Python's own truth testing procedure (stdtypes.rst, "Truth Value
/// Testing"), for the value shapes this domain carries. `None` means
/// undecided for this shape, never a guess.
///
/// This is written HERE rather than taken from
/// `refined_domain::lattice_operations::truthiness`, which implements
/// the ECMAScript rule (`sec-toboolean`: every object is truthy) and so
/// answers `True` for an empty list — Python's rule is the opposite.
/// That divergence is recorded deliberately in
/// `truthiness_conformance.rs`'s own
/// `test_container_truthiness_is_python_owned_and_not_posed`, which
/// states the container rule is the Python adapter's to own.
///
/// The clause's own falsy list: "constants defined to be false: ``None``
/// and ``False``"; "zero of any numeric type"; "empty sequences and
/// collections: ``''``, ``()``, ``[]``, ``{}``, ``set()``, ``range(0)``".
fn python_truth_value(value: &AbstractValue) -> Option<bool> {
    match value.kind {
        // "constants defined to be false: None"
        Kind::Null => Some(false),
        // a known exact string is falsy exactly when it is empty
        Kind::Values if value.kind_tag == Some(PrimitiveKind::String) => Some(!value.values.is_empty()),
        // "zero of any numeric type" — a single known number decides;
        // `False` is the integer zero, so the Boolean tag rides the same
        // row (`bool` is an `int` subclass, stdtypes.rst)
        Kind::Values => {
            let [only] = value.values[..] else { return None };
            Some(only != 0.0)
        }
        // "empty sequences and collections" — a KNOWN list/tuple/set
        // whose element count this domain holds decides by that count
        Kind::List => Some(!value.items.is_empty()),
        Kind::Collection if value.complete => Some(!value.entries.is_empty()),
        // `Kind::Object` is deliberately absent: it carries both a dict
        // display (falsy when empty) and a class instance (truthy unless
        // its own `__bool__`/`__len__` says otherwise, which this domain
        // does not read), so no one rule decides it here.
        _ => None,
    }
}

/// `int(string, /, base)` — the RADIX form, library/functions.rst:
/// "If the argument is not a number or if *base* is given, then it must
/// be a string... representing an integer in radix *base*. Optionally,
/// the string can be preceded by `+` or `-` (with no space in between),
/// have leading zeros, be surrounded by whitespace, and have single
/// underscores interspersed between digits."
///
/// "A base-n integer string contains digits, each representing a value
/// from 0 to n-1. The values 0--9 can be represented by any Unicode
/// decimal digit. The values 10--35 can be represented by `a` to `z`
/// (or `A` to `Z`). ... The allowed bases are 0 and 2--36. Base-2, -8,
/// and -16 strings can be optionally prefixed with `0b`/`0B`,
/// `0o`/`0O`, or `0x`/`0X`." The doc's own worked example
/// `int('FACE', 16)` is `64206`, exactly what this row computes.
///
/// Modeled for a known exact string and a KNOWN base in `2..=36`. Base
/// `0` declines: it "is interpreted in a similar way to an integer
/// literal in code, in that the actual base is 2, 8, 10, or 16 as
/// determined by the prefix" and "also disallows leading zeros" — a
/// different grammar this row does not spell. A string outside the
/// radix grammar declines here too; `expressions`'s own raise channel
/// speaks the `ValueError` for it.
fn int_call_in_radix(text_value: &AbstractValue, base_value: &AbstractValue) -> Option<AbstractValue> {
    let (base, base_sort) = single_known_numeric(base_value)?;
    if base_sort != PrimitiveKind::Integer {
        return None;
    }
    let base = base as u32;
    if !(2..=36).contains(&base) {
        return None;
    }
    if text_value.kind != Kind::Values || text_value.kind_tag != Some(PrimitiveKind::String) {
        // an UNREAD string in a known radix: the digits are unknown, so
        // the value is unknown — but `int(...)` returns a Python `int`
        // whatever the digits are, and this domain states integers with
        // no bound as the Integer-sorted ground. That is A3.xfer.parse's
        // own `parse_hex_outside` claim ("n is ℤ (unbounded)"), a
        // determined sort rather than nothing at all.
        if !is_string_sorted_argument(text_value) {
            return None;
        }
        return Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![refined_sets::refinement_forms::integer()]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        });
    }
    let text: String = text_value.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
    let body = text.trim();
    let (negative, body) = match body.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, body.strip_prefix('+').unwrap_or(body)),
    };
    // the optional radix prefix, allowed only for its own base
    let prefix = match base {
        2 => Some(["0b", "0B"]),
        8 => Some(["0o", "0O"]),
        16 => Some(["0x", "0X"]),
        _ => None,
    };
    let body = match prefix {
        Some([lower, upper]) => body.strip_prefix(lower).or_else(|| body.strip_prefix(upper)).unwrap_or(body),
        None => body,
    };
    // "single underscores interspersed between digits" — never leading,
    // trailing, or doubled
    if body.starts_with('_') || body.ends_with('_') || body.contains("__") {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    if digits.is_empty() {
        return None;
    }
    let mut magnitude: f64 = 0.0;
    for digit in digits.chars() {
        let place = digit.to_digit(base)? as f64;
        magnitude = magnitude * base as f64 + place;
    }
    let parsed = if negative { -magnitude } else { magnitude };
    let grade = derived_trust_level(TrustSpec, &[text_value.clone(), base_value.clone()]);
    Some(known_values(vec![parsed], PrimitiveKind::Integer, grade))
}

/// `int(x)` on a KNOWN Boolean-tagged `Kind::Values` operand, of any
/// member count — the shape `isinstance(x, bool)` seeds
/// (`narrowing.rs::sort_seed`) and any further comparison narrowing
/// still leaves Boolean-tagged (`narrow_isinstance_call`'s own doc: a
/// Values binding stays `Kind::Values`/`Boolean` through narrowing,
/// never promoted to Integer). Every member is already `0.0` or `1.0`
/// (the domain's own Boolean encoding), so `int()`'s truncation is the
/// identity — this maps the member LIST across unchanged, only the tag
/// changes to `Integer`. An empty member list (an infeasible branch,
/// `narrow_isinstance_call`'s own "disagrees with the test" arm) still
/// answers the empty Integer-tagged set, honestly carrying zero
/// admitted values forward rather than declining. Any other tag, or a
/// non-`Kind::Values` shape, declines — `int_call`'s own numeric/string
/// rows own those.
fn boolean_operand_as_int_values(value: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::Boolean) {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(value.values.clone(), PrimitiveKind::Integer, grade))
}

/// `int(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded set
/// another transfer already produced — e.g. `int(math.sqrt(x))`,
/// `math.sqrt`'s own Float-sorted enclosure over a declared parameter
/// range, `math_models.rs`'s `sqrt_call_over_set`): `int_call`'s own
/// row only reads a single known numeric value
/// (`single_known_numeric`), so a Set-shaped argument declines there
/// with no further attempt. This asks the kernel's own `Trunc`
/// transfer directly — the exact mirror of `abs_call_over_set` above
/// (same `TransferQuestion` construction, same `catch_unwind` refusal
/// discipline, same `TransferAnswerKind` match) — library/
/// functions.html#int: "For floating-point numbers, this truncates
/// towards zero," the same trunc-toward-zero rule `int_call`'s
/// single-value row already applies, here posed to `binary64.trunc`
/// (`TransferQuestionOp::Trunc`) instead of computed locally. Unlike
/// `abs_call_over_set` (which preserves the operand's own sort), the
/// result is Integer sort UNCONDITIONALLY — `int(x)` always returns an
/// `int` regardless of its argument's sort, the same rule `int_call`'s
/// own `known_values(..., PrimitiveKind::Integer, ...)` return states.
///
/// A kernel-answered enclosure NOT provably finite
/// (`enclosure_is_provably_finite` false — e.g. `binary64.trunc` over a
/// bare unbounded `float` parameter's own `numbers()` seed,
/// `float_sorted_unknown`'s own doc) does not decline outright: the
/// same non-finite gate `int_call`'s single-value row keeps
/// (`int(float('nan'))`/`int(float('inf'))` both RAISE `ValueError`/
/// `OverflowError` in CPython, never returning a value) rules out ONLY
/// the two non-finite INPUTS, not every finite input the enclosure also
/// admits — those still truncate to SOME integer, so the WEAKER but
/// still TRUE claim over the non-raising outcomes is the unbounded
/// Integer sort (`int_image`'s own image — every row `int(...)`
/// returns at all is an int, library/functions.html#int), not `None`.
/// Answering `None` here left `n = int(x)` for a bare `float`
/// parameter's own guard branches Unknown downstream — one undetermined
/// branch that then poisons a whole function's derived return cases
/// (D5's own `clamp_to_age` helpers, ISSUES.md's fact-export trace).
/// `int_call`'s own single-VALUE row keeps declining outright on a
/// non-finite operand (unchanged): that row reads ONE concrete number,
/// which either raises or does not — there is no "other outcomes" to
/// weaken to when the whole operand IS the non-finite value itself, the
/// same distinction `domain_raise_served_half_value`'s own "straddling
/// vs. entirely-raising" split keeps for a domain-limited math family.
pub(super) fn int_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean) | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Trunc,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    match asked.kind {
        TransferAnswerKind::Values => {
            if !asked.values.iter().all(|v| v.is_finite()) {
                return None;
            }
            Some(known_values(asked.values, PrimitiveKind::Integer, grade))
        }
        TransferAnswerKind::Set => {
            if !enclosure_is_provably_finite(&asked.set) {
                // the finite outcomes still all truncate to an int —
                // `int_image`'s own unbounded Integer ray, the weaker
                // TRUE claim over the non-raising half of this operand
                // (this function's own doc above)
                return int_image();
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(asked.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// Whether a set the kernel answered describes only FINITE values — the
/// set-shaped twin of `is_finite`, for `int_call_over_set`'s own arm
/// that reads a kernel enclosure back as a Python `int` result. A
/// private copy of `math_models.rs`'s identically-named helper: this
/// file's own header states the file-ownership convention already kept
/// for `int_call`'s validity check ("the two files stay independent...
/// the rule is written twice rather than shared across the boundary").
///
/// `±inf` ARE elements of the grammar (`refinement_forms`'s own module
/// note: "+-infinity are elements of R-bar and are admitted"), so a
/// bound or an admitted value can be infinite and the set is still
/// well-formed — it just describes a result no Python `int` can hold.
/// NaN cannot appear at all (`element` panics on it at construction), so
/// there is nothing to check for it here.
///
/// This reads the set's OWN top-level forms, looking through
/// `Union`/`Difference`. A form this recognizer does not understand
/// answers `false` — an unread shape declines rather than being assumed
/// finite, which is the direction that keeps the gate honest.
fn enclosure_is_provably_finite(set: &RefinedSet) -> bool {
    if set.forms.is_empty() {
        // the unconstrained set — every real AND both infinities
        return false;
    }
    let mut bounded_below = false;
    let mut bounded_above = false;
    for form in &set.forms {
        match form.form {
            Form::AtLeast | Form::Above => {
                if !form.a.is_finite() {
                    // `atLeast(-inf)` constrains nothing; `atLeast(+inf)`
                    // admits only +inf
                    return false;
                }
                bounded_below = true;
            }
            Form::AtMost | Form::Below => {
                if !form.a.is_finite() {
                    return false;
                }
                bounded_above = true;
            }
            // an explicit value list is finite exactly when every value is
            Form::OneOf => {
                return form.w.iter().all(|v| v.is_finite());
            }
            Form::Union => {
                let (Some(left), Some(right)) = (form.a_.as_ref(), form.b.as_ref()) else {
                    return false;
                };
                // a union is finite only if BOTH arms are
                return enclosure_is_provably_finite(left) && enclosure_is_provably_finite(right);
            }
            // a difference is finite when its left arm is — removing
            // values never adds an infinity
            Form::Difference => {
                let Some(left) = form.a_.as_ref() else {
                    return false;
                };
                return enclosure_is_provably_finite(left);
            }
            // `Integer`/`MultipleOf` narrow but do not bound; the
            // sequence shapes are not scalar sets at all
            Form::Integer | Form::MultipleOf => {}
            _ => return false,
        }
    }
    bounded_below && bounded_above
}

/// `int(string, base=10)`'s exact parsed value, for the base-10
/// default form ONLY (`int_call`'s own scope — a `base=` keyword
/// changes the digit alphabet entirely and is not read by this row's
/// caller, which never passes one through). functions.rst's own
/// grammar: "the string can be preceded by + or - (with no space in
/// between), have leading zeros, be surrounded by whitespace, and have
/// single underscores interspersed between digits." Returns `None`
/// (never a fabricated value) the moment the text does not parse —
/// `call_provable_raise`'s own `is_valid_base_ten_int_string` is the
/// row that speaks the ValueError this shape raises at runtime.
fn parse_base_ten_int_string(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let negative = trimmed.starts_with('-');
    let digits_and_underscores = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if digits_and_underscores.is_empty() {
        return None;
    }
    let chars: Vec<char> = digits_and_underscores.chars().collect();
    if chars.first() == Some(&'_') || chars.last() == Some(&'_') {
        return None;
    }
    let mut digits = String::new();
    let mut previous_was_underscore = false;
    for &c in &chars {
        if c == '_' {
            if previous_was_underscore {
                return None;
            }
            previous_was_underscore = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return None;
        }
        digits.push(c);
        previous_was_underscore = false;
    }
    if digits.is_empty() {
        return None;
    }
    let magnitude: f64 = digits.parse().ok()?;
    Some(if negative { -magnitude } else { magnitude })
}

/// `float(x)` on a single known numeric or known exact string —
/// library/functions.html#float: "Return a floating-point number
/// constructed from a number or a string." A NUMERIC argument answers
/// its exact value, Float-sorted. A known EXACT string is parsed by
/// `parse_float_literal_string` — that function's own doc cites the
/// grammar (functions.rst's `productionlist:: float`): the `inf`/
/// `Infinity`/`nan` spellings (case-insensitive, optional leading sign)
/// answer the exact infinite/NaN value, and any other text that parses
/// as the grammar's `floatnumber` production answers that exact decimal
/// value. A STRING-sorted argument with no exact text this file can
/// parse (`is_string_sorted_argument`'s own doc — e.g. a captured
/// subprocess `.stdout` read: `expressions.rs`'s own
/// `subprocess_run_construction_value`) still determines a SORT: the
/// same clause states `float`'s return is always a `float` regardless of
/// which of the two argument forms produced it, so `float(<any string>)`
/// answers `float_sorted_unknown()` — sort-known, value-unknown, the
/// same posture every other sort-only row in this file takes rather than
/// decline outright. An EXACT string that fails to parse under the
/// grammar keeps that same sort-only posture rather than decline
/// outright (`is_string_sorted_argument` already reads a
/// `Kind::Values`/`String` argument as string-sorted) — CPython raises
/// `ValueError` for it, which this file has no exception channel for,
/// so the sort-only answer is the honest fallback, not a fabricated
/// value.
pub(super) fn float_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if let Some((value, _sort)) = single_known_numeric(only) {
        if value.is_nan() {
            return Some(nan_value());
        }
        let grade = derived_trust_level(TrustSpec, arguments);
        return Some(known_values(vec![value], PrimitiveKind::Float, grade));
    }
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        let text: String = only.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        if let Some(value) = parse_float_literal_string(&text) {
            // `float("nan")` (and its case/sign variants — parsed by
            // `parse_float_literal_string`'s own grammar reading)
            // answers the domain's NaN state rather than let a bare NaN
            // enter `known_values`, which no refined set admits
            // (`refinement_forms::element`'s own construction-time
            // refusal — the same guard `float_result` keeps in
            // math_models.rs for `math.fabs(nan)`).
            if value.is_nan() {
                return Some(nan_value());
            }
            let grade = derived_trust_level(TrustSpec, arguments);
            return Some(known_values(vec![value], PrimitiveKind::Float, grade));
        }
        return Some(float_sorted_unknown());
    }
    // A Boolean-tagged `Kind::Values` operand — the two-member boolean
    // domain a membership read answers, possibly narrowed to one member
    // — maps each member straight across: `True` is the `int` subclass
    // whose value is exactly `1`, `False` exactly `0` (stdtypes.rst's
    // Boolean Type note), and functions.html#float constructs the same
    // magnitude Float-sorted. The same reading `int_call` takes through
    // `boolean_operand_as_int_values`, Float-tagged here.
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::Boolean) && !only.values.is_empty() {
        let grade = derived_trust_level(TrustSpec, arguments);
        return Some(known_values(only.values.clone(), PrimitiveKind::Float, grade));
    }
    if is_string_sorted_argument(only) {
        return Some(float_sorted_unknown());
    }
    None
}

/// `float(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded set
/// another transfer already produced — e.g. `float(math.floor(x))`,
/// `math.floor`'s own Integer-sorted enclosure over a declared parameter
/// range, `math_models.rs`'s `rounding_call_over_set`): `float_call`'s own
/// row only reads a single known numeric value (`single_known_numeric`),
/// so a Set-shaped argument declines there with no further attempt.
/// Unlike `int_call_over_set`/`abs_call_over_set` (which pose a kernel
/// `TransferQuestion` because their result VALUE differs from their
/// input), `float(x)` on a numeric argument changes only the SORT, never
/// the value (library/functions.html#float: "Return a floating-point
/// number constructed from a number" — the same magnitude, Float-sorted)
/// — so this re-tags the operand's own set in place, no kernel round
/// trip needed. CPython never raises for a numeric argument (only the
/// string-parse form can raise, `float_call`'s own doc), so every
/// Integer/Float/Boolean/Number-sorted set answers here, unconditionally.
pub(super) fn float_call_over_set(value: &AbstractValue) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean) | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    Some(AbstractValue { kind_tag: Some(PrimitiveKind::Float), ..known_set(value.set.clone(), None, grade, SetKindTag::None) })
}

/// `float(string)`'s exact parsed value, for the grammar
/// library/functions.rst's `productionlist:: float` states (read
/// before writing this function): after leading/trailing whitespace is
/// removed, an optional `sign` (`+`/`-`, `+` has no effect), then either
/// `infinity` (`"Infinity"` or `"inf"`, case-insensitive per that
/// section's own "Case is not significant... 'inf', 'Inf', 'INFINITY',
/// and 'iNfINity' are all acceptable spellings"), `nan` (`"nan"`, same
/// case-insensitivity), or a `floatnumber` (`digitpart ["." digitpart]`
/// or `["." digitpart]`, with an optional `(e|E) [sign] digitpart`
/// exponent — underscores between digits allowed, the same grouping
/// `parse_base_ten_int_string` already reads for `int`). Returns `None`
/// or panics on no legitimate value: `None` when the text does not
/// conform to the grammar (`float_call`'s own caller falls back to the
/// sort-only answer for this row, never a fabricated value) or the
/// parse is not itself the exact spelled decimal (never here, since the
/// spellings this function recognizes route straight to `f64::INFINITY`/
/// `f64::NEG_INFINITY`/`f64::NAN`/Rust's own `str::parse::<f64>`, which
/// implements the same decimal grammar).
fn parse_float_literal_string(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let (negative, unsigned) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    if unsigned.is_empty() {
        return None;
    }
    let lowered = unsigned.to_ascii_lowercase();
    if lowered == "inf" || lowered == "infinity" {
        return Some(if negative { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    if lowered == "nan" {
        return Some(f64::NAN);
    }
    // the `floatnumber` production: digits (with single underscores
    // between them, the same grouping rule int()'s own parse allows),
    // an optional decimal point, an optional e/E exponent — Rust's
    // `str::parse::<f64>` reads this same grammar once underscores are
    // stripped, so digit-and-underscore validity is checked by hand
    // first (a stray underscore, e.g. "1__0" or "_1", is invalid Python
    // syntax that `str::parse` would otherwise silently reject anyway,
    // but the explicit check keeps this row's acceptance exactly the
    // documented grammar rather than piggybacking on Rust's own parser
    // leniency).
    let mut digits_only = String::with_capacity(unsigned.len());
    let mut previous_was_underscore = false;
    let mut previous_was_digit = false;
    for c in unsigned.chars() {
        if c == '_' {
            if !previous_was_digit || previous_was_underscore {
                return None;
            }
            previous_was_underscore = true;
            continue;
        }
        digits_only.push(c);
        previous_was_underscore = false;
        previous_was_digit = c.is_ascii_digit();
    }
    if previous_was_underscore {
        return None;
    }
    let value: f64 = digits_only.parse().ok()?;
    Some(if negative { -value } else { value })
}

/// Whether `argument` is a STRING-sorted value: an exact `Kind::Values`
/// tagged `PrimitiveKind::String`, or a `Kind::Set` that is either
/// explicitly tagged String or untagged with a sequence-shaped own set
/// (`assignability.rs`'s own `sequence_shaped` — the SAME "untagged set,
/// sequence-shaped forms read as string-sorted" convention that file's
/// containment law already applies, e.g. `__name__`'s own untagged
/// `strings()` ground in `expressions.rs`).
pub(super) fn is_string_sorted_argument(argument: &AbstractValue) -> bool {
    if argument.kind == Kind::Values {
        return argument.kind_tag == Some(PrimitiveKind::String);
    }
    if argument.kind != Kind::Set {
        return false;
    }
    argument.kind_tag == Some(PrimitiveKind::String)
        || (argument.kind_tag.is_none() && crate::assignability::sequence_shaped(&argument.set))
}

/// `chr(i)` on a known Integer code point — library/functions.html#chr:
/// "Return the string representing a character whose Unicode code
/// point is the integer *i*." A one-code-point exact string, the same
/// `Kind::Values`/`PrimitiveKind::String` shape `string_models.rs`
/// builds for any other exact string. `i` outside the valid code-point
/// range (`0..=0x10FFFF`, the same range `char::from_u32` itself
/// enforces) has no row here: CPython raises `ValueError`, which this
/// domain has no channel for this wave, so this row declines rather
/// than answer a fabricated character.
pub(super) fn chr_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, sort) = single_known_numeric(only)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    if value < 0.0 || value > 0x10FFFF as f64 {
        return None;
    }
    char::from_u32(value as u32)?;
    Some(known_values(vec![value], PrimitiveKind::String, TrustSpec))
}

/// `str(object)` — library/stdtypes.rst's `class:: str(object='')`
/// constructor row: "Return a string version of *object*." Modeled for
/// four known argument shapes: an exact string (the identity
/// conversion — `str(word)` answers `word` unchanged, per the same
/// row's own "If *object* already is a string, it is returned
/// unchanged" behavior), a known Integer (CPython's plain decimal
/// spelling, no `.0` — the same integer-spelling rule
/// `expressions.rs`'s f-string composition already establishes for an
/// interpolated Integer), a known EXCEPTION instance
/// (`expressions.rs`'s `exception_construction_value`, tagged
/// `source == "exception"`, one `args` field holding the constructor's
/// own positional arguments as a `Kind::List`) whose FIRST argument is
/// a known exact string — `str(Exception(message))` answers `message`
/// unchanged: `Doc/tutorial/errors.rst`, "Errors and Exceptions" §8.3,
/// "the exception instance... typically has an `args` attribute...
/// builtin exception types define `__str__` to print all the
/// arguments." A single-string-argument exception's `__str__` is
/// exactly that one string (CPython's own `BaseException.__str__`:
/// zero args -> `''`, one arg -> `str(args[0])`, 2+ args -> the
/// `repr()` of the whole tuple — only the one-string-argument row is
/// modeled here), a NONNEGATIVE BOUNDED Integer window (a seeded
/// parameter range, or a bounded set another transfer produced) —
/// `int.__repr__`'s plain no-leading-zero decimal spelling widened
/// from one value to the whole window, `json_grammar::
/// integer_window_grammar`'s own composition (already built for
/// `json.dumps`'s serialized-text grammar; reused here rather than
/// duplicated) — and a known FLOAT value: `stdtypes.rst`'s own `str(x)`
/// clause for `float` delegates to `repr(x)`'s shortest round-tripping
/// decimal, mandatory `.0` on a whole value, exponent form outside
/// `%g`'s plain-decimal window — exactly `refined_sets::
/// format_string_shapes::format_py_number(value, true)` already builds
/// (A2.xfer.tostring's own three rows: `str(0.5)` == `"0.5"`, `str(0.1
/// + 0.2)` == `"0.30000000000000004"`, `str(1e21)` == `"1e+21"`, all
/// exact matches to that function's own doc). NaN/±∞ are excluded here
/// — `format_py_number`'s `format_js_number` base spells `NaN`/`inf`/
/// `-inf`, CPython's own `str(float("nan"))`/`str(float("inf"))`
/// spellings ("nan"/"inf"/"-inf", lowercase, `stdtypes.rst`'s float
/// constructor note), a DIFFERENT text this row does not yet build —
/// `single_numeric_operand`-style finiteness is checked before
/// spelling rather than emit the wrong-case text.
pub(super) fn str_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        return Some(only.clone());
    }
    if only.kind == Kind::Object && only.source == "exception" {
        return exception_single_string_message(only);
    }
    // `str(None)` is exactly the four-character text `"None"` —
    // library/stdtypes.rst, "The Null Object": "There is exactly one null
    // object, named ``None`` (a built-in name)... It is written as
    // ``None``." One singleton object with one written spelling, so the
    // conversion's image is the singleton string set, never a sort-only
    // claim.
    if only.kind == Kind::Null {
        let code_points: Vec<f64> = "None".chars().map(|c| c as u32 as f64).collect();
        return Some(known_values(code_points, PrimitiveKind::String, TrustSpec));
    }
    if let Some(value) = str_call_over_boolean(only) {
        return Some(value);
    }
    if let Some(value) = str_call_over_known_list(only) {
        return Some(value);
    }
    if only.kind == Kind::Set && only.kind_tag == Some(PrimitiveKind::Integer) {
        return str_call_over_integer_window(only);
    }
    let (value, sort) = single_known_numeric(only)?;
    if sort != PrimitiveKind::Integer {
        if !value.is_finite() {
            return None;
        }
        let spelled = refined_sets::format_string_shapes::format_py_number(value, true);
        let code_points: Vec<f64> = spelled.chars().map(|c| c as u32 as f64).collect();
        return Some(known_values(code_points, PrimitiveKind::String, TrustSpec));
    }
    let spelled = format!("{}", value as i64);
    let code_points: Vec<f64> = spelled.chars().map(|c| c as u32 as f64).collect();
    Some(known_values(code_points, PrimitiveKind::String, TrustSpec))
}

/// `ord(c)` on a known ONE-CHARACTER exact string —
/// library/functions.rst: "Given a string representing one Unicode
/// character, return an integer representing the Unicode code point of
/// that character. For example, `ord('a')` returns the integer `97`...
/// This is the inverse of `chr`." `string_models`'s own exact-string
/// encoding already holds one Unicode code point per `values` element,
/// so a length-1 receiver's code point IS the answer — the exact
/// inverse of `chr_call`'s own row, reading the same vector back.
///
/// A receiver whose length is not exactly 1 declines: CPython raises
/// `TypeError` for it, which this row has no channel to speak.
pub(super) fn ord_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::Values || only.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    let [code_point] = only.values[..] else { return None };
    Some(known_values(vec![code_point], PrimitiveKind::Integer, TrustSpec))
}

/// `str(b)` on a Boolean-tagged operand — `True` and `False` "are
/// alternative ways to spell the integer values 1 and 0, with the
/// single difference that `str` and `repr` return the strings `'True'`
/// and `'False'` instead of `'1'` and `'0'`" (whatsnew/2.3.rst, the
/// `bool` type's own introduction; `bool` "has exactly two constant
/// instances: `True` and `False`", stdtypes.rst).
///
/// A SINGLE-member Boolean operand spells its own one text exactly; a
/// BOTH-members one answers the two-branch alternation
/// `{"True", "False"}` — A3.seed.conversion's own
/// `boolean_to_string_outside` claim. The domain encodes a Boolean as
/// `0.0`/`1.0` (`string_models`'s own `boolean_value`), so the member
/// values map straight onto the two spellings.
///
/// Two operand shapes carry a boolean here: the Boolean-tagged
/// `Kind::Values` a narrowed `isinstance(x, bool)` seeds
/// (`narrowing::sort_seed`), and the `one_of([0.0, 1.0])` `Kind::Set` a
/// bare `b: bool` parameter seeds (`typereading::base_sort`'s own
/// `"bool"` row) — A3.seed.conversion's own parameter is the second.
fn str_call_over_boolean(value: &AbstractValue) -> Option<AbstractValue> {
    let members: Vec<f64> = if value.kind == Kind::Values && value.kind_tag == Some(PrimitiveKind::Boolean) {
        value.values.clone()
    } else if value.kind == Kind::Set {
        boolean_set_members(&value.set)?
    } else {
        return None;
    };
    if let [only] = members[..] {
        let text = if only == 0.0 { "False" } else { "True" };
        return Some(known_values(
            text.chars().map(|c| c as u32 as f64).collect(),
            PrimitiveKind::String,
            TrustSpec,
        ));
    }
    if members.len() != 2 {
        return None;
    }
    let compiled = refined_sets::regex_compiler::format_grammar("^(True|False)$", "");
    if !compiled.ok {
        return None;
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(compiled.set, None, TrustSpec, SetKindTag::None)
    })
}

/// The members of a set that is EXACTLY the boolean domain's own
/// `one_of` shape over a subset of `{0.0, 1.0}` — the spelling
/// `typereading::base_sort`'s `"bool"` row builds. Any other form
/// composition answers `None`: this reader states no boolean for a
/// shape it was not built to read.
fn boolean_set_members(set: &RefinedSet) -> Option<Vec<f64>> {
    let [only] = &set.forms[..] else { return None };
    if only.form != Form::OneOf {
        return None;
    }
    let members = only.w.clone();
    if members.is_empty() || !members.iter().all(|member| *member == 0.0 || *member == 1.0) {
        return None;
    }
    Some(members)
}

/// `str(xs)` on a KNOWN list of KNOWN Integers — `list` defines no
/// `__str__` of its own, so `str(object)` "returns `object.__repr__()`"
/// (stdtypes.rst's `str(object)` row, via `object.__str__`'s own
/// default), and `list.__repr__` spells the elements' own reprs between
/// square brackets separated by `", "`: stdtypes.rst's own worked
/// example at the `GenericAlias` section shows `list[str]([1, 2, 3])`
/// echoing as `[1, 2, 3]` — brackets, comma, one space. Each element is
/// an `int`, whose repr is the same plain no-leading-zero decimal
/// spelling `str_call`'s own Integer row already builds.
///
/// Declines for a list holding anything other than known Integers: a
/// string element's repr carries quoting and escaping rules this row
/// does not spell, and an unread element has no repr text at all.
fn str_call_over_known_list(value: &AbstractValue) -> Option<AbstractValue> {
    if value.kind != Kind::List {
        return None;
    }
    let mut spellings: Vec<String> = Vec::with_capacity(value.items.len());
    for element in &value.items {
        let (number, sort) = single_known_numeric(element)?;
        if sort != PrimitiveKind::Integer {
            return None;
        }
        spellings.push(format!("{}", number as i64));
    }
    let text = format!("[{}]", spellings.join(", "));
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    Some(known_values(code_points, PrimitiveKind::String, TrustSpec))
}

/// `str(n)` on a NONNEGATIVE BOUNDED Integer-sorted `Kind::Set` window
/// `[lo, hi]` (a seeded parameter range, or a bounded set another
/// transfer produced) — the exact digit-count run
/// `json_grammar::integer_window_grammar` already composes for
/// `json.dumps`'s serialized-text grammar, reused unchanged here for
/// `str_call`'s own decimal-spelling row: both are the SAME `int.
/// __repr__` plain decimal spelling (stdtypes.rst's `str(object)`
/// row delegates to `__str__`, which for `int` is `__repr__`'s own
/// no-leading-zero decimal text), just reached from a different
/// caller. The bound is read off the set's own top-level
/// `AtLeast`/`Above`/`AtMost`/`Below` forms syntactically — no kernel
/// ask, the same private-copy convention `json_grammar::
/// integer_set_bounds` already keeps against `expressions.rs`'s own
/// identically-named helper (this file's own AGENT-BRIEF scope keeps
/// it from reaching into either). A negative lower bound, or a bound
/// this reader cannot close (an unbounded ray, a union, a pattern),
/// declines — `integer_window_grammar`'s own `lo < 0` refusal
/// propagates here as a decline rather than a fabricated fallback.
fn str_call_over_integer_window(value: &AbstractValue) -> Option<AbstractValue> {
    let (lo, hi) = integer_set_bounds(value)?;
    let grammar = crate::json_grammar::integer_window_grammar(lo, hi)?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(grammar, None, grade, SetKindTag::None)
    })
}

/// `format(value, spec)` (functions.rst): the one-argument and
/// empty-spec forms are `str(value)`'s own row (`format(value, "")`
/// delegates to `str`), and spec `"x"` is the lowercase hexadecimal
/// presentation of an int (the format-spec mini-language, string.rst).
/// An exact nonnegative int spells its exact hex text; a bounded
/// nonnegative SINGLE-HEX-DIGIT window answers the exact one-character
/// alphabet of its own members (`json_grammar::
/// hex_digit_window_grammar` — `[0, 9]` stays decimal digits,
/// `[10, 15]` is exactly the letters). Any other spec or value shape
/// declines.
pub(super) fn format_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match arguments {
        [only] => str_call(std::slice::from_ref(only)),
        [value, spec] => {
            if spec.kind != Kind::Values || spec.kind_tag != Some(PrimitiveKind::String) {
                return None;
            }
            let spec_text: String =
                spec.values.iter().map(|&point| char::from_u32(point as i64 as u32)).collect::<Option<String>>()?;
            match spec_text.as_str() {
                "" => str_call(std::slice::from_ref(value)),
                "x" => {
                    if let Some((exact, sort)) = single_known_numeric(value) {
                        if sort != PrimitiveKind::Integer || exact < 0.0 {
                            return None;
                        }
                        let spelled = format!("{:x}", exact as i64);
                        let code_points: Vec<f64> = spelled.chars().map(|c| c as u32 as f64).collect();
                        return Some(known_values(code_points, PrimitiveKind::String, TrustSpec));
                    }
                    let (lo, hi) = integer_set_bounds(value)?;
                    let grammar = crate::json_grammar::hex_digit_window_grammar(lo, hi)?;
                    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
                    Some(AbstractValue {
                        kind_tag: Some(PrimitiveKind::String),
                        ..known_set(grammar, None, grade, SetKindTag::None)
                    })
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// The closed integer bound `[lo, hi]` a value states, when the value is
/// a bounded Integer-sorted `Kind::Set` — the same syntactic hull
/// `json_grammar::integer_set_bounds`/`expressions.rs::
/// integer_set_bounds` both already read, duplicated here rather than
/// exported per this file's own file-ownership convention (see either
/// of those two functions' own doc comments).
fn integer_set_bounds(value: &AbstractValue) -> Option<(i64, i64)> {
    if value.kind != Kind::Set || value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &value.set.forms {
        match form.form {
            Form::AtLeast => lo = Some(lo.map_or(form.a, |current: f64| current.max(form.a))),
            Form::Above => lo = Some(lo.map_or(form.a.floor() + 1.0, |current: f64| current.max(form.a.floor() + 1.0))),
            Form::AtMost => hi = Some(hi.map_or(form.a, |current: f64| current.min(form.a))),
            Form::Below => hi = Some(hi.map_or(form.a.ceil() - 1.0, |current: f64| current.min(form.a.ceil() - 1.0))),
            Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, hi) = (lo?, hi?);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// The exact message `str()` of a known exception instance answers, for
/// the ONE constructor-argument shape this file models: an `args`
/// field (`expressions.rs`'s own exception-construction tag) holding a
/// `Kind::List` of exactly one known exact-string element —
/// `BaseException.__str__`'s one-argument row (this function's own
/// caller doc). Any other `args` shape (zero elements, 2+ elements, a
/// non-string element) declines — this file does not build the `repr()`
/// spelling a multi-argument `__str__` would need.
fn exception_single_string_message(instance: &AbstractValue) -> Option<AbstractValue> {
    let args = &instance.keys.iter().find(|key| key.name == "args")?.value;
    if args.kind != Kind::List {
        return None;
    }
    let [only] = args.items.as_slice() else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        return Some(only.clone());
    }
    None
}

/// `int(<anything the rows above declined>)`'s own IMAGE: wherever the
/// call returns at all, it returns an int (library/functions.rst — a
/// non-convertible operand raises instead), so an operand no concrete
/// or kernel row reads still answers the unbounded integer sort. The
/// raise arm is `call_provable_raise`'s business — a provably-raising
/// call's value is unreachable, and an unreachable value carrying the
/// image is sound either way.
pub(super) fn int_image() -> Option<AbstractValue> {
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![
                refined_sets::refinement_forms::integer(),
                refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
            ]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    })
}

/// `unicodedata.normalize(form, unistr)` — library/unicodedata.html:
/// "Return the normal form *form* for the Unicode string *unistr*...
/// Valid values for *form* are 'NFC', 'NFKC', 'NFD', and 'NFKD'." The
/// doc states the return is itself a Python `str`, with no further
/// bound on its content or length (a normalization form can both grow
/// and shrink a string's code-point count relative to its input,
/// library/unicodedata.html's own "Unicode Standard Annex #15"
/// citation) — this row states exactly that sort, the whole-strings
/// ground `Σ*`, matching A3.xfer.normalize's own claim ("result is
/// Σ*"). Modeled for the two-argument form with a known exact-string
/// `form` argument in the doc's own four valid spellings; any other
/// `form` (unknown, or a string outside that set) declines rather than
/// assume the call does not raise.
///
/// An exact-string `unistr` answers the EXACT normalized string, under
/// any of the four forms: the `unicode-normalization` crate carries the
/// Unicode Character Database's own decomposition and composition data,
/// so the answer is Unicode Standard Annex #15's, not an approximation.
/// That is what both halves of A3.xfer.normalize need — `normalize("NFC",
/// "AA")` answering `{"AA"}` inside `Code`, and `normalize("NFC",
/// "e\u{0301}")` composing the two-code-point decomposed spelling to the
/// one code point U+00E9, whose `len()` is `{1}` inside `Unit`. A
/// `unistr` this file cannot read exactly still falls to the `Σ*`
/// sort-only claim.
pub(super) fn unicodedata_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "normalize" {
        return None;
    }
    let [form, unistr] = arguments else { return None };
    if form.kind != Kind::Values || form.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    let form_text: String = form.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
    if !matches!(form_text.as_str(), "NFC" | "NFKC" | "NFD" | "NFKD") {
        return None;
    }
    if unistr.kind == Kind::Values && unistr.kind_tag == Some(PrimitiveKind::String) {
        let subject: Option<String> = unistr.values.iter().map(|point| char::from_u32(*point as i64 as u32)).collect();
        if let Some(subject) = subject {
            let normalized: String = match form_text.as_str() {
                "NFC" => subject.nfc().collect(),
                "NFD" => subject.nfd().collect(),
                "NFKC" => subject.nfkc().collect(),
                "NFKD" => subject.nfkd().collect(),
                _ => unreachable!("form_text was already matched against the four valid forms"),
            };
            return Some(string_literal_value(&normalized));
        }
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, TrustSpec, SetKindTag::None)
    })
}

/// `urllib.parse.quote(string)` — library/urllib.parse.html#urllib.parse.quote:
/// "Replace special characters in *string* using the %xx escape...
/// Letters, digits, and the characters '_.-~' are never quoted." The
/// result is a Python `str` built only from that ASCII subset plus the
/// literal `%` escape triples — narrower than the whole-strings ground,
/// but this row states the SORT-ONLY answer (`Σ*`, String-sorted)
/// rather than the tight percent-encoding grammar: the doc's own
/// `safe='/'` default (a further always-unquoted character this row
/// does not thread through) makes the exact alphabet argument-
/// dependent, so `Σ*` is the sound claim actually made here, matching
/// A3.xfer.url's own claim ("result is Σ* (percent-encoding
/// grammar)"). One-argument form only (no `safe=`/`encoding=`/
/// `errors=` keyword arguments modeled).
///
/// A known exact `string` built ONLY from the doc's own never-quoted
/// characters answers EXACTLY that same string: "Letters, digits, and
/// the characters `_.-~` are never quoted," and the one-argument form's
/// own `safe='/'` default only ADDS `/` to that set, so no character of
/// such a string is replaced by a `%xx` escape and the result is the
/// input unchanged. That is what A3.xfer.url's own `quote_inside`
/// (`quote("AA")` returning `Code`) needs; a string carrying any other
/// character falls to the `Σ*` sort-only claim above.
///
/// Reached through `builtin_call_result`'s own BARE-NAME dispatch, not
/// `stdlib_call_result`'s module-qualified one: the corpus's own row
/// (A3.xfer.url.py) writes `from urllib.parse import quote` then calls
/// the bare name `quote(s)` — `urllib.parse` is not a Python-source
/// module the cross-module resolver reads (`check.rs::
/// bind_or_forget_imported_name`'s own doc), so the import binds
/// nothing and `quote` reaches `evaluate_call`'s `Expr::Call(Expr::Name(...))`
/// arm exactly like an ordinary builtin call.
pub(super) fn urllib_quote_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [string] = arguments else { return None };
    if string.kind == Kind::Values && string.kind_tag == Some(PrimitiveKind::String) && string.values.iter().all(|point| is_never_quoted(*point)) {
        return Some(string.clone());
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, TrustSpec, SetKindTag::None)
    })
}

/// `urllib.parse.parse_qs(qs)` on a known exact query string —
/// library/urllib.parse.rst: "Parse a query string given as a string
/// argument... Data are returned as a dictionary. The dictionary keys
/// are the unique query variable names and the values are lists of
/// values for each name." With every default in force: `separator='&'`,
/// and `keep_blank_values=False` so "blank values are to be ignored and
/// treated as if they were not included."
///
/// Modeled for a query string carrying NO percent escapes and no `+`:
/// the *encoding* and *errors* parameters "specify how to decode
/// percent-encoded sequences into Unicode characters", a decoding this
/// row does not perform — a `%` or `+` anywhere in `qs` declines rather
/// than hand back the undecoded text as if it were the value. A field
/// with no `=` also declines: `strict_parsing=False` "silently ignores"
/// such errors, and this row states no shape for what survives.
pub(super) fn parse_qs_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [query] = arguments else { return None };
    if query.kind != Kind::Values || query.kind_tag != Some(PrimitiveKind::String) {
        return parse_qs_unread_query(query);
    }
    let text: String = query.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
    if text.contains('%') || text.contains('+') {
        return None;
    }
    let mut keys: Vec<ObjectKey> = Vec::new();
    for field in text.split('&') {
        if field.is_empty() {
            continue;
        }
        let (name, value) = field.split_once('=')?;
        if value.is_empty() {
            continue; // keep_blank_values=False drops a blank value
        }
        let element = known_values(value.chars().map(|c| c as u32 as f64).collect(), PrimitiveKind::String, TrustSpec);
        // "the values are lists of values for each name" — a repeated
        // name accumulates into the one list this row builds per key.
        match keys.iter_mut().find(|key| key.name == name) {
            Some(existing) => existing.value.items.push(element),
            None => keys.push(ObjectKey {
                name: name.to_owned(),
                numeric: false,
                value: known_list(vec![element], TrustSpec),
            }),
        }
    }
    Some(known_object(keys, None, true, TrustSpec, false))
}

/// `urllib.parse.parse_qs(qs)` on a query string this file cannot read
/// exactly — a `qs: str` parameter's own `Σ*` seed, the shape
/// A8.seed.boundary's rows pass. The exact key set and the exact values
/// are unknowable without the text, but the cited clause still pins the
/// SHAPE of every result the call can produce: "Data are returned as a
/// dictionary. The dictionary keys are the unique query variable names
/// and the values are lists of values for each name." Both halves are
/// strings — a query string carries no other sort — so the answer is the
/// unbounded-key mapping (`known_dict_star`, the same shape
/// `check::seed_parameters` builds for a `dict[str, X]` parameter) whose
/// value at every present key is a LIST of whole strings.
///
/// The list's own length is unstated (a name may repeat any number of
/// times), so it is the bare unbounded repetition window over `Σ*` —
/// the identical shape a declared `list[str]` parameter seeds, which
/// means `params.get("code")` then reads "a list of strings, or None"
/// through the existing dict-star `.get` arm, and `v[0]` reads `Σ*`
/// through the existing repetition-window subscript arm.
///
/// `None` for a `qs` that is not string-SORTED at all
/// (`is_string_sorted_argument`, the same test `quote`'s row takes) — a
/// numeric or unread argument states nothing this row could shape an
/// answer around.
fn parse_qs_unread_query(query: &AbstractValue) -> Option<AbstractValue> {
    // Read through the same string-sortedness test `quote`'s own row
    // takes: a declared `str` parameter's seed is a sequence-shaped
    // `Kind::Set` that carries NO scalar `kind_tag`, so requiring
    // `Some(PrimitiveKind::String)` here declined exactly the shape
    // A8.seed.boundary's rows pass — `parse_qs(qs)` on a `qs: str`
    // parameter — and left every read through the result with no
    // reading at all.
    if !is_string_sorted_argument(query) {
        return None;
    }
    let value_element = AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, TrustSpec, SetKindTag::None)
    };
    let value_list = AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(
            refined_sets::repetition_window_forms::repetition(value_element.set, 0, None),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let (star, built) = refined_domain::known_constructors::known_dict_star(value_list, TrustSpec);
    built.then_some(star)
}

/// Whether `urllib.parse.quote` leaves this code point untouched: the
/// doc's own never-quoted set ("Letters, digits, and the characters
/// `_.-~` are never quoted") plus the one-argument form's `safe='/'`
/// default. Letters and digits here are the ASCII ones — `quote`
/// percent-encodes every non-ASCII code point via its UTF-8 bytes.
fn is_never_quoted(point: f64) -> bool {
    let Some(character) = char::from_u32(point as i64 as u32) else {
        return false;
    };
    character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '-' | '~' | '/')
}
