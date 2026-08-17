/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Calls to Python builtins with determinable results, answered exactly.
//! One dispatcher — `builtin_call_result` — takes the callee name and the
//! already-evaluated argument values; `None` means "not modeled here" (the
//! caller declines honestly), `Some` is an exact answer. Every modeled row
//! cites its clause of docs.python.org/3.12/library/functions.html; a row
//! with no citation is not written.

use refined_domain::abstract_value::{known_values, AbstractValue, Kind, PrimitiveKind};
use refined_domain::trust_grades::{derived_trust_level, TrustSpec};

/// Read a single known numeric value out of an argument: `Kind::Values`,
/// tagged `Integer` or `Float`, carrying exactly one element. Every row
/// below that needs "one known number" reads through this rather than
/// re-matching the shape.
fn single_known_numeric(argument: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
    if argument.kind != Kind::Values {
        return None;
    }
    if argument.values.len() != 1 {
        return None;
    }
    match argument.kind_tag {
        Some(PrimitiveKind::Integer) => Some((argument.values[0], PrimitiveKind::Integer)),
        Some(PrimitiveKind::Float) => Some((argument.values[0], PrimitiveKind::Float)),
        _ => None,
    }
}

/// `abs(x)` on a single known numeric — library/functions.html#abs:
/// "Return the absolute value of a number." Sort is preserved: an int
/// argument's absolute value is an int, a float's a float — abs never
/// changes the numeric sort of its single argument.
fn abs_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.abs()], sort, grade))
}

/// `round(x)`, single-argument — library/functions.html#round: "If
/// ndigits is omitted or is None, it returns the nearest integer to its
/// input," rounding "toward the even choice" on a tie (banker's
/// rounding — `round(0.5)` and `round(-0.5)` are both `0`, `round(1.5)`
/// is `2`). The two-argument form `round(x, n)` is not modeled: it keeps
/// the input's sort (int stays int, float stays float) rather than
/// always producing an int, a different row this dispatcher does not
/// yet answer.
fn round_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(
        vec![value.round_ties_even()],
        PrimitiveKind::Integer,
        grade,
    ))
}

/// `min`/`max` over two or more known single-numeric arguments —
/// library/functions.html#min and #max: "If two or more positional
/// arguments are provided, the smallest [largest] of the positional
/// arguments is returned." The single-iterable form (`min(some_list)`)
/// is not modeled here — that argument is not a known scalar, so
/// `single_known_numeric` declines it and the whole call declines.
/// Result sort: Python's min/max return the winning ARGUMENT unchanged,
/// so a Float argument winning over Integer arguments keeps Float — the
/// winning value's own sort is threaded through, not fixed at one sort.
fn min_max_call(
    arguments: &[AbstractValue],
    pick: fn(f64, f64) -> bool,
) -> Option<AbstractValue> {
    if arguments.len() < 2 {
        return None;
    }
    let mut best: Option<(f64, PrimitiveKind)> = None;
    for argument in arguments {
        let candidate = single_known_numeric(argument)?;
        best = Some(match best {
            None => candidate,
            Some(current) => {
                if pick(candidate.0, current.0) {
                    candidate
                } else {
                    current
                }
            }
        });
    }
    let (value, sort) = best?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value], sort, grade))
}

/// `int(x)` on a single known numeric — library/functions.html#int:
/// "For floating-point numbers, this truncates towards zero." An
/// already-Integer argument is the identity read under this row (the
/// same trunc-toward-zero rule with no fractional part to discard).
/// `int(str)` is not modeled: a string argument is never a
/// `single_known_numeric`, so the call declines, matching the row's
/// own scope (numeric argument only).
fn int_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.trunc()], PrimitiveKind::Integer, grade))
}

/// `float(x)` on a single known numeric — library/functions.html#float:
/// "Return a floating-point number constructed from a number or a
/// string." Restricted here to the numeric argument: a string argument
/// is never a `single_known_numeric`, so `float(str)` declines rather
/// than being answered by this row.
fn float_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value], PrimitiveKind::Float, grade))
}

/// The dispatcher: a call to Python builtin `function` with already-
/// evaluated `arguments`. `None` means "not modeled here" — the caller
/// declines honestly rather than reading this as "the call is unknown to
/// Python." `Some` is an exact answer at the derived trust grade.
pub fn builtin_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "abs" => abs_call(arguments),
        "round" => round_call(arguments),
        "min" => min_max_call(arguments, |candidate, current| candidate < current),
        "max" => min_max_call(arguments, |candidate, current| candidate > current),
        // len() declines for now: answering it needs container states
        // (string/list/tuple/dict length facts) this domain does not yet
        // carry — single_known_numeric only ever reads a known SCALAR,
        // never a container, so there is no row to write until container
        // states land.
        "len" => None,
        "int" => int_call(arguments),
        "float" => float_call(arguments),
        // sum() declines: it reads an iterable's elements, and iterable
        // states are not yet carried by this domain (the same gap as
        // len()) — no row to write until they are.
        "sum" => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustSpec)
    }

    fn float(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Float, TrustSpec)
    }

    #[test]
    fn round_half_to_even_rounds_up_at_odd_tenths() {
        // round(201.5) == 202: 201.5 sits between 201 and 202; 202 is
        // the even choice.
        let got = builtin_call_result("round", &[float(201.5)]).expect("round(201.5) models");
        assert_eq!(got.values, vec![202.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn round_half_to_even_rounds_down_at_even_tenths() {
        // round(40.5) == 40: 40.5 sits between 40 and 41; 40 is the even
        // choice — the AGENT-BRIEF row-inverting fact against a naive
        // round-half-up reading.
        let got = builtin_call_result("round", &[float(40.5)]).expect("round(40.5) models");
        assert_eq!(got.values, vec![40.0]);
    }

    #[test]
    fn round_two_argument_form_declines() {
        let got = builtin_call_result("round", &[float(40.5), integer(1.0)]);
        assert!(got.is_none(), "round(x, n) should decline: {got:?}");
    }

    #[test]
    fn abs_of_negative_integer_is_positive_integer() {
        let got = builtin_call_result("abs", &[integer(-200.0)]).expect("abs(-200) models");
        assert_eq!(got.values, vec![200.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn int_truncates_toward_zero_on_positive_fraction() {
        let got = builtin_call_result("int", &[float(7.9)]).expect("int(7.9) models");
        assert_eq!(got.values, vec![7.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn int_truncates_toward_zero_on_negative_fraction() {
        // int(-7.9) == -7, not -8: truncation toward zero, not floor.
        let got = builtin_call_result("int", &[float(-7.9)]).expect("int(-7.9) models");
        assert_eq!(got.values, vec![-7.0]);
    }

    #[test]
    fn int_of_string_declines() {
        let string_argument = known_values(
            vec![55.0, 53.0],
            PrimitiveKind::String,
            TrustSpec,
        );
        let got = builtin_call_result("int", &[string_argument]);
        assert!(got.is_none(), "int(str) should decline: {got:?}");
    }

    #[test]
    fn min_over_known_numerics_picks_the_smallest() {
        let got = builtin_call_result("min", &[integer(3.0), integer(-1.0), integer(5.0)])
            .expect("min(...) models");
        assert_eq!(got.values, vec![-1.0]);
    }

    #[test]
    fn max_over_known_numerics_picks_the_largest() {
        let got = builtin_call_result("max", &[integer(3.0), integer(-1.0), integer(5.0)])
            .expect("max(...) models");
        assert_eq!(got.values, vec![5.0]);
    }

    #[test]
    fn max_threads_the_winning_arguments_own_sort() {
        // 4.5 (float) beats 3 (int): the winner's own Float sort carries
        // through, matching Python's min/max returning the argument
        // itself unchanged.
        let got = builtin_call_result("max", &[integer(3.0), float(4.5)]).expect("max(...) models");
        assert_eq!(got.values, vec![4.5]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn min_single_argument_declines() {
        // min(some_list) reads an iterable, not two-or-more scalars —
        // out of this row's modeled shape.
        let got = builtin_call_result("min", &[integer(3.0)]);
        assert!(got.is_none(), "min(x) with one argument should decline: {got:?}");
    }

    #[test]
    fn len_declines() {
        let got = builtin_call_result("len", &[integer(3.0)]);
        assert!(got.is_none());
    }

    #[test]
    fn sum_declines() {
        let got = builtin_call_result("sum", &[integer(3.0)]);
        assert!(got.is_none());
    }

    #[test]
    fn unmodeled_name_declines() {
        let got = builtin_call_result("print", &[integer(3.0)]);
        assert!(got.is_none(), "an unmodeled builtin name should decline: {got:?}");
    }
}
