use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use ruff_python_ast::Operator;

use crate::expressions::sequence_ops::arithmetic_result;
use crate::expressions::sequence_ops::exact_int_arithmetic;
use crate::expressions::sequence_ops::f64_to_exact_i64;

use super::sequence_row::sequence_binop_value;

/// The single numeric value a known abstract value carries, if it
/// carries exactly one, plus the PYTHON ARITHMETIC SORT it reads under.
/// Integer-, Float-, Boolean-, and bare Number-sorted values are all
/// safe to feed into arithmetic: a Boolean operand reads as `Integer`
/// (Python's own `bool` is an `int` subclass, `True + True == 2`,
/// AGENT-BRIEF.md); a bare `Number`-tagged value (a join of an Integer
/// and a Float arm, or a caller that has not yet threaded a Python sort
/// through — `loops.rs`'s own `known_number` helper) has no single
/// Python sort PROVED, so it reads conservatively as `Float` — the same
/// "unproven int reads as the float row" rule AGENT-BRIEF.md's Wave-1
/// recognition facts already name, never widened silently to `Integer`.
/// A String/Array word is the one shape still refused outright.
pub(in crate::expressions) fn single_numeric_value(value: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
    if value.kind != Kind::Values {
        return None;
    }
    if value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Integer) => Some((value.values[0], PrimitiveKind::Integer)),
        Some(PrimitiveKind::Float) => Some((value.values[0], PrimitiveKind::Float)),
        Some(PrimitiveKind::Boolean) => Some((value.values[0], PrimitiveKind::Integer)),
        Some(PrimitiveKind::Number) => Some((value.values[0], PrimitiveKind::Float)),
        _ => None,
    }
}

/// Binary arithmetic over two known numeric operands, for exactly the
/// operators PYREFLY-NUMERIC-B3-B4.md cites a CPython row for: `+ - *
/// / // % **`. Two known SINGLE values answer through
/// `binary_arithmetic_pair` directly; a MULTI-valued `Kind::Values`
/// operand on either side (an ordinary join of admitted literals, e.g.
/// a loop's second judged pass) answers through
/// `multi_value_binary_arithmetic`'s own pointwise cross product
/// instead, before ever falling through to the sequence row. An
/// operator this file does not recognize, or operands this file cannot
/// prove numeric AT ALL, declines to the sequence row below (a
/// non-numeric `+`/`*` — string/list concatenation or repetition,
/// `sequence_binop_value`'s own doc) — the same decline order
/// `evaluate_binop` already reads: numeric first, then sequence.
///
/// EXPORTED: `loops.rs`'s `AugAssign` handling (`total += age`,
/// `label += "c"`) calls this directly so an augmented assignment
/// agrees with the equivalent `total = total + age` / `label = label +
/// "c"` BinOp exactly — one arithmetic-and-sequence transfer, not two
/// independently maintained copies. Without this fallthrough, a string
/// `+=` would silently decline through the numeric-only row alone
/// (`single_numeric_value` never accepts a String-sorted operand),
/// diverging from what the equivalent BinOp answers.
pub fn binary_arithmetic_value(op: Operator, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    // `x ** 0` is exactly `1` for EVERY `x` (expressions.rst's power
    // operator row states no exception for the base) — pinned ahead of
    // BOTH numeric gates below because a `Kind::NaN` base (`float("nan")`)
    // fails `single_numeric_value` and would otherwise fall through to
    // `sequence_binop_value`'s own `_ => unknown()` `Pow` row, losing this
    // closed fact for the one base shape that carries no numeric read at
    // all. Mirrors `pow_over_sets`' own `k = 0` branch (the SET-shaped
    // sibling of this same corner). `left_sort` from `single_numeric_value`
    // when the base itself reads as a known numeric (an exact `0 ** 0`,
    // matching `binary_arithmetic_pair`'s own row for that pair, still
    // reaches the ordinary path below unchanged since it never hits this
    // arm's own `single_numeric_value(left)` failure) — this arm exists
    // ONLY for a base `single_numeric_value` cannot read, so `left_sort`
    // here is always `PrimitiveKind::Float`: the one non-numeric-read base
    // shape admitted, `Kind::NaN`, is always Python `float`.
    if op == Operator::Pow && single_numeric_value(left).is_none() && left.kind == Kind::NaN {
        if let Some((right_value, _)) = single_numeric_value(right) {
            if right_value == 0.0 {
                let grade = refined_domain::trust_grades::derived_trust_level(
                    refined_domain::trust_grades::TrustProved,
                    &[left.clone(), right.clone()],
                );
                return known_values(vec![1.0], PrimitiveKind::Float, grade);
            }
        }
    }
    let Some((left_value, left_sort)) = single_numeric_value(left) else {
        return multi_value_binary_arithmetic(op, left, right).unwrap_or_else(|| sequence_binop_value(op, left, right));
    };
    let Some((right_value, right_sort)) = single_numeric_value(right) else {
        return multi_value_binary_arithmetic(op, left, right).unwrap_or_else(|| sequence_binop_value(op, left, right));
    };
    binary_arithmetic_pair(op, left_value, left_sort, right_value, right_sort)
}

/// Binary arithmetic over two known SINGLE numeric values — the exact
/// per-pair rule every operator in PYREFLY-NUMERIC-B3-B4.md's cited
/// CPython row follows: `+ - * / // % ** << >> & | ^`. Factored out of
/// `binary_arithmetic_value` so `multi_value_binary_arithmetic`'s own
/// cross-product can call the identical per-pair arithmetic CPython
/// itself runs at each admitted combination, rather than re-deriving
/// it. An operator this file does not recognize, or a pair this file
/// cannot prove exact for (a zero divisor, an out-of-2^53-range
/// result, a non-integer bitwise operand, …), answers `unknown()` — the
/// caller's own decline discipline, unchanged from before this split.
pub(in crate::expressions) fn binary_arithmetic_pair(
    op: Operator,
    left_value: f64,
    left_sort: PrimitiveKind,
    right_value: f64,
    right_sort: PrimitiveKind,
) -> AbstractValue {
    // int op int -> int (PYREFLY-NUMERIC-B3-B4.md's own kernel-transfer
    // rows); either operand float -> the result widens to float per
    // stdtypes' mixed-arithmetic rule. `/` overrides this below — true
    // division is ALWAYS float, even int/int.
    let both_int = left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer;
    // Two int-sorted operands compute EXACTLY in i128 first — CPython
    // ints are unbounded, and a value like 2^62 is exactly carried by an
    // f64 even though it sits past the 2^53 window `arithmetic_result`
    // trusts for f64-computed results. The fold answers only when the
    // f64 CARRIES the true integer exactly (the round-trip test inside
    // `exact_int_arithmetic`), so nothing rounded ever folds.
    if both_int {
        if let Some(exact) = exact_int_arithmetic(op, left_value, right_value) {
            return known_values(vec![exact], PrimitiveKind::Integer, refined_domain::trust_grades::TrustProved);
        }
    }
    match op {
        Operator::Add => arithmetic_result(left_value + right_value, both_int),
        Operator::Sub => arithmetic_result(left_value - right_value, both_int),
        Operator::Mult => arithmetic_result(left_value * right_value, both_int),
        // `/` is ALWAYS true division in Python: int/int gives float
        // (expressions §6.7). Division by zero raises ZeroDivisionError
        // rather than producing ±Infinity/NaN — this file has no
        // exception channel, so a zero divisor declines to unknown()
        // rather than answering IEEE's ±Infinity. A non-zero divisor can
        // still divide two infinities (`inf / inf` is NaN, IEEE 754), so
        // this routes through `arithmetic_result` — the same NaN screen
        // every other Float-sorted row here already keeps — rather than
        // build `known_values` directly.
        Operator::Div => {
            if right_value == 0.0 {
                refined_domain::abstract_value::unknown()
            } else {
                arithmetic_result(left_value / right_value, false)
            }
        }
        // `//` floors toward negative infinity for both int and float
        // operands (expressions §6.7 note 1). Division by zero raises;
        // this file declines the same way `/` does.
        Operator::FloorDiv => {
            if right_value == 0.0 {
                refined_domain::abstract_value::unknown()
            } else {
                arithmetic_result((left_value / right_value).floor(), both_int)
            }
        }
        // `%` takes the SIGN OF THE DIVISOR in Python — the opposite of
        // ECMA's dividend-sign remainder (AGENT-BRIEF.md, expressions
        // §6.7). Paired with `//` by `x == (x//y)*y + (x%y)`; computed
        // that way here so the sign identity holds exactly rather than
        // trusting f64 `%`'s own (dividend-sign) convention.
        Operator::Mod => {
            if right_value == 0.0 {
                refined_domain::abstract_value::unknown()
            } else {
                let quotient = (left_value / right_value).floor();
                let remainder = left_value - quotient * right_value;
                arithmetic_result(remainder, both_int)
            }
        }
        // `**` with a non-negative int exponent is exact per §6.5; a
        // negative int exponent converts to float (int ** negative int
        // -> float, PYREFLY-NUMERIC-B3-B4.md) — both rows are pinned, so
        // both are answered; a fractional/negative-base combination that
        // would go complex is outside what an f64 result carries exactly
        // and is left to the general float row below.
        Operator::Pow => {
            if both_int && right_value >= 0.0 && right_value.fract() == 0.0 {
                arithmetic_result(left_value.powf(right_value), true)
            } else {
                arithmetic_result(left_value.powf(right_value), false)
            }
        }
        // `@` has no cited CPython row for exact-value arithmetic
        // transfer in this wave.
        Operator::MatMult => refined_domain::abstract_value::unknown(),
        // `<<`/`>>` on ints are exact per §6.8: `x << n` is `x * 2**n`,
        // `x >> n` is `x // 2**n` (floor division toward negative
        // infinity, matching CPython's own arbitrary-precision shift).
        // A negative shift count raises ValueError in CPython — this
        // file has no exception channel for a binary operator's own
        // decline (the same posture `Div`/`FloorDiv`/`Mod` already take
        // for a zero divisor), so it declines to unknown() rather than
        // claim a value CPython never produces. Both operands must be
        // int-sorted (a float shift operand raises TypeError in
        // CPython) and the shift count must stay small enough that
        // `2**n` is itself f64-exact, or this declines the same way an
        // out-of-2^53-range result already does elsewhere in this
        // function.
        Operator::LShift | Operator::RShift if both_int && right_value >= 0.0 && right_value < 53.0 => {
            let factor = 2f64.powf(right_value);
            if op == Operator::LShift {
                arithmetic_result(left_value * factor, true)
            } else {
                arithmetic_result((left_value / factor).floor(), true)
            }
        }
        Operator::LShift | Operator::RShift => refined_domain::abstract_value::unknown(),
        // `&`/`|`/`^` on ints are exact per §6.8, computed over CPython's
        // conceptually infinite two's-complement representation. Both
        // operands must be int-sorted (a float bitwise operand raises
        // TypeError in CPython) and stay within `i64`'s exact range —
        // this file's carrier is f64, so an operand or result outside
        // i64 declines rather than truncate silently.
        Operator::BitOr | Operator::BitXor | Operator::BitAnd
            if both_int && left_value.fract() == 0.0 && right_value.fract() == 0.0 =>
        {
            let Some(left_int) = f64_to_exact_i64(left_value) else {
                return refined_domain::abstract_value::unknown();
            };
            let Some(right_int) = f64_to_exact_i64(right_value) else {
                return refined_domain::abstract_value::unknown();
            };
            let result = match op {
                Operator::BitOr => left_int | right_int,
                Operator::BitXor => left_int ^ right_int,
                Operator::BitAnd => left_int & right_int,
                _ => unreachable!("guarded to BitOr/BitXor/BitAnd above"),
            };
            arithmetic_result(result as f64, true)
        }
        Operator::BitOr | Operator::BitXor | Operator::BitAnd => refined_domain::abstract_value::unknown(),
    }
}

/// The multi-value cap: two operands whose CROSS PRODUCT exceeds this
/// many combined pairs fall through to the existing set/transfer path
/// unchanged, rather than enumerate an unbounded join as `Kind::Values`
/// — mirrors the boundaryFuelCeiling-style hang guard other kernel-
/// adjacent code in this workspace states explicitly rather than
/// deriving from an existing convention (none was found for THIS
/// carrier: no prior `Kind::Values` cross product exists in this file).
const MULTI_VALUE_CROSS_PRODUCT_CAP: usize = 16;

/// `{a1, a2, ...} op {b1, b2, ...}` — a binary operation over TWO
/// operands where at least one is a MULTI-valued `Kind::Values` binding
/// (an ordinary join of admitted literals, e.g. a loop's second judged
/// pass over `total` after the first pass's own join produced `{0,
/// age}`): the exact pointwise answer, one CPython actually computes at
/// EVERY admitted concrete pair — `{1.0, 2.0} * 2.0` answers `{2.0,
/// 4.0}`, not `unknown()`, because CPython evaluates `1.0 * 2.0` and
/// `2.0 * 2.0` independently at each concrete run and both are exact.
///
/// Every pair goes through the SAME `binary_arithmetic_pair` the
/// single-value path already trusts — this function is a cross product
/// over that function's own answers, never a second arithmetic
/// implementation. A single-valued operand reads as its own one-element
/// list (`single_numeric_value`), so a `{Values} op {single}` pair and
/// a `{Values} op {Values}` pair are the SAME shape here — the
/// single-value CALLER (`binary_arithmetic_value`) never reaches this
/// function at all (it already answers through `binary_arithmetic_pair`
/// directly when BOTH operands are singletons), so this function only
/// ever runs with at least one genuinely multi-valued side.
///
/// A pair `binary_arithmetic_pair` cannot determine (a zero divisor, an
/// out-of-2^53-range result, a non-integer bitwise operand, …) makes
/// the WHOLE cross product decline (`None`) rather than silently drop
/// that one admitted combination from the answer — dropping a value
/// CPython can actually produce would be UNSOUND, the same reason
/// `split_divisor_transfer` never drops its own raise arm silently; it
/// routes that arm to `possible_raise` instead, a channel this function
/// does not own. `Div`/`FloorDiv`/`Mod`'s zero-divisor row therefore
/// still declines the WHOLE combined answer whenever the cross product
/// admits a zero divisor pairing, exactly as `binary_arithmetic_pair`'s
/// own existing single-pair behavior already declines for one. A cross
/// product past `MULTI_VALUE_CROSS_PRODUCT_CAP` also declines here (the
/// caller falls through to the set/transfer path unchanged) rather than
/// enumerate an unbounded join. `None` for every non-multi-valued
/// operand pair (the caller's own single-value path owns those) and
/// for a non-numeric multi-valued operand (a String/Boolean-sorted
/// `Kind::Values` — `single_numeric_value`'s own per-element read
/// already excludes those the same way the single-value path does).
pub(in crate::expressions) fn multi_value_binary_arithmetic(op: Operator, left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let left_pairs = numeric_values_with_sort(left)?;
    let right_pairs = numeric_values_with_sort(right)?;
    if left_pairs.len() <= 1 && right_pairs.len() <= 1 {
        // both sides are already single values — the caller's own
        // `binary_arithmetic_value` path answers this directly through
        // `binary_arithmetic_pair`, never reaching this function
        return None;
    }
    if left_pairs.len().saturating_mul(right_pairs.len()) > MULTI_VALUE_CROSS_PRODUCT_CAP {
        return None;
    }
    let mut combined: Vec<f64> = Vec::with_capacity(left_pairs.len() * right_pairs.len());
    for &(left_value, left_sort) in &left_pairs {
        for &(right_value, right_sort) in &right_pairs {
            let pair_result = binary_arithmetic_pair(op, left_value, left_sort, right_value, right_sort);
            let Some((result_value, _)) = single_numeric_value(&pair_result) else {
                // one admitted combination could not be determined
                // exactly (a zero divisor, an out-of-range result, …) —
                // the whole cross product declines rather than silently
                // omit a value CPython can actually produce
                return None;
            };
            if !combined.contains(&result_value) {
                combined.push(result_value);
            }
        }
    }
    // the RESULT sort follows the same both-int rule
    // `binary_arithmetic_pair` already applies per pair: every pointwise
    // result came back through `arithmetic_result`/`known_values`,
    // which already normalized Integer vs Float per pair — reading the
    // FIRST pair's own kind_tag is sound because every pair shares the
    // same left/right SORTS (not values), so `both_int` (and therefore
    // the result sort) is identical across the whole cross product.
    let result_sort = binary_arithmetic_pair(op, left_pairs[0].0, left_pairs[0].1, right_pairs[0].0, right_pairs[0].1)
        .kind_tag
        .unwrap_or(PrimitiveKind::Float);
    Some(known_values(combined, result_sort, refined_domain::trust_grades::TrustProved))
}

/// Every numeric value a `Kind::Values` binding admits, each paired
/// with the PYTHON ARITHMETIC SORT `single_numeric_value` reads a
/// single value under — the multi-valued generalization of
/// `single_numeric_value` itself (a one-element binding reads
/// identically through either function). `None` for a non-`Kind::Values`
/// operand, an EMPTY `Kind::Values` (nothing to enumerate — should not
/// occur for a real join, but this function makes no assumption), or a
/// non-numeric sort (String/Boolean/anything `single_numeric_value`
/// itself declines) — the same "known operands only" discipline every
/// reader in this file keeps.
pub(in crate::expressions) fn numeric_values_with_sort(value: &AbstractValue) -> Option<Vec<(f64, PrimitiveKind)>> {
    if value.kind != Kind::Values || value.values.is_empty() {
        return None;
    }
    let sort = match value.kind_tag {
        Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
        Some(PrimitiveKind::Float) => PrimitiveKind::Float,
        Some(PrimitiveKind::Boolean) => PrimitiveKind::Integer,
        Some(PrimitiveKind::Number) => PrimitiveKind::Float,
        _ => return None,
    };
    Some(value.values.iter().map(|&v| (v, sort)).collect())
}
