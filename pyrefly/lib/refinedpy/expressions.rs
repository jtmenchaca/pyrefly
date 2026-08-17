/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Expression evaluation into abstract values: literals, name reads
//! from the environment, unary minus, and arithmetic whose CPython row
//! is cited in PYREFLY-NUMERIC-B3-B4.md. This file is the contract the
//! walk calls; the expressions unit fills it in construct by construct.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;

use crate::refinedpy::env::Environment;

/// What this expression evaluates to in this environment. `unknown()`
/// is the honest default for every construct not yet built — an
/// unknown never fires and never silently passes a judgment.
pub fn evaluate_expression(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    match expression {
        // parenthesization carries no AST node of its own — ruff folds
        // `(x)` into `x` at parse time, so there is no case to write here
        Expr::NumberLiteral(literal) => number_literal_value(&literal.value),
        Expr::BooleanLiteral(literal) => {
            known_values(vec![if literal.value { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved)
        }
        // None is Python's one absent value — Kind::Null is the closest
        // faithful representation refined_domain carries (undef and null
        // both exist; None matches null_value's "the exactly-absent
        // marker" shape more than a wrapped maybe)
        Expr::NoneLiteral(_) => null_value(),
        Expr::Name(name) => match environment.read(name.id.as_str()) {
            Some(value) => value.clone(),
            None => unknown(),
        },
        Expr::UnaryOp(unary) => evaluate_unary(unary, environment, kernel),
        Expr::BinOp(binop) => evaluate_binop(binop, environment, kernel),
        _ => unknown(),
    }
}

/// A NumberLiteral's own value: an int that fits i64 tags `Integer`, a
/// float literal tags `Float` — the syntax's own sort, read once at the
/// value's construction rather than re-derived from the AST at every
/// arithmetic site (PYREFLY-NUMERIC-B3-B4.md's "two sorts, never one
/// Number"). A complex literal, or an int too big for i64, is honest
/// unknown rather than a truncated stand-in.
fn number_literal_value(number: &Number) -> AbstractValue {
    match number {
        Number::Int(int) => match int.as_i64() {
            Some(value) => known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved),
            None => unknown(),
        },
        Number::Float(value) => known_values(vec![*value], PrimitiveKind::Float, TrustProved),
        Number::Complex { .. } => unknown(),
    }
}

/// `-x` / `+x` over a known single numeric value. Any other operand
/// (not known, or known but not exactly one number) is unknown — a
/// unary minus over a set or a multi-valued state states nothing exact.
/// The operand's own sort survives unary +/-: `-3` is still `int`
/// (expressions §6.6 — unary arithmetic states no widening), and a
/// Boolean operand (`bool` is an `int` subclass) becomes an ordinary
/// `Integer` result the same way arithmetic on booleans always does.
fn evaluate_unary(
    unary: &ruff_python_ast::ExprUnaryOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let operand = evaluate_expression(&unary.operand, environment, kernel);
    let Some((value, sort)) = single_numeric_value(&operand) else {
        return unknown();
    };
    match unary.op {
        UnaryOp::USub => known_values(vec![-value], sort, TrustProved),
        UnaryOp::UAdd => known_values(vec![value], sort, TrustProved),
        // `~x` and `not x` are not numeric arithmetic transfers this
        // wave builds
        UnaryOp::Invert | UnaryOp::Not => unknown(),
    }
}

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
fn single_numeric_value(value: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
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

/// Binary arithmetic over two known single numeric values, for exactly
/// the operators PYREFLY-NUMERIC-B3-B4.md cites a CPython row for:
/// `+ - * / // % **`. Every row below follows the cited clause exactly;
/// an operator this file does not recognize, or operands this file
/// cannot prove numeric, decline to unknown().
///
/// EXPORTED: `loops.rs`'s `AugAssign` handling (`total += age`) calls
/// this directly so an augmented assignment agrees with the equivalent
/// `total = total + age` BinOp exactly — one arithmetic transfer, not
/// two independently maintained copies.
pub fn binary_arithmetic_value(op: Operator, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    let Some((left_value, left_sort)) = single_numeric_value(left) else {
        return unknown();
    };
    let Some((right_value, right_sort)) = single_numeric_value(right) else {
        return unknown();
    };
    // int op int -> int (PYREFLY-NUMERIC-B3-B4.md's own kernel-transfer
    // rows); either operand float -> the result widens to float per
    // stdtypes' mixed-arithmetic rule. `/` overrides this below — true
    // division is ALWAYS float, even int/int.
    let both_int = left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer;
    match op {
        Operator::Add => arithmetic_result(left_value + right_value, both_int),
        Operator::Sub => arithmetic_result(left_value - right_value, both_int),
        Operator::Mult => arithmetic_result(left_value * right_value, both_int),
        // `/` is ALWAYS true division in Python: int/int gives float
        // (expressions §6.7). Division by zero raises ZeroDivisionError
        // rather than producing ±Infinity/NaN — this file has no
        // exception channel, so a zero divisor declines to unknown()
        // rather than answering IEEE's ±Infinity.
        Operator::Div => {
            if right_value == 0.0 {
                unknown()
            } else {
                known_values(vec![left_value / right_value], PrimitiveKind::Float, TrustProved)
            }
        }
        // `//` floors toward negative infinity for both int and float
        // operands (expressions §6.7 note 1). Division by zero raises;
        // this file declines the same way `/` does.
        Operator::FloorDiv => {
            if right_value == 0.0 {
                unknown()
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
                unknown()
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
        // `@`, shifts, and bitwise ops have no cited CPython row for
        // exact-value arithmetic transfer in this wave
        Operator::MatMult
        | Operator::LShift
        | Operator::RShift
        | Operator::BitOr
        | Operator::BitXor
        | Operator::BitAnd => unknown(),
    }
}

fn evaluate_binop(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let left = evaluate_expression(&binop.left, environment, kernel);
    let right = evaluate_expression(&binop.right, environment, kernel);
    binary_arithmetic_value(binop.op, &left, &right)
}

/// Wraps an arithmetic result as known_values, honestly: an int result
/// stays exact only while it still fits an f64's 53-bit exact-integer
/// range (2^53) — CPython ints are unbounded, but this file's carrier is
/// f64, so a result outside that range is no longer provably exact and
/// declines rather than silently truncating. `both_int` selects the
/// Python sort: `Integer` when both operands were int-sorted (and the
/// value stays exact), `Float` otherwise — the mixed-arithmetic widening
/// rule (stdtypes' Numeric Types) and `/`'s own always-float override
/// both route through this by passing `both_int = false`.
fn arithmetic_result(value: f64, both_int: bool) -> AbstractValue {
    if both_int {
        if value.fract() != 0.0 || value.abs() >= 2f64.powi(53) {
            return unknown();
        }
        return known_values(vec![value], PrimitiveKind::Integer, TrustProved);
    }
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use refined_kernel::kernel_interface::RefinedTSKernel;
    use ruff_python_parser::parse_expression;

    use super::*;

    /// A kernel handle for tests that never ask it — evaluate_expression
    /// takes the parameter for the contract's sake but no construct this
    /// wave asks a question of it. `None` when the native dylib artifact
    /// has not been built (same skip check.rs's own tests use), so this
    /// file's tests run without requiring `pnpm kernel:native` first.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn empty_environment() -> Environment {
        Environment::new(HashSet::new())
    }

    fn eval(source: &str) -> Option<AbstractValue> {
        let kernel = loaded_kernel()?;
        let parsed = parse_expression(source).expect("test source must parse");
        let expression = parsed.into_expr();
        let environment = empty_environment();
        Some(evaluate_expression(&expression, &environment, &kernel))
    }

    #[test]
    fn test_int_literal() {
        let Some(value) = eval("7") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.values, vec![7.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_float_literal() {
        let Some(value) = eval("3.5") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.values, vec![3.5]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_negative_int_literal() {
        let Some(value) = eval("-7") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.values, vec![-7.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_name_bound() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("x").expect("test source must parse");
        let expression = parsed.into_expr();
        let mut environment = empty_environment();
        environment.bind("x", known_values(vec![42.0], PrimitiveKind::Integer, TrustProved));
        let value = evaluate_expression(&expression, &environment, &kernel);
        assert_eq!(value.values, vec![42.0]);
    }

    /// A name bound to an Integer-sorted value keeps the Integer tag
    /// through `a + 1` — the arithmetic transfer reads the BOUND
    /// value's own sort (never re-derives it syntactically from the
    /// name), so `both_int` sees Integer op Integer here.
    #[test]
    fn test_name_bound_int_keeps_integer_sort_through_addition() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("a + 1").expect("test source must parse");
        let expression = parsed.into_expr();
        let mut environment = empty_environment();
        environment.bind("a", known_values(vec![10.0], PrimitiveKind::Integer, TrustProved));
        let value = evaluate_expression(&expression, &environment, &kernel);
        assert_eq!(value.values, vec![11.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_name_unbound() {
        let Some(value) = eval("y") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_add_int() {
        let Some(value) = eval("2 + 3") else { return };
        assert_eq!(value.values, vec![5.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_sub_int() {
        let Some(value) = eval("5 - 8") else { return };
        assert_eq!(value.values, vec![-3.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_mult_int() {
        let Some(value) = eval("4 * 6") else { return };
        assert_eq!(value.values, vec![24.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `/` is ALWAYS true division in Python — the result is Float-sorted
    /// even when both operands are int-sorted and the quotient is whole
    /// (6 / 3 == 2.0, not the int 2). This is the row the mission's
    /// int-sort fire depends on: a Float-tagged `6 / 3` assigned into an
    /// int-sorted alias must fire, not silently pass as if it were `int`.
    #[test]
    fn test_true_division_of_two_ints_is_float_tagged_even_on_a_whole_quotient() {
        let Some(value) = eval("6 / 3") else { return };
        assert_eq!(value.values, vec![2.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_true_division_int_gives_float() {
        // 7 / 2 == 3.5 — Python `/` is always true division
        let Some(value) = eval("7 / 2") else { return };
        assert_eq!(value.values, vec![3.5]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_floor_division_negative_floors_toward_negative_infinity() {
        // -7 // 2 == -4 (not -3, which truncation toward zero would give)
        let Some(value) = eval("-7 // 2") else { return };
        assert_eq!(value.values, vec![-4.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_mod_sign_follows_divisor_negative_divisor() {
        // -7 % 2 == 1 — sign of the result follows the divisor (2, positive)
        let Some(value) = eval("-7 % 2") else { return };
        assert_eq!(value.values, vec![1.0]);
    }

    #[test]
    fn test_mod_sign_follows_divisor_negative_dividend_side() {
        // 7 % -2 == -1 — sign of the result follows the divisor (-2, negative)
        let Some(value) = eval("7 % -2") else { return };
        assert_eq!(value.values, vec![-1.0]);
    }

    #[test]
    fn test_pow_int_exact() {
        let Some(value) = eval("2 ** 10") else { return };
        assert_eq!(value.values, vec![1024.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `int ** negative int` converts to float per §6.5 / stdtypes note
    /// (5) — `10 ** -2 == 0.01`, Float-sorted even though both operands
    /// were Integer-sorted.
    #[test]
    fn test_pow_negative_int_exponent_widens_to_float() {
        let Some(value) = eval("10 ** -2") else { return };
        assert!((value.values[0] - 0.01).abs() < 1e-12);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn test_division_by_zero_declines() {
        let Some(value) = eval("1 / 0") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_boolean_literal_true() {
        let Some(value) = eval("True") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.values, vec![1.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
    }

    /// `True + True == 2` — Python's `bool` is an `int` subclass, so
    /// arithmetic on booleans reads them as Integer and yields an
    /// ordinary Integer-sorted result (AGENT-BRIEF.md).
    #[test]
    fn test_boolean_arithmetic_yields_integer_sort() {
        let Some(value) = eval("True + True") else { return };
        assert_eq!(value.values, vec![2.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_none_literal() {
        let Some(value) = eval("None") else { return };
        assert_eq!(value.kind, Kind::Null);
    }

    #[test]
    fn test_unsupported_construct_is_unknown() {
        // a call is not modeled this wave
        let Some(value) = eval("f(1)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `binary_arithmetic_value` directly, no kernel needed (pure
    /// computation over two known AbstractValues) — pins the exported
    /// signature `loops.rs`'s AugAssign path calls, and the sort rule a
    /// mixed Integer/Float `+` widens to Float per stdtypes' own mixed-
    /// arithmetic rule.
    #[test]
    fn test_binary_arithmetic_value_mixed_sort_widens_to_float() {
        let ten_int = known_values(vec![10.0], PrimitiveKind::Integer, TrustProved);
        let half_float = known_values(vec![0.5], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value(Operator::Add, &ten_int, &half_float);
        assert_eq!(result.values, vec![10.5]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }
}
