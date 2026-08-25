//! Plain literal readers shared across the surface unit: numbers
//! (including constant-folded arithmetic), strings, and lengths.

use ruff_python_ast::{Expr, Number, Operator, UnaryOp};

/// A numeric literal, with unary minus — the readable-RHS gate for
/// this slice; a constant integer expression over literals
/// (`2**53 + 2`, `2**31 - 1`) folds through `literal_integer_fold`
/// and is accepted only when the folded value converts to f64 without
/// rounding, so the computed spelling of a bound reads exactly as its
/// literal spelling would. None anywhere else (an unread value
/// declines, it never guesses).
pub fn literal_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(f) => Some(*f),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
            Some(-literal_number(unary.operand.as_ref())?)
        }
        Expr::BinOp(_) => {
            let folded = literal_integer_fold(expr)?;
            let as_float = folded as f64;
            if as_float as i64 != folded {
                return None;
            }
            Some(as_float)
        }
        _ => None,
    }
}

/// Constant integer arithmetic over literals, folded exactly in i64:
/// `2**53`, `2**31 - 1`, `60 * 60`. Overflow, a float operand, a
/// division, or any non-literal leaf declines — the fold never
/// approximates.
fn literal_integer_fold(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64(),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
            literal_integer_fold(unary.operand.as_ref())?.checked_neg()
        }
        Expr::BinOp(bin) => {
            let left = literal_integer_fold(bin.left.as_ref())?;
            let right = literal_integer_fold(bin.right.as_ref())?;
            match bin.op {
                Operator::Add => left.checked_add(right),
                Operator::Sub => left.checked_sub(right),
                Operator::Mult => left.checked_mul(right),
                Operator::Pow => left.checked_pow(u32::try_from(right).ok()?),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A plain (non-f-string) string literal — the readable-RHS gate for
/// `pattern=r"…"`. None anywhere else, matching `literal_number`'s
/// decline-don't-guess discipline.
pub(super) fn literal_string(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str()),
        _ => None,
    }
}

/// `min_length`/`max_length`'s literal int argument — pydantic's own
/// `StringConstraints`/`Field` types these as `int`, never a float, so
/// a fractional or non-literal value declines rather than truncating.
pub(super) fn literal_length(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64(),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        _ => None,
    }
}
