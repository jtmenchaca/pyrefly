//! `Literal[...]`'s slice read into its int/bool/string member lists —
//! the wire shapes `declared_refinement`'s own `Literal` arm dispatches
//! across.

use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;

/// `Literal[...]`'s slice read as a list of int-literal members: one
/// bare (possibly negated) `NumberLiteral` for a single-member
/// `Literal[40]`, or every element of an `Expr::Tuple` for
/// `Literal[10, 20]`. `None` the moment any member is not a plain int
/// literal (a string, a bool, a float, a name) — the whole subscript
/// declines rather than reading a partial member list.
pub(super) fn int_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(int_literal_value).collect();
    }
    Some(vec![int_literal_value(slice)?])
}

/// `Literal[...]`'s slice read as a list of BOOL-literal members —
/// `True` encodes 1 and `False` 0, the boolean-domain convention. `None`
/// the moment any member is not a bare bool literal, the same
/// all-or-nothing rule the int and string readers keep.
pub(super) fn bool_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    let bool_literal_value = |expr: &Expr| -> Option<f64> {
        match expr {
            Expr::BooleanLiteral(literal) => Some(if literal.value { 1.0 } else { 0.0 }),
            _ => None,
        }
    };
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(bool_literal_value).collect();
    }
    Some(vec![bool_literal_value(slice)?])
}

/// One `Literal[...]` member read as an int, with unary minus
/// (`Literal[-1]`) — the same shape `surface.rs::literal_number` reads,
/// but INTEGER ONLY: a `Number::Float` member declines, since a float
/// value can never be a `Literal[...]` member in the typing grammar
/// (`tmp/cpython Doc/library/typing.rst`, "Literal" — "Literal[3.14]" is
/// not a valid Literal parameter in the first place; only int, str,
/// bytes, bool, and None literals are).
fn int_literal_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => Some(-int_literal_value(unary.operand.as_ref())?),
        _ => None,
    }
}

/// `Literal[...]`'s slice read as a list of STRING-literal members —
/// `int_literal_members`'s twin. `None` the moment any member is not a
/// plain `Expr::StringLiteral` (an int, a bool, a name, an f-string) —
/// a MIXED int/string `Literal[...]` declines whole, the same
/// all-or-nothing rule `int_literal_members` already applies to a
/// mixed int/name member list.
pub(super) fn string_literal_members(slice: &Expr) -> Option<Vec<String>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(string_literal_value).collect();
    }
    Some(vec![string_literal_value(slice)?])
}

/// One `Literal[...]` member read as a plain string literal — no
/// f-string, no concatenation, the same bare shape `int_literal_value`
/// reads on the numeric side.
fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
        _ => None,
    }
}

/// The UNION of every member's own singleton string tuple
/// (`codepoint_sets::string_tuple`) — the unambiguous string-Literal
/// wire shape `int_literal_members` cannot share: a string member's
/// code points would collide with `one_of`'s numeric encoding, so each
/// member gets its own tuple set and the members fold together by
/// `union`, not by one shared `one_of`. A single member is exactly its
/// own tuple set (no union node needed); `members` is never empty —
/// `string_literal_members` always returns at least one element when it
/// returns at all.
pub(super) fn string_literal_set(members: &[String]) -> RefinedSet {
    let mut set = refined_sets::codepoint_sets::string_tuple(&members[0]);
    for member in &members[1..] {
        set = refined_sets::refinement_forms::make_refined_set(vec![union(set, refined_sets::codepoint_sets::string_tuple(member))]);
    }
    set
}
