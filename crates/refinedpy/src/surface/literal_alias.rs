//! `type Pick = Literal[...]` and `type PickUnion = Literal[...] |
//! Literal[...]` module-level aliases — the type-alias-RHS twin of
//! `typereading.rs`'s own `declared_refinement`'s `Literal[...]` arm.

use refined_sets::codepoint_sets::string_tuple;
use refined_sets::refinement_forms::{make_refined_set, one_of, union, RefinedSet};
use ruff_python_ast::{Expr, Number, Operator, UnaryOp};

/// `type Pick = Literal[10, 20, 30]` (or a single-member/string-member
/// form) — the type-alias-RHS twin of `typereading.rs`'s
/// `declared_refinement`'s own `Literal[...]` arm (int members build a
/// numeric `one_of`, string members build the union of each member's
/// own singleton tuple, a mixed int/string member list declines whole).
/// Mirrored locally rather than imported: `surface.rs` is imported BY
/// `typereading.rs` (`annotated_expression_set`), so importing the
/// other direction would cycle.
pub(super) fn literal_alias_set(value: &Expr) -> Option<RefinedSet> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let is_literal = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Literal");
    if !is_literal {
        return None;
    }
    if let Some(members) = int_literal_members(subscript.slice.as_ref()) {
        return Some(make_refined_set(vec![one_of(&members)]));
    }
    if let Some(members) = string_literal_members(subscript.slice.as_ref()) {
        return Some(string_literal_set(&members));
    }
    None
}

/// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]` —
/// exactly two `Literal[...]` arms joined by `|`, each read through
/// `literal_alias_set` and folded together by `refinement_forms::union`.
/// Any other union shape (a non-Literal arm, more than two arms — ruff
/// parses a chained `|` as nested `BinOp`s so a third arm would need a
/// second union node this reader does not build) declines.
pub(super) fn literal_union_alias_set(value: &Expr) -> Option<RefinedSet> {
    let Expr::BinOp(binop) = value else {
        return None;
    };
    if binop.op != Operator::BitOr {
        return None;
    }
    let left = literal_alias_set(binop.left.as_ref())?;
    let right = literal_alias_set(binop.right.as_ref())?;
    Some(make_refined_set(vec![union(left, right)]))
}

/// `Literal[...]`'s slice read as a list of int-literal members —
/// `typereading.rs::int_literal_members`'s exact twin (see that
/// function's doc for the all-or-nothing member-list rule).
fn int_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(int_literal_value).collect();
    }
    Some(vec![int_literal_value(slice)?])
}

/// One `Literal[...]` member read as an int, with unary minus —
/// `typereading.rs::int_literal_value`'s exact twin.
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
/// `typereading.rs::string_literal_members`'s exact twin.
fn string_literal_members(slice: &Expr) -> Option<Vec<String>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(string_literal_value).collect();
    }
    Some(vec![string_literal_value(slice)?])
}

/// One `Literal[...]` member read as a plain string literal —
/// `typereading.rs::string_literal_value`'s exact twin.
fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
        _ => None,
    }
}

/// The UNION of every member's own singleton string tuple —
/// `typereading.rs::string_literal_set`'s exact twin.
fn string_literal_set(members: &[String]) -> RefinedSet {
    let mut set = string_tuple(&members[0]);
    for member in &members[1..] {
        set = make_refined_set(vec![union(set, string_tuple(member))]);
    }
    set
}
