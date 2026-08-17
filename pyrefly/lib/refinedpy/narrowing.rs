/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Condition narrowing: what a test being true (or false) says about
//! the names it mentions. The walk forks an environment per branch arm
//! and asks this module to tighten each fork before walking the arm.
//! This file is the contract the walk calls; the narrowing unit fills
//! it in behind these signatures.
//!
//! Every narrowing here acts on `Kind::Values` bindings only — the
//! exact-values state a name can hold before a branch. A binding of any
//! other kind (including `Kind::Null`) passes through unchanged: the
//! honest default is to narrow nothing, and this wave never builds set
//! machinery (no kernel questions, no RefinedSet unions).
//!
//! Chained comparisons lower to a conjunction of adjacent pairs
//! (`a op1 b op2 c` == `a op1 b and b op2 c`, CPython
//! tmp/cpython/Doc/reference/expressions.rst — Comparisons, "Comparisons
//! can be chained arbitrarily"), so `ExprCompare`'s multi-op form and
//! `and`'s multi-value form share one conjunction helper.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::trust_level_of;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;

use crate::refinedpy::env::Environment;

/// Tighten `environment` by what `condition` being `truth` says.
/// Returns the narrowed environment for that arm. The honest default
/// narrows nothing — an arm walked with the unnarrowed fork is
/// conservative, never wrong. `kernel` is threaded for the frozen
/// signature; nothing this wave asks it a question.
pub fn assume(
    condition: &Expr,
    environment: Environment,
    _kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) -> Environment {
    let mut environment = environment;
    narrow(condition, &mut environment, truth);
    environment
}

/// Tightens `environment` in place by what `condition` being `truth`
/// says, dispatching on the condition's shape. Every arm that cannot
/// narrow simply returns without touching `environment`.
fn narrow(condition: &Expr, environment: &mut Environment, truth: bool) {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            narrow(&unary.operand, environment, !truth);
        }
        Expr::BoolOp(bool_op) => narrow_bool_op(bool_op, environment, truth),
        Expr::Compare(compare) => narrow_compare(compare, environment, truth),
        Expr::Call(call) => narrow_isinstance_call(call, environment, truth),
        // Calls other than isinstance, attributes, walrus, `in`, string
        // comparisons, and everything else this wave does not read: no
        // narrowing, the honest default.
        _ => {}
    }
}

/// `and`/`or` narrowing (mission point 3): `and` under truth, and `or`
/// under falsity (De Morgan — `not (a or b)` == `not a and not b`),
/// both apply every operand in conjunction. `and` under falsity and
/// `or` under truth narrow nothing this wave — either arm alone could
/// have made the whole true/false, so no single operand's negation (or
/// affirmation) is forced, and this wave builds no set-union machinery
/// to hold the "at least one of" case.
fn narrow_bool_op(bool_op: &ruff_python_ast::ExprBoolOp, environment: &mut Environment, truth: bool) {
    let conjunction = match (bool_op.op, truth) {
        (BoolOp::And, true) => true,
        (BoolOp::Or, false) => true,
        _ => false,
    };
    if !conjunction {
        return;
    }
    // `or` under falsity narrows each operand by its own negation
    // (De Morgan); `and` under truth narrows each by its own truth.
    let per_operand_truth = bool_op.op == BoolOp::And;
    for value in &bool_op.values {
        narrow(value, environment, per_operand_truth);
    }
}

/// `ExprCompare` narrowing: chained comparisons lower to the
/// conjunction of adjacent pairs (see the module doc's CPython
/// citation). Under falsity, a conjunction narrows nothing (the same
/// rule `and`-under-falsity follows in `narrow_bool_op`) — the chain's
/// negation is a disjunction over which pair failed, and this wave
/// holds no union.
fn narrow_compare(compare: &ruff_python_ast::ExprCompare, environment: &mut Environment, truth: bool) {
    if !truth {
        // is/is not None still narrows under falsity for a single pair
        // (mission point 5) — handled directly here since it is not a
        // conjunction the same way numeric comparisons are.
        if compare.ops.len() == 1 {
            narrow_one_comparison(&compare.left, compare.ops[0], &compare.comparators[0], environment, false);
        }
        return;
    }
    let mut left = compare.left.as_ref();
    for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
        narrow_one_comparison(left, *op, right, environment, true);
        left = right;
    }
}

/// One comparison pair (`left op right`) as a narrowing leaf: is/is not
/// None (mission point 5), then numeric literal-side comparisons
/// (mission point 1), mirrored so the literal may sit on either side.
/// Anything else — a call, an attribute, a string, two changing names —
/// narrows nothing.
fn narrow_one_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    if matches!(op, CmpOp::Is | CmpOp::IsNot) {
        narrow_is_none(left, op, right, environment, truth);
        return;
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return;
    };
    if let (Some(name), Some(literal)) = (name_of(left), literal_number(right)) {
        narrow_name_against_literal(name, numeric_op, literal, environment, truth);
        return;
    }
    if let (Some(literal), Some(name)) = (literal_number(left), name_of(right)) {
        narrow_name_against_literal(name, mirror_cmp_op(numeric_op), literal, environment, truth);
        return;
    }
}

/// The subset of `CmpOp` this wave's numeric side-bounds filter reads:
/// `< <= > >= == !=`. `is`/`is not`/`in`/`not in` are handled
/// elsewhere or not at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumericCmpOp {
    Lt,
    LtE,
    Gt,
    GtE,
    Eq,
    NotEq,
}

fn numeric_cmp_op(op: CmpOp) -> Option<NumericCmpOp> {
    match op {
        CmpOp::Lt => Some(NumericCmpOp::Lt),
        CmpOp::LtE => Some(NumericCmpOp::LtE),
        CmpOp::Gt => Some(NumericCmpOp::Gt),
        CmpOp::GtE => Some(NumericCmpOp::GtE),
        CmpOp::Eq => Some(NumericCmpOp::Eq),
        CmpOp::NotEq => Some(NumericCmpOp::NotEq),
        CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => None,
    }
}

/// Mirrors the operator when the literal was on the left (`k >= x`
/// means `x <= k`).
fn mirror_cmp_op(op: NumericCmpOp) -> NumericCmpOp {
    match op {
        NumericCmpOp::Lt => NumericCmpOp::Gt,
        NumericCmpOp::LtE => NumericCmpOp::GtE,
        NumericCmpOp::Gt => NumericCmpOp::Lt,
        NumericCmpOp::GtE => NumericCmpOp::LtE,
        NumericCmpOp::Eq => NumericCmpOp::Eq,
        NumericCmpOp::NotEq => NumericCmpOp::NotEq,
    }
}

/// Whether a known single value `v` satisfies `v op literal` — applied
/// pointwise over a Values binding's exact members in
/// `narrow_name_against_literal`.
fn satisfies(value: f64, op: NumericCmpOp, literal: f64) -> bool {
    match op {
        NumericCmpOp::Lt => value < literal,
        NumericCmpOp::LtE => value <= literal,
        NumericCmpOp::Gt => value > literal,
        NumericCmpOp::GtE => value >= literal,
        NumericCmpOp::Eq => value == literal,
        NumericCmpOp::NotEq => value != literal,
    }
}

/// Narrows a Values-kind binding named `name` by `name op literal`
/// being `truth` (mission point 1): keep exactly the members
/// satisfying the (possibly negated) predicate. Zero survivors bind the
/// empty Values state — sound infeasibility (this branch arm's Values
/// set is empty, so any read of `name` inside it answers from no
/// members rather than answering unknown).
fn narrow_name_against_literal(
    name: &str,
    op: NumericCmpOp,
    literal: f64,
    environment: &mut Environment,
    truth: bool,
) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // a numeric-sorted binding (any of Number/Integer/Float — sort-
    // unknown or sort-known — or Boolean, which reads numerically the
    // same way Python's `True + True == 2` does); a String/Array-tagged
    // binding is not read as a number here
    if !is_numeric_or_boolean(kind_tag) {
        return;
    }
    let grade = trust_level_of(&current);
    let kept: Vec<f64> = current
        .values
        .iter()
        .copied()
        .filter(|&value| satisfies(value, op, literal) == truth)
        .collect();
    environment.bind(name, known_values(kept, kind_tag, grade));
}

/// Whether `kind_tag` reads numerically for a literal comparison:
/// Number (sort-unknown), Integer, Float, or Boolean (`True`/`False`
/// compare as `1`/`0`). String and Array are not numeric.
fn is_numeric_or_boolean(kind_tag: PrimitiveKind) -> bool {
    matches!(
        kind_tag,
        PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float | PrimitiveKind::Boolean
    )
}

/// `is None` / `is not None` (mission point 5): only a Values-kind
/// binding is touched — the empty Values state means "provably not
/// None among the tracked exact values," which is sound because None
/// itself is never a member of a Values state (Values carries only
/// host-sorted numbers/booleans/strings/arrays, never the absent
/// marker). A non-Values binding (including one already `Kind::Null`)
/// passes through unchanged, per the mission's instruction that
/// non-Values states pass through everywhere this wave.
fn narrow_is_none(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let is_not = op == CmpOp::IsNot;
    let name = if is_none_literal(right) {
        name_of(left)
    } else if is_none_literal(left) {
        name_of(right)
    } else {
        None
    };
    let Some(name) = name else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // `name is None` true, or `name is not None` false, both mean
    // "this Values binding holds None" — impossible for a Values state,
    // so every member is infeasible: bind the empty set.
    // `name is None` false, or `name is not None` true, both mean
    // "not None" — a Values binding already satisfies that for every
    // member, so it is left as is (still narrows nothing further, which
    // is sound: no member is dropped).
    let means_is_none = truth != is_not;
    if means_is_none {
        let grade = trust_level_of(&current);
        environment.bind(name, known_values(Vec::new(), kind_tag, grade));
    }
}

/// `isinstance(name, int | float | bool)` (mission point 6): filters a
/// Values binding by `kind_tag`. `PrimitiveKind::Number` is the
/// sort-unknown numeric tag (AGENT-BRIEF.md, Wave-1 recognition facts —
/// int-vs-float is not yet distinguished at the value level except
/// where the syntax proves it), so a Number-tagged state passes
/// unfiltered both ways: this wave cannot prove which arm of an
/// int/float isinstance test it falls on.
fn narrow_isinstance_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, truth: bool) {
    let Expr::Name(func_name) = call.func.as_ref() else {
        return;
    };
    if func_name.id.as_str() != "isinstance" {
        return;
    }
    if call.arguments.args.len() != 2 {
        return;
    }
    let Some(name) = name_of(&call.arguments.args[0]) else {
        return;
    };
    let Some(tags) = isinstance_type_tags(&call.arguments.args[1]) else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // a sort-unknown Number state cannot be proved in, or proved out,
    // of an int/float isinstance test — pass through unfiltered
    if kind_tag == PrimitiveKind::Number {
        return;
    }
    let matches_tag = tags.contains(&kind_tag);
    if matches_tag == truth {
        // every member already agrees with the test — nothing to drop
        return;
    }
    // the whole binding disagrees with the test — every member is
    // infeasible under this arm
    let grade = trust_level_of(&current);
    environment.bind(name, known_values(Vec::new(), kind_tag, grade));
}

/// The `PrimitiveKind`s an `isinstance` second argument names, for
/// exactly the shapes mission point 6 covers: a bare type name
/// (`int`/`float`/`bool`), or a `|`-chain of them
/// (`isinstance(x, int | float)`). Any other shape (a tuple form, a
/// non-primitive type) answers `None` — not read this wave.
fn isinstance_type_tags(expression: &Expr) -> Option<Vec<PrimitiveKind>> {
    match expression {
        Expr::Name(name) => primitive_kind_of_type_name(name.id.as_str()).map(|tag| vec![tag]),
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let mut left = isinstance_type_tags(&binop.left)?;
            let right = isinstance_type_tags(&binop.right)?;
            left.extend(right);
            Some(left)
        }
        _ => None,
    }
}

fn primitive_kind_of_type_name(name: &str) -> Option<PrimitiveKind> {
    match name {
        "int" => Some(PrimitiveKind::Integer),
        "float" => Some(PrimitiveKind::Float),
        "bool" => Some(PrimitiveKind::Boolean),
        _ => None,
    }
}

/// Whether `expression` is the bare name of a tracked place — the only
/// shape every narrowing leaf here reads on the tested side (mission's
/// "filter what is known, never invent," matching the Go reference's
/// `onPlace` restriction to a single identifier).
fn name_of(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    }
}

fn is_none_literal(expression: &Expr) -> bool {
    matches!(expression, Expr::NoneLiteral(_))
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression). Mirrors the sibling
/// private helpers of the same shape in loops.rs/expressions.rs/
/// surface.rs — each narrowing-adjacent file keeps its own copy rather
/// than sharing a cross-file dependency for one small leaf reader.
fn literal_number(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            Number::Float(value) => Some(*value),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_number(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_domain::abstract_value::known_values;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_parser::parse_expression;

    use super::*;

    /// A kernel handle for tests that never ask it anything — `assume`
    /// takes the parameter for the frozen signature's sake, but no
    /// construct this wave asks a question of it. `None` when the
    /// native dylib artifact has not been built, so this file's tests
    /// run without requiring `pnpm kernel:native` first.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn environment_with(name: &str, values: Vec<f64>, kind_tag: PrimitiveKind) -> Environment {
        let mut locally_bound = HashSet::new();
        locally_bound.insert(name.to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind(name, known_values(values, kind_tag, TrustProved));
        environment
    }

    fn assumed(source: &str, environment: Environment, truth: bool) -> Option<Environment> {
        let kernel = loaded_kernel()?;
        let parsed = parse_expression(source).expect("test source must parse");
        let expression = parsed.into_expr();
        Some(assume(&expression, environment, &kernel, truth))
    }

    #[test]
    fn test_greater_than_literal_keeps_satisfying_drops_others() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![200.0]);
    }

    #[test]
    fn test_greater_than_literal_falsity_flips_the_kept_side() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x > 100", environment, false) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_chained_comparison_keeps_the_middle_window() {
        let environment = environment_with("x", vec![-5.0, 0.0, 60.0, 120.0, 200.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![0.0, 60.0, 120.0]);
    }

    #[test]
    fn test_equality_against_literal_keeps_only_that_value() {
        let environment = environment_with("x", vec![40.0, 41.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x == 40", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_not_wrapped_comparison_flips_truth() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("not (x > 100)", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_and_narrows_both_names() {
        let mut locally_bound = HashSet::new();
        locally_bound.insert("a".to_owned());
        locally_bound.insert("b".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("a", known_values(vec![-1.0, 5.0], PrimitiveKind::Number, TrustProved));
        environment.bind("b", known_values(vec![-2.0, 7.0], PrimitiveKind::Number, TrustProved));
        let Some(narrowed) = assumed("a > 0 and b > 0", environment, true) else {
            return;
        };
        let a = narrowed.read("a").expect("a still bound");
        let b = narrowed.read("b").expect("b still bound");
        assert_eq!(a.values, vec![5.0]);
        assert_eq!(b.values, vec![7.0]);
    }

    #[test]
    fn test_non_values_binding_untouched() {
        use refined_domain::abstract_value::null_value;
        let mut locally_bound = HashSet::new();
        locally_bound.insert("x".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("x", null_value());
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Null);
    }

    #[test]
    fn test_unbound_name_untouched() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        assert!(narrowed.read("x").is_none());
    }
}
