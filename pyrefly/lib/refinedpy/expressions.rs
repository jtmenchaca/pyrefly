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

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::opaque_value;
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
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::ConversionFlag;
use ruff_python_ast::Expr;
use ruff_python_ast::InterpolatedStringElement;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::refinedpy::builtin_models;
use crate::refinedpy::bytes_models;
use crate::refinedpy::bytes_models::BytesAnswer;
use crate::refinedpy::collection_models;
use crate::refinedpy::env::Environment;
use crate::refinedpy::instances;
use crate::refinedpy::math_models;
use crate::refinedpy::string_models;
use crate::refinedpy::summaries;

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
        // `__name__` is host-defined (the running module's own identity —
        // "__main__" when run as a script, the dotted module path
        // otherwise) but its SORT is always `str`
        // (tmp/cpython/Doc/reference/import.html#__name__ /
        // Doc/reference/datamodel.rst's module-attribute table both state
        // it as a string attribute) — a sort-only claim, never a specific
        // value, since this file has no host-execution-context knowledge
        // of which module is running. Only when the name is not locally
        // bound: an ordinary variable named `__name__` (shadowing the
        // module attribute) reads through the ordinary Name arm instead.
        Expr::Name(name) if name.id.as_str() == "__name__" && environment.read("__name__").is_none() => {
            known_set(strings(), None, TrustSpec, SetKindTag::None)
        }
        Expr::Name(name) => match environment.read(name.id.as_str()) {
            Some(value) => value.clone(),
            None => unknown(),
        },
        Expr::UnaryOp(unary) => evaluate_unary(unary, environment, kernel),
        Expr::BinOp(binop) => evaluate_binop(binop, environment, kernel),
        Expr::StringLiteral(literal) => string_models::string_literal_value(literal.value.to_str()),
        Expr::BytesLiteral(literal) => evaluate_bytes_literal(literal),
        Expr::List(list) => evaluate_list(list, environment, kernel),
        Expr::Set(set) => evaluate_set(set, environment, kernel),
        Expr::Tuple(tuple) => evaluate_tuple(tuple, environment, kernel),
        Expr::Dict(dict) => evaluate_dict(dict, environment, kernel),
        Expr::Subscript(subscript) => evaluate_subscript(subscript, environment, kernel),
        Expr::Compare(compare) => evaluate_compare(compare, environment, kernel),
        Expr::BoolOp(boolop) => evaluate_boolop(boolop, environment, kernel),
        Expr::FString(fstring) => evaluate_fstring(fstring, environment, kernel),
        Expr::If(ternary) => evaluate_ternary(ternary, environment, kernel),
        Expr::Named(named) => evaluate_expression(&named.value, environment, kernel),
        Expr::Call(call) => evaluate_call(call, environment, kernel),
        Expr::Attribute(attribute) => evaluate_attribute_read(attribute, environment, kernel),
        Expr::ListComp(comp) => evaluate_list_or_set_comp(&comp.elt, &comp.generators, environment, kernel),
        Expr::SetComp(comp) => evaluate_list_or_set_comp(&comp.elt, &comp.generators, environment, kernel),
        Expr::Generator(comp) => evaluate_list_or_set_comp(&comp.elt, &comp.generators, environment, kernel),
        Expr::DictComp(comp) => evaluate_dict_comp(comp, environment, kernel),
        Expr::Await(inner) => evaluate_expression(&inner.value, environment, kernel),
        _ => unknown(),
    }
}

/// `b"..."`/`bytes([...])` literal text — bytes_models.rs's own
/// `Kind::List` shape, one Integer-tagged code-unit per byte
/// (stdtypes.rst, "Bytes and Bytearray Objects": bytes objects are
/// sequences of integers 0-255). `literal.value.bytes()` reads the
/// literal's own raw byte sequence off `ruff_python_ast`'s
/// `BytesLiteralValue` the same way `ExprStringLiteral`'s `to_str()` is
/// already read above.
fn evaluate_bytes_literal(literal: &ruff_python_ast::ExprBytesLiteral) -> AbstractValue {
    let bytes: Vec<u8> = literal.value.bytes().collect();
    bytes_models::bytes_literal_value(&bytes)
}

/// `[a, b, c]` — every element evaluated, then handed to
/// `collection_models::list_literal_value`. A `Starred` element
/// (`[*xs, a]`) unpacks an iterable's contents into the literal at parse
/// time (expressions.rst, "List displays") — this file has no iterable-
/// unpacking transfer, so a starred element declines the WHOLE literal
/// rather than treating the starred expression as one ordinary slot.
fn evaluate_list(list: &ruff_python_ast::ExprList, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(elements) = evaluate_display_elements(list.elts.iter(), environment, kernel) else {
        return unknown();
    };
    collection_models::list_literal_value(&elements)
}

/// `(a, b, c)` — the same element-evaluation and starred-element decline
/// as `evaluate_list`; `collection_models::tuple_literal_value` is the
/// one call that differs (both build the same `Kind::List` shape, per
/// that file's own doc).
fn evaluate_tuple(tuple: &ruff_python_ast::ExprTuple, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(elements) = evaluate_display_elements(tuple.elts.iter(), environment, kernel) else {
        return unknown();
    };
    collection_models::tuple_literal_value(&elements)
}

/// `{a, b, c}` — a set DISPLAY, evaluated the same way a list display
/// is: every element in order, a starred element declining the whole
/// literal (expressions.rst, "Set displays" — the same unpacking rule
/// "List displays" states). Read into the identical `Kind::List` shape
/// `list_literal_value` builds — this domain has no dedicated set kind
/// (`collection_models.rs`'s own module doc: a set's own element-
/// uniqueness is invisible to a reader that only ever consumes the
/// sequence via iteration/membership/`len()`), so a set literal and a
/// list literal share one representation, and every `set_method_result`
/// row above reads a `Kind::List` receiver either way.
fn evaluate_set(set: &ruff_python_ast::ExprSet, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(elements) = evaluate_display_elements(set.elts.iter(), environment, kernel) else {
        return unknown();
    };
    collection_models::list_literal_value(&elements)
}

/// Evaluates every element of a list/tuple display in order; `None` the
/// moment a `Starred` element appears (expressions.rst, "List displays"
/// — a starred element unpacks an iterable this file cannot model), so
/// the caller can decline the whole literal rather than mis-slot the
/// starred expression as one ordinary element.
fn evaluate_display_elements<'a>(
    elements: impl Iterator<Item = &'a Expr>,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::new();
    for element in elements {
        if matches!(element, Expr::Starred(_)) {
            return None;
        }
        values.push(evaluate_expression(element, environment, kernel));
    }
    Some(values)
}

/// `{k: v, ...}` — every key expression must be a plain string literal
/// (expressions.rst, "Dictionary displays": a non-literal key, a
/// computed key, or a `**spread` entry has no slot in this domain's
/// string-keyed `ObjectKey.name`); `collection_models::dict_literal_value`
/// itself declines the whole literal the moment any key position is not
/// a string literal, so this function's job is only to build the
/// `(Option<String>, value)` rows in source order.
fn evaluate_dict(dict: &ruff_python_ast::ExprDict, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let mut keys: Vec<Option<String>> = Vec::new();
    let mut values: Vec<AbstractValue> = Vec::new();
    for item in &dict.items {
        // a `**spread` entry parses with `key: None` — no string-literal
        // key exists at that position, so the row stays `None` and the
        // whole literal declines below
        let key_text = match &item.key {
            Some(Expr::StringLiteral(literal)) => Some(literal.value.to_str().to_owned()),
            _ => None,
        };
        keys.push(key_text);
        values.push(evaluate_expression(&item.value, environment, kernel));
    }
    collection_models::dict_literal_value(&keys, &values)
}

/// `container[index]` — expressions.rst, "Subscriptions." A `Slice`
/// index (`s[1:3]`) routes through `evaluate_string_slice` for a known
/// exact-string receiver (list slicing is not modeled this unit —
/// `list_slice` in c-reads-and-values.py stays a named remainder).
fn evaluate_subscript(subscript: &ruff_python_ast::ExprSubscript, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    if let Expr::Slice(slice) = subscript.slice.as_ref() {
        let container = evaluate_expression(&subscript.value, environment, kernel);
        return evaluate_string_slice(&container, slice, environment, kernel);
    }
    let container = evaluate_expression(&subscript.value, environment, kernel);
    let index = evaluate_expression(&subscript.slice, environment, kernel);
    match collection_models::subscript_read(&container, &index) {
        Some(value) => value,
        None => unknown(),
    }
}

/// `s[lower:upper]` on a known exact-string receiver, no `step`
/// (expressions.rst, "Slicings" — a slicing indexes via the same
/// `__getitem__` machinery as a plain subscript, with the slice's own
/// bounds silently CLAMPED to `[0, len(s)]` rather than raising — the
/// one place this domain's plain-index honesty ("out of range
/// declines") does not apply, because a slice never raises for an
/// out-of-range bound the way a single index does). Missing `lower`
/// defaults to 0, missing `upper` defaults to `len(s)` — the same
/// defaults `s[:]`/`s[n:]`/`s[:n]` read under. Negative bounds adjust
/// by the string's own length first, matching the plain-index rule
/// (`known_integer_index`'s own negative-adjustment). A `step` is not
/// modeled (declines outright, per the mission's own scope); a non-
/// string receiver or a non-Integer bound also declines.
fn evaluate_string_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if slice.step.is_some() {
        return unknown();
    }
    let Some(text) = exact_string_values(container) else {
        return unknown();
    };
    let length = text.len() as i64;
    let lower = match &slice.lower {
        Some(expr) => match slice_bound_index(expr, environment, kernel) {
            Some(value) => value,
            None => return unknown(),
        },
        None => 0,
    };
    let upper = match &slice.upper {
        Some(expr) => match slice_bound_index(expr, environment, kernel) {
            Some(value) => value,
            None => return unknown(),
        },
        None => length,
    };
    let clamped_lower = clamp_slice_bound(lower, length);
    let clamped_upper = clamp_slice_bound(upper, length);
    if clamped_lower >= clamped_upper {
        return string_models::string_literal_value("");
    }
    let slice_points = text[clamped_lower as usize..clamped_upper as usize].to_vec();
    known_values(slice_points, PrimitiveKind::String, TrustProved)
}

/// One slice bound's known Integer value, or `None` if it is not a
/// single known Integer-sorted expression — the same acceptance
/// `known_integer_index` (collection_models.rs) gives a plain
/// subscript index, evaluated here instead since a slice bound is an
/// EXPRESSION (`lower_bound: expression`, expressions.rst's own
/// grammar) rather than an already-evaluated AbstractValue.
fn slice_bound_index(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let value = evaluate_expression(expr, environment, kernel);
    let (number, sort) = single_numeric_value(&value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    Some(number as i64)
}

/// Adjusts a slice bound by the sequence's own length (negative bounds
/// count from the end, the same rule a plain index follows) and then
/// CLAMPS to `[0, length]` — a slice bound never raises for landing
/// outside that range, unlike a plain index (expressions.rst,
/// "Slicings"' own silent-clamp behavior).
fn clamp_slice_bound(bound: i64, length: i64) -> i64 {
    let adjusted = if bound < 0 { bound + length } else { bound };
    adjusted.clamp(0, length)
}

/// `x < y <= z` chains as `x < y and y <= z`, evaluating `y` once
/// (expressions.rst, "Comparisons": "x < y <= z is equivalent to x < y
/// and y <= z, except that y is evaluated only once"). Every adjacent
/// pair must decide `True` for the whole chain to decide `True`; the
/// moment one pair cannot be decided, the whole chain is unknown — a
/// chain never answers partial knowledge.
fn evaluate_compare(compare: &ruff_python_ast::ExprCompare, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let left = evaluate_expression(&compare.left, environment, kernel);
    let mut operands = Vec::with_capacity(compare.comparators.len());
    for comparator in compare.comparators.iter() {
        operands.push(evaluate_expression(comparator, environment, kernel));
    }
    let mut previous = &left;
    for (op, operand) in compare.ops.iter().zip(operands.iter()) {
        let Some(result) = compare_pair(*op, previous, operand) else {
            return unknown();
        };
        if result != 1.0 {
            return known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved);
        }
        previous = operand;
    }
    known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved)
}

/// One comparison operator over two already-evaluated operands: `1.0`
/// (True), `0.0` (False), or `None` (not decidable). Every row here
/// requires both operands KNOWN — an unknown operand always declines
/// the whole pair, which `evaluate_compare` turns into unknown() for
/// the whole chain.
fn compare_pair(op: CmpOp, left: &AbstractValue, right: &AbstractValue) -> Option<f64> {
    // `is` / `is not` decide identity against None only: expressions.rst,
    // "Comparisons," `is`/`is not` — "None" is the one CPython value this
    // file can prove identity for without a shared-object model. Either
    // side being the exactly-null state (`Kind::Null`) settles it: None
    // is None (True/False split by op), and a known non-None value is
    // never identical to None.
    if op == CmpOp::Is || op == CmpOp::IsNot {
        let identical = match (left.kind == Kind::Null, right.kind == Kind::Null) {
            (true, true) => true,
            (true, false) | (false, true) => {
                if right.kind == Kind::Unknown || left.kind == Kind::Unknown {
                    return None;
                }
                false
            }
            (false, false) => return None,
        };
        let result = if op == CmpOp::Is { identical } else { !identical };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // `in` / `not in`: a known needle against a known List container of
    // known elements — membership by `==` on each element's own value
    // (expressions.rst, "Comparisons," `in`/`not in`: "For container
    // types such as list, tuple, set, frozenset, dict, or collections.deque,
    // the expression `x in y` is equivalent to `any(x is e or x == e for e
    // in y)`" — this row reads the `==` half, since exact-value equality
    // already decides `is` for two equal known scalars).
    if op == CmpOp::In || op == CmpOp::NotIn {
        if right.kind != Kind::List {
            return None;
        }
        let mut found = false;
        for element in &right.items {
            match single_pair_equal(left, element) {
                Some(true) => {
                    found = true;
                    break;
                }
                Some(false) => continue,
                None => return None,
            }
        }
        let result = if op == CmpOp::In { found } else { !found };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // both single known numeric values: ==, !=, <, <=, >, >= over the
    // f64s directly (expressions.rst, "Comparisons," the numeric-types
    // ordering — CPython orders numbers by mathematical value)
    if let (Some((left_value, _)), Some((right_value, _))) =
        (single_numeric_value(left), single_numeric_value(right))
    {
        let result = match op {
            CmpOp::Eq => left_value == right_value,
            CmpOp::NotEq => left_value != right_value,
            CmpOp::Lt => left_value < right_value,
            CmpOp::LtE => left_value <= right_value,
            CmpOp::Gt => left_value > right_value,
            CmpOp::GtE => left_value >= right_value,
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => unreachable!("handled above"),
        };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // both known exact strings: == and != read the code-point vectors
    // directly; <, <=, >, >= read them lexicographically — CPython
    // orders strings "lexicographically using the numeric equivalents
    // (the result of the built-in function ord()) of their characters"
    // (expressions.rst, "Comparisons," the sequence-types ordering rule).
    if let (Some(left_text), Some(right_text)) = (exact_string_values(left), exact_string_values(right)) {
        let result = match op {
            CmpOp::Eq => left_text == right_text,
            CmpOp::NotEq => left_text != right_text,
            CmpOp::Lt => left_text < right_text,
            CmpOp::LtE => left_text <= right_text,
            CmpOp::Gt => left_text > right_text,
            CmpOp::GtE => left_text >= right_text,
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => unreachable!("handled above"),
        };
        return Some(if result { 1.0 } else { 0.0 });
    }
    None
}

/// Whether two already-evaluated values are `==`, for the `in`/`not in`
/// membership row: single known numerics compare by value, known exact
/// strings compare by their code-point sequence, and anything else (an
/// unknown side, or a shape this file has no equality row for) declines
/// with `None` rather than guessing.
fn single_pair_equal(left: &AbstractValue, right: &AbstractValue) -> Option<bool> {
    if let (Some((left_value, _)), Some((right_value, _))) =
        (single_numeric_value(left), single_numeric_value(right))
    {
        return Some(left_value == right_value);
    }
    if let (Some(left_text), Some(right_text)) = (exact_string_values(left), exact_string_values(right)) {
        return Some(left_text == right_text);
    }
    None
}

/// The code-point vector an AbstractValue carries, if it is a known
/// exact string (`Kind::Values` tagged `PrimitiveKind::String`) — the
/// same shape `string_models.rs` builds; comparing the `Vec<f64>`
/// directly (rather than converting to a Rust `String` first) IS the
/// code-point-by-code-point ordering `str`'s own comparison rule states.
fn exact_string_values(value: &AbstractValue) -> Option<&[f64]> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(&value.values)
}

/// `and`/`or` return an OPERAND, never a coerced bool (expressions.rst,
/// "Boolean operations": "the return value of a short-circuit operator
/// is the last evaluated argument"). Walked left to right: `and` stops
/// at the first definitely-falsy operand (that operand IS the answer)
/// and skips past a definitely-truthy one; `or` mirrors it. The moment
/// an operand's truthiness is not decidable, the whole expression
/// declines — a later operand might still have changed the answer, and
/// this file does not guess. The LAST operand is always evaluated and
/// returned once every earlier operand has been skipped (an `and` chain
/// of all-truthy operands, or an `or` chain of all-falsy ones), matching
/// the same short-circuit rule — a BoolOp always carries at least two
/// values (Python's grammar has no one-operand `and`/`or`), so the loop
/// below always reaches that last operand.
fn evaluate_boolop(boolop: &ruff_python_ast::ExprBoolOp, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let is_and = boolop.op == BoolOp::And;
    let last_index = boolop.values.len().saturating_sub(1);
    for (index, operand_expr) in boolop.values.iter().enumerate() {
        let operand = evaluate_expression(operand_expr, environment, kernel);
        if index == last_index {
            return operand;
        }
        let (value, known) = truthiness(&operand);
        if !known {
            return unknown();
        }
        // `and` stops on a falsy operand; `or` stops on a truthy one —
        // that operand is the short-circuited answer
        if is_and == !value {
            return operand;
        }
    }
    unknown()
}

/// `f"...{expr}..."` composes the literal text and each interpolation's
/// contribution, in source order, into one exact string
/// (expressions.rst, "Formatted string literals"). Only the plainest
/// interpolation shape is modeled: no conversion (`!s`/`!r`/`!a`) and no
/// format spec (`:...`) — either one changes the spelling in ways this
/// file does not compute exactly, so their presence declines the WHOLE
/// f-string rather than composing a partially-wrong string. An
/// interpolated expression that is a known exact string contributes its
/// text; a single known Integer-sorted value contributes its plain
/// integer spelling (`"42"`, no `.0`); a single known Float-sorted
/// value contributes CPython's own `str(float)` spelling via
/// `format_py_number(value, true)` — `str`/`repr` of a float are
/// identical in Python 3 (library/stdtypes.rst's float section states
/// no divergence between the two), and `format_py_number`'s own doc
/// already pins the exact row this needs ("fires spell 30.0 with its
/// .0"), verified against `f"{30.0}"` == `"30.0"` and `f"{3.5}"` ==
/// `"3.5"` by execution. Any other shape declines the whole f-string.
/// An implicitly concatenated f-string (`f"a" f"b"`) is not modeled —
/// only the single-part form (`as_single_part_fstring`) is read.
fn evaluate_fstring(fstring: &ruff_python_ast::ExprFString, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(single) = fstring.as_single_part_fstring() else {
        return unknown();
    };
    let mut composed = String::new();
    for element in &single.elements {
        match element {
            InterpolatedStringElement::Literal(literal) => composed.push_str(&literal.value),
            InterpolatedStringElement::Interpolation(interpolation) => {
                if interpolation.conversion != ConversionFlag::None || interpolation.format_spec.is_some() {
                    return unknown();
                }
                let value = evaluate_expression(&interpolation.expression, environment, kernel);
                if let Some(text) = exact_string_values(&value) {
                    let Some(text) = code_points_to_string(text) else {
                        return unknown();
                    };
                    composed.push_str(&text);
                } else if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(&value) {
                    composed.push_str(&format_integer_spelling(number));
                } else if let Some((number, PrimitiveKind::Float)) = single_numeric_value(&value) {
                    composed.push_str(&refined_sets::format_string_shapes::format_py_number(number, true));
                } else {
                    return unknown();
                }
            }
        }
    }
    string_models::string_literal_value(&composed)
}

/// The `Vec<f64>` code points `string_models.rs` builds, converted back
/// to a Rust `String` — the same conversion `string_models.rs`'s own
/// (private) `exact_string_text` performs; repeated here because this
/// file is out-of-crate from `string_models.rs`'s module (AGENT-BRIEF.md:
/// this wave touches only `expressions.rs`, so no visibility is widened
/// there for this one caller).
fn code_points_to_string(code_points: &[f64]) -> Option<String> {
    code_points
        .iter()
        .map(|point| char::from_u32(*point as i64 as u32))
        .collect()
}

/// A known Integer-sorted value's plain spelling: `"42"`, never `"42.0"`
/// — Python's f-string `str()` conversion of an int has no decimal
/// point (contrast `format_py_number`'s float spelling, which is a
/// different sort this row does not attempt).
fn format_integer_spelling(value: f64) -> String {
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
fn evaluate_ternary(ternary: &ruff_python_ast::ExprIf, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let test = evaluate_expression(&ternary.test, environment, kernel);
    let (value, known) = truthiness(&test);
    if known {
        return if value {
            evaluate_expression(&ternary.body, environment, kernel)
        } else {
            evaluate_expression(&ternary.orelse, environment, kernel)
        };
    }
    let body = evaluate_expression(&ternary.body, environment, kernel);
    let orelse = evaluate_expression(&ternary.orelse, environment, kernel);
    join_known(body, orelse)
}

/// A function/method call — dispatch order: (a) a bare, environment-
/// unbound name naming a SAME-MODULE `def` (`environment.functions()`)
/// summarizes through `summaries::call_result` — checked FIRST, so a
/// module-level `def` shadows a builtin of the same name, matching
/// CPython's own name resolution (a later `def len(...):` at module
/// scope really does shadow the builtin `len`); (b) a bare, unbound name
/// naming a same-module class (`environment.classes()`) is a
/// CONSTRUCTION call — judged through `instances::judge_construction`,
/// but this is a VALUE read: any fire the construction raises is
/// check.rs's own statement-sink business (a nested construction inside
/// a larger expression has no sink of its own here), so the verdict's
/// `fires` are discarded and only `instance` is returned; (c) a bare
/// unbound name calls a builtin (`len` gets its own row into
/// `collection_models::len_result`; everything else goes to
/// `builtin_models::builtin_call_result`); (d) `math.<name>(...)` where
/// `math` is not locally bound calls `math_models::math_call_result`;
/// (e) any other attribute call evaluates its receiver and dispatches by
/// the receiver's own known shape (an exact string's method, or a
/// dict's `.get`); (f) everything else — a lambda call, a bound-name
/// call, an unmodeled builtin, an unmodeled method — is unknown().
/// Keyword or starred arguments are not modeled for any modeled row
/// EXCEPT the function/construction paths, which map keywords to
/// parameter/field position themselves — every other cited
/// builtin/method signature this wave models takes positional arguments
/// only, so the global keyword/starred guard below applies to the
/// builtin/math/method paths.
fn evaluate_call(call: &ruff_python_ast::ExprCall, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    if let Expr::Name(name) = call.func.as_ref() {
        if environment.read(name.id.as_str()).is_none() {
            if let Some(table) = environment.functions() {
                if let Some(def) = table.def(name.id.as_str()) {
                    let Some(positional) = positional_arguments_for_def(call, def, environment, kernel) else {
                        return unknown();
                    };
                    return match summaries::call_result(def, &positional, environment.functions(), kernel, 0) {
                        Some(value) => value,
                        None => unknown(),
                    };
                }
            }
            if let Some(classes) = environment.classes() {
                if let Some(model) = classes.get(name.id.as_str()) {
                    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
                        return unknown();
                    }
                    let positional: Vec<(AbstractValue, TextRange)> = call
                        .arguments
                        .args
                        .iter()
                        .map(|arg| (evaluate_expression(arg, environment, kernel), arg.range()))
                        .collect();
                    let keyword: Vec<(String, AbstractValue, TextRange)> = call
                        .arguments
                        .keywords
                        .iter()
                        .filter_map(|kw| {
                            let arg_name = kw.arg.as_ref()?;
                            Some((
                                arg_name.as_str().to_owned(),
                                evaluate_expression(&kw.value, environment, kernel),
                                kw.value.range(),
                            ))
                        })
                        .collect();
                    // a construction is a VALUE here — the verdict's fires
                    // belong to whichever statement sink hosts this call
                    // expression, not to this nested value read
                    let verdict = instances::judge_construction(model, &positional, &keyword, kernel);
                    return verdict.instance;
                }
            }
        }
    }
    if !call.arguments.keywords.is_empty() {
        return unknown();
    }
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return unknown();
    }
    match call.func.as_ref() {
        Expr::Name(name) if environment.read(name.id.as_str()).is_none() => {
            let arguments: Vec<AbstractValue> = call
                .arguments
                .args
                .iter()
                .map(|arg| evaluate_expression(arg, environment, kernel))
                .collect();
            if name.id.as_str() == "len" {
                let [only] = arguments.as_slice() else { return unknown() };
                return match collection_models::len_result(only) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            match builtin_models::builtin_call_result(name.id.as_str(), &arguments) {
                Some(value) => value,
                None => unknown(),
            }
        }
        Expr::Attribute(attribute) => evaluate_attribute_call(attribute, call, environment, kernel),
        _ => unknown(),
    }
}

/// A same-module `def` call's positional argument values, in parameter
/// order: every positional call argument evaluated in place, then every
/// keyword argument mapped to its parameter's own position by NAME
/// (`summaries::call_result` itself takes only positional values, per
/// its own module doc — "Keyword arguments are the WIRING owner's job").
/// A keyword naming no parameter of `def`, or a starred positional
/// argument, declines the whole call — this file does not guess which
/// position a stray argument might occupy. Positions covered by BOTH a
/// positional and a keyword argument are impossible to build soundly
/// (CPython itself raises `TypeError: multiple values for argument` at
/// that call), so this function does not attempt to detect that
/// conflict — `bind_parameters`'s own arity check will decline once the
/// merged vector's length disagrees with what the call actually
/// supplied where relevant, and any un-caught double-binding is a
/// pre-existing gap this wave does not close.
fn positional_arguments_for_def(
    call: &ruff_python_ast::ExprCall,
    def: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
    let parameter_names: Vec<&str> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .map(|parameter| parameter.parameter.name.id.as_str())
        .collect();
    let mut positional: Vec<Option<AbstractValue>> = vec![None; parameter_names.len().max(call.arguments.args.len())];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        positional[index] = Some(evaluate_expression(arg, environment, kernel));
    }
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            // `**kwargs`-spread call argument: no single parameter name
            // to map it to
            return None;
        };
        let Some(position) = parameter_names.iter().position(|name| *name == arg_name.as_str()) else {
            return None;
        };
        positional[position] = Some(evaluate_expression(&keyword.value, environment, kernel));
    }
    // trailing None slots are parameters this call left for their own
    // default — bind_parameters reads those; only a HOLE before a filled
    // slot (a positional gap no keyword covered) is unbuildable
    while matches!(positional.last(), Some(None)) {
        positional.pop();
    }
    positional.into_iter().collect()
}

/// `receiver.attr(...)` — the known receiver shapes this file
/// dispatches: `math.<name>(...)` / `re.compile(...)` (only when the
/// module name is not shadowed by a local binding) and a method call
/// on an evaluated receiver (an exact string's method, a dict's `.get`
/// or a view method, or a set method).
fn evaluate_attribute_call(
    attribute: &ruff_python_ast::ExprAttribute,
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let arguments: Vec<AbstractValue> = call
        .arguments
        .args
        .iter()
        .map(|arg| evaluate_expression(arg, environment, kernel))
        .collect();
    if let Expr::Name(module_name) = attribute.value.as_ref() {
        if module_name.id.as_str() == "math" && environment.read("math").is_none() {
            return match math_models::math_call_result(attribute.attr.as_str(), &arguments) {
                Some(value) => value,
                None => unknown(),
            };
        }
        // `re.compile(pattern)` — library/re.html, `re.compile`: "Compile
        // a regular expression pattern... into a regular expression
        // object." This domain has no Pattern kind (no regex-engine
        // knowledge is tracked), so the result is answered opaque —
        // the honest "a compiled pattern" sort, never a specific value
        // (b-body-expressions.py's `literal_regex`).
        if module_name.id.as_str() == "re" && environment.read("re").is_none() {
            if attribute.attr.as_str() == "compile" {
                return opaque_value("a compiled pattern");
            }
            // `re.match(pattern, string)` — library/re.html: "Return a
            // corresponding match object" (or None on no match). This
            // file cannot decide WHICH of the two outcomes a real regex
            // engine would reach (no pattern-matching engine is
            // modeled), so it answers the match-object sort ONLY, never
            // the None-on-no-match alternative — an honest over-
            // approximation of "some value came back," matching the
            // fixture row's own sort-mismatch framing
            // (c-reads-and-values.py's `string_match`).
            if attribute.attr.as_str() == "match" {
                return opaque_value("a match object");
            }
        }
    }
    let receiver = evaluate_expression(&attribute.value, environment, kernel);
    if exact_string_values(&receiver).is_some() {
        return match string_models::string_method_result(attribute.attr.as_str(), &receiver, &arguments) {
            Some(value) => value,
            None => unknown(),
        };
    }
    if receiver.kind == Kind::Object {
        if attribute.attr.as_str() == "get" {
            // dict.get(key, default=None, /) — a missing default argument
            // is None (stdtypes.rst, dict's `method:: get`), matching
            // `dict_get_result`'s own `None` reading of an absent default
            let key = arguments.first();
            let default = arguments.get(1);
            if let Some(key) = key {
                return match collection_models::dict_get_result(&receiver, key, default) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            return unknown();
        }
        if arguments.is_empty() {
            if let Some(value) = dict_view_method_result(attribute.attr.as_str(), &receiver) {
                return value;
            }
        }
    }
    if receiver.kind == Kind::List {
        if let Some(value) = set_method_result(attribute.attr.as_str(), &receiver, &arguments) {
            return value;
        }
    }
    unknown()
}

/// `dict.keys()`/`.values()`/`.items()` (no arguments) on a known dict
/// (`Kind::Object`) — library/stdtypes.rst, dict's `method:: keys()`/
/// `method:: values()`/`method:: items()`: "Return a new view of the
/// dictionary's keys/values/items." A VIEW is read here as the flat
/// `Kind::List` of its own elements (this domain has no separate view
/// kind, matching the module's own "iteration values" scope) — `keys()`
/// answers the key strings, `values()` the value AbstractValues, and
/// `items()` a list of 2-element `(key, value)` pair-lists, in the
/// dict's own insertion order (`ObjectKey`'s ordered-Vec shape,
/// `abstract_value.rs`'s own doc: "iteration order is insertion
/// order"). `None` for any other method name — declined, not modeled.
fn dict_view_method_result(method: &str, receiver: &AbstractValue) -> Option<AbstractValue> {
    match method {
        "keys" => {
            let keys: Vec<AbstractValue> = receiver.keys.iter().map(|entry| string_models::string_literal_value(&entry.name)).collect();
            Some(collection_models::list_literal_value(&keys))
        }
        "values" => {
            let values: Vec<AbstractValue> = receiver.keys.iter().map(|entry| entry.value.clone()).collect();
            Some(collection_models::list_literal_value(&values))
        }
        "items" => {
            let pairs: Vec<AbstractValue> = receiver
                .keys
                .iter()
                .map(|entry| {
                    collection_models::list_literal_value(&[string_models::string_literal_value(&entry.name), entry.value.clone()])
                })
                .collect();
            Some(collection_models::list_literal_value(&pairs))
        }
        _ => None,
    }
}

/// `a.union(b)` / `a.intersection(b)` / `a.difference(b)` /
/// `a.symmetric_difference(b)` / `a.issubset(b)` / `a.issuperset(b)` on
/// a known set receiver (`Kind::List` — this domain's one sequence
/// shape, `collection_models.rs`'s own module doc: a set's own
/// element-uniqueness is invisible to a reader that only ever consumes
/// the sequence via iteration/membership) with a known set argument.
/// Every row is cited against library/stdtypes.rst's own set-methods
/// entries: `union(*others)` ("Return a new set with elements from the
/// set and all others"), `intersection(*others)` ("elements common to
/// the set and all others"), `difference(*others)` ("elements in the
/// set that are not in the others"), `symmetric_difference(other)`
/// ("elements in either the set or other but not both"),
/// `issubset(other)` ("Test whether every element in the set is in
/// *other*"), `issuperset(other)` ("Test whether every element in
/// *other* is in the set"). This file's one method dispatches ONLY the
/// TWO-set, one-`other`-argument form (`*others`'s variadic extra
/// arguments are not modeled). `None` for any other method, receiver,
/// or argument shape.
fn set_method_result(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [other] = arguments else { return None };
    if other.kind != Kind::List {
        return None;
    }
    match method {
        "union" => {
            let mut items = receiver.items.clone();
            for candidate in &other.items {
                if !set_contains(&items, candidate)? {
                    items.push(candidate.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "intersection" => {
            let mut items = Vec::new();
            for element in &receiver.items {
                if set_contains(&other.items, element)? {
                    items.push(element.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "difference" => {
            let mut items = Vec::new();
            for element in &receiver.items {
                if !set_contains(&other.items, element)? {
                    items.push(element.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "symmetric_difference" => {
            let mut items = Vec::new();
            for element in &receiver.items {
                if !set_contains(&other.items, element)? {
                    items.push(element.clone());
                }
            }
            for element in &other.items {
                if !set_contains(&receiver.items, element)? {
                    items.push(element.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "issubset" => {
            for element in &receiver.items {
                if !set_contains(&other.items, element)? {
                    return Some(boolean_answer(false));
                }
            }
            Some(boolean_answer(true))
        }
        "issuperset" => {
            for element in &other.items {
                if !set_contains(&receiver.items, element)? {
                    return Some(boolean_answer(false));
                }
            }
            Some(boolean_answer(true))
        }
        _ => None,
    }
}

/// Whether `needle` is a member of `items` by `==` — `single_pair_equal`
/// declines (`None`) the moment one comparison cannot be decided, and
/// this helper propagates that decline through `?` at every call site
/// above rather than silently reading an undecidable member as absent.
fn set_contains(items: &[AbstractValue], needle: &AbstractValue) -> Option<bool> {
    for element in items {
        match single_pair_equal(needle, element) {
            Some(true) => return Some(true),
            Some(false) => continue,
            None => return None,
        }
    }
    Some(false)
}

/// A Boolean AbstractValue — the same `known_values(vec![0.0/1.0],
/// PrimitiveKind::Boolean, TrustProved)` shape every other boolean
/// answer in this file builds (`compare_pair`'s own rows, `not`'s own
/// row).
fn boolean_answer(value: bool) -> AbstractValue {
    known_values(vec![if value { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved)
}

/// `receiver.attr` — a plain attribute READ, not a call. The receiver
/// evaluates first; a known Object (`Kind::Object`) reads the field
/// through `instances::field_read`, the same linear-scan-by-name
/// `judge_construction`'s own `field_read` doc describes. This one arm
/// covers BOTH an instance field read (`person.age`) and a cross-module
/// binding read (`helper.over_years`): `cross_module.rs` builds a module
/// object with the identical `known_object` shape a class instance
/// carries (this file's own module doc note; both are `Kind::Object`
/// with an ordered `ObjectKey` vec), so one dispatch arm serves both
/// without asking which one built the receiver. Any other receiver
/// shape (unknown, a scalar, a list) answers `unknown()` — there is no
/// attribute-read model for it here.
fn evaluate_attribute_read(
    attribute: &ruff_python_ast::ExprAttribute,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    // `__class__` is a universal attribute (datamodel.rst: "instance.__class__
    // is the object's class") — EVERY value has one, the host's own type
    // object, never a program-tracked value; answered opaque regardless
    // of whether the receiver itself is known, since the fact "this
    // reads a host type object" holds independent of the receiver's
    // OWN value (b-body-expressions.py's `wrapper_dunder_class`/
    // `NewTargetProbe` rows)
    if attribute.attr.as_str() == "__class__" {
        return opaque_value("the __class__ object");
    }
    let receiver = evaluate_expression(&attribute.value, environment, kernel);
    if receiver.kind != Kind::Object {
        return unknown();
    }
    match instances::field_read(&receiver, attribute.attr.as_str()) {
        Some(value) => value,
        None => unknown(),
    }
}

/// `[elt for target in iterable if cond ...]` / the same shape for a set
/// display and a generator expression — expressions.rst, "Displays for
/// lists, sets and dictionaries": "the comprehension consists of a
/// single expression, followed by at least one `for` clause." Modeled
/// ONLY the single-clause, bare-Name-target, known-List-iterable shape:
/// exactly one `Comprehension` (a second `for` clause, or an `async for`
/// — `is_async` — declines outright), the target a bare `Expr::Name`
/// (a tuple-unpacking target is not modeled), and the iterable a known
/// `Kind::List` of already-known elements. Each surviving element forks
/// the environment, binds the target, evaluates every `if` condition in
/// order (a `known&&false` truthiness drops the element; `known&&true`
/// keeps checking the rest; anything not fully known makes the WHOLE
/// comprehension unknown — a single undecidable filter means this file
/// cannot say which elements the real list would contain), then
/// evaluates `elt` on that fork. The collected elements build through
/// `collection_models::list_literal_value` for every one of
/// list/set/generator — a set's own element-uniqueness and a
/// generator's own lazy-iteration behavior are both invisible to a
/// caller that only ever consumes the sequence via `len()`/`sum()`/a
/// `for`-loop read, so this file states the shared List shape honestly
/// rather than inventing a `Kind::Set`/generator variant with no reader
/// that would ever tell the difference.
fn evaluate_list_or_set_comp(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let Some(elements) = comprehension_elements(element_expr, generators, environment, kernel) else {
        return unknown();
    };
    collection_models::list_literal_value(&elements)
}

/// `{key: value for target in iterable if cond ...}` — the same
/// single-clause/bare-Name-target/known-List-iterable restriction as
/// `evaluate_list_or_set_comp`, with the additional requirement that
/// `key` evaluates to a known exact String at every surviving element
/// (this domain's dict literal is string-keyed only,
/// `collection_models.rs`'s own documented restriction) — any element
/// whose key is not a known exact string makes the whole comprehension
/// unknown() rather than silently dropping that entry.
fn evaluate_dict_comp(
    comp: &ruff_python_ast::ExprDictComp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let Some(key_expr) = comp.key.as_deref() else {
        // a `**spread` entry inside a dict comprehension has no single
        // key expression to read
        return unknown();
    };
    let Some(rows) = comprehension_rows(key_expr, &comp.value, &comp.generators, environment, kernel) else {
        return unknown();
    };
    let mut keys: Vec<Option<String>> = Vec::with_capacity(rows.len());
    let mut values: Vec<AbstractValue> = Vec::with_capacity(rows.len());
    for (key, value) in rows {
        let Some(key_text) = exact_string_values(&key).and_then(code_points_to_string) else {
            return unknown();
        };
        keys.push(Some(key_text));
        values.push(value);
    }
    collection_models::dict_literal_value(&keys, &values)
}

/// The single-clause comprehension shape shared by every comprehension
/// form: exactly one `Comprehension` clause, synchronous, a bare-Name
/// target, over a known `Kind::List` iterable of already-known elements.
/// `None` for anything outside that shape (multiple clauses, `async
/// for`, a non-Name target, an unknown/non-List iterable) — the honest
/// decline every comprehension form shares before either evaluates its
/// own element/key expression. The target name and the `if` conditions
/// both borrow from `generators` itself (`'a`), so a caller walking the
/// returned elements still has the clause's own filter list in hand
/// with no second destructure of `generators`.
fn comprehension_target_and_elements<'a>(
    generators: &'a [ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(&'a str, &'a [Expr], Vec<AbstractValue>)> {
    let [clause] = generators else {
        return None;
    };
    if clause.is_async {
        return None;
    }
    let Expr::Name(target) = &clause.target else {
        return None;
    };
    let iterable = evaluate_expression(&clause.iter, environment, kernel);
    if iterable.kind != Kind::List {
        return None;
    }
    Some((target.id.as_str(), &clause.ifs, iterable.items))
}

/// The surviving elements of a list/set/generator comprehension: walks
/// `comprehension_target_and_elements`'s own element sequence, forking
/// the environment and binding the target for each one, filtering by
/// every `if` clause's truthiness, and evaluating `element_expr` on the
/// elements that survive every filter. `None` the moment the shape is
/// outside what this file models, OR an `if` clause's truthiness cannot
/// be decided for some element.
fn comprehension_elements(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let (target_name, conditions, source_elements) =
        comprehension_target_and_elements(generators, environment, kernel)?;
    let mut out = Vec::new();
    for element in source_elements {
        let mut fork = environment.fork();
        fork.bind(target_name, element);
        if !comprehension_conditions_hold(conditions, &fork, kernel)? {
            continue;
        }
        out.push(evaluate_expression(element_expr, &fork, kernel));
    }
    Some(out)
}

/// The surviving `(key, value)` pairs of a dict comprehension — the same
/// per-element fork/bind/filter walk `comprehension_elements` performs,
/// evaluating both `key_expr` and `value_expr` on each surviving fork.
fn comprehension_rows(
    key_expr: &Expr,
    value_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(AbstractValue, AbstractValue)>> {
    let (target_name, conditions, source_elements) =
        comprehension_target_and_elements(generators, environment, kernel)?;
    let mut out = Vec::new();
    for element in source_elements {
        let mut fork = environment.fork();
        fork.bind(target_name, element);
        if !comprehension_conditions_hold(conditions, &fork, kernel)? {
            continue;
        }
        let key = evaluate_expression(key_expr, &fork, kernel);
        let value = evaluate_expression(value_expr, &fork, kernel);
        out.push((key, value));
    }
    Some(out)
}

/// Every `if` condition of one comprehension clause, evaluated in order
/// against `environment` (the fork with this element's target already
/// bound): `Some(true)` when every condition is definitely truthy (the
/// element survives), `Some(false)` the moment one condition is
/// definitely falsy (the element is dropped, remaining conditions are
/// not evaluated — matching Python's own left-to-right short-circuit
/// evaluation order for chained comprehension `if`s), `None` the moment
/// one condition's truthiness cannot be decided at all.
fn comprehension_conditions_hold(
    conditions: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<bool> {
    for condition in conditions {
        let value = evaluate_expression(condition, environment, kernel);
        let (truthy, known) = truthiness(&value);
        if !known {
            return None;
        }
        if !truthy {
            return Some(false);
        }
    }
    Some(true)
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

/// `-x` / `+x` / `~x` / `not x`. `not` reads ANY operand's truthiness
/// (expressions.rst §6.6: "not x" — "yields True if x is false, False
/// otherwise"), so it is decided before the numeric-only guard below;
/// every other row needs a known single numeric operand (not known, or
/// known but not exactly one number, is unknown — a unary op over a set
/// or a multi-valued state states nothing exact). `-x`/`+x` preserve the
/// operand's own sort: `-3` is still `int` (expressions §6.6 — unary
/// arithmetic states no widening), and a Boolean operand (`bool` is an
/// `int` subclass) becomes an ordinary `Integer` result the same way
/// arithmetic on booleans always does. `~x` is the bitwise inversion of
/// an integer argument, `-(x+1)` (expressions.rst §6.6: "The unary ~
/// (invert) operator yields the bitwise inversion of its integer
/// argument. The bitwise inversion of x is defined as -(x+1)") — over a
/// Float operand CPython raises `TypeError`, so this file declines
/// rather than answer a value the real call never produces.
fn evaluate_unary(
    unary: &ruff_python_ast::ExprUnaryOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let operand = evaluate_expression(&unary.operand, environment, kernel);
    if unary.op == UnaryOp::Not {
        let (value, known) = truthiness(&operand);
        if !known {
            return unknown();
        }
        return known_values(vec![if value { 0.0 } else { 1.0 }], PrimitiveKind::Boolean, TrustProved);
    }
    let Some((value, sort)) = single_numeric_value(&operand) else {
        return unknown();
    };
    match unary.op {
        UnaryOp::USub => known_values(vec![-value], sort, TrustProved),
        UnaryOp::UAdd => known_values(vec![value], sort, TrustProved),
        UnaryOp::Invert => {
            if sort == PrimitiveKind::Integer {
                known_values(vec![-(value + 1.0)], PrimitiveKind::Integer, TrustProved)
            } else {
                unknown()
            }
        }
        UnaryOp::Not => unreachable!("handled above"),
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
    let arithmetic = binary_arithmetic_value(binop.op, &left, &right);
    if arithmetic.kind != Kind::Unknown {
        return arithmetic;
    }
    sequence_binop_value(binop.op, &left, &right)
}

/// String/list `+`/`*`, and the SET operator spelling of `|`/`&`/`-`/
/// `^` — stdtypes.rst, "Common Sequence Operations": `s + t` is "the
/// concatenation of s and t," and `s * n` (either operand order) is "n
/// shallow copies of s concatenated," with note 2 pinning "values of n
/// less than 0 are treated as 0." The set section states the operator
/// spellings directly beside their method names (`union(*others)`:
/// "set | other | ..."; `intersection`: "set & other & ..."; the
/// `difference`/`symmetric_difference` operator rows the same section
/// states as `-`/`^`) — `set_method_result` already carries every one
/// of those row's semantics, so `|`/`&`/`-`/`^` over two known
/// `Kind::List` operands (both operands, per this domain's shared
/// list/set representation — see `evaluate_set`'s own doc) call
/// through to it under the equivalent method name rather than
/// duplicate the four loops. Read only after `binary_arithmetic_value`
/// has already declined the pair — a numeric `+`/`*`/bitwise op never
/// falls through to here.
fn sequence_binop_value(op: Operator, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    match op {
        Operator::Add => {
            if let (Some(left_text), Some(right_text)) = (exact_string_values(left), exact_string_values(right)) {
                let mut joined = left_text.to_vec();
                joined.extend_from_slice(right_text);
                return known_values(joined, PrimitiveKind::String, TrustProved);
            }
            if left.kind == Kind::List && right.kind == Kind::List {
                let mut joined = left.items.clone();
                joined.extend(right.items.iter().cloned());
                return collection_models::list_literal_value(&joined);
            }
            unknown()
        }
        Operator::BitOr => set_operator_value("union", left, right),
        Operator::BitAnd => set_operator_value("intersection", left, right),
        Operator::Sub => set_operator_value("difference", left, right),
        Operator::BitXor => set_operator_value("symmetric_difference", left, right),
        Operator::Mult => {
            if let Some(result) = sequence_repetition(left, right) {
                return result;
            }
            if let Some(result) = sequence_repetition(right, left) {
                return result;
            }
            unknown()
        }
        _ => unknown(),
    }
}

/// The binary-operator spelling of a set method: both operands must be
/// known `Kind::List` (this domain's shared list/set shape) for the
/// operator to answer at all — a numeric or string operand pair never
/// reaches here (`binary_arithmetic_value` and the `Add`/`Mult` rows
/// above already own those), so this function exists only to route
/// `|`/`&`/`-`/`^` through the exact same `set_method_result` logic a
/// `.union(...)`/`.intersection(...)`/`.difference(...)`/
/// `.symmetric_difference(...)` method call already answers.
fn set_operator_value(method: &str, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    if left.kind != Kind::List || right.kind != Kind::List {
        return unknown();
    }
    match set_method_result(method, left, std::slice::from_ref(right)) {
        Some(value) => value,
        None => unknown(),
    }
}

/// `sequence * n` (one fixed operand order): `sequence` is a known exact
/// string or a known list, `n` is a single known Integer-sorted value.
/// A negative `n` answers the empty sequence (stdtypes.rst note 2); a
/// non-negative `n` repeats the sequence's own elements/code points that
/// many times. `None` when `sequence`/`n` are not this exact shape, so
/// the caller can try the other operand order.
fn sequence_repetition(sequence: &AbstractValue, count: &AbstractValue) -> Option<AbstractValue> {
    let (count_value, count_sort) = single_numeric_value(count)?;
    if count_sort != PrimitiveKind::Integer {
        return None;
    }
    let repeats = if count_value < 0.0 { 0 } else { count_value as usize };
    if let Some(text) = exact_string_values(sequence) {
        let mut repeated = Vec::with_capacity(text.len() * repeats);
        for _ in 0..repeats {
            repeated.extend_from_slice(text);
        }
        return Some(known_values(repeated, PrimitiveKind::String, TrustProved));
    }
    if sequence.kind == Kind::List {
        let mut repeated = Vec::with_capacity(sequence.items.len() * repeats);
        for _ in 0..repeats {
            repeated.extend(sequence.items.iter().cloned());
        }
        return Some(collection_models::list_literal_value(&repeated));
    }
    None
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

/// Whether `expression` (or a sub-expression it evaluates FIRST, in
/// CPython's own left-before-right, receiver-before-arguments order)
/// PROVABLY raises on every run, given every operand it needs is known.
/// `Some((range, message))` names the raising expression's own range and
/// a plain sentence in one voice: "this expression provably raises
/// <ExcType>: <plain detail>" — the same voice `bytes_models.rs`'s own
/// `BytesAnswer::Raises` messages already speak (this function is the
/// caller `check.rs` will route those messages through, unchanged).
/// Anything not provably raising, or any operand this file cannot read,
/// answers `None` — this function never guesses at a raise the way it
/// never guesses at a value.
///
/// Recognized rows, each cited in the function that decides it: zero-
/// divisor arithmetic (`/`, `//`, `%`), an out-of-range/absent
/// subscript on a known List/Object, a bytes-like read/write whose
/// `bytes_models` answer is `BytesAnswer::Raises`, `int(<unparseable
/// known string>)`, `<receiver>.index(<absent known needle>)` on a
/// known string or list receiver, and `math.sqrt(<known negative>)`.
pub fn provable_raise(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    match expression {
        Expr::BinOp(binop) => {
            if let Some(found) = provable_raise(&binop.left, environment, kernel) {
                return Some(found);
            }
            if let Some(found) = provable_raise(&binop.right, environment, kernel) {
                return Some(found);
            }
            binop_provable_raise(binop, environment, kernel)
        }
        Expr::Subscript(subscript) => {
            if let Some(found) = provable_raise(&subscript.value, environment, kernel) {
                return Some(found);
            }
            if let Some(found) = provable_raise(&subscript.slice, environment, kernel) {
                return Some(found);
            }
            subscript_provable_raise(subscript, environment, kernel)
        }
        Expr::Call(call) => {
            if let Some(found) = provable_raise(&call.func, environment, kernel) {
                return Some(found);
            }
            for arg in &call.arguments.args {
                if let Some(found) = provable_raise(arg, environment, kernel) {
                    return Some(found);
                }
            }
            call_provable_raise(call, environment, kernel)
        }
        Expr::Attribute(attribute) => provable_raise(&attribute.value, environment, kernel),
        Expr::UnaryOp(unary) => provable_raise(&unary.operand, environment, kernel),
        Expr::BoolOp(boolop) => boolop
            .values
            .iter()
            .find_map(|operand| provable_raise(operand, environment, kernel)),
        Expr::Compare(compare) => {
            if let Some(found) = provable_raise(&compare.left, environment, kernel) {
                return Some(found);
            }
            compare.comparators.iter().find_map(|comparator| provable_raise(comparator, environment, kernel))
        }
        _ => None,
    }
}

/// `x / 0`, `x // 0`, `x % 0` — a known ZERO divisor provably raises
/// `ZeroDivisionError: division by zero` (expressions.rst §6.7:
/// "raise[s] ZeroDivisionError" for `/`/`//`/`%` when the right operand
/// is zero). The evaluation path (`binary_arithmetic_value`) already
/// declines these to `unknown()` for the VALUE question; this is the
/// same zero-divisor check speaking the fact as a provable raise rather
/// than a silent decline — the value path is unchanged.
fn binop_provable_raise(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if !matches!(binop.op, Operator::Div | Operator::FloorDiv | Operator::Mod) {
        return None;
    }
    let right = evaluate_expression(&binop.right, environment, kernel);
    let (right_value, _) = single_numeric_value(&right)?;
    if right_value != 0.0 {
        return None;
    }
    Some((
        binop.range(),
        "this expression provably raises ZeroDivisionError: division by zero".to_owned(),
    ))
}

/// `container[index]` where `container` and `index` are both KNOWN and
/// the read is provably out of range/absent — the same distinction
/// `collection_models::subscript_read`'s own doc draws between "not
/// modeled" and "known container, known index, provably absent": a
/// `subscript_read` decline on an UNKNOWN container or index states
/// nothing about the real runtime behavior (this function must decline
/// too), while a decline on a KNOWN List with a KNOWN out-of-range
/// Integer index, or a KNOWN Object with a KNOWN string key absent from
/// its own `keys`, is exactly the shape CPython raises
/// `IndexError`/`KeyError` for (expressions.rst, "Subscriptions";
/// stdtypes.rst, dict's `d[key]` row).
///
/// A `Kind::List` receiver tries `bytes_models::bytes_index` FIRST: a
/// `bytes`/`bytearray`/`array.array` value is the identical `Kind::List`
/// shape an ordinary list literal builds (bytes_models.rs's own module
/// doc), and `bytes_index`'s negative-index-adjusted bounds check is the
/// same rule an ordinary list read follows — so its own `Raises` message
/// already speaks correctly for BOTH a bytes-like receiver and a plain
/// list receiver, and this file does not re-derive that bounds
/// arithmetic a second time. `known_container_index_absent` below is
/// reached only for the `Kind::Object` (dict `KeyError`) row, which
/// `bytes_models.rs` has no function for.
fn subscript_provable_raise(
    subscript: &ruff_python_ast::ExprSubscript,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if matches!(subscript.slice.as_ref(), Expr::Slice(_)) {
        // a slice never raises for an out-of-bounds bound (silently
        // clamped, expressions.rst) — nothing to prove here
        return None;
    }
    let container = evaluate_expression(&subscript.value, environment, kernel);
    let index = evaluate_expression(&subscript.slice, environment, kernel);
    if container.kind == Kind::List {
        if let Some(BytesAnswer::Raises(message)) = bytes_models::bytes_index(&container, &index) {
            return Some((subscript.range(), one_voice_raise_message(&message)));
        }
        return None;
    }
    known_container_index_absent(&container, &index).map(|detail| {
        (
            subscript.range(),
            format!("this expression provably raises {detail}"),
        )
    })
}

/// Normalizes a `bytes_models.rs`-voiced raise sentence ("this read/write
/// provably raises...") to `provable_raise`'s own one voice, "this
/// expression provably raises..." — the two files speak the same fact
/// (a provable runtime raise) but were built with slightly different
/// wording for their own subject ("read"/"write" vs. "expression"); this
/// function is the one seam where the two meet, so every message this
/// function hands back reads in exactly one voice regardless of which
/// sibling file decided the raise.
fn one_voice_raise_message(message: &str) -> String {
    match message.split_once("provably raises") {
        Some((_, rest)) => format!("this expression provably raises{rest}"),
        None => message.to_owned(),
    }
}

/// Whether a KNOWN Object `container` provably lacks a KNOWN string
/// `key` — the exact-value companion to
/// `collection_models::subscript_read`'s dict row, deciding the same
/// membership question directly against `container.keys` so a caller
/// can tell "provably absent" apart from "not modeled" (which
/// `subscript_read`'s bare `None` cannot do alone). The `Kind::List` row
/// is handled by `subscript_provable_raise` itself through
/// `bytes_models::bytes_index` (see that function's own doc) — this
/// function covers `Kind::Object` (dict `KeyError`) only. `Some(detail)`
/// names the ExcType and the missing key, in `provable_raise`'s own
/// voice fragment (the `ExcType: detail` half, joined by the caller);
/// `None` for an unknown container/key, or a key that IS present.
fn known_container_index_absent(container: &AbstractValue, index: &AbstractValue) -> Option<String> {
    if container.kind != Kind::Object {
        return None;
    }
    let key = exact_string_values(index).map(code_points_to_string)??;
    let present = container.keys.iter().any(|entry| entry.name == key);
    if present {
        None
    } else {
        Some(format!("KeyError: '{key}'"))
    }
}

/// A call expression's own provable raise, once its callee and every
/// argument have already been checked (by `provable_raise`'s own
/// pre-order walk): a bytes-like element read/write whose
/// `bytes_models` answer is `Raises`, `int(<a known string that does
/// not parse as an int>)`, or `<receiver>.index(<a known needle absent
/// from a known receiver>)`.
fn call_provable_raise(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if let Expr::Name(name) = call.func.as_ref() {
        // a `base=` keyword changes the parsing rules entirely (a
        // non-decimal radix admits letters as digits) — this row only
        // ever decides the base-10 default, so ANY keyword argument
        // (not just a `base=` one) declines rather than risk judging a
        // non-base-10 call by the base-10 grammar
        if name.id.as_str() == "int"
            && environment.read("int").is_none()
            && call.arguments.keywords.is_empty()
        {
            let [only] = &*call.arguments.args else {
                return None;
            };
            let value = evaluate_expression(only, environment, kernel);
            let text = exact_string_values(&value).and_then(code_points_to_string)?;
            // library/functions.rst's `int(string, base=10)` row: "the
            // string can be preceded by + or - (with no space in
            // between), have leading zeros, be surrounded by whitespace,
            // and have single underscores interspersed between digits."
            // The exact ValueError wording ("invalid literal for int()
            // with base 10: '...'") is pinned by library/unittest.rst's
            // own worked test example (`assertRaisesRegex(ValueError,
            // "invalid literal for.*XYZ'$", int, 'XYZ')`) rather than by
            // functions.rst's own int() entry directly — the vendored
            // tree does not restate the message inline there.
            if !is_valid_base_ten_int_string(&text) {
                return Some((
                    call.range(),
                    format!("this expression provably raises ValueError: invalid literal for int() with base 10: '{text}'"),
                ));
            }
            return None;
        }
    }
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        // `math.sqrt(<known negative>)` provably raises `ValueError` —
        // library/math.rst's own module-introduction note: "The current
        // implementation will raise ValueError for invalid operations
        // like sqrt(-1.0)..." (math_models.rs's own
        // `sqrt_argument_is_known_negative` reads the same operand this
        // row checks, so the value dispatch and the raise dispatch
        // agree on exactly which sqrt calls are negative).
        if attribute.attr.as_str() == "sqrt" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    let arguments: Vec<AbstractValue> =
                        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, kernel)).collect();
                    if math_models::sqrt_argument_is_known_negative(&arguments) {
                        return Some((
                            call.range(),
                            "this expression provably raises ValueError: math domain error".to_owned(),
                        ));
                    }
                }
            }
        }
        if attribute.attr.as_str() == "index" {
            let [needle_expr] = &*call.arguments.args else {
                return None;
            };
            let receiver = evaluate_expression(&attribute.value, environment, kernel);
            let needle = evaluate_expression(needle_expr, environment, kernel);
            // str.index/list.index RAISE on a miss (AGENT-BRIEF.md;
            // stdtypes.rst's Common Sequence Operations table, note (8):
            // "index raises ValueError when x is not found in s")
            if let (Some(receiver_text), Some(needle_text)) =
                (exact_string_values(&receiver), exact_string_values(&needle))
            {
                let receiver_text = code_points_to_string(receiver_text)?;
                let needle_text = code_points_to_string(needle_text)?;
                if !receiver_text.contains(&needle_text) {
                    return Some((
                        call.range(),
                        format!("this expression provably raises ValueError: '{needle_text}' is not in string"),
                    ));
                }
                return None;
            }
            if receiver.kind == Kind::List {
                let found = receiver.items.iter().any(|element| single_pair_equal(element, &needle) == Some(true));
                if !found {
                    return Some((
                        call.range(),
                        "this expression provably raises ValueError: value is not in list".to_owned(),
                    ));
                }
            }
        }
    }
    // a bytes-like element access (`data[i]`/`data[i] = v`) already
    // routes through `subscript_provable_raise`'s own container-shaped
    // check above for a READ; a WRITE has no expression-level call site
    // this function walks (an assignment target is a statement-level
    // concern, check.rs's own sink), so `bytes_models::bytes_index`'s
    // `Raises` answer is not reached from a bare call expression here —
    // noted rather than silently unhandled.
    None
}

/// Whether `text` parses as a base-10 `int(str)` argument
/// (library/functions.rst's `int(string, base=10)` row, quoted in
/// `call_provable_raise`'s own doc): optional surrounding whitespace,
/// an optional single leading `+`/`-` with no space before the digits,
/// then one or more ASCII decimal digits with single underscores
/// allowed BETWEEN digits only (never leading, trailing, or doubled —
/// "single underscores interspersed between digits"). An empty digit
/// run (after stripping the sign) is never valid — `int("+")`,
/// `int("-")`, and `int("")` all raise the same way an all-underscore
/// run does.
fn is_valid_base_ten_int_string(text: &str) -> bool {
    let trimmed = text.trim();
    let digits_and_underscores = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if digits_and_underscores.is_empty() {
        return false;
    }
    let chars: Vec<char> = digits_and_underscores.chars().collect();
    if chars.first() == Some(&'_') || chars.last() == Some(&'_') {
        return false;
    }
    let mut previous_was_underscore = false;
    let mut saw_any_digit = false;
    for &c in &chars {
        if c == '_' {
            if previous_was_underscore {
                return false;
            }
            previous_was_underscore = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return false;
        }
        saw_any_digit = true;
        previous_was_underscore = false;
    }
    saw_any_digit
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
        // `f` is an unbound name and not a modeled builtin — the call
        // dispatch declines rather than guessing at an unmodeled callee
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

    #[test]
    fn test_string_literal() {
        let Some(value) = eval("\"ab\"") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
        assert_eq!(value.values, vec!['a' as u32 as f64, 'b' as u32 as f64]);
    }

    #[test]
    fn test_list_tuple_literal_and_subscript_read() {
        let Some(list_value) = eval("[10, 20, 30]") else { return };
        assert_eq!(list_value.kind, Kind::List);
        assert_eq!(list_value.items.len(), 3);

        let Some(tuple_value) = eval("(1, 2)") else { return };
        assert_eq!(tuple_value.kind, Kind::List);
        assert_eq!(tuple_value.items.len(), 2);

        let Some(subscripted) = eval("[10, 20, 30][1]") else { return };
        assert_eq!(subscripted.values, vec![20.0]);
    }

    #[test]
    fn test_dict_literal_and_subscript_read() {
        let Some(value) = eval("{\"a\": 1, \"b\": 2}[\"b\"]") else { return };
        assert_eq!(value.values, vec![2.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// A subscript past the list's bounds declines: CPython raises
    /// `IndexError`, which this file has no channel for
    /// (collection_models.rs's own pinned decline).
    #[test]
    fn test_subscript_out_of_range_declines() {
        let Some(value) = eval("[1, 2][5]") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_chained_comparison_true() {
        let Some(value) = eval("1 < 2 <= 2") else { return };
        assert_eq!(value.values, vec![1.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
    }

    #[test]
    fn test_chained_comparison_false() {
        // 1 < 2 <= 2 is True, but 1 < 2 <= 1 is False (second pair fails)
        let Some(value) = eval("1 < 2 <= 1") else { return };
        assert_eq!(value.values, vec![0.0]);
    }

    #[test]
    fn test_string_comparison() {
        let Some(equal) = eval("\"ab\" == \"ab\"") else { return };
        assert_eq!(equal.values, vec![1.0]);

        let Some(less) = eval("\"ab\" < \"ac\"") else { return };
        assert_eq!(less.values, vec![1.0]);
    }

    #[test]
    fn test_is_none() {
        let Some(is_none) = eval("None is None") else { return };
        assert_eq!(is_none.values, vec![1.0]);

        let Some(value_is_none) = eval("1 is None") else { return };
        assert_eq!(value_is_none.values, vec![0.0]);
    }

    #[test]
    fn test_in_over_list_literal() {
        let Some(present) = eval("2 in [1, 2, 3]") else { return };
        assert_eq!(present.values, vec![1.0]);

        let Some(absent) = eval("5 in [1, 2, 3]") else { return };
        assert_eq!(absent.values, vec![0.0]);
    }

    /// `and`/`or` return an OPERAND, not a coerced bool — `0 and 5`
    /// answers `0` (the falsy left operand), `0 or 5` answers `5` (the
    /// first truthy operand reached).
    #[test]
    fn test_and_or_return_operands() {
        let Some(and_result) = eval("0 and 5") else { return };
        assert_eq!(and_result.values, vec![0.0]);
        assert_eq!(and_result.kind_tag, Some(PrimitiveKind::Integer));

        let Some(or_result) = eval("0 or 5") else { return };
        assert_eq!(or_result.values, vec![5.0]);
    }

    #[test]
    fn test_not_and_invert() {
        let Some(not_result) = eval("not 0") else { return };
        assert_eq!(not_result.values, vec![1.0]);
        assert_eq!(not_result.kind_tag, Some(PrimitiveKind::Boolean));

        // ~5 == -(5+1) == -6
        let Some(invert_result) = eval("~5") else { return };
        assert_eq!(invert_result.values, vec![-6.0]);
        assert_eq!(invert_result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_fstring_composition_int_and_str() {
        let Some(value) = eval("f\"n={1 + 1} s={'ab'}\"") else { return };
        let text: String = value
            .values
            .iter()
            .filter_map(|c| char::from_u32(*c as i64 as u32))
            .collect();
        assert_eq!(text, "n=2 s=ab");
    }

    /// A ternary whose test is not decidable joins both arms' values —
    /// the loosest sound answer once neither arm can be ruled out.
    #[test]
    fn test_ternary_both_arms_join() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("1 if flag else 2").expect("test source must parse");
        let expression = parsed.into_expr();
        let mut environment = empty_environment();
        environment.bind("flag", unknown());
        let value = evaluate_expression(&expression, &environment, &kernel);
        // an Integer 1 joined with an Integer 2 is not exactly-known —
        // the join is not equal to either arm alone
        assert_ne!(value, known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
        assert_ne!(value, known_values(vec![2.0], PrimitiveKind::Integer, TrustProved));
    }

    #[test]
    fn test_ternary_decided_test_answers_one_arm() {
        let Some(value) = eval("1 if True else 2") else { return };
        assert_eq!(value.values, vec![1.0]);
    }

    #[test]
    fn test_len_call() {
        let Some(value) = eval("len([1, 2, 3])") else { return };
        assert_eq!(value.values, vec![3.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_abs_call() {
        let Some(value) = eval("abs(-7)") else { return };
        assert_eq!(value.values, vec![7.0]);
    }

    /// round(40.5) == 40 — round-half-to-even, the AGENT-BRIEF
    /// row-inverting fact against a naive round-half-up reading.
    #[test]
    fn test_round_half_to_even() {
        let Some(value) = eval("round(40.5)") else { return };
        assert_eq!(value.values, vec![40.0]);
    }

    #[test]
    fn test_math_floor_call() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("math.floor(x)").expect("test source must parse");
        let expression = parsed.into_expr();
        let mut environment = empty_environment();
        environment.bind("x", known_values(vec![7.9], PrimitiveKind::Float, TrustProved));
        let value = evaluate_expression(&expression, &environment, &kernel);
        assert_eq!(value.values, vec![7.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_string_upper_method() {
        let Some(value) = eval("\"ab\".upper()") else { return };
        let text: String = value
            .values
            .iter()
            .filter_map(|c| char::from_u32(*c as i64 as u32))
            .collect();
        assert_eq!(text, "AB");
    }

    #[test]
    fn test_string_repetition() {
        let Some(value) = eval("\"ab\" * 3") else { return };
        let text: String = value
            .values
            .iter()
            .filter_map(|c| char::from_u32(*c as i64 as u32))
            .collect();
        assert_eq!(text, "ababab");
    }

    #[test]
    fn test_list_concatenation() {
        let Some(value) = eval("[1, 2] + [3, 4]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items.len(), 4);
    }

    // --- item 1: same-module def calls ---

    /// A bare unbound name naming a same-module `def` summarizes through
    /// `summaries::call_result`, before the builtin path — `double(3)`
    /// answers 6 via the module's own function table, not a builtin.
    #[test]
    fn test_same_module_function_call() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def double(x):\n    return x + x\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("double(3)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![6.0]);
    }

    /// A module-level `def` named `len` shadows the builtin `len` —
    /// dispatch checks `environment.functions()` before the builtin path.
    #[test]
    fn test_same_module_def_shadows_a_builtin_of_the_same_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def len(x):\n    return 999\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("len([1, 2, 3])").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        // the shadowing def always answers 999, never the real length 3
        assert_eq!(value.values, vec![999.0]);
    }

    // --- item 2: construction is a value, not a statement-level fire ---

    /// A same-module class construction call evaluates to its instance
    /// value; any fire the construction would raise is discarded here
    /// (statement-level fires are check.rs's own business).
    #[test]
    fn test_same_module_construction_is_a_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "class Person:\n",
            "    def __init__(self, age):\n",
            "        self.age = age\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let aliases = std::collections::HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let classes = std::sync::Arc::new(crate::refinedpy::instances::class_table(
            &module, &aliases, &imports, &kernel,
        ));
        let mut environment = empty_environment();
        environment.set_classes(classes);
        let parsed = parse_expression("Person(40)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(
            crate::refinedpy::instances::field_read(&value, "age"),
            Some(known_values(vec![40.0], PrimitiveKind::Integer, TrustProved))
        );
    }

    // --- item 3: attribute read ---

    #[test]
    fn test_attribute_read_on_a_known_object() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "class Person:\n",
            "    def __init__(self, age):\n",
            "        self.age = age\n",
            "p = Person(40)\n",
            "value = p.age\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let aliases = std::collections::HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let classes = std::sync::Arc::new(crate::refinedpy::instances::class_table(
            &module, &aliases, &imports, &kernel,
        ));
        let mut environment = empty_environment();
        environment.set_classes(classes);
        let constructed = parse_expression("Person(40)").expect("test source must parse");
        let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
        environment.bind("p", instance);
        let read = parse_expression("p.age").expect("test source must parse");
        let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
        assert_eq!(value, known_values(vec![40.0], PrimitiveKind::Integer, TrustProved));
    }

    // --- item 4: __name__ ---

    #[test]
    fn test_dunder_name_is_a_sort_only_string() {
        let Some(value) = eval("__name__") else { return };
        assert_eq!(value.kind, Kind::Set);
    }

    #[test]
    fn test_dunder_name_shadowed_by_a_local_binding_reads_the_binding() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("__name__").expect("test source must parse");
        let mut environment = empty_environment();
        environment.bind("__name__", known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![1.0]);
    }

    // --- item 5: bytes literal ---

    #[test]
    fn test_bytes_literal() {
        let Some(value) = eval("b\"ab\"") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![97.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![98.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    #[test]
    fn test_bytes_index_reads_an_int() {
        // b"ab"[0] is the int 97 — AGENT-BRIEF.md's own pinned fact
        let Some(value) = eval("b\"ab\"[0]") else { return };
        assert_eq!(value.values, vec![97.0]);
    }

    // --- item 6: comprehensions ---

    #[test]
    fn test_list_comp_over_known_list() {
        let Some(value) = eval("[x for x in [1, 2, 3]]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    #[test]
    fn test_list_comp_with_a_condition_filters_elements() {
        let Some(value) = eval("[x for x in [1, 2, 3, 4] if x > 2]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![4.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    #[test]
    fn test_set_comp_and_generator_share_the_list_shape() {
        let Some(set_value) = eval("{x for x in [1, 2]}") else { return };
        assert_eq!(set_value.kind, Kind::List);
        let Some(gen_value) = eval("(x for x in [1, 2])") else { return };
        assert_eq!(gen_value.kind, Kind::List);
    }

    #[test]
    fn test_dict_comp_over_known_list_with_string_keys() {
        let Some(value) = eval("{str(x): x for x in [1]}") else { return };
        // str(x) is not a modeled builtin call in this domain, so the key
        // expression declines and the whole comprehension is unknown —
        // this pins the "unknown key declines the whole thing" rule
        // rather than asserting a specific dict shape
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_multiple_generator_clauses_decline() {
        let Some(value) = eval("[x for x in [1, 2] for y in [3, 4]]") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    // --- item 7: await ---

    #[test]
    fn test_await_evaluates_the_inner_expression() {
        let Some(value) = eval("await x") else { return };
        // `x` is unbound in the empty test environment, so the await of
        // it is unknown — this pins that await passes THROUGH to the
        // inner expression's own value rather than always answering
        // unknown regardless of the inner expression
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_await_of_a_known_value_passes_it_through() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("await x").expect("test source must parse");
        let mut environment = empty_environment();
        environment.bind("x", known_values(vec![7.0], PrimitiveKind::Integer, TrustProved));
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![7.0]);
    }

    // --- item 8: provable_raise ---

    fn provable_raise_of(source: &str) -> Option<(TextRange, String)> {
        let kernel = loaded_kernel()?;
        let parsed = parse_expression(source).expect("test source must parse");
        let environment = empty_environment();
        provable_raise(&parsed.into_expr(), &environment, &kernel)
    }

    #[test]
    fn test_provable_raise_zero_division() {
        let Some(found) = provable_raise_of("1 / 0") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("1 / 0 must provably raise");
        };
        assert!(found.1.contains("ZeroDivisionError"), "{}", found.1);
        assert!(found.1.contains("division by zero"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_zero_floor_division_and_modulo() {
        let Some(found) = provable_raise_of("1 // 0") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("1 // 0 must provably raise");
        };
        assert!(found.1.contains("ZeroDivisionError"), "{}", found.1);

        let Some(found) = provable_raise_of("1 % 0") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("1 % 0 must provably raise");
        };
        assert!(found.1.contains("ZeroDivisionError"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_out_of_range_subscript() {
        let Some(found) = provable_raise_of("[1, 2][5]") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("[1, 2][5] must provably raise");
        };
        assert!(found.1.contains("IndexError"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_absent_dict_key() {
        let Some(found) = provable_raise_of("{\"a\": 1}[\"missing\"]") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("a missing dict key must provably raise");
        };
        assert!(found.1.contains("KeyError"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_int_of_unparseable_string() {
        let Some(found) = provable_raise_of("int(\"xyz\")") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("int(\"xyz\") must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{}", found.1);
        assert!(found.1.contains("invalid literal"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_int_of_valid_string_declines() {
        assert!(provable_raise_of("int(\"123\")").is_none());
        // the underscore-digit-separator row (functions.rst) must NOT
        // false-positive raise
        assert!(provable_raise_of("int(\"1_000\")").is_none());
    }

    #[test]
    fn test_provable_raise_string_index_miss() {
        let Some(found) = provable_raise_of("\"banana\".index(\"z\")") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("a missing needle's .index() must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_list_index_miss() {
        let Some(found) = provable_raise_of("[1, 2, 3].index(9)") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("a missing element's .index() must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_bytes_out_of_range_index() {
        let Some(found) = provable_raise_of("b\"ab\"[10]") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("an out-of-range bytes index must provably raise");
        };
        assert!(found.1.contains("IndexError"), "{}", found.1);
        // the message speaks in provable_raise's own voice, not
        // bytes_models.rs's "this read provably raises" wording
        assert!(found.1.starts_with("this expression provably raises"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_none_case() {
        assert!(provable_raise_of("1 + 2").is_none());
        assert!(provable_raise_of("[1, 2][0]").is_none());
        assert!(provable_raise_of("1 / 2").is_none());
    }
}
