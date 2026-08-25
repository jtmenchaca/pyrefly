
use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::UnaryOp;

use crate::collection_models;
use crate::env::Environment;

use super::evaluate_expression;
use super::arithmetic::*;
use super::subscript::*;

/// (`[*xs, a]`) unpacks an iterable's contents into the literal at parse
/// time (expressions.rst, "List displays") — modeled ONLY when the
/// starred expression is a known `Kind::List` (this domain's shared
/// list/tuple/set shape), whose own elements splice into the display in
/// place, in order; any other starred-expression shape (unknown, a
/// non-List value) declines the WHOLE literal rather than mis-slot the
/// starred expression as one ordinary element.
pub(super) fn evaluate_list(list: &ruff_python_ast::ExprList, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(elements) = evaluate_display_elements(list.elts.iter(), environment, kernel) else {
        return unknown();
    };
    collection_models::list_literal_value(&elements)
}

/// `(a, b, c)` — the same element-evaluation and starred-element decline
/// as `evaluate_list`; `collection_models::tuple_literal_value` is the
/// one call that differs (both build the same `Kind::List` shape, per
/// that file's own doc).
pub(super) fn evaluate_tuple(tuple: &ruff_python_ast::ExprTuple, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
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
pub(super) fn evaluate_set(set: &ruff_python_ast::ExprSet, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(elements) = evaluate_display_elements(set.elts.iter(), environment, kernel) else {
        return unknown();
    };
    collection_models::list_literal_value(&elements)
}

/// Evaluates every element of a list/tuple display in order, splicing a
/// `Starred` element's own List elements in place (expressions.rst,
/// "List displays" — see `evaluate_list`'s own doc). `None` the moment
/// a starred expression evaluates to anything but a known `Kind::List`
/// — the caller declines the whole literal rather than mis-slot it.
pub(super) fn evaluate_display_elements<'a>(
    elements: impl Iterator<Item = &'a Expr>,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::new();
    for element in elements {
        if let Expr::Starred(starred) = element {
            let spread = evaluate_expression(&starred.value, environment, kernel);
            if spread.kind != Kind::List {
                return None;
            }
            values.extend(spread.items);
            continue;
        }
        values.push(evaluate_expression(element, environment, kernel));
    }
    Some(values)
}

/// `{k: v, ..., **spread}` — every ordinary key expression must be a
/// plain string literal OR evaluate to a known single Integer-sorted
/// value (expressions.rst, "Dictionary displays": any other key shape
/// — a computed key this file cannot reduce to one of those two
/// sorts, a float/bool/tuple key — has no slot in this domain's
/// `ObjectKey.name`/`numeric` pair, `collection_models.rs`'s own
/// module doc); a `**spread` entry (parses with `key: None`, `value`
/// the spread expression) splices a known `Kind::Object`'s own entries
/// in place, in source order — "if a key occurs more than once in the
/// same dictionary display... the last value... becomes the
/// corresponding value" (the SAME doc, its own duplicate-key-in-one-
/// display rule; `dict_literal_value`'s own last-value-wins overwrite
/// already gives a LATER row priority over an earlier one, so
/// spreading a `**spread` entry's rows into `keys`/`values` in source
/// order and letting that shared overwrite rule run is exactly "later
/// keys win," matching `{**a, "k": v}` and `{**a, **b}` alike). A
/// spread expression that is not a known `Kind::Object` declines the
/// WHOLE literal, the same honesty a non-string/non-int ordinary key
/// already carries.
pub(super) fn evaluate_dict(dict: &ruff_python_ast::ExprDict, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let mut keys: Vec<Option<collection_models::DictKey>> = Vec::new();
    let mut values: Vec<AbstractValue> = Vec::new();
    for item in &dict.items {
        match &item.key {
            Some(Expr::StringLiteral(literal)) => {
                keys.push(Some(collection_models::DictKey::string(literal.value.to_str())));
                values.push(evaluate_expression(&item.value, environment, kernel));
            }
            Some(key_expr) => {
                // a non-string-LITERAL key: a known single Integer-sorted
                // VALUE (an int literal, or any expression that reduces to
                // one, e.g. `{age + 1: v}`) still has a slot, a known
                // EXACT STRING value (never a string LITERAL — that shape
                // is the `Expr::StringLiteral` arm above) also has a slot
                // — a COMPUTED key that happens to evaluate to a string —
                // a bare Name bound to a string
                // (h-object-literal-members.py's own `computed_key_other_
                // expression`: `key = "age"`, `{key: 40}`), or any other
                // expression this file can reduce to a known string — is
                // the identical string-keyed dict entry a literal `{"age":
                // 40}` would build. A recognized IDENTITY value (a module-
                // level `object()` sentinel read back by name,
                // `builtin_models::object_call`'s own `source: "object()"`
                // tag) also has a slot, matched by provenance rather than
                // value. `collection_models::known_dict_key` is the SAME
                // reader `subscript_read` uses on the read side, so a
                // dict literal's keys are recognized identically whether
                // this is the build or the later `d[key]` lookup.
                let key_value = evaluate_expression(key_expr, environment, kernel);
                match collection_models::known_dict_key(&key_value) {
                    Some(dict_key) => keys.push(Some(dict_key)),
                    None => {
                        // no slot — decline the whole literal via a
                        // None row, matching dict_literal_value's own
                        // all-keys-must-be-Some check
                        keys.push(None);
                    }
                }
                values.push(evaluate_expression(&item.value, environment, kernel));
            }
            None => {
                let spread = evaluate_expression(&item.value, environment, kernel);
                if spread.kind != Kind::Object {
                    return unknown();
                }
                for entry in spread.keys {
                    let dict_key = if entry.numeric {
                        // every numeric ObjectKey this codebase builds
                        // carries a valid decimal spelling (DictKey::
                        // integer's own doc) — a parse failure here means
                        // the spread's own source built one some other
                        // way, which this file does not trust to splice
                        let Ok(value) = entry.name.parse() else {
                            return unknown();
                        };
                        collection_models::DictKey::integer(value)
                    } else {
                        collection_models::DictKey::string(&entry.name)
                    };
                    keys.push(Some(dict_key));
                    values.push(entry.value);
                }
            }
        }
    }
    collection_models::dict_literal_value(&keys, &values)
}

/// `container[index]` — expressions.rst, "Subscriptions." A `Slice`
/// index (`s[1:3]`) routes through `evaluate_slice` for a known
/// exact-string OR known list/tuple receiver.
pub(super) fn evaluate_subscript(subscript: &ruff_python_ast::ExprSubscript, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    if let Expr::Slice(slice) = subscript.slice.as_ref() {
        let container = evaluate_expression(&subscript.value, environment, kernel);
        return evaluate_slice(&container, slice, environment, kernel);
    }
    let container = evaluate_expression(&subscript.value, environment, kernel);
    let index = evaluate_expression(&subscript.slice, environment, kernel);
    match collection_models::subscript_read(&container, &index) {
        Some(value) => value,
        None => unknown(),
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
pub(super) fn evaluate_unary(
    unary: &ruff_python_ast::ExprUnaryOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let operand = evaluate_expression(&unary.operand, environment, kernel);
    if unary.op == UnaryOp::Not {
        let (value, known) = truthiness(&operand);
        if !known {
            // The operand's truthiness is undecided, so WHICH of the two
            // values `not` yields is undecided — but that it yields one of
            // them is stated outright: expressions.rst, "Boolean
            // operations" — "The operator ``not`` yields ``True`` if its
            // argument is false, ``False`` otherwise." The answer is the
            // exact two-member boolean domain, never `unknown()`.
            return known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec);
        }
        return known_values(vec![if value { 0.0 } else { 1.0 }], PrimitiveKind::Boolean, TrustProved);
    }
    let Some((value, sort)) = single_numeric_value(&operand) else {
        return negate_over_set(unary.op, &operand, kernel).unwrap_or_else(unknown);
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
