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

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::known_object;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
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
use crate::refinedpy::env;
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
        // `lambda: ...` read as a VALUE (bound to a name, returned, or
        // otherwise used directly rather than called) — expressions.rst,
        // "Lambdas": "The expression `lambda parameters: expression`
        // yields a function object." The unnamed object behaves like an
        // ordinary `def`-built function object (datamodel.rst,
        // "User-defined functions"). This domain tracks no
        // function-value Kind (a
        // function is never itself a refined scalar/collection), so the
        // honest answer is opaque — "a function value," never a
        // specific scalar (b-body-expressions.py's
        // `function_stored_as_local`).
        //
        // RETAINED CALLABLE: when `register_retained_callables` has
        // already recorded this exact lambda's own body into
        // `environment` (its statement-level caller runs that scan
        // before reaching this evaluation — `check.rs::sink_value`,
        // `summaries::interpret_body`'s `Stmt::Return` arm), the value
        // additionally encodes the table key on `source`
        // (`env::retained_callable_value`) so a later call through
        // `evaluate_call`'s retained-callable arm can interpret the
        // body instead of declining. A lambda `register_retained_
        // callables` never reached (a shape outside its own recursion,
        // or an environment with no such registration step at all —
        // every existing test environment, unaffected) still answers
        // the plain opaque value exactly as before this table existed.
        Expr::Lambda(lambda) => match environment.retained_callable(lambda.range().start().to_u32()) {
            Some(_) => env::retained_callable_value(lambda.range().start().to_u32()),
            None => opaque_value("a function value"),
        },
        _ => unknown(),
    }
}

/// Walks `expr`'s own subtree for every `Expr::Lambda` reachable
/// WITHOUT crossing a statement boundary (a call's own function/
/// arguments/keywords, an attribute's own receiver, a lambda's own
/// body — the shapes this corpus's five retained-callable rows
/// actually nest a lambda inside: a call argument, a constructor
/// argument), and records each one into `environment` with an EMPTY
/// closure — every lambda literal this scan reaches reads no free
/// name outside its own parameters (`e-class-and-function.py`'s
/// `pick(lambda s: s.age)`, `b-body-expressions.py`'s
/// `Person(lambda: 40)`), so there is nothing to seed. A `Stmt::Return`
/// whose value is a BARE lambda (`return lambda age: age + step`,
/// `make_adder`'s own row) is also reached here — a bare lambda is
/// still an `Expr`, so this scan's own top-level match arm covers it
/// with no separate case.
///
/// Called at the few STATEMENT-level points that hold `&mut
/// Environment` just before the expression evaluates
/// (`check.rs::sink_value`, `summaries::interpret_body`'s `Stmt::Return`
/// arm) — `evaluate_expression` itself only ever reads `&Environment`,
/// so a lambda nested inside a call/constructor argument has no other
/// place to register before `evaluate_call`'s own argument evaluation
/// reads it. Every other expression shape (a `BinOp`, a display, a
/// comprehension, …) is not walked into — a lambda nested THERE is
/// outside this wave's five rows and stays the plain opaque value,
/// never a wrong answer, only a lambda this table does not yet retain.
pub fn register_retained_callables(expr: &Expr, environment: &mut Environment) {
    match expr {
        Expr::Lambda(lambda) => {
            register_retained_callables(lambda.body.as_ref(), environment);
            let key = lambda.range().start().to_u32();
            environment.record_retained_callable(key, env::RetainedCallable::from_lambda(lambda, HashMap::new()));
        }
        Expr::Call(call) => {
            register_retained_callables(call.func.as_ref(), environment);
            for argument in &call.arguments.args {
                register_retained_callables(argument, environment);
            }
            for keyword in &call.arguments.keywords {
                register_retained_callables(&keyword.value, environment);
            }
        }
        Expr::Attribute(attribute) => {
            register_retained_callables(attribute.value.as_ref(), environment);
        }
        _ => {}
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
/// time (expressions.rst, "List displays") — modeled ONLY when the
/// starred expression is a known `Kind::List` (this domain's shared
/// list/tuple/set shape), whose own elements splice into the display in
/// place, in order; any other starred-expression shape (unknown, a
/// non-List value) declines the WHOLE literal rather than mis-slot the
/// starred expression as one ordinary element.
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

/// Evaluates every element of a list/tuple display in order, splicing a
/// `Starred` element's own List elements in place (expressions.rst,
/// "List displays" — see `evaluate_list`'s own doc). `None` the moment
/// a starred expression evaluates to anything but a known `Kind::List`
/// — the caller declines the whole literal rather than mis-slot it.
fn evaluate_display_elements<'a>(
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
fn evaluate_dict(dict: &ruff_python_ast::ExprDict, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
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
                // one, e.g. `{age + 1: v}`) still has a slot — read the
                // same way `evaluate_dict_comp`'s own key row does. A
                // known EXACT STRING value (never a string LITERAL — that
                // shape is the `Expr::StringLiteral` arm above) also has a
                // slot: a COMPUTED key that happens to evaluate to a
                // string — a bare Name bound to a string
                // (h-object-literal-members.py's own `computed_key_other_
                // expression`: `key = "age"`, `{key: 40}`), or any other
                // expression this file can reduce to a known string — is
                // the identical string-keyed dict entry a literal `{"age":
                // 40}` would build; `collection_models::DictKey::string`
                // takes the same plain text either way.
                let key_value = evaluate_expression(key_expr, environment, kernel);
                match single_numeric_value(&key_value) {
                    Some((number, PrimitiveKind::Integer)) => {
                        keys.push(Some(collection_models::DictKey::integer(number as i64)));
                    }
                    _ => match exact_string_values(&key_value).and_then(code_points_to_string) {
                        Some(text) => keys.push(Some(collection_models::DictKey::string(&text))),
                        None => {
                            // no slot — decline the whole literal via a
                            // None row, matching dict_literal_value's own
                            // all-keys-must-be-Some check
                            keys.push(None);
                        }
                    },
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
fn evaluate_subscript(subscript: &ruff_python_ast::ExprSubscript, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
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

/// `s[lower:upper]` / `xs[lower:upper]`, no `step` (expressions.rst,
/// "Slicings" — a slicing indexes via the same `__getitem__` machinery
/// as a plain subscript, with the slice's own bounds silently CLAMPED to
/// `[0, len(s)]` rather than raising — the one place this domain's
/// plain-index honesty ("out of range declines") does not apply, because
/// a slice never raises for an out-of-range bound the way a single index
/// does). Missing `lower` defaults to 0, missing `upper` defaults to
/// `len(s)` — the same defaults `s[:]`/`s[n:]`/`s[:n]` read under.
/// Negative bounds adjust by the sequence's own length first, matching
/// the plain-index rule (`known_integer_index`'s own negative-
/// adjustment). A `step` is not modeled (declines outright, per the
/// mission's own scope). Two known receiver shapes: a known exact
/// string (`Kind::Values` tagged `PrimitiveKind::String`) answers a
/// SLICED STRING (`c-reads-and-values.py`'s own string-slicing rows);
/// a known list/tuple (`Kind::List` — this domain's shared sequence
/// shape, `collection_models.rs`'s own module doc: list slicing shares
/// the identical clamp-not-raise rule "Slicings" states for every
/// built-in sequence, so this is the SAME bound-computation this
/// function already carries, only reading `items` instead of `values`)
/// answers a SLICED LIST (`c-reads-and-values.py`'s `list_slice`:
/// `overs[0:1][0]`, a slice immediately re-subscripted). Any other
/// receiver shape, or a non-Integer bound, declines.
fn evaluate_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if slice.step.is_some() {
        return unknown();
    }
    let length = match container.kind {
        Kind::Values if container.kind_tag == Some(PrimitiveKind::String) => container.values.len() as i64,
        Kind::List => container.items.len() as i64,
        _ => return unknown(),
    };
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
    match container.kind {
        Kind::Values => {
            if clamped_lower >= clamped_upper {
                return string_models::string_literal_value("");
            }
            let slice_points = container.values[clamped_lower as usize..clamped_upper as usize].to_vec();
            known_values(slice_points, PrimitiveKind::String, TrustProved)
        }
        Kind::List => {
            if clamped_lower >= clamped_upper {
                return collection_models::list_literal_value(&[]);
            }
            let slice_items = container.items[clamped_lower as usize..clamped_upper as usize].to_vec();
            collection_models::list_literal_value(&slice_items)
        }
        _ => unreachable!("container.kind checked above in the length match"),
    }
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
            // a SINGLE-PAIR `in`/`not in` chain (`x in y`, never a
            // chained `a in b in c`) against a known List container
            // whose own element equality this file cannot decide (e.g.
            // a container of opaque class-instance elements,
            // weakref.WeakSet's own `.add(key)` shape) still provably
            // answers A BOOLEAN — expressions.rst, "Comparisons": every
            // `in`/`not in` expression's result IS `bool`, regardless of
            // which value it resolves to. Answered opaque (a boolean
            // sort with no further-known value) rather than fully
            // unknown, so a scalar-ground sink (Age) still fires through
            // the opaque law instead of sitting undetermined. Scoped to
            // a ONE-PAIR chain only — a longer chain still declines
            // fully, since an undecided EARLIER pair could still make
            // the later pair's own truthiness matter to which operand
            // Python's short-circuit `and` would even reach.
            if compare.ops.len() == 1 && matches!(op, CmpOp::In | CmpOp::NotIn) && operand.kind == Kind::List {
                return opaque_value("a boolean value");
            }
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
/// contribution, in source order (expressions.rst, "Formatted string
/// literals"). Only the plainest interpolation shape is modeled: no
/// conversion (`!s`/`!r`/`!a`) and no format spec (`:...`) — either one
/// changes the spelling in ways this file does not compute exactly, so
/// their presence declines the WHOLE f-string rather than composing a
/// partially-wrong string. Two tiers, mirroring refined-ts-go's
/// `evaluateTemplate` (walk/literal_values.go): when every interpolation
/// is EXACTLY readable (a known string, a single known Integer-sorted
/// value spelled bare, or a single known Float-sorted value spelled via
/// `format_py_number`), the whole f-string is one exact string, as
/// before this wave. The moment one interpolation is instead a known SET
/// — sort-only, no exact value (a same-module call's declined-body
/// `summaries::return_sort_fallback`, `float_sorted_unknown()`, or a
/// compiled `Label`-shaped string alias) — the f-string steps down to a
/// PATTERN: every part (literal text, an exact interpolation's spelling,
/// or a set interpolation's own admitted spellings) is a `RefinedSet`,
/// folded by `refinement_forms::concatenation` right to left into one
/// set the checker can still judge a declared max-length against. Only
/// when a part is truly UNREADABLE (`evaluate_expression` answers a
/// shape with no exact spelling and no known sort at all — INCLUDING
/// `Kind::Null`, which this function does not yet compose as the exact
/// word "None" the way CPython's own `str(None)` spells it; see this
/// unit's own report) does the whole f-string stay `unknown()`. NOTE:
/// b-body-expressions.py's own `fstring_unread_substitution`
/// (`f"n={unread_number()}"` against `Label`, max_length=8) is NOT moved
/// by this tier — `unread_number`'s ellipsis-only body is not a decline
/// at all (`summaries::return_sort_fallback`'s own doc), so the call
/// answers `Kind::Null` and this f-string still declines to `unknown()`
/// for it, same as before this wave. An implicitly concatenated f-string
/// (`f"a" f"b"`) is not modeled — only the single-part form
/// (`as_single_part_fstring`) is read.
fn evaluate_fstring(fstring: &ruff_python_ast::ExprFString, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let Some(single) = fstring.as_single_part_fstring() else {
        return unknown();
    };
    let mut composed = String::new();
    let mut has_exact = true;
    let mut parts: Vec<RefinedSet> = Vec::new();
    let mut grade = TrustProved;
    for element in &single.elements {
        match element {
            InterpolatedStringElement::Literal(literal) => {
                if has_exact {
                    composed.push_str(&literal.value);
                }
                if !literal.value.is_empty() {
                    parts.push(refined_sets::codepoint_sets::string_tuple(&literal.value));
                }
            }
            InterpolatedStringElement::Interpolation(interpolation) => {
                if interpolation.conversion != ConversionFlag::None || interpolation.format_spec.is_some() {
                    return unknown();
                }
                let value = evaluate_expression(&interpolation.expression, environment, kernel);
                if let Some(text) = exact_string_values(&value) {
                    let Some(text) = code_points_to_string(text) else {
                        return unknown();
                    };
                    if has_exact {
                        composed.push_str(&text);
                    }
                    parts.push(refined_sets::codepoint_sets::string_tuple(&text));
                } else if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(&value) {
                    let spelling = format_integer_spelling(number);
                    if has_exact {
                        composed.push_str(&spelling);
                    }
                    parts.push(refined_sets::codepoint_sets::string_tuple(&spelling));
                } else if let Some((number, PrimitiveKind::Float)) = single_numeric_value(&value) {
                    let spelling = refined_sets::format_string_shapes::format_py_number(number, true);
                    if has_exact {
                        composed.push_str(&spelling);
                    }
                    parts.push(refined_sets::codepoint_sets::string_tuple(&spelling));
                } else if let Some(part) = spellings_of_known_set(&value) {
                    // a sort-only SET (no exact value): the exact-string
                    // composition can no longer track one spelling, so the
                    // f-string steps down to the pattern tier from here on
                    has_exact = false;
                    grade = refined_domain::trust_grades::min_trust_level(grade, TrustSpec);
                    parts.push(part);
                } else {
                    return unknown();
                }
            }
        }
    }
    if has_exact {
        return string_models::string_literal_value(&composed);
    }
    let Some(mut folded) = parts.pop() else {
        return string_models::string_literal_value("");
    };
    while let Some(part) = parts.pop() {
        folded = make_refined_set(vec![refined_sets::refinement_forms::concatenation(part, folded)]);
    }
    known_set(folded, None, grade, SetKindTag::None)
}

/// The set of strings an f-string interpolation admits, once it is known
/// to be a `Kind::Set` but not readable as one exact value — the
/// spellings-of-a-known-set half of `evaluateTemplate`'s own concatenated
/// pattern (walk/literal_values.go's `case known.Kind == KindSet &&
/// stringy`). A STRING-sorted set (`set_kind_tag == SetKindTag::None`
/// with no numeric `kind_tag` — a compiled `Label`-shaped alias, or the
/// `strings()` set an `__name__` read or `str`-return sort fallback
/// already carries) contributes its OWN set verbatim: every spelling the
/// interpolation can hold IS a member of that set already. A NUMERIC-
/// sorted set (Integer or Float `kind_tag` — `summaries::
/// return_sort_fallback`'s int-sort fallback, or `float_sorted_unknown()`)
/// has no numeric-spelling grammar available in this crate (no
/// `refined_sets` constructor turns "every real number" into "every
/// number's decimal spelling," the same gap refined-ts-go's own
/// `TextOfKnown`/`format_string_shapes` bridges for its host); the honest
/// SOUND weaker claim is `codepoint_sets::strings()` — literally every
/// string, since every real spelling of every admitted number IS some
/// string, so the coarse claim never excludes a spelling the true set
/// would include. Any other `Kind::Set` shape (a set carrying no sort tag
/// at all, or one this function does not recognize) declines — the
/// caller's own `unknown()` fallback stays honest for it.
fn spellings_of_known_set(value: &AbstractValue) -> Option<RefinedSet> {
    if value.kind != Kind::Set {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Number) => Some(strings()),
        Some(PrimitiveKind::String) | None => {
            if value.set_kind_tag == SetKindTag::None {
                Some(value.set.clone())
            } else {
                None
            }
        }
        _ => None,
    }
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

/// A function/method call — dispatch order: (a) a bare name that is
/// EITHER environment-unbound OR bound only to an opaque lambda value
/// (`same_module_def_gate_open`), naming a SAME-MODULE `def`
/// (`environment.functions()`), summarizes through `summaries::call_result`
/// — checked FIRST, so a module-level `def` shadows a builtin of the
/// same name, matching CPython's own name resolution (a later `def
/// len(...):` at module scope really does shadow the builtin `len`);
/// (b) a bare, unbound name naming a same-module class
/// (`environment.classes()`) is a
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
/// Keyword arguments are not modeled for any row EXCEPT the
/// function/construction paths, which map keywords to parameter/field
/// position themselves — every other cited builtin/method signature
/// this wave models takes positional arguments only, so the keyword
/// guard below applies to the builtin/math/method paths. A STARRED
/// positional argument (`max(*xs)`) splices in place when it evaluates
/// to a known `Kind::List` (`splice_call_arguments`'s own doc) — an
/// unknown or unbounded starred argument still declines the whole call,
/// since this file cannot guess how many positional slots it fills.
/// The same-module-`def` gate is `same_module_def_gate_open`, not a bare
/// `environment.read(name).is_none()` check — see that function's own
/// doc for why a name bound to an opaque LAMBDA value still needs to
/// reach the function table.
fn same_module_def_gate_open(environment: &Environment, name: &str) -> bool {
    match environment.read(name) {
        None => true,
        // `f = lambda: ...` binds `f` to `opaque_value("a function
        // value")` (this file's own `Expr::Lambda` arm) — an ordinary
        // program-tracked value binding still blocks the same-module-def
        // dispatch (a real value shadows the def name), but a LAMBDA
        // binding carries no scalar/collection value of its own to
        // shadow anything with, so the gate stays open and `f()` still
        // reaches a same-module `def f(...)` if the module happens to
        // declare one of that name (an unusual but legal shadow: Python
        // itself would call whichever binding is live at the call site,
        // and this file tracks no execution-order distinction between
        // the lambda assignment and a module-level `def` of the same
        // name — the function-table dispatch is the more informative
        // answer of the two shapes this file can read).
        // A CLASS-OBJECT binding likewise keeps the gate open: the walk
        // seeds a class's own bare name to its class-object value (a
        // Kind::Object whose `source` is the class's own name —
        // `instances::class_object_value`), and CALLING that binding IS
        // the construction the classes arm below decides. Any other
        // binding shadows the def/class dispatch as before.
        Some(value) => {
            value.kind == Kind::Object
                && (value.kind_word == Some("a function value") || value.source == name)
        }
    }
}

/// Whether `def`'s body is generator-SHAPED: at least one top-level
/// `yield` statement (`Stmt::Expr` wrapping `Expr::Yield`) anywhere in
/// the body, OR a `yield` one level inside a `for`/`async for` loop body
/// (ruff collapses both into one `Stmt::For` node — see that struct's
/// own generated.rs doc, "collapses the synchronous and asynchronous
/// variants into a single type" — so no separate `AsyncFor` arm is
/// needed) — a-statements.py's own `stream()` shape: `for value in (10,
/// 20, 30): yield value`, a loop-bodied generator with no top-level
/// `yield` at all. Recursion stops at one level (a `yield` nested inside
/// a further `if`/`for`/`try` INSIDE that loop body is not walked) —
/// this is a ROUTING check only, matching the same syntactic fact that
/// makes CPython itself compile a function as a generator (datamodel.rst,
/// "Generator functions": "a function... that uses the `yield`
/// statement... is called a generator function"). `instances::
/// generator_yields` still owns deciding whether the body's EXACT shape
/// is one it can interpret (straight-line yields/an early return/a
/// single literal-iterable `for` loop, `Kind::Unknown` on any richer
/// control flow) — a body this function calls generator-shaped but
/// `generator_yields` cannot read answers `unknown()` at the call site,
/// the same decline every other unmodeled body shape in this file
/// already gives.
fn is_generator_def(def: &ruff_python_ast::StmtFunctionDef) -> bool {
    def.body.iter().any(|stmt| match stmt {
        ruff_python_ast::Stmt::Expr(expr_stmt) => matches!(expr_stmt.value.as_ref(), Expr::Yield(_)),
        ruff_python_ast::Stmt::For(for_stmt) => for_stmt.body.iter().any(|inner| {
            matches!(inner, ruff_python_ast::Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::Yield(_)))
        }),
        _ => false,
    })
}

/// `range(...)` read as an EXPRESSION VALUE — library/stdtypes.rst,
/// `class:: range(stop)` / `range(start, stop[, step])`: "The advantage
/// of the range type over a regular list or tuple is that a range
/// object will always take the same (small) amount of memory... range
/// objects implement the `collections.abc.Sequence` ABC." This domain
/// materializes it as a `Kind::List` of Integer-sorted elements — the
/// same eager-materialization choice `loops.rs`'s own `range_call_values`
/// already makes for a `for`-loop iterable, reused here for a
/// non-loop expression context (a comprehension iterable, a bare
/// value, etc.) — `range`'s own elements are always int
/// (stdtypes.rst's `range` entry: "the arguments must be integers").
/// Every argument must be a known single Integer-sorted value (an
/// evaluated expression, not a literal-syntax-only reading — this
/// differs from `loops.rs`'s own syntactic reader, which only serves
/// the `for`-loop path it owns); a non-Integer/unknown argument, a
/// zero step, or an argument count outside 1/2/3 declines.
fn range_expression_value(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let (start, stop, step) = match arguments {
        [stop] => (0.0, range_argument_value(stop)?, 1.0),
        [start, stop] => (range_argument_value(start)?, range_argument_value(stop)?, 1.0),
        [start, stop, step] => (range_argument_value(start)?, range_argument_value(stop)?, range_argument_value(step)?),
        _ => return None,
    };
    if step == 0.0 {
        return None;
    }
    let mut values = Vec::new();
    let mut current = start;
    // r[i] = start + step*i, while r[i] < stop (step > 0) or r[i] > stop
    // (step < 0) — stdtypes.rst's own range formula
    if step > 0.0 {
        while current < stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    } else {
        while current > stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    }
    Some(collection_models::list_literal_value(&values))
}

/// One `range(...)` argument's known Integer value, or `None` if it is
/// not a single known Integer-sorted value.
fn range_argument_value(value: &AbstractValue) -> Option<f64> {
    let (number, sort) = single_numeric_value(value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    Some(number)
}

/// `functools.reduce(function, iterable[, initializer])` —
/// functools.rst's own entry: "Apply *function* of two arguments
/// cumulatively to the items of *iterable*, from left to right... The
/// left argument, *x*, is the accumulated value and the right
/// argument, *y*, is the update value from the *iterable*. If the
/// optional *initializer* is present, it is placed before the items of
/// the iterable... and serves as a default when the iterable is
/// empty." Folded CONCRETELY, one call per element — `iterable` must
/// be a known `Kind::List`; `function` is read as a RAW two-parameter
/// expression (`Expr::Lambda`, or a bare `Expr::Name` resolving to a
/// same-module `def` in the function table), never an already-
/// evaluated value, since this domain's abstract values carry no
/// callable body to fold with once evaluated (`opaque_value("a
/// function value")` states only the SORT, not the body) —
/// `call_two_argument_expression` is the one seam that reads the raw
/// expression instead. Declines the whole call for any other
/// `function`/`iterable` shape, a missing `initializer` on an empty
/// iterable (functools.rst's own "TypeError... reduce() of empty
/// iterable with no initial value"), or a fold step this file cannot
/// evaluate.
fn reduce_expression_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let (function_expr, iterable_expr, initializer_expr) = match &*call.arguments.args {
        [function_expr, iterable_expr] => (function_expr, iterable_expr, None),
        [function_expr, iterable_expr, initializer_expr] => (function_expr, iterable_expr, Some(initializer_expr)),
        _ => return None,
    };
    let iterable = evaluate_expression(iterable_expr, environment, kernel);
    if iterable.kind != Kind::List {
        return None;
    }
    let mut elements = iterable.items.iter();
    let mut accumulator = match initializer_expr {
        Some(expr) => evaluate_expression(expr, environment, kernel),
        None => elements.next()?.clone(),
    };
    for element in elements {
        accumulator = call_two_argument_expression(function_expr, &accumulator, element, environment, kernel)?;
    }
    Some(accumulator)
}

/// One call to a RAW two-parameter callable expression: an
/// `Expr::Lambda` of exactly two parameters (its body is always a
/// single expression, expressions.rst's "Lambdas" — evaluated directly
/// against a fork binding both parameters), or a bare `Expr::Name`
/// resolving to a same-module `def` in the function table (folded
/// through `summaries::call_result`, the same restricted interpreter
/// every other same-module call in this file already uses). Any other
/// callable shape (a builtin name, a method reference, a lambda/def of
/// a different arity) declines.
fn call_two_argument_expression(
    function_expr: &Expr,
    first: &AbstractValue,
    second: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    match function_expr {
        Expr::Lambda(lambda) => {
            let parameters = lambda.parameters.as_deref()?;
            let all_parameters: Vec<_> = parameters.posonlyargs.iter().chain(parameters.args.iter()).collect();
            let [first_parameter, second_parameter] = all_parameters.as_slice() else {
                return None;
            };
            let mut fork = environment.fork();
            fork.bind(first_parameter.parameter.name.id.as_str(), first.clone());
            fork.bind(second_parameter.parameter.name.id.as_str(), second.clone());
            Some(evaluate_expression(&lambda.body, &fork, kernel))
        }
        Expr::Name(name) if environment.read(name.id.as_str()).is_none() => {
            let table = environment.functions()?;
            let def = table.def(name.id.as_str())?;
            summaries::call_result(def, &[first.clone(), second.clone()], environment.functions(), kernel, environment.call_depth())
        }
        _ => None,
    }
}

/// One call to a RAW ONE-parameter callable expression — the `map`/
/// `filter` twin of `call_two_argument_expression`'s own doc, same two
/// callable shapes (a one-parameter `Expr::Lambda`, or a bare
/// `Expr::Name` resolving to a same-module one-parameter `def`), same
/// decline on any other shape.
fn call_one_argument_expression(
    function_expr: &Expr,
    argument: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    match function_expr {
        Expr::Lambda(lambda) => {
            let parameters = lambda.parameters.as_deref()?;
            let all_parameters: Vec<_> = parameters.posonlyargs.iter().chain(parameters.args.iter()).collect();
            let [only_parameter] = all_parameters.as_slice() else {
                return None;
            };
            let mut fork = environment.fork();
            fork.bind(only_parameter.parameter.name.id.as_str(), argument.clone());
            Some(evaluate_expression(&lambda.body, &fork, kernel))
        }
        Expr::Name(name) if environment.read(name.id.as_str()).is_none() => {
            let table = environment.functions()?;
            let def = table.def(name.id.as_str())?;
            summaries::call_result(def, &[argument.clone()], environment.functions(), kernel, environment.call_depth())
        }
        _ => None,
    }
}

/// `map(function, iterable)` — functions.html#map: "Return an iterator
/// that applies *function* to every item of *iterable*, yielding the
/// results." Folded CONCRETELY, one call per element, over a known
/// `Kind::List` iterable (the eager materialization
/// `range_expression_value`'s own doc already establishes for a lazy
/// builtin sequence in this domain) — `function` is read as a RAW
/// one-parameter expression (`call_one_argument_expression`'s own doc),
/// never an already-evaluated value. Declines the whole call for a
/// non-List iterable, or the moment one element's own call cannot be
/// evaluated — no element is silently dropped.
fn map_expression_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [function_expr, iterable_expr] = &*call.arguments.args else {
        return None;
    };
    let iterable = evaluate_expression(iterable_expr, environment, kernel);
    if iterable.kind != Kind::List {
        return None;
    }
    let mut mapped = Vec::with_capacity(iterable.items.len());
    for element in &iterable.items {
        mapped.push(call_one_argument_expression(function_expr, element, environment, kernel)?);
    }
    Some(collection_models::list_literal_value(&mapped))
}

/// `filter(predicate, iterable)` — functions.html#filter: "Construct an
/// iterator from those elements of *iterable* for which *function*
/// returns true." Folded the same way `map_expression_value` is: one
/// predicate call per element over a known `Kind::List`, an element kept
/// only when its own call's truthiness is DEFINITELY true
/// (`lattice_operations::truthiness`) — an undecidable predicate result
/// for any element declines the WHOLE call rather than guess whether
/// that element belongs in the kept list (a single dropped-vs-kept
/// element changes every later index).
fn filter_expression_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [predicate_expr, iterable_expr] = &*call.arguments.args else {
        return None;
    };
    let iterable = evaluate_expression(iterable_expr, environment, kernel);
    if iterable.kind != Kind::List {
        return None;
    }
    let mut kept = Vec::with_capacity(iterable.items.len());
    for element in &iterable.items {
        let outcome = call_one_argument_expression(predicate_expr, element, environment, kernel)?;
        let (truthy, known) = truthiness(&outcome);
        if !known {
            return None;
        }
        if truthy {
            kept.push(element.clone());
        }
    }
    Some(collection_models::list_literal_value(&kept))
}

/// Whether `name` is one of the built-in exception classes this file
/// constructs an `args`-carrying (or, for `ExceptionGroup`, opaque)
/// instance for — `exceptions.rst`'s own class hierarchy: `Exception`,
/// `ValueError`, `RuntimeError`, `TypeError`, and `KeyError` are each a
/// bare `BaseException.__init__(*args)` call with no extra fields of
/// their own (unlike `OSError`'s special-cased constructor, which this
/// file does not model). `ExceptionGroup` is listed here too so
/// `evaluate_call`'s ONE gate covers every recognized exception name,
/// even though it answers opaque rather than the tagged `args` shape
/// the others do (see that call site's own doc).
fn is_builtin_exception_constructor(name: &str) -> bool {
    matches!(name, "Exception" | "ValueError" | "RuntimeError" | "TypeError" | "KeyError" | "ExceptionGroup")
}

/// `Exception(*args)` / `ValueError(*args)` / `RuntimeError(*args)` /
/// `TypeError(*args)` / `KeyError(*args)` — a tagged `Kind::Object`
/// (`source = "exception"`) carrying every positional constructor
/// argument, in order, under one `args` field: tutorial/errors.rst
/// §8.3, "the exception instance... typically has an `args` attribute
/// that stores the arguments." Reads through this tag: `.args[0]` (this
/// file's own `evaluate_attribute_read`'s untagged-instance fallback,
/// since no `ClassModel` is ever registered under the name
/// `"exception"`, so `instances::field_read`'s plain by-name scan
/// answers the `args` `ObjectKey` directly) and `str(...)`
/// (`builtin_models::str_call`'s exception row, reading the SAME `args`
/// field by name).
pub(crate) fn exception_construction_value(arguments: &[AbstractValue]) -> AbstractValue {
    let args = collection_models::list_literal_value(arguments);
    let mut instance = known_object(
        vec![ObjectKey {
            name: "args".to_owned(),
            numeric: false,
            value: args,
        }],
        None,
        true,
        TrustProved,
        false,
    );
    instance.source = "exception".to_owned();
    instance
}

/// The tagged, FIELDLESS exception shape `check.rs`'s own
/// `caught_exception_value` binds a caught exception name to when the
/// try body's own raise cannot be found (a computed exception type, a
/// bare `except:`, more than one matching raise, …): the same
/// `source = "exception"` tag `exception_construction_value` gives a
/// freshly-constructed exception, but with no `args`/`__cause__`
/// field — a read through it (`.args`, `.__cause__`) finds nothing this
/// domain models, the honest "not yet readable" answer, never a false
/// Unknown-is-opaque read that a bare `opaque_value` would give (an
/// opaque value carries no `source` at all, so it cannot even be
/// recognized as an exception by a later `isinstance`/`str()` reader).
pub(crate) fn fieldless_exception_value() -> AbstractValue {
    let mut instance = known_object(Vec::new(), None, true, TrustProved, false);
    instance.source = "exception".to_owned();
    instance
}

/// Every element of a known `Kind::List` receiver, read as a single
/// known Integer in `0..=255` — the shared reader `bytes_like_
/// construction_value`'s own `bytes(<list>)`/`bytearray(<list>)` rows
/// need to turn an already-evaluated argument list back into the raw
/// `u8` sequence `bytes_models::bytes_literal_value` takes. `None` the
/// moment the receiver is not a known list, or any element is not a
/// known Integer in range — CPython itself raises `ValueError: bytes
/// must be in range(0, 256)` for an out-of-range element at
/// CONSTRUCTION time (`bytes_literal_value`'s own doc), a fact this
/// file does not yet speak through a `provable_raise` row for the
/// constructor call itself, so an out-of-range element declines the
/// whole construction rather than silently clamp it.
fn known_byte_sequence(value: &AbstractValue) -> Option<Vec<u8>> {
    if value.kind != Kind::List {
        return None;
    }
    value
        .items
        .iter()
        .map(|item| {
            let (raw, sort) = single_numeric_value(item)?;
            if sort != PrimitiveKind::Integer {
                return None;
            }
            if !(0.0..=255.0).contains(&raw) {
                return None;
            }
            Some(raw as u8)
        })
        .collect()
}

/// `bytes(...)` / `bytearray(...)` / `memoryview(...)` construction —
/// p-typed-array.py's own construction band, wired onto
/// `bytes_models.rs`'s existing element machinery (that file's own
/// module doc: no dedicated bytes/array `Kind` exists or is needed,
/// every one of these values is the identical `Kind::List` an ordinary
/// list literal builds).
///
/// - `bytearray(<known Integer length>)` — `bytearray_from_length`'s
///   own row: stdtypes.rst's `bytearray([source[, encoding[,
///   errors]]])`, "If it is an integer, the array will have that size
///   and will be initialized with null bytes." A length outside
///   `0..=1024` declines (an honest bound against building an
///   unreasonably large element vector for a value this file never
///   needs beyond the corpus's own small fixtures).
/// - `bytes(<known list of known Integers 0..=255>)` /
///   `bytearray(<known list of known Integers 0..=255>)` — `bytes_
///   from_iterable`'s own row: "If it is an iterable, it must be an
///   iterable of integers in the range 0 <= x < 256."
/// - `bytearray(<known bytes-like value>)` / `bytes(<known bytes-like
///   value>)` — `bytes_is_immutable`'s own `frozen = bytes(data)` row
///   (copying a `bytearray` into an immutable `bytes`, or vice versa):
///   the SAME known-list-of-known-Integers shape the row above reads,
///   since a `bytearray`/`bytes` value already IS that shape once
///   built — no separate reader needed.
/// - `memoryview(<known bytearray/bytes value>)` — `memoryview_over_
///   bytearray_reads`'s own row: a view SHARES the underlying buffer
///   (`memoryview(ba)[i]` reads/writes the same elements `ba[i]`
///   would), so this file answers the identical `Kind::List` value
///   unchanged rather than building a distinct wrapper shape — this
///   domain has no separate "view" Kind, and a plain copy-through is
///   sound for every read/len/index this corpus exercises (the
///   shared-buffer WRITE-back-through-the-view effect is check.rs's
///   own statement-sink business, not a value-construction concern).
///
/// Any other argument shape (zero arguments, more than one argument, a
/// non-Integer/out-of-range element, an unknown receiver) declines —
/// this function states nothing beyond the shapes listed above.
fn bytes_like_construction_value(
    constructor: &str,
    args: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let [only] = args else { return None };
    let argument = evaluate_expression(only, environment, kernel);
    if constructor == "memoryview" {
        if argument.kind == Kind::List {
            return Some(argument);
        }
        return None;
    }
    if constructor == "bytearray" {
        if let Some((length, PrimitiveKind::Integer)) = single_numeric_value(&argument) {
            if (0.0..=1024.0).contains(&length) {
                let zeroes = vec![0u8; length as usize];
                return Some(bytes_models::bytes_literal_value(&zeroes));
            }
            return None;
        }
    }
    let bytes = known_byte_sequence(&argument)?;
    Some(bytes_models::bytes_literal_value(&bytes))
}

/// `array.array('d', [...])` — the Float64Array twin,
/// p-typed-array.py's `array_double_from_iterable`/`array_double_
/// write_and_read_back`: `array.rst`'s own `class:: array(typecode[,
/// initializer])`, typecode `'d'` (double). Modeled ONLY for the exact
/// two-argument form with a known exact-string typecode `"d"` and a
/// known list of known numeric (Integer or Float) elements — every
/// element widens to Float on read (`bytes_models::
/// array_double_literal_value`'s own doc: an `array.array('d', ...)`
/// element is ALWAYS a Python `float`, whatever numeric literal built
/// it). Any other typecode, arity, or a non-numeric element declines —
/// this file models the one typecode the corpus's own Float64Array-twin
/// rows use.
fn array_double_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [typecode_expr, initializer_expr] = &*call.arguments.args else {
        return None;
    };
    let typecode = evaluate_expression(typecode_expr, environment, kernel);
    let typecode_text = exact_string_values(&typecode).and_then(code_points_to_string)?;
    if typecode_text != "d" {
        return None;
    }
    let initializer = evaluate_expression(initializer_expr, environment, kernel);
    if initializer.kind != Kind::List {
        return None;
    }
    let elements: Vec<f64> = initializer
        .items
        .iter()
        .map(|item| single_numeric_value(item).map(|(value, _sort)| value))
        .collect::<Option<Vec<f64>>>()?;
    Some(bytes_models::array_double_literal_value(&elements))
}

/// Whether `attribute` is exactly the two-level attribute chain
/// `datetime.datetime` with `datetime` NOT locally shadowed — the
/// receiver shape both the `datetime.datetime(...)` CONSTRUCTION call
/// (this function's caller in `evaluate_call`) and the
/// `datetime.datetime.now()` CLASSMETHOD call
/// (`evaluate_attribute_call`'s own datetime arm) both gate on.
fn is_datetime_datetime_attribute(attribute: &ruff_python_ast::ExprAttribute, environment: &Environment) -> bool {
    if attribute.attr.as_str() != "datetime" {
        return false;
    }
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "datetime" && environment.read("datetime").is_none()
}

/// `datetime.datetime(year, month, day, hour=0, minute=0, second=0, ...,
/// tzinfo=...)` — a tagged `Kind::Object` (`source = "datetime_datetime"`)
/// carrying `year`/`month`/`day`/`hour`/`minute`/`second` as Integer
/// `ObjectKey`s, PLUS an `aware_utc` marker (a Boolean `ObjectKey`) —
/// datetime.rst, `class:: datetime(year, month, day, hour=0, minute=0,
/// second=0, microsecond=0, tzinfo=None, *, fold=0)`. Modeled ONLY when
/// every positional/keyword argument this file reads is a known Integer
/// literal (year/month/day always positional in this corpus;
/// hour/minute/second read from EITHER a positional slot or a keyword,
/// defaulting to 0 when absent, matching the constructor's own
/// defaults) — a `microsecond`/`fold` argument, or ANY argument this
/// file cannot read as a known Integer, declines the WHOLE construction
/// (never a partially-built datetime). `tzinfo=` is read SYNTACTICALLY
/// (the keyword's own value expression, not its evaluated AbstractValue
/// — `datetime.timezone.utc`/`datetime.UTC` have no abstract value this
/// file tracks): `aware_utc` is `true` only when the keyword's value
/// expression is exactly `datetime.timezone.utc` or `datetime.UTC`
/// (datetime.rst's own "Alias for the UTC time zone singleton
/// datetime.timezone.utc" — `UTC` added 3.11, `timezone.utc` older),
/// `false` when `tzinfo` is absent (a NAIVE datetime), and the whole
/// construction declines for any OTHER `tzinfo=` expression (a
/// non-UTC/non-recognized timezone this file cannot prove an exact UTC
/// offset for).
fn datetime_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let positional_names = ["year", "month", "day", "hour", "minute", "second"];
    let mut fields: Vec<Option<i64>> = vec![None; positional_names.len()];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        let slot = fields.get_mut(index)?;
        *slot = Some(datetime_field_argument(arg, environment, kernel)?);
    }
    let mut aware_utc: Option<bool> = None;
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            return None;
        };
        if arg_name.as_str() == "tzinfo" {
            if !is_utc_tzinfo_expression(&keyword.value) {
                // a tzinfo this file cannot prove is exactly UTC —
                // decline the whole construction rather than guess an
                // offset (datetime_construction_value's own doc)
                return None;
            }
            aware_utc = Some(true);
            continue;
        }
        let Some(position) = positional_names.iter().position(|name| *name == arg_name.as_str()) else {
            // `microsecond=`/`fold=` (or any other keyword) — not
            // modeled, decline the whole construction
            return None;
        };
        let slot = fields.get_mut(position)?;
        *slot = Some(datetime_field_argument(&keyword.value, environment, kernel)?);
    }
    // year/month/day have no default (positional-required per the
    // constructor's own signature); hour/minute/second default to 0
    let mut keys = Vec::with_capacity(positional_names.len() + 1);
    for (index, name) in positional_names.iter().enumerate() {
        let value = match fields[index] {
            Some(value) => value,
            None if index < 3 => return None,
            None => 0,
        };
        keys.push(integer_object_key(name, value));
    }
    keys.push(ObjectKey {
        name: "aware_utc".to_owned(),
        numeric: false,
        value: known_values(vec![if aware_utc.unwrap_or(false) { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved),
    });
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_datetime".to_owned();
    Some(instance)
}

/// One `ObjectKey` carrying a known Integer field — the small builder
/// `datetime_construction_value` repeats once per calendar field.
fn integer_object_key(name: &str, value: i64) -> ObjectKey {
    ObjectKey {
        name: name.to_owned(),
        numeric: false,
        value: known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved),
    }
}

/// One `datetime.datetime(...)` constructor argument's known Integer
/// value — every positional/keyword calendar field this file reads
/// (`datetime_construction_value`'s own doc).
fn datetime_field_argument(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let value = evaluate_expression(expr, environment, kernel);
    let (number, sort) = single_numeric_value(&value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    Some(number as i64)
}

/// Whether `expr` is exactly `datetime.timezone.utc` or `datetime.UTC`
/// — the two spellings datetime.rst documents for the UTC singleton
/// (`datetime_construction_value`'s own doc). Read SYNTACTICALLY (the
/// expression's own dotted-name shape), never by evaluating to an
/// AbstractValue — this file tracks no tzinfo value at all.
fn is_utc_tzinfo_expression(expr: &Expr) -> bool {
    // datetime.UTC — a two-level chain, `Name("datetime").UTC`
    if let Expr::Attribute(outer) = expr {
        if outer.attr.as_str() == "UTC" {
            if let Expr::Name(name) = outer.value.as_ref() {
                if name.id.as_str() == "datetime" {
                    return true;
                }
            }
        }
        // datetime.timezone.utc — a three-level chain,
        // `Name("datetime").timezone.utc`
        if outer.attr.as_str() == "utc" {
            if let Expr::Attribute(middle) = outer.value.as_ref() {
                if middle.attr.as_str() == "timezone" {
                    if let Expr::Name(name) = middle.value.as_ref() {
                        return name.id.as_str() == "datetime";
                    }
                }
            }
        }
    }
    false
}

/// The proleptic Gregorian day count from the civil (year, month, day)
/// triple to the POSIX epoch (1970-01-01 = day 0) — Howard Hinnant's
/// `days_from_civil` algorithm, the same closed-form calendar arithmetic
/// `date.toordinal()` computes internally (datetime.rst, `method::
/// date.toordinal()`: "the current proleptic Gregorian ordinal"; day 0
/// here is ordinal `date(1970, 1, 1).toordinal()`, so the DIFFERENCE
/// this function returns is exactly `date(y, m, d).toordinal() -
/// date(1970, 1, 1).toordinal()`, matching `.timestamp()`'s own
/// documented aware-datetime formula, `(dt - datetime(1970, 1, 1,
/// tzinfo=timezone.utc)).total_seconds()`, one calendar step earlier
/// than the seconds-of-day addition `datetime_timestamp_value` performs
/// next). Execution-verified against installed CPython 3.12 for both
/// this file's own corpus dates (1970-01-01 -> 0, 2033-05-18 ->
/// 1999987200.0 once seconds are added) in this wave's own report.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// `<an aware-UTC datetime_datetime instance>.timestamp()` — the EXACT
/// POSIX timestamp: datetime.rst, `method:: datetime.timestamp()`, "For
/// aware datetime instances, the return value is computed as: `(dt -
/// datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()`." UTC has
/// no DST/leap-second adjustment, so that difference reduces to plain
/// calendar-day arithmetic (`days_from_civil`'s own doc) times 86400,
/// plus the wall-clock seconds-of-day. Modeled ONLY for a
/// `datetime_construction_value`-tagged instance whose own `aware_utc`
/// field is `true` — `None` for a NAIVE instance (datetime.rst's own
/// note: "Naive datetime instances are assumed to represent local time
/// and this method relies on the platform C mktime function," a
/// host-dependent conversion this file does not claim to reproduce).
fn datetime_timestamp_value(instance: &AbstractValue) -> Option<AbstractValue> {
    let aware = datetime_field(instance, "aware_utc")?;
    if aware != 1.0 {
        return None;
    }
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let hour = datetime_field(instance, "hour")? as i64;
    let minute = datetime_field(instance, "minute")? as i64;
    let second = datetime_field(instance, "second")? as i64;
    let days = days_from_civil(year, month, day);
    let seconds = days * 86400 + hour * 3600 + minute * 60 + second;
    Some(known_values(vec![seconds as f64], PrimitiveKind::Float, TrustProved))
}

/// One numeric `ObjectKey` field's own value off a tagged instance — the
/// linear scan `datetime_timestamp_value` reads each calendar field
/// through (the same by-name `ObjectKey` shape `instances::field_read`
/// reads for an untagged instance, repeated here as a private single-
/// field helper since every caller already knows the exact field name
/// it wants).
fn datetime_field(instance: &AbstractValue, name: &str) -> Option<f64> {
    let entry = instance.keys.iter().find(|key| key.name == name)?;
    let (value, _) = single_numeric_value(&entry.value)?;
    Some(value)
}

/// `callee(...)` where `callee` is a retained-callable value
/// (`env::retained_callable_key` reads `Some`) — resolves the call
/// through the SAME restricted interpreter an ordinary same-module
/// `def` call already uses (`summaries::call_result_with_enclosing`),
/// never a second one built for this table. `None` ONLY when `callee`
/// is not a retained-callable value at all — the signal `evaluate_
/// call`'s own caller reads to fall through to its other dispatch
/// arms. Once `callee` IS recognized as a retained-callable value,
/// this function always answers `Some` — a table miss (`environment`
/// never recorded this exact key) or an arity/interpretation decline
/// answers `Some(unknown())`, never `None`, so a caller never
/// mistakes "this really is a retained-callable call, and it
/// declined" for "try the ordinary def/builtin dispatch instead,"
/// which could read a stale or wrong same-module def of the same bare
/// name.
///
/// The retained body's own CLOSURE snapshot (free names read from the
/// environment AT THE MOMENT the value was created, `RetainedCallable`'s
/// own doc) seeds a throwaway environment that
/// `call_result_with_enclosing`'s `enclosing` parameter reads free
/// names from — the same closure-reading contract that function
/// already gives an ordinary nested `def`, reused rather than
/// duplicated. Positional arguments read through `positional_
/// arguments_for_def`, the same keyword-to-position mapping and arity
/// checking an ordinary same-module call already gets — one binding
/// law, not a second for retained callables.
fn retained_callable_call_result(
    callee: &AbstractValue,
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let key = env::retained_callable_key(callee)?;
    let Some(retained) = environment.retained_callable(key) else {
        return Some(unknown());
    };
    let def = retained.as_synthetic_def("<retained>", call.range());
    let Some(positional) = positional_arguments_for_def(call, &def, environment, kernel) else {
        return Some(unknown());
    };
    let answer = if retained.closure.is_empty() {
        summaries::call_result_with_enclosing(&def, &positional, environment.functions(), kernel, environment.call_depth(), None)
    } else {
        let mut closure_environment = Environment::new(std::collections::HashSet::new());
        for (name, value) in &retained.closure {
            closure_environment.bind(name, value.clone());
        }
        summaries::call_result_with_enclosing(
            &def,
            &positional,
            environment.functions(),
            kernel,
            environment.call_depth(),
            Some(&closure_environment),
        )
    };
    Some(answer.unwrap_or_else(unknown))
}

fn evaluate_call(call: &ruff_python_ast::ExprCall, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    // A RETAINED-CALLABLE CALL: `name(...)` where `name` reads a value
    // `env::retained_callable_value` built — a lambda or nested `def`
    // that reached this call site through a binding path other than
    // "declared and called in the same body" (returned out of its
    // defining function, passed in as a call argument, read back off
    // an instance field). Tried BEFORE the same-module-def dispatch
    // below: a retained callable's own table entry is a stronger,
    // execution-traced fact than a bare same-module `def` of the same
    // spelling would be, and — for `add_one = make_adder(1)` — there
    // is no module-level `def add_one` for that dispatch to find
    // anyway, so trying this first changes nothing for the shapes that
    // DO have a same-module def of the lambda-bound name
    // (`same_module_def_gate_open`'s own doc already treats that name
    // as open, meaning a real module-level `def` of the same spelling
    // is the intended callee there — this arm never reaches that case
    // since `retained_callable_key` answers `None` for an ordinary,
    // non-retained lambda value).
    if let Expr::Name(name) = call.func.as_ref() {
        if let Some(value) = environment.read(name.id.as_str()) {
            if let Some(result) = retained_callable_call_result(value, call, environment, kernel) {
                return result;
            }
        }
    }
    if let Expr::Name(name) = call.func.as_ref() {
        if same_module_def_gate_open(environment, name.id.as_str()) {
            if let Some(table) = environment.functions() {
                if let Some(def) = table.def(name.id.as_str()) {
                    let Some(positional) = positional_arguments_for_def(call, def, environment, kernel) else {
                        return unknown();
                    };
                    // a GENERATOR function's own call (a body whose
                    // top-level statements are straight-line `yield`s,
                    // `is_generator_def`'s own doc) never reaches
                    // `summaries::call_result` — that restricted
                    // interpreter has no `yield` row and would decline
                    // the whole call. `generator_yields` reads the same
                    // body instead, and the CALL answers the ordered
                    // List of every yielded value (this domain's shared
                    // list/set/generator representation,
                    // `collection_models.rs`'s own module doc), tagged
                    // `source = "generator"` so `next`'s own dispatcher
                    // (`next_call`/`evaluate_call`'s builtin path) can
                    // tell a fresh generator value apart from an
                    // ordinary list — see `next`'s own doc for why a
                    // SECOND `next` on the same value declines rather
                    // than answering the next element (this domain
                    // carries no generator position/exhaustion state).
                    if is_generator_def(def) {
                        return match instances::generator_yields(def, &positional, environment.functions(), kernel, environment.call_depth())
                        {
                            Some(yields) => {
                                let mut value = collection_models::list_literal_value(&yields);
                                value.source = "generator".to_owned();
                                value
                            }
                            None => unknown(),
                        };
                    }
                    // CLOSURE READS: `def` may be a NESTED def (this
                    // call's own `environment` is the enclosing body's
                    // locals at the call site) reading a free name neither
                    // its own parameters nor its own body bind —
                    // `call_result_with_enclosing`'s own doc
                    // (executionmodel.rst's "Naming and binding": a free
                    // variable reads the enclosing scope's binding). Passing
                    // the CALL SITE's `environment` here is sound for a
                    // same-body define-then-call flow (the corpus's own
                    // shape — a nested `def` declared and called inside the
                    // same enclosing body): the enclosing environment at
                    // the point of the call already carries whatever the
                    // enclosing body bound before this call ran. A
                    // module-level `def` (no true enclosing scope) still
                    // answers identically either way — its own
                    // `free_names_read` walk never finds a name the
                    // enclosing environment did not already fail to bind
                    // either, so seeding costs nothing when there is
                    // nothing to seed.
                    return match summaries::call_result_with_enclosing(
                        def,
                        &positional,
                        environment.functions(),
                        kernel,
                        environment.call_depth(),
                        Some(environment),
                    ) {
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
        // A CALLABLE-VARIABLE CALL: `name(...)` where `name` is a bare Name
        // this environment's `callable_returns` table carries (a
        // `Callable[[...], R]`-annotated variable,
        // `typereading::callable_return_refinement` / `walk_ann_assign`'s
        // own recording seam) AND `name` does not ALSO resolve to a
        // same-module def/class. Placed OUTSIDE the `same_module_def_gate_
        // open` block and checked directly here (not by relying on that
        // gate to have excluded a def/class name already): a MODULE-LEVEL
        // `Callable`-typed name read from inside a function body is
        // usually gate-OPEN anyway (the name is never in that function's
        // own `locally_bound` set, so `environment.read` answers `None`
        // there, same as any other unbound outer name), so the def/class
        // dispatch above already tries first and returns early whenever
        // one of them actually matches — this direct check exists for the
        // remaining case, a LOCAL `Callable`-typed rebinding the gate
        // would close (bound to a real value, not an opaque lambda),
        // where the def/class dispatch above is skipped entirely and this
        // arm is the only remaining check standing between it and a wrong
        // answer. This is the same channel
        // `check.rs::callable_variable_call_result` gives `sink_value`'s
        // direct-sink shape (`x: Age = maybe_next_year(40)`) — this arm is
        // the NESTED-expression twin, reached when the call sits inside a
        // larger expression (b-body-expressions.py:79's ternary-guarded
        // `maybe_next_year(40) if maybe_next_year is not None else 0`,
        // where the call is evaluated by `evaluate_ternary`'s own
        // `evaluate_expression` recursion, never by `sink_value`). Answers
        // `R`'s own declared set at TrustSpec — an annotation is the
        // developer's claim, not an execution-proved fact — the same
        // grade `callable_variable_call_result` uses.
        if let Some(declared) = environment.callable_returns().and_then(|table| table.get(name.id.as_str())) {
            let shadowed_by_def = environment.functions().is_some_and(|functions| functions.def(name.id.as_str()).is_some());
            let shadowed_by_class = environment.classes().is_some_and(|classes| classes.contains_key(name.id.as_str()));
            if !shadowed_by_def && !shadowed_by_class {
                return known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None);
            }
        }
    }
    // `receiver.method(...)` on a known INSTANCE (a `Kind::Object` whose
    // `source` names the constructing class, `judge_construction`'s own
    // tag) — the method's own def resolves through `method_def_of`, then
    // `method_call_result` interprets it the same restricted way
    // `summaries::call_result` interprets a plain `def`, with keyword
    // arguments mapped to position first (this is the ONE method-call
    // path that reads keywords — every other method/builtin/math row
    // below still declines any keyword, per the existing guard). Only
    // the RESULT half of `method_call_result`'s `(instance after, result)`
    // pair is read here: the mutated-instance half is check.rs's own
    // statement-sink business (the same "fires/writes belong to the
    // sink" split the construction arm above already draws), so a
    // nested method call inside a larger expression never threads its
    // own receiver mutation back into the environment.
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        // a `math`/`re`/`asyncio` MODULE-name receiver evaluates to
        // `unknown()` here (no binding, no class) and simply misses the
        // `Kind::Object`-with-`source` check below, falling through to
        // `evaluate_attribute_call`'s own module-name arms unaffected.
        let receiver = evaluate_expression(&attribute.value, environment, kernel);
        // A RETAINED-CALLABLE FIELD CALL: `receiver.attr(...)` where
        // `attr` is a STORED field (never a class method — a `def` in
        // the class body resolves through `method_def_of` below
        // instead) holding a retained lambda/def value
        // (b-body-expressions.py's `function_nested_on_object`:
        // `Person(lambda: 40)` stores the lambda as `self.years`, and
        // `person.years()` calls it back). Tried before the
        // class-method dispatch: a field and a method never share a
        // name on the same class (`instances::field_read`/`method_def_
        // of` both key off the class's own single namespace), so this
        // never shadows an actual method call.
        if receiver.kind == Kind::Object {
            if let Some(field) = instances::field_read(&receiver, attribute.attr.as_str()) {
                if let Some(result) = retained_callable_call_result(&field, call, environment, kernel) {
                    return result;
                }
            }
        }
        if receiver.kind == Kind::Object && !receiver.source.is_empty() {
            if let Some(classes) = environment.classes() {
                if let Some(model) = classes.get(receiver.source.as_str()) {
                    if let Some(method) = instances::method_def_of(model, attribute.attr.as_str()) {
                        let Some(positional) = positional_arguments_for_method(call, method, environment, kernel) else {
                            return unknown();
                        };
                        // A GENERATOR METHOD (`class GenAges: def ages(self):
                        // yield 40`, e-class-and-function.py's own
                        // `generator_method`/`async_generator_method`) —
                        // the exact same `Stmt::Expr(Expr::Yield)`-shaped
                        // body `evaluate_call`'s bare-def dispatch already
                        // routes to `instances::generator_yields` rather
                        // than `method_call_result` (that call site's own
                        // doc: "that restricted interpreter has no `yield`
                        // row and would decline the whole call"). A method
                        // body is the identical restricted-interpreter
                        // shape one level down (self bound, otherwise the
                        // same straight-line-yields reading), so this arm
                        // checks the SAME `is_generator_def` gate before
                        // ever trying `method_call_result` — a generator
                        // method reaching that function instead would
                        // simply decline on its first `Stmt::Expr(Expr::
                        // Yield)` statement, the same as a bare generator
                        // def would without this arm. `generator_yields`
                        // binds its OWN `def.parameters` positionally with
                        // no `self`-awareness of its own (it is a plain-def
                        // reader, `instances.rs`'s own doc — "a generator's
                        // parameter list is bound exactly like an ordinary
                        // function's own") — `positional_arguments_for_
                        // method` already EXCLUDES `self` (the receiver is
                        // never a call argument, that function's own doc),
                        // so `self`'s own slot is prepended here with the
                        // RECEIVER value, the same binding `method_call_
                        // result` gives `self` for a non-generator method.
                        if is_generator_def(method) {
                            let mut generator_arguments = Vec::with_capacity(positional.len() + 1);
                            generator_arguments.push(receiver.clone());
                            generator_arguments.extend(positional.iter().cloned());
                            return match instances::generator_yields(method, &generator_arguments, environment.functions(), kernel, environment.call_depth()) {
                                Some(yields) => {
                                    let mut value = collection_models::list_literal_value(&yields);
                                    value.source = "generator".to_owned();
                                    value
                                }
                                None => unknown(),
                            };
                        }
                        return match instances::method_call_result(
                            &receiver,
                            model,
                            method,
                            &positional,
                            environment.functions(),
                            Some(classes),
                            kernel,
                            environment.call_depth(),
                        ) {
                            Some((_instance_after, result)) => result,
                            None => unknown(),
                        };
                    }
                }
            }
        }
    }
    // `functools.reduce(function, iterable[, initializer])` — the ONE
    // call this file folds CONCRETELY step by step, per-element,
    // because `function` is read as a RAW EXPRESSION (a `Lambda` or a
    // bare `Name` naming a same-module `def`) rather than an already-
    // evaluated value the way every other call argument in this file
    // is — see `reduce_expression_value`'s own doc.
    if let Expr::Name(name) = call.func.as_ref() {
        if name.id.as_str() == "reduce" && environment.read("reduce").is_none() {
            if let Some(value) = reduce_expression_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `map(function, iterable)` / `filter(predicate, iterable)` —
        // the two other builtins this file folds CONCRETELY over a RAW
        // callable expression rather than an already-evaluated value,
        // for the same reason `reduce` does (`map_expression_value`/
        // `filter_expression_value`'s own doc). Both return a LAZY
        // iterator (functions.html#map/#filter: "Return an iterator");
        // this domain has no separate iterator Kind, so the answer is
        // the eagerly-materialized `Kind::List` of the iterator's own
        // elements — the same choice `range_expression_value` already
        // makes for `range(...)`'s own lazy sequence, and the shape
        // `list(map(...))`/`list(filter(...))` needs once `list()`
        // (`builtin_models::list_constructor_call`) copies a known
        // `Kind::List` through unchanged.
        if name.id.as_str() == "map" && environment.read("map").is_none() {
            if let Some(value) = map_expression_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        if name.id.as_str() == "filter" && environment.read("filter").is_none() {
            if let Some(value) = filter_expression_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `Exception(message)` / `ValueError(message)` / `RuntimeError(message)`
        // / `TypeError(message)` — a BUILT-IN exception class constructor
        // call (never shadowed by a same-module def/class here, the same
        // `same_module_def_gate_open` gate this whole block is already
        // inside): tutorial/errors.rst §8.3, "the exception instance...
        // typically has an `args` attribute that stores the arguments."
        // Answered as a tagged `Kind::Object` (`exception_construction_value`'s
        // own doc) carrying every positional constructor argument, in
        // order, under one `args` field — `.args[0]` (this file's own
        // `evaluate_attribute_read`'s untagged-instance fallback,
        // `instances::field_read`) and `str(...)`
        // (`builtin_models::str_call`'s new exception row) both read
        // through this ONE construction. `ExceptionGroup(msg, excs)`
        // (PEP 654, `exceptions.rst`) is a DIFFERENT shape this file does
        // not decompose (the message and wrapped exceptions are never
        // read back through a refined sink in this corpus) — answered
        // OPAQUE instead of tagged, so any read through it (this
        // function's own return value, most directly) fires the opaque
        // law rather than silently building a hollow `args` shape nothing
        // reads.
        if is_builtin_exception_constructor(name.id.as_str()) && environment.read(name.id.as_str()).is_none() {
            if name.id.as_str() == "ExceptionGroup" {
                return opaque_value("an ExceptionGroup");
            }
            if !call.arguments.keywords.is_empty() {
                return unknown();
            }
            let Some(arguments) = splice_call_arguments(&call.arguments.args, environment, kernel) else {
                return unknown();
            };
            return exception_construction_value(&arguments);
        }
        // `bytes(...)`/`bytearray(...)`/`memoryview(...)` construction —
        // p-typed-array.py's own construction band. See
        // `bytes_like_construction_value`'s own doc for every recognized
        // argument shape; `None` there means "not one of those shapes,"
        // and this call falls through to the ordinary builtin dispatch
        // below (never a hard decline at this gate alone).
        if matches!(name.id.as_str(), "bytes" | "bytearray" | "memoryview") && environment.read(name.id.as_str()).is_none() {
            if !call.arguments.keywords.is_empty() {
                return unknown();
            }
            if let Some(value) = bytes_like_construction_value(name.id.as_str(), &call.arguments.args, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `datetime.datetime(year, month, day, hour=0, minute=0,
        // second=0, ..., tzinfo=...)` — recognized BEFORE the keyword
        // gate below because the fixture's own construction rows always
        // pass `tzinfo=` as a keyword argument. See
        // `datetime_construction_value`'s own doc for the exact fields
        // read and the aware-UTC-only scope.
    }
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        if is_datetime_datetime_attribute(attribute, environment) {
            if let Some(value) = datetime_construction_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `array.array(typecode, initializer)` — the Float64Array twin,
        // p-typed-array.py's `array_double_from_iterable`/`array_double_
        // write_and_read_back`. Recognized here (an Attribute call,
        // never a bare Name) the same way `datetime.datetime` is:
        // `array` imported as a bare module name (`import array`), not
        // locally shadowed.
        if attribute.attr.as_str() == "array" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "array" && environment.read("array").is_none() {
                    if let Some(value) = array_double_construction_value(call, environment, kernel) {
                        return value;
                    }
                    return unknown();
                }
            }
        }
    }
    if !call.arguments.keywords.is_empty() {
        return unknown();
    }
    let Some(arguments) = splice_call_arguments(&call.arguments.args, environment, kernel) else {
        return unknown();
    };
    match call.func.as_ref() {
        Expr::Name(name) if environment.read(name.id.as_str()).is_none() => {
            if name.id.as_str() == "len" {
                let [only] = arguments.as_slice() else { return unknown() };
                return match collection_models::len_result(only) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            if name.id.as_str() == "range" {
                return match range_expression_value(&arguments) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            if name.id.as_str() == "eval" {
                return match eval_literal_value(&arguments) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            match builtin_models::builtin_call_result(name.id.as_str(), &arguments) {
                Some(value) => value,
                None => unknown(),
            }
        }
        Expr::Attribute(attribute) => evaluate_attribute_call(attribute, &arguments, environment, kernel),
        _ => unknown(),
    }
}

/// Every positional call argument's value, in order, with a `Starred`
/// argument's own known-List elements spliced in place (the same
/// splicing `evaluate_display_elements` performs for a list/tuple/set
/// display — expressions.rst states one "unpacking" rule for both call
/// arguments and displays). `None` the moment a starred argument
/// evaluates to anything but a known `Kind::List` — an UNBOUNDED
/// iterable (an untyped/unknown-length parameter, for instance) has no
/// proven element count to splice, so the whole call declines rather
/// than guess at how many positional slots it fills.
fn splice_call_arguments(
    args: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::new();
    for arg in args {
        if let Expr::Starred(starred) = arg {
            let spread = evaluate_expression(&starred.value, environment, kernel);
            if spread.kind != Kind::List {
                return None;
            }
            values.extend(spread.items);
            continue;
        }
        values.push(evaluate_expression(arg, environment, kernel));
    }
    Some(values)
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
///
/// `def`'s KEYWORD-ONLY parameters (`*, age`) are appended to the
/// name list AFTER `posonlyargs`/`args`, in declaration order — a
/// bare positional call argument can never land on one of those
/// trailing slots (Python's own call-site grammar puts every
/// positional argument before every keyword argument, so
/// `call.arguments.args` never has enough entries to reach past
/// `posonlyargs`/`args`'s own count), so a kwonly name only ever
/// fills from `call.arguments.keywords`'s own position lookup below —
/// the same "the CALLER passed the keyword" reach the mission asks
/// for (`only_keyword(age=200)`, e-class-and-function.py's own
/// `keyword_only_call`). `summaries::bind_parameters` reads this same
/// combined `posonlyargs+args+kwonlyargs` order back apart at its own
/// boundary (that function's own doc).
fn positional_arguments_for_def(
    call: &ruff_python_ast::ExprCall,
    def: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let parameter_names: Vec<&str> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
        .map(|parameter| parameter.parameter.name.id.as_str())
        .collect();
    if def.parameters.kwarg.is_some() {
        return positional_arguments_with_kwargs_dict(call, &parameter_names, environment, kernel);
    }
    positional_arguments_by_names(call, &parameter_names, environment, kernel)
}

/// The same keyword→position mapping `positional_arguments_by_names`
/// gives an ordinary def, PLUS one trailing slot for a `**kwargs`
/// parameter — e-class-and-function.py's own `gather_kwargs(**fields:
/// int)`: "the call site's keyword arguments fill the dict." Every
/// keyword argument that names one of `parameter_names` (a plain or
/// keyword-only parameter) maps to its own position exactly as before;
/// every OTHER named keyword argument (one `**kwargs` would collect at
/// runtime, functions.rst's own `**identifier` row: "receives a
/// dictionary containing... keyword arguments") is instead gathered
/// into ONE dict, built the identical way an ordinary `{...}` literal
/// is (`collection_models::dict_literal_value` — string keys only,
/// this domain's own dict restriction), and appended as the FINAL slot
/// of the returned vector. `summaries::bind_parameters` reads that
/// final slot back and binds it to the `kwarg` parameter's own name
/// (that function's own kwonly-slot doc names the identical trailing-
/// slot convention for kwonly params; this is the same convention one
/// slot further out). A starred positional argument, or a `**spread`
/// keyword argument (`f(**other)` — no single name to attribute to the
/// dict), declines the whole call: this function only ever collects
/// NAMED keyword arguments into the dict, never an unbounded spread.
fn positional_arguments_with_kwargs_dict(
    call: &ruff_python_ast::ExprCall,
    parameter_names: &[&str],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
    let mut positional: Vec<Option<AbstractValue>> = vec![None; parameter_names.len().max(call.arguments.args.len())];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        positional[index] = Some(evaluate_expression(arg, environment, kernel));
    }
    let mut kwargs_keys: Vec<collection_models::DictKey> = Vec::new();
    let mut kwargs_values: Vec<AbstractValue> = Vec::new();
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            // `f(**other)` — an unbounded spread, no single name to
            // attribute into the collected dict
            return None;
        };
        let value = evaluate_expression(&keyword.value, environment, kernel);
        match parameter_names.iter().position(|name| *name == arg_name.as_str()) {
            Some(position) => positional[position] = Some(value),
            None => {
                kwargs_keys.push(collection_models::DictKey::string(arg_name.as_str()));
                kwargs_values.push(value);
            }
        }
    }
    while matches!(positional.last(), Some(None)) {
        positional.pop();
    }
    let mut filled: Vec<AbstractValue> = positional.into_iter().collect::<Option<Vec<_>>>()?;
    let keys: Vec<Option<collection_models::DictKey>> = kwargs_keys.into_iter().map(Some).collect();
    filled.push(collection_models::dict_literal_value(&keys, &kwargs_values));
    Some(filled)
}

/// A same-module METHOD call's positional argument values, keyed by the
/// method's own parameter names WITH `self` EXCLUDED — the receiver is
/// never a call argument, so `method.method_call_result`'s own
/// non-`self` parameter list is the keyword-mapping target, one name
/// per non-receiver argument the call actually supplies.
///
/// `@staticmethod` declares no `self`/receiver slot at all
/// (`instances::method_call_result`'s own doc) — EVERY declared
/// parameter is the keyword-mapping target then, none excluded. Every
/// other member `def` keeps the `self`-splitting shape.
fn positional_arguments_for_method(
    call: &ruff_python_ast::ExprCall,
    method: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let parameters: Vec<_> = method.parameters.posonlyargs.iter().chain(method.parameters.args.iter()).collect();
    let is_static = method
        .decorator_list
        .iter()
        .any(|decorator| matches!(&decorator.expression, Expr::Name(name) if name.id.as_str() == "staticmethod"));
    let rest: Vec<_> = if is_static {
        parameters
    } else {
        let (_self_parameter, rest) = parameters.split_first()?;
        rest.to_vec()
    };
    let parameter_names: Vec<&str> = rest.iter().map(|parameter| parameter.parameter.name.id.as_str()).collect();
    positional_arguments_by_names(call, &parameter_names, environment, kernel)
}

/// The shared keyword→position mapping both `positional_arguments_for_def`
/// and `positional_arguments_for_method` need: every positional call
/// argument evaluated in place against `parameter_names`, then every
/// keyword argument mapped to its own name's position. A starred
/// positional argument, a `**kwargs`-spread keyword, or a keyword naming
/// no parameter all decline the whole call.
fn positional_arguments_by_names(
    call: &ruff_python_ast::ExprCall,
    parameter_names: &[&str],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
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

/// Whether a known exact-string regex pattern contains NO metacharacter
/// `re.escape` would need to escape — library/re.html, `function::
/// escape(pattern)`: "Escape special characters in pattern... useful if
/// you want to match an arbitrary literal string that may have regular
/// expression metacharacters in it," and "The special characters are"
/// (re.html §"Regular Expression Syntax") `. ^ $ * + ? { } [ ] \ | ( )`.
/// A pattern containing none of those characters matches ITSELF and
/// only itself — `re.search`/`re.sub` over such a pattern reduce to a
/// plain substring test/replace, decidable without a regex engine
/// (`re_search_literal_value`/`re.sub`'s own call-site doc). `pattern`
/// must also be a known exact string; a non-string or unknown pattern
/// answers `false` (never metacharacter-free, since it is not even a
/// known literal).
fn is_literal_regex_pattern(pattern: &AbstractValue) -> bool {
    const REGEX_METACHARACTERS: &[char] = &['.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '\\', '|', '(', ')'];
    let Some(text) = exact_string_values(pattern).and_then(code_points_to_string) else {
        return false;
    };
    !text.chars().any(|c| REGEX_METACHARACTERS.contains(&c))
}

/// `re.search(pattern, subject)` reduced to a substring test — modeled
/// ONLY when `pattern` is a known exact string with no regex
/// metacharacter (`is_literal_regex_pattern`) and `subject` is a known
/// exact string (`evaluate_attribute_call`'s own `search` call site
/// doc). A pattern found IN the subject answers the match-object sort
/// (`opaque_value`, the same over-approximation `re.match` already
/// gives); an ABSENT pattern answers the exact `None` `re.search`
/// documents for "no position in the string matches the pattern."
fn re_search_literal_value(pattern: &AbstractValue, subject: &AbstractValue) -> Option<AbstractValue> {
    if !is_literal_regex_pattern(pattern) {
        return None;
    }
    let pattern_text = exact_string_values(pattern).and_then(code_points_to_string)?;
    let subject_text = exact_string_values(subject).and_then(code_points_to_string)?;
    if subject_text.contains(&pattern_text) {
        Some(opaque_value("a match object"))
    } else {
        Some(null_value())
    }
}

/// A JSON SCALAR literal's exact Python value — library/json.rst's own
/// JSON-to-Python conversion table (`evaluate_attribute_call`'s `loads`
/// call site doc): `null` -> `None`, `true`/`false` -> `True`/`False`,
/// a quoted string -> `str` (no escape-sequence decoding — this file
/// only reads the corpus's own escape-free literals), a bare integer
/// literal -> `int`, any other numeric spelling -> `float`. Only the
/// SCALAR productions are parsed — a `[`/`{`-leading text (an array or
/// object) declines, matching this file's own "the corpus's rows never
/// need array/object parsing" scope note.
fn json_scalar_literal_value(text: &str) -> Option<AbstractValue> {
    if text == "null" {
        return Some(null_value());
    }
    if text == "true" {
        return Some(known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved));
    }
    if text == "false" {
        return Some(known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved));
    }
    if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
        return Some(string_models::string_literal_value(&text[1..text.len() - 1]));
    }
    if let Ok(value) = text.parse::<i64>() {
        return Some(known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved));
    }
    if let Ok(value) = text.parse::<f64>() {
        return Some(known_values(vec![value], PrimitiveKind::Float, TrustProved));
    }
    None
}

/// `json.dumps(obj)`'s exact serialized text — library/json.rst's own
/// Python-to-JSON conversion table, default `separators = (', ', ':
/// ')` (`evaluate_attribute_call`'s `dumps` call site doc). Recurses
/// into a known `Kind::Object`'s own values (a nested dict); every
/// OTHER value shape this function cannot serialize (Float, a list, an
/// unknown value) makes the WHOLE call decline, matching this file's
/// "no partial answer" discipline for every other multi-part
/// composition (the f-string's own `has_exact` tier, `dict_literal_value`'s
/// own all-keys-must-parse rule). String quoting borrows Rust's own
/// `Debug` escaping (`format!("{:?}", text)`) rather than a hand-rolled
/// JSON-escape table — exact for the plain-ASCII, no-control-character
/// strings this corpus's own rows use; a string carrying a character
/// JSON and Rust's `Debug` escape differently (e.g. a lone surrogate,
/// or JSON's `\/` convention) is a known gap this file does not close.
fn json_dumps_value(value: &AbstractValue) -> Option<String> {
    if let Some(text) = exact_string_values(value).and_then(code_points_to_string) {
        return Some(format!("{:?}", text));
    }
    if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(value) {
        return Some(format!("{}", number as i64));
    }
    if value.kind == Kind::Object {
        let mut parts = Vec::with_capacity(value.keys.len());
        for entry in &value.keys {
            let serialized_value = json_dumps_value(&entry.value)?;
            parts.push(format!("{:?}: {}", entry.name, serialized_value));
        }
        return Some(format!("{{{}}}", parts.join(", ")));
    }
    None
}

/// `receiver.attr(...)` — the known receiver shapes this file
/// dispatches: `math.<name>(...)` / `re.compile(...)` (only when the
/// module name is not shadowed by a local binding) and a method call
/// on an evaluated receiver (an exact string's method, a dict's `.get`
/// or a view method, or a set method).
fn evaluate_attribute_call(
    attribute: &ruff_python_ast::ExprAttribute,
    arguments: &[AbstractValue],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if let Expr::Name(module_name) = attribute.value.as_ref() {
        if module_name.id.as_str() == "math" && environment.read("math").is_none() {
            return match math_models::math_call_result(attribute.attr.as_str(), arguments) {
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
            // `re.search(pattern, string)` — library/re.html, `function::
            // search(pattern, string, flags=0)`: "Scan through string
            // looking for the first location where the regular
            // expression pattern produces a match, and return a
            // corresponding Match. Return None if no position in the
            // string matches the pattern." Modeled ONLY when `pattern`
            // is a known exact string containing NO regex metacharacter
            // (`is_literal_regex_pattern`'s own doc — `re.escape`'s own
            // documented special-character set) and `string` (the
            // subject) is a known exact string: a metacharacter-free
            // pattern's own regex semantics reduce to a plain SUBSTRING
            // test, decidable without a regex engine. A found substring
            // answers the match-object sort (the same
            // over-approximation `re.match` above already gives — this
            // file cannot build a real Match object); an ABSENT
            // substring answers the exact `None` CPython's own
            // `search` returns on no match — `null_value()`, matching
            // `dict.get`'s own None-on-absent shape.
            if attribute.attr.as_str() == "search" {
                if let [pattern, subject] = arguments {
                    if let Some(value) = re_search_literal_value(pattern, subject) {
                        return value;
                    }
                }
                return unknown();
            }
            // `re.sub(pattern, repl, string)` — library/re.html,
            // `function:: sub(pattern, repl, string, count=0, flags=0)`:
            // "Return the string obtained by replacing the leftmost
            // non-overlapping occurrences of pattern in string by the
            // replacement repl... with no count, every match is
            // replaced" — AGENT-BRIEF.md's own confirmed fact, the twin
            // of JS's GLOBAL `.replace(/…/g)`. Modeled the same way
            // `search` is: `pattern` and `repl` must both be known exact
            // strings with `pattern` metacharacter-free (the same
            // `is_literal_regex_pattern` gate), so the whole call
            // reduces to `string.replace(pattern, repl)` —
            // `string_models::string_method_result`'s own `replace` row
            // already implements the every-occurrence semantics this
            // reduction needs.
            if attribute.attr.as_str() == "sub" {
                if let [pattern, repl, subject] = arguments {
                    if is_literal_regex_pattern(pattern) {
                        if let Some(value) = string_models::string_method_result("replace", subject, &[pattern.clone(), repl.clone()]) {
                            return value;
                        }
                    }
                }
                return unknown();
            }
        }
        // `json.loads(s)` — library/json.rst, `function:: loads(s, ...)`:
        // "deserialize s... to a Python object using this conversion
        // table" (the JSON-to-Python table this function's own doc
        // cites). Modeled ONLY for a known exact-string `s` whose text
        // is one of the JSON SCALAR productions this file parses by hand
        // (`json_scalar_literal_value`'s own doc: an integer, a float, a
        // quoted string, `true`/`false`/`null`) — the corpus's own rows
        // never need array/object parsing, so that grammar is not built.
        if module_name.id.as_str() == "json" && environment.read("json").is_none() {
            if attribute.attr.as_str() == "loads" {
                if let [text] = arguments {
                    if let Some(text) = exact_string_values(text).and_then(code_points_to_string) {
                        if let Some(value) = json_scalar_literal_value(text.trim()) {
                            return value;
                        }
                    }
                }
                return unknown();
            }
            // `json.dumps(obj)` — library/json.rst, `function::
            // dumps(obj, ...)`: "Serialize obj to a JSON formatted str
            // using this conversion table" (the Python-to-JSON table
            // this function's own doc cites), default `separators =
            // (', ', ': ')` (no `indent`). Modeled for a known exact
            // string, a known Integer, or a known `Kind::Object` whose
            // OWN values are each one of those same JSON-serializable
            // shapes (`json_dumps_value`'s own doc) — every other value
            // shape (Float, Boolean, a nested list, an unknown value)
            // declines the whole call.
            if attribute.attr.as_str() == "dumps" {
                if let [value] = arguments {
                    if let Some(text) = json_dumps_value(value) {
                        return string_models::string_literal_value(&text);
                    }
                }
                return unknown();
            }
        }
        // `importlib.import_module(name)` — library/importlib.html,
        // `function:: import_module(name, package=None)`: "Import a
        // module... and return the imported module." This domain has
        // no module-object Kind (the same "no dedicated kind" posture
        // `type(object)`'s own opaque answer takes,
        // `builtin_models::builtin_call_result`'s `"type"` row) — the
        // honest answer is the opaque "a module object" sort, never a
        // specific value: d-module-surface.py's own `dynamic_import`
        // row states exactly that reason ("a module object is never an
        // Age"). Modeled regardless of the argument shape (a dynamic
        // import's own module identity is never read further by this
        // corpus's rows — only that the RESULT is opaque, not a refined
        // scalar).
        if module_name.id.as_str() == "importlib" && environment.read("importlib").is_none() {
            if attribute.attr.as_str() == "import_module" {
                return opaque_value("a module object");
            }
        }
        // `types.MappingProxyType(d)` — library/stdtypes.rst, "Mapping
        // Types — dict": types.MappingProxyType "wraps" a dict in a
        // read-only VIEW; a READ through the proxy returns exactly the
        // wrapped dict's own value (this row only reads — WRITING
        // through the proxy raises `TypeError`, a genuine CPython
        // divergence from a JS `Object.freeze` wrapper this file does
        // not model since no row exercises a write). Answered as the
        // IDENTITY of its one known `Kind::Object` argument — the same
        // pass-through `builtin_models::cast_call` already gives
        // `typing.cast`'s second argument, reused here rather than
        // building a second "read-only dict" tag no reader would ever
        // distinguish from a plain dict (every subscript/`.get()` read
        // this file models is already non-mutating).
        if module_name.id.as_str() == "types" && environment.read("types").is_none() {
            if attribute.attr.as_str() == "MappingProxyType" {
                if let [dict] = arguments {
                    if dict.kind == Kind::Object {
                        return dict.clone();
                    }
                }
                return unknown();
            }
        }
        // `weakref.WeakSet()` / `weakref.WeakKeyDictionary()` — the
        // BARE, zero-argument constructor form only (library/weakref.rst:
        // both classes hold "weak references to its elements/keys," a
        // fact invisible to any reader that only ever consumes the
        // collection via containment/subscript, matching
        // `collection_models.rs`'s own "a set's uniqueness is invisible
        // to a len()/iteration reader" note for an ordinary `set`).
        // `WeakSet()` answers the same empty-list `Kind::List` a bare
        // `set()` does (`builtin_models::set_constructor_call`'s own
        // zero-argument row); `WeakKeyDictionary()` answers the same
        // empty-dict `Kind::Object` a bare `dict()` does. Neither
        // constructor call takes a required argument this file reads —
        // a call WITH an argument (copying from an existing mapping)
        // falls through to `unknown()`, not modeled.
        if module_name.id.as_str() == "weakref" && environment.read("weakref").is_none() {
            if attribute.attr.as_str() == "WeakSet" && arguments.is_empty() {
                return collection_models::list_literal_value(&[]);
            }
            if attribute.attr.as_str() == "WeakKeyDictionary" && arguments.is_empty() {
                return collection_models::dict_literal_value(&[], &[]);
            }
        }
        // `await asyncio.gather(a, b, ...)` — library/asyncio-task.rst,
        // `awaitablefunction:: gather(*aws, ...)`: "If all awaitables
        // are completed successfully, the result is an aggregate list
        // of returned values. The order of result values corresponds
        // to the order of awaitables." Each positional argument here is
        // already the settled value the caller's own `await`/call
        // evaluation produced (a same-module coroutine call summarizes
        // through the ordinary call dispatch above, `async`/`await`
        // carrying no gate of their own — `evaluate_expression`'s
        // `Expr::Await` arm passes its inner value straight through),
        // so this row only needs to collect the already-evaluated
        // arguments into the aggregate List `asyncio.gather` documents.
        // `return_exceptions=`/other keyword arguments are not modeled
        // (the call-site keyword guard above this function's own
        // caller already declines a call carrying any keyword
        // argument).
        if module_name.id.as_str() == "asyncio" && environment.read("asyncio").is_none() {
            if attribute.attr.as_str() == "gather" {
                return collection_models::list_literal_value(arguments);
            }
        }
    }
    // `datetime.datetime.now()` — a TWO-level attribute chain
    // (`Attribute(value=Attribute(value=Name("datetime"),
    // attr="datetime"), attr="now")`), never reaching
    // `is_datetime_datetime_attribute`'s own single-level check (that
    // check gates the CONSTRUCTION call, `datetime.datetime(...)`, whose
    // `call.func` IS the `datetime.datetime` chain itself — here
    // `attribute` is one level further out, `datetime.datetime.now`).
    // classmethod:: datetime.now(tz=None): "Return the current local
    // date and time." — a value that changes every run, never a whole
    // number Age could ever admit (this fixture's own row's reason:
    // "the current moment is not in the set"); answered OPAQUE, the
    // same "not a scalar/set this domain models" honesty every other
    // host-nondeterministic read in this file already carries. The
    // `tz=` argument (if any) is not read — every outcome is equally
    // opaque regardless of which timezone the caller requests.
    if let Expr::Attribute(inner) = attribute.value.as_ref() {
        if is_datetime_datetime_attribute(inner, environment) && attribute.attr.as_str() == "now" {
            return opaque_value("the current datetime");
        }
    }
    let receiver = evaluate_expression(&attribute.value, environment, kernel);
    // A tagged `datetime_datetime` instance's own METHODS —
    // `.timestamp()` (exact, aware-UTC-only, `datetime_timestamp_value`'s
    // own doc) and `.isoformat()` (opaque — datetime.rst's own format
    // composes up to 6 further digits/an offset this file does not
    // spell exactly, and no in-set leg through this sink exists in the
    // corpus this file serves, matching `math.pi`'s "sort-only is
    // enough" reasoning for a row with no in-set leg to prove exact).
    if receiver.kind == Kind::Object && receiver.source == "datetime_datetime" {
        if attribute.attr.as_str() == "timestamp" && arguments.is_empty() {
            return match datetime_timestamp_value(&receiver) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isoformat" {
            return opaque_value("an ISO 8601 datetime string");
        }
    }
    if exact_string_values(&receiver).is_some() {
        return match string_models::string_method_result(attribute.attr.as_str(), &receiver, arguments) {
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
        // dict.setdefault(key, default=None) READ as a VALUE (not
        // through the statement-level mutation sink,
        // `collection_models::mutated_receiver`'s own `setdefault`
        // arm): "If key is in the dictionary, return its value. If not,
        // insert key with a value of default and return default"
        // (stdtypes.rst, dict's method:: setdefault) — the VALUE half
        // of that contract is identical to `dict.get(key, default)`'s
        // own present-wins-over-default row, so this arm reuses
        // `dict_get_result` directly rather than re-derive the same
        // present/absent branch a second time. The receiver's own
        // write (extending it on a miss) is `mutated_receiver`'s
        // business when this call sits in a statement-level write
        // position; a nested read like this one only ever needs the
        // answered value.
        if attribute.attr.as_str() == "setdefault" {
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
    // `(<a known Float/Integer>).is_integer()` — stdtypes.rst, `method::
    // float.is_integer()`: "Return True if the float instance is finite
    // with integral value, and False otherwise" (`int.is_integer()`,
    // added 3.12, "Returns True" always — "Exists for duck type
    // compatibility"). Exact for any known single numeric receiver: an
    // Integer-sorted receiver is always `True` (the int row); a
    // Float-sorted receiver checks `fract() == 0.0 && is_finite()`
    // directly on the known f64.
    if attribute.attr.as_str() == "is_integer" && arguments.is_empty() {
        if let Some((value, sort)) = single_numeric_value(&receiver) {
            let is_integer = sort == PrimitiveKind::Integer || (value.is_finite() && value.fract() == 0.0);
            return boolean_answer(is_integer);
        }
    }
    if receiver.kind == Kind::List {
        if let Some(value) = set_method_result(attribute.attr.as_str(), &receiver, arguments) {
            return value;
        }
        // `list.pop()`/`list.pop(i)` READ as a VALUE (not through the
        // statement-level mutation sink, `collection_models::
        // mutated_receiver`'s own `pop` arm): "retrieves the item at *i*
        // and also removes it from *s*" (stdtypes.rst's Mutable-Sequence-
        // Types table, `s.pop([i])`) — c-reads-and-values.py's
        // `list_pop`'s own RHS shape, `overs.pop()` used directly as a
        // `return` expression rather than first bound to a name. Only the
        // RESULT half of `mutated_receiver`'s `(new receiver, result)`
        // pair is read here: the receiver's own shrink is the write
        // sink's business (`walk_mutating_call_statement`'s statement-
        // level rebind), the same "fires/writes belong to the sink" split
        // every other nested value read in this file already draws
        // (the construction and instance-method-call arms above,
        // `dict.setdefault`'s own value-read arm just above this one).
        if attribute.attr.as_str() == "pop" {
            if let Some((_new_receiver, result)) = collection_models::mutated_receiver("pop", &receiver, arguments) {
                return result;
            }
            return unknown();
        }
        // `xs.sort()` READ AS A VALUE (not through the statement-level
        // mutation sink): "This method sorts the list in place... This
        // method modifies the sequence in place for economy of space
        // when sorting a large sequence. To remind users that it
        // operates by side effect, it does not return the sorted
        // sequence" (stdtypes.rst's Mutable-Sequence-Types table,
        // `s.sort(...)`) — `None` ALWAYS, regardless of whether the
        // receiver's own elements are known (the trap this row exists
        // to name: reading the RETURN VALUE is always a sort mismatch
        // against a refined Age, never the sorted list itself). The
        // sorted LIST is `mutated_receiver`'s own business when this
        // call sits in a statement-level write position — this arm only
        // ever answers the call's own result, matching the "fires/writes
        // belong to the sink" split every other nested value read in
        // this file already draws.
        if attribute.attr.as_str() == "sort" && arguments.is_empty() {
            return null_value();
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
/// evaluates first; a known Object (`Kind::Object`) TAGGED with a class
/// (`source` non-empty, `judge_construction`'s own mark, found in
/// `environment.classes()`) reads through the MODEL — a stored field OR
/// a `@property` alias via `field_read_through_model`, or a bare
/// bound-method reference if the name is neither of those but IS a
/// class method (opaque). An UNTAGGED Object (a cross-module binding:
/// `cross_module.rs` builds a module object with the identical
/// `known_object` shape a class instance carries, this file's own
/// module doc note) falls back to the plain `instances::field_read`
/// linear scan — the same one dispatch arm this function used before
/// class tagging existed, still correct for a receiver with no class
/// to look up. Any other receiver shape (unknown, a scalar, a list)
/// answers `unknown()` — there is no attribute-read model for it here.
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
    // `<a datetime_datetime instance>.year` — datetime.rst,
    // `attribute:: datetime.year`: "Between MINYEAR and MAXYEAR
    // inclusive." Answered OPAQUE rather than reading the exact
    // constructor-argument field this file DOES track internally
    // (`datetime_construction_value`'s own `year` `ObjectKey`): the
    // calendar year a `.year` read reports depends on which timezone the
    // instance carries (a naive vs. aware datetime constructed from the
    // same wall-clock fields can report different years across a
    // day/year boundary), a fact this file's own `aware_utc` marker does
    // not fully resolve (an aware-but-non-UTC instance is already
    // declined at construction, but an aware-UTC instance's `.year`
    // still needs no further computation THIS file's corpus reads
    // through a refined sink) — this row's own fixture framing states
    // the general fact plainly: "not pinned to one calendar value by
    // CPython alone." A calendar year is never inside `Age`'s `[0,120]`
    // window regardless (this file's own `j-stdlib-surfaces.py` row has
    // no in-set leg), so the opaque answer still fires correctly through
    // the opaque law without this file overclaiming precision it does
    // not need for that verdict.
    if attribute.attr.as_str() == "year" {
        let receiver = evaluate_expression(&attribute.value, environment, kernel);
        if receiver.kind == Kind::Object && receiver.source == "datetime_datetime" {
            return opaque_value("a calendar year");
        }
    }
    // `math.pi`/`math.e`/`math.tau`/`math.inf` — an ATTRIBUTE READ on
    // the `math` module name (not shadowed by a local binding), routed
    // through `math_models::math_constant_value` (library/math.rst,
    // "Constants" — see that function's own doc for the sort-only
    // Float answer and the `math.nan` exclusion).
    if let Expr::Name(module_name) = attribute.value.as_ref() {
        if module_name.id.as_str() == "math" && environment.read("math").is_none() {
            if let Some(value) = math_models::math_constant_value(attribute.attr.as_str()) {
                return value;
            }
        }
    }
    // `super().<name>` READ, no call — functions.rst's `super()` entry:
    // "a typical superclass call looks like this: `super().method(arg)`."
    // The receiver `self` is bound to the CURRENT working instance
    // (`instances::method_call_result`'s own environment), and its
    // class's `parent_methods` (never `methods`, which a child override
    // has already replaced) is the map a bare `super().<name>` reference
    // resolves a bound-method name against — the same single-inheritance
    // MRO rule `method_call_result`'s own `super_resolver` reads for a
    // CALLED `super().<method>(...)`, applied here to the un-called
    // attribute reference (b-body-expressions.py's `SuperBareChild.years`
    // row).
    if is_bare_super_call(&attribute.value) {
        if let Some(instance) = environment.read("self") {
            if !instance.source.is_empty() {
                if let Some(classes) = environment.classes() {
                    if let Some(model) = classes.get(instance.source.as_str()) {
                        if model.parent_methods.contains_key(attribute.attr.as_str()) {
                            return opaque_value("a bare bound-method reference");
                        }
                    }
                }
            }
        }
        return unknown();
    }
    let receiver = evaluate_expression(&attribute.value, environment, kernel);
    if receiver.kind != Kind::Object {
        return unknown();
    }
    // A tagged instance (`source` non-empty, `judge_construction`'s own
    // mark) reads through the MODEL, not the bare `field_read` scan: a
    // `@property` name resolves to its backing field's value
    // (`field_read_through_model`'s own doc), and a name that is
    // neither a stored field NOR a property but IS one of the class's
    // own methods is a BARE bound-method reference — `person.next_year`
    // with no call parens names the method object itself, never a
    // program-tracked scalar (datamodel.rst, "Instance methods": "the
    // special thing about methods is that the instance object is
    // prepended to the argument list" — reading the method WITHOUT
    // calling it still answers that bound-method object), so this
    // answers opaque rather than the `unknown()` a plain `field_read`
    // miss would give (c-reads-and-values.py's
    // `read_type_member_method_skip`; `super().<method>`'s own bare-
    // reference row is handled above, before this receiver even
    // evaluates).
    if !receiver.source.is_empty() {
        if let Some(classes) = environment.classes() {
            if let Some(model) = classes.get(receiver.source.as_str()) {
                if let Some(value) = instances::field_read_through_model(model, &receiver, attribute.attr.as_str()) {
                    return value;
                }
                // A CLASS ATTRIBUTE (`ceiling = 40` at class-body top
                // level, read through `cls.ceiling`/`ClassName.ceiling`)
                // lives in the receiver's own `keys` (`instances::
                // class_object_value` builds it there) but never in
                // `model.fields`/`model.properties` — `field_read_
                // through_model` only reads instance-declared fields, so
                // it misses a class attribute by design, not by gap. The
                // plain linear scan still finds it directly off the
                // receiver's own stored value before falling to "this is
                // a bound method" or "unknown."
                if let Some(value) = instances::field_read(&receiver, attribute.attr.as_str()) {
                    return value;
                }
                if instances::method_def_of(model, attribute.attr.as_str()).is_some() {
                    return opaque_value("a bare bound-method reference");
                }
                return unknown();
            }
        }
    }
    match instances::field_read(&receiver, attribute.attr.as_str()) {
        Some(value) => value,
        None => unknown(),
    }
}

/// Whether `expr` is exactly a bare, no-argument `super()` call —
/// `instances.rs`'s own `super_init_call` recognizes the identical
/// shape for `super().__init__(...)`; this is the plain-`Expr::Call`
/// half of that same recognition, reused here for an un-called
/// `super().<name>` attribute reference.
fn is_bare_super_call(expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Name(name) = call.func.as_ref() else {
        return false;
    };
    name.id.as_str() == "super" && call.arguments.args.is_empty() && call.arguments.keywords.is_empty()
}

/// `[elt for target in iterable if cond ...]` / the same shape for a set
/// display and a generator expression — expressions.rst, "Displays for
/// lists, sets and dictionaries": "the comprehension consists of a
/// single expression, followed by at least one `for` clause." Modeled
/// ONLY the single-clause, known-List-iterable shape: exactly one
/// `Comprehension` (a second `for` clause, or an `async for` —
/// `is_async` — declines outright), the target a bare `Expr::Name` or a
/// two-name tuple target (`comprehension_target_names`'s own doc), and
/// the iterable a known `Kind::List` of already-known elements. Each
/// surviving element forks
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
/// single-clause/known-List-iterable restriction as
/// `evaluate_list_or_set_comp` (including its two-name tuple target
/// row, the shape a `{k: v for k, v in d.items()}` walk needs), with
/// the additional requirement that `key` evaluates to a known exact
/// String OR a known single Integer-sorted value at every surviving
/// element (this domain's dict literal accepts string and int keys,
/// `collection_models.rs`'s own documented `DictKey` restriction) —
/// any element whose key is neither of those two sorts makes the whole
/// comprehension unknown() rather than silently dropping that entry.
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
    let mut keys: Vec<Option<collection_models::DictKey>> = Vec::with_capacity(rows.len());
    let mut values: Vec<AbstractValue> = Vec::with_capacity(rows.len());
    for (key, value) in rows {
        // a string-sorted key value builds an ordinary string
        // DictKey; a single known Integer-sorted key value (the
        // comprehension's own mapped element, e.g. `{age: ... for age
        // in [15, 20]}`) builds an int DictKey — the same two key
        // sorts `dict_literal_value` accepts for a plain `{...}`
        // display (`collection_models.rs`'s own module doc). Any
        // other key shape (Float, Boolean, an unread value) declines
        // the whole comprehension, matching `dict_literal_value`'s
        // own "even one unsupported key" honesty.
        let dict_key = if let Some(text) = exact_string_values(&key).and_then(code_points_to_string) {
            collection_models::DictKey::string(&text)
        } else if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(&key) {
            collection_models::DictKey::integer(number as i64)
        } else {
            return unknown();
        };
        keys.push(Some(dict_key));
        values.push(value);
    }
    collection_models::dict_literal_value(&keys, &values)
}

/// The single-clause comprehension shape shared by every comprehension
/// form: exactly one `Comprehension` clause, synchronous, over a known
/// `Kind::List` iterable of already-known elements, with a target that
/// is either a bare `Expr::Name` (one name, bound to the WHOLE element)
/// or a two-element `Expr::Tuple`/`Expr::List` of bare names (bound to
/// a `[first, second]` 2-element `Kind::List` element — the exact shape
/// `.items()`'s own pair-lists build, `dict_view_method_result`'s own
/// doc; a `for k, v in d.items():`-style unpacking target,
/// expressions.rst's "Displays for lists, sets and dictionaries": a
/// comprehension's `for` clause follows the SAME target-list grammar an
/// ordinary `for` statement does). `None` for anything outside that
/// shape (multiple clauses, `async for`, a target of any other arity or
/// shape, an unknown/non-List iterable) — the honest decline every
/// comprehension form shares before either evaluates its own
/// element/key expression. The target names and the `if` conditions
/// both borrow from `generators` itself (`'a`), so a caller walking the
/// returned elements still has the clause's own filter list in hand
/// with no second destructure of `generators`.
fn comprehension_target_and_elements<'a>(
    generators: &'a [ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(Vec<&'a str>, &'a [Expr], Vec<AbstractValue>)> {
    let [clause] = generators else {
        return None;
    };
    if clause.is_async {
        return None;
    }
    let target_names = comprehension_target_names(&clause.target)?;
    let iterable = evaluate_expression(&clause.iter, environment, kernel);
    if iterable.kind != Kind::List {
        return None;
    }
    Some((target_names, &clause.ifs, iterable.items))
}

/// The bare names a comprehension `for` target binds: one name for a
/// plain `Expr::Name` target, or two names for a two-element
/// `Expr::Tuple`/`Expr::List` target of bare names (`for k, v in
/// ...`-style unpacking). `None` for any other target shape (more than
/// two names, a non-Name element, a nested/starred target) — this file
/// does not model general destructuring targets, only the plain and
/// two-name-tuple shapes a dict `.items()` walk needs.
fn comprehension_target_names(target: &Expr) -> Option<Vec<&str>> {
    match target {
        Expr::Name(name) => Some(vec![name.id.as_str()]),
        Expr::Tuple(tuple) => {
            let [Expr::Name(first), Expr::Name(second)] = tuple.elts.as_slice() else {
                return None;
            };
            Some(vec![first.id.as_str(), second.id.as_str()])
        }
        _ => None,
    }
}

/// Binds a comprehension target's names against one source element: a
/// single-name target binds the WHOLE element; a two-name target
/// requires the element to be a known 2-element `Kind::List` (a
/// `.items()` pair, per `comprehension_target_names`'s own doc) and
/// binds each name to its own slot. `false` if a two-name target meets
/// an element that is not that exact shape — the caller must treat that
/// as an undecidable element, not silently bind partial names.
fn bind_comprehension_target(fork: &mut Environment, target_names: &[&str], element: &AbstractValue) -> bool {
    match target_names {
        [name] => {
            fork.bind(name, element.clone());
            true
        }
        [first, second] => {
            if element.kind != Kind::List || element.items.len() != 2 {
                return false;
            }
            fork.bind(first, element.items[0].clone());
            fork.bind(second, element.items[1].clone());
            true
        }
        _ => false,
    }
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
    let (target_names, conditions, source_elements) =
        comprehension_target_and_elements(generators, environment, kernel)?;
    let mut out = Vec::new();
    for element in source_elements {
        let mut fork = environment.fork();
        if !bind_comprehension_target(&mut fork, &target_names, &element) {
            return None;
        }
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
    let (target_names, conditions, source_elements) =
        comprehension_target_and_elements(generators, environment, kernel)?;
    let mut out = Vec::new();
    for element in source_elements {
        let mut fork = environment.fork();
        if !bind_comprehension_target(&mut fork, &target_names, &element) {
            return None;
        }
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
/// cannot prove numeric, declines to the sequence row below (a
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
    let Some((left_value, left_sort)) = single_numeric_value(left) else {
        return sequence_binop_value(op, left, right);
    };
    let Some((right_value, right_sort)) = single_numeric_value(right) else {
        return sequence_binop_value(op, left, right);
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

/// The set operand a kernel arithmetic transfer can pose: a numeric-
/// sorted `Kind::Set` (`kind_tag` Integer or Float — a seeded
/// parameter's declared range, or a sort-only answer like
/// `float_sorted_unknown()`) reads as its own set, and a known single
/// numeric `Kind::Values` (`single_numeric_value`'s own shape) reads as
/// the one-element set `{v}` so a set-vs-known-value pair poses the
/// same two-set question a set-vs-set pair does. Returns the set
/// together with the PYTHON ARITHMETIC SORT it carries — the same
/// Integer/Float split `single_numeric_value` returns, `Boolean`/bare
/// `Number` normalized to `Integer`/`Float` the same conservative way
/// (AGENT-BRIEF.md's "unproven int reads as the float row"). `None` for
/// every other shape (String/Array-sorted, untagged Set, non-numeric
/// Values) — this is a decline, not a guess.
fn transferable_numeric_operand(value: &AbstractValue) -> Option<(RefinedSet, PrimitiveKind)> {
    if let Some((v, sort)) = single_numeric_value(value) {
        return Some((make_refined_set(vec![refined_sets::refinement_forms::one_of(&[v])]), sort));
    }
    if value.kind == Kind::Set {
        let sort = match value.kind_tag {
            Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
            Some(PrimitiveKind::Float) => PrimitiveKind::Float,
            Some(PrimitiveKind::Boolean) => PrimitiveKind::Integer,
            Some(PrimitiveKind::Number) => PrimitiveKind::Float,
            _ => return None,
        };
        return Some((value.set.clone(), sort));
    }
    None
}

/// The kernel `TransferQuestionOp` a Python operator lowers to, or
/// `None` when the operator's kernel row is ECMA-semantics and
/// diverges for Python operands — the same exclusion
/// `loops.rs::lower_counter_step_body`'s own doc states for its
/// Add/Sub-only step shape, extended here to the one further operator
/// this file can also state safely:
///
/// - `Add`/`Sub`/`Mult` lower. The kernel's `transferAdd`/`transferSub`/
///   `transferMul` (set_functions/transfer.lean) are pure IEEE-754
///   float addition/subtraction/multiplication on the operands' real
///   enclosures — no ECMA `ToNumber` coercion, no string/object
///   handling folded in. Python's `+`/`-`/`*` over int/float operands
///   compute the identical IEEE-754 operation once both sides are read
///   as the f64s this file already carries them as (CPython floats ARE
///   IEEE-754 doubles, and `arithmetic_result` already declines an
///   integer result outside the f64-exact 2^53 range rather than claim
///   an inexact one) — so these three rows are semantics-identical
///   between the two languages and safe to lower.
/// - `Div` (`/`), `FloorDiv` (`//`), `Mod` (`%`), and `Pow` (`**`) do
///   NOT lower. Python's `/` is always true division with no ECMA
///   twin-row to ask (the kernel's `Div`/`Rem` rows are ECMA `/`/`%`,
///   dividend-sign remainder); `%` takes the DIVISOR's sign in Python,
///   the opposite of ECMA's dividend-sign remainder
///   (AGENT-BRIEF.md, expressions.rst §6.7) — asking the kernel's `Rem`
///   row for a Python `%` would silently answer the wrong sign on a
///   mixed-sign pair; `//` floors toward negative infinity, which is
///   not one of the kernel's arithmetic transfer rows at all; `**`
///   has no kernel binary-arithmetic-transfer row in this family
///   (`Pow` in `TransferQuestionOp` is the pinned NaN/unknown/set
///   PowOperandWire shape math_transfer.go builds, a different
///   question shape from the plain two-`RefinedSet` rows this
///   function poses). This is the exact set `lower_counter_step_body`
///   already trusts (Add/Sub, "no Python/JS divergence") plus `Mult`,
///   which shares the same no-divergence property arithmetic addition
///   and subtraction do.
fn admitted_transfer_op(op: Operator) -> Option<refined_kernel::transfer_questions::TransferQuestionOp> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    match op {
        Operator::Add => Some(TransferQuestionOp::Add),
        Operator::Sub => Some(TransferQuestionOp::Sub),
        Operator::Mult => Some(TransferQuestionOp::Mul),
        _ => None,
    }
}

/// The SET path over `binary_arithmetic_value`'s own two-known-values
/// decline: when at least one operand carries a numeric SET (seeded
/// parameter range, or a sort-only set-unknown answer) rather than one
/// known single value, this asks the kernel's `transfer` for the
/// admitted operators (`admitted_transfer_op`) instead of losing the
/// determination to `unknown()`. Both operands must read as a numeric
/// set-or-known-value (`transferable_numeric_operand`); a non-numeric
/// or untagged-Set operand still declines. The kernel's own float
/// image (a certified enclosure, `TransferAnswerKind::Set`) or a pair
/// of Integer-marked singletons narrowing to one exact answer
/// (`TransferAnswerKind::Values`) both bind as `known_set`/
/// `known_values` at the WEAKER of the two operands' own trust grades
/// (`derived_trust_level` — the kernel's own answer can never overstate
/// past what its inputs were already trusted at). A kernel refusal
/// (a set shape `transfer` does not decide, e.g. the sequence/string
/// forms in the RefinedSet grammar) is caught exactly as
/// `assignability.rs`'s own containment ask catches one — refusal
/// reads as `None` here (the caller falls back to `unknown()`), never
/// a crash and never a guessed value.
fn transfer_over_sets(
    op: Operator,
    left: &AbstractValue,
    right: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    // gated to the case `binary_arithmetic_value`'s own known-values
    // path cannot already answer: AT LEAST ONE operand must be a
    // `Kind::Set` (a seeded range, or a sort-only set-unknown answer).
    // Two known single values stay on the existing pure-Rust path
    // unchanged — this function never re-derives a determination the
    // fast path already owns.
    if left.kind != Kind::Set && right.kind != Kind::Set {
        return None;
    }
    let transfer_op = admitted_transfer_op(op)?;
    let (left_set, left_sort) = transferable_numeric_operand(left)?;
    let (right_set, right_sort) = transferable_numeric_operand(right)?;
    // `/`'s always-float override has no bearing here (Div is not
    // admitted), so the same both_int rule binary_arithmetic_value's
    // known-values path uses applies unchanged: Integer only when
    // BOTH sides are Integer-sorted.
    let both_int = left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer;
    let result_sort = if both_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    let grade = refined_domain::trust_grades::derived_trust_level(
        refined_domain::trust_grades::TrustProved,
        &[left.clone(), right.clone()],
    );
    let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        (kernel.transfer)(&refined_kernel::transfer_questions::TransferQuestion {
            op: transfer_op,
            a: left_set,
            b: right_set,
            c: 0.0,
            base: refined_kernel::transfer_questions::PowOperandWire {
                kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
                set: make_refined_set(vec![]),
            },
            exp: refined_kernel::transfer_questions::PowOperandWire {
                kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
                set: make_refined_set(vec![]),
            },
        })
    }));
    let answer = asked.ok()?;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    match answer.kind {
        TransferAnswerKind::Values => {
            if both_int && answer.values.iter().any(|v| v.fract() != 0.0) {
                // an Integer-sorted pair whose kernel answer is not itself
                // integral cannot happen under Add/Sub/Mul (both are exact
                // on integer enclosures) — an honest decline rather than a
                // claim the sort rule would contradict, should the kernel
                // ever widen this row
                return None;
            }
            Some(known_values(answer.values, result_sort, grade))
        }
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(result_sort),
            ..known_set(answer.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// `binary_arithmetic_value` already falls through to
/// `sequence_binop_value` for a non-numeric operand pair (that
/// function's own doc — the same fallthrough the AugAssign callers
/// share), so a plain BinOp reads through the one shared entry point
/// too rather than re-run the same numeric-then-sequence dispatch a
/// second time.
fn evaluate_binop(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let left = evaluate_expression(&binop.left, environment, kernel);
    let right = evaluate_expression(&binop.right, environment, kernel);
    binary_arithmetic_value_with_kernel(binop.op, &left, &right, kernel)
}

/// `binary_arithmetic_value` WITH the kernel available: tries the SET
/// path first (`transfer_over_sets` — at least one operand a numeric
/// set, the admitted operators only), then falls through to
/// `binary_arithmetic_value` unchanged for everything else (two known
/// single values, or a non-numeric pair headed for
/// `sequence_binop_value`). Exported for `expressions.rs`'s OWN
/// BinOp evaluation (`evaluate_binop`) — the other call sites
/// (`loops.rs`, `summaries.rs`, `check.rs`'s AugAssign paths) still call
/// the plain `binary_arithmetic_value` today; wiring them onto this
/// function is a follow-on, not a behavior change this function's own
/// landing makes for them.
pub fn binary_arithmetic_value_with_kernel(
    op: Operator,
    left: &AbstractValue,
    right: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if let Some(result) = transfer_over_sets(op, left, right, kernel) {
        return result;
    }
    binary_arithmetic_value(op, left, right)
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
/// duplicate the four loops. Called from `binary_arithmetic_value`'s
/// OWN fallthrough the moment either operand is not a single known
/// numeric value — a numeric `+`/`*`/bitwise op never reaches here, since
/// `binary_arithmetic_value` answers those itself first.
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
    if let Some(detail) = known_string_index_out_of_range(&container, &index) {
        return Some((subscript.range(), format!("this expression provably raises {detail}")));
    }
    known_container_index_absent(&container, &index).map(|detail| {
        (
            subscript.range(),
            format!("this expression provably raises {detail}"),
        )
    })
}

/// Whether a KNOWN exact-string `container` provably has no code point at
/// a KNOWN Integer `index` — the string-receiver row `subscript_read`'s
/// own decline cannot distinguish from "not modeled": `s[i]` follows the
/// same negative-index-adjust-then-bounds-check rule an ordinary list
/// read follows (expressions.rst, "Subscriptions" — the same
/// `__getitem__` machinery every built-in sequence shares), and an
/// adjusted index still outside `0..len` raises `IndexError` exactly the
/// way a list's own out-of-range read does (`bytes_index`'s row, read
/// above, is this same check for a bytes-like `Kind::List` receiver —
/// this function is its string-shaped twin). A `Kind::List`/`Kind::Object`
/// container, or a non-Integer index, is not this function's row —
/// `None`.
fn known_string_index_out_of_range(container: &AbstractValue, index: &AbstractValue) -> Option<String> {
    let text = exact_string_values(container)?;
    let (value, sort) = single_numeric_value(index)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    let position = value as i64;
    let length = text.len() as i64;
    let adjusted = if position < 0 { position + length } else { position };
    if adjusted >= 0 && adjusted < length {
        None
    } else {
        Some("IndexError: string index out of range".to_owned())
    }
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
    // `f(*x)` where `x` is a genuinely unbounded list VALUE (`values:
    // list[int]`, b-body-expressions.py's own `wrapper_spread_call_
    // unbounded`) — CPython's own call path fails past a large
    // positional-argument count (`splice_call_arguments`'s own doc: an
    // unbounded iterable has no proven element count to splice into a
    // fixed argument vector, so the VALUE path declines rather than
    // guess). That silence is honest for the VALUE question, but the
    // SHAPE itself is a fact this checker already knows: an unpack
    // whose own length is unproven can throw at runtime regardless of
    // what the call computes. A KNOWN-length iterable (`Kind::List`,
    // whatever its own elements are known) never fires here —
    // `wrapper_spread_call`'s own `max(*[200, 201])` stays silent,
    // exactly as `splice_call_arguments` already treats it.
    //
    // EXCLUDED: `x` is this body's OWN `*args`/`**kwargs` parameter,
    // forwarded bare (`environment::is_variadic_parameter` — r-ast-
    // census.py's `wrapper(*args: P.args, **kwargs: P.kwargs): return
    // f(*args, **kwargs)`). A ParamSpec-captured vararg forward hands
    // CPython exactly the arguments THIS body itself received; it is
    // never an independently-grown collection whose length could
    // exceed what a real call already survived to reach this body, so
    // it never raises on this shape alone.
    for arg in &call.arguments.args {
        if let Expr::Starred(starred) = arg {
            if let Expr::Name(spread_name) = starred.value.as_ref() {
                if environment.is_variadic_parameter(spread_name.id.as_str()) {
                    continue;
                }
            }
            let spread = evaluate_expression(&starred.value, environment, kernel);
            if spread.kind != Kind::List {
                let callee_name = callee_display_name(call.func.as_ref());
                return Some((
                    call.range(),
                    format!(
                        "this expression provably raises TypeError: the list can hold any number of items, and the unpack hands each to {callee_name} as its own argument"
                    ),
                ));
            }
        }
    }
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

/// A callee expression's own plain name for a raise message — a bare
/// `Name` reads directly (`max`, `sorted`, …); an `Attribute` reads its
/// own trailing name (`obj.method` names `method`, the part CPython's
/// own `TypeError` messages name); anything else (a call result used
/// directly as a callee, for instance) falls back to a generic "the
/// call" rather than guess at a name that is not there.
fn callee_display_name(callee: &Expr) -> String {
    match callee {
        Expr::Name(name) => name.id.as_str().to_owned(),
        Expr::Attribute(attribute) => attribute.attr.as_str().to_owned(),
        _ => "the call".to_owned(),
    }
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
/// `eval(source)` — library/functions.rst's own `eval` entry: "The
/// *source* argument... is parsed and evaluated as a Python
/// expression." `eval` is a HOST BOUNDARY — its source string is
/// evaluated by CPython's own compiler/interpreter at runtime, a
/// dynamic capability this file does not model at all (general
/// parsing, name resolution, and evaluation of an arbitrary expression
/// are all out of scope, matching every other host-boundary row in
/// this file, e.g. `re.match`'s own opaque "a match object" answer).
/// The ENTIRE surface modeled is "a single known-string argument that
/// SYNTACTICALLY reads as a plain int/float literal spelling states
/// that literal's SORT" — never the exact value: even though
/// `eval("40")` is execution-verified to answer the exact int `40`,
/// answering the exact value here would mean this file is quietly
/// interpreting Python source text, which is the general-evaluation
/// capability this file explicitly declines everywhere else. A
/// sort-only answer (the whole-number set for an int-literal spelling,
/// `float_sorted_unknown()` for a float-literal spelling) keeps `eval`
/// honestly in the same "claims a sort, never a value" tier as
/// `math`'s approximated family and a same-module call's declined-body
/// return-annotation fallback (`summaries::return_sort_fallback`) —
/// graded `TrustSpec`, a claim about what KIND of literal the source
/// spells, never a proved fact about the value `eval` would actually
/// produce. Any other spelling (an expression, a call, a name, an
/// operator, a string this file cannot read as a plain literal) still
/// declines outright — `eval` on arbitrary source is never modeled
/// beyond these two literal-SORT rows.
fn eval_literal_value(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let text = exact_string_values(only).and_then(code_points_to_string)?;
    let trimmed = text.trim();
    if is_valid_base_ten_int_string(trimmed) {
        return Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(eval_whole_integers(), None, TrustSpec, SetKindTag::None)
        });
    }
    // a plain float literal spelling: an optional sign, decimal digits,
    // exactly one '.', decimal digits — no exponent/underscore/inf/nan
    // spelling is read (none of those are exercised by any row this
    // file serves, and each would need its own citation)
    let digits_and_sign = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    let is_plain_float_spelling = digits_and_sign.contains('.')
        && digits_and_sign.chars().all(|c| c.is_ascii_digit() || c == '.')
        && digits_and_sign.matches('.').count() == 1;
    if is_plain_float_spelling {
        return Some(float_sorted_unknown());
    }
    None
}

/// The unbounded whole-number set `eval_literal_value`'s int-literal row
/// answers — the identical shape `summaries::whole_integers` builds
/// (`refinement_forms::integer()` conjoined with the unbounded ray),
/// repeated here rather than reaching into `summaries.rs` for one
/// two-line helper (this file is the one every other call-result row
/// already lives in, and `summaries.rs` has no dependency edge back
/// into `expressions.rs` for this single shape).
fn eval_whole_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

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

    /// `lambda: 40` read as a VALUE answers opaque — "a function value,"
    /// never a specific scalar.
    #[test]
    fn test_lambda_as_a_value_is_opaque() {
        let Some(value) = eval("lambda: 40") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("a function value"));
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

    /// `binary_arithmetic_value` on two known STRINGS falls through to
    /// string concatenation — the row `label += "c"`-style AugAssign
    /// calls depend on, matching the equivalent `label = label + "c"`
    /// BinOp exactly.
    #[test]
    fn test_binary_arithmetic_value_falls_through_to_string_concat() {
        let a = string_models::string_literal_value("ab");
        let b = string_models::string_literal_value("c");
        let result = binary_arithmetic_value(Operator::Add, &a, &b);
        assert_eq!(exact_string_values(&result).and_then(code_points_to_string).as_deref(), Some("abc"));
    }

    /// `age + 1` where `age` is a seeded int-sorted SET `[0, 120]` — the
    /// mission's own headline case: the known-values path declines (age
    /// is not one known value), so `binary_arithmetic_value_with_kernel`
    /// takes the SET path and the kernel's `transferAdd` answers the
    /// certified enclosure `[1, 121]`, Integer-sorted (both operands
    /// Integer). Asserted via `scalar_subset` both directions so the
    /// answer set is pinned exactly, not merely "some Set."
    #[test]
    fn test_set_plus_known_int_lowers_through_kernel_transfer() {
        let Some(kernel) = loaded_kernel() else { return };
        let age = known_set(
            make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
        let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Add, &age, &one, &kernel);
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
        let want = make_refined_set(vec![integer(), at_least(1.0), refined_sets::refinement_forms::at_most(121.0)]);
        assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
        assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
    }

    /// The UNBOUNDED float-set row: `float_sorted_unknown() * 2` — the
    /// operand is the whole numeric line (math.sqrt's sort-only shape),
    /// and the kernel's transfer answers no tighter certified image for
    /// an unbounded operand, so the ask decides nothing and the honest
    /// answer stays unknown(). The BOUNDED-set row above is where the
    /// transfer certifies; this row pins that an unbounded operand is
    /// never guessed at.
    #[test]
    fn test_float_sorted_set_times_known_int_stays_unknown_when_unbounded() {
        let Some(kernel) = loaded_kernel() else { return };
        let sqrt_result = float_sorted_unknown();
        let two = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Mult, &sqrt_result, &two, &kernel);
        assert_eq!(result.kind, Kind::Unknown);
    }

    /// The EXCLUDED-operator row: `age % 7` where `age` is a seeded set —
    /// `%` is never admitted (Python's divisor-sign remainder diverges
    /// from the kernel's ECMA dividend-sign `Rem` row), so the set path
    /// declines outright and `binary_arithmetic_value_with_kernel` falls
    /// through to the ordinary known-values path, which also declines
    /// (a Set is not one known value) — the whole call answers
    /// `unknown()`, never a wrong-signed guess.
    #[test]
    fn test_excluded_operator_mod_over_a_set_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let age = known_set(
            make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
        let seven = known_values(vec![7.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Mod, &age, &seven, &kernel);
        assert_eq!(result.kind, Kind::Unknown);
    }

    /// Two known single values over an admitted operator (`+`) still
    /// take the ORIGINAL fast path, never the kernel round-trip — the
    /// set-gate in `transfer_over_sets` declines outright the moment
    /// neither operand is `Kind::Set`, so this stays the pure-Rust
    /// answer `test_binary_arithmetic_value_mixed_sort_widens_to_float`
    /// already pins, unchanged by this wave.
    #[test]
    fn test_two_known_values_skip_the_kernel_set_path() {
        let Some(kernel) = loaded_kernel() else { return };
        let ten_int = known_values(vec![10.0], PrimitiveKind::Integer, TrustProved);
        let half_float = known_values(vec![0.5], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Add, &ten_int, &half_float, &kernel);
        assert_eq!(result.kind, Kind::Values);
        assert_eq!(result.values, vec![10.5]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }

    /// A refusal the kernel `transfer` closure panics on (an untagged
    /// Set — string-sorted by convention, ORIENTATION.md's own
    /// recognition-slice fact) is CAUGHT by `transfer_over_sets`'
    /// `catch_unwind` and answered as a decline, never a crash —
    /// `transferable_numeric_operand` itself already declines an
    /// untagged Set before any kernel ask, so this exercises that
    /// decline path rather than a live kernel panic; the two are
    /// observationally the same "falls back to unknown()" outcome the
    /// mission asks for.
    #[test]
    fn test_untagged_set_declines_before_any_kernel_ask() {
        let Some(kernel) = loaded_kernel() else { return };
        let untagged = known_set(strings(), None, TrustProved, SetKindTag::None);
        let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Add, &untagged, &one, &kernel);
        assert_eq!(result.kind, Kind::Unknown);
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

    /// `[*xs, 30]` splices a known list's own elements in place, in
    /// order (expressions.rst, "List displays").
    #[test]
    fn test_list_display_starred_element_splices_a_known_list() {
        let Some(value) = eval("[*[200, 201], 30]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![201.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![30.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    #[test]
    fn test_list_display_starred_unknown_element_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("[*xs, 30]").expect("test source must parse");
        let environment = empty_environment();
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_dict_literal_and_subscript_read() {
        let Some(value) = eval("{\"a\": 1, \"b\": 2}[\"b\"]") else { return };
        assert_eq!(value.values, vec![2.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `{**base, "age": 41}` splices a known dict's own entries, then a
    /// later ordinary key overwrites the spread's same-named entry —
    /// last-value-wins, matching `dict_literal_value`'s own overwrite
    /// rule.
    #[test]
    fn test_dict_display_double_star_spread_merges_and_later_keys_win() {
        let Some(value) = eval("{**{\"age\": 40, \"name\": \"ann\"}, \"age\": 41}") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.keys.len(), 2);
        let age = value.keys.iter().find(|entry| entry.name == "age").expect("age present");
        assert_eq!(age.value.values, vec![41.0]);
    }

    /// `{**a, **b}` — a LATER spread's same-named key wins over an
    /// earlier spread's.
    #[test]
    fn test_dict_display_two_spreads_later_wins() {
        let Some(value) = eval("{**{\"age\": 40}, **{\"age\": 200}}") else { return };
        assert_eq!(value.keys.len(), 1);
        assert_eq!(value.keys[0].value.values, vec![200.0]);
    }

    /// `dict.setdefault(key, default)` read as a VALUE: a PRESENT key
    /// answers its own value, winning over the default argument.
    #[test]
    fn test_dict_setdefault_present_key_wins_over_the_default() {
        let Some(value) = eval("{\"bea\": 200}.setdefault(\"bea\", 40)") else { return };
        assert_eq!(value.values, vec![200.0]);
    }

    #[test]
    fn test_dict_setdefault_absent_key_answers_the_default() {
        let Some(value) = eval("{\"ann\": 40}.setdefault(\"bea\", 0)") else { return };
        assert_eq!(value.values, vec![0.0]);
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
    fn test_range_one_argument_materializes_stop_exclusive() {
        let Some(value) = eval("range(3)") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![0.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    #[test]
    fn test_range_two_arguments_start_stop() {
        let Some(value) = eval("range(2, 5)") else { return };
        assert_eq!(
            value.items,
            vec![
                known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![4.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    #[test]
    fn test_range_len_over_200() {
        // c-reads-and-values.py's dict_size row: {str(i): i for i in
        // range(200)} — the length is exactly 200
        let Some(value) = eval("len(range(200))") else { return };
        assert_eq!(value.values, vec![200.0]);
    }

    #[test]
    fn test_range_zero_step_declines() {
        let Some(value) = eval("range(0, 10, 0)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `reduce(lambda acc, age: acc + age, [100, 101], 0)` folds
    /// concretely: 0 + 100 + 101 == 201.
    #[test]
    fn test_reduce_with_lambda_and_seed_folds_concretely() {
        let Some(value) = eval("reduce(lambda acc, age: acc + age, [100, 101], 0)") else { return };
        assert_eq!(value.values, vec![201.0]);
    }

    /// `reduce` with no `initializer` on a NON-empty iterable seeds the
    /// accumulator with the FIRST element (functools.rst's own row).
    #[test]
    fn test_reduce_without_initializer_seeds_from_the_first_element() {
        let Some(value) = eval("reduce(lambda acc, age: acc + age, [10, 20, 30])") else { return };
        assert_eq!(value.values, vec![60.0]);
    }

    /// `reduce`'s `function` argument resolving to a same-module `def`
    /// (not only a lambda) folds through `summaries::call_result`.
    #[test]
    fn test_reduce_with_same_module_def_function() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def add(acc, age):\n    return acc + age\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("reduce(add, [10, 20], 0)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![30.0]);
    }

    /// `reduce` over a non-List iterable declines.
    #[test]
    fn test_reduce_non_list_iterable_declines() {
        let Some(value) = eval("reduce(lambda acc, age: acc + age, 5, 0)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `eval("40")` is execution-verified to answer the exact int 40
    /// (`eval("40") == 40`, `type(eval("40")) is int`), but `eval` is a
    /// host boundary this file never interprets: the answer is the
    /// whole-number SET (sort-only), never the exact value — the same
    /// posture `math.sqrt`'s approximated family and a declined
    /// same-module call's return-annotation fallback both take.
    #[test]
    fn test_eval_of_a_plain_int_literal_string_answers_the_whole_number_set() {
        let Some(value) = eval("eval(\"40\")") else { return };
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `eval("3.5")` answers `float_sorted_unknown()`, never the exact
    /// float — the same sort-only posture as the int-literal row above.
    #[test]
    fn test_eval_of_a_plain_float_literal_string_answers_float_sorted_unknown() {
        let Some(value) = eval("eval(\"3.5\")") else { return };
        assert_eq!(value, float_sorted_unknown());
    }

    /// `eval("-7")` still recognizes the leading-sign int spelling and
    /// answers the whole-number set (never the exact -7).
    #[test]
    fn test_eval_of_a_negative_int_literal_string_answers_the_whole_number_set() {
        let Some(value) = eval("eval(\"-7\")") else { return };
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// The whole-number set `eval` answers genuinely admits a value the
    /// Age alias refuses (200, 121, negatives, …) — the CONTAINMENT
    /// question the corpus's `call_eval_bare` row leans on.
    #[test]
    fn test_eval_whole_number_set_is_not_a_subset_of_a_bounded_int_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let Some(value) = eval("eval(\"40\")") else { return };
        let age_window = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(!(kernel.scalar_subset)(&value.set, &age_window));
    }

    /// `eval` on anything past a plain int/float literal string
    /// declines — general expression evaluation is never modeled.
    #[test]
    fn test_eval_of_a_non_literal_expression_declines() {
        let Some(value) = eval("eval(\"1 + 1\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    #[test]
    fn test_abs_call() {
        let Some(value) = eval("abs(-7)") else { return };
        assert_eq!(value.values, vec![7.0]);
    }

    /// `max(*[200, 201])` — a starred CALL argument over a known list
    /// splices its elements into the positional arguments before
    /// dispatch, the same way a starred list-display element does.
    #[test]
    fn test_starred_call_argument_splices_a_known_list() {
        let Some(value) = eval("max(*[200, 201])") else { return };
        assert_eq!(value.values, vec![201.0]);
    }

    /// A starred call argument over an UNBOUND name (no proven element
    /// count) declines the whole call rather than guess how many
    /// positional slots it fills.
    #[test]
    fn test_starred_call_argument_unknown_iterable_declines() {
        let Some(value) = eval("max(*values)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
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

    /// `math.pi` is an attribute read, not a call — the sort-only Float
    /// set `math_models::math_constant_value` answers (library/math.rst,
    /// "Constants").
    #[test]
    fn test_math_pi_attribute_read() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("math.pi").expect("test source must parse");
        let environment = empty_environment();
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
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

    /// A name bound to an opaque LAMBDA value still reaches a
    /// same-module `def` of the same name — the gate widening
    /// `same_module_def_gate_open` states: a lambda binding carries no
    /// scalar/collection value of its own to shadow the def dispatch
    /// with.
    #[test]
    fn test_lambda_bound_name_still_reaches_a_same_module_def_of_the_same_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def double(x):\n    return x + x\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        environment.bind("double", opaque_value("a function value"));
        let parsed = parse_expression("double(3)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![6.0]);
    }

    /// An ORDINARY value binding (not a lambda) still blocks the
    /// same-module-def dispatch — the gate only widens for the opaque
    /// lambda shape, matching the def-shadowing-a-builtin test's own
    /// "a real bound value wins" posture.
    #[test]
    fn test_ordinary_bound_value_still_blocks_the_same_module_def_dispatch() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def double(x):\n    return x + x\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        environment.bind("double", known_values(vec![9.0], PrimitiveKind::Integer, TrustProved));
        let parsed = parse_expression("double(3)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Unknown, "a bound Integer is not callable, and shadows the def dispatch");
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

    // --- generators: a same-module generator def's call ---

    /// `over_ages()` where `over_ages`'s body is straight-line
    /// `yield`s — the CALL answers the ordered List of yields, tagged
    /// `source == "generator"`, never routing through
    /// `summaries::call_result` (which has no `yield` row).
    #[test]
    fn test_generator_def_call_answers_the_ordered_yield_list() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "def over_ages():\n",
            "    yield 200\n",
            "    yield 40\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("over_ages()").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.source.as_str(), "generator");
        assert_eq!(
            value.items,
            vec![
                known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    /// `is_generator_def` routing test — a-statements.py's own `stream()`
    /// shape: a generator whose body is a single `for` loop with the
    /// yield ONE LEVEL inside it (`for value in (10, 20, 30): yield
    /// value`), no top-level `yield` statement at all. Before this
    /// wave's recursion fix, `is_generator_def` never saw the nested
    /// yield and the call would have fallen through to the ORDINARY
    /// `summaries::call_result` path instead (which has no `yield` row
    /// and would decline outright the same way). This test proves the
    /// call now reaches the GENERATOR dispatch: `instances::
    /// generator_yields` does not yet read a `Stmt::For` body (that
    /// extension is a separate owner's work, tracked in this file's own
    /// report), so the call still answers `unknown()` today — but it
    /// answers `unknown()` via `generator_yields`'s own honest decline,
    /// not via `summaries::call_result`'s. Once `generator_yields` gains
    /// the `Stmt::For` reading, this same call site starts answering the
    /// ordered yield list with no further change here.
    #[test]
    fn test_loop_bodied_generator_is_recognized_as_generator_shaped() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "def stream():\n",
            "    for value in (10, 20, 30):\n",
            "        yield value\n",
        ))
        .expect("test module parses")
        .into_syntax();
        assert!(
            is_generator_def(
                module
                    .body
                    .first()
                    .expect("one top-level statement")
                    .as_function_def_stmt()
                    .expect("is a def")
            ),
            "a yield one level inside a for-loop body is generator-shaped"
        );
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("stream()").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        // generator_yields reads the single-for-loop yield shape, so
        // the call answers the ordered yields as the generator's own
        // list-shaped value
        assert_eq!(value.kind, Kind::List);
        let elements: Vec<f64> = value.items.iter().map(|item| item.values[0]).collect();
        assert_eq!(elements, vec![10.0, 20.0, 30.0]);
    }

    /// `next(over_ages())` — the first yielded value, per `next_call`'s
    /// own generator row.
    #[test]
    fn test_next_of_a_generator_call_answers_the_first_yield() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "def over_ages():\n",
            "    yield 200\n",
            "    yield 40\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("next(over_ages())").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![200.0]);
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

    // --- method dispatch (value side) ---

    fn person_next_year_module() -> ruff_python_ast::ModModule {
        ruff_python_parser::parse_module(concat!(
            "class Person:\n",
            "    def __init__(self, age):\n",
            "        self.age = age\n",
            "    def next_year(self, bump=1):\n",
            "        return self.age + bump\n",
        ))
        .expect("test module parses")
        .into_syntax()
    }

    fn environment_with_person_classes(kernel: &Arc<RefinedTSKernel>) -> Environment {
        let module = person_next_year_module();
        let aliases = std::collections::HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::refinedpy::instances::class_table(&module, &aliases, &imports, kernel));
        let mut environment = empty_environment();
        environment.set_classes(classes);
        environment
    }

    /// `person.next_year(40)`-shaped positional call — resolves through
    /// `method_def_of`/`method_call_result`, answering the RESULT value.
    #[test]
    fn test_method_call_positional_answers_the_result_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = environment_with_person_classes(&kernel);
        let constructed = parse_expression("Person(40)").expect("test source must parse");
        let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
        environment.bind("person", instance);
        let call = parse_expression("person.next_year(1)").expect("test source must parse");
        let value = evaluate_expression(&call.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![41.0]);
    }

    /// A method call with a KEYWORD argument maps to position the same
    /// way a plain `def` call does.
    #[test]
    fn test_method_call_keyword_argument_maps_to_position() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = environment_with_person_classes(&kernel);
        let constructed = parse_expression("Person(40)").expect("test source must parse");
        let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
        environment.bind("person", instance);
        let call = parse_expression("person.next_year(bump=2)").expect("test source must parse");
        let value = evaluate_expression(&call.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![42.0]);
    }

    /// A method call's own receiver MUTATION is not threaded back into
    /// the environment from a nested expression read — only the result
    /// is answered here (the mutation half is check.rs's statement-sink
    /// business).
    #[test]
    fn test_method_call_does_not_thread_the_mutated_receiver_back() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "class Counter:\n",
            "    def __init__(self):\n",
            "        self.count = 0\n",
            "    def bump(self):\n",
            "        self.count = self.count + 1\n",
            "        return self.count\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let aliases = std::collections::HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::refinedpy::instances::class_table(&module, &aliases, &imports, &kernel));
        let mut environment = empty_environment();
        environment.set_classes(classes);
        let constructed = parse_expression("Counter()").expect("test source must parse");
        let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
        environment.bind("counter", instance);
        let call = parse_expression("counter.bump()").expect("test source must parse");
        let value = evaluate_expression(&call.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![1.0], "the call answers the result value");
        // the environment's own `counter` binding is UNCHANGED — a
        // nested expression read never writes the mutated instance back
        let still_bound = environment.read("counter").expect("counter remains bound");
        assert_eq!(
            crate::refinedpy::instances::field_read(still_bound, "count"),
            Some(known_values(vec![0.0], PrimitiveKind::Integer, TrustProved))
        );
    }

    /// `@property` read on an instance resolves through
    /// `field_read_through_model` — the alias's backing value, not a
    /// bound-method opaque.
    #[test]
    fn test_property_read_resolves_through_the_model() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "class Person:\n",
            "    def __init__(self, age):\n",
            "        self._age = age\n",
            "    @property\n",
            "    def age(self):\n",
            "        return self._age\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let aliases = std::collections::HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::refinedpy::instances::class_table(&module, &aliases, &imports, &kernel));
        let mut environment = empty_environment();
        environment.set_classes(classes);
        let constructed = parse_expression("Person(40)").expect("test source must parse");
        let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
        environment.bind("person", instance);
        let read = parse_expression("person.age").expect("test source must parse");
        let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![40.0]);
    }

    /// A plain Attribute READ naming a METHOD (no call parens) answers
    /// opaque — "a bare bound-method reference," never the method
    /// object's own scalar-shaped havoc.
    #[test]
    fn test_bare_bound_method_reference_is_opaque() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = environment_with_person_classes(&kernel);
        let constructed = parse_expression("Person(40)").expect("test source must parse");
        let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
        environment.bind("person", instance);
        let read = parse_expression("person.next_year").expect("test source must parse");
        let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("a bare bound-method reference"));
    }

    /// `super().years` — a bare (un-called) reference to a PARENT
    /// method, read from inside a child method's own body: resolves
    /// through `self`'s class's `parent_methods`, answering opaque.
    #[test]
    fn test_super_bare_bound_method_reference_is_opaque() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "class Base:\n",
            "    def years(self):\n",
            "        return 40\n",
            "class Child(Base):\n",
            "    def years(self):\n",
            "        return super().years\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let aliases = std::collections::HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::refinedpy::instances::class_table(&module, &aliases, &imports, &kernel));
        let child = classes.get("Child").expect("Child class recorded");
        let constructed_child = crate::refinedpy::instances::judge_construction(child, &[], &[], &kernel).instance;
        let mut environment = empty_environment();
        environment.set_classes(classes.clone());
        environment.bind("self", constructed_child);
        let read = parse_expression("super().years").expect("test source must parse");
        let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("a bare bound-method reference"));
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
        // str(x) IS a modeled builtin call for a known Integer argument
        // (builtin_models::str_call — CPython's plain decimal spelling),
        // so the key expression is the known string "1" and the whole
        // comprehension builds a known dict, matching CPython's own
        // `{str(x): x for x in [1]}` == `{'1': 1}`
        assert_eq!(value.kind, Kind::Object);
    }

    #[test]
    fn test_multiple_generator_clauses_decline() {
        let Some(value) = eval("[x for x in [1, 2] for y in [3, 4]]") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `{name: age for name, age in d.items()}` — a two-name tuple
    /// target unpacking each `.items()` pair-list; the whole
    /// comprehension re-builds the same dict.
    #[test]
    fn test_dict_comp_two_name_tuple_target_over_items() {
        let Some(value) = eval("{name: age for name, age in {\"ann\": 40, \"bea\": 41}.items()}") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.keys.len(), 2);
    }

    /// A list comprehension can ALSO use a two-name tuple target —
    /// `[age for name, age in d.items()]` reads only the value half.
    #[test]
    fn test_list_comp_two_name_tuple_target_over_items() {
        let Some(value) = eval("[age for name, age in {\"ann\": 40}.items()]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items, vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)]);
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

    /// `await asyncio.gather(a, b)` answers the aggregate List of the
    /// already-evaluated argument values, in call order.
    #[test]
    fn test_asyncio_gather_awaited_answers_the_aggregate_list() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("await asyncio.gather(1, 2)").expect("test source must parse");
        let environment = empty_environment();
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![1.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![2.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
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
    fn test_provable_raise_string_out_of_range_subscript() {
        let Some(found) = provable_raise_of("\"banana\"[99]") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("an out-of-range string subscript must provably raise");
        };
        assert!(found.1.contains("IndexError"), "{}", found.1);
        assert!(found.1.contains("string index out of range"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_string_in_range_subscript_declines() {
        assert!(provable_raise_of("\"banana\"[0]").is_none());
        // a negative in-range index must not false-positive raise
        assert!(provable_raise_of("\"banana\"[-1]").is_none());
    }

    #[test]
    fn test_provable_raise_none_case() {
        assert!(provable_raise_of("1 + 2").is_none());
        assert!(provable_raise_of("[1, 2][0]").is_none());
        assert!(provable_raise_of("1 / 2").is_none());
    }

    #[test]
    fn test_provable_raise_math_sqrt_of_known_negative() {
        let Some(found) = provable_raise_of("math.sqrt(-2)") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("math.sqrt(-2) must provably raise");
        };
        assert!(found.1.contains("ValueError"), "{}", found.1);
        assert!(found.1.contains("math domain error"), "{}", found.1);
    }

    #[test]
    fn test_provable_raise_math_sqrt_of_known_nonnegative_declines() {
        assert!(provable_raise_of("math.sqrt(4)").is_none());
    }

    // --- set display and set operators/methods ---

    #[test]
    fn test_set_display_builds_the_shared_list_shape() {
        let Some(value) = eval("{1, 2, 3}") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items.len(), 3);
    }

    #[test]
    fn test_set_union_operator_and_method_agree() {
        let Some(operator_result) = eval("{1, 2} | {2, 3}") else { return };
        assert_eq!(operator_result.items.len(), 3);
        let Some(method_result) = eval("{1, 2}.union({2, 3})") else { return };
        assert_eq!(method_result.items.len(), 3);
    }

    #[test]
    fn test_set_intersection_operator() {
        let Some(value) = eval("{1, 2, 3} & {2, 3, 4}") else { return };
        assert_eq!(value.items.len(), 2);
    }

    #[test]
    fn test_set_difference_operator() {
        let Some(value) = eval("{1, 2, 3} - {2}") else { return };
        assert_eq!(value.items.len(), 2);
    }

    #[test]
    fn test_set_symmetric_difference_operator() {
        let Some(value) = eval("{1, 2} ^ {2, 3}") else { return };
        assert_eq!(value.items.len(), 2);
    }

    #[test]
    fn test_set_issubset_true() {
        let Some(value) = eval("{1}.issubset({1, 2})") else { return };
        assert_eq!(value.values, vec![1.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
    }

    #[test]
    fn test_set_issubset_false() {
        let Some(value) = eval("{1, 9}.issubset({1, 2})") else { return };
        assert_eq!(value.values, vec![0.0]);
    }

    #[test]
    fn test_set_issuperset() {
        let Some(value) = eval("{1, 2}.issuperset({1})") else { return };
        assert_eq!(value.values, vec![1.0]);
    }

    #[test]
    fn test_in_over_set_display() {
        let Some(present) = eval("2 in {1, 2, 3}") else { return };
        assert_eq!(present.values, vec![1.0]);
    }

    // --- dict view methods ---

    #[test]
    fn test_dict_keys_view() {
        let Some(value) = eval("list({\"a\": 1, \"b\": 2}.keys())") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items.len(), 2);
    }

    #[test]
    fn test_dict_values_view() {
        let Some(value) = eval("list({\"a\": 1, \"b\": 2}.values())[0]") else { return };
        assert_eq!(value.values, vec![1.0]);
    }

    #[test]
    fn test_dict_items_view() {
        let Some(value) = eval("list({\"a\": 1}.items())[0]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items.len(), 2);
        assert_eq!(value.items[1].values, vec![1.0]);
    }

    // --- string slicing ---

    #[test]
    fn test_string_slice_basic_range() {
        let Some(value) = eval("\"abcdefgh\"[0:4]") else { return };
        let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
        assert_eq!(text, "abcd");
    }

    #[test]
    fn test_string_slice_clamps_past_the_end_rather_than_raising() {
        let Some(value) = eval("\"abcdefghij\"[0:99]") else { return };
        let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
        assert_eq!(text, "abcdefghij");
    }

    #[test]
    fn test_string_slice_missing_bounds_default_to_whole_string() {
        let Some(value) = eval("\"ab\"[:]") else { return };
        let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
        assert_eq!(text, "ab");
    }

    #[test]
    fn test_string_slice_with_step_declines() {
        let Some(value) = eval("\"abcdef\"[::2]") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    // --- opaque values ---

    // These four read through `abstract_value::opaque_value`, not
    // `abstract_value::opaque()` — "known kind of thing, unknown
    // contents" builds Kind::Object with a kind_word (never Kind::Unknown
    // / opaque:true, which means "arrived from entirely outside this
    // file's determination"). assignability.rs's OPAQUE law depends on
    // this: a kind_word'd Kind::Object fires against any scalar-ground
    // declared set, so `type(40)` assigned into an int-ground alias
    // fires instead of declining Undetermined.

    #[test]
    fn test_dunder_class_reads_opaque() {
        let Some(value) = eval("object().__class__") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("the __class__ object"));
    }

    #[test]
    fn test_type_call_reads_opaque() {
        let Some(value) = eval("type(40)") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("a type object"));
    }

    #[test]
    fn test_re_compile_reads_opaque() {
        let Some(value) = eval("re.compile(\"a\")") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("a compiled pattern"));
    }

    #[test]
    fn test_re_match_reads_opaque() {
        let Some(value) = eval("re.match(\"a\", \"banana\")") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value.kind_word, Some("a match object"));
    }

    // --- f-string float spelling ---

    #[test]
    fn test_fstring_float_spelling_keeps_the_decimal_point() {
        let Some(value) = eval("f\"{30.0}\"") else { return };
        let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
        assert_eq!(text, "30.0");
    }

    #[test]
    fn test_fstring_float_spelling_non_whole_value() {
        let Some(value) = eval("f\"{3.5}\"") else { return };
        let text: String = value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect();
        assert_eq!(text, "3.5");
    }

    // --- f-string composition over a known SET interpolation (item 2) ---

    /// `f"n={counted(n)}"` where `counted`'s body is a `while` loop (a
    /// genuine `interpret_body` decline, unlike an ellipsis-only stub —
    /// see this unit's own report on why `a-statements.py`'s literal
    /// `unread_number() -> int: ...` does NOT reach this fallback: an
    /// ellipsis body falls through to `Kind::Null`, never a decline).
    /// The declined call's `-> int` annotation answers the whole-number
    /// set (`summaries::return_sort_fallback`, item 1), so the f-string
    /// steps down to the PATTERN tier instead of `unknown()` — a known
    /// `Kind::Set`, never `Kind::Unknown`.
    #[test]
    fn test_fstring_with_a_sort_only_set_interpolation_composes_a_pattern() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def counted(n) -> int:\n    while n > 0:\n        n -= 1\n    return n\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("f\"n={counted(3)}\"").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        // the pattern tier answers a SET (the concatenation of the "n="
        // tuple with the interpolation's spellings), never unknown().
        // Whether that pattern is contained in a bounded length window
        // is the kernel's containment question — its subset decider
        // REFUSES this concatenation-vs-window shape today (assignability
        // catches the refusal and answers Undetermined), so no raw
        // subset ask is made here; the composition itself is the claim
        // this test pins.
        assert_eq!(value.kind, Kind::Set);
        assert!(!value.set.forms.is_empty());
    }

    /// A plain literal-only f-string with no interpolation at all still
    /// composes the exact string it always did — the pattern tier is
    /// never reached when there is nothing to interpolate.
    #[test]
    fn test_fstring_plain_literal_still_answers_exact() {
        let Some(value) = eval("f\"hello\"") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
    }

    // --- list slicing (item 6, c-reads-and-values.py's list_slice) ---

    /// `xs[0:1][0]` — a slice re-subscripted, c-reads-and-values.py's
    /// own `list_slice` shape: the slice answers a known one-element
    /// list, and the following `[0]` reads its sole element back out.
    #[test]
    fn test_list_slice_then_subscript_reads_the_sliced_element() {
        let Some(value) = eval("[200, 201][0:1][0]") else { return };
        assert_eq!(value.values, vec![200.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// An out-of-order slice (`lower >= upper` after clamping) answers
    /// the empty list, matching the string-slice sibling's same row.
    #[test]
    fn test_list_slice_empty_range_answers_the_empty_list() {
        let Some(value) = eval("[1, 2, 3][2:1]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items.len(), 0);
    }

    /// A negative slice bound adjusts by the list's own length first,
    /// the same rule the plain-index and string-slice rows already
    /// follow.
    #[test]
    fn test_list_slice_negative_bound_adjusts_by_length() {
        let Some(value) = eval("[10, 20, 30][-2:]") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items, vec![known_values(vec![20.0], PrimitiveKind::Integer, TrustProved), known_values(vec![30.0], PrimitiveKind::Integer, TrustProved)]);
    }

    // --- list.pop() as an RHS value (item 5) ---

    /// `overs.pop()` used directly as a value (not first bound to a
    /// name) — c-reads-and-values.py's `list_pop` shape: `return
    /// overs.pop()`. The RESULT half of `mutated_receiver`'s pair reads
    /// through the value-call dispatch, answering the popped element.
    #[test]
    fn test_list_pop_as_a_value_expression_answers_the_popped_element() {
        let Some(value) = eval("[200, 201].pop()") else { return };
        assert_eq!(value.values, vec![201.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `xs.pop(0)` — the one-argument indexed form also reads through
    /// the value path.
    #[test]
    fn test_list_pop_with_an_index_as_a_value_expression() {
        let Some(value) = eval("[200, 201].pop(0)") else { return };
        assert_eq!(value.values, vec![200.0]);
    }

    /// `[].pop()` on an empty receiver declines (there is nothing to
    /// pop) — the same honesty `mutated_receiver`'s own statement-sink
    /// row already carries.
    #[test]
    fn test_list_pop_on_an_empty_receiver_declines() {
        let Some(value) = eval("[].pop()") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    // --- sum over a generator (a-statements.py's own generator_expression row) ---

    #[test]
    fn test_sum_over_generator_expression() {
        let Some(value) = eval("sum(age for age in [10, 20, 30])") else { return };
        assert_eq!(value.values, vec![60.0]);
    }

    // --- j-stdlib-surfaces.py: datetime family ---

    /// `datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).timestamp()`
    /// is exactly `0.0` — the POSIX epoch itself.
    #[test]
    fn test_datetime_timestamp_at_the_epoch_is_zero() {
        let Some(value) = eval("datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).timestamp()") else { return };
        assert_eq!(value.values, vec![0.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    /// `datetime.datetime(2033, 5, 18, tzinfo=datetime.timezone.utc).timestamp()`
    /// — the exact later timestamp j-stdlib-surfaces.py's own
    /// `datetime_timestamp` row marks past the Age ceiling.
    #[test]
    fn test_datetime_timestamp_of_a_later_aware_utc_date() {
        let Some(value) = eval("datetime.datetime(2033, 5, 18, tzinfo=datetime.timezone.utc).timestamp()") else { return };
        assert_eq!(value.values, vec![1999987200.0]);
    }

    /// A NAIVE datetime's `.timestamp()` (no `tzinfo=`) declines — this
    /// file does not reproduce the host-local-time `mktime` conversion
    /// datetime.rst documents for the naive row.
    #[test]
    fn test_datetime_timestamp_of_a_naive_datetime_declines() {
        let Some(value) = eval("datetime.datetime(1970, 1, 1).timestamp()") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.datetime.now()` — a value that changes every run, never
    /// pinned to a scalar: answered opaque.
    #[test]
    fn test_datetime_now_is_opaque() {
        let Some(value) = eval("datetime.datetime.now()") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some());
    }

    /// `.year` on a constructed datetime answers opaque (never a
    /// specific value this file claims to pin, per this row's own
    /// fixture framing).
    #[test]
    fn test_datetime_year_is_opaque() {
        let Some(value) = eval("datetime.datetime(1970, 1, 1).year") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some());
    }

    /// `.isoformat()` on a constructed datetime answers opaque.
    #[test]
    fn test_datetime_isoformat_is_opaque() {
        let Some(value) = eval("datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).isoformat()") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some());
    }

    // --- j-stdlib-surfaces.py: re family ---

    /// `re.search("z", "banana")` — the literal pattern "z" never
    /// occurs in "banana," so the exact answer is `None`.
    #[test]
    fn test_re_search_absent_literal_pattern_answers_none() {
        let Some(value) = eval("re.search(\"z\", \"banana\")") else { return };
        assert_eq!(value.kind, Kind::Null);
    }

    /// `re.search("a", "banana")` — the literal pattern IS present, so
    /// the answer is the match-object sort (opaque).
    #[test]
    fn test_re_search_present_literal_pattern_answers_a_match_object() {
        let Some(value) = eval("re.search(\"a\", \"banana\")") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some());
    }

    /// `re.sub("a", "b", "aaaaaaaaaa")` replaces EVERY match — ten "a"s
    /// become ten "b"s.
    #[test]
    fn test_re_sub_literal_pattern_replaces_every_match() {
        let Some(value) = eval("re.sub(\"a\", \"b\", \"aaaaaaaaaa\")") else { return };
        assert_eq!(value.values.len(), 10);
    }

    /// A pattern carrying a regex metacharacter declines — this file
    /// only reduces METACHARACTER-FREE patterns to a substring test.
    #[test]
    fn test_re_search_with_a_metacharacter_pattern_declines() {
        let Some(value) = eval("re.search(\"a.b\", \"axb\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    // --- j-stdlib-surfaces.py: json family ---

    #[test]
    fn test_json_loads_parses_an_integer_literal() {
        let Some(value) = eval("json.loads(\"200\")") else { return };
        assert_eq!(value.values, vec![200.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn test_json_dumps_serializes_a_known_dict() {
        let Some(value) = eval("json.dumps({\"age\": 40})") else { return };
        assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
        assert_eq!(exact_string_values(&value).and_then(code_points_to_string).as_deref(), Some(r#"{"age": 40}"#));
    }

    // --- j-stdlib-surfaces.py: exceptions ---

    /// `str(Exception("failure"))` answers the message unchanged.
    #[test]
    fn test_str_of_exception_answers_the_message() {
        let Some(value) = eval("str(Exception(\"failure\"))") else { return };
        assert_eq!(exact_string_values(&value).and_then(code_points_to_string).as_deref(), Some("failure"));
    }

    /// `ExceptionGroup(...)` answers opaque — its message/wrapped
    /// exceptions are never decomposed by this file.
    #[test]
    fn test_exception_group_construction_is_opaque() {
        let Some(value) = eval("ExceptionGroup(\"many\", [ValueError(\"a\")])") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some());
    }

    // --- j-stdlib-surfaces.py: dict/misc ---

    /// `types.MappingProxyType(d)["age"]` reads through to the wrapped
    /// dict's own value.
    #[test]
    fn test_mapping_proxy_type_reads_through_to_the_wrapped_dict() {
        let Some(value) = eval("types.MappingProxyType({\"age\": 40})[\"age\"]") else { return };
        assert_eq!(value.values, vec![40.0]);
    }

    /// `xs.sort()` used directly as a value expression — the RETURN
    /// VALUE is always `None`, a sort mismatch against a refined Age.
    #[test]
    fn test_list_sort_as_a_value_expression_answers_none() {
        let Some(value) = eval("[41, 40].sort()") else { return };
        assert_eq!(value.kind, Kind::Null);
    }

    /// `list(map(lambda age: age + 1, [39]))[0]` — the materialized map.
    #[test]
    fn test_map_materialized_via_list_answers_the_mapped_elements() {
        let Some(value) = eval("list(map(lambda age: age + 1, [39]))") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items, vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)]);
    }

    /// `list(filter(lambda age: age > 100, [40, 200]))[0]` — the
    /// materialized filter, keeping only the surviving element.
    #[test]
    fn test_filter_materialized_via_list_answers_the_kept_elements() {
        let Some(value) = eval("list(filter(lambda age: age > 100, [40, 200]))") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(value.items, vec![known_values(vec![200.0], PrimitiveKind::Integer, TrustProved)]);
    }

    // --- j-stdlib-surfaces.py: str ---

    /// `long.find("%")` feeding `long[:long_at]` — the fixed `find`
    /// Integer-sort bug this wave closes (`string_models.rs`'s own
    /// `find` row): a `Number`-tagged result used to decline the slice
    /// bound outright.
    #[test]
    fn test_find_result_feeds_a_slice_bound() {
        let Some(value) = eval("\"123456789%\"[:\"123456789%\".find(\"%\")]") else { return };
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::String));
        assert_eq!(exact_string_values(&value).and_then(code_points_to_string).as_deref(), Some("123456789"));
    }

    /// `key in bag` — a known List container whose elements are opaque
    /// class instances (weakref.WeakSet's own `.add(key)` shape,
    /// j-stdlib-surfaces.py's `weak_set_contains` row): element equality
    /// cannot be decided, but the `in` expression's own SORT is still
    /// provably `bool` — answered opaque rather than fully unknown.
    #[test]
    fn test_in_operator_over_opaque_elements_answers_an_opaque_boolean() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("key in bag").expect("test source must parse");
        let mut environment = empty_environment();
        environment.bind("key", opaque_value("a class instance"));
        environment.bind("bag", collection_models::list_literal_value(&[opaque_value("a class instance")]));
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some());
    }

    // --- p-typed-array.py: bytes/bytearray/memoryview/array.array construction ---

    /// `bytes([10, 20, 30])[2]` — p-typed-array.py's own `bytes_from_
    /// iterable` row: the constructor call answers the known list, and
    /// element 2 reads through unchanged.
    #[test]
    fn test_bytes_constructor_from_a_known_list_reads_the_exact_element() {
        let Some(value) = eval("bytes([10, 20, 30])[2]") else { return };
        assert_eq!(value.values, vec![30.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `bytearray(4)[0]` — p-typed-array.py's own `bytearray_from_
    /// length` row: a length-only construction zero-fills every slot.
    #[test]
    fn test_bytearray_constructor_from_a_length_zero_fills() {
        let Some(value) = eval("bytearray(4)[0]") else { return };
        assert_eq!(value.values, vec![0.0]);
    }

    /// `bytearray(b"\x0a\x14")[1]` — a bytes-literal argument to
    /// `bytearray(...)` copies through the same known-list-of-known-
    /// Integers shape a `bytes([...])` display builds.
    #[test]
    fn test_bytearray_constructor_from_a_bytes_literal_reads_the_exact_element() {
        let Some(value) = eval("bytearray(b\"\\x0a\\x14\")[1]") else { return };
        assert_eq!(value.values, vec![20.0]);
    }

    /// `memoryview(bytearray(b"..."))[3]` — p-typed-array.py's own
    /// `memoryview_over_bytearray_reads` row: a view shares the SAME
    /// element sequence as the underlying bytearray.
    #[test]
    fn test_memoryview_constructor_reads_through_the_shared_buffer() {
        let Some(value) = eval("memoryview(bytearray(b\"\\x00\\x01\\x02\\x03\"))[3]") else { return };
        assert_eq!(value.values, vec![3.0]);
    }

    /// `array.array("d", [10.0, 20.0, 30.0])[2]` — p-typed-array.py's
    /// own `array_double_from_iterable` row: every element reads as a
    /// FLOAT, never an int, whatever numeric literal built it.
    #[test]
    fn test_array_double_constructor_reads_a_float_tagged_element() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("import array\n").expect("test module parses").into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("array.array(\"d\", [10.0, 20.0, 30.0])[2]").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![30.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    /// `len(bytearray(10))` — the constructed value's own element count
    /// composes through the ordinary `len()` dispatch once the
    /// constructor answers a known `Kind::List`, with no bytes-specific
    /// `len()` row needed (`collection_models::len_result`'s own generic
    /// `Kind::List` row already covers it).
    #[test]
    fn test_len_of_a_bytearray_constructor_composes() {
        let Some(value) = eval("len(bytearray(10))") else { return };
        assert_eq!(value.values, vec![10.0]);
    }

    // --- h/c-file: computed dict key evaluating to a known string ---

    /// h-object-literal-members.py's own `computed_key_other_expression`
    /// / c-reads-and-values.py's own `read_type_member_computed_name`:
    /// `key = "age"` then `{key: 200}` — a COMPUTED key (a bare Name,
    /// never a string LITERAL) that reduces to a known exact string now
    /// has a slot, the same `DictKey::string` entry a literal `{"age":
    /// 200}` would build.
    #[test]
    fn test_dict_literal_with_a_computed_string_key_builds_and_reads_back() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("{key: 200}[key]").expect("test source must parse");
        let mut environment = empty_environment();
        environment.bind("key", string_models::string_literal_value("age"));
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![200.0]);
    }

    /// The SAME computed-key shape through a ternary — c-reads-and-
    /// values.py's `read_computed_other_key`'s own `"age" if flag else
    /// "years"` construction (proven here directly against a bound
    /// String value, the ternary's own settled answer).
    #[test]
    fn test_dict_literal_with_a_ternary_computed_string_key_builds() {
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("{(\"age\" if flag else \"years\"): 40}[\"age\"]").expect("test source must parse");
        let mut environment = empty_environment();
        environment.bind("flag", known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved));
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![40.0]);
    }

    // --- d-module-surface.py: importlib.import_module ---

    /// `importlib.import_module("d_helper")` — d-module-surface.py's own
    /// `dynamic_import` row: this domain has no module-object Kind, so
    /// the answer is the opaque "a module object" sort.
    #[test]
    fn test_importlib_import_module_answers_opaque() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("import importlib\n").expect("test module parses").into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("importlib.import_module(\"d_helper\")").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert!(value.kind_word.is_some(), "importlib.import_module(...) must answer opaque, not unknown: {value:?}");
    }

    // --- e-class-and-function.py: generator METHODS via next()/anext() ---

    /// e-class-and-function.py's own `generator_method`: `next(GenAges()
    /// .ages())` where `ages(self)` is a generator METHOD, not a bare
    /// def — the method-call dispatch now routes a generator-shaped
    /// method through `instances::generator_yields` (with `self`
    /// prepended to the positional arguments) instead of declining
    /// through `method_call_result`'s own no-`yield`-row interpreter.
    #[test]
    fn test_generator_method_call_answers_the_first_yielded_value_via_next() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "class GenAges:\n",
            "    def ages(self):\n",
            "        yield 40\n",
            "        yield 41\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let empty_aliases = std::collections::HashMap::new();
        let empty_imports = crate::refinedpy::surface::surface_imports(&ruff_python_ast::ModModule {
            node_index: ruff_python_ast::AtomicNodeIndex::NONE,
            range: TextRange::default(),
            body: Vec::new().into(),
        });
        let classes = crate::refinedpy::instances::class_table(&module, &empty_aliases, &empty_imports, &kernel);
        let mut environment = empty_environment();
        environment.set_classes(std::sync::Arc::new(classes));
        let parsed = parse_expression("next(GenAges().ages())").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![40.0], "the generator method's first yield must read through next(): {value:?}");
    }

    /// `anext` dispatches identically to `next` once `await` transparently
    /// unwraps — e-class-and-function.py's own `async_generator_first_
    /// value`/`generator_first_value` pair.
    #[test]
    fn test_anext_of_a_generator_call_answers_the_first_yielded_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "async def async_yield_ages():\n",
            "    yield 40\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("await anext(async_yield_ages())").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![40.0]);
    }

    // --- e-class-and-function.py: keyword-only and **kwargs calls ---

    /// e-class-and-function.py's own `keyword_only_call`: a keyword-only
    /// parameter the CALLER covers by keyword (`only_keyword(age=200)`)
    /// now interprets the body's own exact value, rather than declining
    /// outright.
    #[test]
    fn test_keyword_only_call_binds_and_interprets() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def only_keyword(*, age):\n    return age\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("only_keyword(age=200)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![200.0]);
    }

    /// e-class-and-function.py's own `kwargs_parameter`: `**fields`
    /// collects every keyword the call site passes into a dict, and the
    /// body's own `fields["age"]` reads it back exactly.
    #[test]
    fn test_kwargs_call_collects_keywords_into_a_dict() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("def gather_kwargs(**fields):\n    return fields[\"age\"]\n")
            .expect("test module parses")
            .into_syntax();
        let table = std::sync::Arc::new(crate::refinedpy::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("gather_kwargs(age=200)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![200.0]);
    }
}
