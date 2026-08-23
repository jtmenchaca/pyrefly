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
use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::nan_value;
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
use refined_domain::trust_grades::TrustLevel;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::CalendarQuestion;
use refined_kernel::kernel_interface::CalendarQuestionOp;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::above;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::repeat_of;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::ConversionFlag;
use ruff_python_ast::Expr;
use ruff_python_ast::InterpolatedStringElement;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::UnaryOp;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::assignability;
use crate::builtin_models;
use crate::bytes_models;
use crate::bytes_models::BytesAnswer;
use crate::collection_models;
use crate::diagnostic_sentences;
use crate::env;
use crate::env::Environment;
use crate::foreign_edge;
use crate::instances;
use crate::math_models;
use crate::narrowing;
use crate::string_models;
use crate::summaries;

/// What this expression evaluates to in this environment. `unknown()`
/// is the honest default for every construct not yet built — an
/// unknown never fires and never silently passes a judgment.
pub fn evaluate_expression(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    // A node whose value the walk already proved answers it directly
    // (`Environment::evaluated_node`). The relational sum is the one
    // publisher: a division whose operands the kernel tied together
    // answers more tightly than evaluating the two sides here could,
    // because the tie is a fact of the kernel program rather than of
    // either side. Checked at this one dispatch head, so a published
    // node is found wherever it sits in the tree.
    if let Some(published) = environment.evaluated_node(expression.range()) {
        return published.clone();
    }
    let value = evaluate_expression_dispatch(expression, environment, kernel);
    // Recorded ONLY when a caller asked for it (env.rs's own doc on
    // `evaluations`/`record_evaluation`) — an ordinary check never
    // opts in, so this is a no-op `Option` check for every node on
    // every walk except `check.rs::refined_set_at_position`'s own.
    environment.record_evaluation(expression.range(), value.clone());
    value
}

fn evaluate_expression_dispatch(
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
        // already recorded a creation of this exact lambda into
        // `environment` (its statement-level caller runs that scan
        // before reaching this evaluation — `check.rs::sink_value`,
        // `summaries::interpret_body`'s `Stmt::Return` arm), the value
        // encodes the CURRENT retained-callable key on `source`
        // (`env::retained_callable_value`) so a later call through
        // `evaluate_call`'s retained-callable arm can interpret the
        // body instead of declining. The key is read back through
        // `environment.lambda_key` — never the range itself as a key
        // (`env.rs`'s own doc on why a fresh id is minted per creation,
        // not the AST range) — so two creations of the SAME lambda
        // text (`make_adder(1)` and `make_adder(100)`, each closing
        // over a different `step`) never conflate. A lambda `register_
        // retained_callables` never reached (a shape outside its own
        // recursion, or an environment with no such registration step
        // at all — every existing test environment, unaffected) still
        // answers the plain opaque value exactly as before this table
        // existed.
        Expr::Lambda(lambda) => match environment.lambda_key(lambda.range().start().to_u32()) {
            Some(key) => env::retained_callable_value(key),
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
/// argument, or a bare `return <lambda>`), and records each one into
/// `environment` with a CLOSURE snapshot of every free name its own
/// body reads (`e-class-and-function.py`'s `make_adder`: `return
/// lambda age: age + step` reads `step`, `make_adder`'s own
/// parameter — a lambda is not always closure-free, so this scan
/// always computes the snapshot rather than assuming one is never
/// needed). Reused rather than duplicated: `RetainedCallable::
/// from_lambda` builds the synthetic single-`Return` body first, and
/// `summaries::free_variable_snapshot` reads that SAME body's own free
/// names — the identical free-name reader `Stmt::FunctionDef`'s own
/// retention (`summaries::interpret_body`) already calls for a nested
/// def. Each registration mints a FRESH key
/// (`Environment::next_retained_callable_key`) and publishes it as the
/// lambda's own range's CURRENT key (`Environment::record_lambda_key`)
/// — never keys by the range itself, so a second creation of the same
/// lambda text with a different closure (`make_adder(1)` vs.
/// `make_adder(100)`) never overwrites the first's still-live retained
/// value under a shared key.
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
            let placeholder = env::RetainedCallable::from_lambda(lambda, HashMap::new());
            let synthetic_def = placeholder.as_synthetic_def("<lambda>", lambda.range());
            let closure = summaries::free_variable_snapshot(&synthetic_def, environment);
            let key = environment.next_retained_callable_key();
            environment.record_retained_callable(key, env::RetainedCallable::from_lambda(lambda, closure));
            environment.record_lambda_key(lambda.range().start().to_u32(), key);
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
/// `overs[0:1][0]`, a slice immediately re-subscripted). A THIRD
/// receiver shape answers a narrower claim: a string-shaped SET (a
/// concatenation/repeat window, never an exact literal — those are
/// `Kind::Values` and already answered above) sliced exactly `[:n]`
/// (`lower` absent or the exact literal `0`, `upper` an exact
/// non-negative Integer, no `step`) asks the kernel's `seq_prefix` —
/// `prefixReadOf`'s proved over-approximation of `take n` on every
/// member (boundary/exports_sets.lean's `kernelSeqPrefix`). Any other
/// slice shape over a non-exact receiver — a `step`, a nonzero `lower`,
/// or an `upper` that is not an exact Integer — declines, naming the
/// construct it cannot read rather than guessing.
fn evaluate_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if slice.step.is_some() {
        return unknown();
    }
    if let Some(result) = sequence_prefix_slice(container, slice, environment, kernel) {
        return result;
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

/// The `[:n]` prefix-read arm of `evaluate_slice`: fires only when the
/// receiver is a string-shaped SET (`string_shaped_set` — a concatenation
/// or repeat window; an exact literal is `Kind::Values` and already
/// answered by `evaluate_slice`'s own Values arm, so this never
/// double-answers that case) AND the slice is exactly `[:n]` — `lower`
/// absent or the exact known value `0`, `upper` an exact known
/// non-negative Integer, no `step` (checked by the caller before this
/// runs). Any other slice shape over a set-shaped receiver — a nonzero
/// or unknown `lower`, or an `upper` that is not an exact non-negative
/// Integer — answers `None` immediately, so the caller's own decline
/// stands rather than this function guessing. The kernel itself can
/// ALSO decline once the shape is asked (`kernel.seq_prefix`'s own
/// `None` — the receiver is not `seqOf`-recognized, e.g. a leading
/// concatenation operand that is not a fixed scalar): that decline is an
/// ORDINARY answer, not a fault, and this function answers `None` for it
/// exactly the same way, so the caller falls through to the
/// length-based fallback precisely as if this arm had never matched.
fn sequence_prefix_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    // An exact literal (`Kind::Values`) is already answered exactly by
    // `evaluate_slice`'s own Values arm below; asking the kernel's
    // window-shaped `seq_prefix` for it would answer a WIDER claim (a
    // shape over an unstated member) in place of the exact slice this
    // domain already has, so this arm only reads a genuine window
    // receiver — `string_shaped_set`'s `Kind::Set` branch, never its
    // exact-literal branch (that branch exists for `string_set_concatenation`,
    // not this caller).
    if container.kind != Kind::Set {
        return None;
    }
    let receiver_set = string_shaped_set(container)?;
    if let Some(expr) = &slice.lower {
        if slice_bound_index(expr, environment, kernel) != Some(0) {
            return None;
        }
    }
    let upper_expr = slice.upper.as_ref()?;
    let n = slice_bound_index(upper_expr, environment, kernel)?;
    if n < 0 {
        return None;
    }
    let prefix_set = (kernel.seq_prefix)(&unbounded_repeats(&receiver_set), n)?;
    Some(known_set(prefix_set, None, TrustProved, SetKindTag::None))
}

/// Relaxes every `Repeat`/`RepeatWord` form reachable through the set's
/// own concatenation/union/difference/star operands to its UNBOUNDED
/// twin (`hi: None`) — sound because a bounded window's every member is
/// trivially a member of the same window with its ceiling dropped
/// (widening a claim only ever admits more), and `seqOf`
/// (set_functions/subset_seq_shape.lean) recognizes a `Repeat` position
/// only when `hi` is `none`. `seq_prefix`'s own soundness
/// (`prefixReadOf_sound`) never reads the receiver's ceiling — its
/// premise is `SeqDen`, which states membership per position and the
/// open tail's `Star`, neither of which mentions `hi` — so asking the
/// kernel about this relaxed receiver instead of the tighter original
/// still proves the same `take n` fact for both. A parameter's own
/// `Field(min_length=…, max_length=…)` window is exactly the shape this
/// exists for (`check.rs::seed_parameters`'s own doc): the LENGTH bound
/// still narrows what `evaluate_slice`'s caller reads through `len()`
/// elsewhere; only the prefix READ over-approximates.
fn unbounded_repeats(set: &RefinedSet) -> RefinedSet {
    let forms = set
        .forms
        .iter()
        .map(|form| {
            let mut relaxed = form.clone();
            if matches!(form.form, Form::Repeat | Form::RepeatWord) {
                relaxed.hi = None;
            }
            if let Some(a) = &form.a_ {
                relaxed.a_ = Some(Box::new(unbounded_repeats(a)));
            }
            if let Some(b) = &form.b {
                relaxed.b = Some(Box::new(unbounded_repeats(b)));
            }
            relaxed
        })
        .collect();
    make_refined_set(forms)
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
    // never identical to None. A `Kind::PossiblyUndefined` side (an
    // `Optional[X]`/`X | None`-declared parameter's own seed,
    // `check.rs::seed_parameters`) is NOT a known non-None value the way
    // an ordinary present-only Kind settles as — its own absent side may
    // genuinely BE None at runtime, so its identity against None stays
    // undecided here exactly like Unknown's, leaving `narrowing.rs`'s
    // `narrow_is_none` (the maybe carrier's own unwrap) to state what
    // each fork actually proves instead of this law guessing one arm
    // dead outright.
    if op == CmpOp::Is || op == CmpOp::IsNot {
        let identical = match (left.kind == Kind::Null, right.kind == Kind::Null) {
            (true, true) => true,
            (true, false) | (false, true) => {
                if right.kind == Kind::Unknown
                    || left.kind == Kind::Unknown
                    || right.kind == Kind::PossiblyUndefined
                    || left.kind == Kind::PossiblyUndefined
                {
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
    // one side a single known numeric value, the OTHER a bounded
    // Integer-sorted window (`integer_set_bounds` — `len()`'s own answer
    // over a `Repeat`-shaped receiver, `collection_models::len_result`'s
    // doc: `[window.lo, window.hi]`, never one exact count): the
    // comparison decides only when the WHOLE window agrees, since the
    // window states every value it admits, never which one a given run
    // actually holds. `==`/`!=` decide only on a DEGENERATE window
    // (`lo == hi`, the len-of-a-fixed-length-slice case) — a wider
    // window can never prove `==` true (some other admitted length
    // would disagree) or `!=` true (the target might be the one held
    // length), so both stay undecided there. The four orderings decide
    // whenever the window sits entirely on one side of the target
    // (`hi <op> target` uniform down to `lo`, or the mirror), which a
    // degenerate window trivially satisfies too.
    if let Some(result) = numeric_value_vs_window_compare(op, left, right, false)
        .or_else(|| numeric_value_vs_window_compare(op, right, left, true))
    {
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

/// One comparison operator between a single known numeric value and a
/// bounded Integer-sorted window (`integer_set_bounds`'s own `[lo, hi]`
/// reading), decided only when EVERY value the window admits agrees —
/// the window is a claim over an unstated member, never a promise about
/// which one, so a partial overlap must stay `None` rather than guess.
/// `numeric_side`/`window_side` name which of `compare_pair`'s two
/// operands is being read here. The per-op bodies below read literally
/// as `window_side <op> target` (`Lt` means "every admitted value is
/// below target," matching a name like `low_window` at the call site).
/// `swapped` is `true` when `window_side` is actually `compare_pair`'s
/// LEFT operand, i.e. the original claim already reads `window_side
/// <op> numeric_side` — the same direction the bodies compute, so `op`
/// passes through unchanged. When `swapped` is `false`, the original
/// claim reads `numeric_side <op> window_side` — the mirror of what the
/// bodies compute — so `op` is inverted first (`x < y` read as `y > x`)
/// rather than this function reversing the arithmetic itself.
fn numeric_value_vs_window_compare(
    op: CmpOp,
    numeric_side: &AbstractValue,
    window_side: &AbstractValue,
    swapped: bool,
) -> Option<bool> {
    let (target, _) = single_numeric_value(numeric_side)?;
    let (lo, hi) = integer_set_bounds(window_side)?;
    let (lo, hi) = (lo as f64, hi as f64);
    let effective_op = if swapped {
        op
    } else {
        match op {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::LtE => CmpOp::GtE,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::GtE => CmpOp::LtE,
            other => other,
        }
    };
    match effective_op {
        // every admitted value equals target only when the window is
        // degenerate AND that one value IS target; a wider window can
        // never prove equality (some other admitted length disagrees)
        CmpOp::Eq => {
            if lo == hi {
                Some(lo == target)
            } else {
                None
            }
        }
        // != is decided the same way `==` is (its exact negation, once
        // decidable), plus the case the window misses target ENTIRELY
        // (every admitted value differs, whether or not the window is
        // degenerate)
        CmpOp::NotEq => {
            if hi < target || target < lo {
                Some(true)
            } else if lo == hi {
                Some(lo != target)
            } else {
                None
            }
        }
        CmpOp::Lt => {
            if hi < target {
                Some(true)
            } else if lo >= target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::LtE => {
            if hi <= target {
                Some(true)
            } else if lo > target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Gt => {
            if lo > target {
                Some(true)
            } else if hi <= target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::GtE => {
            if lo >= target {
                Some(true)
            } else if hi < target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => unreachable!("handled above compare_pair's own call site"),
    }
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
                if interpolation.conversion != ConversionFlag::None {
                    return unknown();
                }
                if let Some(format_spec) = &interpolation.format_spec {
                    let value = evaluate_expression(&interpolation.expression, environment, kernel);
                    let Some(part) = zero_padded_decimal_spelling(format_spec, &value) else {
                        return unknown();
                    };
                    has_exact = false;
                    grade = refined_domain::trust_grades::min_trust_level(grade, TrustSpec);
                    parts.push(part);
                    continue;
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

/// `f"{year:04d}"` — an interpolation carrying a ZERO-PADDED DECIMAL
/// format spec (`format_spec.rst`, "Format Specification Mini-Language":
/// `[[fill]align][sign][z][#][0][width][grouping_option][.precision][type]`
/// — this reader recognizes only the plain `0{width}d` spelling: no
/// fill/align/sign/`#`/grouping/precision, `type` exactly `d`). `value`
/// need not be a single known integer — this is the row that fires for
/// a BOUNDED Integer-sorted set (`year: Annotated[int, Field(ge=1970,
/// le=9999)]` seeds `Kind::Set`, never `Kind::Values` —
/// `check.rs::seed_parameters`'s scalar-declared-set arm), which
/// `single_numeric_value`'s exact-value row above never reaches. Exact
/// only when EVERY integer in the set's own closed range needs no
/// padding at all: `min_digit_count`/`max_digit_count` (the decimal
/// digit count of the range's two ends — the monotone extremes, since a
/// wider magnitude never has FEWER digits) both equal `width` exactly,
/// so the zero-fill never actually adds a digit and the plain decimal
/// alphabet is the exact spelling set either way. A range that would
/// need real padding for some members but not others (`ge=8, le=12`
/// against `02d`: "08".."12", where padding does fire) declines rather
/// than approximate — this row states only the sub-case where padding
/// is provably a no-op. `RefinedSet` is a `Repeat` over the plain digit
/// alphabet at EXACTLY `width` positions — a stronger claim than
/// `int_spelling_set`'s own unbounded-length superset, and exact for
/// this admitted case since every member has exactly `width` digits and
/// carries no sign (the range's own `lo` is checked non-negative below).
fn zero_padded_decimal_spelling(
    format_spec: &ruff_python_ast::InterpolatedStringFormatSpec,
    value: &AbstractValue,
) -> Option<RefinedSet> {
    let width = zero_padded_decimal_width(format_spec)?;
    let (lo, hi) = integer_set_bounds(value)?;
    if lo < 0 {
        return None;
    }
    if decimal_digit_count(lo) != width || decimal_digit_count(hi) != width {
        return None;
    }
    Some(make_refined_set(vec![repeat_of(one_char_of("0123456789"), width as i64, Some(width as i64))]))
}

/// The `width` a format spec states, when the spec is EXACTLY the plain
/// `0{width}d` spelling this reader recognizes — a single literal
/// element (no nested interpolation inside the spec itself, which
/// `format_spec.rst` allows but this reader does not model) whose text
/// is `0` followed by one or more digits followed by `d`. Any other
/// spelling (a fill/align/sign/`#`/grouping/precision character, a
/// different `type`, a spec with its own interpolation) answers `None`.
fn zero_padded_decimal_width(format_spec: &ruff_python_ast::InterpolatedStringFormatSpec) -> Option<u32> {
    let [InterpolatedStringElement::Literal(literal)] = &*format_spec.elements else {
        return None;
    };
    let digits = literal.value.strip_prefix('0')?.strip_suffix('d')?;
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The closed integer bound `[lo, hi]` a value states, when the value is
/// a BOUNDED Integer-sorted `Kind::Set` (`seed_parameters`'s scalar
/// arm — never `Kind::Values`, which `single_numeric_value` already
/// reads exactly). Reads the set's own top-level `AtLeast`/`Above`/
/// `AtMost`/`Below` forms, the same syntactic hull
/// `collection_models::integer_range_bounds` reads for its own bounded-
/// index subscript read — duplicated here rather than exported, since
/// the two files' own AGENT-BRIEF scope (`collection_models.rs`'s
/// container reads; this file's expression evaluation) keeps neither
/// reaching into the other's private helpers, the same convention
/// `string_models.rs`'s own `exact_string_text` doc states for this
/// exact situation.
fn integer_set_bounds(value: &AbstractValue) -> Option<(i64, i64)> {
    if value.kind != Kind::Set || value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &value.set.forms {
        match form.form {
            refined_sets::refinement_forms::Form::AtLeast => {
                lo = Some(lo.map_or(form.a, |current: f64| current.max(form.a)))
            }
            refined_sets::refinement_forms::Form::Above => {
                lo = Some(lo.map_or(form.a.floor() + 1.0, |current: f64| current.max(form.a.floor() + 1.0)))
            }
            refined_sets::refinement_forms::Form::AtMost => {
                hi = Some(hi.map_or(form.a, |current: f64| current.min(form.a)))
            }
            refined_sets::refinement_forms::Form::Below => {
                hi = Some(hi.map_or(form.a.ceil() - 1.0, |current: f64| current.min(form.a.ceil() - 1.0)))
            }
            refined_sets::refinement_forms::Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, hi) = (lo?, hi?);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// The number of decimal digits a NONNEGATIVE integer's plain `str()`
/// spelling carries — `0` itself spells one digit ("0"), matching
/// `format_integer_spelling`'s own no-leading-zero convention.
fn decimal_digit_count(value: i64) -> u32 {
    if value == 0 {
        return 1;
    }
    value.unsigned_abs().to_string().len() as u32
}

/// One codepoint drawn from the given ASCII characters — the digit and
/// sign alphabet `int_spelling_set`/`float_spelling_set` repeat.
fn one_char_of(chars: &str) -> RefinedSet {
    let points: Vec<f64> = chars.chars().map(|c| c as u32 as f64).collect();
    make_refined_set(vec![one_of(&points)])
}

/// Every string `str()` can spell for an Integer-sorted value: one or
/// more characters drawn from the digits and `-` — `stdtypes.rst`
/// (`int.__repr__`) states no other characters and no length ceiling
/// (CPython `int` is arbitrary-precision, verified: `str(10**30) ==
/// "1000000000000000000000000000000"`, `str(-5) == "-5"`, `str(0) ==
/// "0"`). A single `Repeat` over the two-character alphabet, rather than
/// a union of a bare digit run and a `-`-prefixed one, admits a few
/// strings `str()` never produces (an interior or repeated `-`, e.g.
/// `"1-2"`) — still a SOUND superset of every real spelling, and the
/// shape the kernel's sequence reader
/// (`set_functions/subset_seq_shape.lean`'s `seqOf`) recognizes directly:
/// a lone `.Repeat A lo none` is read as a positional shape outright
/// (line `some (List.replicate lo A, some A)`), where a `Union` of two
/// concatenation shapes is not — the pattern union routes
/// (`set_functions/pattern_union_routes.lean`'s `leftRouteB`) only
/// distribute a union that is the FIRST piece of an outer concatenation,
/// and this set is always the TRAILING piece once the caller concatenates
/// it after the f-string's own literal text. The alphabet stays bounded
/// even though the length does not — `Repeat` over a finite `one_of`,
/// never `Star` over the whole codepoint ground — so the kernel's
/// counting-window decider (the same route
/// `temporal_string_grammars.rs`'s `TSG_DIGIT`/`tsg_rep` uses for a
/// bounded digit run) can refute containment in a length window instead
/// of falling through to the unresolved general pattern search.
fn int_spelling_set() -> RefinedSet {
    make_refined_set(vec![repeat_of(one_char_of("0123456789-"), 1, None)])
}

/// Every string `str()` can spell for a Float-sorted value: CPython's
/// `repr(float)` alphabet is digits, `-`, `.`, and a lowercase `e`
/// exponent marker (verified: `str(3.5) == "3.5"`, `str(1e+300) ==
/// "1e+300"`, `str(1e-300) == "1e-300"`), or one of the three
/// non-numeric words `inf`, `-inf`, `nan` (verified: `str(float('inf'))
/// == "inf"`, `str(float('-inf')) == "-inf"`, `str(float('nan')) ==
/// "nan"`) — all three of which are themselves spelled only from
/// letters already admitted below (`i`, `n`, `f`, `a`), so folding their
/// three extra letters into the SAME repeated alphabet as the digit/sign
/// run covers every case with one `Repeat`, the shape
/// `int_spelling_set`'s own doc explains the kernel recognizes directly
/// (a bare `Union` embedded as this set's own trailing position, the way
/// a separate words-union would be, is not recognized the same way).
/// CPython never emits an uppercase `E` or a bare `+` outside an
/// exponent, but admitting `+`/`e`/`i`/`n`/`f`/`a` freely only widens the
/// claim, never narrows it past what `str()` can actually produce.
fn float_spelling_set() -> RefinedSet {
    make_refined_set(vec![repeat_of(one_char_of("0123456789.+-einaf"), 1, None)])
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
/// spells through `int_spelling_set`/`float_spelling_set` instead of the
/// unbounded `codepoint_sets::strings()` this used to fall back to: the
/// bare `strings()` claim is sound but its `Star` shape is one the
/// kernel's placement search cannot always decide against a length
/// window (verified: `refinedpy-check` on this file used to panic with
/// "no pattern inclusion proof — the placement search found none" rather
/// than fire), where a bounded-alphabet `Repeat` routes through the
/// proved counting-window decider instead (refined-lean's
/// `set_functions/subset_window.lean`, `RefinedSet.seqAskableB` on a
/// single `.Repeat` form). A bare `Number` tag (no Python sort proved,
/// `summaries.rs`'s int/float join) reads through the FLOAT alphabet —
/// the wider of the two, so a value that could be either sort still gets
/// a sound superset. Any other `Kind::Set` shape (a set carrying no sort
/// tag at all, or one this function does not recognize) declines — the
/// caller's own `unknown()` fallback stays honest for it.
fn spellings_of_known_set(value: &AbstractValue) -> Option<RefinedSet> {
    if value.kind != Kind::Set {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Integer) => Some(int_spelling_set()),
        Some(PrimitiveKind::Float) | Some(PrimitiveKind::Number) => Some(float_spelling_set()),
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
///
/// Each arm is evaluated under its OWN forked, narrowed environment —
/// exactly the fork/`narrowing::assume` pattern `walk_if` runs for an
/// `if`/`else` STATEMENT (check.rs's own `walk_if`), applied here to the
/// expression form instead of duplicating it. `sample if sample is not
/// None else 0.0` forks on `sample is not None`: the true fork narrows
/// `sample` (its possibly-absent tag drops) before `ternary.body` reads
/// it, and the false fork narrows it the other way before
/// `ternary.orelse` reads it. A decided test still narrows before
/// picking the one arm it evaluates, since a name the taken arm reads
/// may depend on that same narrowing (an `isinstance`-proved sort, a
/// walrus-bound comparison, …).
fn evaluate_ternary(ternary: &ruff_python_ast::ExprIf, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let test = evaluate_expression(&ternary.test, environment, kernel);
    let (value, known) = truthiness(&test);
    if known {
        return if value {
            let body_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, true);
            evaluate_expression(&ternary.body, &body_environment, kernel)
        } else {
            let orelse_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, false);
            evaluate_expression(&ternary.orelse, &orelse_environment, kernel)
        };
    }
    let body_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, true);
    let orelse_environment = narrowing::assume(&ternary.test, environment.fork(), kernel, false);
    let body = evaluate_expression(&ternary.body, &body_environment, kernel);
    let orelse = evaluate_expression(&ternary.orelse, &orelse_environment, kernel);
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
            // a view SHARES the underlying buffer (module doc) — writing
            // through the view raises the memoryview-specific wording
            // regardless of which species the wrapped argument itself
            // carried, so this re-tags rather than keeping the argument's
            // own word (`bytes_models::tagged`'s own doc).
            return Some(bytes_models::tagged(argument, bytes_models::MEMORYVIEW_WORD));
        }
        return None;
    }
    if constructor == "bytearray" {
        if let Some((length, PrimitiveKind::Integer)) = single_numeric_value(&argument) {
            if (0.0..=1024.0).contains(&length) {
                let zeroes = vec![0u8; length as usize];
                return Some(bytes_models::tagged(
                    bytes_models::bytes_literal_value(&zeroes),
                    bytes_models::BYTEARRAY_WORD,
                ));
            }
            return None;
        }
    }
    let bytes = known_byte_sequence(&argument)?;
    let word = if constructor == "bytearray" {
        bytes_models::BYTEARRAY_WORD
    } else {
        bytes_models::BYTES_WORD
    };
    Some(bytes_models::tagged(bytes_models::bytes_literal_value(&bytes), word))
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

/// Which datetime construct a local name means, read once from the
/// module's own `import`/`from … import …` statements — the same
/// "one import table, read once" mechanism `surface::SurfaceImports`
/// already carries for the pydantic surface (`surface_imports`'s own
/// doc), scoped here to the `datetime` module family so this file's
/// gates answer by CANONICAL identity rather than the literal spelling
/// `datetime`/`date`/`timedelta`. Three shapes recognize:
/// `import datetime[ as x]` (`module_names`, `x` means the WHOLE
/// module — `x.datetime`/`x.date`/`x.timedelta` all still resolve
/// through it), `from datetime import datetime[ as x]`/`date[ as
/// x]`/`timedelta[ as x]` (each lands in its own class-name set, `x`
/// alone now means that ONE class, no further attribute needed), and
/// no import at all (every set stays empty, and `datetime_imports`'s
/// caller falls back to the literal `datetime.*` spelling unchanged —
/// datetime.rst's classes are named `datetime`/`date`/`timedelta`
/// either way, so a module with no explicit `datetime` import still
/// reads its bare `datetime.date(...)` calls the same as before this
/// table existed).
#[derive(Default)]
pub struct DatetimeImports {
    module_names: HashSet<String>,
    datetime_class_names: HashSet<String>,
    date_class_names: HashSet<String>,
    timedelta_class_names: HashSet<String>,
}

/// Reads `module`'s top-level `import`/`from … import …` statements
/// into a `DatetimeImports` table (see that struct's own doc for the
/// three recognized shapes). Anything else — a re-export, a
/// submodule import (`import datetime.date`, not a real Python
/// shape for this stdlib module anyway), a star import — is out of
/// scope and leaves the corresponding set empty, the same "recognize
/// only the shapes the mission names" discipline `surface_imports`
/// already keeps.
pub(crate) fn datetime_imports(module: &ModModule) -> DatetimeImports {
    let mut table = DatetimeImports::default();
    for stmt in module.body.iter() {
        match stmt {
            Stmt::Import(import) => {
                for alias in &import.names {
                    if alias.name.id.as_str() == "datetime" {
                        let local = alias.asname.as_ref().unwrap_or(&alias.name);
                        table.module_names.insert(local.id.as_str().to_owned());
                    }
                }
            }
            Stmt::ImportFrom(import) => {
                let Some(source) = import.module.as_ref() else {
                    continue;
                };
                if source.id.as_str() != "datetime" || import.level != 0 {
                    continue;
                }
                for alias in &import.names {
                    let local = alias.asname.as_ref().unwrap_or(&alias.name);
                    match alias.name.id.as_str() {
                        "datetime" => {
                            table.datetime_class_names.insert(local.id.as_str().to_owned());
                        }
                        "date" => {
                            table.date_class_names.insert(local.id.as_str().to_owned());
                        }
                        "timedelta" => {
                            table.timedelta_class_names.insert(local.id.as_str().to_owned());
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    table
}

/// Whether `callee` names the `datetime.datetime` class, NOT locally
/// shadowed — resolved by CANONICAL import identity through
/// `environment`'s own `DatetimeImports` table (`datetime_imports`'s
/// own doc) rather than the literal spelling. `callee` is the exact
/// expression a caller wants to prove IS the `datetime.datetime`
/// class — either the CONSTRUCTION call's own callee
/// (`datetime.datetime(...)`) or a classmethod call's own RECEIVER
/// (`datetime.datetime.now()`'s `datetime.datetime`). Two shapes
/// recognize: the qualified attribute chain `datetime.datetime`/
/// `dtm.datetime` (any local name the table's `module_names`
/// resolved to the whole module), and the bare aliased class name
/// (`dt`, from `from datetime import datetime as dt` — the table's
/// `datetime_class_names`). A module with no `DatetimeImports` table
/// at all (`environment.datetime_imports()` answers `None` — a test
/// environment, or a walk that never set one) falls back to the
/// literal `datetime.datetime` spelling only for the qualified shape,
/// and never recognizes a bare name — matching this function's own
/// behavior before the table existed. Shadowing is checked the same
/// way either shape already did: the resolved base name must read
/// `None` from `environment`'s own bindings — a body that locally
/// rebinds `datetime`/`dtm`/`dt` to some other value shadows the
/// import regardless of which spelling reached it.
fn is_datetime_datetime_attribute(callee: &Expr, environment: &Environment) -> bool {
    if let Expr::Attribute(attribute) = callee {
        if attribute.attr.as_str() == "datetime" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if environment.read(module_name.id.as_str()).is_some() {
                    return false;
                }
                if let Some(imports) = environment.datetime_imports() {
                    return imports.module_names.contains(module_name.id.as_str());
                }
                return module_name.id.as_str() == "datetime";
            }
        }
        return false;
    }
    let Expr::Name(name) = callee else {
        return false;
    };
    let Some(imports) = environment.datetime_imports() else {
        return false;
    };
    imports.datetime_class_names.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none()
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

/// `subprocess.run([...], ..., capture_output=True, text=True)` —
/// library/subprocess.rst, `class:: CompletedProcess`: "args, returncode,
/// stdout, stderr" are the instance's own attributes, and `run`'s own
/// entry states `capture_output=True` sets `stdout`/`stderr` to
/// `PIPE`, while `text=True` (an alias for `universal_newlines`) makes
/// every captured stream "opened in text mode" — a `str`, never `bytes`.
/// Modeled ONLY as far as `.stdout`'s own SORT: an OBJECT (`Kind::Object`,
/// untagged `source` — the same untagged shape `cross_module.rs`'s own
/// module object carries, so `evaluate_attribute_read`'s tail falls
/// straight to the plain `instances::field_read` linear scan) with one
/// `ObjectKey` named `stdout`, holding the whole-strings ground
/// (`codepoint_sets::strings()`, `C*`) — the same untagged String-sorted
/// `Kind::Set` `__name__` reads (this file's own `Expr::Name` arm). No
/// OTHER `CompletedProcess` field (`returncode`, `stderr`, `args`) is
/// modeled: this row exists to give `.stdout` a SORT for a body that
/// reads it some other way than `json.loads(...)` (`foreign_edge.rs`'s
/// own `json.loads(result.stdout)` consumer path owns that shape
/// separately, and runs BEFORE this construction ever matters — a
/// recognized foreign edge overrides its own consumer node directly;
/// this row only affects `result` itself and every OTHER read of it,
/// `d-data-legs.py`'s own `level_via_raw_stdout`: `float(result.stdout)`,
/// never parsed as JSON).
///
/// Declines (`None`) unless the module name is `subprocess` (not locally
/// shadowed — the same check every other `subprocess`/module recognizer
/// in this crate applies), the attribute called is `run`, and BOTH
/// `capture_output=True` and `text=True` appear among the call's
/// keywords: away from that exact pair, `.stdout` is not provably a
/// `str` at all (no `capture_output=True` leaves stdout un-captured
/// entirely; no `text=True` leaves it `bytes`), so the whole construction
/// declines rather than guess the sort.
fn subprocess_run_construction_value(attribute: &ruff_python_ast::ExprAttribute, call: &ruff_python_ast::ExprCall, environment: &Environment) -> Option<AbstractValue> {
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "subprocess" || environment.read("subprocess").is_some() {
        return None;
    }
    if attribute.attr.as_str() != "run" {
        return None;
    }
    let mut capture_output_true = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return None;
        };
        match name.as_str() {
            "capture_output" => capture_output_true = foreign_edge::literal_true(&keyword.value),
            "text" => text_true = foreign_edge::literal_true(&keyword.value),
            _ => {}
        }
    }
    if !capture_output_true || !text_true {
        return None;
    }
    let keys = vec![ObjectKey { name: "stdout".to_owned(), numeric: false, value: known_set(strings(), None, TrustSpec, SetKindTag::None) }];
    Some(known_object(keys, None, true, TrustSpec, false))
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
/// triple to the POSIX epoch (1970-01-01 = day 0), asked of the kernel's
/// `calendar` seam (`refined_calendar`'s `"epochDays"` op,
/// `theories/calendar/epoch_days.lean`'s `isoDateToEpochDays`, the SAME
/// anchor `datetime_timestamp_value`'s own doc already cited: day 0 is
/// `date(1970, 1, 1).toordinal()`). The kernel validates the date
/// itself (`isValidISODate`) and the PlainDate day-range limit
/// (`epochDaysWithinLimits`) before answering, so an out-of-range or
/// invalid civil date is a caught refusal here (`ask_kernel`'s
/// `catch_unwind`), not a value this function returns — `None` in that
/// case, matching every other refused kernel ask in this crate.
fn epoch_days_of_civil_date(year: i64, month: i64, day: i64, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::EpochDays,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("days")?.as_i64()
}

/// `<an aware-UTC datetime_datetime instance>.timestamp()` — the EXACT
/// POSIX timestamp: datetime.rst, `method:: datetime.timestamp()`, "For
/// aware datetime instances, the return value is computed as: `(dt -
/// datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()`." UTC has
/// no DST/leap-second adjustment, so that difference reduces to plain
/// calendar-day arithmetic (`epoch_days_of_civil_date`'s kernel ask)
/// times 86400, plus the wall-clock seconds-of-day. Modeled ONLY for a
/// `datetime_construction_value`-tagged instance whose own `aware_utc`
/// field is `true` — `None` for a NAIVE instance (datetime.rst's own
/// note: "Naive datetime instances are assumed to represent local time
/// and this method relies on the platform C mktime function," a
/// host-dependent conversion this file does not claim to reproduce).
fn datetime_timestamp_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
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
    let days = epoch_days_of_civil_date(year, month, day, kernel)?;
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

/// Whether `callee` names the `datetime.date` class, NOT locally
/// shadowed — `date.1`'s own receiver shape, resolved by CANONICAL
/// import identity the same way `is_datetime_datetime_attribute` is
/// for the sibling `datetime` class (that function's own doc — the
/// qualified chain `datetime.date`/`dtm.date` OR the bare aliased
/// name `from datetime import date[ as x]` gave `x`). Gates both the
/// `datetime.date(...)` CONSTRUCTION call and the
/// `datetime.date.fromisoformat(...)` CLASSMETHOD call's own receiver
/// (datetime.rst, `class:: date(year, month, day)`).
fn is_datetime_date_attribute(callee: &Expr, environment: &Environment) -> bool {
    if let Expr::Attribute(attribute) = callee {
        if attribute.attr.as_str() == "date" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if environment.read(module_name.id.as_str()).is_some() {
                    return false;
                }
                if let Some(imports) = environment.datetime_imports() {
                    return imports.module_names.contains(module_name.id.as_str());
                }
                return module_name.id.as_str() == "datetime";
            }
        }
        return false;
    }
    let Expr::Name(name) = callee else {
        return false;
    };
    let Some(imports) = environment.datetime_imports() else {
        return false;
    };
    imports.date_class_names.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none()
}

/// Whether `callee` names the `datetime.timedelta` class, NOT locally
/// shadowed — date.5's own receiver shape, resolved by CANONICAL
/// import identity the same way `is_datetime_datetime_attribute` is
/// for the sibling `datetime` class (that function's own doc — the
/// qualified chain `datetime.timedelta`/`dtm.timedelta` OR the bare
/// aliased name `from datetime import timedelta[ as x]` gave `x`).
/// Gates the `datetime.timedelta(days=n)` CONSTRUCTION call
/// (datetime.rst, `class:: timedelta(days=0, ...)`).
fn is_datetime_timedelta_attribute(callee: &Expr, environment: &Environment) -> bool {
    if let Expr::Attribute(attribute) = callee {
        if attribute.attr.as_str() == "timedelta" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if environment.read(module_name.id.as_str()).is_some() {
                    return false;
                }
                if let Some(imports) = environment.datetime_imports() {
                    return imports.module_names.contains(module_name.id.as_str());
                }
                return module_name.id.as_str() == "datetime";
            }
        }
        return false;
    }
    let Expr::Name(name) = callee else {
        return false;
    };
    let Some(imports) = environment.datetime_imports() else {
        return false;
    };
    imports.timedelta_class_names.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none()
}

/// datetime.rst:88,94 — `MINYEAR` is 1, `MAXYEAR` is 9999 (date.2's own
/// row): "every `date`/`datetime` year satisfies `MINYEAR <= year <=
/// MAXYEAR`." The kernel's OWN range check (`epochDaysWithinLimits`,
/// Temporal's PlainDate window, roughly ±271821 years) is far WIDER
/// than Python's — date.2's row states this directly ("narrower than
/// Temporal's PlainDate day-range limit the JS kernel elects"), and the
/// kernel's `validDate`/`isoDate` ops enforce ONLY their own wider bound
/// (or, for `validDate`, no year bound at all — `isValidISODate` checks
/// month/day-of-month only). Every `datetime_date` construction path in
/// this file therefore asks the kernel's OWN `pyYearInRange` op
/// (`exports_calendar.lean`'s `"pyYearInRange"` arm, `Refinements.
/// pyYearInRange`, `languages/python/dates_durations/year_range.lean`)
/// — one wrapper, three call sites (`date_construction_value`,
/// `date_fromisoformat_value`, `date_shifted_by_timedelta`) unchanged.
/// `None` on a refused ask, matching every other kernel ask in this
/// crate.
fn python_year_in_range(year: i64, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::PyYearInRange,
            year,
            month: 0,
            day: 0,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("valid")?.as_bool()
}

/// `datetime.date(year, month, day)` — a tagged `Kind::Object` (`source =
/// "datetime_date"`) carrying `year`/`month`/`day` Integer `ObjectKey`s.
/// datetime.rst, `class:: date(year, month, day)`: all three arguments
/// are REQUIRED, positional-or-keyword, no defaults — unlike
/// `datetime_construction_value`'s `hour`/`minute`/`second`, a missing
/// field here declines the whole construction rather than defaulting.
/// Validated through TWO kernel asks: `calendar.validDate` (date.1's own
/// seam) for calendar correctness (month/day-of-month), and
/// `python_year_in_range`'s own `pyYearInRange` ask for date.2's
/// `MINYEAR`/`MAXYEAR` window (see that function's own doc for why
/// `validDate` alone does not cover it) — a year/month/day combination
/// either ask refuses answers `None`.
fn date_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let field_names = ["year", "month", "day"];
    let mut fields: Vec<Option<i64>> = vec![None; field_names.len()];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        let slot = fields.get_mut(index)?;
        *slot = Some(datetime_field_argument(arg, environment, kernel)?);
    }
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            return None;
        };
        let position = field_names.iter().position(|name| *name == arg_name.as_str())?;
        let slot = fields.get_mut(position)?;
        *slot = Some(datetime_field_argument(&keyword.value, environment, kernel)?);
    }
    let year = fields[0]?;
    let month = fields[1]?;
    let day = fields[2]?;
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    if !valid_civil_date(year, month, day, kernel)? {
        return None;
    }
    let keys = field_names.iter().zip([year, month, day]).map(|(name, value)| integer_object_key(name, value)).collect();
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_date".to_owned();
    Some(instance)
}

/// `calendar.validDate` — date.1's own kernel seam, asked directly
/// (rather than through `epoch_days_of_civil_date`'s `epochDays` op)
/// because construction only needs the `valid` verdict, not a day
/// count. `None` on a refused ask (the kernel panics on no answer;
/// `ask_kernel` catches that the same way `epoch_days_of_civil_date`
/// does), matching every other refused kernel ask in this crate.
fn valid_civil_date(year: i64, month: i64, day: i64, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::ValidDate,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("valid")?.as_bool()
}

/// `datetime.timedelta(days=n)` — a tagged `Kind::Object` (`source =
/// "datetime_timedelta"`) carrying one `days` Integer `ObjectKey`.
/// datetime.rst, `class:: timedelta(days=0, seconds=0, microseconds=0,
/// milliseconds=0, minutes=0, hours=0, weeks=0)`: only the `days`
/// keyword is modeled — a positional argument or any OTHER keyword
/// (`seconds=`, `weeks=`, …) declines the whole construction, matching
/// this crate's `datetime_construction_value` convention of declining
/// rather than guessing at an argument shape it does not read. Validated
/// through the kernel's `calendar.validDuration` ask (date.5's own
/// seam): the ten-field vector is `(years, months, weeks, days, hours,
/// minutes, seconds, milliseconds, microseconds, nanoseconds)`
/// (`theories/calendar/duration.lean`'s own comment) — every field
/// besides `days` is `0` here, so the magnitude/sign guards the kernel
/// checks only ever bind on the one field this file constructs.
fn timedelta_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.args.is_empty() {
        return None;
    }
    let [keyword] = call.arguments.keywords.as_slice() else {
        return None;
    };
    if keyword.arg.as_ref().map(|name| name.as_str()) != Some("days") {
        return None;
    }
    let days = datetime_field_argument(&keyword.value, environment, kernel)?;
    if !valid_duration_days(days, kernel)? {
        return None;
    }
    let instance_keys = vec![integer_object_key("days", days)];
    let mut instance = known_object(instance_keys, None, true, TrustProved, false);
    instance.source = "datetime_timedelta".to_owned();
    Some(instance)
}

/// `calendar.validDuration` asked over a days-only ten-field vector —
/// `timedelta_construction_value`'s own validity gate (date.5's kernel
/// seam), spelled as its own function so the field-order comment lives
/// beside the one call site that builds the vector.
fn valid_duration_days(days: i64, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let fields = vec![0.0, 0.0, 0.0, days as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::ValidDuration,
            year: 0,
            month: 0,
            day: 0,
            days: 0,
            fields,
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    asked.get("valid")?.as_bool()
}

/// `date.fromisoformat("YYYY-MM-DD")` — datetime.rst, `classmethod::
/// date.fromisoformat(date_string)`. Modeled ONLY for the strict
/// `YYYY-MM-DD` shape date.3's own row states as the committed
/// (non-reduced-precision, non-extended, non-ordinal) grammar — a known
/// exact string this file can split by its two ASCII hyphens into three
/// all-digit runs. The parsed year/month/day is then validated through
/// the SAME two kernel asks `date_construction_value` uses —
/// `python_year_in_range`'s `pyYearInRange` for date.2's window, then
/// `calendar.validDate` for calendar correctness — so a syntactically
/// well-shaped but calendrically invalid string (`"2023-02-30"`)
/// declines the same way a bad `datetime.date(...)` construction does.
/// Any other shape (a non-string argument, an unparseable string, a
/// string with the wrong hyphen count or non-digit runs) answers
/// `None`.
fn date_fromisoformat_value(text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let mut parts = text.split('-');
    let year_text = parts.next()?;
    let month_text = parts.next()?;
    let day_text = parts.next()?;
    if parts.next().is_some() {
        return None;
    }
    if year_text.len() != 4 || month_text.len() != 2 || day_text.len() != 2 {
        return None;
    }
    if !year_text.bytes().all(|b| b.is_ascii_digit())
        || !month_text.bytes().all(|b| b.is_ascii_digit())
        || !day_text.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let year: i64 = year_text.parse().ok()?;
    let month: i64 = month_text.parse().ok()?;
    let day: i64 = day_text.parse().ok()?;
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    if !valid_civil_date(year, month, day, kernel)? {
        return None;
    }
    let keys = vec![integer_object_key("year", year), integer_object_key("month", month), integer_object_key("day", day)];
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_date".to_owned();
    Some(instance)
}

/// The kernel's `epochDays` answer for a tagged `datetime_date`
/// instance's own `year`/`month`/`day` fields — `.weekday()` and
/// `.toordinal()`'s shared first step, both riding the SAME kernel ask
/// `epoch_days_of_civil_date` already makes for `datetime_datetime`
/// (this function reads its own `dayOfWeek` field too, which that
/// function's caller never needed). `None` on a refused ask (an
/// out-of-range or invalid date, though a tagged `datetime_date`
/// instance was already validated at construction).
fn epoch_days_and_day_of_week(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<(i64, i64)> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::EpochDays,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let days = asked.get("days")?.as_i64()?;
    let day_of_week = asked.get("dayOfWeek")?.as_i64()?;
    Some((days, day_of_week))
}

/// `date.weekday()` — datetime.rst:687, "Monday is 0 and Sunday is 6."
/// Asks the kernel's `"weekday"` op directly (`exports_calendar.lean`'s
/// `"weekday"` arm, `Refinements.pyWeekday`, `languages/python/
/// dates_durations/weekday.lean`) over the instance's own `year`/
/// `month`/`day` fields — the kernel answers Python's Monday-0 form
/// itself, so this function poses one ask and reads its `weekday` field
/// unchanged, no local arithmetic.
fn date_weekday_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::Weekday,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let weekday = asked.get("weekday")?.as_i64()?;
    Some(known_values(vec![weekday as f64], PrimitiveKind::Integer, TrustProved))
}

/// `date.isoweekday()` — datetime.rst:694-695, "Monday is 1 and Sunday
/// is 7," ONE more than `.weekday()`'s Monday-0 form (both elections
/// walk the same seven days in the same order — the kernel's `"weekday"`
/// arm already IS the Monday-0 answer this method shifts by one).
/// Reuses `date_weekday_value`'s own ask rather than posing a second
/// one: the ISO-1 form has no dedicated kernel arm of its own, and
/// deriving it from the already-asked Monday-0 answer needs no further
/// kernel round trip.
fn date_isoweekday_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let weekday = date_weekday_value(instance, kernel)?;
    let (monday_zero, _) = single_numeric_value(&weekday)?;
    Some(known_values(vec![monday_zero + 1.0], PrimitiveKind::Integer, TrustProved))
}

/// `date.toordinal()` — datetime.rst:525-526, "January 1 of year 1 has
/// ordinal 1." Asks the kernel's `"toordinal"` op directly
/// (`exports_calendar.lean`'s `"toordinal"` arm, `Refinements.
/// pyToOrdinal`, `languages/python/dates_durations/ordinal.lean`) over
/// the instance's own `year`/`month`/`day` fields — the kernel applies
/// the proved `719163` anchor shift itself, so this function poses one
/// ask and reads its `ordinal` field unchanged, no local arithmetic.
fn date_toordinal_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::ToOrdinal,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let ordinal = asked.get("ordinal")?.as_i64()?;
    Some(known_values(vec![ordinal as f64], PrimitiveKind::Integer, TrustProved))
}

/// `date.isocalendar()` — datetime.rst:699-721, the (ISO year, ISO
/// week, ISO weekday) triple. Asks the kernel's `"isoCalendar"` op
/// directly (`exports_calendar.lean`'s `"isoCalendar"` arm,
/// `Refinements.pyIsoCalendar`, `languages/python/dates_durations/
/// iso_week_date.lean`) over the instance's own `year`/`month`/`day`
/// fields, then binds the three answered ints (`isoYear`, `week`,
/// `weekday`) as a known 3-element tuple through
/// `collection_models::tuple_literal_value` — the same constructor
/// `evaluate_tuple` uses for a literal `(a, b, c)` display, so the
/// answer type-checks identically to a real tuple.
fn date_isocalendar_value(instance: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let year = datetime_field(instance, "year")? as i64;
    let month = datetime_field(instance, "month")? as i64;
    let day = datetime_field(instance, "day")? as i64;
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoCalendar,
            year,
            month,
            day,
            days: 0,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let iso_year = asked.get("isoYear")?.as_i64()?;
    let week = asked.get("week")?.as_i64()?;
    let weekday = asked.get("weekday")?.as_i64()?;
    let elements = [iso_year, week, weekday].map(|value| known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved));
    Some(collection_models::tuple_literal_value(&elements))
}

/// date.12 STAGE 1 — the ISO-equivalent directive subset of
/// `datetime.strptime(date_string, format)` (datetime.rst,
/// `classmethod:: datetime.strptime(date_string, format)`: "Return a
/// datetime corresponding to date_string, parsed according to format").
/// Modeled ONLY for the exact literal format `"%Y-%m-%d"` — the ISO
/// `YYYY-MM-DD` directive sequence date.3's grammar already commits to
/// (`%Y` datetime.rst:2413-2415, `%m` :2407-2409, `%d` :2394-2396, each
/// a zero-padded decimal field) — lowered to EXACTLY the same value
/// `date_fromisoformat_value` binds for the identical text: this
/// function reuses that function outright rather than re-deriving its
/// parse or its two kernel asks (`pyYearInRange` then `validDate`), so
/// `strptime(text, "%Y-%m-%d")` and `date.fromisoformat(text)` produce
/// the SAME `AbstractValue` for the same `text` — a `datetime_date`-
/// tagged instance, not a `datetime_datetime` one: the format carries
/// no time-of-day directive, so the honest value this file can prove is
/// calendar-date-shaped, even though CPython's real return type is
/// `datetime`. EXCLUDED from this stage: any `"%H:%M:%S"`-composite
/// format (`"%Y-%m-%d %H:%M:%S"` and similar) — datetime.rst's own
/// `strftime`/`strptime` directive table gives each of `%H`/`%M`/`%S`
/// (:2416-2430) no existing kernel-crossed bind this file's
/// `datetime_datetime` construction path reads FROM a string (only
/// FROM already-known Integer arguments,
/// `datetime_construction_value`'s own doc), so composing them here
/// would invent a new value shape this stage does not build; a non-ISO
/// literal format or a non-literal (computed) format both decline
/// through this function's own caller, never reaching here.
fn strptime_iso_date_value(text: &str, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    date_fromisoformat_value(text, kernel)
}

/// date.12 STAGE 1 — `date.strftime(format)` (datetime.rst, `method::
/// date.strftime(format)`: "Return a string representing the date,
/// controlled by an explicit format string"). Modeled ONLY for the
/// exact literal format `"%Y-%m-%d"` on a tagged `datetime_date`
/// instance whose OWN `year`/`month`/`day` fields are already known —
/// the ISO rendering `date.isoformat()` also produces (datetime.rst:725,
/// "ISO 8601 format, YYYY-MM-DD"). Declines (`None`) rather than binds
/// an exact string: the kernel's `isoDate` op (`exports_calendar.lean`'s
/// `"isoDate"` arm) answers `{year, month, day, dayOfWeek}` — four
/// INTEGER fields, no rendered string field of any kind — so there is
/// no kernel ask this function can pose for the digit-string render the
/// way `strptime_iso_date_value` poses one for the parse direction. The
/// render direction needs either a kernel export of the zero-padded
/// `YYYY-MM-DD` string form (a NEW `isoDate`/`epochDays` answer field,
/// or a dedicated render op) or an adapter-local zero-pad composition
/// of the three already-known integer fields — out of this stage's
/// scope, which reuses existing kernel asks only.
fn strftime_iso_date_value(_instance: &AbstractValue) -> Option<AbstractValue> {
    None
}

/// `date1 ± timedelta` — datetime.rst's operation table (date.7's own
/// row): shifts by `timedelta.days` (the only field
/// `timedelta_construction_value` ever populates) and answers a NEW
/// tagged `datetime_date` instance, or declines (`None`) exactly where
/// CPython raises `OverflowError`. The kernel's `epochDays`/`isoDate`
/// pair (date.1's seam) computes the shifted day count and certifies it
/// lands back on a calendrically valid date (`isoDate`'s own
/// "self-certification" — `exports_calendar.lean`'s comment), but that
/// certification alone is NOT date.7's `OverflowError` bound: the
/// kernel's own PlainDate window is far wider than Python's
/// `[MINYEAR, MAXYEAR]`, so this function additionally poses the
/// `pyYearInRange` ask (`python_year_in_range`) on the shifted result —
/// a shift the kernel's `isoDate` arm would happily answer but Python
/// would reject (`date(9999, 12, 31) + timedelta(days=1)`, landing on
/// year 10000) still declines here. `negate` flips the shift for
/// `date - timedelta` (`date + timedelta` passes `false`).
fn date_shifted_by_timedelta(date: &AbstractValue, timedelta: &AbstractValue, negate: bool, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let (days, _) = epoch_days_and_day_of_week(date, kernel)?;
    let shift = datetime_field(timedelta, "days")? as i64;
    let shifted_days = if negate { days - shift } else { days + shift };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.calendar)(&CalendarQuestion {
            op: CalendarQuestionOp::IsoDate,
            year: 0,
            month: 0,
            day: 0,
            days: shifted_days,
            fields: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    })
    .ok()?;
    let year = asked.get("year")?.as_i64()?;
    let month = asked.get("month")?.as_i64()?;
    let day = asked.get("day")?.as_i64()?;
    if !python_year_in_range(year, kernel)? {
        return None;
    }
    let keys = vec![integer_object_key("year", year), integer_object_key("month", month), integer_object_key("day", day)];
    let mut instance = known_object(keys, None, true, TrustProved, false);
    instance.source = "datetime_date".to_owned();
    Some(instance)
}

/// A retained-callable call's own positional arguments, given `def`'s
/// synthetic parameter list — tries `positional_arguments_for_def`'s
/// existing exact mapping FIRST (the ordinary, no-splat call shape
/// every other row uses), and only when THAT declines because the
/// call site carries a `Starred` positional argument (`f(*args,
/// **kwargs)`, r-ast-census.py's own `wrapper`: a ParamSpec-forwarding
/// body handing its own received `*args`/`**kwargs` straight to the
/// retained callable it wraps) tries splicing instead: `*args`
/// splices through `splice_call_arguments` (a known `Kind::List`
/// receiver only — the same honest decline on an unbounded iterable
/// that function's own doc states), and a `**kwargs`-spread keyword
/// argument (`keyword.arg.is_none()`) reads its own known `Kind::
/// Object` entries, mapping each by NAME onto `def`'s own parameter
/// list — the same by-name mapping `positional_arguments_with_kwargs_
/// dict` gives an ordinary named keyword, extended to a spread rather
/// than a single name. A `**kwargs` value that is not a known
/// `Kind::Object`, or an entry naming no parameter of `def`, declines
/// the whole call — this reader guesses at neither shape.
fn positional_arguments_for_retained_call(
    call: &ruff_python_ast::ExprCall,
    def: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if let Some(mapped) = positional_arguments_for_def(call, def, environment, kernel) {
        return Some(mapped);
    }
    let has_starred_positional = call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_)));
    let has_kwargs_spread = call.arguments.keywords.iter().any(|keyword| keyword.arg.is_none());
    if !has_starred_positional && !has_kwargs_spread {
        return None;
    }
    let parameter_names: Vec<&str> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
        .map(|parameter| parameter.parameter.name.id.as_str())
        .collect();
    let mut positional = splice_call_arguments(&call.arguments.args, environment, kernel)?;
    for keyword in &call.arguments.keywords {
        match keyword.arg.as_ref() {
            Some(arg_name) => {
                let position = parameter_names.iter().position(|name| *name == arg_name.as_str())?;
                if position < positional.len() {
                    positional[position] = evaluate_expression(&keyword.value, environment, kernel);
                } else {
                    positional.resize_with(position + 1, unknown);
                    positional[position] = evaluate_expression(&keyword.value, environment, kernel);
                }
            }
            None => {
                let spread = evaluate_expression(&keyword.value, environment, kernel);
                if spread.kind != Kind::Object {
                    return None;
                }
                for entry in &spread.keys {
                    let position = parameter_names.iter().position(|name| *name == entry.name.as_str())?;
                    if position < positional.len() {
                        positional[position] = entry.value.clone();
                    } else {
                        positional.resize_with(position + 1, unknown);
                        positional[position] = entry.value.clone();
                    }
                }
            }
        }
    }
    Some(positional)
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
/// arguments_for_retained_call` — the ordinary same-module keyword-
/// to-position mapping and arity checking, PLUS the one splicing
/// fallback a ParamSpec-forwarding wrapper needs (that function's own
/// doc).
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
    let Some(positional) = positional_arguments_for_retained_call(call, &def, environment, kernel) else {
        return Some(unknown());
    };
    // `enclosing` is ALWAYS the call site's own environment, carried
    // through a throwaway wrapper seeded with the retained body's own
    // closure snapshot (empty for a lambda/def that reads no free
    // name, the common case) — never `None` — so `call_result_with_
    // enclosing`'s own `fresh_body_environment` call always inherits
    // this call site's retained-callable table
    // (`Environment::inherit_retained_callables`'s own doc): a
    // retained value the closure carries (r-ast-census.py's `f`) still
    // resolves through the SAME shared table when `def`'s own body
    // calls it, and a retained value THIS call creates is still
    // reachable from `environment` (and everywhere `environment`'s own
    // `Arc` reaches) once this call returns.
    let mut closure_environment = Environment::new(std::collections::HashSet::new());
    closure_environment.inherit_retained_callables(environment);
    for (name, value) in &retained.closure {
        closure_environment.bind(name, value.clone());
    }
    let answer = summaries::call_result_with_enclosing(
        &def,
        &positional,
        environment.functions(),
        kernel,
        environment.call_depth(),
        Some(&closure_environment),
    );
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
            // TWO SIBLING NESTED DEFS DECLARING THE SAME CLASS NAME
            // (b-body-expressions.py's `binary_chained_builder_call`:
            // `make_ok_builder`/`make_over_builder` each declare their
            // own `class Builder`) collide in `environment.classes()` —
            // the caller's own flat, body-wide table can hold only ONE
            // `"Builder"` entry, whichever `check.rs::local_class_table`
            // happened to see first while pre-scanning the caller's body.
            // A CHAINED call's receiver (`make_over_builder().type("x")`)
            // needs the SPECIFIC sibling's own class, not that shared
            // guess, so `receiver_def_local_classes` re-reads the class
            // straight from the same-module def the receiver expression
            // actually traces back to, fresh, with no sibling to collide
            // against. Tried first; `environment.classes()` still answers
            // every other receiver (an ordinary constructed instance, a
            // parameter, a field read) exactly as before.
            let scoped_classes = receiver_def_local_classes(&attribute.value, environment, kernel);
            let classes_for_call = match &scoped_classes {
                Some(scoped) if scoped.contains_key(receiver.source.as_str()) => Some(scoped),
                _ => environment.classes(),
            };
            if let Some(classes) = classes_for_call {
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
                            environment.datetime_imports(),
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
    // The three datetime CONSTRUCTION gates run against `call.func`
    // itself, BEFORE the `Expr::Attribute`-only block below: each gate
    // (`is_datetime_datetime_attribute` and its two siblings) already
    // recognizes both the qualified chain (`datetime.datetime(...)`, an
    // `Expr::Attribute` callee) AND a bare aliased class name
    // (`dt(...)`, an `Expr::Name` callee — `from datetime import
    // datetime as dt`), so trying it here covers both shapes in one
    // place rather than duplicating the bare-Name arm alongside the
    // Attribute-only recognizers further down.
    if is_datetime_datetime_attribute(call.func.as_ref(), environment) {
        if let Some(value) = datetime_construction_value(call, environment, kernel) {
            return value;
        }
        return unknown();
    }
    // `datetime.date(year, month, day)` — date.1's own construction,
    // recognized the same way `datetime.datetime(...)` is (BEFORE the
    // keyword gate below, though this construction reads no keyword
    // this file's corpus does not already handle positionally). See
    // `date_construction_value`'s own doc for the exact fields read
    // and the `calendar.validDate` kernel validation.
    if is_datetime_date_attribute(call.func.as_ref(), environment) {
        if let Some(value) = date_construction_value(call, environment, kernel) {
            return value;
        }
        return unknown();
    }
    // `datetime.timedelta(days=n)` — date.5's own construction,
    // recognized here (BEFORE the keyword gate below) because
    // `days=` always arrives as a keyword argument. See
    // `timedelta_construction_value`'s own doc for the one field
    // read and the `calendar.validDuration` kernel validation.
    if is_datetime_timedelta_attribute(call.func.as_ref(), environment) {
        if let Some(value) = timedelta_construction_value(call, environment, kernel) {
            return value;
        }
        return unknown();
    }
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        // `datetime.date.fromisoformat("YYYY-MM-DD")` — a TWO-level
        // attribute chain the same way `datetime.datetime.now()` is
        // when `date` reached the file qualified (`datetime.date`), OR
        // ONE level when `date` reached it as a bare aliased class name
        // (`date.fromisoformat(...)`, `from datetime import date`) —
        // `is_datetime_date_attribute` resolves `attribute.value`
        // either way. See `date_fromisoformat_value`'s own doc for the
        // exact grammar read.
        if is_datetime_date_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "fromisoformat" {
            if let [text] = &*call.arguments.args {
                if call.arguments.keywords.is_empty() {
                    let argument = evaluate_expression(text, environment, kernel);
                    if let Some(code_points) = exact_string_values(&argument) {
                        if let Some(spelling) = code_points_to_string(code_points) {
                            if let Some(value) = date_fromisoformat_value(&spelling, kernel) {
                                return value;
                            }
                        }
                    }
                }
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
        // `subprocess.run([...], ..., capture_output=True, text=True)` —
        // tried here, alongside `array.array`, so `result`'s own binding
        // carries a `.stdout` field sort even when no `json.loads(...)`
        // consumer exists for `foreign_edge.rs` to recognize
        // (`subprocess_run_construction_value`'s own doc). A call this
        // row does not recognize (a different callee, a missing
        // `capture_output=True`/`text=True` pair) falls through to the
        // ordinary keyword-gated dispatch below unchanged — this row
        // only ever ADDS a sort to `result`, never removes one the
        // generic path would have given.
        if let Some(value) = subprocess_run_construction_value(attribute, call, environment) {
            return value;
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
            match builtin_models::builtin_call_result_with_kernel(name.id.as_str(), &arguments, kernel) {
                Some(value) => value,
                None => unknown(),
            }
        }
        Expr::Attribute(attribute) => evaluate_attribute_call(attribute, &arguments, environment, kernel),
        _ => unknown(),
    }
}

/// The class table a chained method call's RECEIVER expression should
/// resolve its instance's class against, read fresh from the SPECIFIC
/// same-module def the receiver traces back to — never the caller's own
/// shared `environment.classes()`, which can hold only one entry per
/// bare class name and so cannot tell two sibling nested defs' own
/// same-named classes apart (`check.rs::local_class_table`'s own doc:
/// "a class nested inside a NESTED def... is collected too", flattened
/// into one map, first-scanned-wins on a spelling collision).
///
/// `receiver` is peeled one layer at a time: an `Attribute` reads
/// through to its own `.value` (`make_over_builder().type("x")`'s
/// receiver, for the `.size(1)` call, is `make_over_builder().type("x")`
/// itself — another Attribute call, not yet the root), a `Call` whose
/// callee is a bare `Name` naming a same-module def (`environment.
/// functions()`, which already carries every LOCAL nested def merged
/// over the module's own top-level ones — `check.rs::local_function_
/// table`'s own doc) is the root: that def's own body is rescanned for
/// its OWN top-level classes, mirroring `summaries::interpret_class_def`'s
/// exact synthetic-module construction (empty aliases/imports — a
/// body-local class's own field annotations reading a module-level
/// alias is a narrower, still-sound miss the same way that function's
/// own doc already accepts). `None` for every other receiver shape (an
/// ordinary bound name, a field read, a call to anything but a
/// same-module def) — the caller falls back to `environment.classes()`
/// unchanged.
fn receiver_def_local_classes(
    receiver: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<std::sync::Arc<std::collections::HashMap<String, instances::ClassModel>>> {
    match receiver {
        Expr::Attribute(attribute) => receiver_def_local_classes(&attribute.value, environment, kernel),
        // `make_over_builder().type("x")` (the `.size(1)` call's own
        // receiver) is ITSELF a Call whose callee is an Attribute, not
        // yet the root — peel through `.func`'s own receiver the same
        // way the `Expr::Attribute` arm above peels `.value`, so a chain
        // of any length still traces back to the one `Name` call that
        // started it.
        Expr::Call(call) if matches!(call.func.as_ref(), Expr::Attribute(_)) => {
            receiver_def_local_classes(call.func.as_ref(), environment, kernel)
        }
        Expr::Call(call) => {
            let Expr::Name(name) = call.func.as_ref() else {
                return None;
            };
            let def = environment.functions()?.def(name.id.as_str())?;
            let synthetic = ruff_python_ast::ModModule {
                node_index: ruff_python_ast::AtomicNodeIndex::NONE,
                range: TextRange::default(),
                body: def.body.iter().filter(|stmt| matches!(stmt, ruff_python_ast::Stmt::ClassDef(_))).cloned().collect(),
            };
            let empty_aliases = std::collections::HashMap::new();
            let empty_imports = crate::surface::surface_imports(&ruff_python_ast::ModModule {
                node_index: ruff_python_ast::AtomicNodeIndex::NONE,
                range: TextRange::default(),
                body: Vec::new().into(),
            });
            Some(std::sync::Arc::new(instances::class_table(&synthetic, &empty_aliases, &empty_imports, kernel)))
        }
        _ => None,
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

/// `json.loads`'s full return space over an operand this file holds no
/// fact about (ISSUES.md, "generic json.loads of an opaque string
/// answers bare unknown") — library/json.rst's own conversion table,
/// read as ONE honest claim rather than the narrower Float-sorted guess
/// the survey rejected as unsound (a real payload can land on any of
/// the table's rows, and a Float-only claim is false on every other
/// row). `PrimitiveKind::Integer`/`Float` split the JSON `number`
/// production (CPython: `json.loads("1")` is `int`, `json.loads("1.5")`
/// is `float` — `json_scalar_literal_value`'s own doc), so each numeric
/// sort narrows on its own via the ordinary Integer/Float narrowing and
/// judging paths, rather than folding both under the sort-unknown
/// `PrimitiveKind::Number` tag that `isinstance`/`judge` cannot yet
/// place on either side of a test. `str`/`list`/`dict`/`bool`/`None`
/// each carry their own sort, so a downstream `isinstance` or judge
/// call can still tell them apart from the numeric arms — a `list`/
/// `dict` arm is built via `opaque_value` (this file's own established
/// "the kind of thing is known, its contents are not" shape, e.g. the
/// `re.match` result above) rather than an exact-arity `known_list([])`/
/// `known_object([])`, which would falsely claim the parsed value is
/// EMPTY.
fn json_loads_value_space() -> AbstractValue {
    kind_union_of(vec![
        null_value(),
        known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec),
        known_set(strings(), None, TrustSpec, SetKindTag::None),
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(eval_whole_integers(), None, TrustSpec, SetKindTag::None)
        },
        float_sorted_unknown(),
        opaque_value("a list"),
        opaque_value("a dict"),
    ])
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

/// Every module name `evaluate_attribute_call` carries a model for, at
/// least in part — the recognized-module gate every arm in that function
/// already applies one at a time (`module_name.id.as_str() == "math"`,
/// `== "random"`, and so on). Named here as ONE list so a recognizer that
/// needs the COMPLEMENT (rung 1's naming unit, and the manifest reader's
/// own "is this module already modeled here?" check,
/// `python-c-extension-boundary.md`'s build order) reads one table
/// instead of re-deriving it from the arms below. `datetime`'s own three
/// aliases (`date`/`timedelta`) are matched by IDENTITY through
/// `environment.datetime_imports()`, not by this literal list, so they
/// are named here too even though no arm below tests
/// `module_name.id.as_str() == "datetime"` directly.
const MODELED_MODULE_NAMES: &[&str] =
    &["math", "random", "re", "json", "importlib", "types", "weakref", "asyncio", "array", "subprocess", "datetime"];

/// The leftmost `Name` under an attribute-chain receiver (`a.b.c` → `a`;
/// `a` itself → `a`) — `None` when the receiver is not built from a
/// plain name chain at all. The expression-side twin of `check.rs`'s own
/// `receiver_base_name` (private to that file), duplicated rather than
/// exported across the crate boundary this module already keeps thin —
/// both copies read the identical two-line recursion.
fn attribute_chain_root_name(receiver: &Expr) -> Option<&str> {
    match receiver {
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Attribute(attribute) => attribute_chain_root_name(attribute.value.as_ref()),
        _ => None,
    }
}

/// Rung 1 of the compiled-extension recognition ladder
/// (`python-c-extension-boundary.md`'s naming unit): whether `call` is a
/// call on an attribute chain rooted at an imported-but-unmodeled module
/// name — `torch.arange(5)`, `pandas.read_csv(...).head()`'s own
/// receiver — answering that root module's own name for the caller to
/// name in its decline sentence.
///
/// Recognized the same way every modeled-module arm in
/// `evaluate_attribute_call` recognizes ITS OWN module: the chain's root
/// is a bare `Name` that reads UNBOUND in `environment`
/// (`environment.read(name).is_none()` — the identical gate `math`/`re`/
/// `json`/etc. already apply, since an import that resolved to nothing
/// this checker tracks leaves the name unbound, `check.rs::
/// bind_or_forget_imported_name`'s own doc) AND is not itself one of the
/// `MODELED_MODULE_NAMES` this file already carries a model for (a
/// modeled module's own unmodeled FUNCTION — `math.frexp`, say — is a
/// different, narrower gap this naming unit does not claim; only a
/// module with NO model at all is named here). `None` for every other
/// call shape: a bare-name call, a method call on an evaluated (non-
/// module) receiver, or a call whose root name IS bound to a real
/// tracked value (shadowing the module the way every existing arm's own
/// gate already respects).
pub fn unmodeled_module_call_name<'a>(call: &'a ruff_python_ast::ExprCall, environment: &Environment) -> Option<&'a str> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let root = attribute_chain_root_name(attribute.value.as_ref())?;
    if environment.read(root).is_some() {
        return None;
    }
    if MODELED_MODULE_NAMES.contains(&root) {
        return None;
    }
    Some(root)
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
            if let Some(value) = math_models::math_call_result(attribute.attr.as_str(), arguments, kernel) {
                return value;
            }
            // `math_call_result` declined — for the eight domain-limited
            // names (`log`/`log2`/`log10`/`log1p`/`asin`/`acos`/`atanh`/
            // `acosh`), a STRADDLING operand still determines a value
            // over its served half, alongside the fire `possible_raise`
            // (`domain_limited_family_possible_raise`) pushes at the
            // sink — the same "the finding and the value both stand"
            // split `split_divisor_transfer` keeps for a sometimes-zero
            // divisor. An entirely-raising or unreadable operand still
            // answers `unknown()` here, unchanged.
            if let Some(family) = math_models::DomainLimitedFamily::of_function(attribute.attr.as_str()) {
                if let [only] = arguments {
                    if let Some(value) = math_models::domain_raise_served_half_value(family, only, kernel) {
                        return value;
                    }
                }
            }
            return unknown();
        }
        // `random.random()` — the sound `[0.0, 1.0)` range
        // (`math_models::random_call_result`'s own doc, citing
        // library/random.rst). Only this one function of the module is
        // modeled; every other `random.*` call falls through to the
        // generic unmodeled-call path below.
        if module_name.id.as_str() == "random" && environment.read("random").is_none() {
            if let Some(value) = math_models::random_call_result(attribute.attr.as_str(), arguments) {
                return value;
            }
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
        // cites). Modeled for a known exact-string `s` whose text is one
        // of the JSON SCALAR productions this file parses by hand
        // (`json_scalar_literal_value`'s own doc: an integer, a float, a
        // quoted string, `true`/`false`/`null`) — the corpus's own rows
        // never need array/object parsing, so that grammar is not built.
        // An `s` this file holds no fact about (an opaque string — the
        // ISSUES.md b-runners:124 row) answers `json_loads_value_space`
        // instead of bare `unknown()`: every shape `loads` can return is
        // ONE determined claim, never a narrower guess this file cannot
        // back (a Float-sorted answer would be false whenever the real
        // payload is a dict/list/str/bool/None).
        if module_name.id.as_str() == "json" && environment.read("json").is_none() {
            if attribute.attr.as_str() == "loads" {
                if let [text] = arguments {
                    if let Some(text) = exact_string_values(text).and_then(code_points_to_string) {
                        if let Some(value) = json_scalar_literal_value(text.trim()) {
                            return value;
                        }
                    }
                }
                return json_loads_value_space();
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
    // `datetime.datetime.now()` — the receiver (`attribute.value`) is
    // either a TWO-level attribute chain (`Attribute(value=Attribute
    // (value=Name("datetime"), attr="datetime"), attr="now")`) when
    // `datetime` reached the file qualified, or a bare aliased class
    // name (`dt.now()`, `from datetime import datetime as dt`) ONE
    // level — never reaching `is_datetime_datetime_attribute`'s own
    // CONSTRUCTION-callee use (that check ALSO gates
    // `datetime.datetime(...)`, whose `call.func` IS the receiver
    // chain itself; here `attribute` is one level further out,
    // `datetime.datetime.now`/`dt.now`). classmethod:: datetime.now
    // (tz=None): "Return the current local date and time." — a value
    // that changes every run, never a whole number Age could ever
    // admit (this fixture's own row's reason: "the current moment is
    // not in the set"); answered OPAQUE, the same "not a scalar/set
    // this domain models" honesty every other host-nondeterministic
    // read in this file already carries. The `tz=` argument (if any)
    // is not read — every outcome is equally opaque regardless of
    // which timezone the caller requests.
    if is_datetime_datetime_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "now" {
        return opaque_value("the current datetime");
    }
    // `datetime.datetime.strptime(date_string, format)` — date.12
    // STAGE 1, the SAME receiver shape `.now()` reads just above
    // (qualified chain or bare aliased class name). Modeled ONLY when
    // BOTH arguments are known exact strings AND `format` is EXACTLY
    // the literal `"%Y-%m-%d"` (`strptime_iso_date_value`'s own doc —
    // the ISO date-only directive sequence date.3's grammar already
    // commits to). A NON-literal `format` (a parameter, a computed
    // expression, an f-string) is not a string this file can read the
    // DIRECTIVES of at all — this file has no format-code mini-language
    // reader for anything but the exact `"%Y-%m-%d"` spelling — so it
    // declines the same way `date_fromisoformat_value`'s own
    // non-literal-argument row does: no sentence-carrying channel
    // exists on this dispatch path (`evaluate_attribute_call`
    // returns a plain `AbstractValue`, never a message), so the
    // decline is named here, in this comment, the same way every
    // other declining recognizer in this file states its reason in
    // prose beside its own `return unknown()`. A literal format
    // OTHER than `"%Y-%m-%d"` (`"%d/%m/%Y"`, any other directive
    // sequence) names date.12 STAGE 2 — the directive-grammar
    // kernel theory this stage does not build — as its own reason,
    // by the identical convention.
    if is_datetime_datetime_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "strptime" {
        if let [text, format] = arguments {
            if let (Some(text_points), Some(format_points)) = (exact_string_values(text), exact_string_values(format)) {
                if let (Some(text_spelling), Some(format_spelling)) = (code_points_to_string(text_points), code_points_to_string(format_points)) {
                    if format_spelling == "%Y-%m-%d" {
                        return match strptime_iso_date_value(&text_spelling, kernel) {
                            Some(value) => value,
                            None => unknown(),
                        };
                    }
                }
            }
        }
        return unknown();
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
            return match datetime_timestamp_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isoformat" {
            return opaque_value("an ISO 8601 datetime string");
        }
    }
    // A tagged `datetime_date` instance's own METHODS — `.weekday()`
    // (date.8, Monday 0), `.isoweekday()` (date.8, Monday 1),
    // `.toordinal()` (date.9), `.isocalendar()` (date.10) — each exact,
    // each posing its own dedicated kernel ask directly (see
    // `date_weekday_value`/`date_toordinal_value`/
    // `date_isocalendar_value`'s own docs for the exact op).
    if receiver.kind == Kind::Object && receiver.source == "datetime_date" {
        if attribute.attr.as_str() == "weekday" && arguments.is_empty() {
            return match date_weekday_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isoweekday" && arguments.is_empty() {
            return match date_isoweekday_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "toordinal" && arguments.is_empty() {
            return match date_toordinal_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        if attribute.attr.as_str() == "isocalendar" && arguments.is_empty() {
            return match date_isocalendar_value(&receiver, kernel) {
                Some(value) => value,
                None => unknown(),
            };
        }
        // `.strftime(format)` — date.12 STAGE 1. Recognized (the method
        // NAME matches, `format` is a known exact string) so a
        // non-`"%Y-%m-%d"` literal or a computed format can each name
        // their own reason below rather than fall through unrecognized;
        // `strftime_iso_date_value`'s own doc states why even the exact
        // `"%Y-%m-%d"` literal still declines today — the kernel's
        // `isoDate` op answers no rendered-string field, only integers.
        if attribute.attr.as_str() == "strftime" {
            if let [format] = arguments {
                if let Some(format_points) = exact_string_values(format) {
                    if let Some(format_spelling) = code_points_to_string(format_points) {
                        if format_spelling == "%Y-%m-%d" {
                            return match strftime_iso_date_value(&receiver) {
                                Some(value) => value,
                                None => unknown(),
                            };
                        }
                        // a literal format that is not `"%Y-%m-%d"` —
                        // date.12 STAGE 2's own directive-grammar
                        // kernel theory, not built by this stage
                    }
                }
                // a non-literal (computed) format — this file cannot
                // read the directive sequence of an expression it
                // cannot fold to an exact string at all
            }
            return unknown();
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
        // `xs.index(needle)` — stdtypes.rst's Common Sequence Operations
        // table, `s.index(x)`: "index of the first occurrence of x in
        // s." Modeled only on the FOUND leg (a missing needle raises
        // ValueError at runtime instead of returning — that leg is
        // `call_provable_raise`'s own `"index"` row, checked separately
        // against the same `single_pair_equal` equality this row uses,
        // so the two passes agree on exactly which needle is present).
        // Answers the position of the first matching element as an
        // exact Integer.
        if attribute.attr.as_str() == "index" {
            if let [needle] = arguments {
                if let Some(position) = receiver.items.iter().position(|element| single_pair_equal(element, needle) == Some(true)) {
                    return known_values(vec![position as f64], PrimitiveKind::Integer, TrustProved);
                }
            }
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
/// as ONE single-clause shape (exactly one `Comprehension`; a second
/// `for` clause or an `async for` — `is_async` — declines outright; the
/// target a bare `Expr::Name` or a two-name tuple target,
/// `comprehension_target_names`'s own doc) over EITHER of two iterable
/// shapes: a known `Kind::List` of already-known elements (the CONCRETE
/// path, tried first, unchanged from before this function grew a second
/// arm), or an unknown-length sequence known by its element SET
/// (`comprehension_target_and_star_element`'s own doc — a declared/
/// refined parameter with no concrete items, tried only once the
/// concrete path declines). The concrete path forks the environment
/// once per surviving element, binding the target, evaluating every
/// `if` condition in order (a `known&&false` truthiness drops the
/// element; `known&&true` keeps checking the rest; anything not fully
/// known makes the WHOLE comprehension unknown — a single undecidable
/// filter means this file cannot say which elements the real list would
/// contain), then evaluating `elt` on that fork; the collected elements
/// build through `collection_models::list_literal_value`. The star path
/// forks ONCE (`comprehension_star_elements`'s own doc) and answers a
/// star-shaped `Kind::Set`, never a `Kind::List` — a length-unstated
/// result has no exact positional slots to state. Either shape is
/// honest for the same reason: a set's own element-uniqueness and a
/// generator's own lazy-iteration behavior are both invisible to a
/// caller that only ever consumes the sequence via `len()`/`sum()`/a
/// `for`-loop read.
fn evaluate_list_or_set_comp(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if let Some(elements) = comprehension_elements(element_expr, generators, environment, kernel) {
        return collection_models::list_literal_value(&elements);
    }
    if let Some(star) = comprehension_star_elements(element_expr, generators, environment, kernel) {
        return star;
    }
    unknown()
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

/// The single-clause comprehension shape over an UNKNOWN-LENGTH,
/// known-element-set iterable: `Kind::Set` whose only form is the
/// repetition window `as_repetition` reads back
/// (`check.rs::seed_parameters`'s own `list[X]`/`set[X]`/`Sequence[X]`
/// PARAMETER seed builds the bare star, `lo` 0 and `hi` unbounded;
/// `collection_models::star_element_read`'s own doc — the same window
/// shape, read the same way, never a second reader). Every position of
/// a repetition draws from the SAME element set (the grammar's own
/// definition), so there is exactly ONE abstraction to bind the target
/// against and exactly ONE evaluation of `elt` to perform — unlike the
/// concrete path above, which evaluates `elt` once per known item.
/// `None` for the same shape restrictions the concrete path takes (a
/// second `for` clause, `async for`, a target of any other arity), OR
/// when the iterable does not read back as a repetition at all (a
/// union, an unknown value) — the concrete arm and this one are
/// mutually exclusive on `iterable.kind`, so a caller tries the
/// concrete path first and only reaches here on ITS decline. The
/// window's own `{lo, hi}` rides back alongside the element so the
/// caller can restate it on the mapped result.
///
/// The SOURCE NAME — the iterable's own spelling, when `clause.iter` is
/// a plain `Expr::Name` — rides back too, `None` for any other iterable
/// expression (a call, an attribute read, a subscript). The caller uses
/// it to record that the mapped result's own length is proved equal to
/// that name's (`AbstractValue::same_length_as`), which only holds for
/// a plain-name source: an iterable built by an expression has no
/// single binding whose later `len(...)` this value could be tied to.
fn comprehension_target_and_star_element<'a>(
    generators: &'a [ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(Vec<&'a str>, &'a [Expr], AbstractValue, i64, Option<i64>, Option<&'a str>)> {
    let [clause] = generators else {
        return None;
    };
    if clause.is_async {
        return None;
    }
    let target_names = comprehension_target_names(&clause.target)?;
    let source_name = match &clause.iter {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    };
    let iterable = evaluate_expression(&clause.iter, environment, kernel);
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    let repeated = as_repetition(&iterable.set)?;
    // The element's own sort is the SEQUENCE's tag, not re-derived: the
    // sequence value carries `kind_tag` off its declared element sort
    // (`check.rs::seed_parameters`'s sequence-container arm), so peeling
    // one element out of the repetition keeps that same tag rather than
    // rebuilding it — `min_max_scalar_operand` (builtin_models.rs) reads
    // an element pulled this way as a `Kind::Set` operand and needs its
    // own `kind_tag` to answer `min`/`max` over two comprehension-bound
    // names.
    let element = AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(repeated.element, None, TrustSpec, SetKindTag::None)
    };
    Some((target_names, &clause.ifs, element, repeated.lo, repeated.hi, source_name))
}

/// The star-shaped result of a list/set/generator comprehension over an
/// unknown-length, known-element-set iterable
/// (`comprehension_target_and_star_element`'s own doc): binds the
/// target to the ONE element abstraction, evaluates every `if`
/// condition against that single binding, and evaluates `elt` once —
/// there is no per-element enumeration to run since the source length
/// is unstated. `None` when a filter's truthiness cannot be decided FOR
/// THE WHOLE ELEMENT SET (unlike the concrete path, which drops
/// individual elements, a filter here either keeps every position or
/// the comprehension is undecidable, since one shared element stands
/// for all of them). A comprehension preserves the source's own length
/// (mapping every position through `elt` changes no position's
/// presence) — the result carries the SAME `{lo, hi}` window the source
/// read back — UNLESS an `if` clause is present, in which case a filter
/// can drop positions down to zero, so `lo` widens to 0 whenever
/// `conditions` is non-empty; `hi` is unaffected either way (a filter
/// only ever removes positions, never adds them).
///
/// SOUNDNESS LINE for `AbstractValue::same_length_as`: the result's
/// length is proved EQUAL to the source's own length -- not merely
/// bounded by the same window -- only when `conditions` is empty. A
/// filtered comprehension can drop positions, so `len(result) <=
/// len(source)` but never provably `==`; `same_length_as` must NOT be
/// set in that case, on pain of `relational_sum.rs::is_len_of`
/// accepting a division by a count the accumulation never actually ran
/// over. This mirrors the `lo` widening below: both readings state the
/// same fact (whether every position survived), once as a window bound
/// and once as a name link.
fn comprehension_star_elements(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let (target_names, conditions, element, source_lo, source_hi, source_name) =
        comprehension_target_and_star_element(generators, environment, kernel)?;
    let mut fork = environment.fork();
    if !bind_comprehension_target(&mut fork, &target_names, &element) {
        return None;
    }
    if !conditions.is_empty() && comprehension_conditions_hold(conditions, &fork, kernel).is_none() {
        return None;
    }
    let mapped = evaluate_expression(element_expr, &fork, kernel);
    if mapped.kind != Kind::Set {
        return None; // the mapped element must itself name a scalar set to re-window over
    }
    let lo = if conditions.is_empty() { source_lo } else { 0 };
    let window = repetition(mapped.set.clone(), lo, source_hi);
    // `conditions.is_empty()` gates this the SAME way it gates `lo`
    // above: a filter can drop positions, so the length link would be
    // an unproved claim once one is present.
    let same_length_as = if conditions.is_empty() {
        source_name.map(str::to_owned)
    } else {
        None
    };
    Some(AbstractValue {
        kind_tag: mapped.kind_tag,
        same_length_as,
        ..known_set(window, None, TrustSpec, SetKindTag::None)
    })
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

/// `-x` over an INT-SORTED SET operand (a seeded parameter range, or a
/// set another transfer already produced) — the row `evaluate_unary`'s
/// known-single-value path cannot reach. python-pins.md arith.11: "unary
/// `-` yields the numeric negation (`__neg__`)... on ints rides `int.*`
/// exactly," electing `int.neg`, whose kernel arm is
/// `boundary/python.lean`'s `pythonTransferOfOp1`. The answer is
/// Integer-sorted: negation of an integer is an integer, and arith.1's
/// unlimited precision means it never wraps.
///
/// Only `USub` has a row here. `UAdd` over a set is the operand itself
/// and needs no kernel question, but answering it would restate a value
/// this function was handed rather than transfer one, so it is left to
/// the caller's own decline; `Invert` (`~x`) is `-(x+1)`, a composition
/// no pins row states as an `int.*` member; `Not` is decided before the
/// numeric guard entirely.
///
/// A Float-sorted set declines: `binary64.neg` is that row's election
/// (arith.11's own float branch), a different question this function
/// does not pose. A kernel refusal reads as `None` through the same
/// `catch_unwind` discipline `transfer_over_sets` keeps.
fn negate_over_set(op: UnaryOp, operand: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    if op != UnaryOp::USub || operand.kind != Kind::Set {
        return None;
    }
    let (operand_set, sort) = transferable_numeric_operand(operand)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    let grade = refined_domain::trust_grades::derived_trust_level(TrustProved, std::slice::from_ref(operand));
    int_transfer_answer(
        TransferQuestionOp::IntNeg,
        operand_set,
        make_refined_set(vec![]),
        grade,
        kernel,
    )
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
fn binary_arithmetic_pair(
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
                unknown()
            } else {
                arithmetic_result(left_value / right_value, false)
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
        // `@` has no cited CPython row for exact-value arithmetic
        // transfer in this wave.
        Operator::MatMult => unknown(),
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
        Operator::LShift | Operator::RShift => unknown(),
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
                return unknown();
            };
            let Some(right_int) = f64_to_exact_i64(right_value) else {
                return unknown();
            };
            let result = match op {
                Operator::BitOr => left_int | right_int,
                Operator::BitXor => left_int ^ right_int,
                Operator::BitAnd => left_int & right_int,
                _ => unreachable!("guarded to BitOr/BitXor/BitAnd above"),
            };
            arithmetic_result(result as f64, true)
        }
        Operator::BitOr | Operator::BitXor | Operator::BitAnd => unknown(),
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
fn multi_value_binary_arithmetic(op: Operator, left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
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
    Some(known_values(combined, result_sort, TrustProved))
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
fn numeric_values_with_sort(value: &AbstractValue) -> Option<Vec<(f64, PrimitiveKind)>> {
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
/// - `Div` (`/`) lowers EXCEPT at the zero-divisor corner. Python's `/`
///   is always true division — arith.9 (python-pins.md): "Division of
///   int by int (`/`) yields a float — the type is widened even when
///   the arguments are exact integers" — and elects `binary64.div` for
///   exactly this reason, the SAME election the kernel's `Div` row
///   already carries. Away from a zero divisor the two `/`s name the
///   same theorem. AT a zero divisor they diverge: arith.10 makes
///   Python's `/` raise `ZeroDivisionError` (an exception, not a
///   value), while ECMA's `binary64.div` answers a DETERMINED
///   `±Infinity`/NaN (`theories/binary64/div.lean`'s `transferDiv`,
///   proved sound for that theorem — a correct answer to the WRONG
///   question for a Python operand). `transfer_over_sets` gates this:
///   it asks the kernel only when the divisor operand's set provably
///   EXCLUDES zero (`divisor_provably_excludes_zero`); when the
///   divisor's set may admit zero, the value question declines rather
///   than relabel ECMA's answer as Python's. `transfer_over_sets`'s own
///   `result_sort` computation carries the always-Float override for
///   this one op — the `both_int` rule the other three admitted ops
///   share does not apply here.
/// - `FloorDiv` (`//`), `Mod` (`%`), and `Pow` (`**`) do NOT lower to
///   the FLOAT family. `%` takes the DIVISOR's sign in Python, the
///   opposite of ECMA's dividend-sign remainder (AGENT-BRIEF.md,
///   expressions.rst §6.7) — asking the kernel's `Rem` row for a Python
///   `%` would silently answer the wrong sign on a mixed-sign pair; `//`
///   floors toward negative infinity, which is not one of the kernel's
///   float arithmetic transfer rows at all; `**` has no float
///   binary-arithmetic-transfer row in this family (`Pow` in
///   `TransferQuestionOp` is the pinned NaN/unknown/set `PowOperandWire`
///   shape, a different question shape from the plain two-`RefinedSet`
///   rows this function poses).
///
/// `admitted_int_transfer_op` below states the row those three DO have,
/// on the other side of the sort split: the exact `int` theory.
fn admitted_transfer_op(op: Operator) -> Option<refined_kernel::transfer_questions::TransferQuestionOp> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    match op {
        Operator::Add => Some(TransferQuestionOp::Add),
        Operator::Sub => Some(TransferQuestionOp::Sub),
        Operator::Mult => Some(TransferQuestionOp::Mul),
        Operator::Div => Some(TransferQuestionOp::Div),
        _ => None,
    }
}

/// The kernel `TransferQuestionOp` a Python operator lowers to when BOTH
/// operands are INT-SORTED — the exact `int` theory
/// (`boundary/python.lean`'s `pythonTransferOfOp2`), never the
/// `binary64.*` float image `admitted_transfer_op` returns. Python's
/// integers have unlimited precision and never wrap (python-pins.md
/// arith.1), so every row here is exact arithmetic on the mathematical
/// integers, which is what the `int.*` theory proves:
///
/// - `Add`/`Sub`/`Mult` elect `int.add`/`int.sub`/`int.mul` — the exact
///   whole-number operations arith.1 names ("the float transfer is
///   REFUSED for ints and the exact whole-number theory (`int.*`) serves
///   them"). The float image would agree on any operand pair small
///   enough to be f64-exact, but the exact theory is the one the pins
///   elect, and it is the theory that stays right at the edges.
/// - `FloorDiv` elects `int.floorDiv` — arith.7/arith.8: floor division
///   "is always rounded towards minus infinity," paired with `%` by
///   `x == (x//y)*y + (x%y)`. A zero divisor is `ZeroDivisionError`
///   (arith.10), which the kernel arm refuses on rather than answering.
/// - `Mod` elects `rem.divisorSign` — arith.4: "the modulo operator
///   yields a result with the same sign as its SECOND operand (the
///   divisor)." This is the Python-owned remainder, a DIFFERENT theorem
///   from the `rem.truncDividendSign` row JavaScript's `%` elects, so
///   electing it by name is what makes the sign right on a mixed-sign
///   pair.
/// - `Pow` elects `int.pow` — pow.1: "`int ** nonnegative int` yields an
///   exact int (same type as the operands)... a negative int exponent
///   converts both arguments to float and yields a float." The kernel's
///   `int.pow` arm reads its exponent as a nonnegative `Nat`, so
///   `int_transfer_over_sets` below gates this row on
///   `exact_nonnegative_integer` before ever asking — a
///   possibly-negative exponent declines to the float path, which is
///   where pow.1 sends it anyway.
/// - `BitAnd`/`BitOr`/`BitXor` elect `int.bitAnd`/`int.bitOr`/
///   `int.bitXor` — bits.4/bits.5/bits.6: the bitwise operations on
///   UNBOUNDED ints, "never JS's 32-bit wrap view." The `int32.*` family
///   the JavaScript rows elect is the wrong theorem here for exactly
///   that reason.
///
/// `Div` is absent by design: arith.9 widens int/int to float, so `/`
/// never has an int-sorted row at all — it stays on
/// `admitted_transfer_op`'s `binary64.div`. `LShift`/`RShift` are also
/// absent: bits.1/bits.2 define them as `int.floorDiv`/`int.mul` by
/// `2**n` rather than as their own members, so they lower as that
/// COMPOSITION (`shift_as_int_composition` below), not as a direct op.
fn admitted_int_transfer_op(op: Operator) -> Option<refined_kernel::transfer_questions::TransferQuestionOp> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    match op {
        Operator::Add => Some(TransferQuestionOp::IntAdd),
        Operator::Sub => Some(TransferQuestionOp::IntSub),
        Operator::Mult => Some(TransferQuestionOp::IntMul),
        Operator::FloorDiv => Some(TransferQuestionOp::IntFloorDiv),
        Operator::Mod => Some(TransferQuestionOp::RemDivisorSign),
        Operator::Pow => Some(TransferQuestionOp::IntPow),
        Operator::BitAnd => Some(TransferQuestionOp::IntBitAnd),
        Operator::BitOr => Some(TransferQuestionOp::IntBitOr),
        Operator::BitXor => Some(TransferQuestionOp::IntBitXor),
        _ => None,
    }
}

/// A known EXACT NONNEGATIVE INTEGER operand, as its own `f64` — the
/// shape two rows below need before they may ask an `int.*` question:
/// `Pow`'s exponent (pow.1's nonnegative-int branch is the only one the
/// exact `int.pow` theory covers; a negative exponent "converts both
/// arguments to float and yields a float," a different row) and a
/// shift's count (bits.3: "a negative shift count is illegal and raises
/// `ValueError`"). A SET operand answers `None` even when its whole
/// range is nonnegative — the kernel's own `int.*` arms read an operand
/// through `exactIntOf` (a closed singleton, `numeric/enclosure_read.lean`),
/// so a range exponent has nothing to offer them, and proving
/// nonnegativity of a range here would state a gate the row behind it
/// cannot use. The value must also sit inside the f64-exact 2^53 window
/// `arithmetic_result` already trusts, for the same reason it does.
fn exact_nonnegative_integer(value: &AbstractValue) -> Option<f64> {
    let (number, sort) = single_numeric_value(value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    if number < 0.0 || number.fract() != 0.0 || number.abs() >= 2f64.powi(53) {
        return None;
    }
    Some(number)
}

/// Poses one `int.*` question and reads the answer back as an
/// INTEGER-SORTED value. Every `int.*` member answers exact whole
/// numbers (python-pins.md arith.1 — "integer `+ − ×` never overflows
/// and never wraps"), so the answer's sort is Integer unconditionally,
/// with no `both_int` computation of its own: reaching this function AT
/// ALL already required both operands to be int-sorted.
///
/// Two guards the float family does not need:
///
/// - A non-integral value in the answer declines. `int.*` cannot produce
///   one, so this can only mean the wire carried something this row
///   does not understand.
/// - A value outside the f64-exact 2^53 window declines. Python's
///   integers are unbounded (arith.1) and the kernel computes them
///   exactly as `Int`s, but `boundary/encode_sets.lean`'s `encodeNumber`
///   puts every result through `roundNE` before it crosses the wire — so
///   a result past 2^53 arrives ROUNDED, and claiming it as exact would
///   be claiming a value CPython never computes. This is the same window
///   and the same reason `arithmetic_result` already declines on.
///
/// A SET answer must additionally CARRY its own integrality
/// (`requires_integer`) before it is tagged Integer-sorted. Most `int.*`
/// arms answer `.vals`, so this is about the one arm that answers an
/// enclosure — `rem.divisorSign`, whose general-interval branch produces
/// a bound-shaped enclosure. Tagging a set Integer-sorted without that
/// mark would claim an integrality the kernel did not state.
fn int_transfer_answer(
    transfer_op: refined_kernel::transfer_questions::TransferQuestionOp,
    left_set: RefinedSet,
    right_set: RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let nan_operand = refined_kernel::transfer_questions::PowOperandWire {
        kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
        set: make_refined_set(vec![]),
    };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&refined_kernel::transfer_questions::TransferQuestion {
            op: transfer_op,
            a: left_set,
            b: right_set,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    });
    let answer = asked.ok()?;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    match answer.kind {
        TransferAnswerKind::Values => {
            if answer
                .values
                .iter()
                .any(|v| v.fract() != 0.0 || v.abs() >= 2f64.powi(53))
            {
                return None;
            }
            Some(known_values(answer.values, PrimitiveKind::Integer, grade))
        }
        TransferAnswerKind::Set => {
            if !requires_integer(&answer.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(answer.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// The INT-SORTED half of `transfer_over_sets`: when both operands are
/// Integer-sorted, the exact `int` theory serves the operation, not the
/// `binary64.*` float image (python-pins.md arith.1 states this
/// directly — "the float transfer is REFUSED for ints and the exact
/// whole-number theory (`int.*`) serves them").
///
/// Ops and their rows are `admitted_int_transfer_op`'s own doc.
/// The two conditional rows this function gates before asking:
///
/// - `Pow`: the kernel's `int.pow` arm reads its exponent as a
///   nonnegative `Nat`, matching pow.1's own exact branch. An exponent
///   this file cannot prove is an exact nonnegative integer
///   (`exact_nonnegative_integer`) DECLINES here and falls through to
///   the float path, which is where pow.1 puts a negative exponent
///   anyway ("a negative int exponent converts both arguments to float
///   and yields a float").
/// - `LShift`/`RShift`: bits.1/bits.2 define these as compositions
///   rather than as their own kernel members, so they lower as that
///   composition — see `shift_as_int_composition`.
///
/// `Div` never reaches here: arith.9 widens int/int to float, so the
/// caller keeps `/` on the float path unconditionally.
fn int_transfer_over_sets(
    op: Operator,
    right: &AbstractValue,
    left_set: &RefinedSet,
    right_set: &RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if matches!(op, Operator::LShift | Operator::RShift) {
        return shift_as_int_composition(op, left_set, right, grade, kernel);
    }
    if op == Operator::Pow {
        // pow.1's exact branch only — a possibly-negative exponent is
        // the float row, not this one
        exact_nonnegative_integer(right)?;
    }
    let transfer_op = admitted_int_transfer_op(op)?;
    int_transfer_answer(transfer_op, left_set.clone(), right_set.clone(), grade, kernel)
}

/// `x << n` / `x >> n` over int-sorted operands, lowered as the
/// COMPOSITION the pins define them to be rather than as kernel members
/// of their own:
///
/// - bits.2: "`x << n` equals multiplication of `x` by `2**n`" —
///   `int.mul` against the singleton `{2**n}`.
/// - bits.1: "`x >> n` equals floor division of `x` by `2**n`" —
///   `int.floorDiv` against the same singleton.
///
/// The shift count must be a KNOWN exact nonnegative integer
/// (`exact_nonnegative_integer`): bits.3 makes a negative count a
/// `ValueError` rather than a value, and a count this file cannot read
/// exactly gives no `2**n` to compose against. `2**n` itself must also
/// land inside the f64-exact 2^53 window, or the singleton this builds
/// would not be the number it names — the same window every other
/// exactness gate in this file keeps.
fn shift_as_int_composition(
    op: Operator,
    left_set: &RefinedSet,
    right: &AbstractValue,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    let count = exact_nonnegative_integer(right)?;
    let factor = 2f64.powf(count);
    if factor.fract() != 0.0 || factor >= 2f64.powi(53) {
        return None;
    }
    let factor_set = make_refined_set(vec![one_of(&[factor])]);
    let transfer_op = if op == Operator::LShift {
        TransferQuestionOp::IntMul
    } else {
        TransferQuestionOp::IntFloorDiv
    };
    int_transfer_answer(transfer_op, left_set.clone(), factor_set, grade, kernel)
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
///
/// Whether a divisor's set PROVABLY EXCLUDES zero — the gate `/` needs
/// before it may ask `binary64.div` (see `admitted_transfer_op`'s `Div`
/// bullet: away from a zero divisor the two `/`s name the same
/// theorem, but AT one, ECMA answers a determined `±Infinity`/NaN
/// while Python raises `ZeroDivisionError`, arith.10). `0.0` is a
/// member of `divisor` exactly when the kernel's own membership
/// decider says so (`kernel.member`, `x ∈ A` — `memberB_iff`, the same
/// ask `assignability.rs`'s containment law poses); `member` is total
/// over every enclosure this file builds, so there is no refusal shape
/// here to catch the way `scalar_subset`/`seq_subset` need one. A
/// divisor set that DOES admit zero answers `false` here — this
/// function only PROVES the exclusion, it never guesses it, so any
/// doubt routes to "may be zero."
fn divisor_provably_excludes_zero(divisor: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    let asked = crate::kernel_ask::ask_kernel(|| (kernel.member)(divisor, &[0.0]));
    matches!(asked, Ok(false))
}

/// `a / b` for a divisor window `b` that ADMITS zero (`0.0 ∈ b`) but is
/// not itself always zero (`divisor_is_provably_always_zero` already owns
/// the always-zero case as an unconditional raise, in `binop_provable_raise`
/// below). CPython's `/` still raises `ZeroDivisionError` on the zero arm
/// of such a window (arith.10), so this never asks the kernel with `b`
/// itself — that would ask `binary64.div` a question the divisor's own
/// zero member makes unsound for a Python `/`. Instead it splits `b`
/// around zero into its strictly-negative half (`b ∩ below(0)`) and its
/// strictly-positive half (`b ∩ above(0)`) — each half PROVABLY excludes
/// zero by construction — asks `binary64.div` on `a` against each half
/// separately, and unions whichever halves answer into one `RefinedSet`.
/// A half whose intersection with `b` is empty (`kernel.scalar_empty`,
/// e.g. `b` is entirely negative so its positive half is vacuous) is
/// skipped rather than asked, matching `divisor_is_provably_always_zero`'s
/// own empty-set guard.
///
/// This determines the VALUE question on every path that does not raise;
/// the zero arm itself is a MAY-RAISE this function does not speak to —
/// `binop_provable_raise` only fires an unconditional raise when the
/// entire divisor window is zero, so a window that merely ADMITS zero
/// alongside other values raises on SOME inputs and returns a value on
/// others. Reporting that raise arm as its own diagnostic (rather than
/// leaving it to CPython at runtime) is future work; no existing
/// possibly-raising expression in this file reports a partial-raise
/// arm alongside its value determination, so this function returns only
/// the sound value binding over the non-raising split, exactly as every
/// other admitted transfer answer already does.
fn split_divisor_transfer(
    left_set: RefinedSet,
    right_set: &RefinedSet,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<refined_kernel::transfer_questions::TransferAnswer> {
    use refined_kernel::transfer_questions::TransferAnswerKind;
    use refined_kernel::transfer_questions::TransferQuestion;
    use refined_kernel::transfer_questions::TransferQuestionOp;

    let ask_half = |divisor_half: RefinedSet| -> Option<refined_kernel::transfer_questions::TransferAnswer> {
        let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(&divisor_half));
        if matches!(empty, Ok(true)) || empty.is_err() {
            return None;
        }
        let asked = crate::kernel_ask::ask_kernel(|| {
            (kernel.transfer)(&TransferQuestion {
                op: TransferQuestionOp::Div,
                a: left_set.clone(),
                b: divisor_half,
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
        });
        asked.ok()
    };

    let negative_half = make_refined_set({
        let mut forms = right_set.forms.clone();
        forms.push(below(0.0));
        forms
    });
    let positive_half = make_refined_set({
        let mut forms = right_set.forms.clone();
        forms.push(above(0.0));
        forms
    });

    let negative_answer = ask_half(negative_half);
    let positive_answer = ask_half(positive_half);

    // A may-be-NaN answer on either half must never masquerade as a
    // NaN-free result — the whole split declines rather than silently
    // drop the NaN-carrying half's values.
    if matches!(negative_answer.as_ref().map(|a| a.kind), Some(TransferAnswerKind::NaN))
        || matches!(positive_answer.as_ref().map(|a| a.kind), Some(TransferAnswerKind::NaN))
    {
        return None;
    }

    match (negative_answer, positive_answer) {
        (None, None) => None,
        (Some(only), None) | (None, Some(only)) => Some(only),
        (Some(neg), Some(pos)) => Some(union_transfer_answers(neg, pos)),
    }
}

/// Unions two `TransferAnswer`s of the SAME kind family into one answer:
/// `Values` concatenates (both sides are exact singleton sets); `Set`
/// unions the two enclosures via the grammar's own `Union` form; either
/// side reading `Unknown` widens the whole union to `Unknown` (an
/// enclosure the kernel could not narrow on one half narrows nothing
/// once joined with the other). NaN is never passed here —
/// `split_divisor_transfer` already declines before this is called.
fn union_transfer_answers(
    a: refined_kernel::transfer_questions::TransferAnswer,
    b: refined_kernel::transfer_questions::TransferAnswer,
) -> refined_kernel::transfer_questions::TransferAnswer {
    use refined_kernel::transfer_questions::TransferAnswer;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    match (a.kind, b.kind) {
        (TransferAnswerKind::Values, TransferAnswerKind::Values) => {
            let mut values = a.values;
            values.extend(b.values);
            TransferAnswer {
                kind: TransferAnswerKind::Values,
                values,
                set: make_refined_set(vec![]),
            }
        }
        (TransferAnswerKind::Unknown, _) | (_, TransferAnswerKind::Unknown) => TransferAnswer {
            kind: TransferAnswerKind::Unknown,
            values: vec![],
            set: make_refined_set(vec![]),
        },
        _ => {
            let a_set = match a.kind {
                TransferAnswerKind::Values => make_refined_set(vec![one_of(&a.values)]),
                _ => a.set,
            };
            let b_set = match b.kind {
                TransferAnswerKind::Values => make_refined_set(vec![one_of(&b.values)]),
                _ => b.set,
            };
            TransferAnswer {
                kind: TransferAnswerKind::Set,
                values: vec![],
                set: make_refined_set(vec![union(a_set, b_set)]),
            }
        }
    }
}

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
    let (left_set, left_sort) = transferable_numeric_operand(left)?;
    let (right_set, right_sort) = transferable_numeric_operand(right)?;
    let grade = refined_domain::trust_grades::derived_trust_level(
        refined_domain::trust_grades::TrustProved,
        &[left.clone(), right.clone()],
    );
    // BOTH operands int-sorted: the exact `int` theory serves the
    // operation, never the float image (arith.1). `/` is the one
    // exception and stays on the float path below — arith.9 widens
    // int/int to float, so it has no int-sorted row at all. An int row
    // the kernel declines (an operand that is not a closed singleton,
    // a zero divisor, an exponent past the boundary's own fuel ceiling)
    // falls through to the float path unchanged rather than losing the
    // determination outright.
    if op != Operator::Div && left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer {
        if let Some(answer) = int_transfer_over_sets(op, right, &left_set, &right_set, grade, kernel) {
            return Some(answer);
        }
    }
    let transfer_op = admitted_transfer_op(op)?;
    // `Div`'s always-float override (arith.9: "the type is widened even
    // when the arguments are exact integers") beats the both_int rule
    // outright — Python `/` never stays Integer-sorted regardless of
    // its operands' own sorts. Every other admitted op (Add/Sub/Mult)
    // keeps the same both_int rule binary_arithmetic_value's
    // known-values path uses: Integer only when BOTH sides are
    // Integer-sorted.
    let both_int = op != Operator::Div && left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer;
    let result_sort = if both_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    // arith.10's carve-out: a divisor whose set admits zero diverges from
    // `binary64.div` asked directly — ECMA answers a determined
    // `±Infinity`/NaN at zero, Python raises `ZeroDivisionError`
    // (`divisor_is_provably_always_zero`'s window owns the unconditional
    // raise, named in `binop_provable_raise`). A window that admits zero
    // WITHOUT being entirely zero raises only on its zero arm and
    // determines a value on every other input — `split_divisor_transfer`
    // asks `binary64.div` on the zero-excluded negative and positive
    // halves of the divisor separately and unions the two answers, so the
    // value question determines on the non-raising split rather than
    // decline outright. An always-zero divisor has no non-raising half at
    // all (both halves are empty), so it still declines here exactly as
    // before — the raise is the whole answer for that window.
    let answer = if op == Operator::Div && !divisor_provably_excludes_zero(&right_set, kernel) {
        split_divisor_transfer(left_set, &right_set, kernel)?
    } else {
        let asked = crate::kernel_ask::ask_kernel(|| {
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
        });
        asked.ok()?
    };
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
        TransferAnswerKind::Unknown => {
            // The kernel's honest top for this operand pair: no enclosure
            // narrows the result (e.g. a bounded set times an unbounded
            // one), but the SORT rule still holds — the same language-level
            // guarantee float_sorted_unknown carries for the math family —
            // so the answer is sort-known, value-unknown, never nothing.
            // A downstream clamp (max/min) can still bound it, which is
            // exactly how a two-free-name comprehension element derives.
            Some(if both_int {
                AbstractValue {
                    kind_tag: Some(PrimitiveKind::Integer),
                    ..known_set(
                        make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]),
                        None,
                        TrustSpec,
                        SetKindTag::None,
                    )
                }
            } else {
                float_sorted_unknown()
            })
        }
        // A may-be-NaN answer must never masquerade as a NaN-free set.
        TransferAnswerKind::NaN => None,
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
    if let Some(value) = date_timedelta_binop_value(binop.op, &left, &right, kernel) {
        return value;
    }
    binary_arithmetic_value_with_kernel(binop.op, &left, &right, kernel)
}

/// `date ± timedelta` (date.7's own operation-table row) — tried BEFORE
/// the ordinary numeric/sequence dispatch, since neither operand is a
/// single numeric value or a string/list (`binary_arithmetic_value`'s
/// own fallthrough would otherwise reach `sequence_binop_value` and
/// answer `unknown()` for a tagged-Object pair). `date + timedelta` and
/// `timedelta + date` both shift forward (`Operator::Add`, either
/// operand order — datetime.rst states the operation both ways);
/// `date - timedelta` shifts backward (`Operator::Sub`, `date` on the
/// LEFT only — `timedelta - date` is not a datetime.rst operation).
/// `date - date` (the OTHER `date.7` row, an exact `timedelta` result)
/// is NOT built here: no row in this file's construct list asks for it,
/// and `timedelta_construction_value`'s own single `days` field gives no
/// two-instance subtraction a shape to land in without inventing one.
/// `None` for every operand pair that is not exactly one tagged
/// `datetime_date` and one tagged `datetime_timedelta` — the caller
/// falls through to the ordinary dispatch unchanged.
fn date_timedelta_binop_value(op: Operator, left: &AbstractValue, right: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let is_date = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_date";
    let is_timedelta = |value: &AbstractValue| value.kind == Kind::Object && value.source == "datetime_timedelta";
    match op {
        Operator::Add => {
            if is_date(left) && is_timedelta(right) {
                return date_shifted_by_timedelta(left, right, false, kernel);
            }
            if is_timedelta(left) && is_date(right) {
                return date_shifted_by_timedelta(right, left, false, kernel);
            }
            None
        }
        Operator::Sub => {
            if is_date(left) && is_timedelta(right) {
                return date_shifted_by_timedelta(left, right, true, kernel);
            }
            None
        }
        _ => None,
    }
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
            if let Some(result) = string_set_concatenation(left, right) {
                return result;
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

/// `left + right` where at least one side is a STRING-SHAPED SET rather
/// than an exact literal — `seed + "xxxxxxxx"` where `seed` is a
/// parameter carrying a length window (`Annotated[str, Field(min_length=…,
/// max_length=…)]` seeds `Kind::Set` over a `Repeat`/`Star` form,
/// `check.rs::seed_parameters`'s own doc), never `Kind::Values`. The
/// exact-exact row above already answers when BOTH operands are literal
/// strings; this row is what fires the moment either one is not. Composes
/// the same `refinement_forms::concatenation` form `string_tuple`
/// concatenation and the f-string pattern tier already build — pure set
/// composition, no kernel round trip, since concatenation is a GRAMMAR
/// constructor, not a decided question. `None` when either side is not
/// string-shaped at all (a numeric set, an object, an unknown), so the
/// caller's own `unknown()` fallback stays honest for it.
fn string_set_concatenation(left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let left_set = string_shaped_set(left)?;
    let right_set = string_shaped_set(right)?;
    let joined = make_refined_set(vec![refined_sets::refinement_forms::concatenation(left_set, right_set)]);
    let grade = refined_domain::trust_grades::derived_trust_level(TrustProved, &[left.clone(), right.clone()]);
    Some(known_set(joined, None, grade, SetKindTag::None))
}

/// The `RefinedSet` a value states, read ONLY when the value is
/// string-shaped: a known exact string (`Kind::Values` tagged
/// `PrimitiveKind::String` — read through `set_of_known`, which answers
/// the code points' own concatenation form for a multi-character
/// literal), or an untagged `Kind::Set` whose own forms demonstrably
/// carry a sequence shape (`assignability::states_sequence` — the same
/// gate `scalar_case_of`/`seed_parameters` use to tell a string window
/// from a numeric range, since a bare `Kind::Set` carries no sort tag of
/// its own to read instead). A numeric set, an object, or any other
/// shape answers `None` — never guessed at.
fn string_shaped_set(value: &AbstractValue) -> Option<RefinedSet> {
    if value.kind == Kind::Values && value.kind_tag == Some(PrimitiveKind::String) {
        return refined_domain::lattice_operations::set_of_known(value);
    }
    if value.kind == Kind::Set && value.set_kind_tag == SetKindTag::None && assignability::states_sequence(&value.set) {
        return Some(value.set.clone());
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
///
/// A Python `float` arithmetic result CAN be NaN (`inf - inf`, `inf *
/// 0.0` — IEEE 754, arith.9's own doc), so the Float row answers
/// `nan_value()` — the domain's own NaN state — rather than let a bare
/// NaN enter `known_values`, which no refined set admits
/// (`refinement_forms::element`'s own construction-time refusal). The
/// int row cannot reach this: an int-sorted pair's NaN check already
/// takes the `value.fract() != 0.0` decline above (`NaN.fract()` is
/// itself NaN, which is `!= 0.0`), so no int-sorted NaN ever reaches
/// `known_values` here.
fn arithmetic_result(value: f64, both_int: bool) -> AbstractValue {
    if both_int {
        if value.fract() != 0.0 || value.abs() >= 2f64.powi(53) {
            return unknown();
        }
        return known_values(vec![value], PrimitiveKind::Integer, TrustProved);
    }
    if value.is_nan() {
        return nan_value();
    }
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

/// An f64 carrying a whole number, as an `i64` — but only inside the
/// same 2^53 exact-integer window `arithmetic_result` already trusts
/// for every other operator. `&`/`|`/`^` need integer bit patterns
/// rather than f64 arithmetic, so this is the one extra conversion
/// step those three rows take; anything outside the window declines
/// the same way an out-of-range `+`/`-`/`*` result already does.
fn f64_to_exact_i64(value: f64) -> Option<i64> {
    if value.fract() != 0.0 || value.abs() >= 2f64.powi(53) {
        return None;
    }
    Some(value as i64)
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
/// Every row here means every run raises, full stop — `check.rs::
/// sink_value`'s own all-or-nothing gate (a fire here skips the value
/// question entirely). A SOMETIMES-raises escape (some admitted operand
/// values raise, the rest still produce a value) is a DIFFERENT claim
/// and lives in its own function, `possible_raise` below — never a row
/// here.
///
/// Recognized rows, each cited in the function that decides it: zero-
/// divisor arithmetic (`/`, `//`, `%`), an out-of-range/absent
/// subscript on a known List/Object, a bytes-like read/write whose
/// `bytes_models` answer is `BytesAnswer::Raises`, `int(<unparseable
/// known string>)`, `<receiver>.index(<absent known needle>)` on a
/// known string or list receiver, `math.sqrt(<known negative>)`, and
/// `math.floor`/`ceil`/`trunc` of a known non-finite argument
/// (`OverflowError` for an infinity, `ValueError` for NaN — each
/// returns an `Integral`, and no Python `int` holds either).
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
/// is zero). The evaluation path (`binary_arithmetic_value`/
/// `transfer_over_sets`) already declines these to `unknown()` for the
/// VALUE question; this is the same zero-divisor check speaking the
/// fact as a provable raise rather than a silent decline — the value
/// path is unchanged.
///
/// Two shapes prove the divisor is ALWAYS zero, never SOMETIMES zero
/// (`provable_raise`'s own contract — a fire here means every real
/// execution raises, `check.rs::sink_value`'s doc): a known scalar
/// `0.0`/`-0.0` (`single_numeric_value`), or a `Kind::Set` divisor whose
/// entire real range is the singleton `{0.0}` — a seeded window that
/// has narrowed to nothing but zero, not merely a window that ADMITS
/// zero alongside other values (`age - age`'s own `[0, 0]` window
/// against a `/`, for instance). A wider window that only ADMITS zero
/// (e.g. `[0.0, 2.0]`) is a SOMETIMES-raises divisor, which this
/// function must NOT fire on — `possible_raise`/`binop_possible_raise`
/// below is that window's own row; firing an unconditional raise for a
/// mostly-nonzero window here would be a false positive, the same
/// overreach `rounding_argument_raises`' finite-argument gate avoids
/// on the value side.
fn binop_provable_raise(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if !matches!(binop.op, Operator::Div | Operator::FloorDiv | Operator::Mod) {
        return None;
    }
    let right = evaluate_expression(&binop.right, environment, kernel);
    if let Some((right_value, _)) = single_numeric_value(&right) {
        if right_value == 0.0 {
            return Some((
                binop.range(),
                "this expression provably raises ZeroDivisionError: division by zero".to_owned(),
            ));
        }
        return None;
    }
    if right.kind == Kind::Set && divisor_is_provably_always_zero(&right.set, kernel) {
        return Some((
            binop.range(),
            "this expression provably raises ZeroDivisionError: division by zero".to_owned(),
        ));
    }
    None
}

/// Whether `expression` (or a sub-expression `provable_raise`'s own
/// pre-order walk already cleared) has a SOMETIMES-raising corner: some
/// admitted operand values raise, the rest still produce a value this
/// file determines. A DIFFERENT claim from `provable_raise`'s
/// all-or-nothing one, and a DIFFERENT sink discipline follows from it
/// — the finding and the value both stand at whatever sink this
/// expression flows into; the sink decides how to combine them
/// (`check.rs`'s own wiring, not this file's). `Some((range, message))`
/// names the escaping expression's own range and the sentence
/// `diagnostic_sentences.rs` builds for it; `None` when no recognized
/// sometimes-raising shape applies.
///
/// Recognized rows, each cited in the function that decides it: a `/`,
/// `//`, or `%` divisor set that ADMITS zero without being entirely
/// zero (`binop_possible_raise`).
pub fn possible_raise(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    match expression {
        Expr::BinOp(binop) => binop_possible_raise(binop, environment, kernel),
        Expr::Call(call) => domain_limited_family_possible_raise(call, environment, kernel),
        _ => None,
    }
}

/// `math.log(x)`/`log2`/`log10`/`log1p`/`asin`/`acos`/`atanh`/`acosh`
/// where `x`'s window STRADDLES CPython's own raise domain (some
/// admitted values raise, the rest still return a value) —
/// `math_models::DomainRaiseClassification::Straddles`'s own row, the
/// sibling this `call_provable_raise`'s all-or-nothing arm explicitly
/// defers to. The window's ENTIRELY-raising case is `call_provable_
/// raise`'s own row (an unconditional fire, no value question at all);
/// the ENTIRELY-served case fires nothing here and answers its value
/// through `math_models::math_call_result`'s ordinary kernel-backed
/// path, unaffected by this function.
fn domain_limited_family_possible_raise(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let family = math_models::DomainLimitedFamily::of_function(attribute.attr.as_str())?;
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "math" || environment.read("math").is_some() {
        return None;
    }
    let [only_arg] = &*call.arguments.args else {
        return None;
    };
    let argument = evaluate_expression(only_arg, environment, kernel);
    if !matches!(
        math_models::domain_raise_classification(family, &argument, kernel),
        Some(math_models::DomainRaiseClassification::Straddles)
    ) {
        return None;
    }
    Some((call.range(), family.raise_message().to_owned()))
}

/// `x / d`, `x // d`, `x % d` where `d`'s set ADMITS zero without being
/// entirely zero (e.g. `[0.0, 2.0]`) — a SOMETIMES-raises divisor: most
/// real executions clear it, and CPython raises `ZeroDivisionError` on
/// the zero arm of the window for all three operators alike
/// (expressions.rst, "Binary arithmetic operations": "Division by zero
/// raises the ZeroDivisionError exception" for `/`/`//`, "A zero right
/// argument raises the ZeroDivisionError exception" for `%`).
/// `divisor_provably_excludes_zero` gates the value question's OWN
/// silence for `/`'s shape (`transfer_over_sets`); this row asks the
/// same membership question this file already asks there, so the two
/// never disagree about which windows admit zero. `binop_provable_
/// raise`'s own always-zero rows are excluded by construction: a
/// divisor this function reads as NOT provably excluding zero is
/// either always-zero (that row's own claim, made there) or
/// sometimes-zero (this row's claim) — the caller decides which
/// question it is asking by which function it calls.
///
/// The three operators diverge only on the VALUE side of this same
/// corner, never on the RAISE side this function speaks to:
/// `split_divisor_transfer` is `/`'s own fix (`transfer_over_sets`'s own
/// gate, `op == Operator::Div`) — it determines a value over the
/// divisor's zero-excluded halves, so `/`'s fire here rides alongside a
/// determined value. `//` and `%` still ask the kernel over the WHOLE
/// zero-admitting window, which the kernel declines for a non-singleton
/// divisor (`admitted_int_transfer_op`'s row only ever answers over two
/// exact singletons) — so their fire here rides alongside a silent
/// value question, the value side wholly unchanged by this row.
/// `diagnostic_sentences::division_by_a_set_that_admits_zero` already
/// speaks generically to "this expression's divisor set" without
/// naming `/` specifically, so the one sentence serves all three
/// operators without inventing a sibling.
fn binop_possible_raise(
    binop: &ruff_python_ast::ExprBinOp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if !matches!(binop.op, Operator::Div | Operator::FloorDiv | Operator::Mod) {
        return None;
    }
    let right = evaluate_expression(&binop.right, environment, kernel);
    if right.kind != Kind::Set {
        return None;
    }
    if divisor_is_provably_always_zero(&right.set, kernel) {
        // `binop_provable_raise`'s own row already speaks this window
        // as an unconditional raise — not this function's claim to make.
        return None;
    }
    if divisor_provably_excludes_zero(&right.set, kernel) {
        return None;
    }
    Some((binop.range(), diagnostic_sentences::division_by_a_set_that_admits_zero()))
}

/// Whether a divisor SET's entire real range is nothing but zero — a
/// nonempty subset of `{0.0}` (`kernel.scalar_subset`, guarded by
/// `kernel.scalar_empty` since the empty set is vacuously a subset of
/// everything but names no real divisor to raise on). Both closures are
/// total over the scalar shapes this file builds (the same discipline
/// `divisor_provably_excludes_zero` and `assignability.rs`'s own
/// containment ask keep), so there is no refusal to catch here.
fn divisor_is_provably_always_zero(divisor: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    let zero = make_refined_set(vec![one_of(&[0.0])]);
    let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(divisor));
    if matches!(empty, Ok(true)) || empty.is_err() {
        return false;
    }
    let subset = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(divisor, &zero));
    matches!(subset, Ok(true))
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
        // `math.log`/`log2`/`log10`/`log1p`/`asin`/`acos`/`atanh`/`acosh`
        // of a KNOWN operand whose window is ENTIRELY inside CPython's
        // own raise domain provably raises `ValueError: math domain
        // error` — `math_models::DomainLimitedFamily::raise_domain`'s
        // own doc cites the exact `mathmodule.c` clause per family
        // (verified against the vendored source, not against the
        // kernel's own JavaScript-facing `.nan` corner, which disagrees
        // with CPython at one boundary point for `log`/`log2`/`log10`/
        // `log1p`/`atanh`). specifications/python/Doc/library/
        // math.rst:696-698 is the module's own impl-detail note citing
        // `log(0.0)` as its worked example of exactly this row. A
        // window that only STRADDLES the raise domain is `possible_
        // raise`'s own row (`domain_limited_family_possible_raise`
        // below), not this one's — this function's contract is
        // all-or-nothing, so only `EntirelyRaises` fires here.
        if let Some(family) = math_models::DomainLimitedFamily::of_function(attribute.attr.as_str()) {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    if let [only_arg] = &*call.arguments.args {
                        let argument = evaluate_expression(only_arg, environment, kernel);
                        if matches!(
                            math_models::domain_raise_classification(family, &argument, kernel),
                            Some(math_models::DomainRaiseClassification::EntirelyRaises)
                        ) {
                            return Some((call.range(), family.raise_message().to_owned()));
                        }
                    }
                }
            }
        }
        // `math.floor`/`ceil`/`trunc` of a KNOWN NON-FINITE argument
        // provably raises: each returns an `Integral`, and no Python
        // `int` is infinite or NaN. `rounding_argument_raises` names
        // which exception and CPython's own message; it reads the same
        // operand through the same domain gate the value rows use
        // (`integral_domain_admits`), so the value dispatch and this
        // raise dispatch agree on exactly which rounding calls raise.
        if matches!(attribute.attr.as_str(), "floor" | "ceil" | "trunc") {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    let arguments: Vec<AbstractValue> =
                        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, kernel)).collect();
                    if let Some((exception, detail)) =
                        math_models::rounding_argument_raises(attribute.attr.as_str(), &arguments)
                    {
                        return Some((call.range(), format!("this expression provably raises {exception}: {detail}")));
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

    /// `register_retained_callables` scanning a bare `lambda: 40`
    /// (the shape `summaries::interpret_body`'s `Stmt::Return` arm
    /// hands it) makes a LATER read of that SAME `Expr::Lambda` node
    /// answer a retained-callable value rather than the plain opaque
    /// one — and calling that value through `evaluate_call`'s
    /// retained-callable arm interprets the lambda's own body,
    /// answering its exact return value.
    #[test]
    fn test_retained_lambda_call_answers_its_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let lambda_expr = parse_expression("lambda: 40").expect("test source must parse").into_expr();
        let mut environment = empty_environment();
        register_retained_callables(&lambda_expr, &mut environment);
        let retained = evaluate_expression(&lambda_expr, &environment, &kernel);
        assert_eq!(retained.kind, Kind::Object);
        assert_eq!(retained.kind_word, Some("a function value"));
        assert!(!retained.source.is_empty(), "a registered lambda's source carries its table key");

        let call_expr = parse_expression("f()").expect("test source must parse").into_expr();
        let Expr::Call(call) = call_expr else { panic!("expected a call expression") };
        environment.bind("f", retained);
        let result = evaluate_call(&call, &environment, &kernel);
        assert_eq!(result.values, vec![40.0]);
    }

    /// A retained lambda that reads a FREE variable
    /// (`e-class-and-function.py`'s own `make_adder` shape: `lambda
    /// age: age + step` closes over `step`) carries that free name's
    /// value in its own closure snapshot, taken at the moment
    /// `register_retained_callables` runs — a later call answers using
    /// THAT snapshot, not whatever the call site happens to bind the
    /// free name to.
    #[test]
    fn test_retained_lambda_closure_reads_a_free_name_at_creation() {
        let Some(kernel) = loaded_kernel() else { return };
        let lambda_expr = parse_expression("lambda age: age + step").expect("test source must parse").into_expr();
        let mut environment = empty_environment();
        environment.bind("step", known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
        register_retained_callables(&lambda_expr, &mut environment);
        let retained = evaluate_expression(&lambda_expr, &environment, &kernel);

        // rebinding `step` AFTER registration must not affect the
        // already-taken closure snapshot — Python's own closure rule
        // pins the binding to the DEFINING scope, not the call site.
        environment.bind("step", known_values(vec![999.0], PrimitiveKind::Integer, TrustProved));
        environment.bind("f", retained);
        let call_expr = parse_expression("f(40)").expect("test source must parse").into_expr();
        let Expr::Call(call) = call_expr else { panic!("expected a call expression") };
        let result = evaluate_call(&call, &environment, &kernel);
        assert_eq!(result.values, vec![41.0], "must use step=1 from the closure, not step=999 from the call site");
    }

    /// Two creations of the textually SAME lambda (two calls to a
    /// function returning `lambda x: x + step`, each closing over a
    /// different `step`) never conflate: each registration mints its
    /// own key, so the second's closure never overwrites the first's
    /// still-live retained value (`conflation_probe.py`'s own row,
    /// reproduced directly against `register_retained_callables`).
    #[test]
    fn test_two_creations_of_the_same_lambda_text_keep_separate_closures() {
        let Some(kernel) = loaded_kernel() else { return };
        let lambda_expr = parse_expression("lambda x: x + step").expect("test source must parse").into_expr();
        let mut environment = empty_environment();

        environment.bind("step", known_values(vec![1.0], PrimitiveKind::Integer, TrustProved));
        register_retained_callables(&lambda_expr, &mut environment);
        let first = evaluate_expression(&lambda_expr, &environment, &kernel);

        environment.bind("step", known_values(vec![100.0], PrimitiveKind::Integer, TrustProved));
        register_retained_callables(&lambda_expr, &mut environment);
        let second = evaluate_expression(&lambda_expr, &environment, &kernel);

        environment.bind("first", first);
        environment.bind("second", second);
        let call_first = parse_expression("first(40)").expect("test source must parse").into_expr();
        let Expr::Call(call_first) = call_first else { panic!("expected a call expression") };
        let call_second = parse_expression("second(40)").expect("test source must parse").into_expr();
        let Expr::Call(call_second) = call_second else { panic!("expected a call expression") };
        assert_eq!(evaluate_call(&call_first, &environment, &kernel).values, vec![41.0]);
        assert_eq!(evaluate_call(&call_second, &environment, &kernel).values, vec![140.0]);
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

    /// `inf - inf` — a Float `Sub` result that is NaN (IEEE 754). This
    /// must answer the domain's own `Kind::NaN` state rather than panic:
    /// `arithmetic_result`'s Float row screens for NaN and answers
    /// `nan_value()` instead of building `known_values(vec![NaN], ..)`,
    /// which `refinement_forms::element` would refuse at construction
    /// the moment the value crossed into a `one_of` set
    /// (showcase.py's `record_ratio(inf - inf)` row).
    #[test]
    fn test_binary_arithmetic_value_inf_minus_inf_is_the_nan_state_not_a_panic() {
        let positive_infinity = known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value(Operator::Sub, &positive_infinity, &positive_infinity);
        assert_eq!(result.kind, Kind::NaN, "{result:?}");
    }

    /// `inf * 0` — a Float `Mult` result that is NaN (IEEE 754), the
    /// second of showcase.py's three NaN-producing rows
    /// (`record_ratio(inf * 0)`). Same non-panicking `Kind::NaN` answer
    /// as the `Sub` row above.
    #[test]
    fn test_binary_arithmetic_value_inf_times_zero_is_the_nan_state_not_a_panic() {
        let positive_infinity = known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved);
        let zero = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value(Operator::Mult, &positive_infinity, &zero);
        assert_eq!(result.kind, Kind::NaN, "{result:?}");
    }

    /// `inf / inf` — a non-zero divisor (so the `ZeroDivisionError`
    /// decline does not apply), still NaN by IEEE 754. Pins the `Div`
    /// row's own route through `arithmetic_result` rather than a direct
    /// `known_values` call.
    #[test]
    fn test_binary_arithmetic_value_inf_over_inf_is_the_nan_state_not_a_panic() {
        let positive_infinity = known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value(Operator::Div, &positive_infinity, &positive_infinity);
        assert_eq!(result.kind, Kind::NaN, "{result:?}");
    }

    /// `{1.0, 2.0} * 2.0` — a MULTI-valued `Kind::Values` operand
    /// against a single-valued one: the exact pointwise answer `{2.0,
    /// 4.0}`, not `unknown()`. This is the row a loop's second judged
    /// pass needs: a first-pass join can leave `total` bound to exactly
    /// this two-element shape, and a decline here is what collapses a
    /// stabilizing accumulation onto the coarse "not yet walked"
    /// blocker instead of the fixed-point one.
    #[test]
    fn test_binary_arithmetic_value_multi_valued_operand_answers_the_pointwise_cross_product() {
        let one_and_two = known_values(vec![1.0, 2.0], PrimitiveKind::Float, TrustProved);
        let two = known_values(vec![2.0], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value(Operator::Mult, &one_and_two, &two);
        assert_eq!(result.kind, Kind::Values, "{result:?}");
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
        let mut values = result.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![2.0, 4.0]);
    }

    /// `{1.0, 2.0} + {10.0, 20.0}` — BOTH operands multi-valued: the
    /// full cross product, four pointwise sums, deduped (none collide
    /// here) — `1+10, 1+20, 2+10, 2+20`.
    #[test]
    fn test_binary_arithmetic_value_both_operands_multi_valued_answers_the_full_cross_product() {
        let one_and_two = known_values(vec![1.0, 2.0], PrimitiveKind::Float, TrustProved);
        let ten_and_twenty = known_values(vec![10.0, 20.0], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value(Operator::Add, &one_and_two, &ten_and_twenty);
        assert_eq!(result.kind, Kind::Values, "{result:?}");
        let mut values = result.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![11.0, 12.0, 21.0, 22.0]);
    }

    /// A cross product past `MULTI_VALUE_CROSS_PRODUCT_CAP` falls
    /// through to whatever the existing set/transfer path answers today
    /// — pinned as NOT `Kind::Values` (this function's own multi-value
    /// row must not fire), rather than pinning a specific set shape the
    /// set/transfer path's own tests already own.
    #[test]
    fn test_binary_arithmetic_value_cross_product_past_the_cap_falls_through() {
        let left_values: Vec<f64> = (0..5).map(|n| n as f64).collect();
        let right_values: Vec<f64> = (0..5).map(|n| 100.0 + n as f64).collect();
        let left = known_values(left_values, PrimitiveKind::Float, TrustProved);
        let right = known_values(right_values, PrimitiveKind::Float, TrustProved);
        // 5 * 5 = 25 pairs, past the 16-pair cap
        let result = binary_arithmetic_value(Operator::Add, &left, &right);
        assert_ne!(
            result.kind,
            Kind::Values,
            "a cross product past the cap must fall through, not answer Kind::Values: {result:?}"
        );
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

    /// `&`/`|`/`^` on two known int-sorted values are exact per §6.8 —
    /// pins `40 | 200 == 232` (CPython-checked), the exact fold
    /// `compound_bitwise_on_number_slot`'s `age |= 200` depends on to
    /// carry a judgeable value past Age's 120 ceiling instead of
    /// declining to unknown().
    #[test]
    fn test_binary_arithmetic_value_bitwise_or_is_exact() {
        let forty = known_values(vec![40.0], PrimitiveKind::Integer, TrustProved);
        let two_hundred = known_values(vec![200.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value(Operator::BitOr, &forty, &two_hundred);
        assert_eq!(result.values, vec![232.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `&`/`^` follow the same exact two's-complement law as `|` —
    /// CPython-checked: `5 & 3 == 1`, `5 ^ 3 == 6`.
    #[test]
    fn test_binary_arithmetic_value_bitwise_and_xor_are_exact() {
        let five = known_values(vec![5.0], PrimitiveKind::Integer, TrustProved);
        let three = known_values(vec![3.0], PrimitiveKind::Integer, TrustProved);
        let and_result = binary_arithmetic_value(Operator::BitAnd, &five, &three);
        assert_eq!(and_result.values, vec![1.0]);
        let xor_result = binary_arithmetic_value(Operator::BitXor, &five, &three);
        assert_eq!(xor_result.values, vec![6.0]);
    }

    /// `<<`/`>>` on two known int-sorted values are exact per §6.8:
    /// `x << n` is `x * 2**n`, `x >> n` floors `x / 2**n` — CPython-
    /// checked: `1 << 5 == 32`, `(-8) >> 2 == -2` (floors toward
    /// negative infinity, not truncates toward zero).
    #[test]
    fn test_binary_arithmetic_value_shifts_are_exact() {
        let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
        let five = known_values(vec![5.0], PrimitiveKind::Integer, TrustProved);
        let left_shifted = binary_arithmetic_value(Operator::LShift, &one, &five);
        assert_eq!(left_shifted.values, vec![32.0]);

        let negative_eight = known_values(vec![-8.0], PrimitiveKind::Integer, TrustProved);
        let two = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
        let right_shifted = binary_arithmetic_value(Operator::RShift, &negative_eight, &two);
        assert_eq!(right_shifted.values, vec![-2.0]);
    }

    /// A negative shift count raises ValueError in CPython — this file
    /// has no exception channel for a binary operator's own decline, so
    /// it declines to unknown() rather than claim a value CPython never
    /// produces.
    #[test]
    fn test_binary_arithmetic_value_negative_shift_declines() {
        let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
        let negative_one = known_values(vec![-1.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value(Operator::LShift, &one, &negative_one);
        assert_eq!(result.kind, Kind::Unknown);
    }

    /// A float operand to a bitwise op raises TypeError in CPython
    /// (unsupported operand type) — `single_numeric_value` reads a bare
    /// Float-sorted operand as non-int, so `both_int` is false and this
    /// declines rather than guess a two's-complement pattern for a
    /// value that was never an int.
    #[test]
    fn test_binary_arithmetic_value_bitwise_float_operand_declines() {
        let one_float = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
        let one_int = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value(Operator::BitAnd, &one_float, &one_int);
        assert_eq!(result.kind, Kind::Unknown);
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
    /// an unbounded operand. The answer is the SORT-ONLY unbounded set
    /// (the same language-level guarantee the math family carries), not
    /// nothing: the product of two numerics is a numeric, and a
    /// downstream clamp can still bound it. The BOUNDED-set row above
    /// is where the transfer certifies a tight image; this row pins
    /// that an unbounded operand keeps its sort and loses its bounds —
    /// never a guessed value, never a dropped one.
    #[test]
    fn test_float_sorted_set_times_known_int_answers_the_sort_when_unbounded() {
        let Some(kernel) = loaded_kernel() else { return };
        let sqrt_result = float_sorted_unknown();
        let two = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Mult, &sqrt_result, &two, &kernel);
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
        let everything = refined_sets::refinement_forms::numbers();
        assert!((kernel.scalar_subset)(&result.set, &everything), "the sort-only answer must stay inside the numeric line");
        assert!((kernel.scalar_subset)(&everything, &result.set), "the sort-only answer must not invent bounds the transfer never certified");
    }

    /// `age % 7` where `age` is a seeded Integer-sorted set `[0, 120]` —
    /// `admitted_int_transfer_op` elects `rem.divisorSign` for `Mod` on
    /// the int-sorted path (arith.4, the Python-owned remainder), so
    /// `int_transfer_over_sets` asks the kernel rather than declining.
    /// `theories/rem/divisor_sign.lean`'s own general-enclosure branch,
    /// worked by hand for this exact operand pair: `age` is a range (not
    /// a singleton), `7` is a singleton nonzero divisor, so the answer
    /// comes from the `bothSingle = none` arm — `divisorBound = 7`
    /// (finite), both operands Integer-sorted and `7` itself an integer
    /// dyadic, so the TIGHTENED case applies (`magnitude = 7 − 1 = 6`);
    /// the divisor is nonnegative, so the window sits on `[0, magnitude]`
    /// with neither endpoint strict — `[0, 6]`, matching the fixture's
    /// own `int_modulo_over_declared_range` row (`b-body-expressions.py`,
    /// "`count % 7` lands in Remainder's 0..6"). Asserted via
    /// `scalar_subset` both directions, the same pinning style
    /// `test_set_plus_known_int_lowers_through_kernel_transfer` uses.
    #[test]
    fn test_mod_over_an_int_sorted_set_serves_the_divisor_sign_row() {
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
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
        let want = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(6.0)]);
        assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
        assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
    }

    /// The FLOAT-path exclusion: `age % 7.0` where `age` is a
    /// Float-sorted set — `admitted_transfer_op` (the float/mixed-sort
    /// path `int_transfer_over_sets` falls through to whenever either
    /// operand is not Integer-sorted) has no `Mod` arm at all, so
    /// `transfer_over_sets` declines outright, and
    /// `binary_arithmetic_value_with_kernel` falls through to the
    /// ordinary known-values path, which also declines (a Set is not one
    /// known value) — the whole call answers `unknown()`. `%`'s
    /// divisor-sign election is admitted ONLY on the exact int theory
    /// (`rem.divisorSign` has no float-sorted counterpart wired here);
    /// this is the row the now-renamed test above no longer covers.
    #[test]
    fn test_mod_over_a_float_sorted_set_still_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let age = known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let age = AbstractValue { kind_tag: Some(PrimitiveKind::Float), ..age };
        let seven = known_values(vec![7.0], PrimitiveKind::Float, TrustProved);
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let imports = crate::surface::surface_imports(&module);
        let classes = std::sync::Arc::new(crate::instances::class_table(
            &module, &aliases, &imports, &kernel,
        ));
        let mut environment = empty_environment();
        environment.set_classes(classes);
        let parsed = parse_expression("Person(40)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(
            crate::instances::field_read(&value, "age"),
            Some(known_values(vec![40.0], PrimitiveKind::Integer, TrustProved))
        );
    }

    /// b-body-expressions.py's own `binary_chained_builder_call` shape:
    /// TWO same-module defs each declare their own `class Builder`, with
    /// DIFFERENT `size` method bodies. `environment.classes()` is set to
    /// the COLLAPSED table `check.rs::local_class_table`'s own
    /// first-scanned-wins merge would build for the enclosing body
    /// (`make_ok_builder`'s Builder, the one whose `size` returns
    /// `"ab"`) — the exact stale, shared guess a chained call must NOT
    /// trust. `make_over_builder().type("x").size(1)` still answers
    /// `"too-long-str"`, `make_over_builder`'s OWN `size`, proving
    /// `receiver_def_local_classes` traces the chain back to the right
    /// def instead of reading the collapsed table.
    #[test]
    fn test_chained_call_on_a_same_named_sibling_local_class_reads_its_own_def() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "def make_ok_builder():\n",
            "    class Builder:\n",
            "        def type(self, _t):\n",
            "            return self\n",
            "        def size(self, _n):\n",
            "            return \"ab\"\n",
            "    return Builder()\n",
            "def make_over_builder():\n",
            "    class Builder:\n",
            "        def type(self, _t):\n",
            "            return self\n",
            "        def size(self, _n):\n",
            "            return \"too-long-str\"\n",
            "    return Builder()\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
        let aliases = std::collections::HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let ruff_python_ast::Stmt::FunctionDef(make_ok_builder) = &module.body[0] else {
            panic!("module's first statement is def make_ok_builder")
        };
        // the STALE, collapsed table: only `make_ok_builder`'s own
        // `Builder` (whose `size` answers "ab") — the first-scanned-wins
        // shape `local_class_table`'s recursive merge would leave behind
        // for a body enclosing both nested defs.
        let stale_classes = std::sync::Arc::new(crate::instances::class_table(
            &ruff_python_ast::ModModule {
                node_index: ruff_python_ast::AtomicNodeIndex::NONE,
                range: TextRange::default(),
                body: make_ok_builder
                    .body
                    .iter()
                    .filter(|stmt| matches!(stmt, ruff_python_ast::Stmt::ClassDef(_)))
                    .cloned()
                    .collect(),
            },
            &aliases,
            &imports,
            &kernel,
        ));
        let mut environment = empty_environment();
        environment.set_functions(table);
        environment.set_classes(stale_classes);
        let parsed = parse_expression("make_over_builder().type(\"x\").size(1)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, string_models::string_literal_value("too-long-str").values);
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
        let imports = crate::surface::surface_imports(&module);
        let classes = std::sync::Arc::new(crate::instances::class_table(
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
        let imports = crate::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, kernel));
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
        let imports = crate::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, &kernel));
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
            crate::instances::field_read(still_bound, "count"),
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
        let imports = crate::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, &kernel));
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
        let imports = crate::surface::surface_imports(&module);
        let classes =
            std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, &kernel));
        let child = classes.get("Child").expect("Child class recorded");
        let constructed_child = crate::instances::judge_construction(child, &[], &[], &kernel).instance;
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

    /// `math.log(-2)`/`math.log2(-2)`/`math.log10(-2)`: a KNOWN operand
    /// entirely inside CPython's raise domain (`x <= 0`) fires the
    /// determined "math domain error" finding, one shared row per
    /// `DomainLimitedFamily::of_function`.
    #[test]
    fn test_provable_raise_math_log_family_of_a_known_nonpositive() {
        for source in ["math.log(-2)", "math.log2(-2)", "math.log10(-2)"] {
            let Some(found) = provable_raise_of(source) else {
                if loaded_kernel().is_none() {
                    return;
                }
                panic!("{source} must provably raise");
            };
            assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
            assert!(found.1.contains("math domain error"), "{source}: {}", found.1);
        }
    }

    /// `math.log(0.0)` provably raises — the module's own worked
    /// example (specifications/python/Doc/library/math.rst:696-698) and
    /// the exact JS/Python divergence point: the kernel's own `js.log`
    /// arm serves `-inf` there (JavaScript's `Math.log(0) ===
    /// -Infinity`), but CPython's `loghelper`/`math_1` (mathmodule.c)
    /// raises `ValueError` for an infinite result from a finite input.
    #[test]
    fn test_provable_raise_math_log_of_exact_zero_the_python_javascript_divergence_point() {
        let Some(found) = provable_raise_of("math.log(0.0)") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("math.log(0.0) must provably raise — the module's own worked ValueError example");
        };
        assert!(found.1.contains("ValueError"), "{}", found.1);
        assert!(found.1.contains("math domain error"), "{}", found.1);
    }

    /// `math.log1p(-2.0)` (entirely `x <= -1`) provably raises; the
    /// exact boundary point `math.log1p(-1.0)` ALSO raises (the closed
    /// `x <= -1` domain, not the kernel's open `x < -1` NaN corner) —
    /// `jsLog1p` serves `-inf` there, another JS/Python divergence.
    #[test]
    fn test_provable_raise_math_log1p_of_nonpositive_and_its_exact_boundary() {
        for source in ["math.log1p(-2.0)", "math.log1p(-1.0)"] {
            let Some(found) = provable_raise_of(source) else {
                if loaded_kernel().is_none() {
                    return;
                }
                panic!("{source} must provably raise");
            };
            assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
            assert!(found.1.contains("math domain error"), "{source}: {}", found.1);
        }
    }

    /// `math.asin(2.0)`/`math.acos(-2.0)`: entirely outside `[-1, 1]`
    /// fires the determined finding; `math.asin(1.0)` (the CLOSED
    /// boundary) does NOT raise — `asin`/`acos`'s raise domain is the
    /// OPEN ray `|x| > 1`, matching the kernel's own boundary exactly
    /// (no JS/Python divergence for this family).
    #[test]
    fn test_provable_raise_math_asin_acos_outside_domain_and_boundary_declines() {
        let Some(found) = provable_raise_of("math.asin(2.0)") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("math.asin(2.0) must provably raise");
        };
        assert!(found.1.contains("math domain error"), "{}", found.1);

        let Some(found) = provable_raise_of("math.acos(-2.0)") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("math.acos(-2.0) must provably raise");
        };
        assert!(found.1.contains("math domain error"), "{}", found.1);

        assert!(provable_raise_of("math.asin(1.0)").is_none(), "asin(1.0) = pi/2 exactly — must not raise");
        assert!(provable_raise_of("math.acos(-1.0)").is_none(), "acos(-1.0) = pi exactly — must not raise");
    }

    /// `math.atanh(2.0)` (entirely `|x| >= 1`) provably raises; the
    /// exact boundary points `math.atanh(1.0)`/`math.atanh(-1.0)` ALSO
    /// raise (the closed `|x| >= 1` domain) — `jsAtanh` serves `±inf`
    /// there, another JS/Python divergence this family's own raise
    /// domain must be one ray WIDER than the kernel's boundary to catch.
    #[test]
    fn test_provable_raise_math_atanh_outside_domain_and_its_exact_boundary() {
        for source in ["math.atanh(2.0)", "math.atanh(1.0)", "math.atanh(-1.0)"] {
            let Some(found) = provable_raise_of(source) else {
                if loaded_kernel().is_none() {
                    return;
                }
                panic!("{source} must provably raise");
            };
            assert!(found.1.contains("ValueError"), "{source}: {}", found.1);
            assert!(found.1.contains("math domain error"), "{source}: {}", found.1);
        }
    }

    /// `math.acosh(0.5)`: entirely `x < 1` fires; `math.acosh(1.0)` (the
    /// CLOSED boundary, `acosh(1) = 0` exactly) does NOT raise —
    /// `acosh`'s raise domain is the OPEN ray `x < 1`, matching the
    /// kernel's own boundary exactly (no JS/Python divergence).
    #[test]
    fn test_provable_raise_math_acosh_below_one_and_boundary_declines() {
        let Some(found) = provable_raise_of("math.acosh(0.5)") else {
            if loaded_kernel().is_none() {
                return;
            }
            panic!("math.acosh(0.5) must provably raise");
        };
        assert!(found.1.contains("math domain error"), "{}", found.1);
        assert!(provable_raise_of("math.acosh(1.0)").is_none(), "acosh(1.0) = 0 exactly — must not raise");
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
    // Every `.timestamp()` pin below routes its day-count arithmetic
    // through the kernel's `calendar` ask (`refined_calendar`'s
    // `"epochDays"` op, `epoch_days_of_civil_date`) rather than a local
    // Rust reimplementation — a wrong or refused kernel answer fails
    // these pins directly, since `eval` loads the real kernel dylib.

    /// `datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc).timestamp()`
    /// is exactly `0.0` — the POSIX epoch itself, the kernel's own
    /// `epochDays` anchor (`theories/calendar/epoch_days_sound.lean`).
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

    /// `datetime.datetime(2024, 2, 29, tzinfo=datetime.timezone.utc).timestamp()`
    /// — a leap-day date (2024 is divisible by 4, not by 100): the day
    /// count the kernel's `epochDays` ask must cross a Gregorian leap
    /// boundary to answer, execution-verified against installed CPython
    /// 3.12 (`(datetime.datetime(2024, 2, 29, tzinfo=datetime.timezone.utc)
    /// - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc))
    /// .total_seconds() == 1709164800.0`).
    #[test]
    fn test_datetime_timestamp_of_a_leap_day_crosses_the_kernels_calendar() {
        let Some(value) = eval("datetime.datetime(2024, 2, 29, tzinfo=datetime.timezone.utc).timestamp()") else { return };
        assert_eq!(value.values, vec![1709164800.0]);
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

    // --- j-stdlib-surfaces.py: date/timedelta family ---
    // Every pin below routes its calendar arithmetic through the
    // kernel's `calendar` ask — `validDate`/`epochDays`/`isoDate`/
    // `validDuration` (construction and `date ± timedelta`) and
    // `weekday`/`toordinal`/`pyYearInRange`/`isoCalendar` (`.weekday()`/
    // `.isoweekday()`/`.toordinal()`/`.isocalendar()` and the year-range
    // guard) — same as the `datetime_datetime` family above; `eval`
    // loads the real kernel dylib, so a wrong or refused kernel answer
    // fails these pins directly. PIN VALUE PROVENANCE: `date(2024, 3,
    // 1).weekday() == 4` and `.toordinal() == 738946` are the exact
    // values this task's own brief states; every other constant below
    // is derived by `/tmp/date_pin_values.py` (a CPython `datetime`
    // probe) and MUST be cross-checked against that script's printed
    // output before this batch gates — flagged individually below.

    /// `datetime.date(2024, 3, 1)` constructs — a plain valid civil
    /// date, tagged and carrying its own year/month/day fields.
    #[test]
    fn test_date_construction_carries_its_own_fields() {
        let Some(value) = eval("datetime.date(2024, 3, 1)") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(3.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// `datetime.date(2023, 2, 30)` — February has 28 days in 2023 (not
    /// a leap year); the kernel's own `validDate` refuses this, so
    /// construction declines rather than building an invalid instance.
    #[test]
    fn test_date_construction_of_an_invalid_calendar_date_declines() {
        let Some(value) = eval("datetime.date(2023, 2, 30)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    // --- import aliasing: the datetime gates resolve canonical identity,
    // not the literal `datetime`/`date`/`timedelta` spelling ---

    /// One module's `datetime` import table, seeded onto a fresh
    /// environment the same way `check.rs::walk_body_with_self_binding`
    /// seeds it for a real walk — the harness every aliasing pin below
    /// shares.
    fn environment_with_datetime_imports(module: &ruff_python_ast::ModModule) -> Environment {
        let mut environment = empty_environment();
        environment.set_datetime_imports(Arc::new(datetime_imports(module)));
        environment
    }

    /// `from datetime import date` + `date(2024, 3, 1)` — a bare aliased
    /// class name construction. Recognizes IDENTICALLY to the qualified
    /// `datetime.date(2024, 3, 1)` spelling (`test_date_construction_
    /// carries_its_own_fields`'s own pin): same tag, same three fields.
    #[test]
    fn test_bare_imported_date_construction_matches_the_qualified_spelling() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("from datetime import date\n")
            .expect("test module parses")
            .into_syntax();
        let environment = environment_with_datetime_imports(&module);
        let parsed = parse_expression("date(2024, 3, 1)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        let qualified = parse_expression("datetime.date(2024, 3, 1)").expect("test source must parse");
        let qualified_value = evaluate_expression(&qualified.into_expr(), &empty_environment(), &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(value, qualified_value, "an aliased bare-Name construction must equal the qualified spelling's own pin");
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(3.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// `from datetime import datetime as dt` + `dt.strptime("2024-03-01",
    /// "%Y-%m-%d")` — a bare aliased class name's own classmethod call.
    /// Recognizes the same ISO-date STAGE 1 grammar the qualified
    /// `datetime.datetime.strptime(...)` spelling already pins
    /// (`test_strptime_...` rows above), landing on the same
    /// `datetime_datetime`-tagged instance a direct `datetime.datetime(
    /// 2024, 3, 1)` construction gives (`strptime_iso_date_value`'s own
    /// doc: date-only, hour/minute/second all zero).
    #[test]
    fn test_aliased_datetime_strptime_recognizes() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("from datetime import datetime as dt\n")
            .expect("test module parses")
            .into_syntax();
        let environment = environment_with_datetime_imports(&module);
        let parsed = parse_expression("dt.strptime(\"2024-03-01\", \"%Y-%m-%d\")").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(3.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// `import datetime as dtm` + `dtm.date(2024, 3, 1)` — the whole
    /// MODULE aliased (not one class), the qualified-chain shape
    /// resolved through the module alias rather than the literal
    /// `datetime` spelling. Recognizes identically to the unaliased
    /// `datetime.date(2024, 3, 1)` construction.
    #[test]
    fn test_module_aliased_date_construction_recognizes() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("import datetime as dtm\n")
            .expect("test module parses")
            .into_syntax();
        let environment = environment_with_datetime_imports(&module);
        let parsed = parse_expression("dtm.date(2024, 3, 1)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(3.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// A LOCALLY SHADOWED imported name never recognizes — `date` here
    /// is a same-module `def`, never `from datetime import date`
    /// (the import table's own `date_class_names` set stays empty since
    /// no such import statement exists), mirroring `surface.rs`'s own
    /// `locally_defined_field_not_recognized` pin: a same-spelled local
    /// definition that was never the real import is not the shape this
    /// table names, so the same-module-def dispatch (`same_module_def_
    /// gate_open`) answers the call instead of the datetime gate ever
    /// running — `date(2024, 3, 1)` calls the LOCAL zero-argument `def`
    /// (which takes no `year`/`month`/`day`, so the call is unread) and
    /// never reads as a tagged `datetime_date` instance.
    #[test]
    fn test_locally_defined_date_name_not_recognized_as_datetime() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module(concat!(
            "def date():\n",
            "    pass\n",
        ))
        .expect("test module parses")
        .into_syntax();
        let table = Arc::new(crate::function_table::function_table(&module));
        let mut environment = environment_with_datetime_imports(&module);
        environment.set_functions(table);
        let parsed = parse_expression("date(2024, 3, 1)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_ne!(value.kind, Kind::Object, "a locally defined `date` must never read as a datetime_date instance");
    }

    /// A REBOUND imported name never recognizes — `date` is genuinely
    /// `from datetime import date`, but this body's own `date = 40`
    /// rebinds it before the call. `is_datetime_date_attribute`'s own
    /// shadow check (`environment.read(name).is_none()`) must see the
    /// rebinding and decline, the same way the qualified `datetime.date`
    /// spelling already declines when `datetime` itself is rebound.
    #[test]
    fn test_locally_rebound_imported_date_name_not_recognized() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("from datetime import date\n")
            .expect("test module parses")
            .into_syntax();
        let mut environment = environment_with_datetime_imports(&module);
        environment.bind("date", known_values(vec![40.0], PrimitiveKind::Integer, TrustProved));
        let parsed = parse_expression("date(2024, 3, 1)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_ne!(value.kind, Kind::Object, "a locally rebound `date` must never read as a datetime_date instance");
    }

    /// `datetime.date(2024, 3, 1).weekday()` — PIN VALUE FROM THE
    /// TASK BRIEF ITSELF: 4 (Friday), Monday-0 through Sunday-6.
    #[test]
    fn test_date_weekday_of_a_known_friday() {
        let Some(value) = eval("datetime.date(2024, 3, 1).weekday()") else { return };
        assert_eq!(value.values, vec![4.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `datetime.date(2024, 3, 1).isoweekday()` — PIN VALUE DERIVED BY
    /// THE PROBE (`/tmp/date_pin_values.py`'s `isoweekday()` row):
    /// Monday-1 through Sunday-7, one more than `.weekday()`'s Friday-4.
    #[test]
    fn test_date_isoweekday_of_a_known_friday() {
        let Some(value) = eval("datetime.date(2024, 3, 1).isoweekday()") else { return };
        assert_eq!(value.values, vec![5.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `datetime.date(1970, 1, 1).weekday()` — the epoch anchor date,
    /// PIN VALUE DERIVED BY THE PROBE: CPython's own epoch is a
    /// Thursday (`isoDayOfWeek_epoch_thursday`'s proved fact, weekday()
    /// Monday-0 form: Thursday is 3).
    #[test]
    fn test_date_weekday_at_the_epoch_anchor() {
        let Some(value) = eval("datetime.date(1970, 1, 1).weekday()") else { return };
        assert_eq!(value.values, vec![3.0]);
    }

    /// `datetime.date(2024, 3, 1).toordinal()` — PIN VALUE FROM THE
    /// TASK BRIEF ITSELF: 738946.
    #[test]
    fn test_date_toordinal_of_a_known_date() {
        let Some(value) = eval("datetime.date(2024, 3, 1).toordinal()") else { return };
        assert_eq!(value.values, vec![738946.0]);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `datetime.date(1, 1, 1).toordinal()` — PIN VALUE FROM THE KERNEL'S
    /// OWN PROVED THEOREM (`ordinal.lean`'s `pyToOrdinal_anchor_is_one`,
    /// closed by `decide`): exactly 1, "January 1 of year 1 has ordinal
    /// 1" (datetime.rst:525-526).
    #[test]
    fn test_date_toordinal_anchor_is_exactly_one() {
        let Some(value) = eval("datetime.date(1, 1, 1).toordinal()") else { return };
        assert_eq!(value.values, vec![1.0]);
    }

    /// `datetime.timedelta(days=5)` constructs — a plain valid duration,
    /// tagged and carrying its own `days` field.
    #[test]
    fn test_timedelta_construction_carries_its_days_field() {
        let Some(value) = eval("datetime.timedelta(days=5)") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(datetime_field(&value, "days"), Some(5.0));
    }

    /// `datetime.timedelta(hours=5)` — a keyword this file does not
    /// read (only `days=` is modeled); the whole construction declines
    /// rather than silently dropping the field.
    #[test]
    fn test_timedelta_construction_with_an_unmodeled_keyword_declines() {
        let Some(value) = eval("datetime.timedelta(hours=5)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date(2024, 3, 1) + datetime.timedelta(days=31)` — PIN
    /// VALUE DERIVED BY THE PROBE: 2024-03-01 plus 31 days crosses into
    /// April, landing on 2024-04-01.
    #[test]
    fn test_date_plus_timedelta_crosses_a_month_boundary() {
        let Some(value) = eval("datetime.date(2024, 3, 1) + datetime.timedelta(days=31)") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(4.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// `datetime.timedelta(days=31) + datetime.date(2024, 3, 1)` — the
    /// REVERSED operand order (datetime.rst states the operation both
    /// ways); must answer the identical date the forward order gives.
    #[test]
    fn test_timedelta_plus_date_reversed_operand_order_agrees() {
        let Some(value) = eval("datetime.timedelta(days=31) + datetime.date(2024, 3, 1)") else { return };
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(4.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// `datetime.date(2024, 3, 1) - datetime.timedelta(days=1)` — PIN
    /// VALUE DERIVED BY THE PROBE: one day before March 1st on a leap
    /// year is February 29th (2024 IS a leap year).
    #[test]
    fn test_date_minus_timedelta_crosses_back_into_a_leap_february() {
        let Some(value) = eval("datetime.date(2024, 3, 1) - datetime.timedelta(days=1)") else { return };
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(2.0));
        assert_eq!(datetime_field(&value, "day"), Some(29.0));
    }

    /// `datetime.date(9999, 12, 31) + datetime.timedelta(days=1)` —
    /// datetime.rst's own `OverflowError` row (date.7): MAXYEAR is 9999,
    /// so this shift leaves the representable range and declines
    /// through the `pyYearInRange` kernel ask (`python_year_in_range`)
    /// on the shifted result's year (10000) — the kernel's `isoDate` arm
    /// alone would answer this shift (its own PlainDate window is far
    /// wider than Python's), so the decline is `pyYearInRange`'s doing.
    #[test]
    fn test_date_plus_timedelta_past_maxyear_declines() {
        let Some(value) = eval("datetime.date(9999, 12, 31) + datetime.timedelta(days=1)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date.fromisoformat("2024-03-01")` — the strict
    /// `YYYY-MM-DD` grammar (date.3's own committed shape), landing on
    /// the exact same tagged instance `datetime.date(2024, 3, 1)`
    /// constructs directly.
    #[test]
    fn test_date_fromisoformat_parses_the_strict_grammar() {
        let Some(value) = eval("datetime.date.fromisoformat(\"2024-03-01\")") else { return };
        assert_eq!(value.kind, Kind::Object);
        assert_eq!(datetime_field(&value, "year"), Some(2024.0));
        assert_eq!(datetime_field(&value, "month"), Some(3.0));
        assert_eq!(datetime_field(&value, "day"), Some(1.0));
    }

    /// `datetime.date.fromisoformat("2023-02-30")` — syntactically the
    /// right shape (three hyphen-separated all-digit runs of the right
    /// width) but calendrically invalid; declines through the SAME
    /// `calendar.validDate` kernel ask `date_construction_value` uses.
    #[test]
    fn test_date_fromisoformat_of_a_calendrically_invalid_string_declines() {
        let Some(value) = eval("datetime.date.fromisoformat(\"2023-02-30\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date.fromisoformat("2024-3-1")` — the reduced-width
    /// (non-zero-padded) spelling; date.3's own committed grammar is
    /// exactly `YYYY-MM-DD` (fixed widths), so this shape declines
    /// rather than guess a looser parse.
    #[test]
    fn test_date_fromisoformat_of_a_non_zero_padded_string_declines() {
        let Some(value) = eval("datetime.date.fromisoformat(\"2024-3-1\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date(2024, 3, 1).isocalendar()` — PIN VALUE FROM THE
    /// COORDINATOR'S OWN BRIEF (backed by the kernel's `isoCalendar` arm
    /// AND the Lean witness landing alongside it): `(2024, 9, 5)` — ISO
    /// year 2024, ISO week 9, ISO weekday 5 (Friday, the same Friday
    /// `.weekday() == 4`/`.isoweekday() == 5` already pin above). Binds
    /// as a known 3-element tuple, the same `Kind::List` shape a literal
    /// `(a, b, c)` display builds.
    #[test]
    fn test_date_isocalendar_of_a_known_date() {
        let Some(value) = eval("datetime.date(2024, 3, 1).isocalendar()") else { return };
        assert_eq!(value.kind, Kind::List);
        assert_eq!(
            value.items,
            vec![
                known_values(vec![2024.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![9.0], PrimitiveKind::Integer, TrustProved),
                known_values(vec![5.0], PrimitiveKind::Integer, TrustProved),
            ]
        );
    }

    /// `datetime.date(9999, 12, 31) + datetime.timedelta(days=1)` posed
    /// a SECOND way: this is the same construct
    /// `test_date_plus_timedelta_past_maxyear_declines` above already
    /// pins, restated here to name explicitly that the decline is now
    /// the `pyYearInRange` kernel ask's own `valid: false` answer (year
    /// 10000), not an adapter-local bound check.
    #[test]
    fn test_date_plus_timedelta_past_maxyear_declines_via_the_kernel_year_range_ask() {
        let Some(value) = eval("datetime.date(9999, 12, 31) + datetime.timedelta(days=1)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    // --- j-stdlib-surfaces.py: strftime/strptime STAGE 1 (date.12) ---

    /// `datetime.datetime.strptime("2024-03-01", "%Y-%m-%d")` binds the
    /// EXACT SAME value `datetime.date.fromisoformat("2024-03-01")`
    /// does — `strptime_iso_date_value`'s own doc: one recognition, the
    /// existing `date_fromisoformat_value` machinery, no new kernel
    /// question. Asserts equality of the two paths' values directly,
    /// not just their shape.
    #[test]
    fn test_strptime_iso_date_agrees_with_fromisoformat() {
        let Some(via_strptime) = eval("datetime.datetime.strptime(\"2024-03-01\", \"%Y-%m-%d\")") else { return };
        let Some(via_fromisoformat) = eval("datetime.date.fromisoformat(\"2024-03-01\")") else { return };
        assert_eq!(via_strptime, via_fromisoformat);
        assert_eq!(via_strptime.kind, Kind::Object);
        assert_eq!(via_strptime.source.as_str(), "datetime_date");
    }

    /// `datetime.datetime.strptime("2023-02-30", "%Y-%m-%d")` — a
    /// calendrically invalid date (February has 28 days in 2023);
    /// declines through the SAME `validDate` kernel ask
    /// `date.fromisoformat("2023-02-30")` declines through, since
    /// `strptime_iso_date_value` reuses `date_fromisoformat_value`
    /// outright.
    #[test]
    fn test_strptime_of_an_invalid_date_declines_identically_to_fromisoformat() {
        let Some(via_strptime) = eval("datetime.datetime.strptime(\"2023-02-30\", \"%Y-%m-%d\")") else { return };
        let Some(via_fromisoformat) = eval("datetime.date.fromisoformat(\"2023-02-30\")") else { return };
        assert_eq!(via_strptime.kind, Kind::Unknown);
        assert_eq!(via_fromisoformat.kind, Kind::Unknown);
    }

    /// `datetime.datetime.strptime("2024-03-01", fmt)` where `fmt` is a
    /// PARAMETER (a computed format the source cannot name, never a
    /// written literal) — declines; this file has no format-code
    /// mini-language reader for an expression it cannot fold to an
    /// exact string at all.
    #[test]
    fn test_strptime_with_a_computed_format_declines() {
        let Some(value) = eval("datetime.datetime.strptime(\"2024-03-01\", fmt)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.datetime.strptime("01/03/2024", "%d/%m/%Y")` — a
    /// LITERAL format, but not the ISO `"%Y-%m-%d"` sequence this stage
    /// builds; names date.12 STAGE 2 (the directive-grammar kernel
    /// theory) as the reason, per `strptime_iso_date_value`'s own doc.
    #[test]
    fn test_strptime_with_a_non_iso_literal_format_declines() {
        let Some(value) = eval("datetime.datetime.strptime(\"01/03/2024\", \"%d/%m/%Y\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date(2024, 3, 1).strftime("%Y-%m-%d")` — the exact
    /// ISO literal format on a known date; still declines today, per
    /// `strftime_iso_date_value`'s own doc: the kernel's `isoDate` op
    /// answers no rendered-string field, only the four integer fields
    /// `year`/`month`/`day`/`dayOfWeek`.
    #[test]
    fn test_strftime_iso_format_on_a_known_date_declines_pending_a_render_export() {
        let Some(value) = eval("datetime.date(2024, 3, 1).strftime(\"%Y-%m-%d\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date(2024, 3, 1).strftime(fmt)` where `fmt` is a
    /// PARAMETER — declines, the same computed-format reason
    /// `test_strptime_with_a_computed_format_declines` states for the
    /// parse direction.
    #[test]
    fn test_strftime_with_a_computed_format_declines() {
        let Some(value) = eval("datetime.date(2024, 3, 1).strftime(fmt)") else { return };
        assert_eq!(value.kind, Kind::Unknown);
    }

    /// `datetime.date(2024, 3, 1).strftime("%d/%m/%Y")` — a non-ISO
    /// literal directive sequence; names date.12 STAGE 2, the same
    /// reason `test_strptime_with_a_non_iso_literal_format_declines`
    /// states for the parse direction.
    #[test]
    fn test_strftime_with_a_non_iso_literal_format_declines() {
        let Some(value) = eval("datetime.date(2024, 3, 1).strftime(\"%d/%m/%Y\")") else { return };
        assert_eq!(value.kind, Kind::Unknown);
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

    /// `json.loads(x)` over an operand this file holds no fact about (an
    /// unbound name, so `exact_string_values` reads nothing) answers the
    /// full JSON-union — every arm of `json_loads_value_space` — rather
    /// than bare `unknown()` (ISSUES.md, "generic json.loads of an
    /// opaque string answers bare unknown"). All seven shapes
    /// library/json.rst's conversion table admits ride as arms: None,
    /// bool, an unbounded string set, an unbounded int-sorted set, a
    /// value-unknown float-sorted set, and the two opaque list/dict
    /// arms.
    #[test]
    fn test_json_loads_of_an_opaque_operand_answers_the_full_json_union() {
        let Some(value) = eval("json.loads(x)") else { return };
        assert_eq!(value.kind, Kind::KindUnion);
        assert!(value.arms.iter().any(|arm| arm.kind == Kind::Null), "missing the None arm: {value:?}");
        assert!(
            value.arms.iter().any(|arm| arm.kind == Kind::Values && arm.kind_tag == Some(PrimitiveKind::Boolean)),
            "missing the bool arm: {value:?}"
        );
        // the str arm is untagged (kind_tag: None), matching the same
        // convention `__name__`'s own read builds (assignability.rs's
        // doc: an untagged Set whose own set is sequence-shaped reads
        // as string-sorted) — its own set is the full codepoint ground.
        assert!(
            value.arms.iter().any(|arm| arm.kind == Kind::Set && arm.kind_tag.is_none() && arm.set == strings()),
            "missing the str arm: {value:?}"
        );
        assert!(
            value.arms.iter().any(|arm| arm.kind == Kind::Set && arm.kind_tag == Some(PrimitiveKind::Integer)),
            "missing the int arm: {value:?}"
        );
        assert!(
            value.arms.iter().any(|arm| arm.kind == Kind::Set && arm.kind_tag == Some(PrimitiveKind::Float)),
            "missing the float arm: {value:?}"
        );
        assert!(
            value.arms.iter().any(|arm| arm.kind == Kind::Object && arm.kind_word == Some("a list")),
            "missing the list arm: {value:?}"
        );
        assert!(
            value.arms.iter().any(|arm| arm.kind == Kind::Object && arm.kind_word == Some("a dict")),
            "missing the dict arm: {value:?}"
        );
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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

    /// `bytearray(4)`/`bytearray(b"...")`/`bytes([...])`/
    /// `memoryview(bytearray(...))` each carry their own species word
    /// (`bytes_models::tagged`'s own doc) — `check.rs`'s write sink reads
    /// this to decide which of the three write rules applies. A plain
    /// list literal carries none of these words.
    #[test]
    fn test_bytearray_from_length_is_tagged_bytearray() {
        let Some(value) = eval("bytearray(4)") else { return };
        assert_eq!(value.kind_word, Some(bytes_models::BYTEARRAY_WORD));
    }

    #[test]
    fn test_bytearray_from_a_bytes_literal_is_tagged_bytearray() {
        let Some(value) = eval("bytearray(b\"\\x0a\\x14\")") else { return };
        assert_eq!(value.kind_word, Some(bytes_models::BYTEARRAY_WORD));
    }

    #[test]
    fn test_bytes_constructor_is_tagged_bytes() {
        let Some(value) = eval("bytes([10, 20, 30])") else { return };
        assert_eq!(value.kind_word, Some(bytes_models::BYTES_WORD));
    }

    #[test]
    fn test_memoryview_over_bytearray_is_tagged_memoryview_not_bytearray() {
        // the view's OWN word must win — a write through the view raises
        // the memoryview-specific wording, not bytearray's, even though
        // the wrapped argument was itself tagged bytearray.
        let Some(kernel) = loaded_kernel() else { return };
        let parsed = parse_expression("memoryview(bytearray(2))").expect("test source must parse");
        let environment = empty_environment();
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.kind_word, Some(bytes_models::MEMORYVIEW_WORD));
    }

    #[test]
    fn test_plain_list_literal_carries_no_bytes_species_word() {
        let Some(value) = eval("[10, 20, 30]") else { return };
        assert_eq!(value.kind_word, None);
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let empty_imports = crate::surface::surface_imports(&ruff_python_ast::ModModule {
            node_index: ruff_python_ast::AtomicNodeIndex::NONE,
            range: TextRange::default(),
            body: Vec::new().into(),
        });
        let classes = crate::instances::class_table(&module, &empty_aliases, &empty_imports, &kernel);
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
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
        let table = std::sync::Arc::new(crate::function_table::function_table(&module));
        let mut environment = empty_environment();
        environment.set_functions(table);
        let parsed = parse_expression("gather_kwargs(age=200)").expect("test source must parse");
        let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
        assert_eq!(value.values, vec![200.0]);
    }

    // --- `/` at a SET-SHAPED divisor that may admit zero ---

    /// `1.0 / denominator` where `denominator` is a seeded Float-sorted
    /// SET `[0.0, 2.0]` — a WIDE window admitting zero, but NOT entirely
    /// zero (`divisor_is_provably_always_zero` is false — the window
    /// has non-zero members too). `split_divisor_transfer`'s own fix:
    /// the value question no longer declines outright at this shape —
    /// it splits the divisor into its zero-excluded halves (`(0.0, 2.0]`
    /// here; the negative half, `< 0.0`, is empty and skipped) and asks
    /// `binary64.div` on the non-empty half. The kernel's OWN general-
    /// interval branch (`divisorMayBeZero`, `theories/binary64/div.lean`)
    /// still cannot narrow `1.0 / (0.0, 2.0]` to a tight enclosure even
    /// with zero excluded, so the split's own answer is `Unknown` —
    /// which this function reads as `float_sorted_unknown()` (sort-
    /// known, value-unknown), never `Kind::Unknown` outright. The value
    /// question DETERMINES a sort here, on the non-raising split, exactly
    /// as every other admitted transfer answer already does — the raise
    /// arm at `x == 0.0` itself is a separate, unaddressed question
    /// (`binop_provable_raise` only fires when the WHOLE window is zero).
    #[test]
    fn test_div_by_a_set_that_may_admit_zero_determines_the_float_sort_over_the_zero_excluded_split() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            result.kind,
            Kind::Set,
            "the zero-excluded split must determine a value (sort-only, at minimum), never decline outright: {result:?}"
        );
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    }

    /// The SOLE-GUARD row: `1.0 / denominator` where `denominator` is a
    /// DEGENERATE Set carrying nothing but `{0.0}` (`one_of`, `Kind::Set`
    /// rather than the ordinary `Kind::Values` `single_numeric_value`
    /// already reads). Unlike the wide-window row above, the kernel's
    /// OWN `bothSingle` branch (`theories/binary64/div.lean`) answers a
    /// DETERMINED `±Infinity` pair for this exact shape — so this row is
    /// the one `divisor_provably_excludes_zero` alone protects; without
    /// the gate, `transfer_over_sets` would relabel that pair as
    /// Python's answer, which is the unsound row this whole unit fixes.
    #[test]
    fn test_div_by_a_degenerate_zero_only_set_declines_where_the_kernel_would_otherwise_answer() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
        };
        let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            result.kind,
            Kind::Unknown,
            "a degenerate zero-only divisor Set must decline — the kernel's bothSingle branch answers \
             a determined ±Infinity pair here, and relabeling it as Python's answer is exactly the \
             unsoundness this gate exists to prevent: {result:?}"
        );
    }

    /// The mirror row: a divisor set that PROVABLY EXCLUDES zero (a
    /// window `[1.0, 2.0]`, strictly above zero) still lowers through
    /// `binary64.div` — the gate only refuses the zero-admitting case,
    /// it does not disable the SET path outright. `1.0 / [1.0, 2.0]`
    /// certifies to `[0.5, 1.0]`.
    #[test]
    fn test_div_by_a_set_that_provably_excludes_zero_still_lowers_through_the_kernel() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(1.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
        let result = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(result.kind, Kind::Set, "a zero-excluding divisor must still answer: {result:?}");
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
        let want = make_refined_set(vec![at_least(0.5), refined_sets::refinement_forms::at_most(1.0)]);
        assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
        assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
    }

    /// The pinning ask for `divisor_provably_excludes_zero` directly: a
    /// half-open ray `(0.0, ∞)` (strictly positive, zero itself NOT a
    /// member) excludes zero, while `[0.0, ∞)` (zero included) does not.
    #[test]
    fn test_divisor_provably_excludes_zero_reads_strict_vs_inclusive_bounds() {
        let Some(kernel) = loaded_kernel() else { return };
        let strictly_positive = make_refined_set(vec![refined_sets::refinement_forms::above(0.0)]);
        assert!(
            divisor_provably_excludes_zero(&strictly_positive, &kernel),
            "a strictly-positive ray must be proved to exclude zero"
        );
        let nonnegative = make_refined_set(vec![at_least(0.0)]);
        assert!(
            !divisor_provably_excludes_zero(&nonnegative, &kernel),
            "a nonnegative ray admits zero and must not be proved to exclude it"
        );
    }

    // --- `//`/`%` at a SET-SHAPED divisor that may admit zero ---

    /// `age // denominator` where `denominator` is a seeded Integer-sorted
    /// SET `[0, 5]` — the `//`/`%` twin of the `/` corner above, checked
    /// for the SAME hazard. `admitted_int_transfer_op`'s `int.floorDiv`
    /// row only ever answers over TWO EXACT SINGLETONS
    /// (`boundary/python.lean`'s `exactIntOf A, exactIntOf B` match); a
    /// range divisor is not a singleton, so the kernel itself refuses
    /// (`.unknown`) before any zero-admission question is even reached —
    /// this row is sound by construction, with no adapter-side gate
    /// needed. Pinned here so the finding is asserted, not merely
    /// claimed.
    #[test]
    fn test_floor_div_by_a_set_that_may_admit_zero_declines_because_the_kernel_refuses_ranges() {
        let Some(kernel) = loaded_kernel() else { return };
        let age = known_set(
            make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        let result = binary_arithmetic_value_with_kernel(Operator::FloorDiv, &age, &denominator, &kernel);
        assert_eq!(
            result.kind,
            Kind::Unknown,
            "a range divisor has no int.floorDiv row at all (exact singletons only) — declines: {result:?}"
        );
    }

    /// The `%` twin: `age % denominator` over the SAME `[0, 5]` divisor
    /// window. `rem.divisorSign` DOES have a general-interval branch
    /// (unlike `int.floorDiv`), so this is the row that actually
    /// exercises `theories/rem/divisor_sign.lean`'s own `divisorMayBeZero`
    /// gate rather than merely a singleton-only refusal: the kernel
    /// itself declines (`.unknown`) the moment the divisor's range
    /// admits zero, so the adapter's decline here is inherited soundly
    /// from the kernel, with no separate adapter-side gate needed for
    /// `Mod` either.
    #[test]
    fn test_mod_by_a_set_that_may_admit_zero_declines_because_the_kernel_gates_the_interval_branch() {
        let Some(kernel) = loaded_kernel() else { return };
        let age = known_set(
            make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        let result = binary_arithmetic_value_with_kernel(Operator::Mod, &age, &denominator, &kernel);
        assert_eq!(
            result.kind,
            Kind::Unknown,
            "rem.divisorSign's own divisorMayBeZero gate refuses a zero-admitting range: {result:?}"
        );
    }

    // --- `provable_raise` at a SET-SHAPED divisor ---

    /// A divisor set that is ALWAYS zero — a degenerate seeded window
    /// that has narrowed to nothing but `{0.0}` — provably raises, the
    /// SET-shaped twin of the scalar `1 / 0` row `test_provable_raise_
    /// zero_division` already pins. `divisor_is_provably_always_zero`
    /// is the check: the set is a nonempty subset of `{0.0}`.
    #[test]
    fn test_provable_raise_fires_for_a_set_divisor_that_is_always_zero() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        // a degenerate Set that carries nothing but the value zero — the
        // shape a narrowed range can collapse to, distinct from the
        // ordinary Kind::Values `single_numeric_value` already reads;
        // built directly here to pin `divisor_is_provably_always_zero`
        // itself rather than lean on a derived Sub row to produce it
        let always_zero = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
        };
        environment.bind("difference", always_zero);
        let parsed = parse_expression("1 / difference").expect("test source must parse");
        let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
        let found = binop_provable_raise(&binop, &environment, &kernel);
        let Some((_, message)) = found else {
            panic!("a divisor set that is always zero must provably raise");
        };
        assert!(message.contains("ZeroDivisionError"), "{message}");
        assert!(message.contains("division by zero"), "{message}");
    }

    /// The negative row: a divisor set that only SOMETIMES admits zero
    /// (`[0.0, 2.0]`) must NOT provably raise — most real executions
    /// never hit the zero corner, so an unconditional raise finding here
    /// would be a false positive. The VALUE question still declines
    /// (pinned above); this only confirms the RAISE question stays
    /// silent rather than overreaching. `binop_possible_raise` is this
    /// window's own row (pinned below) — a DIFFERENT function, a
    /// DIFFERENT claim, never this one's.
    #[test]
    fn test_provable_raise_stays_silent_for_a_set_divisor_that_only_sometimes_admits_zero() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        environment.bind("denominator", denominator);
        let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
        let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
        assert!(
            binop_provable_raise(&binop, &environment, &kernel).is_none(),
            "a sometimes-zero divisor window must not fire an unconditional raise"
        );
    }

    // --- `possible_raise` at a SET-SHAPED divisor ---

    /// The escape row: a divisor set that only SOMETIMES admits zero
    /// (`[0.0, 2.0]`) fires `binop_possible_raise`'s own sentence — a
    /// DIFFERENT claim from `binop_provable_raise`'s unconditional
    /// wording, and pinned against a DIFFERENT function: most real
    /// executions never hit the zero corner, so an unconditional raise
    /// finding would be a false positive, but the corner itself is a
    /// real escape `split_divisor_transfer`'s own value determination
    /// cannot speak to. Unguarded `1.0 / d` over this exact window still
    /// DERIVES the split value: confirmed directly against
    /// `binary_arithmetic_value_with_kernel` in the same test, so the
    /// fire and the determination are pinned together rather than in
    /// isolation — both stand; this row never withdraws the value, and
    /// which sink decides how to combine the two is `check.rs`'s own
    /// wiring, not this function's.
    #[test]
    fn test_possible_raise_fires_the_escape_sentence_for_a_set_divisor_that_only_sometimes_admits_zero() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        environment.bind("denominator", denominator.clone());
        let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
        let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
        let found = binop_possible_raise(&binop, &environment, &kernel);
        let Some((_, message)) = found else {
            panic!("a sometimes-zero divisor window must fire the escape sentence, not stay silent");
        };
        assert!(message.contains("admits 0"), "{message}");
        assert!(message.contains("ZeroDivisionError"), "{message}");
        assert!(
            !message.contains("this expression provably raises"),
            "the sometimes-zero row must not speak the always-zero rows' unconditional wording: {message}"
        );

        // the value side is not withdrawn: the same window still
        // determines through `split_divisor_transfer`, unaffected by
        // the new fire above
        let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
        let value = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            value.kind,
            Kind::Set,
            "the split value must still determine (never decline) alongside the fire: {value:?}"
        );
    }

    /// The always-zero row must not ALSO fire `possible_raise` — the two
    /// functions' claims are disjoint, keyed by `divisor_is_provably_
    /// always_zero` on one side and its negation on the other, so an
    /// always-zero window belongs to `binop_provable_raise` alone.
    #[test]
    fn test_possible_raise_stays_silent_for_a_divisor_that_is_always_zero() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let always_zero = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
        };
        environment.bind("difference", always_zero);
        let parsed = parse_expression("1 / difference").expect("test source must parse");
        let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
        assert!(
            binop_possible_raise(&binop, &environment, &kernel).is_none(),
            "an always-zero divisor is binop_provable_raise's own claim, not this row's"
        );
    }

    /// The narrowing-interaction row: a divisor already narrowed AWAY
    /// from zero (the shape `if divisor != 0:` leaves bound in
    /// `environment` once the walk consumes that guard) must NOT fire —
    /// `binop_possible_raise` reads `right` fresh off `environment` at
    /// the ask (`evaluate_expression(&binop.right, environment, kernel)`
    /// above), so a zero-excluding narrowed set is exactly what
    /// `divisor_provably_excludes_zero` already reports `true` for, the
    /// same gate the VALUE side reads in `transfer_over_sets` — the two
    /// never disagree about which windows still admit zero.
    #[test]
    fn test_possible_raise_stays_silent_for_a_divisor_narrowed_away_from_zero() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        // the narrowed shape a consumed `if divisor != 0:` (or an
        // equivalent guard) leaves behind: the zero-excluding POSITIVE
        // half of the same window the fire test above admits zero over
        let narrowed = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![
                    refined_sets::refinement_forms::above(0.0),
                    refined_sets::refinement_forms::at_most(2.0),
                ]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        environment.bind("denominator", narrowed);
        let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
        let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
        assert!(
            binop_possible_raise(&binop, &environment, &kernel).is_none(),
            "a divisor narrowed away from zero must not fire — the ask reads the narrowed set, not the pre-guard one"
        );
    }

    /// `//` and `%` fire the SAME escape sentence `/` does over a
    /// sometimes-zero divisor: CPython raises `ZeroDivisionError` on the
    /// zero arm of the window for all three operators alike
    /// (expressions.rst, "Binary arithmetic operations"). `//`/`%` have
    /// no zero-excluded split (`split_divisor_transfer` is `/`'s own
    /// fix), so their VALUE question keeps declining outright over this
    /// same window — only the fire is new here, the value side
    /// unchanged.
    #[test]
    fn test_possible_raise_fires_for_floordiv_and_mod_over_a_sometimes_zero_divisor() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        for source in ["1.0 // denominator", "1.0 % denominator"] {
            environment.bind("denominator", denominator.clone());
            let parsed = parse_expression(source).expect("test source must parse");
            let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
            let found = binop_possible_raise(&binop, &environment, &kernel);
            let Some((_, message)) = found else {
                panic!("{source}: a sometimes-zero divisor window must fire the escape sentence, not stay silent");
            };
            assert!(message.contains("admits 0"), "{source}: {message}");
            assert!(message.contains("ZeroDivisionError"), "{source}: {message}");

            // the value side is unchanged: `//`/`%` still decline outright
            // over this same window, because no split runs for them
            let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
            let op = if source.contains("//") { Operator::FloorDiv } else { Operator::Mod };
            let value = binary_arithmetic_value_with_kernel(op, &one, &denominator, &kernel);
            assert_eq!(
                value.kind,
                Kind::Unknown,
                "{source}: the value question must keep declining outright — no split runs for `//`/`%`: {value:?}"
            );
        }
    }

    // --- `possible_raise` for the domain-limited math family (straddling) ---

    /// `math.log(x)` where `x`'s window is `[-1.0, 1.0]` — STRADDLES the
    /// raise domain (`x <= 0`): the negative-through-zero half raises,
    /// the positive half `(0.0, 1.0]` still returns a value. Fires the
    /// SAME "math domain error" sentence `call_provable_raise`'s
    /// all-or-nothing row speaks, but through `possible_raise` — the
    /// window is not ENTIRELY inside the raise domain, so `call_
    /// provable_raise`'s own row (checked directly below) must stay
    /// silent, exactly the disjointness `test_possible_raise_stays_
    /// silent_for_a_divisor_that_is_always_zero` pins for the division
    /// row. The served half's value stands alongside the fire, read
    /// through `evaluate_attribute_call`'s own wiring (the value side of
    /// `math_call_result`'s decline, not `possible_raise` itself).
    #[test]
    fn test_possible_raise_fires_for_a_log_window_that_straddles_the_raise_domain() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let straddling = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(-1.0), refined_sets::refinement_forms::at_most(1.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        environment.bind("x", straddling);
        let parsed = parse_expression("math.log(x)").expect("test source must parse");
        let expr = parsed.into_expr();

        let found = possible_raise(&expr, &environment, &kernel);
        let Some((_, message)) = found else {
            panic!("a straddling log window must fire the possible-raise sentence, not stay silent");
        };
        assert!(message.contains("ValueError"), "{message}");
        assert!(message.contains("math domain error"), "{message}");

        // the ALL-OR-NOTHING row must stay silent for the same window —
        // the two claims are disjoint, keyed by
        // DomainRaiseClassification::EntirelyRaises vs ::Straddles
        assert!(
            provable_raise(&expr, &environment, &kernel).is_none(),
            "a straddling window is possible_raise's own claim, not provable_raise's"
        );

        // the served half still determines a value, read through the
        // ordinary evaluate_expression path (evaluate_attribute_call's
        // own decline-then-served-half wiring, math_models.rs)
        let value = evaluate_expression(&expr, &environment, &kernel);
        assert_eq!(
            value.kind,
            Kind::Set,
            "the served half (0.0, 1.0] must still determine a window, alongside the fire: {value:?}"
        );
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
    }

    /// An ENTIRELY-served log window (`[1.0, 2.0]`, wholly `x > 0`) must
    /// NOT fire `possible_raise` — the disjointness twin of the fire
    /// test above, mirroring `test_possible_raise_stays_silent_for_a_
    /// divisor_narrowed_away_from_zero`'s own shape for division.
    #[test]
    fn test_possible_raise_stays_silent_for_a_log_window_entirely_served() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let served = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(1.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        environment.bind("x", served);
        let parsed = parse_expression("math.log(x)").expect("test source must parse");
        let expr = parsed.into_expr();
        assert!(
            possible_raise(&expr, &environment, &kernel).is_none(),
            "an entirely-served window must not fire the straddling row"
        );
    }

    /// An ENTIRELY-raising log window (`[-2.0, -1.0]`, wholly `x <= 0`)
    /// must NOT fire `possible_raise` either — that claim belongs to
    /// `provable_raise`'s own all-or-nothing row alone, the same
    /// disjointness `test_possible_raise_stays_silent_for_a_divisor_
    /// that_is_always_zero` pins for the always-zero divisor.
    #[test]
    fn test_possible_raise_stays_silent_for_a_log_window_entirely_raising() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut environment = empty_environment();
        let raising = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![at_least(-2.0), refined_sets::refinement_forms::at_most(-1.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        };
        environment.bind("x", raising);
        let parsed = parse_expression("math.log(x)").expect("test source must parse");
        let expr = parsed.into_expr();
        assert!(
            possible_raise(&expr, &environment, &kernel).is_none(),
            "an entirely-raising window is provable_raise's own claim, not this row's"
        );
        assert!(
            provable_raise(&expr, &environment, &kernel).is_some(),
            "an entirely-raising window must fire provable_raise's own all-or-nothing row"
        );
    }

    // --- string_set_concatenation / string_shaped_set ---

    /// A length-windowed string parameter (`seed`, `Repeat(codepoints,
    /// 1, 8)` — the shape `check.rs::seed_parameters` seeds for
    /// `Annotated[str, Field(min_length=1, max_length=8)]`) concatenated
    /// with a literal: `Add` must compose a `Concatenation` set rather
    /// than falling through to `unknown()`, since neither operand is an
    /// exact string (the literal side is exact; the parameter side is
    /// not, which is what used to make `exact_string_values` refuse the
    /// whole row).
    #[test]
    fn test_add_concatenates_a_string_window_with_a_literal() {
        let seed = AbstractValue {
            kind_tag: None,
            ..known_set(
                make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, Some(8))]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let literal = string_models::string_literal_value("xxxxxxxx");
        let result = sequence_binop_value(Operator::Add, &seed, &literal);
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.set_kind_tag, SetKindTag::None);
        assert!(
            assignability::states_sequence(&result.set),
            "the concatenation must itself carry a sequence form: {:?}",
            result.set
        );
    }

    /// Two known EXACT strings still take the exact-value row above
    /// `string_set_concatenation`'s own fallback (`sequence_binop_value`'s
    /// first check) — this pins that the new fallback never fires for
    /// the case the exact row already answers, so the two rows do not
    /// double-handle the same input.
    #[test]
    fn test_add_two_exact_strings_stays_exact() {
        let a = string_models::string_literal_value("ab");
        let b = string_models::string_literal_value("c");
        let result = sequence_binop_value(Operator::Add, &a, &b);
        assert_eq!(result.kind, Kind::Values);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::String));
    }

    /// A NUMERIC set (never string-shaped) plus a string literal must
    /// stay `unknown()` — `string_shaped_set` refuses the numeric side,
    /// so the concatenation row never fires for a cross-sort operand
    /// pair.
    #[test]
    fn test_add_numeric_set_and_string_literal_stays_unknown() {
        let number_set = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let literal = string_models::string_literal_value("x");
        let result = sequence_binop_value(Operator::Add, &number_set, &literal);
        assert_eq!(result.kind, Kind::Unknown);
    }

    // --- kernel.seq_prefix / evaluate_slice's [:n] arm ---

    /// The kernel ask itself: `seq_prefix` over an UNBOUNDED repetition
    /// window (`Repeat(codepoints, 1, None)` — the shape
    /// set_functions/subset_seq_shape.lean's `seqOf` recognizes directly
    /// via its `.Repeat A lo none` arm) answers a set that itself states
    /// a sequence shape, per `prefixReadOf`'s own over-approximation
    /// (boundary/exports_sets.lean's `kernelSeqPrefix`).
    #[test]
    fn test_kernel_seq_prefix_answers_a_sequence_shaped_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let unbounded_window = make_refined_set(vec![repeat_of(
            refined_sets::codepoint_sets::codepoints(),
            1,
            None,
        )]);
        let Some(answered) = (kernel.seq_prefix)(&unbounded_window, 3) else {
            panic!("seqOf-recognized receiver must not decline");
        };
        assert!(
            assignability::states_sequence(&answered),
            "seq_prefix's answer must itself carry a sequence form: {answered:?}"
        );
    }

    /// The SAME receiver shape `evaluate_slice`'s regression test
    /// exercises end to end, pinned here at the bare ask level: a
    /// `Concatenation` whose leading operand is a `Repeat` window (the
    /// shape `text_label.py`'s own `seed + "xxxxxxxx"` builds).
    ///
    /// Pre-extension, the kernel's `seqOf` recognized a `Concatenation
    /// A B` only when `A.scalarB` — a single fixed scalar, never a
    /// `Repeat`/`Star` window — so this shape declined regardless of
    /// the window's own bound; that was this test's original premise
    /// (`test_kernel_seq_prefix_declines_a_concatenation_with_a_leading_
    /// window`, now renamed). The kernel extension
    /// (`seqWindowOf`/`prefix_read.lean`) now reads a `Concatenation`
    /// with a leading `Repeat` window in either operand order, so this
    /// now ANSWERS the proved window instead of declining.
    #[test]
    fn test_kernel_seq_prefix_admits_a_concatenation_with_a_leading_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let seed_window = make_refined_set(vec![repeat_of(
            refined_sets::codepoint_sets::codepoints(),
            1,
            Some(8),
        )]);
        let literal = refined_sets::codepoint_sets::string_tuple("xxxxxxxx");
        let joined = make_refined_set(vec![refined_sets::refinement_forms::concatenation(
            seed_window,
            literal,
        )]);
        let Some(answered) = (kernel.seq_prefix)(&joined, 3) else {
            panic!("a leading-window concatenation must now be seqOf-recognized, not decline");
        };
        assert!(
            assignability::states_sequence(&answered),
            "seq_prefix's answer must itself carry a sequence form: {answered:?}"
        );
    }

    /// `evaluate_slice`'s `[:n]` admit case: a receiver `Kind::Set` whose
    /// own form is the UNBOUNDED repetition window `seqOf` recognizes,
    /// sliced `[:3]`, asks `seq_prefix` and binds the answered set —
    /// never `unknown()`.
    #[test]
    fn test_slice_prefix_admits_over_a_seq_of_recognized_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let receiver = AbstractValue {
            kind_tag: None,
            ..known_set(
                make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, None)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let mut environment = empty_environment();
        environment.bind("padded", receiver);
        let parsed = parse_expression("padded[:3]").expect("test source must parse");
        let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
        let result = evaluate_subscript(&subscript, &environment, &kernel);
        assert_eq!(result.kind, Kind::Set, "expected a bound prefix set, got {result:?}");
        assert!(
            assignability::states_sequence(&result.set),
            "the bound prefix must itself carry a sequence form: {:?}",
            result.set
        );
    }

    /// A `step` slice over the same set-shaped receiver keeps declining —
    /// `evaluate_slice`'s own `slice.step.is_some()` gate fires before
    /// `sequence_prefix_slice` ever runs, per the mission's own
    /// unmodeled-step scope.
    #[test]
    fn test_slice_prefix_declines_a_step_slice() {
        let Some(kernel) = loaded_kernel() else { return };
        let receiver = AbstractValue {
            kind_tag: None,
            ..known_set(
                make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, None)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let mut environment = empty_environment();
        environment.bind("padded", receiver);
        let parsed = parse_expression("padded[:3:2]").expect("test source must parse");
        let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
        let result = evaluate_subscript(&subscript, &environment, &kernel);
        assert_eq!(result.kind, Kind::Unknown, "a step slice must still decline: {result:?}");
    }

    /// A NEGATIVE `upper` bound over the same set-shaped receiver
    /// declines: `sequence_prefix_slice` refuses `n < 0` rather than
    /// asking the kernel a nonsensical prefix length, and the length-based
    /// fallback below it has no known length for a `Kind::Set` receiver
    /// either, so the whole slice stays `unknown()`.
    #[test]
    fn test_slice_prefix_declines_a_negative_upper_bound() {
        let Some(kernel) = loaded_kernel() else { return };
        let receiver = AbstractValue {
            kind_tag: None,
            ..known_set(
                make_refined_set(vec![repeat_of(refined_sets::codepoint_sets::codepoints(), 1, None)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let mut environment = empty_environment();
        environment.bind("padded", receiver);
        let parsed = parse_expression("padded[:-1]").expect("test source must parse");
        let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
        let result = evaluate_subscript(&subscript, &environment, &kernel);
        assert_eq!(result.kind, Kind::Unknown, "a negative upper bound must decline: {result:?}");
    }

    /// The KERNEL's own decline — not a shape `sequence_prefix_slice`'s
    /// own gate rejects up front, but a receiver that reaches
    /// `kernel.seq_prefix` and gets `None` back from IT — completes
    /// without panicking and keeps the length-based fallback exactly as
    /// if the `[:n]` arm had never matched.
    ///
    /// This test's original premise (before this rewrite) built the
    /// declining operand as a `Union` of two SCALAR string tuples
    /// (`string_tuple("a")`, `string_tuple("b")`), citing
    /// `prefix_read.lean`'s doc that "a Union operand is not read." That
    /// doc names `seqWindowOf`'s own top-level match on `R = Union A B`
    /// — it does not cover a Union appearing as a `Concatenation`
    /// operand. `Refinement.Union A B => A.scalarB && B.scalarB`
    /// (`emptiness.lean`) makes a Union of two scalar sets itself
    /// scalar, so `seqWindowOf`'s `if R.scalarB then some (R, 1, some 1)`
    /// fast path recognized that operand directly — the kernel measured
    /// `Some(...)`, not `None`, so the original premise was stale, not a
    /// regression (`packages/refinedpy/rust/refined_sets/src/../.. /
    /// set_functions/emptiness.lean:40`, `prefix_read.lean:238-252`).
    ///
    /// The genuinely-declining shape is a Union of two NON-scalar
    /// window operands (two `Repeat`s over different alphabets):
    /// `Refinement.Union`'s scalar check fails (neither side is
    /// scalar), so the bare `Union` never matches `seqWindowOf`'s
    /// `if R.scalarB` fast path, and `seqWindowOf`'s own `match R with`
    /// has no arm for a bare `Union` at all — it falls to the wildcard
    /// `_ => none`. Nested as the `Concatenation`'s left operand, the
    /// recursive `seqWindowOf A` call on that Union gets `none` back,
    /// so the whole ask still declines.
    #[test]
    fn test_slice_prefix_completes_without_panic_when_the_kernel_itself_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let window_a = make_refined_set(vec![repeat_of(one_char_of("ab"), 1, Some(4))]);
        let window_b = make_refined_set(vec![repeat_of(one_char_of("cd"), 1, Some(4))]);
        let unrecognized_union_operand =
            make_refined_set(vec![refined_sets::refinement_forms::union(window_a, window_b)]);
        let literal = refined_sets::codepoint_sets::string_tuple("xxxxxxxx");
        let concatenation_with_a_union_operand = make_refined_set(vec![
            refined_sets::refinement_forms::concatenation(unrecognized_union_operand, literal),
        ]);
        // pin the ask-level premise directly: seqWindowOf must still
        // decline this shape, or the rest of the test would be testing
        // nothing
        assert_eq!(
            (kernel.seq_prefix)(&concatenation_with_a_union_operand, 3),
            None,
            "a Concatenation over a Union of non-scalar window operands must still decline (seqWindowOf's own named edge)"
        );
        let receiver = AbstractValue {
            kind_tag: None,
            ..known_set(concatenation_with_a_union_operand, None, TrustProved, SetKindTag::None)
        };
        let mut environment = empty_environment();
        environment.bind("padded", receiver);
        let parsed = parse_expression("padded[:3]").expect("test source must parse");
        let Expr::Subscript(subscript) = parsed.into_expr() else { panic!("expected a Subscript") };
        // the assertion itself is the regression: a prior version of this
        // arm panicked reaching this call ("kernel: the set is not a
        // recognized sequence shape") instead of returning a value
        let result = evaluate_subscript(&subscript, &environment, &kernel);
        assert_eq!(
            result.kind,
            Kind::Unknown,
            "a kernel-declined prefix must fall through to unknown(), not panic: {result:?}"
        );
    }

    // --- numeric_value_vs_window_compare ---

    /// `len(padded) >= 3` where `padded` is a `[:3]` prefix window — the
    /// exact construct `text_label.py`'s `return padded if len(padded)
    /// >= 3 else "xxx"` compares. `len()` over a `Repeat(alphabet, 3, 3)`
    /// window (`collection_models::len_result`'s own reading of a
    /// DEGENERATE bound) answers a bounded Integer `Kind::Set`, `{AtLeast
    /// 3, AtMost 3}` — never a single known value — so this decides only
    /// through `numeric_value_vs_window_compare`'s own window arm, not
    /// `compare_pair`'s exact-numeric row. Every admitted length (all of
    /// them, since the window is degenerate) satisfies `>= 3`, so the
    /// comparison decides `True`.
    #[test]
    fn test_compare_decides_over_a_degenerate_length_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let seed_window = make_refined_set(vec![repeat_of(
            refined_sets::codepoint_sets::codepoints(),
            1,
            Some(8),
        )]);
        let literal = refined_sets::codepoint_sets::string_tuple("xxxxxxxx");
        let concatenation_with_a_leading_window = make_refined_set(vec![
            refined_sets::refinement_forms::concatenation(seed_window, literal),
        ]);
        let receiver = AbstractValue {
            kind_tag: None,
            ..known_set(concatenation_with_a_leading_window, None, TrustProved, SetKindTag::None)
        };
        let mut environment = empty_environment();
        environment.bind("padded", receiver);
        let sliced_parsed = parse_expression("padded[:3]").expect("test source must parse");
        let Expr::Subscript(subscript) = sliced_parsed.into_expr() else { panic!("expected a Subscript") };
        let sliced = evaluate_subscript(&subscript, &environment, &kernel);
        assert_eq!(sliced.kind, Kind::Set, "the [:3] slice must admit now that the kernel recognizes the shape");
        environment.bind("sliced", sliced);

        let compare_parsed = parse_expression("len(sliced) >= 3").expect("test source must parse");
        let compare_value = evaluate_expression(&compare_parsed.into_expr(), &environment, &kernel);
        assert_eq!(compare_value.kind, Kind::Values, "the comparison must decide, not stay unknown: {compare_value:?}");
        assert_eq!(
            compare_value.values,
            vec![1.0],
            "len(a 3-length window) >= 3 must decide True: {compare_value:?}"
        );
    }

    /// A window that only SOMETIMES satisfies the comparison (`[0, 5]`
    /// against `>= 3`) must stay undecided — some admitted lengths (0,
    /// 1, 2) fail the bound while others (3, 4, 5) pass it, and this
    /// function never guesses across a partial overlap.
    #[test]
    fn test_compare_stays_undecided_over_a_window_straddling_the_bound() {
        let straddling_window = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let three = known_values(vec![3.0], PrimitiveKind::Integer, TrustProved);
        assert_eq!(
            compare_pair(CmpOp::GtE, &straddling_window, &three),
            None,
            "a window straddling the bound must not decide >="
        );
        assert_eq!(
            compare_pair(CmpOp::GtE, &three, &straddling_window),
            None,
            "the swapped operand order must not decide either"
        );
    }

    /// A window entirely BELOW the target decides `<`/`<=` true and
    /// `>`/`>=` false — the mirror of the degenerate-window admit case,
    /// pinning the non-degenerate ordering rows and the swapped operand
    /// order together.
    #[test]
    fn test_compare_decides_over_a_window_entirely_below_the_target() {
        let low_window = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let three = known_values(vec![3.0], PrimitiveKind::Integer, TrustProved);
        assert_eq!(compare_pair(CmpOp::Lt, &low_window, &three), Some(1.0), "[0,2] < 3 must decide True");
        assert_eq!(compare_pair(CmpOp::GtE, &low_window, &three), Some(0.0), "[0,2] >= 3 must decide False");
        // swapped: `3 > window` is the same claim as `window < 3`
        assert_eq!(compare_pair(CmpOp::Gt, &three, &low_window), Some(1.0), "3 > [0,2] must decide True");
    }

    // --- zero_padded_decimal_spelling / zero_padded_decimal_width ---

    /// `year: [1970, 9999]` formatted `:04d` — every member already
    /// spells exactly 4 decimal digits, so the zero-fill is a no-op and
    /// the exact digit-window `Repeat(digits, 4, 4)` is sound.
    #[test]
    fn test_zero_padded_decimal_spelling_exact_when_padding_is_a_no_op() {
        let year = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(1970.0), refined_sets::refinement_forms::at_most(9999.0)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let Some(kernel) = loaded_kernel() else { return };
        let source = "f\"{year:04d}\"";
        let parsed = parse_expression(source).expect("test source must parse");
        let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
        let mut environment = empty_environment();
        environment.bind("year", year);
        let result = evaluate_fstring(&fstring, &environment, &kernel);
        assert_eq!(result.kind, Kind::Set);
        assert!(
            assignability::states_sequence(&result.set),
            "a zero-padded bounded-range interpolation must answer a sequence-shaped set: {:?}",
            result.set
        );
    }

    /// A range that WOULD need real padding for some members but not
    /// others (`8..12` against `02d`: "08".."12") declines rather than
    /// approximate — `decimal_digit_count(8) == 1` while
    /// `decimal_digit_count(12) == 2`, so the two ends disagree with the
    /// stated width and the whole interpolation must answer `unknown()`.
    #[test]
    fn test_zero_padded_decimal_spelling_declines_when_padding_would_actually_fire() {
        let count = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(8.0), refined_sets::refinement_forms::at_most(12.0)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let Some(kernel) = loaded_kernel() else { return };
        let source = "f\"{count:02d}\"";
        let parsed = parse_expression(source).expect("test source must parse");
        let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
        let mut environment = empty_environment();
        environment.bind("count", count);
        let result = evaluate_fstring(&fstring, &environment, &kernel);
        assert_eq!(result.kind, Kind::Unknown);
    }

    /// A format spec that is not the recognized `0{width}d` spelling
    /// (here, `.2f`) declines the whole f-string, same as before this
    /// wave's own format-spec gate ever recognized any spec at all.
    #[test]
    fn test_unrecognized_format_spec_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let value = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustSpec, SetKindTag::None)
        };
        let source = "f\"{value:.2f}\"";
        let parsed = parse_expression(source).expect("test source must parse");
        let Expr::FString(fstring) = parsed.into_expr() else { panic!("expected an FString") };
        let mut environment = empty_environment();
        environment.bind("value", value);
        let result = evaluate_fstring(&fstring, &environment, &kernel);
        assert_eq!(result.kind, Kind::Unknown);
    }
}
