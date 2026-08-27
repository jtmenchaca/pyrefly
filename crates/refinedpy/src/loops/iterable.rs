/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::Number;
use ruff_python_ast::Stmt;
use ruff_python_ast::UnaryOp;
use ruff_python_ast::visitor::walk_expr;
use ruff_python_ast::visitor::Visitor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances;

/// A single known, Integer- or Float-sorted for-loop iterate — CPython's
/// own two numeric sorts, never the joined/unknown `PrimitiveKind::Number`
/// (PYREFLY-NUMERIC-B3-B4.md's "two sorts, never one Number"). Binding an
/// iterate this way (rather than the old sort-erasing `known_number`)
/// is what lets a `for age in [10, 20, 30]: total = total + age` row's
/// arithmetic see BOTH operands as Integer and answer an Integer total —
/// `binary_arithmetic_value`'s `single_numeric_value` reads a bare
/// `Number` tag conservatively as Float, which is what previously made
/// an all-int accumulation read as a float and wrongly fire the
/// int-sort law on its own in-set result.
pub(super) fn known_number_sorted(value: f64, sort: PrimitiveKind) -> AbstractValue {
    known_values(vec![value], sort, TrustProved)
}

/// A Python `str`, as this domain's exact-string `AbstractValue` — one
/// code point per `f64` (`string_models.rs`'s documented representation;
/// repeated here rather than reaching into that module's private
/// helper, matching `collection_models.rs`'s own same-crate-different-
/// module precedent for this exact conversion).
pub(super) fn known_string(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    known_values(code_points, PrimitiveKind::String, TrustProved)
}

/// The known elements a `for` loop's iterable expression names, in
/// iteration order, each already carrying its TRUE Python sort:
/// - a literal list/tuple of number literals (Integer or Float per
///   element) or a `range(...)` call (library/stdtypes.html#range,
///   always Integer — `range` accepts only int arguments).
/// - a dict DISPLAY iterated directly (`for k in {...}:`) — CPython
///   iterates a dict's KEYS (library/stdtypes.rst, "Mapping Types —
///   dict": "Iterating views while adding or deleting entries..."; the
///   dict's own `__iter__` "return an iterator over the keys"), so each
///   element is the key's exact String value.
/// - `<dict-valued-name-or-expr>.values()` / `.items()` / `.keys()` on
///   a receiver `evaluate_expression` reads as a known `Kind::Object`
///   (a prior local dict, not necessarily a literal at the call site):
///   `.values()` yields each entry's value, `.keys()` yields each
///   entry's key (String), `.items()` yields a 2-element tuple
///   (`Kind::List` of `[key, value]`) per entry — CPython's own view
///   order, library/stdtypes.rst dict views, "Keys views are set-like...
///   Dictionary views... iterate over `... items in insertion order`".
/// - a same-module (sync or async) generator `def`'s own call
///   (`generator_call_values`, `instances::generator_yields`) — a
///   bare-Name call whose def's body is straight-line `yield`
///   statements; each yielded value becomes one iterate, in yield
///   order.
///
/// Anything else (a name that is not a known dict, a call other than
/// `range`/`.values`/`.items`/`.keys`/a readable same-module generator,
/// a non-literal element whose EVALUATED value is not itself known) is
/// `None`: this function only answers when every iterate is known
/// without running any unmodeled code.
///
/// EVALUATED ELEMENTS: a `List`/`Tuple` display's own elements are read
/// SYNTACTICALLY first (`sorted_number_literal_value` — the exact
/// literal-number path, which also carries the element's true Integer/
/// Float sort); an element that is not a bare number literal falls back
/// to `evaluate_expression`. a-statements.py's `for_over_unread_iterable`:
/// `(unread_number(),)`'s single element is a CALL, and `unread_number`'s
/// own body (`raise NotImplementedError`) is a genuine decline in
/// `summaries::interpret_body` (no `Stmt::Raise` row there) — its call
/// answers `return_sort_fallback`'s `-> int` claim instead, the
/// whole-number SET (`Kind::Set`, Integer-tagged), never `Kind::Null`.
/// Accepted evaluated shapes: ANY known AbstractValue whose `kind` is not
/// `Kind::Unknown` — a known single scalar, `Kind::Null`, or a known SET
/// (Integer/Float/String-sorted, `Kind::Set`) all accepted alike, because
/// the DISPLAY's own element COUNT is syntactic (this is a fixed-arity
/// tuple/list literal, not an iterable whose length depends on a value),
/// so binding the `for` target to each element's own value — whatever
/// shape that value is — and running the body once per element is sound
/// regardless of what sort of value that element turns out to be. Only a
/// truly UNKNOWN element (`Kind::Unknown` — nothing at all is known about
/// it) declines the WHOLE display, same as every other honest refusal in
/// this file. This acceptance is scoped to a DISPLAY's own elements only:
/// a non-display iterable (a bare Name bound to a set-VALUED expression,
/// for instance) has no syntactic element count to fall back on and is
/// not read through this function at all.
pub(super) fn iterable_values(
    iterable: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    match iterable {
        Expr::List(list) => elements_as_values(&list.elts, environment, kernel),
        Expr::Tuple(tuple) => elements_as_values(&tuple.elts, environment, kernel),
        Expr::Call(call) => range_call_values(call)
            .or_else(|| dict_view_call_values(call, environment, kernel))
            .or_else(|| enumerate_call_values(call, environment, kernel))
            .or_else(|| zip_call_values(call, environment, kernel))
            .or_else(|| generator_call_values(call, environment, kernel))
            .or_else(|| finditer_call_values(call, environment, kernel)),
        Expr::Dict(_) => {
            let receiver = evaluate_expression(iterable, environment, kernel);
            dict_keys_as_strings(&receiver)
        }
        // Any other iterable expression (a bare Name most commonly)
        // whose EVALUATED value is a known List of fully-known items:
        // the element count is carried by the value itself, so
        // iterating its items is exactly as sound as a display's — the
        // same acceptance rule elements_as_values applies per element.
        // A known dict value iterates its keys, the same reading the
        // Dict-display arm gives. Anything else stays None.
        other => {
            let receiver = evaluate_expression(other, environment, kernel);
            if receiver.kind == Kind::List
                && receiver.items.iter().all(|item| item.kind != Kind::Unknown)
            {
                return Some(receiver.items.clone());
            }
            dict_keys_as_strings(&receiver)
        }
    }
}

pub(super) fn elements_as_values(
    elements: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::with_capacity(elements.len());
    for element in elements {
        if let Some(literal) = sorted_number_literal_value(element) {
            values.push(literal);
            continue;
        }
        let evaluated = evaluate_expression(element, environment, kernel);
        if evaluated.kind == Kind::Unknown {
            return None;
        }
        values.push(evaluated);
    }
    Some(values)
}

/// `enumerate(<iterable>[, start])` — "the `__next__` method of the
/// iterator returned by enumerate returns a tuple containing a count
/// (from *start* which defaults to 0) and the values obtained from
/// iterating over *iterable*" (library/functions.rst, `enumerate`).
/// Each element is a 2-element `Kind::List` `[count, value]`, the same
/// pair shape `.items()` builds, so `bind_for_target`'s existing
/// tuple-unpack path binds `for i, x in enumerate(xs):` unchanged.
/// `None` when the callee is not the builtin name `enumerate` (a local
/// binding shadows it), when the inner iterable is not one this file
/// already reads concretely, or when `start` is anything but a single
/// known Integer.
fn enumerate_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "enumerate" || environment.read("enumerate").is_some() {
        return None;
    }
    let [inner] = call.arguments.args.as_ref() else {
        return None;
    };
    let start = match call.arguments.find_keyword("start") {
        Some(keyword) => known_integer_argument(&keyword.value, environment, kernel)?,
        None => 0,
    };
    let elements = iterable_values(inner, environment, kernel)?;
    Some(
        elements
            .into_iter()
            .enumerate()
            .map(|(offset, value)| {
                known_list(
                    vec![known_number_sorted((start + offset as i64) as f64, PrimitiveKind::Integer), value],
                    TrustProved,
                )
            })
            .collect(),
    )
}

/// `zip(<iterable>, ...)` — "By default, zip stops when the shortest
/// iterable is exhausted" (library/functions.rst, `zip`). Each element
/// is a `Kind::List` holding one value drawn from each argument at the
/// same offset, and the element count is the MINIMUM of the arguments'
/// own lengths. `None` when the callee is not the builtin name `zip`
/// (a local binding shadows it), when any argument is not an iterable
/// this file already reads concretely, or when `strict=` is present —
/// that keyword raises on a length mismatch rather than truncating, a
/// different rule this row does not decide.
fn zip_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "zip" || environment.read("zip").is_some() {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    if call.arguments.args.is_empty() {
        return None;
    }
    let mut columns: Vec<Vec<AbstractValue>> = Vec::with_capacity(call.arguments.args.len());
    for argument in call.arguments.args.iter() {
        columns.push(iterable_values(argument, environment, kernel)?);
    }
    let length = columns.iter().map(|column| column.len()).min()?;
    Some(
        (0..length)
            .map(|offset| {
                known_list(columns.iter().map(|column| column[offset].clone()).collect(), TrustProved)
            })
            .collect(),
    )
}

/// One known Integer-sorted argument value, read for its exact number —
/// the shape `enumerate`'s own `start` keyword needs. `None` for an
/// unread value or any other sort.
fn known_integer_argument(
    expr: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<i64> {
    let value = evaluate_expression(expr, environment, kernel);
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    if value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    Some(value.values[0] as i64)
}

/// A dict's keys, each as an exact String `AbstractValue`, in the
/// dict's own insertion order — `None` for anything that is not a
/// known `Kind::Object` (an unread dict, a dict built by a non-literal
/// path this domain does not model, library/stdtypes.rst's dict
/// iteration order guarantee applying only to a known key set).
fn dict_keys_as_strings(receiver: &AbstractValue) -> Option<Vec<AbstractValue>> {
    if receiver.kind != Kind::Object {
        return None;
    }
    Some(receiver.keys.iter().map(|entry| known_string(&entry.name)).collect())
}

/// `<dict>.values()` / `<dict>.items()` / `<dict>.keys()` — the
/// receiver expression is evaluated against the CURRENT environment (it
/// may be a prior local variable, not a literal at the call site) and
/// must read as a known `Kind::Object`; every other receiver shape, or
/// a method name other than these three, is `None`. `.items()` builds
/// one 2-element tuple (`Kind::List`) per entry so
/// `bind_for_target`'s existing tuple-unpack path binds `for k, v in
/// d.items():` with no special-casing beyond that.
fn dict_view_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return None;
    }
    let receiver = evaluate_expression(attribute.value.as_ref(), environment, kernel);
    if receiver.kind != Kind::Object {
        return None;
    }
    match attribute.attr.as_str() {
        "values" => Some(receiver.keys.iter().map(|entry| entry.value.clone()).collect()),
        "keys" => dict_keys_as_strings(&receiver),
        "items" => Some(
            receiver
                .keys
                .iter()
                .map(|entry| known_list(vec![known_string(&entry.name), entry.value.clone()], TrustProved))
                .collect(),
        ),
        _ => None,
    }
}

/// `re.finditer(pattern, s)` — library/re.html, `finditer(pattern,
/// string)`: "Return an iterator yielding match objects." Modeled ONLY
/// for a known EXACT-STRING pattern argument (`string_models::
/// match_object_value`'s own gate, the same one `expressions.rs`'s own
/// value-position reading of this identical call already applies), and
/// ONLY as ONE representative match element — `generator_call_values`'s
/// own precedent above: a `for` loop's own body sees at most one bound
/// value per pass regardless of how many times the pattern actually
/// matches, so one representative element (the same unanchored match-
/// object value `expressions.rs`'s own `re.finditer(...).group(0)`
/// reading already answers for a non-loop use of this identical call) is
/// what `for m in re.finditer(pattern, s): ... m.group(0)` needs — the
/// loop body's own `m.group(0)` read then goes through the ALREADY-
/// LANDED `.group` dispatch on that value, unchanged by this function.
/// The receiver is recognized the same way `expressions.rs::
/// evaluate_attribute_call` recognizes every OTHER modeled module call:
/// the chain's root is the bare Name `re`, unshadowed by a local binding
/// (`environment.read("re").is_none()`). `None` for every other shape —
/// a non-`re` receiver, a shadowed `re`, an argument count other than
/// two, or a pattern that is not a known exact string — falling through
/// to `iterable_values`'s own existing decline for this call.
fn finditer_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if attribute.attr.as_str() != "finditer" {
        return None;
    }
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "re" || environment.read("re").is_some() {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [pattern, _subject] = &*call.arguments.args else {
        return None;
    };
    let pattern_value = evaluate_expression(pattern, environment, kernel);
    let pattern_text = code_points_to_string(exact_string_values(&pattern_value)?)?;
    let match_value = crate::string_models::match_object_value(&pattern_text)?;
    Some(vec![match_value])
}

/// The code-point vector an AbstractValue carries, if it is a known
/// exact string (`Kind::Values` tagged `PrimitiveKind::String`) —
/// `expressions.rs::exact_string_values`'s own twin, reimplemented
/// locally rather than imported (this file's own "no importing
/// loops.rs" precedent, `generator_yields`'s own doc, applied to
/// expressions.rs's private helper the same way).
fn exact_string_values(value: &AbstractValue) -> Option<&[f64]> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(&value.values)
}

/// The `Vec<f64>` code points `string_models.rs` builds, converted back
/// to a Rust `String` — `expressions.rs::code_points_to_string`'s own
/// twin, reimplemented locally for the identical reason.
fn code_points_to_string(code_points: &[f64]) -> Option<String> {
    code_points.iter().map(|point| char::from_u32(*point as i64 as u32)).collect()
}

/// The dict name a `for` loop iterates DIRECTLY over its own entries —
/// `for k in d:`/`for k in d.keys():`/`for v in d.values():`/`for k, v
/// in d.items():` — bound to a known `Kind::Object` in `environment`.
/// `Some(name)` only for a bare-Name receiver (a fresh dict literal or a
/// computed expression has no single WRITABLE name a body statement
/// could mutate through, so `dict_size_changing_mutation_range` has
/// nothing to match against); every other iterable shape (a list/tuple
/// display, `range(...)`, a generator call, a dict LITERAL display) is
/// `None` — this reader exists only to feed the iterator-invalidation
/// check below, never `iterable_values`'s own element-reading contract.
pub(super) fn iterated_dict_name<'a>(iterable: &'a Expr, environment: &Environment) -> Option<&'a str> {
    let receiver_expr = match iterable {
        Expr::Name(name) => name.id.as_str(),
        Expr::Call(call) => {
            let Expr::Attribute(attribute) = call.func.as_ref() else {
                return None;
            };
            if !matches!(attribute.attr.as_str(), "keys" | "values" | "items") {
                return None;
            }
            let Expr::Name(name) = attribute.value.as_ref() else {
                return None;
            };
            name.id.as_str()
        }
        _ => return None,
    };
    let receiver = environment.read(receiver_expr)?;
    if receiver.kind != Kind::Object {
        return None;
    }
    Some(receiver_expr)
}

/// Whether `expr` is one of the four dict methods that provably change a
/// dict's own SIZE — `.pop(...)`/`.popitem()`/`.clear()` — called on a
/// bare Name equal to `dict_name`, or a `del <dict_name>[...]` subscript
/// target reads the identical shape one level up in
/// `dict_size_changing_mutation_range`. `d[key] = value` and `.update(...)`
/// are deliberately EXCLUDED: an existing-key assignment never changes
/// size at all (library/stdtypes.rst never raises there), and `.update`'s
/// own size delta is not staticaly provable from its argument alone — this
/// function only ever names a mutation CPython's own dict-views note
/// states unconditionally changes size ("don't add or remove entries").
pub(super) fn is_dict_size_changing_method_call(expr: &Expr, dict_name: &str) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return false;
    };
    if receiver.id.as_str() != dict_name {
        return false;
    }
    matches!(attribute.attr.as_str(), "pop" | "popitem" | "clear")
}

/// Scans `body`'s own TOP-LEVEL statements (mirroring `run_statement_once`'s
/// own straight-line scope — a mutation nested inside an `if`/`for`/`try`
/// one level down is not proved to run on EVERY reachable pass, so it is
/// outside this function's provable claim) for a statement that provably
/// changes `dict_name`'s own size: `del dict_name[...]`,
/// `dict_name.pop(...)`, `dict_name.popitem()`, `dict_name.clear()`
/// (`is_dict_size_changing_method_call`'s own set, as an expression
/// statement). `Some(range)` names the FIRST such statement's own range —
/// the first-blocker-wins convention this file's own `already_fired`
/// dedupe and `check.rs`'s `record_blocker` both keep; `None` when no
/// top-level statement in this body provably changes the dict's size.
pub(super) fn dict_size_changing_mutation_range(body: &[Stmt], dict_name: &str) -> Option<TextRange> {
    for stmt in body {
        match stmt {
            Stmt::Delete(delete) => {
                for target in &delete.targets {
                    if let Expr::Subscript(subscript) = target {
                        if let Expr::Name(receiver) = subscript.value.as_ref() {
                            if receiver.id.as_str() == dict_name {
                                return Some(stmt.range());
                            }
                        }
                    }
                }
            }
            Stmt::Expr(expr_stmt) if is_dict_size_changing_method_call(expr_stmt.value.as_ref(), dict_name) => {
                return Some(stmt.range());
            }
            _ => {}
        }
    }
    None
}

/// The bare Name a `for` loop iterates DIRECTLY over — `for x in lst:`
/// — when `lst` is itself the loop's own iterable expression, no
/// `.keys()`/`.values()`/`.items()` view or other wrapping call
/// involved. `Some(name)` only for this exact bare-Name shape (a
/// computed expression, a literal display, or a view call has no
/// single WRITABLE name a body statement could mutate through, so
/// `list_size_changing_mutation_range` has nothing to match against);
/// mirrors `iterated_dict_name`'s own scoping, one level simpler since
/// a list carries no `.keys()`-style view methods to see through.
pub(super) fn iterated_list_name(iterable: &Expr) -> Option<&str> {
    let Expr::Name(name) = iterable else {
        return None;
    };
    Some(name.id.as_str())
}

/// Whether `expr` is `<list_name>.append(...)` — the one list mutation
/// that unconditionally GROWS the receiver on every call (stdtypes.rst,
/// "list.append(x): Add an item to the end of the list. Equivalent to
/// a[len(a):] = [x]"), called on a bare Name equal to `list_name`.
/// `insert`/`extend`/`+=` also grow a list, but are not read here: this
/// function's own caller (`list_size_changing_mutation_range`) only
/// needs the ONE shape the corpus states as non-terminating —
/// `for x in lst: lst.append(x)`, a self-feeding append that runs the
/// iterator's own internal index into elements the SAME pass just
/// added (`tmp/cpython/Doc/library/stdtypes.rst`'s list iterator has
/// no length snapshot the way a `range(len(...))` counter would) —
/// extending the recognized method set to the wider non-terminating
/// family is a follow-on, not a behavior this one row needs.
fn is_list_growing_append_call(expr: &Expr, list_name: &str) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    if attribute.attr.as_str() != "append" {
        return false;
    }
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return false;
    };
    receiver.id.as_str() == list_name
}

/// Scans `body`'s own TOP-LEVEL statements (the same straight-line
/// scope `dict_size_changing_mutation_range` reads — a nested `.append`
/// one level inside an `if`/`for`/`try` is not proved to run on EVERY
/// reachable pass) for a statement that provably grows `list_name` on
/// every pass: `list_name.append(...)` as an expression statement
/// (`is_list_growing_append_call`). `Some(range)` names the FIRST such
/// statement's own range; `None` when no top-level statement in this
/// body provably appends to the iterated list.
pub(super) fn list_size_changing_mutation_range(body: &[Stmt], list_name: &str) -> Option<TextRange> {
    for stmt in body {
        if let Stmt::Expr(expr_stmt) = stmt {
            if is_list_growing_append_call(expr_stmt.value.as_ref(), list_name) {
                return Some(stmt.range());
            }
        }
    }
    None
}

/// Whether ANY statement in `body`, at any nesting depth, calls a
/// list method on `list_name` that changes its length —
/// `append`/`insert`/`extend`/`pop`/`remove`/`clear`. Unlike
/// `list_size_changing_mutation_range`, which proves a growth on EVERY
/// reachable pass at the top level, this asks the weaker question a
/// CONCRETE element walk needs answered: could the sequence the loop is
/// stepping differ from the snapshot taken before the first pass? A
/// list's iterator holds the live list and re-reads its length on each
/// `__next__` (stdtypes.rst, "Iterator Types" — the iterator keeps a
/// reference and an index, not a copy), so an append reached on any
/// pass, however deeply guarded, adds an element the loop still visits,
/// and a removal drops one it would have. Since this reader cannot say
/// WHICH branches run, the honest answer to a body that can mutate the
/// iterated list is to decline the concrete walk entirely rather than
/// step the stale snapshot and state an exact count that CPython does
/// not produce.
pub(super) fn body_can_resize_iterated_list(body: &[Stmt], list_name: &str) -> bool {
    struct ResizeScan<'a> {
        list_name: &'a str,
        found: bool,
    }
    impl<'a> Visitor<'a> for ResizeScan<'a> {
        fn visit_expr(&mut self, expr: &'a Expr) {
            if let Expr::Call(call) = expr {
                if let Expr::Attribute(attribute) = call.func.as_ref() {
                    if matches!(
                        attribute.attr.as_str(),
                        "append" | "insert" | "extend" | "pop" | "remove" | "clear"
                    ) {
                        if let Expr::Name(receiver) = attribute.value.as_ref() {
                            if receiver.id.as_str() == self.list_name {
                                self.found = true;
                            }
                        }
                    }
                }
            }
            walk_expr(self, expr);
        }
    }
    let mut scan = ResizeScan { list_name, found: false };
    for stmt in body {
        scan.visit_stmt(stmt);
    }
    scan.found
}

/// `some_generator(args...)` — a bare-Name call to a SAME-MODULE `def`
/// (sync or async: `async def stream(): ...` still parses as
/// `StmtFunctionDef`, ruff carries `is_async` as a flag on the def, not
/// a distinct node type) whose body `instances::generator_yields` can
/// read straight-line — `for value in gen(): ...`/`async for value in
/// gen(): ...` both iterate the SAME element sequence a plain call's
/// yields name: compound_stmts.rst, "The `async for` statement" desugars
/// to `TARGET = await type(iter).__anext__(iter)` each pass, and
/// `await` only ever suspends/resumes scheduling — it does not change
/// which values `__anext__` (itself backed by the same generator body's
/// `yield` statements, datamodel.rst's generator-iterator protocol)
/// hands back. `is_async` on `def` is therefore not read here at all:
/// an async generator's yielded elements are the same values a sync
/// generator's would be, only reached through a different awaited
/// protocol. `None` for a non-Name callee, a name with no same-module
/// `def`, any keyword/starred argument (this file does not guess
/// keyword-to-position mapping the way `expressions.rs`'s own
/// `positional_arguments_for_def` does — that helper is private to its
/// module), or a def `generator_yields` itself declines (no top-level
/// `yield`, a conditional yield, a `yield` reached only through a loop
/// or other nested control flow, `yield from` — see that function's own
/// doc for its exact straight-line-body contract).
pub(super) fn generator_call_values(
    call: &ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    if call.arguments.args.iter().any(|argument| matches!(argument, Expr::Starred(_))) {
        return None;
    }
    let table = environment.functions()?;
    let def = table.def(callee.id.as_str())?;
    let mut arguments = Vec::with_capacity(call.arguments.args.len());
    for argument in &call.arguments.args {
        arguments.push(evaluate_expression(argument, environment, kernel));
    }
    let yields = instances::generator_yields(def, &arguments, Some(table), kernel, environment.call_depth())?;
    let mut values = Vec::with_capacity(yields.len());
    for yielded in yields {
        // NOT the same widened acceptance `elements_as_values` now takes
        // for a DISPLAY's own elements: a generator's own yield COUNT is
        // not syntactic the way a tuple/list literal's element count is
        // (`generator_yields` itself already declines any body shape
        // wider than its own two recognized forms before this point is
        // ever reached), so this guard stays at the narrower "a known
        // single scalar or Kind::Null" acceptance — anything wider
        // declines the WHOLE generator's contribution rather than
        // silently narrow it.
        if yielded.kind == Kind::Null || (yielded.kind == Kind::Values && yielded.values.len() == 1) {
            values.push(yielded);
            continue;
        }
        return None;
    }
    Some(values)
}

/// A `range(...)` call's produced values, or `None` when the callee
/// is not the bare name `range`, an argument is not a literal int, or
/// the argument count is not 1/2/3. `step == 0` is `None` — CPython
/// raises `ValueError` there rather than producing a sequence. Every
/// produced value is Integer-sorted — `range` accepts only int
/// arguments (library/stdtypes.html#range), so its elements are never
/// float.
fn range_call_values(call: &ExprCall) -> Option<Vec<AbstractValue>> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "range" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let args = &call.arguments.args;
    let (start, stop, step) = match args.len() {
        1 => (0.0, int_literal_value(&args[0])?, 1.0),
        2 => (int_literal_value(&args[0])?, int_literal_value(&args[1])?, 1.0),
        3 => (
            int_literal_value(&args[0])?,
            int_literal_value(&args[1])?,
            int_literal_value(&args[2])?,
        ),
        _ => return None,
    };
    if step == 0.0 {
        return None;
    }
    let mut values = Vec::new();
    let mut current = start;
    // r[i] = start + step*i, while r[i] < stop (step > 0) or r[i] > stop
    // (step < 0) — library/stdtypes.html#range
    if step > 0.0 {
        while current < stop {
            values.push(known_number_sorted(current, PrimitiveKind::Integer));
            current += step;
        }
    } else {
        while current > stop {
            values.push(known_number_sorted(current, PrimitiveKind::Integer));
            current += step;
        }
    }
    Some(values)
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value,
/// tagged with its own CPython sort (Integer for an int literal, Float
/// for a float literal) — or `None` for anything else (complex, an int
/// too large for i64, a non-literal expression).
fn sorted_number_literal_value(expression: &Expr) -> Option<AbstractValue> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| known_number_sorted(value as f64, PrimitiveKind::Integer)),
            Number::Float(value) => Some(known_number_sorted(*value, PrimitiveKind::Float)),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) => {
            let operand = sorted_number_literal_value(unary.operand.as_ref())?;
            match unary.op {
                UnaryOp::USub => Some(known_number_sorted(-operand.values[0], operand.kind_tag?)),
                UnaryOp::UAdd => Some(operand),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression). Sort-erased: used only by
/// the `while`-counter comparison paths, which read a bound value to
/// compare against, never to bind a fresh iterate.
pub(super) fn number_literal_value(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            Number::Float(value) => Some(*value),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) => {
            let operand = number_literal_value(unary.operand.as_ref())?;
            match unary.op {
                UnaryOp::USub => Some(-operand),
                UnaryOp::UAdd => Some(operand),
                _ => None,
            }
        }
        _ => None,
    }
}

/// A `range()` argument's value, restricted to an INT literal (`range`
/// rejects a float argument at call time — this function will not
/// treat `range(3.0, 5)` as known, staying honest about that CPython
/// restriction rather than silently truncating).
fn int_literal_value(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            _ => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = int_literal_value(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
}
