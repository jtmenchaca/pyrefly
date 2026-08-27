//! `reduce`/`map`/`filter` folded CONCRETELY over a RAW callable
//! expression (a `Lambda` or a bare `Name` naming a same-module `def`)
//! rather than an already-evaluated value — the one seam in this file
//! that reads a call argument's own AST node instead of its value.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::truthiness;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::Expr;

use crate::collection_models;
use crate::env::Environment;
use crate::summaries;

use super::super::evaluate_expression;

/// One call to a RAW two-parameter callable expression: an
/// `Expr::Lambda` of exactly two parameters (its body is always a
/// single expression, expressions.rst's "Lambdas" — evaluated directly
/// against a fork binding both parameters), or a bare `Expr::Name`
/// resolving to a same-module `def` in the function table (folded
/// through `summaries::call_result`, the same restricted interpreter
/// every other same-module call in this file already uses). Any other
/// callable shape (a builtin name, a method reference, a lambda/def of
/// a different arity) declines.
pub(in super::super) fn call_two_argument_expression(
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
pub(crate) fn call_one_argument_expression(
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

/// `sorted(iterable, key=..., reverse=...)` over a receiver whose value
/// is a REPETITION WINDOW — see the call site's own doc in
/// `evaluate_call` for the clause reading and for why a known
/// `Kind::List` receiver is not answered here.
///
/// The window is the whole answer: `as_repetition` reads it back to one
/// element set repeated over an item-count range, which states nothing
/// about the ORDER of the positions, so a reordering leaves it exactly
/// as it was. `key=` and `reverse=` are the only keywords `sorted`
/// accepts (functions.rst pins the signature `sorted(iterable, /, *,
/// key=None, reverse=False)`), and neither changes the item set, so
/// their VALUES are never read here — a `key` this domain could not
/// evaluate would still not change the answer.
///
/// `None` for a receiver that is not a bare repetition window, more than
/// one positional argument, or a keyword outside that pair (a spelling
/// `sorted` itself would reject).
pub(in super::super) fn sorted_over_star_with_keywords(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let [iterable_expr] = &*call.arguments.args else {
        return None;
    };
    for keyword in &call.arguments.keywords {
        let name = keyword.arg.as_ref()?;
        if name.id.as_str() != "key" && name.id.as_str() != "reverse" {
            return None;
        }
    }
    let iterable = evaluate_expression(iterable_expr, environment, kernel);
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    as_repetition(&iterable.set)?;
    Some(iterable)
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
/// evaluate. Called from BOTH `evaluate_call`'s `Expr::Attribute` arm
/// (`functools.reduce(...)`, the qualified spelling `import functools`
/// leaves) and its `Expr::Name` arm (`reduce(...)`, `from functools
/// import reduce`) — this function itself reads `call.arguments` only,
/// never `call.func`, so one fold implementation serves both call
/// shapes.
pub(in super::super) fn reduce_expression_value(
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
pub(in super::super) fn map_expression_value(
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
pub(in super::super) fn filter_expression_value(
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
