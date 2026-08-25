//! Statement-side method calls: `name.method(args)` on a known
//! instance, interpreted through the restricted method interpreter.

use refined_domain::abstract_value::{AbstractValue, Kind};
use ruff_python_ast::Expr;

use crate::check::WalkContext;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances;

/// STATEMENT-SIDE METHOD CALLS: `name.method(args)` where `name` reads
/// as a known instance (`Kind::Object`, a non-empty `source` naming a
/// `ClassModel` in `context.classes` — `instances::judge_construction`'s
/// own tagging) and the class declares `method` (`instances::
/// method_def_of`). Every positional argument evaluates in source
/// order; keyword arguments map onto the method's own remaining
/// parameter positions (`self` excluded) by name
/// (`keyword_arguments_by_position`) — `None` when a keyword names no
/// parameter, two arguments claim the same position, or a position
/// before the last-filled one is left open (this domain has no
/// argument-gap representation to hand `method_call_result`, whose own
/// contract reads a positional PREFIX and falls back to each
/// parameter's default only past the end of it).
/// `instances::method_call_result` interprets the method's body: `Some`
/// REBINDS the receiver to the returned working instance (any
/// `self.<field> = ...` write inside the method survives) and answers
/// the method's own return value as this sink's value; `None` (the
/// method's body or parameter shape is outside the restricted
/// interpreter, or `method`/the receiver's class is not found) declines
/// this path entirely, and the caller falls through to construction
/// then the ordinary `evaluate_expression` reading, exactly as before
/// this law — no receiver forgetting happens here; `sink_value`'s own
/// caller (`walk_return`/`walk_ann_assign`/`walk_assign`) still forgets
/// on the FIRST unproducible value the same way it always did.
///
/// The class table read here is `environment.classes()`, falling back
/// to `context.classes` when the environment carries none: a class
/// defined LOCALLY inside the walked body only lives in
/// `environment.classes()` (`merged_classes_for_body`'s own merge over
/// `context.classes`), so reading `context.classes` alone would miss it
/// — the same locality gap `merged_classes_for_body`'s own doc names
/// for `context.classes` elsewhere.
pub(in crate::check) fn instance_method_call_result(
    expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
) -> Option<AbstractValue> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(receiver_name) = attribute.value.as_ref() else {
        return None;
    };
    let instance = environment.read(receiver_name.id.as_str())?.clone();
    if instance.kind != Kind::Object || instance.source.is_empty() {
        return None;
    }
    let classes = environment.classes().unwrap_or(&context.classes);
    let model = classes.get(instance.source.as_str())?;
    let method = instances::method_def_of(model, attribute.attr.as_str())?;
    let arguments = keyword_arguments_by_position(call, method, context, environment)?;
    let datetime_imports = environment.datetime_imports().unwrap_or(&context.datetime_imports);
    let (new_instance, result) = instances::method_call_result(
        &instance,
        model,
        method,
        &arguments,
        Some(&context.functions),
        Some(classes),
        Some(datetime_imports),
        context.kernel,
        environment.call_depth(),
    )?;
    environment.bind(receiver_name.id.as_str(), new_instance);
    Some(result)
}

/// A method call's own arguments, mapped positionally against `method`'s
/// parameters (`self` excluded) — every positional argument fills the
/// front slots in order; every keyword argument fills its OWN named
/// parameter's slot. `None` when a keyword names no parameter, a
/// position is claimed twice (a positional AND a keyword landing on the
/// same slot), or the filled positions leave a GAP before the
/// last-filled one — `method_call_result`'s own contract only reads a
/// positional PREFIX (`arguments[index]`, falling back to the
/// parameter's own default only past `arguments.len()`), so a gap has
/// no honest representation to hand it.
pub(in crate::check) fn keyword_arguments_by_position(
    call: &ruff_python_ast::ExprCall,
    method: &ruff_python_ast::StmtFunctionDef,
    context: &WalkContext,
    environment: &Environment,
) -> Option<Vec<AbstractValue>> {
    let parameters: Vec<_> = method
        .parameters
        .posonlyargs
        .iter()
        .chain(method.parameters.args.iter())
        .collect();
    // the first parameter is `self` by convention (instances.rs's own
    // stated assumption) — a method with no parameter at all has no
    // receiver slot, so this shape does not apply.
    let (_self_parameter, rest) = parameters.split_first()?;
    if call.arguments.args.len() > rest.len() {
        return None;
    }
    let mut slots: Vec<Option<AbstractValue>> = vec![None; rest.len()];
    for (index, argument) in call.arguments.args.iter().enumerate() {
        slots[index] = Some(evaluate_expression(argument, environment, context.kernel));
    }
    for keyword in &call.arguments.keywords {
        let name = keyword.arg.as_ref()?;
        let position = rest.iter().position(|p| p.parameter.name.id.as_str() == name.as_str())?;
        if slots[position].is_some() {
            return None;
        }
        slots[position] = Some(evaluate_expression(&keyword.value, environment, context.kernel));
    }
    let last_filled = slots.iter().rposition(|slot| slot.is_some());
    let Some(last_filled) = last_filled else {
        return Some(Vec::new());
    };
    let mut filled = Vec::with_capacity(last_filled + 1);
    for slot in slots.into_iter().take(last_filled + 1) {
        filled.push(slot?);
    }
    Some(filled)
}
