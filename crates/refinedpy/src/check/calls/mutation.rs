//! STALE-RECEIVER SOUNDNESS, law (a): an expression-statement method
//! call replays the mutation or forgets the receiver, so a stale
//! pre-call fact never survives it.

use ruff_python_ast::Expr;

use crate::check::WalkContext;
use crate::collection_models::mutated_receiver;
use crate::env::Environment;
use crate::expressions::provable_raise;

use super::construction::evaluate_positional_arguments;

/// STALE-RECEIVER SOUNDNESS, law (a): an expression-statement call shaped
/// `name.method(args)` (an `Attribute` func over a bare-`Name` receiver)
/// is a candidate MUTATION — `ages.append(30)`, `by_name["ann"] = 40`'s
/// sibling method form — and the walk must not let the receiver's
/// PRE-CALL value keep answering reads after it. The receiver and every
/// argument evaluate first (in source order, matching every other call
/// site's own argument evaluation), then
/// `collection_models::mutated_receiver` replays the call: `Some((new
/// receiver, _))` rebinds `name` to the replayed post-call value (the
/// call's own result is discarded here — an expression-statement's value
/// is never read, matching `Stmt::Expr`'s existing convention of
/// discarding `sink_value`'s answer too); `None` FORGETS `name` outright
/// — an unmodeled method may have mutated the receiver in a way this
/// walk cannot replay, so the stale pre-call fact must not survive
/// (a-statements.py's `collection_mutators`; c-reads-and-values.py's
/// `list_append`/`dict_set_item` rows).
///
/// Returns `true` when this shape matched (whether or not
/// `mutated_receiver` itself recognized the method) — the caller then
/// skips its own `sink_value` call, since the receiver name has already
/// been rebound/forgotten here and a plain `evaluate_expression` reading
/// of the call would tell the caller nothing further. Returns `false`
/// for every other statement shape (a bound-name shadowing the receiver,
/// a non-Attribute func, a non-Name receiver, a `Call` whose target is
/// not this shape at all) so the caller falls through to its own
/// `sink_value` path unchanged.
pub(in crate::check) fn walk_mutating_call_statement(
    expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(receiver_name) = attribute.value.as_ref() else {
        return false;
    };
    if provable_raise(expr, environment, context.kernel).is_some() {
        // a provable raise on this same call (e.g. a zero-argument
        // mismatch this walk can prove raises) is sink_value's own
        // channel to speak — decline the mutation shape so the caller's
        // ordinary sink_value path pushes that finding.
        return false;
    }
    let receiver_value = match environment.read(receiver_name.id.as_str()) {
        Some(value) => value.clone(),
        None => return false,
    };
    let method = attribute.attr.as_str();
    let arguments = evaluate_positional_arguments(&call.arguments.args, environment, context.kernel);
    let argument_values: Vec<refined_domain::abstract_value::AbstractValue> =
        arguments.iter().map(|(value, _)| value.clone()).collect();
    match mutated_receiver(method, &receiver_value, &argument_values) {
        Some((new_receiver, _result)) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
    true
}
