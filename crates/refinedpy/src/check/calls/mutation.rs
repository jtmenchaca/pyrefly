//! STALE-RECEIVER SOUNDNESS, law (a): an expression-statement method
//! call replays the mutation or forgets the receiver, so a stale
//! pre-call fact never survives it.

use std::collections::HashMap;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::assignability::judge;
use crate::assignability::Verdict;
use crate::check::Finding;
use crate::check::WalkContext;
use crate::collection_models::mutated_receiver;
use crate::env::Environment;
use crate::expressions::provable_raise;
use crate::typereading::DeclaredRefinement;

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
/// THE ELEMENT SINK: an `append`/`extend` onto a receiver whose own
/// DECLARATION states an element refinement (`xs: list[Age]`, whose
/// `DeclaredRefinement.element` carries `Age`'s window) judges the
/// appended value against THAT element, at the argument's own range,
/// before the replay rebinds anything. Without it, `xs.append(200)`
/// merely WIDENED `xs`'s own element window to admit 200
/// (`set_mutated_receiver`'s join is the right answer for an
/// undeclared receiver) and said nothing at the append itself — the
/// out-of-window value then surfaced at whatever LATER sink read `xs`,
/// which reported the same one defect at a position that is not where
/// it was introduced.
///
/// A judged element also STOPS the widening: where the append is
/// admitted, the receiver keeps its declared element set rather than
/// joining the argument in (the declaration already states that the
/// argument is inside it, so the join adds nothing); where it fires,
/// the receiver likewise keeps its declared set, since the program is
/// being told to fix the append rather than to carry a widened claim
/// downstream. Only a receiver with NO declared element still widens,
/// exactly as before.
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
    declared_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
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
    // THE ELEMENT SINK (this function's own doc): judge the appended /
    // extended value against the receiver's DECLARED element, at the
    // argument's own range, and keep the declared element set rather
    // than widening it.
    if let Some(element) = declared_refinements
        .get(receiver_name.id.as_str())
        .and_then(|declared| declared.element.as_deref())
    {
        let judged: Vec<(&AbstractValue, TextRange)> = match method {
            "append" => arguments.iter().map(|(value, range)| (value, *range)).collect(),
            // `extend`'s argument is the ITERABLE — its own items are what
            // land in the receiver, so a known `Kind::List` argument is
            // judged item by item at the iterable's own range. An
            // unread iterable states no items to judge, and falls
            // through to the ordinary replay unchanged.
            "extend" => match arguments.as_slice() {
                [(iterable, range)] if iterable.kind == Kind::List => {
                    iterable.items.iter().map(|item| (item, *range)).collect()
                }
                _ => Vec::new(),
            },
            _ => Vec::new(),
        };
        if !judged.is_empty() {
            for (value, range) in judged {
                if let Verdict::Fire(message) = judge(value, element, context.kernel) {
                    out.push(Finding { range, code: "RTS7001", message });
                }
            }
            // the declaration already states which elements this receiver
            // holds; the append is judged against it rather than widening
            // it, so the binding stays exactly what the declaration says
            return true;
        }
    }
    match mutated_receiver(method, &receiver_value, &argument_values) {
        Some((new_receiver, _result)) => environment.bind(receiver_name.id.as_str(), new_receiver),
        // An unmodeled method may have mutated the receiver in a way this
        // walk cannot replay: forgets it NAMING the call itself as the
        // cause (`expr`'s own range — the whole `name.method(args)` call)
        // so the LAST-TOUCH LEDGER's later stamp on a declined read of
        // this name reads "havocked by `s.add(x)` @…" rather than the
        // bare "forgotten" a cause-less forget would leave.
        None => environment.forget_with_cause(
            receiver_name.id.as_str(),
            (usize::from(expr.range().start()), usize::from(expr.range().end())),
        ),
    }
    true
}
