use std::collections::HashMap;

use refined_domain::abstract_value::AbstractValue;
use ruff_python_ast::Expr;

use crate::collection_models::dict_get_result;
use crate::collection_models::dict_with_item;
use crate::collection_models::mutated_receiver;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::expressions::possible_raise;
use crate::expressions::provable_raise;
use crate::expressions::register_retained_callables;
use crate::typereading::DeclaredRefinement;

use super::super::Finding;
use super::super::WalkContext;
use super::apply_call_effects;
use super::callable_variable_call_result;
use super::construction_call_verdict;
use super::instance_method_call_result;
use super::manifest_call_fires;
use super::same_module_call_argument_fires;
use super::same_module_def_call_result_already_reported;

/// `receiver.setdefault(key, default).append(appended)` — the manual
/// group-by chain (c-reads-and-values.py's `dict_groupby`:
/// `grouped.setdefault("old" if age > 100 else "young",
/// []).append(age)`, stdtypes.rst's `dict.setdefault` twin of
/// `Map.groupBy`). Composes three EXISTING `collection_models`
/// functions rather than inventing new dict/list machinery: (1)
/// `dict_get_result(receiver, key, Some(default))` reads the entry
/// `setdefault` would have returned — present-key's own value, or
/// `default` on a miss (the identical present/absent rule
/// `dict_mutated_receiver`'s own `"setdefault"` arm already encodes,
/// reused here read-only since this function needs the entry's value
/// TWICE: once to append onto, once implicitly to know whether it was
/// already in `receiver`); (2) `mutated_receiver("append", entry,
/// &[appended])` appends onto that entry, requiring it to be a known
/// `Kind::List` (a `default` this caller did not itself pass as `[]`
/// would decline here, same as any other non-list append target); (3)
/// `dict_with_item(receiver, key, &appended_entry)` writes the grown
/// list back — inserting a NEW entry when `key` was absent, overwriting
/// the existing one otherwise, exactly `setdefault`'s own dual
/// insert-or-return contract PLUS the append, folded into the receiver
/// this single chained statement actually produces. `None` the moment
/// any step declines (a non-dict receiver, a key this walk cannot read
/// exactly, an entry that is not a known list) — the caller must not
/// assume the receiver is unchanged, the same honesty every other
/// decline in this file already keeps.
pub fn setdefault_append(
    receiver: &AbstractValue,
    key: &AbstractValue,
    default: &AbstractValue,
    appended: &AbstractValue,
) -> Option<AbstractValue> {
    let entry = dict_get_result(receiver, key, Some(default))?;
    let (grown_entry, _) = mutated_receiver("append", &entry, &[appended.clone()])?;
    dict_with_item(receiver, key, &grown_entry)
}

/// One `import`/`from…import` local name at its own import statement:
/// bind it to whatever `context.module_bindings` resolved for it (the
/// cross-module surface already did the resolving), or forget it when
/// the surface carries nothing under that name — a function/class
/// import (readable through `environment.functions()`/`.classes()`,
/// not a plain value), an unresolved module, or a star import's own
/// literal `"*"` alias (never a real local name).
pub(in crate::check) fn bind_or_forget_imported_name(local_name: &str, context: &WalkContext, environment: &mut Environment) {
    match context.module_bindings.get(local_name) {
        Some(value) => environment.bind(local_name, value.clone()),
        None => environment.forget(local_name),
    }
}

/// The value a write/return/expression-statement sink's own value
/// expression produces, after three checks the ordinary
/// `evaluate_expression` path does not make on its own:
///
/// 1. A PROVABLE RAISE (`expressions::provable_raise`): a call whose
///    real CPython execution is proven to always raise — pushed as an
///    RTS7001 at the raising expression (the mission's PRODUCT
///    decision: a provable runtime raise is spoken there, not as a
///    silent unknown). The sink then produces NOTHING: `None` here
///    means "unproducible," and every caller forgets its target rather
///    than binding a value, since no execution of this statement ever
///    reaches a value to bind.
/// 2. STATEMENT-SIDE METHOD CALLS (`instance_method_call_result`): a
///    call shaped `name.method(args)` on a bare-Name receiver bound to
///    a known instance — the method's own body interprets through
///    `instances::method_call_result`, REBINDING `name` to the
///    returned (possibly self-mutated) instance, and the sink's value
///    is the method's own return value (b-body-expressions.py's
///    `literal_writing_method`: `outlaw.spoil()` writes `self.age =
///    200` inside the method body, and a LATER `outlaw.age` read must
///    see it). Tried before construction, since a bare-Name call and an
///    attribute call are syntactically disjoint shapes anyway.
/// 3. Statement-level CONSTRUCTION (`construction_call_verdict`): a
///    call recognized as building a same-module or imported
///    `ClassModel` instance. Each fire `judge_construction` returns is
///    pushed as its own RTS7001, and the sink's value is
///    `verdict.instance` — never the plain `evaluate_expression`
///    reading of an unmodeled call.
/// 4. A CALLABLE-VARIABLE CALL (`callable_variable_call_result`): a
///    call on a bare Name this environment's `callable_returns` table
///    carries — a `Callable[[...], R]`-annotated variable
///    (`walk_ann_assign`'s own recording seam). The sink's value is
///    `R`'s own declared set (`known_set`, TrustSpec — an annotation
///    states the developer's claim, not an execution-proved fact), so
///    a call through it judges at whatever sink it flows into
///    (b-body-expressions.py:79's `maybe_next_year(40) if ... else 0`
///    — the containment law fires `R`'s whole-number claim against
///    `Age`). A call to a POSSIBLY-None callable (the variable's own
///    `X | None` wrapper) additionally RAISES if the variable actually
///    holds `None` at the call — not modeled here; this path only
///    answers the value a SUCCESSFUL call produces.
///
/// 5. The CALLEE-EFFECTS CHANNEL (`apply_call_effects`): a bare-Name,
///    same-module call whose body writes an ENCLOSING name (`nonlocal`,
///    or a mutation through a captured free name) — every effect applies
///    against `environment` here, exactly as it does at an
///    expression-statement call site, and the sink's own value is
///    whatever `evaluate_expression`'s ordinary same-module-call path
///    already answers (this channel never changes the RETURNED value,
///    only the enclosing side effects riding alongside it).
///
/// No check applies: falls through to the ordinary `evaluate_expression`
/// reading, unchanged from before this unit.
pub(in crate::check) fn sink_value(
    expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) -> Option<AbstractValue> {
    // RETAINED CALLABLES: a lambda nested in `expr` (a call argument —
    // `pick(lambda s: s.age)` — or a constructor argument — `Person
    // (lambda: 40)`) is registered into `environment` BEFORE any of the
    // immutable evaluation paths below run — `construction_call_
    // verdict`/`evaluate_expression` only ever read `&Environment`, so
    // this is the last point with `&mut Environment` before the lambda
    // is read as a value (`expressions.rs::register_retained_
    // callables`'s own doc).
    register_retained_callables(expr, environment);
    if let Some((range, message)) = provable_raise(expr, environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
        return None;
    }
    // A SOMETIMES-raise fires and evaluation continues: the divisor's
    // set admits 0 among other values, so some runs raise and the rest
    // produce the split value — the finding and the value both stand
    // (`expressions.rs::possible_raise`'s own claim).
    if let Some((range, message)) = possible_raise(expr, environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
    }
    if let Some(result) = instance_method_call_result(expr, context, environment) {
        return Some(result);
    }
    if let Some(verdict) = construction_call_verdict(expr, context, environment) {
        for (range, message) in verdict.fires {
            out.push(Finding { range, code: "RTS7001", message });
        }
        return Some(verdict.instance);
    }
    if let Some(result) = callable_variable_call_result(expr, context, environment) {
        return Some(result);
    }
    // The ARGUMENT-crossing judge runs before the same-module return
    // rung below: that rung returns early with the call's value, and an
    // out-of-set WRITTEN argument (`takes_non_negative(-1)`) is a
    // separate defect at the call's own site that the early return must
    // not swallow.
    same_module_call_argument_fires(expr, context, environment, out);
    if let Some(result) = same_module_def_call_result_already_reported(expr, context, environment) {
        return Some(result);
    }
    // RUNG 2 — THE MANIFEST READER TEMPLATE
    // (`packages/cpp/findings/python-c-extension-boundary.md`): a call
    // recognized against a discovered manifest judges its own written
    // arguments against the manifest's parsed entry contract HERE, at
    // this expression's own sink — an escaping argument fires the same
    // way any other refused write does. The call's own VALUE still falls
    // through to the ordinary `evaluate_expression` reading below
    // (`unknown()`, since no arm in that dispatcher recognizes a
    // manifested module either) — the naming of THAT undetermined value
    // (the "no producer exports its return fact" sentence) happens later,
    // at whichever declared sink judges it, through
    // `name_unmodeled_call_sentence`'s own manifest-aware naming step.
    manifest_call_fires(expr, context, environment, out);
    apply_call_effects(expr, context, environment, aug_assign_refinements, out);
    Some(evaluate_expression(expr, environment, context.kernel))
}
