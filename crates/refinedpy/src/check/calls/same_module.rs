//! Same-module call judging: a call's own written arguments against the
//! callee's declared parameter refinements, a callable-variable's
//! declared return set, and a same-module def's already-reported
//! escaping return.

use refined_domain::abstract_value::{known_set, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::{on_one_tuple_layer, requires_integer};
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;

use refined_domain::abstract_value::AbstractValue;

use crate::assignability::{judge, states_sequence, Verdict};
use crate::check::{Finding, WalkContext};
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::typereading::declared_refinement;

use super::construction::construction_call_verdict;

/// A SAME-MODULE CALL's own written arguments, judged against the
/// callee's OWN declared parameter refinements — `record_ratio(float
/// ("nan"))`'s own shape, where `record_ratio(r: Ratio)` states a
/// refined sink at the parameter position and the caller's argument
/// must cross into it. Mirrors `manifest_call_fires`'s law for a
/// foreign manifest entry, at the same sink: judged HERE, at the call
/// expression's own site, because the call's VALUE (what
/// `evaluate_expression`'s same-module-call path computes from the
/// callee's body) is an entirely separate question from whether the
/// PASSED argument itself crosses into the parameter's declared set —
/// `bind_parameters`/`positional_arguments_for_def` bind and evaluate
/// argument values only to replay the callee's body for its return
/// value, and judge nothing.
///
/// Only a WRITTEN, MATCHED argument's own sort is judged — an arity
/// mismatch, a starred/spread argument, or a keyword naming no
/// parameter contributes no fire here (the same restraint
/// `judge_manifest_call`'s own doc states for its identical shape). A
/// parameter with no annotation this table can read (`declared_
/// refinement` answers `None`) is skipped; nothing is judged against
/// an unrefined slot. `Verdict::Undetermined` is dropped — there is no
/// body-level blocker for a call's own argument crossing, the same
/// "verdict's fires only" restraint `judge_construction`'s own call
/// site takes for a construction's own arguments.
pub(in crate::check) fn same_module_call_argument_fires(expr: &Expr, context: &WalkContext, environment: &Environment, out: &mut Vec<Finding>) {
    let Expr::Call(call) = expr else {
        return;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return;
    };
    // A real value bound to the same name shadows the def — the same
    // narrower re-check `apply_call_effects` takes for its own
    // identical gate, private to `expressions.rs`.
    if environment.read(callee_name.id.as_str()).is_some() {
        return;
    }
    let Some(functions) = environment.functions() else {
        return;
    };
    let Some(def) = functions.def(callee_name.id.as_str()) else {
        return;
    };
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return;
    }
    if call.arguments.keywords.iter().any(|kw| kw.arg.is_none()) {
        return;
    }
    let positional_parameters: Vec<&ruff_python_ast::ParameterWithDefault> =
        def.parameters.posonlyargs.iter().chain(def.parameters.args.iter()).collect();
    for (parameter, arg) in positional_parameters.iter().zip(call.arguments.args.iter()) {
        judge_one_call_argument(parameter, arg, context, environment, out);
    }
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            continue;
        };
        let named = positional_parameters
            .iter()
            .copied()
            .chain(def.parameters.kwonlyargs.iter())
            .find(|parameter| parameter.parameter.name.id.as_str() == arg_name.as_str());
        if let Some(parameter) = named {
            judge_one_call_argument(parameter, &keyword.value, context, environment, out);
        }
    }
}

/// One call argument's own crossing judge against its matched
/// parameter's declared refinement — `same_module_call_argument_
/// fires`'s per-position body, factored out so the positional and
/// keyword loops share it instead of repeating the annotation read
/// and `judge` call.
pub(in crate::check) fn judge_one_call_argument(
    parameter: &ruff_python_ast::ParameterWithDefault,
    argument: &Expr,
    context: &WalkContext,
    environment: &Environment,
    out: &mut Vec<Finding>,
) {
    let Some(annotation) = parameter.parameter.annotation.as_deref() else {
        return;
    };
    // A CLASS-TYPED PARAMETER (`v: Vitals`, a self-authored/pydantic
    // model — `declared_refinement`'s `Expr::Name` arm only ever reads
    // `context.aliases`, never `context.classes`, so a class name
    // answers `None` there and is not a scalar/sequence/tuple
    // refinement this table judges). The parameter itself states no
    // FURTHER scalar set past "an instance of this class" — but the
    // ARGUMENT expression may still be a nested construction call
    // (`Vitals(heart_rate=72, spo2=130)`) whose own per-field crossing
    // is exactly what `construction_call_verdict`/`judge_construction`
    // already check. Without this arm, `declared_refinement` returning
    // `None` for the class name falls straight past the whole function
    // (the ordinary early-return below), and a construction's own
    // out-of-set field never reaches a Finding: showcase.py's own
    // `record_vitals(Vitals(heart_rate=72, spo2=130))` row. Surfaces
    // ONLY the construction's own field fires — there is no outer
    // scalar `judge` call afterward, since a class-typed parameter has
    // no scalar set for the built instance to cross into.
    if let Expr::Name(class_name) = annotation {
        let classes = environment.classes().unwrap_or(&context.classes);
        if classes.contains_key(class_name.id.as_str()) {
            if let Some(verdict) = construction_call_verdict(argument, context, environment) {
                for (range, message) in verdict.fires {
                    out.push(Finding { range, code: "RTS7001", message });
                }
            }
            return;
        }
    }
    let Some(declared) = declared_refinement(annotation, context.aliases, context.imports, environment) else {
        return;
    };
    // A construction nested in argument position has no statement sink
    // hosting it, and evaluate_expression's construction arm discards
    // judge_construction's per-field fires by design — so the verdict
    // is taken here: its fires land at their own field positions, and
    // its built instance is the value the parameter judges.
    let value = match construction_call_verdict(argument, context, environment) {
        Some(verdict) => {
            for (range, message) in verdict.fires {
                out.push(Finding { range, code: "RTS7001", message });
            }
            verdict.instance
        }
        None => evaluate_expression(argument, environment, context.kernel),
    };
    // A call argument crossing that the checker cannot decide is an
    // UNDETERMINED position, per the project's own DETERMINED-or-
    // UNDETERMINED doctrine — this argument's own name binds the
    // parameter's declared refinement, and an unprovable containment
    // there is exactly as much a defect as an unprovable one anywhere
    // else this crate already reports RTS7002 for; it must never be
    // silently dropped the way this call site did before.
    match judge(&value, &declared, context.kernel) {
        Verdict::Fire(message) => {
            out.push(Finding { range: argument.range(), code: "RTS7001", message });
        }
        Verdict::Undetermined(message) => {
            out.push(Finding { range: argument.range(), code: "RTS7002", message });
        }
        Verdict::Silent => {}
    }
}

/// A CALLABLE-VARIABLE CALL: `name(...)` where `name` is a bare Name
/// this environment's `callable_returns` table carries (a
/// `Callable[[...], R]`-annotated variable) AND `name` does not also
/// resolve to a same-module `def` or class — a name shadowing both an
/// (impossible, since one annotation names one thing) is never this
/// call's business, but the gate is kept honest anyway: a resolvable
/// def/class call is ALREADY answered by `evaluate_expression`'s own
/// same-module-call/construction paths (summaries::call_result /
/// instances::judge_construction), which read the callee's ACTUAL body
/// rather than its bare declared return sort, so this path only ever
/// answers a name those paths cannot. Answers `R`'s own declared set at
/// `TrustSpec` — the same grade `seed_parameters` gives a parameter's
/// declared-set seed, since an annotation is a claim, not a
/// proved fact. `None` when `expr` is not a bare-Name call, or the
/// name carries no callable-returns entry, or the name IS a
/// resolvable def/class (the ordinary paths own it instead).
pub(in crate::check) fn callable_variable_call_result(
    expr: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<AbstractValue> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return None;
    };
    let name = callee_name.id.as_str();
    let declared = environment.callable_returns()?.get(name)?;
    if environment.functions().is_some_and(|functions| functions.def(name).is_some()) {
        return None;
    }
    if context.classes.contains_key(name) {
        return None;
    }
    // Tags the numeric sort onward flow needs (the same guarded rule
    // `seed_parameters` applies to a declared set: numeric-ground only,
    // never the `Literal["A", "B"]` string-tuple pun `on_one_tuple_layer`
    // alone would also admit).
    if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
        let sort = if requires_integer(&declared.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        return Some(AbstractValue {
            kind_tag: Some(sort),
            ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
        });
    }
    Some(known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None))
}

/// A SAME-MODULE-DEF CALL whose own body already reports its own
/// escaping return: `findings_for_module_at` walks EVERY module-level
/// `def` on its own (this file's own doc, "each nested `def` gets its
/// own fresh body walk"), so a `def two_hundred() -> Age: return 200`
/// ALWAYS fires RTS7001 at its own `return 200`, independent of any
/// caller. Reading `two_hundred`'s call result at ANOTHER sink
/// (`f = two_hundred; over: Age = f()`, or a direct `over: Age =
/// two_hundred()`) must not ALSO fire there — that would report the
/// SAME defect twice ("one error per defect... at its own construct"),
/// once at the `return` statement and once at every caller.
///
/// Recognizes `expr` as a call whose callee resolves to a same-module
/// `def` — either directly by name, or through a same-module-def alias
/// value (`f = two_hundred`, `env::same_module_def_alias_name`) — and,
/// when that def states its own `-> Annotation` refinement, evaluates
/// the call exactly as `evaluate_expression` would and asks whether the
/// answer escapes it. An escaping answer substitutes the DECLARED SET
/// for the raw value, the same refused-slot substitution `judge_and_
/// bind_naming`'s own Fire arm already makes for a later read in the
/// SAME body — so this sink sees an in-window value and stays silent,
/// leaving the one true fire at the def's own `return`. A value that
/// stays inside the declared window, or a def whose return states no
/// refinement this table reads, or a callee this file cannot resolve to
/// a same-module def at all, all fall through `None` — the caller tries
/// every other channel and ultimately `evaluate_expression` unchanged.
pub(in crate::check) fn same_module_def_call_result_already_reported(
    expr: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<AbstractValue> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return None;
    };
    let functions = environment.functions().or(Some(&context.functions))?;
    let def = match environment.read(callee_name.id.as_str()) {
        Some(value) => {
            let aliased_name = crate::env::same_module_def_alias_name(value)?;
            functions.def(aliased_name)?
        }
        None => functions.def(callee_name.id.as_str())?,
    };
    let returns = def.returns.as_deref()?;
    let declared = declared_refinement(returns, context.aliases, context.imports, environment)?;
    let value = evaluate_expression(expr, environment, context.kernel);
    match judge(&value, &declared, context.kernel) {
        Verdict::Fire(_) => Some(known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)),
        Verdict::Silent | Verdict::Undetermined(_) => Some(value),
    }
}
