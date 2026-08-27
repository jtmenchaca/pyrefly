//! The callee-effects channel: a bare-Name, same-module call whose
//! callee's body writes to a name in this body's own enclosing scope.

use std::collections::HashMap;

use ruff_text_size::Ranged;
use ruff_python_ast::Expr;

use crate::check::{body_may_write_through_parameter, judge_and_bind, Finding, WalkContext};
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::summaries;
use crate::typereading::DeclaredRefinement;

/// CALLEE-EFFECTS CHANNEL: a bare-Name, same-module call
/// (`bump()`/`spoil()` — a-statements.py's own `closure_mutates_
/// flattened_capture`/`nonlocal_rebind` rows) whose callee's body writes
/// to a name in THIS body's own enclosing scope, either through a
/// `nonlocal` declaration or a mutation THROUGH a captured free name
/// (`summaries::call_effects`'s own two effect kinds — see that
/// function's doc for the CPython citations). Every effect the callee
/// reports is applied here, against `environment` — a name this body's
/// own `aug_assign_refinements` table declares (an `age: Age = …` seen
/// earlier in straight-line order) judges the effect value through
/// `judge_and_bind`, exactly as an ordinary straight-line `age = 200`
/// would (this is what makes `nonlocal_rebind`'s own row FIRE: `age` is
/// a declared `Age` slot in the CALLER's own body, and the callee's
/// effect value is 200); every other name simply rebinds. `Some(())`
/// when the call matched this shape (whether or not the callee reported
/// any effects at all — a same-module def with an empty effect list
/// still matched, and the caller must not ALSO try `sink_value`'s own
/// plain-call reading, which would re-evaluate the call through
/// `evaluate_expression` and answer a value with no effects applied);
/// `None` for every other shape (an attribute call, a name with no
/// same-module def, a def `call_effects` itself declines — the depth
/// cap, an unsupported parameter shape, or a body statement the
/// restricted interpreter does not read), so the caller falls through
/// to its own existing dispatch order unchanged. The STALE-ARGUMENT drop
/// below (a positional argument's recorded star entries, dropped when
/// the resolved def's body may write through the matched parameter)
/// runs once `def` itself resolves, regardless of whether `call_effects`
/// goes on to decline — a mutation already applied to `environment` is
/// never undone by this function's own later `None` return.
pub(in crate::check) fn apply_call_effects(
    expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) -> Option<()> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
    // `callee_name` must be genuinely UNBOUND — a real value bound to
    // the same name shadows the def (the same "a real value shadows the
    // def name" rule `expressions.rs`'s own `same_module_def_gate_open`
    // states for its identical gate, private to that module so this
    // narrower re-check covers the ordinary case: bump()/spoil() are
    // never themselves reassigned in the corpus's own rows).
    if environment.read(callee_name.id.as_str()).is_some() {
        return None;
    }
    // reads the CURRENT environment's own function table, not
    // `context.functions` alone — a body-local `def bump(): ...` nested
    // inside the enclosing function (a-statements.py's own
    // `closure_mutates_flattened_capture`/`nonlocal_rebind` shape) is
    // merged into `environment.functions()` by `walk_body_with_self_
    // binding` (`local_function_table` merged over `context.functions`),
    // never present in `context.functions` alone.
    let functions = environment.functions()?.clone();
    let def = functions.def(callee_name.id.as_str())?;
    let arguments: Vec<refined_domain::abstract_value::AbstractValue> =
        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, context.kernel)).collect();
    // STALE-RECEIVER SOUNDNESS FOR A GUARD'S RECORDED ENTRIES
    // (`Environment::forget_recorded_star_entries`'s own doc): a
    // positional argument that is a bare Name bound to `d`'s matched
    // parameter is handed to THIS callee's body — if that body may write
    // through the parameter (`body_may_write_through_parameter`, any
    // subscript/attribute store/delete or method call on it), the
    // argument's own recorded presence facts go stale the moment the
    // callee runs, since the write can happen BEFORE `call_effects`'s own
    // (unrelated, enclosing-scope-only) effect list is computed below.
    // Dropped before `call_effects` runs, not after, so this holds
    // whether or not that channel itself matches this callee at all.
    let positional_parameters: Vec<&str> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .map(|parameter| parameter.parameter.name.id.as_str())
        .collect();
    for (parameter_name, arg) in positional_parameters.iter().zip(call.arguments.args.iter()) {
        let Expr::Name(argument_name) = arg else {
            continue;
        };
        if body_may_write_through_parameter(&def.body, parameter_name) {
            environment.forget_recorded_star_entries(argument_name.id.as_str());
        }
    }
    let (_value, effects) = summaries::call_effects(def, &arguments, Some(&functions), context.kernel, environment.call_depth(), environment)?;
    for (name, effect_value) in effects {
        match aug_assign_refinements.get(name.as_str()) {
            Some(declared) => {
                let declared = declared.clone();
                judge_and_bind(&name, effect_value, &declared, call.range(), context, environment, out);
            }
            None => environment.bind(&name, effect_value),
        }
    }
    Some(())
}
