use std::collections::HashMap;
use std::sync::Arc;

use ruff_python_ast::Expr;
use ruff_python_ast::StmtAssign;
use ruff_text_size::Ranged;

use crate::env::Environment;
use crate::typereading::callable_return_refinement;
use crate::typereading::DeclaredRefinement;

use super::super::Finding;
use super::super::WalkContext;
use super::bind_or_forget_target;
use super::bind_walrus_targets;
use super::forget_target_names;
use super::judge_and_bind;
use super::sink_value;

/// A plain `Assign` (`a = b = value`, or a single-target `a = value`):
/// evaluates the RHS once, then binds each target left to right,
/// exactly matching CPython's own multi-target assignment order
/// (simple_stmts.rst, "Assignment statements": "An assignment statement
/// evaluates the expression list... and assigns the single resulting
/// object to each of the target lists, from left to right"). A
/// bare-Name target with a recorded declared refinement in this body's
/// table (from an earlier `x: Age` or `x: Age = …`) judges the
/// evaluated value against it through the shared refused-write law
/// — `Fire` anchors to the VALUE expression's range, so a chained
/// `a = b = 200` with both `a` and `b` declared fires once per declared
/// target, all at the same value range. A target with no recorded
/// refinement binds (or, for a destructuring target, forgets) exactly
/// as before.
/// `cast(Callable[[...], R], <expr>)` — `typing.cast`'s own docstring
/// ("returns the value unchanged... signals that the return value has
/// the designated type"), read for its FIRST argument's own type
/// expression: `builtin_models::cast_call` only ever sees already-
/// EVALUATED `AbstractValue`s (the identity function over its second
/// argument), so it has no access to the syntactic `Callable[[...], R]`
/// the first argument SPELLS — that annotation is read here instead,
/// straight off the call's own AST, the same `typereading::callable_
/// return_refinement` reader `seed_parameters`/`walk_ann_assign` already
/// use for a `Callable`-shaped parameter/annotation. `None` for any
/// other callee (not a bare `cast` call — no `SurfaceImports` identity
/// for `cast` exists any more than one exists for `Callable` itself,
/// the same no-import-identity convention `callable_return_refinement`'s
/// own doc already takes), wrong arity, or a first argument that is not
/// a `Callable[[...], R]` subscript.
pub(in crate::check) fn cast_to_callable_return(
    value_expr: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    let Expr::Call(call) = value_expr else {
        return None;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return None;
    };
    if callee_name.id.as_str() != "cast" {
        return None;
    }
    let [typ, _val] = &*call.arguments.args else {
        return None;
    };
    callable_return_refinement(typ, context.aliases, context.imports, environment)
}

pub(in crate::check) fn walk_assign(
    assign: &StmtAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    bind_walrus_targets(assign.value.as_ref(), context, aug_assign_refinements, environment, out);
    // CAST-TO-CALLABLE ASSIGNMENT: `g = cast(Callable[[...], R], f)` —
    // tried BEFORE the ordinary `sink_value` read below (`cast_call`'s
    // own doc: the cast is the identity function over its second
    // argument, so `sink_value` still answers `g`'s own VALUE correctly
    // either way; this only ADDS the return-refinement fact a later
    // `g(...)` call site needs, the same channel a `Callable`-annotated
    // target already grows). Scoped to a single bare-Name target — a
    // chained `a = b = cast(...)` or a destructuring target is ordinary
    // Python this channel does not special-case.
    if let [Expr::Name(target_name)] = assign.targets.as_slice() {
        if let Some(callable_declared) = cast_to_callable_return(assign.value.as_ref(), context, environment) {
            let mut callable_returns = environment
                .callable_returns()
                .map(|table| (**table).clone())
                .unwrap_or_default();
            callable_returns.insert(target_name.id.as_str().to_owned(), callable_declared);
            environment.set_callable_returns(Arc::new(callable_returns));
        }
    }
    let Some(value) = sink_value(assign.value.as_ref(), context, environment, aug_assign_refinements, out) else {
        // a provable raise already pushed its own RTS7001 — every
        // target this assignment would have bound holds nothing.
        for target in &assign.targets {
            forget_target_names(target, environment);
        }
        return;
    };
    for target in &assign.targets {
        match target {
            Expr::Name(name) => match aug_assign_refinements.get(name.id.as_str()) {
                Some(declared) => {
                    let declared = declared.clone();
                    judge_and_bind(
                        name.id.as_str(),
                        value.clone(),
                        &declared,
                        assign.value.range(),
                        context,
                        environment,
                        out,
                    );
                }
                None => environment.bind(name.id.as_str(), value.clone()),
            },
            _ => bind_or_forget_target(
                target,
                &value,
                assign.value.range(),
                context,
                aug_assign_refinements,
                environment,
                out,
            ),
        }
    }
}
