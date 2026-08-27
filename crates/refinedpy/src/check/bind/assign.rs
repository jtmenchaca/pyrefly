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

/// Every place an assignment's target list writes, in the ledger's own
/// spelling — `a` for `a = v`, both names for a chained `a = b = v`, and
/// each element name for a destructuring `a, b = v`. A destructuring
/// element's place is filed with the WHOLE right-hand side's derivation,
/// which is the derivation a reader blocked at that name needs to see:
/// the statement that failed to derive the tuple is the construct to go
/// fix. A literal-index subscript write (`d["k"] = v`) names the place
/// `d["k"]` the same way a read of it does. A target this reader does
/// not recognize as a place (a computed-index write, `d[i] = v`) names
/// none and is simply not in the ledger.
///
/// Answers an empty vector without reading a target while tracing is off:
/// the ledger does not exist then, so nothing is walked and nothing is
/// allocated on an ordinary check's path.
fn written_places_of(targets: &[Expr]) -> Vec<String> {
    let mut places: Vec<String> = Vec::new();
    if !crate::trace::is_tracing() {
        return places;
    }
    let mut pending: Vec<&Expr> = targets.iter().collect();
    while let Some(target) = pending.pop() {
        match target {
            Expr::Tuple(tuple) => pending.extend(tuple.elts.iter()),
            Expr::List(list) => pending.extend(list.elts.iter()),
            Expr::Starred(starred) => pending.push(starred.value.as_ref()),
            _ => {
                if let Some(place) = crate::env::tracked_place_of(target) {
                    places.push(place.words());
                }
            }
        }
    }
    places
}

pub(in crate::check) fn walk_assign(
    assign: &StmtAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    // THE BINDING LEDGER's write seam (DERIVATION-TRACE.md, the binding
    // ledger): opened before the right-hand side is read, so every
    // sub-read nests under one span rather than landing as unrelated
    // roots, and keyed by every place this statement writes. An
    // assignment is where a name's value is DERIVED, so this subtree is
    // exactly what a later read blocked at that bare name needs — the
    // ledger files it, and the reclaim hands it back as the trace's
    // `chain`, in ONE explain run.
    //
    // Unlike an ordinary dispatch span this records wherever the binding
    // sits rather than only on the requested line, since the binding
    // statement is almost never the line being explained.
    let _position_span = crate::trace::ledger_scope(
        written_places_of(&assign.targets),
        "check::walk_assign",
        usize::from(assign.value.range().start()),
        usize::from(assign.value.range().end()),
    );
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
    // What this assignment's right-hand side derived — the fact every
    // later read of the bound names depends on. An unread value declines
    // here, which is exactly the span a trace of a later blocked read
    // needs its reader to go find.
    if crate::trace::is_tracing() {
        let spelled = crate::expressions::spelled_value(&value);
        if value.kind == refined_domain::abstract_value::Kind::Unknown {
            crate::trace::record_decline(
                "this assignment's right-hand side derives no value, so every name it binds carries none",
                Some((usize::from(assign.value.range().start()), usize::from(assign.value.range().end()))),
                Some(&spelled),
            );
        } else {
            crate::trace::record_answer(&spelled);
        }
    }
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
        // TEMPORAL OFFSET LEDGER: whatever this write bound, the target
        // no longer holds whatever derivation it held — and it may hold
        // a NEW one (`env::TemporalOffsetDerivation`'s own doc for what
        // the ledger is for).
        if let Expr::Name(name) = target {
            environment.forget_temporal_offset(name.id.as_str());
            if let Some(derivation) = temporal_offset_derivation(assign.value.as_ref(), environment, context.kernel) {
                environment.record_temporal_offset(name.id.as_str(), derivation);
            }
        }
    }
}

/// `(<name> - <instant>) // <timedelta>` read as a temporal offset
/// derivation — A6's own offset spelling. `<name>` must be bound to a
/// window-flowing instant, `<instant>` to a concrete construction whose
/// exact microsecond count this reader can name, and `<timedelta>` to a
/// duration with a known nonzero microsecond total. Anything else
/// declines: the ledger records only derivations it can invert exactly.
fn temporal_offset_derivation(
    value_expr: &Expr,
    environment: &Environment,
    kernel: &Arc<refined_kernel::kernel_interface::RefinedTSKernel>,
) -> Option<crate::env::TemporalOffsetDerivation> {
    let Expr::BinOp(division) = value_expr else {
        return None;
    };
    if division.op != ruff_python_ast::Operator::FloorDiv {
        return None;
    }
    let unit_microseconds = crate::expressions::timedelta_microseconds_of_expression(
        division.right.as_ref(),
        environment,
        kernel,
    )?;
    if unit_microseconds == 0 {
        return None;
    }
    let Expr::BinOp(difference) = division.left.as_ref() else {
        return None;
    };
    if difference.op != ruff_python_ast::Operator::Sub {
        return None;
    }
    let Expr::Name(instant_name) = difference.left.as_ref() else {
        return None;
    };
    let flowing = environment.read(instant_name.id.as_str())?;
    if flowing.source != "temporal_flow" {
        return None;
    }
    let origin_microseconds =
        crate::expressions::exact_instant_microseconds_of_expression(difference.right.as_ref(), environment, kernel)?;
    Some(crate::env::TemporalOffsetDerivation {
        instant_name: instant_name.id.as_str().to_owned(),
        origin_microseconds,
        unit_microseconds,
    })
}
