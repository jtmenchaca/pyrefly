/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The statement walk that judges values against stated sets. Every
//! body gets its own environment, seeded with the names the body
//! itself binds (so a module-level alias name goes dark inside a body
//! that rebinds it — Python's whole-body scoping rule). The walk
//! dispatches on every statement form; a construct it cannot yet walk
//! is the body's blocker — recorded once, as an RTS7002 finding naming
//! the construct in place — and the walk keeps going conservatively
//! (forgetting names it cannot account for) so later determinable rows
//! still judge. The membership question `x ∈ A` always goes to the
//! proved kernel (memberB_iff), never decided host-side; every
//! value-against-set judgment routes through `assignability::judge` so
//! fire wording and undetermined sentences stay uniform across every
//! sink (AnnAssign, plain Assign, return, aug-assign, class field). A
//! write sink that Fires never binds the refused value — `judge_and_bind`
//! is the refused-write law: the slot keeps its DECLARED SET afterward,
//! so a later read judges the declaration against itself (always
//! silent) rather than firing a second time for the same refusal.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use refined_domain::abstract_value::{known_set, unknown, AbstractValue, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::{
    Alias, ExceptHandler, Expr, ModModule, Parameters, Stmt, StmtAnnAssign, StmtAssign,
    StmtAugAssign, StmtClassDef, StmtFunctionDef, StmtIf, StmtReturn, WithItem,
};
use ruff_text_size::{Ranged, TextRange};

use crate::refinedpy::assignability::{judge, Verdict};
use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::{binary_arithmetic_value, evaluate_expression};
use crate::refinedpy::narrowing::assume;
use crate::refinedpy::surface::{compile_aliases, surface_imports};
use crate::refinedpy::typereading::{declared_refinement, DeclaredRefinement};

/// One refinement finding: the range it anchors to, the RTS code, and
/// the rendered message.
pub struct Finding {
    pub range: TextRange,
    pub code: &'static str,
    pub message: String,
}

/// The read-only facts every statement in one module walk shares: the
/// module's compiled aliases, its import identities, and the kernel
/// handle. Bundled so the walk's many recursive calls (one body's
/// `if`/class-body/function-body descent) pass one reference instead
/// of three, without hiding what each field means behind a generic
/// "options" name.
struct WalkContext<'a> {
    aliases: &'a HashMap<String, RefinedSet>,
    imports: &'a crate::refinedpy::surface::SurfaceImports,
    kernel: &'a Arc<RefinedTSKernel>,
}

/// Every finding in one module: compile the module's own aliases and
/// import identities once, then walk its statements (function bodies
/// included — rows live inside fixture functions, and each nested
/// `def` gets its own fresh body walk). Cross-module facts are a later
/// law (L8); this walk is entry-module-local.
pub fn findings_for_module(module: &ModModule, kernel: &Arc<RefinedTSKernel>) -> Vec<Finding> {
    let aliases = compile_aliases(module);
    if aliases.is_empty() {
        return Vec::new();
    }
    let imports = surface_imports(module);
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel,
    };
    let mut out = Vec::new();
    walk_body(&module.body, None, None, &context, &mut out);
    out
}

/// One body's walk: build its environment from every name it locally
/// binds, then dispatch each statement in order. `parameters` seeds a
/// function body's own parameter names into that locally-bound set (a
/// parameter shadows an outer alias exactly as a rebinding would) and,
/// where a parameter's annotation reads, seeds its INITIAL value too
/// (assignability.rs's DeclaredRefinement, at TrustSpec — an annotation
/// states the developer's claim, not an execution-proved fact);
/// `parameters: None` is the module's own top-level body, which has
/// neither. `return_refinement` is the enclosing function's own
/// `-> Annotation` read through `declared_refinement`, threaded down so
/// every `return value` in this body (not in a nested `def`, which
/// reads its own) judges against it; `None` when the function has no
/// return annotation, or this is the module body — ordinary Python,
/// nothing to judge returns against. `blocked` tracks whether this
/// body has already recorded its one RTS7002 — set true the moment the
/// first unwalkable construct is seen, and never reset within this
/// body.
fn walk_body(
    body: &[Stmt],
    parameters: Option<&Parameters>,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    out: &mut Vec<Finding>,
) {
    let mut locally_bound = locally_bound_names(body);
    if let Some(parameters) = parameters {
        collect_parameter_names(parameters, &mut locally_bound);
    }
    let mut environment = Environment::new(locally_bound);
    if let Some(parameters) = parameters {
        seed_parameters(parameters, context, &mut environment);
    }
    let mut blocked = false;
    let mut aug_assign_refinements: HashMap<String, DeclaredRefinement> = HashMap::new();
    for stmt in body {
        walk_statement(
            stmt,
            return_refinement,
            context,
            &mut environment,
            &mut aug_assign_refinements,
            &mut blocked,
            out,
        );
    }
}

/// A function body's own parameters whose annotation reads through
/// `declared_refinement`: bind the name to a set-kind AbstractValue
/// holding the declared set (`known_set`, TrustSpec — the annotation is
/// read, not proved by execution). A parameter whose annotation states
/// nothing this table reads is left unbound (ordinary Python, no seed).
fn seed_parameters(parameters: &Parameters, context: &WalkContext, environment: &mut Environment) {
    for parameter in parameters
        .posonlyargs
        .iter()
        .chain(parameters.args.iter())
        .chain(parameters.kwonlyargs.iter())
    {
        let Some(annotation) = parameter.parameter.annotation.as_deref() else {
            continue;
        };
        let Some(declared) =
            declared_refinement(annotation, context.aliases, context.imports, environment)
        else {
            continue;
        };
        let seeded = known_set(declared.set, None, TrustSpec, SetKindTag::None);
        environment.bind(parameter.parameter.name.id.as_str(), seeded);
    }
}

/// Record this body's one blocker, if it has not already recorded one.
/// Every later call in the same body is a no-op — the FIRST blocker
/// wins, and the walk still keeps going conservatively afterward.
fn record_blocker(blocked: &mut bool, range: TextRange, sentence: String, out: &mut Vec<Finding>) {
    if *blocked {
        return;
    }
    *blocked = true;
    out.push(Finding {
        range,
        code: "RTS7002",
        message: sentence,
    });
}

/// One statement, dispatched by syntactic form. Determinable forms
/// judge or bind/forget as they can; the first form this walk cannot
/// handle names itself as the body's blocker and the walk moves on
/// conservatively. `aug_assign_refinements` is this body's own
/// AnnAssign-recorded refinement table (`x: Age = …` records `x ↦ Age`
/// here) — read back when a later `x += …` needs a set to judge
/// against.
fn walk_statement(
    stmt: &Stmt,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    match stmt {
        Stmt::AnnAssign(assign) => {
            walk_ann_assign(assign, context, environment, aug_assign_refinements, blocked, out);
        }
        Stmt::Assign(assign) => {
            walk_assign(assign, context, environment, aug_assign_refinements, out);
        }
        Stmt::AugAssign(assign) => {
            walk_aug_assign(assign, context, environment, aug_assign_refinements, blocked, out);
        }
        Stmt::Expr(expr_stmt) => {
            evaluate_expression(expr_stmt.value.as_ref(), environment, context.kernel);
        }
        Stmt::Pass(_) => {}
        Stmt::Return(ret) => {
            walk_return(ret, return_refinement, context, environment, blocked, out);
        }
        Stmt::FunctionDef(def) => {
            walk_function_def(def, context, out);
        }
        Stmt::ClassDef(def) => {
            walk_class_def(def, context, out);
        }
        // `del a, b, …` (simple_stmts.rst, "The `del` statement":
        // "Deletion of a target list recursively deletes each target,
        // from left to right") — every named target forgets what the
        // walk knew; no judgment and no blocker either way.
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                forget_target_names(target, environment);
            }
        }
        // `assert test[, msg]`: narrows the environment by the test
        // being true and keeps walking — the same `assume` seam an
        // `if` arm's truthy fork uses. A failing assert raises at
        // runtime and this walk has no exception channel, so only the
        // success continuation is modeled.
        Stmt::Assert(assert) => {
            *environment = assume(assert.test.as_ref(), environment.fork(), context.kernel, true);
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                evaluate_expression(exc, environment, context.kernel);
            }
        }
        Stmt::Global(_) | Stmt::Nonlocal(_) => {}
        Stmt::If(if_stmt) => {
            walk_if(
                if_stmt,
                return_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        // Imports and type aliases are consumed statements, not
        // blockers: the surface reads them (import identities, the
        // alias table) before the walk starts. Walking one only
        // forgets the names it binds, so a stale fact never survives.
        Stmt::Import(import) => {
            for alias in &import.names {
                let local = alias.asname.as_ref().unwrap_or(&alias.name);
                environment.forget(local.id.as_str());
            }
        }
        Stmt::ImportFrom(import) => {
            for alias in &import.names {
                let local = alias.asname.as_ref().unwrap_or(&alias.name);
                environment.forget(local.id.as_str());
            }
        }
        Stmt::TypeAlias(alias) => {
            if let Expr::Name(name) = alias.name.as_ref() {
                environment.forget(name.id.as_str());
            }
        }
        _ => {
            record_blocker(
                blocked,
                stmt.range(),
                format!("{} is not yet walked", statement_kind_name(stmt)),
                out,
            );
        }
    }
}

/// `return value` against the enclosing function's own `-> Annotation`.
/// No annotation (`return_refinement` is `None`) means ordinary Python
/// — nothing judges, matching the mission's "no return annotation → no
/// judging." A bare `return`/`return None` carries no value expression
/// and judges nothing either. `Verdict::Fire` records an RTS7001 at the
/// returned expression's own range; `Undetermined` becomes this body's
/// blocker candidate (never overriding an earlier blocker — the FIRST
/// blocker wins, same as every other sink).
fn walk_return(
    ret: &StmtReturn,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let Some(value_expr) = ret.value.as_deref() else {
        return;
    };
    let value = evaluate_expression(value_expr, environment, context.kernel);
    let Some(declared) = return_refinement else {
        return;
    };
    match judge(&value, declared, context.kernel) {
        Verdict::Fire(message) => out.push(Finding {
            range: value_expr.range(),
            code: "RTS7001",
            message,
        }),
        Verdict::Silent => {}
        Verdict::Undetermined(sentence) => {
            record_blocker(blocked, value_expr.range(), sentence, out);
        }
    }
}

/// A nested `def`: reads its own `-> Annotation` (if any) through
/// `declared_refinement` against the OUTER environment (a return
/// annotation naming a module-level alias resolves the same way any
/// other annotation does; a locally-rebound name here states nothing,
/// same rule as every other annotation read) and walks its body fresh.
fn walk_function_def(def: &StmtFunctionDef, context: &WalkContext, out: &mut Vec<Finding>) {
    let outer_environment = Environment::new(HashSet::new());
    let return_refinement = def.returns.as_deref().and_then(|annotation| {
        declared_refinement(annotation, context.aliases, context.imports, &outer_environment)
    });
    walk_body(
        &def.body,
        Some(def.parameters.as_ref()),
        return_refinement.as_ref(),
        context,
        out,
    );
}

/// A class body: walked as its own body (its own locally-bound prepass,
/// its own environment) — a class-level `AnnAssign` field judges
/// exactly like a module- or function-level one, and a `def` inside the
/// class body recurses as an ordinary function body through
/// `walk_statement`'s own `Stmt::FunctionDef` arm. A class body has no
/// enclosing function, so it carries no return refinement of its own
/// (compound_stmts.rst, "Class definitions": the class body executes
/// in a new namespace with no relation to a function's own scope).
fn walk_class_def(def: &StmtClassDef, context: &WalkContext, out: &mut Vec<Finding>) {
    walk_body(&def.body, None, None, context, out);
}

/// `if test: body [elif test: body ...] [else: body]`
/// (compound_stmts.rst, "The `if` statement"): each arm forks the
/// incoming environment, narrows it by `assume` (the concurrent
/// narrowing unit; today's passthrough default is sound — every arm's
/// fork simply carries the unnarrowed environment forward, which is
/// conservative, never wrong), and walks that arm's body. `elif` is
/// read as CPython reads it: nested `if` in the `else` slot, arm by
/// arm, left to right. An arm whose LAST statement is a bare
/// `return`/`raise` does not rejoin — that arm's fall-through state is
/// unreachable, so only surviving arms' environments join. When every
/// arm falls through this way, the post-if environment is the incoming
/// one (nothing narrowed survives to describe unreachable code).
fn walk_if(
    if_stmt: &StmtIf,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let mut arms: Vec<(Option<&Expr>, &[Stmt])> = Vec::new();
    arms.push((Some(if_stmt.test.as_ref()), if_stmt.body.as_slice()));
    for clause in &if_stmt.elif_else_clauses {
        arms.push((clause.test.as_ref(), clause.body.as_slice()));
    }

    let mut surviving: Vec<Environment> = Vec::new();
    for (test, body) in &arms {
        let mut arm_environment = environment.fork();
        if let Some(test) = test {
            arm_environment = assume(test, arm_environment, context.kernel, true);
        }
        for stmt in *body {
            walk_statement(
                stmt,
                return_refinement,
                context,
                &mut arm_environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        if !arm_terminates(body) {
            surviving.push(arm_environment);
        }
    }

    // `if` with no `else`/final catch-all arm falls through to the
    // post-if point unnarrowed whenever the condition is false — that
    // implicit empty arm always survives.
    let has_catch_all = arms.last().map(|(test, _)| test.is_none()).unwrap_or(false);
    if !has_catch_all {
        surviving.push(environment.fork());
    }

    *environment = match surviving.len() {
        0 => environment.fork(),
        1 => surviving.into_iter().next().unwrap(),
        _ => {
            let mut joined = surviving.remove(0);
            for arm in surviving {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    };
}

/// Whether a body's last statement is a bare `return`/`raise` — an arm
/// ending this way never falls through to the post-if point, so its
/// environment describes only unreachable code and must not join.
fn arm_terminates(body: &[Stmt]) -> bool {
    matches!(body.last(), Some(Stmt::Return(_)) | Some(Stmt::Raise(_)))
}

/// The refused-write law, shared by every write sink that can fire
/// (AnnAssign, Assign, AugAssign): judges `value` against `declared`,
/// pushes a Fire finding at `fire_range` when it fires, and binds
/// `name` in `environment` according to the verdict. A Fire does NOT
/// bind the refused value — the write is refused, so the slot keeps
/// its DECLARED SET (`known_set`, TrustSpec — the same construction
/// `seed_parameters` uses) for onward flow: `a = 200` under `a: Age`
/// fires once, here, and a later `return a` under `-> Age` reads the
/// declared set, which is silent against itself (set ⊆ set), never a
/// second fire for the same refused write. Silent binds the evaluated
/// value as today; Undetermined forgets (a stale fact must not survive
/// an unjudged write) and is returned so the caller may adopt it as
/// this body's blocker.
fn judge_and_bind(
    name: &str,
    value: AbstractValue,
    declared: &DeclaredRefinement,
    fire_range: TextRange,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> Option<String> {
    match judge(&value, declared, context.kernel) {
        Verdict::Fire(message) => {
            out.push(Finding {
                range: fire_range,
                code: "RTS7001",
                message,
            });
            let refused_slot = known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None);
            environment.bind(name, refused_slot);
            None
        }
        Verdict::Silent => {
            environment.bind(name, value);
            None
        }
        Verdict::Undetermined(sentence) => {
            environment.forget(name);
            Some(sentence)
        }
    }
}

/// `x op= v` on a plain name: the new value is `binary_arithmetic_value`
/// (expressions.rs's shared arithmetic transfer — the same one ordinary
/// `x = x op v` rows use, so the two forms agree exactly) folding the
/// target's CURRENT value with the evaluated RHS. Judges against `x`'s
/// own recorded refinement (this body's `x: Age = …` AnnAssign, if any)
/// through the shared refused-write law — `Fire` anchors to the WHOLE
/// statement's range (there is no separate "value expression" the way
/// AnnAssign has one; the fired value is the folded result, not a
/// sub-expression of the source). A name with no recorded refinement
/// binds the folded value directly, same as before. An
/// attribute/subscript aug-target (`obj.x += 1`, `a[0] += 1`) stays
/// this body's blocker — this walk does not track object/element state
/// through an aug-target that is not a bare name.
fn walk_aug_assign(
    assign: &StmtAugAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let Expr::Name(name) = assign.target.as_ref() else {
        record_blocker(
            blocked,
            assign.range(),
            "an augmented assignment to a non-name target is not yet walked".to_owned(),
            out,
        );
        return;
    };
    let current = environment.read(name.id.as_str()).cloned().unwrap_or_else(unknown);
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value(assign.op, &current, &operand);

    match aug_assign_refinements.get(name.id.as_str()) {
        // An Undetermined verdict already forgets the name inside
        // judge_and_bind; a bare-name aug-target is not itself a
        // blocker candidate (blockers here are scoped to non-name
        // targets only, handled above), so the sentence is discarded.
        Some(declared) => {
            let declared = declared.clone();
            judge_and_bind(name.id.as_str(), updated, &declared, assign.range(), context, environment, out);
        }
        None => environment.bind(name.id.as_str(), updated),
    }
}

/// `x: Annotation = value` — the judging channel. Reads the annotation
/// through `declared_refinement` first (the general table); when that
/// states nothing, the direct alias-Name path still runs so existing
/// fires do not regress. An annotation whose Name is an alias but is
/// locally rebound in this body states nothing — that is a blocker
/// candidate naming the rebinding, never a judged 7001. A successfully
/// read declaration is also recorded into `aug_assign_refinements`
/// (keyed on the target's plain name) so a later `x op= v` or plain
/// `x = v` in this same body can judge against it too — recorded even
/// for a VALUE-LESS declaration (`a: Age` alone): simple_stmts.rst,
/// "Annotated assignment statements" states annotated assignment as
/// "the combination, in a single statement, of a variable or attribute
/// annotation AND AN OPTIONAL assignment statement" — the `=` clause is
/// its own separate, optional part of the grammar
/// (`annotated_assignment_stmt: augtarget ":" expression ["=" ...]`),
/// so `a: Age` alone declares the slot's refinement without binding the
/// name to anything, and the slot's declared refinement still exists
/// for later reads/writes to judge against even though nothing binds
/// yet.
fn walk_ann_assign(
    assign: &StmtAnnAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let declared =
        declared_refinement(assign.annotation.as_ref(), context.aliases, context.imports, environment)
            .or_else(|| direct_alias_annotation(assign.annotation.as_ref(), context.aliases, environment));

    let Some(declared) = declared else {
        // An alias name shadowed by a local rebinding is the specific,
        // nameable reason nothing was read here; anything else falls
        // through as a plain "annotation not read" case and is not
        // this body's business to block on (a target lacking a
        // refinement-carrying annotation is ordinary Python).
        if let Expr::Name(annotation_name) = assign.annotation.as_ref()
            && context.aliases.contains_key(annotation_name.id.as_str())
            && !environment.alias_is_visible(annotation_name.id.as_str())
        {
            record_blocker(
                blocked,
                assign.annotation.range(),
                format!(
                    "the annotation's name '{}' is rebound in this body",
                    annotation_name.id.as_str()
                ),
                out,
            );
        }
        bind_target_from_value_expr(assign.target.as_ref(), assign.value.as_deref(), environment, context.kernel);
        return;
    };

    if let Expr::Name(target_name) = assign.target.as_ref() {
        aug_assign_refinements.insert(target_name.id.as_str().to_owned(), declared.clone());
    }

    let Some(value_expr) = assign.value.as_deref() else {
        // `a: Age` alone — the declaration is recorded above; CPython
        // evaluates the annotation but does not bind the name, so
        // nothing judges and nothing binds here.
        bind_target_from_value_expr(assign.target.as_ref(), None, environment, context.kernel);
        return;
    };

    let value = evaluate_expression(value_expr, environment, context.kernel);

    let Expr::Name(target_name) = assign.target.as_ref() else {
        // A non-name AnnAssign target (rare in practice — e.g. an
        // attribute/subscript annotated write): judge for the Fire, but
        // there is no environment slot to rebind under the refused-write
        // law, so fall back to the old bind-the-RHS path.
        match judge(&value, &declared, context.kernel) {
            Verdict::Fire(message) => out.push(Finding {
                range: value_expr.range(),
                code: "RTS7001",
                message,
            }),
            Verdict::Silent => {}
            Verdict::Undetermined(sentence) => {
                record_blocker(blocked, value_expr.range(), sentence, out);
            }
        }
        bind_target_from_value_expr(assign.target.as_ref(), Some(value_expr), environment, context.kernel);
        return;
    };

    if let Some(sentence) =
        judge_and_bind(target_name.id.as_str(), value, &declared, value_expr.range(), context, environment, out)
    {
        record_blocker(blocked, value_expr.range(), sentence, out);
    }
}

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
fn walk_assign(
    assign: &StmtAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    let value = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
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
            _ => bind_or_forget_target(target, &value, environment),
        }
    }
}

/// The pre-typereading path: an annotation that is bare `Name` naming
/// a compiled alias, visible in this body (not locally rebound). Kept
/// alongside `declared_refinement` so the two existing tests' fires
/// keep firing before the general annotation table recognizes this
/// same shape itself.
fn direct_alias_annotation(
    annotation: &Expr,
    aliases: &HashMap<String, RefinedSet>,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    let Expr::Name(name) = annotation else {
        return None;
    };
    if !environment.alias_is_visible(name.id.as_str()) {
        return None;
    }
    let set = aliases.get(name.id.as_str())?;
    Some(DeclaredRefinement {
        set: set.clone(),
        spelling: name.id.as_str().to_owned(),
    })
}

/// After an AnnAssign is judged (or declined), the target still binds:
/// a known value if the RHS was readable, forgotten otherwise so a
/// stale fact never survives an unread write.
fn bind_target_from_value_expr(
    target: &Expr,
    value_expr: Option<&Expr>,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) {
    let Expr::Name(name) = target else {
        return;
    };
    match value_expr {
        Some(expr) => {
            let value = evaluate_expression(expr, environment, kernel);
            environment.bind(name.id.as_str(), value);
        }
        None => environment.forget(name.id.as_str()),
    }
}

/// A plain `Assign` target: binds a plain name to the evaluated value;
/// tuple/list/starred targets forget every name they touch (the walk
/// does not yet destructure a value across positions), and anything
/// else is left alone (an attribute/subscript target writes through
/// another object, not a name this environment tracks).
fn bind_or_forget_target(target: &Expr, value: &AbstractValue, environment: &mut Environment) {
    match target {
        Expr::Name(name) => environment.bind(name.id.as_str(), value.clone()),
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                forget_target_names(element, environment);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                forget_target_names(element, environment);
            }
        }
        Expr::Starred(starred) => forget_target_names(starred.value.as_ref(), environment),
        _ => {}
    }
}

/// Forget every plain name reachable inside a target expression
/// (nested tuple/list/starred targets included) — used where the walk
/// cannot state what value lands in each position, and by `del` (every
/// deleted name is simply forgotten).
fn forget_target_names(target: &Expr, environment: &mut Environment) {
    match target {
        Expr::Name(name) => environment.forget(name.id.as_str()),
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                forget_target_names(element, environment);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                forget_target_names(element, environment);
            }
        }
        Expr::Starred(starred) => forget_target_names(starred.value.as_ref(), environment),
        _ => {}
    }
}

/// A plain prose name for a statement kind, for the blocker sentence —
/// e.g. "a while statement is not yet walked". Never a category label:
/// each name is spoken in place, in the sentence naming this one body's
/// first blocker.
fn statement_kind_name(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::TypeAlias(_) => "a nested type alias statement",
        Stmt::For(_) => "a for statement",
        Stmt::While(_) => "a while statement",
        Stmt::With(_) => "a with statement",
        Stmt::Match(_) => "a match statement",
        Stmt::Try(_) => "a try statement",
        Stmt::Import(_) => "an import statement",
        Stmt::ImportFrom(_) => "an import-from statement",
        Stmt::Break(_) => "a break statement",
        Stmt::Continue(_) => "a continue statement",
        Stmt::IpyEscapeCommand(_) => "an IPython escape command",
        // Handled elsewhere in walk_statement's match — never reaches here.
        Stmt::AnnAssign(_)
        | Stmt::Assign(_)
        | Stmt::AugAssign(_)
        | Stmt::Expr(_)
        | Stmt::Pass(_)
        | Stmt::Return(_)
        | Stmt::FunctionDef(_)
        | Stmt::ClassDef(_)
        | Stmt::Delete(_)
        | Stmt::If(_)
        | Stmt::Assert(_)
        | Stmt::Raise(_)
        | Stmt::Global(_)
        | Stmt::Nonlocal(_) => "a statement",
    }
}

/// Every name this body binds anywhere, at any nesting depth of its
/// OWN statements (not inside a nested `def`/`class` body, which has
/// its own scope) — assignment/for/with-as/except targets, walrus
/// targets in any expression the body evaluates, parameters, and
/// import aliases. A name declared `global`/`nonlocal` is excluded:
/// Python's own rule is that such a name is never local to this body,
/// so a module-level alias sharing its spelling stays visible.
fn locally_bound_names(body: &[Stmt]) -> HashSet<String> {
    let mut bound = HashSet::new();
    let mut excluded = HashSet::new();
    collect_bound_names(body, &mut bound, &mut excluded);
    for name in &excluded {
        bound.remove(name);
    }
    bound
}

fn collect_bound_names(body: &[Stmt], bound: &mut HashSet<String>, excluded: &mut HashSet<String>) {
    for stmt in body {
        collect_bound_names_stmt(stmt, bound, excluded);
    }
}

fn collect_bound_names_stmt(stmt: &Stmt, bound: &mut HashSet<String>, excluded: &mut HashSet<String>) {
    match stmt {
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                collect_target_names(target, bound);
            }
            collect_walrus_names(assign.value.as_ref(), bound);
        }
        Stmt::AnnAssign(assign) => {
            collect_target_names(assign.target.as_ref(), bound);
            if let Some(value) = assign.value.as_deref() {
                collect_walrus_names(value, bound);
            }
        }
        Stmt::AugAssign(assign) => {
            collect_target_names(assign.target.as_ref(), bound);
            collect_walrus_names(assign.value.as_ref(), bound);
        }
        Stmt::For(for_stmt) => {
            collect_target_names(for_stmt.target.as_ref(), bound);
            collect_walrus_names(for_stmt.iter.as_ref(), bound);
            collect_bound_names(&for_stmt.body, bound, excluded);
            collect_bound_names(&for_stmt.orelse, bound, excluded);
        }
        Stmt::While(while_stmt) => {
            collect_walrus_names(while_stmt.test.as_ref(), bound);
            collect_bound_names(&while_stmt.body, bound, excluded);
            collect_bound_names(&while_stmt.orelse, bound, excluded);
        }
        Stmt::If(if_stmt) => {
            collect_walrus_names(if_stmt.test.as_ref(), bound);
            collect_bound_names(&if_stmt.body, bound, excluded);
            for clause in &if_stmt.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    collect_walrus_names(test, bound);
                }
                collect_bound_names(&clause.body, bound, excluded);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                collect_with_item_names(item, bound);
            }
            collect_bound_names(&with_stmt.body, bound, excluded);
        }
        Stmt::Try(try_stmt) => {
            collect_bound_names(&try_stmt.body, bound, excluded);
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                if let Some(name) = handler.name.as_ref() {
                    bound.insert(name.id.as_str().to_owned());
                }
                collect_bound_names(&handler.body, bound, excluded);
            }
            collect_bound_names(&try_stmt.orelse, bound, excluded);
            collect_bound_names(&try_stmt.finalbody, bound, excluded);
        }
        Stmt::FunctionDef(def) => {
            bound.insert(def.name.id.as_str().to_owned());
            // the def's OWN body is a separate scope — its parameters
            // and locals do not leak into this body's bound set
        }
        Stmt::ClassDef(def) => {
            bound.insert(def.name.id.as_str().to_owned());
        }
        Stmt::Import(import) => {
            for alias in &import.names {
                bound.insert(imported_local_name(alias));
            }
        }
        Stmt::ImportFrom(import) => {
            for alias in &import.names {
                bound.insert(imported_local_name(alias));
            }
        }
        Stmt::Global(global) => {
            for name in &global.names {
                excluded.insert(name.id.as_str().to_owned());
            }
        }
        Stmt::Nonlocal(nonlocal) => {
            for name in &nonlocal.names {
                excluded.insert(name.id.as_str().to_owned());
            }
        }
        Stmt::Expr(expr_stmt) => collect_walrus_names(expr_stmt.value.as_ref(), bound),
        Stmt::Return(ret) => {
            if let Some(value) = ret.value.as_deref() {
                collect_walrus_names(value, bound);
            }
        }
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                collect_walrus_names(target, bound);
            }
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                collect_walrus_names(exc, bound);
            }
            if let Some(cause) = raise.cause.as_deref() {
                collect_walrus_names(cause, bound);
            }
        }
        Stmt::Assert(assert) => {
            collect_walrus_names(assert.test.as_ref(), bound);
            if let Some(msg) = assert.msg.as_deref() {
                collect_walrus_names(msg, bound);
            }
        }
        Stmt::Match(match_stmt) => {
            collect_walrus_names(match_stmt.subject.as_ref(), bound);
            for case in &match_stmt.cases {
                collect_bound_names(&case.body, bound, excluded);
            }
        }
        Stmt::TypeAlias(_) | Stmt::Pass(_) | Stmt::Break(_) | Stmt::Continue(_)
        | Stmt::IpyEscapeCommand(_) => {}
    }
}

/// A `for`/`with`-as/assignment target's bound names, including nested
/// tuple/list/starred forms.
fn collect_target_names(target: &Expr, bound: &mut HashSet<String>) {
    match target {
        Expr::Name(name) => {
            bound.insert(name.id.as_str().to_owned());
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_target_names(element, bound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_target_names(element, bound);
            }
        }
        Expr::Starred(starred) => collect_target_names(starred.value.as_ref(), bound),
        _ => {}
    }
}

fn collect_with_item_names(item: &WithItem, bound: &mut HashSet<String>) {
    collect_walrus_names(&item.context_expr, bound);
    if let Some(vars) = item.optional_vars.as_deref() {
        collect_target_names(vars, bound);
    }
}

fn imported_local_name(alias: &Alias) -> String {
    match alias.asname.as_ref() {
        Some(asname) => asname.id.as_str().to_owned(),
        None => {
            // `import a.b.c` binds the top-level name `a` in the local
            // scope; `import a.b.c as x` (the asname arm above) binds
            // `x` instead. A plain `from` import's alias.name is
            // already the single name being bound.
            let full = alias.name.id.as_str();
            full.split('.').next().unwrap_or(full).to_owned()
        }
    }
}

/// Walrus (`:=`) targets anywhere inside an expression the body
/// evaluates — a walrus binds its target into the ENCLOSING scope,
/// wherever it sits (a comprehension, a condition, a call argument).
fn collect_walrus_names(expr: &Expr, bound: &mut HashSet<String>) {
    match expr {
        Expr::Named(named) => {
            collect_target_names(named.target.as_ref(), bound);
            collect_walrus_names(named.value.as_ref(), bound);
        }
        Expr::BoolOp(op) => {
            for value in &op.values {
                collect_walrus_names(value, bound);
            }
        }
        Expr::BinOp(op) => {
            collect_walrus_names(op.left.as_ref(), bound);
            collect_walrus_names(op.right.as_ref(), bound);
        }
        Expr::UnaryOp(op) => collect_walrus_names(op.operand.as_ref(), bound),
        // the lambda's OWN body is a separate scope; a walrus inside it
        // does not bind here
        Expr::Lambda(_) => {}
        Expr::If(if_expr) => {
            collect_walrus_names(if_expr.test.as_ref(), bound);
            collect_walrus_names(if_expr.body.as_ref(), bound);
            collect_walrus_names(if_expr.orelse.as_ref(), bound);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_walrus_names(element, bound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_walrus_names(element, bound);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                collect_walrus_names(element, bound);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    collect_walrus_names(key, bound);
                }
                collect_walrus_names(&item.value, bound);
            }
        }
        Expr::Call(call) => {
            collect_walrus_names(call.func.as_ref(), bound);
            for arg in &call.arguments.args {
                collect_walrus_names(arg, bound);
            }
            for keyword in &call.arguments.keywords {
                collect_walrus_names(&keyword.value, bound);
            }
        }
        Expr::Compare(compare) => {
            collect_walrus_names(compare.left.as_ref(), bound);
            for comparator in &compare.comparators {
                collect_walrus_names(comparator, bound);
            }
        }
        Expr::Attribute(attribute) => collect_walrus_names(attribute.value.as_ref(), bound),
        Expr::Subscript(subscript) => {
            collect_walrus_names(subscript.value.as_ref(), bound);
            collect_walrus_names(subscript.slice.as_ref(), bound);
        }
        Expr::Starred(starred) => collect_walrus_names(starred.value.as_ref(), bound),
        Expr::Slice(slice) => {
            if let Some(lower) = slice.lower.as_deref() {
                collect_walrus_names(lower, bound);
            }
            if let Some(upper) = slice.upper.as_deref() {
                collect_walrus_names(upper, bound);
            }
            if let Some(step) = slice.step.as_deref() {
                collect_walrus_names(step, bound);
            }
        }
        Expr::FString(fstring) => {
            // `.elements()` already flattens every part (single or
            // implicitly-concatenated) down to each part's own
            // elements, literal parts skipped.
            for element in fstring.value.elements() {
                if let Some(interpolation) = element.as_interpolation() {
                    collect_walrus_names(interpolation.expression.as_ref(), bound);
                }
            }
        }
        Expr::Await(inner) => collect_walrus_names(inner.value.as_ref(), bound),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                collect_walrus_names(value, bound);
            }
        }
        Expr::YieldFrom(inner) => collect_walrus_names(inner.value.as_ref(), bound),
        // Comprehensions (ListComp/SetComp/DictComp/Generator) introduce
        // their own scope for their loop variables — a walrus INSIDE the
        // comprehension's element/condition still targets the ENCLOSING
        // scope per PEP 572, but that expression-walking depth is not
        // built in this wave; left unwalked rather than guessed.
        _ => {}
    }
}

/// A body's function-parameter names — every kind (positional-only,
/// normal, `*args`, keyword-only, `**kwargs`).
fn collect_parameter_names(parameters: &Parameters, bound: &mut HashSet<String>) {
    for parameter in parameters.posonlyargs.iter().chain(parameters.args.iter()) {
        bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    for parameter in &parameters.kwonlyargs {
        bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    if let Some(vararg) = parameters.vararg.as_ref() {
        bound.insert(vararg.name.id.as_str().to_owned());
    }
    if let Some(kwarg) = parameters.kwarg.as_ref() {
        bound.insert(kwarg.name.id.as_str().to_owned());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};

    fn parsed(source: &str) -> ModModule {
        ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax()
    }

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    #[test]
    fn an_out_of_set_literal_fires_and_an_in_set_literal_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    good: Age = 42\n",
            "    over: Age = 200\n",
            "    fractional: Age = 7.5\n",
            "    negative: Age = -1\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let messages: Vec<&str> = findings.iter().map(|f| f.message.as_str()).collect();
        assert_eq!(
            findings.len(),
            3,
            "want fires for 200, 7.5, and -1 only: {messages:?}"
        );
        assert!(findings.iter().all(|f| f.code == "RTS7001"));
        assert!(messages[0].contains("'200'"), "{messages:?}");
        assert!(messages[1].contains("'7.5'"), "{messages:?}");
        assert!(messages[2].contains("'-1'"), "{messages:?}");
    }

    #[test]
    fn an_alias_the_table_cannot_lower_declines_whole() {
        let Some(kernel) = loaded_kernel() else { return };
        // json_schema_extra is not on the inert list and not a bound —
        // the alias refuses, so neither line judges.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Odd = Annotated[int, Field(ge=0, json_schema_extra={})]\n",
            "def rows() -> None:\n",
            "    fine: Odd = 5\n",
            "    wild: Odd = -200\n",
        ));
        assert!(findings_for_module(&module, &kernel).is_empty());
    }

    #[test]
    fn a_body_that_rebinds_the_alias_name_blocks_instead_of_judging() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    Age = 5\n",
            "    x: Age = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "a rebound alias name must never judge: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(blockers.len(), 1, "want exactly one blocker: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(
            blockers[0].message.contains("rebound"),
            "{}",
            blockers[0].message
        );
    }

    #[test]
    fn one_blocker_and_the_judged_fire_both_land_in_the_same_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    while True:\n",
            "        pass\n",
            "    over: Age = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            blockers.len(),
            1,
            "want exactly one blocker (the while): {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(
            blockers[0].message.contains("while"),
            "{}",
            blockers[0].message
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "want the judgeable AnnAssign to still fire after the blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_body_never_records_more_than_one_blocker() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    while True:\n",
            "        pass\n",
            "    for i in range(3):\n",
            "        pass\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            blockers.len(),
            1,
            "want at most one blocker per body: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_return_out_of_the_declared_set_fires_at_the_return() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    return 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_seeded_parameter_returned_under_its_own_annotation_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(age: Age) -> Age:\n",
            "    return age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "a parameter within its own declared set must stay silent on return: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn an_if_else_join_carries_an_out_of_set_arm_into_a_judged_row() {
        let Some(kernel) = loaded_kernel() else { return };
        // one arm binds x to an in-set value, the other to an
        // out-of-set value; the join keeps both possibilities, so the
        // kernel must see the union and fire on the out-of-set member.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(flag: bool) -> None:\n",
            "    if flag:\n",
            "        x = 40\n",
            "    else:\n",
            "        x = 200\n",
            "    y: Age = x\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    }

    #[test]
    fn an_aug_assign_out_of_the_recorded_set_fires_at_the_statement() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    x: Age = 40\n",
            "    x += 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.contains("'240'"), "{}", fires[0].message);
    }

    #[test]
    fn a_class_body_out_of_set_field_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Person:\n",
            "    age: Age = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    }

    #[test]
    fn del_and_assert_bodies_record_no_blocker() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    x: Age = 40\n",
            "    assert x\n",
            "    del x\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "assert/del must record no blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_value_less_declaration_then_plain_assign_fires_at_the_assign() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    a: Age\n",
            "    a = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_chained_multi_target_assign_fires_once_per_declared_target() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    a: Age\n",
            "    b: Age\n",
            "    a = b = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            2,
            "both a and b are declared Age, so the chained refusal fires once per target: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_refused_write_keeps_the_declared_set_so_a_later_return_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    a: Age\n",
            "    a = 200\n",
            "    return a\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the write fires once; the return of the refused-but-declared slot must not fire again: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn an_undeclared_names_assign_still_binds_without_judging() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    plain = 200\n",
            "    plain = 300\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an undeclared name's assign must never judge: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }
}
