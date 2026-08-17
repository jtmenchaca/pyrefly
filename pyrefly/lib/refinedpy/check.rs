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

use refined_domain::abstract_value::{known_set, opaque_value, unknown, AbstractValue, Kind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::{
    Alias, AtomicNodeIndex, ExceptHandler, Expr, ExprSubscript, ModModule, Parameters, Stmt,
    StmtAnnAssign, StmtAssign, StmtAugAssign, StmtClassDef, StmtFunctionDef, StmtIf, StmtMatch,
    StmtReturn, StmtTry, StmtWith, WithItem,
};
use ruff_text_size::{Ranged, TextRange};

use crate::refinedpy::assignability::{judge, Verdict};
use crate::refinedpy::collection_models::{dict_with_item, list_literal_value, list_with_item, mutated_receiver};
use crate::refinedpy::cross_module::{module_surface, ModuleResolver, IMPORT_DEPTH_CAP};
use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::{binary_arithmetic_value, evaluate_expression, provable_raise};
use crate::refinedpy::function_table::{function_table, merged, FunctionTable};
use crate::refinedpy::instances::{class_table, judge_construction, ClassModel, ConstructionVerdict};
use crate::refinedpy::loops::loop_final_environment;
use crate::refinedpy::match_arms::match_taken_environment;
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
/// module's compiled aliases, its import identities, the kernel handle,
/// the module's own `def`s merged with every imported `def` (the
/// module's own definition wins on a name collision), the module's own
/// classes merged with every imported class (module-local names win the
/// same way), and every plain top-level binding this module's own
/// surface OR an import statement makes readable. Bundled so the walk's
/// many recursive calls (one body's `if`/class-body/function-body
/// descent) pass one reference instead of many, without hiding what
/// each field means behind a generic "options" name.
struct WalkContext<'a> {
    aliases: &'a HashMap<String, RefinedSet>,
    imports: &'a crate::refinedpy::surface::SurfaceImports,
    kernel: &'a Arc<RefinedTSKernel>,
    functions: Arc<FunctionTable>,
    classes: Arc<HashMap<String, ClassModel>>,
    module_bindings: HashMap<String, AbstractValue>,
}

/// Every finding in one module, resolving no imports — the LSP seam's
/// own entry point, and every existing test's. Behaves exactly as
/// before this unit: calls `findings_for_module_with_resolver` with a
/// resolver that always answers `None`, so no cross-module name ever
/// resolves and the module's own local surface is all that's readable.
pub fn findings_for_module(module: &ModModule, kernel: &Arc<RefinedTSKernel>) -> Vec<Finding> {
    let no_imports: ModuleResolver = &|_: &str| None;
    findings_for_module_with_resolver(module, no_imports, kernel)
}

/// Every finding in one module, resolving imports through `resolver`
/// (the CLI's `disk_resolver`, an LSP's own module graph, or — in a
/// test — an in-memory source map): compile the module's own aliases
/// and import identities, build its cross-module surface
/// (`cross_module::module_surface`, `IMPORT_DEPTH_CAP` hops deep) for
/// the readable functions/classes/bindings an import statement pulls
/// in, then walk its statements (function bodies included — rows live
/// inside fixture functions, and each nested `def` gets its own fresh
/// body walk). `functions`/`classes` merge the module's OWN table over
/// the imported one (`function_table::merged` — a local definition
/// shadows an imported name of the same spelling, same as the
/// module-level alias table already does).
pub fn findings_for_module_with_resolver(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
) -> Vec<Finding> {
    let aliases = compile_aliases(module);
    if aliases.is_empty() {
        return Vec::new();
    }
    let imports = surface_imports(module);
    let surface = module_surface(module, resolver, kernel, IMPORT_DEPTH_CAP);
    let own_functions = function_table(module);
    let functions = Arc::new(merged(&own_functions, surface.functions.as_ref()));
    let own_classes = class_table(module, &aliases, &imports, kernel);
    // `ClassModel` carries no `Clone` (instances.rs's own note: kept
    // minimal, no caller needed it before now), so the merge takes
    // ownership of the imported map rather than cloning it — sound here
    // because `module_surface` just built this `Arc` fresh, with no
    // other clone anywhere yet, so its strong count is exactly 1.
    let mut classes = Arc::try_unwrap(surface.classes)
        .unwrap_or_else(|_| panic!("module_surface's own Arc<classes> has no other owner yet"));
    for (name, model) in own_classes {
        classes.insert(name, model);
    }
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel,
        functions,
        classes: Arc::new(classes),
        module_bindings: surface.bindings,
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
    environment.set_functions(Arc::new(merged(&local_function_table(body), &context.functions)));
    environment.set_classes(context.classes.clone());
    // Every module-level binding (this module's own top-level constants
    // AND every import statement's resolved value) is readable here
    // UNLESS this body itself rebinds the name — a local rebinding
    // shadows the module value, the same rule `alias_is_visible` already
    // applies to the alias table.
    for (name, value) in &context.module_bindings {
        if environment.alias_is_visible(name) {
            environment.bind(name, value.clone());
        }
    }
    if let Some(parameters) = parameters {
        seed_parameters(parameters, context, &mut environment);
    }
    let mut blocked = false;
    let mut aug_assign_refinements: HashMap<String, DeclaredRefinement> = HashMap::new();
    // PROVABLY-UNBOUND READS: every name this straight-line walk has seen
    // declared by a VALUELESS AnnAssign (`x: int`) with no assignment
    // observed since. `walk_statement`'s own `If`/`For`/`While`/`Match`/
    // `With`/`Try`/blocker arms clear this wholesale the moment the walk
    // crosses anything that could bind a name on some path without this
    // loop seeing it directly, so the set only ever names a name that is
    // PROVABLY still unbound along the one path CPython actually ran.
    let mut provably_unbound: HashSet<String> = HashSet::new();
    for stmt in body {
        walk_statement(
            stmt,
            return_refinement,
            context,
            &mut environment,
            &mut aug_assign_refinements,
            &mut provably_unbound,
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
/// against. `provably_unbound` is this body's own PROVABLY-UNBOUND-READS
/// tracking set (`walk_body`'s own doc): every straight-line statement
/// form below that can OBSERVE a name being bound removes it here, and
/// every form whose execution could bind a name on SOME path this walk
/// does not track directly (a branch, a loop, a match, a with, a try, or
/// this body's own first unwalkable construct) clears the set wholesale
/// before dispatching — the conservative "any branch/loop/blocker
/// between declaration and read → no fire" rule the mission states.
fn walk_statement(
    stmt: &Stmt,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    provably_unbound: &mut HashSet<String>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    match stmt {
        Stmt::AnnAssign(assign) => {
            walk_ann_assign(
                assign,
                context,
                environment,
                aug_assign_refinements,
                provably_unbound,
                blocked,
                out,
            );
        }
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                forget_target_from_provably_unbound(target, provably_unbound);
            }
            walk_assign(assign, context, environment, aug_assign_refinements, out);
        }
        Stmt::AugAssign(assign) => {
            forget_target_from_provably_unbound(assign.target.as_ref(), provably_unbound);
            walk_aug_assign(assign, context, environment, aug_assign_refinements, blocked, out);
        }
        Stmt::Expr(expr_stmt) => {
            bind_walrus_targets(expr_stmt.value.as_ref(), context, aug_assign_refinements, environment, out);
            if !walk_mutating_call_statement(expr_stmt.value.as_ref(), context, environment, out) {
                sink_value(expr_stmt.value.as_ref(), context, environment, out);
            }
        }
        Stmt::Pass(_) => {}
        Stmt::Return(ret) => {
            walk_return(
                ret,
                return_refinement,
                context,
                aug_assign_refinements,
                provably_unbound,
                environment,
                blocked,
                out,
            );
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
        // walk knew; no judgment and no blocker either way. A deleted
        // name is UNBOUND again afterward (the same state a valueless
        // AnnAssign leaves), but this table only tracks names a valueless
        // AnnAssign itself declared — `del` on an ordinary name states
        // nothing this law reads, so `provably_unbound` is untouched.
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
            bind_walrus_targets(assert.test.as_ref(), context, aug_assign_refinements, environment, out);
            *environment = assume(assert.test.as_ref(), environment.fork(), context.kernel, true);
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                bind_walrus_targets(exc, context, aug_assign_refinements, environment, out);
                evaluate_expression(exc, environment, context.kernel);
            }
        }
        Stmt::Global(_) | Stmt::Nonlocal(_) => {}
        Stmt::If(if_stmt) => {
            provably_unbound.clear();
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
        // alias table, and — this unit — the cross-module surface) before
        // the walk starts. At MODULE TOP LEVEL, the import statement is
        // where the imported name first becomes live: `context.
        // module_bindings` already carries whatever `cross_module::
        // module_surface` resolved for it, so binding from that table
        // here (rather than merely forgetting) is what makes an imported
        // name readable at its own import site and every statement after
        // it. A name the surface did not resolve (an unresolved module,
        // a name that resolves to a function/class rather than a plain
        // value, a star import's own literal `"*"` alias) still forgets,
        // exactly as before, so no stale fact survives regardless.
        Stmt::Import(import) => {
            for alias in &import.names {
                let local = alias.asname.as_ref().unwrap_or(&alias.name);
                bind_or_forget_imported_name(local.id.as_str(), context, environment);
            }
        }
        Stmt::ImportFrom(import) => {
            for alias in &import.names {
                let local = alias.asname.as_ref().unwrap_or(&alias.name);
                bind_or_forget_imported_name(local.id.as_str(), context, environment);
            }
        }
        Stmt::TypeAlias(alias) => {
            if let Expr::Name(name) = alias.name.as_ref() {
                environment.forget(name.id.as_str());
            }
        }
        Stmt::For(_) | Stmt::While(_) => {
            provably_unbound.clear();
            walk_loop(stmt, context, environment, blocked, out);
        }
        Stmt::Match(match_stmt) => {
            provably_unbound.clear();
            walk_match(
                match_stmt,
                return_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        Stmt::With(with_stmt) => {
            provably_unbound.clear();
            walk_with(
                with_stmt,
                return_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        Stmt::Try(try_stmt) => {
            provably_unbound.clear();
            walk_try(
                try_stmt,
                return_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        _ => {
            provably_unbound.clear();
            record_blocker(
                blocked,
                stmt.range(),
                format!("{} is not yet walked", statement_kind_name(stmt)),
                out,
            );
        }
    }
}

/// Removes every bare name a (possibly destructuring) Assign/AugAssign
/// target touches from `provably_unbound` — an observed WRITE to a name
/// this table is tracking cures it, the same way `judge_and_bind`'s own
/// write-sink laws bind/forget the environment for that name. Applies
/// even to a nested tuple/list/starred target: any target position that
/// names the tracked name is itself proof CPython bound it on this path.
fn forget_target_from_provably_unbound(target: &Expr, provably_unbound: &mut HashSet<String>) {
    match target {
        Expr::Name(name) => {
            provably_unbound.remove(name.id.as_str());
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                forget_target_from_provably_unbound(element, provably_unbound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                forget_target_from_provably_unbound(element, provably_unbound);
            }
        }
        Expr::Starred(starred) => forget_target_from_provably_unbound(starred.value.as_ref(), provably_unbound),
        _ => {}
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
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    provably_unbound: &HashSet<String>,
    environment: &mut Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let Some(value_expr) = ret.value.as_deref() else {
        return;
    };
    // PROVABLY-UNBOUND READS: `return x` where `x` is STILL in
    // `provably_unbound` (a valueless AnnAssign declared it, and no
    // straight-line write since has cured it — walk_statement clears the
    // whole set the moment a branch/loop/blocker could have bound it on
    // some other path) is CPython's own UnboundLocalError at this exact
    // read (executionmodel.rst's local-variable scoping rule). Checked
    // BEFORE the ordinary sink/judge path — `environment.read` already
    // answers `None` for this name (nothing ever bound it), which would
    // otherwise fall through to a silent Undetermined rather than naming
    // the provable raise.
    if let Expr::Name(name) = value_expr {
        if provably_unbound.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none() {
            out.push(Finding {
                range: value_expr.range(),
                code: "RTS7001",
                message: format!(
                    "this read provably raises UnboundLocalError: '{}' is unbound at this point",
                    name.id.as_str()
                ),
            });
            return;
        }
    }
    bind_walrus_targets(value_expr, context, aug_assign_refinements, environment, out);
    let Some(value) = sink_value(value_expr, context, environment, out) else {
        // a provable raise already pushed its own RTS7001 at the
        // raising expression — this return never produces a value to
        // judge, since CPython never reaches the return statement's own
        // completion on this path.
        return;
    };
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

/// LOCAL DEFS: this body's own top-level `def`s (not a nested body's —
/// `function_table`'s own scan only reads `module.body`'s own top-level
/// statements, and this function reuses that exact scan by wrapping
/// `body` in a synthetic `ModModule`, the same construction
/// `cross_module.rs`'s `synthetic_module`/`rename_def` pair already uses
/// to turn an assembled def collection into a real `FunctionTable`
/// through `function_table`'s one public constructor). A `def` nested
/// inside a function/`if`/`for`/`try`/... body is otherwise invisible to
/// `context.functions` (the MODULE-level table `findings_for_module`
/// built once, before any body walk) — `nested_function_definition`'s
/// `years`/`over_years` rows, and `b-body-expressions.py`'s own local
/// defs, need their own body's defs merged in, local name winning over
/// an outer/imported name of the same spelling (`function_table::
/// merged`'s existing base-wins rule, reused unchanged here with the
/// LOCAL table as `base`).
fn local_function_table(body: &[Stmt]) -> FunctionTable {
    let local_defs: Vec<Stmt> = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::FunctionDef(def) => Some(Stmt::FunctionDef(def.clone())),
            _ => None,
        })
        .collect();
    let synthetic = ModModule {
        node_index: AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: local_defs,
    };
    function_table(&synthetic)
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
///
/// DEAD-BRANCH LAW (serves a-statements:400 —
/// `none_test_on_helper_that_never_answers_none`): for each arm that
/// carries a test, the test expression is evaluated first and read
/// through `truthiness`. A test PROVABLY FALSE (`known && !value`)
/// fires an RTS7001 at the test's own range ("this condition is
/// provably false on every run") and that arm's body is never walked —
/// it contributes no surviving environment, since CPython itself never
/// runs it. A test PROVABLY TRUE (`known && value`) walks only that
/// arm, and every LATER arm (including any final `else`) is
/// unreachable — CPython never evaluates a later test once an earlier
/// one is known true — so no later arm is walked and none contributes
/// any finding; the post-if environment is that one arm's alone, and
/// this function returns immediately. An UNKNOWN test (the ordinary
/// case) keeps exactly today's behavior: forked, narrowed, walked, and
/// joined alongside every other surviving arm. Only a provably-FALSE
/// test ever fires — a provably-true test states nothing wrong, so it
/// never does.
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
        if let Some(test) = test {
            let test_value = evaluate_expression(test, environment, context.kernel);
            // WALRUS BINDING: `if (age := 40) > 0:` binds `age` into the
            // ENCLOSING environment BEFORE any arm forks from it — CPython
            // evaluates the test (and its own walrus assignment) once,
            // regardless of which arm the truth value takes, so the bound
            // name is visible both inside the taken arm's body and after
            // the whole `if` (a-statements.py's `walrus_in_condition`).
            bind_walrus_targets(test, context, aug_assign_refinements, environment, out);
            let (truthy, known) = refined_domain::lattice_operations::truthiness(&test_value);
            if known && !truthy {
                out.push(Finding {
                    range: test.range(),
                    code: "RTS7001",
                    message: "this condition is provably false on every run".to_owned(),
                });
                continue;
            }
            if known && truthy {
                let mut arm_environment = environment.fork();
                arm_environment = assume(test, arm_environment, context.kernel, true);
                // A fresh, empty PROVABLY-UNBOUND-READS set per arm body:
                // the outer body's own set was already cleared by
                // `walk_statement`'s `Stmt::If` arm before this function was
                // called, and an arm body's own valueless AnnAssigns start
                // this fresh scan over again — sound because a name this
                // arm itself declares valueless and never assigns is still
                // exactly as provably unbound as it would be in a straight-
                // line body.
                let mut arm_provably_unbound: HashSet<String> = HashSet::new();
                for stmt in *body {
                    walk_statement(
                        stmt,
                        return_refinement,
                        context,
                        &mut arm_environment,
                        aug_assign_refinements,
                        &mut arm_provably_unbound,
                        blocked,
                        out,
                    );
                }
                *environment = if arm_terminates(body) { environment.fork() } else { arm_environment };
                return;
            }
        }
        let mut arm_environment = environment.fork();
        if let Some(test) = test {
            arm_environment = assume(test, arm_environment, context.kernel, true);
        }
        let mut arm_provably_unbound: HashSet<String> = HashSet::new();
        for stmt in *body {
            walk_statement(
                stmt,
                return_refinement,
                context,
                &mut arm_environment,
                aug_assign_refinements,
                &mut arm_provably_unbound,
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

/// `for`/`while`: `loops::loop_final_environment` concretely executes the
/// bounded shapes it recognizes (literal list/tuple/range iterables,
/// bounded counter `while`s) — `Some(env)` replaces the environment
/// outright and the statement is consumed with no blocker. `None` means
/// the shape is outside what that module can run; the walk keeps its own
/// blocker AND forgets every name the loop statement binds anywhere (its
/// target plus every name its body/orelse bind), so a stale pre-loop fact
/// never survives an unmodeled loop that may have rebound it.
fn walk_loop(
    stmt: &Stmt,
    context: &WalkContext,
    environment: &mut Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    if let Some(final_env) = loop_final_environment(stmt, environment, context.kernel) {
        *environment = final_env;
        return;
    }
    record_blocker(
        blocked,
        stmt.range(),
        format!("{} is not yet walked", statement_kind_name(stmt)),
        out,
    );
    forget_names_bound_by_stmt(stmt, environment);
}

/// `match subject: case ... case ...` (compound_stmts.rst, "The `match`
/// statement"): `match_arms::match_taken_environment` decides, for a
/// KNOWN subject, which single arm CPython would take. `Some((index,
/// arm_env))` adopts that arm's environment (already carrying any
/// capture-pattern bindings match_arms.rs made) and walks ONLY that
/// case's body statements in order — the other arms are not taken and
/// are never walked, matching CPython's own first-match semantics.
/// `None` (an unknown subject, an undecidable pattern shape, or no arm
/// decidably reached) keeps the existing blocker and forgets every name
/// bound anywhere in ANY case body, since the walk cannot say which arm
/// (if any) actually ran.
fn walk_match(
    match_stmt: &StmtMatch,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let subject_value = evaluate_expression(match_stmt.subject.as_ref(), environment, context.kernel);
    if let Some((taken_index, mut arm_env)) =
        match_taken_environment(&subject_value, &match_stmt.cases, environment, context.kernel)
    {
        let mut case_provably_unbound: HashSet<String> = HashSet::new();
        for stmt in &match_stmt.cases[taken_index].body {
            walk_statement(
                stmt,
                return_refinement,
                context,
                &mut arm_env,
                aug_assign_refinements,
                &mut case_provably_unbound,
                blocked,
                out,
            );
        }
        *environment = arm_env;
        return;
    }
    record_blocker(
        blocked,
        match_stmt.range(),
        "a match statement is not yet walked".to_owned(),
        out,
    );
    for case in &match_stmt.cases {
        forget_names_bound_in_body(&case.body, environment);
    }
}

/// `with EXPRESSION as TARGET: SUITE` (compound_stmts.rst, "The `with`
/// statement"): this walk models no context-manager protocol, so each
/// item's context expression is evaluated for its own side effects on
/// the environment (a call may read names) and discarded, and its
/// optional `as`-target is forgotten — step 5 of the with-statement's
/// own execution order binds the target to `__enter__`'s return value,
/// a value this walk cannot know, so forgetting is the honest answer
/// rather than guessing. The body then walks inline, on the SAME
/// environment, with no blocker for the with statement itself.
fn walk_with(
    with_stmt: &StmtWith,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    for item in &with_stmt.items {
        evaluate_expression(&item.context_expr, environment, context.kernel);
        if let Some(target) = item.optional_vars.as_deref() {
            forget_target_names(target, environment);
        }
    }
    let mut with_provably_unbound: HashSet<String> = HashSet::new();
    for stmt in &with_stmt.body {
        walk_statement(
            stmt,
            return_refinement,
            context,
            environment,
            aug_assign_refinements,
            &mut with_provably_unbound,
            blocked,
            out,
        );
    }
}

/// `try: BODY (except ... )+ [else: BODY] [finally: BODY]`
/// (compound_stmts.rst, "The `try` statement"): the try body walks on
/// its own fork; each handler starts from a fork of the INCOMING
/// environment with every name the try body binds forgotten — an
/// exception may interrupt the body at any point, so no write in it is
/// guaranteed to have happened, and any write that DID happen
/// invalidates the pre-try fact either way, so forgetting is the one
/// sound answer for both. A handler's `as`-name (if present) is bound by
/// forgetting it (exception objects are not modeled) and forgotten AGAIN
/// after the handler body — "when an exception has been assigned using
/// `as target`, it is cleared at the end of the except clause... as if
/// `except E as N: foo` was translated to... `finally: del N`". `orelse`
/// runs only once the try body completes without raising, continuing
/// that same fork ("the else clause is executed if the control flow
/// leaves the try suite, no exception was raised"). The post-try
/// environment joins every surviving path (the try+orelse fork and each
/// handler fork, `arm_terminates` deciding survival exactly as `if`
/// does); zero survivors keeps the incoming environment, the same
/// convention `walk_if` uses. `finalbody` then always runs, after the
/// join, on the joined environment — "the finally clause is executed...
/// if the try clause is executed, including any except and else
/// clauses."
fn walk_try(
    try_stmt: &StmtTry,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let mut surviving: Vec<Environment> = Vec::new();

    let mut try_env = environment.fork();
    let mut try_provably_unbound: HashSet<String> = HashSet::new();
    for stmt in &try_stmt.body {
        walk_statement(
            stmt,
            return_refinement,
            context,
            &mut try_env,
            aug_assign_refinements,
            &mut try_provably_unbound,
            blocked,
            out,
        );
    }
    // orelse ("the try statement", `else` clause) runs only once the try
    // body completes without raising — walking it here is this analysis's
    // static approximation of that runtime condition, same as every other
    // body this walk always visits regardless of the branch actually
    // taken at runtime. The combined path's survival is decided by the
    // LAST body actually executed along it: orelse's own last statement
    // when orelse is present, otherwise the try body's.
    for stmt in &try_stmt.orelse {
        walk_statement(
            stmt,
            return_refinement,
            context,
            &mut try_env,
            aug_assign_refinements,
            &mut try_provably_unbound,
            blocked,
            out,
        );
    }
    let try_path_terminal_body = if try_stmt.orelse.is_empty() {
        try_stmt.body.as_slice()
    } else {
        try_stmt.orelse.as_slice()
    };
    if !arm_terminates(try_path_terminal_body) {
        surviving.push(try_env);
    }

    for handler in &try_stmt.handlers {
        let ExceptHandler::ExceptHandler(handler) = handler;
        let mut handler_env = environment.fork();
        for stmt in &try_stmt.body {
            forget_names_bound_by_stmt(stmt, &mut handler_env);
        }
        // HANDLER AS-NAME: `except Exception as error:` binds `error` to
        // an OPAQUE caught-exception value at handler entry — not a
        // forget — so a read inside the handler body (e.g.
        // `try_except_binding`'s `age = error`) has something to judge
        // rather than reading Undetermined. `opaque_value` carries the
        // "not a scalar/set/object/list this domain models" standing
        // every existing opaque reader already treats as Unknown, so an
        // assignability ask against a scalar-ground declared set fires
        // through that law once it exists — this table only builds the
        // value and binds it.
        if let Some(name) = handler.name.as_ref() {
            handler_env.bind(name.id.as_str(), opaque_value("a caught exception"));
        }
        let mut handler_provably_unbound: HashSet<String> = HashSet::new();
        for stmt in &handler.body {
            walk_statement(
                stmt,
                return_refinement,
                context,
                &mut handler_env,
                aug_assign_refinements,
                &mut handler_provably_unbound,
                blocked,
                out,
            );
        }
        if let Some(name) = handler.name.as_ref() {
            handler_env.forget(name.id.as_str());
        }
        if !arm_terminates(&handler.body) {
            surviving.push(handler_env);
        }
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

    let mut finally_provably_unbound: HashSet<String> = HashSet::new();
    for stmt in &try_stmt.finalbody {
        walk_statement(
            stmt,
            return_refinement,
            context,
            environment,
            aug_assign_refinements,
            &mut finally_provably_unbound,
            blocked,
            out,
        );
    }
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
    if let Some((range, message)) = provable_raise(assign.value.as_ref(), environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
        // the raise happens before `x op= v` ever folds a value — the
        // target's own current value is untouched by CPython, but this
        // walk has no exception-continuation channel (the same posture
        // `Stmt::Assert`'s doc already states), so the honest answer is
        // to forget rather than assert the pre-raise value still holds
        // past this statement.
        environment.forget(name.id.as_str());
        return;
    }
    bind_walrus_targets(assign.value.as_ref(), context, aug_assign_refinements, environment, out);
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
    provably_unbound: &mut HashSet<String>,
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
        // PROVABLY-UNBOUND READS: `x: int` (valueless, and no declared
        // refinement this table reads) leaves `x` locally bound
        // (locally_bound_names' own scan) but with no environment
        // binding — the exact CPython UnboundLocalError shape
        // (executionmodel.rst's local-variable rule: a name assigned
        // anywhere in a function is local to the WHOLE function, and
        // reading it before any binding executes raises). A value-
        // carrying `x: int = v` cures it the same way an ordinary
        // assignment would.
        if let Expr::Name(target_name) = assign.target.as_ref() {
            if assign.value.is_none() {
                provably_unbound.insert(target_name.id.as_str().to_owned());
            } else {
                provably_unbound.remove(target_name.id.as_str());
            }
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
        // nothing judges and nothing binds here. Tracked the same way
        // the declined-annotation branch above tracks a valueless `x:
        // int` — the DECLARED-set path and the general path share the
        // one CPython fact (annotation-only never binds).
        if let Expr::Name(target_name) = assign.target.as_ref() {
            provably_unbound.insert(target_name.id.as_str().to_owned());
        }
        bind_target_from_value_expr(assign.target.as_ref(), None, environment, context.kernel);
        return;
    };

    if let Expr::Name(target_name) = assign.target.as_ref() {
        provably_unbound.remove(target_name.id.as_str());
    }
    bind_walrus_targets(value_expr, context, aug_assign_refinements, environment, out);
    let Some(value) = sink_value(value_expr, context, environment, out) else {
        // a provable raise already pushed its own RTS7001 at the
        // raising expression — this write never completes on this
        // path, so the target holds nothing: forget it, the same
        // "unproducible" answer Undetermined already forgets to.
        forget_target_names(assign.target.as_ref(), environment);
        return;
    };

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
    bind_walrus_targets(assign.value.as_ref(), context, aug_assign_refinements, environment, out);
    let Some(value) = sink_value(assign.value.as_ref(), context, environment, out) else {
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
        admits_none: false,
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
/// tuple/list targets attempt KNOWN-TUPLE DESTRUCTURING
/// (`bind_known_sequence_target`) when the RHS is a known `Kind::List`,
/// falling back to forgetting every name they touch when it is not (the
/// walk cannot destructure a value it cannot see the length/elements
/// of). An attribute target (`obj.x = v`, `self.x = v`) forgets the
/// RECEIVER's own base
/// name — the leftmost `Name` under the attribute chain
/// (`receiver_base_name`) — rather than leaving it alone: a known
/// instance bound to that name may carry a stale field value for `x`
/// after this write, and this file does not track field-level state
/// through an attribute write, so forgetting the whole receiver is the
/// one sound answer (no field-level tracking this unit).
///
/// STALE-RECEIVER SOUNDNESS, law (b): a subscript target (`name[key] =
/// value`, bare-Name receiver only) evaluates the receiver and the key
/// expressions, then replays the write through
/// `collection_models::dict_with_item` (an Object receiver) or
/// `list_with_item` (a List receiver): `Some(new receiver)` rebinds
/// `name` to it, so a later read sees the write (a-statements.py's
/// `collection_mutators`: `by_name["ann"] = 40` must leave `by_name`
/// holding `{"ann": 40}`, not the stale `{}`); `None` (an unknown
/// receiver, a non-Name receiver, a key/index this walk cannot read
/// exactly, or an index outside the list's current bounds) FORGETS
/// `name` — the pre-write value must not survive an unread write, the
/// same honesty every other decline in this file already keeps.
fn bind_or_forget_target(
    target: &Expr,
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match target {
        Expr::Name(name) => environment.bind(name.id.as_str(), value.clone()),
        Expr::Tuple(tuple) => {
            if !bind_known_sequence_target(
                &tuple.elts,
                value,
                value_range,
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for element in &tuple.elts {
                    forget_target_names(element, environment);
                }
            }
        }
        Expr::List(list) => {
            if !bind_known_sequence_target(
                &list.elts,
                value,
                value_range,
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for element in &list.elts {
                    forget_target_names(element, environment);
                }
            }
        }
        Expr::Starred(starred) => forget_target_names(starred.value.as_ref(), environment),
        Expr::Attribute(attribute) => {
            if let Some(base_name) = receiver_base_name(attribute.value.as_ref()) {
                environment.forget(base_name);
            }
        }
        Expr::Subscript(subscript) => {
            bind_or_forget_subscript_target(subscript, value, context, environment);
        }
        _ => {}
    }
}

/// KNOWN-TUPLE DESTRUCTURING: `(a, b, ...) = value` / `[a, b, ...] =
/// value`, where `value` is a KNOWN `Kind::List` (a-statements.py's
/// `tuple_unpack_ok`/`starred_unpack_ok`/`nested_tuple_unpack_ok` rows —
/// CPython does not distinguish list vs. tuple targets or RHS shape for
/// unpacking, simple_stmts.rst's `target: "(" [target_list] ")" | "["
/// [target_list] "]"` grammar treats both parenthesized and bracketed
/// target lists the same way). Returns `false` (no binding performed,
/// caller falls back to forgetting every name) when `value` is not a
/// known list — an unknown RHS states nothing about how many elements
/// there are, so this law does not apply and the existing forget-all
/// answer is the sound one.
///
/// With no starred element: `elements.len()` must equal `items.len()`
/// exactly — a mismatch is CPython's own `ValueError` ("too many values
/// to unpack (expected N)" / "not enough values to unpack (expected N,
/// got M)", both confirmed by execution against python3.12), fired here
/// as RTS7001 at the RHS value's own range, with EVERY target name
/// forgotten (the assignment never completes, so nothing binds). A
/// length match binds each element positionally: `Expr::Name` targets
/// bind (through `judge_and_bind` when the name carries a recorded
/// declared refinement, exactly like a plain `a = value` target — see
/// `walk_assign`'s own doc), and a nested `Expr::Tuple`/`Expr::List`
/// target recurses on that position's own element value.
///
/// With one starred element (`first, *rest = value` — a `SyntaxError` to
/// have more than one, so this table never needs to detect that case
/// itself): the elements BEFORE the star bind to the LIST'S head
/// positions, the elements AFTER the star bind to its TAIL positions
/// (counted from the end), and the starred name itself binds a
/// `Kind::List` of every element in between (`known_list` — the exact
/// "the middle slice" `first, *rest = years` gives `rest` in CPython).
/// Too few items for the non-starred elements alone (`head.len() +
/// tail.len() > items.len()`) is the starred row's own `ValueError`
/// ("not enough values to unpack (expected at least N, got M)",
/// confirmed by execution) — same fire-and-forget-all answer.
fn bind_known_sequence_target(
    elements: &[Expr],
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> bool {
    if value.kind != Kind::List {
        return false;
    }
    let items = &value.items;
    let starred_position = elements.iter().position(|element| matches!(element, Expr::Starred(_)));

    let Some(star_index) = starred_position else {
        if elements.len() != items.len() {
            out.push(Finding {
                range: value_range,
                code: "RTS7001",
                message: format!(
                    "this expression provably raises ValueError: {}",
                    unpack_mismatch_detail(elements.len(), items.len(), false),
                ),
            });
            for element in elements {
                forget_target_names(element, environment);
            }
            return true;
        }
        for (element, item) in elements.iter().zip(items.iter()) {
            bind_sequence_element(element, item, context, aug_assign_refinements, environment, out);
        }
        return true;
    };

    let head = &elements[..star_index];
    let tail = &elements[star_index + 1..];
    if head.len() + tail.len() > items.len() {
        out.push(Finding {
            range: value_range,
            code: "RTS7001",
            message: format!(
                "this expression provably raises ValueError: {}",
                unpack_mismatch_detail(head.len() + tail.len(), items.len(), true),
            ),
        });
        for element in elements {
            forget_target_names(element, environment);
        }
        return true;
    }
    for (element, item) in head.iter().zip(items.iter()) {
        bind_sequence_element(element, item, context, aug_assign_refinements, environment, out);
    }
    let tail_start = items.len() - tail.len();
    for (element, item) in tail.iter().zip(items[tail_start..].iter()) {
        bind_sequence_element(element, item, context, aug_assign_refinements, environment, out);
    }
    let Expr::Starred(starred) = &elements[star_index] else {
        unreachable!("star_index is the position matched against Expr::Starred above")
    };
    if let Expr::Name(name) = starred.value.as_ref() {
        let middle = list_literal_value(&items[head.len()..tail_start]);
        environment.bind(name.id.as_str(), middle);
    }
    true
}

/// One destructured position's own target: a bare name binds (through
/// `judge_and_bind` when the name carries a recorded declared
/// refinement — the same table an ordinary `x = value` target reads),
/// and a nested `Tuple`/`List` target recurses through
/// `bind_known_sequence_target` on that position's own known element —
/// a non-list element at a nested-tuple position is itself an unknown
/// shape to that recursive call, which then forgets that sub-target's
/// own names, matching the top-level "unknown RHS forgets" rule at
/// whatever depth it occurs.
fn bind_sequence_element(
    element: &Expr,
    item: &AbstractValue,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match element {
        Expr::Name(name) => match aug_assign_refinements.get(name.id.as_str()) {
            Some(declared) => {
                let declared = declared.clone();
                judge_and_bind(name.id.as_str(), item.clone(), &declared, element.range(), context, environment, out);
            }
            None => environment.bind(name.id.as_str(), item.clone()),
        },
        Expr::Tuple(tuple) => {
            if !bind_known_sequence_target(
                &tuple.elts,
                item,
                element.range(),
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for nested in &tuple.elts {
                    forget_target_names(nested, environment);
                }
            }
        }
        Expr::List(list) => {
            if !bind_known_sequence_target(
                &list.elts,
                item,
                element.range(),
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for nested in &list.elts {
                    forget_target_names(nested, environment);
                }
            }
        }
        _ => forget_target_names(element, environment),
    }
}

/// The CPython `ValueError` wording for a length-mismatch unpack,
/// confirmed by execution against python3.12: without a starred target,
/// "too many values to unpack (expected N)" when the RHS has MORE items
/// than targets, else "not enough values to unpack (expected N, got M)";
/// with a starred target (`has_star`), the expected count is a floor —
/// "not enough values to unpack (expected at least N, got M)" (a
/// starred target can never see "too many": it absorbs every surplus
/// element into its own list, so this row only ever under-supplies).
fn unpack_mismatch_detail(expected: usize, got: usize, has_star: bool) -> String {
    if has_star {
        return format!("not enough values to unpack (expected at least {expected}, got {got})");
    }
    if got > expected {
        format!("too many values to unpack (expected {expected})")
    } else {
        format!("not enough values to unpack (expected {expected}, got {got})")
    }
}

/// `name[key] = value` — see `bind_or_forget_target`'s own doc for law
/// (b)'s full contract. Only a bare-`Name` receiver is replayed; any
/// other receiver shape (`obj.attr[key] = v`, a chained subscript) has
/// no single environment slot to rebind and is left untouched, matching
/// this file's existing "no element-level model" posture for a receiver
/// it cannot name.
fn bind_or_forget_subscript_target(
    subscript: &ExprSubscript,
    value: &AbstractValue,
    context: &WalkContext,
    environment: &mut Environment,
) {
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        return;
    };
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    let key_value = evaluate_expression(subscript.slice.as_ref(), environment, context.kernel);
    let written = match receiver_value.kind {
        Kind::Object => dict_with_item(&receiver_value, &key_value, value),
        Kind::List => list_with_item(&receiver_value, &key_value, value),
        _ => None,
    };
    match written {
        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
}

/// The leftmost `Name` under an attribute-chain receiver
/// (`a.b.c` → `a`; `a` itself → `a`) — `None` when the receiver is not
/// built from a plain name chain at all (a call's own result, a
/// subscript, …), which this walk has no base name to forget either
/// way.
fn receiver_base_name(receiver: &Expr) -> Option<&str> {
    match receiver {
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Attribute(attribute) => receiver_base_name(attribute.value.as_ref()),
        _ => None,
    }
}

/// One `import`/`from…import` local name at its own import statement:
/// bind it to whatever `context.module_bindings` resolved for it (the
/// cross-module surface already did the resolving), or forget it when
/// the surface carries nothing under that name — a function/class
/// import (readable through `environment.functions()`/`.classes()`,
/// not a plain value), an unresolved module, or a star import's own
/// literal `"*"` alias (never a real local name).
fn bind_or_forget_imported_name(local_name: &str, context: &WalkContext, environment: &mut Environment) {
    match context.module_bindings.get(local_name) {
        Some(value) => environment.bind(local_name, value.clone()),
        None => environment.forget(local_name),
    }
}

/// The value a write/return/expression-statement sink's own value
/// expression produces, after two checks the ordinary
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
/// 2. Statement-level CONSTRUCTION (`construction_call_verdict`): a
///    call recognized as building a same-module or imported
///    `ClassModel` instance. Each fire `judge_construction` returns is
///    pushed as its own RTS7001, and the sink's value is
///    `verdict.instance` — never the plain `evaluate_expression`
///    reading of an unmodeled call.
///
/// Neither check applies: falls through to the ordinary
/// `evaluate_expression` reading, unchanged from before this unit.
fn sink_value(
    expr: &Expr,
    context: &WalkContext,
    environment: &Environment,
    out: &mut Vec<Finding>,
) -> Option<AbstractValue> {
    if let Some((range, message)) = provable_raise(expr, environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
        return None;
    }
    if let Some(verdict) = construction_call_verdict(expr, context, environment) {
        for (range, message) in verdict.fires {
            out.push(Finding { range, code: "RTS7001", message });
        }
        return Some(verdict.instance);
    }
    Some(evaluate_expression(expr, environment, context.kernel))
}

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
fn walk_mutating_call_statement(
    expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
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
    let argument_values: Vec<AbstractValue> = arguments.iter().map(|(value, _)| value.clone()).collect();
    match mutated_receiver(method, &receiver_value, &argument_values) {
        Some((new_receiver, _result)) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
    true
}

/// Recognizes `expr` as a class-construction call and judges it, or
/// `None` when `expr` is not one of the two recognized construction
/// shapes:
///
/// (a) a bare-Name call (`Person(40)`, `Person(age=40)`) whose callee
///     is UNBOUND in the environment (a bound name shadows the class,
///     same rule `evaluate_call` already applies to a builtin name) and
///     names a `ClassModel` in `context.classes` — every positional
///     argument evaluates in order, every keyword argument evaluates
///     and pairs with its own name.
/// (b) `<ClassName>.model_validate(<dict literal>)` or
///     `TypeAdapter(<ClassName>).validate_python(<dict literal>)` —
///     pydantic's own parse surface (m-pydantic-schema.py's
///     `model_validate`/`TypeAdapter(...).validate_python` rows):
///     `ClassName` must be a bare Name in `context.classes`, and the
///     single argument must be a `Dict` literal so its keys map
///     directly to keyword rows; any other argument shape (a name, a
///     call, a non-literal key) is not construction this function
///     reads, and the call falls through to the ordinary
///     `evaluate_expression` path.
fn construction_call_verdict(
    expr: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<ConstructionVerdict> {
    let Expr::Call(call) = expr else {
        return None;
    };
    if let Expr::Name(callee) = call.func.as_ref() {
        if environment.read(callee.id.as_str()).is_none() {
            if let Some(model) = context.classes.get(callee.id.as_str()) {
                let positional = evaluate_positional_arguments(&call.arguments.args, environment, context.kernel);
                let keyword = evaluate_keyword_arguments(&call.arguments.keywords, environment, context.kernel);
                return Some(judge_construction(model, &positional, &keyword, context.kernel));
            }
        }
        return None;
    }
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if attribute.attr.as_str() == "model_validate" {
        let model = class_model_of_bare_name(attribute.value.as_ref(), context, environment)?;
        let dict_argument = single_dict_argument(&call.arguments)?;
        let keyword = dict_literal_keyword_rows(dict_argument, environment, context.kernel)?;
        return Some(judge_construction(model, &[], &keyword, context.kernel));
    }
    if attribute.attr.as_str() == "validate_python" {
        // `TypeAdapter(<ClassName>).validate_python(<dict literal>)` —
        // the receiver is itself a Call: `TypeAdapter`'s own single
        // positional argument names the class.
        let Expr::Call(adapter_call) = attribute.value.as_ref() else {
            return None;
        };
        let Expr::Name(adapter_name) = adapter_call.func.as_ref() else {
            return None;
        };
        if adapter_name.id.as_str() != "TypeAdapter" {
            return None;
        }
        let [Expr::Name(class_name)] = adapter_call.arguments.args.as_ref() else {
            return None;
        };
        let model = context.classes.get(class_name.id.as_str())?;
        let dict_argument = single_dict_argument(&call.arguments)?;
        let keyword = dict_literal_keyword_rows(dict_argument, environment, context.kernel)?;
        return Some(judge_construction(model, &[], &keyword, context.kernel));
    }
    None
}

/// `<ClassName>` out of a bare-Name expression naming a class in
/// `context.classes` — the receiver shape `<ClassName>.model_validate`
/// reads. `None` for anything else (a non-Name receiver, or a Name that
/// is either environment-bound to something else or simply not a known
/// class).
fn class_model_of_bare_name<'a>(
    expr: &Expr,
    context: &'a WalkContext,
    environment: &Environment,
) -> Option<&'a ClassModel> {
    let Expr::Name(name) = expr else {
        return None;
    };
    if environment.read(name.id.as_str()).is_some() {
        return None;
    }
    context.classes.get(name.id.as_str())
}

/// The single positional argument of a call, when it is a `Dict`
/// literal — `model_validate`/`validate_python`'s own argument shape.
/// `None` for zero/multiple arguments, any keyword argument, or a
/// positional argument that is not a `Dict` display.
fn single_dict_argument(arguments: &ruff_python_ast::Arguments) -> Option<&ruff_python_ast::ExprDict> {
    if !arguments.keywords.is_empty() {
        return None;
    }
    let [Expr::Dict(dict)] = arguments.args.as_ref() else {
        return None;
    };
    Some(dict)
}

/// A `{"key": value, ...}` literal's rows, mapped to `judge_construction`'s
/// own keyword-row shape: each entry's STRING key becomes the field
/// name, its value expression evaluates through `evaluate_expression`,
/// and the row's range is the VALUE expression's own range (so a fire
/// anchors at the value that refused, matching every other sink in this
/// file). `None` the moment any entry's key is not a plain string
/// literal (a computed key, a `**spread` entry) — the same all-or-
/// nothing posture `collection_models::dict_literal_value` already
/// takes for a dict display it cannot read exactly.
fn dict_literal_keyword_rows(
    dict: &ruff_python_ast::ExprDict,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(String, AbstractValue, TextRange)>> {
    let mut rows = Vec::with_capacity(dict.items.len());
    for item in &dict.items {
        let Some(Expr::StringLiteral(key)) = item.key.as_ref() else {
            return None;
        };
        let value = evaluate_expression(&item.value, environment, kernel);
        rows.push((key.value.to_str().to_owned(), value, item.value.range()));
    }
    Some(rows)
}

/// Every positional argument of a construction call, evaluated in
/// order — the same per-argument evaluation `evaluate_call` already
/// does for a builtin, paired here with each argument's own range so
/// `judge_construction`'s fires anchor at the refusing argument.
fn evaluate_positional_arguments(
    args: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Vec<(AbstractValue, TextRange)> {
    args.iter()
        .map(|arg| (evaluate_expression(arg, environment, kernel), arg.range()))
        .collect()
}

/// Every keyword argument of a construction call (`name=value` rows
/// only — `**spread` keywords carry no `arg` identifier and are
/// skipped, since this table cannot know which field a spread's keys
/// would land in).
fn evaluate_keyword_arguments(
    keywords: &[ruff_python_ast::Keyword],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Vec<(String, AbstractValue, TextRange)> {
    keywords
        .iter()
        .filter_map(|keyword| {
            let name = keyword.arg.as_ref()?;
            let value = evaluate_expression(&keyword.value, environment, kernel);
            Some((name.id.as_str().to_owned(), value, keyword.value.range()))
        })
        .collect()
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

/// Forget every name a single statement binds anywhere within its own
/// sub-bodies (its target plus every name any nested body binds) — the
/// blocker-path cleanup for a `for`/`while` the loop module declined:
/// reuses `collect_bound_names_stmt`'s own walk of that statement's
/// shape so the "what does this bind" answer never drifts from the
/// scope prepass's.
fn forget_names_bound_by_stmt(stmt: &Stmt, environment: &mut Environment) {
    let mut bound = HashSet::new();
    let mut excluded = HashSet::new();
    collect_bound_names_stmt(stmt, &mut bound, &mut excluded);
    for name in &excluded {
        bound.remove(name);
    }
    for name in &bound {
        environment.forget(name);
    }
}

/// Forget every name a body binds anywhere within it — the blocker-path
/// cleanup for a `match` the arm-decision module declined to resolve
/// (used per undecided case body, since the walk cannot say which arm,
/// if any, actually ran).
fn forget_names_bound_in_body(body: &[Stmt], environment: &mut Environment) {
    let mut bound = HashSet::new();
    let mut excluded = HashSet::new();
    collect_bound_names(body, &mut bound, &mut excluded);
    for name in &excluded {
        bound.remove(name);
    }
    for name in &bound {
        environment.forget(name);
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

/// WALRUS BINDING: every `:=` reachable inside an expression the walk
/// evaluates binds its bare-Name target into the ENCLOSING environment
/// — the same traversal shape `collect_walrus_names` already walks (for
/// the SCOPE prepass, which only needs the target's spelling), reused
/// here to also BIND the target to its evaluated inner value (what the
/// scope prepass does not need, since it runs before any environment
/// exists to bind into). `evaluate_expression` already reads
/// `Expr::Named` correctly wherever it is nested (it returns the inner
/// value, `expressions.rs`'s own dispatch), so evaluating each found
/// walrus's OWN inner expression here — a second, cheap evaluation of a
/// pure expression tree with no side effects to duplicate — is the
/// direct way to get the exact same value the walrus's surrounding
/// expression already computed from it.
///
/// `aug_assign_refinements` judges a declared name's walrus value
/// through `judge_and_bind` exactly like a plain `x = value` target
/// (`walrus_in_condition`'s own `over := 200` under a later `Age`-typed
/// read is the corpus row this serves); an undeclared target binds
/// directly. A non-Name walrus target is not legal Python grammar
/// (`named_expression: assignment_expression | expression`, PEP 572 —
/// the target is always an identifier) and never reaches this function
/// at all, so there is no "else" case to handle.
fn bind_walrus_targets(
    expr: &Expr,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match expr {
        Expr::Named(named) => {
            if let Expr::Name(target_name) = named.target.as_ref() {
                let inner_value = evaluate_expression(named.value.as_ref(), environment, context.kernel);
                match aug_assign_refinements.get(target_name.id.as_str()) {
                    Some(declared) => {
                        let declared = declared.clone();
                        judge_and_bind(
                            target_name.id.as_str(),
                            inner_value,
                            &declared,
                            named.value.range(),
                            context,
                            environment,
                            out,
                        );
                    }
                    None => environment.bind(target_name.id.as_str(), inner_value),
                }
            }
            bind_walrus_targets(named.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::BoolOp(op) => {
            for value in &op.values {
                bind_walrus_targets(value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::BinOp(op) => {
            bind_walrus_targets(op.left.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(op.right.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::UnaryOp(op) => bind_walrus_targets(op.operand.as_ref(), context, aug_assign_refinements, environment, out),
        // the lambda's OWN body is a separate scope; a walrus inside it
        // does not bind here — mirrors collect_walrus_names exactly.
        Expr::Lambda(_) => {}
        Expr::If(if_expr) => {
            bind_walrus_targets(if_expr.test.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(if_expr.body.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(if_expr.orelse.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                bind_walrus_targets(element, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                bind_walrus_targets(element, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                bind_walrus_targets(element, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    bind_walrus_targets(key, context, aug_assign_refinements, environment, out);
                }
                bind_walrus_targets(&item.value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Call(call) => {
            bind_walrus_targets(call.func.as_ref(), context, aug_assign_refinements, environment, out);
            for arg in &call.arguments.args {
                bind_walrus_targets(arg, context, aug_assign_refinements, environment, out);
            }
            for keyword in &call.arguments.keywords {
                bind_walrus_targets(&keyword.value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Compare(compare) => {
            bind_walrus_targets(compare.left.as_ref(), context, aug_assign_refinements, environment, out);
            for comparator in &compare.comparators {
                bind_walrus_targets(comparator, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Attribute(attribute) => {
            bind_walrus_targets(attribute.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Subscript(subscript) => {
            bind_walrus_targets(subscript.value.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(subscript.slice.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Starred(starred) => {
            bind_walrus_targets(starred.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Slice(slice) => {
            if let Some(lower) = slice.lower.as_deref() {
                bind_walrus_targets(lower, context, aug_assign_refinements, environment, out);
            }
            if let Some(upper) = slice.upper.as_deref() {
                bind_walrus_targets(upper, context, aug_assign_refinements, environment, out);
            }
            if let Some(step) = slice.step.as_deref() {
                bind_walrus_targets(step, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::FString(fstring) => {
            for element in fstring.value.elements() {
                if let Some(interpolation) = element.as_interpolation() {
                    bind_walrus_targets(interpolation.expression.as_ref(), context, aug_assign_refinements, environment, out);
                }
            }
        }
        Expr::Await(inner) => bind_walrus_targets(inner.value.as_ref(), context, aug_assign_refinements, environment, out),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                bind_walrus_targets(value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::YieldFrom(inner) => {
            bind_walrus_targets(inner.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        // Comprehensions introduce their own scope for their loop
        // variables; a walrus inside one still targets the enclosing
        // scope per PEP 572, but (mirroring collect_walrus_names) that
        // expression-walking depth is not built this wave.
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

    #[test]
    fn a_literal_range_for_loop_accumulates_and_the_out_of_set_total_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        // loop_final_environment runs [200] concretely (a single-element
        // literal list, a shape it CAN execute), leaving `total` bound to
        // 200 with no blocker; the read afterward judges that value.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    total: Age = 0\n",
            "    for x in [200]:\n",
            "        total = x\n",
            "    over: Age = total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a concretely-executable for loop must record no blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the post-loop read of total (200, the loop's last element) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_for_loop_over_an_unknown_iterable_blocks_and_forgets_its_stale_binding() {
        let Some(kernel) = loaded_kernel() else { return };
        // `items` is an unannotated parameter, so its value is unknown —
        // literal_iterable_values cannot read it and loop_final_environment
        // declines. `total` held an OUT-OF-SET literal immediately before
        // the loop; had the blocker path left that stale fact bound, the
        // read after the loop would fire a second time on it. The fix
        // forgets `total` (and the loop's own target `x`) at the blocker,
        // so the post-loop read is Undetermined, not a second Fire.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(items) -> None:\n",
            "    total = 200\n",
            "    for x in items:\n",
            "        total = 5\n",
            "    check: Age = total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            blockers.len(),
            1,
            "the unmodeled for loop is this body's one blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(blockers[0].message.contains("for"), "{}", blockers[0].message);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert!(
            fires.is_empty(),
            "total's stale pre-loop value (200) must not survive to fire after an unmodeled loop: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_match_on_a_known_subject_takes_its_arm_and_fires_inside_it() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    x = 1\n",
            "    match x:\n",
            "        case 1:\n",
            "            over: Age = 200\n",
            "        case _:\n",
            "            pass\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a match on a known subject must record no blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "only the taken arm (case 1) is walked, and it fires on 200: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_with_body_still_judges_and_records_no_blocker_for_the_with() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(cm) -> None:\n",
            "    with cm as ctx:\n",
            "        over: Age = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a with statement must record no blocker of its own: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the with body's AnnAssign still fires on 200: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_try_body_out_of_set_ann_assign_fires_with_no_blocker_for_the_try() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    try:\n",
            "        over: Age = 200\n",
            "    except Exception:\n",
            "        pass\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a try statement must record no blocker of its own: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the try body's AnnAssign still fires on 200: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_try_except_join_forgets_the_declared_slots_pre_try_out_of_set_value() {
        let Some(kernel) = loaded_kernel() else { return };
        // total starts OUT of Age's set (200, fires once). The try body
        // rebinds it in-set and then returns, so the try path never
        // survives to the join — only the handler does. The handler's
        // starting environment must forget the try body's bound names
        // (total among them), so the pre-try 200 does not leak through
        // the join into the post-try environment. Had it leaked, the
        // final read below would fire a SECOND time on the stale 200.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    total: Age = 200\n",
            "    try:\n",
            "        total = 40\n",
            "        return\n",
            "    except Exception:\n",
            "        pass\n",
            "    check: Age = total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "only the pre-try declaration's own refusal (200) may fire — total must not carry its stale pre-try value through the join: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
        let try_blockers: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7002" && f.message.contains("try statement"))
            .collect();
        assert!(
            try_blockers.is_empty(),
            "the try statement itself must never be recorded as a blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_same_module_def_call_flows_a_known_return_into_a_declared_sink() {
        let Some(kernel) = loaded_kernel() else { return };
        // `over` is a module-level def, readable through
        // environment.functions() (walk_body seeds it on every body,
        // this module's own): the call resolves through
        // summaries::call_result and its known return (200) fires
        // against Age at the declared sink.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def over() -> int:\n",
            "    return 200\n",
            "def f() -> None:\n",
            "    x: Age = over()\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the same-module call's known return (200) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn an_imported_value_read_through_a_two_module_resolver_fires_at_a_return_sink() {
        let Some(kernel) = loaded_kernel() else { return };
        // A closure resolver over an in-memory map of module name ->
        // source text (cross_module.rs's own test pattern) stands in
        // for disk_resolver: `helper.py` states an out-of-set constant,
        // and the entry module's `from helper import over_years` makes
        // it readable at the return sink through context.module_bindings.
        let mut sources: HashMap<&str, &str> = HashMap::new();
        sources.insert("helper", "over_years = 200\n");
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "from helper import over_years\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    return over_years\n",
        ));
        let resolver: ModuleResolver = &|name: &str| sources.get(name).map(|source| parsed(source));
        let findings = findings_for_module_with_resolver(&module, resolver, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the imported constant (200) must fire at the return sink: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_keyword_construction_call_fires_on_an_out_of_set_field() {
        let Some(kernel) = loaded_kernel() else { return };
        // Person(age=200): a bare-Name construction call naming a
        // same-module class, judged through instances::judge_construction
        // — the keyword argument maps to the age field's own Annotated
        // set and fires. `type Age = ...` is declared (even though the
        // field spells its own inline Annotated[...]) because
        // findings_for_module's own aliases-gate returns nothing at all
        // for a module with zero type-alias statements.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, BaseModel\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Person(BaseModel):\n",
            "    age: Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    p = Person(age=200)\n",
            "    _ = p\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the keyword construction argument (200) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_provably_false_if_test_fires_and_its_body_is_never_walked() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements:400's own shape: a helper whose every real return
        // is a live dict never answers None, so `held is None` is
        // provably false — the dead-branch law fires there, and the
        // out-of-set `return 200` inside that branch must never be
        // walked (no second RTS7001 for it).
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def helper_never_answers_none(adult: bool) -> dict[str, int] | None:\n",
            "    if adult:\n",
            "        return {\"age\": 40}\n",
            "    return {\"age\": 10}\n",
            "def f(adult: bool) -> Age:\n",
            "    held = helper_never_answers_none(adult)\n",
            "    if held is None:\n",
            "        return 200\n",
            "    return 40\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let dead_branch_fires: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
            .collect();
        assert_eq!(
            dead_branch_fires.len(),
            1,
            "the known-false `is None` test must fire exactly once: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let two_hundred_fires: Vec<&Finding> =
            findings.iter().filter(|f| f.code == "RTS7001" && f.message.contains("'200'")).collect();
        assert!(
            two_hundred_fires.is_empty(),
            "the dead branch's own `return 200` must never be walked: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_provable_raise_at_an_expr_statement_fires() {
        // Coded against expressions::provable_raise, landing in a
        // parallel follow-up unit — a known zero divisor
        // (`1 / 0`) is CPython's own unconditional ZeroDivisionError
        // (expressions.rst §6.7, division). Present per the mission's
        // instruction to leave this test in place, noted in the report,
        // rather than stubbing provable_raise here.
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    1 / 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "a known zero divisor is a provable raise and must fire once: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- STALE-RECEIVER SOUNDNESS, law (a): mutating method calls ---

    #[test]
    fn a_list_append_carries_the_new_element_into_a_later_read() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    ages = [40]\n",
            "    ages.append(200)\n",
            "    over: Age = ages[1]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the appended 200 must be visible at ages[1], not the stale pre-append list: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn an_unmodeled_mutating_method_forgets_the_receiver_rather_than_reading_the_stale_value() {
        let Some(kernel) = loaded_kernel() else { return };
        // `sort` is not in collection_models::mutated_receiver's modeled
        // row set — the receiver must be forgotten (Undetermined), never
        // left bound to its pre-call value.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    ages = [40, 200]\n",
            "    ages.sort()\n",
            "    over: Age = ages[0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "an unmodeled mutator must forget the receiver, never fire on its stale value: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- STALE-RECEIVER SOUNDNESS, law (b): subscript-target writes ---

    #[test]
    fn a_dict_item_write_carries_the_new_value_into_a_later_read() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    ages: dict[str, int] = {}\n",
            "    ages[\"ann\"] = 200\n",
            "    over: Age = ages[\"ann\"]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the written 200 must be visible at ages[\"ann\"], not the stale empty dict: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_list_item_write_carries_the_new_value_into_a_later_read() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    ages = [40, 41]\n",
            "    ages[0] = 200\n",
            "    over: Age = ages[0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the written 200 must be visible at ages[0]: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- KNOWN-TUPLE DESTRUCTURING (law 2) ---

    #[test]
    fn a_known_tuple_target_binds_each_position_and_judges_it() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    a: Age\n",
            "    b: Age\n",
            "    a, b = (200, 40)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "only a's position (200) is out of set; b's (40) is in set: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_starred_target_binds_the_head_and_the_middle_list() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    first, *rest = [200, 20, 30]\n",
            "    over: Age = first\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the starred target's head element (200) must bind and judge: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_length_mismatch_unpack_of_a_known_list_fires_value_error_and_forgets_every_target() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    a, b = (1, 2, 3)\n",
            "    over: Age = a\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let raises: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("provably raises ValueError"))
            .collect();
        assert_eq!(
            raises.len(),
            1,
            "a 3-item tuple unpacked into 2 targets provably raises ValueError: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(
            raises[0].message.contains("too many values to unpack (expected 2)"),
            "{}",
            raises[0].message
        );
        let age_fires: Vec<&Finding> =
            findings.iter().filter(|f| f.code == "RTS7001" && f.message.contains("'Age'")).collect();
        assert!(
            age_fires.is_empty(),
            "every target must be forgotten after the raise — no second fire reading 'a': {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- HANDLER AS-NAME (law 3) ---

    #[test]
    fn a_caught_exception_bound_to_a_declared_int_slot_fires_through_the_opaque_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    try:\n",
            "        raise ValueError(1)\n",
            "    except ValueError as error:\n",
            "        return error\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        // The handler's as-name must be bound to something (not forgotten
        // at entry) — an Undetermined blocker at worst, or a Fire once
        // assignability reads the opaque marker. Either way it must not
        // be silently absent from the findings the way "forget" would
        // leave it (no finding at all).
        assert!(
            !findings.is_empty(),
            "a caught exception returned under a declared int-sorted set must not pass silently: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- LOCAL DEFS (law 4) ---

    #[test]
    fn a_locally_defined_function_is_callable_through_its_own_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    def over_years() -> int:\n",
            "        return 200\n",
            "    return over_years()\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the local def's known return (200) must fire through the call: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- WALRUS BINDING (law 5) ---

    #[test]
    fn a_walrus_in_an_if_test_binds_the_target_for_the_rest_of_the_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    if (over := 200) > 0:\n",
            "        return over\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the walrus-bound 200 must be readable inside the taken arm: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- PROVABLY-UNBOUND READS (law 6) ---

    #[test]
    fn a_valueless_annotation_then_a_return_fires_unbound_local_error() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    x: int\n",
            "    return x\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let raises: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("UnboundLocalError"))
            .collect();
        assert_eq!(
            raises.len(),
            1,
            "a valueless declaration read with no intervening assignment provably raises: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(raises[0].message.contains("'x'"), "{}", raises[0].message);
    }

    #[test]
    fn a_valueless_annotation_cured_by_an_assignment_never_fires_unbound() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    x: int\n",
            "    x = 40\n",
            "    return x\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| !f.message.contains("UnboundLocalError")),
            "an assignment between the declaration and the read cures it: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_valueless_annotation_behind_a_branch_never_fires_unbound_conservatively() {
        let Some(kernel) = loaded_kernel() else { return };
        // A branch between the declaration and the read COULD have bound
        // x on some path this straight-line tracking does not follow —
        // the conservative rule says no fire, even though this particular
        // program still never assigns x.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(flag: bool) -> Age:\n",
            "    x: int\n",
            "    if flag:\n",
            "        pass\n",
            "    return x\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| !f.message.contains("UnboundLocalError")),
            "a branch between declaration and read must suppress the fire conservatively: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }
}
