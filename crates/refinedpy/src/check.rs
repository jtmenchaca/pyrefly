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
use std::sync::Mutex;

use refined_domain::abstract_value::{
    known_set, known_values, possibly_absent, unknown, AbsentFlavor, AbstractValue, Kind, ObjectKey, PrimitiveKind,
    SetKindTag,
};
use refined_domain::known_constructors::{known_list, known_object};
use refined_domain::trust_grades::{TrustProved, TrustSpec};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::{make_refined_set, on_one_tuple_layer, one_of, repeat_of, requires_integer, RefinedSet};
use ruff_python_ast::{
    Alias, AtomicNodeIndex, CmpOp, ExceptHandler, ExceptHandlerExceptHandler, Expr, ExprAttribute, ExprSubscript,
    ModModule, Parameters, Stmt, StmtAnnAssign, StmtAssign, StmtAugAssign, StmtClassDef, StmtFunctionDef,
    StmtIf, StmtMatch, StmtRaise, StmtReturn, StmtTry, StmtWith, WithItem,
};
use ruff_text_size::{Ranged, TextRange, TextSize};

use crate::assignability::{judge, states_sequence, Verdict};
use crate::bytes_models::{self, BytesAnswer};
use crate::collection_models::{dict_get_result, dict_with_item, dict_without_item, list_literal_value, list_with_item, mutated_receiver, subscript_read};
use crate::cross_module::{module_surface, ModuleResolver, IMPORT_DEPTH_CAP};
use crate::diagnostic_sentences::{empty_set, loop_accumulation_did_not_stabilize, unhonorable_annotation};
use crate::env::Environment;
use crate::expressions::{binary_arithmetic_value, evaluate_expression, fieldless_exception_value, possible_raise, provable_raise, register_retained_callables};
use crate::foreign_edge;
use crate::function_table::{function_table, merged, FunctionTable};
use crate::instances;
use crate::instances::{class_table, judge_construction, ClassModel, ConstructionVerdict};
use crate::loops::{loop_final_environment, LoopAnswer};
use crate::match_arms;
use crate::match_arms::match_taken_environment;
use crate::narrowing::assume;
use crate::relational_sum;
use crate::summaries;
use crate::surface::{compile_aliases, strict_int_alias_names, surface_imports, AliasEntry};
use crate::typereading::{base_sort_return_refinement, callable_return_refinement, declared_refinement, typed_dict_return_refinement, DeclaredRefinement};

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
    aliases: &'a HashMap<String, AliasEntry>,
    imports: &'a crate::surface::SurfaceImports,
    kernel: &'a Arc<RefinedTSKernel>,
    functions: Arc<FunctionTable>,
    classes: Arc<HashMap<String, ClassModel>>,
    /// The module's own `datetime` import identities — which local
    /// names mean the `datetime` module, `datetime.datetime`,
    /// `datetime.date`, and `datetime.timedelta`
    /// (`expressions::DatetimeImports`'s own doc), built once here the
    /// same "built once before any body walk" posture `functions`/
    /// `classes` already take, and layered onto each body's own
    /// `Environment` (`walk_body_with_self_binding`) so
    /// `expressions.rs`'s datetime gates answer by canonical identity
    /// rather than the literal `datetime`/`date`/`timedelta` spelling.
    datetime_imports: Arc<crate::expressions::DatetimeImports>,
    module_bindings: HashMap<String, AbstractValue>,
    /// Every MODULE-LEVEL callable-variable's own return refinement:
    /// `name: Callable[[...], R] = ...` (or `| None`) at the module's
    /// top level, keyed on `name`, read through
    /// `typereading::callable_return_refinement`. Built once here (the
    /// same "built once before any body walk" posture `functions`/
    /// `classes` already take) so every body — the module body itself,
    /// and every nested `def` reached through the one shared `context`
    /// — starts with the module's own callable declarations visible; a
    /// body-local `Callable`-typed variable is layered on top of this
    /// by `walk_body_with_self_binding`'s own per-body table.
    module_callable_returns: Arc<HashMap<String, DeclaredRefinement>>,
    /// Every module-level `type X = Annotated[StrictInt, …]` alias name
    /// (`surface::strict_int_alias_names`) — the TypeAdapter adapter
    /// route consults this to decide whether a `str` argument against
    /// this alias may coerce (a lax `int` base) or must refuse outright
    /// (a `StrictInt` base never attempts str-to-int coercion,
    /// execution-verified against pydantic 2.13.4).
    strict_int_aliases: &'a HashSet<String>,
    /// Every module-level `class X(TypedDict): name: Annotation, …`
    /// read into its own per-member refinement table
    /// (`instances::typed_dict_table`), keyed on the class's name —
    /// consulted where a `-> X`/`x: X` annotation names a TypedDict
    /// rather than a `type X = …` alias, so a dict literal judged
    /// against it is judged member-by-member (`typed_dict_return_
    /// refinement`) instead of reading as unrefined.
    typed_dicts: Arc<HashMap<String, Vec<(String, DeclaredRefinement)>>>,
    /// Every module-level def's own direct callers' positional argument
    /// lists, keyed by the def's name (`function_table::
    /// caller_argument_positions`) — a def missing from this table either
    /// has no module-level definition or does not qualify (some
    /// occurrence of its name is not a plain positional call, so no
    /// caller of it is safe to join). `seed_parameters` reads this to
    /// fold an UNANNOTATED parameter's every caller argument at that
    /// position into a literal-union seed.
    caller_arguments: Arc<crate::function_table::CallerArguments>,
    /// The checked file's own directory, when the caller knows it — a
    /// foreign edge's relative argv entry (`"./audio_level.ts"`) is
    /// relative to the file that wrote it, never to the eventual
    /// process's cwd, so the recognizer joins against this. `None`
    /// (the resolver-less test entry points) leaves relative targets
    /// unresolved, which declines honestly downstream.
    entry_directory: Option<std::path::PathBuf>,
    /// The whole module walk's shared evaluations recorder, when a
    /// caller asked for one (`refined_set_at_position`'s own doc).
    /// `walk_body_with_self_binding` installs this SAME `Arc` on every
    /// body's fresh `Environment` the moment it builds one
    /// (`Environment::set_evaluations_recorder`), so the module body,
    /// every top-level `def`, and every nested `def`/method all write
    /// into the one `Vec` the caller reads back once the whole walk
    /// finishes. `None` for every ordinary check (`findings_for_
    /// module_at`, `derived_return_values`) — ordinary walks never pay
    /// for recording they never asked for.
    evaluations_recorder: Option<Arc<Mutex<Vec<(TextRange, AbstractValue)>>>>,
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
    findings_for_module_at(module, resolver, kernel, None)
}

/// `findings_for_module_with_resolver` plus the checked file's own
/// directory — the CLI and the LSP seam both know it, and the foreign
/// edge needs it to resolve a relative argv target against the file
/// that wrote it rather than the process's cwd.
pub fn findings_for_module_at(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Vec<Finding> {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    if aliases.is_empty()
        && imports.annotated_names.is_empty()
        && imports.literal_names.is_empty()
    {
        // No module-level `type X = Annotated[...]` alias, no
        // recognized `Annotated` import, and no `Literal` import — no
        // statement in this module can carry refinement vocabulary
        // this checker reads, so this is the fast, zero-cost early
        // return. A module whose only vocabulary is an inline
        // `Annotated[...]` or `Literal[...]` annotation still has the
        // matching import set populated (`surface_imports` reads only
        // import statements, cheap before any walk), so it reaches
        // the walk below rather than being skipped without judgment.
        return Vec::new();
    }
    let surface = module_surface(module, resolver, kernel, IMPORT_DEPTH_CAP);
    let own_functions = function_table(module);
    let functions = Arc::new(merged(&own_functions, surface.functions.as_ref()));
    let own_classes = class_table(module, &aliases, &imports, kernel);
    // `ClassModel` derives `Clone` (instances.rs), but the merge still
    // takes OWNERSHIP of the imported map rather than cloning it —
    // cheaper, and sound here because `module_surface` just built this
    // `Arc` fresh, with no other clone anywhere yet, so its strong count
    // is exactly 1.
    let mut classes = Arc::try_unwrap(surface.classes)
        .unwrap_or_else(|_| panic!("module_surface's own Arc<classes> has no other owner yet"));
    for (name, model) in own_classes {
        classes.insert(name, model);
    }
    // SAME-MODULE-DEF LOCAL CLASSES: a-statements.py's own `device()` — a
    // module-level `def` whose body declares a local class (`_Device`)
    // and returns its construction. `summaries::call_result_with_enclosing`
    // (a completely separate interpretation of `device`'s body, run at the
    // VALUE call site, e.g. `with device() as handle:`) already tags the
    // answered instance `source = "_Device"` through its own
    // `interpret_class_def` — but that class table is scratch state,
    // local to that one interpreted call and discarded when it returns.
    // `enter_method_result`/`instance_method_call_result`/
    // `construction_call_verdict` all resolve an instance's class SOLELY
    // through `context.classes` (`WalkContext`'s own module-wide table,
    // built once here), so a class this checker only ever discovers by
    // interpreting a same-module def's body must ALSO be registered here
    // — otherwise `with device() as handle: return handle.value` can
    // never find `_Device`'s own `__enter__`, even though the instance's
    // own tag names it correctly. Every module-level `def`'s own body is
    // scanned the same way `local_class_table` already scans a body-local
    // class for the body CURRENTLY being walked; local name wins on a
    // spelling collision with a module-level class, matching every other
    // merge in this function.
    for def in module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    }) {
        let def_local_classes = local_class_table(&def.body, &aliases, &imports, kernel);
        for (name, model) in def_local_classes {
            classes.insert(name, model);
        }
    }
    let module_callable_returns = Arc::new(module_level_callable_returns(module, &aliases, &imports));
    let strict_int_aliases = strict_int_alias_names(module);
    let typed_dicts = Arc::new(instances::typed_dict_table(module, &aliases, &imports));
    let caller_arguments = Arc::new(crate::function_table::caller_argument_positions(module));
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel,
        functions,
        classes: Arc::new(classes),
        datetime_imports: Arc::new(crate::expressions::datetime_imports(module)),
        module_bindings: surface.bindings,
        module_callable_returns,
        strict_int_aliases: &strict_int_aliases,
        typed_dicts,
        caller_arguments,
        entry_directory: entry_directory.map(|dir| dir.to_path_buf()),
        evaluations_recorder: None,
    };
    let mut out = Vec::new();
    walk_body(&module.body, None, None, None, &context, &mut out);
    out
}

/// The refined set stated or known at one position in the module — the
/// LSP hover's own query. A STATED refinement first (a parameter's own
/// annotation, or the enclosing function's `-> Annotation` when the
/// position sits inside the `returns` node): the developer wrote this
/// claim, so it answers before anything the flow walk derives, the
/// same preference order refined-ts-go's own `AnswerAt` keeps between
/// a stated annotation and `FlowAnswerAt`'s derived knowledge
/// (service/hover_provider.go). Where nothing is stated, the walk runs
/// with recording enabled (`WalkContext::evaluations_recorder`) and
/// the answer is the SMALLEST recorded expression range containing
/// `position` — the innermost node the walk evaluated there, mirroring
/// how a hover always names the tightest enclosing expression rather
/// than an outer one that merely contains it. `None` where neither
/// says anything: an unreadable module (no `type` alias AND no
/// recognized `Annotated` import — the same early exit
/// `findings_for_module_at` takes), a position outside every annotation
/// and outside every recorded node, or a recorded value this table
/// cannot read back as a set (`abstract_value_as_refined_set`'s own
/// doc).
pub fn refined_set_at_position(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    position: TextSize,
) -> Option<RefinedSet> {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    if aliases.is_empty()
        && imports.annotated_names.is_empty()
        && imports.literal_names.is_empty()
    {
        // Same gate `findings_for_module_at` takes, and for the same
        // reason: a module with no `type` alias, no recognized
        // `Annotated` import, and no `Literal` import carries no
        // refinement vocabulary at all.
        return None;
    }
    if let Some(stated) = stated_refinement_at(module, &aliases, &imports, position) {
        return Some(stated);
    }
    let surface = module_surface(module, resolver, kernel, IMPORT_DEPTH_CAP);
    let own_functions = function_table(module);
    let functions = Arc::new(merged(&own_functions, surface.functions.as_ref()));
    let own_classes = class_table(module, &aliases, &imports, kernel);
    let mut classes = Arc::try_unwrap(surface.classes)
        .unwrap_or_else(|_| panic!("module_surface's own Arc<classes> has no other owner yet"));
    for (name, model) in own_classes {
        classes.insert(name, model);
    }
    for def in module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    }) {
        for (name, model) in local_class_table(&def.body, &aliases, &imports, kernel) {
            classes.insert(name, model);
        }
    }
    let module_callable_returns = Arc::new(module_level_callable_returns(module, &aliases, &imports));
    let strict_int_aliases = strict_int_alias_names(module);
    let typed_dicts = Arc::new(instances::typed_dict_table(module, &aliases, &imports));
    let caller_arguments = Arc::new(crate::function_table::caller_argument_positions(module));
    let recorder = Arc::new(Mutex::new(Vec::new()));
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel,
        functions,
        classes: Arc::new(classes),
        datetime_imports: Arc::new(crate::expressions::datetime_imports(module)),
        module_bindings: surface.bindings,
        module_callable_returns,
        strict_int_aliases: &strict_int_aliases,
        typed_dicts,
        caller_arguments,
        entry_directory: None,
        evaluations_recorder: Some(recorder.clone()),
    };
    let mut discarded_findings = Vec::new();
    walk_body(&module.body, None, None, None, &context, &mut discarded_findings);
    let recorded = recorder
        .lock()
        .expect("evaluations recorder poisoned by an earlier panic")
        .clone();
    smallest_covering_set(&recorded, position)
}

/// A declared refinement stated exactly AT `position`: a function
/// parameter's own annotation (`declared_refinement`, the same read
/// `check.rs::seed_parameters` uses to seed it), a function's own
/// `-> Annotation` when `position` sits inside the `returns` node
/// (`declared_refinement`, falling back to `base_sort_return_refinement`
/// — the same two-reader fallback chain `walk_function_def` already
/// runs to build a body's own `return_refinement`, minus the
/// `typed_dict_return_refinement` arm, which needs `context.typed_dicts`
/// and is not in scope here), a `def`'s own NAME (`declared_refinement`
/// on `def.returns` ALONE — no `base_sort_return_refinement` fallback,
/// so a bare `-> float`/`-> int`/`-> str` answers nothing at the name,
/// exactly as `declared_refinement`'s own doc says a base sort states
/// nothing this table reads — the same reader `return_refinement`
/// itself is built from at every `walk_function_def`/`walk_method_def`
/// call site), or a module-level ALIAS DECLARATION's own name (`type X
/// = …`, `X = Annotated[…]`, `X: TypeAlias = Annotated[…]` — the three
/// spellings `compile_aliases` already reads): the alias's own compiled
/// set from `aliases`, the SAME table `declared_refinement`'s own
/// `Expr::Name` arm consults for a parameter spelled with that alias —
/// one resolution mechanism, not a re-read of the RHS. Recurses into
/// every nested `def`/`class` body so the INNERMOST enclosing
/// construct's own name/parameter/return wins — the same "innermost
/// node" preference the derived-flow fallback keeps for a recorded
/// range.
fn stated_refinement_at(
    module: &ModModule,
    aliases: &HashMap<String, AliasEntry>,
    imports: &crate::surface::SurfaceImports,
    position: TextSize,
) -> Option<RefinedSet> {
    // Alias-declaration names are read ONLY at the module's own top
    // level — the same scope `compile_aliases` reads them from. A
    // function body statement of the same shape (`Age = 40` inside a
    // `def`, rebinding a name that happens to match a module-level
    // alias) is a LOCAL rebinding, not a declaration, and must not
    // answer the alias's set — the same rebinding hazard
    // `declared_refinement`'s own `Expr::Name` arm guards against via
    // `environment.alias_is_visible`.
    for stmt in &module.body {
        if let Some(set) = alias_declaration_set_at(stmt, aliases, position) {
            return Some(set);
        }
    }
    stated_refinement_in_body(&module.body, aliases, imports, position)
}

fn stated_refinement_in_body(
    body: &[Stmt],
    aliases: &HashMap<String, AliasEntry>,
    imports: &crate::surface::SurfaceImports,
    position: TextSize,
) -> Option<RefinedSet> {
    for stmt in body {
        match stmt {
            Stmt::FunctionDef(def) if def.range.contains_inclusive(position) => {
                // the innermost enclosing def wins: a position inside a
                // NESTED def's own body is checked against that nested
                // def's own parameters/return first
                if let Some(inner) = stated_refinement_in_body(&def.body, aliases, imports, position) {
                    return Some(inner);
                }
                let outer_environment = Environment::new(HashSet::new());
                if def.name.range.contains_inclusive(position) {
                    // the def's own name: what a call to it yields, when
                    // that is readable at all — never the base-sort
                    // fallback, so a bare `-> float` states nothing here
                    return def
                        .returns
                        .as_deref()
                        .and_then(|returns| declared_refinement(returns, aliases, imports, &outer_environment))
                        .map(|declared| declared.set);
                }
                for parameter in def
                    .parameters
                    .posonlyargs
                    .iter()
                    .chain(def.parameters.args.iter())
                    .chain(def.parameters.kwonlyargs.iter())
                {
                    let Some(annotation) = parameter.parameter.annotation.as_deref() else {
                        continue;
                    };
                    if annotation.range().contains_inclusive(position) {
                        return declared_refinement(annotation, aliases, imports, &outer_environment)
                            .map(|declared| declared.set);
                    }
                }
                if let Some(returns) = def.returns.as_deref() {
                    if returns.range().contains_inclusive(position) {
                        return declared_refinement(returns, aliases, imports, &outer_environment)
                            .or_else(|| base_sort_return_refinement(returns))
                            .map(|declared| declared.set);
                    }
                }
                return None;
            }
            Stmt::ClassDef(def) if def.range.contains_inclusive(position) => {
                return stated_refinement_in_body(&def.body, aliases, imports, position);
            }
            _ => {}
        }
    }
    None
}

/// `stmt`'s own alias-declaration NAME, when it is one of the three
/// spellings `compile_aliases` reads (`type X = …`, `X = Annotated[…]`,
/// `X: TypeAlias = Annotated[…]`) AND `position` sits exactly on that
/// name AND `aliases` actually compiled an entry for it — reads
/// `aliases[name].set` directly rather than re-lowering the RHS, so
/// this answers the IDENTICAL set a parameter annotated with the same
/// alias name already gets from `declared_refinement`'s own
/// `Expr::Name` arm. `None` for every other statement shape, a
/// position elsewhere on the line, or a name `compile_aliases` could
/// not lower (the declaration states nothing this table reads, same as
/// any other unlowerable alias).
fn alias_declaration_set_at(stmt: &Stmt, aliases: &HashMap<String, AliasEntry>, position: TextSize) -> Option<RefinedSet> {
    let name = match stmt {
        Stmt::TypeAlias(alias) => match alias.name.as_ref() {
            Expr::Name(name) => name,
            _ => return None,
        },
        Stmt::Assign(assign) => match assign.targets.as_slice() {
            [Expr::Name(name)] => name,
            _ => return None,
        },
        Stmt::AnnAssign(annotated) => match annotated.target.as_ref() {
            Expr::Name(name) => name,
            _ => return None,
        },
        _ => return None,
    };
    if !name.range().contains_inclusive(position) {
        return None;
    }
    aliases.get(name.id.as_str()).map(|entry| entry.set.clone())
}

/// The smallest recorded `(range, value)` whose range contains
/// `position`, converted to a `RefinedSet` — the derived-flow fallback
/// once `stated_refinement_at` finds nothing declared there. "Smallest"
/// rather than "first" or "last": several recorded nodes can nest around
/// one position (`total / len(samples)` records the division AND both
/// operands), and the hover should name the innermost one, the same
/// preference a source-map lookup or an LSP's own token-at-position
/// query keeps. `None` when no recorded range covers the position, or
/// the covering value's own kind cannot be read back as a set
/// (`abstract_value_as_refined_set`'s own doc).
fn smallest_covering_set(recorded: &[(TextRange, AbstractValue)], position: TextSize) -> Option<RefinedSet> {
    let mut best: Option<&(TextRange, AbstractValue)> = None;
    for entry in recorded {
        let (range, _) = entry;
        if !range.contains_inclusive(position) {
            continue;
        }
        match best {
            Some((best_range, _)) if best_range.len() <= range.len() => {}
            _ => best = Some(entry),
        }
    }
    let (_, value) = best?;
    abstract_value_as_refined_set(value)
}

/// The refined set a recorded `AbstractValue` states, or `None` when
/// its kind carries no set this table can hand back. `Kind::Set`
/// carries one directly; `Kind::Values` (a scalar literal join —
/// `known_values`'s own shape) is read back as the `one_of` of exactly
/// those values, matching the singleton-membership set the SAME values
/// would compile to if the developer had written them as a `Literal[
/// ...]` annotation. Every other kind (an object, a variable, a
/// callable, …) declines rather than approximate — this table states a
/// REFINED SET, not a general value description.
fn abstract_value_as_refined_set(value: &AbstractValue) -> Option<RefinedSet> {
    match value.kind {
        Kind::Set => Some(value.set.clone()),
        Kind::Values if !value.values.is_empty() => {
            Some(make_refined_set(vec![one_of(&value.values)]))
        }
        _ => None,
    }
}

/// Every top-level `name: Callable[[...], R] [| None] = ...` at the
/// module's own body — b-body-expressions.py's own
/// `maybe_next_year: Callable[[int], int] | None = None` shape — read
/// through `typereading::callable_return_refinement` against a
/// no-locals environment (module-level names are never "locally
/// rebound" at the point this table is built; a body that DOES rebind
/// one shadows it in that body's own environment the same way
/// `alias_is_visible` already shadows an alias name). A name whose
/// annotation is not this shape is simply absent — absence declines
/// judgment, it never approximates.
fn module_level_callable_returns(
    module: &ModModule,
    aliases: &HashMap<String, AliasEntry>,
    imports: &crate::surface::SurfaceImports,
) -> HashMap<String, DeclaredRefinement> {
    let no_locals = Environment::new(HashSet::new());
    let mut out = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::AnnAssign(assign) = stmt else {
            continue;
        };
        let Expr::Name(target_name) = assign.target.as_ref() else {
            continue;
        };
        if let Some(declared) =
            callable_return_refinement(assign.annotation.as_ref(), aliases, imports, &no_locals)
        {
            out.insert(target_name.id.as_str().to_owned(), declared);
        }
    }
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
///
/// BODY-LOCAL CLASSES: `class_table`'s own module-level scan
/// (`instances.rs`) reads only `module.body`'s own top-level
/// `StmtClassDef`s, so a class defined INSIDE a function body (or a
/// method body) is invisible to `context.classes` — the same gap
/// `local_function_table` already closes for a body-local `def`.
/// `local_class_table(body, ...)` mirrors that construction (wrapping
/// this body's own top-level `Stmt::ClassDef`s in a synthetic
/// `ModModule` and reusing `instances::class_table`'s one public
/// constructor) and is merged over `context.classes` here, local name
/// winning on a spelling collision — the same base-wins rule
/// `function_table::merged` already applies. A class nested inside one
/// of THIS body's own local classes is not itself walked as a
/// top-level entry (the same one-level rule `local_function_table`
/// keeps for a nested `def`).
///
/// SELF-SEEDING: `self_model` is `Some(class)` only when this body IS a
/// class body being walked by `walk_class_def` for `class` itself —
/// `None` everywhere else (a plain function, the module body, or any
/// nested body reached through `walk_statement`'s ordinary recursion).
/// When `Some`, this function's own per-statement loop below walks a
/// top-level member `def` whose first parameter is `self` through
/// `walk_method_def` instead of the ordinary `walk_statement` dispatch,
/// seeding `self` with an instance built from `class`'s own declared
/// fields (datamodel.rst, "Instance methods": "the special thing about
/// methods is that the instance object is prepended to the argument
/// list" — the receiver is a real value at every method call, so a
/// body that reads `self.<field>` before typereading can prove which
/// concrete instance called it still has SOMETHING sound to read: the
/// class's own declared shape). Every other statement (a class-body
/// `AnnAssign` field, a nested `class`, an `if`, …) still walks through
/// the ordinary `walk_statement` dispatch, unaffected.
fn walk_body(
    body: &[Stmt],
    parameters: Option<&Parameters>,
    return_refinement: Option<&DeclaredRefinement>,
    self_model: Option<&ClassModel>,
    context: &WalkContext,
    out: &mut Vec<Finding>,
) {
    walk_body_with_self_binding(body, parameters, return_refinement, None, self_model, None, None, None, None, context, out);
}

/// The statement position whose walk must publish a recognized foreign
/// edge's return fact — the STATEMENT HOLDING the edge's own
/// `json.loads(...)` node, found by RANGE CONTAINMENT among the
/// statements after the call at `call_position` (the sole consumer may
/// sit any number of statements later, never assumed to be the very
/// next one). `None` when no later statement in `body` contains
/// `parse_range` at all — the edge recognized a crossing call whose
/// result is never actually parsed anywhere in this body, so there is
/// no position to key an entry under and the recognition publishes
/// nothing.
///
/// Pulled out of `walk_body_with_self_binding`'s per-statement loop so
/// the position math is independently testable: two recognized calls in
/// the same body resolve to two INDEPENDENT positions here (this
/// function reads `body` fresh each call and returns a plain `usize`),
/// which is what lets the caller key `foreign_edge_overrides` per call
/// rather than through one shared slot.
fn foreign_edge_consumer_position(body: &[Stmt], call_position: usize, parse_range: TextRange) -> Option<usize> {
    body.iter()
        .enumerate()
        .skip(call_position + 1)
        .find(|(_, following)| following.range().contains_range(parse_range))
        .map(|(index, _)| index)
}

/// Tries `foreign_edge::foreign_edge_at` at `position` in `body` — the
/// SAME `Stmt::Assign`/`Stmt::With` gate and outcome handling
/// `walk_body_with_self_binding`'s own per-statement loop already runs,
/// pulled out so every statement list the checker walks (a function's own
/// top-level body, an `if`-arm's body, a `with`-block's body) offers this
/// recognition the same way, never a second mechanism. A recognized
/// `Override` is keyed into `foreign_edge_overrides` under its own
/// consumer position (`foreign_edge_consumer_position`'s own doc); a
/// `Fired` records an RTS7001 directly; a `Decline` records this body's
/// one blocker. A statement that is not this shape, or that
/// `foreign_edge_at` does not recognize at all, is a no-op — the caller's
/// own walk of `stmt` is untouched either way.
fn serve_foreign_edge_at(
    body: &[Stmt],
    position: usize,
    environment: &Environment,
    context: &WalkContext,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
    foreign_edge_overrides: &mut HashMap<usize, Vec<(TextRange, AbstractValue)>>,
) {
    if !matches!(body[position], Stmt::Assign(_) | Stmt::With(_)) {
        return;
    }
    let Some(outcome) =
        foreign_edge::foreign_edge_at(body, position, environment, context.kernel, context.entry_directory.as_deref())
    else {
        return;
    };
    match outcome {
        foreign_edge::ForeignEdgeOutcome::Override { parse_range, value, stdout_override } => {
            // The override applies at the STATEMENT HOLDING the
            // `json.loads(...)` node — the sole consumer may sit any
            // number of statements after the call, so find it by range
            // containment rather than assuming a position. Keyed by
            // that consumer's own position: an earlier still-pending
            // entry for a DIFFERENT consumer position is untouched by
            // this insert, so two recognized calls before either is
            // consumed each keep their own entry.
            if let Some(consumer_position) = foreign_edge_consumer_position(body, position, parse_range) {
                let mut published = vec![(parse_range, value)];
                // `stdout_override`'s own node (a `result.stdout`
                // attribute access, or the bound name's own node) is a
                // SUB-NODE of the SAME `json.loads(...)` statement
                // `parse_range` already resolved a position for, so it
                // always shares that position — `foreign_edge_consumer_
                // position` is not asked a second time.
                if let Some((stdout_range, stdout_value)) = stdout_override {
                    published.push((stdout_range, stdout_value));
                }
                foreign_edge_overrides.insert(consumer_position, published);
            }
        }
        foreign_edge::ForeignEdgeOutcome::Fired { message, range, consumer } => {
            out.push(Finding { range, code: "RTS7001", message });
            // The TS convention taken literally: an outbound fire and a
            // bound return fact are independent truths, so the sole
            // consumer of a fired FileRead edge still binds the
            // artifact's own return-leg value (carried on the Fired
            // outcome, built by the same reader the serving path uses)
            // and judges it for real — one edge, its fire, and a
            // determined read.
            if let Some((consumer_range, value)) = consumer {
                if let Some(consumer_position) = foreign_edge_consumer_position(body, position, consumer_range) {
                    foreign_edge_overrides.insert(consumer_position, vec![(consumer_range, value)]);
                }
            }
        }
        foreign_edge::ForeignEdgeOutcome::Decline { message, range } => {
            record_blocker(blocked, range, message, out);
        }
    }
    // recognized whole (an override, a fire, or a named decline) — the
    // statement itself still walks ordinarily afterward so `<name>`
    // binds through the usual Assign path; only the LATER `json.loads`
    // node's value is overridden.
}

/// `walk_body`'s full construction, plus one extra optional step:
/// `self_binding`, when `Some`, binds the name `self` to that value
/// AFTER parameter seeding — `walk_method_def`'s own seam into this
/// function, so a `self.<field>` read inside a method body reaches
/// `evaluate_attribute_read`'s tagged-instance path
/// (`instances::field_read_through_model`) instead of reading an
/// unbound name. `self` carries no annotation in the corpus's own
/// convention, so `seed_parameters` never seeds it itself — this bind
/// is the only writer for that name at body entry.
///
/// `returned_values_out`, when `Some`, is filled with every value this
/// body's `return` statements produced — the SAME values `walk_return`
/// judges, recorded through `env::collect_returned_values` rather than
/// re-derived. `fact_export` is the only caller that asks; every walk in
/// the checker itself passes `None` and behaves exactly as before.
///
/// `enclosing_def_name`, when `Some`, is the MODULE-LEVEL def this body
/// belongs to — `seed_parameters`'s own key into `context.caller_arguments`
/// for an unannotated-parameter caller join. `None` for the module body
/// itself and for a nested/method def (neither is a module-level def
/// `function_table::caller_argument_positions` ever indexes), which
/// simply seeds no caller-joined parameter, matching today's plain
/// Unknown behavior.
///
/// `bare_sort_return_refinement`, when `Some`, is `typereading::
/// base_sort_return_refinement` read off this def's own `-> Annotation`
/// (a bare `int`/`float`/`str`, no `Annotated[...]` refinement) —
/// `None` for every other return shape, including no annotation at all.
/// This is NEVER `return_refinement` itself (a bare-sort return still
/// judges nothing at an ORDINARY return, exactly as today — widening
/// that would turn every unreadable `-> int` helper body into a new
/// undetermined blocker, `typereading::base_sort_return_refinement`'s own
/// doc). It exists ONLY so the per-statement loop below can judge a
/// RECOGNIZED foreign-edge crossing's return leg against the sort even
/// when no refinement narrows it — the one place a bare sort is worth
/// something (sort admission and refinement wideness against a fact the
/// crossing itself proved), read at exactly the position `foreign_edge_
/// overrides` already names, never at any other return in this body.
fn walk_body_with_self_binding(
    body: &[Stmt],
    parameters: Option<&Parameters>,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    self_model: Option<&ClassModel>,
    self_binding: Option<&AbstractValue>,
    returned_values_out: Option<&mut Vec<AbstractValue>>,
    enclosing_def_name: Option<&str>,
    bare_sort_return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    out: &mut Vec<Finding>,
) {
    let mut locally_bound = locally_bound_names(body);
    if let Some(parameters) = parameters {
        collect_parameter_names(parameters, &mut locally_bound);
    }
    let mut environment = Environment::new(locally_bound);
    if returned_values_out.is_some() {
        environment.collect_returned_values();
    }
    // Shares the module walk's ONE evaluations recorder (if the caller
    // asked for one) onto THIS body's fresh environment — the seam
    // `refined_set_at_position` depends on: a nested `def`'s own body
    // gets its own fresh `Environment` here, so without this the
    // recordings from every body but the outermost would be lost.
    if let Some(recorder) = context.evaluations_recorder.clone() {
        environment.set_evaluations_recorder(recorder);
    }
    environment.set_functions(Arc::new(merged(&local_function_table(body), &context.functions)));
    environment.set_classes(merged_classes_for_body(body, context));
    environment.set_datetime_imports(context.datetime_imports.clone());
    if let Some(entry_directory) = context.entry_directory.clone() {
        environment.set_entry_directory(Arc::new(entry_directory));
    }
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
    // Every visible CLASS name seeds its class-object value too — the
    // shadow-on-rebind rule module_bindings takes — so a function body's
    // `Counted.total = 200` write and `Counted.total` read see the class
    // object without a construction anywhere in the body. Calling the
    // seeded name still constructs: the construction gates recognize a
    // name bound to its OWN class object (source == the class name).
    {
        let class_names: Vec<String> = environment
            .classes()
            .map(|classes| classes.keys().cloned().collect())
            .unwrap_or_default();
        for name in class_names {
            if environment.alias_is_visible(&name) && environment.read(&name).is_none() {
                let model = environment
                    .classes()
                    .and_then(|classes| classes.get(&name))
                    .expect("name came from this same table");
                let value = instances::class_object_value(model);
                environment.bind(&name, value);
            }
        }
    }
    if let Some(parameters) = parameters {
        seed_parameters(parameters, enclosing_def_name, context, &mut environment);
        // `*args`/`**kwargs`'s own names — a bare-Name forward of either
        // (`f(*args)`, `f(**kwargs)`) hands CPython exactly what THIS
        // body itself received, never an independently-grown collection
        // (`expressions.rs::call_provable_raise`'s own "unbounded
        // starred argument" check reads this set to stay silent on a
        // ParamSpec-forwarding row like r-ast-census.py's `wrapper`).
        let mut variadic_names = HashSet::new();
        if let Some(vararg) = parameters.vararg.as_ref() {
            variadic_names.insert(vararg.name.id.as_str().to_owned());
        }
        if let Some(kwarg) = parameters.kwarg.as_ref() {
            variadic_names.insert(kwarg.name.id.as_str().to_owned());
        }
        environment.set_variadic_parameter_names(Arc::new(variadic_names));
    }
    if let Some(self_value) = self_binding {
        environment.bind("self", self_value.clone());
    }
    // This body's own CALLABLE-RETURN table, seeded from the module's
    // top-level callable declarations — the same shadow-on-rebind rule
    // `module_bindings` above takes (a body that locally rebinds the
    // name is not seeded with the module-level entry). `walk_ann_assign`
    // grows this table as a body-local `Callable[...]`-typed variable is
    // walked, republishing it onto `environment` itself (rather than a
    // sibling parameter threaded through every statement form the way
    // `aug_assign_refinements` is) so `sink_value`'s call-site read —
    // reachable from every nested branch/loop/match/with/try arm through
    // `environment` alone — sees each new entry as soon as it is walked,
    // with no signature change anywhere along that dispatch tree.
    let module_callable_returns: HashMap<String, DeclaredRefinement> = context
        .module_callable_returns
        .iter()
        .filter(|(name, _)| environment.alias_is_visible(name))
        .map(|(name, declared)| (name.clone(), declared.clone()))
        .collect();
    if !module_callable_returns.is_empty() {
        environment.set_callable_returns(Arc::new(module_callable_returns));
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
    // The positions of the statements a relational sum already folded
    // into its kernel program — the division alone, or a count-alias
    // assignment plus the division that reads it — as a half-open
    // range. `position+1..position+1` (empty) while there is none,
    // since no statement position falls inside an empty range.
    let mut folded_division_at = usize::MAX..usize::MAX;
    // Every foreign-edge return fact still waiting to be published,
    // keyed by the POSITION of the statement holding its own
    // `json.loads(...)` node — never a single slot. Unlike a relational
    // sum's quotient (always the very next statement), the sole
    // `json.loads` consumer a foreign edge finds may sit several
    // statements after the call, so each entry rides across more than
    // one loop iteration rather than being cleared unconditionally
    // after one walk. A body can make more than one recognized crossing
    // call before either result is consumed (the diamond shape: two
    // `subprocess.run` calls, then two `json.loads` reads) — with a
    // single slot, the second recognition would clobber the first's
    // still-pending entry before its own consumer is ever reached, so
    // the map keeps them independently: one entry per consumer
    // position, applied via `environment.set_evaluated_node` only at
    // its own matching `position`, and removed the moment that
    // position's walk runs (an entry whose consumer position never
    // recurs — the same body never reaches it again — simply stays in
    // the map unconsumed and unread, exactly as inert as the single
    // slot's own unconsumed `None` was).
    let mut foreign_edge_overrides: HashMap<usize, Vec<(TextRange, AbstractValue)>> = HashMap::new();
    for (position, stmt) in body.iter().enumerate() {
        if let (Some(class), Stmt::FunctionDef(def)) = (self_model, stmt) {
            if is_self_method(def) {
                walk_method_def(def, class, context, out);
                continue;
            }
        }
        // RELATIONAL SUM: an accumulation over a sequence known only by
        // its element set — either spelling, `total = 0; for x in xs:
        // total += f(x)` or `total = sum(f(x) for x in xs)` — followed by
        // a division of that total by the same sequence's length.
        // Interval division answers this far too weakly (`[0, n] / [1,
        // n]` is `[0, n]`), so the accumulation and the division lower
        // into ONE kernel program where the linear decider ties the
        // total to the count. Recognized here rather than inside
        // `walk_loop` because the division sits in the FOLLOWING
        // statement, which only this statement driver can see. Declining
        // leaves the statement to `walk_statement` exactly as before.
        // A folded division was already walked as part of the kernel
        // program, so the statement itself is skipped rather than
        // re-walked into a second, weaker binding of the same name.
        if folded_division_at.contains(&position) {
            continue;
        }
        // FOREIGN EDGE: `<name> = subprocess.run(["node", "<script>.ts"],
        // input=json.dumps(...), capture_output=True, text=True)` followed
        // by `json.loads(<name>.stdout)` — a cross-language call whose
        // argv NAMES the TypeScript code that runs next, so the checker
        // reads that target's own exported fact and attaches it to the
        // parse node. Tried BEFORE `recognize_generator_sum` below: both
        // recognize a shape at an `Assign`, and the foreign edge's own
        // shape (a `subprocess.run` call) is the more specific of the two,
        // so it must not be shadowed by a generator-sum match that could
        // never apply to a `subprocess.run` value anyway — the fixed
        // order this mission calls for once two Assign-shaped two-
        // statement recognizers coexist. A `With` statement also enters:
        // the temp-file carrier's unit starts at the tempfile with-block
        // (recognize_temp_file_edge), not at an Assign.
        serve_foreign_edge_at(body, position, &environment, context, &mut blocked, out, &mut foreign_edge_overrides);
        let recognized = match stmt {
            Stmt::For(for_stmt) if for_stmt.orelse.is_empty() => {
                relational_sum::recognize_accumulation(for_stmt, &environment)
                    .map(|recognized| (recognized, Some(for_stmt.target.as_ref()), None))
            }
            Stmt::Assign(assign) => relational_sum::recognize_generator_sum(assign, &environment)
                .map(|recognized| (recognized, None, Some(assign.range()))),
            _ => None,
        };
        if let Some((recognized, loop_target, bound_at)) = recognized {
            match walk_relational_sum(
                recognized,
                loop_target,
                bound_at,
                &body[position + 1..],
                &mut environment,
            ) {
                RelationalSum::Declined => {}
                RelationalSum::Consumed => continue,
                RelationalSum::ConsumedWithDivision { skip_statements } => {
                    folded_division_at = (position + 1)..(position + 1 + skip_statements);
                    continue;
                }
            }
        }
        // A foreign edge's return fact is published for the STATEMENT
        // HOLDING its `json.loads(...)` node, wherever that sits later in
        // this body — looked up by THIS position, published immediately
        // before that one statement's walk, and removed from the map
        // right after (the same scoping obligation a relational sum's
        // quotient keeps for the very next statement, widened to
        // whichever position matches). A different pending entry, keyed
        // under a different position, is untouched here — each entry is
        // published only at its own consumer, never at another one's.
        let at_recognized_crossing = foreign_edge_overrides.contains_key(&position);
        if let Some(published) = foreign_edge_overrides.get(&position) {
            environment.set_evaluated_node(published.clone());
        }
        // RELATIONAL SUM AT A BARE RETURN: `return sum(<elt> for <var>
        // in <seq>)` with no assignment in the body at all — the
        // generator-sum recognizer above only ever reads an `Assign`, so
        // this single-statement spelling needs its own publish before
        // the return walks. A decline publishes nothing and the return
        // below evaluates exactly as it already did.
        if let Stmt::Return(ret) = stmt {
            publish_relational_sum_return(ret, &mut environment);
        }
        // BARE-SORT RETURN AT A RECOGNIZED CROSSING: `return_refinement`
        // stays `None` for every ORDINARY bare-sort return (unchanged —
        // the general table deliberately does not read base sorts), but
        // THIS statement is the one a recognized foreign edge just
        // published its own proved fact onto (`at_recognized_crossing`),
        // so the sort itself is worth judging here: a declared `-> float`
        // still refuses a value the crossing's own fact places outside
        // the float ray (there is none — `numbers()` is the whole real
        // line — but `-> int`/`-> str` genuinely narrow), and a widened
        // corner the crossing's own `Optional` fact admits still passes.
        // No other return in this body is affected: the fallback applies
        // ONLY at this one position, and ONLY when `return_refinement`
        // itself named nothing.
        let effective_return_refinement = if at_recognized_crossing && return_refinement.is_none() {
            bare_sort_return_refinement
        } else {
            return_refinement
        };
        let terminates = walk_statement(
            stmt,
            effective_return_refinement,
            yield_refinement,
            context,
            &mut environment,
            &mut aug_assign_refinements,
            &mut provably_unbound,
            &mut blocked,
            out,
        );
        // A relational sum publishes its quotient for the NEXT
        // statement's one division node; a foreign edge's own publish
        // (above) is likewise scoped to exactly the one statement whose
        // walk just ran — either way, the publication ends here and no
        // later node can match it.
        environment.set_evaluated_node(Vec::new());
        // This position's own entry (if it had one) expires the moment
        // its statement's walk runs — the same way the single slot's
        // `None` reset did, widened to a per-key removal so a DIFFERENT
        // still-pending entry (a later consumer this walk has not
        // reached yet) stays in the map rather than being cleared
        // wholesale.
        foreign_edge_overrides.remove(&position);
        // A `try` whose every arm provably raises or returns leaves
        // nothing that reaches whatever follows it in THIS body — the
        // same "unreachable code past a terminal statement" rule
        // `arm_terminates` already applies to a bare `return`/`raise`,
        // extended here to a try/except construct whose own termination
        // is proved rather than syntactic. Stopping here is what keeps a
        // read past the try from reporting an unreadable-value blocker
        // for a name only the (unreachable) fall-through path would have
        // left unbound.
        if terminates {
            break;
        }
    }
    if let Some(returned_values_out) = returned_values_out {
        *returned_values_out = environment
            .returned_values()
            .expect("the recorder was installed above whenever this slot is Some");
    }
}

/// BODY-LOCAL CLASSES, merged: `local_class_table`'s own build for
/// `body`, layered over `context.classes` (local name wins on a
/// spelling collision, the same base-wins rule `function_table::merged`
/// already applies) — returns `context.classes.clone()` UNCHANGED (an
/// `Arc` clone, no allocation) when `body` declares no local class at
/// all, so the common case (no body-local classes) costs nothing beyond
/// the empty scan.
fn merged_classes_for_body(body: &[Stmt], context: &WalkContext) -> Arc<HashMap<String, ClassModel>> {
    let local_classes = local_class_table(body, context.aliases, context.imports, context.kernel);
    if local_classes.is_empty() {
        return context.classes.clone();
    }
    let mut merged_classes = (*context.classes).clone();
    for (name, model) in local_classes {
        merged_classes.insert(name, model);
    }
    Arc::new(merged_classes)
}

/// LOCAL CLASSES: this body's own top-level `class`s, read through
/// `instances::class_table`'s one public constructor over a synthetic
/// `ModModule` wrapping just those definitions — the exact construction
/// `local_function_table` already uses for a body-local `def`
/// (`cross_module.rs`'s `synthetic_module` pattern). Parent-linking via
/// `super().__init__(...)` only resolves against another class in the
/// SAME synthetic table, so a body-local class naming a MODULE-level
/// class as its base is read parent-less here — an acceptable narrowing
/// for a shape outside this wave's fixture rows, not a soundness gap
/// (a parent-less child still reads its own AnnAssign/`__init__`
/// fields correctly, only the inherited-field merge is skipped).
///
/// A class nested inside a NESTED `def` (`body`'s own top-level `def`
/// whose body declares a class one level further down — a nested
/// closure-factory shape returning an instance of a class local to
/// itself) is collected too: every top-level `Stmt::FunctionDef`'s body
/// is scanned the same way, recursively, so a class declared at any
/// nesting depth of nested defs is visible once its instance crosses
/// back out to an outer scope. A direct top-level class NAME wins over
/// a same-named class found one level deeper (the nearer scope shadows
/// the farther one, Python's own scoping rule).
fn local_class_table(
    body: &[Stmt],
    aliases: &HashMap<String, AliasEntry>,
    imports: &crate::surface::SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
) -> HashMap<String, ClassModel> {
    let local_defs = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::ClassDef(def) => Some(Stmt::ClassDef(def.clone())),
            _ => None,
        })
        .collect();
    let synthetic = ModModule {
        node_index: AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: local_defs,
    };
    let mut classes = class_table(&synthetic, aliases, imports, kernel);
    for stmt in body {
        if let Stmt::FunctionDef(def) = stmt {
            let nested = local_class_table(&def.body, aliases, imports, kernel);
            for (name, model) in nested {
                classes.entry(name).or_insert(model);
            }
        }
    }
    classes
}

/// Whether `def`'s first parameter is named `self` — the corpus's own
/// receiver-naming convention (`instances.rs`'s `self_attribute_name`
/// doc makes the same assumption). A member `def` with no parameter at
/// all (a `@staticmethod`, out of this wave's scope) is not a bound
/// instance method this seeding law applies to.
fn is_self_method(def: &StmtFunctionDef) -> bool {
    def.parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .next()
        .is_some_and(|parameter| parameter.parameter.name.id.as_str() == "self")
}

/// A class-body member `def` whose first parameter is `self`: walks
/// exactly like `walk_function_def` (its own `-> Annotation` reads
/// against the OUTER environment, its body walks fresh through
/// `walk_body`), except `self` seeds an INSTANCE built from `class`'s
/// own declared fields — `judge_construction`'s own construction path,
/// called with NO arguments so every field takes its default when
/// present, else its declared set (`known_set`, TrustSpec — the same
/// "declared set stands in for an unread value" law `seed_parameters`
/// already applies to an ordinary parameter), else `unknown()`. This is
/// the METHOD's own declared shape, not the value any particular call
/// site constructed with — sound because a method body reads `self`
/// long before this checker can know which call site reached it;
/// `judge_construction`'s own fires are discarded here (a field outside
/// its declared set is this synthesized self's own business, never a
/// finding — the mission's fires belong to an actual construction/write
/// site, not this seeding).
fn walk_method_def(def: &StmtFunctionDef, class: &ClassModel, context: &WalkContext, out: &mut Vec<Finding>) {
    let outer_environment = Environment::new(HashSet::new());
    let return_refinement = def.returns.as_deref().and_then(|annotation| {
        declared_refinement(annotation, context.aliases, context.imports, &outer_environment)
            .or_else(|| typed_dict_return_refinement(annotation, &context.typed_dicts))
    });
    let (return_refinement, yield_refinement) = generator_body_refinements(def, return_refinement);
    let bare_sort_return_refinement = def.returns.as_deref().and_then(base_sort_return_refinement);
    let self_instance = judge_construction(class, &[], &[], context.kernel).instance;
    walk_body_with_self_binding(
        &def.body,
        Some(def.parameters.as_ref()),
        return_refinement.as_ref(),
        yield_refinement.as_ref(),
        None,
        Some(&self_instance),
        None,
        None,
        bare_sort_return_refinement.as_ref(),
        context,
        out,
    );
}

/// Splits a `def`'s own resolved `-> Annotation` refinement into the two
/// checked positions its BODY judges against, once the body is
/// GENERATOR-shaped (`is_generator_shaped`'s own doc — a `yield`
/// anywhere, straight-line or one level inside a `for`/`async for`).
/// `Generator[Y, S, R]`/`AsyncGenerator[Y, S]`/`Iterator[Y]`/`Iterable[Y]`
/// carry their two positions in `DeclaredRefinement::generator`
/// (`typereading.rs`'s own doc); every `yield <expr>` in this body
/// judges against `generator.yield_type`, every `return <expr>` against
/// `generator.return_type` (`None` for `AsyncGenerator`/`Iterator`/
/// `Iterable` — no return type is judged there at all, the same "no
/// annotation → no judging" rule `walk_return` already states). A
/// NON-generator body, or a generator-shaped body whose own `->
/// Annotation` did not read as one of the four generator forms
/// (`generator` is `None`), returns `declared` UNCHANGED as the return
/// position and no yield position — ordinary Python, nothing new judges.
fn generator_body_refinements(
    def: &StmtFunctionDef,
    declared: Option<DeclaredRefinement>,
) -> (Option<DeclaredRefinement>, Option<DeclaredRefinement>) {
    if !is_generator_shaped(&def.body) {
        return (declared, None);
    }
    let Some(generator) = declared.and_then(|declared| declared.generator) else {
        return (None, None);
    };
    (generator.return_type, Some(generator.yield_type))
}

/// Whether `body` contains a `yield`/`yield from` anywhere that makes
/// CPython compile the enclosing `def` as a generator function
/// (datamodel.rst, "Generator functions") — the SAME routing fact
/// `expressions.rs::is_generator_def` reads for the call-answering side
/// of this feature, reimplemented locally per this file's own
/// "no importing across files for a one-line routing check" convention
/// (`loops.rs`'s own `generator_call_values` doc states the identical
/// precedent). Recognizes a top-level `Stmt::Expr(Expr::Yield |
/// Expr::YieldFrom)` and the same one-level-inside-a-`for`-loop nesting
/// `is_generator_def` reads (ruff collapses `for`/`async for` into one
/// `Stmt::For` node) — this is a ROUTING check only, not a claim about
/// which yields this checker can JUDGE: an unreadable yield shape still
/// walks through the ordinary blocker path once routed here.
fn is_generator_shaped(body: &[Stmt]) -> bool {
    body.iter().any(|stmt| match stmt {
        Stmt::Expr(expr_stmt) => matches!(expr_stmt.value.as_ref(), Expr::Yield(_) | Expr::YieldFrom(_)),
        Stmt::For(for_stmt) => for_stmt.body.iter().any(|inner| {
            matches!(inner, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::Yield(_) | Expr::YieldFrom(_)))
        }),
        _ => false,
    })
}

/// A function body's own parameters whose annotation reads through
/// `declared_refinement`: bind the name to a set-kind AbstractValue
/// holding the declared set (`known_set`, TrustSpec — the annotation is
/// read, not proved by execution). A parameter whose annotation states
/// nothing this table reads instead tries the CALLER-JOINED seed
/// (`unannotated_parameter_caller_seed`) — this def's own recorded direct
/// callers (`enclosing_def_name`'s lookup in `context.caller_arguments`),
/// each contributing whatever exact string its own bound argument at this
/// SAME position folds to; joining every caller's fold is what lets
/// `c-reference-shapes.py`'s `level_via_parameter_path` carry its
/// caller's literal script path onto `script_path` with no annotation at
/// all. Left unbound (ordinary Python, no seed) when no caller-joined
/// value applies either — unannotated, uncalled, or a caller whose own
/// argument does not fold.
fn seed_parameters(parameters: &Parameters, enclosing_def_name: Option<&str>, context: &WalkContext, environment: &mut Environment) {
    // Position, when `Some`, is this parameter's own index among
    // POSITIONAL parameters only (`posonlyargs` then `args`, in the
    // exact order a plain positional call's own `call.arguments.args`
    // fills them) — `CallerArguments`'s own indexing convention. A
    // keyword-only parameter (chained last, always `None` here) can
    // never be filled by a positional call at all, so it carries no
    // position for a caller-joined seed to read.
    let ordered = parameters
        .posonlyargs
        .iter()
        .chain(parameters.args.iter())
        .enumerate()
        .map(|(position, parameter)| (Some(position), parameter))
        .chain(parameters.kwonlyargs.iter().map(|parameter| (None, parameter)));
    for (position, parameter) in ordered {
        let Some(annotation) = parameter.parameter.annotation.as_deref() else {
            if let Some(position) = position {
                if let Some(seed) = unannotated_parameter_caller_seed(enclosing_def_name, position, context) {
                    environment.bind(parameter.parameter.name.id.as_str(), seed);
                }
            }
            continue;
        };
        // A bare CLASS-NAME parameter (`request: AudioRequest`, the class
        // itself declaring annotated fields — self-authored, `@dataclass`,
        // or pydantic `BaseModel` alike) seeds a TAGGED `Kind::Object`
        // instance exactly as `judge_construction` tags one built from a
        // real call: `source` carries the class name so `evaluate_attribute_
        // read` (expressions.rs) finds `model` in `environment.classes()`/
        // `context.classes` and reads through `instances::field_read_
        // through_model`, which — for an ordinary stored field, the only
        // shape a parameter's fields are — resolves to `instances::
        // field_read`'s linear scan of the INSTANCE'S OWN `keys`. The read
        // path never re-derives a field's value from `ClassModel.fields`;
        // it consumes only what this seed puts in `keys`, so every
        // declared field is populated here (per-field, from `ClassField.
        // declared`) rather than left for a later lookup to reconstruct.
        // A name that is not a class in this table falls through to the
        // ordinary `declared_refinement` read below, unchanged.
        if let Expr::Name(class_name) = annotation {
            if let Some(model) = context.classes.get(class_name.id.as_str()) {
                let instance = class_parameter_object(model);
                environment.bind(parameter.parameter.name.id.as_str(), instance);
                continue;
            }
        }
        // A bare `int`/`float`/`str` PARAMETER seeds its sort claim (the
        // whole-int ray etc. — typereading's own base-sort reader), so
        // `age: int` flowing into a refined sink refuses by containment
        // ("a whole int admits values outside the set") unless a guard
        // narrows it. Scoped to parameters ONLY: the general annotation
        // table does not read base sorts, so `-> int` returns stay
        // unjudged and helper bodies gain no new blockers.
        let Some(declared) =
            declared_refinement(annotation, context.aliases, context.imports, environment)
                .or_else(|| crate::typereading::base_sort_return_refinement(annotation))
        else {
            continue;
        };
        // A `list[X]`/`set[X]`/`Sequence[X]` PARAMETER (`declared.element`
        // Some, `declared.set` unused/empty — typereading's own "one
        // active field" convention) seeds a SEQUENCE whose every position
        // draws from X's own set: `Kind::Set` over one `Form::Repeat(X,
        // lo, hi)` refinement — the bare unbounded window (`lo` 0, `hi`
        // `None`, the same shape `star` alone used to build) when the
        // declaration states no length bound, or a TIGHTER window when
        // `declared.element_length` carries one (`Annotated[list[X],
        // Field(min_length=…, max_length=…)]`, `typereading.rs`'s own
        // doc — `repeat_of`/`repetition` already accept an arbitrary
        // `lo`, so this is the SAME constructor family, not a new one).
        // Either way, `subscript_read`'s own `Kind::Set` arm reads any
        // index as "some member of X", the repetition grammar's own
        // definition (a repetition's positions never hold anything
        // outside its element alphabet), so no kernel round trip is ever
        // needed to answer an element read; `relational_sum.rs`'s own
        // `element_and_count_sets` reads the SAME window's `lo`/`hi` into
        // the count set a relational division ties to, so a `min_length`
        // bound here is what gives that division its `count >= 1` fact
        // with no cast. Scoped to a SCALAR/string-sorted element
        // (`element.set` non-empty) — an element that is itself
        // container-shaped (`dict[str, X]`'s own element, another
        // `list[X]`) has no scalar set here to repeat over, and is left
        // for the plain element-carrying seed below, matching today's
        // behavior for that shape.
        let is_sequence_container = declared.spelling.starts_with("list[")
            || declared.spelling.starts_with("set[")
            || declared.spelling.starts_with("Sequence[");
        if is_sequence_container {
            if let Some(element) = &declared.element {
                if !element.set.forms.is_empty() {
                    let (lo, hi) = declared.element_length.unwrap_or((0, None));
                    // The element's own sort (`requires_integer` reads the
                    // `Form::Integer` marker `annotated_expression_set`
                    // pushes for a bare `int`, the same recognizer
                    // `adapter_alias_verdict` above already uses to tell an
                    // int-sorted alias from a float-sorted one) rides onto
                    // the OUTER sequence value's `kind_tag` — `star_numeric_
                    // hull`/`sum_call_over_star`/`min_max_over_star`
                    // (builtin_models.rs) all read the sequence value's own
                    // tag, never the element's, to answer sum/min/max over
                    // an unknown-length iterable.
                    let sort = if requires_integer(&element.set) {
                        PrimitiveKind::Integer
                    } else {
                        PrimitiveKind::Float
                    };
                    let sequence = AbstractValue {
                        kind_tag: Some(sort),
                        ..known_set(
                            make_refined_set(vec![repeat_of(element.set.clone(), lo, hi)]),
                            None,
                            TrustSpec,
                            SetKindTag::None,
                        )
                    };
                    environment.bind(parameter.parameter.name.id.as_str(), sequence);
                    continue;
                }
            }
        }
        // A FIXED-ARITY `tuple[X, Y]` PARAMETER (`declared.positions`
        // Some — typereading's own per-position table) seeds a
        // KNOWN-LENGTH `Kind::List` whose slot `i` is `known_set` over
        // position `i`'s own declared set — the same nested-exact-
        // sequence shape a literal tuple/list display already builds
        // (`collection_models.rs`'s own module doc). Unlike the sequence
        // seed above, every position keeps its OWN set rather than
        // sharing one starred element, so `subscript_read`'s `Kind::List`
        // arm reads slot `i` back exactly, and a spread (`[*item]`) can
        // splice the slots in place — a spread only recognizes a known
        // `Kind::List` receiver (`expressions.rs::evaluate_display_elements`),
        // never the unknown-length `Kind::Set` star shape above.
        if let Some(positions) = &declared.positions {
            let items = positions
                .iter()
                .map(|position| known_set(position.set.clone(), None, TrustSpec, SetKindTag::None))
                .collect();
            let tuple = known_list(items, TrustSpec);
            environment.bind(parameter.parameter.name.id.as_str(), tuple);
            continue;
        }
        // A scalar declared set carries its own numeric sort onto the
        // seeded value's `kind_tag` — the same tag `min_max_scalar_operand`/
        // `star_numeric_hull`/`sum_call_over_star` (builtin_models.rs) read
        // to answer sum/min/max — but ONLY when the declared set is
        // numeric-ground: `on_one_tuple_layer` alone also reads a
        // `Literal["A", "B", "C"]` string-tuple union as "on the one-tuple
        // layer" (the tuple pun `assignability.rs::states_sequence`'s own
        // doc names), so the gate pairs both checks exactly as every sort
        // law in that file does. A string/sequence-shaped declared set
        // (`states_sequence` true, or `on_one_tuple_layer` false) is left
        // untagged, unchanged from today.
        let admits_none = declared.admits_none;
        let seeded = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
            let sort = if requires_integer(&declared.set) {
                PrimitiveKind::Integer
            } else {
                PrimitiveKind::Float
            };
            AbstractValue {
                kind_tag: Some(sort),
                ..known_set(declared.set, None, TrustSpec, SetKindTag::None)
            }
        } else {
            known_set(declared.set, None, TrustSpec, SetKindTag::None)
        };
        // `Optional[X]`/`X | None` (`declared.admits_none`): the parameter
        // genuinely may arrive as `None` at runtime, so the seeded value
        // must carry that admission the same way a JS `undefined`-or-`X`
        // parameter would — wrapped in the maybe carrier
        // (`possibly_absent`), its absent side pinned `NullOnly` (Python's
        // `None`, never JS's `undefined`). Un-wrapped, a bare `Kind::Set`
        // seed has NO shape a narrowing test could ever read as "possibly
        // absent" (`narrowing.rs`'s own `is`/`is not None` arms only ever
        // touch `Kind::Values`/`Kind::PossiblyUndefined`), so `sample is
        // not None` against the un-wrapped seed decided PROVABLY TRUE
        // outright (`expressions.rs::compare_pair`'s `is`/`is not` law
        // settles identity the moment either side is KNOWN and not
        // `Kind::Null` — a bare `Kind::Set` already qualifies) rather than
        // staying the undecided test an Optional parameter's own guard
        // must be.
        let seeded = if admits_none {
            possibly_absent(seeded, AbsentFlavor::NullOnly, Some(TrustSpec), false)
        } else {
            seeded
        };
        environment.bind(parameter.parameter.name.id.as_str(), seeded);
    }
}

/// The caller-joined seed for one UNANNOTATED parameter at `position`
/// (among positional parameters only) in the def named `enclosing_def_
/// name`: `None` when there is no def name (a nested/method body, or the
/// module body itself), the def does not qualify at all (missing from
/// `context.caller_arguments` — some occurrence of its name was not a
/// plain positional call), it has zero recorded callers, some caller's
/// own call is shorter than `position` (an argument this position was
/// never even passed at that site), some caller's own argument does not
/// fold to an exact string (`caller_argument_exact_string`), or two
/// callers fold to DIFFERENT exact strings (a real disagreement — no
/// single literal is true of every call, so this scalar seed cannot
/// state one without inventing a value no caller actually passed).
/// `Some` only when EVERY recorded caller's argument at this position
/// folds to the identical exact string, the one case where "this
/// parameter always holds this text" is something the checker actually
/// proved from the call sites it can see.
fn unannotated_parameter_caller_seed(
    enclosing_def_name: Option<&str>,
    position: usize,
    context: &WalkContext,
) -> Option<AbstractValue> {
    let def_name = enclosing_def_name?;
    let calls = context.caller_arguments.get(def_name)?;
    if calls.is_empty() {
        return None;
    }
    let module_environment = module_scope_environment(context);
    let mut agreed_text: Option<String> = None;
    for call_arguments in calls {
        let argument = call_arguments.get(position)?;
        let text = caller_argument_exact_string(argument, &module_environment, context.kernel)?;
        match &agreed_text {
            None => agreed_text = Some(text),
            Some(existing) if *existing == text => {}
            Some(_) => return None,
        }
    }
    Some(crate::string_models::string_literal_value(&agreed_text?))
}

/// A fresh, empty-locally-bound `Environment` seeded with every MODULE-
/// LEVEL binding this walk already exposes to every ordinary body
/// (`context.module_bindings` — the exact table `walk_body_with_self_
/// binding` layers onto each body's own environment). A caller's argument
/// expression is read in THIS scope, never the calling body's own live
/// one: the escape discipline this whole join rests on only trusts a
/// module-level constant or a written literal, precisely because a
/// caller's own LOCAL variable is not something a def-name-keyed,
/// built-once-per-module table could ever resolve safely.
fn module_scope_environment(context: &WalkContext) -> Environment {
    let mut environment = Environment::new(HashSet::new());
    for (name, value) in &context.module_bindings {
        environment.bind(name, value.clone());
    }
    environment
}

/// The same three const-fold tiers `foreign_edge.rs::const_folded_text_of`
/// reads (a written string literal; a bare `Name` resolving, through
/// `environment`, to a known exact string; or any other expression
/// `evaluate_expression` folds to an exact string — an f-string composed
/// entirely of consts, a `+` concatenation of known exact strings),
/// reimplemented here per this crate's own no-shared-private-helper
/// convention (`foreign_edge.rs::exact_string_text`'s own doc states the
/// identical precedent against `string_models.rs`). `None` on any tier
/// that does not fold — a caller's argument that is itself a local
/// variable, a call, or any other non-exact expression.
fn caller_argument_exact_string(argument: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    if let Expr::StringLiteral(literal) = argument {
        return Some(literal.value.to_str().to_owned());
    }
    if let Expr::Name(name) = argument {
        if let Some(bound) = environment.read(name.id.as_str()) {
            if let Some(text) = caller_exact_string_text(bound) {
                return Some(text);
            }
        }
    }
    let folded = evaluate_expression(argument, environment, kernel);
    caller_exact_string_text(&folded)
}

/// The exact text an `AbstractValue` carries, if it is a `Kind::Values`
/// state sorted `PrimitiveKind::String` — `string_models.rs::
/// exact_string_text`'s own exact twin, reimplemented locally per this
/// crate's convention (see `caller_argument_exact_string`'s own doc).
fn caller_exact_string_text(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect())
}

/// A CLASS-NAME parameter's own instance value: a `Kind::Object` tagged
/// with `source = model.name` (the SAME tag `judge_construction` writes
/// for a real call — the tag `evaluate_attribute_read` and every other
/// `receiver.source`-keyed reader already consult), one `ObjectKey` per
/// declared field IN `model.fields`' own order. A field with no declared
/// refinement (`ClassField.declared` `None`) seeds NOTHING for that key —
/// absent from `keys` entirely, not a fabricated unrefined set — so a
/// later read of it stays undetermined naming the true blocker (there is
/// no declaration to read), rather than reading as a false "any value at
/// all" claim this seed did not earn.
fn class_parameter_object(model: &ClassModel) -> AbstractValue {
    let entries: Vec<ObjectKey> = model
        .fields
        .iter()
        .filter_map(|field| {
            let declared = field.declared.as_ref()?;
            Some(ObjectKey {
                name: field.name.clone(),
                numeric: false,
                value: class_field_value(declared),
            })
        })
        .collect();
    let mut instance = known_object(entries, None, true, TrustSpec, false);
    instance.source = model.name.clone();
    instance
}

/// One declared field's own seeded value — the same two shapes
/// `seed_parameters` already builds for a top-level parameter of the
/// same declared shape, so a field reads back exactly like a parameter
/// would: a sequence-shaped field (`declared.element` Some, a scalar-
/// sorted element — the same gate `seed_parameters`' sequence-container
/// branch reads) becomes a repetition set carrying the element's own
/// numeric sort on `kind_tag`, so `samples: Annotated[list[float],
/// Field(min_length=1)]` flows into sum/min/max/relational reads
/// exactly as a top-level `samples: list[float]` parameter would. A
/// scalar field with a numeric-ground declared set carries that sort on
/// `kind_tag` the same way a scalar parameter does. `set_kind_tag` stays
/// `SetKindTag::None` in both shapes — that field wears the bigint/symbol
/// distinction, unrelated to `kind_tag`'s numeric sort, and this seed
/// states nothing about it either way.
fn class_field_value(declared: &DeclaredRefinement) -> AbstractValue {
    if let Some(element) = &declared.element {
        if !element.set.forms.is_empty() {
            let (lo, hi) = declared.element_length.unwrap_or((0, None));
            let sort = if requires_integer(&element.set) {
                PrimitiveKind::Integer
            } else {
                PrimitiveKind::Float
            };
            return AbstractValue {
                kind_tag: Some(sort),
                ..known_set(
                    make_refined_set(vec![repeat_of(element.set.clone(), lo, hi)]),
                    None,
                    TrustSpec,
                    SetKindTag::None,
                )
            };
        }
    }
    if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
        let sort = if requires_integer(&declared.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        return AbstractValue {
            kind_tag: Some(sort),
            ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
        };
    }
    known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
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
///
/// Returns whether this statement provably never falls through to
/// whatever follows it — true only for a `try` whose own arms all
/// terminate (`walk_try`'s return). Every other form still runs exactly
/// as before and answers `false`: this is not a general reachability
/// signal, only the one fact `walk_try` already computes and this
/// dispatcher passes outward untouched.
fn walk_statement(
    stmt: &Stmt,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    provably_unbound: &mut HashSet<String>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> bool {
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
        // `yield`/`yield from` — tried before the ordinary Stmt::Expr
        // handling below: a yield is not a call and carries no receiver,
        // method, or mutation shape any of those channels recognize, so
        // routing it there would only ever reach `sink_value`'s plain
        // `evaluate_expression` fallback with no yield-position judging
        // at all. `walk_yield` owns judging it against the enclosing
        // generator's own declared yield set.
        Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::Yield(_) | Expr::YieldFrom(_)) => {
            walk_yield(
                expr_stmt.value.as_ref(),
                yield_refinement,
                context,
                aug_assign_refinements,
                environment,
                blocked,
                out,
            );
        }
        Stmt::Expr(expr_stmt) => {
            bind_walrus_targets(expr_stmt.value.as_ref(), context, aug_assign_refinements, environment, out);
            // STATEMENT-SIDE METHOD CALLS tries first: a bare-Name receiver
            // bound to a known instance rebinds through
            // instances::method_call_result, discarding the returned
            // value (an expression-statement's own value is never read,
            // matching sink_value's other callers' discard convention at
            // Stmt::Expr). Declining (None) falls to the CALLEE-EFFECTS
            // CHANNEL (a bare-Name same-module call whose body writes an
            // enclosing name — `apply_call_effects`), then the collection
            // mutated_receiver path, then to sink_value's plain read —
            // exactly the ordering already in place, with the effects
            // channel inserted where a bare same-module call is otherwise
            // indistinguishable from any other unmodeled call.
            if instance_method_call_result(expr_stmt.value.as_ref(), context, environment).is_none()
                && apply_call_effects(expr_stmt.value.as_ref(), context, environment, aug_assign_refinements, out).is_none()
                && !walk_mutating_call_statement(expr_stmt.value.as_ref(), context, environment)
            {
                sink_value(expr_stmt.value.as_ref(), context, environment, aug_assign_refinements, out);
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
            walk_class_def(def, environment, context, out);
        }
        // `del a, b, …` (simple_stmts.rst, "The `del` statement":
        // "Deletion of a target list recursively deletes each target,
        // from left to right") — every named target forgets what the
        // walk knew; no judgment and no blocker either way. A deleted
        // name is UNBOUND again afterward (the same state a valueless
        // AnnAssign leaves), but this table only tracks names a valueless
        // AnnAssign itself declared — `del` on an ordinary name states
        // nothing this law reads, so `provably_unbound` is untouched.
        //
        // `del d[k]` (a `Subscript` target, bare-Name receiver): the
        // MUTATION CONTRACT's own delete-shaped write —
        // `collection_models::dict_without_item` answers the receiver
        // WITHOUT that key, which rebinds `name` through
        // `walk_del_subscript_target` so a later read sees the key's
        // absence; an unresolved receiver/key FORGETS `name` (the stale
        // pre-delete value must not survive), the same honesty every
        // other unresolved write in this file keeps. Every other target
        // shape (bare name, tuple/list/starred, a non-Name-receiver
        // subscript) still forgets through `forget_target_names`,
        // unchanged.
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                if let Expr::Subscript(subscript) = target {
                    walk_del_subscript_target(subscript, context, environment);
                } else {
                    forget_target_names(target, environment);
                }
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
                yield_refinement,
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
            walk_loop(stmt, return_refinement, yield_refinement, context, environment, aug_assign_refinements, blocked, out);
        }
        Stmt::Match(match_stmt) => {
            provably_unbound.clear();
            walk_match(
                match_stmt,
                return_refinement,
                yield_refinement,
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
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                blocked,
                out,
            );
        }
        Stmt::Try(try_stmt) => {
            provably_unbound.clear();
            return walk_try(
                try_stmt,
                return_refinement,
                yield_refinement,
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
    false
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
    let Some(value) = sink_value(value_expr, context, environment, aug_assign_refinements, out) else {
        // a provable raise already pushed its own RTS7001 at the
        // raising expression — this return never produces a value to
        // judge, since CPython never reaches the return statement's own
        // completion on this path.
        return;
    };
    // The value this return produces, handed to whoever asked this walk
    // to collect them (`env::collect_returned_values` — `fact_export`'s
    // own derivation seam). A no-op for every ordinary walk, and the
    // value is exactly the one judged below: the export never runs a
    // second, differently-derived reading of the same return.
    environment.record_returned_value(value.clone());
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
            let sentence = name_unmodeled_call_sentence(sentence, Some(value_expr), environment);
            record_blocker(blocked, value_expr.range(), sentence, out);
        }
    }
}

/// `yield value` / bare `yield` / `yield from value`, against the
/// enclosing generator's own YIELD position (`Generator[Y, S, R]`'s
/// first element, `AsyncGenerator[Y, S]`'s/`Iterator[Y]`'s/
/// `Iterable[Y]`'s only element — `typereading.rs::GeneratorRefinement`,
/// threaded down as `yield_refinement` by `generator_body_refinements`).
/// No declared yield position (`yield_refinement` is `None` — an
/// ordinary, non-generator-annotated body, or a generator body whose own
/// `-> Annotation` did not read as one of the four generator forms)
/// means nothing judges here, the mission's "no annotation → no
/// judging" rule applied to this checked position instead of `return`'s.
/// A BARE `yield` (`Expr::Yield` with no operand) yields `None`
/// (datamodel.rst's generator-iterator protocol: `next()` on a bare
/// `yield` hands back `None`) — judged as `Kind::Null` against the
/// declared yield set exactly like any other absent value, so a
/// non-`Optional` yield type still fires on it. `yield from <expr>`
/// DELEGATES: every value the inner generator yields flows out of this
/// generator too, so EACH ONE judges against this generator's own
/// declared yield set (`delegated_generator_yields`'s own two-reading
/// doc: the callee's actual body-walked yields where they read, its
/// declared annotation's yield set otherwise) — the first Fire wins,
/// matching `judge`'s own dict/list element-law convention of reporting
/// the first escaping member rather than joining every member's verdict.
fn walk_yield(
    yield_expr: &Expr,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let Some(declared) = yield_refinement else {
        return;
    };
    match yield_expr {
        Expr::Yield(yield_node) => {
            let range = yield_node.range();
            let (value, source_expr) = match yield_node.value.as_deref() {
                Some(value_expr) => {
                    bind_walrus_targets(value_expr, context, aug_assign_refinements, environment, out);
                    let Some(value) = sink_value(value_expr, context, environment, aug_assign_refinements, out) else {
                        // a provable raise already pushed its own RTS7001 —
                        // this yield never produces a value to judge.
                        return;
                    };
                    (value, Some(value_expr))
                }
                // bare `yield` — the generator hands back None here.
                None => (refined_domain::abstract_value::null_value(), None),
            };
            judge_at(&value, declared, range, source_expr, context, environment, blocked, out);
        }
        Expr::YieldFrom(yield_from) => {
            let range = yield_from.range();
            let Some(elements) = delegated_generator_yields(yield_from.value.as_ref(), context, environment) else {
                record_blocker(
                    blocked,
                    range,
                    "this yield from's own delegate does not yet state a readable yield set".to_owned(),
                    out,
                );
                return;
            };
            for element in &elements {
                judge_at(element, declared, range, None, context, environment, blocked, out);
                // the first Fire this loop pushes is the row's own
                // verdict — later elements still walk (so a LATER
                // element's own Undetermined can still set the body's
                // blocker when no earlier element fired), but a second
                // Fire at the same range would only restate the same
                // row twice, so this loop does not stop early; judge_at
                // itself never double-reports past `blocked` for the
                // Undetermined branch, and a Fire is idempotent to
                // report once per offending element in the rare case
                // more than one escapes (matching the dict/list element
                // law's own "first Fire" framing loosely, since a
                // delegate's own elements are not individually
                // addressable the way a dict key/list index is).
            }
        }
        _ => {}
    }
}

/// Judges one value at `range` against `declared`, pushing a Fire or
/// recording the body's blocker candidate — `walk_return`'s own
/// Fire/Silent/Undetermined tail, factored out so `walk_yield`'s two
/// call shapes (a plain yield's one value, a delegation's several) share
/// it instead of repeating the match. `source_expr` is the single
/// expression this `value` was read from, when one exists (a plain
/// `yield value_expr`'s own operand) — `None` for a `yield from`
/// delegation's own per-element judging, where no single source
/// expression names any one element. Passed through to
/// `name_unmodeled_call_sentence` so a plain `yield torch.arange(5)`
/// under a declared yield type names the module the same way a return
/// or an assignment does.
fn judge_at(
    value: &AbstractValue,
    declared: &DeclaredRefinement,
    range: TextRange,
    source_expr: Option<&Expr>,
    context: &WalkContext,
    environment: &Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    match judge(value, declared, context.kernel) {
        Verdict::Fire(message) => out.push(Finding { range, code: "RTS7001", message }),
        Verdict::Silent => {}
        Verdict::Undetermined(sentence) => {
            let sentence = name_unmodeled_call_sentence(sentence, source_expr, environment);
            record_blocker(blocked, range, sentence, out);
        }
    }
}

/// `yield from <expr>`'s own delegate, read to the values it hands this
/// generator's caller. `<expr>` must be a bare same-module call
/// (`gen()`) — any other delegate shape (a non-Name callee, a call to a
/// name with no same-module `def`) declines whole, this row's own
/// blocker. Two readings, tried in order, mirroring refined-ts-go's own
/// `GeneratorElementOf` (walk/generator_element.go): the callee's own
/// ACTUAL yields, walked fresh through `instances::generator_yields`
/// (straight-line/single-for-loop bodies only — that function's own
/// doc), state the TIGHTEST claim this checker can prove (a callee
/// declared `-> Generator[int, None, None]` whose body only ever yields
/// `40` hands this delegation exactly `[40]`, not the unbounded `int`
/// ray its own annotation states) and are read FIRST for that reason;
/// the callee's own DECLARED yield set (`typereading::declared_refinement`'s
/// generator arm) is the fallback once the body-walk declines (a
/// conditional yield, a parameter-shaped body, any restricted-
/// interpreter shape `generator_yields` does not read) — the annotation
/// is still a real claim about every value the callee could ever hand
/// back, so a callee whose actual body this walk cannot run still judges
/// against what it PROMISES.
fn delegated_generator_yields(
    delegate: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<Vec<AbstractValue>> {
    let Expr::Call(call) = delegate else {
        return None;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return None;
    };
    let def = context.functions.def(callee_name.id.as_str())?;
    if let Some(yields) =
        instances::generator_yields(def, &[], environment.functions(), context.kernel, environment.call_depth())
    {
        return Some(yields);
    }
    let outer_environment = Environment::new(HashSet::new());
    let declared = declared_refinement(def.returns.as_deref()?, context.aliases, context.imports, &outer_environment)?;
    let yield_type = declared.generator?.yield_type;
    // Tags the numeric sort `min_max_scalar_operand`/`star_numeric_hull`/
    // `sum_call_over_star` (builtin_models.rs) read, the same guarded
    // rule `seed_parameters` applies to a declared set: numeric-ground
    // only (`on_one_tuple_layer` alone also reads a `Literal["A", "B"]`
    // string-tuple union as "on the one-tuple layer", so `states_sequence`
    // must be false too, ruling out that pun). A string/sequence-shaped
    // yield type is left untagged, unchanged from today.
    if on_one_tuple_layer(&yield_type.set) && !states_sequence(&yield_type.set) {
        let sort = if requires_integer(&yield_type.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        return Some(vec![AbstractValue {
            kind_tag: Some(sort),
            ..known_set(yield_type.set, None, TrustSpec, SetKindTag::None)
        }]);
    }
    Some(vec![known_set(yield_type.set, None, TrustSpec, SetKindTag::None)])
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
            .or_else(|| typed_dict_return_refinement(annotation, &context.typed_dicts))
    });
    let (return_refinement, yield_refinement) = generator_body_refinements(def, return_refinement);
    let bare_sort_return_refinement = def.returns.as_deref().and_then(base_sort_return_refinement);
    walk_body_with_self_binding(
        &def.body,
        Some(def.parameters.as_ref()),
        return_refinement.as_ref(),
        yield_refinement.as_ref(),
        None,
        None,
        None,
        Some(def.name.id.as_str()),
        bare_sort_return_refinement.as_ref(),
        context,
        out,
    );
}

/// Every top-level `def`'s own DERIVED return value, keyed by name: the
/// join of every value that def's `return` statements produce, from the
/// SAME walk `findings_for_module_with_resolver` runs over the module —
/// parameters seeded from their declarations by `seed_parameters`, the
/// body walked statement by statement, and each return's value taken
/// from the point `walk_return` already computed it for judging.
///
/// `fact_export`'s derivation seam, and its only entry into this file.
/// The module's shared context (aliases, imports, cross-module surface,
/// function/class tables — everything a body reads must be built the way
/// the checker builds it, never a narrower stand-in) is built ONCE here
/// and every def walks against it, so an N-def module resolves its
/// imports once rather than N times.
///
/// `derived_return_values`'s own answer: every def's derived return
/// value, keyed by name, PLUS — for a def with no entry in that map —
/// the first blocker sentence its own walk recorded, keyed the same way.
/// A def absent from both is a body this walk ran cleanly with no
/// blocker and no `return` statement at all (a bare `pass`, an `if`
/// with no branch reaching a return) — genuinely nothing to name,
/// distinct from a def the walk COULD NOT get through.
pub struct DerivedReturns {
    pub values: HashMap<String, AbstractValue>,
    pub blockers: HashMap<String, String>,
}

/// An empty answer when the module states no `type` alias AND no
/// recognized `Annotated` import (the same early exit
/// `findings_for_module_at` takes). A def whose body produced no
/// `return` value this walk could read is simply absent from `values`
/// — never an entry holding a guessed value. `blockers` names the FIRST
/// construct that stopped a def's own walk (the same RTS7002 sentence
/// `findings_for_module` would report for this body), independent of
/// whether the def's `-> Annotation` was itself readable: an unreadable
/// return annotation gives `return_refinement` as `None`, which is
/// `walk_return`'s own signal to skip JUDGING a return value — it does
/// not, and must not, silence the blocker this body's own walk hit on
/// the way to that return (`walk_loop`/`walk_statement`'s
/// `record_blocker` calls are unconditional on `return_refinement`
/// already; what this function fixes is that its OWN discarded-findings
/// walk used to drop that recorded blocker on the floor rather than
/// handing it back to a caller that needs it).
pub fn derived_return_values(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
) -> DerivedReturns {
    derived_return_values_at(module, resolver, kernel, None)
}

/// `derived_return_values` plus the checked file's own directory — the
/// export seam's twin of `findings_for_module_at`. A relative argv
/// target inside a recognized foreign edge (`subprocess.run(["node",
/// "./audio_level.ts"], ...)`) resolves against this directory, exactly
/// as it does when the SAME body is walked for ordinary findings — a
/// def whose derivation crosses to TypeScript exports the same way a
/// def with no foreign edge does.
pub fn derived_return_values_at(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> DerivedReturns {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    if aliases.is_empty()
        && imports.annotated_names.is_empty()
        && imports.literal_names.is_empty()
    {
        return DerivedReturns { values: HashMap::new(), blockers: HashMap::new() };
    }
    let surface = module_surface(module, resolver, kernel, IMPORT_DEPTH_CAP);
    let own_functions = function_table(module);
    let functions = Arc::new(merged(&own_functions, surface.functions.as_ref()));
    let own_classes = class_table(module, &aliases, &imports, kernel);
    let mut classes = Arc::try_unwrap(surface.classes)
        .unwrap_or_else(|_| panic!("module_surface's own Arc<classes> has no other owner yet"));
    for (class_name, model) in own_classes {
        classes.insert(class_name, model);
    }
    for def in module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    }) {
        for (class_name, model) in local_class_table(&def.body, &aliases, &imports, kernel) {
            classes.insert(class_name, model);
        }
    }
    let module_callable_returns = Arc::new(module_level_callable_returns(module, &aliases, &imports));
    let strict_int_aliases = strict_int_alias_names(module);
    let typed_dicts = Arc::new(instances::typed_dict_table(module, &aliases, &imports));
    let caller_arguments = Arc::new(crate::function_table::caller_argument_positions(module));
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel,
        functions,
        classes: Arc::new(classes),
        datetime_imports: Arc::new(crate::expressions::datetime_imports(module)),
        module_bindings: surface.bindings,
        module_callable_returns,
        strict_int_aliases: &strict_int_aliases,
        typed_dicts,
        caller_arguments,
        entry_directory: entry_directory.map(|dir| dir.to_path_buf()),
        evaluations_recorder: None,
    };
    let mut values = HashMap::new();
    let mut blockers = HashMap::new();
    for def in module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    }) {
        let outer_environment = Environment::new(HashSet::new());
        let return_refinement = def.returns.as_deref().and_then(|annotation| {
            declared_refinement(annotation, context.aliases, context.imports, &outer_environment)
                .or_else(|| typed_dict_return_refinement(annotation, &context.typed_dicts))
        });
        let (return_refinement, yield_refinement) = generator_body_refinements(def, return_refinement);
        let bare_sort_return_refinement = def.returns.as_deref().and_then(base_sort_return_refinement);
        let mut returned_values: Vec<AbstractValue> = Vec::new();
        let mut findings = Vec::new();
        walk_body_with_self_binding(
            &def.body,
            Some(def.parameters.as_ref()),
            return_refinement.as_ref(),
            yield_refinement.as_ref(),
            None,
            None,
            Some(&mut returned_values),
            Some(def.name.id.as_str()),
            bare_sort_return_refinement.as_ref(),
            &context,
            &mut findings,
        );
        let mut answers = returned_values.into_iter();
        let Some(first) = answers.next() else {
            // No `return` this walk could read a value from — named
            // here, independent of whether `-> Annotation` itself read
            // (a bare `-> float` leaves `return_refinement` `None`, but
            // `record_blocker`'s own call sites never gate on that: the
            // FIRST unwalkable construct this body's walk hit is still
            // right here in `findings`, exactly the RTS7002 sentence
            // `findings_for_module` would report for the same body).
            if let Some(blocker) = findings.iter().find(|finding| finding.code == "RTS7002") {
                blockers.insert(def.name.id.as_str().to_owned(), blocker.message.clone());
            }
            continue;
        };
        values.insert(
            def.name.id.as_str().to_owned(),
            answers.fold(first, refined_domain::lattice_operations::join_known),
        );
    }
    DerivedReturns { values, blockers }
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
///
/// LAMBDA-ASSIGN LAW (b-body-expressions.py:578/581's own call sites):
/// `f = lambda x: <expr>` assigned to a bare name is ALSO recorded here,
/// as a synthetic `StmtFunctionDef` (`lambda_as_synthetic_def`) — so a
/// later `f(...)` call resolves through `environment.functions()` and
/// `summaries::call_result` exactly like an ordinary same-module `def`,
/// with no separate call-answering path to maintain. A later plain
/// `def f(...):`/another lambda assign of the SAME name in this body
/// overwrites the synthetic entry the same way `function_table`'s own
/// scan already lets a later `def` win (`Stmt::FunctionDef`/lambda-assign
/// entries share the one `local_defs` list, in source order, and
/// `function_table` keeps whichever inserts last).
fn local_function_table(body: &[Stmt]) -> FunctionTable {
    // collect() infers ModModule's own body container type from the
    // struct field (the same construction cross_module::synthetic_module
    // uses), so no container crate is named here
    let local_defs = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::FunctionDef(def) => Some(Stmt::FunctionDef(def.clone())),
            Stmt::Assign(assign) => lambda_as_synthetic_def(assign).map(Stmt::FunctionDef),
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

/// `name = lambda <params>: <expr>` — a single bare-Name target whose
/// value is a `Lambda` — read as a synthetic `def name(<params>):
/// return <expr>`, so `summaries::call_result` (which only interprets a
/// real `StmtFunctionDef`) can answer a later `name(...)` call through
/// the lambda's own body. `None` for a multi-target assign, a non-Name
/// target, or a non-Lambda value — this law only ever recognizes the
/// exact `f = lambda ...: ...` shape.
fn lambda_as_synthetic_def(assign: &StmtAssign) -> Option<StmtFunctionDef> {
    let [Expr::Name(target_name)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Lambda(lambda) = assign.value.as_ref() else {
        return None;
    };
    let parameters = lambda
        .parameters
        .as_deref()
        .cloned()
        .unwrap_or_else(Parameters::default);
    let return_stmt = Stmt::Return(StmtReturn {
        node_index: AtomicNodeIndex::NONE,
        range: lambda.body.range(),
        value: Some(lambda.body.clone()),
    });
    Some(StmtFunctionDef {
        node_index: AtomicNodeIndex::NONE,
        range: assign.range(),
        is_async: false,
        decorator_list: Default::default(),
        name: ruff_python_ast::Identifier::new(target_name.id.as_str(), target_name.range()),
        type_params: None,
        parameters: Box::new(parameters),
        returns: None,
        body: [return_stmt].into_iter().collect(),
    })
}

/// A class body: walked as its own body (its own locally-bound prepass,
/// its own environment) — a class-level `AnnAssign` field judges
/// exactly like a module- or function-level one, and a `def` inside the
/// class body recurses as an ordinary function body through
/// `walk_statement`'s own `Stmt::FunctionDef` arm EXCEPT for a `self`-
/// taking member, which `walk_body`'s own `self_model` parameter routes
/// through `walk_method_def` instead (the self-seeding law). A class
/// body has no enclosing function, so it carries no return refinement
/// of its own (compound_stmts.rst, "Class definitions": the class body
/// executes in a new namespace with no relation to a function's own
/// scope).
///
/// `self_model`: `def`'s own `ClassModel`, looked up BY NAME out of
/// `enclosing_environment.classes()` — the table `walk_body`/
/// `walk_body_with_self_binding` already built and set on the
/// environment via `merged_classes_for_body` BEFORE this body's own
/// statement loop began dispatching (so `def`'s entry is already
/// present by the time `Stmt::ClassDef(def)` reaches this function).
/// This is the SAME table a sibling `super().__init__(...)`/
/// `super().<method>(...)` call already resolves parent links through
/// (module-level classes keep the full parent chain
/// `findings_for_module_with_resolver` built once over the WHOLE
/// module; a body-local class is parent-linked against any SIBLING
/// body-local class `local_class_table`'s own single build over the
/// whole enclosing body already covers) — looking the model up here,
/// rather than rebuilding a one-class synthetic table from `def` alone,
/// is what keeps `self_model.parent_methods`/inherited fields intact
/// for a self-seeded method body (`call_super_method`'s own
/// `super().years()` shape). `None` when the environment carries no
/// class table at all (should not occur — every walk sets one) or the
/// name is genuinely absent; `walk_body` itself tolerates `None`, a
/// class shape it somehow declines to model still walks its own body
/// with member defs falling back to the ordinary un-seeded
/// `walk_function_def` path.
fn walk_class_def(def: &StmtClassDef, enclosing_environment: &mut Environment, context: &WalkContext, out: &mut Vec<Finding>) {
    // Cloning the Arc (cheap — a refcount bump, not a table copy) frees
    // this table from `enclosing_environment`'s own borrow, so the
    // class-object seed below can mutably bind into it while `self_model`
    // stays alive for `walk_body`'s own read afterward.
    let classes = enclosing_environment.classes().cloned();
    let self_model = classes.as_ref().and_then(|classes| classes.get(def.name.id.as_str()));
    // CLASS-OBJECT SEEDING: the class's own bare name becomes readable,
    // in THIS enclosing scope, as a tagged Kind::Object carrying its
    // class_attributes (`instances::class_object_value`'s own doc) — the
    // same environment slot `Counted.total = 40`/`Counted.total` (a
    // bare-Name attribute write/read, e-class-and-function.py's
    // `class_attribute_write`) already finds and rebinds through
    // `write_named_field`/`field_read_through_model`, with no separate
    // "class object" machinery needed there. A class with no
    // `class_attributes` at all still seeds an empty tagged object — a
    // later `SomeClass.new_attr = v` attribute GAIN is ordinary Python,
    // matching `field_write`'s own "an ordinary Python attribute gain is
    // not a blocker" doc.
    if let Some(model) = self_model {
        enclosing_environment.bind(def.name.id.as_str(), instances::class_object_value(model));
    }
    walk_body(&def.body, None, None, self_model, context, out);
}

/// The one walrus-bound `subprocess.<callee>(...)` call reachable inside
/// an `if`/`elif` TEST, if any — `(target, call)`, the same
/// `Expr::Named::target`/`value` pair `bind_walrus_targets` already
/// destructures, read again here rather than threaded through (this
/// walk is a second, cheap traversal of a pure expression tree, the
/// same posture `bind_walrus_targets`'s own doc already takes for its
/// value-evaluation pass). Mirrors `bind_walrus_targets`'s own recursion
/// shape so a walrus nested anywhere in the test (not only at the top)
/// is found the same way it is already bound; stops at the first
/// `Expr::Named` whose value is an `Expr::Call`, since a walrus can only
/// ever bind once per test in the corpus's own idiom
/// (`level_via_walrus_result`).
fn walrus_bound_call_in_test(expr: &Expr) -> Option<(&ruff_python_ast::ExprName, &ruff_python_ast::ExprCall)> {
    match expr {
        Expr::Named(named) => {
            if let (Expr::Name(target), Expr::Call(call)) = (named.target.as_ref(), named.value.as_ref()) {
                return Some((target, call));
            }
            walrus_bound_call_in_test(named.value.as_ref())
        }
        Expr::BoolOp(op) => op.values.iter().find_map(walrus_bound_call_in_test),
        Expr::BinOp(op) => walrus_bound_call_in_test(op.left.as_ref()).or_else(|| walrus_bound_call_in_test(op.right.as_ref())),
        Expr::UnaryOp(op) => walrus_bound_call_in_test(op.operand.as_ref()),
        Expr::Compare(compare) => walrus_bound_call_in_test(compare.left.as_ref())
            .or_else(|| compare.comparators.iter().find_map(walrus_bound_call_in_test)),
        Expr::Attribute(attribute) => walrus_bound_call_in_test(attribute.value.as_ref()),
        _ => None,
    }
}

/// Offers `arm_body`'s own consumer scan the same foreign-edge
/// recognition a plain `Assign`/`With` statement already gets, for a
/// walrus-bound `subprocess.<callee>(...)` call living in the `if`
/// TEST rather than in the arm body itself
/// (`level_via_walrus_result`'s own shape). Returns the per-consumer-
/// position override map the arm body's own statement loop applies —
/// the SAME map shape and publish/expire discipline
/// `serve_foreign_edge_at`'s callers already keep, so no second
/// mechanism exists for how an override reaches its consumer once
/// recognized.
fn serve_foreign_edge_in_walrus_test(
    test: &Expr,
    arm_body: &[Stmt],
    environment: &Environment,
    context: &WalkContext,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> HashMap<usize, Vec<(TextRange, AbstractValue)>> {
    let mut foreign_edge_overrides = HashMap::new();
    let Some((target, call)) = walrus_bound_call_in_test(test) else {
        return foreign_edge_overrides;
    };
    let Some(outcome) = foreign_edge::foreign_edge_at_walrus_call(
        call,
        target,
        arm_body,
        0,
        environment,
        context.kernel,
        context.entry_directory.as_deref(),
    ) else {
        return foreign_edge_overrides;
    };
    match outcome {
        foreign_edge::ForeignEdgeOutcome::Override { parse_range, value, stdout_override } => {
            if let Some(consumer_position) = arm_body
                .iter()
                .enumerate()
                .find(|(_, statement)| statement.range().contains_range(parse_range))
                .map(|(position, _)| position)
            {
                let mut published = vec![(parse_range, value)];
                // `stdout_override`'s own node sits inside the SAME
                // `json.loads(...)` statement `parse_range` already
                // resolved a position for (`serve_foreign_edge_at`'s own
                // doc on this point).
                if let Some((stdout_range, stdout_value)) = stdout_override {
                    published.push((stdout_range, stdout_value));
                }
                foreign_edge_overrides.insert(consumer_position, published);
            }
        }
        foreign_edge::ForeignEdgeOutcome::Fired { message, range, consumer } => {
            out.push(Finding { range, code: "RTS7001", message });
            // The TS convention taken literally: the fired FileRead
            // edge's sole consumer still binds the artifact's carried
            // return-leg value and judges it for real — one edge, its
            // fire, and a determined read.
            if let Some((consumer_range, value)) = consumer {
                if let Some(consumer_position) = arm_body
                    .iter()
                    .enumerate()
                    .find(|(_, statement)| statement.range().contains_range(consumer_range))
                    .map(|(position, _)| position)
                {
                    foreign_edge_overrides.insert(consumer_position, vec![(consumer_range, value)]);
                }
            }
        }
        foreign_edge::ForeignEdgeOutcome::Decline { message, range } => {
            record_blocker(blocked, range, message, out);
        }
    }
    foreign_edge_overrides
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
///
/// EXCEPTION (`is_admits_none_peel_test`, serves f-type-nodes.py's
/// `optional_annotation`/`pipe_none_annotation`): a provably-false test
/// is never fired when it is the ordinary Optional-PEELING idiom — `if
/// <name> is None:` / `if <name> is not None:` where `<name>` carries a
/// DECLARED refinement that `admits_none` (`Optional[Age]`, `Age | None`).
/// `present: Optional[Age] = 40` then `if present is None:` evaluates the
/// concrete literal 40 against `is None` and comes back provably false —
/// but that falseness is a fact about THIS ONE assignment's own concrete
/// value, not about the DECLARED shape `present` states it carries at
/// every point downstream; peeling a `| None`/`Optional[...]` declaration
/// with an `is None`/`is not None` guard is ordinary, idiomatic narrowing
/// (the same idiom every `X | None` field/parameter read leans on), never
/// dead code, so the dead-branch law must not speak here. This is a
/// narrow, syntactic exception: `none_test_on_helper_that_never_answers_
/// none`'s own row (a-statements.py:400ish) is unaffected because `held`
/// there is bound by a plain `Assign` from a call result, never an
/// AnnAssign, so it carries no entry in `aug_assign_refinements` at all —
/// the exception's own `declared.admits_none` check simply finds nothing
/// and the dead-branch law still fires for it.
fn walk_if(
    if_stmt: &StmtIf,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
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
            if known && !truthy && !is_admits_none_peel_test(test, aug_assign_refinements) {
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
                // FOREIGN EDGE: a walrus-bound `subprocess.<callee>(...)`
                // call living in THIS test (`level_via_walrus_result`'s own
                // shape) is offered to the same recognition an ordinary
                // `Assign`/`With` statement gets — the arm body is where
                // its sole `json.loads(...)` consumer sits.
                let mut foreign_edge_overrides =
                    serve_foreign_edge_in_walrus_test(test, body, &arm_environment, context, blocked, out);
                for (position, stmt) in body.iter().enumerate() {
                    if let Some(published) = foreign_edge_overrides.get(&position) {
                        arm_environment.set_evaluated_node(published.clone());
                    }
                    walk_statement(
                        stmt,
                        return_refinement,
                        yield_refinement,
                        context,
                        &mut arm_environment,
                        aug_assign_refinements,
                        &mut arm_provably_unbound,
                        blocked,
                        out,
                    );
                    arm_environment.set_evaluated_node(Vec::new());
                    foreign_edge_overrides.remove(&position);
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
        // FOREIGN EDGE: same walrus-in-test recognition as the
        // provably-true short-circuit arm above, applied to this
        // ordinary (possibly unknown-truthiness) arm's own body.
        let mut foreign_edge_overrides = test
            .map(|test| serve_foreign_edge_in_walrus_test(test, body, &arm_environment, context, blocked, out))
            .unwrap_or_default();
        for (position, stmt) in body.iter().enumerate() {
            if let Some(published) = foreign_edge_overrides.get(&position) {
                arm_environment.set_evaluated_node(published.clone());
            }
            walk_statement(
                stmt,
                return_refinement,
                yield_refinement,
                context,
                &mut arm_environment,
                aug_assign_refinements,
                &mut arm_provably_unbound,
                blocked,
                out,
            );
            arm_environment.set_evaluated_node(Vec::new());
            foreign_edge_overrides.remove(&position);
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

/// Whether `test` is an `is`/`is not` comparison against a bare `None`
/// literal, with the other side a bare Name carrying a DECLARED
/// refinement (`aug_assign_refinements`, populated at that name's own
/// `AnnAssign`) that admits `None` — the ordinary `Optional[X]`/`X | None`
/// peeling idiom `walk_if`'s DEAD-BRANCH LAW must never treat as dead
/// code, however provably-false the test reads against one concrete
/// assignment (see that law's own doc for the full reasoning and the
/// `none_test_on_helper_that_never_answers_none` row this exception must
/// NOT touch). A chained comparison (more than one `ops`/`comparators`
/// entry), a non-`Is`/`IsNot` op, a `None` on both sides, or a name with
/// no `aug_assign_refinements` entry (or one that does not admit `None`)
/// all fall through to `false` — the dead-branch law fires for every one
/// of those shapes exactly as before this exception existed.
fn is_admits_none_peel_test(test: &Expr, aug_assign_refinements: &HashMap<String, DeclaredRefinement>) -> bool {
    let Expr::Compare(compare) = test else {
        return false;
    };
    let ([op], [comparator]) = (&*compare.ops, &*compare.comparators) else {
        return false;
    };
    if !matches!(op, CmpOp::Is | CmpOp::IsNot) {
        return false;
    }
    let left_is_none = matches!(compare.left.as_ref(), Expr::NoneLiteral(_));
    let right_is_none = matches!(comparator, Expr::NoneLiteral(_));
    if left_is_none == right_is_none {
        // both None, or neither — not the peel shape at all
        return false;
    }
    let name_side = if right_is_none { compare.left.as_ref() } else { comparator };
    let Expr::Name(name) = name_side else {
        return false;
    };
    aug_assign_refinements
        .get(name.id.as_str())
        .is_some_and(|declared| declared.admits_none)
}

/// Whether a body's last statement is a bare `return`/`raise` — an arm
/// ending this way never falls through to the post-if point, so its
/// environment describes only unreachable code and must not join.
fn arm_terminates(body: &[Stmt]) -> bool {
    matches!(body.last(), Some(Stmt::Return(_)) | Some(Stmt::Raise(_)))
}

/// The same termination test as `arm_terminates`, extended with ONE more
/// proven-terminal shape: the body's last statement is not syntactically a
/// `return`/`raise`, but the walk's OWN provable-raise machinery already
/// recorded an RTS7001 finding whose range falls inside that last
/// statement's own range (e.g. `bind_known_sequence_target`'s arity-mismatch
/// fire, anchored at the destructured value's range, inside an `Assign`
/// statement that is itself the try body's last statement). A body ending
/// this way also describes only unreachable code past that point — the
/// exception is provably always raised, so a "falls through normally" path
/// never exists. `findings_before` is `out`'s length captured immediately
/// before this body was walked; only findings recorded DURING that walk are
/// considered. This does not weaken the syntactic check for a body whose
/// last statement was never proven to raise — those still route through
/// `arm_terminates` alone.
fn arm_terminates_or_provably_raises(body: &[Stmt], out: &[Finding], findings_before: usize) -> bool {
    if arm_terminates(body) {
        return true;
    }
    let Some(last) = body.last() else {
        return false;
    };
    let last_range = last.range();
    out[findings_before..]
        .iter()
        .any(|finding| finding.code == "RTS7001" && last_range.contains_range(finding.range))
}

/// RELATIONAL SUM: `total = 0; for x in xs: total += f(x)` over a
/// sequence known only by its ELEMENT SET, optionally followed by a
/// division of that total by the same sequence's own length.
///
/// The division is what forces the lowering. Interval arithmetic knows
/// `total` is in `[0, n]` and the length is in `[1, n]`, and dividing
/// those two enclosures separately gives `[0, n]` — useless. The tight
/// answer needs the RELATION between numerator and denominator, so the
/// accumulation and the division go to the kernel as ONE program
/// (`relational_sum`'s own doc) and the kernel's linear decider narrows
/// the quotient. Nothing is computed here; this function recognizes the
/// shape, asks, and binds what the kernel answered.
///
/// `Consumed` means the accumulation was walked here and the
/// accumulator holds the kernel's total; the following statement, if
/// any, still walks. `ConsumedWithDivision` means one or two FOLLOWING
/// statements — the division alone, or a count-alias assignment plus
/// the division that reads it — were folded into the same program, so
/// the caller skips exactly that many statements rather than walking
/// them a second time. `Declined` leaves everything to `walk_statement`,
/// exactly as before. Recognition — both spellings, and the
/// `for`/`else` gate — happens at the caller; this function receives an
/// already-recognized accumulation.
///
/// A `return` carrying the division is `Consumed`, not
/// `ConsumedWithDivision`: the return still walks and still judges
/// against the enclosing annotation, with the quotient published for
/// its one division node so the surrounding expression (the fixture's
/// `math.sqrt(...)`) evaluates around an already-narrowed value.
///
/// Conservative declines, each one because the fact it would state is
/// not the fact the kernel proved: any statement other than a division-
/// carrying assignment or return sitting immediately after the
/// accumulation, OR after a single count-alias assignment
/// (`count = len(samples)`) immediately after the accumulation (a
/// statement in either of those positions that is not one of these
/// shapes could rebind either name, so nothing further is folded and
/// the division walks ordinarily); a walrus rebinding either name
/// anywhere in that expression; a return whose expression holds the
/// division zero times or more than once (with two, one published
/// answer cannot say which node it belongs to); and a kernel refusal.
fn walk_relational_sum(
    mut recognized: relational_sum::RecognizedAccumulation,
    loop_target: Option<&Expr>,
    bound_at: Option<TextRange>,
    following: &[Stmt],
    environment: &mut Environment,
) -> RelationalSum {
    // The division, when it sits at the very next statement OR one hop
    // later behind a count-alias assignment. Only that immediate
    // lookahead is read: anything else in between could rebind either
    // name, and this pass never reasons about what it did not look at.
    //
    // The count-alias hop: `following.first()` is tried first as the
    // division-carrying statement itself (the direct spelling); when it
    // is instead a plain `<name> = len(<sequence>)` naming THIS
    // accumulation's own sequence (`is_length_alias_assignment`), the
    // alias is recorded (`record_length_alias`) and the division/return
    // match re-runs against `following.get(1)` instead. With the alias
    // at `following.first()` and the division at `following.get(1)`,
    // there is no statement BETWEEN them for
    // `reassigns_alias_or_sequence` to guard — that guard is for a wider
    // gap than this exact one-hop lookahead ever opens.
    //
    // Two shapes carry the division itself. An ASSIGNMENT divides at
    // its top level and names the quotient, so the answer binds to that
    // name and the statement is consumed whole. A RETURN may nest the
    // division anywhere inside the returned expression — the fixture's
    // own `return math.sqrt(total / len(samples))` — so the return is
    // still walked ordinarily, with the quotient published for exactly
    // that one division node (`Environment::set_evaluated_node`) and
    // the surrounding call evaluated around it as usual.
    //
    // A walrus rebinding either name anywhere in the expression
    // declines both shapes: the rebinding happens mid-expression, so
    // the division would be over a value the kernel never tied.
    let mut divided_into: Option<(String, ruff_text_size::TextRange)> = None;
    let mut published_division = None;
    // How many of `following`'s leading statements this walk consumed:
    // 1 for the division alone, 2 when a count-alias assignment was
    // consumed ahead of it. Stays 0 on a decline, so the caller's own
    // skip bookkeeping only ever advances past what was actually folded.
    let mut consumed_statements: usize = 0;
    // The alias name and its own Assign range, held only long enough to
    // bind+publish below once the division actually folds — mirroring
    // `divided_into`'s own tentativeness (an alias with nothing folded
    // behind it is not consumed, so nothing about it is bound either).
    let mut alias_binding: Option<(String, ruff_text_size::TextRange)> = None;
    let mut division_candidate = following.first();
    if let Some(Stmt::Assign(alias_assign)) = following.first() {
        if let Some(alias) = relational_sum::is_length_alias_assignment(alias_assign, &recognized, environment) {
            relational_sum::record_length_alias(&mut recognized, alias.clone());
            alias_binding = Some((alias, alias_assign.range()));
            division_candidate = following.get(1);
            consumed_statements = 1;
        }
    }
    match division_candidate {
        Some(Stmt::Assign(assign)) => {
            if let [Expr::Name(target)] = assign.targets.as_slice() {
                if !rebinds_relational_name(assign.value.as_ref(), &recognized)
                    && relational_sum::fold_division(&mut recognized, assign.value.as_ref(), environment)
                {
                    divided_into = Some((target.id.as_str().to_owned(), assign.range()));
                    consumed_statements += 1;
                }
            }
        }
        Some(Stmt::Return(ret)) => {
            if let Some(value) = ret.value.as_deref() {
                if !rebinds_relational_name(value, &recognized) {
                    if let Some(range) = relational_sum::division_range_in(value, &recognized, environment) {
                        relational_sum::fold_located_division(&mut recognized);
                        published_division = Some(range);
                        consumed_statements += 1;
                    }
                }
            }
        }
        _ => {}
    }
    // The alias hop was tentative until the division actually folded —
    // an alias assignment with nothing behind it to consume must leave
    // `following.first()` for the ordinary walk, exactly as if the alias
    // read had never been tried.
    if divided_into.is_none() && published_division.is_none() {
        consumed_statements = 0;
        alias_binding = None;
    }
    // Plain data dump for the two relational-sum fixtures one exhausted
    // static trace could not tell apart (the const-effect variant
    // determines, the var-effect variant does not, and only the live
    // answer states say why) — gated on REFINEDPY_DEBUG_RELATIONAL so it
    // never runs unasked. Read once per call, matching the crate's other
    // inline `std::env::var` checks (`kernel_path.rs`, `kernel_bridge.rs`)
    // rather than a cached static — this instrument is meant to be
    // removable later, not a permanent flag. `check.rs` never sees the
    // per-slot wire send/receive `ask_walk_relational` performs inside
    // `relational_sum::walk_accumulation` (that call, and the
    // `catch_unwind` around it, live in a different crate), so a genuine
    // "ask panicked" vs. "kernel declined" split is not nameable from
    // here — both collapse to `walk_accumulation` answering `None`. What
    // IS printed is everything `check.rs` actually holds: the recognized
    // names, the entry states as this call is ABOUT to send them
    // (`recognized.entry_states`, `Debug`-formatted — no `Serialize` wire
    // encoder is public outside `refined_kernel`), and the answer as
    // received, or the words "declined" when `walk_accumulation` itself
    // answers `None`.
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        eprintln!(
            "relational_sum: total={} sequence={} entry_states={:?} statements={:?}",
            recognized.total_name, recognized.sequence_name, recognized.entry_states, recognized.statements
        );
    }
    let Some(answer) = relational_sum::walk_accumulation(&recognized) else {
        if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
            eprintln!("relational_sum: declined (walk_accumulation answered None)");
        }
        return RelationalSum::Declined;
    };
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        eprintln!(
            "relational_sum: answered total={:?} quotient={:?}",
            answer.total, answer.quotient
        );
    }
    // The total binds when the kernel answered it; a total whose own
    // enclosure is unbounded (sign-straddling step, unbounded count) is
    // forgotten instead — the ledger's quotient below still stands.
    match answer.total {
        Some(total) => {
            // The SAME publish an ordinary `total = <value>` gets from
            // `evaluate_expression`'s own record_evaluation call
            // (expressions.rs) — this recognizer consumes the whole
            // Assign statement before that ordinary evaluation ever
            // runs, so without this the binding's own position answers
            // no set at `refined_set_at_position` (bound_at is `None`
            // for the For-loop accumulation shape, which has no single
            // Assign statement to record against).
            if let Some(range) = bound_at {
                environment.record_evaluation(range, total.clone());
            }
            environment.bind(&recognized.total_name, total);
        }
        None => environment.forget(&recognized.total_name),
    }
    // The count-alias name (`count = len(samples)`) binds when a division
    // actually folded behind it AND the kernel's own count window is
    // bindable. The SAME publish the total's binding gets above: this
    // recognizer consumes the whole alias Assign before the ordinary
    // evaluator ever runs, so without this the alias binding's own
    // position answers no set at `refined_set_at_position`. A count the
    // machinery cannot state (an empty count window) leaves the alias
    // exactly as today — forgotten, nothing fabricated.
    if let Some((alias_name, alias_range)) = alias_binding {
        match answer.count {
            Some(count) => {
                environment.record_evaluation(alias_range, count.clone());
                environment.bind(&alias_name, count);
            }
            None => environment.forget(&alias_name),
        }
    }
    // The quotient rides its own slot, so the divided name carries
    // exactly what the kernel proved — or, where the kernel answered the
    // total but not the quotient, nothing at all rather than a guess.
    //
    // The RETURN itself is never skipped — it always still walks at its
    // own position, judging against the enclosing annotation — so a
    // return-with-division fold's own statement does not count toward
    // `skip_statements`; only a count-alias assignment consumed AHEAD of
    // that return does. `consumed_statements` counted the return as 1
    // of its own leading-statement tally, so that one is subtracted back
    // out here before it becomes a caller-facing skip count.
    let outcome = match (divided_into, published_division) {
        (Some((target, bound_range)), _) => {
            match answer.quotient {
                Some(quotient) => {
                    // The SAME publish the total's binding gets above:
                    // this recognizer consumes the whole Assign before
                    // the ordinary evaluation runs, so without this the
                    // quotient binding's own position answers no set at
                    // `refined_set_at_position`.
                    environment.record_evaluation(bound_range, quotient.clone());
                    environment.bind(&target, quotient)
                }
                None => environment.forget(&target),
            }
            RelationalSum::ConsumedWithDivision { skip_statements: consumed_statements }
        }
        // The return is NOT consumed — it still walks, judging against
        // the enclosing `-> Annotation` as always. What changes is that
        // its one division node reads the kernel's narrowed quotient
        // instead of being evaluated from two untied enclosures. A
        // quotient the kernel declined publishes nothing, so that node
        // evaluates ordinarily: never a weaker path than before.
        (None, Some(range)) => {
            if let Some(quotient) = answer.quotient {
                environment.set_evaluated_node(vec![(range, quotient)]);
            }
            // A count-alias assignment sat ahead of this return and was
            // folded in — that ONE statement still needs skipping, even
            // though the return itself does not. `consumed_statements`
            // is exactly 1 (the return alone) when no alias hop ran, and
            // exactly 2 (alias plus return) when one did.
            let alias_statements_ahead = consumed_statements.saturating_sub(1);
            if alias_statements_ahead > 0 {
                RelationalSum::ConsumedWithDivision { skip_statements: alias_statements_ahead }
            } else {
                RelationalSum::Consumed
            }
        }
        (None, None) => RelationalSum::Consumed,
    };
    // The loop variable outlives the loop in CPython (compound_stmts,
    // "the for statement"), but this pass never ran a concrete
    // iteration, so which element it ended on is not a fact here — it
    // is forgotten rather than claimed. The generator spelling has no
    // loop target: the generator's own variable never escapes it.
    if let Some(Expr::Name(loop_variable)) = loop_target {
        environment.forget(loop_variable.id.as_str());
    }
    outcome
}

/// RELATIONAL SUM AT A BARE RETURN: `return sum(<elt> for <var> in
/// <seq>)` with no assignment anywhere in sight — the whole body is one
/// statement. `recognize_generator_sum` only ever reads an `Assign`, so
/// this exact expression, spelled `total = sum(...); return total`,
/// already recognizes and judges; spelled as a direct `return`, it fell
/// through to the ordinary evaluator's `sum_call_over_star` row, which
/// needs a known-sign hull and declines on a sign-straddling element.
///
/// There is no name to bind the total into — it routes straight to the
/// return's own evaluated-node seam (`Environment::set_evaluated_node`),
/// the same publish `walk_relational_sum`'s return-with-division arm
/// already uses, except the published range is the WHOLE returned
/// expression rather than a division nested inside it, since the call
/// to `sum` IS the returned expression here. `walk_return` (called by
/// the ordinary `walk_statement` dispatch right after this) then reads
/// the publish at `evaluate_expression`'s own dispatch head and judges
/// it against the declared return set exactly as any other value would
/// be.
///
/// A kernel refusal, or a total the kernel could not bind, publishes
/// nothing: `evaluate_expression` then runs unchanged and the ordinary
/// `sum_call_over_star` row answers whatever it already would.
fn publish_relational_sum_return(ret: &StmtReturn, environment: &mut Environment) {
    let Some(value) = ret.value.as_deref() else {
        return;
    };
    let Some(recognized) = relational_sum::recognize_generator_sum_in_return(value, environment) else {
        return;
    };
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        eprintln!(
            "relational_sum: (bare return) sequence={} entry_states={:?} statements={:?}",
            recognized.sequence_name, recognized.entry_states, recognized.statements
        );
    }
    let Some(answer) = relational_sum::walk_accumulation(&recognized) else {
        if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
            eprintln!("relational_sum: (bare return) declined (walk_accumulation answered None)");
        }
        return;
    };
    if let Some(total) = answer.total {
        environment.set_evaluated_node(vec![(value.range(), total)]);
    }
}

/// What `walk_relational_sum` did with a recognized accumulation.
enum RelationalSum {
    /// Not this pass's shape — the ordinary walk runs.
    Declined,
    /// The accumulation was walked here and the accumulator holds the
    /// total. Whatever follows still walks — including a `return` whose
    /// division was folded, which reads its published quotient.
    Consumed,
    /// The accumulation AND one or two of the leading statements of
    /// `following` were walked as one kernel program, so the caller
    /// skips exactly that many statements rather than walking them a
    /// second time. The count is 1 for a division-carrying ASSIGNMENT
    /// alone, or 2 when a count-alias assignment
    /// (`count = len(samples)`) was folded ahead of it — the alias
    /// statement is consumed the same as the division statement it
    /// feeds, even though neither is a `return` (a `return` is never
    /// skipped; it always still walks, so a folded return-with-division
    /// reports `Consumed` for the division itself, with this variant
    /// used only when a count-alias assignment preceded it — see the
    /// count-alias-then-return arm below).
    ConsumedWithDivision { skip_statements: usize },
}

/// Whether an expression rebinds — through a walrus — either name a
/// recognized accumulation states its relation over. Such a rebinding
/// happens mid-expression, so the division would be taken over a value
/// the kernel's relation was never tied to.
fn rebinds_relational_name(
    expression: &Expr,
    recognized: &relational_sum::RecognizedAccumulation,
) -> bool {
    let mut names = HashSet::new();
    collect_walrus_names(expression, &mut names);
    names.contains(&recognized.total_name) || names.contains(&recognized.sequence_name)
}

/// `for`/`while`: `loops::loop_final_environment` concretely executes the
/// bounded shapes it recognizes (literal list/tuple/range iterables,
/// bounded counter `while`s), judging every declared-slot write inside
/// the body against `aug_assign_refinements` as it runs (loops.rs's own
/// contract — a fire lands in `judged_fires`, deduped by statement
/// range) and reporting whether the loop's `else` clause RUNS
/// (`else_runs`). `Some((env, else_runs))` replaces the environment
/// outright and the statement is consumed with no blocker for the
/// `for`/`while` itself; this function then owns the `orelse` body
/// itself — loops.rs never runs it: DEAD-BRANCH LAW's own sibling, the
/// LOOP ELSE + DEAD-ELSE LAW (serves a-statements:446/472/486): when
/// `else_runs`, `orelse` walks through `walk_statement` exactly like any
/// other body (fully judged — this is what makes an else-arm's own
/// out-of-set write fire); when `!else_runs` (every execution provably
/// `break`s), `orelse` never runs, and this function fires RTS7001 at
/// the orelse body's own first statement instead, naming why. `None`
/// from `loop_final_environment` means the shape is outside what that
/// module can run; the walk keeps its own blocker AND forgets every name
/// the loop statement binds anywhere (its target plus every name its
/// body/orelse bind, PLUS every attribute-call/subscript-store receiver
/// the unmodeled body only ever MUTATES — `forget_mutated_receivers_in_stmt`),
/// so a stale pre-loop fact never survives an unmodeled loop that may
/// have rebound it. The blocker is recorded regardless of whether THIS
/// body has already recorded a declared return/yield position or a
/// declared slot (`aug_assign_refinements`) at the loop's OWN position —
/// a slot declared LATER in the same straight-line body (an AnnAssign
/// after the loop, e.g. `for_loop_over_an_unknown_iterable_blocks_and_
/// forgets_its_stale_binding`'s own `check: Age = total`) still has a
/// real sink waiting on this loop's answer, and `aug_assign_refinements`
/// only ever holds what has been WALKED SO FAR — checking it at the
/// loop's own position can never see a later declaration. Recording the
/// blocker unconditionally is sound either way: a body with truly no
/// checked position anywhere still records one blocker (first-blocker-
/// wins, same as every other construct this walk cannot yet handle),
/// never a second one.
///
/// A `Some` answer is not always blocker-free: a `for` loop's own
/// abstract pass (`loops.rs`'s `stabilized_join`) may reach a real
/// stopping point while still havocing one or more written names to
/// `unknown()`, because their value never settled to a fixed point
/// across its two judged passes — `LoopAnswer.widened_names`, sorted for
/// a reproducible first name. This function records THAT as the body's
/// blocker before ever looking at `return_refinement`, so a bare
/// `-> float`/`-> int`/`-> str` return (unreadable to `declared_
/// refinement`, `typereading.rs`'s own doc) never leaves this loop's own
/// genuine undetermined value unnamed.
fn walk_loop(
    stmt: &Stmt,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let mut judged_fires: Vec<(TextRange, String)> = Vec::new();
    let result = loop_final_environment(stmt, environment, context.kernel, aug_assign_refinements, &mut judged_fires);
    for (range, message) in judged_fires {
        out.push(Finding {
            range,
            code: "RTS7001",
            message,
        });
    }
    if let Some(LoopAnswer { environment: final_env, else_runs, returned, widened_names }) = result {
        *environment = final_env;
        // A `for` loop's own abstract pass reached a real stopping
        // point, but one or more of its OWN written names never settled
        // to a fixed point across its two judged passes
        // (`loops.rs::stabilized_join`'s own doc) and reads as
        // `unknown()` from here on — a genuine blocker this loop itself
        // is the first construct to name, independent of whether the
        // enclosing function's own `-> Annotation` is readable (a bare
        // `-> float` leaves `return_refinement` `None`, which is
        // `walk_return`'s own signal to skip JUDGING a value — it must
        // not also silence THIS body's own record of why that value is
        // unreadable in the first place). `widened_names` is sorted
        // (`stabilized_join`'s own doc), so the first entry is a stable,
        // reproducible choice across runs.
        if let Some(first_widened) = widened_names.first() {
            record_blocker(blocked, stmt.range(), loop_accumulation_did_not_stabilize(first_widened), out);
        }
        // RETURN-THROUGH-LOOP CHANNEL: a value SOME concrete iteration
        // returned judges against the enclosing function's own
        // `-> Annotation`, exactly as `walk_return` judges a
        // straight-line `return` — the same Fire/Silent/Undetermined
        // law, anchored at the range loops.rs carried (the returned
        // expression's own range, or the bare `return` statement's own
        // range). A BARE return (`value: None`, loops.rs's own
        // `walk_return`-matching convention) judges nothing, same as a
        // straight-line bare `return`. Additive: the walk still proceeds
        // to the `else`/dead-else law below on the SAME environment,
        // since loops.rs never tries to prove the statement after the
        // loop unreachable (see `LoopAnswer`'s own doc).
        if let Some((Some(value), range)) = returned {
            if let Some(declared) = return_refinement {
                match judge(&value, declared, context.kernel) {
                    Verdict::Fire(message) => out.push(Finding { range, code: "RTS7001", message }),
                    Verdict::Silent => {}
                    Verdict::Undetermined(sentence) => {
                        record_blocker(blocked, range, sentence, out);
                    }
                }
            }
        }
        let orelse: &[Stmt] = match stmt {
            Stmt::For(for_stmt) => for_stmt.orelse.as_slice(),
            Stmt::While(while_stmt) => while_stmt.orelse.as_slice(),
            _ => &[],
        };
        if orelse.is_empty() {
            return;
        }
        if !else_runs {
            out.push(Finding {
                range: orelse[0].range(),
                code: "RTS7001",
                message: "this else arm provably never runs: the loop above always breaks".to_owned(),
            });
            return;
        }
        let mut orelse_provably_unbound: HashSet<String> = HashSet::new();
        for orelse_stmt in orelse {
            walk_statement(
                orelse_stmt,
                return_refinement,
                yield_refinement,
                context,
                environment,
                aug_assign_refinements,
                &mut orelse_provably_unbound,
                blocked,
                out,
            );
        }
        return;
    }
    record_blocker(
        blocked,
        stmt.range(),
        format!("{} is not yet walked", statement_kind_name(stmt)),
        out,
    );
    forget_names_bound_by_stmt(stmt, environment);
    forget_mutated_receivers_in_stmt(stmt, environment);
}

/// `match subject: case ... case ...` (compound_stmts.rst, "The `match`
/// statement"): `match_arms::match_taken_environment` decides, for a
/// KNOWN subject, which single arm CPython would take. `Some((index,
/// arm_env))` adopts that arm's environment (already carrying any
/// capture-pattern bindings match_arms.rs made) and walks ONLY that
/// case's body statements in order — the other arms are not taken and
/// are never walked, matching CPython's own first-match semantics.
///
/// MATCH JOIN FALLBACK (serves b-body-expressions.py:889/900 — a class
/// pattern like `case int() as n:`, which `match_arms.rs` cannot decide
/// TAKEN/NOT-TAKEN this wave, `Pattern::MatchClass` being Undecidable
/// there regardless of the subject): when `match_taken_environment`
/// answers `None`, this function no longer records a blocker outright.
/// Instead it walks EVERY case on its OWN fork of the incoming
/// environment — `match_arms::pattern_bound_captures` names what a
/// pattern captures SYNTACTICALLY (a question `pattern_outcome` never
/// has to answer, unlike TAKEN/NOT-TAKEN) AND binds each captured name
/// to the most specific value it can prove (an exact literal proof, a
/// sequence element/mapping value/class field read off a KNOWN
/// container subject, or — when none of those apply — `unknown()`,
/// never a guess) — a guarded case still walks under the same fork (the
/// guard's own truth is not decided either; over-approximating past it
/// is sound, never wrong). Every surviving fork (one whose last
/// statement is not `return`/`raise`, `arm_terminates`) joins exactly
/// as `walk_if` joins its own arms. The blocker is kept ONLY when
/// `pattern_bound_captures` itself cannot even name a case's own
/// captures (a `MatchSequence`/`MatchMapping`/`MatchClass` shape past
/// its own flat bare-capture scope, or a `MatchClass` with POSITIONAL
/// sub-patterns — `match_arms.rs`'s own doc names exactly which shapes
/// those are) — that one case, and every case after it (CPython could
/// have reached any of them), forgets its own bound names instead of
/// joining, and the match statement's own blocker still records once.
fn walk_match(
    match_stmt: &StmtMatch,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let subject_value = evaluate_expression(match_stmt.subject.as_ref(), environment, context.kernel);
    // The subject's own bare name, when it has one, so a split arm can
    // rebind it to the arm's intersection — the same name the fallback
    // path below already reads off `Expr::Name`.
    let subject_name = match match_stmt.subject.as_ref() {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    };
    {
        // The arm-body walker match_taken_environment calls per decided
        // arm — the identical per-arm walk `walk_if` runs for a fork:
        // a fresh provably-unbound set, every statement through
        // walk_statement, and the arm's own termination read off its
        // last statement. This walker can always walk, so it never
        // answers `None` (the decline arm is for callers whose own
        // interpreters can genuinely refuse, `summaries.rs`).
        let mut walk_arm = |body: &[Stmt], arm_environment: &mut Environment| -> Option<bool> {
            let mut arm_provably_unbound: HashSet<String> = HashSet::new();
            for stmt in body {
                walk_statement(
                    stmt,
                    return_refinement,
                    yield_refinement,
                    context,
                    arm_environment,
                    aug_assign_refinements,
                    &mut arm_provably_unbound,
                    blocked,
                    out,
                );
            }
            Some(!arm_terminates(body))
        };
        if let Some((post_environment, _falls_through)) = match_taken_environment(
            &subject_value,
            subject_name,
            &match_stmt.cases,
            environment,
            context.kernel,
            &mut walk_arm,
        ) {
            *environment = post_environment;
            return;
        }
    }

    let mut surviving: Vec<Environment> = Vec::new();
    let mut every_case_nameable = true;
    for case in &match_stmt.cases {
        if !every_case_nameable {
            // an earlier case's own captures could not be named — CPython
            // might have reached THIS case too (or any later one), so its
            // bound names are equally unknown from here on
            forget_names_bound_in_body(&case.body, environment);
            forget_mutated_receivers_in_body(&case.body, environment);
            continue;
        }
        let Some(bound_captures) =
            match_arms::pattern_bound_captures(&case.pattern, &subject_value, environment, context.kernel)
        else {
            every_case_nameable = false;
            record_blocker(
                blocked,
                case.pattern.range(),
                "this match arm's own pattern does not yet name its captures".to_owned(),
                out,
            );
            forget_names_bound_in_body(&case.body, environment);
            forget_mutated_receivers_in_body(&case.body, environment);
            continue;
        };
        let mut arm_env = environment.fork();
        // PATTERN-PROVED NARROWING: every capture `pattern_bound_
        // captures` names binds to what its own position/key/field (or,
        // for a literal/singleton/or/as pattern with none of those, the
        // pattern's own PROVED value — `match_arms::pattern_bound_
        // captures`'s own doc) states about a taken arm, tighter than
        // the coarse pre-match claim `subject_value` carries alone. The
        // SUBJECT NAME ITSELF (when the match subject is a bare Name,
        // e.g. `match pick:`) rebinds to `pattern_proved_value`'s own
        // whole-pattern proof, since a body that reads the subject name
        // directly inside a literal arm (`case 18 | 21 | 40: return
        // pick`) is reading exactly what the pattern just proved. A
        // pattern proving nothing whole (a bare capture, a sequence/
        // mapping/class pattern) leaves the subject name unchanged —
        // the honest, unnarrowed subject claim.
        for (name, value) in &bound_captures {
            arm_env.bind(name, value.clone());
        }
        let narrowed = match_arms::pattern_proved_value(&case.pattern, &arm_env, context.kernel);
        if let Expr::Name(subject_name) = match_stmt.subject.as_ref() {
            if let Some(narrowed) = &narrowed {
                arm_env.bind(subject_name.id.as_str(), narrowed.clone());
            }
        }
        // GUARD NARROWING: `case x if x >= 0 and x <= 120:` — the guard
        // is a boolean expression over names the pattern already bound
        // (captures, or the subject itself above); it narrows exactly
        // the way `walk_if`'s own test narrows an `if` arm
        // (`narrowing::assume`, mission point: "a guard narrows"). The
        // guard's own TRUTH is not decided here (unlike `arm_outcome`'s
        // TAKEN/NOT-TAKEN reading) — this walk already treats every
        // case as reachable in the join fallback, so the guard is
        // assumed TRUE for the body it gates, sound because CPython
        // only runs this body when the guard is in fact true.
        if let Some(guard) = case.guard.as_deref() {
            arm_env = assume(guard, arm_env, context.kernel, true);
        }
        let mut arm_provably_unbound: HashSet<String> = HashSet::new();
        for stmt in &case.body {
            walk_statement(
                stmt,
                return_refinement,
                yield_refinement,
                context,
                &mut arm_env,
                aug_assign_refinements,
                &mut arm_provably_unbound,
                blocked,
                out,
            );
        }
        if !arm_terminates(&case.body) {
            surviving.push(arm_env);
        }
    }

    if !every_case_nameable {
        return;
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

/// `with EXPRESSION as TARGET: SUITE` (compound_stmts.rst, "The `with`
/// statement"): step 5 of the with-statement's own execution order binds
/// TARGET to `__enter__`'s (or, for `async with` — `with_stmt.is_async`,
/// the collapsed sync/async node ruff's own generated.rs doc states —
/// `__aenter__`'s) RETURN VALUE. When the context expression reads as a
/// known INSTANCE (`Kind::Object` with a non-empty `source` naming a
/// `ClassModel` — `instances::judge_construction`'s own tag, the same
/// shape `instance_method_call_result` already reads for a statement-side
/// method call) and that class declares the matching enter method,
/// `instances::method_call_result` interprets its body the same way any
/// other zero-argument method call on a known instance does: `Some`
/// REBINDS the receiver (when it is a bare Name) to the returned working
/// instance and binds TARGET to the method's own return value —
/// `with device() as handle:` (`_Device.__enter__` returns `self`) is
/// exactly this shape. `None` (an unmodeled receiver, a class with no
/// matching enter method, a method body/parameter shape outside the
/// restricted interpreter) forgets TARGET instead — the honest answer
/// when this walk cannot know `__enter__`'s own return value, unchanged
/// from before this law. The body then walks inline, on the SAME
/// environment, with no blocker for the with statement itself.
fn walk_with(
    with_stmt: &StmtWith,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    for item in &with_stmt.items {
        let receiver = evaluate_expression(&item.context_expr, environment, context.kernel);
        let Some(target) = item.optional_vars.as_deref() else {
            continue;
        };
        match enter_method_result(&receiver, with_stmt.is_async, &item.context_expr, context, environment) {
            Some(entered) => bind_with_target(target, entered, environment),
            None => forget_target_names(target, environment),
        }
    }
    let mut with_provably_unbound: HashSet<String> = HashSet::new();
    // FOREIGN EDGE: a temp-file leg (or any other recognized crossing
    // call) nested inside this with-block's own body is offered to
    // `serve_foreign_edge_at` the same way the top-level body loop offers
    // it — `with_stmt.body` is itself a statement list, so nesting one
    // level inside another `with` (`level_via_nested_tempdir`) no longer
    // removes it from every scan `foreign_edge_at` ever runs.
    let mut foreign_edge_overrides: HashMap<usize, Vec<(TextRange, AbstractValue)>> = HashMap::new();
    for (position, stmt) in with_stmt.body.iter().enumerate() {
        serve_foreign_edge_at(&with_stmt.body, position, environment, context, blocked, out, &mut foreign_edge_overrides);
        if let Some(published) = foreign_edge_overrides.get(&position) {
            environment.set_evaluated_node(published.clone());
        }
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
            context,
            environment,
            aug_assign_refinements,
            &mut with_provably_unbound,
            blocked,
            out,
        );
        environment.set_evaluated_node(Vec::new());
        foreign_edge_overrides.remove(&position);
    }
}

/// `EXPRESSION.__enter__()` (or `.__aenter__()` when `is_async`) on a
/// known instance — the same receiver-tag/method-lookup/interpretation
/// chain `instance_method_call_result` already runs for a statement-side
/// `name.method(...)` call, reused here with a FIXED, zero-argument
/// method name instead of reading one off the call syntax (`with`'s own
/// context expression is not itself a method call — `device()` names a
/// constructor or a same-module def, never `.__enter__` directly; the
/// PROTOCOL supplies that name, per compound_stmts.rst's with-statement
/// execution steps). `receiver_expr` is threaded through so a bare-Name
/// context expression (`with handle_holder as handle:`, the receiver
/// already a local) can have its OWN binding rebound to the entered
/// working instance, the same "the receiver survives a self-mutating
/// method call" law `instance_method_call_result` keeps; a context
/// expression that is not itself a bare Name (`with device() as handle:`)
/// has no environment slot to rebind, so only the returned value is
/// reported. `None` for a non-instance receiver, a class with no matching
/// enter method, or a method body/parameter shape `method_call_result`
/// itself declines.
fn enter_method_result(
    receiver: &AbstractValue,
    is_async: bool,
    receiver_expr: &Expr,
    context: &WalkContext,
    environment: &mut Environment,
) -> Option<AbstractValue> {
    if receiver.kind != Kind::Object || receiver.source.is_empty() {
        return None;
    }
    let model = context.classes.get(receiver.source.as_str())?;
    let method_name = if is_async { "__aenter__" } else { "__enter__" };
    let method = instances::method_def_of(model, method_name)?;
    let (new_instance, result) = instances::method_call_result(
        receiver,
        model,
        method,
        &[],
        Some(&context.functions),
        Some(&context.classes),
        Some(&context.datetime_imports),
        context.kernel,
        environment.call_depth(),
    )?;
    if let Expr::Name(receiver_name) = receiver_expr {
        environment.bind(receiver_name.id.as_str(), new_instance);
    }
    Some(result)
}

/// Binds a `with ... as TARGET:` target to `__enter__`'s own return
/// value: a bare Name binds directly. Every other target shape (a
/// tuple/list/starred unpack — `with cm() as (a, b):`, out of this
/// corpus's own rows) FORGETS every name it names instead — this
/// function has no positional unpack rule for `__enter__`'s single
/// return value the way `bind_for_target`'s tuple arm has for a
/// known-arity iterate, so the honest answer is the same "no fact
/// survives an unmodeled target" rule `forget_target_names` already
/// states everywhere else in this file, never a silent stale-value
/// pass-through.
fn bind_with_target(target: &Expr, value: AbstractValue, environment: &mut Environment) {
    match target {
        Expr::Name(name) => environment.bind(name.id.as_str(), value),
        _ => forget_target_names(target, environment),
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
///
/// Returns whether the try statement itself provably never falls
/// through: zero surviving arms means every path either raises past an
/// uncaught exception or terminates inside its own handler — the exact
/// fact `surviving.is_empty()` already computes for the join below,
/// surfaced so the caller's own body loop can stop walking statements
/// that follow, the same way it already stops after a bare `return`/
/// `raise` (`arm_terminates`). A `finally` clause still always runs
/// first regardless — CPython runs `finally` even when nothing survives
/// the try/except — so this only governs statements AFTER the whole
/// try/except/else/finally construct, never `finalbody` itself.
fn walk_try(
    try_stmt: &StmtTry,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> bool {
    let mut surviving: Vec<Environment> = Vec::new();

    let mut try_env = environment.fork();
    let mut try_provably_unbound: HashSet<String> = HashSet::new();
    let try_body_findings_before = out.len();
    for stmt in &try_stmt.body {
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
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
    let orelse_findings_before = out.len();
    for stmt in &try_stmt.orelse {
        walk_statement(
            stmt,
            return_refinement,
            yield_refinement,
            context,
            &mut try_env,
            aug_assign_refinements,
            &mut try_provably_unbound,
            blocked,
            out,
        );
    }
    let (try_path_terminal_body, terminal_findings_before) = if try_stmt.orelse.is_empty() {
        (try_stmt.body.as_slice(), try_body_findings_before)
    } else {
        (try_stmt.orelse.as_slice(), orelse_findings_before)
    };
    if !arm_terminates_or_provably_raises(try_path_terminal_body, out, terminal_findings_before) {
        surviving.push(try_env);
    }

    for handler in &try_stmt.handlers {
        let ExceptHandler::ExceptHandler(handler) = handler;
        let mut handler_env = environment.fork();
        for stmt in &try_stmt.body {
            forget_names_bound_by_stmt(stmt, &mut handler_env);
        }
        // HANDLER AS-NAME: `except Exception as error:` binds `error` to
        // a caught-exception value at handler entry — not a forget — so
        // a read inside the handler body (e.g. `try_except_binding`'s
        // `age = error`) has something to judge rather than reading
        // Undetermined. `caught_exception_value` gives the MOST SPECIFIC
        // value this walk can prove: a tagged, `args`-carrying (and,
        // where a sole matching `raise ... from` names it, `__cause__`-
        // carrying) exception when the try body's own raise is
        // findable, else the plain opaque marker every other unmodeled
        // instance already reads as Unknown.
        if let Some(name) = handler.name.as_ref() {
            let caught = caught_exception_value(handler, &try_stmt.body, environment, context);
            handler_env.bind(name.id.as_str(), caught);
        }
        let mut handler_provably_unbound: HashSet<String> = HashSet::new();
        let handler_findings_before = out.len();
        for stmt in &handler.body {
            walk_statement(
                stmt,
                return_refinement,
                yield_refinement,
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
        if !arm_terminates_or_provably_raises(&handler.body, out, handler_findings_before) {
            surviving.push(handler_env);
        }
    }

    let nothing_survives = surviving.is_empty();
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
            yield_refinement,
            context,
            environment,
            aug_assign_refinements,
            &mut finally_provably_unbound,
            blocked,
            out,
        );
    }
    nothing_survives
}

/// The value one `except <type> as <name>:` handler's own `<name>`
/// binds — the most specific value this walk can prove about the
/// exception CPython would actually deliver there. `handler`'s own
/// `type_` (a bare `Name`, the only shape this reader matches — a
/// tuple-of-types `except (A, B):` or an attribute `except mod.Err:`
/// falls through to the fieldless answer) names the exception CLASS the
/// search below looks for: `find_raise_from`, walking `try_body` in
/// source order (recursing into a nested `Stmt::Try`'s own body, since
/// `raise ... from` can sit arbitrarily deep — j-stdlib-surfaces.py's
/// own `exception_cause` nests one level), returns the SOLE `Stmt::Raise`
/// whose `exc` is a call to that exact class name, or `None` when there
/// is none or more than one (an ambiguous try body proves no ONE raise
/// reaches this handler over another). A found raise's `exc` evaluates
/// through the ordinary `evaluate_expression` path — already routed
/// through `exception_construction_value` for a recognized builtin
/// exception constructor (`expressions.rs`'s own `is_builtin_exception_
/// constructor` gate) — and its own `cause`, when present, resolves
/// through `resolve_cause_name`: a bare Name matching an ENCLOSING
/// handler's own `as`-name recurses into THAT handler's nested try via
/// this same function (j-stdlib-surfaces.py's `inner` naming the INNER
/// try's own caught `ValueError`); any other cause shape (a fresh
/// construction, an unrecognized name) evaluates plainly against
/// `environment`. No raise found, no cause resolvable, or `exc` itself
/// not a recognized exception constructor: `fieldless_exception_value`
/// — the honest "an exception, but this walk cannot prove its fields"
/// answer, never the fully opaque `opaque_value` (which cannot even be
/// recognized as an exception downstream).
fn caught_exception_value(
    handler: &ExceptHandlerExceptHandler,
    try_body: &[Stmt],
    environment: &Environment,
    context: &WalkContext,
) -> AbstractValue {
    let Some(Expr::Name(caught_type)) = handler.type_.as_deref() else {
        return fieldless_exception_value();
    };
    let Some(raise) = find_raise_from(try_body, caught_type.id.as_str()) else {
        return fieldless_exception_value();
    };
    let Some(exc) = raise.exc.as_deref() else {
        return fieldless_exception_value();
    };
    let mut value = evaluate_expression(exc, environment, context.kernel);
    if value.kind != Kind::Object || value.source != "exception" {
        return fieldless_exception_value();
    }
    if let Some(cause) = raise.cause.as_deref() {
        let cause_value = resolve_cause_name(cause, try_body, environment, context);
        value.keys.push(ObjectKey {
            name: "__cause__".to_owned(),
            numeric: false,
            value: cause_value,
        });
    }
    value
}

/// The sole `raise <exc> from <cause>` (or bare `raise <exc>`) whose
/// `exc` is a call to `class_name`, searched in source order through
/// `body`'s own statements — recursing into a nested `Stmt::Try`'s
/// `body`/`handlers`/`orelse`/`finalbody` (every place CPython itself
/// could reach a raise from within a try construct) and an `if`
/// statement's own arms (the same shape `walk_if` itself walks
/// unconditionally). `None` when no raise names that class, or more
/// than one does — this function proves nothing about WHICH raise
/// reaches the handler when the body's own control flow leaves that
/// ambiguous, so it answers only the unambiguous one-match case.
fn find_raise_from<'a>(body: &'a [Stmt], class_name: &str) -> Option<&'a StmtRaise> {
    let mut found: Option<&'a StmtRaise> = None;
    for stmt in body {
        for raise in raises_in_stmt(stmt) {
            let Some(Expr::Call(call)) = raise.exc.as_deref() else {
                continue;
            };
            let Expr::Name(callee) = call.func.as_ref() else {
                continue;
            };
            if callee.id.as_str() != class_name {
                continue;
            }
            if found.is_some() {
                // more than one raise names this class — ambiguous
                return None;
            }
            found = Some(raise);
        }
    }
    found
}

/// Every `Stmt::Raise` reachable inside one statement, recursing into
/// the compound shapes a raise can sit under — `Stmt::Try` (its own
/// `body`/`handlers`/`orelse`/`finalbody`, in that order) and `Stmt::If`
/// (its own `body` plus every `elif`/`else` clause's own body) — the two
/// shapes this corpus's own rows nest a raise inside. Any other compound
/// statement (a `for`/`while`/`with`/`match`) is not walked into: this
/// reader is scoped to the try/if nesting `caught_exception_value`'s own
/// rows use, not a general statement-tree walk.
fn raises_in_stmt(stmt: &Stmt) -> Vec<&StmtRaise> {
    match stmt {
        Stmt::Raise(raise) => vec![raise],
        Stmt::Try(try_stmt) => {
            let mut found = Vec::new();
            for inner in &try_stmt.body {
                found.extend(raises_in_stmt(inner));
            }
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    found.extend(raises_in_stmt(inner));
                }
            }
            for inner in &try_stmt.orelse {
                found.extend(raises_in_stmt(inner));
            }
            for inner in &try_stmt.finalbody {
                found.extend(raises_in_stmt(inner));
            }
            found
        }
        Stmt::If(if_stmt) => {
            let mut found = Vec::new();
            for inner in &if_stmt.body {
                found.extend(raises_in_stmt(inner));
            }
            for clause in &if_stmt.elif_else_clauses {
                for inner in &clause.body {
                    found.extend(raises_in_stmt(inner));
                }
            }
            found
        }
        _ => Vec::new(),
    }
}

/// The value a `raise ... from <cause>` expression's own `cause` names —
/// `caught_exception_value`'s own cause-resolution step. A bare Name
/// matching an ENCLOSING `except ... as <name>:` handler nested inside
/// `try_body` recurses into that handler's own nested try (its `type_`
/// naming which class to search for, `caught_exception_value` again),
/// so a chain of `raise ... from` across nested try/except blocks
/// resolves end to end (j-stdlib-surfaces.py's `inner` naming the INNER
/// try's own caught `ValueError`). Any other cause shape (a fresh
/// construction, a name this search cannot trace to an enclosing
/// handler) evaluates plainly through `evaluate_expression` — sound
/// either way, since an unrecognized cause still reads SOME value, just
/// not necessarily the exact tagged exception shape.
fn resolve_cause_name(cause: &Expr, try_body: &[Stmt], environment: &Environment, context: &WalkContext) -> AbstractValue {
    if let Expr::Name(cause_name) = cause {
        if let Some((nested_try, nested_handler)) = enclosing_handler_named(try_body, cause_name.id.as_str()) {
            return caught_exception_value(nested_handler, &nested_try.body, environment, context);
        }
    }
    evaluate_expression(cause, environment, context.kernel)
}

/// The nested `Stmt::Try` and its own `except ... as <name>:` handler,
/// searched inside `body` — `resolve_cause_name`'s own lookup for a
/// `raise ... from <name>` whose `name` was bound by an ENCLOSING
/// handler rather than by ordinary code. Recurses into every nested
/// `Stmt::Try`'s own `body` (the same one-level-deep nesting
/// `find_raise_from`'s search already walks) so a chain of nested
/// try/except resolves at any depth. `None` when no enclosing handler
/// binds that name.
fn enclosing_handler_named<'a>(body: &'a [Stmt], name: &str) -> Option<(&'a StmtTry, &'a ExceptHandlerExceptHandler)> {
    for stmt in body {
        let Stmt::Try(try_stmt) = stmt else {
            continue;
        };
        for handler in &try_stmt.handlers {
            let ExceptHandler::ExceptHandler(handler) = handler;
            if handler.name.as_ref().is_some_and(|handler_name| handler_name.id.as_str() == name) {
                return Some((try_stmt, handler));
            }
        }
        if let Some(found) = enclosing_handler_named(&try_stmt.body, name) {
            return Some(found);
        }
    }
    None
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
    judge_and_bind_naming(name, value, declared, fire_range, None, context, environment, out)
}

/// `judge_and_bind`'s own body, plus the ONE naming step
/// `python-c-extension-boundary.md`'s naming unit adds: when the verdict
/// is the GENERIC undetermined sentence (`SENTENCE.value_not_readable` —
/// `assignability.rs::judge`'s own catch-all, which carries no construct
/// name at all) AND `source_expr` is the exact RHS expression this value
/// came from, a call on an attribute chain rooted at an imported-but-
/// unmodeled module renames the sentence to name that module
/// (`expressions::unmodeled_module_call_name`'s own recognition — the
/// SAME gate `evaluate_attribute_call`'s own module arms already apply,
/// so this never fires for a module the walk actually modeled). Every
/// OTHER undetermined sentence (a kernel refusal, a TypedDict/tuple
/// position, a loop-stabilization blocker, …) already names its own
/// construct and passes through unchanged — this step only ever
/// SHARPENS the one anonymous case, never rewrites a sentence that
/// already has a name.
///
/// Split from `judge_and_bind` itself (rather than adding the parameter
/// there directly) so the many call sites with no single RHS expression
/// to offer (a destructured element, a same-module call's own mutation
/// effect) keep calling the original signature unchanged, and only the
/// sites that DO have the RHS in scope opt into naming.
fn judge_and_bind_naming(
    name: &str,
    value: AbstractValue,
    declared: &DeclaredRefinement,
    fire_range: TextRange,
    source_expr: Option<&Expr>,
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
            // Tags the numeric sort onward flow needs (the same guarded
            // rule `seed_parameters` applies to a declared set:
            // numeric-ground only, never the `Literal["A", "B"]`
            // string-tuple pun `on_one_tuple_layer` alone would also
            // admit).
            let refused_slot = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
                let sort = if requires_integer(&declared.set) {
                    PrimitiveKind::Integer
                } else {
                    PrimitiveKind::Float
                };
                AbstractValue {
                    kind_tag: Some(sort),
                    ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
                }
            } else {
                known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
            };
            environment.bind(name, refused_slot);
            None
        }
        Verdict::Silent => {
            environment.bind(name, value);
            None
        }
        Verdict::Undetermined(sentence) => {
            environment.forget(name);
            Some(name_unmodeled_call_sentence(sentence, source_expr, environment))
        }
    }
}

/// The one naming step `judge_and_bind_naming` applies: `sentence`
/// unchanged UNLESS it is the exact generic `value_not_readable` wording
/// AND `source_expr` names a call this file recognizes SOMETHING about —
/// in which case a narrower sentence replaces it. Two rungs, tried in
/// order (`python-c-extension-boundary.md`'s recognition ladder):
///
/// 1. RUNG 2 — a call on a manifested module's own LISTED function
///    (`binding_manifest::discover_manifest` finds a manifest AND it
///    names an entry for the called function): the manifest states the
///    entry, never the return, so the sentence names the missing
///    producer (`diagnostic_sentences::manifest_entry_names_no_producer`).
///    A manifest that exists but names NO entry for this function is a
///    narrower named decline too
///    (`diagnostic_sentences::manifest_names_no_entry_for`) — still more
///    specific than rung 1's plain "no model at all," since the module
///    itself IS modeled in part.
/// 2. RUNG 1 — every other unmodeled-module call
///    (`expressions::unmodeled_module_call_name`'s own recognition).
///
/// Factored out so the two direct `judge(...)` call sites that also have
/// a source expression in scope (`walk_return`'s own return-position
/// judge, `judge_at`) can apply the identical rule without duplicating
/// it.
fn name_unmodeled_call_sentence(sentence: String, source_expr: Option<&Expr>, environment: &Environment) -> String {
    if sentence != crate::diagnostic_sentences::SENTENCE.value_not_readable {
        return sentence;
    }
    let Some(source_expr) = source_expr else {
        return sentence;
    };
    let Expr::Call(call) = source_expr else {
        return sentence;
    };
    if let Some(named) = manifest_named_sentence(call, environment) {
        return named;
    }
    match crate::expressions::unmodeled_module_call_name(call, environment) {
        Some(module_name) => crate::diagnostic_sentences::unmodeled_module_call(module_name),
        None => sentence,
    }
}

/// Rung 2's own naming: `call` on a bare-Name-rooted attribute chain
/// whose root reads unbound AND has a discovered, readable manifest —
/// `Some` names either the missing entry (the function is not one of
/// the manifest's own listed rows) or the missing producer (the function
/// IS listed, so the entry judged the call's arguments already, and the
/// only remaining gap is the return). `None` for every other shape: no
/// manifest discovered at all (rung 1 owns it), or a manifest that could
/// not be read (this naming step never reports a manifest's own parse
/// failure — that is `discover_manifest`'s own `Err` the export/CLI path
/// surfaces, not a per-call sentence).
fn manifest_named_sentence(call: &ruff_python_ast::ExprCall, environment: &Environment) -> Option<String> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if environment.read(module_name.id.as_str()).is_some() {
        return None;
    }
    let entry_directory = environment.entry_directory().map(|path| path.as_path());
    let manifest = crate::binding_manifest::discover_manifest(module_name.id.as_str(), entry_directory)?.ok()?;
    let function_name = attribute.attr.as_str();
    match manifest.entries.get(function_name) {
        Some(entry) => Some(crate::diagnostic_sentences::manifest_entry_names_no_producer(
            module_name.id.as_str(),
            function_name,
            &entry.producer_symbol,
        )),
        None => Some(crate::diagnostic_sentences::manifest_names_no_entry_for(module_name.id.as_str(), function_name)),
    }
}

/// `x op= v` — dispatches on the target's own syntactic shape. A bare
/// name folds `binary_arithmetic_value` (expressions.rs's shared
/// arithmetic transfer — the same one ordinary `x = x op v` rows use, so
/// the two forms agree exactly) over the target's CURRENT value and the
/// evaluated RHS, then judges against `x`'s own recorded refinement
/// (this body's `x: Age = …` AnnAssign, if any) through the shared
/// refused-write law — `Fire` anchors to the WHOLE statement's range
/// (there is no separate "value expression" the way AnnAssign has one;
/// the fired value is the folded result, not a sub-expression of the
/// source). A name with no recorded refinement binds the folded value
/// directly. An `obj.attr op= v` / `name[key] op= v` target composes the
/// identical read-fold-write shape through `walk_field_aug_assign` /
/// `walk_subscript_aug_assign` — see each function's own doc for what it
/// judges and what it can only compose honestly. Any other target shape
/// (a tuple/list/starred aug-target — not valid Python syntax, so this
/// arm is unreachable in practice) stays this body's blocker.
fn walk_aug_assign(
    assign: &StmtAugAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    match assign.target.as_ref() {
        Expr::Name(name) => {
            walk_name_aug_assign(assign, name.id.as_str(), context, environment, aug_assign_refinements, out);
        }
        Expr::Attribute(attribute) => {
            walk_field_aug_assign(assign, attribute, context, environment, out);
        }
        Expr::Subscript(subscript) => {
            walk_subscript_aug_assign(assign, subscript, context, environment, aug_assign_refinements, out);
        }
        _ => {
            record_blocker(
                blocked,
                assign.range(),
                "an augmented assignment to a non-name target is not yet walked".to_owned(),
                out,
            );
        }
    }
}

/// `x op= v` on a plain name — the original bare-name aug-target law,
/// unchanged: fold the target's current value with the evaluated RHS
/// through `binary_arithmetic_value`, then judge against `x`'s own
/// recorded declared refinement (`aug_assign_refinements`) through the
/// shared refused-write law.
fn walk_name_aug_assign(
    assign: &StmtAugAssign,
    name: &str,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    if let Some((range, message)) = provable_raise(assign.value.as_ref(), environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
        // the raise happens before `x op= v` ever folds a value — the
        // target's own current value is untouched by CPython, but this
        // walk has no exception-continuation channel (the same posture
        // `Stmt::Assert`'s doc already states), so the honest answer is
        // to forget rather than assert the pre-raise value still holds
        // past this statement.
        environment.forget(name);
        return;
    }
    // A SOMETIMES-raise (the divisor's set admits 0 among other values)
    // fires its finding and the walk continues with the split value —
    // some runs raise, the rest produce the value, so neither replaces
    // the other (`expressions.rs::possible_raise`'s own claim).
    if let Some((range, message)) = possible_raise(assign.value.as_ref(), environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
    }
    bind_walrus_targets(assign.value.as_ref(), context, aug_assign_refinements, environment, out);
    let current = environment.read(name).cloned().unwrap_or_else(unknown);
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value(assign.op, &current, &operand);

    match aug_assign_refinements.get(name) {
        // An Undetermined verdict already forgets the name inside
        // judge_and_bind; a bare-name aug-target is not itself a
        // blocker candidate (blockers here are scoped to non-name
        // targets only, handled by the caller), so the sentence is
        // discarded.
        Some(declared) => {
            let declared = declared.clone();
            judge_and_bind(name, updated, &declared, assign.range(), context, environment, out);
        }
        None => environment.bind(name, updated),
    }
}

/// `obj.attr op= v` where `obj` is a bare-Name receiver bound to a
/// tagged instance (i-more-expressions.py's `accessor_compound_read_
/// modify_write`: `box.age += 5` through a `@property` getter/setter
/// pair — the same accessor `write_named_field` already judges for a
/// plain `box.age = v`). Composes three EXISTING reads/writes rather
/// than inventing new field-mutation machinery: the CURRENT value reads
/// through the ordinary `evaluate_expression` attribute path (which
/// already resolves a `@property` name to its backing field via
/// `field_read_through_model`), the fold is the identical
/// `binary_arithmetic_value` transfer every other aug-target uses, and
/// the write-back is `write_named_field` — the same judged-and-rebound
/// law a plain `obj.attr = v` write already gets, so `box.age += 5`
/// fires under EXACTLY the same setter-declared refinement a hand-split
/// `box.age = box.age + 5` would.
///
/// A receiver that is not a bare Name, or a bare Name not bound to a
/// tagged instance whose class this environment can find, composes
/// nothing: this function is a no-op in that case (unlike a bare-name
/// aug-target, an attribute aug-target names no single environment slot
/// to forget on decline — the same "no element-level model" posture
/// `bind_or_forget_target`'s own Attribute arm already takes for a
/// plain `obj.attr = v` write to an untagged receiver).
fn walk_field_aug_assign(
    assign: &StmtAugAssign,
    attribute: &ExprAttribute,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return;
    };
    // `write_named_field` is already generic over the receiver's own
    // environment slot — a method body's `self.age += 5` and a local
    // variable's `box.age += 5` share one judged-and-rebound law under
    // whichever name the receiver actually is, with no separate `self`
    // case needed here.
    let receiver_name = receiver.id.as_str();
    let field = attribute.attr.as_str();
    let current = evaluate_expression(&Expr::Attribute(attribute.clone()), environment, context.kernel);
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value(assign.op, &current, &operand);
    write_named_field(receiver_name, field, &updated, assign.range(), context, environment, out);
}

/// `name[key] op= v` where `name` is a bare-Name receiver bound to a
/// known `Kind::Object`/`Kind::List` (i-more-expressions.py's
/// `compound_array_index_operators`/`list_index_power_compound`:
/// `ages[0] += 190`, `over_ages[0] **= 2`). Composes the identical
/// three-step shape `walk_field_aug_assign` uses: the CURRENT element
/// reads through `collection_models::subscript_read`, the fold is the
/// shared `binary_arithmetic_value` transfer, and the write-back replays
/// through the SAME `dict_with_item`/`list_with_item` pair
/// `bind_or_forget_subscript_target` already uses for a plain
/// `name[key] = v` write — rebinding `name` so a later read in the same
/// straight-line body sees the mutated element.
///
/// ELEMENT-LEVEL JUDGING: when `name`'s own declared refinement
/// (`aug_assign_refinements`, this body's `x: list[Age] = …` table) is
/// element-carrying (`typereading::DeclaredRefinement.element`, read for
/// `list[X]`/`set[X]`/`dict[str, X]`), the folded element value is
/// judged against that inner refinement through the shared `judge`
/// entry point — the same law `judge_and_bind` already applies to a
/// bare-name aug-target, one level down at the element. A Fire pushes
/// the finding and writes the DECLARED element set back into the slot
/// (the refused-write law: the container keeps a fact it can still
/// answer for later reads, never the refused value) rather than
/// `updated`; Silent writes `updated` through unchanged; Undetermined
/// forgets the receiver — this walk has no blocker-sentence channel
/// for a subscript aug-target (`walk_aug_assign`'s own doc already
/// scopes blocker candidates to non-name targets, and an element-level
/// Undetermined is not that either), so the honest answer is silence
/// plus forgetting, the same posture an unresolved element read already
/// takes below. A receiver with no recorded declared refinement, or one
/// that is not element-carrying, composes the write mechanically with
/// no judging — unchanged from before.
///
/// A bytes-like receiver is the one exception: `bytes_models::
/// bytes_write_answer`'s raise is a LANGUAGE fact (CPython itself raises
/// on the write), so it is checked here the same way
/// `bind_or_forget_subscript_target` checks it for a plain `=` write —
/// see that function's own doc for the full three-way rule (a provable
/// raise leaves the receiver untouched and records no finding, success
/// applies through `list_with_item` as today, undecidable forgets).
fn walk_subscript_aug_assign(
    assign: &StmtAugAssign,
    subscript: &ExprSubscript,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        return;
    };
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    let key_value = evaluate_expression(subscript.slice.as_ref(), environment, context.kernel);
    let Some(current) = subscript_read(&receiver_value, &key_value) else {
        // an unresolved element read (an unknown container, a key this
        // walk cannot read exactly, an out-of-bounds index) states
        // nothing to fold — forgetting the receiver is the same honesty
        // `bind_or_forget_subscript_target` already keeps for a decline.
        environment.forget(receiver_name.id.as_str());
        return;
    };
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value(assign.op, &current, &operand);
    if receiver_value.kind == Kind::List && receiver_value.kind_word.is_some() {
        match bytes_models::bytes_write_answer(&receiver_value, &updated) {
            Some(BytesAnswer::Raises(_)) => return,
            Some(BytesAnswer::Value(_)) => {}
            None => {
                environment.forget(receiver_name.id.as_str());
                return;
            }
        }
    }
    let element_declared = aug_assign_refinements
        .get(receiver_name.id.as_str())
        .and_then(|declared| declared.element.as_deref());
    let updated = if let Some(element_declared) = element_declared {
        match judge(&updated, element_declared, context.kernel) {
            Verdict::Fire(message) => {
                out.push(Finding {
                    range: assign.range(),
                    code: "RTS7001",
                    message,
                });
                // Tags the numeric sort onward flow needs (the same
                // guarded rule `seed_parameters` applies to a declared
                // set: numeric-ground only, never the
                // `Literal["A", "B"]` string-tuple pun
                // `on_one_tuple_layer` alone would also admit).
                if on_one_tuple_layer(&element_declared.set) && !states_sequence(&element_declared.set) {
                    let sort = if requires_integer(&element_declared.set) {
                        PrimitiveKind::Integer
                    } else {
                        PrimitiveKind::Float
                    };
                    AbstractValue {
                        kind_tag: Some(sort),
                        ..known_set(element_declared.set.clone(), None, TrustSpec, SetKindTag::None)
                    }
                } else {
                    known_set(element_declared.set.clone(), None, TrustSpec, SetKindTag::None)
                }
            }
            Verdict::Silent => updated,
            Verdict::Undetermined(_) => {
                environment.forget(receiver_name.id.as_str());
                return;
            }
        }
    } else {
        updated
    };
    let written = match receiver_value.kind {
        Kind::Object => dict_with_item(&receiver_value, &key_value, &updated),
        Kind::List => list_with_item(&receiver_value, &key_value, &updated),
        _ => None,
    };
    match written {
        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
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
///
/// A `Callable[[...], R]`-annotated target (`declared_refinement`
/// states nothing for it) is recorded separately, into `environment`'s
/// own `callable_returns` table, through `typereading::
/// callable_return_refinement` — see the CALLABLE-VARIABLE CALL
/// CHANNEL comment inside this function's decline arm.
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
            .or_else(|| direct_alias_annotation(assign.annotation.as_ref(), context.aliases, environment))
            .or_else(|| optional_base_sort_annotation(assign.annotation.as_ref()));

    let Some(declared) = declared else {
        // CALLABLE-VARIABLE CALL CHANNEL: `x: Callable[[...], R] [|
        // None] = ...` states nothing `declared_refinement` reads (a
        // `Callable[...]` subscript is not a set X itself binds to),
        // but a LATER `x(...)` call site still has a fact to judge
        // against — `R`, the callable's own return refinement. Recorded
        // into this environment's `callable_returns` table (keyed on
        // the target's plain name), read back by `check.rs::sink_value`
        // at the call site the same way `aug_assign_refinements` is
        // read back for a later `x op= v`. Tried before the rebound-alias
        // blocker check below: a `Callable`-typed name is ordinary
        // Python with a real fact to state, never this body's blocker.
        if let Expr::Name(target_name) = assign.target.as_ref()
            && let Some(callable_declared) =
                callable_return_refinement(assign.annotation.as_ref(), context.aliases, context.imports, environment)
        {
            let mut callable_returns = environment
                .callable_returns()
                .map(|table| (**table).clone())
                .unwrap_or_default();
            callable_returns.insert(target_name.id.as_str().to_owned(), callable_declared);
            environment.set_callable_returns(Arc::new(callable_returns));
            provably_unbound.remove(target_name.id.as_str());
            bind_target_from_value_expr(assign.target.as_ref(), assign.value.as_deref(), environment, context.kernel);
            return;
        }
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
        } else if let Some(spelling) = unhonorable_annotated_spelling(assign.annotation.as_ref(), context.imports) {
            // UNHONORABLE STATEMENT (RTS7004): the annotation is
            // recognizably this table's OWN vocabulary — an
            // `Annotated[...]` subscript rooted at the module's
            // imported `Annotated` identity (`surface::
            // annotated_expression_set`'s own head-identity gate) —
            // but `declared_refinement` still read nothing from it,
            // meaning something PAST that gate refused (an
            // unrecognized base sort, an unrecognized metadata
            // kwarg/constructor, or a pattern the grammar compiler
            // rejected). Mirrors the Go twin's own gate
            // (`RootsInSurface` in `annotation_file_facts.go`): a
            // recognized-root annotation the checker cannot read is
            // never dropped silently, unlike a plain-Python name this
            // table has no vocabulary for at all (that case falls
            // through below, unchanged).
            out.push(Finding {
                range: assign.annotation.range(),
                code: "RTS7004",
                message: unhonorable_annotation(&spelling),
            });
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

    // THE EMPTY SET (RTS7003): this annotation compiled to a scalar or
    // sequence set the kernel proves admits nothing — the declaration
    // itself can never be honored by any value, independent of what
    // the write assigns. Mirrors the Go twin's own per-occurrence fire
    // (`annotation_file_facts.go`'s `emptiness` ask, immediately after
    // a successful compile): asked once per annotated statement, the
    // same "courtesy, never a blocker" posture the Go doc names — a
    // kernel refusal on this set's shape (caught, never a crash, the
    // same `catch_unwind` idiom `truthiness_conformance.rs`'s own
    // emptiness ask already holds every kernel closure to) answers
    // "not decided" and this diagnostic simply does not fire, the
    // declaration still compiles and judges normally either way.
    if let Some(true) = declared_set_is_empty(&declared.set, context.kernel) {
        out.push(Finding {
            range: assign.annotation.range(),
            code: "RTS7003",
            message: empty_set(&declared.set),
        });
    }

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
    let Some(value) = sink_value(value_expr, context, environment, aug_assign_refinements, out) else {
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
                let sentence = name_unmodeled_call_sentence(sentence, Some(value_expr), environment);
                record_blocker(blocked, value_expr.range(), sentence, out);
            }
        }
        bind_target_from_value_expr(assign.target.as_ref(), Some(value_expr), environment, context.kernel);
        return;
    };

    if let Some(sentence) = judge_and_bind_naming(
        target_name.id.as_str(),
        value,
        &declared,
        value_expr.range(),
        Some(value_expr),
        context,
        environment,
        out,
    ) {
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

/// `Optional[int|float|str]` / `int|float|str | None` — the Optional-
/// peeling idiom over a BARE base sort, with no alias involved.
/// `declared_refinement`'s general table deliberately does not read a
/// bare `int`/`float`/`str` (its own doc: doing so turned every
/// unreadable `-> int` helper into a fresh undetermined blocker), so
/// `over: Optional[int] = 200` reaches this function's caller with
/// `declared` still `None` and NOTHING recorded into
/// `aug_assign_refinements` — leaving `walk_if`'s `is_admits_none_peel_
/// test` unable to find the declared shape and firing the dead-branch
/// law on the ordinary `if over is None:` peel. This reader is scoped
/// to exactly the wrapper shape (`Optional[X]`/`X | None`) around
/// exactly a bare base-sort name, and answers through
/// `base_sort_return_refinement` — the SAME set that sort already
/// states everywhere else it is read (a declined call's return, a
/// `Callable[[...], R]` slot) — so recording it here states nothing
/// new, only lets the ALREADY-STATED fact reach the peel-test
/// exception. A bare `int`/`float`/`str` with no `Optional`/`| None`
/// wrapper still declines (unaffected): this function is reached only
/// through `walk_ann_assign`'s `Optional[X]`/`X | None` peel below.
fn optional_base_sort_annotation(annotation: &Expr) -> Option<DeclaredRefinement> {
    match annotation {
        Expr::Subscript(subscript) => {
            let is_optional = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Optional");
            if !is_optional {
                return None;
            }
            let mut declared = base_sort_return_refinement(subscript.slice.as_ref())?;
            declared.admits_none = true;
            Some(declared)
        }
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            let mut declared = base_sort_return_refinement(other)?;
            declared.admits_none = true;
            Some(declared)
        }
        _ => None,
    }
}

/// The pre-typereading path: an annotation that is bare `Name` naming
/// a compiled alias, visible in this body (not locally rebound). Kept
/// alongside `declared_refinement` so the two existing tests' fires
/// keep firing before the general annotation table recognizes this
/// same shape itself.
fn direct_alias_annotation(
    annotation: &Expr,
    aliases: &HashMap<String, AliasEntry>,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    let Expr::Name(name) = annotation else {
        return None;
    };
    if !environment.alias_is_visible(name.id.as_str()) {
        return None;
    }
    let entry = aliases.get(name.id.as_str())?;
    // Same container carry as `declared_refinement`'s own bare-Name arm
    // (`typereading.rs` doc) — a `Boosted`-shaped alias must not lose its
    // element/length window just because this AnnAssign path reads it
    // before the general table gets a chance to. The element's spelling
    // is its own WRITTEN spelling (`entry.element`'s second tuple slot),
    // never a reformatting of its resolved set — the same fidelity
    // `declared_refinement`'s own bare-Name arm keeps.
    let element = entry.element.as_ref().map(|element_entry| {
        let (element_set, element_spelling) = element_entry.as_ref();
        Box::new(DeclaredRefinement {
            set: element_set.clone(),
            spelling: element_spelling.clone(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        })
    });
    let container_spelling = match (entry.head, &element) {
        (Some(head), Some(element_declared)) => Some(format!("{}[{}]", head, element_declared.spelling)),
        _ => None,
    };
    Some(DeclaredRefinement {
        set: entry.set.clone(),
        spelling: container_spelling.unwrap_or_else(|| name.id.as_str().to_owned()),
        admits_none: false,
        element,
        element_length: entry.length_window,
        generator: None,
        members: None,
        positions: None,
    })
}

/// Recognizes an annotation as this table's OWN `Annotated[...]`
/// vocabulary — the module's imported `Annotated` (or
/// `typing_extensions.Annotated`) identity as the subscript's head —
/// the exact gate `surface::annotated_expression_set` itself opens
/// with before reading any base sort or metadata. Reads `imports`'
/// `annotated_names` set directly rather than re-deriving it, so this
/// stays exactly as narrow as the compiler's own recognition and never
/// drifts from it. Returns the recognized spelling (`"Annotated[...]"`)
/// for the RTS7004 message when the annotation matches this shape;
/// `None` for every other shape (a bare alias name, `dict[...]`,
/// `list[...]`, or any expression this table has no vocabulary for at
/// all) — those are ordinary Python, never this diagnostic's business.
fn unhonorable_annotated_spelling(annotation: &Expr, imports: &crate::surface::SurfaceImports) -> Option<String> {
    let Expr::Subscript(subscript) = annotation else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !imports.annotated_names.contains(head.id.as_str()) {
        return None;
    }
    Some(format!("{}[...]", head.id.as_str()))
}

/// Whether a compiled declared set is proved empty — the courtesy ask
/// RTS7003 fires on. Tries the scalar decider first, then the sequence
/// decider (the same two-decider order `truthiness_conformance.rs`'s
/// own `state_is_uninhabited` takes), each guarded by `catch_unwind` so
/// a kernel refusal on a set shape neither decider speaks to reads as
/// `None` (not decided) rather than a crash. `None` means this
/// diagnostic simply does not fire — the annotation still compiled and
/// still judges normally.
fn declared_set_is_empty(set: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let scalar_asked = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(set));
    if let Ok(empty) = scalar_asked {
        return Some(empty);
    }
    let seq_asked = crate::kernel_ask::ask_kernel(|| (kernel.seq_empty)(set));
    seq_asked.ok()
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
/// of).
///
/// FIELD-WRITE LAW: `<receiver>.<field> = v` where `receiver` is a bare
/// Name bound to a TAGGED instance (`Kind::Object`, a non-empty `source`
/// naming a `ClassModel` this environment can find) is JUDGED, through
/// `write_named_field` — `self` inside a method body walked through
/// `walk_method_def`'s self-seeding (`self_attribute_name`'s own
/// recognition), and any OTHER local name holding a tagged instance
/// (`box.age = 200`, `over_box.age = 200` — e-class-and-function.py's
/// `property_getter_setter`, q-decline-names.py's `setter_effect_read_
/// through_getter`) alike: the receiver's class resolves through
/// `environment.classes()`, and `instances::field_write_judgment` judges
/// `v` against the field's own declared refinement exactly like any
/// other write sink in this file (`Fire` pushes an RTS7001 at the
/// value's own range; `Undetermined` records this body's blocker).
/// Either way the receiver REBINDS to `instances::field_write`'s updated
/// instance (never forgotten) — a later `<receiver>.<field>` read in the
/// SAME straight-line body must see the write, matching every other
/// known-write sink's own read-after-write law in this file.
/// `field_write_judgment` returning `None` (an unrefined field, or a
/// field the model does not declare) still rebinds through `field_write`
/// with no Fire — an ordinary Python attribute gain is not a blocker.
///
/// Falls back to forgetting the RECEIVER's own base name — the leftmost
/// `Name` under the attribute chain (`receiver_base_name`) — when the
/// receiver is not a bare Name bound to a tagged instance at all (an
/// arbitrary attribute chain, an untagged value, a class this
/// environment cannot find): a known instance bound to that name may
/// carry a stale field value for `x` after this write, and this file
/// does not track field-level state through an unresolved attribute
/// write, so forgetting the whole receiver is the one sound answer
/// there.
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
            if let Some(field) = instances::self_attribute_name(target) {
                if write_named_field("self", &field, value, value_range, context, environment, out) {
                    return;
                }
            } else if let Expr::Name(receiver) = attribute.value.as_ref() {
                // NAMED-RECEIVER FIELD WRITE: `box.age = v` where `box`
                // (any bare name, not just `self`) is bound to a tagged
                // instance — e-class-and-function.py's
                // `property_getter_setter` (`over_box.age = 200` through
                // a `@property` setter) and q-decline-names.py's
                // `setter_effect_read_through_getter` both write through a
                // LOCAL variable holding the instance, never `self`. The
                // same judged-and-rebound law `write_named_field` already
                // gives `self` applies unchanged: the receiver name is
                // just a different environment slot to re-read/rebind.
                if write_named_field(
                    receiver.id.as_str(),
                    attribute.attr.as_str(),
                    value,
                    value_range,
                    context,
                    environment,
                    out,
                ) {
                    return;
                }
            }
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

/// The FIELD-WRITE LAW (see `bind_or_forget_target`'s own doc): `<receiver>.<field>
/// = value`, judged and rebound under `receiver_name` — the environment
/// slot a bare-Name receiver is bound under, `self` inside a method body
/// (`self_attribute_name`'s own recognition) or any other local name
/// holding a tagged instance (`box.age = 200`, `over_box.age = 200`).
/// Returns `true` when `receiver_name` reads as a tagged instance whose
/// class this environment can find — the write is fully handled either
/// way (judged and rebound), and the caller must not ALSO run its own
/// forget-the-receiver fallback. Returns `false` when the receiver is
/// unbound, untagged, or its class is not in `environment.classes()` —
/// the caller's existing fallback is the honest answer there.
fn write_named_field(
    receiver_name: &str,
    field: &str,
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> bool {
    let Some(instance) = environment.read(receiver_name) else {
        return false;
    };
    if instance.kind != Kind::Object || instance.source.is_empty() {
        return false;
    }
    let Some(classes) = environment.classes() else {
        return false;
    };
    let Some(model) = classes.get(instance.source.as_str()) else {
        return false;
    };
    if let Some(Verdict::Fire(message)) = instances::field_write_judgment(model, field, value, context.kernel) {
        out.push(Finding {
            range: value_range,
            code: "RTS7001",
            message,
        });
    }
    // A `@property` name is never a stored slot — `field_read_through_model`
    // resolves a read of `field` to the property's `backing` name
    // (`instances.rs`'s own doc), so the write must land on that SAME
    // backing name or a later read through the property sees the instance's
    // pre-write value instead of what was just assigned
    // (e-class-and-function.py's `property_getter_setter`,
    // q-decline-names.py's `setter_effect_read_through_getter`: writing
    // `over_box.age = 200` and reading `over_box.age` back must see 200).
    let write_target = match model.properties.get(field) {
        Some(property) => property.backing.as_str(),
        None => field,
    };
    // re-read after the class-table lookup above (which only borrowed
    // `environment`) so the write below can borrow it mutably; the
    // receiver is still exactly the instance just read, since nothing in
    // between could have rebound it.
    let instance = environment.read(receiver_name).expect("checked Some above").clone();
    if let Some(updated) = instances::field_write(&instance, write_target, value.clone()) {
        environment.bind(receiver_name, updated);
    }
    true
}

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
///
/// A `bytes`/`bytearray`/`memoryview` receiver (`bytes_models::tagged`'s
/// own species word) routes through `bytes_models::bytes_write_answer`
/// FIRST, before the plain `list_with_item` path below ever runs. A
/// write CPython provably raises (an out-of-`0..=255` bytearray/
/// memoryview element, or ANY write onto an immutable `bytes`) leaves
/// the receiver COMPLETELY UNTOUCHED — no bind, no forget, no finding.
/// The write never took effect, so the pre-write contents are still
/// exactly right (rebinding OR forgetting would both be a weaker, wrong
/// answer — a forgotten receiver reads Undetermined past this point even
/// though its value is still fully known), and the raise itself is not
/// this function's own finding to report: it is not a judgment against a
/// declared refinement (`assignability::judge`'s own seam), just a
/// LANGUAGE fact this function uses to keep its model honest
/// (p-typed-array.py's own `bytes_is_immutable` docstring: "No
/// expect-error marker belongs on the raise itself"). A write that
/// provably SUCCEEDS applies through the ordinary `list_with_item` path
/// below unchanged (200 into a bytearray is in `0..=255`, so it writes
/// through even though 200 would refuse against a declared `Age` — a
/// different, later question `judge_and_bind` owns, never this
/// function). An UNDECIDABLE bytes-like write (an unknown value) falls
/// through to the same decline-and-forget the untagged path already
/// takes, honest about a write this function cannot prove either way.
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
    if receiver_value.kind == Kind::List && receiver_value.kind_word.is_some() {
        match bytes_models::bytes_write_answer(&receiver_value, value) {
            Some(BytesAnswer::Raises(_)) => return,
            Some(BytesAnswer::Value(_)) => {
                // falls through to the ordinary list_with_item path below,
                // which performs the identical index-bounds write — this
                // function only needed to know the write does not raise.
            }
            None => {
                environment.forget(receiver_name.id.as_str());
                return;
            }
        }
    }
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

/// `del d[k]` — the delete-shaped sibling of `bind_or_forget_subscript_target`:
/// only a bare-`Name` receiver is replayed (any other receiver shape has
/// no single environment slot to rebind, and is simply left untouched —
/// the same "no element-level model" posture the write-sibling takes).
/// `collection_models::dict_without_item` answers the receiver WITHOUT
/// `key`'s entry: `Some` rebinds `name` to it, so a later read sees the
/// key's absence (b-body-expressions.py's `del_expression`: `del
/// person["age"]` then `person.get("age")` must answer the absent-key
/// default, not the stale pre-delete 40); `None` (an unknown receiver, a
/// key this walk cannot read exactly, or a receiver `Kind` the contract
/// does not own — e.g. a `List`, which has no by-value delete this table
/// models) FORGETS `name` — the pre-delete value must not survive an
/// unresolved delete, the same honesty every other decline in this file
/// already keeps.
fn walk_del_subscript_target(subscript: &ExprSubscript, context: &WalkContext, environment: &mut Environment) {
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        return;
    };
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    let key_value = evaluate_expression(subscript.slice.as_ref(), environment, context.kernel);
    let written = dict_without_item(&receiver_value, &key_value);
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
fn sink_value(
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

/// Recognizes `expr` as a call on a manifested module's own listed
/// function, judges every WRITTEN argument (positional matched by
/// position, keyword matched by name) against the manifest's parsed
/// entry contract, and pushes an RTS7001 for each one that escapes —
/// `binding_manifest::judge_manifest_call`'s own crossing-fit check, the
/// SAME refusal shape the stdio edge fires for an escaping outbound
/// value. A no-op for every call this recognizes nothing about: an
/// unmodeled module with no manifest, a manifest with no row for the
/// called function (rung 1's own plain decline territory either way),
/// or a manifest file this reader could not parse — a bad manifest never
/// crashes the walk, it simply contributes no crossing-fit judging this
/// call.
fn manifest_call_fires(expr: &Expr, context: &WalkContext, environment: &Environment, out: &mut Vec<Finding>) {
    let Expr::Call(call) = expr else {
        return;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return;
    };
    if environment.read(module_name.id.as_str()).is_some() {
        return;
    }
    let entry_directory = environment.entry_directory().map(|path| path.as_path());
    let Some(Ok(manifest)) = crate::binding_manifest::discover_manifest(module_name.id.as_str(), entry_directory) else {
        return;
    };
    let Some(entry) = manifest.entries.get(attribute.attr.as_str()) else {
        return;
    };
    let positional: Vec<(AbstractValue, TextRange)> = call
        .arguments
        .args
        .iter()
        .map(|arg| (evaluate_expression(arg, environment, context.kernel), arg.range()))
        .collect();
    let keyword: Vec<(String, AbstractValue, TextRange)> = call
        .arguments
        .keywords
        .iter()
        .filter_map(|kw| {
            let arg_name = kw.arg.as_ref()?;
            Some((
                arg_name.as_str().to_owned(),
                evaluate_expression(&kw.value, environment, context.kernel),
                kw.value.range(),
            ))
        })
        .collect();
    let outcome =
        crate::binding_manifest::judge_manifest_call(module_name.id.as_str(), entry, &positional, &keyword, context.kernel);
    for (range, message) in outcome.fires {
        out.push(Finding { range, code: "RTS7001", message });
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
fn callable_variable_call_result(
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
fn instance_method_call_result(
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
fn keyword_arguments_by_position(
    call: &ruff_python_ast::ExprCall,
    method: &StmtFunctionDef,
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
/// to its own existing dispatch order unchanged.
fn apply_call_effects(
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
    let arguments: Vec<AbstractValue> =
        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, context.kernel)).collect();
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
        // Unbound, OR bound to its OWN class-object value (the walk seeds
        // every visible class name to `instances::class_object_value`,
        // whose `source` is the class's own name) — calling the class
        // object IS the construction. Any other binding shadows the
        // class name, same rule evaluate_call applies to a builtin name.
        let callee_open = match environment.read(callee.id.as_str()) {
            None => true,
            Some(bound) => {
                bound.kind == refined_domain::abstract_value::Kind::Object
                    && bound.source == callee.id.as_str()
            }
        };
        if callee_open {
            // A class defined LOCALLY inside the walked body only lives in
            // `environment.classes()` (`merged_classes_for_body`'s own merge
            // over `context.classes`) — two different body-local classes
            // sharing a bare name (e.g. two functions each declaring their
            // own `class Person`) collide in the one shared
            // `context.classes` map, so the per-body table must win when
            // present, exactly as `instance_method_call_result` already
            // reads it.
            let classes = environment.classes().unwrap_or(&context.classes);
            if let Some(model) = classes.get(callee.id.as_str()) {
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
        // Same locality rule as the bare-Name construction arm above: a
        // body-local class only lives in `environment.classes()`.
        if let Some(model) = environment.classes().unwrap_or(&context.classes).get(class_name.id.as_str()) {
            let dict_argument = single_dict_argument(&call.arguments)?;
            let keyword = dict_literal_keyword_rows(dict_argument, environment, context.kernel)?;
            return Some(judge_construction(model, &[], &keyword, context.kernel));
        }
        // THE ADAPTER-ALIAS ROUTE: `TypeAdapter(<alias>).validate_python(<scalar
        // expr>)` where `<alias>` is a bare `type X = ...` name
        // (`context.aliases`), not a `ClassModel`. Judges the ARGUMENT
        // expression's own value against the alias's declared set —
        // there is no field-by-field construction here, since the alias
        // names a scalar (or Literal) set, not an object shape.
        return adapter_alias_verdict(class_name, &call.arguments, context, environment);
    }
    None
}

/// `TypeAdapter(<alias name>).validate_python(<argument>)` against a
/// module-level `type <alias> = ...` set — `None` when `<alias>` is not
/// in `context.aliases` (the class route above already tried
/// `context.classes` and missed) or the call does not carry exactly one
/// positional, no-keyword argument (`validate_python`'s own single-value
/// shape).
fn adapter_alias_verdict(
    class_name: &ruff_python_ast::ExprName,
    call_arguments: &ruff_python_ast::Arguments,
    context: &WalkContext,
    environment: &Environment,
) -> Option<ConstructionVerdict> {
    let declared_entry = context.aliases.get(class_name.id.as_str())?;
    if !call_arguments.keywords.is_empty() {
        return None;
    }
    let [argument_expr] = call_arguments.args.as_ref() else {
        return None;
    };
    let declared = DeclaredRefinement {
        set: declared_entry.set.clone(),
        spelling: class_name.id.as_str().to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let range = argument_expr.range();
    let mut value = evaluate_expression(argument_expr, environment, context.kernel);
    // LAX INT COERCION: pydantic's own `int` field (never `StrictInt`,
    // execution-verified 2026-08-17 against pydantic 2.13.4 —
    // `TypeAdapter(Age).validate_python("40")` coerces to `40`,
    // `.validate_python("200")` coerces to `200` and THEN fails the
    // range bound, `.validate_python("abc")`/`""` raise a parse error
    // this table does not model) accepts a plain base-10 digit string
    // (optional leading `-`, ASCII digits only — the narrow shape this
    // row needs; pydantic's fuller grammar also admits whitespace and
    // whole-valued float strings, out of scope here) and coerces it to
    // the int it spells before judging. `StrictInt` never coerces — a
    // `str` argument against a `StrictAge`-shaped alias reaches
    // `assignability::judge`'s own opaque/structural-mismatch law
    // unparsed, firing "not assignable" (StrictInt's own refusal,
    // execution-verified: `.validate_python("40")` raises `int_type`
    // with no coercion attempt).
    //
    // GATED ON A NUMERIC-SORTED ALIAS: this coercion is pydantic's `int`
    // FIELD behavior — it applies only when the alias itself declares an
    // int-sorted set (`requires_integer`, `refined_sets::refinement_
    // forms`'s own recognizer for the `Form::Integer` marker
    // `annotated_expression_set` pushes for `int`). m-pydantic-schema.py's
    // `Digits` (a STR-sorted pattern alias, `type Digits = Annotated[str,
    // Field(pattern=r"^[0-9]+$")]`) must NOT coerce
    // `TypeAdapter(Digits).validate_python("42")` — a digit-only STRING is
    // exactly what a `str`-sorted pattern alias accepts on its own terms,
    // and rewriting it to the int `42` before judging is judging the
    // wrong sort entirely. `plain_digit_string_value` only ever produces
    // an Integer-tagged value, so `requires_integer` is the precise gate:
    // a Float-sorted or str-sorted declared set never coerces.
    if value.kind == Kind::Values
        && value.kind_tag == Some(PrimitiveKind::String)
        && requires_integer(&declared_entry.set)
        && !context.strict_int_aliases.contains(class_name.id.as_str())
    {
        if let Some(parsed) = plain_digit_string_value(&value.values) {
            value = parsed;
        }
    }
    match judge(&value, &declared, context.kernel) {
        Verdict::Fire(message) => Some(ConstructionVerdict {
            fires: vec![(range, message)],
            // THE REFUSED-WRITE LAW (this file's own header note): the
            // answer carries the DECLARED SET, never the refused raw
            // value — this construction's own return type is very often
            // the SAME alias (`-> Age` on a `TypeAdapter(Age)` call), so
            // the outer sink (`walk_return`) judges this instance a
            // SECOND time against that identical declaration; handing
            // back the raw out-of-set value would fire there again for
            // the one refusal this function already reported.
            instance: declared_set_instance(&declared),
        }),
        Verdict::Silent => Some(ConstructionVerdict {
            fires: Vec::new(),
            instance: value,
        }),
        Verdict::Undetermined(_) => Some(ConstructionVerdict {
            fires: Vec::new(),
            // the same "keeps the DECLARED set" answer
            // `judge_construction`'s own Undetermined arm gives a
            // construction field — a later sink judging this value
            // against the SAME declaration (e.g. the function's own `->
            // Age` return annotation) sees a trivial self-match rather
            // than staying stuck on a value this table could not read.
            instance: declared_set_instance(&declared),
        }),
    }
}

/// A declared set as a bound value, tagged with its numeric sort when
/// the ground is provably numeric — the same guarded rule
/// `seed_parameters` applies to a declared set: `on_one_tuple_layer`
/// alone also reads a `Literal["A", "B"]` string-tuple union as "on the
/// one-tuple layer", so `states_sequence` must be false too, ruling out
/// that pun. Shared by `adapter_alias_verdict`'s Fire and Undetermined
/// arms, which both keep the declared set rather than the value this
/// table refused or could not read.
fn declared_set_instance(declared: &DeclaredRefinement) -> AbstractValue {
    if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
        let sort = if requires_integer(&declared.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        return AbstractValue {
            kind_tag: Some(sort),
            ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
        };
    }
    known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
}

/// A plain base-10 digit string's codepoints (optional leading `-`,
/// ASCII digits only — Python's `int()` grammar restricted to the shape
/// this row needs, `expressions.rs::is_valid_base_ten_int_string`'s
/// fuller sibling out of this file's reach) read as the int
/// `AbstractValue` it spells — pydantic's lax `int` coercion parses the
/// SAME digit text before range-judging it (execution-verified: `"200"`
/// coerces to `200`, then fails `le=120`). `None` for anything else
/// (a float string, a non-digit string, an empty string) — this table
/// declines rather than guessing a coercion pydantic itself would
/// refuse.
fn plain_digit_string_value(code_points: &[f64]) -> Option<AbstractValue> {
    let text: String = code_points
        .iter()
        .map(|point| char::from_u32(*point as i64 as u32))
        .collect::<Option<String>>()?;
    let digits = text.strip_prefix('-').unwrap_or(&text);
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let parsed: i64 = text.parse().ok()?;
    Some(known_values(vec![parsed as f64], PrimitiveKind::Integer, TrustProved))
}

/// `<ClassName>` out of a bare-Name expression naming a class in
/// `environment.classes()` (falling back to `context.classes` when the
/// environment carries none — the same locality rule
/// `instance_method_call_result` already applies, since a class defined
/// LOCALLY inside the walked body only lives in the per-body table) — the
/// receiver shape `<ClassName>.model_validate` reads. `None` for anything
/// else (a non-Name receiver, or a Name that is either environment-bound to
/// something else or simply not a known class).
fn class_model_of_bare_name<'a>(
    expr: &Expr,
    context: &'a WalkContext,
    environment: &'a Environment,
) -> Option<&'a ClassModel> {
    let Expr::Name(name) = expr else {
        return None;
    };
    // A name bound to its OWN class-object value (the walk seeds a
    // class's bare name to `instances::class_object_value`, whose
    // `source` is the class's own name) is still the constructor —
    // calling it IS the construction. Any OTHER binding shadows the
    // class name as before.
    if let Some(bound) = environment.read(name.id.as_str()) {
        let is_own_class_object = bound.kind == refined_domain::abstract_value::Kind::Object
            && bound.source == name.id.as_str();
        if !is_own_class_object {
            return None;
        }
    }
    environment.classes().unwrap_or(&context.classes).get(name.id.as_str())
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

/// STALE-RECEIVER SOUNDNESS, unmodeled-body law: `collect_bound_names`
/// (and `collect_bound_names_stmt`) only name the slots a body BINDS —
/// an assignment/for/with-as/except/walrus target, a parameter, an
/// import. A name that is only ever MUTATED inside an unmodeled body
/// (never itself the target of `=`) is invisible to that scan, so the
/// blocker-path forgets above leave its stale pre-loop/pre-match value
/// standing — exactly the shape `grouped.setdefault(...).append(age)`
/// inside a declined `for` takes: `grouped` is never assigned, only
/// mutated through a chained method call, so a post-loop read of
/// `grouped` wrongly kept reading the empty dict from before the loop
/// (c-reads-and-values.py:1008's own WRONG ANSWER: an unmatched
/// "provably raises KeyError" fire on a key the mutation actually
/// wrote).
///
/// This function is the second half of the same forget: a syntactic
/// walk over every statement and expression in `stmt`, collecting the
/// LEFTMOST `Name` reachable under two receiver shapes — an
/// ATTRIBUTE-CALL's receiver (`X.method(...)`, the func of a `Call`
/// being an `Attribute`) and a SUBSCRIPT-STORE's receiver (`X[k] = v`,
/// an assign target that is a `Subscript`) — walking THROUGH a chained
/// call's own func-attribute the way `grouped.setdefault(...).append(...)`
/// requires (the `.append` receiver is itself a Call, whose own func is
/// another Attribute reaching back to `grouped`). Every collected base
/// name is forgotten, on top of (never replacing) `forget_names_bound_by_stmt`'s
/// own bound-name forgets — sound and narrow: this is a syntactic
/// over-approximation (a plain non-mutating method call like
/// `x.keys()` is also swept up), never a false negative, since a stale
/// receiver surviving an unmodeled body is exactly the wrong-answer
/// shape this law exists to close.
fn forget_mutated_receivers_in_stmt(stmt: &Stmt, environment: &mut Environment) {
    let mut receivers = HashSet::new();
    collect_mutation_receiver_names_stmt(stmt, &mut receivers);
    for name in &receivers {
        environment.forget(name);
    }
}

/// The per-case-body sibling of `forget_mutated_receivers_in_stmt`, for
/// a `match` the arm-decision module declined to resolve — one case
/// body at a time, matching `forget_names_bound_in_body`'s own calling
/// convention.
fn forget_mutated_receivers_in_body(body: &[Stmt], environment: &mut Environment) {
    let mut receivers = HashSet::new();
    for stmt in body {
        collect_mutation_receiver_names_stmt(stmt, &mut receivers);
    }
    for name in &receivers {
        environment.forget(name);
    }
}

/// Walks one statement's own sub-bodies and every expression it
/// contains, collecting every attribute-call/subscript-store receiver's
/// leftmost base name into `receivers` — see
/// `forget_mutated_receivers_in_stmt`'s own doc for the exact contract.
fn collect_mutation_receiver_names_stmt(stmt: &Stmt, receivers: &mut HashSet<String>) {
    match stmt {
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                collect_subscript_store_receiver(target, receivers);
            }
            collect_mutation_receiver_names_expr(assign.value.as_ref(), receivers);
        }
        Stmt::AnnAssign(assign) => {
            collect_subscript_store_receiver(assign.target.as_ref(), receivers);
            if let Some(value) = assign.value.as_deref() {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Stmt::AugAssign(assign) => {
            collect_subscript_store_receiver(assign.target.as_ref(), receivers);
            collect_mutation_receiver_names_expr(assign.value.as_ref(), receivers);
        }
        Stmt::Expr(expr_stmt) => collect_mutation_receiver_names_expr(expr_stmt.value.as_ref(), receivers),
        Stmt::Return(ret) => {
            if let Some(value) = ret.value.as_deref() {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                collect_mutation_receiver_names_expr(target, receivers);
            }
        }
        Stmt::Assert(assert) => {
            collect_mutation_receiver_names_expr(assert.test.as_ref(), receivers);
            if let Some(msg) = assert.msg.as_deref() {
                collect_mutation_receiver_names_expr(msg, receivers);
            }
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                collect_mutation_receiver_names_expr(exc, receivers);
            }
            if let Some(cause) = raise.cause.as_deref() {
                collect_mutation_receiver_names_expr(cause, receivers);
            }
        }
        Stmt::If(if_stmt) => {
            collect_mutation_receiver_names_expr(if_stmt.test.as_ref(), receivers);
            for inner in &if_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for clause in &if_stmt.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    collect_mutation_receiver_names_expr(test, receivers);
                }
                for inner in &clause.body {
                    collect_mutation_receiver_names_stmt(inner, receivers);
                }
            }
        }
        Stmt::For(for_stmt) => {
            collect_mutation_receiver_names_expr(for_stmt.iter.as_ref(), receivers);
            for inner in &for_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for inner in &for_stmt.orelse {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::While(while_stmt) => {
            collect_mutation_receiver_names_expr(while_stmt.test.as_ref(), receivers);
            for inner in &while_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for inner in &while_stmt.orelse {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                collect_mutation_receiver_names_expr(&item.context_expr, receivers);
            }
            for inner in &with_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::Try(try_stmt) => {
            for inner in &try_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    collect_mutation_receiver_names_stmt(inner, receivers);
                }
            }
            for inner in &try_stmt.orelse {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for inner in &try_stmt.finalbody {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::Match(match_stmt) => {
            collect_mutation_receiver_names_expr(match_stmt.subject.as_ref(), receivers);
            for case in &match_stmt.cases {
                if let Some(guard) = case.guard.as_deref() {
                    collect_mutation_receiver_names_expr(guard, receivers);
                }
                for inner in &case.body {
                    collect_mutation_receiver_names_stmt(inner, receivers);
                }
            }
        }
        // a nested def/class body has its own scope — the names its own
        // mutations touch are not this outer body's receivers to forget
        Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
        Stmt::Pass(_)
        | Stmt::Break(_)
        | Stmt::Continue(_)
        | Stmt::Global(_)
        | Stmt::Nonlocal(_)
        | Stmt::Import(_)
        | Stmt::ImportFrom(_)
        | Stmt::TypeAlias(_)
        | Stmt::IpyEscapeCommand(_) => {}
    }
}

/// A (possibly destructuring) assign/aug-assign/ann-assign target's own
/// SUBSCRIPT-STORE receivers (`X[k] = v` at any nesting depth of a
/// tuple/list/starred target) — the leftmost base name under each
/// `Subscript.value` collected via `collect_leftmost_receiver_name`.
/// Non-subscript target shapes (a bare name, an attribute write) name no
/// subscript-store receiver here; a bare name's own binding is already
/// covered by `collect_bound_names`'s separate scan, and an attribute
/// write's receiver is covered by this same walk's expression side
/// (`collect_mutation_receiver_names_expr`'s `Expr::Attribute` arm on
/// the RHS/nested reads) — assignment TARGETS reach this function only
/// for their subscript form, which is the one shape `forget_names_bound_by_stmt`
/// cannot already see.
fn collect_subscript_store_receiver(target: &Expr, receivers: &mut HashSet<String>) {
    match target {
        Expr::Subscript(subscript) => {
            collect_leftmost_receiver_name(subscript.value.as_ref(), receivers);
            collect_mutation_receiver_names_expr(subscript.slice.as_ref(), receivers);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_subscript_store_receiver(element, receivers);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_subscript_store_receiver(element, receivers);
            }
        }
        Expr::Starred(starred) => collect_subscript_store_receiver(starred.value.as_ref(), receivers),
        _ => {}
    }
}

/// Walks one expression tree, collecting every ATTRIBUTE-CALL's receiver
/// base name (`X.method(...)` — the func of a `Call` being an
/// `Attribute`) into `receivers`, recursing into every sub-expression a
/// mutation could hide inside (call arguments, comparison operands,
/// boolean/binary/unary operands, container displays, the ternary's
/// three arms, f-string interpolations, comprehension element/iterable/
/// condition parts, await/yield operands) so a nested mutating call
/// anywhere in the tree is caught, not only at the statement's own top
/// level.
fn collect_mutation_receiver_names_expr(expr: &Expr, receivers: &mut HashSet<String>) {
    match expr {
        Expr::Call(call) => {
            if let Expr::Attribute(attribute) = call.func.as_ref() {
                collect_leftmost_receiver_name(attribute.value.as_ref(), receivers);
            }
            collect_mutation_receiver_names_expr(call.func.as_ref(), receivers);
            for arg in &call.arguments.args {
                collect_mutation_receiver_names_expr(arg, receivers);
            }
            for keyword in &call.arguments.keywords {
                collect_mutation_receiver_names_expr(&keyword.value, receivers);
            }
        }
        Expr::Attribute(attribute) => collect_mutation_receiver_names_expr(attribute.value.as_ref(), receivers),
        Expr::Subscript(subscript) => {
            collect_mutation_receiver_names_expr(subscript.value.as_ref(), receivers);
            collect_mutation_receiver_names_expr(subscript.slice.as_ref(), receivers);
        }
        Expr::Named(named) => {
            collect_mutation_receiver_names_expr(named.target.as_ref(), receivers);
            collect_mutation_receiver_names_expr(named.value.as_ref(), receivers);
        }
        Expr::BoolOp(op) => {
            for value in &op.values {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Expr::BinOp(op) => {
            collect_mutation_receiver_names_expr(op.left.as_ref(), receivers);
            collect_mutation_receiver_names_expr(op.right.as_ref(), receivers);
        }
        Expr::UnaryOp(op) => collect_mutation_receiver_names_expr(op.operand.as_ref(), receivers),
        Expr::If(if_expr) => {
            collect_mutation_receiver_names_expr(if_expr.test.as_ref(), receivers);
            collect_mutation_receiver_names_expr(if_expr.body.as_ref(), receivers);
            collect_mutation_receiver_names_expr(if_expr.orelse.as_ref(), receivers);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_mutation_receiver_names_expr(element, receivers);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_mutation_receiver_names_expr(element, receivers);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                collect_mutation_receiver_names_expr(element, receivers);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    collect_mutation_receiver_names_expr(key, receivers);
                }
                collect_mutation_receiver_names_expr(&item.value, receivers);
            }
        }
        Expr::Compare(compare) => {
            collect_mutation_receiver_names_expr(compare.left.as_ref(), receivers);
            for comparator in &compare.comparators {
                collect_mutation_receiver_names_expr(comparator, receivers);
            }
        }
        Expr::Starred(starred) => collect_mutation_receiver_names_expr(starred.value.as_ref(), receivers),
        Expr::Slice(slice) => {
            if let Some(lower) = slice.lower.as_deref() {
                collect_mutation_receiver_names_expr(lower, receivers);
            }
            if let Some(upper) = slice.upper.as_deref() {
                collect_mutation_receiver_names_expr(upper, receivers);
            }
            if let Some(step) = slice.step.as_deref() {
                collect_mutation_receiver_names_expr(step, receivers);
            }
        }
        Expr::FString(fstring) => {
            for element in fstring.value.elements() {
                if let Some(interpolation) = element.as_interpolation() {
                    collect_mutation_receiver_names_expr(interpolation.expression.as_ref(), receivers);
                }
            }
        }
        Expr::Await(inner) => collect_mutation_receiver_names_expr(inner.value.as_ref(), receivers),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Expr::YieldFrom(inner) => collect_mutation_receiver_names_expr(inner.value.as_ref(), receivers),
        Expr::ListComp(comp) => {
            collect_mutation_receiver_names_expr(comp.elt.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        Expr::SetComp(comp) => {
            collect_mutation_receiver_names_expr(comp.elt.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        Expr::DictComp(comp) => {
            if let Some(key) = comp.key.as_deref() {
                collect_mutation_receiver_names_expr(key, receivers);
            }
            collect_mutation_receiver_names_expr(comp.value.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        Expr::Generator(comp) => {
            collect_mutation_receiver_names_expr(comp.elt.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        // a lambda's own body is a separate scope — mirrors
        // collect_walrus_names/bind_walrus_targets's same posture
        Expr::Lambda(_) => {}
        _ => {}
    }
}

/// A comprehension's own generator clauses: each `iter` expression and
/// every `if` condition, in source order — the loop VARIABLE itself
/// introduces no receiver to collect.
fn collect_comprehension_generators(generators: &[ruff_python_ast::Comprehension], receivers: &mut HashSet<String>) {
    for generator in generators {
        collect_mutation_receiver_names_expr(&generator.iter, receivers);
        for condition in &generator.ifs {
            collect_mutation_receiver_names_expr(condition, receivers);
        }
    }
}

/// The leftmost `Name` reachable under a receiver expression, walking
/// THROUGH a chained call's own func-attribute — unlike
/// `receiver_base_name` (which stops at a `Call` and answers `None`),
/// this function keeps walking into a `Call`'s `func` so
/// `grouped.setdefault(...).append(...)`'s outer receiver
/// (`grouped.setdefault(...)`, itself a `Call`) still resolves to
/// `grouped`. Every argument/keyword of a call encountered along the
/// way is ALSO walked for its own nested mutations (a mutation can hide
/// inside an argument expression, e.g. `xs.append(ys.pop())`), and a
/// non-Name/Attribute/Call receiver (a subscript, a literal, …) yields
/// no base name — this function only ever forgets a plain identifier.
fn collect_leftmost_receiver_name(receiver: &Expr, receivers: &mut HashSet<String>) {
    match receiver {
        Expr::Name(name) => {
            receivers.insert(name.id.as_str().to_owned());
        }
        Expr::Attribute(attribute) => collect_leftmost_receiver_name(attribute.value.as_ref(), receivers),
        Expr::Call(call) => {
            collect_leftmost_receiver_name(call.func.as_ref(), receivers);
            for arg in &call.arguments.args {
                collect_mutation_receiver_names_expr(arg, receivers);
            }
            for keyword in &call.arguments.keywords {
                collect_mutation_receiver_names_expr(&keyword.value, receivers);
            }
        }
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

    fn no_imports_resolver() -> ModuleResolver<'static> {
        &|_: &str| None
    }

    /// The byte offset of `needle`'s own first character in `source` —
    /// a readable way to name a test position ("the `s` of `samples`")
    /// rather than a bare integer that says nothing about what it
    /// points at.
    fn offset_of(source: &str, needle: &str) -> TextSize {
        let byte_offset = source.find(needle).unwrap_or_else(|| panic!("{needle:?} not found in fixture"));
        TextSize::try_from(byte_offset).expect("fixture offsets fit in TextSize")
    }

    /// A module whose ONLY refinement vocabulary is a bare
    /// `Literal[...]` annotation — no `type` alias, no `Annotated`
    /// import — is judged: the engagement gate reads the `Literal`
    /// import as refinement vocabulary too. Pins the gate widening;
    /// before it, this module returned zero findings without ever
    /// reaching the walk.
    #[test]
    fn a_literal_only_module_is_judged() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Literal\n",
            "\n",
            "def pick(level: Literal[1, 2, 4]) -> Literal[1, 2]:\n",
            "    return level\n",
        );
        let module = parsed(source);
        let findings = findings_for_module_at(&module, no_imports_resolver(), &kernel, None);
        assert!(
            !findings.is_empty(),
            "a Literal-only module must reach the walk and judge its rows"
        );
    }

    /// `refined_set_at_position` at a PARAMETER's own annotation
    /// position answers the declared set — a-audio-level.py's own
    /// `samples: Annotated[list[Sample], Field(min_length=1)]` row.
    /// The stated branch reads through the SAME `declared_refinement`
    /// `seed_parameters` calls to seed the parameter itself, so this
    /// pins that the query and the seed never drift apart.
    #[test]
    fn a_parameter_annotation_position_answers_its_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "\n",
            "def audio_level(samples: Sample) -> None:\n",
            "    pass\n",
        );
        let module = parsed(source);
        // a position inside the PARAMETER's own annotation name ("Sample"
        // in "samples: Sample")
        let position = offset_of(source, "Sample) -> None");
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected a declared set at the parameter annotation"));
        assert_eq!(format_for_diagnostics(&set), ">= -2 && <= 2");
    }

    /// `refined_set_at_position` at the RETURN annotation's own
    /// position answers the declared set — audio_level.py's own `->
    /// Level` (`Level = Annotated[float, Field(ge=0.0, le=1.0)]`). The
    /// brief that specified this unit named "the Level parameter", but
    /// `Level` is the fixture's RETURN annotation, not a parameter —
    /// this test covers the position the fixture actually has.
    #[test]
    fn a_return_annotation_position_answers_its_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def audio_level(x: float) -> Level:\n",
            "    return x\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "Level:\n    return");
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected a declared set at the return annotation"));
        assert_eq!(format_for_diagnostics(&set), ">= 0 && <= 1");
    }

    /// The hover CLI's own pair — `refined_set_at_position` then
    /// `format_for_hover` — read at the RETURN annotation's own `Level`
    /// spells the same facts `format_for_diagnostics` reads above, in
    /// the hover vocabulary rather than the diagnostic one. Pins that
    /// `refinedpy-check --hover` prints what this seam pair actually
    /// answers, since the bin crate has no test convention of its own
    /// (`src/bin/refinedpy_check.rs` carries no `#[cfg(test)]`) and this
    /// is where every other `refined_set_at_position` position test
    /// already lives.
    #[test]
    fn the_hover_seam_pair_spells_a_declared_alias_at_a_usage_site() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def audio_level(x: float) -> Level:\n",
            "    return x\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "Level:\n    return");
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected a declared set at the return annotation"));
        let spelled = refined_sets::format_for_hover::format_for_hover(&set)
            .unwrap_or_else(|| panic!("expected a hover spelling for Level"));
        assert_eq!(spelled, "{0 ≤ 𝑥 ≤ 1}");
    }

    /// A position at the ALIAS DECLARATION's own name (`type Level =
    /// Annotated[...]`, the `Level` before `=`) answers the alias's own
    /// compiled set — the SAME set a parameter annotated `x: Level`
    /// gets. Glyph spelling copied from `format_for_hover.rs`'s own
    /// fixtures (`"{0 ≤ 𝑥 ≤ 100}"`), never from memory.
    #[test]
    fn an_alias_declarations_own_name_answers_its_compiled_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def audio_level(x: float) -> Level:\n",
            "    return x\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "Level = Annotated");
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected the alias's own compiled set at its declaration name"));
        let spelled = refined_sets::format_for_hover::format_for_hover(&set)
            .unwrap_or_else(|| panic!("expected a hover spelling for Level's declaration"));
        assert_eq!(spelled, "{0 ≤ 𝑥 ≤ 1}");
    }

    /// The `X = Annotated[...]` (no `type` keyword) and `X: TypeAlias =
    /// Annotated[...]` spellings answer identically at their own name —
    /// the other two of the three spellings `compile_aliases` reads,
    /// alongside the `type X = ...` spelling pinned above.
    #[test]
    fn the_plain_and_annotated_alias_spellings_answer_at_their_own_name_too() {
        let Some(kernel) = loaded_kernel() else { return };
        let plain_source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        );
        let plain = parsed(plain_source);
        let plain_position = offset_of(plain_source, "Level = Annotated");
        let plain_set = refined_set_at_position(&plain, no_imports_resolver(), &kernel, plain_position)
            .unwrap_or_else(|| panic!("expected the plain-assignment alias's own set at its name"));
        assert_eq!(format_for_diagnostics(&plain_set), ">= 0 && <= 1");

        let annotated_source = concat!(
            "from typing import Annotated, TypeAlias\n",
            "from pydantic import Field\n",
            "Level: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        );
        let annotated = parsed(annotated_source);
        let annotated_position = offset_of(annotated_source, "Level: TypeAlias");
        let annotated_set = refined_set_at_position(&annotated, no_imports_resolver(), &kernel, annotated_position)
            .unwrap_or_else(|| panic!("expected the `: TypeAlias =` spelling's own set at its name"));
        assert_eq!(format_for_diagnostics(&annotated_set), ">= 0 && <= 1");
    }

    /// A `def`'s own name answers its return refinement, when one is
    /// readable — the same claim a call to it yields.
    #[test]
    fn a_defs_own_name_answers_its_declared_return_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def audio_level(x: float) -> Level:\n",
            "    return x\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "audio_level(x: float)");
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected the def's own declared return set at its name"));
        assert_eq!(format_for_diagnostics(&set), ">= 0 && <= 1");
    }

    /// A `def` with a BARE base-sort return annotation (`-> float`, no
    /// `Annotated`/alias) answers NOTHING at its own name —
    /// `declared_refinement` alone is read there, with no
    /// `base_sort_return_refinement` fallback, exactly matching
    /// `declared_refinement`'s own doc: a base sort states nothing this
    /// table reads. The module still carries an `Annotated` import (on
    /// an unrelated parameter) so the engagement gate is passed for the
    /// same reason the walk reaches this module at all — the point
    /// pinned here is the def-name branch's own decline, not the
    /// module-level early exit `a_position_naming_neither_a_declaration_
    /// nor_a_recorded_expression_answers_nothing` already covers.
    #[test]
    fn a_defs_own_name_with_a_bare_sort_return_answers_nothing() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "\n",
            "def audio_level(x: float) -> float:\n",
            "    return x\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "audio_level(x: float)");
        assert!(
            refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_none(),
            "a bare `-> float` must not fabricate a claim at the def's own name"
        );
    }

    /// Ledger 260: `total`'s own position in `audio_level_unclamped.py`
    /// (`total = sum(s * s for s in samples)`) now answers the derived
    /// total's set — `walk_relational_sum` records the kernel's proved
    /// total at the Assign statement's own range the same way an
    /// ordinary assignment's RHS records its evaluated node
    /// (`expressions.rs::evaluate_expression`'s `record_evaluation`
    /// call), so `refined_set_at_position` finds it as the smallest
    /// covering recorded range. Was pinned the other way
    /// (`the_measured_total_position_answers_no_set_today`,
    /// refinedpy_lsp/src/lib.rs) before this unit threaded the publish
    /// through.
    #[test]
    fn the_measured_total_position_now_answers_the_derived_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "import math\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def audio_level_unclamped(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
            "    total = sum(s * s for s in samples)\n",
            "    return math.sqrt(total / len(samples))\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "total = sum");
        assert!(
            refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
            "total's own position must answer the kernel's derived set now that the recognizer publishes it"
        );
    }

    /// The quotient binding's twin of the test above: `mean = total /
    /// len(samples)` is consumed by the same recognizer
    /// (`walk_relational_sum`'s divided-into arm), which binds the
    /// kernel's quotient — and now records it at the Assign statement's
    /// own range too, so `mean`'s own position answers the quotient's
    /// set instead of nothing.
    #[test]
    fn the_quotient_binding_position_answers_the_derived_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def mean_square(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
            "    total = sum(s * s for s in samples)\n",
            "    mean = total / len(samples)\n",
            "    return mean\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "mean = total");
        assert!(
            refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
            "the quotient binding's own position must answer the kernel's derived set"
        );
    }

    /// A count-alias binding (`count = len(samples)`) consumed ahead of
    /// the division now binds and publishes — the alias's own name reads
    /// an integer-sorted set admitting the sequence's length window, and
    /// its position answers that same set, exactly the way the total's
    /// and the quotient's own positions do above. The window here is
    /// `min_length=1` with no upper bound, so the alias's set is at
    /// least the count-alias fold's own floor and never a fabricated
    /// exact value.
    #[test]
    fn a_count_alias_binding_now_answers_the_derived_length_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "\n",
            "def mean_square(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
            "    total = sum(s * s for s in samples)\n",
            "    count = len(samples)\n",
            "    mean = total / count\n",
            "    return mean\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "count = len");
        assert!(
            refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
            "the count-alias binding's own position must answer the kernel's derived length window"
        );
    }

    /// The count-alias spelling — `count = len(samples)` then
    /// `mean = total / count` — judges IDENTICALLY to the direct
    /// spelling `mean = total / len(samples)`, and the statement after
    /// the consumed alias-and-division pair still walks rather than
    /// being skipped by stale bookkeeping (`walk_relational_sum`'s own
    /// `skip_statements` count, threaded through `check.rs`'s
    /// `folded_division_at` range). Each body assigns the mean into an
    /// `Age`-typed slot with an out-of-set literal fallback (`over`)
    /// immediately after, so a wrong skip count — either leaving the
    /// alias assignment to be walked a second time as an ordinary
    /// statement, or swallowing `over` into the skipped range — would
    /// change the finding count or the position `over`'s own fire lands
    /// at. `mean`'s own position is also read back identically in both
    /// spellings, pinning that the aliased fold answers the same derived
    /// set the direct fold does, not merely the same finding count.
    #[test]
    fn the_count_alias_spelling_judges_identically_to_the_direct_spelling_and_the_next_statement_still_walks() {
        let Some(kernel) = loaded_kernel() else { return };
        let direct_source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=0.0, le=2.0)]\n",
            "Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(samples: Annotated[list[Sample], Field(min_length=1)]) -> None:\n",
            "    total = sum(s for s in samples)\n",
            "    mean = total / len(samples)\n",
            "    over: Age = 200\n",
        );
        let aliased_source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=0.0, le=2.0)]\n",
            "Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(samples: Annotated[list[Sample], Field(min_length=1)]) -> None:\n",
            "    total = sum(s for s in samples)\n",
            "    count = len(samples)\n",
            "    mean = total / count\n",
            "    over: Age = 200\n",
        );
        let direct = parsed(direct_source);
        let aliased = parsed(aliased_source);
        let direct_findings = findings_for_module(&direct, &kernel);
        let aliased_findings = findings_for_module(&aliased, &kernel);
        let direct_messages: Vec<&str> = direct_findings.iter().map(|f| f.message.as_str()).collect();
        let aliased_messages: Vec<&str> = aliased_findings.iter().map(|f| f.message.as_str()).collect();
        // the trailing `over: Age = 200` fires in both spellings — the
        // statement after the consumed pair (one statement wider in the
        // aliased body) still walks rather than being skipped
        assert_eq!(direct_findings.len(), 1, "direct spelling: {direct_messages:?}");
        assert_eq!(aliased_findings.len(), 1, "aliased spelling: {aliased_messages:?}");
        assert_eq!(direct_findings[0].code, "RTS7001");
        assert_eq!(aliased_findings[0].code, "RTS7001");
        assert!(direct_messages[0].contains("'200'"), "{direct_messages:?}");
        assert!(aliased_messages[0].contains("'200'"), "{aliased_messages:?}");
        // The fold's OWN identity (the aliased spelling wiring the same
        // kernel program as the direct one) is pinned at the API level
        // by relational_sum.rs's
        // `the_count_alias_shape_folds_identically_to_the_direct_spelling`
        // — a folded division records no evaluated node at `mean`'s
        // position in EITHER spelling (only the recognized sum's own
        // binding records one), so a position read here would assert a
        // channel the fold does not publish. This pin holds the two
        // walk-level truths: identical single designed findings, and
        // the statement after the consumed pair still walking.
    }

    /// A position that names neither a parameter's own annotation, a
    /// return annotation, nor any expression the walk evaluated (here,
    /// the function's OWN NAME) answers nothing — the stated branch
    /// declines, and no recorded range covers a name the walk never
    /// evaluates as an expression.
    ///
    /// A narrowed LOCAL's own position (`age = 40; if age > 0: <hover
    /// age here>`) is the derived-flow case the brief also asks about;
    /// asserting it needs the walk to have recorded a node exactly at
    /// the READ site, which in turn needs an LSP-shaped identifier
    /// resolution this unit does not build (`evaluate_expression` is
    /// keyed on the EXPRESSION node's own range, and a bare `Expr::Name`
    /// read is one such node, so the machinery here already covers it —
    /// but pinning "reachable" requires walking a guarded branch and
    /// then locating the exact `Expr::Name` byte range the guard body
    /// re-reads, which is fixture-fragile in a way the declared-position
    /// tests above are not). Recorded here as a gap rather than a test
    /// built to look green: the declared-branch tests above are the
    /// ones this unit can assert without guessing at ruff's own byte
    /// offsets for a re-read name.
    #[test]
    fn a_position_naming_neither_a_declaration_nor_a_recorded_expression_answers_nothing() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def audio_level() -> None:\n",
            "    pass\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "audio_level");
        assert!(refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_none());
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

    /// A module whose ONLY refinement vocabulary is an inline
    /// `Annotated[...]` parameter annotation — no `type X = ...`
    /// statement anywhere — still walks and fires. Before this gate
    /// read `imports.annotated_names`, `compile_aliases` alone fed the
    /// early-return check, so this exact shape returned zero findings
    /// for the whole module regardless of what its body did.
    #[test]
    fn an_inline_only_annotated_parameter_with_no_type_alias_still_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "def check_age(age: Annotated[int, Field(ge=0, le=120)]) -> None:\n",
            "    over: Annotated[int, Field(ge=0, le=120)] = 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(
            findings.len(),
            1,
            "want the fire for 200, from a module with no `type` alias at all: {:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
        assert_eq!(findings[0].code, "RTS7001");
        assert!(findings[0].message.contains("'200'"), "{}", findings[0].message);
    }

    /// A module with neither a `type` alias nor a recognized `Annotated`
    /// import — ordinary Python this checker has no vocabulary for —
    /// still returns empty through the same zero-cost early return, never
    /// reaching the walk.
    #[test]
    fn a_module_with_no_aliases_and_no_annotated_import_stays_empty() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def add(a: int, b: int) -> int:\n",
            "    return a + b\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(findings.is_empty(), "{:?}", findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>());
    }

    /// RTS7003 — an inline `Annotated[...]` annotation compiles, but its
    /// two bounds contradict each other, so the kernel proves the
    /// declared set admits nothing. Mirrors the Go twin's own emptiness
    /// fire (`annotation_file_facts.go`, the `emptiness` ask
    /// immediately after a successful compile).
    #[test]
    fn an_annotation_whose_bounds_contradict_denotes_the_empty_set() {
        let Some(kernel) = loaded_kernel() else { return };
        // `findings_for_module_at` returns no findings at all when the
        // module has neither a `type X = ...` alias NOR a recognized
        // `Annotated` import (its own early-return guard) — this
        // fixture's `from typing import Annotated` line alone already
        // clears that guard, so the `type Age` alias here is present to
        // exercise the alias path too, not because it is required.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    impossible: Annotated[int, Field(ge=10, le=5)] = 7\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let empties: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7003").collect();
        assert_eq!(
            empties.len(),
            1,
            "want exactly one empty-set finding: {:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
        assert!(empties[0].message.contains("denotes the empty set"), "{}", empties[0].message);
    }

    /// A ordinary, inhabited `Annotated[...]` annotation never fires
    /// RTS7003 — the emptiness ask is a courtesy on an already-compiled
    /// set, never a blocker on a set that admits values.
    #[test]
    fn an_ordinary_inhabited_annotation_never_fires_the_empty_set_finding() {
        let Some(kernel) = loaded_kernel() else { return };
        // see the `type Age` comment above — the `from typing import
        // Annotated` line alone already clears
        // `findings_for_module_at`'s alias-or-Annotated-import guard.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    fine: Annotated[int, Field(ge=0, le=120)] = 42\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7003"),
            "{:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
    }

    /// RTS7004 — the annotation is recognizably this table's OWN
    /// `Annotated[...]` vocabulary (the imported `Annotated` identity as
    /// the subscript head), but an unrecognized `Field` kwarg refuses
    /// the whole statement. Mirrors the Go twin's own `RootsInSurface`
    /// gate (`annotation_file_facts.go`): a recognized-root annotation
    /// the checker cannot read is never dropped silently.
    #[test]
    fn an_annotated_statement_with_an_unrecognized_field_kwarg_is_unhonorable() {
        let Some(kernel) = loaded_kernel() else { return };
        // see the `type Age` comment above — the `from typing import
        // Annotated` line alone already clears
        // `findings_for_module_at`'s alias-or-Annotated-import guard.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    over: Annotated[int, Field(unknown_kwarg=1)] = 42\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(
            findings.len(),
            1,
            "want exactly one unhonorable-statement finding: {:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
        assert_eq!(findings[0].code, "RTS7004");
        assert!(findings[0].message.contains("Annotated[...]"), "{}", findings[0].message);
    }

    /// A plain, no-vocabulary annotation (an ordinary class name, not
    /// this table's `Annotated[...]` root) never fires RTS7004 — it is
    /// ordinary Python this table has no vocabulary for, not a refused
    /// statement.
    #[test]
    fn a_plain_unrelated_annotation_never_fires_the_unhonorable_statement_finding() {
        let Some(kernel) = loaded_kernel() else { return };
        // see the `type Age` comment above — the `from typing import
        // Annotated` line alone already clears
        // `findings_for_module_at`'s alias-or-Annotated-import guard.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    label: str = \"ok\"\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7004"),
            "{:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
    }

    /// `f-type-nodes.py`'s `list_annotation_parameter` row: a `list[int]`
    /// PARAMETER read through `ages[0]` against `Age` — `int`'s own
    /// unbounded ray admits values outside `Age`'s [0, 120] window, so
    /// this fires (`seed_parameters`' star-of-a-set seed, read back
    /// through `collection_models::subscript_read`'s new `Kind::Set` arm).
    #[test]
    fn a_list_int_parameters_element_read_fires_against_a_narrower_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def list_annotation_parameter(ages: list[int]) -> Age:\n",
            "    return ages[0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7001");
        assert!(findings[0].message.contains("'Age'"), "{}", findings[0].message);
    }

    /// The same shape, but the declared element ITSELF is `Age` (an
    /// alias, not a bare sort): `list[Age]`'s element already resolves
    /// through the ordinary alias path (no fallback needed), and reading
    /// its element against `Age` is a set-equals-itself Silent — pinning
    /// that the star seed only ever WIDENS what silently determines,
    /// never narrows an already-working alias-element row.
    #[test]
    fn a_list_of_the_declared_sets_own_alias_element_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def first_age(ages: list[Age]) -> Age:\n",
            "    return ages[0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 0, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    }

    // --- BARE CLASS-NAME PARAMETERS (`seed_parameters`' class branch) ---

    /// A PARAMETER annotated with a bare class name (`request: AudioRequest`)
    /// whose class declares an annotated scalar field: the field read
    /// through `request.level` is judged against `Level`'s own [0, 100]
    /// window — in range stays Silent, out of range fires RTS7001. Pins
    /// that `seed_parameters`' new class branch binds the parameter as a
    /// tagged `Kind::Object` whose `keys` the ordinary attribute-read path
    /// (`evaluate_attribute_read` → `field_read_through_model` →
    /// `field_read`) actually consumes — this is the fix's whole point:
    /// before it, `request` was never bound at all, and both reads below
    /// stayed silently unknown with nothing firing.
    #[test]
    fn a_bare_class_name_parameters_scalar_field_read_judges_against_its_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Level = Annotated[int, Field(ge=0, le=100)]\n",
            "class AudioRequest:\n",
            "    def __init__(self, level: Level) -> None:\n",
            "        self.level = level\n",
            "def in_range(request: AudioRequest) -> Level:\n",
            "    return request.level\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(
            findings.len(),
            0,
            "a parameter's own declared field read back against its own declared set is Silent: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// The same class-name parameter shape, but the field is read into a
    /// NARROWER declared return than the field's own declaration: the
    /// parameter's `Level` field ([0, 100]) admits values outside a
    /// tighter `Quiet` ([0, 20]) return, so this fires RTS7001 — the
    /// out-of-set leg of the same fix.
    #[test]
    fn a_bare_class_name_parameters_scalar_field_read_fires_against_a_narrower_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Level = Annotated[int, Field(ge=0, le=100)]\n",
            "type Quiet = Annotated[int, Field(ge=0, le=20)]\n",
            "class AudioRequest:\n",
            "    def __init__(self, level: Level) -> None:\n",
            "        self.level = level\n",
            "def maybe_quiet(request: AudioRequest) -> Quiet:\n",
            "    return request.level\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7001");
        assert!(findings[0].message.contains("'Quiet'"), "{}", findings[0].message);
    }

    /// A bare class-name parameter whose class declares a SEQUENCE field
    /// (`samples: Annotated[list[float], Field(min_length=1)]`): indexing
    /// the field proves the sequence machinery sees both the window and
    /// the sort — mirrors `a_list_int_parameters_element_read_fires_
    /// against_a_narrower_declared_set` above (a bare `float`'s own
    /// unbounded ray admits values outside `Score`'s [0.0, 1.0] window),
    /// but the sequence lives one level down, inside a field, rather than
    /// at the parameter itself: `class_field_value`'s sequence arm must
    /// build the SAME `Kind::Set` repetition `seed_parameters`' own
    /// sequence-container branch builds, or `batch.samples[0]` would read
    /// as `unknown()` (undetermined, not a fire) instead.
    #[test]
    fn a_bare_class_name_parameters_sequence_field_element_read_fires_against_a_narrower_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Score = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "class ScoredBatch:\n",
            "    def __init__(self, samples: Annotated[list[float], Field(min_length=1)]) -> None:\n",
            "        self.samples = samples\n",
            "def first_score(batch: ScoredBatch) -> Score:\n",
            "    return batch.samples[0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7001");
        assert!(findings[0].message.contains("'Score'"), "{}", findings[0].message);
    }

    /// A parameter annotated with a name that is NOT a class in this
    /// module's table (an ordinary undeclared name, or a name that never
    /// resolves to any `ClassModel`): the new branch must decline exactly
    /// as today — no binding, no finding — falling through to the ordinary
    /// `declared_refinement` read, which also states nothing for it.
    #[test]
    fn a_class_name_not_in_the_table_still_declines_as_today() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def uses_unknown_class(thing: NotAClass) -> None:\n",
            "    pass\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an annotation naming no known class must decline without fabricating a finding: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// A class-name parameter whose field has NO declared refinement (an
    /// ordinary unannotated `self.level = level` field): `class_parameter_
    /// object` seeds NOTHING for that key (absent, not a fabricated set),
    /// so `request.level` reads back as `unknown()` — flowing into a
    /// declared `-> Level` return, `judge` falls through every typed arm to
    /// its undetermined catch-all, firing RTS7002 naming that blocker
    /// (`"the flowing value is not yet readable"`), never silently
    /// admitted and never a false RTS7001.
    #[test]
    fn a_class_name_parameters_field_with_no_declared_refinement_reads_undetermined() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Level = Annotated[int, Field(ge=0, le=100)]\n",
            "class AudioRequest:\n",
            "    def __init__(self, level) -> None:\n",
            "        self.level = level\n",
            "def read_it(request: AudioRequest) -> Level:\n",
            "    return request.level\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7002");
        assert!(
            findings[0].message.contains("not yet readable"),
            "{}",
            findings[0].message
        );
    }

    /// The naming unit's own pinned shape (`python-c-extension-boundary.md`):
    /// `x: Age = torch.arange(5)` — a call on an attribute chain rooted at
    /// an imported-but-unmodeled module name — reads undetermined naming
    /// `torch` rather than the generic "not yet readable" wording, since
    /// `torch` is never bound to a real tracked value (the walk carries no
    /// same-project module named `torch` for the no-resolver test harness
    /// to find) and is not one of `expressions::MODELED_MODULE_NAMES`.
    #[test]
    fn an_unmodeled_module_call_into_a_declared_position_names_the_module() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "import torch\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def read_it() -> None:\n",
            "    x: Age = torch.arange(5)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7002");
        assert!(findings[0].message.contains("'torch'"), "{}", findings[0].message);
        assert!(findings[0].message.contains("no model for"), "{}", findings[0].message);
    }

    /// The same naming, at a `return` sink rather than an `AnnAssign`:
    /// `return torch.arange(5)` under a declared `-> Age` return names
    /// `torch` too, proving the naming step is not AnnAssign-specific.
    #[test]
    fn an_unmodeled_module_call_returned_from_a_declared_function_names_the_module() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "import torch\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def read_it() -> Age:\n",
            "    return torch.arange(5)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7002");
        assert!(findings[0].message.contains("'torch'"), "{}", findings[0].message);
    }

    /// A MODELED module (`math`, already in `MODELED_MODULE_NAMES`) whose
    /// own unmodeled FUNCTION (`math.frexp`, not one of the rows
    /// `math_call_result` reads) is a different, narrower gap the naming
    /// unit does NOT claim — the sentence stays the generic wording, never
    /// misnaming `math` as if the checker carried no model for it at all.
    #[test]
    fn an_unmodeled_function_on_a_modeled_module_keeps_the_generic_sentence() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "import math\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def read_it() -> None:\n",
            "    x: Age = math.frexp(5.0)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert_eq!(findings[0].code, "RTS7002");
        assert!(!findings[0].message.contains("'math'"), "{}", findings[0].message);
        assert!(findings[0].message.contains("not yet readable"), "{}", findings[0].message);
    }

    /// RUNG 2's own end-to-end fixture: a tiny hand-authored manifest for
    /// a two-function module (`scale(Scalar factor)`, `label(str text)`),
    /// exercised against all four shapes
    /// `python-c-extension-boundary.md`'s manifest-reader unit names —
    /// one module, one temp directory standing in for the checked file's
    /// own `entry_directory`.
    ///
    /// 1. A FITTING call (`widgets.scale(2)`): the entry judges silently
    ///    (no RTS7001), and the return still declines naming the missing
    ///    producer.
    /// 2. An ESCAPING call (`widgets.scale("nope")`): the entry fires
    ///    RTS7001, naming the module, function, parameter, and both
    ///    words.
    /// 3. An UNLISTED function (`widgets.unlisted(1)`): the module HAS a
    ///    manifest, but it names no row for `unlisted` — the narrower
    ///    "manifest names no entry" decline, not rung 1's plain one.
    /// 4. An UNMANIFESTED module (`import other_widgets;
    ///    other_widgets.scale(2)`): no manifest file exists for
    ///    `other_widgets` at all — rung 1's plain "no model for" decline.
    #[test]
    fn the_manifest_reader_template_covers_all_four_recognition_shapes() {
        let Some(kernel) = loaded_kernel() else { return };
        let root = std::env::temp_dir().join(format!(
            "refinedpy_check_binding_manifest_fixture_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("create temp dir");
        let manifest_json = serde_json::json!({
            "scale": {"entry": "scale(Scalar factor)", "producer": "widgets_scale_impl"},
            "label": {"entry": "label(str text)", "producer": "widgets_label_impl"},
        });
        std::fs::write(root.join("widgets.manifest.json"), manifest_json.to_string()).expect("write manifest");

        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "import widgets\n",
            "import other_widgets\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def fitting() -> None:\n",
            "    x: Age = widgets.scale(2)\n",
            "def escaping() -> None:\n",
            "    widgets.scale(\"nope\")\n",
            "def unlisted() -> None:\n",
            "    x: Age = widgets.unlisted(1)\n",
            "def unmanifested() -> None:\n",
            "    x: Age = other_widgets.scale(2)\n",
        ));
        let no_imports: ModuleResolver = &|_: &str| None;
        let findings = findings_for_module_at(&module, no_imports, &kernel, Some(&root));

        // 1. FITTING: no RTS7001 for `fitting`'s own body, and its own
        // RTS7002 blocker names the missing producer.
        let fitting_blocker = findings
            .iter()
            .find(|f| f.code == "RTS7002" && f.message.contains("widgets.scale") && f.message.contains("widgets_scale_impl"))
            .expect(&format!("fitting's own blocker must name the missing producer: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
        assert!(fitting_blocker.message.contains("no producer exports its return fact"), "{}", fitting_blocker.message);

        // 2. ESCAPING: an RTS7001 naming the module, function, parameter,
        // and both value words.
        let escaping_fire = findings
            .iter()
            .find(|f| f.code == "RTS7001" && f.message.contains("widgets.scale"))
            .expect(&format!("the escaping call must fire: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
        assert!(escaping_fire.message.contains("'factor: Scalar'"), "{}", escaping_fire.message);
        assert!(escaping_fire.message.contains("a str"), "{}", escaping_fire.message);

        // 3. UNLISTED: the manifest names no row for `unlisted` — the
        // narrower manifest decline, not the plain module-level one.
        let unlisted_blocker = findings
            .iter()
            .find(|f| f.code == "RTS7002" && f.message.contains("manifest names no entry"))
            .expect(&format!("the unlisted call must name the missing entry: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
        assert!(unlisted_blocker.message.contains("'widgets'"), "{}", unlisted_blocker.message);
        assert!(unlisted_blocker.message.contains("'unlisted'"), "{}", unlisted_blocker.message);

        // 4. UNMANIFESTED: no manifest file exists for `other_widgets` —
        // rung 1's own plain decline.
        let unmanifested_blocker = findings
            .iter()
            .find(|f| f.code == "RTS7002" && f.message.contains("'other_widgets'"))
            .expect(&format!("the unmanifested module must fall back to rung 1's naming: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
        assert!(unmanifested_blocker.message.contains("no model for"), "{}", unmanifested_blocker.message);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_list_element_compound_write_past_the_declared_ceiling_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        // ages[0] is 2 after the `//= 2`; += 190 writes 192, past Age's
        // 120 ceiling — the element-level judging `walk_subscript_
        // aug_assign` now applies through `aug_assign_refinements`'
        // `list[Age]` entry.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    ages: list[Age] = [10, 20]\n",
            "    ages[0] //= 5\n",
            "    ages[0] += 190\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "want exactly one fire for the 192 write: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// UNIT 3, site 4 (`walk_subscript_aug_assign`'s element Fire arm):
    /// the refused element write keeps `Age`'s own numeric-ground set,
    /// tagged with its sort — a later `ages[0]` read reaches
    /// `math.sqrt` (a sort-gated consumer, `sqrt_call_over_set`,
    /// math_models.rs) and derives a value instead of leaving the
    /// return undetermined.
    #[test]
    fn a_refused_element_writes_declared_set_reaches_sqrt_tagged() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "import math\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
            "def rows() -> Root:\n",
            "    ages: list[Age] = [10, 20]\n",
            "    ages[0] //= 5\n",
            "    ages[0] += 190\n",
            "    return math.sqrt(ages[0])\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            fires.len(),
            1,
            "the 192 write still fires once: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(
            blockers.is_empty(),
            "the tagged element set must let math.sqrt derive rather than blocking: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_list_element_compound_write_inside_the_declared_ceiling_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    ages: list[Age] = [10, 20]\n",
            "    ages[0] += 5\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "an in-range element write must never fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
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

    /// UNIT 1 (fully diagnosed, runtime-verified): a bare `return
    /// sum(<generator>)` with no assignment anywhere in the body used to
    /// fall through to the ordinary evaluator's `sum_call_over_star` row
    /// (`builtin_models.rs`), which needs a known-sign element hull and
    /// declines outright on a sign-straddling `[-1, 1]` element — even
    /// though the byte-identical computation, spelled `total =
    /// sum(...); return total`, already recognized and judged through
    /// `recognize_generator_sum`'s own Assign-only reader. Both spellings
    /// here run over `samples: list[Sample]` with `Sample`'s own hull
    /// straddling zero, and both must derive the SAME silent verdict
    /// against `-> Total` (`[-10, 10]`, the relational ledger's own tight
    /// total for up to 10 elements each in `[-1, 1]`, with no
    /// `min_length` stated so the count's own lower bound is 0) — never
    /// a blocker for one and a judged silence for the other.
    #[test]
    fn a_bare_return_of_a_generator_sum_judges_identically_to_its_assign_then_return_twin() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Sample = Annotated[float, Field(ge=-1.0, le=1.0)]\n",
            "type Total = Annotated[float, Field(ge=-10.0, le=10.0)]\n",
            "def bare_return(samples: Annotated[list[Sample], Field(max_length=10)]) -> Total:\n",
            "    return sum(s for s in samples)\n",
            "def assign_then_return(samples: Annotated[list[Sample], Field(max_length=10)]) -> Total:\n",
            "    total = sum(s for s in samples)\n",
            "    return total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "both spellings derive the same silent [-10, 10] verdict, want no findings: {:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
    }

    // --- yield/return inside a Generator[...]-annotated body ---

    /// i-more-expressions.py's own `yield_expression` shape:
    /// `Generator[Age, None, Age]` makes both a `yield 200` and a
    /// `return 200` checked positions — one fire each, an in-set
    /// `yield 40` stays silent.
    #[test]
    fn a_yield_and_a_return_out_of_the_declared_generator_set_each_fire() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Generator\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Generator[Age, None, Age]:\n",
            "    yield 40\n",
            "    yield 200\n",
            "    return 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 2, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
        assert!(fires[1].message.contains("'200'"), "{}", fires[1].message);
    }

    /// A non-generator body's `-> Age` never turns a `yield` inside a
    /// DIFFERENT, non-generator function into a checked position — this
    /// test pins that `yield_refinement` stays `None` outside a
    /// generator-shaped body by checking a plain `-> Age` function's own
    /// return still judges normally alongside an unrelated generator.
    #[test]
    fn a_bare_yield_judges_as_none_against_the_declared_yield_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Generator\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Generator[Age, None, Age]:\n",
            "    yield\n",
            "    return 40\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.to_lowercase().contains("none"), "{}", fires[0].message);
    }

    /// `yield from` delegating to a same-module generator whose own body
    /// yields an out-of-set value: the delegate's ACTUAL yields (read
    /// through `instances::generator_yields`, tighter than its own bare
    /// declared annotation) are what judge — `over_inner()`'s single
    /// `yield 200` fires against the outer `Age` set.
    #[test]
    fn a_yield_from_delegate_whose_own_body_yields_out_of_set_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Generator\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def over_inner() -> Generator[int, None, None]:\n",
            "    yield 200\n",
            "def f() -> Generator[Age, None, None]:\n",
            "    yield from over_inner()\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// A generator body's own IN-SET yields stay silent, including a
    /// `yield from` delegate whose actual yields all sit inside the
    /// outer set.
    #[test]
    fn a_generator_body_entirely_in_set_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Generator\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def inner() -> Generator[int, None, None]:\n",
            "    yield 40\n",
            "def f() -> Generator[Age, None, Age]:\n",
            "    yield 40\n",
            "    yield from inner()\n",
            "    return 40\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an entirely in-set generator body must stay silent: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// UNIT 3, site 2 (`delegated_generator_yields`'s declared-annotation
    /// fallback): a delegate whose body-walk `instances::generator_yields`
    /// permanently declines (a CONDITIONAL yield, `if flag: yield <expr>`
    /// — that function's own doc names this the deliberate boundary) falls
    /// to its bare `-> Generator[Age, None, None]` annotation instead.
    /// `Age`'s own set is numeric-ground, so the delegated value must
    /// carry `kind_tag: Some(Integer)` — the tag `min_max_scalar_operand`/
    /// `star_numeric_hull`/`sum_call_over_star` (builtin_models.rs) read,
    /// and the sort-gated consumers that used to refuse an untagged set.
    #[test]
    fn a_delegates_declared_yield_annotation_is_tagged_when_its_body_walk_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Generator\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def conditional_gen(flag: bool) -> Generator[Age, None, None]:\n",
            "    if flag:\n",
            "        yield 40\n",
        ));
        let def = module
            .body
            .iter()
            .find_map(|stmt| match stmt {
                Stmt::FunctionDef(def) if def.name.id.as_str() == "conditional_gen" => Some(def),
                _ => None,
            })
            .expect("the fixture's own def");
        // `instances::generator_yields` must genuinely decline this body
        // (the conditional yield) so the call below exercises the
        // fallback this test pins, not the body-walked route.
        assert!(
            instances::generator_yields(def, &[], None, &kernel, 0).is_none(),
            "a conditional yield must decline the body-walked route (this test's own premise)"
        );
        let aliases = compile_aliases(&module);
        let imports = surface_imports(&module);
        let functions = Arc::new(function_table(&module));
        let classes = Arc::new(class_table(&module, &aliases, &imports, &kernel));
        let context = WalkContext {
            aliases: &aliases,
            imports: &imports,
            kernel: &kernel,
            functions,
            classes,
            datetime_imports: Arc::new(crate::expressions::datetime_imports(&module)),
            module_bindings: HashMap::new(),
            module_callable_returns: Arc::new(HashMap::new()),
            strict_int_aliases: &HashSet::new(),
            typed_dicts: Arc::new(instances::typed_dict_table(&module, &aliases, &imports)),
            caller_arguments: Arc::new(HashMap::new()),
            entry_directory: None,
            evaluations_recorder: None,
        };
        let environment = Environment::new(HashSet::new());
        let delegate = ruff_python_parser::parse_expression("conditional_gen(True)")
            .expect("the delegate call parses")
            .into_syntax();
        let delegate = *delegate.body;
        let yields = delegated_generator_yields(&delegate, &context, &environment)
            .expect("the declared annotation fallback must still answer");
        let [value] = yields.as_slice() else {
            panic!("want exactly the one declared yield-type reading, got {}", yields.len());
        };
        assert_eq!(
            value.kind_tag,
            Some(PrimitiveKind::Integer),
            "Age's own numeric-ground set must tag the delegated value"
        );
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

    /// UNIT 3, site 3 (`judge_and_bind`'s Fire arm): the refused-but-
    /// declared slot `a` carries `Age`'s own numeric-ground set, tagged
    /// with its sort — so `math.sqrt(a)`, a sort-gated consumer
    /// (`sqrt_call_over_set`, math_models.rs) that refuses an untagged
    /// set, now derives a value instead of leaving the return
    /// undetermined.
    #[test]
    fn a_refused_writes_declared_set_reaches_sqrt_tagged() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "import math\n",
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
            "def f() -> Root:\n",
            "    a: Age\n",
            "    a = 200\n",
            "    return math.sqrt(a)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        // the assign's own refusal fires once; the sqrt-derived return
        // must stay SILENT (Root's own [0, 20] window covers sqrt(Age)'s
        // [0, sqrt(120)] range) rather than adding an undetermined
        // blocker for a return this fix now derives.
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(
            blockers.is_empty(),
            "the tagged slot must let math.sqrt derive rather than blocking: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
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
    fn a_declined_loop_forgets_a_receiver_only_ever_touched_through_a_chained_mutating_call() {
        let Some(kernel) = loaded_kernel() else { return };
        // `grouped` is never itself the target of `=` inside the loop body
        // — it is only read as the receiver of a CHAINED call
        // (`grouped.setdefault(...)` returns a value that `.append(...)`
        // is then called on). `run_expr_statement_once` (loops.rs) only
        // replays a mutating call whose receiver is a bare Name, so this
        // shape declines the whole loop. Before the fix, `grouped` was
        // never named by `collect_bound_names_stmt`'s scan (it is
        // MUTATED, never ASSIGNED), so the blocker path left it bound to
        // its stale pre-loop empty dict — and a post-loop
        // `grouped["young"]` read would then be a WRONG ANSWER: a
        // provable KeyError fire on a key the (unread) mutation actually
        // wrote (c-reads-and-values.py:1008). The fix forgets `grouped`
        // at the blocker, so the post-loop read is Undetermined, not a
        // false provable-raise fire.
        // `.extend` on the setdefault entry is OUTSIDE the executor's
        // recognized `.setdefault(...).append(...)` shape, so this loop
        // still declines — which is exactly what this test needs: the
        // forget rule at the blocker, not the served path.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    grouped: dict[str, list[int]] = {}\n",
            "    for age in [40, 200]:\n",
            "        grouped.setdefault(\"young\", []).extend([age])\n",
            "    check: Age = grouped[\"young\"][0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            blockers.len(),
            1,
            "the unmodeled for loop is this body's one blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let raises: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("KeyError"))
            .collect();
        assert!(
            raises.is_empty(),
            "grouped's stale pre-loop empty dict must not survive to falsely prove a KeyError: {:?}",
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

    /// `arm_terminates_or_provably_raises` treats a body whose last
    /// statement is NOT syntactically `return`/`raise`, but that the
    /// walk's own provable-raise machinery already fired an RTS7001 for,
    /// as terminating — the same as a bare `raise`. A plain `Assign` with
    /// no recorded fire must NOT be treated as terminating; only tacking
    /// a genuine RTS7001 finding, anchored inside that statement's own
    /// range, onto the body flips the answer.
    #[test]
    fn arm_terminates_or_provably_raises_treats_a_provable_raise_as_terminal() {
        let module = parsed(concat!(
            "def f() -> None:\n",
            "    a, b = (1, 2, 3)\n",
        ));
        let Stmt::FunctionDef(def) = &module.body[0] else { panic!("a function def") };
        let Stmt::Assign(assign) = &def.body[0] else { panic!("an assign") };
        let body = std::slice::from_ref(&def.body[0]);

        let no_findings: Vec<Finding> = Vec::new();
        assert!(
            !arm_terminates_or_provably_raises(body, &no_findings, 0),
            "a plain Assign with no recorded raise must not read as terminal"
        );

        let with_a_raise = vec![Finding {
            range: assign.value.range(),
            code: "RTS7001",
            message: "this expression provably raises ValueError: too many values to unpack (expected 2)".to_owned(),
        }];
        assert!(
            arm_terminates_or_provably_raises(body, &with_a_raise, 0),
            "an RTS7001 anchored inside the last statement's own range must count as terminal"
        );
    }

    /// A `try` whose body provably raises (an arity-mismatch unpack) and
    /// whose sole handler itself terminates (`except ValueError: return
    /// first`) leaves `walk_try`'s own `surviving` list empty — this
    /// statement never falls through. Statements written AFTER the try
    /// describe only unreachable code, so they must not be walked for
    /// judgement: a read of a name the try body never got to bind must
    /// not report an unreadable-value blocker, and the try body's own
    /// provable raise is still the sole finding.
    #[test]
    fn a_try_whose_every_arm_terminates_stops_the_body_walk_at_the_try() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    first = 40\n",
            "    triple = (200, 201, 202)\n",
            "    try:\n",
            "        over_first, over_second = triple\n",
            "    except ValueError:\n",
            "        return first\n",
            "    return over_first\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let raises: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("provably raises ValueError"))
            .collect();
        assert_eq!(
            raises.len(),
            1,
            "the try body's own arity-mismatch unpack must still fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "the unreachable `return over_first` past the terminating try must not report a blocker: {:?}",
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

    // --- JUDGED LOOP BODIES (loops.rs's declared-slot judging) ---

    #[test]
    fn a_declared_slot_write_inside_a_while_body_fires_with_no_post_loop_read() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:495's own row: the marker sits INSIDE the loop
        // body, with no post-loop declared read to catch it — the fire
        // must come from loops.rs's own judging, not check.rs's ordinary
        // sink path.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    age: Age = 0\n",
            "    while age < 3:\n",
            "        age = age + 121\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the +121 step leaving the set must fire from inside the loop body: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'121'"), "{}", fires[0].message);
    }

    #[test]
    fn a_declared_slot_write_from_a_dict_key_fires_instead_of_declining() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:508's own row: a String iterate written into a
        // declared Integer-sorted slot now fires through assignability::
        // judge rather than declining the whole loop.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    age: Age = 0\n",
            "    for key in {\"a\": 1, \"b\": 2}:\n",
            "        age = key\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "a string key into a declared int-sorted slot must fire, deduped once across both iterations: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "the loop must still run to completion — no blocker: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- LOOP ELSE + DEAD-ELSE LAW ---

    #[test]
    fn an_else_arm_write_fires_when_the_loop_never_breaks() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:446/472's own row: the else clause runs
        // (the loop never breaks), so its own out-of-set write fires —
        // check.rs walks orelse fully judged, not loops.rs.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    age: Age = 0\n",
            "    n = 0\n",
            "    while n < 3:\n",
            "        age = age + 1\n",
            "        n = n + 1\n",
            "    else:\n",
            "        age = 200\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the else arm's own write (200) must fire since the loop never breaks: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn an_else_arm_never_fires_its_own_write_when_the_loop_always_breaks() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:486's own row: the loop always breaks at i==1,
        // so the else clause never runs — its own out-of-set write
        // (200) must NOT fire; instead the dead-else law fires once,
        // naming why.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    age: Age = 0\n",
            "    for i in range(3):\n",
            "        if i == 1:\n",
            "            break\n",
            "        age = age + 1\n",
            "    else:\n",
            "        age = 200\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let two_hundred_fires: Vec<&Finding> =
            findings.iter().filter(|f| f.code == "RTS7001" && f.message.contains("'200'")).collect();
        assert!(
            two_hundred_fires.is_empty(),
            "the else arm's own write must never be walked when the loop always breaks: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let dead_else_fires: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("never runs"))
            .collect();
        assert_eq!(
            dead_else_fires.len(),
            1,
            "the dead-else law must fire exactly once naming why: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- EVALUATED ITERABLES ---

    #[test]
    fn a_tuple_element_that_evaluates_to_none_fires_into_a_non_optional_declared_slot() {
        let Some(kernel) = loaded_kernel() else { return };
        // a-statements.py:541's own row: `unread_number()`'s body falls
        // off its end with no return, so the call answers None —
        // iterable_values now evaluates a non-literal tuple element
        // rather than declining the whole loop for a syntactic miss.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def unread_number() -> int: ...\n",
            "def f() -> Age:\n",
            "    age: Age = 0\n",
            "    for item in (unread_number(),):\n",
            "        age = item\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "the tuple's evaluated element makes the loop concretely executable: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "None written into a non-Optional declared Age slot must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- MATCH JOIN FALLBACK ---

    #[test]
    fn a_class_pattern_as_capture_fires_inside_its_own_arm_on_an_undecidable_subject() {
        let Some(kernel) = loaded_kernel() else { return };
        // b-body-expressions.py:897-905's own row: `case int() as n:`
        // is a MatchClass wrapped in MatchAs — match_arms.rs cannot
        // decide TAKEN/NOT-TAKEN for a class pattern (Undecidable
        // regardless of the subject), so this fallback walks every arm
        // on a fork with `n` bound to the subject and fires from inside
        // the taken-in-practice arm.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    value = 200\n",
            "    match value:\n",
            "        case int() as n:\n",
            "            return n\n",
            "        case _:\n",
            "            return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a nameable class-pattern capture must not block the whole match: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the captured 200 must fire inside its own arm: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_class_pattern_as_capture_in_set_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        // b-body-expressions.py:886-894's own row: the in-set counterpart
        // — the same fallback must stay silent when the captured value
        // is inside the declared set.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    value = 40\n",
            "    match value:\n",
            "        case int() as n:\n",
            "            ok: Age = n\n",
            "            return ok\n",
            "        case _:\n",
            "            return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an in-set captured value must never fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_sequence_pattern_with_bare_name_elements_no_longer_blocks_the_whole_match() {
        let Some(kernel) = loaded_kernel() else { return };
        // `match_arms::pattern_bound_captures` names `a`/`b` positionally
        // (bare-Name elements over an UNKNOWN subject bind unknown(),
        // never a guess) — the match no longer needs its own blocker, and
        // an unreadable capture never fires (assignability's own law
        // never fires an Unknown value).
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(value) -> None:\n",
            "    match value:\n",
            "        case [a, b]:\n",
            "            pass\n",
            "        case _:\n",
            "            pass\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "a sequence pattern's own bare-Name captures are nameable now: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// t-match-patterns.py's own `match_sequence_out_of_set_element` shape:
    /// a KNOWN list literal subject lets `pattern_bound_captures` read the
    /// bound element's REAL value positionally (`x` binds to `items[0]`,
    /// 200) rather than `unknown()`, so the out-of-set read fires exactly
    /// where the fixture expects — at the return, not at the match.
    #[test]
    fn a_sequence_pattern_over_a_known_list_subject_binds_elements_positionally_and_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    match [200, 10]:\n",
            "        case [x, _y]:\n",
            "            return x\n",
            "        case _:\n",
            "            return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a sequence pattern's own bare-Name captures are nameable: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the bound element 200 must fire at the return: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// t-match-patterns.py's own `match_mapping_key_binding`/`match_
    /// mapping_literal_out_of_set` shapes: a mapping pattern's literal-key
    /// captures are nameable, and a known dict-literal subject lets
    /// `pattern_bound_captures` read the bound key's REAL value.
    #[test]
    fn a_mapping_pattern_over_a_known_dict_subject_binds_the_keyed_value_and_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    match {\"age\": 200}:\n",
            "        case {\"age\": bound_age}:\n",
            "            return bound_age\n",
            "        case _:\n",
            "            return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a mapping pattern's own literal-key captures are nameable: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the bound value 200 must fire at the return: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// t-match-patterns.py's own `match_class_out_of_set_attribute` shape:
    /// a class pattern's KEYWORD sub-pattern captures are nameable, and a
    /// known constructed-instance subject lets `pattern_bound_captures`
    /// read the bound field's REAL value via `instances::field_read`.
    #[test]
    fn a_class_pattern_keyword_subpattern_over_a_known_instance_binds_the_field_and_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import BaseModel, Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Point(BaseModel):\n",
            "    x: int\n",
            "    y: int\n",
            "def f() -> Age:\n",
            "    match Point(x=200, y=10):\n",
            "        case Point(x=px):\n",
            "            return px\n",
            "        case _:\n",
            "            return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "a class pattern's own keyword-subpattern captures are nameable: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the bound field 200 must fire at the return: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// t-match-patterns.py's own `match_class_positional_pattern` shape:
    /// POSITIONAL class-pattern sub-patterns still decline — resolving a
    /// position to a field name needs `__match_args__` order, which
    /// `match_arms::pattern_bound_captures` has no class table to read.
    #[test]
    fn a_class_pattern_with_positional_subpatterns_still_blocks_the_whole_match() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import BaseModel, Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Point(BaseModel):\n",
            "    x: int\n",
            "    y: int\n",
            "def f(shape: object) -> Age:\n",
            "    match shape:\n",
            "        case Point(px, _py):\n",
            "            return px\n",
            "        case _:\n",
            "            return 200\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            blockers.len(),
            1,
            "a positional class-pattern capture is unnameable without __match_args__ order: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- LAMBDA-ASSIGN LAW ---

    #[test]
    fn a_lambda_assigned_to_a_name_is_callable_through_that_name() {
        let Some(kernel) = loaded_kernel() else { return };
        // `f = lambda: 200` registers a synthetic def under `f`
        // (local_function_table) AND binds `f` to an opaque function
        // value; evaluate_call's gate dispatches through the function
        // table for a name bound only to an opaque function value, so
        // `f()` answers 200 end-to-end and the return sink fires.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def g() -> Age:\n",
            "    f = lambda: 200\n",
            "    return f()\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the lambda's 200 flows through f() into the return sink: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn local_function_table_registers_a_lambda_assign_as_a_callable_synthetic_def() {
        // Proves the LAMBDA-ASSIGN LAW's own infrastructure directly,
        // bypassing evaluate_call's environment-binding gate (the gap the
        // test above documents): the synthetic def IS correctly built and
        // IS answerable through summaries::call_result once looked up by
        // name — everything local_function_table itself is responsible
        // for.
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed("def g():\n    add_one = lambda x: x + 1\n    return 0\n");
        let Stmt::FunctionDef(g) = &module.body[0] else {
            panic!("module's one statement is def g")
        };
        let table = local_function_table(&g.body);
        let def = table.def("add_one").expect("the lambda-assign registers a synthetic def named add_one");
        assert_eq!(def.parameters.args.len(), 1, "the lambda's own parameter carries through");
        let result = crate::summaries::call_result(
            def,
            &[refined_domain::abstract_value::known_values(
                vec![120.0],
                refined_domain::abstract_value::PrimitiveKind::Integer,
                refined_domain::trust_grades::TrustProved,
            )],
            None,
            &kernel,
            0,
        )
        .expect("the synthetic def's body (return x + 1) answers through summaries::call_result");
        assert_eq!(result.values, vec![121.0]);
    }

    // --- STATEMENT-SIDE METHOD CALLS ---

    #[test]
    fn a_statement_side_method_call_writes_a_field_a_later_read_sees() {
        let Some(kernel) = loaded_kernel() else { return };
        // b-body-expressions.py:522-547's own row: `outlaw.spoil()` is a
        // bare Expr statement calling a method that writes `self.age =
        // 200` — the receiver must rebind, and the later `outlaw.age`
        // read must see 200, not the stale pre-call 40.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Outlaw:\n",
            "    def __init__(self) -> None:\n",
            "        self.age = 40\n",
            "    def spoil(self) -> None:\n",
            "        self.age = 200\n",
            "def f() -> Age:\n",
            "    outlaw = Outlaw()\n",
            "    outlaw.spoil()\n",
            "    return outlaw.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the method's own write (200) must be visible at outlaw.age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_statement_side_method_call_that_leaves_the_field_in_set_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Person:\n",
            "    def __init__(self) -> None:\n",
            "        self.age = 40\n",
            "    def bump(self) -> None:\n",
            "        self.age = self.age + 1\n",
            "def f() -> Age:\n",
            "    person = Person()\n",
            "    person.bump()\n",
            "    return person.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an in-set write through a statement-side method call must never fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_write_then_read_through_a_declared_sink_uses_the_method_s_own_return_value() {
        let Some(kernel) = loaded_kernel() else { return };
        // sink_value's own method-call channel: an AnnAssign RHS that is
        // a statement-side method call judges the method's OWN return
        // value, not a plain evaluate_expression reading of the call.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Counter:\n",
            "    def __init__(self) -> None:\n",
            "        self.value = 199\n",
            "    def increment(self) -> int:\n",
            "        self.value = self.value + 1\n",
            "        return self.value\n",
            "def f() -> None:\n",
            "    c = Counter()\n",
            "    over: Age = c.increment()\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the method's own returned value (200) must judge at the declared sink: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- NAMED-RECEIVER FIELD WRITE (write_named_field, e:357/q:203) ---

    #[test]
    fn a_property_setter_write_through_a_local_variable_receiver_fires_and_the_return_fires_too() {
        let Some(kernel) = loaded_kernel() else { return };
        // e-class-and-function.py's `property_getter_setter`: `over_box.age
        // = 200` writes through a `@property` setter on a LOCAL variable
        // receiver, never `self` — before write_named_field generalized
        // write_self_field's own judged-and-rebound law past the literal
        // name `self`, this row's write silently forgot `over_box` instead
        // of judging the setter's own declared refinement.
        //
        // TWO fires are correct here, not one: the write's own value
        // range (`write_named_field`'s judgment, `bind_or_forget_target`'s
        // named-receiver branch) is the FIRST verdict, and `f`'s own
        // `-> Age` return sink (`walk_return`) is a SECOND, independent
        // verdict — the rebind the sibling test at :8406
        // (`a_property_setter_write_lands_on_the_backing_field_a_later_
        // getter_read_sees`) pins as sound is exactly what carries this
        // same out-of-set 200 forward to `return over_box.age`, so the
        // return sink judges the identical bad value against the
        // identical `Age` set. Both messages read as the same text
        // (`judge`'s message carries no location), so the ranges — not
        // the text — are what prove these are two determined verdicts
        // over two statements, not one write double-counted.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Aged:\n",
            "    def __init__(self) -> None:\n",
            "        self._held = 40\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self._held\n",
            "    @age.setter\n",
            "    def age(self, value: Age) -> None:\n",
            "        self._held = value\n",
            "def f() -> Age:\n",
            "    over_box = Aged()\n",
            "    over_box.age = 200\n",
            "    return over_box.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            2,
            "the setter's own write (200) must fire against its declared Age refinement, AND \
             the rebind it leaves behind must carry 200 to f's own -> Age return sink, which \
             fires independently: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
        assert!(fires[1].message.contains("'200'"), "{}", fires[1].message);
        assert_ne!(
            fires[0].range.start(),
            fires[1].range.start(),
            "two DIFFERENT sinks must have judged this value — the write's own literal and the \
             return expression — never the same range firing twice: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// A DISCRIMINATING test for the write TARGET, not just the write's
    /// own judgment: the backing field starts at an OUT-OF-SET value
    /// (200), and the setter then writes an IN-SET value (40) through
    /// the property. `write_named_field` writes the property name
    /// itself (`age`) into the instance's keys, but `field_read_through_
    /// model` reads a property through its OWN `backing` name (`_held`)
    /// — a write that lands on `age` instead of `_held` never reaches
    /// the getter's own read at all, so the later `over_box.age` read
    /// would still see the STALE 200 backing value and wrongly fire.
    /// The write must resolve to the SAME backing name the read
    /// resolves to, so this call stays completely silent.
    #[test]
    fn a_property_setter_write_lands_on_the_backing_field_a_later_getter_read_sees() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Aged:\n",
            "    def __init__(self) -> None:\n",
            "        self._held = 200\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self._held\n",
            "    @age.setter\n",
            "    def age(self, value: Age) -> None:\n",
            "        self._held = value\n",
            "def f() -> Age:\n",
            "    box = Aged()\n",
            "    box.age = 40\n",
            "    return box.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "the setter's in-set write (40) must land on the SAME backing field the getter reads, \
             leaving no stale out-of-set value behind: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_plain_field_write_through_a_local_variable_receiver_rebinds_and_a_later_read_sees_it() {
        let Some(kernel) = loaded_kernel() else { return };
        // q-decline-names.py's `setter_effect_read_through_getter`: the
        // same named-receiver write law, over an UNREFINED field (no Fire
        // expected) — pins that write_named_field still rebinds (a later
        // getter read must see the write) even with no declared refinement
        // to judge against.
        let module = parsed(concat!(
            "class AgeBox:\n",
            "    def __init__(self) -> None:\n",
            "        self._age = 10\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self._age\n",
            "    @age.setter\n",
            "    def age(self, value: int) -> None:\n",
            "        self._age = value\n",
            "def f() -> int:\n",
            "    box = AgeBox()\n",
            "    box.age = 40\n",
            "    return box.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an unrefined field write through a local variable receiver must never fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- CLASS-OBJECT ATTRIBUTE STATE (class_object_value, e:485) ---

    #[test]
    fn a_class_object_attribute_write_and_read_composes_with_no_instance_involved() {
        let Some(kernel) = loaded_kernel() else { return };
        // e-class-and-function.py's `class_attribute_write`: `Counted.total
        // = 200` writes through the CLASS ITSELF (no `Counted(...)`
        // construction anywhere on this row), and the later `Counted.total`
        // read must see the write. Before class_object_value seeded the
        // class's own bare name as a tagged Kind::Object, `Counted` read as
        // unknown() and the write silently forgot it.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Counted:\n",
            "    total = 0\n",
            "def f() -> Age:\n",
            "    Counted.total = 200\n",
            "    return Counted.total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the class-object write (200) must be visible at the later Counted.total read: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_class_object_attribute_write_in_range_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Counted:\n",
            "    total = 0\n",
            "def f() -> Age:\n",
            "    Counted.total = 40\n",
            "    return Counted.total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an in-range class-object write must never fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- AugAssign ON A NON-NAME TARGET (walk_field_aug_assign /
    //     walk_subscript_aug_assign, i:233/246/273) ---

    #[test]
    fn a_property_accessor_compound_write_fires_against_the_setters_own_refinement() {
        let Some(kernel) = loaded_kernel() else { return };
        // i-more-expressions.py's `accessor_compound_read_modify_write`:
        // `over_box.age += 195` (10 + 195 = 205) must fire against the
        // setter's own Age refinement, the same fire a hand-split
        // `over_box.age = over_box.age + 195` would give.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class AccessorBox:\n",
            "    def __init__(self) -> None:\n",
            "        self.held = 10\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self.held\n",
            "    @age.setter\n",
            "    def age(self, value: Age) -> None:\n",
            "        self.held = value\n",
            "def f() -> int:\n",
            "    over_box = AccessorBox()\n",
            "    over_box.age += 195\n",
            "    return over_box.held\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the compound write's own folded value (205) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'205'"), "{}", fires[0].message);
    }

    #[test]
    fn a_property_accessor_compound_write_in_range_stays_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class AccessorBox:\n",
            "    def __init__(self) -> None:\n",
            "        self.held = 10\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self.held\n",
            "    @age.setter\n",
            "    def age(self, value: Age) -> None:\n",
            "        self.held = value\n",
            "def f() -> int:\n",
            "    box = AccessorBox()\n",
            "    box.age += 5\n",
            "    return box.held\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an in-range accessor compound write must never fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_subscript_compound_write_composes_and_a_later_read_sees_the_mutated_element() {
        let Some(kernel) = loaded_kernel() else { return };
        // i-more-expressions.py's `compound_array_index_operators`:
        // `ages[0] += 5` must compose (read the element, fold, write back)
        // so a LATER `ages[0]` read sees 15, not the stale pre-write 10 —
        // walk_subscript_aug_assign's own no-element-judging contract still
        // requires the composition itself to be sound.
        let module = parsed(concat!(
            "def f() -> int:\n",
            "    ages = [10, 20]\n",
            "    ages[0] += 5\n",
            "    return ages[0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "walk_subscript_aug_assign never fires (no declared element set to judge against): {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- del d[k] REBIND/FORGET ---

    #[test]
    fn del_subscript_on_a_known_dict_rebinds_and_a_later_read_answers_undetermined() {
        let Some(kernel) = loaded_kernel() else { return };
        // b-body-expressions.py:660-665's own row: `del person["age"]`
        // removes the key from a KNOWN dict; a later `.get("age")` read
        // then answers None (an absent key) rather than the stale
        // pre-delete value — this pins the REBIND half (dict_without_item
        // answers Some), not the specific None-vs-Undetermined judgment
        // downstream, which is dict_get_result's own contract.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> None:\n",
            "    person: dict[str, int] = {\"age\": 40}\n",
            "    del person[\"age\"]\n",
            "    check = person.get(\"age\", 0)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "no fire is expected in this row on its own: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn del_subscript_on_an_unknown_receiver_forgets_it() {
        let Some(kernel) = loaded_kernel() else { return };
        // an unresolved key/receiver shape must FORGET the receiver
        // (Undetermined downstream), never leave the stale pre-delete
        // value standing.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f(key: str) -> None:\n",
            "    person: dict[str, int] = {\"age\": 200}\n",
            "    del person[key]\n",
            "    over: Age = person[\"age\"]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "an unresolved delete key must forget the receiver — the stale 200 must not survive to fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- RETURN-THROUGH-LOOP CHANNEL ---

    #[test]
    fn a_return_inside_a_for_loop_body_fires_at_the_carried_range() {
        let Some(kernel) = loaded_kernel() else { return };
        // c-reads-and-values.py:927/928's own shape: `for age in
        // overs.values(): return age` — every iterate is known, and the
        // loop's own answer must carry the returned value out so
        // walk_loop can judge it against -> Age, exactly as walk_return
        // would for a straight-line return.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    overs = {\"bea\": 200}\n",
            "    for age in overs.values():\n",
            "        return age\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "the loop must still run concretely — the return channel must not decline it: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the returned 200 must fire against the declared -> Age return: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    #[test]
    fn a_conditional_return_inside_a_loop_joins_the_return_path_with_the_normal_completion() {
        let Some(kernel) = loaded_kernel() else { return };
        // the return sits under an `if` that only SOME iterations take
        // (age == 200 never occurs here, so the loop actually completes
        // normally on every iteration and the return path never fires) —
        // this pins that the join keeps the NORMAL completion path alive
        // and does not wrongly treat "a return exists somewhere in the
        // body" as "every path returns."
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    total: Age = 0\n",
            "    for age in [10, 20]:\n",
            "        if age == 999:\n",
            "            return age\n",
            "        total = total + age\n",
            "    return total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "every iterate stays in range on both the conditional-return and the normal-completion path: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_conditional_return_inside_a_loop_that_does_fire_judges_at_the_carried_range() {
        let Some(kernel) = loaded_kernel() else { return };
        // the SAME conditional shape, but the guarded return DOES trigger
        // on one iteration — the returned value must still fire.
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    for age in [10, 200]:\n",
            "        if age > 100:\n",
            "            return age\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the conditional return's own out-of-set value (200) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- BODY-LOCAL CLASS TABLES ---

    /// A class defined INSIDE a function body (b-body-expressions.py's
    /// `new_resolvable` shape): `class_table`'s own module-level scan
    /// never sees it, so before this fix a body-local construction
    /// stayed `unknown()` and the fire never landed. `merged_classes_for_body`
    /// now merges this body's own top-level classes over `context.classes`,
    /// so `Person(200)`'s field carries the summary into the return sink.
    #[test]
    fn a_class_defined_inside_a_function_body_still_judges_its_construction() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def f() -> Age:\n",
            "    class Person:\n",
            "        def __init__(self, age: int) -> None:\n",
            "            self.age = age\n",
            "    ok = Person(40)\n",
            "    good: Age = ok.age\n",
            "    over = Person(200)\n",
            "    return over.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the body-local class's own out-of-set construction (200) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// c-reads-and-values.py's own `read_one_field`/`read_nested_path`
    /// collision: two DIFFERENT functions each declare their own body-local
    /// class named `Person`, with different fields. Both classes collide
    /// under the one shared bare name in `context.classes`
    /// (`findings_for_module_with_resolver`'s own module-wide scan), so
    /// `construction_call_verdict` must read `environment.classes()` (the
    /// per-body table `merged_classes_for_body` built for THIS body) rather
    /// than `context.classes` alone — otherwise `Person(age=40)` matches
    /// whichever class happened to overwrite the shared entry, not the
    /// caller's own local `Person`.
    #[test]
    fn a_body_local_class_construction_uses_its_own_bodys_class_not_a_same_named_sibling() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import BaseModel, Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def read_one_field() -> Age:\n",
            "    class Person(BaseModel):\n",
            "        age: int\n",
            "    over = Person(age=200)\n",
            "    return over.age\n",
            "def other_function_with_same_named_class() -> None:\n",
            "    class Person(BaseModel):\n",
            "        name: str\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            blockers.is_empty(),
            "the same-named sibling class must not shadow this body's own Person: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "read_one_field's own Person(age=200) must fire through its own body-local class: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- SELF-SEEDING ---

    /// `self.age` read inside a method body, with NO call site anywhere
    /// in the module (b-body-expressions.py's `self_field_read`/
    /// `OverPerson` shape) — before this fix, `self` was never bound
    /// during the STATEMENT WALK of a method body (only `method_
    /// call_result`'s separate call-site interpreter seeded it), so this
    /// read answered `Unknown` and stayed silent. `walk_method_def` now
    /// seeds `self` from the class's own declared/default field shape at
    /// the method body's own entry, so the literal self-write inside
    /// `__init__` (captured as the field's DEFAULT, `class_table`'s own
    /// literal-self-write rule) carries into `years`'s own `self.age`
    /// read and judges against the method's `-> Age` annotation.
    #[test]
    fn a_self_field_read_inside_a_method_body_judges_with_no_call_site() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class OverPerson:\n",
            "    def __init__(self) -> None:\n",
            "        self.age = 200\n",
            "    def years(self) -> Age:\n",
            "        return self.age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "self.age's own out-of-set default (200) must fire at the method's own return: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// A bare `self` reference judges too (b-body-expressions.py's
    /// `ThisBare` shape): an Object value against a scalar-ground
    /// declared set is `assignability.rs`'s own "Object/List/Null vs
    /// scalar-ground → Fire" law — reachable only once `self` is bound
    /// to something at all.
    #[test]
    fn a_bare_self_reference_fires_against_a_scalar_ground_return_annotation() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Bare:\n",
            "    def years(self) -> Age:\n",
            "        return self\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "a bare self reference is not a refined Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    // --- setdefault_append (dict_groupby's chained mutation) ---

    #[test]
    fn setdefault_append_extends_a_present_key_and_writes_a_new_one() {
        use refined_domain::abstract_value::{known_values, PrimitiveKind};
        use refined_domain::trust_grades::TrustProved;
        fn integer(v: f64) -> AbstractValue {
            known_values(vec![v], PrimitiveKind::Integer, TrustProved)
        }
        fn string(text: &str) -> AbstractValue {
            let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
            known_values(code_points, PrimitiveKind::String, TrustProved)
        }
        let grouped = crate::collection_models::dict_literal_value(
            &[Some(crate::collection_models::DictKey::string("young"))],
            &[list_literal_value(&[integer(40.0)])],
        );
        // "young" is present: setdefault_append reads its existing list
        // and appends onto it, rather than replacing with the default.
        let after_young = setdefault_append(&grouped, &string("young"), &list_literal_value(&[]), &integer(41.0))
            .expect("appending onto a present key's list must decide");
        assert_eq!(
            crate::collection_models::subscript_read(&after_young, &string("young")),
            Some(list_literal_value(&[integer(40.0), integer(41.0)]))
        );
        // "old" is absent: setdefault_append inserts the default list,
        // then appends onto that fresh list — the exact
        // `grouped.setdefault("old", []).append(200)` shape.
        let after_old = setdefault_append(&after_young, &string("old"), &list_literal_value(&[]), &integer(200.0))
            .expect("appending onto a fresh default list must decide");
        assert_eq!(
            crate::collection_models::subscript_read(&after_old, &string("old")),
            Some(list_literal_value(&[integer(200.0)]))
        );
    }

    // --- Literal[...] int-only inline recognition (typereading.rs) ---

    #[test]
    fn an_int_literal_alias_and_an_inline_literal_annotation_both_judge() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Literal\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def rows() -> None:\n",
            "    small: Literal[10, 20] = 10\n",
            "    good: Age = small\n",
            "    big: Literal[200, 201] = 200\n",
            "    over: Age = big\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "only the Literal[200, 201]-typed `big` read is out of Age's [0, 120] window: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    // --- callable-variable calls (typereading.rs::callable_return_refinement,
    // env.rs::callable_returns, check.rs::callable_variable_call_result) ---

    /// The smallest DIRECT-sink shape: `x: Age = maybe_next_year(40)` puts
    /// the call straight into `sink_value`'s own value expression (no
    /// ternary in between) — `maybe_next_year`'s bare `int` return sort
    /// (`Callable[[int], int]`, no refined alias) is the unbounded
    /// whole-number ray, which is NOT a subset of Age's `[0, 120]`
    /// window, so the containment law fires.
    #[test]
    fn a_direct_callable_variable_call_sink_fires_against_a_declared_alias() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "maybe_next_year: Callable[[int], int] | None = None\n",
            "def rows() -> None:\n",
            "    over: Age = maybe_next_year(40)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the callable's own unrefined int return admits values outside Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// A callable variable whose declared return IS a refined alias
    /// (`Callable[[int], Age]`) reads Age's own set at the call site —
    /// an in-window argument-independent call is silent, since this
    /// channel judges the RETURN refinement, never the call's own
    /// arguments.
    #[test]
    fn a_direct_callable_variable_call_sink_is_silent_when_the_return_is_already_the_declared_alias() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "next_year: Callable[[int], Age] | None = None\n",
            "def rows() -> None:\n",
            "    fine: Age = next_year(40)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "Callable[[int], Age]'s own return is already Age-refined: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// UNIT 3, site 5 (`callable_variable_call_result`): `Age`'s own
    /// declared set is numeric-ground, so the callable's return must
    /// carry `kind_tag: Some(Integer)` once bound to `year` — piping
    /// `year` through `math.sqrt` (a sort-gated consumer,
    /// `sqrt_call_over_set`, math_models.rs) derives a value instead of
    /// leaving the return undetermined. The call sits at its own DIRECT
    /// sink (`year: Age = next_year(40)`) — `callable_variable_call_
    /// result` only reads a call at `sink_value`'s own value-expression
    /// position (this file's own doc: "5. The CALLEE-EFFECTS CHANNEL"),
    /// never a call nested as another call's argument.
    #[test]
    fn a_callable_variable_call_results_declared_set_reaches_sqrt_tagged() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "import math\n",
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
            "next_year: Callable[[int], Age] | None = None\n",
            "def f() -> Root:\n",
            "    year: Age = next_year(40)\n",
            "    return math.sqrt(year)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "the tagged callable return must let math.sqrt derive rather than blocking: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// b-body-expressions.py:38/79's own shape verbatim, EXCEPT the call
    /// sits at a DIRECT sink (no ternary): `maybe_next_year(40)` read
    /// straight into a `return -> Age`. This is the shape this unit's
    /// `sink_value` channel reaches; the fixture row's own
    /// `maybe_next_year(40) if maybe_next_year is not None else 0` ternary
    /// wrapping is a DIFFERENT shape this channel does not reach — see
    /// this unit's report (the call there is evaluated inside
    /// `evaluate_ternary`'s `evaluate_expression`/`evaluate_call`
    /// recursion in expressions.rs, never through `sink_value`).
    #[test]
    fn the_b74_shape_without_its_ternary_wrapper_fires_at_a_return_sink() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "maybe_next_year: Callable[[int], int] | None = None\n",
            "def call_direct() -> Age:\n",
            "    return maybe_next_year(40)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the guarded call's own unrefined int return admits values outside Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// A resolvable same-module `def` of the same name wins over the
    /// callable-returns table — the ordinary `summaries::call_result`
    /// path (which reads the def's ACTUAL body) owns a name that
    /// resolves to a real def, never this fallback.
    #[test]
    fn a_name_resolving_to_a_same_module_def_is_not_read_as_a_callable_variable() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "greet: Callable[[int], int] | None = None\n",
            "def greet(x: int) -> int:\n",
            "    return 40\n",
            "def rows() -> None:\n",
            "    fine: Age = greet(1)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.iter().all(|f| f.code != "RTS7001"),
            "the same-module def `greet` (always returns 40, in-window) must win over the callable-returns fallback: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// b-body-expressions.py:76-79's own shape verbatim: the callable
    /// call sits inside a ternary's `body` arm
    /// (`maybe_next_year(40) if maybe_next_year is not None else 0`),
    /// which `evaluate_ternary` (expressions.rs) evaluates through plain
    /// `evaluate_expression`/`evaluate_call` recursion, never through
    /// `sink_value` — the gap
    /// `the_b74_shape_without_its_ternary_wrapper_fires_at_a_return_sink`
    /// documents as this channel's own remaining shape. This test proves
    /// `evaluate_call`'s own callable-variable-call arm (added alongside
    /// this test) closes it: the ternary's test
    /// (`maybe_next_year is not None`) is not provably decided from a
    /// bare module-level `Callable | None` binding, so both arms
    /// evaluate and `join_known` joins the call's own `known_set`
    /// (`R`'s unbounded whole-number ray, TrustSpec) with the literal
    /// `0` (Kind::Values, Integer) — the untagged-Set-vs-Values join
    /// falls to `join_known`'s bottom numeric-set path (`is_numeric_kind`
    /// admits any non-Values kind, so `Kind::Set` always qualifies) and
    /// answers the union of the two sides' own sets, still admitting
    /// values Age's `[0, 120]` window does not, so the containment law
    /// fires.
    #[test]
    fn the_ternary_wrapped_b79_shape_fires_through_join_known() {
        let Some(kernel) = loaded_kernel() else { return };
        // the VALUELESS module AnnAssign is the faithful twin of TS
        // `declare const maybeNextYear: ... | undefined` — a concrete
        // `= None` initializer would make the guard provably false and
        // the silent answer honest, which is a different row entirely
        let module = parsed(concat!(
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "maybe_next_year: Callable[[int], int] | None\n",
            "def call_optional() -> Age:\n",
            "    return maybe_next_year(40) if maybe_next_year is not None else 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the guarded call still admits a whole number outside the set: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// A callable-variable call reached ONLY through `evaluate_call`
    /// (expressions.rs), never through `sink_value`'s own
    /// `callable_variable_call_result` — `walk_assign`'s value routes
    /// through `sink_value` first (which already answers a bare
    /// `over = maybe_next_year(40)` assignment before `evaluate_call` is
    /// ever reached), so this test nests the call one level deeper, as
    /// the single element of a list display read back by index:
    /// `[maybe_next_year(40)][0]`. `sink_value` reads the WHOLE
    /// subscript expression (not a bare Call node) and declines, falling
    /// through to `evaluate_expression`'s list-display and subscript
    /// arms, which recurse into `evaluate_call` for the display's own
    /// element — the one path this unit's arm, and only this unit's
    /// arm, answers.
    #[test]
    fn a_callable_variable_call_nested_inside_a_list_display_fires_via_evaluate_call() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Callable\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "maybe_next_year: Callable[[int], int] | None = None\n",
            "def call_nested_in_list_display() -> Age:\n",
            "    return [maybe_next_year(40)][0]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the callable's own unrefined int return, read back through the display, still admits values outside Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// a-statements.py's own `with_statement`/`device()` shape: `device()`
    /// is a MODULE-LEVEL `def` whose body declares a LOCAL class
    /// (`_Device`) and returns its construction — `with device() as
    /// handle:` never walks `device`'s body directly (`check.rs` only
    /// EVALUATES the context expression as a value), so the instance
    /// `summaries::call_result_with_enclosing` tags `source = "_Device"`
    /// must be resolvable through `context.classes`, the ONLY table
    /// `enter_method_result` consults — this pins the module-level-def
    /// local-class registration this unit added in
    /// `findings_for_module_with_resolver` (the loop scanning every
    /// top-level `def`'s own body via `local_class_table`). Without it,
    /// `enter_method_result` declines (`context.classes.get("_Device")`
    /// answers `None`), `handle` is forgotten, and `handle.value` never
    /// fires — the ONE fire this test asserts.
    #[test]
    fn with_statement_over_a_same_module_def_returning_a_local_class_instance_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def unread_number() -> int:\n",
            "    raise NotImplementedError\n",
            "def device():\n",
            "    class _Device:\n",
            "        value: int = 0\n",
            "        def __enter__(self):\n",
            "            self.value = unread_number()\n",
            "            return self\n",
            "        def __exit__(self, *exc_info):\n",
            "            return False\n",
            "    return _Device()\n",
            "def with_statement() -> Age:\n",
            "    with device() as handle:\n",
            "        return handle.value\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the __enter__-assigned opaque int admits values outside Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// a-statements.py's own `async_with_statement`/`AsyncDevice` shape:
    /// the class is declared DIRECTLY inside the `async with` statement's
    /// own enclosing function (a body-local class, already reachable
    /// through `local_class_table`/`merged_classes_for_body` — no
    /// same-module-def indirection the way `device()`/`with_statement`
    /// needs), and its `__aenter__` (not `__enter__`) is what
    /// `enter_method_result` must dispatch to for `with_stmt.is_async`.
    /// Proof the `__aenter__` half of that dispatch fires exactly like
    /// the sync `__enter__` half already does.
    #[test]
    fn async_with_statement_over_a_body_local_class_dispatches_aenter_and_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def unread_number() -> int:\n",
            "    raise NotImplementedError\n",
            "async def async_with_statement() -> Age:\n",
            "    class AsyncDevice:\n",
            "        value: int = 0\n",
            "        async def __aenter__(self):\n",
            "            self.value = unread_number()\n",
            "            return self\n",
            "        async def __aexit__(self, *exc_info):\n",
            "            return False\n",
            "    async with AsyncDevice() as handle:\n",
            "        return handle.value\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the __aenter__-assigned opaque int admits values outside Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// a-statements.py's own `nonlocal_rebind` shape end-to-end: `bump()`
    /// rebinds the enclosing `age` in-set (silent), `spoil()` rebinds it
    /// out-of-set (fires) — proof the CALLEE-EFFECTS CHANNEL
    /// (`apply_call_effects`) is wired into the ordinary statement walk,
    /// not merely unit-tested against `summaries::call_effects` in
    /// isolation.
    #[test]
    fn nonlocal_rebind_fires_once_at_the_out_of_set_call_site() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def nonlocal_rebind() -> Age:\n",
            "    age: Age = 10\n",
            "    def bump() -> None:\n",
            "        nonlocal age\n",
            "        age = 15\n",
            "    bump()\n",
            "    def spoil() -> None:\n",
            "        nonlocal age\n",
            "        age = 200\n",
            "    spoil()\n",
            "    return age\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "bump()'s in-set rebind must stay silent; only spoil()'s 200 fires: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// a-statements.py's own `closure_mutates_flattened_capture` shape
    /// end-to-end: `spoil()` mutates a captured dict through a subscript
    /// store with no `nonlocal` declaration at all, and the LATER read
    /// `outlaw["age"]` (never inside `spoil` itself) is what fires —
    /// proof the effect survives back into the caller's own environment
    /// and is read at a plain dict-subscript sink.
    #[test]
    fn closure_mutates_flattened_capture_fires_at_the_later_read() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def closure_mutates_flattened_capture() -> Age:\n",
            "    outlaw = {\"age\": 40}\n",
            "    def spoil() -> None:\n",
            "        outlaw[\"age\"] = 200\n",
            "    spoil()\n",
            "    return outlaw[\"age\"]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the closure's subscript mutation must carry 200 into the later read: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// a-statements.py's own `async_for_over_stream` shape end-to-end:
    /// `stream() -> AsyncIterator[int]` declines concretely (`raise
    /// NotImplementedError`), so the loop only runs through the ABSTRACT
    /// SORT-ELEMENT PASS (`loops::abstract_element_sort_pass`) — proof
    /// the pass is wired into the ordinary loop walk (`walk_loop`), not
    /// merely unit-tested against `loop_final_environment` directly.
    #[test]
    fn async_for_over_stream_fires_through_the_abstract_element_sort_pass() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, AsyncIterator\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "async def stream() -> AsyncIterator[int]:\n",
            "    raise NotImplementedError\n",
            "    yield 0\n",
            "async def async_for_over_stream() -> Age:\n",
            "    age: Age = 0\n",
            "    async for chunk in stream():\n",
            "        age = chunk\n",
            "    return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the whole-int element sort admits values outside Age: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
    }

    /// f-type-nodes.py's own `optional_annotation` shape: `present:
    /// Optional[Age] = 40` then `if present is None:` — `present`'s
    /// concrete value (40) makes the `is None` test provably false, but
    /// `present`'s DECLARED shape admits `None` (`Optional[Age]`), so this
    /// is the ordinary Optional-peeling idiom, never dead code. The
    /// DEAD-BRANCH LAW must not fire RTS7001 here, and the walk must still
    /// reach the later `good: Age = present` read (which stays silent —
    /// 40 is in Age's [0, 120] window).
    #[test]
    fn an_is_none_peel_on_an_admits_none_declared_name_never_fires_the_dead_branch_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Optional\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def optional_annotation() -> Age:\n",
            "    present: Optional[Age] = 40\n",
            "    if present is None:\n",
            "        return 0\n",
            "    good: Age = present\n",
            "    return good\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an Optional-peel test must never fire the dead-branch law, and the in-set \
             read after it must stay silent too: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// The mirror: `Age | None` (the pipe-union spelling of `Optional`)
    /// peeled the same way — `is_admits_none_peel_test` must recognize
    /// both annotation spellings identically, since `typereading::
    /// declared_refinement` reads them to the same `admits_none: true`
    /// shape.
    #[test]
    fn an_is_none_peel_on_a_pipe_none_declared_name_never_fires_the_dead_branch_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def pipe_none_annotation() -> Age:\n",
            "    present: Age | None = 40\n",
            "    if present is None:\n",
            "        return 0\n",
            "    good: Age = present\n",
            "    return good\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "an `Age | None` peel test must never fire the dead-branch law: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// f-type-nodes.py's own `optional_annotation`/`pipe_none_annotation`
    /// SECOND row (`over: Optional[int] = 200`, `if over is None:`): a
    /// bare base-sort wrapped in `Optional`/`| None`, with NO alias
    /// involved at all — `optional_base_sort_annotation`'s own row,
    /// distinct from the `Optional[Age]`/`Age | None` alias shape the two
    /// tests above cover. The dead-branch law must not fire on the peel
    /// test, and the later `return over` must still fire on 200 once
    /// unwrapped — the peel exception silences ONLY the `is None` dead-
    /// branch fire, never the real out-of-set return.
    #[test]
    fn an_is_none_peel_on_a_bare_optional_int_declared_name_never_fires_the_dead_branch_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Optional\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def optional_annotation() -> Age:\n",
            "    over: Optional[int] = 200\n",
            "    if over is None:\n",
            "        return 0\n",
            "    return over\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let dead_branch_fires: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
            .collect();
        assert!(
            dead_branch_fires.is_empty(),
            "a bare Optional[int] peel test must never fire the dead-branch law: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "200 must still fire at the return once unwrapped from Optional: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// The pipe-union mirror: `over: int | None = 200`.
    #[test]
    fn an_is_none_peel_on_a_bare_pipe_none_int_declared_name_never_fires_the_dead_branch_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def pipe_none_annotation() -> Age:\n",
            "    over: int | None = 200\n",
            "    if over is None:\n",
            "        return 0\n",
            "    return over\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let dead_branch_fires: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
            .collect();
        assert!(
            dead_branch_fires.is_empty(),
            "a bare `int | None` peel test must never fire the dead-branch law: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "200 must still fire at the return once unwrapped from the union: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// The exception's own boundary: a-statements.py's own
    /// `none_test_on_helper_that_never_answers_none` shape — `held` is
    /// bound by a plain `Assign` from a call result (never an
    /// `AnnAssign`), so it carries no entry in `aug_assign_refinements` at
    /// all. `is_admits_none_peel_test` must find nothing and the
    /// dead-branch law must still fire here, exactly as before the
    /// exception existed — the exception is scoped to a DECLARED
    /// `admits_none` name, never to every `is None` test whose value
    /// happens to be provably non-null.
    #[test]
    fn an_is_none_test_on_a_plain_assign_target_still_fires_the_dead_branch_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def helper_never_answers_none() -> dict[str, int]:\n",
            "    if True:\n",
            "        return {\"age\": 40}\n",
            "    return {\"age\": 10}\n",
            "def none_test_on_helper_that_never_answers_none() -> Age:\n",
            "    held = helper_never_answers_none()\n",
            "    if held is None:\n",
            "        return 0\n",
            "    return held[\"age\"]\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let dead_branch_fires: Vec<&Finding> = findings
            .iter()
            .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
            .collect();
        assert_eq!(
            dead_branch_fires.len(),
            1,
            "a plain-Assign target carries no aug_assign_refinements entry, so the \
             exception must not suppress this row's own dead-branch fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// e-class-and-function.py's own `first_age`/`rest_parameter` shape
    /// end to end: `*ages: int` genuinely binds a known tuple of the
    /// caller's trailing arguments (`summaries::bind_parameters`'s own
    /// vararg row), so an IN-SET call stays silent and an OUT-OF-SET call
    /// fires exactly once, at the offending argument's own value — never
    /// a wrong fire on the in-set call from `return_sort_fallback`'s own
    /// coarse `-> int` claim (item 1's own regression).
    #[test]
    fn a_vararg_def_interprets_concretely_instead_of_firing_the_coarse_fallback() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def first_age(*ages: int) -> int:\n",
            "    return ages[0]\n",
            "def rest_parameter() -> Age:\n",
            "    good: Age = first_age(40, 41)\n",
            "    _ = good\n",
            "    return first_age(200, 201)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the in-set first_age(40, 41) call must stay silent, and only the \
             out-of-set first_age(200, 201) call must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// e-class-and-function.py's own `unpack_first`/`unpacking_in_body`
    /// shape end to end: `a, _b = ages` (a tuple-unpack `Assign` target)
    /// genuinely binds against the known tuple parameter
    /// (`summaries::bind_unpack_target`), so the in-set call stays silent
    /// and the out-of-set call fires exactly once — never a wrong fire
    /// from the coarse `-> int` fallback on a body that should have
    /// interpreted concretely.
    #[test]
    fn a_tuple_unpack_assign_in_a_summarized_body_interprets_concretely() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def unpack_first(ages: tuple[int, int]) -> int:\n",
            "    a, _b = ages\n",
            "    return a\n",
            "def unpacking_in_body() -> Age:\n",
            "    good: Age = unpack_first((40, 41))\n",
            "    _ = good\n",
            "    return unpack_first((200, 201))\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the in-set unpack_first((40, 41)) call must stay silent, and only the \
             out-of-set unpack_first((200, 201)) call must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    // --- adapter-alias route: TypeAdapter(<alias>).validate_python(<scalar>) ---

    /// m-pydantic-schema.py's `parse_number_chain_ok`/`_over_ceiling` own
    /// shape: `TypeAdapter(Age).validate_python(<int>)` where `Age` is a
    /// bare alias name, not a `BaseModel` class — the class route in
    /// `construction_call_verdict` misses (`context.classes` has no
    /// entry), so the adapter-alias route must judge the argument
    /// directly against `Age`'s own declared set.
    #[test]
    fn type_adapter_validate_python_on_an_alias_judges_the_scalar_argument() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def ok() -> Age:\n",
            "    return TypeAdapter(Age).validate_python(40)\n",
            "def over() -> Age:\n",
            "    return TypeAdapter(Age).validate_python(200)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the in-set validate_python(40) must stay silent, and only the \
             out-of-set validate_python(200) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// UNIT 3, sites 6 and 7 (`adapter_alias_verdict`'s Fire and
    /// Undetermined arms, both built through the shared
    /// `declared_set_instance`): `Age`'s own declared set is
    /// numeric-ground, so the kept instance must carry `kind_tag:
    /// Some(Integer)` in EITHER arm — bound at its own direct sink
    /// (`year: Age = TypeAdapter(Age).validate_python(...)`, since
    /// `construction_call_verdict`, like `callable_variable_call_
    /// result`, only reads a call at `sink_value`'s own value-expression
    /// position), then piped through `math.sqrt` (a sort-gated
    /// consumer, `sqrt_call_over_set`, math_models.rs), which now
    /// derives a value instead of leaving the return undetermined.
    #[test]
    fn adapter_alias_verdicts_fire_arm_kept_instance_reaches_sqrt_tagged() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "import math\n",
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
            "def over() -> Root:\n",
            "    year: Age = TypeAdapter(Age).validate_python(200)\n",
            "    return math.sqrt(year)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            fires.len(),
            1,
            "the out-of-set validate_python(200) still fires once: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(
            blockers.is_empty(),
            "the tagged kept instance must let math.sqrt derive rather than blocking: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// `adapter_alias_verdict`'s Undetermined arm itself records no
    /// finding of its own (only the Fire arm carries one, via
    /// `ConstructionVerdict.fires`) — it keeps `Age`'s own declared set
    /// as the instance, which the enclosing `year: Age = …` AnnAssign
    /// then judges a second time and finds a trivial self-match
    /// (Silent). This test pins that the WHOLE statement stays exactly
    /// as silent as it already was before this unit's tag — the
    /// observable difference the tag makes is downstream, at
    /// `math.sqrt(year)`: an untagged kept instance would leave that
    /// call's own return undetermined; the tagged one lets it derive.
    #[test]
    fn adapter_alias_verdicts_undetermined_arm_kept_instance_reaches_sqrt_tagged() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "import math\n",
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
            "class AudioRequest:\n",
            "    def __init__(self, level) -> None:\n",
            "        self.level = level\n",
            "def f(request: AudioRequest) -> Root:\n",
            // request.level reads back as unknown() — an unmodeled field —
            // so validate_python's own argument judges Undetermined, the
            // arm this test pins.
            "    year: Age = TypeAdapter(Age).validate_python(request.level)\n",
            "    return math.sqrt(year)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "the tagged kept instance must let math.sqrt derive with no fire and no blocker: {:?}",
            findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
        );
    }

    /// m-pydantic-schema.py's `parse_string_chain_over_length` shape: a
    /// STRING-sorted alias (`Label`, min_length/max_length window) judges
    /// its adapter argument the same way.
    #[test]
    fn type_adapter_validate_python_on_a_string_alias_fires_over_length() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type Label = Annotated[str, Field(min_length=1, max_length=8)]\n",
            "def over() -> Label:\n",
            "    return TypeAdapter(Label).validate_python(\"too-long-string\")\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    }

    /// m-pydantic-schema.py's `safe_parse_refused_reified` shape: the
    /// adapter-alias route's own RTS7001 fire, inside a `try` body, is
    /// reified by the SAME try/except machinery every other provable
    /// raise already uses — no special-casing needed once the fire
    /// itself lands.
    #[test]
    fn type_adapter_validate_python_fire_inside_try_is_reified_by_the_except_arm() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "def safe_parse_refused_reified() -> Age:\n",
            "    try:\n",
            "        return TypeAdapter(Age).validate_python(200)\n",
            "    except ValueError:\n",
            "        return 0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// m-pydantic-schema.py's `parse_lax_coercion_ok`/`_out_of_range` own
    /// shape: a lax (non-`StrictInt`) `int` alias coerces a plain digit
    /// string before judging (execution-verified against pydantic 2.13.4:
    /// `"40"` coerces to `40`, `"200"` coerces to `200` and then fails the
    /// range bound).
    #[test]
    fn type_adapter_validate_python_lax_int_alias_coerces_a_digit_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
            "def ok() -> LaxAge:\n",
            "    return TypeAdapter(LaxAge).validate_python(\"40\")\n",
            "def over() -> LaxAge:\n",
            "    return TypeAdapter(LaxAge).validate_python(\"200\")\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the coerced-in-range \"40\" must stay silent, and only the coerced-\
             out-of-range \"200\" must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// m-pydantic-schema.py's `parse_strict_int_ok`/`_refuses_string` own
    /// shape: a `StrictInt`-based alias never coerces a string argument —
    /// a genuine int is admitted, a numeric string fires the ordinary
    /// string-vs-numeric-ground sort mismatch (StrictInt's own refusal,
    /// execution-verified: `.validate_python("40")` raises `int_type` with
    /// no coercion attempt).
    #[test]
    fn type_adapter_validate_python_strict_int_alias_refuses_a_digit_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, StrictInt, TypeAdapter\n",
            "type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]\n",
            "def ok() -> StrictAge:\n",
            "    return TypeAdapter(StrictAge).validate_python(40)\n",
            "def refused() -> StrictAge:\n",
            "    return TypeAdapter(StrictAge).validate_python(\"40\")\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the genuine int 40 must stay silent, and only the numeric string \
             \"40\" must fire (StrictInt never coerces): {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("not assignable"), "{}", fires[0].message);
    }

    /// m-pydantic-schema.py's `parse_pattern_ok` shape: a STR-sorted
    /// pattern alias (`Digits`, `Annotated[str, Field(pattern=r"^[0-9]+$")]`)
    /// must NOT run the lax-int digit-string coercion — a digit-only
    /// STRING is exactly what a `str`-sorted pattern alias accepts on its
    /// own terms, so `TypeAdapter(Digits).validate_python("42")` judges
    /// the string "42" (2 codepoints, inside the pattern/length window)
    /// as a string, never rewritten to the int 42 first. Before gating
    /// `adapter_alias_verdict`'s coercion on `requires_integer(declared_set)`,
    /// this row wrongly fired (the digit-only string coerced to an int,
    /// then the resulting Integer-vs-str-sorted-set mismatch fired) —
    /// this test pins the fix.
    #[test]
    fn type_adapter_validate_python_str_sorted_pattern_alias_never_coerces_a_digit_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type Digits = Annotated[str, Field(min_length=1, max_length=4, pattern=r\"^[0-9]+$\")]\n",
            "def ok() -> Digits:\n",
            "    return TypeAdapter(Digits).validate_python(\"42\")\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert!(
            fires.is_empty(),
            "a digit-only string against a str-sorted pattern alias must judge AS a \
             string, never coerced to an int first: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// m-pydantic-schema.py's `parse_lax_coercion_out_of_range` shape,
    /// re-asserted alongside the str-sorted-alias fix above: an
    /// INT-sorted lax alias must still coerce a digit string and fire
    /// once its coerced value leaves the range — the fix narrows the
    /// coercion to numeric-sorted aliases, it must not also narrow it
    /// away from the int-sorted case that motivated it.
    #[test]
    fn type_adapter_validate_python_int_sorted_lax_alias_still_coerces_and_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, TypeAdapter\n",
            "type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
            "def over() -> LaxAge:\n",
            "    return TypeAdapter(LaxAge).validate_python(\"200\")\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the coerced-out-of-range \"200\" must still fire against an int-sorted alias: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    }

    /// m-pydantic-schema.py's `parse_literal_ok`/`_outside` shape: a bare
    /// `type Pick = Literal[10, 20, 30]` alias (`surface::literal_alias_set`)
    /// judges its adapter argument through the exact same route as a
    /// scalar `Annotated[...]`-compiled alias.
    #[test]
    fn type_adapter_validate_python_on_a_literal_alias_fires_outside_every_member() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Literal\n",
            "from pydantic import TypeAdapter\n",
            "type Pick = Literal[10, 20, 30]\n",
            "def ok() -> Pick:\n",
            "    return TypeAdapter(Pick).validate_python(20)\n",
            "def outside() -> Pick:\n",
            "    return TypeAdapter(Pick).validate_python(25)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the in-set validate_python(20) must stay silent, and only the \
             out-of-set validate_python(25) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'25'"), "{}", fires[0].message);
    }

    /// m-pydantic-schema.py's `parse_union_ok`/`_outside` shape: a
    /// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]`
    /// union alias (`surface::literal_union_alias_set`) judges a member of
    /// EITHER arm as silent and a value in neither arm as a fire — the
    /// kernel's `memberB` derivative walk decides membership over the
    /// whole union set regardless of which arm's sort a given probe value
    /// carries (`RefinedSet.memberB_iff`, refined-ts-lean/set_functions/
    /// membership.lean: total and proved over any concrete tuple).
    #[test]
    fn type_adapter_validate_python_on_a_literal_union_alias_fires_outside_both_arms() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Literal\n",
            "from pydantic import TypeAdapter\n",
            "type PickUnion = Literal[10, 20, 30] | Literal[\"ten\", \"twenty\"]\n",
            "def ok() -> PickUnion:\n",
            "    return TypeAdapter(PickUnion).validate_python(\"ten\")\n",
            "def outside() -> PickUnion:\n",
            "    return TypeAdapter(PickUnion).validate_python(25)\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the in-set validate_python(\"ten\") must stay silent, and only the \
             out-of-both-arms validate_python(25) must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(fires[0].message.contains("'25'"), "{}", fires[0].message);
    }

    /* ── the per-edge foreign-override map ───────────────────────────
     *
     * `walk_body_with_self_binding`'s pending foreign-edge override is
     * keyed by the position of the STATEMENT holding the recognized
     * edge's own `json.loads(...)` consumer (`foreign_edge_overrides:
     * HashMap<usize, Vec<(TextRange, AbstractValue)>>` — a `Vec` rather
     * than a single pair per position, since a discharged crossing whose
     * return is number-sorted publishes a SECOND override at the same
     * position: the intermediate captured-stdout reading's own
     * serialized-string fact, alongside the return fact), never a single
     * unkeyed slot — two recognized crossing calls in the same body,
     * each consumed at a DIFFERENT later statement, must each publish
     * their own fact at their own consumer, with neither clobbering the
     * other's still-pending entry.
     *
     * `findings_for_module_at`'s own route to a recognized `Override`
     * goes through `foreign_edge::foreign_edge_at`, which — in every
     * `cargo test` build of this crate — routes ITS artifact read
     * through `foreign_edge.rs`'s own `#[cfg(test)] read_foreign_ts_
     * artifact`, a crate-wide swap (not scoped to that file's own test
     * module) that answers only from a `FIXTURE_ARTIFACTS` thread-local
     * private to `foreign_edge::tests` (`register_fixture_artifact` is
     * a bare `fn`, `pub(super)` at most — never reachable from this
     * module). So a real on-disk artifact, however written, can never
     * reach `Override` through `foreign_edge_at` from a test in THIS
     * file: the read always fails and the outcome is always `Decline`,
     * which never touches `foreign_edge_overrides` at all. There is no
     * seam left in `check.rs` alone to drive `Override` end to end —
     * fixing that needs either a `pub(crate)` bridge in `foreign_edge
     * .rs` (owned by the other lane) or the corpus's own end-to-end
     * fixture path once its earlier gate (missing chain artifacts, per
     * the brief) is cleared.
     *
     * `foreign_edge_consumer_position` is the one place the ORIGINAL
     * defect actually lived (the position math that used to feed a
     * single `Option` slot) — it takes no artifact, no kernel, and no
     * `Environment`, so it is exactly what these tests can pin
     * end-to-end: TWO recognized calls at different call positions
     * resolve to their OWN independent consumer positions (never one
     * clobbering the other's key), a single recognized call's consumer
     * resolves unchanged, and a call whose result is never parsed
     * anywhere later answers `None` — the same "no position to key an
     * entry under" a single unconsumed slot used to leave as `None`.
     */

    /// A tiny multi-statement body — `first = <call>`; `second =
    /// <call>`; `x = <parse-shaped read>`; `y = <parse-shaped read>` —
    /// built from real parsed statements, so `TextRange`s are genuine
    /// ranges from `ruff_python_parser`, not hand-picked offsets.
    /// Returns the parsed body and, by index, `first`'s call statement,
    /// `second`'s call statement, `x`'s own assignment (whose RHS range
    /// stands in for a `json.loads(first.stdout)` node), and `y`'s own
    /// assignment (standing in for `json.loads(second.stdout)`).
    fn diamond_shaped_body() -> Vec<Stmt> {
        parsed(concat!(
            "def f():\n",
            "    first = call_a()\n",
            "    second = call_b()\n",
            "    x = first\n",
            "    y = second\n",
        ))
        .body
        .into_iter()
        .find_map(|stmt| match stmt {
            Stmt::FunctionDef(def) => Some(def.body.to_vec()),
            _ => None,
        })
        .expect("the fixture's own def body")
    }

    /// The RHS range of the `Assign` at `position` — the stand-in for a
    /// recognized edge's own `json.loads(...)` parse-node range: what
    /// matters for `foreign_edge_consumer_position` is that the range is
    /// CONTAINED by exactly one later statement, not that it is
    /// literally a `json.loads` call.
    fn assign_value_range(body: &[Stmt], position: usize) -> TextRange {
        let Stmt::Assign(assign) = &body[position] else {
            panic!("fixture statement {position} is not an Assign");
        };
        assign.value.range()
    }

    /// THE DIAMOND PIN: two recognized calls (`first` at position 0,
    /// `second` at position 1), each consumed by its OWN later
    /// statement (`x` at position 2 reads `first`'s range, `y` at
    /// position 3 reads `second`'s range) — resolving `second`'s
    /// consumer position must not disturb `first`'s already-resolved
    /// one, and both must resolve to their own distinct, correct
    /// position. A single unkeyed slot has no analogue for "resolve
    /// twice, keep both" at all; this is exactly the map's own
    /// contract.
    #[test]
    fn two_recognized_calls_resolve_to_their_own_independent_consumer_positions() {
        let body = diamond_shaped_body();
        let first_parse_range = assign_value_range(&body, 2); // x = first
        let second_parse_range = assign_value_range(&body, 3); // y = second

        let first_consumer = foreign_edge_consumer_position(&body, 0, first_parse_range);
        let second_consumer = foreign_edge_consumer_position(&body, 1, second_parse_range);

        assert_eq!(first_consumer, Some(2), "first's own consumer must resolve to x's position");
        assert_eq!(second_consumer, Some(3), "second's own consumer must resolve to y's position, unaffected by first's resolution");

        // Keyed exactly as `walk_body_with_self_binding` keys them: two
        // independent map entries, neither overwriting the other.
        let mut overrides: HashMap<usize, TextRange> = HashMap::new();
        overrides.insert(first_consumer.unwrap(), first_parse_range);
        overrides.insert(second_consumer.unwrap(), second_parse_range);
        assert_eq!(overrides.len(), 2, "both entries must coexist in the map, keyed under their own positions");
        assert_eq!(overrides.get(&2), Some(&first_parse_range), "position 2 must still carry first's own range");
        assert_eq!(overrides.get(&3), Some(&second_parse_range), "position 3 must still carry second's own range");
    }

    /// THE SINGLE-EDGE REGRESSION PIN: one recognized call, consumed by
    /// the very next statement — the shape the original single slot
    /// already served correctly — resolves to that next position
    /// unchanged.
    #[test]
    fn a_single_recognized_calls_consumer_resolves_to_the_very_next_statement() {
        let body = diamond_shaped_body();
        let first_parse_range = assign_value_range(&body, 2); // x = first
        let consumer = foreign_edge_consumer_position(&body, 0, first_parse_range);
        assert_eq!(consumer, Some(2), "the single recognized call's consumer must still resolve to its own position");
    }

    /// THE EXPIRY PIN: a recognized call whose own range is never
    /// contained by any LATER statement (no consumer anywhere in the
    /// body reads it) answers `None` — mirroring the single slot's own
    /// behavior for an unconsumed override: nothing is inserted, so
    /// nothing can later leak onto an unrelated statement.
    #[test]
    fn a_call_whose_range_no_later_statement_contains_has_no_consumer_position() {
        let body = diamond_shaped_body();
        // `second`'s own call range (position 1) is never READ by any
        // statement after it in this fixture (only `y`'s RHS, which is
        // the bare name `second`, contains it in the diamond fixture
        // above — here we ask about `first`'s CALL range itself, which
        // no later statement's range contains, since `x = first` reads
        // only the NAME `first`, not the call expression `call_a()`).
        let Stmt::Assign(first_assign) = &body[0] else {
            panic!("fixture statement 0 is not an Assign");
        };
        let call_range = first_assign.value.range();
        let consumer = foreign_edge_consumer_position(&body, 0, call_range);
        assert_eq!(
            consumer, None,
            "a range no later statement contains must answer no consumer position, the same way an override \
             with nothing to key never gets applied anywhere"
        );
    }

    /* ── the export walk's own entry_directory ───────────────────────
     *
     * `derived_return_values_at` threads `entry_directory` into the
     * SAME `WalkContext` field `findings_for_module_at` already
     * populates, so `serve_foreign_edge_at`'s `foreign_edge_at` call
     * (line ~588) resolves a relative argv target the identical way
     * during export as during an ordinary check. `register_fixture_
     * artifact` is private to `foreign_edge::tests` (unreachable from
     * here — the brief's own boundary), so these tests cannot drive a
     * recognized edge all the way to `Override`. What IS reachable: the
     * recognized-but-undischarged edge's own DECLINE sentence, which
     * `discharge_edge_premises` builds from `edge.target_path` AFTER
     * joining it against `entry_directory` when one is given
     * (foreign_edge.rs:395-403). A relative target with no directory
     * stays the bare literal the source wrote; the identical body walked
     * WITH a directory reports the JOINED (absolute) path instead — an
     * observable difference that exists only if the walk actually
     * carried `entry_directory` through to `foreign_edge_at`, which is
     * the one fact these tests pin.
     */

    /// A one-def module whose only statement is a recognized (relative-
    /// target) foreign edge that consumes nothing further — the walk
    /// records it as `blockers`, never `values` (no `return` anywhere in
    /// the body), so the blocker sentence is exactly what
    /// `discharge_edge_premises` names for this decline. Carries a
    /// `type` alias with no other role than clearing
    /// `derived_return_values_at`'s own early exit (`aliases.is_empty()
    /// && imports.annotated_names.is_empty()`) — with neither present
    /// the walk never runs at all, foreign edge included.
    fn foreign_edge_only_module() -> ModModule {
        parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "\n",
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    parsed = json.loads(result.stdout)\n",
        ))
    }

    /// `derived_return_values` (no directory, the pre-existing entry
    /// point) leaves a relative argv target UNJOINED — the blocker names
    /// the bare literal the source wrote, never an absolute path.
    #[test]
    fn derived_return_values_with_no_directory_names_the_bare_relative_target() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = foreign_edge_only_module();
        let derived = derived_return_values(&module, no_imports_resolver(), &kernel);
        let blocker = derived.blockers.get("f").unwrap_or_else(|| {
            panic!("expected a recorded blocker for 'f'; values = {:?}, blockers = {:?}", derived.values, derived.blockers)
        });
        assert!(
            blocker.contains("./audio_level.ts"),
            "with no entry_directory the target must stay the bare relative literal: {blocker:?}"
        );
    }

    /// `derived_return_values_at` WITH a directory joins the SAME
    /// relative target against it before the artifact read — the
    /// blocker now names the absolute (joined) path, proving the export
    /// walk carried `entry_directory` into `foreign_edge_at` exactly as
    /// `findings_for_module_at` already does for an ordinary check.
    #[test]
    fn derived_return_values_at_with_a_directory_joins_the_relative_target_before_declining() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = foreign_edge_only_module();
        let directory = std::path::Path::new("/tmp/refinedpy-export-directory-fixture");
        let derived = derived_return_values_at(&module, no_imports_resolver(), &kernel, Some(directory));
        let blocker = derived.blockers.get("f").unwrap_or_else(|| {
            panic!("expected a recorded blocker for 'f'; values = {:?}, blockers = {:?}", derived.values, derived.blockers)
        });
        // `Path::join` keeps the source's own leading "./" verbatim
        // (foreign_edge.rs's own join: `directory.join(target)` where
        // `target` is `Path::new("./audio_level.ts")`), so the joined
        // spelling is the directory plus that exact relative text, not
        // a normalized form.
        let joined = directory.join("./audio_level.ts");
        assert!(
            blocker.contains(&joined.to_string_lossy().into_owned()),
            "with entry_directory given, the target must be joined against it before the read: {blocker:?}"
        );
    }

    /// A module with no foreign edge at all keeps deriving its return
    /// exactly the same whether or not a directory is given — threading
    /// `entry_directory` through must never change a def's OWN
    /// derivation when nothing in its body ever reads it.
    #[test]
    fn derived_return_values_at_with_a_directory_does_not_disturb_an_ordinary_def() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Age = Annotated[int, Field(ge=0)]\n",
            "\n",
            "def f(x: Age) -> Age:\n",
            "    return x\n",
        );
        let module = parsed(source);
        let no_directory = derived_return_values(&module, no_imports_resolver(), &kernel);
        let with_directory = derived_return_values_at(
            &module,
            no_imports_resolver(),
            &kernel,
            Some(std::path::Path::new("/tmp/refinedpy-export-directory-fixture")),
        );
        assert_eq!(
            format!("{:?}", no_directory.values.get("f")),
            format!("{:?}", with_directory.values.get("f")),
            "an ordinary def's derived return must be identical regardless of entry_directory"
        );
    }

    /// `sample`'s own read INSIDE the `if sample is not None:` arm — the
    /// position `refined_set_at_position` answers for the `sample` name
    /// at the `return sample` site — must read the ANNOTATED, non-None
    /// set (`>= -2 && <= 2`), never the wrapper: `narrow_is_none`'s own
    /// `Kind::PossiblyUndefined` unwrap arm is what this pins.
    #[test]
    fn an_is_not_none_guarded_parameter_narrows_to_its_annotated_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "from typing import Annotated, Optional\n",
            "from pydantic import Field\n",
            "Sample = Optional[Annotated[float, Field(ge=-2.0, le=2.0)]]\n",
            "\n",
            "def f(sample: Sample) -> None:\n",
            "    if sample is not None:\n",
            "        return sample\n",
            "    return None\n",
        );
        let module = parsed(source);
        let position = offset_of(source, "sample\n    return None");
        let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
            .unwrap_or_else(|| panic!("expected the narrowed `sample` read to answer a declared set"));
        assert_eq!(format_for_diagnostics(&set), ">= -2 && <= 2");
    }

    /// The ternary twin of the above: `sample if sample is not None else
    /// 0.0` forks and narrows the SAME way an `if`/`else` STATEMENT does
    /// (`expressions.rs::evaluate_ternary`'s own routing through
    /// `narrowing::assume`) — the joined value is wider than a `Level`
    /// return (`>= 0 && <= 1`), so it fires RTS7001, never sits
    /// undetermined behind RTS7002.
    #[test]
    fn an_is_not_none_guarded_ternary_joins_to_a_wider_set_and_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated, Optional\n",
            "from pydantic import Field\n",
            "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
            "Sample = Optional[Annotated[float, Field(ge=-2.0, le=2.0)]]\n",
            "def f(sample: Sample) -> Level:\n",
            "    return sample if sample is not None else 0.0\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let undetermined: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert!(
            undetermined.is_empty(),
            "the ternary's joined value must be readable, never RTS7002: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
        assert_eq!(
            fires.len(),
            1,
            "the joined set (>= -2 && <= 2, or 0.0) is wider than Level (>= 0 && <= 1) and must fire: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }

    /// Ledger: "Py: loop blockers unnamed when return annotation
    /// unreadable — bare `-> float` never judged; undetermined bodies
    /// must name their blocker regardless." `samples` is a repetition-
    /// window parameter (`repetition_window_element_pass`'s own shape),
    /// so the loop runs through `stabilized_join`'s two-pass abstract
    /// walk rather than a concrete per-element run.
    ///
    /// The accumulation is spelled `total = total + s` — a plain
    /// `Stmt::Assign`, not `total += s`. This matters: `loops.rs`'s
    /// `AugAssign` arm calls the LOOP-LOCAL `binary_arithmetic_value`
    /// directly (declines outright unless BOTH operands reduce to one
    /// exact `Kind::Values` number), while `Stmt::Assign`'s RHS runs
    /// through the ordinary `evaluate_expression` → `evaluate_binop` →
    /// exact-arithmetic path: `total = total * 2.0` reads two exact
    /// `Kind::Values` operands every iteration, so each judged pass
    /// COMPLETES (`run_assign_once` accepts a `Kind::Values` result) and
    /// the two passes bind DIFFERENT exact values ([2.0] then [4.0]) —
    /// a write that provably never settles to a fixed point, reaching
    /// `stabilized_join`'s havoc branch rather than declining the whole
    /// loop before getting there (an element-consuming spelling like
    /// `total + s` evaluates Values + Set to unknown inside the loop
    /// module and declines the pass outright, landing on the older
    /// "is not yet walked" blocker instead). A doubling rebind is
    /// neither a `relational_sum` accumulation (`total += element`) nor
    /// a count spelling (`count = count + 1`), so no more precise
    /// recognizer intercepts it and the walk reaches
    /// `walk_loop`/`stabilized_join` as an ordinary unrecognized rebind.
    ///
    /// The return annotation is a BARE `float`, which `typereading::
    /// declared_refinement` never reads (it only resolves a bare Name
    /// through the module's own alias table), so `return_refinement` is
    /// `None` and `walk_return` judges nothing on its own — before this
    /// fix, that left the body with zero findings despite `total`'s
    /// value never having been determined. The loop itself must still
    /// name that blocker.
    #[test]
    fn a_for_loop_whose_accumulation_does_not_stabilize_names_its_blocker_even_under_a_bare_return_sort() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
            "def f(samples: Annotated[list[Sample], Field(min_length=1)]) -> float:\n",
            "    total = 1.0\n",
            "    for s in samples:\n",
            "        total = total * 2.0\n",
            "        pass\n",
            "    return total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
        assert_eq!(
            blockers.len(),
            1,
            "the loop's own non-stabilizing accumulation must be this body's named blocker, even though \
            the bare `-> float` return gives walk_return nothing to judge: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
        assert!(blockers[0].message.contains("'total'"), "{}", blockers[0].message);
        assert!(
            blockers[0].message.contains("fixed point"),
            "the sentence must name WHY total is unreadable, not just that it is: {}",
            blockers[0].message
        );
    }

    /// The determined twin of the test above: a `for` loop over a
    /// LITERAL list runs through the concrete per-element path
    /// (`iterable_values`), never through `stabilized_join` at all, so
    /// `widened_names` stays empty and the body — bare `-> float` return
    /// included — records no blocker and no fire.
    #[test]
    fn a_for_loop_over_a_literal_list_stays_sentence_free_under_a_bare_return_sort() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def f() -> float:\n",
            "    total = 0.0\n",
            "    for s in [1.0, 2.0, 3.0]:\n",
            "        total += s\n",
            "    return total\n",
        ));
        let findings = findings_for_module(&module, &kernel);
        assert!(
            findings.is_empty(),
            "a concretely-executable for loop over a literal list must stay determined and sentence-free: {:?}",
            findings.iter().map(|f| &f.message).collect::<Vec<_>>()
        );
    }
}
