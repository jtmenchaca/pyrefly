use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use refined_domain::abstract_value::{AbstractValue, Kind};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::{make_refined_set, one_of, RefinedSet};
use ruff_python_ast::{Expr, ModModule, Stmt};
use ruff_text_size::{Ranged, TextRange, TextSize};

use crate::cross_module::{module_surface, ModuleResolver};
use crate::env::Environment;
use crate::function_table::{function_table, merged};
use crate::instances;
use crate::instances::class_table;
use crate::surface::{compile_aliases, strict_int_alias_names, surface_imports, AliasEntry};
use crate::typereading::{base_sort_return_refinement, declared_refinement};

use super::*;

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
    let surface = module_surface(module, resolver, kernel);
    let mut aliases = surface.aliases.clone();
    for (name, alias) in compile_aliases(module) {
        aliases.insert(name, alias);
    }
    let imports = surface_imports(module);
    if aliases.is_empty()
        && imports.annotated_names.is_empty()
        && imports.literal_names.is_empty()
    {
        // Same gate `findings_for_module_at` takes, and for the same
        // reason: a module with no refinement alias (own or imported),
        // no recognized `Annotated` import, and no `Literal` import
        // carries no refinement vocabulary at all.
        return None;
    }
    if let Some(stated) = stated_refinement_at(module, &aliases, &imports, position) {
        return Some(stated);
    }
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
        locale_never_set: crate::expressions::module_never_calls_setlocale(module),
        module_bindings: module_bindings_with_math_imports(surface.bindings, module),
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
pub(super) fn stated_refinement_at(
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

pub(super) fn stated_refinement_in_body(
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
pub(super) fn alias_declaration_set_at(stmt: &Stmt, aliases: &HashMap<String, AliasEntry>, position: TextSize) -> Option<RefinedSet> {
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
pub(super) fn smallest_covering_set(recorded: &[(TextRange, AbstractValue)], position: TextSize) -> Option<RefinedSet> {
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
pub(super) fn abstract_value_as_refined_set(value: &AbstractValue) -> Option<RefinedSet> {
    match value.kind {
        Kind::Set => Some(value.set.clone()),
        Kind::Values if !value.values.is_empty() => {
            Some(make_refined_set(vec![one_of(&value.values)]))
        }
        _ => None,
    }
}
