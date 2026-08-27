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
//! carrying no error) rather than reporting a second time for the same
//! refusal.
//!
//! The `pub(super) use <child>::*;` block below is this module's one
//! door: `check/walk/`'s files reach the other children's rows through
//! it. rustc reports those globs as unreexporting because it reads the
//! target module's own visibility rather than the consumers, so the
//! lint is answered here instead of by deleting lines the walk needs
//! (removing them was measured at 200 compile errors).
#![allow(unused_imports)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::Mutex;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{Expr, ModModule, Stmt};
use ruff_text_size::TextRange;

use crate::cross_module::{module_surface, ModuleResolver};
use crate::env::Environment;
use crate::expressions::math_from_imports;
use crate::function_table::{function_table, merged, FunctionTable};
use crate::instances;
use crate::instances::{class_table, ClassModel};
use crate::surface::{compile_aliases, strict_int_alias_names, surface_imports, AliasEntry};
use crate::typereading::{callable_return_refinement, DeclaredRefinement, TypedDictMember};

mod hover;
mod walk;
mod seed;
mod function_def;
mod class_def;
mod branch;
mod control;
mod bind;
mod aug_assign;
mod calls;
mod pydantic;
mod judge;
#[cfg(test)]
mod tests;

pub use bind::setdefault_append;
pub use function_def::{derived_return_values, derived_return_values_at, DerivedReturns};
pub use hover::refined_set_at_position;

// Sibling-shared helpers: children call these as `super::name` / `use super::*`.
pub(super) use aug_assign::*;
pub(super) use bind::*;
pub(super) use branch::*;
pub(super) use calls::*;
pub(super) use class_def::*;
pub(super) use control::*;
pub(super) use function_def::*;
pub(super) use hover::*;
pub(super) use judge::*;
pub(super) use pydantic::*;
pub(super) use seed::*;
pub(super) use walk::*;

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
pub(super) struct WalkContext<'a> {
    pub(super) aliases: &'a HashMap<String, AliasEntry>,
    pub(super) imports: &'a crate::surface::SurfaceImports,
    pub(super) kernel: &'a Arc<RefinedTSKernel>,
    pub(super) functions: Arc<FunctionTable>,
    pub(super) classes: Arc<HashMap<String, ClassModel>>,
    /// The module's own `datetime` import identities — which local
    /// names mean the `datetime` module, `datetime.datetime`,
    /// `datetime.date`, and `datetime.timedelta`
    /// (`expressions::DatetimeImports`'s own doc), built once here the
    /// same "built once before any body walk" posture `functions`/
    /// `classes` already take, and layered onto each body's own
    /// `Environment` (`walk_body_with_self_binding`) so
    /// `expressions.rs`'s datetime gates answer by canonical identity
    /// rather than the literal `datetime`/`date`/`timedelta` spelling.
    pub(super) datetime_imports: Arc<crate::expressions::DatetimeImports>,
    /// Whether this module never calls `locale.setlocale` anywhere in
    /// its own source (`expressions::module_never_calls_setlocale`'s
    /// own doc), built once here the same "built once before any body
    /// walk" posture `datetime_imports` takes, and layered onto each
    /// body's own `Environment` (`walk_body_with_self_binding`) so
    /// `datetime.strptime`'s `%a` reading can answer the C-locale
    /// premise without a signature change anywhere along the call
    /// chain.
    pub(super) locale_never_set: bool,
    pub(super) module_bindings: HashMap<String, AbstractValue>,
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
    pub(super) module_callable_returns: Arc<HashMap<String, DeclaredRefinement>>,
    /// Every module-level `type X = Annotated[StrictInt, …]` alias name
    /// (`surface::strict_int_alias_names`) — the TypeAdapter adapter
    /// route consults this to decide whether a `str` argument against
    /// this alias may coerce (a lax `int` base) or must refuse outright
    /// (a `StrictInt` base never attempts str-to-int coercion,
    /// execution-verified against pydantic 2.13.4).
    pub(super) strict_int_aliases: &'a HashSet<String>,
    /// Every module-level `class X(TypedDict): name: Annotation, …`
    /// read into its own per-member refinement table
    /// (`instances::typed_dict_table`), keyed on the class's name —
    /// consulted where a `-> X`/`x: X` annotation names a TypedDict
    /// rather than a `type X = …` alias, so a dict literal judged
    /// against it is judged member-by-member (`typed_dict_return_
    /// refinement`) instead of reading as unrefined.
    pub(super) typed_dicts: Arc<HashMap<String, Vec<TypedDictMember>>>,
    /// Every module-level def's own direct callers' positional argument
    /// lists, keyed by the def's name (`function_table::
    /// caller_argument_positions`) — a def missing from this table either
    /// has no module-level definition or does not qualify (some
    /// occurrence of its name is not a plain positional call, so no
    /// caller of it is safe to join). `seed_parameters` reads this to
    /// fold an UNANNOTATED parameter's every caller argument at that
    /// position into a literal-union seed.
    pub(super) caller_arguments: Arc<crate::function_table::CallerArguments>,
    /// The checked file's own directory, when the caller knows it — a
    /// foreign edge's relative argv entry (`"./audio_level.ts"`) is
    /// relative to the file that wrote it, never to the eventual
    /// process's cwd, so the recognizer joins against this. `None`
    /// (the resolver-less test entry points) leaves relative targets
    /// unresolved, which declines honestly downstream.
    pub(super) entry_directory: Option<std::path::PathBuf>,
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
    pub(super) evaluations_recorder: Option<Arc<Mutex<Vec<(TextRange, AbstractValue)>>>>,
    /// The whole module walk's shared derivation-trace collector, when a
    /// caller asked for one (`refinedpy-check --trace-verdict`).
    /// `walk_body_with_self_binding` installs this SAME `Arc` on every
    /// body's fresh `Environment` the moment it builds one, exactly the
    /// way `evaluations_recorder` above is installed and for the same
    /// reason: the blocked position may sit inside any nested `def`'s own
    /// body. `None` for every ordinary check.
    pub(super) trace_collector: Option<Arc<Mutex<crate::trace::TraceCollector>>>,
}

/// `surface.bindings` (the cross-module resolver's own map) plus every
/// `from math import inf/nan/pi/e/tau[ as x]` local name this module's
/// own top-level statements bind (`expressions::math_from_imports`'s own
/// doc) — `math` is a host module with no Python source for the
/// cross-module resolver to read, so `surface.bindings` never carries a
/// `math` name on its own; this is the one place those two tables meet.
/// Math wins on a spelling collision (impossible in practice: the
/// resolver only ever answers a name found in ANOTHER module's own
/// source, and no module actually named `math` exists on disk here), the
/// same "last write wins" merge direction as every other table join in
/// this file. `WalkContext.module_bindings` is what `bind_or_forget_
/// imported_name`'s existing per-import-statement walk and `module_
/// scope_environment`'s existing seed both already read — so a module-
/// level `from math import inf` becomes readable through the SAME
/// mechanism a `from helper import forty` cross-module import already
/// uses, with no new `Environment` field.
pub(super) fn module_bindings_with_math_imports(surface_bindings: HashMap<String, AbstractValue>, module: &ModModule) -> HashMap<String, AbstractValue> {
    let mut bindings = surface_bindings;
    bindings.extend(math_from_imports(module));
    bindings
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
    findings_for_module_traced(module, resolver, kernel, entry_directory, None)
}

/// `findings_for_module_at` plus an optional derivation-trace collector
/// (`trace`'s own module doc). With `None` — every ordinary check, every
/// existing caller — this walks byte-identically to before the trace
/// existed: the collector field is `None` all the way down and every
/// recording entry point returns on its first `Cell<bool>` read.
///
/// With `Some`, the SAME `Arc` is installed on every body's environment
/// AND published into the thread-local slot the two `Environment`-less
/// seams read (`trace::install`), for the whole duration of the walk.
pub fn findings_for_module_traced(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
    trace_collector: Option<Arc<Mutex<crate::trace::TraceCollector>>>,
) -> Vec<Finding> {
    // The guard lives for the whole walk and clears the thread-local on
    // the way out, including on an unwind.
    let _trace_guard = trace_collector.clone().map(crate::trace::install);
    let surface = module_surface(module, resolver, kernel);
    // The module's own aliases, plus every alias an import pulled in
    // under a local name (`from support import Age`); an own alias wins
    // a spelling collision, matching every other merge in this function.
    let mut aliases = surface.aliases.clone();
    for (name, alias) in compile_aliases(module) {
        aliases.insert(name, alias);
    }
    let imports = surface_imports(module);
    // Every module reaches the walk. Refinement vocabulary decides what a
    // STATED set is, never whether this module has anything to judge: a
    // dead guard, a comparison the kernel proves, a temporal transfer,
    // and every designated position are all judgments the walk makes from
    // the statements themselves, with no alias, `Annotated`, or `Literal`
    // anywhere in the file. A module that truly states nothing walks and
    // reports nothing, which is the same answer at the cost of running.
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
        locale_never_set: crate::expressions::module_never_calls_setlocale(module),
        module_bindings: module_bindings_with_math_imports(surface.bindings, module),
        module_callable_returns,
        strict_int_aliases: &strict_int_aliases,
        typed_dicts,
        caller_arguments,
        entry_directory: entry_directory.map(|dir| dir.to_path_buf()),
        evaluations_recorder: None,
        trace_collector,
    };
    let mut out = Vec::new();
    walk_body(&module.body, None, None, None, &context, &mut out);
    out
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
pub(super) fn module_level_callable_returns(
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
