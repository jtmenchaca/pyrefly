/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A module's cross-file surface: what an importing file sees when it
//! reads a name out of another module — a plain top-level binding, a
//! `def` (through the module's own `FunctionTable`), a class (through
//! its `ClassModel`), or the module object itself (`import X`/`import
//! X as y`, read member-by-member the same way an object literal is).
//!
//! `module_surface` never touches the filesystem: the caller hands in
//! a `ModuleResolver` closure that turns a module NAME into an already-
//! parsed `ModModule`. The CLI's resolver (`disk_resolver`, this file)
//! reads sibling `.py` files; an LSP resolver would instead ask the
//! host for an open buffer or its own module graph — this file states
//! only the SHAPE the resolver fills in, matching `check.rs`'s own
//! "the walk reads whatever the caller assembled" seam.
//!
//! `FunctionTable` (`function_table.rs`) is opaque and has no mutator.
//! An imported function is folded into `functions` by cloning its
//! `StmtFunctionDef` under the LOCAL name (a fresh `Identifier`
//! replaces the def's own name), then assembling every collected def
//! into one table through that file's own constructors — never a
//! parallel table type.
//!
//! Each def is assembled BESIDE THE NAME OF THE MODULE that declared
//! it. A def's own identity is its name and its `TextRange`, and a
//! range is a byte offset into one module's source, so two sibling
//! modules that both open with the same `def` are indistinguishable by
//! the def alone — the summary registry (`summaries.rs`) would serve
//! one module's compiled answer at the other's call sites. The module
//! stamp is what makes the identity program-wide, and a re-exported def
//! keeps the stamp of the module that DECLARED it rather than the one
//! it was re-exported through, so a def reached under several local
//! names still keys to exactly one summary.
//!
//! Read `SYNTAX-COVERAGE §D` (`fixtures/language/syntax-coverage-py/
//! d-module-surface.py`) for the exact edges this file serves: named
//! import, renamed import, module import (`import X as y` then
//! `y.member`), re-export (an aliased from-import one hop upstream),
//! star-reexport (`from X import *` with an explicit `__all__`), and
//! namespace-reexport (`import X as y` re-exported by name one hop
//! upstream, read the same way at the entry file).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use refined_domain::abstract_value::{AbstractValue, ObjectKey};
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustLibrary;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{
    Expr, Identifier, ModModule, Stmt, StmtAnnAssign, StmtAssign, StmtFunctionDef, StmtImport,
    StmtImportFrom,
};

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::function_table::{
    function_table_from_module, merged, FunctionTable, ENTRY_MODULE,
};
use crate::instances::{class_table, ClassModel};
use crate::surface::{compile_aliases, surface_imports};

/// How a module NAME becomes a parsed module: the CLI resolves a
/// sibling `.py` file on disk (`disk_resolver`), the LSP resolves
/// through its own open-buffer/module-graph host. `module_surface`
/// itself never reads a file — every resolution goes through this
/// closure, so the surface reader stays host-agnostic.
pub type ModuleResolver<'a> = &'a dyn Fn(&str) -> Option<ModModule>;

/// How deep an import chain is followed before `module_surface` stops
/// resolving further imports (a re-export chain, a star-reexport of a
/// star-reexport, …). A cap rather than cycle detection: the corpus's
/// deepest chain (`d-module-surface.py` → `d_star_reexport.py` →
/// `d_helper.py`, two hops) is far inside it, and a cap is the simpler
/// answer to the same "stop eventually" requirement a visited-set would
/// give, with no `HashSet` threaded through every recursive call.
pub const IMPORT_DEPTH_CAP: u32 = 8;

/// One module's readable surface: its own top-level plain bindings
/// (`x = value`, string-annotated `AnnAssign`s), its own top-level
/// `def`s (via `FunctionTable`), its own top-level classes (via
/// `ClassModel`), AND every binding/function/class an import statement
/// pulled in under a local name — a `from X import a` lands `a` in
/// `bindings`/`functions`/`classes` exactly where a same-module `def`
/// or plain assignment would.
pub struct ModuleSurface {
    pub bindings: HashMap<String, AbstractValue>,
    pub functions: Arc<FunctionTable>,
    pub classes: Arc<HashMap<String, ClassModel>>,
}

/// The pre-`FunctionTable` accumulator this file builds while folding
/// imports: every def this module's surface will answer, keyed by its
/// LOCAL name, each collected beside the NAME OF THE MODULE it was
/// parsed from.
///
/// The module name rides along because a def's own identity — its name
/// and its `TextRange` — is only unique WITHIN one module's source
/// (`function_table.rs`'s own doc): two sibling modules opening with the
/// same `def` produce the same name and the same span, and the summary
/// registry would serve one module's compiled answer at the other's call
/// sites. Folding an import therefore records where the def came from,
/// and `assembled_function_table` rebuilds the real `FunctionTable` with
/// every stamp intact.
type DefsByLocalName = HashMap<String, (StmtFunctionDef, String)>;

/// Builds `module`'s surface: first its own top-level plain bindings,
/// functions, and classes; then folds in every `import`/`from…import`
/// statement at the top level, each resolved through `resolver` at
/// `depth - 1` (an `import` under a `depth` of 0 resolves nothing —
/// only this module's own local declarations remain readable; a plain
/// local assign still reads regardless of depth, since it names
/// nothing to resolve). Statements walk in source order, matching
/// `compile_aliases`'s own "a later alias can point at an earlier one"
/// convention, though no corpus row here depends on import-then-shadow
/// ordering within one module.
pub fn module_surface(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> ModuleSurface {
    module_surface_of(module, ENTRY_MODULE, resolver, kernel, depth)
}

/// `module_surface`'s own construction for a module whose NAME is known
/// — the name a resolver was asked for. Every def this module declares
/// itself is stamped with that name, so its summary key names the module
/// it really came from; an imported def keeps the stamp of ITS own
/// source module, one hop further out.
pub fn module_surface_of(
    module: &ModModule,
    module_name: &str,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> ModuleSurface {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    let mut classes = class_table(module, &aliases, &imports, kernel);

    let mut function_environment = Environment::new(Default::default());
    function_environment.set_functions(Arc::new(function_table_from_module(module, module_name)));

    let mut bindings = HashMap::new();
    for stmt in module.body.iter() {
        match stmt {
            Stmt::Assign(assign) => {
                bind_plain_assign(assign, &function_environment, kernel, &mut bindings);
            }
            Stmt::AnnAssign(assign) => {
                bind_plain_ann_assign(assign, &function_environment, kernel, &mut bindings);
            }
            _ => {}
        }
    }

    let mut defs: DefsByLocalName = own_top_level_defs(module, module_name);

    if depth > 0 {
        for stmt in module.body.iter() {
            match stmt {
                Stmt::ImportFrom(import) if import.level == 0 => {
                    fold_import_from(import, resolver, kernel, depth, &mut bindings, &mut defs, &mut classes);
                }
                Stmt::Import(import) => {
                    fold_import(import, resolver, kernel, depth, &mut bindings);
                }
                _ => {}
            }
        }
    }

    let functions = Arc::new(assembled_function_table(defs));

    ModuleSurface {
        bindings,
        functions,
        classes: Arc::new(classes),
    }
}

/// This module's own top-level `def`s, cloned out under their OWN
/// names and stamped with `module_name` — the seed `defs` starts from
/// before any import is folded in. Reuses `function_table`'s own module
/// scan indirectly by walking `module.body` the same way
/// (`function_table.rs`'s own doc: "Scans `module`'s own top-level
/// statements... for `def` statements"), since there is no accessor to
/// enumerate an already-built `FunctionTable`'s entries by name.
fn own_top_level_defs(module: &ModModule, module_name: &str) -> DefsByLocalName {
    let mut defs = HashMap::new();
    for stmt in module.body.iter() {
        if let Stmt::FunctionDef(def) = stmt {
            defs.insert(
                def.name.id.as_str().to_owned(),
                (def.clone(), module_name.to_owned()),
            );
        }
    }
    defs
}

/// The assembled accumulator as the real `FunctionTable`: one entry per
/// LOCAL name, each still carrying the module its def was parsed from.
///
/// Built by merging one single-entry table per def (`FunctionTable::
/// holding`) rather than by wrapping every def in a synthetic
/// `ModModule` and rescanning it: a rescan reads each def's module from
/// the module it is rescanned in, which for a synthetic wrapper is no
/// module at all, and would relabel every imported def as the importing
/// module's own — the exact conflation the stamp exists to prevent.
/// `merged`'s base-wins rule is irrelevant here (the map's keys are
/// already unique), so the fold order does not matter.
///
/// Each def is renamed to its LOCAL name on the way in, so the table's
/// by-name lookup answers the spelling the importing module actually
/// calls. Every def already arrives under that name
/// (`own_top_level_defs` keys by the def's own name, `pull_member`
/// renames an aliased import), so the rename only ever restates it.
fn assembled_function_table(defs: DefsByLocalName) -> FunctionTable {
    let mut table = FunctionTable::empty();
    for (local_name, (def, module_name)) in defs {
        let renamed = rename_def(&def, &local_name);
        table = merged(&FunctionTable::holding(renamed, &module_name), &table);
    }
    table
}

/// A clone of `def` with its `name` identifier replaced by `local_name`
/// — the shape `function_table`'s by-name indexing reads, so a def
/// imported under an alias (`from d_helper import next_year as fn`)
/// answers `FunctionTable::def("fn")`, not `"next_year"`.
fn rename_def(def: &StmtFunctionDef, local_name: &str) -> StmtFunctionDef {
    let mut renamed = def.clone();
    renamed.name = Identifier::new(local_name, renamed.name.range);
    renamed
}

/// `x = value` at module top level, bare-name target only: the RHS
/// reads through `expressions::evaluate_expression` against a fresh
/// environment carrying this module's own `FunctionTable` (so an
/// initializer that calls a same-module function resolves the same
/// way a function body's own call would). A multi-target
/// (`a = b = value`) or destructuring target binds every bare name it
/// reaches; a non-name target contributes nothing.
fn bind_plain_assign(
    assign: &StmtAssign,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    bindings: &mut HashMap<String, AbstractValue>,
) {
    let value = evaluate_expression(assign.value.as_ref(), environment, kernel);
    for target in &assign.targets {
        if let Expr::Name(name) = target {
            bindings.insert(name.id.as_str().to_owned(), value.clone());
        }
    }
}

/// `x: Annotation = value` at module top level, bare-name target only.
/// A value-less declaration (`x: Age` alone) states no initializer —
/// simple_stmts.rst's "Annotated assignment statements" treats the `=`
/// clause as its own optional grammar part — so nothing binds here,
/// matching `check.rs`'s own "declares but does not bind" reading.
fn bind_plain_ann_assign(
    assign: &StmtAnnAssign,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    bindings: &mut HashMap<String, AbstractValue>,
) {
    let Expr::Name(name) = assign.target.as_ref() else {
        return;
    };
    let Some(value_expr) = assign.value.as_deref() else {
        return;
    };
    let value = evaluate_expression(value_expr, environment, kernel);
    bindings.insert(name.id.as_str().to_owned(), value);
}

/// `from X import a, b as c` / `from X import *` — resolves `X` one
/// hop (at `depth - 1`) and pulls members into the accumulators under
/// their LOCAL names. A star import takes every non-underscore-
/// prefixed public binding/function/class of `X`'s own surface,
/// honoring `X`'s `__all__` when it names one as a literal list of
/// string constants (`tmp/cpython/Doc/reference/simple_stmts.rst`,
/// "The import statement," the wildcard-import rule: "If `__all__` is
/// not defined, the set of public names includes all names found in
/// the module's namespace which do not begin with an underscore
/// character"; when `__all__` IS defined, "it must be a sequence of
/// strings which are considered to be defined in that module" and
/// wildcard import binds exactly that sequence). An unresolved source
/// module (the resolver answers `None`) contributes nothing — no
/// panic, no partial guess.
fn fold_import_from(
    import: &StmtImportFrom,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    bindings: &mut HashMap<String, AbstractValue>,
    defs: &mut DefsByLocalName,
    classes: &mut HashMap<String, ClassModel>,
) {
    let Some(source_name) = import.module.as_ref() else {
        return;
    };
    let Some(source_module) = resolver(source_name.id.as_str()) else {
        return;
    };
    let source_surface =
        module_surface_of(&source_module, source_name.id.as_str(), resolver, kernel, depth - 1);

    for alias in &import.names {
        let imported_name = alias.name.id.as_str();
        if imported_name == "*" {
            fold_star_import(&source_module, &source_surface, bindings, defs, classes);
            continue;
        }
        let local_name = alias.asname.as_ref().unwrap_or(&alias.name).id.as_str();
        pull_member(imported_name, local_name, &source_surface, bindings, defs, classes);
    }
}

/// One `from X import <imported_name>[ as <local_name>]` member: a
/// binding, a function, or a class of `X`'s surface all land under
/// `local_name` in the SAME accumulator a same-module declaration
/// would populate — a re-exported function is callable through the
/// final `functions` table exactly like a local `def`, and a
/// re-exported class constructs through `classes` exactly like a
/// local `class`. A name present in more than one of `X`'s three
/// tables (not possible for a well-formed module, since a single
/// top-level name binds at most one way) copies whichever tables have
/// it — never a guess at which one "wins."
fn pull_member(
    imported_name: &str,
    local_name: &str,
    source: &ModuleSurface,
    bindings: &mut HashMap<String, AbstractValue>,
    defs: &mut DefsByLocalName,
    classes: &mut HashMap<String, ClassModel>,
) {
    if let Some(value) = source.bindings.get(imported_name) {
        bindings.insert(local_name.to_owned(), value.clone());
    }
    if let Some(def) = source.functions.def(imported_name) {
        // the def keeps the module stamp `source`'s own table holds for
        // it, which for a RE-EXPORTED name is the module one hop further
        // upstream that really declared it — never `source` itself
        let origin = source
            .functions
            .module_of(imported_name)
            .expect("a def the table answered has a module stamp")
            .to_owned();
        defs.insert(local_name.to_owned(), (rename_def(def, local_name), origin));
    }
    if let Some(class_model) = source.classes.get(imported_name) {
        classes.insert(
            local_name.to_owned(),
            ClassModel {
                name: local_name.to_owned(),
                fields: class_model
                    .fields
                    .iter()
                    .map(|field| crate::instances::ClassField {
                        name: field.name.clone(),
                        declared: field.declared.clone(),
                        default: field.default.clone(),
                    })
                    .collect(),
                properties: class_model
                    .properties
                    .iter()
                    .map(|(property_name, property)| {
                        (
                            property_name.clone(),
                            crate::instances::PropertyModel {
                                backing: property.backing.clone(),
                                declared: property.declared.clone(),
                            },
                        )
                    })
                    .collect(),
                methods: class_model.methods.clone(),
                parent_methods: class_model.parent_methods.clone(),
                class_attributes: class_model.class_attributes.clone(),
            },
        );
    }
}

/// `from X import *` — every public name of `X`'s surface, filtered by
/// `X`'s own `__all__` when `X` states one as a literal list of string
/// constants; otherwise every non-underscore-prefixed public name
/// across `X`'s bindings and classes. `X`'s FUNCTIONS are not enumerated
/// in the no-`__all__` branch: `FunctionTable` (`function_table.rs`)
/// exposes only `def(&str) -> Option<&StmtFunctionDef>`, a by-name
/// lookup, and no way to list every name it holds — so a star import
/// with no `__all__` cannot discover a source module's public `def`
/// names to pull them in. No corpus row needs this: every §D
/// star/namespace-reexport helper (`d_star_reexport.py`) states an
/// explicit `__all__`, which the branch above already handles exactly.
fn fold_star_import(
    source_module: &ModModule,
    source: &ModuleSurface,
    bindings: &mut HashMap<String, AbstractValue>,
    defs: &mut DefsByLocalName,
    classes: &mut HashMap<String, ClassModel>,
) {
    match literal_dunder_all(source_module) {
        Some(names) => {
            for name in &names {
                pull_member(name, name, source, bindings, defs, classes);
            }
        }
        None => {
            let public_binding_names: Vec<String> =
                source.bindings.keys().filter(|name| !name.starts_with('_')).cloned().collect();
            let public_class_names: Vec<String> =
                source.classes.keys().filter(|name| !name.starts_with('_')).cloned().collect();
            for name in public_binding_names.iter().chain(public_class_names.iter()) {
                pull_member(name, name, source, bindings, defs, classes);
            }
        }
    }
}

/// `X`'s own `__all__ = ["a", "b", …]` — a module-level plain `Assign`
/// to the bare name `__all__` whose RHS is a `List`/`Tuple` display of
/// plain string literals only. Any other shape (a computed entry, a
/// name reference, `__all__ += [...]`, no `__all__` at all) answers
/// `None` — the caller falls back to the non-underscore-prefixed rule,
/// never a partial name list.
fn literal_dunder_all(module: &ModModule) -> Option<Vec<String>> {
    for stmt in module.body.iter() {
        let Stmt::Assign(assign) = stmt else { continue };
        let is_dunder_all = assign
            .targets
            .iter()
            .any(|target| matches!(target, Expr::Name(name) if name.id.as_str() == "__all__"));
        if !is_dunder_all {
            continue;
        }
        let elements: &[Expr] = match assign.value.as_ref() {
            Expr::List(list) => &list.elts,
            Expr::Tuple(tuple) => &tuple.elts,
            _ => return None,
        };
        let mut names = Vec::with_capacity(elements.len());
        for element in elements {
            let Expr::StringLiteral(literal) = element else {
                return None;
            };
            names.push(literal.value.to_str().to_owned());
        }
        return Some(names);
    }
    None
}

/// `import X` / `import X as y` — binds the local name (`y`, or `X`
/// itself with no `as`) to a MODULE-OBJECT value: `known_object` whose
/// `ObjectKey` rows are `X`'s own surface BINDINGS (the plain values a
/// `y.member` read resolves through the same `subscript_read`/
/// `dict_key_read` convention `collection_models.rs` already gives a
/// dict literal's fields). Functions and classes are not folded into
/// this object's keys: no corpus row in §D reads a function or class
/// off a bound module object (`helper.forty_years`/`over_years` are
/// both plain bindings), and `known_object`'s `ObjectKey.value` slot
/// is an `AbstractValue`, which a `FunctionTable`/`ClassModel` entry
/// is not — folding either in would need a value shape this domain
/// does not carry. Only a plain (non-dotted) module name resolves —
/// `import a.b.c` binds the top-level package `a`, whose surface this
/// function cannot assemble from a dotted resolver request, so it
/// contributes nothing for that shape (`resolver` is asked with the
/// dotted spelling and, per `disk_resolver`'s own contract, answers
/// `None`).
fn fold_import(
    import: &StmtImport,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    bindings: &mut HashMap<String, AbstractValue>,
) {
    for alias in &import.names {
        let module_name = alias.name.id.as_str();
        let Some(source_module) = resolver(module_name) else {
            continue;
        };
        let source_surface =
            module_surface_of(&source_module, module_name, resolver, kernel, depth - 1);
        let local_name = alias.asname.as_ref().unwrap_or(&alias.name).id.as_str();
        let entries: Vec<ObjectKey> = source_surface
            .bindings
            .iter()
            .map(|(name, value)| ObjectKey {
                name: name.clone(),
                numeric: false,
                value: value.clone(),
            })
            .collect();
        let module_object = known_object(entries, None, true, TrustLibrary, false);
        bindings.insert(local_name.to_owned(), module_object);
    }
}

/// `<entry_directory>/<module_name>.py`, parsed the same way every
/// other file in this checker parses a module (`ruff_python_parser::
/// parse_module`). Only a plain (non-dotted) module name resolves — a
/// dotted name (`import a.b.c`) has no single sibling file this
/// resolver can name, so it answers `None` rather than guessing at
/// `a/b/c.py` vs `a.b.c.py`. A missing file, or a file that fails to
/// parse, both answer `None` — the caller's fold functions already
/// treat "unresolved" as "contributes nothing."
pub fn disk_resolver(entry_directory: PathBuf) -> impl Fn(&str) -> Option<ModModule> {
    move |module_name: &str| {
        if module_name.contains('.') {
            return None;
        }
        let path = entry_directory.join(format!("{module_name}.py"));
        let source = fs::read_to_string(&path).ok()?;
        let parsed = ruff_python_parser::parse_module(&source).ok()?;
        Some(parsed.into_syntax())
    }
}

#[cfg(test)]
mod tests {
    use refined_domain::abstract_value::{known_values, Kind, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};

    use super::*;

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn parsed(source: &str) -> ModModule {
        ruff_python_parser::parse_module(source).expect("test source parses").into_syntax()
    }

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    /// A resolver over an in-memory map of module name -> source text —
    /// no disk touched, matching the mission's "you never touch the
    /// filesystem in the surface reader itself."
    fn map_resolver(sources: HashMap<&'static str, &'static str>) -> impl Fn(&str) -> Option<ModModule> {
        move |name: &str| sources.get(name).map(|source| parsed(source))
    }

    // --- plain binding read ---

    #[test]
    fn a_plain_top_level_binding_reads() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed("forty = 40\n");
        let resolver = map_resolver(HashMap::new());
        let surface = module_surface(&module, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.bindings.get("forty"), Some(&integer(40.0)));
    }

    // --- from-import ---

    #[test]
    fn a_named_from_import_reads_the_source_modules_binding() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("helper", "forty = 40\n");
        let entry = parsed("from helper import forty\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.bindings.get("forty"), Some(&integer(40.0)));
    }

    // --- renamed from-import ---

    #[test]
    fn a_renamed_from_import_binds_under_the_local_alias() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("helper", "forty = 40\n");
        let entry = parsed("from helper import forty as renamed\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.bindings.get("renamed"), Some(&integer(40.0)));
        assert!(surface.bindings.get("forty").is_none());
    }

    // --- re-export chain: entry imports from A, A imports (renamed) from B ---

    #[test]
    fn a_re_export_chain_resolves_through_the_middle_module() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("b_helper", "forty = 40\n");
        sources.insert("a_reexport", "from b_helper import forty as re_forty\n");
        let entry = parsed("from a_reexport import re_forty\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.bindings.get("re_forty"), Some(&integer(40.0)));
    }

    // --- star re-export, with __all__ ---

    #[test]
    fn a_star_reexport_with_dunder_all_pulls_only_the_named_members() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("helper", "forty = 40\nover = 200\n_private = 1\n");
        sources.insert(
            "star_helper",
            "from helper import *\n__all__ = [\"forty\", \"over\"]\n",
        );
        let entry = parsed("from star_helper import forty, over\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.bindings.get("forty"), Some(&integer(40.0)));
        assert_eq!(surface.bindings.get("over"), Some(&integer(200.0)));
    }

    // --- star re-export, no __all__ (non-underscore rule) ---

    #[test]
    fn a_star_import_with_no_dunder_all_skips_underscore_prefixed_names() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("helper", "forty = 40\n_private = 1\n");
        let entry = parsed("from helper import *\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.bindings.get("forty"), Some(&integer(40.0)));
        assert!(surface.bindings.get("_private").is_none());
    }

    // --- import-as module object with a member row ---

    #[test]
    fn a_module_import_binds_a_module_object_with_member_rows() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("helper", "forty = 40\nover = 200\n");
        let entry = parsed("import helper as h\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        let module_object = surface.bindings.get("h").expect("h binds a module object");
        assert_eq!(module_object.kind, Kind::Object);
        assert_eq!(
            crate::collection_models::subscript_read(
                module_object,
                &known_values(
                    "forty".chars().map(|c| c as u32 as f64).collect(),
                    PrimitiveKind::String,
                    TrustProved
                )
            ),
            Some(integer(40.0))
        );
    }

    // --- depth cap ---

    #[test]
    fn depth_zero_resolves_no_imports_but_still_reads_local_bindings() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("helper", "forty = 40\n");
        let entry = parsed("local = 1\nfrom helper import forty\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, 0);
        assert_eq!(surface.bindings.get("local"), Some(&integer(1.0)));
        assert!(surface.bindings.get("forty").is_none());
    }

    #[test]
    fn depth_one_resolves_exactly_one_hop() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("b_helper", "forty = 40\n");
        sources.insert("a_reexport", "from b_helper import forty as re_forty\n");
        let entry = parsed("from a_reexport import re_forty\n");
        let resolver = map_resolver(sources);
        // depth 1: entry -> a_reexport resolves (hop 1), but a_reexport's
        // OWN from-import of b_helper needs depth 0 there and resolves
        // nothing, so re_forty is never actually bound inside a_reexport's
        // own surface either.
        let surface = module_surface(&entry, &resolver, &kernel, 1);
        assert!(surface.bindings.get("re_forty").is_none());
    }

    #[test]
    fn depth_two_resolves_the_full_two_hop_chain() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("b_helper", "forty = 40\n");
        sources.insert("a_reexport", "from b_helper import forty as re_forty\n");
        let entry = parsed("from a_reexport import re_forty\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, 2);
        assert_eq!(surface.bindings.get("re_forty"), Some(&integer(40.0)));
    }

    // --- module stamps (the cross-module summary identity) ---

    /// A module's OWN def is stamped with that module's own name — the
    /// entry file's, when the check started there.
    #[test]
    fn a_modules_own_def_is_stamped_with_its_own_module_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed("def scale(x):\n    return x * 2\n");
        let resolver = map_resolver(HashMap::new());
        let surface = module_surface(&module, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.functions.module_of("scale"), Some(ENTRY_MODULE));
    }

    /// An IMPORTED def is stamped with the module that declared it, not
    /// with the importing module — the stamp a cross-module call's
    /// summary key reads.
    #[test]
    fn an_imported_def_is_stamped_with_the_module_that_declared_it() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("audio_level", "def scale(x):\n    return x * 2\n");
        let entry = parsed("from audio_level import scale\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert!(surface.functions.def("scale").is_some(), "the imported def is reachable");
        assert_eq!(surface.functions.module_of("scale"), Some("audio_level"));
    }

    /// Two sibling modules whose defs are BYTE-IDENTICAL — same name,
    /// same span — are told apart by their stamps. Without the stamp the
    /// two defs share a summary key entirely, and one module's compiled
    /// answer would serve the other's calls.
    #[test]
    fn two_modules_with_an_identical_def_are_told_apart_by_their_stamps() {
        let Some(kernel) = loaded_kernel() else { return };
        let identical = "def scale(x):\n    return x * 2\n";
        let mut sources = HashMap::new();
        sources.insert("audio_level", identical);
        sources.insert("video_level", identical);
        let entry = parsed("from audio_level import scale\nfrom video_level import scale as other\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        let first = surface.functions.def("scale").expect("scale imports");
        let second = surface.functions.def("other").expect("other imports");
        assert_eq!(
            first.range, second.range,
            "the two defs really are byte-identical — the stamp is the only thing telling them apart"
        );
        assert_eq!(surface.functions.module_of("scale"), Some("audio_level"));
        assert_eq!(surface.functions.module_of("other"), Some("video_level"));
    }

    /// A RENAMED import keeps the declaring module's stamp: the local
    /// name changes, the identity does not.
    #[test]
    fn a_renamed_import_keeps_the_declaring_modules_stamp() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("audio_level", "def scale(x):\n    return x * 2\n");
        let entry = parsed("from audio_level import scale as boost\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.functions.module_of("boost"), Some("audio_level"));
    }

    /// A def reached through a RE-EXPORT chain is stamped with the module
    /// that DECLARED it, never the module it was re-exported through —
    /// so the same def reached by two routes keys to one summary.
    #[test]
    fn a_re_exported_def_keeps_the_declaring_modules_stamp() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("b_helper", "def scale(x):\n    return x * 2\n");
        sources.insert("a_reexport", "from b_helper import scale\n");
        let entry = parsed("from a_reexport import scale\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        assert_eq!(surface.functions.module_of("scale"), Some("b_helper"));
    }

    /// A module's own def and an imported one of the SAME spelling: the
    /// walk resolves this by re-merging the module's own table over this
    /// surface (`check.rs`'s `merged(&own_functions, …)`, base wins), and
    /// the merge carries each entry's stamp with it — so whichever def
    /// wins, its stamp is the stamp of the module that declared it, never
    /// the other's.
    #[test]
    fn merging_a_local_table_over_the_surface_carries_both_stamps() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut sources = HashMap::new();
        sources.insert("audio_level", "def scale(x):\n    return x * 2\n");
        sources.insert("video_level", "def other(x):\n    return x * 3\n");
        let entry = parsed("from audio_level import scale\nfrom video_level import other\ndef scale(x):\n    return x * 4\n");
        let resolver = map_resolver(sources);
        let surface = module_surface(&entry, &resolver, &kernel, IMPORT_DEPTH_CAP);
        let own = function_table_from_module(&entry, ENTRY_MODULE);
        let table = merged(&own, surface.functions.as_ref());
        assert_eq!(table.module_of("scale"), Some(ENTRY_MODULE), "the local def wins, with its own stamp");
        assert_eq!(table.module_of("other"), Some("video_level"), "the imported def keeps its declaring stamp");
    }

    // --- disk_resolver ---

    #[test]
    fn disk_resolver_reads_a_sibling_file_by_module_name() {
        let dir = std::env::temp_dir().join(format!("refinedpy_cross_module_test_{}", std::process::id()));
        fs::create_dir_all(&dir).expect("create temp dir");
        let file_path = dir.join("sibling.py");
        fs::write(&file_path, "forty = 40\n").expect("write sibling module");

        let resolver = disk_resolver(dir.clone());
        let resolved = resolver("sibling").expect("sibling.py resolves");
        assert!(matches!(resolved.body.first(), Some(Stmt::Assign(_))));

        assert!(resolver("dotted.name").is_none());
        assert!(resolver("does_not_exist").is_none());

        let _ = fs::remove_file(&file_path);
        let _ = fs::remove_dir(&dir);
    }
}
