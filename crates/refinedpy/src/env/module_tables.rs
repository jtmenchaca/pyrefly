//! The module-level tables an `Environment` carries so a call evaluated
//! against it (and any environment forked from it) can look up a
//! same-module def/class/alias/import fact with no signature change
//! anywhere along the call chain — set once per module walk and read
//! from wherever the walk reaches.

use std::collections::HashMap;
use std::sync::Arc;

use crate::function_table::FunctionTable;
use crate::instances::ClassModel;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;
use crate::typereading::DeclaredRefinement;

use super::Environment;

impl Environment {
    /// Attaches the module's function table so calls evaluated against
    /// this environment (and any environment forked from it) can look
    /// up a same-module callee by name.
    pub fn set_functions(&mut self, functions: Arc<FunctionTable>) {
        self.functions = Some(functions);
    }

    /// The module's function table, if this environment carries one.
    pub fn functions(&self) -> Option<&Arc<FunctionTable>> {
        self.functions.as_ref()
    }

    /// Attaches the module's `datetime` import identities so a
    /// construction/classmethod call evaluated against this
    /// environment (and any environment forked from it) can resolve
    /// `datetime`/`date`/`timedelta` by canonical identity rather than
    /// literal spelling (`DatetimeImports`'s own doc).
    pub fn set_datetime_imports(&mut self, datetime_imports: Arc<crate::expressions::DatetimeImports>) {
        self.datetime_imports = Some(datetime_imports);
    }

    /// The module's `datetime` import identities, if this environment
    /// carries one.
    pub fn datetime_imports(&self) -> Option<&Arc<crate::expressions::DatetimeImports>> {
        self.datetime_imports.as_ref()
    }

    /// Attaches the module's own `locale.setlocale`-never-called fact
    /// so `datetime.strptime`'s `%a` reading, evaluated against this
    /// environment (and any environment forked from it), can read the
    /// C-locale premise (`module_never_calls_setlocale`'s own doc).
    pub fn set_locale_never_set(&mut self, locale_never_set: bool) {
        self.locale_never_set = Some(locale_never_set);
    }

    /// The module's own `locale.setlocale`-never-called fact, if this
    /// environment carries one.
    pub fn locale_never_set(&self) -> Option<bool> {
        self.locale_never_set
    }

    /// Attaches the checked file's own directory so a call evaluated
    /// against this environment (and any environment forked from it) can
    /// discover a manifest file beside the checked file
    /// (`binding_manifest.rs`'s own discovery convention).
    pub fn set_entry_directory(&mut self, entry_directory: Arc<std::path::PathBuf>) {
        self.entry_directory = Some(entry_directory);
    }

    /// The checked file's own directory, if this environment carries one.
    pub fn entry_directory(&self) -> Option<&Arc<std::path::PathBuf>> {
        self.entry_directory.as_ref()
    }

    /// Attaches the module's class table so a construction call
    /// evaluated against this environment (and any environment forked
    /// from it) can look up a same-module class by name.
    pub fn set_classes(&mut self, classes: Arc<HashMap<String, ClassModel>>) {
        self.classes = Some(classes);
    }

    /// The module's class table, if this environment carries one.
    pub fn classes(&self) -> Option<&Arc<HashMap<String, ClassModel>>> {
        self.classes.as_ref()
    }

    /// Attaches the module's compiled alias table and import identities
    /// so a same-module callee's declared return annotation, evaluated
    /// against this environment (and any environment forked from it),
    /// can be read through `typereading::declared_refinement` — the
    /// same table `check.rs::walk_function_def` already reads a def's
    /// own `-> Annotation` through, made reachable from
    /// `summaries.rs`'s decline path too.
    pub fn set_declared_aliases(&mut self, aliases: Arc<HashMap<String, AliasEntry>>, imports: Arc<SurfaceImports>) {
        self.declared_aliases = Some((aliases, imports));
    }

    /// The module's compiled alias table and import identities, if this
    /// environment carries one.
    pub fn declared_aliases(&self) -> Option<(&Arc<HashMap<String, AliasEntry>>, &Arc<SurfaceImports>)> {
        self.declared_aliases.as_ref().map(|(aliases, imports)| (aliases, imports))
    }

    /// Attaches this body's callable-return table so a call site
    /// evaluated against this environment (and any environment forked
    /// from it) can look up a bare-Name callable's return refinement.
    pub fn set_callable_returns(&mut self, callable_returns: Arc<HashMap<String, DeclaredRefinement>>) {
        self.callable_returns = Some(callable_returns);
    }

    /// This body's callable-return table, if it carries one.
    pub fn callable_returns(&self) -> Option<&Arc<HashMap<String, DeclaredRefinement>>> {
        self.callable_returns.as_ref()
    }

    /// How many interpreted CALLS deep this environment sits — 0 for a
    /// walked body, parent + 1 inside each summaries/instances body
    /// interpretation. Dispatch sites pass this into the interpreters
    /// so the CALL_DEPTH_CAP engages across the evaluate↔summaries
    /// boundary; without it a self-recursive def (`countdown` calling
    /// itself through the function table) re-entered at depth 0 forever
    /// and overflowed the stack.
    pub fn call_depth(&self) -> u32 {
        self.call_depth
    }

    pub fn set_call_depth(&mut self, depth: u32) {
        self.call_depth = depth;
    }

    /// Records this body's own `*args`/`**kwargs` parameter names (see
    /// the field's own doc).
    pub fn set_variadic_parameter_names(&mut self, names: Arc<std::collections::HashSet<String>>) {
        self.variadic_parameter_names = names;
    }

    /// Whether `name` is THIS body's own `*args`/`**kwargs` parameter —
    /// a bare-Name read of one of these is always a FORWARD of exactly
    /// what this body itself received, never an independently-built
    /// value.
    pub fn is_variadic_parameter(&self, name: &str) -> bool {
        self.variadic_parameter_names.contains(name)
    }
}
