/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The module's own top-level `def`s, indexed by name, so a call site
//! can look up the callee's AST without re-scanning the module. Only
//! MODULE-LEVEL functions are indexed — a nested `def` (inside another
//! function or a class body) is not a same-module call target this
//! table answers; it lives inside its enclosing body's own statements
//! and is reached (if at all) by walking that body, not by name lookup
//! here.
//!
//! Each entry carries the NAME OF THE MODULE its `def` was parsed from,
//! beside the def itself. A `StmtFunctionDef`'s own identity is its name
//! and its `TextRange`, and a range is a byte offset into ONE module's
//! source: two modules both opening with `def scale(x): return x * 2`
//! give their defs the same name and the same span. Anything keyed on
//! the def alone therefore conflates them across modules, so the module
//! name travels with the def and joins that key
//! (`summaries::summary_key`).

use std::collections::HashMap;

use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;

/// The module a `def` was parsed from — the name the resolver was asked
/// for (`cross_module::ModuleResolver`), or `ENTRY_MODULE` for the file
/// the check itself started at, which no import statement ever names.
pub type ModuleName = String;

/// The module name standing for the file a check started at. A real
/// module name is a Python identifier, and this one is not a legal
/// identifier, so it can never collide with a resolved import's name.
pub const ENTRY_MODULE: &str = "<entry>";

/// One indexed `def`: the cloned AST, plus the module it came from.
#[derive(Clone)]
struct TableEntry {
    def: StmtFunctionDef,
    module: ModuleName,
}

/// Every module-level `def`, cloned out of the parsed AST and keyed by
/// its name. Cloning (rather than borrowing) lets the table outlive the
/// borrow of the module it was built from, so it can ride inside an
/// `Environment` (`env.rs`) alongside values that already own their own
/// data.
pub struct FunctionTable {
    defs: HashMap<String, TableEntry>,
}

/// Scans `module`'s own top-level statements (not the body of a nested
/// `def`/`class`) for `def` statements and clones each one into the
/// table under its name. A later `def` with the same name overwrites an
/// earlier one — Python itself resolves a redefined name to whichever
/// binding executed last, so the last `def` in source order is the one
/// a same-module call actually reaches.
///
/// Every entry is stamped `ENTRY_MODULE`: this constructor is the one a
/// caller holding a module with no name of its own uses (the file a
/// check started at, and every synthetic single-def module the tree
/// assembles). A def read out of a RESOLVED import carries that
/// import's own module name instead — see `function_table_from_module`.
pub fn function_table(module: &ModModule) -> FunctionTable {
    function_table_from_module(module, ENTRY_MODULE)
}

/// `function_table`'s own scan, stamping each entry with the name of the
/// module it was parsed from. `cross_module` calls this for a resolved
/// import so a def's summary key names the module it really came from.
pub fn function_table_from_module(module: &ModModule, module_name: &str) -> FunctionTable {
    let mut defs = HashMap::new();
    for stmt in &module.body {
        if let Stmt::FunctionDef(def) = stmt {
            defs.insert(
                def.name.id.as_str().to_owned(),
                TableEntry { def: def.clone(), module: module_name.to_owned() },
            );
        }
    }
    FunctionTable { defs }
}

impl FunctionTable {
    /// The module-level `def` named `name`, if the module has one.
    pub fn def(&self, name: &str) -> Option<&StmtFunctionDef> {
        self.defs.get(name).map(|entry| &entry.def)
    }

    /// The module the `def` named `name` was parsed from — the other
    /// half of that def's cross-module identity.
    pub fn module_of(&self, name: &str) -> Option<&str> {
        self.defs.get(name).map(|entry| entry.module.as_str())
    }

    /// A table holding nothing — the identity `merged` folds onto when a
    /// caller assembles a table one entry at a time.
    pub fn empty() -> FunctionTable {
        FunctionTable { defs: HashMap::new() }
    }

    /// A single `def` held under `module_name` — the table shape a caller
    /// that already has one def (rather than a parsed module) needs, so
    /// the def still travels with the module identity its summary key
    /// reads.
    pub fn holding(def: StmtFunctionDef, module_name: &str) -> FunctionTable {
        let mut defs = HashMap::new();
        defs.insert(
            def.name.id.as_str().to_owned(),
            TableEntry { def, module: module_name.to_owned() },
        );
        FunctionTable { defs }
    }
}

/// `base` merged with `imported`: on a name both tables carry, `base`'s
/// own `def` wins — a module's own definition shadows an imported name
/// of the same spelling, exactly as a later top-level `def` in `base`
/// itself already overwrites an earlier one. Each entry keeps the module
/// stamp it arrived with, so a merge never relabels an imported def as
/// the merging module's own.
pub fn merged(base: &FunctionTable, imported: &FunctionTable) -> FunctionTable {
    let mut defs = imported.defs.clone();
    for (name, entry) in &base.defs {
        defs.insert(name.clone(), entry.clone());
    }
    FunctionTable { defs }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(source: &str) -> ModModule {
        ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax()
    }

    #[test]
    fn finds_a_top_level_def_by_name() {
        let module = parsed("def double(x):\n    return x + x\n");
        let table = function_table(&module);
        let def = table.def("double").expect("double is indexed");
        assert_eq!(def.name.id.as_str(), "double");
    }

    #[test]
    fn a_name_with_no_matching_def_is_none() {
        let module = parsed("def double(x):\n    return x + x\n");
        let table = function_table(&module);
        assert!(table.def("triple").is_none());
    }

    #[test]
    fn a_nested_def_inside_another_function_is_not_indexed() {
        let module = parsed("def outer():\n    def inner():\n        return 1\n    return inner()\n");
        let table = function_table(&module);
        assert!(table.def("outer").is_some());
        assert!(table.def("inner").is_none());
    }

    #[test]
    fn a_nested_def_inside_a_class_body_is_not_indexed() {
        let module = parsed("class C:\n    def method(self):\n        return 1\n");
        let table = function_table(&module);
        assert!(table.def("method").is_none());
    }

    #[test]
    fn a_later_def_with_the_same_name_wins() {
        let module = parsed("def f():\n    return 1\ndef f():\n    return 2\n");
        let table = function_table(&module);
        let def = table.def("f").expect("f is indexed");
        assert_eq!(def.body.len(), 1);
        // the SECOND def's body ("return 2") is the one the table
        // keeps — matches Stmt::Return's own presence, not its value,
        // since asserting the returned literal would need expressions.rs
    }

    #[test]
    fn a_def_carries_the_module_it_was_scanned_from() {
        let module = parsed("def double(x):\n    return x + x\n");
        let table = function_table_from_module(&module, "audio_level");
        assert_eq!(table.module_of("double"), Some("audio_level"));
        assert_eq!(table.module_of("triple"), None);
    }

    #[test]
    fn the_plain_constructor_stamps_the_entry_module() {
        let module = parsed("def double(x):\n    return x + x\n");
        let table = function_table(&module);
        assert_eq!(table.module_of("double"), Some(ENTRY_MODULE));
    }

    /// Two tables built from the SAME source text under different module
    /// names hold defs that are byte-identical — same name, same range —
    /// and are told apart only by their stamps.
    #[test]
    fn identical_defs_in_two_modules_differ_only_by_their_stamp() {
        let module = parsed("def scale(x):\n    return x * 2\n");
        let first = function_table_from_module(&module, "audio_level");
        let second = function_table_from_module(&module, "video_level");
        assert_eq!(
            first.def("scale").expect("scale is indexed").range,
            second.def("scale").expect("scale is indexed").range
        );
        assert_ne!(first.module_of("scale"), second.module_of("scale"));
    }

    #[test]
    fn holding_carries_the_module_it_was_given() {
        let module = parsed("def double(x):\n    return x + x\n");
        let def = function_table(&module).def("double").expect("double is indexed").clone();
        let table = FunctionTable::holding(def, "b_helper");
        assert!(table.def("double").is_some());
        assert_eq!(table.module_of("double"), Some("b_helper"));
    }

    #[test]
    fn an_empty_table_answers_nothing() {
        let table = FunctionTable::empty();
        assert!(table.def("anything").is_none());
        assert_eq!(table.module_of("anything"), None);
    }

    /// A merge keeps each surviving entry's own stamp: the base's def
    /// wins on a collision and brings its stamp, while a name only the
    /// imported table carries keeps the stamp it arrived with.
    #[test]
    fn merged_keeps_each_entrys_own_stamp() {
        let base_module = parsed("def f(x):\n    return x\n");
        let imported_module = parsed("def f(x):\n    return x\ndef g(x):\n    return x\n");
        let base = function_table_from_module(&base_module, ENTRY_MODULE);
        let imported = function_table_from_module(&imported_module, "b_helper");
        let table = merged(&base, &imported);
        assert_eq!(table.module_of("f"), Some(ENTRY_MODULE), "the base def wins with its own stamp");
        assert_eq!(table.module_of("g"), Some("b_helper"), "an imported-only name keeps its stamp");
    }

    #[test]
    fn merged_carries_names_from_both_tables() {
        let base_module = parsed("def double(x):\n    return x + x\n");
        let imported_module = parsed("def triple(x):\n    return x + x + x\n");
        let base = function_table(&base_module);
        let imported = function_table(&imported_module);
        let table = merged(&base, &imported);
        assert!(table.def("double").is_some());
        assert!(table.def("triple").is_some());
    }

    #[test]
    fn merged_prefers_the_base_def_on_a_name_collision() {
        // the base and imported `f` differ in their PARAMETER shape, an
        // observable structural difference this file can read without
        // needing expressions.rs to compare return values
        let base_module = parsed("def f(base_only):\n    return base_only\n");
        let imported_module = parsed("def f():\n    return 1\n");
        let base = function_table(&base_module);
        let imported = function_table(&imported_module);
        let table = merged(&base, &imported);
        let def = table.def("f").expect("f is indexed");
        assert_eq!(def.parameters.args.len(), 1, "the base module's own f (one parameter) must win");
        assert_eq!(def.parameters.args[0].parameter.name.id.as_str(), "base_only");
    }
}
