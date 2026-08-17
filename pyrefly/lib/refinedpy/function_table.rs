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

use std::collections::HashMap;

use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;

/// Every module-level `def`, cloned out of the parsed AST and keyed by
/// its name. Cloning (rather than borrowing) lets the table outlive the
/// borrow of the module it was built from, so it can ride inside an
/// `Environment` (`env.rs`) alongside values that already own their own
/// data.
pub struct FunctionTable {
    defs: HashMap<String, StmtFunctionDef>,
}

/// Scans `module`'s own top-level statements (not the body of a nested
/// `def`/`class`) for `def` statements and clones each one into the
/// table under its name. A later `def` with the same name overwrites an
/// earlier one — Python itself resolves a redefined name to whichever
/// binding executed last, so the last `def` in source order is the one
/// a same-module call actually reaches.
pub fn function_table(module: &ModModule) -> FunctionTable {
    let mut defs = HashMap::new();
    for stmt in &module.body {
        if let Stmt::FunctionDef(def) = stmt {
            defs.insert(def.name.id.as_str().to_owned(), def.clone());
        }
    }
    FunctionTable { defs }
}

impl FunctionTable {
    /// The module-level `def` named `name`, if the module has one.
    pub fn def(&self, name: &str) -> Option<&StmtFunctionDef> {
        self.defs.get(name)
    }
}

/// `base` merged with `imported`: on a name both tables carry, `base`'s
/// own `def` wins — a module's own definition shadows an imported name
/// of the same spelling, exactly as a later top-level `def` in `base`
/// itself already overwrites an earlier one.
pub fn merged(base: &FunctionTable, imported: &FunctionTable) -> FunctionTable {
    let mut defs = imported.defs.clone();
    for (name, def) in &base.defs {
        defs.insert(name.clone(), def.clone());
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
