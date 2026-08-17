/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The pydantic surface: `type X = Annotated[int, Field(ge=…, le=…)]`
//! aliases lowered to refined sets, one table (plan-v2 L7).
//!
//! The lowering walks the RAW annotation expression, never the host's
//! resolved `Type`: pydantic `Field`'s stub returns `Any`, so
//! `Type::Annotated`'s metadata slot holds the inferred return type
//! and the `ge`/`le` values are unrecoverable from it
//! (PYREFLY-API-NOTES.md §3).

use std::collections::HashMap;
use std::collections::HashSet;

use refined_sets::refinement_forms::{
    Refinement, RefinedSet, above, at_least, at_most, below, integer, make_refined_set,
    multiple_of,
};
use ruff_python_ast::{Expr, ModModule, Number, Stmt, StmtImport, StmtImportFrom, UnaryOp};

/// Field kwargs that state nothing about the value set — safe to skip.
/// Any OTHER unrecognized kwarg refuses the whole alias: a constraint
/// this table cannot state must not silently widen or narrow the set.
const INERT_FIELD_KWARGS: &[&str] = &[
    "alias",
    "default",
    "description",
    "examples",
    "title",
];

/// Every `type X = Annotated[int|float, Field(…)]` alias at the
/// module's top level, lowered to its refined set, plus alias-of-alias
/// (`type Adult = Age`, where `Age` already named a compiled set).
/// Statements walk in source order so a later alias can point at an
/// earlier one. Aliases the table cannot lower faithfully are absent —
/// absence declines judgment, it never approximates.
pub fn compile_aliases(module: &ModModule) -> HashMap<String, RefinedSet> {
    let imports = surface_imports(module);
    let mut out = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::TypeAlias(alias) = stmt else {
            continue;
        };
        let Expr::Name(name) = alias.name.as_ref() else {
            continue;
        };
        let set = annotated_expression_set(alias.value.as_ref(), &imports).or_else(|| {
            // `type Adult = Age`: the RHS is a bare name that already
            // names a compiled set in this same table.
            let Expr::Name(rhs) = alias.value.as_ref() else {
                return None;
            };
            out.get(rhs.id.as_str()).cloned()
        });
        if let Some(set) = set {
            out.insert(name.id.as_str().to_owned(), set);
        }
    }
    out
}

/// `Annotated[int|float, Field(…), …]` → the stated set, resolved
/// against the module's import identities. The `Annotated` head name
/// must itself resolve to an import of `typing.Annotated` (or
/// `typing_extensions.Annotated`) — a bare `Annotated` that was never
/// imported is not recognized. The `int` sort carries the integer form
/// (int ≠ float is a product law); every metadata element must be a
/// recognized `Field(…)` call (by import identity, not spelling) or
/// the alias refuses.
pub fn annotated_expression_set(value: &Expr, imports: &SurfaceImports) -> Option<RefinedSet> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !imports.annotated_names.contains(head.id.as_str()) {
        return None;
    }
    let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
        return None;
    };
    let (base, metadata) = arguments.elts.split_first()?;
    let mut forms: Vec<Refinement> = match base {
        Expr::Name(sort) if sort.id.as_str() == "int" => vec![integer()],
        Expr::Name(sort) if sort.id.as_str() == "float" => vec![],
        _ => return None,
    };
    for meta in metadata {
        let Expr::Call(call) = meta else {
            return None;
        };
        if !names_field(call.func.as_ref(), imports) {
            return None;
        }
        for keyword in call.arguments.keywords.iter() {
            let name = keyword.arg.as_ref()?;
            match name.as_str() {
                "ge" => forms.push(at_least(literal_number(&keyword.value)?)),
                "gt" => forms.push(above(literal_number(&keyword.value)?)),
                "le" => forms.push(at_most(literal_number(&keyword.value)?)),
                "lt" => forms.push(below(literal_number(&keyword.value)?)),
                "multiple_of" => forms.push(multiple_of(literal_number(&keyword.value)?)),
                other if INERT_FIELD_KWARGS.contains(&other) => {}
                _ => return None,
            }
        }
    }
    Some(make_refined_set(forms))
}

/// A metadata call names pydantic's `Field` when its callee is either
/// a bare name that imports resolved to `Field`, or an attribute whose
/// base is a name that imports resolved to the pydantic module and
/// whose attribute is literally `Field`. A `Field` defined locally or
/// imported from any other module never matches either shape.
fn names_field(func: &Expr, imports: &SurfaceImports) -> bool {
    match func {
        Expr::Name(n) => imports.field_names.contains(n.id.as_str()),
        Expr::Attribute(a) => {
            a.attr.as_str() == "Field"
                && matches!(a.value.as_ref(), Expr::Name(base) if imports.pydantic_modules.contains(base.id.as_str()))
        }
        _ => false,
    }
}

/// The import identities the surface resolves names against: which
/// local names mean pydantic's `Field`, which local names mean the
/// pydantic module itself, and which local names mean `Annotated`
/// (from `typing` or `typing_extensions`).
pub struct SurfaceImports {
    field_names: HashSet<String>,
    pydantic_modules: HashSet<String>,
    annotated_names: HashSet<String>,
}

/// Reads the module's top-level `import`/`from … import …` statements
/// and records the local names that mean pydantic's `Field`, the
/// pydantic module, and `Annotated`. Only the two shapes named in the
/// mission are recognized: `import pydantic[ as x]`,
/// `from pydantic import Field[ as x]`, and the same two shapes for
/// `Annotated` from `typing`/`typing_extensions`. Anything else (a
/// `fields`-style submodule import, a re-export) is out of scope and
/// leaves the corresponding set empty.
pub fn surface_imports(module: &ModModule) -> SurfaceImports {
    let mut field_names = HashSet::new();
    let mut pydantic_modules = HashSet::new();
    let mut annotated_names = HashSet::new();
    for stmt in module.body.iter() {
        match stmt {
            Stmt::Import(StmtImport { names, .. }) => {
                for alias in names {
                    if alias.name.id.as_str() == "pydantic" {
                        let local = alias.asname.as_ref().unwrap_or(&alias.name);
                        pydantic_modules.insert(local.id.as_str().to_owned());
                    }
                }
            }
            Stmt::ImportFrom(StmtImportFrom {
                module: Some(source),
                names,
                level: 0,
                ..
            }) => {
                for alias in names {
                    let local = alias.asname.as_ref().unwrap_or(&alias.name);
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "Field" {
                        field_names.insert(local.id.as_str().to_owned());
                    }
                    if (source.id.as_str() == "typing" || source.id.as_str() == "typing_extensions")
                        && alias.name.id.as_str() == "Annotated"
                    {
                        annotated_names.insert(local.id.as_str().to_owned());
                    }
                }
            }
            _ => {}
        }
    }
    SurfaceImports {
        field_names,
        pydantic_modules,
        annotated_names,
    }
}

/// A numeric literal, with unary minus — the readable-RHS gate for
/// this slice. None anywhere else (an unread value declines, it never
/// guesses).
pub fn literal_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(f) => Some(*f),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
            Some(-literal_number(unary.operand.as_ref())?)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(source: &str) -> ModModule {
        ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax()
    }

    /// `Field as F` — the import alias is still recognized as `Field`.
    #[test]
    fn field_import_alias_recognized() {
        let module = parsed(
            "from pydantic import Field as F\n\
             from typing import Annotated\n\
             type Age = Annotated[int, F(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(out.contains_key("Age"));
    }

    /// `import pydantic as p` + `p.Field(...)` — the module alias is
    /// still recognized as the pydantic module.
    #[test]
    fn pydantic_module_alias_recognized() {
        let module = parsed(
            "import pydantic as p\n\
             from typing import Annotated\n\
             type Age = Annotated[int, p.Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(out.contains_key("Age"));
    }

    /// A locally defined `Field` shadowing the name is never a pydantic
    /// `Field` — no alias compiles.
    #[test]
    fn locally_defined_field_not_recognized() {
        let module = parsed(
            "from typing import Annotated\n\
             def Field(**kwargs):\n\
             \x20\x20\x20\x20pass\n\
             type Age = Annotated[int, Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    /// `from mylib import Field` — a same-named import from any other
    /// module is never recognized as pydantic's `Field`.
    #[test]
    fn field_from_other_module_not_recognized() {
        let module = parsed(
            "from mylib import Field\n\
             from typing import Annotated\n\
             type Age = Annotated[int, Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    /// `Annotated` used bare, with no import naming it, is never
    /// recognized.
    #[test]
    fn annotated_without_import_not_recognized() {
        let module = parsed(
            "from pydantic import Field\n\
             type Age = Annotated[int, Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    /// `type Adult = Age` — alias-of-alias compiles both names to the
    /// same set.
    #[test]
    fn alias_of_alias_compiles_both_names() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Age = Annotated[int, Field(ge=0)]\n\
             type Adult = Age\n",
        );
        let out = compile_aliases(&module);
        assert!(out.contains_key("Age"));
        assert!(out.contains_key("Adult"));
        assert_eq!(out.get("Age"), out.get("Adult"));
    }
}
