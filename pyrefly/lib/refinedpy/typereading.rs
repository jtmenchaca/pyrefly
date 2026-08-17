/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Annotation expressions read into declared refinements: alias names
//! (only where visible), inline `Annotated[...]` forms, alias-of-alias,
//! and string annotations. This file is the contract the walk calls;
//! the typereading unit fills it in behind these signatures.
//!
//! A None never approximates a set — it declines to state one, and the
//! walk decides what silence says at the call site. Nothing here widens
//! or guesses a set it cannot read exactly (the same discipline as
//! refined-ts-go's typereading package: refuse rather than approximate).

use std::collections::HashMap;

use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_parser::parse_expression;

use crate::refinedpy::env::Environment;
use crate::refinedpy::surface::SurfaceImports;
use crate::refinedpy::surface::annotated_expression_set;

/// A refinement an annotation states, with the spelling diagnostics
/// use for it (the alias name, or the formatted set for inline forms).
#[derive(Clone)]
pub struct DeclaredRefinement {
    pub set: RefinedSet,
    pub spelling: String,
}

/// The refinement this annotation states here, or None when it states
/// none this table can read. A None never approximates — it declines.
pub fn declared_refinement(
    annotation: &Expr,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    match annotation {
        // A bare name: only a visible module-level alias states anything.
        // A name the body rebinds locally means something else here, so
        // the alias table must not be consulted for it — the walk owns
        // saying why a rebound name reads as unrefined.
        Expr::Name(name) => {
            let spelling = name.id.as_str();
            if !environment.alias_is_visible(spelling) {
                return None;
            }
            let set = aliases.get(spelling)?;
            Some(DeclaredRefinement {
                set: set.clone(),
                spelling: spelling.to_owned(),
            })
        }
        // An inline `Annotated[...]` form: the surface unit owns
        // recognizing and lowering it against the module's import
        // identities. The spelling here is the set itself, formatted,
        // since there is no alias name standing for it.
        Expr::Subscript(_) => {
            let set = annotated_expression_set(annotation, imports)?;
            let spelling = format_for_diagnostics(&set);
            Some(DeclaredRefinement { set, spelling })
        }
        // A string annotation (a forward reference or a quoted form):
        // parse the string's own contents as an expression and recurse
        // on what it parses to. A string that fails to parse as an
        // expression states nothing this table can read.
        Expr::StringLiteral(literal) => {
            let source = literal.value.to_str();
            let parsed = parse_expression(source).ok()?;
            let inner = parsed.into_syntax().body;
            declared_refinement(&inner, aliases, imports, environment)
        }
        // Every other shape (unions, `Optional[...]`, attribute paths,
        // calls, …) is not a form this table reads. None here declines
        // judgment; it is never read as "no refinement applies".
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::make_refined_set;
    use ruff_python_ast::ExprContext;
    use ruff_python_ast::ExprName;
    use ruff_python_ast::ExprStringLiteral;
    use ruff_python_ast::StringLiteralValue;
    use ruff_text_size::TextRange;

    use super::*;

    fn name_expr(id: &str) -> Expr {
        Expr::Name(ExprName {
            node_index: Default::default(),
            range: TextRange::default(),
            id: id.into(),
            ctx: ExprContext::Load,
        })
    }

    fn string_literal_expr(text: &str) -> Expr {
        Expr::StringLiteral(ExprStringLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
            value: StringLiteralValue::single(ruff_python_ast::StringLiteral {
                range: TextRange::default(),
                node_index: Default::default(),
                value: text.into(),
                flags: ruff_python_ast::StringLiteralFlags::empty(),
            }),
        })
    }

    fn no_locals() -> Environment {
        Environment::new(HashSet::new())
    }

    /// SurfaceImports has private fields — tests build it through the
    /// public constructor on a parsed module with no imports.
    fn no_imports() -> SurfaceImports {
        let module = ruff_python_parser::parse_module("x = 1")
            .expect("test module parses")
            .into_syntax();
        crate::refinedpy::surface::surface_imports(&module)
    }

    #[test]
    fn a_visible_alias_name_resolves_with_its_name_as_spelling() {
        let mut aliases = HashMap::new();
        aliases.insert("PositiveInt".to_owned(), make_refined_set(vec![at_least(1.0)]));
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment)
            .expect("a visible alias resolves");
        assert_eq!(got.spelling, "PositiveInt");
        assert_eq!(got.set, make_refined_set(vec![at_least(1.0)]));
    }

    #[test]
    fn a_locally_rebound_alias_name_states_nothing() {
        let mut aliases = HashMap::new();
        aliases.insert("PositiveInt".to_owned(), make_refined_set(vec![at_least(1.0)]));
        let imports = no_imports();
        let mut locally_bound = HashSet::new();
        locally_bound.insert("PositiveInt".to_owned());
        let environment = Environment::new(locally_bound);

        let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    #[test]
    fn a_string_annotation_naming_a_visible_alias_resolves() {
        let mut aliases = HashMap::new();
        aliases.insert("PositiveInt".to_owned(), make_refined_set(vec![at_least(1.0)]));
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(
            &string_literal_expr("PositiveInt"),
            &aliases,
            &imports,
            &environment,
        )
        .expect("a string annotation naming a visible alias resolves");
        assert_eq!(got.spelling, "PositiveInt");
    }

    #[test]
    fn an_unreadable_annotation_states_nothing() {
        // `int | None` is a BinOp, not a form this table reads.
        let left = name_expr("int");
        let right = Expr::NoneLiteral(ruff_python_ast::ExprNoneLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
        });
        let union = Expr::BinOp(ruff_python_ast::ExprBinOp {
            node_index: Default::default(),
            range: TextRange::default(),
            left: Box::new(left),
            op: ruff_python_ast::Operator::BitOr,
            right: Box::new(right),
        });
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }
}
