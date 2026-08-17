/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Annotation expressions read into declared refinements: alias names
//! (only where visible), inline `Annotated[...]` forms, alias-of-alias,
//! `X | None` / `Optional[X]` (admits-None wrapping the inner read),
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
    /// true when the annotation admits None alongside the set (`X |
    /// None`, `Optional[X]`) — Kind::Null judges Silent against such a
    /// declaration; the set describes the non-None values.
    pub admits_none: bool,
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
                admits_none: false,
            })
        }
        // `Optional[X]` reads X through the ordinary path and marks the
        // result as admitting None — "Optional[X] is equivalent to X |
        // None" (tmp/cpython Doc/library/typing.rst, "Optional").
        // Recognized by bare name only: `SurfaceImports` carries no
        // `typing.Optional` import identity today (only `Field` and
        // `Annotated`), so this matches the same bare-name convention
        // `annotated_expression_set` uses for `int`/`float`/`str` sort
        // names rather than gating on an identity this table cannot
        // check. `subscript.slice` is the single argument directly (not
        // a `Tuple` — ruff wraps in `Tuple` only for multi-element
        // subscripts like `Annotated[a, b]`). An inline `Annotated[...]`
        // form (the OTHER subscript head this table reads) is
        // unaffected: the surface unit owns recognizing and lowering it
        // against the module's import identities, and the spelling
        // there is the set itself, formatted, since no alias name
        // stands for it.
        Expr::Subscript(subscript) => {
            let is_optional = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Optional");
            if is_optional {
                let mut declared =
                    declared_refinement(subscript.slice.as_ref(), aliases, imports, environment)?;
                declared.admits_none = true;
                return Some(declared);
            }
            let set = annotated_expression_set(annotation, imports)?;
            let spelling = format_for_diagnostics(&set);
            Some(DeclaredRefinement {
                set,
                spelling,
                admits_none: false,
            })
        }
        // `X | None` / `None | X` (exactly one side a bare `None`
        // literal): read the OTHER side through the ordinary path and
        // mark the result as admitting None — the `X | None` union
        // syntax (tmp/cpython Doc/library/stdtypes.rst, "Union Type",
        // `types.UnionType`). `None | None` and a union of two non-None
        // shapes (`Age | Label`) are a different unit — a general union
        // of two sets — and decline here exactly as before.
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                // both None, or neither — not this table's one-sided form
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            let mut declared = declared_refinement(other, aliases, imports, environment)?;
            declared.admits_none = true;
            Some(declared)
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
        // Every other shape (a general union of two non-None sets,
        // attribute paths, calls, …) is not a form this table reads.
        // None here declines judgment; it is never read as "no
        // refinement applies".
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
    fn an_alias_name_not_in_the_table_states_nothing_even_as_one_side_of_a_none_union() {
        // `int | None`: `int` is not a compiled alias in this test's
        // table, so the inner read misses and the whole union declines
        // — the same "alias lookup miss" reason a bare `int` would.
        let union = none_union(name_expr("int"));
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    fn none_literal_expr() -> Expr {
        Expr::NoneLiteral(ruff_python_ast::ExprNoneLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
        })
    }

    fn bin_or(left: Expr, right: Expr) -> Expr {
        Expr::BinOp(ruff_python_ast::ExprBinOp {
            node_index: Default::default(),
            range: TextRange::default(),
            left: Box::new(left),
            op: ruff_python_ast::Operator::BitOr,
            right: Box::new(right),
        })
    }

    /// `<other> | None`.
    fn none_union(other: Expr) -> Expr {
        bin_or(other, none_literal_expr())
    }

    /// `None | <other>` — the reversed side.
    fn union_none(other: Expr) -> Expr {
        bin_or(none_literal_expr(), other)
    }

    /// `Optional[<other>]`.
    fn optional_of(other: Expr) -> Expr {
        Expr::Subscript(ruff_python_ast::ExprSubscript {
            node_index: Default::default(),
            range: TextRange::default(),
            value: Box::new(name_expr("Optional")),
            slice: Box::new(other),
            ctx: ExprContext::Load,
        })
    }

    fn age_aliases() -> HashMap<String, RefinedSet> {
        let mut aliases = HashMap::new();
        aliases.insert("Age".to_owned(), make_refined_set(vec![at_least(0.0)]));
        aliases.insert("Label".to_owned(), make_refined_set(vec![at_least(1.0)]));
        aliases
    }

    #[test]
    fn a_plain_alias_name_reads_with_admits_none_false() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&name_expr("Age"), &aliases, &imports, &environment)
            .expect("Age resolves");
        assert!(!got.admits_none);
    }

    #[test]
    fn age_or_none_reads_age_with_admits_none_true() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&none_union(name_expr("Age")), &aliases, &imports, &environment)
            .expect("Age | None resolves");
        assert_eq!(got.spelling, "Age");
        assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
        assert!(got.admits_none);
    }

    #[test]
    fn none_or_age_reversed_reads_age_with_admits_none_true() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&union_none(name_expr("Age")), &aliases, &imports, &environment)
            .expect("None | Age resolves");
        assert_eq!(got.spelling, "Age");
        assert!(got.admits_none);
    }

    #[test]
    fn optional_age_reads_age_with_admits_none_true() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&optional_of(name_expr("Age")), &aliases, &imports, &environment)
            .expect("Optional[Age] resolves");
        assert_eq!(got.spelling, "Age");
        assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
        assert!(got.admits_none);
    }

    #[test]
    fn age_or_label_a_union_of_two_non_none_sets_still_declines_whole() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let union = bin_or(name_expr("Age"), name_expr("Label"));
        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    #[test]
    fn none_or_none_declines_whole() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let union = bin_or(none_literal_expr(), none_literal_expr());
        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    /// `Annotated[int, Field(ge=0)] | None` — the recursion into the
    /// non-None side of a `| None` union reaches an inline `Annotated`
    /// form exactly as it would reach a bare alias name.
    #[test]
    fn annotated_or_none_reads_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             x: Annotated[int, Field(ge=0)] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Annotated[int, Field(ge=0)] | None resolves");
        assert!(got.admits_none);
        assert_eq!(got.set, make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(0.0)]));
    }

    /// `Optional[Annotated[int, Field(ge=0)]]` — the recursion into
    /// `Optional[...]`'s inner expression reaches the same inline
    /// `Annotated` form.
    #[test]
    fn optional_of_annotated_reads_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "from pydantic import Field\n\
             from typing import Annotated, Optional\n\
             x: Optional[Annotated[int, Field(ge=0)]] = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Optional[Annotated[int, Field(ge=0)]] resolves");
        assert!(got.admits_none);
        assert_eq!(got.set, make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(0.0)]));
    }

    /// The parsed module's one `AnnAssign`'s annotation — shared by the
    /// two nested-form tests above.
    fn annotated_or_none_annotation(module: &ruff_python_ast::ModModule) -> &Expr {
        for stmt in module.body.iter() {
            if let ruff_python_ast::Stmt::AnnAssign(ann_assign) = stmt {
                return ann_assign.annotation.as_ref();
            }
        }
        panic!("test module has one AnnAssign statement");
    }
}
