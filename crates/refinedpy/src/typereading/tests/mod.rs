use std::collections::HashSet;

use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::union;
use ruff_python_ast::ExprContext;
use ruff_python_ast::ExprName;
use ruff_python_ast::ExprStringLiteral;
use ruff_python_ast::StringLiteralValue;
use ruff_text_size::TextRange;

use super::*;

mod alias_resolution;
mod aliased_sequence;
mod typed_dict;
mod none_and_optional;
mod literal;
mod callable_return;
mod dict_value;
mod generator;

pub(super) fn name_expr(id: &str) -> Expr {
    Expr::Name(ExprName {
        node_index: Default::default(),
        range: TextRange::default(),
        id: id.into(),
        ctx: ExprContext::Load,
    })
}

pub(super) fn string_literal_expr(text: &str) -> Expr {
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

pub(super) fn no_locals() -> Environment {
    Environment::new(HashSet::new())
}

/// SurfaceImports has private fields — tests build it through the
/// public constructor on a parsed module with no imports.
pub(super) fn no_imports() -> SurfaceImports {
    let module = ruff_python_parser::parse_module("x = 1")
        .expect("test module parses")
        .into_syntax();
    crate::surface::surface_imports(&module)
}

pub(super) fn none_literal_expr() -> Expr {
    Expr::NoneLiteral(ruff_python_ast::ExprNoneLiteral {
        node_index: Default::default(),
        range: TextRange::default(),
    })
}

pub(super) fn bin_or(left: Expr, right: Expr) -> Expr {
    Expr::BinOp(ruff_python_ast::ExprBinOp {
        node_index: Default::default(),
        range: TextRange::default(),
        left: Box::new(left),
        op: ruff_python_ast::Operator::BitOr,
        right: Box::new(right),
    })
}

/// `<other> | None`.
pub(super) fn none_union(other: Expr) -> Expr {
    bin_or(other, none_literal_expr())
}

/// `None | <other>` — the reversed side.
pub(super) fn union_none(other: Expr) -> Expr {
    bin_or(none_literal_expr(), other)
}

/// `Optional[<other>]`.
pub(super) fn optional_of(other: Expr) -> Expr {
    Expr::Subscript(ruff_python_ast::ExprSubscript {
        node_index: Default::default(),
        range: TextRange::default(),
        value: Box::new(name_expr("Optional")),
        slice: Box::new(other),
        ctx: ExprContext::Load,
    })
}

pub(super) fn age_aliases() -> HashMap<String, AliasEntry> {
    let mut aliases = HashMap::new();
    aliases.insert(
        "Age".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(0.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    aliases.insert(
        "Label".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(1.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    aliases
}

/// The parsed module's one `AnnAssign`'s annotation — shared across
/// the none/optional/literal/dict sections.
pub(super) fn annotated_or_none_annotation(module: &ruff_python_ast::ModModule) -> &Expr {
    for stmt in module.body.iter() {
        if let ruff_python_ast::Stmt::AnnAssign(ann_assign) = stmt {
            return ann_assign.annotation.as_ref();
        }
    }
    panic!("test module has one AnnAssign statement");
}

/// The parsed module's one top-level `def`'s own `-> Annotation` —
/// this section's own twin of `annotated_or_none_annotation` for a
/// return-typed function rather than an `AnnAssign`.
pub(super) fn def_return_annotation(module: &ruff_python_ast::ModModule) -> &Expr {
    for stmt in module.body.iter() {
        if let ruff_python_ast::Stmt::FunctionDef(def) = stmt {
            return def.returns.as_deref().expect("test def carries a return annotation");
        }
    }
    panic!("test module has one top-level def");
}
