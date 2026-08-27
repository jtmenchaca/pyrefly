//! The fields an explicit `__init__` derives, and the parent fields a
//! `super().__init__(...)` call links onward — the construction-order
//! half of `class_model_of`'s field merge.

use std::sync::Arc;

use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{Expr, Stmt, StmtFunctionDef};

use crate::env::Environment;
use crate::surface::{AliasEntry, SurfaceImports};
use crate::typereading::declared_refinement;

use super::super::fields::self_attribute_name;
use super::defaults::default_value_of;
use super::types::ClassField;

/// The fields an explicit `def __init__(self, ...)` derives — `None`
/// when `__init__`'s own parameter shape is outside what this reads
/// (`*args`/`**kwargs`/keyword-only parameters, or no parameter at all
/// beyond `self`). Two passes over `__init__`'s TOP-LEVEL body:
///
/// 1. Every `self.<name> = <expr>` / `self.<name>: <Ann> = <expr>`
///    statement becomes one field, classified by its RHS below — a
///    parameter-flowing write, a readable literal, or an unreadable
///    expression.
/// 2. The RESULT VECTOR is then reordered so a parameter-flowing
///    field sits at that parameter's own position (datamodel.rst,
///    "Classes"/`object.__init__` — construction arguments are
///    `__init__`'s own arguments, so parameter position IS
///    construction position); a parameter the body never forwards
///    bare still occupies its slot as its own field (annotation/
///    default read from the PARAMETER, no write needed to justify a
///    position existing); every non-parameter-flowing write (a
///    literal-default write, or an unreadable-RHS write) is appended
///    after every parameter position, in the body's own write order —
///    it has no construction slot of its own, but is still a field
///    `field_read`/`field_write_judgment` can reach by name.
pub(super) fn init_derived_fields(
    init: &StmtFunctionDef,
    aliases: &std::collections::HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<ClassField>> {
    if init.parameters.vararg.is_some() || init.parameters.kwarg.is_some() || !init.parameters.kwonlyargs.is_empty() {
        return None;
    }
    let parameters: Vec<_> = init
        .parameters
        .posonlyargs
        .iter()
        .chain(init.parameters.args.iter())
        .collect();
    // the first parameter is `self` by convention; a `def __init__`
    // with no parameters at all is not a bound instance method this
    // reader can pair a `self.<name>` target against.
    let (_self_param, rest) = parameters.split_first()?;
    let empty_environment = Environment::new(Default::default());

    let mut parameter_slots: Vec<Option<ClassField>> = vec![None; rest.len()];
    let mut trailing: Vec<ClassField> = Vec::new();

    for stmt in &init.body {
        let Some((field_name, value_expr)) = self_write_target(stmt) else {
            continue;
        };
        let matched_parameter = rest
            .iter()
            .position(|parameter| is_bare_name(value_expr, parameter.parameter.name.id.as_str()));

        match matched_parameter {
            // <expr> is a bare Name matching one of __init__'s own
            // parameters: the field's value flows from that
            // parameter — declared/default read from the PARAMETER's
            // own annotation/default, and the field takes that
            // parameter's construction POSITION.
            Some(index) => {
                let parameter = rest[index];
                let declared = parameter
                    .parameter
                    .annotation
                    .as_deref()
                    .and_then(|annotation| declared_refinement(annotation, aliases, imports, &empty_environment));
                let default = parameter
                    .default
                    .as_deref()
                    .map(|default_expr| default_value_of(default_expr, &empty_environment, kernel));
                let base_sort = parameter
                    .parameter
                    .annotation
                    .as_deref()
                    .and_then(crate::typereading::base_sort_return_refinement);
                parameter_slots[index] = Some(ClassField { name: field_name, declared, default, base_sort });
            }
            None => {
                // AnnAssign carries its own annotation even when the
                // RHS is not a bare parameter name.
                let declared = match stmt {
                    Stmt::AnnAssign(assign) => {
                        declared_refinement(assign.annotation.as_ref(), aliases, imports, &empty_environment)
                    }
                    _ => None,
                };
                let base_sort = match stmt {
                    Stmt::AnnAssign(assign) => {
                        crate::typereading::base_sort_return_refinement(assign.annotation.as_ref())
                    }
                    _ => None,
                };
                let literal = default_value_of(value_expr, &empty_environment, kernel);
                let is_readable = literal.kind != Kind::Unknown;
                trailing.push(ClassField {
                    name: field_name,
                    declared,
                    base_sort,
                    // <expr> is anything evaluate_expression answers
                    // (a literal): the field's default is that
                    // constant. Any other RHS: declared: None,
                    // default: None (unless an AnnAssign carried its
                    // own annotation, read above).
                    default: is_readable.then_some(literal),
                });
            }
        }
    }

    // a parameter the body never forwards bare still occupies its own
    // construction slot, carrying its own annotation/default.
    for (index, parameter) in rest.iter().enumerate() {
        if parameter_slots[index].is_some() {
            continue;
        }
        let declared = parameter
            .parameter
            .annotation
            .as_deref()
            .and_then(|annotation| declared_refinement(annotation, aliases, imports, &empty_environment));
        let default = parameter
            .default
            .as_deref()
            .map(|default_expr| default_value_of(default_expr, &empty_environment, kernel));
        let base_sort = parameter
            .parameter
            .annotation
            .as_deref()
            .and_then(crate::typereading::base_sort_return_refinement);
        parameter_slots[index] = Some(ClassField {
            name: parameter.parameter.name.id.as_str().to_owned(),
            declared,
            default,
            base_sort,
        });
    }

    let mut fields: Vec<ClassField> = parameter_slots.into_iter().map(|slot| slot.expect("filled above")).collect();
    fields.extend(trailing);
    Some(fields)
}

/// The parent-derived fields a child's `__init__` inherits through a
/// TOP-LEVEL `super().__init__(<args>)` call statement — `Vec::new()`
/// when `__init__`'s body carries no such statement. `super()` is
/// implemented as part of the binding process for explicit dotted
/// attribute lookups (`tmp/cpython/Doc/library/functions.rst`, the
/// `super()` entry — "a typical superclass call looks like this:
/// `super().method(arg)`… This does the same thing as `super(C,
/// self).method(arg)`"), so `super().__init__(<args>)` recognized
/// syntactically (an `Expr::Call` whose `func` is `Attribute { value:
/// Call(bare `super`, no args), attr: "__init__" }`) is `object`'s own
/// datamodel entry again: the arguments passed are the parent's
/// `__init__`'s own construction arguments.
///
/// Arguments map onto `parent_fields` POSITIONALLY, exactly the way
/// `judge_construction` maps a construction call's own positional
/// arguments — a bare Name argument that names one of the CHILD
/// `__init__`'s own parameters carries THAT parameter's annotation
/// forward as the field's own declared refinement (the child-parameter
/// linkage the mission specifies, so a later child construction
/// argument still flows to this field, matching how
/// `init_derived_fields` reads a parameter-flowing self-write); the
/// PARENT field's own default survives when the child parameter states
/// none of its own. Anything else (a literal, an unreadable
/// expression, or no `super()` call at all) keeps the PARENT field's
/// own declared/default shape unlinked. More `super().__init__(...)`
/// arguments than `parent_fields` — an extra-argument call the
/// parent's own `__init__` would reject at runtime — is not modeled;
/// parent fields answer unlinked, same as the no-match case.
pub(super) fn super_init_fields(
    init: &StmtFunctionDef,
    parent_fields: &[ClassField],
    aliases: &std::collections::HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
) -> Vec<ClassField> {
    let Some(call) = super_init_call(init) else {
        return Vec::new();
    };
    if call.arguments.args.len() > parent_fields.len() {
        return parent_fields.to_vec();
    }
    let child_parameters: Vec<_> = init
        .parameters
        .posonlyargs
        .iter()
        .chain(init.parameters.args.iter())
        .collect();
    let empty_environment = Environment::new(Default::default());

    parent_fields
        .iter()
        .enumerate()
        .map(|(index, parent_field)| {
            let Some(argument) = call.arguments.args.get(index) else {
                return parent_field.clone();
            };
            // a bare Name argument naming one of the CHILD __init__'s
            // own parameters: the field's declared refinement now
            // flows from the CHILD's parameter (a construction
            // argument at this position reaches the parent field
            // through it), keeping the PARENT field's own name (the
            // field is still reached as `field_read(instance,
            // parent_field.name)`) and default (the parent's own
            // default still answers a caller who omits the argument).
            let Expr::Name(name) = argument else {
                return parent_field.clone();
            };
            let Some(child_parameter) =
                child_parameters.iter().find(|p| p.parameter.name.id.as_str() == name.id.as_str())
            else {
                return parent_field.clone();
            };
            let declared = child_parameter
                .parameter
                .annotation
                .as_deref()
                .and_then(|annotation| declared_refinement(annotation, aliases, imports, &empty_environment))
                .or_else(|| parent_field.declared.clone());
            let base_sort = child_parameter
                .parameter
                .annotation
                .as_deref()
                .and_then(crate::typereading::base_sort_return_refinement)
                .or_else(|| parent_field.base_sort.clone());
            ClassField {
                name: parent_field.name.clone(),
                declared,
                default: parent_field.default.clone(),
                base_sort,
            }
        })
        .collect()
}

/// The class body's TOP-LEVEL `super().__init__(<args>)` call
/// statement, if `__init__`'s own body carries exactly that shape
/// (an `Expr` statement whose value is a `Call` on `Attribute { value:
/// a bare, no-argument `Call` to the name `super`, attr: "__init__" }`).
fn super_init_call(init: &StmtFunctionDef) -> Option<&ruff_python_ast::ExprCall> {
    init.body.iter().find_map(|stmt| {
        let Stmt::Expr(expr_stmt) = stmt else {
            return None;
        };
        let Expr::Call(call) = expr_stmt.value.as_ref() else {
            return None;
        };
        let Expr::Attribute(attribute) = call.func.as_ref() else {
            return None;
        };
        if attribute.attr.as_str() != "__init__" {
            return None;
        }
        let Expr::Call(super_call) = attribute.value.as_ref() else {
            return None;
        };
        let Expr::Name(super_name) = super_call.func.as_ref() else {
            return None;
        };
        if super_name.id.as_str() != "super" || !super_call.arguments.args.is_empty() {
            return None;
        }
        Some(call)
    })
}

/// A top-level `self.<name> = <expr>` or `self.<name>: <Ann> = <expr>`
/// statement's field name and RHS value expression — `None` for
/// anything else (a non-`self` target, a destructuring target, a
/// statement that is not one of these two assignment forms). Only a
/// SINGLE `Attribute` target on `Assign` is read (`self.a = self.b =
/// v` is a multi-target assign this function does not pair to one
/// field) — `self.<name>` chained targets are rare enough in the
/// corpus band this file serves that reading only the single-target
/// form is the honest scope, not a guess.
fn self_write_target(stmt: &Stmt) -> Option<(String, &Expr)> {
    match stmt {
        Stmt::Assign(assign) => {
            let [target] = assign.targets.as_slice() else {
                return None;
            };
            let name = self_attribute_name(target)?;
            Some((name, assign.value.as_ref()))
        }
        Stmt::AnnAssign(assign) => {
            let name = self_attribute_name(assign.target.as_ref())?;
            let value_expr = assign.value.as_deref()?;
            Some((name, value_expr))
        }
        _ => None,
    }
}

/// Whether `expr` is exactly the bare name `name` — the "this
/// self-write forwards that parameter untouched" test.
pub(super) fn is_bare_name(expr: &Expr, name: &str) -> bool {
    matches!(expr, Expr::Name(n) if n.id.as_str() == name)
}
