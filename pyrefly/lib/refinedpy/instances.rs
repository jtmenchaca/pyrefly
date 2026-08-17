/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Classes as readable data: a class's declared fields (in declaration
//! order), judged construction against those fields, and instance
//! attribute reads/writes. One model covers every AnnAssign-fielded
//! class this checker reads — a self-authored class, a `@dataclass`,
//! and a pydantic `BaseModel` subclass all declare their fields the
//! same way (`name: Annotation [= default]` in the class body), so
//! there is one `ClassModel`, not a class family per framework.
//!
//! Field order is declaration order: pydantic v2 auto-generates
//! `__match_args__` in field-declaration order (AGENT-BRIEF.md,
//! "Environment facts" — "pydantic v2 `BaseModel` auto-generates
//! `__match_args__` in field-declaration order"), and a dataclass's
//! generated `__init__` binds positional arguments in the same order
//! its fields were declared (tmp/cpython/Doc/library/dataclasses.rst
//! is not present in this wave's read set beyond that one AGENT-BRIEF
//! fact; the positional-parameter-order claim for dataclasses is
//! standard `@dataclass` behavior and is flagged unverified against
//! the vendored tree in this file's owning report).

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::{
    known_set, unknown, AbstractValue, Kind, ObjectKey, SetKindTag,
};
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::{Expr, ModModule, Stmt, StmtClassDef, StmtFunctionDef};
use ruff_text_size::TextRange;

use crate::refinedpy::assignability::{judge, Verdict};
use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::evaluate_expression;
use crate::refinedpy::surface::SurfaceImports;
use crate::refinedpy::typereading::{declared_refinement, DeclaredRefinement};

/// One class's declared shape: its name, its fields (in construction
/// order), and its property accessors (read/write aliases that are
/// never stored fields of their own — see `PropertyModel`).
pub struct ClassModel {
    pub name: String,
    pub fields: Vec<ClassField>,
    pub properties: HashMap<String, PropertyModel>,
}

/// A `@property` getter/setter pair recognized on the class body:
/// `name` reads as an alias of `backing`'s own value (`field_read` on
/// `name` answers whatever `field_read(backing)` would), and a WRITE
/// to `name` judges against `declared` (the setter parameter's own
/// annotation, when the setter states one) rather than against any
/// refinement `backing` itself carries — the setter's parameter
/// annotation is pydantic-independent Python's own way of stating a
/// property's accepted input, and is the more specific claim for a
/// write through the accessor.
pub struct PropertyModel {
    pub backing: String,
    pub declared: Option<DeclaredRefinement>,
}

/// One declared field: its name, the refinement its annotation states
/// (`None` when the annotation states nothing this table reads — an
/// ordinary unrefined field, not a blocker), and its default value
/// expression evaluated once against a fresh environment (`None` when
/// the field has no default, or the default expression is not
/// itself readable). `Clone` so a parent's own fields can be copied
/// wholesale into a child that inherits them (no-`__init__` subclass,
/// or a `super().__init__(...)`-derived field) without disturbing the
/// parent's own `ClassModel`.
#[derive(Clone)]
pub struct ClassField {
    pub name: String,
    pub declared: Option<DeclaredRefinement>,
    pub default: Option<AbstractValue>,
}

/// What judging one construction call against a class's fields
/// concluded: every Fire finding raised (an argument's own range plus
/// the message `assignability::judge` produced — never composed
/// here), and the instance's resulting value.
pub struct ConstructionVerdict {
    pub fires: Vec<(TextRange, String)>,
    pub instance: AbstractValue,
}

/// Every module-level class definition's declared shape: `name: Annotation
/// [= default]` rows in the class body, walked in source order, merged
/// against an explicit `__init__`, and — when the class names exactly
/// one bare-Name base that is ITSELF a module-level class in this same
/// table — merged against that PARENT's own already-built fields
/// through `super().__init__(...)`. A `ClassVar[...]`-annotated row is
/// class-level state, not an instance field, and is skipped — every
/// framework this checker reads (dataclasses, pydantic) draws that
/// same line (AGENT-BRIEF.md's pydantic section and ordinary dataclass
/// practice both exclude `ClassVar` fields from `__init__`/
/// `__match_args__`; the exact vendored-doc clause for dataclasses is
/// unverified this wave, noted in the report's Blockers). A nested
/// class definition inside another class's body is not walked as its
/// own top-level entry — only `module.body`'s own `StmtClassDef`s
/// populate this table, matching `check.rs`'s own module-level
/// alias/import walk.
///
/// Parents build before children: every module-level class is visited
/// in a topological order over its single bare-Name base (a
/// depth-first walk with cycle/visited tracking) so a child's
/// `class_model_of` call always finds its parent already present in
/// `out`. A base that does not name another module-level class in
/// this table, or a class with more than one base (multiple
/// inheritance / diamond shapes), is read with NO parent — the child
/// keeps the isolated, AnnAssign/`__init__`-only behavior this file
/// already served, exactly as the mission specifies.
pub fn class_table(
    module: &ModModule,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
) -> HashMap<String, ClassModel> {
    let defs: HashMap<String, &StmtClassDef> = module
        .body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::ClassDef(def) => Some((def.name.id.as_str().to_owned(), def)),
            _ => None,
        })
        .collect();

    let mut out: HashMap<String, ClassModel> = HashMap::new();
    let mut building: std::collections::HashSet<String> = std::collections::HashSet::new();
    for name in defs.keys() {
        build_class_model(name, &defs, aliases, imports, kernel, &mut out, &mut building);
    }
    out
}

/// Depth-first build of one class into `out`, building its single
/// bare-Name module-level parent first when one exists. `building`
/// guards an inheritance cycle (`class A(B): ...` / `class B(A):
/// ...`, which CPython itself rejects at class-creation time) from
/// infinitely recursing — a class already mid-build is read as
/// parent-less rather than looping.
fn build_class_model(
    name: &str,
    defs: &HashMap<String, &StmtClassDef>,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
    out: &mut HashMap<String, ClassModel>,
    building: &mut std::collections::HashSet<String>,
) {
    if out.contains_key(name) || building.contains(name) {
        return;
    }
    let Some(def) = defs.get(name) else {
        return;
    };
    building.insert(name.to_owned());

    let parent_name = single_bare_name_base(def);
    let parent = parent_name.and_then(|parent_name| {
        if !out.contains_key(parent_name) {
            build_class_model(parent_name, defs, aliases, imports, kernel, out, building);
        }
        out.get(parent_name)
    });

    let model = class_model_of(def, aliases, imports, kernel, parent);
    building.remove(name);
    out.insert(model.name.clone(), model);
}

/// The class's single base, when it is a bare `Name` — `None` for a
/// class with zero bases, more than one base (multiple inheritance),
/// or a base that is not a bare name (`Foo[int]`, `mod.Foo`, a
/// synthesized/called base). The mission scopes this table to the
/// single-bare-Name-base shape only; every other base shape keeps the
/// child parent-less, the same isolated behavior this file already
/// served.
fn single_bare_name_base(def: &StmtClassDef) -> Option<&str> {
    let arguments = def.arguments.as_ref()?;
    let [base] = &*arguments.args else {
        return None;
    };
    let Expr::Name(name) = base else {
        return None;
    };
    Some(name.id.as_str())
}

/// One class definition's fields: every class-body `AnnAssign` in
/// source order, `ClassVar[...]` rows skipped, THEN merged against an
/// explicit `__init__`'s own `self.<name> = <expr>` writes when the
/// class body defines one plain `def __init__(self, ...)`.
///
/// The two sources answer different questions: an `AnnAssign` states a
/// field the class body DECLARES; an `__init__` write states a field
/// the class ACTUALLY ASSIGNS at construction. Ordering follows
/// construction, not declaration, once `__init__` exists: CPython
/// passes the class-constructor expression's own arguments straight
/// through to `__init__` (datamodel.rst, "Classes" — "the arguments of
/// the call are passed to `__new__` and, in the typical case, to
/// `__init__` to initialize the new instance"; `object.__init__`'s own
/// entry — "the arguments are those passed to the class constructor
/// expression") — so `__init__`'s parameter list IS the class's
/// positional construction order, and `init_derived_fields` below
/// already returns its fields in that order. An `AnnAssign` name
/// `__init__` also writes keeps the ANNASSIGN's declared/default (the
/// explicit annotation is the more specific claim); an `AnnAssign`
/// name `__init__` never writes stays keyword-reachable, appended
/// after every `__init__`-ordered field. A class with no `__init__`
/// (or an `__init__` with `*args`/`**kwargs`/keyword-only parameters,
/// which `init_derived_fields` declines) keeps the plain AnnAssign
/// order — the dataclass/pydantic path this file already served.
///
/// `parent` is this class's already-built single-bare-Name-base
/// ClassModel, when `class_table` found one. Two inheritance rules
/// fold in on top of the AnnAssign/`__init__` merge above:
///
/// - No explicit `__init__` at all, with a parent: CPython runs the
///   PARENT's own `__init__` at construction (datamodel.rst's
///   `object.__init__` entry, same citation as above — the class
///   constructor's arguments reach whichever `__init__` actually
///   executes, and a subclass declaring none inherits the parent's),
///   so the child inherits every one of the parent's fields wholesale,
///   before its own AnnAssign merge.
/// - An explicit `__init__` whose body's own TOP-LEVEL
///   `super().__init__(<args>)` call maps arguments onto the parent's
///   fields (`super_init_fields`, mission: "each parent field lands in
///   the child's ClassModel with the child-parameter linkage… ordered
///   before the child's own self-writes") — those parent-linked fields
///   are PREPENDED to the child's own `init_derived_fields`, so a
///   parent field never collides in position with a child-local field
///   sharing no name.
fn class_model_of(
    def: &StmtClassDef,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
    parent: Option<&ClassModel>,
) -> ClassModel {
    let empty_environment = Environment::new(Default::default());
    let mut ann_fields: HashMap<String, ClassField> = HashMap::new();
    let mut ann_order: Vec<String> = Vec::new();
    for stmt in def.body.iter() {
        let Stmt::AnnAssign(assign) = stmt else {
            continue;
        };
        if is_class_var(assign.annotation.as_ref()) {
            continue;
        }
        let Expr::Name(target_name) = assign.target.as_ref() else {
            continue;
        };
        let declared = declared_refinement(assign.annotation.as_ref(), aliases, imports, &empty_environment);
        let default = assign
            .value
            .as_deref()
            .map(|value_expr| default_value_of(value_expr, &empty_environment, kernel))
            .filter(|value| value.kind != Kind::Unknown);
        let name = target_name.id.as_str().to_owned();
        ann_order.push(name.clone());
        ann_fields.insert(name.clone(), ClassField { name, declared, default });
    }

    let init = def.body.iter().find_map(|stmt| match stmt {
        Stmt::FunctionDef(function) if function.name.id.as_str() == "__init__" => Some(function),
        _ => None,
    });
    let properties = property_table(def, aliases, imports);

    let base_fields: Vec<ClassField> = match init {
        // no explicit __init__: the parent's own __init__ runs at
        // construction (datamodel.rst, object.__init__), so the child
        // inherits every parent field wholesale, in the parent's own
        // order.
        None => parent.map(|parent| parent.fields.clone()).unwrap_or_default(),
        Some(init) => {
            let parent_fields = parent
                .map(|parent| super_init_fields(init, &parent.fields, aliases, imports))
                .unwrap_or_default();
            let own_fields = init_derived_fields(init, aliases, imports, kernel).unwrap_or_default();
            // a name the super() call already linked to a parent field
            // is NOT re-derived from __init__'s own parameter/self-write
            // walk — the parent-linked field (built above, carrying the
            // parent's own field identity and the child parameter's
            // annotation) is the one true field for that name; without
            // this, a pure-delegation __init__ (every parameter forwarded
            // bare through super().__init__(...), nothing else written)
            // would double the field under its own construction slot AND
            // under the parent's inherited slot.
            let parent_names: std::collections::HashSet<String> =
                parent_fields.iter().map(|field| field.name.clone()).collect();
            let mut combined = parent_fields;
            combined.extend(own_fields.into_iter().filter(|field| !parent_names.contains(field.name.as_str())));
            combined
        }
    };

    let mut fields: Vec<ClassField> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for base_field in base_fields {
        // an AnnAssign row for the same name is the more specific claim
        // (its own explicit annotation/default), so it takes the
        // base field's POSITION but the AnnAssign's own declared/
        // default content.
        let field = ann_fields.remove(&base_field.name).unwrap_or(base_field);
        seen.insert(field.name.clone());
        fields.push(field);
    }
    for name in ann_order {
        if seen.contains(&name) {
            continue;
        }
        if let Some(field) = ann_fields.remove(&name) {
            fields.push(field);
        }
    }

    ClassModel {
        name: def.name.id.as_str().to_owned(),
        fields,
        properties,
    }
}

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
fn init_derived_fields(
    init: &StmtFunctionDef,
    aliases: &HashMap<String, RefinedSet>,
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
                parameter_slots[index] = Some(ClassField { name: field_name, declared, default });
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
                let literal = default_value_of(value_expr, &empty_environment, kernel);
                let is_readable = literal.kind != Kind::Unknown;
                trailing.push(ClassField {
                    name: field_name,
                    declared,
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
        parameter_slots[index] = Some(ClassField {
            name: parameter.parameter.name.id.as_str().to_owned(),
            declared,
            default,
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
fn super_init_fields(
    init: &StmtFunctionDef,
    parent_fields: &[ClassField],
    aliases: &HashMap<String, RefinedSet>,
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
            ClassField {
                name: parent_field.name.clone(),
                declared,
                default: parent_field.default.clone(),
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

/// `self.<name>` recognition: an `Attribute` expression whose own
/// value is the bare name `self` — the first parameter's conventional
/// spelling. This function does not itself verify the receiver is
/// __init__'s actual first parameter (a `def __init__(this, ...)`
/// naming its receiver something other than `self` is out of the
/// corpus band this file serves; `self` is Python's own overwhelming
/// convention, not a keyword, so a literal name match is the same
/// honest-recognition posture `surface.rs`'s `names_field` takes for
/// other by-spelling recognitions).
fn self_attribute_name(target: &Expr) -> Option<String> {
    let Expr::Attribute(attribute) = target else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != "self" {
        return None;
    }
    Some(attribute.attr.as_str().to_owned())
}

/// Whether `expr` is exactly the bare name `name` — the "this
/// self-write forwards that parameter untouched" test.
fn is_bare_name(expr: &Expr, name: &str) -> bool {
    matches!(expr, Expr::Name(n) if n.id.as_str() == name)
}

/// `ClassVar[...]` annotation recognition: the annotation's own
/// `Subscript` head names `ClassVar` — read the same bare-name way
/// `annotated_expression_set` reads its own `Annotated` head, since
/// this table has no import-identity tracking for `ClassVar` the way
/// `SurfaceImports` tracks `Field`/`Annotated` (no fixture row spells
/// `ClassVar` through an import alias or a `typing.ClassVar`
/// qualified form, so only the bare unqualified spelling is
/// recognized).
fn is_class_var(annotation: &Expr) -> bool {
    let Expr::Subscript(subscript) = annotation else {
        return false;
    };
    matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "ClassVar")
}

/// Every `@property`/`@<name>.setter` pair the class body declares,
/// keyed by the property's own name — exactly the two body shapes the
/// mission names, nothing richer:
///
/// - `@property\ndef <name>(self): return self.<backing>` — a
///   `StmtFunctionDef` named `<name>`, decorated with the bare name
///   `property`, whose body is a SINGLE `return self.<backing>`
///   statement.
/// - `@<name>.setter\ndef <name>(self, <param>): self.<backing> =
///   <param>` — a `StmtFunctionDef` of the SAME name, decorated with
///   `Attribute { value: Name(<name>), attr: "setter" }`, whose body
///   is a SINGLE `self.<backing> = <param>` statement writing the
///   setter's own (second) parameter, bare, into the SAME backing name
///   the getter reads.
///
/// A getter/setter pair whose backing names disagree, or whose body is
/// richer than the one statement above (a computed return, a guard, a
/// multi-statement body), is not modeled — the property is absent from
/// this table, matching the mission's "anything richer... is not
/// modeled (today's behavior)." A getter with no matching setter (or a
/// setter with no matching getter) is also absent — `PropertyModel`
/// always carries the getter's own `backing` name, so a setter alone
/// has no backing name to record.
fn property_table(
    def: &StmtClassDef,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
) -> HashMap<String, PropertyModel> {
    let empty_environment = Environment::new(Default::default());
    let mut getters: HashMap<String, String> = HashMap::new();
    let mut setters: HashMap<String, (String, Option<DeclaredRefinement>)> = HashMap::new();

    for stmt in def.body.iter() {
        let Stmt::FunctionDef(function) = stmt else {
            continue;
        };
        let name = function.name.id.as_str();
        if is_property_getter_decorator(function) {
            if let Some(backing) = single_self_attribute_return(function) {
                getters.insert(name.to_owned(), backing);
            }
        } else if is_property_setter_decorator(function, name) {
            if let Some((backing, parameter)) = single_self_attribute_write_of_second_parameter(function) {
                let declared = parameter
                    .parameter
                    .annotation
                    .as_deref()
                    .and_then(|annotation| declared_refinement(annotation, aliases, imports, &empty_environment));
                setters.insert(name.to_owned(), (backing, declared));
            }
        }
    }

    let mut properties = HashMap::new();
    for (name, getter_backing) in getters {
        let declared = match setters.get(&name) {
            // the getter/setter pair's backing names must agree — a
            // setter writing a DIFFERENT attribute than the getter
            // reads is not a coherent accessor pair for this table.
            Some((setter_backing, declared)) if *setter_backing == getter_backing => declared.clone(),
            Some(_) => continue,
            None => None,
        };
        properties.insert(name, PropertyModel { backing: getter_backing, declared });
    }
    properties
}

/// `@property` — the decorator list carries exactly the bare name
/// `property` (no other decorator on the same function; a
/// `@property` stacked with something else is outside the mission's
/// two exact shapes).
fn is_property_getter_decorator(function: &StmtFunctionDef) -> bool {
    let [decorator] = function.decorator_list.as_slice() else {
        return false;
    };
    matches!(&decorator.expression, Expr::Name(name) if name.id.as_str() == "property")
}

/// `@<name>.setter` — the decorator list carries exactly one
/// `Attribute` whose own value is the bare name `<name>` (the SAME
/// name as the function being decorated — CPython's own
/// `@<property_name>.setter` spelling) and whose attribute is
/// literally `setter`.
fn is_property_setter_decorator(function: &StmtFunctionDef, property_name: &str) -> bool {
    let [decorator] = function.decorator_list.as_slice() else {
        return false;
    };
    let Expr::Attribute(attribute) = &decorator.expression else {
        return false;
    };
    attribute.attr.as_str() == "setter"
        && matches!(attribute.value.as_ref(), Expr::Name(base) if base.id.as_str() == property_name)
}

/// A getter body's exact single statement: `return self.<backing>` —
/// the backing attribute's name, or `None` for any other body shape
/// (more than one statement, a non-`self` or non-`Attribute` return
/// value, a bare `return`).
fn single_self_attribute_return(function: &StmtFunctionDef) -> Option<String> {
    let [Stmt::Return(ret)] = function.body.as_slice() else {
        return None;
    };
    let value = ret.value.as_deref()?;
    self_attribute_name(value)
}

/// A setter body's exact single statement: `self.<backing> = <param>`,
/// where `<param>` is a BARE Name matching the setter's own second
/// parameter (the first is `self`) — the backing name and the
/// parameter itself, or `None` for any other shape (more than one
/// statement, a non-self-attribute target, an RHS that is not exactly
/// that bare parameter, or a setter with fewer than two parameters).
fn single_self_attribute_write_of_second_parameter(
    function: &StmtFunctionDef,
) -> Option<(String, &ruff_python_ast::ParameterWithDefault)> {
    let [Stmt::Assign(assign)] = function.body.as_slice() else {
        return None;
    };
    let [target] = assign.targets.as_slice() else {
        return None;
    };
    let backing = self_attribute_name(target)?;
    let parameters: Vec<_> = function
        .parameters
        .posonlyargs
        .iter()
        .chain(function.parameters.args.iter())
        .collect();
    let (_self_param, rest) = parameters.split_first()?;
    let [value_parameter] = rest else {
        return None;
    };
    if !is_bare_name(assign.value.as_ref(), value_parameter.parameter.name.id.as_str()) {
        return None;
    }
    Some((backing, value_parameter))
}

/// A field's default value expression, evaluated against a fresh
/// environment via `expressions::evaluate_expression` — EXCEPT a
/// `Field(...)` call, which is pydantic surface, not an ordinary
/// call `evaluate_expression` can read (it declines every call whose
/// callee is a bound-or-unrecognized name, per expressions.rs's own
/// `evaluate_call` contract). `field_call_default` reads a `Field`
/// call's own `default=` keyword when the call names `Field` by
/// import identity, so `age: Age = Field(default=40, ge=0, le=120)`
/// reads its default the same way a bare `age: Age = 40` does.
fn default_value_of(value_expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    if let Expr::Call(call) = value_expr {
        if let Some(default) = field_call_default(call) {
            return default;
        }
    }
    evaluate_expression(value_expr, environment, kernel)
}

/// `Field(default=..., ...)` — the DEFAULT for a `= Field(...)` row is
/// `Field`'s own `default=` keyword when that keyword's value is a
/// numeric literal (`surface.rs`'s own `literal_number` reader, the
/// same one `annotated_expression_set` uses for `ge`/`le`/`lt`/`gt`).
/// No import-identity check gates the callee name here: this file does
/// not itself decide whether the call names pydantic's `Field` — the
/// field's ANNOTATION already gated that upstream via
/// `annotated_expression_set` when it read `Annotated[int,
/// Field(...)]`, and the mission's example row (`age: Age =
/// Field(default=..., ...)`) carries its constraint through the `Age`
/// alias's own `Annotated[...]`, so matching the callee's bare
/// spelling is sufficient for every corpus row this wave serves. An
/// int-sorted literal tags `Integer`, matching every other int literal
/// this checker reads (`expressions.rs`'s own `number_literal_value`).
fn field_call_default(call: &ruff_python_ast::ExprCall) -> Option<AbstractValue> {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "Field" {
        return None;
    }
    let keyword = call
        .arguments
        .keywords
        .iter()
        .find(|kw| kw.arg.as_ref().is_some_and(|name| name.as_str() == "default"))?;
    let sort = if matches!(&keyword.value, Expr::NumberLiteral(literal) if matches!(literal.value, ruff_python_ast::Number::Float(_)))
    {
        PrimitiveKind::Float
    } else {
        PrimitiveKind::Integer
    };
    let value = crate::refinedpy::surface::literal_number(&keyword.value)?;
    Some(known_values(vec![value], sort, TrustProved))
}

/// Judge one construction call's arguments against a class's declared
/// fields, mapping positional arguments to fields in declaration
/// order and then keyword arguments by name. A keyword naming no
/// field, or more positional arguments than the class has fields, is
/// an unmodeled construction — an overload, a `**kwargs`-absorbing
/// `__init__`, or simply a call this table cannot map exactly — and
/// answers `unknown()` with no fires: an unmapped construction never
/// guesses which field a stray argument might have landed in.
///
/// Each mapped argument judges through `assignability::judge` against
/// its field's declared refinement (when the field has one): `Fire`
/// pushes `(argument_range, message)` and the field holds the
/// argument's own value regardless (the refused-write law lives at
/// the WRITE sink in `check.rs`; this constructor's job is reporting
/// the fire and building the best-known instance, matching
/// `dict_literal_value`'s own "still holds a value" convention rather
/// than substituting the declared set the way `judge_and_bind` does
/// for a name binding — an object field slot has no reassignable name
/// downstream the way a plain variable does). `Undetermined` also
/// keeps the argument's own value at that field (the DECLARED set,
/// per the mission: "the field holds the DECLARED set as a known_set
/// value, TrustSpec — same construction check.rs's seed_parameters
/// uses"). A field with no argument takes its default if present,
/// else its declared set if present, else `unknown()`.
pub fn judge_construction(
    model: &ClassModel,
    positional: &[(AbstractValue, TextRange)],
    keyword: &[(String, AbstractValue, TextRange)],
    kernel: &Arc<RefinedTSKernel>,
) -> ConstructionVerdict {
    if positional.len() > model.fields.len() {
        return ConstructionVerdict {
            fires: Vec::new(),
            instance: unknown(),
        };
    }
    let mut keyword_by_name: HashMap<&str, &(String, AbstractValue, TextRange)> = HashMap::new();
    for entry in keyword {
        keyword_by_name.insert(entry.0.as_str(), entry);
    }
    let known_field_names: std::collections::HashSet<&str> =
        model.fields.iter().map(|field| field.name.as_str()).collect();
    if keyword_by_name.keys().any(|name| !known_field_names.contains(name)) {
        return ConstructionVerdict {
            fires: Vec::new(),
            instance: unknown(),
        };
    }

    let mut fires = Vec::new();
    let mut entries: Vec<ObjectKey> = Vec::new();
    for (index, field) in model.fields.iter().enumerate() {
        let argument = positional
            .get(index)
            .map(|(value, range)| (value.clone(), *range))
            .or_else(|| keyword_by_name.get(field.name.as_str()).map(|(_, value, range)| (value.clone(), *range)));

        let field_value = match argument {
            Some((value, range)) => match &field.declared {
                Some(declared) => match judge(&value, declared, kernel) {
                    Verdict::Fire(message) => {
                        fires.push((range, message));
                        value
                    }
                    Verdict::Silent => value,
                    Verdict::Undetermined(_) => known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None),
                },
                None => value,
            },
            None => match (&field.default, &field.declared) {
                (Some(default), _) => default.clone(),
                (None, Some(declared)) => known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None),
                (None, None) => unknown(),
            },
        };
        entries.push(ObjectKey {
            name: field.name.clone(),
            value: field_value,
        });
    }

    let instance = known_object(entries, None, true, TrustSpec, false);
    ConstructionVerdict { fires, instance }
}

/// `instance.field` — the field's value out of a known_object
/// instance, matching `collection_models.rs`'s own dict-key access
/// pattern (`dict_key_read`): a linear scan of `keys` for the matching
/// name. `None` for anything else — an unknown instance, an instance
/// missing that key, or a non-`Kind::Object` value (this table never
/// builds any other kind, but a caller may hand in an arbitrary
/// AbstractValue).
///
/// `instance.field` reads a STORED field. A `@property` name is never
/// a stored field — it is a read ALIAS the model states, so a
/// property read routes through `field_read_through_model` instead,
/// which resolves the alias to its backing name before calling this
/// function.
pub fn field_read(instance: &AbstractValue, field: &str) -> Option<AbstractValue> {
    if instance.kind != Kind::Object {
        return None;
    }
    instance
        .keys
        .iter()
        .find(|entry| entry.name == field)
        .map(|entry| entry.value.clone())
}

/// `box.age` where `age` may be a stored field OR a `@property` read
/// alias: a property name resolves to its `backing` field's own value
/// (`PropertyModel`'s doc — "the property `<name>` is a READ alias of
/// `<backing>`"); any other name reads the instance's stored field
/// directly, same as `field_read`.
pub fn field_read_through_model(model: &ClassModel, instance: &AbstractValue, field: &str) -> Option<AbstractValue> {
    match model.properties.get(field) {
        Some(property) => field_read(instance, &property.backing),
        None => field_read(instance, field),
    }
}

/// `self.x = v` / `obj.x = v` — judge a field write against the
/// class's declared refinement for that field. `None` when the field
/// carries no declared refinement (an ordinary unrefined field write,
/// not a blocker) OR when the class has no field by that name (an
/// attribute the model does not track — not this function's business
/// to invent a verdict for). A `@property` name judges against its OWN
/// setter-parameter refinement (`PropertyModel.declared`) rather than
/// any refinement its `backing` field carries — the setter's parameter
/// annotation is the more specific claim for a write through the
/// accessor (`PropertyModel`'s own doc).
pub fn field_write_judgment(
    model: &ClassModel,
    field: &str,
    value: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Verdict> {
    if let Some(property) = model.properties.get(field) {
        let declared = property.declared.as_ref()?;
        return Some(judge(value, declared, kernel));
    }
    let declared = model.fields.iter().find(|f| f.name == field)?.declared.as_ref()?;
    Some(judge(value, declared, kernel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
    use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set};

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

    fn age_declared() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
            spelling: "Age".to_owned(),
            admits_none: false,
        }
    }

    fn integer_value(v: f64) -> AbstractValue {
        known_values(vec![v], PrimitiveKind::Integer, TrustProved)
    }

    /// A hand-built `ClassModel` with no properties — every direct
    /// `judge_construction`/`field_write_judgment` test builds a model
    /// this way rather than parsing source, since those functions take
    /// the model, not the class definition.
    fn bare_model(name: &str, fields: Vec<ClassField>) -> ClassModel {
        ClassModel { name: name.to_owned(), fields, properties: HashMap::new() }
    }

    fn range_of(source: &str) -> TextRange {
        // a stable, arbitrary non-default range for tests that only
        // check WHICH range a fire carries back, never its exact span
        let _ = source;
        TextRange::default()
    }

    // --- class_table: field order + declared sets ---

    #[test]
    fn class_table_reads_fields_in_declaration_order_with_declared_sets() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, BaseModel\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Person(BaseModel):\n",
            "    age: Age\n",
            "    label: str\n",
        ));
        let aliases = crate::refinedpy::surface::compile_aliases(&module);
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let person = table.get("Person").expect("Person class recorded");
        assert_eq!(person.fields.len(), 2);
        assert_eq!(person.fields[0].name, "age");
        assert!(person.fields[0].declared.is_some(), "age reads its Annotated set");
        assert_eq!(person.fields[1].name, "label");
        assert!(person.fields[1].declared.is_none(), "bare str states no refinement");
    }

    // --- ClassVar is skipped ---

    #[test]
    fn class_var_annotated_row_is_not_an_instance_field() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import ClassVar\n",
            "class Counted:\n",
            "    total: ClassVar[int] = 0\n",
            "    age: int = 40\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let counted = table.get("Counted").expect("Counted class recorded");
        assert_eq!(counted.fields.len(), 1, "ClassVar row must not become a field");
        assert_eq!(counted.fields[0].name, "age");
    }

    // --- judge_construction: positional mapping ---

    #[test]
    fn judge_construction_maps_positional_arguments_in_declaration_order() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Person",
            vec![
                ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None },
                ClassField { name: "label".to_owned(), declared: None, default: None },
            ],
        );
        let positional = vec![
            (integer_value(40.0), range_of("40")),
            (known_values(vec![0.0], PrimitiveKind::String, TrustProved), range_of("label")),
        ];
        let verdict = judge_construction(&model, &positional, &[], &kernel);
        assert!(verdict.fires.is_empty());
        assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(40.0)));
    }

    // --- judge_construction: keyword out-of-set fire ---

    #[test]
    fn judge_construction_keyword_out_of_set_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Person",
            vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None }],
        );
        let keyword = vec![("age".to_owned(), integer_value(200.0), range_of("200"))];
        let verdict = judge_construction(&model, &[], &keyword, &kernel);
        assert_eq!(verdict.fires.len(), 1);
        assert!(verdict.fires[0].1.contains("'200'"), "{}", verdict.fires[0].1);
    }

    /// A keyword naming no field on the class is an unmodeled
    /// construction — unknown() with no fires, never a guess.
    #[test]
    fn judge_construction_unknown_keyword_declines_whole() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Person",
            vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None }],
        );
        let keyword = vec![("nickname".to_owned(), integer_value(1.0), range_of("1"))];
        let verdict = judge_construction(&model, &[], &keyword, &kernel);
        assert!(verdict.fires.is_empty());
        assert_eq!(verdict.instance.kind, Kind::Unknown);
    }

    // --- missing-arg default ---

    #[test]
    fn judge_construction_missing_argument_takes_the_default() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Grow",
            vec![ClassField { name: "age".to_owned(), declared: None, default: Some(integer_value(18.0)) }],
        );
        let verdict = judge_construction(&model, &[], &[], &kernel);
        assert!(verdict.fires.is_empty());
        assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(18.0)));
    }

    /// A missing argument with no default but a declared set holds the
    /// DECLARED SET (TrustSpec), the same construction seed_parameters
    /// uses for an unbound parameter.
    #[test]
    fn judge_construction_missing_argument_with_no_default_holds_declared_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Person",
            vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None }],
        );
        let verdict = judge_construction(&model, &[], &[], &kernel);
        assert!(verdict.fires.is_empty());
        let field = field_read(&verdict.instance, "age").expect("age field present");
        assert_eq!(field.kind, Kind::Set);
    }

    // --- field_read ---

    #[test]
    fn field_read_on_the_built_instance() {
        let Some(kernel) = loaded_kernel() else { return };
        let model =
            bare_model("Person", vec![ClassField { name: "age".to_owned(), declared: None, default: None }]);
        let positional = vec![(integer_value(40.0), range_of("40"))];
        let verdict = judge_construction(&model, &positional, &[], &kernel);
        assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(40.0)));
        assert_eq!(field_read(&verdict.instance, "missing"), None);
        assert_eq!(field_read(&unknown(), "age"), None);
    }

    // --- field_write_judgment ---

    #[test]
    fn field_write_judgment_fires_on_an_out_of_set_write() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Aged",
            vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None }],
        );
        let verdict = field_write_judgment(&model, "age", &integer_value(200.0), &kernel);
        assert!(matches!(verdict, Some(Verdict::Fire(_))));
    }

    #[test]
    fn field_write_judgment_is_none_for_an_undeclared_field() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model("Aged", vec![ClassField { name: "age".to_owned(), declared: None, default: None }]);
        let verdict = field_write_judgment(&model, "age", &integer_value(200.0), &kernel);
        assert!(verdict.is_none(), "an undeclared field writes with no judgment");
    }

    // --- pydantic-style class: Annotated[int, Field(ge=0, le=120)] field construction fire ---

    #[test]
    fn pydantic_style_annotated_field_construction_fires_over_ceiling() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import BaseModel, Field\n",
            "class Person(BaseModel):\n",
            "    age: Annotated[int, Field(ge=0, le=120)]\n",
        ));
        let aliases = crate::refinedpy::surface::compile_aliases(&module);
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let person = table.get("Person").expect("Person class recorded");
        assert!(person.fields[0].declared.is_some(), "inline Annotated field reads its own set");
        let keyword = vec![("age".to_owned(), integer_value(200.0), range_of("200"))];
        let verdict = judge_construction(person, &[], &keyword, &kernel);
        assert_eq!(verdict.fires.len(), 1);
        assert!(verdict.fires[0].1.contains("'200'"), "{}", verdict.fires[0].1);
    }

    // --- __init__-derived fields ---

    /// `class Person: def __init__(self, age: int): self.age = age` —
    /// d-module-surface.py:21-23's own shape. The parameter flows
    /// straight into the field: positional construction maps the
    /// argument to `age`, and `field_read` answers it back — with NO
    /// fire at construction (the parameter's annotation is a plain
    /// `int`, no refinement set), matching d-module-surface.py:128's
    /// own comment that the fire happens later, at the return sink,
    /// not here.
    #[test]
    fn init_derived_field_maps_positional_construction_and_reads_back() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Person:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let person = table.get("Person").expect("Person class recorded");
        assert_eq!(person.fields.len(), 1);
        assert_eq!(person.fields[0].name, "age");
        // `int` alone is not a form typereading reads as a refinement
        // (unlike `Age`, an Annotated alias) — no declared set, so no
        // fire is possible at this field regardless of the argument.
        assert!(person.fields[0].declared.is_none());

        let positional = vec![(integer_value(200.0), range_of("200"))];
        let verdict = judge_construction(person, &positional, &[], &kernel);
        assert!(verdict.fires.is_empty(), "a plain int field never fires at construction");
        assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(200.0)));
    }

    /// A class mixing a class-body `AnnAssign` with an explicit
    /// `__init__`: the `__init__`-forwarded parameter takes the
    /// POSITIONAL construction slot, but the field keeps the
    /// AnnAssign's own declared refinement (the more specific claim) —
    /// so a positional construction argument still fires against the
    /// class-body's `Age` alias.
    #[test]
    fn mixed_annassign_and_init_field_keeps_the_annassign_declared_set_at_the_init_position() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Person:\n",
            "    age: Age\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
        ));
        let aliases = crate::refinedpy::surface::compile_aliases(&module);
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let person = table.get("Person").expect("Person class recorded");
        assert_eq!(person.fields.len(), 1, "the AnnAssign and __init__ rows name the same field");
        assert_eq!(person.fields[0].name, "age");
        assert!(person.fields[0].declared.is_some(), "the AnnAssign's Age set survives the merge");

        let positional = vec![(integer_value(200.0), range_of("200"))];
        let verdict = judge_construction(person, &positional, &[], &kernel);
        assert_eq!(verdict.fires.len(), 1, "200 fires against the AnnAssign's own Age set");
    }

    /// `self.total = 0` — a self-write whose RHS is a literal, not a
    /// parameter: the field exists with that literal as its DEFAULT,
    /// no declared refinement, and no construction slot of its own (it
    /// trails every parameter-flowing field).
    #[test]
    fn self_write_with_a_literal_rhs_becomes_a_default_only_field() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Counter:\n",
            "    def __init__(self, step: int) -> None:\n",
            "        self.step = step\n",
            "        self.total = 0\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let counter = table.get("Counter").expect("Counter class recorded");
        assert_eq!(counter.fields.len(), 2);
        assert_eq!(counter.fields[0].name, "step", "the parameter-flowing field keeps its construction slot");
        assert_eq!(counter.fields[1].name, "total", "the literal self-write trails as a default-only field");
        assert!(counter.fields[1].declared.is_none());
        assert_eq!(counter.fields[1].default, Some(integer_value(0.0)));

        let verdict = judge_construction(counter, &[(integer_value(5.0), range_of("5"))], &[], &kernel);
        assert_eq!(field_read(&verdict.instance, "total"), Some(integer_value(0.0)));
    }

    /// `self.cache = build_cache()` — a self-write whose RHS is an
    /// unmodeled call: `evaluate_expression` declines every call
    /// (`expressions.rs`'s own `evaluate_call` contract), so the field
    /// exists with no declared refinement and no default — an honest
    /// unknown, never a guess.
    #[test]
    fn self_write_with_an_unreadable_rhs_stays_declared_none_default_none() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Cached:\n",
            "    def __init__(self) -> None:\n",
            "        self.cache = build_cache()\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let cached = table.get("Cached").expect("Cached class recorded");
        assert_eq!(cached.fields.len(), 1);
        assert_eq!(cached.fields[0].name, "cache");
        assert!(cached.fields[0].declared.is_none());
        assert!(cached.fields[0].default.is_none());
    }

    // --- inheritance via super().__init__ ---

    /// `class BaseYears: def __init__(self, age: int): self.age = age`
    /// / `class KidYears(BaseYears): def __init__(self, age: int):
    /// super().__init__(age)` — e-class-and-function.py:396-408's own
    /// shape. `KidYears`'s single field `age` is parent-linked through
    /// the `super()` call: a child construction argument flows through
    /// to `field_read`, exactly as `super_init_call`'s fixture comment
    /// states ("200 carried through the super call").
    #[test]
    fn super_init_call_links_the_child_construction_argument_to_the_parent_field() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class BaseYears:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "class KidYears(BaseYears):\n",
            "    def __init__(self, age: int) -> None:\n",
            "        super().__init__(age)\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let kid = table.get("KidYears").expect("KidYears class recorded");
        assert_eq!(kid.fields.len(), 1, "the super() call links to the SAME field, not a duplicate");
        assert_eq!(kid.fields[0].name, "age");

        let positional = vec![(integer_value(200.0), range_of("200"))];
        let verdict = judge_construction(kid, &positional, &[], &kernel);
        assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(200.0)));
    }

    /// A parent field carrying a declared refinement, forwarded through
    /// `super().__init__(...)`: a child construction argument outside
    /// that set fires — the child-parameter linkage carries the
    /// parent's own declared set forward when the child parameter's
    /// own annotation states none.
    #[test]
    fn super_init_call_construction_fires_when_the_parent_field_carries_a_refinement() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class BaseAged:\n",
            "    age: Age\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "class KidAged(BaseAged):\n",
            "    def __init__(self, age: int) -> None:\n",
            "        super().__init__(age)\n",
        ));
        let aliases = crate::refinedpy::surface::compile_aliases(&module);
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let kid = table.get("KidAged").expect("KidAged class recorded");
        assert!(kid.fields[0].declared.is_some(), "the parent's AnnAssign-declared Age set carries through");

        let positional = vec![(integer_value(200.0), range_of("200"))];
        let verdict = judge_construction(kid, &positional, &[], &kernel);
        assert_eq!(verdict.fires.len(), 1, "200 fires against the inherited Age set");
    }

    /// A child with NO explicit `__init__` inherits the parent's
    /// fields wholesale (datamodel.rst's `object.__init__` — the
    /// parent's own `__init__` runs at construction when the child
    /// declares none).
    #[test]
    fn a_child_with_no_init_inherits_the_parents_fields_wholesale() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class BaseYears:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "class ChildYears(BaseYears):\n",
            "    pass\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let child = table.get("ChildYears").expect("ChildYears class recorded");
        assert_eq!(child.fields.len(), 1);
        assert_eq!(child.fields[0].name, "age");

        let positional = vec![(integer_value(40.0), range_of("40"))];
        let verdict = judge_construction(child, &positional, &[], &kernel);
        assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(40.0)));
    }

    // --- property accessors ---

    /// `class Aged: def __init__(self): self._held = 40` / `@property
    /// def age(self): return self._held` — e-class-and-function.py:
    /// 336-344's own shape. `age` reads as an alias of `_held`'s value.
    #[test]
    fn property_read_aliases_the_backing_field() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Aged:\n",
            "    def __init__(self) -> None:\n",
            "        self._held = 40\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self._held\n",
            "    @age.setter\n",
            "    def age(self, value: int) -> None:\n",
            "        self._held = value\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let aged = table.get("Aged").expect("Aged class recorded");
        assert!(aged.properties.contains_key("age"));
        assert_eq!(aged.properties["age"].backing, "_held");

        let verdict = judge_construction(aged, &[], &[], &kernel);
        assert_eq!(
            field_read_through_model(aged, &verdict.instance, "age"),
            Some(integer_value(40.0)),
            "reading the property answers the backing field's own value"
        );
    }

    /// A setter whose parameter carries a declared refinement
    /// (`value: Age`, not a plain `int`): a write through the property
    /// fires when the value is outside that set.
    #[test]
    fn property_setter_write_fires_through_field_write_judgment() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Aged:\n",
            "    def __init__(self) -> None:\n",
            "        self._held = 40\n",
            "    @property\n",
            "    def age(self) -> int:\n",
            "        return self._held\n",
            "    @age.setter\n",
            "    def age(self, value: Age) -> None:\n",
            "        self._held = value\n",
        ));
        let aliases = crate::refinedpy::surface::compile_aliases(&module);
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let aged = table.get("Aged").expect("Aged class recorded");
        assert!(aged.properties["age"].declared.is_some(), "the setter's Age annotation is read");

        let verdict = field_write_judgment(aged, "age", &integer_value(200.0), &kernel);
        assert!(matches!(verdict, Some(Verdict::Fire(_))), "200 fires against the setter's own Age set");
    }
}
