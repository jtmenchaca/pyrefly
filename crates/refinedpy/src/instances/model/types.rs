//! The three model structs: one class's declared shape (`ClassModel`),
//! one property accessor pair (`PropertyModel`), and one declared field
//! (`ClassField`).

use std::collections::HashMap;

use refined_domain::abstract_value::AbstractValue;
use ruff_python_ast::StmtFunctionDef;

use crate::typereading::DeclaredRefinement;

/// One class's declared shape: its name, its fields (in construction
/// order), its property accessors (read/write aliases that are never
/// stored fields of their own — see `PropertyModel`), its EFFECTIVE
/// method set (`methods`: every def in the class body, own defs
/// overriding an inherited def of the same name), and its PARENT's own
/// methods unaffected by any child override (`parent_methods`, the
/// `super().<method>(...)` resolution target). `Clone` so a class's
/// own model can be copied wholesale — a re-exported class
/// (`cross_module.rs::pull_member`) and a body-local class table merge
/// (`check.rs::merged_classes_for_body`) both need an owned copy
/// without disturbing the table they read it from.
#[derive(Clone)]
pub struct ClassModel {
    pub name: String,
    pub fields: Vec<ClassField>,
    pub properties: HashMap<String, PropertyModel>,
    pub methods: HashMap<String, StmtFunctionDef>,
    pub parent_methods: HashMap<String, StmtFunctionDef>,
    /// Every class-body top-level PLAIN (unannotated) `name = <literal>`
    /// row — e-class-and-function.py's `Counted`/`Limits`: `total = 0`,
    /// `ceiling = 40`. A class attribute lives on the CLASS OBJECT
    /// ITSELF, never on any one instance (datamodel.rst, "Classes" —
    /// class attributes are looked up on the class, distinct from the
    /// per-instance `__dict__` an `AnnAssign`/`__init__` field populates),
    /// so this table is read into a SEPARATE class-object value
    /// (`check.rs`'s own class-object seeding at `Stmt::ClassDef`), never
    /// folded into `fields`/instance construction. An `AnnAssign` row
    /// (`age: int = 40`, a declared INSTANCE field) is never read here —
    /// the two tables are disjoint by construction (`AnnAssign` only,
    /// `Assign` only), matching the language's own distinction between a
    /// class-level attribute and a declared instance field.
    pub class_attributes: Vec<ClassField>,
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
#[derive(Clone)]
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
    /// The bare `int`/`float`/`str`/`bool` sort the field's annotation
    /// states, when `declared` reads no refinement of its own
    /// (`typereading::base_sort_return_refinement`). READ ONLY BY THE
    /// SEED (`check::seed::class_parameter_object`), so a `a: int`
    /// field of a class-typed parameter starts as the whole-int ray a
    /// bare `raw: int` PARAMETER already starts as, and an ordinary
    /// range guard over `o.a` narrows it the same way. Deliberately
    /// separate from `declared`: the write-judging tables read
    /// `declared` alone, so a bare-`int` field gains no new refusal at
    /// a write, matching `base_sort_return_refinement`'s own
    /// "parameter seeding ONLY" scoping.
    pub base_sort: Option<DeclaredRefinement>,
}
