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
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use refined_domain::abstract_value::{
    known_set, unknown, AbstractValue, Kind, ObjectKey, SetKindTag,
};
use refined_domain::known_constructors::known_object;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::make_refined_set;
use ruff_python_ast::{Expr, ExprCall, ModModule, Number, Stmt, StmtClassDef, StmtFunctionDef, UnaryOp};
use ruff_text_size::TextRange;

use crate::assignability::{judge, Verdict};
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::function_table::FunctionTable;
use crate::surface::{AliasEntry, SurfaceImports};
use crate::typereading::{declared_refinement, DeclaredRefinement};

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
    aliases: &HashMap<String, AliasEntry>,
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

/// Every module-level `class X(TypedDict): name: Annotation, …` read
/// into its own per-member refinement table, keyed by the class's name.
/// A TypedDict class body carries no `__init__`, no methods, and no
/// inheritance chain this table follows — it is exactly the plain
/// AnnAssign rows `class_model_of` already reads for an ordinary class,
/// with none of that function's `__init__`/`super()` machinery, so this
/// is its own small reader rather than a `ClassModel`-shaped one: a
/// TypedDict's checked shape is "each member's own declared refinement,"
/// not "a constructible instance with fields." Recognized by BARE base
/// name `TypedDict` (`single_bare_name_base`, the same one-bare-Name-base
/// rule `class_table` already applies), matching `is_class_var`'s own
/// no-import-identity convention — no fixture row spells `TypedDict`
/// through an import alias.
pub fn typed_dict_table(
    module: &ModModule,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
) -> HashMap<String, Vec<(String, DeclaredRefinement)>> {
    let empty_environment = Environment::new(Default::default());
    let mut out = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::ClassDef(def) = stmt else {
            continue;
        };
        if single_bare_name_base(def) != Some("TypedDict") {
            continue;
        }
        let mut members = Vec::new();
        for member_stmt in def.body.iter() {
            let Stmt::AnnAssign(assign) = member_stmt else {
                continue;
            };
            let Expr::Name(target_name) = assign.target.as_ref() else {
                continue;
            };
            let Some(declared) =
                declared_refinement(assign.annotation.as_ref(), aliases, imports, &empty_environment)
            else {
                continue;
            };
            members.push((target_name.id.as_str().to_owned(), declared));
        }
        out.insert(def.name.id.as_str().to_owned(), members);
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
    aliases: &HashMap<String, AliasEntry>,
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

    // A parent-CLONE (`parent.cloned()`) so `class_model_of`'s own
    // field loop can build/read a DIFFERENT class (a member's own
    // annotation naming another BaseModel — `Resident.address:
    // Address`) through `out`/`building` without holding a live borrow
    // of `parent`'s entry across that nested `build_class_model` call —
    // Rust's borrow checker forbids mutating `out` (to insert the
    // nested class) while `parent` still borrows from it immutably.
    let parent = parent.cloned();
    let model = class_model_of(def, aliases, imports, kernel, parent.as_ref(), defs, out, building);
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
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
    parent: Option<&ClassModel>,
    defs: &HashMap<String, &StmtClassDef>,
    out: &mut HashMap<String, ClassModel>,
    building: &mut std::collections::HashSet<String>,
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
        let declared = declared_refinement(assign.annotation.as_ref(), aliases, imports, &empty_environment)
            // A bare-Name annotation `declared_refinement` reads as
            // nothing this table's own alias/inline-form grammar
            // covers (`Resident.address: Address` — `Address` is a
            // module-level BaseModel class, not a type alias) — the
            // MEMBERS LAW twin of `typed_dict_return_refinement`, but
            // for a class member rather than a return position. The
            // named class is built first when `out` does not have it
            // yet (`build_class_model`'s own lazy pattern, reused
            // verbatim: a class already mid-build reads parent-less/
            // member-less rather than looping on a field cycle), then
            // its OWN just-built fields become this member's per-field
            // table — nested BaseModel membership (`Resident.person:
            // Person`, itself a BaseModel) recurses for free, since
            // `Person` was built the same way and its own `declared`
            // may itself carry `members: Some(...)`.
            .or_else(|| {
                let Expr::Name(class_name) = assign.annotation.as_ref() else {
                    return None;
                };
                if !defs.contains_key(class_name.id.as_str()) {
                    return None;
                }
                if !out.contains_key(class_name.id.as_str()) {
                    build_class_model(class_name.id.as_str(), defs, aliases, imports, kernel, out, building);
                }
                out.get(class_name.id.as_str()).map(model_members_refinement)
            });
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

    let own_methods = own_method_table(def);
    let parent_methods = parent.map(|parent| parent.methods.clone()).unwrap_or_default();
    // the EFFECTIVE method set: every parent method, with this class's
    // own defs of the SAME name overriding it — `HashMap::extend`
    // overwrites on a key collision, keeping the later (own) insertion,
    // which is exactly "own defs override inherited ones."
    let mut methods = parent_methods.clone();
    methods.extend(own_methods);

    let class_attributes = class_attribute_table(def, &empty_environment, kernel);

    ClassModel {
        name: def.name.id.as_str().to_owned(),
        fields,
        properties,
        methods,
        parent_methods,
        class_attributes,
    }
}

/// A module-level class's own fields, wrapped as a `DeclaredRefinement`
/// with `members: Some(...)` — `typed_dict_return_refinement`'s exact
/// shape, built here instead from an already-built `ClassModel` rather
/// than a fresh scan, since a class MEMBER's own declared refinement
/// (`declared: Option<DeclaredRefinement>` per field) is already exactly
/// what `assignability::judge`'s MEMBERS LAW reads. A field whose own
/// annotation states no refinement (`declared: None` — an unrefined
/// `str`/`int`, or an annotation this table cannot read) is left out of
/// the member list entirely, matching `typed_dict_table`'s own
/// `let Some(declared) = ... else { continue }` convention: an absent
/// member states nothing the MEMBERS LAW judges, never a guessed set.
///
/// `pub`: `class_model_of`'s own field loop (below) is not the only
/// caller a bare CLASS-NAME annotation needs.
///
/// A STATEMENT-LEVEL construction (`return Person.model_validate({"age":
/// 200, ...})`, `m-pydantic-schema.py`'s own corpus shape) already fires
/// correctly with NO help from this function: `check.rs::sink_value`'s
/// law 3 (`construction_call_verdict`) surfaces `judge_construction`'s
/// own per-field fires directly at the statement sink, regardless of
/// whether `Person`'s bare-class-name RETURN annotation itself compiles
/// a `DeclaredRefinement` (`declared_refinement` never learns a class
/// name at all — only `typed_dict_return_refinement`'s narrower
/// `TypedDict`-only table, `typereading.rs`'s own doc). That is why the
/// corpus's key-by-key membership rows already measure clean without
/// this reader.
///
/// A construction NESTED inside a call ARGUMENT
/// (`record_vitals(Vitals(heart_rate=72, spo2=130))`, the showcase's own
/// row) is the one shape that still loses its fire: `check.rs::judge_
/// one_call_argument` evaluates each argument through plain
/// `evaluate_expression`, whose own same-module-construction arm
/// (`expressions.rs`) discards `judge_construction`'s fires by design —
/// "a construction's fires belong to whichever statement sink hosts
/// this call expression, not to this nested value read" — because
/// ordinarily SOME enclosing sink (a return, an assignment) already
/// re-fires them through `sink_value`. An argument position is not such
/// a sink today: it neither calls `sink_value` NOR re-derives a
/// `members`-carrying `DeclaredRefinement` for a bare class-name
/// parameter to judge the constructed instance against afterward. THIS
/// function is exactly the second route — the caller can build `v`'s
/// own `DeclaredRefinement` from `context.classes.get("Vitals")` and let
/// `assignability::judge`'s MEMBERS LAW re-judge the already-built
/// instance — but surfacing `construction_call_verdict`'s fires the same
/// way `sink_value` already does for its OWN argument-evaluating step is
/// the more direct fix, since it reuses a verdict already computed
/// correctly rather than re-deriving one. Either fix lands in
/// `check.rs`, not here; this function is exported so whichever route is
/// chosen has the member-table reader ready.
pub fn model_members_refinement(model: &ClassModel) -> DeclaredRefinement {
    let members: Vec<(String, DeclaredRefinement)> = model
        .fields
        .iter()
        .filter_map(|field| field.declared.clone().map(|declared| (field.name.clone(), declared)))
        .collect();
    DeclaredRefinement {
        set: make_refined_set(Vec::new()),
        spelling: model.name.clone(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: Some(members),
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    }
}

/// Every class-body top-level PLAIN `name = <literal>` row (see
/// `ClassModel.class_attributes`'s own doc): a bare-Name target on a
/// SINGLE-target `Stmt::Assign` (no `AnnAssign` — that is an instance
/// field, read separately above), whose RHS reads through
/// `evaluate_expression` (the same reader `default_value_of` uses for an
/// instance field's own default). An unreadable RHS (a call, a name
/// reference, …) is skipped — `check.rs`'s class-object seeding has
/// nothing to bind that attribute to either way, and skipping it here is
/// the same honest omission `init_derived_fields`'s own unreadable-RHS
/// row already takes for an instance field.
fn class_attribute_table(def: &StmtClassDef, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Vec<ClassField> {
    let mut attributes = Vec::new();
    for stmt in def.body.iter() {
        let Stmt::Assign(assign) = stmt else {
            continue;
        };
        let [target] = assign.targets.as_slice() else {
            continue;
        };
        let Expr::Name(target_name) = target else {
            continue;
        };
        let value = evaluate_expression(assign.value.as_ref(), environment, kernel);
        if value.kind == Kind::Unknown {
            continue;
        }
        attributes.push(ClassField {
            name: target_name.id.as_str().to_owned(),
            declared: None,
            default: Some(value),
        });
    }
    attributes
}

/// Every `def` directly in this class's OWN body, keyed by name —
/// `__init__` included (a `super().<method>(...)` call can name
/// `__init__` exactly like any other method, and a class with no
/// override of a given name simply has no entry here, falling through
/// to whatever `parent_methods` states in the caller's `methods` merge
/// above). A class-body `def` is read regardless of its own decorator
/// list (`@property`/`@x.setter` defs are ALSO ordinary callables by
/// name — `property_table` reads the identical two defs for its own
/// alias purpose, and the two readings do not conflict: a property
/// getter/setter is simultaneously an entry in `methods` under its own
/// name).
fn own_method_table(def: &StmtClassDef) -> HashMap<String, StmtFunctionDef> {
    def.body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::FunctionDef(function) => Some((function.name.id.as_str().to_owned(), function.clone())),
            _ => None,
        })
        .collect()
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
    aliases: &HashMap<String, AliasEntry>,
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
    aliases: &HashMap<String, AliasEntry>,
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
/// other by-spelling recognitions). `pub`: both `check.rs`'s
/// `bind_or_forget_target`/`write_named_field` and `summaries.rs`'s
/// restricted-body interpreter (its own `write_self_field`,
/// `interpret_aug_assign`) recognize the identical `self.<name>` shape
/// through this one function, rather than each re-deriving it.
pub fn self_attribute_name(target: &Expr) -> Option<String> {
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
    aliases: &HashMap<String, AliasEntry>,
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
    let value = crate::surface::literal_number(&keyword.value)?;
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

/// The counter `next_instance_identity` draws from — one process-wide
/// sequence, so two constructions anywhere in one checker run (even of
/// different classes, even on different threads if this checker ever
/// becomes concurrent) never mint the same id.
static NEXT_INSTANCE_IDENTITY: AtomicU32 = AtomicU32::new(0);

/// Mints a fresh per-construction identity — unique for the life of the
/// process, never reused. `judge_construction` stamps this onto every
/// instance it builds (`AbstractValue::instance_identity`'s own doc), so
/// two `Holder()` calls (the same class, the same AST call site, two
/// separate executions) always mint two distinct ids, exactly the way
/// `env.rs`'s `next_retained_callable_key` mints a fresh key per lambda/def
/// creation rather than keying by the AST's own range (that module's own
/// doc: a range key would let two creations of the same source text
/// silently conflate).
fn next_instance_identity() -> u32 {
    NEXT_INSTANCE_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

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
            numeric: false,
            value: field_value,
        });
    }

    let mut instance = known_object(entries, None, true, TrustSpec, false);
    // source carries the constructing class's name so a later
    // receiver.method(...) call can find the ClassModel in the
    // environment's class table; empty on every non-instance object.
    instance.source = model.name.clone();
    // instance_identity carries THIS call's own fresh id — distinct from
    // `source`, which every instance of `model` shares. Two `Holder()`
    // calls build two different instances; a dict keyed by one must not
    // answer a lookup by the other (`collection_models::known_dict_key`'s
    // own identity arm reads this field to tell them apart).
    instance.instance_identity = Some(next_instance_identity());

    // pydantic's own post-construction hook: `model_post_init(self,
    // __context)` runs immediately after every field is set
    // (docs/concepts/models.md's own "Post-init processing"), so a
    // dependent check written there (m-pydantic-schema.py's `Range`:
    // `if self.hi < self.lo: raise ValueError(...)`) is this
    // construction's own business, not a later sink's. Anchored at the
    // LAST mapped argument's range — a cross-field check has no single
    // refusing argument to blame, and this is the closest token to the
    // call's own closing paren among the ranges this function already
    // carries.
    if let Some(post_init) = model.methods.get("model_post_init") {
        if let Some(anchor) = keyword.last().map(|(_, _, range)| *range).or_else(|| positional.last().map(|(_, range)| *range)) {
            if let Some(message) = post_init_provable_raise(post_init, &instance, kernel) {
                fires.push((anchor, message));
            }
        }
    }

    ConstructionVerdict { fires, instance }
}

/// `model_post_init`'s own body, read ONLY in the one shape the corpus
/// spells: a SINGLE top-level `if <condition>: raise <exc>` statement
/// (no `elif`/`else`, no other statement before or after it) — the
/// dependent-check shape pydantic's own docs name for this hook
/// (docs/concepts/models.md, "Post-init processing" — the hook "will
/// be called... to perform additional validation"). `self` binds to
/// `instance` (already fully built — every field's own value, judged
/// or not, is in place, matching real pydantic's own construction
/// order: fields set, THEN `model_post_init` runs) and the condition
/// evaluates through `evaluate_expression`'s ordinary comparison
/// reading, restricted to `self.<field>` operands `field_read` already
/// answers.
///
/// `Some(message)` only when the condition is PROVABLY true
/// (`truthiness`'s `(true, true)` answer) — the same honest-decline
/// discipline every other provable-raise reader in this checker takes:
/// an undetermined or provably-false condition never fires here.
/// `raise <exc>`'s own message reads `<exc>`'s single string-literal
/// argument when `<exc>` is a bare `Call` (`ValueError("...")`,
/// `raise <name>` alone, or a computed message, states nothing this
/// reader can quote, so the message falls back to the exception
/// callee's own bare name). Any other body shape (more than one
/// top-level statement, an `elif`/`else` clause, a non-`Raise` `if`
/// body, a body that is not exactly one `if`) declines — `None`, never
/// a guessed fire.
fn post_init_provable_raise(
    post_init: &StmtFunctionDef,
    instance: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<String> {
    let [Stmt::If(if_stmt)] = post_init.body.as_slice() else {
        return None;
    };
    if !if_stmt.elif_else_clauses.is_empty() {
        return None;
    }
    let [Stmt::Raise(raise_stmt)] = if_stmt.body.as_slice() else {
        return None;
    };

    let mut environment = Environment::new(Default::default());
    environment.bind("self", instance.clone());
    let test_value = evaluate_expression(if_stmt.test.as_ref(), &environment, kernel);
    let (truthy, known) = truthiness(&test_value);
    if !known || !truthy {
        return None;
    }

    Some(post_init_raise_message(raise_stmt))
}

/// `model_post_init`'s own construction-site fire message: "this
/// expression provably raises `<ExcType>`: `<plain detail>`" — the
/// same voice `expressions::provable_raise` already speaks, quoting
/// `raise <exc>`'s own exception name and its single string-literal
/// argument when `<exc>` is a bare `Call` (`ValueError("hi must be >=
/// lo")` reads as `ValueError: hi must be >= lo`); any other `<exc>`
/// shape (a bare name, a computed argument, no argument at all) still
/// names the exception type alone.
fn post_init_raise_message(raise_stmt: &ruff_python_ast::StmtRaise) -> String {
    let Some(exc) = raise_stmt.exc.as_deref() else {
        return "this construction provably raises an exception".to_owned();
    };
    let Expr::Call(call) = exc else {
        return "this construction provably raises an exception".to_owned();
    };
    let Expr::Name(exc_name) = call.func.as_ref() else {
        return "this construction provably raises an exception".to_owned();
    };
    let detail = call
        .arguments
        .args
        .first()
        .and_then(|arg| match arg {
            Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
            _ => None,
        });
    match detail {
        Some(detail) => format!("this construction provably raises {}: {}", exc_name.id.as_str(), detail),
        None => format!("this construction provably raises {}", exc_name.id.as_str()),
    }
}

/// The CLASS OBJECT's own initial value — e-class-and-function.py's
/// `class_attribute_write`: `Counted.total = 40` then `Counted.total`
/// read back, a write/read pair that never touches any INSTANCE (no
/// `Counted(...)` construction happens on this row at all). Distinct
/// from `judge_construction`'s instance value: this reads `model.
/// class_attributes` alone (never `model.fields`, which are per-instance
/// slots this class object does not carry), tagged with the SAME
/// `source = model.name` convention `judge_construction` uses so
/// `write_named_field`/`field_read_through_model` (which only ever check
/// `instance.kind == Kind::Object` and a non-empty `source`) read/write
/// through it with NO new machinery — a class object and an instance
/// object share one representation, distinguished only by which table
/// (`class_attributes` vs `fields`) built their starting keys.
///
/// `check.rs`'s own `Stmt::ClassDef` walk binds this value under the
/// class's own bare name, in the ENCLOSING environment (the scope where
/// the class statement itself executes) — the environment slot
/// `Counted.total = 40`'s attribute-write law (a bare-Name receiver
/// bound to a tagged `Kind::Object`) then finds and rebinds, exactly the
/// same way an instance variable already does.
pub fn class_object_value(model: &ClassModel) -> AbstractValue {
    let entries: Vec<ObjectKey> = model
        .class_attributes
        .iter()
        .filter_map(|attribute| {
            attribute.default.clone().map(|value| ObjectKey {
                name: attribute.name.clone(),
                numeric: false,
                value,
            })
        })
        .collect();
    let mut value = known_object(entries, None, true, TrustSpec, false);
    value.source = model.name.clone();
    value
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
        .find(|entry| entry.name == field && !entry.numeric)
        .map(|entry| entry.value.clone())
}

/// `self.<field> = v` — the struct-updated instance with `field` set to
/// `value`, every other stored key AND every other `AbstractValue` field
/// (`source` included — the constructing class's tag must survive a
/// write, since a later `receiver.method(...)` call still needs it to
/// find the `ClassModel`) preserved from `instance` unchanged. `None`
/// for a non-`Kind::Object` instance — there is no field slot to write
/// on anything else this table builds. A field name absent from
/// `instance.keys` is APPENDED as a new entry (an ordinary Python
/// attribute gain, `field_write_judgment`'s own doc: "an ordinary
/// Python attribute gain is not a blocker") rather than declined.
pub fn field_write(instance: &AbstractValue, field: &str, value: AbstractValue) -> Option<AbstractValue> {
    if instance.kind != Kind::Object {
        return None;
    }
    let mut keys = instance.keys.clone();
    match keys.iter_mut().find(|entry| entry.name == field && !entry.numeric) {
        Some(entry) => entry.value = value,
        None => keys.push(ObjectKey {
            name: field.to_owned(),
            numeric: false,
            value,
        }),
    }
    Some(AbstractValue {
        keys,
        ..instance.clone()
    })
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

/// `model`'s own callable named `name` — own-overrides-inherited, since
/// `model.methods` (built in `class_model_of`) already holds the
/// EFFECTIVE set: a child's own def already replaced any inherited def
/// of the same name there. `None` for a name the class declares no
/// method under at all.
pub fn method_def_of<'a>(model: &'a ClassModel, name: &str) -> Option<&'a StmtFunctionDef> {
    model.methods.get(name)
}

/// `receiver.method(arguments)` — the instance AFTER the call (any
/// `self.<field> = ...` write inside the method body survives on it)
/// and the method's own return value, or `None` when the method's body
/// or parameter shape is outside what `summaries`'s restricted body
/// interpreter reads.
///
/// `self` binds to `instance` and the remaining parameters bind to
/// `arguments` positionally (`summaries::bind_parameters`'s own
/// convention: a trailing parameter with no matching argument takes its
/// own default, evaluated fresh; too few arguments with no default, or
/// too many, declines the whole call) — `self` is excluded from that
/// positional binding since it is bound directly to `instance`, never
/// to an entry of `arguments`.
///
/// The body interprets through `summaries::interpret_body`, the SAME
/// restricted statement walk an ordinary same-module call uses, with
/// one addition: a `super_resolver` that answers `super().<name>(args)`
/// by looking `name` up in `model.parent_methods` (the parent's OWN
/// methods, never touched by this child's overrides) and recursively
/// calling `method_call_result` on THAT def, over the SAME working
/// instance the resolver was handed (so a `super().__init__(...)` call
/// early in a child method still writes fields onto the one instance
/// this call is building), with an EMPTY `parent_methods` one level up
/// (a grandparent's own further `super()` chain is out of the
/// single-inheritance band this table builds — `class_table`'s own
/// scope). The resolver reads the CALLER's `super_resolver` parameter
/// (`environment: &Environment`) for `self`'s WORKING value at the
/// point of the call, per `summaries::SuperResolver`'s own doc — every
/// earlier `self.<field> = ...` statement in the SAME method body has
/// already updated it there.
///
/// `self.<field>` reads/writes inside the body route through
/// `summaries::interpret_body`'s own `self`-aware `Assign`/`AugAssign`/
/// `Expr::Name("self")` handling (`field_read`/`field_write`, the same
/// two functions this file exports) — this function does not re-walk
/// the body itself, only sets up the environment and resolver
/// `interpret_body` needs.
///
/// The answer is `(working instance, joined return value)`: every
/// `return` the body's paths could reach joins into one value
/// (`join_known`, `interpret_body`'s own fold), and a body that falls
/// off the end without an explicit `return` contributes `null_value()`
/// to that join — the same fall-through law `summaries::call_result`
/// already applies to an ordinary function. Depth-capped through
/// `summaries::CALL_DEPTH_CAP`, shared with every other same-module
/// call chain so a recursive method (directly, or through a
/// `super()`-chained cycle) declines rather than hangs. Any unsupported
/// body/parameter shape (`*args`/`**kwargs`/keyword-only parameters, a
/// statement `interpret_body` does not read, an unresolved `super()`
/// call, an unreadable return) answers `None` — an honest decline,
/// never a guessed instance.
///
/// `classes` seeds the method body's own environment with the module's
/// class table (`environment.set_classes`), the same way `table` seeds
/// the function table — without it, `self.<other_method>()` and any
/// property read inside the body has no class to resolve `self`'s own
/// type against, and declines even though the class and method both
/// exist. `None` when the caller has no class table to offer (a plain
/// unit test constructing a method call directly, for instance).
///
/// `datetime_imports` seeds the method body's own environment with the
/// module's `datetime` import identities (`environment.set_datetime_
/// imports`), the same explicit-parameter shape `classes` already takes
/// here — this function builds a FRESH `Environment` for the method
/// body (below) and has no enclosing `Environment` in scope to inherit
/// from (unlike `summaries::call_result_with_enclosing`, which inherits
/// `classes`/`datetime_imports` from its own `enclosing: Option<&
/// Environment>` parameter when the callee sets neither of its own), so
/// every caller threads its own module's table through explicitly, the
/// same way every caller already threads `classes`. Without it, a
/// method body's own `date(...)`/`dt.strptime(...)` call (an aliased
/// `datetime` construction/classmethod call — `expressions.rs`'s
/// datetime gates) has no import table to resolve the alias against and
/// declines even though the same call in a plain function recognizes.
/// `None` when the caller has no import table to offer (a plain unit
/// test constructing a method call directly, for instance).
pub fn method_call_result(
    instance: &AbstractValue,
    model: &ClassModel,
    method: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    classes: Option<&Arc<HashMap<String, ClassModel>>>,
    datetime_imports: Option<&Arc<crate::expressions::DatetimeImports>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> Option<(AbstractValue, AbstractValue)> {
    use crate::summaries::{collect_bound_names, interpret_body, CALL_DEPTH_CAP};

    if depth >= CALL_DEPTH_CAP {
        return None;
    }
    if method.parameters.vararg.is_some()
        || method.parameters.kwarg.is_some()
        || !method.parameters.kwonlyargs.is_empty()
    {
        return None;
    }
    // `@staticmethod` declares no `self`/`cls` receiver slot at all
    // (datamodel.rst's own "Static method objects": "a static method
    // does not receive an implicit first argument") — every declared
    // parameter binds positionally to `arguments`, exactly like an
    // ordinary same-module `def`, with no receiver consumed. Every
    // other member `def` (including `@classmethod`, out of this
    // function's own scope — its first parameter is `cls`, still a
    // receiver slot, just bound to the class rather than the instance)
    // keeps the `self`-splitting shape below.
    let is_static = method
        .decorator_list
        .iter()
        .any(|decorator| matches!(&decorator.expression, Expr::Name(name) if name.id.as_str() == "staticmethod"));

    let all_parameters: Vec<_> = method
        .parameters
        .posonlyargs
        .iter()
        .chain(method.parameters.args.iter())
        .collect();
    // the first parameter is `self` by convention for an ordinary
    // instance method (matching `init_derived_fields`'s own
    // first-parameter reading) — a method with no parameter at all is
    // not a bound instance method this function can seed a receiver
    // for. A `@classmethod`'s own first parameter is spelled `cls`
    // instead, still a receiver slot, just bound to the class rather
    // than the instance — `receiver_parameter_name` carries the ACTUAL
    // spelling forward so a `cls.<attr>` read/write inside the body
    // resolves, alongside the literal `"self"` binding every other
    // reader in this file (`self_attribute_name`, `interpret_assign`'s
    // own self-write recognition) already hardcodes.
    let (receiver_parameter_name, rest): (Option<&str>, Vec<_>) = if is_static {
        (None, all_parameters)
    } else {
        let (self_parameter, rest) = all_parameters.split_first()?;
        (Some(self_parameter.parameter.name.id.as_str()), rest.to_vec())
    };
    if arguments.len() > rest.len() {
        return None;
    }

    let mut locally_bound = std::collections::HashSet::new();
    if let Some(receiver_parameter_name) = receiver_parameter_name {
        locally_bound.insert("self".to_owned());
        locally_bound.insert(receiver_parameter_name.to_owned());
    }
    for parameter in &rest {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    collect_bound_names(&method.body, &mut locally_bound);
    let mut environment = Environment::new(locally_bound);
    // one call deeper than the caller — the depth cap engages across
    // the evaluate↔interpreter boundary (see env::call_depth)
    environment.set_call_depth(depth.saturating_add(1));
    if let Some(table) = table {
        environment.set_functions(table.clone());
    }
    if let Some(classes) = classes {
        environment.set_classes(classes.clone());
    }
    if let Some(datetime_imports) = datetime_imports {
        environment.set_datetime_imports(datetime_imports.clone());
    }
    if let Some(receiver_parameter_name) = receiver_parameter_name {
        environment.bind("self", instance.clone());
        if receiver_parameter_name != "self" {
            environment.bind(receiver_parameter_name, instance.clone());
        }
    }

    let default_environment = Environment::new(Default::default());
    for (index, parameter) in rest.iter().enumerate() {
        let value = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            let default_expr = parameter.default.as_deref()?;
            evaluate_expression(default_expr, &default_environment, kernel)
        };
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }

    let parent_methods = model.parent_methods.clone();
    let super_resolver = move |name: &str, args: &[AbstractValue], environment: &Environment| {
        let parent_def = parent_methods.get(name)?;
        let working_instance = environment.read("self")?.clone();
        let parentless = ClassModel {
            name: model.name.clone(),
            fields: Vec::new(),
            properties: HashMap::new(),
            methods: parent_methods.clone(),
            parent_methods: HashMap::new(),
            class_attributes: Vec::new(),
        };
        let (_after, result) = method_call_result(
            &working_instance,
            &parentless,
            parent_def,
            args,
            table,
            classes,
            datetime_imports,
            kernel,
            depth + 1,
        )?;
        Some(result)
    };

    let mut returns: Vec<AbstractValue> = Vec::new();
    let falls_through = interpret_body(&method.body, kernel, depth, &mut environment, &mut returns, Some(&super_resolver))?;
    if falls_through {
        returns.push(refined_domain::abstract_value::null_value());
    }

    let mut answers = returns.into_iter();
    let first = answers.next()?;
    let result = answers.fold(first, |acc, next| refined_domain::lattice_operations::join_known(acc, next));
    // A static method binds no `self` at all — there is no receiver for
    // its own body to mutate, so the "working instance" half of the
    // answer is simply `instance` unchanged (the caller's own
    // class-object value, echoed back rather than read out of an
    // environment slot that was never bound).
    let working_instance = if is_static { instance.clone() } else { environment.read("self")?.clone() };
    Some((working_instance, result))
}

/// A generator body's own yielded values, in order — `Some(Vec::new())`
/// for a body that yields nothing on its only path, `None` when the
/// body is outside the two shapes this function reads (a CONDITIONAL
/// yield, `yield from`, any restricted-body statement this function
/// itself does not walk). Models ONLY the yields themselves; a
/// `next(gen)` call's OWN read of "the first yield" is the WIRING
/// owner's job (`expressions.rs`'s `evaluate_call`) — this function
/// hands back the full ordered list so that caller can index position 0
/// (or answer a join over every yielded value, for a plain `for x in
/// gen():` walk, should that wiring choose to).
///
/// Two accepted top-level statement shapes, walked in source order and
/// merged into one ordered list (a LEADING docstring is skipped first,
/// `yields_of_body`'s own doc):
///
/// 1. A STRAIGHT-LINE `yield <expr>` statement (an `Expr` statement
///    whose value is `Expr::Yield`) — the yielded value evaluates
///    against the current environment and is appended in place. A bare
///    `return` ends iteration without yielding (datamodel.rst's
///    generator-function entry) — no more statements after it are read,
///    and a straight-line body's own return-with-a-value shape
///    (`StopIteration`'s `.value`) is outside this function's scope
///    (never read by `next()`'s own first-value contract).
/// 2. `for <name> in <literal iterable>: yield <expr>` — a-statements.py's
///    `stream()` shape (`for value in (10, 20, 30): yield value`,
///    wrapped in `async def` — this domain collapses `for`/`async for`
///    into the identical `StmtFor` node, ruff's own generated.rs doc:
///    "collapses the synchronous and asynchronous variants into a
///    single type"). Modeled ONLY when the loop's own iterable reads
///    through `literal_iterable_values` below (a literal list/tuple of
///    number literals, or `range(...)` with int-literal args — the same
///    two syntactic shapes `loops.rs`'s own reader accepts, reimplemented
///    LOCALLY per this addendum's own scope rather than importing that
///    file), the target a bare Name, the body EXACTLY one `yield <expr>`
///    statement, and no `else` clause (a `for...else` is outside this
///    shape). Each element binds the SAME environment in turn (the
///    elements are already fully known, so no branch of the walk can
///    see a stale binding) — parameters and any prior straight-line
///    bindings stay visible to the yield expression, matching CPython's
///    own left-to-right iteration order (compound_stmts.rst, "The `for`
///    statement").
///
/// Any other statement shape (an `if`, a `while`, a nested `for` whose
/// iterable is not one of the two literal forms, a `for` whose body is
/// not exactly one `yield`, …) declines the WHOLE body — `None`, never a
/// partial list. A CONDITIONAL yield (`if <test>: yield <expr>`) is a
/// deliberate, permanent decline — q-decline-names.py's own
/// `age_generator` row states this as one of its file's two genuine
/// soundness boundaries: "a generator whose yield sits under a
/// CONDITION is beyond the straight-line summary the checker reads."
/// This function must never join an `if`/`else`'s own yields into one
/// answer, even though the values involved would often be sound to
/// join — the row's own purpose is to teach that this shape stays
/// undetermined.
///
/// `arguments`/`table`/`kernel`/`depth` mirror `summaries::call_result`
/// exactly (parameters bind positionally, the module's function table
/// composes a nested same-module call, the depth cap declines a runaway
/// chain) — a generator's parameter list is bound exactly like an
/// ordinary function's own.
pub fn generator_yields(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> Option<Vec<AbstractValue>> {
    use crate::summaries::CALL_DEPTH_CAP;
    if depth >= CALL_DEPTH_CAP {
        return None;
    }
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() || !def.parameters.kwonlyargs.is_empty() {
        return None;
    }
    let parameters: Vec<_> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .collect();
    if arguments.len() > parameters.len() {
        return None;
    }
    let mut locally_bound = std::collections::HashSet::new();
    for parameter in &parameters {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    let mut environment = Environment::new(locally_bound);
    // one call deeper than the caller — the depth cap engages across
    // the evaluate↔interpreter boundary (see env::call_depth)
    environment.set_call_depth(depth.saturating_add(1));
    if let Some(table) = table {
        environment.set_functions(table.clone());
    }
    let default_environment = Environment::new(Default::default());
    for (index, parameter) in parameters.iter().enumerate() {
        let value = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            let default_expr = parameter.default.as_deref()?;
            evaluate_expression(default_expr, &default_environment, kernel)
        };
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }

    yields_of_body(&def.body, &mut environment, kernel)
}

/// `generator_yields`'s own body walk, over `def.body` — see that
/// function's own doc for the two straight-line-yield shapes, PLUS the
/// one CONDITIONAL shape this function now summarizes: `if <test>: yield
/// <expr>` with no `elif`/`else` clause and no other statement in the
/// `if`'s own body. CPython's real generator-iterator protocol either
/// runs that `yield` (the test is true on this pass) or skips straight to
/// whatever statement follows it (the test is false) — this function has
/// no way to decide WHICH, so it states the sound over-approximation for
/// "the next value `__next__` could produce at this position": the JOIN
/// of the conditional yield's own value with whatever value the REST of
/// the body would produce if this position were skipped entirely
/// (`yields_of_body`'s own recursive call over the statements after the
/// `if`). A conditional yield followed by more yields therefore never
/// widens the overall yielded COUNT — it only widens the VALUE at the one
/// position where the branch and its continuation compete to be "the
/// value read there" (`age_generator`'s own row: `if bool([]): yield 40`
/// then `yield 41` answers ONE position, `join(40, 41)`, never two
/// separate positions). A conditional yield with NOTHING after it (no
/// unconditional yield anywhere later in the body) still declines — the
/// join needs a second value to join against, and a length-zero-or-one
/// generator is a shape this function does not spell (its own `Vec` return
/// has no way to say "zero or one," only "exactly N" — the caller reading
/// `items.first()` for `next()` would otherwise wrongly treat a possibly-
/// empty position as always-present).
///
/// A LEADING docstring (a bare string-literal `Expr` statement,
/// `summaries::first_non_docstring_statement`'s own shape) is skipped
/// before the walk starts — a docstring is documentation, never a
/// readable effect (that function's own doc), so a generator whose body
/// opens with one must not decline solely because its first statement is
/// not `Expr::Yield`.
fn yields_of_body(body: &[Stmt], environment: &mut Environment, kernel: &Arc<RefinedTSKernel>) -> Option<Vec<AbstractValue>> {
    let Some(first) = crate::summaries::first_non_docstring_statement(body) else {
        // nothing but leading docstrings — no yield anywhere
        return Some(Vec::new());
    };
    let skip = body.iter().position(|stmt| std::ptr::eq(stmt, first)).expect("first came from this same body");
    let body = &body[skip..];
    let mut yields = Vec::new();
    for (position, stmt) in body.iter().enumerate() {
        match stmt {
            // `if <test>: yield <expr>` — no `elif`/`else`, exactly one
            // statement in the `if`'s own body — this function's own doc
            // states the join this arm computes. `continuation` is every
            // statement AFTER this `if` in source order, summarized
            // recursively (a FRESH docstring-skip is harmless here: there
            // is no docstring mid-body to skip, `first_non_docstring_
            // statement` simply returns the continuation's own first
            // statement unchanged). `None` from EITHER the conditional
            // arm's own value or the continuation still declines the
            // whole body — this join is sound only when both sides of it
            // are themselves fully known.
            Stmt::If(if_stmt) if if_stmt.elif_else_clauses.is_empty() => {
                let [Stmt::Expr(if_body_expr_stmt)] = if_stmt.body.as_slice() else {
                    return None;
                };
                let Expr::Yield(if_yield_expr) = if_body_expr_stmt.value.as_ref() else {
                    return None;
                };
                let conditional_value = match if_yield_expr.value.as_deref() {
                    Some(value_expr) => evaluate_expression(value_expr, environment, kernel),
                    None => refined_domain::abstract_value::null_value(),
                };
                if conditional_value.kind == Kind::Unknown {
                    return None;
                }
                let continuation = yields_of_body(&body[position + 1..], environment, kernel)?;
                let mut continuation = continuation.into_iter();
                let Some(next_value) = continuation.next() else {
                    // nothing yielded after this conditional position — the
                    // real generator sometimes yields nothing AT ALL past
                    // here (StopIteration on the very first `__next__`
                    // call), a length-zero-or-one shape this function's own
                    // `Vec` return cannot spell (see this function's own
                    // doc) — decline rather than claim a length this
                    // reading did not prove.
                    return None;
                };
                yields.push(refined_domain::lattice_operations::join_known(conditional_value, next_value));
                yields.extend(continuation);
                return Some(yields);
            }
            Stmt::Expr(expr_stmt) => {
                let Expr::Yield(yield_expr) = expr_stmt.value.as_ref() else {
                    return None;
                };
                let value = match yield_expr.value.as_deref() {
                    Some(value_expr) => evaluate_expression(value_expr, environment, kernel),
                    None => refined_domain::abstract_value::null_value(),
                };
                if value.kind == Kind::Unknown {
                    return None;
                }
                yields.push(value);
            }
            // a bare `return` inside a generator ends iteration without
            // yielding (datamodel.rst's generator-function entry) — no
            // more statements after it are read, and a straight-line
            // body's own return-with-a-value shape (`StopIteration`'s
            // `.value`) is outside this function's scope (never read by
            // `next()`'s own first-value contract).
            Stmt::Return(_) => break,
            // `for <name> in <literal iterable>: yield <expr>` — see
            // this function's own doc, shape 2.
            Stmt::For(for_stmt) => {
                if !for_stmt.orelse.is_empty() {
                    return None;
                }
                let Expr::Name(target_name) = for_stmt.target.as_ref() else {
                    return None;
                };
                let [Stmt::Expr(body_expr_stmt)] = for_stmt.body.as_slice() else {
                    return None;
                };
                let Expr::Yield(yield_expr) = body_expr_stmt.value.as_ref() else {
                    return None;
                };
                let Some(value_expr) = yield_expr.value.as_deref() else {
                    return None;
                };
                let elements = literal_iterable_values(for_stmt.iter.as_ref())?;
                for element in elements {
                    environment.bind(target_name.id.as_str(), element);
                    let value = evaluate_expression(value_expr, environment, kernel);
                    if value.kind == Kind::Unknown {
                        return None;
                    }
                    yields.push(value);
                }
            }
            _ => return None,
        }
    }
    Some(yields)
}

/// The elements a generator's own `for <target> in <iterable>: yield
/// <expr>` shape iterates over, restricted to the two syntactic forms
/// this addendum reads: a `List`/`Tuple` DISPLAY of bare number literals
/// (`(10, 20, 30)`, `literal_number_elements`'s own literal-only
/// reading — an element that is not a bare number literal declines the
/// WHOLE iterable rather than falling back to a wider evaluated read,
/// since this reader is deliberately the SMALL syntactic subset the
/// addendum scopes it to), or `range(...)` with 1-3 INT-literal
/// arguments (`range` rejects a float argument at call time — the same
/// restriction `loops.rs`'s own `int_literal_value` states). Every
/// produced value is Integer- or Float-sorted per its own literal syntax
/// (never a joined `PrimitiveKind::Number`). `None` for any other
/// iterable shape — a name, a call to anything but `range`, a
/// non-literal element — this reader declines rather than guess.
fn literal_iterable_values(iterable: &Expr) -> Option<Vec<AbstractValue>> {
    match iterable {
        Expr::List(list) => literal_number_elements(&list.elts),
        Expr::Tuple(tuple) => literal_number_elements(&tuple.elts),
        Expr::Call(call) => literal_range_values(call),
        _ => None,
    }
}

/// Every element of a `List`/`Tuple` display read as a bare (optionally
/// unary +/- wrapped) number literal — `None` the moment one element is
/// not that exact shape.
fn literal_number_elements(elements: &[Expr]) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::with_capacity(elements.len());
    for element in elements {
        values.push(literal_number_value(element)?);
    }
    Some(values)
}

/// A bare (possibly unary +/- wrapped) `NumberLiteral`'s exact value,
/// tagged with its own CPython sort — the same reading `loops.rs`'s own
/// `sorted_number_literal_value` gives, reimplemented locally per this
/// function's own module (the addendum's own "do NOT import loops.rs").
fn literal_number_value(expression: &Expr) -> Option<AbstractValue> {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved)),
            Number::Float(value) => Some(known_values(vec![*value], PrimitiveKind::Float, TrustProved)),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_number_value(unary.operand.as_ref())?;
            let sort = operand.kind_tag?;
            let value = operand.values.first().copied()?;
            let signed = if unary.op == UnaryOp::USub { -value } else { value };
            Some(known_values(vec![signed], sort, TrustProved))
        }
        _ => None,
    }
}

/// A `range(...)` call's produced Integer-sorted values, `None` when the
/// callee is not the bare name `range`, an argument is not an INT
/// literal, the argument count is not 1/2/3, or the step is 0 — the same
/// reading `loops.rs`'s own `range_call_values` gives, reimplemented
/// locally (this function's own module owns no dependency on `loops.rs`
/// per the addendum's scope).
fn literal_range_values(call: &ExprCall) -> Option<Vec<AbstractValue>> {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "range" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let args = &call.arguments.args;
    let (start, stop, step) = match args.len() {
        1 => (0.0, literal_int_value(&args[0])?, 1.0),
        2 => (literal_int_value(&args[0])?, literal_int_value(&args[1])?, 1.0),
        3 => (
            literal_int_value(&args[0])?,
            literal_int_value(&args[1])?,
            literal_int_value(&args[2])?,
        ),
        _ => return None,
    };
    if step == 0.0 {
        return None;
    }
    let mut values = Vec::new();
    let mut current = start;
    // r[i] = start + step*i, while r[i] < stop (step > 0) or r[i] > stop
    // (step < 0) — library/stdtypes.rst's own range formula
    if step > 0.0 {
        while current < stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    } else {
        while current > stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    }
    Some(values)
}

/// A `range()` argument's value, restricted to an INT literal — `range`
/// rejects a float argument at call time, so this reader stays honest
/// about that CPython restriction rather than silently truncating.
fn literal_int_value(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            _ => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_int_value(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
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
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
            spelling: "Age".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    fn integer_value(v: f64) -> AbstractValue {
        known_values(vec![v], PrimitiveKind::Integer, TrustProved)
    }

    /// A hand-built `ClassModel` with no properties and no methods —
    /// every direct `judge_construction`/`field_write_judgment` test
    /// builds a model this way rather than parsing source, since those
    /// functions take the model, not the class definition.
    fn bare_model(name: &str, fields: Vec<ClassField>) -> ClassModel {
        ClassModel {
            name: name.to_owned(),
            fields,
            properties: HashMap::new(),
            methods: HashMap::new(),
            parent_methods: HashMap::new(),
            class_attributes: Vec::new(),
        }
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
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let person = table.get("Person").expect("Person class recorded");
        assert_eq!(person.fields.len(), 2);
        assert_eq!(person.fields[0].name, "age");
        assert!(person.fields[0].declared.is_some(), "age reads its Annotated set");
        assert_eq!(person.fields[1].name, "label");
        assert!(person.fields[1].declared.is_none(), "bare str states no refinement");
    }

    // --- typed_dict_table: per-member refinements ---

    #[test]
    fn typed_dict_table_reads_each_members_own_refinement() {
        let module = parsed(concat!(
            "from typing import Annotated, TypedDict\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class PersonDict(TypedDict):\n",
            "    age: Age\n",
            "    label: str\n",
        ));
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
        let table = typed_dict_table(&module, &aliases, &imports);
        let members = table.get("PersonDict").expect("PersonDict recorded");
        assert_eq!(members.len(), 1, "only age reads a refinement; bare str states none");
        assert_eq!(members[0].0, "age");
        assert_eq!(members[0].1.spelling, "Age");
    }

    #[test]
    fn typed_dict_table_ignores_a_class_with_no_typed_dict_base() {
        let module = parsed(concat!(
            "from pydantic import BaseModel\n",
            "class Person(BaseModel):\n",
            "    age: int\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = typed_dict_table(&module, &aliases, &imports);
        assert!(table.get("Person").is_none(), "a plain BaseModel class is not a TypedDict");
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
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let counted = table.get("Counted").expect("Counted class recorded");
        assert_eq!(counted.fields.len(), 1, "ClassVar row must not become a field");
        assert_eq!(counted.fields[0].name, "age");
    }

    // --- a field annotated with another module-level BaseModel class ---

    /// m-pydantic-schema.py's own `Resident.address: Address` shape:
    /// `Address` is a class, not a `type` alias, so `declared_refinement`'s
    /// bare-Name arm reads nothing for it — `class_model_of`'s own
    /// `.or_else` fallback must build `Address`'s member table instead.
    /// `Resident`'s field carries `members: Some(...)` with `zip_code`'s
    /// own declared set, so a later `judge_construction`/MEMBERS LAW
    /// judgment of a nested dict can see past the bare class name.
    #[test]
    fn class_model_of_reads_a_field_annotated_with_another_module_level_class() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, BaseModel\n",
            "type ZipCode = Annotated[str, Field(min_length=5, max_length=5)]\n",
            "class Address(BaseModel):\n",
            "    zip_code: ZipCode\n",
            "class Resident(BaseModel):\n",
            "    address: Address\n",
        ));
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let resident = table.get("Resident").expect("Resident class recorded");
        let address_field = resident.fields.iter().find(|field| field.name == "address").expect("address field present");
        let declared = address_field.declared.as_ref().expect("Address reads as a member-carrying declaration");
        let members = declared.members.as_ref().expect("a class-typed field carries a per-member table");
        let zip_code = members.iter().find(|(name, _)| name == "zip_code").expect("zip_code member present");
        assert_eq!(zip_code.1.spelling, "ZipCode");
    }

    /// The same shape one level deeper: `Resident.person: Person` where
    /// `Person` is ITSELF a BaseModel with a refined field — nested
    /// membership recurses because `Person` was built through the same
    /// lazy `build_class_model` call, so its own `declared` already
    /// carries `members: Some(...)` by the time `Resident`'s field reads it.
    #[test]
    fn class_model_of_reads_a_doubly_nested_member_class() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from typing import Annotated\n",
            "from pydantic import Field, BaseModel\n",
            "type Age = Annotated[int, Field(ge=0, le=120)]\n",
            "class Person(BaseModel):\n",
            "    age: Age\n",
            "class Resident(BaseModel):\n",
            "    person: Person\n",
        ));
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let resident = table.get("Resident").expect("Resident class recorded");
        let person_field = resident.fields.iter().find(|field| field.name == "person").expect("person field present");
        let declared = person_field.declared.as_ref().expect("Person reads as a member-carrying declaration");
        let members = declared.members.as_ref().expect("a class-typed field carries a per-member table");
        let age = members.iter().find(|(name, _)| name == "age").expect("age member present");
        assert_eq!(age.1.spelling, "Age");
    }

    /// A field annotated with a class name the module never declares
    /// (a typo, or a class defined in another module this table cannot
    /// see) declines exactly as before this unit — `declared: None`,
    /// never a guessed member table.
    #[test]
    fn class_model_of_field_annotated_with_an_unknown_name_stays_undeclared() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from pydantic import BaseModel\n",
            "class Resident(BaseModel):\n",
            "    address: Missing\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let resident = table.get("Resident").expect("Resident class recorded");
        let address_field = resident.fields.iter().find(|field| field.name == "address").expect("address field present");
        assert!(address_field.declared.is_none(), "an undeclared class name states nothing this table reads");
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

    // --- model_post_init: the dependent-check hook ---

    /// m-pydantic-schema.py's own `Range` shape: `model_post_init(self,
    /// __context): if self.hi < self.lo: raise ValueError(...)`. A
    /// construction whose fields provably satisfy `hi >= lo` never
    /// fires here — the post-init condition reads False.
    #[test]
    fn model_post_init_is_silent_when_the_dependent_check_passes() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Range:\n",
            "    lo: int\n",
            "    hi: int\n",
            "    def model_post_init(self, __context) -> None:\n",
            "        if self.hi < self.lo:\n",
            "            raise ValueError(\"hi must be >= lo\")\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let range_model = table.get("Range").expect("Range class recorded");
        let keyword = vec![
            ("lo".to_owned(), integer_value(10.0), range_of("10")),
            ("hi".to_owned(), integer_value(20.0), range_of("20")),
        ];
        let verdict = judge_construction(range_model, &[], &keyword, &kernel);
        assert!(verdict.fires.is_empty(), "hi (20) >= lo (10): the dependent check never raises");
    }

    /// The refused pair: `hi` (5) below `lo` (10) — the post-init
    /// condition provably reads True, so construction fires with the
    /// `ValueError`'s own message.
    #[test]
    fn model_post_init_fires_when_the_dependent_check_provably_raises() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Range:\n",
            "    lo: int\n",
            "    hi: int\n",
            "    def model_post_init(self, __context) -> None:\n",
            "        if self.hi < self.lo:\n",
            "            raise ValueError(\"hi must be >= lo\")\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let range_model = table.get("Range").expect("Range class recorded");
        let keyword = vec![
            ("lo".to_owned(), integer_value(10.0), range_of("10")),
            ("hi".to_owned(), integer_value(5.0), range_of("5")),
        ];
        let verdict = judge_construction(range_model, &[], &keyword, &kernel);
        assert_eq!(verdict.fires.len(), 1, "hi (5) < lo (10): the dependent check provably raises");
        assert!(verdict.fires[0].1.contains("ValueError"), "{}", verdict.fires[0].1);
        assert!(verdict.fires[0].1.contains("hi must be >= lo"), "{}", verdict.fires[0].1);
    }

    /// An undetermined field (no keyword argument at all, so `hi`/`lo`
    /// both hold the declared int base sort, not a concrete value)
    /// never fires — `truthiness` cannot decide the condition, and this
    /// reader's own honest-decline discipline never guesses.
    #[test]
    fn model_post_init_never_fires_on_an_undetermined_condition() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Range:\n",
            "    lo: int\n",
            "    hi: int\n",
            "    def model_post_init(self, __context) -> None:\n",
            "        if self.hi < self.lo:\n",
            "            raise ValueError(\"hi must be >= lo\")\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let range_model = table.get("Range").expect("Range class recorded");
        let verdict = judge_construction(range_model, &[], &[], &kernel);
        assert!(verdict.fires.is_empty(), "an undetermined comparison never guesses a fire");
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
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
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
        let imports = crate::surface::surface_imports(&module);
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
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
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
        let imports = crate::surface::surface_imports(&module);
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
        let imports = crate::surface::surface_imports(&module);
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
        let imports = crate::surface::surface_imports(&module);
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
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
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
        let imports = crate::surface::surface_imports(&module);
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
        let imports = crate::surface::surface_imports(&module);
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
        let aliases = crate::surface::compile_aliases(&module);
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let aged = table.get("Aged").expect("Aged class recorded");
        assert!(aged.properties["age"].declared.is_some(), "the setter's Age annotation is read");

        let verdict = field_write_judgment(aged, "age", &integer_value(200.0), &kernel);
        assert!(matches!(verdict, Some(Verdict::Fire(_))), "200 fires against the setter's own Age set");
    }

    // --- ClassModel is Clone ---

    #[test]
    fn class_model_clones_its_fields_properties_and_methods() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Aged:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "    def next_year(self) -> int:\n",
            "        return self.age + 1\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let aged = table.get("Aged").expect("Aged class recorded");
        let cloned = aged.clone();
        assert_eq!(cloned.name, aged.name);
        assert_eq!(cloned.fields.len(), aged.fields.len());
        assert!(cloned.methods.contains_key("next_year"), "the clone keeps the method table");
    }

    // --- method_def_of: own-overrides-inherited ---

    #[test]
    fn method_def_of_reads_a_class_own_method() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Aged:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "    def next_year(self) -> int:\n",
            "        return self.age + 1\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let aged = table.get("Aged").expect("Aged class recorded");
        let method = method_def_of(aged, "next_year").expect("next_year is a declared method");
        assert_eq!(method.name.id.as_str(), "next_year");
        assert!(method_def_of(aged, "missing").is_none());
    }

    /// A child overriding a parent's method: `method_def_of` on the
    /// child answers the CHILD's own def (its body differs from the
    /// parent's — `label` returns 2, not the parent's 1), while
    /// `parent_methods` still carries the parent's original — the
    /// `super()` resolution target, proven by running both defs through
    /// `method_call_result` and comparing their answers.
    #[test]
    fn method_def_of_prefers_the_childs_own_override_over_the_inherited_def() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class BaseYears:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "    def label(self) -> int:\n",
            "        return 1\n",
            "class KidYears(BaseYears):\n",
            "    def __init__(self, age: int) -> None:\n",
            "        super().__init__(age)\n",
            "    def label(self) -> int:\n",
            "        return 2\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let kid = table.get("KidYears").expect("KidYears class recorded");
        let instance = judge_construction(kid, &[(integer_value(40.0), range_of("40"))], &[], &kernel).instance;

        let effective = method_def_of(kid, "label").expect("label is declared");
        let (_after, effective_result) = method_call_result(&instance, kid, effective, &[], None, None, None, &kernel, 0)
            .expect("the child's own label() must interpret");
        assert_eq!(effective_result, integer_value(2.0), "method_def_of answers the CHILD's own override");

        let inherited = kid.parent_methods.get("label").expect("parent_methods keeps the parent's own def");
        let (_after, inherited_result) = method_call_result(&instance, kid, inherited, &[], None, None, None, &kernel, 0)
            .expect("the parent's own label() must interpret");
        assert_eq!(inherited_result, integer_value(1.0), "parent_methods is unaffected by the child's override");
    }

    // --- method_call_result: write-then-read, and the super() chain ---

    /// `outlaw.spoil()` where `spoil` writes `self.age = 200` and reads
    /// nothing back itself — the RETURNED instance must carry the
    /// write, matching b-body-expressions.py's own
    /// `literal_writing_method` shape (ORIENTATION.md's own citation for
    /// `method_call_result`).
    #[test]
    fn method_call_result_write_then_read_survives_on_the_returned_instance() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class Outlaw:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "    def spoil(self) -> None:\n",
            "        self.age = 200\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let outlaw = table.get("Outlaw").expect("Outlaw class recorded");
        let instance = judge_construction(outlaw, &[(integer_value(40.0), range_of("40"))], &[], &kernel).instance;
        let method = method_def_of(outlaw, "spoil").expect("spoil is declared");
        let (after, _result) = method_call_result(&instance, outlaw, method, &[], None, None, None, &kernel, 0)
            .expect("spoil's straight-line self-write must interpret");
        assert_eq!(field_read(&after, "age"), Some(integer_value(200.0)), "the write survives on the returned instance");
    }

    /// `KidYears(age=200).years()` where `years` calls
    /// `super().years() + 1` — the parent's OWN `years` (never the
    /// child's, since `KidYears` declares no override of that name)
    /// answers through `parent_methods`, and the child's own method adds
    /// 1 to it.
    #[test]
    fn method_call_result_resolves_a_super_call_through_parent_methods() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "class BaseYears:\n",
            "    def __init__(self, age: int) -> None:\n",
            "        self.age = age\n",
            "    def years(self) -> int:\n",
            "        return self.age\n",
            "class KidYears(BaseYears):\n",
            "    def __init__(self, age: int) -> None:\n",
            "        super().__init__(age)\n",
            "    def call_super_method(self) -> int:\n",
            "        return super().years() + 1\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let kid = table.get("KidYears").expect("KidYears class recorded");
        let instance = judge_construction(kid, &[(integer_value(40.0), range_of("40"))], &[], &kernel).instance;
        let method = method_def_of(kid, "call_super_method").expect("call_super_method is declared");
        let (_after, result) = method_call_result(&instance, kid, method, &[], None, None, None, &kernel, 0)
            .expect("the super().years() call must resolve through parent_methods");
        assert_eq!(result, integer_value(41.0), "super().years() answers 40, plus 1");
    }

    /// A class method body's own `from datetime import date`-aliased
    /// `date(2024, 3, 1)` construction recognizes IDENTICALLY to the same
    /// call in a plain function (`expressions.rs`'s own
    /// `test_bare_imported_date_construction_matches_the_qualified_
    /// spelling`, mirrored here for a method body): `method_call_result`
    /// is handed the module's own `datetime_imports` table explicitly
    /// (the same explicit-parameter shape `classes` already takes), so
    /// the method body's fresh `Environment` resolves the bare `date`
    /// alias exactly as a module-level call would, rather than reading
    /// no table at all and declining the construction.
    #[test]
    fn method_body_bare_imported_date_construction_matches_the_qualified_spelling() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "from datetime import date\n",
            "class Anniversary:\n",
            "    def __init__(self, count: int) -> None:\n",
            "        self.count = count\n",
            "    def occasion(self):\n",
            "        return date(2024, 3, 1)\n",
        ));
        let aliases = HashMap::new();
        let imports = crate::surface::surface_imports(&module);
        let table = class_table(&module, &aliases, &imports, &kernel);
        let anniversary = table.get("Anniversary").expect("Anniversary class recorded");
        let instance =
            judge_construction(anniversary, &[(integer_value(1.0), range_of("1"))], &[], &kernel).instance;
        let method = method_def_of(anniversary, "occasion").expect("occasion is declared");
        let datetime_imports = Arc::new(crate::expressions::datetime_imports(&module));

        let plain_environment = {
            let mut environment = Environment::new(Default::default());
            environment.set_datetime_imports(Arc::new(crate::expressions::datetime_imports(&module)));
            environment
        };
        let plain_parsed = ruff_python_parser::parse_expression("date(2024, 3, 1)").expect("test source parses");
        let plain_value = crate::expressions::evaluate_expression(&plain_parsed.into_expr(), &plain_environment, &kernel);

        let (_after, method_value) =
            method_call_result(&instance, anniversary, method, &[], None, None, Some(&datetime_imports), &kernel, 0)
                .expect("the method body's own date(...) construction must interpret");

        assert_eq!(method_value.kind, Kind::Object);
        assert_eq!(method_value, plain_value, "a method body's aliased date(...) construction must equal the same call in a plain function");
    }

    // --- field_write: the source tag survives ---

    #[test]
    fn field_write_preserves_the_instances_source_tag() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model(
            "Aged",
            vec![ClassField { name: "age".to_owned(), declared: None, default: None }],
        );
        let verdict = judge_construction(&model, &[(integer_value(40.0), range_of("40"))], &[], &kernel);
        assert_eq!(verdict.instance.source, "Aged", "judge_construction tags the instance with the class name");
        let written = field_write(&verdict.instance, "age", integer_value(41.0)).expect("write must decide");
        assert_eq!(written.source, "Aged", "the source tag survives a field write");
        assert_eq!(field_read(&written, "age"), Some(integer_value(41.0)));
    }

    // --- judge_construction: instance_identity ---

    /// Two separate construction calls of the SAME class each mint their
    /// own `instance_identity` — a dict keyed by one instance must not
    /// answer a lookup by the other (`collection_models::known_dict_key`'s
    /// own identity arm reads this field to tell two `Holder()` calls
    /// apart, the way `env.rs`'s `next_retained_callable_key` already
    /// tells two lambda/def creations apart).
    #[test]
    fn judge_construction_mints_a_distinct_instance_identity_per_call() {
        let Some(kernel) = loaded_kernel() else { return };
        let model = bare_model("Holder", Vec::new());
        let first = judge_construction(&model, &[], &[], &kernel).instance;
        let second = judge_construction(&model, &[], &[], &kernel).instance;
        assert!(first.instance_identity.is_some(), "a constructed instance carries an identity");
        assert!(second.instance_identity.is_some(), "a constructed instance carries an identity");
        assert_ne!(
            first.instance_identity, second.instance_identity,
            "two separate Holder() calls must not mint the same identity"
        );
    }

    // --- generator_yields: the stream() for-loop shape ---

    /// `async def stream(): for value in (10, 20, 30): yield value` —
    /// a-statements.py:547-549's own shape: a generator whose only
    /// statement is a `for` loop over a literal tuple, yielding the
    /// loop target unmodified. `generator_yields` must answer all three
    /// yields, in order.
    #[test]
    fn generator_yields_reads_the_stream_for_loop_shape_in_order() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "async def stream():\n",
            "    for value in (10, 20, 30):\n",
            "        yield value\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the stream() for-loop shape must decide");
        assert_eq!(yields, vec![integer_value(10.0), integer_value(20.0), integer_value(30.0)]);
    }

    /// The same shape, but the yield expression TRANSFORMS the target
    /// (`yield value + 100`) — the per-iterate binding must be visible
    /// to the yield expression, not just a bare pass-through.
    #[test]
    fn generator_yields_evaluates_the_yield_expression_per_iterate() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def stream():\n",
            "    for value in [10, 20]:\n",
            "        yield value + 100\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the transformed-yield shape must decide");
        assert_eq!(yields, vec![integer_value(110.0), integer_value(120.0)]);
    }

    /// Straight-line top-level yields merge with the for-loop's own
    /// yields, in source order — the addendum's own "merged with any
    /// top-level yields in source order."
    #[test]
    fn generator_yields_merges_straight_line_and_for_loop_yields_in_source_order() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def mixed():\n",
            "    yield 1\n",
            "    for value in (2, 3):\n",
            "        yield value\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the mixed shape must decide");
        assert_eq!(yields, vec![integer_value(1.0), integer_value(2.0), integer_value(3.0)]);
    }

    /// A `for` loop whose iterable is NOT one of the two literal shapes
    /// (a bare name, here) declines the whole body — never a partial
    /// list.
    #[test]
    fn generator_yields_declines_a_for_loop_over_a_non_literal_iterable() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def stream(values):\n",
            "    for value in values:\n",
            "        yield value\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        assert!(generator_yields(&def, &[unknown()], None, &kernel, 0).is_none());
    }

    // --- generator_yields: a conditional yield joins with its continuation ---

    /// q-decline-names.py's own `age_generator` shape: `if bool([]): yield
    /// 40` with NO `else`, followed by an unconditional `yield 41`. Neither
    /// branch of the `if` is provably taken, so the position `next()` would
    /// read first is the JOIN of both outcomes — `{40, 41}` — never a
    /// decline: this is exactly the sound over-approximation
    /// `yields_of_body`'s own doc states for a conditional yield followed
    /// by an unconditional one.
    #[test]
    fn generator_yields_joins_a_conditional_yield_with_its_continuation() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def age_generator():\n",
            "    if bool([]):\n",
            "        yield 40\n",
            "    yield 41\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the conditional-then-unconditional shape must join");
        let [joined] = yields.as_slice() else {
            panic!("want exactly one joined position, got {}", yields.len());
        };
        assert_eq!(joined.kind, Kind::Values);
        let mut values = joined.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![40.0, 41.0]);
    }

    /// A conditional yield with NOTHING unconditional after it has no
    /// continuation to join against — the real generator sometimes
    /// produces nothing at all past this point, a length-zero-or-one
    /// shape `yields_of_body`'s own `Vec` return cannot spell. Still a
    /// genuine decline, distinct from the joined case above.
    #[test]
    fn generator_yields_declines_a_conditional_yield_with_no_continuation() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def maybe_yields():\n",
            "    if bool([]):\n",
            "        yield 40\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        assert!(
            generator_yields(&def, &[], None, &kernel, 0).is_none(),
            "a conditional yield with no unconditional yield after it has no continuation to join against"
        );
    }

    // --- generator_yields: a leading docstring is skipped ---

    /// A generator whose body opens with a docstring, then a plain
    /// straight-line `yield`, must summarize exactly as it would with no
    /// docstring at all — the docstring states no readable effect.
    #[test]
    fn generator_yields_skips_a_leading_docstring_before_a_straight_line_yield() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def documented():\n",
            "    \"\"\"a docstring, not a yield\"\"\"\n",
            "    yield 40\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        let yields = generator_yields(&def, &[], None, &kernel, 0)
            .expect("a leading docstring must not decline the body");
        assert_eq!(yields, vec![integer_value(40.0)]);
    }

    /// The same docstring-skip over the `for`-loop shape (shape 2) —
    /// the docstring sits before the loop, not inside it.
    #[test]
    fn generator_yields_skips_a_leading_docstring_before_a_for_loop_yield() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = parsed(concat!(
            "def documented_stream():\n",
            "    \"\"\"a docstring, not a yield\"\"\"\n",
            "    for value in (10, 20):\n",
            "        yield value\n",
        ));
        let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
        let yields = generator_yields(&def, &[], None, &kernel, 0)
            .expect("a leading docstring must not decline the for-loop shape");
        assert_eq!(yields, vec![integer_value(10.0), integer_value(20.0)]);
    }

}
