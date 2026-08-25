//! The class table build: every module-level class's declared shape,
//! parents built before children, `AnnAssign`/`__init__`/`super()`
//! merged into one field list, and the effective (override-aware)
//! method set.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{Expr, ModModule, Stmt, StmtClassDef, StmtFunctionDef};

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::surface::{AliasEntry, SurfaceImports};
use crate::typereading::declared_refinement;

use super::defaults::default_value_of;
use super::init_fields::{init_derived_fields, super_init_fields};
use super::properties::property_table;
use super::typed_dict::unwrap_required_marker;
use super::types::{ClassField, ClassModel};

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

/// Depth-first build of one class into `out`, building its single
/// bare-Name module-level parent first when one exists. `building`
/// guards an inheritance cycle (`class A(B): ...` / `class B(A):
/// ...`, which CPython itself rejects at class-creation time) from
/// infinitely recursing — a class already mid-build is read as
/// parent-less rather than looping.
pub(super) fn build_class_model(
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
pub(super) fn single_bare_name_base(def: &StmtClassDef) -> Option<&str> {
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
pub(super) fn class_model_of(
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
        // `Required[X]`/`NotRequired[X]` (typing.rst) is valid syntax
        // only inside a TypedDict body — `class_table` builds a
        // `ClassModel` for EVERY module-level class, TypedDict-based
        // ones included (`typed_dict_table`'s own separate reading is
        // the return-position/explicit-TypedDict-annotation path;
        // `seed_parameters` (check.rs) reads a bare `r: Record`
        // PARAMETER through `context.classes` FIRST, so THIS loop is
        // what actually seeds a TypedDict-typed parameter's own
        // per-field table) — so this same peel
        // (`unwrap_required_marker`'s own doc) applies here too,
        // unconditionally: an ordinary (non-TypedDict) class field never
        // spells the wrapper at all, so the peel is a no-op for every
        // other class shape.
        let annotation = unwrap_required_marker(assign.annotation.as_ref());
        let declared = declared_refinement(annotation, aliases, imports, &empty_environment)
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
                out.get(class_name.id.as_str()).map(super::member_refinement::model_members_refinement)
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
