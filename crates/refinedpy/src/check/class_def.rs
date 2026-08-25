use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{
    AtomicNodeIndex, Expr, ModModule, Stmt, StmtClassDef, StmtFunctionDef,
};
use ruff_text_size::TextRange;

use crate::env::Environment;
use crate::instances;
use crate::instances::{class_table, judge_construction, ClassModel};
use crate::surface::AliasEntry;
use crate::typereading::{base_sort_return_refinement, declared_refinement, typed_dict_return_refinement, DeclaredRefinement};

use super::*;

/// BODY-LOCAL CLASSES, merged: `local_class_table`'s own build for
/// `body`, layered over `context.classes` (local name wins on a
/// spelling collision, the same base-wins rule `function_table::merged`
/// already applies) — returns `context.classes.clone()` UNCHANGED (an
/// `Arc` clone, no allocation) when `body` declares no local class at
/// all, so the common case (no body-local classes) costs nothing beyond
/// the empty scan.
pub(super) fn merged_classes_for_body(body: &[Stmt], context: &WalkContext) -> Arc<HashMap<String, ClassModel>> {
    let local_classes = local_class_table(body, context.aliases, context.imports, context.kernel);
    if local_classes.is_empty() {
        return context.classes.clone();
    }
    let mut merged_classes = (*context.classes).clone();
    for (name, model) in local_classes {
        merged_classes.insert(name, model);
    }
    Arc::new(merged_classes)
}

/// LOCAL CLASSES: this body's own top-level `class`s, read through
/// `instances::class_table`'s one public constructor over a synthetic
/// `ModModule` wrapping just those definitions — the exact construction
/// `local_function_table` already uses for a body-local `def`
/// (`cross_module.rs`'s `synthetic_module` pattern). Parent-linking via
/// `super().__init__(...)` only resolves against another class in the
/// SAME synthetic table, so a body-local class naming a MODULE-level
/// class as its base is read parent-less here — an acceptable narrowing
/// for a shape outside this wave's fixture rows, not a soundness gap
/// (a parent-less child still reads its own AnnAssign/`__init__`
/// fields correctly, only the inherited-field merge is skipped).
///
/// A class nested inside a NESTED `def` (`body`'s own top-level `def`
/// whose body declares a class one level further down — a nested
/// closure-factory shape returning an instance of a class local to
/// itself) is collected too: every top-level `Stmt::FunctionDef`'s body
/// is scanned the same way, recursively, so a class declared at any
/// nesting depth of nested defs is visible once its instance crosses
/// back out to an outer scope. A direct top-level class NAME wins over
/// a same-named class found one level deeper (the nearer scope shadows
/// the farther one, Python's own scoping rule).
pub(super) fn local_class_table(
    body: &[Stmt],
    aliases: &HashMap<String, AliasEntry>,
    imports: &crate::surface::SurfaceImports,
    kernel: &Arc<RefinedTSKernel>,
) -> HashMap<String, ClassModel> {
    let local_defs = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::ClassDef(def) => Some(Stmt::ClassDef(def.clone())),
            _ => None,
        })
        .collect();
    let synthetic = ModModule {
        node_index: AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: local_defs,
    };
    let mut classes = class_table(&synthetic, aliases, imports, kernel);
    for stmt in body {
        if let Stmt::FunctionDef(def) = stmt {
            let nested = local_class_table(&def.body, aliases, imports, kernel);
            for (name, model) in nested {
                classes.entry(name).or_insert(model);
            }
        }
    }
    classes
}

/// Whether `def`'s first parameter is named `self` — the corpus's own
/// receiver-naming convention (`instances.rs`'s `self_attribute_name`
/// doc makes the same assumption). A member `def` with no parameter at
/// all (a `@staticmethod`, out of this wave's scope) is not a bound
/// instance method this seeding law applies to.
pub(super) fn is_self_method(def: &StmtFunctionDef) -> bool {
    def.parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .next()
        .is_some_and(|parameter| parameter.parameter.name.id.as_str() == "self")
}

/// A class-body member `def` whose first parameter is `self`: walks
/// exactly like `walk_function_def` (its own `-> Annotation` reads
/// against the OUTER environment, its body walks fresh through
/// `walk_body`), except `self` seeds an INSTANCE built from `class`'s
/// own declared fields — `judge_construction`'s own construction path,
/// called with NO arguments so every field takes its default when
/// present, else its declared set (`known_set`, TrustSpec — the same
/// "declared set stands in for an unread value" law `seed_parameters`
/// already applies to an ordinary parameter), else `unknown()`. This is
/// the METHOD's own declared shape, not the value any particular call
/// site constructed with — sound because a method body reads `self`
/// long before this checker can know which call site reached it;
/// `judge_construction`'s own fires are discarded here (a field outside
/// its declared set is this synthesized self's own business, never a
/// finding — the mission's fires belong to an actual construction/write
/// site, not this seeding).
pub(super) fn walk_method_def(def: &StmtFunctionDef, class: &ClassModel, context: &WalkContext, out: &mut Vec<Finding>) {
    let outer_environment = Environment::new(HashSet::new());
    let return_refinement = def.returns.as_deref().and_then(|annotation| {
        declared_refinement(annotation, context.aliases, context.imports, &outer_environment)
            .or_else(|| typed_dict_return_refinement(annotation, &context.typed_dicts))
    });
    let (return_refinement, yield_refinement) = generator_body_refinements(def, return_refinement);
    let bare_sort_return_refinement = def.returns.as_deref().and_then(base_sort_return_refinement);
    let self_instance = judge_construction(class, &[], &[], context.kernel).instance;
    walk_body_with_self_binding(
        &def.body,
        Some(def.parameters.as_ref()),
        return_refinement.as_ref(),
        yield_refinement.as_ref(),
        None,
        Some(&self_instance),
        None,
        None,
        bare_sort_return_refinement.as_ref(),
        context,
        out,
    );
}

/// Splits a `def`'s own resolved `-> Annotation` refinement into the two
/// checked positions its BODY judges against, once the body is
/// GENERATOR-shaped (`is_generator_shaped`'s own doc — a `yield`
/// anywhere, straight-line or one level inside a `for`/`async for`).
/// `Generator[Y, S, R]`/`AsyncGenerator[Y, S]`/`Iterator[Y]`/`Iterable[Y]`
/// carry their two positions in `DeclaredRefinement::generator`
/// (`typereading.rs`'s own doc); every `yield <expr>` in this body
/// judges against `generator.yield_type`, every `return <expr>` against
/// `generator.return_type` (`None` for `AsyncGenerator`/`Iterator`/
/// `Iterable` — no return type is judged there at all, the same "no
/// annotation → no judging" rule `walk_return` already states). A
/// NON-generator body, or a generator-shaped body whose own `->
/// Annotation` did not read as one of the four generator forms
/// (`generator` is `None`), returns `declared` UNCHANGED as the return
/// position and no yield position — ordinary Python, nothing new judges.
pub(super) fn generator_body_refinements(
    def: &StmtFunctionDef,
    declared: Option<DeclaredRefinement>,
) -> (Option<DeclaredRefinement>, Option<DeclaredRefinement>) {
    if !is_generator_shaped(&def.body) {
        return (declared, None);
    }
    let Some(generator) = declared.and_then(|declared| declared.generator) else {
        return (None, None);
    };
    (generator.return_type, Some(generator.yield_type))
}

/// Whether `body` contains a `yield`/`yield from` anywhere that makes
/// CPython compile the enclosing `def` as a generator function
/// (datamodel.rst, "Generator functions") — the SAME routing fact
/// `expressions.rs::is_generator_def` reads for the call-answering side
/// of this feature, reimplemented locally per this file's own
/// "no importing across files for a one-line routing check" convention
/// (`loops.rs`'s own `generator_call_values` doc states the identical
/// precedent). Recognizes a top-level `Stmt::Expr(Expr::Yield |
/// Expr::YieldFrom)` and the same one-level-inside-a-`for`-loop nesting
/// `is_generator_def` reads (ruff collapses `for`/`async for` into one
/// `Stmt::For` node) — this is a ROUTING check only, not a claim about
/// which yields this checker can JUDGE: an unreadable yield shape still
/// walks through the ordinary blocker path once routed here.
pub(super) fn is_generator_shaped(body: &[Stmt]) -> bool {
    body.iter().any(|stmt| match stmt {
        Stmt::Expr(expr_stmt) => matches!(expr_stmt.value.as_ref(), Expr::Yield(_) | Expr::YieldFrom(_)),
        Stmt::For(for_stmt) => for_stmt.body.iter().any(|inner| {
            matches!(inner, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::Yield(_) | Expr::YieldFrom(_)))
        }),
        _ => false,
    })
}

/// A class body: walked as its own body (its own locally-bound prepass,
/// its own environment) — a class-level `AnnAssign` field judges
/// exactly like a module- or function-level one, and a `def` inside the
/// class body recurses as an ordinary function body through
/// `walk_statement`'s own `Stmt::FunctionDef` arm EXCEPT for a `self`-
/// taking member, which `walk_body`'s own `self_model` parameter routes
/// through `walk_method_def` instead (the self-seeding law). A class
/// body has no enclosing function, so it carries no return refinement
/// of its own (compound_stmts.rst, "Class definitions": the class body
/// executes in a new namespace with no relation to a function's own
/// scope).
///
/// `self_model`: `def`'s own `ClassModel`, looked up BY NAME out of
/// `enclosing_environment.classes()` — the table `walk_body`/
/// `walk_body_with_self_binding` already built and set on the
/// environment via `merged_classes_for_body` BEFORE this body's own
/// statement loop began dispatching (so `def`'s entry is already
/// present by the time `Stmt::ClassDef(def)` reaches this function).
/// This is the SAME table a sibling `super().__init__(...)`/
/// `super().<method>(...)` call already resolves parent links through
/// (module-level classes keep the full parent chain
/// `findings_for_module_with_resolver` built once over the WHOLE
/// module; a body-local class is parent-linked against any SIBLING
/// body-local class `local_class_table`'s own single build over the
/// whole enclosing body already covers) — looking the model up here,
/// rather than rebuilding a one-class synthetic table from `def` alone,
/// is what keeps `self_model.parent_methods`/inherited fields intact
/// for a self-seeded method body (`call_super_method`'s own
/// `super().years()` shape). `None` when the environment carries no
/// class table at all (should not occur — every walk sets one) or the
/// name is genuinely absent; `walk_body` itself tolerates `None`, a
/// class shape it somehow declines to model still walks its own body
/// with member defs falling back to the ordinary un-seeded
/// `walk_function_def` path.
pub(super) fn walk_class_def(def: &StmtClassDef, enclosing_environment: &mut Environment, context: &WalkContext, out: &mut Vec<Finding>) {
    // Cloning the Arc (cheap — a refcount bump, not a table copy) frees
    // this table from `enclosing_environment`'s own borrow, so the
    // class-object seed below can mutably bind into it while `self_model`
    // stays alive for `walk_body`'s own read afterward.
    let classes = enclosing_environment.classes().cloned();
    let self_model = classes.as_ref().and_then(|classes| classes.get(def.name.id.as_str()));
    // CLASS-OBJECT SEEDING: the class's own bare name becomes readable,
    // in THIS enclosing scope, as a tagged Kind::Object carrying its
    // class_attributes (`instances::class_object_value`'s own doc) — the
    // same environment slot `Counted.total = 40`/`Counted.total` (a
    // bare-Name attribute write/read, e-class-and-function.py's
    // `class_attribute_write`) already finds and rebinds through
    // `write_named_field`/`field_read_through_model`, with no separate
    // "class object" machinery needed there. A class with no
    // `class_attributes` at all still seeds an empty tagged object — a
    // later `SomeClass.new_attr = v` attribute GAIN is ordinary Python,
    // matching `field_write`'s own "an ordinary Python attribute gain is
    // not a blocker" doc.
    if let Some(model) = self_model {
        enclosing_environment.bind(def.name.id.as_str(), instances::class_object_value(model));
    }
    walk_body(&def.body, None, None, self_model, context, out);
}
