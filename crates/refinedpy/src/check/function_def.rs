use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use refined_domain::abstract_value::{known_set, AbstractValue, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::{on_one_tuple_layer, requires_integer};
use ruff_python_ast::{
    AtomicNodeIndex, Expr, ModModule, Parameters, Stmt, StmtAssign, StmtFunctionDef, StmtReturn,
};
use ruff_text_size::{Ranged, TextRange};

use crate::assignability::states_sequence;
use crate::cross_module::{module_surface, ModuleResolver};
use crate::env::Environment;
use crate::function_table::{function_table, merged, FunctionTable};
use crate::instances;
use crate::instances::class_table;
use crate::surface::{compile_aliases, strict_int_alias_names, surface_imports};
use crate::typereading::{base_sort_return_refinement, declared_refinement, typed_dict_return_refinement};

use super::*;

/// `yield from <expr>`'s own delegate, read to the values it hands this
/// generator's caller. `<expr>` must be a bare same-module call
/// (`gen()`) — any other delegate shape (a non-Name callee, a call to a
/// name with no same-module `def`) declines whole, this row's own
/// blocker. Two readings, tried in order, mirroring refined-ts-go's own
/// `GeneratorElementOf` (walk/generator_element.go): the callee's own
/// ACTUAL yields, walked fresh through `instances::generator_yields`
/// (straight-line/single-for-loop bodies only — that function's own
/// doc), state the TIGHTEST claim this checker can prove (a callee
/// declared `-> Generator[int, None, None]` whose body only ever yields
/// `40` hands this delegation exactly `[40]`, not the unbounded `int`
/// ray its own annotation states) and are read FIRST for that reason;
/// the callee's own DECLARED yield set (`typereading::declared_refinement`'s
/// generator arm) is the fallback once the body-walk declines (a
/// conditional yield, a parameter-shaped body, any restricted-
/// interpreter shape `generator_yields` does not read) — the annotation
/// is still a real claim about every value the callee could ever hand
/// back, so a callee whose actual body this walk cannot run still judges
/// against what it PROMISES.
pub(super) fn delegated_generator_yields(
    delegate: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<Vec<AbstractValue>> {
    let Expr::Call(call) = delegate else {
        return None;
    };
    let Expr::Name(callee_name) = call.func.as_ref() else {
        return None;
    };
    let def = context.functions.def(callee_name.id.as_str())?;
    if let Some(yields) =
        instances::generator_yields(def, &[], environment.functions(), context.kernel, environment.call_depth())
    {
        return Some(yields);
    }
    let outer_environment = Environment::new(HashSet::new());
    let declared = declared_refinement(def.returns.as_deref()?, context.aliases, context.imports, &outer_environment)?;
    let yield_type = declared.generator?.yield_type;
    // Tags the numeric sort `min_max_scalar_operand`/`star_numeric_hull`/
    // `sum_call_over_star` (builtin_models.rs) read, the same guarded
    // rule `seed_parameters` applies to a declared set: numeric-ground
    // only (`on_one_tuple_layer` alone also reads a `Literal["A", "B"]`
    // string-tuple union as "on the one-tuple layer", so `states_sequence`
    // must be false too, ruling out that pun). A string/sequence-shaped
    // yield type is left untagged, unchanged from today.
    if on_one_tuple_layer(&yield_type.set) && !states_sequence(&yield_type.set) {
        let sort = if requires_integer(&yield_type.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        return Some(vec![AbstractValue {
            kind_tag: Some(sort),
            ..known_set(yield_type.set, None, TrustSpec, SetKindTag::None)
        }]);
    }
    Some(vec![known_set(yield_type.set, None, TrustSpec, SetKindTag::None)])
}

/// A nested `def`: reads its own `-> Annotation` (if any) through
/// `declared_refinement` against the OUTER environment (a return
/// annotation naming a module-level alias resolves the same way any
/// other annotation does; a locally-rebound name here states nothing,
/// same rule as every other annotation read) and walks its body fresh.
pub(super) fn walk_function_def(def: &StmtFunctionDef, context: &WalkContext, out: &mut Vec<Finding>) {
    let outer_environment = Environment::new(HashSet::new());
    let return_refinement = def.returns.as_deref().and_then(|annotation| {
        declared_refinement(annotation, context.aliases, context.imports, &outer_environment)
            .or_else(|| typed_dict_return_refinement(annotation, &context.typed_dicts))
    });
    let (return_refinement, yield_refinement) = generator_body_refinements(def, return_refinement);
    let bare_sort_return_refinement = def.returns.as_deref().and_then(base_sort_return_refinement);
    walk_body_with_self_binding(
        &def.body,
        Some(def.parameters.as_ref()),
        return_refinement.as_ref(),
        yield_refinement.as_ref(),
        None,
        None,
        None,
        Some(def.name.id.as_str()),
        bare_sort_return_refinement.as_ref(),
        context,
        out,
    );
}

/// Every top-level `def`'s own DERIVED return value, keyed by name: the
/// join of every value that def's `return` statements produce, from the
/// SAME walk `findings_for_module_with_resolver` runs over the module —
/// parameters seeded from their declarations by `seed_parameters`, the
/// body walked statement by statement, and each return's value taken
/// from the point `walk_return` already computed it for judging.
///
/// `fact_export`'s derivation seam, and its only entry into this file.
/// The module's shared context (aliases, imports, cross-module surface,
/// function/class tables — everything a body reads must be built the way
/// the checker builds it, never a narrower stand-in) is built ONCE here
/// and every def walks against it, so an N-def module resolves its
/// imports once rather than N times.
///
/// `derived_return_values`'s own answer: every def's derived return
/// value, keyed by name, PLUS — for a def with no entry in that map —
/// the first blocker sentence its own walk recorded, keyed the same way.
/// A def absent from both is a body this walk ran cleanly with no
/// blocker and no `return` statement at all (a bare `pass`, an `if`
/// with no branch reaching a return) — genuinely nothing to name,
/// distinct from a def the walk COULD NOT get through.
pub struct DerivedReturns {
    pub values: HashMap<String, AbstractValue>,
    pub blockers: HashMap<String, String>,
}

/// Every def walks on its own merits, whether or not the module states
/// any `type` alias or recognized `Annotated` import — a def with no
/// refinement vocabulary in scope still has a derivable return value
/// and a nameable blocker (`findings_for_module_at`'s own rule). A
/// def whose body produced no `return` value this walk could read is
/// simply absent from `values`
/// — never an entry holding a guessed value. `blockers` names the FIRST
/// construct that stopped a def's own walk (the same RTS7002 sentence
/// `findings_for_module` would report for this body), independent of
/// whether the def's `-> Annotation` was itself readable: an unreadable
/// return annotation gives `return_refinement` as `None`, which is
/// `walk_return`'s own signal to skip JUDGING a return value — it does
/// not, and must not, silence the blocker this body's own walk hit on
/// the way to that return (`walk_loop`/`walk_statement`'s
/// `record_blocker` calls are unconditional on `return_refinement`
/// already; what this function fixes is that its OWN discarded-findings
/// walk used to drop that recorded blocker on the floor rather than
/// handing it back to a caller that needs it).
pub fn derived_return_values(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
) -> DerivedReturns {
    derived_return_values_at(module, resolver, kernel, None)
}

/// `derived_return_values` plus the checked file's own directory — the
/// export seam's twin of `findings_for_module_at`. A relative argv
/// target inside a recognized foreign edge (`subprocess.run(["node",
/// "./audio_level.ts"], ...)`) resolves against this directory, exactly
/// as it does when the SAME body is walked for ordinary findings — a
/// def whose derivation crosses to TypeScript exports the same way a
/// def with no foreign edge does.
pub fn derived_return_values_at(
    module: &ModModule,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> DerivedReturns {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    // Every module reaches the walk, the same rule
    // `findings_for_module_at` keeps: what a def derives comes from its
    // own statements, so a def with no refinement vocabulary in its
    // module still has a derivable return value and a nameable blocker.
    let surface = module_surface(module, resolver, kernel);
    let own_functions = function_table(module);
    let functions = Arc::new(merged(&own_functions, surface.functions.as_ref()));
    let own_classes = class_table(module, &aliases, &imports, kernel);
    let mut classes = Arc::try_unwrap(surface.classes)
        .unwrap_or_else(|_| panic!("module_surface's own Arc<classes> has no other owner yet"));
    for (class_name, model) in own_classes {
        classes.insert(class_name, model);
    }
    for def in module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    }) {
        for (class_name, model) in local_class_table(&def.body, &aliases, &imports, kernel) {
            classes.insert(class_name, model);
        }
    }
    let module_callable_returns = Arc::new(module_level_callable_returns(module, &aliases, &imports));
    let strict_int_aliases = strict_int_alias_names(module);
    let typed_dicts = Arc::new(instances::typed_dict_table(module, &aliases, &imports));
    let caller_arguments = Arc::new(crate::function_table::caller_argument_positions(module));
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel,
        functions,
        classes: Arc::new(classes),
        datetime_imports: Arc::new(crate::expressions::datetime_imports(module)),
        locale_never_set: crate::expressions::module_never_calls_setlocale(module),
        module_bindings: module_bindings_with_math_imports(surface.bindings, module),
        module_callable_returns,
        strict_int_aliases: &strict_int_aliases,
        typed_dicts,
        caller_arguments,
        entry_directory: entry_directory.map(|dir| dir.to_path_buf()),
        evaluations_recorder: None,
        trace_collector: None,
    };
    let mut values = HashMap::new();
    let mut blockers = HashMap::new();
    for def in module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    }) {
        let outer_environment = Environment::new(HashSet::new());
        let return_refinement = def.returns.as_deref().and_then(|annotation| {
            declared_refinement(annotation, context.aliases, context.imports, &outer_environment)
                .or_else(|| typed_dict_return_refinement(annotation, &context.typed_dicts))
        });
        let (return_refinement, yield_refinement) = generator_body_refinements(def, return_refinement);
        let bare_sort_return_refinement = def.returns.as_deref().and_then(base_sort_return_refinement);
        let mut returned_values: Vec<AbstractValue> = Vec::new();
        let mut findings = Vec::new();
        walk_body_with_self_binding(
            &def.body,
            Some(def.parameters.as_ref()),
            return_refinement.as_ref(),
            yield_refinement.as_ref(),
            None,
            None,
            Some(&mut returned_values),
            Some(def.name.id.as_str()),
            bare_sort_return_refinement.as_ref(),
            &context,
            &mut findings,
        );
        let mut answers = returned_values.into_iter();
        let Some(first) = answers.next() else {
            // No `return` this walk could read a value from — named
            // here, independent of whether `-> Annotation` itself read
            // (a bare `-> float` leaves `return_refinement` `None`, but
            // `record_blocker`'s own call sites never gate on that: the
            // FIRST unwalkable construct this body's walk hit is still
            // right here in `findings`, exactly the RTS7002 sentence
            // `findings_for_module` would report for the same body).
            if let Some(blocker) = findings.iter().find(|finding| finding.code == "RTS7002") {
                blockers.insert(def.name.id.as_str().to_owned(), blocker.message.clone());
            }
            continue;
        };
        values.insert(
            def.name.id.as_str().to_owned(),
            answers.fold(first, refined_domain::lattice_operations::join_known),
        );
    }
    DerivedReturns { values, blockers }
}

/// LOCAL DEFS: this body's own top-level `def`s (not a nested body's —
/// `function_table`'s own scan only reads `module.body`'s own top-level
/// statements, and this function reuses that exact scan by wrapping
/// `body` in a synthetic `ModModule`, the same construction
/// `cross_module.rs`'s `synthetic_module`/`rename_def` pair already uses
/// to turn an assembled def collection into a real `FunctionTable`
/// through `function_table`'s one public constructor). A `def` nested
/// inside a function/`if`/`for`/`try`/... body is otherwise invisible to
/// `context.functions` (the MODULE-level table `findings_for_module`
/// built once, before any body walk) — `nested_function_definition`'s
/// `years`/`over_years` rows, and `b-body-expressions.py`'s own local
/// defs, need their own body's defs merged in, local name winning over
/// an outer/imported name of the same spelling (`function_table::
/// merged`'s existing base-wins rule, reused unchanged here with the
/// LOCAL table as `base`).
///
/// LAMBDA-ASSIGN LAW (b-body-expressions.py:578/581's own call sites):
/// `f = lambda x: <expr>` assigned to a bare name is ALSO recorded here,
/// as a synthetic `StmtFunctionDef` (`lambda_as_synthetic_def`) — so a
/// later `f(...)` call resolves through `environment.functions()` and
/// `summaries::call_result` exactly like an ordinary same-module `def`,
/// with no separate call-answering path to maintain. A later plain
/// `def f(...):`/another lambda assign of the SAME name in this body
/// overwrites the synthetic entry the same way `function_table`'s own
/// scan already lets a later `def` win (`Stmt::FunctionDef`/lambda-assign
/// entries share the one `local_defs` list, in source order, and
/// `function_table` keeps whichever inserts last).
pub(super) fn local_function_table(body: &[Stmt]) -> FunctionTable {
    // collect() infers ModModule's own body container type from the
    // struct field (the same construction cross_module::synthetic_module
    // uses), so no container crate is named here
    let local_defs = body
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::FunctionDef(def) => Some(Stmt::FunctionDef(def.clone())),
            Stmt::Assign(assign) => lambda_as_synthetic_def(assign).map(Stmt::FunctionDef),
            _ => None,
        })
        .collect();
    let synthetic = ModModule {
        node_index: AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: local_defs,
    };
    function_table(&synthetic)
}

/// `name = lambda <params>: <expr>` — a single bare-Name target whose
/// value is a `Lambda` — read as a synthetic `def name(<params>):
/// return <expr>`, so `summaries::call_result` (which only interprets a
/// real `StmtFunctionDef`) can answer a later `name(...)` call through
/// the lambda's own body. `None` for a multi-target assign, a non-Name
/// target, or a non-Lambda value — this law only ever recognizes the
/// exact `f = lambda ...: ...` shape.
pub(super) fn lambda_as_synthetic_def(assign: &StmtAssign) -> Option<StmtFunctionDef> {
    let [Expr::Name(target_name)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Lambda(lambda) = assign.value.as_ref() else {
        return None;
    };
    let parameters = lambda
        .parameters
        .as_deref()
        .cloned()
        .unwrap_or_else(Parameters::default);
    let return_stmt = Stmt::Return(StmtReturn {
        node_index: AtomicNodeIndex::NONE,
        range: lambda.body.range(),
        value: Some(lambda.body.clone()),
    });
    Some(StmtFunctionDef {
        node_index: AtomicNodeIndex::NONE,
        range: assign.range(),
        is_async: false,
        decorator_list: Default::default(),
        name: ruff_python_ast::Identifier::new(target_name.id.as_str(), target_name.range()),
        type_params: None,
        parameters: Box::new(parameters),
        returns: None,
        body: [return_stmt].into_iter().collect(),
    })
}
