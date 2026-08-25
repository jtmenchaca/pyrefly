use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::kernel_seam::ask_bounds_public;
use refined_domain::known_constructors::known_list;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer as integer_form;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_parser::parse_module;
use crate::collection_models;
use crate::env::Environment;
use crate::summaries::iterable_element_sort;
use crate::typereading::DeclaredRefinement;

use super::*;
use super::iterable::known_number_sorted;
use super::loop_final_environment;

mod body_once;
mod widen;
mod for_loop;
mod while_loop;
mod iterable;
mod bind_target;

/// Test-only convenience: a Number-sorted (unsplit-int/float) known
/// value — `known_number_sorted`'s own doc explains why production
/// code now always states the true CPython sort instead (`for age
/// in [10, 20, 30]` binds Integer, not this joined `Number` tag).
pub(super) fn known_number(value: f64) -> AbstractValue {
    known_number_sorted(value, PrimitiveKind::Number)
}

pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    let kernel = load_kernel(&path).expect("load_kernel");
    // The binary installs the domain crate's kernel seams at startup;
    // tests exercising join/stabilization behavior need the same seams
    // seated, or every seam ask answers None and the join falls back.
    crate::kernel_ask::install_kernel_seams(&kernel);
    Some(kernel)
}

/// Parses `source` as a module body and returns its single
/// top-level statement (the loop under test).
pub(super) fn parsed_loop(source: &str) -> Stmt {
    let module = parse_module(source).expect("fixture source parses").into_syntax();
    module.body.into_iter().next().expect("one top-level statement")
}

/// Parses `source` as a module body and returns its single top-level
/// `def` — `iterable_element_sort`'s own test fixture shape, which
/// needs a `&StmtFunctionDef` directly rather than a loop statement.
pub(super) fn parsed_def(source: &str) -> StmtFunctionDef {
    let module = parse_module(source).expect("fixture source parses").into_syntax();
    let stmt = module.body.into_iter().next().expect("one top-level statement");
    stmt.function_def_stmt().expect("top-level statement is a def")
}

pub(super) fn environment_with(bindings: &[(&str, f64)]) -> Environment {
    let locally_bound: HashSet<String> = bindings.iter().map(|(name, _)| name.to_string()).collect();
    let mut environment = Environment::new(locally_bound);
    for (name, value) in bindings {
        environment.bind(name, known_number(*value));
    }
    environment
}

pub(super) fn integer(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustProved)
}

pub(super) fn no_declared() -> HashMap<String, DeclaredRefinement> {
    HashMap::new()
}

/// `type Age = Annotated[int, Field(ge=0, le=120)]` — the one
/// declared refinement this module's judged-write tests need,
/// built directly (this module's tests construct environments and
/// declared tables by hand rather than walking a function
/// signature — matching `check.rs`'s own `age_refinement` test
/// fixture in spirit).
pub(super) fn age_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![integer_form(), at_least(0.0), at_most(120.0)]),
        spelling: "Age".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

pub(super) fn declared_age(name: &str) -> HashMap<String, DeclaredRefinement> {
    let mut declared = HashMap::new();
    declared.insert(name.to_owned(), age_refinement());
    declared
}

/// Runs `loop_final_environment` with no declared table and
/// discards its judged-fires/else_runs/returned — the shape every
/// UNIT 1/2 test above cares about is just the post-loop environment.
pub(super) fn run(stmt: &Stmt, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<Environment> {
    let declared = no_declared();
    let mut out = Vec::new();
    loop_final_environment(stmt, environment, kernel, &declared, &mut out).map(|answer| answer.environment)
}

/// Parses `source` as a module with MULTIPLE top-level statements
/// (a generator `def` plus the loop under test) and returns the
/// LAST statement (the loop) alongside the module's own function
/// table — the generator-call tests need `environment.functions()`
/// to resolve the callee, which `parsed_loop`'s single-statement
/// module cannot carry.
pub(super) fn parsed_loop_with_functions(source: &str) -> (Stmt, Arc<crate::function_table::FunctionTable>) {
    let module = parse_module(source).expect("fixture source parses").into_syntax();
    let table = Arc::new(crate::function_table::function_table(&module));
    let loop_stmt = module.body.into_iter().last().expect("at least one top-level statement");
    (loop_stmt, table)
}

/// An `Age`-shaped declared set (`[0, 120]`, integers) — the same
/// shape `seed_parameters` (check.rs) binds a scalar-typed parameter
/// to, built directly here since this module's tests construct
/// environments by hand rather than walking a function signature.
pub(super) fn age_set() -> refined_sets::refinement_forms::RefinedSet {
    refined_sets::refinement_forms::make_refined_set(vec![
        refined_sets::refinement_forms::at_least(0.0),
        refined_sets::refinement_forms::at_most(120.0),
        refined_sets::refinement_forms::integer(),
    ])
}

/// A `list[Wide]`-shaped parameter — the repetition-window seed
/// `check.rs::seed_parameters` builds for a bare `list[X]` annotation
/// (`AbstractValue { kind_tag: Some(sort), ..known_set(repeat_of(...))
/// }`, that function's own doc) — the SAME shape this test builds by
/// hand so `repetition_window_element_pass` sees exactly what a real
/// `xs: list[Wide]` parameter would.
pub(super) fn wide_list_parameter() -> AbstractValue {
    let element = make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]);
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element, 0, None)]), None, TrustProved, SetKindTag::None)
    }
}

/// A known two-entry dict, `{"a": 10, "b": 20}` — the fixture every
/// iterator-invalidation test below iterates over.
pub(super) fn two_entry_dict() -> AbstractValue {
    known_object(
        vec![
            ObjectKey { name: "a".to_owned(), numeric: false, value: integer(10.0) },
            ObjectKey { name: "b".to_owned(), numeric: false, value: integer(20.0) },
        ],
        None,
        true,
        TrustProved,
        false,
    )
}
