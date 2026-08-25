use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_parser::parse_expression;

use crate::env::Environment;

use super::isinstance_guards::unbounded_integers;
use super::*;

mod values_channel;
mod type_guard;
mod set_channel;
mod access_path;
mod membership;
mod isinstance_seeding;
mod none_wrapper;
mod ascii_alphabet;

/// A kernel handle for tests that never ask it anything — `assume`
/// takes the parameter for the frozen signature's sake, but no
/// construct this wave asks a question of it. `None` when the
/// native dylib artifact has not been built, so this file's tests
/// run without requiring `pnpm kernel:native` first.
pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

pub(super) fn environment_with(name: &str, values: Vec<f64>, kind_tag: PrimitiveKind) -> Environment {
    let mut locally_bound = HashSet::new();
    locally_bound.insert(name.to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind(name, known_values(values, kind_tag, TrustProved));
    environment
}

pub(super) fn assumed(source: &str, environment: Environment, truth: bool) -> Option<Environment> {
    let kernel = loaded_kernel()?;
    let parsed = parse_expression(source).expect("test source must parse");
    let expression = parsed.into_expr();
    Some(assume(&expression, environment, &kernel, truth))
}

/// An environment carrying a same-module function table with one
/// `def`, parsed from `source` — the shape `recognizes_type_guard_
/// call` reads via `environment.functions()`.
pub(super) fn environment_with_function_table(source: &str) -> Environment {
    let module = ruff_python_parser::parse_module(source).expect("test source must parse").into_syntax();
    let table = crate::function_table::function_table(&module);
    let mut environment = Environment::new(HashSet::new());
    environment.set_functions(Arc::new(table));
    environment
}

pub(super) fn environment_with_set(name: &str, set: RefinedSet, kind_tag: PrimitiveKind) -> Environment {
    let mut locally_bound = HashSet::new();
    locally_bound.insert(name.to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind(name, AbstractValue { kind_tag: Some(kind_tag), ..known_set(set, None, TrustProved, SetKindTag::None) });
    environment
}

/// A bare `str` parameter's own seed: `Kind::Set` over the whole
/// string ground (`codepoint_sets::strings()`), untagged
/// (`kind_tag: None` — `check.rs::seed_parameters`'s own choice for
/// a sequence-shaped declared set, `states_sequence` true), matching
/// what `x: str` actually seeds to.
pub(super) fn environment_with_bare_string(name: &str) -> Environment {
    let mut locally_bound = HashSet::new();
    locally_bound.insert(name.to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind(name, known_set(strings(), None, TrustProved, SetKindTag::None));
    environment
}
