use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::unknown;
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
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_parser::parse_module;

use crate::env::Environment;
use crate::function_table::ENTRY_MODULE;
use crate::surface::compile_aliases;
use crate::surface::surface_imports;

use super::effects::record_write_effect;
use super::*;

mod call_result_basics;
mod enclosing_and_globals;
mod return_sort_fallback;
mod declared_return_seed;
mod call_effects;
mod interpret_class_def;
mod summary_route;

pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

/// Parses `source` as a module and returns its single top-level
/// `def` (the function under test).
pub(super) fn parsed_def(source: &str) -> StmtFunctionDef {
    let module = parse_module(source).expect("fixture source parses").into_syntax();
    let stmt = module.body.into_iter().next().expect("one top-level statement");
    stmt.function_def_stmt().expect("top-level statement is a def")
}

pub(super) fn known_int(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustProved)
}

/// A bounded Integer window `[lo, hi]`, `Kind::Set` — the shape a
/// narrowed parameter carries at a call site (`fact_inside`'s own `if
/// 0 <= n <= 5:` guard), never a single concrete value.
pub(super) fn known_integer_window(lo: f64, hi: f64) -> AbstractValue {
    use refined_sets::refinement_forms::at_most;
    known_set(make_refined_set(vec![at_least(lo), at_most(hi), integer()]), None, TrustProved, SetKindTag::None)
}

/// A whole module's own compiled alias table, mirroring exactly what
/// `check.rs::walk_body_with_self_binding` threads onto an
/// `Environment` (`Environment::set_declared_aliases`) — built here
/// so `declared_return_seed`'s alias-aware reading can be exercised
/// directly, without going through the full `check.rs` walk.
pub(super) fn environment_with_module_aliases(source: &str) -> Environment {
    let module = parse_module(source).expect("fixture source parses").into_syntax();
    let aliases = compile_aliases(&module);
    let imports = surface_imports(&module);
    let mut environment = Environment::new(std::collections::HashSet::new());
    environment.set_declared_aliases(Arc::new(aliases), Arc::new(imports));
    environment
}

/// A Python `str` as this domain spells it — one code point per f64,
/// the representation `string_models.rs` documents. Built here rather
/// than reached for, matching `loops.rs`'s own same-crate precedent.
pub(super) fn known_string_value(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    known_values(code_points, PrimitiveKind::String, TrustProved)
}
