use std::collections::HashSet;
use std::sync::Arc;

use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::StmtFunctionDef;
use serde_json::Value;
use serde_json::json;

use crate::cross_module::ModuleResolver;

use super::*;

mod sha256;
mod harness_shapes;
mod surface_export;
mod stdout_purity;
mod return_cases;
mod artifact_structure;

/// The dylib-absence convention every kernel-touching test in this
/// crate follows (`lattice_conformance.rs`'s own helper): a missing
/// artifact prints to stderr and the caller returns early, never
/// failing the run.
pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}
