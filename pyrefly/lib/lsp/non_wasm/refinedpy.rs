/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! RefinedPy: refinement diagnostics layered onto pyrefly's own.
//!
//! RefinedPy judges values against refinement sets (which values are
//! allowed, not just what shape they have) by asking a proved Lean
//! kernel loaded from a native dylib. Its diagnostics carry
//! `source: "refinedpy"` and codes RTS7001-RTS7005, and are appended
//! after pyrefly's own diagnostics on the same read-only transaction
//! the host already validated with `Require::Everything` — the check
//! never calls `set_memory` or `run` itself.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::OnceLock;

use lsp_types::Diagnostic;
use lsp_types::DiagnosticSeverity;
use lsp_types::NumberOrString;
use pyrefly_build::handle::Handle;
use refined_kernel::kernel_bridge::kernel_if_loaded;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;

use crate::refinedpy::check::findings_for_module;
use crate::state::state::Transaction;

/// Resolved once before the LSP loop serves requests; `None` when no
/// kernel artifact could be found, in which case every check declines.
static KERNEL_DYLIB: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Resolve and remember the kernel dylib path. Called once from
/// `lsp_loop` before the event loop starts, so every check that runs
/// sees the same answer.
pub fn configure_kernel_dylib() {
    KERNEL_DYLIB.get_or_init(crate::refinedpy::kernel_path::resolve_kernel_dylib);
}

/// The configured kernel dylib path, if one was found.
pub fn kernel_dylib() -> Option<&'static PathBuf> {
    KERNEL_DYLIB.get().and_then(|found| found.as_ref())
}

/// The one loaded kernel this server asks. `load_kernel` adopts a
/// process-wide singleton, so retries after a successful load are
/// cache hits; a missing artifact means every check declines.
fn kernel() -> Option<Arc<RefinedTSKernel>> {
    if let Some(loaded) = kernel_if_loaded() {
        return Some(loaded);
    }
    load_kernel(kernel_dylib()?).ok()
}

/// Append RefinedPy refinement diagnostics for one open handle. Both
/// diagnostic paths (pull and push) reach this through
/// `append_ide_specific_diagnostics`, so this is the one place
/// refinement findings enter the LSP surface. A missing kernel
/// artifact or a non-`.py` handle appends nothing.
pub fn append_refinedpy_diagnostics(
    transaction: &Transaction<'_>,
    handle: &Handle,
    items: &mut Vec<Diagnostic>,
) {
    if !handle
        .path()
        .as_path()
        .extension()
        .is_some_and(|ext| ext == "py")
    {
        return;
    }
    let Some(kernel) = kernel() else {
        return;
    };
    let Some(ast) = transaction.get_ast(handle) else {
        return;
    };
    let Some(module_info) = transaction.get_module_info(handle) else {
        return;
    };
    for finding in findings_for_module(&ast, &kernel) {
        items.push(Diagnostic {
            range: module_info.to_lsp_range(finding.range),
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String(finding.code.to_owned())),
            code_description: None,
            source: Some("refinedpy".to_owned()),
            message: finding.message.into(),
            related_information: None,
            tags: None,
            data: None,
        });
    }
}
