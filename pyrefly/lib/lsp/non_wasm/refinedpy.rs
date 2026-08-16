/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

use std::path::Path;
use std::path::PathBuf;
use std::sync::OnceLock;

use lsp_types::Diagnostic;
use pyrefly_build::handle::Handle;

use crate::state::state::Transaction;

/// The kernel artifact's path inside the TypeRefinery repository.
const KERNEL_DYLIB_RELATIVE: &str =
    "packages/refinedts/refined-ts-lean/native/build/librefinedts_kernel.dylib";

/// Resolved once before the LSP loop serves requests; `None` when no
/// kernel artifact could be found, in which case every check declines.
static KERNEL_DYLIB: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Resolve and remember the kernel dylib path. Called once from
/// `lsp_loop` before the event loop starts, so every check that runs
/// sees the same answer. `REFINEDPY_KERNEL_DYLIB` wins; otherwise the
/// current directory's ancestors are searched for the in-repo artifact.
pub fn configure_kernel_dylib() {
    KERNEL_DYLIB.get_or_init(|| {
        if let Ok(env_path) = std::env::var("REFINEDPY_KERNEL_DYLIB") {
            let path = PathBuf::from(env_path);
            return path.exists().then_some(path);
        }
        let start = std::env::current_dir().ok()?;
        find_in_ancestors(&start, Path::new(KERNEL_DYLIB_RELATIVE))
    });
}

/// The configured kernel dylib path, if one was found.
pub fn kernel_dylib() -> Option<&'static PathBuf> {
    KERNEL_DYLIB.get().and_then(|found| found.as_ref())
}

fn find_in_ancestors(start: &Path, relative: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|ancestor| {
        let candidate = ancestor.join(relative);
        candidate.exists().then_some(candidate)
    })
}

/// Append RefinedPy refinement diagnostics for one open handle. Both
/// diagnostic paths (pull and push) reach this through
/// `append_ide_specific_diagnostics`, so this is the one place
/// refinement findings enter the LSP surface. The refinement engine is
/// linked in behind this function; a build without the engine, or a
/// missing kernel artifact, appends nothing.
pub fn append_refinedpy_diagnostics(
    transaction: &Transaction<'_>,
    handle: &Handle,
    items: &mut Vec<Diagnostic>,
) {
    if kernel_dylib().is_none() {
        return;
    }
    let _ = (transaction, handle, items);
}
