/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Kernel artifact resolution, shared by the LSP seam and the check
//! CLI: `REFINEDPY_KERNEL_DYLIB` wins; otherwise the current
//! directory's ancestors are searched for the in-repo artifact.

use std::path::Path;
use std::path::PathBuf;

/// The kernel artifact's path inside the TypeRefinery repository.
const KERNEL_DYLIB_RELATIVE: &str =
    "packages/refinedts/refined-ts-lean/native/build/librefinedts_kernel.dylib";

/// Resolve the kernel dylib path, or `None` when no artifact exists —
/// in which case every check declines.
pub fn resolve_kernel_dylib() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("REFINEDPY_KERNEL_DYLIB") {
        let path = PathBuf::from(env_path);
        return path.exists().then_some(path);
    }
    let start = std::env::current_dir().ok()?;
    find_in_ancestors(&start, Path::new(KERNEL_DYLIB_RELATIVE))
}

fn find_in_ancestors(start: &Path, relative: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|ancestor| {
        let candidate = ancestor.join(relative);
        candidate.exists().then_some(candidate)
    })
}
