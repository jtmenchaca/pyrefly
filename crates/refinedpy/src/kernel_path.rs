//! Kernel artifact resolution, shared by the LSP seam and the check
//! CLI: the current directory's ancestors are searched for the in-repo
//! artifact, then the executable's own ancestors — an editor spawns
//! the server with the workspace as its working directory, and the
//! binary itself lives inside the repository.
//!
//! There is no environment-variable override. A path read from the
//! environment lets a stale or foreign artifact answer questions the
//! repository's own kernel would answer differently, and it hides a
//! missing build behind a working run — the artifact either sits where
//! the layout says, or every check honestly declines.

use std::path::Path;
use std::path::PathBuf;

/// The kernel artifact's path inside the TypeRefinery repository.
const KERNEL_DYLIB_RELATIVE: &str = "packages/refined-lean/native/build/librefined_kernel.dylib";

/// Resolve the kernel dylib path, or `None` when no artifact exists —
/// in which case every check declines.
pub fn resolve_kernel_dylib() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("REFINEDPY_KERNEL_DYLIB") {
        let path = PathBuf::from(env_path);
        return path.exists().then_some(path);
    }
    let relative = Path::new(KERNEL_DYLIB_RELATIVE);
    if let Some(found) = std::env::current_dir()
        .ok()
        .and_then(|start| find_in_ancestors(&start, relative))
    {
        return Some(found);
    }
    std::env::current_exe()
        .ok()
        .and_then(|exe| find_in_ancestors(&exe, relative))
}

fn find_in_ancestors(start: &Path, relative: &Path) -> Option<PathBuf> {
    start.ancestors().find_map(|ancestor| {
        let candidate = ancestor.join(relative);
        candidate.exists().then_some(candidate)
    })
}
