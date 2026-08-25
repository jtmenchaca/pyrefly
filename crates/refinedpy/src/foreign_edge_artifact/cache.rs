//! The target's project-cache entry: where an exported fact lands on
//! disk, and the project-root resolution both the cache path and the
//! producer resolver (`producer.rs`) share.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;

/// What the producer appends to the target's path under the project
/// cache: `audio-level.ts` caches as
/// `.refined/cache/<relpath>/audio-level.ts.refined.json`.
pub(super) const FOREIGN_ARTIFACT_SUFFIX: &str = ".refined.json";

/// The project root stated outright (typically `refinedpy-check`'s
/// `--project-root` flag, set by a caller — the `refined` front door —
/// that already resolved it), bypassing the `.git`-walk in
/// `project_root_of` for both the cache path and producer resolution.
/// `None` (the default) leaves the walk in place.
fn project_root_override() -> &'static Mutex<Option<PathBuf>> {
    static OVERRIDE: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();
    OVERRIDE.get_or_init(|| Mutex::new(None))
}

/// States the project root outright. See `project_root_override`.
pub fn set_project_root_override(root: Option<PathBuf>) {
    *project_root_override().lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = root;
}

/// The nearest ancestor of `target`'s directory holding `.git` — the
/// project root (the target's own directory when none is found) —
/// unless `set_project_root_override` named the root outright.
pub fn project_root_of(target: &Path) -> PathBuf {
    if let Some(overridden) = project_root_override().lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone() {
        return overridden;
    }
    let absolute = std::path::absolute(target).unwrap_or_else(|_| target.to_path_buf());
    let start = absolute.parent().unwrap_or(Path::new("."));
    let mut root = start.to_path_buf();
    let mut walk = Some(start);
    while let Some(dir) = walk {
        if dir.join(".git").exists() {
            root = dir.to_path_buf();
            break;
        }
        walk = dir.parent();
    }
    root
}

/// The target's project-cache entry: the nearest ancestor holding
/// `.git` is the project root (the target's own directory when none is
/// found), and the entry mirrors the target's path relative to that
/// root — the SAME derivation the Go consumer and the Python producer's
/// own CLI both compute, so every checker meets at one file without
/// either being told where. Promoted here from
/// `bin/refinedpy_check.rs`'s own `cache_artifact_path` so the CLI and
/// this reader share one implementation.
pub fn cache_artifact_path(target: &str) -> PathBuf {
    let absolute = std::path::absolute(target).unwrap_or_else(|_| PathBuf::from(target));
    let root = project_root_of(Path::new(target));
    let relative = absolute.strip_prefix(&root).unwrap_or_else(|_| {
        Path::new(absolute.file_name().map(|name| name.as_ref()).unwrap_or(Path::new("artifact").as_os_str()))
    });
    let mut entry = root.join(".refined").join("cache").join(relative);
    let file_name = entry
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact".to_owned());
    entry.set_file_name(format!("{file_name}{FOREIGN_ARTIFACT_SUFFIX}"));
    entry
}
