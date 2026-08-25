//! Resolving the producer binary that regenerates a missing or stale
//! artifact, and the auto-export spawn — including the cross-process
//! export-chain cycle guard that stops a producer from recursing back
//! through a hop already in flight.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use super::cache::project_root_of;

/// The command that writes a missing artifact — carried into the
/// diagnostic sentence, so a missing fact reads as a work-queue item
/// rather than a silent nothing.
pub(super) const FOREIGN_EXPORT_COMMAND: &str = "refinedts-check-bin -export-fact";

/// The producer binary's name, searched for under the project root and
/// then on `PATH` — never read from an environment variable (env vars
/// break command-approval permissions; ruling 2026-08-20).
pub(super) const FOREIGN_PRODUCER_NAME: &str = "refinedts-check-bin";

/// Where the producer lives relative to the project root, when it was
/// built in place rather than installed onto `PATH`.
pub(super) const FOREIGN_PRODUCER_RELATIVE: &str = "packages/refinedts/refined-ts-go/refinedts-check-bin";

/// Resolves the producer binary: the CONVENTION build path under the
/// target's own project root first
/// (`<root>/packages/refinedts/refined-ts-go/refinedts-check-bin`), then
/// a `PATH` search for `refinedts-check-bin` — mirroring
/// `kernel_path.rs`'s ancestor-then-search shape, minus its
/// environment-variable arm (env vars break command-approval
/// permissions; ruling 2026-08-20 — there is no override here, ever).
pub(super) fn resolve_foreign_producer(target_path: &str) -> Option<PathBuf> {
    let root = project_root_of(Path::new(target_path));
    let convention_path = root.join(FOREIGN_PRODUCER_RELATIVE);
    if convention_path.exists() {
        return Some(convention_path);
    }
    search_path_for(FOREIGN_PRODUCER_NAME)
}

/// Searches the `PATH` environment variable's own directories for
/// `name`, answering the first one that exists. Std-only: no `which`
/// crate is available (this crate's `Cargo.toml` is autocargo-generated
/// from the Meta build definition, so a hand-added dependency does not
/// survive a regeneration).
fn search_path_for(name: &str) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    std::env::split_paths(&path_var).map(|dir| dir.join(name)).find(|candidate| candidate.exists())
}

/// The environment variable carrying the cross-process auto-export
/// chain: a colon-separated list of absolute target paths, one per
/// export hop already in flight. This is internal state no invocation
/// reads on purpose — it governs WHETHER an auto-export spawns, never
/// WHICH binary runs, and is therefore a wholly separate concern from
/// the no-env-producer-resolution ruling at `FOREIGN_PRODUCER_NAME`'s
/// own doc (2026-08-20: the producer's OWN identity is never read from
/// an environment variable, because that would break command-approval
/// permissions). A Python checker auto-exporting a TypeScript target
/// whose own auto-export recurses back to a Python target already on
/// this chain would otherwise spawn forever, each hop a fresh process
/// neither side's own in-memory recursion guard can see across.
pub(super) const EXPORT_CHAIN_ENV_VAR: &str = "REFINED_EXPORT_CHAIN";

/// Whether `target_path`'s absolute form already appears as a hop in
/// `chain` (the colon-separated `REFINED_EXPORT_CHAIN` value read at
/// this process's own entry point) — `true` means spawning the
/// producer for this target would recurse back through a hop already
/// in flight, and the caller must decline rather than spawn.
pub(super) fn export_chain_contains(chain: &str, target_path: &str) -> bool {
    let absolute_target = std::path::absolute(target_path).unwrap_or_else(|_| PathBuf::from(target_path));
    chain.split(':').any(|hop| !hop.is_empty() && Path::new(hop) == absolute_target)
}

/// The sentence a chain-marked decline states: names the recursing
/// target and the whole chain that led back to it, so a reader sees
/// the cycle rather than a generic refusal.
pub(super) fn export_chain_cycle_sentence(chain: &str, target_path: &str) -> String {
    let absolute_target = std::path::absolute(target_path).unwrap_or_else(|_| PathBuf::from(target_path));
    let mut hops: Vec<&str> = chain.split(':').filter(|hop| !hop.is_empty()).collect();
    hops.push(absolute_target.to_str().unwrap_or(target_path));
    format!(
        "the export of {} recurses back through a target already in flight — the auto-export chain is {}",
        absolute_target.display(),
        hops.join(" \u{2192} ")
    )
}

/// Runs the resolved producer into the cache entry, answering `Ok(())`
/// on success and `Err` naming what stopped it. `export_chain` is this
/// process's own `REFINED_EXPORT_CHAIN` value (read once, at the
/// `-export-fact` entry point, and threaded down here as a plain
/// parameter — never re-read from the environment inside this
/// function) — when `target_path` already appears on it, this declines
/// with the cycle sentence rather than spawning; otherwise the CHILD's
/// own environment carries the chain plus `target_path` appended, so a
/// nested auto-export the child triggers sees the extended chain in
/// turn.
pub(super) fn export_foreign_ts_artifact(target_path: &str, artifact_path: &Path, export_chain: &str) -> Result<(), String> {
    if export_chain_contains(export_chain, target_path) {
        return Err(export_chain_cycle_sentence(export_chain, target_path));
    }
    let Some(producer) = resolve_foreign_producer(target_path) else {
        return Err(format!(
            "no {FOREIGN_PRODUCER_NAME} under the project root and none on PATH"
        ));
    };
    if let Some(parent) = artifact_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| format!("the cache directory could not be created: {err}"))?;
    }
    let absolute_target = std::path::absolute(target_path).unwrap_or_else(|_| PathBuf::from(target_path));
    let absolute_target_words = absolute_target.to_string_lossy().into_owned();
    let child_chain = if export_chain.is_empty() {
        absolute_target_words.clone()
    } else {
        format!("{export_chain}:{absolute_target_words}")
    };
    let output = Command::new(&producer)
        .arg("-export-fact")
        .arg(target_path)
        .arg("-o")
        .arg(artifact_path)
        .env(EXPORT_CHAIN_ENV_VAR, child_chain)
        .output()
        .map_err(|err| format!("the export run failed: {err}"))?;
    if !output.status.success() {
        let message = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(if message.is_empty() {
            format!("the export run failed with {}", output.status)
        } else {
            format!("the export run failed: {message}")
        });
    }
    Ok(())
}
