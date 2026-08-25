//! The fact a foreign TypeScript target exports about itself, read off
//! disk. The mirror of the Go consumer's own reader
//! (`refined-ts-go/internal/refinedts/walk/foreign_edge_artifact.go`):
//! a cross-language call edge (`foreign_edge.rs`) claims something about
//! the code that runs on the other side, and that claim is only as good
//! as the two premises this file checks — the artifact SAYS what the
//! target's entry admits and what its return holds, and the artifact is
//! about the FILE THIS CHECK READS (the content hash, the target-
//! integrity premise, not a convenience).
//!
//! One envelope is admitted (docs/one-checker/schema-v2.md):
//!
//!   {"refined": {"kind": "fact-artifact", "version": 2},
//!    "target": {"file", "contentHash": "sha256:<hex>"},
//!    "language": "typescript",
//!    "runtime": {"band": "es2023+"},
//!    "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "<fn>"}
//!      | {"kind": "argv-json", "argIndex": n, "stdout": "json", "calls": "<fn>"}
//!      | {"kind": "file-json", "argIndex": n, "stdout": "json", "calls": "<fn>"},
//!    "functions": {"<name>": {
//!      "entry": [{"name", "sequence": {"element": <set>, "lengthAtLeast": n}}
//!               |{"name", "set": <set>}],
//!      "return": {"set": <set>, "stdoutPure": bool},
//!      "provenance": {"line": n, "said": "..."}}}}
//!
//! `surface.kind` names which carrier the JSON transport model rides
//! on — a pipe (`stdin-json`), one argv element read directly
//! (`argv-json`), or one argv element naming a FILE the target reads
//! (`file-json`) — `argIndex` naming which argv position, either way
//! (the node convention makes the third argv element `process.argv[2]`).
//! All three apply the identical transport model to the payload (JSON
//! text, the same round-trip premise); only the carrier differs, so
//! `argv-json`/`file-json` carry no `stdin` field at all, and
//! `file-json`'s own `argIndex` names the argv position holding the
//! PATH, not the JSON text itself.
//!
//! Any other (kind, version, language) triple declines, naming the
//! triple it saw and the one form this reader accepts.
//!
//! Every <set> is the kernel's own forms JSON, decoded by
//! `refined_kernel::wire_decode::decode_wire_set` — the SAME decoder
//! every kernel answer goes through, so a set that crossed the edge and
//! a set the kernel answered are the same value. `decode_wire_set`
//! PANICS on an unknown form (its own stated contract: a mistyped wire
//! is a violation, not a value to degrade on), so every decode here runs
//! under `catch_unwind` — an artifact is a FILE, written by another
//! program, and a checker must not die on a malformed one. Every other
//! caller in this crate trusts the kernel and does not wrap; this file
//! must.
//!
//! Nothing here reaches the kernel or the walk. It reads bytes, hashes
//! bytes, and answers a fact or a sentence saying why not.

mod cache;
mod cases;
mod compiled_binary;
mod producer;
mod types;
mod typescript_read;

#[cfg(test)]
mod tests;

pub use cache::cache_artifact_path;
pub use cache::project_root_of;
pub use cache::set_project_root_override;
pub use compiled_binary::compiled_binary_fact_path;
pub use compiled_binary::read_compiled_binary_fact;
pub use types::ForeignCase;
pub use types::ForeignSurface;
pub use types::ForeignTsArtifact;
pub use types::ForeignTsEntry;
pub use types::ForeignTsFunctionFact;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::SystemTime;

use cache::cache_artifact_path as cache_artifact_path_of;
use producer::export_foreign_ts_artifact;
use producer::EXPORT_CHAIN_ENV_VAR;
use typescript_read::read_and_verify_foreign_ts_artifact;

/// One read's whole outcome, memoized: the fact or the sentence that
/// stopped it, plus the artifact file's mtime at fill time so a later
/// read can tell whether the cache changed underneath it
/// (docs/one-checker/fact-freshness.md's mtime-freshness stopgap).
struct ForeignArtifactRow {
    artifact: Option<ForeignTsArtifact>,
    sentence: String,
    mtime: Option<SystemTime>,
}

/// Memoizes `read_foreign_ts_artifact` by target path. The walk reaches
/// one statement many times — the speculative recovery passes, the
/// correlation splits, every inlining of the enclosing function — and
/// each reach would otherwise re-read and re-hash the target. A live LSP
/// can rewrite the cache mid-session (a save-time export), so the row
/// also carries the mtime it was filled at: a read that finds a
/// different mtime treats the memo as stale and re-reads, rather than
/// holding the row for the whole process the way a target this checker
/// never writes could safely be held.
fn foreign_artifacts() -> &'static Mutex<HashMap<String, ForeignArtifactRow>> {
    static ROWS: OnceLock<Mutex<HashMap<String, ForeignArtifactRow>>> = OnceLock::new();
    ROWS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Resolves the target's project-cache entry, filling it through the
/// resolved producer when it is missing or stale, checks every premise
/// this file owns, and answers the harness-called function's fact — or
/// (`None`, one sentence) saying which premise broke.
///
/// The premises discharged HERE, each a real check and none assumed:
///
/// - the artifact EXISTS in the cache (a miss auto-exports when a
///   producer resolves; otherwise the sentence names the file and the
///   command that writes it);
/// - the envelope is this kind and this version;
/// - TARGET INTEGRITY: sha256 of the `.ts` file's ACTUAL BYTES equals
///   the artifact's stated contentHash. A mismatch means the claim is
///   about code that is not the code being checked;
/// - RUNTIME IDENTITY: the stated band is the one the pins commit to;
/// - the harness reads json on stdin and writes json on stdout, and
///   names a function the artifact actually carries a fact for.
///
/// CHANNEL PURITY is NOT checked here: see `ForeignTsFunctionFact::stdout_pure`'s
/// doc comment.
pub fn read_foreign_ts_artifact(target_path: &str) -> Result<ForeignTsArtifact, String> {
    let artifact_path = cache_artifact_path_of(target_path);
    let current_mtime = std::fs::metadata(&artifact_path).and_then(|meta| meta.modified()).ok();

    {
        let rows = foreign_artifacts().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(row) = rows.get(target_path) {
            if row.mtime == current_mtime {
                return row.artifact.clone().ok_or_else(|| row.sentence.clone());
            }
        }
    }

    let (artifact, sentence) = read_foreign_ts_artifact_uncached(target_path, &artifact_path);
    let filled_mtime = std::fs::metadata(&artifact_path).and_then(|meta| meta.modified()).ok();
    let mut rows = foreign_artifacts().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    rows.insert(
        target_path.to_owned(),
        ForeignArtifactRow {
            artifact: artifact.clone(),
            sentence: sentence.clone(),
            mtime: filled_mtime,
        },
    );
    artifact.ok_or(sentence)
}

/// Fills the cache when it can and reads it — every premise checked, no
/// memo consulted. A missing or failed artifact triggers ONE export
/// attempt through the resolved producer; when no producer resolves,
/// the sentence names the file and the command, exactly as before.
/// `REFINED_EXPORT_CHAIN` is read ONCE here, at the point the spawn
/// decision is made — never inside `export_foreign_ts_artifact` itself,
/// which takes the chain as a plain parameter so it is testable without
/// mutating process environment.
fn read_foreign_ts_artifact_uncached(
    target_path: &str,
    artifact_path: &Path,
) -> (Option<ForeignTsArtifact>, String) {
    let (artifact, sentence) = read_and_verify_foreign_ts_artifact(target_path, artifact_path);
    if sentence.is_empty() {
        return (artifact, String::new());
    }
    let export_chain = std::env::var(EXPORT_CHAIN_ENV_VAR).unwrap_or_default();
    if let Err(export_sentence) = export_foreign_ts_artifact(target_path, artifact_path, &export_chain) {
        return (None, format!("{sentence} (auto-export declined: {export_sentence})"));
    }
    read_and_verify_foreign_ts_artifact(target_path, artifact_path)
}
