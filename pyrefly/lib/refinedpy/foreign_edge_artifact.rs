/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::SystemTime;

use refined_kernel::wire_decode::decode_wire_set;
use refined_sets::refinement_forms::RefinedSet;
use serde_json::Value;

use crate::refinedpy::fact_export::sha256_hex;

/// What the producer appends to the target's path under the project
/// cache: `audio-level.ts` caches as
/// `.refined/cache/<relpath>/audio-level.ts.refined.json`.
const FOREIGN_ARTIFACT_SUFFIX: &str = ".refined.json";

/// The one envelope this consumer admits (the RULED cases schema,
/// JT-approved 2026-08-21). `language` is checked alongside `kind` — the
/// kind is shared across every producer language, so the language field
/// is what routes to the right runtime-band pins. NO version field is
/// ever read or written: the schema carries no version ceremony, so a
/// reader strict-parses the CURRENT shape and any other shape (a
/// version field present, a bare "set", the old sequence spelling) is
/// NO-FACT — the same decline every other unreadable artifact earns,
/// re-exported by the existing self-refresh path rather than read
/// best-effort.
const FOREIGN_ARTIFACT_KIND: &str = "fact-artifact";
const FOREIGN_ARTIFACT_LANGUAGE: &str = "typescript";

/// The runtime band this checker's TypeScript pins commit to.
///
/// One JS-family band claiming ECMA-262-level behaviour (ruling,
/// 2026-08-21): every premise the edge discharges is an ECMA-262 claim,
/// so any recognized JS runner (node, deno, bun, npx tsx) satisfies this
/// band premise once the artifact declares it — the band names the
/// SPEC LEVEL the target's checked code runs against, not one runtime
/// binary.
const FOREIGN_RUNTIME_BAND: &str = "es2023+";

/// The command that writes a missing artifact — carried into the
/// diagnostic sentence, so a missing fact reads as a work-queue item
/// rather than a silent nothing.
const FOREIGN_EXPORT_COMMAND: &str = "refinedts-check-bin -export-fact";

/// The producer binary's name, searched for under the project root and
/// then on `PATH` — never read from an environment variable (env vars
/// break command-approval permissions; ruling 2026-08-20).
const FOREIGN_PRODUCER_NAME: &str = "refinedts-check-bin";

/// Where the producer lives relative to the project root, when it was
/// built in place rather than installed onto `PATH`.
const FOREIGN_PRODUCER_RELATIVE: &str = "packages/refinedts/refined-ts-go/refinedts-check-bin";

/// One admitted case the wire carries — the reader's own twin of the
/// writer's `Case` (`fact_export.rs`): the full kernel wire set grammar
/// verbatim for a number/string case, and no set at all for the two
/// whole-sort floors. `decode_wire_set` is the SAME decoder every other
/// kernel answer goes through, so a set that crossed this edge and a set
/// the kernel answered are the same value.
///
/// `Object` carries the RULED object case's own vocabulary (CROSS-
/// LANGUAGE-EDGE.md §17, JT-prioritized 2026-08-21): each member NAME
/// mapped to ITS OWN cases list (recursed through this same enum, so a
/// nested object case is an ordinary `ForeignCase::Object` sitting inside
/// a member's list) and whether the key set is `closed`. Stored here so
/// the CONSUMER-side lowering (`foreign_edge.rs`'s object-case arm, a
/// stated follow-up — not this lane's) has a typed shape to match on
/// rather than re-parsing the JSON a second time.
#[derive(Debug, Clone, PartialEq)]
pub enum ForeignCase {
    Number(RefinedSet),
    String(RefinedSet),
    Boolean,
    Null,
    Object {
        members: Vec<(String, Vec<ForeignCase>)>,
        closed: bool,
    },
}

/// One parameter position the target states: either a SEQUENCE (an
/// element's own cases plus the length floor the body relies on,
/// carried as `(cases, lengthAtLeast)`) or a plain SCALAR cases list.
#[derive(Debug, Clone)]
pub struct ForeignTsEntry {
    pub name: String,
    /// `Some` for a sequence position — the element's own cases and the
    /// declaration's own length floor.
    pub sequence: Option<(Vec<ForeignCase>, i64)>,
    /// `Some` for a scalar position — the position's own cases list,
    /// never empty when present (a single case still spells as a
    /// one-element list, matching the wire's own convention).
    pub scalar: Option<Vec<ForeignCase>>,
}

/// One target function's whole exported fact.
#[derive(Debug, Clone)]
pub struct ForeignTsFunctionFact {
    pub name: String,
    pub entry: Vec<ForeignTsEntry>,
    /// The return's own cases list — one case lowers directly to a
    /// single value; more than one lowers to a `Kind::KindUnion` of
    /// arms (`foreign_edge.rs::foreign_return_value` does the lowering,
    /// the one place a `ForeignCase` list becomes an `AbstractValue`).
    pub return_cases: Vec<ForeignCase>,
    /// CHANNEL PURITY: the target writes NOTHING to stdout but the
    /// serialized result. Not a premise this file discharges — a
    /// property of the consumed function's return, checked where the
    /// edge consumes it (`foreign_edge.rs`), which is where the sentence
    /// can name the call.
    pub stdout_pure: bool,
    /// Where the target's claim was made: the line in the TypeScript
    /// file, and the sentence its checker said there.
    pub provenance_line: usize,
    pub provenance_said: String,
}

/// Which carrier the JSON transport model rides on — the three `surface
/// .kind` tags this reader admits. All three apply the SAME transport
/// model (the value crosses as JSON text; `stdoutPure` and the
/// outbound-leg fit checks apply identically to each): only the carrier
/// differs — a pipe, one argv element read directly, or one argv
/// element naming a file the target reads its JSON from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignSurface {
    /// `{"kind": "stdin-json", "stdin": "json", "stdout": "json"}` — the
    /// payload rides on the process's stdin pipe.
    StdinJson,
    /// `{"kind": "argv-json", "argIndex": n, "stdout": "json"}` — the
    /// payload is `JSON.parse`'d from `process.argv[argIndex]`; there is
    /// no `stdin` field at all (the carriers are mutually exclusive by
    /// construction, never a joint claim).
    ArgvJson { arg_index: i64 },
    /// `{"kind": "file-json", "argIndex": n, "stdout": "json"}` — the
    /// payload is `JSON.parse`'d from the FILE named at
    /// `process.argv[argIndex]` (node's own harness reads it with
    /// `readFileSync(process.argv[argIndex], "utf8")`), not from the
    /// argv element's own text.
    FileJson { arg_index: i64 },
}

/// The artifact as consumed: the runtime band it commits to, which
/// carrier the JSON transport rides on, and the ONE function the
/// harness calls, already selected.
#[derive(Debug, Clone)]
pub struct ForeignTsArtifact {
    /// The artifact file itself, for the diagnostics.
    pub path: PathBuf,
    /// The `.ts` path the artifact is about, as resolved here (not as
    /// the artifact spells it — the hash is what ties them).
    pub target_file: String,
    pub runtime_band: String,
    /// Which carrier the target's `surface` states — `foreign_edge.rs`
    /// checks a recognized call's own channel against this before
    /// applying the outbound-leg fit checks.
    pub surface: ForeignSurface,
    pub called: ForeignTsFunctionFact,
}

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
    let artifact_path = cache_artifact_path(target_path);
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

/// Resolves the producer binary: the CONVENTION build path under the
/// target's own project root first
/// (`<root>/packages/refinedts/refined-ts-go/refinedts-check-bin`), then
/// a `PATH` search for `refinedts-check-bin` — mirroring
/// `kernel_path.rs`'s ancestor-then-search shape, minus its
/// environment-variable arm (env vars break command-approval
/// permissions; ruling 2026-08-20 — there is no override here, ever).
fn resolve_foreign_producer(target_path: &str) -> Option<PathBuf> {
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
const EXPORT_CHAIN_ENV_VAR: &str = "REFINED_EXPORT_CHAIN";

/// Whether `target_path`'s absolute form already appears as a hop in
/// `chain` (the colon-separated `REFINED_EXPORT_CHAIN` value read at
/// this process's own entry point) — `true` means spawning the
/// producer for this target would recurse back through a hop already
/// in flight, and the caller must decline rather than spawn.
fn export_chain_contains(chain: &str, target_path: &str) -> bool {
    let absolute_target = std::path::absolute(target_path).unwrap_or_else(|_| PathBuf::from(target_path));
    chain.split(':').any(|hop| !hop.is_empty() && Path::new(hop) == absolute_target)
}

/// The sentence a chain-marked decline states: names the recursing
/// target and the whole chain that led back to it, so a reader sees
/// the cycle rather than a generic refusal.
fn export_chain_cycle_sentence(chain: &str, target_path: &str) -> String {
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
fn export_foreign_ts_artifact(target_path: &str, artifact_path: &Path, export_chain: &str) -> Result<(), String> {
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

/// The read itself — every premise checked against the given cache
/// entry.
fn read_and_verify_foreign_ts_artifact(
    target_path: &str,
    artifact_path: &Path,
) -> (Option<ForeignTsArtifact>, String) {
    let artifact_path_words = artifact_path.display().to_string();
    let raw = match std::fs::read(artifact_path) {
        Ok(raw) => raw,
        Err(_) => {
            return (
                None,
                format!(
                    "there is no {artifact_path_words}; write it with `{FOREIGN_EXPORT_COMMAND} {target_path}`"
                ),
            );
        }
    };
    let parsed: Value = match serde_json::from_slice(&raw) {
        Ok(parsed) => parsed,
        Err(_) => {
            return (
                None,
                format!("{artifact_path_words} is not readable JSON, so the target states nothing this edge can use"),
            );
        }
    };

    if let Err(sentence) = check_artifact_envelope(&parsed, &artifact_path_words) {
        return (None, sentence);
    }
    // TARGET INTEGRITY: the claim holds of a run only if the code that
    // runs is the code that was checked.
    if let Err(sentence) = check_target_integrity(&parsed, target_path, &artifact_path_words) {
        return (None, sentence);
    }
    // PRODUCER FRESHNESS: an artifact older than the producer binary
    // that would regenerate it is stale, the same as a hash mismatch —
    // the producer may have changed what it exports since this artifact
    // was written, and the content hash alone cannot see that.
    if let Err(sentence) = check_producer_freshness(target_path, artifact_path, &artifact_path_words) {
        return (None, sentence);
    }
    let band = match nested_string(&parsed, "runtime", "band") {
        Some(band) => band,
        None => {
            return (
                None,
                format!(
                    "{artifact_path_words} names no runtime band, and the edge's claim inherits whichever band \
                     the target's pins commit to"
                ),
            );
        }
    };
    if band != FOREIGN_RUNTIME_BAND {
        return (
            None,
            format!(
                "{artifact_path_words} commits to the runtime band {band}, and this checker's TypeScript pins \
                 commit to {FOREIGN_RUNTIME_BAND} — the edge cannot inherit semantics it has not transcribed"
            ),
        );
    }
    let (surface, called_name) = match harness_surface_of(&parsed, &artifact_path_words) {
        Ok(surface_and_name) => surface_and_name,
        Err(sentence) => return (None, sentence),
    };
    let fact = match function_fact_of(&parsed, &called_name, &artifact_path_words) {
        Ok(fact) => fact,
        Err(sentence) => return (None, sentence),
    };

    (
        Some(ForeignTsArtifact {
            path: artifact_path.to_path_buf(),
            target_file: target_path.to_owned(),
            runtime_band: band,
            surface,
            called: fact,
        }),
        String::new(),
    )
}

/// Reads the `refined` envelope and checks the `(kind, language)` pair.
/// NO version field is ever admitted: its PRESENCE is itself a decline
/// (an old-shape artifact carrying `"version"` is exactly the
/// no-version-ceremony rule's own negative case) — a reader strict-
/// parses the current shape, and any other shape is no-fact, never a
/// best-effort read.
fn check_artifact_envelope(parsed: &Value, artifact_path_words: &str) -> Result<(), String> {
    let Some(envelope) = parsed.get("refined").and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} carries no \"refined\" envelope, so nothing identifies it as a fact artifact"
        ));
    };
    let kind = envelope.get("kind").and_then(Value::as_str).unwrap_or("");
    let language = parsed.get("language").and_then(Value::as_str).unwrap_or("");

    if envelope.contains_key("version") {
        return Err(format!(
            "{artifact_path_words} states a \"version\" field in its \"refined\" envelope, and this edge reads \
             only the current cases schema, which carries no version field at all — re-export it with \
             `{FOREIGN_EXPORT_COMMAND} <target>`"
        ));
    }
    if kind != FOREIGN_ARTIFACT_KIND {
        return Err(format!(
            "{artifact_path_words} states (kind \"{kind}\"), and this edge reads only (kind \
             \"{FOREIGN_ARTIFACT_KIND}\") — the field meanings are what the kind pins"
        ));
    }
    if language != FOREIGN_ARTIFACT_LANGUAGE {
        return Err(format!(
            "{artifact_path_words} states (kind \"{kind}\", language {}), and this edge reads language \
             \"{FOREIGN_ARTIFACT_LANGUAGE}\" for that kind — the language field is what selects the \
             runtime-band pins",
            quoted_or_none(language)
        ));
    }
    Ok(())
}

/// CROSS-LANGUAGE-EDGE.md's target-integrity premise, discharged by
/// hashing the file the check will run against and comparing it to the
/// hash the producer recorded. The artifact's own `target.file` string
/// is NOT trusted as the identity — a path can be stale or relative to
/// another root; the hash is the identity.
fn check_target_integrity(parsed: &Value, target_path: &str, artifact_path_words: &str) -> Result<(), String> {
    let Some(stated) = nested_string(parsed, "target", "contentHash") else {
        return Err(format!(
            "{artifact_path_words} records no target contentHash, so nothing ties its claim to {target_path} — \
             the target-integrity premise cannot be discharged"
        ));
    };
    let bytes = std::fs::read(target_path)
        .map_err(|_| format!("the TypeScript target {target_path} cannot be read, so its stated fact cannot be tied to it"))?;
    let actual = format!("sha256:{}", sha256_hex(&bytes));
    if actual != stated {
        return Err(format!(
            "{artifact_path_words} states the fact of a target whose contents hash to {stated}, and \
             {target_path} hashes to {actual} — the exported fact is about different code than the code being \
             checked; re-export it with `{FOREIGN_EXPORT_COMMAND} {target_path}`"
        ));
    }
    Ok(())
}

/// PRODUCER FRESHNESS: an artifact whose mtime predates the producer
/// binary that would regenerate it is stale — the producer may compile
/// a newer decider or a changed export shape since this file was
/// written, and the content hash alone (a fact about the TARGET, not
/// the PRODUCER) cannot see that. Resolves the producer through the
/// SAME `resolve_foreign_producer` the auto-export path already calls
/// — no second resolution rule.
///
/// No stamps, no counters, no version field: a plain mtime comparison.
/// When the producer cannot be resolved, or either mtime cannot be
/// read, this check contributes NOTHING — the hash rule alone governs,
/// never an error from the staleness probe itself.
fn check_producer_freshness(target_path: &str, artifact_path: &Path, artifact_path_words: &str) -> Result<(), String> {
    let Some(producer) = resolve_foreign_producer(target_path) else {
        return Ok(());
    };
    let Ok(producer_mtime) = std::fs::metadata(&producer).and_then(|meta| meta.modified()) else {
        return Ok(());
    };
    let Ok(artifact_mtime) = std::fs::metadata(artifact_path).and_then(|meta| meta.modified()) else {
        return Ok(());
    };
    if producer_mtime > artifact_mtime {
        return Err(format!(
            "{artifact_path_words} predates the producer {} that would regenerate it — the fact may be about \
             a checker the producer no longer is; re-export it with `{FOREIGN_EXPORT_COMMAND} {target_path}`",
            producer.display()
        ));
    }
    Ok(())
}

/// Reads the `surface` object: the wire is JSON, carried on stdin, on
/// one argv element, or in a FILE one argv element names, and one named
/// function is what the entry point calls. The edge's whole claim is
/// about THAT function — a target whose surface reads a different
/// encoding, or calls nothing this artifact names, transports something
/// the JSON model does not describe. `surface` carries a tagged `kind`;
/// only `"stdin-json"`, `"argv-json"`, and `"file-json"` have a
/// transport model here (schema-v2.md: the other sketched kinds have no
/// reader yet).
fn harness_surface_of(parsed: &Value, artifact_path_words: &str) -> Result<(ForeignSurface, String), String> {
    let Some(surface) = parsed.get("surface").and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} describes no surface, so nothing says what the target does with stdin \
             and stdout — the JSON transport model has nothing to apply to"
        ));
    };
    let surface_kind = surface.get("kind").and_then(Value::as_str).unwrap_or("");
    let stdout = surface.get("stdout").and_then(Value::as_str).unwrap_or("");
    let channel = match surface_kind {
        "stdin-json" => {
            let stdin = surface.get("stdin").and_then(Value::as_str).unwrap_or("");
            if stdin != "json" || stdout != "json" {
                return Err(format!(
                    "{artifact_path_words} states a surface reading {} on stdin and writing {} on stdout, and \
                     this edge applies the JSON transport model to both legs",
                    quoted_or_none(stdin),
                    quoted_or_none(stdout)
                ));
            }
            ForeignSurface::StdinJson
        }
        "argv-json" => {
            if stdout != "json" {
                return Err(format!(
                    "{artifact_path_words} states an argv-json surface writing {} on stdout, and this edge \
                     reads only \"json\" on that leg",
                    quoted_or_none(stdout)
                ));
            }
            let Some(arg_index) = surface.get("argIndex").and_then(Value::as_i64) else {
                return Err(format!(
                    "{artifact_path_words} states an argv-json surface with no argIndex, so nothing says \
                     which argv element carries the JSON payload"
                ));
            };
            ForeignSurface::ArgvJson { arg_index }
        }
        "file-json" => {
            if stdout != "json" {
                return Err(format!(
                    "{artifact_path_words} states a file-json surface writing {} on stdout, and this edge \
                     reads only \"json\" on that leg",
                    quoted_or_none(stdout)
                ));
            }
            let Some(arg_index) = surface.get("argIndex").and_then(Value::as_i64) else {
                return Err(format!(
                    "{artifact_path_words} states a file-json surface with no argIndex, so nothing says \
                     which argv element names the file carrying the JSON payload"
                ));
            };
            ForeignSurface::FileJson { arg_index }
        }
        other => {
            return Err(format!(
                "{artifact_path_words} states a surface of kind {}, and this edge reads only \"stdin-json\", \
                 \"argv-json\", or \"file-json\"",
                quoted_or_none(other)
            ));
        }
    };
    let called = surface.get("calls").and_then(Value::as_str).unwrap_or("");
    if called.is_empty() {
        return Err(format!(
            "{artifact_path_words} states no surface.calls function, so nothing names the code that runs \
             when this call executes"
        ));
    }
    Ok((channel, called.to_owned()))
}

/// Reads one named function's row: its entry positions, its return, and
/// the provenance a cross-language message renders. `decode_wire_set`
/// panics on a form it does not know — its own stated contract for
/// kernel answers. An artifact is a file another program wrote, so a
/// malformed form is a decline here, never a crash.
fn function_fact_of(parsed: &Value, name: &str, artifact_path_words: &str) -> Result<ForeignTsFunctionFact, String> {
    let Some(functions) = parsed.get("functions").and_then(Value::as_object) else {
        return Err(format!("{artifact_path_words} carries no \"functions\" object at all"));
    };
    let Some(row) = functions.get(name).and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} names {name} as the harness's called function, but \"functions\" carries no \
             row for it"
        ));
    };
    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| function_fact_of_row(row, name)));
    match decoded {
        Ok(Ok(fact)) => Ok(fact),
        Ok(Err(sentence)) => Err(format!("{artifact_path_words} {sentence}")),
        Err(_) => Err(format!(
            "{artifact_path_words} states a set this checker's kernel grammar does not read, so the fact for \
             {name} cannot be decoded"
        )),
    }
}

/// One function row's fields read out, under the caller's `catch_unwind`
/// — every `decode_wire_set` call here can panic on a malformed form.
fn function_fact_of_row(row: &serde_json::Map<String, Value>, name: &str) -> Result<ForeignTsFunctionFact, String> {
    let entries = artifact_entries_of(row, name)?;
    let Some(returned) = row.get("return").and_then(Value::as_object) else {
        return Err(format!("carries no \"return\" object for {name}, so nothing crosses back from this call"));
    };
    let return_cases = cases_of(returned, &format!("the return for {name}"))?;
    let stdout_pure = returned.get("stdoutPure").and_then(Value::as_bool).unwrap_or(false);
    let (provenance_line, provenance_said) = artifact_provenance_of(row);
    Ok(ForeignTsFunctionFact {
        name: name.to_owned(),
        entry: entries,
        return_cases,
        stdout_pure,
        provenance_line,
        provenance_said,
    })
}

/// Reads the entry rows in the order the artifact spells them — that
/// order IS the positional order of the target's parameters, which is
/// how an argument finds the row it must fit.
fn artifact_entries_of(row: &serde_json::Map<String, Value>, name: &str) -> Result<Vec<ForeignTsEntry>, String> {
    let Some(raw_entries) = row.get("entry").and_then(Value::as_array) else {
        return Err(format!("states no entry positions for {name}, so nothing says what the target admits"));
    };
    let mut entries = Vec::with_capacity(raw_entries.len());
    for (index, raw_entry) in raw_entries.iter().enumerate() {
        let Some(entry_row) = raw_entry.as_object() else {
            return Err(format!("states an unreadable entry position {index} for {name}"));
        };
        let entry_name = entry_row.get("name").and_then(Value::as_str).unwrap_or("").to_owned();
        if let Some(sequence) = entry_row.get("sequence").and_then(Value::as_object) {
            let Some(element) = sequence.get("element").and_then(Value::as_object) else {
                return Err(format!("states a sequence entry {entry_name} for {name} with no element cases"));
            };
            let element_cases = cases_of(element, &format!("the sequence entry {entry_name} for {name}"))?;
            let length_at_least = sequence.get("lengthAtLeast").and_then(Value::as_i64).unwrap_or(0);
            entries.push(ForeignTsEntry {
                name: entry_name,
                sequence: Some((element_cases, length_at_least)),
                scalar: None,
            });
            continue;
        }
        let scalar_cases = cases_of(entry_row, &format!("the entry position {entry_name} for {name}"))?;
        entries.push(ForeignTsEntry {
            name: entry_name,
            sequence: None,
            scalar: Some(scalar_cases),
        });
    }
    Ok(entries)
}

/// Reads a `"cases"` array off an object that carries one — the RULED
/// schema's own unit at both the return position and a scalar entry
/// position (and a sequence entry's `element` object). Strict-parse: a
/// bare `"set"` field (the earlier shape) is NOT read as a one-case
/// fallback — that shape is exactly what the no-version-ceremony rule
/// calls NO-FACT, and the caller's own decline sentence is what a stale
/// artifact earns, never a silent best-effort reinterpretation.
fn cases_of(carrier: &serde_json::Map<String, Value>, described: &str) -> Result<Vec<ForeignCase>, String> {
    let Some(raw_cases) = carrier.get("cases").and_then(Value::as_array) else {
        return Err(format!(
            "{described} states no \"cases\" array, so nothing says what shape the value takes — re-export \
             it with `{FOREIGN_EXPORT_COMMAND} <target>`"
        ));
    };
    cases_array_of(raw_cases, described)
}

/// Reads a cases array directly — `cases_of`'s own body, factored out so
/// an object case's MEMBER (whose value in the wire IS the bare cases
/// array, `fact_export.rs::Case::to_json`'s own `Case::Object` arm: `{name:
/// cases_json(cases)}`, never a `{"cases": [...]}` wrapper) parses through
/// the identical rule rather than a second copy of it.
fn cases_array_of(raw_cases: &[Value], described: &str) -> Result<Vec<ForeignCase>, String> {
    if raw_cases.is_empty() {
        return Err(format!("{described} states an empty \"cases\" array, which admits no value at all"));
    }
    let mut cases = Vec::with_capacity(raw_cases.len());
    for (index, raw_case) in raw_cases.iter().enumerate() {
        let Some(case) = raw_case.as_object() else {
            return Err(format!("{described} states an unreadable case {index}"));
        };
        let sort = case.get("sort").and_then(Value::as_str).unwrap_or("");
        cases.push(match sort {
            "number" => {
                let Some(raw_set) = case.get("set") else {
                    return Err(format!("{described} states a number case {index} with no set"));
                };
                ForeignCase::Number(decode_wire_set(raw_set))
            }
            "string" => {
                let Some(raw_set) = case.get("set") else {
                    return Err(format!("{described} states a string case {index} with no set"));
                };
                ForeignCase::String(decode_wire_set(raw_set))
            }
            "boolean" => ForeignCase::Boolean,
            "null" => ForeignCase::Null,
            "object" => object_case_of(case, &format!("{described}'s case {index}"))?,
            other => {
                return Err(format!(
                    "{described} states a case {index} of sort {}, and this reader admits only \"number\", \
                     \"string\", \"boolean\", \"null\", or \"object\"",
                    quoted_or_none(other)
                ));
            }
        });
    }
    Ok(cases)
}

/// Reads one `{"sort": "object", "members": {...}, "closed": bool}` case
/// — the RULED object case's own strict parse (CROSS-LANGUAGE-EDGE.md
/// §17, JT-prioritized 2026-08-21). `members` must be a JSON OBJECT
/// mapping each key DIRECTLY to its own cases ARRAY (never a `{"cases":
/// [...]}` wrapper — `fact_export.rs::Case::to_json`'s `Case::Object` arm
/// writes `{name: cases_json(cases)}`, the bare array, so the parser's
/// shape must match the writer's exactly), recursed through
/// `cases_array_of` so a nested object case parses through the identical
/// rule; `closed` must be a JSON BOOLEAN. Any deviation — `members`
/// missing or not an object, `closed` missing or not a boolean, a
/// member's own value not itself a cases array — declines by name through
/// the ordinary `Err` path, exactly the same "an artifact is a file, not
/// a promise" discipline every other malformed shape in this file earns;
/// nothing here guesses at a member.
fn object_case_of(case: &serde_json::Map<String, Value>, described: &str) -> Result<ForeignCase, String> {
    let Some(raw_members) = case.get("members").and_then(Value::as_object) else {
        return Err(format!(
            "{described} states an object case with no \"members\" object, so nothing says what keys it admits"
        ));
    };
    let Some(closed) = case.get("closed").and_then(Value::as_bool) else {
        return Err(format!(
            "{described} states an object case with no boolean \"closed\" field, so nothing says whether its \
             key set is exact"
        ));
    };
    let mut members = Vec::with_capacity(raw_members.len());
    for (name, raw_member) in raw_members {
        let Some(member_cases) = raw_member.as_array() else {
            return Err(format!("{described} states a member '{name}' that is not a cases array"));
        };
        let cases = cases_array_of(member_cases, &format!("{described}'s member '{name}'"))?;
        members.push((name.clone(), cases));
    }
    Ok(ForeignCase::Object { members, closed })
}

/// Reads where the target's claim was made. Absent fields answer
/// `(0, "")` rather than declining — provenance makes a message
/// readable; it is not a premise of the crossing.
fn artifact_provenance_of(row: &serde_json::Map<String, Value>) -> (usize, String) {
    let Some(provenance) = row.get("provenance").and_then(Value::as_object) else {
        return (0, String::new());
    };
    let line = provenance.get("line").and_then(Value::as_i64).unwrap_or(0).max(0) as usize;
    let said = provenance.get("said").and_then(Value::as_str).unwrap_or("").to_owned();
    (line, said)
}

/// Reads `parsed[outer][inner]` as a string.
fn nested_string(parsed: &Value, outer: &str, inner: &str) -> Option<String> {
    parsed.get(outer)?.get(inner)?.as_str().map(str::to_owned)
}

/// Spells a harness channel for a message: the word it states, or
/// "nothing" where the field is absent.
fn quoted_or_none(word: &str) -> String {
    if word.is_empty() {
        "nothing".to_owned()
    } else {
        format!("\"{word}\"")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;
    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::integer;
    use refined_sets::refinement_forms::make_refined_set;

    /// A fresh temp directory (unique per test run) marked as a project
    /// root with `.git`, so `cache_artifact_path`/`project_root_of`
    /// resolve exactly this directory rather than whatever ancestor of
    /// the real checkout happens to hold `.git`.
    fn temp_project_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "refinedpy_foreign_edge_artifact_test_{label}_{}_{}",
            std::process::id(),
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).expect("create temp project root");
        fs::create_dir_all(root.join(".git")).expect("mark the temp root as a project root");
        root
    }

    /// A well-built artifact JSON for a one-parameter scalar function,
    /// with the real sha256 of `source` as its contentHash.
    fn well_formed_artifact(source: &[u8], called: &str) -> Value {
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        json!({
            "refined": {"kind": FOREIGN_ARTIFACT_KIND},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": called},
            "functions": {
                called: {
                    "entry": [{"name": "x", "cases": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}]}],
                    "return": {"cases": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}], "stdoutPure": true},
                    "provenance": {"line": 3, "said": "given 'x' is a nonnegative integer, this body's returns derive a nonnegative integer"},
                }
            }
        })
    }

    #[test]
    fn a_well_formed_artifact_answers_the_called_functions_fact() {
        let root = temp_project_root("well_formed");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_artifact(&source, "f").to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("a well-formed artifact reads as a fact");
        assert_eq!(artifact.called.name, "f");
        assert_eq!(artifact.called.entry.len(), 1);
        assert_eq!(artifact.called.entry[0].name, "x");
        assert!(artifact.called.entry[0].scalar.is_some());
        assert!(artifact.called.stdout_pure);
        assert_eq!(artifact.called.provenance_line, 3);

        fs::remove_dir_all(&root).ok();
    }

    /// The `argv-json` sibling of `well_formed_artifact` — the same
    /// function fact, carried on an argv element instead of stdin.
    fn well_formed_argv_json_artifact(source: &[u8], called: &str, arg_index: i64) -> Value {
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        json!({
            "refined": {"kind": FOREIGN_ARTIFACT_KIND},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "argv-json", "argIndex": arg_index, "stdout": "json", "calls": called},
            "functions": {
                called: {
                    "entry": [{"name": "x", "cases": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}]}],
                    "return": {"cases": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}], "stdoutPure": true},
                    "provenance": {"line": 3, "said": "given 'x' is a nonnegative integer, this body's returns derive a nonnegative integer"},
                }
            }
        })
    }

    /// A well-formed `argv-json` artifact reads its channel as
    /// `ForeignSurface::ArgvJson` with the stated `argIndex` carried
    /// through, and every other field reads exactly as the stdin-json
    /// case does — the same transport model, a different carrier.
    #[test]
    fn a_well_formed_argv_json_artifact_reads_its_channel_and_index() {
        let root = temp_project_root("argv_json_well_formed");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_argv_json_artifact(&source, "f", 2).to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("a well-formed argv-json artifact reads as a fact");
        assert_eq!(artifact.surface, ForeignSurface::ArgvJson { arg_index: 2 });
        assert_eq!(artifact.called.name, "f");

        fs::remove_dir_all(&root).ok();
    }

    /// An `argv-json` surface with no `argIndex` at all declines — the
    /// carrier is named but the position it reads from is not.
    #[test]
    fn an_argv_json_surface_with_no_arg_index_declines() {
        let root = temp_project_root("argv_json_missing_index");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_argv_json_artifact(&source, "f", 2);
        artifact["surface"].as_object_mut().unwrap().remove("argIndex");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an argv-json surface with no argIndex must decline");
        assert!(sentence.contains("argIndex"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// The `file-json` sibling of `well_formed_artifact` — the same
    /// function fact, the payload carried in a FILE named at the argv
    /// element rather than on the element's own text.
    fn well_formed_file_json_artifact(source: &[u8], called: &str, arg_index: i64) -> Value {
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        json!({
            "refined": {"kind": FOREIGN_ARTIFACT_KIND},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "file-json", "argIndex": arg_index, "stdout": "json", "calls": called},
            "functions": {
                called: {
                    "entry": [{"name": "x", "cases": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}]}],
                    "return": {"cases": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}], "stdoutPure": true},
                    "provenance": {"line": 3, "said": "given 'x' is a nonnegative integer, this body's returns derive a nonnegative integer"},
                }
            }
        })
    }

    /// A well-formed `file-json` artifact reads its channel as
    /// `ForeignSurface::FileJson` with the stated `argIndex` carried
    /// through — the same transport model as `argv-json`, a different
    /// carrier (a file the argv element names, not the element's own
    /// text).
    #[test]
    fn a_well_formed_file_json_artifact_reads_its_channel_and_index() {
        let root = temp_project_root("file_json_well_formed");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_file_json_artifact(&source, "f", 2).to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("a well-formed file-json artifact reads as a fact");
        assert_eq!(artifact.surface, ForeignSurface::FileJson { arg_index: 2 });
        assert_eq!(artifact.called.name, "f");

        fs::remove_dir_all(&root).ok();
    }

    /// A `file-json` surface with no `argIndex` at all declines — the
    /// carrier is named but the position it reads the file's path from
    /// is not.
    #[test]
    fn a_file_json_surface_with_no_arg_index_declines() {
        let root = temp_project_root("file_json_missing_index");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_file_json_artifact(&source, "f", 2);
        artifact["surface"].as_object_mut().unwrap().remove("argIndex");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("a file-json surface with no argIndex must decline");
        assert!(sentence.contains("argIndex"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_missing_artifact_names_the_cache_path_and_the_command() {
        // an empty project root with no producer anywhere under it and
        // (assuming the test host has no refinedts-check-bin on PATH)
        // no producer resolvable at all — the sentence must still name
        // the artifact path and the export command, whether or not the
        // auto-export attempt itself found a producer.
        let root = temp_project_root("missing");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("no artifact exists and no producer can write one in this temp root");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        assert!(sentence.contains(artifact_path.to_str().unwrap()), "sentence = {sentence:?}");
        assert!(sentence.contains("-export-fact"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_stale_hash_declines_naming_target_integrity() {
        let root = temp_project_root("stale_hash");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        // built against different bytes than what's on disk now
        fs::write(&artifact_path, well_formed_artifact(b"stale source", "f").to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("a hash mismatch must decline");
        assert!(sentence.contains("hash"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// Writes a placeholder file at the producer's CONVENTION path
    /// under `root` (`resolve_foreign_producer`'s own first-checked
    /// location) — its content is never executed by these tests, only
    /// its existence and mtime matter to `check_producer_freshness`.
    fn write_placeholder_producer(root: &Path) -> PathBuf {
        let producer_path = root.join(FOREIGN_PRODUCER_RELATIVE);
        fs::create_dir_all(producer_path.parent().unwrap()).expect("create producer dir");
        fs::write(&producer_path, b"placeholder producer").expect("write placeholder producer");
        producer_path
    }

    /// An artifact older than the producer binary that would regenerate
    /// it is STALE — the same path a hash mismatch takes: the freshness
    /// gate declines, the uncached reader attempts one re-export through
    /// the resolved producer, and (since the placeholder producer here
    /// is not a real executable) that attempt fails, so the sentence
    /// carries both the staleness reason and the auto-export failure.
    #[test]
    fn an_artifact_older_than_the_producer_is_stale_and_attempts_reexport() {
        let root = temp_project_root("stale_producer");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_artifact(&source, "f").to_string()).expect("write artifact");

        // the producer binary is built (or rebuilt) AFTER the artifact
        // already exists — the artifact predates it
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_placeholder_producer(&root);

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an artifact older than its producer must decline as stale");
        assert!(sentence.contains("predates the producer"), "sentence = {sentence:?}");
        assert!(sentence.contains("auto-export declined"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// REGRESSION: an artifact newer than the producer, with a matching
    /// hash, reads fresh — the freshness gate contributes nothing when
    /// the artifact is not stale, and the existing premises (hash, band,
    /// harness shape) still govern exactly as before this gate existed.
    #[test]
    fn an_artifact_newer_than_the_producer_with_a_matching_hash_reads_fresh() {
        let root = temp_project_root("fresh_producer");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");

        // the producer binary exists FIRST, then the artifact is written
        // afterward — the artifact is newer than its producer
        write_placeholder_producer(&root);
        std::thread::sleep(std::time::Duration::from_millis(20));
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_artifact(&source, "f").to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("an artifact newer than its producer, with a matching hash, must read fresh");
        assert_eq!(artifact.called.name, "f");

        fs::remove_dir_all(&root).ok();
    }

    /// REGRESSION: when no producer resolves at all (none at the
    /// convention path, none on `PATH`), the freshness gate contributes
    /// nothing and the hash rule alone governs — an artifact with a
    /// matching hash reads fresh even though staleness can never be
    /// checked in this project root.
    #[test]
    fn an_unresolvable_producer_leaves_the_hash_rule_alone_in_charge() {
        let root = temp_project_root("no_producer");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_artifact(&source, "f").to_string()).expect("write artifact");

        // no producer written anywhere under root, and (assuming the
        // test host has no refinedts-check-bin on PATH) none resolvable
        // at all — the read must still succeed on the hash rule alone
        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("with no producer resolvable, the hash rule alone must still admit the fact");
        assert_eq!(artifact.called.name, "f");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_wrong_band_declines() {
        let root = temp_project_root("wrong_band");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["runtime"]["band"] = json!("node-18");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("a different runtime band must decline");
        assert!(sentence.contains("runtime band"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn a_malformed_set_form_declines_via_the_catch_unwind_not_a_panic() {
        let root = temp_project_root("malformed_set");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["functions"]["f"]["return"]["cases"] =
            json!([{"sort": "number", "set": {"forms": [{"form": "not-a-real-form"}]}}]);
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        // the panic inside decode_wire_set must not tear down the test
        // process — it must surface as a named decline
        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("a malformed form must decline, never panic the process");
        assert!(sentence.contains("kernel grammar"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    // --- the RULED object case (CROSS-LANGUAGE-EDGE.md §17) ----------

    /// A well-formed `{"sort": "object", "members": {...}, "closed": bool}`
    /// case round-trips into the `ForeignCase::Object` shape exactly:
    /// each member's own cases list decoded through the SAME `cases_of`
    /// path a top-level return/entry position goes through, and `closed`
    /// carried through unchanged.
    #[test]
    fn a_well_formed_object_case_round_trips_into_the_foreign_case_shape() {
        let root = temp_project_root("object_case_well_formed");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["functions"]["f"]["return"]["cases"] = json!([{
            "sort": "object",
            "members": {
                "age": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}],
            },
            "closed": true,
        }]);
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("a well-formed object case reads as a fact");
        assert_eq!(artifact.called.return_cases.len(), 1);
        let ForeignCase::Object { members, closed } = &artifact.called.return_cases[0] else {
            panic!("expected an object case, got {:?}", artifact.called.return_cases[0]);
        };
        assert!(*closed, "the artifact states closed: true");
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "age");
        assert_eq!(members[0].1, vec![ForeignCase::Number(scalar)]);

        fs::remove_dir_all(&root).ok();
    }

    /// An object case whose `members` field is not a JSON object (a bare
    /// array, standing in for every other malformed shape) declines by
    /// name — the strict parse never falls back to a best-effort reading.
    #[test]
    fn a_malformed_members_value_declines_by_name() {
        let root = temp_project_root("object_case_malformed_members");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["functions"]["f"]["return"]["cases"] =
            json!([{"sort": "object", "members": ["not", "an", "object"], "closed": true}]);
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("a non-object \"members\" value must decline");
        assert!(sentence.contains("members"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// An object case with no `closed` field at all (or a non-boolean
    /// one) declines by name — the same strict parse as a missing
    /// `members`.
    #[test]
    fn an_object_case_with_no_closed_field_declines_by_name() {
        let root = temp_project_root("object_case_no_closed");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["functions"]["f"]["return"]["cases"] = json!([{"sort": "object", "members": {}}]);
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an object case with no \"closed\" field must decline");
        assert!(sentence.contains("closed"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// An object case NESTED inside another object case's member —
    /// `{"members": {"outer": [{"sort": "object", "members": {"inner": [...]}, "closed": true}]}}`
    /// — parses through the identical recursive rule, so a member can
    /// itself be an object case at any depth.
    #[test]
    fn a_nested_object_in_object_case_parses() {
        let root = temp_project_root("object_case_nested");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["functions"]["f"]["return"]["cases"] = json!([{
            "sort": "object",
            "members": {
                "outer": [{
                    "sort": "object",
                    "members": {
                        "inner": [{"sort": "number", "set": refined_kernel::wire_format::wire_set(&scalar)}],
                    },
                    "closed": true,
                }],
            },
            "closed": false,
        }]);
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let artifact = read.expect("a nested object-in-object case reads as a fact");
        let ForeignCase::Object { members, closed } = &artifact.called.return_cases[0] else {
            panic!("expected the outer object case, got {:?}", artifact.called.return_cases[0]);
        };
        assert!(!closed, "the outer case states closed: false");
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].0, "outer");
        let ForeignCase::Object { members: inner_members, closed: inner_closed } = &members[0].1[0] else {
            panic!("expected the nested object case, got {:?}", members[0].1[0]);
        };
        assert!(*inner_closed, "the inner case states closed: true");
        assert_eq!(inner_members.len(), 1);
        assert_eq!(inner_members[0].0, "inner");
        assert_eq!(inner_members[0].1, vec![ForeignCase::Number(scalar)]);

        fs::remove_dir_all(&root).ok();
    }

    /* ── the premise-conformance corpus ──────────────────────────────
     *
     * docs/one-checker/premise-unification.md names the deliverable: a
     * hand-built fact-artifact corpus — valid and broken, each broken
     * row naming exactly ONE premise — read by BOTH consumers' test
     * suites. Cross-language code cannot be shared between the Go and
     * Rust readers; the corpus (packages/refinedts/edge-premise-fixtures/)
     * and its manifest.json are the shared part.
     *
     * SCOPE: this test exercises only the rows whose premise
     * `read_foreign_ts_artifact` itself discharges — the envelope,
     * target integrity, runtime band, harness shape, and the kernel's
     * set decoder. Two manifest rows (crossing-fit, NaN-admission) are
     * NOT reader premises: they are judged against the WALK's crossing
     * value by `check_outbound_leg`, which needs a live kernel and an
     * `Environment` this reader-focused module does not construct —
     * those rows are exercised by this crate's `foreign_edge.rs` test
     * module instead. The stdout-not-pure row IS read here, but only
     * for the flag the reader carries through; the decline sentence
     * naming channel purity is built one layer up, in
     * `foreign_edge_at`, and is exercised there, not duplicated here.
     * See the corpus's own manifest.json and README.md.
     */

    /// The corpus directory's absolute path, derived from `file!()` at
    /// compile time via `CARGO_MANIFEST_DIR` — the same idiom
    /// `fact_export.rs`'s own `the_tutorial_fixture_exports_its_structure`
    /// test already uses, never an environment variable read at run
    /// time.
    fn conformance_fixtures_dir() -> PathBuf {
        // this crate's manifest: packages/refinedpy/pyrefly/pyrefly
        // the corpus:            packages/refinedts/edge-premise-fixtures
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../refinedts/edge-premise-fixtures"))
    }

    /// One manifest row, read generically off the JSON — only the
    /// fields this test consults.
    struct ConformanceRow {
        id: String,
        artifact: String,
        target_source: String,
        premise_broken: String,
        verdict: String,
        key_phrase_rust: Option<String>,
        exercised_by: Vec<String>,
        /// A row mid-respell to v2 carries `"rustSentenceStale": true`
        /// while its paired `.artifact.json` no longer matches this
        /// reader's actual decline wording — see
        /// `the_conformance_corpus_reads_as_the_manifest_expects`, which
        /// fails loudly on this flag rather than silently comparing
        /// against a phrase the reader has already moved past.
        rust_sentence_stale: bool,
    }

    fn load_conformance_manifest(dir: &Path) -> Vec<ConformanceRow> {
        let raw = fs::read(dir.join("manifest.json")).expect("the conformance corpus manifest is committed");
        let parsed: Value = serde_json::from_slice(&raw).expect("manifest.json parses");
        let rows = parsed.get("rows").and_then(Value::as_array).expect("manifest.json states a rows array");
        rows.iter()
            .map(|row| ConformanceRow {
                id: row.get("id").and_then(Value::as_str).unwrap_or("").to_owned(),
                artifact: row.get("artifact").and_then(Value::as_str).unwrap_or("").to_owned(),
                target_source: row.get("targetSource").and_then(Value::as_str).unwrap_or("").to_owned(),
                premise_broken: row.get("premiseBroken").and_then(Value::as_str).unwrap_or("").to_owned(),
                verdict: row.get("verdict").and_then(Value::as_str).unwrap_or("").to_owned(),
                key_phrase_rust: row
                    .get("keyPhrase")
                    .and_then(Value::as_object)
                    .and_then(|phrases| phrases.get("rust"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                exercised_by: row
                    .get("exercisedBy")
                    .and_then(Value::as_array)
                    .map(|list| list.iter().filter_map(Value::as_str).map(str::to_owned).collect())
                    .unwrap_or_default(),
                rust_sentence_stale: row.get("rustSentenceStale").and_then(Value::as_bool).unwrap_or(false),
            })
            .collect()
    }

    /// Whether a manifest row names this side ("rust-reader") among
    /// what exercises it — the rows whose premise lives one layer
    /// above the reader (crossing-fit, NaN-admission) name only the
    /// existing edge-level tests, and this loop must skip them rather
    /// than force them through a harness that cannot construct a
    /// crossing value.
    fn exercised_by_rust_reader(row: &ConformanceRow) -> bool {
        row.exercised_by.iter().any(|who| who.starts_with("rust-reader"))
    }

    /// Writes the row's artifact (with `{{HASH}}` substituted against
    /// the REAL sha256 of its paired target-source, computed at run
    /// time — a hash cannot be hand-pasted and stay honest) into a
    /// fresh temp project root, and answers the target path to read
    /// the artifact against.
    fn materialize_conformance_row(dir: &Path, row: &ConformanceRow) -> (PathBuf, PathBuf) {
        let root = temp_project_root(&format!("conformance_{}", row.id));
        let mut artifact_text =
            String::from_utf8(fs::read(dir.join(&row.artifact)).expect("reading the row's artifact")).expect("artifact is UTF-8");

        // the target's extension follows the artifact's own declared
        // language field — a row with no target-source (it declines
        // before target integrity) still needs a real file at the
        // resolved cache path, and defaults to .py, since the
        // placeholder's content is never actually read
        let is_typescript = artifact_text.contains(r#""language": "typescript""#);
        let target_name = if is_typescript { "audio_level.ts" } else { "audio_level.py" };
        let target = root.join(target_name);

        if !row.target_source.is_empty() {
            let source = fs::read(dir.join(&row.target_source)).expect("reading the row's target-source");
            fs::write(&target, &source).expect("write target");
            let hash = format!("sha256:{}", sha256_hex(&source));
            artifact_text = artifact_text.replace("{{HASH}}", &hash);
        } else {
            // rows that decline before the hash check (kind/version)
            // carry no paired target-source; write a placeholder target
            // so a read attempt that DID reach the hash check would fail
            // loudly rather than silently pass against a missing file
            fs::write(&target, b"placeholder - this row declines before target integrity\n").expect("write placeholder target");
        }

        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, &artifact_text).expect("writing the artifact");
        (target, root)
    }

    /// Iterates `packages/refinedts/edge-premise-fixtures/manifest.json`
    /// and runs each reader-level row through
    /// `read_foreign_ts_artifact`, asserting the verdict class
    /// (consumed vs. declined) and — for a decline — that the sentence
    /// contains the manifest's own recorded key phrase.
    #[test]
    fn the_conformance_corpus_reads_as_the_manifest_expects() {
        let dir = conformance_fixtures_dir();
        let rows = load_conformance_manifest(&dir);
        assert!(!rows.is_empty(), "the manifest must state at least one row");

        for row in rows.iter().filter(|row| exercised_by_rust_reader(row)) {
            assert!(
                !row.rust_sentence_stale,
                "row {:?} is marked rustSentenceStale — this reader's decline wording changed and the \
                 manifest's keyPhrase.rust was not updated to match; fill in the respelled phrase and clear \
                 the flag before this row can be trusted",
                row.id
            );
            let (target, root) = materialize_conformance_row(&dir, row);
            let read = read_foreign_ts_artifact(target.to_str().unwrap());

            match row.verdict.as_str() {
                "consumed" => {
                    read.unwrap_or_else(|sentence| {
                        panic!("row {:?} (premise {:?}) expected to be consumed, but declined: {sentence}", row.id, row.premise_broken)
                    });
                }
                "declined" => {
                    let sentence = read.expect_err(&format!(
                        "row {:?} (premise {:?}) expected to decline, but was consumed",
                        row.id, row.premise_broken
                    ));
                    let phrase = row
                        .key_phrase_rust
                        .as_deref()
                        .unwrap_or_else(|| panic!("row {:?} declined verdict has no manifest keyPhrase.rust", row.id));
                    assert!(
                        sentence.contains(phrase),
                        "row {:?} sentence {sentence:?} does not contain the manifest's key phrase {phrase:?}",
                        row.id
                    );
                }
                "consumed-with-flag-false" => {
                    let artifact = read.unwrap_or_else(|sentence| {
                        panic!("row {:?} expected to be consumed (stdoutPure false carried through), but declined: {sentence}", row.id)
                    });
                    assert!(!artifact.called.stdout_pure, "row {:?} expected stdout_pure == false, got true", row.id);
                }
                other => panic!("row {:?} states an unhandled verdict class {other:?}", row.id),
            }

            fs::remove_dir_all(&root).ok();
        }
    }

    #[test]
    fn a_changed_mtime_refreshes_the_memo() {
        let root = temp_project_root("mtime_refresh");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_artifact(&source, "f").to_string()).expect("write artifact");

        let first = read_foreign_ts_artifact(target.to_str().unwrap()).expect("first read succeeds");
        assert_eq!(first.called.name, "f");

        // rewrite the artifact under a different called name — a plain
        // process-lifetime memo would still answer "f"; the mtime check
        // must notice the file changed and re-read it
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&artifact_path, well_formed_artifact(&source, "g").to_string()).expect("rewrite artifact");

        let second = read_foreign_ts_artifact(target.to_str().unwrap()).expect("second read succeeds");
        assert_eq!(second.called.name, "g", "the memo must refresh once the artifact's mtime changes");

        fs::remove_dir_all(&root).ok();
    }

    /// An unknown `kind` declines, naming the accepted form — never a
    /// best-effort read of a schema this reader does not know.
    #[test]
    fn an_unknown_kind_declines_naming_the_accepted_form() {
        let root = temp_project_root("unknown_kind");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["refined"] = json!({"kind": "python-fact-artifact"});
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an unknown kind must decline");
        assert!(sentence.contains("python-fact-artifact"), "sentence = {sentence:?}");
        assert!(sentence.contains(FOREIGN_ARTIFACT_KIND), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// AN OLD-SHAPE ARTIFACT (a "version" field present, and the earlier
    /// bare "set" spelling instead of "cases") reads as NO-FACT — the
    /// no-version-ceremony rule's own negative case: the version field's
    /// mere PRESENCE is enough to decline, independent of its value, and
    /// the existing self-refresh re-export path is what a reader facing
    /// this shape falls back on.
    #[test]
    fn an_old_shape_artifact_with_a_version_field_reads_as_no_fact() {
        let root = temp_project_root("old_shape_version_field");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        let artifact = json!({
            "refined": {"kind": FOREIGN_ARTIFACT_KIND, "version": 2},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(&source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "f"},
            "functions": {
                "f": {
                    "entry": [{"name": "x", "set": refined_kernel::wire_format::wire_set(&scalar)}],
                    "return": {"set": refined_kernel::wire_format::wire_set(&scalar), "stdoutPure": true},
                    "provenance": {"line": 3, "said": "the old shape's own sentence"},
                }
            }
        });
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an old-shape artifact carrying a version field must read as no-fact");
        assert!(sentence.contains("version"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// The SAME old-shape refusal when the version field is absent but
    /// the bare "set" spelling remains (never the "cases" list): the
    /// reader's `cases_of` strict-parse declines naming the missing
    /// "cases" array, rather than falling back to reading "set" as a
    /// one-case list.
    #[test]
    fn an_old_shape_artifact_with_a_bare_set_reads_as_no_fact() {
        let root = temp_project_root("old_shape_bare_set");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let scalar = make_refined_set(vec![integer(), at_least(0.0)]);
        let artifact = json!({
            "refined": {"kind": FOREIGN_ARTIFACT_KIND},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(&source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "f"},
            "functions": {
                "f": {
                    "entry": [{"name": "x", "set": refined_kernel::wire_format::wire_set(&scalar)}],
                    "return": {"set": refined_kernel::wire_format::wire_set(&scalar), "stdoutPure": true},
                    "provenance": {"line": 3, "said": "the old shape's own sentence"},
                }
            }
        });
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("a bare \"set\" spelling with no \"cases\" array must read as no-fact");
        assert!(sentence.contains("cases"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// An envelope whose `language` is not "typescript" declines — the
    /// kind is shared across producer languages, so the language field
    /// (not the kind) is what this reader checks per-language.
    #[test]
    fn an_envelope_of_another_language_declines() {
        let root = temp_project_root("wrong_language");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["language"] = json!("python");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an envelope naming a different language must decline");
        assert!(sentence.contains("python"), "sentence = {sentence:?}");
        assert!(sentence.contains("language"), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    // --- THE CROSS-PROCESS EXPORT-CHAIN CYCLE GUARD ----------------------
    //
    // `export_foreign_ts_artifact` takes the chain as a plain `&str`
    // parameter rather than reading `REFINED_EXPORT_CHAIN` itself — the
    // testable design chosen here: `std::env::set_var` is `unsafe` and
    // races across this crate's parallel test runner (many tests in this
    // file already run concurrently, each in its own temp project root),
    // so a test cannot safely set the process environment and expect
    // only ITS OWN call to observe that value. Parameterizing the chain
    // and testing the parameterized function directly needs no process
    // mutation at all.

    /// A target whose absolute path already appears as a hop on the
    /// chain is recognized regardless of a relative spelling at the call
    /// site — the comparison is absolute-to-absolute.
    #[test]
    fn export_chain_contains_finds_a_hop_by_its_absolute_path() {
        let root = temp_project_root("chain_contains");
        let target = root.join("a.py");
        fs::write(&target, b"# placeholder\n").expect("write target");
        let absolute = std::path::absolute(&target).unwrap();
        let chain = absolute.to_string_lossy().into_owned();

        assert!(export_chain_contains(&chain, target.to_str().unwrap()));
        assert!(!export_chain_contains(&chain, root.join("b.py").to_str().unwrap()));
        assert!(!export_chain_contains("", target.to_str().unwrap()), "an empty chain contains no hop");

        fs::remove_dir_all(&root).ok();
    }

    /// A multi-hop chain is read as ':'-separated absolute paths; a
    /// target matching any hop (not only the last) is recognized.
    #[test]
    fn export_chain_contains_checks_every_hop_not_only_the_last() {
        let root = temp_project_root("chain_multi_hop");
        let a = root.join("a.py");
        let b = root.join("b.ts");
        fs::write(&a, b"# placeholder\n").expect("write a");
        fs::write(&b, b"// placeholder\n").expect("write b");
        let chain = format!(
            "{}:{}",
            std::path::absolute(&a).unwrap().to_string_lossy(),
            std::path::absolute(&b).unwrap().to_string_lossy()
        );

        assert!(export_chain_contains(&chain, a.to_str().unwrap()), "the first hop must be recognized");
        assert!(export_chain_contains(&chain, b.to_str().unwrap()), "the last hop must be recognized");

        fs::remove_dir_all(&root).ok();
    }

    /// The cycle sentence names the recursing target and renders the
    /// whole chain that led back to it, the recursing target appended
    /// last — a reader sees the exact cycle, not a generic refusal.
    #[test]
    fn export_chain_cycle_sentence_names_the_recursing_target_and_the_whole_chain() {
        let root = temp_project_root("chain_sentence");
        let a = root.join("a.py");
        let b = root.join("b.ts");
        fs::write(&a, b"# placeholder\n").expect("write a");
        fs::write(&b, b"// placeholder\n").expect("write b");
        let absolute_a = std::path::absolute(&a).unwrap();
        let absolute_b = std::path::absolute(&b).unwrap();
        let chain = format!("{}:{}", absolute_a.to_string_lossy(), absolute_b.to_string_lossy());

        let sentence = export_chain_cycle_sentence(&chain, a.to_str().unwrap());
        assert!(sentence.contains("recurses back"), "sentence = {sentence:?}");
        assert!(sentence.contains(&absolute_a.to_string_lossy().into_owned()), "sentence = {sentence:?}");
        assert!(sentence.contains(&absolute_b.to_string_lossy().into_owned()), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// A target already on the chain declines the spawn outright — the
    /// producer is never resolved or invoked, and the sentence names the
    /// cycle, exactly the ruling this guard exists to apply. Uses a
    /// project root with NO producer at all: if the guard failed to
    /// short-circuit, the failure would read as an unresolved-producer
    /// sentence instead, which this test's own assertion distinguishes
    /// from the cycle sentence it requires.
    #[test]
    fn a_chain_marked_target_declines_the_spawn_with_the_cycle_sentence() {
        let root = temp_project_root("chain_marked_declines");
        let target = root.join("a.py");
        fs::write(&target, b"# placeholder\n").expect("write target");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        let absolute_target = std::path::absolute(&target).unwrap();
        let chain = absolute_target.to_string_lossy().into_owned();

        let result = export_foreign_ts_artifact(target.to_str().unwrap(), &artifact_path, &chain);
        let sentence = result.expect_err("a target already on the chain must decline, not spawn");
        assert!(sentence.contains("recurses back"), "sentence = {sentence:?}");
        assert!(sentence.contains(&absolute_target.to_string_lossy().into_owned()), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }

    /// A clean chain (no hop matching the target) behaves exactly as
    /// before this guard existed: with no producer resolvable in this
    /// temp root, the decline names the ordinary "no producer" reason,
    /// never the cycle sentence — the guard contributes nothing when the
    /// target is not already in flight.
    #[test]
    fn a_clean_chain_spawns_as_today_and_declines_on_the_ordinary_no_producer_reason() {
        let root = temp_project_root("chain_clean_spawns");
        let target = root.join("a.py");
        fs::write(&target, b"# placeholder\n").expect("write target");
        let artifact_path = cache_artifact_path(target.to_str().unwrap());

        let result = export_foreign_ts_artifact(target.to_str().unwrap(), &artifact_path, "");
        let sentence = result.expect_err("no producer resolves in this empty temp root");
        assert!(!sentence.contains("recurses back"), "sentence = {sentence:?}");
        assert!(sentence.contains("no"), "sentence = {sentence:?}");
        assert!(sentence.contains(FOREIGN_PRODUCER_NAME), "sentence = {sentence:?}");

        fs::remove_dir_all(&root).ok();
    }
}
