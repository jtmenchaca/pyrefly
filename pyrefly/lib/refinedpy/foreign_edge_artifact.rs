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
//!    "runtime": {"band": "node-23+"},
//!    "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "<fn>"}
//!      | {"kind": "argv-json", "argIndex": n, "stdout": "json", "calls": "<fn>"},
//!    "functions": {"<name>": {
//!      "entry": [{"name", "sequence": {"element": <set>, "lengthAtLeast": n}}
//!               |{"name", "set": <set>}],
//!      "return": {"set": <set>, "stdoutPure": bool},
//!      "provenance": {"line": n, "said": "..."}}}}
//!
//! `surface.kind` names which carrier the JSON transport model rides
//! on — a pipe (`stdin-json`) or one argv element (`argv-json`,
//! `argIndex` naming which one: the node convention makes the third
//! argv element `process.argv[2]`). Both apply the identical transport
//! model to the payload (JSON text, the same round-trip premise); only
//! the carrier differs, so `argv-json` carries no `stdin` field at all.
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

/// The one envelope this consumer admits (schema-v2.md). `language` is
/// checked alongside `(kind, version)` — the kind is shared across every
/// producer language, so the language field is what routes to the right
/// runtime-band pins. A different triple is a decline, never a
/// best-effort read: the fields' meanings are what the version pins.
const FOREIGN_ARTIFACT_KIND: &str = "fact-artifact";
const FOREIGN_ARTIFACT_VERSION: i64 = 2;
const FOREIGN_ARTIFACT_LANGUAGE: &str = "typescript";

/// The runtime band this checker's TypeScript pins commit to.
///
/// PROVISIONAL: no `js.*` naming ruling has landed yet (reverse-pair.md
/// item 2); "node-23+" is a placeholder until that ruling names the
/// real band string. Changing it is a one-line edit once the ruling
/// lands.
const FOREIGN_RUNTIME_BAND: &str = "node-23+";

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

/// One parameter position the target states: either a SEQUENCE (an
/// element set plus the length floor the body relies on, carried as
/// `(element, lengthAtLeast)`) or a plain SCALAR set.
#[derive(Debug, Clone)]
pub struct ForeignTsEntry {
    pub name: String,
    /// `Some` for a sequence position — the element set and the
    /// declaration's own length floor.
    pub sequence: Option<(RefinedSet, i64)>,
    /// `Some` for a scalar position — the position's own set.
    pub scalar: Option<RefinedSet>,
}

/// One target function's whole exported fact.
#[derive(Debug, Clone)]
pub struct ForeignTsFunctionFact {
    pub name: String,
    pub entry: Vec<ForeignTsEntry>,
    pub return_set: RefinedSet,
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

/// Which carrier the JSON transport model rides on — the two `surface
/// .kind` tags this reader admits. Both apply the SAME transport model
/// (the value crosses as JSON text; `stdoutPure` and the outbound-leg
/// fit checks apply identically to either): only the carrier differs,
/// a pipe versus one argv element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignSurface {
    /// `{"kind": "stdin-json", "stdin": "json", "stdout": "json"}` — the
    /// payload rides on the process's stdin pipe.
    StdinJson,
    /// `{"kind": "argv-json", "argIndex": n, "stdout": "json"}` — the
    /// payload is `JSON.parse`'d from `process.argv[argIndex]`; there is
    /// no `stdin` field at all (the two carriers are mutually exclusive
    /// by construction, never a joint claim).
    ArgvJson { arg_index: i64 },
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
fn read_foreign_ts_artifact_uncached(
    target_path: &str,
    artifact_path: &Path,
) -> (Option<ForeignTsArtifact>, String) {
    let (artifact, sentence) = read_and_verify_foreign_ts_artifact(target_path, artifact_path);
    if sentence.is_empty() {
        return (artifact, String::new());
    }
    if let Err(export_sentence) = export_foreign_ts_artifact(target_path, artifact_path) {
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

/// Runs the resolved producer into the cache entry, answering `Ok(())`
/// on success and `Err` naming what stopped it.
fn export_foreign_ts_artifact(target_path: &str, artifact_path: &Path) -> Result<(), String> {
    let Some(producer) = resolve_foreign_producer(target_path) else {
        return Err(format!(
            "no {FOREIGN_PRODUCER_NAME} under the project root and none on PATH"
        ));
    };
    if let Some(parent) = artifact_path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| format!("the cache directory could not be created: {err}"))?;
    }
    let output = Command::new(&producer)
        .arg("-export-fact")
        .arg(target_path)
        .arg("-o")
        .arg(artifact_path)
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
                    "the TypeScript target {target_path} states no fact for this edge — there is no \
                     {artifact_path_words}; write it with `{FOREIGN_EXPORT_COMMAND} {target_path}`"
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

/// Reads the `refined` envelope and checks the `(kind, version,
/// language)` triple. Any triple outside the one admitted form is a
/// decline naming the triple it saw and the one form this reader
/// accepts.
fn check_artifact_envelope(parsed: &Value, artifact_path_words: &str) -> Result<(), String> {
    let Some(envelope) = parsed.get("refined").and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} carries no \"refined\" envelope, so nothing identifies it as a fact artifact"
        ));
    };
    let kind = envelope.get("kind").and_then(Value::as_str).unwrap_or("");
    let version = envelope.get("version").and_then(Value::as_i64);
    let language = parsed.get("language").and_then(Value::as_str).unwrap_or("");

    if kind == FOREIGN_ARTIFACT_KIND && version == Some(FOREIGN_ARTIFACT_VERSION) {
        if language != FOREIGN_ARTIFACT_LANGUAGE {
            return Err(format!(
                "{artifact_path_words} states (kind \"{kind}\", version {}, language {}), and this edge reads \
                 language \"{FOREIGN_ARTIFACT_LANGUAGE}\" for that pair — the language field is what selects \
                 the runtime-band pins",
                FOREIGN_ARTIFACT_VERSION,
                quoted_or_none(language)
            ));
        }
        return Ok(());
    }

    let stated_version = version.map(|v| v.to_string()).unwrap_or_else(|| "nothing".to_owned());
    Err(format!(
        "{artifact_path_words} states (kind \"{kind}\", version {stated_version}), and this edge reads only \
         (kind \"{FOREIGN_ARTIFACT_KIND}\", version {FOREIGN_ARTIFACT_VERSION}, language \
         \"{FOREIGN_ARTIFACT_LANGUAGE}\") — the field meanings are what the version pins"
    ))
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

/// Reads the `surface` object: the wire is JSON, carried either on
/// stdin or on one argv element, and one named function is what the
/// entry point calls. The edge's whole claim is about THAT function —
/// a target whose surface reads a different encoding, or calls nothing
/// this artifact names, transports something the JSON model does not
/// describe. `surface` carries a tagged `kind`; only `"stdin-json"` and
/// `"argv-json"` have a transport model here (schema-v2.md: the other
/// sketched kinds have no reader yet).
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
        other => {
            return Err(format!(
                "{artifact_path_words} states a surface of kind {}, and this edge reads only \"stdin-json\" \
                 or \"argv-json\"",
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
        return Err(format!("{artifact_path_words} carries no functions, so it states no fact about {name}"));
    };
    let Some(row) = functions.get(name).and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} names {name} as the harness's called function and then states no fact for it"
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
        return Err(format!("states no return fact for {name}, so nothing crosses back from this call"));
    };
    let Some(raw_set) = returned.get("set") else {
        return Err(format!("states a return for {name} with no set, so the value crossing back is unbounded"));
    };
    let stdout_pure = returned.get("stdoutPure").and_then(Value::as_bool).unwrap_or(false);
    let (provenance_line, provenance_said) = artifact_provenance_of(row);
    Ok(ForeignTsFunctionFact {
        name: name.to_owned(),
        entry: entries,
        return_set: decode_wire_set(raw_set),
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
            let Some(raw_element) = sequence.get("element") else {
                return Err(format!("states a sequence entry {entry_name} for {name} with no element set"));
            };
            let length_at_least = sequence.get("lengthAtLeast").and_then(Value::as_i64).unwrap_or(0);
            entries.push(ForeignTsEntry {
                name: entry_name,
                sequence: Some((decode_wire_set(raw_element), length_at_least)),
                scalar: None,
            });
            continue;
        }
        let Some(raw_set) = entry_row.get("set") else {
            return Err(format!("states an entry position {entry_name} for {name} that is neither a sequence nor a set"));
        };
        entries.push(ForeignTsEntry {
            name: entry_name,
            sequence: None,
            scalar: Some(decode_wire_set(raw_set)),
        });
    }
    Ok(entries)
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
            "refined": {"kind": FOREIGN_ARTIFACT_KIND, "version": FOREIGN_ARTIFACT_VERSION},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": called},
            "functions": {
                called: {
                    "entry": [{"name": "x", "set": refined_kernel::wire_format::wire_set(&scalar)}],
                    "return": {"set": refined_kernel::wire_format::wire_set(&scalar), "stdoutPure": true},
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
            "refined": {"kind": FOREIGN_ARTIFACT_KIND, "version": FOREIGN_ARTIFACT_VERSION},
            "target": {"file": "target.ts", "contentHash": format!("sha256:{}", sha256_hex(source))},
            "language": FOREIGN_ARTIFACT_LANGUAGE,
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "surface": {"kind": "argv-json", "argIndex": arg_index, "stdout": "json", "calls": called},
            "functions": {
                called: {
                    "entry": [{"name": "x", "set": refined_kernel::wire_format::wire_set(&scalar)}],
                    "return": {"set": refined_kernel::wire_format::wire_set(&scalar), "stdoutPure": true},
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
        artifact["functions"]["f"]["return"]["set"] = json!({"forms": [{"form": "not-a-real-form"}]});
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

    /// A `(kind, version)` pair outside the one admitted form declines,
    /// naming the triple it saw and the accepted form — never a
    /// best-effort read of a schema this reader does not know.
    #[test]
    fn an_unknown_kind_version_triple_declines_naming_the_accepted_form() {
        let root = temp_project_root("unknown_triple");
        let target = root.join("target.ts");
        fs::write(&target, b"export function f(x: number): number { return x; }\n").expect("write target");
        let source = fs::read(&target).expect("read target back");
        let mut artifact = well_formed_artifact(&source, "f");
        artifact["refined"] = json!({"kind": "fact-artifact", "version": 3});
        let artifact_path = cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, artifact.to_string()).expect("write artifact");

        let read = read_foreign_ts_artifact(target.to_str().unwrap());
        let sentence = read.expect_err("an unknown (kind, version) pair must decline");
        assert!(sentence.contains("fact-artifact"), "sentence = {sentence:?}");
        assert!(sentence.contains('3'), "sentence = {sentence:?}");
        assert!(sentence.contains(FOREIGN_ARTIFACT_KIND), "sentence = {sentence:?}");

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
}
