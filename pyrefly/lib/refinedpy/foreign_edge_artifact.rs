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
//! The schema is frozen (docs/one-checker/reverse-pair.md, "Shared
//! decisions"); the TypeScript side's exporter writes it and this side
//! consumes it verbatim:
//!
//!   {"refined": {"kind": "typescript-fact-artifact", "version": 1},
//!    "target": {"file", "contentHash": "sha256:<hex>"},
//!    "runtime": {"band": "node-23+"},
//!    "harness": {"stdin": "json", "stdout": "json", "calls": "<fn>"},
//!    "functions": {"<name>": {
//!      "entry": [{"name", "sequence": {"element": <set>, "lengthAtLeast": n}}
//!               |{"name", "set": <set>}],
//!      "return": {"set": <set>, "stdoutPure": bool},
//!      "provenance": {"line": n, "said": "..."}}}}
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

/// The envelope this consumer admits. A different kind or version is a
/// decline, never a best-effort read: the fields' meanings are what the
/// version pins.
const FOREIGN_ARTIFACT_KIND: &str = "typescript-fact-artifact";
const FOREIGN_ARTIFACT_VERSION: i64 = 1;

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

/// The artifact as consumed: the runtime band it commits to, and the
/// ONE function the harness calls, already selected.
#[derive(Debug, Clone)]
pub struct ForeignTsArtifact {
    /// The artifact file itself, for the diagnostics.
    pub path: PathBuf,
    /// The `.ts` path the artifact is about, as resolved here (not as
    /// the artifact spells it — the hash is what ties them).
    pub target_file: String,
    pub runtime_band: String,
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

/// The nearest ancestor of `target`'s directory holding `.git` — the
/// project root (the target's own directory when none is found).
pub fn project_root_of(target: &Path) -> PathBuf {
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
    let called_name = match harness_called_name(&parsed, &artifact_path_words) {
        Ok(name) => name,
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
            called: fact,
        }),
        String::new(),
    )
}

/// Reads the `refined` envelope: the kind this consumer knows and the
/// version whose field meanings it was written against.
fn check_artifact_envelope(parsed: &Value, artifact_path_words: &str) -> Result<(), String> {
    let Some(envelope) = parsed.get("refined").and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} carries no \"refined\" envelope, so nothing identifies it as a fact artifact"
        ));
    };
    let kind = envelope.get("kind").and_then(Value::as_str).unwrap_or("");
    if kind != FOREIGN_ARTIFACT_KIND {
        return Err(format!(
            "{artifact_path_words} states the kind \"{kind}\", and this edge consumes \"{FOREIGN_ARTIFACT_KIND}\" \
             — nothing else"
        ));
    }
    let version = envelope.get("version").and_then(Value::as_i64);
    if version != Some(FOREIGN_ARTIFACT_VERSION) {
        let stated = version.map(|v| v.to_string()).unwrap_or_else(|| "nothing".to_owned());
        return Err(format!(
            "{artifact_path_words} states artifact version {stated}, and this edge reads version \
             {FOREIGN_ARTIFACT_VERSION} — the field meanings are what the version pins"
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

/// Reads the stdio harness: the wire is JSON in both directions, and one
/// named function is what the entry point calls. The edge's whole claim
/// is about THAT function — a target whose harness reads a different
/// encoding, or calls nothing this artifact names, transports something
/// the JSON model does not describe.
fn harness_called_name(parsed: &Value, artifact_path_words: &str) -> Result<String, String> {
    let Some(harness) = parsed.get("harness").and_then(Value::as_object) else {
        return Err(format!(
            "{artifact_path_words} describes no harness, so nothing says what the target does with stdin and \
             stdout — the JSON transport model has nothing to apply to"
        ));
    };
    let stdin = harness.get("stdin").and_then(Value::as_str).unwrap_or("");
    let stdout = harness.get("stdout").and_then(Value::as_str).unwrap_or("");
    if stdin != "json" || stdout != "json" {
        return Err(format!(
            "{artifact_path_words} states a harness reading {} on stdin and writing {} on stdout, and this edge \
             applies the JSON transport model to both legs",
            quoted_or_none(stdin),
            quoted_or_none(stdout)
        ));
    }
    let called = harness.get("calls").and_then(Value::as_str).unwrap_or("");
    if called.is_empty() {
        return Err(format!(
            "{artifact_path_words} states no harness.calls function, so nothing names the code that runs when \
             this call executes"
        ));
    }
    Ok(called.to_owned())
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
            "runtime": {"band": FOREIGN_RUNTIME_BAND},
            "harness": {"stdin": "json", "stdout": "json", "calls": called},
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
}
