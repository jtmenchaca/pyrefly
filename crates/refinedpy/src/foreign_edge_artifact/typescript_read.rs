//! Reading and verifying a TypeScript target's exported artifact: the
//! envelope check, target-integrity hash, producer-freshness, and the
//! harness surface — every premise the reader owns before a function's
//! own fact (`cases.rs`) is decoded.

use std::path::Path;

use serde_json::Value;

use crate::fact_export::sha256_hex;

use super::cases::function_fact_of;
use super::producer::resolve_foreign_producer;
use super::producer::FOREIGN_EXPORT_COMMAND;
use super::types::ForeignSurface;
use super::types::ForeignTsArtifact;

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
pub(super) const FOREIGN_ARTIFACT_KIND: &str = "fact-artifact";
pub(super) const FOREIGN_ARTIFACT_LANGUAGE: &str = "typescript";

/// The runtime band this checker's TypeScript pins commit to.
///
/// One JS-family band claiming ECMA-262-level behaviour (ruling,
/// 2026-08-21): every premise the edge discharges is an ECMA-262 claim,
/// so any recognized JS runner (node, deno, bun, npx tsx) satisfies this
/// band premise once the artifact declares it — the band names the
/// SPEC LEVEL the target's checked code runs against, not one runtime
/// binary.
pub(super) const FOREIGN_RUNTIME_BAND: &str = "es2023+";

/// The read itself — every premise checked against the given cache
/// entry.
pub(super) fn read_and_verify_foreign_ts_artifact(
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
    // that would regenerate it is stale — the producer may have changed
    // what it exports since this artifact was written, and the content
    // hash alone cannot see that.
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
pub(super) fn harness_surface_of(parsed: &Value, artifact_path_words: &str) -> Result<(ForeignSurface, String), String> {
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

/// Reads `parsed[outer][inner]` as a string.
pub(super) fn nested_string(parsed: &Value, outer: &str, inner: &str) -> Option<String> {
    parsed.get(outer)?.get(inner)?.as_str().map(str::to_owned)
}

/// Spells a harness channel for a message: the word it states, or
/// "nothing" where the field is absent.
pub(super) fn quoted_or_none(word: &str) -> String {
    if word.is_empty() {
        "nothing".to_owned()
    } else {
        format!("\"{word}\"")
    }
}
