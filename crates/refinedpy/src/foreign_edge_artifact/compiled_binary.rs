//! A compiled binary's fact file, sibling-discovered at
//! `<binary_path>.facts.json` — `cpp_level` reads `cpp_level.facts.json`
//! beside it. This is a SEPARATE convention from the TypeScript reader's
//! project-cache path (`cache.rs::cache_artifact_path`'s `.refined/cache/<relpath>
//! /<name>.refined.json`): a compiled binary has no checkable SOURCE this
//! checker reads (the `.cpp` a human compiled it from is not code this
//! checker ever opens) and no PRODUCER binary that regenerates the fact
//! (`producer.rs::resolve_foreign_producer`'s own `refinedts-check-bin`
//! only exports TypeScript facts) — the cache-path/target-integrity-hash/
//! producer-freshness/auto-export machinery in `typescript_read.rs` exists
//! to serve exactly those two premises, neither of which a compiled binary
//! carries. What a compiled binary's fact CAN state is unchanged: the
//! same RULED cases schema (`docs/one-checker/fact-schema.md`'s "The
//! current envelope (ruled)"), read through the SAME field readers
//! `typescript_read.rs`/`cases.rs` already apply to a TypeScript artifact
//! (`harness_surface_of`, `function_fact_of` — both already generic over
//! the parsed envelope, with no TypeScript-specific reading inside
//! either), with `language` stating `"cpp"` in place of `"typescript"`
//! and `runtime.band` stating this checker's own compiled-C++ pin.

use serde_json::Value;

use super::cases::function_fact_of;
use super::types::ForeignTsArtifact;
use super::typescript_read::harness_surface_of;
use super::typescript_read::nested_string;
use super::typescript_read::FOREIGN_ARTIFACT_KIND;

/// The runtime band a compiled-binary fact commits to — the ISO C++17
/// standard the triangle's own producer states it compiles against
/// (`examples/cross-language/audio-level-triangle/targets/cpp_level.cpp`'s
/// own header comment: `c++ -std=c++17 -o cpp_level cpp_level.cpp`).
/// Mirrors `typescript_read.rs::FOREIGN_RUNTIME_BAND`'s own role for
/// TypeScript: the SPEC LEVEL the target's checked code runs against,
/// not one compiler binary.
const COMPILED_BINARY_RUNTIME_BAND: &str = "c++17";

/// The compiled-binary envelope's own `language` tag.
const COMPILED_BINARY_ARTIFACT_LANGUAGE: &str = "cpp";

/// What a compiled binary's sibling fact file is named: `<binary_path>
/// .facts.json`, never the TypeScript reader's `.refined.json` cache
/// suffix — a compiled binary's fact sits NEXT TO the binary itself
/// (hand- or tool-authored, committed alongside it), not in a derived
/// project cache a producer regenerates.
const COMPILED_BINARY_FACT_SUFFIX: &str = ".facts.json";

/// The sibling fact-file path for a compiled binary: `<binary_path>` with
/// `COMPILED_BINARY_FACT_SUFFIX` appended — `./targets/cpp_level` reads
/// `./targets/cpp_level.facts.json`.
pub fn compiled_binary_fact_path(binary_path: &str) -> std::path::PathBuf {
    let mut path = std::path::PathBuf::from(binary_path);
    let file_name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    path.set_file_name(format!("{file_name}{COMPILED_BINARY_FACT_SUFFIX}"));
    path
}

/// Reads a compiled binary's sibling fact file — the three-rung ladder
/// this construct owns: no fact file at that path (the sentence
/// `diagnostic_sentences::compiled_binary_no_fact` already states, held
/// unchanged by this function's caller for that rung — see
/// `read_compiled_binary_fact`'s own doc), a fact file that exists but
/// fails to parse (this function's `Err`, naming the unreadable file), or
/// a fact file that parses and serves (`Ok`).
///
/// No target-integrity hash, no producer-freshness check, no auto-export
/// attempt: none of the three apply to a compiled binary (see this
/// module's doc comment above).
pub fn read_compiled_binary_fact(binary_path: &str) -> Result<ForeignTsArtifact, String> {
    let fact_path = compiled_binary_fact_path(binary_path);
    let fact_path_words = fact_path.display().to_string();
    let raw = std::fs::read(&fact_path).map_err(|_| {
        format!("there is no {fact_path_words} beside {binary_path}, so the checker has no fact for this compiled binary")
    })?;
    let parsed: Value = serde_json::from_slice(&raw).map_err(|_| {
        format!("{fact_path_words} is not readable JSON, so the target states nothing this edge can use")
    })?;
    check_compiled_binary_envelope(&parsed, &fact_path_words)?;
    let band = nested_string(&parsed, "runtime", "band").ok_or_else(|| {
        format!(
            "{fact_path_words} names no runtime band, and the edge's claim inherits whichever band the \
             target's pins commit to"
        )
    })?;
    if band != COMPILED_BINARY_RUNTIME_BAND {
        return Err(format!(
            "{fact_path_words} commits to the runtime band {band}, and this checker's compiled-binary pins \
             commit to {COMPILED_BINARY_RUNTIME_BAND} — the edge cannot inherit semantics it has not transcribed"
        ));
    }
    let (surface, called_name) = harness_surface_of(&parsed, &fact_path_words)?;
    let fact = function_fact_of(&parsed, &called_name, &fact_path_words)?;
    Ok(ForeignTsArtifact {
        path: fact_path,
        target_file: binary_path.to_owned(),
        runtime_band: band,
        surface,
        called: fact,
    })
}

/// The compiled-binary envelope check — `typescript_read.rs::check_artifact_envelope`'s
/// own twin, admitting `language: "cpp"` in place of `"typescript"` and
/// otherwise identical: no version field ever, the same `kind`.
fn check_compiled_binary_envelope(parsed: &Value, fact_path_words: &str) -> Result<(), String> {
    let Some(envelope) = parsed.get("refined").and_then(Value::as_object) else {
        return Err(format!(
            "{fact_path_words} carries no \"refined\" envelope, so nothing identifies it as a fact artifact"
        ));
    };
    let kind = envelope.get("kind").and_then(Value::as_str).unwrap_or("");
    let language = parsed.get("language").and_then(Value::as_str).unwrap_or("");

    if envelope.contains_key("version") {
        return Err(format!(
            "{fact_path_words} states a \"version\" field in its \"refined\" envelope, and this edge reads \
             only the current cases schema, which carries no version field at all"
        ));
    }
    if kind != FOREIGN_ARTIFACT_KIND {
        return Err(format!(
            "{fact_path_words} states (kind \"{kind}\"), and this edge reads only (kind \
             \"{FOREIGN_ARTIFACT_KIND}\") — the field meanings are what the kind pins"
        ));
    }
    if language != COMPILED_BINARY_ARTIFACT_LANGUAGE {
        return Err(format!(
            "{fact_path_words} states (kind \"{kind}\", language {}), and this edge reads language \
             \"{COMPILED_BINARY_ARTIFACT_LANGUAGE}\" for a compiled binary's fact — the language field is what \
             selects the runtime-band pins",
            super::typescript_read::quoted_or_none(language)
        ));
    }
    Ok(())
}
