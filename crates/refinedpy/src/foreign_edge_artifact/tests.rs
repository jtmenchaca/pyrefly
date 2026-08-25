use std::fs;
use std::path::PathBuf;

use serde_json::json;
use serde_json::Value;

use super::*;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;

use crate::fact_export::sha256_hex;

use super::producer::export_chain_contains;
use super::producer::export_chain_cycle_sentence;
use super::producer::export_foreign_ts_artifact;
use super::producer::FOREIGN_PRODUCER_NAME;
use super::producer::FOREIGN_PRODUCER_RELATIVE;
use super::types::ForeignCase;
use super::types::ForeignSurface;
use super::typescript_read::FOREIGN_ARTIFACT_KIND;
use super::typescript_read::FOREIGN_ARTIFACT_LANGUAGE;
use super::typescript_read::FOREIGN_RUNTIME_BAND;

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
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../refinedts/edge-premise-fixtures"))
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
