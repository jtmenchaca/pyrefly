//! The crossing judgment core: Fired/Override outcomes over the
//! outbound leg, NaN freedom, unbounded returns, missing artifacts,
//! and the disk-backed integration that exercises the real sibling
//! reader end to end.

use super::*;

/// FIRED CROSSING: unchanged behavior — `stdout_override` is never
/// populated on a `Fired` outcome at all (the field lives only on
/// `Override`; a refuted outbound leg answers `ForeignEdgeOutcome::
/// Fired` per `check_outbound_leg`'s own fit refutation, with no
/// `stdout_override` field to check), and the `Fired.consumer` pin
/// this module already keeps (`a_too_wide_outbound_argument_fires`
/// and its siblings) is untouched by this addition.
#[test]
fn a_fired_crossing_carries_no_stdout_override_field_at_all() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    // the entry admits -2.0 .. 2.0; this argument's own element set is
    // the full ray, well outside it — the SAME too-wide construction
    // `a_too_wide_outbound_argument_fires` uses.
    let too_wide = known_set(
        make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let environment = env_with(&[("boosted", too_wide)]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Fired { message, .. } => {
            assert!(message.contains("audioLevel"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire — an unbounded float list must not fit [-2, 2]"),
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
    }
}

/// The corner-rule fix (2026-08-22): a return set admitting +Infinity
/// (an unbounded `atLeast` with no upper ray) binds as a DETERMINED
/// `KindUnion` of the claimed set's own finite portion (clipped to
/// `[0.0, f64::MAX]`) and the null case — never a decline. The target's
/// own `JSON.stringify(Infinity)` answers the bare token `null` on
/// this leg, so the crossed value's honest set is exactly that union,
/// determined rather than trusted-or-refused.
#[test]
fn an_unbounded_return_binds_the_finite_portion_union_null() {
    register_fixture_artifact("./audio_level.ts", audio_level_unbounded_return_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::KindUnion, "{value:?}");
            assert_eq!(value.arms.len(), 2, "{value:?}");
            let number_arm = value.arms.iter().find(|arm| arm.kind == Kind::Set).expect("a Set arm");
            assert_eq!(number_arm.kind_tag, Some(PrimitiveKind::Float));
            assert!(
                foreign_scalar_subset(&kernel, &number_arm.set, &make_refined_set(vec![at_least(0.0), at_most(f64::MAX)]))
                    == Some(true),
                "{number_arm:?}"
            );
            let null_arm = value.arms.iter().find(|arm| arm.kind == Kind::Null).expect("a Null arm");
            assert_eq!(null_arm.kind, Kind::Null);
        }
        ForeignEdgeOutcome::Decline { message, .. } => {
            panic!("wanted a determined finite-portion/null union, got a decline: {message}")
        }
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a determined finite-portion/null union, got a fire: {message}")
        }
    }
}

#[test]
fn a_missing_capture_output_keyword_declines() {
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized as subprocess.run") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("capture_output"), "{message}");
        }
        _ => panic!("wanted a decline naming the missing capture_output keyword"),
    }
}

#[test]
fn a_missing_artifact_records_the_export_command_hint() {
    // NOT registered — the fixture stub answers a "no artifact" error,
    // mirroring a missing on-disk cache entry.
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./nowhere.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("./nowhere.ts"), "{message}");
        }
        _ => panic!("wanted a decline naming the missing artifact"),
    }
}

#[test]
fn a_too_wide_outbound_argument_fires() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    // the entry admits -2.0 .. 2.0; this argument's own element set is
    // the full ray, well outside it
    let too_wide = known_set(
        make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let environment = env_with(&[("boosted", too_wide)]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Fired { message, .. } => {
            assert!(message.contains("audioLevel"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire, got an override"),
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
    }
}

#[test]
fn a_possibly_nan_payload_fires_before_the_crossing_fit_is_asked() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let nan_scalar = possibly_nan(known_values(vec![0.0], PrimitiveKind::Float, TrustProved));
    let environment = env_with(&[("boosted", nan_scalar)]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Fired { message, .. } => {
            assert!(message.contains("NaN"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a NaN-freedom fire, got an override"),
        ForeignEdgeOutcome::Decline { message, .. } => {
            panic!("wanted a NaN-freedom fire, got a decline: {message}")
        }
    }
}

/* ── disk-backed integration: a real artifact, read for real ────── */
//
// These exercise the sibling's own `read_foreign_ts_artifact` against
// a hand-built artifact JSON on disk (mirroring
// `foreign_edge_artifact.rs`'s own `temp_project_root`/`well_formed_
// artifact` test idiom) — a genuine end-to-end read, not this
// module's in-process fixture stub. The fact read back is then
// registered into the fixture stub under the SAME target path this
// recognizer resolves to, so `foreign_edge_at`'s own recognition and
// premise logic runs unchanged over a fact that really came off disk.

use refined_kernel::wire_format::wire_set;
use serde_json::json;

use crate::foreign_edge_artifact;

/// A fresh temp directory marked as a project root with `.git`, so
/// `cache_artifact_path`/`project_root_of` resolve exactly this
/// directory.
fn temp_project_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "refinedpy_foreign_edge_test_{label}_{}_{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::create_dir_all(&root).expect("create temp project root");
    fs::create_dir_all(root.join(".git")).expect("mark the temp root as a project root");
    root
}

/// A well-formed `audioLevel` artifact JSON, with the real sha256 of
/// `source` as its target contentHash — the exact RULED cases schema
/// `foreign_edge_artifact.rs`'s own module doc spells, no version
/// field at all.
fn well_formed_audio_level_artifact(source: &[u8]) -> serde_json::Value {
    let element = make_refined_set(vec![at_least(-2.0), at_most(2.0)]);
    let return_set = make_refined_set(vec![integer(), at_least(0.0), at_most(1.0)]);
    json!({
        "refined": {"kind": "fact-artifact"},
        "target": {"file": "audio_level.ts", "contentHash": format!("sha256:{}", crate::fact_export::sha256_hex(source))},
        "language": "typescript",
        "runtime": {"band": "es2023+"},
        "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "audioLevel"},
        "functions": {
            "audioLevel": {
                "entry": [{"name": "boosted", "sequence": {"element": {"cases": [{"sort": "number", "set": wire_set(&element)}]}, "lengthAtLeast": 1}}],
                "return": {"cases": [{"sort": "number", "set": wire_set(&return_set)}], "stdoutPure": true},
                "provenance": {"line": 30, "said": "audioLevel's own kernel summary"},
            }
        }
    })
}

/// A hand-built artifact really on disk, read through the sibling's
/// own `read_foreign_ts_artifact`, recognizes end to end and binds
/// the proved [0, 1] return.
#[test]
fn a_disk_backed_artifact_reads_through_the_sibling_reader_and_binds() {
    let Some(kernel) = loaded_kernel() else { return };
    let root = temp_project_root("proved");
    let target = root.join("audio_level.ts");
    fs::write(&target, b"export function audioLevel(boosted: number[]): number { return 0; }\n")
        .expect("write target");
    let source = fs::read(&target).expect("read target back");
    let artifact_path = foreign_edge_artifact::cache_artifact_path(target.to_str().unwrap());
    fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
    fs::write(&artifact_path, well_formed_audio_level_artifact(&source).to_string()).expect("write artifact");

    let target_path = target.to_str().unwrap().to_owned();
    let real_artifact =
        foreign_edge_artifact::read_foreign_ts_artifact(&target_path).expect("the disk artifact reads back");
    assert_eq!(real_artifact.called.name, "audioLevel");
    assert!(real_artifact.called.stdout_pure);
    register_fixture_artifact(&target_path, real_artifact);

    let source_body = format!(
        "def audio_level_via_ts(boosted):\n    result = subprocess.run(\n        [\"node\", {target_path:?}],\n        input=json.dumps(boosted),\n        capture_output=True,\n        text=True,\n    )\n    return json.loads(result.stdout)\n"
    );
    let body = def_body(&source_body);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }

    fs::remove_dir_all(&root).ok();
}

/// No artifact on disk: the sibling reader's own missing-artifact
/// sentence names the export command — pinned here so this module's
/// own decline path (which just relays whatever the reader answers)
/// is exercised against the REAL sentence text, not a hand-written
/// stand-in.
#[test]
fn a_missing_disk_artifact_names_the_export_command() {
    let root = temp_project_root("missing");
    let target = root.join("audio_level.ts");
    fs::write(&target, b"export function audioLevel(boosted: number[]): number { return 0; }\n")
        .expect("write target");
    let target_path = target.to_str().unwrap().to_owned();

    let sentence = foreign_edge_artifact::read_foreign_ts_artifact(&target_path)
        .expect_err("no artifact exists and no producer can write one in this temp root");
    assert!(sentence.contains("-export-fact"), "{sentence}");

    fs::remove_dir_all(&root).ok();
}

/// This module's own decline (line ~275, production code, `#[cfg(not(test))]`
/// path) wraps whatever the sibling reader answers with "the target …
/// states no fact for this edge — {reason}". The sibling's own
/// missing-artifact sentence must NOT restate that same claim itself,
/// or the composed sentence carries the phrase twice. Composed here
/// exactly as the production call site composes it (tests exercise
/// the module's fixture stub instead of the sibling reader at
/// `foreign_edge_at`'s own call site, so the composition is repeated
/// here rather than observed through it) against the REAL sibling
/// sentence, so a regression in either side's wording is caught.
#[test]
fn a_missing_disk_artifact_states_no_fact_exactly_once() {
    let root = temp_project_root("missing_once");
    let target = root.join("audio_level.ts");
    fs::write(&target, b"export function audioLevel(boosted: number[]): number { return 0; }\n")
        .expect("write target");
    let target_path = target.to_str().unwrap().to_owned();

    let reason = foreign_edge_artifact::read_foreign_ts_artifact(&target_path)
        .expect_err("no artifact exists and no producer can write one in this temp root");
    let message = "the target ".to_owned() + &target_path + " states no fact for this edge — " + &reason;

    assert_eq!(
        message.matches("states no fact for this edge").count(),
        1,
        "the prefix must appear exactly once: {message}"
    );
    assert!(message.contains("-export-fact"), "{message}");

    fs::remove_dir_all(&root).ok();
}
