//! Argv/CLI entry shapes: the argv-json payload and the temp-file
//! carrier (including channel-mismatch declines between stdin/argv/
//! file carriers, the f-string-wrapped payload, and the nested
//! temp-directory case). Runner words, the compiled-binary row, and
//! os.system live in the `runner` sibling.

use super::*;

/// A fitting argv-json call against a matching argv-json target
/// recognizes and binds the proved return — silent (`Override`).
#[test]
fn a_fitting_argv_json_call_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// An unfitting argv-json payload fires the same RTS7001 the stdin
/// leg fires — the outbound-leg fit checks are shared, unchanged by
/// the carrier.
#[test]
fn an_unfitting_argv_json_call_fires() {
    register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
    let too_wide = known_set(
        make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let environment = env_with(&[("boosted", too_wide)]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
        ForeignEdgeOutcome::Fired { message, .. } => {
            assert!(message.contains("audioLevel"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire, got an override"),
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
    }
}

/// An argv-json call against a `stdin-json` target declines with the
/// channel-mismatch sentence: the call names a real reference and
/// the target states a real fact, but the two carriers do not meet.
#[test]
fn an_argv_json_call_at_a_stdin_json_target_declines_with_the_mismatch_sentence() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("argv element"), "{message}");
            assert!(message.contains("stdin"), "{message}");
            assert!(message.contains("channels do not meet"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a channel-mismatch decline, got a fire: {message}")
        }
    }
}

/// A stdin-json call (`input=json.dumps(...)`, plain two-element
/// argv) against an `argv-json` target declines with the mismatch
/// sentence, symmetrically.
#[test]
fn a_stdin_json_call_at_an_argv_json_target_declines_with_the_mismatch_sentence() {
    register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the stdin shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("stdin"), "{message}");
            assert!(message.contains("argv element"), "{message}");
            assert!(message.contains("channels do not meet"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a channel-mismatch decline, got a fire: {message}")
        }
    }
}

/// `input=json.dumps(...)` alongside an argv-json payload declines
/// naming the double channel — two crossing values are stated and
/// this checker recognizes exactly one transport per call.
#[test]
fn input_keyword_alongside_an_argv_json_payload_declines_the_double_channel() {
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\", json.dumps(boosted)],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("argv element"), "{message}");
            assert!(message.contains("input=json.dumps"), "{message}");
        }
        _ => panic!("wanted a decline naming the double channel"),
    }
}

/* ── the temp-file payload shape ──────────────────────────────── */

/// A fitting temp-file call against a matching `file-json` target
/// recognizes and binds the proved return — silent (`Override`), the
/// same as the stdin-json and argv-json shapes: only the carrier
/// differs.
#[test]
fn a_fitting_temp_file_call_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact(
        "./audio_level.ts",
        ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
    );
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// An out-of-set payload crossing through the temp-file carrier
/// fires the same RTS7001 the stdin and argv-json legs fire — the
/// outbound-leg fit checks are shared, unchanged by the carrier.
#[test]
fn an_out_of_set_temp_file_payload_fires() {
    register_fixture_artifact(
        "./audio_level.ts",
        ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
    );
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
    let too_wide = known_set(
        make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let environment = env_with(&[("boosted", too_wide)]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
        ForeignEdgeOutcome::Fired { message, .. } => {
            assert!(message.contains("audioLevel"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire, got an override"),
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
    }
}

/// A temp-file call against a `stdin-json` target declines with the
/// channel-mismatch sentence: the call names a real reference and
/// the target states a real fact, but the two carriers do not meet.
#[test]
fn a_temp_file_call_at_a_stdin_json_target_declines_with_the_mismatch_sentence() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("temp file"), "{message}");
            assert!(message.contains("stdin"), "{message}");
            assert!(message.contains("channels do not meet"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a channel-mismatch decline, got a fire: {message}")
        }
    }
}

/// A temp-file call against an `argv-json` target declines with the
/// channel-mismatch sentence, symmetrically: the target reads the
/// argv element as the JSON text directly, never as a file path.
#[test]
fn a_temp_file_call_at_an_argv_json_target_declines_with_the_mismatch_sentence() {
    register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("temp file"), "{message}");
            assert!(message.contains("JSON text itself"), "{message}");
            assert!(message.contains("channels do not meet"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a channel-mismatch decline, got a fire: {message}")
        }
    }
}

/// A `stdin-json` call (`input=json.dumps(...)`) against a
/// `file-json` target declines with the mismatch sentence,
/// symmetrically with the temp-file-at-stdin-target row.
#[test]
fn a_stdin_json_call_at_a_file_json_target_declines_with_the_mismatch_sentence() {
    register_fixture_artifact(
        "./audio_level.ts",
        ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
    );
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the stdin shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("stdin"), "{message}");
            assert!(message.contains("file"), "{message}");
            assert!(message.contains("channels do not meet"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a channel-mismatch decline, got a fire: {message}")
        }
    }
}

/// An argv-json call (`json.dumps(...)` directly as the third argv
/// element) against a `file-json` target declines with the mismatch
/// sentence, symmetrically with the temp-file-at-argv-target row.
#[test]
fn an_argv_json_call_at_a_file_json_target_declines_with_the_mismatch_sentence() {
    register_fixture_artifact(
        "./audio_level.ts",
        ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
    );
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("directly as an argv element"), "{message}");
            assert!(message.contains("file path"), "{message}");
            assert!(message.contains("channels do not meet"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a channel-mismatch decline, got a fire: {message}")
        }
    }
}

/// FIX 4: the argv-json payload spelled through an f-string wrapping
/// exactly one interpolation, `f"{json.dumps(boosted)}"`, rather than
/// a bare `json.dumps(...)` call (`level_via_fstring_argv_data`,
/// d-data-legs.py:238). Before this fix, `json_dumps_argument_of`
/// required `Expr::Call` as its very first check; an f-string is
/// `Expr::FString`, which failed that guard immediately, so
/// `argv_json_call_of`'s own payload read answered `None` and the
/// whole argv-json shape declined to match at all — the call fell
/// through to the ordinary two-element stdin shape, which itself
/// declines (a three-element argv is not that shape either), so
/// `foreign_edge_at` answered a decline naming the wrong construct
/// (an unrecognized three-element argv) rather than reading through
/// to the payload. This pins the fix: the trivial single-
/// interpolation f-string wrapper unwraps to its inner
/// `json.dumps(...)` call, and the argv-json shape recognizes and
/// binds the proved return exactly as the bare-call spelling does.
#[test]
fn an_fstring_wrapped_argv_json_payload_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def audio_level_via_fstring_argv(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\", f\"{json.dumps(boosted)}\"],\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None)
        .expect("the f-string-wrapped argv-json shape recognizes")
    {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// REGRESSION PIN: the bare `json.dumps(...)` argv-json spelling
/// still recognizes exactly as before this fix — the f-string
/// wrapper is an ADDITIONAL readable spelling of the same payload
/// position, never a replacement for the direct-call shape
/// `json_dumps_argument_of` already read.
#[test]
fn the_bare_call_argv_json_payload_still_recognizes_after_the_fstring_fix() {
    register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the bare-call argv-json shape recognizes")
    {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// A `temp_path` reassigned between the `with`-block's own dump and
/// the call that reads it back stays undetermined, naming the
/// rebound name — the carrier premise (the bytes dumped are the
/// bytes read) cannot be proved once the name is written again.
#[test]
fn a_reassigned_temp_path_between_dump_and_call_stays_undetermined_naming_it() {
    let source = concat!(
        "def f(boosted):\n",
        "    with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\", delete=False) as handle:\n",
        "        json.dump(boosted, handle)\n",
        "        temp_path = handle.name\n",
        "    temp_path = \"/tmp/other.json\"\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\", temp_path],\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the with-block is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("temp_path"), "{message}");
            assert!(message.contains("written again"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a decline naming the rebind, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => {
            panic!("wanted a decline naming the rebind, got a fire: {message}")
        }
    }
}

/// A with-block whose `NamedTemporaryFile(...)` call is missing
/// `delete=False` declines naming the missing keyword — the file
/// would not survive past the with-block for the call to read.
#[test]
fn a_temp_file_missing_delete_false_declines() {
    let source = concat!(
        "def f(boosted):\n",
        "    with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\") as handle:\n",
        "        json.dump(boosted, handle)\n",
        "        temp_path = handle.name\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\", temp_path],\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the with-block is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("delete=False"), "{message}"),
        _ => panic!("wanted a decline naming the missing delete=False keyword"),
    }
}

/// A shadowed `tempfile` name is not recognized as the module.
#[test]
fn a_shadowed_tempfile_name_is_not_recognized() {
    let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
    let mut environment = env_with(&[("boosted", boosted_sequence_value())]);
    environment.bind("tempfile", known_values(vec![0.0], PrimitiveKind::Integer, TrustProved));
    let Some(kernel) = loaded_kernel() else { return };
    assert!(
        foreign_edge_at(&body, 0, &environment, &kernel, None).is_none(),
        "a locally shadowed tempfile must not be read as the module"
    );
}

/// FIX 3: the exact temp-file unit `TEMP_FILE_FIXTURE_SOURCE` proves,
/// nested one level inside an outer `with tempfile
/// .TemporaryDirectory():` block (`level_via_nested_tempdir`,
/// d-data-legs.py:266). `recognize_temp_file_edge` reads whichever
/// `statements`/`index` it is handed with no assumption about
/// nesting depth — this pins that premise directly: calling
/// `foreign_edge_at` at position 0 of the OUTER with-block's own
/// body (the exact statement list and index `check.rs`'s
/// `walk_with` now offers per statement, after this fix) recognizes
/// and binds the proved return exactly as the top-level case does.
/// Before this fix, `walk_with` walked its own body through a plain
/// per-statement `walk_statement` loop with no call into
/// `foreign_edge_at` at all, so this position was never even
/// offered the recognition this test drives directly.
#[test]
fn a_temp_file_edge_nested_inside_a_temporary_directory_recognizes_and_binds() {
    register_fixture_artifact(
        "./audio_level.ts",
        ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
    );
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    with tempfile.TemporaryDirectory():\n",
        "        with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\", delete=False) as handle:\n",
        "            json.dump(boosted, handle)\n",
        "            temp_path = handle.name\n",
        "        result = subprocess.run(\n",
        "            [\"node\", \"./audio_level.ts\", temp_path],\n",
        "            capture_output=True,\n",
        "            text=True,\n",
        "        )\n",
        "        return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let Stmt::With(outer_with) = &body[0] else {
        panic!("this fixture's own top-level statement must be the outer TemporaryDirectory with-block");
    };
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&outer_with.body, 0, &environment, &kernel, None)
        .expect("the nested temp-file shape recognizes")
    {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}
