//! Parse consumers: how `json.loads`/`json.load` reads the crossing
//! result back — stdout-attribute vs. bare check_output reads, the
//! sole-consumer discipline, and the shape that never reads its
//! result at all.

use super::*;

/// DISCHARGED CROSSING, `ResultRead::StdoutAttribute`
/// (`subprocess.run`'s own `result.stdout` shape): `audio_level_ts_
/// artifact`'s return is entirely number-sorted
/// (`integer, >= 0, <= 1`), so `stdout_override` binds — the `result
/// .stdout` attribute-access node inside `json.loads(result.stdout)`
/// reads as the JSON-number-grammar string set, admitting a text the
/// harness's own serializer actually writes ("0.5\n") and excluding
/// both a non-numeric token ("abc") and the SAME digits without the
/// harness's own trailing newline ("0.5") — the anchored `$` end
/// pins the newline as load-bearing, not optional padding.
#[test]
fn a_discharged_stdout_attribute_crossing_binds_the_serialized_stdout_reading() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes");
    let ForeignEdgeOutcome::Override { stdout_override, .. } = outcome else {
        panic!("wanted an override");
    };
    let (_, stdout_value) = stdout_override.expect("a number-only return binds the intermediate stdout reading");
    assert_eq!(stdout_value.kind, Kind::Set);
    assert!(literal_string_admitted_by(&kernel, "0.5\n", &stdout_value.set), "the grammar must admit a serialized JSON number plus newline");
    assert!(!literal_string_admitted_by(&kernel, "abc", &stdout_value.set), "the grammar must exclude non-numeric text");
    assert!(!literal_string_admitted_by(&kernel, "0.5", &stdout_value.set), "the grammar must exclude the digits without the harness's own trailing newline");
}

/// DISCHARGED CROSSING, `ResultRead::Bare` (`subprocess.check_output`'s
/// own direct-return shape): the same number-only return binds the
/// intermediate reading at the bound name's own node inside `json
/// .loads(result)`, admitting/excluding the identical texts the
/// `StdoutAttribute` pin checks.
#[test]
fn a_discharged_bare_check_output_crossing_binds_the_serialized_stdout_reading() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.check_output(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes");
    let ForeignEdgeOutcome::Override { stdout_override, .. } = outcome else {
        panic!("wanted an override");
    };
    let (_, stdout_value) = stdout_override.expect("a number-only return binds the intermediate stdout reading");
    assert_eq!(stdout_value.kind, Kind::Set);
    assert!(literal_string_admitted_by(&kernel, "0.5\n", &stdout_value.set), "the grammar must admit a serialized JSON number plus newline");
    assert!(!literal_string_admitted_by(&kernel, "abc", &stdout_value.set), "the grammar must exclude non-numeric text");
    assert!(!literal_string_admitted_by(&kernel, "0.5", &stdout_value.set), "the grammar must exclude the digits without the harness's own trailing newline");
}

/// A recognized call whose result NOTHING reads through `json.loads`
/// (`d-data-legs.py`'s own `level_via_raw_stdout` row: the result is
/// read as `float(result.stdout)`, never parsed as JSON) answers plain
/// `None` from `foreign_edge_at` — no override (there is no parse node
/// to attach a fact to) and no decline (the outbound leg already
/// judged cleanly; a body that reads its result some other way owes
/// this route nothing).
#[test]
fn an_unparsed_result_answers_no_outcome_at_all() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def audio_level_via_ts(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return float(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None);
    assert!(outcome.is_none(), "wanted no outcome at all, got {:?}", outcome.map(|o| match o {
        ForeignEdgeOutcome::Override { .. } => "an override",
        ForeignEdgeOutcome::Fired { .. } => "a fire",
        ForeignEdgeOutcome::Decline { .. } => "a decline",
    }));
}

/// The recognized foreign-edge shape's `json.loads(result.stdout)`
/// node is never read through `expressions.rs::
/// json_loads_value_space` — the honest JSON-union built for an
/// OPAQUE operand this file holds no fact about
/// (ISSUES.md, b-runners:124). `foreign_edge_at` builds this
/// `Override` value directly from the target's own kernel-derived
/// return fact (this file's own `foreign_return_value`), entirely
/// separate from `expressions.rs`'s generic `json.loads` handler —
/// `check.rs`'s `Environment::set_evaluated_node` publishes this
/// value at the parse node BEFORE any generic evaluation reaches
/// it, so a recognized target never falls to the union answer this
/// row's own sibling test (`test_json_loads_of_an_opaque_operand_
/// answers_the_full_json_union`, expressions.rs) pins for the
/// UNrecognized case.
#[test]
fn a_recognized_target_never_answers_the_generic_json_union() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes");
    match outcome {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_ne!(value.kind, Kind::KindUnion, "the recognized edge's own fact must win, not the opaque union");
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

#[test]
fn a_result_read_twice_through_json_loads_declines() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    a = json.loads(result.stdout)\n",
        "    b = json.loads(result.stdout)\n",
        "    return a\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("2 times") || message.contains("parsed"), "{message}");
        }
        _ => panic!("wanted a decline naming the sole-consumer violation"),
    }
}

/* ── subprocess.check_output ──────────────────────────────────── */

#[test]
fn check_output_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.check_output(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

#[test]
fn check_output_with_no_text_keyword_declines() {
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.check_output(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "    )\n",
        "    return json.loads(result)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("text=True"), "{message}"),
        _ => panic!("wanted a decline naming the missing text keyword"),
    }
}
