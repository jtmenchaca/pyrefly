//! The runner-word/interpreter argv shapes: deno/bun/npx tsx, the
//! compiled-binary argv row, the const-held literal script path, and
//! os.system's own argv-as-shell-string spelling.

use super::*;

/* ── runner words: deno / bun / npx tsx ──────────────────────────── */

/// A `deno run` call recognizes the reference and, once the
/// artifact's own band check passes (this fixture declares the
/// shared `es2023+` band), proceeds to ordinary premise judging
/// exactly like a `node` call — the runner-word band gap retired
/// with the ruling that the band names an ECMA-262 spec level, not
/// one runtime binary.
#[test]
fn a_deno_run_call_recognizes_the_reference_and_judges_like_node() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"deno\", \"run\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
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

/// A `bun` call recognizes the reference and judges like `node` —
/// same rationale as the `deno run` sibling above.
#[test]
fn a_bun_call_recognizes_the_reference_and_judges_like_node() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"bun\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
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

/// An `npx tsx` call recognizes the reference and judges like `node`
/// — same rationale as the `deno run`/`bun` siblings above.
#[test]
fn an_npx_tsx_call_recognizes_the_reference_and_judges_like_node() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"npx\", \"tsx\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
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
fn a_three_element_argv_with_an_unrecognized_two_word_runner_is_not_this_shape() {
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"yarn\", \"dlx\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    assert!(
        foreign_edge_at(&body, 0, &environment, &kernel, None).is_none(),
        "an unrecognized two-word runner is some other program, nothing owed"
    );
}

/* ── the compiled-binary argv row ─────────────────────────────────── */

/// A single-element, path-shaped argv (`["./targets/cpp_level"]`)
/// recognizes as a compiled-binary invocation and reaches the
/// SIBLING fact-file reader exactly as the triangle's own
/// `level_via_cpp` is designed to once `cpp_level.facts.json` exists
/// beside the binary: a real `<dir>/targets/cpp_level.facts.json`
/// written under a temp `entry_directory` serves the crossing
/// (`Override`), never the in-memory TypeScript stub (which
/// `Runner::CompiledBinary` no longer reaches at all).
#[test]
fn a_bare_binary_argv_recognizes_and_reaches_the_sibling_fact_file() {
    let Some(kernel) = loaded_kernel() else { return };
    let root = temp_binary_dir("well_formed");
    fs::create_dir_all(root.join("targets")).expect("create targets dir");
    fs::write(
        root.join("targets").join("cpp_level.facts.json"),
        compiled_binary_fact_json("level").to_string(),
    )
    .expect("write sibling fact file");

    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"./targets/cpp_level\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, Some(&root)).expect("a bare-binary argv recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            // The fact's return set carries no `integer()` form (a
            // real-valued 0.0…1.0 band, matching `Level`'s own
            // declared type `float`) — `requires_or_reads_integer`
            // reads `PrimitiveKind::Float` here, unlike this test
            // module's OTHER fixtures, whose return sets state
            // `integer()` explicitly.
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }

    fs::remove_dir_all(&root).ok();
}

/// The SAME bare-binary argv with no sibling fact file on disk at
/// all: recognition still reaches the artifact lookup (the call is
/// not declined as an unrecognized shape), and the lookup's own
/// decline names the compiled-binary construct — never the generic
/// "there is no <path>.refined.json; write it with -export-fact"
/// sentence, which names a command that has no meaning for a target
/// that is not TypeScript source. The no-file rung of the ladder.
#[test]
fn a_bare_binary_argv_with_no_artifact_declines_naming_the_compiled_binary() {
    let Some(kernel) = loaded_kernel() else { return };
    let root = temp_binary_dir("no_fact");
    fs::create_dir_all(root.join("targets")).expect("create targets dir");
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"./targets/cpp_level\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, Some(&root)).expect("a bare-binary argv recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("cpp_level"), "{message}");
            assert!(message.contains("compiled binary"), "{message}");
            assert!(!message.contains("-export-fact"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted a decline, got a fire: {message}"),
    }

    fs::remove_dir_all(&root).ok();
}

/// A sibling fact file that EXISTS but is not readable JSON declines
/// naming the unreadable file — the ladder's second rung, distinct
/// from the no-file rung above: `sibling_exists` at the discharge
/// site is what tells the two rungs apart.
#[test]
fn a_sibling_fact_file_that_fails_to_parse_declines_naming_the_file() {
    let Some(kernel) = loaded_kernel() else { return };
    let root = temp_binary_dir("unreadable");
    fs::create_dir_all(root.join("targets")).expect("create targets dir");
    fs::write(root.join("targets").join("cpp_level.facts.json"), b"not json at all").expect("write garbage fact file");

    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"./targets/cpp_level\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, Some(&root)).expect("a bare-binary argv recognizes") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("cpp_level.facts.json"), "{message}");
            assert!(message.contains("not readable JSON"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted a decline, got a fire: {message}"),
    }

    fs::remove_dir_all(&root).ok();
}

/// A bare WORD with no leading path marker (`["cpp_level"]`, no
/// `./`/`../`/`/` prefix) is not path-shaped — the compiled-binary
/// row does not claim it, so this one-element argv is the ordinary
/// "not this shape" decline, naming the same absence a two-or-three-
/// element argv with an unrecognized runner word already names.
#[test]
fn a_bare_word_with_no_path_marker_is_not_the_compiled_binary_shape() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"cpp_level\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes as a decline") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("cannot name the code that runs next"), "{message}");
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a decline, got an override"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted a decline, got a fire: {message}"),
    }
}

/* ── the const-held literal path ──────────────────────────────────── */

#[test]
fn a_module_level_constant_script_path_resolves_and_binds() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", TARGET_PATH],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[
        ("boosted", boosted_sequence_value()),
        ("TARGET_PATH", string_literal_value_for_test("./audio_level.ts")),
    ]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the const-held path resolves") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

#[test]
fn an_fstring_script_path_declines_with_the_law_2_sentence() {
    let source = concat!(
        "def f(boosted):\n",
        "    name = \"audio_level\"\n",
        "    result = subprocess.run(\n",
        "        [\"node\", f\"./{name}.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("the call is still recognized as subprocess.run") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("computed"), "{message}");
            assert!(message.contains("written string literal"), "{message}");
        }
        _ => panic!("wanted the law-2 decline naming a computed script path"),
    }
}

#[test]
fn a_parameter_script_path_declines_with_the_law_2_sentence() {
    let source = concat!(
        "def f(boosted, script_path):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", script_path],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized as subprocess.run") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("computed"), "{message}");
            assert!(message.contains("written string literal"), "{message}");
        }
        _ => panic!("wanted the law-2 decline naming a computed script path"),
    }
}

/* ── os.system ────────────────────────────────────────────────────── */

#[test]
fn os_system_with_both_redirections_but_no_entry_write_names_the_missing_write() {
    // ONE-CHECKER.md item 2's own current text: both redirections are
    // present, so this row is tried as a full file-legs crossing — but
    // nothing in this body writes `in.json` before the call, so the
    // entry leg has no payload to attach, and the decline names
    // exactly that missing piece rather than the old (superseded)
    // "os.system captures no stdout" sentence.
    let source = concat!(
        "def f(boosted):\n",
        "    exit_code = os.system(\"node ./audio_level.ts < in.json > out.json\")\n",
        "    with open(\"out.json\") as handle:\n",
        "        return json.load(handle)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("os.system is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("in.json"), "{message}");
            assert!(message.contains("json.dump"), "{message}");
        }
        _ => panic!("wanted a decline naming the missing entry write"),
    }
}

#[test]
fn os_system_with_only_one_redirection_still_declines_naming_no_stdout_capture() {
    // Only ONE of the two redirections is present — the file-legs
    // shape genuinely needs both, so this stays the older no-stdout-
    // capture decline, unaffected by the file-legs reading.
    let source = concat!(
        "def f(boosted):\n",
        "    exit_code = os.system(\"node ./audio_level.ts < in.json\")\n",
        "    return None\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("os.system is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("captures no stdout"), "{message}");
            assert!(message.contains("subprocess.run"), "{message}");
        }
        _ => panic!("wanted a decline naming the missing captured-stdout leg"),
    }
}

#[test]
fn os_system_with_both_file_legs_present_overrides_the_return_value() {
    // The full shape: a preceding literal-file write binds the entry
    // leg, and a following literal-file read binds the return leg —
    // both redirections present, both legs found, so this is a real,
    // judged crossing, not a decline.
    let source = concat!(
        "def f(boosted):\n",
        "    with open(\"in.json\", \"w\") as infile:\n",
        "        json.dump(boosted, infile)\n",
        "    exit_code = os.system(\"node ./audio_level.ts < in.json > out.json\")\n",
        "    with open(\"out.json\") as handle:\n",
        "        return json.load(handle)\n",
    );
    let body = def_body(source);
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("os.system's file legs are recognized") {
        ForeignEdgeOutcome::Override { .. } => {}
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// `a-invocation-functions.py`'s own `level_via_os_system` shape,
/// EXACTLY: an UNBOUNDED `boosted` element escapes `audio_level.ts`'s
/// stated `[-2, 2]` entry, so the outbound leg fires before the
/// return leg's own `os_system_return_read_of` scan would otherwise
/// run. `finish_recognized_edge` runs that SAME scan under the fire
/// (`scan_sole_consumer`'s `FileRead` arm), so `consumer` binds the
/// `json.load(handle)` node inside the trailing `with open("out.json")`
/// block to `audio_level.ts`'s own stated return set — the SAME
/// value `foreign_return_value_or_undetermined` builds for the green
/// `Override` path, built here from the SAME artifact even though
/// the outbound leg refused. The fire and the bound return fact are
/// independent truths: this row reports one fire, and the consumer
/// judges the real fact rather than falling to an unbound catch-all.
#[test]
fn os_system_with_an_unbounded_element_fires_and_still_binds_the_file_read_consumer() {
    let source = concat!(
        "def f(boosted):\n",
        "    with open(\"in.json\", \"w\") as infile:\n",
        "        json.dump(boosted, infile)\n",
        "    exit_code = os.system(\"node ./audio_level.ts < in.json > out.json\")\n",
        "    with open(\"out.json\") as handle:\n",
        "        return json.load(handle)\n",
    );
    let body = def_body(source);
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let unbounded_boosted = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(repetition(make_refined_set(vec![at_least(f64::NEG_INFINITY)]), 1, None), None, TrustProved, SetKindTag::None)
    };
    let environment = env_with(&[("boosted", unbounded_boosted)]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("os.system's file legs are still recognized") {
        ForeignEdgeOutcome::Fired { message, consumer, .. } => {
            assert!(message.contains("outside the target's stated entry set"), "{message}");
            let (_, value) = consumer.expect("the fired edge's sole json.load(handle) consumer must be bound");
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire — an unbounded float list must not fit [-2, 2]"),
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
    }
}

#[test]
fn os_system_with_a_variable_command_declines_with_the_shell_string_sentence() {
    let source = concat!(
        "def f(boosted):\n",
        "    command = \"node ./audio_level.ts\"\n",
        "    exit_code = os.system(command)\n",
        "    with open(\"out.json\") as handle:\n",
        "        return json.load(handle)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("os.system is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("shell string"), "{message}");
            assert!(message.contains("argv list"), "{message}");
        }
        _ => panic!("wanted the shell-string law-2 decline"),
    }
}

#[test]
fn os_system_with_an_unsupported_trailing_token_names_it() {
    let source = concat!(
        "def f(boosted):\n",
        "    exit_code = os.system(\"node ./audio_level.ts --extra-flag\")\n",
        "    with open(\"out.json\") as handle:\n",
        "        return json.load(handle)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("os.system is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => {
            assert!(message.contains("--extra-flag"), "{message}");
        }
        _ => panic!("wanted a decline naming the unsupported trailing token"),
    }
}
