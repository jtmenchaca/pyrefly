//! Recognition of the foreign-edge shape itself: shadowed module
//! names, missing keywords, and the alternate call spellings
//! (Popen, asyncio, walrus) that must recognize the same crossing.

use super::*;

/// REGRESSION PIN: the finite, int-sorted return
/// (`audio_level_ts_artifact`'s `integer, >= 0, <= 1`) binds exactly
/// as before — a corner-free, explicitly-int-sorted set crosses
/// undegraded, now correctly Integer-tagged (the fixed reading of
/// defect 2, not the pre-fix Float stamp).
#[test]
fn the_exact_shape_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes");
    match outcome {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

#[test]
fn a_shadowed_subprocess_name_is_not_recognized() {
    let body = def_body(FIXTURE_SOURCE);
    let mut environment = env_with(&[("boosted", boosted_sequence_value())]);
    environment.bind("subprocess", known_values(vec![0.0], PrimitiveKind::Integer, TrustProved));
    let Some(kernel) = loaded_kernel() else { return };
    assert!(
        foreign_edge_at(&body, 0, &environment, &kernel, None).is_none(),
        "a locally shadowed subprocess must not be read as the module"
    );
}

/// `level_via_runner_variable`'s own shape, isolated: `runner = "node"`
/// at position 0 (pre-seeded into `environment` the same way the real
/// walk's own `walk_statement` would have bound it by the time
/// `serve_foreign_edge_at` reaches position 1), then `subprocess.run(
/// [runner, "./audio_level.ts"], ...)` at position 1 — a bare `Name`
/// at argv[0] resolving to a known exact string through
/// `interpreter_text_of`'s own const-fold, mirroring `script_text_of`'s
/// already-landed `Name` branch (`a_module_level_constant_script_path_
/// resolves_and_binds`'s own pin, at the SCRIPT position). Before this
/// fix, `argv_runner_and_script`'s `[interpreter, script]` arm read
/// argv[0] through `literal_string` alone (`Expr::StringLiteral` only),
/// so a bare `Name` answered no runner word at all and the whole call
/// stayed unrecognized; the return read as the generic `json.loads`
/// union and fired RTS7001 at `None`. This call fits `audio_level.ts`'s
/// stated entry cleanly, so the fix recognizes it as `Runner::Node` and
/// binds the target's own proved return — silent.
#[test]
fn a_runner_held_in_a_variable_resolves_and_binds() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    runner = \"node\"\n",
        "    result = subprocess.run(\n",
        "        [runner, \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[
        ("boosted", boosted_sequence_value()),
        ("runner", string_literal_value_for_test("node")),
    ]);
    match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("the runner-variable call resolves") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// `b-runners.py`'s own `level_via_runner_variable` shape, EXACTLY —
/// an UNBOUNDED `boosted` element (`list[float]`, no `BoostedSample`
/// bound). Before `interpreter_text_of`, this call's argv[0] (a bare
/// `Name`) answered no runner word at all, so the whole call stayed
/// UNRECOGNIZED and the fixture's own docstring named a single
/// finding: the return's generic `json.loads` union firing on its
/// `None` arm. The fold now recognizes `runner` as `Runner::Node`, so
/// the call proceeds to the OUTBOUND-LEG fit ask — and THAT fires
/// too, at the payload, since an unbounded float list genuinely
/// escapes `audio_level.ts`'s stated `[-2, 2]` entry. `consumer` stays
/// `None` here: this is `ResultRead::StdoutAttribute` (an ordinary
/// `json.loads(result.stdout)` call), whose own unbound walk still
/// reaches `expressions.rs`'s union-of-the-full-JSON-value-space
/// model and fires its OWN determined `None`-arm RTS7001 at the
/// return — the corpus's own designed SECOND finding for this exact
/// row, which a bound `consumer` here would have wrongly replaced
/// with a judge against a narrower set the row never claims.
#[test]
fn a_runner_held_in_a_variable_with_an_unbounded_element_fires_at_the_outbound_leg() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    runner = \"node\"\n",
        "    result = subprocess.run(\n",
        "        [runner, \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );
    let body = def_body(source);
    let unbounded_boosted = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(repetition(make_refined_set(vec![at_least(f64::NEG_INFINITY)]), 1, None), None, TrustProved, SetKindTag::None)
    };
    let environment = env_with(&[("boosted", unbounded_boosted), ("runner", string_literal_value_for_test("node"))]);
    match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("the runner-variable call still recognizes") {
        ForeignEdgeOutcome::Fired { message, consumer, .. } => {
            assert!(message.contains("outside the target's stated entry set"), "{message}");
            assert!(
                consumer.is_none(),
                "an ordinary json.loads(result.stdout) consumer must stay unbound — its own \
                 union-None-arm fire is the row's designed second finding, not a value to replace it with: \
                 {consumer:?}"
            );
        }
        ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire — an unbounded float list must not fit [-2, 2]"),
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
    }
}

/* ── a walrus-bound call inside an `if` test ──────────────────────── */

/// FIX 2: `if (result := subprocess.run(...)).returncode == 0: return
/// json.loads(result.stdout)` (`level_via_walrus_result`, d-data-
/// legs.py:205). Before this fix, `recognize_foreign_edge` dispatched
/// only on `statements[index]` being `Stmt::Assign`/`Stmt::With` —
/// the SAME gate `check.rs`'s own body loop applied before ever
/// calling `foreign_edge_at` at all. This statement is a `Stmt::If`
/// whose TEST embeds the walrus, so the recognizer never fired at
/// all — not recognized-and-blocked, structurally unreached. This
/// pins the fix's own entry point directly: `foreign_edge_at_walrus_
/// call`, given the walrus's own target/call and the taken arm's own
/// body (where the `json.loads(...)` consumer sits), recognizes and
/// binds the proved return exactly as the flat `Assign`-shaped call
/// already does.
#[test]
fn a_walrus_bound_run_call_in_an_if_test_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def audio_level_via_walrus(boosted):\n",
        "    if (result := subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )).returncode == 0:\n",
        "        return json.loads(result.stdout)\n",
        "    return 0\n",
    );
    let body = def_body(source);
    let Stmt::If(if_stmt) = &body[0] else {
        panic!("this fixture's own top-level statement must be the if");
    };
    let (target, call) = walrus_test_target_and_call(if_stmt);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at_walrus_call(&call, &target, &if_stmt.body, 0, &environment, &kernel, None)
        .expect("the walrus-bound run call recognizes");
    match outcome {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// REGRESSION PIN: the flat `Assign`-shaped spelling
/// (`result = subprocess.run(...)`, read through `foreign_edge_at`'s
/// ordinary `Stmt::Assign` dispatch) still recognizes exactly as
/// before this fix — the walrus-in-test entry point is an ADDITIONAL
/// way to reach `recognize_subprocess_callee`, never a replacement
/// for `recognize_foreign_edge`'s own `Stmt::Assign` path.
#[test]
fn the_flat_assign_shaped_run_call_still_recognizes_after_the_walrus_fix() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the flat shape recognizes");
    match outcome {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/* ── subprocess.Popen ─────────────────────────────────────────────── */

#[test]
fn popen_with_communicate_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    process = subprocess.Popen(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        stdin=subprocess.PIPE,\n",
        "        stdout=subprocess.PIPE,\n",
        "        text=True,\n",
        "    )\n",
        "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
        "    return json.loads(stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the Popen pair recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

#[test]
fn popen_with_no_following_communicate_declines() {
    let source = concat!(
        "def f(boosted):\n",
        "    process = subprocess.Popen(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        stdin=subprocess.PIPE,\n",
        "        stdout=subprocess.PIPE,\n",
        "        text=True,\n",
        "    )\n",
        "    return process\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("Popen itself is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("communicate"), "{message}"),
        _ => panic!("wanted a decline naming the missing .communicate() call"),
    }
}

#[test]
fn popen_with_a_missing_stdin_pipe_keyword_declines() {
    let source = concat!(
        "def f(boosted):\n",
        "    process = subprocess.Popen(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        stdout=subprocess.PIPE,\n",
        "        text=True,\n",
        "    )\n",
        "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
        "    return json.loads(stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("Popen itself is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("stdin"), "{message}"),
        _ => panic!("wanted a decline naming the missing stdin=subprocess.PIPE keyword"),
    }
}

/// FIX 1: `with subprocess.Popen([...]) as process:` — the idiomatic
/// context-manager wrapping of the flat Popen/`.communicate()` pair
/// above (`level_via_popen_context_manager`, a-invocation-
/// functions.py:80). Before this fix, `recognize_foreign_edge`'s own
/// `Stmt::With` branch tried only `recognize_temp_file_edge` with no
/// fallthrough, so this call never reached `recognize_subprocess_
/// popen` at all — `foreign_edge_at` answered `None` (structurally
/// unrecognized, not a decline) and `process.communicate(...)`'s
/// result read as an ordinary opaque call. This pins the fix:
/// `foreign_edge_at` now tries the Popen-context-manager shape when
/// the temp-file shape declines the whole with-statement, reading
/// the `.communicate()` assign and its own `json.loads(...)`
/// consumer out of the WITH-BLOCK's own body.
#[test]
fn popen_inside_a_with_block_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    with subprocess.Popen(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        stdin=subprocess.PIPE,\n",
        "        stdout=subprocess.PIPE,\n",
        "        text=True,\n",
        "    ) as process:\n",
        "        stdout, _stderr = process.communicate(json.dumps(boosted))\n",
        "        return json.loads(stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the with-wrapped Popen pair recognizes")
    {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// REGRESSION PIN: the flat (non-with) Popen/`.communicate()` spelling
/// still recognizes exactly as before this fix — the with-wrapped
/// shape is an ADDITIONAL recognized shape, never a replacement for
/// the statement-pair one `recognize_subprocess_popen` already reads.
#[test]
fn the_flat_popen_pair_still_recognizes_after_the_with_fix() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    process = subprocess.Popen(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        stdin=subprocess.PIPE,\n",
        "        stdout=subprocess.PIPE,\n",
        "        text=True,\n",
        "    )\n",
        "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
        "    return json.loads(stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the flat Popen pair still recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/* ── asyncio.create_subprocess_exec ──────────────────────────────── */

/// EDGE-COVERAGE §K's own headline row (`k-async-invocation.py`'s
/// `level_via_async_subprocess`): the awaited twin of
/// `popen_with_communicate_recognizes_and_binds_the_proved_return`,
/// spelled with `await asyncio.create_subprocess_exec(...)` and
/// `await proc.communicate(json.dumps(...).encode())` in place of
/// `subprocess.Popen`/`.communicate(json.dumps(...))`. Pins that the
/// async spelling now recognizes and binds the target's own proved
/// return, exactly like the synchronous shape.
#[test]
fn async_create_subprocess_exec_recognizes_and_binds_the_proved_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "async def f(boosted):\n",
        "    proc = await asyncio.create_subprocess_exec(\n",
        "        \"node\",\n",
        "        \"./audio_level.ts\",\n",
        "        stdin=asyncio.subprocess.PIPE,\n",
        "        stdout=asyncio.subprocess.PIPE,\n",
        "    )\n",
        "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
        "    return json.loads(stdout_bytes)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the async create_subprocess_exec pair recognizes")
    {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// EDGE-COVERAGE §K's second row (`level_via_async_subprocess_optional`):
/// the identical async call, with the SAME recognition — the
/// declared-return widening to `Optional[Level]` this row measures is
/// a return-judge question, downstream of recognition, so this test
/// pins that recognition itself is unaffected by the declared return
/// shape and still binds the same proved fact.
#[test]
fn async_create_subprocess_exec_recognizes_regardless_of_the_declared_return() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "async def f(boosted):\n",
        "    proc = await asyncio.create_subprocess_exec(\n",
        "        \"node\",\n",
        "        \"./audio_level.ts\",\n",
        "        stdin=asyncio.subprocess.PIPE,\n",
        "        stdout=asyncio.subprocess.PIPE,\n",
        "    )\n",
        "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
        "    return json.loads(stdout_bytes)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the async pair recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// The bytes encode/decode unwrapping, pinned on BOTH legs at once:
/// the outbound payload rides `json.dumps(...).encode()` (unwrapped
/// by `unwrap_bytes_encode` before `json_dumps_argument_of` reads it)
/// and the return leg reads `json.loads(stdout_text.decode())`
/// (unwrapped by `unwrap_bytes_decode` before `is_foreign_parse_of`
/// matches the name) — the `.encode()`/`.decode()` wrapper on either
/// leg names the identical JSON text/bytes as the unwrapped spelling
/// pinned above, so this recognizes and binds the same proved return.
#[test]
fn async_create_subprocess_exec_unwraps_encode_and_decode_on_both_legs() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "async def f(boosted):\n",
        "    proc = await asyncio.create_subprocess_exec(\n",
        "        \"node\",\n",
        "        \"./audio_level.ts\",\n",
        "        stdin=asyncio.subprocess.PIPE,\n",
        "        stdout=asyncio.subprocess.PIPE,\n",
        "    )\n",
        "    stdout_text, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
        "    return json.loads(stdout_text.decode())\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the .decode()-wrapped consumer recognizes")
    {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// The bare (non-`.encode()`) payload and bare (non-`.decode()`)
/// return both still recognize — `json_dumps_argument_of`/
/// `is_foreign_parse_of` read through an ABSENT wrapper exactly as
/// readily as a present one, since `unwrap_bytes_encode`/
/// `unwrap_bytes_decode` answer the expression unchanged when no
/// `.encode()`/`.decode()` call wraps it.
#[test]
fn async_create_subprocess_exec_recognizes_without_encode_or_decode() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "async def f(boosted):\n",
        "    proc = await asyncio.create_subprocess_exec(\n",
        "        \"node\",\n",
        "        \"./audio_level.ts\",\n",
        "        stdin=asyncio.subprocess.PIPE,\n",
        "        stdout=asyncio.subprocess.PIPE,\n",
        "    )\n",
        "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted))\n",
        "    return json.loads(stdout_bytes)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the unwrapped pair recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// An explicitly non-PIPE `stdout` (`asyncio.subprocess.DEVNULL`)
/// refuses recognition — this call IS `asyncio.create_subprocess_exec`
/// with a readable runner/script, but the checker cannot read the
/// target's stdout back at all, so it declines with the same
/// channel-refusal sentence family `subprocess_popen_keywords_of`'s
/// own sync check already speaks, never a second sentence for the
/// async spelling.
#[test]
fn async_create_subprocess_exec_with_a_non_pipe_stdout_declines() {
    let source = concat!(
        "async def f(boosted):\n",
        "    proc = await asyncio.create_subprocess_exec(\n",
        "        \"node\",\n",
        "        \"./audio_level.ts\",\n",
        "        stdin=asyncio.subprocess.PIPE,\n",
        "        stdout=asyncio.subprocess.DEVNULL,\n",
        "    )\n",
        "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
        "    return json.loads(stdout_bytes)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    let Some(kernel) = loaded_kernel() else { return };
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call itself is recognized") {
        ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("stdout"), "{message}"),
        _ => panic!("wanted a decline naming the non-PIPE stdout"),
    }
}

/// REGRESSION PIN: every synchronous shape this crate already
/// recognized (`subprocess.run`, `subprocess.Popen`) still recognizes
/// after the asyncio row is added — the awaited path is reached only
/// when the assign's own value is `Expr::Await`, so a bare
/// `Expr::Call` value falls straight through to the unchanged sync
/// dispatch, never through the new asyncio reader at all.
#[test]
fn the_synchronous_subprocess_run_shape_still_recognizes_after_the_asyncio_row() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let body = def_body(FIXTURE_SOURCE);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("subprocess.run still recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}

/// REGRESSION PIN: the synchronous Popen/`.communicate()` pair still
/// recognizes after the asyncio row is added — the SAME pin
/// `the_flat_popen_pair_still_recognizes_after_the_with_fix` already
/// keeps for the with-wrapped fix, rerun here for the asyncio one.
#[test]
fn the_synchronous_popen_shape_still_recognizes_after_the_asyncio_row() {
    register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def f(boosted):\n",
        "    process = subprocess.Popen(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        stdin=subprocess.PIPE,\n",
        "        stdout=subprocess.PIPE,\n",
        "        text=True,\n",
        "    )\n",
        "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
        "    return json.loads(stdout)\n",
    );
    let body = def_body(source);
    let environment = env_with(&[("boosted", boosted_sequence_value())]);
    match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("Popen still recognizes") {
        ForeignEdgeOutcome::Override { value, .. } => {
            assert_eq!(value.kind, Kind::Set);
            assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        }
        ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
        ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
    }
}
