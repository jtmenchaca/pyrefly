use super::*;

/// A `print` anywhere in the body — or in a same-module def the body
/// calls — refuses the channel-purity claim.
#[test]
fn the_stdout_scan_follows_same_module_calls() {
    let module = ruff_python_parser::parse_module(
        "def quiet(x):\n    return x + 1\n\n\ndef loud(x):\n    print(x)\n    return x\n\n\ndef calls_quiet(x):\n    return quiet(x)\n\n\ndef calls_loud(x):\n    return loud(x)\n",
    )
    .expect("test module parses")
    .into_syntax();
    let defs: Vec<&StmtFunctionDef> = top_level_defs(&module).collect();
    let by_name = |wanted: &str| {
        *defs
            .iter()
            .find(|def| def.name.id.as_str() == wanted)
            .expect("the test module declares this def")
    };
    assert!(writes_nothing_to_stdout(by_name("quiet"), &module));
    assert!(!writes_nothing_to_stdout(by_name("loud"), &module));
    assert!(writes_nothing_to_stdout(by_name("calls_quiet"), &module));
    assert!(!writes_nothing_to_stdout(by_name("calls_loud"), &module));
}

/// A `sys.stdout.write(...)` is the same refusal a `print` is.
#[test]
fn the_stdout_scan_catches_a_direct_stdout_write() {
    let module = ruff_python_parser::parse_module(
        "import sys\n\n\ndef writes(x):\n    sys.stdout.write(\"hi\")\n    return x\n",
    )
    .expect("test module parses")
    .into_syntax();
    let def = top_level_defs(&module).next().expect("one def");
    assert!(!writes_nothing_to_stdout(def, &module));
}

/// `subprocess.run(..., capture_output=True)` pipes the child's
/// stdout into `result.stdout`, never the parent's own stdout —
/// `chain_relay.py`'s own anatomy (the fixture named in
/// j-chains-and-diamonds.py's own header): the purity claim
/// discharges even though the body spawns a child.
#[test]
fn a_captured_subprocess_run_discharges_the_purity_claim() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport subprocess\n\n\ndef relay(x):\n    result = subprocess.run(\n        [\"node\", \"./targets/chain_meter.ts\"],\n        input=json.dumps(x),\n        capture_output=True,\n        text=True,\n    )\n    return json.loads(result.stdout)\n",
    )
    .expect("test module parses")
    .into_syntax();
    let def = top_level_defs(&module).next().expect("one def");
    assert!(writes_nothing_to_stdout(def, &module));
}

/// `subprocess.run(...)` WITHOUT `capture_output=True` inherits the
/// parent's own stdout (`library/subprocess.rst`'s documented
/// default, `stdout=None`) — the purity claim still refuses, exactly
/// as an uncaptured spawn should.
#[test]
fn subprocess_run_without_capture_output_still_refuses() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport subprocess\n\n\ndef relay(x):\n    result = subprocess.run(\n        [\"node\", \"./targets/chain_meter.ts\"],\n        input=json.dumps(x),\n        text=True,\n    )\n    return json.loads(result.stdout)\n",
    )
    .expect("test module parses")
    .into_syntax();
    let def = top_level_defs(&module).next().expect("one def");
    assert!(!writes_nothing_to_stdout(def, &module));
}

/// `subprocess.Popen(..., stdout=subprocess.PIPE)` admits — the same
/// PIPE sentinel `run`'s `capture_output=True` shorthand expands to,
/// read directly since `Popen` carries no such shorthand of its own.
#[test]
fn popen_with_stdout_pipe_admits() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport subprocess\n\n\ndef relay(x):\n    process = subprocess.Popen(\n        [\"node\", \"./targets/chain_meter.ts\"],\n        stdin=subprocess.PIPE,\n        stdout=subprocess.PIPE,\n        text=True,\n    )\n    stdout, _stderr = process.communicate(json.dumps(x))\n    return json.loads(stdout)\n",
    )
    .expect("test module parses")
    .into_syntax();
    let def = top_level_defs(&module).next().expect("one def");
    assert!(writes_nothing_to_stdout(def, &module));
}

/// A `print(...)` body still refuses — the captured-spawn admission
/// table added nothing to a body whose own impurity is a genuine
/// stdout write (regression against the pre-existing scan).
#[test]
fn a_print_body_still_refuses() {
    let module = ruff_python_parser::parse_module("def loud(x):\n    print(x)\n    return x\n")
        .expect("test module parses")
        .into_syntax();
    let def = top_level_defs(&module).next().expect("one def");
    assert!(!writes_nothing_to_stdout(def, &module));
}

/// The awaited `asyncio.create_subprocess_exec(...,
/// stdout=asyncio.subprocess.PIPE)` shape admits — the same PIPE
/// contract as the synchronous `Popen`, re-exported under the
/// `asyncio.subprocess` namespace (k-async-invocation.py's own
/// `level_via_async_subprocess` anatomy).
#[test]
fn the_asyncio_captured_shape_admits() {
    let module = ruff_python_parser::parse_module(
        "import asyncio\nimport json\n\n\nasync def relay(x):\n    proc = await asyncio.create_subprocess_exec(\n        \"node\",\n        \"./targets/chain_meter.ts\",\n        stdin=asyncio.subprocess.PIPE,\n        stdout=asyncio.subprocess.PIPE,\n    )\n    stdout_bytes, _ = await proc.communicate(json.dumps(x))\n    return json.loads(stdout_bytes)\n",
    )
    .expect("test module parses")
    .into_syntax();
    let def = top_level_defs(&module).next().expect("one def");
    assert!(writes_nothing_to_stdout(def, &module));
}
