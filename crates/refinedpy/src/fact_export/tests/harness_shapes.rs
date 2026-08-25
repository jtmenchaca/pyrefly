use super::*;

/// The main-block reader recognizes the exact stdin→f→stdout shape
/// and nothing looser.
#[test]
fn the_harness_reader_recognizes_the_json_stdio_shape() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(x): return x\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(json.load(sys.stdin))))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::StdinJson { called }) = harness_shape(&module) else {
        panic!("expected the stdin-JSON shape");
    };
    assert_eq!(called, "f");
}

/// The main-block reader recognizes the stdin-JSON shape spelled with
/// one intermediate assignment — `D5.count.helper.py`'s own anatomy:
/// `value = json.load(sys.stdin)` bound one statement before a call
/// whose sole argument is that bound name.
#[test]
fn the_harness_reader_recognizes_the_stdin_json_shape_via_intermediate_assignment() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(x): return x\n\n\nif __name__ == \"__main__\":\n    value = json.load(sys.stdin)\n    print(json.dumps(f(value)))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::StdinJson { called }) = harness_shape(&module) else {
        panic!("expected the stdin-JSON shape");
    };
    assert_eq!(called, "f");
}

#[test]
fn a_main_block_of_another_shape_states_no_harness() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(x): return x\n\n\nif __name__ == \"__main__\":\n    print(f(1))\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(harness_shape(&module).is_none());

    let no_main = ruff_python_parser::parse_module("def f(x): return x\n")
        .expect("test module parses")
        .into_syntax();
    assert!(harness_shape(&no_main).is_none());
}

/// The main-block reader recognizes the argv-scalar shape spelled
/// exactly as `level_gain_argv.py`'s own main block spells it: one
/// intermediate assignment (`gain = float(sys.argv[1])`), then the
/// called function's sole argument is that bound name.
#[test]
fn the_harness_reader_recognizes_the_argv_scalar_shape_via_intermediate_assignment() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(gain: float) -> float: return gain\n\n\nif __name__ == \"__main__\":\n    gain = float(sys.argv[1])\n    print(json.dumps(f(gain)))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::ArgvScalar { called, arg_index }) = harness_shape(&module) else {
        panic!("expected the argv-scalar shape");
    };
    assert_eq!(called, "f");
    assert_eq!(arg_index, 1);
}

/// The same shape read with the argv expression written inline as
/// the call's sole argument, no intermediate assignment at all.
#[test]
fn the_harness_reader_recognizes_the_argv_scalar_shape_written_inline() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(gain: float) -> float: return gain\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(float(sys.argv[2]))))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::ArgvScalar { called, arg_index }) = harness_shape(&module) else {
        panic!("expected the argv-scalar shape");
    };
    assert_eq!(called, "f");
    assert_eq!(arg_index, 2);
}

/// An `int(...)` parse in the argv position is not the one recognized
/// parse — the block states no harness at all, rather than a
/// guessed `argv-scalar` with the wrong `parse` field.
#[test]
fn an_int_parsing_harness_declines_with_no_shape_recognized() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(n: int) -> int: return n\n\n\nif __name__ == \"__main__\":\n    n = int(sys.argv[1])\n    print(json.dumps(f(n)))\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(harness_shape(&module).is_none());

    let inline = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(n: int) -> int: return n\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(int(sys.argv[1]))))\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(harness_shape(&inline).is_none());
}

/// The main-block reader recognizes the mixed shape spelled exactly
/// as `level_gain_argv.py`'s own main block spells it: `gain =
/// float(sys.argv[1])` bound one statement before a call whose first
/// argument is `json.load(sys.stdin)` and whose second is `gain`.
#[test]
fn the_harness_reader_recognizes_the_stdin_json_argv_scalar_shape() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(samples, gain: float) -> float: return gain\n\n\nif __name__ == \"__main__\":\n    gain = float(sys.argv[1])\n    print(json.dumps(f(json.load(sys.stdin), gain)))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::StdinJsonArgvScalar { called, arg_index }) = harness_shape(&module) else {
        panic!("expected the stdin-json-argv-scalar shape");
    };
    assert_eq!(called, "f");
    assert_eq!(arg_index, 1);
}

/// The same mixed shape read with the argv read written inline as
/// the call's second argument, no intermediate assignment at all.
#[test]
fn the_harness_reader_recognizes_the_stdin_json_argv_scalar_shape_written_inline() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(samples, gain: float) -> float: return gain\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(json.load(sys.stdin), float(sys.argv[2]))))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::StdinJsonArgvScalar { called, arg_index }) = harness_shape(&module) else {
        panic!("expected the stdin-json-argv-scalar shape");
    };
    assert_eq!(called, "f");
    assert_eq!(arg_index, 2);
}

/// The arguments in the OTHER order — the argv read first, stdin's
/// JSON second — is not the recognized shape: position carries the
/// meaning (parameter 0 is stdin, parameter 1 is argv), so the
/// swapped order states no harness at all.
#[test]
fn the_stdin_json_argv_scalar_shape_requires_stdin_first() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(gain: float, samples) -> float: return gain\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(float(sys.argv[1]), json.load(sys.stdin))))\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(harness_shape(&module).is_none());
}

/// An `int(...)` parse in the mixed shape's argv position declines
/// as an omission — no shape recognized — exactly as the pure
/// argv-scalar leg declines it.
#[test]
fn an_int_parsing_mixed_harness_declines_with_no_shape_recognized() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(samples, n: int) -> int: return n\n\n\nif __name__ == \"__main__\":\n    n = int(sys.argv[1])\n    print(json.dumps(f(json.load(sys.stdin), n)))\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(harness_shape(&module).is_none());

    let inline = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(samples, n: int) -> int: return n\n\n\nif __name__ == \"__main__\":\n    print(json.dumps(f(json.load(sys.stdin), int(sys.argv[1]))))\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(harness_shape(&inline).is_none());
}

/// The main-block reader recognizes the file shape spelled exactly
/// as `level_from_file.py`'s own main block spells it: `with
/// open(sys.argv[1]) as payload_file:` binding `payload =
/// json.load(payload_file)`, then `print(json.dumps(f(payload)))`.
#[test]
fn the_harness_reader_recognizes_the_file_json_shape() {
    let module = ruff_python_parser::parse_module(
        "import json\nimport sys\n\n\ndef f(x): return x\n\n\nif __name__ == \"__main__\":\n    with open(sys.argv[1]) as payload_file:\n        payload = json.load(payload_file)\n    print(json.dumps(f(payload)))\n",
    )
    .expect("test module parses")
    .into_syntax();
    let Some(HarnessShape::FileJson { called, arg_index }) = harness_shape(&module) else {
        panic!("expected the file-json shape");
    };
    assert_eq!(called, "f");
    assert_eq!(arg_index, 1);
}
