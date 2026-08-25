use super::*;

/// A module whose `__main__` block IS the recognized stdin-JSON
/// shape exports schema v2's `surface` object exactly: the tagged
/// union's `kind` present alongside the carried-forward
/// stdin/stdout/calls fields.
#[test]
fn a_recognized_main_block_exports_the_v2_surface_object() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = b"def f(x: int) -> int:\n    return x\n\n\nif __name__ == \"__main__\":\n    import json\n    import sys\n    print(json.dumps(f(json.load(sys.stdin))))\n";
    let text = String::from_utf8(source.to_vec()).expect("the fixture is UTF-8");
    let module = ruff_python_parser::parse_module(&text)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source, "f.py", no_imports, &kernel, None);
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    assert_eq!(
        artifact["surface"],
        json!({"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "f"})
    );
}

/// A module whose `__main__` block IS the recognized argv-scalar
/// shape — `level_gain_argv.py`'s own anatomy, one intermediate
/// assignment binding `gain = float(sys.argv[1])` before the call —
/// exports schema v2's `surface` object exactly.
#[test]
fn a_recognized_argv_scalar_main_block_exports_the_v2_surface_object() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = b"def f(gain: float) -> float:\n    return gain\n\n\nif __name__ == \"__main__\":\n    import json\n    import sys\n\n    gain = float(sys.argv[1])\n    print(json.dumps(f(gain)))\n";
    let text = String::from_utf8(source.to_vec()).expect("the fixture is UTF-8");
    let module = ruff_python_parser::parse_module(&text)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source, "f.py", no_imports, &kernel, None);
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    assert_eq!(
        artifact["surface"],
        json!({"kind": "argv-scalar", "argIndex": 1, "parse": "float", "stdout": "json", "calls": "f"})
    );
}

/// A module whose `__main__` block IS the recognized mixed shape
/// exports schema v2's `surface` object exactly — the same
/// two-parameter harness anatomy `level_gain_argv.py` states
/// (`json.load(sys.stdin)` first, `float(sys.argv[1])` second), with
/// a body the walk DOES derive a value for (unlike the real
/// fixture's own body — see
/// `the_real_mixed_fixture_bodys_return_is_still_undetermined`
/// below), so this test pins the surface recognition and the entry
/// ordering independent of that separate, ledgered gap. The called
/// function's `entry` carries the two rows in the order the
/// consumer relies on (entry[0] = stdin's parameter, entry[1] =
/// argv's).
#[test]
fn the_mixed_shape_fixture_exports_the_v2_surface_object() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = b"import sys\nfrom typing import Annotated\n\nfrom pydantic import Field\n\nSample = Annotated[float, Field(ge=-2.0, le=2.0)]\n\n\ndef level_gain_argv(samples: Annotated[list[Sample], Field(min_length=1)], gain: float) -> float:\n    return max(0.0, min(1.0, gain))\n\n\nif __name__ == \"__main__\":\n    import json\n\n    gain = float(sys.argv[1])\n    print(json.dumps(level_gain_argv(json.load(sys.stdin), gain)))\n";
    let text = String::from_utf8(source.to_vec()).expect("the fixture is UTF-8");
    let module = ruff_python_parser::parse_module(&text)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source, "level_gain_argv.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    assert_eq!(
        artifact["surface"],
        json!({
            "kind": "stdin-json-argv-scalar",
            "stdin": "json",
            "argIndex": 1,
            "parse": "float",
            "stdout": "json",
            "calls": "level_gain_argv",
        })
    );
    let entry = artifact["functions"]["level_gain_argv"]["entry"].as_array().unwrap_or_else(|| {
        let reason = export
            .omissions
            .iter()
            .find(|omission| omission.function == "level_gain_argv")
            .map(|omission| omission.reason.as_str())
            .unwrap_or("not found in either functions or omissions");
        panic!("level_gain_argv exports no entry — omission reason: {reason}");
    });
    assert_eq!(entry.len(), 2, "the mixed shape's callee states exactly two rows");
    assert_eq!(entry[0]["name"], "samples");
    assert_eq!(entry[1]["name"], "gain");
}

/// `level_gain_argv.py`'s OWN body — the real fixture, not a
/// determinable stand-in — currently derives no return value: the
/// comprehension `clamped = [max(-1.0, min(1.0, s * gain)) for s in
/// samples]` reads a second free scalar name (`gain`) alongside the
/// loop variable, a shape the single-language walk does not yet
/// carry through to `math.sqrt(total / len(samples))`. This is a
/// determination gap ledgered in ISSUES.md, not this recognizer's
/// defect — the harness-shape reader recognizes the module's
/// `surface` correctly regardless (see the test above); it is the
/// CALLEE's own return that stays underived. Pinned here so the
/// ledgered fix flips this test's own assertion the moment it
/// lands, at which point `level_gain_argv` should be asserted
/// present in `functions` instead.
/// The real mixed fixture's body — the two-free-name comprehension
/// element `s * gain` — derives its return: the transfer's
/// sort-only answer for an unbounded operand pair flows into the
/// max/min clamp, the clamp bounds it to [-1, 1], and the whole
/// sum/len/sqrt chain lands on [0, 1].
#[test]
fn the_real_mixed_fixture_bodys_return_derives() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = b"import math\nimport sys\nfrom typing import Annotated\n\nfrom pydantic import Field\n\nSample = Annotated[float, Field(ge=-2.0, le=2.0)]\nLevel = Annotated[float, Field(ge=0.0, le=1.0)]\n\n\ndef level_gain_argv(samples: Annotated[list[Sample], Field(min_length=1)], gain: float) -> Level:\n    clamped = [max(-1.0, min(1.0, s * gain)) for s in samples]\n    total = sum(s * s for s in clamped)\n    return math.sqrt(total / len(samples))\n\n\nif __name__ == \"__main__\":\n    import json\n\n    gain = float(sys.argv[1])\n    print(json.dumps(level_gain_argv(json.load(sys.stdin), gain)))\n";
    let text = String::from_utf8(source.to_vec()).expect("the fixture is UTF-8");
    let module = ruff_python_parser::parse_module(&text)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source, "level_gain_argv.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let rendered = serde_json::to_value(&export.artifact).expect("the artifact renders");
    let function = &rendered["functions"]["level_gain_argv"];
    assert!(
        !function.is_null(),
        "level_gain_argv must export (its body derives through the sort-only transfer answer)"
    );
    let cases = function["return"]["cases"].as_array().expect("the return states its cases");
    assert_eq!(cases.len(), 1, "a plain numeric return states one case");
    assert_eq!(cases[0]["sort"], "number");
    let return_forms = cases[0]["set"]["forms"].as_array().expect("the number case states its forms");
    assert_eq!(return_forms.len(), 2, "the derived return is the two-sided [0, 1] window");
}

/// UNIT 2 (ISSUES.md: "Py: loop blockers unnamed when return
/// annotation unreadable — bare `-> float` never judged;
/// undetermined bodies must name their blocker regardless"). Two
/// defs, identical bodies (`while True: pass`, no `return`
/// anywhere — an unwalkable construct with nothing for the walk to
/// derive), differing ONLY in their return annotation's own
/// readability: `readable_return`'s `-> Age` is `declared_refinement`'s
/// own vocabulary; `unreadable_return`'s `-> float` is a bare sort
/// name `declared_refinement` states nothing about
/// (`typereading.rs`'s `Expr::Name` arm requires a declared alias).
/// Both omissions must name the SAME blocker — the `while` statement
/// — never a named reason for one and the generic "derived no
/// value" placeholder for the other.
#[test]
fn an_unreadable_return_annotation_still_names_its_bodys_blocker() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def readable_return(n: Age) -> Age:\n",
        "    while True:\n",
        "        pass\n",
        "def unreadable_return(n: Age) -> float:\n",
        "    while True:\n",
        "        pass\n",
    );
    let module = ruff_python_parser::parse_module(source)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "loop_blocker.py", no_imports, &kernel, None);
    let reason_for = |name: &str| {
        export
            .omissions
            .iter()
            .find(|omission| omission.function == name)
            .unwrap_or_else(|| panic!("'{name}' must be named in an omission, got: {:?}", export.omissions.iter().map(|o| (&o.function, &o.reason)).collect::<Vec<_>>()))
            .reason
            .clone()
    };
    let readable_reason = reason_for("readable_return");
    let unreadable_reason = reason_for("unreadable_return");
    assert!(
        readable_reason.contains("while"),
        "the readable-annotation twin must name the while loop: {readable_reason}"
    );
    assert_eq!(
        readable_reason, unreadable_reason,
        "an unreadable return annotation must name the identical blocker its readable twin does"
    );
}

/// A module whose `__main__` block IS the recognized file shape
/// exports schema v2's `surface` object exactly —
/// `level_from_file.py`'s own anatomy end to end through
/// `export_module`.
#[test]
fn the_file_shape_fixture_exports_the_v2_surface_object() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = b"import json\nimport math\nimport sys\nfrom typing import Annotated\n\nfrom pydantic import Field\n\nSample = Annotated[float, Field(ge=-2.0, le=2.0)]\nLevel = Annotated[float, Field(ge=0.0, le=1.0)]\n\n\ndef level_from_file(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n    clamped = [max(-1.0, min(1.0, s)) for s in samples]\n    total = sum(s * s for s in clamped)\n    return math.sqrt(total / len(samples))\n\n\nif __name__ == \"__main__\":\n    with open(sys.argv[1]) as payload_file:\n        payload = json.load(payload_file)\n    print(json.dumps(level_from_file(payload)))\n";
    let text = String::from_utf8(source.to_vec()).expect("the fixture is UTF-8");
    let module = ruff_python_parser::parse_module(&text)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source, "level_from_file.py", no_imports, &kernel, None);
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    assert_eq!(
        artifact["surface"],
        json!({
            "kind": "file-json",
            "argIndex": 1,
            "stdout": "json",
            "calls": "level_from_file",
        })
    );
}
