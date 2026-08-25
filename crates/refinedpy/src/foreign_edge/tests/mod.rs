use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use refined_domain::abstract_value::possibly_nan;
use refined_domain::trust_grades::TrustProved;
use crate::collection_models::subscript_read;
use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::star;
use refined_sets::repetition_window_forms::repetition;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;

use super::*;
use crate::foreign_edge_artifact::ForeignCase;
use crate::foreign_edge_artifact::ForeignSurface;
use crate::foreign_edge_artifact::ForeignTsArtifact;
use crate::foreign_edge_artifact::ForeignTsEntry;
use crate::foreign_edge_artifact::ForeignTsFunctionFact;

use super::crossing::foreign_scalar_subset;

mod recognize;
mod argv;
mod runner;
mod parse_consumer;
mod crossing;
mod cases;

thread_local! {
    static FIXTURE_ARTIFACTS: RefCell<HashMap<String, ForeignTsArtifact>> = RefCell::new(HashMap::new());
}

/// Registers a fixture artifact under `target_path` for
/// `read_foreign_ts_artifact`'s test stub to answer — the in-process
/// stand-in for the sibling's disk-backed reader, so this module's
/// own recognizer/premise logic is exercised without depending on
/// `foreign_edge_artifact.rs`'s landed shape.
pub(super) fn register_fixture_artifact(target_path: &str, artifact: ForeignTsArtifact) {
    FIXTURE_ARTIFACTS.with(|cell| cell.borrow_mut().insert(target_path.to_owned(), artifact));
}

pub(super) fn test_read_foreign_ts_artifact(target_path: &str) -> Result<ForeignTsArtifact, String> {
    FIXTURE_ARTIFACTS.with(|cell| {
        cell.borrow()
            .get(target_path)
            .cloned()
            .ok_or_else(|| format!("there is no artifact for {target_path}"))
    })
}

pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

pub(super) fn parsed_body(source: &str) -> Vec<Stmt> {
    ruff_python_parser::parse_module(source).expect("fixture source parses").into_syntax().body.to_vec()
}

pub(super) fn env_with(bindings: &[(&str, AbstractValue)]) -> Environment {
    let mut environment = Environment::new(HashSet::new());
    for (name, value) in bindings {
        environment.bind(name, value.clone());
    }
    environment
}

pub(super) fn boosted_sequence_value() -> AbstractValue {
    known_set(
        repetition(make_refined_set(vec![at_least(-2.0), at_most(2.0)]), 1, None),
        None,
        TrustProved,
        SetKindTag::None,
    )
}

pub(super) fn audio_level_ts_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        path: PathBuf::from("./audio_level.ts.refined.json"),
        called: ForeignTsFunctionFact {
            name: "audioLevel".to_owned(),
            entry: vec![ForeignTsEntry {
                name: "boosted".to_owned(),
                sequence: Some((
                    vec![ForeignCase::Number(make_refined_set(vec![at_least(-2.0), at_most(2.0)]))],
                    1,
                )),
                scalar: None,
            }],
            return_cases: vec![ForeignCase::Number(make_refined_set(vec![
                integer(),
                at_least(0.0),
                at_most(1.0),
            ]))],
            stdout_pure: true,
            provenance_line: 30,
            provenance_said: "audioLevel's own kernel summary".to_owned(),
        },
        target_file: "./audio_level.ts".to_owned(),
        runtime_band: "es2023+".to_owned(),
        surface: ForeignSurface::StdinJson,
    }
}

/// The same fact, on an `argv-json` target reading its payload at
/// `argv[2]` — the fixture the argv-payload tests register.
pub(super) fn audio_level_argv_json_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact { surface: ForeignSurface::ArgvJson { arg_index: 2 }, ..audio_level_ts_artifact() }
}

/// The same fact, with an unbounded `atLeast` return — the derived
/// window a `Math.max(0, x)`-shaped target's own kernel summary
/// states, admitting +Infinity with no literal spelling needed
/// (the corner-check fixture: `foreign_edge.rs:181`'s Go-twin
/// grounding for the identical premise).
pub(super) fn audio_level_unbounded_return_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        called: ForeignTsFunctionFact {
            return_cases: vec![ForeignCase::Number(make_refined_set(vec![at_least(0.0)]))],
            ..audio_level_ts_artifact().called
        },
        ..audio_level_ts_artifact()
    }
}

/// The same fact, with a float-sorted (no `Integer` form) finite
/// return window — the sibling of the int-sorted default fixture,
/// used to pin that an unmarked numeric return still reads Float.
pub(super) fn audio_level_float_return_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        called: ForeignTsFunctionFact {
            return_cases: vec![ForeignCase::Number(make_refined_set(vec![at_least(0.0), at_most(1.0)]))],
            ..audio_level_ts_artifact().called
        },
        ..audio_level_ts_artifact()
    }
}

/// The same fact, with an all-integer `OneOf` return — the shape
/// `union_levels.ts`'s derived `{1, 2, 4}` Literal-set return
/// carries (f-value-unions.py's own `louder_level_wider_window`
/// pin): no explicit `Integer` form, but every admitted value is a
/// whole number.
pub(super) fn audio_level_one_of_integer_return_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        called: ForeignTsFunctionFact {
            return_cases: vec![ForeignCase::Number(make_refined_set(vec![one_of(&[1.0, 2.0, 4.0])]))],
            ..audio_level_ts_artifact().called
        },
        ..audio_level_ts_artifact()
    }
}

/// The same fact, with a single CLOSED, empty-member OBJECT return
/// case — pins `known_object`'s own shape for the plainest object
/// case (`foreign_case_value`'s own Object arm): no members, and
/// `complete: true` straight from `closed`.
pub(super) fn audio_level_object_return_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        called: ForeignTsFunctionFact {
            return_cases: vec![ForeignCase::Object { members: vec![], closed: true }],
            ..audio_level_ts_artifact().called
        },
        ..audio_level_ts_artifact()
    }
}

/// The Result-shape return: two OBJECT cases in one return list —
/// `{"ok": bool, "value": number in [0, 1]}` and `{"ok": bool,
/// "error": string}` — lowering through the same multi-case
/// `Kind::KindUnion` channel a scalar multi-case return already uses
/// (`foreign_return_value`'s own doc).
pub(super) fn audio_level_result_shape_return_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        called: ForeignTsFunctionFact {
            return_cases: vec![
                ForeignCase::Object {
                    members: vec![
                        ("ok".to_owned(), vec![ForeignCase::Boolean]),
                        (
                            "value".to_owned(),
                            vec![ForeignCase::Number(make_refined_set(vec![at_least(0.0), at_most(1.0)]))],
                        ),
                    ],
                    closed: true,
                },
                ForeignCase::Object {
                    members: vec![
                        ("ok".to_owned(), vec![ForeignCase::Boolean]),
                        (
                            "error".to_owned(),
                            vec![ForeignCase::String(make_refined_set(vec![]))],
                        ),
                    ],
                    closed: true,
                },
            ],
            ..audio_level_ts_artifact().called
        },
        ..audio_level_ts_artifact()
    }
}

/// The same fact, with an OBJECT case at the ENTRY (outbound) leg
/// instead of the return — `admitted_set_of_cases`'s own Object arm,
/// a designed remainder distinct from the return-side lowering.
pub(super) fn audio_level_object_entry_artifact() -> ForeignTsArtifact {
    ForeignTsArtifact {
        called: ForeignTsFunctionFact {
            entry: vec![ForeignTsEntry {
                name: "boosted".to_owned(),
                sequence: None,
                scalar: Some(vec![ForeignCase::Object { members: vec![], closed: true }]),
            }],
            ..audio_level_ts_artifact().called
        },
        ..audio_level_ts_artifact()
    }
}

pub(super) const FIXTURE_SOURCE: &str = concat!(
    "def audio_level_via_ts(boosted):\n",
    "    result = subprocess.run(\n",
    "        [\"node\", \"./audio_level.ts\"],\n",
    "        input=json.dumps(boosted),\n",
    "        capture_output=True,\n",
    "        text=True,\n",
    "    )\n",
    "    return json.loads(result.stdout)\n",
);

pub(super) fn def_body(source: &str) -> Vec<Stmt> {
    let module = parsed_body(source);
    let Stmt::FunctionDef(def) = module.into_iter().next().expect("one top-level def") else {
        panic!("fixture source must be a single def");
    };
    def.body.to_vec()
}

/// Whether `literal` is admitted by `set` — a single-string
/// singleton (`string_tuple`) asked against `set` through the
/// kernel's own `seq_subset` decider, the same routing `foreign_
/// scalar_subset` uses for a string-shaped pair. Panics on a kernel
/// refusal: every literal this module's own tests ask about sits
/// squarely inside the JSON-number grammar's supported shape, so a
/// refusal would itself be the finding, not a reason to skip.
pub(super) fn literal_string_admitted_by(kernel: &Arc<RefinedTSKernel>, literal: &str, set: &RefinedSet) -> bool {
    let singleton = refined_sets::codepoint_sets::string_tuple(literal);
    crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(&singleton, set))
        .expect("seq_subset decides a literal singleton against the JSON-number grammar")
}

pub(super) const ARGV_JSON_FIXTURE_SOURCE: &str = concat!(
    "def audio_level_via_argv(boosted):\n",
    "    result = subprocess.run(\n",
    "        [\"node\", \"./audio_level.ts\", json.dumps(boosted)],\n",
    "        capture_output=True,\n",
    "        text=True,\n",
    "    )\n",
    "    return json.loads(result.stdout)\n",
);

pub(super) const TEMP_FILE_FIXTURE_SOURCE: &str = concat!(
    "def audio_level_via_temp_file(boosted):\n",
    "    with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\", delete=False) as handle:\n",
    "        json.dump(boosted, handle)\n",
    "        temp_path = handle.name\n",
    "    result = subprocess.run(\n",
    "        [\"node\", \"./audio_level.ts\", temp_path],\n",
    "        capture_output=True,\n",
    "        text=True,\n",
    "    )\n",
    "    return json.loads(result.stdout)\n",
);

/// A fresh temp directory (unique per test run), mirroring
/// `foreign_edge_artifact.rs`'s own `temp_project_root` convention —
/// a compiled binary's fact reads from a SIBLING file discovered by
/// path alone (`compiled_binary_fact_path`), so no `.git` marker is
/// needed here (unlike the TypeScript reader's project-cache walk).
pub(super) fn temp_binary_dir(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "refinedpy_foreign_edge_compiled_binary_test_{label}_{}_{}",
        std::process::id(),
        SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
    ));
    fs::create_dir_all(&root).expect("create temp binary dir");
    root
}

/// A well-formed compiled-binary fact file's JSON — the triangle's
/// own `cpp_level` contract in miniature: one sequence entry (numbers
/// -2.0…2.0, at least one element) and a number 0.0…1.0 return,
/// through the identical cases-schema envelope
/// `audio_level_ts_artifact` builds for a TypeScript target, with
/// `language: "cpp"` in place of `"typescript"` and the compiled
/// runtime band in place of the JS one.
pub(super) fn compiled_binary_fact_json(called: &str) -> serde_json::Value {
    use refined_kernel::wire_format::wire_set;
    use serde_json::json;
    let entry_set = make_refined_set(vec![at_least(-2.0), at_most(2.0)]);
    let return_set = make_refined_set(vec![at_least(0.0), at_most(1.0)]);
    json!({
        "refined": {"kind": "fact-artifact"},
        "language": "cpp",
        "runtime": {"band": "c++17"},
        "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": called},
        "functions": {
            called: {
                "entry": [{"name": "samples", "sequence": {"element": {"cases": [
                    {"sort": "number", "set": wire_set(&entry_set)}
                ]}, "lengthAtLeast": 1}}],
                "return": {"cases": [{"sort": "number", "set": wire_set(&return_set)}], "stdoutPure": true},
                "provenance": {"line": 35, "said": "level's own derivation: clamp, mean of squares, sqrt"},
            }
        }
    })
}

/// The exact code-point-vector shape a known string constant carries
/// — the same shape `exact_string_text` decodes, built directly here
/// (this test module has no import rights into `string_models.rs`'s
/// own `string_literal_value`).
pub(super) fn string_literal_value_for_test(text: &str) -> AbstractValue {
    known_values(text.chars().map(|c| c as u32 as f64).collect(), PrimitiveKind::String, TrustProved)
}

/// The walrus's own `Expr::Named::target`/`value` pair, read off the
/// first statement's own `if`-test — the exact destructuring
/// `walk_if`'s `serve_foreign_edge_in_walrus_test` (check.rs) performs
/// before calling `foreign_edge_at_walrus_call`, rebuilt here so this
/// test drives that same entry point directly rather than through the
/// checker's own statement walk (per this module's own artifact-stub
/// constraint: only a test living here can observe an `Override`).
pub(super) fn walrus_test_target_and_call(if_stmt: &ruff_python_ast::StmtIf) -> (ExprName, ExprCall) {
    let Expr::Compare(compare) = if_stmt.test.as_ref() else {
        panic!("this fixture's own if-test must be a comparison wrapping the walrus");
    };
    let Expr::Attribute(attribute) = compare.left.as_ref() else {
        panic!("this fixture's own if-test must read an attribute off the walrus-bound name");
    };
    let Expr::Named(named) = attribute.value.as_ref() else {
        panic!("this fixture's own if-test must embed a walrus binding");
    };
    let Expr::Name(target) = named.target.as_ref() else {
        panic!("the walrus target must be a bare name");
    };
    let Expr::Call(call) = named.value.as_ref() else {
        panic!("the walrus value must be a call");
    };
    (target.clone(), call.clone())
}
