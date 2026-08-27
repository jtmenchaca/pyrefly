use super::*;

/// A plain numeric return exports the RULED cases schema exactly: a
/// one-element `cases` array, its sort "number", carrying the full
/// kernel wire set. `Age`'s own `Annotated[...]` alias is what gives
/// `x`/the return their refined set to derive (`derived_return_values`
/// walks every module regardless of vocabulary; without `Age` here
/// there would be nothing refined for the case to state).
#[test]
fn a_numeric_return_exports_one_number_case() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(x: Age) -> Age:\n",
        "    return x\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "f.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    assert!(!artifact["refined"].as_object().unwrap().contains_key("version"), "no version field, ever");
    let cases = artifact["functions"]["f"]["return"]["cases"].as_array().expect("return states cases");
    assert_eq!(cases.len(), 1, "a plain numeric return states exactly one case");
    assert_eq!(cases[0]["sort"], "number");
    assert!(cases[0].get("set").is_some(), "a number case carries the full kernel wire set");
}

/// An `Optional[...]`-declared RETURN whose body's own join derives
/// `Kind::PossiblyUndefined` exports the INNER case(s) PLUS the null
/// case — the omission "a possibly-absent value has no faithful set
/// reading" is retired by the cases schema: absence now crosses as
/// wire-honest `{"sort": "null"}` alongside the value's own case,
/// rather than dropping the whole def from the artifact.
#[test]
fn an_optional_return_exports_the_inner_case_plus_null() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated, Optional\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def maybe_level(x: Age) -> Optional[Age]:\n",
        "    if x > 0:\n",
        "        return x\n",
        "    return None\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "maybe_level.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    let rendered = serde_json::to_string_pretty(&Value::Object(artifact.clone())).expect("the artifact renders");
    eprintln!("emitted artifact for an Optional[Age] return:\n{rendered}");
    let cases = artifact["functions"]["maybe_level"]["return"]["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("maybe_level must export a return cases list, artifact: {rendered}"));
    assert!(cases.len() >= 2, "a possibly-absent return states its inner case(s) plus null: {cases:?}");
    assert!(
        cases.iter().any(|case| case["sort"] == "null"),
        "a possibly-absent return must carry a null case: {cases:?}"
    );
    assert!(
        cases.iter().any(|case| case["sort"] == "number"),
        "a possibly-absent return over an int body must still carry its inner number case: {cases:?}"
    );
}

/// A `TypedDict`-declared return whose body builds and returns a dict
/// literal exports ONE object case: `members` carries the literal's
/// own key ('age', its declared `Age` set) and `closed: true` — a
/// dict literal always derives `Kind::Object` with `complete: true`
/// (`dict_literal_value`'s own construction: a literal states every
/// key it has, no others possible). Pinned VERBATIM per the mission's
/// own ask: the whole `cases` array this function's return states.
#[test]
fn a_typed_dict_return_exports_one_object_case_with_its_members_and_closed_true() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class PersonDict(TypedDict):\n",
        "    age: Age\n",
        "def make_person(x: Age) -> PersonDict:\n",
        "    return {\"age\": x}\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "make_person.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    let rendered = serde_json::to_string_pretty(&Value::Object(artifact.clone())).expect("the artifact renders");
    eprintln!("emitted artifact for a TypedDict return:\n{rendered}");
    let cases = artifact["functions"]["make_person"]["return"]["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("make_person must export a return cases list, artifact: {rendered}"));
    assert_eq!(cases.len(), 1, "a single-shape TypedDict return states exactly one object case: {cases:?}");
    assert_eq!(cases[0]["sort"], "object");
    assert_eq!(cases[0]["closed"], true, "a dict literal's own completeness states every key it has");
    let members = cases[0]["members"].as_object().expect("the object case states its members");
    assert_eq!(members.len(), 1, "the literal states exactly one key: {members:?}");
    let age_cases = members["age"].as_array().expect("'age' states its own cases list");
    assert_eq!(age_cases.len(), 1, "'age' is a plain Age-declared int, one case");
    assert_eq!(age_cases[0]["sort"], "number");
    assert!(
        age_cases[0].get("set").is_some(),
        "'age' carries the full kernel wire set for its Age declaration"
    );
}

/// A `TypedDict`-declared PARAMETER exports an entry object case with
/// the members the class states — the entry-side twin of
/// `a_typed_dict_return_exports_one_object_case_with_its_members_and_
/// closed_true`. Ledger 210: a TypedDict-annotated parameter never
/// reached an object case before `entry_rows` threaded `typed_dicts`
/// through the same three-reader fallback chain the return side
/// already ran.
#[test]
fn a_typed_dict_parameter_exports_an_entry_object_case_with_its_members() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated, TypedDict\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class PersonDict(TypedDict):\n",
        "    age: Age\n",
        "def person_age(person: PersonDict) -> Age:\n",
        "    return person[\"age\"]\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "person_age.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    let rendered = serde_json::to_string_pretty(&Value::Object(artifact.clone())).expect("the artifact renders");
    eprintln!("emitted artifact for a TypedDict parameter:\n{rendered}");
    let entry = artifact["functions"]["person_age"]["entry"]
        .as_array()
        .unwrap_or_else(|| panic!("person_age must export an entry list, artifact: {rendered}"));
    assert_eq!(entry.len(), 1, "person_age states one parameter: {entry:?}");
    assert_eq!(entry[0]["name"], "person");
    let cases = entry[0]["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("'person' must state its own cases list, entry: {entry:?}"));
    assert_eq!(cases.len(), 1, "a single-shape TypedDict parameter states exactly one object case: {cases:?}");
    assert_eq!(cases[0]["sort"], "object");
    assert_eq!(cases[0]["closed"], true, "a TypedDict declaration states its complete key set");
    let members = cases[0]["members"].as_object().expect("the object case states its members");
    assert_eq!(members.len(), 1, "the class declares exactly one member: {members:?}");
    let age_cases = members["age"].as_array().expect("'age' states its own cases list");
    assert_eq!(age_cases.len(), 1, "'age' is a plain Age-declared int, one case");
    assert_eq!(age_cases[0]["sort"], "number");
    assert!(
        age_cases[0].get("set").is_some(),
        "'age' carries the full kernel wire set for its Age declaration"
    );
}

/// A Result-style two-branch return (`{"ok": true, "value": …}` on one
/// arm, `{"ok": false, "error": …}` on the other) exports TWO object
/// cases in the one cases list — never one case with a nested
/// `variants` field. The join (`lattice_operations.rs::join_known`'s
/// `Kind::Object` arm) builds a `Kind::Object` carrying BOTH full
/// shapes on its own `variants` field precisely because the two
/// branches' key sets differ, and `object_cases_of` reads one case per
/// variant.
#[test]
fn a_result_shaped_two_branch_return_exports_two_object_cases() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def make_result(x: Age) -> dict:\n",
        "    if x > 0:\n",
        "        return {\"ok\": True, \"value\": x}\n",
        "    return {\"ok\": False, \"error\": x}\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "make_result.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    let rendered = serde_json::to_string_pretty(&Value::Object(artifact.clone())).expect("the artifact renders");
    eprintln!("emitted artifact for a Result-shaped return:\n{rendered}");
    let cases = artifact["functions"]["make_result"]["return"]["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("make_result must export a return cases list, artifact: {rendered}"));
    assert_eq!(cases.len(), 2, "a two-branch Result-style return states two object cases: {cases:?}");
    for case in cases {
        assert_eq!(case["sort"], "object");
        let members = case["members"].as_object().expect("each object case states its members");
        assert!(members.contains_key("ok"), "every branch carries its own 'ok' key: {members:?}");
    }
    let has_value_branch = cases.iter().any(|case| case["members"].as_object().unwrap().contains_key("value"));
    let has_error_branch = cases.iter().any(|case| case["members"].as_object().unwrap().contains_key("error"));
    assert!(has_value_branch, "one branch must carry 'value': {cases:?}");
    assert!(has_error_branch, "the other branch must carry 'error': {cases:?}");
}

/// An object the domain genuinely cannot enumerate the members of (an
/// opaque/unknown object, never guessed at) stays an OMISSION naming
/// the construct — the object case never approximates a shape this
/// checker did not derive.
#[test]
fn an_unenumerable_object_return_stays_an_omission_naming_the_construct() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(x: Age) -> dict:\n",
        "    return globals()\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "f.py", no_imports, &kernel, None);
    assert!(
        !export.artifact["functions"].as_object().unwrap().contains_key("f"),
        "an unenumerable object return must not export a guessed cases list"
    );
    let reason = export
        .omissions
        .iter()
        .find(|omission| omission.function == "f")
        .unwrap_or_else(|| panic!("'f' must be named in an omission, got: {:?}", export.omissions.iter().map(|o| (&o.function, &o.reason)).collect::<Vec<_>>()));
    eprintln!("omission for an unenumerable object return: {}", reason.reason);
}

/// A `Literal[...]`-declared entry parameter exports its own cases —
/// the OneOf set the literal states, read as a number case (the
/// entry's declared set carries no sequence form, so `scalar_case_of`
/// reads it numeric).
#[test]
fn a_literal_set_entry_exports_its_cases() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated, Literal\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(level: Literal[1, 2, 4]) -> Age:\n",
        "    return level\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "f.py", no_imports, &kernel, None);
    for omission in &export.omissions {
        eprintln!("omission: '{}' — {}", omission.function, omission.reason);
    }
    let artifact = export.artifact.as_object().expect("the artifact is an object");
    let entry = artifact["functions"]["f"]["entry"].as_array().expect("entry is an array");
    assert_eq!(entry.len(), 1);
    assert_eq!(entry[0]["name"], "level");
    let cases = entry[0]["cases"].as_array().expect("the Literal entry states cases");
    assert_eq!(cases.len(), 1, "a plain Literal[...] entry (no None arm) states one case");
    assert_eq!(cases[0]["sort"], "number");
    let forms = cases[0]["set"]["forms"].as_array().expect("the case's set carries forms");
    assert!(!forms.is_empty(), "the Literal[1, 2, 4] entry states a non-empty OneOf set");
}

/// `["ok", "warn", "error"][code]` where `code` is a BOUNDED (not
/// exact) integer index — the join over all three positions builds a
/// top-level `Union` of `Concatenation` forms, which `states_
/// sequence`'s own non-recursive top-layer check alone misreads as
/// numeric (its top form IS `Union`, never `Concatenation` itself).
/// This pins the export's own return case at `"sort": "string"`,
/// never `"number"`, for exactly the shape `text_status.py`'s
/// `make_status` derives.
#[test]
fn export_reads_a_joined_string_list_index_as_a_string_case_not_a_number_case() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = "from typing import Annotated\n\
         from pydantic import Field\n\
         \n\
         def make_status(code: Annotated[int, Field(ge=0, le=2)]) -> str:\n\
         \x20   return [\"ok\", \"warn\", \"error\"][code]\n";
    let module = ruff_python_parser::parse_module(source).expect("test module parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let export = export_module(&module, source.as_bytes(), "text_status.py", no_imports, &kernel, None);
    assert!(
        export.omissions.is_empty(),
        "make_status must export cleanly, got omissions: {:?}",
        export.omissions.iter().map(|o| (&o.function, &o.reason)).collect::<Vec<_>>()
    );
    let entry = export
        .artifact
        .get("functions")
        .and_then(|functions| functions.get("make_status"))
        .expect("make_status must be present in the artifact");
    let sort = entry
        .get("return")
        .and_then(|value| value.get("cases"))
        .and_then(|cases| cases.get(0))
        .and_then(|case| case.get("sort"))
        .and_then(Value::as_str);
    assert_eq!(sort, Some("string"), "artifact: {entry:?}");
}
