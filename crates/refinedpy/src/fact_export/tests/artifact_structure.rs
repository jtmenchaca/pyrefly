use super::*;

/// The tutorial fixture exported end to end: every artifact key
/// present, the hash prefixed and full-width, and the entry row
/// carrying the sequence shape `samples`' own declaration states.
/// Pinned at the STRUCTURE, not at the set contents — what the
/// checker derives for the return is the derivation lanes' business,
/// and this test must not restate it.
#[test]
fn the_tutorial_fixture_exports_its_structure() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../fixtures/tutorial/audio_level_python_only.py"
    );
    let source = std::fs::read(path).expect("the tutorial fixture is committed beside the checker");
    let text = String::from_utf8(source.clone()).expect("the fixture is UTF-8");
    let module = ruff_python_parser::parse_module(&text)
        .expect("the fixture parses")
        .into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;

    let export = export_module(
        &module,
        &source,
        "audio_level_python_only.py",
        no_imports,
        &kernel,
        None,
    );
    let artifact = export.artifact.as_object().expect("the artifact is an object");

    assert_eq!(artifact["refined"]["kind"], ARTIFACT_KIND);
    assert!(
        artifact["refined"].as_object().expect("refined is an object").get("version").is_none(),
        "the RULED schema carries no version field, ever"
    );
    assert_eq!(artifact["target"]["file"], "audio_level_python_only.py");
    let hash = artifact["target"]["contentHash"]
        .as_str()
        .expect("contentHash is a string");
    assert!(hash.starts_with("sha256:"), "contentHash = {hash}");
    assert_eq!(hash.len(), "sha256:".len() + 64, "contentHash = {hash}");
    assert_eq!(&hash["sha256:".len()..], sha256_hex(&source).as_str());
    assert_eq!(artifact["language"], ARTIFACT_LANGUAGE);
    assert_eq!(artifact["runtime"]["band"], RUNTIME_BAND);
    // the fixture has no `__main__` block, so the surface key is
    // absent rather than guessed
    assert!(!artifact.contains_key("surface"));

    let functions = artifact["functions"]
        .as_object()
        .expect("functions is an object");
    // Every def either exports or is named in an omission — the
    // artifact never silently drops one.
    let exported: HashSet<&str> = functions.keys().map(|key| key.as_str()).collect();
    let omitted: HashSet<&str> = export
        .omissions
        .iter()
        .map(|omission| omission.function.as_str())
        .collect();
    for def in top_level_defs(&module) {
        let name = def.name.id.as_str();
        assert!(
            exported.contains(name) || omitted.contains(name),
            "'{name}' is neither exported nor named in an omission"
        );
    }

    for (name, entry) in functions {
        let rows = entry["entry"].as_array().expect("entry is an array");
        assert_eq!(rows.len(), 1, "'{name}' declares one parameter");
        let row = &rows[0];
        assert_eq!(row["name"], "samples");
        // `Annotated[list[Sample], Field(min_length=1)]` — a
        // sequence row, its element cases present and its length
        // floor the declaration's own 1.
        let sequence = row
            .get("sequence")
            .unwrap_or_else(|| panic!("'{name}' states a sequence entry"));
        assert_eq!(sequence["lengthAtLeast"], 1);
        let element_cases = sequence["element"]["cases"]
            .as_array()
            .expect("the element states its cases");
        assert_eq!(element_cases.len(), 1, "a plain element declaration states one case");
        assert!(
            !element_cases[0]["set"]["forms"]
                .as_array()
                .expect("the element case's set carries forms")
                .is_empty(),
            "'{name}' states an empty element set"
        );
        let returned = &entry["return"];
        let return_cases = returned["cases"].as_array().expect("the return states its cases");
        assert_eq!(return_cases.len(), 1, "a plain numeric return states one case");
        assert!(
            !return_cases[0]["set"]["forms"]
                .as_array()
                .expect("the return case's set carries forms")
                .is_empty(),
            "'{name}' states an empty return set"
        );
        assert!(returned["stdoutPure"].is_boolean());
        assert!(
            entry["provenance"]["line"].as_i64().expect("line is a number") > 0,
            "'{name}' states a 1-based def line"
        );
        let said = entry["provenance"]["said"].as_str().expect("said is a string");
        assert!(said.contains("samples"), "said = {said:?}");
        assert!(said.contains("derive"), "said = {said:?}");
    }
}

/// A def with one annotated parameter passes the gate; a def with no
/// parameters, an unannotated parameter, or a `*args`/`**kwargs`
/// tail does not — the same three obstacles `entry_rows` declines
/// on, checked here without the kernel walk.
#[test]
fn has_exportable_defs_reads_the_same_obstacles_entry_rows_declines_on() {
    let annotated = ruff_python_parser::parse_module("def f(x: int) -> int:\n    return x\n")
        .expect("test module parses")
        .into_syntax();
    assert!(has_exportable_defs(&annotated));

    let no_parameters = ruff_python_parser::parse_module("def f() -> int:\n    return 1\n")
        .expect("test module parses")
        .into_syntax();
    assert!(!has_exportable_defs(&no_parameters));

    let unannotated = ruff_python_parser::parse_module("def f(x) -> int:\n    return x\n")
        .expect("test module parses")
        .into_syntax();
    assert!(!has_exportable_defs(&unannotated));

    let varargs = ruff_python_parser::parse_module("def f(x: int, *args) -> int:\n    return x\n")
        .expect("test module parses")
        .into_syntax();
    assert!(!has_exportable_defs(&varargs));

    let no_defs = ruff_python_parser::parse_module("x = 1\n")
        .expect("test module parses")
        .into_syntax();
    assert!(!has_exportable_defs(&no_defs));
}

/// ONE def out of several carrying a declared entry is enough for
/// the module to pass the gate — the scan need not find every
/// exportable def, only that at least one exists.
#[test]
fn has_exportable_defs_is_true_when_only_one_def_qualifies() {
    let module = ruff_python_parser::parse_module(
        "def unreadable(x) -> int:\n    return x\n\n\ndef readable(x: int) -> int:\n    return x\n",
    )
    .expect("test module parses")
    .into_syntax();
    assert!(has_exportable_defs(&module));
}

/// A missing cache entry, an unreadable one, and one whose hash
/// differs all answer `false`; a cache entry whose
/// `target.contentHash` is the real sha256 of the given bytes
/// answers `true`.
#[test]
fn cached_hash_matches_reads_the_cached_target_content_hash() {
    let dir = std::env::temp_dir().join(format!(
        "refinedpy_fact_export_cached_hash_test_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let source = b"def f(x: int) -> int:\n    return x\n";

    let missing_path = dir.join("missing.refined.json");
    assert!(!cached_hash_matches(&missing_path, source));

    let unreadable_path = dir.join("unreadable.refined.json");
    std::fs::write(&unreadable_path, b"not json").expect("write unreadable cache entry");
    assert!(!cached_hash_matches(&unreadable_path, source));

    let stale_path = dir.join("stale.refined.json");
    std::fs::write(
        &stale_path,
        json!({"target": {"contentHash": format!("sha256:{}", sha256_hex(b"different bytes"))}}).to_string(),
    )
    .expect("write stale cache entry");
    assert!(!cached_hash_matches(&stale_path, source));

    let matching_path = dir.join("matching.refined.json");
    std::fs::write(
        &matching_path,
        json!({"target": {"contentHash": format!("sha256:{}", sha256_hex(source))}}).to_string(),
    )
    .expect("write matching cache entry");
    assert!(cached_hash_matches(&matching_path, source));

    std::fs::remove_dir_all(&dir).ok();
}

/// `export_module` WITH an `entry_directory` joins a relative
/// foreign-edge target against it before the export walk's own
/// artifact read — the export seam's own twin of
/// `check.rs`'s `derived_return_values_at_with_a_directory_joins_the_relative_target_before_declining`,
/// pinned here at the `export_function` level: the def carries a
/// declared entry (`x: Age`), so its own omission is what names the
/// JOINED path, proving `export_module` threads `entry_directory`
/// into `derived_return_values_at` rather than dropping it on the
/// way from the CLI/LSP callers into the shared walk.
#[test]
fn export_module_with_a_directory_joins_the_relative_target_before_declining() {
    let Some(kernel) = loaded_kernel() else {
        return;
    };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "\n",
        "def f(x: Age):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(x),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    parsed = json.loads(result.stdout)\n",
    );
    let module = ruff_python_parser::parse_module(source).expect("the fixture parses").into_syntax();
    let no_imports: ModuleResolver = &|_: &str| None;
    let directory = std::path::Path::new("/tmp/refinedpy-export-directory-fixture");
    let export = export_module(&module, source.as_bytes(), "f.py", no_imports, &kernel, Some(directory));
    let reason = export
        .omissions
        .iter()
        .find(|omission| omission.function == "f")
        .unwrap_or_else(|| {
            panic!(
                "'f' must be named in an omission (its foreign-edge return is undetermined), got: {:?}",
                export.omissions.iter().map(|o| (&o.function, &o.reason)).collect::<Vec<_>>()
            )
        });
    // `Path::join` keeps the source's own leading "./" verbatim
    // (foreign_edge.rs's own join), so the joined spelling is the
    // directory plus that exact relative text, not a normalized form.
    let joined = directory.join("./audio_level.ts");
    assert!(
        reason.reason.contains(&joined.to_string_lossy().into_owned()),
        "with entry_directory given, the export walk must join the target before the read: {}",
        reason.reason
    );
}
