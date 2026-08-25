//! One module's fact-artifact export: the top-level `export_module` entry
//! point, the per-def `export_function` reader it calls, the cheap
//! `has_exportable_defs` gate the save hook checks before paying for the
//! full walk, and the `cached_hash_matches` content-hash short-circuit.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;

use crate::check::derived_return_values_at;
use crate::cross_module::ModuleResolver;
use crate::env::Environment;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;
use crate::surface::compile_aliases;
use crate::surface::surface_imports;
use crate::typereading::DeclaredRefinement;
use crate::typereading::base_sort_return_refinement;
use crate::typereading::declared_refinement;
use crate::typereading::typed_dict_return_refinement;

use super::ARTIFACT_KIND;
use super::ARTIFACT_LANGUAGE;
use super::Export;
use super::Omission;
use super::RUNTIME_BAND;
use super::cases;
use super::cases::cases_json;
use super::cases::return_cases;
use super::entry;
use super::entry::EntryRow;
use super::entry::entry_row_json;
use super::entry::entry_rows;
use super::harness::harness_shape;
use super::harness::harness_shape_json;
use super::positions::line_of;
use super::positions::line_starts_of;
use super::sha256::sha256_hex;
use super::stdout_purity::writes_nothing_to_stdout;

/// `module`'s fact artifact — every top-level `def` with a fully
/// declared refined entry and a derivable return set, in source order.
///
/// `source_bytes` is the target file's exact content (the bytes the hash
/// commits to, and the bytes `module` was parsed from); `basename` is
/// what `target.file` states. `resolver` is the same import resolver the
/// checker's own CLI passes, so a def reading an imported name derives
/// exactly what the checker derives for it. `entry_directory` is the
/// checked file's own directory — threaded into
/// `derived_return_values_at` so a relative foreign-edge target (a
/// `subprocess.run(["node", "./audio_level.ts"], ...)` call) joins
/// against it before the artifact read, exactly as an ordinary check
/// already joins it (`findings_for_module_at`). `None` leaves a relative
/// target unjoined, the same as `derived_return_values`'s own default.
pub fn export_module(
    module: &ModModule,
    source_bytes: &[u8],
    basename: &str,
    resolver: ModuleResolver,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&Path>,
) -> Export {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    // The same per-class member table `check.rs::findings_for_module_
    // with_resolver` builds (`instances::typed_dict_table`) — read here
    // too so a parameter annotated with a TypedDict class name reaches
    // the object case exactly as a TypedDict-declared RETURN already
    // does (`entry_rows`'s own three-reader fallback chain).
    let typed_dicts = crate::instances::typed_dict_table(module, &aliases, &imports);
    // ONE walk of the whole module answers every def's derived return:
    // the shared context (imports resolved, function/class tables built)
    // costs the same whether one def asks or all of them do.
    let derived_returns = derived_return_values_at(module, resolver, kernel, entry_directory);
    let mut functions = Map::new();
    let mut omissions = Vec::new();
    let module_line_starts = line_starts_of(source_bytes);

    for def in top_level_defs(module) {
        let name = def.name.id.as_str().to_owned();
        match export_function(
            def,
            module,
            &module_line_starts,
            &aliases,
            &imports,
            &typed_dicts,
            &derived_returns.values,
            &derived_returns.blockers,
        ) {
            Ok(entry) => {
                functions.insert(name, entry);
            }
            Err(reason) => omissions.push(Omission { function: name, reason }),
        }
    }

    let mut artifact = Map::new();
    // NO version field, ever — the identity marker names the kind only
    // (see ARTIFACT_KIND's own doc).
    artifact.insert("refined".to_owned(), json!({"kind": ARTIFACT_KIND}));
    artifact.insert(
        "target".to_owned(),
        json!({"file": basename, "contentHash": format!("sha256:{}", sha256_hex(source_bytes))}),
    );
    artifact.insert("language".to_owned(), json!(ARTIFACT_LANGUAGE));
    artifact.insert("runtime".to_owned(), json!({"band": RUNTIME_BAND}));
    if let Some(shape) = harness_shape(module) {
        // A surface is exported only when the module's `__main__` block
        // IS one of the two recognized shapes; absence of the key is the
        // consumer's "no surface fact" (§11's own reading), never a
        // guessed default. `kind` is v2's tagged union — `stdin-json` and
        // `argv-scalar` are the two surface kinds this producer states
        // today.
        artifact.insert("surface".to_owned(), harness_shape_json(&shape));
    }
    artifact.insert("functions".to_owned(), Value::Object(functions));

    Export {
        artifact: Value::Object(artifact),
        omissions,
    }
}

/// Whether `module` has ANY top-level `def` this export could carry —
/// the save-hook's own gate (docs/one-checker/fact-freshness.md, "Cheap
/// gate: shallow scan for annotated top-level defs before the full
/// walk"). Cheap on purpose: it reads each parameter's own annotation
/// exactly as `entry_rows` does, but never calls `derived_return_values`
/// (the kernel walk `export_module` pays for every def whether or not
/// this gate would have skipped the module) and never checks
/// `entry_shape`'s container-specific reading (a `list[X]` with no
/// crossable element still passes this gate — the full export is what
/// finds that and omits the def; this gate only answers "worth trying").
///
/// A `false` answer means `export_module` would omit every def in the
/// module (every parameter unannotated, or every annotation unreadable,
/// or the module declaring no top-level def at all) — the save hook
/// skips the full walk rather than paying it for an artifact that would
/// carry no functions.
pub fn has_exportable_defs(module: &ModModule) -> bool {
    let aliases = compile_aliases(module);
    let imports = surface_imports(module);
    let environment = Environment::new(HashSet::new());
    let typed_dicts = crate::instances::typed_dict_table(module, &aliases, &imports);
    top_level_defs(module)
        .any(|def| def_has_a_declared_entry(def, &aliases, &imports, &environment, &typed_dicts))
}

/// Whether `def` states at least one parameter this table can read a
/// refinement from, and carries no `*args`/`**kwargs` tail — the same
/// two obstacles `entry_rows` declines on, checked here without building
/// the `EntryRow` vector or reading `entry_shape`. `typed_dicts` is read
/// the same way `entry_rows` reads it — a bare name naming a recorded
/// TypedDict class counts as a declared entry here too, so this cheap
/// gate never skips a module whose only exportable def takes a
/// TypedDict-typed parameter.
fn def_has_a_declared_entry(
    def: &StmtFunctionDef,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    environment: &Environment,
    typed_dicts: &HashMap<String, Vec<(String, DeclaredRefinement)>>,
) -> bool {
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() {
        return false;
    }
    let parameters: Vec<_> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
        .collect();
    if parameters.is_empty() {
        return false;
    }
    parameters.iter().all(|parameter| {
        let Some(annotation) = parameter.parameter.annotation.as_deref() else {
            return false;
        };
        declared_refinement(annotation, aliases, imports, environment).is_some()
            || base_sort_return_refinement(annotation).is_some()
            || typed_dict_return_refinement(annotation, typed_dicts).is_some()
    })
}

/// Reads the cached artifact at `artifact_path`, when one is present,
/// and answers whether its `target.contentHash` equals sha256 of
/// `source_bytes` — the save hook's content-hash short-circuit
/// (docs/one-checker/fact-freshness.md, "Content-hash short-circuit":
/// skip the export when the cache already states the fact for these
/// exact bytes). `false` for a missing, unreadable, or malformed cache
/// entry — the caller re-exports rather than trusting a cache it cannot
/// read, exactly the same "an artifact is a file, not a promise"
/// discipline `foreign_edge_artifact.rs` applies to a foreign artifact.
pub fn cached_hash_matches(artifact_path: &Path, source_bytes: &[u8]) -> bool {
    let Ok(raw) = std::fs::read(artifact_path) else {
        return false;
    };
    let Ok(parsed) = serde_json::from_slice::<Value>(&raw) else {
        return false;
    };
    let Some(stated) = parsed.get("target").and_then(|target| target.get("contentHash")).and_then(Value::as_str)
    else {
        return false;
    };
    stated == format!("sha256:{}", sha256_hex(source_bytes))
}

/// Every top-level `def` in `module`, in source order.
pub(crate) fn top_level_defs(module: &ModModule) -> impl Iterator<Item = &StmtFunctionDef> {
    module.body.iter().filter_map(|stmt| match stmt {
        Stmt::FunctionDef(def) => Some(def),
        _ => None,
    })
}

/// One def's exported entry, or the reason it cannot be exported.
///
/// Every field is derived here and nothing is defaulted: the entry from
/// the declarations the walk itself seeds from, the return from the walk
/// itself, `stdoutPure` from a scan of the body and the same-module defs
/// it calls, and `said` from the same diagnostics formatter every
/// refinement sentence in this checker is spelled through.
fn export_function(
    def: &StmtFunctionDef,
    module: &ModModule,
    line_starts: &[usize],
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    typed_dicts: &HashMap<String, Vec<(String, DeclaredRefinement)>>,
    derived_returns: &HashMap<String, refined_domain::abstract_value::AbstractValue>,
    derived_blockers: &HashMap<String, String>,
) -> Result<Value, String> {
    let entry = entry_rows(def, aliases, imports, typed_dicts)?;
    let name = def.name.id.as_str();
    let returned = derived_returns.get(name).ok_or_else(|| {
        // A body whose own walk hit an unwalkable construct names THAT
        // construct — the same RTS7002 sentence `findings_for_module`
        // would report for this body, independent of whether `->
        // Annotation` itself read (`derived_return_values`'s own doc:
        // an unreadable return annotation must never leave this body's
        // omission unnamed). Falls back to the generic sentence only
        // for a body the walk ran cleanly through with no blocker and
        // no `return` at all — genuinely nothing to name.
        derived_blockers
            .get(name)
            .cloned()
            .unwrap_or_else(|| "the body's returns derived no value the walk could read".to_owned())
    })?;
    let return_cases = return_cases(returned)?;
    let stdout_pure = writes_nothing_to_stdout(def, module);
    // The def's own NAME identifier, not the statement range: a
    // decorated def's statement range starts at the decorator, and the
    // line a reader means by "the def line" is the one `def <name>` sits
    // on. The name identifier is always on that line.
    let line = line_of(line_starts, def.name.range.start().to_usize());
    let said = provenance_sentence(&entry, &return_cases);

    let entry_json: Vec<Value> = entry.iter().map(entry_row_json).collect();
    Ok(json!({
        "entry": entry_json,
        "return": {"cases": cases_json(&return_cases), "stdoutPure": stdout_pure},
        "provenance": {"line": line, "said": said},
    }))
}

/// The one sentence `provenance.said` states, assembled from the facts
/// the artifact already carries — each entry bound and the derived
/// return, spelled through `format_for_diagnostics` (the same formatter
/// every refinement sentence in this checker is spelled through) for a
/// set-carrying case, and plain words for a boolean/null case.
fn provenance_sentence(entry: &[EntryRow], return_cases: &[cases::Case]) -> String {
    let entry_words: Vec<String> = entry
        .iter()
        .map(|row| match &row.shape {
            entry::EntryShape::Sequence {
                element,
                length_at_least,
            } => format!(
                "'{}' whose every element is {} and whose length is at least {}",
                row.name,
                cases::cases_words(element),
                length_at_least
            ),
            entry::EntryShape::Scalar(cases) => {
                format!("'{}' is {}", row.name, cases::cases_words(cases))
            }
        })
        .collect();
    format!(
        "given {}, this body's returns derive {}",
        entry_words.join(" and "),
        cases::cases_words(return_cases)
    )
}
