//! refinedpy-check: the corpus judge — the same engine the LSP seam
//! runs, over files named on the command line. The twin of
//! refinedts-check-bin's fixture contract:
//!
//! - a fire matched by a `# refinedpy: expect-error` marker on the
//!   line above stays SILENT (the expectation held);
//! - an unmatched fire prints `path:line:col refinement CODE: message`;
//! - an unmatched marker prints `path:line # refinedpy: expect-error: reason`;
//! - a file that does not parse prints `path: the entry file did not parse`.
//!
//! Exit 0 when nothing prints — every expectation held.
//!
//! `--export-fact <file.py>` is the OTHER mode: instead of judging, it
//! writes the module's fact artifact (`refinedpy::fact_export`) into
//! the project cache — `<projectRoot>/.refined/cache/<relpath>.refined.json`,
//! where the project root is the nearest ancestor holding `.git` (the
//! file's own directory when none is found), the same derivation the
//! TypeScript consumer reads by. `-o <path>` overrides the location —
//! internal tooling, not consulted by any consumer. Every def with a
//! fully declared refined entry and a derivable return set is carried;
//! every def that is not is named on stderr with the construct that
//! stopped it. Exit 0 when the artifact was written.
//!
//! `--hover <file.py> [name ...]` is the THIRD mode: what the editor's
//! hover would show, from the terminal — the twin of refined-ts-go's
//! own `refinedts-hover.ts`. It drives the same two seams the LSP
//! splice drives (`refined_set_at_position` then `format_for_hover`,
//! `refinedpy_lsp::splice_refinedpy_hover`'s own pair), never a
//! reimplementation of the rendering. With names given, every
//! occurrence of each name is hovered; with no names, every
//! module-level alias name (`compile_aliases`'s own three spellings)
//! and every `def` name is hovered. Exit 0 always — a position with
//! nothing to say prints "(no refinement hover)" rather than failing.

use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use refinedpy::check::findings_for_module_at;
use refinedpy::check::refined_set_at_position;
use refinedpy::cross_module::disk_resolver;
use refinedpy::fact_export::export_module;
use refinedpy::foreign_edge_artifact::cache_artifact_path;
use refinedpy::foreign_edge_artifact::set_project_root_override;
use refinedpy::kernel_path::resolve_kernel_dylib;
use refinedpy::markers::line_col;
use refinedpy::markers::line_starts_of;
use refinedpy::markers::markers_of;
use refinedpy::surface::compile_aliases;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::format_for_hover::format_for_hover;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextSize;

/// Where one judged file's time went: reading and parsing the source,
/// the analysis walk, and — within the walk — the portion spent inside
/// kernel asks (`kernel_ask_totals` deltas), with the ask count.
struct FileTiming {
    parse_ms: f64,
    analysis_ms: f64,
    kernel_ms: f64,
    asks: u64,
}

const NO_TIME: FileTiming = FileTiming { parse_ms: 0.0, analysis_ms: 0.0, kernel_ms: 0.0, asks: 0 };

fn check_file(path: &str, kernel: &Arc<RefinedTSKernel>) -> (usize, Vec<String>, FileTiming) {
    let parse_started = std::time::Instant::now();
    let Ok(source) = std::fs::read_to_string(path) else {
        return (1, vec![format!("{path}: the entry file did not parse")], NO_TIME);
    };
    let Ok(parsed) = ruff_python_parser::parse_module(&source) else {
        return (1, vec![format!("{path}: the entry file did not parse")], NO_TIME);
    };
    let parse_ms = parse_started.elapsed().as_secs_f64() * 1000.0;
    let module = parsed.into_syntax();
    // The entry file's own parent directory is where a sibling `.py`
    // module it imports lives (`disk_resolver`'s own contract) — a bare
    // filename with no parent (a relative path in the current
    // directory) resolves against `.` instead of an empty path.
    let entry_directory = Path::new(path).parent().filter(|dir| !dir.as_os_str().is_empty());
    let resolver = disk_resolver(entry_directory.unwrap_or_else(|| Path::new(".")).to_path_buf());
    let (kernel_nanos_before, asks_before) = refinedpy::kernel_ask::kernel_ask_totals();
    let analysis_started = std::time::Instant::now();
    let findings = findings_for_module_at(
        &module,
        &resolver,
        kernel,
        Some(entry_directory.unwrap_or_else(|| Path::new("."))),
    );
    let analysis_ms = analysis_started.elapsed().as_secs_f64() * 1000.0;
    let (kernel_nanos_after, asks_after) = refinedpy::kernel_ask::kernel_ask_totals();
    let timing = FileTiming {
        parse_ms,
        analysis_ms,
        kernel_ms: (kernel_nanos_after - kernel_nanos_before) as f64 / 1e6,
        asks: asks_after - asks_before,
    };
    let markers = markers_of(&source);
    let line_starts = line_starts_of(&source);

    let mut lines_output = Vec::new();
    let mut matched_markers = vec![false; markers.len()];
    let mut printed = 0;

    for finding in &findings {
        let (line, col) = line_col(&line_starts, usize::from(finding.range.start()));
        // `Marker::covers` is the one place the RTS7002 exclusion and
        // the optional code narrowing both apply — a marker matching
        // 7002 would silently swallow the row saying nothing was
        // determined, faking progress; RTS7002 always prints.
        let matched = markers
            .iter()
            .enumerate()
            .find(|(_, m)| m.expected_line == line && m.covers(finding.code));
        if let Some((index, _)) = matched {
            matched_markers[index] = true;
            continue;
        }
        printed += 1;
        lines_output.push(format!(
            "{path}:{line}:{col} refinement {}: {}",
            finding.code, finding.message
        ));
    }

    for (index, marker) in markers.iter().enumerate() {
        if matched_markers[index] {
            continue;
        }
        printed += 1;
        lines_output.push(format!(
            "{path}:{} # refinedpy: expect-error: {}",
            marker.marker_line,
            if marker.reason.is_empty() {
                "an error was expected on the next line, and none was reported"
            } else {
                &marker.reason
            }
        ));
    }

    (printed, lines_output, timing)
}

/// One name to hover, at one byte offset — `label` is what prints in
/// the `=== label (file:line:col) ===` header (the queried name, or
/// the alias/def name discovered without one).
struct HoverQuery {
    label: String,
    position: TextSize,
}

/// Every module-level alias name's own name-node position (the three
/// spellings `compile_aliases` reads: `type X = …`, `X = Annotated[…]`,
/// `X: TypeAlias = Annotated[…]`) plus every `def` name's own position,
/// walked recursively so a nested `def` is hovered too. Only names
/// `compile_aliases` actually compiled are included in the alias half —
/// a plain assignment whose RHS is not an alias shape carries no
/// refinement vocabulary, so hovering it would only ever print "(no
/// refinement hover)" and add noise no editor tooltip would show.
///
/// Both halves genuinely serve their own declaration position:
/// `refined_set_at_position`'s stated branch
/// (`check.rs::stated_refinement_at`) reads an alias declaration's own
/// name against `compile_aliases`' own table, and a `def`'s own name
/// against `declared_refinement` on its `-> Annotation` (no base-sort
/// fallback, so a bare `-> float`/`-> int`/`-> str` still answers
/// nothing at the name — that claim is not readable, not fabricated).
fn default_queries(module: &ModModule) -> Vec<HoverQuery> {
    let aliases = compile_aliases(module);
    let mut queries = Vec::new();
    for stmt in &module.body {
        let alias_name = match stmt {
            Stmt::TypeAlias(alias) => match alias.name.as_ref() {
                Expr::Name(name) => Some((name.id.as_str(), name.range())),
                _ => None,
            },
            Stmt::Assign(assign) => match assign.targets.as_slice() {
                [Expr::Name(name)] => Some((name.id.as_str(), name.range())),
                _ => None,
            },
            Stmt::AnnAssign(annotated) => match annotated.target.as_ref() {
                Expr::Name(name) => Some((name.id.as_str(), name.range())),
                _ => None,
            },
            _ => None,
        };
        if let Some((id, range)) = alias_name {
            if aliases.contains_key(id) {
                queries.push(HoverQuery { label: id.to_owned(), position: range.start() });
            }
        }
    }
    collect_def_queries(&module.body, &mut queries);
    queries
}

/// Every `def` name's own position in `body`, recursing into each
/// def's own body so a nested `def` is named too — the hover twin of
/// `collect_bound_names_stmt`'s own `Stmt::FunctionDef` recursion.
fn collect_def_queries(body: &[Stmt], queries: &mut Vec<HoverQuery>) {
    for stmt in body {
        if let Stmt::FunctionDef(def) = stmt {
            queries.push(HoverQuery {
                label: def.name.id.to_string(),
                position: def.name.range.start(),
            });
            collect_def_queries(&def.body, queries);
        }
    }
}

/// Every occurrence of `name` as a whole-word identifier in `source` —
/// a text scan, not an AST resolution: adjacent characters must not
/// themselves be identifier characters, so `level` does not match
/// inside `samples_level`. Acceptable for position discovery per the
/// hover CLI's own brief; the AST is still used for `default_queries`,
/// where a spelled `kind` label matters.
fn find_name_positions(source: &str, name: &str) -> Vec<TextSize> {
    if name.is_empty() {
        return Vec::new();
    }
    let bytes = source.as_bytes();
    let is_ident_byte = |b: u8| b == b'_' || b.is_ascii_alphanumeric();
    let mut positions = Vec::new();
    let mut start = 0;
    while let Some(found) = source[start..].find(name) {
        let offset = start + found;
        let before_ok = offset == 0 || !is_ident_byte(bytes[offset - 1]);
        let after = offset + name.len();
        let after_ok = after >= bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            positions.push(TextSize::try_from(offset).expect("fixture offsets fit in TextSize"));
        }
        start = offset + 1;
    }
    positions
}

/// The composed hover line at `position` — the exact seams the LSP
/// splice drives (`splice_refinedpy_hover` in refinedpy_lsp/src/lib.rs):
/// `refined_set_at_position` for the stated-or-derived set, then
/// `format_for_hover` for its spelling. `None` where either seam
/// answers nothing — never a reimplementation of the rendering, and
/// never an invented fallback spelling.
fn hover_at(module: &ModModule, resolver: &dyn Fn(&str) -> Option<ModModule>, kernel: &Arc<RefinedTSKernel>, position: TextSize) -> Option<String> {
    let set = refined_set_at_position(module, resolver, kernel, position)?;
    format_for_hover(&set)
}

/// Prints one `=== label (path:line:col) ===` section: the composed
/// hover line, or "(no refinement hover)" where the seams answered
/// nothing. The CLI has no host type line to splice a suffix onto (the
/// LSP's own `replaces_host_type` branch only matters where one
/// exists), so the spelling alone prints either way — verbatim what
/// `format_for_hover` returned, never reformatted.
fn print_hover_section(
    path: &str,
    label: &str,
    line_starts: &[usize],
    module: &ModModule,
    resolver: &dyn Fn(&str) -> Option<ModModule>,
    kernel: &Arc<RefinedTSKernel>,
    position: TextSize,
) {
    let (line, col) = line_col(line_starts, usize::from(position));
    println!("\n=== {label} ({path}:{line}:{col}) ===");
    match hover_at(module, resolver, kernel, position) {
        Some(spelled) => println!("{spelled}"),
        None => println!("(no refinement hover)"),
    }
}

/// `--hover <file.py> [name ...]` mode: with names, every occurrence of
/// each name (whole-word text scan); with none, every alias name and
/// every def name (`default_queries`). A name with zero occurrences
/// prints "(not found)", matching `refinedts-hover.ts`'s own shape.
/// Always exits 0 — a hover dump reports what the seams say, it never
/// judges the file.
fn hover_file(path: &str, names: &[String], kernel: &Arc<RefinedTSKernel>) -> ExitCode {
    let Ok(source) = std::fs::read_to_string(path) else {
        eprintln!("{path}: the entry file could not be read");
        return ExitCode::from(2);
    };
    let Ok(parsed) = ruff_python_parser::parse_module(&source) else {
        eprintln!("{path}: the entry file did not parse");
        return ExitCode::from(2);
    };
    let module = parsed.into_syntax();
    let entry_directory = Path::new(path).parent().filter(|dir| !dir.as_os_str().is_empty());
    let resolver = disk_resolver(entry_directory.unwrap_or_else(|| Path::new(".")).to_path_buf());
    let line_starts = line_starts_of(&source);

    if names.is_empty() {
        println!("# RefinedPy hovers — {path}");
        for query in default_queries(&module) {
            print_hover_section(path, &query.label, &line_starts, &module, &resolver, kernel, query.position);
        }
        return ExitCode::SUCCESS;
    }
    for name in names {
        let positions = find_name_positions(&source, name);
        if positions.is_empty() {
            println!("\n=== {name} ===");
            println!("(not found)");
            continue;
        }
        for position in positions {
            print_hover_section(path, name, &line_starts, &module, &resolver, kernel, position);
        }
    }
    ExitCode::SUCCESS
}

/// Writes `path`'s fact artifact to `output` (the path `-o` named, or
/// `<path>.refined.json`). Every omitted def is named on stderr with the
/// construct that stopped it; the artifact itself carries only computed
/// facts. Non-zero only when the file cannot be read, cannot be parsed,
/// or the artifact cannot be written.
fn export_file(path: &str, output: &Path, kernel: &Arc<RefinedTSKernel>) -> ExitCode {
    let Ok(source) = std::fs::read(path) else {
        eprintln!("{path}: the entry file could not be read");
        return ExitCode::from(2);
    };
    let Ok(text) = String::from_utf8(source.clone()) else {
        eprintln!("{path}: the entry file is not UTF-8");
        return ExitCode::from(2);
    };
    let Ok(parsed) = ruff_python_parser::parse_module(&text) else {
        eprintln!("{path}: the entry file did not parse");
        return ExitCode::from(2);
    };
    let module = parsed.into_syntax();
    // The same resolver the judging path uses, so a def reading an
    // imported name derives exactly what the checker derives for it.
    let entry_directory = Path::new(path).parent().filter(|dir| !dir.as_os_str().is_empty());
    let resolver = disk_resolver(entry_directory.unwrap_or_else(|| Path::new(".")).to_path_buf());
    let basename = Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned());

    let export = export_module(&module, &source, &basename, &resolver, kernel, entry_directory);
    for omission in &export.omissions {
        eprintln!(
            "{path}: '{}' is not exported: {}",
            omission.function, omission.reason
        );
    }
    let rendered = match serde_json::to_string_pretty(&export.artifact) {
        Ok(rendered) => rendered,
        Err(err) => {
            eprintln!("{path}: the artifact could not be rendered: {err}");
            return ExitCode::from(2);
        }
    };
    if let Some(parent) = output.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            eprintln!("{}: the cache directory could not be created: {err}", parent.display());
            return ExitCode::from(2);
        }
    }
    // ATOMIC write: a reader (the Go/Rust consumer on the other side of
    // the edge) can observe this file at any moment, including while an
    // LSP save-hook writer is mid-write. A direct fs::write can be read
    // torn (a spurious "not readable JSON" decline); writing a temp file
    // in the SAME directory and renaming over it is atomic on the same
    // volume and needs no new crate (std::fs::rename only). Last-write
    // wins is semantically safe here because every writer derives the
    // artifact from the same target's own disk bytes.
    let file_name = output
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "artifact".to_owned());
    let temp_output = output.with_file_name(format!("{file_name}.tmp.{}", std::process::id()));
    if let Err(err) = std::fs::write(&temp_output, format!("{rendered}\n")) {
        eprintln!("{}: the artifact could not be written: {err}", temp_output.display());
        return ExitCode::from(2);
    }
    if let Err(err) = std::fs::rename(&temp_output, output) {
        eprintln!("{}: the artifact could not be published: {err}", output.display());
        let _ = std::fs::remove_file(&temp_output);
        return ExitCode::from(2);
    }
    ExitCode::SUCCESS
}

/// The command line read into one of the three modes: `--export-fact
/// <file.py> [-o <path>]`, `--hover <file.py> [name ...]`, or the
/// ordinary list of files to judge. A command line that is none of
/// these answers `None` and the caller prints the usage line.
/// `--project-root <path>`, when given, is read here and applied via
/// `set_project_root_override` before any mode runs.
enum Invocation {
    Judge { files: Vec<String>, timing: bool },
    Export { file: String, output: PathBuf },
    Hover { file: String, names: Vec<String> },
}

fn read_invocation(arguments: &[String]) -> Option<Invocation> {
    let mut export_target: Option<String> = None;
    let mut hover_target: Option<String> = None;
    let mut hover_names: Vec<String> = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut project_root: Option<PathBuf> = None;
    let mut timing = false;
    let mut files: Vec<String> = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--timing" => {
                timing = true;
                index += 1;
            }
            "--export-fact" => {
                export_target = Some(arguments.get(index + 1)?.clone());
                index += 2;
            }
            "--hover" => {
                hover_target = Some(arguments.get(index + 1)?.clone());
                index += 2;
                while index < arguments.len() && !arguments[index].starts_with("--") && arguments[index] != "-o" {
                    hover_names.push(arguments[index].clone());
                    index += 1;
                }
            }
            "-o" => {
                output = Some(PathBuf::from(arguments.get(index + 1)?));
                index += 2;
            }
            "--project-root" => {
                project_root = Some(PathBuf::from(arguments.get(index + 1)?));
                index += 2;
            }
            other => {
                files.push(other.to_owned());
                index += 1;
            }
        }
    }
    // set BEFORE the -o default is computed below, so a stated
    // --project-root reaches cache_artifact_path's own .git-walk here
    // exactly as it would reach it inside the checker
    if let Some(root) = &project_root {
        set_project_root_override(Some(root.clone()));
    }
    if let Some(file) = hover_target {
        // --hover owns the whole line; --export-fact/-o/--timing/extra
        // files alongside it would be silently ignored otherwise
        if export_target.is_some() || output.is_some() || timing || !files.is_empty() {
            return None;
        }
        return Some(Invocation::Hover { file, names: hover_names });
    }
    if let Some(file) = export_target {
        // extra positional files or --timing alongside --export-fact
        // would be silently ignored, which is worse than refusing the line
        if !files.is_empty() || timing {
            return None;
        }
        let output = output.unwrap_or_else(|| cache_artifact_path(&file));
        return Some(Invocation::Export { file, output });
    }
    if output.is_some() || files.is_empty() {
        return None;
    }
    Some(Invocation::Judge { files, timing })
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(invocation) = read_invocation(&arguments) else {
        eprintln!("usage: refinedpy-check <file.py> [...] [--timing] [--project-root <path>]");
        eprintln!("       refinedpy-check --export-fact <file.py> [-o <path>] [--project-root <path>]");
        eprintln!("       refinedpy-check --hover <file.py> [name ...]");
        return ExitCode::from(2);
    };
    let Some(dylib) = resolve_kernel_dylib() else {
        eprintln!("refinedpy-check: kernel dylib not found (set REFINEDPY_KERNEL_DYLIB or build it: pnpm kernel:native)");
        return ExitCode::from(2);
    };
    let load_started = std::time::Instant::now();
    let kernel = match load_kernel(&dylib) {
        Ok(kernel) => kernel,
        Err(err) => {
            eprintln!("refinedpy-check: kernel failed to load: {err:?}");
            return ExitCode::from(2);
        }
    };
    let kernel_load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    refinedpy::kernel_ask::install_kernel_seams(&kernel);

    let (files, timing) = match invocation {
        Invocation::Export { file, output } => return export_file(&file, &output, &kernel),
        Invocation::Hover { file, names } => return hover_file(&file, &names, &kernel),
        Invocation::Judge { files, timing } => (files, timing),
    };

    let mut total_printed = 0;
    let mut totals = NO_TIME;
    for file in &files {
        let (printed, lines, file_timing) = check_file(file, &kernel);
        total_printed += printed;
        for line in lines {
            println!("{line}");
        }
        if timing {
            eprintln!(
                "timing: {file} parse_ms={:.2} analysis_ms={:.2} kernel_ms={:.2} asks={}",
                file_timing.parse_ms, file_timing.analysis_ms, file_timing.kernel_ms, file_timing.asks
            );
            totals.parse_ms += file_timing.parse_ms;
            totals.analysis_ms += file_timing.analysis_ms;
            totals.kernel_ms += file_timing.kernel_ms;
            totals.asks += file_timing.asks;
        }
    }
    if timing {
        eprintln!(
            "timing: total files={} kernel_load_ms={:.2} parse_ms={:.2} analysis_ms={:.2} kernel_ms={:.2} asks={}",
            files.len(), kernel_load_ms, totals.parse_ms, totals.analysis_ms, totals.kernel_ms, totals.asks
        );
    }
    if total_printed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
