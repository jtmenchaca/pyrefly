/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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

use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use pyrefly::refinedpy::check::findings_for_module_at;
use pyrefly::refinedpy::cross_module::disk_resolver;
use pyrefly::refinedpy::fact_export::export_module;
use pyrefly::refinedpy::foreign_edge_artifact::cache_artifact_path;
use pyrefly::refinedpy::kernel_path::resolve_kernel_dylib;
use pyrefly::refinedpy::markers::line_col;
use pyrefly::refinedpy::markers::line_starts_of;
use pyrefly::refinedpy::markers::markers_of;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;

fn check_file(path: &str, kernel: &Arc<RefinedTSKernel>) -> (usize, Vec<String>) {
    let Ok(source) = std::fs::read_to_string(path) else {
        return (1, vec![format!("{path}: the entry file did not parse")]);
    };
    let Ok(parsed) = ruff_python_parser::parse_module(&source) else {
        return (1, vec![format!("{path}: the entry file did not parse")]);
    };
    let module = parsed.into_syntax();
    // The entry file's own parent directory is where a sibling `.py`
    // module it imports lives (`disk_resolver`'s own contract) — a bare
    // filename with no parent (a relative path in the current
    // directory) resolves against `.` instead of an empty path.
    let entry_directory = Path::new(path).parent().filter(|dir| !dir.as_os_str().is_empty());
    let resolver = disk_resolver(entry_directory.unwrap_or_else(|| Path::new(".")).to_path_buf());
    let findings = findings_for_module_at(
        &module,
        &resolver,
        kernel,
        Some(entry_directory.unwrap_or_else(|| Path::new("."))),
    );
    let markers = markers_of(&source);
    let line_starts = line_starts_of(&source);

    let mut lines_output = Vec::new();
    let mut matched_markers = vec![false; markers.len()];
    let mut printed = 0;

    for finding in &findings {
        let (line, col) = line_col(&line_starts, usize::from(finding.range.start()));
        // RTS7002 is the undetermined channel — a body's blocker, never
        // an expectation a marker can hold. A marker matching one would
        // silently swallow the very row saying nothing was determined,
        // faking progress; RTS7002 always prints.
        if finding.code != "RTS7002" {
            let matched = markers.iter().enumerate().find(|(_, m)| m.expected_line == line);
            if let Some((index, _)) = matched {
                matched_markers[index] = true;
                continue;
            }
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
                "expected a fire on the next line"
            } else {
                &marker.reason
            }
        ));
    }

    (printed, lines_output)
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

    let export = export_module(&module, &source, &basename, &resolver, kernel);
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

/// The command line read into one of the two modes: `--export-fact
/// <file.py> [-o <path>]`, or the ordinary list of files to judge. A
/// command line that is neither answers `None` and the caller prints the
/// usage line.
enum Invocation {
    Judge(Vec<String>),
    Export { file: String, output: PathBuf },
}

fn read_invocation(arguments: &[String]) -> Option<Invocation> {
    let mut export_target: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut files: Vec<String> = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--export-fact" => {
                export_target = Some(arguments.get(index + 1)?.clone());
                index += 2;
            }
            "-o" => {
                output = Some(PathBuf::from(arguments.get(index + 1)?));
                index += 2;
            }
            other => {
                files.push(other.to_owned());
                index += 1;
            }
        }
    }
    if let Some(file) = export_target {
        // extra positional files alongside --export-fact would be
        // silently ignored, which is worse than refusing the line
        if !files.is_empty() {
            return None;
        }
        let output = output.unwrap_or_else(|| cache_artifact_path(&file));
        return Some(Invocation::Export { file, output });
    }
    if output.is_some() || files.is_empty() {
        return None;
    }
    Some(Invocation::Judge(files))
}

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let Some(invocation) = read_invocation(&arguments) else {
        eprintln!("usage: refinedpy-check <file.py> [...]");
        eprintln!("       refinedpy-check --export-fact <file.py> [-o <path>]");
        return ExitCode::from(2);
    };
    let Some(dylib) = resolve_kernel_dylib() else {
        eprintln!("refinedpy-check: kernel dylib not found (set REFINEDPY_KERNEL_DYLIB or build it: pnpm kernel:native)");
        return ExitCode::from(2);
    };
    let kernel = match load_kernel(&dylib) {
        Ok(kernel) => kernel,
        Err(err) => {
            eprintln!("refinedpy-check: kernel failed to load: {err:?}");
            return ExitCode::from(2);
        }
    };

    let files = match invocation {
        Invocation::Export { file, output } => return export_file(&file, &output, &kernel),
        Invocation::Judge(files) => files,
    };

    let mut total_printed = 0;
    for file in &files {
        let (printed, lines) = check_file(file, &kernel);
        total_printed += printed;
        for line in lines {
            println!("{line}");
        }
    }
    if total_printed == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
