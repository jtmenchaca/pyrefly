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

use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use pyrefly::refinedpy::check::findings_for_module_with_resolver;
use pyrefly::refinedpy::cross_module::disk_resolver;
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
    let findings = findings_for_module_with_resolver(&module, &resolver, kernel);
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

fn main() -> ExitCode {
    let files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        eprintln!("usage: refinedpy-check <file.py> [...]");
        return ExitCode::from(2);
    }
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
