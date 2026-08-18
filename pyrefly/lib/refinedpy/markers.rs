/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `# refinedpy: expect-error` markers, shared by the check CLI and
//! the LSP seam. A marker names the line it expects a fire on; a fire
//! matched by a marker is an expectation held, so both surfaces stay
//! silent on it. RTS7002 (the undetermined channel) is never matched —
//! a marker swallowing "nothing was determined" would fake progress.

/// A `# refinedpy: expect-error` marker: the line it sits on, the line
/// it expects a fire on (the next non-comment line), and its reason.
pub struct Marker {
    pub marker_line: usize,
    pub expected_line: usize,
    pub reason: String,
}

pub fn markers_of(source: &str) -> Vec<Marker> {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("# refinedpy: expect-error") else {
            continue;
        };
        let reason = rest
            .trim_start_matches([' ', '\u{2014}', '-'])
            .trim()
            .to_owned();
        // The expected fire sits on the next line that is not itself a
        // comment (host-marker lines may sit between).
        let expected = lines
            .iter()
            .enumerate()
            .skip(index + 1)
            .find(|(_, l)| !l.trim_start().starts_with('#'))
            .map(|(i, _)| i + 1)
            .unwrap_or(index + 2);
        out.push(Marker {
            marker_line: index + 1,
            expected_line: expected,
            reason,
        });
    }
    out
}

/// 1-based line and column of a byte offset.
pub fn line_col(line_starts: &[usize], offset: usize) -> (usize, usize) {
    let line = line_starts.partition_point(|start| *start <= offset);
    let col = offset - line_starts[line - 1] + 1;
    (line, col)
}

pub fn line_starts_of(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            starts.push(i + 1);
        }
    }
    starts
}
