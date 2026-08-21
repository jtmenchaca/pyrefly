/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `# refinedpy: expect-error` markers, shared by the check CLI and
//! the LSP seam. A standalone marker line names the NEXT line it
//! expects a fire on (host-marker lines may sit between); a trailing
//! marker — code before the `#` on the same line — covers that same
//! line, matching the Go reader's own two coverage modes
//! (expect_error.go). An optional numeric code right after
//! `expect-error` narrows the marker to that code alone; the reason
//! follows the code, the same ordering Go uses. RTS7002 (the
//! undetermined channel) is never matched, coded or not — a marker
//! swallowing "nothing was determined" would fake progress.

/// A `# refinedpy: expect-error` marker: the line it sits on, the line
/// it expects a fire on (the next non-comment line for a standalone
/// marker, or its own line for a trailing one), its optional narrowed
/// code, and its reason.
pub struct Marker {
    pub marker_line: usize,
    pub expected_line: usize,
    pub code: Option<u32>,
    pub reason: String,
}

impl Marker {
    /// Whether this marker holds a fire reporting `finding_code`
    /// (`"RTS7001"`-shaped). RTS7002 — the undetermined channel — is
    /// never held, by any marker, coded or not: a stale marker that
    /// only covered a 7002 line is the honest outcome, not a bug in
    /// the matcher. A coded marker narrows to that code alone; a
    /// code-less marker holds any other code.
    pub fn covers(&self, finding_code: &str) -> bool {
        if finding_code == "RTS7002" {
            return false;
        }
        match self.code {
            Some(code) => finding_code == format!("RTS{code}"),
            None => true,
        }
    }
}

const MARKER_TEXT: &str = "# refinedpy: expect-error";

pub fn markers_of(source: &str) -> Vec<Marker> {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        let leading_ws = line.len() - trimmed.len();
        let Some(marker_at) = line.find(MARKER_TEXT) else {
            continue;
        };
        let rest = &line[marker_at + MARKER_TEXT.len()..];
        // Standalone: nothing but whitespace precedes the marker text,
        // so it covers the next non-comment line. Trailing: real code
        // precedes it on the same line, so it covers that line itself
        // (the Go reader's own two modes, expect_error.go).
        let standalone = marker_at == leading_ws;
        let (code, after_code) = numeric_code_of(rest);
        let reason = after_code
            .trim_start_matches([' ', '\u{2014}', '-'])
            .trim()
            .to_owned();
        let expected = if standalone {
            // The expected fire sits on the next line that is not
            // itself a comment (host-marker lines may sit between).
            lines
                .iter()
                .enumerate()
                .skip(index + 1)
                .find(|(_, l)| !l.trim_start().starts_with('#'))
                .map(|(i, _)| i + 1)
                .unwrap_or(index + 2)
        } else {
            index + 1
        };
        out.push(Marker {
            marker_line: index + 1,
            expected_line: expected,
            code,
            reason,
        });
    }
    out
}

/// Reads an optional numeric code immediately after the marker text —
/// `" RTS7001"` or a bare `" 7001"` — and returns it with the
/// remaining text (where the reason, if any, starts). No digits at
/// the front means no code: the whole remainder is reason text.
fn numeric_code_of(rest: &str) -> (Option<u32>, &str) {
    let after_space = rest.strip_prefix(' ').unwrap_or(rest);
    let digits_start = after_space.strip_prefix("RTS").unwrap_or(after_space);
    let digit_count = digits_start
        .as_bytes()
        .iter()
        .take_while(|b| b.is_ascii_digit())
        .count();
    if digit_count == 0 {
        return (None, rest);
    }
    let (digits, remainder) = digits_start.split_at(digit_count);
    match digits.parse::<u32>() {
        Ok(code) => (Some(code), remainder),
        Err(_) => (None, rest),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_standalone_marker_covers_the_next_line() {
        let markers = markers_of("# refinedpy: expect-error\nover: Age = 200\n");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].marker_line, 1);
        assert_eq!(markers[0].expected_line, 2);
    }

    #[test]
    fn a_trailing_marker_covers_its_own_line() {
        let markers = markers_of("over: Age = 200  # refinedpy: expect-error\n");
        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0].marker_line, 1);
        assert_eq!(markers[0].expected_line, 1);
    }

    #[test]
    fn a_trailing_marker_holds_a_fire_on_its_own_line() {
        let markers = markers_of("over: Age = 200  # refinedpy: expect-error\n");
        assert!(markers[0].covers("RTS7001"));
    }

    #[test]
    fn a_trailing_marker_is_stale_when_nothing_fires_on_its_own_line() {
        // The check CLI/LSP seam only holds a marker when some finding's
        // line equals `expected_line`; with no findings at all here, the
        // marker itself is what the caller reports as stale. This test
        // pins the coverage rule the callers rely on: `covers` says a
        // marker CAN hold a matching code, not that one exists.
        let markers = markers_of("over: Age = 40  # refinedpy: expect-error\n");
        assert_eq!(markers[0].expected_line, 1);
        assert!(markers[0].covers("RTS7001"));
    }

    #[test]
    fn a_coded_marker_matches_only_its_own_code() {
        let markers = markers_of("# refinedpy: expect-error RTS7001\nover: Age = 200\n");
        assert_eq!(markers[0].code, Some(7001));
        assert!(markers[0].covers("RTS7001"));
    }

    #[test]
    fn a_coded_marker_accepts_a_bare_numeric_code() {
        let markers = markers_of("# refinedpy: expect-error 7001\nover: Age = 200\n");
        assert_eq!(markers[0].code, Some(7001));
        assert!(markers[0].covers("RTS7001"));
    }

    #[test]
    fn a_coded_marker_does_not_match_a_different_code() {
        let markers = markers_of("# refinedpy: expect-error RTS7001\nover: Age = 200\n");
        assert!(!markers[0].covers("RTS7003"));
    }

    #[test]
    fn a_coded_rts7002_marker_never_matches() {
        // The hard law (module doc, and markers.rs:11-12's original
        // wording): RTS7002 is never matched by any marker, coded or
        // not. A marker spelled with the undetermined code itself must
        // stay just as unable to swallow a 7002 finding.
        let markers = markers_of("# refinedpy: expect-error RTS7002\nover = unread()\n");
        assert_eq!(markers[0].code, Some(7002));
        assert!(!markers[0].covers("RTS7002"));
    }

    #[test]
    fn a_code_less_marker_still_never_matches_rts7002() {
        let markers = markers_of("# refinedpy: expect-error\nover = unread()\n");
        assert_eq!(markers[0].code, None);
        assert!(!markers[0].covers("RTS7002"));
    }

    #[test]
    fn a_coded_marker_reason_follows_the_code() {
        let markers = markers_of(
            "# refinedpy: expect-error RTS7001 — 200 is outside the set\nover: Age = 200\n",
        );
        assert_eq!(markers[0].code, Some(7001));
        assert_eq!(markers[0].reason, "200 is outside the set");
    }

    #[test]
    fn a_code_less_marker_reason_is_unaffected() {
        let markers = markers_of("# refinedpy: expect-error — 200 is outside the set\nover: Age = 200\n");
        assert_eq!(markers[0].code, None);
        assert_eq!(markers[0].reason, "200 is outside the set");
    }
}
