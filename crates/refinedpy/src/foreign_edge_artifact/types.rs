//! The record/row types the artifact reads decode into: the wire-carried
//! cases, one entry position, one function's whole fact, which carrier
//! the JSON transport rides on, and the artifact as consumed.

use std::path::PathBuf;

use refined_sets::refinement_forms::RefinedSet;

/// One admitted case the wire carries — the reader's own twin of the
/// writer's `Case` (`fact_export.rs`): the full kernel wire set grammar
/// verbatim for a number/string case, and no set at all for the two
/// whole-sort floors. `decode_wire_set` is the SAME decoder every other
/// kernel answer goes through, so a set that crossed this edge and a set
/// the kernel answered are the same value.
///
/// `Object` carries the RULED object case's own vocabulary (CROSS-
/// LANGUAGE-EDGE.md §17, JT-prioritized 2026-08-21): each member NAME
/// mapped to ITS OWN cases list (recursed through this same enum, so a
/// nested object case is an ordinary `ForeignCase::Object` sitting inside
/// a member's list) and whether the key set is `closed`. Stored here so
/// the CONSUMER-side lowering (`foreign_edge.rs`'s object-case arm, a
/// stated follow-up — not this lane's) has a typed shape to match on
/// rather than re-parsing the JSON a second time.
#[derive(Debug, Clone, PartialEq)]
pub enum ForeignCase {
    Number(RefinedSet),
    String(RefinedSet),
    Boolean,
    Null,
    Object {
        members: Vec<(String, Vec<ForeignCase>)>,
        closed: bool,
    },
}

/// One parameter position the target states: either a SEQUENCE (an
/// element's own cases plus the length floor the body relies on,
/// carried as `(cases, lengthAtLeast)`) or a plain SCALAR cases list.
#[derive(Debug, Clone)]
pub struct ForeignTsEntry {
    pub name: String,
    /// `Some` for a sequence position — the element's own cases and the
    /// declaration's own length floor.
    pub sequence: Option<(Vec<ForeignCase>, i64)>,
    /// `Some` for a scalar position — the position's own cases list,
    /// never empty when present (a single case still spells as a
    /// one-element list, matching the wire's own convention).
    pub scalar: Option<Vec<ForeignCase>>,
}

/// One target function's whole exported fact.
#[derive(Debug, Clone)]
pub struct ForeignTsFunctionFact {
    pub name: String,
    pub entry: Vec<ForeignTsEntry>,
    /// The return's own cases list — one case lowers directly to a
    /// single value; more than one lowers to a `Kind::KindUnion` of
    /// arms (`foreign_edge.rs::foreign_return_value` does the lowering,
    /// the one place a `ForeignCase` list becomes an `AbstractValue`).
    pub return_cases: Vec<ForeignCase>,
    /// CHANNEL PURITY: the target writes NOTHING to stdout but the
    /// serialized result. Not a premise this file discharges — a
    /// property of the consumed function's return, checked where the
    /// edge consumes it (`foreign_edge.rs`), which is where the sentence
    /// can name the call.
    pub stdout_pure: bool,
    /// Where the target's claim was made: the line in the TypeScript
    /// file, and the sentence its checker said there.
    pub provenance_line: usize,
    pub provenance_said: String,
}

/// Which carrier the JSON transport model rides on — the three `surface
/// .kind` tags this reader admits. All three apply the SAME transport
/// model (the value crosses as JSON text; `stdoutPure` and the
/// outbound-leg fit checks apply identically to each): only the carrier
/// differs — a pipe, one argv element read directly, or one argv
/// element naming a file the target reads its JSON from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignSurface {
    /// `{"kind": "stdin-json", "stdin": "json", "stdout": "json"}` — the
    /// payload rides on the process's stdin pipe.
    StdinJson,
    /// `{"kind": "argv-json", "argIndex": n, "stdout": "json"}` — the
    /// payload is `JSON.parse`'d from `process.argv[argIndex]`; there is
    /// no `stdin` field at all (the carriers are mutually exclusive by
    /// construction, never a joint claim).
    ArgvJson { arg_index: i64 },
    /// `{"kind": "file-json", "argIndex": n, "stdout": "json"}` — the
    /// payload is `JSON.parse`'d from the FILE named at
    /// `process.argv[argIndex]` (node's own harness reads it with
    /// `readFileSync(process.argv[argIndex], "utf8")`), not from the
    /// argv element's own text.
    FileJson { arg_index: i64 },
}

/// The artifact as consumed: the runtime band it commits to, which
/// carrier the JSON transport rides on, and the ONE function the
/// harness calls, already selected.
#[derive(Debug, Clone)]
pub struct ForeignTsArtifact {
    /// The artifact file itself, for the diagnostics.
    pub path: PathBuf,
    /// The `.ts` path the artifact is about, as resolved here (not as
    /// the artifact spells it — the hash is what ties them).
    pub target_file: String,
    pub runtime_band: String,
    /// Which carrier the target's `surface` states — `foreign_edge.rs`
    /// checks a recognized call's own channel against this before
    /// applying the outbound-leg fit checks.
    pub surface: ForeignSurface,
    pub called: ForeignTsFunctionFact,
}
