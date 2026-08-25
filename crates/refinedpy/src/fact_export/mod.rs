//! The Python fact-export surface: one module's CHECKED facts written as
//! a JSON artifact another language's checker reads across the FFI edge
//! (CROSS-LANGUAGE-EDGE.md §8 — "the edge's return fact — the Python
//! function's kernel summary pushed through the return transport — IS
//! the fact on that expression").
//!
//! What crosses is what the checker DERIVED, never what an annotation
//! claimed: `check::derived_return_values` runs the same walk
//! `findings_for_module_with_resolver` runs (parameters seeded from
//! their declarations by `seed_parameters`, the body walked statement by
//! statement) and hands back the join of every value the body's
//! `return`s produced. The entry sets are the declared refinements the
//! walk itself seeds from (`typereading::declared_refinement`), so a
//! consumer reading the entry and the return reads exactly the two ends
//! of one derivation.
//!
//! EVERY FIELD IS COMPUTED. A def with a parameter carrying no declared
//! refinement, or with a derived return that has no faithful set reading
//! (an object, an unknown), is OMITTED from the artifact with the reason
//! named on stderr — the artifact never carries a stub, a placeholder,
//! or a widened stand-in for a fact this checker did not derive.
//!
//! The premises the artifact's own fields discharge (§5, "Edge
//! premises"):
//!
//! - TARGET INTEGRITY — `target.contentHash` is sha256 over the file's
//!   exact bytes, so a consumer can check that the code that runs is the
//!   code that was checked.
//! - RUNTIME IDENTITY — `runtime.band` states the semantics the Python
//!   pins commit to, which the derived facts inherit.
//! - CHANNEL PURITY — `return.stdoutPure` is the effect fact §5 names:
//!   the target writes nothing else to the channel the wire uses.

mod cases;
mod entry;
mod export;
mod harness;
mod positions;
mod sha256;
mod stdout_purity;
mod traversal;

#[cfg(test)]
mod tests;

// Test module is a sibling of the domain children, so re-export their
// items into this module's namespace for `tests`'s `use super::*`.
#[cfg(test)]
pub(super) use cases::*;
#[cfg(test)]
pub(super) use entry::*;
#[cfg(test)]
pub(super) use harness::*;
#[cfg(test)]
pub(super) use stdout_purity::*;

pub use export::cached_hash_matches;
pub use export::export_module;
pub use export::has_exportable_defs;
pub(super) use export::top_level_defs;
pub(crate) use sha256::sha256_hex;

use serde_json::Value;

/// The artifact's own kind tag — the Go consumer matches on it before
/// reading a single fact. NO version field: the RULED cases schema
/// (JT, 2026-08-21) carries no version ceremony at all — a reader
/// strict-parses the CURRENT shape, and any other shape (a version
/// field, a bare "set", the old sequence spelling) is NO-FACT, read by
/// the same "no fact this reader recognizes" sentence every other
/// unreadable artifact earns. `language` (below) is what tells the
/// consumer which pins to check the runtime band against, so adding a
/// language never adds a new artifact kind.
const ARTIFACT_KIND: &str = "fact-artifact";

/// The language this producer states — schema v2's own field, read
/// beside `runtime.band` so a consumer checks the band against the
/// right pins.
const ARTIFACT_LANGUAGE: &str = "python";

/// The semantics band every fact in this artifact inherits: the Python
/// pins commit to CPython 3.11+ behaviour, not to "Python"
/// (CROSS-LANGUAGE-EDGE.md §5, "Runtime identity").
const RUNTIME_BAND: &str = "cpython-3.11+";

/// One def this export could not carry, and the construct that stopped
/// it — the work-queue row a reader turns into a fix.
pub struct Omission {
    pub function: String,
    pub reason: String,
}

/// One module's export: the artifact, and every def omitted from it.
pub struct Export {
    pub artifact: Value,
    pub omissions: Vec<Omission>,
}
