//! The one judging seam: a flowing value against a declared refinement.
//! Every sink (annotated assignment, argument, return, field write)
//! routes here, so fire wording, silence, and undetermined sentences
//! stay uniform. This file is the contract the walk calls; the
//! assignability unit fills it in behind these signatures.

mod judge;
mod temporal;
mod sequence;
mod scalar;

#[cfg(test)]
mod tests;

pub use judge::judge;
pub(crate) use sequence::sequence_shaped;
pub(crate) use sequence::states_sequence;

/// What judging one value against one declared set concluded.
pub enum Verdict {
    /// The value is provably outside the set — the message is the full
    /// diagnostic text.
    Fire(String),
    /// The value is provably inside the set.
    Silent,
    /// The walk could not read enough to judge — the sentence names
    /// what blocked, in plain per-position prose.
    Undetermined(String),
}
