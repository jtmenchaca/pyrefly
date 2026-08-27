//! THE wording module: every refinement sentence the Python checker
//! emits is built here, and nowhere else invents one — the discipline
//! refined-ts-go keeps in `walk/coverage_sentences.go` ("every coverage
//! sentence lives here"), carried across to this adapter.
//!
//! Two shapes of sentence, matching the Go twin's own two:
//!
//! - A REFUTATION names what the value was derived to be and what the
//!   position requires, in that order. Where the two sides cross SORTS
//!   (a string arriving where a number is stated), the sentence says so
//!   in plain words rather than leaving the reader to compare two
//!   spellings — `worn_set_membership.go`'s `wornCrossSortText`.
//! - A DECLINE (the undetermined channel) names, per position, what
//!   blocked — never a category label. `decline_reasons.go`: "a walk
//!   records the reason at the position it finds one, in a plain
//!   sentence about that position — never a category name."
//!
//! The sink requirement is spelled from the declared set's own contents
//! (`format_for_diagnostics` — `>= 0 && <= 120`) beside the name the
//! annotation gave it, so a reader sees WHAT `Age` requires without
//! opening its definition. Where the spelling and the contents would
//! read as the same words (an inline annotation whose spelling already
//! IS its contents), the contents are not repeated.
//!
//! ## Module layout
//!
//! This file keeps the core refusal vocabulary (`required_words`,
//! `refutation`, `cross_sort_of_value`, `containment_refutation`) and
//! the decline table (`Sentences`/`SENTENCE`) every sink shares.
//! `member_refusals` holds the per-position rewrites a dict/list/tuple
//! element's own refusal earns (`at_key`/`at_member`/`at_index`/
//! `at_slot`/`slot_word`/`element_set_refutation`). `raise_and_blocker`
//! holds the empty-set, unhonorable-annotation, provable-raise, and
//! loop/generator blocker sentences. `foreign_edge_sentences` holds the
//! `os.system`/cross-language-crossing wording. `manifest_sentences`
//! holds the manifest-reader's own decline vocabulary and the
//! `strptime` STAGE 2 directive declines.

mod foreign_edge_sentences;
mod manifest_sentences;
mod member_refusals;
mod raise_and_blocker;

#[cfg(test)]
mod tests;

pub use foreign_edge_sentences::compiled_binary_no_fact;
pub use foreign_edge_sentences::foreign_crossing_refusal;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_argv_at_file_target;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_argv_at_stdin_target;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_argv_index;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_file_at_argv_target;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_file_at_stdin_target;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_file_index;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_stdin_at_argv_target;
pub use foreign_edge_sentences::foreign_edge_channel_mismatch_stdin_at_file_target;
pub use foreign_edge_sentences::foreign_edge_double_channel_declared;
pub use foreign_edge_sentences::os_system_missing_entry_write;
pub use foreign_edge_sentences::os_system_missing_return_read;
pub use foreign_edge_sentences::os_system_no_stdout_capture;
pub use foreign_edge_sentences::os_system_shell_string_unreadable;
pub use foreign_edge_sentences::script_path_not_a_literal;
pub use manifest_sentences::manifest_entry_crossing_refused;
pub use manifest_sentences::manifest_entry_names_no_producer;
pub use manifest_sentences::manifest_names_no_entry_for;
pub use manifest_sentences::manifest_unreadable;
pub use manifest_sentences::strptime_locale_directive;
pub use manifest_sentences::strptime_unread_directive;
pub use member_refusals::at_index;
pub use member_refusals::at_key;
pub use member_refusals::at_member;
pub use member_refusals::at_slot;
pub use member_refusals::element_set_refutation;
pub use member_refusals::missing_required_key;
pub use member_refusals::slot_word;
pub use raise_and_blocker::dict_changed_size_during_iteration;
pub use raise_and_blocker::attribute_on_a_receiver_that_admits_none;
pub use raise_and_blocker::statement_is_unreachable;
pub use raise_and_blocker::division_by_a_set_that_admits_zero;
pub use raise_and_blocker::empty_set;
pub use raise_and_blocker::generator_body_never_summarized;
pub use raise_and_blocker::list_never_terminates_self_append;
pub use raise_and_blocker::loop_accumulation_did_not_stabilize;
pub use raise_and_blocker::stale_marker_refusal;
pub use raise_and_blocker::unhonorable_annotation;
pub use raise_and_blocker::unmodeled_module_call;

// Test module is a sibling of the domain children, so re-export the
// private items it needs into this module's namespace for `tests`'s
// `use super::*`.
#[cfg(test)]
pub(self) use member_refusals::plain_refutation_value_words;

use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::RefinedSet;

/// The sink requirement, spelled: the name the annotation gave it, and
/// what that name REQUIRES, so the reader never has to open the alias
/// to learn what it admits. `Age (>= 0 && <= 120)`.
///
/// The contents are omitted when they would only restate the spelling —
/// an inline annotation (`Literal["a", "b"]`) already spells its own
/// contents, and printing them twice is noise, not reasoning.
pub fn required_words(spelling: &str, set: &RefinedSet) -> String {
    let contents = format_for_diagnostics(set);
    if contents == spelling || spelling.contains(&contents) || contents == "any value" {
        return format!("'{spelling}'");
    }
    format!("'{spelling}' ({contents})")
}

/// The sentence a refuted value earns: what the value is, and what the
/// position requires. Mirrors the Go twin's `wornRefutationText`
/// (`worn_set_membership.go:202`) — "<what> of type '<known>' is not
/// assignable to type '<target>'" — with the target's own contents
/// spelled beside its name.
pub fn refutation(value_words: &str, spelling: &str, set: &RefinedSet) -> String {
    format!(
        "a value of type '{value_words}' is not assignable to type {}",
        required_words(spelling, set)
    )
}

/// The sentence a SORT crossing earns — a string arriving where a
/// number is stated, `30.0` where an integer is, and every mirror of
/// them. Ported from the Go twin's `wornCrossSortText`
/// (`worn_set_membership.go:215`) and `checkExactValues`' own
/// scalar/sequence arms (`set_membership.go:311`): the two sorts named,
/// and the plain statement that one is not allowed where the other is
/// stated. This states the REASON — two sorts no run reconciles — where
/// a bare "not assignable" would leave a reader comparing two spellings
/// that look compatible (`30.0` against `0 ≤ value ≤ 120`).
///
/// Composed the way the Go twin composes its own: the standard
/// refutation clause states the verdict, and the reason follows after
/// an em dash — `wornRefutationText(...) + hint`
/// (`worn_set_membership.go:194`), where the hint names WHY the two
/// sides cannot meet. A reader who scans for "not assignable" finds it;
/// a reader who wants the reason reads on.
pub fn cross_sort_of_value(
    value_words: &str,
    value_said: &str,
    position_said: &str,
    spelling: &str,
    set: &RefinedSet,
) -> String {
    format!(
        "{} — the value is {value_said}, the position states {position_said}, and {value_said} is not allowed here",
        refutation(value_words, spelling, set)
    )
}

/// The reason clause a refuted CONTAINMENT earns: the flowing set admits
/// values the declared set does not. The Go twin spells both sets and
/// leaves finding a witness to the reader (`assignability.rs`'s own
/// containment law doc, ported from `checkWornScalarSubset`).
pub fn containment_refutation(flowing: &RefinedSet, spelling: &str, set: &RefinedSet) -> String {
    format!(
        "a value of type '{}' is not assignable to type {} — the flowing set admits values outside the declared set",
        format_for_diagnostics(flowing),
        required_words(spelling, set)
    )
}

/// Every decline sentence — the undetermined channel's own vocabulary,
/// the twin of `coverage_sentences.go`'s `Sentence` struct. Each reads
/// as a plain statement about THIS position, never a category name.
pub struct Sentences {
    pub kernel_declined_member: &'static str,
    pub kernel_declined_containment: &'static str,
    pub value_not_readable: &'static str,
    pub typed_dict_position: &'static str,
    pub tuple_position: &'static str,
    /// A temporal-declared position (`date`/`timedelta`/`datetime`/
    /// `AwareDatetime`/`NaiveDatetime`) holds a flowing value this table
    /// does not yet read as one of the three recognized constructions
    /// (`datetime_date`/`datetime_timedelta`/`datetime_datetime`).
    pub temporal_position: &'static str,
    /// A temporal position's own exact instant/date/duration could not
    /// be proved against the declared window at all — a kernel refusal,
    /// or a construction (`TzinfoKind::OtherAware`) with no exact
    /// offset to compare.
    pub temporal_unprovable_instant: &'static str,
}

pub const SENTENCE: Sentences = Sentences {
    kernel_declined_member: "the kernel does not yet decide membership for this set shape",
    kernel_declined_containment: "the kernel does not yet decide containment for this set shape",
    value_not_readable: "the flowing value is not yet readable",
    typed_dict_position: "a TypedDict-declared position holds a value this table does not yet read",
    tuple_position: "a fixed-arity-tuple-declared position holds a value this table does not yet read",
    temporal_position: "a temporal-declared position holds a value this table does not yet read as a date/timedelta/datetime construction",
    temporal_unprovable_instant: "the flowing instant is not provable against the declared calendar window",
};
