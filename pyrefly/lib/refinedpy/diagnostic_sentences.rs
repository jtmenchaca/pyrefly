/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
}

pub const SENTENCE: Sentences = Sentences {
    kernel_declined_member: "the kernel does not yet decide membership for this set shape",
    kernel_declined_containment: "the kernel does not yet decide containment for this set shape",
    value_not_readable: "the flowing value is not yet readable",
    typed_dict_position: "a TypedDict-declared position holds a value this table does not yet read",
    tuple_position: "a fixed-arity-tuple-declared position holds a value this table does not yet read",
};

/// A cross-language crossing's refutation: the reason the value cannot
/// cross, with the target's own provenance appended — the second step
/// of the two-language explanation, in the message-text form
/// `foreign_edge.go`'s own `foreignMessage` renders. `provenance_line`
/// of 0 means the target stated no line (the provenance is present but
/// unlocated); an empty `provenance_said` alongside a nonzero line
/// still names WHERE without a quoted claim.
pub fn foreign_crossing_refusal(
    said: &str,
    provenance_file: &str,
    provenance_line: usize,
    provenance_said: &str,
) -> String {
    if provenance_file.is_empty() {
        return said.to_owned();
    }
    let mut where_said = provenance_file.to_owned();
    if provenance_line > 0 {
        where_said.push(':');
        where_said.push_str(&provenance_line.to_string());
    }
    if provenance_said.is_empty() {
        return format!("{said}. the target states this at {where_said}");
    }
    format!("{said}. {where_said} said: {provenance_said}")
}

/// A member's own refutation, keyed by the name it escaped under — the
/// dict/TypedDict element law's own suffix.
pub fn at_key(message: &str, key: &str) -> String {
    format!("{message} (at key '{key}')")
}

/// A member's own refutation, keyed by the index it escaped at — the
/// list/tuple element law's own suffix.
pub fn at_index(message: &str, index: usize) -> String {
    format!("{message} (at index {index})")
}

#[cfg(test)]
mod tests {
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set, union};

    use super::*;

    fn age_set() -> RefinedSet {
        make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)])
    }

    /// The sink requirement names the alias AND what it requires, so a
    /// reader never opens `Age`'s definition to learn its bounds.
    #[test]
    fn a_named_alias_spells_its_own_contents_beside_its_name() {
        let words = required_words("Age", &age_set());
        assert!(words.contains("'Age'"), "{words}");
        assert!(words.contains("0"), "{words}");
        assert!(words.contains("120"), "{words}");
    }

    /// An inline annotation already spells its own contents — printing
    /// them twice is noise, so the contents are not repeated.
    #[test]
    fn a_spelling_that_already_states_its_contents_is_not_repeated() {
        let set = age_set();
        let contents = format_for_diagnostics(&set);
        let words = required_words(&contents, &set);
        assert_eq!(words, format!("'{contents}'"));
    }

    /// A refutation names the value and the requirement, in that order.
    #[test]
    fn a_refutation_names_the_value_then_the_requirement() {
        let message = refutation("200", "Age", &age_set());
        assert!(message.contains("'200'"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("120"), "{message}");
    }

    /// The exact-value sort crossing names the word, keeps the standard
    /// verdict clause, and appends the reason.
    #[test]
    fn an_exact_value_sort_crossing_names_the_word_and_the_reason() {
        let message = cross_sort_of_value("30.0", "a float", "an integer", "Age", &age_set());
        assert!(message.contains("'30.0'"), "{message}");
        assert!(message.contains("not assignable"), "{message}");
        assert!(message.contains("is a float"), "{message}");
        assert!(message.contains("states an integer"), "{message}");
        assert!(message.contains("not allowed here"), "{message}");
    }

    /// A containment refutation spells both sets and says what escapes.
    #[test]
    fn a_containment_refutation_spells_both_sets() {
        let flowing = make_refined_set(vec![integer(), at_least(0.0), at_most(200.0)]);
        let message = containment_refutation(&flowing, "Age", &age_set());
        assert!(message.contains("200"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
        assert!(
            message.contains("admits values outside the declared set"),
            "{message}"
        );
    }

    /// A string-shaped set spells as a string, never as codepoints — the
    /// requirement clause reads the same way the value clause does.
    #[test]
    fn a_string_literal_requirement_spells_readably() {
        let set = make_refined_set(vec![union(string_tuple("A"), string_tuple("B"))]);
        let words = required_words("Grade", &set);
        assert!(words.contains("'Grade'"), "{words}");
    }

    /// The member suffixes name which key or index escaped.
    #[test]
    fn member_suffixes_name_the_offending_position() {
        assert!(at_key("a value", "age").contains("at key 'age'"));
        assert!(at_index("a value", 2).contains("at index 2"));
    }

    /// A crossing refusal appends the target's own located provenance.
    #[test]
    fn a_crossing_refusal_appends_the_located_provenance() {
        let message = foreign_crossing_refusal("the value can escape", "./audio_level.ts", 30, "0.0 … 1.0");
        assert!(message.contains("the value can escape"), "{message}");
        assert!(message.contains("./audio_level.ts:30"), "{message}");
        assert!(message.contains("0.0 … 1.0"), "{message}");
    }

    /// A provenance with no line still names the file, un-located.
    #[test]
    fn a_crossing_refusal_with_no_line_still_names_the_file() {
        let message = foreign_crossing_refusal("the value can escape", "./audio_level.ts", 0, "");
        assert!(message.contains("./audio_level.ts"), "{message}");
        assert!(!message.contains(":0"), "{message}");
        assert!(message.contains("states this at"), "{message}");
    }

    /// No provenance at all leaves the said sentence unchanged.
    #[test]
    fn a_crossing_refusal_with_no_provenance_is_unchanged() {
        let message = foreign_crossing_refusal("the value can escape", "", 0, "");
        assert_eq!(message, "the value can escape");
    }
}
