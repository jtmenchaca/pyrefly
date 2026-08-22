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

/// `os.system`'s own undetermined reason for a command whose runner and
/// script DID read cleanly: even a followed literal shell command has
/// no value channel, because `os.system` never captures stdout — names
/// both the missing captured-stdout leg and the fixable argv-list
/// respelling.
pub fn os_system_no_stdout_capture(runner_and_script: &str) -> String {
    format!(
        "{runner_and_script} runs, but os.system captures no stdout — there is no captured-stdout leg for a \
        return fact to attach to; spell the call as subprocess.run([...], input=..., capture_output=True, \
        text=True) instead"
    )
}

/// The shell-string law-2 decline: `os.system`'s argument is not one
/// written string literal, so its tokens cannot be read at all.
pub fn os_system_shell_string_unreadable() -> String {
    "this command is a shell string the checker cannot read; spell it as an argv list \
    (subprocess.run([\"node\", \"<script>.ts\"], ...))"
        .to_owned()
}

/// The law-2 decline for a script path that is neither a written string
/// literal directly in the argv list nor a module-level constant this
/// body can resolve (an f-string, a parameter, any other computed
/// expression) — reused by every remaining non-literal argv[1] shape.
pub fn script_path_not_a_literal() -> String {
    "the script path is computed; spell it as a written string literal".to_owned()
}

/// The channel-mismatch decline when the call sends its payload on
/// stdin but the target's own fact serves JSON on an argv element — the
/// reverse of `foreign_edge_channel_mismatch_argv_at_stdin_target`.
/// Neither side is malformed; the two simply do not name the same
/// carrier, so the JSON transport model has nothing to apply to.
pub fn foreign_edge_channel_mismatch_stdin_at_argv_target() -> String {
    "the call passes the payload on stdin, but the target's fact serves JSON on an argv element — the \
    channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call sends its payload as an
/// argv element but the target's own fact serves JSON on stdin.
pub fn foreign_edge_channel_mismatch_argv_at_stdin_target() -> String {
    "the call passes the payload as an argv element, but the target's fact serves JSON on stdin — the \
    channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when both sides name an argv carrier
/// but at different indices — the call writes the payload at one
/// position and the target reads it from another.
pub fn foreign_edge_channel_mismatch_argv_index(called_index: i64, declared_index: i64) -> String {
    format!(
        "the call passes the payload at argv[{called_index}], but the target's fact reads its payload at \
        argv[{declared_index}] — the channels do not meet"
    )
}

/// The channel-mismatch decline when the call writes a temp file named
/// at an argv element but the target's own fact serves JSON on stdin —
/// the file carrier's own mismatch against `stdin-json`, symmetric with
/// `foreign_edge_channel_mismatch_argv_at_stdin_target`.
pub fn foreign_edge_channel_mismatch_file_at_stdin_target() -> String {
    "the call passes the payload through a temp file named as an argv element, but the target's fact serves \
    JSON on stdin — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call sends its payload on
/// stdin but the target's own fact reads its JSON from a FILE named at
/// an argv element — the reverse of
/// `foreign_edge_channel_mismatch_file_at_stdin_target`.
pub fn foreign_edge_channel_mismatch_stdin_at_file_target() -> String {
    "the call passes the payload on stdin, but the target's fact reads its JSON from a file named as an \
    argv element — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call writes a temp file named
/// at an argv element but the target's own fact reads its argv element
/// AS the JSON text directly (`argv-json`), never as a file path.
pub fn foreign_edge_channel_mismatch_file_at_argv_target() -> String {
    "the call passes the payload through a temp file named as an argv element, but the target's fact reads \
    that argv element as the JSON text itself — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when the call passes its payload
/// directly as an argv element (`argv-json`), but the target's own fact
/// reads that argv element as a FILE PATH (`file-json`) rather than the
/// JSON text itself.
pub fn foreign_edge_channel_mismatch_argv_at_file_target() -> String {
    "the call passes the payload directly as an argv element, but the target's fact reads that argv element \
    as a file path holding the JSON — the channels do not meet"
        .to_owned()
}

/// The channel-mismatch decline when both sides name a file-carried argv
/// element but at different indices.
pub fn foreign_edge_channel_mismatch_file_index(called_index: i64, declared_index: i64) -> String {
    format!(
        "the call names the temp file at argv[{called_index}], but the target's fact reads the file's path \
        from argv[{declared_index}] — the channels do not meet"
    )
}

/// The double-channel decline when a call names BOTH an argv-json
/// payload and an `input=` keyword — a real ambiguity, not a recognition
/// gap: two crossing values are stated and this checker names one
/// transport per call.
pub fn foreign_edge_double_channel_declared() -> String {
    "this call passes the payload both as an argv element and through input=json.dumps(...) — two crossing \
    channels are named and this checker recognizes exactly one transport per call"
        .to_owned()
}

/// The return-leg corner decline: the target's declared return admits
/// an infinite corner (+Infinity or -Infinity) that the JSON leg
/// cannot carry — `JSON.stringify`/`json.dumps` writes that corner as
/// the bare token `null` (RFC 8259 has no numeral for it), a value
/// outside the claimed set landing at this call's own consumer. Named
/// per corner so the reader sees WHICH end escapes, never a category.
pub fn foreign_edge_return_admits_uncarriable_corner(function_name: &str, corner: &str) -> String {
    format!(
        "the target {function_name}'s stated return admits {corner}, which the JSON stdout leg cannot \
        carry — the crossing cannot be trusted at that corner"
    )
}

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

/// A stale expect-error marker's own diagnostic (the RTS7005 role):
/// the marker expected a fire on its covered line and nothing fired.
/// Mirrors the Go host's editor-view wording; the marker's captured
/// reason text, when present, rides in parentheses so the reader sees
/// what the author expected to be caught.
pub fn stale_marker_refusal(expected_line: usize, reason: Option<&str>) -> String {
    let base = format!(
        "expected a refinement fire on line {expected_line} and nothing fired — remove the '# refinedpy: expect-error' marker or restore the failing code"
    );
    match reason {
        Some(reason) if !reason.is_empty() => format!("{base} ({reason})"),
        _ => base,
    }
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

/// The empty-set sentence — an annotation compiles to a set the kernel
/// proves admits nothing. Mirrors the Go twin's own RTS7003 wording
/// (`annotation_file_facts.go`: `"this annotation denotes the empty
/// set: '" + FormatForDiagnostics(set) + "'"`), spelling the compiled
/// set's own contents so the reader sees WHY, not just THAT, it is
/// empty.
pub fn empty_set(set: &RefinedSet) -> String {
    format!("this annotation denotes the empty set: '{}'", format_for_diagnostics(set))
}

/// The unhonorable-statement sentence — an annotation recognizably
/// spells this table's OWN vocabulary (an `Annotated[...]` rooted at
/// the module's imported `Annotated` identity) but this table could
/// not compile it. Mirrors the Go twin's own RTS7004 wording
/// (`annotation_file_facts.go`'s `compiled.Unsupported` /
/// `compiled.Unsupported.Unsupported` messages): names the spelling
/// so the reader sees which statement was refused.
pub fn unhonorable_annotation(spelling: &str) -> String {
    format!("this annotation '{spelling}' is recognized as a refinement statement but this table could not compile it")
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

    /// The os.system no-stdout-capture sentence names the runner and
    /// script that ran, states the missing captured-stdout leg, and
    /// teaches the fixable argv-list respelling.
    #[test]
    fn the_os_system_no_stdout_capture_sentence_names_the_missing_leg_and_the_fix() {
        let message = os_system_no_stdout_capture("node ./audio_level.ts");
        assert!(message.contains("node ./audio_level.ts"), "{message}");
        assert!(message.contains("captures no stdout"), "{message}");
        assert!(message.contains("subprocess.run"), "{message}");
    }

    /// The shell-string law-2 sentence teaches the argv-list respelling.
    #[test]
    fn the_shell_string_sentence_teaches_the_argv_list_respelling() {
        let message = os_system_shell_string_unreadable();
        assert!(message.contains("shell string"), "{message}");
        assert!(message.contains("argv list"), "{message}");
    }

    /// The computed-script-path law-2 sentence teaches the written
    /// string literal respelling.
    #[test]
    fn the_script_path_sentence_teaches_the_written_literal_respelling() {
        let message = script_path_not_a_literal();
        assert!(message.contains("computed"), "{message}");
        assert!(message.contains("written string literal"), "{message}");
    }

    /// The stdin-payload-at-argv-json-target mismatch names both
    /// carriers and states they do not meet.
    #[test]
    fn the_stdin_at_argv_target_mismatch_names_both_carriers() {
        let message = foreign_edge_channel_mismatch_stdin_at_argv_target();
        assert!(message.contains("stdin"), "{message}");
        assert!(message.contains("argv element"), "{message}");
        assert!(message.contains("do not meet"), "{message}");
    }

    /// The argv-payload-at-stdin-json-target mismatch names both
    /// carriers and states they do not meet.
    #[test]
    fn the_argv_at_stdin_target_mismatch_names_both_carriers() {
        let message = foreign_edge_channel_mismatch_argv_at_stdin_target();
        assert!(message.contains("argv element"), "{message}");
        assert!(message.contains("stdin"), "{message}");
        assert!(message.contains("do not meet"), "{message}");
    }

    /// A mismatched argv index names both positions.
    #[test]
    fn the_argv_index_mismatch_names_both_positions() {
        let message = foreign_edge_channel_mismatch_argv_index(1, 2);
        assert!(message.contains("argv[1]"), "{message}");
        assert!(message.contains("argv[2]"), "{message}");
    }

    /// The file-carrier-at-stdin-target mismatch names the temp-file
    /// carrier and states it does not meet stdin.
    #[test]
    fn the_file_at_stdin_target_mismatch_names_both_carriers() {
        let message = foreign_edge_channel_mismatch_file_at_stdin_target();
        assert!(message.contains("temp file"), "{message}");
        assert!(message.contains("stdin"), "{message}");
        assert!(message.contains("do not meet"), "{message}");
    }

    /// The stdin-at-file-target mismatch names both carriers,
    /// symmetrically.
    #[test]
    fn the_stdin_at_file_target_mismatch_names_both_carriers() {
        let message = foreign_edge_channel_mismatch_stdin_at_file_target();
        assert!(message.contains("stdin"), "{message}");
        assert!(message.contains("file"), "{message}");
        assert!(message.contains("do not meet"), "{message}");
    }

    /// The file-carrier-at-argv-target mismatch names the file carrier
    /// and states the target reads the argv element as JSON text
    /// directly.
    #[test]
    fn the_file_at_argv_target_mismatch_names_both_carriers() {
        let message = foreign_edge_channel_mismatch_file_at_argv_target();
        assert!(message.contains("temp file"), "{message}");
        assert!(message.contains("JSON text itself"), "{message}");
        assert!(message.contains("do not meet"), "{message}");
    }

    /// The argv-carrier-at-file-target mismatch names both carriers,
    /// symmetrically.
    #[test]
    fn the_argv_at_file_target_mismatch_names_both_carriers() {
        let message = foreign_edge_channel_mismatch_argv_at_file_target();
        assert!(message.contains("directly as an argv element"), "{message}");
        assert!(message.contains("file path"), "{message}");
        assert!(message.contains("do not meet"), "{message}");
    }

    /// A mismatched file-carrier index names both positions.
    #[test]
    fn the_file_index_mismatch_names_both_positions() {
        let message = foreign_edge_channel_mismatch_file_index(1, 2);
        assert!(message.contains("argv[1]"), "{message}");
        assert!(message.contains("argv[2]"), "{message}");
    }

    /// The double-channel sentence names both channels as stated.
    #[test]
    fn the_double_channel_sentence_names_both_channels() {
        let message = foreign_edge_double_channel_declared();
        assert!(message.contains("argv element"), "{message}");
        assert!(message.contains("input=json.dumps"), "{message}");
    }

    /// The uncarriable-corner sentence names the function, the corner,
    /// and the leg that cannot carry it.
    #[test]
    fn the_uncarriable_corner_sentence_names_the_function_and_the_corner() {
        let message = foreign_edge_return_admits_uncarriable_corner("audioLevel", "+Infinity");
        assert!(message.contains("audioLevel"), "{message}");
        assert!(message.contains("+Infinity"), "{message}");
        assert!(message.contains("JSON stdout leg"), "{message}");
        assert!(message.contains("cannot be trusted"), "{message}");
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

    /// The empty-set sentence spells the compiled set's own contents.
    #[test]
    fn the_empty_set_sentence_spells_the_sets_own_contents() {
        let set = make_refined_set(vec![integer(), at_least(10.0), at_most(5.0)]);
        let message = empty_set(&set);
        assert!(message.contains("denotes the empty set"), "{message}");
    }

    /// The unhonorable-statement sentence names the refused spelling.
    #[test]
    fn the_unhonorable_annotation_sentence_names_the_spelling() {
        let message = unhonorable_annotation("Annotated[int, Ge(0), NotARealConstructor()]");
        assert!(
            message.contains("Annotated[int, Ge(0), NotARealConstructor()]"),
            "{message}"
        );
        assert!(message.contains("could not compile it"), "{message}");
    }
}
