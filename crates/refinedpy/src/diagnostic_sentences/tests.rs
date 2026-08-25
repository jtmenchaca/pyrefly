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

/// `slot_word` names a 2-slot tuple's positions ordinally, a
/// 3-slot tuple's center as "the middle slot," and a longer
/// tuple's final position as "the last slot."
#[test]
fn slot_word_names_first_second_middle_and_last() {
    assert_eq!(slot_word(0, 2), "the first slot");
    assert_eq!(slot_word(1, 2), "the second slot");
    assert_eq!(slot_word(0, 3), "the first slot");
    assert_eq!(slot_word(1, 3), "the middle slot");
    assert_eq!(slot_word(2, 3), "the last slot");
    assert_eq!(slot_word(3, 5), "the fourth slot");
    assert_eq!(slot_word(4, 5), "the last slot");
}

/// `at_slot` rewrites a plain per-value refutation into the
/// ordinal-slot phrasing — the showcase's own `paint((255, 300,
/// 0))` row.
#[test]
fn at_slot_rewrites_a_plain_refutation_into_ordinal_slot_wording() {
    let channel = make_refined_set(vec![at_least(0.0), at_most(255.0)]);
    let message = refutation("300", "Channel", &channel);
    let rewritten = at_slot(&message, 1, 3, &channel);
    assert_eq!(rewritten, "300 in the middle slot is not assignable to type '>= 0 && <= 255'");
}

/// A message shape `at_slot` does not recognize (a sort-crossing
/// reason clause) falls back to the plain `at_index` suffix rather
/// than being corrupted by a shape-blind rewrite.
#[test]
fn at_slot_falls_back_to_at_index_for_an_unrecognized_message_shape() {
    let channel = make_refined_set(vec![at_least(0.0), at_most(255.0)]);
    let sort_crossing = cross_sort_of_value("30.0", "a float", "an integer", "Channel", &channel);
    let rewritten = at_slot(&sort_crossing, 1, 3, &channel);
    assert_eq!(rewritten, at_index(&sort_crossing, 1));
}

/// The element-set refutation spells "list of (...)" on both sides
/// — Python's own container word, not the shared `refined_sets`
/// "array of" spelling — matching the showcase's `bump_all` row.
#[test]
fn the_element_set_refutation_spells_list_of_on_both_sides() {
    let flowing = make_refined_set(vec![integer(), at_least(2.0), at_most(6.0)]);
    let declared = make_refined_set(vec![integer(), at_least(1.0), at_most(5.0)]);
    let message = element_set_refutation(&flowing, &declared);
    assert_eq!(
        message,
        "a value of type 'list of (>= 2 && <= 6 && integer)' is not assignable to type 'list of (>= 1 && <= 5 && integer)'"
    );
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

/// The zero-admitting-divisor sentence names the raise, cites the
/// library clause, and teaches the guard that discharges it.
#[test]
fn the_division_by_a_set_that_admits_zero_sentence_names_the_raise_and_the_guard() {
    let message = division_by_a_set_that_admits_zero();
    assert!(message.contains("admits 0"), "{message}");
    assert!(message.contains("ZeroDivisionError"), "{message}");
    assert!(message.contains("expressions.rst"), "{message}");
    assert!(message.contains("if divisor != 0:"), "{message}");
}

/// The loop-stabilization sentence names the written name that never
/// settled and says its post-loop value is not yet readable.
#[test]
fn the_loop_accumulation_sentence_names_the_written_name() {
    let message = loop_accumulation_did_not_stabilize("total");
    assert!(message.contains("'total'"), "{message}");
    assert!(message.contains("fixed point"), "{message}");
    assert!(message.contains("not yet readable"), "{message}");
}

/// The unmodeled-module sentence names the module and says plainly
/// this checker carries no model for it — the naming unit's own
/// replacement for the generic `value_not_readable` wording.
#[test]
fn the_unmodeled_module_sentence_names_the_module() {
    let message = unmodeled_module_call("torch");
    assert!(message.contains("'torch'"), "{message}");
    assert!(message.contains("no model for"), "{message}");
}

/// The dict-changed-size-during-iteration sentence names the raise,
/// the raising dict, and the invariant the loop body breaks.
#[test]
fn the_dict_changed_size_during_iteration_sentence_names_the_dict_and_the_raise() {
    let message = dict_changed_size_during_iteration("counts");
    assert!(message.contains("RuntimeError"), "{message}");
    assert!(message.contains("'counts'"), "{message}");
    assert!(message.contains("changed size during"), "{message}");
}

/// The generator-body-never-summarized sentence names the generator's
/// own yield as the unread construct, distinct from
/// `unmodeled_module_call`'s "no model at all" wording.
#[test]
fn the_generator_body_never_summarized_sentence_names_the_yield() {
    let message = generator_body_never_summarized();
    assert!(message.contains("generator body"), "{message}");
    assert!(message.contains("never summarized"), "{message}");
    assert!(message.contains("yield is unread"), "{message}");
}

/// The manifest-names-no-entry sentence names both the module and the
/// unlisted function.
#[test]
fn the_manifest_no_entry_sentence_names_the_module_and_function() {
    let message = manifest_names_no_entry_for("widgets", "unlisted_fn");
    assert!(message.contains("'widgets'"), "{message}");
    assert!(message.contains("'unlisted_fn'"), "{message}");
}

/// The manifest-entry-crossing-refused sentence names the module,
/// function, parameter, and both the value's own words and the
/// declared sort it escaped.
#[test]
fn the_manifest_crossing_refused_sentence_names_every_position() {
    let message = manifest_entry_crossing_refused("widgets", "scale", "factor", "a string", "int");
    assert!(message.contains("widgets.scale"), "{message}");
    assert!(message.contains("'factor: int'"), "{message}");
    assert!(message.contains("a string"), "{message}");
}

/// The manifest-entry-names-no-producer sentence names the module,
/// function, and the missing producer symbol.
#[test]
fn the_manifest_no_producer_sentence_names_the_producer_symbol() {
    let message = manifest_entry_names_no_producer("widgets", "scale", "widgets_scale_impl");
    assert!(message.contains("widgets.scale"), "{message}");
    assert!(message.contains("widgets_scale_impl"), "{message}");
}

/// The strptime STAGE 2 unread-directive sentence names the ONE
/// directive letter that blocked the read, never the whole format
/// string.
#[test]
fn the_strptime_unread_directive_sentence_names_the_one_directive() {
    let message = strptime_unread_directive('z');
    assert!(message.contains("'%z'"), "{message}");
    assert!(message.contains("not yet transcribed"), "{message}");
}

/// The strptime STAGE 2 locale-directive sentence names the
/// directive AND states the distinct reason (no host-independent
/// set exists at all), never the plain "not yet transcribed"
/// wording the unread-directive sentence carries.
#[test]
fn the_strptime_locale_directive_sentence_states_the_distinct_reason() {
    let message = strptime_locale_directive('a');
    assert!(message.contains("'%a'"), "{message}");
    assert!(message.contains("locale"), "{message}");
    assert!(!message.contains("not yet transcribed"), "{message}");
}
