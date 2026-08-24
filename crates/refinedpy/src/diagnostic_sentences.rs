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

/// A single-element, path-shaped argv (`["./targets/cpp_level"]`) names
/// the code that runs next — the compiled binary at that path — so the
/// recognizer reaches the artifact lookup exactly as a `node`/`deno`/
/// `bun`/`npx tsx` row does. No producer in this checker regenerates a
/// compiled binary's fact (`foreign_edge_artifact.rs::read_compiled_
/// binary_fact`'s own doc: no source this checker reads, no producer
/// binary), so the checker looks for a SIBLING fact file at
/// `<binary_path>.facts.json`, hand- or tool-authored; this sentence
/// names that construct when the sibling file is absent, rather than
/// the generic "there is no <path>.refined.json; write it with
/// -export-fact" sentence, which names a command that has no meaning
/// for a target that is not TypeScript source.
pub fn compiled_binary_no_fact(target_path: &str) -> String {
    format!(
        "{target_path} is a compiled binary, and there is no {target_path}.facts.json beside it — the \
        checker can name the code that runs next but has no fact stating what it does"
    )
}

/// `os.system`'s file-legs decline when the redirected IN-FILE has no
/// recognized write preceding the call in the same body: the runner,
/// script, and both redirections read cleanly, but there is no
/// `with open("<infile>", "w") as <handle>: json.dump(<payload>, <handle>)`
/// this checker can find, so no entry fact has anything to attach to.
pub fn os_system_missing_entry_write(infile: &str) -> String {
    format!(
        "this call redirects stdin from {infile}, but no `with open(\"{infile}\", \"w\") as <name>: \
        json.dump(<payload>, <name>)` precedes it in this body — the checker cannot find the value written \
        to the in-file"
    )
}

/// `os.system`'s file-legs decline when the redirected OUT-FILE has no
/// recognized read following the call in the same body: the runner,
/// script, both redirections, and the entry write all read cleanly, but
/// there is no `with open("<outfile>") as <handle>: ... json.load(<handle>)`
/// this checker can find, so the return leg has no consumer to attach a
/// fact to.
pub fn os_system_missing_return_read(outfile: &str) -> String {
    format!(
        "this call redirects stdout to {outfile}, but no `with open(\"{outfile}\") as <name>: ... \
        json.load(<name>)` follows it in this body — the checker cannot find the value read back from the \
        out-file"
    )
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

/// A member's own refutation, PREFIXED by which key escaped and the
/// value that did — "the key 'spo2' of value 130 is not assignable to
/// type '>= 0 && <= 100'" — the showcase's own `Vitals(heart_rate=72,
/// spo2=130)` row, where the value arrives through pydantic's
/// `BaseModel` CONSTRUCTOR rather than a bare dict literal (the shape
/// `at_key` already serves for `m-pydantic-schema.py`'s
/// `Person.model_validate({...})`/`{"age": 200, ...}` rows). Like
/// `at_slot`, this recognizes ONLY `refutation`'s own plain composed
/// shape ("a value of type '<value>' is not assignable to type ...", no
/// reason clause riding after it) and rewrites it; any other shape (a
/// sort crossing, a nested structural mismatch) falls back to the
/// ordinary `at_key` suffix unchanged, so a shape this function does
/// not precisely recognize is never corrupted by a blind rewrite.
pub fn at_member(message: &str, key: &str, member_set: &RefinedSet) -> String {
    if let Some(value_words) = plain_refutation_value_words(message) {
        return format!(
            "the key '{key}' of value {value_words} is not assignable to type '{}'",
            format_for_diagnostics(member_set)
        );
    }
    at_key(message, key)
}

/// A member's own refutation, keyed by the index it escaped at — the
/// list/tuple element law's own suffix.
pub fn at_index(message: &str, index: usize) -> String {
    format!("{message} (at index {index})")
}

/// The ordinal name a fixed-arity tuple's own slot earns in prose: "the
/// first slot"/"the second slot"/… for a short tuple, "the middle slot"
/// for the exact center position of an ODD-length tuple (never a
/// numbered ordinal there — a 3-tuple's own center reads as "the middle
/// slot," matching the fixed-arity POSITIONS LAW's own corpus wording),
/// and "the last slot" for the final position of a tuple longer than
/// two. `length` is the tuple's own arity; `index` is the zero-based
/// slot this word names.
pub fn slot_word(index: usize, length: usize) -> String {
    if length > 2 && length % 2 == 1 && index == length / 2 {
        return "the middle slot".to_owned();
    }
    if length > 2 && index == length - 1 {
        return "the last slot".to_owned();
    }
    const ORDINALS: [&str; 10] = [
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
    ];
    match ORDINALS.get(index) {
        Some(word) => format!("the {word} slot"),
        None => format!("slot {index}"),
    }
}

/// A fixed-arity tuple's own PER-SLOT refutation suffix — the
/// POSITIONS LAW's own twin of `at_index`. The recursive
/// `judge(item, position_declared, kernel)` call that produced
/// `message` may have taken any of `judge`'s own laws (an ordinary
/// scalar refutation, a sort crossing, a nested member/element fire),
/// so this recognizes ONLY the one shape `refutation` itself composes
/// ("a value of type '<value>' is not assignable to type ..." — the
/// plain per-value membership fire, what a scalar-ground slot like
/// `Channel` actually produces) and rewrites it into the ordinal-slot
/// phrasing the showcase's own `paint((255, 300, 0))` row states: "300
/// in the middle slot is not assignable to type '>= 0 && <= 255 &&
/// integer'" — the value bare (no "a value of type" preamble, since the
/// slot word already carries the position), the ordinal slot name, and
/// the slot's own declared contents (never the outer tuple's alias
/// name — a slot's own set is what a reader needs at that exact
/// position). Any OTHER message shape (a sort-crossing reason clause, a
/// nested structural mismatch) falls back to `at_index`'s own plain
/// suffix, unchanged — this never rewrites a shape it does not
/// precisely recognize, so no other law's own wording is corrupted by
/// a shape-blind string edit.
pub fn at_slot(message: &str, index: usize, length: usize, slot_set: &RefinedSet) -> String {
    if let Some(value_words) = plain_refutation_value_words(message) {
        return format!(
            "{value_words} in {} is not assignable to type '{}'",
            slot_word(index, length),
            format_for_diagnostics(slot_set)
        );
    }
    at_index(message, index)
}

/// Recognizes `refutation`'s own exact composed shape — "a value of
/// type '<value>' is not assignable to type ..." — and answers `<value>`
/// bare, or `None` for a message any other law in this file composed
/// (a sort crossing appends its own reason clause after this same
/// prefix, so it is EXCLUDED here on purpose: a slot's sort-crossing
/// fire keeps `at_index`'s ordinary suffix, since the reason clause
/// names both sides' sorts in prose that already reads correctly
/// without an ordinal rewrite).
fn plain_refutation_value_words(message: &str) -> Option<&str> {
    let rest = message.strip_prefix("a value of type '")?;
    let (value_words, rest) = rest.split_once("' is not assignable to type ")?;
    if rest.contains(" — ") {
        return None; // a reason clause rides after this prefix — not the plain shape
    }
    Some(value_words)
}

/// A container-typed sink's own ELEMENT-SET refutation: the flowing
/// value's own element set is not a SUBSET of the declared element
/// set — unlike `at_index`'s per-position naming (one escaping item at
/// one index), this names the whole admitted element range on both
/// sides, the shape a bare `list[X]` sink states when the flowing
/// value's own length is unknown (a comprehension over an
/// unbounded-length parameter, `check.rs::seed_parameters`'s own
/// `Kind::Set` sequence seed) so there is no single index to blame.
/// Spelled "list of (<contents>)" — Python's own container word, never
/// the shared `refined_sets` "array of" spelling
/// (`format_for_diagnostics`'s own `Form::Star` wording, which this
/// sentence does not call for the outer container itself). `judge` is
/// sink-agnostic by this file's own design (one seam every sink routes
/// through), so this names the flowing thing "a value" — the same word
/// every sibling law in this module uses — rather than a sink-specific
/// word (return/argument/key) `judge` has no way to know here.
pub fn element_set_refutation(value_element_set: &RefinedSet, declared_element_set: &RefinedSet) -> String {
    format!(
        "a value of type 'list of ({})' is not assignable to type 'list of ({})'",
        format_for_diagnostics(value_element_set),
        format_for_diagnostics(declared_element_set)
    )
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

/// The zero-admitting-divisor fire — `binop_possible_raise`'s own row
/// for a `/`, `//`, or `%` divisor window that ADMITS zero without
/// being entirely zero: the divisor's set admits `0`, and CPython
/// raises `ZeroDivisionError` there for all three operators alike
/// (`expressions.rst` §6.7, arith.10 — the divergence from ECMA's own
/// determined `±Infinity`/NaN answer at that same corner for `/`). For
/// `/`, `expressions.rs`'s `split_divisor_transfer` keeps determining
/// the value question over the divisor's zero-excluded halves
/// alongside this fire; `//` and `%` have no such split, so their
/// value question keeps declining outright over the same window — this
/// sentence names the escape neither value path can speak to, in one
/// wording shared by all three. Names the guard that discharges it, the
/// same teaching move `os_system_no_stdout_capture` makes for its own
/// fixable respelling.
pub fn division_by_a_set_that_admits_zero() -> String {
    "this expression's divisor set admits 0 — CPython raises ZeroDivisionError there (expressions.rst \
    §6.7); a zero guard on the divisor (for example `if divisor != 0:`) discharges this before the \
    division runs"
        .to_owned()
}

/// A `for` loop's own abstract pass names, per iteration, a written name
/// whose value never reached a fixed point across the two judged passes
/// (`loops.rs::stabilized_join`'s own doc) — the loop reaches a real
/// stopping point, but that name's true accumulated value past it is
/// unreadable. Names the written name so the reader knows which
/// accumulation to widen or bound explicitly, mirroring the plain,
/// per-position wording every other decline in this module already
/// takes.
pub fn loop_accumulation_did_not_stabilize(name: &str) -> String {
    format!(
        "the for loop's own value for '{name}' does not settle to a fixed point across its own two \
        judged passes, so its value past the loop is not yet readable"
    )
}

/// A `for` loop iterating a dict directly (`for k in d:`/`for k in
/// d.keys():`/`for v in d.values():`/`for k, v in d.items():`) whose own
/// body provably CHANGES THAT SAME DICT'S SIZE on every reachable pass —
/// `del d[key]`, `d.pop(...)`, `d.popitem()`, `d.clear()` —
/// `loops.rs::dict_size_changing_mutation_range`'s own recognized set.
/// CPython raises `RuntimeError` the moment the size changes mid-
/// iteration (library/stdtypes.rst, dict views: "the dictionary should
/// not be modified during iteration... it is safe... only if you don't
/// add or remove entries"), a defined behavior this checker states as a
/// provable raise, matching `binop_provable_raise`'s own "every operand
/// known, every run raises" discipline. Names the iterated dict so the
/// reader does not have to re-derive which name the loop reads from the
/// mutation alone.
pub fn dict_changed_size_during_iteration(dict_name: &str) -> String {
    format!(
        "this expression provably raises RuntimeError: dictionary '{dict_name}' changed size during \
        iteration — the loop body changes the same dict's size on every reachable pass"
    )
}

/// A `for` loop iterating a list directly (`for x in lst:`) whose own
/// body provably APPENDS TO THAT SAME LIST on every reachable pass —
/// `loops.rs::list_size_changing_mutation_range`'s own recognized
/// `.append(...)` call. Unlike a dict (which raises `RuntimeError`
/// outright, `dict_changed_size_during_iteration`'s own citation), a
/// list's iterator carries no such guard (library/stdtypes.rst's list
/// iterator has no length snapshot the way a `range(len(...))` counter
/// would) — every pass finds a fresh element the SAME pass just
/// appended, so the loop never reaches its own end. Names the iterated
/// list so the reader does not have to re-derive which name the loop
/// reads from the mutation alone.
pub fn list_never_terminates_self_append(list_name: &str) -> String {
    format!(
        "this loop never terminates: list '{list_name}' is appended to inside its own for-loop body — \
        the iterator keeps finding new elements appended ahead of it"
    )
}

/// The generic `value_not_readable` sentence's own NAMED replacement, for
/// the one shape that generic wording leaves anonymous: a flowing value
/// that reached a sink undetermined because it was produced by a call
/// into an imported module this checker carries no model for
/// (`torch.arange(5)`, `pandas.read_csv(...)`) — the python-c-extension-
/// boundary.md naming unit's own sentence, the first rung of the
/// compiled-extension recognition ladder. Names the module rather than
/// leaving the reader to guess which construct blocked the walk.
pub fn unmodeled_module_call(module_name: &str) -> String {
    format!("a call into '{module_name}', a module this checker has no model for")
}

/// The manifest reader's own DECLINE sentence for a module that IS named
/// in a manifest but the CALLED function is not one of the manifest's
/// listed entries — a narrower named decline than the bare unmodeled-
/// module sentence, since the manifest at least states what it does
/// cover.
pub fn manifest_names_no_entry_for(module_name: &str, function_name: &str) -> String {
    format!(
        "'{module_name}''s manifest names no entry for '{function_name}' — the call is a manifested module's \
        own function this checker still has no contract for"
    )
}

/// `datetime.strptime(text, format)` date.12 STAGE 2's own named decline
/// for a format string naming a directive this round has not
/// transcribed against datetime.rst's format-codes table yet (`%z %Z
/// %I %G %u %V` — `expressions.rs::Strptime2Decline::UnreadDirective`'s
/// own set). Names the ONE directive that blocked the read, never the
/// whole format string — a host-independent value set is buildable for
/// this directive once transcribed; today it simply is not yet.
pub fn strptime_unread_directive(letter: char) -> String {
    format!(
        "this format string names the directive '%{letter}', which this checker has not yet transcribed \
        against datetime.rst's format-codes table"
    )
}

/// `datetime.strptime(text, format)` date.12 STAGE 2's own named decline
/// for a format string naming a LOCALE-dependent directive (`%a %A %b
/// %B %p %c %x %X` — `expressions.rs::Strptime2Decline::LocaleDirective`'s
/// own set) — datetime.rst note (1): "the format depends on the current
/// locale... Field orderings will vary... and the output may contain
/// non-ASCII characters." A genuinely distinct reason from
/// `strptime_unread_directive`'s: no host-independent value set exists
/// for a locale directive AT ALL, not merely one this round left
/// untranscribed.
pub fn strptime_locale_directive(letter: char) -> String {
    format!(
        "this format string names the directive '%{letter}', which reads a value from the host's locale \
        (datetime.rst note 1) — there is no host-independent set for a locale-dependent directive to derive"
    )
}

/// The manifest reader's own DECLINE sentence for a manifest file this
/// reader could not parse at all — the whole manifest is unusable, so
/// every call into the module it would have covered stays the bare
/// unmodeled-module decline instead.
pub fn manifest_unreadable(manifest_path: &str, reason: &str) -> String {
    format!("the manifest {manifest_path} could not be read: {reason}")
}

/// A crossing argument escapes the manifest entry's own declared sort —
/// the manifest lane's own crossing-fit refusal, the same shape the
/// stdio edge's `containment_refutation` fires, restated for a
/// manifest-declared parameter (a plain sort word, never a full
/// `DeclaredRefinement` spelling).
pub fn manifest_entry_crossing_refused(
    module_name: &str,
    function_name: &str,
    parameter_name: &str,
    value_words: &str,
    declared_sort: &str,
) -> String {
    format!(
        "a value of type '{value_words}' is not assignable to '{module_name}.{function_name}''s declared \
        parameter '{parameter_name}: {declared_sort}' — the manifest states the entry contract, and \
        {value_words} escapes it"
    )
}

/// The manifest reader's own DECLINE sentence for a call whose return
/// crosses the entry contract fit but has no producer half yet — the
/// manifest states the ENTRY, never the return; a later unit (the
/// producer half, python-c-extension-boundary.md build order item 3)
/// closes this. Names both the module/function and the missing producer
/// symbol so the decline reads as a work-queue item.
pub fn manifest_entry_names_no_producer(module_name: &str, function_name: &str, producer_symbol: &str) -> String {
    format!(
        "'{module_name}.{function_name}''s manifest names its entry but no producer exports its return fact \
        (the manifest names the producer symbol '{producer_symbol}', and no C++/native adapter has exported a \
        fact for it yet)"
    )
}

/// The generic `value_not_readable` sentence's own NAMED replacement for
/// the generator-body boundary q-decline-names.py's own
/// `generator_body_never_summarized` row teaches: a value read off a
/// generator (directly, or through `next`/`anext`) whose body
/// `instances::generator_yields` declined to summarize (a conditional
/// `yield`, or any other shape outside the straight-line reading that
/// function's own doc describes) — never the plain absence of a model
/// `unmodeled_module_call` names, since the generator IS a same-module
/// def this checker recognizes and attempted to summarize. Mirrors
/// `unmodeled_module_call`'s own naming-unit precedent: the generic
/// wording is sharpened to name the ONE construct that blocked the read.
pub fn generator_body_never_summarized() -> String {
    "the generator body was never summarized, so its yield is unread".to_owned()
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
}
