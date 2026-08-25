//! The empty-set / unhonorable-annotation sentences, the provable-raise
//! wording (`division_by_a_set_that_admits_zero`, the dict/list
//! iteration raises), the loop-stabilization blocker, and the two named
//! replacements for the generic `value_not_readable` decline
//! (`unmodeled_module_call`, `generator_body_never_summarized`).

use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::RefinedSet;

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
