//! The trace's own unit tests: the span shape, the projection template,
//! the emitter's schema conformance, and the off-is-free gate.

use std::sync::Arc;
use std::sync::Mutex;

use super::*;
use super::span::Span;
use super::span::SpanStatus;
use super::span::TraceDocument;

fn a_request(source: &str, line: usize) -> TraceRequest {
    TraceRequest::new(
        "fixture.py".to_owned(),
        source.to_owned(),
        crate::markers::line_starts_of(source),
        line,
    )
}

#[test]
fn tracing_is_off_by_default_and_every_entry_point_is_a_no_op() {
    assert!(!is_tracing(), "an untouched thread never records");
    // Each of these would panic or record if the gate leaked; off, all
    // four return having read one Cell and done nothing.
    let scope = span_scope("a_reader", 0, 10);
    record_answer("[0, 150]");
    record_decline("a gate", Some((0, 4)), Some("a value"));
    record_kernel_ask("member", "a question", Some("decided"));
    drop(scope);
    assert!(!is_tracing());
}

#[test]
fn an_installed_collector_records_one_span_on_the_requested_line() {
    let source = "x = 1\ny = 2\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 2))));
    {
        let _guard = install(collector.clone());
        assert!(is_tracing());
        // line 2 is bytes 6..11 — inside the request
        let scope = span_scope("name_read", 6, 11);
        record_answer("2");
        drop(scope);
    }
    let document = take_trace(collector).expect("a span was recorded on the requested line");
    assert_eq!(document.language, "py");
    assert_eq!(document.position, "fixture.py:2");
    assert_eq!(document.root.name, "name_read");
    assert_eq!(document.root.status, SpanStatus::Answered);
    assert_eq!(document.root.construct, "y = 2");
    assert_eq!(document.root.range, "fixture.py:2:1-2:6");
    assert_eq!(document.root.answer.as_deref(), Some("2"));
}

#[test]
fn a_span_off_the_requested_line_is_not_recorded() {
    let source = "x = 1\ny = 2\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 2))));
    {
        let _guard = install(collector.clone());
        let scope = span_scope("name_read", 0, 5); // line 1
        record_answer("1");
        drop(scope);
    }
    assert!(
        take_trace(collector).is_none(),
        "recording is per position — a range off the requested line records nothing"
    );
}

/// The `record_blocker` shape: a dispatcher-owned span that opens,
/// declines, and closes with no reader span of its own beneath it —
/// `check::walk::statement::record_blocker`'s "blocked_construct" span is
/// exactly this. Reproduced here as a second, later-finished top-level
/// span alongside an EARLIER, answered, childless one — the shape an
/// unrelated internal evaluation (loops.rs evaluating a for-loop's own
/// iterable through `evaluate_expression`, which opens its own top-level
/// `name_read` span with no position span wrapping it) leaves on the same
/// requested line.
///
/// Before the selection rule included the span's own top
/// (`deepest_decline_including_top`), `deepest_decline`'s children-only
/// scan found a decline in NEITHER top-level span (the first has none at
/// all; the second's decline sits on its own top, invisible to a
/// children-only search), so `into_document` fell back to index 0 — the
/// answered, childless span — and published it as an answered root
/// carrying no trace of the real blocker anywhere in the document.
#[test]
fn a_blocker_span_with_no_reader_beneath_it_still_becomes_the_chosen_root() {
    let source = "for x in xs:\n    pass\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        // The stray, unrelated, answered leaf — finishes first.
        let stray = span_scope("name_read", 9, 11); // `xs`
        record_answer("[0, 200] × 0 or more");
        drop(stray);
        // The blocker span — opens, declines on its own top, and closes
        // with no child, exactly as `record_blocker` does.
        let blocker = span_scope("blocked_construct", 0, 22);
        record_decline("a for statement is not yet walked", Some((0, 22)), None);
        drop(blocker);
    }
    let document = take_trace(collector).expect("two top-level spans were recorded");
    assert_eq!(document.root.status, SpanStatus::Answered, "the root rule still holds");
    assert_eq!(
        document.root.children.len(),
        1,
        "the blocker span is folded in as the root's one child, not stripped bare in its place"
    );
    let child = &document.root.children[0];
    assert_eq!(child.name, "blocked_construct");
    assert_eq!(child.status, SpanStatus::Declined);
    assert_eq!(child.gate.as_deref(), Some("a for statement is not yet walked"));
    assert_eq!(
        projection_of_deepest_decline(&document.root).as_deref(),
        Some("for x in xs: pass: a for statement is not yet walked — fixture.py:1:1-3:1"),
        "the projection finds the blocker's own gate through the wrapped child"
    );
}

#[test]
fn the_guard_clears_the_thread_local_on_the_way_out() {
    let source = "x = 1\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        assert!(is_tracing());
    }
    assert!(!is_tracing(), "the Drop guard clears the slot");
}

#[test]
fn a_nested_span_becomes_a_child_in_evaluation_order() {
    let source = "return offset_minutes\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        let outer = span_scope("assignability::judge", 0, 21);
        let inner = span_scope("name_read", 7, 21);
        record_decline("a named gate", Some((7, 21)), Some("no reading"));
        drop(inner);
        record_answer("outer answered");
        drop(outer);
    }
    let document = take_trace(collector).expect("spans were recorded");
    assert_eq!(document.root.name, "assignability::judge");
    assert_eq!(document.root.children.len(), 1);
    assert_eq!(document.root.children[0].name, "name_read");
    assert_eq!(document.root.children[0].status, SpanStatus::Declined);
}

#[test]
fn the_projection_follows_the_specs_template_exactly() {
    // <construct>: <gate> — <operand construct> held <held>
    let mut span = Span::new(
        "s1".to_owned(),
        "assignability::judge".to_owned(),
        "offset_minutes".to_owned(),
        "fixture.py:17:24-17:38".to_owned(),
    );
    span.status = SpanStatus::Declined;
    span.gate = Some("a named premise".to_owned());
    span.operand = Some("fixture.py:15:17-15:23".to_owned());
    span.held = Some("[0, 150]".to_owned());
    assert_eq!(
        project_sentence(&span),
        "offset_minutes: a named premise — fixture.py:15:17-15:23 held [0, 150]"
    );
}

#[test]
fn a_declined_span_with_no_gate_projects_the_specs_stated_fallback() {
    // "A dispatcher-instrumented step that has not yet adopted the
    // decline helper still produces a span ... its projection falls back
    // to `<construct>: <reader> declined`."
    let mut span = Span::new(
        "s1".to_owned(),
        "evaluate_call".to_owned(),
        "d.utcoffset()".to_owned(),
        "fixture.py:13:18-13:31".to_owned(),
    );
    span.status = SpanStatus::Declined;
    assert_eq!(project_sentence(&span), "d.utcoffset(): evaluate_call declined");
}

#[test]
fn the_deepest_declined_span_is_the_one_projected() {
    let mut root = Span::new("s1".to_owned(), "judge".to_owned(), "outer".to_owned(), "f.py:1:1-1:6".to_owned());
    root.status = SpanStatus::Declined;
    root.gate = Some("outer gate".to_owned());
    let mut child = Span::new("s2".to_owned(), "reader".to_owned(), "inner".to_owned(), "f.py:1:2-1:5".to_owned());
    child.status = SpanStatus::Declined;
    child.gate = Some("inner gate".to_owned());
    root.children.push(child);
    assert_eq!(
        projection_of_deepest_decline(&root).as_deref(),
        Some("inner: inner gate"),
        "the DEEPEST decline is the blocker, not the outermost"
    );
}

#[test]
fn a_deeper_sibling_subtrees_decline_outranks_a_shallow_one() {
    // Rule 1: between two declining subtrees under one parent, the one
    // whose own decline sits deeper wins — a bare operand read that
    // declined at depth 1 is a smaller answer than a call whose model
    // declined at depth 2 beneath it.
    let mut root = Span::new("s1".to_owned(), "walk".to_owned(), "stmt".to_owned(), "f.py:1:1-1:9".to_owned());
    root.status = SpanStatus::Declined;
    let mut shallow = Span::new("s2".to_owned(), "name_read".to_owned(), "time".to_owned(), "f.py:1:1-1:5".to_owned());
    shallow.status = SpanStatus::Declined;
    shallow.gate = Some("shallow gate".to_owned());
    let mut call = Span::new("s3".to_owned(), "evaluate_call".to_owned(), "time.f()".to_owned(), "f.py:1:1-1:9".to_owned());
    call.status = SpanStatus::Declined;
    call.gate = Some("no model answers this call".to_owned());
    let mut inner = Span::new("s4".to_owned(), "kernel.member".to_owned(), "time.f()".to_owned(), "f.py:1:1-1:9".to_owned());
    inner.status = SpanStatus::Declined;
    inner.gate = Some("the deepest gate".to_owned());
    call.children.push(inner);
    root.children.push(shallow);
    root.children.push(call);
    assert_eq!(
        projection_of_deepest_decline(&root).as_deref(),
        Some("time.f(): the deepest gate"),
        "the deeper subtree's decline is the blocker, not the earlier shallow sibling"
    );
}

#[test]
fn equal_depth_siblings_break_the_tie_by_evaluation_order() {
    // Rule 2: the FIRST decline wins at equal depth — the standing rule
    // that every undetermined names the FIRST construct that blocked it.
    let mut root = Span::new("s1".to_owned(), "walk".to_owned(), "stmt".to_owned(), "f.py:1:1-1:9".to_owned());
    root.status = SpanStatus::Declined;
    for (id, name, gate) in [("s2", "first", "the first gate"), ("s3", "second", "the second gate")] {
        let mut child = Span::new(id.to_owned(), name.to_owned(), name.to_owned(), "f.py:1:1-1:5".to_owned());
        child.status = SpanStatus::Declined;
        child.gate = Some(gate.to_owned());
        root.children.push(child);
    }
    assert_eq!(
        projection_of_deepest_decline(&root).as_deref(),
        Some("first: the first gate")
    );
}

#[test]
fn the_root_is_never_the_projected_span() {
    // DERIVATION-TRACE.md's root rule: "The root is `answered` ... No
    // decline is ever recorded onto the root." A tree whose ONLY declining
    // span is its own root therefore projects nothing — some reader
    // refused without opening a span, and pointing the sentence at the
    // whole judged statement instead of the construct that blocked is
    // exactly the mispointing the rule exists to prevent.
    let mut root = Span::new("s1".to_owned(), "check::walk_return".to_owned(), "return x".to_owned(), "f.py:1:1-1:9".to_owned());
    root.status = SpanStatus::Declined;
    root.gate = Some("this judged position is undetermined".to_owned());
    let mut answered_child =
        Span::new("s2".to_owned(), "name_read".to_owned(), "x".to_owned(), "f.py:1:8-1:9".to_owned());
    answered_child.status = SpanStatus::Answered;
    answered_child.answer = Some("[0, 150]".to_owned());
    root.children.push(answered_child);
    assert!(
        projection_of_deepest_decline(&root).is_none(),
        "a decline on the root alone is not a leaf the projection may point at"
    );
    // A CHAINED root is a binding's own derivation, not the judged
    // position, so its own top IS a candidate.
    assert_eq!(
        projection_of_chained_root(&root).as_deref(),
        Some("return x: this judged position is undetermined")
    );
}

#[test]
fn a_span_that_records_no_outcome_is_answered_not_declined() {
    // A dispatcher that opens a span and records nothing has refused
    // nothing. A default decline would make it a candidate leaf the
    // projection could point at, with no gate to name.
    let source = "return f()\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        let outer = span_scope("check::walk_return", 0, 10);
        let inner = span_scope("evaluate_call", 7, 10);
        drop(inner);
        drop(outer);
    }
    let document = take_trace(collector).expect("spans were recorded");
    assert_eq!(document.root.status, SpanStatus::Answered);
    assert_eq!(document.root.children[0].status, SpanStatus::Answered);
    assert!(projection_of_deepest_decline(&document.root).is_none());
}

#[test]
fn a_decline_onto_the_position_span_is_refused() {
    // "No decline is ever recorded onto the root — a decline with no open
    // reader span is refused, not attached upward."
    let source = "return f()\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        let judged = position_scope("check::walk_return", 0, 10);
        record_decline("this judged position is undetermined", Some((7, 10)), Some("no reading"));
        drop(judged);
    }
    let document = take_trace(collector).expect("the position span was recorded");
    assert_eq!(document.root.status, SpanStatus::Answered);
    assert_eq!(document.root.gate, None, "the gate never attached upward");
    assert_eq!(document.root.operand, None);
    assert_eq!(document.root.held, None);
}

#[test]
fn a_determined_position_carries_an_answered_root_with_the_requested_lines_own_text() {
    // The A1.guard.arm shape: the children answer and the kernel ask
    // decided, so the position determined — and its root says so, carrying
    // the requested line's own construct and full-line range.
    // `def f(x):\n` is bytes 0..10; line 2 is `    return x`, bytes 10..22,
    // and the bare `x` it returns is byte 21.
    let source = "def f(x):\n    return x\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 2))));
    {
        let _guard = install(collector.clone());
        let judged = position_scope("check::walk_return", 21, 22);
        let name = span_scope("name_read", 21, 22);
        record_answer("{100}");
        drop(name);
        let inner = span_scope("assignability::judge", 21, 22);
        record_kernel_ask("member", "member([0,150], 100)", Some("decided"));
        record_answer("{100}");
        drop(inner);
        drop(judged);
    }
    let document = take_trace(collector).expect("the position span was recorded");
    assert_eq!(document.root.status, SpanStatus::Answered);
    assert_eq!(document.root.gate, None, "a determined root carries no gate");
    assert_eq!(document.root.construct, "return x", "the requested line's own text");
    assert_eq!(document.root.range, "fixture.py:2:1-2:13", "the full-line range");
    assert!(projection_of_deepest_decline(&document.root).is_none());
}

#[test]
fn an_answer_clears_a_declines_attributes_from_the_same_span() {
    // "A span that declines on one arm and then answers CLEARS its
    // gate/operand/held — decline attributes never survive on an answered
    // span."
    let source = "return x\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        let scope = span_scope("evaluate_boolop", 0, 8);
        record_decline("the first arm derived no value", Some((7, 8)), Some("no reading"));
        record_answer("[0, 150]");
        drop(scope);
    }
    let document = take_trace(collector).expect("the span was recorded");
    assert_eq!(document.root.status, SpanStatus::Answered);
    assert_eq!(document.root.gate, None);
    assert_eq!(document.root.operand, None);
    assert_eq!(document.root.held, None);
}

#[test]
fn an_all_answered_tree_projects_no_sentence() {
    let mut root = Span::new("s1".to_owned(), "judge".to_owned(), "x".to_owned(), "f.py:1:1-1:2".to_owned());
    root.status = SpanStatus::Answered;
    root.answer = Some("[0, 150]".to_owned());
    assert!(
        projection_of_deepest_decline(&root).is_none(),
        "a determined position has no undetermined sentence to print"
    );
}

#[test]
fn the_emitted_json_carries_every_required_member_the_schema_names() {
    let mut root = Span::new(
        "s1".to_owned(),
        "assignability::judge".to_owned(),
        "offset_minutes".to_owned(),
        "fixture.py:17:24-17:38".to_owned(),
    );
    root.status = SpanStatus::Declined;
    root.language = Some("py");
    root.position = Some("fixture.py:17".to_owned());
    root.gate = Some("a named premise".to_owned());
    let mut ask = Span::new(
        "s2".to_owned(),
        "kernel.member".to_owned(),
        "offset_minutes".to_owned(),
        "fixture.py:17:24-17:38".to_owned(),
    );
    ask.status = SpanStatus::Answered;
    ask.question = Some("member([0,150], 175)".to_owned());
    ask.answer = Some("decided".to_owned());
    root.children.push(ask);
    let document =
        TraceDocument { language: "py", position: "fixture.py:17".to_owned(), root, chain: Vec::new() };
    let json = render_json(&document);
    for required in [
        "\"language\"",
        "\"position\"",
        "\"root\"",
        "\"id\"",
        "\"name\"",
        "\"status\"",
        "\"attributes\"",
        "\"refinery.construct\"",
        "\"refinery.range\"",
        "\"refinery.language\"",
        "\"refinery.position\"",
        "\"refinery.gate\"",
        "\"refinery.question\"",
        "\"children\"",
    ] {
        assert!(json.contains(required), "the emitted JSON is missing {required}:\n{json}");
    }
    assert!(json.contains("\"status\": \"declined\""), "the two-value status is spelled as the schema names it");
    assert!(json.contains("\"status\": \"answered\""));
}

#[test]
fn the_emitter_escapes_a_construct_carrying_a_quote() {
    let mut root = Span::new(
        "s1".to_owned(),
        "string_literal".to_owned(),
        "\"a\\b\"".to_owned(),
        "f.py:1:1-1:6".to_owned(),
    );
    root.status = SpanStatus::Answered;
    root.answer = Some("\"a\\b\"".to_owned());
    let document =
        TraceDocument { language: "py", position: "f.py:1".to_owned(), root, chain: Vec::new() };
    let json = render_json(&document);
    assert!(json.contains("\\\"a\\\\b\\\""), "quotes and backslashes are escaped:\n{json}");
}

#[test]
fn a_range_is_spelled_the_way_the_schema_pattern_requires() {
    // ^.+:[0-9]+:[0-9]+-[0-9]+:[0-9]+$
    let request = a_request("x = 1\ny = 2\n", 1);
    assert_eq!(request.range_words(0, 5), "fixture.py:1:1-1:6");
    assert_eq!(request.range_words(6, 11), "fixture.py:2:1-2:6");
}

#[test]
fn a_multi_line_construct_collapses_to_one_line() {
    let request = a_request("f(\n  1,\n  2,\n)\n", 1);
    assert_eq!(request.construct_words(0, 14), "f( 1, 2, )");
}

// ---- the binding ledger (DERIVATION-TRACE.md, "The projection rule") ----

/// The two-statement shape every ledger test below is built on:
///
/// ```text
/// line 1: x = f()        the binding whose derivation the ledger files
/// line 2: return x       the read whose leaf is the bare name `x`
/// ```
///
/// Byte offsets: line 1 is 0..7, `f()` is 4..7; line 2 is 8..16, `x` is
/// 15..16.
const TWO_STATEMENTS: &str = "x = f()\nreturn x\n";

/// Records the ledger's write for `x` and the read at line 2, then reads
/// the document back. `bind_declines` says whether the binding's own
/// right-hand side derived nothing (the shape that reclaims something
/// worth reading) or answered.
fn a_ledger_walk(bind_declines: bool) -> Option<TraceDocument> {
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(TWO_STATEMENTS, 2))));
    {
        let _guard = install(collector.clone());
        // The BINDING, off the requested line: recorded anyway, because a
        // ledger scope suspends the position filter.
        {
            let _binding = ledger_scope(vec!["x".to_owned()], "check::walk_assign", 4, 7);
            let call = span_scope("evaluate_call", 4, 7);
            match bind_declines {
                true => record_decline("no model answers this call, so it derives no value", None, Some("no reading")),
                false => record_answer("[0, 150]"),
            }
            drop(call);
            match bind_declines {
                true => record_decline("this assignment's right-hand side derives no value", None, Some("no reading")),
                false => record_answer("[0, 150]"),
            }
        }
        // The READ, on the requested line.
        let judged = position_scope("check::walk_return", 8, 16);
        let name = span_scope("name_read", 15, 16);
        record_read_place("x");
        record_decline("'x' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    take_trace(collector)
}

#[test]
fn a_ledger_entry_is_captured_at_a_write_off_the_requested_line() {
    let document = a_ledger_walk(true).expect("the read on line 2 recorded a span");
    // The binding sits on line 1, which the position filter excludes, so
    // the main tree is the read's own and nothing else.
    assert_eq!(document.root.name, "check::walk_return");
    assert_eq!(document.root.children.len(), 1);
    assert_eq!(document.root.children[0].name, "name_read");
    // The ledger nonetheless captured the write: the chain has it.
    assert_eq!(document.chain.len(), 1, "the write off the requested line was still filed");
    assert_eq!(document.chain[0].name, "check::walk_assign");
}

#[test]
fn a_bare_name_leaf_reclaims_the_binding_that_derived_it() {
    let document = a_ledger_walk(true).expect("the read recorded a span");
    let reclaimed = &document.chain[0];
    assert_eq!(reclaimed.range, "fixture.py:1:5-1:8");
    assert_eq!(
        projection_of_deepest_decline(reclaimed).as_deref(),
        Some("f(): no model answers this call, so it derives no value — held no reading"),
        "the chain names the ORIGINATING construct, which the main root's own leaf never could"
    );
}

#[test]
fn the_chain_never_moves_the_projected_sentence() {
    // "The projection still reads the MAIN root's deepest declined span;
    // the chain sharpens the work item, never the sentence."
    let document = a_ledger_walk(true).expect("the read recorded a span");
    assert_eq!(
        projection_of_deepest_decline(&document.root).as_deref(),
        Some("x: 'x' carries no derived value at this read — held no reading")
    );
}

#[test]
fn a_chained_root_whose_own_leaf_is_a_bare_name_reclaims_recursively() {
    // line 1: a = f()      bytes 0..7,  `f()` 4..7
    // line 2: b = a        bytes 8..13, `a`   12..13
    // line 3: return b     bytes 14..22, `b`  21..22
    let source = "a = f()\nb = a\nreturn b\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 3))));
    {
        let _guard = install(collector.clone());
        {
            let _binding = ledger_scope(vec!["a".to_owned()], "check::walk_assign", 4, 7);
            let call = span_scope("evaluate_call", 4, 7);
            record_decline("no model answers this call", None, Some("no reading"));
            drop(call);
            record_decline("this assignment's right-hand side derives no value", None, Some("no reading"));
        }
        {
            let _binding = ledger_scope(vec!["b".to_owned()], "check::walk_assign", 12, 13);
            let name = span_scope("name_read", 12, 13);
            record_read_place("a");
            record_decline("'a' carries no derived value at this read", None, Some("no reading"));
            drop(name);
            record_decline("this assignment's right-hand side derives no value", None, Some("no reading"));
        }
        let judged = position_scope("check::walk_return", 14, 22);
        let name = span_scope("name_read", 21, 22);
        record_read_place("b");
        record_decline("'b' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read on line 3 recorded a span");
    assert_eq!(document.chain.len(), 2, "nearest binding first, then the one behind it");
    assert_eq!(document.chain[0].range, "fixture.py:2:5-2:6", "`b = a` is the nearest binding");
    assert_eq!(document.chain[1].range, "fixture.py:1:5-1:8", "`a = f()` is reclaimed recursively");
    assert_eq!(
        projection_of_deepest_decline(&document.chain[1]).as_deref(),
        Some("f(): no model answers this call — held no reading"),
        "the LAST chained root names the true originating construct"
    );
}

#[test]
fn a_cycle_between_two_places_is_refused_by_place() {
    // `b = a` then `a = b` — the read of `a` reclaims `a = b`, whose leaf
    // reads `b`, which reclaims `b = a`, whose leaf reads `a` again. That
    // is the cycle, and the walk refuses the place it already reclaimed
    // rather than going round forever.
    // line 1: b = a      bytes 0..5,   `a` 4..5
    // line 2: a = b      bytes 6..11,  `b` 10..11
    // line 3: return a   bytes 12..20, `a` 19..20
    let source = "b = a\na = b\nreturn a\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 3))));
    {
        let _guard = install(collector.clone());
        for (place, read_place, start, end) in
            [("b", "a", 4usize, 5usize), ("a", "b", 10, 11)]
        {
            let _binding = ledger_scope(vec![place.to_owned()], "check::walk_assign", start, end);
            let name = span_scope("name_read", start, end);
            record_read_place(read_place);
            record_decline("carries no derived value at this read", None, Some("no reading"));
            drop(name);
            record_decline("this assignment's right-hand side derives no value", None, Some("no reading"));
        }
        let judged = position_scope("check::walk_return", 12, 20);
        let name = span_scope("name_read", 19, 20);
        record_read_place("a");
        record_decline("'a' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read on line 3 recorded a span");
    assert_eq!(document.chain.len(), 2, "the walk stops at the place it already reclaimed");
    assert_eq!(document.chain[0].range, "fixture.py:2:5-2:6", "`a = b` is `a`'s nearest binding");
    assert_eq!(document.chain[1].range, "fixture.py:1:5-1:6", "`b = a` is reclaimed behind it");
}

#[test]
fn a_leaf_that_is_not_a_bare_name_reclaims_nothing() {
    // The judged position's own leaf is a CALL with no model — it already
    // names its construct, so there is nothing for the ledger to sharpen
    // and the schema's optional `chain` is not emitted.
    let source = "return f()\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        let judged = position_scope("check::walk_return", 0, 10);
        let call = span_scope("evaluate_call", 7, 10);
        record_decline("no model answers this call", None, Some("no reading"));
        drop(call);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read recorded a span");
    assert!(document.chain.is_empty(), "a call leaf names its own construct and reclaims nothing");
    assert!(!render_json(&document).contains("\"chain\""), "`chain` is optional and absent here");
}

#[test]
fn an_answered_binding_is_filed_but_a_determined_read_reclaims_nothing() {
    // The ledger files every write, but only a read whose derivation
    // STOPPED at a bare name reclaims one.
    let document = a_ledger_walk(false).expect("the read recorded a span");
    assert_eq!(document.chain.len(), 1, "the reclaim followed the read's own bare-name leaf");
    assert!(
        projection_of_deepest_decline(&document.chain[0]).is_none(),
        "the reclaimed binding answered, so it projects no undetermined sentence"
    );
}

#[test]
fn the_ledger_does_not_exist_while_tracing_is_off() {
    // "Off = the ledger does not exist." Both ledger entry points read
    // one Cell and return having recorded nothing at all.
    assert!(!is_tracing());
    let scope = ledger_scope(vec!["x".to_owned()], "check::walk_assign", 0, 5);
    record_read_place("x");
    drop(scope);
    assert!(!is_tracing());
    // And a collector that saw no ledger write reclaims nothing.
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(TWO_STATEMENTS, 2))));
    {
        let _guard = install(collector.clone());
        let judged = position_scope("check::walk_return", 8, 16);
        let name = span_scope("name_read", 15, 16);
        record_read_place("x");
        record_decline("'x' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read recorded a span");
    assert!(document.chain.is_empty(), "no write was filed, so nothing is reclaimed");
}

#[test]
fn the_emitted_chain_is_a_sibling_of_root_the_schema_names() {
    let document = a_ledger_walk(true).expect("the read recorded a span");
    let json = render_json(&document);
    assert!(json.contains("\"chain\": ["), "the chain is a top-level array beside `root`:\n{json}");
    // Ordinals run s1, s2, … across the whole document: the main root
    // first, then the chained roots.
    assert!(json.contains("\"id\": \"s1\""));
    assert_eq!(document.root.id, "s1");
    assert_eq!(document.chain[0].id, "s3", "the chain continues the document's ordinals");
}

#[test]
fn a_chained_target_files_the_one_derivation_under_every_place() {
    // `a = b = f()` — one right-hand side derived what both names hold,
    // so a read blocked at either reclaims the same subtree.
    let source = "a = b = f()\nreturn b\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 2))));
    {
        let _guard = install(collector.clone());
        {
            let _binding =
                ledger_scope(vec!["a".to_owned(), "b".to_owned()], "check::walk_assign", 8, 11);
            let call = span_scope("evaluate_call", 8, 11);
            record_decline("no model answers this call", None, Some("no reading"));
            drop(call);
            record_decline("this assignment's right-hand side derives no value", None, Some("no reading"));
        }
        let judged = position_scope("check::walk_return", 12, 20);
        let name = span_scope("name_read", 19, 20);
        record_read_place("b");
        record_decline("'b' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read recorded a span");
    assert_eq!(document.chain.len(), 1);
    assert_eq!(document.chain[0].range, "fixture.py:1:9-1:12");
}

#[test]
fn the_nearest_binding_before_the_read_is_the_one_reclaimed() {
    // Two functions each bind `x`; the read sits between them, so the
    // EARLIER binding is the one it saw. A single-slot ledger would hand
    // back the later one, which the read never reached.
    // line 1: x = f()     bytes 0..7,   `f()` 4..7
    // line 2: return x    bytes 8..16,  `x`   15..16
    // line 3: x = g()     bytes 17..24, `g()` 21..24
    let source = "x = f()\nreturn x\nx = g()\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 2))));
    {
        let _guard = install(collector.clone());
        for (start, end, gate) in [(4usize, 7usize, "no model answers f"), (21, 24, "no model answers g")] {
            let _binding = ledger_scope(vec!["x".to_owned()], "check::walk_assign", start, end);
            let call = span_scope("evaluate_call", start, end);
            record_decline(gate, None, Some("no reading"));
            drop(call);
            record_decline("this assignment's right-hand side derives no value", None, Some("no reading"));
        }
        let judged = position_scope("check::walk_return", 8, 16);
        let name = span_scope("name_read", 15, 16);
        record_read_place("x");
        record_decline("'x' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read recorded a span");
    assert_eq!(document.chain.len(), 1);
    assert_eq!(document.chain[0].range, "fixture.py:1:5-1:8", "the binding BEFORE the read");
}

#[test]
fn a_read_of_a_havocked_binding_carries_the_last_touch_attribute() {
    // line 1: s.add(x)     bytes 0..8,  the whole call `s.add(x)`
    // line 2: return s     bytes 9..17, `s` 16..17
    //
    // The mutation walk (`check::calls::mutation::walk_mutating_call_
    // statement`) replays no successor for `s.add(x)` (an unmodeled
    // method), so it forgets `s` NAMING the call as the cause — exactly
    // `Environment::forget_with_cause`'s own effect, reproduced here at
    // the trace layer since `Environment` is a different crate module
    // this file does not otherwise depend on.
    let source = "s.add(x)\nreturn s\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 2))));
    {
        let _guard = install(collector.clone());
        record_havoc_touch("s", (0, 8));
        let judged = position_scope("check::walk_return", 9, 17);
        let name = span_scope("name_read", 16, 17);
        record_read_place("s");
        record_decline("'s' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read recorded a span");
    let leaf = &document.root.children[0];
    assert_eq!(leaf.name, "name_read");
    assert_eq!(
        leaf.last_touch.as_deref(),
        Some("havocked by s.add(x)  @fixture.py:1:1-1:9"),
        "the leaf names the call that erased the binding, not just that something did"
    );
}

#[test]
fn an_untouched_parameters_read_carries_no_last_touch_attribute() {
    // A read of a name the ledger holds no record for at all — a
    // parameter, or any name this walk never bound or forgot — leaves
    // `last_touch` absent rather than stamping a record for a place
    // nothing ever touched.
    let source = "return value\n";
    let collector = Arc::new(Mutex::new(TraceCollector::new(a_request(source, 1))));
    {
        let _guard = install(collector.clone());
        let judged = position_scope("check::walk_return", 0, 12);
        let name = span_scope("name_read", 7, 12);
        record_read_place("value");
        record_decline("'value' carries no derived value at this read", None, Some("no reading"));
        drop(name);
        drop(judged);
    }
    let document = take_trace(collector).expect("the read recorded a span");
    let leaf = &document.root.children[0];
    assert_eq!(leaf.name, "name_read");
    assert_eq!(leaf.last_touch, None, "nothing touched this place, so nothing is stamped");
}
