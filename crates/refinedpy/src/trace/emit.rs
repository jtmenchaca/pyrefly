//! The hand-rolled emitter: a `TraceDocument` rendered as the JSON
//! `trace.schema.json` defines. Hand-rolled per the spec's "each adapter
//! hand-rolls its emitter against `diagnostics/trace.schema.json`" — no
//! SDK, no exporter, and no serde derive on the span types, so the
//! attribute vocabulary is written out key by key in one readable place
//! and `additionalProperties: false` is satisfied by construction (a key
//! is emitted only where its value is present).

use super::span::Span;
use super::span::TraceDocument;

/// One JSON string literal, escaped per RFC 8259 — the control
/// characters the spec's construct/answer text can carry (a tab in a
/// source slice, a quote in a set spelling) plus the two mandatory
/// escapes.
fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The `attributes` object for one span: every `refinery.*` key whose
/// value is present, in the vocabulary's own order. `refinery.construct`
/// and `refinery.range` are always written — the schema requires both on
/// every span, answered or declined.
fn attributes_json(span: &Span, indent: &str) -> String {
    let mut pairs: Vec<String> = Vec::new();
    if let Some(language) = span.language {
        pairs.push(format!("{indent}  \"refinery.language\": {}", quoted(language)));
    }
    if let Some(position) = &span.position {
        pairs.push(format!("{indent}  \"refinery.position\": {}", quoted(position)));
    }
    pairs.push(format!("{indent}  \"refinery.construct\": {}", quoted(&span.construct)));
    pairs.push(format!("{indent}  \"refinery.range\": {}", quoted(&span.range)));
    if let Some(answer) = &span.answer {
        pairs.push(format!("{indent}  \"refinery.answer\": {}", quoted(answer)));
    }
    if let Some(gate) = &span.gate {
        pairs.push(format!("{indent}  \"refinery.gate\": {}", quoted(gate)));
    }
    if let Some(operand) = &span.operand {
        pairs.push(format!("{indent}  \"refinery.operand\": {}", quoted(operand)));
    }
    if let Some(held) = &span.held {
        pairs.push(format!("{indent}  \"refinery.held\": {}", quoted(held)));
    }
    if let Some(question) = &span.question {
        pairs.push(format!("{indent}  \"refinery.question\": {}", quoted(question)));
    }
    if let Some(last_touch) = &span.last_touch {
        pairs.push(format!("{indent}  \"refinery.last-touch\": {}", quoted(last_touch)));
    }
    format!("{{\n{}\n{indent}}}", pairs.join(",\n"))
}

fn span_json(span: &Span, indent: &str) -> String {
    let inner = format!("{indent}  ");
    let mut fields = vec![
        format!("{inner}\"id\": {}", quoted(&span.id)),
        format!("{inner}\"name\": {}", quoted(&span.name)),
        format!("{inner}\"status\": {}", quoted(span.status.word())),
    ];
    if let Some(duration) = span.duration_ns {
        fields.push(format!("{inner}\"durationNs\": {duration}"));
    }
    fields.push(format!("{inner}\"attributes\": {}", attributes_json(span, &inner)));
    if !span.children.is_empty() {
        let children: Vec<String> = span
            .children
            .iter()
            .map(|child| format!("{inner}  {}", span_json(child, &format!("{inner}  "))))
            .collect();
        fields.push(format!("{inner}\"children\": [\n{}\n{inner}]", children.join(",\n")));
    }
    format!("{{\n{}\n{indent}}}", fields.join(",\n"))
}

/// The whole document as schema-valid JSON, pretty-printed. `chain` is
/// OPTIONAL in the schema and is written only where the binding ledger
/// reclaimed something — a trace whose leaf is not a bare name emits the
/// three required members and nothing more.
pub fn render_json(document: &TraceDocument) -> String {
    let mut members = vec![
        format!("  \"language\": {}", quoted(document.language)),
        format!("  \"position\": {}", quoted(&document.position)),
        format!("  \"root\": {}", span_json(&document.root, "  ")),
    ];
    if !document.chain.is_empty() {
        let roots: Vec<String> = document
            .chain
            .iter()
            .map(|span| format!("    {}", span_json(span, "    ")))
            .collect();
        members.push(format!("  \"chain\": [\n{}\n  ]", roots.join(",\n")));
    }
    format!("{{\n{}\n}}", members.join(",\n"))
}
