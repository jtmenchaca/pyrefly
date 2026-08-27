//! The projection rule: the printed UNDETERMINED sentence IS the
//! projection of the trace's deepest declined span, by this template and
//! no other (DERIVATION-TRACE.md, "The projection rule"):
//!
//! ```text
//! <construct>: <gate> — <operand construct> held <held>
//! ```
//!
//! Adapters do not hand-write decline prose. The standing rule — every
//! undetermined names the first construct that blocked it — is enforced
//! by construction, because the sentence and the trace are one carrier.
//!
//! SCOPE: this projects RTS7002 (undetermined) sentences only. RTS7001
//! error sentences are marker-matched by the corpus and are not this
//! module's business.
//!
//! A declined span with NO gate — a dispatcher-instrumented step that has
//! not yet adopted the decline helper — still projects, by the spec's own
//! stated fallback `<construct>: <reader> declined`. Localization is
//! never lost; the missing gate is the visible work item.

use super::span::Span;
use super::span::SpanStatus;

/// The deepest declined span in this tree — where the derivation actually
/// stopped. Two rules, in this order:
///
/// 1. DEPTH. A declined child is deeper than its declined parent, and it
///    is the child that names the construct that blocked; the parent
///    declined only BECAUSE the child did. A kernel-ask child counts — a
///    refused ask is the innermost thing that refused. Between two
///    declining SUBTREES under one parent, the one whose own decline sits
///    deeper wins: a bare operand read that declined at depth 1 is a
///    smaller answer than a call whose model declined at depth 2 beneath
///    it, and the deeper one is the one that names a construct to go fix.
/// 2. EVALUATION ORDER breaks a tie at equal depth: the FIRST decline
///    wins. Equally deep sibling declines at one position are one
///    dispatcher declining after the reader beneath it already had, so
///    the earlier span stopped the walk and the later one is its
///    consequence. This is the standing rule stated as a tree walk —
///    every undetermined names the FIRST construct that blocked it.
///
/// The ROOT is never the answer. DERIVATION-TRACE.md's root rule makes
/// the root `answered` and keeps declines on reader spans, so a root that
/// is the deepest decline would mean some reader refused without opening
/// a span — and projecting it would point the sentence at the whole
/// judged statement rather than at the construct that blocked. The walk
/// therefore searches the root's CHILDREN, and a tree whose only decline
/// is its own root projects nothing.
///
/// `None` when nothing beneath the root declined at all.
pub fn deepest_decline(span: &Span) -> Option<&Span> {
    deepest_among_children(span, 0).map(|(found, _)| found)
}

/// The best candidate among a span's CHILDREN, by the two rules above.
/// `depth` is the parent's own depth, so each child is compared at
/// `depth + 1`. `>` (never `>=`) is what keeps rule 2's earliest-wins
/// tie-break.
fn deepest_among_children(span: &Span, depth: usize) -> Option<(&Span, usize)> {
    let mut best: Option<(&Span, usize)> = None;
    for child in &span.children {
        if let Some((found, found_depth)) = deepest_decline_at(child, depth + 1) {
            if best.is_none_or(|(_, best_depth)| found_depth > best_depth) {
                best = Some((found, found_depth));
            }
        }
    }
    best
}

/// `deepest_decline`'s recursion, carrying each candidate's own depth so
/// rule 1 can compare two subtrees rather than only a parent against its
/// own children.
fn deepest_decline_at(span: &Span, depth: usize) -> Option<(&Span, usize)> {
    if let Some(best) = deepest_among_children(span, depth) {
        return Some(best);
    }
    if span.status == SpanStatus::Declined {
        return Some((span, depth));
    }
    None
}

/// The deepest declined span in a subtree WHOSE OWN TOP is a candidate —
/// `deepest_decline`'s rule with the root exclusion lifted.
///
/// The exclusion exists for the DOCUMENT's root, which the spec pins to
/// `answered`. A CHAINED root is a binding statement's own derivation,
/// reclaimed into `chain` to sharpen the work item; it is not the judged
/// position, and its own decline ("this assignment's right-hand side
/// derives no value") is a real reader refusal that names a real gate. So
/// the chain reads with this entry point and the main root with the
/// other, and one rule still decides depth and ties for both.
pub fn deepest_decline_including_top(span: &Span) -> Option<&Span> {
    deepest_decline_at(span, 0).map(|(found, _)| found)
}

/// `deepest_decline_including_top`'s sentence — what a chained root
/// projects to.
pub fn projection_of_chained_root(root: &Span) -> Option<String> {
    deepest_decline_including_top(root).map(project_sentence)
}

/// The sentence one declined span projects to, by the spec's template.
/// The trailing clause is present only where the span carries an operand
/// or a held value — a gate with neither states the premise alone.
pub fn project_sentence(span: &Span) -> String {
    let Some(gate) = &span.gate else {
        // The spec's stated fallback for a span whose refusal site has
        // not yet adopted the decline helper.
        return format!("{}: {} declined", span.construct, span.name);
    };
    let head = format!("{}: {gate}", span.construct);
    match (&span.operand, &span.held) {
        (Some(operand), Some(held)) => format!("{head} — {operand} held {held}"),
        (Some(operand), None) => format!("{head} — {operand}"),
        (None, Some(held)) => format!("{head} — held {held}"),
        (None, None) => head,
    }
}

/// The projected sentence for a whole trace: the deepest declined span's
/// projection. `None` where nothing declined — a determined position has
/// no undetermined sentence to print.
pub fn projection_of_deepest_decline(root: &Span) -> Option<String> {
    deepest_decline(root).map(project_sentence)
}
