//! The derivation trace — the Python adapter's implementation of
//! `packages/tests/DERIVATION-TRACE.md`, whose schema is
//! `packages/tests/diagnostics/trace.schema.json`. That document is the
//! reference; this file is one of three implementations of it.
//!
//! Every judged position gets its verdict from a derivation this checker
//! already walks and then discards: dispatcher → reader → sub-reads →
//! kernel asks. Tracing records that walk, so a position can answer where
//! the derivation stopped, what each operand held, who was asked, and
//! which named premise refused — with no probe and no instrumentation
//! edit.
//!
//! ## The root rule
//!
//! DERIVATION-TRACE.md pins the root: "The root is `answered`; its
//! `refinery.construct` is the requested line's own text and its
//! `refinery.range` the full-line range. No decline is ever recorded onto
//! the root — a decline with no open reader span is refused, not attached
//! upward."
//!
//! Three pieces of this module hold that up together, so it cannot drift:
//! `position_scope` marks the judged position's span and `record_decline`
//! refuses to write onto a marked one; `Span::new` starts a span
//! `answered`, so a step that records no outcome never masquerades as a
//! refusal; and `projection::deepest_decline` searches a root's children
//! and never the root itself. A position's undetermined-ness is therefore
//! carried entirely by the declined READER spans beneath it — the ones
//! that name a gate, an operand, and what it held — which is exactly what
//! the printed sentence needs to point at the construct that blocked.
//!
//! ## Where the collector lives
//!
//! `Environment` carries `Option<Arc<Mutex<TraceCollector>>>`, `None` by
//! default, exactly the shape `returned_values` and `evaluations` already
//! take (`env/recording.rs`) and for the identical reason: a fork walking
//! one branch arm, and a nested body's own fresh `Environment`, must both
//! write into the ONE collector the caller reads back. `fork` clones the
//! `Arc` rather than the contents, so the recording is fork-blind.
//!
//! Two seams the spec names as instrumentation points hold no
//! `Environment` at all — `assignability::judge` (a value, a declared
//! refinement, a kernel) and `kernel_ask::ask_kernel` (a closure). Rather
//! than thread an `Environment` parameter through ~100 `judge` call sites
//! and every kernel ask, the environment PUBLISHES its own `Arc` into a
//! thread-local slot for the duration of the walk (`install`, and the
//! `Drop` guard that clears it). That is one channel and one collector,
//! never a second: the thread-local holds a clone of the very same `Arc`
//! the environment holds, so a span recorded through either route lands
//! in the same tree. The walk is single-threaded per file, the same
//! premise `kernel_ask`'s own panic-suppression thread-local already
//! rests on.
//!
//! ## The named-blocker reconciliation
//!
//! This crate already had a naming channel for the generic decline:
//! `check::name_unmodeled_call_sentence` rewrites `SENTENCE.
//! value_not_readable` into a narrower sentence for three recognized
//! shapes (`raise_and_blocker::unmodeled_module_call`,
//! `generator_body_never_summarized`, and the two manifest sentences).
//!
//! The decline helper SUBSUMES that channel rather than sitting beside
//! it as a third one, and the two compose without either being rewritten,
//! because of how that step is gated: it acts ONLY on a sentence equal to
//! the exact generic wording, and passes every other sentence through
//! unchanged. So —
//!
//! - untraced, `judge` answers the generic wording and the naming step
//!   sharpens it for its three shapes exactly as it always has;
//! - traced, `judge` answers the PROJECTION (never the generic wording),
//!   the naming step sees a sentence that already names its construct and
//!   leaves it alone, and the projected sentence is the one printed.
//!
//! The projection is the strictly more specific answer in every case the
//! naming step covers — it names the reader, the construct, and what the
//! operand held, where the naming step names only the module or the
//! generator — so nothing is lost by the trace winning. The naming step's
//! own three sentences remain the untraced answer and are not duplicated
//! here.
//!
//! ## Off is free
//!
//! Every recording entry point begins with a thread-local `Cell<bool>`
//! read that is `false` for an ordinary check; nothing is allocated, no
//! source is sliced, no range is formatted. `span_scope` returns a
//! do-nothing guard. The gate wall is untouched.

mod collector;
mod emit;
mod projection;
mod span;

#[cfg(test)]
mod tests;

pub use collector::install;
pub use collector::is_tracing;
pub use collector::ledger_scope;
pub use collector::record_bind_touch;
pub use collector::record_forget_touch;
pub use collector::record_havoc_touch;
pub use collector::record_read_place;
pub use collector::projected_sentence_of_innermost_decline;
pub use collector::record_answer;
pub use collector::record_decline;
pub use collector::position_scope;
pub use collector::record_kernel_ask;
pub use collector::span_scope;
pub use collector::take_trace;
pub use collector::SpanScope;
pub use collector::TraceCollector;
pub use collector::TraceRequest;
pub use emit::render_json;
pub use projection::deepest_decline_including_top;
pub use projection::project_sentence;
pub use projection::projection_of_chained_root;
pub use projection::projection_of_deepest_decline;
pub use span::Span;
pub use span::SpanStatus;
pub use span::TraceDocument;
