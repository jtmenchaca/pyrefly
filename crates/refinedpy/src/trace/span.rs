//! The span struct and the document that wraps it — the minimal
//! OpenTelemetry-shaped subset `trace.schema.json` defines, and nothing
//! else. No SDK, no exporter, no resource/scope envelope, no events, no
//! links: a span is a small struct this adapter hand-rolls an emitter
//! for (`emit.rs`).

/// A span's own two-value status — the only two states the spec admits.
/// Maps to OTLP OK/ERROR in a converter, which this tree does not carry.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpanStatus {
    Answered,
    Declined,
}

impl SpanStatus {
    /// The schema's own spelling of this status.
    pub fn word(self) -> &'static str {
        match self {
            SpanStatus::Answered => "answered",
            SpanStatus::Declined => "declined",
        }
    }
}

/// One step of the derivation: the reader that owned it, whether it
/// answered or declined, and the `refinery.*` attributes the spec's
/// vocabulary defines for that status. Children nest in evaluation
/// order — the artifact keeps the nesting because it is what makes a
/// conformance diff readable across three adapters.
#[derive(Clone, Debug)]
pub struct Span {
    /// Ordinal within one trace: `s1`, `s2`, … assigned by the collector
    /// in creation order. An OTLP converter would assign hex ids.
    pub id: String,
    /// The adapter-local reader id that owned this step, or
    /// `kernel.<op>` for an ask.
    pub name: String,
    pub status: SpanStatus,
    /// Present only under timing; this adapter does not populate it yet
    /// (the spec has timing riding these same spans under the existing
    /// timing flags, which is a later reconciliation).
    pub duration_ns: Option<u64>,

    // ---- the attribute vocabulary, one field per `refinery.*` key ----
    /// Root span only: `ts` | `py` | `cpp`. Always `py` here.
    pub language: Option<&'static str>,
    /// Root span only: the judged position this trace explains.
    pub position: Option<String>,
    /// Every span: the sub-expression's OWN source spelling, never the
    /// whole statement's.
    pub construct: String,
    /// Every span: `path:line:col-line:col` of that sub-expression.
    pub range: String,
    /// Answered spans and kernel asks: the derived set or window, in the
    /// kernel's own diagnostic spelling.
    pub answer: Option<String>,
    /// Declined spans: the named premise that failed. Its ABSENCE on a
    /// declined span is a visible work item, not an accepted state — the
    /// projection falls back to `<construct>: <reader> declined`.
    pub gate: Option<String>,
    /// Declined spans: the failing operand's range.
    pub operand: Option<String>,
    /// Declined spans: what that operand held, in the kernel's spelling.
    pub held: Option<String>,
    /// Kernel-ask spans: the wire question text.
    pub question: Option<String>,
    /// Declined bare-name leaves: where the place this leaf read was last
    /// touched — `<kind> by <construct>  @<range>`, the LAST-TOUCH
    /// LEDGER's own spelling (`collector::LastTouch::words`). Filled in
    /// at document-assembly time (`TraceCollector::into_document`), from
    /// the ledger keyed by the leaf's own `place`, never at span-open
    /// time — the ledger is not complete until the whole walk has run.
    pub last_touch: Option<String>,

    /// THE ROOT RULE (DERIVATION-TRACE.md, "The attribute vocabulary"):
    /// "The root is `answered` ... No decline is ever recorded onto the
    /// root — a decline with no open reader span is refused, not attached
    /// upward." A span opened by `position_scope` carries this, and
    /// `record_decline` refuses to write onto it. NOT part of the
    /// attribute vocabulary — never emitted, never in the schema.
    pub is_position: bool,

    /// THE BINDING LEDGER's key on this span, and NOT part of the
    /// attribute vocabulary — never emitted, never in the schema.
    ///
    /// On a `name_read` span it is the place that read: a bare-name leaf
    /// carrying this is what the ledger reclaims a binding's derivation
    /// for. Spelled the way `env::TrackedPlace` spells a place, so the
    /// reader and the writer agree by construction (`a`, `a.n`).
    pub place: Option<String>,
    /// THE BINDING LEDGER's key on a binding's own span, and NOT part of
    /// the attribute vocabulary — never emitted, never in the schema.
    /// Every place this statement writes: one for `a = v`, several for a
    /// chained `a = b = v`, all filed with the same derivation because
    /// one right-hand side derived what all of them hold.
    pub written_places: Vec<String>,
    /// This span's own start byte offset in the traced file, and NOT part
    /// of the attribute vocabulary — never emitted, never in the schema.
    /// The binding ledger orders writes and reads by it, so a reclaim
    /// picks the binding that precedes the read rather than whichever
    /// same-named binding the walk finished last.
    pub start: usize,

    pub children: Vec<Span>,
}

impl Span {
    /// A span with only the two attributes every span carries. The
    /// collector fills in the status-specific attributes when the step
    /// returns.
    ///
    /// The status starts ANSWERED: a decline is a claim a reader makes by
    /// calling `record_decline`, never a default a step falls into by
    /// recording nothing. A dispatcher that opens a span and records no
    /// outcome has not refused anything, and a tree whose root carries a
    /// default decline would make `deepest_decline` land on a step that
    /// never named a gate.
    pub fn new(id: String, name: String, construct: String, range: String) -> Span {
        Span {
            id,
            name,
            status: SpanStatus::Answered,
            duration_ns: None,
            language: None,
            position: None,
            construct,
            range,
            answer: None,
            gate: None,
            operand: None,
            held: None,
            question: None,
            last_touch: None,
            place: None,
            written_places: Vec::new(),
            start: 0,
            is_position: false,
            children: Vec::new(),
        }
    }
}

/// One whole trace: the three top-level members the schema requires.
#[derive(Clone, Debug)]
pub struct TraceDocument {
    /// Always `"py"` in this adapter.
    pub language: &'static str,
    /// The judged position this trace explains: `path:line[:col]`.
    pub position: String,
    pub root: Span,
    /// The BINDING LEDGER's reclaimed roots: the derivations of the
    /// binding statements behind this trace's bare-name leaves, nearest
    /// binding first. Empty where the main root's own leaf is not a bare
    /// name, and the schema's `chain` is then not emitted at all — the
    /// member is optional.
    ///
    /// Chained as ADDITIONAL roots, never merged into `root`: the
    /// projection still reads the main root's deepest declined span, and
    /// the chain sharpens the work item rather than the sentence.
    pub chain: Vec<Span>,
}
