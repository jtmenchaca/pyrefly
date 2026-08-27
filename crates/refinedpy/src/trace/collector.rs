//! The collector: a span stack the dispatch seams push onto and record
//! into, plus the thread-local channel that lets the two seams holding no
//! `Environment` (`assignability::judge`, `kernel_ask::ask_kernel`) reach
//! the very same collector — see this module's own doc for why that is
//! one channel rather than two.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use super::span::Span;
use super::span::SpanStatus;
use super::span::TraceDocument;

/// What the caller asked to be traced: the file whose ranges are being
/// recorded, its source (for slicing a construct's own spelling), its
/// line starts (for formatting a range), and the 1-based line the caller
/// wants explained.
///
/// Recording is per POSITION, exactly as the spec's Gating section
/// states: the walk runs normally and a span is recorded only where the
/// current range intersects the requested line.
pub struct TraceRequest {
    pub path: String,
    pub source: String,
    pub line_starts: Vec<usize>,
    pub line: usize,
    /// `path` spelled relative to the repository root, computed once when
    /// the request is built. `None` where `path` has no repository
    /// ancestor, and `emitted_path` then answers `path` itself.
    emitted_path: Option<String>,
}

impl TraceRequest {
    /// A request for `line` of `path`, whose repository-relative spelling
    /// is resolved here so every formatted range shares one answer.
    pub fn new(path: String, source: String, line_starts: Vec<usize>, line: usize) -> TraceRequest {
        let emitted_path = TraceRequest::repository_relative(&path);
        TraceRequest { path, source, line_starts, line, emitted_path }
    }

    /// `path:line:col-line:col` for a byte range — the schema's own
    /// `refinery.range` spelling. Offsets past the source end clamp to
    /// the end, so a synthetic range never panics the formatter.
    pub fn range_words(&self, start: usize, end: usize) -> String {
        let (start_line, start_col) = self.line_col(start);
        let (end_line, end_col) = self.line_col(end);
        format!("{}:{start_line}:{start_col}-{end_line}:{end_col}", self.emitted_path())
    }

    /// The path as the EMITTED DOCUMENT spells it: relative to the
    /// repository root, whatever form the caller handed in
    /// (DERIVATION-TRACE.md's Gating rule — "Paths in `refinery.position`
    /// and `refinery.range` are spelled relative to the repository root in
    /// the emitted document, whatever the adapter's internal form —
    /// conformance diffs across languages depend on one spelling").
    ///
    /// The repository root is the nearest ancestor of the traced file
    /// holding a `.git` entry. A path with no such ancestor — a fixture
    /// with a bare relative name, a file outside any checkout — is
    /// already as short as it can be spelled and passes through.
    pub fn emitted_path(&self) -> &str {
        self.emitted_path.as_deref().unwrap_or(&self.path)
    }

    /// `emitted_path`'s one computation, run once when the request is
    /// built rather than per formatted range: the traced file's path made
    /// relative to the repository root that contains it.
    fn repository_relative(path: &str) -> Option<String> {
        let absolute = std::fs::canonicalize(path).ok()?;
        let mut directory = absolute.parent()?;
        loop {
            if directory.join(".git").exists() {
                let relative = absolute.strip_prefix(directory).ok()?;
                return Some(relative.to_string_lossy().into_owned());
            }
            directory = directory.parent()?;
        }
    }

    /// The FULL-LINE range of the requested line, as byte offsets — the
    /// root span's own range, per the spec's root rule ("its
    /// `refinery.construct` is the requested line's own text and its
    /// `refinery.range` the full-line range"). The end excludes the
    /// newline, so a root's construct is the line's text and not the line
    /// plus a blank.
    /// Lines are 1-based; a request naming line 0 is out of range and
    /// answers the first line, the same clamp `line_col` already applies.
    pub fn requested_line_range(&self) -> (usize, usize) {
        let index = self.line.saturating_sub(1);
        let start = self.line_starts.get(index).copied().unwrap_or(0).min(self.source.len());
        let end = match self.line_starts.get(index + 1) {
            Some(next) => next.saturating_sub(1),
            None => self.source.len(),
        };
        (start, end.min(self.source.len()).max(start))
    }

    /// The source text between two byte offsets — a sub-expression's own
    /// spelling. Newlines inside a multi-line construct collapse to
    /// single spaces so the attribute stays one line, and a clamped or
    /// non-char-boundary range answers the empty string rather than
    /// panicking.
    pub fn construct_words(&self, start: usize, end: usize) -> String {
        let end = end.min(self.source.len());
        let start = start.min(end);
        if !self.source.is_char_boundary(start) || !self.source.is_char_boundary(end) {
            return String::new();
        }
        self.source[start..end].split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// Whether a byte range touches the requested line — the recording
    /// filter the spec's Gating section defines. A range spanning several
    /// lines (a whole `if` body) intersects when the requested line falls
    /// anywhere inside it.
    pub fn intersects(&self, start: usize, end: usize) -> bool {
        let (start_line, _) = self.line_col(start);
        let (end_line, _) = self.line_col(end);
        start_line <= self.line && self.line <= end_line
    }

    fn line_col(&self, offset: usize) -> (usize, usize) {
        let offset = offset.min(self.source.len());
        let line = self.line_starts.partition_point(|start| *start <= offset).max(1);
        let col = offset - self.line_starts[line - 1] + 1;
        (line, col)
    }
}

/// One entry of the LAST-TOUCH LEDGER: what happened to a binding the
/// last time it was written, forgotten, or forgotten because a call
/// mutated it, and — for the two forget cases — the construct that did
/// it and where. `construct`/`range` are `None` for a plain forget with
/// no known cause (`Environment::forget`'s ordinary call sites): the
/// spec's own words, "a forget with no cause records kind alone."
#[derive(Clone, Debug)]
pub struct LastTouch {
    kind: &'static str,
    construct: Option<String>,
    range: Option<String>,
}

impl LastTouch {
    pub fn written() -> LastTouch {
        LastTouch { kind: "written", construct: None, range: None }
    }

    pub fn forgotten() -> LastTouch {
        LastTouch { kind: "forgotten", construct: None, range: None }
    }

    pub fn havocked(construct: String, range: String) -> LastTouch {
        LastTouch { kind: "havocked", construct: Some(construct), range: Some(range) }
    }

    /// This record's own `refinery.last-touch` spelling: `<kind> by
    /// <construct>  @<range>`, the construct/range clause omitted where
    /// the record carries none.
    fn words(&self) -> String {
        match (&self.construct, &self.range) {
            (Some(construct), Some(range)) => format!("{} by {construct}  @{range}", self.kind),
            _ => self.kind.to_owned(),
        }
    }
}

/// The span tree under construction: finished root spans, the stack of
/// spans currently open, the ordinal counter ids are minted from, and the
/// BINDING LEDGER.
///
/// ## The binding ledger
///
/// DERIVATION-TRACE.md: "at every environment write, when tracing, the
/// ledger records the span that produced the written value, keyed by the
/// place written." `ledger` is that map, and it is what lets ONE explain
/// run answer a read whose derivation stops at a bare name — the second
/// run at the binding line is no longer needed.
///
/// A binding statement almost never sits on the requested line, so the
/// position filter that governs `span_scope` would record nothing there.
/// `ledger_scope` therefore opens its span UNCONDITIONALLY and raises
/// `ledger_depth`, and every `span_scope` beneath it records
/// unconditionally too — that whole subtree is the binding's derivation.
/// When the ledger root closes it is FILED into `ledger` rather than
/// pushed onto `finished`, so it never becomes a root of the main trace
/// and the projection is untouched.
pub struct TraceCollector {
    request: TraceRequest,
    open: Vec<Span>,
    finished: Vec<Span>,
    next_id: u32,
    /// Every completed derivation subtree filed against a written place,
    /// keyed the way `env::TrackedPlace` spells a place, in walk order
    /// and paired with the byte offset the write sits at.
    ///
    /// Several writes to ONE place is the ordinary case — two functions
    /// in a file each bind `offset_minutes` — so this holds them all and
    /// the reclaim picks the NEAREST binding: the last write that starts
    /// at or before the requested position. A single-slot map would hand
    /// back whichever function the walk happened to finish last, which is
    /// the wrong derivation for the read being explained.
    ledger: HashMap<String, Vec<(usize, Span)>>,
    /// How many ledger scopes are open. Non-zero means the walk is inside
    /// a binding's right-hand side, so the position filter is suspended.
    ledger_depth: usize,
    /// THE LAST-TOUCH LEDGER: the most recent write/forget recorded
    /// against each binding NAME, in walk order (a later touch simply
    /// overwrites an earlier one — a document only ever asks "what
    /// happened to this name most recently"). Consulted at document
    /// assembly (`into_document`) to stamp `refinery.last-touch` onto
    /// every declined bare-name leaf, keyed by the same `place` the
    /// BINDING LEDGER already reads off that leaf.
    last_touch: HashMap<String, LastTouch>,
}

impl TraceCollector {
    pub fn new(request: TraceRequest) -> TraceCollector {
        TraceCollector {
            request,
            open: Vec::new(),
            finished: Vec::new(),
            next_id: 0,
            ledger: HashMap::new(),
            ledger_depth: 0,
            last_touch: HashMap::new(),
        }
    }

    fn mint_id(&mut self) -> String {
        self.next_id += 1;
        format!("s{}", self.next_id)
    }

    /// Closes the innermost open span, attaching it to its parent — or,
    /// with no parent, to the finished list.
    fn close_top(&mut self) {
        let Some(span) = self.open.pop() else {
            return;
        };
        match self.open.last_mut() {
            Some(parent) => parent.children.push(span),
            None => self.finished.push(span),
        }
    }

    /// Closes an open LEDGER span: the subtree it accumulated is filed
    /// under the place it wrote rather than becoming a root of the main
    /// trace. A ledger span nested inside another open span (a binding
    /// inside a body whose own span is open) is filed AND attached, so a
    /// derivation the requested position happens to contain is not lost.
    fn close_ledger_top(&mut self, start: usize, on_requested_line: bool) {
        self.ledger_depth = self.ledger_depth.saturating_sub(1);
        let Some(span) = self.open.pop() else {
            return;
        };
        for place in span.written_places.clone() {
            self.ledger.entry(place).or_default().push((start, span.clone()));
        }
        // Attached to the main tree only where the binding's own range
        // touches the requested position — an off-position binding
        // recorded purely for the ledger stays out of it, so the main
        // trace is byte-identical to what it was without a ledger.
        if !on_requested_line {
            return;
        }
        match self.open.last_mut() {
            Some(parent) => parent.children.push(span),
            None => self.finished.push(span),
        }
    }

    /// The sentence the CURRENTLY-OPEN judged position projects to.
    ///
    /// The open stack is nested by construction (each open span is the
    /// parent of the next), so this materializes it — innermost first,
    /// each folded into its parent's children — and then projects the
    /// resulting subtree by the same rule
    /// `projection::projection_of_deepest_decline` applies to a finished
    /// tree. That keeps ONE projection rule: the sentence a walk prints
    /// mid-derivation and the sentence a reader derives from the emitted
    /// JSON afterward are computed by the same function.
    ///
    /// The fold is wrapped in a CARRIER span that is never itself a
    /// candidate, because `projection::deepest_decline` searches a root's
    /// children and never the root. Wrapping makes every actually-open
    /// span a child of something, so the outermost open span is eligible
    /// whether or not a position span happens to be open above it — the
    /// mid-derivation sentence is the same one the finished tree projects.
    fn project_open_position(&self) -> Option<String> {
        let mut folded: Option<Span> = None;
        for span in self.open.iter().rev() {
            let mut copy = span.clone();
            if let Some(child) = folded.take() {
                copy.children.push(child);
            }
            folded = Some(copy);
        }
        let mut carrier = Span::new(String::new(), String::new(), String::new(), String::new());
        carrier.children.push(folded?);
        super::projection::projection_of_deepest_decline(&carrier)
    }

    /// Every finished top-level span, wrapped as one schema-valid
    /// document. With several top-level spans (one judged position can be
    /// reached by more than one dispatch), the first one carrying a
    /// decline ANYWHERE in its own tree — its top included — is the root:
    /// that is the tree holding the blocker the caller asked about; with
    /// none declining anywhere, the first one is.
    ///
    /// The top is included in this SELECTION scan (unlike the projection
    /// scan below, which still excludes it) because a blocker recorded by
    /// a dispatcher with no reader of its own beneath it — `record_blocker`'s
    /// "blocked_construct" span, opened and declined in one place with no
    /// child to carry the decline instead — is a top-level FINISHED span
    /// that is itself the decline. Scanning children only would never find
    /// it, and the position span some OTHER, unrelated top-level span
    /// happens to have opened on the same requested line would be chosen
    /// instead, publishing an answered root with no trace of the blocker
    /// at all.
    ///
    /// The root's own status is never consulted for the FINAL published
    /// tree, because the root is always `answered` (the spec's root rule,
    /// applied unconditionally below): what distinguishes a blocked
    /// position from a determined one is a declined READER span beneath
    /// it. Widening the SELECTION rule to include the top cannot change
    /// that: every position the old children-only rule already found a
    /// decline for is still found at the same or an earlier candidate,
    /// since a top-level decline is a strictly larger match set.
    fn into_document(mut self) -> Option<TraceDocument> {
        while !self.open.is_empty() {
            self.close_top();
        }
        let chosen = self
            .finished
            .iter()
            .position(|span| super::projection::deepest_decline_including_top(span).is_some())
            .unwrap_or(0);
        if self.finished.is_empty() {
            return None;
        }
        let mut root = self.finished.swap_remove(chosen);
        let position = format!("{}:{}", self.request.emitted_path(), self.request.line);
        // A span chosen for its OWN decline (`deepest_decline` on it finds
        // nothing among its children, but its own top is declined — the
        // `record_blocker` shape: one dispatcher-owned span that opens and
        // declines with no reader beneath it) is folded in as a CHILD of a
        // synthetic root instead of being stripped bare in place: the root
        // rule below always answers root and clears its own decline
        // attributes, and applying that TO the declining span itself would
        // erase the only evidence the projection has to walk to. Wrapping
        // keeps the same guarantee — the published root is answered — while
        // leaving the decline on a child where `deepest_decline` (the
        // projection's own, children-only, root-excluding rule) still finds
        // it, exactly as it would for any other reader span.
        if super::projection::deepest_decline(&root).is_none()
            && super::projection::deepest_decline_including_top(&root).is_some()
        {
            let wrapper = Span::new(String::new(), String::new(), String::new(), String::new());
            root = wrapper_with_child(wrapper, root);
        }
        // THE ROOT RULE (DERIVATION-TRACE.md): the root is `answered`, its
        // construct is the requested line's own text and its range the
        // full-line range. Applied here, on the one span the document
        // actually publishes as root, so no dispatcher has to know whether
        // its own span will turn out to be one.
        let (line_start, line_end) = self.request.requested_line_range();
        root.status = SpanStatus::Answered;
        root.gate = None;
        root.operand = None;
        root.held = None;
        root.construct = self.request.construct_words(line_start, line_end);
        root.range = self.request.range_words(line_start, line_end);
        root.language = Some("py");
        root.position = Some(position.clone());
        let mut chain = self.reclaim_chain(&root);
        // THE LAST-TOUCH LEDGER's own stamp: run once the whole walk has
        // filed every write/forget it ever will, so a leaf's stamp never
        // depends on how much of the file happened to be walked yet.
        self.stamp_last_touch(&mut root);
        for span in &mut chain {
            self.stamp_last_touch(span);
        }
        // IDS ARE ORDINALS WITHIN ONE TRACE (the schema's own wording), so
        // they are assigned here rather than at open time: the ledger
        // records spans the emitted document never carries (every binding
        // in the file, not just the reclaimed ones), and minting at open
        // time would leave the main root's ordinals depending on how many
        // of those the walk happened to pass. Renumbered main root first,
        // then each chained root in order, the document reads s1, s2, …
        // top to bottom and the main root is what it was before a ledger
        // existed.
        let mut next = 0u32;
        renumber(&mut root, &mut next);
        for span in &mut chain {
            renumber(span, &mut next);
        }
        Some(TraceDocument { language: "py", position, root, chain })
    }

    /// THE RECLAIM (DERIVATION-TRACE.md, the binding ledger): "A read
    /// whose derivation stops at a bare name reclaims that span's subtree
    /// into `chain`; a chained root whose own leaf is again a bare name
    /// reclaims recursively, nearest binding first, cycles refused by
    /// place."
    ///
    /// The leaf consulted is the deepest declined span — the same one the
    /// projection reads — so the chain answers exactly the question the
    /// printed sentence leaves open: which construct left this name
    /// carrying nothing. A leaf that is not a bare name (a call with no
    /// model, an unread operator) already names its own construct and
    /// reclaims nothing.
    ///
    /// Cycles are refused BY PLACE: `a = b` then `b = a` would otherwise
    /// walk forever, so a place already reclaimed is never reclaimed a
    /// second time.
    ///
    /// The MAIN root is read with the root-excluding rule (it is the
    /// judged position, which the spec pins to `answered`); a CHAINED root
    /// is a binding statement's own derivation and is read with its own
    /// top included, so a binding whose only declining span is its
    /// statement-level one still states where it stopped.
    fn reclaim_chain(&self, root: &Span) -> Vec<Span> {
        let mut chain: Vec<Span> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut current = root.clone();
        let mut from_main_root = true;
        loop {
            let found = match from_main_root {
                true => super::projection::deepest_decline(&current),
                false => super::projection::deepest_decline_including_top(&current),
            };
            from_main_root = false;
            let Some(leaf) = found else {
                return chain;
            };
            // A bare-name leaf is exactly a span the reader tagged with
            // the place it read; every other form carries no place.
            let Some(place) = leaf.place.clone() else {
                return chain;
            };
            if !seen.insert(place.clone()) {
                return chain;
            }
            let Some(binding) = self.nearest_binding(&place, leaf.start) else {
                return chain;
            };
            chain.push(binding.clone());
            current = binding;
        }
    }

    /// Records `record` as `place`'s most recent touch — the LAST-TOUCH
    /// LEDGER's one write chokepoint, called from `Environment::bind`/
    /// `forget`/`forget_with_cause` wherever tracing is active.
    fn record_last_touch(&mut self, place: &str, record: LastTouch) {
        self.last_touch.insert(place.to_owned(), record);
    }

    /// Walks `span`'s whole subtree stamping `refinery.last-touch` onto
    /// every DECLINED leaf that carries a `place` — a bare-name read
    /// whose derivation stopped there, exactly the leaf the BINDING
    /// LEDGER already reclaims by the same key. A span the ledger holds
    /// no record for (a parameter, a name never written before this
    /// read) is left untouched: an untouched place has nothing to name.
    fn stamp_last_touch(&self, span: &mut Span) {
        if span.status == SpanStatus::Declined {
            if let Some(place) = &span.place {
                if let Some(record) = self.last_touch.get(place) {
                    span.last_touch = Some(record.words());
                }
            }
        }
        for child in &mut span.children {
            self.stamp_last_touch(child);
        }
    }

    /// The NEAREST binding of `place` for a read at byte offset `read`:
    /// the last write to that place starting at or before the read. A
    /// file where two functions each bind `offset_minutes` files two
    /// derivations under the one place, and this is what picks the one
    /// the read being explained actually saw.
    ///
    /// `None` where the place has no write before the read — a parameter,
    /// or a name bound only later in the file.
    fn nearest_binding(&self, place: &str, read: usize) -> Option<Span> {
        self.ledger
            .get(place)?
            .iter()
            .filter(|(start, _)| *start <= read)
            .max_by_key(|(start, _)| *start)
            .map(|(_, span)| span.clone())
    }
}

/// Folds `child` under `wrapper` as its one child, returning the wrapper.
/// Used once, in `into_document`, to give a self-declining chosen span a
/// parent to nest under before the root rule strips the published root's
/// own decline attributes — the wrapper absorbs that clearing and the
/// original span's decline survives as a child the projection still finds.
fn wrapper_with_child(mut wrapper: Span, child: Span) -> Span {
    wrapper.children.push(child);
    wrapper
}

/// Assigns `s1`, `s2`, … over one span tree in document order, advancing
/// the shared counter so several roots of one document never collide.
fn renumber(span: &mut Span, next: &mut u32) {
    *next += 1;
    span.id = format!("s{next}");
    for child in &mut span.children {
        renumber(child, next);
    }
}

thread_local! {
    /// The one collector this thread's walk records into. A clone of the
    /// very same `Arc` the `Environment` holds — never a second channel.
    static ACTIVE: RefCell<Option<Arc<Mutex<TraceCollector>>>> = const { RefCell::new(None) };
    /// The off-is-free gate: every recording entry point reads this
    /// `Cell` first, and an ordinary check finds `false` and returns
    /// having allocated nothing.
    static TRACING: Cell<bool> = const { Cell::new(false) };
}

/// Whether this thread is recording. The one read every entry point
/// makes before doing any work at all.
pub fn is_tracing() -> bool {
    TRACING.with(|flag| flag.get())
}

/// Publishes `collector` as this thread's active one for the lifetime of
/// the returned guard. The guard's `Drop` clears the slot, so a panic
/// unwinding through the walk still leaves the thread clean — the same
/// discipline `kernel_ask`'s own `SuppressGuard` keeps.
pub fn install(collector: Arc<Mutex<TraceCollector>>) -> InstallGuard {
    ACTIVE.with(|slot| *slot.borrow_mut() = Some(collector));
    TRACING.with(|flag| flag.set(true));
    InstallGuard
}

pub struct InstallGuard;

impl Drop for InstallGuard {
    fn drop(&mut self) {
        TRACING.with(|flag| flag.set(false));
        ACTIVE.with(|slot| *slot.borrow_mut() = None);
    }
}

/// Runs `f` against the active collector, if one is installed and this
/// thread is recording. The `try_lock` is deliberate: a re-entrant
/// recording (a span opened while another is being closed) skips rather
/// than deadlocking a walk that is only being instrumented.
fn with_collector<F, T>(f: F) -> Option<T>
where
    F: FnOnce(&mut TraceCollector) -> T,
{
    if !is_tracing() {
        return None;
    }
    ACTIVE.with(|slot| {
        let borrowed = slot.borrow();
        let collector = borrowed.as_ref()?;
        let mut guard = collector.try_lock().ok()?;
        Some(f(&mut guard))
    })
}

/// Reads the finished document out of `collector`, consuming it. `None`
/// when the walk recorded no span at all for the requested position —
/// which is itself the answer "nothing judged on that line."
pub fn take_trace(collector: Arc<Mutex<TraceCollector>>) -> Option<TraceDocument> {
    let collector = Arc::try_unwrap(collector).ok()?;
    let collector = collector.into_inner().ok()?;
    collector.into_document()
}

/// An open span, closed when this guard drops. A dispatcher holds one
/// across its own body; the reader beneath it never manages spans, per
/// the spec's "Threading: dispatchers, not readers".
///
/// A scope that recorded NOTHING (tracing off, or the range outside the
/// requested position) carries `recorded: false` and its `Drop` does
/// nothing at all — the zero-cost path.
pub struct SpanScope {
    recorded: bool,
    /// A LEDGER scope: its subtree is filed under the place it wrote
    /// (`TraceCollector::close_ledger_top`) rather than becoming a root
    /// of the main trace.
    ledger: bool,
    /// Whether this scope's own range touched the requested line. Only a
    /// ledger scope consults it, to decide whether its subtree also
    /// belongs in the main tree.
    on_requested_line: bool,
    /// The scope's own start byte offset — how the ledger orders one
    /// place's several writes against the read that reclaims one.
    start: usize,
}

impl Drop for SpanScope {
    fn drop(&mut self) {
        if !self.recorded {
            return;
        }
        let ledger = self.ledger;
        let on_requested_line = self.on_requested_line;
        let start = self.start;
        with_collector(|collector| match ledger {
            true => collector.close_ledger_top(start, on_requested_line),
            false => collector.close_top(),
        });
    }
}

/// Opens a span for one dispatch step: `name` is the adapter-local
/// reader id, and the byte range is the sub-expression the step owns —
/// its own spelling and range become `refinery.construct` and
/// `refinery.range`.
///
/// Off, or a range that does not touch the requested position, returns a
/// do-nothing guard having read one `Cell<bool>` and allocated nothing.
pub fn span_scope(name: &str, start: usize, end: usize) -> SpanScope {
    if !is_tracing() {
        return SpanScope { recorded: false, ledger: false, on_requested_line: false, start };
    }
    let recorded = with_collector(|collector| {
        // Inside an open ledger scope the position filter is suspended:
        // the whole subtree beneath a binding IS that binding's
        // derivation, and it is recorded wherever the binding sits.
        if collector.ledger_depth == 0 && !collector.request.intersects(start, end) {
            return false;
        }
        let id = collector.mint_id();
        let construct = collector.request.construct_words(start, end);
        let range = collector.request.range_words(start, end);
        let mut span = Span::new(id, name.to_owned(), construct, range);
        span.start = start;
        collector.open.push(span);
        true
    })
    .unwrap_or(false);
    SpanScope { recorded, ledger: false, on_requested_line: false, start }
}

/// THE JUDGED POSITION's own span — the one that becomes the document's
/// root. Identical to `span_scope` except that it is marked, and a
/// marked span REFUSES a decline: DERIVATION-TRACE.md's root rule states
/// "The root is `answered` ... No decline is ever recorded onto the root
/// — a decline with no open reader span is refused, not attached
/// upward."
///
/// A position's undetermined-ness is carried entirely by the declined
/// READER spans beneath it, which name the gate, the operand, and what
/// it held. The root names only WHICH position was judged, so a
/// projection walking for the deepest decline can never land on it and
/// mispoint the sentence at the whole statement.
pub fn position_scope(name: &str, start: usize, end: usize) -> SpanScope {
    let scope = span_scope(name, start, end);
    if scope.recorded {
        with_collector(|collector| {
            if let Some(span) = collector.open.last_mut() {
                span.is_position = true;
            }
        });
    }
    scope
}

/// THE BINDING LEDGER's write seam: one span for a binding statement's
/// right-hand side, keyed by the place it writes, recorded WHEREVER the
/// binding sits rather than only on the requested line. The subtree this
/// scope accumulates is the derivation of the written value, and it is
/// filed into the ledger when the scope drops.
///
/// `places` are spelled the way `env::TrackedPlace` spells a place, so a
/// bare-name read's own tag (`record_read_place`) and these keys agree by
/// construction. A chained `a = b = value` names several, all filed with
/// the one derivation that produced what all of them hold.
///
/// Off, this is one `Cell<bool>` read and a do-nothing guard — the ledger
/// does not exist while no collector is installed.
pub fn ledger_scope(places: Vec<String>, name: &str, start: usize, end: usize) -> SpanScope {
    if !is_tracing() {
        return SpanScope { recorded: false, ledger: false, on_requested_line: false, start };
    }
    let on_requested_line = with_collector(|collector| collector.request.intersects(start, end)).unwrap_or(false);
    let recorded = with_collector(|collector| {
        let id = collector.mint_id();
        let construct = collector.request.construct_words(start, end);
        let range = collector.request.range_words(start, end);
        let mut span = Span::new(id, name.to_owned(), construct, range);
        span.written_places = places;
        span.start = start;
        collector.open.push(span);
        collector.ledger_depth += 1;
        true
    })
    .unwrap_or(false);
    SpanScope { recorded, ledger: true, on_requested_line, start }
}

/// Tags the innermost open span with the PLACE it read — the ledger's
/// lookup key on a bare-name leaf. A read whose derivation stops here is
/// what reclaims that place's binding derivation into `chain`.
pub fn record_read_place(place: &str) {
    with_collector(|collector| {
        if let Some(span) = collector.open.last_mut() {
            span.place = Some(place.to_owned());
        }
    });
}

/// Records that the innermost open span ANSWERED, carrying the derived
/// set or window in the kernel's own diagnostic spelling.
///
/// The decline attributes are cleared: DERIVATION-TRACE.md's span rule,
/// "A span that declines on one arm and then answers CLEARS its
/// gate/operand/held — decline attributes never survive on an answered
/// span."
pub fn record_answer(answer: &str) {
    with_collector(|collector| {
        let Some(span) = collector.open.last_mut() else {
            return;
        };
        span.status = SpanStatus::Answered;
        span.answer = Some(answer.to_owned());
        span.gate = None;
        span.operand = None;
        span.held = None;
    });
}

/// The decline helper the spec names: records the named gate that
/// failed, the failing operand's range, and what that operand held, onto
/// the innermost open span. The projected sentence is then read back off
/// the same span (`projection::projection_of_deepest_decline`) — the
/// sentence and the trace are ONE carrier and cannot drift.
///
/// `operand` is a byte range in the traced file; passing the span's own
/// range is correct where the failing operand IS the whole construct.
///
/// A decline landing on a POSITION span is REFUSED — the spec's root
/// rule: "No decline is ever recorded onto the root — a decline with no
/// open reader span is refused, not attached upward." A refusal that
/// reaches here with only the position span open is a reader that failed
/// to open its own span, and attaching it upward would make the
/// projection point at the whole statement instead of the construct that
/// blocked.
pub fn record_decline(gate: &str, operand: Option<(usize, usize)>, held: Option<&str>) {
    with_collector(|collector| {
        let operand_words = operand.map(|(start, end)| collector.request.range_words(start, end));
        let Some(span) = collector.open.last_mut() else {
            return;
        };
        if span.is_position {
            return;
        }
        span.status = SpanStatus::Declined;
        span.gate = Some(gate.to_owned());
        if let Some(words) = operand_words {
            span.operand = Some(words);
        }
        if let Some(held) = held {
            span.held = Some(held.to_owned());
        }
    });
}

/// The sentence THIS POSITION's derivation projects to, by the spec's
/// template: the deepest declined span in the currently-open span's own
/// subtree, projected.
///
/// Read back at the moment the judging seam lands on a decline, so the
/// printed sentence and the recorded tree are one carrier and cannot
/// state different things. The subtree searched is the OUTERMOST open
/// span's — the judged position's own span — so a decline recorded on a
/// dispatcher still projects the READER's own construct beneath it, which
/// is the one that blocked.
///
/// `None` when nothing is open (tracing off, or the range outside the
/// requested position), and the caller keeps its own generic sentence.
pub fn projected_sentence_of_innermost_decline() -> Option<String> {
    with_collector(|collector| {
        // The outermost open span is the judged position's own; its
        // already-closed children carry the readers that ran. A decline
        // just recorded on an inner open span is visible too, since
        // `project_position` walks the open stack as well as the closed
        // children.
        collector.project_open_position()
    })
    .flatten()
}

/// THE LAST-TOUCH LEDGER's write seam for an ordinary bind:
/// `Environment::bind` calls this, guarded by its own `is_tracing` check
/// the same way every other recording call site in this tree guards
/// itself, so an ordinary check pays for one `Cell<bool>` read and
/// nothing more.
pub fn record_bind_touch(place: &str) {
    with_collector(|collector| collector.record_last_touch(place, LastTouch::written()));
}

/// THE LAST-TOUCH LEDGER's write seam for a plain forget — no known
/// cause, the spec's "a forget with no cause records kind alone."
/// `Environment::forget`'s ordinary call sites use this.
pub fn record_forget_touch(place: &str) {
    with_collector(|collector| collector.record_last_touch(place, LastTouch::forgotten()));
}

/// THE LAST-TOUCH LEDGER's write seam for a forget WITH a cause: an
/// unmodeled call replayed no successor for the receiver, so the walk
/// forgets it rather than carrying a stale fact — `havocked by
/// <construct> @<range>`. `construct_range` is the CAUSING construct's
/// own byte range (the call expression itself, `s.add(x)`), formatted
/// here against the active request the same way `record_decline` formats
/// an operand — so the caller passes offsets, never a pre-spelled string,
/// and the ledger's spelling always matches what a reader span for the
/// identical range would show. `Environment::forget_with_cause`'s one
/// caller today is `walk_mutating_call_statement`'s own `None` arm.
pub fn record_havoc_touch(place: &str, construct_range: (usize, usize)) {
    with_collector(|collector| {
        let construct = collector.request.construct_words(construct_range.0, construct_range.1);
        let range = collector.request.range_words(construct_range.0, construct_range.1);
        collector.record_last_touch(place, LastTouch::havocked(construct, range));
    });
}

/// Records one kernel ask as a CHILD of the reader that asked — the
/// spec's one-chokepoint rule: `kernel_ask::ask_kernel` calls this, so
/// question 3 is never per-reader work. The child inherits the asking
/// span's own construct and range, since an ask has no source spelling
/// of its own.
pub fn record_kernel_ask(op: &str, question: &str, answer: Option<&str>) {
    with_collector(|collector| {
        let Some(parent) = collector.open.last() else {
            return;
        };
        let construct = parent.construct.clone();
        let range = parent.range.clone();
        let id = collector.mint_id();
        let mut child = Span::new(id, format!("kernel.{op}"), construct, range);
        child.question = Some(question.to_owned());
        match answer {
            Some(answer) => {
                child.status = SpanStatus::Answered;
                child.answer = Some(answer.to_owned());
            }
            None => {
                child.status = SpanStatus::Declined;
                child.gate = Some("the kernel refused this question's set shape".to_owned());
            }
        }
        if let Some(parent) = collector.open.last_mut() {
            parent.children.push(child);
        }
    });
}
