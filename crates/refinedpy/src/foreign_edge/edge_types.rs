//! The recognized edge's own data: which carrier a call's payload rides
//! (`Channel`), how the return leg's sole consumer reads the captured
//! text back (`ResultRead`), the recognized edge itself (`ForeignEdge`),
//! and the outcome `foreign_edge_at`/`foreign_edge_at_walrus_call`
//! answer (`ForeignEdgeOutcome`).

use ruff_python_ast::Expr;
use ruff_text_size::TextRange;

use refined_domain::abstract_value::AbstractValue;

use super::argv::Runner;

/// How the return leg's sole consumer reads the captured text back off
/// the bound name — the two shapes this crate's recognized calls
/// produce. `subprocess.run` binds a result object and the captured
/// text sits at its `.stdout` attribute; `subprocess.check_output`
/// returns the captured text directly, and `subprocess.Popen`'s
/// `.communicate()` tuple-unpacks it into a plain name — both of those
/// are read the same bare way once `result_name` names the right
/// variable.
#[derive(Clone, PartialEq, Eq)]
pub(super) enum ResultRead {
    /// `json.loads(<name>.stdout)` — `subprocess.run`'s own shape.
    StdoutAttribute,
    /// `json.loads(<name>)` — `subprocess.check_output`'s direct return,
    /// and `subprocess.Popen`'s tuple-unpacked stdout name.
    Bare,
    /// `os.system`'s own file-legs shape: the return leg has no BOUND
    /// NAME to scan for at all (`os.system`'s captured target is the
    /// process's exit status, never the crossing's own value) — the
    /// consumer instead sits at a LATER `with open("<outfile>") as
    /// <handle>: ... json.load(<handle>)`, found by the literal outfile
    /// name this variant carries rather than by any name the call
    /// itself bound. `finish_recognized_edge` dispatches on this
    /// variant to `os_system_return_read_of` instead of `sole_parse_
    /// consumer_of`'s bound-name scan, which has nothing to scan for
    /// here.
    FileRead { outfile: String },
}


/// Which carrier a recognized call sends its payload on — the SAME two
/// tags `foreign_edge_artifact.rs`'s `ForeignSurface` names, read here
/// off the call's own spelling rather than off a target's stated fact.
/// `foreign_edge_at` compares this against the artifact's declared
/// surface: a mismatch either way (stdin payload at an argv-json
/// target, or the reverse) declines naming the channel that does not
/// meet, before any outbound-leg fit question is even asked.
///
/// PREMISE: an argv element rides the OS argv byte array rather than a
/// pipe, but the checker's identity claim about the crossing value is
/// the SAME one `stdin-json` already cites — `surface.kind` names which
/// carrier the bytes ride on, not a different transport model. Both
/// carriers move the identical JSON text; the round-trip premise
/// (`json.dumps` on this side, `JSON.parse` on the target's) is shared,
/// which is why `check_outbound_leg`'s own fit checks apply unchanged
/// to either channel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Channel {
    /// `input=json.dumps(<payload>)` — the payload rides the process's
    /// stdin pipe.
    Stdin,
    /// The payload rides one argv element — node's own convention
    /// makes the third argv entry `process.argv[2]`, so a three-element
    /// argv list (`["node", script, json.dumps(payload)]`) reads as
    /// `arg_index == 2`.
    Argv { arg_index: i64 },
    /// The payload is written to a named temp file, and the argv
    /// element carries the file's PATH (a bare name, not
    /// `json.dumps(...)`) rather than the JSON text itself — the target
    /// reads the file at that path. Same `arg_index` convention as
    /// `Argv`.
    File { arg_index: i64 },
}


/// One recognized cross-language call: which node the call is, which
/// `.ts` file it names, which expression crosses out, and which name
/// catches the target's stdout.
pub struct ForeignEdge {
    /// The call: `subprocess.run(...)` — where a fit refutation points.
    pub call: TextRange,
    /// The `.ts` file, resolved against the checked file's own
    /// directory (a relative argv entry is relative to the file that
    /// wrote it, which is the only reading that survives a moved cwd).
    pub target_path: String,
    /// The expression handed to `json.dumps` — the value that actually
    /// crosses out. The SAME JSON-encoded value either channel carries;
    /// only the carrier named by `channel` differs.
    pub payload: Expr,
    /// Which carrier this call's own spelling used — checked against
    /// the artifact's declared surface before the outbound-leg fit is
    /// asked.
    pub(super) channel: Channel,
    /// The name the call's result binds, whose sole consumer (read
    /// per `result_read`) receives the return fact.
    pub result_name: String,
    /// How the sole consumer reads `result_name` back.
    pub(super) result_read: ResultRead,
    /// Where the return-leg's sole-consumer scan starts looking (the
    /// statement AFTER this index): the call's own position for
    /// `subprocess.run`/`subprocess.check_output`, and one further for
    /// `subprocess.Popen` — the `.communicate()` statement its own
    /// recognition already consumed is not itself a consumer to find
    /// again.
    pub(super) consumer_scan_from: usize,
    /// Which runner word this call spelled — carried through for the
    /// decline sentences that name it (an unfit-input decline, an
    /// unrecognized script extension); every recognized runner
    /// discharges the runtime-identity premise identically once the
    /// artifact's own band check (`foreign_edge_artifact.rs`) passes.
    pub(super) runner: Runner,
}


/// What the route decided at one statement. Exactly one of `override_value`
/// and `decline` is meaningful: a green crossing publishes the parse
/// node's fact, and everything else says one sentence naming the
/// premise that stopped it.
///
/// A REFUTED crossing (the outbound value escapes the target's entry)
/// reports RTS7001 and publishes nothing — the call is wrong, so there
/// is no fact to carry back. That case answers `Fired`, distinct from
/// both `Override` and `Decline`.
pub enum ForeignEdgeOutcome {
    /// Every premise came back green: the range of the `json.loads(...)`
    /// node to publish the value under, and the value itself.
    ///
    /// `stdout_override`, when `Some`, is a SECOND, independent binding:
    /// the intermediate captured-stdout reading's own node (a `result
    /// .stdout` attribute access for `ResultRead::StdoutAttribute`, or
    /// the bound name's own node for `ResultRead::Bare`) paired with the
    /// SERIALIZED form of the return fact — the string-sorted JSON-
    /// number-grammar set the harness's own encoder can spell for a
    /// return whose every case is number-sorted
    /// (`foreign_stdout_serialized_value`'s own doc). `None` for every
    /// return shape that derivation does not cover (any non-number
    /// case), and unconditionally `None` for `ResultRead::FileRead`
    /// (`os.system` has no intermediate stdout binding to serialize at
    /// all — the captured target there is the process's exit status).
    Override { parse_range: TextRange, value: AbstractValue, stdout_override: Option<(TextRange, AbstractValue)> },
    /// The crossing escapes what the target states it admits: an
    /// RTS7001 the caller reports at `range` (the payload), never a
    /// decline — the call is wrong, so there is no fact to attach.
    ///
    /// An outbound fire and a bound return fact are INDEPENDENT truths
    /// (the TypeScript checker's own ruled convention: `d-data-legs.ts`
    /// measures exactly one fire with the return fact still bound) —
    /// `consumer`, when `Some`, carries the return leg's own consumer
    /// node AND its bound value, built the SAME way `return_leg_outcome`
    /// builds `Override`'s own value (`foreign_return_value_or_
    /// undetermined`, never re-derived), so the caller publishes that
    /// REAL fact at the consumer node instead of leaving it opaque — the
    /// consumer then judges the return against that determined value
    /// (a refusal there is its own genuine fire, not this one repeated).
    ///
    /// `Some` ONLY for `ResultRead::FileRead` (`os.system`'s own
    /// `json.load(<handle>)` read of a plain `open()` result, a shape
    /// `expressions.rs` never models at all: an ordinary walk of that
    /// node produces a bare opaque value with nothing else to say, so
    /// binding the real fact there adds a determination, never removes
    /// one). For every OTHER shape (`ResultRead::StdoutAttribute`/`Bare`
    /// — an ordinary `json.loads(...)` call), `consumer` is
    /// UNCONDITIONALLY `None`: an ordinary walk of that SAME node, left
    /// unbound, reaches `expressions.rs`'s own `json.loads`-of-an-
    /// untracked-operand model (`json_loads_value_space`, the full
    /// None|bool|str|int|float|list|dict union), whose `None` arm the
    /// return's declared type then genuinely refuses — a second,
    /// DETERMINED RTS7001 this field must never replace with a narrower
    /// bound value (`b-runners.py:159`, `c-reference-shapes.py:104`/
    /// `:146`, `d-data-legs.py:184`'s own designed fires, each: "the
    /// outbound refutation is trailed by the return judge's own union-
    /// `None`-arm fire, since the return leg is never served once the
    /// call itself is refused" — a narrower bound value there would
    /// replace that determined union fire with a judge against a set
    /// the corpus never designed this row to carry, possibly answering
    /// determined-silent where the design calls for a second fire).
    Fired { message: String, range: TextRange, consumer: Option<(TextRange, AbstractValue)> },
    /// The sentence naming the premise that stopped the edge (an
    /// RTS7002 the caller records as this body's blocker), and where it
    /// points.
    Decline { message: String, range: TextRange },
}
