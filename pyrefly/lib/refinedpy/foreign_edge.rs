/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The cross-language call edge, recognized in the walk — the REVERSE
//! pair of refined-ts-go's `walk/foreign_edge.go` (TS calling Python):
//! here Python calls out to a TypeScript body over stdin/stdout JSON,
//! reads back the target's own kernel-derived return fact, and attaches
//! it to the `json.loads(...)` node that reads the captured stdout.
//!
//! Recognized shape (`docs/one-checker/reverse-pair.md`, Half B):
//!
//! ```python
//! result = subprocess.run(
//!     ["node", "./audio_level.ts"],
//!     input=json.dumps(boosted),
//!     capture_output=True,
//!     text=True,
//! )
//! return json.loads(result.stdout)
//! ```
//!
//! Two other `subprocess` callees recognize the same argv/payload shape:
//! `subprocess.check_output(...)` (the captured text is the CALL's own
//! return, read bare — `json.loads(result)`, never `result.stdout`) and
//! the two-statement `subprocess.Popen(...)` / `<stdout>, _ = <name>
//! .communicate(json.dumps(...))` pair, where `.communicate()`'s own
//! call carries the payload the `Popen(...)` call itself does not.
//!
//! The runner word at argv[0] (plus, for a two-word runner, argv[1])
//! also recognizes beyond plain `"node"`: `"deno" "run"`, `"bun"`, and
//! `"npx" "tsx"` all name a real script the same way `"node"` does. The
//! band this checker's TypeScript pins commit to (`es2023+`) names an
//! ECMA-262 spec level, not one runtime binary (ruling, 2026-08-21), so
//! every recognized runner discharges the runtime-identity premise
//! identically once the artifact declares that band — the artifact
//! reader's own band check (`foreign_edge_artifact.rs`) is the only gate,
//! and it applies the same way regardless of which runner the call names.
//!
//! argv[1] (the script) also resolves through a module-level constant
//! this body reads (`TARGET_PATH = "./x.ts"` used as `["node",
//! TARGET_PATH]`) — any other non-literal shape (an f-string, a
//! parameter) declines with the law-2 sentence naming the fixable
//! written-literal respelling.
//!
//! A SIBLING carrier: `subprocess.run(["node", "<script>.ts",
//! json.dumps(<payload>)], capture_output=True, text=True)` — the
//! payload rides the third argv element (`process.argv[2]`, node's own
//! convention) rather than stdin, and carries no `input=` keyword at
//! all (its presence alongside an argv payload is a real double-channel
//! ambiguity, declined rather than silently picking one). The target's
//! own artifact must declare a matching `surface.kind == "argv-json"`
//! (with the same `argIndex`) for this shape to apply; an argv payload
//! against a `stdin-json` target (or the reverse) declines naming the
//! channel mismatch — recognized shapes on both sides, transports that
//! do not meet.
//!
//! A THIRD carrier — a named TEMP FILE — sends the payload through
//! neither a pipe nor an argv element's own text: `with tempfile
//! .NamedTemporaryFile(mode="w", suffix=".json", delete=False) as
//! handle: json.dump(<payload>, handle); temp_path = handle.name`
//! immediately followed by `subprocess.run(["node", "<script>.ts",
//! temp_path], capture_output=True, text=True)`. The argv element
//! carries the file's PATH (a bare name, never `json.dumps(...)`), and
//! the target reads its JSON from that file (node's own
//! `readFileSync(process.argv[2], "utf8")`). This is a THREE-STATEMENT
//! unit (`recognize_temp_file_edge`): the `with`-block itself supplies
//! the payload and the bound path name, and the call one statement
//! later must name that SAME bound name at argv[2] — a reassignment of
//! `temp_path` between the dump and the call leaves the checker unable
//! to prove the file the call reads is the file `json.dump` wrote, so
//! it stays undetermined naming the rebind. The target's own artifact
//! must declare a matching `surface.kind == "file-json"` (with the same
//! `argIndex`) for this shape to apply; a temp-file payload against a
//! `stdin-json` or `argv-json` target (or the reverse) declines naming
//! the channel mismatch, the same way the argv-json sibling does.
//!
//! CROSS-LANGUAGE-EDGE.md §2's corollary makes this a real edge and not
//! a manifest: the argv deterministically NAMES the code that runs
//! next, so the checker treats the call the way it treats an import.
//! §11 is this exact spelling; §4 is the JSON transport model both legs
//! apply; §5 is the list of premises the crossing rests on.
//!
//! WHAT THE ROUTE DOES, in order (mirrors the Go twin's own banner):
//!
//!  1. RECOGNIZE the call: an `Assign` of one name from a recognized
//!     `subprocess` callee, with a written argv list naming a runner and
//!     a script, and every required keyword. Anything unrecognized
//!     declines, and every decline NAMES what broke.
//!  2. READ the target's exported fact off disk through the sibling's
//!     `read_foreign_ts_artifact` — target integrity, runtime identity,
//!     and harness shape are the artifact reader's own premises.
//!  3. DISCHARGE the outbound leg's premises against the value actually
//!     being stringified: NaN-freedom (NaN stringifies to `null`, so the
//!     target never sees the number the caller sent) and the crossing
//!     fit (the argument's element set inside the entry's, its length
//!     floor at or above the entry's). A fit FAILURE is not a decline:
//!     it is a 7001 at the call, because the value can escape what the
//!     target states it admits.
//!  4. DISCHARGE channel purity and ATTACH the return fact to the
//!     `json.loads(result.stdout)` node — the sole consumer of the
//!     captured stdout, found the same way the Go twin's
//!     `soleParseConsumerOf` finds its `JSON.parse` node.
//!
//! The attach rides `Environment::set_evaluated_node`, the seam the
//! relational-sum lane already uses for a value no re-walk can reach —
//! `check.rs`'s own return-position quotient publish (check.rs:1975-1978)
//! is the exact precedent this route follows.
//!
//! TRUST GRADE. The attached fact is stamped `TrustSpec`, not
//! `TrustProved` — the mirror of the Go twin's own reasoning
//! (`foreignReturnValue`'s doc): every premise here is a real check, but
//! the crossing itself rests on cited spec behaviour (the JSON number
//! round-trip) this tree has not proved as a kernel theorem.
//!
//! `json.loads` always answers a Python `float` for a JSON number
//! whose text carries a fractional/exponent part (library/json.rst's
//! conversion table — "number (int)" only when the JSON text itself
//! has no such part AND the loader's own `parse_int` is not
//! overridden, which this checker does not read). The CHECKER's own
//! sort tag on the crossed value does not stamp Float uniformly over
//! this ambiguity; it reads the target's declared return set for its
//! own `Integer` form the same way a declared position's sort is read
//! (`foreign_return_value`'s doc) — an all-integer return reads
//! Integer, and only an unmarked or genuinely fractional return reads
//! Float.
//!
//! CORNER CHECK: the return set's own corner values must be ones the
//! TypeScript target's own serializer actually carries. The mechanism is
//! NOT that legal JSON text (RFC 8259) has no token for ±Infinity — a
//! JSON leg can carry it fine (`1e999` is legal JSON text and parses to
//! Infinity in both runtimes). The mechanism is `JSON.stringify` itself:
//! it serializes a non-finite Number as the bare literal `null`
//! (ECMA-262's `SerializeJSONProperty`, the finiteness check on a Number
//! value), a value outside the claimed numeric set landing at this leg's
//! own `json.loads` consumer. A return set whose corners admit ±Infinity
//! degrades to a named undetermined instead of binding the set as stated
//! (`foreign_return_value_or_undetermined`); NaN is already excluded
//! from every `RefinedSet` at construction (the boundary ruling), so
//! only the two infinite corners need the check.

use std::sync::Arc;

use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::known_object;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::ConversionFlag;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprList;
use ruff_python_ast::ExprName;
use ruff_python_ast::InterpolatedStringElement;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtWith;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::refinedpy::diagnostic_sentences;
use crate::refinedpy::env::Environment;
use crate::refinedpy::foreign_edge_artifact::ForeignCase;
use crate::refinedpy::foreign_edge_artifact::ForeignSurface;
use crate::refinedpy::foreign_edge_artifact::ForeignTsArtifact;
use crate::refinedpy::foreign_edge_artifact::ForeignTsEntry;
#[cfg(test)]
use crate::refinedpy::foreign_edge_artifact::ForeignTsFunctionFact;
#[cfg(not(test))]
use crate::refinedpy::foreign_edge_artifact::read_foreign_ts_artifact as read_foreign_ts_artifact_landed;

/// How the return leg's sole consumer reads the captured text back off
/// the bound name — the two shapes this crate's recognized calls
/// produce. `subprocess.run` binds a result object and the captured
/// text sits at its `.stdout` attribute; `subprocess.check_output`
/// returns the captured text directly, and `subprocess.Popen`'s
/// `.communicate()` tuple-unpacks it into a plain name — both of those
/// are read the same bare way once `result_name` names the right
/// variable.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ResultRead {
    /// `json.loads(<name>.stdout)` — `subprocess.run`'s own shape.
    StdoutAttribute,
    /// `json.loads(<name>)` — `subprocess.check_output`'s direct return,
    /// and `subprocess.Popen`'s tuple-unpacked stdout name.
    Bare,
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
enum Channel {
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
    channel: Channel,
    /// The name the call's result binds, whose sole consumer (read
    /// per `result_read`) receives the return fact.
    pub result_name: String,
    /// How the sole consumer reads `result_name` back.
    result_read: ResultRead,
    /// Where the return-leg's sole-consumer scan starts looking (the
    /// statement AFTER this index): the call's own position for
    /// `subprocess.run`/`subprocess.check_output`, and one further for
    /// `subprocess.Popen` — the `.communicate()` statement its own
    /// recognition already consumed is not itself a consumer to find
    /// again.
    consumer_scan_from: usize,
    /// Which runner word this call spelled — carried through for the
    /// decline sentences that name it (an unfit-input decline, an
    /// unrecognized script extension); every recognized runner
    /// discharges the runtime-identity premise identically once the
    /// artifact's own band check (`foreign_edge_artifact.rs`) passes.
    runner: Runner,
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
    Override { parse_range: TextRange, value: AbstractValue },
    /// The crossing escapes what the target states it admits: an
    /// RTS7001 the caller reports at `range` (the payload), never a
    /// decline — the call is wrong, so there is no fact to attach.
    Fired { message: String, range: TextRange },
    /// The sentence naming the premise that stopped the edge (an
    /// RTS7002 the caller records as this body's blocker), and where it
    /// points.
    Decline { message: String, range: TextRange },
}

/// Recognizes a cross-language call at `statements[index]` and, on all
/// premises green, answers the override the caller publishes for the
/// rest of the body's walk.
///
/// Answers `None` for every statement that is not this shape — the
/// ordinary walk is untouched and pays one recognizer's worth of
/// syntax tests. A recognized edge that cannot be completed answers an
/// outcome carrying a decline sentence: an edge the checker sees and
/// cannot serve is a work-queue item, never a silence.
pub fn foreign_edge_at(
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    // The idiomatic `with subprocess.Popen([...]) as process:` wrapping
    // (`level_via_popen_context_manager`'s own shape) puts its OWN
    // consumer — the `.communicate()` assign and, later, the
    // `json.loads(...)` read — inside the WITH-BLOCK's own body, never
    // as a sibling of the `with` statement in `statements`. Every other
    // recognized shape (a plain `Assign`, or the temp-file `with`) keeps
    // scanning `statements` exactly as before; only this one shape scans
    // its own nested body instead.
    if let Stmt::With(with_stmt) = &statements[index] {
        if recognize_temp_file_edge(with_stmt, statements, index, environment).is_none() {
            if let Some(edge) = recognize_popen_context_manager_edge(with_stmt, environment) {
                return finish_recognized_edge(edge, &with_stmt.body, environment, kernel, entry_directory);
            }
        }
    }
    let edge = recognize_foreign_edge(statements, index, environment)?;
    finish_recognized_edge(edge, statements, environment, kernel, entry_directory)
}

/// Recognizes a cross-language call directly off a walrus-bound `<name>
/// := subprocess.<callee>(...)` inside an `if`/`elif` TEST (`level_via_
/// walrus_result`'s own shape: `Stmt::If`, never an `Assign`/`With`, so
/// `foreign_edge_at`'s own `statements[index]` dispatch structurally
/// never reaches it) and, on all premises green, answers the override
/// the caller publishes for the rest of the ARM body's walk.
///
/// `target`/`call` are the walrus's own `Expr::Named::target`/`value`
/// (already destructured by the caller, which knows the walrus shape);
/// `arm_body` is the taken arm's own statement list — the return leg's
/// sole-consumer scan runs over THAT list (`sole_parse_consumer_of`
/// reads forward from `arm_scan_from`), since the `json.loads(...)`
/// consumer sits inside the arm, never as a sibling of the outer `if`.
/// Answers `None` for every callee this crate does not recognize at all
/// — the same "not this shape, no sentence owed" contract
/// `recognize_foreign_edge` keeps for its own Assign path.
pub fn foreign_edge_at_walrus_call(
    call: &ExprCall,
    target: &ExprName,
    arm_body: &[Stmt],
    arm_scan_from: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    let edge = recognize_subprocess_callee(call, target, arm_body, arm_scan_from, environment)?;
    // The walrus-bound call sits inside the `if` TEST, never as a member
    // of `arm_body` — there is no call STATEMENT for the return leg's
    // scan to skip past, unlike the `Stmt::Assign`/`Stmt::With` shapes
    // `finish_recognized_edge`'s other callers supply. The whole arm
    // body is offered to the consumer scan from its own start.
    finish_recognized_edge_from_start(edge, arm_body, environment, kernel, entry_directory)
}

/// The post-recognition premises every recognized edge shares, whichever
/// syntactic shape (`Stmt::Assign`, `Stmt::With`, or a walrus-bound call
/// inside an `if` test) supplied it: resolve a relative target path,
/// read the target's own artifact, check the carrier identity, discharge
/// the outbound leg, and check channel purity. Answers the discharged
/// `ForeignEdge`/artifact pair once every premise up to (not including)
/// the return leg's own consumer scan is green — the one step that
/// differs between callers (`sole_parse_consumer_of`'s "skip past the
/// call statement" scan for `Stmt::Assign`/`Stmt::With`, versus the
/// walrus path's "scan the whole arm body" one, since there is no call
/// statement to skip past at all), left to each caller's own thin
/// wrapper.
fn discharge_edge_premises(
    edge: Result<ForeignEdge, RecognitionDecline>,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Result<(ForeignEdge, ForeignTsArtifact), ForeignEdgeOutcome> {
    let mut edge = match edge {
        Ok(edge) => edge,
        Err(decline) => {
            return Err(ForeignEdgeOutcome::Decline { message: decline.message, range: decline.range });
        }
    };
    // A relative argv entry is relative to the FILE that wrote it, never
    // to the process's cwd — join it against the checked file's own
    // directory the moment both are in hand.
    if let Some(directory) = entry_directory {
        let target = std::path::Path::new(&edge.target_path);
        if target.is_relative() {
            edge.target_path = directory.join(target).to_string_lossy().into_owned();
        }
    }
    let artifact = match read_foreign_ts_artifact(&edge.target_path) {
        Ok(artifact) => artifact,
        Err(reason) => {
            return Err(ForeignEdgeOutcome::Decline {
                message: "the target ".to_owned() + &edge.target_path + " states no fact for this edge — " + &reason,
                range: edge.call,
            });
        }
    };
    // RUNTIME IDENTITY: the artifact's own band names an ECMA-262 spec
    // level, not one runtime binary (ruling, 2026-08-21) — the sibling
    // reader already checked the band against this checker's pinned
    // `es2023+` string, so any recognized runner (node, deno, bun, npx
    // tsx) discharges this premise identically once that check passed.
    //
    // CARRIER IDENTITY: the call's own spelling states one channel, and
    // the target's surface states the one it actually reads — a JSON
    // transport model applies only when both name the SAME carrier.
    if let Some(mismatch) = channel_mismatch_decline(edge.channel, &artifact.surface) {
        return Err(ForeignEdgeOutcome::Decline { message: mismatch, range: edge.call });
    }
    // the OUTBOUND leg: every premise about what crosses out, discharged
    // against the value the walk holds for it
    if let Some(outcome) = check_outbound_leg(&edge, &artifact, environment, kernel) {
        return Err(outcome);
    }
    // CHANNEL PURITY: the wire is stdout, and the claim assumes stdout
    // carries exactly the serialized result
    if !artifact.called.stdout_pure {
        return Err(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " does not state that it writes "
                + "nothing else to stdout, and this edge reads its result off stdout — "
                + "the channel-purity premise is undischarged",
            range: edge.call,
        });
    }
    Ok((edge, artifact))
}

/// Answers the return leg's own outcome once `sole_parse_consumer_of`
/// (or its inclusive-scan sibling) has already run: the target's own
/// fact, attached to the parse — unless the declared return admits a
/// corner the target's own `JSON.stringify` serializes as `null`, which
/// degrades to a named undetermined instead of binding the set as
/// stated.
fn return_leg_outcome(consumer: Result<TextRange, String>, artifact: &ForeignTsArtifact, call: TextRange) -> ForeignEdgeOutcome {
    match consumer {
        Ok(parse_range) => match foreign_return_value_or_undetermined(artifact) {
            Ok(value) => ForeignEdgeOutcome::Override { parse_range, value },
            Err(message) => ForeignEdgeOutcome::Decline { message, range: parse_range },
        },
        Err(message) => ForeignEdgeOutcome::Decline { message, range: call },
    }
}

/// `discharge_edge_premises` plus the ordinary return-leg scan
/// (`sole_parse_consumer_of`, which skips past the call's own statement
/// at `edge.consumer_scan_from`) — the `Stmt::Assign`/`Stmt::With`
/// callers' own finish, unchanged from before the walrus entry point
/// existed.
fn finish_recognized_edge(
    edge: Result<ForeignEdge, RecognitionDecline>,
    statements: &[Stmt],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    let (edge, artifact) = match discharge_edge_premises(edge, environment, kernel, entry_directory) {
        Ok(discharged) => discharged,
        Err(outcome) => return Some(outcome),
    };
    let consumer = sole_parse_consumer_of(statements, edge.consumer_scan_from, &edge.result_name, edge.result_read);
    Some(return_leg_outcome(consumer, &artifact, edge.call))
}

/// `discharge_edge_premises` plus the INCLUSIVE return-leg scan
/// (`sole_parse_consumer_from`, over the whole of `statements` — no call
/// statement to skip past) — `foreign_edge_at_walrus_call`'s own finish,
/// since its recognized call sits inside the `if` TEST rather than as a
/// member of `statements` at all.
fn finish_recognized_edge_from_start(
    edge: Result<ForeignEdge, RecognitionDecline>,
    statements: &[Stmt],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    let (edge, artifact) = match discharge_edge_premises(edge, environment, kernel, entry_directory) {
        Ok(discharged) => discharged,
        Err(outcome) => return Some(outcome),
    };
    let consumer = sole_parse_consumer_from(statements, &edge.result_name, edge.result_read);
    Some(return_leg_outcome(consumer, &artifact, edge.call))
}

/// Whether the call's own carrier and the target's declared surface
/// name the SAME channel — `None` when they meet, the decline sentence
/// naming the mismatch otherwise. Neither direction is a recognition
/// failure: the call is a real, well-formed shape, and the target's
/// fact is a real, well-formed fact; they simply do not speak the same
/// carrier, so nothing here can apply the JSON transport model.
fn channel_mismatch_decline(channel: Channel, surface: &ForeignSurface) -> Option<String> {
    match (channel, surface) {
        (Channel::Stdin, ForeignSurface::StdinJson) => None,
        (Channel::Argv { arg_index }, ForeignSurface::ArgvJson { arg_index: declared_index })
            if arg_index == *declared_index =>
        {
            None
        }
        (Channel::File { arg_index }, ForeignSurface::FileJson { arg_index: declared_index })
            if arg_index == *declared_index =>
        {
            None
        }
        (Channel::Stdin, ForeignSurface::ArgvJson { .. }) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_stdin_at_argv_target())
        }
        (Channel::Argv { .. }, ForeignSurface::StdinJson) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_argv_at_stdin_target())
        }
        (Channel::Argv { arg_index }, ForeignSurface::ArgvJson { arg_index: declared_index }) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_argv_index(arg_index, *declared_index))
        }
        (Channel::Stdin, ForeignSurface::FileJson { .. }) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_stdin_at_file_target())
        }
        (Channel::File { .. }, ForeignSurface::StdinJson) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_file_at_stdin_target())
        }
        (Channel::Argv { .. }, ForeignSurface::FileJson { .. }) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_argv_at_file_target())
        }
        (Channel::File { .. }, ForeignSurface::ArgvJson { .. }) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_file_at_argv_target())
        }
        (Channel::File { arg_index }, ForeignSurface::FileJson { arg_index: declared_index }) => {
            Some(diagnostic_sentences::foreign_edge_channel_mismatch_file_index(arg_index, *declared_index))
        }
    }
}

/// `read_foreign_ts_artifact` under test: no sibling module to link
/// against, so tests exercise this module's own fixture-backed stub
/// instead. See the module doc's TODO-sibling note.
#[cfg(test)]
fn read_foreign_ts_artifact(target_path: &str) -> Result<ForeignTsArtifact, String> {
    tests::test_read_foreign_ts_artifact(target_path)
}

#[cfg(not(test))]
fn read_foreign_ts_artifact(target_path: &str) -> Result<ForeignTsArtifact, String> {
    read_foreign_ts_artifact_landed(target_path)
}

/// The fact the `json.loads` result wears: the target's stated return
/// cases, at the grade the crossing's weakest cited boundary admits —
/// one case lowers directly to its own value; more than one lowers to a
/// `Kind::KindUnion` of arms (the machinery every consumer of a sort
/// union already shares: `judge`, isinstance/match narrowing). An
/// `ForeignCase::Object` return case is real wire vocabulary now: it
/// lowers to `Kind::Object` through the same `known_object` constructor
/// every dict-literal read already builds (`foreign_case_value`'s own
/// Object arm), so the Result-shape corpus of "several object cases in
/// one return list" reads through this exact same function unchanged —
/// `kind_union_of` is already generic over its arms' own Kind, taking no
/// object-specific branch.
///
/// `TrustSpec`, mirroring the Go twin's `foreignReturnValue`: the value
/// is not this kernel's own decision about this expression, it is
/// another language's claim carried across a transport whose identity
/// is a CITED PREMISE, not a proved theorem.
fn foreign_return_value(artifact: &ForeignTsArtifact) -> Result<AbstractValue, String> {
    foreign_case_list_value(&artifact.called.return_cases, &artifact.called.name)
}

/// A cases LIST lowered to one `AbstractValue` — one case direct, several
/// through `kind_union_of` — the same channel `foreign_return_value`
/// applies at the top level and a member's own `Vec<ForeignCase>`
/// (`ForeignCase::Object`'s own field) applies once per member, since a
/// member's cases list is the identical "one or several wire cases name
/// one value" shape recursed one layer down.
fn foreign_case_list_value(cases: &[ForeignCase], function_name: &str) -> Result<AbstractValue, String> {
    let mut values = Vec::with_capacity(cases.len());
    for case in cases {
        values.push(foreign_case_value(case, function_name)?);
    }
    Ok(kind_union_of(values))
}

/// One case lowered to its own `AbstractValue`: a number/string set
/// case's SORT comes from the case tag itself, never guessed from the
/// set's own forms — the whole point of the wire stating "number" or
/// "string" explicitly is that a crossed return never needs the
/// declared-position sort law's own shape heuristic. `requires_or_
/// reads_integer` still decides Integer vs Float WITHIN a number case
/// (a union-of-integer-literal return like `union_levels.ts`'s derived
/// `{1, 2, 4}` reads Integer; a numeric set stating neither reads
/// Float) — that distinction is orthogonal to the case's own number/
/// string/boolean/null tag.
///
/// `ForeignCase::Object` lowers into the domain's own object vocabulary
/// — the same `Kind::Object` shape `collection_models.rs`'s
/// `dict_literal_value` builds for an ordinary `{...}` display, through
/// the identical `known_object` constructor
/// (`refined_domain::known_constructors::known_object`), never a
/// parallel object representation. Each member's own cases list
/// recurses through `foreign_case_list_value` — the same one-direct/
/// several-union channel this function's own caller applies at the top
/// level — so a member typed as several wire cases (a Result-shape
/// member itself carrying a nested object union) lowers the same way a
/// multi-case return does. `complete` comes straight from the case's own
/// `closed`: a closed case states its member list is the WHOLE key set,
/// which is exactly what `known_object`'s `complete` flag claims.
/// Every member key is a plain string entry (`numeric: false`) — the
/// wire's own member-name vocabulary is always a JSON object key, never
/// a Python int-keyed dict entry. `stated` is `None` (no
/// `ObjectAnnotationRef` for a value this checker derived rather than
/// read off a declared annotation) and `bare_proto` is `false`,
/// matching `dict_literal_value`'s own call.
fn foreign_case_value(case: &ForeignCase, function_name: &str) -> Result<AbstractValue, String> {
    Ok(match case {
        ForeignCase::Number(set) => {
            let sort = if requires_or_reads_integer(set) { PrimitiveKind::Integer } else { PrimitiveKind::Float };
            AbstractValue { kind_tag: Some(sort), ..known_set(set.clone(), None, TrustSpec, SetKindTag::None) }
        }
        ForeignCase::String(set) => known_set(set.clone(), None, TrustSpec, SetKindTag::None),
        ForeignCase::Boolean => known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec),
        ForeignCase::Null => null_value(),
        ForeignCase::Object { members, closed } => {
            let mut keys = Vec::with_capacity(members.len());
            for (name, member_cases) in members {
                let value = foreign_case_list_value(member_cases, function_name)?;
                keys.push(ObjectKey { name: name.clone(), numeric: false, value });
            }
            known_object(keys, None, *closed, TrustSpec, false)
        }
    })
}

/// Whether a set's own forms state an integer sort — `requires_integer`
/// (the explicit `Form::Integer` marker, looking through `Union`/
/// `Difference`) OR every value an `OneOf` form admits is a whole,
/// finite number. A crossed return carries no annotation to attach an
/// explicit `Integer` form to (unlike a declared `int`-based alias), so
/// a derived Literal-set return (`union_levels.ts`'s `{1, 2, 4}`) is
/// only ever an all-integer `OneOf` — this is the wider reading the
/// crossed-value case needs beyond the declared-position law it
/// otherwise mirrors.
fn requires_or_reads_integer(set: &RefinedSet) -> bool {
    if requires_integer(set) {
        return true;
    }
    for form in &set.forms {
        match form.form {
            Form::OneOf => {
                if !form.w.is_empty() && form.w.iter().all(|&w| w.is_finite() && w == w.trunc()) {
                    return true;
                }
            }
            Form::Union | Form::Difference => {
                if requires_or_reads_integer(form.a_.as_ref().unwrap())
                    || form.b.as_ref().is_some_and(|b| requires_or_reads_integer(b))
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// Which infinite corner a NUMBER case's own set admits — `None` when
/// the case's own hull is bounded on both ends. A set is an
/// INTERSECTION of its forms, so a ray form (`AtLeast`/`Above` narrows
/// the hull's lower end up; `AtMost`/`Below` narrows the upper end
/// down) only leaves a side unbounded when NO form in the intersection
/// states a finite bound on that side — the same reading
/// `set_simplification.rs`'s own `hull_of` computes for simplification,
/// done locally here since that reader is private to its crate. A
/// `Union` widens to the LOOSER of its two arms' own hulls (either arm
/// admitting the corner means the union does); a `Difference` reads
/// only its left arm's hull (removing members never widens). NaN is
/// excluded from every `RefinedSet` at construction (the boundary
/// ruling), so only the two infinite corners are asked about here. The
/// case tag itself already says "number" — no `on_one_tuple_layer`/
/// `states_sequence` shape gate is needed to tell a number case from a
/// string/sequence one.
fn uncarriable_corner_of(set: &RefinedSet) -> Option<&'static str> {
    let hull = hull_of(set);
    if hull.lo == f64::NEG_INFINITY {
        return Some("-Infinity");
    }
    if hull.hi == f64::INFINITY {
        return Some("+Infinity");
    }
    None
}

/// The outermost bounds a set's own top-level forms state, read
/// syntactically — unbounded (`NEG_INFINITY`/`INFINITY`) on a side no
/// form narrows. `MultipleOf` states no bound and is skipped;
/// `uncarriable_corner_of`'s own gate keeps this reader off a
/// sequence-shaped set entirely, so no sequence form ever reaches this
/// match.
struct ScalarHull {
    lo: f64,
    hi: f64,
}

fn hull_of(set: &RefinedSet) -> ScalarHull {
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for form in &set.forms {
        match form.form {
            Form::AtLeast | Form::Above => lo = lo.max(form.a),
            Form::AtMost | Form::Below => hi = hi.min(form.a),
            Form::OneOf => {
                if !form.w.is_empty() {
                    lo = lo.max(form.w.iter().copied().fold(form.w[0], f64::min));
                    hi = hi.min(form.w.iter().copied().fold(form.w[0], f64::max));
                }
            }
            Form::Union => {
                let a = hull_of(form.a_.as_ref().unwrap());
                let b = hull_of(form.b.as_ref().unwrap());
                lo = lo.max(a.lo.min(b.lo));
                hi = hi.min(a.hi.max(b.hi));
            }
            Form::Difference => {
                let a = hull_of(form.a_.as_ref().unwrap());
                lo = lo.max(a.lo);
                hi = hi.min(a.hi);
            }
            _ => {}
        }
    }
    ScalarHull { lo, hi }
}

/// The return leg's own fact, degraded to a named undetermined when the
/// target's declared return admits a corner (+Infinity or -Infinity)
/// the target's own `JSON.stringify` serializes as the bare token
/// `null` (ECMA-262's `SerializeJSONProperty`, the finiteness check on a
/// Number value — not an RFC 8259 gap, since `1e999` is legal JSON text
/// that parses to Infinity in both runtimes), a value outside the
/// claimed numeric set landing at this call's own consumer. Every
/// NUMBER case among the return's own cases is checked
/// (a string/boolean/null case states no scalar corner this premise is
/// about); a finite-cornered return binds exactly as `foreign_return_
/// value` already reads it. This is the gate every caller of `foreign_
/// return_value` for a RETURN (never an entry — the outbound leg's own
/// NaN-freedom check is the different, already-landed premise for the
/// value crossing OUT) must pass through first.
fn foreign_return_value_or_undetermined(artifact: &ForeignTsArtifact) -> Result<AbstractValue, String> {
    for case in &artifact.called.return_cases {
        let ForeignCase::Number(set) = case else {
            continue;
        };
        if let Some(corner) = uncarriable_corner_of(set) {
            return Err(diagnostic_sentences::foreign_edge_return_admits_uncarriable_corner(
                &artifact.called.name,
                corner,
            ));
        }
    }
    foreign_return_value(artifact)
}

/* ── recognition ──────────────────────────────────────────────────── */

/// A decline the recognizer already knows enough to name — distinct
/// from "not this shape at all," which owes no sentence.
struct RecognitionDecline {
    message: String,
    range: TextRange,
}

/// The recognized runner words — argv[0] (plus, for a two-word runner,
/// argv[1]) that names the program the target `.ts` file runs under.
/// Every runner recognizes the REFERENCE (the argv genuinely names one
/// script) and, once the artifact declares the shared `es2023+` band,
/// discharges the runtime-identity premise identically — the band names
/// an ECMA-262 spec level, not one runtime binary (ruling, 2026-08-21).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Runner {
    Node,
    Deno,
    Bun,
    NpxTsx,
}

impl Runner {
    /// The exact runner text this call spells — carried into a decline
    /// sentence that names the runner (an unfit-input decline, an
    /// unrecognized script extension), never a category.
    fn word(self) -> &'static str {
        match self {
            Runner::Node => "node",
            Runner::Deno => "deno",
            Runner::Bun => "bun",
            Runner::NpxTsx => "npx tsx",
        }
    }
}

/// One argv list read as `[<runner words>, <script>]` — the runner
/// identified and the script's own literal text resolved, independent
/// of which `subprocess.*` callee is being recognized (`run`,
/// `check_output`, and `Popen` all take the same argv shape).
struct ArgvReading {
    runner: Runner,
    script_text: String,
}

/// Reads one `Expr::List` argv literal as `[runner_word(s), script]`:
/// exactly two elements for `Node`/`Bun` (`["node", script]`), exactly
/// three for `Deno`/`NpxTsx` (`["deno", "run", script]`,
/// `["npx", "tsx", script]`). Any other length, or a two/three-element
/// list whose runner word(s) do not match one of these four rows,
/// answers `None` — "some other program, nothing owed" for a
/// recognized-length list with an unrecognized word, and a decline
/// (owed by the caller, since the shape genuinely does not fit ANY
/// known runner) for every other length.
///
/// `None` is also the answer when argv[0] is not a written string
/// literal — an interpreter read through a variable is not a shape any
/// runner row here recognizes (`level_via_runner_variable`'s own row),
/// so the caller declines naming that specifically.
fn argv_runner_and_script(
    argv_list: &ExprList,
    environment: &Environment,
) -> Option<Result<ArgvReading, RecognitionDecline>> {
    let call_range = argv_list.range();
    match argv_list.elts.as_slice() {
        [interpreter, script] => {
            let Some(interpreter_text) = literal_string(interpreter) else {
                return Some(Err(RecognitionDecline {
                    message: "this call's argv[0] is not a written string literal naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match interpreter_text {
                "node" => Runner::Node,
                "bun" => Runner::Bun,
                _ => return None,
            };
            Some(script_text_of(script, environment).map(|script_text| ArgvReading { runner, script_text }))
        }
        [interpreter, second_word, script] => {
            let (Some(interpreter_text), Some(second_word_text)) =
                (literal_string(interpreter), literal_string(second_word))
            else {
                return Some(Err(RecognitionDecline {
                    message: "this call's argv[0] is not a written string literal naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match (interpreter_text, second_word_text) {
                ("deno", "run") => Runner::Deno,
                ("npx", "tsx") => Runner::NpxTsx,
                _ => return None,
            };
            Some(script_text_of(script, environment).map(|script_text| ArgvReading { runner, script_text }))
        }
        _ => Some(Err(RecognitionDecline {
            message: "this call's argv does not hold exactly [\"node\", \"<script>.ts\"] (or a recognized \
                deno/bun/npx-tsx runner row), so the checker cannot name the code that runs next"
                .to_owned(),
            range: call_range,
        })),
    }
}

/// Reads the script element's own text: a written string literal
/// directly, or a bare `Name` this body never rebinds that resolves
/// (through `environment.read`) to a known exact string — the
/// module-level-constant path (`TARGET_PATH = "./targets/level_ok.ts"`
/// used as `["node", TARGET_PATH]`). Every other shape (an f-string, a
/// parameter, a computed expression) declines with the law-2 sentence:
/// the path is computed, and the fix is to spell it as a written
/// string literal.
///
/// A script position always owes either a resolved reading or the
/// law-2 decline — never a bare `None` (that belongs to the caller's
/// own runner-word match, not to this function).
fn script_text_of(script: &Expr, environment: &Environment) -> Result<String, RecognitionDecline> {
    if let Some(literal) = literal_string(script) {
        return Ok(literal.to_owned());
    }
    if let Expr::Name(name) = script {
        if let Some(bound) = environment.read(name.id.as_str()) {
            if let Some(text) = exact_string_text(bound) {
                return Ok(text);
            }
        }
    }
    Err(RecognitionDecline { message: diagnostic_sentences::script_path_not_a_literal(), range: script.range() })
}

/// The exact text an `AbstractValue` carries, if it is a `Kind::Values`
/// state sorted `PrimitiveKind::String` — the same code-point-vector
/// shape every other file in this crate decodes locally
/// (`string_models.rs::exact_string_text`, reimplemented per file per
/// this crate's own no-shared-private-helper convention rather than
/// widening another module's visibility for one caller).
fn exact_string_text(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect())
}

/// Reads one statement as `<name> = subprocess.run(...)`,
/// `<name> = subprocess.check_output(...)`, the two-statement
/// `<name> = subprocess.Popen(...)` / `<a>, <b> = <name>.communicate(...)`
/// pair, or the two-statement AWAITED twin `<name> = await asyncio
/// .create_subprocess_exec(...)` / `<a>, <b> = await <name>.communicate(
/// ...)` — the four `subprocess`/`asyncio` callees whose argv shape and
/// keywords this checker reads.
///
/// `None` — not this shape at all, no sentence owed (including a plain
/// `subprocess.Popen(...)` whose very first statement is not even a
/// `subprocess` call, which is not this recognizer's concern at all).
/// `Some(Err(...))` — this IS a recognized `subprocess.*`/`asyncio.*` call
/// and something about its spelling stopped the resolution, so the caller
/// owes a sentence naming it.
fn recognize_foreign_edge(
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    // The `with subprocess.Popen(...) as process:` wrapping is handled
    // one level up, in `foreign_edge_at` itself, before this function is
    // even called: that shape's own consumer scan must run over the
    // WITH-BLOCK'S body, never over `statements` (the temp-file shape's
    // consumer, in contrast, is a SIBLING statement after the with-block,
    // which is exactly what this function's own `statements`/`index`
    // scan already serves). Reaching this branch at all with a `With`
    // statement therefore means the Popen wrapping already declined (or
    // this is not it), and only the temp-file shape remains to try.
    if let Stmt::With(with_stmt) = &statements[index] {
        return recognize_temp_file_edge(with_stmt, statements, index, environment);
    }
    let Stmt::Assign(assign) = &statements[index] else {
        return None;
    };
    if let Some(decline) = recognize_os_system(assign, environment) {
        return Some(Err(decline));
    }
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    // `await asyncio.create_subprocess_exec(...)` wraps its call in
    // `Expr::Await` — unwrapped and tried FIRST, since no recognized
    // `subprocess.*` callee is ever itself awaited (the sync callees run
    // to completion synchronously), so a bare `Expr::Call` never matches
    // this row and falls straight through to the unchanged sync path.
    if let Expr::Await(awaited) = assign.value.as_ref() {
        if let Expr::Call(call) = awaited.value.as_ref() {
            if let Some(result) =
                recognize_asyncio_create_subprocess_exec(statements, index, call, target, environment)
            {
                return Some(result);
            }
        }
        return None;
    }
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    recognize_subprocess_callee(call, target, statements, index, environment)
}

/// Reads `<target> = subprocess.<attr>(...)`'s CALLEE off an already-
/// destructured `call`/`target` pair — the module-name/shadow check and
/// the `run`/`check_output`/`Popen` dispatch, shared by
/// `recognize_foreign_edge`'s own `Stmt::Assign` path and
/// `foreign_edge_at_walrus_call`'s walrus-bound path (`if (result :=
/// subprocess.run(...)).returncode == 0:`), which has no `Stmt::Assign`
/// to destructure at all — a `Named` expression binds through
/// `Expr::Named::target`/`value` directly, the identical shape once the
/// wrapping statement is stripped away.
fn recognize_subprocess_callee(
    call: &ExprCall,
    target: &ExprName,
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    // a local binding named `subprocess` is not the module — the same
    // shadow-on-rebind rule every other builtin/module recognition in
    // this crate applies (relational_sum.rs's own `sum` shadow check)
    if module_name.id.as_str() != "subprocess" || environment.read("subprocess").is_some() {
        return None;
    }
    match attribute.attr.as_str() {
        "run" => recognize_subprocess_run(call, target, environment, index),
        "check_output" => recognize_subprocess_check_output(call, target, environment, index),
        "Popen" => recognize_subprocess_popen(statements, index, call, target, environment),
        _ => None,
    }
}

/// `<name> = os.system("<shell command>")` — never an override: `os
/// .system` runs a shell command but captures no stdout at all
/// (`library/os.rst`, `os.system`: "the exit status of the process" is
/// the whole return; nothing here reads the command's output), so even
/// a recognized, followed literal command has no captured-stdout leg
/// for a return fact to attach to. `None` when this is not an
/// `os.system` call at all (no sentence owed); `Some(decline)` — this
/// checker sees the shape and every reachable outcome is undetermined.
///
/// A shadowed `os` name is not the module, mirroring the `subprocess`
/// shadow-on-rebind check the other recognizers apply.
fn recognize_os_system(assign: &StmtAssign, environment: &Environment) -> Option<RecognitionDecline> {
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "os" || environment.read("os").is_some() {
        return None;
    }
    if attribute.attr.as_str() != "system" {
        return None;
    }
    let call_range = call.range();
    let [command] = call.arguments.args.as_ref() else {
        return Some(RecognitionDecline {
            message: "this call passes other than one positional command argument, and the checker \
                models only a single written shell-string argument"
                .to_owned(),
            range: call_range,
        });
    };
    let Some(command_text) = literal_string(command) else {
        return Some(RecognitionDecline {
            message: diagnostic_sentences::os_system_shell_string_unreadable(),
            range: call_range,
        });
    };
    let Some(tokens) = tokenize_shell_command(command_text) else {
        return Some(RecognitionDecline {
            message: diagnostic_sentences::os_system_shell_string_unreadable(),
            range: call_range,
        });
    };
    let Some((runner_and_script, remainder)) = split_runner_and_script(&tokens) else {
        return Some(RecognitionDecline {
            message: diagnostic_sentences::os_system_shell_string_unreadable(),
            range: call_range,
        });
    };
    // "< infile" and "> outfile" are the two redirections this row reads
    // past the runner and script, in either order or both — a command
    // line's own way of naming stdin/stdout files. Any other trailing
    // token is unsupported and named specifically rather than silently
    // accepted.
    let Some(redirection_suffix) = redirection_suffix_of(remainder) else {
        return Some(RecognitionDecline {
            message: format!(
                "{} is followed by {}, which this checker's shell-string reader does not admit — only \
                trailing \"< <file>\"/\"> <file>\" redirections are read past the runner and script",
                runner_and_script,
                remainder.join(" ")
            ),
            range: call_range,
        });
    };
    let runner_and_script = runner_and_script + &redirection_suffix;
    // even a followed literal command has no value channel: os.system
    // never captures stdout, so there is no consumer leg to attach a
    // return fact to, regardless of how cleanly the runner+script read
    Some(RecognitionDecline {
        message: diagnostic_sentences::os_system_no_stdout_capture(&runner_and_script),
        range: call_range,
    })
}

/// Reads zero, one, or both of a trailing `< infile` / `> outfile`
/// redirection, in either order, off the tokens following the runner
/// and script. `None` when the trailing tokens are not exactly this
/// shape (an extra flag, a pipe, anything this reader does not admit).
fn redirection_suffix_of(remainder: &[&str]) -> Option<String> {
    match remainder {
        [] => Some(String::new()),
        ["<", input_file] => Some(format!(" < {input_file}")),
        [">", output_file] => Some(format!(" > {output_file}")),
        ["<", input_file, ">", output_file] => Some(format!(" < {input_file} > {output_file}")),
        [">", output_file, "<", input_file] => Some(format!(" > {output_file} < {input_file}")),
        _ => None,
    }
}

/// Splits a shell command string on single spaces — the narrowest
/// tokenizer this row needs (no quoting). `None` for an empty command.
fn tokenize_shell_command(command_text: &str) -> Option<Vec<&str>> {
    if command_text.is_empty() {
        return None;
    }
    Some(command_text.split(' ').collect())
}

/// Reads the leading `[runner(+word), script]` prefix off a tokenized
/// shell command, answering the recognized "runner script" text and
/// whatever tokens follow it. `None` when the leading tokens are not
/// one of the four recognized runner rows at all.
fn split_runner_and_script<'a>(tokens: &'a [&'a str]) -> Option<(String, &'a [&'a str])> {
    match tokens {
        [runner_word, script, rest @ ..] if is_recognized_runner_word(runner_word) => {
            Some((format!("{runner_word} {script}"), rest))
        }
        [runner_word, second_word, script, rest @ ..] if is_recognized_two_word_runner(runner_word, second_word) => {
            Some((format!("{runner_word} {second_word} {script}"), rest))
        }
        _ => None,
    }
}

/// Whether a token is a recognized one-word runner (`node`, `bun`).
fn is_recognized_runner_word(word: &str) -> bool {
    word == "node" || word == "bun"
}

/// Whether two tokens are a recognized two-word runner (`deno run`,
/// `npx tsx`).
fn is_recognized_two_word_runner(first: &str, second: &str) -> bool {
    (first == "deno" && second == "run") || (first == "npx" && second == "tsx")
}

/// The argv list and its resolved runner/script — read once, shared by
/// every callee's own recognition. `None` propagates a not-this-shape
/// answer (an unrecognized argv[0] at a recognized length: "some other
/// program, nothing owed"); `Some(Err(...))` is a decline the caller
/// returns unchanged.
fn recognized_argv(call: &ExprCall, environment: &Environment) -> Option<Result<ArgvReading, RecognitionDecline>> {
    let call_range = call.range();
    let [argv] = call.arguments.args.as_ref() else {
        return Some(Err(RecognitionDecline {
            message: "this call passes other than one positional argv argument, and the checker models \
                only a written argv list naming one script"
                .to_owned(),
            range: call_range,
        }));
    };
    let Expr::List(argv_list) = argv else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv is not one written list literal, so the checker cannot name the \
                code that runs next — no edge is modeled here"
                .to_owned(),
            range: call_range,
        }));
    };
    argv_runner_and_script(argv_list, environment)
}

/// `<name> = subprocess.run(["node", "<script>.ts"], input=json.dumps(
/// <payload>), capture_output=True, text=True)` — the result reads back
/// at `<name>.stdout`. The sibling argv-json shape (`["node",
/// "<script>.ts", json.dumps(<payload>)]`, no `input=` keyword) is tried
/// first: it is a real ambiguity with the ordinary two-element-argv
/// shape only when BOTH an argv payload and `input=` are present, which
/// `argv_json_call_of` itself declines naming the double channel.
fn recognize_subprocess_run(
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    index: usize,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    if let Some(argv_json) = argv_json_call_of(call, target, environment, index) {
        return Some(argv_json);
    }
    let call_range = call.range();
    let reading = match recognized_argv(call, environment)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    let (payload, keywords_decline) = subprocess_run_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(payload) = payload else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} and sends it no json.dumps(...) input, so nothing crosses out on \
                stdin and the transport model has no outbound leg to apply",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::StdoutAttribute,
        consumer_scan_from: index,
        runner: reading.runner,
    }))
}

/// `<name> = subprocess.run(["node", "<script>.ts", json.dumps(<payload>)],
/// capture_output=True, text=True)` — the payload rides the third argv
/// element (node's own convention: `process.argv[2]`) rather than
/// stdin. `None` when this is not that three-element shape at all (the
/// ordinary two-element stdin call, an unrelated argv arity, or a
/// three-element deno/npx-tsx runner row whose own trailing element is
/// the SCRIPT, not a `json.dumps(...)` payload — `literal_string` on
/// that element fails `json_dumps_argument_of`'s own call-shape check,
/// so it falls through here unchanged). `Some(Err(...))` when the shape
/// reads as an argv payload but something about it stops recognition:
/// an unreadable runner/script, a wrong extension, or `input=` ALSO
/// present (the double-channel decline — two crossing values are named
/// and this checker recognizes exactly one transport per call).
fn argv_json_call_of(
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    index: usize,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let call_range = call.range();
    let [argv] = call.arguments.args.as_ref() else {
        return None;
    };
    let Expr::List(argv_list) = argv else {
        return None;
    };
    let [interpreter, script, third] = argv_list.elts.as_slice() else {
        return None;
    };
    let payload = json_dumps_argument_of(third)?;
    let Some(interpreter_text) = literal_string(interpreter) else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv[0] is not a written string literal naming the interpreter".to_owned(),
            range: call_range,
        }));
    };
    let runner = match interpreter_text {
        "node" => Runner::Node,
        "bun" => Runner::Bun,
        _ => {
            return Some(Err(RecognitionDecline {
                message: format!(
                    "this call's argv names {interpreter_text} as the third-position payload's runner, and \
                    the argv-json shape recognizes only node/bun at that position"
                ),
                range: call_range,
            }));
        }
    };
    let script_text = match script_text_of(script, environment) {
        Ok(text) => text,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&script_text, runner, call_range) {
        return Some(Err(decline));
    }
    let (input_present, keywords_decline) = subprocess_run_argv_json_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    if input_present {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::foreign_edge_double_channel_declared(),
            range: call_range,
        }));
    }
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&script_text),
        payload,
        channel: Channel::Argv { arg_index: 2 },
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::StdoutAttribute,
        consumer_scan_from: index,
        runner,
    }))
}

/// `<name> = subprocess.check_output(["node", "<script>.ts"], input=
/// json.dumps(<payload>), text=True)` — the result IS the captured
/// stdout text directly (`library/subprocess.rst`: "the return value is
/// the command's output"), so the sole consumer reads `<name>` bare,
/// never `<name>.stdout`. No `capture_output` keyword exists for this
/// callee (`check_output` always captures), so it is not read here.
fn recognize_subprocess_check_output(
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    index: usize,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let call_range = call.range();
    let reading = match recognized_argv(call, environment)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    let (payload, keywords_decline) = subprocess_check_output_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(payload) = payload else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} and sends it no json.dumps(...) input, so nothing crosses out on \
                stdin and the transport model has no outbound leg to apply",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::Bare,
        consumer_scan_from: index,
        runner: reading.runner,
    }))
}

/// `<name> = subprocess.Popen(["node", "<script>.ts"], stdin=subprocess
/// .PIPE, stdout=subprocess.PIPE, text=True)` immediately followed by
/// `<stdout_name>, <_> = <name>.communicate(json.dumps(<payload>))` —
/// the SAME two-statement-unit discipline `foreign_edge_at`'s own
/// return-leg scan already applies to the call-and-its-consumer, here
/// applied one statement earlier: `.communicate()`'s own call is not a
/// consumer to find again, it is where the payload and the captured
/// name are read, so recognition consumes it here rather than leaving
/// it for `sole_parse_consumer_of` to (mis)count as a second statement
/// writing the name.
fn recognize_subprocess_popen(
    statements: &[Stmt],
    index: usize,
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let call_range = call.range();
    let reading = match recognized_argv(call, environment)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    if let Some(decline) = subprocess_popen_keywords_of(call) {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(next) = statements.get(index + 1) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} through Popen and nothing follows it in this body, so the \
                checker cannot find the .communicate() call that reads the captured output back",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    let Some((stdout_name, payload)) = communicate_call_of(next, target.id.as_str()) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "the statement after this Popen call is not exactly `<name>, <name> = {}.communicate(\
                json.dumps(...))`, so the checker cannot find the captured output or the outbound payload",
                target.id.as_str()
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: stdout_name,
        result_read: ResultRead::Bare,
        consumer_scan_from: index + 1,
        runner: reading.runner,
    }))
}

/// `<name> = await asyncio.create_subprocess_exec("node", "<script>.ts",
/// stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)`
/// immediately followed by `<stdout_name>, <_> = await <name>.communicate(
/// json.dumps(<payload>).encode())` — the awaited twin of
/// `recognize_subprocess_popen`'s own two-statement Popen/`.communicate()`
/// unit. Three deltas from the sync shape, all read through, never a
/// second pipeline: the runner/script argv rides `create_subprocess_exec`'s
/// own VARIADIC positional arguments (`program, *args`) rather than one
/// list literal; both this call and the `.communicate()` call it awaits
/// are wrapped in `Expr::Await`, unwrapped before each is read; and the
/// payload/return values ride BYTES (`json.dumps(...).encode()` going out,
/// `json.loads(...)` reading a `.decode()`-unwrapped — or bare bytes —
/// binding coming back), so `text=True` is neither passed nor required
/// here, unlike every synchronous shape this crate recognizes.
///
/// `None` — not this shape at all (the awaited call is not
/// `asyncio.create_subprocess_exec(...)`, or a shadowed `asyncio` name).
/// `Some(Err(...))` — this IS the recognized awaited call and something
/// about its own spelling, or the awaited `.communicate()` call that must
/// follow it, stops the resolution.
fn recognize_asyncio_create_subprocess_exec(
    statements: &[Stmt],
    index: usize,
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    // a local binding named `asyncio` is not the module — the same
    // shadow-on-rebind rule every other builtin/module recognition in
    // this crate applies (relational_sum.rs's own `sum` shadow check)
    if module_name.id.as_str() != "asyncio"
        || environment.read("asyncio").is_some()
        || attribute.attr.as_str() != "create_subprocess_exec"
    {
        return None;
    }
    let call_range = call.range();
    let reading = match asyncio_argv_runner_and_script(call, environment)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    if let Some(decline) = asyncio_create_subprocess_exec_keywords_of(call) {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(next) = statements.get(index + 1) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} through asyncio.create_subprocess_exec and nothing follows it in \
                this body, so the checker cannot find the awaited .communicate() call that reads the \
                captured output back",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    let Some((stdout_name, payload)) = awaited_communicate_call_of(next, target.id.as_str()) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "the statement after this asyncio.create_subprocess_exec call is not exactly `<name>, \
                <name> = await {}.communicate(json.dumps(...))` (optionally `.encode()`-wrapped), so the \
                checker cannot find the captured output or the outbound payload",
                target.id.as_str()
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: stdout_name,
        result_read: ResultRead::Bare,
        consumer_scan_from: index + 1,
        runner: reading.runner,
    }))
}

/// `asyncio.create_subprocess_exec`'s own argv reading: the runner and
/// script ride the call's VARIADIC positional arguments (`program, *args`)
/// rather than one list literal — `["node", script]`/`["deno", "run",
/// script]`/`["npx", "tsx", script]` reread as `call.arguments.args`
/// holding exactly two or three positional elements, the same runner-word
/// match and script resolution `argv_runner_and_script` already applies
/// to a list literal's own elements.
fn asyncio_argv_runner_and_script(
    call: &ExprCall,
    environment: &Environment,
) -> Option<Result<ArgvReading, RecognitionDecline>> {
    let call_range = call.range();
    match call.arguments.args.as_ref() {
        [interpreter, script] => {
            let Some(interpreter_text) = literal_string(interpreter) else {
                return Some(Err(RecognitionDecline {
                    message: "this call's leading positional argument is not a written string literal \
                        naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match interpreter_text {
                "node" => Runner::Node,
                "bun" => Runner::Bun,
                _ => return None,
            };
            Some(script_text_of(script, environment).map(|script_text| ArgvReading { runner, script_text }))
        }
        [interpreter, second_word, script] => {
            let (Some(interpreter_text), Some(second_word_text)) =
                (literal_string(interpreter), literal_string(second_word))
            else {
                return Some(Err(RecognitionDecline {
                    message: "this call's leading positional argument is not a written string literal \
                        naming the interpreter"
                        .to_owned(),
                    range: call_range,
                }));
            };
            let runner = match (interpreter_text, second_word_text) {
                ("deno", "run") => Runner::Deno,
                ("npx", "tsx") => Runner::NpxTsx,
                _ => return None,
            };
            Some(script_text_of(script, environment).map(|script_text| ArgvReading { runner, script_text }))
        }
        _ => Some(Err(RecognitionDecline {
            message: "this call's positional arguments do not hold exactly (\"node\", \"<script>.ts\") (or \
                a recognized deno/bun/npx-tsx runner row), so the checker cannot name the code that runs \
                next"
                .to_owned(),
            range: call_range,
        })),
    }
}

/// Reads the `asyncio.create_subprocess_exec` keyword arguments:
/// `stdin=asyncio.subprocess.PIPE` and `stdout=asyncio.subprocess.PIPE` —
/// BOTH required (`.communicate()`'s own call, not this one, carries the
/// payload); any other keyword declines. No `text=True` keyword exists
/// for this callee at all (`library/asyncio-subprocess.rst`: the stream
/// always carries bytes), so it is neither read nor required here, unlike
/// `subprocess_popen_keywords_of`'s own sync check. An explicitly
/// non-PIPE `stdout` (a real file handle, `asyncio.subprocess.DEVNULL`,
/// anything else) refuses recognition with the same "cannot read the
/// target's stdout back" sentence the sync Popen row already speaks —
/// one channel-refusal sentence family, not a second one for the async
/// spelling.
fn asyncio_create_subprocess_exec_keywords_of(call: &ExprCall) -> Option<String> {
    let mut stdin_pipe = false;
    let mut stdout_pipe = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return Some(
                "this call passes a keyword argument through **, which the checker cannot read into a \
                fixed set of premises"
                    .to_owned(),
            );
        };
        match name.as_str() {
            "stdin" => stdin_pipe = is_asyncio_subprocess_pipe(&keyword.value),
            "stdout" => stdout_pipe = is_asyncio_subprocess_pipe(&keyword.value),
            other => {
                return Some(format!(
                    "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                ));
            }
        }
    }
    if !stdin_pipe {
        return Some(
            "this call does not pass stdin=asyncio.subprocess.PIPE, so the checker cannot tell that the \
            payload crosses out on stdin"
                .to_owned(),
        );
    }
    if !stdout_pipe {
        return Some(
            "this call does not pass stdout=asyncio.subprocess.PIPE, so the checker cannot read the \
            target's stdout back"
                .to_owned(),
        );
    }
    None
}

/// Whether an expression is exactly `asyncio.subprocess.PIPE` — the
/// awaited shape's own spelling of the sync `subprocess.PIPE` sentinel
/// (`library/asyncio-subprocess.rst`: `asyncio.subprocess.PIPE` is the
/// same integer constant `subprocess.PIPE` is, re-exported under the
/// `asyncio.subprocess` namespace for this callee's own keywords). A
/// two-level attribute chain (`asyncio.subprocess.PIPE`), unlike the
/// sync shape's one-level `subprocess.PIPE`.
fn is_asyncio_subprocess_pipe(expression: &Expr) -> bool {
    let Expr::Attribute(pipe_attribute) = expression else {
        return false;
    };
    if pipe_attribute.attr.as_str() != "PIPE" {
        return false;
    }
    let Expr::Attribute(subprocess_attribute) = pipe_attribute.value.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = subprocess_attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "asyncio" && subprocess_attribute.attr.as_str() == "subprocess"
}

/// Reads `<a>, <b> = await <process_name>.communicate(json.dumps(<payload>)
/// .encode())` — the awaited twin of `communicate_call_of`: the call
/// itself is wrapped in `Expr::Await` (unwrapped first), and the
/// positional argument is `json.dumps(...)` OPTIONALLY wrapped in
/// `.encode()` — the call sends bytes on the wire (`library/asyncio-
/// subprocess.rst`: `Process.communicate`'s own `input` parameter is
/// `bytes | None`), and `.encode()` carries the identical JSON text
/// `json_dumps_argument_of` already reads through; the wrapper itself is
/// stripped before that shared reader ever sees the expression, so a
/// bare `json.dumps(...)` (no `.encode()` at all — a caller relying on
/// `communicate`'s own str-to-bytes convenience, if the target's own
/// stdlib version admits it) is read exactly the same way.
fn awaited_communicate_call_of(statement: &Stmt, process_name: &str) -> Option<(String, Expr)> {
    let Stmt::Assign(assign) = statement else {
        return None;
    };
    let [Expr::Tuple(targets)] = assign.targets.as_slice() else {
        return None;
    };
    let [Expr::Name(stdout_name), _] = targets.elts.as_slice() else {
        return None;
    };
    let Expr::Await(awaited) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Call(call) = awaited.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != process_name || attribute.attr.as_str() != "communicate" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    let unwrapped = unwrap_bytes_encode(argument);
    let payload = json_dumps_argument_of(unwrapped)?;
    Some((stdout_name.id.as_str().to_owned(), payload))
}

/// Strips a trailing `.encode()` call off an expression — `<expr>.encode(
/// )` with no arguments and no keywords answers `<expr>` itself; every
/// other shape (a bare expression with no `.encode()` at all, or a
/// receiver method call `.encode()` does not directly wrap) answers the
/// expression unchanged. Shared by the outbound payload's own `json.dumps(
/// ...).encode()` unwrap and would apply identically to a return-leg
/// `.decode()` unwrap were one written the same way (`unwrap_bytes_decode`
/// is the separate, differently-shaped reader that one needs, since it
/// reads a NAME rather than a call).
fn unwrap_bytes_encode(expression: &Expr) -> &Expr {
    let Expr::Call(call) = expression else {
        return expression;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return expression;
    };
    if attribute.attr.as_str() != "encode" || !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return expression;
    }
    attribute.value.as_ref()
}

/// `with subprocess.Popen(["node", "<script>.ts"], stdin=subprocess.PIPE,
/// stdout=subprocess.PIPE, text=True) as process: <stdout>, _ = process
/// .communicate(json.dumps(<payload>)); return json.loads(<stdout>)` —
/// the idiomatic context-manager spelling of the flat two-statement
/// Popen/`.communicate()` unit `recognize_subprocess_popen` already
/// reads, here read against the with-block's OWN body instead of the
/// statements that follow it (`level_via_popen_context_manager`'s own
/// shape): the with's context expression is the Popen call itself, the
/// `as` target is the name `.communicate()` is called on, and both the
/// `.communicate()` assign and its own consumer sit INSIDE this
/// with-block's body, never as siblings after it.
///
/// `None` — not this shape at all: the context expression is not
/// `subprocess.Popen(...)` (a shadowed `subprocess` name reads the same
/// as not-this-shape, mirroring every other recognizer's shadow-on-
/// rebind check), or the `with` binds no bare name via `as`. `Some(Err(
/// ...))` — this IS a recognized Popen context manager and something
/// about its spelling, or the `.communicate()` call that must open its
/// own body, stops the resolution.
fn recognize_popen_context_manager_edge(
    with_stmt: &StmtWith,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let with_range = with_stmt.range();
    let [item] = with_stmt.items.as_slice() else {
        return None;
    };
    let Expr::Call(call) = &item.context_expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "subprocess"
        || environment.read("subprocess").is_some()
        || attribute.attr.as_str() != "Popen"
    {
        return None;
    }
    let Some(process_name) = item.optional_vars.as_deref().and_then(as_bare_name) else {
        return Some(Err(RecognitionDecline {
            message: "this subprocess.Popen(...) with-statement binds no bare name via 'as', so the checker \
                cannot find the .communicate() call that reads the captured output back"
                .to_owned(),
            range: with_range,
        }));
    };
    let call_range = call.range();
    let reading = match recognized_argv(call, environment)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    if let Some(decline) = subprocess_popen_keywords_of(call) {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(first) = with_stmt.body.first() else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's body is empty, so the checker cannot find the .communicate() call that \
                reads {process_name}'s captured output back"
            ),
            range: with_range,
        }));
    };
    let Some((stdout_name, payload)) = communicate_call_of(first, process_name) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's first statement is not exactly `<name>, <name> = {process_name}.communicate(\
                json.dumps(...))`, so the checker cannot find the captured output or the outbound payload"
            ),
            range: with_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: stdout_name,
        result_read: ResultRead::Bare,
        // The `.communicate()` statement sits at the with-body's own
        // index 0 — the return leg's sole-consumer scan (over
        // `with_stmt.body`, per `foreign_edge_at`'s own routing) starts
        // looking the statement AFTER it, exactly as the flat Popen
        // shape's `index + 1` does relative to its own statement list.
        consumer_scan_from: 0,
        runner: reading.runner,
    }))
}

/// `with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
/// delete=False) as handle: json.dump(<payload>, handle); temp_path =
/// handle.name` immediately followed by `<name> = subprocess.run(
/// ["node", "<script>.ts", temp_path], capture_output=True, text=True)`
/// — the payload crosses through a NAMED TEMP FILE rather than stdin or
/// a `json.dumps(...)` argv element: the `with`-block's own two
/// statements write the payload to the file and bind its path to
/// `temp_path`, and the call's third argv element is that SAME bound
/// name, read bare (never `json.dumps(...)`, since the file already
/// carries the JSON text).
///
/// This is a THREE-STATEMENT unit, one further than `Popen`'s own
/// two-statement lookahead: the `with` statement (recognized here, at
/// `index`) supplies the payload and the path name, and the call the
/// checker still must find sits at `index + 1` — the same
/// one-statement-further lookahead `Popen`'s own recognition applies
/// for `.communicate()`, widened by the one extra statement the
/// `with`-block's own body contributes before the call is even reached.
///
/// CARRIER PREMISE: the bytes `json.dump` writes to the file are the
/// bytes the target reads back at that same path — no intervening
/// write to the file and no reassignment of `temp_path` between the
/// dump and the call. The JSON transport model this shares with
/// `stdin-json`/`argv-json` rests on the identical round-trip premise;
/// only the carrier (a file, rather than a pipe or an argv element)
/// differs.
///
/// `None` — not this shape at all (the `with`'s own context expression
/// is not `tempfile.NamedTemporaryFile(...)`, or a shadowed `tempfile`
/// name). `Some(Err(...))` — the `with`-block is recognizably this
/// shape and something about its spelling, or the call that must
/// follow it, stops the resolution.
fn recognize_temp_file_edge(
    with_stmt: &StmtWith,
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let with_range = with_stmt.range();
    let [item] = with_stmt.items.as_slice() else {
        return None;
    };
    if !is_named_temporary_file_call(&item.context_expr, environment) {
        return None;
    }
    let Some(handle_name) = item.optional_vars.as_deref().and_then(as_bare_name) else {
        return Some(Err(RecognitionDecline {
            message: "this tempfile.NamedTemporaryFile(...) with-statement binds no bare name via 'as', so \
                the checker cannot find the handle json.dump(...) writes through"
                .to_owned(),
            range: with_range,
        }));
    };
    let temp_file_keywords_decline = temp_file_keywords_of(&item.context_expr);
    if let Some(decline) = temp_file_keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: with_range }));
    }
    let [dump_statement, path_statement] = with_stmt.body.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "this tempfile.NamedTemporaryFile(...) with-block does not hold exactly \
                `json.dump(<payload>, <handle>)` followed by `<name> = <handle>.name`, so the checker \
                cannot find the payload this call writes to the temp file"
                .to_owned(),
            range: with_range,
        }));
    };
    let Some(payload) = json_dump_payload_of(dump_statement, handle_name) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's first statement is not exactly json.dump(<payload>, {handle_name}), so \
                the checker cannot find the value written to the temp file"
            ),
            range: with_range,
        }));
    };
    let Some(temp_path_name) = handle_name_binding_of(path_statement, handle_name) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's second statement is not exactly `<name> = {handle_name}.name`, so the \
                checker cannot find the bound path the call must name"
            ),
            range: with_range,
        }));
    };
    // Scans forward for the subprocess.run(...) call that reads the temp
    // file back — the SAME discipline the return leg's own
    // `sole_parse_consumer_of` applies to its result name: an unrelated
    // statement in between is not itself a blocker
    // (`level_via_call_then_unrelated_then_parse`'s own precedent), but a
    // WRITE to `temp_path_name` before the call is found means the file
    // the call would name is no longer provably the file `json.dump`
    // wrote, so that is named and declined rather than silently
    // skipped past.
    let mut call_statement: Option<(usize, &StmtAssign, &ExprCall)> = None;
    for (offset, statement) in statements[index + 1..].iter().enumerate() {
        if let Stmt::Assign(assign) = statement {
            if let Expr::Call(call) = assign.value.as_ref() {
                if is_subprocess_run_call(call, environment) {
                    call_statement = Some((index + 1 + offset, assign, call));
                    break;
                }
            }
        }
        if statement_writes_name(statement, &temp_path_name) {
            return Some(Err(RecognitionDecline {
                message: format!(
                    "{temp_path_name} is written again before a subprocess.run(...) call reads it, so the \
                    file that call would name is not provably the file json.dump wrote"
                ),
                range: statement.range(),
            }));
        }
    }
    let Some((call_position, assign, call)) = call_statement else {
        return Some(Err(RecognitionDecline {
            message: "this temp-file with-block writes the payload out, and no subprocess.run(...) call \
                follows it in this body to read the file back"
                .to_owned(),
            range: with_range,
        }));
    };
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "the subprocess.run(...) call reading this temp file does not bind one plain name, so \
                the checker cannot find the call's own captured result"
                .to_owned(),
            range: with_range,
        }));
    };
    let call_range = call.range();
    let [argv] = call.arguments.args.as_ref() else {
        return Some(Err(RecognitionDecline {
            message: "this call passes other than one positional argv argument, and the checker models \
                only a written argv list naming one script"
                .to_owned(),
            range: call_range,
        }));
    };
    let Expr::List(argv_list) = argv else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv is not one written list literal, so the checker cannot name the \
                code that runs next — no edge is modeled here"
                .to_owned(),
            range: call_range,
        }));
    };
    let [interpreter, script, third] = argv_list.elts.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv does not hold exactly [\"node\", \"<script>.ts\", <temp path name>], \
                so the checker cannot name the code that runs next"
                .to_owned(),
            range: call_range,
        }));
    };
    // the third element must be a BARE NAME reading the SAME binding the
    // with-block produced — a json.dumps(...) call there is the sibling
    // argv-json shape, not this one, and any other shape does not name
    // the temp file's own path at all
    let Some(third_name) = as_bare_name(third) else {
        return Some(Err(RecognitionDecline {
            message: "this call's third argv element is not the bare name the with-block bound to the \
                temp file's path, so the checker cannot tell that this call reads that file back"
                .to_owned(),
            range: call_range,
        }));
    };
    if third_name != temp_path_name {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call's third argv element names {third_name}, and the with-block bound the temp \
                file's path to {temp_path_name} — the checker cannot tell these name the same file"
            ),
            range: call_range,
        }));
    }
    let Some(interpreter_text) = literal_string(interpreter) else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv[0] is not a written string literal naming the interpreter".to_owned(),
            range: call_range,
        }));
    };
    let runner = match interpreter_text {
        "node" => Runner::Node,
        "bun" => Runner::Bun,
        _ => {
            return Some(Err(RecognitionDecline {
                message: format!(
                    "this call's argv names {interpreter_text} as the temp-file shape's runner, and this \
                    checker recognizes only node/bun at that position"
                ),
                range: call_range,
            }));
        }
    };
    let script_text = match script_text_of(script, environment) {
        Ok(text) => text,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&script_text, runner, call_range) {
        return Some(Err(decline));
    }
    let (input_present, keywords_decline) = subprocess_run_argv_json_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    if input_present {
        return Some(Err(RecognitionDecline {
            message: diagnostic_sentences::foreign_edge_double_channel_declared(),
            range: call_range,
        }));
    }
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&script_text),
        payload,
        channel: Channel::File { arg_index: 2 },
        result_name: target.id.as_str().to_owned(),
        result_read: ResultRead::StdoutAttribute,
        consumer_scan_from: call_position,
        runner,
    }))
}

/// Whether an expression is exactly `tempfile.NamedTemporaryFile(...)` —
/// a shadowed `tempfile` name is not the module, mirroring every other
/// recognizer's shadow-on-rebind check.
fn is_named_temporary_file_call(expression: &Expr, environment: &Environment) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "tempfile"
        && environment.read("tempfile").is_none()
        && attribute.attr.as_str() == "NamedTemporaryFile"
}

/// Reads the `NamedTemporaryFile(...)` keyword arguments: `mode="w"`,
/// `suffix=".json"`, `delete=False` — ALL required, any other keyword
/// declines. `None` when every keyword checks out.
fn temp_file_keywords_of(call_expr: &Expr) -> Option<String> {
    let Expr::Call(call) = call_expr else {
        return Some("this is not a call expression".to_owned());
    };
    let mut mode_w = false;
    let mut suffix_json = false;
    let mut delete_false = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return Some(
                "this call passes a keyword argument through **, which the checker cannot read into a \
                fixed set of premises"
                    .to_owned(),
            );
        };
        match name.as_str() {
            "mode" => mode_w = literal_string(&keyword.value) == Some("w"),
            "suffix" => suffix_json = literal_string(&keyword.value) == Some(".json"),
            "delete" => delete_false = matches!(&keyword.value, Expr::BooleanLiteral(literal) if !literal.value),
            other => {
                return Some(format!(
                    "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                ));
            }
        }
    }
    if !mode_w {
        return Some("this call does not pass mode=\"w\", so the checker cannot tell the handle is opened \
            for text writing"
            .to_owned());
    }
    if !suffix_json {
        return Some("this call does not pass suffix=\".json\", so the checker cannot tell the temp file is \
            named as JSON"
            .to_owned());
    }
    if !delete_false {
        return Some("this call does not pass delete=False, so the checker cannot tell the file survives \
            past the with-block for the call to read"
            .to_owned());
    }
    None
}

/// A bare `Name` expression's own identifier text.
fn as_bare_name(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    }
}

/// Reads `json.dump(<payload>, <handle_name>)` — exactly two positional
/// arguments, no keywords, the second a bare name matching `handle_name`.
/// Answers the payload expression.
fn json_dump_payload_of(statement: &Stmt, handle_name: &str) -> Option<Expr> {
    let Stmt::Expr(expr_stmt) = statement else {
        return None;
    };
    let Expr::Call(call) = expr_stmt.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "dump" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [payload, handle_arg] = call.arguments.args.as_ref() else {
        return None;
    };
    if as_bare_name(handle_arg) != Some(handle_name) {
        return None;
    }
    Some(payload.clone())
}

/// Reads `<name> = <handle_name>.name` — the with-block's second
/// statement, binding the temp file's own path to a plain name. Answers
/// the bound name's own text.
fn handle_name_binding_of(statement: &Stmt, handle_name: &str) -> Option<String> {
    let Stmt::Assign(assign) = statement else {
        return None;
    };
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Attribute(attribute) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != handle_name || attribute.attr.as_str() != "name" {
        return None;
    }
    Some(target.id.as_str().to_owned())
}

/// Whether a call is exactly `subprocess.run(...)` — a shadowed
/// `subprocess` name is not the module, mirroring every other
/// recognizer's shadow-on-rebind check.
fn is_subprocess_run_call(call: &ExprCall, environment: &Environment) -> bool {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "subprocess" && environment.read("subprocess").is_none() && attribute.attr.as_str() == "run"
}

/// Whether the script text names a `.ts` file — the one extension this
/// edge models a fact for.
fn script_extension_decline(script_text: &str, runner: Runner, call_range: TextRange) -> Option<RecognitionDecline> {
    if script_text.ends_with(".ts") {
        return None;
    }
    Some(RecognitionDecline {
        message: format!(
            "this call runs {} on {script_text}, which is not a .ts file — the checker models the edge \
            only where the argv names TypeScript source it can read a fact for",
            runner.word()
        ),
        range: call_range,
    })
}

/// Reads `<a>, <b> = <popen_name>.communicate(json.dumps(<payload>))` —
/// exactly a two-element tuple target, a call to `.communicate` on the
/// exact name Popen bound, with exactly one positional `json.dumps(...)`
/// argument. Answers the first target name (the captured stdout text)
/// and the payload expression.
fn communicate_call_of(statement: &Stmt, popen_name: &str) -> Option<(String, Expr)> {
    let Stmt::Assign(assign) = statement else {
        return None;
    };
    let [Expr::Tuple(targets)] = assign.targets.as_slice() else {
        return None;
    };
    let [Expr::Name(stdout_name), _] = targets.elts.as_slice() else {
        return None;
    };
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != popen_name || attribute.attr.as_str() != "communicate" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    let payload = json_dumps_argument_of(argument)?;
    Some((stdout_name.id.as_str().to_owned(), payload))
}

/// The `.ts` path resolved against the checked file's own directory —
/// mirroring the Go twin's own `foreignEdgeOf`: a relative argv entry
/// is relative to the file that wrote it, never the eventual run's cwd.
///
/// This module has no source-file handle of its own (unlike the Go
/// walk, which reads `ast.GetSourceFileOfNode`), so callers resolve a
/// RELATIVE script name against the checked file's directory at the
/// artifact-read call site; here the raw argv text is kept as-is and
/// resolution happens where the checked file's own path is known
/// (`check.rs`'s entry point already threads that path for module
/// resolution). An absolute script name is returned unchanged.
fn resolve_target_path(script_text: &str) -> String {
    script_text.to_owned()
}

/// Reads the `subprocess.run` keyword arguments: `input=json.dumps(...)`,
/// `capture_output=True`, `text=True` — ALL required, any other keyword
/// declines. Answers the payload expression (`None` when `input` is
/// absent or is not a stringify-shaped call) and, on the FIRST keyword
/// shape that stops recognition, the decline sentence naming it.
fn subprocess_run_keywords_of(call: &ExprCall) -> (Option<Expr>, Option<String>) {
    let mut payload: Option<Expr> = None;
    let mut capture_output_true = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return (
                None,
                Some("this call passes a keyword argument through **, which the checker cannot read \
                    into a fixed set of premises"
                    .to_owned()),
            );
        };
        match name.as_str() {
            "input" => {
                payload = json_dumps_argument_of(&keyword.value);
                if payload.is_none() {
                    return (
                        None,
                        Some("this call's input keyword is not json.dumps(...), so the checker cannot \
                            read what crosses out on stdin"
                            .to_owned()),
                    );
                }
            }
            "capture_output" => capture_output_true = literal_true(&keyword.value),
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return (
                    None,
                    Some(format!(
                        "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                    )),
                );
            }
        }
    }
    if !capture_output_true {
        return (
            None,
            Some("this call does not pass capture_output=True, so the checker cannot read the target's \
                stdout back"
                .to_owned()),
        );
    }
    if !text_true {
        return (
            None,
            Some("this call does not pass text=True, so its result is bytes rather than the target's \
                JSON text — the return leg has no text to parse"
                .to_owned()),
        );
    }
    (payload, None)
}

/// Reads the `subprocess.run` keyword arguments for the argv-json call
/// shape: `capture_output=True` and `text=True` are required exactly as
/// they are for the stdin shape, but `input` is admitted here ONLY to
/// be detected and reported back — its presence alongside an argv
/// payload is the double-channel case the caller declines, never a
/// silent second reading of the payload. Any other keyword declines.
fn subprocess_run_argv_json_keywords_of(call: &ExprCall) -> (bool, Option<String>) {
    let mut input_present = false;
    let mut capture_output_true = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return (
                false,
                Some("this call passes a keyword argument through **, which the checker cannot read \
                    into a fixed set of premises"
                    .to_owned()),
            );
        };
        match name.as_str() {
            "input" => input_present = true,
            "capture_output" => capture_output_true = literal_true(&keyword.value),
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return (
                    false,
                    Some(format!(
                        "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                    )),
                );
            }
        }
    }
    if !capture_output_true {
        return (
            input_present,
            Some("this call does not pass capture_output=True, so the checker cannot read the target's \
                stdout back"
                .to_owned()),
        );
    }
    if !text_true {
        return (
            input_present,
            Some("this call does not pass text=True, so its result is bytes rather than the target's \
                JSON text — the return leg has no text to parse"
                .to_owned()),
        );
    }
    (input_present, None)
}

/// Reads the `subprocess.check_output` keyword arguments:
/// `input=json.dumps(...)` and `text=True` — both required, any other
/// keyword declines. `check_output` has no `capture_output` keyword at
/// all (the callee always captures — `library/subprocess.rst`), so it
/// is not read here.
fn subprocess_check_output_keywords_of(call: &ExprCall) -> (Option<Expr>, Option<String>) {
    let mut payload: Option<Expr> = None;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return (
                None,
                Some("this call passes a keyword argument through **, which the checker cannot read \
                    into a fixed set of premises"
                    .to_owned()),
            );
        };
        match name.as_str() {
            "input" => {
                payload = json_dumps_argument_of(&keyword.value);
                if payload.is_none() {
                    return (
                        None,
                        Some("this call's input keyword is not json.dumps(...), so the checker cannot \
                            read what crosses out on stdin"
                            .to_owned()),
                    );
                }
            }
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return (
                    None,
                    Some(format!(
                        "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                    )),
                );
            }
        }
    }
    if !text_true {
        return (
            None,
            Some("this call does not pass text=True, so its result is bytes rather than the target's \
                JSON text — the return leg has no text to parse"
                .to_owned()),
        );
    }
    (payload, None)
}

/// Reads the `subprocess.Popen` keyword arguments: `stdin=subprocess
/// .PIPE`, `stdout=subprocess.PIPE`, `text=True` — ALL required
/// (`.communicate()`'s own call, not this one, carries the payload), any
/// other keyword declines. Answers the decline sentence naming the
/// first shape that stops recognition, or `None` when every keyword
/// checks out.
fn subprocess_popen_keywords_of(call: &ExprCall) -> Option<String> {
    let mut stdin_pipe = false;
    let mut stdout_pipe = false;
    let mut text_true = false;
    for keyword in call.arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            return Some(
                "this call passes a keyword argument through **, which the checker cannot read into a \
                fixed set of premises"
                    .to_owned(),
            );
        };
        match name.as_str() {
            "stdin" => stdin_pipe = is_subprocess_pipe(&keyword.value),
            "stdout" => stdout_pipe = is_subprocess_pipe(&keyword.value),
            "text" => text_true = literal_true(&keyword.value),
            other => {
                return Some(format!(
                    "this call passes the keyword {other}, which this edge's recognized shape does not admit"
                ));
            }
        }
    }
    if !stdin_pipe {
        return Some(
            "this call does not pass stdin=subprocess.PIPE, so the checker cannot tell that the payload \
            crosses out on stdin"
                .to_owned(),
        );
    }
    if !stdout_pipe {
        return Some(
            "this call does not pass stdout=subprocess.PIPE, so the checker cannot read the target's \
            stdout back"
                .to_owned(),
        );
    }
    if !text_true {
        return Some(
            "this call does not pass text=True, so its result is bytes rather than the target's JSON \
            text — the return leg has no text to parse"
                .to_owned(),
        );
    }
    None
}

/// Whether an expression is exactly `subprocess.PIPE`.
fn is_subprocess_pipe(expression: &Expr) -> bool {
    let Expr::Attribute(attribute) = expression else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "subprocess" && attribute.attr.as_str() == "PIPE"
}

/// Reads `json.dumps(<expr>)` and answers the single argument. An
/// f-string wrapping EXACTLY one interpolation, with no literal text
/// around it and no conversion (`!s`/`!r`/`!a`) or format spec
/// (`level_via_fstring_argv_data`'s own `f"{json.dumps(boosted)}"`
/// shape) unwraps to that interpolation's own inner expression first —
/// the spelling an f-string always produces for a lone substitution, and
/// the only shape `json_dumps_argument_of` reads through; a literal
/// character anywhere alongside the interpolation, more than one
/// interpolation, or a conversion/format spec all decline unchanged
/// (falls through to the `Expr::Call` check below, which then answers
/// `None` for the wrapper).
fn json_dumps_argument_of(expression: &Expr) -> Option<Expr> {
    let expression = single_interpolation_call(expression).unwrap_or_else(|| expression.clone());
    let expression = &expression;
    let Expr::Call(call) = expression else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "dumps" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    Some(argument.clone())
}

/// The sole interpolation's own inner expression, for an f-string that
/// spells EXACTLY one substitution and nothing else — no literal
/// character before, between, or after it, and no conversion or format
/// spec on that one interpolation. `None` for every other f-string shape
/// (mixed literal text, zero or multiple interpolations, a conversion or
/// format spec, or an implicitly concatenated f-string, which
/// `as_single_part_fstring` itself already declines) and for any
/// expression that is not `Expr::FString` at all.
fn single_interpolation_call(expression: &Expr) -> Option<Expr> {
    let Expr::FString(fstring) = expression else {
        return None;
    };
    let single = fstring.as_single_part_fstring()?;
    let elements: &[InterpolatedStringElement] = &single.elements;
    let [InterpolatedStringElement::Interpolation(interpolation)] = elements else {
        return None;
    };
    if interpolation.conversion != ConversionFlag::None || interpolation.format_spec.is_some() {
        return None;
    }
    Some(interpolation.expression.as_ref().clone())
}

/// A written string literal's own text.
fn literal_string(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::StringLiteral(literal) => Some(literal.value.to_str()),
        _ => None,
    }
}

/// Whether an expression is the literal `True`.
fn literal_true(expression: &Expr) -> bool {
    matches!(expression, Expr::BooleanLiteral(literal) if literal.value)
}

/* ── the outbound leg ─────────────────────────────────────────────── */

/// Discharges every premise about the value that crosses OUT, against
/// the value the walk holds for it. Answers `None` where the leg is
/// clean; an outcome (a decline, or `Fired` after an RTS7001) where it
/// is not.
fn check_outbound_leg(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    if artifact.called.entry.is_empty() {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " states no entry position, so "
                + "nothing says what the value crossing out must be",
            range: edge.call,
        });
    }
    // the harness hands the WHOLE parsed stdin value to the called
    // function, so exactly one entry position receives it
    if artifact.called.entry.len() != 1 {
        return Some(ForeignEdgeOutcome::Decline {
            message: format!(
                "the target {} states {} entry positions, and this harness hands it one JSON value from \
                stdin — the checker models no splitting of that value across positions",
                artifact.called.name,
                artifact.called.entry.len()
            ),
            range: edge.call,
        });
    }
    let entry = &artifact.called.entry[0];
    let crossing = crate::refinedpy::expressions::evaluate_expression(&edge.payload, environment, kernel);
    let payload_range = edge.payload.range();
    // NaN-FREEDOM: NaN stringifies to `null` in json.dumps, so the
    // target would receive a value this program never computed
    if let Some(sentence) = nan_freedom_obstacle(&crossing) {
        return Some(fire_at(
            payload_range,
            format!(
                "{sentence} — json.dumps writes NaN as null, so {} would receive a value this program \
                never computed",
                artifact.called.name
            ),
            artifact,
        ));
    }
    if let Some((element_cases, length_at_least)) = &entry.sequence {
        return check_sequence_crossing(edge, artifact, entry, element_cases, *length_at_least, &crossing, kernel);
    }
    if let Some(scalar_cases) = &entry.scalar {
        return check_scalar_crossing(edge, artifact, entry, scalar_cases, &crossing, kernel);
    }
    Some(ForeignEdgeOutcome::Decline {
        message: "the target ".to_owned() + &artifact.called.name + " states an entry position " + &entry.name
            + " that is neither a sequence nor a scalar set — nothing says whether the value fits",
        range: payload_range,
    })
}

/// The union of every NUMBER/STRING case's own set among an entry's
/// cases — the one admitted set a kernel `scalar_subset` ask judges a
/// numeric/string crossing value against. `None` when the cases list
/// carries no set-bearing case at all (every case is Boolean/Null, or an
/// Object case), since there is then no set for a numeric/string
/// crossing to fit — the caller's own existing decline sentence
/// ("nothing says whether the value fits") runs unchanged.
///
/// An `ForeignCase::Object` entry case answers no-set BY DESIGN, not as
/// a staging placeholder: the outbound leg's own question is "does the
/// value CROSSING OUT fit the entry's admitted set," and this checker
/// has no OBJECT-shaped crossing value to ask that of at all today (an
/// outbound Python payload never lowers to `Kind::Object` on this path —
/// `expressions::evaluate_expression` is asked for a Python dict's OWN
/// shape, not the entry's declared one). Fitting an outbound object
/// payload against a declared object entry is a SEPARATE designed unit
/// (its own queue entry: a receiver-shaped fit check, not a return-value
/// lowering) — the consumer-side RETURN lowering this file now carries
/// (`foreign_case_value`'s own Object arm) does not, by itself, give the
/// entry leg anything new to check.
fn admitted_set_of_cases(cases: &[ForeignCase]) -> Option<RefinedSet> {
    let mut union_set: Option<RefinedSet> = None;
    for case in cases {
        let set = match case {
            ForeignCase::Number(set) | ForeignCase::String(set) => set.clone(),
            ForeignCase::Boolean | ForeignCase::Null | ForeignCase::Object { .. } => continue,
        };
        union_set = Some(match union_set {
            None => set,
            Some(rest) => make_refined_set(vec![union(set, rest)]),
        });
    }
    union_set
}

/// Whether a value crossing out may carry NaN — the two ways NaN rides
/// beside a set are the `Kind::NaN`/`Kind::PossiblyNaN` wrapper for a
/// scalar and, for a sequence, the `nan_elements` flag its element
/// reading consults. A derived set excludes NaN by construction (a
/// `RefinedSet` denotes a subset of the reals, and NaN is a member of
/// no refined set), so the check is on the value's SHAPE, mirroring the
/// Go twin's `nanFreedomObstacle` exactly.
fn nan_freedom_obstacle(crossing: &AbstractValue) -> Option<&'static str> {
    match crossing.kind {
        Kind::NaN => Some("the value crossing to the TypeScript target is NaN"),
        Kind::PossiblyNaN => Some("the value crossing to the TypeScript target may be NaN"),
        Kind::Set if crossing.nan_elements => {
            Some("the sequence crossing to the TypeScript target may hold NaN elements")
        }
        _ => None,
    }
}

/// Judges an array payload against a sequence entry: the elements
/// inside the union of the element's own number/string cases, and the
/// length floor at or above the stated one.
fn check_sequence_crossing(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    entry: &ForeignTsEntry,
    element_cases: &[ForeignCase],
    length_at_least: i64,
    crossing: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    let payload_range = edge.payload.range();
    if crossing.kind != Kind::Set || crossing.set_kind_tag != SetKindTag::None {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a sequence at " + &entry.name
                + ", and the value crossing out is not read as one here — nothing says whether it fits",
            range: payload_range,
        });
    }
    let Some(window) = as_repetition(&crossing.set) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: format!(
                "the target {} admits a sequence at {} of at least {} elements, and the value crossing \
                out states no element set or length window — nothing says whether it fits",
                artifact.called.name, entry.name, length_at_least
            ),
            range: payload_range,
        });
    };
    let Some(element_set) = admitted_set_of_cases(element_cases) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a sequence at " + &entry.name
                + " whose element cases carry no number/string set — nothing says whether an element fits",
            range: payload_range,
        });
    };
    // the ELEMENT fit — a real kernel ask
    let fits = match foreign_scalar_subset(kernel, &window.element, &element_set) {
        Some(fits) => fits,
        None => {
            return Some(ForeignEdgeOutcome::Decline {
                message: "the kernel refused the question of whether the elements crossing out fit "
                    .to_owned()
                    + &artifact.called.name
                    + "'s stated entry set, so the crossing is not judged",
                range: payload_range,
            });
        }
    };
    if !fits {
        return Some(fire_at(
            payload_range,
            format!(
                "the elements crossing to {} are outside the target's stated entry set — the value can \
                escape what the target states it accepts",
                artifact.called.name
            ),
            artifact,
        ));
    }
    // the LENGTH floor: the target's body relies on it, so a shorter
    // sequence is a different program
    if window.lo < length_at_least {
        return Some(fire_at(
            payload_range,
            format!(
                "the sequence crossing to {} holds at least {} elements, and the target relies on at \
                least {}",
                artifact.called.name, window.lo, length_at_least
            ),
            artifact,
        ));
    }
    None
}

/// Judges a scalar payload against a scalar entry's own cases: a
/// `Kind::Null` crossing fits when a `Null` case is among them (the
/// `admits_none` entry's own reading); every other crossing is judged
/// against the union of the entry's number/string cases through the
/// same `scalar_subset` kernel ask as before — a `Boolean` case widens
/// that union to admit `0`/`1` (a Python `bool` is an `int` subclass),
/// so a numeric judge already covers a boolean crossing without a
/// separate arm.
fn check_scalar_crossing(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    entry: &ForeignTsEntry,
    entry_cases: &[ForeignCase],
    crossing: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    let payload_range = edge.payload.range();
    if crossing.kind == Kind::Null {
        if entry_cases.iter().any(|case| matches!(case, ForeignCase::Null)) {
            return None;
        }
        return Some(fire_at(
            payload_range,
            format!(
                "the value crossing to {} is None, and the target's stated entry admits no null case",
                artifact.called.name
            ),
            artifact,
        ));
    }
    let Some(crossing_set) = set_of_known(crossing) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a value at " + &entry.name
                + ", and the value crossing out is not read as a set here — nothing says whether it fits",
            range: payload_range,
        });
    };
    let mut entry_set = admitted_set_of_cases(entry_cases);
    if entry_cases.iter().any(|case| matches!(case, ForeignCase::Boolean)) {
        let boolean_set = make_refined_set(vec![one_of(&[0.0, 1.0])]);
        entry_set = Some(match entry_set {
            None => boolean_set,
            Some(rest) => make_refined_set(vec![union(boolean_set, rest)]),
        });
    }
    let Some(entry_set) = entry_set else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a value at " + &entry.name
                + " whose cases carry no number/string/boolean set — nothing says whether the value fits",
            range: payload_range,
        });
    };
    let fits = match foreign_scalar_subset(kernel, &crossing_set, &entry_set) {
        Some(fits) => fits,
        None => {
            return Some(ForeignEdgeOutcome::Decline {
                message: "the kernel refused the question of whether the value crossing out fits "
                    .to_owned()
                    + &artifact.called.name
                    + "'s stated entry set, so the crossing is not judged",
                range: payload_range,
            });
        }
    };
    if !fits {
        return Some(fire_at(
            payload_range,
            format!(
                "the value crossing to {} can escape what the target states it accepts",
                artifact.called.name
            ),
            artifact,
        ));
    }
    None
}

/// Asks the kernel A ⊆ B, answering `Some(fits)`, or `None` when the
/// kernel refuses — the same try/catch shape `assignability.rs`'s own
/// `scalar_subset` call wears (assignability.rs:631-643), so a kernel
/// that cannot decide leaves the crossing unjudged rather than refuting
/// it.
fn foreign_scalar_subset(kernel: &Arc<RefinedTSKernel>, a: &RefinedSet, b: &RefinedSet) -> Option<bool> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (kernel.scalar_subset)(a, b))).ok()
}

/// Builds the `Fired` outcome: an RTS7001 sentence with the target's own
/// provenance appended, the way the Go twin's `foreignMessage` does.
fn fire_at(range: TextRange, said: String, artifact: &ForeignTsArtifact) -> ForeignEdgeOutcome {
    ForeignEdgeOutcome::Fired {
        message: diagnostic_sentences::foreign_crossing_refusal(
            &said,
            &artifact.target_file,
            artifact.called.provenance_line,
            &artifact.called.provenance_said,
        ),
        range,
    }
}

/* ── the return leg ───────────────────────────────────────────────── */

/// Finds the `json.loads(<result_name>.stdout)` (or, for `result_read
/// == Bare`, the plain `json.loads(<result_name>)`) node the target's
/// return fact attaches to, scanning the statements AFTER `index` in
/// the same function — the same same-function, count-the-occurrences
/// discipline the Go twin's `soleParseConsumerOf` uses.
///
/// Errs, each because the fact would land on the wrong value:
///
///   - no parse of the name at all: nothing reads the target's output
///     as JSON here, so there is nothing to attach to;
///   - TWO OR MORE parses: one published fact cannot stand for two
///     nodes, and both would read it;
///   - an intervening WRITE to the name: the value the parse reads is
///     then not the value the call produced.
///
/// A parse inside a nested function body is not counted: that scope
/// runs an unstated number of times, so the fact cannot be pinned to
/// one evaluation.
fn sole_parse_consumer_of(
    statements: &[Stmt],
    index: usize,
    result_name: &str,
    result_read: ResultRead,
) -> Result<TextRange, String> {
    sole_parse_consumer_from(&statements[index + 1..], result_name, result_read)
}

/// `sole_parse_consumer_of`'s own scan, taking the slice to scan
/// directly rather than a call's own index plus one — the walrus-bound
/// entry point (`foreign_edge_at_walrus_call`) has no call STATEMENT to
/// skip past at all (the call sits inside the `if` TEST, not as a member
/// of the arm body), so its whole arm body is scanned from its own
/// start, never offset by one.
fn sole_parse_consumer_from(statements: &[Stmt], result_name: &str, result_read: ResultRead) -> Result<TextRange, String> {
    let mut found: Option<TextRange> = None;
    let mut count = 0usize;
    let mut written = false;
    for statement in statements {
        if statement_writes_name(statement, result_name) {
            written = true;
        }
        foreign_parse_calls_in(statement, result_name, result_read, &mut found, &mut count);
    }
    if written {
        return Err(format!(
            "the result binding {result_name} is written after the call, so the value parsed is not the \
            value the TypeScript target produced — no fact is attached"
        ));
    }
    match count {
        0 => Err(format!(
            "nothing reads {result_name} through json.loads after the call, so the target's stated result \
            has no expression to land on"
        )),
        1 => Ok(found.expect("count == 1 implies found is Some")),
        _ => Err(format!(
            "{result_name} is parsed {count} times after the call, and one stated result cannot stand for \
            more than one expression — no fact is attached"
        )),
    }
}

/// Whether a statement writes `name` directly, at any nesting depth of
/// its OWN statements (not inside a nested `def`/`class`, which has its
/// own scope) — an assignment/for/with-as/aug-assign target naming it.
/// Written fresh for this module's own sole-consumer scan (the Go
/// twin's `AssignedNamesDirect` is the model this mirrors, per the
/// mission's own note that no Rust twin exists yet).
fn statement_writes_name(statement: &Stmt, name: &str) -> bool {
    match statement {
        Stmt::Assign(assign) => assign.targets.iter().any(|target| target_names(target, name)),
        Stmt::AnnAssign(assign) => target_names(assign.target.as_ref(), name),
        Stmt::AugAssign(assign) => target_names(assign.target.as_ref(), name),
        Stmt::For(for_stmt) => {
            target_names(for_stmt.target.as_ref(), name)
                || for_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || for_stmt.orelse.iter().any(|inner| statement_writes_name(inner, name))
        }
        Stmt::While(while_stmt) => {
            while_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || while_stmt.orelse.iter().any(|inner| statement_writes_name(inner, name))
        }
        Stmt::If(if_stmt) => {
            if_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || if_stmt
                    .elif_else_clauses
                    .iter()
                    .any(|clause| clause.body.iter().any(|inner| statement_writes_name(inner, name)))
        }
        Stmt::With(with_stmt) => {
            with_stmt.items.iter().any(|item| item.optional_vars.as_deref().is_some_and(|target| target_names(target, name)))
                || with_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
        }
        Stmt::Try(try_stmt) => {
            try_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || try_stmt.handlers.iter().any(|handler| {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(handler) = handler;
                    handler.body.iter().any(|inner| statement_writes_name(inner, name))
                })
                || try_stmt.orelse.iter().any(|inner| statement_writes_name(inner, name))
                || try_stmt.finalbody.iter().any(|inner| statement_writes_name(inner, name))
        }
        // a nested def/class is its own scope, unreachable from this
        // scan, and every other statement form binds no plain name
        _ => false,
    }
}

/// Whether an assignment/for/with target expression names `name`
/// directly — a bare `Name`, or `name` among a `Tuple`/`List` target's
/// own elements.
fn target_names(target: &Expr, name: &str) -> bool {
    match target {
        Expr::Name(identifier) => identifier.id.as_str() == name,
        Expr::Tuple(tuple) => tuple.elts.iter().any(|element| target_names(element, name)),
        Expr::List(list) => list.elts.iter().any(|element| target_names(element, name)),
        Expr::Starred(starred) => target_names(starred.value.as_ref(), name),
        _ => false,
    }
}

/// Counts every parse of `<name>` (per `result_read`) in a statement,
/// recording the first — never descending into a nested function, the
/// same boundary the Go twin's `foreignParseCallsIn` keeps.
fn foreign_parse_calls_in(
    statement: &Stmt,
    name: &str,
    result_read: ResultRead,
    found: &mut Option<TextRange>,
    count: &mut usize,
) {
    visit_statement_exprs(statement, &mut |expression| {
        if is_foreign_parse_of(expression, name, result_read) {
            if found.is_none() {
                *found = Some(expression.range());
            }
            *count += 1;
        }
    });
}

/// Whether a node is exactly `json.loads(<name>.stdout)` (`result_read
/// == StdoutAttribute`) or `json.loads(<name>)`, OPTIONALLY
/// `.decode()`-wrapped (`result_read == Bare`) — the awaited asyncio
/// shape's `stdout_bytes` binding carries raw bytes
/// (`library/asyncio-subprocess.rst`: `Process.communicate`'s own return
/// is `bytes`, never `str`), so `json.loads(stdout_bytes)` reads bytes
/// directly (`json.loads` accepts `bytes | bytearray | str` per
/// `library/json.rst`) exactly as readily as a `.decode()`-unwrapped
/// text binding — both spellings name the identical captured value, so
/// neither is preferred over the other.
fn is_foreign_parse_of(expression: &Expr, name: &str, result_read: ResultRead) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "loads" {
        return false;
    }
    if !call.arguments.keywords.is_empty() {
        return false;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return false;
    };
    match result_read {
        ResultRead::StdoutAttribute => {
            let Expr::Attribute(result_attribute) = argument else {
                return false;
            };
            let Expr::Name(result_name) = result_attribute.value.as_ref() else {
                return false;
            };
            result_name.id.as_str() == name && result_attribute.attr.as_str() == "stdout"
        }
        ResultRead::Bare => {
            let Expr::Name(result_name) = unwrap_bytes_decode(argument) else {
                return false;
            };
            result_name.id.as_str() == name
        }
    }
}

/// Strips a trailing `.decode()` call off an expression — `<expr>.decode(
/// )` with no arguments and no keywords answers `<expr>` itself; every
/// other shape (a bare expression with no `.decode()` at all) answers the
/// expression unchanged. The return-leg counterpart of
/// `unwrap_bytes_encode`'s outbound unwrap: reads a NAME through the
/// wrapper (`is_foreign_parse_of`'s own use, matching the unwrapped
/// expression against `Expr::Name`) rather than an arbitrary expression,
/// so it answers `&Expr` directly rather than a reference the caller
/// must re-match.
fn unwrap_bytes_decode(expression: &Expr) -> &Expr {
    let Expr::Call(call) = expression else {
        return expression;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return expression;
    };
    if attribute.attr.as_str() != "decode" || !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return expression;
    }
    attribute.value.as_ref()
}

/// Walks every expression reachable from a statement without crossing
/// into a nested function/class body, calling `visit` on each. A small,
/// purpose-built walk (rather than reusing `check.rs`'s own
/// `collect_walrus_names` recursion, which is expression-shaped, not
/// statement-shaped) covering exactly the statement forms that can
/// appear between the call and its return in an ordinary function body.
fn visit_statement_exprs(statement: &Stmt, visit: &mut dyn FnMut(&Expr)) {
    match statement {
        Stmt::Expr(expr_stmt) => visit_expr_exprs(expr_stmt.value.as_ref(), visit),
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                visit_expr_exprs(target, visit);
            }
            visit_expr_exprs(assign.value.as_ref(), visit);
        }
        Stmt::AnnAssign(assign) => {
            visit_expr_exprs(assign.target.as_ref(), visit);
            if let Some(value) = assign.value.as_deref() {
                visit_expr_exprs(value, visit);
            }
        }
        Stmt::AugAssign(assign) => {
            visit_expr_exprs(assign.target.as_ref(), visit);
            visit_expr_exprs(assign.value.as_ref(), visit);
        }
        Stmt::Return(ret) => {
            if let Some(value) = ret.value.as_deref() {
                visit_expr_exprs(value, visit);
            }
        }
        Stmt::If(if_stmt) => {
            visit_expr_exprs(if_stmt.test.as_ref(), visit);
            for inner in &if_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for clause in &if_stmt.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    visit_expr_exprs(test, visit);
                }
                for inner in &clause.body {
                    visit_statement_exprs(inner, visit);
                }
            }
        }
        Stmt::For(for_stmt) => {
            visit_expr_exprs(for_stmt.iter.as_ref(), visit);
            for inner in &for_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for inner in &for_stmt.orelse {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::While(while_stmt) => {
            visit_expr_exprs(while_stmt.test.as_ref(), visit);
            for inner in &while_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for inner in &while_stmt.orelse {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                visit_expr_exprs(&item.context_expr, visit);
            }
            for inner in &with_stmt.body {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::Try(try_stmt) => {
            for inner in &try_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for handler in &try_stmt.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    visit_statement_exprs(inner, visit);
                }
            }
            for inner in &try_stmt.orelse {
                visit_statement_exprs(inner, visit);
            }
            for inner in &try_stmt.finalbody {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::Assert(assert_stmt) => {
            visit_expr_exprs(assert_stmt.test.as_ref(), visit);
            if let Some(message) = assert_stmt.msg.as_deref() {
                visit_expr_exprs(message, visit);
            }
        }
        Stmt::Raise(raise_stmt) => {
            if let Some(exc) = raise_stmt.exc.as_deref() {
                visit_expr_exprs(exc, visit);
            }
            if let Some(cause) = raise_stmt.cause.as_deref() {
                visit_expr_exprs(cause, visit);
            }
        }
        // a nested def/class is its own scope; every other statement
        // form (pass, break, continue, import, global, nonlocal, match,
        // delete, type-alias) carries no expression this scan reaches
        _ => {}
    }
}

/// Visits every subexpression of `expression`, never descending into a
/// lambda body — a lambda's body is its own scope, the same rule the
/// statement-level walk keeps for a nested def.
fn visit_expr_exprs(expression: &Expr, visit: &mut dyn FnMut(&Expr)) {
    visit(expression);
    match expression {
        Expr::Lambda(_) => {}
        Expr::BoolOp(op) => op.values.iter().for_each(|value| visit_expr_exprs(value, visit)),
        Expr::BinOp(op) => {
            visit_expr_exprs(op.left.as_ref(), visit);
            visit_expr_exprs(op.right.as_ref(), visit);
        }
        Expr::UnaryOp(op) => visit_expr_exprs(op.operand.as_ref(), visit),
        Expr::If(ternary) => {
            visit_expr_exprs(ternary.test.as_ref(), visit);
            visit_expr_exprs(ternary.body.as_ref(), visit);
            visit_expr_exprs(ternary.orelse.as_ref(), visit);
        }
        Expr::Tuple(tuple) => tuple.elts.iter().for_each(|element| visit_expr_exprs(element, visit)),
        Expr::List(list) => list.elts.iter().for_each(|element| visit_expr_exprs(element, visit)),
        Expr::Set(set) => set.elts.iter().for_each(|element| visit_expr_exprs(element, visit)),
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    visit_expr_exprs(key, visit);
                }
                visit_expr_exprs(&item.value, visit);
            }
        }
        Expr::Call(call) => {
            visit_expr_exprs(call.func.as_ref(), visit);
            for argument in &call.arguments.args {
                visit_expr_exprs(argument, visit);
            }
            for keyword in &call.arguments.keywords {
                visit_expr_exprs(&keyword.value, visit);
            }
        }
        Expr::Compare(compare) => {
            visit_expr_exprs(compare.left.as_ref(), visit);
            compare.comparators.iter().for_each(|comparator| visit_expr_exprs(comparator, visit));
        }
        Expr::Attribute(attribute) => visit_expr_exprs(attribute.value.as_ref(), visit),
        Expr::Subscript(subscript) => {
            visit_expr_exprs(subscript.value.as_ref(), visit);
            visit_expr_exprs(subscript.slice.as_ref(), visit);
        }
        Expr::Starred(starred) => visit_expr_exprs(starred.value.as_ref(), visit),
        Expr::Named(named) => visit_expr_exprs(named.value.as_ref(), visit),
        Expr::Await(inner) => visit_expr_exprs(inner.value.as_ref(), visit),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                visit_expr_exprs(value, visit);
            }
        }
        Expr::YieldFrom(inner) => visit_expr_exprs(inner.value.as_ref(), visit),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::path::PathBuf;

    use refined_domain::abstract_value::possibly_nan;
    use refined_domain::trust_grades::TrustProved;
    use crate::refinedpy::collection_models::subscript_read;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::at_most;
    use refined_sets::refinement_forms::integer;
    use refined_sets::refinement_forms::star;
    use refined_sets::repetition_window_forms::repetition;

    use super::*;
    use crate::refinedpy::foreign_edge_artifact::ForeignSurface;

    thread_local! {
        static FIXTURE_ARTIFACTS: RefCell<HashMap<String, ForeignTsArtifact>> = RefCell::new(HashMap::new());
    }

    /// Registers a fixture artifact under `target_path` for
    /// `read_foreign_ts_artifact`'s test stub to answer — the in-process
    /// stand-in for the sibling's disk-backed reader, so this module's
    /// own recognizer/premise logic is exercised without depending on
    /// `foreign_edge_artifact.rs`'s landed shape.
    fn register_fixture_artifact(target_path: &str, artifact: ForeignTsArtifact) {
        FIXTURE_ARTIFACTS.with(|cell| cell.borrow_mut().insert(target_path.to_owned(), artifact));
    }

    pub(super) fn test_read_foreign_ts_artifact(target_path: &str) -> Result<ForeignTsArtifact, String> {
        FIXTURE_ARTIFACTS.with(|cell| {
            cell.borrow()
                .get(target_path)
                .cloned()
                .ok_or_else(|| format!("there is no artifact for {target_path}"))
        })
    }

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn parsed_body(source: &str) -> Vec<Stmt> {
        ruff_python_parser::parse_module(source).expect("fixture source parses").into_syntax().body.to_vec()
    }

    fn env_with(bindings: &[(&str, AbstractValue)]) -> Environment {
        let mut environment = Environment::new(HashSet::new());
        for (name, value) in bindings {
            environment.bind(name, value.clone());
        }
        environment
    }

    fn boosted_sequence_value() -> AbstractValue {
        known_set(
            repetition(make_refined_set(vec![at_least(-2.0), at_most(2.0)]), 1, None),
            None,
            TrustProved,
            SetKindTag::None,
        )
    }

    fn audio_level_ts_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            path: PathBuf::from("./audio_level.ts.refined.json"),
            called: ForeignTsFunctionFact {
                name: "audioLevel".to_owned(),
                entry: vec![ForeignTsEntry {
                    name: "boosted".to_owned(),
                    sequence: Some((
                        vec![ForeignCase::Number(make_refined_set(vec![at_least(-2.0), at_most(2.0)]))],
                        1,
                    )),
                    scalar: None,
                }],
                return_cases: vec![ForeignCase::Number(make_refined_set(vec![
                    integer(),
                    at_least(0.0),
                    at_most(1.0),
                ]))],
                stdout_pure: true,
                provenance_line: 30,
                provenance_said: "audioLevel's own kernel summary".to_owned(),
            },
            target_file: "./audio_level.ts".to_owned(),
            runtime_band: "es2023+".to_owned(),
            surface: ForeignSurface::StdinJson,
        }
    }

    /// The same fact, on an `argv-json` target reading its payload at
    /// `argv[2]` — the fixture the argv-payload tests register.
    fn audio_level_argv_json_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact { surface: ForeignSurface::ArgvJson { arg_index: 2 }, ..audio_level_ts_artifact() }
    }

    /// The same fact, with an unbounded `atLeast` return — the derived
    /// window a `Math.max(0, x)`-shaped target's own kernel summary
    /// states, admitting +Infinity with no literal spelling needed
    /// (the corner-check fixture: `foreign_edge.rs:181`'s Go-twin
    /// grounding for the identical premise).
    fn audio_level_unbounded_return_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            called: ForeignTsFunctionFact {
                return_cases: vec![ForeignCase::Number(make_refined_set(vec![at_least(0.0)]))],
                ..audio_level_ts_artifact().called
            },
            ..audio_level_ts_artifact()
        }
    }

    /// The same fact, with a float-sorted (no `Integer` form) finite
    /// return window — the sibling of the int-sorted default fixture,
    /// used to pin that an unmarked numeric return still reads Float.
    fn audio_level_float_return_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            called: ForeignTsFunctionFact {
                return_cases: vec![ForeignCase::Number(make_refined_set(vec![at_least(0.0), at_most(1.0)]))],
                ..audio_level_ts_artifact().called
            },
            ..audio_level_ts_artifact()
        }
    }

    /// The same fact, with an all-integer `OneOf` return — the shape
    /// `union_levels.ts`'s derived `{1, 2, 4}` Literal-set return
    /// carries (f-value-unions.py's own `louder_level_wider_window`
    /// pin): no explicit `Integer` form, but every admitted value is a
    /// whole number.
    fn audio_level_one_of_integer_return_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            called: ForeignTsFunctionFact {
                return_cases: vec![ForeignCase::Number(make_refined_set(vec![one_of(&[1.0, 2.0, 4.0])]))],
                ..audio_level_ts_artifact().called
            },
            ..audio_level_ts_artifact()
        }
    }

    /// The same fact, with a single CLOSED, empty-member OBJECT return
    /// case — pins `known_object`'s own shape for the plainest object
    /// case (`foreign_case_value`'s own Object arm): no members, and
    /// `complete: true` straight from `closed`.
    fn audio_level_object_return_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            called: ForeignTsFunctionFact {
                return_cases: vec![ForeignCase::Object { members: vec![], closed: true }],
                ..audio_level_ts_artifact().called
            },
            ..audio_level_ts_artifact()
        }
    }

    /// The Result-shape return: two OBJECT cases in one return list —
    /// `{"ok": bool, "value": number in [0, 1]}` and `{"ok": bool,
    /// "error": string}` — lowering through the same multi-case
    /// `Kind::KindUnion` channel a scalar multi-case return already uses
    /// (`foreign_return_value`'s own doc).
    fn audio_level_result_shape_return_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            called: ForeignTsFunctionFact {
                return_cases: vec![
                    ForeignCase::Object {
                        members: vec![
                            ("ok".to_owned(), vec![ForeignCase::Boolean]),
                            (
                                "value".to_owned(),
                                vec![ForeignCase::Number(make_refined_set(vec![at_least(0.0), at_most(1.0)]))],
                            ),
                        ],
                        closed: true,
                    },
                    ForeignCase::Object {
                        members: vec![
                            ("ok".to_owned(), vec![ForeignCase::Boolean]),
                            (
                                "error".to_owned(),
                                vec![ForeignCase::String(make_refined_set(vec![]))],
                            ),
                        ],
                        closed: true,
                    },
                ],
                ..audio_level_ts_artifact().called
            },
            ..audio_level_ts_artifact()
        }
    }

    /// The same fact, with an OBJECT case at the ENTRY (outbound) leg
    /// instead of the return — `admitted_set_of_cases`'s own Object arm,
    /// a designed remainder distinct from the return-side lowering.
    fn audio_level_object_entry_artifact() -> ForeignTsArtifact {
        ForeignTsArtifact {
            called: ForeignTsFunctionFact {
                entry: vec![ForeignTsEntry {
                    name: "boosted".to_owned(),
                    sequence: None,
                    scalar: Some(vec![ForeignCase::Object { members: vec![], closed: true }]),
                }],
                ..audio_level_ts_artifact().called
            },
            ..audio_level_ts_artifact()
        }
    }

    const FIXTURE_SOURCE: &str = concat!(
        "def audio_level_via_ts(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\"],\n",
        "        input=json.dumps(boosted),\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );

    fn def_body(source: &str) -> Vec<Stmt> {
        let module = parsed_body(source);
        let Stmt::FunctionDef(def) = module.into_iter().next().expect("one top-level def") else {
            panic!("fixture source must be a single def");
        };
        def.body.to_vec()
    }

    /// REGRESSION PIN: the finite, int-sorted return
    /// (`audio_level_ts_artifact`'s `integer, >= 0, <= 1`) binds exactly
    /// as before — a corner-free, explicitly-int-sorted set crosses
    /// undegraded, now correctly Integer-tagged (the fixed reading of
    /// defect 2, not the pre-fix Float stamp).
    #[test]
    fn the_exact_shape_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes");
        match outcome {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// DEFECT 1's fix: a return set admitting +Infinity (an unbounded
    /// `atLeast` with no upper ray) degrades to a named undetermined
    /// naming the corner and the mechanism, rather than binding the set
    /// as stated — the target's own `JSON.stringify(Infinity)` answers
    /// the bare token `null`, not an RFC 8259 gap.
    #[test]
    fn an_unbounded_return_degrades_to_the_named_undetermined() {
        register_fixture_artifact("./audio_level.ts", audio_level_unbounded_return_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("audioLevel"), "{message}");
                assert!(message.contains("+Infinity"), "{message}");
                assert!(message.contains("JSON.stringify"), "{message}");
                assert!(message.contains("cannot be trusted"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => {
                panic!("wanted the corner-check decline, got an override binding the uncarriable set")
            }
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted the corner-check decline, got a fire: {message}")
            }
        }
    }

    /// A single CLOSED, empty-member OBJECT return case binds through
    /// `known_object` — pins the plainest object shape
    /// (`foreign_case_value`'s own Object arm): `Kind::Object`, no keys,
    /// `complete: true` straight from the case's own `closed`.
    #[test]
    fn a_closed_empty_object_return_case_binds_as_a_complete_object() {
        register_fixture_artifact("./audio_level.ts", audio_level_object_return_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Object);
                assert!(value.keys.is_empty());
                assert!(value.complete);
            }
            ForeignEdgeOutcome::Decline { message, .. } => {
                panic!("wanted an override binding the closed empty object, got a decline: {message}")
            }
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted an override binding the closed empty object, got a fire: {message}")
            }
        }
    }

    /// END-TO-END PIN: the Result-shape return (two OBJECT cases — `{ok,
    /// value}` and `{ok, error}`) binds at `json.loads`, and a member
    /// read (`parsed["value"]`, `collection_models.rs`'s `subscript_read`
    /// own `Kind::Object` arm) reaches a judged verdict against a
    /// declared window: the crossed `"value"` member's own number window
    /// is `[0, 1]`, so asking the kernel whether it fits inside `[0, 2]`
    /// answers true, and outside `[10, 20]` answers false — the
    /// consumer-side judge running unchanged over a value this lane's
    /// lowering produced.
    #[test]
    fn a_result_shape_return_binds_and_its_value_member_judges_against_a_declared_window() {
        register_fixture_artifact("./audio_level.ts", audio_level_result_shape_return_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let value = match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => value,
            ForeignEdgeOutcome::Decline { message, .. } => {
                panic!("wanted an override binding the Result-shape union, got a decline: {message}")
            }
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted an override binding the Result-shape union, got a fire: {message}")
            }
        };
        assert_eq!(value.kind, Kind::KindUnion);
        assert_eq!(value.arms.len(), 2);
        for arm in &value.arms {
            assert_eq!(arm.kind, Kind::Object);
            assert!(arm.complete);
        }
        let value_key = known_values(
            "value".chars().map(|c| c as u32 as f64).collect(),
            PrimitiveKind::String,
            TrustProved,
        );
        let value_member = subscript_read(&value.arms[0], &value_key).expect("the \"value\" member reads");
        assert_eq!(value_member.kind, Kind::Set);
        assert_eq!(value_member.kind_tag, Some(PrimitiveKind::Float));
        let inside_window = make_refined_set(vec![at_least(0.0), at_most(2.0)]);
        let outside_window = make_refined_set(vec![at_least(10.0), at_most(20.0)]);
        assert_eq!(foreign_scalar_subset(&kernel, &value_member.set, &inside_window), Some(true));
        assert_eq!(foreign_scalar_subset(&kernel, &value_member.set, &outside_window), Some(false));
    }

    /// An OBJECT case at the ENTRY (outbound) leg declines through the
    /// existing "nothing says whether the value fits" sentence —
    /// `admitted_set_of_cases` answers no-set for an Object case exactly
    /// as it already does for Boolean/Null, so no new sentence is owed
    /// at this leg.
    #[test]
    fn an_object_entry_case_declines_at_the_outbound_leg() {
        register_fixture_artifact("./audio_level.ts", audio_level_object_entry_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("audioLevel"), "{message}");
                assert!(message.contains("whether the value fits"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => {
                panic!("wanted the entry-leg decline, got an override binding an unlowered case")
            }
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted the entry-leg decline, got a fire: {message}")
            }
        }
    }

    /// DEFECT 2's fix: an unmarked, genuinely float-sorted return window
    /// (no `Integer` form) still reads Float — the sibling row proving
    /// the fix does not over-correct into tagging every crossed return
    /// Integer.
    #[test]
    fn a_float_window_return_reads_float_tagged() {
        register_fixture_artifact("./audio_level.ts", audio_level_float_return_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// DEFECT 2's fix: an all-integer `OneOf` return (`{1, 2, 4}`, the
    /// shape `union_levels.ts`'s derived Literal-set return carries,
    /// f-value-unions.py's `louder_level_wider_window` pin) reads
    /// Integer-tagged and passes an integer-window judge — the crossed
    /// value's own sort read from the set, never a Float stamp.
    #[test]
    fn an_all_integer_one_of_return_reads_integer_and_fits_an_integer_window() {
        register_fixture_artifact("./audio_level.ts", audio_level_one_of_integer_return_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let value = match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => value,
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        };
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        // an integer-window judge: {1, 2, 4} subset-of [0, 10] ∧ integer
        let narrow_window = make_refined_set(vec![integer(), at_least(0.0), at_most(10.0)]);
        let fits = foreign_scalar_subset(&kernel, &value.set, &narrow_window);
        assert_eq!(fits, Some(true), "the all-integer OneOf return must fit an integer-window judge");
    }

    /// The recognized foreign-edge shape's `json.loads(result.stdout)`
    /// node is never read through `expressions.rs::
    /// json_loads_value_space` — the honest JSON-union built for an
    /// OPAQUE operand this file holds no fact about
    /// (ISSUES.md, b-runners:124). `foreign_edge_at` builds this
    /// `Override` value directly from the target's own kernel-derived
    /// return fact (this file's own `foreign_return_value`), entirely
    /// separate from `expressions.rs`'s generic `json.loads` handler —
    /// `check.rs`'s `Environment::set_evaluated_node` publishes this
    /// value at the parse node BEFORE any generic evaluation reaches
    /// it, so a recognized target never falls to the union answer this
    /// row's own sibling test (`test_json_loads_of_an_opaque_operand_
    /// answers_the_full_json_union`, expressions.rs) pins for the
    /// UNrecognized case.
    #[test]
    fn a_recognized_target_never_answers_the_generic_json_union() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes");
        match outcome {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_ne!(value.kind, Kind::KindUnion, "the recognized edge's own fact must win, not the opaque union");
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    #[test]
    fn a_shadowed_subprocess_name_is_not_recognized() {
        let body = def_body(FIXTURE_SOURCE);
        let mut environment = env_with(&[("boosted", boosted_sequence_value())]);
        environment.bind("subprocess", known_values(vec![0.0], PrimitiveKind::Integer, TrustProved));
        let Some(kernel) = loaded_kernel() else { return };
        assert!(
            foreign_edge_at(&body, 0, &environment, &kernel, None).is_none(),
            "a locally shadowed subprocess must not be read as the module"
        );
    }

    #[test]
    fn a_missing_capture_output_keyword_declines() {
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized as subprocess.run") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("capture_output"), "{message}");
            }
            _ => panic!("wanted a decline naming the missing capture_output keyword"),
        }
    }

    #[test]
    fn a_result_read_twice_through_json_loads_declines() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    a = json.loads(result.stdout)\n",
            "    b = json.loads(result.stdout)\n",
            "    return a\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("2 times") || message.contains("parsed"), "{message}");
            }
            _ => panic!("wanted a decline naming the sole-consumer violation"),
        }
    }

    #[test]
    fn a_missing_artifact_records_the_export_command_hint() {
        // NOT registered — the fixture stub answers a "no artifact" error,
        // mirroring a missing on-disk cache entry.
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./nowhere.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("./nowhere.ts"), "{message}");
            }
            _ => panic!("wanted a decline naming the missing artifact"),
        }
    }

    #[test]
    fn a_too_wide_outbound_argument_fires() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        // the entry admits -2.0 .. 2.0; this argument's own element set is
        // the full ray, well outside it
        let too_wide = known_set(
            make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let environment = env_with(&[("boosted", too_wide)]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Fired { message, .. } => {
                assert!(message.contains("audioLevel"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire, got an override"),
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
        }
    }

    #[test]
    fn a_possibly_nan_payload_fires_before_the_crossing_fit_is_asked() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let nan_scalar = possibly_nan(known_values(vec![0.0], PrimitiveKind::Float, TrustProved));
        let environment = env_with(&[("boosted", nan_scalar)]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Fired { message, .. } => {
                assert!(message.contains("NaN"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a NaN-freedom fire, got an override"),
            ForeignEdgeOutcome::Decline { message, .. } => {
                panic!("wanted a NaN-freedom fire, got a decline: {message}")
            }
        }
    }

    /* ── the argv-json payload shape ──────────────────────────────── */

    const ARGV_JSON_FIXTURE_SOURCE: &str = concat!(
        "def audio_level_via_argv(boosted):\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\", json.dumps(boosted)],\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );

    /// A fitting argv-json call against a matching argv-json target
    /// recognizes and binds the proved return — silent (`Override`).
    #[test]
    fn a_fitting_argv_json_call_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// An unfitting argv-json payload fires the same RTS7001 the stdin
    /// leg fires — the outbound-leg fit checks are shared, unchanged by
    /// the carrier.
    #[test]
    fn an_unfitting_argv_json_call_fires() {
        register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
        let too_wide = known_set(
            make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let environment = env_with(&[("boosted", too_wide)]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
            ForeignEdgeOutcome::Fired { message, .. } => {
                assert!(message.contains("audioLevel"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire, got an override"),
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
        }
    }

    /// An argv-json call against a `stdin-json` target declines with the
    /// channel-mismatch sentence: the call names a real reference and
    /// the target states a real fact, but the two carriers do not meet.
    #[test]
    fn an_argv_json_call_at_a_stdin_json_target_declines_with_the_mismatch_sentence() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("argv element"), "{message}");
                assert!(message.contains("stdin"), "{message}");
                assert!(message.contains("channels do not meet"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a channel-mismatch decline, got a fire: {message}")
            }
        }
    }

    /// A stdin-json call (`input=json.dumps(...)`, plain two-element
    /// argv) against an `argv-json` target declines with the mismatch
    /// sentence, symmetrically.
    #[test]
    fn a_stdin_json_call_at_an_argv_json_target_declines_with_the_mismatch_sentence() {
        register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the stdin shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("stdin"), "{message}");
                assert!(message.contains("argv element"), "{message}");
                assert!(message.contains("channels do not meet"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a channel-mismatch decline, got a fire: {message}")
            }
        }
    }

    /// `input=json.dumps(...)` alongside an argv-json payload declines
    /// naming the double channel — two crossing values are stated and
    /// this checker recognizes exactly one transport per call.
    #[test]
    fn input_keyword_alongside_an_argv_json_payload_declines_the_double_channel() {
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\", json.dumps(boosted)],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("argv element"), "{message}");
                assert!(message.contains("input=json.dumps"), "{message}");
            }
            _ => panic!("wanted a decline naming the double channel"),
        }
    }

    /* ── the temp-file payload shape ──────────────────────────────── */

    const TEMP_FILE_FIXTURE_SOURCE: &str = concat!(
        "def audio_level_via_temp_file(boosted):\n",
        "    with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\", delete=False) as handle:\n",
        "        json.dump(boosted, handle)\n",
        "        temp_path = handle.name\n",
        "    result = subprocess.run(\n",
        "        [\"node\", \"./audio_level.ts\", temp_path],\n",
        "        capture_output=True,\n",
        "        text=True,\n",
        "    )\n",
        "    return json.loads(result.stdout)\n",
    );

    /// A fitting temp-file call against a matching `file-json` target
    /// recognizes and binds the proved return — silent (`Override`), the
    /// same as the stdin-json and argv-json shapes: only the carrier
    /// differs.
    #[test]
    fn a_fitting_temp_file_call_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact(
            "./audio_level.ts",
            ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
        );
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// An out-of-set payload crossing through the temp-file carrier
    /// fires the same RTS7001 the stdin and argv-json legs fire — the
    /// outbound-leg fit checks are shared, unchanged by the carrier.
    #[test]
    fn an_out_of_set_temp_file_payload_fires() {
        register_fixture_artifact(
            "./audio_level.ts",
            ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
        );
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
        let too_wide = known_set(
            make_refined_set(vec![star(make_refined_set(vec![at_least(-1000.0), at_most(1000.0)]))]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let environment = env_with(&[("boosted", too_wide)]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
            ForeignEdgeOutcome::Fired { message, .. } => {
                assert!(message.contains("audioLevel"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a fire, got an override"),
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted a fire, got a decline: {message}"),
        }
    }

    /// A temp-file call against a `stdin-json` target declines with the
    /// channel-mismatch sentence: the call names a real reference and
    /// the target states a real fact, but the two carriers do not meet.
    #[test]
    fn a_temp_file_call_at_a_stdin_json_target_declines_with_the_mismatch_sentence() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("temp file"), "{message}");
                assert!(message.contains("stdin"), "{message}");
                assert!(message.contains("channels do not meet"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a channel-mismatch decline, got a fire: {message}")
            }
        }
    }

    /// A temp-file call against an `argv-json` target declines with the
    /// channel-mismatch sentence, symmetrically: the target reads the
    /// argv element as the JSON text directly, never as a file path.
    #[test]
    fn a_temp_file_call_at_an_argv_json_target_declines_with_the_mismatch_sentence() {
        register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the temp-file shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("temp file"), "{message}");
                assert!(message.contains("JSON text itself"), "{message}");
                assert!(message.contains("channels do not meet"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a channel-mismatch decline, got a fire: {message}")
            }
        }
    }

    /// A `stdin-json` call (`input=json.dumps(...)`) against a
    /// `file-json` target declines with the mismatch sentence,
    /// symmetrically with the temp-file-at-stdin-target row.
    #[test]
    fn a_stdin_json_call_at_a_file_json_target_declines_with_the_mismatch_sentence() {
        register_fixture_artifact(
            "./audio_level.ts",
            ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
        );
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the stdin shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("stdin"), "{message}");
                assert!(message.contains("file"), "{message}");
                assert!(message.contains("channels do not meet"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a channel-mismatch decline, got a fire: {message}")
            }
        }
    }

    /// An argv-json call (`json.dumps(...)` directly as the third argv
    /// element) against a `file-json` target declines with the mismatch
    /// sentence, symmetrically with the temp-file-at-argv-target row.
    #[test]
    fn an_argv_json_call_at_a_file_json_target_declines_with_the_mismatch_sentence() {
        register_fixture_artifact(
            "./audio_level.ts",
            ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
        );
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the argv-json shape recognizes") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("directly as an argv element"), "{message}");
                assert!(message.contains("file path"), "{message}");
                assert!(message.contains("channels do not meet"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a channel-mismatch decline, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a channel-mismatch decline, got a fire: {message}")
            }
        }
    }

    /// FIX 4: the argv-json payload spelled through an f-string wrapping
    /// exactly one interpolation, `f"{json.dumps(boosted)}"`, rather than
    /// a bare `json.dumps(...)` call (`level_via_fstring_argv_data`,
    /// d-data-legs.py:238). Before this fix, `json_dumps_argument_of`
    /// required `Expr::Call` as its very first check; an f-string is
    /// `Expr::FString`, which failed that guard immediately, so
    /// `argv_json_call_of`'s own payload read answered `None` and the
    /// whole argv-json shape declined to match at all — the call fell
    /// through to the ordinary two-element stdin shape, which itself
    /// declines (a three-element argv is not that shape either), so
    /// `foreign_edge_at` answered a decline naming the wrong construct
    /// (an unrecognized three-element argv) rather than reading through
    /// to the payload. This pins the fix: the trivial single-
    /// interpolation f-string wrapper unwraps to its inner
    /// `json.dumps(...)` call, and the argv-json shape recognizes and
    /// binds the proved return exactly as the bare-call spelling does.
    #[test]
    fn an_fstring_wrapped_argv_json_payload_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def audio_level_via_fstring_argv(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\", f\"{json.dumps(boosted)}\"],\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None)
            .expect("the f-string-wrapped argv-json shape recognizes")
        {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// REGRESSION PIN: the bare `json.dumps(...)` argv-json spelling
    /// still recognizes exactly as before this fix — the f-string
    /// wrapper is an ADDITIONAL readable spelling of the same payload
    /// position, never a replacement for the direct-call shape
    /// `json_dumps_argument_of` already read.
    #[test]
    fn the_bare_call_argv_json_payload_still_recognizes_after_the_fstring_fix() {
        register_fixture_artifact("./audio_level.ts", audio_level_argv_json_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(ARGV_JSON_FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the bare-call argv-json shape recognizes")
        {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// A `temp_path` reassigned between the `with`-block's own dump and
    /// the call that reads it back stays undetermined, naming the
    /// rebound name — the carrier premise (the bytes dumped are the
    /// bytes read) cannot be proved once the name is written again.
    #[test]
    fn a_reassigned_temp_path_between_dump_and_call_stays_undetermined_naming_it() {
        let source = concat!(
            "def f(boosted):\n",
            "    with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\", delete=False) as handle:\n",
            "        json.dump(boosted, handle)\n",
            "        temp_path = handle.name\n",
            "    temp_path = \"/tmp/other.json\"\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\", temp_path],\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the with-block is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("temp_path"), "{message}");
                assert!(message.contains("written again"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a decline naming the rebind, got an override"),
            ForeignEdgeOutcome::Fired { message, .. } => {
                panic!("wanted a decline naming the rebind, got a fire: {message}")
            }
        }
    }

    /// A with-block whose `NamedTemporaryFile(...)` call is missing
    /// `delete=False` declines naming the missing keyword — the file
    /// would not survive past the with-block for the call to read.
    #[test]
    fn a_temp_file_missing_delete_false_declines() {
        let source = concat!(
            "def f(boosted):\n",
            "    with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\") as handle:\n",
            "        json.dump(boosted, handle)\n",
            "        temp_path = handle.name\n",
            "    result = subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\", temp_path],\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the with-block is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("delete=False"), "{message}"),
            _ => panic!("wanted a decline naming the missing delete=False keyword"),
        }
    }

    /// A shadowed `tempfile` name is not recognized as the module.
    #[test]
    fn a_shadowed_tempfile_name_is_not_recognized() {
        let body = def_body(TEMP_FILE_FIXTURE_SOURCE);
        let mut environment = env_with(&[("boosted", boosted_sequence_value())]);
        environment.bind("tempfile", known_values(vec![0.0], PrimitiveKind::Integer, TrustProved));
        let Some(kernel) = loaded_kernel() else { return };
        assert!(
            foreign_edge_at(&body, 0, &environment, &kernel, None).is_none(),
            "a locally shadowed tempfile must not be read as the module"
        );
    }

    /// FIX 3: the exact temp-file unit `TEMP_FILE_FIXTURE_SOURCE` proves,
    /// nested one level inside an outer `with tempfile
    /// .TemporaryDirectory():` block (`level_via_nested_tempdir`,
    /// d-data-legs.py:266). `recognize_temp_file_edge` reads whichever
    /// `statements`/`index` it is handed with no assumption about
    /// nesting depth — this pins that premise directly: calling
    /// `foreign_edge_at` at position 0 of the OUTER with-block's own
    /// body (the exact statement list and index `check.rs`'s
    /// `walk_with` now offers per statement, after this fix) recognizes
    /// and binds the proved return exactly as the top-level case does.
    /// Before this fix, `walk_with` walked its own body through a plain
    /// per-statement `walk_statement` loop with no call into
    /// `foreign_edge_at` at all, so this position was never even
    /// offered the recognition this test drives directly.
    #[test]
    fn a_temp_file_edge_nested_inside_a_temporary_directory_recognizes_and_binds() {
        register_fixture_artifact(
            "./audio_level.ts",
            ForeignTsArtifact { surface: ForeignSurface::FileJson { arg_index: 2 }, ..audio_level_ts_artifact() },
        );
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    with tempfile.TemporaryDirectory():\n",
            "        with tempfile.NamedTemporaryFile(mode=\"w\", suffix=\".json\", delete=False) as handle:\n",
            "            json.dump(boosted, handle)\n",
            "            temp_path = handle.name\n",
            "        result = subprocess.run(\n",
            "            [\"node\", \"./audio_level.ts\", temp_path],\n",
            "            capture_output=True,\n",
            "            text=True,\n",
            "        )\n",
            "        return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let Stmt::With(outer_with) = &body[0] else {
            panic!("this fixture's own top-level statement must be the outer TemporaryDirectory with-block");
        };
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&outer_with.body, 0, &environment, &kernel, None)
            .expect("the nested temp-file shape recognizes")
        {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /* ── subprocess.check_output ──────────────────────────────────── */

    #[test]
    fn check_output_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.check_output(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    #[test]
    fn check_output_with_no_text_keyword_declines() {
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.check_output(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "    )\n",
            "    return json.loads(result)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("text=True"), "{message}"),
            _ => panic!("wanted a decline naming the missing text keyword"),
        }
    }

    /* ── runner words: deno / bun / npx tsx ──────────────────────────── */

    /// A `deno run` call recognizes the reference and, once the
    /// artifact's own band check passes (this fixture declares the
    /// shared `es2023+` band), proceeds to ordinary premise judging
    /// exactly like a `node` call — the runner-word band gap retired
    /// with the ruling that the band names an ECMA-262 spec level, not
    /// one runtime binary.
    #[test]
    fn a_deno_run_call_recognizes_the_reference_and_judges_like_node() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"deno\", \"run\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// A `bun` call recognizes the reference and judges like `node` —
    /// same rationale as the `deno run` sibling above.
    #[test]
    fn a_bun_call_recognizes_the_reference_and_judges_like_node() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"bun\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// An `npx tsx` call recognizes the reference and judges like `node`
    /// — same rationale as the `deno run`/`bun` siblings above.
    #[test]
    fn an_npx_tsx_call_recognizes_the_reference_and_judges_like_node() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"npx\", \"tsx\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    #[test]
    fn a_three_element_argv_with_an_unrecognized_two_word_runner_is_not_this_shape() {
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"yarn\", \"dlx\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        assert!(
            foreign_edge_at(&body, 0, &environment, &kernel, None).is_none(),
            "an unrecognized two-word runner is some other program, nothing owed"
        );
    }

    /* ── the const-held literal path ──────────────────────────────────── */

    #[test]
    fn a_module_level_constant_script_path_resolves_and_binds() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", TARGET_PATH],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[
            ("boosted", boosted_sequence_value()),
            ("TARGET_PATH", string_literal_value_for_test("./audio_level.ts")),
        ]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the const-held path resolves") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// The exact code-point-vector shape a known string constant carries
    /// — the same shape `exact_string_text` decodes, built directly here
    /// (this test module has no import rights into `string_models.rs`'s
    /// own `string_literal_value`).
    fn string_literal_value_for_test(text: &str) -> AbstractValue {
        known_values(text.chars().map(|c| c as u32 as f64).collect(), PrimitiveKind::String, TrustProved)
    }

    #[test]
    fn an_fstring_script_path_declines_with_the_law_2_sentence() {
        let source = concat!(
            "def f(boosted):\n",
            "    name = \"audio_level\"\n",
            "    result = subprocess.run(\n",
            "        [\"node\", f\"./{name}.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("the call is still recognized as subprocess.run") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("computed"), "{message}");
                assert!(message.contains("written string literal"), "{message}");
            }
            _ => panic!("wanted the law-2 decline naming a computed script path"),
        }
    }

    #[test]
    fn a_parameter_script_path_declines_with_the_law_2_sentence() {
        let source = concat!(
            "def f(boosted, script_path):\n",
            "    result = subprocess.run(\n",
            "        [\"node\", script_path],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )\n",
            "    return json.loads(result.stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call is still recognized as subprocess.run") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("computed"), "{message}");
                assert!(message.contains("written string literal"), "{message}");
            }
            _ => panic!("wanted the law-2 decline naming a computed script path"),
        }
    }

    /* ── os.system ────────────────────────────────────────────────────── */

    #[test]
    fn os_system_with_a_recognized_command_declines_naming_the_missing_stdout_capture() {
        let source = concat!(
            "def f(boosted):\n",
            "    exit_code = os.system(\"node ./audio_level.ts < in.json > out.json\")\n",
            "    with open(\"out.json\") as handle:\n",
            "        return json.load(handle)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("os.system is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("captures no stdout"), "{message}");
                assert!(message.contains("subprocess.run"), "{message}");
            }
            _ => panic!("wanted a decline naming the missing captured-stdout leg"),
        }
    }

    #[test]
    fn os_system_with_a_variable_command_declines_with_the_shell_string_sentence() {
        let source = concat!(
            "def f(boosted):\n",
            "    command = \"node ./audio_level.ts\"\n",
            "    exit_code = os.system(command)\n",
            "    with open(\"out.json\") as handle:\n",
            "        return json.load(handle)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 1, &environment, &kernel, None).expect("os.system is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("shell string"), "{message}");
                assert!(message.contains("argv list"), "{message}");
            }
            _ => panic!("wanted the shell-string law-2 decline"),
        }
    }

    #[test]
    fn os_system_with_an_unsupported_trailing_token_names_it() {
        let source = concat!(
            "def f(boosted):\n",
            "    exit_code = os.system(\"node ./audio_level.ts --extra-flag\")\n",
            "    with open(\"out.json\") as handle:\n",
            "        return json.load(handle)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("os.system is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => {
                assert!(message.contains("--extra-flag"), "{message}");
            }
            _ => panic!("wanted a decline naming the unsupported trailing token"),
        }
    }

    /* ── a walrus-bound call inside an `if` test ──────────────────────── */

    /// The walrus's own `Expr::Named::target`/`value` pair, read off the
    /// first statement's own `if`-test — the exact destructuring
    /// `walk_if`'s `serve_foreign_edge_in_walrus_test` (check.rs) performs
    /// before calling `foreign_edge_at_walrus_call`, rebuilt here so this
    /// test drives that same entry point directly rather than through the
    /// checker's own statement walk (per this module's own artifact-stub
    /// constraint: only a test living here can observe an `Override`).
    fn walrus_test_target_and_call(if_stmt: &ruff_python_ast::StmtIf) -> (ExprName, ExprCall) {
        let Expr::Compare(compare) = if_stmt.test.as_ref() else {
            panic!("this fixture's own if-test must be a comparison wrapping the walrus");
        };
        let Expr::Attribute(attribute) = compare.left.as_ref() else {
            panic!("this fixture's own if-test must read an attribute off the walrus-bound name");
        };
        let Expr::Named(named) = attribute.value.as_ref() else {
            panic!("this fixture's own if-test must embed a walrus binding");
        };
        let Expr::Name(target) = named.target.as_ref() else {
            panic!("the walrus target must be a bare name");
        };
        let Expr::Call(call) = named.value.as_ref() else {
            panic!("the walrus value must be a call");
        };
        (target.clone(), call.clone())
    }

    /// FIX 2: `if (result := subprocess.run(...)).returncode == 0: return
    /// json.loads(result.stdout)` (`level_via_walrus_result`, d-data-
    /// legs.py:205). Before this fix, `recognize_foreign_edge` dispatched
    /// only on `statements[index]` being `Stmt::Assign`/`Stmt::With` —
    /// the SAME gate `check.rs`'s own body loop applied before ever
    /// calling `foreign_edge_at` at all. This statement is a `Stmt::If`
    /// whose TEST embeds the walrus, so the recognizer never fired at
    /// all — not recognized-and-blocked, structurally unreached. This
    /// pins the fix's own entry point directly: `foreign_edge_at_walrus_
    /// call`, given the walrus's own target/call and the taken arm's own
    /// body (where the `json.loads(...)` consumer sits), recognizes and
    /// binds the proved return exactly as the flat `Assign`-shaped call
    /// already does.
    #[test]
    fn a_walrus_bound_run_call_in_an_if_test_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def audio_level_via_walrus(boosted):\n",
            "    if (result := subprocess.run(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        input=json.dumps(boosted),\n",
            "        capture_output=True,\n",
            "        text=True,\n",
            "    )).returncode == 0:\n",
            "        return json.loads(result.stdout)\n",
            "    return 0\n",
        );
        let body = def_body(source);
        let Stmt::If(if_stmt) = &body[0] else {
            panic!("this fixture's own top-level statement must be the if");
        };
        let (target, call) = walrus_test_target_and_call(if_stmt);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let outcome = foreign_edge_at_walrus_call(&call, &target, &if_stmt.body, 0, &environment, &kernel, None)
            .expect("the walrus-bound run call recognizes");
        match outcome {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// REGRESSION PIN: the flat `Assign`-shaped spelling
    /// (`result = subprocess.run(...)`, read through `foreign_edge_at`'s
    /// ordinary `Stmt::Assign` dispatch) still recognizes exactly as
    /// before this fix — the walrus-in-test entry point is an ADDITIONAL
    /// way to reach `recognize_subprocess_callee`, never a replacement
    /// for `recognize_foreign_edge`'s own `Stmt::Assign` path.
    #[test]
    fn the_flat_assign_shaped_run_call_still_recognizes_after_the_walrus_fix() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let outcome = foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the flat shape recognizes");
        match outcome {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /* ── subprocess.Popen ─────────────────────────────────────────────── */

    #[test]
    fn popen_with_communicate_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    process = subprocess.Popen(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        stdin=subprocess.PIPE,\n",
            "        stdout=subprocess.PIPE,\n",
            "        text=True,\n",
            "    )\n",
            "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
            "    return json.loads(stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the Popen pair recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    #[test]
    fn popen_with_no_following_communicate_declines() {
        let source = concat!(
            "def f(boosted):\n",
            "    process = subprocess.Popen(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        stdin=subprocess.PIPE,\n",
            "        stdout=subprocess.PIPE,\n",
            "        text=True,\n",
            "    )\n",
            "    return process\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("Popen itself is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("communicate"), "{message}"),
            _ => panic!("wanted a decline naming the missing .communicate() call"),
        }
    }

    #[test]
    fn popen_with_a_missing_stdin_pipe_keyword_declines() {
        let source = concat!(
            "def f(boosted):\n",
            "    process = subprocess.Popen(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        stdout=subprocess.PIPE,\n",
            "        text=True,\n",
            "    )\n",
            "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
            "    return json.loads(stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("Popen itself is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("stdin"), "{message}"),
            _ => panic!("wanted a decline naming the missing stdin=subprocess.PIPE keyword"),
        }
    }

    /// FIX 1: `with subprocess.Popen([...]) as process:` — the idiomatic
    /// context-manager wrapping of the flat Popen/`.communicate()` pair
    /// above (`level_via_popen_context_manager`, a-invocation-
    /// functions.py:80). Before this fix, `recognize_foreign_edge`'s own
    /// `Stmt::With` branch tried only `recognize_temp_file_edge` with no
    /// fallthrough, so this call never reached `recognize_subprocess_
    /// popen` at all — `foreign_edge_at` answered `None` (structurally
    /// unrecognized, not a decline) and `process.communicate(...)`'s
    /// result read as an ordinary opaque call. This pins the fix:
    /// `foreign_edge_at` now tries the Popen-context-manager shape when
    /// the temp-file shape declines the whole with-statement, reading
    /// the `.communicate()` assign and its own `json.loads(...)`
    /// consumer out of the WITH-BLOCK's own body.
    #[test]
    fn popen_inside_a_with_block_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    with subprocess.Popen(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        stdin=subprocess.PIPE,\n",
            "        stdout=subprocess.PIPE,\n",
            "        text=True,\n",
            "    ) as process:\n",
            "        stdout, _stderr = process.communicate(json.dumps(boosted))\n",
            "        return json.loads(stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the with-wrapped Popen pair recognizes")
        {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// REGRESSION PIN: the flat (non-with) Popen/`.communicate()` spelling
    /// still recognizes exactly as before this fix — the with-wrapped
    /// shape is an ADDITIONAL recognized shape, never a replacement for
    /// the statement-pair one `recognize_subprocess_popen` already reads.
    #[test]
    fn the_flat_popen_pair_still_recognizes_after_the_with_fix() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    process = subprocess.Popen(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        stdin=subprocess.PIPE,\n",
            "        stdout=subprocess.PIPE,\n",
            "        text=True,\n",
            "    )\n",
            "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
            "    return json.loads(stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the flat Popen pair still recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /* ── asyncio.create_subprocess_exec ──────────────────────────────── */

    /// EDGE-COVERAGE §K's own headline row (`k-async-invocation.py`'s
    /// `level_via_async_subprocess`): the awaited twin of
    /// `popen_with_communicate_recognizes_and_binds_the_proved_return`,
    /// spelled with `await asyncio.create_subprocess_exec(...)` and
    /// `await proc.communicate(json.dumps(...).encode())` in place of
    /// `subprocess.Popen`/`.communicate(json.dumps(...))`. Pins that the
    /// async spelling now recognizes and binds the target's own proved
    /// return, exactly like the synchronous shape.
    #[test]
    fn async_create_subprocess_exec_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "async def f(boosted):\n",
            "    proc = await asyncio.create_subprocess_exec(\n",
            "        \"node\",\n",
            "        \"./audio_level.ts\",\n",
            "        stdin=asyncio.subprocess.PIPE,\n",
            "        stdout=asyncio.subprocess.PIPE,\n",
            "    )\n",
            "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
            "    return json.loads(stdout_bytes)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the async create_subprocess_exec pair recognizes")
        {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// EDGE-COVERAGE §K's second row (`level_via_async_subprocess_optional`):
    /// the identical async call, with the SAME recognition — the
    /// declared-return widening to `Optional[Level]` this row measures is
    /// a return-judge question, downstream of recognition, so this test
    /// pins that recognition itself is unaffected by the declared return
    /// shape and still binds the same proved fact.
    #[test]
    fn async_create_subprocess_exec_recognizes_regardless_of_the_declared_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "async def f(boosted):\n",
            "    proc = await asyncio.create_subprocess_exec(\n",
            "        \"node\",\n",
            "        \"./audio_level.ts\",\n",
            "        stdin=asyncio.subprocess.PIPE,\n",
            "        stdout=asyncio.subprocess.PIPE,\n",
            "    )\n",
            "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
            "    return json.loads(stdout_bytes)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the async pair recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// The bytes encode/decode unwrapping, pinned on BOTH legs at once:
    /// the outbound payload rides `json.dumps(...).encode()` (unwrapped
    /// by `unwrap_bytes_encode` before `json_dumps_argument_of` reads it)
    /// and the return leg reads `json.loads(stdout_text.decode())`
    /// (unwrapped by `unwrap_bytes_decode` before `is_foreign_parse_of`
    /// matches the name) — the `.encode()`/`.decode()` wrapper on either
    /// leg names the identical JSON text/bytes as the unwrapped spelling
    /// pinned above, so this recognizes and binds the same proved return.
    #[test]
    fn async_create_subprocess_exec_unwraps_encode_and_decode_on_both_legs() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "async def f(boosted):\n",
            "    proc = await asyncio.create_subprocess_exec(\n",
            "        \"node\",\n",
            "        \"./audio_level.ts\",\n",
            "        stdin=asyncio.subprocess.PIPE,\n",
            "        stdout=asyncio.subprocess.PIPE,\n",
            "    )\n",
            "    stdout_text, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
            "    return json.loads(stdout_text.decode())\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the .decode()-wrapped consumer recognizes")
        {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// The bare (non-`.encode()`) payload and bare (non-`.decode()`)
    /// return both still recognize — `json_dumps_argument_of`/
    /// `is_foreign_parse_of` read through an ABSENT wrapper exactly as
    /// readily as a present one, since `unwrap_bytes_encode`/
    /// `unwrap_bytes_decode` answer the expression unchanged when no
    /// `.encode()`/`.decode()` call wraps it.
    #[test]
    fn async_create_subprocess_exec_recognizes_without_encode_or_decode() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "async def f(boosted):\n",
            "    proc = await asyncio.create_subprocess_exec(\n",
            "        \"node\",\n",
            "        \"./audio_level.ts\",\n",
            "        stdin=asyncio.subprocess.PIPE,\n",
            "        stdout=asyncio.subprocess.PIPE,\n",
            "    )\n",
            "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted))\n",
            "    return json.loads(stdout_bytes)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the unwrapped pair recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// An explicitly non-PIPE `stdout` (`asyncio.subprocess.DEVNULL`)
    /// refuses recognition — this call IS `asyncio.create_subprocess_exec`
    /// with a readable runner/script, but the checker cannot read the
    /// target's stdout back at all, so it declines with the same
    /// channel-refusal sentence family `subprocess_popen_keywords_of`'s
    /// own sync check already speaks, never a second sentence for the
    /// async spelling.
    #[test]
    fn async_create_subprocess_exec_with_a_non_pipe_stdout_declines() {
        let source = concat!(
            "async def f(boosted):\n",
            "    proc = await asyncio.create_subprocess_exec(\n",
            "        \"node\",\n",
            "        \"./audio_level.ts\",\n",
            "        stdin=asyncio.subprocess.PIPE,\n",
            "        stdout=asyncio.subprocess.DEVNULL,\n",
            "    )\n",
            "    stdout_bytes, _stderr = await proc.communicate(json.dumps(boosted).encode())\n",
            "    return json.loads(stdout_bytes)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let Some(kernel) = loaded_kernel() else { return };
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the call itself is recognized") {
            ForeignEdgeOutcome::Decline { message, .. } => assert!(message.contains("stdout"), "{message}"),
            _ => panic!("wanted a decline naming the non-PIPE stdout"),
        }
    }

    /// REGRESSION PIN: every synchronous shape this crate already
    /// recognized (`subprocess.run`, `subprocess.Popen`) still recognizes
    /// after the asyncio row is added — the awaited path is reached only
    /// when the assign's own value is `Expr::Await`, so a bare
    /// `Expr::Call` value falls straight through to the unchanged sync
    /// dispatch, never through the new asyncio reader at all.
    #[test]
    fn the_synchronous_subprocess_run_shape_still_recognizes_after_the_asyncio_row() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("subprocess.run still recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /// REGRESSION PIN: the synchronous Popen/`.communicate()` pair still
    /// recognizes after the asyncio row is added — the SAME pin
    /// `the_flat_popen_pair_still_recognizes_after_the_with_fix` already
    /// keeps for the with-wrapped fix, rerun here for the asyncio one.
    #[test]
    fn the_synchronous_popen_shape_still_recognizes_after_the_asyncio_row() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let source = concat!(
            "def f(boosted):\n",
            "    process = subprocess.Popen(\n",
            "        [\"node\", \"./audio_level.ts\"],\n",
            "        stdin=subprocess.PIPE,\n",
            "        stdout=subprocess.PIPE,\n",
            "        text=True,\n",
            "    )\n",
            "    stdout, _stderr = process.communicate(json.dumps(boosted))\n",
            "    return json.loads(stdout)\n",
        );
        let body = def_body(source);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("Popen still recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }
    }

    /* ── disk-backed integration: a real artifact, read for real ────── */
    //
    // These exercise the sibling's own `read_foreign_ts_artifact` against
    // a hand-built artifact JSON on disk (mirroring
    // `foreign_edge_artifact.rs`'s own `temp_project_root`/`well_formed_
    // artifact` test idiom) — a genuine end-to-end read, not this
    // module's in-process fixture stub. The fact read back is then
    // registered into the fixture stub under the SAME target path this
    // recognizer resolves to, so `foreign_edge_at`'s own recognition and
    // premise logic runs unchanged over a fact that really came off disk.

    use std::fs;

    use refined_kernel::wire_format::wire_set;
    use serde_json::json;

    use crate::refinedpy::foreign_edge_artifact;

    /// A fresh temp directory marked as a project root with `.git`, so
    /// `cache_artifact_path`/`project_root_of` resolve exactly this
    /// directory.
    fn temp_project_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "refinedpy_foreign_edge_test_{label}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&root).expect("create temp project root");
        fs::create_dir_all(root.join(".git")).expect("mark the temp root as a project root");
        root
    }

    /// A well-formed `audioLevel` artifact JSON, with the real sha256 of
    /// `source` as its target contentHash — the exact RULED cases schema
    /// `foreign_edge_artifact.rs`'s own module doc spells, no version
    /// field at all.
    fn well_formed_audio_level_artifact(source: &[u8]) -> serde_json::Value {
        let element = make_refined_set(vec![at_least(-2.0), at_most(2.0)]);
        let return_set = make_refined_set(vec![integer(), at_least(0.0), at_most(1.0)]);
        json!({
            "refined": {"kind": "fact-artifact"},
            "target": {"file": "audio_level.ts", "contentHash": format!("sha256:{}", crate::refinedpy::fact_export::sha256_hex(source))},
            "language": "typescript",
            "runtime": {"band": "es2023+"},
            "surface": {"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": "audioLevel"},
            "functions": {
                "audioLevel": {
                    "entry": [{"name": "boosted", "sequence": {"element": {"cases": [{"sort": "number", "set": wire_set(&element)}]}, "lengthAtLeast": 1}}],
                    "return": {"cases": [{"sort": "number", "set": wire_set(&return_set)}], "stdoutPure": true},
                    "provenance": {"line": 30, "said": "audioLevel's own kernel summary"},
                }
            }
        })
    }

    /// A hand-built artifact really on disk, read through the sibling's
    /// own `read_foreign_ts_artifact`, recognizes end to end and binds
    /// the proved [0, 1] return.
    #[test]
    fn a_disk_backed_artifact_reads_through_the_sibling_reader_and_binds() {
        let Some(kernel) = loaded_kernel() else { return };
        let root = temp_project_root("proved");
        let target = root.join("audio_level.ts");
        fs::write(&target, b"export function audioLevel(boosted: number[]): number { return 0; }\n")
            .expect("write target");
        let source = fs::read(&target).expect("read target back");
        let artifact_path = foreign_edge_artifact::cache_artifact_path(target.to_str().unwrap());
        fs::create_dir_all(artifact_path.parent().unwrap()).expect("create cache dir");
        fs::write(&artifact_path, well_formed_audio_level_artifact(&source).to_string()).expect("write artifact");

        let target_path = target.to_str().unwrap().to_owned();
        let real_artifact =
            foreign_edge_artifact::read_foreign_ts_artifact(&target_path).expect("the disk artifact reads back");
        assert_eq!(real_artifact.called.name, "audioLevel");
        assert!(real_artifact.called.stdout_pure);
        register_fixture_artifact(&target_path, real_artifact);

        let source_body = format!(
            "def audio_level_via_ts(boosted):\n    result = subprocess.run(\n        [\"node\", {target_path:?}],\n        input=json.dumps(boosted),\n        capture_output=True,\n        text=True,\n    )\n    return json.loads(result.stdout)\n"
        );
        let body = def_body(&source_body);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        match foreign_edge_at(&body, 0, &environment, &kernel, None).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
            }
            ForeignEdgeOutcome::Decline { message, .. } => panic!("wanted an override, got a decline: {message}"),
            ForeignEdgeOutcome::Fired { message, .. } => panic!("wanted an override, got a fire: {message}"),
        }

        fs::remove_dir_all(&root).ok();
    }

    /// No artifact on disk: the sibling reader's own missing-artifact
    /// sentence names the export command — pinned here so this module's
    /// own decline path (which just relays whatever the reader answers)
    /// is exercised against the REAL sentence text, not a hand-written
    /// stand-in.
    #[test]
    fn a_missing_disk_artifact_names_the_export_command() {
        let root = temp_project_root("missing");
        let target = root.join("audio_level.ts");
        fs::write(&target, b"export function audioLevel(boosted: number[]): number { return 0; }\n")
            .expect("write target");
        let target_path = target.to_str().unwrap().to_owned();

        let sentence = foreign_edge_artifact::read_foreign_ts_artifact(&target_path)
            .expect_err("no artifact exists and no producer can write one in this temp root");
        assert!(sentence.contains("-export-fact"), "{sentence}");

        fs::remove_dir_all(&root).ok();
    }

    /// This module's own decline (line ~275, production code, `#[cfg(not(test))]`
    /// path) wraps whatever the sibling reader answers with "the target …
    /// states no fact for this edge — {reason}". The sibling's own
    /// missing-artifact sentence must NOT restate that same claim itself,
    /// or the composed sentence carries the phrase twice. Composed here
    /// exactly as the production call site composes it (tests exercise
    /// the module's fixture stub instead of the sibling reader at
    /// `foreign_edge_at`'s own call site, so the composition is repeated
    /// here rather than observed through it) against the REAL sibling
    /// sentence, so a regression in either side's wording is caught.
    #[test]
    fn a_missing_disk_artifact_states_no_fact_exactly_once() {
        let root = temp_project_root("missing_once");
        let target = root.join("audio_level.ts");
        fs::write(&target, b"export function audioLevel(boosted: number[]): number { return 0; }\n")
            .expect("write target");
        let target_path = target.to_str().unwrap().to_owned();

        let reason = foreign_edge_artifact::read_foreign_ts_artifact(&target_path)
            .expect_err("no artifact exists and no producer can write one in this temp root");
        let message = "the target ".to_owned() + &target_path + " states no fact for this edge — " + &reason;

        assert_eq!(
            message.matches("states no fact for this edge").count(),
            1,
            "the prefix must appear exactly once: {message}"
        );
        assert!(message.contains("-export-fact"), "{message}");

        fs::remove_dir_all(&root).ok();
    }
}
