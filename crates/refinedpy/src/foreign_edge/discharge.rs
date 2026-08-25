//! The post-recognition premises every recognized edge shares: resolve
//! a relative target path, read the target's own artifact, check the
//! carrier identity, discharge the outbound leg, check channel purity,
//! and scan for the return leg's sole consumer — the shared finish both
//! `foreign_edge_at`'s and `foreign_edge_at_walrus_call`'s own thin
//! wrappers call.

use std::sync::Arc;

use ruff_python_ast::Stmt;

use refined_kernel::kernel_interface::RefinedTSKernel;

use crate::diagnostic_sentences;
use crate::env::Environment;
use crate::foreign_edge_artifact::ForeignSurface;
use crate::foreign_edge_artifact::ForeignTsArtifact;
use crate::foreign_edge_artifact::read_compiled_binary_fact;
#[cfg(not(test))]
use crate::foreign_edge_artifact::read_foreign_ts_artifact as read_foreign_ts_artifact_landed;

use super::argv::RecognitionDecline;
use super::argv::Runner;
use super::cases::foreign_return_value_or_undetermined;
use super::cases::foreign_stdout_serialized_value;
use super::crossing::check_outbound_leg;
use super::parse_consumer::ParseConsumer;
use super::parse_consumer::foreign_parse_argument_range_of;
use super::parse_consumer::sole_parse_consumer_from;
use super::parse_consumer::sole_parse_consumer_of;
use super::recognize::os_system_return_read_of;
use super::Channel;
use super::ForeignEdge;
use super::ForeignEdgeOutcome;
use super::ResultRead;

/// `discharge_edge_premises`'s own error: a `Decline` never had a
/// consumer scan to run (the edge itself is not returned, since no
/// caller needs it — a decline names a premise no later statement's
/// walk can complete around); a `Fired` DOES need both `edge` AND
/// `artifact` back — the caller (which alone holds `statements`) finds
/// the return leg's sole consumer through `edge`, then builds that
/// consumer's own bound value through `artifact`, the SAME two inputs
/// `return_leg_outcome`'s own `Override` arm already needs for the
/// identical build.
enum EdgeDischargeError {
    Fired { outcome: ForeignEdgeOutcome, edge: ForeignEdge, artifact: ForeignTsArtifact },
    Decline(ForeignEdgeOutcome),
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
) -> Result<(ForeignEdge, ForeignTsArtifact), EdgeDischargeError> {
    let mut edge = match edge {
        Ok(edge) => edge,
        Err(decline) => {
            return Err(EdgeDischargeError::Decline(ForeignEdgeOutcome::Decline {
                message: decline.message,
                range: decline.range,
            }));
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
    let artifact = if edge.runner == Runner::CompiledBinary {
        // A compiled binary reads its fact from a SIBLING file
        // (`<binary_path>.facts.json`), never the TypeScript reader's
        // project-cache path — a compiled binary has no `.refined/cache/`
        // entry any producer in this checker writes.
        match read_compiled_binary_fact(&edge.target_path) {
            Ok(artifact) => artifact,
            Err(reason) => {
                let sibling_exists = crate::foreign_edge_artifact::compiled_binary_fact_path(&edge.target_path).exists();
                let message = if sibling_exists {
                    // The sibling file exists but failed to read as a
                    // fact — name the unreadable file, not the generic
                    // "no producer" sentence, which is only true when
                    // there is no fact file at all.
                    reason
                } else {
                    // The artifact reader's own sentence names a missing
                    // TypeScript-fact file and the `-export-fact` command
                    // that writes one — neither applies here: the target
                    // is a compiled binary, and no producer in this
                    // checker exports a fact for one. Name the construct
                    // that actually blocks, not the generic no-fact
                    // sentence.
                    diagnostic_sentences::compiled_binary_no_fact(&edge.target_path)
                };
                return Err(EdgeDischargeError::Decline(ForeignEdgeOutcome::Decline { message, range: edge.call }));
            }
        }
    } else {
        match read_foreign_ts_artifact(&edge.target_path) {
            Ok(artifact) => artifact,
            Err(reason) => {
                let message = "the target ".to_owned() + &edge.target_path + " states no fact for this edge — " + &reason;
                return Err(EdgeDischargeError::Decline(ForeignEdgeOutcome::Decline { message, range: edge.call }));
            }
        }
    };
    // RUNTIME IDENTITY: the artifact's own band names a spec level, not
    // one runtime binary (ruling, 2026-08-21) — the reader that produced
    // `artifact` already checked its band against the pin for its own
    // language (`es2023+` for a JS runner row, `c++17` for a compiled-
    // binary row), so every recognized row discharges this premise
    // identically once that check passed.
    //
    // CARRIER IDENTITY: the call's own spelling states one channel, and
    // the target's surface states the one it actually reads — a JSON
    // transport model applies only when both name the SAME carrier.
    if let Some(mismatch) = channel_mismatch_decline(edge.channel, &artifact.surface) {
        return Err(EdgeDischargeError::Decline(ForeignEdgeOutcome::Decline { message: mismatch, range: edge.call }));
    }
    // the OUTBOUND leg: every premise about what crosses out, discharged
    // against the value the walk holds for it — a `Fired` outcome here
    // carries `edge` AND `artifact` back out too, so the caller can
    // still find the return leg's sole consumer and bind its own real
    // fact under the fire, the same two inputs the green path needs.
    if let Some(outcome) = check_outbound_leg(&edge, &artifact, environment, kernel) {
        return Err(match outcome {
            ForeignEdgeOutcome::Fired { .. } => EdgeDischargeError::Fired { outcome, edge, artifact },
            other => EdgeDischargeError::Decline(other),
        });
    }
    // CHANNEL PURITY: the wire is stdout, and the claim assumes stdout
    // carries exactly the serialized result
    if !artifact.called.stdout_pure {
        return Err(EdgeDischargeError::Decline(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " does not state that it writes "
                + "nothing else to stdout, and this edge reads its result off stdout — "
                + "the channel-purity premise is undischarged",
            range: edge.call,
        }));
    }
    Ok((edge, artifact))
}


/// Answers the return leg's own outcome once `sole_parse_consumer_of`
/// (or its inclusive-scan sibling) has already run: the target's own
/// fact, attached to the parse — unless the declared return admits a
/// corner the target's own `JSON.stringify` serializes as `null`, which
/// degrades to a named undetermined instead of binding the set as
/// stated.
///
/// A call whose result NOTHING reads through `json.loads` (`ParseConsumer::
/// NoneFound`) answers plain `None` here, never a decline: the outbound
/// leg already discharged cleanly (every premise up through `discharge_
/// edge_premises` ran before this function is ever called), so there is a
/// real, judged crossing — the return leg simply has no expression for
/// the target's return fact to attach to, which is not itself a blocked
/// construct (`d-data-legs.py`'s own `level_via_raw_stdout` row: the
/// value is read as `float(result.stdout)`, never parsed as JSON at all,
/// and that read is free to judge on its own ordinary terms).
///
/// `consumer_scan_statements` is the SAME statement list the caller's
/// own `consumer` scan already ran over — passed again (rather than
/// re-sliced here) so `stdout_override`'s own argument-range scan
/// (`foreign_parse_argument_range_of`) searches the identical range and
/// finds the identical node `consumer`'s own `parse_range` names.
fn return_leg_outcome(
    consumer: ParseConsumer,
    artifact: &ForeignTsArtifact,
    edge: &ForeignEdge,
    consumer_scan_statements: &[Stmt],
) -> Option<ForeignEdgeOutcome> {
    match consumer {
        ParseConsumer::Found(parse_range) => Some(match foreign_return_value_or_undetermined(artifact) {
            Ok(value) => {
                // The intermediate captured-stdout reading's own SECOND
                // override — bound ONLY on a DISCHARGED crossing (this
                // arm is reached exclusively from the green
                // `discharge_edge_premises` path, never `Fired`) and
                // ONLY when every return case is number-sorted
                // (`foreign_stdout_serialized_value`'s own gate). `os
                // .system`'s `ResultRead::FileRead` has no intermediate
                // stdout binding at all — `foreign_parse_argument_range_of`
                // answers `None` for it (`foreign_parse_argument_range`'s
                // own `FileRead` arm), so this stays `None` there
                // unconditionally, leaving that shape's existing
                // return-fact consumer override as its only publish.
                let stdout_override = foreign_stdout_serialized_value(&artifact.called.return_cases).and_then(
                    |serialized| {
                        foreign_parse_argument_range_of(consumer_scan_statements, &edge.result_name, &edge.result_read)
                            .map(|argument_range| (argument_range, serialized))
                    },
                );
                ForeignEdgeOutcome::Override { parse_range, value, stdout_override }
            }
            Err(message) => ForeignEdgeOutcome::Decline { message, range: parse_range },
        }),
        ParseConsumer::NoneFound => None,
        ParseConsumer::Blocked(message) => Some(ForeignEdgeOutcome::Decline { message, range: edge.call }),
    }
}


/// The return leg's sole-consumer scan, run over the statements AFTER
/// `edge.consumer_scan_from` — `os_system_return_read_of`'s literal-
/// outfile scan for `os.system`'s own `ResultRead::FileRead` shape
/// (which has no bound name to scan for at all: the call's own captured
/// target is the process's exit status, never the crossing's value), or
/// `sole_parse_consumer_of`'s bound-name scan for every ordinary shape.
fn scan_sole_consumer(statements: &[Stmt], edge: &ForeignEdge) -> ParseConsumer {
    match &edge.result_read {
        ResultRead::FileRead { outfile } => match os_system_return_read_of(statements, edge.consumer_scan_from, outfile) {
            Some(parse_range) => ParseConsumer::Found(parse_range),
            None => ParseConsumer::Blocked(diagnostic_sentences::os_system_missing_return_read(outfile)),
        },
        _ => sole_parse_consumer_of(statements, edge.consumer_scan_from, &edge.result_name, &edge.result_read),
    }
}


/// Fills a `Fired` outcome's `consumer` field from the scan the caller
/// already ran (the SAME scan the green path reuses), building the
/// consumer's own bound value through `foreign_return_value_or_
/// undetermined(artifact)` — the IDENTICAL call `return_leg_outcome`'s
/// own `Override` arm makes, never re-derived. The found range and
/// value survive into `consumer` ONLY when `edge.result_read` is
/// `ResultRead::FileRead` (`os.system`'s own `json.load(<handle>)` read,
/// a shape `expressions.rs` never models at all: binding the real fact
/// there adds a determination, never removes one). Every OTHER
/// `result_read` shape (`StdoutAttribute`/`Bare` — an ordinary `json.
/// loads(...)` call) has its own real fallback: left unbound, that node
/// reaches `expressions.rs`'s `json.loads`-of-an-untracked-operand
/// model, whose `None` arm the return's declared type genuinely
/// refuses — a second, DETERMINED fire this function must never
/// replace with a narrower bound value (`ForeignEdgeOutcome::Fired`'s
/// own doc names the four corpus rows this exact confusion broke). A
/// `FileRead` scan that comes back `NoneFound`/`Blocked`, or whose own
/// return value itself degrades to a named undetermined (`Err` from
/// `foreign_return_value_or_undetermined`), leaves `consumer` `None` —
/// nothing real to bind. A `StdoutAttribute`/`Bare` scan always leaves
/// `consumer` `None` regardless of its own outcome — the gate is on the
/// SHAPE, never on whether a consumer or a value happens to exist.
fn attach_consumer_to_fire(
    outcome: ForeignEdgeOutcome,
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    consumer: ParseConsumer,
) -> ForeignEdgeOutcome {
    match outcome {
        ForeignEdgeOutcome::Fired { message, range, .. } => {
            let consumer = match (&edge.result_read, consumer) {
                (ResultRead::FileRead { .. }, ParseConsumer::Found(parse_range)) => {
                    foreign_return_value_or_undetermined(artifact).ok().map(|value| (parse_range, value))
                }
                _ => None,
            };
            ForeignEdgeOutcome::Fired { message, range, consumer }
        }
        other => other,
    }
}


/// `discharge_edge_premises` plus the return-leg scan — the
/// `Stmt::Assign`/`Stmt::With` callers' own finish, unchanged from
/// before the walrus entry point existed except for this one dispatch.
/// A `Fired` outbound leg still runs this SAME scan (`scan_sole_
/// consumer`) before answering — `attach_consumer_to_fire` then binds
/// the found consumer to its own real return-leg value ONLY for
/// `os.system`'s own `FileRead` shape, whose consumer node has no
/// fallback fact of its own; every other shape's found consumer is left
/// unbound, since THAT node's own unbound walk is where the union-
/// `None`-arm fire this edge's outbound refusal used to trail still
/// needs to run.
pub(super) fn finish_recognized_edge(
    edge: Result<ForeignEdge, RecognitionDecline>,
    statements: &[Stmt],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    let (edge, artifact) = match discharge_edge_premises(edge, environment, kernel, entry_directory) {
        Ok(discharged) => discharged,
        Err(EdgeDischargeError::Fired { outcome, edge, artifact }) => {
            let consumer = scan_sole_consumer(statements, &edge);
            return Some(attach_consumer_to_fire(outcome, &edge, &artifact, consumer));
        }
        Err(EdgeDischargeError::Decline(outcome)) => return Some(outcome),
    };
    let consumer = scan_sole_consumer(statements, &edge);
    // `scan_sole_consumer`'s own non-`FileRead` arm scans
    // `statements[edge.consumer_scan_from + 1..]` (`sole_parse_consumer_of`'s
    // doc) — sliced identically here so `return_leg_outcome`'s own
    // argument-range scan searches the same range.
    return_leg_outcome(consumer, &artifact, &edge, &statements[edge.consumer_scan_from + 1..])
}


/// `discharge_edge_premises` plus the INCLUSIVE return-leg scan
/// (`sole_parse_consumer_from`, over the whole of `statements` — no call
/// statement to skip past) — `foreign_edge_at_walrus_call`'s own finish,
/// since its recognized call sits inside the `if` TEST rather than as a
/// member of `statements` at all. `foreign_edge_at_walrus_call` only
/// ever reaches `recognize_subprocess_callee` (never `recognize_os_
/// system`), so a `Fired` outbound leg here can never carry `ResultRead
/// ::FileRead` — `attach_consumer_to_fire` always answers `consumer:
/// None` on this path, the same as `finish_recognized_edge`'s own
/// non-`FileRead` shapes, for the identical reason: the found consumer
/// belongs to an ordinary `json.loads(...)` node whose own union-`None`
/// -arm fire must still run unbound.
pub(super) fn finish_recognized_edge_from_start(
    edge: Result<ForeignEdge, RecognitionDecline>,
    statements: &[Stmt],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    entry_directory: Option<&std::path::Path>,
) -> Option<ForeignEdgeOutcome> {
    let (edge, artifact) = match discharge_edge_premises(edge, environment, kernel, entry_directory) {
        Ok(discharged) => discharged,
        Err(EdgeDischargeError::Fired { outcome, edge, artifact }) => {
            let consumer = sole_parse_consumer_from(statements, &edge.result_name, &edge.result_read);
            return Some(attach_consumer_to_fire(outcome, &edge, &artifact, consumer));
        }
        Err(EdgeDischargeError::Decline(outcome)) => return Some(outcome),
    };
    let consumer = sole_parse_consumer_from(statements, &edge.result_name, &edge.result_read);
    // `finish_recognized_edge_from_start`'s own INCLUSIVE scan — the
    // whole `statements` slice, unsliced, matching `sole_parse_consumer_
    // from`'s own call just above.
    return_leg_outcome(consumer, &artifact, &edge, statements)
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
    super::tests::test_read_foreign_ts_artifact(target_path)
}


#[cfg(not(test))]
fn read_foreign_ts_artifact(target_path: &str) -> Result<ForeignTsArtifact, String> {
    read_foreign_ts_artifact_landed(target_path)
}
