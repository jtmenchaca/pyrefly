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
//! CROSS-LANGUAGE-EDGE.md §2's corollary makes this a real edge and not
//! a manifest: the argv deterministically NAMES the code that runs
//! next, so the checker treats the call the way it treats an import.
//! §11 is this exact spelling; §4 is the JSON transport model both legs
//! apply; §5 is the list of premises the crossing rests on.
//!
//! WHAT THE ROUTE DOES, in order (mirrors the Go twin's own banner):
//!
//!  1. RECOGNIZE the call: an `Assign` of one name from `subprocess.run`,
//!     with a literal `["node", "<script>.ts"]` argv and the three
//!     required keywords (`input=json.dumps(...)`, `capture_output=True`,
//!     `text=True`). Anything unrecognized declines, and every decline
//!     NAMES what broke.
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
//! A JSON number crossing to Python is read as `PrimitiveKind::Float`:
//! `json.loads` always answers a Python `float` for a JSON number
//! (library/json.rst's conversion table — "number (int)" only when the
//! JSON text itself has no fractional/exponent part AND the loader's
//! own `parse_int` is not overridden; this checker does not read that
//! override, so the safe, uniform reading here is Float, mirroring the
//! Go twin's own `foreignReturnValue` comment on the identical premise).

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::refinedpy::diagnostic_sentences;
use crate::refinedpy::env::Environment;
use crate::refinedpy::foreign_edge_artifact::ForeignTsArtifact;
use crate::refinedpy::foreign_edge_artifact::ForeignTsEntry;
#[cfg(test)]
use crate::refinedpy::foreign_edge_artifact::ForeignTsFunctionFact;
#[cfg(not(test))]
use crate::refinedpy::foreign_edge_artifact::read_foreign_ts_artifact as read_foreign_ts_artifact_landed;

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
    /// crosses out.
    pub payload: Expr,
    /// The name the call's result binds, whose sole `json.loads(<name>
    /// .stdout)` consumer receives the return fact.
    pub result_name: String,
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
    let edge = recognize_foreign_edge(statements, index, environment)?;
    let mut edge = match edge {
        Ok(edge) => edge,
        Err(decline) => return Some(ForeignEdgeOutcome::Decline { message: decline.message, range: decline.range }),
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
            return Some(ForeignEdgeOutcome::Decline {
                message: "the target ".to_owned() + &edge.target_path + " states no fact for this edge — " + &reason,
                range: edge.call,
            });
        }
    };
    // the OUTBOUND leg: every premise about what crosses out, discharged
    // against the value the walk holds for it
    if let Some(outcome) = check_outbound_leg(&edge, &artifact, environment, kernel) {
        return Some(outcome);
    }
    // CHANNEL PURITY: the wire is stdout, and the claim assumes stdout
    // carries exactly the serialized result
    if !artifact.called.stdout_pure {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " does not state that it writes "
                + "nothing else to stdout, and this edge reads its result off stdout — "
                + "the channel-purity premise is undischarged",
            range: edge.call,
        });
    }
    // the RETURN leg: the target's own fact, attached to the parse
    match sole_parse_consumer_of(statements, index, &edge.result_name) {
        Ok(parse_range) => Some(ForeignEdgeOutcome::Override {
            parse_range,
            value: foreign_return_value(&artifact),
        }),
        Err(message) => Some(ForeignEdgeOutcome::Decline { message, range: edge.call }),
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
/// set, at the grade the crossing's weakest cited boundary admits.
///
/// `TrustSpec`, mirroring the Go twin's `foreignReturnValue`: the value
/// is not this kernel's own decision about this expression, it is
/// another language's claim carried across a transport whose identity
/// is a CITED PREMISE, not a proved theorem.
fn foreign_return_value(artifact: &ForeignTsArtifact) -> AbstractValue {
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(artifact.called.return_set.clone(), None, TrustSpec, SetKindTag::None)
    }
}

/* ── recognition ──────────────────────────────────────────────────── */

/// A decline the recognizer already knows enough to name — distinct
/// from "not this shape at all," which owes no sentence.
struct RecognitionDecline {
    message: String,
    range: TextRange,
}

/// Reads one statement as `<name> = subprocess.run(["node", "<script>
/// .ts"], input=json.dumps(<payload>), capture_output=True, text=True)`.
///
/// `None` — not this shape at all, no sentence owed. `Some(Err(...))` —
/// this IS a `subprocess.run` call and something about its spelling
/// stopped the resolution, so the caller owes a sentence naming it.
fn recognize_foreign_edge(
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let Stmt::Assign(assign) = &statements[index] else {
        return None;
    };
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
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
    if attribute.attr.as_str() != "run" {
        return None;
    }
    // past this point the reader KNOWS it is looking at a subprocess.run
    // call, so every remaining decline names what stopped it
    let call_range = call.range();
    let [argv] = call.arguments.args.as_ref() else {
        return Some(Err(RecognitionDecline {
            message: "this call runs subprocess.run with other than one positional argv argument, and \
                the checker models only a written argv list naming one script"
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
    let [interpreter, script] = argv_list.elts.as_slice() else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv does not hold exactly [\"node\", \"<script>.ts\"], so the checker \
                cannot name the code that runs next"
                .to_owned(),
            range: call_range,
        }));
    };
    let Some(interpreter_text) = literal_string(interpreter) else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv[0] is not a written string literal naming the interpreter".to_owned(),
            range: call_range,
        }));
    };
    if interpreter_text != "node" {
        // some other program: not a TS edge, nothing owed
        return None;
    }
    let Some(script_text) = literal_string(script) else {
        return Some(Err(RecognitionDecline {
            message: "this call's argv is not one written string naming a script — the checker cannot \
                name the code that runs next, so it models no edge here"
                .to_owned(),
            range: call_range,
        }));
    };
    if !script_text.ends_with(".ts") {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs node on {script_text}, which is not a .ts file — the checker models the \
                edge only where the argv names TypeScript source it can read a fact for"
            ),
            range: call_range,
        }));
    }
    let (payload, keywords_decline) = subprocess_run_keywords_of(call);
    if let Some(decline) = keywords_decline {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(payload) = payload else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs node on {script_text} and sends it no json.dumps(...) input, so \
                nothing crosses out on stdin and the transport model has no outbound leg to apply"
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(script_text),
        payload,
        result_name: target.id.as_str().to_owned(),
    }))
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
fn subprocess_run_keywords_of(call: &ruff_python_ast::ExprCall) -> (Option<Expr>, Option<String>) {
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

/// Reads `json.dumps(<expr>)` and answers the single argument.
fn json_dumps_argument_of(expression: &Expr) -> Option<Expr> {
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
    if let Some((element_set, length_at_least)) = &entry.sequence {
        return check_sequence_crossing(edge, artifact, entry, element_set, *length_at_least, &crossing, kernel);
    }
    if let Some(scalar_set) = &entry.scalar {
        return check_scalar_crossing(edge, artifact, entry, scalar_set, &crossing, kernel);
    }
    Some(ForeignEdgeOutcome::Decline {
        message: "the target ".to_owned() + &artifact.called.name + " states an entry position " + &entry.name
            + " that is neither a sequence nor a scalar set — nothing says whether the value fits",
        range: payload_range,
    })
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
/// inside the stated element set, and the length floor at or above the
/// stated one.
fn check_sequence_crossing(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    entry: &ForeignTsEntry,
    element_set: &RefinedSet,
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
    // the ELEMENT fit — a real kernel ask
    let fits = match foreign_scalar_subset(kernel, &window.element, element_set) {
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

/// Judges a scalar payload against a scalar entry — the same
/// `scalar_subset` ask, without a length to carry.
fn check_scalar_crossing(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    entry: &ForeignTsEntry,
    entry_set: &RefinedSet,
    crossing: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    let payload_range = edge.payload.range();
    let Some(crossing_set) = set_of_known(crossing) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a value at " + &entry.name
                + ", and the value crossing out is not read as a set here — nothing says whether it fits",
            range: payload_range,
        });
    };
    let fits = match foreign_scalar_subset(kernel, &crossing_set, entry_set) {
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

/// Finds the `json.loads(<result_name>.stdout)` node the target's
/// return fact attaches to, scanning the statements AFTER the call in
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
fn sole_parse_consumer_of(statements: &[Stmt], index: usize, result_name: &str) -> Result<TextRange, String> {
    let mut found: Option<TextRange> = None;
    let mut count = 0usize;
    let mut written = false;
    for statement in &statements[index + 1..] {
        if statement_writes_name(statement, result_name) {
            written = true;
        }
        foreign_parse_calls_in(statement, result_name, &mut found, &mut count);
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

/// Counts every `json.loads(<name>.stdout)` in a statement, recording
/// the first — never descending into a nested function, the same
/// boundary the Go twin's `foreignParseCallsIn` keeps.
fn foreign_parse_calls_in(statement: &Stmt, name: &str, found: &mut Option<TextRange>, count: &mut usize) {
    visit_statement_exprs(statement, &mut |expression| {
        if is_foreign_parse_of(expression, name) {
            if found.is_none() {
                *found = Some(expression.range());
            }
            *count += 1;
        }
    });
}

/// Whether a node is exactly `json.loads(<name>.stdout)`.
fn is_foreign_parse_of(expression: &Expr, name: &str) -> bool {
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
    let Expr::Attribute(result_attribute) = argument else {
        return false;
    };
    let Expr::Name(result_name) = result_attribute.value.as_ref() else {
        return false;
    };
    result_name.id.as_str() == name && result_attribute.attr.as_str() == "stdout"
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

    use refined_domain::abstract_value::known_values;
    use refined_domain::abstract_value::possibly_nan;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::at_most;
    use refined_sets::refinement_forms::integer;
    use refined_sets::refinement_forms::make_refined_set;
    use refined_sets::refinement_forms::star;
    use refined_sets::repetition_window_forms::repetition;

    use super::*;

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
                    sequence: Some((make_refined_set(vec![at_least(-2.0), at_most(2.0)]), 1)),
                    scalar: None,
                }],
                return_set: make_refined_set(vec![integer(), at_least(0.0), at_most(1.0)]),
                stdout_pure: true,
                provenance_line: 30,
                provenance_said: "audioLevel's own kernel summary".to_owned(),
            },
            target_file: "./audio_level.ts".to_owned(),
            runtime_band: "node-20+".to_owned(),
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

    #[test]
    fn the_exact_shape_recognizes_and_binds_the_proved_return() {
        register_fixture_artifact("./audio_level.ts", audio_level_ts_artifact());
        let Some(kernel) = loaded_kernel() else { return };
        let body = def_body(FIXTURE_SOURCE);
        let environment = env_with(&[("boosted", boosted_sequence_value())]);
        let outcome = foreign_edge_at(&body, 0, &environment, &kernel).expect("the shape recognizes");
        match outcome {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
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
            foreign_edge_at(&body, 0, &environment, &kernel).is_none(),
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
        match foreign_edge_at(&body, 0, &environment, &kernel).expect("the call is still recognized as subprocess.run") {
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
        match foreign_edge_at(&body, 0, &environment, &kernel).expect("the shape recognizes") {
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
        match foreign_edge_at(&body, 0, &environment, &kernel).expect("the shape recognizes") {
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
        match foreign_edge_at(&body, 0, &environment, &kernel).expect("the shape recognizes") {
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
        match foreign_edge_at(&body, 0, &environment, &kernel).expect("the shape recognizes") {
            ForeignEdgeOutcome::Fired { message, .. } => {
                assert!(message.contains("NaN"), "{message}");
            }
            ForeignEdgeOutcome::Override { .. } => panic!("wanted a NaN-freedom fire, got an override"),
            ForeignEdgeOutcome::Decline { message, .. } => {
                panic!("wanted a NaN-freedom fire, got a decline: {message}")
            }
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
    /// `source` as its target contentHash — the exact schema
    /// `foreign_edge_artifact.rs`'s own module doc spells.
    fn well_formed_audio_level_artifact(source: &[u8]) -> serde_json::Value {
        let element = make_refined_set(vec![at_least(-2.0), at_most(2.0)]);
        let return_set = make_refined_set(vec![integer(), at_least(0.0), at_most(1.0)]);
        json!({
            "refined": {"kind": "typescript-fact-artifact", "version": 1},
            "target": {"file": "audio_level.ts", "contentHash": format!("sha256:{}", crate::refinedpy::fact_export::sha256_hex(source))},
            "runtime": {"band": "node-23+"},
            "harness": {"stdin": "json", "stdout": "json", "calls": "audioLevel"},
            "functions": {
                "audioLevel": {
                    "entry": [{"name": "boosted", "sequence": {"element": wire_set(&element), "lengthAtLeast": 1}}],
                    "return": {"set": wire_set(&return_set), "stdoutPure": true},
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
        match foreign_edge_at(&body, 0, &environment, &kernel).expect("the shape recognizes") {
            ForeignEdgeOutcome::Override { value, .. } => {
                assert_eq!(value.kind, Kind::Set);
                assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
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
}
