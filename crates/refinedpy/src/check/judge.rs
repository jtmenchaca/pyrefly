use std::collections::{HashMap, HashSet};

use refined_domain::abstract_value::{known_set, AbstractValue, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::{on_one_tuple_layer, requires_integer};
use ruff_python_ast::{Expr, StmtReturn};
use ruff_text_size::{Ranged, TextRange};

use crate::assignability::{judge, states_sequence, Verdict};
use crate::env::Environment;
use crate::typereading::DeclaredRefinement;

use super::*;

/// `return value` against the enclosing function's own `-> Annotation`.
/// No annotation (`return_refinement` is `None`) means ordinary Python
/// — nothing judges, matching the mission's "no return annotation → no
/// judging." A bare `return`/`return None` carries no value expression
/// and judges nothing either. `Verdict::Fire` records an RTS7001 at the
/// returned expression's own range; `Undetermined` becomes this body's
/// blocker candidate (never overriding an earlier blocker — the FIRST
/// blocker wins, same as every other sink).
pub(super) fn walk_return(
    ret: &StmtReturn,
    return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    provably_unbound: &HashSet<String>,
    environment: &mut Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let Some(value_expr) = ret.value.as_deref() else {
        return;
    };
    // PROVABLY-UNBOUND READS: `return x` where `x` is STILL in
    // `provably_unbound` (a valueless AnnAssign declared it, and no
    // straight-line write since has cured it — walk_statement clears the
    // whole set the moment a branch/loop/blocker could have bound it on
    // some other path) is CPython's own UnboundLocalError at this exact
    // read (executionmodel.rst's local-variable scoping rule). Checked
    // BEFORE the ordinary sink/judge path — `environment.read` already
    // answers `None` for this name (nothing ever bound it), which would
    // otherwise fall through to a silent Undetermined rather than naming
    // the provable raise.
    if let Expr::Name(name) = value_expr {
        if provably_unbound.contains(name.id.as_str()) && environment.read(name.id.as_str()).is_none() {
            out.push(Finding {
                range: value_expr.range(),
                code: "RTS7001",
                message: format!(
                    "this read provably raises UnboundLocalError: '{}' is unbound at this point",
                    name.id.as_str()
                ),
            });
            return;
        }
    }
    bind_walrus_targets(value_expr, context, aug_assign_refinements, environment, out);
    let Some(value) = sink_value(value_expr, context, environment, aug_assign_refinements, out) else {
        // a provable raise already pushed its own RTS7001 at the
        // raising expression — this return never produces a value to
        // judge, since CPython never reaches the return statement's own
        // completion on this path.
        return;
    };
    // The value this return produces, handed to whoever asked this walk
    // to collect them (`env::collect_returned_values` — `fact_export`'s
    // own derivation seam). A no-op for every ordinary walk, and the
    // value is exactly the one judged below: the export never runs a
    // second, differently-derived reading of the same return.
    environment.record_returned_value(value.clone());
    let Some(declared) = return_refinement else {
        return;
    };
    match judge(&value, declared, context.kernel) {
        Verdict::Fire(message) => out.push(Finding {
            range: value_expr.range(),
            code: "RTS7001",
            message,
        }),
        Verdict::Silent => {}
        Verdict::Undetermined(sentence) => {
            let sentence = name_unmodeled_call_sentence(sentence, Some(value_expr), Some(&value), environment);
            record_blocker(blocked, value_expr.range(), sentence, out);
        }
    }
}

/// `yield value` / bare `yield` / `yield from value`, against the
/// enclosing generator's own YIELD position (`Generator[Y, S, R]`'s
/// first element, `AsyncGenerator[Y, S]`'s/`Iterator[Y]`'s/
/// `Iterable[Y]`'s only element — `typereading.rs::GeneratorRefinement`,
/// threaded down as `yield_refinement` by `generator_body_refinements`).
/// No declared yield position (`yield_refinement` is `None` — an
/// ordinary, non-generator-annotated body, or a generator body whose own
/// `-> Annotation` did not read as one of the four generator forms)
/// means nothing judges here, the mission's "no annotation → no
/// judging" rule applied to this checked position instead of `return`'s.
/// A BARE `yield` (`Expr::Yield` with no operand) yields `None`
/// (datamodel.rst's generator-iterator protocol: `next()` on a bare
/// `yield` hands back `None`) — judged as `Kind::Null` against the
/// declared yield set exactly like any other absent value, so a
/// non-`Optional` yield type still fires on it. `yield from <expr>`
/// DELEGATES: every value the inner generator yields flows out of this
/// generator too, so EACH ONE judges against this generator's own
/// declared yield set (`delegated_generator_yields`'s own two-reading
/// doc: the callee's actual body-walked yields where they read, its
/// declared annotation's yield set otherwise) — the first Fire wins,
/// matching `judge`'s own dict/list element-law convention of reporting
/// the first escaping member rather than joining every member's verdict.
pub(super) fn walk_yield(
    yield_expr: &Expr,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let Some(declared) = yield_refinement else {
        return;
    };
    match yield_expr {
        Expr::Yield(yield_node) => {
            let range = yield_node.range();
            let (value, source_expr) = match yield_node.value.as_deref() {
                Some(value_expr) => {
                    bind_walrus_targets(value_expr, context, aug_assign_refinements, environment, out);
                    let Some(value) = sink_value(value_expr, context, environment, aug_assign_refinements, out) else {
                        // a provable raise already pushed its own RTS7001 —
                        // this yield never produces a value to judge.
                        return;
                    };
                    (value, Some(value_expr))
                }
                // bare `yield` — the generator hands back None here.
                None => (refined_domain::abstract_value::null_value(), None),
            };
            judge_at(&value, declared, range, source_expr, context, environment, blocked, out);
        }
        Expr::YieldFrom(yield_from) => {
            let range = yield_from.range();
            let Some(elements) = delegated_generator_yields(yield_from.value.as_ref(), context, environment) else {
                record_blocker(
                    blocked,
                    range,
                    "this yield from's own delegate does not yet state a readable yield set".to_owned(),
                    out,
                );
                return;
            };
            for element in &elements {
                judge_at(element, declared, range, None, context, environment, blocked, out);
                // the first Fire this loop pushes is the row's own
                // verdict — later elements still walk (so a LATER
                // element's own Undetermined can still set the body's
                // blocker when no earlier element fired), but a second
                // Fire at the same range would only restate the same
                // row twice, so this loop does not stop early; judge_at
                // itself never double-reports past `blocked` for the
                // Undetermined branch, and a Fire is idempotent to
                // report once per offending element in the rare case
                // more than one escapes (matching the dict/list element
                // law's own "first Fire" framing loosely, since a
                // delegate's own elements are not individually
                // addressable the way a dict key/list index is).
            }
        }
        _ => {}
    }
}

/// Judges one value at `range` against `declared`, pushing a Fire or
/// recording the body's blocker candidate — `walk_return`'s own
/// Fire/Silent/Undetermined tail, factored out so `walk_yield`'s two
/// call shapes (a plain yield's one value, a delegation's several) share
/// it instead of repeating the match. `source_expr` is the single
/// expression this `value` was read from, when one exists (a plain
/// `yield value_expr`'s own operand) — `None` for a `yield from`
/// delegation's own per-element judging, where no single source
/// expression names any one element. Passed through to
/// `name_unmodeled_call_sentence` so a plain `yield torch.arange(5)`
/// under a declared yield type names the module the same way a return
/// or an assignment does.
pub(super) fn judge_at(
    value: &AbstractValue,
    declared: &DeclaredRefinement,
    range: TextRange,
    source_expr: Option<&Expr>,
    context: &WalkContext,
    environment: &Environment,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    match judge(value, declared, context.kernel) {
        Verdict::Fire(message) => out.push(Finding { range, code: "RTS7001", message }),
        Verdict::Silent => {}
        Verdict::Undetermined(sentence) => {
            let sentence = name_unmodeled_call_sentence(sentence, source_expr, Some(value), environment);
            record_blocker(blocked, range, sentence, out);
        }
    }
}

/// The refused-write law, shared by every write sink that can fire
/// (AnnAssign, Assign, AugAssign): judges `value` against `declared`,
/// pushes a Fire finding at `fire_range` when it fires, and binds
/// `name` in `environment` according to the verdict. A Fire does NOT
/// bind the refused value — the write is refused, so the slot keeps
/// its DECLARED SET (`known_set`, TrustSpec — the same construction
/// `seed_parameters` uses) for onward flow: `a = 200` under `a: Age`
/// fires once, here, and a later `return a` under `-> Age` reads the
/// declared set, which is silent against itself (set ⊆ set), never a
/// second fire for the same refused write. Silent binds the evaluated
/// value as today; Undetermined forgets (a stale fact must not survive
/// an unjudged write) and is returned so the caller may adopt it as
/// this body's blocker.
pub(super) fn judge_and_bind(
    name: &str,
    value: AbstractValue,
    declared: &DeclaredRefinement,
    fire_range: TextRange,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> Option<String> {
    judge_and_bind_naming(name, value, declared, fire_range, None, context, environment, out)
}

/// The `.source` tag `expressions::evaluate_call`'s own generator-call
/// arm sets on the value a `def`-recognized generator's call answers
/// when `instances::generator_yields` declines to summarize its body
/// (a conditional `yield`, or any other shape outside the straight-line
/// reading that function's own doc describes) — the DECLINE twin of the
/// `"generator"` tag the SUCCESS path already sets
/// (`collection_models::list_literal_value`'s result, `value.source =
/// "generator"`). Read here, not set: the tag's producer lives in
/// `expressions.rs`/`builtin_models.rs`, outside this file's own scope
/// (see `generator_declined_sentence`'s own doc for the exact handoff).
pub(super) const GENERATOR_DECLINED_SOURCE_TAG: &str = "generator-declined";

/// `judge_and_bind`'s own body, plus the ONE naming step
/// `python-c-extension-boundary.md`'s naming unit adds: when the verdict
/// is the GENERIC undetermined sentence (`SENTENCE.value_not_readable` —
/// `assignability.rs::judge`'s own catch-all, which carries no construct
/// name at all) AND `source_expr` is the exact RHS expression this value
/// came from, a call on an attribute chain rooted at an imported-but-
/// unmodeled module renames the sentence to name that module
/// (`expressions::unmodeled_module_call_name`'s own recognition — the
/// SAME gate `evaluate_attribute_call`'s own module arms already apply,
/// so this never fires for a module the walk actually modeled). Every
/// OTHER undetermined sentence (a kernel refusal, a TypedDict/tuple
/// position, a loop-stabilization blocker, …) already names its own
/// construct and passes through unchanged — this step only ever
/// SHARPENS the one anonymous case, never rewrites a sentence that
/// already has a name.
///
/// Split from `judge_and_bind` itself (rather than adding the parameter
/// there directly) so the many call sites with no single RHS expression
/// to offer (a destructured element, a same-module call's own mutation
/// effect) keep calling the original signature unchanged, and only the
/// sites that DO have the RHS in scope opt into naming.
pub(super) fn judge_and_bind_naming(
    name: &str,
    value: AbstractValue,
    declared: &DeclaredRefinement,
    fire_range: TextRange,
    source_expr: Option<&Expr>,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> Option<String> {
    match judge(&value, declared, context.kernel) {
        Verdict::Fire(message) => {
            out.push(Finding {
                range: fire_range,
                code: "RTS7001",
                message,
            });
            // Tags the numeric sort onward flow needs (the same guarded
            // rule `seed_parameters` applies to a declared set:
            // numeric-ground only, never the `Literal["A", "B"]`
            // string-tuple pun `on_one_tuple_layer` alone would also
            // admit).
            let refused_slot = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
                let sort = if requires_integer(&declared.set) {
                    PrimitiveKind::Integer
                } else {
                    PrimitiveKind::Float
                };
                AbstractValue {
                    kind_tag: Some(sort),
                    ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
                }
            } else {
                known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
            };
            environment.bind(name, refused_slot);
            None
        }
        Verdict::Silent => {
            environment.bind(name, value);
            None
        }
        Verdict::Undetermined(sentence) => {
            let sentence = name_unmodeled_call_sentence(sentence, source_expr, Some(&value), environment);
            environment.forget(name);
            Some(sentence)
        }
    }
}

/// The one naming step `judge_and_bind_naming` applies: `sentence`
/// unchanged UNLESS it is the exact generic `value_not_readable` wording
/// AND either `value` or `source_expr` names something this file
/// recognizes — in which case a narrower sentence replaces it. Three
/// rungs, tried in order (`python-c-extension-boundary.md`'s recognition
/// ladder, plus the generator-body rung q-decline-names.py's own
/// `generator_body_never_summarized` row teaches):
///
/// 0. THE GENERATOR RUNG — `value`'s own `.source` carries
///    `GENERATOR_DECLINED_SOURCE_TAG`: the undetermined value traces back
///    to a same-module generator's call whose body `instances::
///    generator_yields` declined to summarize. Checked first and against
///    `value` directly rather than `source_expr`'s own syntax, because the
///    blocked read is usually NOT the call itself — `first = next(it);
///    return first` blocks at the bare-Name `return first`, two
///    statements downstream of the actual `age_generator()` call — and the
///    tag survives exactly that far because `walk_assign`'s untyped-target
///    arm binds an evaluated RHS value verbatim (`environment.bind(name,
///    value.clone())`, no declared refinement to judge or forget against).
///    NOT YET WIRED at its producer: `expressions::evaluate_call`'s own
///    generator-call arm still answers a bare `unknown()` (expressions.rs,
///    the `is_generator_def(def)` arm's `None => unknown()` row) rather
///    than a value carrying this tag — this rung is written against the
///    tag name in advance of that producer landing (see this file's own
///    module doc / the handoff this unit reports).
/// 1. RUNG 2 — a call on a manifested module's own LISTED function
///    (`binding_manifest::discover_manifest` finds a manifest AND it
///    names an entry for the called function): the manifest states the
///    entry, never the return, so the sentence names the missing
///    producer (`diagnostic_sentences::manifest_entry_names_no_producer`).
///    A manifest that exists but names NO entry for this function is a
///    narrower named decline too
///    (`diagnostic_sentences::manifest_names_no_entry_for`) — still more
///    specific than rung 1's plain "no model at all," since the module
///    itself IS modeled in part.
/// 2. RUNG 1 — every other unmodeled-module call
///    (`expressions::unmodeled_module_call_name`'s own recognition).
///
/// Factored out so the several direct `judge(...)`/`judge_at` call sites
/// that also have a source expression and the judged value in scope can
/// apply the identical rule without duplicating it.
pub(super) fn name_unmodeled_call_sentence(
    sentence: String,
    source_expr: Option<&Expr>,
    value: Option<&AbstractValue>,
    environment: &Environment,
) -> String {
    if sentence != crate::diagnostic_sentences::SENTENCE.value_not_readable {
        return sentence;
    }
    if let Some(value) = value {
        if value.source == GENERATOR_DECLINED_SOURCE_TAG {
            return crate::diagnostic_sentences::generator_body_never_summarized();
        }
    }
    let Some(source_expr) = source_expr else {
        return sentence;
    };
    let Expr::Call(call) = source_expr else {
        return sentence;
    };
    if let Some(named) = manifest_named_sentence(call, environment) {
        return named;
    }
    match crate::expressions::unmodeled_module_call_name(call, environment) {
        Some(module_name) => crate::diagnostic_sentences::unmodeled_module_call(module_name),
        None => sentence,
    }
}

/// Rung 2's own naming: `call` on a bare-Name-rooted attribute chain
/// whose root reads unbound AND has a discovered, readable manifest —
/// `Some` names either the missing entry (the function is not one of
/// the manifest's own listed rows) or the missing producer (the function
/// IS listed, so the entry judged the call's arguments already, and the
/// only remaining gap is the return). `None` for every other shape: no
/// manifest discovered at all (rung 1 owns it), or a manifest that could
/// not be read (this naming step never reports a manifest's own parse
/// failure — that is `discover_manifest`'s own `Err` the export/CLI path
/// surfaces, not a per-call sentence).
pub(super) fn manifest_named_sentence(call: &ruff_python_ast::ExprCall, environment: &Environment) -> Option<String> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if environment.read(module_name.id.as_str()).is_some() {
        return None;
    }
    let entry_directory = environment.entry_directory().map(|path| path.as_path());
    let manifest = crate::binding_manifest::discover_manifest(module_name.id.as_str(), entry_directory)?.ok()?;
    let function_name = attribute.attr.as_str();
    match manifest.entries.get(function_name) {
        Some(entry) => Some(crate::diagnostic_sentences::manifest_entry_names_no_producer(
            module_name.id.as_str(),
            function_name,
            &entry.producer_symbol,
        )),
        None => Some(crate::diagnostic_sentences::manifest_names_no_entry_for(module_name.id.as_str(), function_name)),
    }
}
