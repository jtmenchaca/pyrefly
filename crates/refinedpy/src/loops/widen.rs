/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::kernel_seam::ask_bounds_public;
use refined_domain::trust_grades::min_trust_level;
use refined_domain::trust_grades::trust_level_of;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::refinement_forms::Refinement;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use crate::env::Environment;

use super::JudgeContext;
use super::bind_target::bind_for_target;
use super::bind_target::target_names;
use super::bind_target::written_names;
use super::body_once::run_body_once;

/// Whether `narrower.set` is provably contained in `wider.set` — the
/// question `stabilized_join` asks when the structural rejoin test
/// cannot answer stability for a `Kind::Set` pair, because `join_known`'s
/// general set-combining path has NO STRUCTURAL FIXPOINT: it always
/// wraps both operand sets in a fresh, unreduced `union(...)` node
/// (`lattice_operations.rs`'s fallback), so `join(J, second_pass)`
/// re-wraps rather than converging back to `J`'s own shape even when the
/// second pass denotes nothing new — a raw element set and that same set
/// folded one layer deeper through a prior `union` never compare equal
/// under `RefinedSet`'s derived `PartialEq`, no matter how many times the
/// rejoin runs. Stability under repetition means exactly that the second
/// pass's set is already covered by the first join's set (join only
/// grows, so "the rejoin adds nothing" and "the second pass's set ⊆ the
/// first join's set" are the same claim) — a question the KERNEL decides
/// on the actual admitted values, not on either side's syntactic form.
///
/// `kernel.scalar_subset` is tried first — the ordinary 1-tuple-layer
/// question, covering the two-passes-of-a-numeric-set case both
/// `g_iter_bind`/`g_iter_mul` are — then `kernel.seq_subset` when
/// `scalar_subset` refuses (a sequence-shaped set the scalar decider
/// cannot read; `assignability.rs`'s own containment law tries the same
/// two asks, ordered by which shape is more likely, with the same
/// fallback-on-refusal posture). Both asks panic inside the kernel
/// closure on a refusal — the crate's established `catch_unwind`/
/// `AssertUnwindSafe` idiom (`assignability.rs`, `lattice_conformance.rs`)
/// catches that and reads it as "no proof," never a crash. `true` from
/// either ask is a theorem; `false`, or a refusal from both, is not a
/// disproof — it is simply no proof of stability, so the caller havocs,
/// the same posture every other refused containment ask in this crate
/// already takes.
pub(super) fn stable_by_containment(narrower: &RefinedSet, wider: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    let scalar_asked = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(narrower, wider));
    if let Ok(subset) = scalar_asked {
        return subset;
    }
    let seq_asked = crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(narrower, wider));
    matches!(seq_asked, Ok(true))
}

/// A closed scalar hull `[lo, hi]` read back off a kernel `Bounds`
/// answer — `Some((lo, hi, is_integer))`, `lo`/`hi` each `None` when
/// that side is unbounded. Reads exactly the two non-strict bound
/// forms and the `Integer` mark the kernel's own `encodeEnclosure`
/// always wires for a `Bounds` answer (`refined-lean/boundary/
/// encode_sets.lean`) — `AtLeast`/`AtMost`, the same two
/// `lattice_operations.rs`'s own private `integral_closed_window`
/// reads. An unbounded edge crosses the wire AS an `AtLeast`/`AtMost`
/// carrying `f64::NEG_INFINITY`/`f64::INFINITY` (`encodeEnclosure`
/// wires both edges unconditionally, `encodeNumber .negInf/.posInf` —
/// there is no "form omitted" wire shape), so an infinite edge here is
/// read back as `None`, matching what "unbounded" means to this
/// reader's own caller. `None` (the whole read declines) on a strict
/// edge (`Above`/`Below` — a hull the kernel proved with a strict
/// bound survives only through `Enclosure.weaken`'s callers, not
/// through this one) or any other form: the caller declines to widen
/// rather than guess at a bound the kernel did not state as a plain
/// window.
pub(super) fn hull_window(hull: &RefinedSet) -> Option<(Option<f64>, Option<f64>, bool)> {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    let mut is_integer = false;
    for form in &hull.forms {
        match form.form {
            Form::AtLeast if form.a == f64::NEG_INFINITY => {}
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost if form.a == f64::INFINITY => {}
            Form::AtMost => hi = Some(form.a),
            Form::Integer => is_integer = true,
            _ => return None,
        }
    }
    Some((lo, hi, is_integer))
}

/// Builds the widened scalar hull `W` for a Set pair whose containment
/// check failed because the accumulation genuinely GROWS: `joined`'s
/// hull (asked from the kernel via `ask_bounds_public` — the actual
/// enclosure of whatever forms `joined`'s set wears, not merely its own
/// syntactic spelling) is the pre-loop-join bound, `second`'s hull is
/// one further body step past it. For each edge, the side that grew
/// (second's hi above joined's hi; second's lo below joined's lo) is
/// DROPPED entirely (no `at_most`/`at_least` on that edge — the true
/// accumulated value could be reached after any number of further
/// iterations, so no finite bound on the growing side is sound), and
/// the side that stayed put keeps `joined`'s own bound. `integer()`
/// survives only when BOTH hulls are integral — the same all-sides-
/// agree rule `lattice_operations.rs`'s own run/hull collapses already
/// use before keeping the integer mark. `None` when either side's
/// `Bounds` ask refuses or reads empty, either hull is unreadable
/// (`hull_window` refused), or the widened window would admit no
/// members (`lo > hi` after widening, which cannot happen from a
/// genuine growth but guards a malformed pair rather than building an
/// impossible set).
fn widened_scalar_hull(joined_set: &RefinedSet, second_set: &RefinedSet) -> Option<RefinedSet> {
    let joined_bounds = ask_bounds_public(joined_set)?;
    let second_bounds = ask_bounds_public(second_set)?;
    if joined_bounds.empty || second_bounds.empty {
        return None;
    }
    let (joined_lo, joined_hi, joined_integer) = hull_window(&joined_bounds.hull)?;
    let (second_lo, second_hi, second_integer) = hull_window(&second_bounds.hull)?;
    // the lower edge GREW when the second pass's own edge reads lower
    // than the joined edge — including the second pass going unbounded
    // outright (`second_lo` is `None`, `hull_window`'s own reading of
    // an infinite edge) while `joined_lo` was still finite; either
    // shape drops the edge from `W` entirely. `joined_lo` already
    // `None` stays `None` (nothing to widen further).
    let lo = match (joined_lo, second_lo) {
        (Some(_), None) => None,
        (Some(j), Some(s)) if s < j => None,
        (j, _) => j,
    };
    let hi = match (joined_hi, second_hi) {
        (Some(_), None) => None,
        (Some(j), Some(s)) if s > j => None,
        (j, _) => j,
    };
    if let (Some(lo), Some(hi)) = (lo, hi) {
        if lo > hi {
            return None;
        }
    }
    let mut forms: Vec<Refinement> = Vec::new();
    if let Some(lo) = lo {
        forms.push(at_least(lo));
    }
    if let Some(hi) = hi {
        forms.push(at_most(hi));
    }
    if joined_integer && second_integer {
        forms.push(integer());
    }
    Some(make_refined_set(forms))
}

/// The widened candidate `W` for one name whose rejoin/containment
/// stability check both failed — `stabilized_join`'s own last resort
/// before havocing. Two shapes are tried: a plain scalar accumulator
/// (both sides' sets read a `Bounds` hull, widened edge-by-edge by
/// `widened_scalar_hull`) and a repetition-shaped accumulator (both
/// sides read back as `as_repetition` — a bounded list/sequence build —
/// widened the same way over the COUNT window, element = `joined`'s own
/// element, per this function's own doc below). Neither shape read
/// declines to the other; `None` when neither reads.
///
/// `W` is a CANDIDATE, not yet trusted: it is verified by rebinding
/// `name` to `W` in a fresh fork of `joined` (the pair's shared,
/// weaker-of-the-two grade, `SetKindTag::None` — the branch's own
/// precondition on both sides) and running the body ONE further time
/// from there. `W` is sound exactly when that run's own value for
/// `name` is contained in `W` (`stable_by_containment`): `W` already
/// contains the join (built by weakening `joined`'s own bound only on
/// the growing edge) and, once ONE more body step from `W` also stays
/// inside `W`, every later step does too by induction — a post-
/// fixpoint, which is the loop's true invariant. `None` when the
/// candidate cannot even be built, the verification run itself does
/// not complete (a statement shape `run_body_once` cannot walk), or the
/// verification containment check fails — `stabilized_join`'s caller
/// falls back to havocing in every one of those cases, exactly as it
/// did before this widening existed.
#[allow(clippy::too_many_arguments)]
pub(super) fn widened_set_candidate(
    joined_value: &AbstractValue,
    second_value: &AbstractValue,
    name: &str,
    joined: &Environment,
    body: &[Stmt],
    target: &Expr,
    element: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<AbstractValue> {
    let candidate_set = if let (Some(joined_repeated), Some(second_repeated)) =
        (as_repetition(&joined_value.set), as_repetition(&second_value.set))
    {
        // a bounded-list accumulator: widen the COUNT window the same
        // way a scalar hull widens, element = `joined`'s own element —
        // the accumulation's element claim is already stable (only the
        // COUNT grows across passes; a growing ELEMENT claim would
        // already have failed `stable_by_containment` on the element's
        // own scalar set, which this branch does not attempt to widen).
        let lo = if second_repeated.lo < joined_repeated.lo { 0 } else { joined_repeated.lo };
        let hi = match (joined_repeated.hi, second_repeated.hi) {
            (Some(joined_hi), Some(second_hi)) if second_hi > joined_hi => None,
            (joined_hi, _) => joined_hi,
        };
        Some(repetition(joined_repeated.element, lo, hi))
    } else {
        widened_scalar_hull(&joined_value.set, &second_value.set)
    }?;

    let grade = min_trust_level(trust_level_of(joined_value), trust_level_of(second_value));
    let candidate_value = known_set(candidate_set.clone(), None, grade, SetKindTag::None);

    let mut verification = joined.fork();
    verification.bind(name, candidate_value.clone());
    if !bind_for_target(target, element, &mut verification) {
        return None;
    }
    run_body_once(body, &mut verification, kernel, judge_context)?;
    let verified_value = verification.read(name)?;
    if verified_value.kind != Kind::Set || verified_value.set_kind_tag != SetKindTag::None {
        return None;
    }
    if !stable_by_containment(&verified_value.set, &candidate_set, kernel) {
        return None;
    }
    Some(candidate_value)
}

/// The stability check every one-pass-plus-join abstract loop pass
/// shares: a body that only REBINDS its written names (`last = s`) sees
/// the same value on a second pass as the first, so joining the pre-loop
/// state with one pass is already a fixpoint. A body that ACCUMULATES
/// (`total += s * s`) does not — a second pass adds another term on top
/// of the first pass's own joined value, so the name a single join would
/// report is a bound the real, unboundedly-many-iteration run can
/// exceed. This function tells the two apart by running the body a
/// SECOND time, from a fork of the join of `environment` (pre-loop) and
/// `one_pass` (the first pass's own environment) — call that join `J` —
/// and testing, for every name the body writes, whether joining the
/// SECOND pass's own value into `J` changes `J` at all: a name is stable
/// when `join(J, second_pass) == J`, since `join_known` is idempotent
/// exactly where the second pass adds no new information beyond what `J`
/// already states. `PartialEq` alone answers this correctly for a
/// `Kind::Values` pair (the same-tag join arms only append values not
/// already present, so an already-covered join reproduces the identical
/// `Vec<f64>`), but `join_known` HAS NO STRUCTURAL FIXPOINT for a
/// `Kind::Set` pair — its general fallback always wraps both sides in a
/// fresh `union(...)` node, so a rejoin that denotes nothing new still
/// produces a NEW, differently-shaped `RefinedSet` that `PartialEq` calls
/// unequal. For that case (both sides `Kind::Set`, `SetKindTag::None`)
/// the structural mismatch is not read as instability outright — the
/// kernel is asked the real question instead, `stable_by_containment`'s
/// own containment verdict: the second pass's set is stable exactly when
/// it is CONTAINED in `J`'s set, which is what "the rejoin adds nothing"
/// actually means once the join no longer has a structural fixed point
/// to compare against. A name whose value is still `PartialEq`-unequal
/// AND (for a Set pair) not kernel-proved contained is REBOUND to
/// `unknown()` in the final environment, since it holds no claim this
/// walk can make; every other name — including one the body never
/// actually touches on this concrete run — keeps its `J` value. The loop
/// target itself is excluded from this comparison and this havoc: it is
/// rebound to a fresh element abstraction every iteration by construction
/// (`bind_for_target`'s own call at each pass), never accumulated, so
/// comparing it across passes would only ever measure two different
/// intentional bindings and never a genuine instability.
///
/// The names compared are every bare name `written_names` finds
/// SYNTACTICALLY in `body` (a superset of what one concrete pass
/// actually writes is safe here — see that function's own doc). For each
/// one, `J`'s own value and the second pass's own value are re-joined
/// through the same `lattice_operations::join_known` every ordinary
/// branch join already uses (via `Environment::join` on two single-
/// binding forks). A value that happens to be `PartialEq`-different from
/// `J` after the re-join, and is not a `Kind::Set` pair the kernel proves
/// contained, is at worst treated as unstable and havoced to unknown,
/// which is always a weaker, still-undetermined answer, never a wrong
/// one — a question the kernel refuses leaves the name havoced, the same
/// as a structural mismatch it never had the chance to ask about.
///
/// Returns `None` when the second pass hits a statement shape `run_body_
/// once` cannot run (the same "this loop is not this module's shape"
/// decline the first pass already uses) — an unwalkable second pass
/// gives no stability answer to trust, so the whole loop declines rather
/// than publish the first pass's own join unchecked. The second pass's
/// own control-flow outcome (`Broke`/`Continued`/`Returned`) is read only
/// as this success/failure signal; its `Returned` value is not itself
/// used to build the answer since the second pass's whole purpose here
/// is the stability comparison, not a fresh answer to return through.
///
/// `Some((environment, widened))` — `widened` names every bare name this
/// pass rebound to `unknown()` because it never reached a fixed point,
/// SORTED (`HashSet` iteration order is not stable, and a body writing
/// more than one such name still needs a single, reproducible FIRST name
/// for the caller's own blocker) — empty when every written name
/// stabilized. This function itself records no finding: `check.rs`'s
/// `walk_loop` owns turning a non-empty `widened` into this body's own
/// blocker, the same way it already owns every other loop-shaped
/// blocker.
pub(super) fn stabilized_join(
    environment: &Environment,
    one_pass: &Environment,
    body: &[Stmt],
    target: &Expr,
    element: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
    judge_context: &mut JudgeContext,
) -> Option<(Environment, Vec<String>)> {
    let joined = Environment::join(environment.fork(), one_pass);

    let mut second_pass = joined.fork();
    if !bind_for_target(target, element, &mut second_pass) {
        return None;
    }
    run_body_once(body, &mut second_pass, kernel, judge_context)?;

    let mut excluded = std::collections::HashSet::new();
    target_names(target, &mut excluded);
    let mut candidates = std::collections::HashSet::new();
    written_names(body, &mut candidates);

    let mut result = joined.fork();
    let mut widened: Vec<String> = Vec::new();
    for name in candidates {
        if excluded.contains(&name) {
            continue;
        }
        let Some(joined_value) = joined.read(&name) else {
            continue;
        };
        let Some(second_value) = second_pass.read(&name) else {
            continue;
        };
        // a single-name fork carrying just this one binding, joined
        // against the same single-name binding off `joined` — the
        // per-name reading of `join(J, second_pass) == J`, built out of
        // the same two-environment `Environment::join` every call site
        // already uses rather than a new per-value join entry point.
        let mut left = joined.fork();
        left.bind(&name, joined_value.clone());
        let mut right = joined.fork();
        right.bind(&name, second_value.clone());
        let rejoined = Environment::join(left, &right);
        let rejoined_value = rejoined.read(&name);
        let mut stable = rejoined_value == joined.read(&name);
        // the structural rejoin has no fixpoint for a Set pair — ask the
        // kernel whether the second pass's set is genuinely covered by
        // `J`'s set before havocing what may be a real determination.
        let is_set_pair = joined_value.kind == Kind::Set
            && second_value.kind == Kind::Set
            && joined_value.set_kind_tag == SetKindTag::None
            && second_value.set_kind_tag == SetKindTag::None;
        if !stable && is_set_pair {
            stable = stable_by_containment(&second_value.set, &joined_value.set, kernel);
        }
        if !stable && is_set_pair {
            // containment refused too — the accumulation GREW rather than
            // merely rejoining. Before havocing, try the widened hull: a
            // post-fixpoint candidate that keeps the stable side's bound
            // and drops the growing side, verified by running the body a
            // THIRD time from the joined environment with the name
            // rebound to the candidate — see `widened_set_candidate`'s
            // own doc for why this is sound.
            if let Some(widened_value) = widened_set_candidate(
                joined_value,
                second_value,
                &name,
                &joined,
                body,
                target,
                element,
                kernel,
                judge_context,
            ) {
                result.bind(&name, widened_value);
                continue;
            }
        }
        if !stable {
            result.bind(&name, unknown());
            widened.push(name);
        }
    }
    widened.sort();
    Some((result, widened))
}
