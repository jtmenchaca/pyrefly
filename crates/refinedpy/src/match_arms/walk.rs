//! Walk every arm of a match statement in order, deciding each with
//! `arm_outcome`, splitting a decidable scalar subject per arm, and
//! joining the arms that survive — `match_taken_environment`, the
//! entry point every caller outside this module reaches.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::MatchCase;
use ruff_python_ast::Pattern;
use ruff_python_ast::Stmt;

use crate::env::Environment;

use super::outcome::arm_outcome;
use super::outcome::ArmOutcome;
use super::values::bare_capture_name;
use super::values::enumerable_numeric_members;
use super::values::guarded_bare_capture_narrowed;
use super::values::is_full_overlap;
use super::values::narrow_maybe_subject_on_none;
use super::values::narrow_scalar_subject;
use super::values::rebind_split_subject;

/// Walk every arm of a match statement in order, deciding each with
/// `arm_outcome`, joining the arms that survive exactly the way
/// `walk_if` joins its own branch environments (`Environment::join`,
/// left-folded over every surviving arm — 1 survivor is that arm alone,
/// 2+ actually joins; see `finalize_survivors`). `walk_arm_body(body,
/// &mut arm_env)` runs the caller's OWN statement walker over one arm's
/// body (`check.rs`'s `walk_statement` loop, `summaries.rs`'s
/// `interpret_body` — this file stays walker-agnostic on purpose, so it
/// never depends on either caller's own types) and answers `Some(true)`
/// when the arm survives (falls through, matching `arm_terminates`'s
/// own reading of "does not end in `return`/`raise`"), `Some(false)`
/// when it terminates, or `None` when the walk itself must decline the
/// WHOLE match (an unsupported statement inside the body) — propagated
/// here by `?`, the same short-circuit `interpret_body`'s own callers
/// already rely on.
///
/// Returns `Some((environment, falls_through))`: the post-match
/// environment (the one taken arm's own, or every surviving split arm's
/// joined together), and whether the match AS A WHOLE falls through —
/// `true` unless every reached arm terminated. `None` means no arm is
/// decidably reached at all (the whole match is undecided from its own
/// start, no body has walked, and the caller's own join-fallback is
/// free to re-walk every case for itself).
///
/// A subject that DOES NOT enumerate (`enumerable_numeric_members`
/// answers `None` — an unbounded `Kind::Set` ray, `Kind::Unknown`, a
/// non-numeric subject, …) never splits: every arm is judged, walked,
/// and joined exactly as before this rule existed — one arm's own
/// `Taken` outcome is unconditional (the same "later arms are dead,
/// stop scanning" this function always gave), and the answer is that
/// one arm's own body's environment.
///
/// A DECIDABLE SCALAR SUBJECT splits per arm — the abstractly-precise
/// reading of what a multi-valued subject actually means: SOME admitted
/// values take this arm, the rest fall through to later ones, the same
/// two-branch shape an `if`/`else` over an unknown boolean already
/// walks. Before an arm is even judged, an ALREADY-EMPTY
/// `remaining_subject` (every earlier arm together consumed every
/// admitted value) makes it — and every arm after it — dead: no body
/// walks, nothing joins, the loop simply runs out (this is the
/// wildcard-sees-emptiness case: `case 1: / case 2 | 4: / case _:` over
/// `{1, 2, 4}` leaves the wildcard's own `remaining_subject` empty, and
/// its body never walks). Otherwise, a Taken arm asks `narrow_scalar_
/// subject`'s own INTERSECTION (`keep_matched: true`) against
/// `remaining_subject`:
///
/// - the intersection is the WHOLE remaining subject (`is_full_overlap`)
///   — this arm alone consumes every value still live, so it behaves as
///   an unconditional Taken: no later arm is ever reached, and this
///   function returns as soon as this one arm's own body is walked
///   (survives or not; a survivor is the sole answer, no join needed).
/// - the intersection is a PROPER, NONEMPTY subset — a genuine split:
///   `rebind_split_subject` narrows the arm's own subject binding and
///   capture(s) down to the intersection, the body walks under THAT
///   narrower claim, and (if it survives) its environment joins the
///   later arms' the same way two `if`/`else` branches join. The
///   DIFFERENCE (`keep_matched: false`) becomes the new
///   `remaining_subject` for every arm still to come, and the walk
///   continues scanning rather than stopping.
/// - `narrow_scalar_subject` answers `None` (the pattern proves no
///   scalar literal — a bare capture/wildcard, a guard-only arm, a
///   class/sequence/mapping pattern) — the pattern is not itself
///   scalar-decidable even though the subject is, so this arm keeps
///   TODAY'S binary semantics: unconditional Taken, no split.
///
/// A GUARDED BARE CAPTURE (`case x if <condition>:`) is tried BEFORE
/// `arm_outcome` is ever called for that arm: `guarded_bare_capture_
/// narrowed` asks `narrowing::guard_narrowed_values` — the SAME
/// comparison-narrowing reader `assume` gives every `if`/`elif` — for
/// the guard's own admitted (`keep_matched: true`) AND excluded
/// (`keep_matched: false`) values, and this path is taken only when BOTH
/// answer `Some` (a lone proved side is never enough — reading the
/// unproved side as "admits/excludes everything" would manufacture a
/// claim the guard's own reader never made); it then supersedes
/// `arm_outcome`'s own binary evaluation, which would otherwise run the
/// guard against a single arm environment holding the WHOLE
/// `remaining_subject` and decline (`Undecidable`) the moment that
/// evaluation is not a single known boolean. `case x if x == 1: / case x
/// if x == 2: / case _:` over `oneOf(1, 2, 4)` therefore splits
/// IDENTICALLY to the literal spelling `case 1: / case 2: / case _:`
/// through this same intersection/difference/full-overlap reading. A
/// guard the reader cannot prove both sides of (both calls answer `None`
/// — e.g. `x in (2, 4)`, membership over a `Kind::Values` binding, which
/// is the SET channel's own leaf and needs `Kind::Set`) falls through to
/// `arm_outcome`'s existing binary semantics unchanged.
///
/// An `Undecidable` arm still poisons every LATER arm exactly as
/// before — this file cannot rule out that arm having actually run.
/// When no EARLIER arm's body has walked yet, the whole match declines
/// (`None`), the same as before this rule existed. Once an earlier
/// split arm's body already walked for real, declining here would let
/// the caller's join-fallback re-walk it a second time (duplicating any
/// side effects that walk already recorded) — so this answers instead
/// with whatever already survived (see the `any_arm_walked` guard on
/// the `Undecidable` arm below).
pub fn match_taken_environment(
    subject_value: &AbstractValue,
    subject_name: Option<&str>,
    cases: &[MatchCase],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    walk_arm_body: &mut dyn FnMut(&[Stmt], &mut Environment) -> Option<bool>,
) -> Option<(Environment, bool)> {
    let mut remaining_subject = subject_value.clone();
    let mut survivors: Vec<Environment> = Vec::new();
    // Whether ANY arm's body was actually walked (an unconditional Taken,
    // or a split arm) — distinguishes "every arm bottomed out at
    // NotTaken/dead, the match is genuinely undecided" (still `None`,
    // the caller's own join-fallback engages, exactly as before this
    // rule existed) from "arms were decided and walked, but every
    // surviving one terminated" (a real answer, falls-through `false` —
    // the same "0 survivors" reading `walk_if`'s own join gives, never
    // a fallback re-walk that would run those bodies' side effects a
    // second time).
    let mut any_arm_walked = false;
    for case in cases {
        // An already-exhausted remaining subject makes this arm (and
        // every arm after it) dead: no runtime value is left for it to
        // ever see, so its body never walks and it never joins.
        if enumerable_numeric_members(&remaining_subject).is_some_and(|members| members.is_empty()) {
            continue;
        }
        // GUARDED BARE-CAPTURE SPLIT: `case x if <condition>:` over a
        // multi-valued `remaining_subject` — `arm_outcome`'s own
        // `apply_guard` evaluates the guard against a SINGLE arm
        // environment where `x` is bound to the WHOLE remaining subject,
        // so a guard like `x == 1` evaluates to a multi-valued/unknown
        // boolean there and `known_boolean` declines, poisoning every
        // later arm through the `Undecidable` branch below — exactly the
        // binary reading a decidable scalar subject must NOT get, the
        // same problem a literal pattern's own split
        // (`narrow_scalar_subject`) already solves for `case 1:`. Tried
        // BEFORE `arm_outcome`, so a guard the reader proves something
        // about never reaches that binary evaluation at all: the guard's
        // own admitted values (`keep_matched: true`) decide this arm's
        // split exactly like a literal pattern's intersection would, and
        // its excluded values (`keep_matched: false`) become the
        // difference threaded onward.
        //
        // BOTH calls must answer `Some` before this path is taken — a
        // lone `Some` (one direction proved, the other not) is never
        // enough: reading the UNPROVED side as "the guard admits/excludes
        // everything" would silently manufacture a claim the guard's own
        // reader never made, exactly the unsoundness a decline exists to
        // prevent. `narrow_name_against_literal`'s own `filter` runs on
        // both `truth` values together for every comparison this reader
        // recognizes, so a genuinely provable guard always answers both;
        // requiring both costs nothing on the shapes this file actually
        // proves and closes the asymmetric gap on every shape it does
        // not. `None` from either call (the guard is not one shape
        // `narrowing::guard_narrowed_values` proves) falls through to
        // `arm_outcome`'s own binary reading unchanged.
        if case.guard.is_some() {
            let guarded_intersection =
                guarded_bare_capture_narrowed(&remaining_subject, &case.pattern, case.guard.as_deref(), true, kernel);
            let guarded_difference =
                guarded_bare_capture_narrowed(&remaining_subject, &case.pattern, case.guard.as_deref(), false, kernel);
            if let (Some(intersected), Some(difference)) = (guarded_intersection, guarded_difference) {
                if intersected.values.is_empty() {
                    // the guard admits none of what remains: this arm is
                    // a dead arm, the same NotTaken every other
                    // unreachable arm answers — thread the difference
                    // onward and keep scanning.
                    remaining_subject = difference;
                    continue;
                }
                let mut arm_env = environment.fork();
                if let Some(name) = bare_capture_name(&case.pattern) {
                    arm_env.bind(name, intersected.clone());
                }
                if is_full_overlap(&intersected, &remaining_subject) {
                    // The guard admits every value still remaining, so no
                    // later arm is ever reached and this is the last body
                    // to walk. It is NOT necessarily the only one: an
                    // earlier arm may have already split off its own slice
                    // and survived, and that survivor is still a real path
                    // through the match. Walk this body, then answer with
                    // the same join every other multi-arm exit takes —
                    // returning this arm alone would drop the earlier
                    // arms' bindings and report one walked body where two
                    // ran.
                    if let Some(name) = subject_name {
                        arm_env.bind(name, intersected.clone());
                    }
                    let survives = walk_arm_body(&case.body, &mut arm_env)?;
                    if survives {
                        survivors.push(arm_env);
                    }
                    return Some(if survivors.is_empty() {
                        (environment.fork(), false)
                    } else {
                        (finalize_survivors(survivors), true)
                    });
                }
                any_arm_walked = true;
                rebind_split_subject(&mut arm_env, subject_name, &case.pattern, &intersected);
                let survives = walk_arm_body(&case.body, &mut arm_env)?;
                if survives {
                    survivors.push(arm_env);
                }
                remaining_subject = difference;
                continue;
            }
        }
        // MAYBE-CARRIER `case None:` SPLIT: `remaining_subject` is a
        // `Kind::PossiblyUndefined` carrier (an `Optional[X]`/`X |
        // None`-declared parameter's own seed) and this arm's pattern is
        // the `None` singleton — `values::narrow_maybe_subject_on_none`
        // reads the two sides the same way `is None`/`is not None`
        // already narrow one: `keep_matched: true` is the exact `None`
        // value, `keep_matched: false` is the carrier's own inner
        // (present) value unwrapped. Tried BEFORE `arm_outcome`, the
        // same reason the guarded-bare-capture split above runs first:
        // `arm_outcome`'s own `match_singleton_outcome` declines a
        // maybe-carrier subject outright (`outcome::subject_is_singleton`'s
        // own doc — neither provably taken nor provably dead), which
        // would otherwise poison every later arm through the
        // `Undecidable` branch below instead of splitting. `None` from
        // either call (a subject that is not this carrier shape, or a
        // pattern that is not `case None:`) falls through to
        // `arm_outcome`'s ordinary reading unchanged.
        if let Pattern::MatchSingleton(singleton) = &case.pattern {
            if singleton.value == ruff_python_ast::Singleton::None && remaining_subject.kind == refined_domain::abstract_value::Kind::PossiblyUndefined {
                if let (Some(intersected), Some(difference)) = (
                    narrow_maybe_subject_on_none(&remaining_subject, &case.pattern, true),
                    narrow_maybe_subject_on_none(&remaining_subject, &case.pattern, false),
                ) {
                    let mut arm_env = environment.fork();
                    if let Some(name) = subject_name {
                        arm_env.bind(name, intersected.clone());
                    }
                    any_arm_walked = true;
                    let survives = walk_arm_body(&case.body, &mut arm_env)?;
                    if survives {
                        survivors.push(arm_env);
                    }
                    remaining_subject = difference;
                    continue;
                }
            }
        }
        match arm_outcome(&case.pattern, case.guard.as_deref(), &remaining_subject, environment, kernel) {
            ArmOutcome::Taken(mut arm_env) => {
                let intersection = narrow_scalar_subject(&remaining_subject, &case.pattern, true, environment, kernel);
                let split = intersection
                    .as_ref()
                    .filter(|intersected| !is_full_overlap(intersected, &remaining_subject));
                let Some(intersected) = split else {
                    // no scalar split for this arm — unconditional Taken,
                    // exactly today's behavior: walk its body once and
                    // commit ITS OWN environment regardless of whether
                    // the body survives (`check.rs`'s pre-existing
                    // decided-arm branch committed `arm_env`
                    // unconditionally too — a body ending in
                    // `return`/`raise` is still the honest post-match
                    // state, the same way a single surviving `if` arm
                    // needs no join). No later arm is ever reached; the
                    // whole match falls through iff this one body does.
                    let survives = walk_arm_body(&case.body, &mut arm_env)?;
                    return Some((arm_env, survives));
                };
                // a genuine partial split: narrow this arm's own subject
                // binding/capture(s) to the intersection, walk under
                // that narrower claim, and thread the difference onward
                // to every arm still to come instead of stopping here.
                any_arm_walked = true;
                rebind_split_subject(&mut arm_env, subject_name, &case.pattern, intersected);
                let survives = walk_arm_body(&case.body, &mut arm_env)?;
                if survives {
                    survivors.push(arm_env);
                }
                if let Some(narrowed) =
                    narrow_scalar_subject(&remaining_subject, &case.pattern, false, environment, kernel)
                {
                    remaining_subject = narrowed;
                }
                continue;
            }
            ArmOutcome::NotTaken => {
                if let Some(narrowed) =
                    narrow_scalar_subject(&remaining_subject, &case.pattern, false, environment, kernel)
                {
                    remaining_subject = narrowed;
                }
                continue;
            }
            // An Undecidable arm poisons every LATER arm exactly as
            // before — this file cannot rule out that arm having
            // actually run. When NO earlier arm's body was walked yet,
            // that is still `None`: the whole match is undecided from
            // its own start, and the caller's join-fallback is free to
            // re-walk every case from scratch. Once an earlier SPLIT
            // arm's body already walked for real (side effects: findings
            // recorded, returns collected), falling back would re-walk
            // it a second time — so this answers with whatever already
            // survived instead (falls-through `true` only when at least
            // one split arm actually survived; `false`, same as the
            // "no survivors" tail below, otherwise).
            ArmOutcome::Undecidable => {
                if !any_arm_walked {
                    return None;
                }
                return Some(if survivors.is_empty() {
                    (environment.fork(), false)
                } else {
                    (finalize_survivors(survivors), true)
                });
            }
        }
    }
    if survivors.is_empty() {
        return if any_arm_walked { Some((environment.fork(), false)) } else { None };
    }
    Some((finalize_survivors(survivors), true))
}

/// The join every split-arm walk in `match_taken_environment` funnels
/// through — the SAME shape `walk_if`'s own tail takes
/// (`check.rs::walk_if`): 1 survivor is that arm's environment alone,
/// 2+ left-folds through `Environment::join` exactly as `walk_if`/
/// today's `walk_match` join their own branch environments — this
/// function never invents a different join. The caller never calls
/// this with an empty `survivors` (it reads that case itself, before
/// ever reaching here — see `match_taken_environment`'s own tail).
fn finalize_survivors(mut survivors: Vec<Environment>) -> Environment {
    match survivors.len() {
        1 => survivors.into_iter().next().expect("len checked above"),
        _ => {
            let mut joined = survivors.remove(0);
            for arm in survivors {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    }
}
