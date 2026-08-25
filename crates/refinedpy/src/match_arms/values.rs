//! The scalar-narrowing/splitting layer: what a decidable scalar
//! subject narrows to after ONE arm's literal or guard decides
//! TAKEN/NOT-TAKEN against it, and the bookkeeping `match_taken_
//! environment`'s per-arm split needs on top of that narrowing.

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::min_trust_level;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Pattern;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::captures::pattern_captures;
use super::value_proof::pattern_proved_value;

/// The admitted numeric members a subject enumerates, if it enumerates
/// any this file can read — the membership-question counterpart
/// `match_value_outcome` asks instead of the plain single-value
/// equality `single_numeric_value` alone can answer. Three shapes:
///
/// - `Kind::Values` (Number/Boolean/Integer/Float-tagged): its own
///   `values` directly — a single known scalar is the `len() == 1`
///   case already handled before this function existed; a
///   multi-valued binding (`{1, 2, 4}`, an ordinary join of several
///   known values — `lattice_operations.rs::join_known`'s same-sort
///   arm) enumerates every value it carries.
/// - `Kind::Set` that enumerates a union-of-singleton-scalars form —
///   `collection_models.rs::scalars_of_union_of_singletons`, reused
///   here rather than re-parsed, the same reader
///   `known_value_of_state` uses to read a kernel-joined dict value
///   back to exact values. A set that does NOT enumerate (a range, a
///   star, a multi-codepoint string tuple) answers `None` — this
///   function never guesses at values that are not actually
///   enumerated.
/// - `Kind::KindUnion` is read one level up, in
///   `kind_union_pattern_outcome` — a union asks per-arm, not through
///   this flat membership list, since an Undecidable arm must poison
///   the whole judgment rather than silently drop out of a merged
///   member list.
pub(super) fn enumerable_numeric_members(subject: &AbstractValue) -> Option<Vec<f64>> {
    if subject.kind == Kind::Values {
        return match subject.kind_tag {
            Some(PrimitiveKind::Number)
            | Some(PrimitiveKind::Boolean)
            | Some(PrimitiveKind::Integer)
            | Some(PrimitiveKind::Float) => Some(subject.values.clone()),
            _ => None,
        };
    }
    if subject.kind == Kind::Set {
        return crate::collection_models::scalars_of_union_of_singletons(&subject.set);
    }
    None
}

/// One `case` pattern's own flat list of numeric literals — every value
/// a `MatchValue`/`MatchOr`-of-numerics/`MatchAs`-wrapping-one names,
/// read via `pattern_proved_value` and unwrapped back to its bare
/// `Vec<f64>` (this function drops the tag/grade `pattern_proved_value`
/// carries, since the two callers below fold the result against a
/// SUBJECT's own tag, never the pattern's). `None` for a pattern
/// `pattern_proved_value` itself does not prove a value for (a bare
/// capture/wildcard, a singleton `None`, a sequence/mapping/class
/// pattern) — the same declines, read through the one existing proof
/// function rather than re-deriving them.
fn pattern_literal_members(pattern: &Pattern, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<Vec<f64>> {
    pattern_proved_value(pattern, environment, kernel).map(|proved| proved.values)
}

/// A decidable scalar subject's own narrowed value after ONE arm's
/// pattern decides TAKEN or NOT-TAKEN against it — the intersection/
/// difference pair `narrowing.rs`'s own isinstance/comparison leaves
/// already spell for a Values binding (`narrow_name_against_literal`'s
/// `filter` by a kept predicate, `narrow_isinstance_call`'s KindUnion
/// `filter` by tag match), applied here to a match arm's own admitted
/// members instead of a comparison/isinstance test:
///
/// - TAKEN (`keep_matched` true): the arm's own environment sees the
///   INTERSECTION — exactly the subject's admitted members that the
///   pattern's own literals also name (`case 1:` over `{1, 2, 4}`
///   narrows to `{1}`; `case 2 | 4:` narrows to `{2, 4}`, the union of
///   admitted alternatives, which IS the intersection of `{1, 2, 4}`
///   with the pattern's own `{2, 4}`).
/// - NOT-TAKEN (`keep_matched` false): the remainder every LATER arm
///   and the wildcard must see is the DIFFERENCE — the subject's
///   admitted members with the pattern's own literals removed.
///
/// `None` when the subject does not enumerate (`enumerable_numeric_
/// members`) or the pattern proves no literal (`pattern_literal_
/// members`) — the caller's own job to fall back to the unnarrowed
/// subject in that case, never to guess. The narrowed result keeps the
/// subject's own `kind_tag` (a pattern's literal tag is never trusted
/// over the subject's, matching `narrow_name_against_literal`'s own
/// "the binding's own tag survives" reading) and the WEAKER of the two
/// trust grades (`min_trust_level` — a narrowing is never claimed
/// stronger than either input that fed it).
pub(super) fn narrow_scalar_subject(
    subject: &AbstractValue,
    pattern: &Pattern,
    keep_matched: bool,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let subject_members = enumerable_numeric_members(subject)?;
    let kind_tag = subject.kind_tag?;
    let pattern_members = pattern_literal_members(pattern, environment, kernel)?;
    let kept: Vec<f64> = subject_members
        .into_iter()
        .filter(|member| pattern_members.contains(member) == keep_matched)
        .collect();
    let grade = min_trust_level(trust_level_of(subject), TrustProved);
    Some(known_values(kept, kind_tag, grade))
}

/// The bare name a `case x:` capture binds — a `Pattern::MatchAs` with
/// no inner sub-pattern and a real (non-wildcard) name. `None` for a
/// wildcard `case _:` (`as_pattern.name` is itself `None` there), an
/// `as`-wrapped sub-pattern (`case <pattern> as x:` — the SUBPATTERN
/// decides this arm, not a bare capture), or any other pattern shape.
/// This is the ONLY pattern shape the guard-narrowing split below reads:
/// a guard on a literal/or/sequence/mapping/class pattern keeps today's
/// binary semantics, exactly as `narrow_scalar_subject` already declines
/// (returns `None`) for every one of those shapes too.
pub(super) fn bare_capture_name(pattern: &Pattern) -> Option<&str> {
    let Pattern::MatchAs(as_pattern) = pattern else {
        return None;
    };
    if as_pattern.pattern.is_some() {
        return None;
    }
    Some(as_pattern.name.as_ref()?.id.as_str())
}

/// The guarded twin of `narrow_scalar_subject`, for the one pattern
/// shape that function's own scalar-literal proof cannot read at all: a
/// BARE CAPTURE (`case x:`) whose arm carries a GUARD that is itself a
/// comparison `narrowing.rs`'s own reader already proves
/// (`narrowing::guard_narrowed_values`, which runs the guard through
/// `assume` — the SAME comparison-narrowing leaf every `if`/`elif`
/// condition in this checker already reads, never reimplemented here).
/// `case x if x == 1: / case x if x == 2: / case _:` over `oneOf(1, 2,
/// 4)` splits identically to the literal spelling `case 1: / case 2:
/// / case _:` through this same function, called the same way
/// `match_taken_environment` already calls `narrow_scalar_subject` for a
/// literal pattern's own split. `case x if x in (2, 4):` splits the same
/// way now too: `narrowing::narrow_name_against_membership` reads the
/// tuple's own literal members directly against the `Kind::Values`
/// binding `guard_narrowed_values`'s sandbox holds, the VALUES channel's
/// own leaf for `in`/`not in` (the SET channel's `membership_leaf_tree_of`
/// is the separate `Kind::Set` reading, used when the sandboxed name is a
/// sort rather than an enumerated value list). A guard shape neither
/// channel reads at all (a call, an attribute, two changing names) is
/// still `guard_narrowed_values`'s own decline, not this function's — it
/// never widens the comparison reader's own coverage on its own account.
///
/// `keep_matched: true` asks the guard's OWN admitted values (`assume`'s
/// `truth: true`) — the arm's narrowed walk, intersected with
/// `remaining_subject` so a guard that (wrongly, or by construction)
/// admits values the subject never carried is never widened past what
/// the subject already stated. `keep_matched: false` asks the guard's
/// OWN excluded values (`truth: false`) — the difference every LATER arm
/// and the wildcard must still see, the same DIFFERENCE role
/// `narrow_scalar_subject`'s own `false` arm plays for a literal
/// pattern's split.
///
/// `None` when the pattern is not a bare capture, there is no guard, the
/// subject does not enumerate, or `guard_narrowed_values` itself declines
/// (a guard shape `narrowing.rs`'s comparison reader does not prove) —
/// the caller's own job to fall back to today's binary guard semantics
/// for that arm, never to guess a split the guard's own reader could not
/// prove. This function answers each `keep_matched` side independently;
/// `match_taken_environment`'s own caller requires BOTH the
/// `keep_matched: true` and `keep_matched: false` calls to answer `Some`
/// before treating the arm as split — a lone side is never enough (see
/// that function's own doc).
pub(super) fn guarded_bare_capture_narrowed(
    subject: &AbstractValue,
    pattern: &Pattern,
    guard: Option<&Expr>,
    keep_matched: bool,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let name = bare_capture_name(pattern)?;
    let guard = guard?;
    enumerable_numeric_members(subject)?;
    let subject_kind_tag = subject.kind_tag?;
    let narrowed = crate::narrowing::guard_narrowed_values(guard, name, subject, kernel, keep_matched)?;
    if narrowed.kind_tag != Some(subject_kind_tag) {
        return None;
    }
    let subject_members = enumerable_numeric_members(subject).expect("checked Some above");
    let kept: Vec<f64> = subject_members
        .into_iter()
        .filter(|member| narrowed.values.contains(member))
        .collect();
    let grade = min_trust_level(trust_level_of(subject), TrustProved);
    Some(known_values(kept, subject_kind_tag, grade))
}

/// Whether `narrowed` is the SAME admitted set as `remaining` — both
/// read through `enumerable_numeric_members` and compared as sets
/// (order-independent: a join can enumerate its members in either
/// order). `narrowed` is always a subset of `remaining` by construction
/// (`narrow_scalar_subject`'s own intersection), so equal LENGTH with
/// every member of one present in the other is exactly set equality
/// here — this is the FULL-OVERLAP test: an arm whose intersection is
/// the whole remaining subject consumes it entirely, the same
/// unconditional Taken this file gave every arm before the per-arm
/// split existed (no later arm can ever be reached, so no join is
/// needed for this arm).
pub(super) fn is_full_overlap(narrowed: &AbstractValue, remaining: &AbstractValue) -> bool {
    let (Some(narrowed_members), Some(remaining_members)) =
        (enumerable_numeric_members(narrowed), enumerable_numeric_members(remaining))
    else {
        return false;
    };
    narrowed_members.len() == remaining_members.len()
        && remaining_members.iter().all(|member| narrowed_members.contains(member))
}

/// Rebinds `subject_name` (the match subject's own name, when the
/// subject expression is a bare `Name`) and every name `pattern`
/// captures (`pattern_captures`) to `intersected` inside `arm_env` — the
/// two slots the PARTIAL-OVERLAP split (`match_taken_environment`'s own
/// doc) must narrow before the arm's body ever walks: a bare `MatchAs`
/// binds its own name to the raw `remaining_subject` when the pattern
/// is Taken (`match_as_outcome`'s own doc), which is correct only for
/// the FULL-overlap case; a split arm's body must see the INTERSECTION
/// instead, on every name that would otherwise still read the coarser
/// pre-split claim. A pattern with no nameable captures
/// (`pattern_captures` answers `None` for a shape past this file's flat
/// scope — never reached here, since only a literal/or pattern that
/// itself proved a value reaches a split) simply rebinds the subject
/// name alone.
pub(super) fn rebind_split_subject(
    arm_env: &mut Environment,
    subject_name: Option<&str>,
    pattern: &Pattern,
    intersected: &AbstractValue,
) {
    if let Some(name) = subject_name {
        arm_env.bind(name, intersected.clone());
    }
    if let Some(captures) = pattern_captures(pattern, None) {
        for name in captures {
            arm_env.bind(&name, intersected.clone());
        }
    }
}
