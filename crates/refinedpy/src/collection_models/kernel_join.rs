//! The kernel-delegate-first fold `dict_key_set_read` uses to join two
//! dict-entry values through the kernel's own proved `join_state` when
//! both sides convert to a scalar kernel state — falling back to the
//! local `join_known` on any refusal.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::trust_grades::min_trust_level;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustLevel;
use refined_kernel::kernel_bridge::kernel_if_loaded;
use refined_kernel::kernel_interface::KnownStateWire;

use super::dict_key::DictKey;
use super::subscript_read::dict_key_read;

/// The kernel state a SCALAR knowledge value denotes — narrowed to the
/// two shapes `dict_key_set_read`'s fold ever hands it: an untagged
/// numeric-or-other `Kind::Values` singleton set, or a plain `Kind::Set`
/// (`set_kind_tag == SetKindTag::None`). Mirrors
/// `lattice_conformance.rs`'s own `state_of_known`, cut down to the
/// scalar-set rows this call site can produce — no Undef/Null/NaN/
/// wrapper arm, since a dict entry's own value never reaches this fold
/// in one of those shapes without already having declined earlier
/// (`dict_key_read` hands back the entry's `AbstractValue` verbatim,
/// and `word_tuples_of`'s gate above only ever supplies exact-string
/// keys, never an absent/NaN VALUE). `set_of_known` is the existing
/// tuple-layer reader every other kernel-asking row in this crate
/// already uses to reach a `RefinedSet`.
fn known_state_of(value: &AbstractValue) -> Option<KnownStateWire> {
    if value.kind != Kind::Values && value.kind != Kind::Set {
        return None;
    }
    if value.kind == Kind::Set && value.set_kind_tag != SetKindTag::None {
        return None; // a worn set (bigint/symbol) carries no ℝ̄ member — set_of_known refuses it too
    }
    let set = set_of_known(value)?;
    Some(KnownStateWire { top: false, set, undef: false, null: false, nan: false, thrown: false })
}

/// A right-fold `Form::Union` tree whose every leaf is a singleton
/// scalar `OneOf` (never a multi-codepoint string tuple, never a bare
/// range/star/etc): the exact values it admits, in no particular
/// order. This is the shape the kernel's `join_state` answers for two
/// (or, folded further, more) distinct scalar values — `{40} ∪ {41}` —
/// the same set `join_known`'s own untagged-numeric arm spells
/// (`lattice_operations.rs`'s `known_set(make_refined_set(vec![union(left,
/// right)]), ...)` tail) before this file's fold hands it to the
/// kernel. A bare (non-union) singleton also qualifies — `word_of`
/// alone reads it — so a one-member fold (no join ever ran) still
/// converts. Any leaf that is not a length-one `word_of` result (a
/// string tuple, a window, a star) fails the whole recognition: the
/// caller must keep the `Kind::Set` form rather than guess at values
/// that are not actually enumerated.
///
/// `pub(crate)`: `match_arms.rs`'s `MatchValue`/`MatchOr` pattern
/// outcome reuses this exact reading for a `Kind::Set` match subject
/// (`case 1:` over a set that enumerates {1, 2, 4}) rather than writing
/// a second set-enumeration parser.
pub(crate) fn scalars_of_union_of_singletons(set: &refined_sets::refinement_forms::RefinedSet) -> Option<Vec<f64>> {
    if let Some(values) = refined_sets::refinement_forms::word_of(set) {
        if values.len() == 1 {
            return Some(values);
        }
        return None;
    }
    if set.forms.len() != 1 || set.forms[0].form != refined_sets::refinement_forms::Form::Union {
        return None;
    }
    let mut values = scalars_of_union_of_singletons(set.forms[0].a_.as_ref().unwrap())?;
    values.extend(scalars_of_union_of_singletons(set.forms[0].b.as_ref().unwrap())?);
    Some(values)
}

/// The reverse of `known_state_of`: a kernel-answered state back to an
/// `AbstractValue`, at the joined trust grade — only for the plain,
/// flag-free state the two scalar-set rows above ever produce or ask
/// about. `top`/`undef`/`null`/`nan`/`thrown` all being unset is the
/// gate: any flag means the answer left the scalar-set world this fold
/// lives in, and the caller falls back to `join_known` rather than
/// misreading a flagged wire as a plain set.
///
/// The kernel's own wire carries no Python sort tag at all
/// (`lattice_conformance.rs`'s module doc), so its answer is always a
/// bare set shape — but when that shape is a union of singleton
/// scalars (`scalars_of_union_of_singletons`), the ANSWER denotes the
/// same exact values `join_known`'s local numeric-tagged arms would
/// have kept as `Kind::Values`: reading it back that way, tagged with
/// `shared_tag` (the caller's own agreement on both operands' Python
/// sort — `Some(Integer)` when both sides were Integer, `Some(Float)`
/// when both were Float, `None` otherwise), recovers the exact-values
/// representation instead of losing it to a poorer `Kind::Set` — every
/// transfer/min/max/sort-law consumer downstream of a kernel-joined
/// dict read gets the richer shape either way. A leaf that is NOT a
/// union of singletons (a range, a star, a multi-codepoint string
/// tuple) stays `Kind::Set` — there are no enumerated values to lift.
pub(super) fn known_value_of_state(
    state: &KnownStateWire,
    grade: TrustLevel,
    shared_tag: Option<PrimitiveKind>,
) -> Option<AbstractValue> {
    if state.top || state.undef || state.null || state.nan || state.thrown {
        return None;
    }
    if let Some(tag) = shared_tag {
        if let Some(values) = scalars_of_union_of_singletons(&state.set) {
            // The kernel's join keeps both operands' members even when
            // they repeat (`Union({40},{40})`); `join_known`'s own
            // same-sort arms merge with a membership check, so the
            // read-back applies the identical rule.
            let mut merged: Vec<f64> = Vec::with_capacity(values.len());
            for v in values {
                if !merged.iter().any(|kept| *kept == v) {
                    merged.push(v);
                }
            }
            return Some(known_values(merged, tag, grade));
        }
    }
    Some(known_set(state.set.clone(), None, grade, SetKindTag::None))
}

/// The delegate-first fold `dict_key_set_read` folds two dict-entry
/// values through: ask the kernel's proved `join_state`
/// (`kernel_interface.rs`'s `join_state` field, the same entry
/// `lattice_conformance.rs`'s own conformance suite holds
/// `refined_domain::lattice_operations::join_known` to) when both sides
/// convert to a scalar kernel state, and use ITS answer over the local
/// `join_known`. `catch_unwind` turns a kernel panic into a refusal
/// rather than a crash — the same discipline `assignability.rs`/
/// `builtin_models.rs` already hold every kernel ask to. On any
/// refusal — no loaded kernel, a non-convertible operand shape, a
/// flagged answer, or a caught panic — `join_known` runs unchanged as
/// the fallback; this function never weakens what `join_known` alone
/// already proves.
///
/// `shared_tag_of` is the same rule `lattice_operations.rs`'s own
/// same-sorted `join_known` arms follow (finding this fold must not
/// special-case): both operands `Kind::Values` tagged the SAME
/// Integer-or-Float sort keeps that tag; anything else (mixed sorts, a
/// non-Values operand, an already-Set operand) states no shared sort,
/// and `known_value_of_state` then keeps the untagged `Kind::Set`
/// form — Integer ⊔ Float (or Integer ⊔ an unrelated set) is the bare
/// "Number" reading, never one side's tag winning by omission.
fn shared_tag_of(a: &AbstractValue, b: &AbstractValue) -> Option<PrimitiveKind> {
    if a.kind != Kind::Values || b.kind != Kind::Values {
        return None;
    }
    match (a.kind_tag, b.kind_tag) {
        (Some(PrimitiveKind::Integer), Some(PrimitiveKind::Integer)) => Some(PrimitiveKind::Integer),
        (Some(PrimitiveKind::Float), Some(PrimitiveKind::Float)) => Some(PrimitiveKind::Float),
        _ => None,
    }
}

pub(super) fn kernel_joined_set(so_far: AbstractValue, found: AbstractValue) -> AbstractValue {
    let fallback = || join_known(so_far.clone(), found.clone());
    let Some(kernel) = kernel_if_loaded() else {
        return fallback();
    };
    let Some(state_a) = known_state_of(&so_far) else {
        return fallback();
    };
    let Some(state_b) = known_state_of(&found) else {
        return fallback();
    };
    let asked = crate::kernel_ask::ask_kernel(|| (kernel.join_state)(&state_a, &state_b));
    let Ok(joined_state) = asked else {
        return fallback();
    };
    let grade = min_trust_level(trust_level_of(&so_far), trust_level_of(&found));
    let shared_tag = shared_tag_of(&so_far, &found);
    match known_value_of_state(&joined_state, grade, shared_tag) {
        Some(value) => value,
        None => fallback(),
    }
}

/// `container[key]` on a known DICT receiver with a key that is a
/// FINITE UNION of known exact strings (`key = "age" if flag else
/// "years"`'s own joined shape, `Kind::Set` — `lattice_operations
/// ::join_known` of two distinct multi-codepoint exact strings builds
/// exactly this union-of-`string_tuple` form, per that function's own
/// tests). `stdtypes.rst`'s mapping-subscription rule reads a single
/// key; this is the SOUND generalization when every branch's own key
/// names a PRESENT entry: `person[key]` with `key` known to be `"age"`
/// OR `"years"`, and both `person["age"]` and `person["years"]` present,
/// answers the join of the two entries' own values — exactly the value
/// the real subscription reads on whichever branch actually ran.  A key
/// naming any string not present in `keys` declines the whole read
/// (`None`, never a partial/guessed answer) — the same honesty a single
/// missing key already gives `dict_key_read`. `word_tuples_of` is the
/// existing exact-word enumerator `refined_sets::codepoint_sets` already
/// proves against a union-of-`string_tuple` set (the string-equality
/// narrowing rows use the identical reader); a set that is not this
/// union-of-known-words shape (an unbounded range, an unrelated form)
/// answers `None` from `word_tuples_of` itself, and this function
/// declines in step. The fold asks the kernel's `join_state` first
/// (`kernel_joined_set`) when both accumulated values are scalar-set
/// shaped, falling back to the local `join_known` otherwise — see that
/// function's own doc.
pub(super) fn dict_key_set_read(keys: &[ObjectKey], index: &AbstractValue) -> Option<AbstractValue> {
    if index.kind != Kind::Set || index.kind_tag.is_some_and(|tag| tag != PrimitiveKind::String) {
        return None;
    }
    let words = refined_sets::codepoint_sets::word_tuples_of(&index.set)?;
    if words.is_empty() {
        return None;
    }
    let mut joined: Option<AbstractValue> = None;
    for points in words {
        let text: String = points.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        let found = dict_key_read(keys, &DictKey::string(&text))?;
        joined = Some(match joined {
            Some(so_far) => kernel_joined_set(so_far, found),
            None => found,
        });
    }
    joined
}
