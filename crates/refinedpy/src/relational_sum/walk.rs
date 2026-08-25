//! Asking the kernel to walk the lowered program and reading back the
//! slots it wrote.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustLevel;
use refined_kernel::narrow_questions::KnownStateWire;
use refined_kernel::summary_questions::ask_walk_relational;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;

use super::division::DivisionOp;
use super::RecognizedAccumulation;
use super::QUOTIENT_SLOT;
use super::TOTAL_SLOT;

/// What the kernel answered: the accumulated total, and the quotient
/// when a division rode along.
pub struct AccumulationAnswer {
    /// The value the accumulator holds after the loop, when the kernel
    /// answered a bindable set for it. `None` when the total's own
    /// enclosure is honestly unbounded (a sign-straddling step times an
    /// unbounded count) while the LEDGER still proved the quotient —
    /// the two claims are independent in both directions.
    pub total: Option<AbstractValue>,
    /// The value the divided name holds, when a division was folded in
    /// and the kernel answered a bindable set for it.
    pub quotient: Option<AbstractValue>,
    /// The sequence's own length, integer-sorted — the value a
    /// count-alias name (`count = len(samples)`) binds to. Read straight
    /// off `RecognizedAccumulation::count_set`, the same window
    /// `entry_states`'s slot 2 already carried into the kernel ask; this
    /// is not a kernel answer, since the length is a fact this checker
    /// already held before asking (the sequence's own repetition
    /// window), not one the kernel derived. `None` only when that window
    /// itself states nothing bindable (an empty form list) — the same
    /// rule `bindable_state` applies to every other slot.
    pub count: Option<AbstractValue>,
}

/// Asks the kernel to walk the lowered program and reads back the slots
/// it wrote.
///
/// The RELATIONAL walk is what is asked, never the certifying one: the
/// certifying walk drops the linear ledger, so the division would be
/// answered by plain interval arithmetic and the whole lowering would
/// buy nothing (`kernel_interface`'s `walk_relational` states why the
/// two paths are exclusive).
///
/// `None` whenever the kernel declines (`ask_walk_relational`'s own
/// refusal discipline: no kernel loaded, or a question the kernel
/// refuses), or when the TOTAL's own answer is not a plain set this
/// checker can bind — an unknown exit state is an honest refusal, never
/// a set to invent. A folded division whose quotient came back unknown
/// leaves `quotient` as `None` while the total still stands: the two
/// claims are independent, and dropping the good one with the bad would
/// state less than the kernel proved.
pub fn walk_accumulation(recognized: &RecognizedAccumulation) -> Option<AccumulationAnswer> {
    let asked = ask_walk_relational(&recognized.entry_states, &recognized.statements, &[]);
    // Debug instrument (REFINEDPY_DEBUG_RELATIONAL): the raw exit
    // states as the kernel answered them, or the ask's own refusal —
    // the split `check.rs`'s instrument cannot see from one layer up.
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        match &asked {
            None => eprintln!("relational_sum: ask_walk_relational answered None (no kernel, or the ask panicked)"),
            Some(states) => eprintln!("relational_sum: kernel exit states={states:?}"),
        }
    }
    let states = asked?;
    // The total wears the sort its seed pinned (Float for a `0.0`
    // seed; nothing otherwise — recognize_accumulation's own rule).
    let total = states
        .get(TOTAL_SLOT as usize)
        .and_then(|state| bindable_state(state, recognized.grade,
            recognized.total_kind_tag));
    let quotient = match recognized.statements.len() {
        // no division rode along, so the quotient slot was never written
        1 => None,
        _ => states
            .get(QUOTIENT_SLOT as usize)
            .and_then(|state| bindable_state(state, recognized.grade, quotient_kind_tag(recognized))),
    };
    // The count is not a kernel answer — it is the sequence's own
    // repetition window, already read at recognition time
    // (`element_and_count_sets`, stashed as `count_set`) and carried into
    // the kernel ask as slot 2's entry state unchanged. A `len(...)` call
    // yields Python's own `int` (library/functions.html#len), so the
    // sort is Integer-tagged, exactly as `integer_set_bounds` reads back.
    // Empty forms is the one case that states nothing bindable, the same
    // rule every other slot answers under.
    let count = if recognized.count_set.forms.is_empty() {
        None
    } else {
        Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(recognized.count_set.clone(), None, recognized.grade, SetKindTag::None)
        })
    };
    // An answer with NEITHER slot bindable claims nothing; either slot
    // alone still stands — a top total must not drop a proved quotient
    // (the ledger ties the quotient to the count even when the total's
    // own enclosure is unbounded), and the reverse held already. The
    // count rides independently of both: it is known whenever the
    // accumulation recognized at all, so it does not gate this decline.
    if total.is_none() && quotient.is_none() {
        return None;
    }
    Some(AccumulationAnswer { total, quotient, count })
}

/// The sort a folded division's quotient wears, once it binds. Two
/// division operators fold here, and they sort differently
/// (expressions.rst, binary arithmetic): `/` "division of integers
/// yields a float" unconditionally, so `Div` is always Float. `//`
/// keeps the OPERANDS' own sort — the count (`len(...)`) is always
/// Integer-sorted (`walk_accumulation`'s own `count` field), so
/// `FloorDiv` is Integer exactly when the total is ALSO Integer-sorted,
/// and Float the moment either side is Float or the total's sort is
/// unknown (`total_kind_tag` of `None` states no claim, never an
/// assumed Integer). This is the sort the sqrt/floor call rows
/// downstream require before they ask the kernel. Pure function of the
/// recognized program's own fields, called only after a division
/// folded (`recognized.quotient_op.is_some()`); on the `None` case
/// (defensive — every caller already gates on `statements.len() > 1`,
/// which only a fold reaches) it answers Float, matching the operator's
/// own historical default.
pub(super) fn quotient_kind_tag(recognized: &RecognizedAccumulation) -> Option<PrimitiveKind> {
    match recognized.quotient_op {
        Some(DivisionOp::FloorDiv) if recognized.total_kind_tag == Some(PrimitiveKind::Integer) => {
            Some(PrimitiveKind::Integer)
        }
        _ => Some(PrimitiveKind::Float),
    }
}

/// One exit state as a value this checker can bind, or `None` when the
/// kernel claimed nothing about that slot. The sort tag is the caller's
/// claim about the slot's language-level sort; `None` states no sort.
///
/// Three shapes answer `None`, all for the same reason — the exit state
/// states nothing an ordinary unnarrowed read did not already have:
/// `state.top` (no knowledge at all), an empty form list (the void, not
/// a claim), and a non-empty form list that nonetheless admits every
/// finite float plus both infinities (`full_float_range`'s own doc) —
/// a set spanning ℝ̄ in its entirety is exactly as uninformative as top,
/// just spelled with forms instead of the top flag.
pub(super) fn bindable_state(
    state: &KnownStateWire,
    grade: TrustLevel,
    kind_tag: Option<PrimitiveKind>,
) -> Option<AbstractValue> {
    if state.top || state.set.forms.is_empty() || full_float_range(&state.set) {
        return None;
    }
    Some(AbstractValue {
        kind_tag,
        ..known_set(state.set.clone(), None, grade, SetKindTag::None)
    })
}

/// Whether a set's forms admit every element of ℝ̄ — no finite float
/// excluded, and both infinities included. Read straight off the forms'
/// own canonical spelling rather than asking the kernel a subset
/// question: the lower ray reaches exactly `-inf` through a NON-STRICT
/// `AtLeast` (an `Above(-inf)` excludes the single point `-inf` and so
/// is NOT the full range), the upper ray reaches exactly `+inf` through
/// a non-strict `AtMost` OR is absent entirely (no `AtMost`/`Below` form
/// at all also states no upper bound — `refinement_forms::numbers`'s own
/// spelling of ℝ̄ is the lone form `at_least(NEG_INFINITY)`, with no
/// paired upper form), and no OTHER form rides alongside those two —
/// an `Integer`/`MultipleOf`/`OneOf`/etc. alongside a full ray still
/// excludes something, so is not this shape.
///
/// Every ray candidate is read directly rather than through
/// `fold_ray_forms`'s tightest-wins fold: a set built from more than one
/// ray per side is not the shape either bound-set builder in this module
/// or the kernel's own division exit state produces, so reading the
/// forms as given (no fold) states exactly what is there.
fn full_float_range(set: &RefinedSet) -> bool {
    let mut reaches_below = false;
    for form in &set.forms {
        match form.form {
            Form::AtLeast if form.a == f64::NEG_INFINITY => reaches_below = true,
            // present or absent, an at-most-infinity upper form states no
            // ceiling either way — it is simply not read as a constraint
            Form::AtMost if form.a == f64::INFINITY => {}
            _ => return false,
        }
    }
    reaches_below
}

/// The element set and the count set of a sequence value, read off the
/// repetition window its own set states. The count rides as a
/// nonnegative INTEGER-sorted number state — a length is a whole count,
/// never a fraction — bounded below by the window's own `lo` and above
/// by its `hi` when the window states one.
///
/// `None` for a value that is not a `Kind::Set`, a set that is not a
/// repetition this reader can peel, or a repetition whose element set
/// states nothing at all. How TIGHT the element set is, is not gated
/// here: the kernel derives what the relation supports, and a hull it
/// cannot tie answers an unknown exit state that `walk_accumulation`
/// declines on. Gating it twice would put an adapter-side judgment in
/// front of the kernel's own.
pub(super) fn element_and_count_sets(value: &AbstractValue) -> Option<(RefinedSet, RefinedSet)> {
    if value.kind != Kind::Set {
        return None;
    }
    let repetition = as_repetition(&value.set)?;
    if repetition.element.forms.is_empty() {
        return None;
    }
    let mut count_forms = vec![integer(), at_least(repetition.lo as f64)];
    if let Some(hi) = repetition.hi {
        count_forms.push(at_most(hi as f64));
    }
    Some((repetition.element, make_refined_set(count_forms)))
}
