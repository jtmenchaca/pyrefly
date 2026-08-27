//! Numeric builtins: `abs`, `round`, `sum`, `min`/`max` — the single-
//! scalar rows, the known-`Kind::List` iterable rows, the unknown-
//! length star-shaped-iterable rows, and the kernel-asked `Kind::Set`
//! rows. Every row cites its clause of docs.python.org/3.12/library/
//! functions.html; a row with no citation is not written.

use std::sync::Arc;

use refined_domain::abstract_value::{known_set, known_values, nan_value, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::{derived_trust_level, TrustSpec};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::{PowOperandKind, PowOperandWire, TransferAnswerKind, TransferQuestion, TransferQuestionOp};
use refined_sets::refinement_forms::{at_least, at_most, make_refined_set, one_of, Form, RefinedSet};
use refined_sets::repetition_window_forms::as_repetition;

/// Read a single known numeric value out of an argument: `Kind::Values`,
/// tagged `Integer` or `Float`, carrying exactly one element. Every row
/// below that needs "one known number" reads through this rather than
/// re-matching the shape.
pub(super) fn single_known_numeric(argument: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
    if argument.kind != Kind::Values {
        return None;
    }
    if argument.values.len() != 1 {
        return None;
    }
    match argument.kind_tag {
        Some(PrimitiveKind::Integer) => Some((argument.values[0], PrimitiveKind::Integer)),
        Some(PrimitiveKind::Float) => Some((argument.values[0], PrimitiveKind::Float)),
        _ => None,
    }
}

/// `abs(x)` on a single known numeric — library/functions.html#abs:
/// "Return the absolute value of a number." Sort is preserved: an int
/// argument's absolute value is an int, a float's a float — abs never
/// changes the numeric sort of its single argument. `abs(float('nan'))`
/// returns `nan` normally in CPython (no exception, the same posture
/// `math.fabs`/`float_result` keep in math_models.rs), so a NaN operand
/// answers the domain's own NaN state (`nan_value()`) rather than let a
/// bare NaN enter `known_values`, which no refined set admits
/// (`refinement_forms::element`'s own construction-time refusal).
pub(super) fn abs_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, sort) = single_known_numeric(only)?;
    if value.is_nan() {
        return Some(nan_value());
    }
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.abs()], sort, grade))
}

/// `abs(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded set
/// another transfer already produced): the kernel's own `Abs` transfer
/// (`javascript-pins.md` arith.7 — "Math.abs: −0→+0, −∞→+∞, otherwise
/// negates negatives," `theories/binary64/abs.lean`'s `transferAbs` —
/// a range straddling zero folds its lower bound to 0, e.g. `[-2, 1]`
/// answers `[0, 2]`) answers the absolute-valued enclosure directly, the
/// exact mirror of `rounding_call_over_set` (`math_models.rs`) — same
/// `TransferQuestion` construction, same `catch_unwind` refusal
/// discipline, same `TransferAnswerKind` match. Sort is preserved (the
/// same rule `abs_call`'s single-value row keeps): the answer keeps the
/// operand's own Integer/Float tag, never fixed at one sort the way
/// `rounding_call_over_set`'s Integer-only result is. A non-numeric-sorted
/// set, or a kernel refusal on this set shape, declines to `None`.
pub(super) fn abs_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    let sort = match value.kind_tag {
        Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
        Some(PrimitiveKind::Float) => PrimitiveKind::Float,
        _ => return None,
    };
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Abs,
            a: value.set.clone(),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, sort, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(sort),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// `round(x)`, single-argument — library/functions.html#round: "If
/// ndigits is omitted or is None, it returns the nearest integer to its
/// input," rounding "toward the even choice" on a tie (banker's
/// rounding — `round(0.5)` and `round(-0.5)` are both `0`, `round(1.5)`
/// is `2`). The two-argument form `round(x, n)` is not modeled: it keeps
/// the input's sort (int stays int, float stays float) rather than
/// always producing an int, a different row this dispatcher does not
/// yet answer.
pub(super) fn round_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(
        vec![value.round_ties_even()],
        PrimitiveKind::Integer,
        grade,
    ))
}

/// The single numeric value out of a KNOWN `Kind::List` element — the
/// same acceptance `single_known_numeric` gives a bare argument, read
/// off one list slot for `sum`/`min`/`max`'s single-iterable rows.
fn single_known_numeric_element(element: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
    single_known_numeric(element)
}

/// The `{lo, hi}` numeric hull a repetition-window `Kind::Set` iterable's
/// own ELEMENT set admits, EITHER side left unbounded
/// (`f64::NEG_INFINITY`/`f64::INFINITY`) when the element states no ray
/// on that side: `iterable` must be the repetition-window `Kind::Set`
/// shape `check.rs::seed_parameters` builds for a declared
/// `list[X]`/`set[X]`/`Sequence[X]` parameter — bare-star (unbounded) or
/// length-bounded, `as_repetition` reads either window shape uniformly
/// (`collection_models::star_element_read`'s own doc — the same window
/// reading, never a second reader) — and the element set itself must be
/// built ONLY from `AtLeast`/`Above`/`AtMost`/`Below` rays, PLUS the
/// `Integer`/`MultipleOf` markers (`math_models.rs`'s own
/// `enclosure_is_provably_finite` keeps the identical exception: these
/// two forms narrow WHICH values the ray admits but state no bound of
/// their own, so they carry no `lo`/`hi` contribution to fold — an
/// int-sorted element like `Age`'s own `[AtLeast(0), AtMost(120),
/// Integer]` (`check.rs::seed_parameters`'s own scalar seeding, which
/// tags `Integer` onto the outer value's `kind_tag` and states the
/// SAME marker on the element set that produced it) is exactly this
/// shape, not a different one this reader has never walked). An outer
/// sound hull: `Above`/`Below`'s own strict bound still bounds the
/// closed `f64` hull correctly, even though the true infimum/supremum
/// is never attained there — any other element form (a union, `OneOf`,
/// a pattern-compiled set) answers `None` rather than guess a hull for a
/// shape this reader does not walk. Each caller states its own
/// requirement on which side(s) must be finite.
pub(super) fn star_numeric_hull(iterable: &AbstractValue) -> Option<(f64, f64)> {
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    if !matches!(iterable.kind_tag, Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float)) {
        return None;
    }
    let repeated = as_repetition(&iterable.set)?;
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for form in &repeated.element.forms {
        match form.form {
            Form::AtLeast | Form::Above => lo = lo.max(form.a),
            Form::AtMost | Form::Below => hi = hi.min(form.a),
            Form::Integer | Form::MultipleOf => {}
            _ => return None,
        }
    }
    Some((lo, hi))
}

/// `sum(iterable, start=0)` over a known `Kind::List` of known single-
/// numeric elements (a known list literal, or the comprehension/
/// generator shape `evaluate_list_or_set_comp` already builds as a
/// `Kind::List`) — library/functions.html#sum: "Sums *start* and the
/// items of an *iterable* from left to right and returns the total."
/// The two-argument `start=` form threads the caller's own start value
/// (defaulting to Integer 0, matching the doc's own default); any
/// non-numeric element declines the whole call rather than skip it.
/// Sort widens to Float the moment any addend (the start value or any
/// element) is Float-sorted, matching ordinary `+` — the same mixed-
/// arithmetic widening `expressions.rs`'s `binary_arithmetic_value`
/// already applies. An UNKNOWN-LENGTH star-shaped iterable (a declared
/// element set, no concrete items) falls to `sum_call_over_star` instead
/// — this row's own `Kind::List` gate declines it outright, matching the
/// `iterable.kind != Kind::List` guard immediately below.
pub(super) fn sum_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let (iterable, start) = match arguments {
        [iterable] => (iterable, None),
        [iterable, start] => (iterable, Some(start)),
        _ => return None,
    };
    if iterable.kind != Kind::List {
        return None;
    }
    let (mut total, mut all_int) = match start {
        Some(start_value) => {
            let (value, sort) = single_known_numeric(start_value)?;
            (value, sort == PrimitiveKind::Integer)
        }
        None => (0.0, true),
    };
    for element in &iterable.items {
        let (value, sort) = single_known_numeric_element(element)?;
        total += value;
        all_int = all_int && sort == PrimitiveKind::Integer;
    }
    // A running Float total can land on NaN even when no single addend
    // was NaN (`inf + (-inf)`, IEEE 754) — the same accumulation-order
    // hazard `arithmetic_result` in expressions.rs guards for `+`. The
    // domain's own NaN state (`nan_value()`) is the answer, never a
    // bare NaN inside `known_values`, which no refined set admits
    // (`refinement_forms::element`'s own construction-time refusal).
    if total.is_nan() {
        return Some(nan_value());
    }
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    let sort = if all_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    Some(known_values(vec![total], sort, grade))
}

/// `sum(iterable, start=0)` over an UNKNOWN-LENGTH star-shaped iterable
/// (`star_numeric_hull`'s own doc — a declared `list[X]`/`set[X]`/
/// `Sequence[X]` parameter, element set known, no concrete items to
/// walk): the EXACT relational sum (`total <= count * elemHi`, the
/// linear fact the kernel's own decider ties to the count,
/// CROSS-LANGUAGE-EDGE.md §7 K1) is a kernel capability this row does
/// not have — what this row states instead is the sound SIGN envelope a
/// sum of any-length nonnegative or non-positive addends always keeps:
/// library/functions.html#sum's own "sums *start* and the items... from
/// left to right" — adding zero or more nonnegative numbers to `start`
/// can only move the total UP from `start` (answers `[start, +inf)`),
/// and adding zero or more non-positive numbers can only move it DOWN
/// (answers `(-inf, start]`). Only ONE side of the element hull needs
/// to be known to pick a branch (`elemLo >= 0.0` alone is enough for
/// the nonnegative branch, regardless of whether `elemHi` is finite);
/// declines only when the element hull straddles zero (`lo < 0.0 <
/// hi`, sign undetermined) — an unbounded-both-sides element ray
/// (`star_numeric_hull` returning `NEG_INFINITY`/`INFINITY`) also falls
/// into this undetermined case, since neither comparison holds. Sort
/// widens to Float the moment either the start value or the element
/// set is Float-sorted, the same rule `sum_call`'s exact row applies —
/// requires the iterable's own element sort to be KNOWN (Integer or
/// Float; anything else, including an unset sort, declines) so the
/// widening rule always has both addend sorts in hand.
pub(super) fn sum_call_over_star(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let (iterable, start) = match arguments {
        [iterable] => (iterable, None),
        [iterable, start] => (iterable, Some(start)),
        _ => return None,
    };
    let (element_lo, element_hi) = star_numeric_hull(iterable)?;
    let element_sort = match iterable.kind_tag {
        Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
        Some(PrimitiveKind::Float) => PrimitiveKind::Float,
        _ => return None,
    };
    let (start_value, start_sort) = match start {
        Some(start_value) => single_known_numeric(start_value)?,
        None => (0.0, PrimitiveKind::Integer),
    };
    let all_int = start_sort == PrimitiveKind::Integer && element_sort == PrimitiveKind::Integer;
    let sort = if all_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    let window = if element_lo >= 0.0 {
        make_refined_set(vec![at_least(start_value)])
    } else if element_hi <= 0.0 {
        make_refined_set(vec![at_most(start_value)])
    } else {
        return None;
    };
    Some(AbstractValue {
        kind_tag: Some(sort),
        ..known_set(window, None, grade, SetKindTag::None)
    })
}

/// `sorted(iterable)` (no `key=`/`reverse=` keyword arguments) over a
/// known `Kind::List` of known single-numeric elements —
/// library/functions.html#sorted: "Return a new sorted list from the
/// items in *iterable*." Ascending numeric order, matching the
/// no-`key`/no-`reverse` default row; a non-numeric element declines
/// the whole call.
pub(super) fn sorted_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::List {
        return None;
    }
    let mut pairs: Vec<(f64, PrimitiveKind)> = Vec::with_capacity(iterable.items.len());
    for element in &iterable.items {
        pairs.push(single_known_numeric_element(element)?);
    }
    // A NaN element makes every comparison false (expressions.rst's
    // ordering rules), so CPython's sort produces an order no law
    // states — a NaN-admitting list yields no order claim, and this
    // arm declines rather than fabricate one (float("nan") is a value
    // float_call now constructs).
    if pairs.iter().any(|(value, _)| value.is_nan()) {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN elements declined above"));
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    let sorted_items: Vec<AbstractValue> = pairs.into_iter().map(|(value, sort)| known_values(vec![value], sort, grade)).collect();
    Some(known_list(sorted_items, grade))
}

/// `sorted(iterable)` / `reversed(sequence)` over an UNKNOWN-LENGTH,
/// known-element receiver — a `Kind::Set` whose only form is a
/// repetition window (`as_repetition`, the shape a declared
/// `list[X]`/`Sequence[X]` parameter seeds and the shape
/// `attribute.rs`'s `sys.argv` read answers). The exact ORDER is
/// unstated once the elements themselves are unread, but both calls'
/// own clauses pin the two facts this domain reads a sequence through:
///
/// - `sorted(iterable)`: "Return a new sorted list from the items in
///   *iterable*" (library/functions.rst) — the result holds exactly the
///   ITEMS of `iterable`, reordered, so every element stays inside the
///   receiver's own alphabet and the count is unchanged.
/// - `reversed(seq)`: "Return a reverse iterator" (the same file) — a
///   reversal is a permutation, so the identical two facts hold.
///
/// The answer is therefore the receiver's own repetition window,
/// unchanged: same element set, same `{lo, hi}` length window. This is
/// exact for the length and for element membership; the POSITION of any
/// particular element is what neither call states over an unread
/// receiver, and a repetition window makes no positional claim to lose.
///
/// `None` when the receiver is not a repetition window, so the caller's
/// own exact-list rows and final decline stand unchanged.
pub(super) fn order_preserving_over_star(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    as_repetition(&iterable.set)?;
    Some(iterable.clone())
}

/// `reversed(sequence)` over a known `Kind::List` —
/// library/functions.html#reversed: "Return a reversed iterator... *seq*
/// must be an object which has a `__reversed__()` method or supports the
/// sequence protocol." This domain has no separate iterator Kind (the
/// same eager-materialization choice `range_expression_value`/
/// `sorted_call` already make for their own lazy/ordered results), so the
/// answer is the elements in reverse positional order, still a
/// `Kind::List` — `A7.xfer.reduce.py`'s own `reversed([a, b, c])` row,
/// `functools.reduce`'s fold over the reversed sequence. `None` for any
/// non-`Kind::List` argument, letting the caller's own decline stand.
pub(super) fn reversed_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::List {
        return None;
    }
    let reversed_items: Vec<AbstractValue> = iterable.items.iter().rev().cloned().collect();
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    Some(known_list(reversed_items, grade))
}

/// `min`/`max` over a SINGLE known `Kind::List` iterable argument —
/// library/functions.html#min/#max: "If one positional argument is
/// provided, it should be an iterable... the largest [smallest] item
/// in the iterable is returned." An empty iterable has no row here:
/// CPython raises `ValueError` on an empty sequence with no `default=`
/// keyword, which this file has no exception channel for this wave —
/// this row declines on an empty list rather than answer a fabricated
/// value. An UNKNOWN-LENGTH star-shaped iterable falls to
/// `min_max_over_star` instead — this row's own `Kind::List` gate
/// declines it outright.
pub(super) fn min_max_over_iterable(arguments: &[AbstractValue], pick: fn(f64, f64) -> bool) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::List || iterable.items.is_empty() {
        return None;
    }
    let mut best: Option<(f64, PrimitiveKind)> = None;
    for element in &iterable.items {
        let candidate = single_known_numeric_element(element)?;
        best = Some(match best {
            None => candidate,
            Some(current) => if pick(candidate.0, current.0) { candidate } else { current },
        });
    }
    let (value, sort) = best?;
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    Some(known_values(vec![value], sort, grade))
}

/// `min`/`max` over a SINGLE UNKNOWN-LENGTH star-shaped iterable
/// (`star_numeric_hull`'s own doc — a declared `list[X]`/`set[X]`/
/// `Sequence[X]` parameter, element set known, no concrete items to
/// walk): library/functions.html#min/#max's own "the largest [smallest]
/// item in the iterable" — every item the real call could draw is a
/// value FROM the element set (the grammar's own definition, the same
/// fact `star_element_read` reads a subscript through), so the element
/// set's own numeric hull `[elemLo, elemHi]` is sound for BOTH `min`
/// and `max`: the true minimum/maximum is some element, and every
/// element sits inside `[elemLo, elemHi]` by construction. Requires a
/// NONEMPTY window (`repeated.lo >= 1`, `as_repetition`'s own `{lo,
/// hi}`): CPython raises `ValueError` on an empty sequence with no
/// `default=` keyword (the same exception channel `min_max_over_iterable`
/// has none of), so a window that COULD be empty (`lo == 0`) declines
/// here exactly as an empty concrete list declines above. Requires BOTH
/// hull sides finite (`star_numeric_hull` returning an infinite side
/// means the element set states no bound on that side, so no hull value
/// is sound to answer) — unlike `sum_call_over_star`, which only ever
/// needs ONE sign-determining side.
pub(super) fn min_max_over_star(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    let repeated = as_repetition(&iterable.set)?;
    if repeated.lo < 1 {
        return None; // the window could be empty; min/max would raise
    }
    let (element_lo, element_hi) = star_numeric_hull(iterable)?;
    if !element_lo.is_finite() || !element_hi.is_finite() {
        return None;
    }
    let sort = match iterable.kind_tag {
        Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
        Some(PrimitiveKind::Float) => PrimitiveKind::Float,
        _ => return None,
    };
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    // the returned window must carry its own `Integer` FORM, not just the
    // outer value's `kind_tag`, when the element sort is Integer — the
    // SAME "a Set answer must additionally carry its own integrality"
    // discipline `int_transfer_answer`'s own doc states for a kernel-
    // answered enclosure (`expressions.rs`): assignability's containment
    // ask (`scalar_subset`) reads the SET's own forms against `Age`'s
    // declared set (which itself carries `Integer`), never the outer
    // `kind_tag` alone — an untagged `[0, 120]` window is not provably a
    // subset of `[0, 120] && integer` even though every element the star
    // grammar admits genuinely is one, since the kernel has no way to
    // read that fact off `kind_tag`.
    let mut forms = vec![at_least(element_lo), at_most(element_hi)];
    if sort == PrimitiveKind::Integer {
        forms.push(refined_sets::refinement_forms::integer());
    }
    let window = make_refined_set(forms);
    Some(AbstractValue {
        kind_tag: Some(sort),
        ..known_set(window, None, grade, SetKindTag::None)
    })
}

/// `min`/`max` over EXACTLY two arguments where at least one is a known
/// NaN (`Kind::NaN` — `float("nan")`'s own value, `assignability.rs`'s
/// `Kind::NaN` arm) — library/functions.html#min and #max's own
/// algorithm (CPython's `min`/`max` walk the arguments left to right,
/// keeping the current winner unless a LATER candidate compares
/// STRICTLY past it — `candidate > current` for `max`, `candidate <
/// current` for `min`): every comparison AGAINST NaN is `False`
/// (IEEE 754), so neither operator's own strict-inequality test ever
/// proves the second argument the new winner, whichever position NaN
/// sits in. The FIRST argument is therefore always kept, position-
/// dependent — `max(0.5, nan)` answers `{0.5}` exactly (the second
/// argument, nan, never displaces it), while `max(nan, 0.5)` answers
/// NaN itself (`0.5 > nan` is `False`, so the first argument, nan,
/// stays). `min` and `max` read IDENTICALLY here: the "first argument
/// wins" behavior comes from EVERY comparison against NaN failing, not
/// from which operator is asked, so this one row serves both — the
/// `pick` closure `min_max_call` takes is never consulted. Declines
/// (`None`) for anything but exactly two arguments (Python's own
/// varargs `min`/`max` walks left to right the same way for three or
/// more, but this corpus's own NaN rows are all two-argument, and a
/// three-plus-argument fold over a NaN operand deserves its own pinned
/// row before this one's scope widens to it) or neither argument being
/// `Kind::NaN` (the ordinary `min_max_call` row already reads that
/// shape).
pub(super) fn min_max_call_with_nan_operand(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [first, second] = arguments else { return None };
    if first.kind != Kind::NaN && second.kind != Kind::NaN {
        return None;
    }
    Some(first.clone())
}

/// `min`/`max` over two or more known single-numeric arguments —
/// library/functions.html#min and #max: "If two or more positional
/// arguments are provided, the smallest [largest] of the positional
/// arguments is returned." The single-iterable form (`min(some_list)`)
/// is not modeled here — that argument is not a known scalar, so
/// `single_known_numeric` declines it and the whole call declines.
/// Result sort: Python's min/max return the winning ARGUMENT unchanged,
/// so a Float argument winning over Integer arguments keeps Float — the
/// winning value's own sort is threaded through, not fixed at one sort.
pub(super) fn min_max_call(
    arguments: &[AbstractValue],
    pick: fn(f64, f64) -> bool,
) -> Option<AbstractValue> {
    if arguments.len() < 2 {
        return None;
    }
    let mut best: Option<(f64, PrimitiveKind)> = None;
    for argument in arguments {
        let candidate = single_known_numeric(argument)?;
        best = Some(match best {
            None => candidate,
            Some(current) => {
                if pick(candidate.0, current.0) {
                    candidate
                } else {
                    current
                }
            }
        });
    }
    let (value, sort) = best?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value], sort, grade))
}

/// A `min`/`max` scalar-form operand read as a `{RefinedSet, sort}`
/// pair, the same two-shape acceptance `expressions.rs`'s
/// `transferable_numeric_operand` gives a kernel arithmetic transfer: a
/// known single numeric (`single_known_numeric`) reads as the
/// one-element set `{v}` (`one_of`, panics on NaN — caught by this
/// row's own `catch_unwind`, never reached in practice since this
/// domain's numeric literals/computed values are never constructed as
/// NaN), and a numeric-sorted `Kind::Set` (a seeded parameter range, or
/// a bounded set another transfer produced) reads as its own set
/// directly. `None` for every other shape.
fn min_max_scalar_operand(value: &AbstractValue) -> Option<(RefinedSet, PrimitiveKind)> {
    if let Some((v, sort)) = single_known_numeric(value) {
        return Some((make_refined_set(vec![one_of(&[v])]), sort));
    }
    if value.kind == Kind::Set {
        let sort = match value.kind_tag {
            Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
            Some(PrimitiveKind::Float) => PrimitiveKind::Float,
            _ => return None,
        };
        return Some((value.set.clone(), sort));
    }
    None
}

/// `min`/`max` over two-or-more arguments where AT LEAST ONE is a
/// `Kind::Set` (a seeded parameter range, or a bounded set another
/// transfer produced) rather than a known scalar — `min_max_call`'s own
/// `single_known_numeric` gate declines this shape outright, so this
/// row asks the kernel's `Min`/`Max` transfer instead, folding left
/// pairwise across `arguments` (`binary64.min`/`binary64.max`,
/// `transfer_questions.rs:95-96`) — the same `TransferQuestion`
/// construction, `catch_unwind` refusal discipline, and
/// `TransferAnswerKind` match `sqrt_call_over_set`
/// (`math_models.rs`) uses.
///
/// NaN DISCHARGE, read before wiring this row (python-pins.md cmp.2,
/// cmp.16): cmp.2 states "any ordered comparison of a number to a
/// not-a-number value is false," which makes CPython's own sequential-
/// comparison `min`/`max` (cmp.16: first-encountered-wins on ties)
/// answer an OPERAND-ORDER-DEPENDENT result the moment any argument is
/// NaN — not a value `binary64.min`/`binary64.max`'s own (order-
/// independent) semantics can be trusted to match. This row never asks
/// the kernel on an operand that could BE NaN: every `Kind::Set`
/// operand reaches this function built only from `AtLeast`/`Above`/
/// `AtMost`/`Below`/`OneOf`/`Union` forms, each of which refuses NaN at
/// construction (`refinement_forms.rs`'s `element` helper, "NaN is not
/// an element of ℝ̄") — so a Set operand is NaN-free by construction,
/// never by a runtime check this row would otherwise need to perform.
/// A known scalar operand is read through `one_of`, which shares the
/// same NaN-refusing construction; the `catch_unwind` below turns that
/// refusal into an honest `None` rather than a crash, for the
/// unreached case where a NaN scalar somehow reaches this call.
pub(super) fn min_max_call_over_sets(
    arguments: &[AbstractValue],
    op: TransferQuestionOp,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if arguments.len() < 2 {
        return None;
    }
    if !arguments.iter().any(|argument| argument.kind == Kind::Set) {
        return None; // min_max_call's own known-scalar path already owns this case
    }
    let mut operands = Vec::with_capacity(arguments.len());
    for argument in arguments {
        operands.push(min_max_scalar_operand(argument)?);
    }
    let grade = derived_trust_level(TrustSpec, arguments);
    let (mut acc_set, first_sort) = operands[0].clone();
    let mut all_int = first_sort == PrimitiveKind::Integer;
    for (next_set, next_sort) in &operands[1..] {
        all_int = all_int && *next_sort == PrimitiveKind::Integer;
        let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
        let asked = crate::kernel_ask::ask_kernel(|| {
            (kernel.transfer)(&TransferQuestion {
                op,
                a: acc_set.clone(),
                b: next_set.clone(),
                c: 0.0,
                base: nan_operand.clone(),
                exp: nan_operand,
            })
        })
        .ok()?;
        match asked.kind {
            TransferAnswerKind::Values => {
                acc_set = make_refined_set(vec![one_of(&asked.values)]);
            }
            TransferAnswerKind::Set => {
                acc_set = asked.set;
            }
            TransferAnswerKind::NaN | TransferAnswerKind::Unknown => return None,
        }
    }
    let sort = if all_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    Some(AbstractValue {
        kind_tag: Some(sort),
        ..known_set(acc_set, None, grade, SetKindTag::None)
    })
}
