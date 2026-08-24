/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Calls to Python builtins with determinable results, answered exactly.
//! Two dispatchers: `builtin_call_result` (pure Rust, no kernel) and
//! `builtin_call_result_with_kernel` (the caller's actual entry point —
//! tries the pure dispatcher first, then the row families that need a
//! kernel ask: `min`/`max` over a Set operand, and `abs` over a Set
//! operand). Both take the callee name and the already-evaluated
//! argument values; `None` means "not modeled here" (the caller
//! declines honestly), `Some` is an exact answer. Every modeled row
//! cites its clause of docs.python.org/3.12/library/functions.html or
//! library/stdtypes.html (the container constructors `list`/`set`/
//! `dict` live in stdtypes.html's own class entries); a row with no
//! citation is not written.

use std::sync::Arc;

use refined_domain::abstract_value::{float_sorted_unknown, known_set, known_values, nan_value, null_value, opaque_value, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::{derived_trust_level, TrustSpec};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::{PowOperandKind, PowOperandWire, TransferAnswerKind, TransferQuestion, TransferQuestionOp};
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::{at_least, at_most, make_refined_set, one_of, Form, RefinedSet};
use refined_sets::repetition_window_forms::as_repetition;

/// Read a single known numeric value out of an argument: `Kind::Values`,
/// tagged `Integer` or `Float`, carrying exactly one element. Every row
/// below that needs "one known number" reads through this rather than
/// re-matching the shape.
fn single_known_numeric(argument: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
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
fn abs_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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
/// exact mirror of `floor_call_over_set` (`math_models.rs`) — same
/// `TransferQuestion` construction, same `catch_unwind` refusal
/// discipline, same `TransferAnswerKind` match. Sort is preserved (the
/// same rule `abs_call`'s single-value row keeps): the answer keeps the
/// operand's own Integer/Float tag, never fixed at one sort the way
/// `floor_call_over_set`'s Integer-only result is. A non-numeric-sorted
/// set, or a kernel refusal on this set shape, declines to `None`.
fn abs_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
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
fn round_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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
fn star_numeric_hull(iterable: &AbstractValue) -> Option<(f64, f64)> {
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
fn sum_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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
fn sum_call_over_star(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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
fn min_max_over_iterable(arguments: &[AbstractValue], pick: fn(f64, f64) -> bool) -> Option<AbstractValue> {
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
fn min_max_over_star(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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

/// `sorted(iterable)` (no `key=`/`reverse=` keyword arguments) over a
/// known `Kind::List` of known single-numeric elements —
/// library/functions.html#sorted: "Return a new sorted list from the
/// items in *iterable*." Ascending numeric order, matching the
/// no-`key`/no-`reverse` default row; a non-numeric element declines
/// the whole call.
fn sorted_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
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

/// `list(iterable)` — library/stdtypes.rst's `class:: list([iterable])`
/// constructor row: "Lists may be constructed... using the type
/// constructor `list()` or `list(iterable)`." A known `Kind::List`
/// argument copies through unchanged (`list`/`tuple`/`set` all share
/// this domain's one `Kind::List` shape, per `collection_models.rs`'s
/// own module doc — `list(some_set)` and `list(some_tuple)` both read
/// through this same row). A `dict.fromkeys(...)` ROUND-TRIP CARRIER
/// argument (`dict_fromkeys_call`'s own doc, `A15.xfer.dedupe`'s
/// `list(dict.fromkeys(xs))` shape) is unwrapped through `dict_fromkeys_
/// keys_view` FIRST, before the `Kind::List` gate below (a carrier is
/// `Kind::Object`, never `Kind::List`, so the two arms never both fire
/// on the same argument).
fn list_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if let Some(keys_view) = dict_fromkeys_keys_view(iterable) {
        return Some(keys_view);
    }
    if iterable.kind != Kind::List {
        return None;
    }
    Some(known_list(iterable.items.clone(), derived_trust_level(TrustSpec, arguments)))
}

/// The `kind_word` tagging a `dict.fromkeys(iterable, value=None)`
/// ROUND-TRIP CARRIER value (`dict_fromkeys_call`'s own doc) — the same
/// "`Kind::Object` plus a distinguishing word plus a payload in `inner`"
/// idiom `json_grammar.rs::JSON_DUMPS_ROUND_TRIP_WORD` and
/// `env.rs::retained_callable_value` both already use.
const DICT_FROMKEYS_WORD: &str = "the keys view of a dict.fromkeys(...) call";

/// `dict.fromkeys(iterable, value=None)` — library/stdtypes.rst's
/// `classmethod:: fromkeys(iterable, value=None, /)`: "Create a new
/// dictionary with keys from *iterable* and values set to *value*...
/// *value* defaults to `None`." This domain's `dict` is `Kind::Object`
/// with a CLOSED, string-named `keys` list (`collection_models.rs`'s
/// own module doc) — it cannot represent a dict whose keys are an
/// unbounded-count, windowed-VALUE set (`xs: list[int]`'s own element
/// window, not a finite set of string names), so this row does not
/// build a real `Kind::Object` dict at all. Modeled ONLY for the shape
/// the corpus needs a value for — `iterable` a `Kind::Set` repetition
/// window (`as_repetition`, the same shape `star_numeric_hull`/
/// `min_max_over_star` already read for a `list[int]`-typed parameter)
/// — and answers a ROUND-TRIP CARRIER (`Kind::Object`, `DICT_FROMKEYS_
/// WORD`, the iterable's own repetition set carried in `inner`) rather
/// than a real dict value: this file's own callers only ever consume a
/// `fromkeys(...)` result through `list(...)`
/// (`dict_fromkeys_keys_view`, `A15.xfer.dedupe`'s own row), never by
/// reading a key/value pair directly, so carrying the iterable through
/// unread is the exact answer for that one consumption path rather than
/// building (and immediately discarding) machinery for dict reads this
/// corpus never exercises. `value` (defaulting to `None`) is not
/// modeled — the DEDUPED KEYS are the only fact `list(dict.fromkeys(xs))`
/// ever needs; a caller that goes on to read a VALUE out of the result
/// finds no dict shape here and declines honestly.
fn dict_fromkeys_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let iterable = match arguments {
        [iterable] => iterable,
        [iterable, _value] => iterable,
        _ => return None,
    };
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    as_repetition(&iterable.set)?;
    Some(AbstractValue {
        inner: Some(Box::new(iterable.clone())),
        ..opaque_value(DICT_FROMKEYS_WORD)
    })
}

/// `list(dict.fromkeys(xs))`'s own value: the DISTINCT elements of
/// `xs`, in insertion order (Python's `dict` preserves insertion order,
/// library/stdtypes.rst's own "Mapping Types — dict" guarantee,
/// `json_grammar.rs`'s identical citation for the same fact) — drawn
/// from the SAME element window `xs` itself carries (dedup drops
/// duplicates, never introduces a new element outside `xs`'s own
/// alphabet), at a count anywhere from `0` (every element could
/// collide down to one, or `xs` could already be empty) up to `xs`'s
/// own upper length bound (dedup never GROWS a sequence). Rebuilds the
/// SAME repetition shape `xs` itself carries (`as_repetition`/
/// `repeat_of`, the identical window a plain `list[int]` parameter
/// already flows through `loops.rs`'s own `for`-loop reader), with
/// `lo` relaxed to `0` and `hi` unchanged — so `for x in
/// list(dict.fromkeys(xs)): ...` binds `x` to exactly `xs`'s own
/// element set through the SAME existing reader, no new loop machinery
/// needed. `None` for any argument that is not a `dict_fromkeys_call`
/// carrier (`dict_fromkeys_call`'s own doc on the one-consumer scope).
fn dict_fromkeys_keys_view(argument: &AbstractValue) -> Option<AbstractValue> {
    if argument.kind != Kind::Object || argument.kind_word != Some(DICT_FROMKEYS_WORD) {
        return None;
    }
    let iterable = argument.inner.as_deref()?;
    let repeated = as_repetition(&iterable.set)?;
    let deduped_set = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(repeated.element, 0, repeated.hi)]);
    Some(AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(deduped_set, None, derived_trust_level(TrustSpec, &[iterable.clone()]), SetKindTag::None)
    })
}

/// `set([iterable])` — library/stdtypes.rst's `class:: set([iterable])`
/// constructor row: "Return a new set... object whose elements are
/// taken from *iterable*." This domain has no dedicated set Kind (the
/// same `Kind::List` shape a list/tuple carries, per
/// `collection_models.rs`'s own module doc — a set's own element-
/// uniqueness is invisible to any reader that only ever consumes the
/// sequence via `len()`/iteration, matching that file's list/set-comp
/// note). The BARE zero-argument form `set()` — the brackets in the
/// doc's own signature mark the argument optional — answers the empty
/// list directly (an empty set has no elements to dedupe); the
/// one-argument form is `list_constructor_call` under a different name;
/// deduplication is NOT modeled for the one-argument form (an already-
/// List argument is assumed unique-enough for this file's callers,
/// since a set LITERAL display is not what feeds this row — only an
/// already-list-shaped iterable is).
fn set_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if arguments.is_empty() {
        return Some(known_list(Vec::new(), TrustSpec));
    }
    list_constructor_call(arguments)
}

/// `dict(pairs)` — one positional argument, an iterable of `(key,
/// value)` 2-element pairs — library/stdtypes.rst's `class:: dict(...)`
/// constructor row: "dict(iterable, **kwargs)... Dictionaries can be
/// created by... providing an iterable of key/value pairs, including
/// tuples: `dict([('foo', 100), ('bar', 200)])`." Modeled ONLY when
/// `pairs` is a known `Kind::List` of known `Kind::List` 2-element
/// pairs whose first slot is a known exact string (this domain's
/// dict's own string-keyed-only restriction, `collection_models.rs`'s
/// module doc) — anything else declines. A repeated key keeps the LAST
/// value, matching the same overwrite rule `dict_literal_value` and
/// the `dict(...)` constructor doc both state.
fn dict_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [pairs] = arguments else { return None };
    // `dict(<existing dict>)` — the copy-constructor form ("providing
    // ... another dictionary", the same class:: dict(...) row): a known
    // Kind::Object argument answers a fresh dict with the same entries.
    if pairs.kind == Kind::Object && pairs.kind_word.is_none() {
        return Some(pairs.clone());
    }
    if pairs.kind != Kind::List {
        return None;
    }
    let mut keys: Vec<Option<crate::collection_models::DictKey>> = Vec::with_capacity(pairs.items.len());
    let mut values: Vec<AbstractValue> = Vec::with_capacity(pairs.items.len());
    for pair in &pairs.items {
        if pair.kind != Kind::List || pair.items.len() != 2 {
            return None;
        }
        let key = &pair.items[0];
        if key.kind != Kind::Values || key.kind_tag != Some(PrimitiveKind::String) {
            return None;
        }
        let key_text: String = key.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        keys.push(Some(crate::collection_models::DictKey::string(&key_text)));
        values.push(pair.items[1].clone());
    }
    // dict_literal_value's own last-value-wins overwrite rule handles a
    // repeated key exactly the way this constructor's own cited row
    // does — this file reaches into collection_models.rs for the one
    // shared building block rather than duplicating that merge loop
    Some(crate::collection_models::dict_literal_value(&keys, &values))
}

/// `iter(object)` (one-argument form, no `sentinel`) — library/functions.html#iter:
/// "Return an iterator object... *object* must be a collection object
/// which supports the iterable protocol." This domain has no separate
/// iterator Kind: an iterator over a known `Kind::List` reads through
/// as the SAME list value (the one shape a caller ever inspects it
/// through — `next_call`'s own row below), matching the module's
/// shared list/set/generator representation
/// (`collection_models.rs`'s own module doc). Any other receiver
/// shape declines.
fn iter_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::List {
        return None;
    }
    Some(only.clone())
}

/// `next(iterator)` (one-argument form, no `default`) — library/functions.html#next:
/// "Retrieve the next item from the iterator by calling its
/// `__next__` method." Modeled ONLY for the `iter_call`-shaped receiver
/// (a known `Kind::List` standing in for its own iterator, per that
/// function's own doc) AND a generator call's own answer
/// (`Kind::List` tagged `source == "generator"`,
/// `instances::generator_yields`'s own doc — a same-module generator
/// `def`'s call answers the ordered List of every yielded value): the
/// FIRST element is the first item `__next__` would ever produce off a
/// freshly-built iterator or a freshly-called generator. An EMPTY list
/// provably raises `StopIteration` ("If *default* is given, it is
/// returned if the iterator is exhausted, otherwise `StopIteration` is
/// raised") — this row declines on an empty receiver rather than answer
/// a fabricated element; the raise itself is `provable_raise`'s own
/// business, not this dispatcher's.
///
/// SCOPE: this domain carries no per-call exhaustion/position state — a
/// generator-tagged List is a fixed VALUE (the full yield sequence),
/// not a stateful cursor, so `next_call` cannot tell "the first read of
/// this generator" apart from "a second read of the SAME already-
/// advanced generator." Every corpus row this file serves calls `next`
/// exactly once per freshly-constructed generator/iterator value
/// (`next(some_gen())`, never `next(g); next(g)` on one bound name), so
/// this row is honest for that shape; a second `next()` against the
/// SAME generator value would answer element 0 again rather than
/// element 1, which is a known gap this file does not claim to close.
fn next_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::List {
        // A generator call whose body `instances::generator_yields`
        // declined to summarize answers an Unknown tagged
        // `source == "generator-declined"` (`expressions::evaluate_call`'s
        // own generator-call arm) rather than a List — `next(it)` on
        // THAT receiver still has no element to answer, but the tag
        // itself must survive the call so `check.rs::
        // name_unmodeled_call_sentence`'s generator rung can trace a
        // later blocked read (`first = next(it); return first`) back to
        // the generator body that was never summarized, instead of the
        // generic "value not readable" wording. Any other non-List,
        // non-tagged receiver keeps declining outright — this is not a
        // general "next answers Unknown" widening, only the one tag's
        // own onward carry.
        if only.kind == Kind::Unknown && only.source == "generator-declined" {
            return Some(only.clone());
        }
        return None;
    }
    only.items.first().cloned()
}

/// `anext(async_iterator)` (one-argument form, no `default`) — the
/// `async`-generator twin of `next(iterator)`: library/functions.html
/// documents `anext` as `next`'s async counterpart. `await anext(gen)`
/// evaluates through `evaluate_expression`'s own `Expr::Await` arm
/// (transparent unwrap — `async`/`await` carry no gate of their own,
/// matching this file's asyncio.gather doc's identical note), so the
/// `anext(...)` call itself lands in this dispatcher exactly like a
/// plain `next(...)` call would. An async generator's yielded elements
/// are the SAME `Kind::List` (tagged `source == "generator"`,
/// `instances::generator_yields`'s own doc) a sync generator's call
/// answers — `datamodel.rst`'s generator-iterator protocol makes no
/// distinction between a sync and an async generator's own yielded
/// VALUES, only in how the caller RECEIVES them (`__anext__` returns
/// an awaitable rather than the value directly) — so this row is
/// `next_call` under a different name, not a separate reading.
fn anext_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    next_call(arguments)
}

/// `typing.cast(typ, val)` — `Lib/typing.py`'s own `cast` docstring:
/// "This returns the value unchanged. To the type checker this signals
/// that the return value has the designated type, but at runtime we
/// intentionally don't check anything." `typ` is never read (a type
/// expression, not a value this file evaluates); `val` passes through
/// exactly, whatever shape it is — the identity function over its
/// second argument.
fn cast_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [_typ, val] = arguments else { return None };
    Some(val.clone())
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
fn min_max_call(
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
fn min_max_call_over_sets(
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

/// `int(x)` — library/functions.html#int: "For floating-point numbers,
/// this truncates towards zero." An already-Integer argument is the
/// identity read under this row (the same trunc-toward-zero rule with
/// no fractional part to discard). A known EXACT STRING parses through
/// `parse_base_ten_int_string` — the base-10 `int(string, base=10)`
/// row (functions.rst): j-stdlib-surfaces.py's own `int_parse`,
/// `int("40")`/`int("200")`, both exact parses this row now answers
/// precisely rather than declining. A string that does not parse as a
/// base-10 integer (`int("abc")`) still declines HERE — CPython raises
/// `ValueError` for it, which `expressions.rs`'s own `call_provable_
/// raise` speaks through the raise channel (its own `is_valid_base_
/// ten_int_string` gate, a parallel/duplicate validity check to this
/// row's own `parse_base_ten_int_string` — the two files stay
/// independent per the mission's own file-ownership split, so the
/// validity rule is written twice rather than shared across the
/// boundary).
fn int_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        let text: String = only.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        let parsed = parse_base_ten_int_string(&text)?;
        let grade = derived_trust_level(TrustSpec, arguments);
        return Some(known_values(vec![parsed], PrimitiveKind::Integer, grade));
    }
    let (value, _sort) = single_known_numeric(only)?;
    // `int(float('nan'))` RAISES `ValueError: cannot convert float NaN
    // to integer` in CPython (library/functions.html#int delegates to
    // `__trunc__`, and `float.__trunc__` raises on a non-finite operand
    // — the same domain gate `math_models.rs`'s `integral_domain_admits`
    // documents for `math.floor`/`ceil`/`trunc`). No value is returned,
    // so this declines outright rather than answer a value the real
    // call never produces.
    if !value.is_finite() {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.trunc()], PrimitiveKind::Integer, grade))
}

/// `int(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded set
/// another transfer already produced — e.g. `int(math.sqrt(x))`,
/// `math.sqrt`'s own Float-sorted enclosure over a declared parameter
/// range, `math_models.rs`'s `sqrt_call_over_set`): `int_call`'s own
/// row only reads a single known numeric value
/// (`single_known_numeric`), so a Set-shaped argument declines there
/// with no further attempt. This asks the kernel's own `Trunc`
/// transfer directly — the exact mirror of `abs_call_over_set` above
/// (same `TransferQuestion` construction, same `catch_unwind` refusal
/// discipline, same `TransferAnswerKind` match) — library/
/// functions.html#int: "For floating-point numbers, this truncates
/// towards zero," the same trunc-toward-zero rule `int_call`'s
/// single-value row already applies, here posed to `binary64.trunc`
/// (`TransferQuestionOp::Trunc`) instead of computed locally. Unlike
/// `abs_call_over_set` (which preserves the operand's own sort), the
/// result is Integer sort UNCONDITIONALLY — `int(x)` always returns an
/// `int` regardless of its argument's sort, the same rule `int_call`'s
/// own `known_values(..., PrimitiveKind::Integer, ...)` return states.
///
/// A kernel-answered enclosure NOT provably finite
/// (`enclosure_is_provably_finite` false — e.g. `binary64.trunc` over a
/// bare unbounded `float` parameter's own `numbers()` seed,
/// `float_sorted_unknown`'s own doc) does not decline outright: the
/// same non-finite gate `int_call`'s single-value row keeps
/// (`int(float('nan'))`/`int(float('inf'))` both RAISE `ValueError`/
/// `OverflowError` in CPython, never returning a value) rules out ONLY
/// the two non-finite INPUTS, not every finite input the enclosure also
/// admits — those still truncate to SOME integer, so the WEAKER but
/// still TRUE claim over the non-raising outcomes is the unbounded
/// Integer sort (`int_image`'s own image — every row `int(...)`
/// returns at all is an int, library/functions.html#int), not `None`.
/// Answering `None` here left `n = int(x)` for a bare `float`
/// parameter's own guard branches Unknown downstream — one undetermined
/// branch that then poisons a whole function's derived return cases
/// (D5's own `clamp_to_age` helpers, ISSUES.md's fact-export trace).
/// `int_call`'s own single-VALUE row keeps declining outright on a
/// non-finite operand (unchanged): that row reads ONE concrete number,
/// which either raises or does not — there is no "other outcomes" to
/// weaken to when the whole operand IS the non-finite value itself, the
/// same distinction `domain_raise_served_half_value`'s own "straddling
/// vs. entirely-raising" split keeps for a domain-limited math family.
fn int_call_over_set(value: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean) | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Trunc,
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
        TransferAnswerKind::Values => {
            if !asked.values.iter().all(|v| v.is_finite()) {
                return None;
            }
            Some(known_values(asked.values, PrimitiveKind::Integer, grade))
        }
        TransferAnswerKind::Set => {
            if !enclosure_is_provably_finite(&asked.set) {
                // the finite outcomes still all truncate to an int —
                // `int_image`'s own unbounded Integer ray, the weaker
                // TRUE claim over the non-raising half of this operand
                // (this function's own doc above)
                return int_image();
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(asked.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// Whether a set the kernel answered describes only FINITE values — the
/// set-shaped twin of `is_finite`, for `int_call_over_set`'s own arm
/// that reads a kernel enclosure back as a Python `int` result. A
/// private copy of `math_models.rs`'s identically-named helper: this
/// file's own header states the file-ownership convention already kept
/// for `int_call`'s validity check ("the two files stay independent...
/// the rule is written twice rather than shared across the boundary").
///
/// `±inf` ARE elements of the grammar (`refinement_forms`'s own module
/// note: "+-infinity are elements of R-bar and are admitted"), so a
/// bound or an admitted value can be infinite and the set is still
/// well-formed — it just describes a result no Python `int` can hold.
/// NaN cannot appear at all (`element` panics on it at construction), so
/// there is nothing to check for it here.
///
/// This reads the set's OWN top-level forms, looking through
/// `Union`/`Difference`. A form this recognizer does not understand
/// answers `false` — an unread shape declines rather than being assumed
/// finite, which is the direction that keeps the gate honest.
fn enclosure_is_provably_finite(set: &RefinedSet) -> bool {
    if set.forms.is_empty() {
        // the unconstrained set — every real AND both infinities
        return false;
    }
    let mut bounded_below = false;
    let mut bounded_above = false;
    for form in &set.forms {
        match form.form {
            Form::AtLeast | Form::Above => {
                if !form.a.is_finite() {
                    // `atLeast(-inf)` constrains nothing; `atLeast(+inf)`
                    // admits only +inf
                    return false;
                }
                bounded_below = true;
            }
            Form::AtMost | Form::Below => {
                if !form.a.is_finite() {
                    return false;
                }
                bounded_above = true;
            }
            // an explicit value list is finite exactly when every value is
            Form::OneOf => {
                return form.w.iter().all(|v| v.is_finite());
            }
            Form::Union => {
                let (Some(left), Some(right)) = (form.a_.as_ref(), form.b.as_ref()) else {
                    return false;
                };
                // a union is finite only if BOTH arms are
                return enclosure_is_provably_finite(left) && enclosure_is_provably_finite(right);
            }
            // a difference is finite when its left arm is — removing
            // values never adds an infinity
            Form::Difference => {
                let Some(left) = form.a_.as_ref() else {
                    return false;
                };
                return enclosure_is_provably_finite(left);
            }
            // `Integer`/`MultipleOf` narrow but do not bound; the
            // sequence shapes are not scalar sets at all
            Form::Integer | Form::MultipleOf => {}
            _ => return false,
        }
    }
    bounded_below && bounded_above
}

/// `int(string, base=10)`'s exact parsed value, for the base-10
/// default form ONLY (`int_call`'s own scope — a `base=` keyword
/// changes the digit alphabet entirely and is not read by this row's
/// caller, which never passes one through). functions.rst's own
/// grammar: "the string can be preceded by + or - (with no space in
/// between), have leading zeros, be surrounded by whitespace, and have
/// single underscores interspersed between digits." Returns `None`
/// (never a fabricated value) the moment the text does not parse —
/// `call_provable_raise`'s own `is_valid_base_ten_int_string` is the
/// row that speaks the ValueError this shape raises at runtime.
fn parse_base_ten_int_string(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let negative = trimmed.starts_with('-');
    let digits_and_underscores = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if digits_and_underscores.is_empty() {
        return None;
    }
    let chars: Vec<char> = digits_and_underscores.chars().collect();
    if chars.first() == Some(&'_') || chars.last() == Some(&'_') {
        return None;
    }
    let mut digits = String::new();
    let mut previous_was_underscore = false;
    for &c in &chars {
        if c == '_' {
            if previous_was_underscore {
                return None;
            }
            previous_was_underscore = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return None;
        }
        digits.push(c);
        previous_was_underscore = false;
    }
    if digits.is_empty() {
        return None;
    }
    let magnitude: f64 = digits.parse().ok()?;
    Some(if negative { -magnitude } else { magnitude })
}

/// `float(x)` on a single known numeric or known exact string —
/// library/functions.html#float: "Return a floating-point number
/// constructed from a number or a string." A NUMERIC argument answers
/// its exact value, Float-sorted. A known EXACT string is parsed by
/// `parse_float_literal_string` — that function's own doc cites the
/// grammar (functions.rst's `productionlist:: float`): the `inf`/
/// `Infinity`/`nan` spellings (case-insensitive, optional leading sign)
/// answer the exact infinite/NaN value, and any other text that parses
/// as the grammar's `floatnumber` production answers that exact decimal
/// value. A STRING-sorted argument with no exact text this file can
/// parse (`is_string_sorted_argument`'s own doc — e.g. a captured
/// subprocess `.stdout` read: `expressions.rs`'s own
/// `subprocess_run_construction_value`) still determines a SORT: the
/// same clause states `float`'s return is always a `float` regardless of
/// which of the two argument forms produced it, so `float(<any string>)`
/// answers `float_sorted_unknown()` — sort-known, value-unknown, the
/// same posture every other sort-only row in this file takes rather than
/// decline outright. An EXACT string that fails to parse under the
/// grammar keeps that same sort-only posture rather than decline
/// outright (`is_string_sorted_argument` already reads a
/// `Kind::Values`/`String` argument as string-sorted) — CPython raises
/// `ValueError` for it, which this file has no exception channel for,
/// so the sort-only answer is the honest fallback, not a fabricated
/// value.
fn float_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if let Some((value, _sort)) = single_known_numeric(only) {
        if value.is_nan() {
            return Some(nan_value());
        }
        let grade = derived_trust_level(TrustSpec, arguments);
        return Some(known_values(vec![value], PrimitiveKind::Float, grade));
    }
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        let text: String = only.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        if let Some(value) = parse_float_literal_string(&text) {
            // `float("nan")` (and its case/sign variants — parsed by
            // `parse_float_literal_string`'s own grammar reading)
            // answers the domain's NaN state rather than let a bare NaN
            // enter `known_values`, which no refined set admits
            // (`refinement_forms::element`'s own construction-time
            // refusal — the same guard `float_result` keeps in
            // math_models.rs for `math.fabs(nan)`).
            if value.is_nan() {
                return Some(nan_value());
            }
            let grade = derived_trust_level(TrustSpec, arguments);
            return Some(known_values(vec![value], PrimitiveKind::Float, grade));
        }
        return Some(float_sorted_unknown());
    }
    if is_string_sorted_argument(only) {
        return Some(float_sorted_unknown());
    }
    None
}

/// `float(x)` on a KNOWN NUMERIC SET (a seeded range, or a bounded set
/// another transfer already produced — e.g. `float(math.floor(x))`,
/// `math.floor`'s own Integer-sorted enclosure over a declared parameter
/// range, `math_models.rs`'s `floor_call_over_set`): `float_call`'s own
/// row only reads a single known numeric value (`single_known_numeric`),
/// so a Set-shaped argument declines there with no further attempt.
/// Unlike `int_call_over_set`/`abs_call_over_set` (which pose a kernel
/// `TransferQuestion` because their result VALUE differs from their
/// input), `float(x)` on a numeric argument changes only the SORT, never
/// the value (library/functions.html#float: "Return a floating-point
/// number constructed from a number" — the same magnitude, Float-sorted)
/// — so this re-tags the operand's own set in place, no kernel round
/// trip needed. CPython never raises for a numeric argument (only the
/// string-parse form can raise, `float_call`'s own doc), so every
/// Integer/Float/Boolean/Number-sorted set answers here, unconditionally.
fn float_call_over_set(value: &AbstractValue) -> Option<AbstractValue> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean) | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    Some(AbstractValue { kind_tag: Some(PrimitiveKind::Float), ..known_set(value.set.clone(), None, grade, SetKindTag::None) })
}

/// `float(string)`'s exact parsed value, for the grammar
/// library/functions.rst's `productionlist:: float` states (read
/// before writing this function): after leading/trailing whitespace is
/// removed, an optional `sign` (`+`/`-`, `+` has no effect), then either
/// `infinity` (`"Infinity"` or `"inf"`, case-insensitive per that
/// section's own "Case is not significant... 'inf', 'Inf', 'INFINITY',
/// and 'iNfINity' are all acceptable spellings"), `nan` (`"nan"`, same
/// case-insensitivity), or a `floatnumber` (`digitpart ["." digitpart]`
/// or `["." digitpart]`, with an optional `(e|E) [sign] digitpart`
/// exponent — underscores between digits allowed, the same grouping
/// `parse_base_ten_int_string` already reads for `int`). Returns `None`
/// or panics on no legitimate value: `None` when the text does not
/// conform to the grammar (`float_call`'s own caller falls back to the
/// sort-only answer for this row, never a fabricated value) or the
/// parse is not itself the exact spelled decimal (never here, since the
/// spellings this function recognizes route straight to `f64::INFINITY`/
/// `f64::NEG_INFINITY`/`f64::NAN`/Rust's own `str::parse::<f64>`, which
/// implements the same decimal grammar).
fn parse_float_literal_string(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let (negative, unsigned) = match trimmed.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, trimmed.strip_prefix('+').unwrap_or(trimmed)),
    };
    if unsigned.is_empty() {
        return None;
    }
    let lowered = unsigned.to_ascii_lowercase();
    if lowered == "inf" || lowered == "infinity" {
        return Some(if negative { f64::NEG_INFINITY } else { f64::INFINITY });
    }
    if lowered == "nan" {
        return Some(f64::NAN);
    }
    // the `floatnumber` production: digits (with single underscores
    // between them, the same grouping rule int()'s own parse allows),
    // an optional decimal point, an optional e/E exponent — Rust's
    // `str::parse::<f64>` reads this same grammar once underscores are
    // stripped, so digit-and-underscore validity is checked by hand
    // first (a stray underscore, e.g. "1__0" or "_1", is invalid Python
    // syntax that `str::parse` would otherwise silently reject anyway,
    // but the explicit check keeps this row's acceptance exactly the
    // documented grammar rather than piggybacking on Rust's own parser
    // leniency).
    let mut digits_only = String::with_capacity(unsigned.len());
    let mut previous_was_underscore = false;
    let mut previous_was_digit = false;
    for c in unsigned.chars() {
        if c == '_' {
            if !previous_was_digit || previous_was_underscore {
                return None;
            }
            previous_was_underscore = true;
            continue;
        }
        digits_only.push(c);
        previous_was_underscore = false;
        previous_was_digit = c.is_ascii_digit();
    }
    if previous_was_underscore {
        return None;
    }
    let value: f64 = digits_only.parse().ok()?;
    Some(if negative { -value } else { value })
}

/// Whether `argument` is a STRING-sorted value: an exact `Kind::Values`
/// tagged `PrimitiveKind::String`, or a `Kind::Set` that is either
/// explicitly tagged String or untagged with a sequence-shaped own set
/// (`assignability.rs`'s own `sequence_shaped` — the SAME "untagged set,
/// sequence-shaped forms read as string-sorted" convention that file's
/// containment law already applies, e.g. `__name__`'s own untagged
/// `strings()` ground in `expressions.rs`).
fn is_string_sorted_argument(argument: &AbstractValue) -> bool {
    if argument.kind == Kind::Values {
        return argument.kind_tag == Some(PrimitiveKind::String);
    }
    if argument.kind != Kind::Set {
        return false;
    }
    argument.kind_tag == Some(PrimitiveKind::String)
        || (argument.kind_tag.is_none() && crate::assignability::sequence_shaped(&argument.set))
}

/// `chr(i)` on a known Integer code point — library/functions.html#chr:
/// "Return the string representing a character whose Unicode code
/// point is the integer *i*." A one-code-point exact string, the same
/// `Kind::Values`/`PrimitiveKind::String` shape `string_models.rs`
/// builds for any other exact string. `i` outside the valid code-point
/// range (`0..=0x10FFFF`, the same range `char::from_u32` itself
/// enforces) has no row here: CPython raises `ValueError`, which this
/// domain has no channel for this wave, so this row declines rather
/// than answer a fabricated character.
fn chr_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, sort) = single_known_numeric(only)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    if value < 0.0 || value > 0x10FFFF as f64 {
        return None;
    }
    char::from_u32(value as u32)?;
    Some(known_values(vec![value], PrimitiveKind::String, TrustSpec))
}

/// `str(object)` — library/stdtypes.rst's `class:: str(object='')`
/// constructor row: "Return a string version of *object*." Modeled for
/// four known argument shapes: an exact string (the identity
/// conversion — `str(word)` answers `word` unchanged, per the same
/// row's own "If *object* already is a string, it is returned
/// unchanged" behavior), a known Integer (CPython's plain decimal
/// spelling, no `.0` — the same integer-spelling rule
/// `expressions.rs`'s f-string composition already establishes for an
/// interpolated Integer), a known EXCEPTION instance
/// (`expressions.rs`'s `exception_construction_value`, tagged
/// `source == "exception"`, one `args` field holding the constructor's
/// own positional arguments as a `Kind::List`) whose FIRST argument is
/// a known exact string — `str(Exception(message))` answers `message`
/// unchanged: `Doc/tutorial/errors.rst`, "Errors and Exceptions" §8.3,
/// "the exception instance... typically has an `args` attribute...
/// builtin exception types define `__str__` to print all the
/// arguments." A single-string-argument exception's `__str__` is
/// exactly that one string (CPython's own `BaseException.__str__`:
/// zero args -> `''`, one arg -> `str(args[0])`, 2+ args -> the
/// `repr()` of the whole tuple — only the one-string-argument row is
/// modeled here), and a NONNEGATIVE BOUNDED Integer window (a seeded
/// parameter range, or a bounded set another transfer produced) —
/// `int.__repr__`'s plain no-leading-zero decimal spelling widened
/// from one value to the whole window, `json_grammar::
/// integer_window_grammar`'s own composition (already built for
/// `json.dumps`'s serialized-text grammar; reused here rather than
/// duplicated). A known FLOAT argument is NOT modeled: the
/// repr-shortest spelling `format_py_number` builds lives in the
/// `refined_sets` crate, out of this file's own dependency edge for
/// this wave, so `str(float)` declines rather than half-build that
/// spelling by hand.
fn str_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        return Some(only.clone());
    }
    if only.kind == Kind::Object && only.source == "exception" {
        return exception_single_string_message(only);
    }
    if only.kind == Kind::Set && only.kind_tag == Some(PrimitiveKind::Integer) {
        return str_call_over_integer_window(only);
    }
    let (value, sort) = single_known_numeric(only)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    let spelled = format!("{}", value as i64);
    let code_points: Vec<f64> = spelled.chars().map(|c| c as u32 as f64).collect();
    Some(known_values(code_points, PrimitiveKind::String, TrustSpec))
}

/// `str(n)` on a NONNEGATIVE BOUNDED Integer-sorted `Kind::Set` window
/// `[lo, hi]` (a seeded parameter range, or a bounded set another
/// transfer produced) — the exact digit-count run
/// `json_grammar::integer_window_grammar` already composes for
/// `json.dumps`'s serialized-text grammar, reused unchanged here for
/// `str_call`'s own decimal-spelling row: both are the SAME `int.
/// __repr__` plain decimal spelling (stdtypes.rst's `str(object)`
/// row delegates to `__str__`, which for `int` is `__repr__`'s own
/// no-leading-zero decimal text), just reached from a different
/// caller. The bound is read off the set's own top-level
/// `AtLeast`/`Above`/`AtMost`/`Below` forms syntactically — no kernel
/// ask, the same private-copy convention `json_grammar::
/// integer_set_bounds` already keeps against `expressions.rs`'s own
/// identically-named helper (this file's own AGENT-BRIEF scope keeps
/// it from reaching into either). A negative lower bound, or a bound
/// this reader cannot close (an unbounded ray, a union, a pattern),
/// declines — `integer_window_grammar`'s own `lo < 0` refusal
/// propagates here as a decline rather than a fabricated fallback.
fn str_call_over_integer_window(value: &AbstractValue) -> Option<AbstractValue> {
    let (lo, hi) = integer_set_bounds(value)?;
    let grammar = crate::json_grammar::integer_window_grammar(lo, hi)?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(grammar, None, grade, SetKindTag::None)
    })
}

/// The closed integer bound `[lo, hi]` a value states, when the value is
/// a bounded Integer-sorted `Kind::Set` — the same syntactic hull
/// `json_grammar::integer_set_bounds`/`expressions.rs::
/// integer_set_bounds` both already read, duplicated here rather than
/// exported per this file's own file-ownership convention (see either
/// of those two functions' own doc comments).
fn integer_set_bounds(value: &AbstractValue) -> Option<(i64, i64)> {
    if value.kind != Kind::Set || value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &value.set.forms {
        match form.form {
            Form::AtLeast => lo = Some(lo.map_or(form.a, |current: f64| current.max(form.a))),
            Form::Above => lo = Some(lo.map_or(form.a.floor() + 1.0, |current: f64| current.max(form.a.floor() + 1.0))),
            Form::AtMost => hi = Some(hi.map_or(form.a, |current: f64| current.min(form.a))),
            Form::Below => hi = Some(hi.map_or(form.a.ceil() - 1.0, |current: f64| current.min(form.a.ceil() - 1.0))),
            Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, hi) = (lo?, hi?);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// The exact message `str()` of a known exception instance answers, for
/// the ONE constructor-argument shape this file models: an `args`
/// field (`expressions.rs`'s own exception-construction tag) holding a
/// `Kind::List` of exactly one known exact-string element —
/// `BaseException.__str__`'s one-argument row (this function's own
/// caller doc). Any other `args` shape (zero elements, 2+ elements, a
/// non-string element) declines — this file does not build the `repr()`
/// spelling a multi-argument `__str__` would need.
fn exception_single_string_message(instance: &AbstractValue) -> Option<AbstractValue> {
    let args = &instance.keys.iter().find(|key| key.name == "args")?.value;
    if args.kind != Kind::List {
        return None;
    }
    let [only] = args.items.as_slice() else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        return Some(only.clone());
    }
    None
}

/// `hash(x)` — library/functions.html#hash: "Return the hash value of
/// the object (if it has one). Hash values are integers... Numeric
/// values that compare equal have the same hash value (even if they
/// are of different types, as is the case for 1 and 1.0)." The doc
/// states only that the result is a Python `int` and that EQUAL
/// operands hash equally — it does NOT state `hash(n) == n` for every
/// int `n` (CPython's real implementation reduces modulo
/// `sys.hash_info.modulus`, a fact outside library/functions.html's own
/// text), so this row answers the SORT the doc actually guarantees —
/// the unbounded integer ground — rather than fabricate an identity
/// claim the cited clause does not make. Modeled for any single
/// argument this file can already read a value or a known Set for
/// (`single_known_numeric`, or a numeric/string-sorted `Kind::Set`/
/// `Kind::Values` argument): `hash` accepts any hashable object, and
/// this row's own claim (unbounded `int`) holds regardless of which
/// hashable shape the argument is, so the argument itself is not
/// otherwise inspected.
///
/// The answer carries an EXPLICIT `AtLeast(-inf)` ray alongside
/// `Integer`, the same two-form shape `narrowing.rs`'s own
/// `unbounded_integers()` and this file's own `int_image()` both build
/// for "the whole integer ground, no bound stated" — never `Integer`
/// alone with zero ray forms. A bare `[Integer]` set is missing the
/// ray form the kernel's scalar deciders key the 1-tuple scalar shape
/// on, which let a one-sided guard's own narrowed window (`hash(x) >=
/// 0`, only a lower ray tightened onto this set) reach
/// `scalar_subset`/`assignability.rs`'s containment ask still missing
/// the upper-boundedness a real `[0, 150]`-declared set requires — the
/// A15.xfer.hash `hash_outside` soundness gap this two-form shape
/// closes.
fn hash_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Unknown {
        return None;
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(f64::NEG_INFINITY)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    })
}

/// `object()` — library/functions.html#object: "This is the ultimate
/// base class of all other classes... When the constructor is called,
/// it returns a new featureless object. The constructor does not accept
/// any arguments." A featureless object has no fields this domain could
/// enumerate, so the answer is `opaque_value` — the same "kind of thing
/// known, contents not" shape `type(object)` already answers above —
/// tagged `source: "object()"` so a dict-display key built from this
/// value (`known_dict_key`'s identity arm, `collection_models.rs`) can
/// recognize it as a stable, non-string/int key: `stdtypes.rst`'s
/// mapping-key rule states a dict key only needs to be hashable, never a
/// string or number, and a fresh `object()` is hashable by identity
/// alone (no `__eq__`/`__hash__` override, `object`'s own doc — "has
/// methods that are common to all instances," none of which redefine
/// equality).
///
/// Scope: this tags every `object()` call the SAME way, so it only
/// answers a sound identity for the corpus shape actually read — ONE
/// `object()` call, bound to a name once and read back by that name
/// (never re-evaluated) — never two DIFFERENT `object()` call sites
/// compared as keys within the same dict. Telling two live `object()`
/// values apart needs a per-call-site tag threaded from the call
/// expression itself (`expressions.rs::evaluate_call`), which this file
/// has no access to (it only sees the callee name and the evaluated,
/// argument-less call).
fn object_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if !arguments.is_empty() {
        return None;
    }
    let mut instance = opaque_value("a featureless object");
    instance.source = "object()".to_owned();
    Some(instance)
}

/// The dispatcher: a call to Python builtin `function` with already-
/// evaluated `arguments`. `None` means "not modeled here" — the caller
/// declines honestly rather than reading this as "the call is unknown to
/// Python." `Some` is an exact answer at the derived trust grade. Pure
/// Rust, no kernel dependency — `builtin_call_result_with_kernel` is the
/// caller's actual entry point, trying the kernel-needing rows first
/// (`min`/`max`'s own set-valued row) and falling back to this
/// dispatcher unchanged; kept separate so every one of this dispatcher's
/// OWN tests keeps asserting with no kernel dylib required, exactly as
/// before `min`/`max` grew a kernel-asked arm.
pub fn builtin_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "abs" => abs_call(arguments),
        "round" => round_call(arguments),
        // two-or-more-argument form first (min_max_call's own `len < 2`
        // guard declines there); the single-iterable form answers
        // through min_max_over_iterable for a known Kind::List, then
        // min_max_over_star for an unknown-length star-shaped iterable
        // (a declared list[X]/set[X]/Sequence[X] parameter). The
        // kernel-asked set-valued row (`min_max_call_over_sets`) is
        // NOT reachable here — see `builtin_call_result_with_kernel`.
        "min" => min_max_call(arguments, |candidate, current| candidate < current)
            .or_else(|| min_max_over_iterable(arguments, |candidate, current| candidate < current))
            .or_else(|| min_max_over_star(arguments)),
        "max" => min_max_call(arguments, |candidate, current| candidate > current)
            .or_else(|| min_max_over_iterable(arguments, |candidate, current| candidate > current))
            .or_else(|| min_max_over_star(arguments)),
        // len() declines for now: answering it needs container states
        // (string/list/tuple/dict length facts) this domain does not yet
        // carry — single_known_numeric only ever reads a known SCALAR,
        // never a container, so there is no row to write until container
        // states land.
        "len" => None,
        "int" => int_call(arguments),
        "float" => float_call(arguments),
        "sum" => sum_call(arguments).or_else(|| sum_call_over_star(arguments)),
        "sorted" => sorted_call(arguments),
        "list" => list_constructor_call(arguments),
        "set" => set_constructor_call(arguments),
        "dict" => dict_constructor_call(arguments),
        "chr" => chr_call(arguments),
        "str" => str_call(arguments),
        "iter" => iter_call(arguments),
        "next" => next_call(arguments),
        "anext" => anext_call(arguments),
        "cast" => cast_call(arguments),
        // `type(object)` (one-argument form) — library/functions.html#type:
        // "With one argument, return the type of an object." This domain
        // has no type-object Kind, so the answer is opaque — the honest
        // "a type object" sort, never a specific value
        // (b-body-expressions.py's `type_as_value`). The three-argument
        // `type(name, bases, dict)` class-creation form is not this row
        // (a different arity, out of scope).
        "type" if arguments.len() == 1 => Some(opaque_value("a type object")),
        "object" => object_call(arguments),
        "hash" => hash_call(arguments),
        // `from urllib.parse import quote` — see `urllib_quote_call`'s
        // own doc for why this bare-name spelling is a builtin row here
        // rather than routed through `stdlib_call_result`.
        "quote" => urllib_quote_call(arguments),
        _ => None,
    }
}

/// The caller's actual entry point (`expressions.rs::evaluate_call`): a
/// call to Python builtin `function`, `kernel` in hand for the row
/// families that need it — `min`/`max`'s two-or-more-argument form when
/// at least one argument is a `Kind::Set` (`min_max_call_over_sets`'s
/// own doc, including the NaN-discharge citation), `abs`'s single
/// Set-seeded operand (`abs_call_over_set`'s own doc), `int`'s
/// single Set-seeded operand (`int_call_over_set`'s own doc — e.g.
/// `int(math.sqrt(x))` over a declared parameter range), and `float`'s
/// single Set-seeded operand (`float_call_over_set`'s own doc — e.g.
/// `float(math.floor(x))`; this one row needs no kernel round trip of
/// its own, only the `kernel` parameter's presence in this dispatcher's
/// signature to sit beside its `int`/`abs` siblings). Every other
/// builtin routes straight through the pure-Rust `builtin_call_result`
/// above, tried FIRST so a known-scalar call never pays a kernel round
/// trip it does not need.
pub fn builtin_call_result_with_kernel(
    function: &str,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    builtin_call_result(function, arguments).or_else(|| match function {
        "min" => min_max_call_over_sets(arguments, TransferQuestionOp::Min, kernel),
        "max" => min_max_call_over_sets(arguments, TransferQuestionOp::Max, kernel),
        "abs" => {
            let [only] = arguments else { return None };
            abs_call_over_set(only, kernel)
        }
        "int" => {
            let [only] = arguments else { return None };
            int_call_over_set(only, kernel).or_else(|| int_image())
        }
        "float" => {
            let [only] = arguments else { return None };
            float_call_over_set(only)
        }
        _ => None,
    })
}

/// `int(<anything the rows above declined>)`'s own IMAGE: wherever the
/// call returns at all, it returns an int (library/functions.rst — a
/// non-convertible operand raises instead), so an operand no concrete
/// or kernel row reads still answers the unbounded integer sort. The
/// raise arm is `call_provable_raise`'s business — a provably-raising
/// call's value is unreachable, and an unreachable value carrying the
/// image is sound either way.
fn int_image() -> Option<AbstractValue> {
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![
                refined_sets::refinement_forms::integer(),
                at_least(f64::NEG_INFINITY),
            ]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    })
}

/// An unbounded, NONNEGATIVE numeric ground — the shared answer shape
/// `time_call_result`'s `time.time` row and `os_call_result`'s
/// `os.open` row both state (a value known only to sit in `[0, +inf)`,
/// tagged `sort`). Composed once here rather than duplicated at each
/// call site. `PrimitiveKind::Integer` additionally carries the
/// `integer()` refinement form in the SET itself, not just the
/// `kind_tag` sort marker — without it, a caller's own guard (e.g.
/// `0 <= fd <= 150`) narrows the range but the narrowed set stays
/// bare `[0, 150]` with no integrality, which fails assignment against
/// a declared alias requiring `integer` (A15.xfer.handle's own
/// `os.open` row). `PrimitiveKind::Float` (`time.time`) never adds
/// this form — a float ground is not integer-valued.
fn nonnegative_ground(sort: PrimitiveKind) -> AbstractValue {
    let mut forms = vec![at_least(0.0)];
    if sort == PrimitiveKind::Integer {
        forms.push(refined_sets::refinement_forms::integer());
    }
    AbstractValue {
        kind_tag: Some(sort),
        ..known_set(make_refined_set(forms), None, TrustSpec, SetKindTag::None)
    }
}

/// `time.time()` — library/time.html#time.time: "Return the time in
/// seconds since the epoch as a floating-point number... Note that even
/// though the time is always returned as a floating-point number, not
/// all systems provide time with a better precision than 1 second." The
/// epoch itself is defined as 1970-01-01 00:00:00 (UTC) on every
/// platform this doc covers, so the returned value is always
/// NONNEGATIVE — this row states exactly that ground: `[0, +inf)`,
/// Float-sorted, never a specific instant (the running clock is not a
/// fact this domain reads). Zero-argument only, per the doc's own
/// signature.
fn time_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "time" || !arguments.is_empty() {
        return None;
    }
    Some(nonnegative_ground(PrimitiveKind::Float))
}

/// `os.open(path, flags)` / `os.close(fd)` — library/os.html:
/// `os.open`: "Return a file descriptor... to be used by other
/// low-level (i.e. os.read()) file operations." A file descriptor is
/// always a NONNEGATIVE `int` (`os.rst`'s own examples index only ever
/// nonnegative values, and CPython raises `OSError` rather than ever
/// returning a negative fd) — this row states the ground `[0, +inf)`,
/// Integer-sorted, never a specific descriptor number, matching
/// A15.xfer.handle's own claim ("a file descriptor opened fresh...
/// carries no identity claim"). `os.close`: "Close file descriptor
/// *fd*... Availability: not Emscripten, not WASI." No return value —
/// CPython's own `os.close` always returns `None`, so this row answers
/// the domain's exact absent state (never Unknown) for ANY single
/// argument, matching a Python function whose only documented effect is
/// closing the descriptor.
fn os_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "open" if arguments.len() == 2 => Some(nonnegative_ground(PrimitiveKind::Integer)),
        "close" if arguments.len() == 1 => Some(null_value()),
        _ => None,
    }
}

/// `unicodedata.normalize(form, unistr)` — library/unicodedata.html:
/// "Return the normal form *form* for the Unicode string *unistr*...
/// Valid values for *form* are 'NFC', 'NFKC', 'NFD', and 'NFKD'." The
/// doc states the return is itself a Python `str`, with no further
/// bound on its content or length (a normalization form can both grow
/// and shrink a string's code-point count relative to its input,
/// library/unicodedata.html's own "Unicode Standard Annex #15"
/// citation) — this row states exactly that sort, the whole-strings
/// ground `Σ*`, matching A3.xfer.normalize's own claim ("result is
/// Σ*"). Modeled for the two-argument form with a known exact-string
/// `form` argument in the doc's own four valid spellings; any other
/// `form` (unknown, or a string outside that set) declines rather than
/// assume the call does not raise.
fn unicodedata_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "normalize" {
        return None;
    }
    let [form, _unistr] = arguments else { return None };
    if form.kind != Kind::Values || form.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    let form_text: String = form.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
    if !matches!(form_text.as_str(), "NFC" | "NFKC" | "NFD" | "NFKD") {
        return None;
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, TrustSpec, SetKindTag::None)
    })
}

/// `urllib.parse.quote(string)` — library/urllib.parse.html#urllib.parse.quote:
/// "Replace special characters in *string* using the %xx escape...
/// Letters, digits, and the characters '_.-~' are never quoted." The
/// result is a Python `str` built only from that ASCII subset plus the
/// literal `%` escape triples — narrower than the whole-strings ground,
/// but this row states the SORT-ONLY answer (`Σ*`, String-sorted)
/// rather than the tight percent-encoding grammar: the doc's own
/// `safe='/'` default (a further always-unquoted character this row
/// does not thread through) makes the exact alphabet argument-
/// dependent, so `Σ*` is the sound claim actually made here, matching
/// A3.xfer.url's own claim ("result is Σ* (percent-encoding
/// grammar)"). One-argument form only (no `safe=`/`encoding=`/
/// `errors=` keyword arguments modeled).
///
/// Reached through `builtin_call_result`'s own BARE-NAME dispatch, not
/// `stdlib_call_result`'s module-qualified one: the corpus's own row
/// (A3.xfer.url.py) writes `from urllib.parse import quote` then calls
/// the bare name `quote(s)` — `urllib.parse` is not a Python-source
/// module the cross-module resolver reads (`check.rs::
/// bind_or_forget_imported_name`'s own doc), so the import binds
/// nothing and `quote` reaches `evaluate_call`'s `Expr::Call(Expr::Name(...))`
/// arm exactly like an ordinary builtin call.
fn urllib_quote_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [_string] = arguments else { return None };
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, TrustSpec, SetKindTag::None)
    })
}

/// The dispatcher for a MODULE-QUALIFIED stdlib call whose result is
/// answered from this file — `time.<function>`, `os.<function>`,
/// `unicodedata.<function>`, `dict.<function>` (a builtin TYPE's own
/// classmethod, gated in `expressions.rs::evaluate_attribute_call` the
/// same way a module name is: a bare `dict` receiver that reads unbound
/// in `environment`, since `dict` is never locally rebound in the
/// corpus's own rows). Callable name `module` (the attribute chain's
/// own root, e.g. `"time"`) and `function` (the called attribute, e.g.
/// `"time"`) are read separately so a caller can gate on the module
/// name exactly the way its own `math`/`re`/`json` arms already do,
/// before ever reaching this dispatcher. `None` means "not modeled
/// here" — the caller's own decline, never a guessed value.
/// `urllib.parse.quote` is NOT reached here — see `urllib_quote_call`'s
/// own doc for why it is a bare-name builtin row instead.
pub fn stdlib_call_result(module: &str, function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match module {
        "time" => time_call_result(function, arguments),
        "os" => os_call_result(function, arguments),
        "unicodedata" => unicodedata_call_result(function, arguments),
        "dict" if function == "fromkeys" => dict_fromkeys_call(arguments),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;

    use super::*;

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustSpec)
    }

    fn float(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Float, TrustSpec)
    }

    /// A kernel handle for tests that ask a `min`/`max`-over-a-set
    /// question — the same skip `math_models.rs`'s own `loaded_kernel`
    /// takes when the native dylib artifact has not been built, so
    /// this file's tests run without requiring `pnpm kernel:native`
    /// first. Every OTHER test in this module keeps calling
    /// `builtin_call_result` directly (pure Rust, no kernel needed) —
    /// its own signature never changed.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    #[test]
    fn round_half_to_even_rounds_up_at_odd_tenths() {
        // round(201.5) == 202: 201.5 sits between 201 and 202; 202 is
        // the even choice.
        let got = builtin_call_result("round", &[float(201.5)]).expect("round(201.5) models");
        assert_eq!(got.values, vec![202.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn round_half_to_even_rounds_down_at_even_tenths() {
        // round(40.5) == 40: 40.5 sits between 40 and 41; 40 is the even
        // choice — the AGENT-BRIEF row-inverting fact against a naive
        // round-half-up reading.
        let got = builtin_call_result("round", &[float(40.5)]).expect("round(40.5) models");
        assert_eq!(got.values, vec![40.0]);
    }

    #[test]
    fn round_two_argument_form_declines() {
        let got = builtin_call_result("round", &[float(40.5), integer(1.0)]);
        assert!(got.is_none(), "round(x, n) should decline: {got:?}");
    }

    #[test]
    fn abs_of_negative_integer_is_positive_integer() {
        let got = builtin_call_result("abs", &[integer(-200.0)]).expect("abs(-200) models");
        assert_eq!(got.values, vec![200.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `abs()` over a Set-seeded operand asks the kernel's `Abs` transfer
    /// (`abs_call_over_set`'s own doc, `javascript-pins.md` arith.7): a
    /// window straddling zero folds its lower bound to 0 — `abs([-2, 1])`
    /// answers `[0, 2]`, `transferAbs`'s own `straddles` branch
    /// (`theories/binary64/abs.lean`: `lo := if straddles then 0 else
    /// min(abs(A.lo), abs(A.hi))`, `hi := max(abs(A.lo), abs(A.hi))` —
    /// here `A.lo = -2, A.hi = 1`, both admitted, so `lo = 0` and
    /// `hi = max(2, 1) = 2`). Asserts the exact enclosure, not merely the
    /// shape, since the window is narrow enough to pin by hand.
    #[test]
    fn abs_over_a_set_operand_asks_the_kernel() {
        let Some(kernel) = loaded_kernel() else { return };
        let window = make_refined_set(vec![at_least(-2.0), at_most(1.0)]);
        let operand = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(window, None, TrustSpec, SetKindTag::None)
        };
        let got = builtin_call_result_with_kernel("abs", &[operand], &kernel)
            .expect("abs([-2, 1]) over a Set operand models through the kernel");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
        let want = make_refined_set(vec![at_least(0.0), at_most(2.0)]);
        assert_eq!(got.set, want, "abs([-2, 1]) should answer [0, 2]: got {:?}", got.set);
    }

    /// `float(x)` over an Integer-sorted Set operand (`floor_call_over_set`'s
    /// own image, `math.floor(x)` over a declared `[2.5, 3.5]` guard —
    /// `float_call_over_set`'s own doc): re-tags the same set Float-sorted,
    /// no kernel round trip and no value change — `float([2, 3])` answers
    /// the identical `{2, 3}` window, only Float-sorted now.
    #[test]
    fn float_over_a_set_operand_re_sorts_without_a_kernel_round_trip() {
        let Some(kernel) = loaded_kernel() else { return };
        let window = make_refined_set(vec![at_least(2.0), at_most(3.0)]);
        let operand =
            AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..known_set(window.clone(), None, TrustSpec, SetKindTag::None) };
        let got = builtin_call_result_with_kernel("float", &[operand], &kernel)
            .expect("float([2, 3]) over a Set operand models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
        assert_eq!(got.set, window, "float() must not change the operand's own set: got {:?}", got.set);
    }

    #[test]
    fn int_truncates_toward_zero_on_positive_fraction() {
        let got = builtin_call_result("int", &[float(7.9)]).expect("int(7.9) models");
        assert_eq!(got.values, vec![7.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn int_truncates_toward_zero_on_negative_fraction() {
        // int(-7.9) == -7, not -8: truncation toward zero, not floor.
        let got = builtin_call_result("int", &[float(-7.9)]).expect("int(-7.9) models");
        assert_eq!(got.values, vec![-7.0]);
    }

    #[test]
    fn int_of_a_base_ten_digit_string_parses_the_exact_value() {
        // int("75") == 75 — j-stdlib-surfaces.py's own int_parse row
        let string_argument = known_values(vec![55.0, 53.0], PrimitiveKind::String, TrustSpec);
        let got = builtin_call_result("int", &[string_argument]).expect("int(\"75\") models");
        assert_eq!(got.values, vec![75.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn int_of_a_non_numeric_string_declines() {
        // int("abc") raises ValueError at runtime — this row never
        // fabricates a value for it; the raise itself is
        // expressions.rs's call_provable_raise's own business
        let string_argument = string_value("abc");
        let got = builtin_call_result("int", &[string_argument]);
        assert!(got.is_none(), "int(\"abc\") should decline: {got:?}");
    }

    #[test]
    fn int_of_a_negative_digit_string_parses_the_exact_negative_value() {
        let string_argument = string_value("-7");
        let got = builtin_call_result("int", &[string_argument]).expect("int(\"-7\") models");
        assert_eq!(got.values, vec![-7.0]);
    }

    /// `int()` over a Float-sorted Set operand asks the kernel's `Trunc`
    /// transfer (`int_call_over_set`'s own doc) — the same shape
    /// `int(math.sqrt(x))` builds over a declared parameter range
    /// (`c-reads-and-values.py`'s `math_sqrt_over_declared_range`: `x`
    /// is `[0, 100]`, `math.sqrt(x)` is `[0, 10]`, `int(...)` of that
    /// stays `[0, 10]` — already integral, so truncation changes
    /// nothing at either endpoint).
    #[test]
    fn int_over_a_set_operand_asks_the_kernel() {
        let Some(kernel) = loaded_kernel() else { return };
        let window = make_refined_set(vec![at_least(0.0), at_most(10.0)]);
        let operand = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(window, None, TrustSpec, SetKindTag::None)
        };
        let got = builtin_call_result_with_kernel("int", &[operand], &kernel)
            .expect("int([0, 10]) over a Float Set operand models through the kernel");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer), "int(...) is always Integer-sorted, regardless of its argument's own sort");
        // `Trunc`'s answer over an already-integral window carries its own
        // `Integer` form — `binary64.trunc` proves the whole result is a
        // whole number here, not just this row's own sort tag
        let want = make_refined_set(vec![at_least(0.0), at_most(10.0), refined_sets::refinement_forms::integer()]);
        assert!((kernel.scalar_subset)(&got.set, &want), "result {:?} not ⊆ want {:?}", got.set, want);
        assert!((kernel.scalar_subset)(&want, &got.set), "want {:?} not ⊆ result {:?}", want, got.set);
    }

    /// `int(x)` over a BARE, UNBOUNDED `float` parameter's own seed
    /// (`float_sorted_unknown()`'s own `numbers()` set, `[NEG_INFINITY,
    /// +inf)`) must still answer the unbounded Integer sort rather than
    /// decline outright: `binary64.trunc`'s own enclosure over an
    /// entirely-unbounded window is never provably finite
    /// (`enclosure_is_provably_finite` false by construction — the empty-
    /// forms/unbounded-ray cases it itself declines), so before this fix
    /// `int_call_over_set` returned `None` here, leaving `n = int(x)`
    /// Unknown downstream (D5's own `clamp_to_age` helpers' fact-export
    /// blocker). The weaker TRUE claim — every non-raising outcome of
    /// `int(x)` is SOME int — is `int_image`'s own image, pinned here
    /// directly.
    #[test]
    fn int_over_an_unbounded_float_operand_answers_the_unbounded_integer_image() {
        let Some(kernel) = loaded_kernel() else { return };
        let operand = float_sorted_unknown();
        let got = builtin_call_result_with_kernel("int", &[operand], &kernel).expect("int(x) over an unbounded float must still decide the image");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
        let want = make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(f64::NEG_INFINITY)]);
        assert_eq!(got.set, want, "the answer is int_image's own unbounded Integer ray, not a decline");
    }

    #[test]
    fn float_of_inf_string_is_positive_infinity() {
        // functions.rst's float() grammar: "inf"/"Infinity" (case-
        // insensitive) spell positive infinity.
        let string_argument = string_value("inf");
        let got = builtin_call_result("float", &[string_argument]).expect("float(\"inf\") models");
        assert_eq!(got.values, vec![f64::INFINITY]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn float_of_negative_inf_string_is_negative_infinity() {
        let string_argument = string_value("-inf");
        let got = builtin_call_result("float", &[string_argument]).expect("float(\"-inf\") models");
        assert_eq!(got.values, vec![f64::NEG_INFINITY]);
    }

    #[test]
    fn float_of_nan_string_is_the_nan_admitting_value() {
        // `float("nan")` answers the domain's own `Kind::NaN` state
        // (`nan_value()`), never a `Kind::Values` list carrying a bare
        // NaN — no refined set admits NaN as an element
        // (`refinement_forms::element`'s own construction-time
        // refusal), so `Kind::Values` must stay NaN-free too.
        let string_argument = string_value("nan");
        let got = builtin_call_result("float", &[string_argument]).expect("float(\"nan\") models");
        assert_eq!(got.kind, Kind::NaN, "float(\"nan\") should answer the domain's NaN state: {got:?}");
    }

    #[test]
    fn float_of_a_decimal_digit_string_parses_the_exact_value() {
        let string_argument = string_value("1.5");
        let got = builtin_call_result("float", &[string_argument]).expect("float(\"1.5\") models");
        assert_eq!(got.values, vec![1.5]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn float_of_infinity_spelling_case_insensitive() {
        // "Case is not significant... 'INFINITY' and 'iNfINity' are all
        // acceptable spellings for positive infinity."
        let string_argument = string_value("Infinity");
        let got = builtin_call_result("float", &[string_argument]).expect("float(\"Infinity\") models");
        assert_eq!(got.values, vec![f64::INFINITY]);
    }

    #[test]
    fn float_of_an_unparseable_string_keeps_the_sort_only_answer() {
        let string_argument = string_value("not a number");
        let got = builtin_call_result("float", &[string_argument]).expect("float(<any string>) models sort-only");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn min_over_known_numerics_picks_the_smallest() {
        let got = builtin_call_result("min", &[integer(3.0), integer(-1.0), integer(5.0)])
            .expect("min(...) models");
        assert_eq!(got.values, vec![-1.0]);
    }

    #[test]
    fn max_over_known_numerics_picks_the_largest() {
        let got = builtin_call_result("max", &[integer(3.0), integer(-1.0), integer(5.0)])
            .expect("max(...) models");
        assert_eq!(got.values, vec![5.0]);
    }

    #[test]
    fn max_threads_the_winning_arguments_own_sort() {
        // 4.5 (float) beats 3 (int): the winner's own Float sort carries
        // through, matching Python's min/max returning the argument
        // itself unchanged.
        let got = builtin_call_result("max", &[integer(3.0), float(4.5)]).expect("max(...) models");
        assert_eq!(got.values, vec![4.5]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    }

    #[test]
    fn min_single_scalar_argument_declines() {
        // min(3) is neither the two-or-more-scalar form nor the
        // single-iterable form — a bare scalar is not a Kind::List.
        let got = builtin_call_result("min", &[integer(3.0)]);
        assert!(got.is_none(), "min(x) with one scalar argument should decline: {got:?}");
    }

    /// A numeric-sorted `Kind::Set` operand in the two-or-more-argument
    /// `max` form declines through `min_max_call` (`single_known_numeric`
    /// refuses a Set) and reaches the kernel-asked arm
    /// (`builtin_call_result_with_kernel`'s own doc). `max(ages, 0)` over
    /// `ages` bounded 0..120 and the known scalar `0` asks
    /// `binary64.max`, answering an enclosure whose own hull sits inside
    /// 0..120 — this test only asserts the SHAPE (a Set-kind Integer
    /// answer), not a specific enclosure, matching the kernel-invocation
    /// exception this file's tests otherwise avoid.
    #[test]
    fn max_over_a_set_operand_asks_the_kernel() {
        let Some(kernel) = loaded_kernel() else { return };
        let ages_window = make_refined_set(vec![at_least(0.0), at_most(120.0)]);
        let ages = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(ages_window, None, TrustSpec, SetKindTag::None)
        };
        let got = builtin_call_result_with_kernel("max", &[ages, integer(0.0)], &kernel)
            .expect("max(ages, 0) over a Set operand models through the kernel");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// The known-scalar path still wins first — `builtin_call_result_with_kernel`
    /// never pays a kernel round trip when `builtin_call_result` alone
    /// already answers (both arguments known scalars here).
    #[test]
    fn max_over_known_scalars_never_reaches_the_kernel_arm() {
        let Some(kernel) = loaded_kernel() else { return };
        let got = builtin_call_result_with_kernel("max", &[integer(3.0), integer(9.0)], &kernel)
            .expect("max(3, 9) models");
        assert_eq!(got.values, vec![9.0]);
    }

    #[test]
    fn min_single_iterable_argument_picks_the_smallest() {
        let list = known_list(vec![integer(3.0), integer(-1.0), integer(5.0)], TrustSpec);
        let got = builtin_call_result("min", &[list]).expect("min([...]) models");
        assert_eq!(got.values, vec![-1.0]);
    }

    #[test]
    fn max_single_iterable_argument_picks_the_largest() {
        let list = known_list(vec![integer(200.0)], TrustSpec);
        let got = builtin_call_result("max", &[list]).expect("max([...]) models");
        assert_eq!(got.values, vec![200.0]);
    }

    #[test]
    fn min_max_empty_iterable_declines() {
        let empty = known_list(vec![], TrustSpec);
        assert!(builtin_call_result("min", &[empty]).is_none());
    }

    #[test]
    fn sum_over_known_list_totals_the_elements() {
        let list = known_list(vec![integer(1.0), integer(2.0), integer(3.0)], TrustSpec);
        let got = builtin_call_result("sum", &[list]).expect("sum([...]) models");
        assert_eq!(got.values, vec![6.0]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn sum_with_a_start_value_adds_it_in() {
        let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
        let got = builtin_call_result("sum", &[list, integer(10.0)]).expect("sum([...], start) models");
        assert_eq!(got.values, vec![13.0]);
    }

    #[test]
    fn sum_widens_to_float_when_any_element_is_float() {
        let list = known_list(vec![integer(1.0), float(2.5)], TrustSpec);
        let got = builtin_call_result("sum", &[list]).expect("sum([...]) models");
        assert_eq!(got.values, vec![3.5]);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    }

    /// `D5.edge.helper.py`'s own `sum(s * s for s in clamped)` shape: a
    /// GENERATOR expression, which `expressions.rs`'s own `Expr::
    /// Generator` arm already routes through `evaluate_list_or_set_comp`
    /// — the SAME star-comprehension path a list/set comprehension takes
    /// — so once `clamped`'s own element window reaches `sum(...)` as a
    /// `Kind::Set` repetition (`s * s` squaring `s ∈ [-1, 1]` down to
    /// `[0, 1]`, a fact `expressions.rs`'s own `*` transfer over Set
    /// operands states, not this file's concern), `sum_call`'s existing
    /// `.or_else(|| sum_call_over_star(arguments))` fallback (this
    /// dispatcher's own wiring, unchanged) already answers it — pinned
    /// here directly on a star-shaped Float window with no concrete
    /// `Kind::List` items, the shape a generator's own star evaluation
    /// produces. No new recognition needed in this file: `sum_call_over_
    /// star`'s own `star_numeric_hull` gate already accepts any
    /// repetition-window `Kind::Set` regardless of whether a generator,
    /// a list comprehension, or a declared `list[X]` parameter produced
    /// it — the three are indistinguishable once evaluated to this
    /// shape.
    #[test]
    fn sum_over_a_star_shaped_nonnegative_float_window_answers_the_lower_bound_ray() {
        let squared = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(
                make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
                    make_refined_set(vec![at_least(0.0), at_most(1.0)]),
                    0,
                    None,
                )]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        };
        let got = builtin_call_result("sum", &[squared]).expect("sum(star-shaped [0,1] window) must decide through sum_call_over_star");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
        // every element is nonnegative, so the running total only ever
        // moves up from the start value (0) — `sum_call_over_star`'s own
        // nonnegative-branch doc
        let want = make_refined_set(vec![at_least(0.0)]);
        assert_eq!(got.set, want);
    }

    #[test]
    fn sorted_over_known_list_ascending() {
        let list = known_list(vec![integer(3.0), integer(1.0), integer(2.0)], TrustSpec);
        let got = builtin_call_result("sorted", &[list]).expect("sorted([...]) models");
        assert_eq!(got.kind, Kind::List);
        assert_eq!(got.items, vec![integer(1.0), integer(2.0), integer(3.0)]);
    }

    #[test]
    fn list_constructor_copies_a_known_list() {
        let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
        let got = builtin_call_result("list", &[list]).expect("list([...]) models");
        assert_eq!(got.kind, Kind::List);
        assert_eq!(got.items, vec![integer(1.0), integer(2.0)]);
    }

    #[test]
    fn set_constructor_copies_a_known_list() {
        let list = known_list(vec![integer(1.0)], TrustSpec);
        let got = builtin_call_result("set", &[list]).expect("set([...]) models");
        assert_eq!(got.kind, Kind::List);
        assert_eq!(got.items, vec![integer(1.0)]);
    }

    #[test]
    fn set_bare_constructor_answers_the_empty_list() {
        let got = builtin_call_result("set", &[]).expect("set() models");
        assert_eq!(got.kind, Kind::List);
        assert_eq!(got.items.len(), 0);
    }

    #[test]
    fn dict_constructor_from_pairs() {
        let pair_a = known_list(vec![string_value("ann"), integer(40.0)], TrustSpec);
        let pair_b = known_list(vec![string_value("bea"), integer(200.0)], TrustSpec);
        let pairs = known_list(vec![pair_a, pair_b], TrustSpec);
        let got = builtin_call_result("dict", &[pairs]).expect("dict([...]) models");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.keys.len(), 2);
    }

    #[test]
    fn dict_constructor_repeated_key_keeps_the_last_value() {
        let pair_a = known_list(vec![string_value("ann"), integer(1.0)], TrustSpec);
        let pair_b = known_list(vec![string_value("ann"), integer(2.0)], TrustSpec);
        let pairs = known_list(vec![pair_a, pair_b], TrustSpec);
        let got = builtin_call_result("dict", &[pairs]).expect("dict([...]) models");
        assert_eq!(got.keys.len(), 1);
        assert_eq!(got.keys[0].value, integer(2.0));
    }

    /// `xs: list[int]`'s own seeded shape — a `Kind::Set` repetition
    /// window (`check.rs::seed_parameters`'s own sequence-container
    /// branch, `loops.rs`'s own `for`-loop reader) — bounded `[lo, hi]`
    /// with element `[element_lo, element_hi]`. This test module's own
    /// stand-in receiver for every `dict.fromkeys`/`list(...)` row below.
    fn integer_repetition_window(element_lo: f64, element_hi: f64, lo: i64, hi: Option<i64>) -> AbstractValue {
        let element = make_refined_set(vec![at_least(element_lo), at_most(element_hi), refined_sets::refinement_forms::integer()]);
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element, lo, hi)]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        }
    }

    /// `A15.xfer.dedupe`'s own `dict.fromkeys(xs)` row: a `list[int]`
    /// bounded `[0, 150]` answers a round-trip carrier value — `Kind::
    /// Object`, `DICT_FROMKEYS_WORD`, `xs` itself carried in `inner` —
    /// never a `Kind::List`/`Kind::Object` dict directly (this domain's
    /// dict cannot represent windowed, non-string keys, `dict_fromkeys_
    /// call`'s own doc).
    #[test]
    fn dict_fromkeys_over_a_windowed_list_answers_a_round_trip_carrier() {
        let xs = integer_repetition_window(0.0, 150.0, 0, None);
        let got = stdlib_call_result("dict", "fromkeys", &[xs.clone()]).expect("dict.fromkeys(xs) must decide");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.kind_word, Some(DICT_FROMKEYS_WORD));
        assert_eq!(got.inner.as_deref(), Some(&xs));
    }

    /// `dict.fromkeys(xs, 0)` — the two-argument form, `value` explicit
    /// rather than defaulted — still reads the SAME iterable `dict_
    /// fromkeys_call`'s own doc states this row does not otherwise
    /// inspect `value` for.
    #[test]
    fn dict_fromkeys_two_argument_form_still_reads_the_iterable() {
        let xs = integer_repetition_window(0.0, 150.0, 0, None);
        let got = stdlib_call_result("dict", "fromkeys", &[xs.clone(), integer(0.0)]).expect("dict.fromkeys(xs, 0) must decide");
        assert_eq!(got.inner.as_deref(), Some(&xs));
    }

    /// A non-repetition-window argument (an exact `Kind::List`, this
    /// domain's own EXACT-arity container — `dict.fromkeys`'s own row is
    /// scoped to the unbounded-count windowed shape only) declines.
    #[test]
    fn dict_fromkeys_over_an_exact_list_declines() {
        let xs = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
        assert_eq!(stdlib_call_result("dict", "fromkeys", &[xs]), None);
    }

    /// `A15.xfer.dedupe`'s own full row: `list(dict.fromkeys(xs))` for
    /// `xs: list[int]` bounded `[0, 150]` answers the SAME element window
    /// at a RELAXED length bound (`lo: 0`, `hi` unchanged) — the
    /// `for x in deduped:` loop that follows reads `x` through the
    /// identical `as_repetition` path a plain `list[int]` parameter
    /// already flows through (`loops.rs`), so `0 <= x <= 150` narrows it
    /// the same way.
    #[test]
    fn list_of_dict_fromkeys_answers_the_deduped_element_window() {
        let xs = integer_repetition_window(0.0, 150.0, 3, Some(10));
        let carrier = stdlib_call_result("dict", "fromkeys", &[xs]).expect("dict.fromkeys(xs) must decide");
        let got = builtin_call_result("list", &[carrier]).expect("list(dict.fromkeys(xs)) must decide");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
        let expected_element = make_refined_set(vec![at_least(0.0), at_most(150.0), refined_sets::refinement_forms::integer()]);
        let expected = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(expected_element, 0, Some(10))]);
        assert_eq!(got.set, expected, "dedup relaxes lo to 0, keeps hi unchanged, keeps the same element window");
    }

    /// `list(...)` over an ORDINARY exact `Kind::List` argument still
    /// takes the pre-existing row unchanged — the carrier check is
    /// gated on `Kind::Object`/`DICT_FROMKEYS_WORD` and never fires for
    /// this shape, so `list_constructor_call`'s own long-standing
    /// behavior is undisturbed.
    #[test]
    fn list_of_an_ordinary_list_is_unaffected_by_the_carrier_check() {
        let items = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
        let got = builtin_call_result("list", &[items]).expect("list([...]) models");
        assert_eq!(got.kind, Kind::List);
        assert_eq!(got.items.len(), 2);
    }

    fn string_value(text: &str) -> AbstractValue {
        let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
        known_values(code_points, PrimitiveKind::String, TrustSpec)
    }

    #[test]
    fn len_declines() {
        let got = builtin_call_result("len", &[integer(3.0)]);
        assert!(got.is_none());
    }

    #[test]
    fn sum_declines() {
        let got = builtin_call_result("sum", &[integer(3.0)]);
        assert!(got.is_none());
    }

    #[test]
    fn unmodeled_name_declines() {
        let got = builtin_call_result("print", &[integer(3.0)]);
        assert!(got.is_none(), "an unmodeled builtin name should decline: {got:?}");
    }

    #[test]
    fn iter_of_a_known_list_reads_as_the_same_list() {
        let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
        let got = builtin_call_result("iter", &[list.clone()]).expect("iter([...]) models");
        assert_eq!(got, list);
    }

    #[test]
    fn iter_of_a_non_list_declines() {
        let got = builtin_call_result("iter", &[integer(1.0)]);
        assert!(got.is_none());
    }

    #[test]
    fn next_of_iter_of_a_known_list_answers_the_first_element() {
        let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
        let iterator = builtin_call_result("iter", &[list]).expect("iter([...]) models");
        let got = builtin_call_result("next", &[iterator]).expect("next(iter([...])) models");
        assert_eq!(got, integer(1.0));
    }

    #[test]
    fn next_of_an_empty_list_declines() {
        let empty = known_list(vec![], TrustSpec);
        let got = builtin_call_result("next", &[empty]);
        assert!(got.is_none(), "next() over an empty iterator should decline: {got:?}");
    }

    /// `anext` — the async twin of `next`, e-class-and-function.py's own
    /// `async_generator_first_value`/`generator_first_value` pair: a
    /// generator-tagged List (or a plain iterator List) answers its
    /// first element identically whether read through `next` or `anext`.
    #[test]
    fn anext_of_a_generator_tagged_list_answers_the_first_yielded_value() {
        let mut generator = known_list(vec![integer(40.0), integer(41.0)], TrustSpec);
        generator.source = "generator".to_owned();
        let got = builtin_call_result("anext", &[generator]).expect("anext(generator) models");
        assert_eq!(got, integer(40.0));
    }

    #[test]
    fn anext_of_an_empty_list_declines() {
        let empty = known_list(vec![], TrustSpec);
        let got = builtin_call_result("anext", &[empty]);
        assert!(got.is_none(), "anext() over an empty generator should decline: {got:?}");
    }

    #[test]
    fn cast_returns_the_value_argument_unchanged() {
        // the `typ` argument is never read by `cast` — an unknown value
        // there does not block the answer
        let unread_type_argument = AbstractValue::default();
        let got = builtin_call_result("cast", &[unread_type_argument, integer(200.0)]).expect("cast(...) models");
        assert_eq!(got, integer(200.0));
    }

    #[test]
    fn cast_wrong_arity_declines() {
        let got = builtin_call_result("cast", &[integer(200.0)]);
        assert!(got.is_none());
    }

    fn exception_instance(message: &str) -> AbstractValue {
        let args = known_list(vec![string_value(message)], TrustSpec);
        let mut instance = known_object_helper(vec![("args", args)]);
        instance.source = "exception".to_owned();
        instance
    }

    fn known_object_helper(entries: Vec<(&str, AbstractValue)>) -> AbstractValue {
        use refined_domain::abstract_value::ObjectKey;
        use refined_domain::known_constructors::known_object;
        let keys = entries
            .into_iter()
            .map(|(name, value)| ObjectKey { name: name.to_owned(), numeric: false, value })
            .collect();
        known_object(keys, None, true, TrustSpec, false)
    }

    #[test]
    fn str_of_a_single_string_argument_exception_answers_the_message() {
        let instance = exception_instance("failure");
        let got = builtin_call_result("str", &[instance]).expect("str(Exception(...)) models");
        assert_eq!(exact_text(&got), "failure");
    }

    fn exact_text(value: &AbstractValue) -> String {
        value.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect()
    }

    #[test]
    fn str_of_an_exception_with_no_args_declines() {
        let mut instance = known_object_helper(vec![("args", known_list(vec![], TrustSpec))]);
        instance.source = "exception".to_owned();
        let got = builtin_call_result("str", &[instance]);
        assert!(got.is_none(), "a zero-argument exception's __str__ (empty string) is not modeled: {got:?}");
    }

    fn integer_window(lo: f64, hi: f64) -> AbstractValue {
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![at_least(lo), at_most(hi), refined_sets::refinement_forms::integer()]),
                None,
                TrustSpec,
                SetKindTag::None,
            )
        }
    }

    #[test]
    fn str_of_a_bounded_integer_window_answers_the_decimal_digit_grammar() {
        // str(n) over n in [0, 255]: the decimal spelling runs 1 to 3
        // digits ("0".."255"), every digit drawn from 0-9 —
        // `integer_window_grammar`'s own composition, reused unchanged.
        let got = builtin_call_result("str", &[integer_window(0.0, 255.0)]).expect("str(n) over [0, 255] models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
        let digits: Vec<f64> = "0123456789".chars().map(|c| c as u32 as f64).collect();
        let expected = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
            make_refined_set(vec![one_of(&digits)]),
            1,
            Some(3),
        )]);
        assert_eq!(got.set, expected);
    }

    #[test]
    fn str_of_a_negative_lower_bound_integer_window_declines() {
        // `integer_window_grammar`'s own `lo < 0` refusal — no signed
        // digit-run grammar is built here.
        let got = builtin_call_result("str", &[integer_window(-5.0, 5.0)]);
        assert!(got.is_none(), "str(n) over a window with a negative lower bound should decline: {got:?}");
    }

    #[test]
    fn object_call_answers_an_opaque_value_tagged_for_identity_keying() {
        let got = builtin_call_result("object", &[]).expect("object() models");
        assert_eq!(got.kind, Kind::Object);
        assert_eq!(got.kind_word, Some("a featureless object"));
        assert_eq!(got.source, "object()");
    }

    #[test]
    fn object_call_with_an_argument_declines() {
        // library/functions.html#object: "The constructor does not
        // accept any arguments."
        let got = builtin_call_result("object", &[integer(1.0)]);
        assert!(got.is_none(), "object(x) should decline: {got:?}");
    }

    #[test]
    fn hash_of_a_bounded_int_answers_the_unbounded_integer_sort() {
        // hash(x) for x: int is a Python int (library/functions.html#hash),
        // but this row states no identity claim beyond the sort — a
        // later band guard is what narrows it, exactly A15.xfer.hash's
        // own fixture shape. The set carries an EXPLICIT AtLeast(-inf)
        // ray alongside Integer — the same two-form "whole integer
        // ground" shape `narrowing.rs::unbounded_integers()` and this
        // file's own `int_image()` both build — never Integer alone
        // with zero ray forms (A15.xfer.hash's own `hash_outside` row:
        // a bare-Integer set with no ray form let a one-sided `>= 0`
        // guard's own narrowed window silently pass a declared-bounded
        // Age sink).
        let bounded = integer_window(0.0, 150.0);
        let got = builtin_call_result("hash", &[bounded]).expect("hash(x) models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
        let want = make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(f64::NEG_INFINITY)]);
        assert_eq!(got.set, want, "hash(x) must carry an explicit unbounded ray, not a bare Integer form: got {:?}", got.set);
    }

    #[test]
    fn hash_wrong_arity_declines() {
        let got = builtin_call_result("hash", &[]);
        assert!(got.is_none());
    }

    /// A15.xfer.hash's own `hash_outside` soundness row, pinned directly
    /// against the kernel: `hash(x)`'s own unbounded-both-ways ground,
    /// narrowed by `h >= 0` alone (the ONE ray a one-sided guard can ever
    /// tighten — `narrowing.rs::meet_set_answer`'s own intersection),
    /// must NOT prove a subset of Age's declared `[0, 150] && integer`
    /// window — an unbounded-above ray is never contained in a set
    /// bounded above, so `assignability.rs`'s own `scalar_subset` ask
    /// (the exact containment question `judge`'s `Kind::Set` arm poses at
    /// the `return h` sink) must answer `false` here. Before this fix,
    /// `hash_call`'s bare-`Integer` set (no ray form at all) reached this
    /// same ask and was silently admitted — the reproducer this test
    /// pins the refusal for.
    #[test]
    fn hash_narrowed_only_below_is_not_a_subset_of_a_bounded_declared_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let bare_int = integer_window(0.0, 150.0);
        let hash_result = builtin_call_result("hash", &[bare_int]).expect("hash(x) models");
        // `h >= 0` narrows only the lower ray — the SAME `meet_set_answer`
        // intersection `narrowing.rs` performs for a one-sided guard,
        // reproduced here directly on the set rather than through the
        // full narrowing walk, since this file's own tests stay
        // kernel-optional and narrowing-free.
        let mut narrowed_forms = hash_result.set.forms.clone();
        narrowed_forms.push(at_least(0.0));
        let narrowed_set = make_refined_set(narrowed_forms);
        let age_declared = make_refined_set(vec![at_least(0.0), at_most(150.0), refined_sets::refinement_forms::integer()]);
        let is_subset = (kernel.scalar_subset)(&narrowed_set, &age_declared);
        assert!(
            !is_subset,
            "hash(x) narrowed only below by `>= 0` must not be a subset of Age's bounded window: narrowed {:?}, declared {:?}",
            narrowed_set, age_declared
        );
    }

    #[test]
    fn time_time_answers_a_nonnegative_float_ground() {
        let got = stdlib_call_result("time", "time", &[]).expect("time.time() models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
        assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
    }

    #[test]
    fn time_time_with_an_argument_declines() {
        let got = stdlib_call_result("time", "time", &[integer(1.0)]);
        assert!(got.is_none());
    }

    #[test]
    fn os_open_answers_a_nonnegative_integer_ground() {
        let got = stdlib_call_result("os", "open", &[string_value("/tmp/x"), integer(0.0)]).expect("os.open(...) models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
        assert_eq!(
            got.set,
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::integer()])
        );
    }

    #[test]
    fn os_close_answers_none() {
        let got = stdlib_call_result("os", "close", &[integer(3.0)]).expect("os.close(fd) models");
        assert_eq!(got.kind, Kind::Null);
    }

    #[test]
    fn unicodedata_normalize_nfc_answers_the_whole_strings_ground() {
        let form = string_value("NFC");
        let subject = string_value("e\u{0301}");
        let got = stdlib_call_result("unicodedata", "normalize", &[form, subject]).expect("unicodedata.normalize(...) models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
    }

    #[test]
    fn unicodedata_normalize_with_an_unknown_form_declines() {
        let form = string_value("bogus");
        let subject = string_value("x");
        let got = stdlib_call_result("unicodedata", "normalize", &[form, subject]);
        assert!(got.is_none(), "an unrecognized normalization form should decline: {got:?}");
    }

    #[test]
    fn urllib_parse_quote_answers_the_whole_strings_ground() {
        // reached as a bare-name builtin (`from urllib.parse import
        // quote` then `quote(s)`), not through stdlib_call_result — see
        // urllib_quote_call's own doc.
        let subject = string_value("a b");
        let got = builtin_call_result("quote", &[subject]).expect("quote(...) models");
        assert_eq!(got.kind, Kind::Set);
        assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
    }

    #[test]
    fn unmodeled_stdlib_module_declines() {
        let got = stdlib_call_result("sys", "exit", &[]);
        assert!(got.is_none());
    }
}
