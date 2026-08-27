
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;

use crate::collection_models;
use crate::env::Environment;
use crate::string_models;

use super::evaluate_expression;
use super::arithmetic::*;
use super::compare::*;
use super::sequence_ops::*;

/// `s[lower:upper]` / `xs[lower:upper]`, no `step` (expressions.rst,
/// "Slicings" — a slicing indexes via the same `__getitem__` machinery
/// as a plain subscript, with the slice's own bounds silently CLAMPED to
/// `[0, len(s)]` rather than raising — the one place this domain's
/// plain-index honesty ("out of range declines") does not apply, because
/// a slice never raises for an out-of-range bound the way a single index
/// does). Missing `lower` defaults to 0, missing `upper` defaults to
/// `len(s)` — the same defaults `s[:]`/`s[n:]`/`s[:n]` read under.
/// Negative bounds adjust by the sequence's own length first, matching
/// the plain-index rule (`known_integer_index`'s own negative-
/// adjustment). A `step` is not modeled (declines outright, per the
/// mission's own scope). Two known receiver shapes: a known exact
/// string (`Kind::Values` tagged `PrimitiveKind::String`) answers a
/// SLICED STRING (`c-reads-and-values.py`'s own string-slicing rows);
/// a known list/tuple (`Kind::List` — this domain's shared sequence
/// shape, `collection_models.rs`'s own module doc: list slicing shares
/// the identical clamp-not-raise rule "Slicings" states for every
/// built-in sequence, so this is the SAME bound-computation this
/// function already carries, only reading `items` instead of `values`)
/// answers a SLICED LIST (`c-reads-and-values.py`'s `list_slice`:
/// `overs[0:1][0]`, a slice immediately re-subscripted). A THIRD
/// receiver shape answers a narrower claim: a string-shaped SET (a
/// concatenation/repeat window, never an exact literal — those are
/// `Kind::Values` and already answered above) sliced exactly `[:n]`
/// (`lower` absent or the exact literal `0`, `upper` an exact
/// non-negative Integer, no `step`) asks the kernel's `seq_prefix` —
/// `prefixReadOf`'s proved over-approximation of `take n` on every
/// member (boundary/exports_sets.lean's `kernelSeqPrefix`). Any other
/// slice shape over a non-exact receiver — a `step`, a nonzero `lower`,
/// or an `upper` that is not an exact Integer — declines, naming the
/// construct it cannot read rather than guessing.
pub(super) fn evaluate_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if slice.step.is_some() {
        return unknown();
    }
    if let Some(result) = sequence_prefix_slice(container, slice, environment, kernel) {
        return result;
    }
    if let Some(result) = repetition_window_slice(container, slice, environment, kernel) {
        return result;
    }
    let length = match container.kind {
        Kind::Values if container.kind_tag == Some(PrimitiveKind::String) => container.values.len() as i64,
        Kind::List => container.items.len() as i64,
        _ => return unknown(),
    };
    let lower = match &slice.lower {
        Some(expr) => match slice_bound_index(expr, environment, kernel) {
            Some(value) => value,
            None => return unknown(),
        },
        None => 0,
    };
    let upper = match &slice.upper {
        Some(expr) => match slice_bound_index(expr, environment, kernel) {
            Some(value) => value,
            None => return unknown(),
        },
        None => length,
    };
    let clamped_lower = clamp_slice_bound(lower, length);
    let clamped_upper = clamp_slice_bound(upper, length);
    match container.kind {
        Kind::Values => {
            if clamped_lower >= clamped_upper {
                return string_models::string_literal_value("");
            }
            let slice_points = container.values[clamped_lower as usize..clamped_upper as usize].to_vec();
            known_values(slice_points, PrimitiveKind::String, TrustProved)
        }
        Kind::List => {
            if clamped_lower >= clamped_upper {
                return collection_models::list_literal_value(&[]);
            }
            let slice_items = container.items[clamped_lower as usize..clamped_upper as usize].to_vec();
            collection_models::list_literal_value(&slice_items)
        }
        _ => unreachable!("container.kind checked above in the length match"),
    }
}

/// The `[:n]` prefix-read arm of `evaluate_slice`: fires only when the
/// receiver is a string-shaped SET (`string_shaped_set` — a concatenation
/// or repeat window; an exact literal is `Kind::Values` and already
/// answered by `evaluate_slice`'s own Values arm, so this never
/// double-answers that case) AND the slice is exactly `[:n]` — `lower`
/// absent or the exact known value `0`, `upper` an exact known
/// non-negative Integer, no `step` (checked by the caller before this
/// runs). Any other slice shape over a set-shaped receiver — a nonzero
/// or unknown `lower`, or an `upper` that is not an exact non-negative
/// Integer — answers `None` immediately, so the caller's own decline
/// stands rather than this function guessing. The kernel itself can
/// ALSO decline once the shape is asked (`kernel.seq_prefix`'s own
/// `None` — the receiver is not `seqOf`-recognized, e.g. a leading
/// concatenation operand that is not a fixed scalar): that decline is an
/// ORDINARY answer, not a fault, and this function answers `None` for it
/// exactly the same way, so the caller falls through to the
/// length-based fallback precisely as if this arm had never matched.
pub(super) fn sequence_prefix_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    // An exact literal (`Kind::Values`) is already answered exactly by
    // `evaluate_slice`'s own Values arm below; asking the kernel's
    // window-shaped `seq_prefix` for it would answer a WIDER claim (a
    // shape over an unstated member) in place of the exact slice this
    // domain already has, so this arm only reads a genuine window
    // receiver — `string_shaped_set`'s `Kind::Set` branch, never its
    // exact-literal branch (that branch exists for `string_set_concatenation`,
    // not this caller).
    if container.kind != Kind::Set {
        return None;
    }
    let receiver_set = string_shaped_set(container)?;
    if let Some(expr) = &slice.lower {
        if slice_bound_index(expr, environment, kernel) != Some(0) {
            return None;
        }
    }
    let upper_expr = slice.upper.as_ref()?;
    let n = slice_bound_index(upper_expr, environment, kernel)?;
    if n < 0 {
        return None;
    }
    let prefix_set = (kernel.seq_prefix)(&unbounded_repeats(&receiver_set), n)?;
    Some(known_set(prefix_set, None, TrustProved, SetKindTag::None))
}

/// `xs[lower:upper]` on an UNKNOWN-LENGTH, known-element sequence — a
/// `Kind::Set` whose only form is a repetition window (`as_repetition`,
/// the shape `check/seed.rs::seed_parameters` builds for a declared
/// `list[X]`/`Sequence[X]` parameter, and the shape `attribute.rs`'s
/// `sys.argv` read answers). Distinct from `sequence_prefix_slice`
/// above, which asks the kernel about a STRING-shaped window's exact
/// prefix grammar: this row answers the LIST-shaped window's own two
/// facts, which need no kernel round trip at all.
///
/// Both facts come from expressions.rst, "Slicings," and stdtypes.rst's
/// "Common Sequence Operations" `s[i:j]` row ("slice of *s* from *i* to
/// *j*"): every element of the slice is an element of `s` (a slice
/// selects positions, it never builds a value outside the sequence's own
/// alphabet), and the slice's length is at most `s`'s own length (a
/// slice never grows a sequence). So the answer is the SAME element set
/// repeated, with the length window relaxed at the low end to `0` — a
/// slice can select nothing at all, since "Slicings" clamps an
/// out-of-range bound rather than raising — and unchanged at the high
/// end.
///
/// The WHOLE slice `s[:]` is its own arm: both bounds absent selects
/// every position, so the receiver copies through unchanged rather than
/// losing its own length window to the relaxation below.
///
/// A KNOWN non-negative `lower` tightens the high end further by
/// dropping that many positions (`hi - lower`, floored at 0), the exact
/// count `s[lower:]` skips. An unknown or negative `lower`, or any
/// `upper` this file cannot read, keeps the receiver's own `hi` — still
/// sound, since neither can make the slice longer than `s`. A `step`
/// never reaches here (the caller returns before this row).
///
/// `None` when the receiver is not a repetition window at all, so the
/// caller's own exact-length rows and final decline stand unchanged.
pub(super) fn repetition_window_slice(
    container: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if container.kind != Kind::Set || container.set_kind_tag != SetKindTag::None {
        return None;
    }
    let repeated = as_repetition(&container.set)?;
    // `s[:]` — BOTH bounds absent — is the whole sequence, so the length
    // is unchanged rather than merely bounded above by it: "Slicings"'
    // own defaults make `lower` 0 and `upper` `len(s)`, which selects
    // every position. This is `A7.xfer.copy`'s own `shallow_copy` shape,
    // where relaxing `lo` would discard a length fact the copy actually
    // keeps.
    let whole_sequence = slice.lower.is_none() && slice.upper.is_none();
    if whole_sequence {
        return Some(container.clone());
    }
    let dropped = match &slice.lower {
        Some(expr) => slice_bound_index(expr, environment, kernel).filter(|bound| *bound >= 0).unwrap_or(0),
        None => 0,
    };
    let high = repeated.hi.map(|hi| (hi - dropped).max(0));
    let sliced = refined_sets::repetition_window_forms::repetition(repeated.element, 0, high);
    Some(AbstractValue {
        kind_tag: container.kind_tag,
        ..known_set(sliced, None, trust_level_of(container), SetKindTag::None)
    })
}

/// Relaxes every `Repeat`/`RepeatWord` form reachable through the set's
/// own concatenation/union/difference/star operands to its UNBOUNDED
/// twin (`hi: None`) — sound because a bounded window's every member is
/// trivially a member of the same window with its ceiling dropped
/// (widening a claim only ever admits more), and `seqOf`
/// (set_functions/subset_seq_shape.lean) recognizes a `Repeat` position
/// only when `hi` is `none`. `seq_prefix`'s own soundness
/// (`prefixReadOf_sound`) never reads the receiver's ceiling — its
/// premise is `SeqDen`, which states membership per position and the
/// open tail's `Star`, neither of which mentions `hi` — so asking the
/// kernel about this relaxed receiver instead of the tighter original
/// still proves the same `take n` fact for both. A parameter's own
/// `Field(min_length=…, max_length=…)` window is exactly the shape this
/// exists for (`check.rs::seed_parameters`'s own doc): the LENGTH bound
/// still narrows what `evaluate_slice`'s caller reads through `len()`
/// elsewhere; only the prefix READ over-approximates.
pub(super) fn unbounded_repeats(set: &RefinedSet) -> RefinedSet {
    let forms = set
        .forms
        .iter()
        .map(|form| {
            let mut relaxed = form.clone();
            if matches!(form.form, Form::Repeat | Form::RepeatWord) {
                relaxed.hi = None;
            }
            if let Some(a) = &form.a_ {
                relaxed.a_ = Some(Box::new(unbounded_repeats(a)));
            }
            if let Some(b) = &form.b {
                relaxed.b = Some(Box::new(unbounded_repeats(b)));
            }
            relaxed
        })
        .collect();
    make_refined_set(forms)
}

/// One slice bound's known Integer value, or `None` if it is not a
/// single known Integer-sorted expression — the same acceptance
/// `known_integer_index` (collection_models.rs) gives a plain
/// subscript index, evaluated here instead since a slice bound is an
/// EXPRESSION (`lower_bound: expression`, expressions.rst's own
/// grammar) rather than an already-evaluated AbstractValue.
pub(crate) fn slice_bound_index(expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<i64> {
    let value = evaluate_expression(expr, environment, kernel);
    let (number, sort) = single_numeric_value(&value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    Some(number as i64)
}

/// Adjusts a slice bound by the sequence's own length (negative bounds
/// count from the end, the same rule a plain index follows) and then
/// CLAMPS to `[0, length]` — a slice bound never raises for landing
/// outside that range, unlike a plain index (expressions.rst,
/// "Slicings"' own silent-clamp behavior).
pub(super) fn clamp_slice_bound(bound: i64, length: i64) -> i64 {
    let adjusted = if bound < 0 { bound + length } else { bound };
    adjusted.clamp(0, length)
}

/// `x < y <= z` chains as `x < y and y <= z`, evaluating `y` once
/// (expressions.rst, "Comparisons": "x < y <= z is equivalent to x < y
/// and y <= z, except that y is evaluated only once"). Every adjacent
/// pair must decide `True` for the whole chain to decide `True`; the
/// moment one pair cannot be decided, the whole chain is unknown — a
/// chain never answers partial knowledge.
pub(super) fn evaluate_compare(compare: &ruff_python_ast::ExprCompare, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let left = evaluate_expression(&compare.left, environment, kernel);
    let mut operands = Vec::with_capacity(compare.comparators.len());
    for comparator in compare.comparators.iter() {
        operands.push(evaluate_expression(comparator, environment, kernel));
    }
    let mut previous = &left;
    for (op, operand) in compare.ops.iter().zip(operands.iter()) {
        let Some(result) = compare_pair(*op, previous, operand, kernel) else {
            // a SINGLE-PAIR `in`/`not in` chain (`x in y`, never a
            // chained `a in b in c`) against a known List container
            // whose own element equality this file cannot decide (e.g.
            // a container of opaque class-instance elements,
            // weakref.WeakSet's own `.add(key)` shape) still provably
            // answers A BOOLEAN — expressions.rst, "Comparisons": every
            // `in`/`not in` expression's result IS `bool`, regardless of
            // which value it resolves to. Answered opaque (a boolean
            // sort with no further-known value) rather than fully
            // unknown, so a scalar-ground sink (Age) still fires through
            // the opaque law instead of sitting undetermined. Scoped to
            // a ONE-PAIR chain only — a longer chain still declines
            // fully, since an undecided EARLIER pair could still make
            // the later pair's own truthiness matter to which operand
            // Python's short-circuit `and` would even reach.
            if compare.ops.len() == 1 && matches!(op, CmpOp::In | CmpOp::NotIn) && operand.kind == Kind::List {
                return opaque_value("a boolean value");
            }
            // Which of the two values this pair resolves to is undecided,
            // but THAT it is one of the two is stated by the language:
            // expressions.rst, "Comparisons" — "Comparisons yield boolean
            // values: ``True`` or ``False``." So the answer is the exact
            // two-member boolean domain, never `unknown()`. A downstream
            // `int(...)` then reads `{0, 1}` through
            // `builtin_models::boolean_operand_as_int_values` instead of
            // widening to `int_image`'s unbounded ray, and a sink declared
            // `Literal[True]` refuses this two-member set instead of
            // sitting undetermined.
            return known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec);
        };
        if result != 1.0 {
            return known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved);
        }
        previous = operand;
    }
    known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved)
}
