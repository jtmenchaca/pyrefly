
use std::sync::Arc;

use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::truthiness;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::evaluate_expression;
use super::arithmetic::*;
use super::fstring::*;

/// One comparison operator over two already-evaluated operands: `1.0`
/// (True), `0.0` (False), or `None` (not decidable). Every row here
/// requires both operands KNOWN — an unknown operand always declines
/// the whole pair, which `evaluate_compare` turns into unknown() for
/// the whole chain.
pub(super) fn compare_pair(op: CmpOp, left: &AbstractValue, right: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<f64> {
    // `is` / `is not` decide identity against None only: expressions.rst,
    // "Comparisons," `is`/`is not` — "None" is the one CPython value this
    // file can prove identity for without a shared-object model. Either
    // side being the exactly-null state (`Kind::Null`) settles it: None
    // is None (True/False split by op), and a known non-None value is
    // never identical to None. A `Kind::PossiblyUndefined` side (an
    // `Optional[X]`/`X | None`-declared parameter's own seed,
    // `check.rs::seed_parameters`) is NOT a known non-None value the way
    // an ordinary present-only Kind settles as — its own absent side may
    // genuinely BE None at runtime, so its identity against None stays
    // undecided here exactly like Unknown's, leaving `narrowing.rs`'s
    // `narrow_is_none` (the maybe carrier's own unwrap) to state what
    // each fork actually proves instead of this law guessing one arm
    // dead outright.
    if op == CmpOp::Is || op == CmpOp::IsNot {
        let identical = match (left.kind == Kind::Null, right.kind == Kind::Null) {
            (true, true) => true,
            (true, false) | (false, true) => {
                if right.kind == Kind::Unknown
                    || left.kind == Kind::Unknown
                    || right.kind == Kind::PossiblyUndefined
                    || left.kind == Kind::PossiblyUndefined
                {
                    return None;
                }
                false
            }
            (false, false) => return None,
        };
        let result = if op == CmpOp::Is { identical } else { !identical };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // `in` / `not in`: a known needle against a known List container of
    // known elements — membership by `==` on each element's own value
    // (expressions.rst, "Comparisons," `in`/`not in`: "For container
    // types such as list, tuple, set, frozenset, dict, or collections.deque,
    // the expression `x in y` is equivalent to `any(x is e or x == e for e
    // in y)`" — this row reads the `==` half, since exact-value equality
    // already decides `is` for two equal known scalars).
    if op == CmpOp::In || op == CmpOp::NotIn {
        // `key in d` on a MAPPING receiver — the same "Comparisons" row
        // names dict among its container types, and stdtypes.rst's own
        // Mapping Types section states what membership means there:
        // "`key in d` — Return `True` if *d* has a key *key*, else
        // `False`." So this is a question about the KEY SET, never about
        // the values, and a closed dict states its key set exactly.
        if right.kind == Kind::Object && right.kind_word.is_none() {
            let key = crate::collection_models::known_dict_key(left)?;
            let present = right.keys.iter().any(|entry| entry.name == key.name && entry.numeric == key.numeric);
            let result = if op == CmpOp::In { present } else { !present };
            return Some(if result { 1.0 } else { 0.0 });
        }
        // An UNBOUNDED-KEY mapping (`Kind::ObjectStar`) states no key set
        // to decide against — except for the keys it was WRITTEN at,
        // which are recorded entries and are therefore PRESENT
        // (`collection_models::dict_with_item`'s own star arm). A key with
        // no recorded entry stays undecided: the declaration never said
        // which keys the mapping arrived holding, so its absence is not
        // provable. That is what A8.xfer.getorinsert's own
        // `presence_after_insert` row needs — `setdefault` inserts the
        // key, so `k in d` afterward is provably True.
        if right.kind == Kind::ObjectStar {
            let key = crate::collection_models::known_dict_key(left)?;
            let present = right.keys.iter().any(|entry| entry.name == key.name && entry.numeric == key.numeric);
            if !present {
                return None;
            }
            return Some(if op == CmpOp::In { 1.0 } else { 0.0 });
        }
        // A `set[X]` parameter's own REPETITION-WINDOW receiver
        // (`Kind::Set` over the bare star/window shape
        // `refined_sets::repetition_window_forms::as_repetition` reads
        // back — `check.rs::seed_parameters`'s own sequence-container
        // seed) states no fixed member list to decide against, the same
        // "no key list of its own" shape `Kind::ObjectStar` reads just
        // above — EXCEPT for an element `set.add(x)` just WROTE, which
        // is a recorded entry (`collection_models::list_set_mutation::
        // set_mutated_receiver`'s own `add` arm) and therefore PROVABLY
        // present, mirroring the dict-star's "written keys are present"
        // row one arm up. An unrecorded element declines here (`None`)
        // rather than guess absent: the window states only what the set
        // MIGHT hold, never that a given value is NOT a member. The
        // caller (`expressions::subscript::evaluate_compare`) turns that
        // decline into the exact two-member boolean domain for a
        // single-pair `in`/`not in` chain — sound, since `in` always
        // evaluates to `bool` (expressions.rst, "Comparisons") even when
        // WHICH of the two values it resolves to is not pinned.
        if right.kind == Kind::Set
            && right.set_kind_tag == SetKindTag::None
            && refined_sets::repetition_window_forms::as_repetition(&right.set).is_some()
        {
            let key = crate::collection_models::known_dict_key(left)?;
            let present = right.keys.iter().any(|entry| entry.name == key.name && entry.numeric == key.numeric);
            if !present {
                return None;
            }
            return Some(if op == CmpOp::In { 1.0 } else { 0.0 });
        }
        if right.kind != Kind::List {
            return None;
        }
        let mut found = false;
        for element in &right.items {
            match single_pair_equal(left, element) {
                Some(true) => {
                    found = true;
                    break;
                }
                Some(false) => continue,
                None => return None,
            }
        }
        let result = if op == CmpOp::In { found } else { !found };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // both single known numeric values: ==, !=, <, <=, >, >= over the
    // f64s directly (expressions.rst, "Comparisons," the numeric-types
    // ordering — CPython orders numbers by mathematical value)
    if let (Some((left_value, _)), Some((right_value, _))) =
        (single_numeric_value(left), single_numeric_value(right))
    {
        let result = match op {
            CmpOp::Eq => left_value == right_value,
            CmpOp::NotEq => left_value != right_value,
            CmpOp::Lt => left_value < right_value,
            CmpOp::LtE => left_value <= right_value,
            CmpOp::Gt => left_value > right_value,
            CmpOp::GtE => left_value >= right_value,
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => unreachable!("handled above"),
        };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // one side a single known numeric value, the OTHER a bounded
    // Integer-sorted window (`integer_set_bounds` — `len()`'s own answer
    // over a `Repeat`-shaped receiver, `collection_models::len_result`'s
    // doc: `[window.lo, window.hi]`, never one exact count): the
    // comparison decides only when the WHOLE window agrees, since the
    // window states every value it admits, never which one a given run
    // actually holds. `==`/`!=` decide only on a DEGENERATE window
    // (`lo == hi`, the len-of-a-fixed-length-slice case) — a wider
    // window can never prove `==` true (some other admitted length
    // would disagree) or `!=` true (the target might be the one held
    // length), so both stay undecided there. The four orderings decide
    // whenever the window sits entirely on one side of the target
    // (`hi <op> target` uniform down to `lo`, or the mirror), which a
    // degenerate window trivially satisfies too.
    if let Some(result) = numeric_value_vs_window_compare(op, left, right, false)
        .or_else(|| numeric_value_vs_window_compare(op, right, left, true))
    {
        return Some(if result { 1.0 } else { 0.0 });
    }
    // one side a single known numeric value, the OTHER a bounded window
    // of ANY numeric sort — not just Integer (`integer_set_bounds`'s own
    // restriction, `numeric_value_vs_window_compare`'s scope above): a
    // `float`-sorted parameter narrowed to an open interval (`0 < x <
    // 1`, A2.sink.dead's own `dead_branch_inside` guard) is exactly this
    // shape. Scoped to `==`/`!=` only, and only the ONE direction those
    // two operators can ever decide without reading the window's
    // interior — the target sits STRICTLY outside `[lo, hi]` (open or
    // closed at each end per the window's own `Above`/`Below` vs
    // `AtLeast`/`AtMost` forms), so `==` is provably false (no member of
    // the window can be `target`) and `!=` is provably true. A target
    // touching or inside the window still declines here exactly as the
    // integer-only path already declines a non-degenerate window's
    // interior — this adds no new claim about window MEMBERS, only the
    // "outside entirely" corner the integer path already proves for
    // `NotEq` (see its own comment) but never proved for `Eq`.
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        let outside = numeric_value_outside_general_window(left, right)
            .or_else(|| numeric_value_outside_general_window(right, left));
        if outside == Some(true) {
            return Some(if op == CmpOp::Eq { 0.0 } else { 1.0 });
        }
    }
    // both known exact strings: == and != read the code-point vectors
    // directly; <, <=, >, >= read them lexicographically — CPython
    // orders strings "lexicographically using the numeric equivalents
    // (the result of the built-in function ord()) of their characters"
    // (expressions.rst, "Comparisons," the sequence-types ordering rule).
    if let (Some(left_text), Some(right_text)) = (exact_string_values(left), exact_string_values(right)) {
        let result = match op {
            CmpOp::Eq => left_text == right_text,
            CmpOp::NotEq => left_text != right_text,
            CmpOp::Lt => left_text < right_text,
            CmpOp::LtE => left_text <= right_text,
            CmpOp::Gt => left_text > right_text,
            CmpOp::GtE => left_text >= right_text,
            CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => unreachable!("handled above"),
        };
        return Some(if result { 1.0 } else { 0.0 });
    }
    // one side a known EXACT string, the other a string-sorted SET (a
    // grammar a guard already narrowed a name to — `re.fullmatch(
    // r"[0-9]+", s)`'s own true arm, A3.sink.dead's own shape). When
    // the set provably does NOT admit that exact string, no run can
    // ever satisfy `==`: every value the name holds is a member of the
    // set, and the target is not one — so `==` is provably False and
    // `!=` provably True. This is the string twin of
    // `numeric_value_outside_general_window` above, and it decides the
    // same ONE direction those two operators can settle without reading
    // the set's interior: a target the set DOES admit still declines,
    // since the set states what values are possible, never which one a
    // given run actually holds.
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        if let Some(admitted) = exact_string_outside_string_set(left, right, kernel)
            .or_else(|| exact_string_outside_string_set(right, left, kernel))
        {
            if !admitted {
                return Some(if op == CmpOp::Eq { 0.0 } else { 1.0 });
            }
        }
    }
    None
}

/// Whether a string-sorted SET admits a KNOWN EXACT string — the
/// kernel's own member decider, the same one
/// `assignability::judge`'s string row asks. `Some(false)` means the
/// set provably excludes that string; `Some(true)` means it admits it;
/// `None` means this pair is not the shape (one exact string, one
/// string-sorted set) or the kernel declined to decide the set's own
/// form. A kernel REFUSAL is caught the same way every other ask in
/// this crate catches one (`kernel_ask::ask_kernel`) and reads as "not
/// decided" rather than a crash.
fn exact_string_outside_string_set(
    string_side: &AbstractValue,
    set_side: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<bool> {
    let code_points = exact_string_values(string_side)?;
    if set_side.kind != Kind::Set {
        return None;
    }
    if !(set_side.kind_tag == Some(PrimitiveKind::String)
        || (set_side.kind_tag.is_none() && crate::assignability::sequence_shaped(&set_side.set)))
    {
        return None;
    }
    crate::kernel_ask::ask_kernel(|| (kernel.member)(&set_side.set, code_points)).ok()
}

/// One comparison operator between a single known numeric value and a
/// bounded Integer-sorted window (`integer_set_bounds`'s own `[lo, hi]`
/// reading), decided only when EVERY value the window admits agrees —
/// the window is a claim over an unstated member, never a promise about
/// which one, so a partial overlap must stay `None` rather than guess.
/// `numeric_side`/`window_side` name which of `compare_pair`'s two
/// operands is being read here. The per-op bodies below read literally
/// as `window_side <op> target` (`Lt` means "every admitted value is
/// below target," matching a name like `low_window` at the call site).
/// `swapped` is `true` when `window_side` is actually `compare_pair`'s
/// LEFT operand, i.e. the original claim already reads `window_side
/// <op> numeric_side` — the same direction the bodies compute, so `op`
/// passes through unchanged. When `swapped` is `false`, the original
/// claim reads `numeric_side <op> window_side` — the mirror of what the
/// bodies compute — so `op` is inverted first (`x < y` read as `y > x`)
/// rather than this function reversing the arithmetic itself.
pub(super) fn numeric_value_vs_window_compare(
    op: CmpOp,
    numeric_side: &AbstractValue,
    window_side: &AbstractValue,
    swapped: bool,
) -> Option<bool> {
    let (target, _) = single_numeric_value(numeric_side)?;
    let (lo, hi) = integer_set_bounds(window_side)?;
    let (lo, hi) = (lo as f64, hi as f64);
    let effective_op = if swapped {
        op
    } else {
        match op {
            CmpOp::Lt => CmpOp::Gt,
            CmpOp::LtE => CmpOp::GtE,
            CmpOp::Gt => CmpOp::Lt,
            CmpOp::GtE => CmpOp::LtE,
            other => other,
        }
    };
    match effective_op {
        // every admitted value equals target only when the window is
        // degenerate AND that one value IS target; a wider window can
        // never prove equality (some other admitted length disagrees)
        CmpOp::Eq => {
            if lo == hi {
                Some(lo == target)
            } else {
                None
            }
        }
        // != is decided the same way `==` is (its exact negation, once
        // decidable), plus the case the window misses target ENTIRELY
        // (every admitted value differs, whether or not the window is
        // degenerate)
        CmpOp::NotEq => {
            if hi < target || target < lo {
                Some(true)
            } else if lo == hi {
                Some(lo != target)
            } else {
                None
            }
        }
        CmpOp::Lt => {
            if hi < target {
                Some(true)
            } else if lo >= target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::LtE => {
            if hi <= target {
                Some(true)
            } else if lo > target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Gt => {
            if lo > target {
                Some(true)
            } else if hi <= target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::GtE => {
            if lo >= target {
                Some(true)
            } else if hi < target {
                Some(false)
            } else {
                None
            }
        }
        CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => unreachable!("handled above compare_pair's own call site"),
    }
}

/// A bounded numeric window's own `(lo, lo_open, hi, hi_open)`, read off
/// any numeric-sorted (`Integer`/`Float`/`Number`) `Kind::Set` — the
/// general-sort twin of `integer_set_bounds`, which restricts to
/// `PrimitiveKind::Integer` and rounds `Above`/`Below` inward by one
/// (sound only for an integer domain, where no value sits strictly
/// between two consecutive integers). A `float`-sorted window keeps its
/// `Above`/`Below` bound EXACTLY at `form.a`, open (`lo_open`/`hi_open`
/// true) — the same real-number reading `refined_sets::refinement_forms`
/// itself gives the two forms, no rounding. `None` for a Set carrying
/// any other form (`MultipleOf`, `OneOf`, a sequence shape) or a
/// non-numeric sort — this reader states only the plain interval a
/// bounded scalar window admits, the shape `dead_branch_inside`'s own
/// `0 < x < 1` guard narrows `x` to.
pub(super) fn general_numeric_set_window(value: &AbstractValue) -> Option<(f64, bool, f64, bool)> {
    if value.kind != Kind::Set {
        return None;
    }
    if !matches!(
        value.kind_tag,
        Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Number)
    ) {
        return None;
    }
    let mut lo: Option<(f64, bool)> = None;
    let mut hi: Option<(f64, bool)> = None;
    for form in &value.set.forms {
        match form.form {
            refined_sets::refinement_forms::Form::AtLeast => {
                lo = Some(match lo {
                    Some((current, open)) if current > form.a => (current, open),
                    _ => (form.a, false),
                })
            }
            refined_sets::refinement_forms::Form::Above => {
                lo = Some(match lo {
                    Some((current, _)) if current > form.a => (current, false),
                    Some((current, _)) if current == form.a => (current, true),
                    _ => (form.a, true),
                })
            }
            refined_sets::refinement_forms::Form::AtMost => {
                hi = Some(match hi {
                    Some((current, open)) if current < form.a => (current, open),
                    _ => (form.a, false),
                })
            }
            refined_sets::refinement_forms::Form::Below => {
                hi = Some(match hi {
                    Some((current, _)) if current < form.a => (current, false),
                    Some((current, _)) if current == form.a => (current, true),
                    _ => (form.a, true),
                })
            }
            refined_sets::refinement_forms::Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, lo_open) = lo?;
    let (hi, hi_open) = hi?;
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo, lo_open, hi, hi_open))
}

/// Whether `numeric_side`'s single known value sits STRICTLY outside
/// `window_side`'s own bounded numeric window — `Some(true)` only, never
/// `Some(false)`: a target inside (or touching a closed bound of) the
/// window is not provably absent from every member, so this reader
/// simply declines (`None`) rather than claim presence, matching
/// `numeric_value_vs_window_compare`'s own "decided only when the WHOLE
/// window agrees" discipline. `general_numeric_set_window` supplies the
/// bounds; either argument order is tried by the caller, so this reads
/// one fixed direction only (`numeric_side` the value, `window_side` the
/// window).
pub(super) fn numeric_value_outside_general_window(numeric_side: &AbstractValue, window_side: &AbstractValue) -> Option<bool> {
    let (target, _) = single_numeric_value(numeric_side)?;
    let (lo, lo_open, hi, hi_open) = general_numeric_set_window(window_side)?;
    let below = if lo_open { target <= lo } else { target < lo };
    let above = if hi_open { target >= hi } else { target > hi };
    if below || above {
        Some(true)
    } else {
        None
    }
}

/// Whether two already-evaluated values are `==`, for the `in`/`not in`
/// membership row: single known numerics compare by value, known exact
/// strings compare by their code-point sequence, and anything else (an
/// unknown side, or a shape this file has no equality row for) declines
/// with `None` rather than guessing.
pub(super) fn single_pair_equal(left: &AbstractValue, right: &AbstractValue) -> Option<bool> {
    if let (Some((left_value, _)), Some((right_value, _))) =
        (single_numeric_value(left), single_numeric_value(right))
    {
        return Some(left_value == right_value);
    }
    if let (Some(left_text), Some(right_text)) = (exact_string_values(left), exact_string_values(right)) {
        return Some(left_text == right_text);
    }
    None
}

/// The code-point vector an AbstractValue carries, if it is a known
/// exact string (`Kind::Values` tagged `PrimitiveKind::String`) — the
/// same shape `string_models.rs` builds; comparing the `Vec<f64>`
/// directly (rather than converting to a Rust `String` first) IS the
/// code-point-by-code-point ordering `str`'s own comparison rule states.
pub(super) fn exact_string_values(value: &AbstractValue) -> Option<&[f64]> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(&value.values)
}

/// `and`/`or` return an OPERAND, never a coerced bool (expressions.rst,
/// "Boolean operations": "the return value of a short-circuit operator
/// is the last evaluated argument"). Walked left to right: `and` stops
/// at the first definitely-falsy operand (that operand IS the answer)
/// and skips past a definitely-truthy one; `or` mirrors it. The moment
/// an operand's truthiness is not decidable, the whole expression
/// declines — a later operand might still have changed the answer, and
/// this file does not guess. The LAST operand is always evaluated and
/// returned once every earlier operand has been skipped (an `and` chain
/// of all-truthy operands, or an `or` chain of all-falsy ones), matching
/// the same short-circuit rule — a BoolOp always carries at least two
/// values (Python's grammar has no one-operand `and`/`or`), so the loop
/// below always reaches that last operand.
pub(super) fn evaluate_boolop(boolop: &ruff_python_ast::ExprBoolOp, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    let is_and = boolop.op == BoolOp::And;
    let last_index = boolop.values.len().saturating_sub(1);
    for (index, operand_expr) in boolop.values.iter().enumerate() {
        let operand = evaluate_expression(operand_expr, environment, kernel);
        if index == last_index {
            return operand;
        }
        let (value, known) = truthiness(&operand);
        if !known {
            // This operand's truthiness decides whether the expression
            // stops HERE (answering this operand) or carries on, and it
            // is undecided — but every outcome is one of the operands
            // this expression can still return, so their JOIN is a true
            // claim about the answer. That is the whole content of the
            // clause: expressions.rst, "Boolean operations" — "neither
            // ``and`` nor ``or`` restrict the value and type they return
            // to ``False`` and ``True``, but rather return the last
            // evaluated argument."
            //
            // For two `bool` operands the join IS the two-member boolean
            // domain, so `int(a and b)` reads `{0, 1}` rather than
            // widening to `int_image`'s unbounded ray; for operands of
            // any other sorts the join states exactly as much as those
            // operands jointly admit, never a boolean the expression
            // does not produce.
            return join_remaining_operands(&boolop.values[index..], environment, kernel);
        }
        // `and` stops on a falsy operand; `or` stops on a truthy one —
        // that operand is the short-circuited answer
        if is_and == !value {
            return operand;
        }
    }
    unknown()
}

/// The join of every operand a short-circuit expression can still
/// return, once an operand's truthiness has gone undecided: from that
/// operand onward, any of them may be the last evaluated argument, and
/// the expression's answer is whichever one that turns out to be. An
/// operand this file cannot read at all makes the whole join unknown —
/// a join with an unread arm states nothing.
fn join_remaining_operands(
    operands: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let mut joined: Option<AbstractValue> = None;
    for operand_expr in operands {
        let operand = evaluate_expression(operand_expr, environment, kernel);
        if operand.kind == Kind::Unknown {
            return unknown();
        }
        joined = Some(match joined {
            None => operand,
            Some(current) => join_known(current, operand),
        });
    }
    joined.unwrap_or_else(unknown)
}
