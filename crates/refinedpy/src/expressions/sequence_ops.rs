
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::nan_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;
use ruff_text_size::TextRange;

use crate::assignability;
use crate::collection_models;
use crate::env::Environment;

use super::arithmetic::*;
use super::attribute::*;
use super::compare::*;

/// The binary-operator spelling of a set method. The ordinary row needs
/// both operands known `Kind::List` (this domain's shared list/set
/// shape) — a numeric or string operand pair never reaches here
/// (`binary_arithmetic_value` and the `Add`/`Mult` rows above already own
/// those) — so it routes `|`/`&`/`-`/`^` through the exact same
/// `set_method_result` logic a `.union(...)`/`.intersection(...)`/
/// `.difference(...)`/`.symmetric_difference(...)` method call already
/// answers. The EMPTY-OPERAND row is tried first and answers a pair the
/// item-by-item logic cannot read — see `set_operator_with_empty`'s own
/// doc.
pub(super) fn set_operator_value(method: &str, left: &AbstractValue, right: &AbstractValue) -> AbstractValue {
    if let Some(value) = set_operator_with_empty(method, left, right) {
        return value;
    }
    if left.kind != Kind::List || right.kind != Kind::List {
        return unknown();
    }
    match set_method_result(method, left, std::slice::from_ref(right)) {
        Some(value) => value,
        None => unknown(),
    }
}

/// A set operator one of whose operands is the EMPTY set, where the
/// other is an unread `set[X]` parameter's own repetition window
/// (`Kind::Set`, `as_repetition` — `check::seed_parameters`' sequence
/// seed, the same shape `star_element_read`/`len_result` already read).
/// The exact member list of the windowed side is unknown, so
/// `set_method_result`'s own item-by-item loops cannot run — but each of
/// stdtypes.rst's four set-operation rows states an IDENTITY against the
/// empty set that holds whatever the other side's members are:
///
/// - `union` (`set | other`, "Return a new set with elements from the
///   set and all others") and `symmetric_difference` (`set ^ other`,
///   "elements in either the set or *other* but not both") both answer
///   the OTHER side unchanged — no member is added by an empty operand,
///   and none of the windowed side's members is in the empty side to
///   cancel.
/// - `difference` (`set - other`, "elements in the set that are not in
///   the others") answers the LEFT unchanged when the RIGHT is empty (no
///   member is removed), and the empty set when the LEFT is empty.
/// - `intersection` (`set & other`, "elements common to the set and all
///   others") answers the EMPTY set either way — nothing is common with
///   a set that holds nothing.
///
/// This is what A8.xfer.setalgebra's own `union_widened_count` needs:
/// `len(a | set())` must still read `a`'s own guarded size of 151,
/// outside Age's `[0, 150]`, rather than the union deriving nothing at
/// all. `None` when neither side is a known EMPTY `Kind::List` or the
/// other side is not a repetition window, so the ordinary exact rows
/// below stand unchanged.
fn set_operator_with_empty(method: &str, left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let is_empty_set = |value: &AbstractValue| value.kind == Kind::List && value.items.is_empty();
    let is_window = |value: &AbstractValue| {
        value.kind == Kind::Set && value.set_kind_tag == SetKindTag::None && refined_sets::repetition_window_forms::as_repetition(&value.set).is_some()
    };
    let empty = || collection_models::list_literal_value(&[]);
    match method {
        "union" | "symmetric_difference" => {
            if is_empty_set(right) && is_window(left) {
                return Some(left.clone());
            }
            if is_empty_set(left) && is_window(right) {
                return Some(right.clone());
            }
            None
        }
        "difference" => {
            if is_empty_set(right) && is_window(left) {
                return Some(left.clone());
            }
            if is_empty_set(left) && is_window(right) {
                return Some(empty());
            }
            None
        }
        "intersection" => {
            if (is_empty_set(right) && is_window(left)) || (is_empty_set(left) && is_window(right)) {
                return Some(empty());
            }
            None
        }
        _ => None,
    }
}

/// `d1 | d2` — stdtypes.rst's Mapping Types section, `d | other`:
/// "Create a new dictionary with the merged keys and values of *d* and
/// *other*, which must both be dictionaries. The values of *other* take
/// priority when *d* and *other* share keys." Both operands must be a
/// known `Kind::Object` dict (this domain's dict shape,
/// `collection_models`'s own module doc) with no `kind_word` — a tagged
/// opaque object (a `dict.fromkeys` carrier, a datetime instance) is not
/// a dict and declines. The merge is entry by entry in the same order
/// `dict_mutated_receiver`'s own `update` row already walks: the left's
/// entries first, then each of the right's entries either OVERWRITING a
/// matching `(name, numeric)` slot or appending a new one — which is
/// exactly the priority rule the cited clause states, and preserves
/// insertion order for the keys only the left holds.
///
/// Neither operand is mutated: `d | other` builds a NEW dict
/// (stdtypes.rst states the in-place spelling separately as `d |= other`),
/// so this row clones the left's entries rather than writing through the
/// receiver — which is what lets A8.xfer.merge's own
/// `originals_unchanged_after_merge` still read `a["a"]` as the
/// pre-merge value afterward.
///
/// `None` when either side is not a plain known dict, so the caller's
/// own set-operator row and final decline stand unchanged.
pub(super) fn dict_union_value(left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let is_plain_dict = |value: &AbstractValue| value.kind == Kind::Object && value.kind_word.is_none();
    if !is_plain_dict(left) || !is_plain_dict(right) {
        return None;
    }
    let mut entries = left.keys.clone();
    for incoming in &right.keys {
        if let Some(existing) = entries
            .iter_mut()
            .find(|entry| entry.name == incoming.name && entry.numeric == incoming.numeric)
        {
            existing.value = incoming.value.clone();
        } else {
            entries.push(incoming.clone());
        }
    }
    let grade = refined_domain::trust_grades::derived_trust_level(TrustProved, &[left.clone(), right.clone()]);
    Some(refined_domain::known_constructors::known_object(entries, None, true, grade, false))
}

/// `sequence * n` (one fixed operand order): `sequence` is a known exact
/// string or a known list, `n` is a single known Integer-sorted value.
/// A negative `n` answers the empty sequence (stdtypes.rst note 2); a
/// non-negative `n` repeats the sequence's own elements/code points that
/// many times. `None` when `sequence`/`n` are not this exact shape, so
/// the caller can try the other operand order.
pub(super) fn sequence_repetition(sequence: &AbstractValue, count: &AbstractValue) -> Option<AbstractValue> {
    let (count_value, count_sort) = single_numeric_value(count)?;
    if count_sort != PrimitiveKind::Integer {
        return None;
    }
    let repeats = if count_value < 0.0 { 0 } else { count_value as usize };
    if let Some(text) = exact_string_values(sequence) {
        let mut repeated = Vec::with_capacity(text.len() * repeats);
        for _ in 0..repeats {
            repeated.extend_from_slice(text);
        }
        return Some(known_values(repeated, PrimitiveKind::String, TrustProved));
    }
    if sequence.kind == Kind::List {
        let mut repeated = Vec::with_capacity(sequence.items.len() * repeats);
        for _ in 0..repeats {
            repeated.extend(sequence.items.iter().cloned());
        }
        return Some(collection_models::list_literal_value(&repeated));
    }
    None
}

/// `sequence * count` where `sequence` is provably STRING-SORTED (an
/// exact string, or a Set `assignability::states_sequence`/
/// `sequence_shaped` reads as a sequence form) but `sequence_
/// repetition`'s own exact row already declined — either `sequence`
/// carries no exact code points to repeat, or `count` is not one known
/// Integer value (a bare, unrefined `n: int` parameter, `Kind::Set`
/// over the unbounded integer ray). Every real `str * int` call
/// (stdtypes.rst, "Common Sequence Operations," note 2 — a negative `n`
/// answers the empty string, never anything else) answers ANOTHER
/// `str`, so `strings()` (`Σ*`) is sound here regardless of which side
/// is unread: this is the same "answer the sort, not a guessed value"
/// row `string_models::string_method_sort_only_result` keeps for a
/// method call over an unbounded receiver, applied to the `*` operator
/// instead of a method name. `count` must still be provably
/// Integer-sorted (`Kind::Values`/`Kind::Set` tagged
/// `PrimitiveKind::Integer`) — a Float or unread count is `str`'s own
/// `TypeError`, not this row's to answer. `None` when `sequence` is not
/// string-shaped at all (a list repetition, or a genuinely unknown
/// operand) — the caller's own final `unknown()` decline, unchanged.
pub(super) fn string_repetition_sort_only(sequence: &AbstractValue, count: &AbstractValue) -> Option<AbstractValue> {
    let sequence_is_string_shaped = exact_string_values(sequence).is_some()
        || (sequence.kind == Kind::Set
            && (crate::assignability::states_sequence(&sequence.set) || crate::assignability::sequence_shaped(&sequence.set)));
    if !sequence_is_string_shaped {
        return None;
    }
    let count_is_integer_sorted = match count.kind {
        Kind::Values => count.kind_tag == Some(PrimitiveKind::Integer),
        Kind::Set => count.kind_tag == Some(PrimitiveKind::Integer),
        _ => false,
    };
    if !count_is_integer_sorted {
        return None;
    }
    Some(known_set(strings(), None, trust_level_of(sequence), SetKindTag::None))
}

/// `xs + ys` where BOTH sides are unknown-length sequences — a
/// `Kind::Set` whose only form is a repetition window (`as_repetition`,
/// the shape a declared `list[X]`/`Sequence[X]` parameter seeds). The
/// exact-list row already answered when both sides carry known items,
/// so this is what fires the moment either one does not.
///
/// stdtypes.rst, "Common Sequence Operations," `s + t`: "the
/// concatenation of *s* and *t*." Concatenation pins both facts a
/// repetition window states. Every element of the result is an element
/// of one side or the other, so the result's element set is the UNION
/// of the two windows' elements — the loosest sound claim, and the
/// exact one, since concatenation introduces no value neither side
/// held. And the length is the SUM of the two lengths, so the result's
/// window is `[lo_a + lo_b, hi_a + hi_b]`, unbounded above the moment
/// either side is.
///
/// This is what lets `combined = a + b` then read `combined[0]` as the
/// two element windows' union — A7.xfer.concat's own
/// `concat_element_outside` claim, `Age`'s `[0, 150]` joined with
/// `Wide`'s `[0, 200]` — rather than the binding deriving nothing.
///
/// STRING windows are NOT this row: a `str` receiver's own window is a
/// repetition of CODEPOINTS, and `string_set_concatenation` below states
/// a strictly narrower claim for that pair — the exact
/// `Concatenation(a, b)` grammar, which preserves the ORDER of the two
/// sides, where this row's union-of-elements would discard it. So a
/// STRING-SHAPED side (`assignability::sequence_shaped` — the same
/// recognizer the sort laws gate on, which reads a repetition as
/// string-shaped exactly when its element demonstrably IS codepoints)
/// declines here and reaches the string row unchanged.
///
/// `None` when either side is not a repetition window, so the caller's
/// own exact rows and final decline stand unchanged.
pub(super) fn sequence_window_concatenation(left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let is_window = |value: &AbstractValue| value.kind == Kind::Set && value.set_kind_tag == SetKindTag::None;
    if !is_window(left) || !is_window(right) {
        return None;
    }
    // A NUMERIC-SORTED window is this row's own pair, whatever its
    // element set happens to look like. `sequence_shaped` reads a set
    // alone, and a numeric element window that lies inside the codepoint
    // range is indistinguishable from a codepoint alphabet by set shape —
    // `Age`'s own `[0, 150]` integers all ARE code points, so
    // `within_codepoint_door` answers true for a `list[Age]` element and
    // the string-shaped decline below would take a pair that is not a
    // string pair at all. The VALUE carries what the set cannot: a
    // `list[X]` seed's own `kind_tag` is X's numeric sort
    // (`check::seed::seed_parameters`' sequence arm), which a string
    // window never has. So both sides being numeric-sorted settles the
    // question ahead of the set-shape reading.
    let both_numeric_sorted = matches!(left.kind_tag, Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float))
        && matches!(right.kind_tag, Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float));
    if !both_numeric_sorted
        && (crate::assignability::sequence_shaped(&left.set) || crate::assignability::sequence_shaped(&right.set))
    {
        return None;
    }
    let left_window = refined_sets::repetition_window_forms::as_repetition(&left.set)?;
    let right_window = refined_sets::repetition_window_forms::as_repetition(&right.set)?;
    let element = make_refined_set(vec![refined_sets::refinement_forms::union(
        left_window.element,
        right_window.element,
    )]);
    let high = match (left_window.hi, right_window.hi) {
        (Some(left_high), Some(right_high)) => Some(left_high + right_high),
        _ => None,
    };
    let joined = refined_sets::repetition_window_forms::repetition(element, left_window.lo + right_window.lo, high);
    let grade = refined_domain::trust_grades::derived_trust_level(TrustProved, &[left.clone(), right.clone()]);
    // the two sides' element sorts agree wherever both state one; a
    // disagreement leaves the result untagged rather than picking a side
    let kind_tag = if left.kind_tag == right.kind_tag { left.kind_tag } else { None };
    Some(AbstractValue {
        kind_tag,
        ..known_set(joined, None, grade, SetKindTag::None)
    })
}

/// `xs * n` where `xs` is a known `Kind::List` but `n` is NOT one known
/// Integer value (a bare `n: int` parameter, `Kind::Set` over the
/// integer ray) — `sequence_repetition`'s own exact row already
/// declined, so the exact item sequence cannot be built. stdtypes.rst,
/// "Common Sequence Operations," `s * n`: "equivalent to adding *s* to
/// itself *n* times," with note (2): "Values of *n* less than ``0`` are
/// treated as ``0`` (which yields an empty sequence of the same type as
/// *s*). Note that items in the sequence *s* are not copied; they are
/// referenced multiple times."
///
/// Two facts survive an unread `n`. Every element of the result is an
/// element of `xs` — repetition only ever re-references `xs`'s own
/// items, never builds a value outside them. And the LENGTH is
/// `len(xs) * n`, unbounded above with `n` unread and floored at `0` by
/// note (2)'s own negative-`n` clause. So the answer is the repetition
/// window over the JOIN of `xs`'s items, length `[0, +inf)` — which is
/// what lets `lst = [0] * n` then read `lst[0]` as exactly `{0}` (a
/// one-item list joins to that single value) rather than the whole
/// binding deriving nothing.
///
/// `n` must still be provably Integer-sorted: `list * float` is
/// CPython's own `TypeError`, not this row's to answer. `None` when
/// `xs` is not a known list, or when any item carries no readable set
/// (`set_of_known` declining) — the caller's own `unknown()` stands.
pub(super) fn list_repetition_sort_only(sequence: &AbstractValue, count: &AbstractValue) -> Option<AbstractValue> {
    if sequence.kind != Kind::List || sequence.items.is_empty() {
        return None;
    }
    let count_is_integer_sorted = matches!(count.kind, Kind::Values | Kind::Set) && count.kind_tag == Some(PrimitiveKind::Integer);
    if !count_is_integer_sorted {
        return None;
    }
    let mut element: Option<AbstractValue> = None;
    for item in &sequence.items {
        element = Some(match element {
            None => item.clone(),
            Some(so_far) => refined_domain::lattice_operations::join_known(so_far, item.clone()),
        });
    }
    let element = element.expect("sequence.items is non-empty, checked above");
    let element_set = refined_domain::lattice_operations::set_of_known(&element)?;
    let repeated = refined_sets::repetition_window_forms::repetition(element_set, 0, None);
    let grade = refined_domain::trust_grades::derived_trust_level(TrustProved, &[sequence.clone(), count.clone()]);
    Some(AbstractValue {
        kind_tag: element.kind_tag,
        ..known_set(repeated, None, grade, SetKindTag::None)
    })
}

/// `left + right` where at least one side is a STRING-SHAPED SET rather
/// than an exact literal — `seed + "xxxxxxxx"` where `seed` is a
/// parameter carrying a length window (`Annotated[str, Field(min_length=…,
/// max_length=…)]` seeds `Kind::Set` over a `Repeat`/`Star` form,
/// `check.rs::seed_parameters`'s own doc), never `Kind::Values`. The
/// exact-exact row above already answers when BOTH operands are literal
/// strings; this row is what fires the moment either one is not. Composes
/// the same `refinement_forms::concatenation` form `string_tuple`
/// concatenation and the f-string pattern tier already build — pure set
/// composition, no kernel round trip, since concatenation is a GRAMMAR
/// constructor, not a decided question. `None` when either side is not
/// string-shaped at all (a numeric set, an object, an unknown), so the
/// caller's own `unknown()` fallback stays honest for it.
pub(super) fn string_set_concatenation(left: &AbstractValue, right: &AbstractValue) -> Option<AbstractValue> {
    let left_set = string_shaped_set(left)?;
    let right_set = string_shaped_set(right)?;
    let joined = make_refined_set(vec![refined_sets::refinement_forms::concatenation(left_set, right_set)]);
    let grade = refined_domain::trust_grades::derived_trust_level(TrustProved, &[left.clone(), right.clone()]);
    Some(known_set(joined, None, grade, SetKindTag::None))
}

/// The `RefinedSet` a value states, read ONLY when the value is
/// string-shaped: a known exact string (`Kind::Values` tagged
/// `PrimitiveKind::String` — read through `set_of_known`, which answers
/// the code points' own concatenation form for a multi-character
/// literal), or an untagged `Kind::Set` whose own forms demonstrably
/// carry a sequence shape (`assignability::states_sequence` — the same
/// gate `scalar_case_of`/`seed_parameters` use to tell a string window
/// from a numeric range, since a bare `Kind::Set` carries no sort tag of
/// its own to read instead). A numeric set, an object, or any other
/// shape answers `None` — never guessed at.
pub(super) fn string_shaped_set(value: &AbstractValue) -> Option<RefinedSet> {
    if value.kind == Kind::Values && value.kind_tag == Some(PrimitiveKind::String) {
        return refined_domain::lattice_operations::set_of_known(value);
    }
    if value.kind == Kind::Set && value.set_kind_tag == SetKindTag::None && assignability::states_sequence(&value.set) {
        return Some(value.set.clone());
    }
    None
}

/// Wraps an arithmetic result as known_values, honestly: an int result
/// stays exact only while it still fits an f64's 53-bit exact-integer
/// range (2^53) — CPython ints are unbounded, but this file's carrier is
/// f64, so a result outside that range is no longer provably exact and
/// declines rather than silently truncating. `both_int` selects the
/// Python sort: `Integer` when both operands were int-sorted (and the
/// value stays exact), `Float` otherwise — the mixed-arithmetic widening
/// rule (stdtypes' Numeric Types) and `/`'s own always-float override
/// both route through this by passing `both_int = false`.
///
/// A Python `float` arithmetic result CAN be NaN (`inf - inf`, `inf *
/// 0.0` — IEEE 754, arith.9's own doc), so the Float row answers
/// `nan_value()` — the domain's own NaN state — rather than let a bare
/// NaN enter `known_values`, which no refined set admits
/// (`refinement_forms::element`'s own construction-time refusal). The
/// int row cannot reach this: an int-sorted pair's NaN check already
/// takes the `value.fract() != 0.0` decline above (`NaN.fract()` is
/// itself NaN, which is `!= 0.0`), so no int-sorted NaN ever reaches
/// `known_values` here.
/// Exact integer arithmetic past the 2^53 window: two int-sorted f64
/// operands that each carry a whole number convert exactly to i128,
/// compute with checked operations (`+ - * **` with a non-negative int
/// exponent), and the result folds only when an f64 CARRIES it exactly
/// — the round-trip test — so 2^62 + 2^62 folds to 2^63 while
/// 2^62 + 1 declines instead of silently rounding. Anything else
/// (overflow, a negative or oversized exponent, a fractional operand)
/// answers None and the caller's ordinary rows decide.
pub(super) fn exact_int_arithmetic(op: Operator, left: f64, right: f64) -> Option<f64> {
    if left.fract() != 0.0 || right.fract() != 0.0 {
        return None;
    }
    if left.abs() >= 2f64.powi(126) || right.abs() >= 2f64.powi(126) {
        return None;
    }
    let a = left as i128;
    let b = right as i128;
    let v = match op {
        Operator::Add => a.checked_add(b)?,
        Operator::Sub => a.checked_sub(b)?,
        Operator::Mult => a.checked_mul(b)?,
        Operator::Pow => {
            if !(0..=1024).contains(&b) {
                return None;
            }
            let mut acc: i128 = 1;
            for _ in 0..b {
                acc = acc.checked_mul(a)?;
            }
            acc
        }
        _ => return None,
    };
    let f = v as f64;
    if f.is_infinite() || f as i128 != v {
        return None;
    }
    Some(f)
}

pub(super) fn arithmetic_result(value: f64, both_int: bool) -> AbstractValue {
    if both_int {
        if value.fract() != 0.0 || value.abs() >= 2f64.powi(53) {
            return unknown();
        }
        return known_values(vec![value], PrimitiveKind::Integer, TrustProved);
    }
    if value.is_nan() {
        return nan_value();
    }
    known_values(vec![value], PrimitiveKind::Float, TrustProved)
}

/// An f64 carrying a whole number, as an `i64` — but only inside the
/// same 2^53 exact-integer window `arithmetic_result` already trusts
/// for every other operator. `&`/`|`/`^` need integer bit patterns
/// rather than f64 arithmetic, so this is the one extra conversion
/// step those three rows take; anything outside the window declines
/// the same way an out-of-range `+`/`-`/`*` result already does.
pub(super) fn f64_to_exact_i64(value: f64) -> Option<i64> {
    if value.fract() != 0.0 || value.abs() >= 2f64.powi(53) {
        return None;
    }
    Some(value as i64)
}

/// Whether `expression` (or a sub-expression it evaluates FIRST, in
/// CPython's own left-before-right, receiver-before-arguments order)
/// PROVABLY raises on every run, given every operand it needs is known.
/// `Some((range, message))` names the raising expression's own range and
/// a plain sentence in one voice: "this expression provably raises
/// <ExcType>: <plain detail>" — the same voice `bytes_models.rs`'s own
/// `BytesAnswer::Raises` messages already speak (this function is the
/// caller `check.rs` will route those messages through, unchanged).
/// Anything not provably raising, or any operand this file cannot read,
/// answers `None` — this function never guesses at a raise the way it
/// never guesses at a value.
///
/// Every row here means every run raises, full stop — `check.rs::
/// sink_value`'s own all-or-nothing gate (a fire here skips the value
/// question entirely). A SOMETIMES-raises escape (some admitted operand
/// values raise, the rest still produce a value) is a DIFFERENT claim
/// and lives in its own function, `possible_raise` below — never a row
/// here.
///
/// Recognized rows, each cited in the function that decides it: zero-
/// divisor arithmetic (`/`, `//`, `%`), an out-of-range/absent
/// subscript on a known List/Object, a bytes-like read/write whose
/// `bytes_models` answer is `BytesAnswer::Raises`, `int(<unparseable
/// known string>)`, `<receiver>.index(<absent known needle>)` on a
/// known string or list receiver, `math.sqrt(<known negative>)`, and
/// `math.floor`/`ceil`/`trunc` of a known non-finite argument
/// (`OverflowError` for an infinity, `ValueError` for NaN — each
/// returns an `Integral`, and no Python `int` holds either).
pub fn provable_raise(
    expression: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    match expression {
        Expr::BinOp(binop) => {
            if let Some(found) = provable_raise(&binop.left, environment, kernel) {
                return Some(found);
            }
            if let Some(found) = provable_raise(&binop.right, environment, kernel) {
                return Some(found);
            }
            binop_provable_raise(binop, environment, kernel)
        }
        Expr::Subscript(subscript) => {
            if let Some(found) = provable_raise(&subscript.value, environment, kernel) {
                return Some(found);
            }
            if let Some(found) = provable_raise(&subscript.slice, environment, kernel) {
                return Some(found);
            }
            subscript_provable_raise(subscript, environment, kernel)
        }
        Expr::Call(call) => {
            if let Some(found) = provable_raise(&call.func, environment, kernel) {
                return Some(found);
            }
            for arg in &call.arguments.args {
                if let Some(found) = provable_raise(arg, environment, kernel) {
                    return Some(found);
                }
            }
            call_provable_raise(call, environment, kernel)
        }
        Expr::Attribute(attribute) => provable_raise(&attribute.value, environment, kernel),
        Expr::UnaryOp(unary) => provable_raise(&unary.operand, environment, kernel),
        Expr::BoolOp(boolop) => boolop
            .values
            .iter()
            .find_map(|operand| provable_raise(operand, environment, kernel)),
        Expr::Compare(compare) => {
            if let Some(found) = provable_raise(&compare.left, environment, kernel) {
                return Some(found);
            }
            compare.comparators.iter().find_map(|comparator| provable_raise(comparator, environment, kernel))
        }
        _ => None,
    }
}
