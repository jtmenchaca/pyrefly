use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustLevel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::RefinedSet;

use super::kernel_transfer::exact_nonnegative_integer;

/// `x ** k` over a SET-shaped `left_set` (a seeded parameter range, or a
/// bounded set another transfer already produced) and a KNOWN exact
/// integer exponent — `transfer_over_sets`'s own `Pow` row, once neither
/// `binary_arithmetic_value`'s two-known-values path nor
/// `int_transfer_over_sets`'s `int.pow` arm has already answered.
/// `int.pow` (`boundary/python.lean`) matches `exactIntOf A` on the
/// base only — a closed singleton, never a window — so an int-sorted
/// range base (e.g. `x: Age` in `x**2`) always reads back `.unknown`
/// from it and reaches here needing a different row entirely.
///
/// Poses the LANGUAGE-NEUTRAL `pow` wire question directly
/// (`TransferQuestionOp::Pow`, `PowOperandWire::Set` on both `base` and
/// `exp` — `transfer_questions.rs`'s own wire test is the citation for
/// this shape) rather than a `python.` or `js.`-namespaced row: pow.lean's
/// `transferPow` is ECMA-262-shaped in its NaN/unpinned branches, but its
/// two WINDOW-READING sub-deciders are pure real-number math with no
/// language dependence at all —
///
/// - `transferIntegerPow` (`theories/pow/binary64.lean`): an
///   INTEGER-sorted base window (`A.int`) under an exact nonnegative
///   integer exponent — repeated multiplication, exact while the
///   corners fit `2^53`. This is the exact `int.pow` row Python's own
///   boundary elects, generalized from a singleton base to a window
///   one — the same "ask the shared theory when the language-owned arm
///   has no window form of its own" composition
///   `float_mul_as_shift_fallback` already performs for `<<`.
/// - `transferRealPow` (`languages/javascript/powers_and_roots/
///   approx_window.lean`): a NONNEGATIVE real (Float) base window under
///   an exact integer exponent `1 ≤ k ≤ 64` — the corner powers bracket
///   the true value, widened by the k-ulp approximation envelope. This
///   is `m ** 2` (`Meters` in showcase.py's `bmi`): `m`'s own window is
///   `[0.5, 2.5]`, entirely nonnegative, so the real-base branch serves
///   it directly. A base window straddling or beneath zero is OUTSIDE
///   what `transferRealPow` reads (`0 ≤ dl.num` its own gate) — Python's
///   `**` on a negative float base and integer exponent is still real
///   (unlike a fractional exponent, which goes complex), but proving that
///   corner sound is future work; this composition declines it rather
///   than guess.
///
/// The exponent is EITHER a known exact integer in `[1, 64]`
/// (`exact_nonnegative_integer`'s own f64-exact gate, further narrowed
/// to the `1..=64` window both Lean deciders share — read as the
/// one-element set `{k}`, the same singleton reading
/// `transferable_numeric_operand` gives any known scalar), OR an
/// Integer-sorted `Kind::Set` WINDOW (a seeded parameter range, or a
/// bounded set another transfer produced) — read as its own set,
/// mirroring exactly how `base_set` below is built off `left_set`/
/// `left_sort`. Either shape rides the wire as `PowOperandWire::Set`;
/// the kernel's own `transferIntegerPow` (`theories/pow/binary64.lean`)
/// is the decider for a windowed exponent (an integer-marked bounded
/// enclosure, nonnegative bounds) exactly as it already is for a
/// windowed BASE — a negative or unbounded exponent window, or any
/// shape the Lean decider does not read, comes back `Unknown`/`NaN`
/// and this function declines the same way it already declines an
/// unread base window. `x ** 0` is its own pinned branch below
/// (answered directly, no kernel round trip — see that branch's own
/// doc for why) and applies ONLY to the known-scalar exponent path,
/// since a window can never be the single value `0`.
/// A negative exponent is `binary_arithmetic_pair`'s own row when both
/// operands are single values, and stays `unknown()` here for a SET
/// base — outside this function's declared scope.
pub(in crate::expressions) fn pow_over_sets(
    left_set: &RefinedSet,
    left_sort: PrimitiveKind,
    right: &AbstractValue,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    use refined_kernel::transfer_questions::PowOperandKind;
    use refined_kernel::transfer_questions::PowOperandWire;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    use refined_kernel::transfer_questions::TransferQuestion;
    use refined_kernel::transfer_questions::TransferQuestionOp;

    let exp_set = if let Some(exponent) = exact_nonnegative_integer(right) {
        // `x ** 0` is exactly `1` for EVERY `x` — expressions.rst's power
        // operator row states no exception (unlike `0 ** negative`, which
        // raises `ZeroDivisionError`, or a negative base under a fractional
        // exponent, which goes complex): this is a closed pinned fact, not a
        // window computation, so it answers directly rather than asking
        // either Lean decider — neither `transferIntegerPow`
        // (`A.int` only) nor `transferRealPow` (`1 ≤ k ≤ 64`, nonnegative
        // base only) reads `k = 0` at all, and the shared classifier's own
        // `k = 0` cell (`powCells`'s `e.may zero → addVal vs 1`) sits behind
        // ECMA-shaped NaN/unpinned guards (`powNaN`'s own finite-negative-
        // base branch) that do not hold for Python's own `**` — Python
        // raises rather than answers NaN on the corner ECMA's classifier
        // reads as NaN, so trusting that branch here would risk claiming a
        // value on an input Python actually raises on. Answering `1`
        // directly for `k = 0` sidesteps both: no base reading is needed at
        // all for this one exponent value. Result sort matches the base's
        // own sort exactly as the two-known-values `binary_arithmetic_pair`
        // row already states for this same corner.
        if exponent == 0.0 {
            return Some(known_values(vec![1.0], left_sort, grade));
        }
        if exponent < 1.0 || exponent > 64.0 {
            return None;
        }
        make_refined_set(vec![one_of(&[exponent])])
    } else if right.kind == Kind::Set && right.kind_tag == Some(PrimitiveKind::Integer) {
        // A WINDOWED exponent — no exact scalar to read, so no `k = 0`/
        // `[1, 64]` pinned corner applies; the kernel's own
        // `transferIntegerPow` window arm decides the whole question
        // (including refusing an out-of-range or negative window), the
        // same "ask the shared theory" posture the base side already
        // takes for a window it cannot read locally.
        let mut forms = right.set.forms.clone();
        if !requires_integer(&right.set) {
            forms.push(integer());
        }
        make_refined_set(forms)
    } else {
        return None;
    };
    let base_forms = if left_sort == PrimitiveKind::Integer {
        let mut forms = left_set.forms.clone();
        if !requires_integer(left_set) {
            forms.push(integer());
        }
        forms
    } else {
        left_set.forms.clone()
    };
    let base_set = make_refined_set(base_forms);

    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Pow,
            a: make_refined_set(vec![]),
            b: make_refined_set(vec![]),
            c: 0.0,
            base: PowOperandWire { kind: PowOperandKind::Set, set: base_set },
            exp: PowOperandWire { kind: PowOperandKind::Set, set: exp_set },
        })
    })
    .ok()?;

    match asked.kind {
        TransferAnswerKind::Values => {
            if left_sort == PrimitiveKind::Integer
                && asked.values.iter().any(|v| v.fract() != 0.0 || v.abs() >= 2f64.powi(53))
            {
                return None;
            }
            Some(known_values(asked.values, left_sort, grade))
        }
        TransferAnswerKind::Set => {
            if left_sort == PrimitiveKind::Integer && !requires_integer(&asked.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(left_sort),
                ..known_set(asked.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}
