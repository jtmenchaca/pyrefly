use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustLevel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Operator;

use super::known_values::single_numeric_value;

/// The set operand a kernel arithmetic transfer can pose: a numeric-
/// sorted `Kind::Set` (`kind_tag` Integer or Float — a seeded
/// parameter's declared range, or a sort-only answer like
/// `float_sorted_unknown()`) reads as its own set, and a known single
/// numeric `Kind::Values` (`single_numeric_value`'s own shape) reads as
/// the one-element set `{v}` so a set-vs-known-value pair poses the
/// same two-set question a set-vs-set pair does. Returns the set
/// together with the PYTHON ARITHMETIC SORT it carries — the same
/// Integer/Float split `single_numeric_value` returns, `Boolean`/bare
/// `Number` normalized to `Integer`/`Float` the same conservative way
/// (AGENT-BRIEF.md's "unproven int reads as the float row"). `None` for
/// every other shape (String/Array-sorted, untagged Set, non-numeric
/// Values, and a SEQUENCE WINDOW whatever element sort it carries — the
/// body's own gate) — this is a decline, not a guess.
pub(in crate::expressions) fn transferable_numeric_operand(value: &AbstractValue) -> Option<(RefinedSet, PrimitiveKind)> {
    if let Some((v, sort)) = single_numeric_value(value) {
        return Some((make_refined_set(vec![refined_sets::refinement_forms::one_of(&[v])]), sort));
    }
    if value.kind == Kind::Set {
        // A SEQUENCE WINDOW is not a numeric operand, whatever element
        // sort it carries. A `list[X]` parameter seeds `Kind::Set` over a
        // repetition of X's own set, tagged with X's numeric sort —
        // `check::seed::seed_parameters`' sequence arm puts the ELEMENT's
        // sort on the OUTER sequence value so `sum`/`min`/`max` over the
        // sequence can read it. That tag makes a `list[int]` operand read
        // as Integer-sorted here, and without this gate `a + b` over two
        // such parameters poses a numeric `int.add` question about two
        // repetition sets — a number question asked of a list
        // concatenation, whose `Unknown` answer then binds the unbounded
        // integer ray in place of the concatenated sequence. Reading the
        // SET's own shape settles it: a repetition (`as_repetition`, the
        // one recognizer for the window grammar) is a sequence, so it
        // declines here and the caller's sequence row
        // (`sequence_binop_value` → `sequence_window_concatenation`)
        // answers the concatenation instead.
        if refined_sets::repetition_window_forms::as_repetition(&value.set).is_some() {
            return None;
        }
        let sort = match value.kind_tag {
            Some(PrimitiveKind::Integer) => PrimitiveKind::Integer,
            Some(PrimitiveKind::Float) => PrimitiveKind::Float,
            Some(PrimitiveKind::Boolean) => PrimitiveKind::Integer,
            Some(PrimitiveKind::Number) => PrimitiveKind::Float,
            // An untagged Set whose FORMS prove the integer sort — a
            // bare `int` parameter's seed carries the `integer` form
            // and no tag, and the form is the stronger claim anyway.
            None if value
                .set
                .forms
                .iter()
                .any(|form| form.form == refined_sets::refinement_forms::Form::Integer) =>
            {
                PrimitiveKind::Integer
            }
            _ => return None,
        };
        return Some((value.set.clone(), sort));
    }
    None
}

/// The kernel `TransferQuestionOp` a Python operator lowers to, or
/// `None` when the operator's kernel row is ECMA-semantics and
/// diverges for Python operands — the same exclusion
/// `loops.rs::lower_counter_step_body`'s own doc states for its
/// Add/Sub-only step shape, extended here to the one further operator
/// this file can also state safely:
///
/// - `Add`/`Sub`/`Mult` lower. The kernel's `transferAdd`/`transferSub`/
///   `transferMul` (set_functions/transfer.lean) are pure IEEE-754
///   float addition/subtraction/multiplication on the operands' real
///   enclosures — no ECMA `ToNumber` coercion, no string/object
///   handling folded in. Python's `+`/`-`/`*` over int/float operands
///   compute the identical IEEE-754 operation once both sides are read
///   as the f64s this file already carries them as (CPython floats ARE
///   IEEE-754 doubles, and `arithmetic_result` already declines an
///   integer result outside the f64-exact 2^53 range rather than claim
///   an inexact one) — so these three rows are semantics-identical
///   between the two languages and safe to lower.
/// - `Div` (`/`) lowers EXCEPT at the zero-divisor corner. Python's `/`
///   is always true division — arith.9 (python-pins.md): "Division of
///   int by int (`/`) yields a float — the type is widened even when
///   the arguments are exact integers" — and elects `binary64.div` for
///   exactly this reason, the SAME election the kernel's `Div` row
///   already carries. Away from a zero divisor the two `/`s name the
///   same theorem. AT a zero divisor they diverge: arith.10 makes
///   Python's `/` raise `ZeroDivisionError` (an exception, not a
///   value), while ECMA's `binary64.div` answers a DETERMINED
///   `±Infinity`/NaN (`theories/binary64/div.lean`'s `transferDiv`,
///   proved sound for that theorem — a correct answer to the WRONG
///   question for a Python operand). `transfer_over_sets` gates this:
///   it asks the kernel only when the divisor operand's set provably
///   EXCLUDES zero (`divisor_provably_excludes_zero`); when the
///   divisor's set may admit zero, the value question declines rather
///   than relabel ECMA's answer as Python's. `transfer_over_sets`'s own
///   `result_sort` computation carries the always-Float override for
///   this one op — the `both_int` rule the other three admitted ops
///   share does not apply here.
/// - `FloorDiv` (`//`), `Mod` (`%`), and `Pow` (`**`) do NOT lower to
///   the FLOAT family. `%` takes the DIVISOR's sign in Python, the
///   opposite of ECMA's dividend-sign remainder (AGENT-BRIEF.md,
///   expressions.rst §6.7) — asking the kernel's `Rem` row for a Python
///   `%` would silently answer the wrong sign on a mixed-sign pair; `//`
///   floors toward negative infinity, which is not one of the kernel's
///   float arithmetic transfer rules at all; `**` has no float
///   binary-arithmetic-transfer row in this family (`Pow` in
///   `TransferQuestionOp` is the pinned NaN/unknown/set `PowOperandWire`
///   shape, a different question shape from the plain two-`RefinedSet`
///   rows this function poses).
///
/// `admitted_int_transfer_op` below states the row those three DO have,
/// on the other side of the sort split: the exact `int` theory.
pub(in crate::expressions) fn admitted_transfer_op(op: Operator) -> Option<refined_kernel::transfer_questions::TransferQuestionOp> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    match op {
        Operator::Add => Some(TransferQuestionOp::Add),
        Operator::Sub => Some(TransferQuestionOp::Sub),
        Operator::Mult => Some(TransferQuestionOp::Mul),
        Operator::Div => Some(TransferQuestionOp::Div),
        _ => None,
    }
}

/// The kernel `TransferQuestionOp` a Python operator lowers to when BOTH
/// operands are INT-SORTED — the exact `int` theory
/// (`boundary/python.lean`'s `pythonTransferOfOp2`), never the
/// `binary64.*` float image `admitted_transfer_op` returns. Python's
/// integers have unlimited precision and never wrap (python-pins.md
/// arith.1), so every row here is exact arithmetic on the mathematical
/// integers, which is what the `int.*` theory proves:
///
/// - `Add`/`Sub`/`Mult` elect `int.add`/`int.sub`/`int.mul` — the exact
///   whole-number operations arith.1 names ("the float transfer is
///   REFUSED for ints and the exact whole-number theory (`int.*`) serves
///   them"). The float image would agree on any operand pair small
///   enough to be f64-exact, but the exact theory is the one the pins
///   elect, and it is the theory that stays right at the edges.
/// - `FloorDiv` elects `int.floorDiv` — arith.7/arith.8: floor division
///   "is always rounded towards minus infinity," paired with `%` by
///   `x == (x//y)*y + (x%y)`. A zero divisor is `ZeroDivisionError`
///   (arith.10), which the kernel arm refuses on rather than answering.
/// - `Mod` elects `rem.divisorSign` — arith.4: "the modulo operator
///   yields a result with the same sign as its SECOND operand (the
///   divisor)." This is the Python-owned remainder, a DIFFERENT theorem
///   from the `rem.truncDividendSign` row JavaScript's `%` elects, so
///   electing it by name is what makes the sign right on a mixed-sign
///   pair.
/// - `Pow` elects `int.pow` — pow.1: "`int ** nonnegative int` yields an
///   exact int (same type as the operands)... a negative int exponent
///   converts both arguments to float and yields a float." The kernel's
///   `int.pow` arm reads its exponent as a nonnegative `Nat`, so
///   `int_transfer_over_sets` below gates this row on
///   `exact_nonnegative_integer` before ever asking — a
///   possibly-negative exponent declines to the float path, which is
///   where pow.1 sends it anyway.
/// - `BitAnd`/`BitOr`/`BitXor` elect `int.bitAnd`/`int.bitOr`/
///   `int.bitXor` — bits.4/bits.5/bits.6: the bitwise operations on
///   UNBOUNDED ints, "never JS's 32-bit wrap view." The `int32.*` family
///   the JavaScript rows elect is the wrong theorem here for exactly
///   that reason.
///
/// `Div` is absent by design: arith.9 widens int/int to float, so `/`
/// never has an int-sorted row at all — it stays on
/// `admitted_transfer_op`'s `binary64.div`. `LShift`/`RShift` are also
/// absent: bits.1/bits.2 define them as `int.floorDiv`/`int.mul` by
/// `2**n` rather than as their own members, so they lower as that
/// COMPOSITION (`shift_as_int_composition` below), not as a direct op.
pub(in crate::expressions) fn admitted_int_transfer_op(op: Operator) -> Option<refined_kernel::transfer_questions::TransferQuestionOp> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    match op {
        Operator::Add => Some(TransferQuestionOp::IntAdd),
        Operator::Sub => Some(TransferQuestionOp::IntSub),
        Operator::Mult => Some(TransferQuestionOp::IntMul),
        Operator::FloorDiv => Some(TransferQuestionOp::IntFloorDiv),
        Operator::Mod => Some(TransferQuestionOp::RemDivisorSign),
        Operator::Pow => Some(TransferQuestionOp::IntPow),
        Operator::BitAnd => Some(TransferQuestionOp::IntBitAnd),
        Operator::BitOr => Some(TransferQuestionOp::IntBitOr),
        Operator::BitXor => Some(TransferQuestionOp::IntBitXor),
        _ => None,
    }
}

/// A known EXACT NONNEGATIVE INTEGER operand, as its own `f64` — the
/// shape two rows below need before they may ask an `int.*` question:
/// `Pow`'s exponent (pow.1's nonnegative-int branch is the only one the
/// exact `int.pow` theory covers; a negative exponent "converts both
/// arguments to float and yields a float," a different row) and a
/// shift's count (bits.3: "a negative shift count is illegal and raises
/// `ValueError`"). A SET operand answers `None` even when its whole
/// range is nonnegative — the kernel's own `int.*` arms read an operand
/// through `exactIntOf` (a closed singleton, `numeric/enclosure_read.lean`),
/// so a range exponent has nothing to offer them, and proving
/// nonnegativity of a range here would state a gate the row behind it
/// cannot use. The value must also sit inside the f64-exact 2^53 window
/// `arithmetic_result` already trusts, for the same reason it does.
pub(in crate::expressions) fn exact_nonnegative_integer(value: &AbstractValue) -> Option<f64> {
    let (number, sort) = single_numeric_value(value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    if number < 0.0 || number.fract() != 0.0 || number.abs() >= 2f64.powi(53) {
        return None;
    }
    Some(number)
}

/// Poses one `int.*` question and reads the answer back as an
/// INTEGER-SORTED value. Every `int.*` member answers exact whole
/// numbers (python-pins.md arith.1 — "integer `+ − ×` never overflows
/// and never wraps"), so the answer's sort is Integer unconditionally,
/// with no `both_int` computation of its own: reaching this function AT
/// ALL already required both operands to be int-sorted.
///
/// Two guards the float family does not need:
///
/// - A non-integral value in the answer declines. `int.*` cannot produce
///   one, so this can only mean the wire carried something this row
///   does not understand.
/// - A value outside the f64-exact 2^53 window declines. Python's
///   integers are unbounded (arith.1) and the kernel computes them
///   exactly as `Int`s, but `boundary/encode_sets.lean`'s `encodeNumber`
///   puts every result through `roundNE` before it crosses the wire — so
///   a result past 2^53 arrives ROUNDED, and claiming it as exact would
///   be claiming a value CPython never computes. This is the same window
///   and the same reason `arithmetic_result` already declines on.
///
/// A SET answer must additionally CARRY its own integrality
/// (`requires_integer`) before it is tagged Integer-sorted. Most `int.*`
/// arms answer `.vals`, so this is about the one arm that answers an
/// enclosure — `rem.divisorSign`, whose general-interval branch produces
/// a bound-shaped enclosure. Tagging a set Integer-sorted without that
/// mark would claim an integrality the kernel did not state.
pub(in crate::expressions) fn int_transfer_answer(
    transfer_op: refined_kernel::transfer_questions::TransferQuestionOp,
    left_set: RefinedSet,
    right_set: RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let nan_operand = refined_kernel::transfer_questions::PowOperandWire {
        kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
        set: make_refined_set(vec![]),
    };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&refined_kernel::transfer_questions::TransferQuestion {
            op: transfer_op,
            a: left_set,
            b: right_set,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    });
    let answer = asked.ok()?;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    match answer.kind {
        TransferAnswerKind::Values => {
            if answer
                .values
                .iter()
                .any(|v| v.fract() != 0.0 || v.abs() >= 2f64.powi(53))
            {
                return None;
            }
            Some(known_values(answer.values, PrimitiveKind::Integer, grade))
        }
        TransferAnswerKind::Set => {
            if !requires_integer(&answer.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(answer.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// The INT-SORTED half of `transfer_over_sets`: when both operands are
/// Integer-sorted, the exact `int` theory serves the operation, not the
/// `binary64.*` float image (python-pins.md arith.1 states this
/// directly — "the float transfer is REFUSED for ints and the exact
/// whole-number theory (`int.*`) serves them").
///
/// Ops and their rows are `admitted_int_transfer_op`'s own doc.
/// The two conditional rows this function gates before asking:
///
/// - `Pow`: the kernel's `int.pow` arm reads its exponent as a
///   nonnegative `Nat`, matching pow.1's own exact branch. An exponent
///   this file cannot prove is an exact nonnegative integer
///   (`exact_nonnegative_integer`) DECLINES here and falls through to
///   the float path, which is where pow.1 puts a negative exponent
///   anyway ("a negative int exponent converts both arguments to float
///   and yields a float").
/// - `LShift`/`RShift`: bits.1/bits.2 define these as compositions
///   rather than as their own kernel members, so they lower as that
///   composition — see `shift_as_int_composition`.
///
/// `Div` never reaches here: arith.9 widens int/int to float, so the
/// caller keeps `/` on the float path unconditionally.
pub(in crate::expressions) fn int_transfer_over_sets(
    op: Operator,
    right: &AbstractValue,
    left_set: &RefinedSet,
    right_set: &RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if matches!(op, Operator::LShift | Operator::RShift) {
        return shift_as_int_composition(op, left_set, right, grade, kernel);
    }
    if op == Operator::Pow {
        // pow.1's exact branch only — a possibly-negative exponent is
        // the float row, not this one
        exact_nonnegative_integer(right)?;
    }
    let transfer_op = admitted_int_transfer_op(op)?;
    int_transfer_answer(transfer_op, left_set.clone(), right_set.clone(), grade, kernel)
}

/// `x << n` / `x >> n` over int-sorted operands, lowered as the
/// COMPOSITION the pins define them to be rather than as kernel members
/// of their own:
///
/// - bits.2: "`x << n` equals multiplication of `x` by `2**n`" —
///   `int.mul` against the singleton `{2**n}`.
/// - bits.1: "`x >> n` equals floor division of `x` by `2**n`" —
///   `int.floorDiv` against the same singleton.
///
/// The shift count must be a KNOWN exact nonnegative integer
/// (`exact_nonnegative_integer`): bits.3 makes a negative count a
/// `ValueError` rather than a value, and a count this file cannot read
/// exactly gives no `2**n` to compose against. `2**n` itself must also
/// land inside the f64-exact 2^53 window, or the singleton this builds
/// would not be the number it names — the same window every other
/// exactness gate in this file keeps.
///
/// `int.mul` (`boundary/python.lean`) only ever matches `exactIntOf`
/// on BOTH sides — there is no general-window arm the way
/// `int.floorDiv` carries (`intFloorDivWindow`), so `x << n` over a
/// non-singleton `left_set` (the ordinary case: a seeded parameter
/// range, never one known value) always reads back `.unknown` from
/// `int.mul` and would decline outright with no further attempt. `x >>
/// n` never needs this fallback — `int.floorDiv`'s own window arm
/// already narrows a bounded integer dividend against the single-signed
/// non-zero singleton divisor `factor_set` states.
///
/// The fallback asks the FLOAT image instead (`TransferQuestionOp::Mul`,
/// `binary64.mul`'s enclosure decider, the same row `count * 2` already
/// rides when `int.mul` declines on it — `transfer_over_sets`'s own
/// `admitted_transfer_op` fallthrough). Multiplying an integer-sorted
/// window by an exact power-of-two singleton already inside the
/// f64-exact window (`factor`'s own gate above) never rounds — every
/// product the float enclosure narrows to is the same exact integer
/// `int.mul` would have named, had it read windows at all — so the
/// answer re-tags Integer-sorted unconditionally, the identical
/// `both_int` re-tagging `transfer_over_sets`'s general path performs
/// for `Add`/`Sub`/`Mult` today.
pub(in crate::expressions) fn shift_as_int_composition(
    op: Operator,
    left_set: &RefinedSet,
    right: &AbstractValue,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    let count = exact_nonnegative_integer(right)?;
    let factor = 2f64.powf(count);
    if factor.fract() != 0.0 || factor >= 2f64.powi(53) {
        return None;
    }
    let factor_set = make_refined_set(vec![one_of(&[factor])]);
    if op == Operator::RShift {
        return int_transfer_answer(TransferQuestionOp::IntFloorDiv, left_set.clone(), factor_set, grade, kernel);
    }
    if let Some(answer) = int_transfer_answer(TransferQuestionOp::IntMul, left_set.clone(), factor_set.clone(), grade, kernel) {
        return Some(answer);
    }
    float_mul_as_shift_fallback(left_set, &factor_set, grade, kernel)
}

/// The float-image `Mul` retry `shift_as_int_composition` takes for
/// `<<` alone, once `int.mul`'s exact-singleton-only arm has already
/// declined on a non-singleton `left_set`. Asks `binary64.mul`
/// (`TransferQuestionOp::Mul`) the same two sets and re-tags whatever
/// enclosure or exact values it narrows to as Integer-sorted — sound
/// because `factor` is already gated to an exact power of two inside
/// the f64-exact 2^53 window before this is ever called, so the float
/// product of an integer-window dividend by that singleton never
/// rounds away from the integer `int.mul` would have named. A
/// non-integral or out-of-window float answer still declines
/// (`int_transfer_answer`'s own guard, mirrored here since this
/// function bypasses that helper to reuse the float `Mul` row instead
/// of an `int.*` one).
pub(in crate::expressions) fn float_mul_as_shift_fallback(
    left_set: &RefinedSet,
    factor_set: &RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    use refined_kernel::transfer_questions::TransferQuestionOp;
    use refined_kernel::transfer_questions::TransferAnswerKind;
    let nan_operand = refined_kernel::transfer_questions::PowOperandWire {
        kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
        set: make_refined_set(vec![]),
    };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&refined_kernel::transfer_questions::TransferQuestion {
            op: TransferQuestionOp::Mul,
            a: left_set.clone(),
            b: factor_set.clone(),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    });
    let answer = asked.ok()?;
    match answer.kind {
        TransferAnswerKind::Values => {
            if answer.values.iter().any(|v| v.fract() != 0.0 || v.abs() >= 2f64.powi(53)) {
                return None;
            }
            Some(known_values(answer.values, PrimitiveKind::Integer, grade))
        }
        TransferAnswerKind::Set => {
            if !requires_integer(&answer.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(answer.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// `binary_arithmetic_value`'s own two-known-values decline, taken to
/// the SET path: when at least one operand carries a numeric SET
/// (seeded parameter range, or a sort-only set-unknown answer) rather
/// than one known single value, this asks the kernel's `transfer` for
/// the admitted operators (`admitted_transfer_op`) instead of losing
/// the determination to `unknown()`. Both operands must read as a
/// numeric set-or-known-value (`transferable_numeric_operand`); a
/// non-numeric or untagged-Set operand still declines. The kernel's own
/// float image (a certified enclosure, `TransferAnswerKind::Set`) or a
/// pair of Integer-marked singletons narrowing to one exact answer
/// (`TransferAnswerKind::Values`) both bind as `known_set`/
/// `known_values` at the WEAKER of the two operands' own trust grades
/// (`derived_trust_level` — the kernel's own answer can never overstate
/// past what its inputs were already trusted at). A kernel refusal
/// (a set shape `transfer` does not decide, e.g. the sequence/string
/// forms in the RefinedSet grammar) is caught exactly as
/// `assignability.rs`'s own containment ask catches one — refusal
/// reads as `None` here (the caller falls back to `unknown()`), never
/// a crash and never a guessed value.
///
/// Whether a divisor's set PROVABLY EXCLUDES zero — the gate `/` needs
/// before it may ask `binary64.div` (see `admitted_transfer_op`'s `Div`
/// bullet: away from a zero divisor the two `/`s name the same
/// theorem, but AT one, ECMA answers a determined `±Infinity`/NaN
/// while Python raises `ZeroDivisionError`, arith.10). `0.0` is a
/// member of `divisor` exactly when the kernel's own membership
/// decider says so (`kernel.member`, `x ∈ A` — `memberB_iff`, the same
/// ask `assignability.rs`'s containment law poses); `member` is total
/// over every enclosure this file builds, so there is no refusal shape
/// here to catch the way `scalar_subset`/`seq_subset` need one. A
/// divisor set that DOES admit zero answers `false` here — this
/// function only PROVES the exclusion, it never guesses it, so any
/// doubt routes to "may be zero."
pub(in crate::expressions) fn divisor_provably_excludes_zero(divisor: &RefinedSet, kernel: &Arc<RefinedTSKernel>) -> bool {
    let asked = crate::kernel_ask::ask_kernel(|| (kernel.member)(divisor, &[0.0]));
    matches!(asked, Ok(false))
}

pub(in crate::expressions) fn transfer_over_sets(
    op: Operator,
    left: &AbstractValue,
    right: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    // gated to the case `binary_arithmetic_value`'s own known-values
    // path cannot already answer: AT LEAST ONE operand must be a
    // `Kind::Set` (a seeded range, or a sort-only set-unknown answer).
    // Two known single values stay on the existing pure-Rust path
    // unchanged — this function never re-derives a determination the
    // fast path already owns.
    if left.kind != Kind::Set && right.kind != Kind::Set {
        return None;
    }
    let (left_set, left_sort) = transferable_numeric_operand(left)?;
    let (right_set, right_sort) = transferable_numeric_operand(right)?;
    let grade = refined_domain::trust_grades::derived_trust_level(
        refined_domain::trust_grades::TrustProved,
        &[left.clone(), right.clone()],
    );
    // BOTH operands int-sorted: the exact `int` theory serves the
    // operation, never the float image (arith.1). `/` is the one
    // exception and stays on the float path below — arith.9 widens
    // int/int to float, so it has no int-sorted row at all. An int row
    // the kernel declines (an operand that is not a closed singleton,
    // a zero divisor, an exponent past the boundary's own fuel ceiling)
    // falls through to the float path unchanged rather than losing the
    // determination outright.
    if op != Operator::Div && left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer {
        if let Some(answer) = int_transfer_over_sets(op, right, &left_set, &right_set, grade, kernel) {
            return Some(answer);
        }
    }
    // `**` is absent from `admitted_transfer_op` (that row's own doc:
    // "`Pow` has no float binary-arithmetic-transfer row in this
    // family... a different question shape from the plain two-`RefinedSet`
    // rows this function poses") — falling through to `admitted_transfer_op(op)?`
    // below would decline `**` outright for any SET-shaped base, never
    // asking the kernel at all. `pow_over_sets` poses the real `pow` wire
    // question (`TransferQuestionOp::Pow`, `PowOperandWire::Set` on both
    // sides) instead, so this returns here rather than joining the
    // `admitted_transfer_op` dispatch.
    if op == Operator::Pow {
        return super::power::pow_over_sets(&left_set, left_sort, right, grade, kernel);
    }
    let transfer_op = admitted_transfer_op(op)?;
    // `Div`'s always-float override (arith.9: "the type is widened even
    // when the arguments are exact integers") beats the both_int rule
    // outright — Python `/` never stays Integer-sorted regardless of
    // its operands' own sorts. Every other admitted op (Add/Sub/Mult)
    // keeps the same both_int rule binary_arithmetic_value's
    // known-values path uses: Integer only when BOTH sides are
    // Integer-sorted.
    let both_int = op != Operator::Div && left_sort == PrimitiveKind::Integer && right_sort == PrimitiveKind::Integer;
    let result_sort = if both_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    // arith.10's carve-out: a divisor whose set admits zero diverges from
    // `binary64.div` asked directly — ECMA answers a determined
    // `±Infinity`/NaN at zero, Python raises `ZeroDivisionError`
    // (`divisor_is_provably_always_zero`'s window owns the unconditional
    // raise, named in `binop_provable_raise`). A window that admits zero
    // WITHOUT being entirely zero raises only on its zero arm and
    // determines a value on every other input — `split_divisor_transfer`
    // asks `binary64.div` on the zero-excluded negative and positive
    // halves of the divisor separately and unions the two answers, so the
    // value question determines on the non-raising split rather than
    // decline outright. An always-zero divisor has no non-raising half at
    // all (both halves are empty), so it still declines here exactly as
    // before — the raise is the whole answer for that window.
    let answer = if op == Operator::Div && !divisor_provably_excludes_zero(&right_set, kernel) {
        super::division::split_divisor_transfer(left_set, &right_set, kernel)?
    } else {
        let asked = crate::kernel_ask::ask_kernel(|| {
            (kernel.transfer)(&refined_kernel::transfer_questions::TransferQuestion {
                op: transfer_op,
                a: left_set,
                b: right_set,
                c: 0.0,
                base: refined_kernel::transfer_questions::PowOperandWire {
                    kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
                    set: make_refined_set(vec![]),
                },
                exp: refined_kernel::transfer_questions::PowOperandWire {
                    kind: refined_kernel::transfer_questions::PowOperandKind::NaN,
                    set: make_refined_set(vec![]),
                },
            })
        });
        asked.ok()?
    };
    use refined_kernel::transfer_questions::TransferAnswerKind;
    match answer.kind {
        TransferAnswerKind::Values => {
            if both_int && answer.values.iter().any(|v| v.fract() != 0.0) {
                // an Integer-sorted pair whose kernel answer is not itself
                // integral cannot happen under Add/Sub/Mul (both are exact
                // on integer enclosures) — an honest decline rather than a
                // claim the sort rule would contradict, should the kernel
                // ever widen this row
                return None;
            }
            Some(known_values(answer.values, result_sort, grade))
        }
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(result_sort),
            ..known_set(answer.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::Unknown => {
            // The kernel's honest top for this operand pair: no enclosure
            // narrows the result (e.g. a bounded set times an unbounded
            // one), but the SORT rule still holds — the same language-level
            // guarantee float_sorted_unknown carries for the math family —
            // so the answer is sort-known, value-unknown, never nothing.
            // A downstream clamp (max/min) can still bound it, which is
            // exactly how a two-free-name comprehension element derives.
            Some(if both_int {
                AbstractValue {
                    kind_tag: Some(PrimitiveKind::Integer),
                    ..known_set(
                        make_refined_set(vec![
                            refined_sets::refinement_forms::integer(),
                            refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
                        ]),
                        None,
                        refined_domain::trust_grades::TrustSpec,
                        SetKindTag::None,
                    )
                }
            } else {
                refined_domain::abstract_value::float_sorted_unknown()
            })
        }
        // A may-be-NaN answer must never masquerade as a NaN-free set.
        TransferAnswerKind::NaN => None,
    }
}
