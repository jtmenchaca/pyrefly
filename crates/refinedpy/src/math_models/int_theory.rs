use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustLevel;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::PowOperandKind;
use refined_kernel::transfer_questions::PowOperandWire;
use refined_kernel::transfer_questions::TransferAnswerKind;
use refined_kernel::transfer_questions::TransferQuestion;
use refined_kernel::transfer_questions::TransferQuestionOp;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::RefinedSet;

use super::single_numeric_operand;
use super::sqrt_call_over_set;
use super::rounding_call_over_set;

/// Poses one `int.*` question — the exact unbounded-integer theory
/// (`boundary/python.lean`'s `pythonTransferOfOp1`/`pythonTransferOfOp2`)
/// — and reads the answer back as an INTEGER-SORTED value. The exact
/// mirror of `rounding_call_over_set` above: same `TransferQuestion`
/// construction, same `catch_unwind` refusal discipline, same
/// `TransferAnswerKind` match. `b` is the empty set for the one-operand
/// members.
///
/// Two guards `rounding_call_over_set` does not need, both about the
/// unboundedness python-pins.md arith.1 states ("integers have unlimited
/// precision"): a non-integral answer declines (no `int.*` member can
/// produce one), and an answer past the f64-exact 2^53 window declines
/// because `boundary/encode_sets.lean`'s `encodeNumber` puts every
/// result through `roundNE` before it crosses the wire — a bigger result
/// arrives ROUNDED, and this file's carrier is f64, so claiming it as
/// exact would claim a value CPython never computes.
pub(super) fn int_transfer_call(
    op: TransferQuestionOp,
    a: RefinedSet,
    b: RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op,
            a,
            b,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    match asked.kind {
        TransferAnswerKind::Values => {
            if asked.values.iter().any(|v| v.fract() != 0.0 || v.abs() >= 2f64.powi(53)) {
                return None;
            }
            Some(known_values(asked.values, PrimitiveKind::Integer, grade))
        }
        // a SET answer must carry its own integrality before it is
        // tagged Integer-sorted — tagging one without that mark would
        // claim an integrality the kernel did not state
        TransferAnswerKind::Set => {
            if !requires_integer(&asked.set) {
                return None;
            }
            Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(asked.set, None, grade, SetKindTag::None)
            })
        }
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// An operand an `int.*` question can be posed over: an int-sorted
/// `Kind::Set` reads as its own set, and a known single int-sorted value
/// reads as the one-element set `{v}`, so a set-vs-known-value pair
/// poses the same question a set-vs-set pair does — the same reading
/// `expressions.rs::transferable_numeric_operand` performs, narrowed to
/// the INT sort because every `int.*` member's domain is the integers
/// (python-pins.md arith.1). A Float-sorted operand declines: `math.gcd`
/// and friends raise `TypeError` on one, so there is no value to answer.
pub(super) fn int_transferable_operand(value: &AbstractValue) -> Option<RefinedSet> {
    if let Some((number, is_int)) = single_numeric_operand(value) {
        if !is_int || number.fract() != 0.0 {
            return None;
        }
        return Some(make_refined_set(vec![one_of(&[number])]));
    }
    if value.kind == Kind::Set
        && matches!(value.kind_tag, Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Boolean))
    {
        return Some(value.set.clone());
    }
    None
}

/// The `int.*` rows this file serves where its own concrete paths
/// decline — a SET operand, or a known-value pair no pure-Rust row above
/// computes. Each names the pins row that elects it:
///
/// - `isqrt` → `int.isqrt` (pow.4: "the integer square root of the
///   nonnegative int n... the floor of the exact square root"). Tried
///   only after `isqrt_call`'s concrete path declines, so a known
///   nonnegative int still answers without a kernel round trip.
/// - `factorial` → `int.factorial` (arith.21: "exact int factorial,
///   raises `ValueError` if n is not integral or negative").
/// - `gcd`/`lcm` → `int.gcd`/`int.lcm` (arith.20: "exact
///   greatest-common-divisor / least-common-multiple... on the unbounded
///   `int` theory"). CPython's own signature is variadic
///   (`math.gcd(*integers)`); the kernel members are binary, so this
///   folds the arguments left-to-right through repeated asks —
///   associativity of gcd/lcm is what makes the fold equal the variadic
///   call, and a fold step the kernel declines declines the whole call
///   rather than answering a partial product.
/// - `comb`/`perm` → `int.comb`/`int.perm` (arith.21: "exact
///   combinatorial counts, same int theory"). `math.perm(n)` with the
///   count omitted defaults to `k = n`, per the same clause's
///   `perm(n, k=None)` signature.
///
/// A negative operand is NOT filtered here: the kernel arms read their
/// `Nat`-domain operands through `exactNatOf` (`boundary/python.lean`),
/// which answers `none` on a negative exact integer rather than
/// extending the theory function silently — so the refusal that
/// corresponds to CPython's `ValueError` is the kernel's own, not a
/// condition this file restates.
pub(super) fn int_theory_call(
    function: &str,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let grade = derived_trust_level(TrustProved, arguments);
    let empty = make_refined_set(vec![]);
    match function {
        "isqrt" => {
            let [only] = arguments else { return None };
            let operand = int_transferable_operand(only)?;
            if let Some(answer) = int_transfer_call(TransferQuestionOp::IntIsqrt, operand.clone(), empty, grade, kernel) {
                return Some(answer);
            }
            isqrt_as_sqrt_floor_composition(&operand, grade, kernel)
        }
        "factorial" => {
            let [only] = arguments else { return None };
            int_transfer_call(
                TransferQuestionOp::IntFactorial,
                int_transferable_operand(only)?,
                empty,
                grade,
                kernel,
            )
        }
        // the variadic fold — gcd/lcm are associative, so folding the
        // binary member left-to-right computes the same value the
        // variadic call does
        "gcd" | "lcm" => {
            let op = if function == "gcd" { TransferQuestionOp::IntGcd } else { TransferQuestionOp::IntLcm };
            let (first, rest) = arguments.split_first()?;
            if rest.is_empty() {
                return None;
            }
            let mut accumulated = int_transferable_operand(first)?;
            let mut answer = None;
            for argument in rest {
                let next = int_transferable_operand(argument)?;
                let step = int_transfer_call(op, accumulated, next, grade, kernel)?;
                accumulated = int_transferable_operand(&step)?;
                answer = Some(step);
            }
            answer
        }
        "comb" => {
            let [n, k] = arguments else { return None };
            int_transfer_call(
                TransferQuestionOp::IntComb,
                int_transferable_operand(n)?,
                int_transferable_operand(k)?,
                grade,
                kernel,
            )
        }
        // `math.perm(n)` defaults k to n (functions' own `perm(n,
        // k=None)` signature, arith.21's clause)
        "perm" => {
            let n = arguments.first()?;
            let n_set = int_transferable_operand(n)?;
            let k_set = match arguments.get(1) {
                Some(k) => int_transferable_operand(k)?,
                None if arguments.len() == 1 => n_set.clone(),
                None => return None,
            };
            int_transfer_call(TransferQuestionOp::IntPerm, n_set, k_set, grade, kernel)
        }
        _ => None,
    }
}

/// `math.isqrt(n)` over a NON-SINGLETON `RefinedSet` (a seeded
/// parameter range, or a bounded set another transfer already
/// produced) — the FALLBACK `int_theory_call`'s own `isqrt` row takes
/// once `int.isqrt` (`boundary/python.lean`) has already declined on
/// it. `int.isqrt` matches `exactNatOf` on its one operand only — it
/// carries no general-window arm the way `int.floorDiv` does
/// (`shift_as_int_composition`'s own doc names the identical gap for
/// `int.mul`), so a bounded, non-singleton `n` always reads back
/// `.unknown` from `int.isqrt` and would decline outright with no
/// further attempt.
///
/// pow.4 states `math.isqrt(n)` IS "the floor of the exact square
/// root" — the same composition CPython's own implementation performs
/// — so this asks `binary64.sqrt` (`sqrt_call_over_set`'s own kernel
/// row, which DOES carry a general-window decider) over `operand`,
/// then floors that Float-sorted enclosure through `binary64.floor`
/// (`rounding_call_over_set`'s own `Floor` row) back to Integer sort. Both rows
/// already exist and are already exercised by `math.sqrt`/`math.floor`
/// over a declared range elsewhere in this file; this composes them
/// rather than adding a third kernel-transfer implementation.
///
/// Gated on `operand` PROVABLY EXCLUDING every negative value first
/// (`kernel.scalar_subset(operand, [0, +inf))`, the same "prove, never
/// guess" discipline `divisor_provably_excludes_zero`
/// (`expressions.rs`) keeps for its own zero exclusion): `math.isqrt`
/// raises `ValueError` on a negative n (`isqrt_call`'s own doc), and
/// `math.sqrt` raises `ValueError` on one too (pow.3) — composing
/// sqrt+floor over a window that ADMITS a negative would either
/// silently answer a value the real call never produces (the negative
/// arm raises) or hand `binary64.sqrt` an operand its own domain
/// excludes. `int_transferable_operand`'s caller already only reaches
/// this fallback with an Integer-sorted `operand`, which `int.isqrt`
/// itself already requires — the composition adds no new sort
/// admission, only a wider VALUE reading of the same sort.
pub(super) fn isqrt_as_sqrt_floor_composition(
    operand: &RefinedSet,
    grade: TrustLevel,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let nonneg = make_refined_set(vec![at_least(0.0)]);
    if !(kernel.scalar_subset)(operand, &nonneg) {
        return None;
    }
    let operand_value = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..known_set(operand.clone(), None, grade, SetKindTag::None) };
    let root = sqrt_call_over_set(&operand_value, kernel)?;
    rounding_call_over_set(TransferQuestionOp::Floor, &root, kernel)
}
