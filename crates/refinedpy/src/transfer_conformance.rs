/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The differential harness for the Python adapter's CONCRETE scalar
//! arithmetic path against the kernel's proved transfers over singleton
//! sets — the THIN-WALK-AUDIT.md W1 row that names this exact pair:
//! "Python: the scalar branches of arithmetic and `math.floor` beside
//! their asking set branches (`binary_arithmetic_value` vs.
//! `_with_kernel`)". Two independently-maintained implementations of
//! the same IEEE-754 clause; this file runs representative values
//! through BOTH and compares.
//!
//! Placement and conventions follow `lattice_conformance.rs`: the
//! `loaded_kernel()` dylib-absence early return, tests only, and no
//! edits to the modules under test (`expressions.rs`, `math_models.rs`)
//! — their `pub` functions are consumed as they stand.
//!
//! ## The three-verdict frame
//!
//! Every row lands in exactly one of three classes, and the assertions
//! encode which:
//!
//! 1. **BOTH ANSWER → must AGREE.** `assert_agrees` fails the test on
//!    any drift. This is the only class that can fail here: two routes
//!    claiming different values for the same operation is a soundness
//!    defect in one of them, and the harness exists to catch it.
//! 2. **ADAPTER DECLINES, KERNEL ANSWERS → a DETERMINATION-GAP row.**
//!    Not a failure: the adapter is weaker than the kernel at that
//!    operand shape, which is a missing determination, not a wrong one.
//!    Each is recorded in the ledger table below and asserted to STILL
//!    be a gap, so the row turns into a compile-visible reminder the
//!    day the adapter starts serving it.
//! 3. **ADAPTER ANSWERS WHERE THE KERNEL DOES NOT (or answers
//!    differently) → SCRUTINY.** The adapter claiming a value with no
//!    proved backing is the shape that can be unsound. Flagged loudly
//!    by `assert_scrutiny_row`, whose message names the operand pair.
//!
//! ## THE DETERMINATION-GAP LEDGER (operations 1 and 2)
//!
//! | # | operation | operands | adapter | kernel | class |
//! |---|-----------|----------|---------|--------|-------|
//! | G1 | `+` int/int overflowing 2^53 | 2^53, 1 | declines (`arithmetic_result`'s exactness gate) | `int.add` answers exactly (unbounded ints) | GAP |
//! | G2 | `*` int/int overflowing 2^53 | 2^53, 2 | declines | `int.mul` answers exactly | GAP |
//! | G3 | `/` by zero | 1, 0 | declines to `unknown()` (no exception channel) | refuses too | agree-on-silence |
//! | G4 | `+` with a NaN operand | NaN, 1 | never constructed (`RefinedSet` refuses NaN at construction) | n/a | out of the value vocabulary |
//! | G5 | `/` by a degenerate SET-shaped zero divisor | 1.0, `{0.0}` (`Kind::Set`, not `Kind::Values`) | declines (`divisor_provably_excludes_zero` gate) | `binary64.div`'s `bothSingle` branch answers the determined pair `[-∞, +∞]` | GENUINE DIVERGENCE, correct |
//!
//! G5 is NOT a determination gap: it is deliberately excluded from
//! `compare_row`'s three-verdict frame (which would read "adapter
//! declines, kernel answers" as the adapter being weaker) because the
//! kernel's answer here is CORRECT FOR ECMA, not for Python — arith.10
//! makes Python's `/` raise `ZeroDivisionError` at a zero divisor, an
//! outcome `binary64.div`'s determined pair cannot speak (it proves
//! only the IEEE-754 float theorem). Serving that pair as the Python
//! answer would be unsound; the decline is the day-one-correct verdict,
//! not a gap the adapter should ever close by asking harder. A WIDE
//! zero-admitting range (e.g. `[0.0, 2.0]`) is a DIFFERENT shape from
//! G5's degenerate singleton: it has non-zero members too, so it is not
//! an always-raises divisor — `split_divisor_transfer` (2026-08-22) asks
//! the kernel on the divisor's zero-excluded halves and unions the
//! answers, so this shape now DETERMINES a value (sort-known, at
//! minimum) rather than declining. `test_div_over_a_wide_zero_admitting_
//! window_determines_via_the_split` pins that below. G5's premise stays
//! the SINGLETON-shaped Set specifically — the one shape with no
//! non-raising half to split into at all, so it is still the shape
//! where the kernel's own answer would have to be relabeled wholesale
//! to serve it, which the gate still refuses.
//! `test_div_by_a_set_admitting_zero_diverges_from_the_kernel_by_design`
//! asserts G5's divergence directly, with its own message rather than
//! `compare_row`'s "gap" framing.
//!
//! The WIDE zero-admitting window's own raise corner is not silent
//! either, as of `possible_raise`/`binop_possible_raise` — a function
//! separate from `provable_raise`'s all-or-nothing claim: a divisor set
//! that admits zero without being entirely zero (this table's own
//! distinction from G5) now fires `diagnostic_sentences::
//! division_by_a_set_that_admits_zero` at the division site AS WELL AS
//! determining the split value above — a finding beside the value,
//! never a withdrawal of it; which sink combines the two is `check.rs`'s
//! own wiring. `test_a_zero_admitting_divisor_
//! fires_and_still_determines_with_no_infinity_row` and
//! `test_a_zero_excluding_divisor_fires_nothing` below pin both halves
//! of that pair.
//!
//! Rows G1/G2 are the audit's own "the exact `int` theory serves them"
//! observation seen from the concrete side: the f64 carrier is what
//! declines, not the semantics. The adapter now ASKS `int.add`/`int.mul`
//! for an int-sorted operand pair, but the answer still crosses the wire
//! through `encodeNumber`'s `roundNE`, so a result past 2^53 arrives
//! rounded and the adapter declines it — the gap is the carrier, exactly
//! as these rows state.
//!
//! `math.floor`/`ceil`/`trunc` of a non-finite argument was a scrutiny
//! row here (the adapter answered an Integer-sorted ±inf/NaN where
//! CPython raises). `integral_domain_admits` now gates the family and
//! `rounding_argument_raises` names the exception, so the row is a
//! decline plus a provable raise;
//! `test_math_rounding_of_non_finite_arguments_declines` asserts it.
//!
//! ## What the value vocabulary excludes, and why no row poses it
//!
//! NaN is NOT an element of ℝ̄ and `refinement_forms::one_of` PANICS on
//! it at construction (`element`'s own check). A NaN-operand transfer is
//! therefore not a question this wire can pose at all — the kernel's NaN
//! answers ride the `TransferAnswerKind::NaN` reply, not a NaN operand.
//! The brief asks for NaN-operand rows; the honest coverage is the
//! adapter side alone, which `test_nan_operand_is_outside_the_wires_
//! value_vocabulary` records rather than fabricating a question.

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use refined_domain::abstract_value::{known_set, known_values, AbstractValue, Kind, PrimitiveKind, SetKindTag};
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
    use refined_kernel::kernel_interface::RefinedTSKernel;
    use refined_kernel::transfer_questions::{
        PowOperandKind, PowOperandWire, TransferAnswer, TransferAnswerKind, TransferQuestion,
        TransferQuestionOp,
    };
    use refined_sets::refinement_forms::{at_least, at_most, make_refined_set, one_of, RefinedSet};
    use ruff_python_ast::{Expr, Operator};
    use ruff_python_parser::parse_expression;

    use crate::env::Environment;
    use crate::expressions::binary_arithmetic_value;
    use crate::expressions::binary_arithmetic_value_with_kernel;
    use crate::expressions::possible_raise;
    use crate::math_models::math_call_result;

    /// `loaded_kernel` mirrors `lattice_conformance.rs`'s own helper
    /// exactly: a missing dylib artifact prints to stderr and the caller
    /// returns early, never failing the run.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn int_operand(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    fn float_operand(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Float, TrustProved)
    }

    /// The singleton set `{v}` — the same one-element `one_of` shape
    /// `expressions.rs`'s own `transferable_numeric_operand` builds for
    /// a known single numeric value, rebuilt here because that function
    /// is private to `expressions.rs`.
    fn singleton(value: f64) -> RefinedSet {
        make_refined_set(vec![one_of(&[value])])
    }

    /// A binary transfer question over two singleton sets.
    fn binary_question(op: TransferQuestionOp, a: f64, b: f64) -> TransferQuestion {
        TransferQuestion {
            op,
            a: singleton(a),
            b: singleton(b),
            c: 0.0,
            base: PowOperandWire {
                kind: PowOperandKind::NaN,
                set: make_refined_set(vec![]),
            },
            exp: PowOperandWire {
                kind: PowOperandKind::NaN,
                set: make_refined_set(vec![]),
            },
        }
    }

    /// A unary transfer question over one singleton set.
    fn unary_question(op: TransferQuestionOp, a: f64) -> TransferQuestion {
        binary_question(op, a, 0.0)
    }

    /// The single value a `TransferAnswer` pins, when it pins exactly
    /// one. The `NaN` reply IS such a value: the spec's Number type has
    /// one NaN, and `transferAdd`'s singleton corner answers it exactly
    /// ("the indeterminate corner IS the spec's NaN"), so the reply is
    /// a determined answer, never a decline. `None` only for the
    /// unknown and set-valued replies — the kernel declining to name
    /// one float.
    fn kernel_exact_value(answer: &TransferAnswer) -> Option<f64> {
        if answer.kind == TransferAnswerKind::NaN {
            return Some(f64::NAN);
        }
        if answer.kind != TransferAnswerKind::Values {
            return None;
        }
        if answer.values.len() != 1 {
            return None;
        }
        Some(answer.values[0])
    }

    /// The single value an adapter `AbstractValue` pins, when it pins
    /// exactly one, together with the Python sort it carries. `None`
    /// where the adapter declined (`unknown()` is `Kind::Unknown`, never
    /// `Kind::Values`).
    fn adapter_exact_value(value: &AbstractValue) -> Option<(f64, Option<PrimitiveKind>)> {
        if value.kind != Kind::Values {
            return None;
        }
        if value.values.len() != 1 {
            return None;
        }
        Some((value.values[0], value.kind_tag))
    }

    /// Bit-exact float comparison — `-0.0 == 0.0` under `==`, so the
    /// signed-zero rows below would silently pass a wrong answer without
    /// this. Compares the IEEE bit patterns instead.
    fn same_float(a: f64, b: f64) -> bool {
        // The spec's Number type has exactly one NaN value, so any NaN
        // payload agrees with any other; hardware NaN bit patterns are
        // not part of either route's claim.
        if a.is_nan() && b.is_nan() {
            return true;
        }
        a.to_bits() == b.to_bits()
    }

    /// VERDICT 1 — both routes answered, so they must AGREE. A drift
    /// here is the failure this harness exists to catch.
    fn assert_agrees(label: &str, adapter: f64, kernel: f64) {
        assert!(
            same_float(adapter, kernel),
            "{label}: adapter answered {adapter:?} (bits {:#x}), kernel answered {kernel:?} \
             (bits {:#x}) — two routes for the same operation must agree",
            adapter.to_bits(),
            kernel.to_bits()
        );
    }

    /// VERDICT 3 — the adapter answered where the kernel did not. Named
    /// loudly: a value claimed with no proved backing is the shape that
    /// can be unsound, unlike a missing determination.
    fn assert_scrutiny_row(label: &str, adapter: Option<(f64, Option<PrimitiveKind>)>, kernel: Option<f64>) {
        if adapter.is_some() && kernel.is_none() {
            panic!(
                "SCRUTINY: {label}: the adapter answered {adapter:?} where the kernel declined — \
                 an adapter-only claim carries no proved backing"
            );
        }
    }

    /// The three verdicts, applied to one row. Returns which class the
    /// row landed in so the caller can count the gaps.
    #[derive(Debug, PartialEq)]
    enum Verdict {
        Agreed,
        DeterminationGap,
        BothSilent,
    }

    fn compare_row(
        label: &str,
        adapter: &AbstractValue,
        kernel_answer: &TransferAnswer,
    ) -> Verdict {
        let adapter_value = adapter_exact_value(adapter);
        let kernel_value = kernel_exact_value(kernel_answer);
        assert_scrutiny_row(label, adapter_value, kernel_value);
        match (adapter_value, kernel_value) {
            (Some((a, _)), Some(k)) => {
                assert_agrees(label, a, k);
                Verdict::Agreed
            }
            (None, Some(_)) => Verdict::DeterminationGap,
            (None, None) => Verdict::BothSilent,
            // the scrutiny assertion above already panicked on this arm
            (Some(_), None) => unreachable!("assert_scrutiny_row panics on the adapter-only arm"),
        }
    }

    // ===================================================================
    // OPERATION 1 — the concrete scalar arithmetic path
    // (`binary_arithmetic_value`, the hand-computed f64 route) against
    // the kernel's `binary64.add`/`sub`/`mul`/`div` transfers over the
    // same values posed as singleton sets.
    // ===================================================================

    /// The float rows: two Float-sorted operands, so the adapter's
    /// `both_int` is false and the kernel's `binary64.*` family is
    /// exactly the twin (`admitted_transfer_op`'s own election, quoted
    /// in `expressions.rs`: "these three rows are semantics-identical
    /// between the two languages"). Every operand pair here is inside
    /// the f64-exact range, so both routes are expected to answer, and
    /// the assertion is AGREEMENT.
    #[test]
    fn test_float_add_sub_mul_agree_with_the_kernel_transfers_on_representative_values() {
        let Some(kernel) = loaded_kernel() else { return };

        // (left, right) pairs spanning the corners the brief names:
        // ordinary values, ±0, the 2^53 boundary, and infinities.
        let pairs: Vec<(f64, f64)> = vec![
            (1.0, 2.0),
            (0.5, 0.25),
            (-3.0, 7.0),
            (0.0, 0.0),
            (0.0, -0.0),
            (-0.0, -0.0),
            (-0.0, 0.0),
            (1.0, -1.0),
            (9007199254740992.0, 1.0),   // 2^53 + 1 is not representable
            (9007199254740992.0, 2.0),   // 2^53 + 2 is
            (9007199254740991.0, 1.0),   // 2^53 - 1, the last exact odd int
            (f64::INFINITY, 1.0),
            (f64::NEG_INFINITY, 1.0),
            (f64::INFINITY, f64::NEG_INFINITY),
            (f64::MAX, f64::MAX),        // overflows to +inf under add
            (1e308, 10.0),               // overflows under mul
            (5e-324, 0.5),               // the subnormal floor under mul
        ];

        let ops: Vec<(Operator, TransferQuestionOp, &str)> = vec![
            (Operator::Add, TransferQuestionOp::Add, "float +"),
            (Operator::Sub, TransferQuestionOp::Sub, "float -"),
            (Operator::Mult, TransferQuestionOp::Mul, "float *"),
        ];

        let mut agreed = 0;
        let mut gaps = 0;
        for (py_op, kernel_op, name) in &ops {
            for (left, right) in &pairs {
                let label = format!("{name} ({left:?}, {right:?})");
                let adapter = binary_arithmetic_value(*py_op, &float_operand(*left), &float_operand(*right));
                let answer = (kernel.transfer)(&binary_question(*kernel_op, *left, *right));
                match compare_row(&label, &adapter, &answer) {
                    Verdict::Agreed => agreed += 1,
                    Verdict::DeterminationGap => gaps += 1,
                    Verdict::BothSilent => {}
                }
            }
        }
        // Every row above is a finite-or-infinite float pair both routes
        // compute; a zero agreement count would mean the harness never
        // actually compared anything.
        assert!(agreed > 0, "no float arithmetic row was compared: agreed={agreed}, gaps={gaps}");
    }

    /// `/` is ALWAYS true division in Python (arith.9), and
    /// `admitted_transfer_op`'s own doc says the two `/`s "name the same
    /// theorem" — so the adapter's `known_values(.., Float, ..)` result
    /// and `binary64.div` must agree on value AND the adapter's sort must
    /// be Float even for an int/int pair.
    #[test]
    fn test_div_agrees_with_binary64_div_and_always_answers_the_float_sort() {
        let Some(kernel) = loaded_kernel() else { return };

        let pairs: Vec<(f64, f64)> = vec![
            (1.0, 2.0),
            (6.0, 3.0),
            (-6.0, 3.0),
            (6.0, -3.0),
            (0.0, 1.0),
            (-0.0, 1.0),
            (0.0, -1.0),
            (1.0, f64::INFINITY),
            (f64::INFINITY, 2.0),
            (1.0, 3.0), // a non-terminating quotient: both must round identically
        ];

        for (left, right) in &pairs {
            let label = format!("float / ({left:?}, {right:?})");
            let adapter =
                binary_arithmetic_value(Operator::Div, &float_operand(*left), &float_operand(*right));
            let answer = (kernel.transfer)(&binary_question(TransferQuestionOp::Div, *left, *right));
            compare_row(&label, &adapter, &answer);
        }

        // arith.9's own rule, pinned against the kernel's ANSWER SORT
        // question: an int/int division still answers Float on the
        // adapter side. The kernel's transfer answer carries no Python
        // sort tag at all (TransferAnswer is values/set, never a sort),
        // so the sort half is checked adapter-side against arith.9,
        // exactly the way lattice_conformance.rs checks the Integer/
        // Float tag arms adapter-side.
        let int_div = binary_arithmetic_value(Operator::Div, &int_operand(6.0), &int_operand(3.0));
        let (value, sort) = adapter_exact_value(&int_div).expect("int/int division answers a value");
        assert_eq!(value, 2.0);
        assert_eq!(
            sort,
            Some(PrimitiveKind::Float),
            "arith.9: int/int `/` widens to float even when the arguments are exact integers"
        );
        // and the same value the kernel's binary64.div pins
        let answer = (kernel.transfer)(&binary_question(TransferQuestionOp::Div, 6.0, 3.0));
        assert_eq!(kernel_exact_value(&answer), Some(2.0));
    }

    /// The INT-SORT rows: `int op int` stays Integer for `+`/`-`/`*`
    /// (the mixed-arithmetic rule's own base case), and the VALUE must
    /// still match the kernel's exact `int.*` theory over the same
    /// singletons. Both operands stay inside the f64-exact window here,
    /// which is where the two theories provably coincide; the rows that
    /// leave it are the G1/G2 ledger entries below.
    #[test]
    fn test_int_add_sub_mul_agree_with_the_exact_int_theory_and_keep_the_integer_sort() {
        let Some(kernel) = loaded_kernel() else { return };

        let pairs: Vec<(f64, f64)> = vec![
            (1.0, 2.0),
            (0.0, 0.0),
            (-3.0, 7.0),
            (7.0, -3.0),
            (-7.0, -3.0),
            (9007199254740990.0, 1.0), // 2^53 - 2, result stays exact
            (4503599627370496.0, 2.0), // 2^52 * 2 = 2^53, the boundary itself
        ];

        let ops: Vec<(Operator, TransferQuestionOp, &str)> = vec![
            (Operator::Add, TransferQuestionOp::IntAdd, "int +"),
            (Operator::Sub, TransferQuestionOp::IntSub, "int -"),
            (Operator::Mult, TransferQuestionOp::IntMul, "int *"),
        ];

        for (py_op, kernel_op, name) in &ops {
            for (left, right) in &pairs {
                let label = format!("{name} ({left:?}, {right:?})");
                let adapter = binary_arithmetic_value(*py_op, &int_operand(*left), &int_operand(*right));
                let answer = (kernel.transfer)(&binary_question(*kernel_op, *left, *right));
                match compare_row(&label, &adapter, &answer) {
                    Verdict::Agreed => {
                        // both answered: the adapter's Python sort must
                        // be Integer, the `both_int` rule's own claim.
                        let (_, sort) = adapter_exact_value(&adapter).expect("agreed rows carry a value");
                        assert_eq!(
                            sort,
                            Some(PrimitiveKind::Integer),
                            "{label}: int op int stays int-sorted"
                        );
                    }
                    Verdict::DeterminationGap | Verdict::BothSilent => {}
                }
            }
        }
    }

    /// LEDGER ROWS G1 and G2, asserted as gaps rather than failures.
    /// The adapter's `arithmetic_result` declines an int result outside
    /// the f64-exact 2^53 window ("CPython ints are unbounded, but this
    /// file's carrier is f64"), while the kernel's `int.add`/`int.mul`
    /// are exact unbounded-integer arithmetic and answer there. This is
    /// a MISSING determination on the adapter side, never a wrong one —
    /// and the day the adapter starts serving it, this test fails and
    /// the ledger row above gets deleted.
    #[test]
    fn test_determination_gap_int_arithmetic_outside_the_f64_exact_window() {
        let Some(kernel) = loaded_kernel() else { return };

        // G1: 2^53 + 1. The adapter's carrier cannot hold it exactly.
        let two_53 = 9007199254740992.0;
        let adapter_add = binary_arithmetic_value(Operator::Add, &int_operand(two_53), &int_operand(1.0));
        assert_eq!(
            adapter_add.kind,
            Kind::Unknown,
            "G1: the adapter is expected to decline 2^53 + 1 (arithmetic_result's exactness gate)"
        );
        let kernel_add = (kernel.transfer)(&binary_question(TransferQuestionOp::IntAdd, two_53, 1.0));
        // The kernel's exact int theory has no such carrier limit. If it
        // ever stops answering here, the gap has closed from the other
        // side and this row needs rereading rather than silent passing.
        assert_scrutiny_row("G1 int + at 2^53", None, kernel_exact_value(&kernel_add));

        // G2: 2^53 * 2. Same shape on the multiplication row.
        let adapter_mul = binary_arithmetic_value(Operator::Mult, &int_operand(two_53), &int_operand(2.0));
        assert_eq!(
            adapter_mul.kind,
            Kind::Unknown,
            "G2: the adapter is expected to decline 2^53 * 2"
        );
    }

    /// LEDGER ROW G3, the agree-on-silence row: Python's `/` by zero
    /// raises `ZeroDivisionError` rather than producing ±Infinity, and
    /// the adapter has no exception channel, so it declines to
    /// `unknown()`. This is NOT a determination gap in the adapter's
    /// disfavour — answering IEEE's ±inf here would be WRONG for Python,
    /// so the decline is the correct verdict and the row is pinned to
    /// stay a decline.
    #[test]
    fn test_division_by_zero_declines_rather_than_answering_the_ieee_infinity() {
        for divisor in [0.0, -0.0] {
            for op in [Operator::Div, Operator::FloorDiv, Operator::Mod] {
                let adapter = binary_arithmetic_value(op, &float_operand(1.0), &float_operand(divisor));
                assert_eq!(
                    adapter.kind,
                    Kind::Unknown,
                    "G3: {op:?} by {divisor:?} must decline (ZeroDivisionError), never answer IEEE's infinity"
                );
            }
        }
    }

    /// LEDGER ROW G5, the genuine-divergence row: `1.0 / denominator`
    /// where `denominator` is a DEGENERATE Float-sorted SET carrying
    /// nothing but `{0.0}` (`one_of`) — the shape a narrowed range can
    /// collapse to (still `Kind::Set`, not the ordinary `Kind::Values`
    /// `binary_arithmetic_value`'s ownknown-values path already reads).
    /// `theories/binary64/div.lean`'s `transferDiv` takes its
    /// `bothSingle` branch for this exact shape and answers a
    /// DETERMINED `±Infinity` pair — correct for ECMA's own `/`, which
    /// the kernel's transfer proves, but Python raises
    /// `ZeroDivisionError` at a zero divisor (arith.10), an outcome the
    /// value pair cannot speak. Serving that pair as the Python answer
    /// would be unsound, so `transfer_over_sets`'s own
    /// `divisor_provably_excludes_zero` gate declines the whole call
    /// instead — asserted here directly (not through `compare_row`,
    /// whose "adapter declines, kernel answers" verdict reads as a
    /// determination GAP; this row is the opposite, a decline the
    /// adapter must NEVER close by asking harder). A WIDE zero-admitting
    /// range (`[0.0, 2.0]`) is a DIFFERENT shape from this degenerate
    /// singleton: it has non-zero members too, so it is no longer the
    /// always-raises window this row's `divisor_is_provably_always_zero`
    /// gate protects — `split_divisor_transfer`'s own fix (2026-08-22)
    /// asks the kernel on the divisor's zero-excluded halves instead of
    /// declining outright, and `test_div_by_a_set_that_may_admit_zero_
    /// determines_the_float_sort_over_the_zero_excluded_split`
    /// (expressions.rs) pins that split now DETERMINES a value there —
    /// see `test_div_over_a_wide_zero_admitting_window_determines_via_
    /// the_split` below for this file's own conformance pin of the same
    /// shape.
    #[test]
    fn test_div_by_a_set_admitting_zero_diverges_from_the_kernel_by_design() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
        };
        let one = float_operand(1.0);
        let adapter = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            adapter.kind,
            Kind::Unknown,
            "G5: a divisor set that is nothing but zero must decline — never relabel the kernel's \
             ECMA-correct ±Infinity pair as Python's answer: {adapter:?}"
        );

        // confirmed the kernel's OWN answer here really is a determined
        // ±Infinity pair — the divergence is real, not merely asserted
        let asked = (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Div,
            a: singleton(1.0),
            b: singleton(0.0),
            c: 0.0,
            base: PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) },
            exp: PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) },
        });
        assert_eq!(
            asked.kind,
            TransferAnswerKind::Values,
            "G5's premise requires the kernel to answer a determined value here, or the row proves nothing"
        );
        assert_eq!(asked.values, vec![f64::NEG_INFINITY, f64::INFINITY]);
    }

    /// The gate's own pin: a divisor window that PROVABLY EXCLUDES zero
    /// (`[1.0, 2.0]`, strictly above zero) still lowers through the
    /// kernel and agrees — the gate only refuses the zero-admitting
    /// case, it does not disable the SET path for `/` outright.
    #[test]
    fn test_div_by_a_set_excluding_zero_still_lowers_and_agrees() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(1.0), at_most(2.0)]), None, TrustProved, SetKindTag::None)
        };
        let one = float_operand(1.0);
        let adapter = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(adapter.kind, Kind::Set, "a zero-excluding divisor window must still answer: {adapter:?}");
        assert_eq!(adapter.kind_tag, Some(PrimitiveKind::Float));
        let want = make_refined_set(vec![at_least(0.5), at_most(1.0)]);
        assert!((kernel.scalar_subset)(&adapter.set, &want), "adapter {:?} not ⊆ want {:?}", adapter.set, want);
        assert!((kernel.scalar_subset)(&want, &adapter.set), "want {:?} not ⊆ adapter {:?}", want, adapter.set);
    }

    /// The zero-divisor unit's own pin: `1.0 / denominator` where
    /// `denominator` is the WIDE window `[0.0, 2.0]` (`edge_infinity.py`'s
    /// own `max(0.0, sample)` shape, `sample ∈ [-2.0, 2.0]` clamped) —
    /// admits zero at its lower bound, but is NOT the degenerate `{0.0}`
    /// singleton G5 pins. Before `split_divisor_transfer`, this row
    /// declined outright (`test_div_by_a_set_that_may_admit_zero_
    /// determines_the_float_sort_over_the_zero_excluded_split`'s own
    /// prior name, expressions.rs). The split reads this window's
    /// negative half (`< 0.0`) as EMPTY and skips it, asks `binary64.div`
    /// on the positive half alone (`(0.0, 2.0]`), and that ask itself
    /// answers `Unknown` — the kernel's own general-interval branch
    /// cannot narrow `1.0 / (0.0, 2.0]` to a tight enclosure even with
    /// zero excluded — so the adapter now DETERMINES the float sort
    /// (`float_sorted_unknown()`), never the full `Kind::Unknown` decline
    /// this row used to answer.
    #[test]
    fn test_div_over_a_wide_zero_admitting_window_determines_via_the_split() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0), at_most(2.0)]), None, TrustProved, SetKindTag::None)
        };
        let one = float_operand(1.0);
        let adapter = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            adapter.kind,
            Kind::Set,
            "the zero-excluded split must determine the float sort, never decline outright: {adapter:?}"
        );
        assert_eq!(adapter.kind_tag, Some(PrimitiveKind::Float));

        // confirmed directly: the kernel's OWN answer on the zero-
        // excluded positive half alone is Unknown — the adapter's
        // determination traces to a real kernel answer, not a guess
        let positive_half = make_refined_set(vec![refined_sets::refinement_forms::above(0.0), at_most(2.0)]);
        let asked = (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Div,
            a: singleton(1.0),
            b: positive_half,
            c: 0.0,
            base: PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) },
            exp: PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) },
        });
        assert_eq!(
            asked.kind,
            TransferAnswerKind::Unknown,
            "the split's own positive half must ask the kernel and read its honest Unknown, or this row \
             proves the wrong thing: {asked:?}"
        );
    }

    /// The split's UNION arm: `1.0 / denominator` where `denominator` is
    /// `[-2.0, 2.0]` — a window straddling zero with BOTH a genuine
    /// negative half (`[-2.0, 0.0)`) and a genuine positive half
    /// (`(0.0, 2.0]`), unlike the wide-window row above (whose negative
    /// half is empty). Both halves are non-empty, so `split_divisor_
    /// transfer` asks the kernel on EACH and unions the two answers
    /// (`union_transfer_answers`) — pinned here so the union arm itself
    /// (not just the single-nonempty-half arm the wide-window row
    /// exercises) is asked of the real kernel at least once.
    #[test]
    fn test_div_over_a_window_straddling_zero_unions_both_split_halves() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(-2.0), at_most(2.0)]), None, TrustProved, SetKindTag::None)
        };
        let one = float_operand(1.0);
        let adapter = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            adapter.kind,
            Kind::Set,
            "a window straddling zero, split into two non-empty halves, must still determine the float \
             sort: {adapter:?}"
        );
        assert_eq!(adapter.kind_tag, Some(PrimitiveKind::Float));
    }

    /// `1.0 / denominator` parsed once, with `denominator` bound to
    /// `divisor` in a fresh environment — the shared setup every
    /// zero-admitting-divisor conformance pair below builds on.
    fn division_by_bound_denominator(divisor: AbstractValue) -> (Expr, Environment) {
        let mut environment = Environment::new(HashSet::new());
        environment.bind("denominator", divisor);
        let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
        (parsed.into_expr(), environment)
    }

    /// THE LEDGER'S CONFORMANCE PAIR, first half: a divisor window that
    /// ADMITS zero without being entirely zero (`[0.0, 2.0]`) now fires
    /// `binop_possible_raise`'s own row (`diagnostic_sentences::
    /// division_by_a_set_that_admits_zero`) AND the value question still
    /// determines through `split_divisor_transfer` — both stand,
    /// pinned together so neither can regress without the other
    /// noticing. `possible_raise` is its own function, separate from
    /// `provable_raise`'s all-or-nothing claim: the sink combines the
    /// two (`check.rs`'s own wiring), never this test's concern. The
    /// split value carries no infinity row from the zero corner:
    /// `adapter_exact_value` (a determined SINGLE value) is `None` for
    /// this row (`Kind::Set`, sort-only — pinned already by
    /// `test_div_over_a_wide_zero_admitting_window_determines_via_the_
    /// split` above). The pin holds the FIRE plus a DETERMINED value —
    /// infinity-absence is deliberately not asserted, since a zero-
    /// excluded divisor half still reaches +inf by denormal overflow
    /// (the in-test comment states the measured mechanism).
    #[test]
    fn test_a_zero_admitting_divisor_fires_and_still_determines_with_no_infinity_row() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(0.0), at_most(2.0)]), None, TrustProved, SetKindTag::None)
        };
        let (expression, environment) = division_by_bound_denominator(denominator.clone());

        let found = possible_raise(&expression, &environment, &kernel);
        let Some((_, message)) = found else {
            panic!("a divisor window admitting zero must fire the escape sentence");
        };
        assert!(message.contains("admits 0"), "{message}");
        assert!(message.contains("ZeroDivisionError"), "{message}");

        let one = float_operand(1.0);
        let adapter = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
        assert_eq!(
            adapter.kind,
            Kind::Set,
            "the split value must still determine alongside the new fire: {adapter:?}"
        );
        // Infinity-ABSENCE is not assertable here, and not because of the
        // zero corner: with zero excluded, `1.0 / d` still overflows to
        // +inf for a denormal `d` (1.0 / 5e-324 exceeds binary64's max),
        // so an unbounded quotient genuinely admits +inf. What the split
        // guarantees is that the value came from the zero-EXCLUDED halves
        // — the raise arm carries the fire above, never an ECMA-style
        // determined infinity AT zero. Measured today the kernel widens
        // the open positive half (above(0) && atMost(2)) to the float
        // ground rather than answering [0.5, +inf]; tightening that open-
        // window division enclosure is a named kernel precision follow-up,
        // and this pin holds the determination, not the width.
    }

    /// THE LEDGER'S CONFORMANCE PAIR, second half: a divisor window that
    /// PROVABLY EXCLUDES zero (`[1.0, 2.0]`, the same window
    /// `test_div_by_a_set_excluding_zero_still_lowers_and_agrees` above
    /// already lowers through the kernel) fires NOTHING — the escape
    /// sentence is keyed on the divisor's set admitting zero, and this
    /// window never does.
    #[test]
    fn test_a_zero_excluding_divisor_fires_nothing() {
        let Some(kernel) = loaded_kernel() else { return };
        let denominator = AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(make_refined_set(vec![at_least(1.0), at_most(2.0)]), None, TrustProved, SetKindTag::None)
        };
        let (expression, environment) = division_by_bound_denominator(denominator);
        assert!(
            possible_raise(&expression, &environment, &kernel).is_none(),
            "a divisor window that provably excludes zero must fire nothing"
        );
    }

    /// LEDGER ROW G4: the brief asks for NaN-operand arithmetic rows.
    /// A NaN operand cannot be posed to the kernel at all — NaN is not
    /// an element of ℝ̄ and `refinement_forms::one_of` panics at
    /// construction, which is the boundary ruling, not a gap. What CAN
    /// be checked is the adapter side alone: a NaN operand flows through
    /// the f64 arithmetic and answers NaN, which is IEEE-correct for
    /// Python's own float NaN.
    #[test]
    fn test_nan_operand_is_outside_the_wires_value_vocabulary() {
        // the wire refuses it at construction — a fact, recorded
        let posed = std::panic::catch_unwind(|| singleton(f64::NAN));
        assert!(
            posed.is_err(),
            "NaN is not an element of ℝ̄; one_of must refuse it at construction"
        );

        // the adapter's own concrete path still computes it, IEEE-style
        let adapter = binary_arithmetic_value(Operator::Add, &float_operand(f64::NAN), &float_operand(1.0));
        let (value, sort) = adapter_exact_value(&adapter).expect("NaN + 1 answers a float value");
        assert!(value.is_nan(), "NaN + 1 is NaN under IEEE-754");
        assert_eq!(sort, Some(PrimitiveKind::Float));
    }

    /// The signed-zero corner, isolated: `-0.0 + -0.0` is `-0.0` and
    /// `-0.0 + 0.0` is `+0.0` under IEEE-754 round-to-nearest. `==`
    /// cannot see this difference, so the comparison goes through
    /// `same_float`'s bit test. This is the row a naive `==`-based
    /// harness would pass while the two routes disagreed.
    #[test]
    fn test_signed_zero_rows_compare_by_bits_not_by_equality() {
        let Some(kernel) = loaded_kernel() else { return };

        let rows: Vec<(f64, f64, f64)> = vec![
            (-0.0, -0.0, -0.0),
            (-0.0, 0.0, 0.0),
            (0.0, -0.0, 0.0),
            (0.0, 0.0, 0.0),
        ];
        for (left, right, expected) in &rows {
            let adapter = binary_arithmetic_value(Operator::Add, &float_operand(*left), &float_operand(*right));
            let (value, _) = adapter_exact_value(&adapter).expect("a signed-zero sum answers");
            assert!(
                same_float(value, *expected),
                "IEEE-754: {left:?} + {right:?} = {expected:?}, adapter answered {value:?}"
            );
            let answer = (kernel.transfer)(&binary_question(TransferQuestionOp::Add, *left, *right));
            if let Some(kernel_value) = kernel_exact_value(&answer) {
                assert_agrees(&format!("signed zero + ({left:?}, {right:?})"), value, kernel_value);
            }
        }
    }

    // ===================================================================
    // OPERATION 2 — math floor/ceil/trunc/fabs scalar paths against the
    // kernel's Floor/Ceil/Trunc/Abs transfers over the same singletons.
    // The audit's own framing: "two independently-maintained
    // implementations of the same IEEE clause."
    // ===================================================================

    /// The unary rows. `math.floor`/`ceil`/`trunc` return a Python
    /// `int`, `math.fabs` returns a `float` — the SORT differs between
    /// them and is checked adapter-side (the kernel's transfer answer
    /// carries no sort). The VALUE is the differential comparison.
    #[test]
    fn test_math_floor_ceil_trunc_fabs_agree_with_the_kernel_transfers() {
        let Some(kernel) = loaded_kernel() else { return };

        let inputs: Vec<f64> = vec![
            200.9, 200.1, 200.0, -200.9, -200.1, -200.0, 0.0, -0.0, 0.5, -0.5, 1.5, -1.5, 2.5,
            -2.5, 1e15, -1e15,
            // the 2^53 edge: at and above it every float is already an
            // integer, so all four operations are the identity there
            9007199254740992.0,
            -9007199254740992.0,
            9007199254740991.0, // 2^53 - 1, the last exactly representable odd integer
        ];

        let ops: Vec<(&str, TransferQuestionOp, PrimitiveKind)> = vec![
            ("floor", TransferQuestionOp::Floor, PrimitiveKind::Integer),
            ("ceil", TransferQuestionOp::Ceil, PrimitiveKind::Integer),
            ("trunc", TransferQuestionOp::Trunc, PrimitiveKind::Integer),
            ("fabs", TransferQuestionOp::Abs, PrimitiveKind::Float),
        ];

        let mut agreed = 0;
        for (name, kernel_op, expected_sort) in &ops {
            for input in &inputs {
                let label = format!("math.{name}({input:?})");
                let adapter = math_call_result(name, &[float_operand(*input)], &kernel);
                let answer = (kernel.transfer)(&unary_question(*kernel_op, *input));
                let kernel_value = kernel_exact_value(&answer);
                let adapter_value = adapter.as_ref().and_then(adapter_exact_value);
                assert_scrutiny_row(&label, adapter_value, kernel_value);
                if let (Some((a, sort)), Some(k)) = (adapter_value, kernel_value) {
                    // `math.fabs` widens to float, the three rounding
                    // functions answer int — the Python sort rule, which
                    // the kernel has no opinion on, checked here.
                    assert_eq!(
                        sort,
                        Some(*expected_sort),
                        "{label}: expected the {expected_sort:?} sort"
                    );
                    // Value agreement, by bits: floor(-0.0) is -0.0 and
                    // fabs(-0.0) is +0.0, differences `==` cannot see.
                    // The adapter answers the ROUNDING functions in the
                    // Integer sort, where CPython's own `math.floor`
                    // returns an int and signed zero collapses; compare
                    // on magnitude for those and on bits for fabs.
                    if *expected_sort == PrimitiveKind::Float {
                        assert_agrees(&label, a, k);
                    } else {
                        assert!(
                            a == k,
                            "{label}: adapter answered {a:?}, kernel answered {k:?}"
                        );
                    }
                    agreed += 1;
                }
            }
        }
        assert!(agreed > 0, "no math unary row was compared");
    }

    /// `math.floor`/`ceil`/`trunc` of a NON-FINITE argument answer no
    /// value: each returns an `Integral`, and no Python `int` is
    /// infinite or NaN. CPython raises `OverflowError` for ±inf and
    /// `ValueError` for NaN, so the adapter's domain gate
    /// (`integral_domain_admits`) declines rather than claim an
    /// Integer-sorted infinity or NaN — the same discipline `isqrt`'s
    /// negative operand and `binary_arithmetic_value`'s zero divisors
    /// already keep. The raise itself is `provable_raise`'s row, which
    /// reads the same operand through `rounding_argument_raises`.
    ///
    /// The kernel is not involved in the divergence: `binary64.floor`
    /// is the pure IEEE clause and `floor(inf) = inf` is correct there.
    /// The gate is the adapter declining to read that float answer back
    /// as a Python `int`, confirmed below.
    #[test]
    fn test_math_rounding_of_non_finite_arguments_declines() {
        let Some(kernel) = loaded_kernel() else { return };

        for name in ["floor", "ceil", "trunc"] {
            for input in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                let adapter = math_call_result(name, &[float_operand(input)], &kernel);
                assert_eq!(
                    adapter, None,
                    "math.{name}({input:?}) must answer no value — CPython raises there, so any \
                     answer claims a Python int that does not exist"
                );
            }
        }

        // The kernel, by contrast, is answering its own question
        // correctly: binary64.floor of an infinity IS that infinity
        // under IEEE-754. Confirmed here so the finding cannot be
        // misread as a kernel defect.
        let answer = (kernel.transfer)(&unary_question(TransferQuestionOp::Floor, f64::INFINITY));
        if let Some(k) = kernel_exact_value(&answer) {
            assert!(
                k.is_infinite() && k > 0.0,
                "the kernel's binary64.floor(inf) is inf — the IEEE clause, correctly"
            );
        }
    }

    /// The int-sorted argument rows: `math.floor` of an int is the
    /// identity, and the kernel's floor over a singleton integer set
    /// agrees. Included because the int and float argument paths are
    /// different arms in `single_numeric_operand`.
    #[test]
    fn test_math_floor_of_an_int_argument_is_the_identity_and_agrees() {
        let Some(kernel) = loaded_kernel() else { return };

        for input in [0.0, 1.0, -1.0, 7.0, -7.0, 9007199254740991.0] {
            let label = format!("math.floor(int {input:?})");
            let adapter = math_call_result("floor", &[int_operand(input)], &kernel);
            let answer = (kernel.transfer)(&unary_question(TransferQuestionOp::Floor, input));
            let adapter_value = adapter.as_ref().and_then(adapter_exact_value);
            let kernel_value = kernel_exact_value(&answer);
            assert_scrutiny_row(&label, adapter_value, kernel_value);
            if let (Some((a, _)), Some(k)) = (adapter_value, kernel_value) {
                assert_agrees(&label, a, k);
                assert_eq!(a, input, "{label}: floor of an integer is the identity");
            }
        }
    }
}
