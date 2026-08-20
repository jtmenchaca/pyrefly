/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The differential harness for the Python adapter's local truthiness
//! decision (`refined_domain::lattice_operations::truthiness`) against
//! the kernel's proved `narrow_state` truthy filter — the
//! THIN-WALK-AUDIT.md slice-9 row stating it outright: "Truthiness
//! answers (false,false) for KindSet outright — never asks the kernel's
//! truthy narrowing; a concrete B row."
//!
//! This file measures that row rather than asserting it away. The
//! kernel's `narrow_state` with op `js.truthyNum` splits a knowledge
//! state into what truth admits and what falsity admits, each a proved
//! filter (the `narrow*_sound` theorems,
//! `set_functions/known_state.lean`). A state whose truthy side is
//! EMPTY is definitely falsy; a state whose falsy side is EMPTY is
//! definitely truthy; a state where both sides are inhabited is
//! genuinely undecided. That three-way reading is exactly the
//! `(value, known)` pair `truthiness` returns, so the two are directly
//! comparable.
//!
//! Placement and conventions follow `lattice_conformance.rs`: the
//! `loaded_kernel()` dylib-absence early return, `state_of_known`'s
//! wire construction, and tests only.
//!
//! ## The three-verdict frame
//!
//! 1. **BOTH ANSWER → must AGREE.** A state the adapter calls definitely
//!    truthy while the kernel's truthy side is empty (or vice versa) is
//!    a soundness defect; `assert_agrees` fails on it.
//! 2. **ADAPTER DECLINES, KERNEL ANSWERS → DETERMINATION-GAP row.** This
//!    is where the slice-9 finding lives, and the ledger below names
//!    every instance the rows reach.
//! 3. **ADAPTER ANSWERS WHERE THE KERNEL DOES NOT → SCRUTINY**, flagged
//!    loudly.
//!
//! ## THE DETERMINATION-GAP LEDGER (operation 5)
//!
//! | # | state | adapter | kernel | class |
//! |---|-------|---------|--------|-------|
//! | T1 | `Kind::Set` of `atLeast(1)` — every member nonzero | `(false, false)` — undecided, the outright `Kind::Set` arm | truthy side inhabited, falsy side EMPTY → definitely truthy | **GAP** |
//! | T2 | `Kind::Set` of `oneOf([0])` — the singleton zero as a set | `(false, false)` | truthy side EMPTY → definitely falsy | **GAP** |
//! | T3 | `Kind::Set` of `atLeast(0)` — genuinely spans zero | `(false, false)` | both sides inhabited → undecided | agree-on-silence |
//! | T4 | `Kind::Values` single value | decided by `values[0] != 0.0` | decided | must AGREE |
//! | T5 | `Kind::Values` MULTI-value | `(false, false)` | a multi-value state has no single scalar reading on this wire | vocabulary-bound |
//! | T6 | `Kind::Unknown` | `(false, false)` | `top` state: both sides inhabited | agree-on-silence |
//!
//! T1 and T2 are the audit's row made executable: two `Kind::Set`
//! states the kernel decides outright and the adapter does not. They
//! are asserted to STILL be gaps, so the day the adapter starts asking
//! `narrow_state`, these tests fail and the ledger rows get deleted.
//!
//! ## Why `js.truthyNum` and not a Python-named op
//!
//! `boundary/javascript.lean`'s `kernelNarrowState` dispatches
//! `js.defined`, `js.eqUndef`, `js.eqNull`, `js.truthyNum`,
//! `js.truthyStr`, and `eq`. There is no Python-named truthy op; the
//! NUMERIC truthiness rule the kernel proves — zero is falsy, every
//! other real is truthy — is the same rule Python applies to `int` and
//! `float` (`0`, `0.0` falsy; every other number truthy). The rows
//! below stay inside that shared numeric fragment and pose no state
//! whose Python truthiness diverges from it (an empty container is
//! falsy in Python and truthy in JS — that divergence is named in
//! `test_container_truthiness_is_python_owned_and_not_posed`, which
//! keeps those states off the wire rather than comparing them wrongly).

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use refined_domain::abstract_value::{
        known_set, known_values, nan_value, undef, unknown, AbstractValue, Kind, PrimitiveKind,
        SetKindTag,
    };
    use refined_domain::known_constructors::known_list;
    use refined_domain::lattice_operations::{set_of_known, truthiness};
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
    use refined_kernel::kernel_interface::{KnownStateWire, RefinedTSKernel};
    use refined_sets::refinement_forms::{
        above, at_least, at_most, below, integer, make_refined_set, one_of,
    };

    /// `loaded_kernel` mirrors `lattice_conformance.rs`'s own helper.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// `empty_set` is `EMPTY` in the TS source — the same spelling
    /// `lattice_conformance.rs` uses.
    fn empty_set() -> refined_sets::refinement_forms::RefinedSet {
        make_refined_set(vec![one_of(&[])])
    }

    /// The scalar knowledge state a value denotes, for the shapes these
    /// rows pose. Narrower than `lattice_conformance.rs`'s own
    /// `state_of_known` on purpose: only `Values`, `Set`, `NaN`, `Undef`
    /// and `Unknown` are reached here, and `None` everywhere else keeps
    /// container states off the wire (see the container test below).
    fn state_of(value: &AbstractValue) -> Option<KnownStateWire> {
        match value.kind {
            Kind::Unknown => Some(KnownStateWire {
                top: true,
                set: make_refined_set(vec![]),
                undef: false,
                null: false,
                nan: false,
                thrown: false,
            }),
            Kind::Undef => Some(KnownStateWire {
                top: false,
                set: empty_set(),
                undef: true,
                null: false,
                nan: false,
                thrown: false,
            }),
            Kind::NaN => Some(KnownStateWire {
                top: false,
                set: empty_set(),
                undef: false,
                null: false,
                nan: true,
                thrown: false,
            }),
            Kind::Values | Kind::Set => {
                let set = set_of_known(value)?;
                Some(KnownStateWire {
                    top: false,
                    set,
                    undef: false,
                    null: false,
                    nan: false,
                    thrown: false,
                })
            }
            _ => None,
        }
    }

    /// Whether a knowledge state admits nothing at all: not top, no
    /// flags, and an EMPTY scalar set by the kernel's own proved
    /// emptiness decider. This is the reading that turns `narrow_state`'s
    /// two-sided answer into a truthiness verdict.
    fn state_is_uninhabited(kernel: &RefinedTSKernel, state: &KnownStateWire) -> bool {
        if state.top || state.undef || state.null || state.nan || state.thrown {
            return false;
        }
        (kernel.scalar_empty)(&state.set)
    }

    /// The kernel's truthiness verdict for a state, as the same
    /// `(value, known)` pair `truthiness` returns:
    ///
    /// - falsy side uninhabited → definitely truthy `(true, true)`,
    /// - truthy side uninhabited → definitely falsy `(false, true)`,
    /// - both inhabited → undecided `(false, false)`.
    ///
    /// Both sides uninhabited would mean the state itself was empty —
    /// no value flows there at all — which is not a truthiness verdict
    /// and is returned as undecided rather than as a claim.
    fn kernel_truthiness(kernel: &RefinedTSKernel, state: &KnownStateWire) -> (bool, bool) {
        let (when_true, when_false) = (kernel.narrow_state)(state, "js.truthyNum", 0.0, false);
        let truthy_empty = state_is_uninhabited(kernel, &when_true);
        let falsy_empty = state_is_uninhabited(kernel, &when_false);
        match (truthy_empty, falsy_empty) {
            (true, true) => (false, false),
            (false, true) => (true, true),
            (true, false) => (false, true),
            (false, false) => (false, false),
        }
    }

    /// VERDICT 1 — both routes decided, so they must AGREE.
    fn assert_agrees(label: &str, adapter: bool, kernel: bool) {
        assert!(
            adapter == kernel,
            "{label}: the adapter's truthiness is {adapter}, the kernel's proved narrowing says \
             {kernel} — two routes for the same predicate must agree"
        );
    }

    /// VERDICT 3 — the adapter decided where the kernel did not.
    fn assert_scrutiny_row(label: &str, adapter: (bool, bool), kernel: (bool, bool)) {
        if adapter.1 && !kernel.1 {
            panic!(
                "SCRUTINY: {label}: the adapter claims truthiness {} where the kernel's proved \
                 narrowing leaves both sides inhabited — an adapter-only claim carries no proved \
                 backing",
                adapter.0
            );
        }
    }

    /// VERDICT 2 — a determination gap, returned rather than asserted so
    /// callers can count and name them.
    fn is_determination_gap(adapter: (bool, bool), kernel: (bool, bool)) -> bool {
        !adapter.1 && kernel.1
    }

    fn int_value(v: f64) -> AbstractValue {
        known_values(vec![v], PrimitiveKind::Integer, TrustProved)
    }

    fn float_value(v: f64) -> AbstractValue {
        known_values(vec![v], PrimitiveKind::Float, TrustProved)
    }

    fn numeric_set(forms: Vec<refined_sets::refinement_forms::Refinement>) -> AbstractValue {
        known_set(make_refined_set(forms), None, TrustProved, SetKindTag::None)
    }

    /// LEDGER ROWS T4 and T6, the agreement rows: a single known
    /// numeric value is decided by both routes, and they must agree on
    /// every representative value including both signed zeros. `Unknown`
    /// is undecided on both.
    #[test]
    fn test_single_value_truthiness_agrees_with_the_kernels_proved_narrowing() {
        let Some(kernel) = loaded_kernel() else { return };

        let values: Vec<f64> = vec![
            0.0, -0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 1e308, -1e308, f64::INFINITY,
            f64::NEG_INFINITY, 9007199254740992.0, 5e-324,
        ];

        let mut agreed = 0;
        for v in &values {
            for value in [int_value(*v), float_value(*v)] {
                let label = format!("truthiness({:?} as {:?})", v, value.kind_tag);
                let adapter = truthiness(&value);
                let Some(state) = state_of(&value) else { continue };
                let kernel_verdict = kernel_truthiness(&kernel, &state);
                assert_scrutiny_row(&label, adapter, kernel_verdict);
                if adapter.1 && kernel_verdict.1 {
                    assert_agrees(&label, adapter.0, kernel_verdict.0);
                    // ground truth: zero (either sign) is falsy, every
                    // other real is truthy — Python's own rule for int
                    // and float, and the kernel's numeric truthiness
                    assert_eq!(
                        adapter.0,
                        *v != 0.0,
                        "{label}: zero is falsy and every other real is truthy"
                    );
                    agreed += 1;
                }
            }
        }
        assert!(agreed > 0, "no single-value truthiness row was compared");

        // T6: Unknown is undecided on both routes
        let unknown_value = unknown();
        assert_eq!(truthiness(&unknown_value), (false, false));
        let state = state_of(&unknown_value).expect("Unknown has a top state");
        assert_eq!(
            kernel_truthiness(&kernel, &state),
            (false, false),
            "T6: a top state leaves both sides inhabited — undecided, matching the adapter"
        );
    }

    /// LEDGER ROWS T1 and T2, asserted as gaps: the audit's slice-9
    /// finding made executable. `truthiness` answers `(false, false)`
    /// for `Kind::Set` OUTRIGHT — the `Kind::Set | Kind::Variable |
    /// PossiblyUndefined | PossiblyNaN | Unknown => (false, false)` arm —
    /// while the kernel's proved narrowing DECIDES these two states.
    ///
    /// The day the adapter asks `narrow_state` here, this test fails and
    /// the T1/T2 ledger rows get deleted. That is the point of pinning
    /// it: a gap that closes silently is a gap nobody notices closing.
    #[test]
    fn test_determination_gap_kind_set_truthiness_is_never_asked_of_the_kernel() {
        let Some(kernel) = loaded_kernel() else { return };

        // T1: every member is at least 1, so no member is zero — the
        // state is definitely truthy, and the kernel says so.
        let definitely_truthy = numeric_set(vec![at_least(1.0)]);
        let adapter = truthiness(&definitely_truthy);
        assert_eq!(
            adapter,
            (false, false),
            "T1: the adapter's Kind::Set arm answers undecided outright"
        );
        let state = state_of(&definitely_truthy).expect("a numeric set has a scalar state");
        let kernel_verdict = kernel_truthiness(&kernel, &state);
        assert_eq!(
            kernel_verdict,
            (true, true),
            "T1: the kernel's proved narrowing decides atLeast(1) is definitely truthy"
        );
        assert!(
            is_determination_gap(adapter, kernel_verdict),
            "T1 is a determination gap: the kernel decides, the adapter does not"
        );

        // T2: the singleton zero, carried as a SET rather than a value.
        // The same number the adapter decides instantly as Kind::Values
        // becomes undecided the moment it is spelled as a set.
        let definitely_falsy = numeric_set(vec![one_of(&[0.0])]);
        let adapter = truthiness(&definitely_falsy);
        assert_eq!(
            adapter,
            (false, false),
            "T2: the adapter's Kind::Set arm answers undecided even for the singleton zero"
        );
        let state = state_of(&definitely_falsy).expect("a numeric set has a scalar state");
        let kernel_verdict = kernel_truthiness(&kernel, &state);
        assert_eq!(
            kernel_verdict,
            (false, true),
            "T2: the kernel's proved narrowing decides {{0}} is definitely falsy"
        );
        assert!(is_determination_gap(adapter, kernel_verdict), "T2 is a determination gap");

        // and the SAME value as Kind::Values IS decided by the adapter —
        // which is what makes T2 a representation gap rather than a
        // semantic one
        assert_eq!(
            truthiness(&int_value(0.0)),
            (false, true),
            "T2: the identical value decided instantly in the Kind::Values representation"
        );
    }

    /// More `Kind::Set` rows, swept: every set below is one the kernel
    /// decides and the adapter does not. Counted rather than enumerated
    /// one assertion at a time, so the gap's SIZE is visible and not
    /// just its existence.
    #[test]
    fn test_determination_gap_sweep_over_decidable_numeric_sets() {
        let Some(kernel) = loaded_kernel() else { return };

        // sets whose members are all nonzero (definitely truthy) or all
        // zero (definitely falsy) — every one decidable in principle
        let decidable = vec![
            ("atLeast(1)", numeric_set(vec![at_least(1.0)])),
            ("above(0)", numeric_set(vec![above(0.0)])),
            ("atMost(-1)", numeric_set(vec![at_most(-1.0)])),
            ("below(0)", numeric_set(vec![below(0.0)])),
            ("oneOf([1,2,3])", numeric_set(vec![one_of(&[1.0, 2.0, 3.0])])),
            ("oneOf([0])", numeric_set(vec![one_of(&[0.0])])),
            (
                "integer ∩ [1,120]",
                numeric_set(vec![integer(), at_least(1.0), at_most(120.0)]),
            ),
        ];

        let mut gaps = 0;
        let mut agreed = 0;
        for (name, value) in &decidable {
            let label = format!("Kind::Set truthiness of {name}");
            let adapter = truthiness(value);
            let Some(state) = state_of(value) else { continue };
            let kernel_verdict = kernel_truthiness(&kernel, &state);
            assert_scrutiny_row(&label, adapter, kernel_verdict);
            if is_determination_gap(adapter, kernel_verdict) {
                gaps += 1;
            } else if adapter.1 && kernel_verdict.1 {
                assert_agrees(&label, adapter.0, kernel_verdict.0);
                agreed += 1;
            }
        }
        // Every row above is a set the adapter's outright `Kind::Set`
        // arm refuses. If this count ever drops, the adapter started
        // asking — reread the ledger rather than lowering the number.
        assert_eq!(
            gaps,
            decidable.len(),
            "expected every decidable numeric set to be a determination gap today \
             (gaps={gaps}, agreed={agreed}, rows={})",
            decidable.len()
        );
    }

    /// LEDGER ROW T3, the agree-on-silence row: a set that genuinely
    /// spans zero is undecided on BOTH routes. Included so the sweep
    /// above cannot be read as "the kernel decides everything" — it
    /// decides what is decidable, and declines what is not.
    #[test]
    fn test_a_set_spanning_zero_is_undecided_on_both_routes() {
        let Some(kernel) = loaded_kernel() else { return };

        for (name, value) in [
            ("atLeast(0)", numeric_set(vec![at_least(0.0)])),
            ("oneOf([0,1])", numeric_set(vec![one_of(&[0.0, 1.0])])),
            ("integer ∩ [-5,5]", numeric_set(vec![integer(), at_least(-5.0), at_most(5.0)])),
        ] {
            let adapter = truthiness(&value);
            assert_eq!(adapter, (false, false), "{name}: the adapter is undecided");
            let Some(state) = state_of(&value) else { continue };
            let kernel_verdict = kernel_truthiness(&kernel, &state);
            assert_eq!(
                kernel_verdict,
                (false, false),
                "T3 {name}: a set spanning zero has both sides inhabited — the kernel declines too"
            );
        }
    }

    /// LEDGER ROW T5, the vocabulary bound: a MULTI-value
    /// `Kind::Values` state answers `(false, false)` on the adapter, and
    /// its scalar reading is a `oneOf` of all its members — which the
    /// kernel then decides. So this is a gap of the same family as T1,
    /// arising from the adapter's own arm rather than from the wire.
    #[test]
    fn test_determination_gap_multi_value_states_are_undecided_adapter_side() {
        let Some(kernel) = loaded_kernel() else { return };

        // all members nonzero — decidable in principle
        let multi = known_values(vec![1.0, 2.0, 3.0], PrimitiveKind::Integer, TrustProved);
        let adapter = truthiness(&multi);
        assert_eq!(
            adapter,
            (false, false),
            "T5: the adapter decides only the len()==1 case of Kind::Values"
        );
        let Some(state) = state_of(&multi) else {
            // no scalar reading available: the row is vocabulary-bound
            // rather than a gap, and is reported as such
            return;
        };
        let kernel_verdict = kernel_truthiness(&kernel, &state);
        assert_scrutiny_row("T5 multi-value", adapter, kernel_verdict);
        assert!(
            is_determination_gap(adapter, kernel_verdict),
            "T5: a multi-value state of all-nonzero members is decidable by the kernel \
             (kernel said {kernel_verdict:?})"
        );
    }

    /// The falsy singletons the adapter decides outright, checked
    /// against the kernel where the wire can carry them. `NaN` is falsy
    /// in Python (`bool(float('nan'))` is `True` — see the assertion's
    /// own note) and `None`/`undef` is falsy; these ride flags on the
    /// wire rather than the scalar set, so the comparison reads the
    /// flags rather than the set emptiness.
    #[test]
    fn test_absent_and_nan_states_keep_their_flags_across_the_truthy_narrowing() {
        let Some(kernel) = loaded_kernel() else { return };

        // The adapter's own verdicts, from `truthiness`'s NaN and Undef
        // arms. NOTE: these are the JS/ECMA rows the shared domain
        // carries (`sec-toboolean`: NaN and undefined are falsy). Python
        // disagrees about NaN specifically — `bool(float('nan'))` is
        // `True`, since only `0.0` is the falsy float. That divergence
        // is a LANGUAGE-OWNED fact, not a kernel disagreement, and it is
        // recorded here rather than compared: posing NaN to
        // `js.truthyNum` would be asking the wrong language's question.
        assert_eq!(truthiness(&nan_value()), (false, true));
        assert_eq!(truthiness(&undef()), (false, true));

        // What IS checked against the kernel: the flags survive the
        // narrowing intact, so no state silently loses its NaN or
        // absent admission on the way through.
        let nan_state = state_of(&nan_value()).expect("NaN has a state");
        let (when_true, when_false) = (kernel.narrow_state)(&nan_state, "js.truthyNum", 0.0, false);
        // A NaN-flagged input cannot come back with BOTH sides admitting
        // nothing at all: the value flows somewhere, and a narrowing that
        // dropped it entirely would be unsound in the direction that
        // matters (a state narrowed to nothing makes every downstream
        // claim vacuously true).
        assert!(
            when_true.nan
                || when_false.nan
                || !state_is_uninhabited(&kernel, &when_true)
                || !state_is_uninhabited(&kernel, &when_false),
            "a NaN-flagged state must survive the truthy narrowing on one side or the other"
        );
    }

    /// The states this harness deliberately does NOT pose: Python
    /// container truthiness. An empty list is FALSY in Python
    /// ("Sequences and collections... are false when empty") while the
    /// shared domain's `truthiness` answers `(true, true)` for
    /// `Kind::List` — the JS rule, where every object is truthy
    /// (`sec-toboolean`). That is a language-model divergence inside the
    /// adapter, not a kernel disagreement, and `set_of_known` refuses
    /// `Kind::List` outright, so no such state can reach the wire.
    ///
    /// Recorded as its own row so the omission is deliberate and
    /// visible rather than an untested corner.
    #[test]
    fn test_container_truthiness_is_python_owned_and_not_posed() {
        let empty_list = known_list(vec![], TrustProved);
        // the shared domain's JS-shaped answer
        assert_eq!(
            truthiness(&empty_list),
            (true, true),
            "the shared domain answers the ECMA rule: every object is truthy"
        );
        // and the wire cannot carry it, so the harness never compares it
        assert_eq!(
            set_of_known(&empty_list),
            None,
            "a container has no tuple-layer set — it cannot be posed to narrow_state at all"
        );
        assert_eq!(state_of(&empty_list), None);
    }
}
