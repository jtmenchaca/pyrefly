/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The mirror `kernel_interface.rs:181` names: RefinedPy's own
//! `join_known` (`refined_domain::lattice_operations`), held to the
//! kernel's proved `join_state` entry, over every pair of a hand-picked
//! scalar knowledge row. Exact by `join_exact`
//! (`set_functions/known_state.lean`); the two routes may spell the
//! same set two ways, so sets are compared by mutual subset — the
//! Go twin's `lattice_conformance_test.go` convention, ported here.
//!
//! Two rows here have no Go twin: `Integer`- and `Float`-tagged values.
//! `join_known`'s Rust port carries tag-preservation arms Go's checker
//! lacks (same-sort keeps the Python `int`/`float` tag through a join;
//! a mixed or plain-`Number` join loses it — `PYREFLY-NUMERIC-B3-B4.md`).
//! The kernel has no notion of that tag at all — `KnownStateWire` carries
//! a `RefinedSet` and three flags, nothing sort-specific — so the kernel
//! comparison for those two rows checks the SET content only, and the
//! Rust-side tag-preservation rule is checked separately, by the adapter,
//! against `join_known`'s own documented rule.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use refined_domain::abstract_value::{
        known_set, known_values, nan_value, possibly_absent, possibly_nan, undef, unknown,
        AbsentFlavor, AbstractValue, Kind, PrimitiveKind, SetKindTag,
    };
    use refined_domain::lattice_operations::{join_known, set_of_known};
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
    use refined_kernel::kernel_interface::{KnownStateWire, RefinedTSKernel};
    use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set, one_of};

    /// `loaded_kernel` mirrors `cross_module.rs`'s own test helper
    /// exactly: the dylib-absence convention every kernel-touching test
    /// in this crate follows — a missing artifact prints to stderr and
    /// the caller returns early, never failing the run.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// `empty_set` is `EMPTY` in the TS source and the Go twin's
    /// `emptySet`.
    fn empty_set() -> refined_sets::refinement_forms::RefinedSet {
        make_refined_set(vec![one_of(&[])])
    }

    /// `state_of_known` is `stateOfKnown` in the TS source (the Go
    /// twin's own `stateOfKnown`): the kernel state a scalar knowledge
    /// value denotes, or `None` where the knowledge leaves the scalar
    /// world (objects, sorts, sequences) — no row here reaches that
    /// case, but the signature stays total, matching the Go port.
    fn state_of_known(k: &AbstractValue) -> Option<KnownStateWire> {
        match k.kind {
            Kind::Unknown => Some(KnownStateWire {
                top: true,
                set: make_refined_set(vec![]),
                undef: false,
                null: false,
                nan: false,
                thrown: false,
            }),
            Kind::Undef => {
                // mirrors the Go twin's own note: every exact Kind::Undef
                // site means exactly the undefined value, never null, so
                // the wire claims Undef alone.
                Some(KnownStateWire {
                    top: false,
                    set: empty_set(),
                    undef: true,
                    null: false,
                    nan: false,
                    thrown: false,
                })
            }
            Kind::NaN => Some(KnownStateWire {
                top: false,
                set: empty_set(),
                undef: false,
                null: false,
                nan: true,
                thrown: false,
            }),
            Kind::PossiblyUndefined => {
                let inner = state_of_known(k.inner.as_deref().expect("possiblyUndefined carries Inner"))?;
                if inner.top {
                    return Some(KnownStateWire {
                        top: true,
                        set: make_refined_set(vec![]),
                        undef: false,
                        null: false,
                        nan: false,
                        thrown: false,
                    });
                }
                // mirrors the Go twin's own note: a flavored wrapper's own
                // absent side admits exactly the flag its flavor names;
                // AbsentFlavor::Conflated (the zero value) keeps sending
                // both, unchanged.
                let mut inner = inner;
                match k.absent_side {
                    AbsentFlavor::UndefOnly => inner.undef = true,
                    AbsentFlavor::NullOnly => inner.null = true,
                    AbsentFlavor::Conflated => {
                        inner.undef = true;
                        inner.null = true;
                    }
                }
                Some(inner)
            }
            Kind::PossiblyNaN => {
                let inner = state_of_known(k.inner.as_deref().expect("possiblyNaN carries Inner"))?;
                if inner.top {
                    return Some(KnownStateWire {
                        top: true,
                        set: make_refined_set(vec![]),
                        undef: false,
                        null: false,
                        nan: false,
                        thrown: false,
                    });
                }
                let mut inner = inner;
                inner.nan = true;
                Some(inner)
            }
            Kind::Values | Kind::Set => {
                if k.kind == Kind::Values && k.kind_tag != Some(PrimitiveKind::Number) {
                    // the Integer/Float-tagged rows: the kernel's own wire
                    // carries no sort tag at all, so a state comparison here
                    // reads the SET only — the tag itself is checked
                    // adapter-side by test_join_known_preserves_the_python_
                    // numeric_tag_on_matching_sorts below, against
                    // join_known's own documented rule, never against the
                    // kernel.
                    let set = set_of_known(k)?;
                    return Some(KnownStateWire {
                        top: false,
                        set,
                        undef: false,
                        null: false,
                        nan: false,
                        thrown: false,
                    });
                }
                if k.kind == Kind::Set && k.set_kind_tag != SetKindTag::None {
                    return None;
                }
                let set = set_of_known(k)?;
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

    /// `same_state` is `sameState` in the TS source (the Go twin's own
    /// `sameState`): both top, or equal flags and mutually contained
    /// sets.
    ///
    /// `scalar_subset` speaks only to scalar-shaped sets (`scalarB`,
    /// `set_functions/emptiness.lean:46`, needs a non-empty refinements
    /// list) and refuses — panics inside the kernel closure — on a set
    /// outside that shape. That refusal is caught here, the same
    /// `catch_unwind`/`AssertUnwindSafe` idiom `assignability.rs`'s
    /// containment ask already holds every kernel closure to, rather
    /// than left to crash the test process. On a refusal, the fallback
    /// is the encode-level check: the two wire sets' own `PartialEq`
    /// (`RefinedSet` derives it, `refinement_forms.rs`). Equal spellings
    /// still agree; unequal spellings with a refused subset ask carry no
    /// proof either way, so this FAILS naming the set the subset decider
    /// refused rather than silently reporting disagreement — the row
    /// gets pinned on the next run instead of the panic hiding it.
    fn same_state(kernel: &RefinedTSKernel, a: &KnownStateWire, b: &KnownStateWire) -> bool {
        if a.top || b.top {
            return a.top && b.top;
        }
        if a.undef != b.undef || a.null != b.null || a.nan != b.nan {
            return false;
        }
        let a_subset_b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (kernel.scalar_subset)(&a.set, &b.set)
        }));
        let b_subset_a = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (kernel.scalar_subset)(&b.set, &a.set)
        }));
        match (a_subset_b, b_subset_a) {
            (Ok(a_in_b), Ok(b_in_a)) => a_in_b && b_in_a,
            _ if a.set == b.set => true,
            (refused_a, refused_b) => panic!(
                "same_state: scalar_subset refused a set shape it does not decide and the two \
                 wire sets disagree by spelling — a={:?} (subset refused: {}), b={:?} (subset \
                 refused: {})",
                a.set,
                refused_a.is_err(),
                b.set,
                refused_b.is_err()
            ),
        }
    }

    /// The 12 hand-picked scalar rows the Go twin carries, ported
    /// faithfully, plus two Rust-only rows (Integer-tagged and
    /// Float-tagged values — see the module doc).
    fn rows() -> Vec<AbstractValue> {
        vec![
            known_values(vec![1.0], PrimitiveKind::Number, TrustProved),
            known_values(vec![2.0], PrimitiveKind::Number, TrustProved),
            known_values(vec![0.0], PrimitiveKind::Number, TrustProved),
            known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None),
            known_set(
                make_refined_set(vec![at_most(10.0), integer()]),
                None,
                TrustProved,
                SetKindTag::None,
            ),
            undef(),
            nan_value(),
            possibly_absent(
                known_values(vec![1.0], PrimitiveKind::Number, TrustProved),
                AbsentFlavor::Conflated,
                None,
                false,
            ),
            possibly_absent(
                known_set(make_refined_set(vec![at_least(5.0)]), None, TrustProved, SetKindTag::None),
                AbsentFlavor::Conflated,
                None,
                false,
            ),
            possibly_nan(known_set(
                make_refined_set(vec![at_least(0.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )),
            possibly_absent(
                possibly_nan(known_set(make_refined_set(vec![integer()]), None, TrustProved, SetKindTag::None)),
                AbsentFlavor::Conflated,
                None,
                false,
            ),
            unknown(),
            // Rust-only: same-sort Integer join keeps the Integer tag
            // (join_known's own tag-preservation arm, no Go twin).
            known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![4.0], PrimitiveKind::Integer, TrustProved),
            // Rust-only: same-sort Float join keeps the Float tag.
            known_values(vec![1.5], PrimitiveKind::Float, TrustProved),
            known_values(vec![2.5], PrimitiveKind::Float, TrustProved),
        ]
    }

    /// The mirror `kernel_interface.rs:181` names: `join_known` agrees
    /// with the kernel's proved `join_state` on every scalar pair.
    #[test]
    fn test_join_known_agrees_with_the_kernels_proved_join_on_every_scalar_pair() {
        let Some(kernel) = loaded_kernel() else { return };

        let rows = rows();
        let mut compared = 0;
        for a in &rows {
            for b in &rows {
                let sa = state_of_known(a).expect("state_of_known(a) = None, want Some");
                let sb = state_of_known(b).expect("state_of_known(b) = None, want Some");
                let joined_known = join_known(a.clone(), b.clone());
                let joined =
                    state_of_known(&joined_known).expect("state_of_known(join_known(a, b)) = None, want Some");
                let kernel_joined = (kernel.join_state)(&sa, &sb);
                assert!(
                    same_state(&kernel, &joined, &kernel_joined),
                    "same_state(joined, kernel_joined) = false, want true (a={a:?}, b={b:?})"
                );
                compared += 1;
            }
        }
        assert_eq!(compared, rows.len() * rows.len());
    }

    /// The tag-preservation rule the Rust port carries and Go's checker
    /// does not (`lattice_operations.rs`'s own doc on its Integer/Float
    /// arms): same-sort Integer joined with Integer keeps the Integer
    /// tag, same-sort Float joined with Float keeps the Float tag. The
    /// kernel has no opinion on this — it is checked here, adapter-side,
    /// against `join_known`'s own documented rule, not against the
    /// kernel's wire (which carries no sort tag at all).
    #[test]
    fn test_join_known_preserves_the_python_numeric_tag_on_matching_sorts() {
        let joined_int = join_known(
            known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![4.0], PrimitiveKind::Integer, TrustProved),
        );
        assert_eq!(joined_int.kind, Kind::Values);
        assert_eq!(joined_int.kind_tag, Some(PrimitiveKind::Integer));

        let joined_float = join_known(
            known_values(vec![1.5], PrimitiveKind::Float, TrustProved),
            known_values(vec![2.5], PrimitiveKind::Float, TrustProved),
        );
        assert_eq!(joined_float.kind, Kind::Values);
        assert_eq!(joined_float.kind_tag, Some(PrimitiveKind::Float));

        // a MIXED Integer/Float join falls through to the untagged
        // numeric-set path and loses the tag — Integer ⊔ Float = Number.
        let joined_mixed = join_known(
            known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![2.5], PrimitiveKind::Float, TrustProved),
        );
        assert_ne!(joined_mixed.kind_tag, Some(PrimitiveKind::Integer));
        assert_ne!(joined_mixed.kind_tag, Some(PrimitiveKind::Float));
    }
}
