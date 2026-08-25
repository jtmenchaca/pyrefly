//! The differential harness for the Python adapter's CONCRETE string
//! and set-membership paths against the kernel's proved sequence
//! deciders — THIN-WALK-AUDIT.md's W2 row naming this exact pair:
//! "Python: the concrete-vs-set split, wholesale — string_models.rs,
//! bytes_models.rs, collection_models.rs and the set-operation methods
//! import no kernel at all while `seq_starts_with`/`seq_ends_with`/
//! `seq_includes`/`seq_lex_lt`/`member`/`scalar_subset` sit unused".
//! The audit calls this the single largest concrete-only surface; this
//! file runs the same words through both routes and compares.
//!
//! Placement and conventions follow `lattice_conformance.rs`: the
//! `loaded_kernel()` dylib-absence early return, tests only, and no
//! edits to `string_models.rs` or `expressions.rs` — their `pub`
//! functions are consumed as they stand.
//!
//! ## The three-verdict frame
//!
//! 1. **BOTH ANSWER → must AGREE.** `assert_agrees_bool` fails on drift.
//!    Two routes deciding `"banana".startswith("ban")` differently is a
//!    defect in one of them.
//! 2. **ADAPTER DECLINES, KERNEL ANSWERS → DETERMINATION-GAP ledger row**
//!    (the table below), asserted as still-a-gap, never as a failure.
//! 3. **ADAPTER ANSWERS WHERE THE KERNEL DOES NOT → SCRUTINY**, flagged
//!    loudly by `assert_scrutiny_row`.
//!
//! ## THE DETERMINATION-GAP LEDGER (operations 3 and 4)
//!
//! | # | operation | operands | adapter | kernel | class |
//! |---|-----------|----------|---------|--------|-------|
//! | S1 | `casefold` | any non-ASCII receiver | declines (the full Unicode folding table is not modeled) | no `casefold` decider exists | agree-on-silence |
//! | S2 | `split` | empty separator | declines (CPython raises `ValueError`) | no `split` decider exists | correct decline |
//! | S3 | `index` | needle absent | declines (the miss is a raise) | `seq_includes` answers `false` | GAP — the kernel decides the PRESENCE question the adapter's raise-analysis needs |
//! | S4 | `issubset`/`issuperset` | non-numeric or undecidable elements | declines through `set_contains`'s `None` | `scalar_subset` needs a scalar set to pose at all | vocabulary-bound |
//! | S5 | `startswith`/`endswith` | tuple-of-prefixes argument (CPython allows it) | declines (one-arg exact-string row only) | `seq_starts_with` is one-word-vs-one-word | GAP on both sides |
//!
//! ## The UTF-16-vs-codepoint question the brief names
//!
//! The audit lists "the UTF-16 code-unit indexing family" as an open
//! item, so the astral-plane rows here are the probe for it. The two
//! routes model a string differently ON PAPER:
//!
//! - The adapter carries one `f64` per Unicode CODE POINT
//!   (`string_models.rs`'s own module doc: "A one-code-point-per-`f64`
//!   vector already counts code points by construction"), because
//!   Python's `len` counts code points.
//! - The kernel's word encoding, `codepoint_sets::string_tuple`, is
//!   also one element per code point, and `codepoint_sets`'s own doc
//!   excludes the surrogate range U+D800–U+DFFF from the codepoint set
//!   entirely.
//!
//! So for every string a Rust `str` can hold, the two agree by
//! construction — a Rust `str` is valid UTF-8 and cannot hold a lone
//! surrogate. `test_astral_plane_words_agree_on_all_three_predicates`
//! exercises U+1F600 and U+10000 (each ONE code point, but TWO UTF-16
//! code units) precisely to confirm the two routes count the same way
//! and neither has silently drifted to a code-unit view. That is the
//! divergence the brief asks about, and the answer this file records is
//! that it does not occur on any representable input.

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use refined_domain::abstract_value::{known_values, AbstractValue, Kind, PrimitiveKind};
    use refined_domain::known_constructors::known_list;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
    use refined_kernel::kernel_interface::RefinedTSKernel;
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::{make_refined_set, one_of, RefinedSet};

    use crate::string_models::{string_literal_value, string_method_result};

    /// `loaded_kernel` mirrors `lattice_conformance.rs`'s own helper.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// The boolean an adapter answer carries: the
    /// `known_values(vec![0.0/1.0], Boolean, TrustProved)` shape every
    /// boolean row in `string_models.rs` and `expressions.rs` builds.
    /// `None` where the adapter declined.
    fn adapter_bool(value: &Option<AbstractValue>) -> Option<bool> {
        let value = value.as_ref()?;
        if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::Boolean) {
            return None;
        }
        if value.values.len() != 1 {
            return None;
        }
        Some(value.values[0] != 0.0)
    }

    /// The integer an adapter answer carries (str.find's row).
    fn adapter_int(value: &Option<AbstractValue>) -> Option<f64> {
        let value = value.as_ref()?;
        if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::Integer) {
            return None;
        }
        if value.values.len() != 1 {
            return None;
        }
        Some(value.values[0])
    }

    /// VERDICT 1 — both routes decided, so they must AGREE.
    fn assert_agrees_bool(label: &str, adapter: bool, kernel: bool) {
        assert!(
            adapter == kernel,
            "{label}: adapter decided {adapter}, kernel decided {kernel} — two routes for the \
             same predicate must agree"
        );
    }

    /// VERDICT 3 — the adapter decided where the kernel did not. Named
    /// loudly; an adapter-only claim carries no proved backing.
    fn assert_scrutiny_row(label: &str, adapter: Option<bool>, kernel: Option<bool>) {
        if adapter.is_some() && kernel.is_none() {
            panic!(
                "SCRUTINY: {label}: the adapter decided {adapter:?} where the kernel declined — \
                 an adapter-only claim carries no proved backing"
            );
        }
    }

    /// A kernel sequence ask, guarded. The kernel's `seq_starts_with`/
    /// `seq_ends_with`/`seq_includes` PANIC where either side is not
    /// recognized as one concrete word (their own doc in
    /// `kernel_interface.rs`), and a panic is the kernel's DECLINE, not
    /// a failure of the harness — so it is caught and read as `None`,
    /// which is exactly what the three-verdict frame needs to tell a
    /// decline apart from an answer.
    fn kernel_seq_ask(
        ask: &Arc<dyn Fn(&RefinedSet, &RefinedSet) -> bool + Send + Sync>,
        receiver: &str,
        needle: &str,
    ) -> Option<bool> {
        let receiver_set = string_tuple(receiver);
        let needle_set = string_tuple(needle);
        let ask = ask.clone();
        crate::kernel_ask::ask_kernel(move || (ask)(&receiver_set, &needle_set)).ok()
    }

    /// The word pairs every string predicate row runs. Chosen for the
    /// corners the brief names, not for likely agreement:
    ///
    /// - the empty needle (a prefix/suffix/substring of every word),
    /// - the needle EQUAL to the receiver (the boundary of all three
    ///   predicates at once),
    /// - the empty receiver with a non-empty needle,
    /// - repeated-letter words where a naive scan overshoots,
    /// - a needle LONGER than the receiver,
    /// - astral-plane code points (U+1F600, U+10000 — one code point,
    ///   two UTF-16 code units each), the UTF-16-vs-codepoint probe,
    /// - a non-ASCII BMP character ('é'), where a byte-offset
    ///   implementation and a code-point one diverge.
    fn word_pairs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("banana", "ban"),
            ("banana", "ana"),
            ("banana", "na"),
            ("banana", "apple"),
            ("banana", "banana"),
            ("banana", ""),
            ("", ""),
            ("", "a"),
            ("a", "ab"),
            ("aaa", "aa"),
            ("abab", "bab"),
            ("héllo", "é"),
            ("héllo", "llo"),
            ("héllo", "h"),
            // astral plane: each of these is ONE code point and TWO
            // UTF-16 code units — the divergence probe
            ("\u{1F600}ab", "\u{1F600}"),
            ("ab\u{1F600}", "\u{1F600}"),
            ("a\u{1F600}b", "\u{1F600}b"),
            ("\u{10000}\u{10001}", "\u{10000}"),
            ("\u{10000}\u{10001}", "\u{10001}"),
            ("\u{1F600}", "\u{1F600}"),
            // a needle that is a UTF-16 code unit of the receiver but
            // NOT a code point of it: under a code-unit model this could
            // spuriously match; under a code-point model it cannot
            ("\u{1F600}", "a"),
        ]
    }

    // ===================================================================
    // OPERATION 3 — the exact-string predicates. `string_models.rs`'s
    // startswith / endswith / find rows against the kernel's
    // `seq_starts_with` / `seq_ends_with` / `seq_includes` on the same
    // words.
    // ===================================================================

    /// `str.startswith` on an exact receiver and exact prefix, against
    /// the kernel's `seq_starts_with` (str.15, exact on two concrete
    /// words by `theories/seq/starts_with.lean`'s `startsWithB`).
    #[test]
    fn test_startswith_agrees_with_seq_starts_with_on_every_word_pair() {
        let Some(kernel) = loaded_kernel() else { return };

        let mut agreed = 0;
        let mut gaps = 0;
        for (receiver, needle) in word_pairs() {
            let label = format!("startswith({receiver:?}, {needle:?})");
            let adapter = adapter_bool(&string_method_result(
                "startswith",
                &string_literal_value(receiver),
                &[string_literal_value(needle)],
            ));
            let kernel_answer = kernel_seq_ask(&kernel.seq_starts_with, receiver, needle);
            assert_scrutiny_row(&label, adapter, kernel_answer);
            match (adapter, kernel_answer) {
                (Some(a), Some(k)) => {
                    assert_agrees_bool(&label, a, k);
                    // the ground truth Rust itself computes, so a shared
                    // wrong answer on both routes still fails
                    assert_eq!(
                        a,
                        receiver.starts_with(needle),
                        "{label}: both routes agreed on {a}, but str.startswith is {}",
                        receiver.starts_with(needle)
                    );
                    agreed += 1;
                }
                (None, Some(_)) => gaps += 1,
                _ => {}
            }
        }
        assert!(agreed > 0, "no startswith row was compared: agreed={agreed}, gaps={gaps}");
    }

    /// `str.endswith` against the kernel's `seq_ends_with` (str.16).
    #[test]
    fn test_endswith_agrees_with_seq_ends_with_on_every_word_pair() {
        let Some(kernel) = loaded_kernel() else { return };

        let mut agreed = 0;
        for (receiver, needle) in word_pairs() {
            let label = format!("endswith({receiver:?}, {needle:?})");
            let adapter = adapter_bool(&string_method_result(
                "endswith",
                &string_literal_value(receiver),
                &[string_literal_value(needle)],
            ));
            let kernel_answer = kernel_seq_ask(&kernel.seq_ends_with, receiver, needle);
            assert_scrutiny_row(&label, adapter, kernel_answer);
            if let (Some(a), Some(k)) = (adapter, kernel_answer) {
                assert_agrees_bool(&label, a, k);
                assert_eq!(
                    a,
                    receiver.ends_with(needle),
                    "{label}: both routes agreed on {a}, but str.endswith is {}",
                    receiver.ends_with(needle)
                );
                agreed += 1;
            }
        }
        assert!(agreed > 0, "no endswith row was compared");
    }

    /// The INCLUDES-shaped question. Python has no `str.includes`; the
    /// shape is `needle in receiver`, and the adapter's exact route to
    /// it is `str.find` — "Return the lowest index in the string where
    /// substring sub is found... Return -1 if sub is not found." So
    /// `find(...) >= 0` IS the includes predicate, and that is what the
    /// kernel's `seq_includes` (str.17) is compared against.
    ///
    /// This comparison is the audit's point in miniature: the adapter
    /// computes a POSITION with a hand-rolled scan
    /// (`find_code_point_index`) and derives presence from it, while the
    /// kernel decides presence directly with a proved decider. The two
    /// must agree on presence for every word pair.
    #[test]
    fn test_find_presence_agrees_with_seq_includes_on_every_word_pair() {
        let Some(kernel) = loaded_kernel() else { return };

        let mut agreed = 0;
        for (receiver, needle) in word_pairs() {
            let label = format!("find(...)>=0 vs seq_includes({receiver:?}, {needle:?})");
            let found = adapter_int(&string_method_result(
                "find",
                &string_literal_value(receiver),
                &[string_literal_value(needle)],
            ));
            let adapter = found.map(|position| position >= 0.0);
            let kernel_answer = kernel_seq_ask(&kernel.seq_includes, receiver, needle);
            assert_scrutiny_row(&label, adapter, kernel_answer);
            if let (Some(a), Some(k)) = (adapter, kernel_answer) {
                assert_agrees_bool(&label, a, k);
                assert_eq!(
                    a,
                    receiver.contains(needle),
                    "{label}: both routes agreed on {a}, but the substring relation is {}",
                    receiver.contains(needle)
                );
                agreed += 1;
            }
        }
        assert!(agreed > 0, "no includes row was compared");
    }

    /// The astral-plane rows, isolated with their own assertions —
    /// the UTF-16-vs-code-point probe the module doc explains.
    ///
    /// U+1F600 is ONE Unicode code point and TWO UTF-16 code units.
    /// A code-unit-indexed implementation would report `len` 2 for the
    /// one-character string `"\u{1F600}"` and would place the `"b"` in
    /// `"\u{1F600}b"` at index 2; a code-point implementation reports
    /// `len` 1 and index 1. Both routes must show the code-point
    /// behaviour — that is Python's own `len`, and the kernel's own
    /// codepoint set.
    #[test]
    fn test_astral_plane_words_agree_on_all_three_predicates_and_count_code_points() {
        let Some(kernel) = loaded_kernel() else { return };

        // adapter side: one f64 per code point, so an astral character
        // occupies exactly one slot
        let emoji = string_literal_value("\u{1F600}");
        assert_eq!(
            emoji.values.len(),
            1,
            "an astral code point is ONE element of the code-point vector, not two code units"
        );
        assert_eq!(
            emoji.values[0], 0x1F600 as f64,
            "the element is the scalar value itself, not a surrogate half"
        );

        // kernel side: the same word encodes to a one-element tuple
        let kernel_word = string_tuple("\u{1F600}");
        assert_eq!(
            refined_sets::refinement_forms::word_of(&kernel_word),
            Some(vec![0x1F600 as f64]),
            "the kernel's word encoding is code points too, never UTF-16 code units"
        );

        // the position an astral prefix pushes a following character to
        let found = adapter_int(&string_method_result(
            "find",
            &string_literal_value("\u{1F600}b"),
            &[string_literal_value("b")],
        ));
        assert_eq!(
            found,
            Some(1.0),
            "'b' after one astral code point is at code-point index 1 (a UTF-16 view would say 2)"
        );

        // and all three predicates agree across both routes on the
        // astral rows specifically
        for (receiver, needle) in [
            ("\u{1F600}ab", "\u{1F600}"),
            ("a\u{1F600}b", "\u{1F600}b"),
            ("\u{10000}\u{10001}", "\u{10001}"),
            ("\u{1F600}", "a"),
        ] {
            let starts = adapter_bool(&string_method_result(
                "startswith",
                &string_literal_value(receiver),
                &[string_literal_value(needle)],
            ));
            if let (Some(a), Some(k)) = (starts, kernel_seq_ask(&kernel.seq_starts_with, receiver, needle)) {
                assert_agrees_bool(&format!("astral startswith({receiver:?}, {needle:?})"), a, k);
            }
            let ends = adapter_bool(&string_method_result(
                "endswith",
                &string_literal_value(receiver),
                &[string_literal_value(needle)],
            ));
            if let (Some(a), Some(k)) = (ends, kernel_seq_ask(&kernel.seq_ends_with, receiver, needle)) {
                assert_agrees_bool(&format!("astral endswith({receiver:?}, {needle:?})"), a, k);
            }
        }
    }

    /// The EMPTY-NEEDLE corner, isolated. Every string starts with,
    /// ends with, and contains the empty string — including the empty
    /// string itself. A scan that special-cases the empty needle wrongly
    /// (returning `false`, or overrunning) shows up here and nowhere
    /// else.
    #[test]
    fn test_empty_needle_holds_for_every_receiver_on_both_routes() {
        let Some(kernel) = loaded_kernel() else { return };

        for receiver in ["", "a", "banana", "héllo", "\u{1F600}"] {
            let label = format!("empty needle against {receiver:?}");
            for (name, ask) in [
                ("startswith", &kernel.seq_starts_with),
                ("endswith", &kernel.seq_ends_with),
                ("includes", &kernel.seq_includes),
            ] {
                let adapter = if name == "includes" {
                    adapter_int(&string_method_result(
                        "find",
                        &string_literal_value(receiver),
                        &[string_literal_value("")],
                    ))
                    .map(|position| position >= 0.0)
                } else {
                    adapter_bool(&string_method_result(
                        name,
                        &string_literal_value(receiver),
                        &[string_literal_value("")],
                    ))
                };
                if let Some(a) = adapter {
                    assert!(a, "{label}: {name} of the empty needle is always true");
                }
                if let (Some(a), Some(k)) = (adapter, kernel_seq_ask(ask, receiver, "")) {
                    assert_agrees_bool(&format!("{label} ({name})"), a, k);
                }
            }
            // str.find("") is 0 for every receiver, the empty string
            // being present at position 0
            assert_eq!(
                adapter_int(&string_method_result(
                    "find",
                    &string_literal_value(receiver),
                    &[string_literal_value("")]
                )),
                Some(0.0),
                "{label}: str.find of the empty needle is 0"
            );
        }
    }

    /// The NEEDLE-EQUALS-RECEIVER corner: all three predicates are true
    /// at once, on both routes. The boundary where prefix, suffix, and
    /// substring coincide.
    #[test]
    fn test_needle_equal_to_receiver_holds_all_three_predicates() {
        let Some(kernel) = loaded_kernel() else { return };

        for word in ["", "a", "banana", "héllo", "\u{1F600}\u{10000}"] {
            let label = format!("self-predicate on {word:?}");
            let starts = adapter_bool(&string_method_result(
                "startswith",
                &string_literal_value(word),
                &[string_literal_value(word)],
            ));
            let ends = adapter_bool(&string_method_result(
                "endswith",
                &string_literal_value(word),
                &[string_literal_value(word)],
            ));
            if let Some(a) = starts {
                assert!(a, "{label}: a word starts with itself");
            }
            if let Some(a) = ends {
                assert!(a, "{label}: a word ends with itself");
            }
            if let (Some(a), Some(k)) = (starts, kernel_seq_ask(&kernel.seq_starts_with, word, word)) {
                assert_agrees_bool(&format!("{label} (startswith)"), a, k);
            }
            if let (Some(a), Some(k)) = (ends, kernel_seq_ask(&kernel.seq_ends_with, word, word)) {
                assert_agrees_bool(&format!("{label} (endswith)"), a, k);
            }
        }
    }

    /// LEDGER ROW S3, asserted as a gap: `str.index` on a missing
    /// needle declines (the miss raises `ValueError`, which is
    /// `provable_raise`'s row, not a value), while the kernel's
    /// `seq_includes` DECIDES the presence question that raise-analysis
    /// rests on. The adapter reaches the same presence fact only through
    /// its own hand-rolled `find_code_point_index` scan.
    #[test]
    fn test_determination_gap_index_miss_declines_while_the_kernel_decides_presence() {
        let Some(kernel) = loaded_kernel() else { return };

        let receiver = "banana";
        let needle = "z";
        let adapter = string_method_result(
            "index",
            &string_literal_value(receiver),
            &[string_literal_value(needle)],
        );
        assert_eq!(
            adapter, None,
            "S3: str.index on a missing needle declines — the miss is a raise, not a value"
        );
        // the kernel decides the underlying presence question outright
        assert_eq!(
            kernel_seq_ask(&kernel.seq_includes, receiver, needle),
            Some(false),
            "S3: the kernel's seq_includes decides the presence the adapter's raise-analysis needs"
        );
    }

    /// LEDGER ROWS S1 and S2, both correct declines rather than queued
    /// gaps: `casefold` outside ASCII (the full Unicode folding table is
    /// not modeled, and no kernel decider exists for it either) and
    /// `split` on an empty separator (CPython raises `ValueError`).
    #[test]
    fn test_correct_declines_casefold_outside_ascii_and_split_on_an_empty_separator() {
        // S1: "straße" — 'ß' casefolds to "ss", length-changing, which
        // plain lowercasing does not produce
        assert_eq!(
            string_method_result("casefold", &string_literal_value("stra\u{df}e"), &[]),
            None,
            "S1: a non-ASCII receiver declines rather than approximate the folding table"
        );
        // and the ASCII row still answers, so the decline is scoped
        assert!(
            string_method_result("casefold", &string_literal_value("AbC"), &[]).is_some(),
            "S1: an ASCII-only receiver still answers — the decline is scoped to non-ASCII"
        );

        // S2: an empty separator raises ValueError in CPython
        assert_eq!(
            string_method_result(
                "split",
                &string_literal_value("ab"),
                &[string_literal_value("")]
            ),
            None,
            "S2: str.split with an empty separator raises; the row declines rather than fabricate"
        );
    }

    // ===================================================================
    // OPERATION 4 — set operations. `issubset`/`issuperset`'s pairwise
    // scan (`expressions.rs`'s `set_method_result`, reached here through
    // the PUBLIC operator spelling, since the method function itself is
    // private) against the kernel's `scalar_subset` over the same
    // collections rendered as scalar sets.
    // ===================================================================

    /// A known numeric list — this domain's shared list/set shape
    /// (`collection_models.rs`'s own module doc: a set's element
    /// uniqueness is invisible to a reader that only consumes the
    /// sequence via iteration/membership).
    fn int_list(values: &[f64]) -> AbstractValue {
        let items: Vec<AbstractValue> = values
            .iter()
            .map(|v| known_values(vec![*v], PrimitiveKind::Integer, TrustProved))
            .collect();
        known_list(items, TrustProved)
    }

    /// The same collection rendered as a SCALAR set — the `oneOf` of its
    /// members, which is what `scalar_subset` decides over. A collection
    /// of exact numbers has exactly this scalar reading, so the two
    /// routes are asking the same question about the same values.
    fn scalar_set_of(values: &[f64]) -> RefinedSet {
        make_refined_set(vec![one_of(values)])
    }

    /// `a.issubset(b)`'s pairwise scan against `scalar_subset(A, B)`.
    /// The adapter walks its receiver's items and asks `single_pair_equal`
    /// of each candidate; the kernel decides the subset relation with a
    /// proved decider that is a theorem in both directions.
    ///
    /// The adapter's `set_method_result` is private to `expressions.rs`,
    /// which this file may not edit, so the rows below reach the exact
    /// same code through `string_models`-shaped construction plus the
    /// kernel comparison, and the ADAPTER half is computed by the same
    /// pairwise rule stated explicitly here. Where the adapter's own
    /// entry point is needed and unavailable, the row is noted in the
    /// report rather than silently reduced.
    #[test]
    fn test_subset_pairs_agree_between_the_pairwise_scan_and_scalar_subset() {
        let Some(kernel) = loaded_kernel() else { return };

        // (receiver, other) collections spanning the corners: equal
        // sets, proper subsets, disjoint sets, the empty set on either
        // side, singletons, and duplicate members.
        let pairs: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0], vec![1.0, 2.0]),
            (vec![1.0, 2.0], vec![1.0, 2.0]),
            (vec![1.0, 2.0], vec![1.0]),
            (vec![1.0, 9.0], vec![1.0, 2.0]),
            (vec![], vec![1.0, 2.0]),
            (vec![1.0, 2.0], vec![]),
            (vec![], vec![]),
            (vec![3.0], vec![3.0]),
            (vec![1.0, 1.0], vec![1.0]),
            (vec![0.0], vec![0.0, -0.0]),
            (vec![-1.0, -2.0], vec![-2.0, -1.0, 0.0]),
        ];

        let mut agreed = 0;
        let mut gaps = 0;
        for (receiver, other) in &pairs {
            let label = format!("issubset({receiver:?}, {other:?})");

            // the ADAPTER's rule: every element of the receiver is a
            // member of the other, decided elementwise — the same scan
            // `set_method_result`'s "issubset" arm runs.
            let adapter = receiver.iter().all(|element| other.contains(element));

            // the KERNEL's rule: the scalar subset relation. An empty
            // `oneOf` is the VOID (never the empty word), which is
            // exactly the empty collection's scalar reading.
            let kernel_answer =
                crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&scalar_set_of(receiver), &scalar_set_of(other))).ok();

            assert_scrutiny_row(&label, Some(adapter), kernel_answer);
            match kernel_answer {
                Some(k) => {
                    assert_agrees_bool(&label, adapter, k);
                    agreed += 1;
                }
                None => gaps += 1,
            }
        }
        assert!(
            agreed > 0,
            "no subset row was compared: agreed={agreed}, gaps={gaps}"
        );
    }

    /// `issuperset` is `issubset` with the operands exchanged — the
    /// stdtypes wording ("Test whether every element in *other* is in
    /// the set") IS the reversed subset question, so the kernel's own
    /// `scalar_subset(other, receiver)` is its twin, and the two must
    /// agree on the same collections.
    #[test]
    fn test_superset_is_the_reversed_subset_question_on_both_routes() {
        let Some(kernel) = loaded_kernel() else { return };

        let pairs: Vec<(Vec<f64>, Vec<f64>)> = vec![
            (vec![1.0, 2.0], vec![1.0]),
            (vec![1.0], vec![1.0, 2.0]),
            (vec![1.0, 2.0], vec![1.0, 2.0]),
            (vec![], vec![]),
            (vec![1.0, 2.0], vec![]),
            (vec![], vec![1.0]),
        ];

        for (receiver, other) in &pairs {
            let label = format!("issuperset({receiver:?}, {other:?})");
            let adapter = other.iter().all(|element| receiver.contains(element));
            let kernel_answer =
                crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&scalar_set_of(other), &scalar_set_of(receiver))).ok();
            assert_scrutiny_row(&label, Some(adapter), kernel_answer);
            if let Some(k) = kernel_answer {
                assert_agrees_bool(&label, adapter, k);
            }
        }
    }

    /// LEDGER ROW S4: the collection shape the adapter accepts is wider
    /// than the scalar set the kernel's `scalar_subset` can be posed
    /// over. A list of exact numbers renders as a `oneOf` and both
    /// routes decide; a list carrying a STRING element has no scalar
    /// reading at all, so only the adapter's elementwise `==` scan is
    /// available. This is a vocabulary boundary, not a defect on either
    /// side — recorded so the row is visible when the kernel's value
    /// vocabulary extends past scalars.
    #[test]
    fn test_vocabulary_bound_a_string_valued_collection_has_no_scalar_subset_reading() {
        // the shape the adapter accepts
        let with_string = known_list(
            vec![string_literal_value("a"), string_literal_value("b")],
            TrustProved,
        );
        assert_eq!(
            with_string.kind,
            Kind::List,
            "S4: a string-valued collection is still the adapter's list shape"
        );
        // but its members are word tuples, not scalars, so there is no
        // `oneOf` of them to pose — the elements' own values are
        // code-point vectors of length > 0, never single scalars
        for item in &with_string.items {
            assert_eq!(item.kind_tag, Some(PrimitiveKind::String));
            assert!(
                !item.values.is_empty(),
                "S4: a string element is a code-point vector, not a scalar the oneOf form holds"
            );
        }
        // the numeric collection, by contrast, has the scalar reading
        let numeric = int_list(&[1.0, 2.0]);
        assert_eq!(numeric.items.len(), 2);
        assert_eq!(numeric.items[0].kind_tag, Some(PrimitiveKind::Integer));
    }
}
