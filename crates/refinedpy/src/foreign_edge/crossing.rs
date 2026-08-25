use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::set_of_known;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_text_size::Ranged;

use crate::assignability;
use crate::env::Environment;
use crate::foreign_edge_artifact::ForeignCase;
use crate::foreign_edge_artifact::ForeignTsArtifact;
use crate::foreign_edge_artifact::ForeignTsEntry;

use super::parse_consumer::fire_at;
use super::ForeignEdge;
use super::ForeignEdgeOutcome;

/// Which infinite corner a NUMBER case's own set admits — `None` when
/// the case's own hull is bounded on both ends. A set is an
/// INTERSECTION of its forms, so a ray form (`AtLeast`/`Above` narrows
/// the hull's lower end up; `AtMost`/`Below` narrows the upper end
/// down) only leaves a side unbounded when NO form in the intersection
/// states a finite bound on that side — the same reading
/// `set_simplification.rs`'s own `hull_of` computes for simplification,
/// done locally here since that reader is private to its crate. A
/// `Union` widens to the LOOSER of its two arms' own hulls (either arm
/// admitting the corner means the union does); a `Difference` reads
/// only its left arm's hull (removing members never widens). NaN is
/// excluded from every `RefinedSet` at construction (the boundary
/// ruling), so only the two infinite corners are asked about here. The
/// case tag itself already says "number" — no `on_one_tuple_layer`/
/// `states_sequence` shape gate is needed to tell a number case from a
/// string/sequence one.
pub(super) fn uncarriable_corner_of(set: &RefinedSet) -> Option<&'static str> {
    let hull = hull_of(set);
    if hull.lo == f64::NEG_INFINITY {
        return Some("-Infinity");
    }
    if hull.hi == f64::INFINITY {
        return Some("+Infinity");
    }
    None
}

/// The outermost bounds a set's own top-level forms state, read
/// syntactically — unbounded (`NEG_INFINITY`/`INFINITY`) on a side no
/// form narrows. `MultipleOf` states no bound and is skipped;
/// `uncarriable_corner_of`'s own gate keeps this reader off a
/// sequence-shaped set entirely, so no sequence form ever reaches this
/// match.
pub(super) struct ScalarHull {
    lo: f64,
    hi: f64,
}

pub(super) fn hull_of(set: &RefinedSet) -> ScalarHull {
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    for form in &set.forms {
        match form.form {
            Form::AtLeast | Form::Above => lo = lo.max(form.a),
            Form::AtMost | Form::Below => hi = hi.min(form.a),
            Form::OneOf => {
                if !form.w.is_empty() {
                    lo = lo.max(form.w.iter().copied().fold(form.w[0], f64::min));
                    hi = hi.min(form.w.iter().copied().fold(form.w[0], f64::max));
                }
            }
            Form::Union => {
                let a = hull_of(form.a_.as_ref().unwrap());
                let b = hull_of(form.b.as_ref().unwrap());
                lo = lo.max(a.lo.min(b.lo));
                hi = hi.min(a.hi.max(b.hi));
            }
            Form::Difference => {
                let a = hull_of(form.a_.as_ref().unwrap());
                lo = lo.max(a.lo);
                hi = hi.min(a.hi);
            }
            _ => {}
        }
    }
    ScalarHull { lo, hi }
}

/* ── the outbound leg ─────────────────────────────────────────────── */

/// Discharges every premise about the value that crosses OUT, against
/// the value the walk holds for it. Answers `None` where the leg is
/// clean; an outcome (a decline, or `Fired` after an RTS7001) where it
/// is not.
pub(super) fn check_outbound_leg(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    if artifact.called.entry.is_empty() {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " states no entry position, so "
                + "nothing says what the value crossing out must be",
            range: edge.call,
        });
    }
    // the harness hands the WHOLE parsed stdin value to the called
    // function, so exactly one entry position receives it
    if artifact.called.entry.len() != 1 {
        return Some(ForeignEdgeOutcome::Decline {
            message: format!(
                "the target {} states {} entry positions, and this harness hands it one JSON value from \
                stdin — the checker models no splitting of that value across positions",
                artifact.called.name,
                artifact.called.entry.len()
            ),
            range: edge.call,
        });
    }
    let entry = &artifact.called.entry[0];
    let crossing = crate::expressions::evaluate_expression(&edge.payload, environment, kernel);
    let payload_range = edge.payload.range();
    // NaN-FREEDOM: NaN stringifies to `null` in json.dumps, so the
    // target would receive a value this program never computed
    if let Some(sentence) = nan_freedom_obstacle(&crossing) {
        return Some(fire_at(
            payload_range,
            format!(
                "{sentence} — json.dumps writes NaN as null, so {} would receive a value this program \
                never computed",
                artifact.called.name
            ),
            artifact,
        ));
    }
    if let Some((element_cases, length_at_least)) = &entry.sequence {
        return check_sequence_crossing(edge, artifact, entry, element_cases, *length_at_least, &crossing, kernel);
    }
    if let Some(scalar_cases) = &entry.scalar {
        return check_scalar_crossing(edge, artifact, entry, scalar_cases, &crossing, kernel);
    }
    Some(ForeignEdgeOutcome::Decline {
        message: "the target ".to_owned() + &artifact.called.name + " states an entry position " + &entry.name
            + " that is neither a sequence nor a scalar set — nothing says whether the value fits",
        range: payload_range,
    })
}

/// The union of every NUMBER/STRING case's own set among an entry's
/// cases — the one admitted set a kernel `scalar_subset` ask judges a
/// numeric/string crossing value against. `None` when the cases list
/// carries no set-bearing case at all (every case is Boolean/Null, or an
/// Object case), since there is then no set for a numeric/string
/// crossing to fit — the caller's own existing decline sentence
/// ("nothing says whether the value fits") runs unchanged.
///
/// An `ForeignCase::Object` entry case answers no-set BY DESIGN, not as
/// a staging placeholder: the outbound leg's own question is "does the
/// value CROSSING OUT fit the entry's admitted set," and this checker
/// has no OBJECT-shaped crossing value to ask that of at all today (an
/// outbound Python payload never lowers to `Kind::Object` on this path —
/// `expressions::evaluate_expression` is asked for a Python dict's OWN
/// shape, not the entry's declared one). Fitting an outbound object
/// payload against a declared object entry is a SEPARATE designed unit
/// (its own queue entry: a receiver-shaped fit check, not a return-value
/// lowering) — the consumer-side RETURN lowering this file now carries
/// (`foreign_case_value`'s own Object arm) does not, by itself, give the
/// entry leg anything new to check.
pub(super) fn admitted_set_of_cases(cases: &[ForeignCase]) -> Option<RefinedSet> {
    let mut union_set: Option<RefinedSet> = None;
    for case in cases {
        let set = match case {
            ForeignCase::Number(set) | ForeignCase::String(set) => set.clone(),
            ForeignCase::Boolean | ForeignCase::Null | ForeignCase::Object { .. } => continue,
        };
        union_set = Some(match union_set {
            None => set,
            Some(rest) => make_refined_set(vec![union(set, rest)]),
        });
    }
    union_set
}

/// Whether a value crossing out may carry NaN — the two ways NaN rides
/// beside a set are the `Kind::NaN`/`Kind::PossiblyNaN` wrapper for a
/// scalar and, for a sequence, the `nan_elements` flag its element
/// reading consults. A derived set excludes NaN by construction (a
/// `RefinedSet` denotes a subset of the reals, and NaN is a member of
/// no refined set), so the check is on the value's SHAPE, mirroring the
/// Go twin's `nanFreedomObstacle` exactly.
pub(super) fn nan_freedom_obstacle(crossing: &AbstractValue) -> Option<&'static str> {
    match crossing.kind {
        Kind::NaN => Some("the value crossing to the TypeScript target is NaN"),
        Kind::PossiblyNaN => Some("the value crossing to the TypeScript target may be NaN"),
        Kind::Set if crossing.nan_elements => {
            Some("the sequence crossing to the TypeScript target may hold NaN elements")
        }
        _ => None,
    }
}

/// Judges an array payload against a sequence entry: the elements
/// inside the union of the element's own number/string cases, and the
/// length floor at or above the stated one.
pub(super) fn check_sequence_crossing(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    entry: &ForeignTsEntry,
    element_cases: &[ForeignCase],
    length_at_least: i64,
    crossing: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    let payload_range = edge.payload.range();
    if crossing.kind != Kind::Set || crossing.set_kind_tag != SetKindTag::None {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a sequence at " + &entry.name
                + ", and the value crossing out is not read as one here — nothing says whether it fits",
            range: payload_range,
        });
    }
    let Some(window) = as_repetition(&crossing.set) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: format!(
                "the target {} admits a sequence at {} of at least {} elements, and the value crossing \
                out states no element set or length window — nothing says whether it fits",
                artifact.called.name, entry.name, length_at_least
            ),
            range: payload_range,
        });
    };
    let Some(element_set) = admitted_set_of_cases(element_cases) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a sequence at " + &entry.name
                + " whose element cases carry no number/string set — nothing says whether an element fits",
            range: payload_range,
        });
    };
    // the ELEMENT fit — a real kernel ask
    let fits = match foreign_scalar_subset(kernel, &window.element, &element_set) {
        Some(fits) => fits,
        None => {
            return Some(ForeignEdgeOutcome::Decline {
                message: "the kernel refused the question of whether the elements crossing out fit "
                    .to_owned()
                    + &artifact.called.name
                    + "'s stated entry set, so the crossing is not judged",
                range: payload_range,
            });
        }
    };
    if !fits {
        return Some(fire_at(
            payload_range,
            format!(
                "the elements crossing to {} are outside the target's stated entry set — the value can \
                escape what the target states it accepts",
                artifact.called.name
            ),
            artifact,
        ));
    }
    // the LENGTH floor: the target's body relies on it, so a shorter
    // sequence is a different program
    if window.lo < length_at_least {
        return Some(fire_at(
            payload_range,
            format!(
                "the sequence crossing to {} holds at least {} elements, and the target relies on at \
                least {}",
                artifact.called.name, window.lo, length_at_least
            ),
            artifact,
        ));
    }
    None
}

/// Judges a scalar payload against a scalar entry's own cases: a
/// `Kind::Null` crossing fits when a `Null` case is among them (the
/// `admits_none` entry's own reading); every other crossing is judged
/// against the union of the entry's number/string cases through the
/// same `scalar_subset` kernel ask as before — a `Boolean` case widens
/// that union to admit `0`/`1` (a Python `bool` is an `int` subclass),
/// so a numeric judge already covers a boolean crossing without a
/// separate arm.
pub(super) fn check_scalar_crossing(
    edge: &ForeignEdge,
    artifact: &ForeignTsArtifact,
    entry: &ForeignTsEntry,
    entry_cases: &[ForeignCase],
    crossing: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<ForeignEdgeOutcome> {
    let payload_range = edge.payload.range();
    if crossing.kind == Kind::Null {
        if entry_cases.iter().any(|case| matches!(case, ForeignCase::Null)) {
            return None;
        }
        return Some(fire_at(
            payload_range,
            format!(
                "the value crossing to {} is None, and the target's stated entry admits no null case",
                artifact.called.name
            ),
            artifact,
        ));
    }
    let Some(crossing_set) = set_of_known(crossing) else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a value at " + &entry.name
                + ", and the value crossing out is not read as a set here — nothing says whether it fits",
            range: payload_range,
        });
    };
    let mut entry_set = admitted_set_of_cases(entry_cases);
    if entry_cases.iter().any(|case| matches!(case, ForeignCase::Boolean)) {
        let boolean_set = make_refined_set(vec![one_of(&[0.0, 1.0])]);
        entry_set = Some(match entry_set {
            None => boolean_set,
            Some(rest) => make_refined_set(vec![union(boolean_set, rest)]),
        });
    }
    let Some(entry_set) = entry_set else {
        return Some(ForeignEdgeOutcome::Decline {
            message: "the target ".to_owned() + &artifact.called.name + " admits a value at " + &entry.name
                + " whose cases carry no number/string/boolean set — nothing says whether the value fits",
            range: payload_range,
        });
    };
    let fits = match foreign_scalar_subset(kernel, &crossing_set, &entry_set) {
        Some(fits) => fits,
        None => {
            return Some(ForeignEdgeOutcome::Decline {
                message: "the kernel refused the question of whether the value crossing out fits "
                    .to_owned()
                    + &artifact.called.name
                    + "'s stated entry set, so the crossing is not judged",
                range: payload_range,
            });
        }
    };
    if !fits {
        return Some(fire_at(
            payload_range,
            format!(
                "the value crossing to {} can escape what the target states it accepts",
                artifact.called.name
            ),
            artifact,
        ));
    }
    None
}

/// Asks the kernel A ⊆ B, answering `Some(fits)`, or `None` when the
/// kernel refuses — the same try/catch shape `assignability.rs`'s own
/// `scalar_subset` call wears (assignability.rs:631-643), so a kernel
/// that cannot decide leaves the crossing unjudged rather than refuting
/// it.
///
/// The question is picked by the operands' sort, mirroring refined-ts-
/// go's own `foreignScalarSubset` (walk/foreign_edge.go): a sequence-
/// shaped operand (a string window, a concatenation, a union of words —
/// `states_sequence`'s fast top-level test OR `sequence_shaped`'s
/// recursive one, on EITHER side) asks `seq_subset` — the decider whose
/// grammar reads those shapes — and a scalar pair asks `scalar_subset`.
/// Sending a string set through the scalar decider is a question the
/// kernel rightly refuses (`"kernel: subset is decided for scalar and
/// sequence shapes today"`), which this function used to read as an
/// ordinary refusal rather than a misrouted question.
pub(super) fn foreign_scalar_subset(kernel: &Arc<RefinedTSKernel>, a: &RefinedSet, b: &RefinedSet) -> Option<bool> {
    let sequence_question = assignability::states_sequence(a)
        || assignability::sequence_shaped(a)
        || assignability::states_sequence(b)
        || assignability::sequence_shaped(b);
    if sequence_question {
        if let Ok(fits) = crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(a, b)) {
            return Some(fits);
        }
    }
    crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(a, b)).ok()
}

