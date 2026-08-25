//! The one judging seam: a flowing value against a declared refinement.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::is_codepoint_alphabet;
use refined_sets::format_string_shapes::format_py_number;
use refined_sets::refinement_forms::on_one_tuple_layer;
use refined_sets::refinement_forms::requires_integer;

use crate::diagnostic_sentences::at_index;
use crate::diagnostic_sentences::at_key;
use crate::diagnostic_sentences::at_member;
use crate::diagnostic_sentences::at_slot;
use crate::diagnostic_sentences::containment_refutation;
use crate::diagnostic_sentences::cross_sort_of_value;
use crate::diagnostic_sentences::element_set_refutation;
use crate::diagnostic_sentences::refutation;
use crate::diagnostic_sentences::required_words;
use crate::diagnostic_sentences::SENTENCE;
use crate::typereading::DeclaredRefinement;

use super::scalar::scalar_or_string_shaped;
use super::scalar::spelled_boolean_word;
use super::scalar::spelled_string_word;
use super::scalar::within_codepoint_door;
use super::sequence::sequence_shaped_safely;
use super::sequence::states_sequence;
use super::temporal::temporal_admission_refusal;
use super::temporal::temporal_alert_sentence;
use super::temporal::temporal_refutation;
use super::Verdict;

/// Judge a flowing value against a declared refinement.
///
/// The OPAQUE law runs first, before any other case: a value whose KIND
/// OF THING is known but whose contents are not (`kind_word: Some(word)`
/// — `abstract_value::opaque_value`, built on `Kind::Object`) against
/// ANY scalar-ground declared set (numeric-ground `on_one_tuple_layer`,
/// or string-ground `is_string_ground`) fires with the honest word
/// ("a function value is not assignable to type 'Age'"), so it wins
/// over the generic Object law's "a dict" below. A declared set that is
/// not scalar-ground declines and falls through.
///
/// `Kind::Values` (a known scalar, or a known tuple word) asks the
/// kernel per value — EXCEPT three SORT laws, judged before any kernel
/// question, because all three are facts only the checker's own tags
/// state (the kernel's `member` decides "is this real number/word
/// inside this real set," and never carries the checker's own sort
/// tags):
///
/// - The mission's int-sort law: a Float-tagged value against a
///   declared set that carries the `int` form (`requires_integer`)
///   fires outright (PYREFLY-NUMERIC-B3-B4.md's "two sorts, never one
///   Number") — `30.0`'s real value of 30 sits inside `[0, 120]`
///   exactly as `30`'s does, so the sort mismatch is never visible to
///   `kernel.member` itself.
/// - The string/numeric ground law: a String-tagged value (one whole
///   word, `codepoint_sets`'s "a string value IS its codepoint tuple")
///   against a declared set that is NUMERIC-ground
///   (`on_one_tuple_layer`, the scalar ray/point forms) fires, and the
///   mirror — a numeric value against a declared set that DEMONSTRABLY
///   STATES A SEQUENCE (`states_sequence` — a Star/Concatenation/
///   Repeat/RepeatWord/EmptyTuple form present) fires too. A tuple of
///   code points and a tuple of numbers share the same wire shape; only
///   the two sides' own sort tags can tell a string from a number, so
///   this is judged the same way as the int-sort law rather than asked
///   of the kernel. THE TUPLE PUN: a 1-character string's own tuple
///   (`codepoint_sets::string_tuple`'s length-1 encoding is a bare
///   `OneOf`, never wrapped in `Concatenation`) sits on the one-tuple
///   layer exactly like a numeric point does, so `Literal["A", "B",
///   "C"]` (a `Union` of single-codepoint `OneOf`s, `surface.rs`'s
///   `string_literal_set`) reads as numeric-ground under
///   `on_one_tuple_layer` alone with no way to tell it from a bare
///   numeric `one_of([65, 66, 67])`. The string-into-numeric-ground
///   direction is therefore gated on `states_sequence(declared.set) ||
///   !within_codepoint_door(declared.set, false)` (ported from
///   refined-ts-go's `checkExactValues`/`StatesSequence`/
///   `WithinCodepointDoor`, walk/set_membership.go +
///   walk/sequence_measures.go): a target that does not demonstrably
///   state a sequence AND sits wholly inside the codepoint alphabet
///   (every admitted value is a valid single codepoint — `Grade`'s own
///   shape) is indistinguishable from a union of one-character strings,
///   so a String-tagged value may legitimately be one of its members
///   and the law declines, falling through to the ordinary whole-word
///   kernel membership ask below.
/// - The BOOLEAN PRODUCT LAW: a Boolean-tagged value against a declared
///   set that requires the `int` form fires — `True`/`False` are `bool`,
///   and bool is excluded from the int sort by product law
///   (fixtures/language/syntax-coverage-py/b-body-expressions.py:744,
///   c-reads-and-values.py:999), even though `True`'s numeric value 1
///   sits inside every ordinary int range. Judged BEFORE the per-value
///   kernel membership ask below, so `True` never silently passes as
///   `1`. Scoped to `requires_integer` only — arithmetic still reads a
///   Boolean operand as Integer (`True + True == 2`, unchanged); this
///   law is about a Boolean-tagged value ARRIVING at a judgment, not
///   about arithmetic transfer.
///
/// Every other Values case asks the kernel: a String-tagged value asks
/// ONE membership question over the whole word (`value.values` IS the
/// one string, never per-code-point askings), spelling the fire with
/// the string decoded back to text (`format_string_shapes::from_points`
/// — the same JSON-quoted spelling `format_string_literal` uses for a
/// set's own literal chain). Every non-string Values case (Integer-
/// tagged, or bare Number where the sort is unknown) asks the kernel
/// per value exactly as before, spelling the fire message the Python
/// way via `format_py_number`.
///
/// `Kind::Set` (a refined set of possible values, not one exact word)
/// carries its own SORT laws first, mirroring the Values-side
/// string/numeric ground law and judged the same way — before any
/// kernel ask, because the sort is a fact only the checker's own tags
/// (or, for an untagged Set, the SET'S OWN SHAPE) state, never something
/// `scalar_subset`/`scalar_disjoint` can see:
///
/// - A STRING-sorted Set (`kind_tag: Some(PrimitiveKind::String)`, or an
///   UNTAGGED Set — `kind_tag: None` — whose own set is SEQUENCE-shaped,
///   `sequence_shaped` below, the untagged-Set-reads-as-string-sorted
///   convention `AGENT-BRIEF.md`/`ORIENTATION.md` both pin) against a
///   declared set that is NUMERIC-ground (`on_one_tuple_layer`) AND
///   (per the same tuple-pun gate the Values-side law above takes)
///   either demonstrably states a sequence or sits outside the
///   codepoint door fires: a string-sorted value is never a member of
///   an int-sorted set, regardless of which characters either side
///   admits. The `__name__` read (`expressions.rs`, `known_set(strings(),
///   None, TrustSpec, SetKindTag::None)` — untagged, the full string
///   ground) flowing into an `Age`-shaped return is exactly this shape.
/// - The mirror: a NUMERIC-sorted Set (`kind_tag: Some(Integer/Float/
///   Number)`, or an untagged Set whose own set sits `on_one_tuple_layer`
///   — the shape a bare `integer()`/range Set carries with no producer
///   ever tagging it) against a declared set that DEMONSTRABLY STATES A
///   SEQUENCE (`states_sequence`) fires the same way.
///
/// Only when both sides share a sort (both string-shaped, or both
/// numeric-shaped, or the value's shape is neither recognized law's
/// antecedent) does judgment fall through past these two laws to the
/// FLOAT-SORT law and then the containment ask below — so a same-sort
/// pair the kernel cannot yet decide (e.g. the full string ground
/// against a bounded length window) still reaches the CONTAINMENT
/// REFUSAL catch rather than being wrongly waved through or wrongly
/// fired by a sort law that does not apply to it.
///
/// After the two sort-shape laws, a Set carries its own FLOAT-SORT law:
/// a Float-sorted Set (`kind_tag:
/// Some(PrimitiveKind::Float)` — `abstract_value::float_sorted_unknown`,
/// the shape `math.sqrt`'s value-unknown result carries) against a
/// declared set that requires the `int` form fires outright, the same
/// reasoning as the Values-side int-sort law: a float-sorted value is
/// never a member of an int-sorted set, regardless of which real
/// numbers either side admits. A Float-sorted Set against a
/// NON-integer-sorted declared set declines this law and falls through
/// to the CONTAINMENT-REFUTATION law (adopted from
/// refined-ts-go/internal/refinedts/walk/check_assignability.go's own
/// three-outcome vocabulary: "proved says nothing, a refutation reports
/// 7001 with the counterexample spelled, and an undetermined verdict
/// reports 7002"): a checked position IS the claim `flowing ⊆ declared`,
/// so `scalar_subset` proving it holds is silent, and ANY proof that it
/// does NOT hold — subset false, whether the two sets are disjoint or
/// merely overlapping — is a refutation and fires. The two-ask form
/// below (`scalar_subset` then `scalar_disjoint`) is not a second
/// three-way split; it exists only because the disjoint case gets its
/// own message emphasis (no member of either set lines up with the
/// other), while the overlap case's fire message spells both sets and
/// lets the reader see the escaping region. Both closures are total
/// over the scalar (1-tuple) shape `Kind::Set` carries here: the Lean
/// kernel's `kernelScalarSubset`/`kernelScalarDisjoint`
/// (refined-ts-lean/boundary/exports.lean) are proved theorems "in both
/// directions" over scalar-shaped sets and only `fail` (a Lean-level
/// refusal) on a non-scalar shape; the Rust closures
/// (`refined_kernel::kernel_asks`) turn any such refusal into a `panic!`
/// rather than a `false` — so a `false` this file ever observes from
/// `scalar_subset` or `scalar_disjoint` is always a DECIDED refutation,
/// never a refusal in disguise (matching Go's own
/// `containedInAsked`/`scalarSubsetAsked`, which documents "the scalar
/// decider is a theorem in both directions, so its false is a verdict,
/// not a refusal" — refused kernel asks there are a `recover()`d panic,
/// exactly the same shape as this crate's `.unwrap_or_else(|err| panic!
/// ...)`). No search loop names a counterexample element; the fire
/// message spells both sets via `format_for_diagnostics` and leaves
/// finding a witness to the reader — the only existing helper that
/// could name one, `refinement_forms::word_of`, reads a SINGLETON
/// shape's own tuple and has no bearing on naming a member of the
/// flowing set that escapes the declared set.
///
/// `Kind::Object` / `Kind::List` (a dict, or a list/tuple) can never be
/// a member of a SCALAR declared set (numeric-ground or string-ground)
/// — neither is a number or a string, so this is a structural sort
/// mismatch and fires outright rather than sitting undetermined. A
/// declared set that is not recognizably scalar-ground (numeric or
/// string) declines this law and falls through to the general
/// undetermined answer below.
///
/// `Kind::Null` (Python's `None`) is the same structural mismatch
/// UNLESS the declaration itself admits absence (`declared.admits_none`
/// — `Optional[X]`/`X | None`, set by `typereading.rs`): an admitted
/// `None` is silent, a `None` against a plain (non-`Optional`)
/// declared set fires the same as Object/List.
///
/// `Kind::KindUnion` (a sort union — `json.loads`'s own honest return
/// space over an opaque string is the one producer today) judges each
/// arm against `declared` through this same function and takes the
/// first Fire; every arm Silent is Silent; any arm Undetermined makes
/// the whole judgment Undetermined. See the function body's own comment
/// at that arm for the full rule.
///
/// Anything else (`Kind::Unknown`, and every other not-yet-known shape)
/// is undetermined with a sentence the caller may adopt as the body's
/// blocker.
pub fn judge(
    value: &AbstractValue,
    declared: &DeclaredRefinement,
    kernel: &Arc<RefinedTSKernel>,
) -> Verdict {
    // The ELEMENT LAW: a container declaration (`dict[str, X]`,
    // `declared.element` Some, `declared.set` unused/empty) judges
    // every MEMBER VALUE against its element refinement rather than
    // judging the container itself against a scalar/sequence set —
    // `declared.set` carries nothing a dict could ever be a member of.
    // A known Object (a dict literal, `Kind::Object` with no
    // `kind_word` — the opaque-object law above already owns the
    // kind_word case) walks every key in order and asks THIS SAME
    // `judge` of each member's value; the first Fire is the verdict,
    // its message naming the offending key so the reader sees which
    // member escaped. All-Silent members is Silent. Any Undetermined
    // member makes the whole judgment Undetermined, carrying that
    // member's own sentence (the walk cannot claim more than its
    // least-determined member knows). `None`/a list against an
    // element-carrying declaration are their own explicit arms below —
    // a dict declaration is not scalar-shaped, so the ordinary
    // structural law (further down, gated on `scalar_or_string_shaped`)
    // must never see them: `declared.set` is empty, which
    // `scalar_or_string_shaped` reads as "not scalar-ground" and would
    // leave None/a list Undetermined instead of firing the honest
    // structural mismatch. `None`'s `admits_none` check comes first — a
    // `dict[str, X] | None` declaration is still element-carrying (the
    // `| None` wrapper only sets `admits_none`, never clears `element`,
    // per typereading.rs's union-arm recursion), and `admits_none` wins
    // the same way it does for every other declaration shape.
    //
    // `Kind::PossiblyUndefined` (an `Optional[X]`/`X | None`-declared
    // PARAMETER's own seed, `check.rs::seed_parameters`, narrowed or not
    // — an un-narrowed wrapper reaches here exactly like a bare
    // un-narrowed `Kind::Null` already does): the wrapper's absent side
    // is Python's `None`, judged the SAME way a bare `Kind::Null` already
    // is right below (`declared.admits_none` wins, or fires the
    // structural mismatch) — a flowing `None` is `None` whether it
    // arrived as the exact null_value or as this wrapper's own absent
    // admission. The wrapper's PRESENT side (`value.inner`) judges
    // through this SAME seam recursively, so a parameter's own annotated
    // set still fires/silences/blocks exactly as it would un-wrapped —
    // the maybe carrier changes nothing about what its present side
    // states.
    if value.kind == Kind::PossiblyUndefined {
        if !declared.admits_none {
            return Verdict::Fire(refutation("None", &declared.spelling, &declared.set));
        }
        let inner = value.inner.as_deref().expect("Kind::PossiblyUndefined always carries an inner value");
        return judge(inner, declared, kernel);
    }
    // `Kind::NaN` (the arithmetic layer's own answer for `inf - inf`,
    // `inf * 0`, `inf / inf`, `float("nan")`): a `RefinedSet` denotes a
    // subset of the reals, and NaN is a member of no refined set (the
    // boundary ruling `foreign_edge.rs::nan_freedom_obstacle` already
    // states for the cross-language crossing) — so a provably-NaN value
    // is provably outside EVERY declared refinement this table judges,
    // unconditionally. No declared-side spelling admits NaN into a set:
    // `PYREFLY-PYDANTIC-SURFACE.md`'s own row for `allow_inf_nan=True`
    // (float/Decimal's pydantic default) reads "honesty: do not admit
    // NaN into sets, no zod twin" — `typereading.rs` never compiles that
    // knob into `DeclaredRefinement` at all, so this fires outright, the
    // same unconditional way a dict/list fires against a scalar-ground
    // declared set below, never gated on `declared.admits_none` or any
    // other declared-side field. `Kind::PossiblyNaN` (the wrapper a
    // narrowing has not yet stripped, e.g. an unguarded `x / y` where `y`
    // may be zero) is not provably NaN — the PRESENT side may or may not
    // sit inside the declared set — so it judges its inner value through
    // this SAME seam recursively, exactly the way `Kind::PossiblyUndefined`
    // above judges its own present side: the maybe-NaN carrier changes
    // nothing about what the inner value states, and cannot itself be
    // read as a fire (that would refuse every real value the wrapper may
    // also hold) or a silence (that would let an actually-NaN run escape
    // undetected).
    if value.kind == Kind::NaN {
        return Verdict::Fire(format!(
            "{} — a value that is NaN is a member of no refined set",
            refutation("NaN", &declared.spelling, &declared.set),
        ));
    }
    if value.kind == Kind::PossiblyNaN {
        let inner = value.inner.as_deref().expect("Kind::PossiblyNaN always carries an inner value");
        return judge(inner, declared, kernel);
    }
    // THE TEMPORAL LAW: a `date`/`timedelta`/`datetime`/`AwareDatetime`/
    // `NaiveDatetime` declaration (`declared.temporal` Some, `declared.
    // set` unused/empty, the same "one active field" convention `element`/
    // `positions`/`members` already keep) judges a flowing `Kind::Object`
    // construction (`source` one of `"datetime_date"`/
    // `"datetime_timedelta"`/`"datetime_datetime"` — expressions.rs's own
    // three temporal constructors) against the declared calendar window,
    // through the kernel's calendar seam (`bounds_verdict_of`,
    // calendar_interpreter.rs). `None`'s `admits_none` check comes first,
    // the same rule every other container-shaped declaration keeps.
    if let Some(declared_temporal) = &declared.temporal {
        if value.kind == Kind::Null {
            return if declared.admits_none {
                Verdict::Silent
            } else {
                Verdict::Fire(refutation("None", &declared.spelling, &declared.set))
            };
        }
        // A WINDOW-FLOWING value (`source == "temporal_flow"` —
        // `check.rs::seed_parameters`'s own temporal seed: a temporal-
        // declared PARAMETER's own value, representing "any member of
        // its own declared window," never one concrete construction) is
        // judged by IMPLICATION (`bounds_imply`) rather than by exact-
        // point containment: does every value the flowing window admits
        // also sit inside the declared window. This is the shape a
        // temporal parameter flowing into ANOTHER temporal-declared
        // parameter takes (`record_visit`'s own `in_period(v)`,
        // `backwards`'s own `narrow(p)`) — there is no single concrete
        // instant to spell, only the flowing declaration's own bound.
        if value.kind == Kind::Object && value.source == "temporal_flow" {
            let Some(flowing_temporal) = &value.temporal else {
                return Verdict::Undetermined(SENTENCE.temporal_position.to_owned());
            };
            let calendar_ask = refined_kernel::calendar_adapter::calendar_ask_of(kernel);
            let asked = crate::kernel_ask::ask_kernel(|| {
                refined_sets::calendar_interpreter::bounds_imply(&*calendar_ask, flowing_temporal, declared_temporal)
            });
            return match asked {
                Ok(refined_sets::calendar_interpreter::BoundsVerdict::Proved) => Verdict::Silent,
                Ok(refined_sets::calendar_interpreter::BoundsVerdict::Refuted(_side)) => {
                    Verdict::Fire(temporal_refutation(flowing_temporal, declared, declared_temporal))
                }
                Ok(refined_sets::calendar_interpreter::BoundsVerdict::Alert(why)) => {
                    Verdict::Undetermined(temporal_alert_sentence(&why))
                }
                Err(_) => Verdict::Undetermined(SENTENCE.temporal_unprovable_instant.to_owned()),
            };
        }
        let is_temporal_construction = value.kind == Kind::Object
            && matches!(value.source.as_str(), "datetime_date" | "datetime_timedelta" | "datetime_datetime");
        if !is_temporal_construction {
            return Verdict::Undetermined(SENTENCE.temporal_position.to_owned());
        }
        // THE AWARE/NAIVE ADMISSION LAW: `AwareDatetime` states its own
        // documented refusal of a naive construction OUTRIGHT (pydantic's
        // docs, cited at `temporal_admission_refusal`'s own call site) —
        // decided from the construction's OWN fields (whether
        // `instance.temporal` is populated at all — `datetime_construction_
        // value`'s own doc: `None` only for `TzinfoKind::OtherAware`, never
        // for `Naive`), checked BEFORE `bounds_verdict_of`, since a naive
        // value has no exact instant to compare bounds against in the
        // first place.
        if let Some(fire) = temporal_admission_refusal(value, declared) {
            return Verdict::Fire(fire);
        }
        let Some(value_temporal) = &value.temporal else {
            // `TzinfoKind::OtherAware` (a recognized-but-unresolvable
            // tzinfo, e.g. `zoneinfo.ZoneInfo(...)`) — admitted as AWARE
            // by the law above, but its own exact instant is not provable
            // against ANY bound (`chart_reading`'s own `Instant` arm would
            // read it `Unprovable` regardless), so this position is
            // undetermined rather than guessed either way.
            return Verdict::Undetermined(SENTENCE.temporal_unprovable_instant.to_owned());
        };
        let calendar_ask = refined_kernel::calendar_adapter::calendar_ask_of(kernel);
        let asked = crate::kernel_ask::ask_kernel(|| {
            refined_sets::calendar_interpreter::bounds_verdict_of(&*calendar_ask, declared_temporal, value_temporal.min.as_deref().unwrap_or(""))
        });
        return match asked {
            Ok(refined_sets::calendar_interpreter::BoundsVerdict::Proved) => Verdict::Silent,
            Ok(refined_sets::calendar_interpreter::BoundsVerdict::Refuted(_side)) => {
                Verdict::Fire(temporal_refutation(value_temporal, declared, declared_temporal))
            }
            Ok(refined_sets::calendar_interpreter::BoundsVerdict::Alert(why)) => {
                Verdict::Undetermined(temporal_alert_sentence(&why))
            }
            Err(_) => Verdict::Undetermined(SENTENCE.temporal_unprovable_instant.to_owned()),
        };
    }
    if let Some(element) = &declared.element {
        if value.kind == Kind::Null {
            if declared.admits_none {
                return Verdict::Silent;
            }
            return Verdict::Fire(refutation("None", &declared.spelling, &declared.set));
        }
        // WHICH container the declaration names is read off its own
        // spelling (both element-carrying constructors build it —
        // "dict[str, X]" vs "list[X]"/"set[X]"): a dict declaration
        // judges an Object's MEMBER VALUES, a list/set declaration
        // judges a List's ITEMS, and the mismatched container kind
        // fires the structural mismatch. Spelling-based dispatch is a
        // stopgap the doc comment owns honestly — a container tag on
        // DeclaredRefinement is the clean form once a third container
        // arrives.
        let declares_sequence =
            declared.spelling.starts_with("list[") || declared.spelling.starts_with("set[");
        if value.kind == Kind::Object && value.kind_word.is_none() {
            if declares_sequence {
                return Verdict::Fire(refutation("a dict", &declared.spelling, &declared.set));
            }
            for key in &value.keys {
                match judge(&key.value, element, kernel) {
                    Verdict::Fire(message) => {
                        return Verdict::Fire(at_key(&message, &key.name));
                    }
                    Verdict::Undetermined(sentence) => return Verdict::Undetermined(sentence),
                    Verdict::Silent => {}
                }
            }
            return Verdict::Silent;
        }
        if value.kind == Kind::List {
            if !declares_sequence {
                return Verdict::Fire(refutation("a list", &declared.spelling, &declared.set));
            }
            for (index, item) in value.items.iter().enumerate() {
                match judge(item, element, kernel) {
                    Verdict::Fire(message) => {
                        return Verdict::Fire(at_index(&message, index));
                    }
                    Verdict::Undetermined(sentence) => return Verdict::Undetermined(sentence),
                    Verdict::Silent => {}
                }
            }
            return Verdict::Silent;
        }
        // The ELEMENT-SET LAW: a `Kind::Set` value under a container
        // declaration is an UNKNOWN-LENGTH sequence — a repetition
        // window over its own element set, `check.rs::seed_parameters`'s
        // own seed for a `list[X]`/`set[X]`/`Sequence[X]` PARAMETER, and
        // the shape a comprehension over such a parameter still carries
        // when it returns (`expressions.rs::comprehension_star_elements`,
        // which re-windows the mapped element but keeps the outer value
        // a `Kind::Set` repetition — there is no concrete per-item list
        // to walk the way the `Kind::List` arm above does). There is no
        // single escaping INDEX to blame, so this asks whether the
        // value's own admitted ELEMENT SET sits inside the declared
        // element set as a whole — the SAME `seq_subset`-when-sequence-
        // shaped, `scalar_subset`-otherwise routing the CONTAINMENT-
        // REFUTATION law takes further below in this file for a bare
        // `Kind::Set` value (see that law's own doc): `sequence_shaped_
        // safely` on EITHER side decides which ask actually applies —
        // `Ratings`'s own int-bounded element sits on the 1-tuple layer,
        // so it takes `scalar_subset` directly, while an element that is
        // itself sequence-shaped (a `list[str]`'s own element window)
        // takes `seq_subset`, falling back to `scalar_subset` only on a
        // REFUSAL (a kernel decline, never a decided `false`).
        if value.kind == Kind::Set && declares_sequence {
            let Some(repeated) = refined_sets::repetition_window_forms::as_repetition(&value.set) else {
                return Verdict::Undetermined(SENTENCE.value_not_readable.to_owned());
            };
            if sequence_shaped_safely(&repeated.element, kernel) || sequence_shaped_safely(&element.set, kernel) {
                let seq_asked =
                    crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(&repeated.element, &element.set));
                match seq_asked {
                    Ok(true) => return Verdict::Silent,
                    Ok(false) => {
                        return Verdict::Fire(element_set_refutation(&repeated.element, &element.set));
                    }
                    Err(_) => {}
                }
            }
            let scalar_asked =
                crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&repeated.element, &element.set));
            return match scalar_asked {
                Ok(true) => Verdict::Silent,
                Ok(false) => Verdict::Fire(element_set_refutation(&repeated.element, &element.set)),
                Err(_) => Verdict::Undetermined(SENTENCE.kernel_declined_containment.to_owned()),
            };
        }
    }
    // The POSITIONS LAW: a fixed-arity tuple declaration (`declared.
    // positions` Some, `declared.set` unused/empty, the same "one active
    // field" convention `element` keeps) judges EACH SLOT against ITS OWN
    // refinement, keyed by index rather than by name — unlike the ELEMENT
    // LAW above, which shares one refinement across every position of a
    // `list[X]`/`set[X]`, a fixed-arity tuple's positions are checked
    // separately. Only a known List with the SAME LENGTH as `positions`
    // is judged this way; a length mismatch is the same structural
    // mismatch CPython itself would reject the annotation over, so it
    // fires rather than sitting undetermined. `None` is its own explicit
    // arm (`admits_none` wins exactly like the element/members laws); any
    // other value shape (an opaque object, an unresolved value) falls
    // through undetermined rather than guessing a structural mismatch
    // that may not hold. The first Fire among the positions wins, naming
    // the offending index; any Undetermined position makes the whole
    // judgment Undetermined; every position Silent is Silent.
    if let Some(positions) = &declared.positions {
        if value.kind == Kind::Null {
            return if declared.admits_none {
                Verdict::Silent
            } else {
                Verdict::Fire(refutation("None", &declared.spelling, &declared.set))
            };
        }
        if value.kind == Kind::List {
            if value.items.len() != positions.len() {
                let count = value.items.len();
                let plural = if count == 1 { "" } else { "s" };
                return Verdict::Fire(format!(
                    "a value of {count} element{plural} is not assignable to type {} — the position states {} element{}",
                    required_words(&declared.spelling, &declared.set),
                    positions.len(),
                    if positions.len() == 1 { "" } else { "s" },
                ));
            }
            for (index, (item, position_declared)) in value.items.iter().zip(positions.iter()).enumerate() {
                match judge(item, position_declared, kernel) {
                    Verdict::Fire(message) => {
                        return Verdict::Fire(at_slot(&message, index, positions.len(), &position_declared.set));
                    }
                    Verdict::Undetermined(sentence) => return Verdict::Undetermined(sentence),
                    Verdict::Silent => {}
                }
            }
            return Verdict::Silent;
        }
        return Verdict::Undetermined(SENTENCE.tuple_position.to_owned());
    }
    // The MEMBERS LAW: a TypedDict declaration (`declared.members` Some,
    // `declared.set` unused/empty, the same "one active field"
    // convention `element` keeps) judges EACH NAMED MEMBER against ITS
    // OWN refinement — unlike the ELEMENT LAW above, which shares one
    // refinement across every key, a TypedDict's members are
    // heterogeneous, so each declared name carries its own set. Only a
    // known Object with no `kind_word` (a dict literal) is judged this
    // way; `None` is its own explicit arm (`admits_none` wins exactly
    // like the element law); a `Kind::List` is the same structural
    // mismatch the element law already fires for a container
    // declaration. Anything else this table has not yet read (an opaque
    // object, an unresolved value) falls through undetermined rather
    // than guessing a structural mismatch that may not hold. A declared
    // member ABSENT from the value's own keys is not judged at all (an
    // absent key states nothing this table can read into a member's own
    // set, matching `judge_construction`'s honest-absence convention
    // elsewhere in this checker); an extra key the declaration does not
    // name is likewise not judged — TypedDict's own `total=True` default
    // requires every declared key to be present at runtime, but a
    // structural extra-key refusal is a different check this row does
    // not ask for. The first Fire among the declared members wins,
    // naming the offending member; any Undetermined member makes the
    // whole judgment Undetermined; all declared members present and
    // Silent is Silent.
    if let Some(members) = &declared.members {
        if value.kind == Kind::Null {
            return if declared.admits_none {
                Verdict::Silent
            } else {
                Verdict::Fire(refutation("None", &declared.spelling, &declared.set))
            };
        }
        if value.kind == Kind::List {
            return Verdict::Fire(refutation("a list", &declared.spelling, &declared.set));
        }
        if value.kind == Kind::Object && value.kind_word.is_none() {
            for (member_name, member_declared) in members {
                let Some(member_value) = value.keys.iter().find(|key| key.name == *member_name && !key.numeric)
                else {
                    continue;
                };
                match judge(&member_value.value, member_declared, kernel) {
                    Verdict::Fire(message) => {
                        return Verdict::Fire(at_member(&message, member_name, &member_declared.set));
                    }
                    Verdict::Undetermined(sentence) => return Verdict::Undetermined(sentence),
                    Verdict::Silent => {}
                }
            }
            return Verdict::Silent;
        }
        return Verdict::Undetermined(SENTENCE.typed_dict_position.to_owned());
    }
    if value.kind == Kind::Object && value.kind_word.is_some() {
        let scalar_ground = scalar_or_string_shaped(&declared.set);
        if scalar_ground {
            let word = value.kind_word.expect("checked Some above");
            return Verdict::Fire(format!(
                "a value of kind '{word}' is not assignable to type {} — {word} is neither a number nor a string, and this position states one",
                required_words(&declared.spelling, &declared.set),
            ));
        }
    }
    if value.kind == Kind::Values {
        let is_string = value.kind_tag == Some(PrimitiveKind::String);
        let is_float_sorted = value.kind_tag == Some(PrimitiveKind::Float);
        let is_boolean = value.kind_tag == Some(PrimitiveKind::Boolean);
        // The TUPLE PUN: a 1-character string's own tuple (`string_tuple`'s
        // `OneOf([codepoint])`, no `Concatenation` wrapper for a
        // length-1 word) sits ON THE ONE-TUPLE LAYER exactly like a bare
        // numeric point does — `Literal["A", "B", "C"]` compiles through
        // nothing but `OneOf`/`Union` (surface.rs's `string_literal_set`,
        // typereading.rs's own twin), so `on_one_tuple_layer` alone
        // cannot tell "B"'s target set from a numeric one-of. Ported
        // from refined-ts-go's own two-part gate
        // (walk/set_membership.go's `checkExactValues`,
        // walk/sequence_measures.go's `StatesSequence`/
        // `WithinCodepointDoor`): the string-into-numeric-ground fire
        // requires the target to DEMONSTRABLY state a sequence shape
        // (`states_sequence` — a Star/Concatenation/Repeat/RepeatWord/
        // EmptyTuple form present, never inferred from the layer alone)
        // OR sit outside the codepoint door (`within_codepoint_door` —
        // every value the target admits is a valid single codepoint, so
        // the set is indistinguishable from a union of one-character
        // strings and a string value may legitimately be one of its
        // members).
        // An explicit `Integer` form is numeric INTENT by construction —
        // no string set ever carries it (`string_tuple`/`string_literal_set`
        // build OneOf/Concatenation/Union only), so `requires_integer`
        // opens the sort law even INSIDE the codepoint door: `Age`'s
        // `[0,120] ∧ integer` refuses every String-tagged value outright
        // (the corpus's "a string key is not in the set"), while `Grade`'s
        // integer-less union of one-codepoint OneOfs still declines to the
        // membership ask below.
        //
        // ONE CARVE-OUT inside that: `is_codepoint_alphabet` — the
        // declared set being STRUCTURALLY the codepoint alphabet itself
        // (`codepoint_sets::codepoints()`), not merely a narrower
        // Integer-bounded window inside it. This is what `SingleCharacter
        // = Annotated[str, Field(min_length=1, max_length=1)]` compiles
        // to: `repetition_window_forms::repetition`'s own (1,1) collapse
        // hands back the bare codepoint-ground element (its documented
        // "a 1-element sequence IS the scalar layer" rule), so the
        // sequence marker that would otherwise gate this law
        // (`states_sequence`) never survives the collapse, and the
        // element's own `Form::Integer` (part of the codepoint alphabet's
        // definition, not a declared int base) makes `requires_integer`
        // true exactly the way `Age`'s genuine int base does. The two are
        // representationally identical except for their BOUNDS: `Age`'s
        // window is a strict subset of the alphabet, the collapsed
        // element IS the alphabet — `is_codepoint_alphabet` is the one
        // test that tells them apart, so the fire below is reached only
        // for a declared set that is NOT the alphabet outright.
        if is_string
            && on_one_tuple_layer(&declared.set)
            && !is_codepoint_alphabet(&declared.set)
            && (requires_integer(&declared.set)
                || states_sequence(&declared.set)
                || !within_codepoint_door(&declared.set, false))
        {
            // The position's own sort is named as precisely as the set
            // states it: an explicit `Integer` form says "an integer",
            // a bare numeric layer says "a number".
            let position_said = if requires_integer(&declared.set) {
                "an integer"
            } else {
                "a number"
            };
            return Verdict::Fire(cross_sort_of_value(
                &spelled_string_word(&value.values),
                "a string",
                position_said,
                &declared.spelling,
                &declared.set,
            ));
        }
        if !is_string && states_sequence(&declared.set) {
            for v in &value.values {
                return Verdict::Fire(cross_sort_of_value(
                    &format_py_number(*v, is_float_sorted),
                    "a number",
                    "a string",
                    &declared.spelling,
                    &declared.set,
                ));
            }
            return Verdict::Silent; // an empty tuple word has no value to fire on
        }
        if is_float_sorted && requires_integer(&declared.set) {
            for v in &value.values {
                return Verdict::Fire(cross_sort_of_value(
                    &format_py_number(*v, true),
                    "a float",
                    "an integer",
                    &declared.spelling,
                    &declared.set,
                ));
            }
            return Verdict::Silent; // an empty tuple word has no value to fire on
        }
        if is_boolean && requires_integer(&declared.set) {
            return Verdict::Fire(format!(
                "{} — the value is a boolean, the position states an integer, and bool is excluded from the int sort by product law",
                refutation(
                    &spelled_boolean_word(&value.values),
                    &declared.spelling,
                    &declared.set,
                ),
            ));
        }
        // Every member ask is wrapped like the containment ask below: a
        // kernel REFUSAL (a set shape the member decider does not
        // decide) panics inside the closure; caught here it answers
        // Undetermined naming the refusal — never a crash that silences
        // the rest of the file's judging, and never misread as a
        // verdict.
        if is_string {
            let asked = crate::kernel_ask::ask_kernel(|| (kernel.member)(&declared.set, &value.values));
            return match asked {
                Ok(true) => Verdict::Silent,
                Ok(false) => Verdict::Fire(refutation(
                    &spelled_string_word(&value.values),
                    &declared.spelling,
                    &declared.set,
                )),
                Err(_) => Verdict::Undetermined(SENTENCE.kernel_declined_member.to_owned()),
            };
        }
        for v in &value.values {
            let asked = crate::kernel_ask::ask_kernel(|| (kernel.member)(&declared.set, &[*v]));
            match asked {
                Ok(true) => {}
                Ok(false) => {
                    return Verdict::Fire(refutation(
                        &format_py_number(*v, false),
                        &declared.spelling,
                        &declared.set,
                    ));
                }
                Err(_) => {
                    return Verdict::Undetermined(SENTENCE.kernel_declined_member.to_owned());
                }
            }
        }
        return Verdict::Silent;
    }
    if value.kind == Kind::Set {
        let is_string_sorted_set = value.kind_tag == Some(PrimitiveKind::String)
            || (value.kind_tag.is_none() && sequence_shaped_safely(&value.set, kernel));
        let is_numeric_sorted_set = matches!(
            value.kind_tag,
            Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Number)
        ) || (value.kind_tag.is_none() && on_one_tuple_layer(&value.set));
        // The same TUPLE PUN as the Values-side law above applies to a
        // Set-kind value's own sort tag: gated on `states_sequence`/
        // `within_codepoint_door`, never `on_one_tuple_layer` alone —
        // see that law's doc comment for the ported refined-ts-go
        // source.
        // the same `requires_integer` opening as the Values-side law —
        // an explicit Integer form is numeric intent no string set carries
        // — and the same `is_codepoint_alphabet` carve-out inside it (see
        // the Values-side law's doc comment for the collapsed-
        // `SingleCharacter` shape this protects).
        if is_string_sorted_set
            && on_one_tuple_layer(&declared.set)
            && !is_codepoint_alphabet(&declared.set)
            && (requires_integer(&declared.set)
                || states_sequence(&declared.set)
                || !within_codepoint_door(&declared.set, false))
        {
            // The position's own sort, named as precisely as the set
            // states it — the same reading the Values-side law takes.
            let position_said = if requires_integer(&declared.set) {
                "an integer"
            } else {
                "a number"
            };
            return Verdict::Fire(cross_sort_of_value(
                &refined_sets::format_for_diagnostics::format_for_diagnostics(&value.set),
                "a string",
                position_said,
                &declared.spelling,
                &declared.set,
            ));
        }
        if is_numeric_sorted_set && states_sequence(&declared.set) {
            return Verdict::Fire(cross_sort_of_value(
                &refined_sets::format_for_diagnostics::format_for_diagnostics(&value.set),
                "a number",
                "a string",
                &declared.spelling,
                &declared.set,
            ));
        }
        let is_float_sorted = value.kind_tag == Some(PrimitiveKind::Float);
        if is_float_sorted && requires_integer(&declared.set) {
            return Verdict::Fire(cross_sort_of_value(
                &refined_sets::format_for_diagnostics::format_for_diagnostics(&value.set),
                "a float",
                "an integer",
                &declared.spelling,
                &declared.set,
            ));
        }
        // CONTAINMENT-REFUTATION LAW: the checked position IS the claim
        // `flowing ⊆ declared`. `scalar_subset` proves it on the 1-tuple
        // layer (silent on `true`); a decided `false` refutes it —
        // whether by a proved disjoint or a proved overlap, both fire. A
        // REFUSAL (a set shape the kernel's subset decider does not
        // decide — e.g. a concatenation pattern against a length window
        // today) PANICS inside the kernel closure rather than returning a
        // boolean; that panic is caught here and answered as Undetermined
        // naming the refusal — never read as a refutation (refined-ts-go's
        // containedInAsked recover()s the same way).
        //
        // SEQUENCE-SHAPED SETS (a string-literal union, e.g. `anchor_of`'s
        // own `Literal["end", "start", "middle"]` narrowed by a `match` to
        // one member) are never scalar (1-tuple) shaped, so
        // `scalar_subset` always refuses them — `kernelScalarSubset`'s own
        // export fails outright unless BOTH sides are `scalarB`
        // (exports_sets.lean). Either side being sequence-shaped
        // (`sequence_shaped`, this file's own recursive Star/Concatenation/
        // Repeat/RepeatWord/EmptyTuple/Union/Difference test) tries
        // `seq_subset` FIRST — the kernel's own sequence-shape decider
        // (`kernelSeqSubset`'s doc: "a theorem in both directions" for
        // every branch that answers at all; the pattern-placement route's
        // `false` is likewise a proved separating witness, never a bare
        // "no proof found" — only a witness-less search still refuses).
        // `scalar_subset` is tried only as the FALLBACK when `seq_subset`
        // itself refuses (both panics caught the same way), since a set
        // that is sequence-shaped by this file's own test may still sit
        // on the 1-tuple layer too (the tuple pun) and the scalar decider
        // can occasionally settle what the sequence route could not.
        //
        // A NUMERIC repetition (`list[int]`'s own element-read shape, a
        // `Star`/`Repeat`/`RepeatWord`/`EmptyTuple`/`Concatenation` form
        // whose element is NOT codepoints — e.g. `Age`'s own int-sorted
        // element) reads as `sequence_shaped: false` (that recognizer
        // gates a repetition form on `repetition_element_is_codepoints`)
        // and `on_one_tuple_layer: false` too (a repetition form falls to
        // that recognizer's `_ => false` arm) — so it is neither string-
        // sorted nor numeric-sorted by this block's own two set-shape
        // tests above, and the `sequence_question` gate above never
        // fires for it. Left unrouted, this pair falls straight to
        // `scalar_subset`, whose kernel export refuses any non-1-tuple
        // shape outright — the star never reaches a decider that could
        // answer it. `states_sequence` (the POSITIVE, non-recursive top-
        // layer test, sort-blind by construction) catches exactly this:
        // the flowing value stating a repetition form at all, regardless
        // of its element's sort, against a declared side that IS scalar-
        // shaped (`on_one_tuple_layer` — `Age`'s own `[0, 120]` window).
        // `seq_subset` decides this pair today (`rayVsScalarRefuteB`: an
        // unbounded ray read as a sequence's own element domain against a
        // scalar right side) — tried first, with the same fallback-to-
        // `scalar_subset`-on-refusal discipline as the sequence_question
        // gate above.
        let sequence_question =
            sequence_shaped_safely(&value.set, kernel) || sequence_shaped_safely(&declared.set, kernel);
        let numeric_repetition_into_scalar =
            states_sequence(&value.set) && on_one_tuple_layer(&declared.set);
        if sequence_question || numeric_repetition_into_scalar {
            let seq_asked = crate::kernel_ask::ask_kernel(|| (kernel.seq_subset)(&value.set, &declared.set));
            match seq_asked {
                Ok(true) => return Verdict::Silent,
                Ok(false) => {
                    return Verdict::Fire(containment_refutation(
                        &value.set,
                        &declared.spelling,
                        &declared.set,
                    ));
                }
                Err(_) => {}
            }
        }
        let asked = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&value.set, &declared.set));
        return match asked {
            Ok(true) => Verdict::Silent,
            Ok(false) => Verdict::Fire(containment_refutation(
                &value.set,
                &declared.spelling,
                &declared.set,
            )),
            Err(_) => Verdict::Undetermined(SENTENCE.kernel_declined_containment.to_owned()),
        };
    }
    if value.kind == Kind::Null && declared.admits_none {
        return Verdict::Silent;
    }
    if matches!(value.kind, Kind::Object | Kind::List | Kind::Null)
        && scalar_or_string_shaped(&declared.set)
    {
        let value_word = match value.kind {
            Kind::Object => value.kind_word.unwrap_or("a dict"),
            Kind::List => "a list",
            Kind::Null => "None",
            _ => unreachable!("matches! above admits only Object, List, Null"),
        };
        // The structural mismatch names the sort crossing outright: a
        // dict/list/None is neither a number nor a string, so no run
        // reconciles it with a scalar-ground position — the Go twin's
        // "is not allowed here" reason clause rather than a bare
        // "not assignable".
        let position_said = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
            "a number"
        } else {
            "a string"
        };
        return Verdict::Fire(format!(
            "a value of type '{value_word}' is not assignable to type {} — the position states {position_said}, and {value_word} is not allowed here",
            required_words(&declared.spelling, &declared.set),
        ));
    }
    // A KindUnion (json.loads's own honest return space over an opaque
    // string, `expressions.rs::json_loads_value_space` — every JSON
    // shape rides as one arm) judges EACH ARM against the same
    // `declared` refinement, recursively through this same seam: the
    // union claims the runtime value is SOME arm, never which one, so a
    // declared numeric position is escaped the moment ANY arm is not a
    // number — the first Fire among the arms is the verdict, naming
    // that arm's own word (the None/list/dict/str arms already fire via
    // the Null/Object/scalar-ground laws above; a numeric arm is
    // Silent). Any Undetermined arm makes the whole judgment
    // Undetermined, matching the ELEMENT/MEMBERS/POSITIONS laws' own
    // "cannot claim more than the least-determined part knows" rule.
    // All arms Silent (every possible shape fits the declared set) is
    // Silent.
    if value.kind == Kind::KindUnion {
        for arm in &value.arms {
            match judge(arm, declared, kernel) {
                Verdict::Fire(message) => return Verdict::Fire(message),
                Verdict::Undetermined(sentence) => return Verdict::Undetermined(sentence),
                Verdict::Silent => {}
            }
        }
        return Verdict::Silent;
    }
    Verdict::Undetermined(SENTENCE.value_not_readable.to_owned())
}
