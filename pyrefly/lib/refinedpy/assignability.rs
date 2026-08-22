/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The one judging seam: a flowing value against a declared refinement.
//! Every sink (annotated assignment, argument, return, field write)
//! routes here, so fire wording, silence, and undetermined sentences
//! stay uniform. This file is the contract the walk calls; the
//! assignability unit fills it in behind these signatures.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::is_string_ground;
use refined_sets::format_string_shapes::format_py_number;
use refined_sets::refinement_forms::on_one_tuple_layer;
use refined_sets::refinement_forms::requires_integer;

use crate::refinedpy::diagnostic_sentences::at_index;
use crate::refinedpy::diagnostic_sentences::at_key;
use crate::refinedpy::diagnostic_sentences::containment_refutation;
use crate::refinedpy::diagnostic_sentences::cross_sort_of_value;
use crate::refinedpy::diagnostic_sentences::refutation;
use crate::refinedpy::diagnostic_sentences::required_words;
use crate::refinedpy::diagnostic_sentences::SENTENCE;
use crate::refinedpy::typereading::DeclaredRefinement;

/// What judging one value against one declared set concluded.
pub enum Verdict {
    /// The value is provably outside the set — the message is the full
    /// diagnostic text.
    Fire(String),
    /// The value is provably inside the set.
    Silent,
    /// The walk could not read enough to judge — the sentence names
    /// what blocked, in plain per-position prose.
    Undetermined(String),
}

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
                        return Verdict::Fire(at_index(&message, index));
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
                        return Verdict::Fire(at_key(&message, member_name));
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
        if is_string
            && on_one_tuple_layer(&declared.set)
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
            let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (kernel.member)(&declared.set, &value.values)
            }));
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
            let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (kernel.member)(&declared.set, &[*v])
            }));
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
            || (value.kind_tag.is_none() && sequence_shaped(&value.set));
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
        if is_string_sorted_set
            && on_one_tuple_layer(&declared.set)
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
        let sequence_question = sequence_shaped(&value.set) || sequence_shaped(&declared.set);
        let numeric_repetition_into_scalar =
            states_sequence(&value.set) && on_one_tuple_layer(&declared.set);
        if sequence_question || numeric_repetition_into_scalar {
            let seq_asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (kernel.seq_subset)(&value.set, &declared.set)
            }));
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
        let asked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            (kernel.scalar_subset)(&value.set, &declared.set)
        }));
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

/// Whether the declared set names scalars or strings — the shapes a
/// dict/list/None/opaque value can NEVER inhabit. Three recognizers:
/// numeric 1-tuple forms (`on_one_tuple_layer`), the full string ground
/// (`is_string_ground`), and SEQUENCE-SHAPED sets (every top-level form
/// is a string/sequence form — EmptyTuple/Concatenation/Star/Repeat/
/// RepeatWord — or a Union/Difference of sequence-shaped operands),
/// which is what a `Literal["a", "b"]` union of string tuples compiles
/// to. A set none of the three recognize declines the structural laws
/// and falls through to the general undetermined answer.
fn scalar_or_string_shaped(set: &refined_sets::refinement_forms::RefinedSet) -> bool {
    on_one_tuple_layer(set) || is_string_ground(set) || sequence_shaped(set)
}

/// A `Star`/`Repeat`/`RepeatWord` form's own element sits inside the
/// codepoint alphabet — this crate's grammar reuses `Star`/`Repeat` for a
/// NUMERIC element too (`check.rs::seed_parameters`'s `list[int]`/
/// `set[int]`/`Sequence[int]` parameter seed, `Form::Star(int's own
/// set)`), so a bare repetition form is string-shaped only when its
/// element demonstrably IS codepoints, never merely because it wears one
/// of these forms.
///
/// Two spellings pass: `is_character` — the element IS the whole
/// alphabet exactly (a plain `list[str]`/`Sequence[str]` element seed) —
/// or `within_codepoint_door` — the element is a NARROWER codepoint-only
/// subset, the shape a `.regex(...)`-compiled character class actually
/// produces (`regex_compiler.rs`'s `code_range`/`character_class`:
/// `[a-z]` compiles to `Integer ∧ AtLeast(0x61) ∧ AtMost(0x7A)`, an
/// `Integer`-window wholly inside the codepoint alphabet, never the full
/// alphabet itself). Before this second spelling, a `Repeat`/`Star` over
/// a narrowed class (`LabelPattern = Annotated[str,
/// Field(pattern=r"^[a-z]{3,8}$")]`'s own compiled `Repeat`) read as
/// NEITHER string- nor number-shaped by this file's own two tests, so
/// `judge`'s `sequence_question` gate never tried `seq_subset` at all and
/// fell straight to `scalar_subset`, which the kernel refuses outright
/// for a non-1-tuple shape — the undetermined `g-strings-and-formats.py`
/// row this fixes (`kernel_declined_containment`, RTS7002).
fn repetition_element_is_codepoints(form: &refined_sets::refinement_forms::Refinement) -> bool {
    let one = refined_sets::refinement_forms::make_refined_set(vec![form.clone()]);
    match refined_sets::repetition_window_forms::as_repetition(&one) {
        Some(repeated) => {
            refined_sets::format_string_shapes::is_character(&repeated.element)
                || within_codepoint_door(&repeated.element, false)
        }
        None => false,
    }
}

pub(crate) fn sequence_shaped(set: &refined_sets::refinement_forms::RefinedSet) -> bool {
    use refined_sets::refinement_forms::Form;
    !set.forms.is_empty()
        && set.forms.iter().all(|form| match form.form {
            // EmptyTuple/Concatenation carry no separate "element sort" of
            // their own (an EmptyTuple names no element at all; a
            // Concatenation's operands are themselves nested sets this
            // crate only ever builds over codepoints — `string_tuple`'s
            // own encoding) — string-shaped exactly as before.
            Form::EmptyTuple | Form::Concatenation => true,
            Form::Star | Form::Repeat | Form::RepeatWord => repetition_element_is_codepoints(form),
            Form::Union | Form::Difference => {
                form.a_.as_deref().map(sequence_shaped).unwrap_or(false)
                    && form.b.as_deref().map(sequence_shaped).unwrap_or(false)
            }
            Form::AtLeast
            | Form::Above
            | Form::AtMost
            | Form::Below
            | Form::Integer
            | Form::MultipleOf
            | Form::OneOf => false,
        })
}

/// Whether a set's OWN top-level forms DEMONSTRABLY state a sequence —
/// a `Star`/`Concatenation`/`Repeat`/`RepeatWord`/`EmptyTuple` form
/// sits among them. Ported from refined-ts-go's `StatesSequence`
/// (walk/sequence_measures.go): a POSITIVE, non-recursive test — unlike
/// `sequence_shaped` above (which requires EVERY form, recursing
/// through Union/Difference, and serves the Object/List/Null
/// structural-mismatch law), `states_sequence` only asks whether the
/// set's own top layer carries a sequence form at all, and is what
/// gates the string-vs-numeric-ground SORT laws below: `on_one_tuple_
/// layer` alone cannot tell a numeric one-of from a union of
/// single-character string tuples (the tuple pun — `string_tuple`'s
/// length-1 encoding is bare `OneOf`, no `Concatenation` wrapper), so
/// the sort law must see an actual sequence form before it may read
/// "on the one-tuple layer" as "numeric."
pub(crate) fn states_sequence(set: &refined_sets::refinement_forms::RefinedSet) -> bool {
    use refined_sets::refinement_forms::Form;
    set.forms.iter().any(|form| {
        matches!(
            form.form,
            Form::Star | Form::Concatenation | Form::Repeat | Form::RepeatWord | Form::EmptyTuple
        )
    })
}

/// One admitted scalar that could be a one-character string's own
/// codepoint — ported from refined-ts-go's `CodepointScalar`
/// (walk/sequence_measures.go): a natural number inside the codepoint
/// alphabet (`codepoint_sets::codepoints`'s own surrogate-gap-excluding
/// range), never a negative, fractional, or out-of-range value.
fn codepoint_scalar(v: f64) -> bool {
    v == v.trunc() && v >= 0.0 && (v <= 0xD7FF as f64 || (v >= 0xE000 as f64 && v <= 0x10FFFF as f64))
}

/// Whether EVERY value a scalar set admits sits inside the codepoint
/// alphabet — ported from refined-ts-go's `WithinCodepointDoor`
/// (walk/sequence_measures.go): such a set is indistinguishable from a
/// union of one-character strings, so the string-vs-numeric sort laws
/// must not refute a string value against it on shape alone. Two
/// spellings pass: enumerated codepoints (`OneOf`), and
/// INTEGER-constrained windows wholly inside one side of the surrogate
/// gap (`Field(pattern=r"^[\x00-\x7f]$")`'s own shape). Windows without
/// the `Integer` form answer false (they admit non-codepoint reals),
/// and the window test is conservative (`Above`/`Below` widen to their
/// closed bounds) so the door never opens wrongly. `integer_inherited`
/// carries the ancestor's own `Integer` form down through a `Union`
/// (the same recursion refined-ts-go's Go source takes), since a bound
/// form nested under a `Union` reads its sort from the branch that
/// states it, not from its own immediate siblings.
fn within_codepoint_door(
    set: &refined_sets::refinement_forms::RefinedSet,
    integer_inherited: bool,
) -> bool {
    use refined_sets::refinement_forms::Form;
    if set.forms.is_empty() {
        return false;
    }
    let mut integer = integer_inherited;
    if !integer {
        integer = set.forms.iter().any(|form| form.form == Form::Integer);
    }
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    let mut content = false;
    for form in &set.forms {
        match form.form {
            Form::Integer => {}
            Form::OneOf => {
                if !form.w.iter().all(|&w| codepoint_scalar(w)) {
                    return false;
                }
                content = true;
            }
            Form::AtLeast | Form::Above => lo = lo.max(form.a),
            Form::AtMost | Form::Below => hi = hi.min(form.a),
            Form::Union => {
                let a = form.a_.as_deref();
                let b = form.b.as_deref();
                let a_ok = a.map(|s| within_codepoint_door(s, integer)).unwrap_or(false);
                let b_ok = b.map(|s| within_codepoint_door(s, integer)).unwrap_or(false);
                if !a_ok || !b_ok {
                    return false;
                }
                content = true;
            }
            _ => return false,
        }
    }
    if lo != f64::NEG_INFINITY || hi != f64::INFINITY {
        if !integer || lo > hi {
            return false;
        }
        let in_low = lo >= 0.0 && hi <= 0xD7FF as f64;
        let in_high = lo >= 0xE000 as f64 && hi <= 0x10FFFF as f64;
        if !in_low && !in_high {
            return false;
        }
        content = true;
    }
    content
}

/// The readable spelling of a string word for a fire message: the code
/// points decoded back to text and JSON-quoted, the same spelling
/// `format_string_shapes::format_string_literal` gives a set's own
/// literal chain. Falls back to the Python `repr`-style bare digits
/// only if the points sit outside the representable scalar range (an
/// honest label rather than a silent drop) — `from_points` returns
/// `None` there.
fn spelled_string_word(points: &[f64]) -> String {
    refined_sets::format_string_shapes::from_points(points)
        .unwrap_or_else(|| format!("{:?}", points))
}

/// The readable spelling of a Boolean-tagged value for a fire message:
/// the Python literal `True`/`False`, never the bare `1`/`0` a numeric
/// spelling would give (`format_py_number` reads the sort tag, not the
/// PRODUCT-LAW distinction this file's Boolean law exists to state).
/// Falls back to the bare digit only for the unreached case of an
/// empty or multi-valued Boolean word (a boolean is always exactly one
/// value, `expressions.rs`'s own `BooleanLiteral` encoding) — an honest
/// label rather than a silent drop.
fn spelled_boolean_word(values: &[f64]) -> String {
    match values {
        [v] if *v == 1.0 => "True".to_owned(),
        [v] if *v == 0.0 => "False".to_owned(),
        _ => format!("{:?}", values),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use refined_domain::abstract_value::SetKindTag;
    use refined_domain::abstract_value::known_set;
    use refined_domain::abstract_value::known_values;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::at_most;
    use refined_sets::refinement_forms::below;
    use refined_sets::refinement_forms::integer;
    use refined_sets::refinement_forms::make_refined_set;

    use super::*;

    /// A kernel handle for tests that ask it — same skip-when-unbuilt
    /// pattern check.rs and expressions.rs already use, so this file's
    /// tests run without requiring `pnpm kernel:native` first.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// `type Age = Annotated[int, Field(ge=0, le=120)]` — an int-sorted
    /// alias, the shape surface.rs's annotated_expression_set builds.
    fn age_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
            spelling: "Age".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// `type OptionalAge = Age | None` — the same int-sorted ray, but
    /// the declaration admits absence.
    fn optional_age_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
            spelling: "Age | None".to_owned(),
            admits_none: true,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    fn fire_message(verdict: Verdict) -> String {
        match verdict {
            Verdict::Fire(message) => message,
            Verdict::Silent => panic!("expected Fire, got Silent"),
            Verdict::Undetermined(sentence) => panic!("expected Fire, got Undetermined({sentence})"),
        }
    }

    /// `x: Age = 30.0` fires — Age is int-sorted, and 30.0 is
    /// Float-tagged, so the sort law fires even though the real value
    /// 30 sits inside [0, 120]. The message spells the number the
    /// Python way: "30.0" keeps its trailing ".0", never bare "30".
    #[test]
    fn a_float_tagged_whole_value_into_an_int_sorted_alias_fires_spelled_with_its_dot_zero() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let thirty_float = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
        let message = fire_message(judge(&thirty_float, &declared, &kernel));
        assert!(message.contains("'30.0'"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
    }

    /// `x: Age = 30` (Integer-tagged) is silent — the ordinary kernel
    /// membership path, no sort law involved.
    #[test]
    fn an_integer_tagged_in_range_value_into_an_int_sorted_alias_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let thirty_int = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
        assert!(matches!(judge(&thirty_int, &declared, &kernel), Verdict::Silent));
    }

    /// `6 / 3` evaluates to a Float-tagged 2.0 (Python's `/` is always
    /// true division, expressions.rs's own pinned test) — assigned into
    /// an int-sorted alias, THAT Float tag is what makes this fire, not
    /// the real value 2 being out of range.
    #[test]
    fn true_division_of_two_ints_still_fires_into_an_int_sorted_alias() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let two_float = crate::refinedpy::expressions::binary_arithmetic_value(
            ruff_python_ast::Operator::Div,
            &known_values(vec![6.0], PrimitiveKind::Integer, TrustProved),
            &known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
        );
        assert_eq!(two_float.kind_tag, Some(PrimitiveKind::Float));
        let message = fire_message(judge(&two_float, &declared, &kernel));
        assert!(message.contains("'2.0'"), "{message}");
    }

    /// A declared Float-sorted alias (no `integer()` form) never fires
    /// the sort law — the int-sort gate is specific to a declared set
    /// that actually carries the `int` form.
    #[test]
    fn a_float_tagged_value_into_a_float_sorted_alias_never_hits_the_sort_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = DeclaredRefinement {
            set: make_refined_set(vec![at_least(0.0)]),
            spelling: "Weight".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let thirty_float = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
        assert!(matches!(judge(&thirty_float, &declared, &kernel), Verdict::Silent));
    }

    /// `type Label = Annotated[str, Field(max_length=8)]` — a bounded
    /// string alias: the codepoint alphabet repeated, capped at 8. Not
    /// the full string GROUND (`is_string_ground` requires an
    /// unbounded repetition), so a String-tagged value against it flows
    /// past both sort laws to the ordinary kernel membership ask.
    fn label_refinement() -> DeclaredRefinement {
        use refined_sets::codepoint_sets::codepoints;
        use refined_sets::refinement_forms::repeat_of;
        DeclaredRefinement {
            set: make_refined_set(vec![repeat_of(codepoints(), 0, Some(8))]),
            spelling: "Label".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// `type AnyString = Annotated[str, Field()]` — the bare string
    /// GROUND itself (`z.string()`'s Python twin, unbounded): one star
    /// over the codepoint alphabet, the exact shape `is_string_ground`
    /// recognizes.
    fn any_string_refinement() -> DeclaredRefinement {
        use refined_sets::codepoint_sets::strings;
        DeclaredRefinement {
            set: strings(),
            spelling: "AnyString".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// `x: Label = "hi"` — a whole String-tagged word asked ONCE against
    /// the alias, silent because "hi" (2 code points) sits under the
    /// 8-character ceiling.
    #[test]
    fn a_string_value_member_of_a_string_set_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = label_refinement();
        let value = known_values(hi_points("hi"), PrimitiveKind::String, TrustProved);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// `x: Label = "too-long-string"` (15 code points, over the 8-char
    /// ceiling) fires ONE membership question over the whole word, and
    /// the message quotes the string readably rather than spelling code
    /// points.
    #[test]
    fn a_string_value_not_a_member_of_a_string_set_fires_quoting_the_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = label_refinement();
        let value = known_values(
            hi_points("too-long-string"),
            PrimitiveKind::String,
            TrustProved,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("\"too-long-string\""), "{message}");
        assert!(message.contains("'Label'"), "{message}");
    }

    /// `type ChartLayout = Literal["horizontal", "vertical", "centric",
    /// "radial"]` — c-reads-and-values.py:1182's own alias, the UNION of
    /// four singleton string tuples `typereading::string_literal_set`
    /// builds. Untagged `Kind::Set` (`set_kind_tag: SetKindTag::None`)
    /// reads as string-sorted by convention (ORIENTATION.md's own
    /// recognition-slice fact) — no kind_tag field on a `RefinedSet`.
    fn chart_layout_refinement() -> DeclaredRefinement {
        use refined_sets::codepoint_sets::string_tuple;
        use refined_sets::refinement_forms::union;
        let set = make_refined_set(vec![union(
            make_refined_set(vec![union(
                make_refined_set(vec![union(string_tuple("horizontal"), string_tuple("vertical"))]),
                string_tuple("centric"),
            )]),
            string_tuple("radial"),
        )]);
        DeclaredRefinement {
            set,
            spelling: "ChartLayout".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// `c-reads-and-values.py:1197`'s HELD arm — `layout` narrowed to
    /// `"horizontal"` (a String-tagged whole word) against
    /// `Literal["horizontal", "vertical", "centric", "radial"]`: ONE
    /// membership ask over the whole word (line 208's `is_string` arm),
    /// silent because "horizontal" is one of the four tuples the union
    /// spells.
    #[test]
    fn a_literal_union_member_string_value_is_silent_via_whole_word_membership() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = chart_layout_refinement();
        let value = known_values(hi_points("horizontal"), PrimitiveKind::String, TrustProved);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// The mirror: a String-tagged whole word NOT among the union's four
    /// tuples fires the ordinary kernel membership ask, quoting the
    /// string readably.
    #[test]
    fn a_literal_union_non_member_string_value_fires_quoting_the_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = chart_layout_refinement();
        let value = known_values(hi_points("diagonal"), PrimitiveKind::String, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("\"diagonal\""), "{message}");
        assert!(message.contains("'ChartLayout'"), "{message}");
    }

    /// c-reads-and-values.py:1199's own shape: `return None` under a
    /// declared `Literal["horizontal", "vertical"]` — NOT `| None`
    /// wrapped, so `declared.admits_none` is false. `Kind::Null` reaches
    /// A Literal union of specific string tuples is SEQUENCE-SHAPED
    /// (`sequence_shaped`: a Union of Concatenation forms), so the
    /// structural-mismatch law recognizes it and `None` against a
    /// non-admitting Literal union FIRES — None is provably not a
    /// string of any spelling (c-reads-and-values.py's fall-through-
    /// to-None row).
    #[test]
    fn none_against_a_literal_union_that_does_not_admit_none_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let two_member_declared = DeclaredRefinement {
            set: make_refined_set(vec![refined_sets::refinement_forms::union(
                refined_sets::codepoint_sets::string_tuple("horizontal"),
                refined_sets::codepoint_sets::string_tuple("vertical"),
            )]),
            spelling: "Literal['horizontal', 'vertical']".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let value = refined_domain::abstract_value::null_value();
        let Verdict::Fire(message) = judge(&value, &two_member_declared, &kernel) else {
            panic!("None against a non-admitting string-Literal union fires the structural law");
        };
        assert!(message.contains("None"), "{message}");
    }

    /// A String-tagged value against a NUMERIC-ground alias (Age, an
    /// int-sorted ray) still fires — but via the ORDINARY whole-word
    /// kernel membership ask, not the sort law: `Age`'s own range
    /// `[0, 120]` sits WITHIN the codepoint door (every value it admits
    /// is a valid single codepoint), so the sort law declines per the
    /// tuple-pun gate (`within_codepoint_door`) and falls through.
    /// `"30"` is a 2-CODEPOINT tuple, never a member of `Age`'s
    /// 1-tuple-shaped set regardless, so the kernel's own derivative
    /// walk refutes it — the fire message is identical either way, this
    /// test only pins that the value is still refused.
    #[test]
    fn a_string_value_into_a_numeric_ground_alias_still_fires_via_the_kernel_ask() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_values(hi_points("30"), PrimitiveKind::String, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
    }

    /// The TUPLE-PUN fix's own pin: `"B"` (a String-tagged whole word)
    /// into `Grade = Literal["A", "B", "C"]` — a `Union` of
    /// single-codepoint `OneOf`s (`surface.rs`'s `string_literal_set`
    /// over 1-character members) — is SILENT. Before the fix, the
    /// string-vs-numeric-ground sort law read `Grade`'s shape as
    /// numeric-ground (`on_one_tuple_layer` alone, blind to the
    /// single-character tuple pun) and fired outright on every real
    /// member; `Grade`'s own range sits wholly inside the codepoint
    /// door (every member is a valid codepoint) with no sequence form
    /// present, so the law now declines and the ordinary whole-word
    /// kernel membership ask decides it correctly.
    #[test]
    fn a_single_character_literal_union_member_is_silent_not_the_numeric_ground_sort_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = grade_refinement();
        let value = known_values(hi_points("B"), PrimitiveKind::String, TrustProved);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// The mirror: `"F"` (outside `Grade`'s three members) fires via the
    /// ordinary whole-word kernel ask, quoting the string readably —
    /// never the numeric-ground sort law's own wording.
    #[test]
    fn a_single_character_literal_union_non_member_fires_quoting_the_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = grade_refinement();
        let value = known_values(hi_points("F"), PrimitiveKind::String, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("\"F\""), "{message}");
        assert!(message.contains("'Grade'"), "{message}");
    }

    /// `type Grade = Literal["A", "B", "C"]` — o-grammar-refinements.py's
    /// own alias, `surface.rs`'s `literal_alias_set`/
    /// `string_literal_set`'s exact fold: `union(union(string_tuple("A"),
    /// string_tuple("B")), string_tuple("C"))`, every member a single
    /// character so every `string_tuple` call is a bare `OneOf` (no
    /// `Concatenation` wrapper for a length-1 word).
    fn grade_refinement() -> DeclaredRefinement {
        use refined_sets::codepoint_sets::string_tuple;
        use refined_sets::refinement_forms::union;
        let set = make_refined_set(vec![union(
            make_refined_set(vec![union(string_tuple("A"), string_tuple("B"))]),
            string_tuple("C"),
        )]);
        DeclaredRefinement {
            set,
            spelling: "Grade".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// The mirror: an Integer-tagged value against the STRING-ground
    /// alias fires the sort law — a number is never a member of a
    /// string-ground set, regardless of whether its real value would
    /// pass a bare membership ask.
    #[test]
    fn a_numeric_value_into_a_string_ground_alias_fires_the_sort_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = any_string_refinement();
        let value = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'AnyString'"), "{message}");
    }

    /// A dict (`Kind::Object`) can never be a member of a numeric-ground
    /// declared set — fires outright, never undetermined.
    #[test]
    fn a_dict_value_into_a_numeric_ground_alias_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = refined_domain::known_constructors::known_object(
            Vec::new(),
            Default::default(),
            false,
            TrustProved,
            false,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.to_lowercase().contains("dict"), "{message}");
    }

    /// A list (`Kind::List`) can never be a member of a numeric-ground
    /// declared set — fires outright.
    #[test]
    fn a_list_value_into_a_numeric_ground_alias_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = refined_domain::known_constructors::known_list(Vec::new(), TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
    }

    /// `return None` under a plain (non-`Optional`) declared set
    /// (`declared.admits_none == false`) fires outright — `Kind::Null`
    /// is this crate's representation of Python's `None`, and a plain
    /// declaration never admits it.
    #[test]
    fn none_into_a_plain_declaration_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        assert!(!declared.admits_none);
        let value = refined_domain::abstract_value::null_value();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.to_lowercase().contains("none"), "{message}");
    }

    /// `return None` under an `Optional[Age]`/`Age | None` declared set
    /// (`declared.admits_none == true`) is silent — the admitted
    /// absence is in the declaration, so `None` is a member.
    #[test]
    fn none_into_an_admits_none_declaration_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = optional_age_refinement();
        assert!(declared.admits_none);
        let value = refined_domain::abstract_value::null_value();
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// The code points a string literal spells, reused across the
    /// string-ground tests so each stays a one-line construction over
    /// `string_models.rs`'s own encoding (one code point per char).
    fn hi_points(text: &str) -> Vec<f64> {
        text.chars().map(|c| c as u32 as f64).collect()
    }

    /// The OPAQUE law: a function object (opaque_value's first-cited
    /// kind) against Age (numeric-ground) fires with the honest word,
    /// never "a dict" — no kernel ask needed, the same short-circuit
    /// the sort laws take.
    #[test]
    fn an_opaque_function_value_into_a_numeric_ground_alias_fires_with_its_word() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = refined_domain::abstract_value::opaque_value("a function value");
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("a function value"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
    }

    /// The honest JSON-union `json.loads` answers over an opaque string
    /// (`expressions.rs::json_loads_value_space`, ISSUES.md b-runners:124)
    /// judged against a numeric-ground alias FIRES, naming the first
    /// non-numeric arm (`None`, this union's own first arm) — the
    /// honest verdict for an opaque payload, since the union claims the
    /// runtime value is SOME arm and a JSON `null` genuinely escapes an
    /// `int`-sorted position. Built inline with the same seven arms
    /// `json_loads_value_space` builds, mirroring the isinstance
    /// narrowing test's own construction (narrowing.rs).
    #[test]
    fn a_json_loads_union_into_a_numeric_ground_alias_fires_naming_the_non_numeric_arm() {
        use refined_domain::abstract_value::float_sorted_unknown;
        use refined_domain::abstract_value::kind_union_of;
        use refined_domain::abstract_value::null_value;
        use refined_domain::abstract_value::opaque_value;
        use refined_sets::codepoint_sets::strings;

        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let integer_arm = AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]), None, TrustProved, SetKindTag::None)
        };
        let union = kind_union_of(vec![
            null_value(),
            known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustProved),
            known_set(strings(), None, TrustProved, SetKindTag::None),
            integer_arm,
            float_sorted_unknown(),
            opaque_value("a list"),
            opaque_value("a dict"),
        ]);
        assert_eq!(union.kind, Kind::KindUnion);
        let message = fire_message(judge(&union, &declared, &kernel));
        assert!(message.contains("None"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
    }

    /// The mirror: an opaque value against the STRING-ground alias fires
    /// too — a function is never a member of a string-ground set either.
    #[test]
    fn an_opaque_value_into_a_string_ground_alias_fires_with_its_word() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = any_string_refinement();
        let value = refined_domain::abstract_value::opaque_value("a caught exception");
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("a caught exception"), "{message}");
        assert!(message.contains("'AnyString'"), "{message}");
    }

    /// An opaque value against a declared set that is NOT scalar-ground
    /// (neither numeric- nor string-ground) declines the opaque law and
    /// falls through to the general undetermined answer — the same
    /// decline the Object/List/Null law already takes for a non-scalar
    /// declared set.
    #[test]
    fn an_opaque_value_into_a_non_scalar_ground_alias_is_undetermined() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = DeclaredRefinement {
            set: make_refined_set(Vec::new()),
            spelling: "Anything".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let value = refined_domain::abstract_value::opaque_value("a function value");
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Undetermined(_)));
    }

    /// The STRING-SORTED-SET law declines for a target whose own range
    /// sits WITHIN THE CODEPOINT DOOR: an UNTAGGED Set holding the full
    /// string ground (`kind_tag: None`, the exact shape `expressions.rs`'s
    /// `__name__` read carries — `known_set(strings(), None, TrustSpec,
    /// SetKindTag::None)`) against Age FIRES via the sort law: `Age`
    /// carries the explicit `Integer` form, which is numeric INTENT by
    /// construction (no string set ever builds one), so the
    /// `requires_integer` opening decides the sort mismatch even though
    /// Age's `[0, 120]` range sits inside the codepoint door — the
    /// d-module-surface row's own expectation ("a host-defined string is
    /// not in an int-sorted set").
    #[test]
    fn an_untagged_string_shaped_set_into_an_integer_formed_alias_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::codepoint_sets::strings;
        let declared = age_refinement();
        let value = known_set(strings(), None, TrustProved, SetKindTag::None);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
    }

    /// The same law, explicitly String-tagged: a Set carrying `kind_tag:
    /// Some(PrimitiveKind::String)` against Age fires identically — the
    /// law reads either the explicit tag or the untagged-Set convention,
    /// and Age's explicit `Integer` form opens the sort law for both.
    #[test]
    fn a_string_tagged_set_into_an_integer_formed_alias_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::codepoint_sets::strings;
        let declared = age_refinement();
        let value = AbstractValue {
            kind_tag: Some(PrimitiveKind::String),
            ..known_set(strings(), None, TrustProved, SetKindTag::None)
        };
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
    }

    /// The tuple-pun gate's own Set-kind pin: an EXPLICITLY String-tagged
    /// Set built from a single-codepoint `OneOf` (`{66}`, "B") against
    /// `Grade` (single-codepoint `OneOf`/`Union` forms only,
    /// `on_one_tuple_layer` true, no demonstrable sequence form) reaches
    /// the CONTAINMENT ask rather than the sort law, because `Grade`
    /// sits wholly inside the codepoint door — `{66}` IS `scalar_subset`
    /// of `Grade`'s set (both are scalar/1-tuple shaped here), so this
    /// is Silent, not a sort-law Fire. (An UNTAGGED bare `OneOf` Set
    /// reads as NUMERIC-sorted by the codebase's own convention — this
    /// test tags String explicitly so it exercises the string-sorted
    /// branch, mirroring `a_string_tagged_set_into_a_codepoint_door_
    /// alias_is_undetermined` above but with a value that IS
    /// scalar-shaped, so the containment ask decides rather than
    /// refuses.)
    #[test]
    fn a_single_codepoint_string_tagged_set_wholly_inside_a_single_character_literal_union_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::refinement_forms::one_of;
        let declared = grade_refinement();
        let value = AbstractValue {
            kind_tag: Some(PrimitiveKind::String),
            ..known_set(make_refined_set(vec![one_of(&[66.0])]), None, TrustProved, SetKindTag::None)
        };
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// The mirror: a NUMERIC-sorted Set (an untagged Set whose own set is
    /// `on_one_tuple_layer`, e.g. the bare `integer()` line) against the
    /// STRING-ground alias fires the sort law before any kernel ask — a
    /// number is never a member of a string-ground set, regardless of
    /// which real numbers the set admits.
    #[test]
    fn a_numeric_shaped_set_into_a_string_ground_alias_fires_the_sort_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = any_string_refinement();
        let value = known_set(make_refined_set(vec![integer()]), None, TrustProved, SetKindTag::None);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'AnyString'"), "{message}");
    }

    /// A NUMERIC star (`list[int]`'s own element-read shape,
    /// `Form::Star(int's set)` — `refinedpy::collection_models::
    /// subscript_read`'s new `Kind::Set` arm hands this exact shape back
    /// for `ages[0]`) against `Age` must NOT take the string-sort law:
    /// before `sequence_shaped` learned to check the star's own alphabet,
    /// ANY `Form::Star` read as string-shaped regardless of its element,
    /// which would have wrongly fired "a string-sorted value is never in
    /// an int-sorted set" here even though the element is a whole number.
    /// The correct path is the CONTAINMENT law: the unbounded int ray is
    /// not a subset of Age's `[0, 120]` window, so this fires the
    /// CONTAINMENT message instead.
    #[test]
    fn a_numeric_star_shaped_set_into_age_fires_containment_not_the_sort_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let whole_ints = make_refined_set(vec![integer(), refined_sets::refinement_forms::at_least(f64::NEG_INFINITY)]);
        let numeric_star = make_refined_set(vec![refined_sets::refinement_forms::star(whole_ints)]);
        let value = known_set(numeric_star, None, TrustProved, SetKindTag::None);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(
            message.contains("admits values outside"),
            "must fire the CONTAINMENT message, not the string-sort law: {message}"
        );
    }

    /// The FLOAT-SORT SET law: a Float-sorted Set (`float_sorted_unknown`
    /// — the shape `math.sqrt`'s result carries) against Age (int-sorted)
    /// fires — a float-sorted value is never a member of an int-sorted
    /// set, regardless of what real numbers the set admits.
    #[test]
    fn a_float_sorted_set_into_an_int_sorted_alias_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = refined_domain::abstract_value::float_sorted_unknown();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.to_lowercase().contains("float"), "{message}");
    }

    /// CONTAINMENT-REFUTATION LAW, the overlap case: a Float-sorted Set
    /// against a float-TOLERANT (non-integer-sorted) declared set skips
    /// the sort law (specific to `requires_integer`) and falls to the
    /// ordinary Set path — R-bar (`float_sorted_unknown`'s set, the
    /// whole real line) is NOT a subset of `Weight`'s `[0, ∞)` ray (it
    /// admits negatives the declared set excludes) and is NOT disjoint
    /// from it either (they overlap on `[0, ∞)`). Before this law, that
    /// overlap sat Undetermined; the law now fires it, because
    /// `scalar_subset` proving false over decided scalar forms IS a
    /// refutation of the checked position's containment claim, whether
    /// the two sets are disjoint or merely overlapping.
    #[test]
    fn a_float_sorted_set_overlapping_a_non_integer_sorted_alias_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = DeclaredRefinement {
            set: make_refined_set(vec![at_least(0.0)]),
            spelling: "Weight".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let value = refined_domain::abstract_value::float_sorted_unknown();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Weight'"), "{message}");
        assert!(message.contains("admits values outside"), "{message}");
    }

    /// CONTAINMENT-REFUTATION LAW, the subset case: an int-sorted Set
    /// `[10, 20]` (wholly inside Age's `[0, 120]` window) is still
    /// Silent — `scalar_subset` proves the containment claim outright,
    /// unchanged by this law.
    #[test]
    fn an_int_sorted_set_wholly_inside_the_declared_window_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_set(
            make_refined_set(vec![integer(), at_least(10.0), at_most(20.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// CONTAINMENT-REFUTATION LAW, the decided-disjoint case: an
    /// int-sorted Set entirely below Age's floor (`< 0`) still fires —
    /// `scalar_disjoint` proves no member of either set can ever be the
    /// other's, the sharpest form of refutation the law covers.
    #[test]
    fn an_int_sorted_set_disjoint_from_the_declared_window_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_set(
            make_refined_set(vec![integer(), below(0.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("admits values outside"), "{message}");
    }

    /// CONTAINMENT-REFUTATION LAW, the overlap case (int-sort whole set
    /// vs Age window): the unrestricted integer line is NOT a subset of
    /// Age's `[0, 120]` window (it admits negatives and values above
    /// 120) and NOT disjoint from it either (10 is a member of both).
    /// Before this law, that overlap sat Undetermined; the law now fires
    /// it — `scalar_subset` proving false over decided scalar forms is a
    /// refutation of the checked position's containment claim regardless
    /// of whether the two sets are disjoint or merely overlapping.
    #[test]
    fn an_int_sort_whole_set_overlapping_the_age_window_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_set(
            make_refined_set(vec![integer()]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("admits values outside"), "{message}");
    }

    /// CONTAINMENT REFUTATION, the sequence case: the kernel's
    /// `seq_subset` decider now DECIDES the `strings()`-vs-length-window
    /// pair (the sequence-containment fragment grew past the earlier
    /// refusal this test used to pin — its old panic message, "subset is
    /// decided for scalar and sequence shapes today," is no longer
    /// reachable for this shape). `strings()` is the full, UNBOUNDED
    /// codepoint ground; `Label`'s declared set caps length at 8
    /// (`repeat_of(codepoints(), 0, Some(8))`), so the unbounded set is
    /// never a subset of the capped one — `seq_subset` proves `false`,
    /// a decided refutation, and `judge` fires the CONTAINMENT-REFUTATION
    /// message ("the flowing set admits values outside the declared
    /// set").
    #[test]
    fn an_unbounded_string_set_against_a_max_length_window_fires_containment() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::codepoint_sets::strings;
        let declared = label_refinement();
        let value = known_set(strings(), None, TrustProved, SetKindTag::None);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Label'"), "{message}");
        assert!(message.contains("admits values outside"), "{message}");
    }

    /// The SEQ_SUBSET ROUTING law's own pin: `e-class-and-function.py:610`'s
    /// shape — a narrowed union of specific string members
    /// (`or_narrowed_branch_call`'s own `Literal["insideStart",
    /// "insideEnd", "end"]`, `sequence_shaped` true — each member's own
    /// multi-character `string_tuple` is a `Concatenation`, not the
    /// tuple-pun bare `OneOf` `Grade`'s single-character members build)
    /// flowing into a declared set that admits every one of those members
    /// PLUS more (`position_label`'s own four-member
    /// `Literal["insideStart", "insideEnd", "end", "outside"]`) is a
    /// genuine subset. Before this law, `scalar_subset` refused the pair
    /// outright (neither side is 1-tuple/scalar shaped) and the row sat
    /// Undetermined; `seq_subset` decides it Silent.
    #[test]
    fn a_narrowed_string_literal_union_subset_of_a_wider_one_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::codepoint_sets::string_tuple;
        use refined_sets::refinement_forms::union;
        let narrowed = make_refined_set(vec![union(
            make_refined_set(vec![union(string_tuple("insideStart"), string_tuple("insideEnd"))]),
            string_tuple("end"),
        )]);
        let wider = make_refined_set(vec![union(
            make_refined_set(vec![union(
                make_refined_set(vec![union(string_tuple("insideStart"), string_tuple("insideEnd"))]),
                string_tuple("end"),
            )]),
            string_tuple("outside"),
        )]);
        let declared = DeclaredRefinement {
            set: wider,
            spelling: "PositionLabel".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let value = known_set(narrowed, None, TrustProved, SetKindTag::None);
        assert!(
            matches!(judge(&value, &declared, &kernel), Verdict::Silent),
            "a narrowed string-literal union wholly inside a wider one must be Silent, not Undetermined"
        );
    }

    /// The mirror: a member OUTSIDE the declared union
    /// (`string_to_literal_union_parameter`'s own shape, widened to a Set
    /// rather than that row's single-value read) fires — `seq_subset`
    /// proving false over recognized sequence shapes is a decided
    /// refutation, the same "false is a verdict, never a refusal in
    /// disguise" reading `scalar_subset`'s own law doc states.
    #[test]
    fn a_string_literal_union_with_a_member_outside_the_declared_set_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        use refined_sets::codepoint_sets::string_tuple;
        use refined_sets::refinement_forms::union;
        let declared = DeclaredRefinement {
            set: make_refined_set(vec![union(string_tuple("node"), string_tuple("link"))]),
            spelling: "Tag".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let value = known_set(
            make_refined_set(vec![union(string_tuple("node"), string_tuple("other"))]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Tag'"), "{message}");
        assert!(message.contains("admits values outside"), "{message}");
    }

    /// The BOOLEAN PRODUCT LAW: `True` (Boolean-tagged) against Age
    /// (int-sorted) fires — bool is excluded from the int sort by
    /// product law, the fixture rows' own reason
    /// (b-body-expressions.py:744, c-reads-and-values.py:999). Before
    /// this law, a Boolean-tagged value flowed to the per-value kernel
    /// membership ask and passed silently (1.0 sits inside [0, 120]).
    #[test]
    fn a_boolean_true_into_an_int_sorted_alias_fires_by_product_law() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("True"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.to_lowercase().contains("product law"), "{message}");
    }

    /// The non-firing neighbor: a Boolean-tagged value against a
    /// NON-integer-sorted declared set is unchanged — the product law
    /// gates on `requires_integer` alone, so a float-tolerant alias
    /// still asks the kernel per value the ordinary way (1.0 is a member
    /// of `[0, ∞)`).
    #[test]
    fn a_boolean_true_into_a_non_integer_sorted_alias_is_unchanged() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = DeclaredRefinement {
            set: make_refined_set(vec![at_least(0.0)]),
            spelling: "Weight".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        };
        let value = known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    // --- m-pydantic-schema.py's Digits: pattern-and-window intersection ---

    /// `type Digits = Annotated[str, Field(min_length=1, max_length=4,
    /// pattern=r"^[0-9]+$")]` — `surface.rs`'s `annotated_expression_set`
    /// own fold: the compiled `[0-9]+` grammar's `Repeat` form (the
    /// digit range, unbounded repetition) INTERSECTED with the
    /// `min_length`/`max_length` window's own `Repeat` form (the full
    /// codepoint ground, length `[1, 4]`) — two `Repeat` forms over
    /// DIFFERENT element sets, never `on_one_tuple_layer` (each is a
    /// `Form::Repeat`, not a scalar form), so this alias never reaches
    /// the tuple-pun sort law this file's other Digits/Grade tests pin;
    /// it flows straight to the ordinary whole-word kernel membership
    /// ask.
    fn digits_refinement() -> DeclaredRefinement {
        use refined_sets::codepoint_sets::codepoints;
        use refined_sets::refinement_forms::repeat_of;
        let digit_range = make_refined_set(vec![integer(), at_least(0x30 as f64), at_most(0x39 as f64)]);
        DeclaredRefinement {
            set: make_refined_set(vec![
                repeat_of(digit_range, 1, None),
                repeat_of(codepoints(), 1, Some(4)),
            ]),
            spelling: "Digits".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// `TypeAdapter(Digits).validate_python("42")` — m-pydantic-schema.py:65's
    /// own row. `"42"` (a String-tagged whole word, 2 code points, both
    /// ASCII digits) is a genuine member of `Digits`'s pattern-and-window
    /// set: `judge()` proves this Silent via the ordinary whole-word
    /// kernel membership ask, root-causing this row's OWN reported false
    /// Fire to `check.rs`'s adapter-alias route (`adapter_alias_verdict`'s
    /// LAX INT COERCION), not to this file — that coercion is gated only
    /// on the value being a digit string and the alias not being a
    /// `StrictInt` name, with no check that the alias's declared set is
    /// even NUMERIC-sorted, so `"42"` is silently rewritten to the
    /// Integer value `42` before `judge()` ever sees it, and `42`
    /// (correctly) fails membership in a codepoint-tuple-shaped set. This
    /// test pins that `judge()` itself, given the UN-coerced String
    /// value, decides the row correctly.
    #[test]
    fn a_string_value_member_of_the_digits_pattern_and_window_set_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = digits_refinement();
        let value = known_values(hi_points("42"), PrimitiveKind::String, TrustProved);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// The mirror: `"ab"` (letters, outside the digit pattern) fires via
    /// the ordinary whole-word kernel ask, quoting the string readably —
    /// m-pydantic-schema.py:71's own row.
    #[test]
    fn a_string_value_outside_the_digits_pattern_fires_quoting_the_string() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = digits_refinement();
        let value = known_values(hi_points("ab"), PrimitiveKind::String, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("\"ab\""), "{message}");
        assert!(message.contains("'Digits'"), "{message}");
    }

    // --- the ELEMENT LAW: dict[str, X]'s value-slot judgment ---

    /// `dict[str, Age]` — the `element`-carrying declaration
    /// `a-statements.py`'s `return_dict_members` needs, `element` set to
    /// the same `age_refinement` every other test in this file shares.
    fn dict_of_age_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(Vec::new()),
            spelling: "dict[str, Age]".to_owned(),
            admits_none: false,
            element: Some(Box::new(age_refinement())),
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }

    /// `return {"age": 200}` under `-> dict[str, Age]` — an Object with
    /// one out-of-set member fires, naming the key so the reader sees
    /// which member escaped ("(at key 'age')").
    #[test]
    fn a_dict_with_an_out_of_set_member_fires_naming_the_key() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = dict_of_age_refinement();
        let value = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("(at key 'age')"), "{message}");
    }

    /// `return {"age": 40}` under `-> dict[str, Age]` — every member sits
    /// inside the element refinement, so the whole dict is Silent.
    #[test]
    fn a_dict_with_every_member_in_set_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = dict_of_age_refinement();
        let value = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// `None` against a plain (non-`Optional`) `dict[str, Age]` fires —
    /// a dict declaration is not scalar-shaped, so this exercises the
    /// element law's own explicit Null arm rather than the ordinary
    /// structural law (which would decline: `declared.set` is empty for
    /// an element-carrying declaration).
    #[test]
    fn none_against_a_plain_dict_declaration_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = dict_of_age_refinement();
        assert!(!declared.admits_none);
        let value = refined_domain::abstract_value::null_value();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'dict[str, Age]'"), "{message}");
        assert!(message.to_lowercase().contains("none"), "{message}");
    }

    /// `None` against `dict[str, Age] | None` (`admits_none` true, still
    /// element-carrying) is Silent — the admits_none check wins before
    /// the element law's Null arm would otherwise fire.
    #[test]
    fn none_against_an_admits_none_dict_declaration_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut declared = dict_of_age_refinement();
        declared.admits_none = true;
        let value = refined_domain::abstract_value::null_value();
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// A list against `dict[str, Age]` fires — the element law's own
    /// explicit List arm, kind-worded.
    #[test]
    fn a_list_against_a_dict_declaration_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = dict_of_age_refinement();
        let value = refined_domain::known_constructors::known_list(Vec::new(), TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'dict[str, Age]'"), "{message}");
        assert!(message.to_lowercase().contains("list"), "{message}");
    }

    // --- the MEMBERS LAW: a TypedDict's per-member judgment ---

    /// `PersonDict`'s own `age: Age` member table — the `members`-carrying
    /// declaration h-object-literal-members.py's `dict_return_member`
    /// needs, `age` set to the same `age_refinement` every other test in
    /// this file shares.
    fn person_dict_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(Vec::new()),
            spelling: "PersonDict".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: Some(vec![("age".to_owned(), age_refinement())]),
            positions: None,
        }
    }

    /// `return {"age": 200}` under `-> PersonDict` — the declared member's
    /// own out-of-set value fires, naming the key exactly like the
    /// element law's own key-naming convention.
    #[test]
    fn a_typed_dict_with_an_out_of_set_member_fires_naming_the_key() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = person_dict_refinement();
        let value = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("(at key 'age')"), "{message}");
    }

    /// `return {"age": 40}` under `-> PersonDict` — the member sits inside
    /// its own declared set, so the whole dict is Silent.
    #[test]
    fn a_typed_dict_with_its_member_in_set_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = person_dict_refinement();
        let value = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// A declared member the dict literal never writes is not judged at
    /// all — the honest-absence rule this law's own doc states — so a
    /// dict missing `age` entirely is still Silent rather than
    /// Undetermined or a false Fire.
    #[test]
    fn a_typed_dict_missing_a_declared_member_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = person_dict_refinement();
        let value = refined_domain::known_constructors::known_object(Vec::new(), None, true, TrustProved, false);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// `None` against a plain (non-`Optional`) TypedDict declaration
    /// fires — the members law's own explicit Null arm, mirroring the
    /// element law's.
    #[test]
    fn none_against_a_plain_typed_dict_declaration_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = person_dict_refinement();
        assert!(!declared.admits_none);
        let value = refined_domain::abstract_value::null_value();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'PersonDict'"), "{message}");
        assert!(message.to_lowercase().contains("none"), "{message}");
    }

    /// A list against a TypedDict declaration fires — the members law's
    /// own explicit List arm.
    #[test]
    fn a_list_against_a_typed_dict_declaration_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = person_dict_refinement();
        let value = refined_domain::known_constructors::known_list(Vec::new(), TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'PersonDict'"), "{message}");
        assert!(message.to_lowercase().contains("list"), "{message}");
    }

    // --- the POSITIONS LAW: a fixed-arity tuple's per-slot judgment ---

    /// `tuple[Age, Label]` — the `positions`-carrying declaration
    /// `c-reads-and-values.py`'s own fixed-arity-tuple rows need, each
    /// slot set to a DIFFERENT one of this file's shared refinements —
    /// unlike the element law's one shared refinement, each position
    /// keeps its own set.
    fn age_label_tuple_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(Vec::new()),
            spelling: "tuple[Age, Label]".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: Some(vec![age_refinement(), label_refinement()]),
        }
    }

    /// A list of two values, both inside their own position's set, is
    /// Silent.
    #[test]
    fn a_two_slot_list_with_every_position_in_set_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_label_tuple_refinement();
        let value = refined_domain::known_constructors::known_list(
            vec![
                known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
                known_values(hi_points("ok"), PrimitiveKind::String, TrustProved),
            ],
            TrustProved,
        );
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    /// Slot 0 out of its own declared set fires, naming the offending
    /// index — the positions law's own twin of the element law's
    /// key-naming convention.
    #[test]
    fn a_two_slot_list_with_slot_zero_out_of_set_fires_naming_the_index() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_label_tuple_refinement();
        let value = refined_domain::known_constructors::known_list(
            vec![
                known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
                known_values(hi_points("ok"), PrimitiveKind::String, TrustProved),
            ],
            TrustProved,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("(at index 0)"), "{message}");
    }

    /// A list of the WRONG LENGTH (one slot, not two) fires as a
    /// structural mismatch rather than sitting undetermined or judging
    /// past the end of `positions`.
    #[test]
    fn a_list_of_the_wrong_length_fires_as_a_structural_mismatch() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_label_tuple_refinement();
        let value = refined_domain::known_constructors::known_list(
            vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)],
            TrustProved,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'tuple[Age, Label]'"), "{message}");
    }

    /// `None` against a plain (non-`Optional`) fixed-arity tuple
    /// declaration fires — the positions law's own explicit Null arm,
    /// mirroring the element/members laws.
    #[test]
    fn none_against_a_plain_positions_declaration_fires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_label_tuple_refinement();
        assert!(!declared.admits_none);
        let value = refined_domain::abstract_value::null_value();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'tuple[Age, Label]'"), "{message}");
        assert!(message.to_lowercase().contains("none"), "{message}");
    }

    /// `None` against `Optional[tuple[Age, Label]]` (`admits_none` true)
    /// is Silent — the same admits_none precedence the element/members
    /// laws already give their own Null arm.
    #[test]
    fn none_against_an_admits_none_positions_declaration_is_silent() {
        let Some(kernel) = loaded_kernel() else { return };
        let mut declared = age_label_tuple_refinement();
        declared.admits_none = true;
        let value = refined_domain::abstract_value::null_value();
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }

    // --- the REASONED SENTENCE: what the value is, what the sink
    // requires. Every fire below is already pinned above for its
    // VERDICT; these pin the WORDING the reader gets.

    /// A refutation states the sink's own REQUIREMENT, not just its
    /// name: `Age`'s bounds ride beside it, so a reader never opens the
    /// alias to learn what it admits.
    #[test]
    fn a_refutation_spells_what_the_sink_requires_beside_its_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_values(vec![200.0], PrimitiveKind::Integer, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'200'"), "{message}");
        assert!(message.contains("'Age'"), "{message}");
        assert!(message.contains("120"), "names Age's own ceiling: {message}");
    }

    /// A SORT crossing states the reason in plain words — the value's
    /// sort, the position's sort, and that no run reconciles them. The
    /// Go twin's "— <said> is not allowed here" clause.
    #[test]
    fn a_float_into_an_int_sorted_alias_states_both_sorts_and_the_reason() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("not assignable"), "{message}");
        assert!(message.contains("the value is a float"), "{message}");
        assert!(message.contains("states an integer"), "{message}");
        assert!(message.contains("not allowed here"), "{message}");
    }

    /// The mirror direction: a number arriving where a string is stated
    /// says so as a sort crossing, never as a bare "not assignable".
    #[test]
    fn a_number_into_a_string_ground_alias_states_the_sort_crossing() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = any_string_refinement();
        let value = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("not assignable"), "{message}");
        assert!(message.contains("the value is a number"), "{message}");
        assert!(message.contains("states a string"), "{message}");
        assert!(message.contains("not allowed here"), "{message}");
    }

    /// A structural mismatch (a dict where a scalar is stated) names
    /// what the position states and why the value cannot sit there.
    #[test]
    fn a_dict_into_a_numeric_ground_alias_states_what_the_position_requires() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = refined_domain::known_constructors::known_object(
            Vec::new(),
            Default::default(),
            false,
            TrustProved,
            false,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("the position states"), "{message}");
        assert!(message.contains("not allowed here"), "{message}");
    }

    /// A container declaration carries an EMPTY outer set, so the
    /// requirement clause must not append a vacuous "(any value)" — the
    /// name stands alone.
    #[test]
    fn a_container_declaration_names_itself_without_a_vacuous_contents_clause() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = dict_of_age_refinement();
        let value = refined_domain::abstract_value::null_value();
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'dict[str, Age]'"), "{message}");
        assert!(!message.contains("any value"), "{message}");
    }

    /// An arity mismatch says how many elements arrived AND how many the
    /// position states — the reader sees both counts.
    #[test]
    fn an_arity_mismatch_states_both_counts() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_label_tuple_refinement();
        let value = refined_domain::known_constructors::known_list(
            vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)],
            TrustProved,
        );
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("1 element"), "{message}");
        assert!(message.contains("states 2 element"), "{message}");
    }
}
