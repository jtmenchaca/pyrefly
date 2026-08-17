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
///   mirror — a numeric value against a declared set that is
///   STRING-ground (`is_string_ground`, the codepoint-alphabet star)
///   fires too. A tuple of code points and a tuple of numbers share the
///   same wire shape; only the two sides' own sort tags can tell a
///   string from a number, so this is judged the same way as the
///   int-sort law rather than asked of the kernel.
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
/// carries its own FLOAT-SORT law first: a Float-sorted Set (`kind_tag:
/// Some(PrimitiveKind::Float)` — `abstract_value::float_sorted_unknown`,
/// the shape `math.sqrt`'s value-unknown result carries) against a
/// declared set that requires the `int` form fires outright, the same
/// reasoning as the Values-side int-sort law: a float-sorted value is
/// never a member of an int-sorted set, regardless of which real
/// numbers either side admits. A Float-sorted Set against a
/// NON-integer-sorted declared set declines this law and falls through
/// to the kernel's own set-relationship questions: `scalar_subset`
/// proves the whole flowing set lands inside the declared set (silent),
/// `scalar_disjoint` proves no member of the flowing set can ever land
/// inside it (fire), and neither proof going through is an honest
/// overlap the walk cannot resolve further — undetermined, naming what
/// blocked in the sentence the mission specifies verbatim.
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
/// Anything else (`Kind::Unknown`, `Kind::KindUnion`, and every other
/// not-yet-known shape) is undetermined with a sentence the caller may
/// adopt as the body's blocker.
pub fn judge(
    value: &AbstractValue,
    declared: &DeclaredRefinement,
    kernel: &Arc<RefinedTSKernel>,
) -> Verdict {
    if value.kind == Kind::Object && value.kind_word.is_some() {
        let scalar_ground = on_one_tuple_layer(&declared.set) || is_string_ground(&declared.set);
        if scalar_ground {
            let word = value.kind_word.expect("checked Some above");
            return Verdict::Fire(format!(
                "a value of kind '{}' is not assignable to type '{}'",
                word, declared.spelling,
            ));
        }
    }
    if value.kind == Kind::Values {
        let is_string = value.kind_tag == Some(PrimitiveKind::String);
        let is_float_sorted = value.kind_tag == Some(PrimitiveKind::Float);
        let is_boolean = value.kind_tag == Some(PrimitiveKind::Boolean);
        if is_string && on_one_tuple_layer(&declared.set) {
            return Verdict::Fire(format!(
                "a value of type '{}' is not assignable to type '{}'",
                spelled_string_word(&value.values),
                declared.spelling,
            ));
        }
        if !is_string && is_string_ground(&declared.set) {
            for v in &value.values {
                return Verdict::Fire(format!(
                    "a value of type '{}' is not assignable to type '{}'",
                    format_py_number(*v, is_float_sorted),
                    declared.spelling,
                ));
            }
            return Verdict::Silent; // an empty tuple word has no value to fire on
        }
        if is_float_sorted && requires_integer(&declared.set) {
            for v in &value.values {
                return Verdict::Fire(format!(
                    "a value of type '{}' is not assignable to type '{}'",
                    format_py_number(*v, true),
                    declared.spelling,
                ));
            }
            return Verdict::Silent; // an empty tuple word has no value to fire on
        }
        if is_boolean && requires_integer(&declared.set) {
            return Verdict::Fire(format!(
                "a value of type '{}' is not assignable to type '{}' — bool is excluded from the int sort by product law",
                spelled_boolean_word(&value.values),
                declared.spelling,
            ));
        }
        if is_string {
            if !(kernel.member)(&declared.set, &value.values) {
                return Verdict::Fire(format!(
                    "a value of type '{}' is not assignable to type '{}'",
                    spelled_string_word(&value.values),
                    declared.spelling,
                ));
            }
            return Verdict::Silent;
        }
        for v in &value.values {
            if !(kernel.member)(&declared.set, &[*v]) {
                return Verdict::Fire(format!(
                    "a value of type '{}' is not assignable to type '{}'",
                    format_py_number(*v, false),
                    declared.spelling,
                ));
            }
        }
        return Verdict::Silent;
    }
    if value.kind == Kind::Set {
        let is_float_sorted = value.kind_tag == Some(PrimitiveKind::Float);
        if is_float_sorted && requires_integer(&declared.set) {
            return Verdict::Fire(format!(
                "a value of type '{}' is not assignable to type '{}' — a float-sorted value is never in an int-sorted set",
                refined_sets::format_for_diagnostics::format_for_diagnostics(&value.set),
                declared.spelling,
            ));
        }
        if (kernel.scalar_subset)(&value.set, &declared.set) {
            return Verdict::Silent;
        }
        if (kernel.scalar_disjoint)(&value.set, &declared.set) {
            return Verdict::Fire(format!(
                "a value of type '{}' is not assignable to type '{}'",
                refined_sets::format_for_diagnostics::format_for_diagnostics(&value.set),
                declared.spelling,
            ));
        }
        return Verdict::Undetermined(
            "the flowing value's set is not contained in the declared set".to_owned(),
        );
    }
    if value.kind == Kind::Null && declared.admits_none {
        return Verdict::Silent;
    }
    if matches!(value.kind, Kind::Object | Kind::List | Kind::Null)
        && (on_one_tuple_layer(&declared.set) || is_string_ground(&declared.set))
    {
        let value_word = match value.kind {
            Kind::Object => value.kind_word.unwrap_or("a dict"),
            Kind::List => "a list",
            Kind::Null => "None",
            _ => unreachable!("matches! above admits only Object, List, Null"),
        };
        return Verdict::Fire(format!(
            "a value of type '{}' is not assignable to type '{}'",
            value_word, declared.spelling,
        ));
    }
    Verdict::Undetermined("the flowing value is not yet readable".to_owned())
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

    use refined_domain::abstract_value::known_values;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::at_most;
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
        }
    }

    /// `type OptionalAge = Age | None` — the same int-sorted ray, but
    /// the declaration admits absence.
    fn optional_age_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
            spelling: "Age | None".to_owned(),
            admits_none: true,
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

    /// A String-tagged value against a NUMERIC-ground alias (Age, an
    /// int-sorted ray) fires the sort law before any kernel question —
    /// a string is never a member of an int-sorted set, regardless of
    /// what the code points spell. No kernel ask is needed to decide
    /// this, so the message is asserted without requiring the kernel be
    /// built (the sort law short-circuits before `(kernel.member)` is
    /// ever called).
    #[test]
    fn a_string_value_into_a_numeric_ground_alias_fires_the_sort_law_before_any_kernel_ask() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = age_refinement();
        let value = known_values(hi_points("30"), PrimitiveKind::String, TrustProved);
        let message = fire_message(judge(&value, &declared, &kernel));
        assert!(message.contains("'Age'"), "{message}");
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
        };
        let value = refined_domain::abstract_value::opaque_value("a function value");
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Undetermined(_)));
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

    /// The non-firing neighbor: a Float-sorted Set against a
    /// float-TOLERANT (non-integer-sorted) declared set stays on today's
    /// ordinary Set path — undetermined here (R-bar is not a subset of,
    /// nor disjoint from, `Weight`'s `[0, ∞)` ray), never fired by the
    /// sort law, which is specific to `requires_integer`.
    #[test]
    fn a_float_sorted_set_into_a_non_integer_sorted_alias_stays_on_the_set_path() {
        let Some(kernel) = loaded_kernel() else { return };
        let declared = DeclaredRefinement {
            set: make_refined_set(vec![at_least(0.0)]),
            spelling: "Weight".to_owned(),
            admits_none: false,
        };
        let value = refined_domain::abstract_value::float_sorted_unknown();
        assert!(matches!(
            judge(&value, &declared, &kernel),
            Verdict::Undetermined(_)
        ));
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
        };
        let value = known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved);
        assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
    }
}
