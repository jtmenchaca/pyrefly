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
use refined_sets::format_string_shapes::format_py_number;
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
/// `Kind::Values` (a known scalar, or a known tuple word) asks the
/// kernel per value — EXCEPT the mission's own int-sort law: a
/// Float-tagged value against a declared set that carries the `int`
/// form (`requires_integer`) fires outright, never asked of
/// `kernel.member` at all, because `int ≠ float` is a SORT law
/// (PYREFLY-NUMERIC-B3-B4.md's "two sorts, never one Number") that the
/// kernel's own scalar membership question does not carry — the
/// kernel's `member` decides "is this real number inside this real
/// set," and `30.0`'s real value of 30 sits inside `[0, 120]` exactly
/// as `30`'s does; the sort mismatch is a fact ONLY the checker's own
/// PrimitiveKind tag states, so it must be judged before, not through,
/// the kernel question. Every other Values case (Integer-tagged, or
/// bare Number where the sort is unknown) asks the kernel exactly as
/// before, spelling the fire message the Python way via
/// `format_py_number`.
///
/// `Kind::Set` (a refined set of possible values, not one exact word)
/// asks the kernel's own set-relationship questions: `scalar_subset`
/// proves the whole flowing set lands inside the declared set (silent),
/// `scalar_disjoint` proves no member of the flowing set can ever land
/// inside it (fire), and neither proof going through is an honest
/// overlap the walk cannot resolve further — undetermined, naming what
/// blocked in the sentence the mission specifies verbatim.
///
/// Anything else (not yet a known value or set) is undetermined with a
/// sentence the caller may adopt as the body's blocker.
pub fn judge(
    value: &AbstractValue,
    declared: &DeclaredRefinement,
    kernel: &Arc<RefinedTSKernel>,
) -> Verdict {
    if value.kind == Kind::Values {
        let is_float_sorted = value.kind_tag == Some(PrimitiveKind::Float);
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
    Verdict::Undetermined("the flowing value is not yet readable".to_owned())
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
        };
        let thirty_float = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
        assert!(matches!(judge(&thirty_float, &declared, &kernel), Verdict::Silent));
    }
}
