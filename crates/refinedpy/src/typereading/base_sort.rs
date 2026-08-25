//! The bare `int`/`float`/`str`/`bool` base-sort fallback, used inside
//! `declared_refinement`'s own container/union arms and exported for
//! `check.rs`'s parameter seeding.

use refined_sets::refinement_forms::make_refined_set;
use ruff_python_ast::Expr;

use super::declared_refinement::DeclaredRefinement;

/// The bare `int`/`float`/`str`/`bool` return-annotation fallback,
/// matched to `summaries.rs::return_sort_fallback`'s own sets exactly:
/// `int` is the unbounded whole-number ray (`integer()` conjoined with
/// the unbounded `at_least(NEG_INFINITY)` ray, the same "no
/// ceiling/floor" shape that fallback builds), `float` is the unbounded
/// real ray (`numbers()`, the same set `float_sorted_unknown()`
/// carries), `str` is the whole-strings ground
/// (`codepoint_sets::strings()`), `bool` is the exact two-member domain
/// (`oneOf{0, 1}`, the boolean-domain convention).
/// EXPORTED for check.rs's parameter seeding ONLY: a bare-`int`
/// parameter seeds the whole-int sort claim ("a whole int admits
/// values outside the set", the corpus's own reason). The general
/// declared_refinement table deliberately does NOT read base sorts —
/// doing so made every `-> int` helper return judge, turning each
/// unreadable helper body into a new undetermined blocker.
pub fn base_sort_return_refinement(returns: &Expr) -> Option<DeclaredRefinement> {
    let Expr::Name(sort) = returns else {
        return None;
    };
    let set = match sort.id.as_str() {
        "int" => make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
        ]),
        "float" => refined_sets::refinement_forms::numbers(),
        "str" => refined_sets::codepoint_sets::strings(),
        // `bool`'s whole domain is the two exact values 0 and 1 (the
        // boolean-domain convention `bool_literal_members` and
        // `narrow_isinstance_call` both read), so a bare `bool`
        // parameter seeds the exact two-member set rather than a ray.
        "bool" => make_refined_set(vec![refined_sets::refinement_forms::one_of(&[0.0, 1.0])]),
        _ => return None,
    };
    let spelling = sort.id.as_str().to_owned();
    Some(DeclaredRefinement {
        set,
        spelling,
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    })
}
