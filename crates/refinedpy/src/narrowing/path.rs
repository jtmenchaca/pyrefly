//! Access-path channel: numeric comparisons over attribute chains
//! (`TrackedPlace`), tightening WINDOW bounds at paths.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::compare::mirror_cmp_op;
use super::compare::numeric_cmp_op;
use super::compare::NumericCmpOp;
use super::isinstance_guards::unbounded_integers;
use super::literal_number;

/// The ACCESS-PATH channel's own entry point (`env.rs`'s own
/// `TrackedPlace`/`bind_path`/`read_path` doc): for every numeric
/// comparison `condition` folds through `and` (chained comparisons and
/// `and`-conjunctions only — an `or`'s own operand alone could have made
/// the whole thing true, so it states nothing about any single operand,
/// the same rule the VALUES channel's `narrow_bool_op` already follows),
/// whose tested side is an ATTRIBUTE CHAIN rather than a bare name
/// (`a.n`, `env::tracked_place_of`'s own doc), tighten a WINDOW bound at
/// that path. Run after the VALUES and SET channels, for the identical
/// reason `narrow_set_kind_names` runs after the VALUES channel: nothing
/// in this wave SEEDS a path fact from an `isinstance` test the way a
/// bare name can, so there is no ordering dependency the other direction,
/// but keeping the SAME position after the two name-keyed channels keeps
/// the three channels' relative order stable and easy to reason about.
pub(super) fn narrow_path_comparisons(condition: &Expr, environment: &mut Environment, truth: bool) {
    match condition {
        Expr::BoolOp(bool_op) if bool_op.op == BoolOp::And && truth => {
            for value in &bool_op.values {
                narrow_path_comparisons(value, environment, truth);
            }
        }
        Expr::Compare(compare) if truth => {
            let mut left = compare.left.as_ref();
            for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
                narrow_one_path_comparison(left, *op, right, environment);
                left = right;
            }
        }
        // `not`, `or`, a single-pair falsity read, a call, anything else:
        // no shape this channel narrows — the honest "narrows nothing"
        // default every leaf in this file keeps. Falsity is not read at
        // all here (unlike the VALUES channel's single-pair `is`/`is not
        // None` exception): a comparison's negation over an unenumerated
        // WINDOW fact has no single bound this channel can tighten to,
        // the same reason `narrow_name_length_against_literal`'s own
        // falsity path folds through `negate_numeric_cmp_op` instead of
        // being read leaf-by-leaf — the path channel keeps that
        // tightening scoped to the truth arm only, matching the mission's
        // narrower ask.
        _ => {}
    }
}

/// One comparison pair (`left op right`) as an ACCESS-PATH narrowing leaf:
/// a numeric literal on one side, an attribute chain on the other
/// (`env::tracked_place_of`), tightens that chain's own WINDOW bound —
/// the identical `{lo, hi}` tightening `narrow_name_length_against_literal`
/// already gives a length window, applied here to a path's own numeric
/// fact instead of a `len(...)` call's result. `is`/`is not`, `in`/`not
/// in`, and a non-numeric operator narrow nothing this leaf reads (the
/// VALUES/SET channels' own leaves already cover those over a BARE name;
/// a path is scoped to the numeric-comparison construct the mission
/// names). Two changing paths (`a.n < b.m`), or a side that is neither a
/// path nor a literal, narrow nothing — the honest default.
pub(super) fn narrow_one_path_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment) {
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return;
    };
    let (place, on_place, literal) = if let (Some(place), Some(literal)) =
        (crate::env::tracked_place_of(left), literal_number(right))
    {
        (place, true, literal)
    } else if let (Some(literal), Some(place)) =
        (literal_number(left), crate::env::tracked_place_of(right))
    {
        (place, false, literal)
    } else {
        return;
    };
    // a bare name is already the VALUES/SET channels' own business —
    // this leaf is scoped to a GENUINE multi-segment path, the shape
    // those two channels cannot bind at all (`bindings` is keyed on one
    // name)
    if place.path.is_empty() {
        return;
    }
    let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
    narrow_path_window(&place, effective_op, literal, environment);
}

/// Tightens the WINDOW bound recorded for `place` by `place op literal`
/// holding — the path-keyed twin of `narrow_name_length_against_literal`'s
/// own `{lo, hi}` tightening, reused here rather than duplicated: a path
/// fact is read back (or seeded fresh, the unbounded integer ray — every
/// `Age`-annotated instance field this construct's own rows use is an
/// int, and this wave states no path fact for a non-integer field) then
/// tightened the identical way that function tightens a length window,
/// through the SAME `{lo, hi}` triple. `!=` tightens nothing (the same
/// "no shape for a single excluded point" decline that function's own
/// `NotEq` arm gives); a tightened-to-empty window is left UNBOUND rather
/// than rebound (an infeasible path fact is the walk's own dead-branch
/// business, never a narrowing claim this leaf makes).
pub(super) fn narrow_path_window(place: &crate::env::TrackedPlace, op: NumericCmpOp, literal: f64, environment: &mut Environment) {
    let current = environment.read_path(place).cloned().unwrap_or_else(|| AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(unbounded_integers(), None, TrustSpec, SetKindTag::None)
    });
    if current.kind != Kind::Set {
        return;
    }
    let Some(repeated_or_window) = numeric_window_of(&current.set) else {
        return;
    };
    let (mut lo, mut hi) = repeated_or_window;
    // integer-only bounds (this channel seeds nothing but the unbounded
    // integer ray — `narrow_path_window`'s own doc), the same `± 1`
    // strict-inequality reading `narrow_name_length_against_literal`
    // already takes for a length window.
    match op {
        NumericCmpOp::GtE => lo = lo.max(literal),
        NumericCmpOp::Gt => lo = lo.max(literal + 1.0),
        NumericCmpOp::LtE => hi = Some(hi.map_or(literal, |current_hi| current_hi.min(literal))),
        NumericCmpOp::Lt => hi = Some(hi.map_or(literal - 1.0, |current_hi| current_hi.min(literal - 1.0))),
        NumericCmpOp::Eq => {
            lo = lo.max(literal);
            hi = Some(hi.map_or(literal, |current_hi| current_hi.min(literal)));
        }
        NumericCmpOp::NotEq => return,
    }
    if let Some(h) = hi {
        if h < lo {
            // the window is now provably empty — leave the path fact
            // unchanged rather than rebind to an empty claim; the walk's
            // own dead-branch handling is what skips an unreachable arm,
            // not a narrowed-to-empty path fact here
            return;
        }
    }
    let mut forms = vec![integer()];
    forms.push(at_least(lo));
    if let Some(h) = hi {
        forms.push(at_most(h));
    }
    environment.bind_path(
        place,
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(make_refined_set(forms), None, TrustSpec, SetKindTag::None)
        },
    );
}

/// A `RefinedSet`'s own `{lo, hi}` numeric window, read from its
/// `AtLeast`/`AtMost`/`Integer` forms — the SAME shape
/// `narrow_name_length_against_literal` reads off a length window,
/// applied here to a path's own numeric fact. `None` for any set shape
/// other than exactly these three forms (a `oneOf`, a string ground) —
/// this wave's path channel only ever builds this one shape itself, so a
/// set built any other way is not one it can tighten.
pub(super) fn numeric_window_of(set: &RefinedSet) -> Option<(f64, Option<f64>)> {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &set.forms {
        match form.form {
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost => hi = Some(form.a),
            Form::Integer => {}
            _ => return None,
        }
    }
    Some((lo.unwrap_or(f64::NEG_INFINITY), hi))
}

