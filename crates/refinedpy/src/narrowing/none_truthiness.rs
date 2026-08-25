//! `is None` / bool-literal / bare-name truthiness leaves.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::trust_level_of;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::Form;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::is_none_literal;
use super::name_of;

/// `is None` / `is not None` (mission point 5): a Values-kind binding
/// narrows by emptying (see below); a `Kind::PossiblyUndefined` binding
/// — an `Optional[X]`/`X | None`-declared parameter's own seed
/// (`check.rs::seed_parameters`) — narrows by UNWRAPPING, the maybe
/// carrier's own reason for existing. A non-Values, non-wrapper binding
/// (including one already `Kind::Null`) passes through unchanged, per
/// the mission's instruction that non-Values states pass through
/// everywhere this wave.
/// `P is None` as a single-pair comparison (either operand order) —
/// the shape the cross-channel disjunction reader in `narrow_bool_op`
/// recognizes as its absence side. The tested bare name, or None for
/// any other shape.
pub(super) fn is_none_test_name(e: &Expr) -> Option<&str> {
    let Expr::Compare(compare) = e else {
        return None;
    };
    if compare.ops.len() != 1 || compare.ops[0] != CmpOp::Is {
        return None;
    }
    if is_none_literal(&compare.comparators[0]) {
        return name_of(&compare.left);
    }
    if is_none_literal(&compare.left) {
        return name_of(&compare.comparators[0]);
    }
    None
}

pub(super) fn narrow_is_none(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let is_not = op == CmpOp::IsNot;
    let name = if is_none_literal(right) {
        name_of(left)
    } else if is_none_literal(left) {
        name_of(right)
    } else {
        None
    };
    let Some(name) = name else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    // `name is None` true, or `name is not None` false, both mean
    // "None": a `Kind::PossiblyUndefined` wrapper's own absent side
    // proves this reachable, so the TRUE reading of "None" rebinds to
    // the exact null_value (matching what `assignability::judge` reads
    // directly for an admits_none declaration) — never the wrapper
    // itself, since the wrapper's present side is now proved
    // unreachable on this fork.
    // `name is None` false, or `name is not None` true, both mean "not
    // None": the wrapper's own INNER value is what remains — unwrapped,
    // so a later read sees the plain present-side value (the annotated
    // set, a plain scalar, …) rather than the maybe carrier.
    let means_is_none = truth != is_not;
    if current.kind == Kind::PossiblyUndefined {
        let inner = current.inner.as_deref().expect("Kind::PossiblyUndefined always carries an inner value");
        let narrowed = if means_is_none { null_value() } else { inner.clone() };
        environment.bind(name, narrowed);
        return;
    }
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // `name is None` true, or `name is not None` false, both mean
    // "this Values binding holds None" — impossible for a Values state,
    // so every member is infeasible: bind the empty set.
    // `name is None` false, or `name is not None` true, both mean
    // "not None" — a Values binding already satisfies that for every
    // member, so it is left as is (still narrows nothing further, which
    // is sound: no member is dropped).
    if means_is_none {
        let grade = trust_level_of(&current);
        environment.bind(name, known_values(Vec::new(), kind_tag, grade));
    }
}

/// `name is True` / `name is False` — and the `==`/`!=` spellings of
/// the same pair — against a binding already scoped to the BOOL domain
/// (every member 0 or 1: `bool` seeds `oneOf{0, 1}`, `Literal[...]`
/// bool members and `isinstance(x, bool)` build the same two-value
/// reading). CPython interns exactly two bool objects (datamodel.rst,
/// "Booleans": "The two objects representing the values False and
/// True"), so identity and equality coincide on this domain and one
/// leaf reads all four operators. A binding admitting ANY other member
/// declines whole: `1 is True` is False (distinct objects), so
/// identity against a literal keeps nothing pointwise for a general
/// int, and equality on a wider set is the numeric paths' own
/// business. A filter that would empty the members also declines — an
/// unreachable arm is the walk's provably-false business, not a
/// narrowing claim.
pub(super) fn narrow_bool_literal_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let bool_literal_value = |expr: &Expr| -> Option<f64> {
        match expr {
            Expr::BooleanLiteral(literal) => Some(if literal.value { 1.0 } else { 0.0 }),
            _ => None,
        }
    };
    let (name, literal) = match (name_of(left), bool_literal_value(right)) {
        (Some(name), Some(literal)) => (name, literal),
        _ => match (bool_literal_value(left), name_of(right)) {
            (Some(literal), Some(name)) => (name, literal),
            _ => return,
        },
    };
    let keep_equal = matches!(op, CmpOp::Is | CmpOp::Eq) == truth;
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    let bool_domain = |members: &[f64]| members.iter().all(|member| *member == 0.0 || *member == 1.0);
    if current.kind == Kind::Values {
        if current.values.is_empty() || !bool_domain(&current.values) {
            return;
        }
        let kept: Vec<f64> = current.values.iter().copied().filter(|member| (*member == literal) == keep_equal).collect();
        if kept.is_empty() {
            return;
        }
        let Some(kind_tag) = current.kind_tag else {
            return;
        };
        environment.bind(name, known_values(kept, kind_tag, trust_level_of(&current)));
        return;
    }
    if current.kind == Kind::Set {
        let [form] = current.set.forms.as_slice() else {
            return;
        };
        if form.form != Form::OneOf || form.w.is_empty() || !bool_domain(&form.w) {
            return;
        }
        let kept: Vec<f64> = form.w.iter().copied().filter(|member| (*member == literal) == keep_equal).collect();
        if kept.is_empty() {
            return;
        }
        let narrowed = AbstractValue {
            kind_tag: current.kind_tag,
            ..known_set(make_refined_set(vec![one_of(&kept)]), None, trust_level_of(&current), current.set_kind_tag)
        };
        environment.bind(name, narrowed);
    }
}

/// A bare name as the whole condition (`if x:`): Python truthiness.
/// `x` truthy proves `x is not None` AND `x != 0` (`bool(None)` and
/// `bool(0)` are both False; every other int/float, NaN included, is
/// truthy). A `Kind::PossiblyUndefined` binding — an `Optional[X]`/
/// `X | None` seed — unwraps to its inner value on the truthy arm,
/// exactly as `narrow_is_none`'s not-None side does; a Values inner
/// (or a bare Values binding) additionally keeps only its truthy
/// members. The falsy arm keeps a wrapper unchanged when its inner
/// could itself be falsy (0 in the annotated set) — None and a falsy
/// inner member are then both live; when the inner is a Values set
/// with no falsy member, falsity proves None exactly. A Set-kind or
/// otherwise unread binding narrows nothing — the inner set may hold
/// 0, and dropping nothing is conservative, never wrong.
pub(super) fn narrow_name_truthiness(condition: &Expr, environment: &mut Environment, truth: bool) {
    let Some(name) = name_of(condition) else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    let truthy_members = |value: &AbstractValue| -> AbstractValue {
        if value.kind == Kind::Values {
            if let Some(kind_tag) = value.kind_tag {
                let kept: Vec<f64> = value.values.iter().copied().filter(|member| *member != 0.0).collect();
                return known_values(kept, kind_tag, trust_level_of(value));
            }
        }
        value.clone()
    };
    if current.kind == Kind::PossiblyUndefined {
        let inner = current.inner.as_deref().expect("Kind::PossiblyUndefined always carries an inner value");
        if truth {
            environment.bind(name, truthy_members(inner));
        } else if inner.kind == Kind::Values && inner.values.iter().all(|member| *member != 0.0) {
            environment.bind(name, null_value());
        }
        return;
    }
    if current.kind == Kind::Set {
        // Exact-member form first: a `oneOf` keeps the members whose
        // truthiness matches (`b: bool` seeds `oneOf{0, 1}`, so
        // `if not b:` proves `{0}`). A filter that would empty the
        // members proves nothing this arm states (the arm is then
        // unreachable — the walk's own provably-false business).
        if current.set.forms.iter().any(|form| form.form == Form::OneOf) {
            let mut forms = current.set.forms.clone();
            let mut rewrote = false;
            for form in &mut forms {
                if form.form == Form::OneOf {
                    let kept: Vec<f64> = form.w.iter().copied().filter(|member| (*member != 0.0) == truth).collect();
                    if kept.is_empty() {
                        return;
                    }
                    if kept.len() != form.w.len() {
                        rewrote = true;
                    }
                    form.w = kept;
                }
            }
            if rewrote {
                let narrowed = AbstractValue {
                    kind_tag: current.kind_tag,
                    ..known_set(make_refined_set(forms), None, trust_level_of(&current), current.set_kind_tag)
                };
                environment.bind(name, narrowed);
            }
            return;
        }
        // WINDOW form ([atLeast, atMost, integer]): truthiness on the
        // integer domain is exactly "≠ 0" (datamodel.rst, truth value
        // testing — a zero of any numeric type is false). The TRUE arm
        // trims a 0 edge off the window (an interior 0 is a hole one
        // window cannot state and trims nothing); the FALSE arm IS the
        // value 0 whenever the window admits it.
        if !current.set.forms.iter().any(|form| form.form == Form::Integer) {
            return;
        }
        let mut lo: Option<f64> = None;
        let mut hi: Option<f64> = None;
        for form in &current.set.forms {
            match form.form {
                Form::AtLeast => lo = Some(form.a),
                Form::AtMost => hi = Some(form.a),
                Form::Integer => {}
                _ => return,
            }
        }
        if truth {
            let mut forms = current.set.forms.clone();
            let mut rewrote = false;
            for form in &mut forms {
                if form.form == Form::AtLeast && form.a == 0.0 {
                    form.a = 1.0;
                    rewrote = true;
                }
                if form.form == Form::AtMost && form.a == 0.0 {
                    form.a = -1.0;
                    rewrote = true;
                }
            }
            if rewrote {
                let narrowed = AbstractValue {
                    kind_tag: current.kind_tag,
                    ..known_set(make_refined_set(forms), None, trust_level_of(&current), current.set_kind_tag)
                };
                environment.bind(name, narrowed);
            }
        } else {
            let admits_zero = lo.is_none_or(|floor| floor <= 0.0) && hi.is_none_or(|ceiling| ceiling >= 0.0);
            if admits_zero {
                environment.bind(
                    name,
                    known_values(vec![0.0], current.kind_tag.unwrap_or(PrimitiveKind::Integer), trust_level_of(&current)),
                );
            }
        }
        return;
    }
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    let kept: Vec<f64> = current.values.iter().copied().filter(|member| (*member != 0.0) == truth).collect();
    environment.bind(name, known_values(kept, kind_tag, trust_level_of(&current)));
}
