//! Lowering the per-trip added expression into the kernel's effect
//! grammar.

use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;

use super::ELEMENT_SLOT;

/// Lowers the expression added on each trip into the kernel's effect
/// grammar, over the element slot. Recognizes the loop variable itself,
/// a numeric literal, and the arithmetic combinations of those the
/// kernel has proved transfers for.
///
/// INVARIANT, and the reason no separate term gate is needed: the only
/// `Var` effect this function can emit is `ELEMENT_SLOT`, because the
/// loop variable is the one name it matches and every other name
/// declines. So a lowered term reads the element and constants and
/// nothing else — which is exactly the premise the kernel's relation
/// rests on (`total <= count * termHi`, with `termHi` read off the
/// element alone). A term reading the running total would be a
/// recurrence rather than a sum, and one reading the count would tie
/// the total to a factor the relation does not carry; neither can be
/// built here.
///
/// `None` for anything wider — a call, an attribute read, a name that
/// is neither the loop variable nor a literal. This module never
/// approximates a body step it cannot state exactly; the accumulation
/// declines whole and the existing paths run unchanged.
pub(super) fn lower_added_expression(expression: &Expr, loop_variable: &str) -> Option<LoopEffect> {
    match expression {
        Expr::Name(name) if name.id.as_str() == loop_variable => Some(super::slot(ELEMENT_SLOT)),
        Expr::NumberLiteral(literal) => {
            // an int past i64 is not a value this reader states exactly,
            // the same ceiling every other literal reader here keeps
            let value = match &literal.value {
                Number::Int(int) => int.as_i64()? as f64,
                Number::Float(float) => *float,
                Number::Complex { .. } => return None,
            };
            Some(LoopEffect {
                kind: LoopEffectKind::Const,
                set: make_refined_set(vec![one_of(&[value])]),
                ..Default::default()
            })
        }
        // `s * s` — both operands the loop variable, the SAME source
        // variable — is a structural square: the kernel's `Effect.sq`
        // answers the correlated image `[0, max²]`, which the general
        // product `Binary(Mul, Var(i), Var(i))` cannot, since the
        // kernel no longer recognizes x*x by syntax (unsound under
        // renaming). Read directly off the source AST, before either
        // side lowers: this is the one place the identifier binding is
        // honestly known, and a lowered `LoopEffect` has already
        // thrown that identity away. Gated on the loop variable
        // specifically (not just "the same name as each other"): a
        // product of some OTHER shared free name would otherwise
        // misread as squaring the element it never named.
        Expr::BinOp(binop) if is_same_name_square(binop, loop_variable) => Some(LoopEffect {
            kind: LoopEffectKind::Sq,
            index: ELEMENT_SLOT,
            ..Default::default()
        }),
        Expr::BinOp(binop) => {
            let op = match binop.op {
                Operator::Add => LoopEffectOp::Add,
                Operator::Sub => LoopEffectOp::Sub,
                Operator::Mult => LoopEffectOp::Mul,
                // Div/FloorDiv/Mod/Pow are deliberately absent: each
                // carries a Python/JS divergence or a zero-denominator
                // premise this reader does not vouch for inside a body
                // it is stating a relation about.
                _ => return None,
            };
            let left = lower_added_expression(binop.left.as_ref(), loop_variable)?;
            let right = lower_added_expression(binop.right.as_ref(), loop_variable)?;
            Some(LoopEffect {
                kind: LoopEffectKind::Binary,
                op,
                a: Some(Box::new(left)),
                b: Some(Box::new(right)),
                ..Default::default()
            })
        }
        _ => None,
    }
}

/// Whether a `BinOp` is `<loop_variable> * <loop_variable>` — decided
/// from the source AST's own two `Expr::Name` nodes, never from a
/// lowered effect, which has already erased which variable a term came
/// from. Gated on the LOOP variable specifically, not merely "the same
/// name as each other": some other shared free name would decline
/// anyway (this module's own invariant — the only `Var` this reader
/// emits is the element slot), and must keep declining rather than
/// being misread as squaring the element.
pub(super) fn is_same_name_square(binop: &ruff_python_ast::ExprBinOp, loop_variable: &str) -> bool {
    if !matches!(binop.op, Operator::Mult) {
        return false;
    }
    let (Expr::Name(left), Expr::Name(right)) = (binop.left.as_ref(), binop.right.as_ref()) else {
        return false;
    };
    left.id.as_str() == loop_variable && right.id.as_str() == loop_variable
}
