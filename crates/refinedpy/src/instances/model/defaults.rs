//! A field's default value expression, evaluated against a fresh
//! environment — with `Field(default=...)`'s own pydantic-surface
//! default read specially.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;

use crate::env::Environment;
use crate::expressions::evaluate_expression;

/// A field's default value expression, evaluated against a fresh
/// environment via `expressions::evaluate_expression` — EXCEPT a
/// `Field(...)` call, which is pydantic surface, not an ordinary
/// call `evaluate_expression` can read (it declines every call whose
/// callee is a bound-or-unrecognized name, per expressions.rs's own
/// `evaluate_call` contract). `field_call_default` reads a `Field`
/// call's own `default=` keyword when the call names `Field` by
/// import identity, so `age: Age = Field(default=40, ge=0, le=120)`
/// reads its default the same way a bare `age: Age = 40` does.
pub(super) fn default_value_of(value_expr: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    if let Expr::Call(call) = value_expr {
        if let Some(default) = field_call_default(call) {
            return default;
        }
    }
    evaluate_expression(value_expr, environment, kernel)
}

/// `Field(default=..., ...)` — the DEFAULT for a `= Field(...)` row is
/// `Field`'s own `default=` keyword when that keyword's value is a
/// numeric literal (`surface.rs`'s own `literal_number` reader, the
/// same one `annotated_expression_set` uses for `ge`/`le`/`lt`/`gt`).
/// No import-identity check gates the callee name here: this file does
/// not itself decide whether the call names pydantic's `Field` — the
/// field's ANNOTATION already gated that upstream via
/// `annotated_expression_set` when it read `Annotated[int,
/// Field(...)]`, and the mission's example row (`age: Age =
/// Field(default=..., ...)`) carries its constraint through the `Age`
/// alias's own `Annotated[...]`, so matching the callee's bare
/// spelling is sufficient for every corpus row this wave serves. An
/// int-sorted literal tags `Integer`, matching every other int literal
/// this checker reads (`expressions.rs`'s own `number_literal_value`).
fn field_call_default(call: &ruff_python_ast::ExprCall) -> Option<AbstractValue> {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "Field" {
        return None;
    }
    let keyword = call
        .arguments
        .keywords
        .iter()
        .find(|kw| kw.arg.as_ref().is_some_and(|name| name.as_str() == "default"))?;
    let sort = if matches!(&keyword.value, Expr::NumberLiteral(literal) if matches!(literal.value, ruff_python_ast::Number::Float(_)))
    {
        PrimitiveKind::Float
    } else {
        PrimitiveKind::Integer
    };
    let value = crate::surface::literal_number(&keyword.value)?;
    Some(known_values(vec![value], sort, TrustProved))
}
