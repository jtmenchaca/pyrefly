//! A retained-callable call's own positional arguments, including the
//! starred/kwargs-spread splice path — lives beside the datetime family
//! for historical reasons (the file this module was split from), not
//! because it models a temporal construct.

use std::sync::Arc;

use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;

use crate::env::Environment;

use super::super::evaluate_expression;
use super::super::call::*;

/// A retained-callable call's own positional arguments, given `def`'s
/// synthetic parameter list — tries `positional_arguments_for_def`'s
/// existing exact mapping FIRST (the ordinary, no-splat call shape
/// every other row uses), and only when THAT declines because the
/// call site carries a `Starred` positional argument (`f(*args,
/// **kwargs)`, r-ast-census.py's own `wrapper`: a ParamSpec-forwarding
/// body handing its own received `*args`/`**kwargs` straight to the
/// retained callable it wraps) tries splicing instead: `*args`
/// splices through `splice_call_arguments` (a known `Kind::List`
/// receiver only — the same honest decline on an unbounded iterable
/// that function's own doc states), and a `**kwargs`-spread keyword
/// argument (`keyword.arg.is_none()`) reads its own known `Kind::
/// Object` entries, mapping each by NAME onto `def`'s own parameter
/// list — the same by-name mapping `positional_arguments_with_kwargs_
/// dict` gives an ordinary named keyword, extended to a spread rather
/// than a single name. A `**kwargs` value that is not a known
/// `Kind::Object`, or an entry naming no parameter of `def`, declines
/// the whole call — this reader guesses at neither shape.
pub(in crate::expressions) fn positional_arguments_for_retained_call(
    call: &ruff_python_ast::ExprCall,
    def: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if let Some(mapped) = positional_arguments_for_def(call, def, environment, kernel) {
        return Some(mapped);
    }
    let has_starred_positional = call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_)));
    let has_kwargs_spread = call.arguments.keywords.iter().any(|keyword| keyword.arg.is_none());
    if !has_starred_positional && !has_kwargs_spread {
        return None;
    }
    let parameter_names: Vec<&str> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
        .map(|parameter| parameter.parameter.name.id.as_str())
        .collect();
    let mut positional = splice_call_arguments(&call.arguments.args, environment, kernel)?;
    for keyword in &call.arguments.keywords {
        match keyword.arg.as_ref() {
            Some(arg_name) => {
                let position = parameter_names.iter().position(|name| *name == arg_name.as_str())?;
                if position < positional.len() {
                    positional[position] = evaluate_expression(&keyword.value, environment, kernel);
                } else {
                    positional.resize_with(position + 1, unknown);
                    positional[position] = evaluate_expression(&keyword.value, environment, kernel);
                }
            }
            None => {
                let spread = evaluate_expression(&keyword.value, environment, kernel);
                if spread.kind != Kind::Object {
                    return None;
                }
                for entry in &spread.keys {
                    let position = parameter_names.iter().position(|name| *name == entry.name.as_str())?;
                    if position < positional.len() {
                        positional[position] = entry.value.clone();
                    } else {
                        positional.resize_with(position + 1, unknown);
                        positional[position] = entry.value.clone();
                    }
                }
            }
        }
    }
    Some(positional)
}
