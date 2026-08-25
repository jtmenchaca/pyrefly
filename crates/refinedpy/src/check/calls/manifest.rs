//! Foreign manifest call judging: a call on a manifested module's own
//! listed function, judged against the manifest's parsed entry contract.

use refined_domain::abstract_value::AbstractValue;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;

use crate::check::{Finding, WalkContext};
use crate::env::Environment;
use crate::expressions::evaluate_expression;

/// Recognizes `expr` as a call on a manifested module's own listed
/// function, judges every WRITTEN argument (positional matched by
/// position, keyword matched by name) against the manifest's parsed
/// entry contract, and pushes an RTS7001 for each one that escapes —
/// `binding_manifest::judge_manifest_call`'s own crossing-fit check, the
/// SAME refusal shape the stdio edge fires for an escaping outbound
/// value. A no-op for every call this recognizes nothing about: an
/// unmodeled module with no manifest, a manifest with no row for the
/// called function (rung 1's own plain decline territory either way),
/// or a manifest file this reader could not parse — a bad manifest never
/// crashes the walk, it simply contributes no crossing-fit judging this
/// call.
pub(in crate::check) fn manifest_call_fires(expr: &Expr, context: &WalkContext, environment: &Environment, out: &mut Vec<Finding>) {
    let Expr::Call(call) = expr else {
        return;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return;
    };
    if environment.read(module_name.id.as_str()).is_some() {
        return;
    }
    let entry_directory = environment.entry_directory().map(|path| path.as_path());
    let Some(Ok(manifest)) = crate::binding_manifest::discover_manifest(module_name.id.as_str(), entry_directory) else {
        return;
    };
    let Some(entry) = manifest.entries.get(attribute.attr.as_str()) else {
        return;
    };
    let positional: Vec<(AbstractValue, ruff_text_size::TextRange)> = call
        .arguments
        .args
        .iter()
        .map(|arg| (evaluate_expression(arg, environment, context.kernel), arg.range()))
        .collect();
    let keyword: Vec<(String, AbstractValue, ruff_text_size::TextRange)> = call
        .arguments
        .keywords
        .iter()
        .filter_map(|kw| {
            let arg_name = kw.arg.as_ref()?;
            Some((
                arg_name.as_str().to_owned(),
                evaluate_expression(&kw.value, environment, context.kernel),
                kw.value.range(),
            ))
        })
        .collect();
    let outcome =
        crate::binding_manifest::judge_manifest_call(module_name.id.as_str(), entry, &positional, &keyword, context.kernel);
    for (range, message) in outcome.fires {
        out.push(Finding { range, code: "RTS7001", message });
    }
}
