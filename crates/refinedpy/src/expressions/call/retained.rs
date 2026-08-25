//! Retained-callable registration and call resolution, and the shared
//! keyword→position argument-binding helpers `evaluate_call`'s
//! same-module-def/method/retained-call arms all read from.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;

use crate::collection_models;
use crate::env;
use crate::env::Environment;
use crate::summaries;

use super::super::evaluate_expression;

/// Walks `expr`'s own subtree for every `Expr::Lambda` reachable
/// WITHOUT crossing a statement boundary (a call's own function/
/// arguments/keywords, an attribute's own receiver, a lambda's own
/// body — the shapes this corpus's five retained-callable rows
/// actually nest a lambda inside: a call argument, a constructor
/// argument, or a bare `return <lambda>`), and records each one into
/// `environment` with a CLOSURE snapshot of every free name its own
/// body reads (`e-class-and-function.py`'s `make_adder`: `return
/// lambda age: age + step` reads `step`, `make_adder`'s own
/// parameter — a lambda is not always closure-free, so this scan
/// always computes the snapshot rather than assuming one is never
/// needed). Reused rather than duplicated: `RetainedCallable::
/// from_lambda` builds the synthetic single-`Return` body first, and
/// `summaries::free_variable_snapshot` reads that SAME body's own free
/// names — the identical free-name reader `Stmt::FunctionDef`'s own
/// retention (`summaries::interpret_body`) already calls for a nested
/// def. Each registration mints a FRESH key
/// (`Environment::next_retained_callable_key`) and publishes it as the
/// lambda's own range's CURRENT key (`Environment::record_lambda_key`)
/// — never keys by the range itself, so a second creation of the same
/// lambda text with a different closure (`make_adder(1)` vs.
/// `make_adder(100)`) never overwrites the first's still-live retained
/// value under a shared key.
///
/// Called at the few STATEMENT-level points that hold `&mut
/// Environment` just before the expression evaluates
/// (`check.rs::sink_value`, `summaries::interpret_body`'s `Stmt::Return`
/// arm) — `evaluate_expression` itself only ever reads `&Environment`,
/// so a lambda nested inside a call/constructor argument has no other
/// place to register before `evaluate_call`'s own argument evaluation
/// reads it. Every other expression shape (a `BinOp`, a display, a
/// comprehension, …) is not walked into — a lambda nested THERE is
/// outside this wave's five rows and stays the plain opaque value,
/// never a wrong answer, only a lambda this table does not yet retain.
pub fn register_retained_callables(expr: &Expr, environment: &mut Environment) {
    match expr {
        Expr::Lambda(lambda) => {
            register_retained_callables(lambda.body.as_ref(), environment);
            let placeholder = env::RetainedCallable::from_lambda(lambda, HashMap::new());
            let synthetic_def = placeholder.as_synthetic_def("<lambda>", lambda.range());
            let closure = summaries::free_variable_snapshot(&synthetic_def, environment);
            let key = environment.next_retained_callable_key();
            environment.record_retained_callable(key, env::RetainedCallable::from_lambda(lambda, closure));
            environment.record_lambda_key(lambda.range().start().to_u32(), key);
        }
        Expr::Call(call) => {
            register_retained_callables(call.func.as_ref(), environment);
            for argument in &call.arguments.args {
                register_retained_callables(argument, environment);
            }
            for keyword in &call.arguments.keywords {
                register_retained_callables(&keyword.value, environment);
            }
        }
        Expr::Attribute(attribute) => {
            register_retained_callables(attribute.value.as_ref(), environment);
        }
        _ => {}
    }
}

/// `callee(...)` where `callee` is a retained-callable value
/// (`env::retained_callable_key` reads `Some`) — resolves the call
/// through the SAME restricted interpreter an ordinary same-module
/// `def` call already uses (`summaries::call_result_with_enclosing`),
/// never a second one built for this table. `None` ONLY when `callee`
/// is not a retained-callable value at all — the signal `evaluate_
/// call`'s own caller reads to fall through to its other dispatch
/// arms. Once `callee` IS recognized as a retained-callable value,
/// this function always answers `Some` — a table miss (`environment`
/// never recorded this exact key) or an arity/interpretation decline
/// answers `Some(unknown())`, never `None`, so a caller never
/// mistakes "this really is a retained-callable call, and it
/// declined" for "try the ordinary def/builtin dispatch instead,"
/// which could read a stale or wrong same-module def of the same bare
/// name.
///
/// The retained body's own CLOSURE snapshot (free names read from the
/// environment AT THE MOMENT the value was created, `RetainedCallable`'s
/// own doc) seeds a throwaway environment that
/// `call_result_with_enclosing`'s `enclosing` parameter reads free
/// names from — the same closure-reading contract that function
/// already gives an ordinary nested `def`, reused rather than
/// duplicated. Positional arguments read through `positional_
/// arguments_for_retained_call` — the ordinary same-module keyword-
/// to-position mapping and arity checking, PLUS the one splicing
/// fallback a ParamSpec-forwarding wrapper needs (that function's own
/// doc).
pub(in super::super) fn retained_callable_call_result(
    callee: &AbstractValue,
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let key = env::retained_callable_key(callee)?;
    let Some(retained) = environment.retained_callable(key) else {
        return Some(unknown());
    };
    let def = retained.as_synthetic_def("<retained>", call.range());
    let Some(positional) =
        super::super::datetime::positional_arguments_for_retained_call(call, &def, environment, kernel)
    else {
        return Some(unknown());
    };
    // `enclosing` is ALWAYS the call site's own environment, carried
    // through a throwaway wrapper seeded with the retained body's own
    // closure snapshot (empty for a lambda/def that reads no free
    // name, the common case) — never `None` — so `call_result_with_
    // enclosing`'s own `fresh_body_environment` call always inherits
    // this call site's retained-callable table
    // (`Environment::inherit_retained_callables`'s own doc): a
    // retained value the closure carries (r-ast-census.py's `f`) still
    // resolves through the SAME shared table when `def`'s own body
    // calls it, and a retained value THIS call creates is still
    // reachable from `environment` (and everywhere `environment`'s own
    // `Arc` reaches) once this call returns.
    let mut closure_environment = Environment::new(std::collections::HashSet::new());
    closure_environment.inherit_retained_callables(environment);
    for (name, value) in &retained.closure {
        closure_environment.bind(name, value.clone());
    }
    let answer = summaries::call_result_with_enclosing(
        &def,
        &positional,
        environment.functions(),
        kernel,
        environment.call_depth(),
        Some(&closure_environment),
    );
    Some(answer.unwrap_or_else(unknown))
}

/// A same-module `def` call's positional argument values, in parameter
/// order: every positional call argument evaluated in place, then every
/// keyword argument mapped to its parameter's own position by NAME
/// (`summaries::call_result` itself takes only positional values, per
/// its own module doc — "Keyword arguments are the WIRING owner's job").
/// A keyword naming no parameter of `def`, or a starred positional
/// argument, declines the whole call — this file does not guess which
/// position a stray argument might occupy. Positions covered by BOTH a
/// positional and a keyword argument are impossible to build soundly
/// (CPython itself raises `TypeError: multiple values for argument` at
/// that call), so this function does not attempt to detect that
/// conflict — `bind_parameters`'s own arity check will decline once the
/// merged vector's length disagrees with what the call actually
/// supplied where relevant, and any un-caught double-binding is a
/// pre-existing gap this wave does not close.
///
/// `def`'s KEYWORD-ONLY parameters (`*, age`) are appended to the
/// name list AFTER `posonlyargs`/`args`, in declaration order — a
/// bare positional call argument can never land on one of those
/// trailing slots (Python's own call-site grammar puts every
/// positional argument before every keyword argument, so
/// `call.arguments.args` never has enough entries to reach past
/// `posonlyargs`/`args`'s own count), so a kwonly name only ever
/// fills from `call.arguments.keywords`'s own position lookup below —
/// the same "the CALLER passed the keyword" reach the mission asks
/// for (`only_keyword(age=200)`, e-class-and-function.py's own
/// `keyword_only_call`). `summaries::bind_parameters` reads this same
/// combined `posonlyargs+args+kwonlyargs` order back apart at its own
/// boundary (that function's own doc).
pub(in super::super) fn positional_arguments_for_def(
    call: &ruff_python_ast::ExprCall,
    def: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let parameter_names: Vec<&str> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
        .map(|parameter| parameter.parameter.name.id.as_str())
        .collect();
    if def.parameters.kwarg.is_some() {
        return positional_arguments_with_kwargs_dict(call, &parameter_names, environment, kernel);
    }
    positional_arguments_by_names(call, &parameter_names, environment, kernel)
}

/// The same keyword→position mapping `positional_arguments_by_names`
/// gives an ordinary def, PLUS one trailing slot for a `**kwargs`
/// parameter — e-class-and-function.py's own `gather_kwargs(**fields:
/// int)`: "the call site's keyword arguments fill the dict." Every
/// keyword argument that names one of `parameter_names` (a plain or
/// keyword-only parameter) maps to its own position exactly as before;
/// every OTHER named keyword argument (one `**kwargs` would collect at
/// runtime, functions.rst's own `**identifier` row: "receives a
/// dictionary containing... keyword arguments") is instead gathered
/// into ONE dict, built the identical way an ordinary `{...}` literal
/// is (`collection_models::dict_literal_value` — string keys only,
/// this domain's own dict restriction), and appended as the FINAL slot
/// of the returned vector. `summaries::bind_parameters` reads that
/// final slot back and binds it to the `kwarg` parameter's own name
/// (that function's own kwonly-slot doc names the identical trailing-
/// slot convention for kwonly params; this is the same convention one
/// slot further out). A starred positional argument, or a `**spread`
/// keyword argument (`f(**other)` — no single name to attribute to the
/// dict), declines the whole call: this function only ever collects
/// NAMED keyword arguments into the dict, never an unbounded spread.
pub(in super::super) fn positional_arguments_with_kwargs_dict(
    call: &ruff_python_ast::ExprCall,
    parameter_names: &[&str],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
    let mut positional: Vec<Option<AbstractValue>> = vec![None; parameter_names.len().max(call.arguments.args.len())];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        positional[index] = Some(evaluate_expression(arg, environment, kernel));
    }
    let mut kwargs_keys: Vec<collection_models::DictKey> = Vec::new();
    let mut kwargs_values: Vec<AbstractValue> = Vec::new();
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            // `f(**other)` — an unbounded spread, no single name to
            // attribute into the collected dict
            return None;
        };
        let value = evaluate_expression(&keyword.value, environment, kernel);
        match parameter_names.iter().position(|name| *name == arg_name.as_str()) {
            Some(position) => positional[position] = Some(value),
            None => {
                kwargs_keys.push(collection_models::DictKey::string(arg_name.as_str()));
                kwargs_values.push(value);
            }
        }
    }
    while matches!(positional.last(), Some(None)) {
        positional.pop();
    }
    let mut filled: Vec<AbstractValue> = positional.into_iter().collect::<Option<Vec<_>>>()?;
    let keys: Vec<Option<collection_models::DictKey>> = kwargs_keys.into_iter().map(Some).collect();
    filled.push(collection_models::dict_literal_value(&keys, &kwargs_values));
    Some(filled)
}

/// A same-module METHOD call's positional argument values, keyed by the
/// method's own parameter names WITH `self` EXCLUDED — the receiver is
/// never a call argument, so `method.method_call_result`'s own
/// non-`self` parameter list is the keyword-mapping target, one name
/// per non-receiver argument the call actually supplies.
///
/// `@staticmethod` declares no `self`/receiver slot at all
/// (`instances::method_call_result`'s own doc) — EVERY declared
/// parameter is the keyword-mapping target then, none excluded. Every
/// other member `def` keeps the `self`-splitting shape.
pub(in super::super) fn positional_arguments_for_method(
    call: &ruff_python_ast::ExprCall,
    method: &ruff_python_ast::StmtFunctionDef,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let parameters: Vec<_> = method.parameters.posonlyargs.iter().chain(method.parameters.args.iter()).collect();
    let is_static = method
        .decorator_list
        .iter()
        .any(|decorator| matches!(&decorator.expression, Expr::Name(name) if name.id.as_str() == "staticmethod"));
    let rest: Vec<_> = if is_static {
        parameters
    } else {
        let (_self_parameter, rest) = parameters.split_first()?;
        rest.to_vec()
    };
    let parameter_names: Vec<&str> = rest.iter().map(|parameter| parameter.parameter.name.id.as_str()).collect();
    positional_arguments_by_names(call, &parameter_names, environment, kernel)
}

/// The shared keyword→position mapping both `positional_arguments_for_def`
/// and `positional_arguments_for_method` need: every positional call
/// argument evaluated in place against `parameter_names`, then every
/// keyword argument mapped to its own name's position. A starred
/// positional argument, a `**kwargs`-spread keyword, or a keyword naming
/// no parameter all decline the whole call.
pub(in super::super) fn positional_arguments_by_names(
    call: &ruff_python_ast::ExprCall,
    parameter_names: &[&str],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
    let mut positional: Vec<Option<AbstractValue>> = vec![None; parameter_names.len().max(call.arguments.args.len())];
    for (index, arg) in call.arguments.args.iter().enumerate() {
        positional[index] = Some(evaluate_expression(arg, environment, kernel));
    }
    for keyword in &call.arguments.keywords {
        let Some(arg_name) = keyword.arg.as_ref() else {
            // `**kwargs`-spread call argument: no single parameter name
            // to map it to
            return None;
        };
        let Some(position) = parameter_names.iter().position(|name| *name == arg_name.as_str()) else {
            return None;
        };
        positional[position] = Some(evaluate_expression(&keyword.value, environment, kernel));
    }
    // trailing None slots are parameters this call left for their own
    // default — bind_parameters reads those; only a HOLE before a filled
    // slot (a positional gap no keyword covered) is unbuildable
    while matches!(positional.last(), Some(None)) {
        positional.pop();
    }
    positional.into_iter().collect()
}
