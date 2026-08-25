use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtWith;
use ruff_text_size::Ranged;

use crate::env::Environment;

use super::super::argv::RecognitionDecline;
use super::super::argv::recognized_argv;
use super::super::argv::resolve_target_path;
use super::super::argv::script_extension_decline;
use super::super::keywords::json_dumps_argument_of;
use super::super::keywords::subprocess_popen_keywords_of;
use super::super::Channel;
use super::super::ForeignEdge;
use super::super::ResultRead;
use super::as_bare_name;

/// `<name> = subprocess.Popen(["node", "<script>.ts"], stdin=subprocess
/// .PIPE, stdout=subprocess.PIPE, text=True)` immediately followed by
/// `<stdout_name>, <_> = <name>.communicate(json.dumps(<payload>))` —
/// the SAME two-statement-unit discipline `foreign_edge_at`'s own
/// return-leg scan already applies to the call-and-its-consumer, here
/// applied one statement earlier: `.communicate()`'s own call is not a
/// consumer to find again, it is where the payload and the captured
/// name are read, so recognition consumes it here rather than leaving
/// it for `sole_parse_consumer_of` to (mis)count as a second statement
/// writing the name.
pub(super) fn recognize_subprocess_popen(
    statements: &[Stmt],
    index: usize,
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let call_range = call.range();
    let reading = match recognized_argv(call, environment, kernel)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    if let Some(decline) = subprocess_popen_keywords_of(call) {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(next) = statements.get(index + 1) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} through Popen and nothing follows it in this body, so the \
                checker cannot find the .communicate() call that reads the captured output back",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    let Some((stdout_name, payload)) = communicate_call_of(next, target.id.as_str()) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "the statement after this Popen call is not exactly `<name>, <name> = {}.communicate(\
                json.dumps(...))`, so the checker cannot find the captured output or the outbound payload",
                target.id.as_str()
            ),
            range: call_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: stdout_name,
        result_read: ResultRead::Bare,
        consumer_scan_from: index + 1,
        runner: reading.runner,
    }))
}

/// `with subprocess.Popen(["node", "<script>.ts"], stdin=subprocess.PIPE,
/// stdout=subprocess.PIPE, text=True) as process: <stdout>, _ = process
/// .communicate(json.dumps(<payload>)); return json.loads(<stdout>)` —
/// the idiomatic context-manager spelling of the flat two-statement
/// Popen/`.communicate()` unit `recognize_subprocess_popen` already
/// reads, here read against the with-block's OWN body instead of the
/// statements that follow it (`level_via_popen_context_manager`'s own
/// shape): the with's context expression is the Popen call itself, the
/// `as` target is the name `.communicate()` is called on, and both the
/// `.communicate()` assign and its own consumer sit INSIDE this
/// with-block's body, never as siblings after it.
///
/// `None` — not this shape at all: the context expression is not
/// `subprocess.Popen(...)` (a shadowed `subprocess` name reads the same
/// as not-this-shape, mirroring every other recognizer's shadow-on-
/// rebind check), or the `with` binds no bare name via `as`. `Some(Err(
/// ...))` — this IS a recognized Popen context manager and something
/// about its spelling, or the `.communicate()` call that must open its
/// own body, stops the resolution.
pub(in crate::foreign_edge) fn recognize_popen_context_manager_edge(
    with_stmt: &StmtWith,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let with_range = with_stmt.range();
    let [item] = with_stmt.items.as_slice() else {
        return None;
    };
    let Expr::Call(call) = &item.context_expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "subprocess"
        || environment.read("subprocess").is_some()
        || attribute.attr.as_str() != "Popen"
    {
        return None;
    }
    let Some(process_name) = item.optional_vars.as_deref().and_then(as_bare_name) else {
        return Some(Err(RecognitionDecline {
            message: "this subprocess.Popen(...) with-statement binds no bare name via 'as', so the checker \
                cannot find the .communicate() call that reads the captured output back"
                .to_owned(),
            range: with_range,
        }));
    };
    let call_range = call.range();
    let reading = match recognized_argv(call, environment, kernel)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    if let Some(decline) = subprocess_popen_keywords_of(call) {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(first) = with_stmt.body.first() else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's body is empty, so the checker cannot find the .communicate() call that \
                reads {process_name}'s captured output back"
            ),
            range: with_range,
        }));
    };
    let Some((stdout_name, payload)) = communicate_call_of(first, process_name) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this with-block's first statement is not exactly `<name>, <name> = {process_name}.communicate(\
                json.dumps(...))`, so the checker cannot find the captured output or the outbound payload"
            ),
            range: with_range,
        }));
    };
    Some(Ok(ForeignEdge {
        call: call_range,
        target_path: resolve_target_path(&reading.script_text),
        payload,
        channel: Channel::Stdin,
        result_name: stdout_name,
        result_read: ResultRead::Bare,
        // The `.communicate()` statement sits at the with-body's own
        // index 0 — the return leg's sole-consumer scan (over
        // `with_stmt.body`, per `foreign_edge_at`'s own routing) starts
        // looking the statement AFTER it, exactly as the flat Popen
        // shape's `index + 1` does relative to its own statement list.
        consumer_scan_from: 0,
        runner: reading.runner,
    }))
}

/// Whether a call is exactly `subprocess.run(...)` — a shadowed
/// `subprocess` name is not the module, mirroring every other
/// recognizer's shadow-on-rebind check.
pub(super) fn is_subprocess_run_call(call: &ExprCall, environment: &Environment) -> bool {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "subprocess" && environment.read("subprocess").is_none() && attribute.attr.as_str() == "run"
}

/// Reads `<a>, <b> = <popen_name>.communicate(json.dumps(<payload>))` —
/// exactly a two-element tuple target, a call to `.communicate` on the
/// exact name Popen bound, with exactly one positional `json.dumps(...)`
/// argument. Answers the first target name (the captured stdout text)
/// and the payload expression.
pub(super) fn communicate_call_of(statement: &Stmt, popen_name: &str) -> Option<(String, Expr)> {
    let Stmt::Assign(assign) = statement else {
        return None;
    };
    let [Expr::Tuple(targets)] = assign.targets.as_slice() else {
        return None;
    };
    let [Expr::Name(stdout_name), _] = targets.elts.as_slice() else {
        return None;
    };
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != popen_name || attribute.attr.as_str() != "communicate" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    let payload = json_dumps_argument_of(argument)?;
    Some((stdout_name.id.as_str().to_owned(), payload))
}
