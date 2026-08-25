use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;

use crate::env::Environment;

use super::super::argv::RecognitionDecline;
use super::super::argv::asyncio_argv_runner_and_script;
use super::super::argv::resolve_target_path;
use super::super::argv::script_extension_decline;
use super::super::keywords::asyncio_create_subprocess_exec_keywords_of;
use super::super::keywords::json_dumps_argument_of;
use super::super::Channel;
use super::super::ForeignEdge;
use super::super::ResultRead;

/// `<name> = await asyncio.create_subprocess_exec("node", "<script>.ts",
/// stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)`
/// immediately followed by `<stdout_name>, <_> = await <name>.communicate(
/// json.dumps(<payload>).encode())` — the awaited twin of
/// `recognize_subprocess_popen`'s own two-statement Popen/`.communicate()`
/// unit. Three deltas from the sync shape, all read through, never a
/// second pipeline: the runner/script argv rides `create_subprocess_exec`'s
/// own VARIADIC positional arguments (`program, *args`) rather than one
/// list literal; both this call and the `.communicate()` call it awaits
/// are wrapped in `Expr::Await`, unwrapped before each is read; and the
/// payload/return values ride BYTES (`json.dumps(...).encode()` going out,
/// `json.loads(...)` reading a `.decode()`-unwrapped — or bare bytes —
/// binding coming back), so `text=True` is neither passed nor required
/// here, unlike every synchronous shape this crate recognizes.
///
/// `None` — not this shape at all (the awaited call is not
/// `asyncio.create_subprocess_exec(...)`, or a shadowed `asyncio` name).
/// `Some(Err(...))` — this IS the recognized awaited call and something
/// about its own spelling, or the awaited `.communicate()` call that must
/// follow it, stops the resolution.
pub(super) fn recognize_asyncio_create_subprocess_exec(
    statements: &[Stmt],
    index: usize,
    call: &ExprCall,
    target: &ExprName,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    // a local binding named `asyncio` is not the module — the same
    // shadow-on-rebind rule every other builtin/module recognition in
    // this crate applies (relational_sum.rs's own `sum` shadow check)
    if module_name.id.as_str() != "asyncio"
        || environment.read("asyncio").is_some()
        || attribute.attr.as_str() != "create_subprocess_exec"
    {
        return None;
    }
    let call_range = call.range();
    let reading = match asyncio_argv_runner_and_script(call, environment, kernel)? {
        Ok(reading) => reading,
        Err(decline) => return Some(Err(decline)),
    };
    if let Some(decline) = script_extension_decline(&reading.script_text, reading.runner, call_range) {
        return Some(Err(decline));
    }
    if let Some(decline) = asyncio_create_subprocess_exec_keywords_of(call) {
        return Some(Err(RecognitionDecline { message: decline, range: call_range }));
    }
    let Some(next) = statements.get(index + 1) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "this call runs {} on {} through asyncio.create_subprocess_exec and nothing follows it in \
                this body, so the checker cannot find the awaited .communicate() call that reads the \
                captured output back",
                reading.runner.word(),
                reading.script_text
            ),
            range: call_range,
        }));
    };
    let Some((stdout_name, payload)) = awaited_communicate_call_of(next, target.id.as_str()) else {
        return Some(Err(RecognitionDecline {
            message: format!(
                "the statement after this asyncio.create_subprocess_exec call is not exactly `<name>, \
                <name> = await {}.communicate(json.dumps(...))` (optionally `.encode()`-wrapped), so the \
                checker cannot find the captured output or the outbound payload",
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

/// Reads `<a>, <b> = await <process_name>.communicate(json.dumps(<payload>)
/// .encode())` — the awaited twin of `communicate_call_of`: the call
/// itself is wrapped in `Expr::Await` (unwrapped first), and the
/// positional argument is `json.dumps(...)` OPTIONALLY wrapped in
/// `.encode()` — the call sends bytes on the wire (`library/asyncio-
/// subprocess.rst`: `Process.communicate`'s own `input` parameter is
/// `bytes | None`), and `.encode()` carries the identical JSON text
/// `json_dumps_argument_of` already reads through; the wrapper itself is
/// stripped before that shared reader ever sees the expression, so a
/// bare `json.dumps(...)` (no `.encode()` at all — a caller relying on
/// `communicate`'s own str-to-bytes convenience, if the target's own
/// stdlib version admits it) is read exactly the same way.
pub(super) fn awaited_communicate_call_of(statement: &Stmt, process_name: &str) -> Option<(String, Expr)> {
    let Stmt::Assign(assign) = statement else {
        return None;
    };
    let [Expr::Tuple(targets)] = assign.targets.as_slice() else {
        return None;
    };
    let [Expr::Name(stdout_name), _] = targets.elts.as_slice() else {
        return None;
    };
    let Expr::Await(awaited) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Call(call) = awaited.value.as_ref() else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != process_name || attribute.attr.as_str() != "communicate" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    let unwrapped = unwrap_bytes_encode(argument);
    let payload = json_dumps_argument_of(unwrapped)?;
    Some((stdout_name.id.as_str().to_owned(), payload))
}

/// Strips a trailing `.encode()` call off an expression — `<expr>.encode(
/// )` with no arguments and no keywords answers `<expr>` itself; every
/// other shape (a bare expression with no `.encode()` at all, or a
/// receiver method call `.encode()` does not directly wrap) answers the
/// expression unchanged. Shared by the outbound payload's own `json.dumps(
/// ...).encode()` unwrap and would apply identically to a return-leg
/// `.decode()` unwrap were one written the same way (`unwrap_bytes_decode`
/// is the separate, differently-shaped reader that one needs, since it
/// reads a NAME rather than a call).
pub(super) fn unwrap_bytes_encode(expression: &Expr) -> &Expr {
    let Expr::Call(call) = expression else {
        return expression;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return expression;
    };
    if attribute.attr.as_str() != "encode" || !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return expression;
    }
    attribute.value.as_ref()
}
