//! Recognizes a cross-language call that crosses into untrusted
//! territory — the `subprocess`/`os`/`asyncio`/`tempfile` shapes this
//! checker reads an argv, a payload, and a return leg from. One family
//! per sibling: `os_system` (`os.system("<runner> <script> < <infile> >
//! <outfile>")`'s file-legs crossing), `subprocess_run` (the stdin-json
//! and argv-json `subprocess.run(...)` shapes), `subprocess_check_output`
//! (the bare-result twin of `subprocess.run`), `popen` (the two-statement
//! `subprocess.Popen(...)`/`.communicate()` unit, flat and context-
//! manager), `asyncio_exec` (the awaited `asyncio.create_subprocess_exec`
//! twin), and `temp_file` (the three-statement named-temp-file carrier).
//!
//! `recognize_foreign_edge` is this module's own dispatcher: it reads
//! `statements[index]`'s own shape (an `Assign`, a walrus-bound call
//! already destructured by the caller, or a `With`) and routes to
//! whichever family's own recognizer can read it.

use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ExprName;
use ruff_python_ast::Stmt;

use crate::env::Environment;

use super::argv::RecognitionDecline;
use super::ForeignEdge;

mod asyncio_exec;
mod os_system;
mod popen;
mod subprocess_check_output;
mod subprocess_run;
mod temp_file;

use asyncio_exec::recognize_asyncio_create_subprocess_exec;
use os_system::recognize_os_system;
use popen::recognize_subprocess_popen;
use subprocess_check_output::recognize_subprocess_check_output;
use subprocess_run::recognize_subprocess_run;
use temp_file::json_dump_payload_of;

pub(in crate::foreign_edge) use os_system::os_system_return_read_of;
pub(in crate::foreign_edge) use popen::recognize_popen_context_manager_edge;
pub(in crate::foreign_edge) use temp_file::recognize_temp_file_edge;

/// A bare `Name` expression's own identifier text.
fn as_bare_name(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    }
}

/// Reads one statement as `<name> = subprocess.run(...)`,
/// `<name> = subprocess.check_output(...)`, the two-statement
/// `<name> = subprocess.Popen(...)` / `<a>, <b> = <name>.communicate(...)`
/// pair, or the two-statement AWAITED twin `<name> = await asyncio
/// .create_subprocess_exec(...)` / `<a>, <b> = await <name>.communicate(
/// ...)` — the four `subprocess`/`asyncio` callees whose argv shape and
/// keywords this checker reads.
///
/// `None` — not this shape at all, no sentence owed (including a plain
/// `subprocess.Popen(...)` whose very first statement is not even a
/// `subprocess` call, which is not this recognizer's concern at all).
/// `Some(Err(...))` — this IS a recognized `subprocess.*`/`asyncio.*` call
/// and something about its spelling stopped the resolution, so the caller
/// owes a sentence naming it.
pub(in crate::foreign_edge) fn recognize_foreign_edge(
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    // The `with subprocess.Popen(...) as process:` wrapping is handled
    // one level up, in `foreign_edge_at` itself, before this function is
    // even called: that shape's own consumer scan must run over the
    // WITH-BLOCK'S body, never over `statements` (the temp-file shape's
    // consumer, in contrast, is a SIBLING statement after the with-block,
    // which is exactly what this function's own `statements`/`index`
    // scan already serves). Reaching this branch at all with a `With`
    // statement therefore means the Popen wrapping already declined (or
    // this is not it), and only the temp-file shape remains to try.
    if let Stmt::With(with_stmt) = &statements[index] {
        return recognize_temp_file_edge(with_stmt, statements, index, environment, kernel);
    }
    let Stmt::Assign(assign) = &statements[index] else {
        return None;
    };
    if let Some(result) = recognize_os_system(statements, index, assign, environment) {
        return Some(result);
    }
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    // `await asyncio.create_subprocess_exec(...)` wraps its call in
    // `Expr::Await` — unwrapped and tried FIRST, since no recognized
    // `subprocess.*` callee is ever itself awaited (the sync callees run
    // to completion synchronously), so a bare `Expr::Call` never matches
    // this row and falls straight through to the unchanged sync path.
    if let Expr::Await(awaited) = assign.value.as_ref() {
        if let Expr::Call(call) = awaited.value.as_ref() {
            if let Some(result) =
                recognize_asyncio_create_subprocess_exec(statements, index, call, target, environment, kernel)
            {
                return Some(result);
            }
        }
        return None;
    }
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    recognize_subprocess_callee(call, target, statements, index, environment, kernel)
}

/// Reads `<target> = subprocess.<attr>(...)`'s CALLEE off an already-
/// destructured `call`/`target` pair — the module-name/shadow check and
/// the `run`/`check_output`/`Popen` dispatch, shared by
/// `recognize_foreign_edge`'s own `Stmt::Assign` path and
/// `foreign_edge_at_walrus_call`'s walrus-bound path (`if (result :=
/// subprocess.run(...)).returncode == 0:`), which has no `Stmt::Assign`
/// to destructure at all — a `Named` expression binds through
/// `Expr::Named::target`/`value` directly, the identical shape once the
/// wrapping statement is stripped away.
pub(in crate::foreign_edge) fn recognize_subprocess_callee(
    call: &ExprCall,
    target: &ExprName,
    statements: &[Stmt],
    index: usize,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Result<ForeignEdge, RecognitionDecline>> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    // a local binding named `subprocess` is not the module — the same
    // shadow-on-rebind rule every other builtin/module recognition in
    // this crate applies (relational_sum.rs's own `sum` shadow check)
    if module_name.id.as_str() != "subprocess" || environment.read("subprocess").is_some() {
        return None;
    }
    match attribute.attr.as_str() {
        "run" => recognize_subprocess_run(call, target, environment, index, kernel),
        "check_output" => recognize_subprocess_check_output(call, target, environment, index, kernel),
        "Popen" => recognize_subprocess_popen(statements, index, call, target, environment, kernel),
        _ => None,
    }
}
