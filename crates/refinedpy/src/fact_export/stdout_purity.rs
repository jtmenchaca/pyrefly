//! CHANNEL PURITY: whether a def's body writes nothing to stdout — the
//! effect fact CROSS-LANGUAGE-EDGE.md §5 names, and the premise the
//! wire's own claim rests on (the wire IS stdout, so a stray write
//! inside the target corrupts the payload). A conservative syntactic
//! scan, transitive through same-module calls.

use std::collections::HashMap;
use std::collections::HashSet;

use ruff_python_ast::Arguments;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprCall;
use ruff_python_ast::ModModule;
use ruff_python_ast::StmtFunctionDef;

use super::traversal::walk_statement_expressions;
use super::top_level_defs;

/// Whether `def`'s body writes nothing to stdout — the effect fact
/// CROSS-LANGUAGE-EDGE.md §5 ("Channel purity") names, and the premise
/// the wire's own claim rests on: the wire IS stdout, so a stray write
/// inside the target corrupts the payload.
///
/// A CONSERVATIVE syntactic scan: any `print(...)` call, any
/// `sys.stdout.<anything>(...)` / `sys.stdout.write` reference, and any
/// `.write(...)` on a receiver spelled `stdout` counts as a write.
/// Transitive through the SAME-MODULE defs the body calls, capped at the
/// module (a call this module does not declare — an import, a builtin
/// this scan does not model — makes the answer false, since the scan
/// cannot see that body).
pub(super) fn writes_nothing_to_stdout(def: &StmtFunctionDef, module: &ModModule) -> bool {
    let module_defs: HashMap<&str, &StmtFunctionDef> = top_level_defs(module)
        .map(|candidate| (candidate.name.id.as_str(), candidate))
        .collect();
    let mut visited: HashSet<String> = HashSet::new();
    body_is_stdout_pure(&def.body, &module_defs, &mut visited)
}

/// One body's scan, following every same-module call it makes.
/// `visited` names the defs already scanned, so a recursive or mutually
/// recursive call terminates (a def already being scanned adds nothing
/// new to the answer).
fn body_is_stdout_pure(
    body: &[ruff_python_ast::Stmt],
    module_defs: &HashMap<&str, &StmtFunctionDef>,
    visited: &mut HashSet<String>,
) -> bool {
    // The scan collects first and decides second: the decision recurses
    // into a callee's own body, which would otherwise need `visited`
    // borrowed inside the traversal closure that already borrows it.
    let mut writes_stdout = false;
    let mut called_names: Vec<String> = Vec::new();
    let mut has_opaque_call = false;
    for stmt in body {
        walk_statement_expressions(stmt, &mut |expr| {
            if expression_writes_stdout(expr) {
                writes_stdout = true;
                return;
            }
            let Expr::Call(call) = expr else {
                return;
            };
            // A captured-stdout spawn call (`subprocess.run(...,
            // capture_output=True)`, `subprocess.Popen(...,
            // stdout=subprocess.PIPE)`, `subprocess.check_output(...)`, the
            // awaited `asyncio.create_subprocess_exec(...,
            // stdout=asyncio.subprocess.PIPE)`) pipes the CHILD's stdout
            // into this call's own return value; it writes nothing to the
            // PARENT's stdout on its own account, so it is checked BEFORE
            // the opaque-attribute-call fallback below — a shape this
            // table does not admit still falls through to that fallback
            // and refuses, exactly as before.
            if is_captured_stdout_spawn_call(call) {
                return;
            }
            match call.func.as_ref() {
                Expr::Name(callee) => called_names.push(callee.id.as_str().to_owned()),
                // `<process>.communicate(...)` is the ONLY way either
                // admitted PIPE row (`subprocess.Popen`, the awaited
                // `asyncio.create_subprocess_exec`) ever reads its
                // captured stdout back — `library/subprocess.rst`,
                // `Popen.communicate`: "the data will be … strings if
                // streams were opened in text mode", read from the pipe
                // this call's own captured-spawn row already admitted;
                // it writes nothing to the PARENT's stdout on its own
                // account either, so it is admitted here BY METHOD NAME
                // alone (this scan carries no alias table tying the
                // receiver name back to its own `Popen`/
                // `create_subprocess_exec` call site) — the same
                // by-name-only posture the Go twin's own banner accepts
                // for its three synchronous spawn names.
                Expr::Attribute(attribute) if attribute.attr.as_str() == "communicate" => {}
                // an attribute call (`obj.method(...)`, `math.sqrt(...)`)
                // reaches no same-module def this scan can follow; the
                // stdout-writing attribute shapes are already caught by
                // `expression_writes_stdout` above, and a method body on
                // an instance is out of this scan's reach — so any
                // receiver outside the modelled stdlib list refuses the
                // claim.
                other => {
                    if is_opaque_receiver_call(other) {
                        has_opaque_call = true;
                    }
                }
            }
        });
    }
    if writes_stdout || has_opaque_call {
        return false;
    }
    for name in called_names {
        if is_pure_builtin(&name) {
            continue;
        }
        let Some(callee_def) = module_defs.get(name.as_str()) else {
            // a name this module does not declare: an import, or a
            // builtin outside the modelled list. The scan cannot see
            // that body, so it cannot claim the channel is clean.
            return false;
        };
        // a def already being scanned adds nothing new to the answer,
        // which is what makes a recursive or mutually recursive call
        // terminate here
        if !visited.insert(name) {
            continue;
        }
        if !body_is_stdout_pure(&callee_def.body, module_defs, visited) {
            return false;
        }
    }
    true
}

/// Whether `expr` is itself a write to stdout: `print(...)`, a
/// `sys.stdout` attribute path, or a `.write(...)` whose receiver is
/// spelled `stdout`.
fn expression_writes_stdout(expr: &Expr) -> bool {
    match expr {
        // `print(...)`, or any call whose receiver path names stdout
        // (`sys.stdout.write(...)`, `sys.stdout.flush()`).
        Expr::Call(call) => match call.func.as_ref() {
            Expr::Name(callee) => callee.id.as_str() == "print",
            Expr::Attribute(attribute) => attribute_path_reaches_stdout(attribute.value.as_ref()),
            _ => false,
        },
        // A bare `sys.stdout` reference (handed to a writer this scan
        // cannot follow) is itself enough to refuse the claim. This is
        // the actual STREAM object — `is_sys_stdout_path` reads the
        // receiver chain back to `sys` (or a `from sys import stdout`
        // bare name) rather than matching any identifier merely NAMED
        // `.stdout` (a captured subprocess result's own `result.stdout`
        // — the field `subprocess.run(..., capture_output=True)` pipes
        // the child's captured bytes into — is a plain read of THAT
        // field, never a reference to the parent's own stream, and must
        // not trip this rule).
        Expr::Attribute(_) => is_sys_stdout_path(expr),
        _ => false,
    }
}

/// Whether `expr` is itself a write to stdout (called with `expr` as the
/// receiver of `.write(...)`, `.flush()`, etc — see the call above) —
/// resolves the SAME `sys.stdout` stream object `attribute_path_reaches_
/// stdout`'s callers already assume, kept as its own name since a
/// receiver check and a bare-reference check read the identical path.
fn attribute_path_reaches_stdout(expr: &Expr) -> bool {
    is_sys_stdout_path(expr)
}

/// Whether `expr` is exactly `sys.stdout`, or a bare `stdout` a `from sys
/// import stdout` would bind — the parent's own stream object, never any
/// OTHER attribute merely spelled `.stdout` (a captured subprocess
/// result's `result.stdout`, a `Popen` object's `.stdout` pipe handle).
/// The root of the attribute chain must be the name `sys`; a chain
/// rooted in any other name (a local variable, a captured result) does
/// not match here regardless of how many `.stdout`-named links follow.
fn is_sys_stdout_path(expr: &Expr) -> bool {
    match expr {
        Expr::Name(name) => name.id.as_str() == "stdout",
        Expr::Attribute(attribute) => {
            attribute.attr.as_str() == "stdout"
                && matches!(attribute.value.as_ref(), Expr::Name(receiver) if receiver.id.as_str() == "sys")
        }
        _ => false,
    }
}

/// Whether an attribute-callee shape is one this scan cannot follow and
/// therefore refuses the purity claim for. A `math.<fn>(...)` /
/// `json.<fn>(...)` call on a stdlib module this checker already models
/// writes nothing to the channel; every other receiver (an instance
/// method, an imported module's function) is opaque here.
fn is_opaque_receiver_call(func: &Expr) -> bool {
    let Expr::Attribute(attribute) = func else {
        return true;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return true;
    };
    !matches!(receiver.id.as_str(), "math" | "json")
}

/// Whether `call` spawns a child process through a form whose stdout is
/// CAPTURED rather than written to the parent's own stdout — the twin of
/// the Go scan's `isCapturedStdoutSpawnCall`
/// (fact_export_purity.go:309-362), admitted per the SAME documented
/// `subprocess`/`asyncio.subprocess` capture semantics, cited per row:
///
/// - `subprocess.run(..., capture_output=True)`: `capture_output=True` is
///   shorthand for `stdout=PIPE, stderr=PIPE` (`library/subprocess.rst`,
///   `subprocess.run`), which pipes the child's stdout into the returned
///   `CompletedProcess.stdout` rather than writing it to the parent's own
///   stdout. WITHOUT `capture_output=True` (or an equivalent explicit
///   `stdout=PIPE`), the default is `stdout=None`, which means the child
///   INHERITS the parent's stdout — the call is then a real write, and
///   this table refuses it.
/// - `subprocess.check_output(...)`: captures by definition —
///   `library/subprocess.rst`, `subprocess.check_output`: "the return
///   value ... is the stdout." Admitted UNCONDITIONALLY: the call has no
///   `stdout` keyword to override at all, so no shape defeats the
///   capture.
/// - `subprocess.Popen(..., stdout=subprocess.PIPE)`: the same PIPE
///   sentinel as `run`'s shorthand expands to, read directly here since
///   `Popen` has no `capture_output` convenience of its own — the child's
///   stdout is a pipe this call's own returned `Popen` object reads back
///   from (`.communicate()`/`.stdout`), never the parent's channel.
///   WITHOUT `stdout=subprocess.PIPE`, the default is `stdout=None`
///   (inherited), refused the same as `run`'s uncaptured default.
/// - the awaited `asyncio.create_subprocess_exec(...,
///   stdout=asyncio.subprocess.PIPE)`: `library/asyncio-subprocess.rst`
///   states the identical PIPE-sentinel contract as the synchronous
///   `Popen`, re-exported under the `asyncio.subprocess` namespace — the
///   child's stdout is captured into the process object's own
///   `.communicate()`/`.stdout`, never the parent's stdout.
///
/// A call this table does not recognize as one of these four exact
/// shapes (a different receiver, a different attribute name, a `stdio`
/// keyword this reader cannot read as one of the two admitted literal
/// forms) answers `false` — the caller's own opaque-call fallback then
/// refuses the purity claim, the same conservative-only-admits posture
/// the rest of this scan takes.
fn is_captured_stdout_spawn_call(call: &ExprCall) -> bool {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return false;
    };
    match (receiver.id.as_str(), attribute.attr.as_str()) {
        ("subprocess", "run") => keyword_bool(&call.arguments, "capture_output") == Some(true),
        // check_output always captures: no keyword of its own could
        // defeat it, so every call shape reaches this row
        ("subprocess", "check_output") => true,
        ("subprocess", "Popen") => {
            keyword_matches(&call.arguments, "stdout", is_subprocess_pipe_sentinel)
        }
        ("asyncio", "create_subprocess_exec") => {
            keyword_matches(&call.arguments, "stdout", is_asyncio_subprocess_pipe_sentinel)
        }
        _ => false,
    }
}

/// `arguments`'s own `name=` keyword read as a boolean literal (`True`/
/// `False`) — `Some(value)` when the keyword is present and its value is
/// a plain `BooleanLiteral`; `None` when the keyword is absent OR its
/// value is not a literal this reader can pin down (a computed
/// expression, a name), which this table's callers always treat the same
/// as "the keyword is absent" — a shape this scan cannot read never
/// registers as the safe case on invented grounds.
fn keyword_bool(arguments: &Arguments, name: &str) -> Option<bool> {
    arguments.keywords.iter().find_map(|keyword| {
        if keyword.arg.as_ref()?.as_str() != name {
            return None;
        }
        match &keyword.value {
            Expr::BooleanLiteral(literal) => Some(literal.value),
            _ => None,
        }
    })
}

/// Whether `arguments` carries a `name=` keyword whose value satisfies
/// `predicate` — the read `is_captured_stdout_spawn_call` uses for the
/// `stdout=subprocess.PIPE` / `stdout=asyncio.subprocess.PIPE` rows,
/// where absence of the keyword (the inherited-stdout default) and a
/// present-but-different value both answer `false` alike.
fn keyword_matches(arguments: &Arguments, name: &str, predicate: fn(&Expr) -> bool) -> bool {
    arguments
        .keywords
        .iter()
        .any(|keyword| keyword.arg.as_ref().is_some_and(|arg| arg.as_str() == name) && predicate(&keyword.value))
}

/// Whether `expr` is exactly `subprocess.PIPE`.
fn is_subprocess_pipe_sentinel(expr: &Expr) -> bool {
    let Expr::Attribute(attribute) = expr else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "subprocess" && attribute.attr.as_str() == "PIPE"
}

/// Whether `expr` is exactly `asyncio.subprocess.PIPE` — the awaited
/// shape's own two-level spelling of the same PIPE sentinel
/// (`library/asyncio-subprocess.rst`: re-exported under the
/// `asyncio.subprocess` namespace), unlike the sync shape's one-level
/// `subprocess.PIPE`.
fn is_asyncio_subprocess_pipe_sentinel(expr: &Expr) -> bool {
    let Expr::Attribute(pipe_attribute) = expr else {
        return false;
    };
    if pipe_attribute.attr.as_str() != "PIPE" {
        return false;
    }
    let Expr::Attribute(subprocess_attribute) = pipe_attribute.value.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = subprocess_attribute.value.as_ref() else {
        return false;
    };
    module_name.id.as_str() == "asyncio" && subprocess_attribute.attr.as_str() == "subprocess"
}

/// The builtins this scan knows write nothing to stdout. A name outside
/// this list and outside the module's own defs refuses the claim, so the
/// list only ever ADMITS a fact; it never widens one.
fn is_pure_builtin(name: &str) -> bool {
    matches!(
        name,
        "abs" | "all"
            | "any"
            | "bool"
            | "dict"
            | "divmod"
            | "enumerate"
            | "float"
            | "int"
            | "len"
            | "list"
            | "max"
            | "min"
            | "pow"
            | "range"
            | "round"
            | "set"
            | "sorted"
            | "str"
            | "sum"
            | "tuple"
            | "zip"
    )
}
