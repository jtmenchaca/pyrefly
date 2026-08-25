//! The four harness shapes this reader recognizes in a module's
//! `if __name__ == "__main__":` block — the inbound channel(s) the
//! module reads and the JSON stdout it writes.

use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Number;
use ruff_python_ast::Stmt;
use serde_json::Value;
use serde_json::json;

/// The four harness shapes this reader recognizes in a module's
/// `if __name__ == "__main__":` block. Every other shape is not a
/// harness fact at all — `harness_shape` answers `None` for it, and the
/// artifact omits the `surface` key entirely.
pub(super) enum HarnessShape {
    /// `print(json.dumps(<f>(json.load(sys.stdin))))` — the inbound
    /// channel is stdin, one JSON payload.
    StdinJson { called: String },
    /// `print(json.dumps(<f>(float(sys.argv[<n>]))))`, the argv read
    /// possibly bound through one intermediate assignment first — the
    /// inbound channel is one argv string, parsed as a float.
    ArgvScalar { called: String, arg_index: i64 },
    /// `print(json.dumps(<f>(json.load(sys.stdin), <argv-read>)))` —
    /// TWO inbound channels, stdin's JSON payload handed to the callee's
    /// parameter 0 and the argv float handed to parameter 1
    /// (`level_gain_argv.py`'s own anatomy). The invariant a consumer
    /// relies on: `<f>`'s exported `entry` carries exactly two rows in
    /// this same order (entry[0] = the stdin parameter, entry[1] = the
    /// argv parameter), because this shape is recognized only when the
    /// call's two positional arguments match that order.
    StdinJsonArgvScalar { called: String, arg_index: i64 },
    /// `with open(sys.argv[<n>]) as <handle>: <payload> =
    /// json.load(<handle>)` followed by
    /// `print(json.dumps(<f>(<payload>)))` — the inbound channel is the
    /// JSON file named at that argv position, not stdin.
    FileJson { called: String, arg_index: i64 },
}

/// `harness_shape_json`'s JSON for one recognized shape — schema v2's
/// tagged union, the `stdin-json` and `argv-scalar` rows plus this
/// batch's `stdin-json-argv-scalar` and `file-json` rows.
pub(super) fn harness_shape_json(shape: &HarnessShape) -> Value {
    match shape {
        HarnessShape::StdinJson { called } => {
            json!({"kind": "stdin-json", "stdin": "json", "stdout": "json", "calls": called})
        }
        HarnessShape::ArgvScalar { called, arg_index } => {
            json!({"kind": "argv-scalar", "argIndex": arg_index, "parse": "float", "stdout": "json", "calls": called})
        }
        HarnessShape::StdinJsonArgvScalar { called, arg_index } => {
            json!({"kind": "stdin-json-argv-scalar", "stdin": "json", "argIndex": arg_index, "parse": "float", "stdout": "json", "calls": called})
        }
        HarnessShape::FileJson { called, arg_index } => {
            json!({"kind": "file-json", "argIndex": arg_index, "stdout": "json", "calls": called})
        }
    }
}

/// The module's `if __name__ == "__main__":` block read for one of the
/// four recognized harness shapes. `None` for a module with no main
/// block, or with a block of any other shape — the artifact omits the
/// harness key entirely then, and the consumer reads that absence as "no
/// harness fact".
pub(super) fn harness_shape(module: &ModModule) -> Option<HarnessShape> {
    for stmt in &module.body {
        let Stmt::If(if_stmt) = stmt else {
            continue;
        };
        if !is_main_guard(if_stmt.test.as_ref()) {
            continue;
        }
        return stdin_json_argv_scalar_harness(&if_stmt.body)
            .or_else(|| argv_scalar_harness(&if_stmt.body))
            .or_else(|| stdin_json_harness(&if_stmt.body))
            .or_else(|| file_json_harness(&if_stmt.body));
    }
    None
}

/// The stdin-JSON shape: some statement in the block is
/// `print(json.dumps(<f>(json.load(sys.stdin))))`, the `json.load(sys.stdin)`
/// read either written inline as the call's sole argument, or bound one
/// statement earlier by a plain assignment (`value = json.load(sys.stdin)`)
/// and then referenced by that same name — `D5.count.helper.py`'s own
/// anatomy (`level_gain_argv.py`'s already-recognized argv leg has the
/// identical one-statement indirection; this is the stdin leg's twin,
/// read through `argv_scalar_harness`'s exact intermediate-assignment
/// pattern rather than a new one). Any other shape — a second statement of
/// indirection, an assignment target that is not a bare name, an assigned
/// value that is not exactly `json.load(sys.stdin)` — answers `None`.
fn stdin_json_harness(body: &[Stmt]) -> Option<HarnessShape> {
    for (index, inner) in body.iter().enumerate() {
        let Stmt::Expr(expr_stmt) = inner else {
            continue;
        };
        if let Some(called) = harness_shape_call(expr_stmt.value.as_ref()) {
            return Some(HarnessShape::StdinJson { called });
        }
        let Some((called, argument)) = harness_sole_argument_call(expr_stmt.value.as_ref()) else {
            continue;
        };
        // The one intermediate assignment the brief allows, read exactly
        // as `argv_scalar_harness` reads it for its own single argument:
        // the statement directly before this one binds a bare name to
        // `json.load(sys.stdin)`, and this call's sole argument is that
        // same name.
        let Expr::Name(referenced) = argument else {
            continue;
        };
        let Some(Stmt::Assign(assign)) = index.checked_sub(1).and_then(|previous| body.get(previous)) else {
            continue;
        };
        let [target] = assign.targets.as_slice() else {
            continue;
        };
        let Expr::Name(bound) = target else {
            continue;
        };
        if bound.id.as_str() != referenced.id.as_str() {
            continue;
        }
        if is_stdin_json_load(assign.value.as_ref()) {
            return Some(HarnessShape::StdinJson { called });
        }
    }
    None
}

/// The argv-scalar shape: some statement in the block is
/// `print(json.dumps(<f>(<argv-read>)))`, where `<argv-read>` is
/// `float(sys.argv[<literal int>])` either written inline as the call's
/// sole argument, or bound one statement earlier by a plain assignment
/// (`gain = float(sys.argv[1])`) and then referenced by that same name.
/// Any other shape — a second argument beside the argv read, `int(...)`
/// or `str(...)` in place of `float(...)`, an argv-read expression this
/// reader does not recognize — answers `None`.
fn argv_scalar_harness(body: &[Stmt]) -> Option<HarnessShape> {
    for (index, inner) in body.iter().enumerate() {
        let Stmt::Expr(expr_stmt) = inner else {
            continue;
        };
        let Some((called, argument)) = harness_sole_argument_call(expr_stmt.value.as_ref()) else {
            continue;
        };
        if let Some(arg_index) = argv_float_read(argument) {
            return Some(HarnessShape::ArgvScalar { called, arg_index });
        }
        // The one intermediate assignment the brief allows: the
        // statement directly before this one binds a bare name to
        // `float(sys.argv[<n>])`, and this call's sole argument is that
        // same name.
        let Expr::Name(referenced) = argument else {
            continue;
        };
        let Some(Stmt::Assign(assign)) = index.checked_sub(1).and_then(|previous| body.get(previous)) else {
            continue;
        };
        let [target] = assign.targets.as_slice() else {
            continue;
        };
        let Expr::Name(bound) = target else {
            continue;
        };
        if bound.id.as_str() != referenced.id.as_str() {
            continue;
        }
        if let Some(arg_index) = argv_float_read(assign.value.as_ref()) {
            return Some(HarnessShape::ArgvScalar { called, arg_index });
        }
    }
    None
}

/// The mixed shape: some statement in the block is
/// `print(json.dumps(<f>(json.load(sys.stdin), <argv-read>)))` — a call
/// of exactly TWO positional arguments, the first `json.load(sys.stdin)`
/// and the second an argv float read (inline, or bound one statement
/// earlier by a plain assignment, the same intermediate-assignment leg
/// `argv_scalar_harness` reads) — `level_gain_argv.py`'s own anatomy.
/// Position is what carries the meaning: stdin's parse goes to the
/// callee's parameter 0, the argv float to parameter 1, which is why
/// this reader checks the arguments in that exact order rather than
/// accepting either order. Any other shape — one argument, a third
/// argument, the two arguments swapped, an `int(...)`/`str(...)` parse
/// — answers `None`.
fn stdin_json_argv_scalar_harness(body: &[Stmt]) -> Option<HarnessShape> {
    for (index, inner) in body.iter().enumerate() {
        let Stmt::Expr(expr_stmt) = inner else {
            continue;
        };
        let Some((called, first_argument, second_argument)) = harness_two_argument_call(expr_stmt.value.as_ref())
        else {
            continue;
        };
        if !is_stdin_json_load(first_argument) {
            continue;
        }
        if let Some(arg_index) = argv_float_read(second_argument) {
            return Some(HarnessShape::StdinJsonArgvScalar { called, arg_index });
        }
        // The one intermediate assignment the brief allows, read exactly
        // as `argv_scalar_harness` reads it for its own single argument.
        let Expr::Name(referenced) = second_argument else {
            continue;
        };
        let Some(Stmt::Assign(assign)) = index.checked_sub(1).and_then(|previous| body.get(previous)) else {
            continue;
        };
        let [target] = assign.targets.as_slice() else {
            continue;
        };
        let Expr::Name(bound) = target else {
            continue;
        };
        if bound.id.as_str() != referenced.id.as_str() {
            continue;
        }
        if let Some(arg_index) = argv_float_read(assign.value.as_ref()) {
            return Some(HarnessShape::StdinJsonArgvScalar { called, arg_index });
        }
    }
    None
}

/// The file-JSON shape: a `with open(sys.argv[<n>]) as <handle>:` block
/// whose body is exactly one assignment binding a bare name to
/// `json.load(<handle>)`, followed (with or without an intermediate
/// statement gap the scan does not require to be adjacent — the search
/// below just looks at every remaining statement) by
/// `print(json.dumps(<f>(<payload>)))` where `<payload>` is that same
/// bound name — `level_from_file.py`'s own anatomy. Any other shape (a
/// `with` body with more than the one assignment, a `json.load` receiver
/// other than the `with` target, a call whose sole argument is not the
/// loaded payload) answers `None`.
fn file_json_harness(body: &[Stmt]) -> Option<HarnessShape> {
    for (index, inner) in body.iter().enumerate() {
        let Stmt::With(with_stmt) = inner else {
            continue;
        };
        let [item] = with_stmt.items.as_slice() else {
            continue;
        };
        let Some(handle) = item.optional_vars.as_deref() else {
            continue;
        };
        let Expr::Name(handle_name) = handle else {
            continue;
        };
        let Some(opened) = single_argument_of(&item.context_expr, &CalleeSpelling::BareName("open")) else {
            continue;
        };
        let Some(arg_index) = argv_subscript_index(opened) else {
            continue;
        };
        let [Stmt::Assign(payload_assign)] = with_stmt.body.as_slice() else {
            continue;
        };
        let [payload_target] = payload_assign.targets.as_slice() else {
            continue;
        };
        let Expr::Name(payload_name) = payload_target else {
            continue;
        };
        let Some(loaded) = single_argument_of(payload_assign.value.as_ref(), &CalleeSpelling::Attribute("json", "load"))
        else {
            continue;
        };
        let Expr::Name(loaded_receiver) = loaded else {
            continue;
        };
        if loaded_receiver.id.as_str() != handle_name.id.as_str() {
            continue;
        }
        // The payload is read from every remaining statement after this
        // `with` block, not only the very next one — the brief's own
        // fixture places the `print` immediately after, and this reader
        // does not require that adjacency to hold for the shape to
        // count.
        for later in &body[index + 1..] {
            let Stmt::Expr(expr_stmt) = later else {
                continue;
            };
            let Some((called, argument)) = harness_sole_argument_call(expr_stmt.value.as_ref()) else {
                continue;
            };
            let Expr::Name(referenced) = argument else {
                continue;
            };
            if referenced.id.as_str() == payload_name.id.as_str() {
                return Some(HarnessShape::FileJson { called, arg_index });
            }
        }
    }
    None
}

/// `print(json.dumps(<f>(<argument>)))` read for `<f>` and its sole
/// argument — the same print/dumps wrapping `harness_shape_call` reads,
/// stopping short of reading what the innermost argument itself is
/// (`argv_float_read` and the caller's assignment-following do that).
fn harness_sole_argument_call<'a>(expr: &'a Expr) -> Option<(String, &'a Expr)> {
    let printed = single_argument_of(expr, &CalleeSpelling::BareName("print"))?;
    let dumped = single_argument_of(printed, &CalleeSpelling::Attribute("json", "dumps"))?;
    let Expr::Call(inner) = dumped else {
        return None;
    };
    let Expr::Name(called) = inner.func.as_ref() else {
        return None;
    };
    if !inner.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = inner.arguments.args.as_ref() else {
        return None;
    };
    Some((called.id.as_str().to_owned(), argument))
}

/// `print(json.dumps(<f>(<first>, <second>)))` read for `<f>` and its
/// exactly-two positional arguments, in order — the two-argument
/// counterpart of `harness_sole_argument_call`, for the mixed
/// stdin+argv shape where argument POSITION is the fact (parameter 0 is
/// the stdin parse, parameter 1 is the argv parse).
fn harness_two_argument_call<'a>(expr: &'a Expr) -> Option<(String, &'a Expr, &'a Expr)> {
    let printed = single_argument_of(expr, &CalleeSpelling::BareName("print"))?;
    let dumped = single_argument_of(printed, &CalleeSpelling::Attribute("json", "dumps"))?;
    let Expr::Call(inner) = dumped else {
        return None;
    };
    let Expr::Name(called) = inner.func.as_ref() else {
        return None;
    };
    if !inner.arguments.keywords.is_empty() {
        return None;
    }
    let [first, second] = inner.arguments.args.as_ref() else {
        return None;
    };
    Some((called.id.as_str().to_owned(), first, second))
}

/// Whether `expr` is exactly `json.load(sys.stdin)` — the mixed shape's
/// first argument, read without the print/dumps wrapping
/// `harness_shape_call` reads it under.
fn is_stdin_json_load(expr: &Expr) -> bool {
    let Some(stdin) = single_argument_of(expr, &CalleeSpelling::Attribute("json", "load")) else {
        return false;
    };
    is_sys_stdin(stdin)
}

/// Whether `expr` is exactly `sys.argv[<literal int>]`, read for that
/// literal index — the bare subscript `argv_float_read` reads underneath
/// its `float(...)` wrapping, and what the file shape's `with
/// open(sys.argv[<n>])` reads directly (no `float(...)` wrapping there).
fn argv_subscript_index(expr: &Expr) -> Option<i64> {
    let Expr::Subscript(subscript) = expr else {
        return None;
    };
    if !is_sys_argv(subscript.value.as_ref()) {
        return None;
    }
    let Expr::NumberLiteral(literal) = subscript.slice.as_ref() else {
        return None;
    };
    match &literal.value {
        Number::Int(value) => value.as_i64(),
        Number::Float(_) | Number::Complex { .. } => None,
    }
}

/// Whether `expr` is exactly `float(sys.argv[<literal int>])`, read for
/// that literal index. `int(...)`/`str(...)` in the parse position, a
/// non-literal or negative subscript, or any other spelling answers
/// `None` — this reader states only the one parse this unit recognizes.
fn argv_float_read(expr: &Expr) -> Option<i64> {
    let argument = single_argument_of(expr, &CalleeSpelling::BareName("float"))?;
    argv_subscript_index(argument)
}

/// Whether `expr` is `sys.argv` (or a bare `argv` a `from sys import
/// argv` would bind).
fn is_sys_argv(expr: &Expr) -> bool {
    match expr {
        Expr::Name(name) => name.id.as_str() == "argv",
        Expr::Attribute(attribute) => {
            attribute.attr.as_str() == "argv"
                && matches!(attribute.value.as_ref(), Expr::Name(receiver) if receiver.id.as_str() == "sys")
        }
        _ => false,
    }
}

/// Whether `test` is `__name__ == "__main__"` (either order).
fn is_main_guard(test: &Expr) -> bool {
    let Expr::Compare(compare) = test else {
        return false;
    };
    let ([CmpOp::Eq], [right]) = (compare.ops.as_ref(), compare.comparators.as_ref()) else {
        return false;
    };
    let names_dunder_name = |expr: &Expr| matches!(expr, Expr::Name(name) if name.id.as_str() == "__name__");
    let names_main = |expr: &Expr| {
        matches!(expr, Expr::StringLiteral(literal) if literal.value.to_str() == "__main__")
    };
    (names_dunder_name(compare.left.as_ref()) && names_main(right))
        || (names_main(compare.left.as_ref()) && names_dunder_name(right))
}

/// `print(json.dumps(<f>(json.load(sys.stdin))))` read for its `<f>`.
/// Every layer must match: a bare `print` call of one argument, a
/// `json.dumps` call of one argument, a bare-Name call of one argument,
/// and a `json.load(sys.stdin)` innermost. Any deviation answers `None`
/// — a harness this reader half-recognizes is not a harness fact.
fn harness_shape_call(expr: &Expr) -> Option<String> {
    let (called, argument) = harness_sole_argument_call(expr)?;
    let stdin = single_argument_of(argument, &CalleeSpelling::Attribute("json", "load"))?;
    if !is_sys_stdin(stdin) {
        return None;
    }
    Some(called)
}

/// How a harness layer's callee must be spelled.
enum CalleeSpelling {
    BareName(&'static str),
    Attribute(&'static str, &'static str),
}

/// `expr`'s single positional argument, when `expr` is a call to the
/// named callee with exactly one positional argument and no keywords.
fn single_argument_of<'a>(expr: &'a Expr, callee: &CalleeSpelling) -> Option<&'a Expr> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let matches_callee = match (callee, call.func.as_ref()) {
        (CalleeSpelling::BareName(wanted), Expr::Name(name)) => name.id.as_str() == *wanted,
        (CalleeSpelling::Attribute(module, attribute), Expr::Attribute(path)) => {
            path.attr.as_str() == *attribute
                && matches!(path.value.as_ref(), Expr::Name(receiver) if receiver.id.as_str() == *module)
        }
        _ => false,
    };
    if !matches_callee || !call.arguments.keywords.is_empty() {
        return None;
    }
    let [only] = call.arguments.args.as_ref() else {
        return None;
    };
    Some(only)
}

/// Whether `expr` is `sys.stdin` (or a bare `stdin` a `from sys import
/// stdin` would bind).
fn is_sys_stdin(expr: &Expr) -> bool {
    match expr {
        Expr::Name(name) => name.id.as_str() == "stdin",
        Expr::Attribute(attribute) => {
            attribute.attr.as_str() == "stdin"
                && matches!(attribute.value.as_ref(), Expr::Name(receiver) if receiver.id.as_str() == "sys")
        }
        _ => false,
    }
}
