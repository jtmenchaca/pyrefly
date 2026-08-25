use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::diagnostic_sentences;
use crate::foreign_edge_artifact::ForeignTsArtifact;

use super::ResultRead;
use super::ForeignEdgeOutcome;

/// Builds the `Fired` outcome: an RTS7001 sentence with the target's own
/// provenance appended, the way the Go twin's `foreignMessage` does.
/// `consumer` always starts `None` here — `fire_at` runs from inside the
/// OUTBOUND leg's own check, before the return leg's consumer scan has
/// run at all; `finish_recognized_edge`/`finish_recognized_edge_from_start`
/// fill it in once discharge answers `Fired`, running that scan the same
/// way they would for a green crossing.
pub(super) fn fire_at(range: TextRange, said: String, artifact: &ForeignTsArtifact) -> ForeignEdgeOutcome {
    ForeignEdgeOutcome::Fired {
        message: diagnostic_sentences::foreign_crossing_refusal(
            &said,
            &artifact.target_file,
            artifact.called.provenance_line,
            &artifact.called.provenance_said,
        ),
        range,
        consumer: None,
    }
}

/* ── the return leg ───────────────────────────────────────────────── */

/// What the return leg's own consumer scan found — three outcomes, only
/// two of which are a blocker:
///
///   - `Found`: exactly one `json.loads(...)` reads the result, and no
///     write to the name intervenes — the target's return fact attaches
///     here.
///   - `NoneFound`: nothing parses the result as JSON at all — NOT a
///     blocker (`return_leg_outcome`'s own doc): the outbound leg already
///     judged, and a result the body reads some other way (`d-data-legs
///     .py`'s own `level_via_raw_stdout`: `float(result.stdout)`, never
///     `json.loads`) owes no return fact and no decline either.
///   - `Blocked`: TWO OR MORE parses (one published fact cannot stand for
///     two nodes) or an intervening WRITE to the name (the value a parse
///     would read is then not the value the call produced) — a real
///     blocker, named.
///
/// A parse inside a nested function body is not counted: that scope
/// runs an unstated number of times, so the fact cannot be pinned to one
/// evaluation.
pub(super) enum ParseConsumer {
    Found(TextRange),
    NoneFound,
    Blocked(String),
}

/// Finds the `json.loads(<result_name>.stdout)` (or, for `result_read
/// == Bare`, the plain `json.loads(<result_name>)`) node the target's
/// return fact attaches to, scanning the statements AFTER `index` in
/// the same function — the same same-function, count-the-occurrences
/// discipline the Go twin's `soleParseConsumerOf` uses.
pub(super) fn sole_parse_consumer_of(
    statements: &[Stmt],
    index: usize,
    result_name: &str,
    result_read: &ResultRead,
) -> ParseConsumer {
    sole_parse_consumer_from(&statements[index + 1..], result_name, result_read)
}

/// `sole_parse_consumer_of`'s own scan, taking the slice to scan
/// directly rather than a call's own index plus one — the walrus-bound
/// entry point (`foreign_edge_at_walrus_call`) has no call STATEMENT to
/// skip past at all (the call sits inside the `if` TEST, not as a member
/// of the arm body), so its whole arm body is scanned from its own
/// start, never offset by one.
pub(super) fn sole_parse_consumer_from(statements: &[Stmt], result_name: &str, result_read: &ResultRead) -> ParseConsumer {
    let mut found: Option<TextRange> = None;
    let mut count = 0usize;
    let mut written = false;
    for statement in statements {
        if statement_writes_name(statement, result_name) {
            written = true;
        }
        foreign_parse_calls_in(statement, result_name, result_read, &mut found, &mut count);
    }
    // NOTHING reads the result as JSON here at all: not a blocker — the
    // outbound leg already judged (`return_leg_outcome`'s own doc), and a
    // result read some other way (or not read at all) owes no return
    // fact and no decline. A WRITE to the name is only a real hazard for
    // a parse that actually happened; a written-but-never-parsed name
    // reads nothing stale, since nothing reads it at all.
    if count == 0 {
        return ParseConsumer::NoneFound;
    }
    if written {
        return ParseConsumer::Blocked(format!(
            "the result binding {result_name} is written after the call, so the value parsed is not the \
            value the TypeScript target produced — no fact is attached"
        ));
    }
    match count {
        1 => ParseConsumer::Found(found.expect("count == 1 implies found is Some")),
        _ => ParseConsumer::Blocked(format!(
            "{result_name} is parsed {count} times after the call, and one stated result cannot stand for \
            more than one expression — no fact is attached"
        )),
    }
}

/// Whether a statement writes `name` directly, at any nesting depth of
/// its OWN statements (not inside a nested `def`/`class`, which has its
/// own scope) — an assignment/for/with-as/aug-assign target naming it.
/// Written fresh for this module's own sole-consumer scan (the Go
/// twin's `AssignedNamesDirect` is the model this mirrors, per the
/// mission's own note that no Rust twin exists yet).
pub(super) fn statement_writes_name(statement: &Stmt, name: &str) -> bool {
    match statement {
        Stmt::Assign(assign) => assign.targets.iter().any(|target| target_names(target, name)),
        Stmt::AnnAssign(assign) => target_names(assign.target.as_ref(), name),
        Stmt::AugAssign(assign) => target_names(assign.target.as_ref(), name),
        Stmt::For(for_stmt) => {
            target_names(for_stmt.target.as_ref(), name)
                || for_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || for_stmt.orelse.iter().any(|inner| statement_writes_name(inner, name))
        }
        Stmt::While(while_stmt) => {
            while_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || while_stmt.orelse.iter().any(|inner| statement_writes_name(inner, name))
        }
        Stmt::If(if_stmt) => {
            if_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || if_stmt
                    .elif_else_clauses
                    .iter()
                    .any(|clause| clause.body.iter().any(|inner| statement_writes_name(inner, name)))
        }
        Stmt::With(with_stmt) => {
            with_stmt.items.iter().any(|item| item.optional_vars.as_deref().is_some_and(|target| target_names(target, name)))
                || with_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
        }
        Stmt::Try(try_stmt) => {
            try_stmt.body.iter().any(|inner| statement_writes_name(inner, name))
                || try_stmt.handlers.iter().any(|handler| {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(handler) = handler;
                    handler.body.iter().any(|inner| statement_writes_name(inner, name))
                })
                || try_stmt.orelse.iter().any(|inner| statement_writes_name(inner, name))
                || try_stmt.finalbody.iter().any(|inner| statement_writes_name(inner, name))
        }
        // a nested def/class is its own scope, unreachable from this
        // scan, and every other statement form binds no plain name
        _ => false,
    }
}

/// Whether an assignment/for/with target expression names `name`
/// directly — a bare `Name`, or `name` among a `Tuple`/`List` target's
/// own elements.
pub(super) fn target_names(target: &Expr, name: &str) -> bool {
    match target {
        Expr::Name(identifier) => identifier.id.as_str() == name,
        Expr::Tuple(tuple) => tuple.elts.iter().any(|element| target_names(element, name)),
        Expr::List(list) => list.elts.iter().any(|element| target_names(element, name)),
        Expr::Starred(starred) => target_names(starred.value.as_ref(), name),
        _ => false,
    }
}

/// Counts every parse of `<name>` (per `result_read`) in a statement,
/// recording the first — never descending into a nested function, the
/// same boundary the Go twin's `foreignParseCallsIn` keeps.
pub(super) fn foreign_parse_calls_in(
    statement: &Stmt,
    name: &str,
    result_read: &ResultRead,
    found: &mut Option<TextRange>,
    count: &mut usize,
) {
    visit_statement_exprs(statement, &mut |expression| {
        if is_foreign_parse_of(expression, name, result_read) {
            if found.is_none() {
                *found = Some(expression.range());
            }
            *count += 1;
        }
    });
}

/// The intermediate captured-stdout READING's own node — `json.loads(...)`'s
/// sole ARGUMENT, one layer inside the call `sole_parse_consumer_from`
/// already found and proved unique (`ParseConsumer::Found`'s own
/// `count == 1` guarantee, re-run here rather than threaded through,
/// since this asks a strictly narrower question of the identical node).
/// `ResultRead::StdoutAttribute` answers the `<name>.stdout` attribute-
/// access node; `ResultRead::Bare` answers the bound name's own node —
/// the `Expr::Name` `json.loads(...)` actually reads, UNDER any
/// `.decode()` wrapper (the wrapper call itself is never the override
/// target: `evaluate_expression`'s node-override seam matches by exact
/// range, and only the inner `Expr::Name` is the node a plain `<name>`
/// read — or `<name>.decode()`'s own inner operand — evaluates through).
/// `None` for `ResultRead::FileRead`, which never reaches `json.loads`
/// at all (`is_foreign_parse_of`'s own `false` arm), and `None` when no
/// such node is found (the caller's own `ParseConsumer::Found` already
/// guarantees one exists on every live call path; `None` here is inert
/// rather than a caller-visible failure).
pub(super) fn foreign_parse_argument_range_of(statements: &[Stmt], name: &str, result_read: &ResultRead) -> Option<TextRange> {
    let mut found: Option<TextRange> = None;
    for statement in statements {
        visit_statement_exprs(statement, &mut |expression| {
            if found.is_some() {
                return;
            }
            if let Some(argument_range) = foreign_parse_argument_range(expression, name, result_read) {
                found = Some(argument_range);
            }
        });
        if found.is_some() {
            break;
        }
    }
    found
}

/// The argument-range half of `is_foreign_parse_of`'s own match — kept
/// as a SEPARATE reader rather than widening `is_foreign_parse_of`
/// itself to return the range, so that function's existing bool
/// contract (and every caller matching on it) is untouched.
pub(super) fn foreign_parse_argument_range(expression: &Expr, name: &str, result_read: &ResultRead) -> Option<TextRange> {
    let Expr::Call(call) = expression else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "loads" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    match result_read {
        ResultRead::StdoutAttribute => {
            let Expr::Attribute(result_attribute) = argument else {
                return None;
            };
            let Expr::Name(result_name) = result_attribute.value.as_ref() else {
                return None;
            };
            if result_name.id.as_str() != name || result_attribute.attr.as_str() != "stdout" {
                return None;
            }
            Some(argument.range())
        }
        ResultRead::Bare => {
            let unwrapped = unwrap_bytes_decode(argument);
            let Expr::Name(result_name) = unwrapped else {
                return None;
            };
            if result_name.id.as_str() != name {
                return None;
            }
            Some(unwrapped.range())
        }
        ResultRead::FileRead { .. } => None,
    }
}

/// Whether a node is exactly `json.loads(<name>.stdout)` (`result_read
/// == StdoutAttribute`) or `json.loads(<name>)`, OPTIONALLY
/// `.decode()`-wrapped (`result_read == Bare`) — the awaited asyncio
/// shape's `stdout_bytes` binding carries raw bytes
/// (`library/asyncio-subprocess.rst`: `Process.communicate`'s own return
/// is `bytes`, never `str`), so `json.loads(stdout_bytes)` reads bytes
/// directly (`json.loads` accepts `bytes | bytearray | str` per
/// `library/json.rst`) exactly as readily as a `.decode()`-unwrapped
/// text binding — both spellings name the identical captured value, so
/// neither is preferred over the other.
pub(super) fn is_foreign_parse_of(expression: &Expr, name: &str, result_read: &ResultRead) -> bool {
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return false;
    };
    if module_name.id.as_str() != "json" || attribute.attr.as_str() != "loads" {
        return false;
    }
    if !call.arguments.keywords.is_empty() {
        return false;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return false;
    };
    match result_read {
        ResultRead::StdoutAttribute => {
            let Expr::Attribute(result_attribute) = argument else {
                return false;
            };
            let Expr::Name(result_name) = result_attribute.value.as_ref() else {
                return false;
            };
            result_name.id.as_str() == name && result_attribute.attr.as_str() == "stdout"
        }
        ResultRead::Bare => {
            let Expr::Name(result_name) = unwrap_bytes_decode(argument) else {
                return false;
            };
            result_name.id.as_str() == name
        }
        // `os.system`'s own file-legs shape never reaches this function —
        // `finish_recognized_edge` dispatches `FileRead` to `os_system_
        // return_read_of` instead, which looks for `json.load` (singular),
        // never `json.loads`. Unreachable in practice; `false` rather than
        // a panic, since a plain bool answer costs nothing and needs no
        // caller-side unwrapping.
        ResultRead::FileRead { .. } => false,
    }
}

/// Strips a trailing `.decode()` call off an expression — `<expr>.decode(
/// )` with no arguments and no keywords answers `<expr>` itself; every
/// other shape (a bare expression with no `.decode()` at all) answers the
/// expression unchanged. The return-leg counterpart of
/// `unwrap_bytes_encode`'s outbound unwrap: reads a NAME through the
/// wrapper (`is_foreign_parse_of`'s own use, matching the unwrapped
/// expression against `Expr::Name`) rather than an arbitrary expression,
/// so it answers `&Expr` directly rather than a reference the caller
/// must re-match.
pub(super) fn unwrap_bytes_decode(expression: &Expr) -> &Expr {
    let Expr::Call(call) = expression else {
        return expression;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return expression;
    };
    if attribute.attr.as_str() != "decode" || !call.arguments.args.is_empty() || !call.arguments.keywords.is_empty() {
        return expression;
    }
    attribute.value.as_ref()
}

/// Walks every expression reachable from a statement without crossing
/// into a nested function/class body, calling `visit` on each. A small,
/// purpose-built walk (rather than reusing `check.rs`'s own
/// `collect_walrus_names` recursion, which is expression-shaped, not
/// statement-shaped) covering exactly the statement forms that can
/// appear between the call and its return in an ordinary function body.
pub(super) fn visit_statement_exprs(statement: &Stmt, visit: &mut dyn FnMut(&Expr)) {
    match statement {
        Stmt::Expr(expr_stmt) => visit_expr_exprs(expr_stmt.value.as_ref(), visit),
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                visit_expr_exprs(target, visit);
            }
            visit_expr_exprs(assign.value.as_ref(), visit);
        }
        Stmt::AnnAssign(assign) => {
            visit_expr_exprs(assign.target.as_ref(), visit);
            if let Some(value) = assign.value.as_deref() {
                visit_expr_exprs(value, visit);
            }
        }
        Stmt::AugAssign(assign) => {
            visit_expr_exprs(assign.target.as_ref(), visit);
            visit_expr_exprs(assign.value.as_ref(), visit);
        }
        Stmt::Return(ret) => {
            if let Some(value) = ret.value.as_deref() {
                visit_expr_exprs(value, visit);
            }
        }
        Stmt::If(if_stmt) => {
            visit_expr_exprs(if_stmt.test.as_ref(), visit);
            for inner in &if_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for clause in &if_stmt.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    visit_expr_exprs(test, visit);
                }
                for inner in &clause.body {
                    visit_statement_exprs(inner, visit);
                }
            }
        }
        Stmt::For(for_stmt) => {
            visit_expr_exprs(for_stmt.iter.as_ref(), visit);
            for inner in &for_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for inner in &for_stmt.orelse {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::While(while_stmt) => {
            visit_expr_exprs(while_stmt.test.as_ref(), visit);
            for inner in &while_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for inner in &while_stmt.orelse {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                visit_expr_exprs(&item.context_expr, visit);
            }
            for inner in &with_stmt.body {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::Try(try_stmt) => {
            for inner in &try_stmt.body {
                visit_statement_exprs(inner, visit);
            }
            for handler in &try_stmt.handlers {
                let ruff_python_ast::ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    visit_statement_exprs(inner, visit);
                }
            }
            for inner in &try_stmt.orelse {
                visit_statement_exprs(inner, visit);
            }
            for inner in &try_stmt.finalbody {
                visit_statement_exprs(inner, visit);
            }
        }
        Stmt::Assert(assert_stmt) => {
            visit_expr_exprs(assert_stmt.test.as_ref(), visit);
            if let Some(message) = assert_stmt.msg.as_deref() {
                visit_expr_exprs(message, visit);
            }
        }
        Stmt::Raise(raise_stmt) => {
            if let Some(exc) = raise_stmt.exc.as_deref() {
                visit_expr_exprs(exc, visit);
            }
            if let Some(cause) = raise_stmt.cause.as_deref() {
                visit_expr_exprs(cause, visit);
            }
        }
        // a nested def/class is its own scope; every other statement
        // form (pass, break, continue, import, global, nonlocal, match,
        // delete, type-alias) carries no expression this scan reaches
        _ => {}
    }
}

/// Visits every subexpression of `expression`, never descending into a
/// lambda body — a lambda's body is its own scope, the same rule the
/// statement-level walk keeps for a nested def.
pub(super) fn visit_expr_exprs(expression: &Expr, visit: &mut dyn FnMut(&Expr)) {
    visit(expression);
    match expression {
        Expr::Lambda(_) => {}
        Expr::BoolOp(op) => op.values.iter().for_each(|value| visit_expr_exprs(value, visit)),
        Expr::BinOp(op) => {
            visit_expr_exprs(op.left.as_ref(), visit);
            visit_expr_exprs(op.right.as_ref(), visit);
        }
        Expr::UnaryOp(op) => visit_expr_exprs(op.operand.as_ref(), visit),
        Expr::If(ternary) => {
            visit_expr_exprs(ternary.test.as_ref(), visit);
            visit_expr_exprs(ternary.body.as_ref(), visit);
            visit_expr_exprs(ternary.orelse.as_ref(), visit);
        }
        Expr::Tuple(tuple) => tuple.elts.iter().for_each(|element| visit_expr_exprs(element, visit)),
        Expr::List(list) => list.elts.iter().for_each(|element| visit_expr_exprs(element, visit)),
        Expr::Set(set) => set.elts.iter().for_each(|element| visit_expr_exprs(element, visit)),
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    visit_expr_exprs(key, visit);
                }
                visit_expr_exprs(&item.value, visit);
            }
        }
        Expr::Call(call) => {
            visit_expr_exprs(call.func.as_ref(), visit);
            for argument in &call.arguments.args {
                visit_expr_exprs(argument, visit);
            }
            for keyword in &call.arguments.keywords {
                visit_expr_exprs(&keyword.value, visit);
            }
        }
        Expr::Compare(compare) => {
            visit_expr_exprs(compare.left.as_ref(), visit);
            compare.comparators.iter().for_each(|comparator| visit_expr_exprs(comparator, visit));
        }
        Expr::Attribute(attribute) => visit_expr_exprs(attribute.value.as_ref(), visit),
        Expr::Subscript(subscript) => {
            visit_expr_exprs(subscript.value.as_ref(), visit);
            visit_expr_exprs(subscript.slice.as_ref(), visit);
        }
        Expr::Starred(starred) => visit_expr_exprs(starred.value.as_ref(), visit),
        Expr::Named(named) => visit_expr_exprs(named.value.as_ref(), visit),
        Expr::Await(inner) => visit_expr_exprs(inner.value.as_ref(), visit),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                visit_expr_exprs(value, visit);
            }
        }
        Expr::YieldFrom(inner) => visit_expr_exprs(inner.value.as_ref(), visit),
        _ => {}
    }
}

