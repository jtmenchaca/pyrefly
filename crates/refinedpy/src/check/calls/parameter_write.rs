//! STALE-ARGUMENT SOUNDNESS: whether a same-module def's own body may
//! write THROUGH one of its parameters — a subscript/attribute store or
//! delete whose base names the parameter, or a not-proven-read-only
//! method call on it. A caller handing that parameter's argument binding
//! into such a call must not keep facts recorded on the pre-call value
//! (`check::calls::effects::apply_call_effects`, this file's own
//! consumer, drops them there).

use ruff_python_ast::{Expr, Stmt};

/// Method names this crate already models as READ-ONLY at the
/// evaluation layer (`expressions::call::attribute_call`'s own `"get"`/
/// `dict_view_method_result`/`set_method_result` rows) — the only names
/// this scan excludes from "possibly writes." Every other method name,
/// including one this crate does not model at all, is treated as
/// possibly-mutating (see this file's own doc).
const KNOWN_READ_ONLY_METHOD_NAMES: [&str; 7] = ["get", "keys", "values", "items", "copy", "count", "index"];

/// True when `body` may, on SOME execution path, write through `name` —
/// a `Subscript`/`Attribute` assignment or `del` target whose base is
/// `Expr::Name(name)`, or a `name.method(...)` call whose method is
/// either a KNOWN mutating name or not proven read-only (this file's own
/// two lists). Recurses into every nested statement form (`if`/`for`/
/// `while`/`with`/`try`/`match`) — unlike `collect_bound_names`'s
/// restricted-interpreter reach, this is a soundness scan and must see a
/// write buried in any control-flow shape the body contains, not just
/// the forms the restricted interpreter itself replays.
///
/// `name` shadowed by a nested `def`/`lambda` parameter of the same name,
/// or reassigned to a different object before the write (`d = {}`,
/// `d.pop(...)`), is not distinguished from an unshadowed write — the
/// scan asks only "does the SYNTAX `name.<sub>`/`name.<method>()` occur,"
/// which over-approximates in the caller's favor (treating a callee as
/// possibly-mutating when it is not costs a dropped guard fact, never an
/// unsound one).
pub(in crate::check) fn body_may_write_through_parameter(body: &[Stmt], name: &str) -> bool {
    body.iter().any(|stmt| statement_may_write_through_parameter(stmt, name))
}

fn statement_may_write_through_parameter(stmt: &Stmt, name: &str) -> bool {
    match stmt {
        Stmt::Assign(assign) => {
            assign.targets.iter().any(|target| target_writes_through(target, name))
                || expr_may_write_through_parameter(&assign.value, name)
        }
        Stmt::AugAssign(assign) => {
            target_writes_through(&assign.target, name) || expr_may_write_through_parameter(&assign.value, name)
        }
        Stmt::AnnAssign(assign) => {
            target_writes_through(&assign.target, name)
                || assign.value.as_deref().is_some_and(|value| expr_may_write_through_parameter(value, name))
        }
        Stmt::Delete(delete) => delete.targets.iter().any(|target| target_writes_through(target, name)),
        Stmt::Expr(expr_stmt) => expr_may_write_through_parameter(&expr_stmt.value, name),
        Stmt::Return(ret) => ret.value.as_deref().is_some_and(|value| expr_may_write_through_parameter(value, name)),
        Stmt::If(if_stmt) => {
            body_may_write_through_parameter(&if_stmt.body, name)
                || if_stmt.elif_else_clauses.iter().any(|clause| body_may_write_through_parameter(&clause.body, name))
        }
        Stmt::For(for_stmt) => {
            body_may_write_through_parameter(&for_stmt.body, name)
                || body_may_write_through_parameter(&for_stmt.orelse, name)
        }
        Stmt::While(while_stmt) => {
            body_may_write_through_parameter(&while_stmt.body, name)
                || body_may_write_through_parameter(&while_stmt.orelse, name)
        }
        Stmt::With(with_stmt) => body_may_write_through_parameter(&with_stmt.body, name),
        Stmt::Try(try_stmt) => {
            body_may_write_through_parameter(&try_stmt.body, name)
                || try_stmt.handlers.iter().any(|handler| {
                    let ruff_python_ast::ExceptHandler::ExceptHandler(handler) = handler;
                    body_may_write_through_parameter(&handler.body, name)
                })
                || body_may_write_through_parameter(&try_stmt.orelse, name)
                || body_may_write_through_parameter(&try_stmt.finalbody, name)
        }
        Stmt::Match(match_stmt) => match_stmt.cases.iter().any(|case| body_may_write_through_parameter(&case.body, name)),
        _ => false,
    }
}

/// Whether `target` itself is a write reaching through `name` — a
/// `Subscript`/`Attribute` whose base is `Expr::Name(name)`. A bare
/// `Expr::Name` target REBINDS the parameter to a new object rather than
/// writing through the old one (the guard's recorded entries describe
/// the OLD object; a rebind replaces it wholesale, which is exactly what
/// `Environment::bind`'s own doc already treats as a plain, harmless
/// rebind, not a stale-entry hazard) so it is not counted here. A
/// tuple/list unpack target recurses over its own elements for the same
/// reason `collect_unpack_target_names` does.
fn target_writes_through(target: &Expr, name: &str) -> bool {
    match target {
        Expr::Subscript(subscript) => matches!(subscript.value.as_ref(), Expr::Name(base) if base.id.as_str() == name),
        Expr::Attribute(attribute) => matches!(attribute.value.as_ref(), Expr::Name(base) if base.id.as_str() == name),
        Expr::Tuple(tuple) => tuple.elts.iter().any(|element| target_writes_through(element, name)),
        Expr::List(list) => list.elts.iter().any(|element| target_writes_through(element, name)),
        _ => false,
    }
}

/// Whether `expr` contains a `name.method(...)` call anywhere within it
/// (a call argument, an operand, a nested call) whose method is not
/// PROVEN read-only (`KNOWN_READ_ONLY_METHOD_NAMES`) — this file's own
/// doc states why an unrecognized method counts as possibly-mutating
/// rather than assumed safe.
fn expr_may_write_through_parameter(expr: &Expr, name: &str) -> bool {
    match expr {
        Expr::Call(call) => {
            let callee_is_write_method_on_name = matches!(
                call.func.as_ref(),
                Expr::Attribute(attribute)
                    if matches!(attribute.value.as_ref(), Expr::Name(base) if base.id.as_str() == name)
                        && !KNOWN_READ_ONLY_METHOD_NAMES.contains(&attribute.attr.as_str())
            );
            callee_is_write_method_on_name
                || expr_may_write_through_parameter(&call.func, name)
                || call.arguments.args.iter().any(|arg| expr_may_write_through_parameter(arg, name))
                || call.arguments.keywords.iter().any(|kw| expr_may_write_through_parameter(&kw.value, name))
        }
        Expr::BoolOp(op) => op.values.iter().any(|value| expr_may_write_through_parameter(value, name)),
        Expr::BinOp(op) => {
            expr_may_write_through_parameter(&op.left, name) || expr_may_write_through_parameter(&op.right, name)
        }
        Expr::UnaryOp(op) => expr_may_write_through_parameter(&op.operand, name),
        Expr::If(if_expr) => {
            expr_may_write_through_parameter(&if_expr.test, name)
                || expr_may_write_through_parameter(&if_expr.body, name)
                || expr_may_write_through_parameter(&if_expr.orelse, name)
        }
        Expr::Compare(compare) => {
            expr_may_write_through_parameter(&compare.left, name)
                || compare.comparators.iter().any(|value| expr_may_write_through_parameter(value, name))
        }
        _ => false,
    }
}
