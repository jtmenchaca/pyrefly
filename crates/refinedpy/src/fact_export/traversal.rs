//! Every expression inside a statement or body, visited parent-before-
//! child — the traversal the stdout-purity scan walks.

use ruff_python_ast::ExceptHandler;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;

/// Every expression inside one statement, including every nested
/// statement's own — the traversal the stdout scan walks. Visits each
/// expression node once, parents before children.
pub(super) fn walk_statement_expressions(stmt: &Stmt, visit: &mut dyn FnMut(&Expr)) {
    match stmt {
        Stmt::Expr(node) => walk_expression(node.value.as_ref(), visit),
        Stmt::Return(node) => {
            if let Some(value) = node.value.as_deref() {
                walk_expression(value, visit);
            }
        }
        Stmt::Assign(node) => {
            for target in &node.targets {
                walk_expression(target, visit);
            }
            walk_expression(node.value.as_ref(), visit);
        }
        Stmt::AnnAssign(node) => {
            if let Some(value) = node.value.as_deref() {
                walk_expression(value, visit);
            }
        }
        Stmt::AugAssign(node) => walk_expression(node.value.as_ref(), visit),
        Stmt::If(node) => {
            walk_expression(node.test.as_ref(), visit);
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
            for clause in &node.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    walk_expression(test, visit);
                }
                for inner in &clause.body {
                    walk_statement_expressions(inner, visit);
                }
            }
        }
        Stmt::For(node) => {
            walk_expression(node.iter.as_ref(), visit);
            for inner in node.body.iter().chain(node.orelse.iter()) {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::While(node) => {
            walk_expression(node.test.as_ref(), visit);
            for inner in node.body.iter().chain(node.orelse.iter()) {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::With(node) => {
            for item in &node.items {
                walk_expression(&item.context_expr, visit);
            }
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::Try(node) => {
            for inner in node.body.iter().chain(node.orelse.iter()).chain(node.finalbody.iter()) {
                walk_statement_expressions(inner, visit);
            }
            for handler in &node.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    walk_statement_expressions(inner, visit);
                }
            }
        }
        Stmt::Match(node) => {
            walk_expression(node.subject.as_ref(), visit);
            for case in &node.cases {
                for inner in &case.body {
                    walk_statement_expressions(inner, visit);
                }
            }
        }
        Stmt::FunctionDef(node) => {
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::ClassDef(node) => {
            for inner in &node.body {
                walk_statement_expressions(inner, visit);
            }
        }
        Stmt::Raise(node) => {
            if let Some(exception) = node.exc.as_deref() {
                walk_expression(exception, visit);
            }
            if let Some(cause) = node.cause.as_deref() {
                walk_expression(cause, visit);
            }
        }
        Stmt::Assert(node) => {
            walk_expression(node.test.as_ref(), visit);
            if let Some(message) = node.msg.as_deref() {
                walk_expression(message, visit);
            }
        }
        Stmt::Delete(node) => {
            for target in &node.targets {
                walk_expression(target, visit);
            }
        }
        // Every remaining form (`pass`, `break`, `continue`, `global`,
        // `nonlocal`, `import`, `from ... import`, a type alias) holds no
        // expression a stdout write could hide in.
        _ => {}
    }
}

/// One expression and every expression nested inside it, parent first.
pub(super) fn walk_expression(expr: &Expr, visit: &mut dyn FnMut(&Expr)) {
    visit(expr);
    match expr {
        Expr::Call(node) => {
            walk_expression(node.func.as_ref(), visit);
            for argument in node.arguments.args.iter() {
                walk_expression(argument, visit);
            }
            for keyword in node.arguments.keywords.iter() {
                walk_expression(&keyword.value, visit);
            }
        }
        Expr::Attribute(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Subscript(node) => {
            walk_expression(node.value.as_ref(), visit);
            walk_expression(node.slice.as_ref(), visit);
        }
        Expr::BinOp(node) => {
            walk_expression(node.left.as_ref(), visit);
            walk_expression(node.right.as_ref(), visit);
        }
        Expr::UnaryOp(node) => walk_expression(node.operand.as_ref(), visit),
        Expr::BoolOp(node) => {
            for value in &node.values {
                walk_expression(value, visit);
            }
        }
        Expr::Compare(node) => {
            walk_expression(node.left.as_ref(), visit);
            for comparator in node.comparators.iter() {
                walk_expression(comparator, visit);
            }
        }
        Expr::If(node) => {
            walk_expression(node.test.as_ref(), visit);
            walk_expression(node.body.as_ref(), visit);
            walk_expression(node.orelse.as_ref(), visit);
        }
        Expr::Tuple(node) => {
            for element in &node.elts {
                walk_expression(element, visit);
            }
        }
        Expr::List(node) => {
            for element in &node.elts {
                walk_expression(element, visit);
            }
        }
        Expr::Set(node) => {
            for element in &node.elts {
                walk_expression(element, visit);
            }
        }
        Expr::Dict(node) => {
            for item in &node.items {
                if let Some(key) = item.key.as_ref() {
                    walk_expression(key, visit);
                }
                walk_expression(&item.value, visit);
            }
        }
        Expr::ListComp(node) => {
            walk_expression(node.elt.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::SetComp(node) => {
            walk_expression(node.elt.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::Generator(node) => {
            walk_expression(node.elt.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::DictComp(node) => {
            if let Some(key) = node.key.as_deref() {
                walk_expression(key, visit);
            }
            walk_expression(node.value.as_ref(), visit);
            for generator in &node.generators {
                walk_expression(&generator.iter, visit);
                for condition in &generator.ifs {
                    walk_expression(condition, visit);
                }
            }
        }
        Expr::Starred(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Await(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Yield(node) => {
            if let Some(value) = node.value.as_deref() {
                walk_expression(value, visit);
            }
        }
        Expr::YieldFrom(node) => walk_expression(node.value.as_ref(), visit),
        Expr::Named(node) => {
            walk_expression(node.target.as_ref(), visit);
            walk_expression(node.value.as_ref(), visit);
        }
        Expr::Lambda(node) => walk_expression(node.body.as_ref(), visit),
        Expr::Slice(node) => {
            for part in [node.lower.as_deref(), node.upper.as_deref(), node.step.as_deref()]
                .into_iter()
                .flatten()
            {
                walk_expression(part, visit);
            }
        }
        Expr::FString(node) => {
            for element in node.value.elements().filter_map(|element| element.as_interpolation()) {
                walk_expression(element.expression.as_ref(), visit);
            }
        }
        // Every remaining form is a leaf (a name, a literal, an
        // ellipsis) with nothing nested inside it.
        _ => {}
    }
}
