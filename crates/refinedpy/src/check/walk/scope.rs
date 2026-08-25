//! The scope prepass: every name a body binds anywhere within its own
//! statements (not inside a nested `def`/`class`, which has its own
//! scope), used to seed each body's fresh `Environment` with which
//! names go dark on a module-level alias — Python's whole-body scoping
//! rule. Also owns the walrus-target readers (scope collection AND the
//! actual bind), since both walk the identical expression shape.

use std::collections::HashMap;
use std::collections::HashSet;

use ruff_python_ast::{ExceptHandler, Expr, Parameters, Stmt, WithItem};
use ruff_text_size::Ranged;

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::typereading::DeclaredRefinement;

use crate::check::{judge_and_bind, Finding, WalkContext};

/// A plain prose name for a statement kind, for the blocker sentence —
/// e.g. "a while statement is not yet walked". Never a category label:
/// each name is spoken in place, in the sentence naming this one body's
/// first blocker.
pub(in crate::check) fn statement_kind_name(stmt: &Stmt) -> &'static str {
    match stmt {
        Stmt::TypeAlias(_) => "a nested type alias statement",
        Stmt::For(_) => "a for statement",
        Stmt::While(_) => "a while statement",
        Stmt::With(_) => "a with statement",
        Stmt::Match(_) => "a match statement",
        Stmt::Try(_) => "a try statement",
        Stmt::Import(_) => "an import statement",
        Stmt::ImportFrom(_) => "an import-from statement",
        Stmt::Break(_) => "a break statement",
        Stmt::Continue(_) => "a continue statement",
        Stmt::IpyEscapeCommand(_) => "an IPython escape command",
        // Handled elsewhere in walk_statement's match — never reaches here.
        Stmt::AnnAssign(_)
        | Stmt::Assign(_)
        | Stmt::AugAssign(_)
        | Stmt::Expr(_)
        | Stmt::Pass(_)
        | Stmt::Return(_)
        | Stmt::FunctionDef(_)
        | Stmt::ClassDef(_)
        | Stmt::Delete(_)
        | Stmt::If(_)
        | Stmt::Assert(_)
        | Stmt::Raise(_)
        | Stmt::Global(_)
        | Stmt::Nonlocal(_) => "a statement",
    }
}

/// Every name this body binds anywhere, at any nesting depth of its
/// OWN statements (not inside a nested `def`/`class` body, which has
/// its own scope) — assignment/for/with-as/except targets, walrus
/// targets in any expression the body evaluates, parameters, and
/// import aliases. A name declared `global`/`nonlocal` is excluded:
/// Python's own rule is that such a name is never local to this body,
/// so a module-level alias sharing its spelling stays visible.
pub(in crate::check) fn locally_bound_names(body: &[Stmt]) -> HashSet<String> {
    let mut bound = HashSet::new();
    let mut excluded = HashSet::new();
    collect_bound_names(body, &mut bound, &mut excluded);
    for name in &excluded {
        bound.remove(name);
    }
    bound
}

pub(in crate::check) fn collect_bound_names(body: &[Stmt], bound: &mut HashSet<String>, excluded: &mut HashSet<String>) {
    for stmt in body {
        collect_bound_names_stmt(stmt, bound, excluded);
    }
}

pub(in crate::check) fn collect_bound_names_stmt(stmt: &Stmt, bound: &mut HashSet<String>, excluded: &mut HashSet<String>) {
    match stmt {
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                collect_target_names(target, bound);
            }
            collect_walrus_names(assign.value.as_ref(), bound);
        }
        Stmt::AnnAssign(assign) => {
            collect_target_names(assign.target.as_ref(), bound);
            if let Some(value) = assign.value.as_deref() {
                collect_walrus_names(value, bound);
            }
        }
        Stmt::AugAssign(assign) => {
            collect_target_names(assign.target.as_ref(), bound);
            collect_walrus_names(assign.value.as_ref(), bound);
        }
        Stmt::For(for_stmt) => {
            collect_target_names(for_stmt.target.as_ref(), bound);
            collect_walrus_names(for_stmt.iter.as_ref(), bound);
            collect_bound_names(&for_stmt.body, bound, excluded);
            collect_bound_names(&for_stmt.orelse, bound, excluded);
        }
        Stmt::While(while_stmt) => {
            collect_walrus_names(while_stmt.test.as_ref(), bound);
            collect_bound_names(&while_stmt.body, bound, excluded);
            collect_bound_names(&while_stmt.orelse, bound, excluded);
        }
        Stmt::If(if_stmt) => {
            collect_walrus_names(if_stmt.test.as_ref(), bound);
            collect_bound_names(&if_stmt.body, bound, excluded);
            for clause in &if_stmt.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    collect_walrus_names(test, bound);
                }
                collect_bound_names(&clause.body, bound, excluded);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                collect_with_item_names(item, bound);
            }
            collect_bound_names(&with_stmt.body, bound, excluded);
        }
        Stmt::Try(try_stmt) => {
            collect_bound_names(&try_stmt.body, bound, excluded);
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                if let Some(name) = handler.name.as_ref() {
                    bound.insert(name.id.as_str().to_owned());
                }
                collect_bound_names(&handler.body, bound, excluded);
            }
            collect_bound_names(&try_stmt.orelse, bound, excluded);
            collect_bound_names(&try_stmt.finalbody, bound, excluded);
        }
        Stmt::FunctionDef(def) => {
            bound.insert(def.name.id.as_str().to_owned());
            // the def's OWN body is a separate scope — its parameters
            // and locals do not leak into this body's bound set
        }
        Stmt::ClassDef(def) => {
            bound.insert(def.name.id.as_str().to_owned());
        }
        // An import DECLARES a name; it never REBINDS one — the
        // rebinding gate this collector feeds
        // (`Environment::alias_is_visible`) exists to catch a body's OWN
        // assignment/for/with/except target shadowing a module-level
        // alias, never the import statement that IS how the alias
        // itself becomes visible in the first place (C1.scope.py's own
        // `from support.py.refined import Age` at module top level —
        // the alias's own establishing import — must never read as
        // shadowing `Age` at its own declaration site, the false
        // positive this arm once produced). A local import shadowing an
        // OUTER alias by the same spelling inside a nested body is a
        // real Python fact this collector does not separately state
        // today; no fixture in the corpus exercises that shape, and
        // leaving it unstated is conservative (an unshadowed alias name
        // still resolves the caller's normal way), never a false report
        // the other direction.
        Stmt::Import(_) | Stmt::ImportFrom(_) => {}
        Stmt::Global(global) => {
            for name in &global.names {
                excluded.insert(name.id.as_str().to_owned());
            }
        }
        Stmt::Nonlocal(nonlocal) => {
            for name in &nonlocal.names {
                excluded.insert(name.id.as_str().to_owned());
            }
        }
        Stmt::Expr(expr_stmt) => collect_walrus_names(expr_stmt.value.as_ref(), bound),
        Stmt::Return(ret) => {
            if let Some(value) = ret.value.as_deref() {
                collect_walrus_names(value, bound);
            }
        }
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                collect_walrus_names(target, bound);
            }
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                collect_walrus_names(exc, bound);
            }
            if let Some(cause) = raise.cause.as_deref() {
                collect_walrus_names(cause, bound);
            }
        }
        Stmt::Assert(assert) => {
            collect_walrus_names(assert.test.as_ref(), bound);
            if let Some(msg) = assert.msg.as_deref() {
                collect_walrus_names(msg, bound);
            }
        }
        Stmt::Match(match_stmt) => {
            collect_walrus_names(match_stmt.subject.as_ref(), bound);
            for case in &match_stmt.cases {
                collect_bound_names(&case.body, bound, excluded);
            }
        }
        Stmt::TypeAlias(_) | Stmt::Pass(_) | Stmt::Break(_) | Stmt::Continue(_)
        | Stmt::IpyEscapeCommand(_) => {}
    }
}

/// A `for`/`with`-as/assignment target's bound names, including nested
/// tuple/list/starred forms.
pub(in crate::check) fn collect_target_names(target: &Expr, bound: &mut HashSet<String>) {
    match target {
        Expr::Name(name) => {
            bound.insert(name.id.as_str().to_owned());
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_target_names(element, bound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_target_names(element, bound);
            }
        }
        Expr::Starred(starred) => collect_target_names(starred.value.as_ref(), bound),
        _ => {}
    }
}

pub(in crate::check) fn collect_with_item_names(item: &WithItem, bound: &mut HashSet<String>) {
    collect_walrus_names(&item.context_expr, bound);
    if let Some(vars) = item.optional_vars.as_deref() {
        collect_target_names(vars, bound);
    }
}

/// Walrus (`:=`) targets anywhere inside an expression the body
/// evaluates — a walrus binds its target into the ENCLOSING scope,
/// wherever it sits (a comprehension, a condition, a call argument).
pub(in crate::check) fn collect_walrus_names(expr: &Expr, bound: &mut HashSet<String>) {
    match expr {
        Expr::Named(named) => {
            collect_target_names(named.target.as_ref(), bound);
            collect_walrus_names(named.value.as_ref(), bound);
        }
        Expr::BoolOp(op) => {
            for value in &op.values {
                collect_walrus_names(value, bound);
            }
        }
        Expr::BinOp(op) => {
            collect_walrus_names(op.left.as_ref(), bound);
            collect_walrus_names(op.right.as_ref(), bound);
        }
        Expr::UnaryOp(op) => collect_walrus_names(op.operand.as_ref(), bound),
        // the lambda's OWN body is a separate scope; a walrus inside it
        // does not bind here
        Expr::Lambda(_) => {}
        Expr::If(if_expr) => {
            collect_walrus_names(if_expr.test.as_ref(), bound);
            collect_walrus_names(if_expr.body.as_ref(), bound);
            collect_walrus_names(if_expr.orelse.as_ref(), bound);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_walrus_names(element, bound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_walrus_names(element, bound);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                collect_walrus_names(element, bound);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    collect_walrus_names(key, bound);
                }
                collect_walrus_names(&item.value, bound);
            }
        }
        Expr::Call(call) => {
            collect_walrus_names(call.func.as_ref(), bound);
            for arg in &call.arguments.args {
                collect_walrus_names(arg, bound);
            }
            for keyword in &call.arguments.keywords {
                collect_walrus_names(&keyword.value, bound);
            }
        }
        Expr::Compare(compare) => {
            collect_walrus_names(compare.left.as_ref(), bound);
            for comparator in &compare.comparators {
                collect_walrus_names(comparator, bound);
            }
        }
        Expr::Attribute(attribute) => collect_walrus_names(attribute.value.as_ref(), bound),
        Expr::Subscript(subscript) => {
            collect_walrus_names(subscript.value.as_ref(), bound);
            collect_walrus_names(subscript.slice.as_ref(), bound);
        }
        Expr::Starred(starred) => collect_walrus_names(starred.value.as_ref(), bound),
        Expr::Slice(slice) => {
            if let Some(lower) = slice.lower.as_deref() {
                collect_walrus_names(lower, bound);
            }
            if let Some(upper) = slice.upper.as_deref() {
                collect_walrus_names(upper, bound);
            }
            if let Some(step) = slice.step.as_deref() {
                collect_walrus_names(step, bound);
            }
        }
        Expr::FString(fstring) => {
            // `.elements()` already flattens every part (single or
            // implicitly-concatenated) down to each part's own
            // elements, literal parts skipped.
            for element in fstring.value.elements() {
                if let Some(interpolation) = element.as_interpolation() {
                    collect_walrus_names(interpolation.expression.as_ref(), bound);
                }
            }
        }
        Expr::Await(inner) => collect_walrus_names(inner.value.as_ref(), bound),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                collect_walrus_names(value, bound);
            }
        }
        Expr::YieldFrom(inner) => collect_walrus_names(inner.value.as_ref(), bound),
        // Comprehensions (ListComp/SetComp/DictComp/Generator) introduce
        // their own scope for their loop variables — a walrus INSIDE the
        // comprehension's element/condition still targets the ENCLOSING
        // scope per PEP 572, but that expression-walking depth is not
        // built in this wave; left unwalked rather than guessed.
        _ => {}
    }
}

/// WALRUS BINDING: every `:=` reachable inside an expression the walk
/// evaluates binds its bare-Name target into the ENCLOSING environment
/// — the same traversal shape `collect_walrus_names` already walks (for
/// the SCOPE prepass, which only needs the target's spelling), reused
/// here to also BIND the target to its evaluated inner value (what the
/// scope prepass does not need, since it runs before any environment
/// exists to bind into). `evaluate_expression` already reads
/// `Expr::Named` correctly wherever it is nested (it returns the inner
/// value, `expressions.rs`'s own dispatch), so evaluating each found
/// walrus's OWN inner expression here — a second, cheap evaluation of a
/// pure expression tree with no side effects to duplicate — is the
/// direct way to get the exact same value the walrus's surrounding
/// expression already computed from it.
///
/// `aug_assign_refinements` judges a declared name's walrus value
/// through `judge_and_bind` exactly like a plain `x = value` target
/// (`walrus_in_condition`'s own `over := 200` under a later `Age`-typed
/// read is the corpus row this serves); an undeclared target binds
/// directly. A non-Name walrus target is not legal Python grammar
/// (`named_expression: assignment_expression | expression`, PEP 572 —
/// the target is always an identifier) and never reaches this function
/// at all, so there is no "else" case to handle.
pub(in crate::check) fn bind_walrus_targets(
    expr: &Expr,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match expr {
        Expr::Named(named) => {
            if let Expr::Name(target_name) = named.target.as_ref() {
                let inner_value = evaluate_expression(named.value.as_ref(), environment, context.kernel);
                match aug_assign_refinements.get(target_name.id.as_str()) {
                    Some(declared) => {
                        let declared = declared.clone();
                        judge_and_bind(
                            target_name.id.as_str(),
                            inner_value,
                            &declared,
                            named.value.range(),
                            context,
                            environment,
                            out,
                        );
                    }
                    None => environment.bind(target_name.id.as_str(), inner_value),
                }
            }
            bind_walrus_targets(named.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::BoolOp(op) => {
            for value in &op.values {
                bind_walrus_targets(value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::BinOp(op) => {
            bind_walrus_targets(op.left.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(op.right.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::UnaryOp(op) => bind_walrus_targets(op.operand.as_ref(), context, aug_assign_refinements, environment, out),
        // the lambda's OWN body is a separate scope; a walrus inside it
        // does not bind here — mirrors collect_walrus_names exactly.
        Expr::Lambda(_) => {}
        Expr::If(if_expr) => {
            bind_walrus_targets(if_expr.test.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(if_expr.body.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(if_expr.orelse.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                bind_walrus_targets(element, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                bind_walrus_targets(element, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                bind_walrus_targets(element, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    bind_walrus_targets(key, context, aug_assign_refinements, environment, out);
                }
                bind_walrus_targets(&item.value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Call(call) => {
            bind_walrus_targets(call.func.as_ref(), context, aug_assign_refinements, environment, out);
            for arg in &call.arguments.args {
                bind_walrus_targets(arg, context, aug_assign_refinements, environment, out);
            }
            for keyword in &call.arguments.keywords {
                bind_walrus_targets(&keyword.value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Compare(compare) => {
            bind_walrus_targets(compare.left.as_ref(), context, aug_assign_refinements, environment, out);
            for comparator in &compare.comparators {
                bind_walrus_targets(comparator, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::Attribute(attribute) => {
            bind_walrus_targets(attribute.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Subscript(subscript) => {
            bind_walrus_targets(subscript.value.as_ref(), context, aug_assign_refinements, environment, out);
            bind_walrus_targets(subscript.slice.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Starred(starred) => {
            bind_walrus_targets(starred.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        Expr::Slice(slice) => {
            if let Some(lower) = slice.lower.as_deref() {
                bind_walrus_targets(lower, context, aug_assign_refinements, environment, out);
            }
            if let Some(upper) = slice.upper.as_deref() {
                bind_walrus_targets(upper, context, aug_assign_refinements, environment, out);
            }
            if let Some(step) = slice.step.as_deref() {
                bind_walrus_targets(step, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::FString(fstring) => {
            for element in fstring.value.elements() {
                if let Some(interpolation) = element.as_interpolation() {
                    bind_walrus_targets(interpolation.expression.as_ref(), context, aug_assign_refinements, environment, out);
                }
            }
        }
        Expr::Await(inner) => bind_walrus_targets(inner.value.as_ref(), context, aug_assign_refinements, environment, out),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                bind_walrus_targets(value, context, aug_assign_refinements, environment, out);
            }
        }
        Expr::YieldFrom(inner) => {
            bind_walrus_targets(inner.value.as_ref(), context, aug_assign_refinements, environment, out);
        }
        // Comprehensions introduce their own scope for their loop
        // variables; a walrus inside one still targets the enclosing
        // scope per PEP 572, but (mirroring collect_walrus_names) that
        // expression-walking depth is not built this wave.
        _ => {}
    }
}

/// A body's function-parameter names — every kind (positional-only,
/// normal, `*args`, keyword-only, `**kwargs`).
pub(in crate::check) fn collect_parameter_names(parameters: &Parameters, bound: &mut HashSet<String>) {
    for parameter in parameters.posonlyargs.iter().chain(parameters.args.iter()) {
        bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    for parameter in &parameters.kwonlyargs {
        bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    if let Some(vararg) = parameters.vararg.as_ref() {
        bound.insert(vararg.name.id.as_str().to_owned());
    }
    if let Some(kwarg) = parameters.kwarg.as_ref() {
        bound.insert(kwarg.name.id.as_str().to_owned());
    }
}
