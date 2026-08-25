//! Blocker-path forgets: what an unmodeled branch/loop/match arm binds
//! or mutates, so the walk stays conservative rather than reading a
//! stale pre-branch value past a construct it declined to walk.

use std::collections::HashSet;

use ruff_python_ast::{ExceptHandler, Expr, Stmt};

use crate::env::Environment;

/// Removes every bare name a (possibly destructuring) Assign/AugAssign
/// target touches from `provably_unbound` — an observed WRITE to a name
/// this table is tracking cures it, the same way `judge_and_bind`'s own
/// write-sink laws bind/forget the environment for that name. Applies
/// even to a nested tuple/list/starred target: any target position that
/// names the tracked name is itself proof CPython bound it on this path.
pub(in crate::check) fn forget_target_from_provably_unbound(target: &Expr, provably_unbound: &mut HashSet<String>) {
    match target {
        Expr::Name(name) => {
            provably_unbound.remove(name.id.as_str());
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                forget_target_from_provably_unbound(element, provably_unbound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                forget_target_from_provably_unbound(element, provably_unbound);
            }
        }
        Expr::Starred(starred) => forget_target_from_provably_unbound(starred.value.as_ref(), provably_unbound),
        _ => {}
    }
}

/// Forget every plain name reachable inside a target expression
/// (nested tuple/list/starred targets included) — used where the walk
/// cannot state what value lands in each position, and by `del` (every
/// deleted name is simply forgotten).
pub(in crate::check) fn forget_target_names(target: &Expr, environment: &mut Environment) {
    match target {
        Expr::Name(name) => environment.forget(name.id.as_str()),
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                forget_target_names(element, environment);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                forget_target_names(element, environment);
            }
        }
        Expr::Starred(starred) => forget_target_names(starred.value.as_ref(), environment),
        _ => {}
    }
}

/// Forget every name a single statement binds anywhere within its own
/// sub-bodies (its target plus every name any nested body binds) — the
/// blocker-path cleanup for a `for`/`while` the loop module declined:
/// reuses `collect_bound_names_stmt`'s own walk of that statement's
/// shape so the "what does this bind" answer never drifts from the
/// scope prepass's.
pub(in crate::check) fn forget_names_bound_by_stmt(stmt: &Stmt, environment: &mut Environment) {
    let mut bound = HashSet::new();
    let mut excluded = HashSet::new();
    super::collect_bound_names_stmt(stmt, &mut bound, &mut excluded);
    for name in &excluded {
        bound.remove(name);
    }
    for name in &bound {
        environment.forget(name);
    }
}

/// Forget every name a body binds anywhere within it — the blocker-path
/// cleanup for a `match` the arm-decision module declined to resolve
/// (used per undecided case body, since the walk cannot say which arm,
/// if any, actually ran).
pub(in crate::check) fn forget_names_bound_in_body(body: &[Stmt], environment: &mut Environment) {
    let mut bound = HashSet::new();
    let mut excluded = HashSet::new();
    super::collect_bound_names(body, &mut bound, &mut excluded);
    for name in &excluded {
        bound.remove(name);
    }
    for name in &bound {
        environment.forget(name);
    }
}

/// STALE-RECEIVER SOUNDNESS, unmodeled-body law: `collect_bound_names`
/// (and `collect_bound_names_stmt`) only name the slots a body BINDS —
/// an assignment/for/with-as/except/walrus target, a parameter, an
/// import. A name that is only ever MUTATED inside an unmodeled body
/// (never itself the target of `=`) is invisible to that scan, so the
/// blocker-path forgets above leave its stale pre-loop/pre-match value
/// standing — exactly the shape `grouped.setdefault(...).append(age)`
/// inside a declined `for` takes: `grouped` is never assigned, only
/// mutated through a chained method call, so a post-loop read of
/// `grouped` wrongly kept reading the empty dict from before the loop
/// (c-reads-and-values.py:1008's own WRONG ANSWER: an unmatched
/// "provably raises KeyError" fire on a key the mutation actually
/// wrote).
///
/// This function is the second half of the same forget: a syntactic
/// walk over every statement and expression in `stmt`, collecting the
/// LEFTMOST `Name` reachable under two receiver shapes — an
/// ATTRIBUTE-CALL's receiver (`X.method(...)`, the func of a `Call`
/// being an `Attribute`) and a SUBSCRIPT-STORE's receiver (`X[k] = v`,
/// an assign target that is a `Subscript`) — walking THROUGH a chained
/// call's own func-attribute the way `grouped.setdefault(...).append(...)`
/// requires (the `.append` receiver is itself a Call, whose own func is
/// another Attribute reaching back to `grouped`). Every collected base
/// name is forgotten, on top of (never replacing) `forget_names_bound_by_stmt`'s
/// own bound-name forgets — sound and narrow: this is a syntactic
/// over-approximation (a plain non-mutating method call like
/// `x.keys()` is also swept up), never a false negative, since a stale
/// receiver surviving an unmodeled body is exactly the wrong-answer
/// shape this law exists to close.
pub(in crate::check) fn forget_mutated_receivers_in_stmt(stmt: &Stmt, environment: &mut Environment) {
    let mut receivers = HashSet::new();
    collect_mutation_receiver_names_stmt(stmt, &mut receivers);
    for name in &receivers {
        environment.forget(name);
    }
}

/// The per-case-body sibling of `forget_mutated_receivers_in_stmt`, for
/// a `match` the arm-decision module declined to resolve — one case
/// body at a time, matching `forget_names_bound_in_body`'s own calling
/// convention.
pub(in crate::check) fn forget_mutated_receivers_in_body(body: &[Stmt], environment: &mut Environment) {
    let mut receivers = HashSet::new();
    for stmt in body {
        collect_mutation_receiver_names_stmt(stmt, &mut receivers);
    }
    for name in &receivers {
        environment.forget(name);
    }
}

/// Walks one statement's own sub-bodies and every expression it
/// contains, collecting every attribute-call/subscript-store receiver's
/// leftmost base name into `receivers` — see
/// `forget_mutated_receivers_in_stmt`'s own doc for the exact contract.
pub(in crate::check) fn collect_mutation_receiver_names_stmt(stmt: &Stmt, receivers: &mut HashSet<String>) {
    match stmt {
        Stmt::Assign(assign) => {
            for target in &assign.targets {
                collect_subscript_store_receiver(target, receivers);
            }
            collect_mutation_receiver_names_expr(assign.value.as_ref(), receivers);
        }
        Stmt::AnnAssign(assign) => {
            collect_subscript_store_receiver(assign.target.as_ref(), receivers);
            if let Some(value) = assign.value.as_deref() {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Stmt::AugAssign(assign) => {
            collect_subscript_store_receiver(assign.target.as_ref(), receivers);
            collect_mutation_receiver_names_expr(assign.value.as_ref(), receivers);
        }
        Stmt::Expr(expr_stmt) => collect_mutation_receiver_names_expr(expr_stmt.value.as_ref(), receivers),
        Stmt::Return(ret) => {
            if let Some(value) = ret.value.as_deref() {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Stmt::Delete(delete) => {
            for target in &delete.targets {
                collect_mutation_receiver_names_expr(target, receivers);
            }
        }
        Stmt::Assert(assert) => {
            collect_mutation_receiver_names_expr(assert.test.as_ref(), receivers);
            if let Some(msg) = assert.msg.as_deref() {
                collect_mutation_receiver_names_expr(msg, receivers);
            }
        }
        Stmt::Raise(raise) => {
            if let Some(exc) = raise.exc.as_deref() {
                collect_mutation_receiver_names_expr(exc, receivers);
            }
            if let Some(cause) = raise.cause.as_deref() {
                collect_mutation_receiver_names_expr(cause, receivers);
            }
        }
        Stmt::If(if_stmt) => {
            collect_mutation_receiver_names_expr(if_stmt.test.as_ref(), receivers);
            for inner in &if_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for clause in &if_stmt.elif_else_clauses {
                if let Some(test) = clause.test.as_ref() {
                    collect_mutation_receiver_names_expr(test, receivers);
                }
                for inner in &clause.body {
                    collect_mutation_receiver_names_stmt(inner, receivers);
                }
            }
        }
        Stmt::For(for_stmt) => {
            collect_mutation_receiver_names_expr(for_stmt.iter.as_ref(), receivers);
            for inner in &for_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for inner in &for_stmt.orelse {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::While(while_stmt) => {
            collect_mutation_receiver_names_expr(while_stmt.test.as_ref(), receivers);
            for inner in &while_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for inner in &while_stmt.orelse {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::With(with_stmt) => {
            for item in &with_stmt.items {
                collect_mutation_receiver_names_expr(&item.context_expr, receivers);
            }
            for inner in &with_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::Try(try_stmt) => {
            for inner in &try_stmt.body {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for handler in &try_stmt.handlers {
                let ExceptHandler::ExceptHandler(handler) = handler;
                for inner in &handler.body {
                    collect_mutation_receiver_names_stmt(inner, receivers);
                }
            }
            for inner in &try_stmt.orelse {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
            for inner in &try_stmt.finalbody {
                collect_mutation_receiver_names_stmt(inner, receivers);
            }
        }
        Stmt::Match(match_stmt) => {
            collect_mutation_receiver_names_expr(match_stmt.subject.as_ref(), receivers);
            for case in &match_stmt.cases {
                if let Some(guard) = case.guard.as_deref() {
                    collect_mutation_receiver_names_expr(guard, receivers);
                }
                for inner in &case.body {
                    collect_mutation_receiver_names_stmt(inner, receivers);
                }
            }
        }
        // a nested def/class body has its own scope — the names its own
        // mutations touch are not this outer body's receivers to forget
        Stmt::FunctionDef(_) | Stmt::ClassDef(_) => {}
        Stmt::Pass(_)
        | Stmt::Break(_)
        | Stmt::Continue(_)
        | Stmt::Global(_)
        | Stmt::Nonlocal(_)
        | Stmt::Import(_)
        | Stmt::ImportFrom(_)
        | Stmt::TypeAlias(_)
        | Stmt::IpyEscapeCommand(_) => {}
    }
}

/// A (possibly destructuring) assign/aug-assign/ann-assign target's own
/// SUBSCRIPT-STORE receivers (`X[k] = v` at any nesting depth of a
/// tuple/list/starred target) — the leftmost base name under each
/// `Subscript.value` collected via `collect_leftmost_receiver_name`.
/// Non-subscript target shapes (a bare name, an attribute write) name no
/// subscript-store receiver here; a bare name's own binding is already
/// covered by `collect_bound_names`'s separate scan, and an attribute
/// write's receiver is covered by this same walk's expression side
/// (`collect_mutation_receiver_names_expr`'s `Expr::Attribute` arm on
/// the RHS/nested reads) — assignment TARGETS reach this function only
/// for their subscript form, which is the one shape `forget_names_bound_by_stmt`
/// cannot already see.
pub(in crate::check) fn collect_subscript_store_receiver(target: &Expr, receivers: &mut HashSet<String>) {
    match target {
        Expr::Subscript(subscript) => {
            collect_leftmost_receiver_name(subscript.value.as_ref(), receivers);
            collect_mutation_receiver_names_expr(subscript.slice.as_ref(), receivers);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_subscript_store_receiver(element, receivers);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_subscript_store_receiver(element, receivers);
            }
        }
        Expr::Starred(starred) => collect_subscript_store_receiver(starred.value.as_ref(), receivers),
        _ => {}
    }
}

/// Walks one expression tree, collecting every ATTRIBUTE-CALL's receiver
/// base name (`X.method(...)` — the func of a `Call` being an
/// `Attribute`) into `receivers`, recursing into every sub-expression a
/// mutation could hide inside (call arguments, comparison operands,
/// boolean/binary/unary operands, container displays, the ternary's
/// three arms, f-string interpolations, comprehension element/iterable/
/// condition parts, await/yield operands) so a nested mutating call
/// anywhere in the tree is caught, not only at the statement's own top
/// level.
pub(in crate::check) fn collect_mutation_receiver_names_expr(expr: &Expr, receivers: &mut HashSet<String>) {
    match expr {
        Expr::Call(call) => {
            if let Expr::Attribute(attribute) = call.func.as_ref() {
                collect_leftmost_receiver_name(attribute.value.as_ref(), receivers);
            }
            collect_mutation_receiver_names_expr(call.func.as_ref(), receivers);
            for arg in &call.arguments.args {
                collect_mutation_receiver_names_expr(arg, receivers);
            }
            for keyword in &call.arguments.keywords {
                collect_mutation_receiver_names_expr(&keyword.value, receivers);
            }
        }
        Expr::Attribute(attribute) => collect_mutation_receiver_names_expr(attribute.value.as_ref(), receivers),
        Expr::Subscript(subscript) => {
            collect_mutation_receiver_names_expr(subscript.value.as_ref(), receivers);
            collect_mutation_receiver_names_expr(subscript.slice.as_ref(), receivers);
        }
        Expr::Named(named) => {
            collect_mutation_receiver_names_expr(named.target.as_ref(), receivers);
            collect_mutation_receiver_names_expr(named.value.as_ref(), receivers);
        }
        Expr::BoolOp(op) => {
            for value in &op.values {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Expr::BinOp(op) => {
            collect_mutation_receiver_names_expr(op.left.as_ref(), receivers);
            collect_mutation_receiver_names_expr(op.right.as_ref(), receivers);
        }
        Expr::UnaryOp(op) => collect_mutation_receiver_names_expr(op.operand.as_ref(), receivers),
        Expr::If(if_expr) => {
            collect_mutation_receiver_names_expr(if_expr.test.as_ref(), receivers);
            collect_mutation_receiver_names_expr(if_expr.body.as_ref(), receivers);
            collect_mutation_receiver_names_expr(if_expr.orelse.as_ref(), receivers);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_mutation_receiver_names_expr(element, receivers);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_mutation_receiver_names_expr(element, receivers);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                collect_mutation_receiver_names_expr(element, receivers);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    collect_mutation_receiver_names_expr(key, receivers);
                }
                collect_mutation_receiver_names_expr(&item.value, receivers);
            }
        }
        Expr::Compare(compare) => {
            collect_mutation_receiver_names_expr(compare.left.as_ref(), receivers);
            for comparator in &compare.comparators {
                collect_mutation_receiver_names_expr(comparator, receivers);
            }
        }
        Expr::Starred(starred) => collect_mutation_receiver_names_expr(starred.value.as_ref(), receivers),
        Expr::Slice(slice) => {
            if let Some(lower) = slice.lower.as_deref() {
                collect_mutation_receiver_names_expr(lower, receivers);
            }
            if let Some(upper) = slice.upper.as_deref() {
                collect_mutation_receiver_names_expr(upper, receivers);
            }
            if let Some(step) = slice.step.as_deref() {
                collect_mutation_receiver_names_expr(step, receivers);
            }
        }
        Expr::FString(fstring) => {
            for element in fstring.value.elements() {
                if let Some(interpolation) = element.as_interpolation() {
                    collect_mutation_receiver_names_expr(interpolation.expression.as_ref(), receivers);
                }
            }
        }
        Expr::Await(inner) => collect_mutation_receiver_names_expr(inner.value.as_ref(), receivers),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                collect_mutation_receiver_names_expr(value, receivers);
            }
        }
        Expr::YieldFrom(inner) => collect_mutation_receiver_names_expr(inner.value.as_ref(), receivers),
        Expr::ListComp(comp) => {
            collect_mutation_receiver_names_expr(comp.elt.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        Expr::SetComp(comp) => {
            collect_mutation_receiver_names_expr(comp.elt.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        Expr::DictComp(comp) => {
            if let Some(key) = comp.key.as_deref() {
                collect_mutation_receiver_names_expr(key, receivers);
            }
            collect_mutation_receiver_names_expr(comp.value.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        Expr::Generator(comp) => {
            collect_mutation_receiver_names_expr(comp.elt.as_ref(), receivers);
            collect_comprehension_generators(&comp.generators, receivers);
        }
        // a lambda's own body is a separate scope — mirrors
        // collect_walrus_names/bind_walrus_targets's same posture
        Expr::Lambda(_) => {}
        _ => {}
    }
}

/// A comprehension's own generator clauses: each `iter` expression and
/// every `if` condition, in source order — the loop VARIABLE itself
/// introduces no receiver to collect.
pub(in crate::check) fn collect_comprehension_generators(generators: &[ruff_python_ast::Comprehension], receivers: &mut HashSet<String>) {
    for generator in generators {
        collect_mutation_receiver_names_expr(&generator.iter, receivers);
        for condition in &generator.ifs {
            collect_mutation_receiver_names_expr(condition, receivers);
        }
    }
}

/// The leftmost `Name` reachable under a receiver expression, walking
/// THROUGH a chained call's own func-attribute — unlike
/// `receiver_base_name` (which stops at a `Call` and answers `None`),
/// this function keeps walking into a `Call`'s `func` so
/// `grouped.setdefault(...).append(...)`'s outer receiver
/// (`grouped.setdefault(...)`, itself a `Call`) still resolves to
/// `grouped`. Every argument/keyword of a call encountered along the
/// way is ALSO walked for its own nested mutations (a mutation can hide
/// inside an argument expression, e.g. `xs.append(ys.pop())`), and a
/// non-Name/Attribute/Call receiver (a subscript, a literal, …) yields
/// no base name — this function only ever forgets a plain identifier.
pub(in crate::check) fn collect_leftmost_receiver_name(receiver: &Expr, receivers: &mut HashSet<String>) {
    match receiver {
        Expr::Name(name) => {
            receivers.insert(name.id.as_str().to_owned());
        }
        Expr::Attribute(attribute) => collect_leftmost_receiver_name(attribute.value.as_ref(), receivers),
        Expr::Call(call) => {
            collect_leftmost_receiver_name(call.func.as_ref(), receivers);
            for arg in &call.arguments.args {
                collect_mutation_receiver_names_expr(arg, receivers);
            }
            for keyword in &call.arguments.keywords {
                collect_mutation_receiver_names_expr(&keyword.value, receivers);
            }
        }
        _ => {}
    }
}
