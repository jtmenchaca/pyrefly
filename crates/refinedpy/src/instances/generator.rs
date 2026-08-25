//! Generator yield summarization.

use std::sync::Arc;

use refined_domain::abstract_value::{AbstractValue, Kind};
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{Expr, ExprCall, Number, Stmt, StmtFunctionDef, UnaryOp};

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::function_table::FunctionTable;

/// A generator body's own yielded values, in order — `Some(Vec::new())`
/// for a body that yields nothing on its only path, `None` when the
/// body is outside the two shapes this function reads (a CONDITIONAL
/// yield, `yield from`, any restricted-body statement this function
/// itself does not walk). Models ONLY the yields themselves; a
/// `next(gen)` call's OWN read of "the first yield" is the WIRING
/// owner's job (`expressions.rs`'s `evaluate_call`) — this function
/// hands back the full ordered list so that caller can index position 0
/// (or answer a join over every yielded value, for a plain `for x in
/// gen():` walk, should that wiring choose to).
///
/// Two accepted top-level statement shapes, walked in source order and
/// merged into one ordered list (a LEADING docstring is skipped first,
/// `yields_of_body`'s own doc):
///
/// 1. A STRAIGHT-LINE `yield <expr>` statement (an `Expr` statement
///    whose value is `Expr::Yield`) — the yielded value evaluates
///    against the current environment and is appended in place. A bare
///    `return` ends iteration without yielding (datamodel.rst's
///    generator-function entry) — no more statements after it are read,
///    and a straight-line body's own return-with-a-value shape
///    (`StopIteration`'s `.value`) is outside this function's scope
///    (never read by `next()`'s own first-value contract).
/// 2. `for <name> in <literal iterable>: yield <expr>` — a-statements.py's
///    `stream()` shape (`for value in (10, 20, 30): yield value`,
///    wrapped in `async def` — this domain collapses `for`/`async for`
///    into the identical `StmtFor` node, ruff's own generated.rs doc:
///    "collapses the synchronous and asynchronous variants into a
///    single type"). Modeled ONLY when the loop's own iterable reads
///    through `literal_iterable_values` below (a literal list/tuple of
///    number literals, or `range(...)` with int-literal args — the same
///    two syntactic shapes `loops.rs`'s own reader accepts, reimplemented
///    LOCALLY per this addendum's own scope rather than importing that
///    file), the target a bare Name, the body EXACTLY one `yield <expr>`
///    statement, and no `else` clause (a `for...else` is outside this
///    shape). Each element binds the SAME environment in turn (the
///    elements are already fully known, so no branch of the walk can
///    see a stale binding) — parameters and any prior straight-line
///    bindings stay visible to the yield expression, matching CPython's
///    own left-to-right iteration order (compound_stmts.rst, "The `for`
///    statement").
///
/// Any other statement shape (an `if`, a `while`, a nested `for` whose
/// iterable is not one of the two literal forms, a `for` whose body is
/// not exactly one `yield`, …) declines the WHOLE body — `None`, never a
/// partial list. A CONDITIONAL yield (`if <test>: yield <expr>`) is a
/// deliberate, permanent decline — q-decline-names.py's own
/// `age_generator` row states this as one of its file's two genuine
/// soundness boundaries: "a generator whose yield sits under a
/// CONDITION is beyond the straight-line summary the checker reads."
/// This function must never join an `if`/`else`'s own yields into one
/// answer, even though the values involved would often be sound to
/// join — the row's own purpose is to teach that this shape stays
/// undetermined.
///
/// `arguments`/`table`/`kernel`/`depth` mirror `summaries::call_result`
/// exactly (parameters bind positionally, the module's function table
/// composes a nested same-module call, the depth cap declines a runaway
/// chain) — a generator's parameter list is bound exactly like an
/// ordinary function's own.
pub fn generator_yields(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> Option<Vec<AbstractValue>> {
    use crate::summaries::CALL_DEPTH_CAP;
    if depth >= CALL_DEPTH_CAP {
        return None;
    }
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() || !def.parameters.kwonlyargs.is_empty() {
        return None;
    }
    let parameters: Vec<_> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .collect();
    if arguments.len() > parameters.len() {
        return None;
    }
    let mut locally_bound = std::collections::HashSet::new();
    for parameter in &parameters {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    let mut environment = Environment::new(locally_bound);
    // one call deeper than the caller — the depth cap engages across
    // the evaluate↔interpreter boundary (see env::call_depth)
    environment.set_call_depth(depth.saturating_add(1));
    if let Some(table) = table {
        environment.set_functions(table.clone());
    }
    let default_environment = Environment::new(Default::default());
    for (index, parameter) in parameters.iter().enumerate() {
        let value = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            let default_expr = parameter.default.as_deref()?;
            evaluate_expression(default_expr, &default_environment, kernel)
        };
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }

    yields_of_body(&def.body, &mut environment, kernel)
}

/// `generator_yields`'s own body walk, over `def.body` — see that
/// function's own doc for the two straight-line-yield shapes, PLUS the
/// one CONDITIONAL shape this function now summarizes: `if <test>: yield
/// <expr>` with no `elif`/`else` clause and no other statement in the
/// `if`'s own body. CPython's real generator-iterator protocol either
/// runs that `yield` (the test is true on this pass) or skips straight to
/// whatever statement follows it (the test is false) — this function has
/// no way to decide WHICH, so it states the sound over-approximation for
/// "the next value `__next__` could produce at this position": the JOIN
/// of the conditional yield's own value with whatever value the REST of
/// the body would produce if this position were skipped entirely
/// (`yields_of_body`'s own recursive call over the statements after the
/// `if`). A conditional yield followed by more yields therefore never
/// widens the overall yielded COUNT — it only widens the VALUE at the one
/// position where the branch and its continuation compete to be "the
/// value read there" (`age_generator`'s own row: `if bool([]): yield 40`
/// then `yield 41` answers ONE position, `join(40, 41)`, never two
/// separate positions). A conditional yield with NOTHING after it (no
/// unconditional yield anywhere later in the body) still declines — the
/// join needs a second value to join against, and a length-zero-or-one
/// generator is a shape this function does not spell (its own `Vec` return
/// has no way to say "zero or one," only "exactly N" — the caller reading
/// `items.first()` for `next()` would otherwise wrongly treat a possibly-
/// empty position as always-present).
///
/// A LEADING docstring (a bare string-literal `Expr` statement,
/// `summaries::first_non_docstring_statement`'s own shape) is skipped
/// before the walk starts — a docstring is documentation, never a
/// readable effect (that function's own doc), so a generator whose body
/// opens with one must not decline solely because its first statement is
/// not `Expr::Yield`.
fn yields_of_body(body: &[Stmt], environment: &mut Environment, kernel: &Arc<RefinedTSKernel>) -> Option<Vec<AbstractValue>> {
    let Some(first) = crate::summaries::first_non_docstring_statement(body) else {
        // nothing but leading docstrings — no yield anywhere
        return Some(Vec::new());
    };
    let skip = body.iter().position(|stmt| std::ptr::eq(stmt, first)).expect("first came from this same body");
    let body = &body[skip..];
    let mut yields = Vec::new();
    for (position, stmt) in body.iter().enumerate() {
        match stmt {
            // `if <test>: yield <expr>` — no `elif`/`else`, exactly one
            // statement in the `if`'s own body — this function's own doc
            // states the join this arm computes. `continuation` is every
            // statement AFTER this `if` in source order, summarized
            // recursively (a FRESH docstring-skip is harmless here: there
            // is no docstring mid-body to skip, `first_non_docstring_
            // statement` simply returns the continuation's own first
            // statement unchanged). `None` from EITHER the conditional
            // arm's own value or the continuation still declines the
            // whole body — this join is sound only when both sides of it
            // are themselves fully known.
            Stmt::If(if_stmt) if if_stmt.elif_else_clauses.is_empty() => {
                let [Stmt::Expr(if_body_expr_stmt)] = if_stmt.body.as_slice() else {
                    return None;
                };
                let Expr::Yield(if_yield_expr) = if_body_expr_stmt.value.as_ref() else {
                    return None;
                };
                let conditional_value = match if_yield_expr.value.as_deref() {
                    Some(value_expr) => evaluate_expression(value_expr, environment, kernel),
                    None => refined_domain::abstract_value::null_value(),
                };
                if conditional_value.kind == Kind::Unknown {
                    return None;
                }
                let continuation = yields_of_body(&body[position + 1..], environment, kernel)?;
                let mut continuation = continuation.into_iter();
                let Some(next_value) = continuation.next() else {
                    // nothing yielded after this conditional position — the
                    // real generator sometimes yields nothing AT ALL past
                    // here (StopIteration on the very first `__next__`
                    // call), a length-zero-or-one shape this function's own
                    // `Vec` return cannot spell (see this function's own
                    // doc) — decline rather than claim a length this
                    // reading did not prove.
                    return None;
                };
                yields.push(refined_domain::lattice_operations::join_known(conditional_value, next_value));
                yields.extend(continuation);
                return Some(yields);
            }
            Stmt::Expr(expr_stmt) => {
                let Expr::Yield(yield_expr) = expr_stmt.value.as_ref() else {
                    return None;
                };
                let value = match yield_expr.value.as_deref() {
                    Some(value_expr) => evaluate_expression(value_expr, environment, kernel),
                    None => refined_domain::abstract_value::null_value(),
                };
                if value.kind == Kind::Unknown {
                    return None;
                }
                yields.push(value);
            }
            // a bare `return` inside a generator ends iteration without
            // yielding (datamodel.rst's generator-function entry) — no
            // more statements after it are read, and a straight-line
            // body's own return-with-a-value shape (`StopIteration`'s
            // `.value`) is outside this function's scope (never read by
            // `next()`'s own first-value contract).
            Stmt::Return(_) => break,
            // `for <name> in <literal iterable>: yield <expr>` — see
            // this function's own doc, shape 2.
            Stmt::For(for_stmt) => {
                if !for_stmt.orelse.is_empty() {
                    return None;
                }
                let Expr::Name(target_name) = for_stmt.target.as_ref() else {
                    return None;
                };
                let [Stmt::Expr(body_expr_stmt)] = for_stmt.body.as_slice() else {
                    return None;
                };
                let Expr::Yield(yield_expr) = body_expr_stmt.value.as_ref() else {
                    return None;
                };
                let Some(value_expr) = yield_expr.value.as_deref() else {
                    return None;
                };
                let elements = literal_iterable_values(for_stmt.iter.as_ref())?;
                for element in elements {
                    environment.bind(target_name.id.as_str(), element);
                    let value = evaluate_expression(value_expr, environment, kernel);
                    if value.kind == Kind::Unknown {
                        return None;
                    }
                    yields.push(value);
                }
            }
            _ => return None,
        }
    }
    Some(yields)
}

/// The elements a generator's own `for <target> in <iterable>: yield
/// <expr>` shape iterates over, restricted to the two syntactic forms
/// this addendum reads: a `List`/`Tuple` DISPLAY of bare number literals
/// (`(10, 20, 30)`, `literal_number_elements`'s own literal-only
/// reading — an element that is not a bare number literal declines the
/// WHOLE iterable rather than falling back to a wider evaluated read,
/// since this reader is deliberately the SMALL syntactic subset the
/// addendum scopes it to), or `range(...)` with 1-3 INT-literal
/// arguments (`range` rejects a float argument at call time — the same
/// restriction `loops.rs`'s own `int_literal_value` states). Every
/// produced value is Integer- or Float-sorted per its own literal syntax
/// (never a joined `PrimitiveKind::Number`). `None` for any other
/// iterable shape — a name, a call to anything but `range`, a
/// non-literal element — this reader declines rather than guess.
fn literal_iterable_values(iterable: &Expr) -> Option<Vec<AbstractValue>> {
    match iterable {
        Expr::List(list) => literal_number_elements(&list.elts),
        Expr::Tuple(tuple) => literal_number_elements(&tuple.elts),
        Expr::Call(call) => literal_range_values(call),
        _ => None,
    }
}

/// Every element of a `List`/`Tuple` display read as a bare (optionally
/// unary +/- wrapped) number literal — `None` the moment one element is
/// not that exact shape.
fn literal_number_elements(elements: &[Expr]) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::with_capacity(elements.len());
    for element in elements {
        values.push(literal_number_value(element)?);
    }
    Some(values)
}

/// A bare (possibly unary +/- wrapped) `NumberLiteral`'s exact value,
/// tagged with its own CPython sort — the same reading `loops.rs`'s own
/// `sorted_number_literal_value` gives, reimplemented locally per this
/// function's own module (the addendum's own "do NOT import loops.rs").
fn literal_number_value(expression: &Expr) -> Option<AbstractValue> {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved)),
            Number::Float(value) => Some(known_values(vec![*value], PrimitiveKind::Float, TrustProved)),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_number_value(unary.operand.as_ref())?;
            let sort = operand.kind_tag?;
            let value = operand.values.first().copied()?;
            let signed = if unary.op == UnaryOp::USub { -value } else { value };
            Some(known_values(vec![signed], sort, TrustProved))
        }
        _ => None,
    }
}

/// A `range(...)` call's produced Integer-sorted values, `None` when the
/// callee is not the bare name `range`, an argument is not an INT
/// literal, the argument count is not 1/2/3, or the step is 0 — the same
/// reading `loops.rs`'s own `range_call_values` gives, reimplemented
/// locally (this function's own module owns no dependency on `loops.rs`
/// per the addendum's scope).
fn literal_range_values(call: &ExprCall) -> Option<Vec<AbstractValue>> {
    use refined_domain::abstract_value::{known_values, PrimitiveKind};
    use refined_domain::trust_grades::TrustProved;
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "range" {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let args = &call.arguments.args;
    let (start, stop, step) = match args.len() {
        1 => (0.0, literal_int_value(&args[0])?, 1.0),
        2 => (literal_int_value(&args[0])?, literal_int_value(&args[1])?, 1.0),
        3 => (
            literal_int_value(&args[0])?,
            literal_int_value(&args[1])?,
            literal_int_value(&args[2])?,
        ),
        _ => return None,
    };
    if step == 0.0 {
        return None;
    }
    let mut values = Vec::new();
    let mut current = start;
    // r[i] = start + step*i, while r[i] < stop (step > 0) or r[i] > stop
    // (step < 0) — library/stdtypes.rst's own range formula
    if step > 0.0 {
        while current < stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    } else {
        while current > stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    }
    Some(values)
}

/// A `range()` argument's value, restricted to an INT literal — `range`
/// rejects a float argument at call time, so this reader stays honest
/// about that CPython restriction rather than silently truncating.
fn literal_int_value(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            _ => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_int_value(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
}
