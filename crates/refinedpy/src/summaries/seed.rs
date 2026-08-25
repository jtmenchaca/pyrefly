/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::function_table::FunctionTable;

use super::effects::collect_nonlocal_names;
use super::interpret::collect_bound_names;

/// `body`'s own first statement, SKIPPING a leading string-literal
/// `Expr` statement (a docstring) — the probe target `call_result_with_
/// enclosing`'s own decline handler reads to tell "the body never got
/// off the ground" apart from "the body read concretely for a while,
/// then declined." A docstring is documentation, not a readable
/// effect: `Doc/reference/compound_stmts.rst`'s `funcdef` grammar
/// states no special docstring statement at all — it is an ordinary
/// bare string-literal expression statement that CPython happens to
/// bind to `__doc__` — so `interpret_body` always succeeds on it alone
/// (the same `Stmt::Expr` arm any other bare expression statement
/// takes), and probing it in isolation would wrongly read as "this
/// body is readable" for a body whose only OTHER statement is a raise.
/// Skips every LEADING docstring-shaped statement (never just the
/// first one), though CPython itself recognizes at most one — a
/// second string-literal statement right after the first is an
/// ordinary (if unusual) expression statement, and skipping it too
/// costs nothing since it is equally not a readable effect. `None`
/// when the body is empty, or contains nothing but docstring-shaped
/// statements.
pub(crate) fn first_non_docstring_statement(body: &[Stmt]) -> Option<&Stmt> {
    body.iter().find(|stmt| !is_bare_string_literal_statement(stmt))
}

/// Whether `stmt` is a bare string-literal expression statement — the
/// docstring shape `first_non_docstring_statement` skips.
fn is_bare_string_literal_statement(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::StringLiteral(_)))
}

/// Whether `body` is a STUB body — PEP 484's "Stub Files" convention
/// (typeshed's own written form for a declaration with no runtime
/// implementation), read here for an INLINE `def` rather than a `.pyi`
/// file: a body whose only non-docstring statement is a bare `...`
/// (`Expr::EllipsisLiteral`), and nothing follows it. `first_non_
/// docstring_statement`'s own leading-docstring skip applies first
/// (`def f() -> Age:\n    """docs"""\n    ...\n` is a stub exactly as
/// much as one with no docstring), so this checks the body's own FIRST
/// REAL statement, then requires it be the body's LAST statement too —
/// `def f() -> Age:\n    ...\n    return 200\n` is an ordinary body
/// that merely opens with a stray `...` expression, not a stub, and
/// still interprets through `interpret_body`'s ordinary `Stmt::Expr`
/// arm unchanged.
pub(super) fn is_stub_body(body: &[Stmt]) -> bool {
    let Some(first_statement) = first_non_docstring_statement(body) else {
        return false;
    };
    let is_ellipsis = matches!(first_statement, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::EllipsisLiteral(_)));
    is_ellipsis && std::ptr::eq(first_statement, body.last().expect("first_non_docstring_statement found a statement, so body is non-empty"))
}

/// Copies every name `enclosing` binds that `def`'s own body does NOT
/// itself bind (checked against the same locally-bound set
/// `fresh_body_environment` builds — parameters plus every
/// `collect_bound_names` target) into `into`. A parameter always wins
/// its own slot regardless of what `enclosing` holds (`bind_parameters`
/// runs AFTER this and overwrites), so the seeding order is safe either
/// way; running it first keeps this function's own job to one thing —
/// copying free names — rather than also re-deriving the parameter
/// list.
pub(super) fn seed_free_variables(def: &StmtFunctionDef, enclosing: &Environment, into: &mut Environment) {
    for (name, value) in free_variable_snapshot(def, enclosing) {
        into.bind(&name, value);
    }
}

/// `def`'s own free-name reads, each paired with whatever value
/// `enclosing` currently holds for it — the same copy `seed_free_
/// variables` performs, but returned as a standalone snapshot rather
/// than written directly into a callee environment. `env.rs`'s
/// `closure_snapshot` calls this at the moment a nested def/lambda
/// VALUE is created (rather than at the moment it is CALLED), so a
/// retained callable's closure is pinned to its own definition site,
/// matching Python's own scoping rule instead of whatever happens to
/// be bound wherever it is later invoked.
pub(crate) fn free_variable_snapshot(
    def: &StmtFunctionDef,
    enclosing: &Environment,
) -> std::collections::HashMap<String, AbstractValue> {
    let mut snapshot = std::collections::HashMap::new();
    for free_name in free_names_read(&def.body, &locally_bound_names(def)) {
        if let Some(value) = enclosing.read(&free_name) {
            snapshot.insert(free_name, value.clone());
        }
    }
    snapshot
}

/// Every name `def` binds itself: its parameters of all four flavors,
/// then every name its body binds — EXCEPT a name the body declares
/// `nonlocal` (`collect_nonlocal_names`, the same reach `call_effects`'s
/// own local exclusion already gives its OWN copy of this set — folded
/// in here so the ordinary VALUE-ONLY route shares it too). CPython's own
/// rule is that a `nonlocal`-declared name is NEVER local to this body
/// (executionmodel.rst, "the nonlocal statement causes the listed
/// identifiers to refer to previously bound variables in the nearest
/// enclosing scope"), so `collect_bound_names`' own "any assignment
/// target is a local" default must be corrected for exactly this set —
/// otherwise a `nonlocal n; n += 1` body (`A10.xfer.closure`'s own
/// `counter`/`next_value`) reads `n` as an ordinary local `AugAssign`
/// would never seed from `enclosing`, and the read of `n`'s CURRENT
/// value before the `+= 1` finds nothing bound at all.
///
/// This is the set that decides which of the body's reads are FREE — the
/// complement of it, over the body's own reads, is what `seed_free_
/// variables` copies from the caller and what `needs_enclosing_scope`
/// tests for existence. Both read it here so the gate and the machinery
/// it guards share one definition.
pub(super) fn locally_bound_names(def: &StmtFunctionDef) -> std::collections::HashSet<String> {
    let mut bound = std::collections::HashSet::new();
    for parameter in def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
    {
        bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    if let Some(vararg) = def.parameters.vararg.as_ref() {
        bound.insert(vararg.name.id.as_str().to_owned());
    }
    if let Some(kwarg) = def.parameters.kwarg.as_ref() {
        bound.insert(kwarg.name.id.as_str().to_owned());
    }
    collect_bound_names(&def.body, &mut bound);
    let mut nonlocal_names = std::collections::HashSet::new();
    collect_nonlocal_names(&def.body, &mut nonlocal_names);
    for nonlocal_name in &nonlocal_names {
        bound.remove(nonlocal_name);
    }
    bound
}

/// Every bare `Expr::Name` a parameter's own default expression reads —
/// the candidate names `bind_parameters` tries against the call site's
/// `enclosing` environment before evaluating any default. A default
/// expression can only ever reference an outer name (never one of
/// `def`'s own parameters or locals, which do not exist yet at def
/// time), so this walks with an EMPTY locally-bound set, unlike
/// `free_names_read`'s own body-wide walk.
fn default_expression_free_names(parameters: &[&ruff_python_ast::ParameterWithDefault]) -> Vec<String> {
    let empty = std::collections::HashSet::new();
    let mut names = Vec::new();
    for parameter in parameters {
        if let Some(default_expr) = parameter.default.as_deref() {
            collect_names_in_expr(default_expr, &empty, &mut names);
        }
    }
    names
}

/// Every bare `Expr::Name` read inside `body` whose name is NOT in
/// `locally_bound` — the candidate free variables `seed_free_variables`
/// tries against `enclosing`. Over-approximates safely: a name walked
/// here that `enclosing` never bound either simply finds nothing to
/// copy (`Environment::read` already answers `None` for it, same as
/// before this wave); a name that IS a free read gets its value copied.
/// Walks only the expression positions the restricted interpreter
/// itself reaches (assignment RHS, `if` tests, `return` values) — the
/// same statement forms `interpret_body` recognizes, so this collector
/// never visits a form the interpreter would have declined on anyway.
pub(super) fn free_names_read(body: &[Stmt], locally_bound: &std::collections::HashSet<String>) -> Vec<String> {
    let mut names = Vec::new();
    collect_names_in_body(body, locally_bound, &mut names);
    names
}

fn collect_names_in_body(body: &[Stmt], locally_bound: &std::collections::HashSet<String>, names: &mut Vec<String>) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                collect_names_in_expr(assign.value.as_ref(), locally_bound, names);
                for target in &assign.targets {
                    collect_write_target_base_name(target, locally_bound, names);
                }
            }
            Stmt::AnnAssign(assign) => {
                if let Some(value) = assign.value.as_deref() {
                    collect_names_in_expr(value, locally_bound, names);
                }
            }
            Stmt::AugAssign(assign) => {
                collect_names_in_expr(assign.value.as_ref(), locally_bound, names);
                collect_write_target_base_name(assign.target.as_ref(), locally_bound, names);
            }
            Stmt::Expr(expr_stmt) => collect_names_in_expr(expr_stmt.value.as_ref(), locally_bound, names),
            Stmt::Return(ret) => {
                if let Some(value) = ret.value.as_deref() {
                    collect_names_in_expr(value, locally_bound, names);
                }
            }
            Stmt::If(if_stmt) => {
                collect_names_in_expr(if_stmt.test.as_ref(), locally_bound, names);
                collect_names_in_body(&if_stmt.body, locally_bound, names);
                for clause in &if_stmt.elif_else_clauses {
                    if let Some(test) = clause.test.as_ref() {
                        collect_names_in_expr(test, locally_bound, names);
                    }
                    collect_names_in_body(&clause.body, locally_bound, names);
                }
            }
            _ => {}
        }
    }
}

/// A write TARGET's own free-read candidate: `outlaw["age"] = 200`'s
/// target is `Expr::Subscript { value: Name("outlaw"), slice: "age" }` —
/// `outlaw` is READ (its current value is looked up before the write
/// composes a new one, `write_subscript_target`'s own contract) even
/// though the STATEMENT as a whole is a write, so it is a free-read
/// candidate exactly like any other name appearing on an RHS. Without
/// this walk, `outlaw` — appearing ONLY as a subscript/attribute target's
/// own base, never on any statement's RHS — would never be seeded by
/// `seed_free_variables`, and `write_subscript_target`'s own
/// `environment.read(name)` would find nothing, declining the whole call
/// (this is the captured-receiver-mutation half of the CALLEE-EFFECTS
/// CHANNEL, `call_effects`'s own doc). A bare `Expr::Name` target still
/// bound in `locally_bound` is NOT walked here — that shape is an
/// ORDINARY local bind (`collect_bound_names`'s own job), never a free
/// read of the pre-existing value. A bare `Expr::Name` target NOT in
/// `locally_bound` is the `nonlocal` shape (`locally_bound_names` already
/// removed it): `nonlocal n; n += 1` reads `n`'s pre-existing value from
/// the enclosing scope before the write composes a new one, the exact
/// same "read before write" story the subscript/attribute arms already
/// tell for a captured receiver — so it is walked as a free-read
/// candidate too, the one exception to the "bare Name target is a bind"
/// default. The subscript's own KEY expression (`"age"`) is also walked,
/// on the chance it is itself a free name (`outlaw[key] = 200` where
/// `key` is a captured local) — walked through the ordinary `collect_
/// names_in_expr`, since a key expression is always a READ, never a
/// target.
fn collect_write_target_base_name(target: &Expr, locally_bound: &std::collections::HashSet<String>, names: &mut Vec<String>) {
    match target {
        Expr::Name(name) => {
            if !locally_bound.contains(name.id.as_str()) {
                names.push(name.id.as_str().to_owned());
            }
        }
        Expr::Subscript(subscript) => {
            collect_names_in_expr(subscript.value.as_ref(), locally_bound, names);
            collect_names_in_expr(subscript.slice.as_ref(), locally_bound, names);
        }
        Expr::Attribute(attribute) => {
            // `self.<field> = ...` is handled by this file's own
            // self-aware write path, never through the captured-free-name
            // channel — `self` is always a parameter (method_call_result's
            // own binding), never a free read, so walking it here would be
            // harmless but pointless; every OTHER attribute base (a free
            // name's own field write, out of this wave's fixture rows but
            // not precluded) is still walked the same way a subscript's
            // base is, for the identical reason.
            collect_names_in_expr(attribute.value.as_ref(), locally_bound, names);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_write_target_base_name(element, locally_bound, names);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_write_target_base_name(element, locally_bound, names);
            }
        }
        _ => {}
    }
}

/// A shallow-enough walk over one expression's own bare-Name reads:
/// every `Expr::Name` reached through the operator/call/attribute/
/// subscript/comparison/bool-op/ternary shapes a restricted body's own
/// expressions build from. Not a full AST visitor (this crate has none
/// generic enough to filter by `locally_bound` mid-walk) — it covers
/// the expression shapes the corpus's closure rows actually build
/// (`a.b`, `a[b]`, `a + b`, `a if b else c`, `f(a, b)`), and a shape
/// outside this list simply contributes no candidate name, which is
/// always SAFE (a missed free name just fails to seed, matching this
/// wave's pre-existing "unbound name reads unknown()" behavior) rather
/// than wrong.
fn collect_names_in_expr(expr: &Expr, locally_bound: &std::collections::HashSet<String>, names: &mut Vec<String>) {
    match expr {
        Expr::Name(name) => {
            if !locally_bound.contains(name.id.as_str()) {
                names.push(name.id.as_str().to_owned());
            }
        }
        Expr::UnaryOp(unary) => collect_names_in_expr(unary.operand.as_ref(), locally_bound, names),
        Expr::BinOp(binop) => {
            collect_names_in_expr(binop.left.as_ref(), locally_bound, names);
            collect_names_in_expr(binop.right.as_ref(), locally_bound, names);
        }
        Expr::BoolOp(boolop) => {
            for value in &boolop.values {
                collect_names_in_expr(value, locally_bound, names);
            }
        }
        Expr::Compare(compare) => {
            collect_names_in_expr(compare.left.as_ref(), locally_bound, names);
            for comparator in &compare.comparators {
                collect_names_in_expr(comparator, locally_bound, names);
            }
        }
        Expr::If(ternary) => {
            collect_names_in_expr(ternary.test.as_ref(), locally_bound, names);
            collect_names_in_expr(ternary.body.as_ref(), locally_bound, names);
            collect_names_in_expr(ternary.orelse.as_ref(), locally_bound, names);
        }
        Expr::Attribute(attribute) => collect_names_in_expr(attribute.value.as_ref(), locally_bound, names),
        Expr::Subscript(subscript) => {
            collect_names_in_expr(subscript.value.as_ref(), locally_bound, names);
            collect_names_in_expr(subscript.slice.as_ref(), locally_bound, names);
        }
        Expr::Call(call) => {
            collect_names_in_expr(call.func.as_ref(), locally_bound, names);
            for arg in &call.arguments.args {
                collect_names_in_expr(arg, locally_bound, names);
            }
            for keyword in &call.arguments.keywords {
                collect_names_in_expr(&keyword.value, locally_bound, names);
            }
        }
        _ => {}
    }
}

/// A fresh environment for the callee's body: every parameter name plus
/// every name the body itself binds (this file's own collector, not
/// check.rs's — the two stay independent per the mission's file
/// ownership), the module's function table carried forward so a nested
/// same-module call composes through `evaluate_expression`'s dispatch
/// once that wiring lands.
pub(super) fn fresh_body_environment(def: &StmtFunctionDef, table: Option<&Arc<FunctionTable>>, depth: u32) -> Environment {
    let mut locally_bound = std::collections::HashSet::new();
    for parameter in def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
    {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    // a `*args` parameter's own name is bound too — `bind_parameters`
    // below fills it with the caller's trailing-argument tuple, the same
    // way an ordinary positional parameter's own name is filled.
    if let Some(vararg) = def.parameters.vararg.as_ref() {
        locally_bound.insert(vararg.name.id.as_str().to_owned());
    }
    // a `**kwargs` parameter's own name is bound the same way — `bind_
    // parameters` fills it with the caller's own collected keyword dict.
    if let Some(kwarg) = def.parameters.kwarg.as_ref() {
        locally_bound.insert(kwarg.name.id.as_str().to_owned());
    }
    collect_bound_names(&def.body, &mut locally_bound);
    let mut environment = Environment::new(locally_bound);
    // the CHILD interpretation sits one call deeper than its caller —
    // evaluate_expression's dispatch reads this back so the depth cap
    // engages across the evaluate↔summaries boundary (a self-recursive
    // def would otherwise re-enter at depth 0 forever)
    environment.set_call_depth(depth.saturating_add(1));
    if let Some(table) = table {
        environment.set_functions(table.clone());
    }
    environment
}

/// Binds `arguments` to `def`'s posonlyargs+args in order, THEN a
/// trailing `*args` parameter (when `def` declares one) to every
/// remaining caller argument past the plain positional slots, composed
/// into ONE tuple (`collection_models::tuple_literal_value` — Python's
/// own vararg binding: functions.rst's own "if the syntax `*identifier`
/// is present, it is initialized to a tuple receiving any excess
/// positional parameters"). The call SITE's own argument COUNT and every
/// argument's own VALUE are both fully known at the point this file
/// interprets a call (`positional_arguments_for_def`'s own caller already
/// evaluated every argument in order), so the tail's own length is never
/// an unknown-length abstraction — e-class-and-function.py's
/// `first_age(40, 41)` binds `ages` to the known 2-tuple `(40, 41)`,
/// exactly the shape `ages[0]` needs to read through.
///
/// A trailing plain parameter with no matching argument uses its own
/// default, evaluated in a FRESH (name-less) environment — a default
/// expression may only reference literals/builtins, never an enclosing
/// name, so no name this call knows is visible while reading it. Too few
/// arguments to fill every plain parameter (with an unevaluable or absent
/// default), or too many arguments when `def` declares no `*args` tail at
/// all, declines the whole call.
///
/// `def`'s keyword-only parameters bind from `arguments`' own trailing
/// slots, at positions `plain_parameters.len()..plain_parameters.len()
/// + kwonlyargs.len()` — the exact layout `expressions.rs`'s
/// `positional_arguments_for_def` builds (posonlyargs+args first, then
/// kwonlyargs in declaration order). EVERY kwonly parameter must have a
/// slot there (`arguments.get(...)` answering `None`, meaning the
/// CALLER never covered it by keyword, declines the whole call rather
/// than read a kwonly parameter's own default — this file does not yet
/// carry a "kwonly param defaulted, not supplied" reading path, so a
/// def with an optional kwonly parameter the caller genuinely omits
/// still declines here, a narrower contract than CPython's own but
/// never wrong). A `*args` tail, when `def` also declares one, collects
/// whatever is left AFTER both the plain parameters' own slots AND the
/// kwonly slots — the two features do not collide in practice (a
/// caller passing enough positional arguments to spill into a kwonly
/// slot is a `SyntaxError` at the call site, never a real value this
/// function would see), so reading kwonly's slots out of the tail
/// before the vararg does is always the correct order.
///
/// A `**kwargs` parameter, when `def` declares one, binds from the
/// VERY LAST slot of `arguments` — the collected dict
/// `expressions.rs`'s `positional_arguments_with_kwargs_dict` appends
/// after every plain and kwonly slot (that function's own doc). That
/// final slot is excluded from the plain/kwonly/vararg arithmetic
/// above (it is popped off `arguments` before any other binding reads
/// the tail), so a def combining `**kwargs` with `*args` or kwonly
/// parameters — out of this corpus's own rows, but not precluded —
/// still binds every slot in the right place.
pub(super) fn bind_parameters(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
    enclosing: Option<&Environment>,
) -> Option<()> {
    let (kwargs_value, arguments) = match def.parameters.kwarg.as_ref() {
        Some(_) => {
            let (last, rest) = arguments.split_last()?;
            (Some(last.clone()), rest)
        }
        None => (None, arguments),
    };
    let parameters: Vec<_> = def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .collect();
    let kwonly_parameters: Vec<_> = def.parameters.kwonlyargs.iter().collect();
    let covered = parameters.len() + kwonly_parameters.len();
    if def.parameters.vararg.is_none() && arguments.len() > covered {
        return None;
    }
    // A default expression reads against the CALL SITE's own enclosing
    // environment (module-level bindings, and any name a nested def's
    // own outer scope holds) — `_DEFAULT_BUCKET` in `bucket: list[int] =
    // _DEFAULT_BUCKET` is a module-level name, not a parameter or local
    // of the def itself, so a bare empty environment can never read it.
    // Copying `enclosing`'s OWN bindings wholesale is safe here (never
    // `def`'s own locally-bound names, since a default expression is
    // evaluated once at def time, before any of `def`'s own parameters
    // or body statements exist) — the same one-directional copy
    // `seed_free_variables` performs for a nested def's free reads.
    let mut default_environment = Environment::new(std::collections::HashSet::new());
    if let Some(enclosing) = enclosing {
        for free_name in default_expression_free_names(&parameters) {
            if let Some(value) = enclosing.read(&free_name) {
                default_environment.bind(&free_name, value.clone());
            }
        }
    }
    for (index, parameter) in parameters.iter().enumerate() {
        let value = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            let default_expr = parameter.default.as_deref()?;
            evaluate_expression(default_expr, &default_environment, kernel)
        };
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }
    if let Some(kwarg) = def.parameters.kwarg.as_ref() {
        let value = kwargs_value.expect("split_last above must have set this whenever kwarg.is_some()");
        environment.bind(kwarg.name.id.as_str(), value);
    }
    for (offset, parameter) in kwonly_parameters.iter().enumerate() {
        let value = arguments.get(parameters.len() + offset)?.clone();
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }
    if let Some(vararg) = def.parameters.vararg.as_ref() {
        let tail: Vec<AbstractValue> = arguments.iter().skip(covered).cloned().collect();
        let tail_value = crate::collection_models::tuple_literal_value(&tail);
        environment.bind(vararg.name.id.as_str(), tail_value);
    }
    Some(())
}
