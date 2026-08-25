/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::lattice_operations::join_known;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;

use crate::env::Environment;
use crate::function_table::FunctionTable;
use crate::function_table::ENTRY_MODULE;

use super::compile::kernel_summary_result;
use super::interpret::interpret_body;
use super::seed::bind_parameters;
use super::seed::first_non_docstring_statement;
use super::seed::free_names_read;
use super::seed::fresh_body_environment;
use super::seed::is_stub_body;
use super::seed::locally_bound_names;
use super::seed::seed_free_variables;
use super::sorts::declared_return_seed;
use super::sorts::return_sort_fallback;

/// The deepest a call chain interprets before declining outright. A
/// same-module call whose body calls itself (directly or through a
/// cycle of same-module calls) would otherwise interpret forever; the
/// cap turns that into an honest decline rather than a hang, matching
/// the corpus's recursion row (n-file).
pub const CALL_DEPTH_CAP: u32 = 8;

/// `def`'s answer for one call with `arguments` bound positionally, or
/// `None` when the body (or its parameter shape) is outside what this
/// file interprets. See the module doc for the body forms interpreted
/// and the a-statements:399-404 seam this unblocks.
///
/// A thin wrapper over `call_result_with_enclosing` passing `None` — no
/// enclosing environment, so a free name inside `def`'s body (one
/// neither a parameter nor a name the body itself binds) reads as
/// `unknown()` exactly as before this wave.
pub fn call_result(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> Option<AbstractValue> {
    call_result_with_enclosing(def, arguments, table, kernel, depth, None)
}

/// `call_result`'s own answer, PLUS a closure's read of an ENCLOSING
/// local: `enclosing` is the call-SITE's own environment (the caller's
/// locals at the point `def` — a nested `def` — is invoked), read only
/// for a name `def`'s body itself never binds (not a parameter, not an
/// `Assign`/`AnnAssign`/`AugAssign`/`if`-arm target) — Python's own
/// scoping rule (`tmp/cpython/Doc/reference/executionmodel.rst`,
/// "Naming and binding" — "if a name is bound in a block, it is a local
/// variable of that block... free variables may refer to bindings in
/// the enclosing function scope"). Every such free name still bound in
/// `enclosing` is copied into the callee's fresh environment BEFORE
/// interpretation starts (`Environment` has no scope-chain lookup of
/// its own — `evaluate_expression`'s `Expr::Name` arm reads one flat
/// `bindings` map — so pre-seeding is the one way a captured read
/// succeeds without widening `Environment`'s own shape). A WRITE to an
/// enclosing name from inside the callee (`nonlocal`, a-statements.py's
/// `nonlocal_rebind`) is not modeled: the copy is one-directional, into
/// the fresh environment only, and nothing here reads `nonlocal`
/// declarations or propagates a write back to `enclosing` — a caller
/// needing that is out of this function's scope (report's Blockers).
pub fn call_result_with_enclosing(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    enclosing: Option<&Environment>,
) -> Option<AbstractValue> {
    if depth >= CALL_DEPTH_CAP {
        return return_sort_fallback(def);
    }
    // RECURSION WITH A MEASURE (A10.xfer.recursion.py's own `fact`/
    // `fact_inside`): a SELF-recursive def (`fact` calls `fact` in its
    // own body) called with a WINDOW argument (`n ∈ [0, 5]`, never a
    // single concrete value) cannot resolve through the ordinary
    // interpreter below at all — `n <= 1` stays undecided the whole way
    // down, so BOTH arms stay live at every depth, the recursive call's
    // own argument (`n - 1`) stays a window too at every step, and the
    // depth cap is reached with `n` STILL unresolved, falling back to
    // `return_sort_fallback`'s bare, unbounded sort (never `[1, 120]`).
    // Tried BEFORE the kernel-summary route below (which the module doc's
    // own `kernel_summary_result` — "the lowering reaches no callee
    // today — a call declines the body" — already declines for any
    // self-recursive def, so trying this first costs nothing the kernel
    // route would have answered instead): a small bounded window ENUMERATES
    // to its own concrete members (`enumerated_recursive_call`), and each
    // concrete call recurses through this SAME function, one call per
    // member, joined — the sound answer a purely symbolic unroll can never
    // reach, since it is EXACTLY what CPython itself computes, one call
    // per admitted `n`. A call whose arguments do not match this shape
    // (no self-recursion, no lone bounded-window integer argument, a
    // window too large to enumerate) answers `None` here and falls
    // through to every route below unchanged.
    if let Some(answer) = enumerated_recursive_call(def, arguments, table, kernel, depth, enclosing) {
        return Some(answer);
    }
    // THE KERNEL SUMMARY ROUTE, tried ahead of the concrete
    // interpretation below. The body is lowered and compiled ONCE per
    // `def` (`summary_registry`), and this call sends only its own
    // argument states; the answer is the kernel's, carrying the same
    // soundness the walk carries (`summarize_eq`). Everything it cannot
    // serve — a body outside the lowering's grammar, an argument the
    // state wire cannot spell, a kernel that declines — falls through to
    // the interpreter unchanged, so this route only ever ADDS answers.
    //
    // GATED ON A PROPERTY OF THE DEF, never on whether this CALLER
    // happened to supply an environment. Every ordinary def call passes
    // one (`expressions.rs`'s own call arm), so a gate reading
    // `enclosing.is_some()` would exclude every ordinary call and leave
    // the route reachable only from the callback arms. What the
    // exclusion is really about is the enclosing MACHINERY below — free-
    // variable seeding, retained-callable inheritance, class-table
    // inheritance — and whether that machinery has anything to do is
    // decided by the def's own body (`needs_enclosing_scope`).
    //
    // The def's MODULE comes from the table that answered it: a def
    // reached through an import carries the stamp of the module that
    // DECLARED it (`function_table.rs`), so a cross-module call keys to
    // the declaring module's own summary and never to a same-named,
    // same-spanned def in another file. A def reached with no table at
    // all is the calling module's own, so it keys under `ENTRY_MODULE`.
    if !needs_enclosing_scope(def) {
        let module = table
            .and_then(|table| table.module_of(def.name.id.as_str()))
            .unwrap_or(ENTRY_MODULE);
        if let Some(answer) = kernel_summary_result(def, module, arguments) {
            return Some(answer);
        }
    }
    // A `*args` tail binds to a KNOWN-LENGTH tuple of the caller's own
    // trailing positional arguments (`bind_parameters`'s own vararg row,
    // below) — this file models it exactly, not as a decline, since the
    // tail's own element count and each element's own value are both
    // fully known at the call site (`first_age(40, 41)`'s `*ages` binds
    // to the 2-tuple `(40, 41)`, not an unknown-length abstraction). A
    // keyword-only parameter is likewise no longer a hard decline:
    // `expressions.rs`'s `positional_arguments_for_def` maps every
    // keyword-only param the CALLER covered by name onto a trailing
    // slot of `arguments` (that function's own doc — declaration order,
    // appended after `posonlyargs`/`args`), so `bind_parameters` below
    // reads those trailing slots back apart by position; a kwonly param
    // the caller left uncovered (no keyword, no default read here)
    // still declines through that function's own arity check. A
    // `**kwargs` parameter is the SAME story one slot further out:
    // `expressions.rs`'s `positional_arguments_with_kwargs_dict`
    // collects every keyword the call site passes that names no plain
    // or kwonly parameter into ONE dict and appends it as the FINAL
    // slot of `arguments` — `bind_parameters` below reads that final
    // slot and binds it to the `kwarg` parameter's own name.
    let mut environment = fresh_body_environment(def, table, depth);
    if let Some(enclosing) = enclosing {
        seed_free_variables(def, enclosing, &mut environment);
        // RETAINED CALLABLES: this call's own environment shares the
        // CALLER's retained-callable table (the same `Arc<Mutex<...>>>`,
        // never a copy) rather than starting a fresh, empty one — a
        // nested def this call's own body creates
        // (`interpret_body`'s `Stmt::FunctionDef` arm, r-ast-census.py's
        // `wrapper`) is returned OUT of this call and invoked later
        // from the CALLER's own environment, which must still be able
        // to look its table entry back up at that later call site
        // (`env::Environment::inherit_retained_callables`'s own doc).
        environment.inherit_retained_callables(enclosing);
        // CLASSES: this call's own environment ALSO inherits the
        // caller's class table when it never set one of its own — a
        // same-module def interpreted here may itself construct a
        // class instance (e-class-and-function.py's `pick`: `store =
        // Store(40)`, called through `pick(lambda s: s.age)` — the
        // retained lambda's own body reads `s.age` off that instance),
        // and `evaluate_call`'s construction arm only ever resolves a
        // class by reading `environment.classes()` — `None` here
        // otherwise, since `fresh_body_environment` never populates it
        // on its own.
        if environment.classes().is_none() {
            if let Some(classes) = enclosing.classes() {
                environment.set_classes(classes.clone());
            }
        }
        // DECLARED ALIASES: the same inherit-when-unset rule `classes`
        // just took, for the reason `declared_return_seed`'s own doc
        // states — `fresh_body_environment` never populates this table
        // on its own, so without inheriting it here, a call this file
        // cannot interpret (a stub body, a genuine decline) would answer
        // only the three bare `int`/`float`/`str` sorts even when the
        // CALLER's own environment carries the full alias table
        // `check.rs::walk_body_with_self_binding` seeded it with.
        if environment.declared_aliases().is_none() {
            if let Some((aliases, imports)) = enclosing.declared_aliases() {
                environment.set_declared_aliases(aliases.clone(), imports.clone());
            }
        }
        // DATETIME IMPORTS: the same inherit-when-unset rule `classes`
        // just took, for the identical reason — a same-module def
        // interpreted here may itself construct/call a `datetime`
        // class the CALLER's own module aliased (`from datetime import
        // date as d`), and `evaluate_call`'s datetime gates only ever
        // resolve that alias by reading `environment.datetime_imports()`
        // — `None` here otherwise, since `fresh_body_environment` never
        // populates it on its own.
        if environment.datetime_imports().is_none() {
            if let Some(datetime_imports) = enclosing.datetime_imports() {
                environment.set_datetime_imports(datetime_imports.clone());
            }
        }
        // LOCALE PREMISE: the same inherit-when-unset rule, for the
        // identical reason — a same-module def interpreted here may
        // itself call `datetime.strptime` with a `%a` directive, and
        // that reading needs the caller's own module-wide
        // `locale.setlocale`-never-called premise
        // (`module_never_calls_setlocale`'s own doc), not a fresh
        // `None` this interpreted body's own `Environment::new` would
        // otherwise carry.
        if environment.locale_never_set().is_none() {
            if let Some(locale_never_set) = enclosing.locale_never_set() {
                environment.set_locale_never_set(locale_never_set);
            }
        }
    }
    let Some(()) = bind_parameters(def, arguments, kernel, &mut environment, enclosing) else {
        return declared_return_seed(def, &environment).or_else(|| return_sort_fallback(def));
    };

    // A stub body (PEP 484's "Stub Files" convention, restated for an
    // inline definition by typing.rst's own `...` placeholder example:
    // a body that is exactly one `Expr::EllipsisLiteral` statement,
    // optionally preceded by a leading docstring) is DECLARATION-ONLY —
    // it states no runtime behavior for `interpret_body` to read.
    // Recognized here, before the ordinary interpretation below, so a
    // stub answers its own declared return annotation
    // (`declared_return_seed`/`return_sort_fallback`) the same way any
    // other body this interpreter cannot get off the ground already
    // does (`raise NotImplementedError`'s own first-statement-declines
    // path, further down) — never `interpret_body`'s ordinary
    // `Stmt::Expr` arm, which would evaluate the bare `...` and discard
    // it like `pass`, falling off the end into a fabricated
    // `null_value()` return that carries no relation to what the
    // annotation actually declares.
    if is_stub_body(&def.body) {
        return declared_return_seed(def, &environment).or_else(|| return_sort_fallback(def));
    }

    let mut returns: Vec<AbstractValue> = Vec::new();
    let Some(falls_through) = interpret_body(&def.body, kernel, depth, &mut environment, &mut returns, None) else {
        // The body declined SOMEWHERE inside `interpret_body`'s statement
        // walk — but WHERE matters: a def opaque from its very first
        // statement (`unread_number`'s `raise NotImplementedError`,
        // a-statements.py:34) never produced any readable effect, so the
        // bare `-> int`/`float`/`str` annotation is the only claim left to
        // make, and `return_sort_fallback` is honest. A def whose body
        // interprets one or more statements CONCRETELY before the decline
        // (e-class-and-function.py's `grow_into_bucket`: `bucket.append(age)`
        // reads fine, only the later `return bucket[0]` decides on an
        // unknown() value because the mutable-default parameter's value
        // is opaque) is NOT opaque — it is a genuinely unread VALUE inside
        // an otherwise-readable body, and the coarse whole-sort claim would
        // overstate what this interpreter actually knows. Re-running the
        // interpreter on just the body's own FIRST REAL statement (a
        // fresh, throwaway environment/returns pair — this probe never
        // contributes to the real answer) tells the two cases apart:
        // still declining there means the def never got off the ground;
        // succeeding there means the later decline was mid-body, and the
        // honest answer is unknown(), never a guessed sort. "First REAL"
        // skips a LEADING docstring (`first_non_docstring_statement`'s
        // own doc): `unread_number`'s body is a docstring followed by
        // `raise NotImplementedError` (a-statements.py:34-38), and the
        // docstring ALONE always interprets fine (`Stmt::Expr` evaluates
        // and discards its string-literal value, same as any other bare
        // expression statement) — probing the docstring by itself would
        // wrongly read as "the body got off the ground," masking that the
        // body's first REAL statement is the one that declines. A
        // docstring is documentation, never a readable effect; a body
        // that is nothing but a docstring then a decline is exactly as
        // opaque as a body that declines immediately.
        let Some(first_statement) = first_non_docstring_statement(&def.body) else {
            return declared_return_seed(def, &environment).or_else(|| return_sort_fallback(def));
        };
        let mut probe_environment = fresh_body_environment(def, table, depth);
        if let Some(enclosing) = enclosing {
            seed_free_variables(def, enclosing, &mut probe_environment);
        }
        if bind_parameters(def, arguments, kernel, &mut probe_environment, enclosing).is_none() {
            return declared_return_seed(def, &environment).or_else(|| return_sort_fallback(def));
        }
        let mut probe_returns: Vec<AbstractValue> = Vec::new();
        let first_statement_declines = interpret_body(
            std::slice::from_ref(first_statement),
            kernel,
            depth,
            &mut probe_environment,
            &mut probe_returns,
            None,
        )
        .is_none();
        if first_statement_declines {
            return declared_return_seed(def, &environment).or_else(|| return_sort_fallback(def));
        }
        return None;
    };
    if falls_through {
        returns.push(null_value());
    }

    let mut answers = returns.into_iter();
    let Some(first) = answers.next() else {
        return declared_return_seed(def, &environment).or_else(|| return_sort_fallback(def));
    };
    let joined = answers.fold(first, |acc, next| join_known(acc, next));
    Some(joined)
}

// --- RECURSION WITH A MEASURE, UNROLLED BY ENUMERATION --------------
//
// `fact(n)` for `n` a WINDOW (never a single value) cannot be answered
// by walking the body once: `n <= 1` never decides, so every level of
// the depth-capped interpreter stays symbolic and the whole call
// bottoms out at `return_sort_fallback`'s bare sort. The one sound
// route left is the one CPython itself takes: run the call once per
// admitted `n`, concretely, and join every answer. This is enumeration
// at the CALL SITE, never a change to `interpret_body`'s own statement
// walk or to `CALL_DEPTH_CAP` — a concrete `fact(5)` still recurses
// through the ordinary depth-capped path, five levels deep, each level
// now holding a single concrete `n` the body's own `n <= 1` test DOES
// decide.

/// The largest window this file enumerates concretely — chosen to match
/// `expressions.rs`'s own `MULTI_VALUE_CROSS_PRODUCT_CAP` (16), the one
/// existing "how many members is too many to enumerate" convention in
/// this crate, rather than a fresh number invented for this route alone.
const RECURSIVE_ENUMERATION_CAP: usize = 16;

/// `enumerated_recursive_call`'s own gate: `def` is SELF-recursive (its
/// body calls its own name somewhere, direct — not through a cycle of
/// OTHER same-module defs, which this reader does not chase) AND
/// `arguments` carries EXACTLY ONE argument, currently `Kind::Set` over
/// an Integer-tagged window with a provable `[lo, hi]` (`window_of`) of
/// width `<= RECURSIVE_ENUMERATION_CAP` members. Scoped to exactly one
/// argument — a multi-parameter recursive def (an accumulator-style
/// `fact(n, acc)`) is out of this reader's own reach, matching the
/// fixture's own shape (`fact(n: int) -> int`) rather than a guessed
/// wider contract. Returns the enumerable argument's own `[lo, hi]`
/// window; `None` for every other shape (not self-recursive, zero or
/// two-plus arguments, the sole argument not a bounded integer window,
/// or a window wider than the cap) falls through to every other route
/// unchanged.
fn enumerable_recursive_argument(def: &StmtFunctionDef, arguments: &[AbstractValue]) -> Option<(f64, f64)> {
    if !calls_own_name(&def.body, def.name.id.as_str()) {
        return None;
    }
    let [only_argument] = arguments else {
        return None;
    };
    if only_argument.kind != Kind::Set {
        return None;
    }
    if !only_argument.set.forms.iter().any(|form| form.form == Form::Integer) {
        return None;
    }
    let (lo, hi) = window_of(&only_argument.set)?;
    if hi < lo {
        return None;
    }
    // the width comparison stays in floats: a huge window cast to
    // usize saturates and the +1 overflows — comparing before any
    // cast declines the wide window instead
    let width = (hi - lo).round();
    if width < 0.0 || width + 1.0 > RECURSIVE_ENUMERATION_CAP as f64 {
        return None;
    }
    Some((lo, hi))
}

/// `set`'s own `[lo, hi]` window when it carries exactly one `AtLeast`
/// lower form and one `AtMost` upper form (an `Integer`/`MultipleOf` form
/// alongside them states the sort and never widens the window) — the
/// same shape `check.rs`'s own `aug_assign_window` reads, restated here
/// since this file owns no dependency on `check.rs`'s private readers.
fn window_of(set: &RefinedSet) -> Option<(f64, f64)> {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &set.forms {
        match form.form {
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost => hi = Some(form.a),
            Form::Integer | Form::MultipleOf => {}
            _ => return None,
        }
    }
    Some((lo?, hi?))
}

/// Whether `body` calls a bare name `own_name` anywhere — the same
/// restricted statement/expression reach `free_names_read`'s own
/// collectors give (`Assign`/`AnnAssign`/`AugAssign`/`If`/`Return`/
/// `Expr`, one level of `if`/elif/else nesting), since a call outside
/// that reach is a shape `interpret_body` cannot walk at all and the
/// enumeration route below would decline through the ordinary
/// interpreter on its own first concrete call anyway.
fn calls_own_name(body: &[Stmt], own_name: &str) -> bool {
    body.iter().any(|stmt| statement_calls_own_name(stmt, own_name))
}

fn statement_calls_own_name(stmt: &Stmt, own_name: &str) -> bool {
    match stmt {
        Stmt::Assign(assign) => expr_calls_own_name(assign.value.as_ref(), own_name),
        Stmt::AnnAssign(assign) => assign.value.as_deref().is_some_and(|value| expr_calls_own_name(value, own_name)),
        Stmt::AugAssign(assign) => expr_calls_own_name(assign.value.as_ref(), own_name),
        Stmt::Expr(expr_stmt) => expr_calls_own_name(expr_stmt.value.as_ref(), own_name),
        Stmt::Return(ret) => ret.value.as_deref().is_some_and(|value| expr_calls_own_name(value, own_name)),
        Stmt::If(if_stmt) => {
            calls_own_name(&if_stmt.body, own_name)
                || if_stmt.elif_else_clauses.iter().any(|clause| calls_own_name(&clause.body, own_name))
        }
        _ => false,
    }
}

/// A shallow expression walk over the same shapes `collect_names_in_expr`
/// reaches, asking only whether a `Call` node's own callee names
/// `own_name` — never resolving whether that name is truly a recursive
/// self-call (a local rebinding of the same spelling would still count,
/// the same over-approximation `free_names_read`'s own doc calls always
/// SAFE: a false-positive "is recursive" answer only sends a call through
/// this route's own extra window check, which itself declines harmlessly
/// on a shape it cannot enumerate).
fn expr_calls_own_name(expr: &Expr, own_name: &str) -> bool {
    match expr {
        Expr::Call(call) => {
            let callee_matches = matches!(call.func.as_ref(), Expr::Name(name) if name.id.as_str() == own_name);
            callee_matches
                || expr_calls_own_name(call.func.as_ref(), own_name)
                || call.arguments.args.iter().any(|arg| expr_calls_own_name(arg, own_name))
                || call.arguments.keywords.iter().any(|keyword| expr_calls_own_name(&keyword.value, own_name))
        }
        Expr::UnaryOp(unary) => expr_calls_own_name(unary.operand.as_ref(), own_name),
        Expr::BinOp(binop) => expr_calls_own_name(binop.left.as_ref(), own_name) || expr_calls_own_name(binop.right.as_ref(), own_name),
        Expr::BoolOp(boolop) => boolop.values.iter().any(|value| expr_calls_own_name(value, own_name)),
        Expr::Compare(compare) => {
            expr_calls_own_name(compare.left.as_ref(), own_name) || compare.comparators.iter().any(|comparator| expr_calls_own_name(comparator, own_name))
        }
        Expr::If(ternary) => {
            expr_calls_own_name(ternary.test.as_ref(), own_name)
                || expr_calls_own_name(ternary.body.as_ref(), own_name)
                || expr_calls_own_name(ternary.orelse.as_ref(), own_name)
        }
        Expr::Attribute(attribute) => expr_calls_own_name(attribute.value.as_ref(), own_name),
        Expr::Subscript(subscript) => {
            expr_calls_own_name(subscript.value.as_ref(), own_name) || expr_calls_own_name(subscript.slice.as_ref(), own_name)
        }
        _ => false,
    }
}

/// `enumerable_recursive_argument`'s own answer, folded into one call per
/// admitted `n` in `[lo, hi]`: each member replaces the sole argument
/// with its own exact `Kind::Values` singleton, calls `call_result_with_
/// enclosing` at the SAME `depth` (this is a call-SITE transform, not an
/// extra level of interpretation — the enumerated call's own body still
/// recurses through the ordinary depth-capped path from here), and every
/// answer joins through `join_known`, the same fold `call_result_with_
/// enclosing`'s own multi-return join already uses. `None` when the
/// gate itself declines, OR when even ONE enumerated member's own call
/// declines (an honest "this whole claim is unproven" rather than
/// joining a partial answer that quietly drops a member CPython could
/// actually reach).
fn enumerated_recursive_call(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    enclosing: Option<&Environment>,
) -> Option<AbstractValue> {
    let (lo, hi) = enumerable_recursive_argument(def, arguments)?;
    // `enumerable_recursive_argument` gates on the window carrying
    // `Form::Integer` — the only sort this reader admits — so every
    // enumerated member is Integer-sorted, whether or not the window's
    // own `kind_tag` happens to carry that tag explicitly (a `Kind::Set`
    // value's `kind_tag` is sometimes `Some(PrimitiveKind::Integer)`
    // (a declared-refinement-seeded parameter) and sometimes `None`
    // (`known_set`'s own bare construction) — the GATE already proved
    // the sort either way, so the fallback below is never a guess.
    let sort = arguments[0].kind_tag.unwrap_or(PrimitiveKind::Integer);
    let member_count = (hi - lo).round() as usize + 1;
    let mut answers: Vec<AbstractValue> = Vec::with_capacity(member_count);
    let mut member = lo;
    while member <= hi {
        let member_value = known_values(vec![member], sort, TrustProved);
        let member_arguments = vec![member_value];
        let answer = call_result_with_enclosing(def, &member_arguments, table, kernel, depth, enclosing)?;
        answers.push(answer);
        member += 1.0;
    }
    let mut answers = answers.into_iter();
    let first = answers.next()?;
    Some(answers.fold(first, |acc, next| join_known(acc, next)))
}

// --- THE KERNEL SUMMARY ROUTE ---------------------------------------
//
// A `def`'s body is lowered to the kernel's flow IR and COMPILED once
// (`refined_kernel::summary_questions::ask_summarize`); every call after
// that sends only its own entry states (`ask_apply_summary`) and reads
// the exit at the result slot. The interpreter above re-walks the body
// per call; this route walks it never.
//
// The store is keyed by the `def` ALONE: a summary quantifies over every
// entry, so it is context-free and one `def` has exactly one compiled
// answer whatever any call passes. The key is the def's MODULE, its
// NAME, and its own source RANGE — `FunctionTable` hands out CLONES of a
// parsed def, so a pointer would be a different identity at every call
// site while the module/name/range triple is the same for every clone of
// one source def and different for any two source defs.
//
// The MODULE is what makes the key unique across a whole program rather
// than within one file. A `TextRange` is a byte offset into ONE module's
// source, so two sibling modules that both open with `def scale(x):
// return x * 2` give their defs the same name and the same span; without
// the module, one module's compiled summary would answer the other's
// calls. `FunctionTable` carries each def's own module for exactly this
// (`function_table.rs`'s own doc), and an imported def keeps the stamp of
// the module that DECLARED it, so a def reached through a re-export chain
// keys to one summary however many local names it is reached under.

/// Whether interpreting `def` needs the CALLER's environment at all —
/// the def-level property the kernel-summary gate reads.
///
/// Four pieces of machinery read the caller's environment, and each has
/// a precondition that is a property of the DEF rather than of the call:
///
/// 1. FREE-VARIABLE SEEDING copies every name the body reads that the
///    body itself does not bind. It has something to do exactly when
///    `free_names_read` finds such a name — so this asks that same
///    question, over the same locally-bound set `free_variable_snapshot`
///    builds, and answers true when any free read exists.
/// 2. RETAINED-CALLABLE INHERITANCE matters only for a body that creates
///    or returns a callable.
/// 3. CLASS-TABLE INHERITANCE matters only for a body that constructs a
///    class instance, which it can only do by CALLING one.
/// 4. A PARAMETER DEFAULT is evaluated against a copy of the caller's
///    bindings (`bind_parameters`), so a default naming an outer name
///    reads the enclosing scope as surely as a body read does.
///
/// Only (1) is tested here. The other three are already impossible for
/// any def that compiles to a summary at all: the lowering is
/// total-or-decline and spells no nested `def`, no `lambda`, no call of
/// any kind, and no defaulted parameter — so a body reaching one of them
/// never reaches the apply path whatever this answers. Testing them
/// again would be a second statement of the same invariant, and the two
/// could drift.
///
/// This function's remaining job is therefore (1), plus skipping the
/// kernel attempt cheaply for the bodies that will need the interpreter
/// anyway.
///
/// A def this answers TRUE for keeps the concrete interpreter outright,
/// exactly as before. A def it answers FALSE for reads only its own
/// parameters and locals, so the summary's entry vector carries
/// everything the body can see and the caller's environment adds nothing.
pub(super) fn needs_enclosing_scope(def: &StmtFunctionDef) -> bool {
    // reads the SAME free-name question the seeding itself asks —
    // `locally_bound_names` is the set `free_variable_snapshot` builds
    // before its own copy, so the gate and the machinery it guards can
    // never disagree about which names are free
    !free_names_read(&def.body, &locally_bound_names(def)).is_empty()
}
