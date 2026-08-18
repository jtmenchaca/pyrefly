/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A same-module `def`'s answer for one call: concrete evaluation of a
//! BOUNDED body — the same posture `loops.rs`'s `run_restricted_body`
//! takes for loop bodies, extended to the restricted statement forms a
//! function body needs (branching and `return`, which a loop body never
//! has). `call_result` binds the callee's parameters to the caller's
//! argument values, interprets the body statements it recognizes, and
//! answers the join of every value the body could return — or declines
//! (`None`) the moment the body does something this file does not
//! interpret, so a caller never gets a guessed answer.
//!
//! This is the a-statements:399-404 seam: `helper_never_answers_none`
//! returns a dict literal on both the `if` arm and the fall-through —
//! `{"age": 40}` and `{"age": 10}`. Once `expressions.rs` evaluates
//! dict literals, this file's `if`/`else` handling joins those two
//! Object values into one Object answer that is never `Kind::Null`,
//! which is exactly what lets the walk prove `held is None` false at
//! `none_test_on_helper_that_never_answers_none`'s call site.
//!
//! Keyword arguments are the WIRING owner's job: `call_result` takes
//! only POSITIONAL argument values, in parameter order. A caller with a
//! keyword call maps each keyword to its parameter's position before
//! calling this function; this file has no keyword-name matching of
//! its own.
//!
//! `interpret_assign`/`interpret_aug_assign` also recognize a
//! `self.<field> = <expr>` / `self.<field> += <expr>` target: when
//! `self` is bound to a known instance (only true inside
//! `instances::method_call_result`'s own environment, never inside an
//! ordinary `call_result`), the write updates the WORKING instance
//! through `instances::field_write` and rebinds `self` so a later
//! `self.<field>` read in the same body sees it. This is the one seam
//! `instances.rs`'s method interpreter shares with this file's
//! restricted body walk, rather than duplicating `interpret_body`'s
//! statement dispatch.

use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAnnAssign;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtIf;
use ruff_text_size::TextRange;

use crate::refinedpy::collection_models::dict_with_item;
use crate::refinedpy::collection_models::list_with_item;
use crate::refinedpy::env::Environment;
use crate::refinedpy::expressions::binary_arithmetic_value;
use crate::refinedpy::expressions::evaluate_expression;
use crate::refinedpy::function_table::FunctionTable;
use crate::refinedpy::instances::class_table;
use crate::refinedpy::instances::field_read;
use crate::refinedpy::instances::field_write;
use crate::refinedpy::instances::self_attribute_name;
use crate::refinedpy::instances::ClassModel;
use crate::refinedpy::surface::surface_imports;

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
    }
    let Some(()) = bind_parameters(def, arguments, kernel, &mut environment) else {
        return return_sort_fallback(def);
    };

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
            return return_sort_fallback(def);
        };
        let mut probe_environment = fresh_body_environment(def, table, depth);
        if let Some(enclosing) = enclosing {
            seed_free_variables(def, enclosing, &mut probe_environment);
        }
        if bind_parameters(def, arguments, kernel, &mut probe_environment).is_none() {
            return return_sort_fallback(def);
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
            return return_sort_fallback(def);
        }
        return None;
    };
    if falls_through {
        returns.push(null_value());
    }

    let mut answers = returns.into_iter();
    let Some(first) = answers.next() else {
        return return_sort_fallback(def);
    };
    let joined = answers.fold(first, |acc, next| join_known(acc, next));
    Some(joined)
}

/// `call_result_with_enclosing`'s own answer, PLUS every ENCLOSING-SCOPE
/// write the body itself performs — the channel that `call_result_with_
/// enclosing`'s own doc names as out of its scope ("A WRITE to an
/// enclosing name from inside the callee... is not modeled"):
/// a-statements.py's `nonlocal_rebind` (`nonlocal age` then `age = 200`)
/// and `closure_mutates_flattened_capture` (`outlaw["age"] = 200`, a
/// mutation THROUGH a captured free name, no `nonlocal` needed since the
/// write never rebinds `outlaw` itself — CPython's own rule,
/// executionmodel.rst's "Naming and binding": "if a name is bound in a
/// block, it is a local variable of that block" applies to the NAME
/// `outlaw`, never to a subscript/attribute STORE through it, so no
/// `nonlocal` declaration is needed or read for that shape).
///
/// Two kinds of effect, both read against the SAME interpreted run
/// `call_result_with_enclosing` would produce (this function re-runs the
/// body rather than sharing state with that call, since the two answers
/// serve different callers — a value-only call site never needs the
/// effect list, and building it costs one extra interpretation of an
/// already-bounded, already depth-capped body):
///
/// 1. A `nonlocal <name>` declaration anywhere at this body's own
///    TOP LEVEL (`collect_nonlocal_names`, one level of `if`/elif/else
///    nesting included, matching `interpret_if`'s own reach) followed by
///    a plain `name = <expr>` / `name op= <expr>` assignment: the
///    ENCLOSING scope's own `age` is what CPython actually rebinds
///    (executionmodel.rst: "The nonlocal statement causes... names to
///    refer to previously bound variables in the nearest enclosing
///    scope"), so the effect is the assignment's own evaluated value —
///    judged by the CALLER (`check.rs`'s statement-level dispatch)
///    against the enclosing body's OWN declared table exactly as a
///    straight-line `age = 200` would be, which is what makes
///    `nonlocal_rebind`'s own row FIRE: the outer `age` is a declared
///    `Age` slot, and 200 is the effect value judged against it.
/// 2. A STORE THROUGH A FREE NAME: `<free-name>[<key>] = <value>` or
///    `<free-name>.<field> = <value>` where `<free-name>` is neither a
///    parameter nor a name this body's own statements bind (the same
///    `locally_bound` set `fresh_body_environment` builds) — composes
///    the receiver's NEW value via `collection_models::dict_with_item`/
///    `list_with_item` (subscript) or `instances::field_write`
///    (attribute), reading the free name's CURRENT value from
///    `enclosing` first (so two writes to the same captured name inside
///    one call compose, matching real execution order) — a store this
///    function cannot compose (a receiver shape neither helper answers,
///    or a free name `enclosing` never bound) answers that name
///    `unknown()` instead of dropping the effect silently: the caller
///    MUST forget a name this function could not account for, never
///    keep a stale pre-call value.
///
/// Returns `None` under the exact same conditions
/// `call_result_with_enclosing` would decline outright (the depth cap,
/// an unsupported parameter shape, or `interpret_body` declining the
/// body) — an effect list is only ever built alongside a value this
/// call genuinely answers, never as a consolation prize for an otherwise
/// declined call.
pub fn call_effects(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    enclosing: &Environment,
) -> Option<(AbstractValue, Vec<(String, AbstractValue)>)> {
    let value = call_result_with_enclosing(def, arguments, table, kernel, depth, Some(enclosing))?;

    let mut nonlocal_names = std::collections::HashSet::new();
    collect_nonlocal_names(&def.body, &mut nonlocal_names);

    // `collect_bound_names` reads any `name = ...` target as a LOCAL
    // binding — it has no `nonlocal` awareness of its own (a restricted
    // body never had one to read before this channel existed). A name
    // this body declares `nonlocal` is, by CPython's own scoping rule,
    // NEVER local (executionmodel.rst: "the nonlocal statement causes
    // the listed identifiers to refer to previously bound variables in
    // the nearest enclosing scope"), so it is removed here — this is
    // what lets `seed_free_variables` (below) copy its CURRENT value in
    // from `enclosing` for a shape like `nonlocal age; age = age + 1`
    // to read correctly, and what lets `record_write_effect`'s own
    // subscript/attribute arms treat it as a free base name too.
    let mut locally_bound = std::collections::HashSet::new();
    for parameter in def.parameters.posonlyargs.iter().chain(def.parameters.args.iter()) {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    collect_bound_names(&def.body, &mut locally_bound);
    for nonlocal_name in &nonlocal_names {
        locally_bound.remove(nonlocal_name);
    }

    let mut effect_environment = fresh_body_environment(def, table, depth);
    seed_free_variables(def, enclosing, &mut effect_environment);
    if bind_parameters(def, arguments, kernel, &mut effect_environment).is_none() {
        return Some((value, Vec::new()));
    }
    let mut effects: Vec<(String, AbstractValue)> = Vec::new();
    collect_call_effects(&def.body, kernel, &mut effect_environment, &nonlocal_names, &locally_bound, &mut effects);
    Some((value, effects))
}

/// Every name declared `nonlocal` anywhere at `body`'s own top level or
/// one level inside an `if`/elif/else arm — the same reach
/// `interpret_if`/`interpret_undecided_arms` give an ordinary statement,
/// since a `nonlocal` declaration inside an untaken arm still applies to
/// this scope regardless of which arm executes (CPython resolves
/// `nonlocal` at COMPILE time, not at the declaring statement's own
/// runtime position — executionmodel.rst, "the nonlocal statement...
/// applies to the entire scope of a function or class body").
fn collect_nonlocal_names(body: &[Stmt], names: &mut std::collections::HashSet<String>) {
    for stmt in body {
        match stmt {
            Stmt::Nonlocal(nonlocal) => {
                for name in &nonlocal.names {
                    names.insert(name.id.as_str().to_owned());
                }
            }
            Stmt::If(if_stmt) => {
                collect_nonlocal_names(&if_stmt.body, names);
                for clause in &if_stmt.elif_else_clauses {
                    collect_nonlocal_names(&clause.body, names);
                }
            }
            _ => {}
        }
    }
}

/// Walks `body`'s own top-level statements (plus one level of `if` arms)
/// evaluating each against `environment` IN PLACE — the same restricted
/// forms `interpret_body` reads, but this walk's OWN job is recording
/// `effects`, not answering a return value, so it never declines: a
/// statement shape it does not specifically recognize is simply skipped
/// (its own value-producing behavior is already accounted for by
/// `call_result_with_enclosing`'s own separate, complete interpretation;
/// this second pass only needs to notice WRITES that escape the callee's
/// own local scope). `declared` name resolution is not this function's
/// job — every effect is reported as a plain value, judged by the
/// CALLER against ITS OWN declared table, exactly as `bind_checked` in
/// `loops.rs` judges a loop body's declared-slot writes.
fn collect_call_effects(
    body: &[Stmt],
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
    nonlocal_names: &std::collections::HashSet<String>,
    locally_bound: &std::collections::HashSet<String>,
    effects: &mut Vec<(String, AbstractValue)>,
) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                let [target] = assign.targets.as_slice() else {
                    continue;
                };
                record_write_effect(target, assign.value.as_ref(), kernel, environment, nonlocal_names, locally_bound, effects);
            }
            Stmt::AugAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    if nonlocal_names.contains(name.id.as_str()) {
                        let current = environment.read(name.id.as_str()).cloned().unwrap_or_else(unknown);
                        let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
                        let updated = binary_arithmetic_value(assign.op, &current, &operand);
                        environment.bind(name.id.as_str(), updated.clone());
                        effects.push((name.id.as_str().to_owned(), updated));
                    }
                }
            }
            Stmt::If(if_stmt) => {
                let test_value = evaluate_expression(if_stmt.test.as_ref(), environment, kernel);
                let (truthy, known) = truthiness(&test_value);
                if known {
                    let body = if truthy {
                        Some(if_stmt.body.as_slice())
                    } else {
                        if_stmt
                            .elif_else_clauses
                            .iter()
                            .find(|clause| clause.test.is_none())
                            .map(|clause| clause.body.as_slice())
                    };
                    if let Some(body) = body {
                        collect_call_effects(body, kernel, environment, nonlocal_names, locally_bound, effects);
                    }
                    continue;
                }
                // an undecidable test: both arms may have run under real
                // execution, so both are scanned for effects (on a shared
                // fork each, never rejoined — this function reports every
                // POSSIBLE effect, and the caller's own judging handles an
                // over-approximated value the same honest way a loop's
                // Undetermined-declines-the-whole-run posture does not
                // need to apply here, since an effect is additive
                // information, not a replacement for the value answer).
                let mut arm_environment = environment.fork();
                collect_call_effects(&if_stmt.body, kernel, &mut arm_environment, nonlocal_names, locally_bound, effects);
                for clause in &if_stmt.elif_else_clauses {
                    let mut clause_environment = environment.fork();
                    collect_call_effects(&clause.body, kernel, &mut clause_environment, nonlocal_names, locally_bound, effects);
                }
            }
            _ => {}
        }
    }
}

/// One `Assign` target's own effect, when it is a shape this channel
/// tracks: a bare `nonlocal` name, or a subscript/attribute store whose
/// BASE is a free name (neither a parameter nor a name this body's own
/// statements bind). Every other target shape (a locally-bound plain
/// name, a tuple/list unpack, a store through a non-Name base) records
/// no effect — that write is either purely local (already answered by
/// `call_result_with_enclosing`'s own value) or outside this channel's
/// read shapes.
fn record_write_effect(
    target: &Expr,
    value_expr: &Expr,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
    nonlocal_names: &std::collections::HashSet<String>,
    locally_bound: &std::collections::HashSet<String>,
    effects: &mut Vec<(String, AbstractValue)>,
) {
    match target {
        Expr::Name(name) if nonlocal_names.contains(name.id.as_str()) => {
            let value = evaluate_expression(value_expr, environment, kernel);
            environment.bind(name.id.as_str(), value.clone());
            effects.push((name.id.as_str().to_owned(), value));
        }
        Expr::Subscript(subscript) => {
            let Expr::Name(base) = subscript.value.as_ref() else {
                return;
            };
            if locally_bound.contains(base.id.as_str()) {
                return;
            }
            let value = evaluate_expression(value_expr, environment, kernel);
            let Some(receiver) = environment.read(base.id.as_str()).cloned() else {
                effects.push((base.id.as_str().to_owned(), unknown()));
                return;
            };
            let key = evaluate_expression(subscript.slice.as_ref(), environment, kernel);
            let composed = match receiver.kind {
                Kind::Object => dict_with_item(&receiver, &key, &value),
                Kind::List => list_with_item(&receiver, &key, &value),
                _ => None,
            };
            let new_receiver = composed.unwrap_or_else(unknown);
            environment.bind(base.id.as_str(), new_receiver.clone());
            effects.push((base.id.as_str().to_owned(), new_receiver));
        }
        Expr::Attribute(attribute) => {
            let Expr::Name(base) = attribute.value.as_ref() else {
                return;
            };
            if locally_bound.contains(base.id.as_str()) {
                return;
            }
            let value = evaluate_expression(value_expr, environment, kernel);
            let Some(receiver) = environment.read(base.id.as_str()).cloned() else {
                effects.push((base.id.as_str().to_owned(), unknown()));
                return;
            };
            let new_receiver = field_write(&receiver, attribute.attr.as_str(), value).unwrap_or_else(unknown);
            environment.bind(base.id.as_str(), new_receiver.clone());
            effects.push((base.id.as_str().to_owned(), new_receiver));
        }
        _ => {}
    }
}

/// The SORT SET a same-module call's return annotation states, for a
/// caller that explicitly wants a coarse "some value of this sort"
/// CLAIM rather than the call's own (possibly declined) VALUE — never
/// called from `call_result`/`call_result_with_enclosing`'s own decline
/// points (both answer `None` outright on a genuine decline; see that
/// function's own doc). The one caller today is `evaluate_fstring`'s
/// PATTERN tier: an f-string interpolation only ever COMPOSES this set
/// into a concatenated pattern (never checks it for exact containment
/// against a narrow declared sink), so a fabricated sort-only claim is
/// safe there in a way it is NOT safe as an ordinary call's return value
/// — flowing this set into `assignability.rs`'s CONTAINMENT-REFUTATION
/// law as if it were a checkable fact FIRES the checker's own admission
/// of ignorance against a narrow sink on an otherwise IN-SET call
/// (item 1's own regression: e-class-and-function.py's
/// `first_age(40, 41)`, i-more-expressions.py's
/// `rest_identifier_parameter(40, 41)`, and others — see
/// `call_result_with_enclosing`'s own doc for the full list). This is
/// why the fallback is no longer wired into that function's decline
/// points and is instead exposed here as its own named capability, for
/// `evaluate_fstring` to call directly on a bare same-module call whose
/// ordinary `evaluate_expression` reading already came back `unknown()`.
///
/// NOT reached by `a-statements.py`'s own `def unread_number() -> int:
/// ...`: an ellipsis-only body is NOT a decline in `interpret_body` — a
/// bare `...` is an ordinary `Stmt::Expr` (evaluated and discarded, like
/// `pass`), so the body falls off the end and contributes `null_value()`
/// instead, matching CPython itself (execution-verified: `def f() -> int:
/// ...` really returns `None` at runtime). That call already answers
/// `Kind::Null`, a DIFFERENT existing law's business (`assignability.rs`'s
/// Null-vs-scalar-ground fire) — `evaluate_fstring` only ever retries
/// THIS fallback when the plain reading answered `Kind::Unknown`, so an
/// ellipsis-bodied call's own `Kind::Null` answer never reaches it either.
/// Recognizes only a BARE `int`/`float`/`str` return annotation — the
/// same three base-sort names `surface.rs::annotated_expression_set`
/// matches on an `Annotated[...]` base (that function's own `Expr::Name`
/// arms), reused here by the identical convention rather than re-deriving
/// a different one. `int` answers the whole-number SET (every integer,
/// unbounded — `whole_integers()` below, the same "R-bar itself, no
/// range narrows it" shape `float_sorted_unknown` builds for the float
/// case, but Integer-tagged instead of Float-tagged) rather than one
/// exact value: CPython's own runtime enforces NOTHING about a return
/// annotation (`tmp/cpython/Doc/reference/compound_stmts.rst`'s `funcdef`
/// grammar states `-> expression` as a syntactic annotation only), so
/// this is a language/library-level CLAIM about the def's own contract —
/// graded `TrustSpec` for that reason, matching `float_sorted_unknown`'s
/// identical grading rationale for the `math` family. `float` answers
/// `float_sorted_unknown()` directly. `str` answers the whole-strings set
/// (`codepoint_sets::strings()`, `C*`) at the same Spec grade. Any other
/// return annotation shape (a compiled alias name, `None`, a missing
/// annotation, a `dict[...]`/`list[...]` subscript, …) declines — this
/// fallback states nothing beyond the three base sorts a bare name can
/// spell.
pub fn return_sort_fallback(def: &StmtFunctionDef) -> Option<AbstractValue> {
    let Expr::Name(sort) = def.returns.as_deref()? else {
        return None;
    };
    match sort.id.as_str() {
        "int" => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(whole_integers(), None, TrustSpec, SetKindTag::None)
        }),
        "float" => Some(float_sorted_unknown()),
        "str" => Some(known_set(strings(), None, TrustSpec, SetKindTag::None)),
        _ => None,
    }
}

/// R-bar (`refinement_forms::numbers()`'s own unbounded ray) conjoined
/// with the `int` form — the unbounded whole-number set: every integer,
/// no ceiling/floor. The same shape `surface.rs::annotated_expression_set`
/// builds for a bare `Annotated[int, Field(…)]` with no `ge`/`le` kwarg
/// (`vec![integer()]`, which the kernel already reads as "integer, no
/// other bound" — `numbers()`'s own `at_least(NEG_INFINITY)` form states
/// the identical "unbounded" fact explicitly, so conjoining it changes
/// nothing about which values the set admits and only makes the
/// unbounded-ness textually visible here, mirroring `float_sorted_unknown`'s
/// own `numbers()` base).
fn whole_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

/// The ELEMENT sort a same-module GENERATOR/STREAM def's return
/// annotation states, once the body's own straight-line interpretation
/// GENUINELY declines it — a-statements.py's own `stream() ->
/// AsyncIterator[int]: raise NotImplementedError; yield 0` (the `yield`
/// after the `raise` marks this def as an async generator syntactically,
/// datamodel.rst's generator-iterator protocol, but is never reached at
/// runtime; `interpret_body` has no `Stmt::Raise` row, so calling it on
/// this body already answers `None`, the same genuine-decline `loops.rs`'s
/// own for-loop reader hits). Unlike a same-module call's own declined
/// return value (`call_result`/`call_result_with_enclosing`, which answer
/// `None` outright on a genuine decline — a fabricated sort-only claim is
/// never safe to check for exact containment against a narrow sink, since
/// the checker never actually read the body it would be claiming a fact
/// about), a `for`/`async for` loop's own ITERATION count is bounded
/// separately by `loops.rs`'s own cap machinery, so stating the element's
/// bare SORT here (never a value) is a fact the loop reader can use
/// without that same soundness hazard — see `loops.rs` for how the
/// element sort composes with the loop's own iteration bound.
///
/// Recognizes `AsyncIterator[T]` / `Iterator[T]` / `Iterable[T]` — a
/// `Subscript` whose HEAD is one of those three bare names (no import-
/// identity check — this table has no `typing.AsyncIterator`/`Iterator`/
/// `Iterable` import identity to check against, matching `Optional`/
/// `Literal`'s own no-identity reading in `typereading.rs`) — and `T` is
/// itself one of three base-sort names (`int` → the unbounded whole-number
/// set, Integer-tagged; `float` → `float_sorted_unknown()`; `str` → the
/// whole-strings set — the same three base-sort names
/// `surface.rs::annotated_expression_set` matches on an `Annotated[...]`
/// base). Any other subscript head, a `T` that is not one of the three
/// base sorts, or a non-`Subscript` annotation (a missing annotation, a
/// bare name, `None`) declines — this fallback states nothing beyond the
/// three base sorts one level down.
pub fn iterable_element_sort(def: &StmtFunctionDef) -> Option<AbstractValue> {
    let Expr::Subscript(subscript) = def.returns.as_deref()? else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !matches!(head.id.as_str(), "AsyncIterator" | "Iterator" | "Iterable") {
        return None;
    }
    let Expr::Name(element_sort) = subscript.slice.as_ref() else {
        return None;
    };
    match element_sort.id.as_str() {
        "int" => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(whole_integers(), None, TrustSpec, SetKindTag::None)
        }),
        "float" => Some(float_sorted_unknown()),
        "str" => Some(known_set(strings(), None, TrustSpec, SetKindTag::None)),
        _ => None,
    }
}

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
fn first_non_docstring_statement(body: &[Stmt]) -> Option<&Stmt> {
    body.iter().find(|stmt| !is_bare_string_literal_statement(stmt))
}

/// Whether `stmt` is a bare string-literal expression statement — the
/// docstring shape `first_non_docstring_statement` skips.
fn is_bare_string_literal_statement(stmt: &Stmt) -> bool {
    matches!(stmt, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::StringLiteral(_)))
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
fn seed_free_variables(def: &StmtFunctionDef, enclosing: &Environment, into: &mut Environment) {
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
    if let Some(vararg) = def.parameters.vararg.as_ref() {
        locally_bound.insert(vararg.name.id.as_str().to_owned());
    }
    if let Some(kwarg) = def.parameters.kwarg.as_ref() {
        locally_bound.insert(kwarg.name.id.as_str().to_owned());
    }
    collect_bound_names(&def.body, &mut locally_bound);
    for free_name in free_names_read(&def.body, &locally_bound) {
        if let Some(value) = enclosing.read(&free_name) {
            into.bind(&free_name, value.clone());
        }
    }
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
fn free_names_read(body: &[Stmt], locally_bound: &std::collections::HashSet<String>) -> Vec<String> {
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
/// CHANNEL, `call_effects`'s own doc). A bare `Expr::Name` target is NOT
/// walked here — that shape is a LOCAL bind (`collect_bound_names`'s own
/// job), never a free read of the pre-existing value. The subscript's own
/// KEY expression (`"age"`) is also walked, on the chance it is itself a
/// free name (`outlaw[key] = 200` where `key` is a captured local) —
/// walked through the ordinary `collect_names_in_expr`, since a key
/// expression is always a READ, never a target.
fn collect_write_target_base_name(target: &Expr, locally_bound: &std::collections::HashSet<String>, names: &mut Vec<String>) {
    match target {
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
fn fresh_body_environment(def: &StmtFunctionDef, table: Option<&Arc<FunctionTable>>, depth: u32) -> Environment {
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
fn bind_parameters(
    def: &StmtFunctionDef,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
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
    let default_environment = Environment::new(std::collections::HashSet::new());
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
        let tail_value = crate::refinedpy::collection_models::tuple_literal_value(&tail);
        environment.bind(vararg.name.id.as_str(), tail_value);
    }
    Some(())
}

/// A `super().<method>(<args>)` call recognized inside a RETURN
/// expression: the method name, the argument VALUES (already evaluated
/// against the interpreting body's own environment), and the CURRENT
/// environment (so the resolver reads `self`'s WORKING value — any
/// earlier `self.<field> = ...` statement in the same method body
/// already updated it — rather than a value captured once at method
/// entry) — answers the call's return value, or `None` when it is not
/// a super call this resolver's owner (`instances::method_call_result`)
/// can answer. Threaded through
/// `interpret_body`/`interpret_if`/`interpret_undecided_arms` so a
/// plain `call_result` (which never sets one) keeps declining any body
/// with a `super()` call exactly as before — only a method
/// interpretation supplies a resolver.
pub(crate) type SuperResolver<'a> = dyn Fn(&str, &[AbstractValue], &Environment) -> Option<AbstractValue> + 'a;

/// Interprets `body`'s statements in order against `environment`,
/// restricted forms only. Returns `Some(true)` when control can fall
/// off the end of `body` (so the caller should contribute a
/// `null_value()` return), `Some(false)` when every path through `body`
/// ends in a recorded `Return`, and `None` the moment a statement
/// outside the restricted forms is met — the whole call declines then,
/// matching `loops.rs::run_restricted_body`'s all-or-nothing posture.
///
/// `super_resolver` is `Some` only when `instances::method_call_result`
/// is interpreting a method body; a bare `call_result` passes `None`
/// and a `super()` call inside it still declines exactly as before this
/// wave (`Stmt::Return`'s own `evaluate_expression` fallback has no
/// model for a `super()` receiver, matching `evaluate_call`'s own
/// unknown() answer for any callee shape it does not recognize).
pub(crate) fn interpret_body(
    body: &[Stmt],
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    environment: &mut Environment,
    returns: &mut Vec<AbstractValue>,
    super_resolver: Option<&SuperResolver>,
) -> Option<bool> {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => interpret_assign(assign, kernel, environment)?,
            Stmt::AnnAssign(assign) => interpret_ann_assign(assign, kernel, environment)?,
            Stmt::AugAssign(assign) => interpret_aug_assign(assign, kernel, environment)?,
            Stmt::Pass(_) => {}
            Stmt::Expr(expr_stmt) => {
                evaluate_expression(expr_stmt.value.as_ref(), environment, kernel);
            }
            Stmt::If(if_stmt) => {
                let falls_through = interpret_if(if_stmt, kernel, depth, environment, returns, super_resolver)?;
                if !falls_through {
                    return Some(false);
                }
            }
            Stmt::Return(ret) => {
                let value = match ret.value.as_deref() {
                    Some(value_expr) => {
                        evaluate_return_value(value_expr, environment, kernel, super_resolver)?
                    }
                    None => null_value(),
                };
                if value.kind == Kind::Unknown {
                    return None;
                }
                returns.push(value);
                return Some(false);
            }
            Stmt::ClassDef(def) => interpret_class_def(def, kernel, environment)?,
            // `nonlocal <name>[, ...]` — a DECLARATION, not a value-producing
            // or value-binding statement on its own (simple_stmts.rst, "The
            // `nonlocal` statement": it only "causes the listed identifiers
            // to refer to previously bound variables in the nearest
            // enclosing scope"). This interpreter tracks no scope chain of
            // its own (`Environment` is one flat map, `call_result_with_
            // enclosing`'s own doc), so the declaration itself is a no-op
            // here, exactly like `Stmt::Pass` — it neither reads nor writes
            // a value. Recognizing it is what lets a body OPENING with
            // `nonlocal age` (a-statements.py's own `nonlocal_rebind`/
            // `spoil`) reach its own `age = 200` statement at all: before
            // this arm, `nonlocal age` alone hit the catch-all `_ => return
            // None` and declined the WHOLE call before the write it
            // introduces was ever interpreted. `call_effects` (this file's
            // own CALLEE-EFFECTS CHANNEL) is the ONE place a `nonlocal`
            // declaration's own outward-write MEANING is read
            // (`collect_nonlocal_names`) — this interpreter's job stops at
            // "not declining," never reporting the effect itself, matching
            // `call_result`/`call_result_with_enclosing`'s own doc: "A WRITE
            // to an enclosing name from inside the callee... is not
            // modeled" by this path.
            Stmt::Nonlocal(_) => {}
            _ => return None,
        }
    }
    Some(true)
}

/// A `return <expr>` value, with ONE extra recognized shape a plain
/// `evaluate_expression` cannot answer: a bare `super().<method>(...)`
/// call, or that call as one operand of a `BinOp` (`super().years() +
/// 1`, the corpus's own `call_super_method` shape) — both routed
/// through `super_resolver` for the call's own answer, then combined
/// through `binary_arithmetic_value` the same way any other BinOp
/// would be. `None` when `super_resolver` is absent (a plain
/// `call_result`, which has no model for a `super()` receiver at all)
/// and the expression names one, OR when the resolver itself declines.
/// Every other expression shape evaluates exactly as before, through
/// the ordinary dispatcher.
fn evaluate_return_value(
    value_expr: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    super_resolver: Option<&SuperResolver>,
) -> Option<AbstractValue> {
    if let Some(resolver) = super_resolver {
        if let Some(value) = try_super_call(value_expr, environment, kernel, resolver) {
            return Some(value);
        }
        if let Expr::BinOp(binop) = value_expr {
            if let Some(left) = try_super_call(binop.left.as_ref(), environment, kernel, resolver) {
                let right = evaluate_expression(binop.right.as_ref(), environment, kernel);
                return Some(binary_arithmetic_value(binop.op, &left, &right));
            }
            if let Some(right) = try_super_call(binop.right.as_ref(), environment, kernel, resolver) {
                let left = evaluate_expression(binop.left.as_ref(), environment, kernel);
                return Some(binary_arithmetic_value(binop.op, &left, &right));
            }
        }
    }
    Some(evaluate_expression(value_expr, environment, kernel))
}

/// `super().<method>(<args>)` recognized syntactically — an `Expr::Call`
/// whose `func` is `Attribute { value: a bare, no-argument `Call` to
/// the name `super`, attr: <method> }`, the same shape
/// `instances::super_init_call` recognizes for `super().__init__(...)`
/// (`tmp/cpython/Doc/library/functions.rst`'s `super()` entry cited
/// there). `None` when `expr` is not that shape, OR when any argument
/// is starred/keyword (this resolver's own positional-only contract).
fn try_super_call(
    expr: &Expr,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
    resolver: &SuperResolver,
) -> Option<AbstractValue> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Call(super_call) = attribute.value.as_ref() else {
        return None;
    };
    let Expr::Name(super_name) = super_call.func.as_ref() else {
        return None;
    };
    if super_name.id.as_str() != "super" || !super_call.arguments.args.is_empty() {
        return None;
    }
    if !call.arguments.keywords.is_empty() || call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
        return None;
    }
    let arguments: Vec<AbstractValue> = call
        .arguments
        .args
        .iter()
        .map(|arg| evaluate_expression(arg, environment, kernel))
        .collect();
    resolver(attribute.attr.as_str(), &arguments, environment)
}

/// A `class` statement inside a summarized body — a-statements.py's own
/// `device()`/`with_statement` shape: `device()`'s body declares a local
/// class `_Device`, then `return _Device()` constructs it. Without this
/// row, `Stmt::ClassDef` fell to `interpret_body`'s catch-all `_ => return
/// None`, declining `device()`'s whole call — `evaluate_call`'s own
/// construction arm only ever finds a class by reading
/// `environment.classes()` (`expressions.rs`'s module doc, dispatch order
/// (b)), and a `call_result`-built environment never carried one before
/// this row (`fresh_body_environment` only ever calls `set_functions`).
///
/// Builds `def`'s own `ClassModel` the same way `check.rs`'s
/// `local_class_table` builds a body-local class: `def` alone, wrapped in
/// a synthetic single-class `ModModule`, through
/// `instances::class_table`'s one public constructor — the exact
/// construction the mission names ("the same synthetic-module pattern
/// check.rs's local_class_table uses"). `aliases`/`imports` are read
/// EMPTY here (`summaries::call_result` carries neither the module's
/// alias table nor its import identities — only `WalkContext`, built in
/// `check.rs`, has them), so a field annotated with a module-level `type
/// Age = …` alias or a pydantic `Annotated[...]` form reads as
/// undeclared (`declared: None`) inside a same-module-call-summarized
/// class — narrower than `check.rs`'s own body-local reading, never
/// wrong: an undeclared field write raises no fire, it simply carries the
/// value through unjudged, which is what this row's own fixture rows
/// need (`_Device.value: int` — a bare `int` annotation reads through
/// the alias table too, `typereading::declared_refinement`'s `Expr::Name`
/// arm, and is UNDECLARED there regardless of whether the table is
/// populated, since `int`/`str`/`float` are base sorts, never alias
/// entries).
///
/// Inserted into `environment`'s own class table via `Environment::
/// set_classes`, merged over whatever the environment already carries
/// (a caller's own classes, when `call_result_with_enclosing`'s future
/// callers seed one) so a LATER class in the same body naming an
/// EARLIER one as its base — out of this wave's fixture rows, but not
/// precluded — still finds it. Always succeeds (`Some(())`): a
/// `ClassDef` statement itself never fails to interpret, whatever its
/// body contains — the class's own construction/field rules are judged
/// later, at each construction/field-write SITE, not here.
fn interpret_class_def(def: &StmtClassDef, kernel: &Arc<RefinedTSKernel>, environment: &mut Environment) -> Option<()> {
    let synthetic = ModModule {
        node_index: AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: vec![Stmt::ClassDef(def.clone())].into(),
    };
    let empty_aliases = std::collections::HashMap::new();
    let empty_imports = surface_imports(&ModModule {
        node_index: AtomicNodeIndex::NONE,
        range: TextRange::default(),
        body: Vec::new().into(),
    });
    let local_classes = class_table(&synthetic, &empty_aliases, &empty_imports, kernel);
    let mut merged_classes: std::collections::HashMap<String, ClassModel> = match environment.classes() {
        Some(existing) => (**existing).clone(),
        None => std::collections::HashMap::new(),
    };
    for (name, model) in local_classes {
        merged_classes.insert(name, model);
    }
    environment.set_classes(Arc::new(merged_classes));
    Some(())
}

fn interpret_assign(assign: &StmtAssign, kernel: &Arc<RefinedTSKernel>, environment: &mut Environment) -> Option<()> {
    let [target] = assign.targets.as_slice() else {
        return None;
    };
    if let Expr::Name(name) = target {
        let value = evaluate_expression(assign.value.as_ref(), environment, kernel);
        environment.bind(name.id.as_str(), value);
        return Some(());
    }
    if let Expr::Subscript(subscript) = target {
        if let Some(()) = write_subscript_target(subscript, assign.value.as_ref(), kernel, environment) {
            return Some(());
        }
    }
    if matches!(target, Expr::Tuple(_) | Expr::List(_)) {
        let value = evaluate_expression(assign.value.as_ref(), environment, kernel);
        return bind_unpack_target(target, &value, environment);
    }
    // `self.<field> = <expr>` — a method body's own field write, live
    // only when `self` is bound to a known instance (an ordinary
    // function body has no such binding, so this arm is a no-op outside
    // `method_call_result`'s own environment setup).
    write_self_field(target, assign.value.as_ref(), kernel, environment)
}

/// `(a, b, ...) = value` / `[a, b, ...] = value` inside a restricted
/// body — e-class-and-function.py's own `unpack_first`: `a, _b = ages`
/// where `ages` is the def's own tuple-typed PARAMETER (`ages: tuple[int,
/// int]`), a known `Kind::List` value bound at call time. No starred
/// element (`a, *rest = value` is out of this restricted interpreter's
/// scope — the mission names no fixture row needing it here, and
/// `check.rs::bind_known_sequence_target` already owns that shape for the
/// ordinary walk); every target must be a bare `Expr::Name` (a nested
/// tuple/list sub-target is also out of scope, same reasoning). `None`
/// (the whole call declines) when `value` is not a known `Kind::List`,
/// the element COUNT does not match the target list's own length exactly
/// (CPython's own `ValueError` — this restricted interpreter has no
/// finding sink to report it through, so a mismatch is an honest decline
/// rather than a silently-wrong bind), or any target is not a bare name.
fn bind_unpack_target(target: &Expr, value: &AbstractValue, environment: &mut Environment) -> Option<()> {
    let elements: &[Expr] = match target {
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::List(list) => &list.elts,
        _ => return None,
    };
    if value.kind != Kind::List || elements.len() != value.items.len() {
        return None;
    }
    for (element, item) in elements.iter().zip(value.items.iter()) {
        let Expr::Name(name) = element else {
            return None;
        };
        environment.bind(name.id.as_str(), item.clone());
    }
    Some(())
}

/// `name[key] = value` inside a restricted body — the CAPTURED-RECEIVER
/// mutation shape a-statements.py's `spoil` closure builds
/// (`outlaw["age"] = 200`, a free name `outlaw` read from the enclosing
/// scope through `call_effects`'s own seeding). `name` must already be
/// bound to a known receiver (a dict or list — the module-level
/// `collection_models::dict_with_item`/`list_with_item` mutation
/// contract, the same one `loops.rs::run_subscript_assign_once` uses for
/// the identical shape inside a loop body); the written-through receiver
/// rebinds `name` in place. `None` for anything the contract does not
/// resolve — an unbound name, a receiver kind neither function owns, or
/// a key/value shape the contract declines — leaving the caller's own
/// `write_self_field` fallback to answer whether this is instead a
/// `self.<field>` write (a `Subscript` target is never that shape, so
/// the fallback simply also answers `None`, and the whole statement
/// declines, unchanged from before this function existed).
fn write_subscript_target(
    subscript: &ruff_python_ast::ExprSubscript,
    value_expr: &Expr,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    let Expr::Name(name) = subscript.value.as_ref() else {
        return None;
    };
    let receiver = environment.read(name.id.as_str())?.clone();
    let key = evaluate_expression(subscript.slice.as_ref(), environment, kernel);
    let value = evaluate_expression(value_expr, environment, kernel);
    let new_receiver = match receiver.kind {
        Kind::Object => crate::refinedpy::collection_models::dict_with_item(&receiver, &key, &value)?,
        Kind::List => crate::refinedpy::collection_models::list_with_item(&receiver, &key, &value)?,
        _ => return None,
    };
    environment.bind(name.id.as_str(), new_receiver);
    Some(())
}

/// `self.<field> = <expr>` shared by both a plain `Assign` and an
/// `AugAssign`'s pre-computed RHS value: resolves the field name,
/// evaluates `value_expr` against `environment` (the CALLER already
/// substitutes the augmented value when this is an `AugAssign`),
/// updates the WORKING instance through `instances::field_write`, and
/// rebinds `self` in `environment` to the updated instance so a later
/// `self.<field>` read in the same body sees the write. Declines
/// (`None`) when the target is not `self.<field>`, or `self` is not
/// bound to a known `Kind::Object` — the same all-or-nothing posture
/// every other restricted form takes.
fn write_self_field(
    target: &Expr,
    value_expr: &Expr,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    let field = self_attribute_name(target)?;
    let instance = environment.read("self")?.clone();
    let value = evaluate_expression(value_expr, environment, kernel);
    let updated = field_write(&instance, &field, value)?;
    environment.bind("self", updated);
    Some(())
}

fn interpret_ann_assign(
    assign: &StmtAnnAssign,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    let Expr::Name(name) = assign.target.as_ref() else {
        return None;
    };
    let Some(value_expr) = assign.value.as_deref() else {
        // a value-less `x: T` declares nothing to bind — CPython
        // evaluates the annotation but never assigns the name
        // (simple_stmts.rst, "Annotated assignment statements")
        return Some(());
    };
    let value = evaluate_expression(value_expr, environment, kernel);
    environment.bind(name.id.as_str(), value);
    Some(())
}

fn interpret_aug_assign(
    assign: &StmtAugAssign,
    kernel: &Arc<RefinedTSKernel>,
    environment: &mut Environment,
) -> Option<()> {
    if let Expr::Name(name) = assign.target.as_ref() {
        let current = environment.read(name.id.as_str()).cloned().unwrap_or_else(unknown);
        let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
        let updated = binary_arithmetic_value(assign.op, &current, &operand);
        environment.bind(name.id.as_str(), updated);
        return Some(());
    }
    // `self.<field> += <expr>` — read the field's CURRENT value off the
    // working instance, combine it with the operand, then write the
    // result back the same way a plain `self.<field> = ...` does.
    let field = self_attribute_name(assign.target.as_ref())?;
    let instance = environment.read("self")?.clone();
    let current = field_read(&instance, &field).unwrap_or_else(unknown);
    let operand = evaluate_expression(assign.value.as_ref(), environment, kernel);
    let updated_value = binary_arithmetic_value(assign.op, &current, &operand);
    let updated_instance = field_write(&instance, &field, updated_value)?;
    environment.bind("self", updated_instance);
    Some(())
}

/// `if test: body [elif ...] [else: body]` inside a summarized call
/// body. A definitely-true/false test interprets only the live arm on
/// the SAME environment (no fork needed — only one arm's writes ever
/// happen). An undecidable test interprets BOTH arms on forked
/// environments and rejoins the surviving ones through
/// `Environment::join`, mirroring `check.rs::walk_if`/`arm_terminates`:
/// an arm ending in `Return` contributes its value(s) to `returns` but
/// does not rejoin, since its fall-through state is unreachable.
/// Returns `Some(true)` if the post-if point is reachable (so the
/// caller keeps interpreting later statements), `Some(false)` if every
/// live arm returned, `None` if any visited arm is outside the
/// restricted forms.
fn interpret_if(
    if_stmt: &StmtIf,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    environment: &mut Environment,
    returns: &mut Vec<AbstractValue>,
    super_resolver: Option<&SuperResolver>,
) -> Option<bool> {
    let mut arms: Vec<(Option<&Expr>, &[Stmt])> = Vec::new();
    arms.push((Some(if_stmt.test.as_ref()), if_stmt.body.as_slice()));
    for clause in &if_stmt.elif_else_clauses {
        arms.push((clause.test.as_ref(), clause.body.as_slice()));
    }

    // a definite verdict short-circuits to the one live arm, evaluated
    // in place — walrus/side effects on the test itself are read once,
    // through the caller's own environment
    for (test, body) in &arms {
        if let Some(test_expr) = test {
            let test_value = evaluate_expression(test_expr, environment, kernel);
            let (truthy, known) = truthiness(&test_value);
            if known {
                if truthy {
                    return interpret_body(body, kernel, depth, environment, returns, super_resolver);
                }
                continue;
            }
            // the FIRST undecidable test is where both-arms interpretation
            // starts — every arm from here on (including any later elif)
            // is undetermined territory, handled below
            return interpret_undecided_arms(&arms, kernel, depth, environment, returns, super_resolver);
        }
        // a bare `else`/catch-all arm reached with every earlier test
        // known false: this is the one live arm
        return interpret_body(body, kernel, depth, environment, returns, super_resolver);
    }
    // every test was known false and there was no catch-all arm: the
    // whole `if` falls through untouched
    Some(true)
}

/// Interprets every arm on its own fork once a test could not be
/// decided — used from the first undecidable test onward, since a
/// later arm's own reachability itself depends on the undecided one.
fn interpret_undecided_arms(
    arms: &[(Option<&Expr>, &[Stmt])],
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
    environment: &mut Environment,
    returns: &mut Vec<AbstractValue>,
    super_resolver: Option<&SuperResolver>,
) -> Option<bool> {
    let mut surviving: Vec<Environment> = Vec::new();
    let mut has_catch_all = false;
    for (test, body) in arms {
        has_catch_all = has_catch_all || test.is_none();
        let mut arm_environment = environment.fork();
        let falls_through = interpret_body(body, kernel, depth, &mut arm_environment, returns, super_resolver)?;
        if falls_through {
            surviving.push(arm_environment);
        }
    }
    if !has_catch_all {
        surviving.push(environment.fork());
    }

    *environment = match surviving.len() {
        0 => return Some(false),
        1 => surviving.into_iter().next().unwrap(),
        _ => {
            let mut joined = surviving.remove(0);
            for arm in surviving {
                joined = Environment::join(joined, &arm);
            }
            joined
        }
    };
    Some(true)
}

/// Every bare name this body's own statements bind — `Assign`/
/// `AnnAssign`/`AugAssign` targets (including a tuple/list UNPACK
/// target's own leaf names, `interpret_assign`'s own `bind_unpack_target`
/// row — e-class-and-function.py's `unpack_first`'s `a, _b = ages`) and
/// `if`/`elif`/`else` bodies, recursively. A restricted body never
/// contains anything else that binds a name (no `for`/`with`/`import`/
/// nested `def`), so this collector only walks the forms `interpret_body`
/// itself recognizes.
pub(crate) fn collect_bound_names(body: &[Stmt], bound: &mut std::collections::HashSet<String>) {
    for stmt in body {
        match stmt {
            Stmt::Assign(assign) => {
                for target in &assign.targets {
                    collect_unpack_target_names(target, bound);
                }
            }
            Stmt::AnnAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    bound.insert(name.id.as_str().to_owned());
                }
            }
            Stmt::AugAssign(assign) => {
                if let Expr::Name(name) = assign.target.as_ref() {
                    bound.insert(name.id.as_str().to_owned());
                }
            }
            Stmt::If(if_stmt) => {
                collect_bound_names(&if_stmt.body, bound);
                for clause in &if_stmt.elif_else_clauses {
                    collect_bound_names(&clause.body, bound);
                }
            }
            _ => {}
        }
    }
}

/// One `Assign` target's own bound leaf names: a bare `Expr::Name` binds
/// itself; a `Tuple`/`List` UNPACK target recurses over its own elements
/// (`bind_unpack_target`'s identical shape — every element there is
/// itself required to be a bare name, so this walk never needs to go
/// deeper than one level, but recurses anyway for the same honest-over-
/// approximation reason `check.rs::forget_target_from_provably_unbound`
/// recurses on its own tuple/list targets). Every other target shape (a
/// `Subscript`/`Attribute` write, out of `collect_bound_names`'s own
/// scope — neither is a NAME binding) contributes nothing.
fn collect_unpack_target_names(target: &Expr, bound: &mut std::collections::HashSet<String>) {
    match target {
        Expr::Name(name) => {
            bound.insert(name.id.as_str().to_owned());
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                collect_unpack_target_names(element, bound);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                collect_unpack_target_names(element, bound);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use refined_domain::abstract_value::known_values;
    use refined_domain::abstract_value::PrimitiveKind;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_parser::parse_module;

    use super::*;

    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    /// Parses `source` as a module and returns its single top-level
    /// `def` (the function under test).
    fn parsed_def(source: &str) -> StmtFunctionDef {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        let stmt = module.body.into_iter().next().expect("one top-level statement");
        stmt.function_def_stmt().expect("top-level statement is a def")
    }

    fn known_int(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustProved)
    }

    #[test]
    fn straight_line_body_answers_the_returned_expression() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def double(x):\n    return x + x\n");
        let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0).expect("straight-line body answers");
        assert_eq!(result.values, vec![6.0]);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    #[test]
    fn a_trailing_default_parameter_is_evaluated_when_no_argument_covers_it() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def add(x, y=10):\n    return x + y\n");
        let result = call_result(&def, &[known_int(5.0)], None, &kernel, 0).expect("default parameter fills in");
        assert_eq!(result.values, vec![15.0]);
    }

    #[test]
    fn an_if_else_where_both_arms_return_known_values_joins_both_possibilities() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(
            "def pick(flag):\n    if flag:\n        return 3\n    else:\n        return 5\n",
        );
        let result =
            call_result(&def, &[unknown()], None, &kernel, 0).expect("both known-value arms join to an answer");
        // an undecidable flag interprets both arms; the join of 3 and 5
        // under one Integer tag is the two-value carrier
        // join_known's own test (test_join_known_like_sort_keeps_the_tag_mixed_sort_loses_it)
        // pins for two same-sort Values joins
        assert_eq!(result.kind, Kind::Values);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
        let mut values = result.values.clone();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(values, vec![3.0, 5.0]);
    }

    #[test]
    fn a_body_that_falls_off_the_end_contributes_null_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def maybe_none(flag):\n    if flag:\n        return 3\n    x = 1\n");
        let result = call_result(&def, &[known_int(1.0)], None, &kernel, 0)
            .expect("a known-true flag still interprets the fall-through arm's shape honestly");
        // flag is KNOWN true here, so only the `return 3` arm runs and the
        // fall-through never contributes — this pins the definite-branch
        // path specifically; the undecidable-flag fall-through case is
        // covered by the next test
        assert_eq!(result.values, vec![3.0]);
    }

    #[test]
    fn an_undecidable_flag_whose_false_arm_falls_off_the_end_joins_in_null() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def maybe_none(flag):\n    if flag:\n        return 3\n    x = 1\n");
        let result = call_result(&def, &[unknown()], None, &kernel, 0)
            .expect("an undecidable flag interprets both the return arm and the fall-through");
        // the true arm returns 3; the false arm falls off the end,
        // contributing null_value() — the join of an Integer with Null
        // is neither a bare Integer (Kind::Values) nor a bare Null
        assert_ne!(result.kind, Kind::Unknown);
        assert_ne!(result.kind, Kind::Values);
        assert_ne!(result.kind, Kind::Null);
    }

    #[test]
    fn a_body_with_a_while_loop_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def counted(n):\n    while n > 0:\n        n -= 1\n    return n\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
    }

    #[test]
    fn the_depth_cap_declines_before_interpreting_the_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def double(x):\n    return x + x\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, CALL_DEPTH_CAP).is_none());
    }

    #[test]
    fn a_return_with_an_unknown_value_declines_the_whole_call() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def opaque(x):\n    return f(x)\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
    }

    #[test]
    fn too_many_arguments_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def one_arg(x):\n    return x\n");
        assert!(call_result(&def, &[known_int(1.0), known_int(2.0)], None, &kernel, 0).is_none());
    }

    /// `*args` genuinely interprets — bound to the caller's own trailing
    /// arguments as a known tuple (`bind_parameters`'s own vararg row) —
    /// rather than declining outright. This body never reads `args` at
    /// all, so the call answers the literal `1` regardless of what
    /// arguments the caller passed.
    #[test]
    fn varargs_with_no_argument_reads_interprets_the_body() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def variadic(*args):\n    return 1\n");
        let result = call_result(&def, &[], None, &kernel, 0).expect("a *args parameter is no longer a decline");
        assert_eq!(result.values, vec![1.0]);
    }

    /// e-class-and-function.py's own `first_age` shape: `*ages: int`
    /// bound to the caller's own trailing arguments as a tuple, then
    /// `ages[0]` reads the first one through the ordinary subscript path
    /// — the regression this pins: `first_age(40, 41)` (an IN-SET call
    /// under `Age`) answers the exact value 40, never a coarse fallback
    /// set the containment law would wrongly fire against a narrow sink.
    #[test]
    fn varargs_binds_a_known_tuple_of_the_trailing_arguments() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def first_age(*ages):\n    return ages[0]\n");
        let result = call_result(&def, &[known_int(40.0), known_int(41.0)], None, &kernel, 0)
            .expect("*ages binds to the known (40, 41) tuple, and ages[0] reads through it");
        assert_eq!(result, known_int(40.0));
    }

    /// A def with both a plain parameter and a `*args` tail: the plain
    /// parameter takes the first argument, `*args` collects the rest.
    #[test]
    fn varargs_after_a_plain_parameter_collects_only_the_remaining_arguments() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def first_and_rest(first, *rest):\n    return rest[0]\n");
        let result = call_result(&def, &[known_int(1.0), known_int(2.0), known_int(3.0)], None, &kernel, 0)
            .expect("rest binds to the known (2, 3) tuple");
        assert_eq!(result, known_int(2.0));
    }

    // --- call_result_with_enclosing: closure reads ---

    /// `def read_age(): return age` nested inside a body that bound
    /// `age` — a-statements.py's own closure-read shape
    /// (`closure_mutates_flattened_capture`'s cousin, minus the write):
    /// `age` is free in `read_age`'s own body, so `call_result` alone
    /// (no enclosing environment) declines it as an unbound name read
    /// (`unknown()`, which `interpret_body`'s `Return` arm rejects);
    /// `call_result_with_enclosing` answers it once the call site's
    /// environment is threaded through.
    #[test]
    fn call_result_with_enclosing_reads_a_free_enclosing_local() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def read_age():\n    return age\n");

        let mut enclosing = Environment::new(std::collections::HashSet::from(["age".to_owned()]));
        enclosing.bind("age", known_int(40.0));

        assert!(
            call_result(&def, &[], None, &kernel, 0).is_none(),
            "with no enclosing environment, the free read of `age` stays unbound"
        );
        let result = call_result_with_enclosing(&def, &[], None, &kernel, 0, Some(&enclosing))
            .expect("the enclosing environment's `age` binding answers the free read");
        assert_eq!(result, known_int(40.0));
    }

    /// A name the callee body ITSELF binds (a parameter, or an
    /// assignment target) is never seeded from `enclosing`, even when
    /// `enclosing` happens to bind the same name — ordinary Python
    /// scoping (the body's own binding shadows the enclosing one for
    /// its whole extent, `executionmodel.rst`'s "Naming and binding").
    #[test]
    fn call_result_with_enclosing_does_not_shadow_a_locally_bound_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def shadow():\n    age = 10\n    return age\n");

        let mut enclosing = Environment::new(std::collections::HashSet::from(["age".to_owned()]));
        enclosing.bind("age", known_int(999.0));

        let result = call_result_with_enclosing(&def, &[], None, &kernel, 0, Some(&enclosing))
            .expect("the body's own local binding answers the read");
        assert_eq!(result, known_int(10.0), "the callee's own `age = 10` wins, never the enclosing 999");
    }

    // --- return_sort_fallback: declined-call sort fallback ---
    //
    // A body `interpret_body` genuinely declines (a `while` loop, `**kwargs`/
    // a keyword-only parameter, the depth cap, or an unbindable argument
    // list — a `*args` parameter is NO LONGER one of these, see the
    // `varargs_*` tests above) still states its return annotation's bare
    // SORT rather than declining outright to `None` — item 1's own
    // regression was never this fallback firing per se; it was the
    // vararg/tuple-unpack/isinstance-narrowed bodies genuinely declining
    // when they should have interpreted (or, for the vararg case,
    // genuinely bound a known tuple). `for_over_unread_iterable`
    // (a-statements.py) and `fstring_unread_substitution`
    // (b-body-expressions.py) both lean on this fallback reaching a real
    // sink and correctly FIRING there — see `loops.rs`'s own
    // `iterable_values` doc and `expressions.rs`'s own `evaluate_fstring`
    // doc for why a coarse sort-only claim is sound to flow all the way
    // to a sink in those two cases (the checker's own admitted-coarse
    // claim is what the row is testing, not a smuggled-in wrong answer).
    #[test]
    fn a_declined_while_loop_body_with_a_bare_int_return_annotation_answers_the_whole_number_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def counted(n) -> int:\n    while n > 0:\n        n -= 1\n    return n\n");
        let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0)
            .expect("the -> int annotation answers the whole-number set on a declined body");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// `-> float` reads through to the existing `float_sorted_unknown()`
    /// shape — the same Float-tagged all-numbers set `math.sqrt` answers.
    #[test]
    fn a_declined_while_loop_body_with_a_bare_float_return_annotation_answers_float_sorted_unknown() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def counted(n) -> float:\n    while n > 0:\n        n -= 1\n    return n\n");
        let result = call_result(&def, &[known_int(3.0)], None, &kernel, 0)
            .expect("the -> float annotation answers float_sorted_unknown on a declined body");
        assert_eq!(result, float_sorted_unknown());
    }

    /// A return annotation that is not a bare `int`/`float`/`str` name
    /// (a compiled alias name, `Age`) still declines outright on a
    /// genuinely-declining body — the fallback states nothing beyond the
    /// three base sorts.
    #[test]
    fn a_declined_while_loop_body_with_a_non_base_sort_annotation_still_declines() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def counted(n) -> Age:\n    while n > 0:\n        n -= 1\n    return n\n");
        assert!(call_result(&def, &[known_int(3.0)], None, &kernel, 0).is_none());
    }

    /// A def with a keyword-only parameter the CALLER never covers (no
    /// slot in `arguments` at all — the shape `bind_parameters` sees
    /// when a caller genuinely omits it, e.g. an optional kwonly with a
    /// default this file does not yet read) still reaches the coarse
    /// `-> int` fallback, since `bind_parameters`'s own arity check
    /// finds no slot for it.
    #[test]
    fn a_keyword_only_def_with_no_covering_slot_answers_the_whole_number_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def only_keyword(*, age) -> int:\n    return age\n");
        let result = call_result(&def, &[], None, &kernel, 0)
            .expect("the -> int annotation answers the whole-number set when no slot covers the kwonly param");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// e-class-and-function.py's own `keyword_only_call` regression: a
    /// keyword-only parameter the CALLER covers by keyword is no longer
    /// a hard decline — `expressions.rs`'s `positional_arguments_for_
    /// def` maps the caller's `age=200` onto this def's own trailing
    /// kwonly slot (that function's own doc), and `call_result` (called
    /// here exactly the way that mapping would hand it off) answers the
    /// body's own exact value, never the coarse fallback.
    #[test]
    fn a_keyword_only_def_with_a_covering_slot_answers_the_bodys_exact_value() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def only_keyword(*, age):\n    return age\n");
        let result = call_result(&def, &[known_int(200.0)], None, &kernel, 0)
            .expect("a covering slot binds the kwonly parameter and interprets the body");
        assert_eq!(result, known_int(200.0));
    }

    /// A plain parameter THEN a keyword-only one — the two families
    /// bind from adjacent slots in the SAME `arguments` vector
    /// (`bind_parameters`'s own doc: kwonly slots sit right after the
    /// plain parameters' own).
    #[test]
    fn a_plain_parameter_and_a_trailing_keyword_only_parameter_bind_from_adjacent_slots() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def mixed(first, *, second):\n    return first + second\n");
        let result = call_result(&def, &[known_int(1.0), known_int(2.0)], None, &kernel, 0)
            .expect("first binds positionally, second binds from the trailing kwonly slot");
        assert_eq!(result, known_int(3.0));
    }

    /// e-class-and-function.py's own `kwargs_parameter` regression: a
    /// `**kwargs` parameter binds from the VERY LAST slot of
    /// `arguments` — the collected dict `expressions.rs`'s
    /// `positional_arguments_with_kwargs_dict` would build and append
    /// there. `fields["age"]` reads the collected dict back through the
    /// ordinary subscript-read path once bound.
    #[test]
    fn a_kwargs_parameter_binds_the_final_slot_as_a_dict() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def gather_kwargs(**fields):\n    return fields[\"age\"]\n");
        let collected = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_int(200.0),
            }],
            None,
            true,
            TrustSpec,
            false,
        );
        let result = call_result(&def, &[collected], None, &kernel, 0)
            .expect("the final slot binds to fields, and fields[\"age\"] reads through");
        assert_eq!(result, known_int(200.0));
    }

    /// The depth cap's own decline point reaches the fallback too.
    #[test]
    fn the_depth_cap_decline_with_a_bare_int_return_annotation_answers_the_whole_number_set() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def double(x) -> int:\n    return x + x\n");
        let result = call_result(&def, &[known_int(3.0)], None, &kernel, CALL_DEPTH_CAP)
            .expect("the -> int annotation answers the whole-number set at the depth cap");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// The whole-number set genuinely admits a value the Age alias
    /// refuses — this is the CONTAINMENT check `for_over_unread_iterable`
    /// leans on: `whole_integers()` is not a subset of `Age`'s [0, 120]
    /// window (it admits 200, 121, negative values, …), so `scalar_subset`
    /// must answer false, matching `float_sorted_unknown`'s own sibling
    /// test in refined_domain.
    #[test]
    fn whole_integers_is_not_a_subset_of_a_bounded_int_window() {
        let Some(kernel) = loaded_kernel() else { return };
        let bounded = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(!(kernel.scalar_subset)(&whole_integers(), &bounded));
    }

    /// A body that reads CONCRETELY for one or more statements before
    /// declining is NOT opaque — the coarse `-> int` fallback must not
    /// fire. e-class-and-function.py's own `grow_into_bucket` shape:
    /// `bucket.append(age)` is an ordinary expression statement
    /// `interpret_body` reads fine (its result is simply discarded, per
    /// that arm's own doc); the decline happens only later, at
    /// `return bucket[0]`, because `bucket` itself is `unknown()` (its
    /// caller passed no argument, so `bind_parameters` evaluated the
    /// PARAMETER DEFAULT — a bare module-level name — against a fresh,
    /// name-less environment, per that function's own doc). Firing the
    /// coarse whole-number-set fallback here would overstate what this
    /// interpreter actually determined; the honest answer is `None`
    /// (`unknown()` at the call site), matching every other genuinely
    /// unread value this file declines rather than guesses at.
    #[test]
    fn a_body_that_reads_one_statement_before_declining_does_not_reach_the_coarse_fallback() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(concat!(
            "def grow_into_bucket(age, bucket=_DEFAULT_BUCKET) -> int:\n",
            "    bucket.append(age)\n",
            "    return bucket[0]\n",
        ));
        let result = call_result(&def, &[known_int(41.0)], None, &kernel, 0);
        assert!(
            result.is_none(),
            "a mid-body decline after a concretely-read statement must stay None, never the coarse -> int set: {result:?}"
        );
    }

    /// The CONTRASTING case, pinned alongside the one above so the two
    /// never drift apart: a body that declines on its very FIRST
    /// statement (never producing any readable effect) still reaches the
    /// coarse fallback — `unread_number`'s own shape
    /// (a-statements.py:34), `raise NotImplementedError` as the sole
    /// statement.
    #[test]
    fn a_body_that_declines_on_its_first_statement_still_reaches_the_coarse_fallback() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def unread_number() -> int:\n    raise NotImplementedError\n");
        let result = call_result(&def, &[], None, &kernel, 0)
            .expect("a first-statement decline is genuinely opaque, so the -> int fallback must still fire");
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// THE DOCSTRING GATE BUG's own regression: `unread_number`'s REAL
    /// body (a-statements.py:34-38) is a docstring FOLLOWED BY `raise
    /// NotImplementedError` — a docstring-only probe of "the first
    /// statement" would wrongly succeed (`Stmt::Expr` on a string
    /// literal always interprets fine) and mask that the body's first
    /// REAL statement is the one that declines, sending this def down
    /// the `None` path instead of the coarse `-> int` fallback. This
    /// pins the fix: `first_non_docstring_statement` skips the leading
    /// docstring, so the probe reaches `raise NotImplementedError` and
    /// correctly declines there.
    #[test]
    fn a_docstring_before_a_first_statement_decline_still_reaches_the_coarse_fallback() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(concat!(
            "def unread_number() -> int:\n",
            "    \"\"\"an opaque int source\"\"\"\n",
            "    raise NotImplementedError\n",
        ));
        let result = call_result(&def, &[], None, &kernel, 0).expect(
            "a docstring is not a readable effect — the def is still opaque from its first REAL statement, so the -> int fallback must fire",
        );
        assert_eq!(result.kind, Kind::Set);
        assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    }

    /// The CONTRASTING case the gate exists for stays out of the
    /// fallback even WITH a leading docstring: e-class-and-function.py's
    /// own `grow_into_bucket` shape, now with a docstring prepended — a
    /// concretely-read statement (`bucket.append(age)`) after the
    /// docstring still marks the body as genuinely readable, not opaque,
    /// so the answer stays `None` rather than the coarse fallback.
    #[test]
    fn a_docstring_before_a_concretely_read_statement_does_not_reach_the_coarse_fallback() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(concat!(
            "def grow_into_bucket(age, bucket=_DEFAULT_BUCKET) -> int:\n",
            "    \"\"\"mutable default\"\"\"\n",
            "    bucket.append(age)\n",
            "    return bucket[0]\n",
        ));
        let result = call_result(&def, &[known_int(41.0)], None, &kernel, 0);
        assert!(
            result.is_none(),
            "a docstring plus a mid-body decline after a concretely-read statement must stay None: {result:?}"
        );
    }

    /// A def whose body is NOTHING BUT a docstring (no statement after
    /// it at all) still reaches the coarse fallback — the same "first
    /// REAL statement" absence `first_non_docstring_statement`'s own
    /// `None` row declines through.
    #[test]
    fn a_body_that_is_only_a_docstring_reaches_the_coarse_fallback() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def only_documented() -> int:\n    \"\"\"nothing else here\"\"\"\n");
        let result = call_result(&def, &[], None, &kernel, 0);
        // a docstring-only body falls off the end (Kind::Null, the
        // Null-vs-scalar-ground law's own business) — this pins that the
        // docstring-only shape does not crash or mis-answer, without
        // asserting which existing law owns the resulting verdict
        assert!(result.is_some(), "a docstring-only body still answers something (falls through to None): {result:?}");
    }

    // --- call_effects: the CALLEE-EFFECTS CHANNEL ---

    /// a-statements.py's own `nonlocal_rebind`/`spoil`: `nonlocal age` then
    /// `age = 200` — the effect list must carry `("age", 200)`, the
    /// ENCLOSING name's own new value, not merely `spoil`'s own (Null)
    /// return.
    #[test]
    fn call_effects_reports_a_nonlocal_declared_write() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def spoil():\n    nonlocal age\n    age = 200\n");
        let mut enclosing = Environment::new(std::collections::HashSet::from(["age".to_owned()]));
        enclosing.bind("age", known_int(10.0));

        let (_value, effects) =
            call_effects(&def, &[], None, &kernel, 0, &enclosing).expect("a nonlocal write is a readable effect");
        assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
        assert_eq!(effects[0].0, "age");
        assert_eq!(effects[0].1, known_int(200.0));
    }

    /// a-statements.py's own `closure_mutates_flattened_capture`/`spoil`:
    /// `outlaw["age"] = 200` — a mutation THROUGH a captured free name,
    /// with no `nonlocal` declaration at all (CPython never requires one
    /// for a subscript/attribute STORE, only for rebinding the name
    /// itself). The effect is the WRITTEN-THROUGH dict, keyed on `outlaw`.
    #[test]
    fn call_effects_reports_a_captured_receiver_subscript_mutation() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def spoil():\n    outlaw[\"age\"] = 200\n");
        let mut enclosing = Environment::new(std::collections::HashSet::from(["outlaw".to_owned()]));
        let dict_value = refined_domain::known_constructors::known_object(
            vec![refined_domain::abstract_value::ObjectKey {
                name: "age".to_owned(),
                numeric: false,
                value: known_int(40.0),
            }],
            None,
            true,
            TrustProved,
            false,
        );
        enclosing.bind("outlaw", dict_value);

        let (_value, effects) =
            call_effects(&def, &[], None, &kernel, 0, &enclosing).expect("a captured-receiver mutation is readable");
        assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
        assert_eq!(effects[0].0, "outlaw");
        assert_eq!(effects[0].1.kind, Kind::Object);
        let written = effects[0].1.keys.iter().find(|entry| entry.name == "age").expect("age entry survives the write");
        assert_eq!(written.value, known_int(200.0));
    }

    /// A body with no `nonlocal` declaration and no captured-receiver
    /// mutation — an ordinary local write — reports an EMPTY effect list;
    /// `call_effects` never invents an effect for a purely local rebind
    /// (Python's own scoping rule: a plain `Assign` target with no
    /// `nonlocal` always creates a fresh local, never writes outward).
    #[test]
    fn call_effects_reports_no_effects_for_a_purely_local_write() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def("def bump():\n    age = 15\n    return age\n");
        let enclosing = Environment::new(std::collections::HashSet::new());
        let (value, effects) =
            call_effects(&def, &[], None, &kernel, 0, &enclosing).expect("a purely local write still answers");
        assert_eq!(value, known_int(15.0));
        assert!(effects.is_empty(), "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
    }

    /// A captured-receiver store this channel CANNOT compose (the free
    /// name's current value is a scalar Integer, not a dict/list —
    /// `dict_with_item`/`list_with_item` both answer `None` for it)
    /// answers an effect whose VALUE is `unknown()` — the caller MUST
    /// forget the name rather than keep its stale pre-call value
    /// (`call_effects`'s own doc: "a store you cannot compose answers
    /// that name unknown() so the caller FORGETS it — an effect is never
    /// silently dropped"). Exercised directly against `record_write_
    /// effect` (the law's own owning function) rather than through the
    /// full `call_effects` pipeline: `interpret_body`'s own subscript-
    /// write recognition (`write_subscript_target`, a sibling law added
    /// this same wave) reads the identical seeded free-name value and
    /// therefore ALREADY declines this exact body shape at the VALUE
    /// pass, before `call_effects`'s own second pass ever runs — so this
    /// unknown()-forget answer is not reachable through `call_effects`'s
    /// public surface on TODAY's fixture rows, but is real defensive
    /// code for a store shape the value pass might one day recognize
    /// more narrowly than the effects pass does; testing it directly
    /// keeps the law honest without asserting a false end-to-end claim.
    #[test]
    fn record_write_effect_answers_unknown_for_an_uncomposable_captured_receiver_store() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("outlaw[\"age\"] = 200\n")
            .expect("statement source parses")
            .into_syntax();
        let Stmt::Assign(assign) = module.body.into_iter().next().expect("one statement") else {
            panic!("expected an Assign statement");
        };
        let mut environment = Environment::new(std::collections::HashSet::new());
        environment.bind("outlaw", known_int(999.0));
        let nonlocal_names = std::collections::HashSet::new();
        let locally_bound = std::collections::HashSet::new();
        let mut effects: Vec<(String, AbstractValue)> = Vec::new();
        let [target] = assign.targets.as_slice() else { panic!("one target") };
        record_write_effect(target, assign.value.as_ref(), &kernel, &mut environment, &nonlocal_names, &locally_bound, &mut effects);
        assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
        assert_eq!(effects[0].0, "outlaw");
        assert_eq!(effects[0].1.kind, Kind::Unknown, "an uncomposable store forgets, never keeps a stale value");
    }

    /// A captured-receiver store on a free name never bound at all — the
    /// same `unknown()`-forgets answer, for the OTHER uncomposable shape
    /// (no current value to compose against, rather than a wrong-shaped
    /// one). Same direct-against-`record_write_effect` posture as above.
    #[test]
    fn record_write_effect_answers_unknown_for_a_store_through_a_never_bound_free_name() {
        let Some(kernel) = loaded_kernel() else { return };
        let module = ruff_python_parser::parse_module("outlaw[\"age\"] = 200\n")
            .expect("statement source parses")
            .into_syntax();
        let Stmt::Assign(assign) = module.body.into_iter().next().expect("one statement") else {
            panic!("expected an Assign statement");
        };
        let mut environment = Environment::new(std::collections::HashSet::new());
        let nonlocal_names = std::collections::HashSet::new();
        let locally_bound = std::collections::HashSet::new();
        let mut effects: Vec<(String, AbstractValue)> = Vec::new();
        let [target] = assign.targets.as_slice() else { panic!("one target") };
        record_write_effect(target, assign.value.as_ref(), &kernel, &mut environment, &nonlocal_names, &locally_bound, &mut effects);
        assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
        assert_eq!(effects[0].0, "outlaw");
        assert_eq!(effects[0].1.kind, Kind::Unknown);
    }

    // --- interpret_class_def: ClassDef-in-summary construction ---

    /// a-statements.py's own `device()` shape: a body-local class,
    /// constructed and returned. `call_result` must answer a TAGGED
    /// instance (`source == "_Device"`) carrying the field's own default
    /// — proof `Stmt::ClassDef` no longer falls to `interpret_body`'s
    /// catch-all decline.
    #[test]
    fn call_result_answers_a_tagged_instance_for_a_body_local_class_construction() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(concat!(
            "def device():\n",
            "    class _Device:\n",
            "        value: int = 0\n",
            "    return _Device()\n",
        ));
        let result = call_result(&def, &[], None, &kernel, 0).expect("a body-local ClassDef no longer declines");
        assert_eq!(result.kind, Kind::Object);
        assert_eq!(result.source, "_Device");
        let value_field = result.keys.iter().find(|entry| entry.name == "value").expect("value field present");
        assert_eq!(value_field.value, known_int(0.0));
    }

    /// The constructed instance's class is ALSO readable off
    /// `environment.classes()` inside the SAME call (not merely the
    /// returned value) — `_Device`'s own `__init__`-free field defaults
    /// still resolve when a later statement in the same body (out of this
    /// wave's fixture rows, but not precluded) constructs a second
    /// instance of the same class.
    #[test]
    fn interpret_class_def_registers_the_class_before_the_return_statement_runs() {
        let Some(kernel) = loaded_kernel() else { return };
        let def = parsed_def(concat!(
            "def two_devices():\n",
            "    class _Device:\n",
            "        value: int = 0\n",
            "    first = _Device()\n",
            "    return first\n",
        ));
        let result = call_result(&def, &[], None, &kernel, 0)
            .expect("a second construction of the same body-local class still resolves");
        assert_eq!(result.kind, Kind::Object);
        assert_eq!(result.source, "_Device");
    }
}
