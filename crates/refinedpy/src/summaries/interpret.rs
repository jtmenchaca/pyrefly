/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::trust_level_of;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAnnAssign;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtAugAssign;
use ruff_python_ast::StmtClassDef;
use ruff_python_ast::StmtIf;
use ruff_text_size::TextRange;

use crate::env::Environment;
use crate::expressions::binary_arithmetic_value;
use crate::expressions::call_one_argument_expression;
use crate::expressions::evaluate_expression;
use crate::instances::class_table;
use crate::instances::field_read;
use crate::instances::field_write;
use crate::instances::self_attribute_name;
use crate::instances::ClassModel;
use crate::match_arms;
use crate::narrowing;
use crate::surface::surface_imports;

use super::seed::free_variable_snapshot;

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
/// restricted forms only (`Assign`/`AnnAssign`/`AugAssign`/`Pass`/`Expr`/
/// `If`/`Return`/`ClassDef`/`Nonlocal`/a bounded `For` over a known
/// `Kind::List` — see `Stmt::For`'s own arm below). Returns `Some(true)`
/// when control can fall off the end of `body` (so the caller should
/// contribute a `null_value()` return), `Some(false)` when every path
/// through `body` ends in a recorded `Return`, and `None` the moment a
/// statement outside the restricted forms is met — the whole call
/// declines then, matching `loops.rs::run_restricted_body`'s all-or-
/// nothing posture.
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
                // A `name.method(args)` expression-statement is tried as a
                // MUTATION first (`write_mutating_call_expr`, the same
                // receiver-rebinding contract `check.rs`'s own top-level
                // walk applies) — `bucket.append(age)` must carry its
                // written element into a LATER read in this same body
                // (`grow_into_bucket`'s own `return bucket[0]`), not leave
                // `bucket` bound to its stale pre-call value. Only when the
                // expression is not this shape at all (`Err` from the
                // `Ok`/`Err` split below — the call's func is not a
                // Name-receiver Attribute call) does this fall back to the
                // ordinary evaluate-and-discard `interpret_body` always
                // used before this arm existed; a shape that IS this call
                // form but that `mutated_receiver` does not recognize
                // declines the whole interpretation, matching `write_
                // subscript_target`'s identical all-or-nothing posture,
                // rather than silently keeping a stale receiver bound.
                if is_mutating_call_expr_shape(expr_stmt.value.as_ref()) {
                    write_mutating_call_expr(expr_stmt.value.as_ref(), kernel, environment)?;
                } else {
                    evaluate_expression(expr_stmt.value.as_ref(), environment, kernel);
                }
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
                        // RETAINED CALLABLES: a bare `return lambda ...:
                        // ...` (e-class-and-function.py's `make_adder`)
                        // registers the lambda's own body into
                        // `environment` before the immutable `evaluate_
                        // return_value`/`evaluate_expression` path below
                        // reads it as a value — the same "register just
                        // before the immutable read" rule `check.rs::
                        // sink_value` follows for its own statement
                        // forms.
                        crate::expressions::register_retained_callables(value_expr, environment);
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
            // A NESTED `def` INSIDE A SUMMARIZED BODY (e-class-and-
            // function.py's `make_counter`'s own `def bump(...)`,
            // r-ast-census.py's `with_paramspec_presence`'s own `def
            // wrapper(...)`): retains the def's own body under a FRESH
            // counter key (`next_retained_callable_key` — never the AST
            // range, unlike a lambda's own registration: `env.rs`'s own
            // doc on why a def's key must be minted per call), with a
            // CLOSURE snapshot of every free name the def's body reads
            // (`free_variable_snapshot`) — taken HERE, at the moment the
            // def statement executes, never at the moment a later call
            // reaches it (`RetainedCallable`'s own doc: Python pins a
            // closure to its DEFINING scope). The name binds to the
            // retained-callable value the same way an ordinary
            // `Stmt::Assign` binds a name to whatever it evaluates to —
            // a later `return bump`/`return wrapper` reads this binding
            // through the ordinary `Expr::Name` arm, no special case
            // needed there.
            Stmt::FunctionDef(def) => {
                let closure = free_variable_snapshot(def, environment);
                let key = environment.next_retained_callable_key();
                environment.record_retained_callable(key, crate::env::RetainedCallable::from_def(def, closure));
                environment.bind(def.name.id.as_str(), crate::env::retained_callable_value(key));
            }
            // `for <name> in <iterable>: <body>` — bounded to a KNOWN
            // `Kind::List` receiver with every item known (the same
            // honesty `loops.rs::iterable_values`'s catch-all arm gives a
            // bare-Name iterable, reimplemented locally per this file's
            // own "no importing loops.rs" precedent, `generator_yields`'s
            // own doc). A `*rest: int` vararg parameter binds exactly
            // this shape at a CALL SITE (`bind_parameters`'s own vararg
            // row — a known-length tuple of the caller's own trailing
            // arguments, `tuple_literal_value` producing `Kind::List`),
            // so a callee whose body sums its own rest parameter now
            // summarizes instead of declining the whole call. The body
            // runs once per element, in order, on the SAME environment
            // (each element's own binding overwrites the last, matching
            // `loops.rs`'s own left-to-right iteration order) — a
            // `Stmt::Return` on any iteration ends the loop immediately
            // (real CPython: a `return` inside a `for` body exits the
            // function, no further elements bind), reported through the
            // ordinary `returns` accumulator.
            //
            // A REPETITION WINDOW receiver has no element list to step and
            // is read abstractly instead — ONE pass over the window's own
            // element set, the arm's own doc below. Any OTHER iterable
            // shape (unknown, a non-List non-window value, an element that
            // is itself unknown), a non-bare-Name target, or a non-empty
            // `else` clause declines the WHOLE call — never a partial
            // summary.
            //
            // `for key, group in groupby(...)` is the ONE other shape
            // read here: a tuple target over an iterable whose grouping
            // is unread, answered ABSTRACTLY (one pass over the key
            // image and a group window) rather than concretely — see
            // `groupby_pass_bindings`' own doc for the clause reading.
            // Tried FIRST, since its target is a tuple the bare-Name
            // gate below would decline outright.
            Stmt::For(for_stmt) => {
                if !for_stmt.orelse.is_empty() {
                    return None;
                }
                if let Some(bindings) = groupby_pass_bindings(for_stmt, environment, kernel) {
                    for (name, value) in bindings {
                        environment.bind(&name, value);
                    }
                    let falls_through =
                        interpret_body(&for_stmt.body, kernel, depth, environment, returns, super_resolver)?;
                    if !falls_through {
                        return Some(false);
                    }
                    continue;
                }
                let Expr::Name(target_name) = for_stmt.target.as_ref() else {
                    return None;
                };
                let receiver = evaluate_expression(for_stmt.iter.as_ref(), environment, kernel);
                // A REPETITION WINDOW receiver — `out.splitlines()` over an
                // UNREAD `out` (`string_models::sort_only`'s own
                // `splitlines` row answers `repetition(strings(), 0,
                // None)`), or a declared `list[X]` parameter's own seed.
                // There is no element LIST to step: the window states one
                // element set and no count. Every position draws from that
                // SAME set (`repetition_window_forms::as_repetition`), so
                // one pass over the element is the whole reading — exactly
                // the stand-in `loops::for_loop::repetition_window_element_
                // pass` makes for the identical receiver, and the same
                // one-abstract-pass posture `groupby_pass_bindings` above
                // already takes in this file.
                //
                // The pass runs on the SAME environment rather than a join
                // of the zero-iteration and one-pass states: this
                // interpreter has no per-name join channel, and the values
                // a body accumulates through it are read back through
                // claims that already cover the zero-iteration case (a
                // dict written at an unread key widens to the
                // unbounded-key star, whose own `len` is the floor `[0,
                // +inf)` — true of the empty dict too).
                if receiver.kind == Kind::Set && receiver.set_kind_tag == SetKindTag::None {
                    let repeated = as_repetition(&receiver.set)?;
                    let element = AbstractValue {
                        kind_tag: receiver.kind_tag,
                        ..known_set(repeated.element, None, trust_level_of(&receiver), SetKindTag::None)
                    };
                    environment.bind(target_name.id.as_str(), element);
                    let falls_through =
                        interpret_body(&for_stmt.body, kernel, depth, environment, returns, super_resolver)?;
                    if !falls_through {
                        return Some(false);
                    }
                    continue;
                }
                if receiver.kind != Kind::List || receiver.items.iter().any(|item| item.kind == Kind::Unknown) {
                    return None;
                }
                let mut ended_early = false;
                for element in receiver.items.clone() {
                    environment.bind(target_name.id.as_str(), element);
                    let falls_through = interpret_body(&for_stmt.body, kernel, depth, environment, returns, super_resolver)?;
                    if !falls_through {
                        ended_early = true;
                        break;
                    }
                }
                if ended_early {
                    return Some(false);
                }
            }
            // `match subject: case ... case ...` — mirrors `check.rs::
            // walk_match`'s own two-path reading, restricted to this
            // interpreter's return-collecting shape. A DECIDED subject
            // (`match_arms::match_taken_environment`) walks every arm its
            // own per-arm scalar split reaches (an unconditional single
            // arm, or several partial-overlap arms joined the way
            // `Environment::join` already joins any two branches) via the
            // closure below, which delegates to THIS function's own
            // `interpret_body` — `declined` catches an inner decline
            // (`interpret_body` answering `None`) so it propagates as
            // this whole call's own decline rather than being misread as
            // "the match was undecided." Every match this corpus's
            // callee bodies build uses a STRING-literal `MatchValue`
            // pattern (`case "left":`), which `match_arms.rs`'s scalar
            // narrowing never decides (its own `enumerable_numeric_
            // members` reads Number/Boolean/Integer/Float-tagged subjects
            // only — see that file's own doc), so in practice this call
            // always falls to the JOIN path below: every case forks the
            // incoming environment, binds whatever `match_arms::
            // pattern_bound_captures` can name (a plain literal/wildcard
            // pattern names none — `Some(Vec::new())` — so this never
            // actually blocks on an unnameable capture for the shapes
            // this corpus builds), interprets that arm's body, and every
            // surviving arm (one that falls through rather than
            // returning) joins through `Environment::join`, the same
            // discipline `interpret_undecided_arms` gives an `if`/`elif`/
            // `else` chain. A case whose own pattern cannot even be
            // NAMED (a sequence/mapping/class pattern past `pattern_
            // bound_captures`'s own flat-capture scope) declines the
            // whole call — this restricted interpreter has no
            // blocker-recording channel to fall back to the way
            // `check.rs`'s full walk does.
            Stmt::Match(match_stmt) => {
                let subject_value = evaluate_expression(match_stmt.subject.as_ref(), environment, kernel);
                let subject_name = match match_stmt.subject.as_ref() {
                    Expr::Name(name) => Some(name.id.as_str()),
                    _ => None,
                };
                let mut declined = false;
                let decided = match_arms::match_taken_environment(
                    &subject_value,
                    subject_name,
                    &match_stmt.cases,
                    environment,
                    kernel,
                    &mut |body, arm_env| {
                        let result = interpret_body(body, kernel, depth, arm_env, returns, super_resolver);
                        if result.is_none() {
                            declined = true;
                        }
                        result
                    },
                );
                if declined {
                    return None;
                }
                if let Some((arm_env, falls_through)) = decided {
                    *environment = arm_env;
                    if !falls_through {
                        return Some(false);
                    }
                    continue;
                }
                let mut surviving: Vec<Environment> = Vec::new();
                for case in &match_stmt.cases {
                    let bound_captures =
                        match_arms::pattern_bound_captures(&case.pattern, &subject_value, environment, kernel)?;
                    let mut arm_environment = environment.fork();
                    for (name, value) in bound_captures {
                        arm_environment.bind(&name, value);
                    }
                    let falls_through =
                        interpret_body(&case.body, kernel, depth, &mut arm_environment, returns, super_resolver)?;
                    if falls_through {
                        surviving.push(arm_environment);
                    }
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
            // `global <name>[, ...]` — the same declaration-only shape as
            // `nonlocal`, just naming the MODULE scope instead of an
            // enclosing function scope (simple_stmts.rst, "The `global`
            // statement": it "causes the listed identifiers to be
            // interpreted as globals"). This interpreter still tracks no
            // scope chain, so the declaration itself neither reads nor
            // writes a value — recognizing it, exactly like `Stmt::Nonlocal`,
            // is what lets a body OPENING with `global _module_age` reach its
            // own following statements at all, rather than declining the
            // whole call on the declaration alone.
            Stmt::Global(_) => {}
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

/// The `(key, group)` pair one pass of `for key, group in
/// groupby(<iterable>[, key=<callable>]):` binds, as `(name, value)`
/// rows ready to bind — A8.seed.library's own `group_by_parity`, read
/// inside a summarized body. The same clause reading
/// `loops::for_loop::groupby_element_pass` states for the ordinary walk,
/// reimplemented locally per this file's own "no importing loops.rs"
/// precedent:
///
/// library/itertools.rst, `groupby(iterable, key=None)`: "Make an
/// iterator that returns consecutive keys and groups from the
/// *iterable*. The *key* is a function computing a key value for each
/// element. If not specified or is ``None``, *key* defaults to an
/// identity function and returns the element unchanged." And on the
/// group: "The returned group is itself an iterator that shares the
/// underlying iterable," yielding values drawn from it.
///
/// Over an iterable this domain reads only as a REPETITION WINDOW, no
/// exact grouping exists — the element values decide both the group
/// count and where the breaks fall, and neither is read. What the entry
/// pins, given the element set: the KEY is the key function's IMAGE over
/// that set (the element set itself when `key=` is absent or `None`, the
/// entry's own identity default), and the GROUP is a sequence of
/// elements of the same iterable — a repetition window over that same
/// element set, starting at 1 since every group `groupby` emits holds at
/// least the element that created it.
///
/// One pass, never a per-group walk: the group COUNT is exactly what is
/// unread, so there is no element list to step. `None` for any other
/// shape — a non-`groupby` iterable, a shadowed `groupby`/`itertools`
/// name, a receiver that is not a bare repetition window, a target that
/// is not a two-name tuple, a `key=` whose image cannot be read, or a
/// keyword the entry's signature does not name.
fn groupby_pass_bindings(
    for_stmt: &ruff_python_ast::StmtFor,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(String, AbstractValue)>> {
    let Expr::Tuple(target) = for_stmt.target.as_ref() else {
        return None;
    };
    let [key_target, group_target] = &*target.elts else {
        return None;
    };
    let (Expr::Name(key_name), Expr::Name(group_name)) = (key_target, group_target) else {
        return None;
    };
    let Expr::Call(call) = for_stmt.iter.as_ref() else {
        return None;
    };
    let bare = matches!(call.func.as_ref(), Expr::Name(name) if name.id.as_str() == "groupby")
        && environment.read("groupby").is_none();
    let qualified = match call.func.as_ref() {
        Expr::Attribute(attribute) if attribute.attr.as_str() == "groupby" => {
            matches!(attribute.value.as_ref(), Expr::Name(module) if module.id.as_str() == "itertools")
                && environment.read("itertools").is_none()
        }
        _ => false,
    };
    if !bare && !qualified {
        return None;
    }
    let [iterable_expr] = &*call.arguments.args else {
        return None;
    };
    let iterable = evaluate_expression(iterable_expr, environment, kernel);
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    let repeated = as_repetition(&iterable.set)?;
    let grade = trust_level_of(&iterable);
    let element = AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(repeated.element.clone(), None, grade, SetKindTag::None)
    };
    let mut key_expression: Option<&Expr> = None;
    for keyword in &call.arguments.keywords {
        let name = keyword.arg.as_ref()?;
        if name.id.as_str() != "key" {
            return None;
        }
        key_expression = Some(&keyword.value);
    }
    let key_value = match key_expression {
        None => element.clone(),
        Some(Expr::NoneLiteral(_)) => element.clone(),
        Some(expression) => call_one_argument_expression(expression, &element, environment, kernel)?,
    };
    if key_value.kind == Kind::Unknown {
        return None;
    }
    let group_value = AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(repetition(repeated.element, 1, None), None, grade, SetKindTag::None)
    };
    Some(vec![
        (key_name.id.to_string(), key_value),
        (group_name.id.to_string(), group_value),
    ])
}

/// `(a, b, ...) = value` / `[a, b, ...] = value` inside a restricted
/// body — e-class-and-function.py's own `unpack_first`: `a, _b = ages`
/// where `ages` is the def's own tuple-typed PARAMETER (`ages: tuple[int,
/// int]`), a known `Kind::List` value bound at call time; and
/// A8.edge.process's own `k, v = line.split("=", 1)`. No starred element
/// (`a, *rest = value` is out of this restricted interpreter's scope —
/// the mission names no fixture row needing it here, and
/// `check.rs::bind_known_sequence_target` already owns that shape for the
/// ordinary walk); every target must be a bare `Expr::Name` (a nested
/// tuple/list sub-target is also out of scope, same reasoning).
///
/// simple_stmts.rst, "Assignment statements", states the rule for a
/// target list that is not a single target: "The object must be an
/// iterable with the same number of items as there are targets in the
/// target list, and the items are assigned, from left to right, to the
/// corresponding targets." Two right-side shapes carry that reading, the
/// same two `loops::body_once::run_unpack_assign_once` reads for the
/// identical statement inside a loop body:
///
/// - an EXACT `Kind::List`: the arity is known, so a matching count
///   binds positionally and a mismatch is CPython's own `ValueError`,
///   which this restricted interpreter has no finding sink for — an
///   honest decline rather than a silently-wrong bind.
/// - a REPETITION WINDOW (`Kind::Set` reading back through
///   `as_repetition` — `line.split("=", 1)` over an unread `line`): the
///   window states no exact item count, but every position draws from
///   the SAME element set, so on every run whose arity does match — the
///   only runs that do not raise — each target's item is somewhere in
///   that one element set, and binding every target to the element is
///   the claim the window supports.
///
/// `None` (the whole call declines) for any other right-side value.
fn bind_unpack_target(target: &Expr, value: &AbstractValue, environment: &mut Environment) -> Option<()> {
    let elements: &[Expr] = match target {
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::List(list) => &list.elts,
        _ => return None,
    };
    let mut names: Vec<&str> = Vec::with_capacity(elements.len());
    for element in elements {
        let Expr::Name(name) = element else {
            return None;
        };
        names.push(name.id.as_str());
    }
    if value.kind == Kind::Set && value.set_kind_tag == SetKindTag::None {
        let repeated = as_repetition(&value.set)?;
        let element = AbstractValue {
            kind_tag: value.kind_tag,
            ..known_set(repeated.element, None, trust_level_of(value), SetKindTag::None)
        };
        for name in &names {
            environment.bind(name, element.clone());
        }
        return Some(());
    }
    if value.kind != Kind::List || names.len() != value.items.len() {
        return None;
    }
    for (name, item) in names.iter().zip(value.items.iter()) {
        environment.bind(name, item.clone());
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
        Kind::Object => crate::collection_models::dict_with_item(&receiver, &key, &value)?,
        Kind::List => crate::collection_models::list_with_item(&receiver, &key, &value)?,
        _ => return None,
    };
    environment.bind(name.id.as_str(), new_receiver);
    Some(())
}

/// Whether `expr` is the `name.method(args)` shape `write_mutating_call_expr`
/// knows how to attempt — a syntactic check only (never reads `environment`),
/// so `interpret_body`'s `Stmt::Expr` arm can tell "not this shape, fall back
/// to evaluate-and-discard" apart from "this shape, but the mutation itself
/// is unresolvable, decline the whole call."
fn is_mutating_call_expr_shape(expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return false;
    };
    matches!(attribute.value.as_ref(), Expr::Name(_))
}

/// `name.method(args)` as its own expression-statement inside a restricted
/// body — e-class-and-function.py's own `grow_into_bucket`:
/// `bucket.append(age)` mutating a parameter bound from a module-level
/// default (`bucket: list[int] = _DEFAULT_BUCKET`). `name` must already be
/// bound to a known receiver; `collection_models::mutated_receiver` (the
/// SAME contract `check.rs::walk_mutating_call_statement` uses for the
/// ordinary top-level walk) replays the call and answers the updated
/// receiver, which rebinds `name` so a LATER read in the same body (this
/// function's own `return bucket[0]`) sees the write rather than the
/// stale pre-call value. `None` when `name` is unbound or `mutated_receiver`
/// does not recognize the method on a KNOWN receiver kind — this is only
/// ever called once `is_mutating_call_expr_shape` has already confirmed the
/// syntactic shape, so a `None` here always means "this interpreter's own
/// contract cannot replay this specific mutation," and the whole call
/// declines rather than silently keeping a stale receiver bound.
///
/// An UNKNOWN receiver (`grow_into_bucket`'s own shape when `bucket`'s
/// module-level default is out of reach — no `enclosing` environment
/// carries `_DEFAULT_BUCKET`) is not this same "unrecognized shape"
/// decline: the statement syntactically IS a mutating method call, so it
/// is a genuinely recognized, concretely-attempted effect whose OUTCOME
/// happens to stay unknown — the receiver rebinds to `unknown()` rather
/// than the whole call declining here. A later read of that same name
/// (`return bucket[0]`) still declines on its own terms
/// (`evaluate_expression`'s subscript-on-unknown reading), which is the
/// honest place for THIS body's opacity to surface, not the mutation
/// statement that merely could not resolve to a concrete value.
fn write_mutating_call_expr(expr: &Expr, kernel: &Arc<RefinedTSKernel>, environment: &mut Environment) -> Option<()> {
    let Expr::Call(call) = expr else {
        return None;
    };
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let Expr::Name(receiver_name) = attribute.value.as_ref() else {
        return None;
    };
    let receiver = environment.read(receiver_name.id.as_str())?.clone();
    let arguments: Vec<AbstractValue> =
        call.arguments.args.iter().map(|argument| evaluate_expression(argument, environment, kernel)).collect();
    if receiver.kind == Kind::Unknown {
        environment.bind(receiver_name.id.as_str(), unknown());
        return Some(());
    }
    let (new_receiver, _result) =
        crate::collection_models::mutated_receiver(attribute.attr.as_str(), &receiver, &arguments)?;
    environment.bind(receiver_name.id.as_str(), new_receiver);
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
///
/// Each arm's own fork is narrowed by `narrowing::assume` before its
/// body interprets — CPython only reaches arm N once every EARLIER
/// test proved false, so arm N's fork is narrowed `false` by each of
/// those, THEN `true` by its own test (when it has one; a bare `else`
/// arm carries no test of its own to narrow by). This is what lets
/// e-class-and-function.py's `pick_years` read `return value` inside
/// `if isinstance(value, int):` with `value` still carrying its
/// concrete argument (`isinstance`'s own test is undecidable at the
/// TRUTHINESS level — `evaluate_expression` has no `isinstance` model
/// — but `assume`'s narrowing channel reads the SAME call shape and
/// tightens the binding directly), mirroring `check.rs::walk_if`'s own
/// per-arm `assume` call for the ordinary walk.
///
/// An arm the narrowing just proved IMPOSSIBLE for this call's concrete
/// arguments (`narrowing::arm_is_infeasible`) is skipped WITHOUT
/// interpreting its body — the same `pick_years(200)` call's own
/// `isinstance(value, int)` FALSE arm narrows `value` to the empty set
/// (200 genuinely is an int), and CPython never runs `return
/// len(value)` for this call at all; interpreting it anyway and letting
/// its own unmodeled `len()` call decline would sink the WHOLE call
/// (the `?` on `interpret_body`'s result) even though the arm actually
/// taken (`return value`) is fully readable. A dead arm contributes no
/// surviving fork and no return value — the same as any other
/// terminating arm, just reached for a different reason.
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
    for (arm_index, (test, body)) in arms.iter().enumerate() {
        has_catch_all = has_catch_all || test.is_none();
        let mut arm_environment = environment.fork();
        let mut infeasible = false;
        for (earlier_test, _) in arms.iter().take(arm_index) {
            if let Some(earlier_test) = earlier_test {
                arm_environment = narrowing::assume(earlier_test, arm_environment, kernel, false);
                infeasible = infeasible || narrowing::arm_is_infeasible(earlier_test, &arm_environment);
            }
        }
        if let Some(test_expr) = test {
            arm_environment = narrowing::assume(test_expr, arm_environment, kernel, true);
            infeasible = infeasible || narrowing::arm_is_infeasible(test_expr, &arm_environment);
        }
        if infeasible {
            continue;
        }
        let falls_through = interpret_body(body, kernel, depth, &mut arm_environment, returns, super_resolver)?;
        if falls_through {
            surviving.push(arm_environment);
        }
    }
    if !has_catch_all {
        // No `else` at all (`if test: return ...` falling straight into
        // the NEXT statement, e-class-and-function.py's `pick_years` —
        // `if isinstance(value, int): return value` with no `else`,
        // `return len(value)` is simply the statement after the `if`,
        // not a second arm) — the implicit fallthrough is reached only
        // when EVERY test in `arms` was false, so it is narrowed by all
        // of them the same way an explicit later arm would be.
        let mut fallthrough_environment = environment.fork();
        let mut fallthrough_infeasible = false;
        for (test, _) in arms {
            if let Some(test_expr) = test {
                fallthrough_environment = narrowing::assume(test_expr, fallthrough_environment, kernel, false);
                fallthrough_infeasible =
                    fallthrough_infeasible || narrowing::arm_is_infeasible(test_expr, &fallthrough_environment);
            }
        }
        // A fallthrough narrowing already proven impossible for this
        // call's concrete arguments (`pick_years(200)`'s own `value`
        // narrowed to the empty Integer set once `isinstance(value,
        // int)` proved true) is never reached by CPython — the
        // statement after the `if` (`return len(value)`) is dead code
        // for THIS call, so it must not contribute a surviving fork
        // (or be walked at all): a surviving-but-impossible fork is
        // exactly what let an unrelated, unmodeled construct in dead
        // code decline the whole call.
        if !fallthrough_infeasible {
            surviving.push(fallthrough_environment);
        }
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
