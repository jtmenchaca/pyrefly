//! Python call-expression evaluation: the `evaluate_call` dispatcher and
//! its siblings — same-module user function calls (summaries/inlining),
//! retained-callable and lambda calls, builtin/stdlib/module dispatch,
//! method calls on known receivers, constructor calls, and the argument-
//! evaluation helpers those paths share.
//!
//! The re-export block below is this module's one door: every row its
//! children implement is named there, whether or not a caller outside
//! the module reads that particular row today. A row with no current
//! reader is still part of the stated interface, so the block carries
//! `allow(unused_imports)` rather than being trimmed to today's
//! callers and re-grown one line at a time as callers appear.
#![allow(unused_imports)]

mod attribute_call;
mod construction;
mod functional;
mod helpers;
mod retained;

pub(super) use attribute_call::evaluate_attribute_call;
pub(super) use attribute_call::is_literal_regex_pattern;
pub(super) use attribute_call::receiver_def_local_classes;
pub use attribute_call::unmodeled_module_call_name;
pub(crate) use construction::exception_construction_value;
pub(crate) use construction::fieldless_exception_value;
pub(super) use construction::is_builtin_exception_constructor;
pub(super) use construction::is_utc_tzinfo_expression;
pub(super) use construction::known_byte_sequence;
pub(crate) use construction::math_from_imports;
pub(super) use functional::call_one_argument_expression;
pub(super) use functional::call_two_argument_expression;
pub(super) use functional::filter_expression_value;
pub(super) use functional::map_expression_value;
pub(super) use functional::reduce_expression_value;
pub(super) use helpers::eval_whole_integers;
pub(super) use helpers::evaluate_bytes_literal;
pub(super) use helpers::is_generator_def;
pub(super) use helpers::is_valid_base_ten_int_string;
pub(super) use helpers::range_argument_value;
pub(super) use helpers::range_expression_value;
pub(super) use helpers::splice_call_arguments;
pub use retained::register_retained_callables;
pub(super) use retained::positional_arguments_by_names;
pub(super) use retained::positional_arguments_for_def;
pub(super) use retained::positional_arguments_for_method;
pub(super) use retained::positional_arguments_with_kwargs_dict;
pub(super) use retained::retained_callable_call_result;

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::builtin_models;
use crate::collection_models;
use crate::env;
use crate::env::Environment;
use crate::instances;
use crate::summaries;

use super::arithmetic;
use super::evaluate_expression;
use super::boolop_ternary::same_module_def_gate_open;
use super::compare::exact_string_values;
use super::datetime::date_construction_value;
use super::datetime::date_fromisoformat_value;
use super::datetime::datetime_construction_value;
use super::datetime::datetime_fromtimestamp_value;
use super::datetime::is_datetime_date_attribute;
use super::datetime::is_datetime_datetime_attribute;
use super::datetime::is_datetime_timedelta_attribute;
use super::datetime::subprocess_run_construction_value;
use super::datetime::timedelta_construction_value;
use super::fstring::code_points_to_string;

pub(super) fn evaluate_call(call: &ruff_python_ast::ExprCall, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> AbstractValue {
    // A RETAINED-CALLABLE CALL: `name(...)` where `name` reads a value
    // `env::retained_callable_value` built — a lambda or nested `def`
    // that reached this call site through a binding path other than
    // "declared and called in the same body" (returned out of its
    // defining function, passed in as a call argument, read back off
    // an instance field). Tried BEFORE the same-module-def dispatch
    // below: a retained callable's own table entry is a stronger,
    // execution-traced fact than a bare same-module `def` of the same
    // spelling would be, and — for `add_one = make_adder(1)` — there
    // is no module-level `def add_one` for that dispatch to find
    // anyway, so trying this first changes nothing for the shapes that
    // DO have a same-module def of the lambda-bound name
    // (`same_module_def_gate_open`'s own doc already treats that name
    // as open, meaning a real module-level `def` of the same spelling
    // is the intended callee there — this arm never reaches that case
    // since `retained_callable_key` answers `None` for an ordinary,
    // non-retained lambda value).
    if let Expr::Name(name) = call.func.as_ref() {
        if let Some(value) = environment.read(name.id.as_str()) {
            if let Some(result) = retained_callable_call_result(value, call, environment, kernel) {
                return result;
            }
        }
    }
    // A SAME-MODULE-DEF ALIAS CALL: `f = identity; f(x)` — `f` reads a
    // value `env::same_module_def_alias_value` built (this file's own
    // `Expr::Name` read arm, for a bare reference to a module-level
    // `def`). Tried alongside the retained-callable arm above, for the
    // identical reason: the value already NAMES its own callee (the
    // def's own name, read back through `env::same_module_def_alias_
    // name`), a stronger fact than the bare `f` spelling `table.def`
    // would look up next — and `table.def("f")` would find nothing
    // anyway unless the module happens to ALSO declare a `def f`. No
    // retained-body table entry, no closure snapshot: the aliased def
    // is a MODULE-LEVEL def, already fully resolvable by name through
    // `environment.functions()`, so this calls `summaries::call_result_
    // with_enclosing` directly on it, exactly the way the same-module-
    // def dispatch just below calls a def reached by its own literal
    // name.
    if let Expr::Name(name) = call.func.as_ref() {
        if let Some(value) = environment.read(name.id.as_str()) {
            if let Some(aliased_name) = env::same_module_def_alias_name(value) {
                if let Some(table) = environment.functions() {
                    if let Some(def) = table.def(aliased_name) {
                        let Some(positional) = positional_arguments_for_def(call, def, environment, kernel) else {
                            return unknown();
                        };
                        let answer = summaries::call_result_with_enclosing(
                            def,
                            &positional,
                            environment.functions(),
                            kernel,
                            environment.call_depth(),
                            Some(environment),
                        );
                        return answer.unwrap_or_else(unknown);
                    }
                }
            }
        }
    }
    if let Expr::Name(name) = call.func.as_ref() {
        if same_module_def_gate_open(environment, name.id.as_str()) {
            if let Some(table) = environment.functions() {
                if let Some(def) = table.def(name.id.as_str()) {
                    let Some(positional) = positional_arguments_for_def(call, def, environment, kernel) else {
                        return unknown();
                    };
                    // a GENERATOR function's own call (a body whose
                    // top-level statements are straight-line `yield`s,
                    // `is_generator_def`'s own doc) never reaches
                    // `summaries::call_result` — that restricted
                    // interpreter has no `yield` row and would decline
                    // the whole call. `generator_yields` reads the same
                    // body instead, and the CALL answers the ordered
                    // List of every yielded value (this domain's shared
                    // list/set/generator representation,
                    // `collection_models.rs`'s own module doc), tagged
                    // `source = "generator"` so `next`'s own dispatcher
                    // (`next_call`/`evaluate_call`'s builtin path) can
                    // tell a fresh generator value apart from an
                    // ordinary list — see `next`'s own doc for why a
                    // SECOND `next` on the same value declines rather
                    // than answering the next element (this domain
                    // carries no generator position/exhaustion state).
                    if is_generator_def(def) {
                        return match instances::generator_yields(def, &positional, environment.functions(), kernel, environment.call_depth())
                        {
                            Some(yields) => {
                                let mut value = collection_models::list_literal_value(&yields);
                                value.source = "generator".to_owned();
                                value
                            }
                            // The DECLINE twin of the tagged success above:
                            // `generator_yields` could not summarize this
                            // body (a conditional `yield`, or any other
                            // shape outside its own straight-line reading).
                            // Tagged `source = GENERATOR_DECLINED_SOURCE_TAG`
                            // (`check.rs`'s own constant, mirrored here as a
                            // literal since `expressions.rs` is upstream of
                            // `check.rs` in this crate's dependency order —
                            // matching the SAME literal spelling, not the
                            // constant itself) rather than a bare unknown,
                            // so `check.rs::name_unmodeled_call_sentence`'s
                            // generator rung can trace an undetermined read
                            // this call feeds back to its own cause, instead
                            // of the generic "value not readable" wording.
                            None => AbstractValue {
                                source: "generator-declined".to_owned(),
                                ..unknown()
                            },
                        };
                    }
                    // CLOSURE READS: `def` may be a NESTED def (this
                    // call's own `environment` is the enclosing body's
                    // locals at the call site) reading a free name neither
                    // its own parameters nor its own body bind —
                    // `call_result_with_enclosing`'s own doc
                    // (executionmodel.rst's "Naming and binding": a free
                    // variable reads the enclosing scope's binding). Passing
                    // the CALL SITE's `environment` here is sound for a
                    // same-body define-then-call flow (the corpus's own
                    // shape — a nested `def` declared and called inside the
                    // same enclosing body): the enclosing environment at
                    // the point of the call already carries whatever the
                    // enclosing body bound before this call ran. A
                    // module-level `def` (no true enclosing scope) still
                    // answers identically either way — its own
                    // `free_names_read` walk never finds a name the
                    // enclosing environment did not already fail to bind
                    // either, so seeding costs nothing when there is
                    // nothing to seed.
                    return match summaries::call_result_with_enclosing(
                        def,
                        &positional,
                        environment.functions(),
                        kernel,
                        environment.call_depth(),
                        Some(environment),
                    ) {
                        Some(value) => value,
                        None => unknown(),
                    };
                }
            }
            if let Some(classes) = environment.classes() {
                if let Some(model) = classes.get(name.id.as_str()) {
                    if call.arguments.args.iter().any(|arg| matches!(arg, Expr::Starred(_))) {
                        return unknown();
                    }
                    let positional: Vec<(AbstractValue, TextRange)> = call
                        .arguments
                        .args
                        .iter()
                        .map(|arg| (evaluate_expression(arg, environment, kernel), arg.range()))
                        .collect();
                    let keyword: Vec<(String, AbstractValue, TextRange)> = call
                        .arguments
                        .keywords
                        .iter()
                        .filter_map(|kw| {
                            let arg_name = kw.arg.as_ref()?;
                            Some((
                                arg_name.as_str().to_owned(),
                                evaluate_expression(&kw.value, environment, kernel),
                                kw.value.range(),
                            ))
                        })
                        .collect();
                    // a construction is a VALUE here — the verdict's fires
                    // belong to whichever statement sink hosts this call
                    // expression, not to this nested value read
                    let verdict = instances::judge_construction(model, &positional, &keyword, kernel);
                    return verdict.instance;
                }
            }
        }
        // A CALLABLE-VARIABLE CALL: `name(...)` where `name` is a bare Name
        // this environment's `callable_returns` table carries (a
        // `Callable[[...], R]`-annotated variable,
        // `typereading::callable_return_refinement` / `walk_ann_assign`'s
        // own recording seam) AND `name` does not ALSO resolve to a
        // same-module def/class. Placed OUTSIDE the `same_module_def_gate_
        // open` block and checked directly here (not by relying on that
        // gate to have excluded a def/class name already): a MODULE-LEVEL
        // `Callable`-typed name read from inside a function body is
        // usually gate-OPEN anyway (the name is never in that function's
        // own `locally_bound` set, so `environment.read` answers `None`
        // there, same as any other unbound outer name), so the def/class
        // dispatch above already tries first and returns early whenever
        // one of them actually matches — this direct check exists for the
        // remaining case, a LOCAL `Callable`-typed rebinding the gate
        // would close (bound to a real value, not an opaque lambda),
        // where the def/class dispatch above is skipped entirely and this
        // arm is the only remaining check standing between it and a wrong
        // answer. This is the same channel
        // `check.rs::callable_variable_call_result` gives `sink_value`'s
        // direct-sink shape (`x: Age = maybe_next_year(40)`) — this arm is
        // the NESTED-expression twin, reached when the call sits inside a
        // larger expression (b-body-expressions.py:79's ternary-guarded
        // `maybe_next_year(40) if maybe_next_year is not None else 0`,
        // where the call is evaluated by `evaluate_ternary`'s own
        // `evaluate_expression` recursion, never by `sink_value`). Answers
        // `R`'s own declared set at TrustSpec — an annotation is the
        // developer's claim, not an execution-proved fact — the same
        // grade `callable_variable_call_result` uses.
        if let Some(declared) = environment.callable_returns().and_then(|table| table.get(name.id.as_str())) {
            let shadowed_by_def = environment.functions().is_some_and(|functions| functions.def(name.id.as_str()).is_some());
            let shadowed_by_class = environment.classes().is_some_and(|classes| classes.contains_key(name.id.as_str()));
            if !shadowed_by_def && !shadowed_by_class {
                return known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None);
            }
        }
    }
    // `receiver.method(...)` on a known INSTANCE (a `Kind::Object` whose
    // `source` names the constructing class, `judge_construction`'s own
    // tag) — the method's own def resolves through `method_def_of`, then
    // `method_call_result` interprets it the same restricted way
    // `summaries::call_result` interprets a plain `def`, with keyword
    // arguments mapped to position first (this is the ONE method-call
    // path that reads keywords — every other method/builtin/math row
    // below still declines any keyword, per the existing guard). Only
    // the RESULT half of `method_call_result`'s `(instance after, result)`
    // pair is read here: the mutated-instance half is check.rs's own
    // statement-sink business (the same "fires/writes belong to the
    // sink" split the construction arm above already draws), so a
    // nested method call inside a larger expression never threads its
    // own receiver mutation back into the environment.
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        // a `math`/`re`/`asyncio` MODULE-name receiver evaluates to
        // `unknown()` here (no binding, no class) and simply misses the
        // `Kind::Object`-with-`source` check below, falling through to
        // `evaluate_attribute_call`'s own module-name arms unaffected.
        let receiver = evaluate_expression(&attribute.value, environment, kernel);
        // A RETAINED-CALLABLE FIELD CALL: `receiver.attr(...)` where
        // `attr` is a STORED field (never a class method — a `def` in
        // the class body resolves through `method_def_of` below
        // instead) holding a retained lambda/def value
        // (b-body-expressions.py's `function_nested_on_object`:
        // `Person(lambda: 40)` stores the lambda as `self.years`, and
        // `person.years()` calls it back). Tried before the
        // class-method dispatch: a field and a method never share a
        // name on the same class (`instances::field_read`/`method_def_
        // of` both key off the class's own single namespace), so this
        // never shadows an actual method call.
        if receiver.kind == Kind::Object {
            if let Some(field) = instances::field_read(&receiver, attribute.attr.as_str()) {
                if let Some(result) = retained_callable_call_result(&field, call, environment, kernel) {
                    return result;
                }
            }
        }
        if receiver.kind == Kind::Object && !receiver.source.is_empty() {
            // TWO SIBLING NESTED DEFS DECLARING THE SAME CLASS NAME
            // (b-body-expressions.py's `binary_chained_builder_call`:
            // `make_ok_builder`/`make_over_builder` each declare their
            // own `class Builder`) collide in `environment.classes()` —
            // the caller's own flat, body-wide table can hold only ONE
            // `"Builder"` entry, whichever `check.rs::local_class_table`
            // happened to see first while pre-scanning the caller's body.
            // A CHAINED call's receiver (`make_over_builder().type("x")`)
            // needs the SPECIFIC sibling's own class, not that shared
            // guess, so `receiver_def_local_classes` re-reads the class
            // straight from the same-module def the receiver expression
            // actually traces back to, fresh, with no sibling to collide
            // against. Tried first; `environment.classes()` still answers
            // every other receiver (an ordinary constructed instance, a
            // parameter, a field read) exactly as before.
            let scoped_classes = receiver_def_local_classes(&attribute.value, environment, kernel);
            let classes_for_call = match &scoped_classes {
                Some(scoped) if scoped.contains_key(receiver.source.as_str()) => Some(scoped),
                _ => environment.classes(),
            };
            if let Some(classes) = classes_for_call {
                if let Some(model) = classes.get(receiver.source.as_str()) {
                    if let Some(method) = instances::method_def_of(model, attribute.attr.as_str()) {
                        let Some(positional) = positional_arguments_for_method(call, method, environment, kernel) else {
                            return unknown();
                        };
                        // A GENERATOR METHOD (`class GenAges: def ages(self):
                        // yield 40`, e-class-and-function.py's own
                        // `generator_method`/`async_generator_method`) —
                        // the exact same `Stmt::Expr(Expr::Yield)`-shaped
                        // body `evaluate_call`'s bare-def dispatch already
                        // routes to `instances::generator_yields` rather
                        // than `method_call_result` (that call site's own
                        // doc: "that restricted interpreter has no `yield`
                        // row and would decline the whole call"). A method
                        // body is the identical restricted-interpreter
                        // shape one level down (self bound, otherwise the
                        // same straight-line-yields reading), so this arm
                        // checks the SAME `is_generator_def` gate before
                        // ever trying `method_call_result` — a generator
                        // method reaching that function instead would
                        // simply decline on its first `Stmt::Expr(Expr::
                        // Yield)` statement, the same as a bare generator
                        // def would without this arm. `generator_yields`
                        // binds its OWN `def.parameters` positionally with
                        // no `self`-awareness of its own (it is a plain-def
                        // reader, `instances.rs`'s own doc — "a generator's
                        // parameter list is bound exactly like an ordinary
                        // function's own") — `positional_arguments_for_
                        // method` already EXCLUDES `self` (the receiver is
                        // never a call argument, that function's own doc),
                        // so `self`'s own slot is prepended here with the
                        // RECEIVER value, the same binding `method_call_
                        // result` gives `self` for a non-generator method.
                        if is_generator_def(method) {
                            let mut generator_arguments = Vec::with_capacity(positional.len() + 1);
                            generator_arguments.push(receiver.clone());
                            generator_arguments.extend(positional.iter().cloned());
                            return match instances::generator_yields(method, &generator_arguments, environment.functions(), kernel, environment.call_depth()) {
                                Some(yields) => {
                                    let mut value = collection_models::list_literal_value(&yields);
                                    value.source = "generator".to_owned();
                                    value
                                }
                                None => unknown(),
                            };
                        }
                        return match instances::method_call_result(
                            &receiver,
                            model,
                            method,
                            &positional,
                            environment.functions(),
                            Some(classes),
                            environment.datetime_imports(),
                            kernel,
                            environment.call_depth(),
                        ) {
                            Some((_instance_after, result)) => result,
                            None => unknown(),
                        };
                    }
                }
            }
        }
    }
    // `reduce(function, iterable[, initializer])` — the FROM-IMPORT
    // spelling (`from functools import reduce`) of the same call the
    // `Expr::Attribute` block above recognizes as `functools.reduce(...)`.
    // The ONE call this file folds CONCRETELY step by step, per-element,
    // because `function` is read as a RAW EXPRESSION (a `Lambda` or a
    // bare `Name` naming a same-module `def`) rather than an already-
    // evaluated value the way every other call argument in this file
    // is — see `reduce_expression_value`'s own doc.
    if let Expr::Name(name) = call.func.as_ref() {
        if name.id.as_str() == "reduce" && environment.read("reduce").is_none() {
            if let Some(value) = reduce_expression_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `map(function, iterable)` / `filter(predicate, iterable)` —
        // the two other builtins this file folds CONCRETELY over a RAW
        // callable expression rather than an already-evaluated value,
        // for the same reason `reduce` does (`map_expression_value`/
        // `filter_expression_value`'s own doc). Both return a LAZY
        // iterator (functions.html#map/#filter: "Return an iterator");
        // this domain has no separate iterator Kind, so the answer is
        // the eagerly-materialized `Kind::List` of the iterator's own
        // elements — the same choice `range_expression_value` already
        // makes for `range(...)`'s own lazy sequence, and the shape
        // `list(map(...))`/`list(filter(...))` needs once `list()`
        // (`builtin_models::list_constructor_call`) copies a known
        // `Kind::List` through unchanged.
        if name.id.as_str() == "map" && environment.read("map").is_none() {
            if let Some(value) = map_expression_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        if name.id.as_str() == "filter" && environment.read("filter").is_none() {
            if let Some(value) = filter_expression_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `Exception(message)` / `ValueError(message)` / `RuntimeError(message)`
        // / `TypeError(message)` — a BUILT-IN exception class constructor
        // call (never shadowed by a same-module def/class here, the same
        // `same_module_def_gate_open` gate this whole block is already
        // inside): tutorial/errors.rst §8.3, "the exception instance...
        // typically has an `args` attribute that stores the arguments."
        // Answered as a tagged `Kind::Object` (`exception_construction_value`'s
        // own doc) carrying every positional constructor argument, in
        // order, under one `args` field — `.args[0]` (this file's own
        // `evaluate_attribute_read`'s untagged-instance fallback,
        // `instances::field_read`) and `str(...)`
        // (`builtin_models::str_call`'s new exception row) both read
        // through this ONE construction. `ExceptionGroup(msg, excs)`
        // (PEP 654, `exceptions.rst`) is a DIFFERENT shape this file does
        // not decompose (the message and wrapped exceptions are never
        // read back through a refined sink in this corpus) — answered
        // OPAQUE instead of tagged, so any read through it (this
        // function's own return value, most directly) fires the opaque
        // law rather than silently building a hollow `args` shape nothing
        // reads.
        if is_builtin_exception_constructor(name.id.as_str()) && environment.read(name.id.as_str()).is_none() {
            if name.id.as_str() == "ExceptionGroup" {
                return opaque_value("an ExceptionGroup");
            }
            if !call.arguments.keywords.is_empty() {
                return unknown();
            }
            let Some(arguments) = splice_call_arguments(&call.arguments.args, environment, kernel) else {
                return unknown();
            };
            return exception_construction_value(&arguments);
        }
        // `bytes(...)`/`bytearray(...)`/`memoryview(...)` construction —
        // p-typed-array.py's own construction band. See
        // `bytes_like_construction_value`'s own doc for every recognized
        // argument shape; `None` there means "not one of those shapes,"
        // and this call falls through to the ordinary builtin dispatch
        // below (never a hard decline at this gate alone).
        if matches!(name.id.as_str(), "bytes" | "bytearray" | "memoryview") && environment.read(name.id.as_str()).is_none() {
            if !call.arguments.keywords.is_empty() {
                return unknown();
            }
            if let Some(value) = construction::bytes_like_construction_value(name.id.as_str(), &call.arguments.args, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `datetime.datetime(year, month, day, hour=0, minute=0,
        // second=0, ..., tzinfo=...)` — recognized BEFORE the keyword
        // gate below because the fixture's own construction rows always
        // pass `tzinfo=` as a keyword argument. See
        // `datetime_construction_value`'s own doc for the exact fields
        // read and the aware-UTC-only scope.
    }
    // The three datetime CONSTRUCTION gates run against `call.func`
    // itself, BEFORE the `Expr::Attribute`-only block below: each gate
    // (`is_datetime_datetime_attribute` and its two siblings) already
    // recognizes both the qualified chain (`datetime.datetime(...)`, an
    // `Expr::Attribute` callee) AND a bare aliased class name
    // (`dt(...)`, an `Expr::Name` callee — `from datetime import
    // datetime as dt`), so trying it here covers both shapes in one
    // place rather than duplicating the bare-Name arm alongside the
    // Attribute-only recognizers further down.
    if is_datetime_datetime_attribute(call.func.as_ref(), environment) {
        if let Some(value) = datetime_construction_value(call, environment, kernel) {
            return value;
        }
        return unknown();
    }
    // `datetime.date(year, month, day)` — date.1's own construction,
    // recognized the same way `datetime.datetime(...)` is (BEFORE the
    // keyword gate below, though this construction reads no keyword
    // this file's corpus does not already handle positionally). See
    // `date_construction_value`'s own doc for the exact fields read
    // and the `calendar.validDate` kernel validation.
    if is_datetime_date_attribute(call.func.as_ref(), environment) {
        if let Some(value) = date_construction_value(call, environment, kernel) {
            return value;
        }
        return unknown();
    }
    // `datetime.timedelta(days=n)` — date.5's own construction,
    // recognized here (BEFORE the keyword gate below) because
    // `days=` always arrives as a keyword argument. See
    // `timedelta_construction_value`'s own doc for the one field
    // read and the `calendar.validDuration` kernel validation.
    if is_datetime_timedelta_attribute(call.func.as_ref(), environment) {
        if let Some(value) = timedelta_construction_value(call, environment, kernel) {
            return value;
        }
        return unknown();
    }
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        // `functools.reduce(function, iterable[, initializer])` — the
        // QUALIFIED spelling (`import functools` then
        // `functools.reduce(...)`, A7.xfer.reduce.py's own import shape)
        // of the same call the bare-Name arm below folds through
        // `reduce_expression_value`. Recognized HERE, ahead of the
        // ordinary `splice_call_arguments` dispatch further down, for the
        // identical reason the bare-Name arm runs before that dispatch:
        // `reduce_expression_value` reads `function` as a RAW unevaluated
        // expression (a `Lambda`'s own AST node) rather than an already-
        // evaluated value, and `splice_call_arguments` would evaluate it
        // first, destroying the lambda's raw form before this file could
        // fold over it — the same reason `array.array`/`datetime.date`
        // are recognized in THIS Attribute block rather than falling
        // through to the generic path below.
        if attribute.attr.as_str() == "reduce" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "functools" && environment.read("functools").is_none() {
                    if let Some(value) = reduce_expression_value(call, environment, kernel) {
                        return value;
                    }
                    return unknown();
                }
            }
        }
        // `datetime.date.fromisoformat("YYYY-MM-DD")` — a TWO-level
        // attribute chain the same way `datetime.datetime.now()` is
        // when `date` reached the file qualified (`datetime.date`), OR
        // ONE level when `date` reached it as a bare aliased class name
        // (`date.fromisoformat(...)`, `from datetime import date`) —
        // `is_datetime_date_attribute` resolves `attribute.value`
        // either way. See `date_fromisoformat_value`'s own doc for the
        // exact grammar read.
        if is_datetime_date_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "fromisoformat" {
            if let [text] = &*call.arguments.args {
                if call.arguments.keywords.is_empty() {
                    let argument = evaluate_expression(text, environment, kernel);
                    if let Some(code_points) = exact_string_values(&argument) {
                        if let Some(spelling) = code_points_to_string(code_points) {
                            if let Some(value) = date_fromisoformat_value(&spelling, kernel) {
                                return value;
                            }
                        }
                    }
                }
            }
            return unknown();
        }
        // `datetime.datetime.fromtimestamp(ts, tz=...)` — the same
        // receiver shape `datetime.datetime.now()`/`.strptime(...)`
        // read (a two-level qualified chain, or a one-level bare
        // aliased class name). Recognized HERE rather than in
        // `evaluate_attribute_call` because `tz=`'s value is read as a
        // RAW unevaluated expression (`classify_tzinfo_expression`
        // recognizes `timezone.utc`/`timezone(timedelta(...))`
        // syntactically), which the evaluated-argument dispatch cannot
        // hand over — the same reason `datetime.date.fromisoformat` and
        // `array.array` are recognized in this block.
        if is_datetime_datetime_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "fromtimestamp" {
            if let Some(value) = datetime_fromtimestamp_value(call, environment, kernel) {
                return value;
            }
            return unknown();
        }
        // `array.array(typecode, initializer)` — the Float64Array twin,
        // p-typed-array.py's `array_double_from_iterable`/`array_double_
        // write_and_read_back`. Recognized here (an Attribute call,
        // never a bare Name) the same way `datetime.datetime` is:
        // `array` imported as a bare module name (`import array`), not
        // locally shadowed.
        if attribute.attr.as_str() == "array" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "array" && environment.read("array").is_none() {
                    if let Some(value) = construction::array_double_construction_value(call, environment, kernel) {
                        return value;
                    }
                    return unknown();
                }
            }
        }
        // `subprocess.run([...], ..., capture_output=True, text=True)` —
        // tried here, alongside `array.array`, so `result`'s own binding
        // carries a `.stdout` field sort even when no `json.loads(...)`
        // consumer exists for `foreign_edge.rs` to recognize
        // (`subprocess_run_construction_value`'s own doc). A call this
        // row does not recognize (a different callee, a missing
        // `capture_output=True`/`text=True` pair) falls through to the
        // ordinary keyword-gated dispatch below unchanged — this row
        // only ever ADDS a sort to `result`, never removes one the
        // generic path would have given.
        if let Some(value) = subprocess_run_construction_value(attribute, call, environment) {
            return value;
        }
    }
    if !call.arguments.keywords.is_empty() {
        return unknown();
    }
    // `isinstance(object, classinfo)` is read BEFORE the arguments are
    // evaluated: its second argument is a TYPE expression, not a value,
    // so the ordinary argument evaluation has nothing to hand a builtin
    // row. functions.rst states the return outright — "Return ``True``
    // if the *object* argument is an instance of the *classinfo*
    // argument... If *object* is not an object of the given type, the
    // function always returns ``False``" — so the answer is always one
    // of the two values.
    if let Expr::Name(name) = call.func.as_ref() {
        if name.id.as_str() == "isinstance" && environment.read("isinstance").is_none() {
            if let [subject, classinfo] = &call.arguments.args[..] {
                let value = evaluate_expression(subject, environment, kernel);
                return isinstance_value(&value, classinfo);
            }
        }
    }
    let Some(arguments) = splice_call_arguments(&call.arguments.args, environment, kernel) else {
        return unknown();
    };
    match call.func.as_ref() {
        Expr::Name(name) if environment.read(name.id.as_str()).is_none() => {
            if name.id.as_str() == "len" {
                let [only] = arguments.as_slice() else { return unknown() };
                return match collection_models::len_result(only) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            if name.id.as_str() == "range" {
                return match range_expression_value(&arguments) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            if name.id.as_str() == "eval" {
                return match super::arithmetic::eval_literal_value(&arguments) {
                    Some(value) => value,
                    None => unknown(),
                };
            }
            match builtin_models::builtin_call_result_with_kernel(name.id.as_str(), &arguments, kernel) {
                Some(value) => value,
                None => unknown(),
            }
        }
        Expr::Attribute(attribute) => evaluate_attribute_call(attribute, &arguments, environment, kernel),
        _ => unknown(),
    }
}

/// `isinstance(object, classinfo)`'s own value — functions.rst: "Return
/// ``True`` if the *object* argument is an instance of the *classinfo*
/// argument, or of a (direct, indirect, or virtual) subclass thereof. If
/// *object* is not an object of the given type, the function always
/// returns ``False``."
///
/// Decided when BOTH the subject's own sort is known (a `Kind::Values`
/// carries it on `kind_tag`) and `classinfo` names the primitive sorts
/// `narrowing::isinstance_guards` already reads — the same
/// `isinstance_type_tags` reader that decides the GUARD form, so the
/// value form and the narrowing form agree on which spellings they
/// recognize by construction.
///
/// `bool` is a subclass of `int` (stdtypes.rst, "Boolean Type"), so a
/// Boolean subject satisfies an `int` classinfo — the "or of a subclass
/// thereof" half of the clause. The converse does not hold: an Integer
/// subject is not a `bool`.
///
/// Any shape this reader does not decide still answers the exact
/// two-member boolean domain rather than declining, since the return is
/// one of the two values whatever the arguments are.
fn isinstance_value(subject: &AbstractValue, classinfo: &Expr) -> AbstractValue {
    let decided = match (subject.kind, crate::narrowing::isinstance_type_tags(classinfo)) {
        (Kind::Values, Some(tags)) => subject.kind_tag.map(|sort| {
            tags.iter().any(|tag| {
                *tag == sort || (*tag == PrimitiveKind::Integer && sort == PrimitiveKind::Boolean)
            })
        }),
        _ => None,
    };
    match decided {
        Some(answer) => known_values(vec![if answer { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved),
        None => known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec),
    }
}
