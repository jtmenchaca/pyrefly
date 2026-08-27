//! One body's walk: build its environment from every name it locally
//! binds, then dispatch each statement in order — the module's own top
//! level, a function body, a class body. Also owns the two recognizers
//! that fold a later statement's own work into the current one
//! (a relational sum's division, a foreign edge's `json.loads` return
//! fact), since both need the same per-statement position bookkeeping
//! `walk_body_with_self_binding`'s own loop keeps.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::{known_set, AbstractValue, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use ruff_python_ast::{Parameters, Stmt};
use ruff_text_size::{Ranged, TextRange};

use refined_sets::refinement_forms::{repeat_of, requires_integer};

use crate::env::Environment;
use crate::foreign_edge;
use crate::function_table::merged;
use crate::instances;
use crate::instances::ClassModel;
use crate::relational_sum;
use crate::typereading::{base_sort_return_refinement, DeclaredRefinement};

use crate::check::{
    is_self_method, local_function_table, merged_classes_for_body, publish_relational_sum_return,
    seed_parameters, walk_method_def, walk_relational_sum, Finding, RelationalSum, WalkContext,
};

use super::{record_blocker, walk_statement, FallsThrough};

/// One body's walk: build its environment from every name it locally
/// binds, then dispatch each statement in order. `parameters` seeds a
/// function body's own parameter names into that locally-bound set (a
/// parameter shadows an outer alias exactly as a rebinding would) and,
/// where a parameter's annotation reads, seeds its INITIAL value too
/// (assignability.rs's DeclaredRefinement, at TrustSpec — an annotation
/// states the developer's claim, not an execution-proved fact);
/// `parameters: None` is the module's own top-level body, which has
/// neither. `return_refinement` is the enclosing function's own
/// `-> Annotation` read through `declared_refinement`, threaded down so
/// every `return value` in this body (not in a nested `def`, which
/// reads its own) judges against it; `None` when the function has no
/// return annotation, or this is the module body — ordinary Python,
/// nothing to judge returns against. `blocked` tracks whether this
/// body has already recorded its one RTS7002 — set true the moment the
/// first unwalkable construct is seen, and never reset within this
/// body.
///
/// BODY-LOCAL CLASSES: `class_table`'s own module-level scan
/// (`instances.rs`) reads only `module.body`'s own top-level
/// `StmtClassDef`s, so a class defined INSIDE a function body (or a
/// method body) is invisible to `context.classes` — the same gap
/// `local_function_table` already closes for a body-local `def`.
/// `local_class_table(body, ...)` mirrors that construction (wrapping
/// this body's own top-level `Stmt::ClassDef`s in a synthetic
/// `ModModule` and reusing `instances::class_table`'s one public
/// constructor) and is merged over `context.classes` here, local name
/// winning on a spelling collision — the same base-wins rule
/// `function_table::merged` already applies. A class nested inside one
/// of THIS body's own local classes is not itself walked as a
/// top-level entry (the same one-level rule `local_function_table`
/// keeps for a nested `def`).
///
/// SELF-SEEDING: `self_model` is `Some(class)` only when this body IS a
/// class body being walked by `walk_class_def` for `class` itself —
/// `None` everywhere else (a plain function, the module body, or any
/// nested body reached through `walk_statement`'s ordinary recursion).
/// When `Some`, this function's own per-statement loop below walks a
/// top-level member `def` whose first parameter is `self` through
/// `walk_method_def` instead of the ordinary `walk_statement` dispatch,
/// seeding `self` with an instance built from `class`'s own declared
/// fields (datamodel.rst, "Instance methods": "the special thing about
/// methods is that the instance object is prepended to the argument
/// list" — the receiver is a real value at every method call, so a
/// body that reads `self.<field>` before typereading can prove which
/// concrete instance called it still has SOMETHING sound to read: the
/// class's own declared shape). Every other statement (a class-body
/// `AnnAssign` field, a nested `class`, an `if`, …) still walks through
/// the ordinary `walk_statement` dispatch, unaffected.
pub(in crate::check) fn walk_body(
    body: &[Stmt],
    parameters: Option<&Parameters>,
    return_refinement: Option<&DeclaredRefinement>,
    self_model: Option<&ClassModel>,
    context: &WalkContext,
    out: &mut Vec<Finding>,
) {
    walk_body_with_self_binding(body, parameters, return_refinement, None, self_model, None, None, None, None, context, out);
}

/// The statement position whose walk must publish a recognized foreign
/// edge's return fact — the STATEMENT HOLDING the edge's own
/// `json.loads(...)` node, found by RANGE CONTAINMENT among the
/// statements after the call at `call_position` (the sole consumer may
/// sit any number of statements later, never assumed to be the very
/// next one). `None` when no later statement in `body` contains
/// `parse_range` at all — the edge recognized a crossing call whose
/// result is never actually parsed anywhere in this body, so there is
/// no position to key an entry under and the recognition publishes
/// nothing.
///
/// Pulled out of `walk_body_with_self_binding`'s per-statement loop so
/// the position math is independently testable: two recognized calls in
/// the same body resolve to two INDEPENDENT positions here (this
/// function reads `body` fresh each call and returns a plain `usize`),
/// which is what lets the caller key `foreign_edge_overrides` per call
/// rather than through one shared slot.
pub(in crate::check) fn foreign_edge_consumer_position(body: &[Stmt], call_position: usize, parse_range: TextRange) -> Option<usize> {
    body.iter()
        .enumerate()
        .skip(call_position + 1)
        .find(|(_, following)| following.range().contains_range(parse_range))
        .map(|(index, _)| index)
}

/// Tries `foreign_edge::foreign_edge_at` at `position` in `body` — the
/// SAME `Stmt::Assign`/`Stmt::With` gate and outcome handling
/// `walk_body_with_self_binding`'s own per-statement loop already runs,
/// pulled out so every statement list the checker walks (a function's own
/// top-level body, an `if`-arm's body, a `with`-block's body) offers this
/// recognition the same way, never a second mechanism. A recognized
/// `Override` is keyed into `foreign_edge_overrides` under its own
/// consumer position (`foreign_edge_consumer_position`'s own doc); a
/// `Fired` records an RTS7001 directly; a `Decline` records this body's
/// one blocker. A statement that is not this shape, or that
/// `foreign_edge_at` does not recognize at all, is a no-op — the caller's
/// own walk of `stmt` is untouched either way.
pub(in crate::check) fn serve_foreign_edge_at(
    body: &[Stmt],
    position: usize,
    environment: &Environment,
    context: &WalkContext,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
    foreign_edge_overrides: &mut HashMap<usize, Vec<(TextRange, AbstractValue)>>,
) {
    if !matches!(body[position], Stmt::Assign(_) | Stmt::With(_)) {
        return;
    }
    let Some(outcome) =
        foreign_edge::foreign_edge_at(body, position, environment, context.kernel, context.entry_directory.as_deref())
    else {
        return;
    };
    match outcome {
        foreign_edge::ForeignEdgeOutcome::Override { parse_range, value, stdout_override } => {
            // The override applies at the STATEMENT HOLDING the
            // `json.loads(...)` node — the sole consumer may sit any
            // number of statements after the call, so find it by range
            // containment rather than assuming a position. Keyed by
            // that consumer's own position: an earlier still-pending
            // entry for a DIFFERENT consumer position is untouched by
            // this insert, so two recognized calls before either is
            // consumed each keep their own entry.
            if let Some(consumer_position) = foreign_edge_consumer_position(body, position, parse_range) {
                let mut published = vec![(parse_range, value)];
                // `stdout_override`'s own node (a `result.stdout`
                // attribute access, or the bound name's own node) is a
                // SUB-NODE of the SAME `json.loads(...)` statement
                // `parse_range` already resolved a position for, so it
                // always shares that position — `foreign_edge_consumer_
                // position` is not asked a second time.
                if let Some((stdout_range, stdout_value)) = stdout_override {
                    published.push((stdout_range, stdout_value));
                }
                foreign_edge_overrides.insert(consumer_position, published);
            }
        }
        foreign_edge::ForeignEdgeOutcome::Fired { message, range, consumer } => {
            out.push(Finding { range, code: "RTS7001", message });
            // The TS convention taken literally: an outbound fire and a
            // bound return fact are independent truths, so the sole
            // consumer of a fired FileRead edge still binds the
            // artifact's own return-leg value (carried on the Fired
            // outcome, built by the same reader the serving path uses)
            // and judges it for real — one edge, its fire, and a
            // determined read.
            if let Some((consumer_range, value)) = consumer {
                if let Some(consumer_position) = foreign_edge_consumer_position(body, position, consumer_range) {
                    foreign_edge_overrides.insert(consumer_position, vec![(consumer_range, value)]);
                }
            }
        }
        foreign_edge::ForeignEdgeOutcome::Decline { message, range } => {
            record_blocker(blocked, range, message, out);
        }
    }
    // recognized whole (an override, a fire, or a named decline) — the
    // statement itself still walks ordinarily afterward so `<name>`
    // binds through the usual Assign path; only the LATER `json.loads`
    // node's value is overridden.
}

/// `walk_body`'s full construction, plus one extra optional step:
/// `self_binding`, when `Some`, binds the name `self` to that value
/// AFTER parameter seeding — `walk_method_def`'s own seam into this
/// function, so a `self.<field>` read inside a method body reaches
/// `evaluate_attribute_read`'s tagged-instance path
/// (`instances::field_read_through_model`) instead of reading an
/// unbound name. `self` carries no annotation in the corpus's own
/// convention, so `seed_parameters` never seeds it itself — this bind
/// is the only writer for that name at body entry.
///
/// `returned_values_out`, when `Some`, is filled with every value this
/// body's `return` statements produced — the SAME values `walk_return`
/// judges, recorded through `env::collect_returned_values` rather than
/// re-derived. `fact_export` is the only caller that asks; every walk in
/// the checker itself passes `None` and behaves exactly as before.
///
/// `enclosing_def_name`, when `Some`, is the MODULE-LEVEL def this body
/// belongs to — `seed_parameters`'s own key into `context.caller_arguments`
/// for an unannotated-parameter caller join. `None` for the module body
/// itself and for a nested/method def (neither is a module-level def
/// `function_table::caller_argument_positions` ever indexes), which
/// simply seeds no caller-joined parameter, matching today's plain
/// Unknown behavior.
///
/// `bare_sort_return_refinement`, when `Some`, is `typereading::
/// base_sort_return_refinement` read off this def's own `-> Annotation`
/// (a bare `int`/`float`/`str`, no `Annotated[...]` refinement) —
/// `None` for every other return shape, including no annotation at all.
/// This is NEVER `return_refinement` itself (a bare-sort return still
/// judges nothing at an ORDINARY return, exactly as today — widening
/// that would turn every unreadable `-> int` helper body into a new
/// undetermined blocker, `typereading::base_sort_return_refinement`'s own
/// doc). It exists ONLY so the per-statement loop below can judge a
/// RECOGNIZED foreign-edge crossing's return leg against the sort even
/// when no refinement narrows it — the one place a bare sort is worth
/// something (sort admission and refinement wideness against a fact the
/// crossing itself proved), read at exactly the position `foreign_edge_
/// overrides` already names, never at any other return in this body.
#[allow(clippy::too_many_arguments)]
pub(in crate::check) fn walk_body_with_self_binding(
    body: &[Stmt],
    parameters: Option<&Parameters>,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    self_model: Option<&ClassModel>,
    self_binding: Option<&AbstractValue>,
    returned_values_out: Option<&mut Vec<AbstractValue>>,
    enclosing_def_name: Option<&str>,
    bare_sort_return_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    out: &mut Vec<Finding>,
) {
    let mut locally_bound = super::locally_bound_names(body);
    if let Some(parameters) = parameters {
        super::collect_parameter_names(parameters, &mut locally_bound);
    }
    let mut environment = Environment::new(locally_bound);
    // Declared BEFORE `seed_parameters` runs (rather than beside the
    // straight-line statement loop below, where an AnnAssign-only table
    // would suffice) so a PARAMETER's own declared refinement can be
    // recorded into it too — `seed_parameters`'s own insert, mirroring
    // `walk_ann_assign`'s identical insert for a body-local `x: Age =
    // ...` target. A LATER `x += 1`/`x = 200` against either kind of
    // declared target then judges against the same one table.
    let mut aug_assign_refinements: HashMap<String, DeclaredRefinement> = HashMap::new();
    if returned_values_out.is_some() {
        environment.collect_returned_values();
    }
    // Shares the module walk's ONE evaluations recorder (if the caller
    // asked for one) onto THIS body's fresh environment — the seam
    // `refined_set_at_position` depends on: a nested `def`'s own body
    // gets its own fresh `Environment` here, so without this the
    // recordings from every body but the outermost would be lost.
    if let Some(recorder) = context.evaluations_recorder.clone() {
        environment.set_evaluations_recorder(recorder);
    }
    // The same sharing, for the same reason, for the derivation-trace
    // collector: a nested `def`'s own body builds a fresh `Environment`
    // here, and the blocked position the caller asked about may sit
    // inside it.
    if let Some(collector) = context.trace_collector.clone() {
        environment.set_trace_collector(collector);
    }
    environment.set_functions(Arc::new(merged(&local_function_table(body), &context.functions)));
    environment.set_classes(merged_classes_for_body(body, context));
    environment.set_declared_aliases(Arc::new(context.aliases.clone()), Arc::new(context.imports.clone()));
    environment.set_datetime_imports(context.datetime_imports.clone());
    environment.set_locale_never_set(context.locale_never_set);
    if let Some(entry_directory) = context.entry_directory.clone() {
        environment.set_entry_directory(Arc::new(entry_directory));
    }
    // Every module-level binding (this module's own top-level constants
    // AND every import statement's resolved value) is readable here
    // UNLESS this body itself rebinds the name — a local rebinding
    // shadows the module value, the same rule `alias_is_visible` already
    // applies to the alias table.
    for (name, value) in &context.module_bindings {
        if environment.alias_is_visible(name) {
            environment.bind(name, value.clone());
        }
    }
    // Every visible CLASS name seeds its class-object value too — the
    // shadow-on-rebind rule module_bindings takes — so a function body's
    // `Counted.total = 200` write and `Counted.total` read see the class
    // object without a construction anywhere in the body. Calling the
    // seeded name still constructs: the construction gates recognize a
    // name bound to its OWN class object (source == the class name).
    {
        let class_names: Vec<String> = environment
            .classes()
            .map(|classes| classes.keys().cloned().collect())
            .unwrap_or_default();
        for name in class_names {
            if environment.alias_is_visible(&name) && environment.read(&name).is_none() {
                let model = environment
                    .classes()
                    .and_then(|classes| classes.get(&name))
                    .expect("name came from this same table");
                let value = instances::class_object_value(model);
                environment.bind(&name, value);
            }
        }
    }
    if let Some(parameters) = parameters {
        seed_parameters(parameters, enclosing_def_name, context, &mut environment, &mut aug_assign_refinements);
        // `*args`/`**kwargs`'s own names — a bare-Name forward of either
        // (`f(*args)`, `f(**kwargs)`) hands CPython exactly what THIS
        // body itself received, never an independently-grown collection
        // (`expressions.rs::call_provable_raise`'s own "unbounded
        // starred argument" check reads this set to stay silent on a
        // ParamSpec-forwarding row like r-ast-census.py's `wrapper`).
        let mut variadic_names = HashSet::new();
        if let Some(vararg) = parameters.vararg.as_ref() {
            variadic_names.insert(vararg.name.id.as_str().to_owned());
            // `*rest: int` collects zero or more `int` arguments into a
            // tuple CPython builds at call time (functions.rst, "if the
            // syntax `*identifier` is present") — the same unbounded-
            // length shape `list[X]`/`Sequence[X]` seeds below
            // (`is_sequence_container`'s own repetition window), built
            // from the SCALAR sort `base_sort_return_refinement` reads
            // off the vararg's own bare annotation rather than a
            // container annotation (a vararg is never itself spelled
            // `list[int]`). `iterable_values`/`repetition_window_element_
            // pass` (loops.rs) already read this exact `Kind::Set`-over-
            // `repeat_of` shape for `for value in rest:`, so seeding it
            // here is what lets a body walked WITHOUT a call site (this
            // def's own straight-line walk, not `summaries::bind_
            // parameters`'s call-site tuple) iterate its own rest
            // parameter. An annotation this reader does not recognize
            // (a non-`int`/`float`/`str` bare name, or none at all)
            // leaves the vararg unseeded, unchanged from before this
            // law — an unrecognized element sort states nothing sound
            // to repeat.
            if let Some(annotation) = vararg.annotation.as_deref() {
                if let Some(declared) = base_sort_return_refinement(annotation) {
                    let sort = if requires_integer(&declared.set) { PrimitiveKind::Integer } else { PrimitiveKind::Float };
                    let sequence = AbstractValue {
                        kind_tag: Some(sort),
                        ..known_set(refined_sets::refinement_forms::make_refined_set(vec![repeat_of(declared.set, 0, None)]), None, TrustSpec, SetKindTag::None)
                    };
                    environment.bind(vararg.name.id.as_str(), sequence);
                }
            }
        }
        if let Some(kwarg) = parameters.kwarg.as_ref() {
            variadic_names.insert(kwarg.name.id.as_str().to_owned());
        }
        environment.set_variadic_parameter_names(Arc::new(variadic_names));
    }
    if let Some(self_value) = self_binding {
        environment.bind("self", self_value.clone());
    }
    // This body's own CALLABLE-RETURN table, seeded from the module's
    // top-level callable declarations — the same shadow-on-rebind rule
    // `module_bindings` above takes (a body that locally rebinds the
    // name is not seeded with the module-level entry). `walk_ann_assign`
    // grows this table as a body-local `Callable[...]`-typed variable is
    // walked, republishing it onto `environment` itself (rather than a
    // sibling parameter threaded through every statement form the way
    // `aug_assign_refinements` is) so `sink_value`'s call-site read —
    // reachable from every nested branch/loop/match/with/try arm through
    // `environment` alone — sees each new entry as soon as it is walked,
    // with no signature change anywhere along that dispatch tree.
    let module_callable_returns: HashMap<String, DeclaredRefinement> = context
        .module_callable_returns
        .iter()
        .filter(|(name, _)| environment.alias_is_visible(name))
        .map(|(name, declared)| (name.clone(), declared.clone()))
        .collect();
    if !module_callable_returns.is_empty() {
        environment.set_callable_returns(Arc::new(module_callable_returns));
    }
    let mut blocked = false;
    // PROVABLY-UNBOUND READS: every name this straight-line walk has seen
    // declared by a VALUELESS AnnAssign (`x: int`) with no assignment
    // observed since. `walk_statement`'s own `If`/`For`/`While`/`Match`/
    // `With`/`Try`/blocker arms clear this wholesale the moment the walk
    // crosses anything that could bind a name on some path without this
    // loop seeing it directly, so the set only ever names a name that is
    // PROVABLY still unbound along the one path CPython actually ran.
    let mut provably_unbound: HashSet<String> = HashSet::new();
    // The positions of the statements a relational sum already folded
    // into its kernel program — the division alone, or a count-alias
    // assignment plus the division that reads it — as a half-open
    // range. `position+1..position+1` (empty) while there is none,
    // since no statement position falls inside an empty range.
    let mut folded_division_at = usize::MAX..usize::MAX;
    // Every foreign-edge return fact still waiting to be published,
    // keyed by the POSITION of the statement holding its own
    // `json.loads(...)` node — never a single slot. Unlike a relational
    // sum's quotient (always the very next statement), the sole
    // `json.loads` consumer a foreign edge finds may sit several
    // statements after the call, so each entry rides across more than
    // one loop iteration rather than being cleared unconditionally
    // after one walk. A body can make more than one recognized crossing
    // call before either result is consumed (the diamond shape: two
    // `subprocess.run` calls, then two `json.loads` reads) — with a
    // single slot, the second recognition would clobber the first's
    // still-pending entry before its own consumer is ever reached, so
    // the map keeps them independently: one entry per consumer
    // position, applied via `environment.set_evaluated_node` only at
    // its own matching `position`, and removed the moment that
    // position's walk runs (an entry whose consumer position never
    // recurs — the same body never reaches it again — simply stays in
    // the map unconsumed and unread, exactly as inert as the single
    // slot's own unconsumed `None` was).
    let mut foreign_edge_overrides: HashMap<usize, Vec<(TextRange, AbstractValue)>> = HashMap::new();
    for (position, stmt) in body.iter().enumerate() {
        if let (Some(class), Stmt::FunctionDef(def)) = (self_model, stmt) {
            if is_self_method(def) {
                walk_method_def(def, class, context, out);
                continue;
            }
        }
        // RELATIONAL SUM: an accumulation over a sequence known only by
        // its element set — either spelling, `total = 0; for x in xs:
        // total += f(x)` or `total = sum(f(x) for x in xs)` — followed by
        // a division of that total by the same sequence's length.
        // Interval division answers this far too weakly (`[0, n] / [1,
        // n]` is `[0, n]`), so the accumulation and the division lower
        // into ONE kernel program where the linear decider ties the
        // total to the count. Recognized here rather than inside
        // `walk_loop` because the division sits in the FOLLOWING
        // statement, which only this statement driver can see. Declining
        // leaves the statement to `walk_statement` exactly as before.
        // A folded division was already walked as part of the kernel
        // program, so the statement itself is skipped rather than
        // re-walked into a second, weaker binding of the same name.
        if folded_division_at.contains(&position) {
            continue;
        }
        // FOREIGN EDGE: `<name> = subprocess.run(["node", "<script>.ts"],
        // input=json.dumps(...), capture_output=True, text=True)` followed
        // by `json.loads(<name>.stdout)` — a cross-language call whose
        // argv NAMES the TypeScript code that runs next, so the checker
        // reads that target's own exported fact and attaches it to the
        // parse node. Tried BEFORE `recognize_generator_sum` below: both
        // recognize a shape at an `Assign`, and the foreign edge's own
        // shape (a `subprocess.run` call) is the more specific of the two,
        // so it must not be shadowed by a generator-sum match that could
        // never apply to a `subprocess.run` value anyway — the fixed
        // order this mission calls for once two Assign-shaped two-
        // statement recognizers coexist. A `With` statement also enters:
        // the temp-file carrier's unit starts at the tempfile with-block
        // (recognize_temp_file_edge), not at an Assign.
        serve_foreign_edge_at(body, position, &environment, context, &mut blocked, out, &mut foreign_edge_overrides);
        let recognized = match stmt {
            Stmt::For(for_stmt) if for_stmt.orelse.is_empty() => {
                relational_sum::recognize_accumulation(for_stmt, &environment)
                    .map(|recognized| (recognized, Some(for_stmt.target.as_ref()), None))
            }
            Stmt::Assign(assign) => relational_sum::recognize_generator_sum(assign, &environment)
                .or_else(|| relational_sum::recognize_sum_over_name(assign, &environment))
                .map(|recognized| (recognized, None, Some(assign.range()))),
            _ => None,
        };
        if let Some((recognized, loop_target, bound_at)) = recognized {
            match walk_relational_sum(
                recognized,
                loop_target,
                bound_at,
                &body[position + 1..],
                &mut environment,
            ) {
                RelationalSum::Declined => {}
                RelationalSum::Consumed => continue,
                RelationalSum::ConsumedWithDivision { skip_statements } => {
                    folded_division_at = (position + 1)..(position + 1 + skip_statements);
                    continue;
                }
            }
        }
        // A foreign edge's return fact is published for the STATEMENT
        // HOLDING its `json.loads(...)` node, wherever that sits later in
        // this body — looked up by THIS position, published immediately
        // before that one statement's walk, and removed from the map
        // right after (the same scoping obligation a relational sum's
        // quotient keeps for the very next statement, widened to
        // whichever position matches). A different pending entry, keyed
        // under a different position, is untouched here — each entry is
        // published only at its own consumer, never at another one's.
        let at_recognized_crossing = foreign_edge_overrides.contains_key(&position);
        if let Some(published) = foreign_edge_overrides.get(&position) {
            environment.set_evaluated_node(published.clone());
        }
        // RELATIONAL SUM AT A BARE RETURN: `return sum(<elt> for <var>
        // in <seq>)` with no assignment in the body at all — the
        // generator-sum recognizer above only ever reads an `Assign`, so
        // this single-statement spelling needs its own publish before
        // the return walks. A decline publishes nothing and the return
        // below evaluates exactly as it already did.
        if let Stmt::Return(ret) = stmt {
            publish_relational_sum_return(ret, &mut environment);
        }
        // BARE-SORT RETURN AT A RECOGNIZED CROSSING: `return_refinement`
        // stays `None` for every ORDINARY bare-sort return (unchanged —
        // the general table deliberately does not read base sorts), but
        // THIS statement is the one a recognized foreign edge just
        // published its own proved fact onto (`at_recognized_crossing`),
        // so the sort itself is worth judging here: a declared `-> float`
        // still refuses a value the crossing's own fact places outside
        // the float ray (there is none — `numbers()` is the whole real
        // line — but `-> int`/`-> str` genuinely narrow), and a widened
        // corner the crossing's own `Optional` fact admits still passes.
        // No other return in this body is affected: the fallback applies
        // ONLY at this one position, and ONLY when `return_refinement`
        // itself named nothing.
        let effective_return_refinement = if at_recognized_crossing && return_refinement.is_none() {
            bare_sort_return_refinement
        } else {
            return_refinement
        };
        let falls_through = walk_statement(
            stmt,
            effective_return_refinement,
            yield_refinement,
            context,
            &mut environment,
            &mut aug_assign_refinements,
            &mut provably_unbound,
            &mut blocked,
            out,
        );
        // A relational sum publishes its quotient for the NEXT
        // statement's one division node; a foreign edge's own publish
        // (above) is likewise scoped to exactly the one statement whose
        // walk just ran — either way, the publication ends here and no
        // later node can match it.
        environment.set_evaluated_node(Vec::new());
        // This position's own entry (if it had one) expires the moment
        // its statement's walk runs — the same way the single slot's
        // `None` reset did, widened to a per-key removal so a DIFFERENT
        // still-pending entry (a later consumer this walk has not
        // reached yet) stays in the map rather than being cleared
        // wholesale.
        foreign_edge_overrides.remove(&position);
        // A `try` whose every arm provably raises or returns leaves
        // nothing that reaches whatever follows it in THIS body — the
        // same "unreachable code past a terminal statement" rule
        // `arm_terminates` already applies to a bare `return`/`raise`,
        // extended here to a try/except construct whose own termination
        // is proved rather than syntactic. Stopping here is what keeps a
        // read past the try from reporting an unreadable-value blocker
        // for a name only the (unreachable) fall-through path would have
        // left unbound.
        // A statement that provably never falls through ends this body's
        // walk, whichever of the two reasons carried it: a try whose arms
        // all terminate, or an if whose test proved true and whose arm
        // terminates. In the second case whatever follows is provably
        // unreachable — but no corpus row designates an unreachable
        // STATEMENT (the dead-code convention designates the CONDITION,
        // the sink.dead rows' own shape), so the walk stops without
        // reporting rather than landing a true determination at a
        // position no designation covers.
        if falls_through != FallsThrough::Yes {
            break;
        }
    }
    if let Some(returned_values_out) = returned_values_out {
        *returned_values_out = environment
            .returned_values()
            .expect("the recorder was installed above whenever this slot is Some");
    }
}
