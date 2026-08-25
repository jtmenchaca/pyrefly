use std::collections::{HashMap, HashSet};

use refined_domain::abstract_value::{AbstractValue, Kind};
use ruff_python_ast::{BoolOp, CmpOp, Expr, Stmt, StmtIf, StmtReturn};
use ruff_text_size::{Ranged, TextRange};

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::foreign_edge;
use crate::narrowing::assume;
use crate::relational_sum;
use crate::typereading::DeclaredRefinement;

use super::*;

/// The one walrus-bound `subprocess.<callee>(...)` call reachable inside
/// an `if`/`elif` TEST, if any — `(target, call)`, the same
/// `Expr::Named::target`/`value` pair `bind_walrus_targets` already
/// destructures, read again here rather than threaded through (this
/// walk is a second, cheap traversal of a pure expression tree, the
/// same posture `bind_walrus_targets`'s own doc already takes for its
/// value-evaluation pass). Mirrors `bind_walrus_targets`'s own recursion
/// shape so a walrus nested anywhere in the test (not only at the top)
/// is found the same way it is already bound; stops at the first
/// `Expr::Named` whose value is an `Expr::Call`, since a walrus can only
/// ever bind once per test in the corpus's own idiom
/// (`level_via_walrus_result`).
pub(super) fn walrus_bound_call_in_test(expr: &Expr) -> Option<(&ruff_python_ast::ExprName, &ruff_python_ast::ExprCall)> {
    match expr {
        Expr::Named(named) => {
            if let (Expr::Name(target), Expr::Call(call)) = (named.target.as_ref(), named.value.as_ref()) {
                return Some((target, call));
            }
            walrus_bound_call_in_test(named.value.as_ref())
        }
        Expr::BoolOp(op) => op.values.iter().find_map(walrus_bound_call_in_test),
        Expr::BinOp(op) => walrus_bound_call_in_test(op.left.as_ref()).or_else(|| walrus_bound_call_in_test(op.right.as_ref())),
        Expr::UnaryOp(op) => walrus_bound_call_in_test(op.operand.as_ref()),
        Expr::Compare(compare) => walrus_bound_call_in_test(compare.left.as_ref())
            .or_else(|| compare.comparators.iter().find_map(walrus_bound_call_in_test)),
        Expr::Attribute(attribute) => walrus_bound_call_in_test(attribute.value.as_ref()),
        _ => None,
    }
}

/// Offers `arm_body`'s own consumer scan the same foreign-edge
/// recognition a plain `Assign`/`With` statement already gets, for a
/// walrus-bound `subprocess.<callee>(...)` call living in the `if`
/// TEST rather than in the arm body itself
/// (`level_via_walrus_result`'s own shape). Returns the per-consumer-
/// position override map the arm body's own statement loop applies —
/// the SAME map shape and publish/expire discipline
/// `serve_foreign_edge_at`'s callers already keep, so no second
/// mechanism exists for how an override reaches its consumer once
/// recognized.
pub(super) fn serve_foreign_edge_in_walrus_test(
    test: &Expr,
    arm_body: &[Stmt],
    environment: &Environment,
    context: &WalkContext,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) -> HashMap<usize, Vec<(TextRange, AbstractValue)>> {
    let mut foreign_edge_overrides = HashMap::new();
    let Some((target, call)) = walrus_bound_call_in_test(test) else {
        return foreign_edge_overrides;
    };
    let Some(outcome) = foreign_edge::foreign_edge_at_walrus_call(
        call,
        target,
        arm_body,
        0,
        environment,
        context.kernel,
        context.entry_directory.as_deref(),
    ) else {
        return foreign_edge_overrides;
    };
    match outcome {
        foreign_edge::ForeignEdgeOutcome::Override { parse_range, value, stdout_override } => {
            if let Some(consumer_position) = arm_body
                .iter()
                .enumerate()
                .find(|(_, statement)| statement.range().contains_range(parse_range))
                .map(|(position, _)| position)
            {
                let mut published = vec![(parse_range, value)];
                // `stdout_override`'s own node sits inside the SAME
                // `json.loads(...)` statement `parse_range` already
                // resolved a position for (`serve_foreign_edge_at`'s own
                // doc on this point).
                if let Some((stdout_range, stdout_value)) = stdout_override {
                    published.push((stdout_range, stdout_value));
                }
                foreign_edge_overrides.insert(consumer_position, published);
            }
        }
        foreign_edge::ForeignEdgeOutcome::Fired { message, range, consumer } => {
            out.push(Finding { range, code: "RTS7001", message });
            // The TS convention taken literally: the fired FileRead
            // edge's sole consumer still binds the artifact's carried
            // return-leg value and judges it for real — one edge, its
            // fire, and a determined read.
            if let Some((consumer_range, value)) = consumer {
                if let Some(consumer_position) = arm_body
                    .iter()
                    .enumerate()
                    .find(|(_, statement)| statement.range().contains_range(consumer_range))
                    .map(|(position, _)| position)
                {
                    foreign_edge_overrides.insert(consumer_position, vec![(consumer_range, value)]);
                }
            }
        }
        foreign_edge::ForeignEdgeOutcome::Decline { message, range } => {
            record_blocker(blocked, range, message, out);
        }
    }
    foreign_edge_overrides
}

/// `if test: body [elif test: body ...] [else: body]`
/// (compound_stmts.rst, "The `if` statement"): each arm forks the
/// incoming environment, narrows it by `assume` (the concurrent
/// narrowing unit; today's passthrough default is sound — every arm's
/// fork simply carries the unnarrowed environment forward, which is
/// conservative, never wrong), and walks that arm's body. `elif` is
/// read as CPython reads it: nested `if` in the `else` slot, arm by
/// arm, left to right. An arm whose LAST statement is a bare
/// `return`/`raise` does not rejoin — that arm's fall-through state is
/// unreachable, so only surviving arms' environments join. When every
/// arm falls through this way, the post-if environment is the incoming
/// one (nothing narrowed survives to describe unreachable code).
///
/// DEAD-BRANCH LAW (serves a-statements:400 —
/// `none_test_on_helper_that_never_answers_none`): for each arm that
/// carries a test, the test expression is evaluated first and read
/// through `truthiness`. A test PROVABLY FALSE (`known && !value`)
/// fires an RTS7001 at the test's own range ("this condition is
/// provably false on every run") and that arm's body is never walked —
/// it contributes no surviving environment, since CPython itself never
/// runs it. A test PROVABLY TRUE (`known && value`) walks only that
/// arm, and every LATER arm (including any final `else`) is
/// unreachable — CPython never evaluates a later test once an earlier
/// one is known true — so no later arm is walked and none contributes
/// any finding; the post-if environment is that one arm's alone, and
/// this function returns immediately. An UNKNOWN test (the ordinary
/// case) keeps exactly today's behavior: forked, narrowed, walked, and
/// joined alongside every other surviving arm. Only a provably-FALSE
/// test ever fires — a provably-true test states nothing wrong, so it
/// never does.
///
/// EXCEPTION (`is_admits_none_peel_test`, serves f-type-nodes.py's
/// `optional_annotation`/`pipe_none_annotation`): a provably-false test
/// is never fired when it is the ordinary Optional-PEELING idiom — `if
/// <name> is None:` / `if <name> is not None:` where `<name>` carries a
/// DECLARED refinement that `admits_none` (`Optional[Age]`, `Age | None`).
/// `present: Optional[Age] = 40` then `if present is None:` evaluates the
/// concrete literal 40 against `is None` and comes back provably false —
/// but that falseness is a fact about THIS ONE assignment's own concrete
/// value, not about the DECLARED shape `present` states it carries at
/// every point downstream; peeling a `| None`/`Optional[...]` declaration
/// with an `is None`/`is not None` guard is ordinary, idiomatic narrowing
/// (the same idiom every `X | None` field/parameter read leans on), never
/// dead code, so the dead-branch law must not speak here. This is a
/// narrow, syntactic exception: `none_test_on_helper_that_never_answers_
/// none`'s own row (a-statements.py:400ish) is unaffected because `held`
/// there is bound by a plain `Assign` from a call result, never an
/// AnnAssign, so it carries no entry in `aug_assign_refinements` at all —
/// the exception's own `declared.admits_none` check simply finds nothing
/// and the dead-branch law still fires for it.
pub(super) fn walk_if(
    if_stmt: &StmtIf,
    return_refinement: Option<&DeclaredRefinement>,
    yield_refinement: Option<&DeclaredRefinement>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let mut arms: Vec<(Option<&Expr>, &[Stmt])> = Vec::new();
    arms.push((Some(if_stmt.test.as_ref()), if_stmt.body.as_slice()));
    for clause in &if_stmt.elif_else_clauses {
        arms.push((clause.test.as_ref(), clause.body.as_slice()));
    }

    // The PATH environment: the state in which the NEXT arm's test is
    // evaluated. An arm's body runs only when its test is true, and the
    // next test is evaluated only when this one's was false — so after
    // each tested arm, the path assumes that test false. This is what
    // makes an early-exit arm narrow the fall-through (`if x < k:
    // return` leaves x ∈ [k, ∞) below) and an elif arm carry the
    // earlier tests' complements.
    let mut path = environment.fork();
    let mut surviving: Vec<Environment> = Vec::new();
    // Set when an arm's test proves TRUE under the path (every earlier
    // test already assumed false): no later arm and no fall-through can
    // run, so the post-if state is the join of the surviving arms alone.
    let mut chain_exhausted = false;
    for (test, body) in &arms {
        if let Some(test) = test {
            let test_value = evaluate_expression(test, &mut path, context.kernel);
            // WALRUS BINDING: `if (age := 40) > 0:` binds `age` into the
            // path BEFORE any arm forks from it — CPython evaluates the
            // test (and its own walrus assignment) once, regardless of
            // which arm the truth value takes, so the bound name is
            // visible both inside the taken arm's body and after the
            // whole `if` (a-statements.py's `walrus_in_condition`).
            bind_walrus_targets(test, context, aug_assign_refinements, &mut path, out);
            let (truthy, known) = refined_domain::lattice_operations::truthiness(&test_value);
            if known && !truthy && !is_admits_none_peel_test(test, aug_assign_refinements) {
                out.push(Finding {
                    range: test.range(),
                    code: "RTS7001",
                    message: "this condition is provably false on every run".to_owned(),
                });
                continue;
            }
            if known && truthy {
                let mut arm_environment = path.fork();
                arm_environment = assume(test, arm_environment, context.kernel, true);
                // A fresh, empty PROVABLY-UNBOUND-READS set per arm body:
                // the outer body's own set was already cleared by
                // `walk_statement`'s `Stmt::If` arm before this function was
                // called, and an arm body's own valueless AnnAssigns start
                // this fresh scan over again — sound because a name this
                // arm itself declares valueless and never assigns is still
                // exactly as provably unbound as it would be in a straight-
                // line body.
                let mut arm_provably_unbound: HashSet<String> = HashSet::new();
                // FOREIGN EDGE: a walrus-bound `subprocess.<callee>(...)`
                // call living in THIS test (`level_via_walrus_result`'s own
                // shape) is offered to the same recognition an ordinary
                // `Assign`/`With` statement gets — the arm body is where
                // its sole `json.loads(...)` consumer sits.
                let mut foreign_edge_overrides =
                    serve_foreign_edge_in_walrus_test(test, body, &arm_environment, context, blocked, out);
                for (position, stmt) in body.iter().enumerate() {
                    if let Some(published) = foreign_edge_overrides.get(&position) {
                        arm_environment.set_evaluated_node(published.clone());
                    }
                    walk_statement(
                        stmt,
                        return_refinement,
                        yield_refinement,
                        context,
                        &mut arm_environment,
                        aug_assign_refinements,
                        &mut arm_provably_unbound,
                        blocked,
                        out,
                    );
                    arm_environment.set_evaluated_node(Vec::new());
                    foreign_edge_overrides.remove(&position);
                }
                // The arm provably runs, but only on the forks where every
                // EARLIER test was false — an earlier maybe-taken arm's
                // surviving environment is still a live possibility, so
                // this arm JOINS `surviving` like any other rather than
                // replacing the whole statement's answer.
                if !arm_terminates(body) {
                    surviving.push(arm_environment);
                }
                chain_exhausted = true;
                break;
            }
        }
        let mut arm_environment = path.fork();
        if let Some(test) = test {
            arm_environment = assume(test, arm_environment, context.kernel, true);
        }
        // THE RELATIONAL LEDGER (B1.keep.write's own `increment_weakens_
        // to_le`): `i < n` between two Set-kind names is a relation the
        // SET channel's own `condition_tree_of` cannot express (its
        // every leaf is scoped to ONE `place` against a LITERAL —
        // `narrowing.rs`'s own module doc names "two changing names" as
        // one of the shapes that lowers to `other_tree()`), so `assume`
        // above narrows neither `i` nor `n` at all: `i`'s window going
        // into this arm is still its bare declared `[0, 150]`, never
        // intersected with `n`'s own current ceiling. `relational_
        // narrow_upper_bound` reads that comparison directly off the
        // test's own syntax and REBINDS the left name (`i`) to its
        // current window intersected with the bound the comparison
        // proves — the same "narrow the BINDING, not the declared
        // refinement" law `narrowing.rs::meet_set_answer` already keeps
        // for a single-name comparison, extended here to a comparison
        // against ANOTHER name's current window. A LATER `i += 1` inside
        // this arm then folds the kernel's own arithmetic over the
        // TIGHTENED current binding, so `updated`'s own ceiling already
        // reflects the relation — no separate "intersect after the
        // fact" step, and the kernel arithmetic itself is untouched.
        if let Some(test) = test {
            relational_narrow_upper_bounds(test, body, &mut arm_environment);
        }
        let mut arm_provably_unbound: HashSet<String> = HashSet::new();
        // FOREIGN EDGE: same walrus-in-test recognition as the
        // provably-true short-circuit arm above, applied to this
        // ordinary (possibly unknown-truthiness) arm's own body.
        let mut foreign_edge_overrides = test
            .map(|test| serve_foreign_edge_in_walrus_test(test, body, &arm_environment, context, blocked, out))
            .unwrap_or_default();
        for (position, stmt) in body.iter().enumerate() {
            if let Some(published) = foreign_edge_overrides.get(&position) {
                arm_environment.set_evaluated_node(published.clone());
            }
            walk_statement(
                stmt,
                return_refinement,
                yield_refinement,
                context,
                &mut arm_environment,
                aug_assign_refinements,
                &mut arm_provably_unbound,
                blocked,
                out,
            );
            arm_environment.set_evaluated_node(Vec::new());
            foreign_edge_overrides.remove(&position);
        }
        if !arm_terminates(body) {
            surviving.push(arm_environment);
        }
        // Below this arm, its test was false — the next arm's test (and
        // the fall-through) live in that complement.
        if let Some(test) = test {
            path = assume(test, path, context.kernel, false);
        }
    }

    // `if` with no `else`/final catch-all arm falls through whenever
    // every test was false — the implicit empty arm survives carrying
    // exactly those complements (`if x < k: return` leaves x ∈ [k, ∞)
    // at the post-if point). An exhausted chain (a test proved true)
    // provably never falls through, so no such survivor is added.
    let has_catch_all = arms.last().map(|(test, _)| test.is_none()).unwrap_or(false);
    if !has_catch_all && !chain_exhausted {
        surviving.push(path.fork());
    }

    *environment = match surviving.len() {
        0 => path,
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

/// THE RELATIONAL LEDGER's own fact collector: every `left < right` /
/// `left <= right` two-Name comparison `test` states directly, gathered
/// from two shapes — an `and`-conjunction of separate `Compare` nodes
/// (`n > 150 and i < n`, `i >= 0 and 0 <= n <= 9 and i < n`), and a
/// CHAINED comparison's own adjacent pairs (`lo <= x <= hi` states BOTH
/// `lo <= x` AND `x <= hi` — CPython's own chaining rule, `tmp/cpython/
/// Doc/reference/expressions.rst`'s "Comparisons can be chained
/// arbitrarily": `a op1 b op2 c` means `a op1 b and b op2 c`). Recurses
/// through NESTED `and`s (`(p and q) and r`) but never descends into an
/// `or` — an `or`'s own two arms are not both live at once, so a fact
/// true on one arm is not a fact of the whole test. A conjunct/pair whose
/// two sides are not both bare Names, or whose op is not `Lt`/`LtE`,
/// contributes no fact — the same "narrows nothing" default every other
/// leaf here keeps for a shape it does not recognize.
pub(super) fn relational_ceiling_facts(test: &Expr) -> Vec<(String, CmpOp, String)> {
    let mut facts = Vec::new();
    collect_relational_ceiling_facts(test, &mut facts);
    facts
}

pub(super) fn collect_relational_ceiling_facts(test: &Expr, facts: &mut Vec<(String, CmpOp, String)>) {
    match test {
        Expr::BoolOp(bool_op) if bool_op.op == BoolOp::And => {
            for value in &bool_op.values {
                collect_relational_ceiling_facts(value, facts);
            }
        }
        Expr::Compare(compare) => {
            let mut left = compare.left.as_ref();
            for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
                if matches!(op, CmpOp::Lt | CmpOp::LtE) {
                    if let (Expr::Name(left_name), Expr::Name(right_name)) = (left, right) {
                        facts.push((left_name.id.as_str().to_owned(), *op, right_name.id.as_str().to_owned()));
                    }
                }
                left = right;
            }
        }
        _ => {}
    }
}

/// `i < n` / `i <= n` (`i`/`n` both bare Names, both currently bound
/// `Kind::Set` over INTEGER-tagged windows) rebinds `i` — the SAME
/// binding `walk_if`'s own arm-body walk sees, so a LATER aug-assign
/// inside this arm folds the kernel's arithmetic over the tightened
/// window — to its current window INTERSECTED with an upper bound `n`'s
/// own current ceiling proves: `i < n` proves `i ≤ n's ceiling − 1`
/// (strict order between two integers — CPython `int` has no value
/// between `k` and `k+1`, `tmp/cpython/Doc/library/stdtypes.rst`'s
/// integer type states no fractional members); `i <= n` proves `i ≤ n`'s
/// ceiling directly. Requires BOTH windows to carry `Form::Integer` —
/// the `-1` step is unsound over a Float-tagged window, which this
/// reader declines rather than narrow wrongly.
///
/// GENERALIZED beyond one bare `Compare` (B1.est.guard's own
/// `conjunction_inside`, B1.keep.trans's `transitivity_holds`, B1.use.
/// project's `projection_stays_inside`, B1.use.sink's `between_bounds_
/// admitted`): `relational_ceiling_facts` collects every two-Name `<`/`<=`
/// fact the test states, through an `and`-conjunction of separate
/// comparisons AND a chained comparison's own adjacent pairs, and each is
/// applied here in turn. Applied TWICE over the same fact list — the
/// second pass is ONE transitivity step: `i < n and n <= m` first
/// narrows `i` against `n`'s window as it stood at the START of the
/// first pass (still `n`'s bare declared window), and separately narrows
/// `n` against `m`; the second pass then re-narrows `i` against `n`'s
/// NOW-TIGHTENED window, so `i`'s own final ceiling reflects `m`
/// transitively without a general fixpoint loop. A third fact chained
/// off a second transitive step is out of this pass's own reach — two
/// passes is the one step the cluster asks for, never a fixpoint.
///
/// Each fact is proved once, at arm entry, over the right name's CURRENT
/// window — so it must not survive a LATER write to that name anywhere
/// in `body` (B1.keep.write.py's own `reassign_forgets_relation`: `if i
/// < n: n = 0; return i` — the `i < n` relation the guard proved is
/// STALE the moment `n` is reassigned, and `i` must NOT be judged
/// against the relation's now-invalid bound). Gated on `locally_bound_
/// names(body)` never naming the right name — the same whole-body,
/// any-nesting-depth scan the walk's own scoping (`walk_body_with_self_
/// binding`'s own doc) already uses to decide what a body binds, checked
/// here BEFORE the arm body walks rather than invalidated statement-by-
/// statement, the conservative direction: a body that writes the right
/// name on some path never gets that one fact's narrowing at all, even
/// on a path that does not.
///
/// A fact whose left name is not currently `Kind::Set` over an
/// Integer-tagged window, whose right name is unbound or not Integer-
/// tagged, or whose right window carries no provable ceiling
/// (`aug_assign_ceiling`) contributes nothing — the honest "narrows
/// nothing" default every other narrowing channel in this file already
/// keeps, applied fact-by-fact rather than declining the whole test for
/// one unreadable conjunct.
pub(super) fn relational_narrow_upper_bounds(test: &Expr, body: &[Stmt], arm_environment: &mut Environment) {
    let facts = relational_ceiling_facts(test);
    if facts.is_empty() {
        return;
    }
    let locally_bound = locally_bound_names(body);
    for _ in 0..2 {
        for (left_name, op, right_name) in &facts {
            apply_relational_ceiling_fact(left_name, *op, right_name, &locally_bound, arm_environment);
        }
    }
}

/// One fact's own narrowing step — `relational_narrow_upper_bounds`'s
/// per-fact body, split out so the two-pass loop above stays a plain
/// fact-list iteration rather than a hand-inlined nest.
///
/// `left_name op right_name` is symmetric information: it states BOTH an
/// upper bound on `left_name` (`right_name`'s own ceiling) AND a lower
/// bound on `right_name` (`left_name`'s own floor) — `lo <= x` narrows
/// `x`'s ceiling to nothing new (a floor gives no ceiling), but ALSO
/// narrows `x`'s FLOOR to `lo`'s own floor, exactly as `x <= hi` narrows
/// `x`'s ceiling to `hi`'s. B1.use.sink's own `between_bounds_admitted`
/// (`lo <= x <= hi`, `lo`/`hi` both `Age`) needs both halves at once: the
/// ceiling half alone leaves `x` floorless, which a `Kind::Set` sink
/// judge correctly refuses to admit into `Age`'s own `[0, 150]` window.
/// Applying the ceiling narrowing to `left_name` and the floor narrowing
/// to `right_name` from the SAME fact is the sound symmetric reading of
/// one relational fact, not two independent claims.
pub(super) fn apply_relational_ceiling_fact(
    left_name: &str,
    op: CmpOp,
    right_name: &str,
    locally_bound: &HashSet<String>,
    arm_environment: &mut Environment,
) {
    use refined_sets::refinement_forms::{at_least, at_most, Form};
    fn integer_windowed(value: &AbstractValue) -> bool {
        value.kind == Kind::Set && value.set.forms.iter().any(|form| form.form == Form::Integer)
    }
    // the ceiling half: narrow `left_name` by `right_name`'s own ceiling —
    // read both sides, THEN bind, so the two immutable reads above never
    // overlap the mutable bind below.
    if !locally_bound.contains(right_name) {
        let current = arm_environment.read(left_name).cloned();
        let other = arm_environment.read(right_name).cloned();
        if let (Some(current), Some(other)) = (current, other) {
            if integer_windowed(&current) && integer_windowed(&other) {
                if let Some(other_ceiling) = aug_assign_ceiling(&other) {
                    let bound = if matches!(op, CmpOp::Lt) { other_ceiling - 1.0 } else { other_ceiling };
                    let mut narrowed = current;
                    narrowed.set.forms.push(at_most(bound));
                    arm_environment.bind(left_name, narrowed);
                }
            }
        }
    }
    // the floor half: narrow `right_name` by `left_name`'s own floor — the
    // same fact's mirrored claim, see this function's own doc.
    if !locally_bound.contains(left_name) {
        let current = arm_environment.read(left_name).cloned();
        let other = arm_environment.read(right_name).cloned();
        if let (Some(current), Some(other)) = (current, other) {
            if integer_windowed(&current) && integer_windowed(&other) {
                if let Some(current_floor) = aug_assign_floor(&current) {
                    let bound = if matches!(op, CmpOp::Lt) { current_floor + 1.0 } else { current_floor };
                    let mut narrowed = other;
                    narrowed.set.forms.push(at_least(bound));
                    arm_environment.bind(right_name, narrowed);
                }
            }
        }
    }
}

/// The lowest value `value` could hold: the TIGHTEST `AtLeast`-form
/// bound of a `Kind::Set` window, mirroring `aug_assign_ceiling`'s own
/// `AtMost` read — a window can carry more than one `AtLeast` form for
/// the same reason (a later narrowing pass pushes a tighter bound
/// without removing the stale one), so this reads the MAXIMUM over
/// every `AtLeast` form, never the first one found. `None` for a set
/// with no `AtLeast` form (unbounded below) or any non-`Kind::Set`
/// shape — the caller narrows nothing rather than guess a floor the
/// value does not state.
pub(super) fn aug_assign_floor(value: &AbstractValue) -> Option<f64> {
    use refined_sets::refinement_forms::Form;
    match value.kind {
        Kind::Set => value
            .set
            .forms
            .iter()
            .filter(|form| form.form == Form::AtLeast)
            .map(|form| form.a)
            .reduce(f64::max),
        _ => None,
    }
}

/// Whether `test` is an `is`/`is not` comparison against a bare `None`
/// literal, with the other side a bare Name carrying a DECLARED
/// refinement (`aug_assign_refinements`, populated at that name's own
/// `AnnAssign`) that admits `None` — the ordinary `Optional[X]`/`X | None`
/// peeling idiom `walk_if`'s DEAD-BRANCH LAW must never treat as dead
/// code, however provably-false the test reads against one concrete
/// assignment (see that law's own doc for the full reasoning and the
/// `none_test_on_helper_that_never_answers_none` row this exception must
/// NOT touch). A chained comparison (more than one `ops`/`comparators`
/// entry), a non-`Is`/`IsNot` op, a `None` on both sides, or a name with
/// no `aug_assign_refinements` entry (or one that does not admit `None`)
/// all fall through to `false` — the dead-branch law fires for every one
/// of those shapes exactly as before this exception existed.
pub(super) fn is_admits_none_peel_test(test: &Expr, aug_assign_refinements: &HashMap<String, DeclaredRefinement>) -> bool {
    let Expr::Compare(compare) = test else {
        return false;
    };
    let ([op], [comparator]) = (&*compare.ops, &*compare.comparators) else {
        return false;
    };
    if !matches!(op, CmpOp::Is | CmpOp::IsNot) {
        return false;
    }
    let left_is_none = matches!(compare.left.as_ref(), Expr::NoneLiteral(_));
    let right_is_none = matches!(comparator, Expr::NoneLiteral(_));
    if left_is_none == right_is_none {
        // both None, or neither — not the peel shape at all
        return false;
    }
    let name_side = if right_is_none { compare.left.as_ref() } else { comparator };
    let Expr::Name(name) = name_side else {
        return false;
    };
    aug_assign_refinements
        .get(name.id.as_str())
        .is_some_and(|declared| declared.admits_none)
}

/// Whether a body's last statement is a bare `return`/`raise` — an arm
/// ending this way never falls through to the post-if point, so its
/// environment describes only unreachable code and must not join.
pub(super) fn arm_terminates(body: &[Stmt]) -> bool {
    matches!(body.last(), Some(Stmt::Return(_)) | Some(Stmt::Raise(_)))
}

/// The same termination test as `arm_terminates`, extended with ONE more
/// proven-terminal shape: the body's last statement is not syntactically a
/// `return`/`raise`, but the walk's OWN provable-raise machinery already
/// recorded an RTS7001 finding whose range falls inside that last
/// statement's own range (e.g. `bind_known_sequence_target`'s arity-mismatch
/// fire, anchored at the destructured value's range, inside an `Assign`
/// statement that is itself the try body's last statement). A body ending
/// this way also describes only unreachable code past that point — the
/// exception is provably always raised, so a "falls through normally" path
/// never exists. `findings_before` is `out`'s length captured immediately
/// before this body was walked; only findings recorded DURING that walk are
/// considered. This does not weaken the syntactic check for a body whose
/// last statement was never proven to raise — those still route through
/// `arm_terminates` alone.
pub(super) fn arm_terminates_or_provably_raises(body: &[Stmt], out: &[Finding], findings_before: usize) -> bool {
    if arm_terminates(body) {
        return true;
    }
    let Some(last) = body.last() else {
        return false;
    };
    let last_range = last.range();
    out[findings_before..]
        .iter()
        .any(|finding| finding.code == "RTS7001" && last_range.contains_range(finding.range))
}

/// RELATIONAL SUM: `total = 0; for x in xs: total += f(x)` over a
/// sequence known only by its ELEMENT SET, optionally followed by a
/// division of that total by the same sequence's own length.
///
/// The division is what forces the lowering. Interval arithmetic knows
/// `total` is in `[0, n]` and the length is in `[1, n]`, and dividing
/// those two enclosures separately gives `[0, n]` — useless. The tight
/// answer needs the RELATION between numerator and denominator, so the
/// accumulation and the division go to the kernel as ONE program
/// (`relational_sum`'s own doc) and the kernel's linear decider narrows
/// the quotient. Nothing is computed here; this function recognizes the
/// shape, asks, and binds what the kernel answered.
///
/// `Consumed` means the accumulation was walked here and the
/// accumulator holds the kernel's total; the following statement, if
/// any, still walks. `ConsumedWithDivision` means one or two FOLLOWING
/// statements — the division alone, or a count-alias assignment plus
/// the division that reads it — were folded into the same program, so
/// the caller skips exactly that many statements rather than walking
/// them a second time. `Declined` leaves everything to `walk_statement`,
/// exactly as before. Recognition — both spellings, and the
/// `for`/`else` gate — happens at the caller; this function receives an
/// already-recognized accumulation.
///
/// A `return` carrying the division is `Consumed`, not
/// `ConsumedWithDivision`: the return still walks and still judges
/// against the enclosing annotation, with the quotient published for
/// its one division node so the surrounding expression (the fixture's
/// `math.sqrt(...)`) evaluates around an already-narrowed value.
///
/// Conservative declines, each one because the fact it would state is
/// not the fact the kernel proved: any statement other than a division-
/// carrying assignment or return sitting immediately after the
/// accumulation, OR after a single count-alias assignment
/// (`count = len(samples)`) immediately after the accumulation (a
/// statement in either of those positions that is not one of these
/// shapes could rebind either name, so nothing further is folded and
/// the division walks ordinarily); a walrus rebinding either name
/// anywhere in that expression; a return whose expression holds the
/// division zero times or more than once (with two, one published
/// answer cannot say which node it belongs to); and a kernel refusal.
pub(super) fn walk_relational_sum(
    mut recognized: relational_sum::RecognizedAccumulation,
    loop_target: Option<&Expr>,
    bound_at: Option<TextRange>,
    following: &[Stmt],
    environment: &mut Environment,
) -> RelationalSum {
    // The division, when it sits at the very next statement OR one hop
    // later behind a count-alias assignment. Only that immediate
    // lookahead is read: anything else in between could rebind either
    // name, and this pass never reasons about what it did not look at.
    //
    // The count-alias hop: `following.first()` is tried first as the
    // division-carrying statement itself (the direct spelling); when it
    // is instead a plain `<name> = len(<sequence>)` naming THIS
    // accumulation's own sequence (`is_length_alias_assignment`), the
    // alias is recorded (`record_length_alias`) and the division/return
    // match re-runs against `following.get(1)` instead. With the alias
    // at `following.first()` and the division at `following.get(1)`,
    // there is no statement BETWEEN them for
    // `reassigns_alias_or_sequence` to guard — that guard is for a wider
    // gap than this exact one-hop lookahead ever opens.
    //
    // Two shapes carry the division itself. An ASSIGNMENT divides at
    // its top level and names the quotient, so the answer binds to that
    // name and the statement is consumed whole. A RETURN may nest the
    // division anywhere inside the returned expression — the fixture's
    // own `return math.sqrt(total / len(samples))` — so the return is
    // still walked ordinarily, with the quotient published for exactly
    // that one division node (`Environment::set_evaluated_node`) and
    // the surrounding call evaluated around it as usual.
    //
    // A walrus rebinding either name anywhere in the expression
    // declines both shapes: the rebinding happens mid-expression, so
    // the division would be over a value the kernel never tied.
    let mut divided_into: Option<(String, ruff_text_size::TextRange)> = None;
    let mut published_division = None;
    // How many of `following`'s leading statements this walk consumed:
    // 1 for the division alone, 2 when a count-alias assignment was
    // consumed ahead of it. Stays 0 on a decline, so the caller's own
    // skip bookkeeping only ever advances past what was actually folded.
    let mut consumed_statements: usize = 0;
    // The alias name and its own Assign range, held only long enough to
    // bind+publish below once the division actually folds — mirroring
    // `divided_into`'s own tentativeness (an alias with nothing folded
    // behind it is not consumed, so nothing about it is bound either).
    let mut alias_binding: Option<(String, ruff_text_size::TextRange)> = None;
    let mut division_candidate = following.first();
    if let Some(Stmt::Assign(alias_assign)) = following.first() {
        if let Some(alias) = relational_sum::is_length_alias_assignment(alias_assign, &recognized, environment) {
            relational_sum::record_length_alias(&mut recognized, alias.clone());
            alias_binding = Some((alias, alias_assign.range()));
            division_candidate = following.get(1);
            consumed_statements = 1;
        }
    }
    match division_candidate {
        Some(Stmt::Assign(assign)) => {
            if let [Expr::Name(target)] = assign.targets.as_slice() {
                if !rebinds_relational_name(assign.value.as_ref(), &recognized)
                    && relational_sum::fold_division(&mut recognized, assign.value.as_ref(), environment)
                {
                    divided_into = Some((target.id.as_str().to_owned(), assign.range()));
                    consumed_statements += 1;
                }
            }
        }
        Some(Stmt::Return(ret)) => {
            if let Some(value) = ret.value.as_deref() {
                if !rebinds_relational_name(value, &recognized) {
                    if let Some((range, op)) = relational_sum::division_range_in(value, &recognized, environment) {
                        relational_sum::fold_located_division(&mut recognized, op);
                        published_division = Some(range);
                        consumed_statements += 1;
                    }
                }
            }
        }
        _ => {}
    }
    // The alias hop was tentative until the division actually folded —
    // an alias assignment with nothing behind it to consume must leave
    // `following.first()` for the ordinary walk, exactly as if the alias
    // read had never been tried.
    if divided_into.is_none() && published_division.is_none() {
        consumed_statements = 0;
        alias_binding = None;
    }
    // Plain data dump for the two relational-sum fixtures one exhausted
    // static trace could not tell apart (the const-effect variant
    // determines, the var-effect variant does not, and only the live
    // answer states say why) — gated on REFINEDPY_DEBUG_RELATIONAL so it
    // never runs unasked. Read once per call, matching the crate's other
    // inline `std::env::var` checks (`kernel_path.rs`, `kernel_bridge.rs`)
    // rather than a cached static — this instrument is meant to be
    // removable later, not a permanent flag. `check.rs` never sees the
    // per-slot wire send/receive `ask_walk_relational` performs inside
    // `relational_sum::walk_accumulation` (that call, and the
    // `catch_unwind` around it, live in a different crate), so a genuine
    // "ask panicked" vs. "kernel declined" split is not nameable from
    // here — both collapse to `walk_accumulation` answering `None`. What
    // IS printed is everything `check.rs` actually holds: the recognized
    // names, the entry states as this call is ABOUT to send them
    // (`recognized.entry_states`, `Debug`-formatted — no `Serialize` wire
    // encoder is public outside `refined_kernel`), and the answer as
    // received, or the words "declined" when `walk_accumulation` itself
    // answers `None`.
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        eprintln!(
            "relational_sum: total={} sequence={} entry_states={:?} statements={:?}",
            recognized.total_name, recognized.sequence_name, recognized.entry_states, recognized.statements
        );
    }
    let Some(answer) = relational_sum::walk_accumulation(&recognized) else {
        if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
            eprintln!("relational_sum: declined (walk_accumulation answered None)");
        }
        return RelationalSum::Declined;
    };
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        eprintln!(
            "relational_sum: answered total={:?} quotient={:?}",
            answer.total, answer.quotient
        );
    }
    // The total binds when the kernel answered it; a total whose own
    // enclosure is unbounded (sign-straddling step, unbounded count) is
    // forgotten instead — the ledger's quotient below still stands.
    match answer.total {
        Some(total) => {
            // The SAME publish an ordinary `total = <value>` gets from
            // `evaluate_expression`'s own record_evaluation call
            // (expressions.rs) — this recognizer consumes the whole
            // Assign statement before that ordinary evaluation ever
            // runs, so without this the binding's own position answers
            // no set at `refined_set_at_position` (bound_at is `None`
            // for the For-loop accumulation shape, which has no single
            // Assign statement to record against).
            if let Some(range) = bound_at {
                environment.record_evaluation(range, total.clone());
            }
            environment.bind(&recognized.total_name, total);
        }
        None => environment.forget(&recognized.total_name),
    }
    // The count-alias name (`count = len(samples)`) binds when a division
    // actually folded behind it AND the kernel's own count window is
    // bindable. The SAME publish the total's binding gets above: this
    // recognizer consumes the whole alias Assign before the ordinary
    // evaluator ever runs, so without this the alias binding's own
    // position answers no set at `refined_set_at_position`. A count the
    // machinery cannot state (an empty count window) leaves the alias
    // exactly as today — forgotten, nothing fabricated.
    if let Some((alias_name, alias_range)) = alias_binding {
        match answer.count {
            Some(count) => {
                environment.record_evaluation(alias_range, count.clone());
                environment.bind(&alias_name, count);
            }
            None => environment.forget(&alias_name),
        }
    }
    // The quotient rides its own slot, so the divided name carries
    // exactly what the kernel proved — or, where the kernel answered the
    // total but not the quotient, nothing at all rather than a guess.
    //
    // The RETURN itself is never skipped — it always still walks at its
    // own position, judging against the enclosing annotation — so a
    // return-with-division fold's own statement does not count toward
    // `skip_statements`; only a count-alias assignment consumed AHEAD of
    // that return does. `consumed_statements` counted the return as 1
    // of its own leading-statement tally, so that one is subtracted back
    // out here before it becomes a caller-facing skip count.
    let outcome = match (divided_into, published_division) {
        (Some((target, bound_range)), _) => {
            match answer.quotient {
                Some(quotient) => {
                    // The SAME publish the total's binding gets above:
                    // this recognizer consumes the whole Assign before
                    // the ordinary evaluation runs, so without this the
                    // quotient binding's own position answers no set at
                    // `refined_set_at_position`.
                    environment.record_evaluation(bound_range, quotient.clone());
                    environment.bind(&target, quotient)
                }
                None => environment.forget(&target),
            }
            RelationalSum::ConsumedWithDivision { skip_statements: consumed_statements }
        }
        // The return is NOT consumed — it still walks, judging against
        // the enclosing `-> Annotation` as always. What changes is that
        // its one division node reads the kernel's narrowed quotient
        // instead of being evaluated from two untied enclosures. A
        // quotient the kernel declined publishes nothing, so that node
        // evaluates ordinarily: never a weaker path than before.
        (None, Some(range)) => {
            if let Some(quotient) = answer.quotient {
                environment.set_evaluated_node(vec![(range, quotient)]);
            }
            // A count-alias assignment sat ahead of this return and was
            // folded in — that ONE statement still needs skipping, even
            // though the return itself does not. `consumed_statements`
            // is exactly 1 (the return alone) when no alias hop ran, and
            // exactly 2 (alias plus return) when one did.
            let alias_statements_ahead = consumed_statements.saturating_sub(1);
            if alias_statements_ahead > 0 {
                RelationalSum::ConsumedWithDivision { skip_statements: alias_statements_ahead }
            } else {
                RelationalSum::Consumed
            }
        }
        (None, None) => RelationalSum::Consumed,
    };
    // The loop variable outlives the loop in CPython (compound_stmts,
    // "the for statement"), but this pass never ran a concrete
    // iteration, so which element it ended on is not a fact here — it
    // is forgotten rather than claimed. The generator spelling has no
    // loop target: the generator's own variable never escapes it.
    if let Some(Expr::Name(loop_variable)) = loop_target {
        environment.forget(loop_variable.id.as_str());
    }
    outcome
}

/// RELATIONAL SUM AT A BARE RETURN: `return sum(<elt> for <var> in
/// <seq>)` with no assignment anywhere in sight — the whole body is one
/// statement. `recognize_generator_sum` only ever reads an `Assign`, so
/// this exact expression, spelled `total = sum(...); return total`,
/// already recognizes and judges; spelled as a direct `return`, it fell
/// through to the ordinary evaluator's `sum_call_over_star` row, which
/// needs a known-sign hull and declines on a sign-straddling element.
///
/// There is no name to bind the total into — it routes straight to the
/// return's own evaluated-node seam (`Environment::set_evaluated_node`),
/// the same publish `walk_relational_sum`'s return-with-division arm
/// already uses, except the published range is the WHOLE returned
/// expression rather than a division nested inside it, since the call
/// to `sum` IS the returned expression here. `walk_return` (called by
/// the ordinary `walk_statement` dispatch right after this) then reads
/// the publish at `evaluate_expression`'s own dispatch head and judges
/// it against the declared return set exactly as any other value would
/// be.
///
/// A kernel refusal, or a total the kernel could not bind, publishes
/// nothing: `evaluate_expression` then runs unchanged and the ordinary
/// `sum_call_over_star` row answers whatever it already would.
pub(super) fn publish_relational_sum_return(ret: &StmtReturn, environment: &mut Environment) {
    let Some(value) = ret.value.as_deref() else {
        return;
    };
    let Some(recognized) = relational_sum::recognize_generator_sum_in_return(value, environment) else {
        return;
    };
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        eprintln!(
            "relational_sum: (bare return) sequence={} entry_states={:?} statements={:?}",
            recognized.sequence_name, recognized.entry_states, recognized.statements
        );
    }
    let Some(answer) = relational_sum::walk_accumulation(&recognized) else {
        if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
            eprintln!("relational_sum: (bare return) declined (walk_accumulation answered None)");
        }
        return;
    };
    if let Some(total) = answer.total {
        environment.set_evaluated_node(vec![(value.range(), total)]);
    }
}

/// What `walk_relational_sum` did with a recognized accumulation.
pub(super) enum RelationalSum {
    /// Not this pass's shape — the ordinary walk runs.
    Declined,
    /// The accumulation was walked here and the accumulator holds the
    /// total. Whatever follows still walks — including a `return` whose
    /// division was folded, which reads its published quotient.
    Consumed,
    /// The accumulation AND one or two of the leading statements of
    /// `following` were walked as one kernel program, so the caller
    /// skips exactly that many statements rather than walking them a
    /// second time. The count is 1 for a division-carrying ASSIGNMENT
    /// alone, or 2 when a count-alias assignment
    /// (`count = len(samples)`) was folded ahead of it — the alias
    /// statement is consumed the same as the division statement it
    /// feeds, even though neither is a `return` (a `return` is never
    /// skipped; it always still walks, so a folded return-with-division
    /// reports `Consumed` for the division itself, with this variant
    /// used only when a count-alias assignment preceded it — see the
    /// count-alias-then-return arm below).
    ConsumedWithDivision { skip_statements: usize },
}

/// Whether an expression rebinds — through a walrus — either name a
/// recognized accumulation states its relation over. Such a rebinding
/// happens mid-expression, so the division would be taken over a value
/// the kernel's relation was never tied to.
pub(super) fn rebinds_relational_name(
    expression: &Expr,
    recognized: &relational_sum::RecognizedAccumulation,
) -> bool {
    let mut names = HashSet::new();
    collect_walrus_names(expression, &mut names);
    names.contains(&recognized.total_name) || names.contains(&recognized.sequence_name)
}
