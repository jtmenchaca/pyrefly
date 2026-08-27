//! For-loop shape: concrete literal/range iteration, else_runs,
//! sort preservation, if/elif/else and break/continue inside the
//! body, judged writes into a declared slot, the return-through-loop
//! channel, statement-level mutation contracts, and non-loop/nested
//! declines.

use super::*;

#[test]
fn for_over_literal_list_sums_and_keeps_last_target_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [60, 61]:\n    total += age\n");
    let environment = environment_with(&[("total", 0.0), ("age", 0.0)]);
    let result = run(&stmt, &environment, &kernel).expect("shape is concrete");
    assert_eq!(result.read("total").unwrap().values, vec![121.0]);
    // the target stays bound to the LAST element after the loop —
    // never reset or deleted (compound_stmts.html "the for statement")
    assert_eq!(result.read("age").unwrap().values, vec![61.0]);
}

#[test]
fn for_over_range_three_sums_zero_one_two() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for i in range(3):\n    total += i\n");
    let environment = environment_with(&[("total", 0.0)]);
    let result = run(&stmt, &environment, &kernel).expect("range(3) is concrete");
    assert_eq!(result.read("total").unwrap().values, vec![3.0]);
    assert_eq!(result.read("i").unwrap().values, vec![2.0]);
}

#[test]
fn body_with_a_call_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1, 2]:\n    total = f(x)\n");
    let environment = environment_with(&[("total", 0.0)]);
    assert!(run(&stmt, &environment, &kernel).is_none());
}

#[test]
fn for_else_reports_else_runs_true_after_exhaustion() {
    let Some(kernel) = loaded_kernel() else { return };
    // this module no longer runs the else body itself (check.rs
    // owns that, fully judged) — it only reports else_runs: true,
    // since the loop is exhausted with no break.
    let stmt = parsed_loop("for x in [1, 2]:\n    total += x\nelse:\n    done = 1\n");
    let environment = environment_with(&[("total", 0.0), ("done", 0.0)]);
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("body runs, else_runs reported");
    assert_eq!(answer.environment.read("total").unwrap().values, vec![3.0]);
    assert!(answer.else_runs, "the loop exhausts with no break — the else clause runs");
    assert!(answer.returned.is_none(), "no return fires in this row");
    // the orelse body (`done = 1`) never runs HERE — this module
    // only reports else_runs; check.rs walks the orelse. `done`
    // therefore still carries its PRE-loop binding (0.0), proving
    // the executor did not run the else itself.
    assert_eq!(answer.environment.read("done").unwrap().values, vec![0.0]);
}

#[test]
fn empty_literal_list_leaves_target_unbound_when_it_was_never_bound() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in []:\n    total += x\n");
    let environment = environment_with(&[("total", 0.0)]);
    let result = run(&stmt, &environment, &kernel).expect("empty literal list is concrete");
    // x was never assigned by the loop (compound_stmts.html): it
    // carries forward whatever the pre-loop environment held, which
    // here is nothing
    assert!(result.read("x").is_none());
    assert_eq!(result.read("total").unwrap().values, vec![0.0]);
}

/// UNIT: `run_if_once_over_unknown_test`'s own join path — `for x in
/// xs: if 0 <= x <= 149: out.append(x + 1) else: out.append(0)`
/// against `xs: list[Wide]` ([0, 200]). Before `run_if_once`'s own
/// Set-narrowing fallback, this whole loop declined with the coarse
/// "not yet walked" blocker, because `0 <= x <= 149` never resolves
/// to one known boolean against a Set-bound `x`. This pins the fixed
/// mechanism the wave's A10.seed.library/A15.xfer.dedupe/A15.xfer.inject
/// rows share: the loop now runs, joining both narrowed arms.
#[test]
fn if_else_over_a_set_bound_loop_element_joins_both_narrowed_arms() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(
        "for x in xs:\n    if 0 <= x <= 149:\n        out.append(x + 1)\n    else:\n        out.append(0)\n",
    );
    let mut environment = Environment::new(HashSet::from(["xs".to_owned(), "x".to_owned(), "out".to_owned()]));
    environment.bind("xs", wide_list_parameter());
    environment.bind("out", collection_models::list_literal_value(&[]));
    let result = run(&stmt, &environment, &kernel).expect("the if/else over a Set-bound element now runs");
    // `out` never widens to unknown(): `stabilized_join`'s outer join
    // compares the pre-loop `out` (0 items) against the one-pass `out`
    // (1 item, already the if/else arms' own join) — a DIFFERENT-LENGTH
    // List/List pair, which `join_lists_of_different_length`
    // (lattice_operations.rs) answers as a repetition window over the
    // joined element, `Kind::Set` — the join produced a real, weaker-
    // true answer, not a decline-shaped stand-in.
    assert_eq!(result.read("out").unwrap().kind, Kind::Set);
}

/// UNIT: `run_if_once`'s own EXISTING contract is unchanged for a
/// test this file's narrowing channels do not recognize at all —
/// `if f():` over a CONCRETE per-element iterate never reaches the
/// new join fallback (no name in the test is Set-bound), so the
/// whole loop still declines exactly as before this wave's fix.
#[test]
fn unknown_if_test_on_a_concrete_iterate_still_declines_the_whole_loop() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1, 2]:\n    if f():\n        total = total + x\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
    environment.bind("total", integer(0.0));
    assert!(run(&stmt, &environment, &kernel).is_none(), "an opaque call still declines — nothing here for assume to narrow");
}

/// UNIT: the AugAssign kernel-aware fix — `total += x` where `x` is
/// the abstract pass's own Set-bound element (never one concrete
/// number). Before wiring `binary_arithmetic_value_with_kernel` in,
/// this AugAssign's `updated.kind != Kind::Values` guard declined
/// the whole loop the moment the operand was Set-shaped — the
/// mechanism E1.loop/E2.loop/B2.est.loop/B3.est.loop share.
#[test]
fn aug_assign_folds_a_set_shaped_operand_through_the_kernel_aware_transfer() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in xs:\n    total += x\n");
    let mut environment = Environment::new(HashSet::from(["xs".to_owned(), "x".to_owned(), "total".to_owned()]));
    environment.bind("xs", wide_list_parameter());
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("the Set-shaped accumulation now runs");
    // total widens to a known Set/Values answer, never unknown()
    assert_ne!(result.read("total").unwrap().kind, Kind::Unknown);
}

/// UNIT: `live_list_element_walk` — a `for` over a KNOWN list whose body
/// CONDITIONALLY appends to that same list. stdtypes.rst, "Common
/// Sequence Operations": "Forward and reversed iterators over mutable
/// sequences access values using an index. That index will continue to
/// march forward... even if the underlying sequence is mutated. The
/// iterator terminates only when an :exc:`IndexError` or a
/// :exc:`StopIteration` is encountered." So the element appended during
/// pass 0 IS visited: the body runs exactly twice over an initially
/// one-element list, and `count` lands on exactly `{2}`.
#[test]
fn a_list_grown_inside_its_own_loop_iterates_the_appended_element() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in lst:\n    count += 1\n    if len(lst) < 2:\n        lst.append(2)\n");
    let mut environment = Environment::new(HashSet::from(["lst".to_owned(), "x".to_owned(), "count".to_owned()]));
    environment.bind("lst", known_list(vec![integer(1.0)], TrustProved));
    environment.bind("count", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("the live list steps exactly");
    assert_eq!(
        result.read("count").unwrap().values,
        vec![2.0],
        "the appended element is visited — the index marches past the original length"
    );
    assert_eq!(
        result.read("lst").unwrap().items.len(),
        2,
        "the list grew once and then the guard stopped it growing"
    );
}

/// UNIT: the same walk over a body that never appends at all still ends
/// at the original length — the live re-read agrees with the snapshot
/// whenever nothing mutates, so this row proves the index rule and not
/// an off-by-one of its own.
#[test]
fn a_conditional_append_that_never_fires_leaves_the_count_at_the_original_length() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in lst:\n    count += 1\n    if len(lst) < 1:\n        lst.append(9)\n");
    let mut environment = Environment::new(HashSet::from(["lst".to_owned(), "x".to_owned(), "count".to_owned()]));
    environment.bind("lst", known_list(vec![integer(1.0), integer(2.0)], TrustProved));
    environment.bind("count", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("the live list steps exactly");
    assert_eq!(result.read("count").unwrap().values, vec![2.0]);
    assert_eq!(result.read("lst").unwrap().items.len(), 2, "the guard never held, so nothing was appended");
}

/// UNIT: `list_size_changing_mutation_range`'s own fire —
/// `for x in lst: lst.append(x)` on a `list[int]`-shaped (repetition-
/// window) parameter provably never terminates. Pins C5.rangefor's
/// own mechanism: the fire is recorded and the loop declines, rather
/// than silently running the abstract pass over a receiver the body
/// itself keeps growing.
#[test]
fn list_appended_to_inside_its_own_for_loop_fires_and_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in lst:\n    lst.append(x)\n");
    let mut environment = Environment::new(HashSet::from(["lst".to_owned(), "x".to_owned()]));
    environment.bind("lst", wide_list_parameter());
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out);
    assert!(answer.is_none(), "a self-feeding append never terminates — the loop must decline");
    assert_eq!(out.len(), 1, "exactly one fire names the non-termination: {out:?}");
    assert!(out[0].1.contains("never terminates"), "the fire names non-termination: {:?}", out[0].1);
}

#[test]
fn non_loop_statement_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("total = 1\n");
    let environment = environment_with(&[("total", 0.0)]);
    assert!(run(&stmt, &environment, &kernel).is_none());
}

// --- sort preservation (UNIT 1) ---

#[test]
fn for_over_int_literal_list_binds_the_iterate_as_integer_sorted() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [10, 20, 30]:\n    total = total + age\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "age".to_owned()]));
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("int list is concrete");
    let total = result.read("total").expect("total stays bound");
    assert_eq!(total.values, vec![60.0]);
    // the fix under test: an all-int accumulation answers an
    // Integer-tagged total, not a Float-tagged one — a Float 60.0
    // wrongly fires the int-sort law against an Age slot even
    // though 60 is in range (a-statements.py:515)
    assert_eq!(total.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn range_iterate_is_integer_sorted() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for i in range(3):\n    total = total + i\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("range is concrete");
    assert_eq!(result.read("total").unwrap().kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn for_over_float_literal_list_binds_the_iterate_as_float_sorted() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1.5, 2.5]:\n    total = total + x\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
    environment.bind("total", known_values(vec![0.0], PrimitiveKind::Float, TrustProved));
    let result = run(&stmt, &environment, &kernel).expect("float list is concrete");
    let total = result.read("total").expect("total stays bound");
    assert_eq!(total.values, vec![4.0]);
    assert_eq!(total.kind_tag, Some(PrimitiveKind::Float));
}

// --- if / elif / else inside a body (UNIT 2) ---

#[test]
fn if_arm_runs_only_when_the_test_holds() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1, 2, 3]:\n    if x > 1:\n        total = total + x\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("if inside body is concrete");
    // x=1: test false, no-op; x=2: total=2; x=3: total=5
    assert_eq!(result.read("total").unwrap().values, vec![5.0]);
}

#[test]
fn else_arm_runs_when_no_test_holds() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(
        "for x in [1, 2]:\n    if x > 100:\n        total = total + 1\n    else:\n        total = total + x\n",
    );
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("if/else inside body is concrete");
    assert_eq!(result.read("total").unwrap().values, vec![3.0]);
}

#[test]
fn unknown_if_test_on_any_iteration_declines_the_whole_loop() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1, 2]:\n    if f():\n        total = total + x\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
    environment.bind("total", integer(0.0));
    assert!(run(&stmt, &environment, &kernel).is_none());
}

// --- break / continue / else_runs (UNIT 2, extended for the LOOP
// ELSE + DEAD-ELSE LAW) ---

#[test]
fn break_stops_the_loop_and_reports_else_runs_false() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(
        "for i in range(3):\n    if i == 1:\n        break\n    total = total + 1\nelse:\n    total = 200\n",
    );
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
    environment.bind("total", integer(0.0));
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("break inside body is concrete");
    // i=0: total=1; i=1: breaks before total += 1 runs
    assert_eq!(answer.environment.read("total").unwrap().values, vec![1.0]);
    assert_eq!(answer.environment.read("i").unwrap().values, vec![1.0]);
    assert!(!answer.else_runs, "a break must report else_runs: false");
}

#[test]
fn continue_skips_the_rest_of_that_iteration_only() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(
        "for i in range(4):\n    if i == 2:\n        continue\n    total = total + i\n",
    );
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "i".to_owned()]));
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("continue inside body is concrete");
    // 0 + 1 + (skip 2) + 3 = 4
    assert_eq!(result.read("total").unwrap().values, vec![4.0]);
}

// --- dict-shaped iteration (UNIT 2) ---

#[test]
fn dict_literal_iteration_into_a_declared_int_slot_fires_through_judge() {
    let Some(kernel) = loaded_kernel() else { return };
    // `age: Age = 0` pre-binds age as an Integer; writing a dict key
    // (a String) into it is now JUDGED through assignability::judge
    // — a-statements.py:508's own row — rather than declining the
    // whole loop the way the old cross-family guard did.
    let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    age = key\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned(), "key".to_owned()]));
    environment.bind("age", integer(0.0));
    let declared = declared_age("age");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop still runs concretely — the write fires, it does not decline");
    assert!(!out.is_empty(), "a String into a declared int-sorted Age slot must fire");
    // the refused write keeps the declared set afterward (refused-
    // write law) — a later read of `age` is silent against Age
    let age = answer.environment.read("age").expect("age stays bound to the declared set");
    assert_eq!(age.kind, Kind::Set);
}

#[test]
fn dedupe_by_range_fires_once_per_syntactic_row_across_many_iterations() {
    let Some(kernel) = loaded_kernel() else { return };
    // the loop iterates twice; both keys are strings, so the SAME
    // syntactic write (`age = key`) would fire twice without the
    // dedupe-by-range rule. Only ONE fire must land.
    let stmt = parsed_loop("for key in {\"a\": 1, \"b\": 2}:\n    age = key\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned(), "key".to_owned()]));
    environment.bind("age", integer(0.0));
    let declared = declared_age("age");
    let mut out = Vec::new();
    loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop runs concretely");
    assert_eq!(out.len(), 1, "one syntactic row fires once, however many iterations run: {out:?}");
}

#[test]
fn a_declared_slot_write_that_stays_in_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [10, 20]:\n    age = x\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned(), "x".to_owned()]));
    environment.bind("age", integer(0.0));
    let declared = declared_age("age");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop runs concretely");
    assert!(out.is_empty(), "every in-set write must stay silent: {out:?}");
    assert_eq!(answer.environment.read("age").unwrap().values, vec![20.0]);
}

#[test]
fn a_declared_slot_write_of_none_fires_rather_than_declining_the_loop() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:541's own shape: an evaluated (non-literal)
    // iterate that is Kind::Null must still reach bind_checked's own
    // judging — run_assign_once's kind guard used to reject
    // Kind::Null outright (only Values/List/Object were accepted),
    // which declined the WHOLE loop before any judging ever ran.
    let stmt = parsed_loop("for item in [x]:\n    age = item\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned(), "item".to_owned(), "x".to_owned()]));
    environment.bind("age", integer(0.0));
    environment.bind("x", refined_domain::abstract_value::null_value());
    let declared = declared_age("age");
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("a Kind::Null iterate must still run the loop concretely, not decline it");
    assert_eq!(out.len(), 1, "None into a non-Optional declared Age slot must fire: {out:?}");
    let age = answer.environment.read("age").expect("age stays bound to the declared set after the refused write");
    assert_eq!(age.kind, Kind::Set);
}

// --- RETURN-THROUGH-LOOP CHANNEL ---

#[test]
fn a_return_on_the_first_iteration_ends_the_loop_and_carries_the_value_out() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [40, 200]:\n    return age\n");
    let environment = Environment::new(HashSet::from(["age".to_owned()]));
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("a return inside the body is still a concretely-executable shape");
    let (value, _range) = answer.returned.expect("the first iteration's return must be carried out");
    assert_eq!(
        value.expect("return age carries a value, not a bare return").values,
        vec![40.0],
        "only the FIRST iterate's return fires — the loop ends right there"
    );
    assert!(!answer.else_runs, "a return, like a break, never lets the else clause run");
}

#[test]
fn a_return_under_an_if_that_never_triggers_reports_no_return() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [10, 20]:\n    if age == 999:\n        return age\n    total = total + age\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned(), "total".to_owned()]));
    environment.bind("total", integer(0.0));
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop runs concretely — the guarded return never fires");
    assert!(answer.returned.is_none(), "the guard is false on every concrete iterate, so no return fires");
    assert_eq!(answer.environment.read("total").unwrap().values, vec![30.0]);
}

#[test]
fn a_return_under_an_if_that_triggers_on_a_later_iterate_ends_the_loop_there() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [10, 200]:\n    if age > 100:\n        return age\n    total = total + age\n");
    let mut environment = Environment::new(HashSet::from(["age".to_owned(), "total".to_owned()]));
    environment.bind("total", integer(0.0));
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("the loop runs concretely up to the returning iterate");
    let (value, _range) = answer.returned.expect("age=200 triggers the guard and returns");
    assert_eq!(value.expect("return age carries a value").values, vec![200.0]);
    // the first iterate (age=10) ran total = total + age BEFORE the
    // second iterate's return fired — the environment still reflects
    // that, even though the returned value is what check.rs judges
    assert_eq!(answer.environment.read("total").unwrap().values, vec![10.0]);
}

#[test]
fn a_bare_return_inside_a_loop_carries_no_value_to_judge() {
    let Some(kernel) = loaded_kernel() else { return };
    // matches check.rs's own walk_return convention: a bare `return`
    // (no expression) judges nothing — this channel must not invent
    // a Null value the way a straight-line bare return never would.
    let stmt = parsed_loop("for age in [40]:\n    return\n");
    let environment = Environment::new(HashSet::from(["age".to_owned()]));
    let declared = no_declared();
    let mut out = Vec::new();
    let answer = loop_final_environment(&stmt, &environment, &kernel, &declared, &mut out)
        .expect("a bare return inside the body is still concretely executable");
    let (value, _range) = answer.returned.expect("the bare return must still end the loop and be carried out");
    assert!(value.is_none(), "a bare `return` carries no value to judge, matching walk_return's own convention");
}

// --- statement-level mutation contract (UNIT 2) ---

#[test]
fn a_recognized_mutating_call_rebinds_the_receiver() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1, 2]:\n    xs.append(x)\n");
    let mut environment = Environment::new(HashSet::from(["xs".to_owned(), "x".to_owned()]));
    environment.bind("xs", known_list(vec![], TrustProved));
    // `mutated_receiver` is the concurrent collection_models.rs
    // wave's own contract; whatever it answers for "append" is what
    // this loop must adopt (Some rebinds, None declines) — this
    // test only pins that the call reaches the contract and does
    // not crash, not a specific collection_models.rs answer shape.
    let _ = run(&stmt, &environment, &kernel);
}

#[test]
fn a_recognized_subscript_write_rebinds_the_dict_receiver() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [40, 41]:\n    ages[\"latest\"] = age\n");
    let mut environment = Environment::new(HashSet::from(["ages".to_owned(), "age".to_owned()]));
    environment.bind("ages", collection_models::dict_literal_value(&[], &[]));
    // `dict_with_item` is the concurrent collection_models.rs wave's
    // own contract; this test pins that a subscript-target write
    // reaches it (Some rebinds, None declines), not a specific
    // answer shape.
    let _ = run(&stmt, &environment, &kernel);
}

#[test]
fn nested_for_in_body_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for x in [1, 2]:\n    for y in [1]:\n        total = total + y\n");
    let environment = environment_with(&[("total", 0.0)]);
    assert!(run(&stmt, &environment, &kernel).is_none());
}

// --- async for over a concrete iterable (UNIT 3) ---

#[test]
fn async_for_over_a_known_literal_tuple_runs_concretely() {
    let Some(kernel) = loaded_kernel() else { return };
    // `is_async` alone must never decline — the same literal-tuple
    // shape a plain `for` already runs concretely.
    let stmt = parsed_loop("async for x in (10, 20, 30):\n    total = total + x\n");
    let mut environment = Environment::new(HashSet::from(["total".to_owned(), "x".to_owned()]));
    environment.bind("total", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("a known literal tuple runs under async for too");
    assert_eq!(result.read("total").unwrap().values, vec![60.0]);
}

#[test]
fn async_for_over_an_unmodeled_call_receiver_still_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:555's own shape: `stream()` is neither `range`
    // nor a `.values()`/`.items()`/`.keys()` dict-view call —
    // iterable_values cannot read it regardless of is_async, so this
    // must still decline, exactly as an equivalent sync receiver
    // would (body_with_a_call_declines, above).
    let stmt = parsed_loop("async for chunk in stream():\n    age = chunk\n");
    let environment = Environment::new(HashSet::from(["age".to_owned(), "chunk".to_owned()]));
    assert!(run(&stmt, &environment, &kernel).is_none());
}

// --- setdefault(...).append(...) composition (UNIT 3) ---

#[test]
fn setdefault_append_extends_an_absent_key_with_the_default_and_the_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [40]:\n    grouped.setdefault(\"young\", []).append(age)\n");
    let mut environment = Environment::new(HashSet::from(["grouped".to_owned(), "age".to_owned()]));
    environment.bind("grouped", collection_models::dict_literal_value(&[], &[]));
    let result = run(&stmt, &environment, &kernel).expect("the chained mutation is a recognized statement shape");
    let grouped = result.read("grouped").expect("grouped stays bound");
    assert_eq!(grouped.kind, Kind::Object);
    assert_eq!(grouped.keys.len(), 1);
    assert_eq!(grouped.keys[0].name, "young");
    assert_eq!(grouped.keys[0].value.items.len(), 1);
    assert_eq!(grouped.keys[0].value.items[0].values, vec![40.0]);
}

#[test]
fn setdefault_append_appends_to_a_present_key_without_losing_earlier_entries() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for age in [40, 200]:\n    grouped.setdefault(\"young\", []).append(age)\n");
    let mut environment = Environment::new(HashSet::from(["grouped".to_owned(), "age".to_owned()]));
    environment.bind("grouped", collection_models::dict_literal_value(&[], &[]));
    let result = run(&stmt, &environment, &kernel).expect("two iterates over the same key both compose");
    let grouped = result.read("grouped").expect("grouped stays bound");
    assert_eq!(grouped.keys.len(), 1, "one key, both appends land on it");
    assert_eq!(
        grouped.keys[0].value.items.iter().map(|v| v.values[0]).collect::<Vec<_>>(),
        vec![40.0, 200.0]
    );
}

#[test]
fn setdefault_append_over_a_ternary_key_groups_by_the_per_iterate_branch() {
    let Some(kernel) = loaded_kernel() else { return };
    // c-reads-and-values.py:1007's own dict_groupby shape: the key
    // expression is a ternary that reads differently PER ITERATE.
    let stmt = parsed_loop(
        "for age in [40, 200]:\n    grouped.setdefault(\"old\" if age > 100 else \"young\", []).append(age)\n",
    );
    let mut environment = Environment::new(HashSet::from(["grouped".to_owned(), "age".to_owned()]));
    environment.bind("grouped", collection_models::dict_literal_value(&[], &[]));
    let result = run(&stmt, &environment, &kernel).expect("the ternary key resolves per iterate");
    let grouped = result.read("grouped").expect("grouped stays bound");
    assert_eq!(grouped.keys.len(), 2, "40 groups under young, 200 groups under old");
    let young = grouped.keys.iter().find(|k| k.name == "young").expect("young key exists");
    assert_eq!(young.value.items[0].values, vec![40.0]);
    let old = grouped.keys.iter().find(|k| k.name == "old").expect("old key exists");
    assert_eq!(old.value.items[0].values, vec![200.0]);
}

/// A1.xfer.loop's own mechanism: `for i in range(n)` over a bounded
/// scalar `n` keeps the counter's UPPER bound. Two readers can answer
/// this loop — `windowed_range_element_pass`, which reads `n`'s own
/// `atMost 200` edge and states the counter `[0, 199]`, and
/// `repetition_window_element_pass`, which reads what `range(n)`
/// evaluates to and states the sort-only window `integer ∧ [0, +inf)`.
/// The range reader is consulted first, so the counter left bound after
/// the loop carries `<= 199` rather than being unbounded above.
#[test]
fn TestA1_xfer_loop_RangeOverBoundedScalarKeepsCounterUpper() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for i in range(n):\n    last = i\n");
    let mut environment = Environment::new(HashSet::from(["n".to_owned(), "i".to_owned(), "last".to_owned()]));
    environment.bind(
        "n",
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        },
    );
    environment.bind("last", integer(0.0));
    let result = run(&stmt, &environment, &kernel).expect("range over a bounded scalar runs");
    let last = result.read("last").expect("last stays bound");
    let upper = last
        .set
        .forms
        .iter()
        .find(|form| form.form == refined_sets::refinement_forms::Form::AtMost)
        .expect("the counter carries an upper edge — the reader that drops it would leave none");
    assert_eq!(upper.a, 199.0, "range(n) with n <= 200 yields counters at most 199");
}

// --- unpack targets and unread-key dict writes (A8.edge.process) ---

/// The whole-strings ground `Σ*`, repeated from zero — what
/// `string_models::sort_only`'s own `splitlines` row answers for an
/// unread `str` receiver, and what `check.rs::seed_parameters` seeds a
/// declared `list[str]` parameter with.
fn string_window(low: i64) -> AbstractValue {
    known_set(
        repetition(refined_sets::codepoint_sets::strings(), low, None),
        None,
        TrustProved,
        SetKindTag::None,
    )
}

/// A8.edge.process's own body shape, one statement at a time: `k, v =
/// line.split("=", 1)` over an EXACT line binds both names to the exact
/// pieces the split states — simple_stmts.rst's "the items are assigned,
/// from left to right, to the corresponding targets."
#[test]
fn an_unpack_target_over_an_exact_split_binds_each_piece_positionally() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for line in [\"a=1\", \"b=2\"]:\n    k, v = line.split(\"=\", 1)\n");
    let environment = Environment::new(HashSet::from(["line".to_owned(), "k".to_owned(), "v".to_owned()]));
    let result = run(&stmt, &environment, &kernel).expect("an exact-split unpack runs");
    // the LAST iterate's own pieces, the target's documented post-loop
    // binding (compound_stmts.rst, "the for statement")
    let key = result.read("k").expect("k stays bound");
    let value = result.read("v").expect("v stays bound");
    assert_eq!(key.values, vec!['b' as u32 as f64], "the last line's key is exactly \"b\"");
    assert_eq!(value.values, vec!['2' as u32 as f64], "the last line's value is exactly \"2\"");
}

/// The same statement over an UNREAD line: `line.split("=", 1)` answers
/// a repetition window of unread strings, whose every position draws
/// from ONE element set, so both targets bind that element — never a
/// decline, which is what left A8.edge.process:11 undetermined.
///
/// The loop is walked by `repetition_window_element_pass`, whose answer
/// is `stabilized_join`'s join of the PRE-LOOP environment with the one
/// judged pass — the loop's own zero-or-more honesty, and
/// `Environment::join`'s own rule that only a name BOTH sides know
/// survives. `k`/`v` are therefore seeded here with an exact word, the
/// way real code binds a name before a loop that may run zero times.
/// The seeded word is strictly narrower than the split's own `Σ*`
/// element, so the post-loop set being EXACTLY `Σ*` is the loop's own
/// contribution — the string-ground absorption arm of `join_known`
/// answers the ground itself once the kernel proves the seeded word is
/// inside it.
#[test]
fn an_unpack_target_over_a_repetition_window_binds_every_name_to_the_element() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for line in lines:\n    k, v = line.split(\"=\", 1)\n");
    let mut environment =
        Environment::new(HashSet::from(["lines".to_owned(), "line".to_owned(), "k".to_owned(), "v".to_owned()]));
    environment.bind("lines", string_window(0));
    environment.bind("k", known_string("seed"));
    environment.bind("v", known_string("seed"));
    let result = run(&stmt, &environment, &kernel).expect("a window-split unpack runs");
    for name in ["k", "v"] {
        let bound = result.read(name).unwrap_or_else(|| panic!("{name} stays bound"));
        assert_eq!(bound.kind, Kind::Set, "{name} binds the window's element set, not one value");
        assert_eq!(
            bound.set,
            refined_sets::codepoint_sets::strings(),
            "{name} binds the split's own ELEMENT — the whole-strings ground — never the window one nesting level above it"
        );
    }
}

/// A8.edge.process's whole loop: an unread-key dict write inside a
/// window walk. The result is an unbounded-key dict whose one element
/// claim covers every value written — the dict really was built, so a
/// later read through it answers a set rather than nothing at all.
#[test]
fn a_dict_written_at_an_unread_key_inside_a_loop_answers_an_unbounded_key_dict() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for line in lines:\n    k, v = line.split(\"=\", 1)\n    result[k] = v\n");
    let mut environment = Environment::new(HashSet::from([
        "lines".to_owned(),
        "line".to_owned(),
        "k".to_owned(),
        "v".to_owned(),
        "result".to_owned(),
    ]));
    environment.bind("lines", string_window(0));
    environment.bind("result", known_object(vec![], None, true, TrustProved, false));
    let answer = run(&stmt, &environment, &kernel).expect("the dict-accumulation loop runs");
    let built = answer.read("result").expect("result stays bound");
    assert_eq!(
        built.kind,
        Kind::ObjectStar,
        "an unread key leaves no key list to record, so the dict states one claim about every present key"
    );
    let element = refined_domain::known_constructors::element_of_object_star(&built)
        .expect("the star wraps the written value's own set");
    assert_eq!(element.kind, Kind::Set, "the values written were the split's own unread pieces");
}

// --- itertools.groupby (A8.seed.library) ---

/// itertools.rst's `groupby` entry, read over an unread iterable: the
/// KEY is the key function's image over the element set — for the
/// fixture's `lambda x: "even" if x % 2 == 0 else "odd"`, the ternary
/// joins both arms into a CLOSED two-member set, not a sort — and the
/// GROUP is a sequence of the iterable's own elements.
///
/// `groupby_element_pass` answers through `stabilized_join`, so what a
/// name holds AFTER the loop is the join of its PRE-LOOP value with the
/// one judged pass's — the loop's own zero-or-more honesty, and
/// `Environment::join`'s own rule that only a name BOTH sides know
/// survives a join. Both read names are therefore seeded here the way
/// real code binds a name before a loop that may run zero times, each
/// seeded with the value the entry's own clause says the pass must
/// produce: the ternary's two-arm image for the key, and the element
/// set repeated from one for the group. A pass that answered ANY other
/// value would join to something else and havoc the name to `unknown()`
/// (`stabilized_join`'s own containment path), so the seed pins the
/// pass's answer exactly rather than merely surviving alongside it.
#[test]
fn groupby_binds_the_key_functions_image_and_a_group_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop(
        "for key, group in groupby(sorted(xs, key=lambda x: x % 2), key=lambda x: \"even\" if x % 2 == 0 else \"odd\"):\n    last_key = key\n    last_group = group\n",
    );
    // the ternary's own image, built through the SAME domain join the
    // lambda body's two arms take — never through the pass under test
    let ternary_image = refined_domain::lattice_operations::join_known(known_string("even"), known_string("odd"));
    let element_set = make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]);
    let group_window = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(repetition(element_set, 1, None), None, TrustProved, SetKindTag::None)
    };
    let mut environment = Environment::new(HashSet::from([
        "xs".to_owned(),
        "key".to_owned(),
        "group".to_owned(),
        "last_key".to_owned(),
        "last_group".to_owned(),
    ]));
    environment.bind("xs", wide_list_parameter());
    environment.bind("last_key", ternary_image.clone());
    environment.bind("last_group", group_window);
    let result = run(&stmt, &environment, &kernel).expect("the groupby pass runs");
    let bound_key = result.read("last_key").expect("last_key stays bound");
    assert_ne!(bound_key.kind, Kind::Unknown, "the key set is the lambda's own image, never nothing");
    assert_eq!(
        bound_key.set, ternary_image.set,
        "the key set is the ternary's own two-arm image — a CLOSED member set, never a bare string sort"
    );
    let bound_group = result.read("last_group").expect("last_group stays bound");
    assert_eq!(bound_group.kind, Kind::Set, "a group is a sequence of the iterable's elements");
    let window = as_repetition(&bound_group.set).expect("the group is a repetition over the element set");
    assert_eq!(window.lo, 1, "every group groupby emits holds at least the element that created it");
    assert_eq!(window.hi, None, "the group count is exactly what groupby leaves unread");
}

/// No `key=` at all — itertools.rst's own default: "If not specified or
/// is ``None``, *key* defaults to an identity function and returns the
/// element unchanged," so the key set IS the element set.
///
/// `last_key` is seeded pre-loop for the reason the two-key pin above
/// states — `stabilized_join`'s answer keeps only names the PRE-LOOP
/// environment knows too — and the seed is `xs`'s own element set, the
/// exact value the identity default must bind. A pass answering the
/// WINDOW instead of the element (one nesting level too many) would
/// join to a different value and havoc the name.
#[test]
fn groupby_with_no_key_function_binds_the_element_set_itself() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for key, group in groupby(xs):\n    last_key = key\n");
    let element_set = make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]);
    let element = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(element_set.clone(), None, TrustProved, SetKindTag::None)
    };
    let mut environment =
        Environment::new(HashSet::from(["xs".to_owned(), "key".to_owned(), "group".to_owned(), "last_key".to_owned()]));
    environment.bind("xs", wide_list_parameter());
    environment.bind("last_key", element);
    let result = run(&stmt, &environment, &kernel).expect("the identity-key groupby pass runs");
    let bound_key = result.read("last_key").expect("last_key stays bound");
    assert_eq!(bound_key.kind, Kind::Set, "the identity default makes the key set the element set");
    assert_eq!(
        bound_key.set, element_set,
        "the key is one ELEMENT of the iterable, never the whole window"
    );
    assert!(
        as_repetition(&bound_key.set).is_none(),
        "the key is one ELEMENT of the iterable, never the whole window"
    );
}

/// A locally bound `groupby` name is not itertools' own — the same
/// shadow gate every module-call row in this checker keeps.
#[test]
fn a_shadowed_groupby_name_is_not_read_as_itertools_groupby() {
    let Some(kernel) = loaded_kernel() else { return };
    let stmt = parsed_loop("for key, group in groupby(xs):\n    last_key = key\n");
    let mut environment =
        Environment::new(HashSet::from(["xs".to_owned(), "groupby".to_owned(), "key".to_owned(), "group".to_owned()]));
    environment.bind("xs", wide_list_parameter());
    environment.bind("groupby", integer(1.0));
    assert!(
        run(&stmt, &environment, &kernel).is_none(),
        "a shadowed name states nothing about itertools' own grouping"
    );
}
