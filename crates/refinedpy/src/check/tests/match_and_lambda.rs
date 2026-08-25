use super::*;

#[test]
fn a_locally_defined_function_is_callable_through_its_own_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    def over_years() -> int:\n",
        "        return 200\n",
        "    return over_years()\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the local def's known return (200) must fire through the call: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

// --- WALRUS BINDING (law 5) ---

#[test]
fn a_walrus_in_an_if_test_binds_the_target_for_the_rest_of_the_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    if (over := 200) > 0:\n",
        "        return over\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the walrus-bound 200 must be readable inside the taken arm: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

// --- PROVABLY-UNBOUND READS (law 6) ---

#[test]
fn a_valueless_annotation_then_a_return_fires_unbound_local_error() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    x: int\n",
        "    return x\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let raises: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("UnboundLocalError"))
        .collect();
    assert_eq!(
        raises.len(),
        1,
        "a valueless declaration read with no intervening assignment provably raises: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(raises[0].message.contains("'x'"), "{}", raises[0].message);
}

#[test]
fn a_valueless_annotation_cured_by_an_assignment_never_fires_unbound() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    x: int\n",
        "    x = 40\n",
        "    return x\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| !f.message.contains("UnboundLocalError")),
        "an assignment between the declaration and the read cures it: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_valueless_annotation_behind_a_branch_never_fires_unbound_conservatively() {
    let Some(kernel) = loaded_kernel() else { return };
    // A branch between the declaration and the read COULD have bound
    // x on some path this straight-line tracking does not follow —
    // the conservative rule says no fire, even though this particular
    // program still never assigns x.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(flag: bool) -> Age:\n",
        "    x: int\n",
        "    if flag:\n",
        "        pass\n",
        "    return x\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| !f.message.contains("UnboundLocalError")),
        "a branch between declaration and read must suppress the fire conservatively: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- JUDGED LOOP BODIES (loops.rs's declared-slot judging) ---

#[test]
fn a_declared_slot_write_inside_a_while_body_fires_with_no_post_loop_read() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:495's own row: the marker sits INSIDE the loop
    // body, with no post-loop declared read to catch it — the fire
    // must come from loops.rs's own judging, not check.rs's ordinary
    // sink path.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    age: Age = 0\n",
        "    while age < 3:\n",
        "        age = age + 121\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the +121 step leaving the set must fire from inside the loop body: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'121'"), "{}", fires[0].message);
}

#[test]
fn a_declared_slot_write_from_a_dict_key_fires_instead_of_declining() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:508's own row: a String iterate written into a
    // declared Integer-sorted slot now fires through assignability::
    // judge rather than declining the whole loop.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    age: Age = 0\n",
        "    for key in {\"a\": 1, \"b\": 2}:\n",
        "        age = key\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "a string key into a declared int-sorted slot must fire, deduped once across both iterations: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the loop must still run to completion — no blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- LOOP ELSE + DEAD-ELSE LAW ---

#[test]
fn an_else_arm_write_fires_when_the_loop_never_breaks() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:446/472's own row: the else clause runs
    // (the loop never breaks), so its own out-of-set write fires —
    // check.rs walks orelse fully judged, not loops.rs.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    age: Age = 0\n",
        "    n = 0\n",
        "    while n < 3:\n",
        "        age = age + 1\n",
        "        n = n + 1\n",
        "    else:\n",
        "        age = 200\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the else arm's own write (200) must fire since the loop never breaks: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn an_else_arm_never_fires_its_own_write_when_the_loop_always_breaks() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:486's own row: the loop always breaks at i==1,
    // so the else clause never runs — its own out-of-set write
    // (200) must NOT fire; instead the dead-else law fires once,
    // naming why.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    age: Age = 0\n",
        "    for i in range(3):\n",
        "        if i == 1:\n",
        "            break\n",
        "        age = age + 1\n",
        "    else:\n",
        "        age = 200\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let two_hundred_fires: Vec<&Finding> =
        findings.iter().filter(|f| f.code == "RTS7001" && f.message.contains("'200'")).collect();
    assert!(
        two_hundred_fires.is_empty(),
        "the else arm's own write must never be walked when the loop always breaks: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let dead_else_fires: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("never runs"))
        .collect();
    assert_eq!(
        dead_else_fires.len(),
        1,
        "the dead-else law must fire exactly once naming why: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- EVALUATED ITERABLES ---

#[test]
fn a_tuple_element_that_evaluates_to_none_fires_into_a_non_optional_declared_slot() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements.py:541's own row: `unread_number()`'s body falls
    // off its end with no return, so the call answers None —
    // iterable_values now evaluates a non-literal tuple element
    // rather than declining the whole loop for a syntactic miss.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def unread_number() -> int: ...\n",
        "def f() -> Age:\n",
        "    age: Age = 0\n",
        "    for item in (unread_number(),):\n",
        "        age = item\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the tuple's evaluated element makes the loop concretely executable: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "None written into a non-Optional declared Age slot must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- MATCH JOIN FALLBACK ---

#[test]
fn a_class_pattern_as_capture_fires_inside_its_own_arm_on_an_undecidable_subject() {
    let Some(kernel) = loaded_kernel() else { return };
    // b-body-expressions.py:897-905's own row: `case int() as n:`
    // is a MatchClass wrapped in MatchAs — match_arms.rs cannot
    // decide TAKEN/NOT-TAKEN for a class pattern (Undecidable
    // regardless of the subject), so this fallback walks every arm
    // on a fork with `n` bound to the subject and fires from inside
    // the taken-in-practice arm.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    value = 200\n",
        "    match value:\n",
        "        case int() as n:\n",
        "            return n\n",
        "        case _:\n",
        "            return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a nameable class-pattern capture must not block the whole match: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the captured 200 must fire inside its own arm: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_class_pattern_as_capture_in_set_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    // b-body-expressions.py:886-894's own row: the in-set counterpart
    // — the same fallback must stay silent when the captured value
    // is inside the declared set.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    value = 40\n",
        "    match value:\n",
        "        case int() as n:\n",
        "            ok: Age = n\n",
        "            return ok\n",
        "        case _:\n",
        "            return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an in-set captured value must never fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_sequence_pattern_with_bare_name_elements_no_longer_blocks_the_whole_match() {
    let Some(kernel) = loaded_kernel() else { return };
    // `match_arms::pattern_bound_captures` names `a`/`b` positionally
    // (bare-Name elements over an UNKNOWN subject bind unknown(),
    // never a guess) — the match no longer needs its own blocker, and
    // an unreadable capture never fires (assignability's own law
    // never fires an Unknown value).
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(value) -> None:\n",
        "    match value:\n",
        "        case [a, b]:\n",
        "            pass\n",
        "        case _:\n",
        "            pass\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a sequence pattern's own bare-Name captures are nameable now: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// t-match-patterns.py's own `match_sequence_out_of_set_element` shape:
/// a KNOWN list literal subject lets `pattern_bound_captures` read the
/// bound element's REAL value positionally (`x` binds to `items[0]`,
/// 200) rather than `unknown()`, so the out-of-set read fires exactly
/// where the fixture expects — at the return, not at the match.
#[test]
fn a_sequence_pattern_over_a_known_list_subject_binds_elements_positionally_and_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    match [200, 10]:\n",
        "        case [x, _y]:\n",
        "            return x\n",
        "        case _:\n",
        "            return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a sequence pattern's own bare-Name captures are nameable: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the bound element 200 must fire at the return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// t-match-patterns.py's own `match_mapping_key_binding`/`match_
/// mapping_literal_out_of_set` shapes: a mapping pattern's literal-key
/// captures are nameable, and a known dict-literal subject lets
/// `pattern_bound_captures` read the bound key's REAL value.
#[test]
fn a_mapping_pattern_over_a_known_dict_subject_binds_the_keyed_value_and_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    match {\"age\": 200}:\n",
        "        case {\"age\": bound_age}:\n",
        "            return bound_age\n",
        "        case _:\n",
        "            return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a mapping pattern's own literal-key captures are nameable: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the bound value 200 must fire at the return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// t-match-patterns.py's own `match_class_out_of_set_attribute` shape:
/// a class pattern's KEYWORD sub-pattern captures are nameable, and a
/// known constructed-instance subject lets `pattern_bound_captures`
/// read the bound field's REAL value via `instances::field_read`.
#[test]
fn a_class_pattern_keyword_subpattern_over_a_known_instance_binds_the_field_and_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import BaseModel, Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Point(BaseModel):\n",
        "    x: int\n",
        "    y: int\n",
        "def f() -> Age:\n",
        "    match Point(x=200, y=10):\n",
        "        case Point(x=px):\n",
        "            return px\n",
        "        case _:\n",
        "            return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a class pattern's own keyword-subpattern captures are nameable: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the bound field 200 must fire at the return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// t-match-patterns.py's own `match_class_positional_pattern` shape:
/// POSITIONAL class-pattern sub-patterns still decline — resolving a
/// position to a field name needs `__match_args__` order, which
/// `match_arms::pattern_bound_captures` has no class table to read.
#[test]
fn a_class_pattern_with_positional_subpatterns_still_blocks_the_whole_match() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import BaseModel, Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Point(BaseModel):\n",
        "    x: int\n",
        "    y: int\n",
        "def f(shape: object) -> Age:\n",
        "    match shape:\n",
        "        case Point(px, _py):\n",
        "            return px\n",
        "        case _:\n",
        "            return 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        blockers.len(),
        1,
        "a positional class-pattern capture is unnameable without __match_args__ order: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- LAMBDA-ASSIGN LAW ---

#[test]
fn a_lambda_assigned_to_a_name_is_callable_through_that_name() {
    let Some(kernel) = loaded_kernel() else { return };
    // `f = lambda: 200` registers a synthetic def under `f`
    // (local_function_table) AND binds `f` to an opaque function
    // value; evaluate_call's gate dispatches through the function
    // table for a name bound only to an opaque function value, so
    // `f()` answers 200 end-to-end and the return sink fires.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def g() -> Age:\n",
        "    f = lambda: 200\n",
        "    return f()\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the lambda's 200 flows through f() into the return sink: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn local_function_table_registers_a_lambda_assign_as_a_callable_synthetic_def() {
    // Proves the LAMBDA-ASSIGN LAW's own infrastructure directly,
    // bypassing evaluate_call's environment-binding gate (the gap the
    // test above documents): the synthetic def IS correctly built and
    // IS answerable through summaries::call_result once looked up by
    // name — everything local_function_table itself is responsible
    // for.
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed("def g():\n    add_one = lambda x: x + 1\n    return 0\n");
    let Stmt::FunctionDef(g) = &module.body[0] else {
        panic!("module's one statement is def g")
    };
    let table = local_function_table(&g.body);
    let def = table.def("add_one").expect("the lambda-assign registers a synthetic def named add_one");
    assert_eq!(def.parameters.args.len(), 1, "the lambda's own parameter carries through");
    let result = crate::summaries::call_result(
        def,
        &[refined_domain::abstract_value::known_values(
            vec![120.0],
            refined_domain::abstract_value::PrimitiveKind::Integer,
            refined_domain::trust_grades::TrustProved,
        )],
        None,
        &kernel,
        0,
    )
    .expect("the synthetic def's body (return x + 1) answers through summaries::call_result");
    assert_eq!(result.values, vec![121.0]);
}
