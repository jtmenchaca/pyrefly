use super::*;

#[test]
fn an_if_else_join_carries_an_out_of_set_arm_into_a_judged_row() {
    let Some(kernel) = loaded_kernel() else { return };
    // one arm binds x to an in-set value, the other to an
    // out-of-set value; the join keeps both possibilities, so the
    // kernel must see the union and fire on the out-of-set member.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(flag: bool) -> None:\n",
        "    if flag:\n",
        "        x = 40\n",
        "    else:\n",
        "        x = 200\n",
        "    y: Age = x\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
}

#[test]
fn an_aug_assign_out_of_the_recorded_set_fires_at_the_statement() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    x: Age = 40\n",
        "    x += 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.contains("may write 240,"), "{}", fires[0].message);
}

#[test]
fn a_class_body_out_of_set_field_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Person:\n",
        "    age: Age = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
}

#[test]
fn del_and_assert_bodies_record_no_blocker() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    x: Age = 40\n",
        "    assert x\n",
        "    del x\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "assert/del must record no blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_value_less_declaration_then_plain_assign_fires_at_the_assign() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    a: Age\n",
        "    a = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_chained_multi_target_assign_fires_once_per_declared_target() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    a: Age\n",
        "    b: Age\n",
        "    a = b = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        2,
        "both a and b are declared Age, so the chained refusal fires once per target: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_refused_write_keeps_the_declared_set_so_a_later_return_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    a: Age\n",
        "    a = 200\n",
        "    return a\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the write fires once; the return of the refused-but-declared slot must not fire again: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// UNIT 3, site 3 (`judge_and_bind`'s Fire arm): the refused-but-
/// declared slot `a` carries `Age`'s own numeric-ground set, tagged
/// with its sort — so `math.sqrt(a)`, a sort-gated consumer
/// (`sqrt_call_over_set`, math_models.rs) that refuses an untagged
/// set, now derives a value instead of leaving the return
/// undetermined.
#[test]
fn a_refused_writes_declared_set_reaches_sqrt_tagged() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "import math\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
        "def f() -> Root:\n",
        "    a: Age\n",
        "    a = 200\n",
        "    return math.sqrt(a)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    // the assign's own refusal fires once; the sqrt-derived return
    // must stay SILENT (Root's own [0, 20] window covers sqrt(Age)'s
    // [0, sqrt(120)] range) rather than adding an undetermined
    // blocker for a return this fix now derives.
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(
        blockers.is_empty(),
        "the tagged slot must let math.sqrt derive rather than blocking: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn an_undeclared_names_assign_still_binds_without_judging() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    plain = 200\n",
        "    plain = 300\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an undeclared name's assign must never judge: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_literal_range_for_loop_accumulates_and_the_out_of_set_total_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    // loop_final_environment runs [200] concretely (a single-element
    // literal list, a shape it CAN execute), leaving `total` bound to
    // 200 with no blocker; the read afterward judges that value.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    total: Age = 0\n",
        "    for x in [200]:\n",
        "        total = x\n",
        "    over: Age = total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a concretely-executable for loop must record no blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the post-loop read of total (200, the loop's last element) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_for_loop_over_an_unknown_iterable_blocks_and_forgets_its_stale_binding() {
    let Some(kernel) = loaded_kernel() else { return };
    // `items` is an unannotated parameter, so its value is unknown —
    // literal_iterable_values cannot read it and loop_final_environment
    // declines. `total` held an OUT-OF-SET literal immediately before
    // the loop; had the blocker path left that stale fact bound, the
    // read after the loop would fire a second time on it. The fix
    // forgets `total` (and the loop's own target `x`) at the blocker,
    // so the post-loop read is Undetermined, not a second Fire.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(items) -> None:\n",
        "    total = 200\n",
        "    for x in items:\n",
        "        total = 5\n",
        "    check: Age = total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        blockers.len(),
        1,
        "the unmodeled for loop is this body's one blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(blockers[0].message.contains("for"), "{}", blockers[0].message);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert!(
        fires.is_empty(),
        "total's stale pre-loop value (200) must not survive to fire after an unmodeled loop: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_declined_loop_forgets_a_receiver_only_ever_touched_through_a_chained_mutating_call() {
    let Some(kernel) = loaded_kernel() else { return };
    // `grouped` is never itself the target of `=` inside the loop body
    // — it is only read as the receiver of a CHAINED call
    // (`grouped.setdefault(...)` returns a value that `.append(...)`
    // is then called on). `run_expr_statement_once` (loops.rs) only
    // replays a mutating call whose receiver is a bare Name, so this
    // shape declines the whole loop. Before the fix, `grouped` was
    // never named by `collect_bound_names_stmt`'s scan (it is
    // MUTATED, never ASSIGNED), so the blocker path left it bound to
    // its stale pre-loop empty dict — and a post-loop
    // `grouped["young"]` read would then be a WRONG ANSWER: a
    // provable KeyError fire on a key the (unread) mutation actually
    // wrote (c-reads-and-values.py:1008). The fix forgets `grouped`
    // at the blocker, so the post-loop read is Undetermined, not a
    // false provable-raise fire.
    // `.extend` on the setdefault entry is OUTSIDE the executor's
    // recognized `.setdefault(...).append(...)` shape, so this loop
    // still declines — which is exactly what this test needs: the
    // forget rule at the blocker, not the served path.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    grouped: dict[str, list[int]] = {}\n",
        "    for age in [40, 200]:\n",
        "        grouped.setdefault(\"young\", []).extend([age])\n",
        "    check: Age = grouped[\"young\"][0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        blockers.len(),
        1,
        "the unmodeled for loop is this body's one blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let raises: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("KeyError"))
        .collect();
    assert!(
        raises.is_empty(),
        "grouped's stale pre-loop empty dict must not survive to falsely prove a KeyError: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_match_on_a_known_subject_takes_its_arm_and_fires_inside_it() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    x = 1\n",
        "    match x:\n",
        "        case 1:\n",
        "            over: Age = 200\n",
        "        case _:\n",
        "            pass\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a match on a known subject must record no blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "only the taken arm (case 1) is walked, and it fires on 200: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_with_body_still_judges_and_records_no_blocker_for_the_with() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(cm) -> None:\n",
        "    with cm as ctx:\n",
        "        over: Age = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a with statement must record no blocker of its own: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the with body's AnnAssign still fires on 200: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_try_body_out_of_set_ann_assign_fires_with_no_blocker_for_the_try() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    try:\n",
        "        over: Age = 200\n",
        "    except Exception:\n",
        "        pass\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "a try statement must record no blocker of its own: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the try body's AnnAssign still fires on 200: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_try_except_join_does_not_carry_the_declared_slots_pre_try_out_of_set_value() {
    let Some(kernel) = loaded_kernel() else { return };
    // total starts OUT of Age's set (200, fires once). The try body
    // rebinds it in-set and then returns, so the try path never
    // survives to the join — only the handler does. `total` is bound
    // BEFORE the try and written inside it, so the handler holds the
    // join of its pre-try value with unknown, which is unknown — the
    // stale 200 is not a value the post-try read can judge. Had the
    // pre-try value carried through instead, the final read below would
    // fire a SECOND time on it.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    total: Age = 200\n",
        "    try:\n",
        "        total = 40\n",
        "        return\n",
        "    except Exception:\n",
        "        pass\n",
        "    check: Age = total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "only the pre-try declaration's own refusal (200) may fire — total must not carry its stale pre-try value through the join: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    let try_blockers: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7002" && f.message.contains("try statement"))
        .collect();
    assert!(
        try_blockers.is_empty(),
        "the try statement itself must never be recorded as a blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A name bound BEFORE the try and never written inside it stays BOUND
/// in the handler — the write set is what the handler adjusts, and this
/// name is not in it. The pre-try declaration fires once on 200; the
/// refused-write law then keeps the DECLARED set on the name, so the
/// handler's own read judges against Age's set and fires nothing — one
/// defect, reported once at the statement that introduced it.
#[test]
fn a_name_the_try_body_never_writes_keeps_its_value_in_the_handler() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    kept: Age = 200\n",
        "    try:\n",
        "        other = 1\n",
        "    except Exception:\n",
        "        check: Age = kept\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "exactly the pre-try declaration's fire on 200: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let undetermined: Vec<&Finding> =
        findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        undetermined.is_empty(),
        "the handler's read of the untouched name is determined, never blocked: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A name FIRST bound inside the try body stays forgotten in the
/// handler: the handler may run before that statement ever binds it
/// (compound_stmts.rst, "The `try` statement" — an exception may
/// interrupt the body at any point), so there is no value to serve and
/// no pre-try value to join with.
#[test]
fn a_name_first_bound_inside_the_try_body_is_forgotten_in_the_handler() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    try:\n",
        "        fresh = 200\n",
        "    except Exception:\n",
        "        check: Age = fresh\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert!(
        fires.is_empty(),
        "the handler read has no value to judge, so nothing fires there: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
