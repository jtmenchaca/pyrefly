use super::*;
use ruff_text_size::Ranged;

#[test]
fn a_same_module_def_call_flows_a_known_return_into_a_declared_sink() {
    let Some(kernel) = loaded_kernel() else { return };
    // `over` is a module-level def, readable through
    // environment.functions() (walk_body seeds it on every body,
    // this module's own): the call resolves through
    // summaries::call_result and its known return (200) fires
    // against Age at the declared sink.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def over() -> int:\n",
        "    return 200\n",
        "def f() -> None:\n",
        "    x: Age = over()\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the same-module call's known return (200) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn an_imported_value_read_through_a_two_module_resolver_fires_at_a_return_sink() {
    let Some(kernel) = loaded_kernel() else { return };
    // A closure resolver over an in-memory map of module name ->
    // source text (cross_module.rs's own test pattern) stands in
    // for disk_resolver: `helper.py` states an out-of-set constant,
    // and the entry module's `from helper import over_years` makes
    // it readable at the return sink through context.module_bindings.
    let mut sources: HashMap<&str, &str> = HashMap::new();
    sources.insert("helper", "over_years = 200\n");
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "from helper import over_years\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    return over_years\n",
    ));
    let resolver: ModuleResolver = &|name: &str| sources.get(name).map(|source| parsed(source));
    let findings = findings_for_module_with_resolver(&module, resolver, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the imported constant (200) must fire at the return sink: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_keyword_construction_call_fires_on_an_out_of_set_field() {
    let Some(kernel) = loaded_kernel() else { return };
    // Person(age=200): a bare-Name construction call naming a
    // same-module class, judged through instances::judge_construction
    // — the keyword argument maps to the age field's own Annotated
    // set and fires. `type Age = ...` is declared (even though the
    // field spells its own inline Annotated[...]) because
    // findings_for_module's own aliases-gate returns nothing at all
    // for a module with zero type-alias statements.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, BaseModel\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Person(BaseModel):\n",
        "    age: Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    p = Person(age=200)\n",
        "    _ = p\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the keyword construction argument (200) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_provably_false_if_test_fires_and_its_body_is_never_walked() {
    let Some(kernel) = loaded_kernel() else { return };
    // a-statements:400's own shape: a helper whose every real return
    // is a live dict never answers None, so `held is None` is
    // provably false — the dead-branch law fires there, and the
    // out-of-set `return 200` inside that branch must never be
    // walked (no second RTS7001 for it).
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def helper_never_answers_none(adult: bool) -> dict[str, int] | None:\n",
        "    if adult:\n",
        "        return {\"age\": 40}\n",
        "    return {\"age\": 10}\n",
        "def f(adult: bool) -> Age:\n",
        "    held = helper_never_answers_none(adult)\n",
        "    if held is None:\n",
        "        return 200\n",
        "    return 40\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let dead_branch_fires: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
        .collect();
    assert_eq!(
        dead_branch_fires.len(),
        1,
        "the known-false `is None` test must fire exactly once: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let two_hundred_fires: Vec<&Finding> =
        findings.iter().filter(|f| f.code == "RTS7001" && f.message.contains("'200'")).collect();
    assert!(
        two_hundred_fires.is_empty(),
        "the dead branch's own `return 200` must never be walked: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_provable_raise_at_an_expr_statement_fires() {
    // Coded against expressions::provable_raise, landing in a
    // parallel follow-up unit — a known zero divisor
    // (`1 / 0`) is CPython's own unconditional ZeroDivisionError
    // (expressions.rst §6.7, division). Present per the mission's
    // instruction to leave this test in place, noted in the report,
    // rather than stubbing provable_raise here.
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    1 / 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "a known zero divisor is a provable raise and must fire once: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- STALE-RECEIVER SOUNDNESS, law (a): mutating method calls ---

#[test]
fn a_list_append_carries_the_new_element_into_a_later_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    ages = [40]\n",
        "    ages.append(200)\n",
        "    over: Age = ages[1]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the appended 200 must be visible at ages[1], not the stale pre-append list: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn an_unmodeled_mutating_method_forgets_the_receiver_rather_than_reading_the_stale_value() {
    let Some(kernel) = loaded_kernel() else { return };
    // `sort` is not in collection_models::mutated_receiver's modeled
    // row set — the receiver must be forgotten (Undetermined), never
    // left bound to its pre-call value.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    ages = [40, 200]\n",
        "    ages.sort()\n",
        "    over: Age = ages[0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "an unmodeled mutator must forget the receiver, never fire on its stale value: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- STALE-RECEIVER SOUNDNESS, law (b): subscript-target writes ---

#[test]
fn a_dict_item_write_carries_the_new_value_into_a_later_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    ages: dict[str, int] = {}\n",
        "    ages[\"ann\"] = 200\n",
        "    over: Age = ages[\"ann\"]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the written 200 must be visible at ages[\"ann\"], not the stale empty dict: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_list_item_write_carries_the_new_value_into_a_later_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    ages = [40, 41]\n",
        "    ages[0] = 200\n",
        "    over: Age = ages[0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the written 200 must be visible at ages[0]: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

// --- KNOWN-TUPLE DESTRUCTURING (law 2) ---

#[test]
fn a_known_tuple_target_binds_each_position_and_judges_it() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    a: Age\n",
        "    b: Age\n",
        "    a, b = (200, 40)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "only a's position (200) is out of set; b's (40) is in set: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_starred_target_binds_the_head_and_the_middle_list() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    first, *rest = [200, 20, 30]\n",
        "    over: Age = first\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the starred target's head element (200) must bind and judge: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_length_mismatch_unpack_of_a_known_list_fires_value_error_and_forgets_every_target() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    a, b = (1, 2, 3)\n",
        "    over: Age = a\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let raises: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably raises ValueError"))
        .collect();
    assert_eq!(
        raises.len(),
        1,
        "a 3-item tuple unpacked into 2 targets provably raises ValueError: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(
        raises[0].message.contains("too many values to unpack (expected 2)"),
        "{}",
        raises[0].message
    );
    let age_fires: Vec<&Finding> =
        findings.iter().filter(|f| f.code == "RTS7001" && f.message.contains("'Age'")).collect();
    assert!(
        age_fires.is_empty(),
        "every target must be forgotten after the raise — no second fire reading 'a': {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// `arm_terminates_or_provably_raises` treats a body whose last
/// statement is NOT syntactically `return`/`raise`, but that the
/// walk's own provable-raise machinery already fired an RTS7001 for,
/// as terminating — the same as a bare `raise`. A plain `Assign` with
/// no recorded fire must NOT be treated as terminating; only tacking
/// a genuine RTS7001 finding, anchored inside that statement's own
/// range, onto the body flips the answer.
#[test]
fn arm_terminates_or_provably_raises_treats_a_provable_raise_as_terminal() {
    let module = parsed(concat!(
        "def f() -> None:\n",
        "    a, b = (1, 2, 3)\n",
    ));
    let Stmt::FunctionDef(def) = &module.body[0] else { panic!("a function def") };
    let Stmt::Assign(assign) = &def.body[0] else { panic!("an assign") };
    let body = std::slice::from_ref(&def.body[0]);

    let no_findings: Vec<Finding> = Vec::new();
    assert!(
        !arm_terminates_or_provably_raises(body, &no_findings, 0),
        "a plain Assign with no recorded raise must not read as terminal"
    );

    let with_a_raise = vec![Finding {
        range: assign.value.range(),
        code: "RTS7001",
        message: "this expression provably raises ValueError: too many values to unpack (expected 2)".to_owned(),
    }];
    assert!(
        arm_terminates_or_provably_raises(body, &with_a_raise, 0),
        "an RTS7001 anchored inside the last statement's own range must count as terminal"
    );
}

/// A `try` whose body provably raises (an arity-mismatch unpack) and
/// whose sole handler itself terminates (`except ValueError: return
/// first`) leaves `walk_try`'s own `surviving` list empty — this
/// statement never falls through. Statements written AFTER the try
/// describe only unreachable code, so they must not be walked for
/// judgement: a read of a name the try body never got to bind must
/// not report an unreadable-value blocker.
///
/// The raise itself is SPOKEN: `except ValueError` catching the
/// `ValueError` the unpack provably raises changes where control goes,
/// never whether the raise is reported — a provable raise is always
/// spoken (`walk_try`'s own caught-raise doc). The uncaught twin below
/// fires the same finding and differs only in reachability.
#[test]
fn a_try_whose_every_arm_terminates_stops_the_body_walk_at_the_try() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    first = 40\n",
        "    triple = (200, 201, 202)\n",
        "    try:\n",
        "        over_first, over_second = triple\n",
        "    except ValueError:\n",
        "        return first\n",
        "    return over_first\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let raises: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably raises ValueError"))
        .collect();
    assert_eq!(
        raises.len(),
        1,
        "a provable raise is spoken even when `except ValueError` catches it: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the unreachable `return over_first` past the terminating try must not report a blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The UNCAUGHT twin of the row above: the same arity-mismatch unpack in
/// the same `try`, but the handler names a DIFFERENT exception class
/// (`except KeyError`), which never catches a `ValueError`. Nothing
/// transfers to the handler and the raise escapes the function, so
/// besides firing the same finding the body walks on and the try path
/// is decided by `arm_terminates_or_provably_raises`.
#[test]
fn a_provable_raise_no_handler_catches_still_fires_inside_a_try() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    first = 40\n",
        "    triple = (200, 201, 202)\n",
        "    try:\n",
        "        over_first, over_second = triple\n",
        "    except KeyError:\n",
        "        return first\n",
        "    return first\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let raises: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably raises ValueError"))
        .collect();
    assert_eq!(
        raises.len(),
        1,
        "a ValueError no `except KeyError` catches must still fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The caught-raise rule's two halves in one body, so neither can be
/// traded for the other. The unpack provably raises `ValueError` and
/// `except ValueError` catches it: the raise's own finding is SPOKEN
/// (reporting does not depend on the catch), and the statement AFTER
/// it in the try body — an out-of-set write to a declared `Age` slot
/// that would fire on its own — is never walked, because control left
/// the body at the raise.
#[test]
fn a_caught_provable_raise_is_spoken_and_stops_the_try_body_walk() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> None:\n",
        "    triple = (200, 201, 202)\n",
        "    try:\n",
        "        over_first, over_second = triple\n",
        "        past: Age = 300\n",
        "    except ValueError:\n",
        "        return\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let messages: Vec<&String> = findings.iter().map(|f| &f.message).collect();
    let raises: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably raises ValueError"))
        .collect();
    assert_eq!(
        raises.len(),
        1,
        "the caught raise is spoken: {:?}",
        messages
    );
    assert!(
        !messages.iter().any(|message| message.contains("'300'")),
        "the statement after the raise never runs, so its own out-of-set write must not be judged: {:?}",
        messages
    );
}

// --- HANDLER AS-NAME (law 3) ---

#[test]
fn a_caught_exception_bound_to_a_declared_int_slot_fires_through_the_opaque_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    try:\n",
        "        raise ValueError(1)\n",
        "    except ValueError as error:\n",
        "        return error\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    // The handler's as-name must be bound to something (not forgotten
    // at entry) — an Undetermined blocker at worst, or a Fire once
    // assignability reads the opaque marker. Either way it must not
    // be silently absent from the findings the way "forget" would
    // leave it (no finding at all).
    assert!(
        !findings.is_empty(),
        "a caught exception returned under a declared int-sorted set must not pass silently: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
