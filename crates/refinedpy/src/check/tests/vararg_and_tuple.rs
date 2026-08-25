use super::*;

/// e-class-and-function.py's own `first_age`/`rest_parameter` shape
/// end to end: `*ages: int` genuinely binds a known tuple of the
/// caller's trailing arguments (`summaries::bind_parameters`'s own
/// vararg row), so an IN-SET call stays silent and an OUT-OF-SET call
/// fires exactly once, at the offending argument's own value — never
/// a wrong fire on the in-set call from `return_sort_fallback`'s own
/// coarse `-> int` claim (item 1's own regression).
#[test]
fn a_vararg_def_interprets_concretely_instead_of_firing_the_coarse_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def first_age(*ages: int) -> int:\n",
        "    return ages[0]\n",
        "def rest_parameter() -> Age:\n",
        "    good: Age = first_age(40, 41)\n",
        "    _ = good\n",
        "    return first_age(200, 201)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the in-set first_age(40, 41) call must stay silent, and only the \
         out-of-set first_age(200, 201) call must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// q-decline-names.py:131-144's own `sum_rest`/`rest_parameter_
/// coverage` shape: a `for value in rest:` loop over a `*rest: int`
/// vararg, walked as `sum_rest`'s OWN straight-line body — no call
/// site here for `summaries::bind_parameters` to seed the tuple from
/// (that path is exercised by `a_vararg_def_interprets_concretely_
/// instead_of_firing_the_coarse_fallback` above). `seed_parameters`
/// must seed `rest` itself as an unbounded int-sorted repetition
/// window so `loops.rs::repetition_window_element_pass` can iterate
/// it, rather than leaving `rest` unbound and declining the whole
/// loop with the coarse "a for statement is not yet walked" blocker.
#[test]
fn a_vararg_rest_parameter_iterates_in_its_own_straight_line_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def sum_rest(first: int, *rest: int) -> int:\n",
        "    total = first\n",
        "    for value in rest:\n",
        "        total = total + value\n",
        "    return total\n",
        "def rest_parameter_coverage() -> Age:\n",
        "    ok: Age = sum_rest(40, 0)\n",
        "    del ok\n",
        "    return sum_rest(200, 0)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the vararg's own body walk must not decline the for loop over `rest`: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// e-class-and-function.py's own `unpack_first`/`unpacking_in_body`
/// shape end to end: `a, _b = ages` (a tuple-unpack `Assign` target)
/// genuinely binds against the known tuple parameter
/// (`summaries::bind_unpack_target`), so the in-set call stays silent
/// and the out-of-set call fires exactly once — never a wrong fire
/// from the coarse `-> int` fallback on a body that should have
/// interpreted concretely.
#[test]
fn a_tuple_unpack_assign_in_a_summarized_body_interprets_concretely() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def unpack_first(ages: tuple[int, int]) -> int:\n",
        "    a, _b = ages\n",
        "    return a\n",
        "def unpacking_in_body() -> Age:\n",
        "    good: Age = unpack_first((40, 41))\n",
        "    _ = good\n",
        "    return unpack_first((200, 201))\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the in-set unpack_first((40, 41)) call must stay silent, and only the \
         out-of-set unpack_first((200, 201)) call must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}
