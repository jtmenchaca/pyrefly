use super::*;

/// m-pydantic-schema.py's `parse_number_chain_ok`/`_over_ceiling` own
/// shape: `TypeAdapter(Age).validate_python(<int>)` where `Age` is a
/// bare alias name, not a `BaseModel` class — the class route in
/// `construction_call_verdict` misses (`context.classes` has no
/// entry), so the adapter-alias route must judge the argument
/// directly against `Age`'s own declared set.
#[test]
fn type_adapter_validate_python_on_an_alias_judges_the_scalar_argument() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def ok() -> Age:\n",
        "    return TypeAdapter(Age).validate_python(40)\n",
        "def over() -> Age:\n",
        "    return TypeAdapter(Age).validate_python(200)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the in-set validate_python(40) must stay silent, and only the \
         out-of-set validate_python(200) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// UNIT 3, sites 6 and 7 (`adapter_alias_verdict`'s Fire and
/// Undetermined arms, both built through the shared
/// `declared_set_instance`): `Age`'s own declared set is
/// numeric-ground, so the kept instance must carry `kind_tag:
/// Some(Integer)` in EITHER arm — bound at its own direct sink
/// (`year: Age = TypeAdapter(Age).validate_python(...)`, since
/// `construction_call_verdict`, like `callable_variable_call_
/// result`, only reads a call at `sink_value`'s own value-expression
/// position), then piped through `math.sqrt` (a sort-gated
/// consumer, `sqrt_call_over_set`, math_models.rs), which now
/// derives a value instead of leaving the return undetermined.
#[test]
fn adapter_alias_verdicts_fire_arm_kept_instance_reaches_sqrt_tagged() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "import math\n",
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
        "def over() -> Root:\n",
        "    year: Age = TypeAdapter(Age).validate_python(200)\n",
        "    return math.sqrt(year)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        fires.len(),
        1,
        "the out-of-set validate_python(200) still fires once: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(
        blockers.is_empty(),
        "the tagged kept instance must let math.sqrt derive rather than blocking: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// `adapter_alias_verdict`'s Undetermined arm itself records no
/// finding of its own (only the Fire arm carries one, via
/// `ConstructionVerdict.fires`) — it keeps `Age`'s own declared set
/// as the instance, which the enclosing `year: Age = …` AnnAssign
/// then judges a second time and finds a trivial self-match
/// (Silent). This test pins that the WHOLE statement stays exactly
/// as silent as it already was before this unit's tag — the
/// observable difference the tag makes is downstream, at
/// `math.sqrt(year)`: an untagged kept instance would leave that
/// call's own return undetermined; the tagged one lets it derive.
#[test]
fn adapter_alias_verdicts_undetermined_arm_kept_instance_reaches_sqrt_tagged() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "import math\n",
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
        "class AudioRequest:\n",
        "    def __init__(self, level) -> None:\n",
        "        self.level = level\n",
        "def f(request: AudioRequest) -> Root:\n",
        // request.level reads back as unknown() — an unmodeled field —
        // so validate_python's own argument judges Undetermined, the
        // arm this test pins.
        "    year: Age = TypeAdapter(Age).validate_python(request.level)\n",
        "    return math.sqrt(year)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "the tagged kept instance must let math.sqrt derive with no fire and no blocker: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
}

/// m-pydantic-schema.py's `parse_string_chain_over_length` shape: a
/// STRING-sorted alias (`Label`, min_length/max_length window) judges
/// its adapter argument the same way.
#[test]
fn type_adapter_validate_python_on_a_string_alias_fires_over_length() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type Label = Annotated[str, Field(min_length=1, max_length=8)]\n",
        "def over() -> Label:\n",
        "    return TypeAdapter(Label).validate_python(\"too-long-string\")\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
}

/// m-pydantic-schema.py's `safe_parse_refused_reified` shape: the
/// adapter-alias route's own RTS7001 fire, inside a `try` body, is
/// reified by the SAME try/except machinery every other provable
/// raise already uses — no special-casing needed once the fire
/// itself lands.
#[test]
fn type_adapter_validate_python_fire_inside_try_is_reified_by_the_except_arm() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def safe_parse_refused_reified() -> Age:\n",
        "    try:\n",
        "        return TypeAdapter(Age).validate_python(200)\n",
        "    except ValueError:\n",
        "        return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// m-pydantic-schema.py's `parse_lax_coercion_ok`/`_out_of_range` own
/// shape: a lax (non-`StrictInt`) `int` alias coerces a plain digit
/// string before judging (execution-verified against pydantic 2.13.4:
/// `"40"` coerces to `40`, `"200"` coerces to `200` and then fails the
/// range bound).
#[test]
fn type_adapter_validate_python_lax_int_alias_coerces_a_digit_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
        "def ok() -> LaxAge:\n",
        "    return TypeAdapter(LaxAge).validate_python(\"40\")\n",
        "def over() -> LaxAge:\n",
        "    return TypeAdapter(LaxAge).validate_python(\"200\")\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the coerced-in-range \"40\" must stay silent, and only the coerced-\
         out-of-range \"200\" must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// m-pydantic-schema.py's `parse_strict_int_ok`/`_refuses_string` own
/// shape: a `StrictInt`-based alias never coerces a string argument —
/// a genuine int is admitted, a numeric string fires the ordinary
/// string-vs-numeric-ground sort mismatch (StrictInt's own refusal,
/// execution-verified: `.validate_python("40")` raises `int_type` with
/// no coercion attempt).
#[test]
fn type_adapter_validate_python_strict_int_alias_refuses_a_digit_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, StrictInt, TypeAdapter\n",
        "type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]\n",
        "def ok() -> StrictAge:\n",
        "    return TypeAdapter(StrictAge).validate_python(40)\n",
        "def refused() -> StrictAge:\n",
        "    return TypeAdapter(StrictAge).validate_python(\"40\")\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the genuine int 40 must stay silent, and only the numeric string \
         \"40\" must fire (StrictInt never coerces): {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("not assignable"), "{}", fires[0].message);
}

/// m-pydantic-schema.py's `parse_pattern_ok` shape: a STR-sorted
/// pattern alias (`Digits`, `Annotated[str, Field(pattern=r"^[0-9]+$")]`)
/// must NOT run the lax-int digit-string coercion — a digit-only
/// STRING is exactly what a `str`-sorted pattern alias accepts on its
/// own terms, so `TypeAdapter(Digits).validate_python("42")` judges
/// the string "42" (2 codepoints, inside the pattern/length window)
/// as a string, never rewritten to the int 42 first. Before gating
/// `adapter_alias_verdict`'s coercion on `requires_integer(declared_set)`,
/// this row wrongly fired (the digit-only string coerced to an int,
/// then the resulting Integer-vs-str-sorted-set mismatch fired) —
/// this test pins the fix.
#[test]
fn type_adapter_validate_python_str_sorted_pattern_alias_never_coerces_a_digit_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type Digits = Annotated[str, Field(min_length=1, max_length=4, pattern=r\"^[0-9]+$\")]\n",
        "def ok() -> Digits:\n",
        "    return TypeAdapter(Digits).validate_python(\"42\")\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert!(
        fires.is_empty(),
        "a digit-only string against a str-sorted pattern alias must judge AS a \
         string, never coerced to an int first: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// m-pydantic-schema.py's `parse_lax_coercion_out_of_range` shape,
/// re-asserted alongside the str-sorted-alias fix above: an
/// INT-sorted lax alias must still coerce a digit string and fire
/// once its coerced value leaves the range — the fix narrows the
/// coercion to numeric-sorted aliases, it must not also narrow it
/// away from the int-sorted case that motivated it.
#[test]
fn type_adapter_validate_python_int_sorted_lax_alias_still_coerces_and_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field, TypeAdapter\n",
        "type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
        "def over() -> LaxAge:\n",
        "    return TypeAdapter(LaxAge).validate_python(\"200\")\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the coerced-out-of-range \"200\" must still fire against an int-sorted alias: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// m-pydantic-schema.py's `parse_literal_ok`/`_outside` shape: a bare
/// `type Pick = Literal[10, 20, 30]` alias (`surface::literal_alias_set`)
/// judges its adapter argument through the exact same route as a
/// scalar `Annotated[...]`-compiled alias.
#[test]
fn type_adapter_validate_python_on_a_literal_alias_fires_outside_every_member() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Literal\n",
        "from pydantic import TypeAdapter\n",
        "type Pick = Literal[10, 20, 30]\n",
        "def ok() -> Pick:\n",
        "    return TypeAdapter(Pick).validate_python(20)\n",
        "def outside() -> Pick:\n",
        "    return TypeAdapter(Pick).validate_python(25)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the in-set validate_python(20) must stay silent, and only the \
         out-of-set validate_python(25) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'25'"), "{}", fires[0].message);
}

/// m-pydantic-schema.py's `parse_union_ok`/`_outside` shape: a
/// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]`
/// union alias (`surface::literal_union_alias_set`) judges a member of
/// EITHER arm as silent and a value in neither arm as a fire — the
/// kernel's `memberB` derivative walk decides membership over the
/// whole union set regardless of which arm's sort a given probe value
/// carries (`RefinedSet.memberB_iff`, refined-ts-lean/set_functions/
/// membership.lean: total and proved over any concrete tuple).
#[test]
fn type_adapter_validate_python_on_a_literal_union_alias_fires_outside_both_arms() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Literal\n",
        "from pydantic import TypeAdapter\n",
        "type PickUnion = Literal[10, 20, 30] | Literal[\"ten\", \"twenty\"]\n",
        "def ok() -> PickUnion:\n",
        "    return TypeAdapter(PickUnion).validate_python(\"ten\")\n",
        "def outside() -> PickUnion:\n",
        "    return TypeAdapter(PickUnion).validate_python(25)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the in-set validate_python(\"ten\") must stay silent, and only the \
         out-of-both-arms validate_python(25) must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'25'"), "{}", fires[0].message);
}
