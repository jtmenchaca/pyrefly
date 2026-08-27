use super::*;
use refined_domain::abstract_value::PrimitiveKind;

#[test]
fn a_list_element_compound_write_past_the_declared_ceiling_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    // ages[0] is 2 after the `//= 2`; += 190 writes 192, past Age's
    // 120 ceiling — the element-level judging `walk_subscript_
    // aug_assign` now applies through `aug_assign_refinements`'
    // `list[Age]` entry.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    ages: list[Age] = [10, 20]\n",
        "    ages[0] //= 5\n",
        "    ages[0] += 190\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "want exactly one fire for the 192 write: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// UNIT 3, site 4 (`walk_subscript_aug_assign`'s element Fire arm):
/// the refused element write keeps `Age`'s own numeric-ground set,
/// tagged with its sort — a later `ages[0]` read reaches
/// `math.sqrt` (a sort-gated consumer, `sqrt_call_over_set`,
/// math_models.rs) and derives a value instead of leaving the
/// return undetermined.
#[test]
fn a_refused_element_writes_declared_set_reaches_sqrt_tagged() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "import math\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
        "def rows() -> Root:\n",
        "    ages: list[Age] = [10, 20]\n",
        "    ages[0] //= 5\n",
        "    ages[0] += 190\n",
        "    return math.sqrt(ages[0])\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        fires.len(),
        1,
        "the 192 write still fires once: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(
        blockers.is_empty(),
        "the tagged element set must let math.sqrt derive rather than blocking: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_list_element_compound_write_inside_the_declared_ceiling_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    ages: list[Age] = [10, 20]\n",
        "    ages[0] += 5\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "an in-range element write must never fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn an_alias_the_table_cannot_lower_declines_whole() {
    let Some(kernel) = loaded_kernel() else { return };
    // json_schema_extra is not on the inert list and not a bound —
    // the alias refuses, so neither line judges.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Odd = Annotated[int, Field(ge=0, json_schema_extra={})]\n",
        "def rows() -> None:\n",
        "    fine: Odd = 5\n",
        "    wild: Odd = -200\n",
    ));
    assert!(findings_for_module(&module, &kernel).is_empty());
}

#[test]
fn a_body_that_rebinds_the_alias_name_blocks_instead_of_judging() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    Age = 5\n",
        "    x: Age = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "a rebound alias name must never judge: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(blockers.len(), 1, "want exactly one blocker: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(
        blockers[0].message.contains("rebound"),
        "{}",
        blockers[0].message
    );
}

#[test]
fn one_blocker_and_the_judged_fire_both_land_in_the_same_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    while True:\n",
        "        pass\n",
        "    over: Age = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        blockers.len(),
        1,
        "want exactly one blocker (the while): {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(
        blockers[0].message.contains("while"),
        "{}",
        blockers[0].message
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "want the judgeable AnnAssign to still fire after the blocker: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

#[test]
fn a_body_never_records_more_than_one_blocker() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    while True:\n",
        "        pass\n",
        "    for i in range(3):\n",
        "        pass\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert_eq!(
        blockers.len(),
        1,
        "want at most one blocker per body: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

#[test]
fn a_return_out_of_the_declared_set_fires_at_the_return() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Age:\n",
        "    return 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// UNIT 1 (fully diagnosed, runtime-verified): a bare `return
/// sum(<generator>)` with no assignment anywhere in the body used to
/// fall through to the ordinary evaluator's `sum_call_over_star` row
/// (`builtin_models.rs`), which needs a known-sign element hull and
/// declines outright on a sign-straddling `[-1, 1]` element — even
/// though the byte-identical computation, spelled `total =
/// sum(...); return total`, already recognized and judged through
/// `recognize_generator_sum`'s own Assign-only reader. Both spellings
/// here run over `samples: list[Sample]` with `Sample`'s own hull
/// straddling zero, and both must derive the SAME silent verdict
/// against `-> Total` (`[-10, 10]`, the relational ledger's own tight
/// total for up to 10 elements each in `[-1, 1]`, with no
/// `min_length` stated so the count's own lower bound is 0) — never
/// a blocker for one and a judged silence for the other.
#[test]
fn a_bare_return_of_a_generator_sum_judges_identically_to_its_assign_then_return_twin() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Sample = Annotated[float, Field(ge=-1.0, le=1.0)]\n",
        "type Total = Annotated[float, Field(ge=-10.0, le=10.0)]\n",
        "def bare_return(samples: Annotated[list[Sample], Field(max_length=10)]) -> Total:\n",
        "    return sum(s for s in samples)\n",
        "def assign_then_return(samples: Annotated[list[Sample], Field(max_length=10)]) -> Total:\n",
        "    total = sum(s for s in samples)\n",
        "    return total\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "both spellings derive the same silent [-10, 10] verdict, want no findings: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
}

// --- yield/return inside a Generator[...]-annotated body ---

/// i-more-expressions.py's own `yield_expression` shape:
/// `Generator[Age, None, Age]` makes both a `yield 200` and a
/// `return 200` checked positions — one fire each, an in-set
/// `yield 40` stays silent.
#[test]
fn a_yield_and_a_return_out_of_the_declared_generator_set_each_fire() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Generator\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Generator[Age, None, Age]:\n",
        "    yield 40\n",
        "    yield 200\n",
        "    return 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 2, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
    assert!(fires[1].message.contains("'200'"), "{}", fires[1].message);
}

/// A non-generator body's `-> Age` never turns a `yield` inside a
/// DIFFERENT, non-generator function into a checked position — this
/// test pins that `yield_refinement` stays `None` outside a
/// generator-shaped body by checking a plain `-> Age` function's own
/// return still judges normally alongside an unrelated generator.
#[test]
fn a_bare_yield_judges_as_none_against_the_declared_yield_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Generator\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f() -> Generator[Age, None, Age]:\n",
        "    yield\n",
        "    return 40\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.to_lowercase().contains("none"), "{}", fires[0].message);
}

/// `yield from` delegating to a same-module generator whose own body
/// yields an out-of-set value: the delegate's ACTUAL yields (read
/// through `instances::generator_yields`, tighter than its own bare
/// declared annotation) are what judge — `over_inner()`'s single
/// `yield 200` fires against the outer `Age` set.
#[test]
fn a_yield_from_delegate_whose_own_body_yields_out_of_set_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Generator\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def over_inner() -> Generator[int, None, None]:\n",
        "    yield 200\n",
        "def f() -> Generator[Age, None, None]:\n",
        "    yield from over_inner()\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(fires.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// A generator body's own IN-SET yields stay silent, including a
/// `yield from` delegate whose actual yields all sit inside the
/// outer set.
#[test]
fn a_generator_body_entirely_in_set_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Generator\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def inner() -> Generator[int, None, None]:\n",
        "    yield 40\n",
        "def f() -> Generator[Age, None, Age]:\n",
        "    yield 40\n",
        "    yield from inner()\n",
        "    return 40\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an entirely in-set generator body must stay silent: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// UNIT 3, site 2 (`delegated_generator_yields`'s declared-annotation
/// fallback): a delegate whose body-walk `instances::generator_yields`
/// permanently declines (a CONDITIONAL yield, `if flag: yield <expr>`
/// — that function's own doc names this the deliberate boundary) falls
/// to its bare `-> Generator[Age, None, None]` annotation instead.
/// `Age`'s own set is numeric-ground, so the delegated value must
/// carry `kind_tag: Some(Integer)` — the tag `min_max_scalar_operand`/
/// `star_numeric_hull`/`sum_call_over_star` (builtin_models.rs) read,
/// and the sort-gated consumers that used to refuse an untagged set.
#[test]
fn a_delegates_declared_yield_annotation_is_tagged_when_its_body_walk_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Generator\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def conditional_gen(flag: bool) -> Generator[Age, None, None]:\n",
        "    if flag:\n",
        "        yield 40\n",
    ));
    let def = module
        .body
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::FunctionDef(def) if def.name.id.as_str() == "conditional_gen" => Some(def),
            _ => None,
        })
        .expect("the fixture's own def");
    // `instances::generator_yields` must genuinely decline this body
    // (the conditional yield) so the call below exercises the
    // fallback this test pins, not the body-walked route.
    assert!(
        instances::generator_yields(def, &[], None, &kernel, 0).is_none(),
        "a conditional yield must decline the body-walked route (this test's own premise)"
    );
    let aliases = compile_aliases(&module);
    let imports = surface_imports(&module);
    let functions = Arc::new(function_table(&module));
    let classes = Arc::new(class_table(&module, &aliases, &imports, &kernel));
    let context = WalkContext {
        aliases: &aliases,
        imports: &imports,
        kernel: &kernel,
        functions,
        classes,
        datetime_imports: Arc::new(crate::expressions::datetime_imports(&module)),
        locale_never_set: crate::expressions::module_never_calls_setlocale(&module),
        module_bindings: HashMap::new(),
        module_callable_returns: Arc::new(HashMap::new()),
        strict_int_aliases: &HashSet::new(),
        typed_dicts: Arc::new(instances::typed_dict_table(&module, &aliases, &imports)),
        caller_arguments: Arc::new(HashMap::new()),
        entry_directory: None,
        evaluations_recorder: None,
        trace_collector: None,
    };
    let environment = Environment::new(HashSet::new());
    let delegate = ruff_python_parser::parse_expression("conditional_gen(True)")
        .expect("the delegate call parses")
        .into_syntax();
    let delegate = *delegate.body;
    let yields = delegated_generator_yields(&delegate, &context, &environment)
        .expect("the declared annotation fallback must still answer");
    let [value] = yields.as_slice() else {
        panic!("want exactly the one declared yield-type reading, got {}", yields.len());
    };
    assert_eq!(
        value.kind_tag,
        Some(PrimitiveKind::Integer),
        "Age's own numeric-ground set must tag the delegated value"
    );
}

/// THE VALUE SINK (A8.sink.arg's own shape): `d["x"] = 200` on a
/// `dict[str, Age]` parameter is judged AT THE WRITE, against the
/// declaration's own member refinement — not carried forward to fire
/// at whatever later sink reads `d`. The in-window write beside it
/// stays silent, and neither call site fires, since the receiver keeps
/// what its declaration says about its members either way.
#[test]
fn a_dict_member_write_past_the_declared_ceiling_fires_at_the_write() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def sink(d: dict[str, Age]) -> None:\n",
        "    pass\n",
        "def over(d: dict[str, Age]) -> None:\n",
        "    d[\"x\"] = 200\n",
        "    sink(d)\n",
        "def within(d: dict[str, Age]) -> None:\n",
        "    d[\"x\"] = 150\n",
        "    sink(d)\n",
    );
    let module = parsed(source);
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "want exactly one fire, for the 200 write: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert_eq!(
        fires[0].range.start(),
        offset_of(source, "200"),
        "the fire must land on the written value, never on the later sink call"
    );
}

/// THE PROVABLY-RAISING DELETE (A8.xfer.delete's
/// `read_widened_after_delete`): `del d["z"]` on a fully-known dict
/// with no `"z"` raises `KeyError`, so the delete never takes effect
/// and `d` keeps exactly what it held. The later `d["a"]` read must
/// still answer 200 and fire against `Age`, rather than going
/// undetermined on a forgotten receiver.
#[test]
fn a_provably_raising_delete_leaves_the_receiver_readable() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def rows() -> Age:\n",
        "    d: dict[str, int] = {\"a\": 200}\n",
        "    try:\n",
        "        del d[\"z\"]\n",
        "    except KeyError:\n",
        "        pass\n",
        "    value = d[\"a\"]\n",
        "    return value\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    let blockers: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        blockers.is_empty(),
        "the skipped delete must leave d fully readable: {:?}",
        blockers.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert_eq!(
        fires.len(),
        1,
        "d[\"a\"] still answers 200, outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
