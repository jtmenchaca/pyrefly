use super::*;

/// f-type-nodes.py's own `optional_annotation` shape: `present:
/// Optional[Age] = 40` then `if present is None:` — `present`'s
/// concrete value (40) makes the `is None` test provably false, but
/// `present`'s DECLARED shape admits `None` (`Optional[Age]`), so this
/// is the ordinary Optional-peeling idiom, never dead code. The
/// DEAD-BRANCH LAW must not fire RTS7001 here, and the walk must still
/// reach the later `good: Age = present` read (which stays silent —
/// 40 is in Age's [0, 120] window).
#[test]
fn an_is_none_peel_on_an_admits_none_declared_name_never_fires_the_dead_branch_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Optional\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def optional_annotation() -> Age:\n",
        "    present: Optional[Age] = 40\n",
        "    if present is None:\n",
        "        return 0\n",
        "    good: Age = present\n",
        "    return good\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an Optional-peel test must never fire the dead-branch law, and the in-set \
         read after it must stay silent too: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The mirror: `Age | None` (the pipe-union spelling of `Optional`)
/// peeled the same way — `is_admits_none_peel_test` must recognize
/// both annotation spellings identically, since `typereading::
/// declared_refinement` reads them to the same `admits_none: true`
/// shape.
#[test]
fn an_is_none_peel_on_a_pipe_none_declared_name_never_fires_the_dead_branch_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def pipe_none_annotation() -> Age:\n",
        "    present: Age | None = 40\n",
        "    if present is None:\n",
        "        return 0\n",
        "    good: Age = present\n",
        "    return good\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an `Age | None` peel test must never fire the dead-branch law: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// f-type-nodes.py's own `optional_annotation`/`pipe_none_annotation`
/// SECOND row (`over: Optional[int] = 200`, `if over is None:`): a
/// bare base-sort wrapped in `Optional`/`| None`, with NO alias
/// involved at all — `optional_base_sort_annotation`'s own row,
/// distinct from the `Optional[Age]`/`Age | None` alias shape the two
/// tests above cover. The dead-branch law must not fire on the peel
/// test, and the later `return over` must still fire on 200 once
/// unwrapped — the peel exception silences ONLY the `is None` dead-
/// branch fire, never the real out-of-set return.
#[test]
fn an_is_none_peel_on_a_bare_optional_int_declared_name_never_fires_the_dead_branch_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Optional\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def optional_annotation() -> Age:\n",
        "    over: Optional[int] = 200\n",
        "    if over is None:\n",
        "        return 0\n",
        "    return over\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let dead_branch_fires: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
        .collect();
    assert!(
        dead_branch_fires.is_empty(),
        "a bare Optional[int] peel test must never fire the dead-branch law: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "200 must still fire at the return once unwrapped from Optional: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// The pipe-union mirror: `over: int | None = 200`.
#[test]
fn an_is_none_peel_on_a_bare_pipe_none_int_declared_name_never_fires_the_dead_branch_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def pipe_none_annotation() -> Age:\n",
        "    over: int | None = 200\n",
        "    if over is None:\n",
        "        return 0\n",
        "    return over\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let dead_branch_fires: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
        .collect();
    assert!(
        dead_branch_fires.is_empty(),
        "a bare `int | None` peel test must never fire the dead-branch law: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "200 must still fire at the return once unwrapped from the union: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'200'"), "{}", fires[0].message);
}

/// The exception's own boundary: a-statements.py's own
/// `none_test_on_helper_that_never_answers_none` shape — `held` is
/// bound by a plain `Assign` from a call result (never an
/// `AnnAssign`), so it carries no entry in `aug_assign_refinements` at
/// all. `is_admits_none_peel_test` must find nothing and the
/// dead-branch law must still fire here, exactly as before the
/// exception existed — the exception is scoped to a DECLARED
/// `admits_none` name, never to every `is None` test whose value
/// happens to be provably non-null.
#[test]
fn an_is_none_test_on_a_plain_assign_target_still_fires_the_dead_branch_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def helper_never_answers_none() -> dict[str, int]:\n",
        "    if True:\n",
        "        return {\"age\": 40}\n",
        "    return {\"age\": 10}\n",
        "def none_test_on_helper_that_never_answers_none() -> Age:\n",
        "    held = helper_never_answers_none()\n",
        "    if held is None:\n",
        "        return 0\n",
        "    return held[\"age\"]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let dead_branch_fires: Vec<&Finding> = findings
        .iter()
        .filter(|f| f.code == "RTS7001" && f.message.contains("provably false"))
        .collect();
    assert_eq!(
        dead_branch_fires.len(),
        1,
        "a plain-Assign target carries no aug_assign_refinements entry, so the \
         exception must not suppress this row's own dead-branch fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// `sample`'s own read INSIDE the `if sample is not None:` arm — the
/// position `refined_set_at_position` answers for the `sample` name
/// at the `return sample` site — must read the ANNOTATED, non-None
/// set (`>= -2 && <= 2`), never the wrapper: `narrow_is_none`'s own
/// `Kind::PossiblyUndefined` unwrap arm is what this pins.
#[test]
fn an_is_not_none_guarded_parameter_narrows_to_its_annotated_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated, Optional\n",
        "from pydantic import Field\n",
        "Sample = Optional[Annotated[float, Field(ge=-2.0, le=2.0)]]\n",
        "\n",
        "def f(sample: Sample) -> None:\n",
        "    if sample is not None:\n",
        "        return sample\n",
        "    return None\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "sample\n    return None");
    let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
        .unwrap_or_else(|| panic!("expected the narrowed `sample` read to answer a declared set"));
    assert_eq!(format_for_diagnostics(&set), ">= -2 && <= 2");
}

/// The ternary twin of the above: `sample if sample is not None else
/// 0.0` forks and narrows the SAME way an `if`/`else` STATEMENT does
/// (`expressions.rs::evaluate_ternary`'s own routing through
/// `narrowing::assume`) — the joined value is wider than a `Level`
/// return (`>= 0 && <= 1`), so it fires RTS7001, never sits
/// undetermined behind RTS7002.
#[test]
fn an_is_not_none_guarded_ternary_joins_to_a_wider_set_and_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Optional\n",
        "from pydantic import Field\n",
        "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "Sample = Optional[Annotated[float, Field(ge=-2.0, le=2.0)]]\n",
        "def f(sample: Sample) -> Level:\n",
        "    return sample if sample is not None else 0.0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let undetermined: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7002").collect();
    assert!(
        undetermined.is_empty(),
        "the ternary's joined value must be readable, never RTS7002: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the joined set (>= -2 && <= 2, or 0.0) is wider than Level (>= 0 && <= 1) and must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
