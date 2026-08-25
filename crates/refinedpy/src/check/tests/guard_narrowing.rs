use super::*;

#[test]
fn a_seeded_parameter_returned_under_its_own_annotation_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(age: Age) -> Age:\n",
        "    return age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a parameter within its own declared set must stay silent on return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// PEP 484's "Stub Files" convention, restated for an INLINE `def`
/// (typing.rst's own `...` placeholder example for a declaration
/// with no runtime implementation): `crossed_from_fact`'s body is a
/// single `...` — declaration-only, PEP 484's own words — so a
/// caller reading its return must seed from the DECLARED `-> Age`
/// annotation, not from a body `interpret_body` never genuinely
/// interprets. Before `summaries::is_stub_body`, a bare `...`
/// statement fell through `interpret_body`'s ordinary `Stmt::Expr`
/// arm (evaluated and discarded, like `pass`), landing on a
/// fabricated `null_value()` return that fired RTS7001 against the
/// declared `-> Age` sink — this pins the fix: the crossing local
/// (`x: Age` flowing through the stub, unchanged, into another
/// `-> Age` return) stays silent.
#[test]
fn a_stub_bodied_call_seeds_its_declared_return_instead_of_none() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def crossed_from_fact(x: Age) -> Age: ...\n",
        "def fact_inside(x: Age) -> Age:\n",
        "    return crossed_from_fact(x)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a stub callee must answer its declared -> Age return, not a fabricated None: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The stub convention applies with a leading docstring too
/// (`first_non_docstring_statement`'s own leading-docstring skip) —
/// `def crossed_from_fact(x: Age) -> Age:\n    """docs"""\n    ...\n`
/// is a stub exactly as much as one with no docstring.
#[test]
fn a_docstring_then_ellipsis_stub_body_still_seeds_its_declared_return() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def crossed_from_fact(x: Age) -> Age:\n",
        "    \"\"\"a proved contract crossed as a fact.\"\"\"\n",
        "    ...\n",
        "def fact_inside(x: Age) -> Age:\n",
        "    return crossed_from_fact(x)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a docstring-then-ellipsis stub must still seed its declared -> Age return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A body that merely OPENS with a stray `...` expression, then goes
/// on to return something OUT of the declared set, is an ORDINARY
/// body — not a stub (`is_stub_body`'s own doc: the ellipsis must be
/// the body's own LAST statement) — and must still interpret
/// concretely and fire on its real out-of-set return, never read
/// through the stub's declared-return seed.
#[test]
fn a_leading_ellipsis_followed_by_a_real_statement_is_not_a_stub() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def not_a_stub() -> Age:\n",
        "    ...\n",
        "    return 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "a stray leading ellipsis must not mask the body's own out-of-set return: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A guard re-establishes a sort over an UNKNOWN value — the local's
/// own unknowable origin (a subscript into `json.loads`'s honest
/// return space, which this file's `collection_models::subscript_
/// read` does not read a `Kind::KindUnion` container through) must
/// not matter once `isinstance(value, int) and 0 <= value <= 150`
/// proves the whole window. Before `narrowing::narrow_isinstance_call`
/// treated a `Kind::Unknown` binding the same "no information yet"
/// way an entirely-unbound name is treated, `value`'s `Kind::Unknown`
/// binding passed through the isinstance test unchanged (the
/// function's own "existing binding" arm, which reads only
/// `Kind::Values`/`Kind::KindUnion`), so `return value` against `->
/// Age` fired RTS7002 ("not yet readable") rather than reading the
/// guard's own proof.
#[test]
fn an_isinstance_guard_narrows_an_unknown_valued_local_to_its_declared_return() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "import json\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def json_inside() -> Age:\n",
        "    record = {\"value\": 42}\n",
        "    text = json.dumps(record)\n",
        "    parsed = json.loads(text)\n",
        "    value = parsed[\"value\"]\n",
        "    if isinstance(value, int) and 0 <= value <= 150:\n",
        "        return value\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "the isinstance-and-comparison guard must prove value's window over its unknown origin: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- E2.operator: the AugAssign write-site check, restored beside the kernel-computed fold ---

/// E2.operator.py's own `compound_assign_outside_set`: `x: Age`
/// (a PARAMETER, never an AnnAssign local) then `x += 1` — the
/// write-site check must fire AT the `+=` line, with the marker's
/// own sentence as a verbatim prefix ("x may be 150; x += 1 may
/// write 151, outside Age's [0, 150]"), not at the later `return x`
/// with `judge`'s generic "not assignable" wording. Pins
/// `seed_parameters`'s own `aug_assign_refinements` insert (a
/// parameter's declared refinement was, before this fix, invisible
/// to `walk_name_aug_assign`'s lookup, which only ever saw an
/// AnnAssign target's entry) AND `judge_and_bind_aug_assign_write`'s
/// own write-specific message.
#[test]
fn a_compound_assign_past_a_declared_parameters_ceiling_fires_at_the_write_with_the_marker_sentence() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def compound_assign_outside_set(x: Age) -> Age:\n",
        "    x += 1\n",
        "    return x\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "exactly one fire, at the aug-assign write: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(
        fires[0].message.starts_with("x may be 150; x += 1 may write 151, outside Age's [0, 150]"),
        "want the marker's own sentence as a verbatim prefix, got: {:?}",
        fires[0].message
    );
}

/// E2.operator.py's own silent sibling `compound_assign_in_set`: a
/// bounded guard (`x < 149`) before the compound write keeps the
/// result inside `Age` — pins that the write-site check's OWN Fire
/// composition never fires on a genuinely in-set write; the SET
/// channel's ordinary comparison-against-literal narrowing (`x <
/// 149`, unaffected by this unit) still applies before
/// `judge_and_bind_aug_assign_write` runs.
#[test]
fn a_compound_assign_kept_in_set_by_a_prior_guard_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def compound_assign_in_set(x: Age) -> Age:\n",
        "    if x < 149:\n",
        "        x += 1\n",
        "        return x\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a guarded compound assign that stays inside Age must not fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- B1.keep.write: the relational ledger, restored beside the kernel-computed fold ---

/// B1.keep.write.py's own `increment_weakens_to_le`: under `i < n`
/// (both `Age`), `i += 1` gives `i ≤ n ≤ 150` — `i` stays inside
/// `Age`. Pins `relational_narrow_upper_bounds`'s own intersection of
/// the guard's relation with `i`'s current window, consulted BEFORE
/// the kernel-computed fold `judge_and_bind_aug_assign_write` still
/// runs unchanged.
#[test]
fn an_increment_under_a_strict_less_than_guard_weakens_to_le_and_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def increment_weakens_to_le(i: Age, n: Age) -> Age:\n",
        "    if i < n:\n",
        "        i += 1\n",
        "        return i\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "i < n weakening to i <= n after the increment must keep i inside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// B1.keep.write.py's own `reassign_forgets_relation`: the SAME `i <
/// n` guard, but `n` is reassigned to `0` inside the arm before `i`
/// is read back — the relation the guard proved is stale, and `i`
/// (declared `Wide`, `[0, 200]`) must NOT be judged against the
/// now-invalid `i < n` bound. Pins `relational_narrow_upper_bounds`'s
/// own `locally_bound_names(body)` gate: a body that reassigns the
/// relation's own right-hand name never gets the narrowing at all.
#[test]
fn a_relation_is_forgotten_once_its_own_right_hand_name_is_reassigned() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Wide = Annotated[int, Field(ge=0, le=200)]\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def reassign_forgets_relation(i: Wide, n: Age) -> Age:\n",
        "    if i < n:\n",
        "        n = 0\n",
        "        return i\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the stale i < n relation must not silence i's own out-of-Age window: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// B1.est.guard.py's own `conjunction_inside`: a CHAINED comparison
/// `lo <= x <= hi` states two facts at once — `relational_ceiling_
/// facts`'s own chained-comparison pairing must read the SECOND pair
/// (`x <= hi`) and narrow `x`'s ceiling to `hi`'s own [0, 150]
/// ceiling, keeping `x` inside `Age` on the return.
#[test]
fn a_chained_comparison_narrows_the_middle_names_ceiling_from_its_own_upper_pair() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "type Wide = Annotated[int, Field(ge=0, le=200)]\n",
        "def conjunction_inside(lo: Age, x: Wide, hi: Age) -> Age:\n",
        "    if lo <= x <= hi:\n",
        "        return x\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "lo <= x <= hi must narrow x's ceiling to hi's own Age window: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// B1.keep.trans.py's own `transitivity_holds`: `i < n and n <= m`
/// combines two two-Name facts from ONE `and`-conjunction — pins the
/// TWO-PASS transitivity step: the first pass narrows `i` against
/// `n`'s own bare `Wide` [0, 200] ceiling and separately narrows `n`
/// against `m`'s `Age` [0, 150] ceiling; the second pass then
/// re-narrows `i` against `n`'s now-tightened ceiling, so `i`'s own
/// final bound reflects `m` transitively and stays inside `Age`.
#[test]
fn an_and_conjunction_of_two_facts_narrows_transitively_in_one_step() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "type Wide = Annotated[int, Field(ge=0, le=200)]\n",
        "def transitivity_holds(i: Wide, n: Wide, m: Age) -> Age:\n",
        "    if i < n and n <= m:\n",
        "        return i\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "i < n and n <= m must transitively bound i by m's own Age ceiling: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// B1.use.project.py's own `projection_stays_inside`: the two-Name
/// fact `i < n` sits as the THIRD conjunct of a three-way `and`
/// (`i >= 0 and 0 <= n <= 9 and i < n`) — pins that `relational_
/// ceiling_facts` finds the fact wherever it sits among conjuncts of
/// mixed shapes (a single-name literal bound, a chained single-name
/// bound, then the two-Name fact), narrowing `i`'s ceiling to `n`'s
/// own (already SET-channel-narrowed) ceiling of 9.
#[test]
fn a_two_name_fact_is_found_among_mixed_conjuncts_in_an_and_chain() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def projection_stays_inside(i: int, n: int) -> Age:\n",
        "    if i >= 0 and 0 <= n <= 9 and i < n:\n",
        "        return i\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "i < n with n's own ceiling narrowed to 9 must keep i inside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// B1.use.sink.py's own `between_bounds_admitted`: `lo <= x <= hi`
/// (`lo`, `hi` both `Age`) narrows `x`'s ceiling to `hi`'s own
/// [0, 150] window BEFORE the arm's own `a: Age = x` sink judges it —
/// pins that the chained-comparison ceiling narrowing applies ahead
/// of an ordinary declared-target assignment, not only a `return`.
#[test]
fn a_chained_comparisons_ceiling_narrowing_admits_a_later_declared_sink() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def between_bounds_admitted(lo: Age, x: int, hi: Age) -> Age:\n",
        "    if lo <= x <= hi:\n",
        "        a: Age = x\n",
        "        return a\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "lo <= x <= hi must narrow x's ceiling so a: Age = x is admitted: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}
