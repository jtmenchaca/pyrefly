use super::*;

// --- declared_return_seed / is_stub_body: the stub-body decline path ---

/// A body whose only statement is a bare `...` is a stub
/// (PEP 484's "Stub Files" convention, restated for an inline `def`).
#[test]
fn is_stub_body_recognizes_a_bare_ellipsis_body() {
    let def = parsed_def("def crossed_from_fact(x) -> None: ...\n");
    assert!(is_stub_body(&def.body));
}

/// A leading docstring before the `...` is still a stub —
/// `first_non_docstring_statement`'s own skip applies first.
#[test]
fn is_stub_body_recognizes_a_docstring_then_ellipsis_body() {
    let def = parsed_def("def crossed_from_fact(x) -> None:\n    \"\"\"docs\"\"\"\n    ...\n");
    assert!(is_stub_body(&def.body));
}

/// A body that opens with `...` but goes on to a REAL statement is
/// NOT a stub — the ellipsis must be the body's own LAST statement.
#[test]
fn is_stub_body_refuses_an_ellipsis_followed_by_a_real_statement() {
    let def = parsed_def("def not_a_stub() -> None:\n    ...\n    return None\n");
    assert!(!is_stub_body(&def.body));
}

/// An ordinary body (no ellipsis at all) is not a stub.
#[test]
fn is_stub_body_refuses_an_ordinary_body() {
    let def = parsed_def("def f(x):\n    return x\n");
    assert!(!is_stub_body(&def.body));
}

/// `declared_return_seed` reads a same-module callee's `-> Age`
/// stub return through the alias table, the same scalar seed
/// `check.rs::seed_parameters` builds for an `Age`-typed parameter:
/// `Age`'s own set (`[0, 150]`, Integer-tagged), grade TrustSpec.
#[test]
fn declared_return_seed_reads_an_alias_typed_stub_return() {
    let environment = environment_with_module_aliases(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def crossed_from_fact(x: Age) -> Age: ...\n",
    ));
    let def = parsed_def("def crossed_from_fact(x: Age) -> Age: ...\n");
    let seeded = declared_return_seed(&def, &environment).expect("Age resolves through the alias table");
    assert_eq!(seeded.kind_tag, Some(PrimitiveKind::Integer));
    assert_eq!(seeded.set, refined_sets::refinement_forms::make_refined_set(vec![
        refined_sets::refinement_forms::at_least(0.0),
        refined_sets::refinement_forms::at_most(150.0),
        refined_sets::refinement_forms::integer(),
    ]));
}

/// `declared_return_seed` answers `None` when the environment carries
/// no alias table at all — the caller's own `.or_else(|| return_sort_
/// fallback(def))` is what a bare-sort return still falls back to.
#[test]
fn declared_return_seed_declines_with_no_alias_table() {
    let environment = Environment::new(std::collections::HashSet::new());
    let def = parsed_def("def crossed_from_fact(x) -> Age: ...\n");
    assert!(declared_return_seed(&def, &environment).is_none());
}

/// A caller's own contract crosses through a stub callee end to end:
/// `fact_inside` calls `crossed_from_fact`, whose body is a bare
/// `...` — `call_result_with_enclosing`'s own `is_stub_body` check
/// reads the declared `-> Age` return rather than interpreting the
/// stub body (which would otherwise fall through to a fabricated
/// `null_value()`). Threads the SAME alias table `check.rs`'s own
/// walk threads, through `enclosing`, exactly the way a real call
/// site's environment carries it (`call_result_with_enclosing`'s own
/// `enclosing.declared_aliases()`-reachable seam is `environment`
/// itself, built fresh per call — this test pins that the def's OWN
/// `Environment`, not `enclosing`'s, is what `declared_return_seed`
/// reads, matching `walk_body_with_self_binding`'s per-body seeding).
#[test]
fn a_stub_bodied_call_answers_its_declared_return_not_none() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parse_module(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def crossed_from_fact(x: Age) -> Age: ...\n",
    ))
    .expect("fixture source parses")
    .into_syntax();
    let def = module
        .body
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::FunctionDef(def) if def.name.id.as_str() == "crossed_from_fact" => Some(def.clone()),
            _ => None,
        })
        .expect("the fixture's own def");
    let aliases = compile_aliases(&module);
    let imports = surface_imports(&module);
    let mut caller_environment = Environment::new(std::collections::HashSet::new());
    caller_environment.set_declared_aliases(Arc::new(aliases), Arc::new(imports));
    let result = call_result_with_enclosing(&def, &[known_int(40.0)], None, &kernel, 0, Some(&caller_environment))
        .expect("a stub callee must answer its declared return, not decline outright");
    assert_ne!(result.kind, Kind::Null, "a stub body must never fabricate a None return");
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// A def with a keyword-only parameter the CALLER never covers (no
/// slot in `arguments` at all — the shape `bind_parameters` sees
/// when a caller genuinely omits it, e.g. an optional kwonly with a
/// default this file does not yet read) still reaches the coarse
/// `-> int` fallback, since `bind_parameters`'s own arity check
/// finds no slot for it.
#[test]
fn a_keyword_only_def_with_no_covering_slot_answers_the_whole_number_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def only_keyword(*, age) -> int:\n    return age\n");
    let result = call_result(&def, &[], None, &kernel, 0)
        .expect("the -> int annotation answers the whole-number set when no slot covers the kwonly param");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// e-class-and-function.py's own `keyword_only_call` regression: a
/// keyword-only parameter the CALLER covers by keyword is no longer
/// a hard decline — `expressions.rs`'s `positional_arguments_for_
/// def` maps the caller's `age=200` onto this def's own trailing
/// kwonly slot (that function's own doc), and `call_result` (called
/// here exactly the way that mapping would hand it off) answers the
/// body's own exact value, never the coarse fallback.
#[test]
fn a_keyword_only_def_with_a_covering_slot_answers_the_bodys_exact_value() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def only_keyword(*, age):\n    return age\n");
    let result = call_result(&def, &[known_int(200.0)], None, &kernel, 0)
        .expect("a covering slot binds the kwonly parameter and interprets the body");
    assert_eq!(result, known_int(200.0));
}

/// A plain parameter THEN a keyword-only one — the two families
/// bind from adjacent slots in the SAME `arguments` vector
/// (`bind_parameters`'s own doc: kwonly slots sit right after the
/// plain parameters' own).
#[test]
fn a_plain_parameter_and_a_trailing_keyword_only_parameter_bind_from_adjacent_slots() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def mixed(first, *, second):\n    return first + second\n");
    let result = call_result(&def, &[known_int(1.0), known_int(2.0)], None, &kernel, 0)
        .expect("first binds positionally, second binds from the trailing kwonly slot");
    assert_eq!(result, known_int(3.0));
}

/// e-class-and-function.py's own `kwargs_parameter` regression: a
/// `**kwargs` parameter binds from the VERY LAST slot of
/// `arguments` — the collected dict `expressions.rs`'s
/// `positional_arguments_with_kwargs_dict` would build and append
/// there. `fields["age"]` reads the collected dict back through the
/// ordinary subscript-read path once bound.
#[test]
fn a_kwargs_parameter_binds_the_final_slot_as_a_dict() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def gather_kwargs(**fields):\n    return fields[\"age\"]\n");
    let collected = refined_domain::known_constructors::known_object(
        vec![refined_domain::abstract_value::ObjectKey {
            name: "age".to_owned(),
            numeric: false,
            value: known_int(200.0),
        }],
        None,
        true,
        TrustSpec,
        false,
    );
    let result = call_result(&def, &[collected], None, &kernel, 0)
        .expect("the final slot binds to fields, and fields[\"age\"] reads through");
    assert_eq!(result, known_int(200.0));
}

/// A body that reads CONCRETELY for one or more statements before
/// declining is NOT opaque — the coarse `-> int` fallback must not
/// fire. e-class-and-function.py's own `grow_into_bucket` shape:
/// `bucket.append(age)` is an ordinary expression statement
/// `interpret_body` reads fine (its result is simply discarded, per
/// that arm's own doc); the decline happens only later, at
/// `return bucket[0]`, because `bucket` itself is `unknown()` (its
/// caller passed no argument, so `bind_parameters` evaluated the
/// PARAMETER DEFAULT — a bare module-level name — against a fresh,
/// name-less environment, per that function's own doc). Firing the
/// coarse whole-number-set fallback here would overstate what this
/// interpreter actually determined; the honest answer is `None`
/// (`unknown()` at the call site), matching every other genuinely
/// unread value this file declines rather than guesses at.
#[test]
fn a_body_that_reads_one_statement_before_declining_does_not_reach_the_coarse_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def grow_into_bucket(age, bucket=_DEFAULT_BUCKET) -> int:\n",
        "    bucket.append(age)\n",
        "    return bucket[0]\n",
    ));
    let result = call_result(&def, &[known_int(41.0)], None, &kernel, 0);
    assert!(
        result.is_none(),
        "a mid-body decline after a concretely-read statement must stay None, never the coarse -> int set: {result:?}"
    );
}

/// The CONTRASTING case, pinned alongside the one above so the two
/// never drift apart: a body that declines on its very FIRST
/// statement (never producing any readable effect) still reaches the
/// coarse fallback — `unread_number`'s own shape
/// (a-statements.py:34), `raise NotImplementedError` as the sole
/// statement.
#[test]
fn a_body_that_declines_on_its_first_statement_still_reaches_the_coarse_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def unread_number() -> int:\n    raise NotImplementedError\n");
    let result = call_result(&def, &[], None, &kernel, 0)
        .expect("a first-statement decline is genuinely opaque, so the -> int fallback must still fire");
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// THE DOCSTRING GATE BUG's own regression: `unread_number`'s REAL
/// body (a-statements.py:34-38) is a docstring FOLLOWED BY `raise
/// NotImplementedError` — a docstring-only probe of "the first
/// statement" would wrongly succeed (`Stmt::Expr` on a string
/// literal always interprets fine) and mask that the body's first
/// REAL statement is the one that declines, sending this def down
/// the `None` path instead of the coarse `-> int` fallback. This
/// pins the fix: `first_non_docstring_statement` skips the leading
/// docstring, so the probe reaches `raise NotImplementedError` and
/// correctly declines there.
#[test]
fn a_docstring_before_a_first_statement_decline_still_reaches_the_coarse_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def unread_number() -> int:\n",
        "    \"\"\"an opaque int source\"\"\"\n",
        "    raise NotImplementedError\n",
    ));
    let result = call_result(&def, &[], None, &kernel, 0).expect(
        "a docstring is not a readable effect — the def is still opaque from its first REAL statement, so the -> int fallback must fire",
    );
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// The CONTRASTING case the gate exists for stays out of the
/// fallback even WITH a leading docstring: e-class-and-function.py's
/// own `grow_into_bucket` shape, now with a docstring prepended — a
/// concretely-read statement (`bucket.append(age)`) after the
/// docstring still marks the body as genuinely readable, not opaque,
/// so the answer stays `None` rather than the coarse fallback.
#[test]
fn a_docstring_before_a_concretely_read_statement_does_not_reach_the_coarse_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def grow_into_bucket(age, bucket=_DEFAULT_BUCKET) -> int:\n",
        "    \"\"\"mutable default\"\"\"\n",
        "    bucket.append(age)\n",
        "    return bucket[0]\n",
    ));
    let result = call_result(&def, &[known_int(41.0)], None, &kernel, 0);
    assert!(
        result.is_none(),
        "a docstring plus a mid-body decline after a concretely-read statement must stay None: {result:?}"
    );
}

/// A def whose body is NOTHING BUT a docstring (no statement after
/// it at all) still reaches the coarse fallback — the same "first
/// REAL statement" absence `first_non_docstring_statement`'s own
/// `None` row declines through.
#[test]
fn a_body_that_is_only_a_docstring_reaches_the_coarse_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def only_documented() -> int:\n    \"\"\"nothing else here\"\"\"\n");
    let result = call_result(&def, &[], None, &kernel, 0);
    // a docstring-only body falls off the end (Kind::Null, the
    // Null-vs-scalar-ground law's own business) — this pins that the
    // docstring-only shape does not crash or mis-answer, without
    // asserting which existing law owns the resulting verdict
    assert!(result.is_some(), "a docstring-only body still answers something (falls through to None): {result:?}");
}
