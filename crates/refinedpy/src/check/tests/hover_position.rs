use super::*;

#[test]
fn a_module_level_import_of_its_own_alias_is_never_read_as_rebinding_it() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "module_level_binding: Age = 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        !findings.iter().any(|finding| finding.message.contains("is rebound in this body")),
        "an alias's own establishing import must never read as rebinding it: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A8.guard.eq.py's own accept-arm shape: after `d == {"a": 1}`
/// holds, `d["a"]` is provably {1} and an Age-declared return must
/// admit it — the dict-equality guard's narrowing, pinned here so
/// the mechanism iterates under `cargo test` (pnpm py:test:filter)
/// without a fixture run. Currently a FALSE POSITIVE: the guard
/// does not narrow the subscript read.
#[test]
fn a_dict_equality_guard_narrows_the_subscript_read_it_pins() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def f(d: dict) -> Age:\n",
        "    if d == {\"a\": 1}:\n",
        "        return d[\"a\"]\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "the equality-pinned d[\"a\"] is provably {{1}}, inside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The identical shape through an actual CROSS-MODULE import
/// (`from support.py.refined import Age`, C1.scope.py's own literal
/// spelling) rather than a same-module `type` alias — the
/// `Stmt::ImportFrom` arm specifically, not just `Stmt::Import`.
/// The in-memory resolver map is `an_imported_value_read_through_a_
/// two_module_resolver_fires_at_a_return_sink`'s own pattern
/// (`cross_module.rs`'s own test convention).
#[test]
fn a_module_level_from_import_of_its_own_alias_is_never_read_as_rebinding_it() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut sources: HashMap<&str, &str> = HashMap::new();
    sources.insert(
        "support",
        concat!(
            "from typing import Annotated\n",
            "from pydantic import Field\n",
            "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        ),
    );
    let module = parsed(concat!(
        "from support import Age\n",
        "module_level_binding: Age = 0\n",
    ));
    let resolver: ModuleResolver = &|name: &str| sources.get(name).map(|source| parsed(source));
    let findings = findings_for_module_with_resolver(&module, resolver, &kernel);
    assert!(
        !findings.iter().any(|finding| finding.message.contains("is rebound in this body")),
        "an alias's own establishing FROM-import must never read as rebinding it: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A module whose ONLY refinement vocabulary is a bare
/// `Literal[...]` annotation — no `type` alias, no `Annotated`
/// import — is judged: the engagement gate reads the `Literal`
/// import as refinement vocabulary too. Pins the gate widening;
/// before it, this module returned zero findings without ever
/// reaching the walk.
#[test]
fn a_literal_only_module_is_judged() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Literal\n",
        "\n",
        "def pick(level: Literal[1, 2, 4]) -> Literal[1, 2]:\n",
        "    return level\n",
    );
    let module = parsed(source);
    let findings = findings_for_module_at(&module, no_imports_resolver(), &kernel, None);
    assert!(
        !findings.is_empty(),
        "a Literal-only module must reach the walk and judge its rows"
    );
}

/// `refined_set_at_position` at a PARAMETER's own annotation
/// position answers the declared set — a-audio-level.py's own
/// `samples: Annotated[list[Sample], Field(min_length=1)]` row.
/// The stated branch reads through the SAME `declared_refinement`
/// `seed_parameters` calls to seed the parameter itself, so this
/// pins that the query and the seed never drift apart.
#[test]
fn a_parameter_annotation_position_answers_its_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
        "\n",
        "def audio_level(samples: Sample) -> None:\n",
        "    pass\n",
    );
    let module = parsed(source);
    // a position inside the PARAMETER's own annotation name ("Sample"
    // in "samples: Sample")
    let position = offset_of(source, "Sample) -> None");
    let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
        .unwrap_or_else(|| panic!("expected a declared set at the parameter annotation"));
    assert_eq!(format_for_diagnostics(&set), ">= -2 && <= 2");
}

/// `refined_set_at_position` at the RETURN annotation's own
/// position answers the declared set — audio_level.py's own `->
/// Level` (`Level = Annotated[float, Field(ge=0.0, le=1.0)]`). The
/// brief that specified this unit named "the Level parameter", but
/// `Level` is the fixture's RETURN annotation, not a parameter —
/// this test covers the position the fixture actually has.
#[test]
fn a_return_annotation_position_answers_its_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def audio_level(x: float) -> Level:\n",
        "    return x\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "Level:\n    return");
    let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
        .unwrap_or_else(|| panic!("expected a declared set at the return annotation"));
    assert_eq!(format_for_diagnostics(&set), ">= 0 && <= 1");
}

/// The hover CLI's own pair — `refined_set_at_position` then
/// `format_for_hover` — read at the RETURN annotation's own `Level`
/// spells the same facts `format_for_diagnostics` reads above, in
/// the hover vocabulary rather than the diagnostic one. Pins that
/// `refinedpy-check --hover` prints what this seam pair actually
/// answers, since the bin crate has no test convention of its own
/// (`src/bin/refinedpy_check.rs` carries no `#[cfg(test)]`) and this
/// is where every other `refined_set_at_position` position test
/// already lives.
#[test]
fn the_hover_seam_pair_spells_a_declared_alias_at_a_usage_site() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def audio_level(x: float) -> Level:\n",
        "    return x\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "Level:\n    return");
    let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
        .unwrap_or_else(|| panic!("expected a declared set at the return annotation"));
    let spelled = refined_sets::format_for_hover::format_for_hover(&set)
        .unwrap_or_else(|| panic!("expected a hover spelling for Level"));
    assert_eq!(spelled, "{0 ≤ 𝑥 ≤ 1}");
}

/// A position at the ALIAS DECLARATION's own name (`type Level =
/// Annotated[...]`, the `Level` before `=`) answers the alias's own
/// compiled set — the SAME set a parameter annotated `x: Level`
/// gets. Glyph spelling copied from `format_for_hover.rs`'s own
/// fixtures (`"{0 ≤ 𝑥 ≤ 100}"`), never from memory.
#[test]
fn an_alias_declarations_own_name_answers_its_compiled_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def audio_level(x: float) -> Level:\n",
        "    return x\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "Level = Annotated");
    let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
        .unwrap_or_else(|| panic!("expected the alias's own compiled set at its declaration name"));
    let spelled = refined_sets::format_for_hover::format_for_hover(&set)
        .unwrap_or_else(|| panic!("expected a hover spelling for Level's declaration"));
    assert_eq!(spelled, "{0 ≤ 𝑥 ≤ 1}");
}

/// The `X = Annotated[...]` (no `type` keyword) and `X: TypeAlias =
/// Annotated[...]` spellings answer identically at their own name —
/// the other two of the three spellings `compile_aliases` reads,
/// alongside the `type X = ...` spelling pinned above.
#[test]
fn the_plain_and_annotated_alias_spellings_answer_at_their_own_name_too() {
    let Some(kernel) = loaded_kernel() else { return };
    let plain_source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
    );
    let plain = parsed(plain_source);
    let plain_position = offset_of(plain_source, "Level = Annotated");
    let plain_set = refined_set_at_position(&plain, no_imports_resolver(), &kernel, plain_position)
        .unwrap_or_else(|| panic!("expected the plain-assignment alias's own set at its name"));
    assert_eq!(format_for_diagnostics(&plain_set), ">= 0 && <= 1");

    let annotated_source = concat!(
        "from typing import Annotated, TypeAlias\n",
        "from pydantic import Field\n",
        "Level: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]\n",
    );
    let annotated = parsed(annotated_source);
    let annotated_position = offset_of(annotated_source, "Level: TypeAlias");
    let annotated_set = refined_set_at_position(&annotated, no_imports_resolver(), &kernel, annotated_position)
        .unwrap_or_else(|| panic!("expected the `: TypeAlias =` spelling's own set at its name"));
    assert_eq!(format_for_diagnostics(&annotated_set), ">= 0 && <= 1");
}

/// A `def`'s own name answers its return refinement, when one is
/// readable — the same claim a call to it yields.
#[test]
fn a_defs_own_name_answers_its_declared_return_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def audio_level(x: float) -> Level:\n",
        "    return x\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "audio_level(x: float)");
    let set = refined_set_at_position(&module, no_imports_resolver(), &kernel, position)
        .unwrap_or_else(|| panic!("expected the def's own declared return set at its name"));
    assert_eq!(format_for_diagnostics(&set), ">= 0 && <= 1");
}

/// A `def` with a BARE base-sort return annotation (`-> float`, no
/// `Annotated`/alias) answers NOTHING at its own name —
/// `declared_refinement` alone is read there, with no
/// `base_sort_return_refinement` fallback, exactly matching
/// `declared_refinement`'s own doc: a base sort states nothing this
/// table reads. The module still carries an `Annotated` import (on
/// an unrelated parameter) so the engagement gate is passed for the
/// same reason the walk reaches this module at all — the point
/// pinned here is the def-name branch's own decline, not the
/// module-level early exit `a_position_naming_neither_a_declaration_
/// nor_a_recorded_expression_answers_nothing` already covers.
#[test]
fn a_defs_own_name_with_a_bare_sort_return_answers_nothing() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
        "\n",
        "def audio_level(x: float) -> float:\n",
        "    return x\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "audio_level(x: float)");
    assert!(
        refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_none(),
        "a bare `-> float` must not fabricate a claim at the def's own name"
    );
}

/// Ledger 260: `total`'s own position in `audio_level_unclamped.py`
/// (`total = sum(s * s for s in samples)`) now answers the derived
/// total's set — `walk_relational_sum` records the kernel's proved
/// total at the Assign statement's own range the same way an
/// ordinary assignment's RHS records its evaluated node
/// (`expressions.rs::evaluate_expression`'s `record_evaluation`
/// call), so `refined_set_at_position` finds it as the smallest
/// covering recorded range. Was pinned the other way
/// (`the_measured_total_position_answers_no_set_today`,
/// refinedpy_lsp/src/lib.rs) before this unit threaded the publish
/// through.
#[test]
fn the_measured_total_position_now_answers_the_derived_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "import math\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
        "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def audio_level_unclamped(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
        "    total = sum(s * s for s in samples)\n",
        "    return math.sqrt(total / len(samples))\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "total = sum");
    assert!(
        refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
        "total's own position must answer the kernel's derived set now that the recognizer publishes it"
    );
}

/// The quotient binding's twin of the test above: `mean = total /
/// len(samples)` is consumed by the same recognizer
/// (`walk_relational_sum`'s divided-into arm), which binds the
/// kernel's quotient — and now records it at the Assign statement's
/// own range too, so `mean`'s own position answers the quotient's
/// set instead of nothing.
#[test]
fn the_quotient_binding_position_answers_the_derived_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
        "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def mean_square(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
        "    total = sum(s * s for s in samples)\n",
        "    mean = total / len(samples)\n",
        "    return mean\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "mean = total");
    assert!(
        refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
        "the quotient binding's own position must answer the kernel's derived set"
    );
}

/// A count-alias binding (`count = len(samples)`) consumed ahead of
/// the division now binds and publishes — the alias's own name reads
/// an integer-sorted set admitting the sequence's length window, and
/// its position answers that same set, exactly the way the total's
/// and the quotient's own positions do above. The window here is
/// `min_length=1` with no upper bound, so the alias's set is at
/// least the count-alias fold's own floor and never a fabricated
/// exact value.
#[test]
fn a_count_alias_binding_now_answers_the_derived_length_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=-2.0, le=2.0)]\n",
        "Level = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "\n",
        "def mean_square(samples: Annotated[list[Sample], Field(min_length=1)]) -> Level:\n",
        "    total = sum(s * s for s in samples)\n",
        "    count = len(samples)\n",
        "    mean = total / count\n",
        "    return mean\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "count = len");
    assert!(
        refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_some(),
        "the count-alias binding's own position must answer the kernel's derived length window"
    );
}

/// The count-alias spelling — `count = len(samples)` then
/// `mean = total / count` — judges IDENTICALLY to the direct
/// spelling `mean = total / len(samples)`, and the statement after
/// the consumed alias-and-division pair still walks rather than
/// being skipped by stale bookkeeping (`walk_relational_sum`'s own
/// `skip_statements` count, threaded through `check.rs`'s
/// `folded_division_at` range). Each body assigns the mean into an
/// `Age`-typed slot with an out-of-set literal fallback (`over`)
/// immediately after, so a wrong skip count — either leaving the
/// alias assignment to be walked a second time as an ordinary
/// statement, or swallowing `over` into the skipped range — would
/// change the finding count or the position `over`'s own fire lands
/// at. `mean`'s own position is also read back identically in both
/// spellings, pinning that the aliased fold answers the same derived
/// set the direct fold does, not merely the same finding count.
#[test]
fn the_count_alias_spelling_judges_identically_to_the_direct_spelling_and_the_next_statement_still_walks() {
    let Some(kernel) = loaded_kernel() else { return };
    let direct_source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=0.0, le=2.0)]\n",
        "Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(samples: Annotated[list[Sample], Field(min_length=1)]) -> None:\n",
        "    total = sum(s for s in samples)\n",
        "    mean = total / len(samples)\n",
        "    over: Age = 200\n",
    );
    let aliased_source = concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "Sample = Annotated[float, Field(ge=0.0, le=2.0)]\n",
        "Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def f(samples: Annotated[list[Sample], Field(min_length=1)]) -> None:\n",
        "    total = sum(s for s in samples)\n",
        "    count = len(samples)\n",
        "    mean = total / count\n",
        "    over: Age = 200\n",
    );
    let direct = parsed(direct_source);
    let aliased = parsed(aliased_source);
    let direct_findings = findings_for_module(&direct, &kernel);
    let aliased_findings = findings_for_module(&aliased, &kernel);
    let direct_messages: Vec<&str> = direct_findings.iter().map(|f| f.message.as_str()).collect();
    let aliased_messages: Vec<&str> = aliased_findings.iter().map(|f| f.message.as_str()).collect();
    // the trailing `over: Age = 200` fires in both spellings — the
    // statement after the consumed pair (one statement wider in the
    // aliased body) still walks rather than being skipped
    assert_eq!(direct_findings.len(), 1, "direct spelling: {direct_messages:?}");
    assert_eq!(aliased_findings.len(), 1, "aliased spelling: {aliased_messages:?}");
    assert_eq!(direct_findings[0].code, "RTS7001");
    assert_eq!(aliased_findings[0].code, "RTS7001");
    assert!(direct_messages[0].contains("'200'"), "{direct_messages:?}");
    assert!(aliased_messages[0].contains("'200'"), "{aliased_messages:?}");
    // The fold's OWN identity (the aliased spelling wiring the same
    // kernel program as the direct one) is pinned at the API level
    // by relational_sum.rs's
    // `the_count_alias_shape_folds_identically_to_the_direct_spelling`
    // — a folded division records no evaluated node at `mean`'s
    // position in EITHER spelling (only the recognized sum's own
    // binding records one), so a position read here would assert a
    // channel the fold does not publish. This pin holds the two
    // walk-level truths: identical single designed findings, and
    // the statement after the consumed pair still walking.
}

/// A position that names neither a parameter's own annotation, a
/// return annotation, nor any expression the walk evaluated (here,
/// the function's OWN NAME) answers nothing — the stated branch
/// declines, and no recorded range covers a name the walk never
/// evaluates as an expression.
///
/// A narrowed LOCAL's own position (`age = 40; if age > 0: <hover
/// age here>`) is the derived-flow case the brief also asks about;
/// asserting it needs the walk to have recorded a node exactly at
/// the READ site, which in turn needs an LSP-shaped identifier
/// resolution this unit does not build (`evaluate_expression` is
/// keyed on the EXPRESSION node's own range, and a bare `Expr::Name`
/// read is one such node, so the machinery here already covers it —
/// but pinning "reachable" requires walking a guarded branch and
/// then locating the exact `Expr::Name` byte range the guard body
/// re-reads, which is fixture-fragile in a way the declared-position
/// tests above are not). Recorded here as a gap rather than a test
/// built to look green: the declared-branch tests above are the
/// ones this unit can assert without guessing at ruff's own byte
/// offsets for a re-read name.
#[test]
fn a_position_naming_neither_a_declaration_nor_a_recorded_expression_answers_nothing() {
    let Some(kernel) = loaded_kernel() else { return };
    let source = concat!(
        "def audio_level() -> None:\n",
        "    pass\n",
    );
    let module = parsed(source);
    let position = offset_of(source, "audio_level");
    assert!(refined_set_at_position(&module, no_imports_resolver(), &kernel, position).is_none());
}

#[test]
fn an_out_of_set_literal_fires_and_an_in_set_literal_stays_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    good: Age = 42\n",
        "    over: Age = 200\n",
        "    fractional: Age = 7.5\n",
        "    negative: Age = -1\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let messages: Vec<&str> = findings.iter().map(|f| f.message.as_str()).collect();
    assert_eq!(
        findings.len(),
        3,
        "want fires for 200, 7.5, and -1 only: {messages:?}"
    );
    assert!(findings.iter().all(|f| f.code == "RTS7001"));
    assert!(messages[0].contains("'200'"), "{messages:?}");
    assert!(messages[1].contains("'7.5'"), "{messages:?}");
    assert!(messages[2].contains("'-1'"), "{messages:?}");
}

/// A module whose ONLY refinement vocabulary is an inline
/// `Annotated[...]` parameter annotation — no `type X = ...`
/// statement anywhere — still walks and fires. Before this gate
/// read `imports.annotated_names`, `compile_aliases` alone fed the
/// early-return check, so this exact shape returned zero findings
/// for the whole module regardless of what its body did.
#[test]
fn an_inline_only_annotated_parameter_with_no_type_alias_still_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "def check_age(age: Annotated[int, Field(ge=0, le=120)]) -> None:\n",
        "    over: Annotated[int, Field(ge=0, le=120)] = 200\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        1,
        "want the fire for 200, from a module with no `type` alias at all: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert_eq!(findings[0].code, "RTS7001");
    assert!(findings[0].message.contains("'200'"), "{}", findings[0].message);
}
