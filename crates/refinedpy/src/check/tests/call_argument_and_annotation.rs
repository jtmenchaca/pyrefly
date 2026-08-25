use super::*;

/// `record_ratio(float("nan"))` — showcase.py's own designated NaN
/// row shape, mirrored: a same-module call whose WRITTEN argument is
/// a NaN-producing expression, passed directly (no intervening named
/// binding) into a declared, refined parameter. Before `same_module_
/// call_argument_fires` existed, nothing judged a call's own argument
/// against the callee's declared parameter set — `sink_value`'s
/// same-module-call fallthrough only ever computed the call's RETURN
/// value (`evaluate_expression`), never judging what flowed IN. NaN
/// is a member of no refined set (`assignability.rs`'s `Kind::NaN`
/// arm), so this must fire at the argument's own position.
#[test]
fn a_nan_call_argument_fires_against_the_callees_declared_parameter() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Ratio = Annotated[float, Field(ge=0, le=1)]\n",
        "def record_ratio(r: Ratio) -> float:\n",
        "    return r\n",
        "record_ratio(float(\"nan\"))\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        1,
        "want the fire for float(\"nan\") at the call argument: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert_eq!(findings[0].code, "RTS7001");
    assert!(findings[0].message.contains("NaN"), "{}", findings[0].message);
}

/// The same call-argument shape, for `inf - inf` and `inf * 0` — the
/// two other arithmetic-layer producers of `Kind::NaN`
/// (`expressions.rs::arithmetic_result`), spelled `float("inf")`
/// here; the bare `inf`/`math.inf` spellings also bind the concrete
/// `f64::INFINITY` now (`math_models::math_constant_value` and
/// `expressions.rs::math_from_imports`), with their own pinned
/// tests beside those readers. `float("inf")` likewise
/// parse to the exact `f64::INFINITY` value (`builtin_models::
/// float_call`'s own grammar reading), so it is the one spelling
/// that actually reaches `arithmetic_result`'s concrete-f64 NaN
/// check through the live evaluator, matching showcase.py's own
/// `inf - inf`/`inf * 0` rows' ARITHMETIC shape exactly, just not
/// their exact `inf`-name spelling. Both must fire at their own
/// call argument, one finding per statement.
#[test]
fn inf_arithmetic_nan_call_arguments_fire_against_the_callees_declared_parameter() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Ratio = Annotated[float, Field(ge=0, le=1)]\n",
        "def record_ratio(r: Ratio) -> float:\n",
        "    return r\n",
        "record_ratio(float(\"inf\") - float(\"inf\"))\n",
        "record_ratio(float(\"inf\") * 0)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        2,
        "want one fire per NaN call argument (inf - inf, inf * 0): {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert!(findings.iter().all(|f| f.code == "RTS7001"));
    assert!(
        findings.iter().all(|f| f.message.contains("NaN")),
        "{:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
}

/// A non-NaN call-argument fire — `record_ratio(2.0)`, an ordinary
/// out-of-set float — pinning that `same_module_call_argument_fires`
/// is a GENERAL call-argument judging site, not a NaN-only special
/// case: the same discard that lost every NaN row would equally lose
/// this ordinary containment fire, since both reach the sink through
/// the identical unjudged fallthrough.
#[test]
fn an_ordinary_out_of_set_call_argument_fires_against_the_callees_declared_parameter() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Ratio = Annotated[float, Field(ge=0, le=1)]\n",
        "def record_ratio(r: Ratio) -> float:\n",
        "    return r\n",
        "record_ratio(2.0)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        1,
        "want the fire for 2.0 at the call argument: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert_eq!(findings[0].code, "RTS7001");
    assert!(findings[0].message.contains("'2.0'") || findings[0].message.contains("'2'"), "{}", findings[0].message);
}

/// A2.sink.arg's own shape: the WRITTEN argument is an UNBOUND
/// `float` PARAMETER (`Kind::Set`, the unbounded real ray) rather
/// than a literal or a NaN-producing call — `record_ratio(x)` where
/// `x: float` carries no bound at all. `judge_one_call_argument`
/// reads this exactly like any other argument value
/// (`evaluate_expression`), so an unbounded Set crossing into
/// `Ratio`'s `[0, 1]` window must fire the SAME way the literal
/// `2.0` row above does — pinning that the call-argument judge
/// reads a Set-kind argument, not only a Values-kind literal/NaN
/// one.
#[test]
fn an_unbounded_set_call_argument_fires_against_the_callees_declared_parameter() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Ratio = Annotated[float, Field(ge=0, le=1)]\n",
        "def record_ratio(r: Ratio) -> float:\n",
        "    return r\n",
        "def relay(x: float) -> float:\n",
        "    return record_ratio(x)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        1,
        "want the fire for the unbounded float argument crossing into Ratio: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert_eq!(findings[0].code, "RTS7001", "{:?}", findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>());
}

/// showcase.py's own `Vitals`/`record_vitals` shape (examples/
/// showcase.py:305-307): a CLASS-TYPED parameter (`v: Vitals`) whose
/// own argument is a nested construction call
/// (`Vitals(heart_rate=72, spo2=130)`). `declared_refinement`
/// answers `None` for a class name (it only ever reads `context.
/// aliases`, never `context.classes`), so before `judge_one_call_
/// argument`'s class-typed-parameter arm, the whole function
/// returned at that `None` with no judging at all — the
/// construction's own out-of-set `spo2=130` field never reached a
/// Finding, even though `judge_construction` proves it escapes.
#[test]
fn a_construction_nested_in_a_class_typed_call_argument_fires_on_its_own_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import BaseModel, Field\n",
        "class Vitals(BaseModel):\n",
        "    heart_rate: Annotated[int, Field(ge=20, le=250)]\n",
        "    spo2: Annotated[float, Field(ge=0, le=100)]\n",
        "def record_vitals(v: Vitals) -> int:\n",
        "    return 0\n",
        "record_vitals(Vitals(heart_rate=72, spo2=98))\n",
        "record_vitals(Vitals(heart_rate=72, spo2=130))\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the in-set spo2=98 construction must stay silent, and only the \
         out-of-set spo2=130 construction must fire: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'130'"), "{}", fires[0].message);
}

/// A module with neither a `type` alias nor a recognized `Annotated`
/// import — ordinary Python this checker has no vocabulary for —
/// still returns empty through the same zero-cost early return, never
/// reaching the walk.
#[test]
fn a_module_with_no_aliases_and_no_annotated_import_stays_empty() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def add(a: int, b: int) -> int:\n",
        "    return a + b\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(findings.is_empty(), "{:?}", findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>());
}

/// RTS7003 — an inline `Annotated[...]` annotation compiles, but its
/// two bounds contradict each other, so the kernel proves the
/// declared set admits nothing. Mirrors the Go twin's own emptiness
/// fire (`annotation_file_facts.go`, the `emptiness` ask
/// immediately after a successful compile).
#[test]
fn an_annotation_whose_bounds_contradict_denotes_the_empty_set() {
    let Some(kernel) = loaded_kernel() else { return };
    // `findings_for_module_at` returns no findings at all when the
    // module has neither a `type X = ...` alias NOR a recognized
    // `Annotated` import (its own early-return guard) — this
    // fixture's `from typing import Annotated` line alone already
    // clears that guard, so the `type Age` alias here is present to
    // exercise the alias path too, not because it is required.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    impossible: Annotated[int, Field(ge=10, le=5)] = 7\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let empties: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7003").collect();
    assert_eq!(
        empties.len(),
        1,
        "want exactly one empty-set finding: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert!(empties[0].message.contains("denotes the empty set"), "{}", empties[0].message);
}

/// A ordinary, inhabited `Annotated[...]` annotation never fires
/// RTS7003 — the emptiness ask is a courtesy on an already-compiled
/// set, never a blocker on a set that admits values.
#[test]
fn an_ordinary_inhabited_annotation_never_fires_the_empty_set_finding() {
    let Some(kernel) = loaded_kernel() else { return };
    // see the `type Age` comment above — the `from typing import
    // Annotated` line alone already clears
    // `findings_for_module_at`'s alias-or-Annotated-import guard.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    fine: Annotated[int, Field(ge=0, le=120)] = 42\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7003"),
        "{:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
}

/// RTS7004 — the annotation is recognizably this table's OWN
/// `Annotated[...]` vocabulary (the imported `Annotated` identity as
/// the subscript head), but an unrecognized `Field` kwarg refuses
/// the whole statement. Mirrors the Go twin's own `RootsInSurface`
/// gate (`annotation_file_facts.go`): a recognized-root annotation
/// the checker cannot read is never dropped silently.
#[test]
fn an_annotated_statement_with_an_unrecognized_field_kwarg_is_unhonorable() {
    let Some(kernel) = loaded_kernel() else { return };
    // see the `type Age` comment above — the `from typing import
    // Annotated` line alone already clears
    // `findings_for_module_at`'s alias-or-Annotated-import guard.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    over: Annotated[int, Field(unknown_kwarg=1)] = 42\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        1,
        "want exactly one unhonorable-statement finding: {:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
    assert_eq!(findings[0].code, "RTS7004");
    assert!(findings[0].message.contains("Annotated[...]"), "{}", findings[0].message);
}

/// A plain, no-vocabulary annotation (an ordinary class name, not
/// this table's `Annotated[...]` root) never fires RTS7004 — it is
/// ordinary Python this table has no vocabulary for, not a refused
/// statement.
#[test]
fn a_plain_unrelated_annotation_never_fires_the_unhonorable_statement_finding() {
    let Some(kernel) = loaded_kernel() else { return };
    // see the `type Age` comment above — the `from typing import
    // Annotated` line alone already clears
    // `findings_for_module_at`'s alias-or-Annotated-import guard.
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    label: str = \"ok\"\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7004"),
        "{:?}",
        findings.iter().map(|f| (&f.code, &f.message)).collect::<Vec<_>>()
    );
}

/// `f-type-nodes.py`'s `list_annotation_parameter` row: a `list[int]`
/// PARAMETER read through `ages[0]` against `Age` — `int`'s own
/// unbounded ray admits values outside `Age`'s [0, 120] window, so
/// this fires (`seed_parameters`' star-of-a-set seed, read back
/// through `collection_models::subscript_read`'s new `Kind::Set` arm).
#[test]
fn a_list_int_parameters_element_read_fires_against_a_narrower_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def list_annotation_parameter(ages: list[int]) -> Age:\n",
        "    return ages[0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7001");
    assert!(findings[0].message.contains("'Age'"), "{}", findings[0].message);
}

/// The same shape, but the declared element ITSELF is `Age` (an
/// alias, not a bare sort): `list[Age]`'s element already resolves
/// through the ordinary alias path (no fallback needed), and reading
/// its element against `Age` is a set-equals-itself Silent — pinning
/// that the star seed only ever WIDENS what silently determines,
/// never narrows an already-working alias-element row.
#[test]
fn a_list_of_the_declared_sets_own_alias_element_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def first_age(ages: list[Age]) -> Age:\n",
        "    return ages[0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 0, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
}

// --- BARE CLASS-NAME PARAMETERS (`seed_parameters`' class branch) ---

/// A PARAMETER annotated with a bare class name (`request: AudioRequest`)
/// whose class declares an annotated scalar field: the field read
/// through `request.level` is judged against `Level`'s own [0, 100]
/// window — in range stays Silent, out of range fires RTS7001. Pins
/// that `seed_parameters`' new class branch binds the parameter as a
/// tagged `Kind::Object` whose `keys` the ordinary attribute-read path
/// (`evaluate_attribute_read` → `field_read_through_model` →
/// `field_read`) actually consumes — this is the fix's whole point:
/// before it, `request` was never bound at all, and both reads below
/// stayed silently unknown with nothing firing.
#[test]
fn a_bare_class_name_parameters_scalar_field_read_judges_against_its_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Level = Annotated[int, Field(ge=0, le=100)]\n",
        "class AudioRequest:\n",
        "    def __init__(self, level: Level) -> None:\n",
        "        self.level = level\n",
        "def in_range(request: AudioRequest) -> Level:\n",
        "    return request.level\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(
        findings.len(),
        0,
        "a parameter's own declared field read back against its own declared set is Silent: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// The same class-name parameter shape, but the field is read into a
/// NARROWER declared return than the field's own declaration: the
/// parameter's `Level` field ([0, 100]) admits values outside a
/// tighter `Quiet` ([0, 20]) return, so this fires RTS7001 — the
/// out-of-set leg of the same fix.
#[test]
fn a_bare_class_name_parameters_scalar_field_read_fires_against_a_narrower_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Level = Annotated[int, Field(ge=0, le=100)]\n",
        "type Quiet = Annotated[int, Field(ge=0, le=20)]\n",
        "class AudioRequest:\n",
        "    def __init__(self, level: Level) -> None:\n",
        "        self.level = level\n",
        "def maybe_quiet(request: AudioRequest) -> Quiet:\n",
        "    return request.level\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7001");
    assert!(findings[0].message.contains("'Quiet'"), "{}", findings[0].message);
}

/// A bare class-name parameter whose class declares a SEQUENCE field
/// (`samples: Annotated[list[float], Field(min_length=1)]`): indexing
/// the field proves the sequence machinery sees both the window and
/// the sort — mirrors `a_list_int_parameters_element_read_fires_
/// against_a_narrower_declared_set` above (a bare `float`'s own
/// unbounded ray admits values outside `Score`'s [0.0, 1.0] window),
/// but the sequence lives one level down, inside a field, rather than
/// at the parameter itself: `class_field_value`'s sequence arm must
/// build the SAME `Kind::Set` repetition `seed_parameters`' own
/// sequence-container branch builds, or `batch.samples[0]` would read
/// as `unknown()` (undetermined, not a fire) instead.
#[test]
fn a_bare_class_name_parameters_sequence_field_element_read_fires_against_a_narrower_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Score = Annotated[float, Field(ge=0.0, le=1.0)]\n",
        "class ScoredBatch:\n",
        "    def __init__(self, samples: Annotated[list[float], Field(min_length=1)]) -> None:\n",
        "        self.samples = samples\n",
        "def first_score(batch: ScoredBatch) -> Score:\n",
        "    return batch.samples[0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7001");
    assert!(findings[0].message.contains("'Score'"), "{}", findings[0].message);
}

/// A parameter annotated with a name that is NOT a class in this
/// module's table (an ordinary undeclared name, or a name that never
/// resolves to any `ClassModel`): the new branch must decline exactly
/// as today — no binding, no finding — falling through to the ordinary
/// `declared_refinement` read, which also states nothing for it.
#[test]
fn a_class_name_not_in_the_table_still_declines_as_today() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def uses_unknown_class(thing: NotAClass) -> None:\n",
        "    pass\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "an annotation naming no known class must decline without fabricating a finding: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A class-name parameter whose field has NO declared refinement (an
/// ordinary unannotated `self.level = level` field): `class_parameter_
/// object` seeds NOTHING for that key (absent, not a fabricated set),
/// so `request.level` reads back as `unknown()` — flowing into a
/// declared `-> Level` return, `judge` falls through every typed arm to
/// its undetermined catch-all, firing RTS7002 naming that blocker
/// (`"the flowing value is not yet readable"`), never silently
/// admitted and never a false RTS7001.
#[test]
fn a_class_name_parameters_field_with_no_declared_refinement_reads_undetermined() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Level = Annotated[int, Field(ge=0, le=100)]\n",
        "class AudioRequest:\n",
        "    def __init__(self, level) -> None:\n",
        "        self.level = level\n",
        "def read_it(request: AudioRequest) -> Level:\n",
        "    return request.level\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7002");
    assert!(
        findings[0].message.contains("not yet readable"),
        "{}",
        findings[0].message
    );
}

/// The naming unit's own pinned shape (`python-c-extension-boundary.md`):
/// `x: Age = torch.arange(5)` — a call on an attribute chain rooted at
/// an imported-but-unmodeled module name — reads undetermined naming
/// `torch` rather than the generic "not yet readable" wording, since
/// `torch` is never bound to a real tracked value (the walk carries no
/// same-project module named `torch` for the no-resolver test harness
/// to find) and is not one of `expressions::MODELED_MODULE_NAMES`.
#[test]
fn an_unmodeled_module_call_into_a_declared_position_names_the_module() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "import torch\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def read_it() -> None:\n",
        "    x: Age = torch.arange(5)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7002");
    assert!(findings[0].message.contains("'torch'"), "{}", findings[0].message);
    assert!(findings[0].message.contains("no model for"), "{}", findings[0].message);
}

/// The same naming, at a `return` sink rather than an `AnnAssign`:
/// `return torch.arange(5)` under a declared `-> Age` return names
/// `torch` too, proving the naming step is not AnnAssign-specific.
#[test]
fn an_unmodeled_module_call_returned_from_a_declared_function_names_the_module() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "import torch\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def read_it() -> Age:\n",
        "    return torch.arange(5)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7002");
    assert!(findings[0].message.contains("'torch'"), "{}", findings[0].message);
}

/// A MODELED module (`math`, already in `MODELED_MODULE_NAMES`) whose
/// own unmodeled FUNCTION (`math.frexp`, not one of the rows
/// `math_call_result` reads) is a different, narrower gap the naming
/// unit does NOT claim — the sentence stays the generic wording, never
/// misnaming `math` as if the checker carried no model for it at all.
#[test]
fn an_unmodeled_function_on_a_modeled_module_keeps_the_generic_sentence() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "import math\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def read_it() -> None:\n",
        "    x: Age = math.frexp(5.0)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert_eq!(findings.len(), 1, "{:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>());
    assert_eq!(findings[0].code, "RTS7002");
    assert!(!findings[0].message.contains("'math'"), "{}", findings[0].message);
    assert!(findings[0].message.contains("not yet readable"), "{}", findings[0].message);
}

/// RUNG 2's own end-to-end fixture: a tiny hand-authored manifest for
/// a two-function module (`scale(Scalar factor)`, `label(str text)`),
/// exercised against all four shapes
/// `python-c-extension-boundary.md`'s manifest-reader unit names —
/// one module, one temp directory standing in for the checked file's
/// own `entry_directory`.
///
/// 1. A FITTING call (`widgets.scale(2)`): the entry judges silently
///    (no RTS7001), and the return still declines naming the missing
///    producer.
/// 2. An ESCAPING call (`widgets.scale("nope")`): the entry fires
///    RTS7001, naming the module, function, parameter, and both
///    words.
/// 3. An UNLISTED function (`widgets.unlisted(1)`): the module HAS a
///    manifest, but it names no row for `unlisted` — the narrower
///    "manifest names no entry" decline, not rung 1's plain one.
/// 4. An UNMANIFESTED module (`import other_widgets;
///    other_widgets.scale(2)`): no manifest file exists for
///    `other_widgets` at all — rung 1's plain "no model for" decline.
#[test]
fn the_manifest_reader_template_covers_all_four_recognition_shapes() {
    let Some(kernel) = loaded_kernel() else { return };
    let root = std::env::temp_dir().join(format!(
        "refinedpy_check_binding_manifest_fixture_{}_{}",
        std::process::id(),
        std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos()
    ));
    std::fs::create_dir_all(&root).expect("create temp dir");
    let manifest_json = serde_json::json!({
        "scale": {"entry": "scale(Scalar factor)", "producer": "widgets_scale_impl"},
        "label": {"entry": "label(str text)", "producer": "widgets_label_impl"},
    });
    std::fs::write(root.join("widgets.manifest.json"), manifest_json.to_string()).expect("write manifest");

    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "import widgets\n",
        "import other_widgets\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def fitting() -> None:\n",
        "    x: Age = widgets.scale(2)\n",
        "def escaping() -> None:\n",
        "    widgets.scale(\"nope\")\n",
        "def unlisted() -> None:\n",
        "    x: Age = widgets.unlisted(1)\n",
        "def unmanifested() -> None:\n",
        "    x: Age = other_widgets.scale(2)\n",
    ));
    let no_imports: ModuleResolver = &|_: &str| None;
    let findings = findings_for_module_at(&module, no_imports, &kernel, Some(&root));

    // 1. FITTING: no RTS7001 for `fitting`'s own body, and its own
    // RTS7002 blocker names the missing producer.
    let fitting_blocker = findings
        .iter()
        .find(|f| f.code == "RTS7002" && f.message.contains("widgets.scale") && f.message.contains("widgets_scale_impl"))
        .expect(&format!("fitting's own blocker must name the missing producer: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
    assert!(fitting_blocker.message.contains("no producer exports its return fact"), "{}", fitting_blocker.message);

    // 2. ESCAPING: an RTS7001 naming the module, function, parameter,
    // and both value words.
    let escaping_fire = findings
        .iter()
        .find(|f| f.code == "RTS7001" && f.message.contains("widgets.scale"))
        .expect(&format!("the escaping call must fire: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
    assert!(escaping_fire.message.contains("'factor: Scalar'"), "{}", escaping_fire.message);
    assert!(escaping_fire.message.contains("a str"), "{}", escaping_fire.message);

    // 3. UNLISTED: the manifest names no row for `unlisted` — the
    // narrower manifest decline, not the plain module-level one.
    let unlisted_blocker = findings
        .iter()
        .find(|f| f.code == "RTS7002" && f.message.contains("manifest names no entry"))
        .expect(&format!("the unlisted call must name the missing entry: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
    assert!(unlisted_blocker.message.contains("'widgets'"), "{}", unlisted_blocker.message);
    assert!(unlisted_blocker.message.contains("'unlisted'"), "{}", unlisted_blocker.message);

    // 4. UNMANIFESTED: no manifest file exists for `other_widgets` —
    // rung 1's own plain decline.
    let unmanifested_blocker = findings
        .iter()
        .find(|f| f.code == "RTS7002" && f.message.contains("'other_widgets'"))
        .expect(&format!("the unmanifested module must fall back to rung 1's naming: {:?}", findings.iter().map(|f| &f.message).collect::<Vec<_>>()));
    assert!(unmanifested_blocker.message.contains("no model for"), "{}", unmanifested_blocker.message);

    std::fs::remove_dir_all(&root).ok();
}
