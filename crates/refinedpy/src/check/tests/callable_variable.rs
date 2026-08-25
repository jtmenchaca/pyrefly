use super::*;

// --- Literal[...] int-only inline recognition (typereading.rs) ---

#[test]
fn an_int_literal_alias_and_an_inline_literal_annotation_both_judge() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Literal\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def rows() -> None:\n",
        "    small: Literal[10, 20] = 10\n",
        "    good: Age = small\n",
        "    big: Literal[200, 201] = 200\n",
        "    over: Age = big\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "only the Literal[200, 201]-typed `big` read is out of Age's [0, 120] window: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

// --- callable-variable calls (typereading.rs::callable_return_refinement,
// env.rs::callable_returns, check.rs::callable_variable_call_result) ---

/// The smallest DIRECT-sink shape: `x: Age = maybe_next_year(40)` puts
/// the call straight into `sink_value`'s own value expression (no
/// ternary in between) — `maybe_next_year`'s bare `int` return sort
/// (`Callable[[int], int]`, no refined alias) is the unbounded
/// whole-number ray, which is NOT a subset of Age's `[0, 120]`
/// window, so the containment law fires.
#[test]
fn a_direct_callable_variable_call_sink_fires_against_a_declared_alias() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "maybe_next_year: Callable[[int], int] | None = None\n",
        "def rows() -> None:\n",
        "    over: Age = maybe_next_year(40)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the callable's own unrefined int return admits values outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// A callable variable whose declared return IS a refined alias
/// (`Callable[[int], Age]`) reads Age's own set at the call site —
/// an in-window argument-independent call is silent, since this
/// channel judges the RETURN refinement, never the call's own
/// arguments.
#[test]
fn a_direct_callable_variable_call_sink_is_silent_when_the_return_is_already_the_declared_alias() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "next_year: Callable[[int], Age] | None = None\n",
        "def rows() -> None:\n",
        "    fine: Age = next_year(40)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "Callable[[int], Age]'s own return is already Age-refined: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

// --- Callable-annotated PARAMETER / cast() seeding (A10.guard rows) ---

/// A10.guard.eq — a `Callable[[Age], Age]`-annotated PARAMETER (not
/// a `x: Callable[...] = ...` body-local, the shape the tests above
/// already cover) seeds `f`'s own callable-returns entry through
/// `seed_parameters`'s new arm; `f is known` needs no identity
/// narrowing at all — the declared Callable's own `R` (`Age`)
/// already states the whole fact `return f(x)` needs.
#[test]
fn a_callable_annotated_parameter_seeds_its_declared_return_after_an_identity_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Callable\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def known(x: Age) -> Age:\n",
        "    return x\n",
        "def after_identity_inside(f: Callable[[Age], Age], x: Age) -> Age:\n",
        "    if f is known:\n",
        "        return f(x)\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a Callable-annotated parameter's own declared return must silence the guarded call: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A10.guard.ne — the inequality complement: `f is not known and f is
/// other` still calls through the SAME Callable-annotated `f`, so the
/// identical seeding applies with no narrowing on either identity leg.
#[test]
fn a_callable_annotated_parameter_seeds_its_declared_return_after_an_inequality_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Callable\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def known(x: Age) -> Age:\n",
        "    return x\n",
        "def other(x: Age) -> Age:\n",
        "    if 0 <= x <= 150:\n",
        "        return x\n",
        "    return 0\n",
        "def after_inequality_inside(f: Callable[[Age], Age], x: Age) -> Age:\n",
        "    if f is not known and f is other:\n",
        "        return f(x)\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "a Callable-annotated parameter's own declared return must silence the guarded call: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A10.guard.exit — `g = cast(Callable[[Age], Age], f)` records `g`'s
/// own callable-returns entry through `walk_assign`'s new cast
/// recognizer, reached below an early `not callable(f)` exit.
#[test]
fn a_cast_to_callable_seeds_its_declared_return_below_a_callable_exit_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Any, Callable, cast\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def below_exit_inside(f: Any, x: Age) -> Age:\n",
        "    if not callable(f):\n",
        "        return 0\n",
        "    g = cast(Callable[[Age], Age], f)\n",
        "    if 0 <= x <= 150:\n",
        "        return g(x)\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "cast(Callable[[Age], Age], f)'s own declared return must silence the call through g: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A10.guard.sort — the same cast shape, guarded by `callable(f) and
/// 0 <= x <= 150` in one `if` test instead of an early exit.
#[test]
fn a_cast_to_callable_seeds_its_declared_return_after_a_callable_guard() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Any, Callable, cast\n",
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def after_callable_guard_inside(f: Any, x: Age) -> Age:\n",
        "    if callable(f) and 0 <= x <= 150:\n",
        "        g = cast(Callable[[Age], Age], f)\n",
        "        return g(x)\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "cast(Callable[[Age], Age], f)'s own declared return must silence the call through g: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A10.guard.truthy — `f = identity` (a module `def`, not a
/// `Callable`-annotated parameter) binds `f` to a same-module-def
/// alias value (`env::same_module_def_alias_value`); `f(x)` resolves
/// through `evaluate_call`'s new alias-call arm to `identity`'s own
/// body, interpreted exactly as a direct `identity(x)` call would be
/// — a function value is always truthy, so `if f:` always takes the
/// call branch.
#[test]
fn a_name_bound_to_a_same_module_def_calls_through_to_that_defs_own_body() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def identity(x: Age) -> Age:\n",
        "    return x\n",
        "def truthy_always_inside(x: Age) -> Age:\n",
        "    f = identity\n",
        "    if f:\n",
        "        if 0 <= x <= 150:\n",
        "            return f(x)\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "f = identity must call through to identity's own body: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A name bound to a same-module def still calls through when the
/// def's own body would fire against a DIFFERENT declared alias —
/// pins that the alias call genuinely reaches the real body (through
/// `call_result_with_enclosing`) rather than silently reading as an
/// unjudged `unknown()` that a containment law never gets to see.
#[test]
fn a_name_bound_to_a_same_module_def_still_fires_when_the_defs_body_would() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=150)]\n",
        "def two_hundred() -> Age:\n",
        "    return 200\n",
        "def rows() -> None:\n",
        "    f = two_hundred\n",
        "    over: Age = f()\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "a same-module-def alias call must still reach the real out-of-set body: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// UNIT 3, site 5 (`callable_variable_call_result`): `Age`'s own
/// declared set is numeric-ground, so the callable's return must
/// carry `kind_tag: Some(Integer)` once bound to `year` — piping
/// `year` through `math.sqrt` (a sort-gated consumer,
/// `sqrt_call_over_set`, math_models.rs) derives a value instead of
/// leaving the return undetermined. The call sits at its own DIRECT
/// sink (`year: Age = next_year(40)`) — `callable_variable_call_
/// result` only reads a call at `sink_value`'s own value-expression
/// position (this file's own doc: "5. The CALLEE-EFFECTS CHANNEL"),
/// never a call nested as another call's argument.
#[test]
fn a_callable_variable_call_results_declared_set_reaches_sqrt_tagged() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "import math\n",
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "type Root = Annotated[float, Field(ge=0.0, le=20.0)]\n",
        "next_year: Callable[[int], Age] | None = None\n",
        "def f() -> Root:\n",
        "    year: Age = next_year(40)\n",
        "    return math.sqrt(year)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.is_empty(),
        "the tagged callable return must let math.sqrt derive rather than blocking: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// b-body-expressions.py:38/79's own shape verbatim, EXCEPT the call
/// sits at a DIRECT sink (no ternary): `maybe_next_year(40)` read
/// straight into a `return -> Age`. This is the shape this unit's
/// `sink_value` channel reaches; the fixture row's own
/// `maybe_next_year(40) if maybe_next_year is not None else 0` ternary
/// wrapping is a DIFFERENT shape this channel does not reach — see
/// this unit's report (the call there is evaluated inside
/// `evaluate_ternary`'s `evaluate_expression`/`evaluate_call`
/// recursion in expressions.rs, never through `sink_value`).
#[test]
fn the_b74_shape_without_its_ternary_wrapper_fires_at_a_return_sink() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "maybe_next_year: Callable[[int], int] | None = None\n",
        "def call_direct() -> Age:\n",
        "    return maybe_next_year(40)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the guarded call's own unrefined int return admits values outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// A resolvable same-module `def` of the same name wins over the
/// callable-returns table — the ordinary `summaries::call_result`
/// path (which reads the def's ACTUAL body) owns a name that
/// resolves to a real def, never this fallback.
#[test]
fn a_name_resolving_to_a_same_module_def_is_not_read_as_a_callable_variable() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "greet: Callable[[int], int] | None = None\n",
        "def greet(x: int) -> int:\n",
        "    return 40\n",
        "def rows() -> None:\n",
        "    fine: Age = greet(1)\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    assert!(
        findings.iter().all(|f| f.code != "RTS7001"),
        "the same-module def `greet` (always returns 40, in-window) must win over the callable-returns fallback: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
}

/// b-body-expressions.py:76-79's own shape verbatim: the callable
/// call sits inside a ternary's `body` arm
/// (`maybe_next_year(40) if maybe_next_year is not None else 0`),
/// which `evaluate_ternary` (expressions.rs) evaluates through plain
/// `evaluate_expression`/`evaluate_call` recursion, never through
/// `sink_value` — the gap
/// `the_b74_shape_without_its_ternary_wrapper_fires_at_a_return_sink`
/// documents as this channel's own remaining shape. This test proves
/// `evaluate_call`'s own callable-variable-call arm (added alongside
/// this test) closes it: the ternary's test
/// (`maybe_next_year is not None`) is not provably decided from a
/// bare module-level `Callable | None` binding, so both arms
/// evaluate and `join_known` joins the call's own `known_set`
/// (`R`'s unbounded whole-number ray, TrustSpec) with the literal
/// `0` (Kind::Values, Integer) — the untagged-Set-vs-Values join
/// falls to `join_known`'s bottom numeric-set path (`is_numeric_kind`
/// admits any non-Values kind, so `Kind::Set` always qualifies) and
/// answers the union of the two sides' own sets, still admitting
/// values Age's `[0, 120]` window does not, so the containment law
/// fires.
#[test]
fn the_ternary_wrapped_b79_shape_fires_through_join_known() {
    let Some(kernel) = loaded_kernel() else { return };
    // the VALUELESS module AnnAssign is the faithful twin of TS
    // `declare const maybeNextYear: ... | undefined` — a concrete
    // `= None` initializer would make the guard provably false and
    // the silent answer honest, which is a different row entirely
    let module = parsed(concat!(
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "maybe_next_year: Callable[[int], int] | None\n",
        "def call_optional() -> Age:\n",
        "    return maybe_next_year(40) if maybe_next_year is not None else 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the guarded call still admits a whole number outside the set: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// A callable-variable call reached ONLY through `evaluate_call`
/// (expressions.rs), never through `sink_value`'s own
/// `callable_variable_call_result` — `walk_assign`'s value routes
/// through `sink_value` first (which already answers a bare
/// `over = maybe_next_year(40)` assignment before `evaluate_call` is
/// ever reached), so this test nests the call one level deeper, as
/// the single element of a list display read back by index:
/// `[maybe_next_year(40)][0]`. `sink_value` reads the WHOLE
/// subscript expression (not a bare Call node) and declines, falling
/// through to `evaluate_expression`'s list-display and subscript
/// arms, which recurse into `evaluate_call` for the display's own
/// element — the one path this unit's arm, and only this unit's
/// arm, answers.
#[test]
fn a_callable_variable_call_nested_inside_a_list_display_fires_via_evaluate_call() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, Callable\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "maybe_next_year: Callable[[int], int] | None = None\n",
        "def call_nested_in_list_display() -> Age:\n",
        "    return [maybe_next_year(40)][0]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the callable's own unrefined int return, read back through the display, still admits values outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// a-statements.py's own `with_statement`/`device()` shape: `device()`
/// is a MODULE-LEVEL `def` whose body declares a LOCAL class
/// (`_Device`) and returns its construction — `with device() as
/// handle:` never walks `device`'s body directly (`check.rs` only
/// EVALUATES the context expression as a value), so the instance
/// `summaries::call_result_with_enclosing` tags `source = "_Device"`
/// must be resolvable through `context.classes`, the ONLY table
/// `enter_method_result` consults — this pins the module-level-def
/// local-class registration this unit added in
/// `findings_for_module_with_resolver` (the loop scanning every
/// top-level `def`'s own body via `local_class_table`). Without it,
/// `enter_method_result` declines (`context.classes.get("_Device")`
/// answers `None`), `handle` is forgotten, and `handle.value` never
/// fires — the ONE fire this test asserts.
#[test]
fn with_statement_over_a_same_module_def_returning_a_local_class_instance_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def unread_number() -> int:\n",
        "    raise NotImplementedError\n",
        "def device():\n",
        "    class _Device:\n",
        "        value: int = 0\n",
        "        def __enter__(self):\n",
        "            self.value = unread_number()\n",
        "            return self\n",
        "        def __exit__(self, *exc_info):\n",
        "            return False\n",
        "    return _Device()\n",
        "def with_statement() -> Age:\n",
        "    with device() as handle:\n",
        "        return handle.value\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the __enter__-assigned opaque int admits values outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// a-statements.py's own `async_with_statement`/`AsyncDevice` shape:
/// the class is declared DIRECTLY inside the `async with` statement's
/// own enclosing function (a body-local class, already reachable
/// through `local_class_table`/`merged_classes_for_body` — no
/// same-module-def indirection the way `device()`/`with_statement`
/// needs), and its `__aenter__` (not `__enter__`) is what
/// `enter_method_result` must dispatch to for `with_stmt.is_async`.
/// Proof the `__aenter__` half of that dispatch fires exactly like
/// the sync `__enter__` half already does.
#[test]
fn async_with_statement_over_a_body_local_class_dispatches_aenter_and_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def unread_number() -> int:\n",
        "    raise NotImplementedError\n",
        "async def async_with_statement() -> Age:\n",
        "    class AsyncDevice:\n",
        "        value: int = 0\n",
        "        async def __aenter__(self):\n",
        "            self.value = unread_number()\n",
        "            return self\n",
        "        async def __aexit__(self, *exc_info):\n",
        "            return False\n",
        "    async with AsyncDevice() as handle:\n",
        "        return handle.value\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the __aenter__-assigned opaque int admits values outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// a-statements.py's own `nonlocal_rebind` shape end-to-end: `bump()`
/// rebinds the enclosing `age` in-set (silent), `spoil()` rebinds it
/// out-of-set (fires) — proof the CALLEE-EFFECTS CHANNEL
/// (`apply_call_effects`) is wired into the ordinary statement walk,
/// not merely unit-tested against `summaries::call_effects` in
/// isolation.
#[test]
fn nonlocal_rebind_fires_once_at_the_out_of_set_call_site() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def nonlocal_rebind() -> Age:\n",
        "    age: Age = 10\n",
        "    def bump() -> None:\n",
        "        nonlocal age\n",
        "        age = 15\n",
        "    bump()\n",
        "    def spoil() -> None:\n",
        "        nonlocal age\n",
        "        age = 200\n",
        "    spoil()\n",
        "    return age\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "bump()'s in-set rebind must stay silent; only spoil()'s 200 fires: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// a-statements.py's own `closure_mutates_flattened_capture` shape
/// end-to-end: `spoil()` mutates a captured dict through a subscript
/// store with no `nonlocal` declaration at all, and the LATER read
/// `outlaw["age"]` (never inside `spoil` itself) is what fires —
/// proof the effect survives back into the caller's own environment
/// and is read at a plain dict-subscript sink.
#[test]
fn closure_mutates_flattened_capture_fires_at_the_later_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "def closure_mutates_flattened_capture() -> Age:\n",
        "    outlaw = {\"age\": 40}\n",
        "    def spoil() -> None:\n",
        "        outlaw[\"age\"] = 200\n",
        "    spoil()\n",
        "    return outlaw[\"age\"]\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the closure's subscript mutation must carry 200 into the later read: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}

/// a-statements.py's own `async_for_over_stream` shape end-to-end:
/// `stream() -> AsyncIterator[int]` declines concretely (`raise
/// NotImplementedError`), so the loop only runs through the ABSTRACT
/// SORT-ELEMENT PASS (`loops::abstract_element_sort_pass`) — proof
/// the pass is wired into the ordinary loop walk (`walk_loop`), not
/// merely unit-tested against `loop_final_environment` directly.
#[test]
fn async_for_over_stream_fires_through_the_abstract_element_sort_pass() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated, AsyncIterator\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "async def stream() -> AsyncIterator[int]:\n",
        "    raise NotImplementedError\n",
        "    yield 0\n",
        "async def async_for_over_stream() -> Age:\n",
        "    age: Age = 0\n",
        "    async for chunk in stream():\n",
        "        age = chunk\n",
        "    return 0\n",
    ));
    let findings = findings_for_module(&module, &kernel);
    let fires: Vec<&Finding> = findings.iter().filter(|f| f.code == "RTS7001").collect();
    assert_eq!(
        fires.len(),
        1,
        "the whole-int element sort admits values outside Age: {:?}",
        findings.iter().map(|f| &f.message).collect::<Vec<_>>()
    );
    assert!(fires[0].message.contains("'Age'"), "{}", fires[0].message);
}
