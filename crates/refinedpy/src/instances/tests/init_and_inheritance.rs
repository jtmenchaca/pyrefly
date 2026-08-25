use super::*;

// --- __init__-derived fields ---

/// `class Person: def __init__(self, age: int): self.age = age` —
/// d-module-surface.py:21-23's own shape. The parameter flows
/// straight into the field: positional construction maps the
/// argument to `age`, and `field_read` answers it back — with NO
/// fire at construction (the parameter's annotation is a plain
/// `int`, no refinement set), matching d-module-surface.py:128's
/// own comment that the fire happens later, at the return sink,
/// not here.
#[test]
fn init_derived_field_maps_positional_construction_and_reads_back() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Person:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let person = table.get("Person").expect("Person class recorded");
    assert_eq!(person.fields.len(), 1);
    assert_eq!(person.fields[0].name, "age");
    // `int` alone is not a form typereading reads as a refinement
    // (unlike `Age`, an Annotated alias) — no declared set, so no
    // fire is possible at this field regardless of the argument.
    assert!(person.fields[0].declared.is_none());

    let positional = vec![(integer_value(200.0), range_of("200"))];
    let verdict = judge_construction(person, &positional, &[], &kernel);
    assert!(verdict.fires.is_empty(), "a plain int field never fires at construction");
    assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(200.0)));
}

/// A class mixing a class-body `AnnAssign` with an explicit
/// `__init__`: the `__init__`-forwarded parameter takes the
/// POSITIONAL construction slot, but the field keeps the
/// AnnAssign's own declared refinement (the more specific claim) —
/// so a positional construction argument still fires against the
/// class-body's `Age` alias.
#[test]
fn mixed_annassign_and_init_field_keeps_the_annassign_declared_set_at_the_init_position() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Person:\n",
        "    age: Age\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let person = table.get("Person").expect("Person class recorded");
    assert_eq!(person.fields.len(), 1, "the AnnAssign and __init__ rows name the same field");
    assert_eq!(person.fields[0].name, "age");
    assert!(person.fields[0].declared.is_some(), "the AnnAssign's Age set survives the merge");

    let positional = vec![(integer_value(200.0), range_of("200"))];
    let verdict = judge_construction(person, &positional, &[], &kernel);
    assert_eq!(verdict.fires.len(), 1, "200 fires against the AnnAssign's own Age set");
}

/// `self.total = 0` — a self-write whose RHS is a literal, not a
/// parameter: the field exists with that literal as its DEFAULT,
/// no declared refinement, and no construction slot of its own (it
/// trails every parameter-flowing field).
#[test]
fn self_write_with_a_literal_rhs_becomes_a_default_only_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Counter:\n",
        "    def __init__(self, step: int) -> None:\n",
        "        self.step = step\n",
        "        self.total = 0\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let counter = table.get("Counter").expect("Counter class recorded");
    assert_eq!(counter.fields.len(), 2);
    assert_eq!(counter.fields[0].name, "step", "the parameter-flowing field keeps its construction slot");
    assert_eq!(counter.fields[1].name, "total", "the literal self-write trails as a default-only field");
    assert!(counter.fields[1].declared.is_none());
    assert_eq!(counter.fields[1].default, Some(integer_value(0.0)));

    let verdict = judge_construction(counter, &[(integer_value(5.0), range_of("5"))], &[], &kernel);
    assert_eq!(field_read(&verdict.instance, "total"), Some(integer_value(0.0)));
}

/// `self.cache = build_cache()` — a self-write whose RHS is an
/// unmodeled call: `evaluate_expression` declines every call
/// (`expressions.rs`'s own `evaluate_call` contract), so the field
/// exists with no declared refinement and no default — an honest
/// unknown, never a guess.
#[test]
fn self_write_with_an_unreadable_rhs_stays_declared_none_default_none() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Cached:\n",
        "    def __init__(self) -> None:\n",
        "        self.cache = build_cache()\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let cached = table.get("Cached").expect("Cached class recorded");
    assert_eq!(cached.fields.len(), 1);
    assert_eq!(cached.fields[0].name, "cache");
    assert!(cached.fields[0].declared.is_none());
    assert!(cached.fields[0].default.is_none());
}

// --- inheritance via super().__init__ ---

/// `class BaseYears: def __init__(self, age: int): self.age = age`
/// / `class KidYears(BaseYears): def __init__(self, age: int):
/// super().__init__(age)` — e-class-and-function.py:396-408's own
/// shape. `KidYears`'s single field `age` is parent-linked through
/// the `super()` call: a child construction argument flows through
/// to `field_read`, exactly as `super_init_call`'s fixture comment
/// states ("200 carried through the super call").
#[test]
fn super_init_call_links_the_child_construction_argument_to_the_parent_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class BaseYears:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "class KidYears(BaseYears):\n",
        "    def __init__(self, age: int) -> None:\n",
        "        super().__init__(age)\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let kid = table.get("KidYears").expect("KidYears class recorded");
    assert_eq!(kid.fields.len(), 1, "the super() call links to the SAME field, not a duplicate");
    assert_eq!(kid.fields[0].name, "age");

    let positional = vec![(integer_value(200.0), range_of("200"))];
    let verdict = judge_construction(kid, &positional, &[], &kernel);
    assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(200.0)));
}

/// A parent field carrying a declared refinement, forwarded through
/// `super().__init__(...)`: a child construction argument outside
/// that set fires — the child-parameter linkage carries the
/// parent's own declared set forward when the child parameter's
/// own annotation states none.
#[test]
fn super_init_call_construction_fires_when_the_parent_field_carries_a_refinement() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class BaseAged:\n",
        "    age: Age\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "class KidAged(BaseAged):\n",
        "    def __init__(self, age: int) -> None:\n",
        "        super().__init__(age)\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let kid = table.get("KidAged").expect("KidAged class recorded");
    assert!(kid.fields[0].declared.is_some(), "the parent's AnnAssign-declared Age set carries through");

    let positional = vec![(integer_value(200.0), range_of("200"))];
    let verdict = judge_construction(kid, &positional, &[], &kernel);
    assert_eq!(verdict.fires.len(), 1, "200 fires against the inherited Age set");
}

/// A child with NO explicit `__init__` inherits the parent's
/// fields wholesale (datamodel.rst's `object.__init__` — the
/// parent's own `__init__` runs at construction when the child
/// declares none).
#[test]
fn a_child_with_no_init_inherits_the_parents_fields_wholesale() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class BaseYears:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "class ChildYears(BaseYears):\n",
        "    pass\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let child = table.get("ChildYears").expect("ChildYears class recorded");
    assert_eq!(child.fields.len(), 1);
    assert_eq!(child.fields[0].name, "age");

    let positional = vec![(integer_value(40.0), range_of("40"))];
    let verdict = judge_construction(child, &positional, &[], &kernel);
    assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(40.0)));
}
