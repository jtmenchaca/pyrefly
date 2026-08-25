use super::*;

// --- property accessors ---

/// `class Aged: def __init__(self): self._held = 40` / `@property
/// def age(self): return self._held` — e-class-and-function.py:
/// 336-344's own shape. `age` reads as an alias of `_held`'s value.
#[test]
fn property_read_aliases_the_backing_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Aged:\n",
        "    def __init__(self) -> None:\n",
        "        self._held = 40\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self._held\n",
        "    @age.setter\n",
        "    def age(self, value: int) -> None:\n",
        "        self._held = value\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let aged = table.get("Aged").expect("Aged class recorded");
    assert!(aged.properties.contains_key("age"));
    assert_eq!(aged.properties["age"].backing, "_held");

    let verdict = judge_construction(aged, &[], &[], &kernel);
    assert_eq!(
        field_read_through_model(aged, &verdict.instance, "age"),
        Some(integer_value(40.0)),
        "reading the property answers the backing field's own value"
    );
}

/// A setter whose parameter carries a declared refinement
/// (`value: Age`, not a plain `int`): a write through the property
/// fires when the value is outside that set.
#[test]
fn property_setter_write_fires_through_field_write_judgment() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import Field\n",
        "type Age = Annotated[int, Field(ge=0, le=120)]\n",
        "class Aged:\n",
        "    def __init__(self) -> None:\n",
        "        self._held = 40\n",
        "    @property\n",
        "    def age(self) -> int:\n",
        "        return self._held\n",
        "    @age.setter\n",
        "    def age(self, value: Age) -> None:\n",
        "        self._held = value\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let aged = table.get("Aged").expect("Aged class recorded");
    assert!(aged.properties["age"].declared.is_some(), "the setter's Age annotation is read");

    let verdict = field_write_judgment(aged, "age", &integer_value(200.0), &kernel);
    assert!(matches!(verdict, Some(Verdict::Fire(_))), "200 fires against the setter's own Age set");
}

// --- ClassModel is Clone ---

#[test]
fn class_model_clones_its_fields_properties_and_methods() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Aged:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "    def next_year(self) -> int:\n",
        "        return self.age + 1\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let aged = table.get("Aged").expect("Aged class recorded");
    let cloned = aged.clone();
    assert_eq!(cloned.name, aged.name);
    assert_eq!(cloned.fields.len(), aged.fields.len());
    assert!(cloned.methods.contains_key("next_year"), "the clone keeps the method table");
}

// --- method_def_of: own-overrides-inherited ---

#[test]
fn method_def_of_reads_a_class_own_method() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Aged:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "    def next_year(self) -> int:\n",
        "        return self.age + 1\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let aged = table.get("Aged").expect("Aged class recorded");
    let method = method_def_of(aged, "next_year").expect("next_year is a declared method");
    assert_eq!(method.name.id.as_str(), "next_year");
    assert!(method_def_of(aged, "missing").is_none());
}

/// A child overriding a parent's method: `method_def_of` on the
/// child answers the CHILD's own def (its body differs from the
/// parent's — `label` returns 2, not the parent's 1), while
/// `parent_methods` still carries the parent's original — the
/// `super()` resolution target, proven by running both defs through
/// `method_call_result` and comparing their answers.
#[test]
fn method_def_of_prefers_the_childs_own_override_over_the_inherited_def() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class BaseYears:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "    def label(self) -> int:\n",
        "        return 1\n",
        "class KidYears(BaseYears):\n",
        "    def __init__(self, age: int) -> None:\n",
        "        super().__init__(age)\n",
        "    def label(self) -> int:\n",
        "        return 2\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let kid = table.get("KidYears").expect("KidYears class recorded");
    let instance = judge_construction(kid, &[(integer_value(40.0), range_of("40"))], &[], &kernel).instance;

    let effective = method_def_of(kid, "label").expect("label is declared");
    let (_after, effective_result) = method_call_result(&instance, kid, effective, &[], None, None, None, &kernel, 0)
        .expect("the child's own label() must interpret");
    assert_eq!(effective_result, integer_value(2.0), "method_def_of answers the CHILD's own override");

    let inherited = kid.parent_methods.get("label").expect("parent_methods keeps the parent's own def");
    let (_after, inherited_result) = method_call_result(&instance, kid, inherited, &[], None, None, None, &kernel, 0)
        .expect("the parent's own label() must interpret");
    assert_eq!(inherited_result, integer_value(1.0), "parent_methods is unaffected by the child's override");
}

// --- method_call_result: write-then-read, and the super() chain ---

/// `outlaw.spoil()` where `spoil` writes `self.age = 200` and reads
/// nothing back itself — the RETURNED instance must carry the
/// write, matching b-body-expressions.py's own
/// `literal_writing_method` shape (ORIENTATION.md's own citation for
/// `method_call_result`).
#[test]
fn method_call_result_write_then_read_survives_on_the_returned_instance() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Outlaw:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "    def spoil(self) -> None:\n",
        "        self.age = 200\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let outlaw = table.get("Outlaw").expect("Outlaw class recorded");
    let instance = judge_construction(outlaw, &[(integer_value(40.0), range_of("40"))], &[], &kernel).instance;
    let method = method_def_of(outlaw, "spoil").expect("spoil is declared");
    let (after, _result) = method_call_result(&instance, outlaw, method, &[], None, None, None, &kernel, 0)
        .expect("spoil's straight-line self-write must interpret");
    assert_eq!(field_read(&after, "age"), Some(integer_value(200.0)), "the write survives on the returned instance");
}

/// `KidYears(age=200).years()` where `years` calls
/// `super().years() + 1` — the parent's OWN `years` (never the
/// child's, since `KidYears` declares no override of that name)
/// answers through `parent_methods`, and the child's own method adds
/// 1 to it.
#[test]
fn method_call_result_resolves_a_super_call_through_parent_methods() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class BaseYears:\n",
        "    def __init__(self, age: int) -> None:\n",
        "        self.age = age\n",
        "    def years(self) -> int:\n",
        "        return self.age\n",
        "class KidYears(BaseYears):\n",
        "    def __init__(self, age: int) -> None:\n",
        "        super().__init__(age)\n",
        "    def call_super_method(self) -> int:\n",
        "        return super().years() + 1\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let kid = table.get("KidYears").expect("KidYears class recorded");
    let instance = judge_construction(kid, &[(integer_value(40.0), range_of("40"))], &[], &kernel).instance;
    let method = method_def_of(kid, "call_super_method").expect("call_super_method is declared");
    let (_after, result) = method_call_result(&instance, kid, method, &[], None, None, None, &kernel, 0)
        .expect("the super().years() call must resolve through parent_methods");
    assert_eq!(result, integer_value(41.0), "super().years() answers 40, plus 1");
}

/// A class method body's own `from datetime import date`-aliased
/// `date(2024, 3, 1)` construction recognizes IDENTICALLY to the same
/// call in a plain function (`expressions.rs`'s own
/// `test_bare_imported_date_construction_matches_the_qualified_
/// spelling`, mirrored here for a method body): `method_call_result`
/// is handed the module's own `datetime_imports` table explicitly
/// (the same explicit-parameter shape `classes` already takes), so
/// the method body's fresh `Environment` resolves the bare `date`
/// alias exactly as a module-level call would, rather than reading
/// no table at all and declining the construction.
#[test]
fn method_body_bare_imported_date_construction_matches_the_qualified_spelling() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from datetime import date\n",
        "class Anniversary:\n",
        "    def __init__(self, count: int) -> None:\n",
        "        self.count = count\n",
        "    def occasion(self):\n",
        "        return date(2024, 3, 1)\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let anniversary = table.get("Anniversary").expect("Anniversary class recorded");
    let instance =
        judge_construction(anniversary, &[(integer_value(1.0), range_of("1"))], &[], &kernel).instance;
    let method = method_def_of(anniversary, "occasion").expect("occasion is declared");
    let datetime_imports = Arc::new(crate::expressions::datetime_imports(&module));

    let plain_environment = {
        let mut environment = Environment::new(Default::default());
        environment.set_datetime_imports(Arc::new(crate::expressions::datetime_imports(&module)));
        environment
    };
    let plain_parsed = ruff_python_parser::parse_expression("date(2024, 3, 1)").expect("test source parses");
    let plain_value = crate::expressions::evaluate_expression(&plain_parsed.into_expr(), &plain_environment, &kernel);

    let (_after, method_value) =
        method_call_result(&instance, anniversary, method, &[], None, None, Some(&datetime_imports), &kernel, 0)
            .expect("the method body's own date(...) construction must interpret");

    assert_eq!(method_value.kind, Kind::Object);
    assert_eq!(method_value, plain_value, "a method body's aliased date(...) construction must equal the same call in a plain function");
}
