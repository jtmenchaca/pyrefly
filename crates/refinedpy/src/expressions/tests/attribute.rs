use super::*;

/// `math.pi` is an attribute read, not a call — the exact
/// `std::f64::consts::PI` value `math_models::math_constant_value`
/// answers (library/math.rst, "Constants").
#[test]
fn test_math_pi_attribute_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("math.pi").expect("test source must parse");
    let environment = empty_environment();
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![std::f64::consts::PI]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

/// `sys.float_info.max` is a TWO-LEVEL attribute chain, not a call —
/// the exact `f64::MAX` value (library/sys.rst's `float_info.max`
/// row: "The maximum representable positive finite float," `DBL_MAX`
/// — the same platform-independent IEEE 754 identity `math.pi`'s own
/// test above relies on).
#[test]
fn test_sys_float_info_max_attribute_read() {
    let Some(kernel) = loaded_kernel() else { return };
    let parsed = parse_expression("sys.float_info.max").expect("test source must parse");
    let environment = empty_environment();
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![f64::MAX]);
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

/// One module's `from math import ...` table, seeded onto a fresh
/// environment the SAME WAY `check.rs::module_bindings_with_math_
/// imports` seeds `context.module_bindings` for a real walk — the
/// harness every from-import pin below shares.
fn environment_with_math_imports(module: &ruff_python_ast::ModModule) -> Environment {
    let mut environment = empty_environment();
    for (name, value) in math_from_imports(module) {
        environment.bind(&name, value);
    }
    environment
}

/// `from math import inf` then `inf - inf` — IEEE 754's own NaN
/// production (arith.9's doc, cited by `arithmetic_result`), reached
/// only once the bare name `inf` resolves to the CONCRETE
/// `f64::INFINITY` value a from-import now carries, matching
/// `math.inf`'s own attribute spelling.
#[test]
fn test_from_import_inf_minus_inf_is_nan() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from math import inf\n").expect("test module parses").into_syntax();
    let environment = environment_with_math_imports(&module);
    let parsed = parse_expression("inf - inf").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::NaN);
}

/// `from math import inf` then `inf * 0` — the other IEEE 754 NaN
/// producer arith.9 names alongside `inf - inf`.
#[test]
fn test_from_import_inf_times_zero_is_nan() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from math import inf\n").expect("test module parses").into_syntax();
    let environment = environment_with_math_imports(&module);
    let parsed = parse_expression("inf * 0").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::NaN);
}

/// `math.inf - math.inf` — the ATTRIBUTE spelling, same NaN result as
/// the from-import spelling above: both routes resolve to the same
/// concrete `f64::INFINITY` value.
#[test]
fn test_math_inf_attribute_minus_math_inf_is_nan() {
    let Some(kernel) = loaded_kernel() else { return };
    let environment = empty_environment();
    let parsed = parse_expression("math.inf - math.inf").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::NaN);
}

/// `math.inf * 0` — the attribute spelling's own `inf * 0` row.
#[test]
fn test_math_inf_attribute_times_zero_is_nan() {
    let Some(kernel) = loaded_kernel() else { return };
    let environment = empty_environment();
    let parsed = parse_expression("math.inf * 0").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::NaN);
}

/// `from math import nan` — the from-import spelling of the NaN
/// constant itself, matching `math.nan`'s own attribute spelling
/// (`math_models::math_constant_value`'s own doc: `Kind::NaN`, never
/// a value inside `known_values`).
#[test]
fn test_from_import_nan_is_nan_kind() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from math import nan\n").expect("test module parses").into_syntax();
    let environment = environment_with_math_imports(&module);
    let parsed = parse_expression("nan").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::NaN);
}

/// A rename (`from math import inf as infinity`) binds the LOCAL
/// name, not the original `inf` spelling — the same `asname` rule
/// `datetime_imports` already keeps for its own from-import shapes.
#[test]
fn test_from_import_inf_as_rename_binds_the_local_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("from math import inf as infinity\n")
        .expect("test module parses")
        .into_syntax();
    let environment = environment_with_math_imports(&module);
    let parsed = parse_expression("infinity").expect("test source must parse");
    let value = evaluate_expression(&parsed.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Values);
    assert_eq!(value.values, vec![f64::INFINITY]);
}

// --- item 3: attribute read ---

#[test]
fn test_attribute_read_on_a_known_object() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "class Person:\n",
        "    def __init__(self, age):\n",
        "        self.age = age\n",
        "p = Person(40)\n",
        "value = p.age\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let classes = std::sync::Arc::new(crate::instances::class_table(
        &module, &aliases, &imports, &kernel,
    ));
    let mut environment = empty_environment();
    environment.set_classes(classes);
    let constructed = parse_expression("Person(40)").expect("test source must parse");
    let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
    environment.bind("p", instance);
    let read = parse_expression("p.age").expect("test source must parse");
    let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
    assert_eq!(value, known_values(vec![40.0], PrimitiveKind::Integer, TrustProved));
}

/// `@property` read on an instance resolves through
/// `field_read_through_model` — the alias's backing value, not a
/// bound-method opaque.
#[test]
fn test_property_read_resolves_through_the_model() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "class Person:\n",
        "    def __init__(self, age):\n",
        "        self._age = age\n",
        "    @property\n",
        "    def age(self):\n",
        "        return self._age\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let classes =
        std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, &kernel));
    let mut environment = empty_environment();
    environment.set_classes(classes);
    let constructed = parse_expression("Person(40)").expect("test source must parse");
    let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
    environment.bind("person", instance);
    let read = parse_expression("person.age").expect("test source must parse");
    let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
    assert_eq!(value.values, vec![40.0]);
}

/// A plain Attribute READ naming a METHOD (no call parens) answers
/// opaque — "a bare bound-method reference," never the method
/// object's own scalar-shaped havoc.
#[test]
fn test_bare_bound_method_reference_is_opaque() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = environment_with_person_classes(&kernel);
    let constructed = parse_expression("Person(40)").expect("test source must parse");
    let instance = evaluate_expression(&constructed.into_expr(), &environment, &kernel);
    environment.bind("person", instance);
    let read = parse_expression("person.next_year").expect("test source must parse");
    let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("a bare bound-method reference"));
}

/// `super().years` — a bare (un-called) reference to a PARENT
/// method, read from inside a child method's own body: resolves
/// through `self`'s class's `parent_methods`, answering opaque.
#[test]
fn test_super_bare_bound_method_reference_is_opaque() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module(concat!(
        "class Base:\n",
        "    def years(self):\n",
        "        return 40\n",
        "class Child(Base):\n",
        "    def years(self):\n",
        "        return super().years\n",
    ))
    .expect("test module parses")
    .into_syntax();
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let classes =
        std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, &kernel));
    let child = classes.get("Child").expect("Child class recorded");
    let constructed_child = crate::instances::judge_construction(
        child,
        &[],
        &[],
        crate::instances::ConstructionKind::DirectCall,
        &kernel,
    )
    .instance;
    let mut environment = empty_environment();
    environment.set_classes(classes.clone());
    environment.bind("self", constructed_child);
    let read = parse_expression("super().years").expect("test source must parse");
    let value = evaluate_expression(&read.into_expr(), &environment, &kernel);
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("a bare bound-method reference"));
}

// --- opaque values ---

// These four read through `abstract_value::opaque_value`, not
// `abstract_value::opaque()` — "known kind of thing, unknown
// contents" builds Kind::Object with a kind_word (never Kind::Unknown
// / opaque:true, which means "arrived from entirely outside this
// file's determination"). assignability.rs's OPAQUE law depends on
// this: a kind_word'd Kind::Object fires against any scalar-ground
// declared set, so `type(40)` assigned into an int-ground alias
// fires instead of declining Undetermined.

#[test]
fn test_dunder_class_reads_opaque() {
    let Some(value) = eval("object().__class__") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("the __class__ object"));
}

#[test]
fn test_type_call_reads_opaque() {
    let Some(value) = eval("type(40)") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("a type object"));
}

#[test]
fn test_re_compile_reads_opaque() {
    let Some(value) = eval("re.compile(\"a\")") else { return };
    assert_eq!(value.kind, Kind::Object);
    assert_eq!(value.kind_word, Some("a compiled pattern"));
}

/// `re.match` answers the match object OR `None` — library/re.rst,
/// `function::match`: "Return `None` if the string does not match the
/// pattern." The answer is the maybe carrier over the match-object
/// sort, never the match-only claim a caller could read `.group()` off
/// unguarded.
#[test]
fn test_re_match_reads_a_match_object_or_none() {
    let Some(value) = eval("re.match(\"a\", \"banana\")") else { return };
    assert_eq!(value.kind, Kind::PossiblyUndefined);
    let present = value.inner.as_deref().expect("the maybe carrier holds its present side");
    assert_eq!(present.kind, Kind::Object);
    // `"a"` compiles as a pattern, so the present side is the readable-
    // groups match object rather than the bare opaque match sort.
    assert_eq!(present.kind_word, Some("a match object with readable groups"));
}
