use super::*;

// --- judge_construction: positional mapping ---

#[test]
fn judge_construction_maps_positional_arguments_in_declaration_order() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Person",
        vec![
            ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None, base_sort: None },
            ClassField { name: "label".to_owned(), declared: None, default: None, base_sort: None },
        ],
    );
    let positional = vec![
        (integer_value(40.0), range_of("40")),
        (known_values(vec![0.0], PrimitiveKind::String, TrustProved), range_of("label")),
    ];
    let verdict = judge_construction(&model, &positional, &[], ConstructionKind::DirectCall, &kernel);
    assert!(verdict.fires.is_empty());
    assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(40.0)));
}

// --- judge_construction: keyword out-of-set fire ---

#[test]
fn judge_construction_keyword_out_of_set_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Person",
        vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None, base_sort: None }],
    );
    let keyword = vec![("age".to_owned(), integer_value(200.0), range_of("200"))];
    let verdict = judge_construction(&model, &[], &keyword, ConstructionKind::DirectCall, &kernel);
    assert_eq!(verdict.fires.len(), 1);
    assert!(verdict.fires[0].1.contains("'200'"), "{}", verdict.fires[0].1);
}

/// A keyword naming no field on the class is an unmodeled
/// construction — unknown() with no fires, never a guess.
#[test]
fn judge_construction_unknown_keyword_declines_whole() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Person",
        vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None, base_sort: None }],
    );
    let keyword = vec![("nickname".to_owned(), integer_value(1.0), range_of("1"))];
    let verdict = judge_construction(&model, &[], &keyword, ConstructionKind::DirectCall, &kernel);
    assert!(verdict.fires.is_empty());
    assert_eq!(verdict.instance.kind, Kind::Unknown);
}

// --- missing-arg default ---

#[test]
fn judge_construction_missing_argument_takes_the_default() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Grow",
        vec![ClassField { name: "age".to_owned(), declared: None, default: Some(integer_value(18.0)), base_sort: None }],
    );
    let verdict = judge_construction(&model, &[], &[], ConstructionKind::DirectCall, &kernel);
    assert!(verdict.fires.is_empty());
    assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(18.0)));
}

/// A missing argument with no default but a declared set holds the
/// DECLARED SET (TrustSpec), the same construction seed_parameters
/// uses for an unbound parameter.
#[test]
fn judge_construction_missing_argument_with_no_default_holds_declared_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Person",
        vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None, base_sort: None }],
    );
    let verdict = judge_construction(&model, &[], &[], ConstructionKind::DirectCall, &kernel);
    assert!(verdict.fires.is_empty());
    let field = field_read(&verdict.instance, "age").expect("age field present");
    assert_eq!(field.kind, Kind::Set);
}

// --- model_post_init: the dependent-check hook ---

/// m-pydantic-schema.py's own `Range` shape: `model_post_init(self,
/// __context): if self.hi < self.lo: raise ValueError(...)`. A
/// construction whose fields provably satisfy `hi >= lo` never
/// fires here — the post-init condition reads False.
#[test]
fn model_post_init_is_silent_when_the_dependent_check_passes() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Range:\n",
        "    lo: int\n",
        "    hi: int\n",
        "    def model_post_init(self, __context) -> None:\n",
        "        if self.hi < self.lo:\n",
        "            raise ValueError(\"hi must be >= lo\")\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let range_model = table.get("Range").expect("Range class recorded");
    let keyword = vec![
        ("lo".to_owned(), integer_value(10.0), range_of("10")),
        ("hi".to_owned(), integer_value(20.0), range_of("20")),
    ];
    let verdict = judge_construction(range_model, &[], &keyword, ConstructionKind::DirectCall, &kernel);
    assert!(verdict.fires.is_empty(), "hi (20) >= lo (10): the dependent check never raises");
}

/// The refused pair: `hi` (5) below `lo` (10) — the post-init
/// condition provably reads True, so construction fires with the
/// `ValueError`'s own message.
#[test]
fn model_post_init_fires_when_the_dependent_check_provably_raises() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Range:\n",
        "    lo: int\n",
        "    hi: int\n",
        "    def model_post_init(self, __context) -> None:\n",
        "        if self.hi < self.lo:\n",
        "            raise ValueError(\"hi must be >= lo\")\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let range_model = table.get("Range").expect("Range class recorded");
    let keyword = vec![
        ("lo".to_owned(), integer_value(10.0), range_of("10")),
        ("hi".to_owned(), integer_value(5.0), range_of("5")),
    ];
    let verdict = judge_construction(range_model, &[], &keyword, ConstructionKind::DirectCall, &kernel);
    assert_eq!(verdict.fires.len(), 1, "hi (5) < lo (10): the dependent check provably raises");
    assert!(verdict.fires[0].1.contains("ValueError"), "{}", verdict.fires[0].1);
    assert!(verdict.fires[0].1.contains("hi must be >= lo"), "{}", verdict.fires[0].1);
}

/// An undetermined field (no keyword argument at all, so `hi`/`lo`
/// both hold the declared int base sort, not a concrete value)
/// never fires — `truthiness` cannot decide the condition, and this
/// reader's own honest-decline discipline never guesses.
#[test]
fn model_post_init_never_fires_on_an_undetermined_condition() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "class Range:\n",
        "    lo: int\n",
        "    hi: int\n",
        "    def model_post_init(self, __context) -> None:\n",
        "        if self.hi < self.lo:\n",
        "            raise ValueError(\"hi must be >= lo\")\n",
    ));
    let aliases = HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let range_model = table.get("Range").expect("Range class recorded");
    let verdict = judge_construction(range_model, &[], &[], ConstructionKind::DirectCall, &kernel);
    assert!(verdict.fires.is_empty(), "an undetermined comparison never guesses a fire");
}

// --- field_read ---

#[test]
fn field_read_on_the_built_instance() {
    let Some(kernel) = loaded_kernel() else { return };
    let model =
        bare_model("Person", vec![ClassField { name: "age".to_owned(), declared: None, default: None, base_sort: None }]);
    let positional = vec![(integer_value(40.0), range_of("40"))];
    let verdict = judge_construction(&model, &positional, &[], ConstructionKind::DirectCall, &kernel);
    assert_eq!(field_read(&verdict.instance, "age"), Some(integer_value(40.0)));
    assert_eq!(field_read(&verdict.instance, "missing"), None);
    assert_eq!(field_read(&unknown(), "age"), None);
}

// --- field_write_judgment ---

#[test]
fn field_write_judgment_fires_on_an_out_of_set_write() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Aged",
        vec![ClassField { name: "age".to_owned(), declared: Some(age_declared()), default: None, base_sort: None }],
    );
    let verdict = field_write_judgment(&model, "age", &integer_value(200.0), &kernel);
    assert!(matches!(verdict, Some(Verdict::Fire(_))));
}

#[test]
fn field_write_judgment_is_none_for_an_undeclared_field() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model("Aged", vec![ClassField { name: "age".to_owned(), declared: None, default: None, base_sort: None }]);
    let verdict = field_write_judgment(&model, "age", &integer_value(200.0), &kernel);
    assert!(verdict.is_none(), "an undeclared field writes with no judgment");
}

// --- pydantic-style class: Annotated[int, Field(ge=0, le=120)] field construction fire ---

#[test]
fn pydantic_style_annotated_field_construction_fires_over_ceiling() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "from typing import Annotated\n",
        "from pydantic import BaseModel, Field\n",
        "class Person(BaseModel):\n",
        "    age: Annotated[int, Field(ge=0, le=120)]\n",
    ));
    let aliases = crate::surface::compile_aliases(&module);
    let imports = crate::surface::surface_imports(&module);
    let table = class_table(&module, &aliases, &imports, &kernel);
    let person = table.get("Person").expect("Person class recorded");
    assert!(person.fields[0].declared.is_some(), "inline Annotated field reads its own set");
    let keyword = vec![("age".to_owned(), integer_value(200.0), range_of("200"))];
    let verdict = judge_construction(person, &[], &keyword, ConstructionKind::DirectCall, &kernel);
    assert_eq!(verdict.fires.len(), 1);
    assert!(verdict.fires[0].1.contains("'200'"), "{}", verdict.fires[0].1);
}
