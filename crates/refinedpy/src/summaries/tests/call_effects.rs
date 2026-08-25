use super::*;

// --- call_effects: the CALLEE-EFFECTS CHANNEL ---

/// a-statements.py's own `nonlocal_rebind`/`spoil`: `nonlocal age` then
/// `age = 200` — the effect list must carry `("age", 200)`, the
/// ENCLOSING name's own new value, not merely `spoil`'s own (Null)
/// return.
#[test]
fn call_effects_reports_a_nonlocal_declared_write() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def spoil():\n    nonlocal age\n    age = 200\n");
    let mut enclosing = Environment::new(std::collections::HashSet::from(["age".to_owned()]));
    enclosing.bind("age", known_int(10.0));

    let (_value, effects) =
        call_effects(&def, &[], None, &kernel, 0, &enclosing).expect("a nonlocal write is a readable effect");
    assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
    assert_eq!(effects[0].0, "age");
    assert_eq!(effects[0].1, known_int(200.0));
}

/// a-statements.py's own `closure_mutates_flattened_capture`/`spoil`:
/// `outlaw["age"] = 200` — a mutation THROUGH a captured free name,
/// with no `nonlocal` declaration at all (CPython never requires one
/// for a subscript/attribute STORE, only for rebinding the name
/// itself). The effect is the WRITTEN-THROUGH dict, keyed on `outlaw`.
#[test]
fn call_effects_reports_a_captured_receiver_subscript_mutation() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def spoil():\n    outlaw[\"age\"] = 200\n");
    let mut enclosing = Environment::new(std::collections::HashSet::from(["outlaw".to_owned()]));
    let dict_value = refined_domain::known_constructors::known_object(
        vec![refined_domain::abstract_value::ObjectKey {
            name: "age".to_owned(),
            numeric: false,
            value: known_int(40.0),
        }],
        None,
        true,
        TrustProved,
        false,
    );
    enclosing.bind("outlaw", dict_value);

    let (_value, effects) =
        call_effects(&def, &[], None, &kernel, 0, &enclosing).expect("a captured-receiver mutation is readable");
    assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
    assert_eq!(effects[0].0, "outlaw");
    assert_eq!(effects[0].1.kind, Kind::Object);
    let written = effects[0].1.keys.iter().find(|entry| entry.name == "age").expect("age entry survives the write");
    assert_eq!(written.value, known_int(200.0));
}

/// A body with no `nonlocal` declaration and no captured-receiver
/// mutation — an ordinary local write — reports an EMPTY effect list;
/// `call_effects` never invents an effect for a purely local rebind
/// (Python's own scoping rule: a plain `Assign` target with no
/// `nonlocal` always creates a fresh local, never writes outward).
#[test]
fn call_effects_reports_no_effects_for_a_purely_local_write() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def bump():\n    age = 15\n    return age\n");
    let enclosing = Environment::new(std::collections::HashSet::new());
    let (value, effects) =
        call_effects(&def, &[], None, &kernel, 0, &enclosing).expect("a purely local write still answers");
    assert_eq!(value, known_int(15.0));
    assert!(effects.is_empty(), "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
}

/// A captured-receiver store this channel CANNOT compose (the free
/// name's current value is a scalar Integer, not a dict/list —
/// `dict_with_item`/`list_with_item` both answer `None` for it)
/// answers an effect whose VALUE is `unknown()` — the caller MUST
/// forget the name rather than keep its stale pre-call value
/// (`call_effects`'s own doc: "a store you cannot compose answers
/// that name unknown() so the caller FORGETS it — an effect is never
/// silently dropped"). Exercised directly against `record_write_
/// effect` (the law's own owning function) rather than through the
/// full `call_effects` pipeline: `interpret_body`'s own subscript-
/// write recognition (`write_subscript_target`, a sibling law added
/// this same wave) reads the identical seeded free-name value and
/// therefore ALREADY declines this exact body shape at the VALUE
/// pass, before `call_effects`'s own second pass ever runs — so this
/// unknown()-forget answer is not reachable through `call_effects`'s
/// public surface on TODAY's fixture rows, but is real defensive
/// code for a store shape the value pass might one day recognize
/// more narrowly than the effects pass does; testing it directly
/// keeps the law honest without asserting a false end-to-end claim.
#[test]
fn record_write_effect_answers_unknown_for_an_uncomposable_captured_receiver_store() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("outlaw[\"age\"] = 200\n")
        .expect("statement source parses")
        .into_syntax();
    let Stmt::Assign(assign) = module.body.into_iter().next().expect("one statement") else {
        panic!("expected an Assign statement");
    };
    let mut environment = Environment::new(std::collections::HashSet::new());
    environment.bind("outlaw", known_int(999.0));
    let nonlocal_names = std::collections::HashSet::new();
    let locally_bound = std::collections::HashSet::new();
    let mut effects: Vec<(String, AbstractValue)> = Vec::new();
    let [target] = assign.targets.as_slice() else { panic!("one target") };
    record_write_effect(target, assign.value.as_ref(), &kernel, &mut environment, &nonlocal_names, &locally_bound, &mut effects);
    assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
    assert_eq!(effects[0].0, "outlaw");
    assert_eq!(effects[0].1.kind, Kind::Unknown, "an uncomposable store forgets, never keeps a stale value");
}

/// A captured-receiver store on a free name never bound at all — the
/// same `unknown()`-forgets answer, for the OTHER uncomposable shape
/// (no current value to compose against, rather than a wrong-shaped
/// one). Same direct-against-`record_write_effect` posture as above.
#[test]
fn record_write_effect_answers_unknown_for_a_store_through_a_never_bound_free_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = ruff_python_parser::parse_module("outlaw[\"age\"] = 200\n")
        .expect("statement source parses")
        .into_syntax();
    let Stmt::Assign(assign) = module.body.into_iter().next().expect("one statement") else {
        panic!("expected an Assign statement");
    };
    let mut environment = Environment::new(std::collections::HashSet::new());
    let nonlocal_names = std::collections::HashSet::new();
    let locally_bound = std::collections::HashSet::new();
    let mut effects: Vec<(String, AbstractValue)> = Vec::new();
    let [target] = assign.targets.as_slice() else { panic!("one target") };
    record_write_effect(target, assign.value.as_ref(), &kernel, &mut environment, &nonlocal_names, &locally_bound, &mut effects);
    assert_eq!(effects.len(), 1, "{:?}", effects.iter().map(|(name, _)| name).collect::<Vec<_>>());
    assert_eq!(effects[0].0, "outlaw");
    assert_eq!(effects[0].1.kind, Kind::Unknown);
}
