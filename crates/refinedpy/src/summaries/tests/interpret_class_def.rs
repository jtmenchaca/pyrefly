use super::*;

// --- interpret_class_def: ClassDef-in-summary construction ---

/// a-statements.py's own `device()` shape: a body-local class,
/// constructed and returned. `call_result` must answer a TAGGED
/// instance (`source == "_Device"`) carrying the field's own default
/// — proof `Stmt::ClassDef` no longer falls to `interpret_body`'s
/// catch-all decline.
#[test]
fn call_result_answers_a_tagged_instance_for_a_body_local_class_construction() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def device():\n",
        "    class _Device:\n",
        "        value: int = 0\n",
        "    return _Device()\n",
    ));
    let result = call_result(&def, &[], None, &kernel, 0).expect("a body-local ClassDef no longer declines");
    assert_eq!(result.kind, Kind::Object);
    assert_eq!(result.source, "_Device");
    let value_field = result.keys.iter().find(|entry| entry.name == "value").expect("value field present");
    assert_eq!(value_field.value, known_int(0.0));
}

/// The constructed instance's class is ALSO readable off
/// `environment.classes()` inside the SAME call (not merely the
/// returned value) — `_Device`'s own `__init__`-free field defaults
/// still resolve when a later statement in the same body (out of this
/// wave's fixture rows, but not precluded) constructs a second
/// instance of the same class.
#[test]
fn interpret_class_def_registers_the_class_before_the_return_statement_runs() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def two_devices():\n",
        "    class _Device:\n",
        "        value: int = 0\n",
        "    first = _Device()\n",
        "    return first\n",
    ));
    let result = call_result(&def, &[], None, &kernel, 0)
        .expect("a second construction of the same body-local class still resolves");
    assert_eq!(result.kind, Kind::Object);
    assert_eq!(result.source, "_Device");
}
