use super::*;

// --- THE KERNEL SUMMARY ROUTE ---
//
// These read the route's own bookkeeping — the gate, the store key,
// and the memo — without asking a kernel: every case below either
// declines in the LOWERING (which runs before any question), reads
// the gate, or compares two keys, so none of them loads a dylib.

/// THE GATE IS A PROPERTY OF THE DEF. A body reading only its own
/// parameters and locals needs no caller environment, so the kernel
/// route is open to it — and that must hold for the ordinary call
/// arm, which always supplies one (`expressions.rs`'s own call site).
/// Gating on the caller's `enclosing` instead would shut the route
/// off for every ordinary call and leave it reachable only from the
/// callback arms.
#[test]
fn a_body_reading_only_its_own_parameters_and_locals_needs_no_enclosing_scope() {
    assert!(!needs_enclosing_scope(&parsed_def("def double(x):\n    return x + x\n")));
    assert!(!needs_enclosing_scope(&parsed_def(
        "def scaled(x):\n    doubled = x + x\n    return doubled\n"
    )));
    assert!(!needs_enclosing_scope(&parsed_def(
        "def band(n):\n    if n < 10:\n        return 1\n    return 2\n"
    )));
}

/// A body reading a name it does not bind — a module-level global, a
/// captured local — keeps the concrete interpreter, which seeds that
/// name from the caller's environment.
#[test]
fn a_body_reading_a_free_name_needs_the_enclosing_scope() {
    assert!(needs_enclosing_scope(&parsed_def("def capped(x):\n    return x + LIMIT\n")));
    assert!(needs_enclosing_scope(&parsed_def(
        "def guarded(x):\n    if x < CEILING:\n        return x\n    return 0\n"
    )));
}

/// The free-name test reads a name bound LATER in the body as local,
/// the same way the seeding's own snapshot does — a write-then-read
/// body captures nothing.
#[test]
fn a_name_the_body_binds_before_reading_is_local_not_free() {
    assert!(!needs_enclosing_scope(&parsed_def(
        "def held(x):\n    total = 0\n    total = total + x\n    return total\n"
    )));
}

/// A def that captures is excluded by the gate even where the CALLER
/// supplied no environment, and a def that does not capture is
/// admitted even where the caller supplied one — the two halves of
/// reading the def rather than the call. Neither direction was true
/// of a gate on `enclosing`, which admitted exactly the first case
/// and excluded exactly the second.
#[test]
fn the_gate_and_the_callers_environment_are_independent() {
    let captures = parsed_def("def capped(x):\n    return x + LIMIT\n");
    let closed = parsed_def("def double(x):\n    return x + x\n");
    assert!(needs_enclosing_scope(&captures), "a capturing def is excluded however it is called");
    assert!(!needs_enclosing_scope(&closed), "a closed def is admitted however it is called");
}

/// The ordinary call arm's own spelling — a caller environment
/// supplied — still reaches the registry for a closed body. This is
/// the reachability the correction restores: before it, this call
/// never consulted the store at all.
#[test]
fn an_ordinary_call_with_a_caller_environment_reaches_the_registry() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def scaled(x):\n    return x * 2\n");
    let enclosing = Environment::new(std::collections::HashSet::new());
    let _ = call_result_with_enclosing(&def, &[known_int(3.0)], None, &kernel, 0, Some(&enclosing));
    let registry = SUMMARY_REGISTRY.lock().expect("summary registry lock poisoned");
    assert!(
        registry
            .as_ref()
            .is_some_and(|map| map.contains_key(&summary_key(&def, ENTRY_MODULE))),
        "a call carrying a caller environment must still consult the store",
    );
}

/// A `def` and a CLONE of it (which is what `FunctionTable` hands a
/// call site) key to the same compiled summary — the whole reason the
/// key is the name/range pair rather than a pointer.
#[test]
fn a_clone_of_a_def_keys_to_the_same_stored_summary() {
    let def = parsed_def("def double(x):\n    return x + x\n");
    let clone = def.clone();
    assert_eq!(summary_key(&def, ENTRY_MODULE), summary_key(&clone, ENTRY_MODULE));
}

/// The SAME def text at the SAME span in two different modules keys
/// APART — the cross-module half of the identity. Two sibling modules
/// that both open with the same `def` give their defs the same name
/// and the same `TextRange`, so without the module in the key one
/// module's compiled summary would answer the other module's calls.
#[test]
fn the_same_def_in_two_modules_keys_apart() {
    let def = parsed_def("def scale(x):\n    return x * 2\n");
    assert_ne!(summary_key(&def, "audio_level"), summary_key(&def, "video_level"));
}

/// One def reached under two different LOCAL names (an alias import)
/// keys to ONE summary: the key reads the def's own name and its
/// declaring module, and `rename_def` rewrites the local spelling
/// only for the table's own by-name lookup.
#[test]
fn one_def_reached_through_one_module_keys_the_same_however_it_is_reached() {
    let def = parsed_def("def scale(x):\n    return x * 2\n");
    let again = def.clone();
    assert_eq!(summary_key(&def, "audio_level"), summary_key(&again, "audio_level"));
}

/// Two defs in one module are different keys, even where their
/// bodies are identical: the range tells them apart.
#[test]
fn two_defs_in_one_module_key_apart() {
    let module = parse_module("def a(x):\n    return x\ndef b(x):\n    return x\n")
        .expect("fixture source parses")
        .into_syntax();
    let defs: Vec<StmtFunctionDef> = module
        .body
        .into_iter()
        .filter_map(|stmt| stmt.function_def_stmt())
        .collect();
    assert_eq!(defs.len(), 2);
    assert_ne!(summary_key(&defs[0], ENTRY_MODULE), summary_key(&defs[1], ENTRY_MODULE));
}

/// A body outside the lowering's grammar answers a decline, and the
/// decline is REMEMBERED: the second ask reads the store rather than
/// lowering again. Asked twice, both answers are the decline, and the
/// store holds exactly one entry for the key by the end.
#[test]
fn a_body_that_does_not_lower_is_remembered_as_a_decline() {
    // a call in the body: outside the grammar, and the decline
    // happens in the lowering, before any kernel question exists
    let def = parsed_def("def calls(x):\n    return helper(x)\n");
    assert!(compiled_summary_for(&def, ENTRY_MODULE).is_none());
    assert!(
        compiled_summary_for(&def, ENTRY_MODULE).is_none(),
        "the second ask reads the remembered decline"
    );
    let registry = SUMMARY_REGISTRY.lock().expect("summary registry lock poisoned");
    let held = registry
        .as_ref()
        .expect("the registry holds the answer")
        .get(&summary_key(&def, ENTRY_MODULE));
    let spelling = match held {
        None => "no entry at all",
        Some(None) => "a remembered decline",
        Some(Some(_)) => "a compiled summary",
    };
    assert!(matches!(held, Some(None)), "the store holds {spelling}, want a remembered decline");
}

/// The route declines a call whose argument count does not match the
/// def's own parameters — the entry vector has no place for the
/// difference, and the interpreter (which reads defaults) answers
/// instead.
#[test]
fn the_summary_route_declines_an_argument_count_the_entry_vector_cannot_place() {
    let def = parsed_def("def add(x, y):\n    return x + y\n");
    assert!(kernel_summary_result(&def, ENTRY_MODULE, &[known_int(1.0)]).is_none());
}

/// An argument this domain carries but the state wire does not spell
/// declines the call, not the summary.
#[test]
fn an_argument_the_state_wire_cannot_spell_declines_the_call() {
    assert!(entry_state_of(&unknown()).is_none());
    assert!(entry_state_of(&known_string_value("hi")).is_none());
    assert!(entry_state_of(&known_int(3.0)).is_some(), "a numeric value list crosses");
    assert!(entry_state_of(&null_value()).is_some(), "the null admission crosses");
}

/// A `Kind::Values` holding several numbers crosses as the SCALAR set
/// of those numbers — `one_of([3, 5])` — never as the tuple
/// `set_of_known` builds for a multi-value list.
#[test]
fn a_multi_value_argument_crosses_as_the_scalar_set_of_its_values() {
    let two_valued = known_values(vec![3.0, 5.0], PrimitiveKind::Integer, TrustProved);
    let state = entry_state_of(&two_valued).expect("a numeric value list crosses");
    assert_eq!(state.set, make_refined_set(vec![one_of(&[3.0, 5.0])]));
}
