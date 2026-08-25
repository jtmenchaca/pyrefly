use super::*;

// --- call_result_with_enclosing: closure reads ---

/// `def read_age(): return age` nested inside a body that bound
/// `age` — a-statements.py's own closure-read shape
/// (`closure_mutates_flattened_capture`'s cousin, minus the write):
/// `age` is free in `read_age`'s own body, so `call_result` alone
/// (no enclosing environment) declines it as an unbound name read
/// (`unknown()`, which `interpret_body`'s `Return` arm rejects);
/// `call_result_with_enclosing` answers it once the call site's
/// environment is threaded through.
#[test]
fn call_result_with_enclosing_reads_a_free_enclosing_local() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def read_age():\n    return age\n");

    let mut enclosing = Environment::new(std::collections::HashSet::from(["age".to_owned()]));
    enclosing.bind("age", known_int(40.0));

    assert!(
        call_result(&def, &[], None, &kernel, 0).is_none(),
        "with no enclosing environment, the free read of `age` stays unbound"
    );
    let result = call_result_with_enclosing(&def, &[], None, &kernel, 0, Some(&enclosing))
        .expect("the enclosing environment's `age` binding answers the free read");
    assert_eq!(result, known_int(40.0));
}

/// A name the callee body ITSELF binds (a parameter, or an
/// assignment target) is never seeded from `enclosing`, even when
/// `enclosing` happens to bind the same name — ordinary Python
/// scoping (the body's own binding shadows the enclosing one for
/// its whole extent, `executionmodel.rst`'s "Naming and binding").
#[test]
fn call_result_with_enclosing_does_not_shadow_a_locally_bound_name() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def shadow():\n    age = 10\n    return age\n");

    let mut enclosing = Environment::new(std::collections::HashSet::from(["age".to_owned()]));
    enclosing.bind("age", known_int(999.0));

    let result = call_result_with_enclosing(&def, &[], None, &kernel, 0, Some(&enclosing))
        .expect("the body's own local binding answers the read");
    assert_eq!(result, known_int(10.0), "the callee's own `age = 10` wins, never the enclosing 999");
}

/// a-statements.py's own `global_rebind`/`bump`: `global _module_age`
/// then `_module_age = 15` then `return _module_age` — the `global`
/// declaration must not decline the whole call the way an unrecognized
/// statement would. This interpreter tracks no scope chain, so the
/// write and the read both land in the SAME flat environment; the
/// declaration itself is a no-op, exactly like `Stmt::Nonlocal`.
#[test]
fn interpret_body_reaches_past_a_global_declaration() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def("def bump():\n    global _module_age\n    _module_age = 15\n    return _module_age\n");

    let result = call_result(&def, &[], None, &kernel, 0)
        .expect("the `global` declaration is a no-op; the following write/read resolve normally");
    assert_eq!(result, known_int(15.0));
}

/// e-class-and-function.py's own `pick_years` shape: `if
/// isinstance(value, int): return value` with no `else`, followed by
/// `return len(value)` as the NEXT top-level statement. Calling with
/// a concrete int argument (200) takes the isinstance-true arm; the
/// FALSE arm's own fallthrough narrows `value` to the empty set (200
/// really is an int), so `return len(value)` — unmodeled on a
/// non-string `Kind::Values` — is dead code for this call and must
/// never run. Before `interpret_undecided_arms`/the fallthrough
/// branch recognized that dead arm as unreachable, walking it anyway
/// let `len`'s own decline sink the WHOLE call to `None`, even
/// though the arm actually taken answers cleanly.
#[test]
fn call_result_skips_a_fallthrough_arm_narrowing_proves_unreachable() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def pick_years(value):\n",
        "    if isinstance(value, int):\n",
        "        return value\n",
        "    return len(value)\n",
    ));
    let result = call_result(&def, &[known_int(200.0)], None, &kernel, 0)
        .expect("the isinstance-true arm answers the call; the dead len(value) arm must not decline it");
    assert_eq!(result, known_int(200.0));
}

/// The same shape's OTHER branch: an explicit `elif`/second arm
/// (rather than an implicit fallthrough) that is itself narrowed
/// infeasible must also be skipped rather than interpreted — pins
/// `interpret_undecided_arms`'s own per-arm infeasibility check
/// (`narrowing::arm_is_infeasible`), not just the fallthrough one.
#[test]
fn call_result_skips_an_explicit_elif_arm_narrowing_proves_unreachable() {
    let Some(kernel) = loaded_kernel() else { return };
    let def = parsed_def(concat!(
        "def pick_years(value):\n",
        "    if isinstance(value, int):\n",
        "        return value\n",
        "    else:\n",
        "        return len(value)\n",
    ));
    let result = call_result(&def, &[known_int(200.0)], None, &kernel, 0)
        .expect("the isinstance-true arm answers the call; the dead else arm must not decline it");
    assert_eq!(result, known_int(200.0));
}
