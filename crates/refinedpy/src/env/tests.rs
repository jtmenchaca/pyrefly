use std::collections::HashSet;

use refined_domain::abstract_value::opaque_value;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_parser::parse_module;

use super::*;

/// Parses `source` as a module and returns its single top-level
/// `def` — the same helper `summaries.rs`'s own tests use, repeated
/// here since this module's tests need it too and the two files
/// stay independent per the mission's file-ownership convention.
fn parsed_def(source: &str) -> StmtFunctionDef {
    let module = parse_module(source).expect("fixture source parses").into_syntax();
    let stmt = module.body.into_iter().next().expect("one top-level statement");
    stmt.function_def_stmt().expect("top-level statement is a def")
}

fn bare_retained_callable() -> RetainedCallable {
    let def = parsed_def("def f(x):\n    return x\n");
    RetainedCallable::from_def(&def, HashMap::new())
}

/// A key `record_retained_callable` wrote is readable back through
/// `retained_callable` on the SAME environment.
#[test]
fn test_record_and_read_retained_callable_round_trips() {
    let mut environment = Environment::new(HashSet::new());
    let key = environment.next_retained_callable_key();
    environment.record_retained_callable(key, bare_retained_callable());
    assert!(environment.retained_callable(key).is_some());
    assert!(environment.retained_callable(key + 1).is_none());
}

/// `fork` shares the SAME underlying table (never a copy): a write
/// made through the forked environment is visible back through the
/// original — the property `summaries::fresh_body_environment`'s
/// own `inherit_retained_callables` call depends on to let a called
/// function's own retained value outlive its own call frame.
#[test]
fn test_fork_shares_the_retained_callable_table() {
    let original = Environment::new(HashSet::new());
    let mut forked = original.fork();
    let key = forked.next_retained_callable_key();
    forked.record_retained_callable(key, bare_retained_callable());
    assert!(original.retained_callable(key).is_some());
}

/// `join` carries `a`'s own retained-callable table forward — since
/// both arms of a join were forked from the same parent (`fork`'s
/// own doc), a key either arm wrote is visible in the joined
/// environment.
#[test]
fn test_join_keeps_the_retained_callable_table() {
    let parent = Environment::new(HashSet::new());
    let mut arm_a = parent.fork();
    let arm_b = parent.fork();
    let key = arm_a.next_retained_callable_key();
    arm_a.record_retained_callable(key, bare_retained_callable());
    let joined = Environment::join(arm_a, &arm_b);
    assert!(joined.retained_callable(key).is_some());
}

/// `inherit_retained_callables` replaces this environment's own
/// table with `enclosing`'s SAME `Arc` — a key this environment
/// later records is then visible from `enclosing` too, the exact
/// property a called function's own interpretation environment
/// needs so a def/lambda IT creates survives past its own call
/// frame back into the caller.
#[test]
fn test_inherit_retained_callables_shares_writes_both_ways() {
    let enclosing = Environment::new(HashSet::new());
    let mut callee = Environment::new(HashSet::new());
    callee.inherit_retained_callables(&enclosing);
    let key = callee.next_retained_callable_key();
    callee.record_retained_callable(key, bare_retained_callable());
    assert!(enclosing.retained_callable(key).is_some());
}

/// `next_retained_callable_key` never repeats within one shared
/// counter, even across environments that inherited it from each
/// other — the property that keeps two creations of the same
/// lambda/def text (two calls to the same enclosing function) from
/// landing in the same table slot.
#[test]
fn test_next_retained_callable_key_never_repeats() {
    let environment = Environment::new(HashSet::new());
    let first = environment.next_retained_callable_key();
    let second = environment.next_retained_callable_key();
    assert_ne!(first, second);
}

/// `record_lambda_key`/`lambda_key` round-trip, and a SECOND
/// registration of the same range overwrites the mapping to the
/// newer key — the property `expressions.rs::register_retained_
/// callables` depends on so a lambda re-created with a different
/// closure (`make_adder(1)` then `make_adder(100)`) is read back
/// under its OWN creation's key, never a stale earlier one.
#[test]
fn test_record_lambda_key_overwrites_on_a_later_creation() {
    let mut environment = Environment::new(HashSet::new());
    let range_start = 42u32;
    let first_key = environment.next_retained_callable_key();
    environment.record_lambda_key(range_start, first_key);
    assert_eq!(environment.lambda_key(range_start), Some(first_key));
    let second_key = environment.next_retained_callable_key();
    environment.record_lambda_key(range_start, second_key);
    assert_eq!(environment.lambda_key(range_start), Some(second_key));
}

/// `retained_callable_value`/`retained_callable_key` round-trip:
/// building a value from a key and reading the key back off it
/// answers the same key, and an ordinary opaque lambda value (no
/// retained body) reads back `None`.
#[test]
fn test_retained_callable_value_key_round_trip() {
    let value = retained_callable_value(7);
    assert_eq!(retained_callable_key(&value), Some(7));
    let plain = opaque_value(FUNCTION_VALUE_WORD);
    assert_eq!(retained_callable_key(&plain), None);
}

/// `same_module_def_alias_value`/`same_module_def_alias_name`
/// round-trip: building a value from a def's own name and reading
/// the name back off it answers the same name, and an ordinary
/// opaque lambda value (empty `source`) reads back `None`.
#[test]
fn test_same_module_def_alias_value_name_round_trip() {
    let value = same_module_def_alias_value("identity");
    assert_eq!(same_module_def_alias_name(&value), Some("identity"));
    let plain = opaque_value(FUNCTION_VALUE_WORD);
    assert_eq!(same_module_def_alias_name(&plain), None);
}

/// The two `source` encodings never collide: a retained-callable
/// value's numeric key never reads back as a def-alias name (Python
/// identifiers cannot start with a digit, so no real def name is
/// ever numeric), and a def-alias value's own name never parses as
/// a retained-callable key.
#[test]
fn test_retained_callable_and_def_alias_encodings_never_collide() {
    let retained = retained_callable_value(7);
    assert_eq!(same_module_def_alias_name(&retained), None);
    let aliased = same_module_def_alias_value("identity");
    assert_eq!(retained_callable_key(&aliased), None);
}

// ── access-path bindings ────────────────────────────────────────

fn dummy_value() -> AbstractValue {
    refined_domain::abstract_value::known_values(
        vec![40.0],
        refined_domain::abstract_value::PrimitiveKind::Integer,
        refined_domain::trust_grades::TrustProved,
    )
}

/// `a.n.x` reads through `tracked_place_of` the same way
/// `A15.guard.eq`/`A15.guard.ne`'s own `a.n` construct does, one
/// segment shallower — the recursive `Expr::Attribute` reading walks
/// down to the base `Expr::Name` and builds the path back up in
/// order.
#[test]
fn test_tracked_place_of_reads_a_multi_segment_attribute_chain() {
    let parsed = ruff_python_parser::parse_expression("a.n.x").expect("test source must parse");
    let place = tracked_place_of(&parsed.into_expr()).expect("a.n.x is a readable attribute chain");
    assert_eq!(place.binding, "a");
    assert_eq!(place.path, vec!["n".to_owned(), "x".to_owned()]);
}

/// A bare name reads as a `TrackedPlace` with an empty path — the
/// same "no segments" shape the Go type's own `Path: nil` gives a
/// plain identifier.
#[test]
fn test_tracked_place_of_a_bare_name_has_an_empty_path() {
    let parsed = ruff_python_parser::parse_expression("a").expect("test source must parse");
    let place = tracked_place_of(&parsed.into_expr()).expect("a bare name is a readable place");
    assert_eq!(place.binding, "a");
    assert!(place.path.is_empty());
}

/// A call, a subscript, or any other root names no place at all —
/// the checker cannot say the chain survives past a shape this
/// reader does not recognize.
#[test]
fn test_tracked_place_of_declines_a_non_attribute_root() {
    let parsed = ruff_python_parser::parse_expression("f().n").expect("test source must parse");
    assert!(tracked_place_of(&parsed.into_expr()).is_none());
}

/// `bind_path`/`read_path` round-trip a fact recorded at a path.
#[test]
fn test_bind_and_read_path_round_trips() {
    let mut environment = Environment::new(HashSet::new());
    let place = TrackedPlace::bare("a").extend("n");
    environment.bind_path(&place, dummy_value());
    assert!(environment.read_path(&place).is_some());
}

/// `TrackedPlace::extends` — the containment test `forget_path_base`
/// relies on: a path extends itself, extends a shorter prefix of the
/// same binding, and does NOT extend a sibling path or a different
/// binding's path entirely.
#[test]
fn test_tracked_place_extends_prefixes_of_the_same_binding_only() {
    let a = TrackedPlace::bare("a");
    let a_n = a.extend("n");
    let a_n_x = a_n.extend("x");
    let a_m = a.extend("m");
    let b_n = TrackedPlace::bare("b").extend("n");
    assert!(a_n.extends(&a_n), "a place extends itself");
    assert!(a_n_x.extends(&a_n), "a.n.x continues the shorter prefix a.n");
    assert!(!a_n.extends(&a_n_x), "a.n does not continue the LONGER path a.n.x");
    assert!(!a_m.extends(&a_n), "a.m is a sibling of a.n, not a continuation");
    assert!(!b_n.extends(&a_n), "a different binding's path never extends this one");
}

/// `Environment::forget` on the base name drops every path fact
/// rooted at it — `forget`'s own doc states the rule this pins.
#[test]
fn test_forget_drops_every_path_fact_rooted_at_the_base_name() {
    let mut environment = Environment::new(HashSet::new());
    let place = TrackedPlace::bare("a").extend("n");
    environment.bind_path(&place, dummy_value());
    environment.forget("a");
    assert!(environment.read_path(&place).is_none());
}

/// `forget_path_base` on a deeper prefix drops a continuation but
/// leaves an unrelated sibling standing — the same distinction
/// `test_tracked_place_extends_prefixes_of_the_same_binding_only`
/// pins for `extends` itself, exercised here through the actual
/// forget call.
#[test]
fn test_forget_path_base_drops_continuations_not_siblings() {
    let mut environment = Environment::new(HashSet::new());
    let a_n = TrackedPlace::bare("a").extend("n");
    let a_n_x = a_n.extend("x");
    let a_m = TrackedPlace::bare("a").extend("m");
    environment.bind_path(&a_n_x, dummy_value());
    environment.bind_path(&a_m, dummy_value());
    environment.forget_path_base(&a_n);
    assert!(environment.read_path(&a_n_x).is_none(), "a.n.x continues the written prefix a.n");
    assert!(environment.read_path(&a_m).is_some(), "a.m is unrelated to a.n");
}

/// `fork` shares no mutable state for path bindings (unlike the
/// retained-callable table) — a write through the forked
/// environment's OWN path map must not reach the original, matching
/// `bindings`' own by-value clone semantics.
#[test]
fn test_fork_clones_path_bindings_independently() {
    let original = Environment::new(HashSet::new());
    let mut forked = original.fork();
    let place = TrackedPlace::bare("a").extend("n");
    forked.bind_path(&place, dummy_value());
    assert!(forked.read_path(&place).is_some());
    assert!(original.read_path(&place).is_none(), "fork must not share the path map by reference");
}

/// `join` keeps a path fact only when BOTH arms still hold one for
/// the identical place — the same rule `bindings`' own join already
/// follows.
#[test]
fn test_join_keeps_a_path_fact_only_when_both_arms_hold_it() {
    let parent = Environment::new(HashSet::new());
    let mut arm_a = parent.fork();
    let mut arm_b = parent.fork();
    let shared = TrackedPlace::bare("a").extend("n");
    let only_a = TrackedPlace::bare("a").extend("m");
    arm_a.bind_path(&shared, dummy_value());
    arm_b.bind_path(&shared, dummy_value());
    arm_a.bind_path(&only_a, dummy_value());
    let joined = Environment::join(arm_a, &arm_b);
    assert!(joined.read_path(&shared).is_some(), "both arms held a fact about the shared place");
    assert!(joined.read_path(&only_a).is_none(), "only one arm held a fact about this place");
}
