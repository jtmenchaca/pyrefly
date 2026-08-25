use super::*;

#[test]
fn test_greater_than_literal_keeps_satisfying_drops_others() {
    let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
    let Some(narrowed) = assumed("x > 100", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.values, vec![200.0]);
}

#[test]
fn test_greater_than_literal_falsity_flips_the_kept_side() {
    let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
    let Some(narrowed) = assumed("x > 100", environment, false) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.values, vec![40.0]);
}

#[test]
fn test_chained_comparison_keeps_the_middle_window() {
    let environment = environment_with("x", vec![-5.0, 0.0, 60.0, 120.0, 200.0], PrimitiveKind::Number);
    let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.values, vec![0.0, 60.0, 120.0]);
}

#[test]
fn test_equality_against_literal_keeps_only_that_value() {
    let environment = environment_with("x", vec![40.0, 41.0], PrimitiveKind::Number);
    let Some(narrowed) = assumed("x == 40", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.values, vec![40.0]);
}

/// `guard_narrowed_values`'s own pin — a match arm's guard read as a
/// narrowing through the SAME `assume` machinery
/// `test_equality_against_literal_keeps_only_that_value` above
/// exercises directly, but through the sandbox-and-read-back path
/// `match_arms.rs`'s guarded bare-capture split calls: `x == 1` over
/// `{1, 2, 4}` narrows to exactly `{1}` on the admitted (`truth:
/// true`) side.
#[test]
fn test_guard_narrowed_values_keeps_the_admitted_side() {
    let Some(kernel) = loaded_kernel() else { return };
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let parsed = parse_expression("x == 1").expect("test source must parse");
    let narrowed = guard_narrowed_values(&parsed.into_expr(), "x", &subject, &kernel, true)
        .expect("a single equality comparison is a guard shape this reader proves");
    assert_eq!(narrowed.values, vec![1.0]);
}

/// The excluded (`truth: false`) side of the same guard: `x == 1`
/// being false over `{1, 2, 4}` leaves exactly `{2, 4}`.
#[test]
fn test_guard_narrowed_values_keeps_the_excluded_side() {
    let Some(kernel) = loaded_kernel() else { return };
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let parsed = parse_expression("x == 1").expect("test source must parse");
    let narrowed = guard_narrowed_values(&parsed.into_expr(), "x", &subject, &kernel, false)
        .expect("a single equality comparison is a guard shape this reader proves");
    let mut values = narrowed.values.clone();
    values.sort_by(f64::total_cmp);
    assert_eq!(values, vec![2.0, 4.0]);
}

/// A guard shape `assume` narrows nothing for (`x.bit_length() > 0` —
/// a method call on the guard's own subject, which none of this
/// file's comparison, membership, or type-guard leaves recognize)
/// leaves the binding UNCHANGED, so `guard_narrowed_values` declines
/// outright — an unchanged binding is never read as a proof every
/// member survives; it is the absence of a proof.
#[test]
fn test_guard_narrowed_values_declines_when_assume_narrows_nothing() {
    let Some(kernel) = loaded_kernel() else { return };
    let subject = known_values(vec![1.0, 2.0, 4.0], PrimitiveKind::Integer, TrustProved);
    let parsed = parse_expression("x.bit_length() > 0").expect("test source must parse");
    let narrowed = guard_narrowed_values(&parsed.into_expr(), "x", &subject, &kernel, true);
    assert!(
        narrowed.is_none(),
        "an unproved guard shape leaves the binding unchanged — never read as a genuine narrowing"
    );
}

#[test]
fn test_not_wrapped_comparison_flips_truth() {
    let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
    let Some(narrowed) = assumed("not (x > 100)", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.values, vec![40.0]);
}

#[test]
fn test_and_narrows_both_names() {
    let mut locally_bound = HashSet::new();
    locally_bound.insert("a".to_owned());
    locally_bound.insert("b".to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind("a", known_values(vec![-1.0, 5.0], PrimitiveKind::Number, TrustProved));
    environment.bind("b", known_values(vec![-2.0, 7.0], PrimitiveKind::Number, TrustProved));
    let Some(narrowed) = assumed("a > 0 and b > 0", environment, true) else {
        return;
    };
    let a = narrowed.read("a").expect("a still bound");
    let b = narrowed.read("b").expect("b still bound");
    assert_eq!(a.values, vec![5.0]);
    assert_eq!(b.values, vec![7.0]);
}

#[test]
fn test_non_values_binding_untouched() {
    use refined_domain::abstract_value::null_value;
    let mut locally_bound = HashSet::new();
    locally_bound.insert("x".to_owned());
    let mut environment = Environment::new(locally_bound);
    environment.bind("x", null_value());
    let Some(narrowed) = assumed("x > 100", environment, true) else {
        return;
    };
    let x = narrowed.read("x").expect("x still bound");
    assert_eq!(x.kind, Kind::Null);
}

#[test]
fn test_unbound_name_untouched() {
    let environment = Environment::new(HashSet::new());
    let Some(narrowed) = assumed("x > 100", environment, true) else {
        return;
    };
    assert!(narrowed.read("x").is_none());
}
