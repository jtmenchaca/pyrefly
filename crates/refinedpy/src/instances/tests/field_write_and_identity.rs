use super::*;

// --- field_write: the source tag survives ---

#[test]
fn field_write_preserves_the_instances_source_tag() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model(
        "Aged",
        vec![ClassField { name: "age".to_owned(), declared: None, default: None }],
    );
    let verdict = judge_construction(&model, &[(integer_value(40.0), range_of("40"))], &[], &kernel);
    assert_eq!(verdict.instance.source, "Aged", "judge_construction tags the instance with the class name");
    let written = field_write(&verdict.instance, "age", integer_value(41.0)).expect("write must decide");
    assert_eq!(written.source, "Aged", "the source tag survives a field write");
    assert_eq!(field_read(&written, "age"), Some(integer_value(41.0)));
}

// --- judge_construction: instance_identity ---

/// Two separate construction calls of the SAME class each mint their
/// own `instance_identity` — a dict keyed by one instance must not
/// answer a lookup by the other (`collection_models::known_dict_key`'s
/// own identity arm reads this field to tell two `Holder()` calls
/// apart, the way `env.rs`'s `next_retained_callable_key` already
/// tells two lambda/def creations apart).
#[test]
fn judge_construction_mints_a_distinct_instance_identity_per_call() {
    let Some(kernel) = loaded_kernel() else { return };
    let model = bare_model("Holder", Vec::new());
    let first = judge_construction(&model, &[], &[], &kernel).instance;
    let second = judge_construction(&model, &[], &[], &kernel).instance;
    assert!(first.instance_identity.is_some(), "a constructed instance carries an identity");
    assert!(second.instance_identity.is_some(), "a constructed instance carries an identity");
    assert_ne!(
        first.instance_identity, second.instance_identity,
        "two separate Holder() calls must not mint the same identity"
    );
}
