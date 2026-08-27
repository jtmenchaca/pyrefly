use super::*;

// --- dict[str, X]'s value-slot reading ---

/// `dict[str, Age]` — a-statements.py's `return_dict_members` own
/// shape: the outer declaration carries no set of its own (`element`
/// Some, `set` empty) and the element is `Age` read through the
/// ordinary alias recursion.
#[test]
fn dict_of_str_to_age_reads_age_as_the_element() {
    let module = ruff_python_parser::parse_module(
        "x: dict[str, Age] = {}\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("dict[str, Age] resolves");
    assert!(!got.admits_none);
    assert_eq!(got.spelling, "dict[str, Age]");
    let element = got.element.expect("dict[str, Age] carries an element refinement");
    assert_eq!(element.spelling, "Age");
    assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
}

/// `dict[str, Age] | None` — composes with the existing
/// `admits_none` machinery for free: the union arm recurses into
/// this same dict read, then marks `admits_none` true, without
/// touching `element`.
#[test]
fn dict_of_str_to_age_or_none_reads_the_element_with_admits_none_true() {
    let module = ruff_python_parser::parse_module(
        "x: dict[str, Age] | None = None\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("dict[str, Age] | None resolves");
    assert!(got.admits_none);
    assert_eq!(got.spelling, "dict[str, Age]");
    let element = got.element.expect("dict[str, Age] | None still carries an element refinement");
    assert_eq!(element.spelling, "Age");
}

/// `dict[int, Age]` — a non-`str` key reads the same way a `str` key
/// does: the key's sort does not change what the VALUES hold, and the
/// value refinement is what the read carries. An earlier shape declined
/// every non-`str` key, which left `dict[int, X]` and `dict[object, X]`
/// parameters with no value at all.
#[test]
fn dict_of_int_to_age_reads_its_value_refinement() {
    let module = ruff_python_parser::parse_module(
        "x: dict[int, Age] = {}\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("dict[int, Age] resolves");
    let element = got.element.expect("dict[int, Age] carries its value refinement");
    assert_eq!(element.spelling, "Age");
}

/// `dict[str, Unreadable]` — a value type this table cannot read
/// (no alias by that name) declines the whole subscript.
#[test]
fn dict_of_str_to_an_unreadable_value_type_declines() {
    let module = ruff_python_parser::parse_module(
        "x: dict[str, Unreadable] = {}\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment);
    assert!(got.is_none());
}

// --- weakref.WeakKeyDictionary[K, V] / WeakValueDictionary[K, V]'s
// value-slot reading (A8.xfer.weak's own `guarded_weak_read` shape) ---

/// `weakref.WeakKeyDictionary[_Key, Age]` reads its VALUE slot (argument
/// 2 of 2) exactly the way `dict[str, Age]` reads its own value slot —
/// the "weak" half is a lifetime fact about the KEY, invisible to this
/// table, which states only what a present key's value reads back as.
/// The spelling reuses the plain dict arm's own `"dict[object, X]"`
/// shape so every reader keyed on the `"dict["` prefix rides this
/// annotation with no separate case of its own.
#[test]
fn weak_key_dictionary_reads_age_as_the_value_slot() {
    let module = ruff_python_parser::parse_module(
        "x: weakref.WeakKeyDictionary[_Key, Age] = weakref.WeakKeyDictionary()\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("weakref.WeakKeyDictionary[_Key, Age] resolves");
    assert!(!got.admits_none);
    assert_eq!(got.spelling, "dict[object, Age]");
    let element = got.element.expect("WeakKeyDictionary carries an element refinement");
    assert_eq!(element.spelling, "Age");
    assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
}

/// `weakref.WeakValueDictionary[_Key, Age]` — the sibling class: its
/// VALUES are weak instead of its keys, which changes nothing about the
/// value slot's own type, so it reads identically.
#[test]
fn weak_value_dictionary_reads_age_as_the_value_slot() {
    let module = ruff_python_parser::parse_module(
        "x: weakref.WeakValueDictionary[_Key, Age] = weakref.WeakValueDictionary()\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("weakref.WeakValueDictionary[_Key, Age] resolves");
    assert_eq!(got.spelling, "dict[object, Age]");
    let element = got.element.expect("WeakValueDictionary carries an element refinement");
    assert_eq!(element.spelling, "Age");
}

/// A STRING annotation (`"weakref.WeakKeyDictionary[_Key, Age]"`, the
/// forward-reference spelling A8.xfer.weak's own `guarded_weak_read`
/// parameter uses) resolves through the ordinary string-literal arm —
/// parse the quoted contents, then recurse — landing on the identical
/// value slot the unquoted form reads.
#[test]
fn weak_key_dictionary_as_a_string_annotation_reads_the_same_value_slot() {
    let module = ruff_python_parser::parse_module(
        "x: \"weakref.WeakKeyDictionary[_Key, Age]\" = weakref.WeakKeyDictionary()\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let environment = no_locals();

    let got = declared_refinement(annotation, &aliases, &imports, &environment)
        .expect("the quoted forward reference resolves the same way");
    assert_eq!(got.spelling, "dict[object, Age]");
    let element = got.element.expect("the string form still carries an element refinement");
    assert_eq!(element.spelling, "Age");
}

/// A body that rebinds the name `weakref` to something else (shadowing
/// the module) must not have this arm fire on its annotation — the same
/// module-not-shadowed guard `attribute_call.rs`'s own bare-constructor
/// row already takes for the CALL side of `weakref.WeakKeyDictionary()`.
#[test]
fn a_shadowed_weakref_name_declines_the_weak_dict_arm() {
    let module = ruff_python_parser::parse_module(
        "x: weakref.WeakKeyDictionary[_Key, Age] = weakref.WeakKeyDictionary()\n",
    )
    .expect("test module parses")
    .into_syntax();
    let imports = crate::surface::surface_imports(&module);
    let annotation = annotated_or_none_annotation(&module);
    let aliases = age_aliases();
    let mut environment = no_locals();
    environment.bind("weakref", refined_domain::abstract_value::null_value());

    let got = declared_refinement(annotation, &aliases, &imports, &environment);
    assert!(got.is_none());
}
