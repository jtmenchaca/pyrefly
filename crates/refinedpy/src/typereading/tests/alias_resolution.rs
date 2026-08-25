use super::*;

#[test]
fn a_visible_alias_name_resolves_with_its_name_as_spelling() {
    let mut aliases = HashMap::new();
    aliases.insert(
        "PositiveInt".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(1.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment)
        .expect("a visible alias resolves");
    assert_eq!(got.spelling, "PositiveInt");
    assert_eq!(got.set, make_refined_set(vec![at_least(1.0)]));
}

/// `list[int]`'s element position resolves through the bare-sort
/// fallback: `int` is not a module-level alias, so without the
/// `base_sort_return_refinement` fallback at this one call site the
/// whole `list[int]` subscript declines (`f-type-nodes.py`'s
/// `list_annotation_parameter` row, undetermined before this fix).
#[test]
fn list_of_a_bare_int_resolves_its_element_through_the_base_sort_fallback() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("list[int]").expect("test source must parse");
    let annotation = parsed.into_expr();

    let got = declared_refinement(&annotation, &aliases, &imports, &environment)
        .expect("list[int]'s element must resolve through the base-sort fallback");
    assert_eq!(got.spelling, "list[int]");
    let element = got.element.expect("list[X] carries its element, not a scalar set");
    assert_eq!(element.spelling, "int");
    assert!(!element.set.forms.is_empty(), "int's own set must not be empty");
}

/// `set[str]` and `Sequence[float]` take the identical fallback path
/// — the same `is_element_container` arm, keyed only on the head
/// name.
#[test]
fn set_and_sequence_of_a_bare_base_sort_also_resolve_their_element() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();

    let set_parsed = parse_expression("set[str]").expect("test source must parse");
    let set_got = declared_refinement(&set_parsed.into_expr(), &aliases, &imports, &environment)
        .expect("set[str]'s element must resolve");
    assert_eq!(set_got.element.expect("element present").spelling, "str");

    let sequence_parsed = parse_expression("Sequence[float]").expect("test source must parse");
    let sequence_got = declared_refinement(&sequence_parsed.into_expr(), &aliases, &imports, &environment)
        .expect("Sequence[float]'s element must resolve");
    assert_eq!(sequence_got.element.expect("element present").spelling, "float");
}

/// `tuple[int, int]` — a FIXED-ARITY tuple of two bare base sorts:
/// each position reads through the same base-sort fallback the
/// element-container arm above takes, kept SEPARATE per position
/// (unlike `list[int]`'s one shared element) — `c-reads-and-values.py`'s
/// `ternary_spread_copies_optional_list` own parameter shape.
#[test]
fn fixed_arity_tuple_of_two_bare_ints_resolves_each_position_through_the_base_sort_fallback() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("tuple[int, int]").expect("test source must parse");

    let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
        .expect("tuple[int, int]'s positions must resolve through the base-sort fallback");
    assert_eq!(got.spelling, "tuple[int, int]");
    let positions = got.positions.expect("a fixed-arity tuple carries its positions, not a scalar set");
    assert_eq!(positions.len(), 2);
    assert_eq!(positions[0].spelling, "int");
    assert_eq!(positions[1].spelling, "int");
}

/// `tuple[Age, Label]` — mixed alias positions each read through the
/// ordinary alias recursion, keeping their own distinct sets.
#[test]
fn fixed_arity_tuple_of_two_aliases_resolves_each_positions_own_set() {
    let aliases = age_aliases();
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("tuple[Age, Label]").expect("test source must parse");

    let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
        .expect("tuple[Age, Label]'s positions must resolve");
    let positions = got.positions.expect("positions present");
    assert_eq!(positions[0].spelling, "Age");
    assert_eq!(positions[0].set, make_refined_set(vec![at_least(0.0)]));
    assert_eq!(positions[1].spelling, "Label");
    assert_eq!(positions[1].set, make_refined_set(vec![at_least(1.0)]));
}

/// showcase.py's own `Color = tuple[Channel, Channel, Channel]` row:
/// a bare ALIAS NAME whose `AliasEntry` carries `positions` Some
/// (`surface::compile_aliases`'s own tuple arm) resolves through
/// this SAME bare-Name arm that reads `element`/`head` for a
/// `list[X]`-shaped alias — forwarding the alias's own per-position
/// table onto the returned `DeclaredRefinement`, spelled `"tuple[
/// Channel, Channel, Channel]"` (the alias's OWN slot spellings
/// joined, the identical spelling an inline `c: tuple[Channel,
/// Channel, Channel]` parameter would carry — `all_three_alias_
/// spellings_carry_the_identical_sequence_window`'s own doc states
/// the same equivalence for a `list[X]` alias). Before this
/// forwarding, the hardcoded `positions: None` here made `Color`
/// resolve as a scalar with an EMPTY set, so `paint((255, 300, 0))`
/// never reached the POSITIONS LAW at all.
#[test]
fn a_bare_alias_name_forwards_its_compiled_tuple_positions() {
    let mut aliases = HashMap::new();
    aliases.insert(
        "Channel".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(0.0), at_most(255.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    let channel_set = aliases.get("Channel").expect("just inserted").set.clone();
    aliases.insert(
        "Color".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(Vec::new()),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: Some(vec![
                (channel_set.clone(), "Channel".to_owned()),
                (channel_set.clone(), "Channel".to_owned()),
                (channel_set, "Channel".to_owned()),
            ]),
        },
    );
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&name_expr("Color"), &aliases, &imports, &environment)
        .expect("Color resolves through the alias table");
    assert_eq!(got.spelling, "tuple[Channel, Channel, Channel]");
    let positions = got.positions.expect("Color carries a per-position table, not a scalar set");
    assert_eq!(positions.len(), 3);
    assert_eq!(positions[1].spelling, "Channel");
    assert_eq!(positions[1].set, aliases.get("Channel").expect("still present").set);
}

/// `tuple[int, Unreadable]` — one position this table cannot read
/// declines the WHOLE tuple, the same all-or-nothing rule
/// `dict[str, Unreadable]` already takes for its own value slot.
#[test]
fn fixed_arity_tuple_with_one_unreadable_position_declines_whole() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("tuple[int, Unreadable]").expect("test source must parse");

    let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment);
    assert!(got.is_none());
}

/// `tuple[int, ...]` — a VARIADIC tuple (the slice ends in a bare
/// `...`) is a different, unbounded-length shape this reader does not
/// recognize; it declines rather than misreading the ellipsis as a
/// second fixed position.
#[test]
fn variadic_tuple_declines_the_fixed_arity_reader() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("tuple[int, ...]").expect("test source must parse");

    let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment);
    assert!(got.is_none());
}

/// `tuple[int]` — a SINGLE-element tuple has no `Tuple`-wrapped slice
/// (ruff only wraps a multi-element subscript), so this reads as a
/// one-position tuple, not the element-container `list[X]` shape.
#[test]
fn single_element_tuple_resolves_one_position() {
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("tuple[int]").expect("test source must parse");

    let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
        .expect("tuple[int]'s one position must resolve");
    let positions = got.positions.expect("positions present");
    assert_eq!(positions.len(), 1);
    assert_eq!(positions[0].spelling, "int");
}

/// `list[Age]` (an alias element, not a bare sort) is unaffected by
/// the fallback — it still resolves through the ordinary alias path,
/// the same as before this fix.
#[test]
fn list_of_an_alias_element_still_resolves_through_the_alias_path() {
    let mut aliases = HashMap::new();
    aliases.insert(
        "Age".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(0.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    let imports = no_imports();
    let environment = no_locals();
    let parsed = parse_expression("list[Age]").expect("test source must parse");

    let got = declared_refinement(&parsed.into_expr(), &aliases, &imports, &environment)
        .expect("list[Age]'s element must resolve through the alias table");
    let element = got.element.expect("element present");
    assert_eq!(element.spelling, "Age");
    assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
}

#[test]
fn a_locally_rebound_alias_name_states_nothing() {
    let mut aliases = HashMap::new();
    aliases.insert(
        "PositiveInt".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(1.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    let imports = no_imports();
    let mut locally_bound = HashSet::new();
    locally_bound.insert("PositiveInt".to_owned());
    let environment = Environment::new(locally_bound);

    let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment);
    assert!(got.is_none());
}

#[test]
fn a_string_annotation_naming_a_visible_alias_resolves() {
    let mut aliases = HashMap::new();
    aliases.insert(
        "PositiveInt".to_owned(),
        AliasEntry {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![at_least(1.0)]),
            head: None,
            element: None,
            length_window: None,
            admits_none: false,
            positions: None,
        },
    );
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(
        &string_literal_expr("PositiveInt"),
        &aliases,
        &imports,
        &environment,
    )
    .expect("a string annotation naming a visible alias resolves");
    assert_eq!(got.spelling, "PositiveInt");
}

#[test]
fn an_alias_name_not_in_the_table_states_nothing_even_as_one_side_of_a_none_union() {
    // `NotAnAlias | None`: `NotAnAlias` is not a compiled alias in
    // this test's table, AND not one of the bare sorts
    // (`int`/`float`/`str`/`bool`) the union arm's own base-sort
    // fallback reads (`declared_refinement`'s `Expr::BinOp` arm doc:
    // inside a `X | None` union, an unresolved `X` falls back to
    // `base_sort_return_refinement` before declining) — so both the
    // alias lookup AND the base-sort fallback miss, and the whole
    // union states nothing, the same "alias lookup miss" reason a
    // bare `NotAnAlias` would give outside a union too.
    let union = none_union(name_expr("NotAnAlias"));
    let aliases = HashMap::new();
    let imports = no_imports();
    let environment = no_locals();

    let got = declared_refinement(&union, &aliases, &imports, &environment);
    assert!(got.is_none());
}
