//! Scalar-set judgment: the sort laws (int/float/string/boolean),
//! the tuple-pun gate, the codepoint-alphabet carve-out, and the
//! containment-refutation law over plain scalar and shaped sets.

use super::*;

/// `x: Age = 30.0` fires — Age is int-sorted, and 30.0 is
/// Float-tagged, so the sort law fires even though the real value
/// 30 sits inside [0, 120]. The message spells the number the
/// Python way: "30.0" keeps its trailing ".0", never bare "30".
#[test]
fn a_float_tagged_whole_value_into_an_int_sorted_alias_fires_spelled_with_its_dot_zero() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let thirty_float = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
    let message = fire_message(judge(&thirty_float, &declared, &kernel));
    assert!(message.contains("'30.0'"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
}

/// `x: Age = 30` (Integer-tagged) is silent — the ordinary kernel
/// membership path, no sort law involved.
#[test]
fn an_integer_tagged_in_range_value_into_an_int_sorted_alias_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let thirty_int = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
    assert!(matches!(judge(&thirty_int, &declared, &kernel), Verdict::Silent));
}

/// `6 / 3` evaluates to a Float-tagged 2.0 (Python's `/` is always
/// true division, expressions.rs's own pinned test) — assigned into
/// an int-sorted alias, THAT Float tag is what makes this fire, not
/// the real value 2 being out of range.
#[test]
fn true_division_of_two_ints_still_fires_into_an_int_sorted_alias() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let two_float = crate::expressions::binary_arithmetic_value(
        ruff_python_ast::Operator::Div,
        &known_values(vec![6.0], PrimitiveKind::Integer, TrustProved),
        &known_values(vec![3.0], PrimitiveKind::Integer, TrustProved),
    );
    assert_eq!(two_float.kind_tag, Some(PrimitiveKind::Float));
    let message = fire_message(judge(&two_float, &declared, &kernel));
    assert!(message.contains("'2.0'"), "{message}");
}

/// A declared Float-sorted alias (no `integer()` form) never fires
/// the sort law — the int-sort gate is specific to a declared set
/// that actually carries the `int` form.
#[test]
fn a_float_tagged_value_into_a_float_sorted_alias_never_hits_the_sort_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![at_least(0.0)]),
        spelling: "Weight".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let thirty_float = known_values(vec![30.0], PrimitiveKind::Float, TrustProved);
    assert!(matches!(judge(&thirty_float, &declared, &kernel), Verdict::Silent));
}

/// `x: Label = "hi"` — a whole String-tagged word asked ONCE against
/// the alias, silent because "hi" (2 code points) sits under the
/// 8-character ceiling.
#[test]
fn a_string_value_member_of_a_string_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = label_refinement();
    let value = known_values(hi_points("hi"), PrimitiveKind::String, TrustProved);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// `x: Label = "too-long-string"` (15 code points, over the 8-char
/// ceiling) fires ONE membership question over the whole word, and
/// the message quotes the string readably rather than spelling code
/// points.
#[test]
fn a_string_value_not_a_member_of_a_string_set_fires_quoting_the_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = label_refinement();
    let value = known_values(
        hi_points("too-long-string"),
        PrimitiveKind::String,
        TrustProved,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("\"too-long-string\""), "{message}");
    assert!(message.contains("'Label'"), "{message}");
}

/// `c-reads-and-values.py:1197`'s HELD arm — `layout` narrowed to
/// `"horizontal"` (a String-tagged whole word) against
/// `Literal["horizontal", "vertical", "centric", "radial"]`: ONE
/// membership ask over the whole word (line 208's `is_string` arm),
/// silent because "horizontal" is one of the four tuples the union
/// spells.
#[test]
fn a_literal_union_member_string_value_is_silent_via_whole_word_membership() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = chart_layout_refinement();
    let value = known_values(hi_points("horizontal"), PrimitiveKind::String, TrustProved);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// The mirror: a String-tagged whole word NOT among the union's four
/// tuples fires the ordinary kernel membership ask, quoting the
/// string readably.
#[test]
fn a_literal_union_non_member_string_value_fires_quoting_the_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = chart_layout_refinement();
    let value = known_values(hi_points("diagonal"), PrimitiveKind::String, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("\"diagonal\""), "{message}");
    assert!(message.contains("'ChartLayout'"), "{message}");
}

/// c-reads-and-values.py:1199's own shape: `return None` under a
/// declared `Literal["horizontal", "vertical"]` — NOT `| None`
/// wrapped, so `declared.admits_none` is false. `Kind::Null` reaches
/// A Literal union of specific string tuples is SEQUENCE-SHAPED
/// (`sequence_shaped`: a Union of Concatenation forms), so the
/// structural-mismatch law recognizes it and `None` against a
/// non-admitting Literal union FIRES — None is provably not a
/// string of any spelling (c-reads-and-values.py's fall-through-
/// to-None row).
#[test]
fn none_against_a_literal_union_that_does_not_admit_none_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let two_member_declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![refined_sets::refinement_forms::union(
            refined_sets::codepoint_sets::string_tuple("horizontal"),
            refined_sets::codepoint_sets::string_tuple("vertical"),
        )]),
        spelling: "Literal['horizontal', 'vertical']".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = refined_domain::abstract_value::null_value();
    let Verdict::Fire(message) = judge(&value, &two_member_declared, &kernel) else {
        panic!("None against a non-admitting string-Literal union fires the structural law");
    };
    assert!(message.contains("None"), "{message}");
}

/// A String-tagged value against a NUMERIC-ground alias (Age, an
/// int-sorted ray) still fires — but via the ORDINARY whole-word
/// kernel membership ask, not the sort law: `Age`'s own range
/// `[0, 120]` sits WITHIN the codepoint door (every value it admits
/// is a valid single codepoint), so the sort law declines per the
/// tuple-pun gate (`within_codepoint_door`) and falls through.
/// `"30"` is a 2-CODEPOINT tuple, never a member of `Age`'s
/// 1-tuple-shaped set regardless, so the kernel's own derivative
/// walk refutes it — the fire message is identical either way, this
/// test only pins that the value is still refused.
#[test]
fn a_string_value_into_a_numeric_ground_alias_still_fires_via_the_kernel_ask() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_values(hi_points("30"), PrimitiveKind::String, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
}

/// The TUPLE-PUN fix's own pin: `"B"` (a String-tagged whole word)
/// into `Grade = Literal["A", "B", "C"]` — a `Union` of
/// single-codepoint `OneOf`s (`surface.rs`'s `string_literal_set`
/// over 1-character members) — is SILENT. Before the fix, the
/// string-vs-numeric-ground sort law read `Grade`'s shape as
/// numeric-ground (`on_one_tuple_layer` alone, blind to the
/// single-character tuple pun) and fired outright on every real
/// member; `Grade`'s own range sits wholly inside the codepoint
/// door (every member is a valid codepoint) with no sequence form
/// present, so the law now declines and the ordinary whole-word
/// kernel membership ask decides it correctly.
#[test]
fn a_single_character_literal_union_member_is_silent_not_the_numeric_ground_sort_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = grade_refinement();
    let value = known_values(hi_points("B"), PrimitiveKind::String, TrustProved);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// The mirror: `"F"` (outside `Grade`'s three members) fires via the
/// ordinary whole-word kernel ask, quoting the string readably —
/// never the numeric-ground sort law's own wording.
#[test]
fn a_single_character_literal_union_non_member_fires_quoting_the_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = grade_refinement();
    let value = known_values(hi_points("F"), PrimitiveKind::String, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("\"F\""), "{message}");
    assert!(message.contains("'Grade'"), "{message}");
}

/// The bug this fix closes: showcase.py:258's own row, `initial("🎉")`
/// — one codepoint, a genuine member of `SingleCharacter`'s collapsed
/// codepoint-ground element. Before the `is_codepoint_alphabet`
/// carve-out, `requires_integer` read the collapsed element's own
/// `Form::Integer` (part of the codepoint alphabet's definition, not
/// a declared int base) the same way it reads `Age`'s genuine int
/// base, and fired "a string where an integer is expected" on a
/// value that IS a member. Silent now: the alphabet identity check
/// lets this fall through to the ordinary whole-word kernel
/// membership ask, which the tuple-pun law already trusts for
/// `Grade`'s narrower codepoint-doored sets.
#[test]
fn a_single_codepoint_string_inside_the_character_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = single_character_refinement();
    let value = known_values(hi_points("🎉"), PrimitiveKind::String, TrustProved);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// A single codepoint OUTSIDE a character set: `"5"` (not a-z)
/// against `LowerAsciiChar` fires via the ordinary whole-word kernel
/// ask, quoting the string readably — never the sort law's "is not
/// allowed here" wording, matching `Grade`'s own non-member pin.
#[test]
fn a_single_codepoint_string_outside_the_character_set_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = lower_ascii_char_refinement();
    let value = known_values(hi_points("5"), PrimitiveKind::String, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("\"5\""), "{message}");
    assert!(message.contains("'LowerAsciiChar'"), "{message}");
}

/// showcase.py:260's own designated sibling: `initial("👨‍👩‍👧")`, a
/// 5-codepoint ZWJ family emoji — the OUT-OF-SET leg this fix must
/// leave untouched. Reaches the identical whole-word kernel
/// membership ask the in-set leg above now also reaches (both fall
/// through the same `is_codepoint_alphabet` carve-out), refused on
/// length rather than range, with the ordinary `refutation()`
/// sentence — never the sort law's "is not allowed here" wording the
/// bug fired on the in-set leg.
#[test]
fn a_multi_codepoint_string_against_the_character_set_fires_with_the_length_based_sentence() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = single_character_refinement();
    let value = known_values(hi_points("👨‍👩‍👧"), PrimitiveKind::String, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("is not assignable to type"), "{message}");
    assert!(message.contains("'SingleCharacter'"), "{message}");
    assert!(!message.contains("is not allowed here"), "{message}");
}

/// The mirror: an Integer-tagged value against the STRING-ground
/// alias fires the sort law — a number is never a member of a
/// string-ground set, regardless of whether its real value would
/// pass a bare membership ask.
#[test]
fn a_numeric_value_into_a_string_ground_alias_fires_the_sort_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = any_string_refinement();
    let value = known_values(vec![30.0], PrimitiveKind::Integer, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'AnyString'"), "{message}");
}

/// The STRING-SORTED-SET law declines for a target whose own range
/// sits WITHIN THE CODEPOINT DOOR: an UNTAGGED Set holding the full
/// string ground (`kind_tag: None`, the exact shape `expressions.rs`'s
/// `__name__` read carries — `known_set(strings(), None, TrustSpec,
/// SetKindTag::None)`) against Age FIRES via the sort law: `Age`
/// carries the explicit `Integer` form, which is numeric INTENT by
/// construction (no string set ever builds one), so the
/// `requires_integer` opening decides the sort mismatch even though
/// Age's `[0, 120]` range sits inside the codepoint door — the
/// d-module-surface row's own expectation ("a host-defined string is
/// not in an int-sorted set").
#[test]
fn an_untagged_string_shaped_set_into_an_integer_formed_alias_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::codepoint_sets::strings;
    let declared = age_refinement();
    let value = known_set(strings(), None, TrustProved, SetKindTag::None);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
}

/// The same law, explicitly String-tagged: a Set carrying `kind_tag:
/// Some(PrimitiveKind::String)` against Age fires identically — the
/// law reads either the explicit tag or the untagged-Set convention,
/// and Age's explicit `Integer` form opens the sort law for both.
#[test]
fn a_string_tagged_set_into_an_integer_formed_alias_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::codepoint_sets::strings;
    let declared = age_refinement();
    let value = AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, TrustProved, SetKindTag::None)
    };
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
}

/// The tuple-pun gate's own Set-kind pin: an EXPLICITLY String-tagged
/// Set built from a single-codepoint `OneOf` (`{66}`, "B") against
/// `Grade` (single-codepoint `OneOf`/`Union` forms only,
/// `on_one_tuple_layer` true, no demonstrable sequence form) reaches
/// the CONTAINMENT ask rather than the sort law, because `Grade`
/// sits wholly inside the codepoint door — `{66}` IS `scalar_subset`
/// of `Grade`'s set (both are scalar/1-tuple shaped here), so this
/// is Silent, not a sort-law Fire. (An UNTAGGED bare `OneOf` Set
/// reads as NUMERIC-sorted by the codebase's own convention — this
/// test tags String explicitly so it exercises the string-sorted
/// branch, mirroring `a_string_tagged_set_into_a_codepoint_door_
/// alias_is_undetermined` above but with a value that IS
/// scalar-shaped, so the containment ask decides rather than
/// refuses.)
#[test]
fn a_single_codepoint_string_tagged_set_wholly_inside_a_single_character_literal_union_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::refinement_forms::one_of;
    let declared = grade_refinement();
    let value = AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(make_refined_set(vec![one_of(&[66.0])]), None, TrustProved, SetKindTag::None)
    };
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// The mirror: a NUMERIC-sorted Set (an untagged Set whose own set is
/// `on_one_tuple_layer`, e.g. the bare `integer()` line) against the
/// STRING-ground alias fires the sort law before any kernel ask — a
/// number is never a member of a string-ground set, regardless of
/// which real numbers the set admits.
#[test]
fn a_numeric_shaped_set_into_a_string_ground_alias_fires_the_sort_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = any_string_refinement();
    let value = known_set(make_refined_set(vec![integer()]), None, TrustProved, SetKindTag::None);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'AnyString'"), "{message}");
}

/// A NUMERIC star (`list[int]`'s own element-read shape,
/// `Form::Star(int's set)` — `refinedpy::collection_models::
/// subscript_read`'s new `Kind::Set` arm hands this exact shape back
/// for `ages[0]`) against `Age` must NOT take the string-sort law:
/// before `sequence_shaped` learned to check the star's own alphabet,
/// ANY `Form::Star` read as string-shaped regardless of its element,
/// which would have wrongly fired "a string-sorted value is never in
/// an int-sorted set" here even though the element is a whole number.
/// The correct path is the CONTAINMENT law: the unbounded int ray is
/// not a subset of Age's `[0, 120]` window, so this fires the
/// CONTAINMENT message instead.
#[test]
fn a_numeric_star_shaped_set_into_age_fires_containment_not_the_sort_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let whole_ints = make_refined_set(vec![integer(), refined_sets::refinement_forms::at_least(f64::NEG_INFINITY)]);
    let numeric_star = make_refined_set(vec![refined_sets::refinement_forms::star(whole_ints)]);
    let value = known_set(numeric_star, None, TrustProved, SetKindTag::None);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(
        message.contains("admits values outside"),
        "must fire the CONTAINMENT message, not the string-sort law: {message}"
    );
}

/// The FLOAT-SORT SET law: a Float-sorted Set (`float_sorted_unknown`
/// — the shape `math.sqrt`'s result carries) against Age (int-sorted)
/// fires — a float-sorted value is never a member of an int-sorted
/// set, regardless of what real numbers the set admits.
#[test]
fn a_float_sorted_set_into_an_int_sorted_alias_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = refined_domain::abstract_value::float_sorted_unknown();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.to_lowercase().contains("float"), "{message}");
}

/// CONTAINMENT-REFUTATION LAW, the overlap case: a Float-sorted Set
/// against a float-TOLERANT (non-integer-sorted) declared set skips
/// the sort law (specific to `requires_integer`) and falls to the
/// ordinary Set path — R-bar (`float_sorted_unknown`'s set, the
/// whole real line) is NOT a subset of `Weight`'s `[0, ∞)` ray (it
/// admits negatives the declared set excludes) and is NOT disjoint
/// from it either (they overlap on `[0, ∞)`). Before this law, that
/// overlap sat Undetermined; the law now fires it, because
/// `scalar_subset` proving false over decided scalar forms IS a
/// refutation of the checked position's containment claim, whether
/// the two sets are disjoint or merely overlapping.
#[test]
fn a_float_sorted_set_overlapping_a_non_integer_sorted_alias_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![at_least(0.0)]),
        spelling: "Weight".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = refined_domain::abstract_value::float_sorted_unknown();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Weight'"), "{message}");
    assert!(message.contains("admits values outside"), "{message}");
}

/// CONTAINMENT-REFUTATION LAW, the subset case: an int-sorted Set
/// `[10, 20]` (wholly inside Age's `[0, 120]` window) is still
/// Silent — `scalar_subset` proves the containment claim outright,
/// unchanged by this law.
#[test]
fn an_int_sorted_set_wholly_inside_the_declared_window_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_set(
        make_refined_set(vec![integer(), at_least(10.0), at_most(20.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// CONTAINMENT-REFUTATION LAW, the decided-disjoint case: an
/// int-sorted Set entirely below Age's floor (`< 0`) still fires —
/// `scalar_disjoint` proves no member of either set can ever be the
/// other's, the sharpest form of refutation the law covers.
#[test]
fn an_int_sorted_set_disjoint_from_the_declared_window_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_set(
        make_refined_set(vec![integer(), below(0.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.contains("admits values outside"), "{message}");
}

/// CONTAINMENT-REFUTATION LAW, the overlap case (int-sort whole set
/// vs Age window): the unrestricted integer line is NOT a subset of
/// Age's `[0, 120]` window (it admits negatives and values above
/// 120) and NOT disjoint from it either (10 is a member of both).
/// Before this law, that overlap sat Undetermined; the law now fires
/// it — `scalar_subset` proving false over decided scalar forms is a
/// refutation of the checked position's containment claim regardless
/// of whether the two sets are disjoint or merely overlapping.
#[test]
fn an_int_sort_whole_set_overlapping_the_age_window_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_set(
        make_refined_set(vec![integer()]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.contains("admits values outside"), "{message}");
}

/// The BOOLEAN PRODUCT LAW: `True` (Boolean-tagged) against Age
/// (int-sorted) fires — bool is excluded from the int sort by
/// product law, the fixture rows' own reason
/// (b-body-expressions.py:744, c-reads-and-values.py:999). Before
/// this law, a Boolean-tagged value flowed to the per-value kernel
/// membership ask and passed silently (1.0 sits inside [0, 120]).
#[test]
fn a_boolean_true_into_an_int_sorted_alias_fires_by_product_law() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("True"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.to_lowercase().contains("product law"), "{message}");
}

/// The non-firing neighbor: a Boolean-tagged value against a
/// NON-integer-sorted declared set is unchanged — the product law
/// gates on `requires_integer` alone, so a float-tolerant alias
/// still asks the kernel per value the ordinary way (1.0 is a member
/// of `[0, ∞)`).
#[test]
fn a_boolean_true_into_a_non_integer_sorted_alias_is_unchanged() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![at_least(0.0)]),
        spelling: "Weight".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// `TypeAdapter(Digits).validate_python("42")` — m-pydantic-schema.py:65's
/// own row. `"42"` (a String-tagged whole word, 2 code points, both
/// ASCII digits) is a genuine member of `Digits`'s pattern-and-window
/// set: `judge()` proves this Silent via the ordinary whole-word
/// kernel membership ask, root-causing this row's OWN reported false
/// Fire to `check.rs`'s adapter-alias route (`adapter_alias_verdict`'s
/// LAX INT COERCION), not to this file — that coercion is gated only
/// on the value being a digit string and the alias not being a
/// `StrictInt` name, with no check that the alias's declared set is
/// even NUMERIC-sorted, so `"42"` is silently rewritten to the
/// Integer value `42` before `judge()` ever sees it, and `42`
/// (correctly) fails membership in a codepoint-tuple-shaped set. This
/// test pins that `judge()` itself, given the UN-coerced String
/// value, decides the row correctly.
#[test]
fn a_string_value_member_of_the_digits_pattern_and_window_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = digits_refinement();
    let value = known_values(hi_points("42"), PrimitiveKind::String, TrustProved);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// The mirror: `"ab"` (letters, outside the digit pattern) fires via
/// the ordinary whole-word kernel ask, quoting the string readably —
/// m-pydantic-schema.py:71's own row.
#[test]
fn a_string_value_outside_the_digits_pattern_fires_quoting_the_string() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = digits_refinement();
    let value = known_values(hi_points("ab"), PrimitiveKind::String, TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("\"ab\""), "{message}");
    assert!(message.contains("'Digits'"), "{message}");
}
