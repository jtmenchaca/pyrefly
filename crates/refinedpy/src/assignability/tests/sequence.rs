//! Sequence-shaped judgment: sequence containment/subset over
//! strings, the element/members/positions laws for dict/list/
//! typed-dict/fixed-arity tuple declarations, and the Object/List
//! structural-mismatch arms.

use super::*;

/// A dict (`Kind::Object`) can never be a member of a numeric-ground
/// declared set — fires outright, never undetermined.
#[test]
fn a_dict_value_into_a_numeric_ground_alias_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = refined_domain::known_constructors::known_object(
        Vec::new(),
        Default::default(),
        false,
        TrustProved,
        false,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.to_lowercase().contains("dict"), "{message}");
}

/// A list (`Kind::List`) can never be a member of a numeric-ground
/// declared set — fires outright.
#[test]
fn a_list_value_into_a_numeric_ground_alias_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = refined_domain::known_constructors::known_list(Vec::new(), TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
}

/// `return None` under a plain (non-`Optional`) declared set
/// (`declared.admits_none == false`) fires outright — `Kind::Null`
/// is this crate's representation of Python's `None`, and a plain
/// declaration never admits it.
#[test]
fn none_into_a_plain_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    assert!(!declared.admits_none);
    let value = refined_domain::abstract_value::null_value();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.to_lowercase().contains("none"), "{message}");
}

/// `return None` under an `Optional[Age]`/`Age | None` declared set
/// (`declared.admits_none == true`) is silent — the admitted
/// absence is in the declaration, so `None` is a member.
#[test]
fn none_into_an_admits_none_declaration_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = optional_age_refinement();
    assert!(declared.admits_none);
    let value = refined_domain::abstract_value::null_value();
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// The OPAQUE law: a function object (opaque_value's first-cited
/// kind) against Age (numeric-ground) fires with the honest word,
/// never "a dict" — no kernel ask needed, the same short-circuit
/// the sort laws take.
#[test]
fn an_opaque_function_value_into_a_numeric_ground_alias_fires_with_its_word() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let value = refined_domain::abstract_value::opaque_value("a function value");
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("a function value"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
}

/// The honest JSON-union `json.loads` answers over an opaque string
/// (`expressions.rs::json_loads_value_space`, ISSUES.md b-runners:124)
/// judged against a numeric-ground alias FIRES, naming the first
/// non-numeric arm (`None`, this union's own first arm) — the
/// honest verdict for an opaque payload, since the union claims the
/// runtime value is SOME arm and a JSON `null` genuinely escapes an
/// `int`-sorted position. Built inline with the same seven arms
/// `json_loads_value_space` builds, mirroring the isinstance
/// narrowing test's own construction (narrowing.rs).
#[test]
fn a_json_loads_union_into_a_numeric_ground_alias_fires_naming_the_non_numeric_arm() {
    use refined_domain::abstract_value::float_sorted_unknown;
    use refined_domain::abstract_value::kind_union_of;
    use refined_domain::abstract_value::null_value;
    use refined_domain::abstract_value::opaque_value;
    use refined_sets::codepoint_sets::strings;

    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_refinement();
    let integer_arm = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)]), None, TrustProved, SetKindTag::None)
    };
    let union = kind_union_of(vec![
        null_value(),
        known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustProved),
        known_set(strings(), None, TrustProved, SetKindTag::None),
        integer_arm,
        float_sorted_unknown(),
        opaque_value("a list"),
        opaque_value("a dict"),
    ]);
    assert_eq!(union.kind, Kind::KindUnion);
    let message = fire_message(judge(&union, &declared, &kernel));
    assert!(message.contains("None"), "{message}");
    assert!(message.contains("'Age'"), "{message}");
}

/// The mirror: an opaque value against the STRING-ground alias fires
/// too — a function is never a member of a string-ground set either.
#[test]
fn an_opaque_value_into_a_string_ground_alias_fires_with_its_word() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = any_string_refinement();
    let value = refined_domain::abstract_value::opaque_value("a caught exception");
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("a caught exception"), "{message}");
    assert!(message.contains("'AnyString'"), "{message}");
}

/// An opaque value against a declared set that is NOT scalar-ground
/// (neither numeric- nor string-ground) declines the opaque law and
/// falls through to the general undetermined answer — the same
/// decline the Object/List/Null law already takes for a non-scalar
/// declared set.
#[test]
fn an_opaque_value_into_a_non_scalar_ground_alias_is_undetermined() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "Anything".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = refined_domain::abstract_value::opaque_value("a function value");
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Undetermined(_)));
}

/// CONTAINMENT REFUTATION, the sequence case: the kernel's
/// `seq_subset` decider now DECIDES the `strings()`-vs-length-window
/// pair (the sequence-containment fragment grew past the earlier
/// refusal this test used to pin — its old panic message, "subset is
/// decided for scalar and sequence shapes today," is no longer
/// reachable for this shape). `strings()` is the full, UNBOUNDED
/// codepoint ground; `Label`'s declared set caps length at 8
/// (`repeat_of(codepoints(), 0, Some(8))`), so the unbounded set is
/// never a subset of the capped one — `seq_subset` proves `false`,
/// a decided refutation, and `judge` fires the CONTAINMENT-REFUTATION
/// message ("the flowing set admits values outside the declared
/// set").
#[test]
fn an_unbounded_string_set_against_a_max_length_window_fires_containment() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::codepoint_sets::strings;
    let declared = label_refinement();
    let value = known_set(strings(), None, TrustProved, SetKindTag::None);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Label'"), "{message}");
    assert!(message.contains("admits values outside"), "{message}");
}

/// The SEQ_SUBSET ROUTING law's own pin: `e-class-and-function.py:610`'s
/// shape — a narrowed union of specific string members
/// (`or_narrowed_branch_call`'s own `Literal["insideStart",
/// "insideEnd", "end"]`, `sequence_shaped` true — each member's own
/// multi-character `string_tuple` is a `Concatenation`, not the
/// tuple-pun bare `OneOf` `Grade`'s single-character members build)
/// flowing into a declared set that admits every one of those members
/// PLUS more (`position_label`'s own four-member
/// `Literal["insideStart", "insideEnd", "end", "outside"]`) is a
/// genuine subset. Before this law, `scalar_subset` refused the pair
/// outright (neither side is 1-tuple/scalar shaped) and the row sat
/// Undetermined; `seq_subset` decides it Silent.
#[test]
fn a_narrowed_string_literal_union_subset_of_a_wider_one_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::union;
    let narrowed = make_refined_set(vec![union(
        make_refined_set(vec![union(string_tuple("insideStart"), string_tuple("insideEnd"))]),
        string_tuple("end"),
    )]);
    let wider = make_refined_set(vec![union(
        make_refined_set(vec![union(
            make_refined_set(vec![union(string_tuple("insideStart"), string_tuple("insideEnd"))]),
            string_tuple("end"),
        )]),
        string_tuple("outside"),
    )]);
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: wider,
        spelling: "PositionLabel".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = known_set(narrowed, None, TrustProved, SetKindTag::None);
    assert!(
        matches!(judge(&value, &declared, &kernel), Verdict::Silent),
        "a narrowed string-literal union wholly inside a wider one must be Silent, not Undetermined"
    );
}

/// SEQ_NO_SCALAR_REREAD PARITY (ledger 315): `sequence_shaped_safely`
/// asks `kernel.seq_no_scalar_reread` before falling back to
/// `sequence_shaped`'s own local recursion — the same kernel-first
/// order refined-ts-go's `noScalarReread`
/// (`abstractdomain/lattice_operations.go`) already takes. This pins
/// the wiring at both judgment sites the ask now reaches: an untagged
/// Set whose only form is a UNION of two multi-character string
/// tuples (`Union(Concatenation..., Concatenation...)`, a shape the
/// kernel's own `noScalarRereadF` recursion proves reread-safe by
/// walking into each concatenation's own operands) is read as
/// string-sorted at the `is_string_sorted_set` law, and the same
/// shape drives the `sequence_question` routing gate to `seq_subset`
/// rather than `scalar_subset` — both routes this pin already covers
/// via `judge`'s own observable verdict, now proved through the
/// kernel ask rather than the local recursion alone. A member wholly
/// inside a wider declared union of the same shape is Silent.
#[test]
fn seq_no_scalar_reread_parity_string_union_is_silent_through_the_kernel_ask() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::union;
    let narrowed = make_refined_set(vec![union(string_tuple("insideStart"), string_tuple("insideEnd"))]);
    let wider = make_refined_set(vec![union(
        make_refined_set(vec![union(string_tuple("insideStart"), string_tuple("insideEnd"))]),
        string_tuple("end"),
    )]);
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: wider,
        spelling: "PositionLabel".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = known_set(narrowed, None, TrustProved, SetKindTag::None);
    assert!(
        matches!(judge(&value, &declared, &kernel), Verdict::Silent),
        "a string-tuple union wholly inside a wider one must be Silent through the kernel-first reread-safety ask"
    );
}

/// The mirror: a member OUTSIDE the declared union
/// (`string_to_literal_union_parameter`'s own shape, widened to a Set
/// rather than that row's single-value read) fires — `seq_subset`
/// proving false over recognized sequence shapes is a decided
/// refutation, the same "false is a verdict, never a refusal in
/// disguise" reading `scalar_subset`'s own law doc states.
#[test]
fn a_string_literal_union_with_a_member_outside_the_declared_set_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::union;
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![union(string_tuple("node"), string_tuple("link"))]),
        spelling: "Tag".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let value = known_set(
        make_refined_set(vec![union(string_tuple("node"), string_tuple("other"))]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Tag'"), "{message}");
    assert!(message.contains("admits values outside"), "{message}");
}

// --- the ELEMENT LAW: dict[str, X]'s value-slot judgment ---

/// `return {"age": 200}` under `-> dict[str, Age]` — an Object with
/// one out-of-set member fires, naming the key so the reader sees
/// which member escaped ("(at key 'age')").
#[test]
fn a_dict_with_an_out_of_set_member_fires_naming_the_key() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = dict_of_age_refinement();
    let value = refined_domain::known_constructors::known_object(
        vec![refined_domain::abstract_value::ObjectKey {
            name: "age".to_owned(),
            numeric: false,
            value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
        }],
        None,
        true,
        TrustProved,
        false,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'Age'"), "{message}");
    assert!(message.contains("(at key 'age')"), "{message}");
}

/// `return {"age": 40}` under `-> dict[str, Age]` — every member sits
/// inside the element refinement, so the whole dict is Silent.
#[test]
fn a_dict_with_every_member_in_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = dict_of_age_refinement();
    let value = refined_domain::known_constructors::known_object(
        vec![refined_domain::abstract_value::ObjectKey {
            name: "age".to_owned(),
            numeric: false,
            value: known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
        }],
        None,
        true,
        TrustProved,
        false,
    );
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// `None` against a plain (non-`Optional`) `dict[str, Age]` fires —
/// a dict declaration is not scalar-shaped, so this exercises the
/// element law's own explicit Null arm rather than the ordinary
/// structural law (which would decline: `declared.set` is empty for
/// an element-carrying declaration).
#[test]
fn none_against_a_plain_dict_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = dict_of_age_refinement();
    assert!(!declared.admits_none);
    let value = refined_domain::abstract_value::null_value();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'dict[str, Age]'"), "{message}");
    assert!(message.to_lowercase().contains("none"), "{message}");
}

/// `None` against `dict[str, Age] | None` (`admits_none` true, still
/// element-carrying) is Silent — the admits_none check wins before
/// the element law's Null arm would otherwise fire.
#[test]
fn none_against_an_admits_none_dict_declaration_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut declared = dict_of_age_refinement();
    declared.admits_none = true;
    let value = refined_domain::abstract_value::null_value();
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// A list against `dict[str, Age]` fires — the element law's own
/// explicit List arm, kind-worded.
#[test]
fn a_list_against_a_dict_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = dict_of_age_refinement();
    let value = refined_domain::known_constructors::known_list(Vec::new(), TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'dict[str, Age]'"), "{message}");
    assert!(message.to_lowercase().contains("list"), "{message}");
}

// --- the ELEMENT-SET LAW: an unknown-length Kind::Set sequence's
// own element set against a list[X] declaration's element set ---

/// `bump_all`'s own showcase row: `[r + 1 for r in rs]` widens the
/// declared `1..=5` window to `2..=6`, which is NOT a subset of the
/// declared element set — fires the whole-container ELEMENT-SET
/// sentence, no single index to blame.
#[test]
fn a_sequence_whose_element_set_escapes_the_declared_element_set_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = ratings_refinement();
    let value = ratings_sequence_value(2.0, 6.0);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert_eq!(
        message,
        "a value of type 'list of (>= 2 && <= 6 && integer)' is not assignable to type 'list of (>= 1 && <= 5 && integer)'"
    );
}

/// `bump_all_clamped`'s own showcase twin: `[min(5, r + 1) for r in
/// rs]` keeps every mapped value inside the declared `1..=5`
/// window (clamped at 5) — the flowing element set sits inside the
/// declared one, so this is Silent.
#[test]
fn a_sequence_whose_element_set_sits_inside_the_declared_element_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = ratings_refinement();
    let value = ratings_sequence_value(2.0, 5.0);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

// --- the MEMBERS LAW: a TypedDict's per-member judgment ---

/// `return {"age": 200}` under `-> PersonDict` — the declared member's
/// own out-of-set value fires, naming the key and the value bare, up
/// front: "the key 'age' of value 200 is not assignable to type
/// '...'" — never the outer TypedDict's own alias name, since a
/// reader at this exact member needs to see what THIS member admits.
#[test]
fn a_typed_dict_with_an_out_of_set_member_fires_naming_the_key() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = person_dict_refinement();
    let value = refined_domain::known_constructors::known_object(
        vec![refined_domain::abstract_value::ObjectKey {
            name: "age".to_owned(),
            numeric: false,
            value: known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
        }],
        None,
        true,
        TrustProved,
        false,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("the key 'age' of value 200"), "{message}");
    assert!(message.contains("120"), "names Age's own ceiling: {message}");
}

/// The showcase's own `record_vitals(Vitals(heart_rate=72,
/// spo2=130))` row: a `spo2` member over its own `0..=100` ceiling
/// fires the exact wording the marker states. This pins the WORDING
/// half of the fix (`assignability::judge`'s MEMBERS LAW, given a
/// `members`-carrying declaration) — the SURFACING half is a
/// `check.rs` gap outside this file: a STATEMENT-LEVEL construction
/// (`return Person.model_validate(...)`, the m-pydantic corpus's own
/// shape) already fires correctly today, because `check.rs::sink_
/// value`'s law 3 surfaces `judge_construction`'s own per-field
/// fires directly at the sink with NO dependency on `Vitals`'s own
/// return/parameter annotation compiling a `DeclaredRefinement` at
/// all. A construction NESTED inside a call argument, this
/// showcase row's own shape, is the one construct that still loses
/// its fire: `check.rs::judge_one_call_argument` evaluates each
/// argument through plain `evaluate_expression`, whose same-module-
/// construction arm discards `judge_construction`'s fires by design
/// (they belong to whichever sink hosts the call, and an argument
/// position is not currently such a sink). `instances::model_
/// members_refinement` (already `pub`) builds exactly the `members:
/// Some(...)` shape this test constructs by hand, ready for a
/// `check.rs` fix to adopt either as a re-judged sink or to surface
/// `judge_construction`'s own already-correct verdict directly.
#[test]
fn a_vitals_construction_with_spo2_out_of_set_fires_the_shown_words_key_wording() {
    let Some(kernel) = loaded_kernel() else { return };
    let heart_rate_refinement = || DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![integer(), at_least(20.0), at_most(250.0)]),
        spelling: ">= 20 && <= 250 && integer".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let spo2_refinement = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![at_least(0.0), at_most(100.0)]),
        spelling: ">= 0 && <= 100".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    };
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "Vitals".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        // `required: false` on both, mirroring `instances::
        // model_members_refinement` — `Vitals` is an ordinary class, not
        // a TypedDict, so its member table states no totality.
        members: Some(vec![
            TypedDictMember {
                name: "heart_rate".to_owned(),
                required: false,
                declared: heart_rate_refinement(),
            },
            TypedDictMember {
                name: "spo2".to_owned(),
                required: false,
                declared: spo2_refinement,
            },
        ]),
        positions: None,
    };
    let value = refined_domain::known_constructors::known_object(
        vec![
            refined_domain::abstract_value::ObjectKey {
                name: "heart_rate".to_owned(),
                numeric: false,
                value: known_values(vec![72.0], PrimitiveKind::Integer, TrustProved),
            },
            refined_domain::abstract_value::ObjectKey {
                name: "spo2".to_owned(),
                numeric: false,
                // `spo2=130` is a bare Python int literal — evaluates
                // Integer-tagged regardless of the field's own
                // `float`-typed annotation (this checker's literal
                // reader does not itself coerce; only the MEMBERS
                // LAW's per-value judgment below does).
                value: known_values(vec![130.0], PrimitiveKind::Integer, TrustProved),
            },
        ],
        None,
        true,
        TrustProved,
        false,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert_eq!(message, "the key 'spo2' of value 130 is not assignable to type '>= 0 && <= 100'");
}

/// `return {"age": 40}` under `-> PersonDict` — the member sits inside
/// its own declared set, so the whole dict is Silent.
#[test]
fn a_typed_dict_with_its_member_in_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = person_dict_refinement();
    let value = refined_domain::known_constructors::known_object(
        vec![refined_domain::abstract_value::ObjectKey {
            name: "age".to_owned(),
            numeric: false,
            value: known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
        }],
        None,
        true,
        TrustProved,
        false,
    );
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// A CLOSED dict literal missing a REQUIRED declared member fires,
/// naming the missing key — `PersonDict` states no `total=` keyword, so
/// `age` is required (library/typing.rst, `TypedDict`: "By default, all
/// keys must be present in a ``TypedDict``"), and an empty dict literal
/// states its complete key set, so the key is proved absent.
#[test]
fn a_closed_typed_dict_missing_a_required_member_fires_naming_the_key() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = person_dict_refinement();
    let value = refined_domain::known_constructors::known_object(Vec::new(), None, true, TrustProved, false);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'age'"), "{message}");
    assert!(message.contains("missing the required key"), "{message}");
}

/// The same missing member on an OPEN value states nothing: an
/// incomplete key set cannot prove a key absent, only unread, so the
/// lenient path holds and the judgment is Silent.
#[test]
fn an_open_value_missing_a_required_member_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = person_dict_refinement();
    let value = refined_domain::known_constructors::known_object(Vec::new(), None, false, TrustProved, false);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// A `total=False` TypedDict requires nothing, so a closed dict literal
/// missing every declared member is Silent — library/typing.rst,
/// `TypedDict`: "a ``Point2D`` ``TypedDict`` can have any of the keys
/// omitted."
#[test]
fn a_total_false_typed_dict_missing_its_member_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut declared = person_dict_refinement();
    declared.members = Some(vec![TypedDictMember {
        name: "age".to_owned(),
        required: false,
        declared: age_refinement(),
    }]);
    let value = refined_domain::known_constructors::known_object(Vec::new(), None, true, TrustProved, false);
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// `None` against a plain (non-`Optional`) TypedDict declaration
/// fires — the members law's own explicit Null arm, mirroring the
/// element law's.
#[test]
fn none_against_a_plain_typed_dict_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = person_dict_refinement();
    assert!(!declared.admits_none);
    let value = refined_domain::abstract_value::null_value();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'PersonDict'"), "{message}");
    assert!(message.to_lowercase().contains("none"), "{message}");
}

/// A list against a TypedDict declaration fires — the members law's
/// own explicit List arm.
#[test]
fn a_list_against_a_typed_dict_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = person_dict_refinement();
    let value = refined_domain::known_constructors::known_list(Vec::new(), TrustProved);
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'PersonDict'"), "{message}");
    assert!(message.to_lowercase().contains("list"), "{message}");
}

// --- the POSITIONS LAW: a fixed-arity tuple's per-slot judgment ---

/// A list of two values, both inside their own position's set, is
/// Silent.
#[test]
fn a_two_slot_list_with_every_position_in_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_label_tuple_refinement();
    let value = refined_domain::known_constructors::known_list(
        vec![
            known_values(vec![40.0], PrimitiveKind::Integer, TrustProved),
            known_values(hi_points("ok"), PrimitiveKind::String, TrustProved),
        ],
        TrustProved,
    );
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// Slot 0 out of its own declared set fires, naming the offending
/// slot by its ordinal word — a 2-slot tuple's own slot 0 reads as
/// "the first slot," and the message states the slot's own bare
/// contents (never the outer tuple's alias name — a reader at this
/// exact position needs to see what THIS slot admits).
#[test]
fn a_two_slot_list_with_slot_zero_out_of_set_fires_naming_the_slot() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_label_tuple_refinement();
    let value = refined_domain::known_constructors::known_list(
        vec![
            known_values(vec![200.0], PrimitiveKind::Integer, TrustProved),
            known_values(hi_points("ok"), PrimitiveKind::String, TrustProved),
        ],
        TrustProved,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("200 in the first slot"), "{message}");
    assert!(message.contains("120"), "names Age's own ceiling: {message}");
}

/// `paint((255, 300, 0))` — the showcase's own 3-tuple row: slot 1
/// (the exact center of a 3-slot tuple) reads as "the middle slot,"
/// and the message states the slot's own bare contents with no "a
/// value of type" preamble: "300 in the middle slot is not
/// assignable to type '>= 0 && <= 255 && integer'".
#[test]
fn a_three_slot_tuples_middle_slot_out_of_set_fires_naming_it_the_middle_slot() {
    let Some(kernel) = loaded_kernel() else { return };
    fn channel_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(255.0)]),
            spelling: "Channel".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "tuple[Channel, Channel, Channel]".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: Some(vec![channel_refinement(), channel_refinement(), channel_refinement()]),
    };
    let value = refined_domain::known_constructors::known_list(
        vec![
            known_values(vec![255.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![300.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![0.0], PrimitiveKind::Integer, TrustProved),
        ],
        TrustProved,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert_eq!(message, "300 in the middle slot is not assignable to type '>= 0 && <= 255 && integer'");
}

/// `paint((255, 128, 0))` — every slot of the 3-tuple inside its own
/// declared set is Silent, the showcase's own in-set twin of the
/// middle-slot fire above.
#[test]
fn a_three_slot_tuple_with_every_slot_in_set_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    fn channel_refinement() -> DeclaredRefinement {
        DeclaredRefinement {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: make_refined_set(vec![integer(), at_least(0.0), at_most(255.0)]),
            spelling: "Channel".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        }
    }
    let declared = DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "tuple[Channel, Channel, Channel]".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: Some(vec![channel_refinement(), channel_refinement(), channel_refinement()]),
    };
    let value = refined_domain::known_constructors::known_list(
        vec![
            known_values(vec![255.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![128.0], PrimitiveKind::Integer, TrustProved),
            known_values(vec![0.0], PrimitiveKind::Integer, TrustProved),
        ],
        TrustProved,
    );
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}

/// A list of the WRONG LENGTH (one slot, not two) fires as a
/// structural mismatch rather than sitting undetermined or judging
/// past the end of `positions`.
#[test]
fn a_list_of_the_wrong_length_fires_as_a_structural_mismatch() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_label_tuple_refinement();
    let value = refined_domain::known_constructors::known_list(
        vec![known_values(vec![40.0], PrimitiveKind::Integer, TrustProved)],
        TrustProved,
    );
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'tuple[Age, Label]'"), "{message}");
}

/// A `tuple[Age, Age]`-declared sink, for the unknown-length arm below
/// — both slots the same set, so the ARITY is the only thing under
/// test rather than a per-slot sort mismatch.
fn age_pair_tuple_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "tuple[Age, Age]".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: Some(vec![age_refinement(), age_refinement()]),
    }
}

/// One unknown-length sequence carrying `Age`'s own element window —
/// the shape a declared `list[Age]`/`Sequence[Age]` parameter seeds.
fn age_window(lo: i64, hi: Option<i64>) -> AbstractValue {
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(
            refined_sets::repetition_window_forms::repetition(age_refinement().set, lo, hi),
            None,
            TrustProved,
            refined_domain::abstract_value::SetKindTag::None,
        )
    }
}

/// An UNBOUNDED-length sequence against a fixed-arity tuple fires as a
/// structural mismatch: `list[int]`'s `[0, +inf)` always admits a
/// sequence longer than any fixed arity. Before this arm the judgment
/// sat undetermined and said nothing — A7.sink.assign's own
/// `assign_to_tuple` and A7.sink.ret's own `returns_three`.
#[test]
fn an_unbounded_length_sequence_against_a_fixed_arity_tuple_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_pair_tuple_refinement();
    let message = fire_message(judge(&age_window(0, None), &declared, &kernel));
    assert!(message.contains("'tuple[Age, Age]'"), "{message}");
    assert!(message.contains("2 elements"), "{message}");
}

/// A window whose length is pinned to EXACTLY the declared arity is
/// admitted, and each declared position is judged against the window's
/// one element set.
#[test]
fn a_sequence_pinned_to_the_declared_arity_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_pair_tuple_refinement();
    assert!(matches!(judge(&age_window(2, Some(2)), &declared, &kernel), Verdict::Silent));
}

/// A window pinned to a DIFFERENT exact length still fires — the same
/// structural reading the known-List arm gives a length mismatch.
#[test]
fn a_sequence_pinned_to_the_wrong_arity_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_pair_tuple_refinement();
    let message = fire_message(judge(&age_window(3, Some(3)), &declared, &kernel));
    assert!(message.contains("exactly 3 elements"), "{message}");
}

/// `None` against a plain (non-`Optional`) fixed-arity tuple
/// declaration fires — the positions law's own explicit Null arm,
/// mirroring the element/members laws.
#[test]
fn none_against_a_plain_positions_declaration_fires() {
    let Some(kernel) = loaded_kernel() else { return };
    let declared = age_label_tuple_refinement();
    assert!(!declared.admits_none);
    let value = refined_domain::abstract_value::null_value();
    let message = fire_message(judge(&value, &declared, &kernel));
    assert!(message.contains("'tuple[Age, Label]'"), "{message}");
    assert!(message.to_lowercase().contains("none"), "{message}");
}

/// `None` against `Optional[tuple[Age, Label]]` (`admits_none` true)
/// is Silent — the same admits_none precedence the element/members
/// laws already give their own Null arm.
#[test]
fn none_against_an_admits_none_positions_declaration_is_silent() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut declared = age_label_tuple_refinement();
    declared.admits_none = true;
    let value = refined_domain::abstract_value::null_value();
    assert!(matches!(judge(&value, &declared, &kernel), Verdict::Silent));
}
