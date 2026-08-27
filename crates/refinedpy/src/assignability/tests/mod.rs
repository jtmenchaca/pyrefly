//! Assignability unit tests.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;

use crate::typereading::DeclaredRefinement;
use crate::typereading::TypedDictMember;

use super::*;

mod scalar;
mod sequence;
mod judge;

/// A kernel handle for tests that ask it — same skip-when-unbuilt
/// pattern check.rs and expressions.rs already use, so this file's
/// tests run without requiring `pnpm kernel:native` first.
pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

/// `type Age = Annotated[int, Field(ge=0, le=120)]` — an int-sorted
/// alias, the shape surface.rs's annotated_expression_set builds.
pub(super) fn age_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
        spelling: "Age".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type OptionalAge = Age | None` — the same int-sorted ray, but
/// the declaration admits absence.
pub(super) fn optional_age_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
        spelling: "Age | None".to_owned(),
        admits_none: true,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

pub(super) fn fire_message(verdict: Verdict) -> String {
    match verdict {
        Verdict::Fire(message) => message,
        Verdict::Silent => panic!("expected Fire, got Silent"),
        Verdict::Undetermined(sentence) => panic!("expected Fire, got Undetermined({sentence})"),
    }
}

/// `type Label = Annotated[str, Field(max_length=8)]` — a bounded
/// string alias: the codepoint alphabet repeated, capped at 8. Not
/// the full string GROUND (`is_string_ground` requires an
/// unbounded repetition), so a String-tagged value against it flows
/// past both sort laws to the ordinary kernel membership ask.
pub(super) fn label_refinement() -> DeclaredRefinement {
    use refined_sets::codepoint_sets::codepoints;
    use refined_sets::refinement_forms::repeat_of;
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![repeat_of(codepoints(), 0, Some(8))]),
        spelling: "Label".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type AnyString = Annotated[str, Field()]` — the bare string
/// GROUND itself (`z.string()`'s Python twin, unbounded): one star
/// over the codepoint alphabet, the exact shape `is_string_ground`
/// recognizes.
pub(super) fn any_string_refinement() -> DeclaredRefinement {
    use refined_sets::codepoint_sets::strings;
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: strings(),
        spelling: "AnyString".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type ChartLayout = Literal["horizontal", "vertical", "centric",
/// "radial"]` — c-reads-and-values.py:1182's own alias, the UNION of
/// four singleton string tuples `typereading::string_literal_set`
/// builds. Untagged `Kind::Set` (`set_kind_tag: SetKindTag::None`)
/// reads as string-sorted by convention (ORIENTATION.md's own
/// recognition-slice fact) — no kind_tag field on a `RefinedSet`.
pub(super) fn chart_layout_refinement() -> DeclaredRefinement {
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::union;
    let set = make_refined_set(vec![union(
        make_refined_set(vec![union(
            make_refined_set(vec![union(string_tuple("horizontal"), string_tuple("vertical"))]),
            string_tuple("centric"),
        )]),
        string_tuple("radial"),
    )]);
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set,
        spelling: "ChartLayout".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type SingleCharacter = Annotated[str, Field(min_length=1,
/// max_length=1)]` — showcase.py's own alias, built through the real
/// `repetition()` function (not hand-rolled) so this pins the exact
/// shape `annotated_expression_set` compiles: `repetition`'s own
/// `(1, 1)` collapse ("a 1-element sequence IS the scalar layer")
/// hands back the bare codepoint-ground element, `Integer ∧
/// ((>=0 ∧ <=0xD7FF) ∪ (>=0xE000 ∧ <=0x10FFFF))`, with no surviving
/// Repeat/Concatenation/Star form.
pub(super) fn single_character_refinement() -> DeclaredRefinement {
    let set = refined_sets::repetition_window_forms::repetition(
        refined_sets::codepoint_sets::codepoints(),
        1,
        Some(1),
    );
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set,
        spelling: "SingleCharacter".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type LowerAsciiChar = Annotated[str, Field(pattern=r"^[a-z]$")]`
/// — a NARROWER character class than `SingleCharacter`'s bare length
/// window: `regex_compiler.rs`'s own `[a-z]` compilation, `Integer ∧
/// AtLeast(0x61) ∧ AtMost(0x7A)`, wholly inside the codepoint door but
/// not equal to the full alphabet (`is_codepoint_alphabet` is false
/// for it, unlike `single_character_refinement`'s own set), so this
/// pins the ordinary in-door, non-alphabet membership path stays
/// exercised by the fix.
pub(super) fn lower_ascii_char_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![integer(), at_least(0x61 as f64), at_most(0x7A as f64)]),
        spelling: "LowerAsciiChar".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type Grade = Literal["A", "B", "C"]` — o-grammar-refinements.py's
/// own alias, `surface.rs`'s `literal_alias_set`/
/// `string_literal_set`'s exact fold: `union(union(string_tuple("A"),
/// string_tuple("B")), string_tuple("C"))`, every member a single
/// character so every `string_tuple` call is a bare `OneOf` (no
/// `Concatenation` wrapper for a length-1 word).
pub(super) fn grade_refinement() -> DeclaredRefinement {
    use refined_sets::codepoint_sets::string_tuple;
    use refined_sets::refinement_forms::union;
    let set = make_refined_set(vec![union(
        make_refined_set(vec![union(string_tuple("A"), string_tuple("B"))]),
        string_tuple("C"),
    )]);
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set,
        spelling: "Grade".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// The code points a string literal spells, reused across the
/// string-ground tests so each stays a one-line construction over
/// `string_models.rs`'s own encoding (one code point per char).
pub(super) fn hi_points(text: &str) -> Vec<f64> {
    text.chars().map(|c| c as u32 as f64).collect()
}

/// `type Digits = Annotated[str, Field(min_length=1, max_length=4,
/// pattern=r"^[0-9]+$")]` — `surface.rs`'s `annotated_expression_set`
/// own fold: the compiled `[0-9]+` grammar's `Repeat` form (the
/// digit range, unbounded repetition) INTERSECTED with the
/// `min_length`/`max_length` window's own `Repeat` form (the full
/// codepoint ground, length `[1, 4]`) — two `Repeat` forms over
/// DIFFERENT element sets, never `on_one_tuple_layer` (each is a
/// `Form::Repeat`, not a scalar form), so this alias never reaches
/// the tuple-pun sort law this file's other Digits/Grade tests pin;
/// it flows straight to the ordinary whole-word kernel membership
/// ask.
pub(super) fn digits_refinement() -> DeclaredRefinement {
    use refined_sets::codepoint_sets::codepoints;
    use refined_sets::refinement_forms::repeat_of;
    let digit_range = make_refined_set(vec![integer(), at_least(0x30 as f64), at_most(0x39 as f64)]);
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![
            repeat_of(digit_range, 1, None),
            repeat_of(codepoints(), 1, Some(4)),
        ]),
        spelling: "Digits".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `dict[str, Age]` — the `element`-carrying declaration
/// `a-statements.py`'s `return_dict_members` needs, `element` set to
/// the same `age_refinement` every other test in this file shares.
pub(super) fn dict_of_age_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "dict[str, Age]".to_owned(),
        admits_none: false,
        element: Some(Box::new(age_refinement())),
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// `type Ratings = list[Annotated[int, Field(ge=1, le=5)]]` — the
/// `element`-carrying declaration the showcase's `bump_all` return
/// position states, `element` set to the bounded 1-to-5 window
/// `bump_all_clamped`'s in-set twin needs.
pub(super) fn ratings_refinement() -> DeclaredRefinement {
    let element_set = make_refined_set(vec![integer(), at_least(1.0), at_most(5.0)]);
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "list[Ratings]".to_owned(),
        admits_none: false,
        element: Some(Box::new(DeclaredRefinement {
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
            set: element_set,
            spelling: ">= 1 && <= 5 && integer".to_owned(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
        })),
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

/// A `Kind::Set` sequence with NO length bound — the same star shape
/// `repeat_of(element, lo, hi)` builds — wrapping `element`'s own
/// set, tagged Integer: `check.rs::seed_parameters`'s own seed for a
/// `list[X]` PARAMETER, and what a comprehension over it still
/// carries at its own return (`expressions.rs::comprehension_star_
/// elements`'s own re-windowed result).
pub(super) fn ratings_sequence_value(lo: f64, hi: f64) -> AbstractValue {
    use refined_sets::refinement_forms::repeat_of;
    let element = make_refined_set(vec![integer(), at_least(lo), at_most(hi)]);
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![repeat_of(element, 0, None)]), None, TrustSpec, SetKindTag::None)
    }
}

/// `PersonDict`'s own `age: Age` member table — the `members`-carrying
/// declaration h-object-literal-members.py's `dict_return_member`
/// needs, `age` set to the same `age_refinement` every other test in
/// this file shares. `age` is REQUIRED: `class PersonDict(TypedDict):
/// age: Age` states no `total=` keyword and no `NotRequired[...]`
/// marker, and library/typing.rst's `TypedDict` makes every such member
/// required ("``True`` is the default, and makes all items defined in
/// the class body required").
pub(super) fn person_dict_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "PersonDict".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: Some(vec![TypedDictMember {
            name: "age".to_owned(),
            required: true,
            declared: age_refinement(),
        }]),
        positions: None,
    }
}

/// `tuple[Age, Label]` — the `positions`-carrying declaration
/// `c-reads-and-values.py`'s own fixed-arity-tuple rows need, each
/// slot set to a DIFFERENT one of this file's shared refinements —
/// unlike the element law's one shared refinement, each position
/// keeps its own set.
pub(super) fn age_label_tuple_refinement() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(Vec::new()),
        spelling: "tuple[Age, Label]".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: Some(vec![age_refinement(), label_refinement()]),
    }
}
