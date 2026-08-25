//! What a pattern's own LITERAL shape proves about a taken arm's
//! subject, independent of whether the concrete subject is known —
//! `pattern_proved_value` and the scalar/string readers it shares with
//! `outcome.rs`'s own `match_value_outcome`.

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::concatenation;
use refined_sets::refinement_forms::empty_tuple;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Pattern;
use ruff_python_ast::Singleton;

use crate::env::Environment;
use crate::expressions::evaluate_expression;

/// The exact value a pattern's own LITERAL shape proves about a taken
/// arm's subject — independent of whether the concrete subject is
/// known (unlike `pattern_outcome`, which requires a known subject to
/// decide TAKEN/NOT-TAKEN). This is the pattern's proof read
/// syntactically: a `MatchValue` proves exactly its own literal
/// expression's value, TAGGED as that literal's own evaluated
/// `PrimitiveKind` (`evaluate_expression`'s `number_literal_value`
/// convention — an int literal tags `Integer`, a float literal tags
/// `Float` — so `case 40:` proves an `Integer`-tagged 40, never a
/// bare `Number`; a STRING literal tags `String`, its `values` the
/// whole codepoint tuple); a `MatchSingleton` proves `True`/`False` as
/// the Boolean-tagged 1.0/0.0 CPython's `is`-identity singletons
/// (`None` proves no NUMERIC value — a null subject is never a member
/// of a numeric refined set, so it contributes nothing here, matching
/// `narrowing.rs`'s own "None is never a Values member" reading);
/// `MatchOr` proves the UNION of every alternative's own proof (PEP
/// 634's rule that all alternatives bind the same names does not
/// extend to proving the same value — `18 | 21 | 40` proves any of the
/// three, `"a" | "b"` proves either word as a `Kind::Set` —
/// `string_pattern_or_value`'s own doc) — every alternative must prove
/// the SAME tag, or the whole pattern declines (an honest narrow scope:
/// this function never invents a `KindUnion` to paper over a genuinely
/// mixed-sort alternative list); `MatchAs` recurses into its own inner
/// pattern when present, or proves NOTHING when it is a bare capture/
/// wildcard (a bare `case x:` states no literal fact about the subject
/// at all — the caller's job to leave the subject unnarrowed in that
/// case, never to invent a value). Every other pattern shape
/// (Sequence/Mapping/Class/Star) proves nothing this function reads —
/// `None`.
///
/// `check.rs`'s match-join fallback (`walk_match`) calls this to
/// narrow a captured name — or the subject itself when the pattern
/// captures nothing — down from the coarse pre-match claim to exactly
/// what the arm's own pattern proves, the same "a narrowing must be
/// the pattern's own proved claim" discipline `narrowing.rs`'s
/// isinstance/comparison leaves already follow. The returned value's
/// trust grade is `TrustProved` — the pattern's own literal is read
/// exactly, the same grade `number_literal_value` gives every numeric
/// literal.
pub fn pattern_proved_value(pattern: &Pattern, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    match pattern {
        Pattern::MatchValue(value_pattern) => {
            let literal_value = evaluate_expression(&value_pattern.value, environment, kernel);
            if literal_value.kind != Kind::Values {
                return None;
            }
            let kind_tag = literal_value.kind_tag?;
            if kind_tag == PrimitiveKind::String {
                // a string literal's own `values` IS its whole codepoint
                // tuple (one value, however many code points) — never
                // read against the `len() == 1` numeric-member check
                // below, which counts SCALAR members, not codepoints.
                return Some(literal_value);
            }
            if literal_value.values.len() != 1 {
                return None;
            }
            if !matches!(
                kind_tag,
                PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float | PrimitiveKind::Boolean
            ) {
                return None;
            }
            Some(literal_value)
        }
        Pattern::MatchSingleton(singleton_pattern) => match singleton_pattern.value {
            Singleton::True => Some(known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved)),
            Singleton::False => Some(known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved)),
            Singleton::None => None,
        },
        Pattern::MatchOr(or_pattern) => {
            let mut alternatives = or_pattern.patterns.iter();
            let first = pattern_proved_value(alternatives.next()?, environment, kernel)?;
            if first.kind_tag == Some(PrimitiveKind::String) {
                return string_pattern_or_value(first, alternatives, environment, kernel);
            }
            let mut values = first.values.clone();
            let kind_tag = first.kind_tag;
            for alternative in alternatives {
                let proved = pattern_proved_value(alternative, environment, kernel)?;
                if proved.kind_tag != kind_tag {
                    // a genuinely mixed-sort alternative list — never
                    // guessed at, an honest decline
                    return None;
                }
                for value in proved.values {
                    if !values.contains(&value) {
                        values.push(value);
                    }
                }
            }
            Some(known_values(values, kind_tag?, TrustProved))
        }
        Pattern::MatchAs(as_pattern) => match as_pattern.pattern.as_deref() {
            Some(inner) => pattern_proved_value(inner, environment, kernel),
            None => None,
        },
        Pattern::MatchSequence(_) | Pattern::MatchMapping(_) | Pattern::MatchClass(_) | Pattern::MatchStar(_) => None,
    }
}

/// The STRING twin of `pattern_proved_value`'s `MatchOr` numeric-merge
/// arm: `first` is already proved String-tagged (the caller's own
/// check), so this builds the UNION of every alternative's own word
/// tuple as a `Kind::Set` — `"axis" | "item"` proves `{"axis", "item"}`
/// as a set of exact words, never a flat `values` merge (a string's
/// `values` is one whole codepoint tuple, not a list of scalar members
/// — `word_tuple_of_codepoints`'s own doc). Every alternative must also
/// prove String-tagged, or the whole pattern declines, the same
/// "genuinely mixed-sort alternative list, never guessed at" honesty
/// the numeric arm keeps.
fn string_pattern_or_value<'a>(
    first: AbstractValue,
    rest: impl Iterator<Item = &'a Pattern>,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let mut combined = word_tuple_of_codepoints(&first.values);
    for alternative in rest {
        let proved = pattern_proved_value(alternative, environment, kernel)?;
        if proved.kind_tag != Some(PrimitiveKind::String) {
            return None;
        }
        combined = make_refined_set(vec![union(combined, word_tuple_of_codepoints(&proved.values))]);
    }
    Some(known_set(combined, None, TrustProved, SetKindTag::None))
}

/// A codepoint tuple's own singleton set — the SAME shape
/// `refined_sets::codepoint_sets::string_tuple` builds from a `&str`,
/// built here from `&[f64]` directly (a `Kind::Values` String-tagged
/// value's own `values` field, already codepoints — no lossy round trip
/// through `char::from_u32`/`String` needed to build the set itself).
fn word_tuple_of_codepoints(points: &[f64]) -> RefinedSet {
    if points.is_empty() {
        return make_refined_set(vec![empty_tuple()]);
    }
    let mut set = make_refined_set(vec![one_of(&[points[points.len() - 1]])]);
    for point in points[..points.len() - 1].iter().rev() {
        set = make_refined_set(vec![concatenation(make_refined_set(vec![one_of(&[*point])]), set)]);
    }
    set
}

/// The single numeric value a known abstract value carries, if it
/// carries exactly one — Number- or Boolean-tagged only, matching
/// `expressions.rs`'s `single_numeric_value` (CPython's own
/// `bool`-is-an-`int` reading: `True == 1`).
pub(super) fn single_numeric_value(value: &AbstractValue) -> Option<f64> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Number)
        | Some(PrimitiveKind::Boolean)
        | Some(PrimitiveKind::Integer)
        | Some(PrimitiveKind::Float) => Some(value.values[0]),
        _ => None,
    }
}

/// The code-point vector an AbstractValue carries, if it is a known
/// exact string (`Kind::Values` tagged `PrimitiveKind::String`) —
/// `expressions.rs::exact_string_values`'s own twin, reimplemented
/// locally rather than imported (this file's own "no importing
/// loops.rs" precedent, `generator_yields`'s own doc, applied to
/// expressions.rs's private helper the same way).
pub(super) fn exact_string_values(value: &AbstractValue) -> Option<&[f64]> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(&value.values)
}
