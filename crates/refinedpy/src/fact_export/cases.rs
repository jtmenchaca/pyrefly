//! The RULED cases schema's own unit (CROSS-LANGUAGE-EDGE.md's fact-
//! artifact cases design, JT-approved 2026-08-21): a scalar or object
//! position's whole meaning is a LIST of these cases, never a bare set.
//! This file holds the `Case` type, the reader that builds a cases list
//! from a derived `AbstractValue`, and the provenance-sentence words
//! each case spells.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::lattice_operations::set_of_known;
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::RefinedSet;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;

use crate::assignability::sequence_shaped;
use crate::assignability::states_sequence;

/// One admitted return/entry shape, the RULED cases schema's own unit
/// (CROSS-LANGUAGE-EDGE.md's fact-artifact cases design, JT-approved
/// 2026-08-21): a scalar position's whole meaning is a LIST of these,
/// never a bare set — the wire's `Case := {"sort":"number","set":<wire>}
/// | {"sort":"string","set":<wire>} | {"sort":"boolean"} | {"sort":"null"}
/// | {"sort":"object","members":{...},"closed":bool}`. A single case
/// still spells as a one-element list — one shape, no special casing for
/// "just one."
pub(super) enum Case {
    /// A numeric set, encoded through the SAME kernel wire codec every
    /// other set in this artifact goes through — never a private
    /// artifact subset, which is what lets the full grammar (unions,
    /// oneOf literal sets, multipleOf steps, …) cross for free.
    Number(RefinedSet),
    /// A string set — the identical wire codec, read by the consumer as
    /// the string sort rather than guessed from the set's own shape.
    String(RefinedSet),
    /// The whole boolean sort — a floor case for now (a named future
    /// extension narrows this to an admitted member subset); no `set`
    /// field at all, since there is nothing narrower stated yet.
    Boolean,
    /// Absence — the wire-honest reading of a possibly-undefined value:
    /// what actually crosses the JSON transport for "no value here" is
    /// the bare token `null`, so this case names exactly that rather
    /// than omitting the position or approximating it as a set.
    Null,
    /// An object-shaped value with known member structure — the RULED
    /// object case (CROSS-LANGUAGE-EDGE.md §17, JT-prioritized
    /// 2026-08-21): each declared/known key maps to ITS OWN cases list,
    /// fully recursive (a member can itself be an object case). `closed`
    /// is `true` when the producer states the exact key set (every key
    /// listed, no others possible — a dict literal's own `complete: true`,
    /// or a TypedDict's declared member table); `false` when keys beyond
    /// those listed may exist. A Result-style union (two differently-
    /// shaped branches) is never ONE object case with a `variants` field
    /// of its own — it is TWO object cases riding in the same cases list,
    /// which is what `object_cases_of` builds for a `Kind::Object` value
    /// carrying `variants`.
    Object {
        members: Vec<(String, Vec<Case>)>,
        closed: bool,
    },
}

impl Case {
    pub(super) fn to_json(&self) -> Value {
        match self {
            Case::Number(set) => json!({"sort": "number", "set": refined_kernel::wire_format::wire_set(set)}),
            Case::String(set) => json!({"sort": "string", "set": refined_kernel::wire_format::wire_set(set)}),
            Case::Boolean => json!({"sort": "boolean"}),
            Case::Null => json!({"sort": "null"}),
            Case::Object { members, closed } => {
                let mut members_json = Map::new();
                for (name, cases) in members {
                    members_json.insert(name.clone(), cases_json(cases));
                }
                json!({"sort": "object", "members": Value::Object(members_json), "closed": closed})
            }
        }
    }
}

/// Whether `set` states a string rather than a number — `states_sequence`
/// (a set/`str` DECLARATION always carries its own sequence form at the
/// TOP level) alone misses a DERIVED string whose top form is a `Union`
/// of sequence-shaped branches (`["ok", "warn", "error"][code]`'s own
/// join over a bounded index — three known strings joined pairwise
/// through `join_known`'s string-union path build `Union(Concatenation,
/// Concatenation)` at the top, never a bare `Concatenation` itself), so
/// `sequence_shaped`'s own recursive reading (EVERY top form is itself a
/// sequence form, or a `Union`/`Difference` of two sequence-shaped
/// operands) is checked too. Either test firing is enough — `states_
/// sequence` stays the fast, non-recursive first check for the common
/// case, and `sequence_shaped` catches the union/difference join this
/// export's own new joined-index reads produce.
pub(super) fn is_string_shaped(set: &RefinedSet) -> bool {
    states_sequence(set) || sequence_shaped(set)
}

/// A `RefinedSet` this checker already read as a boundary-narrowed,
/// string-or-numeric scalar, spelled as its one case — `is_string_shaped`
/// tells string from number the same way `foreign_edge.rs`'s own sort
/// laws already do: a scalar position's set can carry a sequence form
/// (`Star`/`Concatenation`/`Repeat`/`RepeatWord`/`EmptyTuple`, possibly
/// under a `Union`/`Difference` join) only when it is a `str`
/// declaration or derivation, since every container spelling
/// (`list[X]`/`set[X]`/`Sequence[X]`) routes to the SEQUENCE entry shape
/// before this function ever sees the set.
pub(super) fn scalar_case_of(set: &RefinedSet) -> Case {
    if is_string_shaped(set) {
        Case::String(set.clone())
    } else {
        Case::Number(set.clone())
    }
}

/// The cases list one derived return value spells — the writer's own
/// half of the RULED schema: a plain numeric/string set is one case; a
/// possibly-absent return (`Kind::PossiblyUndefined`) is the INNER
/// value's own case(s) PLUS a null case, since "a possibly-absent value
/// has no faithful set reading" is exactly the omission this schema
/// retires. An OBJECT-shaped return (`Kind::Object`) reads through
/// `object_cases_of` — one case when the value states a single shape, two
/// (or more, up to the join's own four-arm ceiling) when the value carries
/// `variants` (a Result-style union of differently-shaped branches).
/// Every other unreadable shape (an unknown, an object whose members this
/// table cannot itself read) keeps its named omission exactly as before —
/// this function answers `Err` for those, unchanged from
/// `faithful_return_set`'s own refusal.
pub(super) fn return_cases(returned: &AbstractValue) -> Result<Vec<Case>, String> {
    if returned.kind == Kind::PossiblyUndefined {
        let inner = returned
            .inner
            .as_ref()
            .expect("Kind::PossiblyUndefined always carries an inner value");
        let mut cases = return_cases(inner)?;
        cases.push(Case::Null);
        return Ok(cases);
    }
    if returned.kind == Kind::Null || returned.kind == Kind::Undef {
        return Ok(vec![Case::Null]);
    }
    if returned.kind == Kind::Values && returned.kind_tag == Some(PrimitiveKind::Boolean) {
        return Ok(vec![Case::Boolean]);
    }
    if returned.kind == Kind::Object {
        return object_cases_of(returned);
    }
    let set = faithful_return_set(returned)?;
    let is_string = returned.kind_tag == Some(PrimitiveKind::String) || is_string_shaped(&set);
    Ok(vec![if is_string { Case::String(set) } else { Case::Number(set) }])
}

/// One object-shaped value's own cases list — ONE case for a value
/// stating a single shape, or ONE PER VARIANT (`AbstractValue::variants`,
/// `known_with_variants`'s own field) when the value's derivation joined
/// two or more differently-shaped branches (a Result-style union:
/// `{"ok": true, "value": …}` or `{"ok": false, "error": …}` is exactly
/// TWO object cases riding in the one cases list this function answers,
/// never one case with a nested variants field). The `variants` list
/// always holds full stand-alone `Kind::Object` values (`sides_of`'s own
/// convention in `lattice_operations.rs`), so each is read through
/// `object_case_of` exactly as the un-joined value would be.
pub(super) fn object_cases_of(returned: &AbstractValue) -> Result<Vec<Case>, String> {
    if returned.variants.is_empty() {
        return Ok(vec![object_case_of(returned)?]);
    }
    returned.variants.iter().map(object_case_of).collect()
}

/// One `Kind::Object` value's own single case: each of its known keys
/// mapped to that key's OWN cases list (recursing through `return_cases`
/// so a member that is itself object-shaped becomes a nested object case,
/// and a possibly-absent member reads its inner case(s) plus null exactly
/// as a possibly-absent RETURN does — the two positions share one rule),
/// and `closed` read directly off the value's own `complete` bit — the
/// domain's own completeness fact (`known_object`'s doc: a dict literal
/// always builds `complete: true`, since a literal states every key it
/// has; a join of two differing key sets drops to `complete: false`,
/// since a key present on only one arm may or may not exist on the
/// runtime value). A key whose own value has no cases list this table can
/// read (an unknown, a nested unreadable shape) stops the WHOLE object
/// case — this function answers `Err` naming that key, and the caller's
/// own omission carries the sentence forward exactly as an unreadable
/// scalar return already does. Never guesses at a member it cannot state.
pub(super) fn object_case_of(returned: &AbstractValue) -> Result<Case, String> {
    let mut members = Vec::with_capacity(returned.keys.len());
    for key in &returned.keys {
        let cases = return_cases(&key.value).map_err(|reason| {
            format!("its member '{}' is {reason}, which has no faithful cases reading", key.name)
        })?;
        members.push((key.name.clone(), cases));
    }
    Ok(Case::Object {
        members,
        closed: returned.complete,
    })
}

/// The cases list as the artifact spells it — a JSON array, never a
/// bare set, so a single case still reads as the one-element list the
/// schema requires.
pub(super) fn cases_json(cases: &[Case]) -> Value {
    Value::Array(cases.iter().map(Case::to_json).collect())
}

/// The derived return read as a set, or the reason it has no faithful
/// reading. `set_of_known` is the one converter — an object, an unknown,
/// a nested sequence answers `None` there, and this states which.
pub(super) fn faithful_return_set(
    returned: &AbstractValue,
) -> Result<RefinedSet, String> {
    if let Some(set) = set_of_known(returned) {
        if set.forms.is_empty() {
            return Err("the derived return is the empty set, which states no crossable fact".to_owned());
        }
        return Ok(set);
    }
    Err(format!(
        "the derived return is {}, which has no faithful set reading",
        return_kind_words(returned)
    ))
}

/// Plain words for what a return derived to, for the omission row.
pub(super) fn return_kind_words(returned: &AbstractValue) -> &'static str {
    // a set-kinded value only reaches here when it wears a sort tag —
    // `set_of_known` answers Some for an untagged one
    if returned.kind == Kind::Set {
        return "a set of values whose members are not plain numbers";
    }
    match returned.kind {
        Kind::Unknown => "a value this walk never determined",
        Kind::Object | Kind::ObjectStar => "an object",
        Kind::List | Kind::Collection => "a nested sequence",
        Kind::Promise => "an awaitable",
        Kind::Date => "a date",
        Kind::Symbol => "a symbol",
        Kind::HostFunction => "a function",
        Kind::Bigints => "an arbitrary-width integer",
        Kind::Regex => "a regular expression",
        Kind::Undef | Kind::Null => "the absent value",
        Kind::NaN => "NaN",
        Kind::PossiblyUndefined => "a possibly-absent value",
        Kind::PossiblyNaN => "a possibly-NaN value",
        Kind::KindUnion => "a union of sorts",
        Kind::ArrayHoles => "a sequence of holes",
        // the empty tuple is the one Values shape with no set spelling
        Kind::Values => "the empty tuple",
        // handled above, and answered by set_of_known otherwise
        Kind::Set | Kind::Variable => "a value with no set reading",
    }
}

/// One case's own words, for a provenance sentence — `format_for_
/// diagnostics` for the two set-carrying cases, plain words for the two
/// that carry none, and each member's own words (recursively) for an
/// object case.
pub(super) fn case_words(case: &Case) -> String {
    match case {
        Case::Number(set) | Case::String(set) => format_for_diagnostics(set),
        Case::Boolean => "a boolean".to_owned(),
        Case::Null => "absent".to_owned(),
        Case::Object { members, closed } => {
            let member_words: Vec<String> = members
                .iter()
                .map(|(name, cases)| format!("'{name}' is {}", cases_words(cases)))
                .collect();
            let openness = if *closed { "no other keys" } else { "possibly other keys" };
            format!("an object whose {} ({openness})", member_words.join(" and "))
        }
    }
}

/// A cases list's own words — one case reads as its own words; more than
/// one joins with "or", the plain reading of "this position is one of
/// these cases."
pub(super) fn cases_words(cases: &[Case]) -> String {
    cases.iter().map(case_words).collect::<Vec<_>>().join(" or ")
}
