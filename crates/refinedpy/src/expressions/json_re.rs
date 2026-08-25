
use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::codepoint_sets::strings;
use ruff_python_ast::Expr;

use crate::string_models;

use super::arithmetic::*;
use super::call::*;
use super::compare::*;
use super::fstring::*;

/// `re.search(pattern, subject)` reduced to a substring test — modeled
/// ONLY when `pattern` is a known exact string with no regex
/// metacharacter (`is_literal_regex_pattern`) and `subject` is a known
/// exact string (`evaluate_attribute_call`'s own `search` call site
/// doc). A pattern found IN the subject answers the match-object sort
/// (`opaque_value`, the same over-approximation `re.match` already
/// gives); an ABSENT pattern answers the exact `None` `re.search`
/// documents for "no position in the string matches the pattern."
pub(super) fn re_search_literal_value(pattern: &AbstractValue, subject: &AbstractValue) -> Option<AbstractValue> {
    if !is_literal_regex_pattern(pattern) {
        return None;
    }
    let pattern_text = exact_string_values(pattern).and_then(code_points_to_string)?;
    let subject_text = exact_string_values(subject).and_then(code_points_to_string)?;
    if subject_text.contains(&pattern_text) {
        Some(opaque_value("a match object"))
    } else {
        Some(null_value())
    }
}

/// A JSON SCALAR literal's exact Python value — library/json.rst's own
/// JSON-to-Python conversion table (`evaluate_attribute_call`'s `loads`
/// call site doc): `null` -> `None`, `true`/`false` -> `True`/`False`,
/// a quoted string -> `str` (no escape-sequence decoding — this file
/// only reads the corpus's own escape-free literals), a bare integer
/// literal -> `int`, any other numeric spelling -> `float`. Only the
/// SCALAR productions are parsed — a `[`/`{`-leading text (an array or
/// object) declines, matching this file's own "the corpus's rows never
/// need array/object parsing" scope note.
pub(super) fn json_scalar_literal_value(text: &str) -> Option<AbstractValue> {
    if text == "null" {
        return Some(null_value());
    }
    if text == "true" {
        return Some(known_values(vec![1.0], PrimitiveKind::Boolean, TrustProved));
    }
    if text == "false" {
        return Some(known_values(vec![0.0], PrimitiveKind::Boolean, TrustProved));
    }
    if text.len() >= 2 && text.starts_with('"') && text.ends_with('"') {
        return Some(string_models::string_literal_value(&text[1..text.len() - 1]));
    }
    if let Ok(value) = text.parse::<i64>() {
        return Some(known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved));
    }
    if let Ok(value) = text.parse::<f64>() {
        return Some(known_values(vec![value], PrimitiveKind::Float, TrustProved));
    }
    None
}

/// `json.loads`'s full return space over an operand this file holds no
/// fact about (ISSUES.md, "generic json.loads of an opaque string
/// answers bare unknown") — library/json.rst's own conversion table,
/// read as ONE honest claim rather than the narrower Float-sorted guess
/// the survey rejected as unsound (a real payload can land on any of
/// the table's rows, and a Float-only claim is false on every other
/// row). `PrimitiveKind::Integer`/`Float` split the JSON `number`
/// production (CPython: `json.loads("1")` is `int`, `json.loads("1.5")`
/// is `float` — `json_scalar_literal_value`'s own doc), so each numeric
/// sort narrows on its own via the ordinary Integer/Float narrowing and
/// judging paths, rather than folding both under the sort-unknown
/// `PrimitiveKind::Number` tag that `isinstance`/`judge` cannot yet
/// place on either side of a test. `str`/`list`/`dict`/`bool`/`None`
/// each carry their own sort, so a downstream `isinstance` or judge
/// call can still tell them apart from the numeric arms — a `list`/
/// `dict` arm is built via `opaque_value` (this file's own established
/// "the kind of thing is known, its contents are not" shape, e.g. the
/// `re.match` result above) rather than an exact-arity `known_list([])`/
/// `known_object([])`, which would falsely claim the parsed value is
/// EMPTY.
pub(super) fn json_loads_value_space() -> AbstractValue {
    kind_union_of(vec![
        null_value(),
        known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec),
        known_set(strings(), None, TrustSpec, SetKindTag::None),
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(eval_whole_integers(), None, TrustSpec, SetKindTag::None)
        },
        float_sorted_unknown(),
        opaque_value("a list"),
        opaque_value("a dict"),
    ])
}

/// `json.dumps(obj)`'s exact serialized text — library/json.rst's own
/// Python-to-JSON conversion table, default `separators = (', ', ':
/// ')` (`evaluate_attribute_call`'s `dumps` call site doc). Recurses
/// into a known `Kind::Object`'s own values (a nested dict); every
/// OTHER value shape this function cannot serialize (Float, a list, an
/// unknown value) makes the WHOLE call decline, matching this file's
/// "no partial answer" discipline for every other multi-part
/// composition (the f-string's own `has_exact` tier, `dict_literal_value`'s
/// own all-keys-must-parse rule). String quoting borrows Rust's own
/// `Debug` escaping (`format!("{:?}", text)`) rather than a hand-rolled
/// JSON-escape table — exact for the plain-ASCII, no-control-character
/// strings this corpus's own rows use; a string carrying a character
/// JSON and Rust's `Debug` escape differently (e.g. a lone surrogate,
/// or JSON's `\/` convention) is a known gap this file does not close.
pub(super) fn json_dumps_value(value: &AbstractValue) -> Option<String> {
    if let Some(text) = exact_string_values(value).and_then(code_points_to_string) {
        return Some(format!("{:?}", text));
    }
    if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(value) {
        return Some(format!("{}", number as i64));
    }
    if value.kind == Kind::Object {
        let mut parts = Vec::with_capacity(value.keys.len());
        for entry in &value.keys {
            let serialized_value = json_dumps_value(&entry.value)?;
            parts.push(format!("{:?}: {}", entry.name, serialized_value));
        }
        return Some(format!("{{{}}}", parts.join(", ")));
    }
    None
}

/// Every module name `evaluate_attribute_call` carries a model for, at
/// least in part — the recognized-module gate every arm in that function
/// already applies one at a time (`module_name.id.as_str() == "math"`,
/// `== "random"`, and so on). Named here as ONE list so a recognizer that
/// needs the COMPLEMENT (rung 1's naming unit, and the manifest reader's
/// own "is this module already modeled here?" check,
/// `python-c-extension-boundary.md`'s build order) reads one table
/// instead of re-deriving it from the arms below. `datetime`'s own three
/// aliases (`date`/`timedelta`) are matched by IDENTITY through
/// `environment.datetime_imports()`, not by this literal list, so they
/// are named here too even though no arm below tests
/// `module_name.id.as_str() == "datetime"` directly.
pub(super) const MODELED_MODULE_NAMES: &[&str] = &[
    "math", "random", "re", "json", "importlib", "types", "weakref", "asyncio", "array", "subprocess", "datetime", "os", "time", "unicodedata", "base64", "struct",
];

/// The leftmost `Name` under an attribute-chain receiver (`a.b.c` → `a`;
/// `a` itself → `a`) — `None` when the receiver is not built from a
/// plain name chain at all. The expression-side twin of `check.rs`'s own
/// `receiver_base_name` (private to that file), duplicated rather than
/// exported across the crate boundary this module already keeps thin —
/// both copies read the identical two-line recursion.
pub(super) fn attribute_chain_root_name(receiver: &Expr) -> Option<&str> {
    match receiver {
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Attribute(attribute) => attribute_chain_root_name(attribute.value.as_ref()),
        _ => None,
    }
}
