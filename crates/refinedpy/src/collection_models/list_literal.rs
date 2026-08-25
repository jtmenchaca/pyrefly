//! The `list`/`tuple` display constructors: both map to `Kind::List`,
//! this domain's one sequence kind — see `collection_models`'s own
//! module doc for why there is no dedicated tuple variant.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::TrustProved;

/// A Python `list` display (`[a, b, c]`): `Kind::List` with one exact
/// slot per element, in source order. `known_list`'s own floor logic
/// already carries a weaker-grade element's trust up to the whole
/// list — this constructor states nothing further about grade.
pub fn list_literal_value(elements: &[AbstractValue]) -> AbstractValue {
    known_list(elements.to_vec(), TrustProved)
}

/// A Python `tuple` display (`(a, b, c)`): the same exact-positional-
/// slots shape a `list` display carries — `Kind::List` is this
/// domain's one sequence kind (module doc: no dedicated tuple
/// variant exists, matching the TS twin, which has no tuple sort to
/// port from either). A one-element tuple `(a,)` and a zero-element
/// tuple `()` both pass through unchanged; the caller's own parse
/// already resolved the trailing-comma/parenthesized-expression
/// grammar before this function sees the element list.
pub fn tuple_literal_value(elements: &[AbstractValue]) -> AbstractValue {
    known_list(elements.to_vec(), TrustProved)
}
