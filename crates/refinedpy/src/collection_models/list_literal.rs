//! The `list`/`tuple` display constructors: both map to `Kind::List`,
//! this domain's one sequence kind — see `collection_models`'s own
//! module doc for why there is no dedicated tuple variant.

use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
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

/// The counter `with_referent_identities` draws from. It starts above
/// the range `instances::judge_construction` counts through from zero,
/// so a container's referent id and a class instance's construction id
/// share the one `AbstractValue::instance_identity` field without ever
/// colliding.
static NEXT_REFERENT_IDENTITY: AtomicU32 = AtomicU32::new(1 << 24);

/// Stamps every CONTAINER element of a display with a per-construction
/// referent identity (`AbstractValue::instance_identity`), which is what
/// makes a SHALLOW COPY of a nested container observable through both
/// names.
///
/// `outer = [[1, 2]]` builds one inner list object; `copy = outer[:]`
/// builds a NEW outer list whose slot 0 holds that SAME inner object
/// (library/copy.rst states the sharing outright — "A shallow copy
/// constructs a new compound object and then ... inserts *references*
/// into it to the objects found in the original"; library/stdtypes.rst's
/// "Common Sequence Operations" gives `s[i:j]` as the slice that
/// performs it). This domain represents a list BY VALUE, so a slice
/// clones the item values and a later write through `copy[0][0]` would
/// be invisible at `outer[0][0]`. The identity is what closes that gap:
/// the clone carries the SAME id, so the write path
/// (`check::bind::destructure::bind_or_forget_subscript_target`) can
/// find every other binding holding an item with that id and bring it
/// back in step, exactly as `Environment::rebind_aliases_of_instance`
/// already does for a class instance written through an alias name.
///
/// Only the ELEMENTS are stamped, and only the container-shaped ones: a
/// scalar element is copied by value in the real interpreter too, and
/// the display's own outer value is a fresh object nothing else can yet
/// hold a second name for. An element already carrying an identity (a
/// name read back into a display, `[inner]` where `inner` was itself
/// built as a display) keeps the id it already has — that is the same
/// object, and re-stamping it would break the very sharing this
/// establishes.
pub fn with_referent_identities(elements: Vec<AbstractValue>) -> Vec<AbstractValue> {
    elements
        .into_iter()
        .map(|element| {
            if element.instance_identity.is_some() || !matches!(element.kind, Kind::List | Kind::Object) {
                return element;
            }
            AbstractValue {
                instance_identity: Some(NEXT_REFERENT_IDENTITY.fetch_add(1, Ordering::Relaxed)),
                ..element
            }
        })
        .collect()
}
