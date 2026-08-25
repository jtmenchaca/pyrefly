//! `list`/`set` mutating method calls: `append`/`extend`/`insert`/
//! `pop`/`clear`/`sort`/`reverse` on a `Kind::List` receiver, PLUS the
//! set-only method names `add`/`discard`/`remove`/`update` (a set
//! shares the same `Kind::List` receiver shape — `collection_models`'s
//! own module doc), and `append`/`extend` on a REPETITION-SHAPED
//! `Kind::Set` receiver (the `list[X]`/`set[X]`/`Sequence[X]` parameter
//! seed's own star shape, which has no concrete items to index into).
//! See `mutated_receiver`'s own doc (in `collection_models/mod.rs`) for
//! the cited row-by-row contract.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::join_known;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustLevel;
use refined_sets::repetition_window_forms::as_repetition;

use super::list_literal::list_literal_value;
use super::subscript_read::known_integer_index;
use super::subscript_read::list_index_read;

/// `list.append`/`extend`/`insert`/`pop`/`clear`, PLUS the set-only
/// method names `add`/`discard`/`remove`/`update` — see
/// `mutated_receiver`'s own doc for the cited row-by-row contract. Both
/// families dispatch through this one function because a set and a
/// list share the identical `Kind::List` receiver shape in this domain
/// (this file's own module doc) — there is no separate set Kind to
/// route on, so the METHOD NAME alone tells a set call apart from a
/// list call, and both live in the same match.
pub(super) fn list_mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
    match method {
        "append" => {
            let [element] = arguments else { return None };
            let mut items = receiver.items.clone();
            items.push(element.clone());
            Some((list_literal_value(&items), null_value()))
        }
        "extend" | "update" => {
            let [other] = arguments else { return None };
            if other.kind != Kind::List {
                return None;
            }
            let mut items = receiver.items.clone();
            for candidate in &other.items {
                // `update`'s own set-union-in-place semantics skip a
                // duplicate; `extend`'s own list semantics do not — the
                // method name itself decides which rule applies
                if method == "update" && element_contains(&items, candidate)? {
                    continue;
                }
                items.push(candidate.clone());
            }
            Some((list_literal_value(&items), null_value()))
        }
        "insert" => {
            let [index, element] = arguments else { return None };
            let position = known_integer_index(index)?;
            let length = receiver.items.len() as i64;
            // out-of-range clamps to the nearest end rather than raising
            // (stdtypes.rst's `s.insert(i, x)` row states no bounds check,
            // matching CPython's own clamp-not-raise behavior)
            let clamped = if position < 0 {
                (length + position).max(0)
            } else {
                position.min(length)
            } as usize;
            let mut items = receiver.items.clone();
            items.insert(clamped, element.clone());
            Some((list_literal_value(&items), null_value()))
        }
        "pop" if arguments.is_empty() => {
            let popped = receiver.items.last().cloned()?;
            let mut items = receiver.items.clone();
            items.pop();
            Some((list_literal_value(&items), popped))
        }
        "pop" => {
            let [index] = arguments else { return None };
            let position = known_integer_index(index)?;
            let popped = list_index_read(&receiver.items, position)?;
            let length = receiver.items.len() as i64;
            let adjusted = if position < 0 { position + length } else { position } as usize;
            let mut items = receiver.items.clone();
            items.remove(adjusted);
            Some((list_literal_value(&items), popped))
        }
        "clear" if arguments.is_empty() => Some((list_literal_value(&[]), null_value())),
        // set.add(elem) — "Add element *elem* to the set." A duplicate
        // (already-present) element is a silent no-op (set membership,
        // not list append).
        "add" => {
            let [element] = arguments else { return None };
            if element_contains(&receiver.items, element)? {
                return Some((receiver.clone(), null_value()));
            }
            let mut items = receiver.items.clone();
            items.push(element.clone());
            Some((list_literal_value(&items), null_value()))
        }
        // set.discard(elem) — "Remove element *elem* from the set if it
        // is present." A MISSING element is a silent no-op (unlike
        // `remove`, which raises on a miss). `remove_first_element`
        // removes only the FIRST match, which is exactly "the one
        // occurrence" for a set (no duplicates by construction) and
        // also the correct `list.remove`/`list.discard`-shaped
        // behavior if this receiver happens to be a plain list with
        // duplicate elements.
        "discard" => {
            let [element] = arguments else { return None };
            let items = remove_first_element(&receiver.items, element)?;
            Some((list_literal_value(&items), null_value()))
        }
        // `list.remove(x)`/`set.remove(elem)` — stdtypes.rst's
        // Mutable-Sequence-Types table: "removes the first item from
        // *s* where `s[i]` is equal to *x*"; the set section: "Remove
        // element *elem* from the set. Raises KeyError if *elem* is not
        // contained in the set." An ABSENT element declines the whole
        // call rather than mutate on a raise (`provable_raise` is the
        // raise channel, not this function) — sound for BOTH receiver
        // shapes, since a list `.remove` on a missing element raises
        // `ValueError` the same way a set `.remove` raises `KeyError`.
        "remove" => {
            let [element] = arguments else { return None };
            if !element_contains(&receiver.items, element)? {
                return None;
            }
            let items = remove_first_element(&receiver.items, element)?;
            Some((list_literal_value(&items), null_value()))
        }
        // `list.sort(*, key=None, reverse=False)` — the no-keyword-argument
        // default row: ascending order, only `<` comparisons over known
        // single-numeric elements (stdtypes.rst's own method entry).
        // `key=`/`reverse=` keyword arguments never reach `arguments` at
        // all (both statement-position callers of `mutated_receiver` —
        // `check.rs::walk_mutating_call_statement`, `summaries.rs`'s own
        // mutating-call arm — collect POSITIONAL call arguments only), so
        // this same guard already covers `lst.sort(key=lambda x: x)`: the
        // real call's keyword is dropped before this file ever sees it,
        // and `key=lambda x: x` is Python's own identity key, which sorts
        // by the SAME order the no-key row already answers.
        //
        // `sorted_numeric_items` needs every element to be one known
        // EXACT scalar; a list of WINDOWS (`a`/`b`/`c` each guarded to
        // `[0, 9]`, still a bounded `Kind::Set`, never narrowed to one
        // value) falls through to `sorted_membership_preserving_items`
        // instead — see that function's own doc for why every slot
        // answers the JOIN of every element's window (the only per-slot
        // claim sound without a real order), not each element's own
        // pre-sort window.
        "sort" if arguments.is_empty() => {
            if let Some(sorted_items) = sorted_numeric_items(&receiver.items) {
                return Some((list_literal_value(&sorted_items), null_value()));
            }
            let same_items = sorted_membership_preserving_items(&receiver.items)?;
            Some((list_literal_value(&same_items), null_value()))
        }
        // `list.reverse()` — "reverses the items of *s* in place"
        // (stdtypes.rst's Mutable-Sequence-Types table, `s.reverse()`).
        "reverse" if arguments.is_empty() => {
            let mut items = receiver.items.clone();
            items.reverse();
            Some((list_literal_value(&items), null_value()))
        }
        _ => None,
    }
}

/// `set.append(x)`/`set.extend(iterable)` on a REPETITION-SHAPED receiver
/// (`Kind::Set` whose own set is the bare star/window `as_repetition`
/// reads back — `star_element_read`'s own doc, the shape
/// `check.rs::seed_parameters` seeds for a `list[X]`/`set[X]`/
/// `Sequence[X]` parameter with no concrete items). There is no
/// receiver-side count to index into (unlike `list_mutated_receiver`'s
/// `Kind::List`, which has exact items to push/remove), so a mutation
/// widens the window's OWN {lo, hi} bounds instead of pushing an item:
///
/// - `append(x)`: the element set joins with `x`'s own set (the same
///   lattice join `join_repetition_sets` — lattice_operations.rs — takes
///   when two repetition sets union), and the count grows by exactly
///   one: `lo + 1`, `hi + 1` when `hi` is finite, or unbounded when it
///   already was (one more element never turns an unbounded window
///   bounded).
/// - `extend(iterable)`: the element set joins with the iterable's own
///   element set (a `Kind::List` argument's items, each folded in by
///   `join_known`, or a `Kind::Set` repetition argument's own element),
///   and the iterable's own COUNT WINDOW adds onto `[lo, hi]`: `lo` sums
///   (the iterable contributes at least its own minimum), `hi` sums when
///   both sides are finite, else unbounded.
///
/// Every other method name declines (`None`) — this receiver shape has
/// no concrete slot to `pop`/`insert`/`sort` against, and `add`/
/// `discard`/`remove`/`clear` have no way to test or drop one member of
/// a window that states no items, only a shape.
pub(super) fn set_mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
    if receiver.set_kind_tag != SetKindTag::None {
        return None;
    }
    let window = as_repetition(&receiver.set)?;
    let grade = trust_level_of(receiver);
    match method {
        "append" => {
            let [element] = arguments else { return None };
            let element_set = joined_element_set(window.element.clone(), element, grade)?;
            let hi = window.hi.map(|h| h + 1);
            Some((
                repetition_receiver(receiver, element_set, window.lo + 1, hi, grade),
                null_value(),
            ))
        }
        "extend" => {
            let [iterable] = arguments else { return None };
            let (iterable_element, iterable_lo, iterable_hi) = count_window_of(iterable)?;
            // An iterable PROVABLY EMPTY (hi == Some(0), so lo == 0 too)
            // contributes no element claim at all — joining against it
            // would falsely widen the receiver's own element set with
            // whatever placeholder an empty argument reads as.
            let element_set = if iterable_hi == Some(0) {
                window.element.clone()
            } else {
                joined_element_set(window.element.clone(), &iterable_element, grade)?
            };
            let lo = window.lo + iterable_lo;
            let hi = match (window.hi, iterable_hi) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            };
            Some((
                repetition_receiver(receiver, element_set, lo, hi, grade),
                null_value(),
            ))
        }
        _ => None,
    }
}

/// The receiver's own element set joined with one new element's set —
/// the same lattice join `join_repetition_sets` (lattice_operations.rs)
/// takes when two repetition windows union, applied here to one element
/// at a time via `known_set`/`join_known`/`set_of_known` rather than
/// hand-rolled union arithmetic. `None` the moment the join keeps no
/// tuple-layer set (`set_of_known`'s own refusal — an object/list
/// element, say), matching every other decline in this file.
fn joined_element_set(
    element: refined_sets::refinement_forms::RefinedSet,
    new_element: &AbstractValue,
    grade: TrustLevel,
) -> Option<refined_sets::refinement_forms::RefinedSet> {
    let element_value = known_set(element, None, grade, SetKindTag::None);
    let joined = join_known(element_value, new_element.clone());
    set_of_known(&joined)
}

/// An iterable argument's own (element set, lo, hi) count window —
/// `extend`'s own "add the iterable's count window" needs the SAME
/// shape `len_result` already reads for a receiver, applied here to an
/// ARGUMENT instead: a `Kind::List` argument's element set is every item
/// joined together (`join_object_star_with_list`'s own fold,
/// lattice_operations.rs) with an EXACT count `[len, len]`; a `Kind::Set`
/// repetition argument reads back its own `{element, lo, hi}` directly
/// via `as_repetition`. Every other argument shape declines.
fn count_window_of(iterable: &AbstractValue) -> Option<(AbstractValue, i64, Option<i64>)> {
    match iterable.kind {
        Kind::List => {
            let count = iterable.items.len() as i64;
            // An EMPTY list contributes no element claim and no count —
            // the same vacuous-star reading `join_object_star_with_list`
            // takes for a zero-length array (lattice_operations.rs):
            // there is no item to fold in, so the receiver's own element
            // set is untouched rather than joined against a fabricated
            // placeholder.
            let Some((first, rest)) = iterable.items.split_first() else {
                return Some((refined_domain::abstract_value::unknown(), 0, Some(0)));
            };
            let mut element = first.clone();
            for item in rest {
                element = join_known(element, item.clone());
            }
            Some((element, count, Some(count)))
        }
        Kind::Set if iterable.set_kind_tag == SetKindTag::None => {
            let window = as_repetition(&iterable.set)?;
            let grade = trust_level_of(iterable);
            let element_value = AbstractValue {
                kind_tag: iterable.kind_tag,
                ..known_set(window.element, None, grade, SetKindTag::None)
            };
            Some((element_value, window.lo, window.hi))
        }
        _ => None,
    }
}

/// The mutated receiver `append`/`extend` both build: the same
/// repetition shape (`refined_sets::repetition_window_forms::
/// repetition`), the receiver's own `kind_tag` carried across
/// (`star_element_read`'s own convention — a numeric-tagged window stays
/// numeric-tagged after a widening mutation), at the joined grade.
fn repetition_receiver(
    receiver: &AbstractValue,
    element_set: refined_sets::refinement_forms::RefinedSet,
    lo: i64,
    hi: Option<i64>,
    grade: TrustLevel,
) -> AbstractValue {
    AbstractValue {
        kind_tag: receiver.kind_tag,
        ..known_set(
            refined_sets::repetition_window_forms::repetition(element_set, lo, hi),
            None,
            grade,
            SetKindTag::None,
        )
    }
}

/// `items` sorted ascending by numeric value, or `None` the moment one
/// element is not a single known Integer/Float/Boolean-sorted value —
/// the same "known numeric elements only" acceptance
/// `builtin_models::sorted_call` reads for the free `sorted()` function,
/// repeated here rather than reaching across the crate boundary for one
/// small helper (this file owns no dependency on `builtin_models.rs`).
fn sorted_numeric_items(items: &[AbstractValue]) -> Option<Vec<AbstractValue>> {
    let mut pairs: Vec<(f64, AbstractValue)> = Vec::with_capacity(items.len());
    for element in items {
        if element.kind != Kind::Values {
            return None;
        }
        if element.values.len() != 1 {
            return None;
        }
        if !matches!(
            element.kind_tag,
            Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean)
        ) {
            return None;
        }
        pairs.push((element.values[0], element.clone()));
    }
    // A NaN element makes every comparison false, so CPython's sort
    // produces an order no law states — a NaN-admitting list yields no
    // order claim, and this reader declines rather than fabricate one
    // (float("nan") is a value builtin_models::float_call constructs).
    if pairs.iter().any(|(value, _)| value.is_nan()) {
        return None;
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("NaN elements declined above"));
    Some(pairs.into_iter().map(|(_, value)| value).collect())
}

/// `list.sort()` over elements this file cannot place into an EXACT
/// per-slot order — `sorted_numeric_items`'s own decline, e.g. a WINDOW
/// element (`Kind::Set`, a guarded parameter's own `[0, 9]` bound, never
/// narrowed to one value). `<` over two unresolved windows has no answer
/// this file can prove (either window could sit first), so no PER-SLOT
/// ordering claim is made — unlike an EXACT sort, a real run can move
/// ANY window to ANY position (a window with a low lower bound but a
/// high upper bound can still sort last if its actual runtime value
/// turns out large), so answering slot `i` with element `i`'s own
/// PRE-sort window would be unsound the moment two windows overlap or
/// invert: `[100, 200]` sitting before `[0, 50]` pre-sort must sort
/// AFTER it, and keeping `[100, 200]` at slot 0 would misstate the real
/// post-sort value there. The only per-slot claim every position is
/// PROVABLY inside, regardless of which permutation CPython's `<`
/// chooses, is the JOIN of every element's own window — every slot
/// still holds exactly length `items.len()` elements (`list.sort()`
/// never loses or adds one), each one somewhere inside that same join,
/// so filling every slot with the join is sound (if looser than the
/// per-slot answer this file cannot prove). Every element must still be
/// NUMERIC-sorted (`<` is well-defined at runtime only between two
/// numbers, stdtypes.rst's own `sort()` entry: "using only `<`
/// comparisons between items") — a single known scalar (the same
/// acceptance `sorted_numeric_items` gives) OR a numeric-tagged
/// `Kind::Set` window (`Integer`/`Float`/`Boolean`) both qualify; any
/// other element shape (a string, a nested list, an opaque value)
/// declines the whole call, matching `sorted_numeric_items`'s own
/// all-or-nothing posture.
fn sorted_membership_preserving_items(items: &[AbstractValue]) -> Option<Vec<AbstractValue>> {
    let mut joined: Option<AbstractValue> = None;
    for element in items {
        let is_numeric_scalar = element.kind == Kind::Values
            && element.values.len() == 1
            && matches!(
                element.kind_tag,
                Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean)
            );
        let is_numeric_window = element.kind == Kind::Set
            && element.set_kind_tag == SetKindTag::None
            && matches!(
                element.kind_tag,
                Some(PrimitiveKind::Integer) | Some(PrimitiveKind::Float) | Some(PrimitiveKind::Boolean)
            );
        if !is_numeric_scalar && !is_numeric_window {
            return None;
        }
        joined = Some(match joined {
            None => element.clone(),
            Some(so_far) => join_known(so_far, element.clone()),
        });
    }
    let joined = joined?;
    Some(vec![joined; items.len()])
}

/// Whether `needle` is a member of `items` by exact-value equality —
/// scalar values (`Kind::Values`) compare by their `values`/`kind_tag`
/// pair; every other shape declines (`None`) rather than guess at
/// equality for a shape this file has no comparison row for. This is
/// the SAME membership question `expressions.rs`'s own `set_contains`
/// answers for the read-side set methods, kept as a separate small copy
/// here rather than reaching across the module boundary for one helper
/// (this file owns no dependency on `expressions.rs`, and adding one
/// would invert the existing `expressions.rs -> collection_models.rs`
/// direction into a cycle).
fn element_contains(items: &[AbstractValue], needle: &AbstractValue) -> Option<bool> {
    // an EMPTY collection contains nothing, regardless of the needle's
    // own shape — this is trivially true by the definition of
    // membership, so a needle this file otherwise cannot compare
    // equality for (e.g. a class instance, `Kind::Object`) still
    // answers `false` against an empty receiver rather than declining
    // (weakref.WeakSet's own `bag.add(key)` on a freshly-built empty
    // set, `expressions.rs`'s corpus this function serves).
    if items.is_empty() {
        return Some(false);
    }
    if needle.kind != Kind::Values {
        return None;
    }
    for element in items {
        if element.kind != Kind::Values {
            return None;
        }
        if element.kind_tag != needle.kind_tag {
            continue;
        }
        if element.values == needle.values {
            return Some(true);
        }
    }
    Some(false)
}

/// `items` with the FIRST element EQUAL to `needle` removed — correct
/// for a set (there is at most one match, no duplicates by
/// construction) and for a plain list's own `.remove`/`.discard`
/// semantics ("removes the first item... where `s[i]` is equal to
/// *x*," stdtypes.rst's Mutable-Sequence-Types table). `None` the
/// moment `element_contains`'s own equality question cannot be decided
/// for some element scanned before the match.
fn remove_first_element(items: &[AbstractValue], needle: &AbstractValue) -> Option<Vec<AbstractValue>> {
    if needle.kind != Kind::Values {
        return None;
    }
    let mut kept = Vec::with_capacity(items.len());
    let mut removed_one = false;
    for element in items {
        if element.kind != Kind::Values {
            return None;
        }
        if !removed_one && element.kind_tag == needle.kind_tag && element.values == needle.values {
            removed_one = true;
            continue;
        }
        kept.push(element.clone());
    }
    Some(kept)
}
