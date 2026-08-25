//! `list[index] = value` — the list write channel.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;

use super::list_literal::list_literal_value;
use super::subscript_read::known_integer_index;

/// `list[index] = value` — the written-through list, known shapes
/// only: a known `Kind::List` receiver and a known Integer index that
/// (after the same negative-index adjustment `list_index_read` reads
/// by) lands inside the list's current bounds (expressions.rst,
/// "Subscriptions" — item assignment on a sequence follows the same
/// negative-index rule as a read; an index past the end raises
/// `IndexError`, which this domain has no channel for, so it declines
/// rather than silently extending the list the way `append` would).
///
/// Carries `receiver`'s own `kind_word` forward onto the written-through
/// list — a bytes-like receiver (`bytes_models::tagged`'s own species
/// word) stays tagged after a write that mutated its contents, so a
/// SECOND write to the same name still reads which of the three
/// bytes-like write rules applies rather than losing the tag the moment
/// this function rebuilds the list.
pub fn list_with_item(receiver: &AbstractValue, index: &AbstractValue, value: &AbstractValue) -> Option<AbstractValue> {
    if receiver.kind != Kind::List {
        return None;
    }
    let position = known_integer_index(index)?;
    let length = receiver.items.len() as i64;
    let adjusted = if position < 0 { position + length } else { position };
    if adjusted < 0 || adjusted >= length {
        return None;
    }
    let mut items = receiver.items.clone();
    items[adjusted as usize] = value.clone();
    let mut written = list_literal_value(&items);
    written.kind_word = receiver.kind_word;
    Some(written)
}
