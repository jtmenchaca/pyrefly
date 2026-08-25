//! Container VALUE states: `list`/`tuple`/`dict` literals, subscript
//! reads (`s[i]`, `d[key]`), `len()`, `dict.get`, and the mutation
//! contract (`mutated_receiver`, `dict_with_item`, `list_with_item`)
//! the walk's World calls to thread a write's new receiver value
//! through. Every mutation row answers `None` the moment the receiver
//! or an argument is not fully known — an unknown write is silently
//! dropped only by returning no new state, never guessed at (see
//! `mutated_receiver`'s own doc).
//!
//! ## How the domain carries a container
//!
//! `refined_domain::abstract_value::AbstractValue` has no dedicated
//! tuple variant, and Python's `list`/`tuple` both map to
//! `Kind::List` (`known_constructors::known_list`, "a nested exact
//! sequence") — the same "exact positional slots" shape, indexed the
//! same way, so this module's `tuple_literal_value` is `list_literal_value`
//! under a different name (the TS twin has no tuple either: JS has no
//! tuple type, so `known_constructors.rs` never split one out).
//!
//! `dict` maps to `Kind::Object` (`known_constructors::known_object`,
//! "rooted-keys record") — an ordered `Vec<ObjectKey>` of
//! `{name: String, numeric: bool, value: AbstractValue}` pairs, never
//! a JS-style prototype-bearing map. This is a deliberate choice over
//! `Kind::Collection`/`Flavor::Map` (`abstract_value.rs`): the
//! `Collection`/`Flavor` pair is the TS twin's carry-over for a JS
//! `Map`/`Set` INSTANCE built via `new Map()` — the AGENT-BRIEF's
//! `AbstractValue` fields doc calls it "a built Map or Set" — not for
//! a `{...}` object literal read positionally by name, which is what a
//! Python `dict` LITERAL is. `known_object`'s ordered-`Vec` shape
//! already matches a dict literal exactly, and `pyrefly`'s translated
//! domain has no caller of either constructor yet, so this module is
//! the first to decide the mapping. A `dict` built by a non-literal
//! path (`dict(...)`, a comprehension) is out of this module's scope —
//! only `dict_literal_value` (a literal `{...}` display) is modeled.
//!
//! String- and int-keyed entries: a Python dict key that is a string
//! literal OR a single known `Integer`-sorted value has a slot in
//! `ObjectKey` — `ObjectKey.name` carries the key's spelling (a
//! string's own text, or an int key's plain decimal digits) and
//! `ObjectKey.numeric` tells the two apart (`abstract_value.rs`'s own
//! `ObjectKey` doc: an int key and a string key of the same spelling
//! are DIFFERENT Python dict keys — `1 == "1"` is `False`). Any other
//! key shape (a computed key this module cannot reduce to one of those
//! two sorts, a tuple key, a float/bool key — this domain does not
//! yet fold `1.0`/`True` into the same slot `1` occupies, per
//! stdtypes.rst's "values that compare equal... can be used
//! interchangeably") has no slot to occupy: `dict_literal_value` takes
//! `keys: &[Option<DictKey>]` — `None` at a position means "this key
//! expression is not a supported literal" — and that entire literal
//! answers `unknown()` rather than silently dropping the unsupported
//! entry (dropping would misreport the dict's key set to every later
//! read).
//!
//! `len()` is modeled for known lists/tuples/dicts (their slot/key
//! count) and exact strings (`values.len()`, one code point per
//! `f64` — `string_models.rs`'s documented representation, cited
//! there against library/stdtypes.html's Text Sequence Type section:
//! "Strings are immutable sequences of Unicode code points").
//!
//! ## Coverage cited against the vendored CPython 3.12 docs
//!
//! - Subscription negative-index rule: `Doc/reference/expressions.rst`,
//!   section "Subscriptions" — "built-in sequences all provide a
//!   `__getitem__` method that interprets negative indices by adding
//!   the length of the sequence to the index... The resulting value
//!   must be a nonnegative integer less than the number of items in
//!   the sequence." An index that is still out of range after that
//!   adjustment has no row here: CPython raises `IndexError`, and this
//!   domain carries no exception channel this wave (per the brief) —
//!   `subscript_read` answers `None`, the same "not modeled" honesty
//!   every other decline in this module uses.
//! - Mapping subscription: same section — "the expression list must
//!   evaluate to an object whose value is one of the keys of the
//!   mapping, and the subscription selects the value in the mapping
//!   that corresponds to that key."
//! - `d[key]` on a missing key: `Doc/library/stdtypes.rst`, "Mapping
//!   Types — dict" — "Raises a `KeyError` if key is not in the map."
//!   Again no exception channel this wave, so a missing string key
//!   answers `None` from `subscript_read`, not a fabricated value.
//! - `len(d)`: same section, `describe:: len(d)` — "Return the number
//!   of items in the dictionary d."
//! - `dict.get`: same section, `method:: get(key, default=None, /)` —
//!   "Return the value for key if key is in the dictionary, else
//!   default. If default is not given, it defaults to None, so that
//!   this method never raises a KeyError."

mod dict_key;
mod dict_literal;
mod dict_mutation;
mod dict_write;
mod kernel_join;
mod len_and_get;
mod list_literal;
mod list_set_mutation;
mod list_write;
mod subscript_read;

#[cfg(test)]
mod tests;

pub use dict_key::known_dict_key;
pub use dict_key::DictKey;
pub use dict_literal::dict_literal_value;
pub use dict_write::dict_with_item;
pub use dict_write::dict_without_item;
pub use dict_write::sliced_delete_receiver;
pub(crate) use kernel_join::scalars_of_union_of_singletons;
pub use len_and_get::dict_get_result;
pub use len_and_get::len_result;
pub use list_literal::list_literal_value;
pub use list_literal::tuple_literal_value;
pub use list_write::list_with_item;
pub use subscript_read::subscript_read;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;

use dict_mutation::dict_mutated_receiver as dict_mutated;
use list_set_mutation::list_mutated_receiver;
use list_set_mutation::set_mutated_receiver;

/// A mutating container-method call's (new receiver, call result) pair
/// — the walk's own write channel: `check.rs`/`loops.rs` write the
/// returned receiver back into the environment binding the method was
/// called on, and use the call result the same way any other
/// expression value is used. `None` means "not modeled" (the call is
/// silently NOT threaded as a write — the caller must not assume the
/// receiver is unchanged, matching every other decline in this
/// module); every row below requires the receiver AND every argument
/// fully known, per the mission's own scope — a receiver or argument
/// this module cannot read never answers a guessed write.
///
/// Modeled, each cited against library/stdtypes.rst's own method
/// entry:
/// - list: `append(x)` ("appends *x* to the end of the sequence"),
///   `extend(t)` ("extends *s* with the contents of *t*"),
///   `insert(i, x)` ("inserts *x* into *s* at the index given by *i*"
///   — clamped to `[0, len]`, matching `list.insert`'s own
///   out-of-range-index clamping rather than `IndexError`), `pop()`/
///   `pop(i)` ("retrieves the item at *i* and also removes it from
///   *s*" — no-arg defaults to the LAST item), `clear()` ("removes all
///   items from *s*"), `remove(x)` ("removes the first item from *s*
///   where `s[i]` is equal to *x*" — an ABSENT element declines rather
///   than mutate on the real call's `ValueError`), `sort()` (ascending,
///   known single-numeric elements only), `reverse()` (in place).
/// - set: `add(elem)` ("Add element *elem* to the set"), `discard(elem)`
///   ("Remove element *elem* from the set if it is present" — silent
///   no-op on a miss), `remove(elem)` ("Remove element *elem* from the
///   set. Raises `KeyError` if *elem* is not contained in the set" —
///   an ABSENT element declines the whole call, since the real call
///   raises rather than mutates; `provable_raise` is the raise
///   channel), `update(other)` ("Update the set, adding elements from
///   all others" — the two-arg union-in-place, skipping a duplicate),
///   `clear()`.
/// - dict: `update(other)` ("Update the dictionary with the key/value
///   pairs from *other*, overwriting existing keys" — merges a known
///   dict argument entry by entry), `clear()`, `setdefault(key,
///   default=None)` ("If *key* is in the dictionary, return its
///   value. If not, insert *key* with a value of *default* and return
///   *default*" — the ONE row whose receiver AND call result both
///   change: an absent key both extends the receiver and answers
///   `default`), `pop(key)`/`pop(key, default)` ("If *key* is in the
///   dictionary, remove it and return its value, else return
///   *default*. If *default* is not given and *key* is not in the
///   dictionary, a `KeyError` is raised" — a missing key with no
///   default declines the whole call, matching `set.remove`'s same
///   raise-not-mutate honesty), `popitem()` ("Remove and return a
///   `(key, value)` pair... in LIFO order" — the LAST inserted entry).
///
/// `list.sort()` (no `key=`/`reverse=` keyword arguments) sorts a known
/// list of known single-numeric elements ascending, the same order
/// `builtin_models::sorted_call` already reads for the free function —
/// `list.sort(*, key=None, reverse=False)`: "This method sorts the list
/// in place, using only `<` comparisons between items" (stdtypes.rst).
/// `list.reverse()` reverses a known list's elements in place —
/// stdtypes.rst's Mutable-Sequence-Types table, `s.reverse()`:
/// "reverses the items of *s* in place." Both answer `null_value()` as
/// the call result (neither method returns a value). `list`/`set` share
/// the identical `Kind::List` receiver shape (this module's own doc),
/// so `add`/`discard`/`remove`/`update` on a plain-list receiver
/// also answer through the same rows — this domain has no separate set
/// Kind to gate that on, and the method NAME is the only signal that a
/// call is set-shaped.
pub fn mutated_receiver(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<(AbstractValue, AbstractValue)> {
    match receiver.kind {
        Kind::List => list_mutated_receiver(method, receiver, arguments),
        Kind::Object => dict_mutated(method, receiver, arguments),
        Kind::Set => set_mutated_receiver(method, receiver, arguments),
        _ => None,
    }
}
