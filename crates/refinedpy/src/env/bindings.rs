//! Name-keyed bindings: recording, reading, and forgetting what a name
//! holds, plus aliasing rebinds across instance identity.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;

use crate::collection_models::DictKey;

use super::tracked_place::TrackedPlace;
use super::Environment;

impl Environment {
    /// Record what a name holds after a statement the walk understood.
    ///
    /// A REBIND of `name` also strips any BINDING-KEYED presence entry
    /// naming `name` from every OTHER `Kind::ObjectStar` receiver this
    /// environment holds (`narrowing::compare::narrow_dict_membership_
    /// against_literal_key`'s own doc for what such an entry is and why
    /// it exists): `key in m` records that fact by `key`'s own binding
    /// identity, valid only while neither `m` nor `key` is written
    /// between the guard and a later `m[key]` read. A write to `m`
    /// itself already drops the fact — this call REPLACES `m`'s whole
    /// value, the recorded entry along with it — but a write to `key`
    /// touches a DIFFERENT binding's slot, so `m`'s own entry would
    /// otherwise survive untouched and answer a lookup for a runtime
    /// object `key` no longer names. This is `bind`'s own half of the
    /// same discipline `forget`'s doc states for the ordinary case: the
    /// one universal write chokepoint every rebind (successful or not,
    /// `forget` and `bind` between them) routes through, so no call site
    /// has to remember to invalidate this fact itself.
    ///
    /// Scoped to a plain rebind of `name`: rebinding `m` itself needs no
    /// special case here (its own slot is simply overwritten, taking any
    /// entries with it), so this only ever removes entries from OTHER
    /// bindings, never `name`'s own just-written slot.
    pub fn bind(&mut self, name: &str, value: AbstractValue) {
        self.forget_binding_keyed_star_entries(name);
        self.bindings.insert(name.to_owned(), value);
        if crate::trace::is_tracing() {
            crate::trace::record_bind_touch(name);
        }
    }

    /// Strips every `ObjectKey` entry tagged as `name`'s own binding
    /// identity (`DictKey::identity(&format!("binding:{name}"))`,
    /// `bind`'s own doc) from every `Kind::ObjectStar` value this
    /// environment currently binds — the cross-binding half of
    /// invalidating a membership guard's fact when the KEY binding (not
    /// the receiver) is what gets rewritten. A full scan over every
    /// binding, the same style `rebind_aliases_of_instance`/
    /// `rebind_referents_of_item` already sweep the whole table for
    /// their own cross-binding invalidation.
    ///
    /// Every binding-keyed entry this checker records is GUARD
    /// provenance (`narrowing::compare::narrow_dict_membership_against_
    /// literal_key` wraps it in `DictKey::guarded`, `DictKey::guarded`'s
    /// own doc) — a `binding:<name>` tag is never itself a WRITE key.
    /// This scan matches the tag under its guard wrapper
    /// (`DictKey::guarded(&tag)`, the identical spelling the guard
    /// itself recorded), not the bare tag, so a rewrite of the key
    /// binding still finds and strips the entry.
    fn forget_binding_keyed_star_entries(&mut self, name: &str) {
        let tag = DictKey::guarded(&DictKey::identity(&format!("binding:{name}")));
        for bound in self.bindings.values_mut() {
            if bound.kind != Kind::ObjectStar {
                continue;
            }
            bound.keys.retain(|entry| !(entry.name == tag.name && entry.numeric == tag.numeric));
        }
    }

    /// What the name holds here, if the walk bound it.
    pub fn read(&self, name: &str) -> Option<&AbstractValue> {
        self.bindings.get(name)
    }

    /// Drops every recorded `ObjectKey` entry from `name`'s OWN
    /// `Kind::ObjectStar` value, keeping the star kind (and its element
    /// set) intact — both provenances a recorded entry can carry
    /// (`collection_models::DictKey`'s own doc: a WRITE's exact value,
    /// `dict_write::dict_with_item`'s star arm, or a GUARD's "present at
    /// the guard" claim, `DictKey::guarded`) go stale the moment `name`
    /// is handed to a callee not proven to leave it unwritten, since the
    /// callee may delete the very key either one named. A write entry is
    /// dropped here for the same reason as a guard entry: the callee may
    /// have overwritten or deleted it just as easily as it could remove a
    /// guarded key, and this call has no way to tell which entries the
    /// callee actually touched. The callee's own effect on the dict's
    /// SHAPE beyond that (whether it still holds SOME entries of the
    /// declared element set) is unaffected, so the star itself survives.
    /// A no-op on any other kind, or a name this environment does not
    /// currently bind.
    pub fn forget_recorded_star_entries(&mut self, name: &str) {
        if let Some(bound) = self.bindings.get_mut(name) {
            if bound.kind == Kind::ObjectStar {
                bound.keys.clear();
            }
        }
    }

    /// Whether a module-level alias name still means the alias in this
    /// body: true only when the body never rebinds the name.
    pub fn alias_is_visible(&self, name: &str) -> bool {
        !self.locally_bound.contains(name)
    }

    /// Drop what was known about a name (an unmodeled write may have
    /// changed it). Also drops every access-path fact rooted at this
    /// name (`forget_path_base`'s own doc) — a write to `a` invalidates
    /// whatever this environment knew about `a.n` exactly as it
    /// invalidates `a` itself — and every OTHER receiver's binding-keyed
    /// presence entry naming `a` (`bind`'s own doc), the identical
    /// cross-binding invalidation an unmodeled write needs just as much
    /// as an ordinary successful rebind does.
    pub fn forget(&mut self, name: &str) {
        self.forget_binding_keyed_star_entries(name);
        self.bindings.remove(name);
        self.forget_path_base(&TrackedPlace::bare(name));
        if crate::trace::is_tracing() {
            crate::trace::record_forget_touch(name);
        }
    }

    /// `forget`, naming the CONSTRUCT that forced it — an unmodeled call
    /// whose replay produced no successor value for the receiver, so the
    /// walk cannot let the pre-call fact survive. Records the LAST-TOUCH
    /// LEDGER's `havocked by <construct> @<range>` entry instead of the
    /// plain `forgotten` a cause-less forget leaves: a later declined
    /// read of `name` then names the call that erased it, not just the
    /// fact that something did.
    ///
    /// `construct_range` is the causing construct's own byte range (the
    /// call expression itself) — `crate::trace::record_havoc_touch`
    /// spells it the same way a reader span over that range would, so the
    /// ledger and any span covering the identical call never disagree.
    pub fn forget_with_cause(&mut self, name: &str, construct_range: (usize, usize)) {
        self.forget_binding_keyed_star_entries(name);
        self.bindings.remove(name);
        self.forget_path_base(&TrackedPlace::bare(name));
        if crate::trace::is_tracing() {
            crate::trace::record_havoc_touch(name, construct_range);
        }
    }

    /// ALIASING: rebind every name currently holding a class instance
    /// with the given `identity` (`AbstractValue::instance_identity`,
    /// `instances::judge_construction`'s own per-construction tag) to
    /// `updated` — the SAME instance read back through a DIFFERENT name.
    /// `Environment` tracks a value per NAME, so `same = account;
    /// same.balance = -20` writing through `same`'s own slot alone
    /// leaves `account`'s slot holding the pre-write instance; this is
    /// what `check.rs::write_named_field` calls, after its own write
    /// updates `receiver_name`'s slot directly, to bring every OTHER
    /// alias of the identical runtime object back in step (showcase.py's
    /// own `same = account; same.balance = -20; spend(account.balance)`
    /// row — written through `same`, read through `account`). Skips
    /// `receiver_name` itself (the caller's own direct rebind already
    /// covers that slot, and re-cloning `updated` into it here would be
    /// redundant, not wrong) and any name whose bound value carries no
    /// `instance_identity` at all (an ordinary object with no per-
    /// construction id can never alias one that has one).
    pub fn rebind_aliases_of_instance(&mut self, identity: u32, receiver_name: &str, updated: &AbstractValue) {
        for (name, bound) in self.bindings.iter_mut() {
            if name == receiver_name {
                continue;
            }
            if bound.instance_identity == Some(identity) {
                *bound = updated.clone();
            }
        }
    }

    /// SHARED REFERENT ALIASING: replace every ITEM, at any nesting depth
    /// under any binding, carrying `identity` with `updated` — the same
    /// inner object read back through a different outer container.
    ///
    /// `outer = [[1, 2]]; copy = outer[:]` makes `copy`'s slot 0 and
    /// `outer`'s slot 0 the SAME inner list object (library/copy.rst: a
    /// shallow copy "inserts *references* into it to the objects found in
    /// the original"), which this domain records by giving that inner
    /// value one referent identity that the slice's own item clone
    /// carries along (`collection_models::with_referent_identities`).
    /// Writing `copy[0][0] = 200` rebuilds `copy`'s own slot; this brings
    /// `outer`'s slot — and any other container holding that same inner
    /// object — back in step, the item-level twin of
    /// `rebind_aliases_of_instance`'s name-level sweep.
    ///
    /// The walk is over every binding including `written_name`: a
    /// container can hold the same inner object at more than one slot
    /// (`pair = [inner, inner]`), and the caller's own rebuild only
    /// touched the one slot the write named.
    pub fn rebind_referents_of_item(&mut self, identity: u32, updated: &AbstractValue) {
        for bound in self.bindings.values_mut() {
            replace_referent(bound, identity, updated);
        }
    }
}

/// Replaces every item under `value` whose own `instance_identity` is
/// `identity` with `updated`, recursing into the items that are not
/// themselves the target. A value that IS the target is replaced whole
/// and not recursed into — `updated` already carries whatever the write
/// established beneath it.
fn replace_referent(value: &mut AbstractValue, identity: u32, updated: &AbstractValue) {
    for item in value.items.iter_mut() {
        if item.instance_identity == Some(identity) {
            *item = updated.clone();
        } else {
            replace_referent(item, identity, updated);
        }
    }
}
