//! Name-keyed bindings: recording, reading, and forgetting what a name
//! holds, plus aliasing rebinds across instance identity.

use refined_domain::abstract_value::AbstractValue;

use super::tracked_place::TrackedPlace;
use super::Environment;

impl Environment {
    /// Record what a name holds after a statement the walk understood.
    pub fn bind(&mut self, name: &str, value: AbstractValue) {
        self.bindings.insert(name.to_owned(), value);
    }

    /// What the name holds here, if the walk bound it.
    pub fn read(&self, name: &str) -> Option<&AbstractValue> {
        self.bindings.get(name)
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
    /// invalidates `a` itself.
    pub fn forget(&mut self, name: &str) {
        self.bindings.remove(name);
        self.forget_path_base(&TrackedPlace::bare(name));
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
}
