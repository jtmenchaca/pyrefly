/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The per-body environment: names bound to abstract values, plus the
//! set of names the body itself rebinds. A module-level alias states a
//! refinement inside a body only where that body never rebinds the
//! alias's name — Python scoping makes a name local for the whole body
//! if any statement in the body binds it.

use std::collections::HashMap;
use std::collections::HashSet;

use refined_domain::abstract_value::AbstractValue;

pub struct Environment {
    bindings: HashMap<String, AbstractValue>,
    locally_bound: HashSet<String>,
}

impl Environment {
    /// A fresh environment for one body, given every name the body
    /// binds anywhere (assignments, targets, parameters, defs,
    /// imports, `for`/`with`/walrus targets).
    pub fn new(locally_bound: HashSet<String>) -> Environment {
        Environment {
            bindings: HashMap::new(),
            locally_bound,
        }
    }

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
    /// changed it).
    pub fn forget(&mut self, name: &str) {
        self.bindings.remove(name);
    }

    /// A copy of this environment for walking one branch arm — same
    /// locally-bound set, same current bindings.
    pub fn fork(&self) -> Environment {
        Environment {
            bindings: self.bindings.clone(),
            locally_bound: self.locally_bound.clone(),
        }
    }

    /// Rejoin two branch arms: only names both arms still know survive,
    /// each joined through the lattice. The locally-bound set is scope
    /// structure, not flow state — it is identical in both arms.
    pub fn join(a: Environment, b: &Environment) -> Environment {
        let mut bindings = HashMap::new();
        let locally_bound = a.locally_bound;
        for (name, value_a) in a.bindings {
            if let Some(value_b) = b.bindings.get(&name) {
                bindings.insert(
                    name,
                    refined_domain::lattice_operations::join_known(value_a, value_b.clone()),
                );
            }
        }
        Environment {
            bindings,
            locally_bound,
        }
    }
}
