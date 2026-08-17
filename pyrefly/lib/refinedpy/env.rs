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
use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;

use crate::refinedpy::function_table::FunctionTable;
use crate::refinedpy::instances::ClassModel;

pub struct Environment {
    bindings: HashMap<String, AbstractValue>,
    locally_bound: HashSet<String>,
    /// The module's own top-level `def`s, if this environment's walk
    /// has one to offer. Riding the table on the environment (rather
    /// than adding a parameter to every call site) is the whole point:
    /// `evaluate_expression(&Expr, &Environment, kernel)` can answer a
    /// same-module `Call` by reading `environment.functions()`, with no
    /// signature change anywhere along the call chain. `None` for a
    /// walk that never set one (a test environment, or a body reached
    /// before the table is threaded through).
    functions: Option<Arc<FunctionTable>>,
    /// The module's own class table, by class name, if this
    /// environment's walk has one to offer. Rides the environment for
    /// the same reason `functions` does: `evaluate_expression` can
    /// answer a same-module construction call (`Person(age=40)`) by
    /// reading `environment.classes()`, with no signature change
    /// anywhere along the call chain. `None` for a walk that never set
    /// one.
    classes: Option<Arc<HashMap<String, ClassModel>>>,
}

impl Environment {
    /// A fresh environment for one body, given every name the body
    /// binds anywhere (assignments, targets, parameters, defs,
    /// imports, `for`/`with`/walrus targets).
    pub fn new(locally_bound: HashSet<String>) -> Environment {
        Environment {
            bindings: HashMap::new(),
            locally_bound,
            functions: None,
            classes: None,
        }
    }

    /// Attaches the module's function table so calls evaluated against
    /// this environment (and any environment forked from it) can look
    /// up a same-module callee by name.
    pub fn set_functions(&mut self, functions: Arc<FunctionTable>) {
        self.functions = Some(functions);
    }

    /// The module's function table, if this environment carries one.
    pub fn functions(&self) -> Option<&Arc<FunctionTable>> {
        self.functions.as_ref()
    }

    /// Attaches the module's class table so a construction call
    /// evaluated against this environment (and any environment forked
    /// from it) can look up a same-module class by name.
    pub fn set_classes(&mut self, classes: Arc<HashMap<String, ClassModel>>) {
        self.classes = Some(classes);
    }

    /// The module's class table, if this environment carries one.
    pub fn classes(&self) -> Option<&Arc<HashMap<String, ClassModel>>> {
        self.classes.as_ref()
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
    /// locally-bound set, same current bindings, same function and
    /// class tables (`Arc` clones, cheap: both arms of one body's fork
    /// always share the one module tables, never a copy of their
    /// contents).
    pub fn fork(&self) -> Environment {
        Environment {
            bindings: self.bindings.clone(),
            locally_bound: self.locally_bound.clone(),
            functions: self.functions.clone(),
            classes: self.classes.clone(),
        }
    }

    /// Rejoin two branch arms: only names both arms still know survive,
    /// each joined through the lattice. The locally-bound set is scope
    /// structure, not flow state — it is identical in both arms. The
    /// function and class tables are likewise identical in both arms
    /// (both forked from the same body's one environment, which
    /// carries the one module tables), so the joined environment
    /// simply carries `a`'s.
    pub fn join(a: Environment, b: &Environment) -> Environment {
        let mut bindings = HashMap::new();
        let locally_bound = a.locally_bound;
        let functions = a.functions;
        let classes = a.classes;
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
            functions,
            classes,
        }
    }
}
