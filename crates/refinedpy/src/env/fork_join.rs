//! Forking an environment for one branch arm's walk, and joining two
//! arms back together once both have run.

use std::collections::HashMap;

use super::Environment;

impl Environment {
    /// A copy of this environment for walking one branch arm — same
    /// locally-bound set, same current bindings, same function, class,
    /// and callable-return tables (`Arc` clones, cheap: both arms of
    /// one body's fork always share the one module/body tables, never
    /// a copy of their contents).
    pub fn fork(&self) -> Environment {
        Environment {
            bindings: self.bindings.clone(),
            path_bindings: self.path_bindings.clone(),
            locally_bound: self.locally_bound.clone(),
            functions: self.functions.clone(),
            classes: self.classes.clone(),
            declared_aliases: self.declared_aliases.clone(),
            datetime_imports: self.datetime_imports.clone(),
            entry_directory: self.entry_directory.clone(),
            callable_returns: self.callable_returns.clone(),
            call_depth: self.call_depth,
            variadic_parameter_names: self.variadic_parameter_names.clone(),
            retained_callables: self.retained_callables.clone(),
            retained_callable_counter: self.retained_callable_counter.clone(),
            lambda_keys_by_range: self.lambda_keys_by_range.clone(),
            // a module-level premise, identical in every arm of the module
            locale_never_set: self.locale_never_set,
            // a fork walks part of the SAME statement (a comprehension's
            // own element pass, a branch arm), so the published nodes
            // travel with it
            evaluated_node: self.evaluated_node.clone(),
            // the SAME recorder, never a copy: a `return` inside the arm
            // this fork walks must reach the asker (the field's own doc)
            returned_values: self.returned_values.clone(),
            // the SAME recorder too, for the identical reason — a node
            // evaluated inside this arm must still reach whoever asked
            // for the whole module walk's recordings
            evaluations: self.evaluations.clone(),
        }
    }

    /// Rejoin two branch arms: only names both arms still know survive,
    /// each joined through the lattice. The locally-bound set is scope
    /// structure, not flow state — it is identical in both arms. The
    /// function, class, datetime-import, callable-return, and
    /// retained-callable tables are likewise identical in both arms
    /// (both forked from the same body's one environment, sharing the
    /// very same `Arc`s — `fork`'s own doc — so `a`'s and `b`'s own
    /// retained-callable tables are not merely equal, they are the
    /// SAME underlying map), so the joined environment simply carries
    /// `a`'s.
    pub fn join(a: Environment, b: &Environment) -> Environment {
        let mut bindings = HashMap::new();
        let locally_bound = a.locally_bound;
        let functions = a.functions;
        let classes = a.classes;
        let declared_aliases = a.declared_aliases;
        let datetime_imports = a.datetime_imports;
        let entry_directory = a.entry_directory;
        let callable_returns = a.callable_returns;
        let call_depth = a.call_depth;
        let variadic_parameter_names = a.variadic_parameter_names;
        let retained_callables = a.retained_callables;
        let retained_callable_counter = a.retained_callable_counter;
        let lambda_keys_by_range = a.lambda_keys_by_range;
        let locale_never_set = a.locale_never_set;
        // both arms forked from one environment, so they hold the SAME
        // recorder `Arc` — carrying `a`'s carries both arms' recordings
        let returned_values = a.returned_values;
        // same reasoning: both arms share the one evaluations `Arc`,
        // so carrying `a`'s loses nothing either arm recorded
        let evaluations = a.evaluations;
        for (name, value_a) in a.bindings {
            if let Some(value_b) = b.bindings.get(&name) {
                bindings.insert(
                    name,
                    refined_domain::lattice_operations::join_known(value_a, value_b.clone()),
                );
            }
        }
        // access-path facts join the same way: only a path BOTH arms
        // still hold a fact about survives, through the identical
        // lattice join `bindings` itself takes.
        let mut path_bindings = HashMap::new();
        for (place, value_a) in a.path_bindings {
            if let Some(value_b) = b.path_bindings.get(&place) {
                path_bindings.insert(
                    place,
                    refined_domain::lattice_operations::join_known(value_a, value_b.clone()),
                );
            }
        }
        Environment {
            bindings,
            path_bindings,
            locally_bound,
            functions,
            classes,
            declared_aliases,
            datetime_imports,
            entry_directory,
            callable_returns,
            call_depth,
            variadic_parameter_names,
            retained_callables,
            retained_callable_counter,
            lambda_keys_by_range,
            locale_never_set,
            // a join lands past the statement whose walk published a
            // node, so nothing carries forward
            evaluated_node: Vec::new(),
            returned_values,
            evaluations,
        }
    }
}
