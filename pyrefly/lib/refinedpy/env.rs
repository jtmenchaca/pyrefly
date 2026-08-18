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

use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::AbstractValue;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::ExprLambda;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtReturn;
use ruff_text_size::Ranged;

use crate::refinedpy::function_table::FunctionTable;
use crate::refinedpy::instances::ClassModel;
use crate::refinedpy::typereading::DeclaredRefinement;

/// The word every retained-callable `AbstractValue` carries on
/// `kind_word` — the same word `expressions.rs`'s own `Expr::Lambda`
/// arm already answers for an un-retained lambda read, so a retained
/// callable still reads as "a function value" everywhere the WORD
/// alone matters (a fire message against a scalar-ground sink, the
/// `same_module_def_gate_open` check). The distinguishing fact is
/// `source`, which carries the retained-body table key instead of
/// staying empty.
pub const FUNCTION_VALUE_WORD: &str = "a function value";

/// The bound `AbstractValue` a retained lambda/def reads as: the same
/// `kind_word` an ordinary (non-retained) lambda value carries, plus
/// the table key encoded into `source` so a later call can look the
/// body back up (`Environment::retained_callable`). `retained_callable_key`
/// reads the key back out of a value built this way.
pub fn retained_callable_value(key: u32) -> AbstractValue {
    AbstractValue {
        source: key.to_string(),
        ..opaque_value(FUNCTION_VALUE_WORD)
    }
}

/// The retained-body table key `value` carries, if `value` is a
/// retained-callable value `retained_callable_value` built (`kind_word`
/// is the function-value word AND `source` parses as the key it
/// encodes). `None` for an ordinary lambda value with no retained body
/// (`source` stays empty — `opaque_value`'s own default), or any other
/// value entirely.
pub fn retained_callable_key(value: &AbstractValue) -> Option<u32> {
    if value.kind != refined_domain::abstract_value::Kind::Object || value.kind_word != Some(FUNCTION_VALUE_WORD) {
        return None;
    }
    value.source.parse::<u32>().ok()
}

/// A lambda's or nested `def`'s own body, retained so a call reached
/// through a NAME the value travels through (returned, assigned,
/// passed as an argument, stored as a field) can still interpret it —
/// `expressions.rs`'s `Expr::Lambda` arm otherwise collapses a lambda
/// to `opaque_value("a function value")` the moment it is read as a
/// value rather than called at its own definition site, discarding the
/// AST a later call would need.
///
/// A lambda's own body (a single expression) is folded into the same
/// shape a nested `def` already has — one `Return` statement wrapping
/// the expression — so both retained forms interpret through the one
/// existing restricted interpreter (`summaries::call_result_with_
/// enclosing`), never a second one.
///
/// `closure` is the free-name snapshot taken at the moment this value
/// was CREATED (`record_retained_callable`'s own call site), never the
/// call site's own environment — Python's own closure rule
/// (executionmodel.rst, "Naming and binding": "if a name is bound in
/// a block... free variables may refer to bindings in the enclosing
/// function scope") pins the binding to the scope the function was
/// DEFINED in, not wherever it is later invoked. Empty for a
/// lambda/def that reads no free name — the common case, and every
/// row this table exists for except a `nonlocal`-free closure over an
/// outer parameter (r-ast-census.py's `wrapper` closing over `f`).
#[derive(Clone)]
pub struct RetainedCallable {
    pub parameters: Box<Parameters>,
    pub body: Vec<Stmt>,
    pub closure: HashMap<String, AbstractValue>,
}

impl RetainedCallable {
    /// A lambda's own retained body: its parameters unchanged, its
    /// single body EXPRESSION wrapped as one `Return` statement — the
    /// same synthetic shape `check.rs::lambda_as_synthetic_def` builds
    /// for the ONE lambda-assign law that already worked before this
    /// table existed (`f = lambda x: <expr>`). A parameterless lambda
    /// (`lambda: 40`) reads `lambda.parameters` as `None`; ruff's own
    /// `Parameters::default()` is the same empty-parameter-list value
    /// an ordinary parameterless `def` carries.
    pub fn from_lambda(lambda: &ExprLambda, closure: HashMap<String, AbstractValue>) -> RetainedCallable {
        let parameters = lambda.parameters.as_deref().cloned().unwrap_or_default();
        let return_stmt = Stmt::Return(StmtReturn {
            node_index: AtomicNodeIndex::NONE,
            range: lambda.body.range(),
            value: Some(lambda.body.clone()),
        });
        RetainedCallable {
            parameters: Box::new(parameters),
            body: vec![return_stmt],
            closure,
        }
    }

    /// A nested `def`'s own retained body: its parameters and body
    /// statements, cloned out of the AST unchanged.
    pub fn from_def(def: &StmtFunctionDef, closure: HashMap<String, AbstractValue>) -> RetainedCallable {
        RetainedCallable {
            parameters: def.parameters.clone(),
            body: def.body.to_vec(),
            closure,
        }
    }

    /// This retained body as a synthetic `StmtFunctionDef` named `name`
    /// — the shape `summaries::call_result_with_enclosing`/
    /// `expressions::positional_arguments_for_def` already interpret,
    /// so a retained-callable call reuses that ONE interpreter rather
    /// than a second one built for this table. `name` need not be the
    /// original lambda's/def's own name (a lambda has none, and a
    /// retained def may be called through a different bound name than
    /// the one it was defined with) — nothing downstream reads it
    /// except error messages, which this file's callers do not surface
    /// for a synthetic def.
    pub fn as_synthetic_def(&self, name: &str, range: ruff_text_size::TextRange) -> StmtFunctionDef {
        StmtFunctionDef {
            node_index: AtomicNodeIndex::NONE,
            range,
            is_async: false,
            decorator_list: Default::default(),
            name: ruff_python_ast::Identifier::new(name, range),
            type_params: None,
            parameters: self.parameters.clone(),
            returns: None,
            body: self.body.iter().cloned().collect(),
        }
    }
}

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
    /// Every CALLABLE-VARIABLE name this walk has recorded a return
    /// refinement for — `x: Callable[[...], R] = ...`'s own `R`, keyed
    /// on `x`. Rides the environment for the same reason
    /// `functions`/`classes` do: a call-site sink
    /// (`check.rs::sink_value`) can answer `name(...)` on a bare Name
    /// found here by reading `environment.callable_returns()`, with no
    /// signature change anywhere along the call chain. `None` for a
    /// walk that never set one.
    callable_returns: Option<Arc<HashMap<String, DeclaredRefinement>>>,
    /// How many interpreted CALLS deep this environment sits — 0 for a
    /// walked body, parent + 1 inside each interpreter child body.
    call_depth: u32,
    /// The names of THIS body's own `*args`/`**kwargs` parameters (empty
    /// for a body with neither) — a plain bare-Name FORWARD of one of
    /// these (`f(*args)`, `f(**kwargs)`) is CPython re-handing the exact
    /// arguments this body itself received, never an independently-built
    /// collection with its own unproven length. `expressions.rs`'s
    /// `call_provable_raise` reads this to tell "the caller's own vararg
    /// slot, forwarded" apart from "a genuinely unbounded list value" —
    /// r-ast-census.py's `with_paramspec_presence`'s own `def wrapper(*args:
    /// P.args, **kwargs: P.kwargs): return f(*args, **kwargs)` forwards
    /// `wrapper`'s own received arguments, never splats an independently-
    /// grown list whose length this checker cannot bound.
    variadic_parameter_names: Arc<std::collections::HashSet<String>>,
    /// Every lambda's/nested def's own retained body this walk has
    /// recorded, keyed by the AST node's own range START offset (unique
    /// within one module — two distinct nodes never share a start
    /// offset). An OWNED, per-environment map — like `bindings`, never
    /// an `Arc`-shared table like `functions`/`classes` — because a new
    /// entry is inserted DURING the walk, the moment a lambda/def value
    /// is created (`record_retained_callable`'s own call sites in
    /// `check.rs`/`summaries.rs`), not built once up front. Cloned
    /// wholesale on `fork`, same as `bindings`.
    retained_callables: HashMap<u32, RetainedCallable>,
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
            callable_returns: None,
            call_depth: 0,
            variadic_parameter_names: Arc::new(HashSet::new()),
            retained_callables: HashMap::new(),
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

    /// Attaches this body's callable-return table so a call site
    /// evaluated against this environment (and any environment forked
    /// from it) can look up a bare-Name callable's return refinement.
    pub fn set_callable_returns(&mut self, callable_returns: Arc<HashMap<String, DeclaredRefinement>>) {
        self.callable_returns = Some(callable_returns);
    }

    /// This body's callable-return table, if it carries one.
    pub fn callable_returns(&self) -> Option<&Arc<HashMap<String, DeclaredRefinement>>> {
        self.callable_returns.as_ref()
    }

    /// How many interpreted CALLS deep this environment sits — 0 for a
    /// walked body, parent + 1 inside each summaries/instances body
    /// interpretation. Dispatch sites pass this into the interpreters
    /// so the CALL_DEPTH_CAP engages across the evaluate↔summaries
    /// boundary; without it a self-recursive def (`countdown` calling
    /// itself through the function table) re-entered at depth 0 forever
    /// and overflowed the stack.
    pub fn call_depth(&self) -> u32 {
        self.call_depth
    }

    pub fn set_call_depth(&mut self, depth: u32) {
        self.call_depth = depth;
    }

    /// Records this body's own `*args`/`**kwargs` parameter names (see
    /// the field's own doc).
    pub fn set_variadic_parameter_names(&mut self, names: Arc<std::collections::HashSet<String>>) {
        self.variadic_parameter_names = names;
    }

    /// Whether `name` is THIS body's own `*args`/`**kwargs` parameter —
    /// a bare-Name read of one of these is always a FORWARD of exactly
    /// what this body itself received, never an independently-built
    /// value.
    pub fn is_variadic_parameter(&self, name: &str) -> bool {
        self.variadic_parameter_names.contains(name)
    }

    /// Records a lambda's/nested def's own retained body under its AST
    /// range's start offset — the key `retained_callable_value`
    /// encodes into the bound `AbstractValue`'s own `source` field. A
    /// later call to `record_retained_callable` with the SAME key
    /// overwrites the earlier entry (the same "last write wins" rule
    /// `bindings` itself already follows) — sound because two distinct
    /// AST nodes never share a start offset, so a repeat key means the
    /// SAME lambda/def is being retained again (a loop iteration
    /// re-evaluating the same `lambda` literal, for instance), with
    /// whatever closure snapshot is current now.
    pub fn record_retained_callable(&mut self, key: u32, callable: RetainedCallable) {
        self.retained_callables.insert(key, callable);
    }

    /// The retained body for `key`, if this walk has recorded one — a
    /// call site reads this after finding the key encoded in a bound
    /// value's `source` field.
    pub fn retained_callable(&self, key: u32) -> Option<&RetainedCallable> {
        self.retained_callables.get(&key)
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
    /// locally-bound set, same current bindings, same function, class,
    /// and callable-return tables (`Arc` clones, cheap: both arms of
    /// one body's fork always share the one module/body tables, never
    /// a copy of their contents).
    pub fn fork(&self) -> Environment {
        Environment {
            bindings: self.bindings.clone(),
            locally_bound: self.locally_bound.clone(),
            functions: self.functions.clone(),
            classes: self.classes.clone(),
            callable_returns: self.callable_returns.clone(),
            call_depth: self.call_depth,
            variadic_parameter_names: self.variadic_parameter_names.clone(),
            retained_callables: self.retained_callables.clone(),
        }
    }

    /// Rejoin two branch arms: only names both arms still know survive,
    /// each joined through the lattice. The locally-bound set is scope
    /// structure, not flow state — it is identical in both arms. The
    /// function, class, and callable-return tables are likewise
    /// identical in both arms (both forked from the same body's one
    /// environment, which carries the one module/body tables), so the
    /// joined environment simply carries `a`'s. The retained-callable
    /// table UNIONS both arms (rather than intersecting, the way
    /// `bindings` does): a key is an AST node's own range start, so a
    /// key both arms recorded always carries the SAME node's content —
    /// there is nothing to reconcile — and a key only one arm recorded
    /// (that arm's own branch is the only one that executed the
    /// lambda/def) is still a true fact after the join, unlike a plain
    /// VALUE binding, which the other arm may have rebound to something
    /// else entirely.
    pub fn join(a: Environment, b: &Environment) -> Environment {
        let mut bindings = HashMap::new();
        let locally_bound = a.locally_bound;
        let functions = a.functions;
        let classes = a.classes;
        let callable_returns = a.callable_returns;
        let call_depth = a.call_depth;
        let variadic_parameter_names = a.variadic_parameter_names;
        for (name, value_a) in a.bindings {
            if let Some(value_b) = b.bindings.get(&name) {
                bindings.insert(
                    name,
                    refined_domain::lattice_operations::join_known(value_a, value_b.clone()),
                );
            }
        }
        let mut retained_callables = a.retained_callables;
        for (key, callable) in &b.retained_callables {
            retained_callables.entry(*key).or_insert_with(|| callable.clone());
        }
        Environment {
            bindings,
            locally_bound,
            functions,
            classes,
            callable_returns,
            call_depth,
            variadic_parameter_names,
            retained_callables,
        }
    }
}
