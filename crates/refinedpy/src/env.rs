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
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;

use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::AbstractValue;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::ExprLambda;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtReturn;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::function_table::FunctionTable;
use crate::instances::ClassModel;
use crate::typereading::DeclaredRefinement;

/// The word every retained-callable `AbstractValue` carries on
/// `kind_word` — the same word `expressions.rs`'s own `Expr::Lambda`
/// arm already answers for an un-retained lambda read, so a retained
/// callable still reads as "a function value" everywhere the WORD
/// alone matters (a fire message against a scalar-ground sink, the
/// `same_module_def_gate_open` check). The distinguishing fact is
/// `source`, which carries the retained-body table key instead of
/// staying empty.
pub const FUNCTION_VALUE_WORD: &str = "a function value";

/// Every retained-callable table entry (a lambda's or a nested def's)
/// is keyed by a fresh id `next_retained_callable_key` mints — NEVER
/// by the AST's own range, for either shape. A range key would let
/// TWO creations of the textually same lambda/def silently conflate:
/// `make_adder(1)` and `make_adder(100)` each build `lambda age: age +
/// step` from the SAME AST node but close over a DIFFERENT `step`, so
/// the second call's own registration would overwrite the first's
/// still-live retained value under a range key, corrupting whichever
/// call's own returned callable is invoked later (`conflation_probe.py`
/// once pinned exactly this: `add_one(40)` answered 140, the OTHER
/// closure's arithmetic, when lambdas were range-keyed). Minting a
/// fresh key per CREATION, the same discipline this table already
/// gives a nested def, closes that gap for a lambda too.
///
/// Because the evaluation site (`expressions.rs`'s `Expr::Lambda` arm,
/// which only reads `&Environment`) knows a lambda only by its own
/// AST range — never by the fresh key `register_retained_callables`
/// minted for its CURRENT creation — `Environment` also carries
/// `lambda_keys_by_range`: a small, separately-shared index from a
/// lambda's own range start to whichever fresh key is CURRENT for it.
/// `register_retained_callables` mints a fresh key and updates this
/// index every time it registers a lambda; the `Expr::Lambda` arm
/// looks the CURRENT key up through this index rather than assuming
/// the range IS the key.

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
    /// The module's own `datetime` import identities — which local
    /// names mean the `datetime` module, the `datetime.datetime`
    /// class, the `datetime.date` class, and the `datetime.timedelta`
    /// class (`expressions::DatetimeImports`'s own doc), if this
    /// environment's walk has one to offer. Rides the environment for
    /// the same reason `functions`/`classes` do: `expressions.rs`'s
    /// datetime gates (`is_datetime_datetime_attribute` and its two
    /// siblings) answer a construction/classmethod call by reading
    /// `environment.datetime_imports()`, with no signature change
    /// anywhere along the call chain. `None` for a walk that never set
    /// one (a test environment, or a body reached before the table is
    /// threaded through) — those gates fall back to the literal
    /// `datetime.*` spelling only.
    datetime_imports: Option<Arc<crate::expressions::DatetimeImports>>,
    /// Whether this module never calls `locale.setlocale` anywhere in
    /// its own source (`expressions::module_never_calls_setlocale`'s
    /// own doc) — the C-locale premise `%a`'s weekday-abbreviation
    /// reading in `datetime.strptime` needs. Rides the environment for
    /// the same reason `datetime_imports` does: computed once per
    /// module (`check.rs`'s module-setup site,
    /// `walk_body_with_self_binding`) and read at the call-evaluation
    /// site with no signature change anywhere along the call chain.
    /// `None` for a walk that never set one (a test environment, or a
    /// body reached before the fact is threaded through) — the caller
    /// (`evaluate_attribute_call`'s `%a` arm) treats `None` the same as
    /// `Some(false)`: the premise is not KNOWN true, so `%a` stays
    /// undetermined rather than assuming the C locale.
    locale_never_set: Option<bool>,
    /// The checked file's own directory, if this environment's walk has
    /// one to offer. Rides the environment for the same reason
    /// `functions`/`classes`/`datetime_imports` do: `expressions.rs`'s
    /// binding-manifest reader (`binding_manifest.rs`) discovers a
    /// module's manifest file beside the checked file, and needs that
    /// directory at the call-evaluation site
    /// (`evaluate_attribute_call`'s own fallthrough), which reads only
    /// `&Environment` and has no `WalkContext` of its own to read
    /// `entry_directory` off — mirroring `check.rs::WalkContext`'s own
    /// field of the identical name, populated at the same one site
    /// `datetime_imports` is (`walk_body_with_self_binding`). `None` for
    /// a walk that never set one (a test environment, or a body reached
    /// before the field is threaded through) — the manifest reader
    /// declines to discover any manifest at all in that case, the same
    /// "nothing to offer" reading every other `None` on this struct
    /// already carries.
    entry_directory: Option<Arc<std::path::PathBuf>>,
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
    /// recorded, keyed by a fresh id `next_retained_callable_key`
    /// mints per CREATION (never the AST's own range — see this
    /// module's own doc on why a range key would conflate two
    /// creations of the same lambda/def text). `Arc<Mutex<...>>`,
    /// shared (never `Arc::make_mut`-copied) across `fork` AND across
    /// the interpreter boundary (`summaries::fresh_body_environment`
    /// inherits the CALLING environment's own `Arc`, rather than
    /// starting a fresh, empty, disposable table) — the one property
    /// `functions`/`classes`'s plain `Arc<T>` sharing does not give:
    /// a table entry a CALLED function's own interpretation creates
    /// (r-ast-census.py's `wrapper`, created inside `with_paramspec_
    /// presence`'s interpreted body) must still be readable AFTER that
    /// call returns, from the CALLER's own environment, since the
    /// returned retained-callable value travels out of the call and is
    /// invoked later from there (`wrapped(200)`, `param_spec_presence`'s
    /// own row). A plain per-environment `HashMap` — sound for every
    /// other table on this struct, all of which either never mutate
    /// after construction or never need a write to outlive their own
    /// call frame — cannot give a written-inside-a-call entry that
    /// reach; this is the one field on `Environment` an ordinary
    /// `Arc`/owned-map choice does not cover.
    retained_callables: Arc<Mutex<HashMap<u32, RetainedCallable>>>,
    /// The shared counter `next_retained_callable_key` draws from —
    /// riding the SAME `Arc` reach as `retained_callables` itself
    /// (cloned together, never separately), so every environment
    /// reachable from one top-level walk mints keys off the one
    /// sequence and two calls (even to unrelated functions) never
    /// collide.
    retained_callable_counter: Arc<AtomicU32>,
    /// A lambda's own AST range start, mapped to whichever fresh
    /// retained-callable key is CURRENT for it — this module's own doc
    /// explains why this index exists: `expressions.rs`'s `Expr::
    /// Lambda` evaluation arm only reads `&Environment` and knows a
    /// lambda only by its own range, never by the fresh key its most
    /// recent creation minted, so `register_retained_callables` (which
    /// DOES hold `&mut Environment` at the moment it registers a fresh
    /// creation) publishes the current key here for that arm to read
    /// back. Shares the same `Arc<Mutex<...>>` reach as `retained_
    /// callables` itself, for the identical cross-call-boundary reason.
    lambda_keys_by_range: Arc<Mutex<HashMap<u32, u32>>>,
    /// Every expression node's already-computed value published for the
    /// walk of a single statement, keyed by that node's own AST range:
    /// `evaluate_expression` answers a matching range directly instead
    /// of evaluating the node. Set for the walk of exactly one statement
    /// and cleared after.
    ///
    /// Two writers publish here. The relational sum (`check.rs`'s
    /// `walk_relational_sum`) publishes a division `total / len(xs)`
    /// whose two operands the kernel tied together — that answers far
    /// more tightly than evaluating the node here could, since the tie
    /// is a fact of the kernel program, not of either operand. A
    /// recognized foreign-edge crossing (`check.rs`'s `serve_foreign_
    /// edge_at`) publishes the return fact at its `json.loads(...)`
    /// node AND, on a discharged crossing whose return is number-sorted,
    /// the serialized stdout string at a SECOND node sitting in the
    /// SAME statement's walk (`foreign_edge.rs`'s `ForeignEdgeOutcome::
    /// stdout_override` doc) — a single slot cannot hold both, so this
    /// carries up to two entries rather than one. The range identifies
    /// each node the way `lambda_keys_by_range` already identifies a
    /// lambda, and for the same reason: the evaluation site reads only
    /// `&Environment` and knows a node by its range.
    ///
    /// A plain owned field rather than the shared `Arc<Mutex<...>>` the
    /// retained-callable tables need: these values never have to outlive
    /// the statement whose walk set them.
    evaluated_node: Vec<(TextRange, AbstractValue)>,
    /// Every value this body's `return` statements produced, in walk
    /// order — `None` unless a caller asked for them
    /// (`collect_returned_values`). `check.rs::walk_return` records the
    /// value it already computed for judging; nothing else writes here,
    /// and a walk that never opts in pays one `Option` check per return.
    ///
    /// `Arc<Mutex<...>>` for the same reason `retained_callables` is:
    /// a `return` inside an `if`/`for`/`try` arm runs against a FORK of
    /// this environment and the fork's own writes must still reach the
    /// asker after the arms rejoin — `join` keeps only `a`'s tables, so
    /// a per-environment `Vec` would silently drop every branch arm's
    /// returns. Sharing the one map makes the recording fork-blind.
    returned_values: Option<Arc<Mutex<Vec<AbstractValue>>>>,
    /// Every expression node this walk evaluated, paired with its own
    /// AST range, in evaluation order — `None` unless a caller installed
    /// a recorder (`set_evaluations_recorder`). `expressions.rs::
    /// evaluate_expression` records the value at its own exit path, for
    /// every node its own dispatch actually ran (a node `evaluated_node`
    /// short-circuits returns early and is NOT recorded here — its value
    /// was already recorded at whichever earlier call published it). An
    /// ordinary check never opts in, so this costs one `Option` check
    /// per node evaluated and nothing more.
    ///
    /// `Arc<Mutex<...>>` for the same fork-blind reason `returned_
    /// values` is: a position the caller asks about may sit inside any
    /// arm of an `if`/`for`/`try`, or inside a nested `def`'s own body
    /// — `check.rs::refined_set_at_position` shares ONE recorder across
    /// the whole module walk by threading it through `WalkContext`
    /// (read by `walk_body_with_self_binding` at the moment it builds
    /// each body's own fresh `Environment`), so every nested body's
    /// walk writes into the same `Vec` the caller reads back afterward.
    evaluations: Option<Arc<Mutex<Vec<(TextRange, AbstractValue)>>>>,
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
            datetime_imports: None,
            locale_never_set: None,
            entry_directory: None,
            callable_returns: None,
            call_depth: 0,
            variadic_parameter_names: Arc::new(HashSet::new()),
            retained_callables: Arc::new(Mutex::new(HashMap::new())),
            retained_callable_counter: Arc::new(AtomicU32::new(0)),
            lambda_keys_by_range: Arc::new(Mutex::new(HashMap::new())),
            evaluated_node: Vec::new(),
            returned_values: None,
            evaluations: None,
        }
    }

    /// Inherits `enclosing`'s own retained-callable table (the SAME
    /// `Arc`, not a copy) — `summaries::fresh_body_environment`'s own
    /// call site, so a def/lambda this call's own interpretation
    /// creates (`interpret_body`'s `Stmt::FunctionDef` arm) is still
    /// readable from `enclosing` (and everywhere `enclosing`'s own
    /// `Arc` already reaches) once this call returns. A callee reached
    /// with NO caller environment at all (`call_result`'s own `None`
    /// enclosing, or a bare test environment) keeps its own fresh,
    /// independent table instead — sound either way, just unable to
    /// share entries with a caller that was never named.
    pub fn inherit_retained_callables(&mut self, enclosing: &Environment) {
        self.retained_callables = enclosing.retained_callables.clone();
        self.retained_callable_counter = enclosing.retained_callable_counter.clone();
        self.lambda_keys_by_range = enclosing.lambda_keys_by_range.clone();
    }

    /// Mints the next retained-callable table key, unique for as long
    /// as this environment's own `Arc<AtomicU32>` is shared (the whole
    /// reach of one top-level walk, `inherit_retained_callables`'s own
    /// doc) — never reused, so two creations of the textually same
    /// lambda/def (two calls to the same enclosing function) always
    /// land in two distinct table slots.
    pub fn next_retained_callable_key(&self) -> u32 {
        self.retained_callable_counter.fetch_add(1, Ordering::Relaxed)
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

    /// Attaches the module's `datetime` import identities so a
    /// construction/classmethod call evaluated against this
    /// environment (and any environment forked from it) can resolve
    /// `datetime`/`date`/`timedelta` by canonical identity rather than
    /// literal spelling (`DatetimeImports`'s own doc).
    pub fn set_datetime_imports(&mut self, datetime_imports: Arc<crate::expressions::DatetimeImports>) {
        self.datetime_imports = Some(datetime_imports);
    }

    /// The module's `datetime` import identities, if this environment
    /// carries one.
    pub fn datetime_imports(&self) -> Option<&Arc<crate::expressions::DatetimeImports>> {
        self.datetime_imports.as_ref()
    }

    /// Attaches the module's own `locale.setlocale`-never-called fact
    /// so `datetime.strptime`'s `%a` reading, evaluated against this
    /// environment (and any environment forked from it), can read the
    /// C-locale premise (`module_never_calls_setlocale`'s own doc).
    pub fn set_locale_never_set(&mut self, locale_never_set: bool) {
        self.locale_never_set = Some(locale_never_set);
    }

    /// The module's own `locale.setlocale`-never-called fact, if this
    /// environment carries one.
    pub fn locale_never_set(&self) -> Option<bool> {
        self.locale_never_set
    }

    /// Attaches the checked file's own directory so a call evaluated
    /// against this environment (and any environment forked from it) can
    /// discover a manifest file beside the checked file
    /// (`binding_manifest.rs`'s own discovery convention).
    pub fn set_entry_directory(&mut self, entry_directory: Arc<std::path::PathBuf>) {
        self.entry_directory = Some(entry_directory);
    }

    /// The checked file's own directory, if this environment carries one.
    pub fn entry_directory(&self) -> Option<&Arc<std::path::PathBuf>> {
        self.entry_directory.as_ref()
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

    /// Records a lambda's/nested def's own retained body under a fresh
    /// key (`next_retained_callable_key`) — the key `retained_callable_
    /// value` encodes into the bound `AbstractValue`'s own `source`
    /// field. Writes through the shared `Arc<Mutex<...>>` (`fork`'s own
    /// doc), so the entry is visible from every environment sharing
    /// that `Arc` — including, after this call returns, the CALLER's
    /// own environment, when this write happened inside an interpreted
    /// call (`summaries::interpret_body`'s `Stmt::FunctionDef` arm).
    pub fn record_retained_callable(&mut self, key: u32, callable: RetainedCallable) {
        self.retained_callables
            .lock()
            .expect("retained-callables table poisoned by an earlier panic")
            .insert(key, callable);
    }

    /// The retained body for `key`, if this walk has recorded one — a
    /// call site reads this after finding the key encoded in a bound
    /// value's `source` field. Returns an owned clone (never a
    /// reference into the lock) so the lock is held only for the
    /// lookup itself.
    pub fn retained_callable(&self, key: u32) -> Option<RetainedCallable> {
        self.retained_callables
            .lock()
            .expect("retained-callables table poisoned by an earlier panic")
            .get(&key)
            .cloned()
    }

    /// Publishes `key` as the CURRENT retained-callable key for the
    /// lambda whose own AST range starts at `range_start` —
    /// `register_retained_callables`'s own call site, run every time it
    /// registers a fresh creation of that lambda. A later creation of
    /// the SAME lambda literal (a second call to its enclosing
    /// function) overwrites the mapping, so the index always answers
    /// the MOST RECENT creation's key — sound because `evaluate_
    /// expression`'s `Expr::Lambda` arm only ever reads this index at
    /// the moment that exact `Expr::Lambda` node is evaluated as a
    /// value, which always happens during the SAME statement's
    /// evaluation that `register_retained_callables` just ran ahead of
    /// (`register_retained_callables`'s own doc: it runs immediately
    /// before the immutable read, never long before it).
    pub fn record_lambda_key(&mut self, range_start: u32, key: u32) {
        self.lambda_keys_by_range
            .lock()
            .expect("lambda-key index poisoned by an earlier panic")
            .insert(range_start, key);
    }

    /// The CURRENT retained-callable key for the lambda whose own AST
    /// range starts at `range_start`, if `register_retained_callables`
    /// has registered a creation of it. `None` when it never has (a
    /// lambda shape `register_retained_callables`'s own recursion does
    /// not reach, or an environment with no such registration step at
    /// all) — the caller falls back to the plain opaque lambda value.
    pub fn lambda_key(&self, range_start: u32) -> Option<u32> {
        self.lambda_keys_by_range
            .lock()
            .expect("lambda-key index poisoned by an earlier panic")
            .get(&range_start)
            .copied()
    }

    /// Publishes up to two expression nodes' already-computed values for
    /// the walk of a single statement (see the field's own doc). An
    /// empty `Vec` clears it, which every caller does once that
    /// statement is walked.
    pub fn set_evaluated_node(&mut self, evaluated: Vec<(TextRange, AbstractValue)>) {
        self.evaluated_node = evaluated;
    }

    /// The published value for the node at `range`, if one was set for
    /// this walk. Every other node reads `None` and evaluates normally.
    pub fn evaluated_node(&self, range: TextRange) -> Option<&AbstractValue> {
        self.evaluated_node
            .iter()
            .find(|(published, _)| *published == range)
            .map(|(_, value)| value)
    }

    /// Asks this body's walk to record every value its `return`
    /// statements produce (`returned_values`'s own doc). Called once,
    /// before the body walks; every fork made afterwards shares the one
    /// recorder.
    pub fn collect_returned_values(&mut self) {
        self.returned_values = Some(Arc::new(Mutex::new(Vec::new())));
    }

    /// Records one `return`'s value, when this walk was asked for them.
    /// A no-op otherwise, which is every ordinary walk.
    pub fn record_returned_value(&self, value: AbstractValue) {
        let Some(recorder) = self.returned_values.as_ref() else {
            return;
        };
        recorder
            .lock()
            .expect("returned-values recorder poisoned by an earlier panic")
            .push(value);
    }

    /// Every value this body's `return` statements produced, in walk
    /// order — an empty vector for a walk that recorded none, `None`
    /// for a walk that was never asked to record.
    pub fn returned_values(&self) -> Option<Vec<AbstractValue>> {
        Some(
            self.returned_values
                .as_ref()?
                .lock()
                .expect("returned-values recorder poisoned by an earlier panic")
                .clone(),
        )
    }

    /// Installs `recorder` as this environment's own evaluations sink —
    /// the SAME `Arc` a caller (`check.rs::refined_set_at_position`,
    /// through `WalkContext::evaluations_recorder`) already holds, so
    /// every write this body's walk makes lands in the one `Vec` the
    /// caller reads back once the whole module walk finishes. Unlike
    /// `collect_returned_values` (which mints a FRESH recorder scoped
    /// to one body), this shares an EXISTING one across every body the
    /// module walk reaches — the aggregation `refined_set_at_position`
    /// needs, since the asked-about position may sit inside any nested
    /// `def`'s own body, each of which builds its own fresh
    /// `Environment`.
    pub fn set_evaluations_recorder(&mut self, recorder: Arc<Mutex<Vec<(TextRange, AbstractValue)>>>) {
        self.evaluations = Some(recorder);
    }

    /// Records one expression node's own range and value, when this
    /// walk was asked to collect them. A no-op otherwise, which is
    /// every ordinary check.
    pub fn record_evaluation(&self, range: TextRange, value: AbstractValue) {
        let Some(recorder) = self.evaluations.as_ref() else {
            return;
        };
        recorder
            .lock()
            .expect("evaluations recorder poisoned by an earlier panic")
            .push((range, value));
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
        Environment {
            bindings,
            locally_bound,
            functions,
            classes,
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

#[cfg(test)]
mod tests {
    use ruff_python_parser::parse_module;

    use super::*;

    /// Parses `source` as a module and returns its single top-level
    /// `def` — the same helper `summaries.rs`'s own tests use, repeated
    /// here since this module's tests need it too and the two files
    /// stay independent per the mission's file-ownership convention.
    fn parsed_def(source: &str) -> StmtFunctionDef {
        let module = parse_module(source).expect("fixture source parses").into_syntax();
        let stmt = module.body.into_iter().next().expect("one top-level statement");
        stmt.function_def_stmt().expect("top-level statement is a def")
    }

    fn bare_retained_callable() -> RetainedCallable {
        let def = parsed_def("def f(x):\n    return x\n");
        RetainedCallable::from_def(&def, HashMap::new())
    }

    /// A key `record_retained_callable` wrote is readable back through
    /// `retained_callable` on the SAME environment.
    #[test]
    fn test_record_and_read_retained_callable_round_trips() {
        let mut environment = Environment::new(HashSet::new());
        let key = environment.next_retained_callable_key();
        environment.record_retained_callable(key, bare_retained_callable());
        assert!(environment.retained_callable(key).is_some());
        assert!(environment.retained_callable(key + 1).is_none());
    }

    /// `fork` shares the SAME underlying table (never a copy): a write
    /// made through the forked environment is visible back through the
    /// original — the property `summaries::fresh_body_environment`'s
    /// own `inherit_retained_callables` call depends on to let a called
    /// function's own retained value outlive its own call frame.
    #[test]
    fn test_fork_shares_the_retained_callable_table() {
        let original = Environment::new(HashSet::new());
        let mut forked = original.fork();
        let key = forked.next_retained_callable_key();
        forked.record_retained_callable(key, bare_retained_callable());
        assert!(original.retained_callable(key).is_some());
    }

    /// `join` carries `a`'s own retained-callable table forward — since
    /// both arms of a join were forked from the same parent (`fork`'s
    /// own doc), a key either arm wrote is visible in the joined
    /// environment.
    #[test]
    fn test_join_keeps_the_retained_callable_table() {
        let parent = Environment::new(HashSet::new());
        let mut arm_a = parent.fork();
        let arm_b = parent.fork();
        let key = arm_a.next_retained_callable_key();
        arm_a.record_retained_callable(key, bare_retained_callable());
        let joined = Environment::join(arm_a, &arm_b);
        assert!(joined.retained_callable(key).is_some());
    }

    /// `inherit_retained_callables` replaces this environment's own
    /// table with `enclosing`'s SAME `Arc` — a key this environment
    /// later records is then visible from `enclosing` too, the exact
    /// property a called function's own interpretation environment
    /// needs so a def/lambda IT creates survives past its own call
    /// frame back into the caller.
    #[test]
    fn test_inherit_retained_callables_shares_writes_both_ways() {
        let enclosing = Environment::new(HashSet::new());
        let mut callee = Environment::new(HashSet::new());
        callee.inherit_retained_callables(&enclosing);
        let key = callee.next_retained_callable_key();
        callee.record_retained_callable(key, bare_retained_callable());
        assert!(enclosing.retained_callable(key).is_some());
    }

    /// `next_retained_callable_key` never repeats within one shared
    /// counter, even across environments that inherited it from each
    /// other — the property that keeps two creations of the same
    /// lambda/def text (two calls to the same enclosing function) from
    /// landing in the same table slot.
    #[test]
    fn test_next_retained_callable_key_never_repeats() {
        let environment = Environment::new(HashSet::new());
        let first = environment.next_retained_callable_key();
        let second = environment.next_retained_callable_key();
        assert_ne!(first, second);
    }

    /// `record_lambda_key`/`lambda_key` round-trip, and a SECOND
    /// registration of the same range overwrites the mapping to the
    /// newer key — the property `expressions.rs::register_retained_
    /// callables` depends on so a lambda re-created with a different
    /// closure (`make_adder(1)` then `make_adder(100)`) is read back
    /// under its OWN creation's key, never a stale earlier one.
    #[test]
    fn test_record_lambda_key_overwrites_on_a_later_creation() {
        let mut environment = Environment::new(HashSet::new());
        let range_start = 42u32;
        let first_key = environment.next_retained_callable_key();
        environment.record_lambda_key(range_start, first_key);
        assert_eq!(environment.lambda_key(range_start), Some(first_key));
        let second_key = environment.next_retained_callable_key();
        environment.record_lambda_key(range_start, second_key);
        assert_eq!(environment.lambda_key(range_start), Some(second_key));
    }

    /// `retained_callable_value`/`retained_callable_key` round-trip:
    /// building a value from a key and reading the key back off it
    /// answers the same key, and an ordinary opaque lambda value (no
    /// retained body) reads back `None`.
    #[test]
    fn test_retained_callable_value_key_round_trip() {
        let value = retained_callable_value(7);
        assert_eq!(retained_callable_key(&value), Some(7));
        let plain = opaque_value(FUNCTION_VALUE_WORD);
        assert_eq!(retained_callable_key(&plain), None);
    }
}
