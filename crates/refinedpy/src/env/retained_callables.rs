//! Retained lambda/nested-def bodies: the table an `Environment` shares
//! across `fork` and across the interpreter boundary so a value that
//! travels out of a call (returned, assigned, passed as an argument)
//! can still be interpreted later by whichever name it is called
//! through.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::AbstractValue;
use ruff_python_ast::AtomicNodeIndex;
use ruff_python_ast::ExprLambda;
use ruff_python_ast::Parameters;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtFunctionDef;
use ruff_python_ast::StmtReturn;
use ruff_text_size::Ranged;

use super::Environment;

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

/// The bound `AbstractValue` a bare reference to a SAME-MODULE `def`
/// reads as — `f = identity` (a-statements.py-style aliasing, never a
/// CALL of `identity`, just naming it): `Expr::Name`'s ordinary read
/// arm (`expressions.rs::evaluate_expression_dispatch`) finds `identity`
/// unbound in `environment.bindings` (a module-level `def` is indexed
/// in `environment.functions()`, never separately bound as a value) and
/// would otherwise answer bare `unknown()`, discarding which function
/// `f` actually names. This carries the def's own NAME in `source` —
/// the identical `source = <identity>` convention `instances::
/// class_object_value` already uses for a class's own bare-name value —
/// rather than a table key: no registration, no mutation, and no
/// retained-body table entry is needed, since the def is ALREADY fully
/// resolvable by name through `environment.functions()` at the later
/// call site (`expressions.rs::evaluate_call`'s own alias-call arm).
/// Shares `FUNCTION_VALUE_WORD` with `retained_callable_value` so `f`
/// still reads as "a function value" everywhere that word alone matters
/// (a fire message against a scalar-ground sink, `same_module_def_gate_
/// open`'s own `kind_word` check) — the two shapes are told apart by
/// `source`: `retained_callable_key`'s own numeric parse fails on a def
/// name (`"identity".parse::<u32>()` errors), so a value this function
/// builds is never misread as a retained-callable key, and vice versa
/// (a numeric `source` never matches a real Python identifier, which
/// cannot begin with a digit — `tmp/cpython/Doc/reference/lexical_
/// analysis.rst`'s `identifier` grammar).
pub fn same_module_def_alias_value(name: &str) -> AbstractValue {
    AbstractValue {
        source: name.to_owned(),
        ..opaque_value(FUNCTION_VALUE_WORD)
    }
}

/// The def name `value` carries, if `value` is a same-module-def alias
/// value `same_module_def_alias_value` built (`kind_word` is the
/// function-value word AND `source` is non-empty and does NOT parse as
/// a retained-callable key — the same disambiguation
/// `same_module_def_alias_value`'s own doc states). `None` for an
/// ordinary opaque lambda value (`source` stays empty) or a retained-
/// callable value (`source` parses as a key).
pub fn same_module_def_alias_name(value: &AbstractValue) -> Option<&str> {
    if value.kind != refined_domain::abstract_value::Kind::Object || value.kind_word != Some(FUNCTION_VALUE_WORD) {
        return None;
    }
    if value.source.is_empty() || value.source.parse::<u32>().is_ok() {
        return None;
    }
    Some(value.source.as_str())
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

impl Environment {
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
}
