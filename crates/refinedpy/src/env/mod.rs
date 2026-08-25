//! The per-body environment: names bound to abstract values, plus the
//! set of names the body itself rebinds. A module-level alias states a
//! refinement inside a body only where that body never rebinds the
//! alias's name — Python scoping makes a name local for the whole body
//! if any statement in the body binds it.
//!
//! ACCESS-PATH BINDINGS: alongside the name-keyed `bindings` map, this
//! environment separately tracks a fact about a PATH — a base binding
//! plus a chain of attribute segments (`a.n`, `d.tzinfo`) — the same
//! PLACE identity the Go adapter's `dataflowfacts.TrackedPlace` already
//! spells (`refined-ts-go/internal/refinedts/dataflowfacts/access_paths.go`),
//! mirrored here as `TrackedPlace` per the one-path-vocabulary rule: a
//! comparison whose tested side is an attribute chain, not a bare name
//! (`0 <= a.n <= 150`), has nowhere to record what it proves without
//! this — `bindings` is keyed on a single name, and `a.n` names no
//! binding at all. `bind_path`/`read_path` record and read a fact
//! keyed on the whole chain; `forget_path_base` drops every path
//! binding whose OWN base name, or any PREFIX of its path, was written
//! — the one forget resolver every write channel routes through, so no
//! write ever leaves a stale path fact behind (`Environment::forget`'s
//! own doc states the identical rule for a bare name; this is its
//! path-shaped twin).

mod bindings;
mod fork_join;
mod module_tables;
mod recording;
mod retained_callables;
mod tracked_place;

#[cfg(test)]
mod tests;

pub use retained_callables::same_module_def_alias_name;
pub use retained_callables::same_module_def_alias_value;
pub use retained_callables::retained_callable_key;
pub use retained_callables::retained_callable_value;
pub use retained_callables::RetainedCallable;
pub use retained_callables::FUNCTION_VALUE_WORD;
pub use tracked_place::tracked_place_of;
pub use tracked_place::TrackedPlace;

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::sync::Mutex;

use refined_domain::abstract_value::AbstractValue;
use ruff_text_size::TextRange;

use crate::function_table::FunctionTable;
use crate::instances::ClassModel;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;
use crate::typereading::DeclaredRefinement;

pub struct Environment {
    bindings: HashMap<String, AbstractValue>,
    /// Facts recorded about an ACCESS PATH (`TrackedPlace`'s own doc) —
    /// a comparison's own narrowing of `a.n`, kept separate from
    /// `bindings` (which is keyed on a single name) since a path names
    /// no single environment slot. `forget_path_base` is the one place
    /// that removes an entry; every write channel routes through it, the
    /// same discipline `forget` already keeps for a bare name.
    path_bindings: HashMap<TrackedPlace, AbstractValue>,
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
    /// The module's own compiled alias table and import identities, if
    /// this environment's walk has one to offer. Rides the environment
    /// for the same reason `functions`/`classes` do:
    /// `summaries::declared_return_seed`, called from `call_result_with_
    /// enclosing`'s own decline points, can read a same-module callee's
    /// own `-> Age`-shaped return annotation through the ordinary
    /// `typereading::declared_refinement` table with no signature
    /// change anywhere along the call chain. `None` for a walk that
    /// never set one (a test environment, or a body reached before the
    /// table is threaded through) — that decline path falls back to
    /// `return_sort_fallback`'s bare `int`/`float`/`str` reading only,
    /// exactly as it did before this field existed.
    declared_aliases: Option<(Arc<HashMap<String, AliasEntry>>, Arc<SurfaceImports>)>,
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
            path_bindings: HashMap::new(),
            locally_bound,
            functions: None,
            classes: None,
            declared_aliases: None,
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
}
