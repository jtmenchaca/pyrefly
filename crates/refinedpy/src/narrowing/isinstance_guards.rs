//! `isinstance` and `TypeGuard`/`TypeIs` call leaves, plus sort seeds.

use std::sync::Arc;

use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::Stmt;

use crate::env::Environment;

use super::assume;
use super::name_of;

/// Whether `call` is a same-module call to a function whose OWN return
/// annotation is `TypeGuard[X]`/`TypeIs[X]` (typing.rst's user-defined
/// type guard: "a special form... that can be used to annotate the
/// return type of a user-defined type guard function") — recognized
/// SYNTACTICALLY only. `typing.TypeGuard`/`TypeIs` state a CLAIM the
/// function's own signature makes; this recognizer's caller
/// (`narrow_type_guard_call`) never trusts that claim on its own —
/// f-type-nodes.py's own `dishonest_predicate` row is exactly why:
/// `claims_age`'s signature states `TypeGuard[Age]`, but its body only
/// proves `isinstance(v, int)`, strictly weaker than `Age`; trusting the
/// claim would wrongly narrow `value` all the way to `Age` and read the
/// row SILENT, when the row expects a fire. This function's ONLY job is
/// to recognize the shape so the `Expr::Call` dispatch knows to ATTEMPT
/// body-proof narrowing at all — the claimed `X` itself is never read
/// anywhere in this recognizer or its caller.
pub(super) fn recognizes_type_guard_call(call: &ruff_python_ast::ExprCall, environment: &Environment) -> bool {
    let Expr::Name(callee) = call.func.as_ref() else {
        return false;
    };
    let Some(def) = environment.functions().and_then(|table| table.def(callee.id.as_str())) else {
        return false;
    };
    let Some(returns) = def.returns.as_deref() else {
        return false;
    };
    let Expr::Subscript(subscript) = returns else {
        return false;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return false;
    };
    matches!(head.id.as_str(), "TypeGuard" | "TypeIs")
}

/// Narrows `call`'s own first argument by what a `TypeGuard[X]`/`TypeIs[X]`-
/// annotated predicate's OWN BODY proves, never by the annotation's claimed
/// `X` — `recognizes_type_guard_call`'s own doc names why trusting the
/// claim alone is unsound. The proof: when the predicate's body is exactly
/// one statement, `return <condition>` (`is_age`/`claims_age`'s own shape —
/// a boolean expression naming the predicate's own first parameter), that
/// `<condition>` is handed to THIS SAME `assume` function, in a fresh
/// sandbox environment where the predicate's own parameter name starts
/// UNBOUND (mirroring a real call, where `check.rs::seed_parameters` states
/// nothing for `object`-typed parameters), asked under `truth = true` (the
/// question this narrowing site itself is asking: "given the call proved
/// True, what does that say"). Whatever the predicate's own parameter name
/// ends up bound to in that sandbox IS the proven set — read back and
/// copied onto the CALL's own first argument name in the real environment.
/// `is_age`'s `isinstance(v, int) and not isinstance(v, bool) and 0 <= v <=
/// 120` proves `v` down to exactly `Age`'s own set through this same
/// mechanism the ordinary top-level walk already uses for a seeded
/// parameter; `claims_age`'s bare `isinstance(v, int)` proves only the
/// unbounded `int` sort, which is NOT a subset of `Age` — so `return value`
/// against `-> Age` still fires, exactly as the row expects. A predicate
/// whose body is not this single-`return`-of-a-condition shape, or whose
/// own parameter never ends up bound in the sandbox (the condition proved
/// nothing this file's narrowing channels read), leaves the call's argument
/// untouched — the same "narrows nothing" default as any other declined
/// leaf.
pub(super) fn narrow_type_guard_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>) {
    let Expr::Name(callee) = call.func.as_ref() else {
        return;
    };
    let Some(argument) = call.arguments.args.first() else {
        return;
    };
    let Some(argument_name) = name_of(argument) else {
        return;
    };
    if environment.read(argument_name).is_some() {
        return;
    }
    let Some(def) = environment.functions().and_then(|table| table.def(callee.id.as_str())) else {
        return;
    };
    // Skip every leading docstring (`is_age`'s own `"""honest TypeGuard...
    // """`) before requiring the SOLE remaining statement be a bare
    // `return` — the same docstring-shaped skip
    // `summaries::first_non_docstring_statement` applies to a callee body
    // elsewhere, inlined here since that function answers only the FIRST
    // such statement, not the remaining slice this needs.
    let non_docstring_body: Vec<&Stmt> = def
        .body
        .iter()
        .skip_while(|stmt| matches!(stmt, Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::StringLiteral(_))))
        .collect();
    let [Stmt::Return(ret)] = non_docstring_body.as_slice() else {
        return;
    };
    let Some(condition) = ret.value.as_deref() else {
        return;
    };
    let Some(parameter) = def.parameters.posonlyargs.iter().chain(def.parameters.args.iter()).next() else {
        return;
    };
    let parameter_name = parameter.parameter.name.id.as_str();

    let sandbox = Environment::new(std::collections::HashSet::new());
    let sandbox = assume(condition, sandbox, kernel, true);
    let Some(proven) = sandbox.read(parameter_name) else {
        return;
    };
    environment.bind(argument_name, proven.clone());
}

/// `isinstance(name, int | float | bool)` (mission point 6): filters a
/// Values binding by `kind_tag`. `PrimitiveKind::Number` is the
/// sort-unknown numeric tag (AGENT-BRIEF.md, Wave-1 recognition facts —
/// int-vs-float is not yet distinguished at the value level except
/// where the syntax proves it), so a Number-tagged state passes
/// unfiltered both ways: this wave cannot prove which arm of an
/// int/float isinstance test it falls on.
///
/// A name the environment has NOT bound at all (an `object`-typed
/// parameter — `check.rs::seed_parameters` states nothing for the bare
/// `object` annotation, since no alias names it) is a SEPARATE case
/// from an existing Values binding: `environment.read` answers `None`,
/// not a Values state to filter. A name bound to `Kind::Unknown` — a
/// read this file genuinely determined NOTHING about (a subscript into
/// an unrecognized container shape, an unmodeled call's own result,
/// `abstract_value::unknown`'s own doc: "no fact reads through it at
/// all") — carries the identical absence of information, so it takes
/// the SAME seeding path rather than the "existing binding" arm below:
/// an `Unknown` value is not a state with members to filter, and
/// treating it as one that "already agrees" or "wholly disagrees"
/// with the test would be a claim this file never derived. Both cases
/// converge here as `no_information`. `isinstance(value, int)`/`float`
/// PROVING true (`truth` and no information) is itself the first fact
/// this environment learns about `value` — it seeds a fresh
/// `Kind::Set` binding holding the unbounded sort (the same set
/// `summaries.rs::return_sort_fallback`/`expressions.rs`'s `int(...)`
/// row build for a proved-but-unbounded `int`/`float`), grade
/// `TrustSpec` (the isinstance test is read, not executed — the same
/// grade `seed_parameters`'s own annotation-read seeding uses).
/// `isinstance(value, bool)` seeds `Kind::Values` instead: `bool`'s
/// domain is the two exact values `{0, 1}` (`string_models.rs`'s
/// `boolean_value` convention), not an unbounded ray, so it is not a
/// Set-kind sort seed. Proving FALSE, or a name already bound to a
/// READABLE value (however far from the sort being tested), never
/// seeds here — a falsified test says nothing positive about which
/// sort `value` DOES hold, and an existing readable binding is this
/// function's other, unchanged, arm below.
pub(super) fn narrow_isinstance_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, truth: bool) {
    let Expr::Name(func_name) = call.func.as_ref() else {
        return;
    };
    if func_name.id.as_str() != "isinstance" {
        return;
    }
    if call.arguments.args.len() != 2 {
        return;
    }
    let Some(name) = name_of(&call.arguments.args[0]) else {
        return;
    };
    // A CONTAINER test (`isinstance(x, list)`, `isinstance(x, dict)`)
    // asks a shape question, not a sort question, so it is read first and
    // routed to the arm filter that answers by shape. Every other
    // classinfo shape falls through to the scalar-sort reader below.
    if let Some(container_names) = isinstance_container_names(&call.arguments.args[1]) {
        narrow_union_by_container(name, &container_names, environment, truth);
        return;
    }
    let Some(tags) = isinstance_type_tags(&call.arguments.args[1]) else {
        return;
    };
    let current = environment.read(name).cloned();
    let no_information = match &current {
        None => true,
        Some(value) => value.kind == Kind::Unknown,
    };
    if no_information {
        if truth {
            if let [tag] = tags.as_slice() {
                if let Some(seeded) = sort_seed(*tag) {
                    environment.bind(name, seeded);
                }
            }
        }
        return;
    }
    let current = current.expect("no_information false means Some was read above");
    // A KindUnion binding (json.loads's own honest return space,
    // `expressions.rs::json_loads_value_space`) narrows arm-by-arm: each
    // arm already carries the `kind_tag` an ordinary Values/Set binding
    // does, so `isinstance(x, float)` keeps only the arms whose tag
    // matches (`truth`) or excludes them (`!truth`) — the same filter
    // this function already runs on a single Values binding, applied
    // per arm instead of once. An arm with no `kind_tag` at all (the
    // list/dict arms, built via `opaque_value` on `Kind::Object`) never
    // matches a primitive tag either way, so it survives a `truth` test
    // only when the test is proving the union does NOT hold that sort
    // (`!truth` keeps it) and is dropped when `truth` asks for a sort it
    // cannot be. `kind_union_of` collapses the result: one surviving arm
    // answers bare, and no dropped arm decides the fold "no member
    // left standing" — that reading belongs to `arm_is_infeasible`
    // (Values-only today), not to this narrowing.
    if current.kind == Kind::KindUnion {
        let kept: Vec<AbstractValue> = current
            .arms
            .iter()
            .filter(|arm| {
                // A CONTAINER arm can carry a scalar `kind_tag` — a
                // `list[Age]` arm is tagged Integer off its ELEMENT's own
                // sort (`check::seed::union_arm_seed`, matching the
                // reading `star_numeric_hull` and its siblings take) — so
                // the tag alone would read `isinstance(x, int)` as TRUE
                // of a list of ints. The shape question settles it first:
                // a container is never an instance of a scalar sort.
                let matches_tag = !arm_is_container(arm, &["list", "dict"])
                    && arm.kind_tag.is_some_and(|tag| tags.contains(&tag));
                matches_tag == truth
            })
            .cloned()
            .collect();
        environment.bind(name, kind_union_of(kept));
        return;
    }
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // a sort-unknown Number state cannot be proved in, or proved out,
    // of an int/float isinstance test — pass through unfiltered
    if kind_tag == PrimitiveKind::Number {
        return;
    }
    let matches_tag = tags.contains(&kind_tag);
    if matches_tag == truth {
        // every member already agrees with the test — nothing to drop
        return;
    }
    // the whole binding disagrees with the test — every member is
    // infeasible under this arm
    let grade = trust_level_of(&current);
    environment.bind(name, known_values(Vec::new(), kind_tag, grade));
}

/// `isinstance(x, list)` / `isinstance(x, dict)` on a name bound to a
/// `Kind::KindUnion` — the shape a general union PARAMETER seeds
/// (`check::seed::union_parameter_seed`). Keeps the arms whose own shape
/// matches the named container when the test proves TRUE, and the arms
/// whose shape does NOT when it proves FALSE (`x: int | list[Age]` under
/// a false `isinstance(x, int)` is the `list[Age]` arm — A8.guard.sort's
/// own `element_of_excluded_sequence_inside` row reaches its else-branch
/// exactly that way). `kind_union_of` collapses a single surviving arm to
/// that arm bare, so `x[0]` past the guard reads through the ordinary
/// sequence channel with no union-aware subscript reader needed.
///
/// A name bound to anything OTHER than a KindUnion is left untouched:
/// this domain has no shape for "not a list" to record against a single
/// bound value, and a value that already reads as one definite shape
/// gains nothing from a test it either already satisfies or already
/// refutes.
fn narrow_union_by_container(
    name: &str,
    container_names: &[&'static str],
    environment: &mut Environment,
    truth: bool,
) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::KindUnion {
        return;
    }
    let kept: Vec<AbstractValue> = current
        .arms
        .iter()
        .filter(|arm| arm_is_container(arm, container_names) == truth)
        .cloned()
        .collect();
    environment.bind(name, kind_union_of(kept));
}

/// The fresh binding a PROVED `isinstance(x, tag)` seeds for a name the
/// environment held nothing about at all (see `narrow_isinstance_call`'s
/// doc): the unbounded `Kind::Set` ray for `int`/`float`/`str`, or `None`
/// for `bool` (and `Number`, which no `isinstance` argument ever names —
/// `isinstance_type_tags` only ever answers Integer/Float/Boolean/String)
/// — `bool`'s own two-value seed is built directly at the call site
/// instead, since it is `Kind::Values`, a different constructor
/// entirely. `str` seeds the whole-strings ground
/// (`codepoint_sets::strings()`), untagged (`kind_tag: None` — matching
/// `check.rs::seed_parameters`'s own choice for a bare `str` parameter,
/// `narrowing.rs::environment_with_bare_string`'s own doc): without this
/// arm, `isinstance(x, str)` proving true on an `Any`/`object`-typed `x`
/// left `x` unbound (`environment.read` still answers `None` past the
/// test), so a later `Code`-declared sink read `x` as never-narrowed and
/// the checker determined nothing at all (RTS7002) — A3.guard.sort's own
/// `isinstance_str_outside` row, which needs `x` to become the WHOLE
/// string ground precisely so the subsequent `Code` containment check
/// can refuse it (Σ* ⊄ /^[A-Z]{2}$/), not stay silent.
pub(super) fn sort_seed(tag: PrimitiveKind) -> Option<AbstractValue> {
    match tag {
        PrimitiveKind::Integer => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(unbounded_integers(), None, TrustSpec, SetKindTag::None)
        }),
        PrimitiveKind::Float => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(refined_sets::refinement_forms::numbers(), None, TrustSpec, SetKindTag::None)
        }),
        PrimitiveKind::Boolean => Some(known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec)),
        PrimitiveKind::String => Some(known_set(strings(), None, TrustSpec, SetKindTag::None)),
        PrimitiveKind::Number | PrimitiveKind::Array => None,
    }
}

/// The unbounded whole-number ray (every integer, no floor or ceiling)
/// — the same shape `summaries.rs`'s private `whole_integers` builds;
/// copied here rather than shared cross-file, matching this file's own
/// documented precedent (`literal_number`'s doc) for a small leaf
/// reader every narrowing-adjacent file keeps its own copy of.
pub(super) fn unbounded_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

/// The `PrimitiveKind`s an `isinstance` second argument names, for
/// exactly the shapes mission point 6 covers: a bare type name
/// (`int`/`float`/`bool`/`str`), or a `|`-chain of them
/// (`isinstance(x, int | float)`). Any other shape (a tuple form, a
/// non-primitive type) answers `None` — not read this wave.
pub(crate) fn isinstance_type_tags(expression: &Expr) -> Option<Vec<PrimitiveKind>> {
    match expression {
        Expr::Name(name) => primitive_kind_of_type_name(name.id.as_str()).map(|tag| vec![tag]),
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let mut left = isinstance_type_tags(&binop.left)?;
            let right = isinstance_type_tags(&binop.right)?;
            left.extend(right);
            Some(left)
        }
        _ => None,
    }
}

/// The CONTAINER type names an `isinstance` second argument can name —
/// `list`, `tuple`, `set`, `frozenset`, and `dict` (functions.rst,
/// `isinstance`; stdtypes.rst gives each as a built-in type). These are
/// not `PrimitiveKind`s: a container is not a scalar sort, and this
/// domain tells them apart by an `AbstractValue`'s own `Kind`, never by
/// a `kind_tag`. A `|`-chain of container names folds the same way
/// `isinstance_type_tags` folds a scalar chain; `None` for any other
/// shape, including a chain MIXING a container name with a scalar one
/// (`list | int`), which names two different shape questions this
/// reader has no single answer for.
pub(crate) fn isinstance_container_names(expression: &Expr) -> Option<Vec<&'static str>> {
    match expression {
        Expr::Name(name) => container_type_name(name.id.as_str()).map(|word| vec![word]),
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let mut left = isinstance_container_names(&binop.left)?;
            let right = isinstance_container_names(&binop.right)?;
            left.extend(right);
            Some(left)
        }
        _ => None,
    }
}

fn container_type_name(name: &str) -> Option<&'static str> {
    match name {
        "list" => Some("list"),
        "tuple" => Some("tuple"),
        "set" => Some("set"),
        "frozenset" => Some("frozenset"),
        "dict" => Some("dict"),
        _ => None,
    }
}

/// Whether one union arm holds a value of one of the CONTAINER shapes
/// `names` lists.
///
/// This domain represents every sequence — `list`, `tuple`, `set`,
/// `frozenset` alike — as one shape (`collection_models`'s own module
/// doc: no dedicated tuple or set variant exists, since a set's element
/// uniqueness and a tuple's immutability are invisible to a reader that
/// consumes the value by index, membership, and `len()`). Concretely
/// that is a `Kind::List` for a known-length display and a `Kind::Set`
/// repetition window for a declared `list[X]` parameter. So any of the
/// four sequence names answers true for either of those two shapes: the
/// domain does not hold the distinction, and claiming it does — reading
/// `isinstance(x, list)` as FALSE on a value this file recorded as a
/// tuple — would refuse a program that is correct.
///
/// A `dict` arm is a `Kind::Object` (a known key table) or a
/// `Kind::ObjectStar` (a `dict[str, X]` parameter's unbounded-key seed),
/// which no sequence name matches and which matches no scalar sort.
fn arm_is_container(arm: &AbstractValue, names: &[&'static str]) -> bool {
    let sequence = matches!(arm.kind, Kind::List)
        || (arm.kind == Kind::Set
            && arm.set_kind_tag == SetKindTag::None
            && refined_sets::repetition_window_forms::as_repetition(&arm.set).is_some());
    let mapping = matches!(arm.kind, Kind::Object | Kind::ObjectStar);
    names.iter().any(|name| match *name {
        "dict" => mapping,
        _ => sequence,
    })
}

pub(super) fn primitive_kind_of_type_name(name: &str) -> Option<PrimitiveKind> {
    match name {
        "int" => Some(PrimitiveKind::Integer),
        "float" => Some(PrimitiveKind::Float),
        "bool" => Some(PrimitiveKind::Boolean),
        "str" => Some(PrimitiveKind::String),
        _ => None,
    }
}
