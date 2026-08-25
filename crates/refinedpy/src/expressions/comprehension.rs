
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;

use crate::collection_models;
use crate::env::Environment;
use crate::narrowing;

use super::evaluate_expression;
use super::arithmetic::*;
use super::compare::*;
use super::fstring::*;

/// `[elt for target in iterable if cond ...]` / the same shape for a set
/// display and a generator expression — expressions.rst, "Displays for
/// lists, sets and dictionaries": "the comprehension consists of a
/// single expression, followed by at least one `for` clause." Modeled
/// as ONE single-clause shape (exactly one `Comprehension`; a second
/// `for` clause or an `async for` — `is_async` — declines outright; the
/// target a bare `Expr::Name` or a two-name tuple target,
/// `comprehension_target_names`'s own doc) over EITHER of two iterable
/// shapes: a known `Kind::List` of already-known elements (the CONCRETE
/// path, tried first, unchanged from before this function grew a second
/// arm), or an unknown-length sequence known by its element SET
/// (`comprehension_target_and_star_element`'s own doc — a declared/
/// refined parameter with no concrete items, tried only once the
/// concrete path declines). The concrete path forks the environment
/// once per surviving element, binding the target, evaluating every
/// `if` condition in order (a `known&&false` truthiness drops the
/// element; `known&&true` keeps checking the rest; anything not fully
/// known makes the WHOLE comprehension unknown — a single undecidable
/// filter means this file cannot say which elements the real list would
/// contain), then evaluating `elt` on that fork; the collected elements
/// build through `collection_models::list_literal_value`. The star path
/// forks ONCE (`comprehension_star_elements`'s own doc) and answers a
/// star-shaped `Kind::Set`, never a `Kind::List` — a length-unstated
/// result has no exact positional slots to state. Either shape is
/// honest for the same reason: a set's own element-uniqueness and a
/// generator's own lazy-iteration behavior are both invisible to a
/// caller that only ever consumes the sequence via `len()`/`sum()`/a
/// `for`-loop read.
pub(super) fn evaluate_list_or_set_comp(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    if let Some(elements) = comprehension_elements(element_expr, generators, environment, kernel) {
        return collection_models::list_literal_value(&elements);
    }
    if let Some(star) = comprehension_star_elements(element_expr, generators, environment, kernel) {
        return star;
    }
    unknown()
}

/// `{key: value for target in iterable if cond ...}` — the same
/// single-clause/known-List-iterable restriction as
/// `evaluate_list_or_set_comp` (including its two-name tuple target
/// row, the shape a `{k: v for k, v in d.items()}` walk needs), with
/// the additional requirement that `key` evaluates to a known exact
/// String OR a known single Integer-sorted value at every surviving
/// element (this domain's dict literal accepts string and int keys,
/// `collection_models.rs`'s own documented `DictKey` restriction) —
/// any element whose key is neither of those two sorts makes the whole
/// comprehension unknown() rather than silently dropping that entry.
pub(super) fn evaluate_dict_comp(
    comp: &ruff_python_ast::ExprDictComp,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    let Some(key_expr) = comp.key.as_deref() else {
        // a `**spread` entry inside a dict comprehension has no single
        // key expression to read
        return unknown();
    };
    let Some(rows) = comprehension_rows(key_expr, &comp.value, &comp.generators, environment, kernel) else {
        return unknown();
    };
    let mut keys: Vec<Option<collection_models::DictKey>> = Vec::with_capacity(rows.len());
    let mut values: Vec<AbstractValue> = Vec::with_capacity(rows.len());
    for (key, value) in rows {
        // a string-sorted key value builds an ordinary string
        // DictKey; a single known Integer-sorted key value (the
        // comprehension's own mapped element, e.g. `{age: ... for age
        // in [15, 20]}`) builds an int DictKey — the same two key
        // sorts `dict_literal_value` accepts for a plain `{...}`
        // display (`collection_models.rs`'s own module doc). Any
        // other key shape (Float, Boolean, an unread value) declines
        // the whole comprehension, matching `dict_literal_value`'s
        // own "even one unsupported key" honesty.
        let dict_key = if let Some(text) = exact_string_values(&key).and_then(code_points_to_string) {
            collection_models::DictKey::string(&text)
        } else if let Some((number, PrimitiveKind::Integer)) = single_numeric_value(&key) {
            collection_models::DictKey::integer(number as i64)
        } else {
            return unknown();
        };
        keys.push(Some(dict_key));
        values.push(value);
    }
    collection_models::dict_literal_value(&keys, &values)
}

/// The single-clause comprehension shape shared by every comprehension
/// form: exactly one `Comprehension` clause, synchronous, over a known
/// `Kind::List` iterable of already-known elements, with a target that
/// is either a bare `Expr::Name` (one name, bound to the WHOLE element)
/// or a two-element `Expr::Tuple`/`Expr::List` of bare names (bound to
/// a `[first, second]` 2-element `Kind::List` element — the exact shape
/// `.items()`'s own pair-lists build, `dict_view_method_result`'s own
/// doc; a `for k, v in d.items():`-style unpacking target,
/// expressions.rst's "Displays for lists, sets and dictionaries": a
/// comprehension's `for` clause follows the SAME target-list grammar an
/// ordinary `for` statement does). `None` for anything outside that
/// shape (multiple clauses, `async for`, a target of any other arity or
/// shape, an unknown/non-List iterable) — the honest decline every
/// comprehension form shares before either evaluates its own
/// element/key expression. The target names and the `if` conditions
/// both borrow from `generators` itself (`'a`), so a caller walking the
/// returned elements still has the clause's own filter list in hand
/// with no second destructure of `generators`.
pub(super) fn comprehension_target_and_elements<'a>(
    generators: &'a [ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(Vec<&'a str>, &'a [Expr], Vec<AbstractValue>)> {
    let [clause] = generators else {
        return None;
    };
    if clause.is_async {
        return None;
    }
    let target_names = comprehension_target_names(&clause.target)?;
    let iterable = evaluate_expression(&clause.iter, environment, kernel);
    if iterable.kind != Kind::List {
        return None;
    }
    Some((target_names, &clause.ifs, iterable.items))
}

/// The single-clause comprehension shape over an UNKNOWN-LENGTH,
/// known-element-set iterable: `Kind::Set` whose only form is the
/// repetition window `as_repetition` reads back
/// (`check.rs::seed_parameters`'s own `list[X]`/`set[X]`/`Sequence[X]`
/// PARAMETER seed builds the bare star, `lo` 0 and `hi` unbounded;
/// `collection_models::star_element_read`'s own doc — the same window
/// shape, read the same way, never a second reader). Every position of
/// a repetition draws from the SAME element set (the grammar's own
/// definition), so there is exactly ONE abstraction to bind the target
/// against and exactly ONE evaluation of `elt` to perform — unlike the
/// concrete path above, which evaluates `elt` once per known item.
/// `None` for the same shape restrictions the concrete path takes (a
/// second `for` clause, `async for`, a target of any other arity), OR
/// when the iterable does not read back as a repetition at all (a
/// union, an unknown value) — the concrete arm and this one are
/// mutually exclusive on `iterable.kind`, so a caller tries the
/// concrete path first and only reaches here on ITS decline. The
/// window's own `{lo, hi}` rides back alongside the element so the
/// caller can restate it on the mapped result.
///
/// The SOURCE NAME — the iterable's own spelling, when `clause.iter` is
/// a plain `Expr::Name` — rides back too, `None` for any other iterable
/// expression (a call, an attribute read, a subscript). The caller uses
/// it to record that the mapped result's own length is proved equal to
/// that name's (`AbstractValue::same_length_as`), which only holds for
/// a plain-name source: an iterable built by an expression has no
/// single binding whose later `len(...)` this value could be tied to.
pub(super) fn comprehension_target_and_star_element<'a>(
    generators: &'a [ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(Vec<&'a str>, &'a [Expr], AbstractValue, i64, Option<i64>, Option<&'a str>)> {
    let [clause] = generators else {
        return None;
    };
    if clause.is_async {
        return None;
    }
    let target_names = comprehension_target_names(&clause.target)?;
    let source_name = match &clause.iter {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    };
    let iterable = evaluate_expression(&clause.iter, environment, kernel);
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    let repeated = as_repetition(&iterable.set)?;
    // The element's own sort is the SEQUENCE's tag, not re-derived: the
    // sequence value carries `kind_tag` off its declared element sort
    // (`check.rs::seed_parameters`'s sequence-container arm), so peeling
    // one element out of the repetition keeps that same tag rather than
    // rebuilding it — `min_max_scalar_operand` (builtin_models.rs) reads
    // an element pulled this way as a `Kind::Set` operand and needs its
    // own `kind_tag` to answer `min`/`max` over two comprehension-bound
    // names.
    let element = AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(repeated.element, None, TrustSpec, SetKindTag::None)
    };
    Some((target_names, &clause.ifs, element, repeated.lo, repeated.hi, source_name))
}

/// The star-shaped result of a list/set/generator comprehension over an
/// unknown-length, known-element-set iterable
/// (`comprehension_target_and_star_element`'s own doc): binds the
/// target to the ONE element abstraction and evaluates `elt` once —
/// there is no per-element enumeration to run since the source length
/// is unstated. The `if` clauses are NEVER evaluated for their
/// truthiness here (unlike the concrete path, which drops individual
/// elements one at a time): whether a given filter keeps or drops any
/// one position is unknowable from a single shared element abstraction
/// standing for the whole source, but that unknowability is exactly
/// what the `lo` widening below already states — "some positions may
/// not survive" — so a filter this file cannot decide narrows nothing
/// FURTHER than a filter it could. `c-reads-and-values.py`'s own
/// `math_min_max_over_declared_element_set` doc states the general
/// law this composes: "every item the real call could draw IS a member
/// of [the element set] (the star grammar's own definition)" — a
/// filter can only ever SHRINK which positions of that same element
/// set survive, never admit a value the source's own element set did
/// not already admit, so the mapped result's window is sound whether
/// or not the filter's own truthiness is legible. A comprehension
/// preserves the source's own length (mapping every position through
/// `elt` changes no position's presence) — the result carries the SAME
/// `{lo, hi}` window the source read back — UNLESS an `if` clause is
/// present, in which case a filter can drop positions down to zero, so
/// `lo` widens to 0 whenever `conditions` is non-empty; `hi` is
/// unaffected either way (a filter only ever removes positions, never
/// adds them).
///
/// SOUNDNESS LINE for `AbstractValue::same_length_as`: the result's
/// length is proved EQUAL to the source's own length -- not merely
/// bounded by the same window -- only when `conditions` is empty. A
/// filtered comprehension can drop positions, so `len(result) <=
/// len(source)` but never provably `==`; `same_length_as` must NOT be
/// set in that case, on pain of `relational_sum.rs::is_len_of`
/// accepting a division by a count the accumulation never actually ran
/// over. This mirrors the `lo` widening below: both readings state the
/// same fact (whether every position survived), once as a window bound
/// and once as a name link.
pub(super) fn comprehension_star_elements(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let (target_names, conditions, element, source_lo, source_hi, source_name) =
        comprehension_target_and_star_element(generators, environment, kernel)?;
    let mut fork = environment.fork();
    if !bind_comprehension_target(&mut fork, &target_names, &element) {
        return None;
    }
    // Every `if` clause the target's own SET binding survives narrows
    // that binding before `elt` is evaluated — `[x for x in lst if x >
    // 4]`'s own surviving-element shape (A7.xfer.filter.py): a
    // one-name target's own current binding is exactly the SAME
    // `Kind::Set` window `narrowing::assume`'s SET channel already
    // narrows for an ordinary `if` guard, so this folds each clause
    // through that identical channel rather than a second reader.
    // Two-name (`.items()` pair) targets bind through TWO names at
    // once, which `assume`'s own name-keyed channels already read
    // independently per name — no special casing needed here. A
    // condition this channel does not recognize narrows nothing,
    // matching every other declined leaf's honest default; the
    // resulting `elt` reading is then exactly as wide as before this
    // filter existed, never narrower than sound.
    for condition in conditions {
        fork = narrowing::assume(condition, fork, kernel, true);
    }
    let mapped = evaluate_expression(element_expr, &fork, kernel);
    if mapped.kind != Kind::Set {
        return None; // the mapped element must itself name a scalar set to re-window over
    }
    let lo = if conditions.is_empty() { source_lo } else { 0 };
    let window = repetition(mapped.set.clone(), lo, source_hi);
    // `conditions.is_empty()` gates this the SAME way it gates `lo`
    // above: a filter can drop positions, so the length link would be
    // an unproved claim once one is present.
    let same_length_as = if conditions.is_empty() {
        source_name.map(str::to_owned)
    } else {
        None
    };
    Some(AbstractValue {
        kind_tag: mapped.kind_tag,
        same_length_as,
        ..known_set(window, None, TrustSpec, SetKindTag::None)
    })
}

/// The bare names a comprehension `for` target binds: one name for a
/// plain `Expr::Name` target, or two names for a two-element
/// `Expr::Tuple`/`Expr::List` target of bare names (`for k, v in
/// ...`-style unpacking). `None` for any other target shape (more than
/// two names, a non-Name element, a nested/starred target) — this file
/// does not model general destructuring targets, only the plain and
/// two-name-tuple shapes a dict `.items()` walk needs.
pub(super) fn comprehension_target_names(target: &Expr) -> Option<Vec<&str>> {
    match target {
        Expr::Name(name) => Some(vec![name.id.as_str()]),
        Expr::Tuple(tuple) => {
            let [Expr::Name(first), Expr::Name(second)] = tuple.elts.as_slice() else {
                return None;
            };
            Some(vec![first.id.as_str(), second.id.as_str()])
        }
        _ => None,
    }
}

/// Binds a comprehension target's names against one source element: a
/// single-name target binds the WHOLE element; a two-name target
/// requires the element to be a known 2-element `Kind::List` (a
/// `.items()` pair, per `comprehension_target_names`'s own doc) and
/// binds each name to its own slot. `false` if a two-name target meets
/// an element that is not that exact shape — the caller must treat that
/// as an undecidable element, not silently bind partial names.
pub(super) fn bind_comprehension_target(fork: &mut Environment, target_names: &[&str], element: &AbstractValue) -> bool {
    match target_names {
        [name] => {
            fork.bind(name, element.clone());
            true
        }
        [first, second] => {
            if element.kind != Kind::List || element.items.len() != 2 {
                return false;
            }
            fork.bind(first, element.items[0].clone());
            fork.bind(second, element.items[1].clone());
            true
        }
        _ => false,
    }
}

/// The surviving elements of a list/set/generator comprehension: walks
/// `comprehension_target_and_elements`'s own element sequence, forking
/// the environment and binding the target for each one, filtering by
/// every `if` clause's truthiness, and evaluating `element_expr` on the
/// elements that survive every filter. `None` the moment the shape is
/// outside what this file models, OR an `if` clause's truthiness cannot
/// be decided for some element.
pub(super) fn comprehension_elements(
    element_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let (target_names, conditions, source_elements) =
        comprehension_target_and_elements(generators, environment, kernel)?;
    let mut out = Vec::new();
    for element in source_elements {
        let mut fork = environment.fork();
        if !bind_comprehension_target(&mut fork, &target_names, &element) {
            return None;
        }
        if !comprehension_conditions_hold(conditions, &fork, kernel)? {
            continue;
        }
        out.push(evaluate_expression(element_expr, &fork, kernel));
    }
    Some(out)
}

/// The surviving `(key, value)` pairs of a dict comprehension — the same
/// per-element fork/bind/filter walk `comprehension_elements` performs,
/// evaluating both `key_expr` and `value_expr` on each surviving fork.
pub(super) fn comprehension_rows(
    key_expr: &Expr,
    value_expr: &Expr,
    generators: &[ruff_python_ast::Comprehension],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(AbstractValue, AbstractValue)>> {
    let (target_names, conditions, source_elements) =
        comprehension_target_and_elements(generators, environment, kernel)?;
    let mut out = Vec::new();
    for element in source_elements {
        let mut fork = environment.fork();
        if !bind_comprehension_target(&mut fork, &target_names, &element) {
            return None;
        }
        if !comprehension_conditions_hold(conditions, &fork, kernel)? {
            continue;
        }
        let key = evaluate_expression(key_expr, &fork, kernel);
        let value = evaluate_expression(value_expr, &fork, kernel);
        out.push((key, value));
    }
    Some(out)
}

/// Every `if` condition of one comprehension clause, evaluated in order
/// against `environment` (the fork with this element's target already
/// bound): `Some(true)` when every condition is definitely truthy (the
/// element survives), `Some(false)` the moment one condition is
/// definitely falsy (the element is dropped, remaining conditions are
/// not evaluated — matching Python's own left-to-right short-circuit
/// evaluation order for chained comprehension `if`s), `None` the moment
/// one condition's truthiness cannot be decided at all.
pub(super) fn comprehension_conditions_hold(
    conditions: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<bool> {
    for condition in conditions {
        let value = evaluate_expression(condition, environment, kernel);
        let (truthy, known) = truthiness(&value);
        if !known {
            return None;
        }
        if !truthy {
            return Some(false);
        }
    }
    Some(true)
}

/// A NumberLiteral's own value: an int that fits i64 tags `Integer`, a
/// float literal tags `Float` — the syntax's own sort, read once at the
/// value's construction rather than re-derived from the AST at every
/// arithmetic site (PYREFLY-NUMERIC-B3-B4.md's "two sorts, never one
/// Number"). A complex literal, or an int too big for i64, is honest
/// unknown rather than a truncated stand-in.
pub(super) fn number_literal_value(number: &Number) -> AbstractValue {
    match number {
        Number::Int(int) => match int.as_i64() {
            Some(value) => known_values(vec![value as f64], PrimitiveKind::Integer, TrustProved),
            None => unknown(),
        },
        Number::Float(value) => known_values(vec![*value], PrimitiveKind::Float, TrustProved),
        Number::Complex { .. } => unknown(),
    }
}
