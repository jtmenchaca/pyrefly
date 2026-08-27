//! Comparison leaves: numeric/literal/membership/length narrows and
//! `NumericCmpOp`.

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::element_of_object_star;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;

use crate::collection_models::DictKey;
use crate::env::Environment;

use super::literal_number;
use super::name_of;
use super::none_truthiness::narrow_bool_literal_comparison;
use super::none_truthiness::narrow_is_none;

/// `ExprCompare` narrowing: chained comparisons lower to the
/// conjunction of adjacent pairs (see the module doc's CPython
/// citation). Under falsity, a conjunction narrows nothing (the same
/// rule `and`-under-falsity follows in `narrow_bool_op`) — the chain's
/// negation is a disjunction over which pair failed, and this wave
/// holds no union.
pub(super) fn narrow_compare(compare: &ruff_python_ast::ExprCompare, environment: &mut Environment, truth: bool) {
    if !truth {
        // is/is not None still narrows under falsity for a single pair
        // (mission point 5) — handled directly here since it is not a
        // conjunction the same way numeric comparisons are.
        if compare.ops.len() == 1 {
            narrow_one_comparison(&compare.left, compare.ops[0], &compare.comparators[0], environment, false);
        }
        return;
    }
    let mut left = compare.left.as_ref();
    for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
        narrow_one_comparison(left, *op, right, environment, true);
        left = right;
    }
}

/// One comparison pair (`left op right`) as a narrowing leaf: is/is not
/// None (mission point 5), then numeric literal-side comparisons
/// (mission point 1), mirrored so the literal may sit on either side, then
/// membership against a literal collection (`in`/`not in` — closes the
/// match-guard lane's own scope note: `x in (2, 4)` now narrows a
/// `Kind::Values` binding the same pointwise way a numeric comparison
/// does). Anything else — a call, an attribute, a string, two changing
/// names — narrows nothing.
pub(super) fn narrow_one_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    if matches!(op, CmpOp::Is | CmpOp::IsNot) {
        narrow_is_none(left, op, right, environment, truth);
        narrow_bool_literal_comparison(left, op, right, environment, truth);
        return;
    }
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        // `b == True` on a bool-domain binding — read by the same leaf as
        // `b is True` (CPython interns the two bools, so the pair
        // coincides there); the numeric paths below still read the same
        // op for every other operand shape.
        narrow_bool_literal_comparison(left, op, right, environment, truth);
        narrow_name_against_list_literal(left, op, right, environment, truth);
    }
    if matches!(op, CmpOp::In | CmpOp::NotIn) {
        if let Some(name) = name_of(left) {
            narrow_name_against_membership(name, right, environment, op == CmpOp::In, truth);
        }
        narrow_dict_membership_against_literal_key(left, right, environment, op == CmpOp::In, truth);
        return;
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return;
    };
    if let (Some(name), Some(literal)) = (len_call_name(left), literal_number(right)) {
        narrow_name_length_against_literal(name, numeric_op, literal, environment, truth);
        return;
    }
    if let (Some(literal), Some(name)) = (literal_number(left), len_call_name(right)) {
        narrow_name_length_against_literal(name, mirror_cmp_op(numeric_op), literal, environment, truth);
        return;
    }
    if let (Some(name), Some(literal)) = (name_of(left), literal_number(right)) {
        narrow_name_against_literal(name, numeric_op, literal, environment, truth);
        return;
    }
    if let (Some(literal), Some(name)) = (literal_number(left), name_of(right)) {
        narrow_name_against_literal(name, mirror_cmp_op(numeric_op), literal, environment, truth);
        return;
    }
}

/// `name == [k1, k2, …]` / `name != [...]` (expressions.rst,
/// "Comparisons" — sequences compare by VALUE in Python, unlike JS/C++
/// pointer identity, C8's own module-doc citation): the true arm of `==`
/// (or the FALSE arm of `!=`, its complement — `narrow_is_none`'s own
/// "false arm still states a fact" precedent) proves `name` is EXACTLY
/// the literal list, so `name` rebinds to a fresh `Kind::List` whose own
/// items are the literal's own number-literal elements, each a
/// `Kind::Values` singleton — the same shape `expressions.rs::evaluate_
/// list` builds for an ordinary list display, read back through
/// `subscript_read`'s existing `Kind::List` arm with no new reader
/// needed. Only a literal list of plain number literals on the OTHER
/// side of a bare-name comparison narrows (mirroring `literal_numeric_
/// collection`'s own scope): a non-list literal, a non-numeric element, a
/// name on both sides, or the OPPOSITE truth arm of either operator
/// (`==` false, `!=` true — a value merely EXCLUDED from being this one
/// exact list has no single narrower shape this window can name) narrows
/// nothing, the honest default.
pub(super) fn narrow_name_against_list_literal(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let proves_equal = (op == CmpOp::Eq) == truth;
    if !proves_equal {
        return;
    }
    let (name, literal_side) = if let Some(name) = name_of(left) {
        (name, right)
    } else if let Some(name) = name_of(right) {
        (name, left)
    } else {
        return;
    };
    let Expr::List(list) = literal_side else {
        return;
    };
    let Some(numbers): Option<Vec<f64>> = list.elts.iter().map(literal_number).collect() else {
        return;
    };
    let items: Vec<AbstractValue> = numbers
        .into_iter()
        .map(|value| known_values(vec![value], PrimitiveKind::Integer, TrustSpec))
        .collect();
    environment.bind(name, refined_domain::known_constructors::known_list(items, TrustSpec));
}

/// `name == {k1: v1, ...}` / `name != {...}` — the dict-display twin of
/// `narrow_name_against_list_literal`: the true arm of `==` (or the FALSE
/// arm of `!=`, its complement, the same reading `narrow_name_against_
/// list_literal` already gives) proves `name` is EXACTLY the literal
/// dict, so `name` rebinds to a fresh `Kind::Object` built the identical
/// way `expressions.rs::evaluate_dict` builds an ordinary dict display —
/// `collection_models::known_dict_key` reads each key expression through
/// the SAME table a later `d["a"]` subscript read uses
/// (`collection_models::subscript_read`'s own `Kind::Object` arm), so a
/// key this guard proves and a key a later read asks for match by
/// identical spelling. A `**spread` entry (`item.key` is `None`), a key
/// this table cannot reduce to a string/int/identity slot, or a value
/// expression `evaluate_expression` cannot itself resolve declines the
/// WHOLE literal (`collection_models::dict_literal_value`'s own
/// all-keys-must-be-Some honesty) — a partially-known dict has no exact
/// shape this leaf can rebind `name` to. Only a literal dict display on
/// the OTHER side of a bare-name comparison narrows, and only the arm
/// that proves equality: a non-dict literal, a name on both sides, or the
/// OPPOSITE truth arm of either operator narrows nothing, the honest
/// default.
pub(super) fn narrow_name_against_dict_literal(
    compare: &ruff_python_ast::ExprCompare,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) {
    if compare.ops.len() != 1 || compare.comparators.len() != 1 {
        return;
    }
    let op = compare.ops[0];
    if !matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        return;
    }
    let proves_equal = (op == CmpOp::Eq) == truth;
    if !proves_equal {
        return;
    }
    let left = compare.left.as_ref();
    let right = &compare.comparators[0];
    let (name, literal_side) = if let Some(name) = name_of(left) {
        (name, right)
    } else if let Some(name) = name_of(right) {
        (name, left)
    } else {
        return;
    };
    let Expr::Dict(dict) = literal_side else {
        return;
    };
    let mut keys: Vec<Option<crate::collection_models::DictKey>> = Vec::new();
    let mut values: Vec<AbstractValue> = Vec::new();
    for item in &dict.items {
        let Some(key_expr) = item.key.as_ref() else {
            // a `**spread` entry — this leaf reads only a fully literal
            // display, the same restriction `dict_literal_value`'s own
            // "even one unsupported key" honesty already gives
            return;
        };
        let key_value = crate::expressions::evaluate_expression(key_expr, environment, kernel);
        let Some(dict_key) = crate::collection_models::known_dict_key(&key_value) else {
            return;
        };
        keys.push(Some(dict_key));
        values.push(crate::expressions::evaluate_expression(&item.value, environment, kernel));
    }
    let built = crate::collection_models::dict_literal_value(&keys, &values);
    if built.kind != Kind::Object {
        // dict_literal_value itself declines (a repeated key collapsed
        // to a shape it still refused, or an internal inconsistency) —
        // narrow nothing rather than bind a name to an unknown() value
        return;
    }
    environment.bind(name, built);
}

/// Whether `expression` is `len(<bare name>)` — the one shape
/// `narrow_name_length_against_literal` reads on the tested side, the
/// `len(...)`-wrapped twin of `name_of`'s bare-identifier restriction.
/// `len` called on anything other than a single bare name (an
/// attribute, a call, a literal) is not this leaf's business —
/// `narrow_one_comparison` falls through unchanged for it, the same
/// "narrows nothing" default every unread leaf shape keeps.
pub(super) fn len_call_name(expression: &Expr) -> Option<&str> {
    let Expr::Call(call) = expression else { return None };
    let Expr::Name(func_name) = call.func.as_ref() else { return None };
    if func_name.id.as_str() != "len" {
        return None;
    }
    let [only] = &*call.arguments.args else { return None };
    name_of(only)
}

/// Narrows a Set-kind binding named `name` by `len(name) op literal`
/// being `truth`: `ages: list[Age]` (no `min_length`/`max_length` in
/// its own surface) seeds `check.rs::seed_parameters`'s star-repetition
/// shape (`refined_sets::refinement_forms::repeat_of`, `lo` 0, `hi`
/// `None` — the bare unbounded window, `typereading.rs`'s own doc for
/// a length-unconstrained sequence parameter) — a window `min_max_over_
/// star` (`builtin_models.rs`) REFUSES to read for `min`/`max` while
/// `lo` could still be 0 (CPython's `ValueError` on an empty sequence).
/// `if len(ages) >= 1:` is exactly the guard that fixture's own doc
/// names as "what a real caller must write to make this call safe at
/// all" — this is the narrowing that makes the checker SEE that guard:
/// under `len(name) >= k` truth (or the mirrored `k <= len(name)`),
/// `lo` tightens to `max(lo, k)`; under `len(name) <= k`/`== k`, `hi`
/// tightens to `min(hi.unwrap_or(k), k)`; under `len(name) > k`, `lo`
/// tightens to `max(lo, k + 1)`; under `len(name) < k`, `hi` tightens
/// the same way one below `k`. Falsity mirrors through `satisfies`'
/// own negation the same way `narrow_name_against_literal` does — a
/// COMPARISON's false arm still states a fact (`not (len(ages) >= 1)`
/// is `len(ages) < 1`, i.e. `len(ages) == 0`, the empty-window case).
///
/// Reads and rebuilds through `as_repetition`/`repeat_of`
/// (`refined_sets::refinement_forms`, `repetition_window_forms`) — the
/// same {element, lo, hi} triple `check.rs`'s own seeding and
/// `min_max_over_star`'s own reading already agree on, so this adds no
/// new window shape, only a narrower `lo`/`hi` on the existing one. A
/// binding that is not `Kind::Set`, or a `Kind::Set` whose own top
/// layer is not this exact repetition shape (a plain numeric range —
/// `len` has no meaning there — or a fixed-arity `Kind::List` already
/// read through the Values-shaped element channel), narrows nothing.
pub(super) fn narrow_name_length_against_literal(name: &str, op: NumericCmpOp, literal: f64, environment: &mut Environment, truth: bool) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Set {
        return;
    }
    let Some(repeated) = refined_sets::repetition_window_forms::as_repetition(&current.set) else {
        return;
    };
    if literal < 0.0 || literal.fract() != 0.0 {
        // a length is never negative or fractional; a comparison
        // against one is either vacuous or a construct this leaf does
        // not read — narrow nothing rather than guess
        return;
    }
    let k = literal as i64;
    // `op`/`truth` folds to the single EFFECTIVE operator this leaf
    // narrows under — `satisfies`'s own truth-table, applied once at
    // the operator level rather than per element (a length window has
    // no member list to filter, only two bounds to tighten)
    let effective = if truth { op } else { negate_numeric_cmp_op(op) };
    let (mut lo, mut hi) = (repeated.lo, repeated.hi);
    match effective {
        NumericCmpOp::GtE => lo = lo.max(k),
        NumericCmpOp::Gt => lo = lo.max(k + 1),
        NumericCmpOp::LtE => hi = Some(hi.map_or(k, |current_hi| current_hi.min(k))),
        NumericCmpOp::Lt => hi = Some(hi.map_or(k - 1, |current_hi| current_hi.min(k - 1))),
        NumericCmpOp::Eq => {
            lo = lo.max(k);
            hi = Some(hi.map_or(k, |current_hi| current_hi.min(k)));
        }
        // `!=` excludes one point from an interval, which the {lo, hi}
        // window vocabulary cannot state — narrows nothing, the same
        // "no shape for this" decline `narrow_name_against_literal`'s
        // own Values channel never needs (it filters pointwise instead)
        NumericCmpOp::NotEq => return,
    }
    if let Some(h) = hi {
        if h < lo {
            // the window is now provably empty — every leaf in this
            // file leaves an infeasible-branch binding UNCHANGED
            // (`narrow_name_against_literal`'s own "zero survivors"
            // comment states the twin case for a Values binding); the
            // walk's own dead-branch handling is what skips the body,
            // not a narrowed-to-empty rebind here
            return;
        }
    }
    let grade = trust_level_of(&current);
    let narrowed_set = refined_sets::refinement_forms::make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
        repeated.element,
        lo,
        hi,
    )]);
    environment.bind(
        name,
        AbstractValue {
            kind_tag: current.kind_tag,
            ..known_set(narrowed_set, None, grade, SetKindTag::None)
        },
    );
}

/// The strict negation of one `NumericCmpOp` — `not (x >= k)` is
/// `x < k`, etc. — the operator-level mirror of `satisfies`'s own
/// per-element negation (`satisfies(value, op, literal) == truth`),
/// needed here because a length window narrows by tightening a BOUND,
/// not by filtering a member list, so the falsity case folds to a
/// different effective operator up front rather than at each element.
pub(super) fn negate_numeric_cmp_op(op: NumericCmpOp) -> NumericCmpOp {
    match op {
        NumericCmpOp::Lt => NumericCmpOp::GtE,
        NumericCmpOp::LtE => NumericCmpOp::Gt,
        NumericCmpOp::Gt => NumericCmpOp::LtE,
        NumericCmpOp::GtE => NumericCmpOp::Lt,
        NumericCmpOp::Eq => NumericCmpOp::NotEq,
        NumericCmpOp::NotEq => NumericCmpOp::Eq,
    }
}

/// The subset of `CmpOp` this wave's numeric side-bounds filter reads:
/// `< <= > >= == !=`. `is`/`is not` are handled by `narrow_is_none`;
/// `in`/`not in` by `narrow_name_against_membership` on a Values binding, and
/// by the SET channel's own `membership_leaf_tree_of` on a Set binding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum NumericCmpOp {
    Lt,
    LtE,
    Gt,
    GtE,
    Eq,
    NotEq,
}

pub(super) fn numeric_cmp_op(op: CmpOp) -> Option<NumericCmpOp> {
    match op {
        CmpOp::Lt => Some(NumericCmpOp::Lt),
        CmpOp::LtE => Some(NumericCmpOp::LtE),
        CmpOp::Gt => Some(NumericCmpOp::Gt),
        CmpOp::GtE => Some(NumericCmpOp::GtE),
        CmpOp::Eq => Some(NumericCmpOp::Eq),
        CmpOp::NotEq => Some(NumericCmpOp::NotEq),
        CmpOp::Is | CmpOp::IsNot | CmpOp::In | CmpOp::NotIn => None,
    }
}

/// Mirrors the operator when the literal was on the left (`k >= x`
/// means `x <= k`).
pub(super) fn mirror_cmp_op(op: NumericCmpOp) -> NumericCmpOp {
    match op {
        NumericCmpOp::Lt => NumericCmpOp::Gt,
        NumericCmpOp::LtE => NumericCmpOp::GtE,
        NumericCmpOp::Gt => NumericCmpOp::Lt,
        NumericCmpOp::GtE => NumericCmpOp::LtE,
        NumericCmpOp::Eq => NumericCmpOp::Eq,
        NumericCmpOp::NotEq => NumericCmpOp::NotEq,
    }
}

/// Whether a known single value `v` satisfies `v op literal` — applied
/// pointwise over a Values binding's exact members in
/// `narrow_name_against_literal`.
pub(super) fn satisfies(value: f64, op: NumericCmpOp, literal: f64) -> bool {
    match op {
        NumericCmpOp::Lt => value < literal,
        NumericCmpOp::LtE => value <= literal,
        NumericCmpOp::Gt => value > literal,
        NumericCmpOp::GtE => value >= literal,
        NumericCmpOp::Eq => value == literal,
        NumericCmpOp::NotEq => value != literal,
    }
}

/// Narrows a Values-kind binding named `name` by `name op literal`
/// being `truth` (mission point 1): keep exactly the members
/// satisfying the (possibly negated) predicate. Zero survivors bind the
/// empty Values state — sound infeasibility (this branch arm's Values
/// set is empty, so any read of `name` inside it answers from no
/// members rather than answering unknown).
pub(super) fn narrow_name_against_literal(
    name: &str,
    op: NumericCmpOp,
    literal: f64,
    environment: &mut Environment,
    truth: bool,
) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // a numeric-sorted binding (any of Number/Integer/Float — sort-
    // unknown or sort-known — or Boolean, which reads numerically the
    // same way Python's `True + True == 2` does); a String/Array-tagged
    // binding is not read as a number here
    if !is_numeric_or_boolean(kind_tag) {
        return;
    }
    let grade = trust_level_of(&current);
    let kept: Vec<f64> = current
        .values
        .iter()
        .copied()
        .filter(|&value| satisfies(value, op, literal) == truth)
        .collect();
    environment.bind(name, known_values(kept, kind_tag, grade));
}

/// Whether `kind_tag` reads numerically for a literal comparison:
/// Number (sort-unknown), Integer, Float, or Boolean (`True`/`False`
/// compare as `1`/`0`). String and Array are not numeric.
pub(super) fn is_numeric_or_boolean(kind_tag: PrimitiveKind) -> bool {
    matches!(
        kind_tag,
        PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float | PrimitiveKind::Boolean
    )
}

/// Narrows a Values-kind binding named `name` by `name in <collection>` (or
/// `not in`, `is_in: false`) being `truth`: keep exactly the members that
/// are (`is_in`) or are not (`!is_in`) themselves a member of `collection`'s
/// own literal elements, mirroring `narrow_name_against_literal`'s pointwise
/// filter one-for-one — membership is read directly against the collection's
/// literal numbers here rather than through the kernel's `NarrowTree`/`assume`
/// ask that channel takes for a `Kind::Set` binding (`membership_leaf_tree_of`
/// builds that tree for the SET channel; this is the VALUES channel's own
/// leaf, over an already-enumerated binding, so the members are just read and
/// filtered, the same way every other Values leaf in this file narrows).
/// `collection` must be a literal list/tuple/set of plain number literals
/// (mirroring `membership_leaf_tree_of`'s own numeric half) — anything else
/// (a name, a comprehension, a mixed or string collection) narrows nothing,
/// the same "no shape this file reads" default every other declined leaf
/// gives. Zero survivors bind the empty Values state, the same sound
/// infeasibility `narrow_name_against_literal` gives.
pub(super) fn narrow_name_against_membership(name: &str, collection: &Expr, environment: &mut Environment, is_in: bool, truth: bool) {
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    if !is_numeric_or_boolean(kind_tag) {
        return;
    }
    let Some(members) = literal_numeric_collection(collection) else {
        return;
    };
    // `name in <collection>` true, or `name not in <collection>` false,
    // both mean "keep the members present in the collection"; the other
    // two combinations keep the members ABSENT from it — the same
    // `is_in == truth` flip `narrow_name_against_literal`'s own
    // `satisfies(...) == truth` gives a single predicate.
    let keep_present = is_in == truth;
    let grade = trust_level_of(&current);
    let kept: Vec<f64> = current
        .values
        .iter()
        .copied()
        .filter(|value| members.contains(value) == keep_present)
        .collect();
    environment.bind(name, known_values(kept, kind_tag, grade));
}

/// `"a" in d` / `"a" not in d` / `k in d` / `k not in d` — the mirror of
/// `narrow_name_against_membership`'s shape: the tested NAME is here the
/// dict `d`, on the RIGHT of `in`/`not in`, with either a plain string
/// literal (`"a" not in d`, A8.xfer.delete's own exit-guard row) or a
/// plain NAME (`k in d`, A8.guard.forget's own `read_after_key_rebind`;
/// `k in m` over a WEAK map, A8.xfer.weak's own `guarded_weak_read`) on
/// the left. Reads only an unbounded-key dict binding (`Kind::ObjectStar`
/// — `check.rs::seed_parameters`'s `known_dict_star` seed for a
/// `dict[str, X]`/`weakref.WeakKeyDictionary[K, X]` parameter), the shape
/// a raise-guard needs: the star states no key list of its own, so
/// `known_container_index_absent` cannot prove absence for a later read
/// the way a closed `Kind::Object` dict already can
/// (`expressions::compare::compare_pair`'s own `Kind::Object` arm reads
/// the key set directly there).
///
/// The arm that proves PRESENCE (`in` true, or `not in` false) records a
/// fresh `ObjectKey` entry, the same (name, numeric) identity
/// `known_dict_key`/`ObjectKey` already share
/// (`collection_models::dict_write`'s own star-write arm records the
/// identical shape for `d[key] = value`) — every entry the receiver
/// already recorded survives untouched, since a membership test writes
/// nothing to the dict itself (stdtypes.rst, "Mapping Types," `key in
/// d`: "Return `True` if *d* has a key *key*, else `False`" — a pure
/// read). The recorded value is the star's own declared element
/// (`element_of_object_star`) — the ONE claim the parameter's own
/// declaration already makes about every present key's value, not a
/// narrower one this test could not have proved. A key already
/// recorded (an earlier write or an earlier membership guard) is left
/// as-is rather than overwritten, since this leaf has no narrower value
/// to add.
///
/// A STRING LITERAL on the left keys the entry by its own text
/// (`DictKey::string`) — the same value-comparable key `d[key] = value`
/// records. A plain NAME on the left keys the entry by that BINDING's
/// own identity (`DictKey::identity`, tagged `"binding:<name>"`) rather
/// than by anything the name's VALUE states — this is what lets a
/// class-instance key (a weak-referenceable `_Key` parameter, no
/// `instance_identity` of its own to read `known_dict_key`'s ordinary
/// identity arm) still record presence: `key in m` and `m[key]` name the
/// same runtime object because the same UNWRITTEN BINDING supplies it on
/// both sides (stdtypes.rst's Mapping Types section: membership and
/// subscript consult the same keys), not because the value itself is
/// spellable. `Environment::bind`'s own doc states the staleness half:
/// any write to the key binding, OR to the receiver binding, drops the
/// fact — a write to `d`/`m` replaces its whole `Kind::ObjectStar` value
/// (this entry along with it), and a write to `k`/`key` strips every
/// `"binding:<name>"`-tagged entry naming it, from every receiver, the
/// moment the name is rebound.
///
/// The arm that proves ABSENCE (`in` false, or `not in` true) narrows
/// nothing: `Kind::ObjectStar` has no key-list slot to record a missing
/// key in (the same asymmetry `expressions::compare::compare_pair`'s
/// own star arm documents — presence is recordable, absence is not),
/// and inventing one would let a later `del`/write silently resurrect a
/// key this checker had wrongly called impossible.
///
/// Any other left/right shape (the left is neither a string literal nor
/// a plain name, the name is not currently `Kind::ObjectStar`, or the
/// receiver's star element cannot be read) narrows nothing — the honest
/// default.
pub(super) fn narrow_dict_membership_against_literal_key(
    left: &Expr,
    right: &Expr,
    environment: &mut Environment,
    is_in: bool,
    truth: bool,
) {
    let Some(name) = name_of(right) else {
        return;
    };
    let inner_key = match left {
        Expr::StringLiteral(literal) => DictKey::string(literal.value.to_str()),
        Expr::Name(key_name) => DictKey::identity(&format!("binding:{}", key_name.id.as_str())),
        _ => return,
    };
    // Recorded under the GUARD provenance, never the inner key's own
    // plain spelling: this test proves presence AT THE GUARD, not that
    // the key survives to the read (`DictKey::guarded`'s own doc) — a
    // mutation between here and a later read, including one inside a
    // callee handed the receiver, can remove it, which the WRITTEN-key
    // shortcut (`dict_star_get_result`) must never assume away.
    let key = DictKey::guarded(&inner_key);
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::ObjectStar {
        return;
    }
    let proves_present = is_in == truth;
    if !proves_present {
        return;
    }
    let Some(element) = element_of_object_star(&current) else {
        return;
    };
    if current.keys.iter().any(|entry| entry.name == key.name && entry.numeric == key.numeric) {
        return;
    }
    let mut narrowed = current;
    narrowed.keys.push(ObjectKey {
        name: key.name,
        numeric: key.numeric,
        value: element,
    });
    environment.bind(name, narrowed);
}

/// A literal list/tuple/set of plain number literals, read as `f64`s — the
/// numeric half of `membership_leaf_tree_of`'s own element reading, kept as
/// a separate small reader here since this file's own convention
/// (`literal_number`'s doc) is a leaf reader per file rather than a shared
/// cross-file helper. An empty collection, a non-literal collection, or one
/// with any non-numeric member (a string, a name, a nested expression)
/// answers `None` — declined, never partially read.
pub(super) fn literal_numeric_collection(collection: &Expr) -> Option<Vec<f64>> {
    let elements: &[Expr] = match collection {
        Expr::List(list) => &list.elts,
        Expr::Tuple(tuple) => &tuple.elts,
        Expr::Set(set) => &set.elts,
        _ => return None,
    };
    if elements.is_empty() {
        return None;
    }
    elements.iter().map(literal_number).collect()
}
