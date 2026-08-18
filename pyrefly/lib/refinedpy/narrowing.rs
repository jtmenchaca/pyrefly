/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Condition narrowing: what a test being true (or false) says about
//! the names it mentions. The walk forks an environment per branch arm
//! and asks this module to tighten each fork before walking the arm.
//! This file is the contract the walk calls; the narrowing unit fills
//! it in behind these signatures.
//!
//! Two channels narrow a name, in order:
//!
//! - The SET channel (refined-ts-go's condition-tree design): every
//!   name the condition mentions that is CURRENTLY bound `Kind::Set`
//!   (a seeded parameter, or a sort-set `isinstance` just built — see
//!   below) has the whole condition lowered, relative to that one
//!   name, into the kernel's own `NarrowTree` grammar
//!   (`condition_tree_of`) — `!`/`and`/`or` fold exactly as
//!   `refined-ts-go/internal/refinedts/conditiontree` folds them, and
//!   every leaf this file cannot express (a call other than
//!   `isinstance`, two changing names, a string test, …) lowers to
//!   `NarrowTreeKind::Other`, which the kernel treats as "no claim." A
//!   tree that says nothing (`says_anything` false) or that the kernel
//!   REFUSES (panics — an unrefined set shape; caught, never read as a
//!   refutation) leaves the binding untouched. A tree the kernel
//!   answers narrows the CURRENT set by intersection (this file's own
//!   `meet_set_answer`), never widens it, and keeps the binding's own
//!   trust floor — the kernel's answer is never claimed stronger than
//!   what already flowed in.
//! - The VALUES channel (today's ad-hoc path, unchanged): a name
//!   currently bound `Kind::Values` keeps being narrowed leaf-by-leaf
//!   exactly as before this wave — comparisons, `is`/`is not None`,
//!   `isinstance`. The two channels never overlap on one binding: a
//!   name is Values-kind or Set-kind, never both, so a leaf reads
//!   whichever channel its current binding is in.
//!
//! `isinstance(x, int | float)` proving true ALSO seeds a fresh
//! `Kind::Set` binding for a name the environment has not bound at all
//! (an `object`-typed parameter, never seeded by `check.rs::
//! seed_parameters` — no alias states anything for the bare `object`
//! annotation) — this is the "(a seeded parameter, a sort-set)" case
//! the mission names: the SORT itself is what `isinstance` proves, and
//! a sort with no further bound IS a `Kind::Set` (the unbounded
//! integer/float ray), exactly the shape `summaries.rs`'s own
//! `return_sort_fallback` builds for a declined `-> int` return. Only
//! `int`/`float` seed a Set this way; `bool`'s domain is the two exact
//! values `{0, 1}`, so a proved `isinstance(x, bool)` seeds
//! `Kind::Values` instead, matching every other Boolean value this
//! domain carries.
//!
//! A binding of any other kind (`Kind::Null`, `Kind::Object`, …)
//! passes through unchanged on both channels: the honest default is to
//! narrow nothing.
//!
//! Chained comparisons lower to a conjunction of adjacent pairs
//! (`a op1 b op2 c` == `a op1 b and b op2 c`, CPython
//! tmp/cpython/Doc/reference/expressions.rst — Comparisons, "Comparisons
//! can be chained arbitrarily"), so `ExprCompare`'s multi-op form and
//! `and`'s multi-value form share one conjunction helper on both
//! channels.

use std::panic::catch_unwind;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::NarrowTree;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::narrow_questions::NarrowCmpOp;
use refined_kernel::narrow_questions::NarrowTreeKind;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::BoolOp;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;

use crate::refinedpy::env::Environment;

/// Tighten `environment` by what `condition` being `truth` says.
/// Returns the narrowed environment for that arm. The honest default
/// narrows nothing — an arm walked with the unnarrowed fork is
/// conservative, never wrong.
pub fn assume(
    condition: &Expr,
    environment: Environment,
    kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) -> Environment {
    let mut environment = environment;
    // the VALUES channel runs FIRST: it is what SEEDS a fresh Set-kind
    // binding from an `isinstance` test on a name the environment held
    // nothing about (`narrow_isinstance_call`'s own doc) — the SAME
    // `and` chain that proves the sort (`isinstance(value, int) and …
    // and 0 <= value <= 120`) also states the comparison bound in this
    // one `assume` call, so the SET channel must see the seeded
    // binding, not run before it exists.
    narrow(condition, &mut environment, kernel, truth);
    narrow_set_kind_names(condition, &mut environment, kernel, truth);
    environment
}

/// The SET channel's own entry point: for every name `condition`
/// mentions (`collect_names`) that is CURRENTLY bound `Kind::Set`,
/// lower the whole condition relative to that name and ask the kernel
/// once. Run AFTER the VALUES channel's leaf walk (`narrow`, `assume`'s
/// own doc explains why) so a name `narrow`'s own `narrow_isinstance_
/// call` just seeded is already `Kind::Set` by the time this runs — one
/// ask per name, by the WHOLE condition at once (`0 <= value <= 120`
/// intersects both bounds in one ask, matching an `and`'s conjunction
/// reading), rather than once per leaf.
fn narrow_set_kind_names(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    let mut names = Vec::new();
    collect_names(condition, &mut names);
    for name in names {
        let Some(current) = environment.read(name.as_str()) else {
            continue;
        };
        if current.kind != Kind::Set {
            continue;
        }
        let Some(tree) = condition_tree_of(condition, name.as_str()) else {
            continue;
        };
        if !says_anything(&tree) {
            continue;
        }
        // a REFUSED question (a set shape the kernel's narrowing decider
        // does not decide) panics inside the kernel closure rather than
        // returning a claim; caught here and treated as "narrows
        // nothing" — never read as a refutation, mirroring
        // assignability.rs's own containment-ask recover() (and
        // refined-ts-go's narrowRefusable).
        let asked = catch_unwind(AssertUnwindSafe(|| (kernel.narrow)(&tree)));
        let Ok(answer) = asked else {
            continue;
        };
        let claim = if truth { answer.when_true } else { answer.when_false };
        let Some(claim) = claim else {
            continue;
        };
        let current = environment.read(name.as_str()).expect("just read above").clone();
        let narrowed = meet_set_answer(&current, &claim.set);
        environment.bind(name.as_str(), narrowed);
    }
}

/// Every bare name a leaf this file recognizes tests — the SET
/// channel's own place collector, scoped to exactly the shapes
/// `condition_tree_of` folds (`!`/`and`/`or`, a `Compare`'s two sides,
/// an `isinstance` call's first argument): wide enough to find every
/// name a tree might be built for, never wider — a name mentioned only
/// inside a shape this file does not read (a call other than
/// `isinstance`, an attribute) is never added, matching
/// `condition_tree_of`'s own leaf vocabulary.
fn collect_names(condition: &Expr, out: &mut Vec<String>) {
    let add = |name: &str, out: &mut Vec<String>| {
        if !out.iter().any(|held| held == name) {
            out.push(name.to_owned());
        }
    };
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => collect_names(&unary.operand, out),
        Expr::BoolOp(bool_op) => {
            for value in &bool_op.values {
                collect_names(value, out);
            }
        }
        Expr::Compare(compare) => {
            if let Some(name) = name_of(&compare.left) {
                add(name, out);
            }
            for comparator in &compare.comparators {
                if let Some(name) = name_of(comparator) {
                    add(name, out);
                }
            }
        }
        Expr::Call(call) => {
            if let Expr::Name(func_name) = call.func.as_ref() {
                if func_name.id.as_str() == "isinstance" && call.arguments.args.len() == 2 {
                    if let Some(name) = name_of(&call.arguments.args[0]) {
                        add(name, out);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Tightens `environment` in place by what `condition` being `truth`
/// says, dispatching on the condition's shape. Every arm that cannot
/// narrow simply returns without touching `environment`. This is the
/// VALUES channel — see the module doc; `kernel` is threaded through
/// only so `narrow_isinstance_call` can seed a fresh Set-kind binding
/// (the "(a seeded parameter, a sort-set)" case), never asked a
/// per-comparison question here — that is the SET channel's own job
/// (`narrow_set_kind_names`), run by `assume` right after this.
fn narrow(condition: &Expr, environment: &mut Environment, kernel: &Arc<RefinedTSKernel>, truth: bool) {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            narrow(&unary.operand, environment, kernel, !truth);
        }
        Expr::BoolOp(bool_op) => narrow_bool_op(bool_op, environment, kernel, truth),
        Expr::Compare(compare) => narrow_compare(compare, environment, truth),
        Expr::Call(call) => narrow_isinstance_call(call, environment, truth),
        // Calls other than isinstance, attributes, walrus, `in`, string
        // comparisons, and everything else this wave does not read: no
        // narrowing, the honest default.
        _ => {}
    }
}

/// `and`/`or` narrowing (mission point 3): `and` under truth, and `or`
/// under falsity (De Morgan — `not (a or b)` == `not a and not b`),
/// both apply every operand in conjunction. `and` under falsity and
/// `or` under truth narrow nothing this wave — either arm alone could
/// have made the whole true/false, so no single operand's negation (or
/// affirmation) is forced, and this wave builds no set-union machinery
/// to hold the "at least one of" case.
fn narrow_bool_op(
    bool_op: &ruff_python_ast::ExprBoolOp,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
    truth: bool,
) {
    let conjunction = match (bool_op.op, truth) {
        (BoolOp::And, true) => true,
        (BoolOp::Or, false) => true,
        _ => false,
    };
    if !conjunction {
        return;
    }
    // `or` under falsity narrows each operand by its own negation
    // (De Morgan); `and` under truth narrows each by its own truth.
    let per_operand_truth = bool_op.op == BoolOp::And;
    for value in &bool_op.values {
        narrow(value, environment, kernel, per_operand_truth);
    }
}

/// `ExprCompare` narrowing: chained comparisons lower to the
/// conjunction of adjacent pairs (see the module doc's CPython
/// citation). Under falsity, a conjunction narrows nothing (the same
/// rule `and`-under-falsity follows in `narrow_bool_op`) — the chain's
/// negation is a disjunction over which pair failed, and this wave
/// holds no union.
fn narrow_compare(compare: &ruff_python_ast::ExprCompare, environment: &mut Environment, truth: bool) {
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
/// (mission point 1), mirrored so the literal may sit on either side.
/// Anything else — a call, an attribute, a string, two changing names —
/// narrows nothing.
fn narrow_one_comparison(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    if matches!(op, CmpOp::Is | CmpOp::IsNot) {
        narrow_is_none(left, op, right, environment, truth);
        return;
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return;
    };
    if let (Some(name), Some(literal)) = (name_of(left), literal_number(right)) {
        narrow_name_against_literal(name, numeric_op, literal, environment, truth);
        return;
    }
    if let (Some(literal), Some(name)) = (literal_number(left), name_of(right)) {
        narrow_name_against_literal(name, mirror_cmp_op(numeric_op), literal, environment, truth);
        return;
    }
}

/// The subset of `CmpOp` this wave's numeric side-bounds filter reads:
/// `< <= > >= == !=`. `is`/`is not`/`in`/`not in` are handled
/// elsewhere or not at all.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NumericCmpOp {
    Lt,
    LtE,
    Gt,
    GtE,
    Eq,
    NotEq,
}

fn numeric_cmp_op(op: CmpOp) -> Option<NumericCmpOp> {
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
fn mirror_cmp_op(op: NumericCmpOp) -> NumericCmpOp {
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
fn satisfies(value: f64, op: NumericCmpOp, literal: f64) -> bool {
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
fn narrow_name_against_literal(
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
fn is_numeric_or_boolean(kind_tag: PrimitiveKind) -> bool {
    matches!(
        kind_tag,
        PrimitiveKind::Number | PrimitiveKind::Integer | PrimitiveKind::Float | PrimitiveKind::Boolean
    )
}

/// `is None` / `is not None` (mission point 5): only a Values-kind
/// binding is touched — the empty Values state means "provably not
/// None among the tracked exact values," which is sound because None
/// itself is never a member of a Values state (Values carries only
/// host-sorted numbers/booleans/strings/arrays, never the absent
/// marker). A non-Values binding (including one already `Kind::Null`)
/// passes through unchanged, per the mission's instruction that
/// non-Values states pass through everywhere this wave.
fn narrow_is_none(left: &Expr, op: CmpOp, right: &Expr, environment: &mut Environment, truth: bool) {
    let is_not = op == CmpOp::IsNot;
    let name = if is_none_literal(right) {
        name_of(left)
    } else if is_none_literal(left) {
        name_of(right)
    } else {
        None
    };
    let Some(name) = name else {
        return;
    };
    let Some(current) = environment.read(name).cloned() else {
        return;
    };
    if current.kind != Kind::Values {
        return;
    }
    let Some(kind_tag) = current.kind_tag else {
        return;
    };
    // `name is None` true, or `name is not None` false, both mean
    // "this Values binding holds None" — impossible for a Values state,
    // so every member is infeasible: bind the empty set.
    // `name is None` false, or `name is not None` true, both mean
    // "not None" — a Values binding already satisfies that for every
    // member, so it is left as is (still narrows nothing further, which
    // is sound: no member is dropped).
    let means_is_none = truth != is_not;
    if means_is_none {
        let grade = trust_level_of(&current);
        environment.bind(name, known_values(Vec::new(), kind_tag, grade));
    }
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
/// not a Values state to filter. `isinstance(value, int)`/`float`
/// PROVING true (`truth` and no prior binding) is itself the first
/// fact this environment learns about `value` — it seeds a fresh
/// `Kind::Set` binding holding the unbounded sort (the same set
/// `summaries.rs::return_sort_fallback`/`expressions.rs`'s `int(...)`
/// row build for a proved-but-unbounded `int`/`float`), grade
/// `TrustSpec` (the isinstance test is read, not executed — the same
/// grade `seed_parameters`'s own annotation-read seeding uses).
/// `isinstance(value, bool)` seeds `Kind::Values` instead: `bool`'s
/// domain is the two exact values `{0, 1}` (`string_models.rs`'s
/// `boolean_value` convention), not an unbounded ray, so it is not a
/// Set-kind sort seed. Proving FALSE, or a name already bound to
/// SOMETHING (however unreadable), never seeds here — a falsified test
/// says nothing positive about which sort `value` DOES hold, and an
/// existing binding is this function's other, unchanged, arm below.
fn narrow_isinstance_call(call: &ruff_python_ast::ExprCall, environment: &mut Environment, truth: bool) {
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
    let Some(tags) = isinstance_type_tags(&call.arguments.args[1]) else {
        return;
    };
    let current = environment.read(name).cloned();
    if current.is_none() {
        if truth {
            if let [tag] = tags.as_slice() {
                if let Some(seeded) = sort_seed(*tag) {
                    environment.bind(name, seeded);
                }
            }
        }
        return;
    }
    let current = current.expect("checked Some above");
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

/// The fresh binding a PROVED `isinstance(x, tag)` seeds for a name the
/// environment held nothing about at all (see `narrow_isinstance_call`'s
/// doc): the unbounded `Kind::Set` ray for `int`/`float`, or `None` for
/// `bool` (and `Number`, which no `isinstance` argument ever names —
/// `isinstance_type_tags` only ever answers Integer/Float/Boolean) —
/// `bool`'s own two-value seed is built directly at the call site
/// instead, since it is `Kind::Values`, a different constructor
/// entirely.
fn sort_seed(tag: PrimitiveKind) -> Option<AbstractValue> {
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
        PrimitiveKind::Number | PrimitiveKind::String | PrimitiveKind::Array => None,
    }
}

/// The unbounded whole-number ray (every integer, no floor or ceiling)
/// — the same shape `summaries.rs`'s private `whole_integers` builds;
/// copied here rather than shared cross-file, matching this file's own
/// documented precedent (`literal_number`'s doc) for a small leaf
/// reader every narrowing-adjacent file keeps its own copy of.
fn unbounded_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

/// The `PrimitiveKind`s an `isinstance` second argument names, for
/// exactly the shapes mission point 6 covers: a bare type name
/// (`int`/`float`/`bool`), or a `|`-chain of them
/// (`isinstance(x, int | float)`). Any other shape (a tuple form, a
/// non-primitive type) answers `None` — not read this wave.
fn isinstance_type_tags(expression: &Expr) -> Option<Vec<PrimitiveKind>> {
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

fn primitive_kind_of_type_name(name: &str) -> Option<PrimitiveKind> {
    match name {
        "int" => Some(PrimitiveKind::Integer),
        "float" => Some(PrimitiveKind::Float),
        "bool" => Some(PrimitiveKind::Boolean),
        _ => None,
    }
}

/// Whether `expression` is the bare name of a tracked place — the only
/// shape every narrowing leaf here reads on the tested side (mission's
/// "filter what is known, never invent," matching the Go reference's
/// `onPlace` restriction to a single identifier).
fn name_of(expression: &Expr) -> Option<&str> {
    match expression {
        Expr::Name(name) => Some(name.id.as_str()),
        _ => None,
    }
}

fn is_none_literal(expression: &Expr) -> bool {
    matches!(expression, Expr::NoneLiteral(_))
}

/// A bare (possibly unary +/- wrapped) NumberLiteral's exact value —
/// int or float — or `None` for anything else (complex, an int too
/// large for i64, a non-literal expression). Mirrors the sibling
/// private helpers of the same shape in loops.rs/expressions.rs/
/// surface.rs — each narrowing-adjacent file keeps its own copy rather
/// than sharing a cross-file dependency for one small leaf reader.
fn literal_number(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(int) => int.as_i64().map(|value| value as f64),
            Number::Float(value) => Some(*value),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if matches!(unary.op, UnaryOp::USub | UnaryOp::UAdd) => {
            let operand = literal_number(unary.operand.as_ref())?;
            Some(if unary.op == UnaryOp::USub { -operand } else { operand })
        }
        _ => None,
    }
}

// ── the SET channel: condition → NarrowTree ─────────────────────────

/// A `NarrowTree` leaf that claims nothing — the kernel's own "no
/// reading" leaf (`gate_narrow`/`narrow_wire`'s `Other` arm never reads
/// its other fields), matching `refined-ts-go/internal/refinedts/
/// narrowing/type_guard_recognizers.go`'s package-level `Other` value.
fn other_tree() -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Other,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: None,
        b: None,
    }
}

/// One `Cmp` leaf (`name op literal`) — the bare-fields constructor
/// every other `NarrowTree` variant this file never builds also needs,
/// since the struct derives no `Default` (`refined_kernel::
/// narrow_questions` — every field explicit at every call site, the
/// same discipline that module's own tests follow).
fn cmp_tree(op: NarrowCmpOp, k: f64) -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Cmp,
        op: Some(op),
        k,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: None,
        b: None,
    }
}

fn not_tree(a: NarrowTree) -> NarrowTree {
    NarrowTree {
        kind: NarrowTreeKind::Not,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: Some(Box::new(a)),
        b: None,
    }
}

fn and_or_tree(kind: NarrowTreeKind, a: NarrowTree, b: NarrowTree) -> NarrowTree {
    NarrowTree {
        kind,
        op: None,
        k: 0.0,
        lo: 0.0,
        hi: 0.0,
        d: 0.0,
        points: Vec::new(),
        set: RefinedSet::default(),
        a: Some(Box::new(a)),
        b: Some(Box::new(b)),
    }
}

/// `NumericCmpOp` → the kernel's own `NarrowCmpOp` — `Eq`/`NotEq` are
/// NOT `Cmp` leaves (the kernel's `Cmp` carries only the four ORDER
/// operators; equality is its own `NarrowTreeKind::Eq`/its negation),
/// so this reads only the four this file's `Cmp` leaf can name; `None`
/// for `Eq`/`NotEq`, read directly by `condition_tree_of` instead.
fn narrow_cmp_op_of(op: NumericCmpOp) -> Option<NarrowCmpOp> {
    match op {
        NumericCmpOp::Lt => Some(NarrowCmpOp::Lt),
        NumericCmpOp::LtE => Some(NarrowCmpOp::Le),
        NumericCmpOp::Gt => Some(NarrowCmpOp::Gt),
        NumericCmpOp::GtE => Some(NarrowCmpOp::Ge),
        NumericCmpOp::Eq | NumericCmpOp::NotEq => None,
    }
}

/// Lowers `condition`, RELATIVE TO `place` (the one name the kernel's
/// narrowing question is always scoped to — `refined-ts-go`'s own
/// `TreeOf`/`leafTreeOf` convention), into the kernel's `NarrowTree`
/// grammar: `!`/`and`/`or` fold the same shape/structure the condition
/// itself has (a `not` node wraps its operand's own tree UNCHANGED —
/// the KERNEL's own `narrowQ` is what swaps a `not`'s when-true/
/// when-false pair, `set_functions/narrow.lean`'s `.not t => (p.2,
/// p.1)`, so this builder never needs to track which polarity it is
/// "under" the way the VALUES channel's `narrow`/`narrow_bool_op` does
/// — the caller (`narrow_set_kind_names`) reads whichever side of the
/// ONE resulting `NarrowAnswer` its own branch truth names).
///
/// Any leaf NOT on `place`, or NOT one of this file's recognized
/// shapes (a call other than `isinstance`, a string test, two changing
/// names, `is`/`is not None` — Python's `None` is never a member of a
/// `Kind::Set`'s own domain, so an absence test states nothing a set
/// claim could narrow), lowers to `other_tree()` — the honest "no
/// claim" leaf, never a guess. The WHOLE tree is `None` only when
/// `condition` itself has no shape this function reads at all (an
/// unreachable case today — every `Expr` arm below returns `Some`,
/// down to the catch-all `other_tree()` — kept `Option` so a future
/// leaf that must genuinely decline has the same "no tree at all" exit
/// `narrow_set_kind_names`'s `let Some(tree) = … else { continue }`
/// already expects).
fn condition_tree_of(condition: &Expr, place: &str) -> Option<NarrowTree> {
    match condition {
        Expr::UnaryOp(unary) if unary.op == UnaryOp::Not => {
            condition_tree_of(&unary.operand, place).map(not_tree)
        }
        Expr::BoolOp(bool_op) => {
            let kind = match bool_op.op {
                BoolOp::And => NarrowTreeKind::And,
                BoolOp::Or => NarrowTreeKind::Or,
            };
            let mut trees = bool_op.values.iter().map(|value| condition_tree_of(value, place));
            let mut folded = trees.next()??;
            for next in trees {
                folded = and_or_tree(kind, folded, next?);
            }
            Some(folded)
        }
        Expr::Compare(compare) => Some(compare_tree_of(compare, place)),
        Expr::Call(call) => Some(isinstance_leaf_tree_of(call, place)),
        _ => Some(other_tree()),
    }
}

/// `ExprCompare` → a `NarrowTree`: a chained comparison folds to the
/// `And` of its adjacent pairs (same CPython citation the VALUES
/// channel's `narrow_compare` follows) — this reading does not depend
/// on `truth` the way the VALUES channel's falsity short-circuit does,
/// since the kernel's own answer already carries BOTH the `whenTrue`
/// (the chain held) and `whenFalse` (the chain's negation — a
/// disjunction over which pair failed, which the kernel proves
/// directly rather than this file approximating it as "narrows
/// nothing").
fn compare_tree_of(compare: &ruff_python_ast::ExprCompare, place: &str) -> NarrowTree {
    let mut left = compare.left.as_ref();
    let mut folded: Option<NarrowTree> = None;
    for (op, right) in compare.ops.iter().zip(compare.comparators.iter()) {
        let leaf = comparison_leaf_tree_of(left, *op, right, place);
        folded = Some(match folded {
            Some(existing) => and_or_tree(NarrowTreeKind::And, existing, leaf),
            None => leaf,
        });
        left = right;
    }
    folded.unwrap_or_else(other_tree)
}

/// One comparison pair (`left op right`) → a `NarrowTree` leaf, scoped
/// to `place`: a numeric literal on the other side lowers to `Cmp`
/// (`<`/`<=`/`>`/`>=`) or `Eq`/`not Eq` (`==`/`!=`); anything else
/// (`is`/`is not` — read separately by the VALUES channel only, since
/// `None` is never a `Kind::Set` member; two changing names; a string
/// literal) is `other_tree()`.
fn comparison_leaf_tree_of(left: &Expr, op: CmpOp, right: &Expr, place: &str) -> NarrowTree {
    // A STRING-literal equality (`layout == "horizontal"`) lowers to the
    // kernel's own EqSeq leaf — the word's code points ride `points`
    // (set_functions/narrow.lean's `.eqSeq`), so a string-tuple-union
    // set (a `Literal["…", …]` alias) narrows to the named member on
    // the when-true side and its complement on the when-false side.
    // `!=` is the same leaf under Not. Ordering (`<` etc.) over strings
    // has no kernel leaf and stays `other_tree()` below.
    if matches!(op, CmpOp::Eq | CmpOp::NotEq) {
        let word = if name_of(left) == Some(place) {
            string_literal_points(right)
        } else if name_of(right) == Some(place) {
            string_literal_points(left)
        } else {
            None
        };
        if let Some(points) = word {
            let leaf = NarrowTree { kind: NarrowTreeKind::EqSeq, points, ..other_tree() };
            return if op == CmpOp::Eq { leaf } else { not_tree(leaf) };
        }
    }
    let Some(numeric_op) = numeric_cmp_op(op) else {
        return other_tree();
    };
    let (on_place, literal) = if name_of(left) == Some(place) {
        (true, literal_number(right))
    } else if name_of(right) == Some(place) {
        (false, literal_number(left))
    } else {
        return other_tree();
    };
    let Some(literal) = literal else {
        return other_tree();
    };
    let effective_op = if on_place { numeric_op } else { mirror_cmp_op(numeric_op) };
    match effective_op {
        NumericCmpOp::Eq => NarrowTree { kind: NarrowTreeKind::Eq, k: literal, ..other_tree() },
        NumericCmpOp::NotEq => not_tree(NarrowTree { kind: NarrowTreeKind::Eq, k: literal, ..other_tree() }),
        _ => {
            let kernel_op = narrow_cmp_op_of(effective_op).expect("Eq/NotEq handled above");
            cmp_tree(kernel_op, literal)
        }
    }
}

/// A plain string literal's code points, one f64 per point — the word
/// an `EqSeq` leaf carries. Any other expression (an f-string, a
/// concatenation, a name) is `None`; only the literal's own spelling
/// is a proved word.
fn string_literal_points(expr: &Expr) -> Option<Vec<f64>> {
    let Expr::StringLiteral(literal) = expr else {
        return None;
    };
    Some(literal.value.to_str().chars().map(|c| c as u32 as f64).collect())
}

/// `isinstance(place, ...)` as a `NarrowTree` leaf: the kernel's own
/// grammar has no "Python sort" leaf (`IsInt` tests integrality within
/// ℝ̄, not "is this Python `int`") — a sort claim is entirely this
/// file's own `narrow_isinstance_call`/`sort_seed` job, on the VALUES
/// channel or at seeding time, never the kernel's. `other_tree()`
/// always: an `isinstance` test says nothing further the kernel's
/// COMPARISON/MEMBERSHIP vocabulary can express about a Set already
/// scoped to that sort.
fn isinstance_leaf_tree_of(_call: &ruff_python_ast::ExprCall, _place: &str) -> NarrowTree {
    other_tree()
}

/// Whether `tree` states anything at all — an all-`Other` tree asks
/// the kernel a question with no answer worth having (`refined-ts-go`'s
/// own `SaysAnything`).
fn says_anything(tree: &NarrowTree) -> bool {
    match tree.kind {
        NarrowTreeKind::Other => false,
        NarrowTreeKind::Not => says_anything(tree.a.as_deref().expect("not carries A")),
        NarrowTreeKind::And | NarrowTreeKind::Or => {
            says_anything(tree.a.as_deref().expect("and/or carries A"))
                || says_anything(tree.b.as_deref().expect("and/or carries B"))
        }
        _ => true,
    }
}

/// Meets a kernel narrowing claim into `current`'s own set: the
/// INTERSECTION of `current.set`'s forms with `claim_set`'s forms
/// (`RefinedSet`'s forms conjoin — the same reading
/// `refined_domain::lattice_operations::meet_known`'s own Set×Set
/// branch takes), keeping `current`'s `kind_tag` (the kernel's claim
/// carries no Python sort tag of its own) and `current`'s own trust
/// grade (never claimed stronger by a narrowing than the value that
/// flowed in — `loops.rs`'s `kernel_bounded_counter_environment` binds
/// its own kernel answer at the SAME grade the entry binding carried,
/// the matching precedent).
fn meet_set_answer(current: &AbstractValue, claim_set: &RefinedSet) -> AbstractValue {
    let mut combined = current.set.forms.clone();
    combined.extend(claim_set.forms.clone());
    let grade = trust_level_of(current);
    AbstractValue {
        kind_tag: current.kind_tag,
        ..known_set(make_refined_set(combined), None, grade, current.set_kind_tag)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_domain::abstract_value::known_values;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::kernel_bridge::dylib_path;
    use refined_kernel::kernel_bridge::kernel_artifacts_present;
    use refined_kernel::kernel_bridge::load_kernel;
    use ruff_python_parser::parse_expression;

    use super::*;

    /// A kernel handle for tests that never ask it anything — `assume`
    /// takes the parameter for the frozen signature's sake, but no
    /// construct this wave asks a question of it. `None` when the
    /// native dylib artifact has not been built, so this file's tests
    /// run without requiring `pnpm kernel:native` first.
    fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
        let path = dylib_path();
        if !kernel_artifacts_present(&path) {
            eprintln!("native kernel dylib absent — build it first");
            return None;
        }
        Some(load_kernel(&path).expect("load_kernel"))
    }

    fn environment_with(name: &str, values: Vec<f64>, kind_tag: PrimitiveKind) -> Environment {
        let mut locally_bound = HashSet::new();
        locally_bound.insert(name.to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind(name, known_values(values, kind_tag, TrustProved));
        environment
    }

    fn assumed(source: &str, environment: Environment, truth: bool) -> Option<Environment> {
        let kernel = loaded_kernel()?;
        let parsed = parse_expression(source).expect("test source must parse");
        let expression = parsed.into_expr();
        Some(assume(&expression, environment, &kernel, truth))
    }

    #[test]
    fn test_greater_than_literal_keeps_satisfying_drops_others() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![200.0]);
    }

    #[test]
    fn test_greater_than_literal_falsity_flips_the_kept_side() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x > 100", environment, false) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_chained_comparison_keeps_the_middle_window() {
        let environment = environment_with("x", vec![-5.0, 0.0, 60.0, 120.0, 200.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![0.0, 60.0, 120.0]);
    }

    #[test]
    fn test_equality_against_literal_keeps_only_that_value() {
        let environment = environment_with("x", vec![40.0, 41.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("x == 40", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_not_wrapped_comparison_flips_truth() {
        let environment = environment_with("x", vec![200.0, 40.0], PrimitiveKind::Number);
        let Some(narrowed) = assumed("not (x > 100)", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.values, vec![40.0]);
    }

    #[test]
    fn test_and_narrows_both_names() {
        let mut locally_bound = HashSet::new();
        locally_bound.insert("a".to_owned());
        locally_bound.insert("b".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("a", known_values(vec![-1.0, 5.0], PrimitiveKind::Number, TrustProved));
        environment.bind("b", known_values(vec![-2.0, 7.0], PrimitiveKind::Number, TrustProved));
        let Some(narrowed) = assumed("a > 0 and b > 0", environment, true) else {
            return;
        };
        let a = narrowed.read("a").expect("a still bound");
        let b = narrowed.read("b").expect("b still bound");
        assert_eq!(a.values, vec![5.0]);
        assert_eq!(b.values, vec![7.0]);
    }

    #[test]
    fn test_non_values_binding_untouched() {
        use refined_domain::abstract_value::null_value;
        let mut locally_bound = HashSet::new();
        locally_bound.insert("x".to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind("x", null_value());
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Null);
    }

    #[test]
    fn test_unbound_name_untouched() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("x > 100", environment, true) else {
            return;
        };
        assert!(narrowed.read("x").is_none());
    }

    // ── the SET channel ──────────────────────────────────────────────

    fn environment_with_set(name: &str, set: RefinedSet, kind_tag: PrimitiveKind) -> Environment {
        let mut locally_bound = HashSet::new();
        locally_bound.insert(name.to_owned());
        let mut environment = Environment::new(locally_bound);
        environment.bind(name, AbstractValue { kind_tag: Some(kind_tag), ..known_set(set, None, TrustProved, SetKindTag::None) });
        environment
    }

    /// `>` on a Set-kind binding intersects the kernel's claim into the
    /// current set — `x > 0` on the unbounded integer ray narrows to
    /// the open-above-zero integer ray, which the assignability law's
    /// own containment ask would then judge against a declared window.
    #[test]
    fn test_set_kind_greater_than_literal_intersects_the_kernel_claim() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x > 0", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.kind, Kind::Set);
        assert_eq!(x.kind_tag, Some(PrimitiveKind::Integer));
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), refined_sets::refinement_forms::above(0.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            expected
        );
    }

    /// `>=` mirrors the same leaf with the inclusive operator.
    #[test]
    fn test_set_kind_greater_than_or_equal_intersects() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x >= 0", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), at_least(0.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            expected
        );
    }

    /// `0 <= x <= 120` — b-body-expressions.py's `len_in_guard` shape
    /// (~b:649) once `x` already carries the unbounded integer ray:
    /// the chained comparison's `And` tree intersects BOTH bounds in
    /// one kernel ask, landing the exact `[0, 120]` integer window
    /// `Age` (the fixture's own declared alias) admits.
    #[test]
    fn test_set_kind_chained_comparison_intersects_both_bounds() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("0 <= x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &age) && (kernel.scalar_subset)(&age, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            age
        );
    }

    /// `x >= 0` alone (no upper bound) — b-body-expressions.py's
    /// `guard_over_ceiling` shape (~b:656): the narrowed set is `[0,
    /// ∞) ∩ ℤ`, which is NOT a subset of `Age`'s `[0, 120]` window (it
    /// still admits 200) — this is the fixture's own marked fire,
    /// proved here at the SET level (the assignability law's own
    /// containment ask is what actually fires it at the sink; this
    /// test pins that the narrowed set is exactly what admits the
    /// ceiling violation, not something already tighter).
    #[test]
    fn test_set_kind_single_sided_bound_still_admits_the_ceiling() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x >= 0", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            !(kernel.scalar_subset)(&x.set, &age),
            "x.set = {:?} must still admit values above 120 (200, …)",
            x.set
        );
    }

    /// `and` composes two Set-kind leaves on the SAME name into one
    /// tree, same as the chained-comparison test above but spelled as
    /// an explicit `and`.
    #[test]
    fn test_set_kind_and_composes_both_leaves() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x >= 0 and x <= 120", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &age) && (kernel.scalar_subset)(&age, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            age
        );
    }

    /// `not (x > 120)` on a Set-kind binding — De Morgan through `not`
    /// folds to the kernel's own `whenFalse` claim for `x > 120`
    /// (`¬(x > 120)` is `x <= 120`), landing the at-most-120 half of
    /// the integer ray.
    #[test]
    fn test_set_kind_not_wrapped_comparison_uses_the_kernel_negation() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("not (x > 120)", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        let Some(kernel) = loaded_kernel() else { return };
        let expected = make_refined_set(vec![integer(), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&x.set, &expected) && (kernel.scalar_subset)(&expected, &x.set),
            "x.set = {:?}, want the same set as {:?}",
            x.set,
            expected
        );
    }

    /// A leaf this file cannot read at all (`x in "abc"` — membership
    /// against a receiver that is not this file's `isinstance`/
    /// comparison vocabulary) lowers to `other_tree()`; the whole tree
    /// says nothing (`says_anything` false), so the binding is left
    /// exactly as it was — never narrowed, never refused.
    #[test]
    fn test_an_unreadable_leaf_shape_leaves_the_set_binding_untouched() {
        let environment = environment_with_set("x", unbounded_integers(), PrimitiveKind::Integer);
        let Some(narrowed) = assumed("x in y", environment, true) else {
            return;
        };
        let x = narrowed.read("x").expect("x still bound");
        assert_eq!(x.set, unbounded_integers());
    }

    /// `isinstance(value, int)` on a name the environment has bound
    /// NOTHING for (the `object`-typed parameter shape,
    /// b-body-expressions.py's `len_in_guard`/`guard_over_ceiling`)
    /// seeds a fresh `Kind::Set` holding the unbounded integer ray —
    /// the "(a seeded parameter, a sort-set)" case `assume`'s own
    /// module doc names.
    #[test]
    fn test_isinstance_int_seeds_a_fresh_integer_set_from_unbound() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("isinstance(value, int)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("isinstance seeded value");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        assert_eq!(value.set, unbounded_integers());
    }

    /// `isinstance(value, bool)` seeds `Kind::Values` over the two
    /// exact booleans, never a `Kind::Set` — `bool`'s domain is exactly
    /// `{0, 1}`.
    #[test]
    fn test_isinstance_bool_seeds_the_two_boolean_values_from_unbound() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("isinstance(value, bool)", environment, true) else {
            return;
        };
        let value = narrowed.read("value").expect("isinstance seeded value");
        assert_eq!(value.kind, Kind::Values);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Boolean));
        let mut values = value.values.clone();
        values.sort_by(f64::total_cmp);
        assert_eq!(values, vec![0.0, 1.0]);
    }

    /// `isinstance(value, int)` proving FALSE seeds nothing — a
    /// falsified test says which sort `value` is NOT, never which sort
    /// it IS.
    #[test]
    fn test_isinstance_proving_false_seeds_nothing() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed("isinstance(value, int)", environment, false) else {
            return;
        };
        assert!(narrowed.read("value").is_none());
    }

    /// The full `len_in_guard` guard
    /// (`isinstance(value, int) and not isinstance(value, bool) and
    /// 0 <= value <= 120`) run as ONE `assume` call: the isinstance
    /// seed and the chained-comparison narrowing compose end to end,
    /// landing the exact `[0, 120]` integer window `Age` admits.
    #[test]
    fn test_len_in_guard_shape_narrows_to_the_zero_to_120_integer_window() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed(
            "isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 120",
            environment,
            true,
        ) else {
            return;
        };
        let value = narrowed.read("value").expect("value bound by the guard");
        assert_eq!(value.kind, Kind::Set);
        assert_eq!(value.kind_tag, Some(PrimitiveKind::Integer));
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            (kernel.scalar_subset)(&value.set, &age) && (kernel.scalar_subset)(&age, &value.set),
            "value.set = {:?}, want the same set as {:?}",
            value.set,
            age
        );
    }

    /// The `guard_over_ceiling` shape — same guard, but only the
    /// single-sided `value >= 0` bound: the narrowed set still admits
    /// 200 (not a subset of `Age`), matching the fixture's own marked
    /// "the guard does not bound the ceiling" fire.
    #[test]
    fn test_guard_over_ceiling_shape_still_admits_the_ceiling_violation() {
        let environment = Environment::new(HashSet::new());
        let Some(narrowed) = assumed(
            "isinstance(value, int) and not isinstance(value, bool) and value >= 0",
            environment,
            true,
        ) else {
            return;
        };
        let value = narrowed.read("value").expect("value bound by the guard");
        let Some(kernel) = loaded_kernel() else { return };
        let age = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]);
        assert!(
            !(kernel.scalar_subset)(&value.set, &age),
            "value.set = {:?} must still admit values above 120 (200, …)",
            value.set
        );
    }
}
