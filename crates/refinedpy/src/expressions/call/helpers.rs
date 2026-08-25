//! Small leaf readers `evaluate_call` and its siblings share: the
//! bytes-literal reader, the generator-shape test, `range(...)`'s own
//! materialized-value builder, positional-argument splicing, the
//! unbounded-whole-integers set, and the base-ten integer-string test.

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;

use crate::bytes_models;
use crate::collection_models;
use crate::env::Environment;

use super::super::arithmetic::single_numeric_value;
use super::super::evaluate_expression;

/// `b"..."`/`bytes([...])` literal text — bytes_models.rs's own
/// `Kind::List` shape, one Integer-tagged code-unit per byte
/// (stdtypes.rst, "Bytes and Bytearray Objects": bytes objects are
/// sequences of integers 0-255). `literal.value.bytes()` reads the
/// literal's own raw byte sequence off `ruff_python_ast`'s
/// `BytesLiteralValue` the same way `ExprStringLiteral`'s `to_str()` is
/// already read above.
pub(in super::super) fn evaluate_bytes_literal(literal: &ruff_python_ast::ExprBytesLiteral) -> AbstractValue {
    let bytes: Vec<u8> = literal.value.bytes().collect();
    bytes_models::bytes_literal_value(&bytes)
}

/// Whether `def`'s body is generator-SHAPED: at least one top-level
/// `yield` statement (`Stmt::Expr` wrapping `Expr::Yield`) anywhere in
/// the body, OR a `yield` one level inside a `for`/`async for` loop body
/// (ruff collapses both into one `Stmt::For` node — see that struct's
/// own generated.rs doc, "collapses the synchronous and asynchronous
/// variants into a single type" — so no separate `AsyncFor` arm is
/// needed) — a-statements.py's own `stream()` shape: `for value in (10,
/// 20, 30): yield value`, a loop-bodied generator with no top-level
/// `yield` at all. Recursion stops at one level (a `yield` nested inside
/// a further `if`/`for`/`try` INSIDE that loop body is not walked) —
/// this is a ROUTING check only, matching the same syntactic fact that
/// makes CPython itself compile a function as a generator (datamodel.rst,
/// "Generator functions": "a function... that uses the `yield`
/// statement... is called a generator function"). `instances::
/// generator_yields` still owns deciding whether the body's EXACT shape
/// is one it can interpret (straight-line yields/an early return/a
/// single literal-iterable `for` loop, `Kind::Unknown` on any richer
/// control flow) — a body this function calls generator-shaped but
/// `generator_yields` cannot read answers `unknown()` at the call site,
/// the same decline every other unmodeled body shape in this file
/// already gives.
pub(in super::super) fn is_generator_def(def: &ruff_python_ast::StmtFunctionDef) -> bool {
    def.body.iter().any(|stmt| match stmt {
        ruff_python_ast::Stmt::Expr(expr_stmt) => matches!(expr_stmt.value.as_ref(), Expr::Yield(_)),
        ruff_python_ast::Stmt::For(for_stmt) => for_stmt.body.iter().any(|inner| {
            matches!(inner, ruff_python_ast::Stmt::Expr(expr_stmt) if matches!(expr_stmt.value.as_ref(), Expr::Yield(_)))
        }),
        _ => false,
    })
}

/// `range(...)` read as an EXPRESSION VALUE — library/stdtypes.rst,
/// `class:: range(stop)` / `range(start, stop[, step])`: "The advantage
/// of the range type over a regular list or tuple is that a range
/// object will always take the same (small) amount of memory... range
/// objects implement the `collections.abc.Sequence` ABC." This domain
/// materializes it as a `Kind::List` of Integer-sorted elements — the
/// same eager-materialization choice `loops.rs`'s own `range_call_values`
/// already makes for a `for`-loop iterable, reused here for a
/// non-loop expression context (a comprehension iterable, a bare
/// value, etc.) — `range`'s own elements are always int
/// (stdtypes.rst's `range` entry: "the arguments must be integers").
/// Every argument must be a known single Integer-sorted value (an
/// evaluated expression, not a literal-syntax-only reading — this
/// differs from `loops.rs`'s own syntactic reader, which only serves
/// the `for`-loop path it owns); a non-Integer/unknown argument, a
/// zero step, or an argument count outside 1/2/3 declines to that same
/// materialized-list path — EXCEPT the one-argument `range(n)` form
/// with an Integer-sorted `n` this file cannot read as a single known
/// scalar, which answers a SORT-ONLY repetition window instead (see
/// this function's own body for the exact fallback and why only the
/// one-argument form gets it).
pub(in super::super) fn range_expression_value(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    // `range(n)` — the ONE-ARGUMENT form — with `n` an Integer-sorted
    // value this file cannot read as a single known scalar (an unbounded
    // `n: int` parameter, say): the exact element sequence is unstated,
    // but stdtypes.rst's own formula (`r[i] = start + step*i` from
    // `start = 0`, `step = 1`) still pins two SORT facts regardless of
    // `n`'s own sign — every element is Integer AND nonnegative (`n`
    // negative or zero just gives the empty range, never a negative
    // element), and the count is `max(n, 0)`, unstated but bounded below
    // by 0 — so `[0, +inf)` for both, the same bare-star repetition
    // window `list_of_unknown_string_characters` already answers for an
    // unbounded string. Only the one-argument form gets this fallback:
    // a two/three-argument `range(start, stop[, step])` with a
    // non-exact bound has no fixed `start`/`step` to pin the element
    // window to (a negative `start` or `step` makes even the
    // nonnegative-element claim false), so those forms keep declining
    // through `range_argument_value`'s own `?` below, unchanged.
    if let [stop] = arguments {
        if range_argument_value(stop).is_none() && stop.kind == Kind::Set && stop.kind_tag == Some(PrimitiveKind::Integer) {
            let grade = derived_trust_level(TrustSpec, arguments);
            let element = make_refined_set(vec![integer(), at_least(0.0)]);
            return Some(AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(refined_sets::repetition_window_forms::repetition(element, 0, None), None, grade, SetKindTag::None)
            });
        }
    }
    let (start, stop, step) = match arguments {
        [stop] => (0.0, range_argument_value(stop)?, 1.0),
        [start, stop] => (range_argument_value(start)?, range_argument_value(stop)?, 1.0),
        [start, stop, step] => (range_argument_value(start)?, range_argument_value(stop)?, range_argument_value(step)?),
        _ => return None,
    };
    if step == 0.0 {
        return None;
    }
    let mut values = Vec::new();
    let mut current = start;
    // r[i] = start + step*i, while r[i] < stop (step > 0) or r[i] > stop
    // (step < 0) — stdtypes.rst's own range formula
    if step > 0.0 {
        while current < stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    } else {
        while current > stop {
            values.push(known_values(vec![current], PrimitiveKind::Integer, TrustProved));
            current += step;
        }
    }
    Some(collection_models::list_literal_value(&values))
}

/// One `range(...)` argument's known Integer value, or `None` if it is
/// not a single known Integer-sorted value.
pub(in super::super) fn range_argument_value(value: &AbstractValue) -> Option<f64> {
    let (number, sort) = single_numeric_value(value)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    Some(number)
}

/// Every positional call argument's value, in order, with a `Starred`
/// argument's own known-List elements spliced in place (the same
/// splicing `evaluate_display_elements` performs for a list/tuple/set
/// display — expressions.rst states one "unpacking" rule for both call
/// arguments and displays). `None` the moment a starred argument
/// evaluates to anything but a known `Kind::List` — an UNBOUNDED
/// iterable (an untyped/unknown-length parameter, for instance) has no
/// proven element count to splice, so the whole call declines rather
/// than guess at how many positional slots it fills.
pub(in super::super) fn splice_call_arguments(
    args: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<AbstractValue>> {
    let mut values = Vec::new();
    for arg in args {
        if let Expr::Starred(starred) = arg {
            let spread = evaluate_expression(&starred.value, environment, kernel);
            if spread.kind != Kind::List {
                return None;
            }
            values.extend(spread.items);
            continue;
        }
        values.push(evaluate_expression(arg, environment, kernel));
    }
    Some(values)
}

/// The unbounded whole-number set `eval_literal_value`'s int-literal row
/// answers — the identical shape `summaries::whole_integers` builds
/// (`refinement_forms::integer()` conjoined with the unbounded ray),
/// repeated here rather than reaching into `summaries.rs` for one
/// two-line helper (this file is the one every other call-result row
/// already lives in, and `summaries.rs` has no dependency edge back
/// into `expressions.rs` for this single shape).
pub(in super::super) fn eval_whole_integers() -> RefinedSet {
    make_refined_set(vec![integer(), at_least(f64::NEG_INFINITY)])
}

pub(in super::super) fn is_valid_base_ten_int_string(text: &str) -> bool {
    let trimmed = text.trim();
    let digits_and_underscores = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if digits_and_underscores.is_empty() {
        return false;
    }
    let chars: Vec<char> = digits_and_underscores.chars().collect();
    if chars.first() == Some(&'_') || chars.last() == Some(&'_') {
        return false;
    }
    let mut previous_was_underscore = false;
    let mut saw_any_digit = false;
    for &c in &chars {
        if c == '_' {
            if previous_was_underscore {
                return false;
            }
            previous_was_underscore = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return false;
        }
        saw_any_digit = true;
        previous_was_underscore = false;
    }
    saw_any_digit
}
