use refined_domain::abstract_value::{nan_value, float_sorted_unknown, known_values, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_sets::refinement_forms::{at_least, at_most, make_refined_set, one_of, Form};

use super::*;

fn integer(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustSpec)
}

fn float(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Float, TrustSpec)
}

/// A kernel handle for tests that ask a `min`/`max`-over-a-set
/// question — the same skip `math_models.rs`'s own `loaded_kernel`
/// takes when the native dylib artifact has not been built, so
/// this file's tests run without requiring `pnpm kernel:native`
/// first. Every OTHER test in this module keeps calling
/// `builtin_call_result` directly (pure Rust, no kernel needed) —
/// its own signature never changed.
fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

#[test]
fn round_half_to_even_rounds_up_at_odd_tenths() {
    // round(201.5) == 202: 201.5 sits between 201 and 202; 202 is
    // the even choice.
    let got = builtin_call_result("round", &[float(201.5)]).expect("round(201.5) models");
    assert_eq!(got.values, vec![202.0]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn round_half_to_even_rounds_down_at_even_tenths() {
    // round(40.5) == 40: 40.5 sits between 40 and 41; 40 is the even
    // choice — the AGENT-BRIEF row-inverting fact against a naive
    // round-half-up reading.
    let got = builtin_call_result("round", &[float(40.5)]).expect("round(40.5) models");
    assert_eq!(got.values, vec![40.0]);
}

#[test]
fn round_two_argument_form_declines() {
    let got = builtin_call_result("round", &[float(40.5), integer(1.0)]);
    assert!(got.is_none(), "round(x, n) should decline: {got:?}");
}

#[test]
fn abs_of_negative_integer_is_positive_integer() {
    let got = builtin_call_result("abs", &[integer(-200.0)]).expect("abs(-200) models");
    assert_eq!(got.values, vec![200.0]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

/// `abs()` over a Set-seeded operand asks the kernel's `Abs` transfer
/// (`abs_call_over_set`'s own doc, `javascript-pins.md` arith.7): a
/// window straddling zero folds its lower bound to 0 — `abs([-2, 1])`
/// answers `[0, 2]`, `transferAbs`'s own `straddles` branch
/// (`theories/binary64/abs.lean`: `lo := if straddles then 0 else
/// min(abs(A.lo), abs(A.hi))`, `hi := max(abs(A.lo), abs(A.hi))` —
/// here `A.lo = -2, A.hi = 1`, both admitted, so `lo = 0` and
/// `hi = max(2, 1) = 2`). Asserts the exact enclosure, not merely the
/// shape, since the window is narrow enough to pin by hand.
#[test]
fn abs_over_a_set_operand_asks_the_kernel() {
    let Some(kernel) = loaded_kernel() else { return };
    let window = make_refined_set(vec![at_least(-2.0), at_most(1.0)]);
    let operand = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(window, None, TrustSpec, SetKindTag::None)
    };
    let got = builtin_call_result_with_kernel("abs", &[operand], &kernel)
        .expect("abs([-2, 1]) over a Set operand models through the kernel");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![at_least(0.0), at_most(2.0)]);
    assert_eq!(got.set, want, "abs([-2, 1]) should answer [0, 2]: got {:?}", got.set);
}

/// `float(x)` over an Integer-sorted Set operand (`rounding_call_over_set`'s
/// own image, `math.floor(x)` over a declared `[2.5, 3.5]` guard —
/// `float_call_over_set`'s own doc): re-tags the same set Float-sorted,
/// no kernel round trip and no value change — `float([2, 3])` answers
/// the identical `{2, 3}` window, only Float-sorted now.
#[test]
fn float_over_a_set_operand_re_sorts_without_a_kernel_round_trip() {
    let Some(kernel) = loaded_kernel() else { return };
    let window = make_refined_set(vec![at_least(2.0), at_most(3.0)]);
    let operand = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(window.clone(), None, TrustSpec, SetKindTag::None)
    };
    let got = builtin_call_result_with_kernel("float", &[operand], &kernel)
        .expect("float([2, 3]) over a Set operand models");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    assert_eq!(got.set, window, "float() must not change the operand's own set: got {:?}", got.set);
}

#[test]
fn int_truncates_toward_zero_on_positive_fraction() {
    let got = builtin_call_result("int", &[float(7.9)]).expect("int(7.9) models");
    assert_eq!(got.values, vec![7.0]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn int_truncates_toward_zero_on_negative_fraction() {
    // int(-7.9) == -7, not -8: truncation toward zero, not floor.
    let got = builtin_call_result("int", &[float(-7.9)]).expect("int(-7.9) models");
    assert_eq!(got.values, vec![-7.0]);
}

#[test]
fn int_of_a_base_ten_digit_string_parses_the_exact_value() {
    // int("75") == 75 — j-stdlib-surfaces.py's own int_parse row
    let string_argument = known_values(vec![55.0, 53.0], PrimitiveKind::String, TrustSpec);
    let got = builtin_call_result("int", &[string_argument]).expect("int(\"75\") models");
    assert_eq!(got.values, vec![75.0]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn int_of_a_non_numeric_string_declines() {
    // int("abc") raises ValueError at runtime — this row never
    // fabricates a value for it; the raise itself is
    // expressions.rs's call_provable_raise's own business
    let string_argument = string_value("abc");
    let got = builtin_call_result("int", &[string_argument]);
    assert!(got.is_none(), "int(\"abc\") should decline: {got:?}");
}

#[test]
fn int_of_a_negative_digit_string_parses_the_exact_negative_value() {
    let string_argument = string_value("-7");
    let got = builtin_call_result("int", &[string_argument]).expect("int(\"-7\") models");
    assert_eq!(got.values, vec![-7.0]);
}

/// `int()` over a Float-sorted Set operand asks the kernel's `Trunc`
/// transfer (`int_call_over_set`'s own doc) — the same shape
/// `int(math.sqrt(x))` builds over a declared parameter range
/// (`c-reads-and-values.py`'s `math_sqrt_over_declared_range`: `x`
/// is `[0, 100]`, `math.sqrt(x)` is `[0, 10]`, `int(...)` of that
/// stays `[0, 10]` — already integral, so truncation changes
/// nothing at either endpoint).
#[test]
fn int_over_a_set_operand_asks_the_kernel() {
    let Some(kernel) = loaded_kernel() else { return };
    let window = make_refined_set(vec![at_least(0.0), at_most(10.0)]);
    let operand = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..refined_domain::abstract_value::known_set(window, None, TrustSpec, SetKindTag::None)
    };
    let got = builtin_call_result_with_kernel("int", &[operand], &kernel)
        .expect("int([0, 10]) over a Float Set operand models through the kernel");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer), "int(...) is always Integer-sorted, regardless of its argument's own sort");
    // `Trunc`'s answer over an already-integral window carries its own
    // `Integer` form — `binary64.trunc` proves the whole result is a
    // whole number here, not just this row's own sort tag
    let want = make_refined_set(vec![at_least(0.0), at_most(10.0), refined_sets::refinement_forms::integer()]);
    assert!((kernel.scalar_subset)(&got.set, &want), "result {:?} not ⊆ want {:?}", got.set, want);
    assert!((kernel.scalar_subset)(&want, &got.set), "want {:?} not ⊆ result {:?}", want, got.set);
}

/// `int(x)` over a BARE, UNBOUNDED `float` parameter's own seed
/// (`float_sorted_unknown()`'s own `numbers()` set, `[NEG_INFINITY,
/// +inf)`) must still answer the unbounded Integer sort rather than
/// decline outright: `binary64.trunc`'s own enclosure over an
/// entirely-unbounded window is never provably finite
/// (`enclosure_is_provably_finite` false by construction — the empty-
/// forms/unbounded-ray cases it itself declines), so before this fix
/// `int_call_over_set` returned `None` here, leaving `n = int(x)`
/// Unknown downstream (D5's own `clamp_to_age` helpers' fact-export
/// blocker). The weaker TRUE claim — every non-raising outcome of
/// `int(x)` is SOME int — is `int_image`'s own image, pinned here
/// directly.
#[test]
fn int_over_an_unbounded_float_operand_answers_the_unbounded_integer_image() {
    let Some(kernel) = loaded_kernel() else { return };
    let operand = float_sorted_unknown();
    let got = builtin_call_result_with_kernel("int", &[operand], &kernel).expect("int(x) over an unbounded float must still decide the image");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(f64::NEG_INFINITY)]);
    assert_eq!(got.set, want, "the answer is int_image's own unbounded Integer ray, not a decline");
}

#[test]
fn float_of_inf_string_is_positive_infinity() {
    // functions.rst's float() grammar: "inf"/"Infinity" (case-
    // insensitive) spell positive infinity.
    let string_argument = string_value("inf");
    let got = builtin_call_result("float", &[string_argument]).expect("float(\"inf\") models");
    assert_eq!(got.values, vec![f64::INFINITY]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn float_of_negative_inf_string_is_negative_infinity() {
    let string_argument = string_value("-inf");
    let got = builtin_call_result("float", &[string_argument]).expect("float(\"-inf\") models");
    assert_eq!(got.values, vec![f64::NEG_INFINITY]);
}

#[test]
fn float_of_nan_string_is_the_nan_admitting_value() {
    // `float("nan")` answers the domain's own `Kind::NaN` state
    // (`nan_value()`), never a `Kind::Values` list carrying a bare
    // NaN — no refined set admits NaN as an element
    // (`refinement_forms::element`'s own construction-time
    // refusal), so `Kind::Values` must stay NaN-free too.
    let string_argument = string_value("nan");
    let got = builtin_call_result("float", &[string_argument]).expect("float(\"nan\") models");
    assert_eq!(got.kind, Kind::NaN, "float(\"nan\") should answer the domain's NaN state: {got:?}");
}

#[test]
fn float_of_a_decimal_digit_string_parses_the_exact_value() {
    let string_argument = string_value("1.5");
    let got = builtin_call_result("float", &[string_argument]).expect("float(\"1.5\") models");
    assert_eq!(got.values, vec![1.5]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn float_of_infinity_spelling_case_insensitive() {
    // "Case is not significant... 'INFINITY' and 'iNfINity' are all
    // acceptable spellings for positive infinity."
    let string_argument = string_value("Infinity");
    let got = builtin_call_result("float", &[string_argument]).expect("float(\"Infinity\") models");
    assert_eq!(got.values, vec![f64::INFINITY]);
}

#[test]
fn float_of_an_unparseable_string_keeps_the_sort_only_answer() {
    let string_argument = string_value("not a number");
    let got = builtin_call_result("float", &[string_argument]).expect("float(<any string>) models sort-only");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
}

/// A2.xfer.minmax's own `max_nan_second_inside` row: `max(0.5,
/// float("nan"))` — the SECOND argument is NaN, and every comparison
/// against it is False, so the FIRST argument (0.5) is kept exactly.
/// Before `min_max_call_with_nan_operand`, `single_known_numeric`
/// declined the NaN operand outright and the whole call answered
/// `None` (undetermined at the sink), never reading the position-
/// dependent value CPython actually produces.
#[test]
fn max_of_a_known_value_and_a_trailing_nan_keeps_the_first_argument() {
    let got = builtin_call_result("max", &[float(0.5), nan_value()]).expect("max(0.5, nan) models");
    assert_eq!(got.values, vec![0.5]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
}

/// The mirror: `max(float("nan"), 0.5)` — NaN sits FIRST this time,
/// and `0.5 > nan` is also False, so the first argument (nan itself)
/// is what stays. Position-dependent by construction: this row and
/// the one above pass the SAME two values in opposite order and
/// answer two different results, exactly matching CPython's own
/// left-to-right `max`/`min` walk over a NaN operand.
#[test]
fn max_of_a_leading_nan_and_a_known_value_keeps_the_nan() {
    let got = builtin_call_result("max", &[nan_value(), float(0.5)]).expect("max(nan, 0.5) models");
    assert_eq!(got.kind, Kind::NaN, "max(nan, 0.5) should stay the domain's NaN state: {got:?}");
}

/// `min` reads identically: NaN fails every comparison regardless of
/// which operator asked it, so `min` and `max` both keep the first
/// argument over a NaN operand. Pins that `min_max_call_with_nan_
/// operand` is genuinely operator-independent, not a `max`-only fix.
#[test]
fn min_of_a_known_value_and_a_trailing_nan_keeps_the_first_argument() {
    let got = builtin_call_result("min", &[float(0.5), nan_value()]).expect("min(0.5, nan) models");
    assert_eq!(got.values, vec![0.5]);
}

#[test]
fn min_over_known_numerics_picks_the_smallest() {
    let got = builtin_call_result("min", &[integer(3.0), integer(-1.0), integer(5.0)])
        .expect("min(...) models");
    assert_eq!(got.values, vec![-1.0]);
}

#[test]
fn max_over_known_numerics_picks_the_largest() {
    let got = builtin_call_result("max", &[integer(3.0), integer(-1.0), integer(5.0)])
        .expect("max(...) models");
    assert_eq!(got.values, vec![5.0]);
}

#[test]
fn max_threads_the_winning_arguments_own_sort() {
    // 4.5 (float) beats 3 (int): the winner's own Float sort carries
    // through, matching Python's min/max returning the argument
    // itself unchanged.
    let got = builtin_call_result("max", &[integer(3.0), float(4.5)]).expect("max(...) models");
    assert_eq!(got.values, vec![4.5]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
}

#[test]
fn min_single_scalar_argument_declines() {
    // min(3) is neither the two-or-more-scalar form nor the
    // single-iterable form — a bare scalar is not a Kind::List.
    let got = builtin_call_result("min", &[integer(3.0)]);
    assert!(got.is_none(), "min(x) with one scalar argument should decline: {got:?}");
}

/// A numeric-sorted `Kind::Set` operand in the two-or-more-argument
/// `max` form declines through `min_max_call` (`single_known_numeric`
/// refuses a Set) and reaches the kernel-asked arm
/// (`builtin_call_result_with_kernel`'s own doc). `max(ages, 0)` over
/// `ages` bounded 0..120 and the known scalar `0` asks
/// `binary64.max`, answering an enclosure whose own hull sits inside
/// 0..120 — this test only asserts the SHAPE (a Set-kind Integer
/// answer), not a specific enclosure, matching the kernel-invocation
/// exception this file's tests otherwise avoid.
#[test]
fn max_over_a_set_operand_asks_the_kernel() {
    let Some(kernel) = loaded_kernel() else { return };
    let ages_window = make_refined_set(vec![at_least(0.0), at_most(120.0)]);
    let ages = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(ages_window, None, TrustSpec, SetKindTag::None)
    };
    let got = builtin_call_result_with_kernel("max", &[ages, integer(0.0)], &kernel)
        .expect("max(ages, 0) over a Set operand models through the kernel");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

/// The known-scalar path still wins first — `builtin_call_result_with_kernel`
/// never pays a kernel round trip when `builtin_call_result` alone
/// already answers (both arguments known scalars here).
#[test]
fn max_over_known_scalars_never_reaches_the_kernel_arm() {
    let Some(kernel) = loaded_kernel() else { return };
    let got = builtin_call_result_with_kernel("max", &[integer(3.0), integer(9.0)], &kernel)
        .expect("max(3, 9) models");
    assert_eq!(got.values, vec![9.0]);
}

#[test]
fn min_single_iterable_argument_picks_the_smallest() {
    let list = known_list(vec![integer(3.0), integer(-1.0), integer(5.0)], TrustSpec);
    let got = builtin_call_result("min", &[list]).expect("min([...]) models");
    assert_eq!(got.values, vec![-1.0]);
}

#[test]
fn max_single_iterable_argument_picks_the_largest() {
    let list = known_list(vec![integer(200.0)], TrustSpec);
    let got = builtin_call_result("max", &[list]).expect("max([...]) models");
    assert_eq!(got.values, vec![200.0]);
}

#[test]
fn min_max_empty_iterable_declines() {
    let empty = known_list(vec![], TrustSpec);
    assert!(builtin_call_result("min", &[empty]).is_none());
}

#[test]
fn sum_over_known_list_totals_the_elements() {
    let list = known_list(vec![integer(1.0), integer(2.0), integer(3.0)], TrustSpec);
    let got = builtin_call_result("sum", &[list]).expect("sum([...]) models");
    assert_eq!(got.values, vec![6.0]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
}

#[test]
fn sum_with_a_start_value_adds_it_in() {
    let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
    let got = builtin_call_result("sum", &[list, integer(10.0)]).expect("sum([...], start) models");
    assert_eq!(got.values, vec![13.0]);
}

#[test]
fn sum_widens_to_float_when_any_element_is_float() {
    let list = known_list(vec![integer(1.0), float(2.5)], TrustSpec);
    let got = builtin_call_result("sum", &[list]).expect("sum([...]) models");
    assert_eq!(got.values, vec![3.5]);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
}

/// `D5.edge.helper.py`'s own `sum(s * s for s in clamped)` shape: a
/// GENERATOR expression, which `expressions.rs`'s own `Expr::
/// Generator` arm already routes through `evaluate_list_or_set_comp`
/// — the SAME star-comprehension path a list/set comprehension takes
/// — so once `clamped`'s own element window reaches `sum(...)` as a
/// `Kind::Set` repetition (`s * s` squaring `s ∈ [-1, 1]` down to
/// `[0, 1]`, a fact `expressions.rs`'s own `*` transfer over Set
/// operands states, not this file's concern), `sum_call`'s existing
/// `.or_else(|| sum_call_over_star(arguments))` fallback (this
/// dispatcher's own wiring, unchanged) already answers it — pinned
/// here directly on a star-shaped Float window with no concrete
/// `Kind::List` items, the shape a generator's own star evaluation
/// produces. No new recognition needed in this file: `sum_call_over_
/// star`'s own `star_numeric_hull` gate already accepts any
/// repetition-window `Kind::Set` regardless of whether a generator,
/// a list comprehension, or a declared `list[X]` parameter produced
/// it — the three are indistinguishable once evaluated to this
/// shape.
#[test]
fn sum_over_a_star_shaped_nonnegative_float_window_answers_the_lower_bound_ray() {
    let squared = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..refined_domain::abstract_value::known_set(
            make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
                make_refined_set(vec![at_least(0.0), at_most(1.0)]),
                0,
                None,
            )]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let got = builtin_call_result("sum", &[squared]).expect("sum(star-shaped [0,1] window) must decide through sum_call_over_star");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    // every element is nonnegative, so the running total only ever
    // moves up from the start value (0) — `sum_call_over_star`'s own
    // nonnegative-branch doc
    let want = make_refined_set(vec![at_least(0.0)]);
    assert_eq!(got.set, want);
}

#[test]
fn sorted_over_known_list_ascending() {
    let list = known_list(vec![integer(3.0), integer(1.0), integer(2.0)], TrustSpec);
    let got = builtin_call_result("sorted", &[list]).expect("sorted([...]) models");
    assert_eq!(got.kind, Kind::List);
    assert_eq!(got.items, vec![integer(1.0), integer(2.0), integer(3.0)]);
}

#[test]
fn list_constructor_copies_a_known_list() {
    let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
    let got = builtin_call_result("list", &[list]).expect("list([...]) models");
    assert_eq!(got.kind, Kind::List);
    assert_eq!(got.items, vec![integer(1.0), integer(2.0)]);
}

#[test]
fn set_constructor_copies_a_known_list() {
    let list = known_list(vec![integer(1.0)], TrustSpec);
    let got = builtin_call_result("set", &[list]).expect("set([...]) models");
    assert_eq!(got.kind, Kind::List);
    assert_eq!(got.items, vec![integer(1.0)]);
}

#[test]
fn set_bare_constructor_answers_the_empty_list() {
    let got = builtin_call_result("set", &[]).expect("set() models");
    assert_eq!(got.kind, Kind::List);
    assert_eq!(got.items.len(), 0);
}

#[test]
fn dict_constructor_from_pairs() {
    let pair_a = known_list(vec![string_value("ann"), integer(40.0)], TrustSpec);
    let pair_b = known_list(vec![string_value("bea"), integer(200.0)], TrustSpec);
    let pairs = known_list(vec![pair_a, pair_b], TrustSpec);
    let got = builtin_call_result("dict", &[pairs]).expect("dict([...]) models");
    assert_eq!(got.kind, Kind::Object);
    assert_eq!(got.keys.len(), 2);
}

#[test]
fn dict_constructor_repeated_key_keeps_the_last_value() {
    let pair_a = known_list(vec![string_value("ann"), integer(1.0)], TrustSpec);
    let pair_b = known_list(vec![string_value("ann"), integer(2.0)], TrustSpec);
    let pairs = known_list(vec![pair_a, pair_b], TrustSpec);
    let got = builtin_call_result("dict", &[pairs]).expect("dict([...]) models");
    assert_eq!(got.keys.len(), 1);
    assert_eq!(got.keys[0].value, integer(2.0));
}

/// `xs: list[int]`'s own seeded shape — a `Kind::Set` repetition
/// window (`check.rs::seed_parameters`'s own sequence-container
/// branch, `loops.rs`'s own `for`-loop reader) — bounded `[lo, hi]`
/// with element `[element_lo, element_hi]`. This test module's own
/// stand-in receiver for every `dict.fromkeys`/`list(...)` row below.
fn integer_repetition_window(element_lo: f64, element_hi: f64, lo: i64, hi: Option<i64>) -> AbstractValue {
    let element = make_refined_set(vec![at_least(element_lo), at_most(element_hi), refined_sets::refinement_forms::integer()]);
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(
            make_refined_set(vec![refined_sets::refinement_forms::repeat_of(element, lo, hi)]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    }
}

/// `A15.xfer.dedupe`'s own `dict.fromkeys(xs)` row: a `list[int]`
/// bounded `[0, 150]` answers a round-trip carrier value — `Kind::
/// Object`, `DICT_FROMKEYS_WORD`, `xs` itself carried in `inner` —
/// never a `Kind::List`/`Kind::Object` dict directly (this domain's
/// dict cannot represent windowed, non-string keys, `dict_fromkeys_
/// call`'s own doc).
#[test]
fn dict_fromkeys_over_a_windowed_list_answers_a_round_trip_carrier() {
    let xs = integer_repetition_window(0.0, 150.0, 0, None);
    let got = stdlib_call_result("dict", "fromkeys", &[xs.clone()]).expect("dict.fromkeys(xs) must decide");
    assert_eq!(got.kind, Kind::Object);
    assert_eq!(got.kind_word, Some(DICT_FROMKEYS_WORD));
    assert_eq!(got.inner.as_deref(), Some(&xs));
}

/// `dict.fromkeys(xs, 0)` — the two-argument form, `value` explicit
/// rather than defaulted — still reads the SAME iterable `dict_
/// fromkeys_call`'s own doc states this row does not otherwise
/// inspect `value` for.
#[test]
fn dict_fromkeys_two_argument_form_still_reads_the_iterable() {
    let xs = integer_repetition_window(0.0, 150.0, 0, None);
    let got = stdlib_call_result("dict", "fromkeys", &[xs.clone(), integer(0.0)]).expect("dict.fromkeys(xs, 0) must decide");
    assert_eq!(got.inner.as_deref(), Some(&xs));
}

/// A non-repetition-window argument (an exact `Kind::List`, this
/// domain's own EXACT-arity container — `dict.fromkeys`'s own row is
/// scoped to the unbounded-count windowed shape only) declines.
#[test]
fn dict_fromkeys_over_an_exact_list_declines() {
    let xs = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
    assert_eq!(stdlib_call_result("dict", "fromkeys", &[xs]), None);
}

/// `A15.xfer.dedupe`'s own full row: `list(dict.fromkeys(xs))` for
/// `xs: list[int]` bounded `[0, 150]` answers the SAME element window
/// at a RELAXED length bound (`lo: 0`, `hi` unchanged) — the
/// `for x in deduped:` loop that follows reads `x` through the
/// identical `as_repetition` path a plain `list[int]` parameter
/// already flows through (`loops.rs`), so `0 <= x <= 150` narrows it
/// the same way.
#[test]
fn list_of_dict_fromkeys_answers_the_deduped_element_window() {
    let xs = integer_repetition_window(0.0, 150.0, 3, Some(10));
    let carrier = stdlib_call_result("dict", "fromkeys", &[xs]).expect("dict.fromkeys(xs) must decide");
    let got = builtin_call_result("list", &[carrier]).expect("list(dict.fromkeys(xs)) must decide");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    let expected_element = make_refined_set(vec![at_least(0.0), at_most(150.0), refined_sets::refinement_forms::integer()]);
    let expected = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(expected_element, 0, Some(10))]);
    assert_eq!(got.set, expected, "dedup relaxes lo to 0, keeps hi unchanged, keeps the same element window");
}

/// `list(...)` over an ORDINARY exact `Kind::List` argument still
/// takes the pre-existing row unchanged — the carrier check is
/// gated on `Kind::Object`/`DICT_FROMKEYS_WORD` and never fires for
/// this shape, so `list_constructor_call`'s own long-standing
/// behavior is undisturbed.
#[test]
fn list_of_an_ordinary_list_is_unaffected_by_the_carrier_check() {
    let items = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
    let got = builtin_call_result("list", &[items]).expect("list([...]) models");
    assert_eq!(got.kind, Kind::List);
    assert_eq!(got.items.len(), 2);
}

fn string_value(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    known_values(code_points, PrimitiveKind::String, TrustSpec)
}

#[test]
fn len_declines() {
    let got = builtin_call_result("len", &[integer(3.0)]);
    assert!(got.is_none());
}

#[test]
fn sum_declines() {
    let got = builtin_call_result("sum", &[integer(3.0)]);
    assert!(got.is_none());
}

#[test]
fn unmodeled_name_declines() {
    let got = builtin_call_result("print", &[integer(3.0)]);
    assert!(got.is_none(), "an unmodeled builtin name should decline: {got:?}");
}

#[test]
fn iter_of_a_known_list_reads_as_the_same_list() {
    let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
    let got = builtin_call_result("iter", &[list.clone()]).expect("iter([...]) models");
    assert_eq!(got, list);
}

#[test]
fn iter_of_a_non_list_declines() {
    let got = builtin_call_result("iter", &[integer(1.0)]);
    assert!(got.is_none());
}

#[test]
fn next_of_iter_of_a_known_list_answers_the_first_element() {
    let list = known_list(vec![integer(1.0), integer(2.0)], TrustSpec);
    let iterator = builtin_call_result("iter", &[list]).expect("iter([...]) models");
    let got = builtin_call_result("next", &[iterator]).expect("next(iter([...])) models");
    assert_eq!(got, integer(1.0));
}

#[test]
fn next_of_an_empty_list_declines() {
    let empty = known_list(vec![], TrustSpec);
    let got = builtin_call_result("next", &[empty]);
    assert!(got.is_none(), "next() over an empty iterator should decline: {got:?}");
}

/// `anext` — the async twin of `next`, e-class-and-function.py's own
/// `async_generator_first_value`/`generator_first_value` pair: a
/// generator-tagged List (or a plain iterator List) answers its
/// first element identically whether read through `next` or `anext`.
#[test]
fn anext_of_a_generator_tagged_list_answers_the_first_yielded_value() {
    let mut generator = known_list(vec![integer(40.0), integer(41.0)], TrustSpec);
    generator.source = "generator".to_owned();
    let got = builtin_call_result("anext", &[generator]).expect("anext(generator) models");
    assert_eq!(got, integer(40.0));
}

#[test]
fn anext_of_an_empty_list_declines() {
    let empty = known_list(vec![], TrustSpec);
    let got = builtin_call_result("anext", &[empty]);
    assert!(got.is_none(), "anext() over an empty generator should decline: {got:?}");
}

#[test]
fn cast_returns_the_value_argument_unchanged() {
    // the `typ` argument is never read by `cast` — an unknown value
    // there does not block the answer
    let unread_type_argument = AbstractValue::default();
    let got = builtin_call_result("cast", &[unread_type_argument, integer(200.0)]).expect("cast(...) models");
    assert_eq!(got, integer(200.0));
}

#[test]
fn cast_wrong_arity_declines() {
    let got = builtin_call_result("cast", &[integer(200.0)]);
    assert!(got.is_none());
}

fn exception_instance(message: &str) -> AbstractValue {
    let args = known_list(vec![string_value(message)], TrustSpec);
    let mut instance = known_object_helper(vec![("args", args)]);
    instance.source = "exception".to_owned();
    instance
}

fn known_object_helper(entries: Vec<(&str, AbstractValue)>) -> AbstractValue {
    use refined_domain::abstract_value::ObjectKey;
    use refined_domain::known_constructors::known_object;
    let keys = entries
        .into_iter()
        .map(|(name, value)| ObjectKey { name: name.to_owned(), numeric: false, value })
        .collect();
    known_object(keys, None, true, TrustSpec, false)
}

#[test]
fn str_of_a_single_string_argument_exception_answers_the_message() {
    let instance = exception_instance("failure");
    let got = builtin_call_result("str", &[instance]).expect("str(Exception(...)) models");
    assert_eq!(exact_text(&got), "failure");
}

fn exact_text(value: &AbstractValue) -> String {
    value.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect()
}

#[test]
fn str_of_an_exception_with_no_args_declines() {
    let mut instance = known_object_helper(vec![("args", known_list(vec![], TrustSpec))]);
    instance.source = "exception".to_owned();
    let got = builtin_call_result("str", &[instance]);
    assert!(got.is_none(), "a zero-argument exception's __str__ (empty string) is not modeled: {got:?}");
}

fn integer_window(lo: f64, hi: f64) -> AbstractValue {
    AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..refined_domain::abstract_value::known_set(
            make_refined_set(vec![at_least(lo), at_most(hi), refined_sets::refinement_forms::integer()]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    }
}

#[test]
fn str_of_a_bounded_integer_window_answers_the_decimal_digit_grammar() {
    // str(n) over n in [0, 255]: the decimal spelling runs 1 to 3
    // digits ("0".."255"), every digit drawn from 0-9 —
    // `integer_window_grammar`'s own composition, reused unchanged.
    let got = builtin_call_result("str", &[integer_window(0.0, 255.0)]).expect("str(n) over [0, 255] models");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
    let digits: Vec<f64> = "0123456789".chars().map(|c| c as u32 as f64).collect();
    let expected = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(
        make_refined_set(vec![one_of(&digits)]),
        1,
        Some(3),
    )]);
    assert_eq!(got.set, expected);
}

#[test]
fn str_of_a_negative_lower_bound_integer_window_declines() {
    // `integer_window_grammar`'s own `lo < 0` refusal — no signed
    // digit-run grammar is built here.
    let got = builtin_call_result("str", &[integer_window(-5.0, 5.0)]);
    assert!(got.is_none(), "str(n) over a window with a negative lower bound should decline: {got:?}");
}

#[test]
fn object_call_answers_an_opaque_value_tagged_for_identity_keying() {
    let got = builtin_call_result("object", &[]).expect("object() models");
    assert_eq!(got.kind, Kind::Object);
    assert_eq!(got.kind_word, Some("a featureless object"));
    assert_eq!(got.source, "object()");
}

#[test]
fn object_call_with_an_argument_declines() {
    // library/functions.html#object: "The constructor does not
    // accept any arguments."
    let got = builtin_call_result("object", &[integer(1.0)]);
    assert!(got.is_none(), "object(x) should decline: {got:?}");
}

#[test]
fn hash_of_a_bounded_int_answers_the_unbounded_integer_sort() {
    // hash(x) for x: int is a Python int (library/functions.html#hash),
    // but this row states no identity claim beyond the sort — a
    // later band guard is what narrows it, exactly A15.xfer.hash's
    // own fixture shape. The set carries an EXPLICIT AtLeast(-inf)
    // ray alongside Integer — the same two-form "whole integer
    // ground" shape `narrowing.rs::unbounded_integers()` and this
    // file's own `int_image()` both build — never Integer alone
    // with zero ray forms (A15.xfer.hash's own `hash_outside` row:
    // a bare-Integer set with no ray form let a one-sided `>= 0`
    // guard's own narrowed window silently pass a declared-bounded
    // Age sink).
    let bounded = integer_window(0.0, 150.0);
    let got = builtin_call_result("hash", &[bounded]).expect("hash(x) models");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(f64::NEG_INFINITY)]);
    assert_eq!(got.set, want, "hash(x) must carry an explicit unbounded ray, not a bare Integer form: got {:?}", got.set);
}

#[test]
fn hash_wrong_arity_declines() {
    let got = builtin_call_result("hash", &[]);
    assert!(got.is_none());
}

/// A15.xfer.hash's own `hash_outside` soundness row, pinned directly
/// against the kernel: `hash(x)`'s own unbounded-both-ways ground,
/// narrowed by `h >= 0` alone (the ONE ray a one-sided guard can ever
/// tighten — `narrowing.rs::meet_set_answer`'s own intersection),
/// must NOT prove a subset of Age's declared `[0, 150] && integer`
/// window — an unbounded-above ray is never contained in a set
/// bounded above, so `assignability.rs`'s own `scalar_subset` ask
/// (the exact containment question `judge`'s `Kind::Set` arm poses at
/// the `return h` sink) must answer `false` here. Before this fix,
/// `hash_call`'s bare-`Integer` set (no ray form at all) reached this
/// same ask and was silently admitted — the reproducer this test
/// pins the refusal for.
#[test]
fn hash_narrowed_only_below_is_not_a_subset_of_a_bounded_declared_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let bare_int = integer_window(0.0, 150.0);
    let hash_result = builtin_call_result("hash", &[bare_int]).expect("hash(x) models");
    // `h >= 0` narrows only the lower ray — the SAME `meet_set_answer`
    // intersection `narrowing.rs` performs for a one-sided guard,
    // reproduced here directly on the set rather than through the
    // full narrowing walk, since this file's own tests stay
    // kernel-optional and narrowing-free.
    let mut narrowed_forms = hash_result.set.forms.clone();
    narrowed_forms.push(at_least(0.0));
    let narrowed_set = make_refined_set(narrowed_forms);
    let age_declared = make_refined_set(vec![at_least(0.0), at_most(150.0), refined_sets::refinement_forms::integer()]);
    let is_subset = (kernel.scalar_subset)(&narrowed_set, &age_declared);
    assert!(
        !is_subset,
        "hash(x) narrowed only below by `>= 0` must not be a subset of Age's bounded window: narrowed {:?}, declared {:?}",
        narrowed_set, age_declared
    );
}

#[test]
fn time_time_answers_a_nonnegative_float_ground() {
    let got = stdlib_call_result("time", "time", &[]).expect("time.time() models");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Float));
    assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
}

#[test]
fn time_time_with_an_argument_declines() {
    let got = stdlib_call_result("time", "time", &[integer(1.0)]);
    assert!(got.is_none());
}

#[test]
fn os_open_answers_a_nonnegative_integer_ground() {
    let got = stdlib_call_result("os", "open", &[string_value("/tmp/x"), integer(0.0)]).expect("os.open(...) models");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::Integer));
    assert_eq!(
        got.set,
        make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::integer()])
    );
}

#[test]
fn os_close_answers_none() {
    let got = stdlib_call_result("os", "close", &[integer(3.0)]).expect("os.close(fd) models");
    assert_eq!(got.kind, Kind::Null);
}

#[test]
fn test_A3_xfer_normalize_nfc_composes_the_decomposed_pair() {
    // "e" + U+0301 is the decomposed spelling of "é"; NFC composes the
    // two code points into the one code point U+00E9, so the answer's
    // own len() is 1 — A3.xfer.normalize's `composed_length_inside`.
    let form = string_value("NFC");
    let subject = string_value("e\u{0301}");
    let got = stdlib_call_result("unicodedata", "normalize", &[form, subject]).expect("unicodedata.normalize(...) models");
    assert_eq!(got.kind, Kind::Values);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
    assert_eq!(exact_text(&got), "\u{00E9}");
    assert_eq!(got.values.len(), 1);
}

#[test]
fn test_A3_xfer_normalize_nfd_decomposes_the_composed_character() {
    let form = string_value("NFD");
    let subject = string_value("\u{00E9}");
    let got = stdlib_call_result("unicodedata", "normalize", &[form, subject]).expect("unicodedata.normalize(...) models");
    assert_eq!(exact_text(&got), "e\u{0301}");
    assert_eq!(got.values.len(), 2);
}

#[test]
fn test_A3_xfer_normalize_leaves_an_ascii_string_exactly_as_it_stands() {
    let form = string_value("NFC");
    let subject = string_value("AA");
    let got = stdlib_call_result("unicodedata", "normalize", &[form, subject]).expect("unicodedata.normalize(...) models");
    assert_eq!(exact_text(&got), "AA");
}

#[test]
fn unicodedata_normalize_with_an_unknown_form_declines() {
    let form = string_value("bogus");
    let subject = string_value("x");
    let got = stdlib_call_result("unicodedata", "normalize", &[form, subject]);
    assert!(got.is_none(), "an unrecognized normalization form should decline: {got:?}");
}

#[test]
fn urllib_parse_quote_answers_the_whole_strings_ground() {
    // reached as a bare-name builtin (`from urllib.parse import
    // quote` then `quote(s)`), not through stdlib_call_result — see
    // urllib_quote_call's own doc.
    let subject = string_value("a b");
    let got = builtin_call_result("quote", &[subject]).expect("quote(...) models");
    assert_eq!(got.kind, Kind::Set);
    assert_eq!(got.kind_tag, Some(PrimitiveKind::String));
}

#[test]
fn unmodeled_stdlib_module_declines() {
    let got = stdlib_call_result("sys", "exit", &[]);
    assert!(got.is_none());
}
