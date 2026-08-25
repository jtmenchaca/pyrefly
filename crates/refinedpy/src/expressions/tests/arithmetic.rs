use super::*;

/// `binary_arithmetic_value` directly, no kernel needed (pure
/// computation over two known AbstractValues) — pins the exported
/// signature `loops.rs`'s AugAssign path calls, and the sort rule a
/// mixed Integer/Float `+` widens to Float per stdtypes' own mixed-
/// arithmetic rule.
#[test]
fn test_binary_arithmetic_value_mixed_sort_widens_to_float() {
    let ten_int = known_values(vec![10.0], PrimitiveKind::Integer, TrustProved);
    let half_float = known_values(vec![0.5], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value(Operator::Add, &ten_int, &half_float);
    assert_eq!(result.values, vec![10.5]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// `inf - inf` — a Float `Sub` result that is NaN (IEEE 754). This
/// must answer the domain's own `Kind::NaN` state rather than panic:
/// `arithmetic_result`'s Float row screens for NaN and answers
/// `nan_value()` instead of building `known_values(vec![NaN], ..)`,
/// which `refinement_forms::element` would refuse at construction
/// the moment the value crossed into a `one_of` set
/// (showcase.py's `record_ratio(inf - inf)` row).
#[test]
fn test_binary_arithmetic_value_inf_minus_inf_is_the_nan_state_not_a_panic() {
    let positive_infinity = known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value(Operator::Sub, &positive_infinity, &positive_infinity);
    assert_eq!(result.kind, Kind::NaN, "{result:?}");
}

/// `inf * 0` — a Float `Mult` result that is NaN (IEEE 754), the
/// second of showcase.py's three NaN-producing rows
/// (`record_ratio(inf * 0)`). Same non-panicking `Kind::NaN` answer
/// as the `Sub` row above.
#[test]
fn test_binary_arithmetic_value_inf_times_zero_is_the_nan_state_not_a_panic() {
    let positive_infinity = known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved);
    let zero = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value(Operator::Mult, &positive_infinity, &zero);
    assert_eq!(result.kind, Kind::NaN, "{result:?}");
}

/// `inf / inf` — a non-zero divisor (so the `ZeroDivisionError`
/// decline does not apply), still NaN by IEEE 754. Pins the `Div`
/// row's own route through `arithmetic_result` rather than a direct
/// `known_values` call.
#[test]
fn test_binary_arithmetic_value_inf_over_inf_is_the_nan_state_not_a_panic() {
    let positive_infinity = known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value(Operator::Div, &positive_infinity, &positive_infinity);
    assert_eq!(result.kind, Kind::NaN, "{result:?}");
}

/// `{1.0, 2.0} * 2.0` — a MULTI-valued `Kind::Values` operand
/// against a single-valued one: the exact pointwise answer `{2.0,
/// 4.0}`, not `unknown()`. This is the row a loop's second judged
/// pass needs: a first-pass join can leave `total` bound to exactly
/// this two-element shape, and a decline here is what collapses a
/// stabilizing accumulation onto the coarse "not yet walked"
/// blocker instead of the fixed-point one.
#[test]
fn test_binary_arithmetic_value_multi_valued_operand_answers_the_pointwise_cross_product() {
    let one_and_two = known_values(vec![1.0, 2.0], PrimitiveKind::Float, TrustProved);
    let two = known_values(vec![2.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value(Operator::Mult, &one_and_two, &two);
    assert_eq!(result.kind, Kind::Values, "{result:?}");
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    let mut values = result.values.clone();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![2.0, 4.0]);
}

/// `{1.0, 2.0} + {10.0, 20.0}` — BOTH operands multi-valued: the
/// full cross product, four pointwise sums, deduped (none collide
/// here) — `1+10, 1+20, 2+10, 2+20`.
#[test]
fn test_binary_arithmetic_value_both_operands_multi_valued_answers_the_full_cross_product() {
    let one_and_two = known_values(vec![1.0, 2.0], PrimitiveKind::Float, TrustProved);
    let ten_and_twenty = known_values(vec![10.0, 20.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value(Operator::Add, &one_and_two, &ten_and_twenty);
    assert_eq!(result.kind, Kind::Values, "{result:?}");
    let mut values = result.values.clone();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![11.0, 12.0, 21.0, 22.0]);
}

/// A cross product past `MULTI_VALUE_CROSS_PRODUCT_CAP` falls
/// through to whatever the existing set/transfer path answers today
/// — pinned as NOT `Kind::Values` (this function's own multi-value
/// row must not fire), rather than pinning a specific set shape the
/// set/transfer path's own tests already own.
#[test]
fn test_binary_arithmetic_value_cross_product_past_the_cap_falls_through() {
    let left_values: Vec<f64> = (0..5).map(|n| n as f64).collect();
    let right_values: Vec<f64> = (0..5).map(|n| 100.0 + n as f64).collect();
    let left = known_values(left_values, PrimitiveKind::Float, TrustProved);
    let right = known_values(right_values, PrimitiveKind::Float, TrustProved);
    // 5 * 5 = 25 pairs, past the 16-pair cap
    let result = binary_arithmetic_value(Operator::Add, &left, &right);
    assert_ne!(
        result.kind,
        Kind::Values,
        "a cross product past the cap must fall through, not answer Kind::Values: {result:?}"
    );
}

/// `binary_arithmetic_value` on two known STRINGS falls through to
/// string concatenation — the row `label += "c"`-style AugAssign
/// calls depend on, matching the equivalent `label = label + "c"`
/// BinOp exactly.
#[test]
fn test_binary_arithmetic_value_falls_through_to_string_concat() {
    let a = string_models::string_literal_value("ab");
    let b = string_models::string_literal_value("c");
    let result = binary_arithmetic_value(Operator::Add, &a, &b);
    assert_eq!(exact_string_values(&result).and_then(code_points_to_string).as_deref(), Some("abc"));
}

/// `&`/`|`/`^` on two known int-sorted values are exact per §6.8 —
/// pins `40 | 200 == 232` (CPython-checked), the exact fold
/// `compound_bitwise_on_number_slot`'s `age |= 200` depends on to
/// carry a judgeable value past Age's 120 ceiling instead of
/// declining to unknown().
#[test]
fn test_binary_arithmetic_value_bitwise_or_is_exact() {
    let forty = known_values(vec![40.0], PrimitiveKind::Integer, TrustProved);
    let two_hundred = known_values(vec![200.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value(Operator::BitOr, &forty, &two_hundred);
    assert_eq!(result.values, vec![232.0]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// `&`/`^` follow the same exact two's-complement law as `|` —
/// CPython-checked: `5 & 3 == 1`, `5 ^ 3 == 6`.
#[test]
fn test_binary_arithmetic_value_bitwise_and_xor_are_exact() {
    let five = known_values(vec![5.0], PrimitiveKind::Integer, TrustProved);
    let three = known_values(vec![3.0], PrimitiveKind::Integer, TrustProved);
    let and_result = binary_arithmetic_value(Operator::BitAnd, &five, &three);
    assert_eq!(and_result.values, vec![1.0]);
    let xor_result = binary_arithmetic_value(Operator::BitXor, &five, &three);
    assert_eq!(xor_result.values, vec![6.0]);
}

/// `<<`/`>>` on two known int-sorted values are exact per §6.8:
/// `x << n` is `x * 2**n`, `x >> n` floors `x / 2**n` — CPython-
/// checked: `1 << 5 == 32`, `(-8) >> 2 == -2` (floors toward
/// negative infinity, not truncates toward zero).
#[test]
fn test_binary_arithmetic_value_shifts_are_exact() {
    let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let five = known_values(vec![5.0], PrimitiveKind::Integer, TrustProved);
    let left_shifted = binary_arithmetic_value(Operator::LShift, &one, &five);
    assert_eq!(left_shifted.values, vec![32.0]);

    let negative_eight = known_values(vec![-8.0], PrimitiveKind::Integer, TrustProved);
    let two = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
    let right_shifted = binary_arithmetic_value(Operator::RShift, &negative_eight, &two);
    assert_eq!(right_shifted.values, vec![-2.0]);
}

/// A negative shift count raises ValueError in CPython — this file
/// has no exception channel for a binary operator's own decline, so
/// it declines to unknown() rather than claim a value CPython never
/// produces.
#[test]
fn test_binary_arithmetic_value_negative_shift_declines() {
    let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let negative_one = known_values(vec![-1.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value(Operator::LShift, &one, &negative_one);
    assert_eq!(result.kind, Kind::Unknown);
}

/// A float operand to a bitwise op raises TypeError in CPython
/// (unsupported operand type) — `single_numeric_value` reads a bare
/// Float-sorted operand as non-int, so `both_int` is false and this
/// declines rather than guess a two's-complement pattern for a
/// value that was never an int.
#[test]
fn test_binary_arithmetic_value_bitwise_float_operand_declines() {
    let one_float = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
    let one_int = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value(Operator::BitAnd, &one_float, &one_int);
    assert_eq!(result.kind, Kind::Unknown);
}

/// `age + 1` where `age` is a seeded int-sorted SET `[0, 120]` — the
/// mission's own headline case: the known-values path declines (age
/// is not one known value), so `binary_arithmetic_value_with_kernel`
/// takes the SET path and the kernel's `transferAdd` answers the
/// certified enclosure `[1, 121]`, Integer-sorted (both operands
/// Integer). Asserted via `scalar_subset` both directions so the
/// answer set is pinned exactly, not merely "some Set."
#[test]
fn test_set_plus_known_int_lowers_through_kernel_transfer() {
    let Some(kernel) = loaded_kernel() else { return };
    let age = known_set(
        make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
    let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Add, &age, &one, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![integer(), at_least(1.0), refined_sets::refinement_forms::at_most(121.0)]);
    assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
    assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
}

/// The UNBOUNDED float-set row: `float_sorted_unknown() * 2` — the
/// operand is the whole numeric line (math.sqrt's sort-only shape),
/// and the kernel's transfer answers no tighter certified image for
/// an unbounded operand. The answer is the SORT-ONLY unbounded set
/// (the same language-level guarantee the math family carries), not
/// nothing: the product of two numerics is a numeric, and a
/// downstream clamp can still bound it. The BOUNDED-set row above
/// is where the transfer certifies a tight image; this row pins
/// that an unbounded operand keeps its sort and loses its bounds —
/// never a guessed value, never a dropped one.
#[test]
fn test_float_sorted_set_times_known_int_answers_the_sort_when_unbounded() {
    let Some(kernel) = loaded_kernel() else { return };
    let sqrt_result = float_sorted_unknown();
    let two = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Mult, &sqrt_result, &two, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    let everything = refined_sets::refinement_forms::numbers();
    assert!((kernel.scalar_subset)(&result.set, &everything), "the sort-only answer must stay inside the numeric line");
    assert!((kernel.scalar_subset)(&everything, &result.set), "the sort-only answer must not invent bounds the transfer never certified");
}

/// `age % 7` where `age` is a seeded Integer-sorted set `[0, 120]` —
/// `admitted_int_transfer_op` elects `rem.divisorSign` for `Mod` on
/// the int-sorted path (arith.4, the Python-owned remainder), so
/// `int_transfer_over_sets` asks the kernel rather than declining.
/// `theories/rem/divisor_sign.lean`'s own general-enclosure branch,
/// worked by hand for this exact operand pair: `age` is a range (not
/// a singleton), `7` is a singleton nonzero divisor, so the answer
/// comes from the `bothSingle = none` arm — `divisorBound = 7`
/// (finite), both operands Integer-sorted and `7` itself an integer
/// dyadic, so the TIGHTENED case applies (`magnitude = 7 − 1 = 6`);
/// the divisor is nonnegative, so the window sits on `[0, magnitude]`
/// with neither endpoint strict — `[0, 6]`, matching the fixture's
/// own `int_modulo_over_declared_range` row (`b-body-expressions.py`,
/// "`count % 7` lands in Remainder's 0..6"). Asserted via
/// `scalar_subset` both directions, the same pinning style
/// `test_set_plus_known_int_lowers_through_kernel_transfer` uses.
#[test]
fn test_mod_over_an_int_sorted_set_serves_the_divisor_sign_row() {
    let Some(kernel) = loaded_kernel() else { return };
    let age = known_set(
        make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
    let seven = known_values(vec![7.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Mod, &age, &seven, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(6.0)]);
    assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
    assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
}

/// The FLOAT-path exclusion: `age % 7.0` where `age` is a
/// Float-sorted set — `admitted_transfer_op` (the float/mixed-sort
/// path `int_transfer_over_sets` falls through to whenever either
/// operand is not Integer-sorted) has no `Mod` arm at all, so
/// `transfer_over_sets` declines outright, and
/// `binary_arithmetic_value_with_kernel` falls through to the
/// ordinary known-values path, which also declines (a Set is not one
/// known value) — the whole call answers `unknown()`. `%`'s
/// divisor-sign election is admitted ONLY on the exact int theory
/// (`rem.divisorSign` has no float-sorted counterpart wired here);
/// this is the row the now-renamed test above no longer covers.
#[test]
fn test_mod_over_a_float_sorted_set_still_declines() {
    let Some(kernel) = loaded_kernel() else { return };
    let age = known_set(
        make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let age = AbstractValue { kind_tag: Some(PrimitiveKind::Float), ..age };
    let seven = known_values(vec![7.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Mod, &age, &seven, &kernel);
    assert_eq!(result.kind, Kind::Unknown);
}

/// `count << 1` where `count` is a seeded Integer-sorted set `[0, 10]`
/// (`SmallCount` in `b-body-expressions.py`) — `shift_as_int_composition`
/// lowers `LShift` as `int.mul` against the singleton `{2**1}` (bits.2)
/// first, but `int.mul` (`boundary/python.lean`) only ever matches
/// `exactIntOf` on BOTH sides — it has no general-window arm the way
/// `int.floorDiv` does, so a non-singleton `left_set` like this one
/// always reads back `.unknown` from `int.mul` itself. The row this
/// test actually pins is the FALLBACK: `float_mul_as_shift_fallback`
/// retries the same two sets against the float image's `Mul`
/// (`binary64.mul`), which DOES narrow a bounded window times a
/// singleton, and re-tags the result Integer-sorted (sound: `factor`
/// is gated to an exact power of two inside the f64-exact window, so
/// the float product never rounds away from the integer `int.mul`
/// would have named). The fixture's own
/// `int_left_shift_over_declared_range` row states the window this
/// pins: "`count << 1` is `count * 2`, 0..20" — worked by hand,
/// `binary64.mul([0, 10], {2})` is the exact range `[0, 20]`.
#[test]
fn test_left_shift_over_an_int_sorted_set_serves_the_float_mul_fallback() {
    let Some(kernel) = loaded_kernel() else { return };
    let count = known_set(
        make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(10.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let count = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..count };
    let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::LShift, &count, &one, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(20.0)]);
    assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
    assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
}

/// `n >> 1` — the same composition on the OTHER shift, `int.floorDiv`
/// against `{2**1}` (bits.1). `count` reuses `int_left_shift...`'s own
/// `[0, 10]` set; `int.floorDiv([0, 10], {2})` is `[0, 5]`.
#[test]
fn test_right_shift_over_an_int_sorted_set_serves_the_floor_div_composition() {
    let Some(kernel) = loaded_kernel() else { return };
    let count = known_set(
        make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(10.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let count = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..count };
    let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::RShift, &count, &one, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
    let want = make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]);
    assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
    assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
}

/// `m ** 2` where `m` is a Float-sorted set `[0.5, 2.5]` (showcase.py's
/// `Meters`, `bmi`'s own `m**2` step) — `pow_over_sets`' nonnegative-
/// real-base composition (`transferRealPow`): `pow([0.5, 2.5], {2})`
/// widens by the k-ulp approximation envelope around the exact
/// corners `0.25`/`6.25`. This is the row that was UNREACHABLE before
/// this wave's own wire-op fix (`transfer_questions.rs::transfer_wire`
/// spelled the `Pow` op `"pow"`, never matching the boundary's own
/// `"pow.binary64"` dispatch arm — every `Pow` transfer question, from
/// any caller, silently read back `.unknown` regardless of the
/// operand shape).
#[test]
fn test_float_pow_over_a_set_base_serves_the_real_pow_window() {
    let Some(kernel) = loaded_kernel() else { return };
    let m = known_set(
        make_refined_set(vec![at_least(0.5), refined_sets::refinement_forms::at_most(2.5)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let m = AbstractValue { kind_tag: Some(PrimitiveKind::Float), ..m };
    let two = known_values(vec![2.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Pow, &m, &two, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    // the k-ulp envelope widens slightly past the exact corners —
    // asserting CONTAINMENT of the exact window rather than equality,
    // since the precise widened bound is the kernel's own approximated
    // step, not a value this file computes independently
    let exact = make_refined_set(vec![at_least(0.25), refined_sets::refinement_forms::at_most(6.25)]);
    assert!((kernel.scalar_subset)(&exact, &result.set), "exact window {:?} not ⊆ result {:?}", exact, result.set);
}

/// `x ** 0` is exactly `1` for EVERY `x`, including an UNBOUNDED
/// Float-sorted set — `pow_over_sets`' own pinned `k = 0` branch,
/// answered directly rather than through either Lean window decider
/// (neither reads an exponent of `0`). showcase.py's own
/// `anything_to_the_zeroth(x: float) -> ExactlyOne` is this row: `x`
/// carries no declared bound at all.
#[test]
fn test_pow_zero_exponent_over_an_unbounded_set_answers_one() {
    let Some(kernel) = loaded_kernel() else { return };
    let x = AbstractValue { kind_tag: Some(PrimitiveKind::Float), ..known_set(make_refined_set(vec![]), None, TrustProved, SetKindTag::None) };
    let zero = known_values(vec![0.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Pow, &x, &zero, &kernel);
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    assert_eq!(result.values, vec![1.0]);
}

/// `[0, 9] ** [0, 9]` — BOTH the base and the exponent are Integer-
/// sorted `Kind::Set` windows, neither a known scalar —
/// `pow_over_sets`' own windowed-exponent arm: no `exact_nonnegative_
/// integer` reading applies (there is no single `k` to pin `x ** 0`
/// or the `[1, 64]` corner against), so `exp_set` rides the wire
/// exactly as `base_set` already does, and `transferIntegerPow`
/// (`theories/pow/binary64.lean`) is the decider for the whole
/// question. Only the answer's SHAPE is asserted (a determined set,
/// never a decline) — the exact corner hull `[0, 387420489]` is the
/// kernel's own composed bound, not a value this file computes
/// independently.
#[test]
fn test_pow_over_a_windowed_base_and_a_windowed_exponent_answers_a_set() {
    let Some(kernel) = loaded_kernel() else { return };
    let window = || {
        AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(
                make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(9.0)]),
                None,
                TrustProved,
                SetKindTag::None,
            )
        }
    };
    let base = window();
    let exponent = window();
    let result = binary_arithmetic_value_with_kernel(Operator::Pow, &base, &exponent, &kernel);
    assert_eq!(result.kind, Kind::Set);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Integer));
}

/// Two known single values over an admitted operator (`+`) still
/// take the ORIGINAL fast path, never the kernel round-trip — the
/// set-gate in `transfer_over_sets` declines outright the moment
/// neither operand is `Kind::Set`, so this stays the pure-Rust
/// answer `test_binary_arithmetic_value_mixed_sort_widens_to_float`
/// already pins, unchanged by this wave.
#[test]
fn test_two_known_values_skip_the_kernel_set_path() {
    let Some(kernel) = loaded_kernel() else { return };
    let ten_int = known_values(vec![10.0], PrimitiveKind::Integer, TrustProved);
    let half_float = known_values(vec![0.5], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Add, &ten_int, &half_float, &kernel);
    assert_eq!(result.kind, Kind::Values);
    assert_eq!(result.values, vec![10.5]);
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// A refusal the kernel `transfer` closure panics on (an untagged
/// Set — string-sorted by convention, ORIENTATION.md's own
/// recognition-slice fact) is CAUGHT by `transfer_over_sets`'
/// `catch_unwind` and answered as a decline, never a crash —
/// `transferable_numeric_operand` itself already declines an
/// untagged Set before any kernel ask, so this exercises that
/// decline path rather than a live kernel panic; the two are
/// observationally the same "falls back to unknown()" outcome the
/// mission asks for.
#[test]
fn test_untagged_set_declines_before_any_kernel_ask() {
    let Some(kernel) = loaded_kernel() else { return };
    let untagged = known_set(strings(), None, TrustProved, SetKindTag::None);
    let one = known_values(vec![1.0], PrimitiveKind::Integer, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Add, &untagged, &one, &kernel);
    assert_eq!(result.kind, Kind::Unknown);
}

// --- `/` at a SET-SHAPED divisor that may admit zero ---

/// `1.0 / denominator` where `denominator` is a seeded Float-sorted
/// SET `[0.0, 2.0]` — a WIDE window admitting zero, but NOT entirely
/// zero (`divisor_is_provably_always_zero` is false — the window
/// has non-zero members too). `split_divisor_transfer`'s own fix:
/// the value question no longer declines outright at this shape —
/// it splits the divisor into its zero-excluded halves (`(0.0, 2.0]`
/// here; the negative half, `< 0.0`, is empty and skipped) and asks
/// `binary64.div` on the non-empty half. The kernel's OWN general-
/// interval branch (`divisorMayBeZero`, `theories/binary64/div.lean`)
/// still cannot narrow `1.0 / (0.0, 2.0]` to a tight enclosure even
/// with zero excluded, so the split's own answer is `Unknown` —
/// which this function reads as `float_sorted_unknown()` (sort-
/// known, value-unknown), never `Kind::Unknown` outright. The value
/// question DETERMINES a sort here, on the non-raising split, exactly
/// as every other admitted transfer answer already does — the raise
/// arm at `x == 0.0` itself is a separate, unaddressed question
/// (`binop_provable_raise` only fires when the WHOLE window is zero).
#[test]
fn test_div_by_a_set_that_may_admit_zero_determines_the_float_sort_over_the_zero_excluded_split() {
    let Some(kernel) = loaded_kernel() else { return };
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
    assert_eq!(
        result.kind,
        Kind::Set,
        "the zero-excluded split must determine a value (sort-only, at minimum), never decline outright: {result:?}"
    );
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
}

/// The SOLE-GUARD row: `1.0 / denominator` where `denominator` is a
/// DEGENERATE Set carrying nothing but `{0.0}` (`one_of`, `Kind::Set`
/// rather than the ordinary `Kind::Values` `single_numeric_value`
/// already reads). Unlike the wide-window row above, the kernel's
/// OWN `bothSingle` branch (`theories/binary64/div.lean`) answers a
/// DETERMINED `±Infinity` pair for this exact shape — so this row is
/// the one `divisor_provably_excludes_zero` alone protects; without
/// the gate, `transfer_over_sets` would relabel that pair as
/// Python's answer, which is the unsound row this whole unit fixes.
#[test]
fn test_div_by_a_degenerate_zero_only_set_declines_where_the_kernel_would_otherwise_answer() {
    let Some(kernel) = loaded_kernel() else { return };
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
    };
    let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
    assert_eq!(
        result.kind,
        Kind::Unknown,
        "a degenerate zero-only divisor Set must decline — the kernel's bothSingle branch answers \
         a determined ±Infinity pair here, and relabeling it as Python's answer is exactly the \
         unsoundness this gate exists to prevent: {result:?}"
    );
}

/// The mirror row: a divisor set that PROVABLY EXCLUDES zero (a
/// window `[1.0, 2.0]`, strictly above zero) still lowers through
/// `binary64.div` — the gate only refuses the zero-admitting case,
/// it does not disable the SET path outright. `1.0 / [1.0, 2.0]`
/// certifies to `[0.5, 1.0]`.
#[test]
fn test_div_by_a_set_that_provably_excludes_zero_still_lowers_through_the_kernel() {
    let Some(kernel) = loaded_kernel() else { return };
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(1.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
    let result = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
    assert_eq!(result.kind, Kind::Set, "a zero-excluding divisor must still answer: {result:?}");
    assert_eq!(result.kind_tag, Some(PrimitiveKind::Float));
    let want = make_refined_set(vec![at_least(0.5), refined_sets::refinement_forms::at_most(1.0)]);
    assert!((kernel.scalar_subset)(&result.set, &want), "result {:?} not ⊆ want {:?}", result.set, want);
    assert!((kernel.scalar_subset)(&want, &result.set), "want {:?} not ⊆ result {:?}", want, result.set);
}

/// The pinning ask for `divisor_provably_excludes_zero` directly: a
/// half-open ray `(0.0, ∞)` (strictly positive, zero itself NOT a
/// member) excludes zero, while `[0.0, ∞)` (zero included) does not.
#[test]
fn test_divisor_provably_excludes_zero_reads_strict_vs_inclusive_bounds() {
    let Some(kernel) = loaded_kernel() else { return };
    let strictly_positive = make_refined_set(vec![refined_sets::refinement_forms::above(0.0)]);
    assert!(
        divisor_provably_excludes_zero(&strictly_positive, &kernel),
        "a strictly-positive ray must be proved to exclude zero"
    );
    let nonnegative = make_refined_set(vec![at_least(0.0)]);
    assert!(
        !divisor_provably_excludes_zero(&nonnegative, &kernel),
        "a nonnegative ray admits zero and must not be proved to exclude it"
    );
}

// --- `//`/`%` at a SET-SHAPED divisor that may admit zero ---

/// `age // denominator` where `denominator` is a seeded Integer-sorted
/// SET `[0, 5]` — the `//`/`%` twin of the `/` corner above, checked
/// for the SAME hazard. `admitted_int_transfer_op`'s `int.floorDiv`
/// row only ever answers over TWO EXACT SINGLETONS
/// (`boundary/python.lean`'s `exactIntOf A, exactIntOf B` match); a
/// range divisor is not a singleton, so the kernel itself refuses
/// (`.unknown`) before any zero-admission question is even reached —
/// this row is sound by construction, with no adapter-side gate
/// needed. Pinned here so the finding is asserted, not merely
/// claimed.
#[test]
fn test_floor_div_by_a_set_that_may_admit_zero_declines_because_the_kernel_refuses_ranges() {
    let Some(kernel) = loaded_kernel() else { return };
    let age = known_set(
        make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    let result = binary_arithmetic_value_with_kernel(Operator::FloorDiv, &age, &denominator, &kernel);
    assert_eq!(
        result.kind,
        Kind::Unknown,
        "a range divisor has no int.floorDiv row at all (exact singletons only) — declines: {result:?}"
    );
}

/// The `%` twin: `age % denominator` over the SAME `[0, 5]` divisor
/// window. `rem.divisorSign` DOES have a general-interval branch
/// (unlike `int.floorDiv`), so this is the row that actually
/// exercises `theories/rem/divisor_sign.lean`'s own `divisorMayBeZero`
/// gate rather than merely a singleton-only refusal: the kernel
/// itself declines (`.unknown`) the moment the divisor's range
/// admits zero, so the adapter's decline here is inherited soundly
/// from the kernel, with no separate adapter-side gate needed for
/// `Mod` either.
#[test]
fn test_mod_by_a_set_that_may_admit_zero_declines_because_the_kernel_gates_the_interval_branch() {
    let Some(kernel) = loaded_kernel() else { return };
    let age = known_set(
        make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(120.0)]),
        None,
        TrustProved,
        SetKindTag::None,
    );
    let age = AbstractValue { kind_tag: Some(PrimitiveKind::Integer), ..age };
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![integer(), at_least(0.0), refined_sets::refinement_forms::at_most(5.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    let result = binary_arithmetic_value_with_kernel(Operator::Mod, &age, &denominator, &kernel);
    assert_eq!(
        result.kind,
        Kind::Unknown,
        "rem.divisorSign's own divisorMayBeZero gate refuses a zero-admitting range: {result:?}"
    );
}

// --- `provable_raise` at a SET-SHAPED divisor ---

/// A divisor set that is ALWAYS zero — a degenerate seeded window
/// that has narrowed to nothing but `{0.0}` — provably raises, the
/// SET-shaped twin of the scalar `1 / 0` row `test_provable_raise_
/// zero_division` already pins. `divisor_is_provably_always_zero`
/// is the check: the set is a nonempty subset of `{0.0}`.
#[test]
fn test_provable_raise_fires_for_a_set_divisor_that_is_always_zero() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    // a degenerate Set that carries nothing but the value zero — the
    // shape a narrowed range can collapse to, distinct from the
    // ordinary Kind::Values `single_numeric_value` already reads;
    // built directly here to pin `divisor_is_provably_always_zero`
    // itself rather than lean on a derived Sub row to produce it
    let always_zero = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
    };
    environment.bind("difference", always_zero);
    let parsed = parse_expression("1 / difference").expect("test source must parse");
    let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
    let found = binop_provable_raise(&binop, &environment, &kernel);
    let Some((_, message)) = found else {
        panic!("a divisor set that is always zero must provably raise");
    };
    assert!(message.contains("ZeroDivisionError"), "{message}");
    assert!(message.contains("division by zero"), "{message}");
}

/// The negative row: a divisor set that only SOMETIMES admits zero
/// (`[0.0, 2.0]`) must NOT provably raise — most real executions
/// never hit the zero corner, so an unconditional raise finding here
/// would be a false positive. The VALUE question still declines
/// (pinned above); this only confirms the RAISE question stays
/// silent rather than overreaching. `binop_possible_raise` is this
/// window's own row (pinned below) — a DIFFERENT function, a
/// DIFFERENT claim, never this one's.
#[test]
fn test_provable_raise_stays_silent_for_a_set_divisor_that_only_sometimes_admits_zero() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    environment.bind("denominator", denominator);
    let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
    let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
    assert!(
        binop_provable_raise(&binop, &environment, &kernel).is_none(),
        "a sometimes-zero divisor window must not fire an unconditional raise"
    );
}

// --- `possible_raise` at a SET-SHAPED divisor ---

/// The escape row: a divisor set that only SOMETIMES admits zero
/// (`[0.0, 2.0]`) fires `binop_possible_raise`'s own sentence — a
/// DIFFERENT claim from `binop_provable_raise`'s unconditional
/// wording, and pinned against a DIFFERENT function: most real
/// executions never hit the zero corner, so an unconditional raise
/// finding would be a false positive, but the corner itself is a
/// real escape `split_divisor_transfer`'s own value determination
/// cannot speak to. Unguarded `1.0 / d` over this exact window still
/// DERIVES the split value: confirmed directly against
/// `binary_arithmetic_value_with_kernel` in the same test, so the
/// fire and the determination are pinned together rather than in
/// isolation — both stand; this row never withdraws the value, and
/// which sink decides how to combine the two is `check.rs`'s own
/// wiring, not this function's.
#[test]
fn test_possible_raise_fires_the_escape_sentence_for_a_set_divisor_that_only_sometimes_admits_zero() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    environment.bind("denominator", denominator.clone());
    let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
    let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
    let found = binop_possible_raise(&binop, &environment, &kernel);
    let Some((_, message)) = found else {
        panic!("a sometimes-zero divisor window must fire the escape sentence, not stay silent");
    };
    assert!(message.contains("admits 0"), "{message}");
    assert!(message.contains("ZeroDivisionError"), "{message}");
    assert!(
        !message.contains("this expression provably raises"),
        "the sometimes-zero row must not speak the always-zero rows' unconditional wording: {message}"
    );

    // the value side is not withdrawn: the same window still
    // determines through `split_divisor_transfer`, unaffected by
    // the new fire above
    let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
    let value = binary_arithmetic_value_with_kernel(Operator::Div, &one, &denominator, &kernel);
    assert_eq!(
        value.kind,
        Kind::Set,
        "the split value must still determine (never decline) alongside the fire: {value:?}"
    );
}

/// The always-zero row must not ALSO fire `possible_raise` — the two
/// functions' claims are disjoint, keyed by `divisor_is_provably_
/// always_zero` on one side and its negation on the other, so an
/// always-zero window belongs to `binop_provable_raise` alone.
#[test]
fn test_possible_raise_stays_silent_for_a_divisor_that_is_always_zero() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let always_zero = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![one_of(&[0.0])]), None, TrustProved, SetKindTag::None)
    };
    environment.bind("difference", always_zero);
    let parsed = parse_expression("1 / difference").expect("test source must parse");
    let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
    assert!(
        binop_possible_raise(&binop, &environment, &kernel).is_none(),
        "an always-zero divisor is binop_provable_raise's own claim, not this row's"
    );
}

/// The narrowing-interaction row: a divisor already narrowed AWAY
/// from zero (the shape `if divisor != 0:` leaves bound in
/// `environment` once the walk consumes that guard) must NOT fire —
/// `binop_possible_raise` reads `right` fresh off `environment` at
/// the ask (`evaluate_expression(&binop.right, environment, kernel)`
/// above), so a zero-excluding narrowed set is exactly what
/// `divisor_provably_excludes_zero` already reports `true` for, the
/// same gate the VALUE side reads in `transfer_over_sets` — the two
/// never disagree about which windows still admit zero.
#[test]
fn test_possible_raise_stays_silent_for_a_divisor_narrowed_away_from_zero() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    // the narrowed shape a consumed `if divisor != 0:` (or an
    // equivalent guard) leaves behind: the zero-excluding POSITIVE
    // half of the same window the fire test above admits zero over
    let narrowed = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![
                refined_sets::refinement_forms::above(0.0),
                refined_sets::refinement_forms::at_most(2.0),
            ]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    environment.bind("denominator", narrowed);
    let parsed = parse_expression("1.0 / denominator").expect("test source must parse");
    let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
    assert!(
        binop_possible_raise(&binop, &environment, &kernel).is_none(),
        "a divisor narrowed away from zero must not fire — the ask reads the narrowed set, not the pre-guard one"
    );
}

/// `//` and `%` fire the SAME escape sentence `/` does over a
/// sometimes-zero divisor: CPython raises `ZeroDivisionError` on the
/// zero arm of the window for all three operators alike
/// (expressions.rst, "Binary arithmetic operations"). `//`/`%` have
/// no zero-excluded split (`split_divisor_transfer` is `/`'s own
/// fix), so their VALUE question keeps declining outright over this
/// same window — only the fire is new here, the value side
/// unchanged.
#[test]
fn test_possible_raise_fires_for_floordiv_and_mod_over_a_sometimes_zero_divisor() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let denominator = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(0.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    for source in ["1.0 // denominator", "1.0 % denominator"] {
        environment.bind("denominator", denominator.clone());
        let parsed = parse_expression(source).expect("test source must parse");
        let Expr::BinOp(binop) = parsed.into_expr() else { panic!("expected a BinOp") };
        let found = binop_possible_raise(&binop, &environment, &kernel);
        let Some((_, message)) = found else {
            panic!("{source}: a sometimes-zero divisor window must fire the escape sentence, not stay silent");
        };
        assert!(message.contains("admits 0"), "{source}: {message}");
        assert!(message.contains("ZeroDivisionError"), "{source}: {message}");

        // the value side is unchanged: `//`/`%` still decline outright
        // over this same window, because no split runs for them
        let one = known_values(vec![1.0], PrimitiveKind::Float, TrustProved);
        let op = if source.contains("//") { Operator::FloorDiv } else { Operator::Mod };
        let value = binary_arithmetic_value_with_kernel(op, &one, &denominator, &kernel);
        assert_eq!(
            value.kind,
            Kind::Unknown,
            "{source}: the value question must keep declining outright — no split runs for `//`/`%`: {value:?}"
        );
    }
}

// --- `possible_raise` for the domain-limited math family (straddling) ---

/// `math.log(x)` where `x`'s window is `[-1.0, 1.0]` — STRADDLES the
/// raise domain (`x <= 0`): the negative-through-zero half raises,
/// the positive half `(0.0, 1.0]` still returns a value. Fires the
/// SAME "math domain error" sentence `call_provable_raise`'s
/// all-or-nothing row speaks, but through `possible_raise` — the
/// window is not ENTIRELY inside the raise domain, so `call_
/// provable_raise`'s own row (checked directly below) must stay
/// silent, exactly the disjointness `test_possible_raise_stays_
/// silent_for_a_divisor_that_is_always_zero` pins for the division
/// row. The served half's value stands alongside the fire, read
/// through `evaluate_attribute_call`'s own wiring (the value side of
/// `math_call_result`'s decline, not `possible_raise` itself).
#[test]
fn test_possible_raise_fires_for_a_log_window_that_straddles_the_raise_domain() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let straddling = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(-1.0), refined_sets::refinement_forms::at_most(1.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    environment.bind("x", straddling);
    let parsed = parse_expression("math.log(x)").expect("test source must parse");
    let expr = parsed.into_expr();

    let found = possible_raise(&expr, &environment, &kernel);
    let Some((_, message)) = found else {
        panic!("a straddling log window must fire the possible-raise sentence, not stay silent");
    };
    assert!(message.contains("ValueError"), "{message}");
    assert!(message.contains("math domain error"), "{message}");

    // the ALL-OR-NOTHING row must stay silent for the same window —
    // the two claims are disjoint, keyed by
    // DomainRaiseClassification::EntirelyRaises vs ::Straddles
    assert!(
        provable_raise(&expr, &environment, &kernel).is_none(),
        "a straddling window is possible_raise's own claim, not provable_raise's"
    );

    // the served half still determines a value, read through the
    // ordinary evaluate_expression path (evaluate_attribute_call's
    // own decline-then-served-half wiring, math_models.rs)
    let value = evaluate_expression(&expr, &environment, &kernel);
    assert_eq!(
        value.kind,
        Kind::Set,
        "the served half (0.0, 1.0] must still determine a window, alongside the fire: {value:?}"
    );
    assert_eq!(value.kind_tag, Some(PrimitiveKind::Float));
}

/// An ENTIRELY-served log window (`[1.0, 2.0]`, wholly `x > 0`) must
/// NOT fire `possible_raise` — the disjointness twin of the fire
/// test above, mirroring `test_possible_raise_stays_silent_for_a_
/// divisor_narrowed_away_from_zero`'s own shape for division.
#[test]
fn test_possible_raise_stays_silent_for_a_log_window_entirely_served() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let served = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(1.0), refined_sets::refinement_forms::at_most(2.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    environment.bind("x", served);
    let parsed = parse_expression("math.log(x)").expect("test source must parse");
    let expr = parsed.into_expr();
    assert!(
        possible_raise(&expr, &environment, &kernel).is_none(),
        "an entirely-served window must not fire the straddling row"
    );
}

/// An ENTIRELY-raising log window (`[-2.0, -1.0]`, wholly `x <= 0`)
/// must NOT fire `possible_raise` either — that claim belongs to
/// `provable_raise`'s own all-or-nothing row alone, the same
/// disjointness `test_possible_raise_stays_silent_for_a_divisor_
/// that_is_always_zero` pins for the always-zero divisor.
#[test]
fn test_possible_raise_stays_silent_for_a_log_window_entirely_raising() {
    let Some(kernel) = loaded_kernel() else { return };
    let mut environment = empty_environment();
    let raising = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(
            make_refined_set(vec![at_least(-2.0), refined_sets::refinement_forms::at_most(-1.0)]),
            None,
            TrustProved,
            SetKindTag::None,
        )
    };
    environment.bind("x", raising);
    let parsed = parse_expression("math.log(x)").expect("test source must parse");
    let expr = parsed.into_expr();
    assert!(
        possible_raise(&expr, &environment, &kernel).is_none(),
        "an entirely-raising window is provable_raise's own claim, not this row's"
    );
    assert!(
        provable_raise(&expr, &environment, &kernel).is_some(),
        "an entirely-raising window must fire provable_raise's own all-or-nothing row"
    );
}
