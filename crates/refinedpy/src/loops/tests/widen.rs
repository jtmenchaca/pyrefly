//! Widening: `stabilized_join`'s own fixed-point/widening behavior
//! over scalar and list-count accumulators, plus the stepwise
//! diagnostic chain for the join/transfer path a widened join relies
//! on.

use super::*;

/// `stabilized_join`'s widening, pinned at its own layer: a second
/// pass that binds a DIFFERENT exact value than the first proves the
/// name never reached a fixed point, so the join rebinds it to
/// unknown and names it in `widened` — the list `check.rs`'s
/// `walk_loop` turns into the body's fixed-point blocker.
#[test]
fn stabilized_join_names_the_name_that_never_reaches_a_fixed_point() {
    let Some(kernel) = loaded_kernel() else { return };
    let for_stmt = parsed_loop("for s in samples:\n    total = total * 2.0\n");
    let Stmt::For(for_stmt) = for_stmt else {
        panic!("fixture is a for statement");
    };
    let environment = environment_with(&[("total", 1.0)]);
    let one_pass = environment_with(&[("total", 1.0)]);
    let declared = no_declared();
    let mut judge_context = JudgeContext {
        declared: &declared,
        newly_declared: HashMap::new(),
        already_fired: std::collections::HashSet::new(),
        fires: Vec::new(),
    };
    let element = known_number(0.0);
    let (result, widened) = stabilized_join(
        &environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        &kernel,
        &mut judge_context,
    )
    .expect("both judged passes complete for an exact rebind");
    assert_eq!(widened, vec!["total".to_owned()], "the non-stabilizing name is named");
    assert_eq!(
        result.read("total").map(|v| v.kind),
        Some(Kind::Unknown),
        "the unstable name holds no claim past the loop"
    );
}

/// `stabilized_join`'s widening, pinned at its own layer: a scalar
/// accumulator whose containment check fails because the second
/// pass's hull GREW past the first join's own hull — `total += x`
/// over `x` bound to `[0, 200] ∩ ℤ`, starting `environment` AND
/// `one_pass` at the SAME `total = [0, 200] ∩ ℤ` (so `joined` is
/// exactly that set, via `same_known`'s fast path — no dependence
/// on how `Environment::join`'s own Values/Set collapse would
/// otherwise read a `{0}`/`[0,200]` pair). Running `total += x`
/// once more from `[0, 200] ∩ ℤ` grows the hull to `[0, 400] ∩ ℤ`,
/// which `stable_by_containment` correctly refuses (400 is not
/// covered by `[0, 200]`) — this is the shape `stabilized_join`'s
/// widening exists for: the upper edge GREW, so `W` drops it
/// entirely and keeps the stable lower edge and the integer mark,
/// verified sound by one further body step staying inside `W`.
#[test]
fn stabilized_join_widens_a_growing_scalar_accumulator_to_the_ray() {
    let Some(kernel) = loaded_kernel() else { return };
    let for_stmt = parsed_loop("for x in xs:\n    total += x\n");
    let Stmt::For(for_stmt) = for_stmt else {
        panic!("fixture is a for statement");
    };
    let bounded_total = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]), None, TrustProved, SetKindTag::None)
    };
    let locally_bound: HashSet<String> = HashSet::from(["total".to_owned(), "x".to_owned()]);
    let mut environment = Environment::new(locally_bound.clone());
    environment.bind("total", bounded_total.clone());
    let mut one_pass = Environment::new(locally_bound);
    one_pass.bind("total", bounded_total);
    let declared = no_declared();
    let mut judge_context = JudgeContext {
        declared: &declared,
        newly_declared: HashMap::new(),
        already_fired: std::collections::HashSet::new(),
        fires: Vec::new(),
    };
    let element = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]), None, TrustProved, SetKindTag::None)
    };
    let (result, widened) = stabilized_join(
        &environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        &kernel,
        &mut judge_context,
    )
    .expect("the widened candidate's own verification pass completes");
    assert_eq!(widened, Vec::<String>::new(), "the widened hull is trusted — the name is never havoced");
    let total = result.read("total").expect("total stays bound");
    assert_eq!(total.kind, Kind::Set, "the accumulator stays a determined Set, never unknown()");
    let hull = ask_bounds_public(&total.set).expect("the widened hull is a plain scalar window");
    assert!(!hull.empty);
    let (lo, hi, is_integer) = hull_window(&hull.hull).expect("the widened set is a plain AtLeast/AtMost/Integer window");
    assert_eq!(lo, Some(0.0), "the stable lower edge survives widening");
    assert_eq!(hi, None, "the growing upper edge is dropped, not merely raised");
    assert!(is_integer, "both hulls were integral, so the widened window stays integral");
}

/// `stabilized_join`'s widening over the OTHER shape it reads: a
/// repetition-shaped (bounded-list) accumulator whose COUNT window
/// grew rather than a scalar hull — `out.append(...)` inside an
/// `if`/`else`, mirroring `if_else_over_a_set_bound_loop_element_
/// joins_both_narrowed_arms`'s own fixture but calling
/// `stabilized_join` directly so the widened window's own shape can
/// be asserted precisely, not just "stays Kind::Set". `environment`
/// and `one_pass` both start `out` at the SAME repetition window
/// (`[0, 1]` copies of the element, `same_known`'s fast path again
/// keeping `joined` exactly that shape); one further `append` grows
/// the count to `[1, 2]`, which is not contained in `[0, 1]` — the
/// widening drops the grown upper edge, keeping `lo = 0` and
/// `hi = None` (an unbounded count, the star shape).
#[test]
fn stabilized_join_widens_a_growing_list_accumulator_count() {
    let Some(kernel) = loaded_kernel() else { return };
    let for_stmt = parsed_loop("for x in xs:\n    if 0 <= x <= 149:\n        out.append(x + 1)\n    else:\n        out.append(0)\n");
    let Stmt::For(for_stmt) = for_stmt else {
        panic!("fixture is a for statement");
    };
    let element_set = make_refined_set(vec![integer_form(), at_least(0.0), at_most(200.0)]);
    let out_window = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(repetition(element_set.clone(), 0, Some(1)), None, TrustProved, SetKindTag::None)
    };
    let locally_bound: HashSet<String> = HashSet::from(["out".to_owned(), "x".to_owned()]);
    let mut environment = Environment::new(locally_bound.clone());
    environment.bind("out", out_window.clone());
    let mut one_pass = Environment::new(locally_bound);
    one_pass.bind("out", out_window);
    let declared = no_declared();
    let mut judge_context = JudgeContext {
        declared: &declared,
        newly_declared: HashMap::new(),
        already_fired: std::collections::HashSet::new(),
        fires: Vec::new(),
    };
    let element = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(element_set, None, TrustProved, SetKindTag::None)
    };
    let (result, widened) = stabilized_join(
        &environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        &kernel,
        &mut judge_context,
    )
    .expect("the widened candidate's own verification pass completes");
    assert_eq!(widened, Vec::<String>::new(), "the widened count window is trusted — the name is never havoced");
    let out = result.read("out").expect("out stays bound");
    assert_eq!(out.kind, Kind::Set, "the accumulator stays a determined Set, never unknown()");
    let window = as_repetition(&out.set).expect("the widened value stays a repetition window");
    assert_eq!(window.lo, 0, "the stable lower edge of the count survives widening");
    assert_eq!(window.hi, None, "the growing upper edge of the count is dropped — an unbounded count");
}

/// `stabilized_join`'s widening does NOT paper over a genuinely
/// unstable name: a body that unconditionally REBINDS `total` to a
/// fixed literal every pass (`total = 1000`, never reading `total`
/// or `x` at all) starts `environment`/`one_pass` at the SAME
/// `Kind::Set` window (so `joined` is that Set exactly, via
/// `same_known`'s fast path, same construction as the two widening
/// pins above), but the second pass's own literal assign rebinds
/// `total` to `Kind::Values [1000.0]` — a DIFFERENT Kind entirely,
/// oscillating between disjoint value kinds the way the module's
/// own doc names as the genuinely unstable shape. This pair never
/// reaches `widened_set_candidate` at all (`is_set_pair` requires
/// BOTH sides `Kind::Set`, which the second value no longer is) —
/// the plain rejoin/containment path this widening sits beside
/// still havocs it, exactly as before this widening existed.
#[test]
fn stabilized_join_still_havocs_a_genuinely_unstable_set_pair() {
    let Some(kernel) = loaded_kernel() else { return };
    let for_stmt = parsed_loop("for x in xs:\n    total = 1000\n");
    let Stmt::For(for_stmt) = for_stmt else {
        panic!("fixture is a for statement");
    };
    let bounded_total = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(make_refined_set(vec![integer_form(), at_least(0.0), at_most(10.0)]), None, TrustProved, SetKindTag::None)
    };
    let locally_bound: HashSet<String> = HashSet::from(["total".to_owned(), "x".to_owned()]);
    let mut environment = Environment::new(locally_bound.clone());
    environment.bind("total", bounded_total.clone());
    let mut one_pass = Environment::new(locally_bound);
    one_pass.bind("total", bounded_total);
    let declared = no_declared();
    let mut judge_context = JudgeContext {
        declared: &declared,
        newly_declared: HashMap::new(),
        already_fired: std::collections::HashSet::new(),
        fires: Vec::new(),
    };
    let element = known_number_sorted(0.0, PrimitiveKind::Integer);
    let (result, widened) = stabilized_join(
        &environment,
        &one_pass,
        &for_stmt.body,
        for_stmt.target.as_ref(),
        &element,
        &kernel,
        &mut judge_context,
    )
    .expect("both judged passes complete");
    assert_eq!(widened, vec!["total".to_owned()], "a genuinely unstable pair is still named, never silently widened");
    assert_eq!(
        result.read("total").map(|v| v.kind),
        Some(Kind::Unknown),
        "the unstable name holds no claim past the loop, exactly as before this widening existed"
    );
}

// --- STEPWISE DIAGNOSTIC CHAIN for showcase.py's own `total = total +
// amount` shape (invoice_total/refund_everything) — the two pins at
// check.rs's own test module (`a_plain_rebind_accumulation_over_a_
// float_list_parameter_walks_the_loop` and its subtracting twin) still
// fail with the coarser "a for statement is not yet walked" blocker
// after the join's own numeric-fallback union was fixed to thread a
// shared `kind_tag` (lattice_operations.rs) — these three tests
// measure each link of the chain in isolation rather than inferring
// which one still declines from the pins' own outer failure.

/// STEP 1: `join_known` directly, on the EXACT pair `stabilized_join`
/// builds for `total` after the loop's first pass — the pre-loop
/// binding (`total = 0.0`, `Kind::Values`, Float-tagged) against a
/// pass-one Set (`total + amount` — the same `[0, +inf)`-shaped
/// Float-tagged window `transfer_over_sets`'s `TransferAnswerKind::Set`
/// row answers for a non-negative Float set added to `{0.0}`, built by
/// hand here rather than round-tripped through the kernel, matching
/// this test's own narrow question: what the JOIN does with this
/// shape, not what the transfer computes). Asserts the joined value's
/// kind, kind_tag, and set forms — the fixed `shared_kind_tag` should
/// carry `Some(Float)` through onto the union.
#[test]
fn join_known_of_preloop_total_and_pass_one_set_keeps_the_float_tag() {
    let preloop_total = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
    let pass_one_total = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    };
    let joined = refined_domain::lattice_operations::join_known(preloop_total, pass_one_total);
    assert_eq!(joined.kind, Kind::Set, "a Values/Set numeric pair joins to a Set: {joined:?}");
    assert_eq!(
        joined.kind_tag,
        Some(PrimitiveKind::Float),
        "the join must thread the shared Float tag onto the union rather than drop it: {joined:?}"
    );
    assert_eq!(joined.set_kind_tag, SetKindTag::None, "a plain numeric set carries no worn tag: {joined:?}");
}

/// STEP 2: feeds STEP 1's joined value as the LEFT operand of the same
/// `total + amount` binary op the loop's SECOND pass evaluates,
/// through `binary_arithmetic_value_with_kernel` — the exact function
/// `evaluate_binop` calls (expressions.rs) — asking what the kernel's
/// own `transfer_over_sets` path does with a UNION-shaped operand
/// (`union({0.0}, [0, +inf))`) now that it carries a tag. Two
/// possibilities distinguish the remaining failing link: if this
/// answers `Kind::Unknown`, the tag fix alone was not enough — the
/// kernel's own `transfer` closure declines a Union-form operand
/// outright (unfolded), and the fix is a fold at the ask site; if this
/// answers `Kind::Set`/`Kind::Values`, the join/transfer chain itself
/// is clean and the remaining defect is downstream (`run_assign_once`/
/// `stabilized_join`'s own comparison, or `bind_checked`).
#[test]
fn transfer_over_the_joined_union_set_plus_amount_measures_the_kernel_answer() {
    let Some(kernel) = loaded_kernel() else { return };
    let preloop_total = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
    let pass_one_total = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    };
    let joined_total = refined_domain::lattice_operations::join_known(preloop_total, pass_one_total);
    let amount = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    };
    let result = crate::expressions::binary_arithmetic_value_with_kernel(Operator::Add, &joined_total, &amount, &kernel);
    eprintln!("STEP 2 measured answer: {result:?}");
    assert_ne!(
        result.kind,
        Kind::Unknown,
        "if this fails, the kernel's transfer declines the joined UNION operand outright — the \
        remaining fix is a fold (fold_ray_forms) of the joined set before the ask, either at \
        transfer_over_sets' own call site or at join_known's union-building arm: {result:?}"
    );
}

/// STEP 3: the same union-shaped joined value, folded through
/// `refined_sets::refinement_forms::fold_ray_forms` BEFORE the ask —
/// the Rust twin of the Go adapter's `FoldRayForms`/`CanonicalScalarForms`
/// hygiene (refinement_forms.go's own doc: "posing the folded question
/// saves the kernel the redundant forms... while asking for the same
/// set"). `{0.0} ∪ [0, +inf)` folds to the single ray `[0, +inf)` —
/// `at_least(0.0)` dominates the singleton, so the fold both simplifies
/// AND stays semantically identical. If STEP 2 shows the kernel
/// declining the unfolded union but this step's folded ask determines,
/// the fix site is confirmed: fold the joined set's forms before
/// `transfer_over_sets` asks the kernel (or fold at `join_known`'s own
/// union-building arms directly, so every caller of `join_known`
/// inherits the same hygiene without a second call site).
#[test]
fn transfer_over_the_folded_joined_set_plus_amount_measures_the_kernel_answer() {
    let Some(kernel) = loaded_kernel() else { return };
    let preloop_total = known_values(vec![0.0], PrimitiveKind::Float, TrustProved);
    let pass_one_total = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    };
    let joined_total = refined_domain::lattice_operations::join_known(preloop_total, pass_one_total);
    let folded_forms = refined_sets::refinement_forms::fold_ray_forms(&joined_total.set.forms);
    let folded_total = AbstractValue {
        set: make_refined_set(folded_forms),
        ..joined_total
    };
    let amount = AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(make_refined_set(vec![at_least(0.0)]), None, TrustProved, SetKindTag::None)
    };
    let result = crate::expressions::binary_arithmetic_value_with_kernel(Operator::Add, &folded_total, &amount, &kernel);
    eprintln!("STEP 3 measured answer (folded operand): {result:?}");
    assert_ne!(
        result.kind,
        Kind::Unknown,
        "the folded ray form still declines — the remaining defect is not the union's redundant \
        forms: {result:?}"
    );
}
