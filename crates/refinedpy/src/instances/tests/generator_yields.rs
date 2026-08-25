use super::*;

// --- generator_yields: the stream() for-loop shape ---

/// `async def stream(): for value in (10, 20, 30): yield value` —
/// a-statements.py:547-549's own shape: a generator whose only
/// statement is a `for` loop over a literal tuple, yielding the
/// loop target unmodified. `generator_yields` must answer all three
/// yields, in order.
#[test]
fn generator_yields_reads_the_stream_for_loop_shape_in_order() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "async def stream():\n",
        "    for value in (10, 20, 30):\n",
        "        yield value\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the stream() for-loop shape must decide");
    assert_eq!(yields, vec![integer_value(10.0), integer_value(20.0), integer_value(30.0)]);
}

/// The same shape, but the yield expression TRANSFORMS the target
/// (`yield value + 100`) — the per-iterate binding must be visible
/// to the yield expression, not just a bare pass-through.
#[test]
fn generator_yields_evaluates_the_yield_expression_per_iterate() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def stream():\n",
        "    for value in [10, 20]:\n",
        "        yield value + 100\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the transformed-yield shape must decide");
    assert_eq!(yields, vec![integer_value(110.0), integer_value(120.0)]);
}

/// Straight-line top-level yields merge with the for-loop's own
/// yields, in source order — the addendum's own "merged with any
/// top-level yields in source order."
#[test]
fn generator_yields_merges_straight_line_and_for_loop_yields_in_source_order() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def mixed():\n",
        "    yield 1\n",
        "    for value in (2, 3):\n",
        "        yield value\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the mixed shape must decide");
    assert_eq!(yields, vec![integer_value(1.0), integer_value(2.0), integer_value(3.0)]);
}

/// A `for` loop whose iterable is NOT one of the two literal shapes
/// (a bare name, here) declines the whole body — never a partial
/// list.
#[test]
fn generator_yields_declines_a_for_loop_over_a_non_literal_iterable() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def stream(values):\n",
        "    for value in values:\n",
        "        yield value\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    assert!(generator_yields(&def, &[unknown()], None, &kernel, 0).is_none());
}

// --- generator_yields: a conditional yield joins with its continuation ---

/// q-decline-names.py's own `age_generator` shape: `if bool([]): yield
/// 40` with NO `else`, followed by an unconditional `yield 41`. Neither
/// branch of the `if` is provably taken, so the position `next()` would
/// read first is the JOIN of both outcomes — `{40, 41}` — never a
/// decline: this is exactly the sound over-approximation
/// `yields_of_body`'s own doc states for a conditional yield followed
/// by an unconditional one.
#[test]
fn generator_yields_joins_a_conditional_yield_with_its_continuation() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def age_generator():\n",
        "    if bool([]):\n",
        "        yield 40\n",
        "    yield 41\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    let yields = generator_yields(&def, &[], None, &kernel, 0).expect("the conditional-then-unconditional shape must join");
    let [joined] = yields.as_slice() else {
        panic!("want exactly one joined position, got {}", yields.len());
    };
    assert_eq!(joined.kind, Kind::Values);
    let mut values = joined.values.clone();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![40.0, 41.0]);
}

/// A conditional yield with NOTHING unconditional after it has no
/// continuation to join against — the real generator sometimes
/// produces nothing at all past this point, a length-zero-or-one
/// shape `yields_of_body`'s own `Vec` return cannot spell. Still a
/// genuine decline, distinct from the joined case above.
#[test]
fn generator_yields_declines_a_conditional_yield_with_no_continuation() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def maybe_yields():\n",
        "    if bool([]):\n",
        "        yield 40\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    assert!(
        generator_yields(&def, &[], None, &kernel, 0).is_none(),
        "a conditional yield with no unconditional yield after it has no continuation to join against"
    );
}

// --- generator_yields: a leading docstring is skipped ---

/// A generator whose body opens with a docstring, then a plain
/// straight-line `yield`, must summarize exactly as it would with no
/// docstring at all — the docstring states no readable effect.
#[test]
fn generator_yields_skips_a_leading_docstring_before_a_straight_line_yield() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def documented():\n",
        "    \"\"\"a docstring, not a yield\"\"\"\n",
        "    yield 40\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    let yields = generator_yields(&def, &[], None, &kernel, 0)
        .expect("a leading docstring must not decline the body");
    assert_eq!(yields, vec![integer_value(40.0)]);
}

/// The same docstring-skip over the `for`-loop shape (shape 2) —
/// the docstring sits before the loop, not inside it.
#[test]
fn generator_yields_skips_a_leading_docstring_before_a_for_loop_yield() {
    let Some(kernel) = loaded_kernel() else { return };
    let module = parsed(concat!(
        "def documented_stream():\n",
        "    \"\"\"a docstring, not a yield\"\"\"\n",
        "    for value in (10, 20):\n",
        "        yield value\n",
    ));
    let def = module.body.into_iter().next().expect("one top-level def").function_def_stmt().expect("is a def");
    let yields = generator_yields(&def, &[], None, &kernel, 0)
        .expect("a leading docstring must not decline the for-loop shape");
    assert_eq!(yields, vec![integer_value(10.0), integer_value(20.0)]);
}
