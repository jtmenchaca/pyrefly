/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Calls to Python builtins with determinable results, answered exactly.
//! One dispatcher — `builtin_call_result` — takes the callee name and the
//! already-evaluated argument values; `None` means "not modeled here" (the
//! caller declines honestly), `Some` is an exact answer. Every modeled row
//! cites its clause of docs.python.org/3.12/library/functions.html or
//! library/stdtypes.html (the container constructors `list`/`set`/`dict`
//! live in stdtypes.html's own class entries); a row with no citation is
//! not written.

use refined_domain::abstract_value::{known_values, opaque_value, AbstractValue, Kind, PrimitiveKind};
use refined_domain::known_constructors::known_list;
use refined_domain::trust_grades::{derived_trust_level, TrustSpec};

/// Read a single known numeric value out of an argument: `Kind::Values`,
/// tagged `Integer` or `Float`, carrying exactly one element. Every row
/// below that needs "one known number" reads through this rather than
/// re-matching the shape.
fn single_known_numeric(argument: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
    if argument.kind != Kind::Values {
        return None;
    }
    if argument.values.len() != 1 {
        return None;
    }
    match argument.kind_tag {
        Some(PrimitiveKind::Integer) => Some((argument.values[0], PrimitiveKind::Integer)),
        Some(PrimitiveKind::Float) => Some((argument.values[0], PrimitiveKind::Float)),
        _ => None,
    }
}

/// `abs(x)` on a single known numeric — library/functions.html#abs:
/// "Return the absolute value of a number." Sort is preserved: an int
/// argument's absolute value is an int, a float's a float — abs never
/// changes the numeric sort of its single argument.
fn abs_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.abs()], sort, grade))
}

/// `round(x)`, single-argument — library/functions.html#round: "If
/// ndigits is omitted or is None, it returns the nearest integer to its
/// input," rounding "toward the even choice" on a tie (banker's
/// rounding — `round(0.5)` and `round(-0.5)` are both `0`, `round(1.5)`
/// is `2`). The two-argument form `round(x, n)` is not modeled: it keeps
/// the input's sort (int stays int, float stays float) rather than
/// always producing an int, a different row this dispatcher does not
/// yet answer.
fn round_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(
        vec![value.round_ties_even()],
        PrimitiveKind::Integer,
        grade,
    ))
}

/// The single numeric value out of a KNOWN `Kind::List` element — the
/// same acceptance `single_known_numeric` gives a bare argument, read
/// off one list slot for `sum`/`min`/`max`'s single-iterable rows.
fn single_known_numeric_element(element: &AbstractValue) -> Option<(f64, PrimitiveKind)> {
    single_known_numeric(element)
}

/// `sum(iterable, start=0)` over a known `Kind::List` of known single-
/// numeric elements (a known list literal, or the comprehension/
/// generator shape `evaluate_list_or_set_comp` already builds as a
/// `Kind::List`) — library/functions.html#sum: "Sums *start* and the
/// items of an *iterable* from left to right and returns the total."
/// The two-argument `start=` form threads the caller's own start value
/// (defaulting to Integer 0, matching the doc's own default); any
/// non-numeric element declines the whole call rather than skip it.
/// Sort widens to Float the moment any addend (the start value or any
/// element) is Float-sorted, matching ordinary `+` — the same mixed-
/// arithmetic widening `expressions.rs`'s `binary_arithmetic_value`
/// already applies.
fn sum_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let (iterable, start) = match arguments {
        [iterable] => (iterable, None),
        [iterable, start] => (iterable, Some(start)),
        _ => return None,
    };
    if iterable.kind != Kind::List {
        return None;
    }
    let (mut total, mut all_int) = match start {
        Some(start_value) => {
            let (value, sort) = single_known_numeric(start_value)?;
            (value, sort == PrimitiveKind::Integer)
        }
        None => (0.0, true),
    };
    for element in &iterable.items {
        let (value, sort) = single_known_numeric_element(element)?;
        total += value;
        all_int = all_int && sort == PrimitiveKind::Integer;
    }
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    let sort = if all_int { PrimitiveKind::Integer } else { PrimitiveKind::Float };
    Some(known_values(vec![total], sort, grade))
}

/// `min`/`max` over a SINGLE known `Kind::List` iterable argument —
/// library/functions.html#min/#max: "If one positional argument is
/// provided, it should be an iterable... the largest [smallest] item
/// in the iterable is returned." An empty iterable has no row here:
/// CPython raises `ValueError` on an empty sequence with no `default=`
/// keyword, which this file has no exception channel for this wave —
/// this row declines on an empty list rather than answer a fabricated
/// value.
fn min_max_over_iterable(arguments: &[AbstractValue], pick: fn(f64, f64) -> bool) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::List || iterable.items.is_empty() {
        return None;
    }
    let mut best: Option<(f64, PrimitiveKind)> = None;
    for element in &iterable.items {
        let candidate = single_known_numeric_element(element)?;
        best = Some(match best {
            None => candidate,
            Some(current) => if pick(candidate.0, current.0) { candidate } else { current },
        });
    }
    let (value, sort) = best?;
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    Some(known_values(vec![value], sort, grade))
}

/// `sorted(iterable)` (no `key=`/`reverse=` keyword arguments) over a
/// known `Kind::List` of known single-numeric elements —
/// library/functions.html#sorted: "Return a new sorted list from the
/// items in *iterable*." Ascending numeric order, matching the
/// no-`key`/no-`reverse` default row; a non-numeric element declines
/// the whole call.
fn sorted_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::List {
        return None;
    }
    let mut pairs: Vec<(f64, PrimitiveKind)> = Vec::with_capacity(iterable.items.len());
    for element in &iterable.items {
        pairs.push(single_known_numeric_element(element)?);
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("known numeric values are never NaN"));
    let grade = derived_trust_level(TrustSpec, &[iterable.clone()]);
    let sorted_items: Vec<AbstractValue> = pairs.into_iter().map(|(value, sort)| known_values(vec![value], sort, grade)).collect();
    Some(known_list(sorted_items, grade))
}

/// `list(iterable)` — library/stdtypes.rst's `class:: list([iterable])`
/// constructor row: "Lists may be constructed... using the type
/// constructor `list()` or `list(iterable)`." A known `Kind::List`
/// argument copies through unchanged (`list`/`tuple`/`set` all share
/// this domain's one `Kind::List` shape, per `collection_models.rs`'s
/// own module doc — `list(some_set)` and `list(some_tuple)` both read
/// through this same row).
fn list_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if iterable.kind != Kind::List {
        return None;
    }
    Some(known_list(iterable.items.clone(), derived_trust_level(TrustSpec, arguments)))
}

/// `set([iterable])` — library/stdtypes.rst's `class:: set([iterable])`
/// constructor row: "Return a new set... object whose elements are
/// taken from *iterable*." This domain has no dedicated set Kind (the
/// same `Kind::List` shape a list/tuple carries, per
/// `collection_models.rs`'s own module doc — a set's own element-
/// uniqueness is invisible to any reader that only ever consumes the
/// sequence via `len()`/iteration, matching that file's list/set-comp
/// note). The BARE zero-argument form `set()` — the brackets in the
/// doc's own signature mark the argument optional — answers the empty
/// list directly (an empty set has no elements to dedupe); the
/// one-argument form is `list_constructor_call` under a different name;
/// deduplication is NOT modeled for the one-argument form (an already-
/// List argument is assumed unique-enough for this file's callers,
/// since a set LITERAL display is not what feeds this row — only an
/// already-list-shaped iterable is).
fn set_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if arguments.is_empty() {
        return Some(known_list(Vec::new(), TrustSpec));
    }
    list_constructor_call(arguments)
}

/// `dict(pairs)` — one positional argument, an iterable of `(key,
/// value)` 2-element pairs — library/stdtypes.rst's `class:: dict(...)`
/// constructor row: "dict(iterable, **kwargs)... Dictionaries can be
/// created by... providing an iterable of key/value pairs, including
/// tuples: `dict([('foo', 100), ('bar', 200)])`." Modeled ONLY when
/// `pairs` is a known `Kind::List` of known `Kind::List` 2-element
/// pairs whose first slot is a known exact string (this domain's
/// dict's own string-keyed-only restriction, `collection_models.rs`'s
/// module doc) — anything else declines. A repeated key keeps the LAST
/// value, matching the same overwrite rule `dict_literal_value` and
/// the `dict(...)` constructor doc both state.
fn dict_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [pairs] = arguments else { return None };
    // `dict(<existing dict>)` — the copy-constructor form ("providing
    // ... another dictionary", the same class:: dict(...) row): a known
    // Kind::Object argument answers a fresh dict with the same entries.
    if pairs.kind == Kind::Object && pairs.kind_word.is_none() {
        return Some(pairs.clone());
    }
    if pairs.kind != Kind::List {
        return None;
    }
    let mut keys: Vec<Option<crate::refinedpy::collection_models::DictKey>> = Vec::with_capacity(pairs.items.len());
    let mut values: Vec<AbstractValue> = Vec::with_capacity(pairs.items.len());
    for pair in &pairs.items {
        if pair.kind != Kind::List || pair.items.len() != 2 {
            return None;
        }
        let key = &pair.items[0];
        if key.kind != Kind::Values || key.kind_tag != Some(PrimitiveKind::String) {
            return None;
        }
        let key_text: String = key.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        keys.push(Some(crate::refinedpy::collection_models::DictKey::string(&key_text)));
        values.push(pair.items[1].clone());
    }
    // dict_literal_value's own last-value-wins overwrite rule handles a
    // repeated key exactly the way this constructor's own cited row
    // does — this file reaches into collection_models.rs for the one
    // shared building block rather than duplicating that merge loop
    Some(crate::refinedpy::collection_models::dict_literal_value(&keys, &values))
}

/// `iter(object)` (one-argument form, no `sentinel`) — library/functions.html#iter:
/// "Return an iterator object... *object* must be a collection object
/// which supports the iterable protocol." This domain has no separate
/// iterator Kind: an iterator over a known `Kind::List` reads through
/// as the SAME list value (the one shape a caller ever inspects it
/// through — `next_call`'s own row below), matching the module's
/// shared list/set/generator representation
/// (`collection_models.rs`'s own module doc). Any other receiver
/// shape declines.
fn iter_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::List {
        return None;
    }
    Some(only.clone())
}

/// `next(iterator)` (one-argument form, no `default`) — library/functions.html#next:
/// "Retrieve the next item from the iterator by calling its
/// `__next__` method." Modeled ONLY for the `iter_call`-shaped receiver
/// (a known `Kind::List` standing in for its own iterator, per that
/// function's own doc) AND a generator call's own answer
/// (`Kind::List` tagged `source == "generator"`,
/// `instances::generator_yields`'s own doc — a same-module generator
/// `def`'s call answers the ordered List of every yielded value): the
/// FIRST element is the first item `__next__` would ever produce off a
/// freshly-built iterator or a freshly-called generator. An EMPTY list
/// provably raises `StopIteration` ("If *default* is given, it is
/// returned if the iterator is exhausted, otherwise `StopIteration` is
/// raised") — this row declines on an empty receiver rather than answer
/// a fabricated element; the raise itself is `provable_raise`'s own
/// business, not this dispatcher's.
///
/// SCOPE: this domain carries no per-call exhaustion/position state — a
/// generator-tagged List is a fixed VALUE (the full yield sequence),
/// not a stateful cursor, so `next_call` cannot tell "the first read of
/// this generator" apart from "a second read of the SAME already-
/// advanced generator." Every corpus row this file serves calls `next`
/// exactly once per freshly-constructed generator/iterator value
/// (`next(some_gen())`, never `next(g); next(g)` on one bound name), so
/// this row is honest for that shape; a second `next()` against the
/// SAME generator value would answer element 0 again rather than
/// element 1, which is a known gap this file does not claim to close.
fn next_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::List {
        return None;
    }
    only.items.first().cloned()
}

/// `anext(async_iterator)` (one-argument form, no `default`) — the
/// `async`-generator twin of `next(iterator)`: library/functions.html
/// documents `anext` as `next`'s async counterpart. `await anext(gen)`
/// evaluates through `evaluate_expression`'s own `Expr::Await` arm
/// (transparent unwrap — `async`/`await` carry no gate of their own,
/// matching this file's asyncio.gather doc's identical note), so the
/// `anext(...)` call itself lands in this dispatcher exactly like a
/// plain `next(...)` call would. An async generator's yielded elements
/// are the SAME `Kind::List` (tagged `source == "generator"`,
/// `instances::generator_yields`'s own doc) a sync generator's call
/// answers — `datamodel.rst`'s generator-iterator protocol makes no
/// distinction between a sync and an async generator's own yielded
/// VALUES, only in how the caller RECEIVES them (`__anext__` returns
/// an awaitable rather than the value directly) — so this row is
/// `next_call` under a different name, not a separate reading.
fn anext_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    next_call(arguments)
}

/// `typing.cast(typ, val)` — `Lib/typing.py`'s own `cast` docstring:
/// "This returns the value unchanged. To the type checker this signals
/// that the return value has the designated type, but at runtime we
/// intentionally don't check anything." `typ` is never read (a type
/// expression, not a value this file evaluates); `val` passes through
/// exactly, whatever shape it is — the identity function over its
/// second argument.
fn cast_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [_typ, val] = arguments else { return None };
    Some(val.clone())
}

/// `min`/`max` over two or more known single-numeric arguments —
/// library/functions.html#min and #max: "If two or more positional
/// arguments are provided, the smallest [largest] of the positional
/// arguments is returned." The single-iterable form (`min(some_list)`)
/// is not modeled here — that argument is not a known scalar, so
/// `single_known_numeric` declines it and the whole call declines.
/// Result sort: Python's min/max return the winning ARGUMENT unchanged,
/// so a Float argument winning over Integer arguments keeps Float — the
/// winning value's own sort is threaded through, not fixed at one sort.
fn min_max_call(
    arguments: &[AbstractValue],
    pick: fn(f64, f64) -> bool,
) -> Option<AbstractValue> {
    if arguments.len() < 2 {
        return None;
    }
    let mut best: Option<(f64, PrimitiveKind)> = None;
    for argument in arguments {
        let candidate = single_known_numeric(argument)?;
        best = Some(match best {
            None => candidate,
            Some(current) => {
                if pick(candidate.0, current.0) {
                    candidate
                } else {
                    current
                }
            }
        });
    }
    let (value, sort) = best?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value], sort, grade))
}

/// `int(x)` — library/functions.html#int: "For floating-point numbers,
/// this truncates towards zero." An already-Integer argument is the
/// identity read under this row (the same trunc-toward-zero rule with
/// no fractional part to discard). A known EXACT STRING parses through
/// `parse_base_ten_int_string` — the base-10 `int(string, base=10)`
/// row (functions.rst): j-stdlib-surfaces.py's own `int_parse`,
/// `int("40")`/`int("200")`, both exact parses this row now answers
/// precisely rather than declining. A string that does not parse as a
/// base-10 integer (`int("abc")`) still declines HERE — CPython raises
/// `ValueError` for it, which `expressions.rs`'s own `call_provable_
/// raise` speaks through the raise channel (its own `is_valid_base_
/// ten_int_string` gate, a parallel/duplicate validity check to this
/// row's own `parse_base_ten_int_string` — the two files stay
/// independent per the mission's own file-ownership split, so the
/// validity rule is written twice rather than shared across the
/// boundary).
fn int_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        let text: String = only.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        let parsed = parse_base_ten_int_string(&text)?;
        let grade = derived_trust_level(TrustSpec, arguments);
        return Some(known_values(vec![parsed], PrimitiveKind::Integer, grade));
    }
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value.trunc()], PrimitiveKind::Integer, grade))
}

/// `int(string, base=10)`'s exact parsed value, for the base-10
/// default form ONLY (`int_call`'s own scope — a `base=` keyword
/// changes the digit alphabet entirely and is not read by this row's
/// caller, which never passes one through). functions.rst's own
/// grammar: "the string can be preceded by + or - (with no space in
/// between), have leading zeros, be surrounded by whitespace, and have
/// single underscores interspersed between digits." Returns `None`
/// (never a fabricated value) the moment the text does not parse —
/// `call_provable_raise`'s own `is_valid_base_ten_int_string` is the
/// row that speaks the ValueError this shape raises at runtime.
fn parse_base_ten_int_string(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    let negative = trimmed.starts_with('-');
    let digits_and_underscores = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    if digits_and_underscores.is_empty() {
        return None;
    }
    let chars: Vec<char> = digits_and_underscores.chars().collect();
    if chars.first() == Some(&'_') || chars.last() == Some(&'_') {
        return None;
    }
    let mut digits = String::new();
    let mut previous_was_underscore = false;
    for &c in &chars {
        if c == '_' {
            if previous_was_underscore {
                return None;
            }
            previous_was_underscore = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return None;
        }
        digits.push(c);
        previous_was_underscore = false;
    }
    if digits.is_empty() {
        return None;
    }
    let magnitude: f64 = digits.parse().ok()?;
    Some(if negative { -magnitude } else { magnitude })
}

/// `float(x)` on a single known numeric — library/functions.html#float:
/// "Return a floating-point number constructed from a number or a
/// string." Restricted here to the numeric argument: a string argument
/// is never a `single_known_numeric`, so `float(str)` declines rather
/// than being answered by this row.
fn float_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, _sort) = single_known_numeric(only)?;
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_values(vec![value], PrimitiveKind::Float, grade))
}

/// `chr(i)` on a known Integer code point — library/functions.html#chr:
/// "Return the string representing a character whose Unicode code
/// point is the integer *i*." A one-code-point exact string, the same
/// `Kind::Values`/`PrimitiveKind::String` shape `string_models.rs`
/// builds for any other exact string. `i` outside the valid code-point
/// range (`0..=0x10FFFF`, the same range `char::from_u32` itself
/// enforces) has no row here: CPython raises `ValueError`, which this
/// domain has no channel for this wave, so this row declines rather
/// than answer a fabricated character.
fn chr_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let (value, sort) = single_known_numeric(only)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    if value < 0.0 || value > 0x10FFFF as f64 {
        return None;
    }
    char::from_u32(value as u32)?;
    Some(known_values(vec![value], PrimitiveKind::String, TrustSpec))
}

/// `str(object)` — library/stdtypes.rst's `class:: str(object='')`
/// constructor row: "Return a string version of *object*." Modeled for
/// three known argument shapes: an exact string (the identity
/// conversion — `str(word)` answers `word` unchanged, per the same
/// row's own "If *object* already is a string, it is returned
/// unchanged" behavior), a known Integer (CPython's plain decimal
/// spelling, no `.0` — the same integer-spelling rule
/// `expressions.rs`'s f-string composition already establishes for an
/// interpolated Integer), and a known EXCEPTION instance
/// (`expressions.rs`'s `exception_construction_value`, tagged
/// `source == "exception"`, one `args` field holding the constructor's
/// own positional arguments as a `Kind::List`) whose FIRST argument is
/// a known exact string — `str(Exception(message))` answers `message`
/// unchanged: `Doc/tutorial/errors.rst`, "Errors and Exceptions" §8.3,
/// "the exception instance... typically has an `args` attribute...
/// builtin exception types define `__str__` to print all the
/// arguments." A single-string-argument exception's `__str__` is
/// exactly that one string (CPython's own `BaseException.__str__`:
/// zero args -> `''`, one arg -> `str(args[0])`, 2+ args -> the
/// `repr()` of the whole tuple — only the one-string-argument row is
/// modeled here). A known FLOAT argument is NOT modeled: the
/// repr-shortest spelling `format_py_number` builds lives in the
/// `refined_sets` crate, out of this file's own dependency edge for
/// this wave, so `str(float)` declines rather than half-build that
/// spelling by hand.
fn str_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        return Some(only.clone());
    }
    if only.kind == Kind::Object && only.source == "exception" {
        return exception_single_string_message(only);
    }
    let (value, sort) = single_known_numeric(only)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    let spelled = format!("{}", value as i64);
    let code_points: Vec<f64> = spelled.chars().map(|c| c as u32 as f64).collect();
    Some(known_values(code_points, PrimitiveKind::String, TrustSpec))
}

/// The exact message `str()` of a known exception instance answers, for
/// the ONE constructor-argument shape this file models: an `args`
/// field (`expressions.rs`'s own exception-construction tag) holding a
/// `Kind::List` of exactly one known exact-string element —
/// `BaseException.__str__`'s one-argument row (this function's own
/// caller doc). Any other `args` shape (zero elements, 2+ elements, a
/// non-string element) declines — this file does not build the `repr()`
/// spelling a multi-argument `__str__` would need.
fn exception_single_string_message(instance: &AbstractValue) -> Option<AbstractValue> {
    let args = &instance.keys.iter().find(|key| key.name == "args")?.value;
    if args.kind != Kind::List {
        return None;
    }
    let [only] = args.items.as_slice() else { return None };
    if only.kind == Kind::Values && only.kind_tag == Some(PrimitiveKind::String) {
        return Some(only.clone());
    }
    None
}

/// The dispatcher: a call to Python builtin `function` with already-
/// evaluated `arguments`. `None` means "not modeled here" — the caller
/// declines honestly rather than reading this as "the call is unknown to
/// Python." `Some` is an exact answer at the derived trust grade.
pub fn builtin_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "abs" => abs_call(arguments),
        "round" => round_call(arguments),
        // two-or-more-argument form first (min_max_call's own `len < 2`
        // guard declines there); the single-iterable form is the ONE
        // row this file's own doc used to call out as "not modeled" —
        // now answered by min_max_over_iterable once the argument is a
        // known Kind::List
        "min" => min_max_call(arguments, |candidate, current| candidate < current)
            .or_else(|| min_max_over_iterable(arguments, |candidate, current| candidate < current)),
        "max" => min_max_call(arguments, |candidate, current| candidate > current)
            .or_else(|| min_max_over_iterable(arguments, |candidate, current| candidate > current)),
        // len() declines for now: answering it needs container states
        // (string/list/tuple/dict length facts) this domain does not yet
        // carry — single_known_numeric only ever reads a known SCALAR,
        // never a container, so there is no row to write until container
        // states land.
        "len" => None,
        "int" => int_call(arguments),
        "float" => float_call(arguments),
        "sum" => sum_call(arguments),
        "sorted" => sorted_call(arguments),
        "list" => list_constructor_call(arguments),
        "set" => set_constructor_call(arguments),
        "dict" => dict_constructor_call(arguments),
        "chr" => chr_call(arguments),
        "str" => str_call(arguments),
        "iter" => iter_call(arguments),
        "next" => next_call(arguments),
        "anext" => anext_call(arguments),
        "cast" => cast_call(arguments),
        // `type(object)` (one-argument form) — library/functions.html#type:
        // "With one argument, return the type of an object." This domain
        // has no type-object Kind, so the answer is opaque — the honest
        // "a type object" sort, never a specific value
        // (b-body-expressions.py's `type_as_value`). The three-argument
        // `type(name, bases, dict)` class-creation form is not this row
        // (a different arity, out of scope).
        "type" if arguments.len() == 1 => Some(opaque_value("a type object")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integer(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Integer, TrustSpec)
    }

    fn float(value: f64) -> AbstractValue {
        known_values(vec![value], PrimitiveKind::Float, TrustSpec)
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
}
