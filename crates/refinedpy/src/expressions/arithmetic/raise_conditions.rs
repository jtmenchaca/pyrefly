use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::bytes_models;
use crate::bytes_models::BytesAnswer;
use crate::env::Environment;
use crate::expressions::call::eval_whole_integers;
use crate::expressions::call::is_valid_base_ten_int_string;
use crate::expressions::compare::exact_string_values;
use crate::expressions::compare::single_pair_equal;
use crate::expressions::datetime::date_fromisoformat_raises;
use crate::expressions::datetime::is_datetime_date_attribute;
use crate::expressions::evaluate_expression;
use crate::expressions::fstring::code_points_to_string;
use crate::math_models;

use super::known_values::single_numeric_value;

/// `container[index]` where `container` and `index` are both KNOWN and
/// the read is provably out of range/absent — the same distinction
/// `collection_models::subscript_read`'s own doc draws between "not
/// modeled" and "known container, known index, provably absent": a
/// `subscript_read` decline on an UNKNOWN container or index states
/// nothing about the real runtime behavior (this function must decline
/// too), while a decline on a KNOWN List with a KNOWN out-of-range
/// Integer index, or a KNOWN Object with a KNOWN string key absent from
/// its own `keys`, is exactly the shape CPython raises
/// `IndexError`/`KeyError` for (expressions.rst, "Subscriptions";
/// stdtypes.rst, dict's `d[key]` row).
///
/// A `Kind::List` receiver tries `bytes_models::bytes_index` FIRST: a
/// `bytes`/`bytearray`/`array.array` value is the identical `Kind::List`
/// shape an ordinary list literal builds (bytes_models.rs's own module
/// doc), and `bytes_index`'s negative-index-adjusted bounds check is the
/// same rule an ordinary list read follows — so its own `Raises` message
/// already speaks correctly for BOTH a bytes-like receiver and a plain
/// list receiver, and this file does not re-derive that bounds
/// arithmetic a second time. `known_container_index_absent` below is
/// reached only for the `Kind::Object` (dict `KeyError`) row, which
/// `bytes_models.rs` has no function for.
pub(in crate::expressions) fn subscript_provable_raise(
    subscript: &ruff_python_ast::ExprSubscript,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    if matches!(subscript.slice.as_ref(), Expr::Slice(_)) {
        // a slice never raises for an out-of-bounds bound (silently
        // clamped, expressions.rst) — nothing to prove here
        return None;
    }
    let container = evaluate_expression(&subscript.value, environment, kernel);
    let index = evaluate_expression(&subscript.slice, environment, kernel);
    if container.kind == Kind::List {
        if let Some(BytesAnswer::Raises(message)) = bytes_models::bytes_index(&container, &index) {
            return Some((subscript.range(), one_voice_raise_message(&message)));
        }
        return None;
    }
    if let Some(detail) = known_string_index_out_of_range(&container, &index) {
        return Some((subscript.range(), format!("this expression provably raises {detail}")));
    }
    known_container_index_absent(&container, &index).map(|detail| {
        (
            subscript.range(),
            format!("this expression provably raises {detail}"),
        )
    })
}

/// Whether a KNOWN exact-string `container` provably has no code point at
/// a KNOWN Integer `index` — the string-receiver row `subscript_read`'s
/// own decline cannot distinguish from "not modeled": `s[i]` follows the
/// same negative-index-adjust-then-bounds-check rule an ordinary list
/// read follows (expressions.rst, "Subscriptions" — the same
/// `__getitem__` machinery every built-in sequence shares), and an
/// adjusted index still outside `0..len` raises `IndexError` exactly the
/// way a list's own out-of-range read does (`bytes_index`'s row, read
/// above, is this same check for a bytes-like `Kind::List` receiver —
/// this function is its string-shaped twin). A `Kind::List`/`Kind::Object`
/// container, or a non-Integer index, is not this function's row —
/// `None`.
pub(in crate::expressions) fn known_string_index_out_of_range(container: &AbstractValue, index: &AbstractValue) -> Option<String> {
    let text = exact_string_values(container)?;
    let (value, sort) = single_numeric_value(index)?;
    if sort != PrimitiveKind::Integer {
        return None;
    }
    let position = value as i64;
    let length = text.len() as i64;
    let adjusted = if position < 0 { position + length } else { position };
    if adjusted >= 0 && adjusted < length {
        None
    } else {
        Some("IndexError: string index out of range".to_owned())
    }
}

/// Normalizes a `bytes_models.rs`-voiced raise sentence ("this read/write
/// provably raises...") to `provable_raise`'s own one voice, "this
/// expression provably raises..." — the two files speak the same fact
/// (a provable runtime raise) but were built with slightly different
/// wording for their own subject ("read"/"write" vs. "expression"); this
/// function is the one seam where the two meet, so every message this
/// function hands back reads in exactly one voice regardless of which
/// sibling file decided the raise.
pub(in crate::expressions) fn one_voice_raise_message(message: &str) -> String {
    match message.split_once("provably raises") {
        Some((_, rest)) => format!("this expression provably raises{rest}"),
        None => message.to_owned(),
    }
}

/// Whether a KNOWN Object `container` provably lacks a KNOWN string
/// `key` — the exact-value companion to
/// `collection_models::subscript_read`'s dict row, deciding the same
/// membership question directly against `container.keys` so a caller
/// can tell "provably absent" apart from "not modeled" (which
/// `subscript_read`'s bare `None` cannot do alone). The `Kind::List` row
/// is handled by `subscript_provable_raise` itself through
/// `bytes_models::bytes_index` (see that function's own doc) — this
/// function covers `Kind::Object` (dict `KeyError`) only. `Some(detail)`
/// names the ExcType and the missing key, in `provable_raise`'s own
/// voice fragment (the `ExcType: detail` half, joined by the caller);
/// `None` for an unknown container/key, or a key that IS present.
pub(in crate::expressions) fn known_container_index_absent(container: &AbstractValue, index: &AbstractValue) -> Option<String> {
    if container.kind != Kind::Object {
        return None;
    }
    let key = exact_string_values(index).map(code_points_to_string)??;
    let present = container.keys.iter().any(|entry| entry.name == key);
    if present {
        None
    } else {
        Some(format!("KeyError: '{key}'"))
    }
}

/// `math.log(x)`/`log2`/`log10`/`log1p`/`asin`/`acos`/`atanh`/`acosh`
/// where `x`'s window STRADDLES CPython's own raise domain (some
/// admitted values raise, the rest still return a value) —
/// `math_models::DomainRaiseClassification::Straddles`'s own row, the
/// sibling this `call_provable_raise`'s all-or-nothing arm explicitly
/// defers to. The window's ENTIRELY-raising case is `call_provable_
/// raise`'s own row (an unconditional fire, no value question at all);
/// the ENTIRELY-served case fires nothing here and answers its value
/// through `math_models::math_call_result`'s ordinary kernel-backed
/// path, unaffected by this function.
pub(in crate::expressions) fn domain_limited_family_possible_raise(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    let family = math_models::DomainLimitedFamily::of_function(attribute.attr.as_str())?;
    let Expr::Name(module_name) = attribute.value.as_ref() else {
        return None;
    };
    if module_name.id.as_str() != "math" || environment.read("math").is_some() {
        return None;
    }
    let [only_arg] = &*call.arguments.args else {
        return None;
    };
    let argument = evaluate_expression(only_arg, environment, kernel);
    if !matches!(
        math_models::domain_raise_classification(family, &argument, kernel),
        Some(math_models::DomainRaiseClassification::Straddles)
    ) {
        return None;
    }
    Some((call.range(), family.raise_message().to_owned()))
}

/// A call expression's own provable raise, once its callee and every
/// argument have already been checked (by `provable_raise`'s own
/// pre-order walk): a bytes-like element read/write whose
/// `bytes_models` answer is `Raises`, `int(<a known string that does
/// not parse as an int>)`, or `<receiver>.index(<a known needle absent
/// from a known receiver>)`.
pub(in crate::expressions) fn call_provable_raise(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<(TextRange, String)> {
    // `f(*x)` where `x` is a genuinely unbounded list VALUE (`values:
    // list[int]`, b-body-expressions.py's own `wrapper_spread_call_
    // unbounded`) — CPython's own call path fails past a large
    // positional-argument count (`splice_call_arguments`'s own doc: an
    // unbounded iterable has no proven element count to splice into a
    // fixed argument vector, so the VALUE path declines rather than
    // guess). That silence is honest for the VALUE question, but the
    // SHAPE itself is a fact this checker already knows: an unpack
    // whose own length is unproven can throw at runtime regardless of
    // what the call computes. A KNOWN-length iterable (`Kind::List`,
    // whatever its own elements are known) never fires here —
    // `wrapper_spread_call`'s own `max(*[200, 201])` stays silent,
    // exactly as `splice_call_arguments` already treats it.
    //
    // EXCLUDED: `x` is this body's OWN `*args`/`**kwargs` parameter,
    // forwarded bare (`environment::is_variadic_parameter` — r-ast-
    // census.py's `wrapper(*args: P.args, **kwargs: P.kwargs): return
    // f(*args, **kwargs)`). A ParamSpec-captured vararg forward hands
    // CPython exactly the arguments THIS body itself received; it is
    // never an independently-grown collection whose length could
    // exceed what a real call already survived to reach this body, so
    // it never raises on this shape alone.
    for arg in &call.arguments.args {
        if let Expr::Starred(starred) = arg {
            if let Expr::Name(spread_name) = starred.value.as_ref() {
                if environment.is_variadic_parameter(spread_name.id.as_str()) {
                    continue;
                }
            }
            let spread = evaluate_expression(&starred.value, environment, kernel);
            if spread.kind != Kind::List {
                let callee_name = callee_display_name(call.func.as_ref());
                return Some((
                    call.range(),
                    format!(
                        "this expression provably raises TypeError: the list can hold any number of items, and the unpack hands each to {callee_name} as its own argument"
                    ),
                ));
            }
        }
    }
    if let Expr::Name(name) = call.func.as_ref() {
        // a `base=` keyword changes the parsing rules entirely (a
        // non-decimal radix admits letters as digits) — this row only
        // ever decides the base-10 default, so ANY keyword argument
        // (not just a `base=` one) declines rather than risk judging a
        // non-base-10 call by the base-10 grammar
        if name.id.as_str() == "int"
            && environment.read("int").is_none()
            && call.arguments.keywords.is_empty()
        {
            let [only] = &*call.arguments.args else {
                return None;
            };
            let value = evaluate_expression(only, environment, kernel);
            let text = exact_string_values(&value).and_then(code_points_to_string)?;
            // library/functions.rst's `int(string, base=10)` row: "the
            // string can be preceded by + or - (with no space in
            // between), have leading zeros, be surrounded by whitespace,
            // and have single underscores interspersed between digits."
            // The exact ValueError wording ("invalid literal for int()
            // with base 10: '...'") is pinned by library/unittest.rst's
            // own worked test example (`assertRaisesRegex(ValueError,
            // "invalid literal for.*XYZ'$", int, 'XYZ')`) rather than by
            // functions.rst's own int() entry directly — the vendored
            // tree does not restate the message inline there.
            if !is_valid_base_ten_int_string(&text) {
                return Some((
                    call.range(),
                    format!("this expression provably raises ValueError: invalid literal for int() with base 10: '{text}'"),
                ));
            }
            return None;
        }
        // `float(<a known string outside the documented grammar>)`
        // provably raises `ValueError` — library/functions.rst's own
        // `float(string, /)` entry: "the input must conform to the
        // floatvalue production rule in the following grammar, after
        // leading and trailing whitespace characters are removed."
        // `is_valid_float_string` spells that production exactly. The
        // production admits NO trailing garbage and has no empty
        // alternative, which is why `float("1.5abc")` and `float("")`
        // both raise — the two facts A3.xfer.parse's own
        // `parse_float_lenient_raises` and `float_empty_raises` state,
        // each a place Python differs from JS (`parseFloat` stops at
        // the garbage; `Number("")` answers 0). As with the `int` row
        // above, a keyword argument declines rather than judge a call
        // this grammar does not describe.
        if name.id.as_str() == "float" && environment.read("float").is_none() && call.arguments.keywords.is_empty() {
            let [only] = &*call.arguments.args else {
                return None;
            };
            let value = evaluate_expression(only, environment, kernel);
            let text = exact_string_values(&value).and_then(code_points_to_string)?;
            if !is_valid_float_string(&text) {
                return Some((
                    call.range(),
                    format!("this expression provably raises ValueError: could not convert string to float: '{text}'"),
                ));
            }
            return None;
        }
    }
    if let Expr::Attribute(attribute) = call.func.as_ref() {
        // `math.sqrt(<known negative>)` provably raises `ValueError` —
        // library/math.rst's own module-introduction note: "The current
        // implementation will raise ValueError for invalid operations
        // like sqrt(-1.0)..." (math_models.rs's own
        // `sqrt_argument_is_known_negative` reads the same operand this
        // row checks, so the value dispatch and the raise dispatch
        // agree on exactly which sqrt calls are negative).
        if attribute.attr.as_str() == "sqrt" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    let arguments: Vec<AbstractValue> =
                        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, kernel)).collect();
                    if math_models::sqrt_argument_is_known_negative(&arguments) {
                        return Some((
                            call.range(),
                            "this expression provably raises ValueError: math domain error".to_owned(),
                        ));
                    }
                }
            }
        }
        // `math.sin`/`math.cos`/`math.tan(<known infinite>)` provably
        // raises `ValueError` — the SAME "invalid operations" module-
        // introduction note the `sqrt` row above cites, extended to the
        // platform C99 Annex F domain error an infinite argument gives
        // `sin`/`cos`/`tan` (`math_models::trig_argument_is_known_
        // infinite`'s own doc names the exact clause and scope).
        if matches!(attribute.attr.as_str(), "sin" | "cos" | "tan") {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    let arguments: Vec<AbstractValue> =
                        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, kernel)).collect();
                    if math_models::trig_argument_is_known_infinite(attribute.attr.as_str(), &arguments) {
                        return Some((
                            call.range(),
                            "this expression provably raises ValueError: math domain error".to_owned(),
                        ));
                    }
                }
            }
        }
        // `math.pow(<known negative finite base>, <known finite
        // non-integer exponent>)` provably raises `ValueError` —
        // library/math.rst's own `pow(x, y)` clause: "If both x and y
        // are finite, x is negative, and y is not an integer then
        // pow(x, y) is undefined, and raises ValueError."
        // (`math_models.rs`'s own `pow_arguments_provably_raise` reads
        // the same two operands this row checks, in the same order and
        // under the same `pow(1.0, x)`/`pow(x, 0.0)` precedence the
        // doc's own clause states, so the value dispatch and this raise
        // dispatch agree on exactly which `math.pow` calls raise.)
        if attribute.attr.as_str() == "pow" {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    let arguments: Vec<AbstractValue> =
                        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, kernel)).collect();
                    if math_models::pow_arguments_provably_raise(&arguments) {
                        return Some((
                            call.range(),
                            "this expression provably raises ValueError: math domain error".to_owned(),
                        ));
                    }
                }
            }
        }
        // `date.fromisoformat(<known malformed or calendrically invalid
        // string>)` provably raises `ValueError` — see
        // `date_fromisoformat_raises`'s own doc for the exact grammar and
        // why a malformed string (`"13:45"`) raises the same as a
        // syntactically-shaped but invalid one (`"2023-02-29"`). Reads the
        // SAME receiver recognition (`is_datetime_date_attribute`) and
        // argument shape (a single keyword-free positional) the value
        // dispatch's own `date.fromisoformat` row uses, so the two
        // dispatches agree on exactly which calls this construct owns.
        if is_datetime_date_attribute(attribute.value.as_ref(), environment) && attribute.attr.as_str() == "fromisoformat" {
            if let [text] = &*call.arguments.args {
                if call.arguments.keywords.is_empty() {
                    let argument = evaluate_expression(text, environment, kernel);
                    if let Some(code_points) = exact_string_values(&argument) {
                        if let Some(spelling) = code_points_to_string(code_points) {
                            if date_fromisoformat_raises(&spelling, kernel) == Some(true) {
                                return Some((
                                    call.range(),
                                    "this expression provably raises ValueError: Invalid isoformat string".to_owned(),
                                ));
                            }
                        }
                    }
                }
            }
        }
        // `math.log`/`log2`/`log10`/`log1p`/`asin`/`acos`/`atanh`/`acosh`
        // of a KNOWN operand whose window is ENTIRELY inside CPython's
        // own raise domain provably raises `ValueError: math domain
        // error` — `math_models::DomainLimitedFamily::raise_domain`'s
        // own doc cites the exact `mathmodule.c` clause per family
        // (verified against the vendored source, not against the
        // kernel's own JavaScript-facing `.nan` corner, which disagrees
        // with CPython at one boundary point for `log`/`log2`/`log10`/
        // `log1p`/`atanh`). specifications/python/Doc/library/
        // math.rst:696-698 is the module's own impl-detail note citing
        // `log(0.0)` as its worked example of exactly this row. A
        // window that only STRADDLES the raise domain is `possible_
        // raise`'s own row (`domain_limited_family_possible_raise`
        // below), not this one's — this function's contract is
        // all-or-nothing, so only `EntirelyRaises` fires here.
        if let Some(family) = math_models::DomainLimitedFamily::of_function(attribute.attr.as_str()) {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    if let [only_arg] = &*call.arguments.args {
                        let argument = evaluate_expression(only_arg, environment, kernel);
                        if matches!(
                            math_models::domain_raise_classification(family, &argument, kernel),
                            Some(math_models::DomainRaiseClassification::EntirelyRaises)
                        ) {
                            return Some((call.range(), family.raise_message().to_owned()));
                        }
                    }
                }
            }
        }
        // `math.floor`/`ceil`/`trunc` of a KNOWN NON-FINITE argument
        // provably raises: each returns an `Integral`, and no Python
        // `int` is infinite or NaN. `rounding_argument_raises` names
        // which exception and CPython's own message; it reads the same
        // operand through the same domain gate the value rows use
        // (`integral_domain_admits`), so the value dispatch and this
        // raise dispatch agree on exactly which rounding calls raise.
        //
        // `rounding_argument_raises` reads a `Kind::Values` operand only
        // (`single_numeric_operand`'s own gate) — a NaN-PRODUCING argument
        // (`float("nan")`, `math.inf - math.inf`, …) never reaches that
        // shape at all: `refinement_forms::element` refuses NaN at
        // construction, so the domain answers the distinct `Kind::NaN`
        // state (`nan_value()`) instead of a `Kind::Values` list holding a
        // NaN element. This arm reads that state directly, the same
        // pairing `binary_arithmetic_value`'s `inf - inf` row keeps for
        // its own callers.
        if matches!(attribute.attr.as_str(), "floor" | "ceil" | "trunc") {
            if let Expr::Name(module_name) = attribute.value.as_ref() {
                if module_name.id.as_str() == "math" && environment.read("math").is_none() {
                    let arguments: Vec<AbstractValue> =
                        call.arguments.args.iter().map(|arg| evaluate_expression(arg, environment, kernel)).collect();
                    if let [only] = arguments.as_slice() {
                        if only.kind == Kind::NaN {
                            return Some((
                                call.range(),
                                "this expression provably raises ValueError: cannot convert float NaN to integer"
                                    .to_owned(),
                            ));
                        }
                    }
                    if let Some((exception, detail)) =
                        math_models::rounding_argument_raises(attribute.attr.as_str(), &arguments)
                    {
                        return Some((call.range(), format!("this expression provably raises {exception}: {detail}")));
                    }
                }
            }
        }
        // `<known bytes>.decode("ascii")` where some byte is outside
        // `[0, 127]` provably raises `UnicodeDecodeError` — the "ascii"
        // codec maps exactly the seven-bit range
        // (`Doc/library/codecs.rst`'s own codec table), so a byte past
        // `0x7F` has no ASCII character at all and the strict default
        // error handler raises rather than substitute one. Reads the
        // SAME receiver and encoding shapes `bytes_models::
        // bytes_decode_call`'s own value row reads, so the value
        // dispatch and this raise dispatch agree on exactly which
        // decodes this construct owns.
        if attribute.attr.as_str() == "decode" {
            if let [encoding_expr] = &*call.arguments.args {
                let encoding = evaluate_expression(encoding_expr, environment, kernel);
                let encoding_text = exact_string_values(&encoding).and_then(code_points_to_string);
                let receiver = evaluate_expression(&attribute.value, environment, kernel);
                if encoding_text.as_deref() == Some("ascii") && receiver.kind == Kind::List {
                    let offending = receiver.items.iter().enumerate().find(|(_, element)| {
                        element.kind == Kind::Values && element.values.len() == 1 && !(0.0..=127.0).contains(&element.values[0])
                    });
                    if let Some((position, element)) = offending {
                        let byte = element.values[0] as i64;
                        return Some((
                            call.range(),
                            format!(
                                "this expression provably raises UnicodeDecodeError: 'ascii' codec can't decode byte 0x{byte:02x} in position {position}: ordinal not in range(128)"
                            ),
                        ));
                    }
                }
            }
        }
        if attribute.attr.as_str() == "index" {
            let [needle_expr] = &*call.arguments.args else {
                return None;
            };
            let receiver = evaluate_expression(&attribute.value, environment, kernel);
            let needle = evaluate_expression(needle_expr, environment, kernel);
            // str.index/list.index RAISE on a miss (AGENT-BRIEF.md;
            // stdtypes.rst's Common Sequence Operations table, note (8):
            // "index raises ValueError when x is not found in s")
            if let (Some(receiver_text), Some(needle_text)) =
                (exact_string_values(&receiver), exact_string_values(&needle))
            {
                let receiver_text = code_points_to_string(receiver_text)?;
                let needle_text = code_points_to_string(needle_text)?;
                if !receiver_text.contains(&needle_text) {
                    return Some((
                        call.range(),
                        format!("this expression provably raises ValueError: '{needle_text}' is not in string"),
                    ));
                }
                return None;
            }
            if receiver.kind == Kind::List {
                let found = receiver.items.iter().any(|element| single_pair_equal(element, &needle) == Some(true));
                if !found {
                    return Some((
                        call.range(),
                        "this expression provably raises ValueError: value is not in list".to_owned(),
                    ));
                }
            }
        }
    }
    // a bytes-like element access (`data[i]`/`data[i] = v`) already
    // routes through `subscript_provable_raise`'s own container-shaped
    // check above for a READ; a WRITE has no expression-level call site
    // this function walks (an assignment target is a statement-level
    // concern, check.rs's own sink), so `bytes_models::bytes_index`'s
    // `Raises` answer is not reached from a bare call expression here —
    // noted rather than silently unhandled.
    None
}

/// Whether `text` conforms to `float(string, /)`'s own `floatvalue`
/// production (library/functions.rst, transcribed):
///
/// ```text
/// sign: "+" | "-"
/// infinity: "Infinity" | "inf"
/// nan: "nan"
/// digit: <a Unicode decimal digit, i.e. characters in Unicode general category Nd>
/// digitpart: digit (["_"] digit)*
/// number: [digitpart] "." digitpart | digitpart ["."]
/// exponent: ("e" | "E") [sign] digitpart
/// floatnumber: number [exponent]
/// absfloatvalue: floatnumber | infinity | nan
/// floatvalue: [sign] absfloatvalue
/// ```
///
/// applied "after leading and trailing whitespace characters are
/// removed," and with "case is not significant" for the `infinity` and
/// `nan` spellings. The production has no empty alternative and admits
/// no trailing text, so `""` and `"1.5abc"` both fail here.
fn is_valid_float_string(text: &str) -> bool {
    let body = text.trim();
    let body = body.strip_prefix(['+', '-']).unwrap_or(body);
    let lowered = body.to_ascii_lowercase();
    if matches!(lowered.as_str(), "infinity" | "inf" | "nan") {
        return true;
    }
    // floatnumber: number [exponent] — split at the first `e`/`E` that
    // begins a well-formed exponent tail.
    let (number, exponent) = match body.find(['e', 'E']) {
        Some(at) => (&body[..at], Some(&body[at + 1..])),
        None => (body, None),
    };
    if let Some(exponent) = exponent {
        let digits = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
        if !is_digitpart(digits) {
            return false;
        }
    }
    // number: [digitpart] "." digitpart | digitpart ["."]
    match number.split_once('.') {
        Some((whole, fraction)) => {
            if fraction.is_empty() {
                // digitpart "."
                is_digitpart(whole)
            } else if whole.is_empty() {
                // "." digitpart
                is_digitpart(fraction)
            } else {
                is_digitpart(whole) && is_digitpart(fraction)
            }
        }
        None => is_digitpart(number),
    }
}

/// Whether `text` is a `digitpart`: `digit (["_"] digit)*` — one or more
/// Unicode decimal digits with single underscores allowed only BETWEEN
/// digits, never leading, trailing, or doubled.
///
/// The production's `digit` is "a Unicode decimal digit, i.e. characters
/// in Unicode general category Nd". `char::to_digit(10)` answers `Some`
/// for exactly the characters Rust classifies as decimal digits, which
/// is narrower than `char::is_numeric` (that one additionally admits
/// categories Nl and No — Roman numerals, superscripts — which this
/// production does not).
fn is_digitpart(text: &str) -> bool {
    let mut characters = text.chars();
    let Some(first) = characters.next() else {
        return false;
    };
    if !is_decimal_digit(first) {
        return false;
    }
    while let Some(character) = characters.next() {
        if character == '_' {
            // an underscore must be followed by a digit
            match characters.next() {
                Some(following) if is_decimal_digit(following) => continue,
                _ => return false,
            }
        }
        if !is_decimal_digit(character) {
            return false;
        }
    }
    true
}

/// Whether `character` is a Unicode decimal digit (general category Nd).
fn is_decimal_digit(character: char) -> bool {
    character.is_ascii_digit() || (character.is_numeric() && character.to_digit(10).is_some())
}

/// A callee expression's own plain name for a raise message — a bare
/// `Name` reads directly (`max`, `sorted`, …); an `Attribute` reads its
/// own trailing name (`obj.method` names `method`, the part CPython's
/// own `TypeError` messages name); anything else (a call result used
/// directly as a callee, for instance) falls back to a generic "the
/// call" rather than guess at a name that is not there.
pub(in crate::expressions) fn callee_display_name(callee: &Expr) -> String {
    match callee {
        Expr::Name(name) => name.id.as_str().to_owned(),
        Expr::Attribute(attribute) => attribute.attr.as_str().to_owned(),
        _ => "the call".to_owned(),
    }
}

/// Whether `text` parses as a base-10 `int(str)` argument
/// (library/functions.rst's `int(string, base=10)` row, quoted in
/// `call_provable_raise`'s own doc): optional surrounding whitespace,
/// an optional single leading `+`/`-` with no space before the digits,
/// then one or more ASCII decimal digits with single underscores
/// allowed BETWEEN digits only (never leading, trailing, or doubled —
/// "single underscores interspersed between digits"). An empty digit
/// run (after stripping the sign) is never valid — `int("+")`,
/// `int("-")`, and `int("")` all raise the same way an all-underscore
/// run does.
/// `eval(source)` — library/functions.rst's own `eval` entry: "The
/// *source* argument... is parsed and evaluated as a Python
/// expression." `eval` is a HOST BOUNDARY — its source string is
/// evaluated by CPython's own compiler/interpreter at runtime, a
/// dynamic capability this file does not model at all (general
/// parsing, name resolution, and evaluation of an arbitrary expression
/// are all out of scope, matching every other host-boundary row in
/// this file, e.g. `re.match`'s own opaque "a match object" answer).
/// The ENTIRE surface modeled is "a single known-string argument that
/// SYNTACTICALLY reads as a plain int/float literal spelling states
/// that literal's SORT" — never the exact value: even though
/// `eval("40")` is execution-verified to answer the exact int `40`,
/// answering the exact value here would mean this file is quietly
/// interpreting Python source text, which is the general-evaluation
/// capability this file explicitly declines everywhere else. A
/// sort-only answer (the whole-number set for an int-literal spelling,
/// `float_sorted_unknown()` for a float-literal spelling) keeps `eval`
/// honestly in the same "claims a sort, never a value" tier as
/// `math`'s approximated family and a same-module call's declined-body
/// return-annotation fallback (`summaries::return_sort_fallback`) —
/// graded `TrustSpec`, a claim about what KIND of literal the source
/// spells, never a proved fact about the value `eval` would actually
/// produce. Any other spelling (an expression, a call, a name, an
/// operator, a string this file cannot read as a plain literal) still
/// declines outright — `eval` on arbitrary source is never modeled
/// beyond these two literal-SORT rows.
pub(in crate::expressions) fn eval_literal_value(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    let text = exact_string_values(only).and_then(code_points_to_string)?;
    let trimmed = text.trim();
    if is_valid_base_ten_int_string(trimmed) {
        return Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Integer),
            ..known_set(eval_whole_integers(), None, TrustSpec, SetKindTag::None)
        });
    }
    // a plain float literal spelling: an optional sign, decimal digits,
    // exactly one '.', decimal digits — no exponent/underscore/inf/nan
    // spelling is read (none of those are exercised by any row this
    // file serves, and each would need its own citation)
    let digits_and_sign = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    let is_plain_float_spelling = digits_and_sign.contains('.')
        && digits_and_sign.chars().all(|c| c.is_ascii_digit() || c == '.')
        && digits_and_sign.matches('.').count() == 1;
    if is_plain_float_spelling {
        return Some(float_sorted_unknown());
    }
    None
}
