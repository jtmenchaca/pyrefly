//! Calls to Python builtins with determinable results, answered exactly.
//! Two dispatchers: `builtin_call_result` (pure Rust, no kernel) and
//! `builtin_call_result_with_kernel` (the caller's actual entry point —
//! tries the pure dispatcher first, then the row families that need a
//! kernel ask: `min`/`max` over a Set operand, and `abs` over a Set
//! operand). Both take the callee name and the already-evaluated
//! argument values; `None` means "not modeled here" (the caller
//! declines honestly), `Some` is an exact answer. Every modeled row
//! cites its clause of docs.python.org/3.12/library/functions.html or
//! library/stdtypes.html (the container constructors `list`/`set`/
//! `dict` live in stdtypes.html's own class entries); a row with no
//! citation is not written.

mod containers;
mod conversions;
mod numeric;
mod stdlib;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use refined_domain::abstract_value::{known_set, opaque_value, AbstractValue, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::TransferQuestionOp;

use containers::{
    anext_call, cast_call, dict_constructor_call, dict_fromkeys_call, hash_call, iter_call, list_constructor_call,
    next_call, object_call, set_constructor_call, struct_call_result,
};
use conversions::{
    bool_call, chr_call, float_call, float_call_over_set, format_call, int_call, int_call_over_set, int_image,
    ord_call, parse_qs_call, str_call, unicodedata_call_result, urllib_quote_call,
};
use numeric::{
    abs_call, abs_call_over_set, min_max_call, min_max_call_over_sets, min_max_call_with_nan_operand,
    min_max_over_iterable, min_max_over_star, reversed_call, round_call, sorted_call, sum_call, sum_call_over_star,
};
use stdlib::{os_call_result, time_call_result};

// Test module is a sibling of the domain children, so re-export their
// items into this module's namespace for `tests`'s `use super::*`.
#[cfg(test)]
pub(self) use containers::DICT_FROMKEYS_WORD;

/// The dispatcher: a call to Python builtin `function` with already-
/// evaluated `arguments`. `None` means "not modeled here" — the caller
/// declines honestly rather than reading this as "the call is unknown to
/// Python." `Some` is an exact answer at the derived trust grade. Pure
/// Rust, no kernel dependency — `builtin_call_result_with_kernel` is the
/// caller's actual entry point, trying the kernel-needing rows first
/// (`min`/`max`'s own set-valued row) and falling back to this
/// dispatcher unchanged; kept separate so every one of this dispatcher's
/// OWN tests keeps asserting with no kernel dylib required, exactly as
/// before `min`/`max` grew a kernel-asked arm.
pub fn builtin_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "abs" => abs_call(arguments),
        "round" => round_call(arguments),
        // two-or-more-argument form first (min_max_call's own `len < 2`
        // guard declines there); the single-iterable form answers
        // through min_max_over_iterable for a known Kind::List, then
        // min_max_over_star for an unknown-length star-shaped iterable
        // (a declared list[X]/set[X]/Sequence[X] parameter). The
        // kernel-asked set-valued row (`min_max_call_over_sets`) is
        // NOT reachable here — see `builtin_call_result_with_kernel`.
        "min" => min_max_call_with_nan_operand(arguments)
            .or_else(|| min_max_call(arguments, |candidate, current| candidate < current))
            .or_else(|| min_max_over_iterable(arguments, |candidate, current| candidate < current))
            .or_else(|| min_max_over_star(arguments)),
        "max" => min_max_call_with_nan_operand(arguments)
            .or_else(|| min_max_call(arguments, |candidate, current| candidate > current))
            .or_else(|| min_max_over_iterable(arguments, |candidate, current| candidate > current))
            .or_else(|| min_max_over_star(arguments)),
        // len() declines for now: answering it needs container states
        // (string/list/tuple/dict length facts) this domain does not yet
        // carry — single_known_numeric only ever reads a known SCALAR,
        // never a container, so there is no row to write until container
        // states land.
        "len" => None,
        "int" => int_call(arguments),
        // `bool(x)` always answers one of the two values — never a
        // decline, see `bool_call`'s own clause citation.
        "bool" => bool_call(arguments),
        "float" => float_call(arguments),
        "sum" => sum_call(arguments).or_else(|| sum_call_over_star(arguments)),
        "sorted" => sorted_call(arguments),
        "reversed" => reversed_call(arguments),
        "list" => list_constructor_call(arguments),
        "set" => set_constructor_call(arguments),
        "dict" => dict_constructor_call(arguments),
        "chr" => chr_call(arguments),
        "ord" => ord_call(arguments),
        // `input()`/`input(prompt)` — library/functions.rst: "The
        // function then reads a line from input, converts it to a
        // string (stripping a trailing newline), and returns that." The
        // line's own content comes from outside the program, so the
        // answer is the whole-strings ground `Σ*` — A3.seed.boundary's
        // own `line_from_input_outside` claim: a determined `str`-sorted
        // state, not a decline.
        "input" if arguments.len() <= 1 => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::String),
            ..known_set(refined_sets::codepoint_sets::strings(), None, TrustSpec, SetKindTag::None)
        }),
        "str" => str_call(arguments),
        "format" => format_call(arguments),
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
        "object" => object_call(arguments),
        "hash" => hash_call(arguments),
        // `from urllib.parse import quote` — see `urllib_quote_call`'s
        // own doc for why this bare-name spelling is a builtin row here
        // rather than routed through `stdlib_call_result`.
        "quote" => urllib_quote_call(arguments),
        // `from urllib.parse import parse_qs` — the same bare-name
        // spelling `quote` above takes, for the same reason.
        "parse_qs" => parse_qs_call(arguments),
        _ => None,
    }
}

/// The caller's actual entry point (`expressions.rs::evaluate_call`): a
/// call to Python builtin `function`, `kernel` in hand for the row
/// families that need it — `min`/`max`'s two-or-more-argument form when
/// at least one argument is a `Kind::Set` (`min_max_call_over_sets`'s
/// own doc, including the NaN-discharge citation), `abs`'s single
/// Set-seeded operand (`abs_call_over_set`'s own doc), `int`'s
/// single Set-seeded operand (`int_call_over_set`'s own doc — e.g.
/// `int(math.sqrt(x))` over a declared parameter range), and `float`'s
/// single Set-seeded operand (`float_call_over_set`'s own doc — e.g.
/// `float(math.floor(x))`; this one row needs no kernel round trip of
/// its own, only the `kernel` parameter's presence in this dispatcher's
/// signature to sit beside its `int`/`abs` siblings). Every other
/// builtin routes straight through the pure-Rust `builtin_call_result`
/// above, tried FIRST so a known-scalar call never pays a kernel round
/// trip it does not need.
pub fn builtin_call_result_with_kernel(
    function: &str,
    arguments: &[AbstractValue],
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    builtin_call_result(function, arguments).or_else(|| match function {
        "min" => min_max_call_over_sets(arguments, TransferQuestionOp::Min, kernel),
        "max" => min_max_call_over_sets(arguments, TransferQuestionOp::Max, kernel),
        "abs" => {
            let [only] = arguments else { return None };
            abs_call_over_set(only, kernel)
        }
        "int" => {
            let [only] = arguments else { return None };
            int_call_over_set(only, kernel).or_else(|| int_image())
        }
        "float" => {
            let [only] = arguments else { return None };
            float_call_over_set(only)
        }
        _ => None,
    })
}

/// The dispatcher for a MODULE-QUALIFIED stdlib call whose result is
/// answered from this file — `time.<function>`, `os.<function>`,
/// `unicodedata.<function>`, `dict.<function>` (a builtin TYPE's own
/// classmethod, gated in `expressions.rs::evaluate_attribute_call` the
/// same way a module name is: a bare `dict` receiver that reads unbound
/// in `environment`, since `dict` is never locally rebound in the
/// corpus's own rows). Callable name `module` (the attribute chain's
/// own root, e.g. `"time"`) and `function` (the called attribute, e.g.
/// `"time"`) are read separately so a caller can gate on the module
/// name exactly the way its own `math`/`re`/`json` arms already do,
/// before ever reaching this dispatcher. `None` means "not modeled
/// here" — the caller's own decline, never a guessed value.
/// `urllib.parse.quote` is NOT reached here — see `urllib_quote_call`'s
/// own doc for why it is a bare-name builtin row instead.
pub fn stdlib_call_result(module: &str, function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match module {
        "time" => time_call_result(function, arguments),
        "os" => os_call_result(function, arguments),
        "unicodedata" => unicodedata_call_result(function, arguments),
        "dict" if function == "fromkeys" => dict_fromkeys_call(arguments),
        "struct" => struct_call_result(function, arguments),
        _ => None,
    }
}
