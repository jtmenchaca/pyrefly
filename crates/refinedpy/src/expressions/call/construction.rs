//! Exception/bytes/array construction calls, the `math`-from-import
//! constant table, and the `datetime.timezone.utc` syntactic recognizer
//! — the value-building rows `evaluate_call` reaches before its generic
//! builtin dispatch.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::ModModule;
use ruff_python_ast::Stmt;

use crate::bytes_models;
use crate::collection_models;
use crate::env::Environment;
use crate::math_models;

use super::super::arithmetic::single_numeric_value;
use super::super::compare::exact_string_values;
use super::super::evaluate_expression;
use super::super::fstring::code_points_to_string;

/// Whether `name` is one of the built-in exception classes this file
/// constructs an `args`-carrying (or, for `ExceptionGroup`, opaque)
/// instance for — `exceptions.rst`'s own class hierarchy: `Exception`,
/// `ValueError`, `RuntimeError`, `TypeError`, and `KeyError` are each a
/// bare `BaseException.__init__(*args)` call with no extra fields of
/// their own (unlike `OSError`'s special-cased constructor, which this
/// file does not model). `ExceptionGroup` is listed here too so
/// `evaluate_call`'s ONE gate covers every recognized exception name,
/// even though it answers opaque rather than the tagged `args` shape
/// the others do (see that call site's own doc).
pub(in super::super) fn is_builtin_exception_constructor(name: &str) -> bool {
    matches!(name, "Exception" | "ValueError" | "RuntimeError" | "TypeError" | "KeyError" | "ExceptionGroup")
}

/// `Exception(*args)` / `ValueError(*args)` / `RuntimeError(*args)` /
/// `TypeError(*args)` / `KeyError(*args)` — a tagged `Kind::Object`
/// (`source = "exception"`) carrying every positional constructor
/// argument, in order, under one `args` field: tutorial/errors.rst
/// §8.3, "the exception instance... typically has an `args` attribute
/// that stores the arguments." Reads through this tag: `.args[0]` (this
/// file's own `evaluate_attribute_read`'s untagged-instance fallback,
/// since no `ClassModel` is ever registered under the name
/// `"exception"`, so `instances::field_read`'s plain by-name scan
/// answers the `args` `ObjectKey` directly) and `str(...)`
/// (`builtin_models::str_call`'s exception row, reading the SAME `args`
/// field by name).
pub(crate) fn exception_construction_value(arguments: &[AbstractValue]) -> AbstractValue {
    let args = collection_models::list_literal_value(arguments);
    let mut instance = known_object(
        vec![ObjectKey {
            name: "args".to_owned(),
            numeric: false,
            value: args,
        }],
        None,
        true,
        TrustProved,
        false,
    );
    instance.source = "exception".to_owned();
    instance
}

/// The tagged, FIELDLESS exception shape `check.rs`'s own
/// `caught_exception_value` binds a caught exception name to when the
/// try body's own raise cannot be found (a computed exception type, a
/// bare `except:`, more than one matching raise, …): the same
/// `source = "exception"` tag `exception_construction_value` gives a
/// freshly-constructed exception, but with no `args`/`__cause__`
/// field — a read through it (`.args`, `.__cause__`) finds nothing this
/// domain models, the honest "not yet readable" answer, never a false
/// Unknown-is-opaque read that a bare `opaque_value` would give (an
/// opaque value carries no `source` at all, so it cannot even be
/// recognized as an exception by a later `isinstance`/`str()` reader).
pub(crate) fn fieldless_exception_value() -> AbstractValue {
    let mut instance = known_object(Vec::new(), None, true, TrustProved, false);
    instance.source = "exception".to_owned();
    instance
}

/// Every element of a known `Kind::List` receiver, read as a single
/// known Integer in `0..=255` — the shared reader `bytes_like_
/// construction_value`'s own `bytes(<list>)`/`bytearray(<list>)` rows
/// need to turn an already-evaluated argument list back into the raw
/// `u8` sequence `bytes_models::bytes_literal_value` takes. `None` the
/// moment the receiver is not a known list, or any element is not a
/// known Integer in range — CPython itself raises `ValueError: bytes
/// must be in range(0, 256)` for an out-of-range element at
/// CONSTRUCTION time (`bytes_literal_value`'s own doc), a fact this
/// file does not yet speak through a `provable_raise` row for the
/// constructor call itself, so an out-of-range element declines the
/// whole construction rather than silently clamp it.
pub(in super::super) fn known_byte_sequence(value: &AbstractValue) -> Option<Vec<u8>> {
    if value.kind != Kind::List {
        return None;
    }
    value
        .items
        .iter()
        .map(|item| {
            let (raw, sort) = single_numeric_value(item)?;
            if sort != PrimitiveKind::Integer {
                return None;
            }
            if !(0.0..=255.0).contains(&raw) {
                return None;
            }
            Some(raw as u8)
        })
        .collect()
}

/// `bytes(...)` / `bytearray(...)` / `memoryview(...)` construction —
/// p-typed-array.py's own construction band, wired onto
/// `bytes_models.rs`'s existing element machinery (that file's own
/// module doc: no dedicated bytes/array `Kind` exists or is needed,
/// every one of these values is the identical `Kind::List` an ordinary
/// list literal builds).
///
/// - `bytearray(<known Integer length>)` — `bytearray_from_length`'s
///   own row: stdtypes.rst's `bytearray([source[, encoding[,
///   errors]]])`, "If it is an integer, the array will have that size
///   and will be initialized with null bytes." A length outside
///   `0..=1024` declines (an honest bound against building an
///   unreasonably large element vector for a value this file never
///   needs beyond the corpus's own small fixtures).
/// - `bytes(<known list of known Integers 0..=255>)` /
///   `bytearray(<known list of known Integers 0..=255>)` — `bytes_
///   from_iterable`'s own row: "If it is an iterable, it must be an
///   iterable of integers in the range 0 <= x < 256."
/// - `bytearray(<known bytes-like value>)` / `bytes(<known bytes-like
///   value>)` — `bytes_is_immutable`'s own `frozen = bytes(data)` row
///   (copying a `bytearray` into an immutable `bytes`, or vice versa):
///   the SAME known-list-of-known-Integers shape the row above reads,
///   since a `bytearray`/`bytes` value already IS that shape once
///   built — no separate reader needed.
/// - `memoryview(<known bytearray/bytes value>)` — `memoryview_over_
///   bytearray_reads`'s own row: a view SHARES the underlying buffer
///   (`memoryview(ba)[i]` reads/writes the same elements `ba[i]`
///   would), so this file answers the identical `Kind::List` value
///   unchanged rather than building a distinct wrapper shape — this
///   domain has no separate "view" Kind, and a plain copy-through is
///   sound for every read/len/index this corpus exercises (the
///   shared-buffer WRITE-back-through-the-view effect is check.rs's
///   own statement-sink business, not a value-construction concern).
///
/// Any other argument shape (zero arguments, more than one argument, a
/// non-Integer/out-of-range element, an unknown receiver) declines —
/// this function states nothing beyond the shapes listed above.
pub(in super::super) fn bytes_like_construction_value(
    constructor: &str,
    args: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let [only] = args else { return None };
    let argument = evaluate_expression(only, environment, kernel);
    if constructor == "memoryview" {
        if argument.kind == Kind::List {
            // a view SHARES the underlying buffer (module doc) — writing
            // through the view raises the memoryview-specific wording
            // regardless of which species the wrapped argument itself
            // carried, so this re-tags rather than keeping the argument's
            // own word (`bytes_models::tagged`'s own doc).
            return Some(bytes_models::tagged(argument, bytes_models::MEMORYVIEW_WORD));
        }
        return None;
    }
    if constructor == "bytearray" {
        if let Some((length, PrimitiveKind::Integer)) = single_numeric_value(&argument) {
            if (0.0..=1024.0).contains(&length) {
                let zeroes = vec![0u8; length as usize];
                return Some(bytes_models::tagged(
                    bytes_models::bytes_literal_value(&zeroes),
                    bytes_models::BYTEARRAY_WORD,
                ));
            }
            return None;
        }
    }
    let bytes = known_byte_sequence(&argument)?;
    let word = if constructor == "bytearray" {
        bytes_models::BYTEARRAY_WORD
    } else {
        bytes_models::BYTES_WORD
    };
    Some(bytes_models::tagged(bytes_models::bytes_literal_value(&bytes), word))
}

/// `array.array('d', [...])` — the Float64Array twin,
/// p-typed-array.py's `array_double_from_iterable`/`array_double_
/// write_and_read_back`: `array.rst`'s own `class:: array(typecode[,
/// initializer])`, typecode `'d'` (double). Modeled ONLY for the exact
/// two-argument form with a known exact-string typecode `"d"` and a
/// known list of known numeric (Integer or Float) elements — every
/// element widens to Float on read (`bytes_models::
/// array_double_literal_value`'s own doc: an `array.array('d', ...)`
/// element is ALWAYS a Python `float`, whatever numeric literal built
/// it). Any other typecode, arity, or a non-numeric element declines —
/// this file models the one typecode the corpus's own Float64Array-twin
/// rows use.
pub(in super::super) fn array_double_construction_value(
    call: &ruff_python_ast::ExprCall,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [typecode_expr, initializer_expr] = &*call.arguments.args else {
        return None;
    };
    let typecode = evaluate_expression(typecode_expr, environment, kernel);
    let typecode_text = exact_string_values(&typecode).and_then(code_points_to_string)?;
    if typecode_text != "d" {
        return None;
    }
    let initializer = evaluate_expression(initializer_expr, environment, kernel);
    if initializer.kind != Kind::List {
        return None;
    }
    let elements: Vec<f64> = initializer
        .items
        .iter()
        .map(|item| single_numeric_value(item).map(|(value, _sort)| value))
        .collect::<Option<Vec<f64>>>()?;
    Some(bytes_models::array_double_literal_value(&elements))
}

/// Every `from math import inf/nan/pi/e/tau[ as x]` local name, bound to
/// the SAME CONCRETE VALUE `math_models::math_constant_value` answers
/// for the attribute spelling — read once from the module's own
/// top-level `from … import …` statements, the same "read the module's
/// import statements once, resolve by canonical identity" shape
/// `datetime_imports` below already keeps for the `datetime` module
/// family. `import math` (the plain module import, no `from`) needs no
/// entry here: `math.inf`/`math.pi`/… already resolve through the
/// ATTRIBUTE arm (`evaluate_attribute`'s own `math_constant_value` call)
/// regardless of this table, since that arm reads the literal `math`
/// module name directly. Only `from math import <name>` strips the
/// `math.` qualifier entirely, leaving a bare `Name` node with no module
/// to route through — this table is what lets `evaluate_expression`'s
/// bare-name arm resolve `inf`/`nan`/`pi`/`e`/`tau` after that import,
/// the same gap `bind_or_forget_imported_name` closes for `datetime`'s
/// own from-imports via `context.module_bindings`. Built once per module
/// (`math_from_imports`) and merged into that same `module_bindings`
/// table at `check.rs`'s three `WalkContext` construction sites, so a
/// module-level import statement, `bind_or_forget_imported_name`'s
/// existing walk, and `module_scope_environment`'s existing seed all
/// pick this up with no new `Environment` field or thread of their own.
/// A rebound name (`inf = 200`) still shadows correctly: `module_
/// bindings` only seeds where `environment.alias_is_visible` allows, the
/// same rule every other module-level binding already obeys.
pub(crate) fn math_from_imports(module: &ModModule) -> HashMap<String, AbstractValue> {
    let mut table = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::ImportFrom(import) = stmt else {
            continue;
        };
        let Some(source) = import.module.as_ref() else {
            continue;
        };
        if source.id.as_str() != "math" || import.level != 0 {
            continue;
        }
        for alias in &import.names {
            let Some(value) = math_models::math_constant_value(alias.name.id.as_str()) else {
                continue;
            };
            let local = alias.asname.as_ref().unwrap_or(&alias.name);
            table.insert(local.id.as_str().to_owned(), value);
        }
    }
    table
}

/// Whether `expr` is exactly `datetime.timezone.utc` or `datetime.UTC`
/// — the two spellings datetime.rst documents for the UTC singleton
/// (`datetime_construction_value`'s own doc). Read SYNTACTICALLY (the
/// expression's own dotted-name shape), never by evaluating to an
/// AbstractValue — this file tracks no tzinfo value at all.
pub(in super::super) fn is_utc_tzinfo_expression(expr: &Expr) -> bool {
    // datetime.UTC — a two-level chain, `Name("datetime").UTC`
    if let Expr::Attribute(outer) = expr {
        if outer.attr.as_str() == "UTC" {
            if let Expr::Name(name) = outer.value.as_ref() {
                if name.id.as_str() == "datetime" {
                    return true;
                }
            }
        }
        if outer.attr.as_str() == "utc" {
            // datetime.timezone.utc — a three-level chain,
            // `Name("datetime").timezone.utc`
            if let Expr::Attribute(middle) = outer.value.as_ref() {
                if middle.attr.as_str() == "timezone" {
                    if let Expr::Name(name) = middle.value.as_ref() {
                        if name.id.as_str() == "datetime" {
                            return true;
                        }
                    }
                }
            }
            // `timezone.utc` — a two-level chain, `Name("timezone").utc`,
            // the shape `from datetime import timezone` gives (showcase.py's
            // own spelling); recognized by bare name only, the same
            // no-import-identity convention this function already takes
            // for `datetime.UTC`/`datetime.timezone.utc`.
            if let Expr::Name(name) = outer.value.as_ref() {
                if name.id.as_str() == "timezone" {
                    return true;
                }
            }
        }
    }
    false
}
