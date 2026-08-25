//! Module-qualified stdlib builtins answered from this crate:
//! `time.time`, `os.open`/`os.close`. Every row cites its clause of
//! docs.python.org/3.12/library/time.html or library/os.html; a row
//! with no citation is not written.

use refined_domain::abstract_value::{null_value, AbstractValue, PrimitiveKind, SetKindTag};
use refined_domain::abstract_value::known_set;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::{at_least, make_refined_set};

/// An unbounded, NONNEGATIVE numeric ground — the shared answer shape
/// `time_call_result`'s `time.time` row and `os_call_result`'s
/// `os.open` row both state (a value known only to sit in `[0, +inf)`,
/// tagged `sort`). Composed once here rather than duplicated at each
/// call site. `PrimitiveKind::Integer` additionally carries the
/// `integer()` refinement form in the SET itself, not just the
/// `kind_tag` sort marker — without it, a caller's own guard (e.g.
/// `0 <= fd <= 150`) narrows the range but the narrowed set stays
/// bare `[0, 150]` with no integrality, which fails assignment against
/// a declared alias requiring `integer` (A15.xfer.handle's own
/// `os.open` row). `PrimitiveKind::Float` (`time.time`) never adds
/// this form — a float ground is not integer-valued.
fn nonnegative_ground(sort: PrimitiveKind) -> AbstractValue {
    let mut forms = vec![at_least(0.0)];
    if sort == PrimitiveKind::Integer {
        forms.push(refined_sets::refinement_forms::integer());
    }
    AbstractValue {
        kind_tag: Some(sort),
        ..known_set(make_refined_set(forms), None, TrustSpec, SetKindTag::None)
    }
}

/// `time.time()` — library/time.html#time.time: "Return the time in
/// seconds since the epoch as a floating-point number... Note that even
/// though the time is always returned as a floating-point number, not
/// all systems provide time with a better precision than 1 second." The
/// epoch itself is defined as 1970-01-01 00:00:00 (UTC) on every
/// platform this doc covers, so the returned value is always
/// NONNEGATIVE — this row states exactly that ground: `[0, +inf)`,
/// Float-sorted, never a specific instant (the running clock is not a
/// fact this domain reads). Zero-argument only, per the doc's own
/// signature.
pub(super) fn time_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "time" || !arguments.is_empty() {
        return None;
    }
    Some(nonnegative_ground(PrimitiveKind::Float))
}

/// `os.open(path, flags)` / `os.close(fd)` — library/os.html:
/// `os.open`: "Return a file descriptor... to be used by other
/// low-level (i.e. os.read()) file operations." A file descriptor is
/// always a NONNEGATIVE `int` (`os.rst`'s own examples index only ever
/// nonnegative values, and CPython raises `OSError` rather than ever
/// returning a negative fd) — this row states the ground `[0, +inf)`,
/// Integer-sorted, never a specific descriptor number, matching
/// A15.xfer.handle's own claim ("a file descriptor opened fresh...
/// carries no identity claim"). `os.close`: "Close file descriptor
/// *fd*... Availability: not Emscripten, not WASI." No return value —
/// CPython's own `os.close` always returns `None`, so this row answers
/// the domain's exact absent state (never Unknown) for ANY single
/// argument, matching a Python function whose only documented effect is
/// closing the descriptor.
pub(super) fn os_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "open" if arguments.len() == 2 => Some(nonnegative_ground(PrimitiveKind::Integer)),
        "close" if arguments.len() == 1 => Some(null_value()),
        _ => None,
    }
}
