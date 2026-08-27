//! Module-qualified stdlib builtins answered from this crate:
//! `time.time`, `time.monotonic_ns`, `os.open`/`os.close`,
//! `copy.copy`/`copy.deepcopy`. Every row cites its clause of
//! docs.python.org/3.12/library/time.html, library/os.html, or
//! library/copy.html; a row with no citation is not written.

use refined_domain::abstract_value::{null_value, AbstractValue, Kind, PrimitiveKind, SetKindTag};
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
    if !arguments.is_empty() {
        return None;
    }
    match function {
        "time" => Some(nonnegative_ground(PrimitiveKind::Float)),
        // `time.monotonic_ns()` — library/time.html#time.monotonic_ns:
        // "Similar to monotonic(), but return time as nanoseconds." §"Clock
        // ID Constants" and monotonic()'s own doc: "The reference point of
        // the returned value is undefined, so that only the difference
        // between the results of two calls is valid" and "the value...is
        // undefined, so only the difference between the results of
        // consecutive calls is valid" — a single reading carries no
        // identity claim (never a specific instant), but its own type is
        // pinned: "Return the value... in nanoseconds" is always a whole
        // `int`. This crate reads the monotonicity guarantee itself (each
        // reading nondecreasing across the SAME clock) as the sound but
        // weaker per-call ground `[0, +inf)`, Integer-sorted — the same
        // NONNEGATIVE, unspecific-instant shape `time.time()`'s own row
        // states for the wall clock, restricted to whole nanoseconds
        // rather than a float second count.
        "monotonic_ns" => Some(nonnegative_ground(PrimitiveKind::Integer)),
        _ => None,
    }
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
/// `copy.copy(x)` / `copy.deepcopy(x)` — library/copy.rst: "Return a
/// shallow copy of *x*" and "Return a deep copy of *x*." The module's own
/// opening paragraph states the difference the two make: "A shallow copy
/// constructs a new compound object and then... inserts references into
/// it to the objects found in the original. A deep copy constructs a new
/// compound object and then... inserts copies into it of the objects
/// found in the original." Either way the copy holds the SAME VALUES the
/// original held — copying is defined to preserve contents, and the two
/// spellings differ only in whether the NESTED objects are shared or
/// themselves copied, never in which values are present. This domain
/// reads a container by its contents, so both rows answer the argument's
/// own value.
///
/// The copy is a DIFFERENT referent from the original (copy.rst's own
/// "constructs a new compound object"), so a `Kind::List`/`Kind::Object`
/// answer takes a FRESH `instance_identity` — the same per-referent tag
/// `with_referent_identities` already stamps on a container built into a
/// list literal, and the fact `d is cloned` must answer False on. A
/// scalar argument carries no referent identity to refresh and copies
/// through as itself, matching copy.rst's own note that the two functions
/// return immutable atomic values unchanged.
///
/// `None` for any argument this file cannot read at all (an Unknown) —
/// copying an unread value states nothing new about it.
pub(super) fn copy_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if !matches!(function, "copy" | "deepcopy") {
        return None;
    }
    let [only] = arguments else { return None };
    if only.kind == Kind::Unknown {
        return None;
    }
    if !matches!(only.kind, Kind::List | Kind::Object) {
        return Some(only.clone());
    }
    let copied = crate::collection_models::with_referent_identities(vec![AbstractValue {
        instance_identity: None,
        ..only.clone()
    }]);
    copied.into_iter().next()
}

pub(super) fn os_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    match function {
        "open" if arguments.len() == 2 => Some(nonnegative_ground(PrimitiveKind::Integer)),
        "close" if arguments.len() == 1 => Some(null_value()),
        // `os.listdir(path='.')` — library/os.rst, `function:: listdir`:
        // "Return a list containing the names of the entries in the
        // directory given by *path*. The list is in arbitrary order..."
        // and, on the element type: "*path* may be a path-like object.
        // If *path* is of type ``bytes``... the filenames returned will
        // also be of type ``bytes``; in all other circumstances, they
        // will be of type ``str``."
        //
        // The names come from the filesystem, so no content is known —
        // every element is `Σ*`, the whole-strings ground. The COUNT is
        // likewise unstated (the same clause makes even the membership
        // of a concurrently-changed entry "unspecified"), so the length
        // window is unbounded from zero: an empty directory listing is
        // an empty list. The answer is the unbounded repetition of
        // unread strings — the same shape `attribute.rs`'s `sys.argv`
        // read answers for its own external-origin sequence — so
        // `os.listdir(d)[0]` reads `Σ*` through `subscript_read`'s own
        // `star_element_read` rather than declining.
        //
        // The BYTES half of the element-type clause is not this row: it
        // fires only for a `bytes` *path*, and this row is written for
        // the `str`-path circumstance the clause states covers "all
        // other circumstances." So the argument must be provably
        // string-shaped — an exact string, or a `Kind::Set` whose forms
        // carry a sequence shape (the same `states_sequence`/
        // `sequence_shaped` test `attribute_call.rs` gates its own
        // string-method sort-only rows on, never a second recognizer for
        // the same question). A bytes or unread path declines rather
        // than claim `Σ*` elements the bytes case would not have.
        "listdir" if arguments.len() <= 1 => {
            if let [path] = arguments {
                let path_is_string_shaped = path.kind_tag == Some(PrimitiveKind::String)
                    || (path.kind == Kind::Set
                        && (crate::assignability::states_sequence(&path.set) || crate::assignability::sequence_shaped(&path.set)));
                if !path_is_string_shaped {
                    return None;
                }
            }
            Some(known_set(
                refined_sets::repetition_window_forms::repetition(refined_sets::codepoint_sets::strings(), 0, None),
                None,
                TrustSpec,
                SetKindTag::None,
            ))
        }
        _ => None,
    }
}
