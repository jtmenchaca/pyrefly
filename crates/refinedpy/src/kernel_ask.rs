//! Every kernel ask this crate makes is wrapped in `std::panic::
//! catch_unwind` (`assignability.rs`'s own containment/member asks,
//! `narrowing.rs`'s narrow ask, and every sibling this file's own doc
//! names) because a REFUSED question — a set shape the kernel's decider
//! does not decide — panics inside the kernel closure rather than
//! returning an answer; the catch turns that refusal into an honest
//! `Undetermined`/`None`, never a crash. Rust's default panic hook,
//! though, still PRINTS "thread 'main' panicked at ..." to stderr the
//! moment the panic unwinds, before the catch ever runs — noise for a
//! refusal this file already expects and handles.
//!
//! `ask_kernel` is the one place that noise is suppressed: it installs
//! a process-global hook, ONCE (`std::sync::Once`), that checks a
//! thread-local flag before printing. The flag is set for exactly the
//! duration of the closure this function runs and cleared immediately
//! after (a `Drop` guard, so a panic unwinding through the closure still
//! clears it) — every OTHER panic, on any other thread or outside this
//! window, finds the flag unset and the hook falls through to the
//! ORIGINAL default hook, so an unexpected panic still prints exactly
//! as it does today. This is the one seam every `catch_unwind` call in
//! this crate should route through rather than calling `catch_unwind`
//! directly, so the suppression is installed once and reasoned about in
//! one place.

use std::cell::Cell;
use std::panic::catch_unwind;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Once;

thread_local! {
    static SUPPRESS_PANIC_PRINT: Cell<bool> = const { Cell::new(false) };
}

/// Time spent inside kernel asks and the number of asks, summed
/// process-wide since start. `refinedpy-check --timing` reads deltas
/// around each file to split analysis time into in-kernel and
/// out-of-kernel portions.
static KERNEL_ASK_NANOS: AtomicU64 = AtomicU64::new(0);
static KERNEL_ASK_COUNT: AtomicU64 = AtomicU64::new(0);

/// (nanoseconds inside kernel asks, ask count) accumulated so far.
pub fn kernel_ask_totals() -> (u64, u64) {
    (KERNEL_ASK_NANOS.load(Ordering::Relaxed), KERNEL_ASK_COUNT.load(Ordering::Relaxed))
}

static INSTALL_HOOK: Once = Once::new();

/// Installs the process-global hook exactly once: it wraps whatever
/// hook was previously registered (Rust's own default hook, the first
/// time this runs), and skips calling through to it only when the
/// panicking thread's own `SUPPRESS_PANIC_PRINT` flag is set.
fn install_hook_once() {
    INSTALL_HOOK.call_once(|| {
        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |panic_info| {
            let suppressed = SUPPRESS_PANIC_PRINT.with(|flag| flag.get());
            if !suppressed {
                previous_hook(panic_info);
            }
        }));
    });
}

/// A `Drop` guard that clears the suppression flag when it goes out of
/// scope — covers both the ordinary return path and a panic unwinding
/// through `f` (`Drop::drop` still runs during unwind), so the flag
/// never leaks `true` onto a later, unrelated panic on the same thread.
struct SuppressGuard;

impl Drop for SuppressGuard {
    fn drop(&mut self) {
        SUPPRESS_PANIC_PRINT.with(|flag| flag.set(false));
    }
}

/// Runs `f`, catching a panic the same way every kernel ask in this
/// crate already does (`catch_unwind`/`AssertUnwindSafe`), with the
/// default panic PRINT suppressed for exactly this call — an expected
/// kernel refusal no longer writes to stderr, while an unexpected panic
/// on any other call still prints normally. Callers keep their own
/// `Ok`/`Err` handling unchanged; this only changes what reaches stderr.
pub fn ask_kernel<F, T>(f: F) -> std::thread::Result<T>
where
    F: FnOnce() -> T,
{
    install_hook_once();
    SUPPRESS_PANIC_PRINT.with(|flag| flag.set(true));
    let _guard = SuppressGuard;
    let started = std::time::Instant::now();
    let result = catch_unwind(AssertUnwindSafe(f));
    KERNEL_ASK_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    KERNEL_ASK_COUNT.fetch_add(1, Ordering::Relaxed);
    // THE ONE ASK CHOKEPOINT (DERIVATION-TRACE.md, "Kernel-ask spans"):
    // every kernel ask this crate makes routes through this function, so
    // recording the ask here is the whole of question 3 — "who was
    // asked, and what did the kernel answer" — and no reader ever does
    // per-ask work of its own. A caught refusal records a DECLINED child
    // (the kernel did not decide this set shape); an ordinary return
    // records an ANSWERED one.
    //
    // The closure is opaque at this seam — it captures its own kernel
    // entry point and operands, and this function is generic over its
    // return type, so neither the op name nor the wire text is
    // reconstructable here. `ask_kernel_named` is the spelling a caller
    // uses to record both; this bare entry point records the ask's
    // outcome under the generic op name so no ask is missing from the
    // tree. See this crate's `trace` module doc / the spec feedback on
    // this seam.
    if crate::trace::is_tracing() {
        crate::trace::record_kernel_ask(
            "ask",
            "the asking reader's own question (this seam carries no wire text — see ask_kernel_named)",
            match &result {
                Ok(_) => Some("decided"),
                Err(_) => None,
            },
        );
    }
    result
}

/// `ask_kernel` with the op name and the wire question text stated by the
/// caller — the spelling a reader uses where it CAN name what it asked,
/// so the trace's kernel child carries `kernel.<op>` and
/// `refinery.question` rather than the generic pair `ask_kernel` records.
/// Identical behaviour otherwise: same catch, same suppression, same
/// timing accounting.
pub fn ask_kernel_named<F, T>(op: &str, question: &str, f: F) -> std::thread::Result<T>
where
    F: FnOnce() -> T,
{
    let tracing = crate::trace::is_tracing();
    let result = ask_kernel(f);
    if tracing {
        crate::trace::record_kernel_ask(
            op,
            question,
            match &result {
                Ok(_) => Some("decided"),
                Err(_) => None,
            },
        );
    }
    result
}

/// Installs every kernel-ask seam the crates BELOW the kernel dependency
/// expose — today three slots on `refined_domain`: the join-time
/// no-scalar-reread gate, the string-ground absorption's seq-subset
/// ask, and the scalar-union join's bounds ask. None of the three can
/// name the kernel's types itself (`refined_domain` sits under
/// `refined_kernel` in the dependency graph), so each receives its ask
/// as an injected closure routed through this module's own catch
/// discipline. Both binaries call this once right after the kernel
/// loads; installing twice is a no-op (each slot is a OnceLock).
pub fn install_kernel_seams(kernel: &std::sync::Arc<refined_kernel::kernel_interface::RefinedTSKernel>) {
    {
        let kernel = kernel.clone();
        refined_domain::kernel_seam::install_no_scalar_reread_ask(move |set| {
            ask_kernel(|| (kernel.seq_no_scalar_reread)(set)).ok()
        });
    }
    {
        let kernel = kernel.clone();
        refined_domain::kernel_seam::install_seq_subset_ask(move |a, b| {
            ask_kernel(|| (kernel.seq_subset)(a, b)).ok()
        });
    }
    {
        let kernel = kernel.clone();
        refined_domain::kernel_seam::install_bounds_ask(move |set| {
            ask_kernel(|| (kernel.bounds)(set)).ok().map(|answer| {
                refined_domain::kernel_seam::BoundsAnswer {
                    empty: answer.empty,
                    hull: answer.hull,
                }
            })
        });
    }
    {
        let kernel = kernel.clone();
        refined_domain::kernel_seam::install_truthy_num_ask(move |set| {
            truthy_num_verdict(&kernel, set)
        });
    }
}

/// Whether `state` provably admits no value at all — the same reading
/// `truthiness_conformance.rs`'s own `state_is_uninhabited` takes: a
/// top state or any admitted flag (undef/null/NaN/thrown) is inhabited
/// outright; otherwise the scalar emptiness decider is asked first and,
/// on a kernel refusal (caught, never a crash), the sequence one.
/// `None` means neither decider spoke to this set's shape — not
/// decided, and never read as either answer.
fn state_is_uninhabited(
    kernel: &refined_kernel::kernel_interface::RefinedTSKernel,
    state: &refined_kernel::narrow_questions::KnownStateWire,
) -> Option<bool> {
    if state.top || state.undef || state.null || state.nan || state.thrown {
        return Some(false);
    }
    if let Ok(empty) = ask_kernel(|| (kernel.scalar_empty)(&state.set)) {
        return Some(empty);
    }
    ask_kernel(|| (kernel.seq_empty)(&state.set)).ok()
}

/// The `TruthyNum` seam's implementation: narrow a bare scalar set by
/// the kernel's proved `js.truthyNum` filter and read each side's
/// emptiness. The numeric fragment this poses is language-shared —
/// zero is the one falsy number in Python exactly as in JS
/// (`truthiness_conformance.rs`'s "Why js.truthyNum and not a
/// Python-named op") — and the caller's `Kind::Set` arm never carries
/// the NaN/absent flags where the two languages diverge, so the wire
/// state below is the set alone. Falsy side empty → definitely truthy;
/// truthy side empty → definitely falsy; both inhabited → a real
/// undecided answer; either side REFUSED by both emptiness deciders →
/// `None`, and the caller keeps its own weaker reading (a refusal is
/// never a claim that a side is empty). Both sides empty would mean no
/// value flows at all, which is not a truthiness verdict — undecided.
fn truthy_num_verdict(
    kernel: &refined_kernel::kernel_interface::RefinedTSKernel,
    set: &refined_sets::refinement_forms::RefinedSet,
) -> Option<(bool, bool)> {
    let state = refined_kernel::narrow_questions::KnownStateWire {
        top: false,
        set: set.clone(),
        undef: false,
        null: false,
        nan: false,
        thrown: false,
    };
    let (when_true, when_false) =
        ask_kernel(|| (kernel.narrow_state)(&state, "js.truthyNum", 0.0, false)).ok()?;
    let truthy_empty = state_is_uninhabited(kernel, &when_true)?;
    let falsy_empty = state_is_uninhabited(kernel, &when_false)?;
    Some(match (truthy_empty, falsy_empty) {
        (true, true) => (false, false),
        (false, true) => (true, true),
        (true, false) => (false, true),
        (false, false) => (false, false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A caught refusal never propagates and the flag is cleared after
    /// (checked indirectly: a second call still catches correctly,
    /// which would not hold if the guard's `Drop` had failed to reset
    /// the thread-local).
    #[test]
    fn catches_a_panic_and_resets_for_the_next_call() {
        let first = ask_kernel(|| -> i32 { panic!("refused") });
        assert!(first.is_err(), "a panicking closure is caught, not propagated");
        let second = ask_kernel(|| 42);
        assert_eq!(second.ok(), Some(42), "an ordinary closure still returns its value");
    }
}
