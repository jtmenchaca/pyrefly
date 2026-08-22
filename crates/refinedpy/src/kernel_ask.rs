/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

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
use std::sync::Once;

thread_local! {
    static SUPPRESS_PANIC_PRINT: Cell<bool> = const { Cell::new(false) };
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
    catch_unwind(AssertUnwindSafe(f))
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
