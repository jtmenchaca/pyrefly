//! The seam pyrefly's library calls through to reach RefinedPy, without
//! depending on the RefinedPy engine itself. `server.rs`'s four call
//! sites (kernel setup, diagnostics, save-time export, hover) call the
//! functions stored here rather than a `crate::lsp::non_wasm::refinedpy`
//! module directly. A binary that wants RefinedPy behavior calls
//! `register` once, before serving any request, with the four
//! implementations from `refinedpy_lsp::register_refinedpy_hooks`. A
//! binary that never registers gets the no-op defaults: pyrefly runs
//! exactly as it does with no RefinedPy dependency at all.

use std::sync::OnceLock;

use lsp_types::Diagnostic;
use lsp_types::Hover;
use pyrefly_build::handle::Handle;
use ruff_text_size::TextSize;

use crate::state::state::Transaction;

/// One function pointer per `server.rs` call site. Each matches that
/// site's exact call shape, so registering is a straight swap-in.
pub struct RefinementHooks {
    /// Resolves and remembers the kernel dylib path once, before the
    /// event loop starts (`lsp_loop`, `server.rs:1507`).
    pub configure_kernel_dylib: fn(),
    /// Appends RefinedPy diagnostics onto pyrefly's own for one open
    /// handle (`append_ide_specific_diagnostics`, `server.rs:3121`).
    pub append_refinedpy_diagnostics: fn(transaction: &Transaction<'_>, handle: &Handle, diagnostics: &mut Vec<Diagnostic>),
    /// Save-time fact export for one path (`did_save`, `server.rs:3697`).
    pub export_fact_on_save: fn(path: &std::path::Path),
    /// Splices RefinedPy's refinement spelling into a hover already built
    /// by the host (`server.rs:5251`).
    pub splice_refinedpy_hover: fn(transaction: &Transaction<'_>, handle: &Handle, position: TextSize, hover: &mut Hover),
}

/// Every hook as a no-op: the default before any binary registers.
const NOOP_HOOKS: RefinementHooks = RefinementHooks {
    configure_kernel_dylib: || {},
    append_refinedpy_diagnostics: |_, _, _| {},
    export_fact_on_save: |_| {},
    splice_refinedpy_hover: |_, _, _, _| {},
};

/// The process-wide registry. Unregistered call sites read `NOOP_HOOKS`.
static HOOKS: OnceLock<RefinementHooks> = OnceLock::new();

/// Installs `hooks` as the process-wide registry. Callable once, before
/// serving any request — a binary that wants RefinedPy behavior calls
/// this from its own `main` before `lsp_loop` ever runs. A second call
/// is a no-op: `OnceLock::set` cannot overwrite an already-set registry.
pub fn register(hooks: RefinementHooks) {
    let _ = HOOKS.set(hooks);
}

/// The registered hooks, or the no-op defaults when nothing registered.
pub fn hooks() -> &'static RefinementHooks {
    HOOKS.get_or_init(|| NOOP_HOOKS)
}
