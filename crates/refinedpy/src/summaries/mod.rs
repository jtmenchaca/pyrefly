/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A same-module `def`'s answer for one call: concrete evaluation of a
//! BOUNDED body — the same posture `loops.rs`'s `run_restricted_body`
//! takes for loop bodies, extended to the restricted statement forms a
//! function body needs (branching and `return`, which a loop body never
//! has). `call_result` binds the callee's parameters to the caller's
//! argument values, interprets the body statements it recognizes, and
//! answers the join of every value the body could return — or declines
//! (`None`) the moment the body does something this file does not
//! interpret, so a caller never gets a guessed answer.
//!
//! This is the a-statements:399-404 seam: `helper_never_answers_none`
//! returns a dict literal on both the `if` arm and the fall-through —
//! `{"age": 40}` and `{"age": 10}`. Once `expressions.rs` evaluates
//! dict literals, this file's `if`/`else` handling joins those two
//! Object values into one Object answer that is never `Kind::Null`,
//! which is exactly what lets the walk prove `held is None` false at
//! `none_test_on_helper_that_never_answers_none`'s call site.
//!
//! Keyword arguments are the WIRING owner's job: `call_result` takes
//! only POSITIONAL argument values, in parameter order. A caller with a
//! keyword call maps each keyword to its parameter's position before
//! calling this function; this file has no keyword-name matching of
//! its own.
//!
//! `interpret_assign`/`interpret_aug_assign` also recognize a
//! `self.<field> = <expr>` / `self.<field> += <expr>` target: when
//! `self` is bound to a known instance (only true inside
//! `instances::method_call_result`'s own environment, never inside an
//! ordinary `call_result`), the write updates the WORKING instance
//! through `instances::field_write` and rebinds `self` so a later
//! `self.<field>` read in the same body sees it. This is the one seam
//! `instances.rs`'s method interpreter shares with this file's
//! restricted body walk, rather than duplicating `interpret_body`'s
//! statement dispatch.

mod call_result;
mod compile;
mod effects;
mod interpret;
mod seed;
mod sorts;

#[cfg(test)]
mod tests;

pub use call_result::call_result;
pub use call_result::call_result_with_enclosing;
pub use call_result::CALL_DEPTH_CAP;
pub use effects::call_effects;
pub use sorts::declared_return_seed;
pub use sorts::iterable_element_sort;
pub use sorts::return_sort_fallback;

pub(crate) use interpret::collect_bound_names;
pub(crate) use interpret::interpret_body;
pub(crate) use seed::first_non_docstring_statement;
pub(crate) use seed::free_variable_snapshot;

// Private parent bindings so in-crate tests keep calling helpers by the
// unqualified names they used when this unit was one file.
#[cfg(test)]
use call_result::needs_enclosing_scope;
#[cfg(test)]
use compile::compiled_summary_for;
#[cfg(test)]
use compile::entry_state_of;
#[cfg(test)]
use compile::kernel_summary_result;
#[cfg(test)]
use compile::summary_key;
#[cfg(test)]
use compile::SUMMARY_REGISTRY;
#[cfg(test)]
use seed::is_stub_body;
#[cfg(test)]
use sorts::whole_integers;
