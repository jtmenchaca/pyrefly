//! `match` statement arm resolution: given a subject with a KNOWN value
//! state, decide which arm a case's pattern (plus its guard) takes, and
//! what names that arm binds. CPython 3.12 semantics
//! (reference/compound_stmts.html#the-match-statement):
//!
//! - A literal/value pattern (`MatchValue`) compares with `==`.
//! - A singleton pattern (`MatchSingleton`, i.e. `True`/`False`/`None`)
//!   compares with `is` — "For the singletons `None`, `True` and
//!   `False`, the `is` operator is used." A subject that is
//!   Boolean-tagged 1.0 IS `True`; a subject that is Number-tagged 1 is
//!   NOT `True` (identity, not equality) — the fact AGENT-BRIEF.md
//!   states as "subject 1 falls through `case True:` but takes
//!   `case True | 1:` via the value alternative."
//! - A capture pattern (bare `case x:`) "always succeeds," binding the
//!   name to the subject.
//! - A wildcard `case _:` "always succeeds... and binds no name."
//! - An OR pattern "matches each of its subpatterns in turn... until
//!   one succeeds" — first Taken wins, left to right.
//! - A guard runs only after its pattern succeeds; "If the guard
//!   condition evaluates as false, the case block is not selected" and
//!   matching continues to the next case.
//!
//! Sequence/Mapping/Class patterns are Undecidable for TAKEN/NOT-TAKEN
//! this wave (`pattern_outcome`, in `outcome.rs`) — deciding which arm
//! runs would need a structural equality/length/key-presence question
//! this file does not ask yet. Their CAPTURES, though, are nameable and
//! (for a known List/Object subject) provable: `pattern_captures` names
//! every bare-Name/star element a sequence pattern binds, every
//! literal-key Name value (plus an optional `**rest`) a mapping pattern
//! binds, and every keyword OR positional sub-pattern Name a class
//! pattern binds. `pattern_bound_captures` reads the actual
//! element/key/field value off a KNOWN List/Object subject when one is
//! available. A class pattern's POSITIONAL sub-patterns (`Point(px,
//! py)`) resolve through the class's own `__match_args__` order
//! (`ClassModel.fields`, `class_pattern_fields`'s own doc) when a class
//! table is available; a keyword sub-pattern needs no such lookup,
//! since the keyword's own `attr` IS the field name.
//!
//! `PrimitiveKind` carries `Integer`/`Float` tags, but nothing in this
//! package's expression evaluator (`expressions.rs`) emits them yet —
//! every numeric literal and arithmetic result reads as
//! `PrimitiveKind::Number` (`Kind::Values` with one `f64`); only a
//! boolean literal reads as `PrimitiveKind::Boolean` (`true` as `1.0`,
//! `false` as `0.0`, matching CPython's `bool` being an `int`
//! subclass). Singleton identity in this file is decided off `kind_tag`
//! + the value: only a Boolean-tagged 1.0/0.0 subject IS `True`/`False`,
//! and only `Kind::Null` IS `None`. When a producer starts tagging
//! `Integer`/`Float` on the values this file reads, `subject_is_singleton`
//! is the one place that gains a new arm — every other function here
//! goes through it rather than re-deriving the identity check.
//!
//! A `MatchValue`/`MatchOr` subject is not always one known scalar.
//! `enumerable_subject_members` reads the admitted numeric members off
//! THREE subject shapes: a multi-valued `Kind::Values` (`{1, 2, 4}`
//! read directly off `subject.values`); a `Kind::Set` that enumerates a
//! union-of-singleton-scalars form (`scalars_of_union_of_singletons`,
//! `collection_models.rs`'s own reader for exactly this shape, reused
//! rather than re-parsed); and, per arm, a `Kind::KindUnion`'s own
//! Values-kind arms. `match_value_outcome` then asks MEMBERSHIP rather
//! than the single-value equality it used to: a pattern literal that IS
//! a member is Taken, one that is NOT is NotTaken (a dead arm — the
//! same NotTaken every other unreachable arm answers, never a new
//! label), and a subject this reading cannot enumerate stays
//! Undecidable exactly as before. `pattern_outcome`'s own `Kind::KindUnion`
//! arm judges the pattern against EACH arm through this same recursive
//! core (mirroring `assignability.rs`'s KindUnion judge: a Fire/Taken
//! arm decides, an Undetermined/Undecidable arm poisons the whole
//! union, and the union is NotTaken only when every arm is) — the same
//! "apply per arm, keep what the pattern admits" reading
//! `narrow_isinstance_call`'s own KindUnion filter (`narrowing.rs`)
//! already uses for `isinstance`, applied here to `match`.

mod captures;
mod outcome;
mod value_proof;
mod values;
mod walk;

#[cfg(test)]
mod tests;

pub use captures::pattern_bound_captures;
pub use captures::pattern_captures;
pub use outcome::arm_outcome;
pub use outcome::ArmOutcome;
pub use value_proof::pattern_proved_value;
pub use walk::match_taken_environment;

// Test module is a sibling of the domain children, so re-export their
// items into this module's namespace for `tests`'s `use super::*`.
#[cfg(test)]
use std::sync::Arc;

#[cfg(test)]
use refined_domain::abstract_value::AbstractValue;
#[cfg(test)]
use refined_domain::abstract_value::Kind;
#[cfg(test)]
use refined_domain::abstract_value::PrimitiveKind;
#[cfg(test)]
use refined_kernel::kernel_interface::RefinedTSKernel;
#[cfg(test)]
use ruff_python_ast::Expr;
#[cfg(test)]
use ruff_python_ast::MatchCase;
#[cfg(test)]
use ruff_python_ast::Pattern;
#[cfg(test)]
use ruff_python_ast::Singleton;
#[cfg(test)]
use ruff_python_ast::Stmt;

#[cfg(test)]
use crate::env::Environment;
#[cfg(test)]
use crate::instances::ClassModel;

#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
pub(self) use values::bare_capture_name;
#[cfg(test)]
pub(self) use values::guarded_bare_capture_narrowed;
#[cfg(test)]
pub(self) use values::narrow_scalar_subject;
