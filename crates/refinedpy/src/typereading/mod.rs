//! Annotation expressions read into declared refinements: alias names
//! (only where visible), inline `Annotated[...]` forms, alias-of-alias,
//! `X | None` / `Optional[X]` (admits-None wrapping the inner read),
//! and string annotations. This file is the contract the walk calls;
//! the typereading unit fills it in behind these signatures.
//!
//! A None never approximates a set — it declines to state one, and the
//! walk decides what silence says at the call site. Nothing here widens
//! or guesses a set it cannot read exactly (the same discipline as
//! refined-ts-go's typereading package: refuse rather than approximate).

mod base_sort;
mod callable;
mod declared_refinement;
mod generator;
mod literal_members;
mod typed_dict;

#[cfg(test)]
mod tests;

pub use base_sort::base_sort_return_refinement;
pub use callable::callable_return_refinement;
pub use declared_refinement::declared_refinement;
pub use declared_refinement::DeclaredRefinement;
pub use declared_refinement::GeneratorRefinement;
pub use typed_dict::typed_dict_return_refinement;

// Test module is a sibling of the domain children, so re-export the
// items its `use super::*;` needs into this module's namespace.
#[cfg(test)]
use std::collections::HashMap;

#[cfg(test)]
use ruff_python_ast::Expr;
#[cfg(test)]
use ruff_python_parser::parse_expression;

#[cfg(test)]
use crate::env::Environment;
#[cfg(test)]
use crate::surface::AliasEntry;
#[cfg(test)]
use crate::surface::SurfaceImports;
