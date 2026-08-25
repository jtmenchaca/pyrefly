//! The pydantic surface: `type X = Annotated[int, Field(ge=…, le=…)]`
//! aliases lowered to refined sets, one table (plan-v2 L7).
//!
//! The lowering walks the RAW annotation expression, never the host's
//! resolved `Type`: pydantic `Field`'s stub returns `Any`, so
//! `Type::Annotated`'s metadata slot holds the inferred return type
//! and the `ge`/`le` values are unrecoverable from it
//! (PYREFLY-API-NOTES.md §3).

mod aliases;
mod annotated_set;
mod imports;
mod literal_alias;
mod literals;
mod strict_int;
mod temporal;

#[cfg(test)]
mod tests;

pub use aliases::{compile_aliases, AliasEntry, TemporalAwareness};
pub use annotated_set::annotated_expression_set;
pub use imports::{surface_imports, SurfaceImports};
pub use literals::literal_number;
pub use strict_int::strict_int_alias_names;
pub use temporal::temporal_inline_annotation;
