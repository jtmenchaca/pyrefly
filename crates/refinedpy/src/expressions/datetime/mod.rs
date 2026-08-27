//! `datetime`/`date`/`time`/`timedelta` and related calls: import
//! identity, construction, parsing, formatting, component reads, and
//! arithmetic on temporal values. Every sibling in `expressions/` reads
//! this module's whole surface through `use super::datetime::*;`, so
//! every item below is re-exported at that same `pub(in crate::
//! expressions)` scope regardless of which leaf file defines it.

mod arithmetic;
mod components;
mod construction;
mod formatting;
mod parsing;
mod retained_call;
mod subprocess;

pub(in crate::expressions) use arithmetic::*;
pub(in crate::expressions) use components::*;
pub(in crate::expressions) use construction::*;
pub(in crate::expressions) use formatting::*;
pub(in crate::expressions) use parsing::*;
pub(in crate::expressions) use retained_call::*;
pub(in crate::expressions) use subprocess::*;

// These four items carry wider visibility than the rest of this
// module's surface (`expressions/mod.rs` re-exports them `pub`/
// `pub(crate)` for crate-wide callers), so they need their own named
// re-export here — a glob re-export caps every item at the glob
// statement's own visibility, regardless of the item's declared one.
pub use arithmetic::binary_arithmetic_value_with_kernel;
pub use arithmetic::exact_instant_microseconds_of_expression;
pub use arithmetic::instant_stepped_by_microseconds;
pub use arithmetic::timedelta_microseconds_of_expression;
pub use arithmetic::utc_iso_microseconds;
pub(crate) use construction::datetime_imports;
pub(crate) use construction::module_never_calls_setlocale;
pub use construction::DatetimeImports;
