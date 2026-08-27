//! Classes as readable data: a class's declared fields (in declaration
//! order), judged construction against those fields, and instance
//! attribute reads/writes. One model covers every AnnAssign-fielded
//! class this checker reads — a self-authored class, a `@dataclass`,
//! and a pydantic `BaseModel` subclass all declare their fields the
//! same way (`name: Annotation [= default]` in the class body), so
//! there is one `ClassModel`, not a class family per framework.
//!
//! Field order is declaration order: pydantic v2 auto-generates
//! `__match_args__` in field-declaration order (AGENT-BRIEF.md,
//! "Environment facts" — "pydantic v2 `BaseModel` auto-generates
//! `__match_args__` in field-declaration order"), and a dataclass's
//! generated `__init__` binds positional arguments in the same order
//! its fields were declared (tmp/cpython/Doc/library/dataclasses.rst
//! is not present in this wave's read set beyond that one AGENT-BRIEF
//! fact; the positional-parameter-order claim for dataclasses is
//! standard `@dataclass` behavior and is flagged unverified against
//! the vendored tree in this file's owning report).


mod model;
mod construction;
mod fields;
mod methods;
mod generator;

#[cfg(test)]
mod tests;

pub use construction::{class_object_value, judge_construction, ConstructionKind, ConstructionVerdict};
pub use fields::{
    field_read, field_read_through_model, field_write, field_write_judgment, self_attribute_name,
};
pub use generator::generator_yields;
pub use methods::{method_call_result, method_def_of};
pub use model::{
    class_table, model_members_refinement, typed_dict_table, ClassField, ClassModel, PropertyModel,
};
