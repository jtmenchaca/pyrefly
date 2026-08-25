// Container VALUE state tests, split by concern. Every sibling module
// carries `use super::*;` to reach the imports and shared fixture
// helpers gathered here — the same symbols `collection_models::mod.rs`
// re-exports under `#[cfg(test)] pub(self) use ...` for this module's
// own `use super::*`, spelled out explicitly since a sibling one level
// deeper cannot glob past this file to the grandparent.

use super::*;

pub use refined_domain::abstract_value::AbstractValue;
pub use refined_domain::abstract_value::Kind;
pub use refined_domain::abstract_value::PrimitiveKind;
pub use refined_domain::abstract_value::SetKindTag;
pub use refined_domain::abstract_value::known_set;
pub use refined_domain::abstract_value::known_values;
pub use refined_domain::abstract_value::null_value;
pub use refined_domain::abstract_value::unknown;
pub use refined_domain::known_constructors::known_object;
pub use refined_domain::lattice_operations::join_known;
pub use refined_domain::lattice_operations::set_of_known;
pub use refined_domain::trust_grades::TrustProved;
pub use refined_kernel::kernel_interface::KnownStateWire;
pub use refined_sets::refinement_forms::at_least;
pub use refined_sets::refinement_forms::at_most;
pub use refined_sets::refinement_forms::make_refined_set;
pub use refined_sets::repetition_window_forms::as_repetition;

pub use super::dict_key::DictKey;
pub use super::dict_key::known_dict_key;
pub use super::dict_literal::dict_literal_value;
pub use super::dict_write::dict_with_item;
pub use super::dict_write::dict_without_item;
pub(in crate::collection_models) use super::kernel_join::kernel_joined_set;
pub(in crate::collection_models) use super::kernel_join::known_value_of_state;
pub use super::len_and_get::dict_get_result;
pub use super::len_and_get::len_result;
pub use super::list_literal::list_literal_value;
pub use super::list_literal::tuple_literal_value;
pub use super::list_write::list_with_item;
pub use super::subscript_read::subscript_read;
pub use super::mutated_receiver;

mod literal_and_dict_key;
mod kernel_join_tests;
mod subscript_and_len;
mod get_and_write;
mod mutation;

pub fn integer(value: f64) -> AbstractValue {
    known_values(vec![value], PrimitiveKind::Integer, TrustProved)
}

pub fn string(text: &str) -> AbstractValue {
    let code_points: Vec<f64> = text.chars().map(|c| c as u32 as f64).collect();
    known_values(code_points, PrimitiveKind::String, TrustProved)
}

pub fn key(text: &str) -> DictKey {
    DictKey::string(text)
}
