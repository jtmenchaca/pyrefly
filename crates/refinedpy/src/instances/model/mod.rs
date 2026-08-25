//! Class models: declared fields, properties, methods, and class tables.

mod class_table;
mod defaults;
mod init_fields;
mod member_refinement;
mod properties;
mod typed_dict;
mod types;

pub use class_table::class_table;
pub use member_refinement::model_members_refinement;
pub use typed_dict::typed_dict_table;
pub use types::{ClassField, ClassModel, PropertyModel};
