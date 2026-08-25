//! Binds names to values during the check walk: plain `Assign`,
//! annotated `AnnAssign`, tuple/list destructuring, subscript/attribute
//! writes and deletes, and the sink that judges a write's own value
//! expression before any of those bind it. Split by binding shape:
//! `ann_assign` (the `x: Annotation = value` judging channel),
//! `assign` (the plain `a = b = value` channel, plus the
//! `cast(Callable[...], f)` return-refinement seam), `destructure`
//! (tuple/list targets, attribute/subscript writes and deletes), and
//! `sink` (`sink_value`, the shared value-producing seam every write
//! statement calls before binding).

mod ann_assign;
mod assign;
mod destructure;
mod sink;

pub use sink::setdefault_append;

// Sibling-shared helpers: children call these as `super::name`.
pub(super) use ann_assign::bind_target_from_value_expr;
pub(super) use ann_assign::declared_set_is_empty;
pub(super) use ann_assign::direct_alias_annotation;
pub(super) use ann_assign::optional_base_sort_annotation;
pub(super) use ann_assign::unhonorable_annotated_spelling;
pub(super) use ann_assign::walk_ann_assign;
pub(super) use assign::cast_to_callable_return;
pub(super) use assign::walk_assign;
pub(super) use destructure::bind_known_sequence_target;
pub(super) use destructure::bind_or_forget_subscript_target;
pub(super) use destructure::bind_or_forget_target;
pub(super) use destructure::bind_sequence_element;
pub(super) use destructure::receiver_base_name;
pub(super) use destructure::unpack_mismatch_detail;
pub(super) use destructure::walk_del_subscript_target;
pub(super) use destructure::write_named_field;
pub(super) use sink::bind_or_forget_imported_name;
pub(super) use sink::sink_value;

// Cross-module helpers every binding-shape sibling reads, re-exported
// here under `super::name` so a sibling's own `use super::X;` line
// does not need to know which check/ module actually defines it.
pub(super) use super::apply_call_effects;
pub(super) use super::bind_walrus_targets;
pub(super) use super::callable_variable_call_result;
pub(super) use super::construction_call_verdict;
pub(super) use super::forget_target_names;
pub(super) use super::instance_method_call_result;
pub(super) use super::judge_and_bind;
pub(super) use super::judge_and_bind_naming;
pub(super) use super::manifest_call_fires;
pub(super) use super::name_unmodeled_call_sentence;
pub(super) use super::record_blocker;
pub(super) use super::same_module_call_argument_fires;
pub(super) use super::same_module_def_call_result_already_reported;
