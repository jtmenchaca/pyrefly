//! Call-site judging during the walk: a foreign manifest call, a
//! same-module def's argument crossing, a callable-variable's declared
//! return, an already-reported same-module escape, a statement-side
//! instance method call, the callee-effects channel, the stale-receiver
//! mutation law, and class-construction judging (bare-Name construction
//! plus pydantic's `model_validate`/`model_validate_json`/
//! `TypeAdapter(...).validate_python` family).

mod construction;
mod effects;
mod manifest;
mod method_call;
mod mutation;
mod parameter_write;
mod same_module;

pub(in crate::check) use construction::{
    adapter_alias_verdict, class_model_of_bare_name, construction_call_verdict, declared_set_instance,
    dict_literal_keyword_rows, evaluate_keyword_arguments, evaluate_positional_arguments, plain_digit_string_value,
    single_dict_argument,
};
pub(in crate::check) use effects::apply_call_effects;
pub(in crate::check) use manifest::manifest_call_fires;
pub(in crate::check) use method_call::{instance_method_call_result, keyword_arguments_by_position};
pub(in crate::check) use mutation::walk_mutating_call_statement;
pub(in crate::check) use parameter_write::body_may_write_through_parameter;
pub(in crate::check) use same_module::{
    callable_variable_call_result, judge_one_call_argument, same_module_call_argument_fires,
    same_module_def_call_result_already_reported,
};
