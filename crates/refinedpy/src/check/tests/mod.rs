use super::*;
use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
use refined_sets::format_for_diagnostics::format_for_diagnostics;
use ruff_text_size::TextSize;

mod hover_position;
mod call_argument_and_annotation;
mod compound_write_and_return;
mod guard_narrowing;
mod statement_forms;
mod call_and_construction_flow;
mod match_and_lambda;
mod method_and_property;
mod loop_and_nested_class;
mod callable_variable;
mod optional_narrowing;
mod vararg_and_tuple;
mod pydantic_adapter;
mod foreign_edge_consumer;
mod flow_and_accumulation;

pub(super) fn parsed(source: &str) -> ModModule {
    ruff_python_parser::parse_module(source)
        .expect("fixture source parses")
        .into_syntax()
}

pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

pub(super) fn no_imports_resolver() -> ModuleResolver<'static> {
    &|_: &str| None
}

/// The byte offset of `needle`'s own first character in `source` —
/// a readable way to name a test position ("the `s` of `samples`")
/// rather than a bare integer that says nothing about what it
/// points at.
pub(super) fn offset_of(source: &str, needle: &str) -> TextSize {
    let byte_offset = source.find(needle).unwrap_or_else(|| panic!("{needle:?} not found in fixture"));
    TextSize::try_from(byte_offset).expect("fixture offsets fit in TextSize")
}
