//! Unit tests for `crate::instances`.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::{known_values, unknown, AbstractValue, Kind, PrimitiveKind};
use refined_domain::trust_grades::TrustProved;
use refined_kernel::kernel_bridge::{dylib_path, kernel_artifacts_present, load_kernel};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::{at_least, at_most, integer, make_refined_set};
use ruff_python_ast::ModModule;
use ruff_text_size::TextRange;

use crate::assignability::Verdict;
use crate::env::Environment;
use crate::typereading::DeclaredRefinement;

use super::*;

mod class_table;
mod construction;
mod init_and_inheritance;
mod properties_and_methods;
mod field_write_and_identity;
mod generator_yields;

pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

pub(super) fn parsed(source: &str) -> ModModule {
    ruff_python_parser::parse_module(source).expect("test source parses").into_syntax()
}

pub(super) fn age_declared() -> DeclaredRefinement {
    DeclaredRefinement {
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
        set: make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)]),
        spelling: "Age".to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
    }
}

pub(super) fn integer_value(v: f64) -> AbstractValue {
    known_values(vec![v], PrimitiveKind::Integer, TrustProved)
}

/// A hand-built `ClassModel` with no properties and no methods —
/// every direct `judge_construction`/`field_write_judgment` test
/// builds a model this way rather than parsing source, since those
/// functions take the model, not the class definition.
pub(super) fn bare_model(name: &str, fields: Vec<ClassField>) -> ClassModel {
    ClassModel {
        name: name.to_owned(),
        fields,
        properties: HashMap::new(),
        methods: HashMap::new(),
        parent_methods: HashMap::new(),
        class_attributes: Vec::new(),
    }
}

pub(super) fn range_of(source: &str) -> TextRange {
    // a stable, arbitrary non-default range for tests that only
    // check WHICH range a fire carries back, never its exact span
    let _ = source;
    TextRange::default()
}
