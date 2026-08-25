//! The RefinedPy recognition engine: Python syntax recognized and
//! lowered to refined-set questions the proved Lean kernel answers.
//! The LSP seam (`lsp::non_wasm::refinedpy`) calls into `check`; the
//! host's CFG and types are sites and sort oracles only — every
//! refined value and every judgment is owned here (plan-v2 laws L3-L5).

pub mod assignability;
pub mod binding_manifest;
pub mod builtin_models;
pub mod bytes_models;
pub mod check;
pub mod collection_models;
pub mod cross_adapter_twins;
pub mod cross_module;
pub mod diagnostic_sentences;
pub mod env;
pub mod expressions;
pub mod fact_export;
pub mod foreign_edge;
pub mod foreign_edge_artifact;
pub mod function_table;
pub mod instances;
pub mod json_grammar;
pub mod kernel_ask;
pub mod kernel_path;
pub mod lattice_conformance;
pub mod loops;
pub mod markers;
pub mod match_arms;
pub mod math_models;
pub mod narrowing;
pub mod relational_sum;
pub mod sequence_conformance;
pub mod string_models;
pub mod summaries;
pub mod summary_lowering;
pub mod surface;
pub mod transfer_conformance;
pub mod truthiness_conformance;
pub mod typereading;
pub mod tzif;
