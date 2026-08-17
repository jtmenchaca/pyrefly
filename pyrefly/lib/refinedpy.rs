/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The RefinedPy recognition engine: Python syntax recognized and
//! lowered to refined-set questions the proved Lean kernel answers.
//! The LSP seam (`lsp::non_wasm::refinedpy`) calls into `check`; the
//! host's CFG and types are sites and sort oracles only — every
//! refined value and every judgment is owned here (plan-v2 laws L3-L5).

pub mod assignability;
pub mod builtin_models;
pub mod check;
pub mod collection_models;
pub mod env;
pub mod expressions;
pub mod kernel_path;
pub mod loops;
pub mod match_arms;
pub mod math_models;
pub mod narrowing;
pub mod string_models;
pub mod surface;
pub mod typereading;
