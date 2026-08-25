use std::collections::HashSet;
use std::sync::Arc;

use refined_domain::abstract_value::float_sorted_unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_kernel::kernel_bridge::dylib_path;
use refined_kernel::kernel_bridge::kernel_artifacts_present;
use refined_kernel::kernel_bridge::load_kernel;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::above;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::repeat_of;
use ruff_python_ast::CmpOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;
use ruff_python_parser::parse_expression;
use ruff_text_size::TextRange;

use crate::assignability;
use crate::bytes_models;
use crate::collection_models;
use crate::env::Environment;
use crate::expressions::arithmetic::binary_arithmetic_value;
use crate::expressions::arithmetic::binop_possible_raise;
use crate::expressions::arithmetic::binop_provable_raise;
use crate::expressions::arithmetic::divisor_provably_excludes_zero;
use crate::expressions::arithmetic::possible_raise;
use crate::expressions::datetime::binary_arithmetic_value_with_kernel;
use crate::expressions::sequence_ops::provable_raise;

use super::*;

mod arithmetic;
mod attribute;
mod boolop_ternary;
mod call;
mod compare;
mod comprehension;
mod datetime;
mod fstring;
mod json_re;
mod literals;
mod raise_conditions;
mod sequence_ops;
mod subscript;

/// A kernel handle for tests that never ask it — evaluate_expression
/// takes the parameter for the contract's sake but no construct this
/// wave asks a question of it. `None` when the native dylib artifact
/// has not been built (same skip check.rs's own tests use), so this
/// file's tests run without requiring `pnpm kernel:native` first.
pub(super) fn loaded_kernel() -> Option<Arc<RefinedTSKernel>> {
    let path = dylib_path();
    if !kernel_artifacts_present(&path) {
        eprintln!("native kernel dylib absent — build it first");
        return None;
    }
    Some(load_kernel(&path).expect("load_kernel"))
}

pub(super) fn empty_environment() -> Environment {
    Environment::new(HashSet::new())
}

pub(super) fn eval(source: &str) -> Option<AbstractValue> {
    let kernel = loaded_kernel()?;
    let parsed = parse_expression(source).expect("test source must parse");
    let expression = parsed.into_expr();
    let environment = empty_environment();
    Some(evaluate_expression(&expression, &environment, &kernel))
}

/// A `Person` class with a `next_year(self, bump=1)` method — shared
/// between call.rs's method-dispatch tests and attribute.rs's
/// bare-bound-method-reference tests.
pub(super) fn person_next_year_module() -> ruff_python_ast::ModModule {
    ruff_python_parser::parse_module(concat!(
        "class Person:\n",
        "    def __init__(self, age):\n",
        "        self.age = age\n",
        "    def next_year(self, bump=1):\n",
        "        return self.age + bump\n",
    ))
    .expect("test module parses")
    .into_syntax()
}

pub(super) fn environment_with_person_classes(kernel: &Arc<RefinedTSKernel>) -> Environment {
    let module = person_next_year_module();
    let aliases = std::collections::HashMap::new();
    let imports = crate::surface::surface_imports(&module);
    let classes =
        std::sync::Arc::new(crate::instances::class_table(&module, &aliases, &imports, kernel));
    let mut environment = empty_environment();
    environment.set_classes(classes);
    environment
}
