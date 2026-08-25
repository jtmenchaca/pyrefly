//! Method lookup and call results.

use std::collections::HashMap;
use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{Expr, StmtFunctionDef};

use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::function_table::FunctionTable;

use super::model::ClassModel;

/// `model`'s own callable named `name` — own-overrides-inherited, since
/// `model.methods` (built in `class_model_of`) already holds the
/// EFFECTIVE set: a child's own def already replaced any inherited def
/// of the same name there. `None` for a name the class declares no
/// method under at all.
pub fn method_def_of<'a>(model: &'a ClassModel, name: &str) -> Option<&'a StmtFunctionDef> {
    model.methods.get(name)
}

/// `receiver.method(arguments)` — the instance AFTER the call (any
/// `self.<field> = ...` write inside the method body survives on it)
/// and the method's own return value, or `None` when the method's body
/// or parameter shape is outside what `summaries`'s restricted body
/// interpreter reads.
///
/// `self` binds to `instance` and the remaining parameters bind to
/// `arguments` positionally (`summaries::bind_parameters`'s own
/// convention: a trailing parameter with no matching argument takes its
/// own default, evaluated fresh; too few arguments with no default, or
/// too many, declines the whole call) — `self` is excluded from that
/// positional binding since it is bound directly to `instance`, never
/// to an entry of `arguments`.
///
/// The body interprets through `summaries::interpret_body`, the SAME
/// restricted statement walk an ordinary same-module call uses, with
/// one addition: a `super_resolver` that answers `super().<name>(args)`
/// by looking `name` up in `model.parent_methods` (the parent's OWN
/// methods, never touched by this child's overrides) and recursively
/// calling `method_call_result` on THAT def, over the SAME working
/// instance the resolver was handed (so a `super().__init__(...)` call
/// early in a child method still writes fields onto the one instance
/// this call is building), with an EMPTY `parent_methods` one level up
/// (a grandparent's own further `super()` chain is out of the
/// single-inheritance band this table builds — `class_table`'s own
/// scope). The resolver reads the CALLER's `super_resolver` parameter
/// (`environment: &Environment`) for `self`'s WORKING value at the
/// point of the call, per `summaries::SuperResolver`'s own doc — every
/// earlier `self.<field> = ...` statement in the SAME method body has
/// already updated it there.
///
/// `self.<field>` reads/writes inside the body route through
/// `summaries::interpret_body`'s own `self`-aware `Assign`/`AugAssign`/
/// `Expr::Name("self")` handling (`field_read`/`field_write`, the same
/// two functions this file exports) — this function does not re-walk
/// the body itself, only sets up the environment and resolver
/// `interpret_body` needs.
///
/// The answer is `(working instance, joined return value)`: every
/// `return` the body's paths could reach joins into one value
/// (`join_known`, `interpret_body`'s own fold), and a body that falls
/// off the end without an explicit `return` contributes `null_value()`
/// to that join — the same fall-through law `summaries::call_result`
/// already applies to an ordinary function. Depth-capped through
/// `summaries::CALL_DEPTH_CAP`, shared with every other same-module
/// call chain so a recursive method (directly, or through a
/// `super()`-chained cycle) declines rather than hangs. Any unsupported
/// body/parameter shape (`*args`/`**kwargs`/keyword-only parameters, a
/// statement `interpret_body` does not read, an unresolved `super()`
/// call, an unreadable return) answers `None` — an honest decline,
/// never a guessed instance.
///
/// `classes` seeds the method body's own environment with the module's
/// class table (`environment.set_classes`), the same way `table` seeds
/// the function table — without it, `self.<other_method>()` and any
/// property read inside the body has no class to resolve `self`'s own
/// type against, and declines even though the class and method both
/// exist. `None` when the caller has no class table to offer (a plain
/// unit test constructing a method call directly, for instance).
///
/// `datetime_imports` seeds the method body's own environment with the
/// module's `datetime` import identities (`environment.set_datetime_
/// imports`), the same explicit-parameter shape `classes` already takes
/// here — this function builds a FRESH `Environment` for the method
/// body (below) and has no enclosing `Environment` in scope to inherit
/// from (unlike `summaries::call_result_with_enclosing`, which inherits
/// `classes`/`datetime_imports` from its own `enclosing: Option<&
/// Environment>` parameter when the callee sets neither of its own), so
/// every caller threads its own module's table through explicitly, the
/// same way every caller already threads `classes`. Without it, a
/// method body's own `date(...)`/`dt.strptime(...)` call (an aliased
/// `datetime` construction/classmethod call — `expressions.rs`'s
/// datetime gates) has no import table to resolve the alias against and
/// declines even though the same call in a plain function recognizes.
/// `None` when the caller has no import table to offer (a plain unit
/// test constructing a method call directly, for instance).
pub fn method_call_result(
    instance: &AbstractValue,
    model: &ClassModel,
    method: &StmtFunctionDef,
    arguments: &[AbstractValue],
    table: Option<&Arc<FunctionTable>>,
    classes: Option<&Arc<HashMap<String, ClassModel>>>,
    datetime_imports: Option<&Arc<crate::expressions::DatetimeImports>>,
    kernel: &Arc<RefinedTSKernel>,
    depth: u32,
) -> Option<(AbstractValue, AbstractValue)> {
    use crate::summaries::{collect_bound_names, interpret_body, CALL_DEPTH_CAP};

    if depth >= CALL_DEPTH_CAP {
        return None;
    }
    if method.parameters.vararg.is_some()
        || method.parameters.kwarg.is_some()
        || !method.parameters.kwonlyargs.is_empty()
    {
        return None;
    }
    // `@staticmethod` declares no `self`/`cls` receiver slot at all
    // (datamodel.rst's own "Static method objects": "a static method
    // does not receive an implicit first argument") — every declared
    // parameter binds positionally to `arguments`, exactly like an
    // ordinary same-module `def`, with no receiver consumed. Every
    // other member `def` (including `@classmethod`, out of this
    // function's own scope — its first parameter is `cls`, still a
    // receiver slot, just bound to the class rather than the instance)
    // keeps the `self`-splitting shape below.
    let is_static = method
        .decorator_list
        .iter()
        .any(|decorator| matches!(&decorator.expression, Expr::Name(name) if name.id.as_str() == "staticmethod"));

    let all_parameters: Vec<_> = method
        .parameters
        .posonlyargs
        .iter()
        .chain(method.parameters.args.iter())
        .collect();
    // the first parameter is `self` by convention for an ordinary
    // instance method (matching `init_derived_fields`'s own
    // first-parameter reading) — a method with no parameter at all is
    // not a bound instance method this function can seed a receiver
    // for. A `@classmethod`'s own first parameter is spelled `cls`
    // instead, still a receiver slot, just bound to the class rather
    // than the instance — `receiver_parameter_name` carries the ACTUAL
    // spelling forward so a `cls.<attr>` read/write inside the body
    // resolves, alongside the literal `"self"` binding every other
    // reader in this file (`self_attribute_name`, `interpret_assign`'s
    // own self-write recognition) already hardcodes.
    let (receiver_parameter_name, rest): (Option<&str>, Vec<_>) = if is_static {
        (None, all_parameters)
    } else {
        let (self_parameter, rest) = all_parameters.split_first()?;
        (Some(self_parameter.parameter.name.id.as_str()), rest.to_vec())
    };
    if arguments.len() > rest.len() {
        return None;
    }

    let mut locally_bound = std::collections::HashSet::new();
    if let Some(receiver_parameter_name) = receiver_parameter_name {
        locally_bound.insert("self".to_owned());
        locally_bound.insert(receiver_parameter_name.to_owned());
    }
    for parameter in &rest {
        locally_bound.insert(parameter.parameter.name.id.as_str().to_owned());
    }
    collect_bound_names(&method.body, &mut locally_bound);
    let mut environment = Environment::new(locally_bound);
    // one call deeper than the caller — the depth cap engages across
    // the evaluate↔interpreter boundary (see env::call_depth)
    environment.set_call_depth(depth.saturating_add(1));
    if let Some(table) = table {
        environment.set_functions(table.clone());
    }
    if let Some(classes) = classes {
        environment.set_classes(classes.clone());
    }
    if let Some(datetime_imports) = datetime_imports {
        environment.set_datetime_imports(datetime_imports.clone());
    }
    if let Some(receiver_parameter_name) = receiver_parameter_name {
        environment.bind("self", instance.clone());
        if receiver_parameter_name != "self" {
            environment.bind(receiver_parameter_name, instance.clone());
        }
    }

    let default_environment = Environment::new(Default::default());
    for (index, parameter) in rest.iter().enumerate() {
        let value = if let Some(argument) = arguments.get(index) {
            argument.clone()
        } else {
            let default_expr = parameter.default.as_deref()?;
            evaluate_expression(default_expr, &default_environment, kernel)
        };
        environment.bind(parameter.parameter.name.id.as_str(), value);
    }

    let parent_methods = model.parent_methods.clone();
    let super_resolver = move |name: &str, args: &[AbstractValue], environment: &Environment| {
        let parent_def = parent_methods.get(name)?;
        let working_instance = environment.read("self")?.clone();
        let parentless = ClassModel {
            name: model.name.clone(),
            fields: Vec::new(),
            properties: HashMap::new(),
            methods: parent_methods.clone(),
            parent_methods: HashMap::new(),
            class_attributes: Vec::new(),
        };
        let (_after, result) = method_call_result(
            &working_instance,
            &parentless,
            parent_def,
            args,
            table,
            classes,
            datetime_imports,
            kernel,
            depth + 1,
        )?;
        Some(result)
    };

    let mut returns: Vec<AbstractValue> = Vec::new();
    let falls_through = interpret_body(&method.body, kernel, depth, &mut environment, &mut returns, Some(&super_resolver))?;
    if falls_through {
        returns.push(refined_domain::abstract_value::null_value());
    }

    let mut answers = returns.into_iter();
    let first = answers.next()?;
    let result = answers.fold(first, |acc, next| refined_domain::lattice_operations::join_known(acc, next));
    // A static method binds no `self` at all — there is no receiver for
    // its own body to mutate, so the "working instance" half of the
    // answer is simply `instance` unchanged (the caller's own
    // class-object value, echoed back rather than read out of an
    // environment slot that was never bound).
    let working_instance = if is_static { instance.clone() } else { environment.read("self")?.clone() };
    Some((working_instance, result))
}
