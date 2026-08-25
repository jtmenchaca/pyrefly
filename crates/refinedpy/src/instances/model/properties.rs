//! `@property`/`@<name>.setter` pair recognition: the two exact body
//! shapes this table reads into a `PropertyModel`.

use std::collections::HashMap;

use ruff_python_ast::{Expr, Stmt, StmtClassDef, StmtFunctionDef};

use crate::env::Environment;
use crate::surface::{AliasEntry, SurfaceImports};
use crate::typereading::{declared_refinement, DeclaredRefinement};

use super::init_fields::is_bare_name;
use super::types::PropertyModel;

/// Every `@property`/`@<name>.setter` pair the class body declares,
/// keyed by the property's own name — exactly the two body shapes the
/// mission names, nothing richer:
///
/// - `@property\ndef <name>(self): return self.<backing>` — a
///   `StmtFunctionDef` named `<name>`, decorated with the bare name
///   `property`, whose body is a SINGLE `return self.<backing>`
///   statement.
/// - `@<name>.setter\ndef <name>(self, <param>): self.<backing> =
///   <param>` — a `StmtFunctionDef` of the SAME name, decorated with
///   `Attribute { value: Name(<name>), attr: "setter" }`, whose body
///   is a SINGLE `self.<backing> = <param>` statement writing the
///   setter's own (second) parameter, bare, into the SAME backing name
///   the getter reads.
///
/// A getter/setter pair whose backing names disagree, or whose body is
/// richer than the one statement above (a computed return, a guard, a
/// multi-statement body), is not modeled — the property is absent from
/// this table, matching the mission's "anything richer... is not
/// modeled (today's behavior)." A getter with no matching setter (or a
/// setter with no matching getter) is also absent — `PropertyModel`
/// always carries the getter's own `backing` name, so a setter alone
/// has no backing name to record.
pub(super) fn property_table(
    def: &StmtClassDef,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
) -> HashMap<String, PropertyModel> {
    let empty_environment = Environment::new(Default::default());
    let mut getters: HashMap<String, String> = HashMap::new();
    let mut setters: HashMap<String, (String, Option<DeclaredRefinement>)> = HashMap::new();

    for stmt in def.body.iter() {
        let Stmt::FunctionDef(function) = stmt else {
            continue;
        };
        let name = function.name.id.as_str();
        if is_property_getter_decorator(function) {
            if let Some(backing) = single_self_attribute_return(function) {
                getters.insert(name.to_owned(), backing);
            }
        } else if is_property_setter_decorator(function, name) {
            if let Some((backing, parameter)) = single_self_attribute_write_of_second_parameter(function) {
                let declared = parameter
                    .parameter
                    .annotation
                    .as_deref()
                    .and_then(|annotation| declared_refinement(annotation, aliases, imports, &empty_environment));
                setters.insert(name.to_owned(), (backing, declared));
            }
        }
    }

    let mut properties = HashMap::new();
    for (name, getter_backing) in getters {
        let declared = match setters.get(&name) {
            // the getter/setter pair's backing names must agree — a
            // setter writing a DIFFERENT attribute than the getter
            // reads is not a coherent accessor pair for this table.
            Some((setter_backing, declared)) if *setter_backing == getter_backing => declared.clone(),
            Some(_) => continue,
            None => None,
        };
        properties.insert(name, PropertyModel { backing: getter_backing, declared });
    }
    properties
}

/// `@property` — the decorator list carries exactly the bare name
/// `property` (no other decorator on the same function; a
/// `@property` stacked with something else is outside the mission's
/// two exact shapes).
fn is_property_getter_decorator(function: &StmtFunctionDef) -> bool {
    let [decorator] = function.decorator_list.as_slice() else {
        return false;
    };
    matches!(&decorator.expression, Expr::Name(name) if name.id.as_str() == "property")
}

/// `@<name>.setter` — the decorator list carries exactly one
/// `Attribute` whose own value is the bare name `<name>` (the SAME
/// name as the function being decorated — CPython's own
/// `@<property_name>.setter` spelling) and whose attribute is
/// literally `setter`.
fn is_property_setter_decorator(function: &StmtFunctionDef, property_name: &str) -> bool {
    let [decorator] = function.decorator_list.as_slice() else {
        return false;
    };
    let Expr::Attribute(attribute) = &decorator.expression else {
        return false;
    };
    attribute.attr.as_str() == "setter"
        && matches!(attribute.value.as_ref(), Expr::Name(base) if base.id.as_str() == property_name)
}

/// A getter body's exact single statement: `return self.<backing>` —
/// the backing attribute's name, or `None` for any other body shape
/// (more than one statement, a non-`self` or non-`Attribute` return
/// value, a bare `return`).
fn single_self_attribute_return(function: &StmtFunctionDef) -> Option<String> {
    let [Stmt::Return(ret)] = function.body.as_slice() else {
        return None;
    };
    let value = ret.value.as_deref()?;
    super::super::fields::self_attribute_name(value)
}

/// A setter body's exact single statement: `self.<backing> = <param>`,
/// where `<param>` is a BARE Name matching the setter's own second
/// parameter (the first is `self`) — the backing name and the
/// parameter itself, or `None` for any other shape (more than one
/// statement, a non-self-attribute target, an RHS that is not exactly
/// that bare parameter, or a setter with fewer than two parameters).
fn single_self_attribute_write_of_second_parameter(
    function: &StmtFunctionDef,
) -> Option<(String, &ruff_python_ast::ParameterWithDefault)> {
    let [Stmt::Assign(assign)] = function.body.as_slice() else {
        return None;
    };
    let [target] = assign.targets.as_slice() else {
        return None;
    };
    let backing = super::super::fields::self_attribute_name(target)?;
    let parameters: Vec<_> = function
        .parameters
        .posonlyargs
        .iter()
        .chain(function.parameters.args.iter())
        .collect();
    let (_self_param, rest) = parameters.split_first()?;
    let [value_parameter] = rest else {
        return None;
    };
    if !is_bare_name(assign.value.as_ref(), value_parameter.parameter.name.id.as_str()) {
        return None;
    }
    Some((backing, value_parameter))
}
