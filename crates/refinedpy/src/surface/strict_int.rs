//! Which module-level aliases have a `StrictInt` base — the check.rs
//! adapter route's own coercion-gate lookup.

use std::collections::HashSet;

use ruff_python_ast::{Expr, ModModule, Stmt};

use super::imports::surface_imports;

/// Every `type X = Annotated[StrictInt, …]` alias name at the module's
/// top level — check.rs's `TypeAdapter(<alias>).validate_python(...)`
/// adapter route consults this to decide whether a `str` argument may
/// COERCE (a lax `int` base) or must REFUSE outright (a `StrictInt`
/// base, execution-verified against pydantic 2.13.4: `StrictInt` never
/// attempts str-to-int coercion, unlike bare `int`). Scans the SAME
/// `Annotated[...]` subscript shape `annotated_expression_set` reads,
/// but only the base sort — `compile_aliases`' own `RefinedSet` answer
/// carries no strictness bit, since `int` and `StrictInt` compile to
/// the identical integer form.
pub fn strict_int_alias_names(module: &ModModule) -> HashSet<String> {
    let imports = surface_imports(module);
    let mut out = HashSet::new();
    for stmt in module.body.iter() {
        // the same three alias spellings compile_aliases admits
        let (name, value) = match stmt {
            Stmt::TypeAlias(alias) => {
                let Expr::Name(name) = alias.name.as_ref() else {
                    continue;
                };
                (name, alias.value.as_ref())
            }
            Stmt::Assign(assign) => {
                let [Expr::Name(name)] = assign.targets.as_slice() else {
                    continue;
                };
                (name, assign.value.as_ref())
            }
            Stmt::AnnAssign(annotated) => {
                let Expr::Name(name) = annotated.target.as_ref() else {
                    continue;
                };
                let Some(value) = annotated.value.as_deref() else {
                    continue;
                };
                (name, value)
            }
            _ => continue,
        };
        let Expr::Subscript(subscript) = value else {
            continue;
        };
        let Expr::Name(head) = subscript.value.as_ref() else {
            continue;
        };
        if !imports.annotated_names.contains(head.id.as_str()) {
            continue;
        }
        let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
            continue;
        };
        let Some(Expr::Name(base)) = arguments.elts.first() else {
            continue;
        };
        if imports.strict_int_names.contains(base.id.as_str()) {
            out.insert(name.id.as_str().to_owned());
        }
    }
    out
}
