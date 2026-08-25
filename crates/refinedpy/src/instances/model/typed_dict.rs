//! Every module-level `class X(TypedDict): name: Annotation, …` read
//! into its own per-member refinement table.

use std::collections::HashMap;

use ruff_python_ast::{Expr, ModModule, Stmt};

use crate::env::Environment;
use crate::surface::{AliasEntry, SurfaceImports};
use crate::typereading::{declared_refinement, DeclaredRefinement};

use super::class_table::single_bare_name_base;

/// Every module-level `class X(TypedDict): name: Annotation, …` read
/// into its own per-member refinement table, keyed by the class's name.
/// A TypedDict class body carries no `__init__`, no methods, and no
/// inheritance chain this table follows — it is exactly the plain
/// AnnAssign rows `class_model_of` already reads for an ordinary class,
/// with none of that function's `__init__`/`super()` machinery, so this
/// is its own small reader rather than a `ClassModel`-shaped one: a
/// TypedDict's checked shape is "each member's own declared refinement,"
/// not "a constructible instance with fields." Recognized by BARE base
/// name `TypedDict` (`single_bare_name_base`, the same one-bare-Name-base
/// rule `class_table` already applies), matching `is_class_var`'s own
/// no-import-identity convention — no fixture row spells `TypedDict`
/// through an import alias.
pub fn typed_dict_table(
    module: &ModModule,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
) -> HashMap<String, Vec<(String, DeclaredRefinement)>> {
    let empty_environment = Environment::new(Default::default());
    let mut out = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::ClassDef(def) = stmt else {
            continue;
        };
        if single_bare_name_base(def) != Some("TypedDict") {
            continue;
        }
        let mut members = Vec::new();
        for member_stmt in def.body.iter() {
            let Stmt::AnnAssign(assign) = member_stmt else {
                continue;
            };
            let Expr::Name(target_name) = assign.target.as_ref() else {
                continue;
            };
            let annotation = unwrap_required_marker(assign.annotation.as_ref());
            let Some(declared) = declared_refinement(annotation, aliases, imports, &empty_environment) else {
                continue;
            };
            members.push((target_name.id.as_str().to_owned(), declared));
        }
        out.insert(def.name.id.as_str().to_owned(), members);
    }
    out
}

/// `Required[X]` / `NotRequired[X]` (typing.rst, "Required" /
/// "NotRequired" — TypedDict's own per-key presence override on a
/// `total=False` / `total=True` class) peeled down to `X` — the bare
/// annotation `declared_refinement` already knows how to read. Recognized
/// by bare name only, the same no-import-identity convention
/// `declared_refinement`'s own `Optional`/`Literal` arms already take
/// (`SurfaceImports` carries no `typing.Required`/`typing.NotRequired`
/// identity to gate on).
///
/// A member's PRESENCE is not itself a fact this table tracks either way
/// — `class_parameter_object` (check.rs) seeds a `Kind::Object` key for
/// every member THIS table records, so a member `declared_refinement`
/// cannot read stays entirely OFF the seeded value's own `keys`,
/// indistinguishable there from a genuinely absent key, and a later
/// `r["a"]` read on it wrongly proves `KeyError` — the exact defect
/// `Required[Age]` (a Record's own `total=False` class marking one key
/// ALWAYS present) surfaced: `Required`'s wrapper, left unpeeled, made
/// `declared_refinement` decline the whole member (a `Subscript` head it
/// does not otherwise recognize), so the member never reached `keys` at
/// all. Peeling to the wrapped annotation is what the MEMBERS LAW
/// (`assignability.rs`) then judges the member's value AGAINST — the
/// same set the annotation would state with no `Required`/`NotRequired`
/// wrapper at all, since neither marker narrows or widens the KEY'S OWN
/// value set, only whether the key must appear.
pub(super) fn unwrap_required_marker(annotation: &Expr) -> &Expr {
    let Expr::Subscript(subscript) = annotation else {
        return annotation;
    };
    let is_presence_marker = matches!(
        subscript.value.as_ref(),
        Expr::Name(head) if head.id.as_str() == "Required" || head.id.as_str() == "NotRequired"
    );
    if is_presence_marker {
        subscript.slice.as_ref()
    } else {
        annotation
    }
}
