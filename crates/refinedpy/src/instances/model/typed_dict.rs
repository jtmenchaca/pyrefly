//! Every module-level `class X(TypedDict): name: Annotation, …` read
//! into its own per-member refinement table.

use std::collections::HashMap;

use ruff_python_ast::{Expr, ModModule, Stmt, StmtClassDef};

use crate::env::Environment;
use crate::surface::{AliasEntry, SurfaceImports};
use crate::typereading::{base_sort_return_refinement, declared_refinement, TypedDictMember};

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
///
/// Each member records whether the declaration REQUIRES its key to be
/// present: the class's own `total=` keyword sets the default
/// (`class_totality`), and a per-key `Required[...]`/`NotRequired[...]`
/// marker overrides it for that key alone (`presence_marker`). The
/// MEMBERS LAW (`assignability::judge`) reads this to decide whether a
/// declared key ABSENT from a closed flowing value is a refusal.
///
/// A member's refinement is its annotation's own
/// (`declared_refinement`), falling back to its BASE SORT
/// (`base_sort_return_refinement`) for a plain builtin like `a: int` —
/// the same two-source rule `class_table`'s `ClassField.declared`/
/// `base_sort` pair already applies to an ordinary class field. Only a
/// member stating NEITHER (a class name this table does not model, an
/// unread generic) is left out of the list, matching the honest-absence
/// convention elsewhere in this checker: an absent member states nothing
/// the MEMBERS LAW judges, never a guessed set.
pub fn typed_dict_table(
    module: &ModModule,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
) -> HashMap<String, Vec<TypedDictMember>> {
    let empty_environment = Environment::new(Default::default());
    let mut out = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::ClassDef(def) = stmt else {
            continue;
        };
        if single_bare_name_base(def) != Some("TypedDict") {
            continue;
        }
        let total = class_totality(def);
        let mut members = Vec::new();
        for member_stmt in def.body.iter() {
            let Stmt::AnnAssign(assign) = member_stmt else {
                continue;
            };
            let Expr::Name(target_name) = assign.target.as_ref() else {
                continue;
            };
            let marker = presence_marker(assign.annotation.as_ref());
            let annotation = unwrap_required_marker(assign.annotation.as_ref());
            // A member whose annotation states no refinement of its own
            // falls back to its BASE SORT — the whole-int ray for
            // `a: int`, the same fallback `class_table`'s own
            // `ClassField.base_sort` already records for an ordinary
            // class field. Without it a `class P(TypedDict): a: int`
            // records NO members at all (`declared_refinement` declines
            // a bare `int`: it is not an alias), the member table is
            // empty, and the MEMBERS LAW iterates zero times and answers
            // Silent — so neither a member's own out-of-set value nor a
            // missing REQUIRED key can ever fire for a TypedDict whose
            // fields are plain builtin sorts.
            let Some(declared) = declared_refinement(annotation, aliases, imports, &empty_environment)
                .or_else(|| base_sort_return_refinement(annotation))
            else {
                continue;
            };
            members.push(TypedDictMember {
                name: target_name.id.as_str().to_owned(),
                required: marker.unwrap_or(total),
                declared,
            });
        }
        out.insert(def.name.id.as_str().to_owned(), members);
    }
    out
}

/// The class's own totality — `True` when the class states no `total=`
/// keyword or states `total=True`, `False` for `total=False`. From
/// library/typing.rst, `TypedDict`: "``True`` is the default, and makes
/// all items defined in the class body required," and "It is also
/// possible to mark all keys as non-required by default by specifying a
/// totality of ``False``." The same clause pins the reading to a literal:
/// "A type checker is only expected to support a literal ``False`` or
/// ``True`` as the value of the ``total`` argument," so a `total=` whose
/// value is anything else keeps the `True` default rather than guessing.
fn class_totality(def: &StmtClassDef) -> bool {
    let Some(arguments) = def.arguments.as_ref() else {
        return true;
    };
    for keyword in arguments.keywords.iter() {
        let Some(name) = keyword.arg.as_ref() else {
            continue;
        };
        if name.as_str() != "total" {
            continue;
        }
        if let Expr::BooleanLiteral(literal) = &keyword.value {
            return literal.value;
        }
    }
    true
}

/// A per-key presence marker read off the member's own annotation:
/// `Some(true)` for `Required[X]`, `Some(false)` for `NotRequired[X]`,
/// `None` when the annotation wears neither and the key takes the
/// class's totality instead. Recognized by bare name, the same
/// no-import-identity convention `unwrap_required_marker` below takes.
fn presence_marker(annotation: &Expr) -> Option<bool> {
    let Expr::Subscript(subscript) = annotation else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    match head.id.as_str() {
        "Required" => Some(true),
        "NotRequired" => Some(false),
        _ => None,
    }
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
/// A member's PRESENCE is recorded separately, on `TypedDictMember::
/// required` (`presence_marker` above reads the marker this function
/// peels, and the class's `total=` supplies the default) — this function
/// answers only WHAT SET the key holds, which neither marker changes.
/// Peeling matters independently of that recording:
/// `class_parameter_object` (check.rs) seeds a `Kind::Object` key for
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
