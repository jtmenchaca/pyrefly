//! A bare-Name annotation naming a module-level TypedDict class, read
//! into its own per-field `DeclaredRefinement` table.

use std::collections::HashMap;

use refined_sets::refinement_forms::make_refined_set;
use ruff_python_ast::Expr;

use super::declared_refinement::{DeclaredRefinement, TypedDictMember};

/// A bare-Name return/AnnAssign annotation naming a module-level
/// TypedDict class (`instances::typed_dict_table`'s own keys) —
/// `PersonDict`'s own per-member table, wrapped as a `DeclaredRefinement`
/// with `members: Some(...)` so `assignability::judge`'s MEMBERS law can
/// judge a dict literal against it field-by-field. `None` for anything
/// else (an alias name, a class that is not a recognized TypedDict, a
/// non-bare-Name annotation) — the ordinary `declared_refinement` path
/// already owns every other shape, and this function is a narrow
/// addition alongside it, not a replacement.
pub fn typed_dict_return_refinement(
    annotation: &Expr,
    typed_dicts: &HashMap<String, Vec<TypedDictMember>>,
) -> Option<DeclaredRefinement> {
    let Expr::Name(name) = annotation else {
        return None;
    };
    let members = typed_dicts.get(name.id.as_str())?;
    Some(DeclaredRefinement {
        set: make_refined_set(Vec::new()),
        spelling: name.id.as_str().to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: Some(members.clone()),
        positions: None,
        temporal: None,
        temporal_awareness: crate::surface::TemporalAwareness::Any,
    })
}
