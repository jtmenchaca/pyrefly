//! The exported parameter entry rows: reading each `def`'s declared
//! parameters into the shape the artifact carries — a SEQUENCE (an
//! element set plus a length floor) or a SCALAR (one cases list).

use std::collections::HashMap;
use std::collections::HashSet;

use ruff_python_ast::StmtFunctionDef;
use serde_json::Value;
use serde_json::json;

use crate::env::Environment;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;
use crate::typereading::DeclaredRefinement;
use crate::typereading::base_sort_return_refinement;
use crate::typereading::declared_refinement;
use crate::typereading::typed_dict_return_refinement;

use super::cases::Case;
use super::cases::cases_json;
use super::cases::scalar_case_of;

/// One exported parameter: its name, and the shape its declaration
/// states — a SEQUENCE (an element set plus the length floor its own
/// declaration carries) or a SCALAR (one set).
pub(super) struct EntryRow {
    pub(super) name: String,
    pub(super) shape: EntryShape,
}

pub(super) enum EntryShape {
    /// `list[X]`/`set[X]`/`Sequence[X]` — the element's own cases, and
    /// the declaration's own length floor (`element_length`'s `lo`, 0
    /// when the declaration states no bound, exactly the default
    /// `check::seed_parameters` seeds the repetition window with).
    Sequence {
        element: Vec<Case>,
        length_at_least: i64,
    },
    /// Every other readable declaration: its own cases — a plain
    /// declaration is one case; an `admits_none` declaration (`X |
    /// None`, `Optional[X]`) carries the inner case(s) PLUS the null
    /// case, the same "the flag stops being dropped" fix the writer
    /// applies to a derived possibly-absent RETURN.
    Scalar(Vec<Case>),
}

/// `def`'s parameters read into exported entry rows, or the reason the
/// whole def cannot be exported.
///
/// Reads each parameter through the SAME three-step fallback chain
/// `check.rs::walk_function_def` runs to build a return refinement —
/// `declared_refinement`, then the bare `int`/`float`/`str` sort
/// reading, then `typed_dict_return_refinement` against `typed_dicts`
/// (built once by the caller, `instances::typed_dict_table`) — so a
/// parameter annotated with a recorded TypedDict class name reaches the
/// object case exactly as a TypedDict-declared RETURN already does.
/// A parameter the walk leaves unseeded by all three readers omits the
/// def rather than crossing an unfounded set.
pub(super) fn entry_rows(
    def: &StmtFunctionDef,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    typed_dicts: &HashMap<String, Vec<(String, DeclaredRefinement)>>,
) -> Result<Vec<EntryRow>, String> {
    // A variadic tail has no fixed arity and therefore no entry row a
    // caller on the other side of the wire could fill — checked before
    // any row is read, so the reason names the real obstacle.
    if def.parameters.vararg.is_some() || def.parameters.kwarg.is_some() {
        return Err("a '*args'/'**kwargs' parameter states no crossable entry shape".to_owned());
    }
    // The same no-locals environment every module-level annotation read
    // in this checker uses: a top-level def's own annotation is never
    // shadowed by a body-local rebinding.
    let environment = Environment::new(HashSet::new());
    let mut rows = Vec::new();
    for parameter in def
        .parameters
        .posonlyargs
        .iter()
        .chain(def.parameters.args.iter())
        .chain(def.parameters.kwonlyargs.iter())
    {
        let parameter_name = parameter.parameter.name.id.as_str().to_owned();
        let Some(annotation) = parameter.parameter.annotation.as_deref() else {
            return Err(format!("parameter '{parameter_name}' carries no annotation"));
        };
        let Some(declared) = declared_refinement(annotation, aliases, imports, &environment)
            .or_else(|| base_sort_return_refinement(annotation))
            .or_else(|| typed_dict_return_refinement(annotation, typed_dicts))
        else {
            return Err(format!(
                "parameter '{parameter_name}' carries no refinement this checker reads"
            ));
        };
        rows.push(EntryRow {
            name: parameter_name,
            shape: entry_shape(&declared)?,
        });
    }
    if rows.is_empty() {
        return Err("the def declares no parameters, so it states no refined entry".to_owned());
    }
    Ok(rows)
}

/// One declaration read into the shape this artifact carries — the same
/// SEQUENCE/SCALAR split `check::seed_parameters` makes when it seeds
/// the parameter (a container spelling with a non-empty element set
/// seeds a repetition window; everything else seeds the set itself).
/// `admits_none` (`X | None`, `Optional[X]`) appends the null case to a
/// SCALAR declaration's own cases — the container arm has no `None`
/// reading of its own to extend (a `dict[str, X] | None` still routes
/// through `element`, unaffected by this function's scalar branch). A
/// TypedDict declaration (`declared.members` Some) reads as a single
/// OBJECT case through `declared_object_case`, mirroring the RETURN
/// side's own `object_case_of` — a parameter's declared member table
/// crosses exactly the same `{"sort":"object","members":...}` shape a
/// derived TypedDict return already does.
pub(super) fn entry_shape(declared: &DeclaredRefinement) -> Result<EntryShape, String> {
    let is_sequence_container = declared.spelling.starts_with("list[")
        || declared.spelling.starts_with("set[")
        || declared.spelling.starts_with("Sequence[");
    if is_sequence_container {
        let Some(element) = declared.element.as_ref() else {
            return Err(format!(
                "'{}' states a container with no element refinement",
                declared.spelling
            ));
        };
        if element.set.forms.is_empty() {
            return Err(format!(
                "'{}' states a container whose element is itself a container, which crosses no set",
                declared.spelling
            ));
        }
        let (lo, _hi) = declared.element_length.unwrap_or((0, None));
        return Ok(EntryShape::Sequence {
            element: vec![scalar_case_of(&element.set)],
            length_at_least: lo,
        });
    }
    if let Some(members) = &declared.members {
        let mut cases = vec![declared_object_case(members)?];
        if declared.admits_none {
            cases.push(Case::Null);
        }
        return Ok(EntryShape::Scalar(cases));
    }
    if declared.set.forms.is_empty() {
        return Err(format!(
            "'{}' states a shape (a dict, a tuple, a generator) that crosses no single set",
            declared.spelling
        ));
    }
    let mut cases = vec![scalar_case_of(&declared.set)];
    if declared.admits_none {
        cases.push(Case::Null);
    }
    Ok(EntryShape::Scalar(cases))
}

/// A TypedDict declaration's own member table read into ONE object
/// case — the entry-side twin of `object_case_of`, reading a per-field
/// `DeclaredRefinement` instead of a derived `AbstractValue`. Each
/// declared member recurses through `entry_shape` so a nested
/// TypedDict-typed field becomes its own nested object case, exactly as
/// a nested member does on the return side. `closed: true`
/// unconditionally: a `class X(TypedDict): ...` declaration states its
/// complete key set by construction (this table reads no
/// `NotRequired`/`total=False` relaxation), unlike the return side's
/// `closed` (read off a runtime value's own `complete` bit) — there is
/// no literal here to read completeness FROM, so the class declaration
/// itself is the fact this case states. A member whose own declared
/// refinement has no crossable shape (a plain `dict`, a generator, a
/// bare tuple — a nested TypedDict member recurses through
/// `entry_shape` instead of hitting this case) stops the WHOLE object
/// case, naming that member — the same all-or-nothing rule
/// `object_case_of` already applies to a derived member.
fn declared_object_case(members: &[(String, DeclaredRefinement)]) -> Result<Case, String> {
    let mut cases = Vec::with_capacity(members.len());
    for (name, declared) in members {
        let shape = entry_shape(declared).map_err(|reason| {
            format!("its member '{name}' is {reason}, which has no faithful cases reading")
        })?;
        let member_cases = match shape {
            EntryShape::Scalar(cases) => cases,
            // A sequence-shaped member states its own element cases plus
            // its length floor — neither of which this object case's
            // recursive cases list can carry (a `Case` states one value's
            // own shape, not a repetition window), so a container-typed
            // TypedDict member is not yet a shape this table crosses.
            EntryShape::Sequence { .. } => {
                return Err(format!(
                    "its member '{name}' is a container, which crosses no single cases reading"
                ));
            }
        };
        cases.push((name.clone(), member_cases));
    }
    Ok(Case::Object {
        members: cases,
        closed: true,
    })
}

/// One entry row as the artifact spells it.
pub(super) fn entry_row_json(row: &EntryRow) -> Value {
    match &row.shape {
        EntryShape::Sequence {
            element,
            length_at_least,
        } => json!({
            "name": row.name,
            "sequence": {"element": {"cases": cases_json(element)}, "lengthAtLeast": length_at_least},
        }),
        EntryShape::Scalar(cases) => json!({"name": row.name, "cases": cases_json(cases)}),
    }
}
