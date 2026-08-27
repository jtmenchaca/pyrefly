//! The import identities the surface resolves names against.

use std::collections::HashSet;

use ruff_python_ast::{ModModule, Stmt, StmtImport, StmtImportFrom};

/// The import identities the surface resolves names against: which
/// local names mean pydantic's `Field`, which local names mean the
/// pydantic module itself, which local names mean `Annotated` (from
/// `typing` or `typing_extensions`), which local name means pydantic's
/// `StrictInt`, and one set per recognized `annotated_types`
/// constructor (`Ge`/`Gt`/`Le`/`Lt`/`MultipleOf`/`MinLen`/`MaxLen`).
#[derive(Clone)]
pub struct SurfaceImports {
    pub(super) field_names: HashSet<String>,
    pub(super) pydantic_modules: HashSet<String>,
    pub(crate) annotated_names: HashSet<String>,
    pub(super) strict_int_names: HashSet<String>,
    pub(super) annotated_types_ge: HashSet<String>,
    pub(super) annotated_types_gt: HashSet<String>,
    pub(super) annotated_types_le: HashSet<String>,
    pub(super) annotated_types_lt: HashSet<String>,
    pub(super) annotated_types_multiple_of: HashSet<String>,
    pub(super) annotated_types_min_len: HashSet<String>,
    pub(super) annotated_types_max_len: HashSet<String>,
    /// `from datetime import date[ as x]` — the local names that mean
    /// the stdlib `date` class (datetime.rst, `class:: date(year,
    /// month, day)`), read by `temporal_alias_annotation` the same
    /// no-import-identity-required-elsewhere-but-checked-here way this
    /// table already gates `Field`/`Annotated`/`StrictInt`.
    pub(crate) date_names: HashSet<String>,
    /// `from datetime import timedelta[ as x]`.
    pub(crate) timedelta_names: HashSet<String>,
    /// `from datetime import datetime[ as x]`.
    pub(crate) datetime_names: HashSet<String>,
    /// `from pydantic import AwareDatetime[ as x]` — a `datetime` typed
    /// `AwareDatetime` REFUSES a naive construction outright (pydantic's
    /// own documented behavior, cited at `temporal_alias_annotation`'s
    /// own call site).
    pub(crate) aware_datetime_names: HashSet<String>,
    /// `from pydantic import NaiveDatetime[ as x]` — the mirror: refuses
    /// an AWARE construction outright.
    pub(crate) naive_datetime_names: HashSet<String>,
}

/// Reads the module's top-level `import`/`from … import …` statements
/// and records the local names that mean pydantic's `Field`, the
/// pydantic module, `Annotated`, pydantic's `StrictInt`, and each
/// recognized `annotated_types` constructor. Only the shapes named in
/// the mission are recognized: `import pydantic[ as x]`,
/// `from pydantic import Field[ as x]` (and `StrictInt[ as x]`), the
/// same two shapes for `Annotated` from `typing`/`typing_extensions`,
/// and `from annotated_types import Ge[ as x]` (and its six siblings).
/// Anything else (a `fields`-style submodule import, a re-export) is
/// out of scope and leaves the corresponding set empty.
pub fn surface_imports(module: &ModModule) -> SurfaceImports {
    let mut field_names = HashSet::new();
    let mut pydantic_modules = HashSet::new();
    let mut annotated_names = HashSet::new();
    let mut strict_int_names = HashSet::new();
    let mut annotated_types_ge = HashSet::new();
    let mut annotated_types_gt = HashSet::new();
    let mut annotated_types_le = HashSet::new();
    let mut annotated_types_lt = HashSet::new();
    let mut annotated_types_multiple_of = HashSet::new();
    let mut annotated_types_min_len = HashSet::new();
    let mut annotated_types_max_len = HashSet::new();
    let mut date_names = HashSet::new();
    let mut timedelta_names = HashSet::new();
    let mut datetime_names = HashSet::new();
    let mut aware_datetime_names = HashSet::new();
    let mut naive_datetime_names = HashSet::new();
    for stmt in module.body.iter() {
        match stmt {
            Stmt::Import(StmtImport { names, .. }) => {
                for alias in names {
                    if alias.name.id.as_str() == "pydantic" {
                        let local = alias.asname.as_ref().unwrap_or(&alias.name);
                        pydantic_modules.insert(local.id.as_str().to_owned());
                    }
                }
            }
            Stmt::ImportFrom(StmtImportFrom {
                module: Some(source),
                names,
                level: 0,
                ..
            }) => {
                for alias in names {
                    let local = alias.asname.as_ref().unwrap_or(&alias.name);
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "Field" {
                        field_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "StrictInt" {
                        strict_int_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "AwareDatetime" {
                        aware_datetime_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "pydantic" && alias.name.id.as_str() == "NaiveDatetime" {
                        naive_datetime_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "datetime" && alias.name.id.as_str() == "date" {
                        date_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "datetime" && alias.name.id.as_str() == "timedelta" {
                        timedelta_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "datetime" && alias.name.id.as_str() == "datetime" {
                        datetime_names.insert(local.id.as_str().to_owned());
                    }
                    if (source.id.as_str() == "typing" || source.id.as_str() == "typing_extensions")
                        && alias.name.id.as_str() == "Annotated"
                    {
                        annotated_names.insert(local.id.as_str().to_owned());
                    }
                    if source.id.as_str() == "annotated_types" {
                        match alias.name.id.as_str() {
                            "Ge" => {
                                annotated_types_ge.insert(local.id.as_str().to_owned());
                            }
                            "Gt" => {
                                annotated_types_gt.insert(local.id.as_str().to_owned());
                            }
                            "Le" => {
                                annotated_types_le.insert(local.id.as_str().to_owned());
                            }
                            "Lt" => {
                                annotated_types_lt.insert(local.id.as_str().to_owned());
                            }
                            "MultipleOf" => {
                                annotated_types_multiple_of.insert(local.id.as_str().to_owned());
                            }
                            "MinLen" => {
                                annotated_types_min_len.insert(local.id.as_str().to_owned());
                            }
                            "MaxLen" => {
                                annotated_types_max_len.insert(local.id.as_str().to_owned());
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    SurfaceImports {
        field_names,
        pydantic_modules,
        annotated_names,
        strict_int_names,
        annotated_types_ge,
        annotated_types_gt,
        annotated_types_le,
        annotated_types_lt,
        annotated_types_multiple_of,
        annotated_types_min_len,
        annotated_types_max_len,
        date_names,
        timedelta_names,
        datetime_names,
        aware_datetime_names,
        naive_datetime_names,
    }
}
