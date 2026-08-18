/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The pydantic surface: `type X = Annotated[int, Field(ge=…, le=…)]`
//! aliases lowered to refined sets, one table (plan-v2 L7).
//!
//! The lowering walks the RAW annotation expression, never the host's
//! resolved `Type`: pydantic `Field`'s stub returns `Any`, so
//! `Type::Annotated`'s metadata slot holds the inferred return type
//! and the `ge`/`le` values are unrecoverable from it
//! (PYREFLY-API-NOTES.md §3).

use std::collections::HashMap;
use std::collections::HashSet;

use refined_sets::codepoint_sets::{string_tuple, strings};
use refined_sets::regex_compiler::format_grammar;
use refined_sets::refinement_forms::{
    Refinement, RefinedSet, above, at_least, at_most, below, integer, make_refined_set,
    multiple_of, one_of, union,
};
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::{Expr, ModModule, Number, Operator, Stmt, StmtImport, StmtImportFrom, UnaryOp};

/// Field kwargs that state nothing about the value set — safe to skip.
/// Any OTHER unrecognized kwarg refuses the whole alias: a constraint
/// this table cannot state must not silently widen or narrow the set.
const INERT_FIELD_KWARGS: &[&str] = &[
    "alias",
    "default",
    "description",
    "examples",
    "title",
];

/// Every `type X = Annotated[int|float, Field(…)]` alias at the
/// module's top level, lowered to its refined set, plus alias-of-alias
/// (`type Adult = Age`, where `Age` already named a compiled set),
/// plus a bare `type Pick = Literal[…]` alias (`literal_alias_set`),
/// plus a union of two `Literal[…]` aliases (`type PickUnion =
/// Literal[10, 20, 30] | Literal["ten", "twenty"]`,
/// `literal_union_alias_set`). Statements walk in source order so a
/// later alias can point at an earlier one. Aliases the table cannot
/// lower faithfully are absent — absence declines judgment, it never
/// approximates.
pub fn compile_aliases(module: &ModModule) -> HashMap<String, RefinedSet> {
    let imports = surface_imports(module);
    let mut out = HashMap::new();
    for stmt in module.body.iter() {
        let Stmt::TypeAlias(alias) = stmt else {
            continue;
        };
        let Expr::Name(name) = alias.name.as_ref() else {
            continue;
        };
        let set = annotated_expression_set(alias.value.as_ref(), &imports)
            .or_else(|| literal_alias_set(alias.value.as_ref()))
            .or_else(|| literal_union_alias_set(alias.value.as_ref()))
            .or_else(|| {
                // `type Adult = Age`: the RHS is a bare name that already
                // names a compiled set in this same table.
                let Expr::Name(rhs) = alias.value.as_ref() else {
                    return None;
                };
                out.get(rhs.id.as_str()).cloned()
            });
        if let Some(set) = set {
            out.insert(name.id.as_str().to_owned(), set);
        }
    }
    out
}

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
        let Stmt::TypeAlias(alias) = stmt else {
            continue;
        };
        let Expr::Name(name) = alias.name.as_ref() else {
            continue;
        };
        let Expr::Subscript(subscript) = alias.value.as_ref() else {
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

/// `type Pick = Literal[10, 20, 30]` (or a single-member/string-member
/// form) — the type-alias-RHS twin of `typereading.rs`'s
/// `declared_refinement`'s own `Literal[...]` arm (int members build a
/// numeric `one_of`, string members build the union of each member's
/// own singleton tuple, a mixed int/string member list declines whole).
/// Mirrored locally rather than imported: `surface.rs` is imported BY
/// `typereading.rs` (`annotated_expression_set`), so importing the
/// other direction would cycle.
fn literal_alias_set(value: &Expr) -> Option<RefinedSet> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let is_literal = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Literal");
    if !is_literal {
        return None;
    }
    if let Some(members) = int_literal_members(subscript.slice.as_ref()) {
        return Some(make_refined_set(vec![one_of(&members)]));
    }
    if let Some(members) = string_literal_members(subscript.slice.as_ref()) {
        return Some(string_literal_set(&members));
    }
    None
}

/// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]` —
/// exactly two `Literal[...]` arms joined by `|`, each read through
/// `literal_alias_set` and folded together by `refinement_forms::union`.
/// Any other union shape (a non-Literal arm, more than two arms — ruff
/// parses a chained `|` as nested `BinOp`s so a third arm would need a
/// second union node this reader does not build) declines.
fn literal_union_alias_set(value: &Expr) -> Option<RefinedSet> {
    let Expr::BinOp(binop) = value else {
        return None;
    };
    if binop.op != Operator::BitOr {
        return None;
    }
    let left = literal_alias_set(binop.left.as_ref())?;
    let right = literal_alias_set(binop.right.as_ref())?;
    Some(make_refined_set(vec![union(left, right)]))
}

/// `Literal[...]`'s slice read as a list of int-literal members —
/// `typereading.rs::int_literal_members`'s exact twin (see that
/// function's doc for the all-or-nothing member-list rule).
fn int_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(int_literal_value).collect();
    }
    Some(vec![int_literal_value(slice)?])
}

/// One `Literal[...]` member read as an int, with unary minus —
/// `typereading.rs::int_literal_value`'s exact twin.
fn int_literal_value(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => Some(-int_literal_value(unary.operand.as_ref())?),
        _ => None,
    }
}

/// `Literal[...]`'s slice read as a list of STRING-literal members —
/// `typereading.rs::string_literal_members`'s exact twin.
fn string_literal_members(slice: &Expr) -> Option<Vec<String>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(string_literal_value).collect();
    }
    Some(vec![string_literal_value(slice)?])
}

/// One `Literal[...]` member read as a plain string literal —
/// `typereading.rs::string_literal_value`'s exact twin.
fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
        _ => None,
    }
}

/// The UNION of every member's own singleton string tuple —
/// `typereading.rs::string_literal_set`'s exact twin.
fn string_literal_set(members: &[String]) -> RefinedSet {
    let mut set = string_tuple(&members[0]);
    for member in &members[1..] {
        set = make_refined_set(vec![union(set, string_tuple(member))]);
    }
    set
}

/// `Annotated[int|float|str, Field(…), …]` → the stated set, resolved
/// against the module's import identities. The `Annotated` head name
/// must itself resolve to an import of `typing.Annotated` (or
/// `typing_extensions.Annotated`) — a bare `Annotated` that was never
/// imported is not recognized. The `int` sort carries the integer form
/// (int ≠ float is a product law); the `str` sort carries the string
/// ground (`C*`, codepoint_sets::strings) so a bare `Annotated[str,
/// Field(…)]` with no length/pattern kwarg still names a set (every
/// string). Every metadata element must be a recognized `Field(…)`
/// call (by import identity, not spelling) or the alias refuses.
///
/// `min_length`/`max_length` fold into ONE repetition window over the
/// codepoint ground rather than stacking a form per kwarg — pydantic
/// itself reads them as one window's two edges
/// (`StringConstraints`/`Len`, PYREFLY-PYDANTIC-SURFACE.md §2.3), and
/// `tighten_repetition`'s own reading of chained `.min`/`.max` folds
/// the same way. `pattern` intersects the compiled grammar set
/// (`format_grammar`, unanchored search semantics per
/// AGENT-BRIEF.md's pydantic surface facts) as its own conjoined form
/// — a length window and a pattern on the same alias both hold at
/// once, exactly like pydantic validates both constraints on the same
/// field.
pub fn annotated_expression_set(value: &Expr, imports: &SurfaceImports) -> Option<RefinedSet> {
    let Expr::Subscript(subscript) = value else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !imports.annotated_names.contains(head.id.as_str()) {
        return None;
    }
    let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
        return None;
    };
    let (base, metadata) = arguments.elts.split_first()?;
    let is_string_sort = matches!(base, Expr::Name(sort) if sort.id.as_str() == "str");
    let mut forms: Vec<Refinement> = match base {
        Expr::Name(sort) if sort.id.as_str() == "int" => vec![integer()],
        // `StrictInt` (pydantic's own strict-mode int type, imported by
        // identity via `imports.strict_int_names`) carries the same
        // integer form as bare `int` — strictness itself (no str-to-int
        // coercion) is not a SET fact this table's refinement carries;
        // check.rs's adapter route consults `imports.strict_int_names`
        // directly to decide whether a coercion is admitted.
        Expr::Name(sort) if imports.strict_int_names.contains(sort.id.as_str()) => vec![integer()],
        Expr::Name(sort) if sort.id.as_str() == "float" => vec![],
        // RefinedSet carries an iterative Drop, so `.forms` cannot move
        // out of a set — std::mem::take is the house pattern (AGENT-BRIEF)
        Expr::Name(sort) if sort.id.as_str() == "str" => std::mem::take(&mut strings().forms),
        _ => return None,
    };
    let mut min_length: Option<i64> = None;
    let mut max_length: Option<i64> = None;
    for meta in metadata {
        let Expr::Call(call) = meta else {
            return None;
        };
        if names_field(call.func.as_ref(), imports) {
            for keyword in call.arguments.keywords.iter() {
                let name = keyword.arg.as_ref()?;
                match name.as_str() {
                    "ge" => forms.push(at_least(literal_number(&keyword.value)?)),
                    "gt" => forms.push(above(literal_number(&keyword.value)?)),
                    "le" => forms.push(at_most(literal_number(&keyword.value)?)),
                    "lt" => forms.push(below(literal_number(&keyword.value)?)),
                    "multiple_of" => forms.push(multiple_of(literal_number(&keyword.value)?)),
                    "min_length" if is_string_sort => {
                        min_length = Some(literal_length(&keyword.value)?);
                    }
                    "max_length" if is_string_sort => {
                        max_length = Some(literal_length(&keyword.value)?);
                    }
                    "pattern" if is_string_sort => {
                        let pattern = literal_string(&keyword.value)?;
                        let mut grammar = format_grammar(pattern, "");
                        if !grammar.ok {
                            // a pattern this table cannot compile refuses the
                            // WHOLE alias, the same decline the table gives
                            // any other unrecognized kwarg — never a partial
                            // set missing the pattern conjunct.
                            return None;
                        }
                        forms.extend(std::mem::take(&mut grammar.set.forms));
                    }
                    other if INERT_FIELD_KWARGS.contains(&other) => {}
                    _ => return None,
                }
            }
            continue;
        }
        // `annotated_types` constructors (`Ge`/`Gt`/`Le`/`Lt`/
        // `MultipleOf`/`MinLen`/`MaxLen`, each a one-positional-argument
        // dataclass — probe-verified against the installed
        // `annotated_types` package's own `inspect.signature`, 2026-08-17)
        // — pydantic recognizes these as `Field(...)`'s own kwarg
        // equivalents when they appear as `Annotated[...]` metadata
        // (execution-verified against pydantic 2.13.4: `Annotated[int,
        // Ge(0), Le(120)]` enforces the identical bound `Field(ge=0,
        // le=120)` would). `MinLen`/`MaxLen` route through the SAME
        // `min_length`/`max_length` window variables `Field`'s own kwargs
        // set, so a `MinLen(...)` and a later `Field(max_length=...)` on
        // the same alias (m-pydantic-schema.py's `LabelAT`) still fold
        // into one window.
        let Some((argument, at_kind)) = annotated_types_argument(call, imports) else {
            return None;
        };
        match at_kind {
            AnnotatedTypesCtor::Ge => forms.push(at_least(literal_number(argument)?)),
            AnnotatedTypesCtor::Gt => forms.push(above(literal_number(argument)?)),
            AnnotatedTypesCtor::Le => forms.push(at_most(literal_number(argument)?)),
            AnnotatedTypesCtor::Lt => forms.push(below(literal_number(argument)?)),
            AnnotatedTypesCtor::MultipleOf => forms.push(multiple_of(literal_number(argument)?)),
            AnnotatedTypesCtor::MinLen if is_string_sort => {
                min_length = Some(literal_length(argument)?);
            }
            AnnotatedTypesCtor::MaxLen if is_string_sort => {
                max_length = Some(literal_length(argument)?);
            }
            AnnotatedTypesCtor::MinLen | AnnotatedTypesCtor::MaxLen => return None,
        }
    }
    if min_length.is_some() || max_length.is_some() {
        // the window REPLACES the plain C* ground rather than joining
        // it (a length window is strictly tighter), so the ground
        // conjunct is dropped before the window is added — leaving
        // exactly one repetition form when no pattern conjunct is
        // present, and the pattern conjunct plus the window when one
        // is (codepoint_sets::without_string_ground keeps the ground
        // when it is the ONLY form, the opposite of what a REPLACING
        // window needs, so this drops it unconditionally instead).
        let mut window = repetition(strings_codepoint_ground(), min_length.unwrap_or(0), max_length);
        let ground = strings();
        let plain_ground = &ground.forms[0];
        forms.retain(|f| f != plain_ground);
        forms.extend(std::mem::take(&mut window.forms));
    }
    Some(make_refined_set(forms))
}

/// The codepoint ground (`C`, one scalar) `min_length`/`max_length`
/// repeat over — `repetition_window_forms::repetition` takes the
/// ELEMENT set, not the already-starred string ground.
fn strings_codepoint_ground() -> RefinedSet {
    refined_sets::codepoint_sets::codepoints()
}

/// A metadata call names pydantic's `Field` when its callee is either
/// a bare name that imports resolved to `Field`, or an attribute whose
/// base is a name that imports resolved to the pydantic module and
/// whose attribute is literally `Field`. A `Field` defined locally or
/// imported from any other module never matches either shape.
fn names_field(func: &Expr, imports: &SurfaceImports) -> bool {
    match func {
        Expr::Name(n) => imports.field_names.contains(n.id.as_str()),
        Expr::Attribute(a) => {
            a.attr.as_str() == "Field"
                && matches!(a.value.as_ref(), Expr::Name(base) if imports.pydantic_modules.contains(base.id.as_str()))
        }
        _ => false,
    }
}

/// One `annotated_types` constructor this table lowers — each is a
/// one-positional-argument dataclass (probe-verified against the
/// installed package, 2026-08-17: `Ge(ge)`, `Gt(gt)`, `Le(le)`,
/// `Lt(lt)`, `MultipleOf(multiple_of)`, `MinLen(min_length)`,
/// `MaxLen(max_length)`), so the constructor identity alone decides
/// which `Field` kwarg it plays.
#[derive(Clone, Copy)]
enum AnnotatedTypesCtor {
    Ge,
    Gt,
    Le,
    Lt,
    MultipleOf,
    MinLen,
    MaxLen,
}

/// A metadata call names a recognized `annotated_types` constructor —
/// by IMPORT IDENTITY, the same discipline `names_field` holds for
/// `Field` — and reads its single positional argument. `None` for a
/// call with zero or more-than-one positional argument, any keyword
/// argument (`annotated_types`' own constructors take one positional
/// value; a keyword-spelled call is not the shape these rows use), or a
/// callee this table's import table does not name.
fn annotated_types_argument<'a>(call: &'a ruff_python_ast::ExprCall, imports: &SurfaceImports) -> Option<(&'a Expr, AnnotatedTypesCtor)> {
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [argument] = call.arguments.args.as_ref() else {
        return None;
    };
    let kind = if imports.annotated_types_ge.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Ge
    } else if imports.annotated_types_gt.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Gt
    } else if imports.annotated_types_le.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Le
    } else if imports.annotated_types_lt.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::Lt
    } else if imports.annotated_types_multiple_of.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::MultipleOf
    } else if imports.annotated_types_min_len.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::MinLen
    } else if imports.annotated_types_max_len.contains(callee.id.as_str()) {
        AnnotatedTypesCtor::MaxLen
    } else {
        return None;
    };
    Some((argument, kind))
}

/// The import identities the surface resolves names against: which
/// local names mean pydantic's `Field`, which local names mean the
/// pydantic module itself, which local names mean `Annotated` (from
/// `typing` or `typing_extensions`), which local name means pydantic's
/// `StrictInt`, and one set per recognized `annotated_types`
/// constructor (`Ge`/`Gt`/`Le`/`Lt`/`MultipleOf`/`MinLen`/`MaxLen`).
pub struct SurfaceImports {
    field_names: HashSet<String>,
    pydantic_modules: HashSet<String>,
    annotated_names: HashSet<String>,
    strict_int_names: HashSet<String>,
    annotated_types_ge: HashSet<String>,
    annotated_types_gt: HashSet<String>,
    annotated_types_le: HashSet<String>,
    annotated_types_lt: HashSet<String>,
    annotated_types_multiple_of: HashSet<String>,
    annotated_types_min_len: HashSet<String>,
    annotated_types_max_len: HashSet<String>,
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
    }
}

/// A numeric literal, with unary minus — the readable-RHS gate for
/// this slice. None anywhere else (an unread value declines, it never
/// guesses).
pub fn literal_number(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64().map(|v| v as f64),
            Number::Float(f) => Some(*f),
            Number::Complex { .. } => None,
        },
        Expr::UnaryOp(unary) if unary.op == UnaryOp::USub => {
            Some(-literal_number(unary.operand.as_ref())?)
        }
        _ => None,
    }
}

/// A plain (non-f-string) string literal — the readable-RHS gate for
/// `pattern=r"…"`. None anywhere else, matching `literal_number`'s
/// decline-don't-guess discipline.
fn literal_string(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str()),
        _ => None,
    }
}

/// `min_length`/`max_length`'s literal int argument — pydantic's own
/// `StringConstraints`/`Field` types these as `int`, never a float, so
/// a fractional or non-literal value declines rather than truncating.
fn literal_length(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::NumberLiteral(literal) => match &literal.value {
            Number::Int(i) => i.as_i64(),
            Number::Float(_) | Number::Complex { .. } => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(source: &str) -> ModModule {
        ruff_python_parser::parse_module(source)
            .expect("fixture source parses")
            .into_syntax()
    }

    /// `Field as F` — the import alias is still recognized as `Field`.
    #[test]
    fn field_import_alias_recognized() {
        let module = parsed(
            "from pydantic import Field as F\n\
             from typing import Annotated\n\
             type Age = Annotated[int, F(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(out.contains_key("Age"));
    }

    /// `import pydantic as p` + `p.Field(...)` — the module alias is
    /// still recognized as the pydantic module.
    #[test]
    fn pydantic_module_alias_recognized() {
        let module = parsed(
            "import pydantic as p\n\
             from typing import Annotated\n\
             type Age = Annotated[int, p.Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(out.contains_key("Age"));
    }

    /// A locally defined `Field` shadowing the name is never a pydantic
    /// `Field` — no alias compiles.
    #[test]
    fn locally_defined_field_not_recognized() {
        let module = parsed(
            "from typing import Annotated\n\
             def Field(**kwargs):\n\
             \x20\x20\x20\x20pass\n\
             type Age = Annotated[int, Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    /// `from mylib import Field` — a same-named import from any other
    /// module is never recognized as pydantic's `Field`.
    #[test]
    fn field_from_other_module_not_recognized() {
        let module = parsed(
            "from mylib import Field\n\
             from typing import Annotated\n\
             type Age = Annotated[int, Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    /// `Annotated` used bare, with no import naming it, is never
    /// recognized.
    #[test]
    fn annotated_without_import_not_recognized() {
        let module = parsed(
            "from pydantic import Field\n\
             type Age = Annotated[int, Field(ge=0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    /// `type Adult = Age` — alias-of-alias compiles both names to the
    /// same set.
    #[test]
    fn alias_of_alias_compiles_both_names() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Age = Annotated[int, Field(ge=0)]\n\
             type Adult = Age\n",
        );
        let out = compile_aliases(&module);
        assert!(out.contains_key("Age"));
        assert!(out.contains_key("Adult"));
        assert_eq!(out.get("Age"), out.get("Adult"));
    }

    /// An anchored `pattern=r"^[0-9a-f]+$"` compiles — the alias's set
    /// is exactly what `format_grammar` gives the same pattern string
    /// directly, so a matching literal ("1a2b", o-file's in-set row)
    /// and a non-matching one ("zz", the o-file's out-of-set row) judge
    /// against the identical compiled set the standalone grammar
    /// reader would give either literal.
    #[test]
    fn anchored_pattern_compiles_to_the_grammar_reader_own_set() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Hex = Annotated[str, Field(min_length=1, max_length=6, pattern=r\"^[0-9a-f]+$\")]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("Hex").expect("Hex compiles");
        let direct = format_grammar("^[0-9a-f]+$", "");
        assert!(direct.ok);
        // the pattern conjunct is present verbatim in the compiled
        // alias's forms (matching o-file's "1a2b" is a hex string, "zz"
        // is not — both judge against this same conjunct at check time)
        assert!(
            compiled.forms.iter().any(|f| direct.set.forms.contains(f)),
            "the anchored pattern's own compiled form must appear in Hex's forms"
        );
    }

    /// An unanchored `pattern=r"^id-"` (anchored only at the start, the
    /// o-file's `Anchored` row) compiles to a set whose top-level shape
    /// is the padded concatenation `format_grammar` gives that same
    /// pattern directly (prefix, then any suffix) — not the exact
    /// two-sided anchored shape.
    #[test]
    fn unanchored_pattern_pads_the_open_side() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Anchored = Annotated[str, Field(min_length=3, max_length=10, pattern=r\"^id-\")]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("Anchored").expect("Anchored compiles");
        let direct = format_grammar("^id-", "");
        assert!(direct.ok);
        assert!(
            compiled.forms.iter().any(|f| direct.set.forms.contains(f)),
            "the unanchored pattern's own padded form must appear in Anchored's forms"
        );
    }

    /// A pattern `format_grammar` refuses (a backreference, which does
    /// not denote a regular language) declines the WHOLE alias — no
    /// partial set missing just the pattern conjunct.
    #[test]
    fn a_pattern_the_grammar_refuses_declines_the_whole_alias() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Bad = Annotated[str, Field(min_length=1, pattern=r\"(a)\\1\")]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Bad"));
    }

    /// `min_length`/`max_length` on a `str` alias (the o-file's
    /// `Handle` row) compile to ONE repetition window over the
    /// codepoint ground — `as_repetition` reads the compiled set back
    /// with the exact [lo, hi] the two kwargs stated.
    #[test]
    fn string_length_window_compiles_to_one_repetition_form() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Handle = Annotated[str, Field(min_length=2, max_length=6)]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("Handle").expect("Handle compiles");
        let read_back = refined_sets::repetition_window_forms::as_repetition(compiled)
            .expect("a length-window-only str alias reads back as one repetition");
        assert_eq!(read_back.lo, 2);
        assert_eq!(read_back.hi, Some(6));
    }

    /// `min_length` with no `max_length` (an open ceiling) reads back
    /// unbounded on the high side.
    #[test]
    fn string_min_length_alone_is_an_open_upper_bound() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type AtLeastTwo = Annotated[str, Field(min_length=2)]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("AtLeastTwo").expect("AtLeastTwo compiles");
        let read_back = refined_sets::repetition_window_forms::as_repetition(compiled)
            .expect("a min_length-only str alias reads back as one repetition");
        assert_eq!(read_back.lo, 2);
        assert_eq!(read_back.hi, None);
    }

    /// An unrecognized kwarg on a `str` alias (`json_schema_extra`,
    /// never on the inert list and never a bound) declines the whole
    /// alias — the same discipline as the existing int-sort test
    /// `an_alias_the_table_cannot_lower_declines_whole` in check.rs.
    #[test]
    fn an_unrecognized_string_kwarg_declines_the_whole_alias() {
        let module = parsed(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             type Odd = Annotated[str, Field(min_length=1, json_schema_extra={})]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Odd"));
    }

    // --- Literal alias / Literal-union alias (m-pydantic-schema.py's Pick/PickUnion) ---

    /// `type Pick = Literal[10, 20, 30]` compiles to a `one_of` set over
    /// exactly those three members.
    #[test]
    fn a_bare_int_literal_alias_compiles_to_one_of_its_members() {
        let module = parsed("from typing import Literal\ntype Pick = Literal[10, 20, 30]\n");
        let out = compile_aliases(&module);
        let compiled = out.get("Pick").expect("Pick compiles");
        assert_eq!(
            compiled,
            &make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0, 30.0])])
        );
    }

    /// `type PickUnion = Literal[10, 20, 30] | Literal["ten", "twenty"]`
    /// compiles to the union of the int-Literal's `one_of` and the
    /// string-Literal's own tuple union.
    #[test]
    fn a_literal_union_alias_compiles_to_the_union_of_both_arms() {
        let module = parsed(
            "from typing import Literal\n\
             type PickUnion = Literal[10, 20, 30] | Literal[\"ten\", \"twenty\"]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("PickUnion").expect("PickUnion compiles");
        let int_arm = make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0, 30.0])]);
        let string_arm = make_refined_set(vec![union(
            string_tuple("ten"),
            string_tuple("twenty"),
        )]);
        assert_eq!(compiled, &make_refined_set(vec![union(int_arm, string_arm)]));
    }

    /// A union of a Literal arm and a non-Literal arm declines whole —
    /// `literal_union_alias_set` only reads a TWO-Literal-arm union.
    #[test]
    fn a_literal_union_with_a_non_literal_arm_declines() {
        let module = parsed(
            "from typing import Literal\n\
             type Bad = Literal[10, 20] | int\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Bad"));
    }

    // --- annotated_types constructors (Ge/Gt/Le/Lt/MultipleOf/MinLen/MaxLen) ---

    /// `Annotated[int, Ge(0), Le(120)]` compiles the same set
    /// `Annotated[int, Field(ge=0, le=120)]` would — m-pydantic-schema.py's
    /// `AgeAT` shape.
    #[test]
    fn ge_and_le_constructors_compile_the_same_set_field_kwargs_would() {
        let module = parsed(
            "from annotated_types import Ge, Le\n\
             from typing import Annotated\n\
             type AgeAT = Annotated[int, Ge(0), Le(120)]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("AgeAT").expect("AgeAT compiles");
        assert_eq!(
            compiled,
            &make_refined_set(vec![integer(), at_least(0.0), at_most(120.0)])
        );
    }

    /// `Annotated[str, MinLen(1), Field(max_length=8)]` — a `MinLen`
    /// constructor and a `Field(max_length=...)` kwarg on the SAME alias
    /// fold into one repetition window, m-pydantic-schema.py's `LabelAT`
    /// shape.
    #[test]
    fn min_len_constructor_and_field_max_length_fold_into_one_window() {
        let module = parsed(
            "from annotated_types import MinLen\n\
             from pydantic import Field\n\
             from typing import Annotated\n\
             type LabelAT = Annotated[str, MinLen(1), Field(max_length=8)]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("LabelAT").expect("LabelAT compiles");
        let read_back = refined_sets::repetition_window_forms::as_repetition(compiled)
            .expect("MinLen + Field(max_length) folds to one repetition window");
        assert_eq!(read_back.lo, 1);
        assert_eq!(read_back.hi, Some(8));
    }

    /// `Gt`/`Lt`/`MultipleOf` each recognized by their own import
    /// identity, matching `Field`'s `gt`/`lt`/`multiple_of` kwargs.
    #[test]
    fn gt_lt_and_multiple_of_constructors_compile_the_matching_forms() {
        let module = parsed(
            "from annotated_types import Gt, Lt, MultipleOf\n\
             from typing import Annotated\n\
             type EvenAge = Annotated[int, Gt(0), Lt(120), MultipleOf(2)]\n",
        );
        let out = compile_aliases(&module);
        let compiled = out.get("EvenAge").expect("EvenAge compiles");
        assert_eq!(
            compiled,
            &make_refined_set(vec![integer(), above(0.0), below(120.0), multiple_of(2.0)])
        );
    }

    /// An `annotated_types` constructor imported from any OTHER module
    /// (never `annotated_types` itself) is not recognized — the same
    /// import-identity discipline `names_field` already holds for `Field`.
    #[test]
    fn an_annotated_types_name_from_another_module_is_not_recognized() {
        let module = parsed(
            "from mylib import Ge\n\
             from typing import Annotated\n\
             type Age = Annotated[int, Ge(0)]\n",
        );
        let out = compile_aliases(&module);
        assert!(!out.contains_key("Age"));
    }

    // --- StrictInt base sort / strict_int_alias_names ---

    /// `type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]`
    /// compiles the identical integer-ground set a plain `int` base would
    /// — strictness is not a SET fact, it is check.rs's own coercion-gate
    /// consult against `strict_int_alias_names`.
    #[test]
    fn strict_int_base_compiles_the_same_set_a_plain_int_base_would() {
        let module = parsed(
            "from pydantic import Field, StrictInt\n\
             from typing import Annotated\n\
             type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]\n\
             type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
        );
        let out = compile_aliases(&module);
        assert_eq!(out.get("StrictAge"), out.get("LaxAge"));
    }

    /// `strict_int_alias_names` names exactly the `StrictInt`-based alias,
    /// never the plain `int`-based one.
    #[test]
    fn strict_int_alias_names_names_only_the_strict_int_based_alias() {
        let module = parsed(
            "from pydantic import Field, StrictInt\n\
             from typing import Annotated\n\
             type StrictAge = Annotated[StrictInt, Field(ge=0, le=120)]\n\
             type LaxAge = Annotated[int, Field(ge=0, le=120)]\n",
        );
        let strict_names = strict_int_alias_names(&module);
        assert!(strict_names.contains("StrictAge"));
        assert!(!strict_names.contains("LaxAge"));
    }
}
