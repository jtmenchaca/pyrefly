/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Annotation expressions read into declared refinements: alias names
//! (only where visible), inline `Annotated[...]` forms, alias-of-alias,
//! `X | None` / `Optional[X]` (admits-None wrapping the inner read),
//! and string annotations. This file is the contract the walk calls;
//! the typereading unit fills it in behind these signatures.
//!
//! A None never approximates a set — it declines to state one, and the
//! walk decides what silence says at the call site. Nothing here widens
//! or guesses a set it cannot read exactly (the same discipline as
//! refined-ts-go's typereading package: refuse rather than approximate).

use std::collections::HashMap;

use refined_sets::format_for_diagnostics::format_for_diagnostics;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::RefinedSet;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::UnaryOp;
use ruff_python_parser::parse_expression;

use crate::refinedpy::env::Environment;
use crate::refinedpy::surface::SurfaceImports;
use crate::refinedpy::surface::annotated_expression_set;

/// A refinement an annotation states, with the spelling diagnostics
/// use for it (the alias name, or the formatted set for inline forms).
#[derive(Clone)]
pub struct DeclaredRefinement {
    pub set: RefinedSet,
    pub spelling: String,
    /// true when the annotation admits None alongside the set (`X |
    /// None`, `Optional[X]`) — Kind::Null judges Silent against such a
    /// declaration; the set describes the non-None values.
    pub admits_none: bool,
    /// The per-member refinement of a container declaration —
    /// `dict[str, X]`'s VALUE slot; the outer `set` field is unused
    /// (empty) when `element` is Some.
    pub element: Option<Box<DeclaredRefinement>>,
}

/// The refinement this annotation states here, or None when it states
/// none this table can read. A None never approximates — it declines.
pub fn declared_refinement(
    annotation: &Expr,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    match annotation {
        // A bare name: only a visible module-level alias states anything.
        // A name the body rebinds locally means something else here, so
        // the alias table must not be consulted for it — the walk owns
        // saying why a rebound name reads as unrefined.
        Expr::Name(name) => {
            let spelling = name.id.as_str();
            if !environment.alias_is_visible(spelling) {
                return None;
            }
            let set = aliases.get(spelling)?;
            Some(DeclaredRefinement {
                set: set.clone(),
                spelling: spelling.to_owned(),
                admits_none: false,
                element: None,
            })
        }
        // `Optional[X]` reads X through the ordinary path and marks the
        // result as admitting None — "Optional[X] is equivalent to X |
        // None" (tmp/cpython Doc/library/typing.rst, "Optional").
        // Recognized by bare name only: `SurfaceImports` carries no
        // `typing.Optional` import identity today (only `Field` and
        // `Annotated`), so this matches the same bare-name convention
        // `annotated_expression_set` uses for `int`/`float`/`str` sort
        // names rather than gating on an identity this table cannot
        // check. `subscript.slice` is the single argument directly (not
        // a `Tuple` — ruff wraps in `Tuple` only for multi-element
        // subscripts like `Annotated[a, b]`). An inline `Annotated[...]`
        // form (the OTHER subscript head this table reads) is
        // unaffected: the surface unit owns recognizing and lowering it
        // against the module's import identities, and the spelling
        // there is the set itself, formatted, since no alias name
        // stands for it.
        Expr::Subscript(subscript) => {
            let is_optional = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Optional");
            if is_optional {
                let mut declared =
                    declared_refinement(subscript.slice.as_ref(), aliases, imports, environment)?;
                declared.admits_none = true;
                return Some(declared);
            }
            // `Literal[...]` (tmp/cpython Doc/library/typing.rst,
            // "Literal"): recognized by bare name only, the same
            // no-import-identity convention `Optional` above takes —
            // `SurfaceImports` carries no `typing.Literal` identity
            // today either. `subscript.slice` is either one bare member
            // (`Literal[40]`, no `Tuple` wrap — ruff only wraps a
            // MULTI-element subscript, the same rule
            // `annotated_expression_set` documents for `Annotated`) or
            // an `Expr::Tuple` of them (`Literal[10, 20]`). INT members
            // build a numeric `one_of` (`int_literal_members`); STRING
            // members build the union of each member's own singleton
            // string tuple (`string_literal_members` /
            // `string_literal_set`) — the two wire shapes cannot share
            // one reading, since a string member's code points would
            // collide with `one_of`'s numeric encoding, so each sort
            // gets its OWN member reader and only one of the two may
            // recognize a given member list; a MIXED int/string
            // `Literal[...]` matches neither reader (every element of
            // `int_literal_members`'s map must be int, every element of
            // `string_literal_members`'s map must be string) and
            // declines whole. Any other non-literal member (a name, an
            // expression, a bool, a float, a bytes literal) declines
            // both readers too, same as `annotated_expression_set`'s
            // own metadata gate.
            let is_literal = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Literal");
            if is_literal {
                if let Some(members) = int_literal_members(subscript.slice.as_ref()) {
                    let set = make_refined_set(vec![one_of(&members)]);
                    let spelling = format_for_diagnostics(&set);
                    return Some(DeclaredRefinement {
                        set,
                        spelling,
                        admits_none: false,
                        element: None,
                    });
                }
                if let Some(members) = string_literal_members(subscript.slice.as_ref()) {
                    let set = string_literal_set(&members);
                    let spelling = format_for_diagnostics(&set);
                    return Some(DeclaredRefinement {
                        set,
                        spelling,
                        admits_none: false,
                        element: None,
                    });
                }
                return None;
            }
            // `dict[str, X]`: the container itself carries no scalar
            // set — its VALUE SLOT does. Recognized by bare-Name head
            // `dict` with a two-element `Tuple` slice (ruff always
            // wraps a multi-element subscript, the same rule
            // `Callable[[...], R]` above documents) whose FIRST member
            // is the bare name `str` (the only key sort this reader
            // states a member law for) and whose SECOND member reads
            // through the ordinary `declared_refinement` recursion —
            // so `dict[str, Age]` reads `Age` exactly, including a
            // nested `X | None` value slot. Any other shape (a
            // non-`str` key, an unreadable value type, no `Tuple` at
            // all) declines this arm and falls through to
            // `annotated_expression_set` below, which also declines
            // (its own head-identity gate never matches `dict`) — so
            // the whole subscript states nothing, as it did before this
            // arm existed.
            let is_dict = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "dict");
            if is_dict {
                if let Expr::Tuple(arguments) = subscript.slice.as_ref() {
                    if let [key, value] = arguments.elts.as_slice() {
                        let key_is_str = matches!(key, Expr::Name(sort) if sort.id.as_str() == "str");
                        if key_is_str {
                            if let Some(value_declared) =
                                declared_refinement(value, aliases, imports, environment)
                            {
                                let spelling = format!("dict[str, {}]", value_declared.spelling);
                                return Some(DeclaredRefinement {
                                    set: make_refined_set(Vec::new()),
                                    spelling,
                                    admits_none: false,
                                    element: Some(Box::new(value_declared)),
                                });
                            }
                        }
                    }
                }
                return None;
            }
            // `list[X]` / `set[X]` — the same one-element-slot shape
            // `dict[str, X]` reads for its VALUE slot: the container
            // itself carries no scalar set, its ELEMENT does. The slice
            // is the single element annotation directly (no Tuple wrap
            // for a one-argument subscript, the same ruff rule the
            // Optional arm above documents).
            let is_element_container = matches!(
                subscript.value.as_ref(),
                Expr::Name(head) if head.id.as_str() == "list" || head.id.as_str() == "set"
            );
            if is_element_container {
                let head = match subscript.value.as_ref() {
                    Expr::Name(head) => head.id.as_str(),
                    _ => unreachable!("matched Name above"),
                };
                if let Some(element_declared) =
                    declared_refinement(subscript.slice.as_ref(), aliases, imports, environment)
                {
                    let spelling = format!("{}[{}]", head, element_declared.spelling);
                    return Some(DeclaredRefinement {
                        set: make_refined_set(Vec::new()),
                        spelling,
                        admits_none: false,
                        element: Some(Box::new(element_declared)),
                    });
                }
                return None;
            }
            let set = annotated_expression_set(annotation, imports)?;
            let spelling = format_for_diagnostics(&set);
            Some(DeclaredRefinement {
                set,
                spelling,
                admits_none: false,
                element: None,
            })
        }
        // `X | None` / `None | X` (exactly one side a bare `None`
        // literal): read the OTHER side through the ordinary path and
        // mark the result as admitting None — the `X | None` union
        // syntax (tmp/cpython Doc/library/stdtypes.rst, "Union Type",
        // `types.UnionType`). `None | None` and a union of two non-None
        // shapes (`Age | Label`) are a different unit — a general union
        // of two sets — and decline here exactly as before.
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                // both None, or neither — not this table's one-sided form
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            let mut declared = declared_refinement(other, aliases, imports, environment)?;
            declared.admits_none = true;
            Some(declared)
        }
        // A string annotation (a forward reference or a quoted form):
        // parse the string's own contents as an expression and recurse
        // on what it parses to. A string that fails to parse as an
        // expression states nothing this table can read.
        Expr::StringLiteral(literal) => {
            let source = literal.value.to_str();
            let parsed = parse_expression(source).ok()?;
            let inner = parsed.into_syntax().body;
            declared_refinement(&inner, aliases, imports, environment)
        }
        // Every other shape (a general union of two non-None sets,
        // attribute paths, calls, …) is not a form this table reads.
        // None here declines judgment; it is never read as "no
        // refinement applies".
        _ => None,
    }
}

/// A CALLABLE-VARIABLE's own RETURN refinement: `Callable[[...], R]`
/// (typing's `Callable`, tmp/cpython Doc/library/typing.rst,
/// "Callable" — `Callable[[int], str]` is "a function of (int) ->
/// str"), read the same bare-name-subscript-head way `Literal`/
/// `Optional` are above (no `SurfaceImports` identity for `Callable`
/// exists yet either), plus its `| None` wrapper (`X | None`/
/// `Optional[X]`). `admits_none` on the RETURNED `DeclaredRefinement`
/// here is never set true by the `| None` wrapper: that `None` means
/// the CALLABLE VARIABLE itself may be `None` (a fact `env.rs`'s
/// caller judges at the call site — a call through a possibly-None
/// callable additionally RAISES if the variable actually holds `None`,
/// out of this function's scope), not that `R` admits `None` — `R`'s
/// own refinement is read through the ordinary `declared_refinement`
/// path (so `Callable[[int], Age]` reads `Age` exactly, including ITS
/// own `admits_none` if `Age` were `Optional`), falling back to the
/// same bare `int`/`float`/`str` base-sort reading
/// `summaries.rs::return_sort_fallback` gives a declined call's return
/// annotation — matched here to the identical sets (`int` → the
/// unbounded whole-number ray, `float` → the unbounded real ray,
/// `str` → the whole-strings ground) so a callable-typed slot and an
/// ordinary same-module `def`'s declined body agree on what a bare
/// base-sort return annotation states.
pub fn callable_return_refinement(
    annotation: &Expr,
    aliases: &HashMap<String, RefinedSet>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    match annotation {
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            // the variable's OWN possible-None-ness is not carried onto
            // its return refinement — see the doc comment above.
            callable_return_refinement(other, aliases, imports, environment)
        }
        Expr::Subscript(subscript) => {
            let is_callable = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Callable");
            if !is_callable {
                return None;
            }
            let Expr::Tuple(arguments) = subscript.slice.as_ref() else {
                return None;
            };
            // `Callable[[params...], R]` — ruff always wraps the
            // two-element (params-list, return) slice in a Tuple; the
            // params element itself must be a `List` (an ellipsis
            // `Callable[..., R]` is a different, unparameterized shape
            // this reader does not recognize).
            let [params, returns] = arguments.elts.as_slice() else {
                return None;
            };
            if !matches!(params, Expr::List(_)) {
                return None;
            }
            declared_refinement(returns, aliases, imports, environment)
                .or_else(|| base_sort_return_refinement(returns))
        }
        _ => None,
    }
}

/// The bare `int`/`float`/`str` return-annotation fallback, matched to
/// `summaries.rs::return_sort_fallback`'s own sets exactly: `int` is
/// the unbounded whole-number ray (`integer()` conjoined with the
/// unbounded `at_least(NEG_INFINITY)` ray, the same "no ceiling/floor"
/// shape that fallback builds), `float` is the unbounded real ray
/// (`numbers()`, the same set `float_sorted_unknown()` carries), `str`
/// is the whole-strings ground (`codepoint_sets::strings()`).
/// EXPORTED for check.rs's parameter seeding ONLY: a bare-`int`
/// parameter seeds the whole-int sort claim ("a whole int admits
/// values outside the set", the corpus's own reason). The general
/// declared_refinement table deliberately does NOT read base sorts —
/// doing so made every `-> int` helper return judge, turning each
/// unreadable helper body into a new undetermined blocker.
pub fn base_sort_return_refinement(returns: &Expr) -> Option<DeclaredRefinement> {
    let Expr::Name(sort) = returns else {
        return None;
    };
    let set = match sort.id.as_str() {
        "int" => make_refined_set(vec![
            refined_sets::refinement_forms::integer(),
            refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
        ]),
        "float" => refined_sets::refinement_forms::numbers(),
        "str" => refined_sets::codepoint_sets::strings(),
        _ => return None,
    };
    let spelling = sort.id.as_str().to_owned();
    Some(DeclaredRefinement {
        set,
        spelling,
        admits_none: false,
        element: None,
    })
}

/// `Literal[...]`'s slice read as a list of int-literal members: one
/// bare (possibly negated) `NumberLiteral` for a single-member
/// `Literal[40]`, or every element of an `Expr::Tuple` for
/// `Literal[10, 20]`. `None` the moment any member is not a plain int
/// literal (a string, a bool, a float, a name) — the whole subscript
/// declines rather than reading a partial member list.
fn int_literal_members(slice: &Expr) -> Option<Vec<f64>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(int_literal_value).collect();
    }
    Some(vec![int_literal_value(slice)?])
}

/// One `Literal[...]` member read as an int, with unary minus
/// (`Literal[-1]`) — the same shape `surface.rs::literal_number` reads,
/// but INTEGER ONLY: a `Number::Float` member declines, since a float
/// value can never be a `Literal[...]` member in the typing grammar
/// (`tmp/cpython Doc/library/typing.rst`, "Literal" — "Literal[3.14]" is
/// not a valid Literal parameter in the first place; only int, str,
/// bytes, bool, and None literals are).
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
/// `int_literal_members`'s twin. `None` the moment any member is not a
/// plain `Expr::StringLiteral` (an int, a bool, a name, an f-string) —
/// a MIXED int/string `Literal[...]` declines whole, the same
/// all-or-nothing rule `int_literal_members` already applies to a
/// mixed int/name member list.
fn string_literal_members(slice: &Expr) -> Option<Vec<String>> {
    if let Expr::Tuple(tuple) = slice {
        return tuple.elts.iter().map(string_literal_value).collect();
    }
    Some(vec![string_literal_value(slice)?])
}

/// One `Literal[...]` member read as a plain string literal — no
/// f-string, no concatenation, the same bare shape `int_literal_value`
/// reads on the numeric side.
fn string_literal_value(expr: &Expr) -> Option<String> {
    match expr {
        Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
        _ => None,
    }
}

/// The UNION of every member's own singleton string tuple
/// (`codepoint_sets::string_tuple`) — the unambiguous string-Literal
/// wire shape `int_literal_members` cannot share: a string member's
/// code points would collide with `one_of`'s numeric encoding, so each
/// member gets its own tuple set and the members fold together by
/// `union`, not by one shared `one_of`. A single member is exactly its
/// own tuple set (no union node needed); `members` is never empty —
/// `string_literal_members` always returns at least one element when it
/// returns at all.
fn string_literal_set(members: &[String]) -> RefinedSet {
    let mut set = refined_sets::codepoint_sets::string_tuple(&members[0]);
    for member in &members[1..] {
        set = make_refined_set(vec![union(set, refined_sets::codepoint_sets::string_tuple(member))]);
    }
    set
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use refined_sets::refinement_forms::at_least;
    use refined_sets::refinement_forms::make_refined_set;
    use ruff_python_ast::ExprContext;
    use ruff_python_ast::ExprName;
    use ruff_python_ast::ExprStringLiteral;
    use ruff_python_ast::StringLiteralValue;
    use ruff_text_size::TextRange;

    use super::*;

    fn name_expr(id: &str) -> Expr {
        Expr::Name(ExprName {
            node_index: Default::default(),
            range: TextRange::default(),
            id: id.into(),
            ctx: ExprContext::Load,
        })
    }

    fn string_literal_expr(text: &str) -> Expr {
        Expr::StringLiteral(ExprStringLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
            value: StringLiteralValue::single(ruff_python_ast::StringLiteral {
                range: TextRange::default(),
                node_index: Default::default(),
                value: text.into(),
                flags: ruff_python_ast::StringLiteralFlags::empty(),
            }),
        })
    }

    fn no_locals() -> Environment {
        Environment::new(HashSet::new())
    }

    /// SurfaceImports has private fields — tests build it through the
    /// public constructor on a parsed module with no imports.
    fn no_imports() -> SurfaceImports {
        let module = ruff_python_parser::parse_module("x = 1")
            .expect("test module parses")
            .into_syntax();
        crate::refinedpy::surface::surface_imports(&module)
    }

    #[test]
    fn a_visible_alias_name_resolves_with_its_name_as_spelling() {
        let mut aliases = HashMap::new();
        aliases.insert("PositiveInt".to_owned(), make_refined_set(vec![at_least(1.0)]));
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment)
            .expect("a visible alias resolves");
        assert_eq!(got.spelling, "PositiveInt");
        assert_eq!(got.set, make_refined_set(vec![at_least(1.0)]));
    }

    #[test]
    fn a_locally_rebound_alias_name_states_nothing() {
        let mut aliases = HashMap::new();
        aliases.insert("PositiveInt".to_owned(), make_refined_set(vec![at_least(1.0)]));
        let imports = no_imports();
        let mut locally_bound = HashSet::new();
        locally_bound.insert("PositiveInt".to_owned());
        let environment = Environment::new(locally_bound);

        let got = declared_refinement(&name_expr("PositiveInt"), &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    #[test]
    fn a_string_annotation_naming_a_visible_alias_resolves() {
        let mut aliases = HashMap::new();
        aliases.insert("PositiveInt".to_owned(), make_refined_set(vec![at_least(1.0)]));
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(
            &string_literal_expr("PositiveInt"),
            &aliases,
            &imports,
            &environment,
        )
        .expect("a string annotation naming a visible alias resolves");
        assert_eq!(got.spelling, "PositiveInt");
    }

    #[test]
    fn an_alias_name_not_in_the_table_states_nothing_even_as_one_side_of_a_none_union() {
        // `int | None`: `int` is not a compiled alias in this test's
        // table, so the inner read misses and the whole union declines
        // — the same "alias lookup miss" reason a bare `int` would.
        let union = none_union(name_expr("int"));
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    fn none_literal_expr() -> Expr {
        Expr::NoneLiteral(ruff_python_ast::ExprNoneLiteral {
            node_index: Default::default(),
            range: TextRange::default(),
        })
    }

    fn bin_or(left: Expr, right: Expr) -> Expr {
        Expr::BinOp(ruff_python_ast::ExprBinOp {
            node_index: Default::default(),
            range: TextRange::default(),
            left: Box::new(left),
            op: ruff_python_ast::Operator::BitOr,
            right: Box::new(right),
        })
    }

    /// `<other> | None`.
    fn none_union(other: Expr) -> Expr {
        bin_or(other, none_literal_expr())
    }

    /// `None | <other>` — the reversed side.
    fn union_none(other: Expr) -> Expr {
        bin_or(none_literal_expr(), other)
    }

    /// `Optional[<other>]`.
    fn optional_of(other: Expr) -> Expr {
        Expr::Subscript(ruff_python_ast::ExprSubscript {
            node_index: Default::default(),
            range: TextRange::default(),
            value: Box::new(name_expr("Optional")),
            slice: Box::new(other),
            ctx: ExprContext::Load,
        })
    }

    fn age_aliases() -> HashMap<String, RefinedSet> {
        let mut aliases = HashMap::new();
        aliases.insert("Age".to_owned(), make_refined_set(vec![at_least(0.0)]));
        aliases.insert("Label".to_owned(), make_refined_set(vec![at_least(1.0)]));
        aliases
    }

    #[test]
    fn a_plain_alias_name_reads_with_admits_none_false() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&name_expr("Age"), &aliases, &imports, &environment)
            .expect("Age resolves");
        assert!(!got.admits_none);
    }

    #[test]
    fn age_or_none_reads_age_with_admits_none_true() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&none_union(name_expr("Age")), &aliases, &imports, &environment)
            .expect("Age | None resolves");
        assert_eq!(got.spelling, "Age");
        assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
        assert!(got.admits_none);
    }

    #[test]
    fn none_or_age_reversed_reads_age_with_admits_none_true() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&union_none(name_expr("Age")), &aliases, &imports, &environment)
            .expect("None | Age resolves");
        assert_eq!(got.spelling, "Age");
        assert!(got.admits_none);
    }

    #[test]
    fn optional_age_reads_age_with_admits_none_true() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = declared_refinement(&optional_of(name_expr("Age")), &aliases, &imports, &environment)
            .expect("Optional[Age] resolves");
        assert_eq!(got.spelling, "Age");
        assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
        assert!(got.admits_none);
    }

    #[test]
    fn age_or_label_a_union_of_two_non_none_sets_still_declines_whole() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let union = bin_or(name_expr("Age"), name_expr("Label"));
        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    #[test]
    fn none_or_none_declines_whole() {
        let aliases = HashMap::new();
        let imports = no_imports();
        let environment = no_locals();

        let union = bin_or(none_literal_expr(), none_literal_expr());
        let got = declared_refinement(&union, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    /// `Annotated[int, Field(ge=0)] | None` — the recursion into the
    /// non-None side of a `| None` union reaches an inline `Annotated`
    /// form exactly as it would reach a bare alias name.
    #[test]
    fn annotated_or_none_reads_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "from pydantic import Field\n\
             from typing import Annotated\n\
             x: Annotated[int, Field(ge=0)] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Annotated[int, Field(ge=0)] | None resolves");
        assert!(got.admits_none);
        assert_eq!(got.set, make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(0.0)]));
    }

    /// `Optional[Annotated[int, Field(ge=0)]]` — the recursion into
    /// `Optional[...]`'s inner expression reaches the same inline
    /// `Annotated` form.
    #[test]
    fn optional_of_annotated_reads_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "from pydantic import Field\n\
             from typing import Annotated, Optional\n\
             x: Optional[Annotated[int, Field(ge=0)]] = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Optional[Annotated[int, Field(ge=0)]] resolves");
        assert!(got.admits_none);
        assert_eq!(got.set, make_refined_set(vec![refined_sets::refinement_forms::integer(), at_least(0.0)]));
    }

    /// `Literal[10, 20]` — a multi-member int Literal (ruff wraps the
    /// slice in a `Tuple`) compiles to a `one_of` set over exactly those
    /// two values, admitting neither.
    #[test]
    fn literal_of_two_ints_compiles_to_one_of_those_values() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[10, 20] = 10\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[10, 20] resolves");
        assert!(!got.admits_none);
        assert_eq!(
            got.set,
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0])])
        );
    }

    /// `Literal[40]` — a single-member Literal (no `Tuple` wrap) reads
    /// the same way.
    #[test]
    fn literal_of_one_int_compiles_to_one_of_that_single_value() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[40] = 40\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[40] resolves");
        assert_eq!(
            got.set,
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[40.0])])
        );
    }

    /// `Literal[10, 20] | None` — composes with the existing
    /// `admits_none` machinery for free: the union arm recurses into
    /// this same Literal read, then marks `admits_none` true, exactly
    /// as it does for an alias name or an inline `Annotated` form.
    #[test]
    fn literal_or_none_reads_the_literal_set_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[10, 20] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[10, 20] | None resolves");
        assert!(got.admits_none);
        assert_eq!(
            got.set,
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[10.0, 20.0])])
        );
    }

    /// `Literal["horizontal", "vertical"]` — a multi-member STRING
    /// Literal compiles to the UNION of each member's own singleton
    /// string tuple (`string_literal_set`), the unambiguous form the
    /// numeric `one_of` reader cannot share.
    #[test]
    fn literal_of_two_strings_compiles_to_the_union_of_their_tuples() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[\"horizontal\", \"vertical\"] = \"horizontal\"\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[\"horizontal\", \"vertical\"] resolves");
        assert!(!got.admits_none);
        assert_eq!(
            got.set,
            make_refined_set(vec![union(
                refined_sets::codepoint_sets::string_tuple("horizontal"),
                refined_sets::codepoint_sets::string_tuple("vertical"),
            )])
        );
    }

    /// `Literal["horizontal"]` — a single-member string Literal (no
    /// `Tuple` wrap) reads as exactly that member's own tuple set, no
    /// union node needed.
    #[test]
    fn literal_of_one_string_compiles_to_that_single_tuple() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[\"horizontal\"] = \"horizontal\"\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[\"horizontal\"] resolves");
        assert_eq!(got.set, refined_sets::codepoint_sets::string_tuple("horizontal"));
    }

    /// `Literal["horizontal", "vertical"] | None` — composes with the
    /// existing `admits_none` machinery for free, the string Literal's
    /// own twin of `literal_or_none_reads_the_literal_set_with_admits_none_true`.
    #[test]
    fn string_literal_or_none_reads_the_literal_set_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[\"horizontal\", \"vertical\"] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[\"horizontal\", \"vertical\"] | None resolves");
        assert!(got.admits_none);
        assert_eq!(
            got.set,
            make_refined_set(vec![union(
                refined_sets::codepoint_sets::string_tuple("horizontal"),
                refined_sets::codepoint_sets::string_tuple("vertical"),
            )])
        );
    }

    /// A MIXED int/string `Literal[...]` member list declines whole:
    /// neither `int_literal_members` (one member is a string) nor
    /// `string_literal_members` (one member is an int) matches every
    /// element, so no reading is built for either sort.
    #[test]
    fn a_mixed_int_and_string_literal_declines() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[40, \"horizontal\"] = 40\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    /// A negative int Literal member (`Literal[-1]`) reads through the
    /// same unary-minus recognition `int_literal_value` shares with
    /// `surface.rs::literal_number`.
    #[test]
    fn a_negative_int_literal_member_reads() {
        let module = ruff_python_parser::parse_module(
            "from typing import Literal\n\
             x: Literal[-1, 1] = -1\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("Literal[-1, 1] resolves");
        assert_eq!(
            got.set,
            make_refined_set(vec![refined_sets::refinement_forms::one_of(&[-1.0, 1.0])])
        );
    }

    /// The parsed module's one `AnnAssign`'s annotation — shared by the
    /// two nested-form tests above.
    fn annotated_or_none_annotation(module: &ruff_python_ast::ModModule) -> &Expr {
        for stmt in module.body.iter() {
            if let ruff_python_ast::Stmt::AnnAssign(ann_assign) = stmt {
                return ann_assign.annotation.as_ref();
            }
        }
        panic!("test module has one AnnAssign statement");
    }

    // --- callable_return_refinement ---

    /// `Callable[[int], Age] | None` — b-body-expressions.py:38's own
    /// shape but with a refined return (`Age`, `int` in the row) rather
    /// than the row's plain `int`: the return reads through the
    /// ordinary `declared_refinement` path, and the `| None` wrapper is
    /// dropped from the RETURN refinement (it describes the variable,
    /// not `R`) — `admits_none` on the answer is false.
    #[test]
    fn callable_return_reads_a_declared_alias_return_dropping_the_variable_none() {
        let module = ruff_python_parser::parse_module(
            "from typing import Callable\n\
             f: Callable[[int], Age] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = age_aliases();
        let environment = no_locals();

        let got = callable_return_refinement(annotation, &aliases, &imports, &environment)
            .expect("Callable[[int], Age] | None resolves a return refinement");
        assert!(!got.admits_none);
        assert_eq!(got.spelling, "Age");
        assert_eq!(got.set, make_refined_set(vec![at_least(0.0)]));
    }

    /// b-body-expressions.py:38's EXACT shape:
    /// `Callable[[int], int] | None` — the return has no refined alias,
    /// so it falls back to the bare `int` base sort
    /// (`summaries.rs::return_sort_fallback`'s own unbounded
    /// whole-number ray), matching `call_optional`'s marker ("the
    /// guarded call still admits a whole number outside the set").
    #[test]
    fn callable_return_falls_back_to_the_bare_int_base_sort() {
        let module = ruff_python_parser::parse_module(
            "from typing import Callable\n\
             maybe_next_year: Callable[[int], int] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = callable_return_refinement(annotation, &aliases, &imports, &environment)
            .expect("Callable[[int], int] | None falls back to the int base sort");
        assert!(!got.admits_none);
        assert_eq!(
            got.set,
            make_refined_set(vec![
                refined_sets::refinement_forms::integer(),
                at_least(f64::NEG_INFINITY)
            ])
        );
    }

    /// No `| None` wrapper at all — `Callable[[int], int]` reads the
    /// same return refinement directly.
    #[test]
    fn callable_return_reads_without_the_none_wrapper() {
        let module = ruff_python_parser::parse_module(
            "from typing import Callable\n\
             f: Callable[[int], int] = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = HashMap::new();
        let environment = no_locals();

        let got = callable_return_refinement(annotation, &aliases, &imports, &environment)
            .expect("Callable[[int], int] resolves");
        assert_eq!(
            got.set,
            make_refined_set(vec![
                refined_sets::refinement_forms::integer(),
                at_least(f64::NEG_INFINITY)
            ])
        );
    }

    /// A non-Callable annotation (a plain alias name) declines — this
    /// reader is specific to the `Callable[...]` subscript shape.
    #[test]
    fn a_non_callable_annotation_declines() {
        let aliases = age_aliases();
        let imports = no_imports();
        let environment = no_locals();

        let got = callable_return_refinement(&name_expr("Age"), &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    // --- dict[str, X]'s value-slot reading ---

    /// `dict[str, Age]` — a-statements.py's `return_dict_members` own
    /// shape: the outer declaration carries no set of its own (`element`
    /// Some, `set` empty) and the element is `Age` read through the
    /// ordinary alias recursion.
    #[test]
    fn dict_of_str_to_age_reads_age_as_the_element() {
        let module = ruff_python_parser::parse_module(
            "x: dict[str, Age] = {}\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = age_aliases();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("dict[str, Age] resolves");
        assert!(!got.admits_none);
        assert_eq!(got.spelling, "dict[str, Age]");
        let element = got.element.expect("dict[str, Age] carries an element refinement");
        assert_eq!(element.spelling, "Age");
        assert_eq!(element.set, make_refined_set(vec![at_least(0.0)]));
    }

    /// `dict[str, Age] | None` — composes with the existing
    /// `admits_none` machinery for free: the union arm recurses into
    /// this same dict read, then marks `admits_none` true, without
    /// touching `element`.
    #[test]
    fn dict_of_str_to_age_or_none_reads_the_element_with_admits_none_true() {
        let module = ruff_python_parser::parse_module(
            "x: dict[str, Age] | None = None\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = age_aliases();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment)
            .expect("dict[str, Age] | None resolves");
        assert!(got.admits_none);
        assert_eq!(got.spelling, "dict[str, Age]");
        let element = got.element.expect("dict[str, Age] | None still carries an element refinement");
        assert_eq!(element.spelling, "Age");
    }

    /// `dict[int, Age]` — a non-`str` key declines the whole subscript,
    /// same as any other unrecognized dict shape.
    #[test]
    fn dict_of_int_to_age_declines() {
        let module = ruff_python_parser::parse_module(
            "x: dict[int, Age] = {}\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = age_aliases();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment);
        assert!(got.is_none());
    }

    /// `dict[str, Unreadable]` — a value type this table cannot read
    /// (no alias by that name) declines the whole subscript.
    #[test]
    fn dict_of_str_to_an_unreadable_value_type_declines() {
        let module = ruff_python_parser::parse_module(
            "x: dict[str, Unreadable] = {}\n",
        )
        .expect("test module parses")
        .into_syntax();
        let imports = crate::refinedpy::surface::surface_imports(&module);
        let annotation = annotated_or_none_annotation(&module);
        let aliases = age_aliases();
        let environment = no_locals();

        let got = declared_refinement(annotation, &aliases, &imports, &environment);
        assert!(got.is_none());
    }
}
