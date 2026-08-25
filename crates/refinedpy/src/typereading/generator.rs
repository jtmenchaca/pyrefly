//! One generator-family subscript's slice read into its yield/return
//! checked positions — `Generator`/`AsyncGenerator`/`Iterator`/
//! `Iterable`.

use std::collections::HashMap;

use ruff_python_ast::Expr;

use crate::env::Environment;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;

use super::base_sort::base_sort_return_refinement;
use super::declared_refinement::declared_refinement;
use super::declared_refinement::DeclaredRefinement;
use super::declared_refinement::GeneratorRefinement;

/// One generator-family subscript's slice read into its yield/return
/// checked positions — `declared_refinement`'s own generator arm. `head`
/// is the bare subscript-head name (`Generator`/`AsyncGenerator`/
/// `Iterator`/`Iterable`), already matched as a Name by the caller.
/// `Generator[Y, S, R]` reads a 3-element `Tuple` slice, yield type
/// first, return type third (the SEND type, second, states nothing this
/// reader judges — a generator's `.send()` argument is outside the
/// checker's scope); `AsyncGenerator[Y, S]` reads a 2-element `Tuple`
/// the same way but never carries a return type (datamodel.rst's
/// asynchronous generator functions cannot use a value-carrying
/// `return`); `Iterator[Y]`/`Iterable[Y]` read the single element
/// directly (no `Tuple` wrap for a one-argument subscript) with no
/// return type at all. Any other head name, or a slice shape that does
/// not match the head's own arity, declines — `None`, never a partial
/// reading.
///
/// Each position falls back to `base_sort_return_refinement` when
/// `declared_refinement` itself declines (a bare `int`/`float`/`str`
/// argument, e.g. `Generator[int, None, None]`) — the SAME fallback
/// `callable_return_refinement`'s own `R` position already takes, and
/// for the identical reason: the generator's own annotation is what
/// MAKES a yield/return a checked position in the first place (this
/// file's own module doc), so a bare base-sort argument here must
/// still state its ordinary whole-sort claim rather than silently
/// declining the position — unlike `declared_refinement`'s own general
/// table, which deliberately does NOT read base sorts for an ordinary
/// (non-generator) return annotation, to avoid turning every unrelated
/// `-> int` helper into a new blocker.
pub(super) fn generator_refinement(
    head: &str,
    slice: &Expr,
    aliases: &HashMap<String, AliasEntry>,
    imports: &SurfaceImports,
    environment: &Environment,
) -> Option<GeneratorRefinement> {
    let read_position = |argument: &Expr| -> Option<DeclaredRefinement> {
        declared_refinement(argument, aliases, imports, environment).or_else(|| base_sort_return_refinement(argument))
    };
    match head {
        "Generator" => {
            let Expr::Tuple(members) = slice else {
                return None;
            };
            let [yield_type, _send_type, return_type] = members.elts.as_slice() else {
                return None;
            };
            let yield_type = read_position(yield_type)?;
            let return_type = read_position(return_type);
            Some(GeneratorRefinement { yield_type, return_type })
        }
        "AsyncGenerator" => {
            let Expr::Tuple(members) = slice else {
                return None;
            };
            let [yield_type, _send_type] = members.elts.as_slice() else {
                return None;
            };
            let yield_type = read_position(yield_type)?;
            Some(GeneratorRefinement { yield_type, return_type: None })
        }
        "Iterator" | "Iterable" => {
            let yield_type = read_position(slice)?;
            Some(GeneratorRefinement { yield_type, return_type: None })
        }
        _ => None,
    }
}
