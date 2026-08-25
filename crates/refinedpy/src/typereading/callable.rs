//! A CALLABLE-VARIABLE's own RETURN refinement: `Callable[[...], R]`.

use std::collections::HashMap;

use ruff_python_ast::Expr;

use crate::env::Environment;
use crate::surface::AliasEntry;
use crate::surface::SurfaceImports;

use super::base_sort::base_sort_return_refinement;
use super::declared_refinement::declared_refinement;
use super::declared_refinement::DeclaredRefinement;

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
    aliases: &HashMap<String, AliasEntry>,
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
