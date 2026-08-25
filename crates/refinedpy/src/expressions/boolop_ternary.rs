
use refined_domain::abstract_value::Kind;

use crate::env::Environment;


/// A function/method call — dispatch order: (a) a bare name that is
/// EITHER environment-unbound OR bound only to an opaque lambda value
/// (`same_module_def_gate_open`), naming a SAME-MODULE `def`
/// (`environment.functions()`), summarizes through `summaries::call_result`
/// — checked FIRST, so a module-level `def` shadows a builtin of the
/// same name, matching CPython's own name resolution (a later `def
/// len(...):` at module scope really does shadow the builtin `len`);
/// (b) a bare, unbound name naming a same-module class
/// (`environment.classes()`) is a
/// CONSTRUCTION call — judged through `instances::judge_construction`,
/// but this is a VALUE read: any fire the construction raises is
/// check.rs's own statement-sink business (a nested construction inside
/// a larger expression has no sink of its own here), so the verdict's
/// `fires` are discarded and only `instance` is returned; (c) a bare
/// unbound name calls a builtin (`len` gets its own row into
/// `collection_models::len_result`; everything else goes to
/// `builtin_models::builtin_call_result`); (d) `math.<name>(...)` where
/// `math` is not locally bound calls `math_models::math_call_result`;
/// (e) any other attribute call evaluates its receiver and dispatches by
/// the receiver's own known shape (an exact string's method, or a
/// dict's `.get`); (f) everything else — a lambda call, a bound-name
/// call, an unmodeled builtin, an unmodeled method — is unknown().
/// Keyword arguments are not modeled for any row EXCEPT the
/// function/construction paths, which map keywords to parameter/field
/// position themselves — every other cited builtin/method signature
/// this wave models takes positional arguments only, so the keyword
/// guard below applies to the builtin/math/method paths. A STARRED
/// positional argument (`max(*xs)`) splices in place when it evaluates
/// to a known `Kind::List` (`splice_call_arguments`'s own doc) — an
/// unknown or unbounded starred argument still declines the whole call,
/// since this file cannot guess how many positional slots it fills.
/// The same-module-`def` gate is `same_module_def_gate_open`, not a bare
/// `environment.read(name).is_none()` check — see that function's own
/// doc for why a name bound to an opaque LAMBDA value still needs to
/// reach the function table.
pub(super) fn same_module_def_gate_open(environment: &Environment, name: &str) -> bool {
    match environment.read(name) {
        None => true,
        // `f = lambda: ...` binds `f` to `opaque_value("a function
        // value")` (this file's own `Expr::Lambda` arm) — an ordinary
        // program-tracked value binding still blocks the same-module-def
        // dispatch (a real value shadows the def name), but a LAMBDA
        // binding carries no scalar/collection value of its own to
        // shadow anything with, so the gate stays open and `f()` still
        // reaches a same-module `def f(...)` if the module happens to
        // declare one of that name (an unusual but legal shadow: Python
        // itself would call whichever binding is live at the call site,
        // and this file tracks no execution-order distinction between
        // the lambda assignment and a module-level `def` of the same
        // name — the function-table dispatch is the more informative
        // answer of the two shapes this file can read).
        // A CLASS-OBJECT binding likewise keeps the gate open: the walk
        // seeds a class's own bare name to its class-object value (a
        // Kind::Object whose `source` is the class's own name —
        // `instances::class_object_value`), and CALLING that binding IS
        // the construction the classes arm below decides. Any other
        // binding shadows the def/class dispatch as before.
        Some(value) => {
            value.kind == Kind::Object
                && (value.kind_word == Some("a function value") || value.source == name)
        }
    }
}
