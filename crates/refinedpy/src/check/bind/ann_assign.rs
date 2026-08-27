use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::StmtAnnAssign;
use ruff_text_size::Ranged;

use crate::assignability::judge;
use crate::assignability::Verdict;
use crate::diagnostic_sentences::empty_set;
use crate::diagnostic_sentences::unhonorable_annotation;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::surface::AliasEntry;
use crate::typereading::base_sort_return_refinement;
use crate::typereading::callable_return_refinement;
use crate::typereading::declared_refinement;
use crate::typereading::typed_dict_return_refinement;
use crate::typereading::DeclaredRefinement;

use super::super::Finding;
use super::super::WalkContext;
use super::bind_walrus_targets;
use super::judge_and_bind_naming;
use super::name_unmodeled_call_sentence;
use super::record_blocker;
use super::sink_value;
use super::forget_target_names;

/// `x: Annotation = value` — the judging channel. Reads the annotation
/// through `declared_refinement` first (the general table), then through
/// `typed_dict_return_refinement` (a bare Name naming a module-level
/// TypedDict class); when both state nothing, the direct alias-Name path
/// still runs so existing fires do not regress. An annotation whose Name is an alias but is
/// locally rebound in this body states nothing — that is a blocker
/// candidate naming the rebinding, never a judged 7001. A successfully
/// read declaration is also recorded into `aug_assign_refinements`
/// (keyed on the target's plain name) so a later `x op= v` or plain
/// `x = v` in this same body can judge against it too — recorded even
/// for a VALUE-LESS declaration (`a: Age` alone): simple_stmts.rst,
/// "Annotated assignment statements" states annotated assignment as
/// "the combination, in a single statement, of a variable or attribute
/// annotation AND AN OPTIONAL assignment statement" — the `=` clause is
/// its own separate, optional part of the grammar
/// (`annotated_assignment_stmt: augtarget ":" expression ["=" ...]`),
/// so `a: Age` alone declares the slot's refinement without binding the
/// name to anything, and the slot's declared refinement still exists
/// for later reads/writes to judge against even though nothing binds
/// yet.
///
/// A `Callable[[...], R]`-annotated target (`declared_refinement`
/// states nothing for it) is recorded separately, into `environment`'s
/// own `callable_returns` table, through `typereading::
/// callable_return_refinement` — see the CALLABLE-VARIABLE CALL
/// CHANNEL comment inside this function's decline arm.
pub(in crate::check) fn walk_ann_assign(
    assign: &StmtAnnAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    provably_unbound: &mut HashSet<String>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    let declared =
        declared_refinement(assign.annotation.as_ref(), context.aliases, context.imports, environment)
            // A bare Name naming a module-level TypedDict class states
            // its own per-member table, which `declared_refinement`'s
            // general table does not read (a class name is not an alias).
            // The SAME `.or_else` step `check::function_def` already
            // takes for a `-> X` return annotation, so an annotated
            // BINDING declared with a TypedDict judges its dict literal
            // member-by-member exactly as a returned one does.
            .or_else(|| typed_dict_return_refinement(assign.annotation.as_ref(), &context.typed_dicts))
            .or_else(|| direct_alias_annotation(assign.annotation.as_ref(), context.aliases, environment))
            .or_else(|| optional_base_sort_annotation(assign.annotation.as_ref()));

    let Some(declared) = declared else {
        // CALLABLE-VARIABLE CALL CHANNEL: `x: Callable[[...], R] [|
        // None] = ...` states nothing `declared_refinement` reads (a
        // `Callable[...]` subscript is not a set X itself binds to),
        // but a LATER `x(...)` call site still has a fact to judge
        // against — `R`, the callable's own return refinement. Recorded
        // into this environment's `callable_returns` table (keyed on
        // the target's plain name), read back by `check.rs::sink_value`
        // at the call site the same way `aug_assign_refinements` is
        // read back for a later `x op= v`. Tried before the rebound-alias
        // blocker check below: a `Callable`-typed name is ordinary
        // Python with a real fact to state, never this body's blocker.
        if let Expr::Name(target_name) = assign.target.as_ref()
            && let Some(callable_declared) =
                callable_return_refinement(assign.annotation.as_ref(), context.aliases, context.imports, environment)
        {
            let mut callable_returns = environment
                .callable_returns()
                .map(|table| (**table).clone())
                .unwrap_or_default();
            callable_returns.insert(target_name.id.as_str().to_owned(), callable_declared);
            environment.set_callable_returns(Arc::new(callable_returns));
            provably_unbound.remove(target_name.id.as_str());
            bind_target_from_value_expr(assign.target.as_ref(), assign.value.as_deref(), environment, context.kernel);
            return;
        }
        // An alias name shadowed by a local rebinding is the specific,
        // nameable reason nothing was read here; anything else falls
        // through as a plain "annotation not read" case and is not
        // this body's business to block on (a target lacking a
        // refinement-carrying annotation is ordinary Python).
        if let Expr::Name(annotation_name) = assign.annotation.as_ref()
            && context.aliases.contains_key(annotation_name.id.as_str())
            && !environment.alias_is_visible(annotation_name.id.as_str())
        {
            record_blocker(
                blocked,
                assign.annotation.range(),
                format!(
                    "the annotation's name '{}' is rebound in this body",
                    annotation_name.id.as_str()
                ),
                out,
            );
        } else if let Some(spelling) = unhonorable_annotated_spelling(assign.annotation.as_ref(), context.imports) {
            // UNHONORABLE STATEMENT (RTS7004): the annotation is
            // recognizably this table's OWN vocabulary — an
            // `Annotated[...]` subscript rooted at the module's
            // imported `Annotated` identity (`surface::
            // annotated_expression_set`'s own head-identity gate) —
            // but `declared_refinement` still read nothing from it,
            // meaning something PAST that gate refused (an
            // unrecognized base sort, an unrecognized metadata
            // kwarg/constructor, or a pattern the grammar compiler
            // rejected). Mirrors the Go twin's own gate
            // (`RootsInSurface` in `annotation_file_facts.go`): a
            // recognized-root annotation the checker cannot read is
            // never dropped silently, unlike a plain-Python name this
            // table has no vocabulary for at all (that case falls
            // through below, unchanged).
            out.push(Finding {
                range: assign.annotation.range(),
                code: "RTS7004",
                message: unhonorable_annotation(&spelling),
            });
        }
        // PROVABLY-UNBOUND READS: `x: int` (valueless, and no declared
        // refinement this table reads) leaves `x` locally bound
        // (locally_bound_names' own scan) but with no environment
        // binding — the exact CPython UnboundLocalError shape
        // (executionmodel.rst's local-variable rule: a name assigned
        // anywhere in a function is local to the WHOLE function, and
        // reading it before any binding executes raises). A value-
        // carrying `x: int = v` cures it the same way an ordinary
        // assignment would.
        if let Expr::Name(target_name) = assign.target.as_ref() {
            if assign.value.is_none() {
                provably_unbound.insert(target_name.id.as_str().to_owned());
            } else {
                provably_unbound.remove(target_name.id.as_str());
            }
        }
        bind_target_from_value_expr(assign.target.as_ref(), assign.value.as_deref(), environment, context.kernel);
        return;
    };

    // THE EMPTY SET (RTS7003): this annotation compiled to a scalar or
    // sequence set the kernel proves admits nothing — the declaration
    // itself can never be honored by any value, independent of what
    // the write assigns. Mirrors the Go twin's own per-occurrence fire
    // (`annotation_file_facts.go`'s `emptiness` ask, immediately after
    // a successful compile): asked once per annotated statement, the
    // same "courtesy, never a blocker" posture the Go doc names — a
    // kernel refusal on this set's shape (caught, never a crash, the
    // same `catch_unwind` idiom `truthiness_conformance.rs`'s own
    // emptiness ask already holds every kernel closure to) answers
    // "not decided" and this diagnostic simply does not fire, the
    // declaration still compiles and judges normally either way.
    if let Some(true) = declared_set_is_empty(&declared.set, context.kernel) {
        out.push(Finding {
            range: assign.annotation.range(),
            code: "RTS7003",
            message: empty_set(&declared.set),
        });
    }

    if let Expr::Name(target_name) = assign.target.as_ref() {
        aug_assign_refinements.insert(target_name.id.as_str().to_owned(), declared.clone());
    }

    let Some(value_expr) = assign.value.as_deref() else {
        // `a: Age` alone — the declaration is recorded above; CPython
        // evaluates the annotation but does not bind the name, so
        // nothing judges and nothing binds here. Tracked the same way
        // the declined-annotation branch above tracks a valueless `x:
        // int` — the DECLARED-set path and the general path share the
        // one CPython fact (annotation-only never binds).
        if let Expr::Name(target_name) = assign.target.as_ref() {
            provably_unbound.insert(target_name.id.as_str().to_owned());
        }
        bind_target_from_value_expr(assign.target.as_ref(), None, environment, context.kernel);
        return;
    };

    if let Expr::Name(target_name) = assign.target.as_ref() {
        provably_unbound.remove(target_name.id.as_str());
    }
    bind_walrus_targets(value_expr, context, aug_assign_refinements, environment, out);
    let Some(value) = sink_value(value_expr, context, environment, aug_assign_refinements, out) else {
        // a provable raise already pushed its own RTS7001 at the
        // raising expression — this write never completes on this
        // path, so the target holds nothing: forget it, the same
        // "unproducible" answer Undetermined already forgets to.
        forget_target_names(assign.target.as_ref(), environment);
        return;
    };

    let Expr::Name(target_name) = assign.target.as_ref() else {
        // A non-name AnnAssign target (rare in practice — e.g. an
        // attribute/subscript annotated write): judge for the Fire, but
        // there is no environment slot to rebind under the refused-write
        // law, so fall back to the old bind-the-RHS path.
        match judge(&value, &declared, context.kernel) {
            Verdict::Fire(message) => out.push(Finding {
                range: value_expr.range(),
                code: "RTS7001",
                message,
            }),
            Verdict::Silent => {}
            Verdict::Undetermined(sentence) => {
                let sentence = name_unmodeled_call_sentence(sentence, Some(value_expr), Some(&value), environment);
                record_blocker(blocked, value_expr.range(), sentence, out);
            }
        }
        bind_target_from_value_expr(assign.target.as_ref(), Some(value_expr), environment, context.kernel);
        return;
    };

    if let Some(sentence) = judge_and_bind_naming(
        target_name.id.as_str(),
        value,
        &declared,
        value_expr.range(),
        Some(value_expr),
        context,
        environment,
        out,
    ) {
        record_blocker(blocked, value_expr.range(), sentence, out);
    }
}

/// `Optional[int|float|str]` / `int|float|str | None` — the Optional-
/// peeling idiom over a BARE base sort, with no alias involved.
/// `declared_refinement`'s general table deliberately does not read a
/// bare `int`/`float`/`str` (its own doc: doing so turned every
/// unreadable `-> int` helper into a fresh undetermined blocker), so
/// `over: Optional[int] = 200` reaches this function's caller with
/// `declared` still `None` and NOTHING recorded into
/// `aug_assign_refinements` — leaving `walk_if`'s `is_admits_none_peel_
/// test` unable to find the declared shape and firing the dead-branch
/// law on the ordinary `if over is None:` peel. This reader is scoped
/// to exactly the wrapper shape (`Optional[X]`/`X | None`) around
/// exactly a bare base-sort name, and answers through
/// `base_sort_return_refinement` — the SAME set that sort already
/// states everywhere else it is read (a declined call's return, a
/// `Callable[[...], R]` slot) — so recording it here states nothing
/// new, only lets the ALREADY-STATED fact reach the peel-test
/// exception. A bare `int`/`float`/`str` with no `Optional`/`| None`
/// wrapper still declines (unaffected): this function is reached only
/// through `walk_ann_assign`'s `Optional[X]`/`X | None` peel below.
pub(in crate::check) fn optional_base_sort_annotation(annotation: &Expr) -> Option<DeclaredRefinement> {
    match annotation {
        Expr::Subscript(subscript) => {
            let is_optional = matches!(subscript.value.as_ref(), Expr::Name(head) if head.id.as_str() == "Optional");
            if !is_optional {
                return None;
            }
            let mut declared = base_sort_return_refinement(subscript.slice.as_ref())?;
            declared.admits_none = true;
            Some(declared)
        }
        Expr::BinOp(binop) if binop.op == ruff_python_ast::Operator::BitOr => {
            let left_is_none = matches!(binop.left.as_ref(), Expr::NoneLiteral(_));
            let right_is_none = matches!(binop.right.as_ref(), Expr::NoneLiteral(_));
            if left_is_none == right_is_none {
                return None;
            }
            let other = if right_is_none { binop.left.as_ref() } else { binop.right.as_ref() };
            let mut declared = base_sort_return_refinement(other)?;
            declared.admits_none = true;
            Some(declared)
        }
        _ => None,
    }
}

/// The pre-typereading path: an annotation that is bare `Name` naming
/// a compiled alias, visible in this body (not locally rebound). Kept
/// alongside `declared_refinement` so the two existing tests' fires
/// keep firing before the general annotation table recognizes this
/// same shape itself.
pub(in crate::check) fn direct_alias_annotation(
    annotation: &Expr,
    aliases: &HashMap<String, AliasEntry>,
    environment: &Environment,
) -> Option<DeclaredRefinement> {
    let Expr::Name(name) = annotation else {
        return None;
    };
    if !environment.alias_is_visible(name.id.as_str()) {
        return None;
    }
    let entry = aliases.get(name.id.as_str())?;
    // Same container carry as `declared_refinement`'s own bare-Name arm
    // (`typereading.rs` doc) — a `Boosted`-shaped alias must not lose its
    // element/length window just because this AnnAssign path reads it
    // before the general table gets a chance to. The element's spelling
    // is its own WRITTEN spelling (`entry.element`'s second tuple slot),
    // never a reformatting of its resolved set — the same fidelity
    // `declared_refinement`'s own bare-Name arm keeps.
    let element = entry.element.as_ref().map(|element_entry| {
        let (element_set, element_spelling) = element_entry.as_ref();
        Box::new(DeclaredRefinement {
            set: element_set.clone(),
            spelling: element_spelling.clone(),
            admits_none: false,
            element: None,
            element_length: None,
            generator: None,
            members: None,
            positions: None,
            temporal: None,
            temporal_awareness: crate::surface::TemporalAwareness::Any,
        })
    });
    let container_spelling = match (entry.head, &element) {
        (Some(head), Some(element_declared)) => Some(format!("{}[{}]", head, element_declared.spelling)),
        _ => None,
    };
    Some(DeclaredRefinement {
        set: entry.set.clone(),
        spelling: container_spelling.unwrap_or_else(|| name.id.as_str().to_owned()),
        admits_none: false,
        element,
        element_length: entry.length_window,
        generator: None,
        members: None,
        positions: None,
        temporal: entry.temporal.clone(),
        temporal_awareness: entry.temporal_awareness,
    })
}

/// Recognizes an annotation as this table's OWN `Annotated[...]`
/// vocabulary — the module's imported `Annotated` (or
/// `typing_extensions.Annotated`) identity as the subscript's head —
/// the exact gate `surface::annotated_expression_set` itself opens
/// with before reading any base sort or metadata. Reads `imports`'
/// `annotated_names` set directly rather than re-deriving it, so this
/// stays exactly as narrow as the compiler's own recognition and never
/// drifts from it. Returns the recognized spelling (`"Annotated[...]"`)
/// for the RTS7004 message when the annotation matches this shape;
/// `None` for every other shape (a bare alias name, `dict[...]`,
/// `list[...]`, or any expression this table has no vocabulary for at
/// all) — those are ordinary Python, never this diagnostic's business.
pub(in crate::check) fn unhonorable_annotated_spelling(annotation: &Expr, imports: &crate::surface::SurfaceImports) -> Option<String> {
    let Expr::Subscript(subscript) = annotation else {
        return None;
    };
    let Expr::Name(head) = subscript.value.as_ref() else {
        return None;
    };
    if !imports.annotated_names.contains(head.id.as_str()) {
        return None;
    }
    Some(format!("{}[...]", head.id.as_str()))
}

/// Whether a compiled declared set is proved empty — the courtesy ask
/// RTS7003 fires on. Tries the scalar decider first, then the sequence
/// decider (the same two-decider order `truthiness_conformance.rs`'s
/// own `state_is_uninhabited` takes), each guarded by `catch_unwind` so
/// a kernel refusal on a set shape neither decider speaks to reads as
/// `None` (not decided) rather than a crash. `None` means this
/// diagnostic simply does not fire — the annotation still compiled and
/// still judges normally.
pub(in crate::check) fn declared_set_is_empty(set: &refined_sets::refinement_forms::RefinedSet, kernel: &Arc<RefinedTSKernel>) -> Option<bool> {
    let scalar_asked = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(set));
    if let Ok(empty) = scalar_asked {
        return Some(empty);
    }
    let seq_asked = crate::kernel_ask::ask_kernel(|| (kernel.seq_empty)(set));
    seq_asked.ok()
}

/// After an AnnAssign is judged (or declined), the target still binds:
/// a known value if the RHS was readable, forgotten otherwise so a
/// stale fact never survives an unread write.
pub(in crate::check) fn bind_target_from_value_expr(
    target: &Expr,
    value_expr: Option<&Expr>,
    environment: &mut Environment,
    kernel: &Arc<RefinedTSKernel>,
) {
    let Expr::Name(name) = target else {
        return;
    };
    match value_expr {
        Some(expr) => {
            let value = evaluate_expression(expr, environment, kernel);
            environment.bind(name.id.as_str(), value);
        }
        None => environment.forget(name.id.as_str()),
    }
}
