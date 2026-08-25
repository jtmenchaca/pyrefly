use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use refined_domain::abstract_value::{
    known_set, possibly_absent, AbsentFlavor, AbstractValue, Kind, ObjectKey, PrimitiveKind, SetKindTag,
};
use refined_domain::known_constructors::{known_dict_star, known_list, known_object};
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::{make_refined_set, on_one_tuple_layer, repeat_of, requires_integer};
use ruff_python_ast::{Expr, Parameters};

use crate::assignability::states_sequence;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances::ClassModel;
use crate::typereading::{callable_return_refinement, declared_refinement, DeclaredRefinement};

use super::*;

/// A function body's own parameters whose annotation reads through
/// `declared_refinement`: bind the name to a set-kind AbstractValue
/// holding the declared set (`known_set`, TrustSpec — the annotation is
/// read, not proved by execution). A parameter whose annotation states
/// nothing this table reads instead tries the CALLER-JOINED seed
/// (`unannotated_parameter_caller_seed`) — this def's own recorded direct
/// callers (`enclosing_def_name`'s lookup in `context.caller_arguments`),
/// each contributing whatever exact string its own bound argument at this
/// SAME position folds to; joining every caller's fold is what lets
/// `c-reference-shapes.py`'s `level_via_parameter_path` carry its
/// caller's literal script path onto `script_path` with no annotation at
/// all. Left unbound (ordinary Python, no seed) when no caller-joined
/// value applies either — unannotated, uncalled, or a caller whose own
/// argument does not fold.
pub(super) fn seed_parameters(
    parameters: &Parameters,
    enclosing_def_name: Option<&str>,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
) {
    // Position, when `Some`, is this parameter's own index among
    // POSITIONAL parameters only (`posonlyargs` then `args`, in the
    // exact order a plain positional call's own `call.arguments.args`
    // fills them) — `CallerArguments`'s own indexing convention. A
    // keyword-only parameter (chained last, always `None` here) can
    // never be filled by a positional call at all, so it carries no
    // position for a caller-joined seed to read.
    let ordered = parameters
        .posonlyargs
        .iter()
        .chain(parameters.args.iter())
        .enumerate()
        .map(|(position, parameter)| (Some(position), parameter))
        .chain(parameters.kwonlyargs.iter().map(|parameter| (None, parameter)));
    for (position, parameter) in ordered {
        let Some(annotation) = parameter.parameter.annotation.as_deref() else {
            if let Some(position) = position {
                if let Some(seed) = unannotated_parameter_caller_seed(enclosing_def_name, position, context) {
                    environment.bind(parameter.parameter.name.id.as_str(), seed);
                }
            }
            continue;
        };
        // A bare CLASS-NAME parameter (`request: AudioRequest`, the class
        // itself declaring annotated fields — self-authored, `@dataclass`,
        // or pydantic `BaseModel` alike) seeds a TAGGED `Kind::Object`
        // instance exactly as `judge_construction` tags one built from a
        // real call: `source` carries the class name so `evaluate_attribute_
        // read` (expressions.rs) finds `model` in `environment.classes()`/
        // `context.classes` and reads through `instances::field_read_
        // through_model`, which — for an ordinary stored field, the only
        // shape a parameter's fields are — resolves to `instances::
        // field_read`'s linear scan of the INSTANCE'S OWN `keys`. The read
        // path never re-derives a field's value from `ClassModel.fields`;
        // it consumes only what this seed puts in `keys`, so every
        // declared field is populated here (per-field, from `ClassField.
        // declared`) rather than left for a later lookup to reconstruct.
        // A name that is not a class in this table falls through to the
        // ordinary `declared_refinement` read below, unchanged.
        if let Expr::Name(class_name) = annotation {
            if let Some(model) = context.classes.get(class_name.id.as_str()) {
                let instance = class_parameter_object(model);
                environment.bind(parameter.parameter.name.id.as_str(), instance);
                continue;
            }
        }
        // A `Callable[[...], R]`-ANNOTATED PARAMETER (`declared_refinement`
        // states nothing for it — a `Callable[...]` subscript is not a set
        // the parameter itself binds to) states a fact a LATER `f(...)`
        // call site still needs: `R`'s own declared return refinement.
        // Recorded into this environment's `callable_returns` table, keyed
        // on the parameter's own name — the SAME table `walk_ann_assign`'s
        // own CALLABLE-VARIABLE CALL CHANNEL grows for a body-local
        // `x: Callable[[...], R] = ...`, read back by the identical
        // `sink_value`/`evaluate_call` call-site channels with no new
        // dispatch needed. Bound to a plain `FUNCTION_VALUE_WORD`-tagged
        // opaque value (`same_module_def_gate_open`'s own `kind_word`
        // check reads this as "a function value" everywhere that matters —
        // truthy, `is`-comparable, never a scalar/collection value with a
        // fire message of its own) rather than left unbound: a parameter
        // must hold SOME value for `f is known`/`if f:` guards to read at
        // all, and this file has no way to know WHICH function a caller
        // actually passes, so the value carries no specific def identity
        // (unlike `env::same_module_def_alias_value`, which names one).
        if let Some(callable_declared) =
            callable_return_refinement(annotation, context.aliases, context.imports, environment)
        {
            let mut callable_returns = environment
                .callable_returns()
                .map(|table| (**table).clone())
                .unwrap_or_default();
            callable_returns.insert(parameter.parameter.name.id.as_str().to_owned(), callable_declared);
            environment.set_callable_returns(Arc::new(callable_returns));
            environment.bind(
                parameter.parameter.name.id.as_str(),
                refined_domain::abstract_value::opaque_value(crate::env::FUNCTION_VALUE_WORD),
            );
            continue;
        }
        // A bare `int`/`float`/`str` PARAMETER seeds its sort claim (the
        // whole-int ray etc. — typereading's own base-sort reader), so
        // `age: int` flowing into a refined sink refuses by containment
        // ("a whole int admits values outside the set") unless a guard
        // narrows it. Scoped to parameters ONLY: the general annotation
        // table does not read base sorts, so `-> int` returns stay
        // unjudged and helper bodies gain no new blockers.
        let Some(declared) =
            declared_refinement(annotation, context.aliases, context.imports, environment)
                .or_else(|| crate::typereading::base_sort_return_refinement(annotation))
        else {
            continue;
        };
        // WRITE-SITE CHECK ELIGIBILITY: a parameter's own declared
        // refinement is recorded into `aug_assign_refinements` — the SAME
        // table `walk_ann_assign` inserts a body-local `x: Age = ...`
        // target's own `declared` into (that function's own doc, right
        // after its identical `declared_refinement` read succeeds, before
        // any shape-specific branching) — so a LATER `x += 1`/`x = 200`
        // against this PARAMETER judges against its declared set at the
        // write, not only at whichever sink the value eventually flows
        // to. Without this, `walk_name_aug_assign`'s own `aug_assign_
        // refinements.get(name)` lookup finds nothing for a parameter (the
        // table was, until this insert, populated by AnnAssign targets
        // only), so `x += 1` on a declared-refinement parameter bound
        // silently instead of being judged at the `+=` itself
        // (E2.operator.py's own `compound_assign_outside_set`).
        aug_assign_refinements.insert(parameter.parameter.name.id.as_str().to_owned(), declared.clone());
        // A `list[X]`/`set[X]`/`Sequence[X]` PARAMETER (`declared.element`
        // Some, `declared.set` unused/empty — typereading's own "one
        // active field" convention) seeds a SEQUENCE whose every position
        // draws from X's own set: `Kind::Set` over one `Form::Repeat(X,
        // lo, hi)` refinement — the bare unbounded window (`lo` 0, `hi`
        // `None`, the same shape `star` alone used to build) when the
        // declaration states no length bound, or a TIGHTER window when
        // `declared.element_length` carries one (`Annotated[list[X],
        // Field(min_length=…, max_length=…)]`, `typereading.rs`'s own
        // doc — `repeat_of`/`repetition` already accept an arbitrary
        // `lo`, so this is the SAME constructor family, not a new one).
        // Either way, `subscript_read`'s own `Kind::Set` arm reads any
        // index as "some member of X", the repetition grammar's own
        // definition (a repetition's positions never hold anything
        // outside its element alphabet), so no kernel round trip is ever
        // needed to answer an element read; `relational_sum.rs`'s own
        // `element_and_count_sets` reads the SAME window's `lo`/`hi` into
        // the count set a relational division ties to, so a `min_length`
        // bound here is what gives that division its `count >= 1` fact
        // with no cast. Scoped to a SCALAR/string-sorted element
        // (`element.set` non-empty) — an element that is itself
        // container-shaped (`dict[str, X]`'s own element, another
        // `list[X]`) has no scalar set here to repeat over, and is left
        // for the plain element-carrying seed below, matching today's
        // behavior for that shape.
        let is_sequence_container = declared.spelling.starts_with("list[")
            || declared.spelling.starts_with("set[")
            || declared.spelling.starts_with("Sequence[");
        if is_sequence_container {
            if let Some(element) = &declared.element {
                if !element.set.forms.is_empty() {
                    let (lo, hi) = declared.element_length.unwrap_or((0, None));
                    // The element's own sort (`requires_integer` reads the
                    // `Form::Integer` marker `annotated_expression_set`
                    // pushes for a bare `int`, the same recognizer
                    // `adapter_alias_verdict` above already uses to tell an
                    // int-sorted alias from a float-sorted one) rides onto
                    // the OUTER sequence value's `kind_tag` — `star_numeric_
                    // hull`/`sum_call_over_star`/`min_max_over_star`
                    // (builtin_models.rs) all read the sequence value's own
                    // tag, never the element's, to answer sum/min/max over
                    // an unknown-length iterable.
                    let sort = if requires_integer(&element.set) {
                        PrimitiveKind::Integer
                    } else {
                        PrimitiveKind::Float
                    };
                    let sequence = AbstractValue {
                        kind_tag: Some(sort),
                        ..known_set(
                            make_refined_set(vec![repeat_of(element.set.clone(), lo, hi)]),
                            None,
                            TrustSpec,
                            SetKindTag::None,
                        )
                    };
                    environment.bind(parameter.parameter.name.id.as_str(), sequence);
                    continue;
                }
            }
        }
        // A `dict[str, X]` PARAMETER (`declared.element` Some, spelling
        // `"dict[str, …]"` — `typereading.rs`'s own dict arm) seeds an
        // unbounded-key object: `dict_star_value_seed` (below) reads X —
        // scalar, or itself another `dict[str, Y]` — and wraps it as the
        // claim every STRING key, if present, reads back as, the dict
        // twin of the sequence-star seed just above, but keyed by string
        // identity rather than position.
        // `collection_models.rs::dict_get_result`/`subscript_read` read
        // an `ObjectStar` receiver by unwrapping the element back off
        // this same shape.
        let is_dict_container = declared.spelling.starts_with("dict[str, ");
        if is_dict_container {
            if let Some(element) = &declared.element {
                if let Some(star) = dict_star_value_seed(element) {
                    environment.bind(parameter.parameter.name.id.as_str(), star);
                    continue;
                }
            }
        }
        // A FIXED-ARITY `tuple[X, Y]` PARAMETER (`declared.positions`
        // Some — typereading's own per-position table) seeds a
        // KNOWN-LENGTH `Kind::List` whose slot `i` is `known_set` over
        // position `i`'s own declared set — the same nested-exact-
        // sequence shape a literal tuple/list display already builds
        // (`collection_models.rs`'s own module doc). Unlike the sequence
        // seed above, every position keeps its OWN set rather than
        // sharing one starred element, so `subscript_read`'s `Kind::List`
        // arm reads slot `i` back exactly, and a spread (`[*item]`) can
        // splice the slots in place — a spread only recognizes a known
        // `Kind::List` receiver (`expressions.rs::evaluate_display_elements`),
        // never the unknown-length `Kind::Set` star shape above.
        if let Some(positions) = &declared.positions {
            let items = positions
                .iter()
                .map(|position| known_set(position.set.clone(), None, TrustSpec, SetKindTag::None))
                .collect();
            let tuple = known_list(items, TrustSpec);
            environment.bind(parameter.parameter.name.id.as_str(), tuple);
            continue;
        }
        // A temporal declaration (`declared.temporal` Some, `declared.set`
        // unused/empty — the same "one active field" convention every
        // other container shape here keeps) seeds a WINDOW value, tagged
        // `source = "temporal_flow"` — `assignability.rs`'s own temporal
        // law reads this tag to route through `bounds_imply` (window vs
        // window: does THIS parameter's own declared bound sit inside a
        // narrower call target's bound) rather than `bounds_verdict_of`
        // (point vs window, the shape a CONSTRUCTION's own exact value
        // takes). Carries the SAME `TemporalAnnotation` the declaration
        // itself states — `record_visit`'s own `narrow(p)` row is exactly
        // this: `p`'s seeded value is Period's own [2020, 2024] window,
        // checked against `narrow`'s own Visit-declared [2021, 2021]
        // parameter.
        if let Some(declared_temporal) = &declared.temporal {
            let mut instance = known_object(Vec::new(), None, true, TrustSpec, false);
            instance.source = "temporal_flow".to_owned();
            instance.temporal = Some(Box::new(declared_temporal.clone()));
            environment.bind(parameter.parameter.name.id.as_str(), instance);
            continue;
        }
        // A scalar declared set carries its own numeric sort onto the
        // seeded value's `kind_tag` — the same tag `min_max_scalar_operand`/
        // `star_numeric_hull`/`sum_call_over_star` (builtin_models.rs) read
        // to answer sum/min/max — but ONLY when the declared set is
        // numeric-ground: `on_one_tuple_layer` alone also reads a
        // `Literal["A", "B", "C"]` string-tuple union as "on the one-tuple
        // layer" (the tuple pun `assignability.rs::states_sequence`'s own
        // doc names), so the gate pairs both checks exactly as every sort
        // law in that file does. A string/sequence-shaped declared set
        // (`states_sequence` true, or `on_one_tuple_layer` false) is left
        // untagged, unchanged from today.
        let admits_none = declared.admits_none;
        let seeded = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
            let sort = if requires_integer(&declared.set) {
                PrimitiveKind::Integer
            } else {
                PrimitiveKind::Float
            };
            AbstractValue {
                kind_tag: Some(sort),
                ..known_set(declared.set, None, TrustSpec, SetKindTag::None)
            }
        } else {
            known_set(declared.set, None, TrustSpec, SetKindTag::None)
        };
        // `Optional[X]`/`X | None` (`declared.admits_none`): the parameter
        // genuinely may arrive as `None` at runtime, so the seeded value
        // must carry that admission the same way a JS `undefined`-or-`X`
        // parameter would — wrapped in the maybe carrier
        // (`possibly_absent`), its absent side pinned `NullOnly` (Python's
        // `None`, never JS's `undefined`). Un-wrapped, a bare `Kind::Set`
        // seed has NO shape a narrowing test could ever read as "possibly
        // absent" (`narrowing.rs`'s own `is`/`is not None` arms only ever
        // touch `Kind::Values`/`Kind::PossiblyUndefined`), so `sample is
        // not None` against the un-wrapped seed decided PROVABLY TRUE
        // outright (`expressions.rs::compare_pair`'s `is`/`is not` law
        // settles identity the moment either side is KNOWN and not
        // `Kind::Null` — a bare `Kind::Set` already qualifies) rather than
        // staying the undecided test an Optional parameter's own guard
        // must be.
        let seeded = if admits_none {
            possibly_absent(seeded, AbsentFlavor::NullOnly, Some(TrustSpec), false)
        } else {
            seeded
        };
        environment.bind(parameter.parameter.name.id.as_str(), seeded);
    }
}

/// The unbounded-key dict-star value a `dict[str, X]` PARAMETER's own
/// VALUE SLOT declaration (`element`, a `DeclaredRefinement`) seeds —
/// `seed_parameters`' own dict arm calls this, and it recurses for a
/// NESTED `dict[str, Y]` value slot (`dict[str, dict[str, int]]`, or
/// (via `Optional`) `dict[str, Optional[dict[str, int]]]` —
/// c-reads-and-values.py's own `read_optional_chain_deeper_step` row),
/// since `element` itself carries the identical `DeclaredRefinement`
/// shape a top-level `dict[str, X]` parameter's own `declared` does.
///
/// Two value shapes:
/// - SCALAR (`element.set` non-empty — a plain/refined `int`/`str`/
///   alias value, never another container): `known_dict_star` wraps the
///   value's own set directly, tagged with its numeric sort when the
///   set is numeric-ground (the same `requires_integer` gate the
///   sequence-star seed above applies).
/// - NESTED DICT (`element.spelling` itself starts `"dict[str, "` —
///   `element.element` is Some, `element.set` empty, the same "one
///   active field" convention every container declaration keeps): this
///   function recurses on `element.element` to build the INNER
///   dict-star first, then wraps THAT as the outer star's own element —
///   `dict[str, dict[str, int]]`'s outer star holds an inner star at
///   every string key, exactly as a `list[list[int]]` parameter would
///   nest two star levels if this crate modeled that shape.
///
/// Either way, `element.admits_none` (`dict[str, Optional[X]]`) wraps
/// the starred element in the maybe carrier first, `NullOnly` —
/// Python's `None`, the same admission `seed_parameters`' own
/// scalar-parameter tail wraps with — so a present key's own value may
/// itself be `None`, matching `dict[str, Optional[dict[str, int]]]`'s
/// own declared admission. `None` when `element` is neither shape this
/// function reads (an element that is itself a `list[X]`/tuple/
/// TypedDict/generator declaration, none of which nest inside a dict
/// value slot today) — the caller's own job to leave the parameter
/// unseeded in that case, never to guess.
pub(super) fn dict_star_value_seed(element: &DeclaredRefinement) -> Option<AbstractValue> {
    let value_seed = if !element.set.forms.is_empty() {
        let sort = if requires_integer(&element.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        AbstractValue {
            kind_tag: Some(sort),
            ..known_set(element.set.clone(), None, TrustSpec, SetKindTag::None)
        }
    } else if element.spelling.starts_with("dict[str, ") {
        let nested = element.element.as_deref()?;
        dict_star_value_seed(nested)?
    } else {
        return None;
    };
    let value_seed = if element.admits_none {
        possibly_absent(value_seed, AbsentFlavor::NullOnly, Some(TrustSpec), false)
    } else {
        value_seed
    };
    let (star, ok) = known_dict_star(value_seed, TrustSpec);
    if ok {
        Some(star)
    } else {
        None
    }
}

/// The caller-joined seed for one UNANNOTATED parameter at `position`
/// (among positional parameters only) in the def named `enclosing_def_
/// name`: `None` when there is no def name (a nested/method body, or the
/// module body itself), the def does not qualify at all (missing from
/// `context.caller_arguments` — some occurrence of its name was not a
/// plain positional call), it has zero recorded callers, some caller's
/// own call is shorter than `position` (an argument this position was
/// never even passed at that site), some caller's own argument does not
/// fold to an exact string (`caller_argument_exact_string`), or two
/// callers fold to DIFFERENT exact strings (a real disagreement — no
/// single literal is true of every call, so this scalar seed cannot
/// state one without inventing a value no caller actually passed).
/// `Some` only when EVERY recorded caller's argument at this position
/// folds to the identical exact string, the one case where "this
/// parameter always holds this text" is something the checker actually
/// proved from the call sites it can see.
pub(super) fn unannotated_parameter_caller_seed(
    enclosing_def_name: Option<&str>,
    position: usize,
    context: &WalkContext,
) -> Option<AbstractValue> {
    let def_name = enclosing_def_name?;
    let calls = context.caller_arguments.get(def_name)?;
    if calls.is_empty() {
        return None;
    }
    let module_environment = module_scope_environment(context);
    let mut agreed_text: Option<String> = None;
    for call_arguments in calls {
        let argument = call_arguments.get(position)?;
        let text = caller_argument_exact_string(argument, &module_environment, context.kernel)?;
        match &agreed_text {
            None => agreed_text = Some(text),
            Some(existing) if *existing == text => {}
            Some(_) => return None,
        }
    }
    Some(crate::string_models::string_literal_value(&agreed_text?))
}

/// A fresh, empty-locally-bound `Environment` seeded with every MODULE-
/// LEVEL binding this walk already exposes to every ordinary body
/// (`context.module_bindings` — the exact table `walk_body_with_self_
/// binding` layers onto each body's own environment). A caller's argument
/// expression is read in THIS scope, never the calling body's own live
/// one: the escape discipline this whole join rests on only trusts a
/// module-level constant or a written literal, precisely because a
/// caller's own LOCAL variable is not something a def-name-keyed,
/// built-once-per-module table could ever resolve safely.
pub(super) fn module_scope_environment(context: &WalkContext) -> Environment {
    let mut environment = Environment::new(HashSet::new());
    for (name, value) in &context.module_bindings {
        environment.bind(name, value.clone());
    }
    environment
}

/// The same three const-fold tiers `foreign_edge.rs::const_folded_text_of`
/// reads (a written string literal; a bare `Name` resolving, through
/// `environment`, to a known exact string; or any other expression
/// `evaluate_expression` folds to an exact string — an f-string composed
/// entirely of consts, a `+` concatenation of known exact strings),
/// reimplemented here per this crate's own no-shared-private-helper
/// convention (`foreign_edge.rs::exact_string_text`'s own doc states the
/// identical precedent against `string_models.rs`). `None` on any tier
/// that does not fold — a caller's argument that is itself a local
/// variable, a call, or any other non-exact expression.
pub(super) fn caller_argument_exact_string(argument: &Expr, environment: &Environment, kernel: &Arc<RefinedTSKernel>) -> Option<String> {
    if let Expr::StringLiteral(literal) = argument {
        return Some(literal.value.to_str().to_owned());
    }
    if let Expr::Name(name) = argument {
        if let Some(bound) = environment.read(name.id.as_str()) {
            if let Some(text) = caller_exact_string_text(bound) {
                return Some(text);
            }
        }
    }
    let folded = evaluate_expression(argument, environment, kernel);
    caller_exact_string_text(&folded)
}

/// The exact text an `AbstractValue` carries, if it is a `Kind::Values`
/// state sorted `PrimitiveKind::String` — `string_models.rs::
/// exact_string_text`'s own exact twin, reimplemented locally per this
/// crate's convention (see `caller_argument_exact_string`'s own doc).
pub(super) fn caller_exact_string_text(value: &AbstractValue) -> Option<String> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    Some(value.values.iter().filter_map(|c| char::from_u32(*c as i64 as u32)).collect())
}

/// A CLASS-NAME parameter's own instance value: a `Kind::Object` tagged
/// with `source = model.name` (the SAME tag `judge_construction` writes
/// for a real call — the tag `evaluate_attribute_read` and every other
/// `receiver.source`-keyed reader already consult), one `ObjectKey` per
/// declared field IN `model.fields`' own order. A field with no declared
/// refinement (`ClassField.declared` `None`) seeds NOTHING for that key —
/// absent from `keys` entirely, not a fabricated unrefined set — so a
/// later read of it stays undetermined naming the true blocker (there is
/// no declaration to read), rather than reading as a false "any value at
/// all" claim this seed did not earn.
pub(super) fn class_parameter_object(model: &ClassModel) -> AbstractValue {
    let entries: Vec<ObjectKey> = model
        .fields
        .iter()
        .filter_map(|field| {
            let declared = field.declared.as_ref()?;
            Some(ObjectKey {
                name: field.name.clone(),
                numeric: false,
                value: class_field_value(declared),
            })
        })
        .collect();
    let mut instance = known_object(entries, None, true, TrustSpec, false);
    instance.source = model.name.clone();
    instance
}

/// One declared field's own seeded value — the same two shapes
/// `seed_parameters` already builds for a top-level parameter of the
/// same declared shape, so a field reads back exactly like a parameter
/// would: a sequence-shaped field (`declared.element` Some, a scalar-
/// sorted element — the same gate `seed_parameters`' sequence-container
/// branch reads) becomes a repetition set carrying the element's own
/// numeric sort on `kind_tag`, so `samples: Annotated[list[float],
/// Field(min_length=1)]` flows into sum/min/max/relational reads
/// exactly as a top-level `samples: list[float]` parameter would. A
/// scalar field with a numeric-ground declared set carries that sort on
/// `kind_tag` the same way a scalar parameter does. `set_kind_tag` stays
/// `SetKindTag::None` in both shapes — that field wears the bigint/symbol
/// distinction, unrelated to `kind_tag`'s numeric sort, and this seed
/// states nothing about it either way.
pub(super) fn class_field_value(declared: &DeclaredRefinement) -> AbstractValue {
    if let Some(element) = &declared.element {
        if !element.set.forms.is_empty() {
            let (lo, hi) = declared.element_length.unwrap_or((0, None));
            let sort = if requires_integer(&element.set) {
                PrimitiveKind::Integer
            } else {
                PrimitiveKind::Float
            };
            return AbstractValue {
                kind_tag: Some(sort),
                ..known_set(
                    make_refined_set(vec![repeat_of(element.set.clone(), lo, hi)]),
                    None,
                    TrustSpec,
                    SetKindTag::None,
                )
            };
        }
    }
    if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
        let sort = if requires_integer(&declared.set) {
            PrimitiveKind::Integer
        } else {
            PrimitiveKind::Float
        };
        return AbstractValue {
            kind_tag: Some(sort),
            ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
        };
    }
    known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
}
