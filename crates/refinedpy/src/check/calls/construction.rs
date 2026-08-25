//! Construction-call recognition and judging: a bare-Name class call, or
//! pydantic's own `model_validate`/`model_validate_json`/
//! `TypeAdapter(...).validate_python` parse surface — plus the shared
//! argument-evaluation helpers every call site in this module uses.

use std::sync::Arc;

use refined_domain::abstract_value::{known_set, known_values, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::{TrustProved, TrustSpec};
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::{on_one_tuple_layer, requires_integer};
use ruff_python_ast::Expr;
use ruff_text_size::{Ranged, TextRange};

use crate::assignability::{judge, states_sequence, Verdict};
use crate::check::WalkContext;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::instances::{judge_construction, ClassModel, ConstructionVerdict};
use crate::typereading::DeclaredRefinement;

/// Recognizes `expr` as a class-construction call and judges it, or
/// `None` when `expr` is not one of the two recognized construction
/// shapes:
///
/// (a) a bare-Name call (`Person(40)`, `Person(age=40)`) whose callee
///     is UNBOUND in the environment (a bound name shadows the class,
///     same rule `evaluate_call` already applies to a builtin name) and
///     names a `ClassModel` in `context.classes` — every positional
///     argument evaluates in order, every keyword argument evaluates
///     and pairs with its own name.
/// (b) `<ClassName>.model_validate(<dict literal>)` or
///     `TypeAdapter(<ClassName>).validate_python(<dict literal>)` —
///     pydantic's own parse surface (m-pydantic-schema.py's
///     `model_validate`/`TypeAdapter(...).validate_python` rows):
///     `ClassName` must be a bare Name in `context.classes`, and the
///     single argument must be a `Dict` literal so its keys map
///     directly to keyword rows; any other argument shape (a name, a
///     call, a non-literal key) is not construction this function
///     reads, and the call falls through to the ordinary
///     `evaluate_expression` path.
/// (c) `<ClassName>.model_validate_json(<any single argument>)` —
///     pydantic's JSON-text parse entry (`library/json.rst`'s own
///     cross-reference to `BaseModel.model_validate_json`; docs/
///     concepts/json.md, "you can also validate JSON directly... via
///     model_validate_json"). Unlike `model_validate`, the argument is
///     an opaque JSON STRING — a `text: str` parameter, most often —
///     so there is no literal content to read fields from at all;
///     every field falls straight to its own declared set the same
///     way `judge_construction` already answers a MISSING argument
///     (`(None, Some(declared)) => known_set(declared.set...)`),
///     which is exactly what calling it with empty positional and
///     keyword rows produces. `ClassName` must be a bare Name in
///     `context.classes`, and the call must carry exactly one
///     argument, no keywords — `model_validate_json`'s own single-
///     value signature; the argument's own shape is never read, only
///     its presence.
pub(in crate::check) fn construction_call_verdict(
    expr: &Expr,
    context: &WalkContext,
    environment: &Environment,
) -> Option<ConstructionVerdict> {
    let Expr::Call(call) = expr else {
        return None;
    };
    if let Expr::Name(callee) = call.func.as_ref() {
        // Unbound, OR bound to its OWN class-object value (the walk seeds
        // every visible class name to `instances::class_object_value`,
        // whose `source` is the class's own name) — calling the class
        // object IS the construction. Any other binding shadows the
        // class name, same rule evaluate_call applies to a builtin name.
        let callee_open = match environment.read(callee.id.as_str()) {
            None => true,
            Some(bound) => {
                bound.kind == refined_domain::abstract_value::Kind::Object
                    && bound.source == callee.id.as_str()
            }
        };
        if callee_open {
            // A class defined LOCALLY inside the walked body only lives in
            // `environment.classes()` (`merged_classes_for_body`'s own merge
            // over `context.classes`) — two different body-local classes
            // sharing a bare name (e.g. two functions each declaring their
            // own `class Person`) collide in the one shared
            // `context.classes` map, so the per-body table must win when
            // present, exactly as `instance_method_call_result` already
            // reads it.
            let classes = environment.classes().unwrap_or(&context.classes);
            if let Some(model) = classes.get(callee.id.as_str()) {
                let positional = evaluate_positional_arguments(&call.arguments.args, environment, context.kernel);
                let keyword = evaluate_keyword_arguments(&call.arguments.keywords, environment, context.kernel);
                return Some(judge_construction(model, &positional, &keyword, context.kernel));
            }
        }
        return None;
    }
    let Expr::Attribute(attribute) = call.func.as_ref() else {
        return None;
    };
    if attribute.attr.as_str() == "model_validate" {
        let model = class_model_of_bare_name(attribute.value.as_ref(), context, environment)?;
        let dict_argument = single_dict_argument(&call.arguments)?;
        let keyword = dict_literal_keyword_rows(dict_argument, environment, context.kernel)?;
        return Some(judge_construction(model, &[], &keyword, context.kernel));
    }
    if attribute.attr.as_str() == "model_validate_json" {
        let model = class_model_of_bare_name(attribute.value.as_ref(), context, environment)?;
        if !call.arguments.keywords.is_empty() {
            return None;
        }
        let [_single_argument] = call.arguments.args.as_ref() else {
            return None;
        };
        return Some(judge_construction(model, &[], &[], context.kernel));
    }
    if attribute.attr.as_str() == "validate_python" {
        // `TypeAdapter(<ClassName>).validate_python(<dict literal>)` —
        // the receiver is itself a Call: `TypeAdapter`'s own single
        // positional argument names the class.
        let Expr::Call(adapter_call) = attribute.value.as_ref() else {
            return None;
        };
        let Expr::Name(adapter_name) = adapter_call.func.as_ref() else {
            return None;
        };
        if adapter_name.id.as_str() != "TypeAdapter" {
            return None;
        }
        let [Expr::Name(class_name)] = adapter_call.arguments.args.as_ref() else {
            return None;
        };
        // Same locality rule as the bare-Name construction arm above: a
        // body-local class only lives in `environment.classes()`.
        if let Some(model) = environment.classes().unwrap_or(&context.classes).get(class_name.id.as_str()) {
            let dict_argument = single_dict_argument(&call.arguments)?;
            let keyword = dict_literal_keyword_rows(dict_argument, environment, context.kernel)?;
            return Some(judge_construction(model, &[], &keyword, context.kernel));
        }
        // THE ADAPTER-ALIAS ROUTE: `TypeAdapter(<alias>).validate_python(<scalar
        // expr>)` where `<alias>` is a bare `type X = ...` name
        // (`context.aliases`), not a `ClassModel`. Judges the ARGUMENT
        // expression's own value against the alias's declared set —
        // there is no field-by-field construction here, since the alias
        // names a scalar (or Literal) set, not an object shape.
        return adapter_alias_verdict(class_name, &call.arguments, context, environment);
    }
    None
}

/// `TypeAdapter(<alias name>).validate_python(<argument>)` against a
/// module-level `type <alias> = ...` set — `None` when `<alias>` is not
/// in `context.aliases` (the class route above already tried
/// `context.classes` and missed) or the call does not carry exactly one
/// positional, no-keyword argument (`validate_python`'s own single-value
/// shape).
pub(in crate::check) fn adapter_alias_verdict(
    class_name: &ruff_python_ast::ExprName,
    call_arguments: &ruff_python_ast::Arguments,
    context: &WalkContext,
    environment: &Environment,
) -> Option<ConstructionVerdict> {
    let declared_entry = context.aliases.get(class_name.id.as_str())?;
    if !call_arguments.keywords.is_empty() {
        return None;
    }
    let [argument_expr] = call_arguments.args.as_ref() else {
        return None;
    };
    let declared = DeclaredRefinement {
        set: declared_entry.set.clone(),
        spelling: class_name.id.as_str().to_owned(),
        admits_none: false,
        element: None,
        element_length: None,
        generator: None,
        members: None,
        positions: None,
        temporal: declared_entry.temporal.clone(),
        temporal_awareness: declared_entry.temporal_awareness,
    };
    let range = argument_expr.range();
    let mut value = evaluate_expression(argument_expr, environment, context.kernel);
    // A TEMPORAL alias (`declared.temporal` Some) reads its own STRING
    // argument as pydantic's own ISO-8601 parse — `TypeAdapter(FollowUp)
    // .validate_python("P30D")` parses a `timedelta`, an `Instant`-chart
    // alias parses a `datetime` — rather than judging the raw string
    // against the alias's own (empty, `declared.set` unused) scalar set.
    // Any other value shape (already-evaluated to a temporal
    // construction — a `datetime.date(...)`/`timedelta(...)`/
    // `datetime.datetime(...)` argument expression the ordinary
    // `evaluate_expression` path already tagged) reaches `judge`'s own
    // temporal law unchanged.
    if let Some(declared_temporal) = declared.temporal.clone() {
        if let Some(text) = crate::check::caller_exact_string_text(&value) {
            match crate::check::pydantic_temporal_parse(&text, declared_temporal.chart) {
                Some(parsed) => value = parsed,
                // THE GRAMMAR FIRE: a Duration-chart alias whose own
                // argument text is not even loosely ISO-8601-duration-
                // shaped (`pydantic_duration_value`'s own decline) is a
                // pydantic-core PARSE ERROR — "not a duration"
                // (showcase.py's own row), a designated fire this
                // function decides directly rather than leaving `judge`
                // to read the unparsed string as an ordinary structural
                // mismatch against the (empty) declared set.
                None if declared_temporal.chart == refined_sets::calendar_interpreter::TemporalChart::Duration => {
                    return Some(ConstructionVerdict {
                        fires: vec![(
                            range,
                            format!(
                                "a value of type '\"{text}\"' is not assignable to type '{}' — {text:?} is not ISO 8601 duration grammar",
                                class_name.id.as_str(),
                            ),
                        )],
                        instance: declared_set_instance(&declared),
                    });
                }
                None => {}
            }
        }
    }
    // LAX INT COERCION: pydantic's own `int` field (never `StrictInt`,
    // execution-verified 2026-08-17 against pydantic 2.13.4 —
    // `TypeAdapter(Age).validate_python("40")` coerces to `40`,
    // `.validate_python("200")` coerces to `200` and THEN fails the
    // range bound, `.validate_python("abc")`/`""` raise a parse error
    // this table does not model) accepts a plain base-10 digit string
    // (optional leading `-`, ASCII digits only — the narrow shape this
    // row needs; pydantic's fuller grammar also admits whitespace and
    // whole-valued float strings, out of scope here) and coerces it to
    // the int it spells before judging. `StrictInt` never coerces — a
    // `str` argument against a `StrictAge`-shaped alias reaches
    // `assignability::judge`'s own opaque/structural-mismatch law
    // unparsed, firing "not assignable" (StrictInt's own refusal,
    // execution-verified: `.validate_python("40")` raises `int_type`
    // with no coercion attempt).
    //
    // GATED ON A NUMERIC-SORTED ALIAS: this coercion is pydantic's `int`
    // FIELD behavior — it applies only when the alias itself declares an
    // int-sorted set (`requires_integer`, `refined_sets::refinement_
    // forms`'s own recognizer for the `Form::Integer` marker
    // `annotated_expression_set` pushes for `int`). m-pydantic-schema.py's
    // `Digits` (a STR-sorted pattern alias, `type Digits = Annotated[str,
    // Field(pattern=r"^[0-9]+$")]`) must NOT coerce
    // `TypeAdapter(Digits).validate_python("42")` — a digit-only STRING is
    // exactly what a `str`-sorted pattern alias accepts on its own terms,
    // and rewriting it to the int `42` before judging is judging the
    // wrong sort entirely. `plain_digit_string_value` only ever produces
    // an Integer-tagged value, so `requires_integer` is the precise gate:
    // a Float-sorted or str-sorted declared set never coerces.
    if value.kind == Kind::Values
        && value.kind_tag == Some(PrimitiveKind::String)
        && requires_integer(&declared_entry.set)
        && !context.strict_int_aliases.contains(class_name.id.as_str())
    {
        if let Some(parsed) = plain_digit_string_value(&value.values) {
            value = parsed;
        }
    }
    match judge(&value, &declared, context.kernel) {
        Verdict::Fire(message) => Some(ConstructionVerdict {
            fires: vec![(range, message)],
            // THE REFUSED-WRITE LAW (this file's own header note): the
            // answer carries the DECLARED SET, never the refused raw
            // value — this construction's own return type is very often
            // the SAME alias (`-> Age` on a `TypeAdapter(Age)` call), so
            // the outer sink (`walk_return`) judges this instance a
            // SECOND time against that identical declaration; handing
            // back the raw out-of-set value would fire there again for
            // the one refusal this function already reported.
            instance: declared_set_instance(&declared),
        }),
        Verdict::Silent => Some(ConstructionVerdict {
            fires: Vec::new(),
            instance: value,
        }),
        Verdict::Undetermined(_) => Some(ConstructionVerdict {
            fires: Vec::new(),
            // the same "keeps the DECLARED set" answer
            // `judge_construction`'s own Undetermined arm gives a
            // construction field — a later sink judging this value
            // against the SAME declaration (e.g. the function's own `->
            // Age` return annotation) sees a trivial self-match rather
            // than staying stuck on a value this table could not read.
            instance: declared_set_instance(&declared),
        }),
    }
}

/// A declared set as a bound value, tagged with its numeric sort when
/// the ground is provably numeric — the same guarded rule
/// `seed_parameters` applies to a declared set: `on_one_tuple_layer`
/// alone also reads a `Literal["A", "B"]` string-tuple union as "on the
/// one-tuple layer", so `states_sequence` must be false too, ruling out
/// that pun. Shared by `adapter_alias_verdict`'s Fire and Undetermined
/// arms, which both keep the declared set rather than the value this
/// table refused or could not read.
pub(in crate::check) fn declared_set_instance(declared: &DeclaredRefinement) -> AbstractValue {
    // A TEMPORAL declaration (`declared.set` unused/empty, `declared.
    // temporal` Some) — the same `"temporal_flow"`-tagged WINDOW value
    // `seed_parameters` seeds a temporal parameter with, carrying the
    // declaration's OWN bound. `assignability.rs`'s temporal law reads
    // this tag through `bounds_imply` (a later sink judging this
    // instance a second time against the SAME or a wider declaration
    // still proves, exactly the same self-match every other declared-
    // set fallback here already gives). Checked FIRST, mirroring the
    // "one active field" convention every other container-shaped
    // declaration's own fallback already keeps.
    if let Some(declared_temporal) = &declared.temporal {
        let mut instance = known_object(Vec::new(), None, true, TrustSpec, false);
        instance.source = "temporal_flow".to_owned();
        instance.temporal = Some(Box::new(declared_temporal.clone()));
        return instance;
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

/// A plain base-10 digit string's codepoints (optional leading `-`,
/// ASCII digits only — Python's `int()` grammar restricted to the shape
/// this row needs, `expressions.rs::is_valid_base_ten_int_string`'s
/// fuller sibling out of this file's reach) read as the int
/// `AbstractValue` it spells — pydantic's lax `int` coercion parses the
/// SAME digit text before range-judging it (execution-verified: `"200"`
/// coerces to `200`, then fails `le=120`). `None` for anything else
/// (a float string, a non-digit string, an empty string) — this table
/// declines rather than guessing a coercion pydantic itself would
/// refuse.
pub(in crate::check) fn plain_digit_string_value(code_points: &[f64]) -> Option<AbstractValue> {
    let text: String = code_points
        .iter()
        .map(|point| char::from_u32(*point as i64 as u32))
        .collect::<Option<String>>()?;
    let digits = text.strip_prefix('-').unwrap_or(&text);
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let parsed: i64 = text.parse().ok()?;
    Some(known_values(vec![parsed as f64], PrimitiveKind::Integer, TrustProved))
}

/// `<ClassName>` out of a bare-Name expression naming a class in
/// `environment.classes()` (falling back to `context.classes` when the
/// environment carries none — the same locality rule
/// `instance_method_call_result` already applies, since a class defined
/// LOCALLY inside the walked body only lives in the per-body table) — the
/// receiver shape `<ClassName>.model_validate` reads. `None` for anything
/// else (a non-Name receiver, or a Name that is either environment-bound to
/// something else or simply not a known class).
pub(in crate::check) fn class_model_of_bare_name<'a>(
    expr: &Expr,
    context: &'a WalkContext,
    environment: &'a Environment,
) -> Option<&'a ClassModel> {
    let Expr::Name(name) = expr else {
        return None;
    };
    // A name bound to its OWN class-object value (the walk seeds a
    // class's bare name to `instances::class_object_value`, whose
    // `source` is the class's own name) is still the constructor —
    // calling it IS the construction. Any OTHER binding shadows the
    // class name as before.
    if let Some(bound) = environment.read(name.id.as_str()) {
        let is_own_class_object = bound.kind == refined_domain::abstract_value::Kind::Object
            && bound.source == name.id.as_str();
        if !is_own_class_object {
            return None;
        }
    }
    environment.classes().unwrap_or(&context.classes).get(name.id.as_str())
}

/// The single positional argument of a call, when it is a `Dict`
/// literal — `model_validate`/`validate_python`'s own argument shape.
/// `None` for zero/multiple arguments, any keyword argument, or a
/// positional argument that is not a `Dict` display.
pub(in crate::check) fn single_dict_argument(arguments: &ruff_python_ast::Arguments) -> Option<&ruff_python_ast::ExprDict> {
    if !arguments.keywords.is_empty() {
        return None;
    }
    let [Expr::Dict(dict)] = arguments.args.as_ref() else {
        return None;
    };
    Some(dict)
}

/// A `{"key": value, ...}` literal's rows, mapped to `judge_construction`'s
/// own keyword-row shape: each entry's STRING key becomes the field
/// name, its value expression evaluates through `evaluate_expression`,
/// and the row's range is the VALUE expression's own range (so a fire
/// anchors at the value that refused, matching every other sink in this
/// file). `None` the moment any entry's key is not a plain string
/// literal (a computed key, a `**spread` entry) — the same all-or-
/// nothing posture `collection_models::dict_literal_value` already
/// takes for a dict display it cannot read exactly.
pub(in crate::check) fn dict_literal_keyword_rows(
    dict: &ruff_python_ast::ExprDict,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Vec<(String, AbstractValue, TextRange)>> {
    let mut rows = Vec::with_capacity(dict.items.len());
    for item in &dict.items {
        let Some(Expr::StringLiteral(key)) = item.key.as_ref() else {
            return None;
        };
        let value = evaluate_expression(&item.value, environment, kernel);
        rows.push((key.value.to_str().to_owned(), value, item.value.range()));
    }
    Some(rows)
}

/// Every positional argument of a construction call, evaluated in
/// order — the same per-argument evaluation `evaluate_call` already
/// does for a builtin, paired here with each argument's own range so
/// `judge_construction`'s fires anchor at the refusing argument.
pub(in crate::check) fn evaluate_positional_arguments(
    args: &[Expr],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Vec<(AbstractValue, TextRange)> {
    args.iter()
        .map(|arg| (evaluate_expression(arg, environment, kernel), arg.range()))
        .collect()
}

/// Every keyword argument of a construction call (`name=value` rows
/// only — `**spread` keywords carry no `arg` identifier and are
/// skipped, since this table cannot know which field a spread's keys
/// would land in).
pub(in crate::check) fn evaluate_keyword_arguments(
    keywords: &[ruff_python_ast::Keyword],
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> Vec<(String, AbstractValue, TextRange)> {
    keywords
        .iter()
        .filter_map(|keyword| {
            let name = keyword.arg.as_ref()?;
            let value = evaluate_expression(&keyword.value, environment, kernel);
            Some((name.id.as_str().to_owned(), value, keyword.value.range()))
        })
        .collect()
}
