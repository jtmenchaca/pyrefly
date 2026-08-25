//! Construction judgment and class-object values.

use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use refined_domain::abstract_value::{
    known_set, unknown, AbstractValue, ObjectKey, SetKindTag,
};
use refined_domain::known_constructors::known_object;
use refined_domain::lattice_operations::truthiness;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::{Expr, Stmt, StmtFunctionDef};
use ruff_text_size::TextRange;

use crate::assignability::{judge, Verdict};
use crate::env::Environment;
use crate::expressions::evaluate_expression;

use super::model::ClassModel;

/// What judging one construction call against a class's fields
/// concluded: every Fire finding raised (an argument's own range plus
/// the message `assignability::judge` produced — never composed
/// here), and the instance's resulting value.
pub struct ConstructionVerdict {
    pub fires: Vec<(TextRange, String)>,
    pub instance: AbstractValue,
}

/// Judge one construction call's arguments against a class's declared
/// fields, mapping positional arguments to fields in declaration
/// order and then keyword arguments by name. A keyword naming no
/// field, or more positional arguments than the class has fields, is
/// an unmodeled construction — an overload, a `**kwargs`-absorbing
/// `__init__`, or simply a call this table cannot map exactly — and
/// answers `unknown()` with no fires: an unmapped construction never
/// guesses which field a stray argument might have landed in.
///
/// Each mapped argument judges through `assignability::judge` against
/// its field's declared refinement (when the field has one): `Fire`
/// pushes `(argument_range, message)` and the field holds the
/// argument's own value regardless (the refused-write law lives at
/// the WRITE sink in `check.rs`; this constructor's job is reporting
/// the fire and building the best-known instance, matching
/// `dict_literal_value`'s own "still holds a value" convention rather
/// than substituting the declared set the way `judge_and_bind` does
/// for a name binding — an object field slot has no reassignable name
/// downstream the way a plain variable does). `Undetermined` also
/// keeps the argument's own value at that field (the DECLARED set,
/// per the mission: "the field holds the DECLARED set as a known_set
/// value, TrustSpec — same construction check.rs's seed_parameters
/// uses"). A field with no argument takes its default if present,
/// else its declared set if present, else `unknown()`.

/// The counter `next_instance_identity` draws from — one process-wide
/// sequence, so two constructions anywhere in one checker run (even of
/// different classes, even on different threads if this checker ever
/// becomes concurrent) never mint the same id.
static NEXT_INSTANCE_IDENTITY: AtomicU32 = AtomicU32::new(0);

/// Mints a fresh per-construction identity — unique for the life of the
/// process, never reused. `judge_construction` stamps this onto every
/// instance it builds (`AbstractValue::instance_identity`'s own doc), so
/// two `Holder()` calls (the same class, the same AST call site, two
/// separate executions) always mint two distinct ids, exactly the way
/// `env.rs`'s `next_retained_callable_key` mints a fresh key per lambda/def
/// creation rather than keying by the AST's own range (that module's own
/// doc: a range key would let two creations of the same source text
/// silently conflate).
fn next_instance_identity() -> u32 {
    NEXT_INSTANCE_IDENTITY.fetch_add(1, Ordering::Relaxed)
}

pub fn judge_construction(
    model: &ClassModel,
    positional: &[(AbstractValue, TextRange)],
    keyword: &[(String, AbstractValue, TextRange)],
    kernel: &Arc<RefinedTSKernel>,
) -> ConstructionVerdict {
    if positional.len() > model.fields.len() {
        return ConstructionVerdict {
            fires: Vec::new(),
            instance: unknown(),
        };
    }
    let mut keyword_by_name: HashMap<&str, &(String, AbstractValue, TextRange)> = HashMap::new();
    for entry in keyword {
        keyword_by_name.insert(entry.0.as_str(), entry);
    }
    let known_field_names: std::collections::HashSet<&str> =
        model.fields.iter().map(|field| field.name.as_str()).collect();
    if keyword_by_name.keys().any(|name| !known_field_names.contains(name)) {
        return ConstructionVerdict {
            fires: Vec::new(),
            instance: unknown(),
        };
    }

    let mut fires = Vec::new();
    let mut entries: Vec<ObjectKey> = Vec::new();
    for (index, field) in model.fields.iter().enumerate() {
        let argument = positional
            .get(index)
            .map(|(value, range)| (value.clone(), *range))
            .or_else(|| keyword_by_name.get(field.name.as_str()).map(|(_, value, range)| (value.clone(), *range)));

        let field_value = match argument {
            Some((value, range)) => match &field.declared {
                Some(declared) => match judge(&value, declared, kernel) {
                    Verdict::Fire(message) => {
                        fires.push((range, message));
                        value
                    }
                    Verdict::Silent => value,
                    Verdict::Undetermined(_) => known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None),
                },
                None => value,
            },
            None => match (&field.default, &field.declared) {
                (Some(default), _) => default.clone(),
                (None, Some(declared)) => known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None),
                (None, None) => unknown(),
            },
        };
        entries.push(ObjectKey {
            name: field.name.clone(),
            numeric: false,
            value: field_value,
        });
    }

    let mut instance = known_object(entries, None, true, TrustSpec, false);
    // source carries the constructing class's name so a later
    // receiver.method(...) call can find the ClassModel in the
    // environment's class table; empty on every non-instance object.
    instance.source = model.name.clone();
    // instance_identity carries THIS call's own fresh id — distinct from
    // `source`, which every instance of `model` shares. Two `Holder()`
    // calls build two different instances; a dict keyed by one must not
    // answer a lookup by the other (`collection_models::known_dict_key`'s
    // own identity arm reads this field to tell them apart).
    instance.instance_identity = Some(next_instance_identity());

    // pydantic's own post-construction hook: `model_post_init(self,
    // __context)` runs immediately after every field is set
    // (docs/concepts/models.md's own "Post-init processing"), so a
    // dependent check written there (m-pydantic-schema.py's `Range`:
    // `if self.hi < self.lo: raise ValueError(...)`) is this
    // construction's own business, not a later sink's. Anchored at the
    // LAST mapped argument's range — a cross-field check has no single
    // refusing argument to blame, and this is the closest token to the
    // call's own closing paren among the ranges this function already
    // carries.
    if let Some(post_init) = model.methods.get("model_post_init") {
        if let Some(anchor) = keyword.last().map(|(_, _, range)| *range).or_else(|| positional.last().map(|(_, range)| *range)) {
            if let Some(message) = post_init_provable_raise(post_init, &instance, kernel) {
                fires.push((anchor, message));
            }
        }
    }

    ConstructionVerdict { fires, instance }
}

/// `model_post_init`'s own body, read ONLY in the one shape the corpus
/// spells: a SINGLE top-level `if <condition>: raise <exc>` statement
/// (no `elif`/`else`, no other statement before or after it) — the
/// dependent-check shape pydantic's own docs name for this hook
/// (docs/concepts/models.md, "Post-init processing" — the hook "will
/// be called... to perform additional validation"). `self` binds to
/// `instance` (already fully built — every field's own value, judged
/// or not, is in place, matching real pydantic's own construction
/// order: fields set, THEN `model_post_init` runs) and the condition
/// evaluates through `evaluate_expression`'s ordinary comparison
/// reading, restricted to `self.<field>` operands `field_read` already
/// answers.
///
/// `Some(message)` only when the condition is PROVABLY true
/// (`truthiness`'s `(true, true)` answer) — the same honest-decline
/// discipline every other provable-raise reader in this checker takes:
/// an undetermined or provably-false condition never fires here.
/// `raise <exc>`'s own message reads `<exc>`'s single string-literal
/// argument when `<exc>` is a bare `Call` (`ValueError("...")`,
/// `raise <name>` alone, or a computed message, states nothing this
/// reader can quote, so the message falls back to the exception
/// callee's own bare name). Any other body shape (more than one
/// top-level statement, an `elif`/`else` clause, a non-`Raise` `if`
/// body, a body that is not exactly one `if`) declines — `None`, never
/// a guessed fire.
fn post_init_provable_raise(
    post_init: &StmtFunctionDef,
    instance: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<String> {
    let [Stmt::If(if_stmt)] = post_init.body.as_slice() else {
        return None;
    };
    if !if_stmt.elif_else_clauses.is_empty() {
        return None;
    }
    let [Stmt::Raise(raise_stmt)] = if_stmt.body.as_slice() else {
        return None;
    };

    let mut environment = Environment::new(Default::default());
    environment.bind("self", instance.clone());
    let test_value = evaluate_expression(if_stmt.test.as_ref(), &environment, kernel);
    let (truthy, known) = truthiness(&test_value);
    if !known || !truthy {
        return None;
    }

    Some(post_init_raise_message(raise_stmt))
}

/// `model_post_init`'s own construction-site fire message: "this
/// expression provably raises `<ExcType>`: `<plain detail>`" — the
/// same voice `expressions::provable_raise` already speaks, quoting
/// `raise <exc>`'s own exception name and its single string-literal
/// argument when `<exc>` is a bare `Call` (`ValueError("hi must be >=
/// lo")` reads as `ValueError: hi must be >= lo`); any other `<exc>`
/// shape (a bare name, a computed argument, no argument at all) still
/// names the exception type alone.
fn post_init_raise_message(raise_stmt: &ruff_python_ast::StmtRaise) -> String {
    let Some(exc) = raise_stmt.exc.as_deref() else {
        return "this construction provably raises an exception".to_owned();
    };
    let Expr::Call(call) = exc else {
        return "this construction provably raises an exception".to_owned();
    };
    let Expr::Name(exc_name) = call.func.as_ref() else {
        return "this construction provably raises an exception".to_owned();
    };
    let detail = call
        .arguments
        .args
        .first()
        .and_then(|arg| match arg {
            Expr::StringLiteral(literal) => Some(literal.value.to_str().to_owned()),
            _ => None,
        });
    match detail {
        Some(detail) => format!("this construction provably raises {}: {}", exc_name.id.as_str(), detail),
        None => format!("this construction provably raises {}", exc_name.id.as_str()),
    }
}

/// The CLASS OBJECT's own initial value — e-class-and-function.py's
/// `class_attribute_write`: `Counted.total = 40` then `Counted.total`
/// read back, a write/read pair that never touches any INSTANCE (no
/// `Counted(...)` construction happens on this row at all). Distinct
/// from `judge_construction`'s instance value: this reads `model.
/// class_attributes` alone (never `model.fields`, which are per-instance
/// slots this class object does not carry), tagged with the SAME
/// `source = model.name` convention `judge_construction` uses so
/// `write_named_field`/`field_read_through_model` (which only ever check
/// `instance.kind == Kind::Object` and a non-empty `source`) read/write
/// through it with NO new machinery — a class object and an instance
/// object share one representation, distinguished only by which table
/// (`class_attributes` vs `fields`) built their starting keys.
///
/// `check.rs`'s own `Stmt::ClassDef` walk binds this value under the
/// class's own bare name, in the ENCLOSING environment (the scope where
/// the class statement itself executes) — the environment slot
/// `Counted.total = 40`'s attribute-write law (a bare-Name receiver
/// bound to a tagged `Kind::Object`) then finds and rebinds, exactly the
/// same way an instance variable already does.
pub fn class_object_value(model: &ClassModel) -> AbstractValue {
    let entries: Vec<ObjectKey> = model
        .class_attributes
        .iter()
        .filter_map(|attribute| {
            attribute.default.clone().map(|value| ObjectKey {
                name: attribute.name.clone(),
                numeric: false,
                value,
            })
        })
        .collect();
    let mut value = known_object(entries, None, true, TrustSpec, false);
    value.source = model.name.clone();
    value
}
