//! Instance field reads, writes, and judgments.

use std::sync::Arc;

use refined_domain::abstract_value::{AbstractValue, Kind, ObjectKey};
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;

use crate::assignability::{judge, Verdict};

use super::model::ClassModel;

/// `self.<name>` recognition: an `Attribute` expression whose own
/// value is the bare name `self` — the first parameter's conventional
/// spelling. This function does not itself verify the receiver is
/// __init__'s actual first parameter (a `def __init__(this, ...)`
/// naming its receiver something other than `self` is out of the
/// corpus band this file serves; `self` is Python's own overwhelming
/// convention, not a keyword, so a literal name match is the same
/// honest-recognition posture `surface.rs`'s `names_field` takes for
/// other by-spelling recognitions). `pub`: both `check.rs`'s
/// `bind_or_forget_target`/`write_named_field` and `summaries.rs`'s
/// restricted-body interpreter (its own `write_self_field`,
/// `interpret_aug_assign`) recognize the identical `self.<name>` shape
/// through this one function, rather than each re-deriving it.
pub fn self_attribute_name(target: &Expr) -> Option<String> {
    let Expr::Attribute(attribute) = target else {
        return None;
    };
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return None;
    };
    if receiver.id.as_str() != "self" {
        return None;
    }
    Some(attribute.attr.as_str().to_owned())
}

/// `instance.field` — the field's value out of a known_object
/// instance, matching `collection_models.rs`'s own dict-key access
/// pattern (`dict_key_read`): a linear scan of `keys` for the matching
/// name. `None` for anything else — an unknown instance, an instance
/// missing that key, or a non-`Kind::Object` value (this table never
/// builds any other kind, but a caller may hand in an arbitrary
/// AbstractValue).
///
/// `instance.field` reads a STORED field. A `@property` name is never
/// a stored field — it is a read ALIAS the model states, so a
/// property read routes through `field_read_through_model` instead,
/// which resolves the alias to its backing name before calling this
/// function.
pub fn field_read(instance: &AbstractValue, field: &str) -> Option<AbstractValue> {
    if instance.kind != Kind::Object {
        return None;
    }
    instance
        .keys
        .iter()
        .find(|entry| entry.name == field && !entry.numeric)
        .map(|entry| entry.value.clone())
}

/// `self.<field> = v` — the struct-updated instance with `field` set to
/// `value`, every other stored key AND every other `AbstractValue` field
/// (`source` included — the constructing class's tag must survive a
/// write, since a later `receiver.method(...)` call still needs it to
/// find the `ClassModel`) preserved from `instance` unchanged. `None`
/// for a non-`Kind::Object` instance — there is no field slot to write
/// on anything else this table builds. A field name absent from
/// `instance.keys` is APPENDED as a new entry (an ordinary Python
/// attribute gain, `field_write_judgment`'s own doc: "an ordinary
/// Python attribute gain is not a blocker") rather than declined.
pub fn field_write(instance: &AbstractValue, field: &str, value: AbstractValue) -> Option<AbstractValue> {
    if instance.kind != Kind::Object {
        return None;
    }
    let mut keys = instance.keys.clone();
    match keys.iter_mut().find(|entry| entry.name == field && !entry.numeric) {
        Some(entry) => entry.value = value,
        None => keys.push(ObjectKey {
            name: field.to_owned(),
            numeric: false,
            value,
        }),
    }
    Some(AbstractValue {
        keys,
        ..instance.clone()
    })
}

/// `box.age` where `age` may be a stored field OR a `@property` read
/// alias: a property name resolves to its `backing` field's own value
/// (`PropertyModel`'s doc — "the property `<name>` is a READ alias of
/// `<backing>`"); any other name reads the instance's stored field
/// directly, same as `field_read`.
pub fn field_read_through_model(model: &ClassModel, instance: &AbstractValue, field: &str) -> Option<AbstractValue> {
    match model.properties.get(field) {
        Some(property) => field_read(instance, &property.backing),
        None => field_read(instance, field),
    }
}

/// `self.x = v` / `obj.x = v` — judge a field write against the
/// class's declared refinement for that field. `None` when the field
/// carries no declared refinement (an ordinary unrefined field write,
/// not a blocker) OR when the class has no field by that name (an
/// attribute the model does not track — not this function's business
/// to invent a verdict for). A `@property` name judges against its OWN
/// setter-parameter refinement (`PropertyModel.declared`) rather than
/// any refinement its `backing` field carries — the setter's parameter
/// annotation is the more specific claim for a write through the
/// accessor (`PropertyModel`'s own doc).
pub fn field_write_judgment(
    model: &ClassModel,
    field: &str,
    value: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<Verdict> {
    if let Some(property) = model.properties.get(field) {
        let declared = property.declared.as_ref()?;
        return Some(judge(value, declared, kernel));
    }
    let declared = model.fields.iter().find(|f| f.name == field)?.declared.as_ref()?;
    Some(judge(value, declared, kernel))
}
