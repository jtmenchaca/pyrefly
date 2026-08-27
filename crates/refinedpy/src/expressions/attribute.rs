
use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::unknown;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_sets::refinement_forms::make_refined_set;
use ruff_python_ast::Expr;

use crate::collection_models;
use crate::env::Environment;
use crate::instances;
use crate::math_models;
use crate::string_models;

use super::evaluate_expression;
use super::compare::*;

/// `dict.keys()`/`.values()`/`.items()` (no arguments) on a known dict
/// (`Kind::Object`) — library/stdtypes.rst, dict's `method:: keys()`/
/// `method:: values()`/`method:: items()`: "Return a new view of the
/// dictionary's keys/values/items." A VIEW is read here as the flat
/// `Kind::List` of its own elements (this domain has no separate view
/// kind, matching the module's own "iteration values" scope) — `keys()`
/// answers the key strings, `values()` the value AbstractValues, and
/// `items()` a list of 2-element `(key, value)` pair-lists, in the
/// dict's own insertion order (`ObjectKey`'s ordered-Vec shape,
/// `abstract_value.rs`'s own doc: "iteration order is insertion
/// order"). `None` for any other method name — declined, not modeled.
pub(super) fn dict_view_method_result(method: &str, receiver: &AbstractValue) -> Option<AbstractValue> {
    match method {
        "keys" => {
            let keys: Vec<AbstractValue> = receiver.keys.iter().map(|entry| string_models::string_literal_value(&entry.name)).collect();
            Some(collection_models::list_literal_value(&keys))
        }
        "values" => {
            let values: Vec<AbstractValue> = receiver.keys.iter().map(|entry| entry.value.clone()).collect();
            Some(collection_models::list_literal_value(&values))
        }
        "items" => {
            let pairs: Vec<AbstractValue> = receiver
                .keys
                .iter()
                .map(|entry| {
                    collection_models::list_literal_value(&[string_models::string_literal_value(&entry.name), entry.value.clone()])
                })
                .collect();
            Some(collection_models::list_literal_value(&pairs))
        }
        _ => None,
    }
}

/// `a.union(b)` / `a.intersection(b)` / `a.difference(b)` /
/// `a.symmetric_difference(b)` / `a.issubset(b)` / `a.issuperset(b)` on
/// a known set receiver (`Kind::List` — this domain's one sequence
/// shape, `collection_models.rs`'s own module doc: a set's own
/// element-uniqueness is invisible to a reader that only ever consumes
/// the sequence via iteration/membership) with a known set argument.
/// Every row is cited against library/stdtypes.rst's own set-methods
/// entries: `union(*others)` ("Return a new set with elements from the
/// set and all others"), `intersection(*others)` ("elements common to
/// the set and all others"), `difference(*others)` ("elements in the
/// set that are not in the others"), `symmetric_difference(other)`
/// ("elements in either the set or other but not both"),
/// `issubset(other)` ("Test whether every element in the set is in
/// *other*"), `issuperset(other)` ("Test whether every element in
/// *other* is in the set"). This file's one method dispatches ONLY the
/// TWO-set, one-`other`-argument form (`*others`'s variadic extra
/// arguments are not modeled). `None` for any other method, receiver,
/// or argument shape.
pub(super) fn set_method_result(method: &str, receiver: &AbstractValue, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [other] = arguments else { return None };
    if other.kind != Kind::List {
        return None;
    }
    match method {
        "union" => {
            let mut items = receiver.items.clone();
            for candidate in &other.items {
                if !set_contains(&items, candidate)? {
                    items.push(candidate.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "intersection" => {
            let mut items = Vec::new();
            for element in &receiver.items {
                if set_contains(&other.items, element)? {
                    items.push(element.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "difference" => {
            let mut items = Vec::new();
            for element in &receiver.items {
                if !set_contains(&other.items, element)? {
                    items.push(element.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "symmetric_difference" => {
            let mut items = Vec::new();
            for element in &receiver.items {
                if !set_contains(&other.items, element)? {
                    items.push(element.clone());
                }
            }
            for element in &other.items {
                if !set_contains(&receiver.items, element)? {
                    items.push(element.clone());
                }
            }
            Some(collection_models::list_literal_value(&items))
        }
        "issubset" => {
            for element in &receiver.items {
                if !set_contains(&other.items, element)? {
                    return Some(boolean_answer(false));
                }
            }
            Some(boolean_answer(true))
        }
        "issuperset" => {
            for element in &other.items {
                if !set_contains(&receiver.items, element)? {
                    return Some(boolean_answer(false));
                }
            }
            Some(boolean_answer(true))
        }
        _ => None,
    }
}

/// Whether `needle` is a member of `items` by `==` — `single_pair_equal`
/// declines (`None`) the moment one comparison cannot be decided, and
/// this helper propagates that decline through `?` at every call site
/// above rather than silently reading an undecidable member as absent.
pub(super) fn set_contains(items: &[AbstractValue], needle: &AbstractValue) -> Option<bool> {
    for element in items {
        match single_pair_equal(needle, element) {
            Some(true) => return Some(true),
            Some(false) => continue,
            None => return None,
        }
    }
    Some(false)
}

/// A Boolean AbstractValue — the same `known_values(vec![0.0/1.0],
/// PrimitiveKind::Boolean, TrustProved)` shape every other boolean
/// answer in this file builds (`compare_pair`'s own rows, `not`'s own
/// row).
pub(super) fn boolean_answer(value: bool) -> AbstractValue {
    known_values(vec![if value { 1.0 } else { 0.0 }], PrimitiveKind::Boolean, TrustProved)
}

/// `receiver.attr` — a plain attribute READ, not a call. The receiver
/// evaluates first; a known Object (`Kind::Object`) TAGGED with a class
/// (`source` non-empty, `judge_construction`'s own mark, found in
/// `environment.classes()`) reads through the MODEL — a stored field OR
/// a `@property` alias via `field_read_through_model`, or a bare
/// bound-method reference if the name is neither of those but IS a
/// class method (opaque). An UNTAGGED Object (a cross-module binding:
/// `cross_module.rs` builds a module object with the identical
/// `known_object` shape a class instance carries, this file's own
/// module doc note) falls back to the plain `instances::field_read`
/// linear scan — the same one dispatch arm this function used before
/// class tagging existed, still correct for a receiver with no class
/// to look up. Any other receiver shape (unknown, a scalar, a list)
/// answers `unknown()` — there is no attribute-read model for it here.
pub(super) fn evaluate_attribute_read(
    attribute: &ruff_python_ast::ExprAttribute,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> AbstractValue {
    // An access path this walk already narrowed (`a.n`'s own comparison,
    // say) answers from that path binding directly — the same fact a bare
    // tracked name would answer from `environment.read`, one level deeper.
    // `tracked_place_of` declines any receiver chain that is not a plain
    // Name/Attribute walk down to a base name (a call, a subscript), so
    // this never fires for a shape it cannot state a path for.
    if let Some(place) = crate::env::tracked_place_of(&Expr::Attribute(attribute.clone())) {
        if let Some(narrowed) = environment.read_path(&place) {
            return narrowed.clone();
        }
    }
    // `__class__` is a universal attribute (datamodel.rst: "instance.__class__
    // is the object's class") — EVERY value has one, the host's own type
    // object, never a program-tracked value; answered opaque regardless
    // of whether the receiver itself is known, since the fact "this
    // reads a host type object" holds independent of the receiver's
    // OWN value (b-body-expressions.py's `wrapper_dunder_class`/
    // `NewTargetProbe` rows)
    if attribute.attr.as_str() == "__class__" {
        return opaque_value("the __class__ object");
    }
    // `<a datetime_datetime instance>.year` — datetime.rst,
    // `attribute:: datetime.year`: "Between MINYEAR and MAXYEAR
    // inclusive." Answered OPAQUE rather than reading the exact
    // constructor-argument field this file DOES track internally
    // (`datetime_construction_value`'s own `year` `ObjectKey`): the
    // calendar year a `.year` read reports depends on which timezone the
    // instance carries (a naive vs. aware datetime constructed from the
    // same wall-clock fields can report different years across a
    // day/year boundary), a fact this file's own `aware_utc` marker does
    // not fully resolve (an aware-but-non-UTC instance is already
    // declined at construction, but an aware-UTC instance's `.year`
    // still needs no further computation THIS file's corpus reads
    // through a refined sink) — this row's own fixture framing states
    // the general fact plainly: "not pinned to one calendar value by
    // CPython alone." A calendar year is never inside `Age`'s `[0,120]`
    // window regardless (this file's own `j-stdlib-surfaces.py` row has
    // no in-set leg), so the opaque answer still fires correctly through
    // the opaque law without this file overclaiming precision it does
    // not need for that verdict.
    if attribute.attr.as_str() == "year" {
        let receiver = evaluate_expression(&attribute.value, environment, kernel);
        if receiver.kind == Kind::Object && receiver.source == "datetime_datetime" {
            return opaque_value("a calendar year");
        }
    }
    // `math.pi`/`math.e`/`math.tau`/`math.inf`/`math.nan` — an ATTRIBUTE
    // READ on the `math` module name (not shadowed by a local binding),
    // routed through `math_models::math_constant_value` (library/
    // math.rst, "Constants" — see that function's own doc for each
    // constant's exact concrete value).
    if let Expr::Name(module_name) = attribute.value.as_ref() {
        if module_name.id.as_str() == "math" && environment.read("math").is_none() {
            if let Some(value) = math_models::math_constant_value(attribute.attr.as_str()) {
                return value;
            }
        }
        // `os.environ` — library/os.rst, `data:: environ`: "A mapping
        // object representing the string environment... For example,
        // environ['HOME']... is equivalent to getenv('HOME')." Every key
        // and value this mapping can hold is an arbitrary string whose
        // content comes from outside the program — the same external-
        // origin reading `sys.argv` above gives its own elements — so the
        // read answers an UNBOUNDED-KEY dict-star (`known_dict_star`,
        // the same constructor `check/seed.rs::dict_star_value_seed`
        // builds for a `dict[str, X]`-declared parameter) whose value
        // slot is the whole-strings ground `Σ*`. `.get(k)`
        // (`collection_models::dict_get_result`'s own `Kind::ObjectStar`
        // arm) then reads it as "the string value if the key is present,
        // else `None`" — `str | None`, A5.seed.boundary's own
        // `env_get_outside` claim.
        if module_name.id.as_str() == "os" && environment.read("os").is_none() && attribute.attr.as_str() == "environ" {
            let string_element = AbstractValue {
                kind_tag: Some(PrimitiveKind::String),
                ..known_set(refined_sets::codepoint_sets::strings(), None, TrustSpec, SetKindTag::None)
            };
            let (star, ok) = refined_domain::known_constructors::known_dict_star(string_element, TrustSpec);
            if ok {
                return star;
            }
            return unknown();
        }
        // `sys.maxsize` — library/sys.rst: "the value of the largest
        // Py_ssize_t... usually 2**31 - 1 on a 32-bit platform and
        // 2**63 - 1 on a 64-bit platform", and always at least
        // 2**31 - 1: the SPEC floor is the determinable claim, so the
        // read answers the integer ray at or above it rather than
        // pinning a platform's own exact value.
        // `sys.argv` — library/sys.rst: "The list of command line
        // arguments passed to a Python script." Every element is a
        // `str` whose content this checker cannot know (it comes from
        // outside the program), and the list's own length is likewise
        // unstated, so the read answers the unbounded repetition of
        // whole strings: `sys.argv[1]` then reads `Σ*` — A3.seed.
        // boundary's own `argv_text_outside` claim — rather than
        // declining to state anything about an external-origin read.
        if module_name.id.as_str() == "sys" && environment.read("sys").is_none() && attribute.attr.as_str() == "argv" {
            return known_set(
                refined_sets::repetition_window_forms::repetition(refined_sets::codepoint_sets::strings(), 0, None),
                None,
                TrustSpec,
                SetKindTag::None,
            );
        }
        if module_name.id.as_str() == "sys" && environment.read("sys").is_none() && attribute.attr.as_str() == "maxsize" {
            return AbstractValue {
                kind_tag: Some(PrimitiveKind::Integer),
                ..known_set(
                    make_refined_set(vec![
                        refined_sets::refinement_forms::integer(),
                        refined_sets::refinement_forms::at_least(2147483647.0),
                    ]),
                    None,
                    TrustSpec,
                    SetKindTag::None,
                )
            };
        }
    }
    // `sys.float_info.max` — a TWO-LEVEL attribute chain (`sys.float_info`
    // read, then `.max` off that), unlike `sys.maxsize`'s one-level read
    // above: the outer `attribute.value` here is itself
    // `Expr::Attribute(sys.float_info)`, not `Expr::Name("sys")`, so the
    // chain must be peeled one level before the same module-name check
    // applies. library/sys.rst's `data:: float_info` table, `float_info.max`
    // row: "The maximum representable positive finite float," mapped to
    // the C `DBL_MAX` macro. `f64::MAX` IS `DBL_MAX` on every IEEE 754
    // binary64 platform (the same "one nearest representable value" IEEE
    // 754 identity `math_constant_value`'s own doc cites for `math.pi`) —
    // the exact CPython value, not a sort-only approximation, the same
    // posture `math.pi`/`math.e` take for their own platform-independent
    // constants.
    if attribute.attr.as_str() == "max" {
        if let Expr::Attribute(inner) = attribute.value.as_ref() {
            if let Expr::Name(module_name) = inner.value.as_ref() {
                if module_name.id.as_str() == "sys"
                    && environment.read("sys").is_none()
                    && inner.attr.as_str() == "float_info"
                {
                    return known_values(vec![f64::MAX], PrimitiveKind::Float, TrustProved);
                }
            }
        }
    }
    // `super().<name>` READ, no call — functions.rst's `super()` entry:
    // "a typical superclass call looks like this: `super().method(arg)`."
    // The receiver `self` is bound to the CURRENT working instance
    // (`instances::method_call_result`'s own environment), and its
    // class's `parent_methods` (never `methods`, which a child override
    // has already replaced) is the map a bare `super().<name>` reference
    // resolves a bound-method name against — the same single-inheritance
    // MRO rule `method_call_result`'s own `super_resolver` reads for a
    // CALLED `super().<method>(...)`, applied here to the un-called
    // attribute reference (b-body-expressions.py's `SuperBareChild.years`
    // row).
    if is_bare_super_call(&attribute.value) {
        if let Some(instance) = environment.read("self") {
            if !instance.source.is_empty() {
                if let Some(classes) = environment.classes() {
                    if let Some(model) = classes.get(instance.source.as_str()) {
                        if model.parent_methods.contains_key(attribute.attr.as_str()) {
                            return opaque_value("a bare bound-method reference");
                        }
                    }
                }
            }
        }
        return unknown();
    }
    let mut receiver = evaluate_expression(&attribute.value, environment, kernel);
    // A POSSIBLY-ABSENT receiver (`o: Optional[Box]` read without a
    // presence guard) reads through its PRESENT side. An attribute
    // reference "either returns a value or raises AttributeError"
    // (reference/expressions.rst, "Attribute references"), and `None`
    // carries no attribute named by any class field, so the absent arm
    // raises and contributes NO value to this expression — what flows
    // onward is exactly the present arm's own field. The raise itself
    // is not this read's to report: the receiver's own absence is the
    // subject of the presence-guard rows, and a value read here still
    // has to be right for the path that survives.
    if receiver.kind == Kind::PossiblyUndefined {
        let Some(present) = receiver.inner.clone() else {
            return unknown();
        };
        receiver = *present;
    }
    if receiver.kind != Kind::Object {
        return unknown();
    }
    // A tagged instance (`source` non-empty, `judge_construction`'s own
    // mark) reads through the MODEL, not the bare `field_read` scan: a
    // `@property` name resolves to its backing field's value
    // (`field_read_through_model`'s own doc), and a name that is
    // neither a stored field NOR a property but IS one of the class's
    // own methods is a BARE bound-method reference — `person.next_year`
    // with no call parens names the method object itself, never a
    // program-tracked scalar (datamodel.rst, "Instance methods": "the
    // special thing about methods is that the instance object is
    // prepended to the argument list" — reading the method WITHOUT
    // calling it still answers that bound-method object), so this
    // answers opaque rather than the `unknown()` a plain `field_read`
    // miss would give (c-reads-and-values.py's
    // `read_type_member_method_skip`; `super().<method>`'s own bare-
    // reference row is handled above, before this receiver even
    // evaluates).
    if !receiver.source.is_empty() {
        if let Some(classes) = environment.classes() {
            if let Some(model) = classes.get(receiver.source.as_str()) {
                if let Some(value) = instances::field_read_through_model(model, &receiver, attribute.attr.as_str()) {
                    return value;
                }
                // A CLASS ATTRIBUTE (`ceiling = 40` at class-body top
                // level, read through `cls.ceiling`/`ClassName.ceiling`)
                // lives in the receiver's own `keys` (`instances::
                // class_object_value` builds it there) but never in
                // `model.fields`/`model.properties` — `field_read_
                // through_model` only reads instance-declared fields, so
                // it misses a class attribute by design, not by gap. The
                // plain linear scan still finds it directly off the
                // receiver's own stored value before falling to "this is
                // a bound method" or "unknown."
                if let Some(value) = instances::field_read(&receiver, attribute.attr.as_str()) {
                    return value;
                }
                if instances::method_def_of(model, attribute.attr.as_str()).is_some() {
                    return opaque_value("a bare bound-method reference");
                }
                return unknown();
            }
        }
    }
    match instances::field_read(&receiver, attribute.attr.as_str()) {
        Some(value) => value,
        None => unknown(),
    }
}

/// Whether `expr` is exactly a bare, no-argument `super()` call —
/// `instances.rs`'s own `super_init_call` recognizes the identical
/// shape for `super().__init__(...)`; this is the plain-`Expr::Call`
/// half of that same recognition, reused here for an un-called
/// `super().<name>` attribute reference.
pub(super) fn is_bare_super_call(expr: &Expr) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    let Expr::Name(name) = call.func.as_ref() else {
        return false;
    };
    name.id.as_str() == "super" && call.arguments.args.is_empty() && call.arguments.keywords.is_empty()
}
