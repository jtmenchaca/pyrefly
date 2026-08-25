use std::collections::HashMap;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprSubscript;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::assignability::Verdict;
use crate::bytes_models;
use crate::bytes_models::BytesAnswer;
use crate::collection_models::dict_with_item;
use crate::collection_models::dict_without_item;
use crate::collection_models::list_literal_value;
use crate::collection_models::list_with_item;
use crate::collection_models::sliced_delete_receiver;
use crate::env::Environment;
use crate::expressions::evaluate_expression;
use crate::expressions::slice_bound_index;
use crate::instances;
use crate::typereading::DeclaredRefinement;

use super::super::Finding;
use super::super::WalkContext;
use super::forget_target_names;
use super::judge_and_bind;

/// A plain `Assign` target: binds a plain name to the evaluated value;
/// tuple/list targets attempt KNOWN-TUPLE DESTRUCTURING
/// (`bind_known_sequence_target`) when the RHS is a known `Kind::List`,
/// falling back to forgetting every name they touch when it is not (the
/// walk cannot destructure a value it cannot see the length/elements
/// of).
///
/// FIELD-WRITE LAW: `<receiver>.<field> = v` where `receiver` is a bare
/// Name bound to a TAGGED instance (`Kind::Object`, a non-empty `source`
/// naming a `ClassModel` this environment can find) is JUDGED, through
/// `write_named_field` — `self` inside a method body walked through
/// `walk_method_def`'s self-seeding (`self_attribute_name`'s own
/// recognition), and any OTHER local name holding a tagged instance
/// (`box.age = 200`, `over_box.age = 200` — e-class-and-function.py's
/// `property_getter_setter`, q-decline-names.py's `setter_effect_read_
/// through_getter`) alike: the receiver's class resolves through
/// `environment.classes()`, and `instances::field_write_judgment` judges
/// `v` against the field's own declared refinement exactly like any
/// other write sink in this file (`Fire` pushes an RTS7001 at the
/// value's own range; `Undetermined` records this body's blocker).
/// Either way the receiver REBINDS to `instances::field_write`'s updated
/// instance (never forgotten) — a later `<receiver>.<field>` read in the
/// SAME straight-line body must see the write, matching every other
/// known-write sink's own read-after-write law in this file.
/// `field_write_judgment` returning `None` (an unrefined field, or a
/// field the model does not declare) still rebinds through `field_write`
/// with no Fire — an ordinary Python attribute gain is not a blocker.
///
/// Falls back to forgetting the RECEIVER's own base name — the leftmost
/// `Name` under the attribute chain (`receiver_base_name`) — when the
/// receiver is not a bare Name bound to a tagged instance at all (an
/// arbitrary attribute chain, an untagged value, a class this
/// environment cannot find): a known instance bound to that name may
/// carry a stale field value for `x` after this write, and this file
/// does not track field-level state through an unresolved attribute
/// write, so forgetting the whole receiver is the one sound answer
/// there.
/// STALE-RECEIVER SOUNDNESS, law (b): a subscript target (`name[key] =
/// value`, bare-Name receiver only) evaluates the receiver and the key
/// expressions, then replays the write through
/// `collection_models::dict_with_item` (an Object receiver) or
/// `list_with_item` (a List receiver): `Some(new receiver)` rebinds
/// `name` to it, so a later read sees the write (a-statements.py's
/// `collection_mutators`: `by_name["ann"] = 40` must leave `by_name`
/// holding `{"ann": 40}`, not the stale `{}`); `None` (an unknown
/// receiver, a non-Name receiver, a key/index this walk cannot read
/// exactly, or an index outside the list's current bounds) FORGETS
/// `name` — the pre-write value must not survive an unread write, the
/// same honesty every other decline in this file already keeps.
pub(in crate::check) fn bind_or_forget_target(
    target: &Expr,
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match target {
        Expr::Name(name) => environment.bind(name.id.as_str(), value.clone()),
        Expr::Tuple(tuple) => {
            if !bind_known_sequence_target(
                &tuple.elts,
                value,
                value_range,
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for element in &tuple.elts {
                    forget_target_names(element, environment);
                }
            }
        }
        Expr::List(list) => {
            if !bind_known_sequence_target(
                &list.elts,
                value,
                value_range,
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for element in &list.elts {
                    forget_target_names(element, environment);
                }
            }
        }
        Expr::Starred(starred) => forget_target_names(starred.value.as_ref(), environment),
        Expr::Attribute(attribute) => {
            if let Some(field) = instances::self_attribute_name(target) {
                if write_named_field("self", &field, value, value_range, context, environment, out) {
                    return;
                }
            } else if let Expr::Name(receiver) = attribute.value.as_ref() {
                // NAMED-RECEIVER FIELD WRITE: `box.age = v` where `box`
                // (any bare name, not just `self`) is bound to a tagged
                // instance — e-class-and-function.py's
                // `property_getter_setter` (`over_box.age = 200` through
                // a `@property` setter) and q-decline-names.py's
                // `setter_effect_read_through_getter` both write through a
                // LOCAL variable holding the instance, never `self`. The
                // same judged-and-rebound law `write_named_field` already
                // gives `self` applies unchanged: the receiver name is
                // just a different environment slot to re-read/rebind.
                if write_named_field(
                    receiver.id.as_str(),
                    attribute.attr.as_str(),
                    value,
                    value_range,
                    context,
                    environment,
                    out,
                ) {
                    return;
                }
            }
            if let Some(base_name) = receiver_base_name(attribute.value.as_ref()) {
                environment.forget(base_name);
            }
        }
        Expr::Subscript(subscript) => {
            bind_or_forget_subscript_target(subscript, value, context, environment);
        }
        _ => {}
    }
}

/// The FIELD-WRITE LAW (see `bind_or_forget_target`'s own doc): `<receiver>.<field>
/// = value`, judged and rebound under `receiver_name` — the environment
/// slot a bare-Name receiver is bound under, `self` inside a method body
/// (`self_attribute_name`'s own recognition) or any other local name
/// holding a tagged instance (`box.age = 200`, `over_box.age = 200`).
/// Returns `true` when `receiver_name` reads as a tagged instance whose
/// class this environment can find — the write is fully handled either
/// way (judged and rebound), and the caller must not ALSO run its own
/// forget-the-receiver fallback. Returns `false` when the receiver is
/// unbound, untagged, or its class is not in `environment.classes()` —
/// the caller's existing fallback is the honest answer there.
pub(in crate::check) fn write_named_field(
    receiver_name: &str,
    field: &str,
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> bool {
    let Some(instance) = environment.read(receiver_name) else {
        return false;
    };
    if instance.kind != Kind::Object || instance.source.is_empty() {
        return false;
    }
    let Some(classes) = environment.classes() else {
        return false;
    };
    let Some(model) = classes.get(instance.source.as_str()) else {
        return false;
    };
    if let Some(Verdict::Fire(message)) = instances::field_write_judgment(model, field, value, context.kernel) {
        out.push(Finding {
            range: value_range,
            code: "RTS7001",
            message,
        });
    }
    // A `@property` name is never a stored slot — `field_read_through_model`
    // resolves a read of `field` to the property's `backing` name
    // (`instances.rs`'s own doc), so the write must land on that SAME
    // backing name or a later read through the property sees the instance's
    // pre-write value instead of what was just assigned
    // (e-class-and-function.py's `property_getter_setter`,
    // q-decline-names.py's `setter_effect_read_through_getter`: writing
    // `over_box.age = 200` and reading `over_box.age` back must see 200).
    let write_target = match model.properties.get(field) {
        Some(property) => property.backing.as_str(),
        None => field,
    };
    // re-read after the class-table lookup above (which only borrowed
    // `environment`) so the write below can borrow it mutably; the
    // receiver is still exactly the instance just read, since nothing in
    // between could have rebound it.
    let instance = environment.read(receiver_name).expect("checked Some above").clone();
    if let Some(updated) = instances::field_write(&instance, write_target, value.clone()) {
        // ALIASING: `same = account; same.balance = -20` must leave
        // `account`'s own slot reading the written-through instance too
        // — `Environment` tracks a value per NAME, so the direct rebind
        // below only ever touches `receiver_name`'s own slot.
        // `instance_identity` (`instances::judge_construction`'s own
        // per-construction tag) is how two different names holding the
        // SAME construction are told apart from two names holding two
        // separate `Holder()` calls of the same class; `rebind_aliases_
        // of_instance` finds every OTHER name carrying the identical id
        // and brings it back in step with this write. An instance with
        // no `instance_identity` at all (built some way other than
        // `judge_construction`) has no alias set to reconcile, and the
        // sweep below is then simply a no-op for every other name.
        if let Some(identity) = instance.instance_identity {
            environment.rebind_aliases_of_instance(identity, receiver_name, &updated);
        }
        environment.bind(receiver_name, updated);
    }
    true
}

/// KNOWN-TUPLE DESTRUCTURING: `(a, b, ...) = value` / `[a, b, ...] =
/// value`, where `value` is a KNOWN `Kind::List` (a-statements.py's
/// `tuple_unpack_ok`/`starred_unpack_ok`/`nested_tuple_unpack_ok` rows —
/// CPython does not distinguish list vs. tuple targets or RHS shape for
/// unpacking, simple_stmts.rst's `target: "(" [target_list] ")" | "["
/// [target_list] "]"` grammar treats both parenthesized and bracketed
/// target lists the same way). Returns `false` (no binding performed,
/// caller falls back to forgetting every name) when `value` is not a
/// known list — an unknown RHS states nothing about how many elements
/// there are, so this law does not apply and the existing forget-all
/// answer is the sound one.
///
/// With no starred element: `elements.len()` must equal `items.len()`
/// exactly — a mismatch is CPython's own `ValueError` ("too many values
/// to unpack (expected N)" / "not enough values to unpack (expected N,
/// got M)", both confirmed by execution against python3.12), fired here
/// as RTS7001 at the RHS value's own range, with EVERY target name
/// forgotten (the assignment never completes, so nothing binds). A
/// length match binds each element positionally: `Expr::Name` targets
/// bind (through `judge_and_bind` when the name carries a recorded
/// declared refinement, exactly like a plain `a = value` target — see
/// `walk_assign`'s own doc), and a nested `Expr::Tuple`/`Expr::List`
/// target recurses on that position's own element value.
///
/// With one starred element (`first, *rest = value` — a `SyntaxError` to
/// have more than one, so this table never needs to detect that case
/// itself): the elements BEFORE the star bind to the LIST'S head
/// positions, the elements AFTER the star bind to its TAIL positions
/// (counted from the end), and the starred name itself binds a
/// `Kind::List` of every element in between (`known_list` — the exact
/// "the middle slice" `first, *rest = years` gives `rest` in CPython).
/// Too few items for the non-starred elements alone (`head.len() +
/// tail.len() > items.len()`) is the starred row's own `ValueError`
/// ("not enough values to unpack (expected at least N, got M)",
/// confirmed by execution) — same fire-and-forget-all answer.
pub(in crate::check) fn bind_known_sequence_target(
    elements: &[Expr],
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> bool {
    if value.kind != Kind::List {
        return false;
    }
    let items = &value.items;
    let starred_position = elements.iter().position(|element| matches!(element, Expr::Starred(_)));

    let Some(star_index) = starred_position else {
        if elements.len() != items.len() {
            out.push(Finding {
                range: value_range,
                code: "RTS7001",
                message: format!(
                    "this expression provably raises ValueError: {}",
                    unpack_mismatch_detail(elements.len(), items.len(), false),
                ),
            });
            for element in elements {
                forget_target_names(element, environment);
            }
            return true;
        }
        for (element, item) in elements.iter().zip(items.iter()) {
            bind_sequence_element(element, item, context, aug_assign_refinements, environment, out);
        }
        return true;
    };

    let head = &elements[..star_index];
    let tail = &elements[star_index + 1..];
    if head.len() + tail.len() > items.len() {
        out.push(Finding {
            range: value_range,
            code: "RTS7001",
            message: format!(
                "this expression provably raises ValueError: {}",
                unpack_mismatch_detail(head.len() + tail.len(), items.len(), true),
            ),
        });
        for element in elements {
            forget_target_names(element, environment);
        }
        return true;
    }
    for (element, item) in head.iter().zip(items.iter()) {
        bind_sequence_element(element, item, context, aug_assign_refinements, environment, out);
    }
    let tail_start = items.len() - tail.len();
    for (element, item) in tail.iter().zip(items[tail_start..].iter()) {
        bind_sequence_element(element, item, context, aug_assign_refinements, environment, out);
    }
    let Expr::Starred(starred) = &elements[star_index] else {
        unreachable!("star_index is the position matched against Expr::Starred above")
    };
    if let Expr::Name(name) = starred.value.as_ref() {
        let middle = list_literal_value(&items[head.len()..tail_start]);
        environment.bind(name.id.as_str(), middle);
    }
    true
}

/// One destructured position's own target: a bare name binds (through
/// `judge_and_bind` when the name carries a recorded declared
/// refinement — the same table an ordinary `x = value` target reads),
/// and a nested `Tuple`/`List` target recurses through
/// `bind_known_sequence_target` on that position's own known element —
/// a non-list element at a nested-tuple position is itself an unknown
/// shape to that recursive call, which then forgets that sub-target's
/// own names, matching the top-level "unknown RHS forgets" rule at
/// whatever depth it occurs.
pub(in crate::check) fn bind_sequence_element(
    element: &Expr,
    item: &AbstractValue,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match element {
        Expr::Name(name) => match aug_assign_refinements.get(name.id.as_str()) {
            Some(declared) => {
                let declared = declared.clone();
                judge_and_bind(name.id.as_str(), item.clone(), &declared, element.range(), context, environment, out);
            }
            None => environment.bind(name.id.as_str(), item.clone()),
        },
        Expr::Tuple(tuple) => {
            if !bind_known_sequence_target(
                &tuple.elts,
                item,
                element.range(),
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for nested in &tuple.elts {
                    forget_target_names(nested, environment);
                }
            }
        }
        Expr::List(list) => {
            if !bind_known_sequence_target(
                &list.elts,
                item,
                element.range(),
                context,
                aug_assign_refinements,
                environment,
                out,
            ) {
                for nested in &list.elts {
                    forget_target_names(nested, environment);
                }
            }
        }
        _ => forget_target_names(element, environment),
    }
}

/// The CPython `ValueError` wording for a length-mismatch unpack,
/// confirmed by execution against python3.12: without a starred target,
/// "too many values to unpack (expected N)" when the RHS has MORE items
/// than targets, else "not enough values to unpack (expected N, got M)";
/// with a starred target (`has_star`), the expected count is a floor —
/// "not enough values to unpack (expected at least N, got M)" (a
/// starred target can never see "too many": it absorbs every surplus
/// element into its own list, so this row only ever under-supplies).
pub(in crate::check) fn unpack_mismatch_detail(expected: usize, got: usize, has_star: bool) -> String {
    if has_star {
        return format!("not enough values to unpack (expected at least {expected}, got {got})");
    }
    if got > expected {
        format!("too many values to unpack (expected {expected})")
    } else {
        format!("not enough values to unpack (expected {expected}, got {got})")
    }
}

/// `name[key] = value` — see `bind_or_forget_target`'s own doc for law
/// (b)'s full contract. Only a bare-`Name` receiver is replayed; any
/// other receiver shape (`obj.attr[key] = v`, a chained subscript) has
/// no single environment slot to rebind and is left untouched, matching
/// this file's existing "no element-level model" posture for a receiver
/// it cannot name.
///
/// A `bytes`/`bytearray`/`memoryview` receiver (`bytes_models::tagged`'s
/// own species word) routes through `bytes_models::bytes_write_answer`
/// FIRST, before the plain `list_with_item` path below ever runs. A
/// write CPython provably raises (an out-of-`0..=255` bytearray/
/// memoryview element, or ANY write onto an immutable `bytes`) leaves
/// the receiver COMPLETELY UNTOUCHED — no bind, no forget, no finding.
/// The write never took effect, so the pre-write contents are still
/// exactly right (rebinding OR forgetting would both be a weaker, wrong
/// answer — a forgotten receiver reads Undetermined past this point even
/// though its value is still fully known), and the raise itself is not
/// this function's own finding to report: it is not a judgment against a
/// declared refinement (`assignability::judge`'s own seam), just a
/// LANGUAGE fact this function uses to keep its model honest
/// (p-typed-array.py's own `bytes_is_immutable` docstring: "No
/// expect-error marker belongs on the raise itself"). A write that
/// provably SUCCEEDS applies through the ordinary `list_with_item` path
/// below unchanged (200 into a bytearray is in `0..=255`, so it writes
/// through even though 200 would refuse against a declared `Age` — a
/// different, later question `judge_and_bind` owns, never this
/// function). An UNDECIDABLE bytes-like write (an unknown value) falls
/// through to the same decline-and-forget the untagged path already
/// takes, honest about a write this function cannot prove either way.
pub(in crate::check) fn bind_or_forget_subscript_target(
    subscript: &ExprSubscript,
    value: &AbstractValue,
    context: &WalkContext,
    environment: &mut Environment,
) {
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        return;
    };
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    let key_value = evaluate_expression(subscript.slice.as_ref(), environment, context.kernel);
    if receiver_value.kind == Kind::List && receiver_value.kind_word.is_some() {
        match bytes_models::bytes_write_answer(&receiver_value, value) {
            Some(BytesAnswer::Raises(_)) => return,
            Some(BytesAnswer::Value(_)) => {
                // falls through to the ordinary list_with_item path below,
                // which performs the identical index-bounds write — this
                // function only needed to know the write does not raise.
            }
            None => {
                environment.forget(receiver_name.id.as_str());
                return;
            }
        }
    }
    let written = match receiver_value.kind {
        Kind::Object => dict_with_item(&receiver_value, &key_value, value),
        Kind::List => list_with_item(&receiver_value, &key_value, value),
        _ => None,
    };
    match written {
        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
}

/// `del d[k]` / `del lst[lower:]` — the delete-shaped sibling of
/// `bind_or_forget_subscript_target`: only a bare-`Name` receiver is
/// replayed (any other receiver shape has no single environment slot to
/// rebind, and is simply left untouched — the same "no element-level
/// model" posture the write-sibling takes).
///
/// An `Expr::Slice` index tries the ONE slice-deletion shape
/// `collection_models::sliced_delete_receiver` models first: no `upper`,
/// no `step`, `lower` a known Integer read through
/// `expressions::slice_bound_index`. `Some` rebinds `name` to the
/// receiver with everything from `lower` onward removed; a slice this
/// function cannot read that way (an `upper`, a `step`, an unknown
/// `lower`) or a receiver `sliced_delete_receiver` declines FORGETS
/// `name`.
///
/// Any other index shape reads as a plain key/index:
/// `collection_models::dict_without_item` answers the receiver WITHOUT
/// `key`'s entry: `Some` rebinds `name` to it, so a later read sees the
/// key's absence (b-body-expressions.py's `del_expression`: `del
/// person["age"]` then `person.get("age")` must answer the absent-key
/// default, not the stale pre-delete 40); `None` (an unknown receiver, a
/// key this walk cannot read exactly, or a receiver `Kind` the contract
/// does not own — e.g. a `List`, which has no by-value delete this table
/// models) FORGETS `name` — the pre-delete value must not survive an
/// unresolved delete, the same honesty every other decline in this file
/// already keeps.
pub(in crate::check) fn walk_del_subscript_target(subscript: &ExprSubscript, context: &WalkContext, environment: &mut Environment) {
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        return;
    };
    // `del name[lower:]` — a Slice index with a known nonnegative `lower`,
    // no `upper`, no `step` — is the ONE slice-deletion shape this walk
    // reads: `collection_models::sliced_delete_receiver`'s own doc states
    // the exact contract. Any other slice shape (an `upper`, a `step`, an
    // unknown `lower`) falls through to the ordinary forget below, same
    // honesty every other unresolved write in this file keeps.
    if let Expr::Slice(slice) = subscript.slice.as_ref() {
        if slice.upper.is_none() && slice.step.is_none() {
            if let Some(lower_expr) = slice.lower.as_deref() {
                if let Some(lower) = slice_bound_index(lower_expr, environment, context.kernel) {
                    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
                    match sliced_delete_receiver(&receiver_value, lower) {
                        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
                        None => environment.forget(receiver_name.id.as_str()),
                    }
                    return;
                }
            }
        }
        environment.forget(receiver_name.id.as_str());
        return;
    }
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    let key_value = evaluate_expression(subscript.slice.as_ref(), environment, context.kernel);
    let written = dict_without_item(&receiver_value, &key_value);
    match written {
        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
}

/// The leftmost `Name` under an attribute-chain receiver
/// (`a.b.c` → `a`; `a` itself → `a`) — `None` when the receiver is not
/// built from a plain name chain at all (a call's own result, a
/// subscript, …), which this walk has no base name to forget either
/// way.
pub(in crate::check) fn receiver_base_name(receiver: &Expr) -> Option<&str> {
    match receiver {
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Attribute(attribute) => receiver_base_name(attribute.value.as_ref()),
        _ => None,
    }
}
