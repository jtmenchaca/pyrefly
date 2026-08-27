use std::collections::HashMap;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::repetition_window_forms::as_repetition;
use refined_sets::repetition_window_forms::repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::ExprSubscript;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::assignability::judge;
use crate::assignability::Verdict;
use crate::bytes_models;
use crate::bytes_models::BytesAnswer;
use crate::collection_models::dict_with_item;
use crate::collection_models::dict_without_item;
use crate::collection_models::list_literal_value;
use crate::collection_models::list_with_item;
use crate::collection_models::sliced_delete_receiver;
use crate::collection_models::subscript_read;
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
            bind_or_forget_subscript_target(
                subscript,
                value,
                value_range,
                context,
                aug_assign_refinements,
                environment,
                out,
            );
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
/// target lists the same way). A value that is not a known list is
/// offered to `bind_window_sequence_target`, which states the weaker
/// per-element claim an unknown-length repetition window supports; only
/// when THAT declines too does this return `false` (no binding performed,
/// caller falls back to forgetting every name) — an RHS with neither
/// exact items nor a known element set states nothing about what any
/// target receives, and the forget-all answer is the sound one.
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
        return bind_window_sequence_target(elements, value, context, aug_assign_refinements, environment, out);
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

/// `a, b, *rest = xs` where `xs` is an UNKNOWN-LENGTH sequence known by
/// its element set — the repetition window a declared `list[X]`/`set[X]`/
/// `Sequence[X]` parameter seeds (`check::seed::seed_parameters`).
///
/// The concrete path above states exact positional slots and is right to
/// decline: WHICH values sit at which position is unknown, and the length
/// may not even be enough for the targets, in which case CPython raises
/// `ValueError` (simple_stmts.rst, "Assignment statements": the sequence
/// "must have the same number of items as there are targets"). A raise is
/// not a fire this walk reports — the value never flows past it — so what
/// remains to state is what holds on the runs that DO complete, and on
/// those runs every non-starred target draws from the window's own
/// element set (every position of a repetition draws from the same
/// element, the grammar's own definition) and the starred target holds a
/// sequence of those same elements.
///
/// The starred target's own window is `[max(0, lo - n), hi - n]` where
/// `n` is the count of non-starred targets: those `n` positions are
/// consumed off the source, so the remainder is shorter by exactly `n`,
/// and a source that could be as short as `lo` leaves as few as zero.
///
/// `false` (the caller forgets every target name) when the value is not a
/// repetition window at all — the honest decline for a shape with no
/// element to bind.
fn bind_window_sequence_target(
    elements: &[Expr],
    value: &AbstractValue,
    context: &WalkContext,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) -> bool {
    if value.kind != Kind::Set || value.set_kind_tag != SetKindTag::None {
        return false;
    }
    let Some(window) = as_repetition(&value.set) else {
        return false;
    };
    let element = AbstractValue {
        kind_tag: value.kind_tag,
        ..known_set(window.element.clone(), None, TrustSpec, SetKindTag::None)
    };
    let plain_count = elements
        .iter()
        .filter(|target| !matches!(target, Expr::Starred(_)))
        .count() as i64;
    for target in elements {
        match target {
            Expr::Starred(starred) => {
                let Expr::Name(name) = starred.value.as_ref() else {
                    forget_target_names(target, environment);
                    continue;
                };
                let low = (window.lo - plain_count).max(0);
                let high = window.hi.map(|hi| (hi - plain_count).max(low));
                let rest = AbstractValue {
                    kind_tag: value.kind_tag,
                    ..known_set(repetition(window.element.clone(), low, high), None, TrustSpec, SetKindTag::None)
                };
                environment.bind(name.id.as_str(), rest);
            }
            _ => bind_sequence_element(target, &element, context, aug_assign_refinements, environment, out),
        }
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
/// (b)'s full contract.
///
/// A CHAINED subscript receiver (`name[i][j] = v`) is written through
/// `write_chained_subscript` first: the write lands on the INNER
/// container, which is one shared object every other name holding it
/// still holds (library/copy.rst's shallow-copy sharing), so the rebuild
/// walks back out to `name`'s own slot and then sweeps every other
/// binding holding that same inner object. A receiver shape with no
/// single environment slot to rebind at all (`obj.attr[key] = v`, a
/// chain whose base is not a bare Name, a chain this walk cannot read an
/// index of) FORGETS the chain's own base name — the pre-write value
/// must not survive a write this walk could not replay, the same honesty
/// every other decline in this file keeps.
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
///
/// THE VALUE SINK: a write onto a receiver whose own DECLARATION states
/// a member refinement (`d: dict[str, Age]`, whose `DeclaredRefinement.
/// element` carries `Age`'s window) judges the written value against
/// THAT refinement, at the value's own range, before the replay records
/// anything — the same law `check::calls::mutation`'s element sink
/// gives an `append`/`extend`. Without it, `d["x"] = 200` merely
/// RECORDED an out-of-window entry, and the defect surfaced at whatever
/// later sink read `d` (`needs_age_dict(d)`), reporting the one defect
/// at a position that is not where it was introduced.
///
/// A FIRED member also stops the recording: the program is being told
/// to fix the write, so the receiver keeps what its declaration says
/// about its members rather than carrying the refused value to the next
/// sink, where it would report the same one defect a second time. This
/// is the identical refused-write law `adapter_alias_verdict` keeps for
/// a refused parse. An ADMITTED write still records its entry through
/// the ordinary replay below, unchanged: the written value is a real
/// fact about that key, and later rows read it back exactly
/// (A8.xfer.set's `replace_widened_value` writes 200 into a
/// `dict[str, int]` — inside `int`, so no fire — and its own later
/// `d["a"]` read must still answer 200).
pub(in crate::check) fn bind_or_forget_subscript_target(
    subscript: &ExprSubscript,
    value: &AbstractValue,
    value_range: TextRange,
    context: &WalkContext,
    declared_refinements: &HashMap<String, DeclaredRefinement>,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    if let Expr::Name(receiver_name) = subscript.value.as_ref() {
        // THE VALUE SINK (this function's own doc), read before the
        // replay so the finding lands at the write rather than at a
        // later read of the receiver.
        if let Some(member) = declared_refinements
            .get(receiver_name.id.as_str())
            .and_then(|declared| declared.element.as_deref())
        {
            if let Verdict::Fire(message) = judge(value, member, context.kernel) {
                out.push(Finding {
                    range: value_range,
                    code: "RTS7001",
                    message,
                });
                // THE REFUSED-WRITE LAW: the refusal is reported here,
                // at the write, so the receiver must not carry the
                // refused value to a later sink that would report the
                // same defect again. The declaration already states
                // what this receiver's members hold, and that is what
                // the binding keeps.
                return;
            }
        }
    }
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        if let Some(base_name) = subscript_chain_base_name(subscript.value.as_ref()) {
            if !write_chained_subscript(subscript, value, context, environment) {
                environment.forget(base_name);
            }
        }
        return;
    };
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    // `name[lower:upper] = value` — SLICE ASSIGNMENT, which replaces a
    // whole run of positions rather than one (stdtypes.rst's
    // Mutable-Sequence-Types table, `s[i:j] = t`: "slice of *s* from *i*
    // to *j* is replaced by the contents of the iterable *t*"). Read
    // through `sliced_write_receiver`, which owns the bound reading;
    // a slice shape or receiver it declines forgets the name, the same
    // honesty every other unresolved write here keeps.
    // A bytes-like receiver (`bytes_models::tagged`'s own species word)
    // is NOT this arm's: its own three write rules — including the
    // immutability of `bytes` itself — are read by the tagged path
    // below, which this arm must not step in front of.
    if receiver_value.kind_word.is_none() {
        if let Expr::Slice(slice) = subscript.slice.as_ref() {
            match sliced_write_receiver(&receiver_value, slice, value, environment, context) {
                Some(written) => environment.bind(receiver_name.id.as_str(), written),
                None => environment.forget(receiver_name.id.as_str()),
            }
            return;
        }
    }
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
        // `Kind::ObjectStar` — a `dict[str, X]` parameter's own unbounded-key
        // seed — writes through the same `dict_with_item` contract, which
        // records the written key's own entry beside the star's law for
        // every other key (that function's own dict-star arm states why).
        Kind::Object | Kind::ObjectStar => dict_with_item(&receiver_value, &key_value, value),
        Kind::List => list_with_item(&receiver_value, &key_value, value),
        _ => None,
    };
    match written {
        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
}

/// `name[lower:upper] = value` on a KNOWN `Kind::List` receiver —
/// stdtypes.rst's Mutable-Sequence-Types table, `s[i:j] = t`: "slice of
/// *s* from *i* to *j* is replaced by the contents of the iterable *t*".
/// The written-through list is the receiver's head up to `lower`, then
/// every item of `value`, then the receiver's tail from `upper` on.
/// Unlike a single-index write, the result's LENGTH changes whenever the
/// replacement's own item count differs from the replaced run's — the
/// clause states a replacement of contents, not of positions.
///
/// The bounds read by the same rules a slice READ takes
/// (`expressions::slice_bound_index` for each stated bound, `evaluate_
/// slice`'s own defaults for an omitted one — `lower` 0, `upper` the
/// receiver's length — and the same clamp to `[0, len]`, since a slice
/// never raises for an out-of-range bound). A `step` is not modeled and
/// declines, matching the read side.
///
/// `None` — the caller forgets the name — for a receiver that is not a
/// known list, a replacement that is not a known list, a `step`, or a
/// bound this walk cannot read exactly.
fn sliced_write_receiver(
    receiver: &AbstractValue,
    slice: &ruff_python_ast::ExprSlice,
    value: &AbstractValue,
    environment: &Environment,
    context: &WalkContext,
) -> Option<AbstractValue> {
    if slice.step.is_some() || receiver.kind != Kind::List || value.kind != Kind::List {
        return None;
    }
    let length = receiver.items.len() as i64;
    let lower = match slice.lower.as_deref() {
        Some(expr) => slice_bound_index(expr, environment, context.kernel)?,
        None => 0,
    };
    let upper = match slice.upper.as_deref() {
        Some(expr) => slice_bound_index(expr, environment, context.kernel)?,
        None => length,
    };
    let clamp = |bound: i64| {
        let adjusted = if bound < 0 { bound + length } else { bound };
        adjusted.clamp(0, length) as usize
    };
    let start = clamp(lower);
    let end = clamp(upper).max(start);
    let mut items = receiver.items[..start].to_vec();
    items.extend(value.items.iter().cloned());
    items.extend(receiver.items[end..].iter().cloned());
    let mut written = list_literal_value(&items);
    written.kind_word = receiver.kind_word;
    written.instance_identity = receiver.instance_identity;
    Some(written)
}

/// The bare `Name` a subscript chain is rooted at (`a[i][j]` → `a`;
/// `a[i]` → `a`; `a` → `a`) — `None` for a chain rooted at anything else
/// (a call's result, an attribute read), which has no environment slot
/// this walk could rebind or forget.
fn subscript_chain_base_name(receiver: &Expr) -> Option<&str> {
    match receiver {
        Expr::Name(name) => Some(name.id.as_str()),
        Expr::Subscript(subscript) => subscript_chain_base_name(subscript.value.as_ref()),
        _ => None,
    }
}

/// `name[i][j] = value` and deeper — the CHAINED subscript write.
///
/// The write mutates the INNERMOST container the chain names, and that
/// container is one object other names may hold a reference to: `outer =
/// [[1, 2]]; copy = outer[:]` makes `copy[0]` and `outer[0]` the same
/// inner list (library/copy.rst, `copy.copy`: a shallow copy "inserts
/// *references* into it to the objects found in the original"), so
/// `copy[0][0] = 200` is observable at `outer[0][0]`. This domain
/// records that sharing as a referent identity minted on every container
/// element of a display (`collection_models::with_referent_identities`)
/// and carried along by the item clone a slice performs.
///
/// So the write runs in two parts. First the rebuild: read the base
/// name's own value, walk DOWN the chain's indices collecting each level
/// receiver, apply the write at the bottom, and re-apply
/// `list_with_item`/`dict_with_item` on the way back UP so the base
/// name's slot holds the whole written-through structure. Then the
/// sweep: the innermost written container carries its own referent
/// identity forward (`list_with_item`'s own doc — an item assignment
/// mutates in place, so the object is the same one afterward), and
/// `Environment::rebind_referents_of_item` replaces that object wherever
/// any OTHER binding holds it.
///
/// `false` — nothing bound, the caller forgets the base name — whenever
/// any level of the walk is unreadable: an unbound base, an index this
/// walk cannot read exactly, an intermediate value that is not a known
/// container, or a write the container's own contract declines (an
/// out-of-bounds index, `list_with_item`'s own decline).
fn write_chained_subscript(
    subscript: &ExprSubscript,
    value: &AbstractValue,
    context: &WalkContext,
    environment: &mut Environment,
) -> bool {
    // The chain's levels, outermost first: `a[i][j]` collects the index
    // expressions `i` then `j`, with `a` as the base.
    let mut indices: Vec<&Expr> = vec![subscript.slice.as_ref()];
    let mut walker = subscript.value.as_ref();
    let base_name = loop {
        match walker {
            Expr::Name(name) => break name.id.as_str(),
            Expr::Subscript(inner) => {
                indices.push(inner.slice.as_ref());
                walker = inner.value.as_ref();
            }
            _ => return false,
        }
    };
    indices.reverse();
    let Some(base) = environment.read(base_name).cloned() else {
        return false;
    };
    // Down the chain: every level's receiver, base first, so the write
    // below can rebuild them bottom-up.
    let mut receivers: Vec<AbstractValue> = vec![base];
    let mut keys: Vec<AbstractValue> = Vec::with_capacity(indices.len());
    for index in &indices {
        let key = evaluate_expression(index, environment, context.kernel);
        let next = {
            let receiver = receivers.last().expect("seeded with the base above");
            read_container_item(receiver, &key)
        };
        let Some(next) = next else {
            return false;
        };
        keys.push(key);
        receivers.push(next);
    }
    // The deepest receiver is the container the write lands in; the value
    // read out of it (`receivers`' own last entry) is what gets replaced.
    receivers.pop();
    let mut written = value.clone();
    let mut rebuilt: Vec<AbstractValue> = Vec::with_capacity(indices.len());
    while let (Some(receiver), Some(key)) = (receivers.pop(), keys.pop()) {
        let Some(updated) = container_with_item(&receiver, &key, &written) else {
            return false;
        };
        rebuilt.push(updated.clone());
        written = updated;
    }
    // Every level the write passed through is the SAME object it was
    // before (an item assignment mutates in place), so every level whose
    // identity another binding also holds is brought back in step — not
    // just the innermost one, since a chain three deep shares each of its
    // interior containers independently.
    for level in &rebuilt {
        if let Some(identity) = level.instance_identity {
            environment.rebind_referents_of_item(identity, level);
        }
    }
    environment.bind(base_name, written);
    true
}

/// One level of a subscript chain, read: the item `receiver[key]` holds,
/// for the two container shapes this walk's write path rebuilds through.
/// `None` for any other receiver kind or a key this walk cannot read
/// exactly — the caller's own decline.
fn read_container_item(receiver: &AbstractValue, key: &AbstractValue) -> Option<AbstractValue> {
    match receiver.kind {
        Kind::List | Kind::Object | Kind::ObjectStar => subscript_read(receiver, key),
        _ => None,
    }
}

/// One level of a subscript chain, written: the same two container
/// contracts `bind_or_forget_subscript_target`'s own bare-Name path
/// dispatches between, applied at an interior level of the chain.
fn container_with_item(receiver: &AbstractValue, key: &AbstractValue, value: &AbstractValue) -> Option<AbstractValue> {
    match receiver.kind {
        Kind::Object | Kind::ObjectStar => dict_with_item(receiver, key, value),
        Kind::List => list_with_item(receiver, key, value),
        _ => None,
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
///
/// THE PROVABLY-RAISING DELETE is the one decline that does NOT forget:
/// a fully-known dict and a known key that is provably ABSENT means
/// CPython raises `KeyError` and the delete never takes effect
/// (`dict_without_item`'s own doc for why it answers `None` there), so
/// the pre-delete contents are still exactly right. Forgetting would be
/// a strictly weaker, wrong answer — the receiver's value is fully
/// known, and every read past the `del` would go undetermined for a
/// statement that changed nothing. This is the same law the write
/// sibling already keeps for a provably-raising bytes write ("leaves
/// the receiver COMPLETELY UNTOUCHED"), and A8.xfer.delete's
/// `read_widened_after_delete` is the row that needs it: `del d["z"]`
/// on `{"a": 200}` raises and is skipped, and `d["a"]` still answers
/// 200. The raise itself is `provable_raise`'s own row to report, never
/// this function's.
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
        // THE PROVABLY-RAISING DELETE (this function's own doc): the
        // delete never took effect, so the receiver keeps exactly what
        // it held. Every other decline still forgets.
        None if delete_provably_raises(&receiver_value, &key_value) => {}
        None => environment.forget(receiver_name.id.as_str()),
    }
}

/// Whether `del receiver[key]` provably raises `KeyError` — a fully
/// known `Kind::Object` dict (a CLOSED key set) and a key this domain
/// reads exactly, which that key set does not carry. Both halves must
/// be known: a star receiver states nothing about which keys are
/// present, and a key with no exact spelling names no entry to prove
/// absent, so neither is ever read as a proof of a raise.
fn delete_provably_raises(receiver: &AbstractValue, key: &AbstractValue) -> bool {
    if receiver.kind != Kind::Object {
        return false;
    }
    let Some(key) = crate::collection_models::known_dict_key(key) else {
        return false;
    };
    !receiver.keys.iter().any(|entry| entry.name == key.name && entry.numeric == key.numeric)
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
