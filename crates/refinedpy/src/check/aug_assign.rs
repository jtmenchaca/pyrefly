use std::collections::HashMap;

use refined_domain::abstract_value::{known_set, unknown, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::{on_one_tuple_layer, requires_integer, RefinedSet};
use ruff_python_ast::{Expr, ExprAttribute, ExprSubscript, StmtAugAssign};
use ruff_text_size::Ranged;

use crate::assignability::{judge, states_sequence, Verdict};
use crate::bytes_models::{self, BytesAnswer};
use crate::collection_models::{dict_with_item, list_with_item, subscript_read};
use crate::env::Environment;
use crate::expressions::{binary_arithmetic_value_with_kernel, evaluate_expression, possible_raise, provable_raise};
use crate::typereading::DeclaredRefinement;

use super::*;

/// `x op= v` — dispatches on the target's own syntactic shape. A bare
/// name folds `binary_arithmetic_value` (expressions.rs's shared
/// arithmetic transfer — the same one ordinary `x = x op v` rows use, so
/// the two forms agree exactly) over the target's CURRENT value and the
/// evaluated RHS, then judges against `x`'s own recorded refinement
/// (this body's `x: Age = …` AnnAssign, if any) through the shared
/// refused-write law — `Fire` anchors to the WHOLE statement's range
/// (there is no separate "value expression" the way AnnAssign has one;
/// the fired value is the folded result, not a sub-expression of the
/// source). A name with no recorded refinement binds the folded value
/// directly. An `obj.attr op= v` / `name[key] op= v` target composes the
/// identical read-fold-write shape through `walk_field_aug_assign` /
/// `walk_subscript_aug_assign` — see each function's own doc for what it
/// judges and what it can only compose honestly. Any other target shape
/// (a tuple/list/starred aug-target — not valid Python syntax, so this
/// arm is unreachable in practice) stays this body's blocker.
pub(super) fn walk_aug_assign(
    assign: &StmtAugAssign,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    blocked: &mut bool,
    out: &mut Vec<Finding>,
) {
    match assign.target.as_ref() {
        Expr::Name(name) => {
            walk_name_aug_assign(assign, name.id.as_str(), context, environment, aug_assign_refinements, out);
        }
        Expr::Attribute(attribute) => {
            walk_field_aug_assign(assign, attribute, context, environment, out);
        }
        Expr::Subscript(subscript) => {
            walk_subscript_aug_assign(assign, subscript, context, environment, aug_assign_refinements, out);
        }
        _ => {
            record_blocker(
                blocked,
                assign.range(),
                "an augmented assignment to a non-name target is not yet walked".to_owned(),
                out,
            );
        }
    }
}

/// `x op= v` on a plain name: fold the target's current value with the
/// evaluated RHS through `binary_arithmetic_value_with_kernel` (the
/// kernel-computed SET transfer, tried first, falling through to the
/// plain arithmetic dispatcher for everything the SET path does not
/// serve), then judge against `x`'s own recorded declared refinement
/// (`aug_assign_refinements` — populated for a body-local `x: Age = ...`
/// by `walk_ann_assign` AND for a declared-refinement PARAMETER by
/// `seed_parameters`) through `judge_and_bind_aug_assign_write`'s own
/// write-site law — the refused-write binding `judge_and_bind` gives
/// every other sink, but with a Fire message naming the WRITE itself
/// (the pre-write ceiling, the operator expression, the post-write
/// ceiling) rather than only the escaped result.
pub(super) fn walk_name_aug_assign(
    assign: &StmtAugAssign,
    name: &str,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &mut HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    if let Some((range, message)) = provable_raise(assign.value.as_ref(), environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
        // the raise happens before `x op= v` ever folds a value — the
        // target's own current value is untouched by CPython, but this
        // walk has no exception-continuation channel (the same posture
        // `Stmt::Assert`'s doc already states), so the honest answer is
        // to forget rather than assert the pre-raise value still holds
        // past this statement.
        environment.forget(name);
        return;
    }
    // A SOMETIMES-raise (the divisor's set admits 0 among other values)
    // fires its finding and the walk continues with the split value —
    // some runs raise, the rest produce the value, so neither replaces
    // the other (`expressions.rs::possible_raise`'s own claim).
    if let Some((range, message)) = possible_raise(assign.value.as_ref(), environment, context.kernel) {
        out.push(Finding { range, code: "RTS7001", message });
    }
    bind_walrus_targets(assign.value.as_ref(), context, aug_assign_refinements, environment, out);
    let current = environment.read(name).cloned().unwrap_or_else(unknown);
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value_with_kernel(assign.op, &current, &operand, context.kernel);

    match aug_assign_refinements.get(name) {
        // An Undetermined verdict already forgets the name inside
        // judge_and_bind; a bare-name aug-target is not itself a
        // blocker candidate (blockers here are scoped to non-name
        // targets only, handled by the caller), so the sentence is
        // discarded.
        Some(declared) => {
            let declared = declared.clone();
            judge_and_bind_aug_assign_write(name, &current, updated, assign, &declared, context, environment, out);
        }
        None => environment.bind(name, updated),
    }
}

/// The WRITE-SITE check for a bare-name aug-target (`x += 1`): judges
/// `updated` — the kernel-computed fold `binary_arithmetic_value_with_
/// kernel` already produced (E2.operator.py's own row exists to pin real
/// arithmetic transfer defects the plain `binary_arithmetic_value`
/// dispatcher does not reach; this function never re-derives that value,
/// only judges and reports it) — against `declared`, exactly the verdict
/// `judge_and_bind` itself would reach, but with a WRITE-SPECIFIC Fire
/// message naming three things a reader needs at the WRITE, not just the
/// escaped result: what the target may have HELD going in (`x may be
/// 150`), the operator expression that wrote it (`x += 1`), and what that
/// write may have PRODUCED, past the declared window (`may write 151,
/// outside Age's [0, 150]`) — `aug_assign_write_refutation`'s own doc.
/// Silent/Undetermined bind/forget exactly as `judge_and_bind_naming`
/// does; only the Fire arm's message differs.
pub(super) fn judge_and_bind_aug_assign_write(
    name: &str,
    current: &AbstractValue,
    updated: AbstractValue,
    assign: &StmtAugAssign,
    declared: &DeclaredRefinement,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    match judge(&updated, declared, context.kernel) {
        Verdict::Fire(judge_message) => {
            let message =
                aug_assign_write_refutation(name, current, &updated, assign, declared).unwrap_or(judge_message);
            out.push(Finding {
                range: assign.range(),
                code: "RTS7001",
                message,
            });
            // the same refused-slot binding `judge_and_bind_naming`'s own
            // Fire arm leaves onward flow with: the declared set itself,
            // tagged the identical numeric-ground way, so a later read of
            // `name` in this same body is not judged twice for the one
            // refused write.
            let refused_slot = if on_one_tuple_layer(&declared.set) && !states_sequence(&declared.set) {
                let sort = if requires_integer(&declared.set) {
                    PrimitiveKind::Integer
                } else {
                    PrimitiveKind::Float
                };
                AbstractValue {
                    kind_tag: Some(sort),
                    ..known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
                }
            } else {
                known_set(declared.set.clone(), None, TrustSpec, SetKindTag::None)
            };
            environment.bind(name, refused_slot);
        }
        Verdict::Silent => {
            environment.bind(name, updated);
        }
        Verdict::Undetermined(_) => {
            // a bare-name aug-target is not itself a blocker candidate
            // (`walk_name_aug_assign`'s own doc — blockers here are scoped
            // to non-name targets), so the sentence is discarded, matching
            // `judge_and_bind`'s call sites that ignore its `Option<String>`
            // return.
            environment.forget(name);
        }
    }
}

/// The write-site Fire message for `judge_and_bind_aug_assign_write`:
/// "`x` may be `150`; `x += 1` may write `151`, outside `Age`'s `[0,
/// 150]`" — the target's own pre-write ceiling, the operator expression
/// as written, the post-write ceiling the kernel-computed fold reached,
/// and the declared window it escapes. `None` when any of the three
/// windows this composes (`current`'s, `updated`'s, `declared`'s) is not
/// a plain `[lo, hi]` scalar window (`aug_assign_window`'s own doc), or
/// the RHS is not a plain integer literal (`aug_assign_exact_operand_
/// spelling`'s own doc) — the caller falls back to `judge`'s own
/// already-computed Fire message then, a courtesy for a shape this
/// composer does not spell exactly, never a guessed number.
pub(super) fn aug_assign_write_refutation(
    name: &str,
    current: &AbstractValue,
    updated: &AbstractValue,
    assign: &StmtAugAssign,
    declared: &DeclaredRefinement,
) -> Option<String> {
    let current_ceiling = aug_assign_ceiling(current)?;
    let updated_ceiling = aug_assign_ceiling(updated)?;
    let (lo, hi) = aug_assign_window(&declared.set)?;
    let operand_spelling = aug_assign_exact_operand_spelling(assign.value.as_ref())?;
    Some(format!(
        "{name} may be {}; {name} {}= {operand_spelling} may write {}, outside {}'s [{}, {}]",
        format_aug_assign_number(current_ceiling),
        aug_assign_op_spelling(assign.op),
        format_aug_assign_number(updated_ceiling),
        declared.spelling,
        format_aug_assign_number(lo),
        format_aug_assign_number(hi),
    ))
}

/// The highest value `value` could hold: the maximum of an exact
/// `Kind::Values` set, or the TIGHTEST `AtMost`-form bound of a
/// `Kind::Set` window (`aug_assign_window`'s own doc — the same `[lo,
/// hi]` shape this file's `integer_window_minus_edge_literals` already
/// reads `Form::AtMost` off of). A window can carry more than one
/// `AtMost` form — a later narrowing pass pushes a tighter bound onto
/// the SAME window without removing the earlier, now-stale one
/// (`apply_relational_ceiling_fact`'s two-pass transitivity step) — so
/// this reads the MINIMUM over every `AtMost` form, never the first
/// one found. `None` for anything else (`Kind::Unknown`, an unbounded
/// set with no `AtMost` form, an empty exact set) — the caller falls
/// back to the generic refutation wording then.
pub(super) fn aug_assign_ceiling(value: &AbstractValue) -> Option<f64> {
    use refined_sets::refinement_forms::Form;
    match value.kind {
        Kind::Values => value.values.iter().copied().reduce(f64::max),
        Kind::Set => value
            .set
            .forms
            .iter()
            .filter(|form| form.form == Form::AtMost)
            .map(|form| form.a)
            .reduce(f64::min),
        _ => None,
    }
}

/// `set`'s own `[lo, hi]` window, when it carries EXACTLY one `AtLeast`
/// lower form and one `AtMost` upper form (an `Integer`/`MultipleOf` form
/// alongside them, if present, states the sort and never widens the
/// window — the same forms `integer_window_minus_edge_literals` already
/// reads past) — `Age`'s own shape (`Annotated[int, Field(ge=0,
/// le=150)]` compiles to `[at_least(0), at_most(150), integer()]`,
/// `surface.rs::ge_and_le_constructors_compile_the_same_set_field_kwargs_
/// would`'s own pinned form order). `None` for any other shape (a
/// one-sided ray, a union, a `Literal[...]` tuple set, …) — this reader
/// states the common bounded-window case only, never a guess at a shape
/// it cannot spell as `[lo, hi]`.
pub(super) fn aug_assign_window(set: &RefinedSet) -> Option<(f64, f64)> {
    use refined_sets::refinement_forms::Form;
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &set.forms {
        match form.form {
            Form::AtLeast => lo = Some(form.a),
            Form::AtMost => hi = Some(form.a),
            Form::Integer | Form::MultipleOf => {}
            _ => return None,
        }
    }
    lo.zip(hi)
}

/// The RHS's own exact numeric spelling, when `value_expr` is a plain
/// integer literal (`x += 1`'s own `1`) — the one operand shape this
/// write-site message composes; every other RHS shape (a name, a call, a
/// float literal) falls back to the caller's own generic wording rather
/// than guessing a spelling this reader cannot state exactly.
pub(super) fn aug_assign_exact_operand_spelling(value_expr: &Expr) -> Option<String> {
    let Expr::NumberLiteral(literal) = value_expr else {
        return None;
    };
    match &literal.value {
        ruff_python_ast::Number::Int(int) => int.as_i64().map(|value| value.to_string()),
        _ => None,
    }
}

/// `n` spelled as a plain integer when it is a whole number (`150`, never
/// `150.0`) — every window this composer reads (`Age`'s own `[0, 150]`,
/// an aug-assign's own before/after ceiling) is integer-ground in every
/// row this message composes for, matching the marker sentence's own
/// plain-integer spelling.
pub(super) fn format_aug_assign_number(n: f64) -> String {
    if n.fract() == 0.0 { format!("{}", n as i64) } else { format!("{n}") }
}

/// `+=`/`-=`/… — the AugAssign operator's own token spelling
/// (simple_stmts.rst's `augassign` production), reconstructed from
/// `ruff_python_ast::Operator` since this file never carries the raw
/// source text to slice the written token back out of.
pub(super) fn aug_assign_op_spelling(op: ruff_python_ast::Operator) -> &'static str {
    match op {
        ruff_python_ast::Operator::Add => "+",
        ruff_python_ast::Operator::Sub => "-",
        ruff_python_ast::Operator::Mult => "*",
        ruff_python_ast::Operator::MatMult => "@",
        ruff_python_ast::Operator::Div => "/",
        ruff_python_ast::Operator::Mod => "%",
        ruff_python_ast::Operator::Pow => "**",
        ruff_python_ast::Operator::LShift => "<<",
        ruff_python_ast::Operator::RShift => ">>",
        ruff_python_ast::Operator::BitOr => "|",
        ruff_python_ast::Operator::BitXor => "^",
        ruff_python_ast::Operator::BitAnd => "&",
        ruff_python_ast::Operator::FloorDiv => "//",
    }
}

/// `obj.attr op= v` where `obj` is a bare-Name receiver bound to a
/// tagged instance (i-more-expressions.py's `accessor_compound_read_
/// modify_write`: `box.age += 5` through a `@property` getter/setter
/// pair — the same accessor `write_named_field` already judges for a
/// plain `box.age = v`). Composes three EXISTING reads/writes rather
/// than inventing new field-mutation machinery: the CURRENT value reads
/// through the ordinary `evaluate_expression` attribute path (which
/// already resolves a `@property` name to its backing field via
/// `field_read_through_model`), the fold is the identical
/// `binary_arithmetic_value` transfer every other aug-target uses, and
/// the write-back is `write_named_field` — the same judged-and-rebound
/// law a plain `obj.attr = v` write already gets, so `box.age += 5`
/// fires under EXACTLY the same setter-declared refinement a hand-split
/// `box.age = box.age + 5` would.
///
/// A receiver that is not a bare Name, or a bare Name not bound to a
/// tagged instance whose class this environment can find, composes
/// nothing: this function is a no-op in that case (unlike a bare-name
/// aug-target, an attribute aug-target names no single environment slot
/// to forget on decline — the same "no element-level model" posture
/// `bind_or_forget_target`'s own Attribute arm already takes for a
/// plain `obj.attr = v` write to an untagged receiver).
pub(super) fn walk_field_aug_assign(
    assign: &StmtAugAssign,
    attribute: &ExprAttribute,
    context: &WalkContext,
    environment: &mut Environment,
    out: &mut Vec<Finding>,
) {
    let Expr::Name(receiver) = attribute.value.as_ref() else {
        return;
    };
    // `write_named_field` is already generic over the receiver's own
    // environment slot — a method body's `self.age += 5` and a local
    // variable's `box.age += 5` share one judged-and-rebound law under
    // whichever name the receiver actually is, with no separate `self`
    // case needed here.
    let receiver_name = receiver.id.as_str();
    let field = attribute.attr.as_str();
    let current = evaluate_expression(&Expr::Attribute(attribute.clone()), environment, context.kernel);
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value_with_kernel(assign.op, &current, &operand, context.kernel);
    write_named_field(receiver_name, field, &updated, assign.range(), context, environment, out);
}

/// `name[key] op= v` where `name` is a bare-Name receiver bound to a
/// known `Kind::Object`/`Kind::List` (i-more-expressions.py's
/// `compound_array_index_operators`/`list_index_power_compound`:
/// `ages[0] += 190`, `over_ages[0] **= 2`). Composes the identical
/// three-step shape `walk_field_aug_assign` uses: the CURRENT element
/// reads through `collection_models::subscript_read`, the fold is the
/// shared `binary_arithmetic_value` transfer, and the write-back replays
/// through the SAME `dict_with_item`/`list_with_item` pair
/// `bind_or_forget_subscript_target` already uses for a plain
/// `name[key] = v` write — rebinding `name` so a later read in the same
/// straight-line body sees the mutated element.
///
/// ELEMENT-LEVEL JUDGING: when `name`'s own declared refinement
/// (`aug_assign_refinements`, this body's `x: list[Age] = …` table) is
/// element-carrying (`typereading::DeclaredRefinement.element`, read for
/// `list[X]`/`set[X]`/`dict[str, X]`), the folded element value is
/// judged against that inner refinement through the shared `judge`
/// entry point — the same law `judge_and_bind` already applies to a
/// bare-name aug-target, one level down at the element. A Fire pushes
/// the finding and writes the DECLARED element set back into the slot
/// (the refused-write law: the container keeps a fact it can still
/// answer for later reads, never the refused value) rather than
/// `updated`; Silent writes `updated` through unchanged; Undetermined
/// forgets the receiver — this walk has no blocker-sentence channel
/// for a subscript aug-target (`walk_aug_assign`'s own doc already
/// scopes blocker candidates to non-name targets, and an element-level
/// Undetermined is not that either), so the honest answer is silence
/// plus forgetting, the same posture an unresolved element read already
/// takes below. A receiver with no recorded declared refinement, or one
/// that is not element-carrying, composes the write mechanically with
/// no judging — unchanged from before.
///
/// A bytes-like receiver is the one exception: `bytes_models::
/// bytes_write_answer`'s raise is a LANGUAGE fact (CPython itself raises
/// on the write), so it is checked here the same way
/// `bind_or_forget_subscript_target` checks it for a plain `=` write —
/// see that function's own doc for the full three-way rule (a provable
/// raise leaves the receiver untouched and records no finding, success
/// applies through `list_with_item` as today, undecidable forgets).
pub(super) fn walk_subscript_aug_assign(
    assign: &StmtAugAssign,
    subscript: &ExprSubscript,
    context: &WalkContext,
    environment: &mut Environment,
    aug_assign_refinements: &HashMap<String, DeclaredRefinement>,
    out: &mut Vec<Finding>,
) {
    let Expr::Name(receiver_name) = subscript.value.as_ref() else {
        return;
    };
    let receiver_value = evaluate_expression(subscript.value.as_ref(), environment, context.kernel);
    let key_value = evaluate_expression(subscript.slice.as_ref(), environment, context.kernel);
    let Some(current) = subscript_read(&receiver_value, &key_value) else {
        // an unresolved element read (an unknown container, a key this
        // walk cannot read exactly, an out-of-bounds index) states
        // nothing to fold — forgetting the receiver is the same honesty
        // `bind_or_forget_subscript_target` already keeps for a decline.
        environment.forget(receiver_name.id.as_str());
        return;
    };
    let operand = evaluate_expression(assign.value.as_ref(), environment, context.kernel);
    let updated = binary_arithmetic_value_with_kernel(assign.op, &current, &operand, context.kernel);
    if receiver_value.kind == Kind::List && receiver_value.kind_word.is_some() {
        match bytes_models::bytes_write_answer(&receiver_value, &updated) {
            Some(BytesAnswer::Raises(_)) => return,
            Some(BytesAnswer::Value(_)) => {}
            None => {
                environment.forget(receiver_name.id.as_str());
                return;
            }
        }
    }
    let element_declared = aug_assign_refinements
        .get(receiver_name.id.as_str())
        .and_then(|declared| declared.element.as_deref());
    let updated = if let Some(element_declared) = element_declared {
        match judge(&updated, element_declared, context.kernel) {
            Verdict::Fire(message) => {
                out.push(Finding {
                    range: assign.range(),
                    code: "RTS7001",
                    message,
                });
                // Tags the numeric sort onward flow needs (the same
                // guarded rule `seed_parameters` applies to a declared
                // set: numeric-ground only, never the
                // `Literal["A", "B"]` string-tuple pun
                // `on_one_tuple_layer` alone would also admit).
                if on_one_tuple_layer(&element_declared.set) && !states_sequence(&element_declared.set) {
                    let sort = if requires_integer(&element_declared.set) {
                        PrimitiveKind::Integer
                    } else {
                        PrimitiveKind::Float
                    };
                    AbstractValue {
                        kind_tag: Some(sort),
                        ..known_set(element_declared.set.clone(), None, TrustSpec, SetKindTag::None)
                    }
                } else {
                    known_set(element_declared.set.clone(), None, TrustSpec, SetKindTag::None)
                }
            }
            Verdict::Silent => updated,
            Verdict::Undetermined(_) => {
                environment.forget(receiver_name.id.as_str());
                return;
            }
        }
    } else {
        updated
    };
    let written = match receiver_value.kind {
        Kind::Object => dict_with_item(&receiver_value, &key_value, &updated),
        Kind::List => list_with_item(&receiver_value, &key_value, &updated),
        _ => None,
    };
    match written {
        Some(new_receiver) => environment.bind(receiver_name.id.as_str(), new_receiver),
        None => environment.forget(receiver_name.id.as_str()),
    }
}
