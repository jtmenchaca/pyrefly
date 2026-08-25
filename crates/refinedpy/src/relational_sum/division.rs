//! Folding a division of the accumulated total by the sequence's own
//! length into the same lowered program, and finding that division
//! inside an arbitrary expression (the return-position spelling).

use refined_kernel::loop_questions::IrStatement;
use refined_kernel::loop_questions::IrStatementKind;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use ruff_python_ast::Expr;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::env::Environment;

use super::slot;
use super::RecognizedAccumulation;
use super::COUNT_SLOT;
use super::QUOTIENT_SLOT;
use super::TOTAL_SLOT;

/// Folds a division of the accumulated total by the sequence's own
/// length into the SAME lowered program, as the statement after the
/// accumulation. This is what makes the relation pay: the kernel's
/// linear decider narrows a division whose numerator it tied to its
/// denominator one statement earlier, where a separate question would
/// see only two unrelated enclosures.
///
/// Recognizes `<total> / len(<sequence>)` OR `<total> // len(<sequence>)`
/// for exactly the accumulator and sequence this accumulation named — OR
/// a sequence a comprehension built 1:1 over it with no filter
/// (`AbstractValue::same_length_as`, `is_len_of`'s own doc) — or a bare
/// name the caller already recorded in `recognized.length_aliases` as
/// equal to `len(<sequence>)` by a plain `count = len(samples)`
/// assignment (`is_length_alias_assignment`, `record_length_alias`).
/// `false` — leaving the program as the accumulation alone — for any
/// other shape: a different, unlinked name on either side, a length
/// taken of some other sequence, an operator that is neither true nor
/// floor division.
pub fn fold_division(
    recognized: &mut RecognizedAccumulation,
    expression: &Expr,
    environment: &Environment,
) -> bool {
    let Some(op) = relational_division_op(expression, recognized, environment) else {
        return false;
    };
    recognized.statements.push(division_statement(op));
    recognized.quotient_op = Some(op);
    true
}

/// Which of Python's two division operators a folded division carries —
/// `/` (true division, always Float) or `//` (floor division, Integer
/// when both operands are Integer-sorted, Float otherwise) —
/// `division_statement` reads this to choose the kernel effect, and
/// `walk_accumulation` reads it to choose the quotient's own sort.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionOp {
    Div,
    FloorDiv,
}

/// The division assignment both folding routes push: the total slot
/// over the count slot, into the quotient's own slot. The quotient
/// rides its own slot because both names survive these two statements
/// in Python, so both exit states are read back and the total is never
/// left holding a value it never had.
///
/// `//` wraps the same `Div` effect in the kernel's existing `Floor`
/// unary (`binary64.floor`): Python's floor division "is always rounded
/// towards minus infinity" (expressions.rst, binary arithmetic), and for
/// this module's own domain — `total` a sum of same-signed elements,
/// `count = len(sequence) >= 0` — `total` and `count` never straddle a
/// sign the way a general `//` would need to worry about, so plain
/// `floor(total / count)` is exact; no separate floor-division kernel op
/// is needed.
fn division_statement(op: DivisionOp) -> IrStatement {
    let divide = LoopEffect {
        kind: LoopEffectKind::Binary,
        op: LoopEffectOp::Div,
        a: Some(Box::new(slot(TOTAL_SLOT))),
        b: Some(Box::new(slot(COUNT_SLOT))),
        ..Default::default()
    };
    let effect = match op {
        DivisionOp::Div => divide,
        DivisionOp::FloorDiv => LoopEffect {
            kind: LoopEffectKind::Unary,
            op: LoopEffectOp::Floor,
            a: Some(Box::new(divide)),
            ..Default::default()
        },
    };
    IrStatement {
        kind: IrStatementKind::Assign,
        target: QUOTIENT_SLOT,
        effect,
        ..Default::default()
    }
}

/// Folds the division into the program without re-matching the node —
/// the caller already located it with `division_range_in`, which
/// matched the same shape `fold_division` checks and returns the
/// operator it matched alongside the range. Used by the return form,
/// where the division sits nested inside the returned expression rather
/// than being that expression.
pub fn fold_located_division(recognized: &mut RecognizedAccumulation, op: DivisionOp) {
    recognized.statements.push(division_statement(op));
    recognized.quotient_op = Some(op);
}

/// Whether a statement is exactly `<name> = len(<sequence>)` for THIS
/// accumulation's own sequence — the plain assignment that binds a
/// count alias the way `count = len(samples)` does, one hop before a
/// later `total / count`. The caller is responsible for the guard this
/// function does not itself see: the assignment must sit in the SAME
/// body, immediately (or with only intervening statements the caller
/// has itself checked do not reassign `<name>` or the sequence), between
/// the accumulation and the division — this function answers only
/// whether the ONE statement handed to it has the count-alias shape,
/// never whether it is safe to trust across the statements around it.
///
/// `None` — not this shape — when: the statement is not a single-name
/// assignment; the value is not a call to a name `len` no local binding
/// shadows; the call carries a keyword argument or an argument count
/// other than one; or the argument is not a plain name equal to this
/// accumulation's own `sequence_name`. Returns the bound name (`<name>`)
/// on a match, for the caller to pass to `record_length_alias`.
pub fn is_length_alias_assignment(
    assign: &StmtAssign,
    recognized: &RecognizedAccumulation,
    environment: &Environment,
) -> Option<String> {
    let [Expr::Name(target)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    if callee.id.as_str() != "len" || environment.read(callee.id.as_str()).is_some() {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let [Expr::Name(argument)] = call.arguments.args.as_ref() else {
        return None;
    };
    if argument.id.as_str() != recognized.sequence_name {
        return None;
    }
    Some(target.id.as_str().to_owned())
}

/// Records that `alias` is proved equal to `len(<sequence_name>)` — the
/// one writer of `RecognizedAccumulation::length_aliases`, called by the
/// caller after `is_length_alias_assignment` matched AND the caller's
/// own reassignment guard held over every statement between the
/// assignment and the division.
pub fn record_length_alias(recognized: &mut RecognizedAccumulation, alias: String) {
    let sequence_name = recognized.sequence_name.clone();
    recognized.length_aliases.insert(alias, sequence_name);
}

/// Whether a statement WRITES either `alias` or `recognized.sequence_name`
/// — the reassignment guard `is_length_alias_assignment`'s own doc
/// requires the caller to hold over every statement between the count
/// alias's own assignment and the division that reads it. No general
/// rebind detector exists for this shape (`rebinds_relational_name` in
/// `check.rs` is walrus-only, and this guard is over a PLAIN statement,
/// not an expression), so this is the smallest check sufficient for the
/// one hop the count-alias fold needs: an assignment target, an
/// augmented-assignment target, or a `for` loop target naming either
/// watched name — the shapes an ordinary statement uses to rebind a
/// plain name. A `del` of either name, a nonlocal/global rebinding, or
/// any other statement kind this checker does not itself special-case
/// for rebinding a bare name is deliberately NOT read here: the caller's
/// own one-hop scan only ever advances past statements it already
/// trusts not to touch either name, and a statement kind so unusual it
/// falls outside this list should stop the scan by not being tried
/// against this guard, never pass through it silently.
pub fn reassigns_alias_or_sequence(stmt: &Stmt, alias: &str, sequence_name: &str) -> bool {
    fn names_target(target: &Expr, watched: &[&str]) -> bool {
        match target {
            Expr::Name(name) => watched.contains(&name.id.as_str()),
            Expr::Tuple(tuple) => tuple.elts.iter().any(|element| names_target(element, watched)),
            Expr::List(list) => list.elts.iter().any(|element| names_target(element, watched)),
            Expr::Starred(starred) => names_target(starred.value.as_ref(), watched),
            _ => false,
        }
    }
    let watched = [alias, sequence_name];
    match stmt {
        Stmt::Assign(assign) => assign.targets.iter().any(|target| names_target(target, &watched)),
        Stmt::AugAssign(assign) => names_target(assign.target.as_ref(), &watched),
        Stmt::AnnAssign(assign) => names_target(assign.target.as_ref(), &watched),
        Stmt::For(for_stmt) => names_target(for_stmt.target.as_ref(), &watched),
        _ => false,
    }
}

/// The range and operator of the ONE `<total> / len(<sequence>)` or
/// `<total> // len(<sequence>)` division inside `expression`, at any
/// depth — the shape a `return` wraps in the fixture (`return
/// math.sqrt(total / len(samples))`, `audio_level.py:25`), where the
/// division is a call argument rather than the returned expression
/// itself.
///
/// `None` unless the count is EXACTLY one. Zero means there is nothing
/// to fold. Two or more means the caller would have to say which
/// occurrence its single published answer belongs to, and both would
/// read the same value — so the honest move is to fold neither and let
/// the ordinary walk evaluate them all.
pub fn division_range_in(
    expression: &Expr,
    recognized: &RecognizedAccumulation,
    environment: &Environment,
) -> Option<(TextRange, DivisionOp)> {
    let mut found: Option<(TextRange, DivisionOp)> = None;
    let mut count = 0;
    find_divisions(expression, recognized, environment, &mut found, &mut count);
    match count {
        1 => found,
        _ => None,
    }
}

/// Walks every subexpression, recording each `<total> / len(<seq>)` or
/// `<total> // len(<seq>)` it meets, alongside which operator it was.
/// Counts past one so the caller can tell "exactly one" from "more than
/// one"; the walk never stops early, because that distinction is the
/// whole point.
///
/// The recursion mirrors `check.rs`'s own `collect_walrus_names`,
/// including its scope rule: a `lambda`'s body is a separate scope
/// whose own `total` is a different binding, so no division inside one
/// is this accumulation's.
fn find_divisions(
    expression: &Expr,
    recognized: &RecognizedAccumulation,
    environment: &Environment,
    found: &mut Option<(TextRange, DivisionOp)>,
    count: &mut usize,
) {
    if let Some(op) = relational_division_op(expression, recognized, environment) {
        *found = Some((expression.range(), op));
        *count += 1;
        // the operands are a name and a `len` call: neither can hold a
        // second occurrence, so the walk stops here
        return;
    }
    match expression {
        Expr::Named(named) => find_divisions(named.value.as_ref(), recognized, environment, found, count),
        Expr::BoolOp(op) => {
            for value in &op.values {
                find_divisions(value, recognized, environment, found, count);
            }
        }
        Expr::BinOp(op) => {
            find_divisions(op.left.as_ref(), recognized, environment, found, count);
            find_divisions(op.right.as_ref(), recognized, environment, found, count);
        }
        Expr::UnaryOp(op) => find_divisions(op.operand.as_ref(), recognized, environment, found, count),
        Expr::Lambda(_) => {}
        Expr::If(ternary) => {
            find_divisions(ternary.test.as_ref(), recognized, environment, found, count);
            find_divisions(ternary.body.as_ref(), recognized, environment, found, count);
            find_divisions(ternary.orelse.as_ref(), recognized, environment, found, count);
        }
        Expr::Tuple(tuple) => {
            for element in &tuple.elts {
                find_divisions(element, recognized, environment, found, count);
            }
        }
        Expr::List(list) => {
            for element in &list.elts {
                find_divisions(element, recognized, environment, found, count);
            }
        }
        Expr::Set(set) => {
            for element in &set.elts {
                find_divisions(element, recognized, environment, found, count);
            }
        }
        Expr::Dict(dict) => {
            for item in &dict.items {
                if let Some(key) = item.key.as_ref() {
                    find_divisions(key, recognized, environment, found, count);
                }
                find_divisions(&item.value, recognized, environment, found, count);
            }
        }
        Expr::Call(call) => {
            find_divisions(call.func.as_ref(), recognized, environment, found, count);
            for argument in &call.arguments.args {
                find_divisions(argument, recognized, environment, found, count);
            }
            for keyword in &call.arguments.keywords {
                find_divisions(&keyword.value, recognized, environment, found, count);
            }
        }
        Expr::Compare(compare) => {
            find_divisions(compare.left.as_ref(), recognized, environment, found, count);
            for comparator in &compare.comparators {
                find_divisions(comparator, recognized, environment, found, count);
            }
        }
        Expr::Attribute(attribute) => {
            find_divisions(attribute.value.as_ref(), recognized, environment, found, count)
        }
        Expr::Subscript(subscript) => {
            find_divisions(subscript.value.as_ref(), recognized, environment, found, count);
            find_divisions(subscript.slice.as_ref(), recognized, environment, found, count);
        }
        Expr::Starred(starred) => find_divisions(starred.value.as_ref(), recognized, environment, found, count),
        Expr::Slice(slice) => {
            for part in [slice.lower.as_deref(), slice.upper.as_deref(), slice.step.as_deref()] {
                if let Some(part) = part {
                    find_divisions(part, recognized, environment, found, count);
                }
            }
        }
        Expr::Await(inner) => find_divisions(inner.value.as_ref(), recognized, environment, found, count),
        Expr::Yield(inner) => {
            if let Some(value) = inner.value.as_deref() {
                find_divisions(value, recognized, environment, found, count);
            }
        }
        Expr::YieldFrom(inner) => find_divisions(inner.value.as_ref(), recognized, environment, found, count),
        // Leaves hold no subexpression. A comprehension
        // (ListComp/SetComp/DictComp/Generator) runs its body an
        // unstated number of times, so a division inside one cannot be
        // shown to evaluate exactly once and is left unwalked rather
        // than folded — the same depth `collect_walrus_names` declines
        // to walk, for a related reason.
        _ => {}
    }
}

/// The division operator an expression carries, when the expression is
/// exactly `<total> / len(<sequence>)` or `<total> // len(<sequence>)`
/// for the accumulator and sequence this accumulation named — or `len`
/// of a DIFFERENT name whose value the environment holds with
/// `same_length_as` proved equal to the accumulation's own sequence, or
/// a bare name proved by a plain assignment to equal `len(<sequence>)`
/// (`is_len_of`'s own doc, both links). `None` for any other operator,
/// including every operator besides `/` and `//`.
fn relational_division_op(
    expression: &Expr,
    recognized: &RecognizedAccumulation,
    environment: &Environment,
) -> Option<DivisionOp> {
    let Expr::BinOp(binop) = expression else {
        return None;
    };
    let op = match binop.op {
        Operator::Div => DivisionOp::Div,
        Operator::FloorDiv => DivisionOp::FloorDiv,
        _ => return None,
    };
    let Expr::Name(numerator) = binop.left.as_ref() else {
        return None;
    };
    if numerator.id.as_str() == recognized.total_name && is_len_of(binop.right.as_ref(), recognized, environment) {
        Some(op)
    } else {
        None
    }
}

/// Whether an expression is `len(<name>)` for exactly this sequence —
/// OR `len(<other name>)` where `<other name>`'s own value carries
/// `AbstractValue::same_length_as == Some(sequence_name)`: a name a
/// comprehension built by mapping every position of `sequence_name` 1:1
/// with no filter, which proves `len(<other name>) == len(sequence_name)`
/// exactly (`comprehension_star_elements`'s own soundness-line comment,
/// expressions.rs — the same fact stated there as a window bound, here
/// read back as a name link) — OR a bare `Expr::Name` recorded in
/// `recognized.length_aliases` as equal to `len(sequence_name)` by a
/// plain `<name> = len(<sequence>)` assignment the caller's own one-hop
/// scan found (`is_length_alias_assignment`, `record_length_alias`): the
/// COUNT-ALIAS shape (`count = len(samples)`, then `total / count`), a
/// different link from the comprehension one above — the aliased name
/// there is not itself a `len(...)` call, it IS the count. A name with
/// no recorded link either way, or one linked to some THIRD sequence,
/// still declines: only an exact proof of equal length licenses folding
/// the division into this program.
fn is_len_of(expression: &Expr, recognized: &RecognizedAccumulation, environment: &Environment) -> bool {
    let sequence_name = recognized.sequence_name.as_str();
    if let Expr::Name(bare) = expression {
        return recognized.length_aliases.get(bare.id.as_str()).map(String::as_str) == Some(sequence_name);
    }
    let Expr::Call(call) = expression else {
        return false;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return false;
    };
    if callee.id.as_str() != "len" || !call.arguments.keywords.is_empty() {
        return false;
    }
    let [Expr::Name(argument)] = call.arguments.args.as_ref() else {
        return false;
    };
    if argument.id.as_str() == sequence_name {
        return true;
    }
    // The link runs in either direction: the len() argument may be a
    // comprehension over the looped sequence, OR the looped sequence
    // may be a comprehension over the len() argument — the fixture's
    // own shape (loop over `clamped`, divide by `len(samples)`, with
    // `clamped` the 1:1 comprehension over `samples`). Both spell the
    // same proved equality of the two lengths.
    if environment
        .read(argument.id.as_str())
        .and_then(|value| value.same_length_as.as_deref())
        == Some(sequence_name)
    {
        return true;
    }
    environment
        .read(sequence_name)
        .and_then(|value| value.same_length_as.as_deref())
        == Some(argument.id.as_str())
}
