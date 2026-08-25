//! Recognition: turning the loop / generator-sum / bare-name-sum
//! spellings of an accumulation into a `RecognizedAccumulation` and its
//! lowered `loopAccum` program.

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::trust_level_of;
use refined_kernel::loop_questions::IrStatement;
use refined_kernel::loop_questions::IrStatementKind;
use refined_kernel::narrow_questions::KnownStateWire;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtFor;

use crate::env::Environment;

use super::number_state;
use super::walk::element_and_count_sets;
use super::RecognizedAccumulation;
use super::COUNT_SLOT;
use super::ELEMENT_SLOT;
use super::TOTAL_SLOT;

/// Recognizes `for <var> in <name>: <total> += <expr over var>` as a
/// relational accumulation, given that `<total>` already holds exactly
/// 0 and `<name>` holds the element-set star shape.
///
/// `None` — declining to whatever path already ran — when: the target
/// is not one plain name; the iterable is not a plain name; that name
/// holds anything but a `Kind::Set` whose single form is a repetition
/// this reader can peel; the element set states nothing; the
/// accumulator is not currently exactly 0; the body is not exactly one
/// accumulating statement; or the added expression is not one this
/// module can lower exactly.
pub fn recognize_accumulation(
    for_stmt: &StmtFor,
    environment: &Environment,
) -> Option<RecognizedAccumulation> {
    let Expr::Name(loop_variable) = for_stmt.target.as_ref() else {
        return None;
    };
    let Expr::Name(sequence) = for_stmt.iter.as_ref() else {
        return None;
    };
    let (total_name, added) = accumulating_body(&for_stmt.body)?;
    // The accumulator must ALREADY be exactly 0: the kernel's own
    // relation starts the total at 0, so an accumulator carrying
    // anything else (a partial sum, an unknown, a set) is a different
    // computation and this reader states nothing about it.
    let seed = environment.read(&total_name)?;
    if !is_exactly_zero(seed) {
        return None;
    }
    // a float seed pins the total's sort; anything else states none
    let total_kind_tag = match seed.kind_tag {
        Some(PrimitiveKind::Float) => Some(PrimitiveKind::Float),
        _ => None,
    };
    accumulation_program(
        total_name,
        sequence.id.as_str(),
        loop_variable.id.as_str(),
        added,
        environment,
        total_kind_tag,
    )
}

/// Recognizes `<total> = sum(<elt> for <var> in <name>)` — the
/// generator-sum spelling of the same computation the explicit loop
/// spells, and the one the cross-language fixture uses
/// (`audio_level.py:19`). It lowers to the IDENTICAL `loopAccum`
/// program: `sum` starts its total at 0 by definition
/// (library/functions.html#sum, "Sums *start* and the items of an
/// *iterable* from left to right"), so no prior binding of `<total>` is
/// read or required.
///
/// `None` — leaving `builtin_call_result_with_kernel`'s own sum-over-star
/// sign envelope as the fallback it is meant to be — when: the statement
/// is not a single-name assignment; the value is not a direct call to a
/// name `sum` that no local binding shadows; a `start` argument is
/// present and is not exactly 0 (a nonzero start shifts the total off
/// the relation's own zero base); any keyword argument rides; the
/// argument is not a bare generator expression (`sum([...])` over a
/// list comprehension declines outright — the eager path already
/// materializes it, and this reader must not race it); the generator
/// has more than one `for` clause, any `if` clause, or is an `async
/// for`; the target is not one plain name; or the shared program
/// builder below declines.
pub fn recognize_generator_sum(
    assign: &StmtAssign,
    environment: &Environment,
) -> Option<RecognizedAccumulation> {
    let [Expr::Name(total)] = assign.targets.as_slice() else {
        return None;
    };
    recognize_generator_sum_call(assign.value.as_ref(), total.id.as_str().to_owned(), environment)
}

/// Recognizes `sum(<elt> for <var> in <name>)` at a position with no
/// naming target of its own — a bare `return sum(...)`, where the total
/// is never bound to a name and instead routes straight to the return
/// sink. `total_name` still fills `RecognizedAccumulation`'s own field
/// (`accumulation_program`'s distinctness guard, and a division-folding
/// match neither caller of this function exercises for an unnamed
/// total), but nothing here binds it — the caller decides what a
/// synthetic name should be.
///
/// The recognition rules are exactly `recognize_generator_sum`'s own —
/// this is the shared body both the assignment and the return spelling
/// call, split out so a return position never has to synthesize a
/// `StmtAssign` it does not have.
fn recognize_generator_sum_call(
    value: &Expr,
    total_name: String,
    environment: &Environment,
) -> Option<RecognizedAccumulation> {
    let Expr::Call(call) = value else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    // a local binding named `sum` is not the builtin, the same
    // shadow-on-rebind rule every other builtin recognition applies
    if callee.id.as_str() != "sum" || environment.read(callee.id.as_str()).is_some() {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let generator = match call.arguments.args.as_ref() {
        [generator] => generator,
        // `sum(gen, start)` — only a start of exactly 0 is the relation's
        // own base; anything else is a different total
        [generator, start] if is_zero_literal(start) => generator,
        _ => return None,
    };
    // a bare generator ONLY: a list/set display argument is already
    // materialized eagerly elsewhere
    let Expr::Generator(generator) = generator else {
        return None;
    };
    let [clause] = generator.generators.as_slice() else {
        return None;
    };
    if clause.is_async || !clause.ifs.is_empty() {
        return None;
    }
    let Expr::Name(loop_variable) = &clause.target else {
        return None;
    };
    let Expr::Name(sequence) = &clause.iter else {
        return None;
    };
    accumulation_program(
        total_name,
        sequence.id.as_str(),
        loop_variable.id.as_str(),
        generator.elt.as_ref(),
        environment,
        // sum() has no seed binding to read a sort off
        None,
    )
}

/// Recognizes a bare `return sum(<elt> for <var> in <name>)` — the same
/// generator-sum shape `recognize_generator_sum` reads at an
/// assignment, at the return position instead. There is no assignment
/// target to read a total's name from, since the total is never bound —
/// it routes straight to the return sink (`check.rs`'s own
/// `Environment::set_evaluated_node` seam, the same one a folded
/// division's quotient already rides in return position). `total_name`
/// carries a placeholder no real Python identifier can equal (Python
/// identifiers hold no `$`, `identifiers.rst`), since nothing reads it
/// as a name to bind here — it exists only to satisfy
/// `accumulation_program`'s distinctness guard against the sequence and
/// loop-variable names.
///
/// `None` under the exact same conditions `recognize_generator_sum`
/// declines under.
pub fn recognize_generator_sum_in_return(
    value: &Expr,
    environment: &Environment,
) -> Option<RecognizedAccumulation> {
    recognize_generator_sum_call(value, "$return".to_owned(), environment)
}

/// Recognizes `<total> = sum(<name>)` — `sum` called directly on a plain
/// name bound to the element-set star shape, with no generator and no
/// per-element transform. This is the shape `sum([s * s for s in
/// samples])`'s own doc calls "already materialized eagerly elsewhere":
/// unlike that list-display argument, a BARE name argument names no
/// concrete items to walk, so the eager path can only fall back to plain
/// interval arithmetic over the sequence's element hull — the same
/// weak-division problem `accumulation_program`'s own module doc states
/// for the generator spelling. Sharing the identical `loopAccum` lowering
/// with the generator forms is what ties the total to the count here
/// too.
///
/// There is no per-element expression to lower: `sum(xs)` reads each
/// element unchanged, so the per-trip effect is always exactly
/// `slot(ELEMENT_SLOT)` — there is no source AST node for a transform,
/// so this does not go through `accumulation_program`'s own
/// `lower_added_expression` call, which requires one.
///
/// The total's sort is known here in a way neither other recognized form
/// states: `sum(xs)` performs no per-element transform, so the total's
/// sort is exactly the sequence's own element sort, read off `<name>`'s
/// own `kind_tag` — the same field `builtin_models.rs`'s
/// `sum_call_over_star` reads for its own (non-relational) sign-envelope
/// row on this identical star-shaped-iterable case.
///
/// `None` — declining to the eager path — under the exact same
/// conditions `recognize_generator_sum` declines under, plus: the
/// argument is not a single bare name (a list/set display, a generator,
/// or any other expression is a different shape read elsewhere); or that
/// name's own value states no element sort (`kind_tag` is `None` or
/// anything but `Integer`/`Float`) — the total's sort must be known
/// exactly, the same requirement `sum_call_over_star` states for its own
/// row.
pub fn recognize_sum_over_name(
    assign: &StmtAssign,
    environment: &Environment,
) -> Option<RecognizedAccumulation> {
    let [Expr::Name(total)] = assign.targets.as_slice() else {
        return None;
    };
    let Expr::Call(call) = assign.value.as_ref() else {
        return None;
    };
    let Expr::Name(callee) = call.func.as_ref() else {
        return None;
    };
    // a local binding named `sum` is not the builtin, the same
    // shadow-on-rebind rule every other builtin recognition applies
    if callee.id.as_str() != "sum" || environment.read(callee.id.as_str()).is_some() {
        return None;
    }
    if !call.arguments.keywords.is_empty() {
        return None;
    }
    let sequence = match call.arguments.args.as_ref() {
        [Expr::Name(sequence)] => sequence,
        // `sum(xs, start)` — only a start of exactly 0 is the relation's
        // own base; anything else is a different total
        [Expr::Name(sequence), start] if is_zero_literal(start) => sequence,
        _ => return None,
    };
    let total_name = total.id.as_str().to_owned();
    let sequence_name = sequence.id.as_str();
    let loop_variable = "$elt";
    if total_name == sequence_name || total_name == loop_variable || sequence_name == loop_variable {
        return None;
    }
    let sequence_value = environment.read(sequence_name)?;
    let total_kind_tag = match sequence_value.kind_tag {
        Some(PrimitiveKind::Integer) => Some(PrimitiveKind::Integer),
        Some(PrimitiveKind::Float) => Some(PrimitiveKind::Float),
        _ => return None,
    };
    let (element_set, count_set) = element_and_count_sets(sequence_value)?;
    Some(RecognizedAccumulation {
        total_name,
        sequence_name: sequence_name.to_owned(),
        entry_states: vec![
            number_state(make_refined_set(vec![one_of(&[0.0])])),
            number_state(element_set),
            number_state(count_set.clone()),
            KnownStateWire {
                top: true,
                set: make_refined_set(vec![]),
                undef: false,
                null: false,
                nan: false,
                thrown: false,
            },
        ],
        statements: vec![IrStatement {
            kind: IrStatementKind::LoopAccum,
            target: TOTAL_SLOT,
            accum_src: ELEMENT_SLOT,
            accum_len: COUNT_SLOT,
            effect: super::slot(ELEMENT_SLOT),
            ..Default::default()
        }],
        grade: trust_level_of(sequence_value),
        total_kind_tag,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set,
    })
}

/// The program both recognized forms build: the four entry states and
/// the one `loopAccum` statement. Everything here comes from knowledge
/// the walk already holds — the sequence's own element set and count
/// window — so nothing invents a fact the ordinary walk did not carry.
///
/// `None` when the sequence name holds no value, holds something other
/// than a peelable element-set repetition, or the per-trip expression
/// does not lower. The accumulator, the sequence, and the loop target
/// must also be three DISTINCT names: a body summing a sequence into
/// itself, or into its own iteration variable, is not this shape.
fn accumulation_program(
    total_name: String,
    sequence_name: &str,
    loop_variable: &str,
    added: &Expr,
    environment: &Environment,
    total_kind_tag: Option<PrimitiveKind>,
) -> Option<RecognizedAccumulation> {
    if total_name == sequence_name || total_name == loop_variable || sequence_name == loop_variable {
        return None;
    }
    let sequence_value = environment.read(sequence_name)?;
    let (element_set, count_set) = element_and_count_sets(sequence_value)?;
    let body = super::lowering::lower_added_expression(added, loop_variable)?;
    Some(RecognizedAccumulation {
        total_name,
        sequence_name: sequence_name.to_owned(),
        entry_states: vec![
            // slot 0: the total, entering at exactly 0
            number_state(make_refined_set(vec![one_of(&[0.0])])),
            // slot 1: the element abstraction the body reads each trip
            number_state(element_set),
            // slot 2: the count
            number_state(count_set.clone()),
            // slot 3: the quotient slot, holding nothing until a
            // division writes it
            KnownStateWire {
                top: true,
                set: make_refined_set(vec![]),
                undef: false,
                null: false,
                nan: false,
                thrown: false,
            },
        ],
        statements: vec![IrStatement {
            kind: IrStatementKind::LoopAccum,
            target: TOTAL_SLOT,
            accum_src: ELEMENT_SLOT,
            accum_len: COUNT_SLOT,
            effect: body,
            ..Default::default()
        }],
        grade: trust_level_of(sequence_value),
        total_kind_tag,
        quotient_op: None,
        length_aliases: std::collections::HashMap::new(),
        count_set,
    })
}

/// The one accumulating statement a recognized body holds: `<total> +=
/// <expr>`, or its longhand `<total> = <total> + <expr>`. Answers the
/// accumulator's name beside the expression being added.
///
/// `None` for a body of any other length or shape. A second statement
/// could write the accumulator or the sequence a second time, and this
/// module states the relation for exactly one addition per trip.
fn accumulating_body(body: &[Stmt]) -> Option<(String, &Expr)> {
    let [statement] = body else {
        return None;
    };
    match statement {
        Stmt::AugAssign(assign) => {
            if !matches!(assign.op, Operator::Add) {
                return None;
            }
            let Expr::Name(target) = assign.target.as_ref() else {
                return None;
            };
            Some((target.id.as_str().to_owned(), assign.value.as_ref()))
        }
        Stmt::Assign(assign) => {
            let [Expr::Name(target)] = assign.targets.as_slice() else {
                return None;
            };
            let Expr::BinOp(binop) = assign.value.as_ref() else {
                return None;
            };
            if !matches!(binop.op, Operator::Add) {
                return None;
            }
            // `total = total + <expr>` — the left operand must be the
            // accumulator itself, or this is not an accumulation
            let Expr::Name(left) = binop.left.as_ref() else {
                return None;
            };
            if left.id != target.id {
                return None;
            }
            Some((target.id.as_str().to_owned(), binop.right.as_ref()))
        }
        _ => None,
    }
}

/// Whether a value is the exact number 0 — the only accumulator start
/// the kernel's relation is stated from.
fn is_exactly_zero(value: &AbstractValue) -> bool {
    value.kind == Kind::Values && value.values.as_slice() == [0.0]
}

/// Whether an expression is the literal `0` — `sum`'s own default
/// start, and the only start the relation's zero base admits. Read
/// syntactically rather than through the environment: a `start`
/// argument is an expression at the call, not a binding this reader
/// tracks.
fn is_zero_literal(expression: &Expr) -> bool {
    let Expr::NumberLiteral(literal) = expression else {
        return false;
    };
    match &literal.value {
        Number::Int(int) => int.as_i64() == Some(0),
        Number::Float(float) => *float == 0.0,
        Number::Complex { .. } => false,
    }
}
