/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Accumulate-then-divide recognition: `total = 0`, a loop adding one
//! value per element of a sequence into `total`, then a division of
//! `total` by that same sequence's length.
//!
//! Interval arithmetic alone answers this division far too weakly. A
//! loop over a sequence of `n` elements each in `[0, 1]` leaves `total`
//! in `[0, n]`, and `n` is a runtime length, so plain division of one
//! enclosure by another gives `[0, n] / [1, n]` — which is `[0, n]`,
//! not `[0, 1]`. The tight answer needs the RELATION between the total
//! and the count, not just their separate ranges.
//!
//! So this module does not compute the answer. It recognizes the shape
//! and lowers it to the kernel's `loopAccum` statement, which ties
//! `total` to the count as linear facts (`total <= count * elemHi` and
//! the lower twin); a division lowered as the NEXT statement of the
//! SAME kernel program is then narrowed by the kernel's linear decider.
//! Every fact about the result is derived kernel-side and read back off
//! the walk's own exit states. No shortcut answers the division here —
//! a mean-of-bounded fold computed adapter-side is exactly what
//! CROSS-LANGUAGE-EDGE.md's ruling R2 rejects.
//!
//! ## The slot environment
//!
//! The lowered program uses four slots, fixed here because the whole
//! program is built here — nothing else allocates into this vector:
//!
//! | slot | holds                                                |
//! |------|------------------------------------------------------|
//! | 0    | `total` — the running sum, entering at exactly 0      |
//! | 1    | the sequence's ELEMENT abstraction, read once a trip  |
//! | 2    | the count — the sequence's length                    |
//! | 3    | the quotient, when a division rode along              |
//!
//! The quotient gets its OWN slot rather than overwriting the total's:
//! both names survive the two statements in Python, so both exit states
//! are read back and the total is never left holding a value it never
//! had.
//!
//! ## What is recognized
//!
//! Statement one is EITHER of two spellings of the same computation:
//!
//! - the explicit loop — `for <x> in <seq>: <total> += <elt>`, with
//!   `<total>` already holding exactly 0;
//! - the generator sum — `<total> = sum(<elt> for <x> in <seq>)`, one
//!   generator, no `if` clauses, no nonzero `start`. This is the
//!   fixture's own spelling (`audio_level.py:19`), and it needs no
//!   prior binding of `<total>`: `sum` starts at 0 by definition
//!   (library/functions.html#sum).
//!
//! Both lower to the IDENTICAL `loopAccum` program. In both, `<seq>`
//! must be a plain name bound to the element-set star shape
//! (`Form::Star`, what `check.rs`'s `seed_parameters` builds for a
//! `list[X]`/`Sequence[X]` parameter), and `<elt>` must read the loop
//! target and numeric literals only, combined with `+`, `-`, and `*`.
//!
//! Statement two, optional and the same in both forms, carries the
//! division `<total> / len(<seq>)` in either of two positions:
//!
//! - an assignment — `<mean> = <total> / len(<seq>)`, the division at
//!   the top level, naming the quotient;
//! - a return — `return <expr containing the division>`, where the
//!   division may sit nested at any depth, as the fixture's
//!   `return math.sqrt(total / len(samples))` does
//!   (`audio_level.py:25`). Exactly one occurrence, so a single
//!   published answer can never be ambiguous about which node it
//!   belongs to.
//!
//! Everything else declines to the paths that already existed,
//! unchanged — including `sum([...])` over a list comprehension, which
//! the eager path already materializes.

use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::trust_level_of;
use refined_domain::trust_grades::TrustLevel;
use refined_kernel::loop_questions::IrStatement;
use refined_kernel::loop_questions::IrStatementKind;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::loop_questions::LoopEffectOp;
use refined_kernel::narrow_questions::KnownStateWire;
use refined_kernel::summary_questions::ask_walk_relational;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::integer;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::repetition_window_forms::as_repetition;
use ruff_python_ast::Expr;
use ruff_python_ast::Number;
use ruff_python_ast::Operator;
use ruff_python_ast::Stmt;
use ruff_python_ast::StmtAssign;
use ruff_python_ast::StmtFor;
use ruff_text_size::Ranged;
use ruff_text_size::TextRange;

use crate::refinedpy::env::Environment;

/// The slot the running total lives in.
const TOTAL_SLOT: i64 = 0;
/// The slot the sequence's element abstraction lives in.
const ELEMENT_SLOT: i64 = 1;
/// The slot the count lives in.
const COUNT_SLOT: i64 = 2;
/// The slot a folded division writes its quotient into.
const QUOTIENT_SLOT: i64 = 3;

/// What a recognized accumulation names: the accumulator, the sequence
/// it ran over, and the lowered program the kernel walks.
pub struct RecognizedAccumulation {
    /// The name the running total is bound to.
    pub total_name: String,
    /// The name the sequence is bound to — a later division by
    /// `len(<this name>)` is the same sequence's count.
    pub sequence_name: String,
    /// The entry states, one per slot, in slot order.
    pub entry_states: Vec<KnownStateWire>,
    /// The lowered statements: the accumulation, plus the division
    /// when one was folded in.
    pub statements: Vec<IrStatement>,
    /// The trust grade the sequence's own value carried, which the
    /// answer inherits — a spec-read element set never yields a proved
    /// total.
    pub grade: TrustLevel,
    /// The total's language-level sort, when the SEED states it: a
    /// float seed (`total = 0.0`) makes every later `total += _` a
    /// float — float absorbs each numeric add — so Float is exact. An
    /// int seed or the seedless generator-sum spelling states nothing:
    /// the elements' sort could go either way and no correct
    /// per-element sort survives the repetition-window read.
    pub total_kind_tag: Option<PrimitiveKind>,
}

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
        total.id.as_str().to_owned(),
        sequence.id.as_str(),
        loop_variable.id.as_str(),
        generator.elt.as_ref(),
        environment,
        // sum() has no seed binding to read a sort off
        None,
    )
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
    let body = lower_added_expression(added, loop_variable)?;
    Some(RecognizedAccumulation {
        total_name,
        sequence_name: sequence_name.to_owned(),
        entry_states: vec![
            // slot 0: the total, entering at exactly 0
            number_state(make_refined_set(vec![one_of(&[0.0])])),
            // slot 1: the element abstraction the body reads each trip
            number_state(element_set),
            // slot 2: the count
            number_state(count_set),
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
    })
}

/// Folds a division of the accumulated total by the sequence's own
/// length into the SAME lowered program, as the statement after the
/// accumulation. This is what makes the relation pay: the kernel's
/// linear decider narrows a division whose numerator it tied to its
/// denominator one statement earlier, where a separate question would
/// see only two unrelated enclosures.
///
/// Recognizes `<total> / len(<sequence>)` for exactly the accumulator
/// and sequence this accumulation named — OR a sequence a comprehension
/// built 1:1 over it with no filter (`AbstractValue::same_length_as`,
/// `is_len_of`'s own doc). `false` — leaving the program as the
/// accumulation alone — for any other shape: a different, unlinked name
/// on either side, a length taken of some other sequence, an operator
/// that is not true division.
pub fn fold_division(
    recognized: &mut RecognizedAccumulation,
    expression: &Expr,
    environment: &Environment,
) -> bool {
    if !is_relational_division(expression, recognized, environment) {
        return false;
    }
    recognized.statements.push(division_statement());
    true
}

/// The division assignment both folding routes push: the total slot
/// over the count slot, into the quotient's own slot. The quotient
/// rides its own slot because both names survive these two statements
/// in Python, so both exit states are read back and the total is never
/// left holding a value it never had.
fn division_statement() -> IrStatement {
    IrStatement {
        kind: IrStatementKind::Assign,
        target: QUOTIENT_SLOT,
        effect: LoopEffect {
            kind: LoopEffectKind::Binary,
            op: LoopEffectOp::Div,
            a: Some(Box::new(slot(TOTAL_SLOT))),
            b: Some(Box::new(slot(COUNT_SLOT))),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Folds the division into the program without re-matching the node —
/// the caller already located it with `division_range_in`, which
/// matched the same shape `fold_division` checks. Used by the return
/// form, where the division sits nested inside the returned expression
/// rather than being that expression.
pub fn fold_located_division(recognized: &mut RecognizedAccumulation) {
    recognized.statements.push(division_statement());
}

/// The range of the ONE `<total> / len(<sequence>)` division inside
/// `expression`, at any depth — the shape a `return` wraps in the
/// fixture (`return math.sqrt(total / len(samples))`,
/// `audio_level.py:25`), where the division is a call argument rather
/// than the returned expression itself.
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
) -> Option<TextRange> {
    let mut found: Option<TextRange> = None;
    let mut count = 0;
    find_divisions(expression, recognized, environment, &mut found, &mut count);
    match count {
        1 => found,
        _ => None,
    }
}

/// Walks every subexpression, recording each `<total> / len(<seq>)` it
/// meets. Counts past one so the caller can tell "exactly one" from
/// "more than one"; the walk never stops early, because that
/// distinction is the whole point.
///
/// The recursion mirrors `check.rs`'s own `collect_walrus_names`,
/// including its scope rule: a `lambda`'s body is a separate scope
/// whose own `total` is a different binding, so no division inside one
/// is this accumulation's.
fn find_divisions(
    expression: &Expr,
    recognized: &RecognizedAccumulation,
    environment: &Environment,
    found: &mut Option<TextRange>,
    count: &mut usize,
) {
    if is_relational_division(expression, recognized, environment) {
        *found = Some(expression.range());
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

/// Whether an expression is exactly `<total> / len(<sequence>)` for the
/// accumulator and sequence this accumulation named — or `len` of a
/// DIFFERENT name whose value the environment holds with
/// `same_length_as` proved equal to the accumulation's own sequence
/// (`is_len_of`'s own doc).
fn is_relational_division(
    expression: &Expr,
    recognized: &RecognizedAccumulation,
    environment: &Environment,
) -> bool {
    let Expr::BinOp(binop) = expression else {
        return false;
    };
    if !matches!(binop.op, Operator::Div) {
        return false;
    }
    let Expr::Name(numerator) = binop.left.as_ref() else {
        return false;
    };
    numerator.id.as_str() == recognized.total_name
        && is_len_of(binop.right.as_ref(), &recognized.sequence_name, environment)
}

/// What the kernel answered: the accumulated total, and the quotient
/// when a division rode along.
pub struct AccumulationAnswer {
    /// The value the accumulator holds after the loop, when the kernel
    /// answered a bindable set for it. `None` when the total's own
    /// enclosure is honestly unbounded (a sign-straddling step times an
    /// unbounded count) while the LEDGER still proved the quotient —
    /// the two claims are independent in both directions.
    pub total: Option<AbstractValue>,
    /// The value the divided name holds, when a division was folded in
    /// and the kernel answered a bindable set for it.
    pub quotient: Option<AbstractValue>,
}

/// Asks the kernel to walk the lowered program and reads back the slots
/// it wrote.
///
/// The RELATIONAL walk is what is asked, never the certifying one: the
/// certifying walk drops the linear ledger, so the division would be
/// answered by plain interval arithmetic and the whole lowering would
/// buy nothing (`kernel_interface`'s `walk_relational` states why the
/// two paths are exclusive).
///
/// `None` whenever the kernel declines (`ask_walk_relational`'s own
/// refusal discipline: no kernel loaded, or a question the kernel
/// refuses), or when the TOTAL's own answer is not a plain set this
/// checker can bind — an unknown exit state is an honest refusal, never
/// a set to invent. A folded division whose quotient came back unknown
/// leaves `quotient` as `None` while the total still stands: the two
/// claims are independent, and dropping the good one with the bad would
/// state less than the kernel proved.
pub fn walk_accumulation(recognized: &RecognizedAccumulation) -> Option<AccumulationAnswer> {
    let asked = ask_walk_relational(&recognized.entry_states, &recognized.statements, &[]);
    // Debug instrument (REFINEDPY_DEBUG_RELATIONAL): the raw exit
    // states as the kernel answered them, or the ask's own refusal —
    // the split `check.rs`'s instrument cannot see from one layer up.
    if std::env::var("REFINEDPY_DEBUG_RELATIONAL").is_ok() {
        match &asked {
            None => eprintln!("relational_sum: ask_walk_relational answered None (no kernel, or the ask panicked)"),
            Some(states) => eprintln!("relational_sum: kernel exit states={states:?}"),
        }
    }
    let states = asked?;
    // The total wears the sort its seed pinned (Float for a `0.0`
    // seed; nothing otherwise — recognize_accumulation's own rule).
    let total = states
        .get(TOTAL_SLOT as usize)
        .and_then(|state| bindable_state(state, recognized.grade,
            recognized.total_kind_tag));
    let quotient = match recognized.statements.len() {
        // no division rode along, so the quotient slot was never written
        1 => None,
        // the folded division is Python's `/`, which yields a float for
        // every numeric operand pair (expressions.rst, binary arithmetic:
        // "division of integers yields a float"), so the quotient is
        // Float-sorted unconditionally — the sort the sqrt/floor call
        // rows downstream require before they ask the kernel
        _ => states
            .get(QUOTIENT_SLOT as usize)
            .and_then(|state| bindable_state(state, recognized.grade, Some(PrimitiveKind::Float))),
    };
    // An answer with NEITHER slot bindable claims nothing; either slot
    // alone still stands — a top total must not drop a proved quotient
    // (the ledger ties the quotient to the count even when the total's
    // own enclosure is unbounded), and the reverse held already.
    if total.is_none() && quotient.is_none() {
        return None;
    }
    Some(AccumulationAnswer { total, quotient })
}

/// One exit state as a value this checker can bind, or `None` when the
/// kernel claimed nothing about that slot. The sort tag is the caller's
/// claim about the slot's language-level sort; `None` states no sort.
fn bindable_state(
    state: &KnownStateWire,
    grade: TrustLevel,
    kind_tag: Option<PrimitiveKind>,
) -> Option<AbstractValue> {
    if state.top || state.set.forms.is_empty() {
        return None;
    }
    Some(AbstractValue {
        kind_tag,
        ..known_set(state.set.clone(), None, grade, SetKindTag::None)
    })
}

/// The element set and the count set of a sequence value, read off the
/// repetition window its own set states. The count rides as a
/// nonnegative INTEGER-sorted number state — a length is a whole count,
/// never a fraction — bounded below by the window's own `lo` and above
/// by its `hi` when the window states one.
///
/// `None` for a value that is not a `Kind::Set`, a set that is not a
/// repetition this reader can peel, or a repetition whose element set
/// states nothing at all. How TIGHT the element set is, is not gated
/// here: the kernel derives what the relation supports, and a hull it
/// cannot tie answers an unknown exit state that `walk_accumulation`
/// declines on. Gating it twice would put an adapter-side judgment in
/// front of the kernel's own.
fn element_and_count_sets(value: &AbstractValue) -> Option<(RefinedSet, RefinedSet)> {
    if value.kind != Kind::Set {
        return None;
    }
    let repetition = as_repetition(&value.set)?;
    if repetition.element.forms.is_empty() {
        return None;
    }
    let mut count_forms = vec![integer(), at_least(repetition.lo as f64)];
    if let Some(hi) = repetition.hi {
        count_forms.push(at_most(hi as f64));
    }
    Some((repetition.element, make_refined_set(count_forms)))
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

/// Lowers the expression added on each trip into the kernel's effect
/// grammar, over the element slot. Recognizes the loop variable itself,
/// a numeric literal, and the arithmetic combinations of those the
/// kernel has proved transfers for.
///
/// INVARIANT, and the reason no separate term gate is needed: the only
/// `Var` effect this function can emit is `ELEMENT_SLOT`, because the
/// loop variable is the one name it matches and every other name
/// declines. So a lowered term reads the element and constants and
/// nothing else — which is exactly the premise the kernel's relation
/// rests on (`total <= count * termHi`, with `termHi` read off the
/// element alone). A term reading the running total would be a
/// recurrence rather than a sum, and one reading the count would tie
/// the total to a factor the relation does not carry; neither can be
/// built here.
///
/// `None` for anything wider — a call, an attribute read, a name that
/// is neither the loop variable nor a literal. This module never
/// approximates a body step it cannot state exactly; the accumulation
/// declines whole and the existing paths run unchanged.
fn lower_added_expression(expression: &Expr, loop_variable: &str) -> Option<LoopEffect> {
    match expression {
        Expr::Name(name) if name.id.as_str() == loop_variable => Some(slot(ELEMENT_SLOT)),
        Expr::NumberLiteral(literal) => {
            // an int past i64 is not a value this reader states exactly,
            // the same ceiling every other literal reader here keeps
            let value = match &literal.value {
                Number::Int(int) => int.as_i64()? as f64,
                Number::Float(float) => *float,
                Number::Complex { .. } => return None,
            };
            Some(LoopEffect {
                kind: LoopEffectKind::Const,
                set: make_refined_set(vec![one_of(&[value])]),
                ..Default::default()
            })
        }
        // `s * s` — both operands the loop variable, the SAME source
        // variable — is a structural square: the kernel's `Effect.sq`
        // answers the correlated image `[0, max²]`, which the general
        // product `Binary(Mul, Var(i), Var(i))` cannot, since the
        // kernel no longer recognizes x*x by syntax (unsound under
        // renaming). Read directly off the source AST, before either
        // side lowers: this is the one place the identifier binding is
        // honestly known, and a lowered `LoopEffect` has already
        // thrown that identity away. Gated on the loop variable
        // specifically (not just "the same name as each other"): a
        // product of some OTHER shared free name would otherwise
        // misread as squaring the element it never named.
        Expr::BinOp(binop) if is_same_name_square(binop, loop_variable) => Some(LoopEffect {
            kind: LoopEffectKind::Sq,
            index: ELEMENT_SLOT,
            ..Default::default()
        }),
        Expr::BinOp(binop) => {
            let op = match binop.op {
                Operator::Add => LoopEffectOp::Add,
                Operator::Sub => LoopEffectOp::Sub,
                Operator::Mult => LoopEffectOp::Mul,
                // Div/FloorDiv/Mod/Pow are deliberately absent: each
                // carries a Python/JS divergence or a zero-denominator
                // premise this reader does not vouch for inside a body
                // it is stating a relation about.
                _ => return None,
            };
            let left = lower_added_expression(binop.left.as_ref(), loop_variable)?;
            let right = lower_added_expression(binop.right.as_ref(), loop_variable)?;
            Some(LoopEffect {
                kind: LoopEffectKind::Binary,
                op,
                a: Some(Box::new(left)),
                b: Some(Box::new(right)),
                ..Default::default()
            })
        }
        _ => None,
    }
}

/// Whether a `BinOp` is `<loop_variable> * <loop_variable>` — decided
/// from the source AST's own two `Expr::Name` nodes, never from a
/// lowered effect, which has already erased which variable a term came
/// from. Gated on the LOOP variable specifically, not merely "the same
/// name as each other": some other shared free name would decline
/// anyway (this module's own invariant — the only `Var` this reader
/// emits is the element slot), and must keep declining rather than
/// being misread as squaring the element.
fn is_same_name_square(binop: &ruff_python_ast::ExprBinOp, loop_variable: &str) -> bool {
    if !matches!(binop.op, Operator::Mult) {
        return false;
    }
    let (Expr::Name(left), Expr::Name(right)) = (binop.left.as_ref(), binop.right.as_ref()) else {
        return false;
    };
    left.id.as_str() == loop_variable && right.id.as_str() == loop_variable
}

/// Whether an expression is `len(<name>)` for exactly this sequence —
/// OR `len(<other name>)` where `<other name>`'s own value carries
/// `AbstractValue::same_length_as == Some(sequence_name)`: a name a
/// comprehension built by mapping every position of `sequence_name` 1:1
/// with no filter, which proves `len(<other name>) == len(sequence_name)`
/// exactly (`comprehension_star_elements`'s own soundness-line comment,
/// expressions.rs — the same fact stated there as a window bound, here
/// read back as a name link). A name with no recorded link, or one
/// linked to some THIRD sequence, still declines: only an exact proof
/// of equal length licenses folding the division into this program.
fn is_len_of(expression: &Expr, sequence_name: &str, environment: &Environment) -> bool {
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

/// A read of one slot.
fn slot(index: i64) -> LoopEffect {
    LoopEffect {
        kind: LoopEffectKind::Var,
        index,
        ..Default::default()
    }
}

/// A plain number state: a set, admitting neither absence nor NaN nor
/// a thrown exit.
fn number_state(set: RefinedSet) -> KnownStateWire {
    KnownStateWire {
        top: false,
        set,
        undef: false,
        null: false,
        nan: false,
        thrown: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use refined_domain::trust_grades::TrustProved;
    use refined_kernel::loop_questions::stmt_wire;

    // `total += s * s` over an element slot — the fixture's own body
    // (audio_level.py:19), lowered.
    fn squared_element() -> LoopEffect {
        LoopEffect {
            kind: LoopEffectKind::Binary,
            op: LoopEffectOp::Mul,
            a: Some(Box::new(slot(ELEMENT_SLOT))),
            b: Some(Box::new(slot(ELEMENT_SLOT))),
            ..Default::default()
        }
    }

    #[test]
    fn the_accumulation_lowers_to_the_loop_accum_statement() {
        let statement = IrStatement {
            kind: IrStatementKind::LoopAccum,
            target: TOTAL_SLOT,
            accum_src: ELEMENT_SLOT,
            accum_len: COUNT_SLOT,
            effect: squared_element(),
            ..Default::default()
        };
        let got = stmt_wire(&statement);
        let want = r#"{"loopAccum":{"total":0,"src":1,"len":2,"body":{"op":"binary64.mul","A":{"var":1},"B":{"var":1}}}}"#;
        assert_eq!(got, want, "stmt_wire(accumulation) = {got:?}, want {want:?}");
    }

    #[test]
    fn the_folded_division_divides_the_total_slot_by_the_count_slot() {
        let mut recognized = RecognizedAccumulation {
            total_name: "total".to_owned(),
            sequence_name: "samples".to_owned(),
            entry_states: Vec::new(),
            statements: Vec::new(),
            grade: TrustProved,
            total_kind_tag: None,
        };
        let expression = division_expression("total", "samples");
        assert!(
            fold_division(&mut recognized, &expression, &environment_with_samples()),
            "fold_division declined `total / len(samples)`"
        );
        let [division] = recognized.statements.as_slice() else {
            panic!("want exactly the division statement, got {:?}", recognized.statements.len());
        };
        let got = stmt_wire(division);
        // the quotient lands in slot 3, its own — the total's slot 0 is
        // read as the numerator and left holding the total
        let want = r#"{"assign":{"target":3,"e":{"op":"binary64.div","A":{"var":0},"B":{"var":2}}}}"#;
        assert_eq!(got, want, "stmt_wire(division) = {got:?}, want {want:?}");
    }

    #[test]
    fn a_division_by_another_sequences_length_is_not_folded() {
        let mut recognized = RecognizedAccumulation {
            total_name: "total".to_owned(),
            sequence_name: "samples".to_owned(),
            entry_states: Vec::new(),
            statements: Vec::new(),
            grade: TrustProved,
            total_kind_tag: None,
        };
        let expression = division_expression("total", "others");
        assert!(
            !fold_division(&mut recognized, &expression, &environment_with_samples()),
            "fold_division accepted a length taken of a different sequence"
        );
        assert!(
            recognized.statements.is_empty(),
            "a declined division must leave the program alone"
        );
    }

    #[test]
    fn a_division_by_a_1_to_1_comprehensions_length_is_folded() {
        // `clamped = [max(-1.0, min(1.0, s)) for s in samples]` proves
        // `len(clamped) == len(samples)` exactly, so `total / len(clamped)`
        // ties to the SAME accumulation the loop over `samples` ran.
        let mut recognized = RecognizedAccumulation {
            total_name: "total".to_owned(),
            sequence_name: "samples".to_owned(),
            entry_states: Vec::new(),
            statements: Vec::new(),
            grade: TrustProved,
            total_kind_tag: None,
        };
        let mut environment = environment_with_samples();
        environment.bind(
            "clamped",
            AbstractValue {
                same_length_as: Some("samples".to_owned()),
                ..known_set(
                    make_refined_set(vec![refined_sets::refinement_forms::star(
                        make_refined_set(vec![at_least(-1.0), at_most(1.0)]),
                    )]),
                    None,
                    TrustProved,
                    SetKindTag::None,
                )
            },
        );
        let expression = division_expression("total", "clamped");
        assert!(
            fold_division(&mut recognized, &expression, &environment),
            "fold_division declined a length proved equal via same_length_as"
        );
    }

    #[test]
    fn a_division_by_a_filtered_comprehensions_length_is_not_folded() {
        // a filtered comprehension's own builder (expressions.rs's
        // `comprehension_star_elements`) leaves `same_length_as` unset —
        // this pins the CONSUMING side of that same soundness line: an
        // unlinked name still declines, exactly as an unrelated name does.
        let mut recognized = RecognizedAccumulation {
            total_name: "total".to_owned(),
            sequence_name: "samples".to_owned(),
            entry_states: Vec::new(),
            statements: Vec::new(),
            grade: TrustProved,
            total_kind_tag: None,
        };
        let mut environment = environment_with_samples();
        environment.bind(
            "positives",
            known_set(
                make_refined_set(vec![refined_sets::refinement_forms::star(
                    make_refined_set(vec![at_least(0.0), at_most(1.0)]),
                )]),
                None,
                TrustProved,
                SetKindTag::None,
            ),
        );
        let expression = division_expression("total", "positives");
        assert!(
            !fold_division(&mut recognized, &expression, &environment),
            "fold_division accepted a length with no proved link"
        );
        assert!(
            recognized.statements.is_empty(),
            "a declined division must leave the program alone"
        );
    }

    // `<numerator> / len(<sequence>)`, parsed rather than hand-built:
    // the AST shapes carry ranges and parenthesization this module
    // reads through, so a parsed expression is the honest input.
    fn division_expression(numerator: &str, sequence: &str) -> Expr {
        let source = format!("{numerator} / len({sequence})");
        let parsed = ruff_python_parser::parse_expression(&source)
            .expect("the test's own source parses");
        *parsed.into_syntax().body
    }

    // A module whose single statement is the assignment under test.
    fn parsed_assignment(source: &str) -> StmtAssign {
        let module = ruff_python_parser::parse_module(source)
            .expect("the test's own source parses")
            .into_syntax();
        let Some(Stmt::Assign(assign)) = module.body.into_iter().next() else {
            panic!("the test's own source must be a single assignment");
        };
        assign
    }

    // An environment holding `samples` as an unknown-length sequence of
    // -1.0 … 1.0 — the star shape `seed_parameters` builds for a
    // `Sequence[float]` parameter, and what the fixture's own body sees.
    fn environment_with_samples() -> Environment {
        let element = make_refined_set(vec![at_least(-1.0), at_most(1.0)]);
        let mut environment = Environment::new(std::collections::HashSet::from([
            "total".to_owned(),
            "samples".to_owned(),
        ]));
        environment.bind(
            "samples",
            known_set(
                make_refined_set(vec![refined_sets::refinement_forms::star(element)]),
                None,
                TrustProved,
                SetKindTag::None,
            ),
        );
        environment
    }

    #[test]
    fn the_generator_sum_lowers_to_the_same_program_the_explicit_loop_does() {
        // the fixture's own statement (audio_level.py:19) — `s * s` is
        // the SAME source variable on both sides, so this lowers to
        // the structural `sq` effect, not the general `mul` of two
        // vars.
        let assign = parsed_assignment("total = sum(s * s for s in samples)\n");
        let recognized = recognize_generator_sum(&assign, &environment_with_samples())
            .expect("the generator sum over a star sequence recognizes");
        assert_eq!(recognized.total_name, "total");
        assert_eq!(recognized.sequence_name, "samples");
        let [accumulation] = recognized.statements.as_slice() else {
            panic!("want exactly the accumulation, got {}", recognized.statements.len());
        };
        let got = stmt_wire(accumulation);
        let want = r#"{"loopAccum":{"total":0,"src":1,"len":2,"body":{"sq":1}}}"#;
        assert_eq!(got, want, "stmt_wire(generator sum) = {got:?}, want {want:?}");
    }

    #[test]
    fn the_generator_sums_entry_states_start_the_total_at_zero() {
        let assign = parsed_assignment("total = sum(s * s for s in samples)\n");
        let recognized = recognize_generator_sum(&assign, &environment_with_samples())
            .expect("the generator sum recognizes");
        assert_eq!(recognized.entry_states.len(), 4, "want one state per slot");
        let total = &recognized.entry_states[TOTAL_SLOT as usize];
        assert_eq!(
            total.set,
            make_refined_set(vec![one_of(&[0.0])]),
            "the total enters at exactly 0, want {:?}",
            total.set
        );
        assert!(
            recognized.entry_states[QUOTIENT_SLOT as usize].top,
            "the quotient slot holds nothing until a division writes it"
        );
    }

    #[test]
    fn a_generator_sum_with_a_nonzero_start_is_declined() {
        // the generator is parenthesized because a bare one cannot be
        // followed by another argument (expressions.rst, "Generator
        // expressions")
        let assign = parsed_assignment("total = sum((s * s for s in samples), 5)\n");
        assert!(
            recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
            "a nonzero start shifts the total off the relation's zero base"
        );
    }

    #[test]
    fn a_generator_sum_with_an_explicit_zero_start_is_recognized() {
        let assign = parsed_assignment("total = sum((s * s for s in samples), 0)\n");
        assert!(
            recognize_generator_sum(&assign, &environment_with_samples()).is_some(),
            "an explicit start of 0 is sum's own default and stays recognized"
        );
    }

    #[test]
    fn a_filtered_generator_is_declined() {
        // an `if` clause drops elements, so the count the relation ties
        // the total to is no longer the sequence's own length
        let assign = parsed_assignment("total = sum(s * s for s in samples if s > 0)\n");
        assert!(
            recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
            "a filtered generator sums over an unstated count"
        );
    }

    #[test]
    fn a_list_comprehension_argument_is_left_to_the_eager_path() {
        let assign = parsed_assignment("total = sum([s * s for s in samples])\n");
        assert!(
            recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
            "sum([...]) is already materialized eagerly and must not be raced"
        );
    }

    #[test]
    fn a_locally_bound_sum_is_not_the_builtin() {
        let assign = parsed_assignment("total = sum(s * s for s in samples)\n");
        let mut environment = environment_with_samples();
        environment.bind("sum", known_set(make_refined_set(vec![]), None, TrustProved, SetKindTag::None));
        assert!(
            recognize_generator_sum(&assign, &environment).is_none(),
            "a shadowed `sum` name is not the builtin this reader models"
        );
    }

    #[test]
    fn a_generator_body_reading_a_free_name_is_declined() {
        let assign = parsed_assignment("total = sum(s * gain for s in samples)\n");
        assert!(
            recognize_generator_sum(&assign, &environment_with_samples()).is_none(),
            "a term reading a name outside the element cannot be lowered exactly"
        );
    }

    // A recognized accumulation over `total` and `samples`, with no
    // program built — enough for the division readers, which only ever
    // consult the two names.
    fn recognized_over_samples() -> RecognizedAccumulation {
        RecognizedAccumulation {
            total_name: "total".to_owned(),
            sequence_name: "samples".to_owned(),
            entry_states: Vec::new(),
            statements: Vec::new(),
            grade: TrustProved,
            total_kind_tag: None,
        }
    }

    // The expression of a `return <source>` inside a def — the position
    // a return actually occupies, so the parse carries no diagnostic of
    // its own.
    fn returned_expression(source: &str) -> Expr {
        let module = ruff_python_parser::parse_module(&format!("def f():\n    return {source}\n"))
            .expect("the test's own source parses")
            .into_syntax();
        let Some(Stmt::FunctionDef(def)) = module.body.into_iter().next() else {
            panic!("the test's own source must be a single def");
        };
        let Some(Stmt::Return(ret)) = def.body.into_iter().next() else {
            panic!("the def's body must be a single return");
        };
        *ret.value.expect("the return carries a value")
    }

    #[test]
    fn the_division_is_found_nested_inside_a_call_argument() {
        // the fixture's own return (audio_level.py:25)
        let returned = returned_expression("math.sqrt(total / len(samples))");
        let range = division_range_in(&returned, &recognized_over_samples(), &environment_with_samples())
            .expect("the nested division is found");
        // the located node is the inner division, strictly inside the
        // call that wraps it
        assert!(
            range.start() > returned.range().start(),
            "want the inner division's range, not the whole call's: {range:?}"
        );
        assert!(
            range.end() < returned.range().end(),
            "want the inner division's range, not the whole call's: {range:?}"
        );
    }

    #[test]
    fn a_bare_division_in_return_position_is_found() {
        let returned = returned_expression("total / len(samples)");
        let range = division_range_in(&returned, &recognized_over_samples(), &environment_with_samples())
            .expect("a top-level division is found");
        assert_eq!(
            range,
            returned.range(),
            "the whole returned expression IS the division"
        );
    }

    #[test]
    fn a_return_holding_two_divisions_is_declined() {
        // one published answer cannot say which node it belongs to
        let returned = returned_expression("total / len(samples) + total / len(samples)");
        assert!(
            division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
            "two occurrences must decline rather than pick one"
        );
    }

    #[test]
    fn a_return_holding_no_division_is_declined() {
        let returned = returned_expression("math.sqrt(total)");
        assert!(
            division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
            "there is nothing to fold"
        );
    }

    #[test]
    fn a_division_by_a_different_sequences_length_is_not_found() {
        let returned = returned_expression("math.sqrt(total / len(others))");
        assert!(
            division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
            "a length taken of another sequence carries no relation"
        );
    }

    #[test]
    fn a_division_inside_a_lambda_body_is_not_this_accumulations() {
        // the lambda's body is its own scope, so its `total` is a
        // different binding
        let returned = returned_expression("lambda: total / len(samples)");
        assert!(
            division_range_in(&returned, &recognized_over_samples(), &environment_with_samples()).is_none(),
            "a lambda body is a separate scope"
        );
    }

    #[test]
    fn the_located_division_folds_the_same_statement_the_assignment_form_does() {
        let mut recognized = recognized_over_samples();
        fold_located_division(&mut recognized);
        let [division] = recognized.statements.as_slice() else {
            panic!("want exactly the division statement, got {}", recognized.statements.len());
        };
        let got = stmt_wire(division);
        let want = r#"{"assign":{"target":3,"e":{"op":"binary64.div","A":{"var":0},"B":{"var":2}}}}"#;
        assert_eq!(got, want, "stmt_wire(located division) = {got:?}, want {want:?}");
    }

    #[test]
    fn the_count_state_is_a_nonnegative_integer_bounded_by_the_window() {
        let element = make_refined_set(vec![at_least(-1.0), at_most(1.0)]);
        let sequence = known_set(
            make_refined_set(vec![refined_sets::refinement_forms::star(element.clone())]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        let (read_element, count) =
            element_and_count_sets(&sequence).expect("a star sequence reads its window");
        assert_eq!(read_element, element, "the element set reads back unchanged");
        // the star window is (0, unbounded): a whole count, at least 0,
        // with no upper bound to state
        let want = make_refined_set(vec![integer(), at_least(0.0)]);
        assert_eq!(count, want, "count = {count:?}, want {want:?}");
    }

    #[test]
    fn a_sequence_with_no_element_set_is_declined() {
        let empty = known_set(
            make_refined_set(vec![refined_sets::refinement_forms::star(
                make_refined_set(vec![]),
            )]),
            None,
            TrustProved,
            SetKindTag::None,
        );
        assert!(
            element_and_count_sets(&empty).is_none(),
            "an element set stating nothing must decline"
        );
    }
}
