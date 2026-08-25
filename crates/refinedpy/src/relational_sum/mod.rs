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
//! Statement two, optional and the same in both forms, carries a
//! division of `<total>` by `len(<seq>)` in either of two positions, in
//! either of two operators — `/` (true division, always Float) or `//`
//! (floor division, Integer when the total is Integer-sorted, Float
//! otherwise — expressions.rst, binary arithmetic):
//!
//! - an assignment — `<mean> = <total> / len(<seq>)` or `<total> //
//!   len(<seq>)`, the division at the top level, naming the quotient;
//! - a return — `return <expr containing the division>`, where the
//!   division may sit nested at any depth, as the fixture's
//!   `return math.sqrt(total / len(samples))` does
//!   (`audio_level.py:25`). Exactly one occurrence, so a single
//!   published answer can never be ambiguous about which node it
//!   belongs to.
//!
//! A THIRD spelling of statement one exists alongside the loop and the
//! generator: `<total> = sum(<name>)`, a bare name argument with no
//! per-element transform (`recognize_sum_over_name`). It lowers to the
//! identical `loopAccum` program with the per-trip effect fixed to the
//! element slot itself, and it is the one form whose total sort is
//! known outright — the sequence's own element sort, with no per-trip
//! expression to widen it.
//!
//! Everything else declines to the paths that already existed,
//! unchanged — including `sum([...])` over a list comprehension, which
//! the eager path already materializes.

mod division;
mod lowering;
mod recognize;
mod walk;

#[cfg(test)]
mod tests;

pub use division::division_range_in;
pub use division::fold_division;
pub use division::fold_located_division;
pub use division::is_length_alias_assignment;
pub use division::record_length_alias;
pub use division::reassigns_alias_or_sequence;
pub use division::DivisionOp;

pub use recognize::recognize_accumulation;
pub use recognize::recognize_generator_sum;
pub use recognize::recognize_generator_sum_in_return;
pub use recognize::recognize_sum_over_name;

pub use walk::walk_accumulation;
pub use walk::AccumulationAnswer;

#[cfg(test)]
pub(self) use lowering::is_same_name_square;
#[cfg(test)]
pub(self) use walk::bindable_state;
#[cfg(test)]
pub(self) use walk::element_and_count_sets;
#[cfg(test)]
pub(self) use walk::quotient_kind_tag;

use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::trust_grades::TrustLevel;
use refined_kernel::loop_questions::IrStatement;
use refined_kernel::loop_questions::LoopEffect;
use refined_kernel::loop_questions::LoopEffectKind;
use refined_kernel::narrow_questions::KnownStateWire;
use refined_sets::refinement_forms::RefinedSet;

/// The slot the running total lives in.
pub(super) const TOTAL_SLOT: i64 = 0;
/// The slot the sequence's element abstraction lives in.
pub(super) const ELEMENT_SLOT: i64 = 1;
/// The slot the count lives in.
pub(super) const COUNT_SLOT: i64 = 2;
/// The slot a folded division writes its quotient into.
pub(super) const QUOTIENT_SLOT: i64 = 3;

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
    /// The total's language-level sort, when it is known exactly: a
    /// float seed (`total = 0.0`) makes every later `total += _` a
    /// float — float absorbs each numeric add — so Float is exact. An
    /// int seed or the seedless generator-sum spelling states nothing
    /// (`None`): the elements' sort could go either way and no correct
    /// per-element sort survives the repetition-window read. The bare
    /// `sum(<name>)` spelling (`recognize_sum_over_name`) is the one
    /// path that states Integer: with no per-element expression to
    /// widen the sort, the total's sort is exactly the sequence's own
    /// element sort, read off `<name>`'s own `kind_tag`
    /// (`sum_call_over_star`'s own reading of the same field).
    pub total_kind_tag: Option<PrimitiveKind>,
    /// Which division operator folded, when one did — `None` until
    /// `fold_division`/`fold_located_division` runs. Read by
    /// `walk_accumulation` to pick the quotient's own sort: `Div` is
    /// always Float (Python's `/` "division of integers yields a
    /// float", expressions.rst); `FloorDiv` is Integer exactly when
    /// both the total and the count are Integer-sorted, Float
    /// otherwise (Python's `//` on any Float operand yields Float,
    /// same clause).
    pub quotient_op: Option<DivisionOp>,
    /// Names proved to equal `len(<sequence_name>)` by a plain
    /// assignment sitting between the accumulation and the division —
    /// `count = len(samples)`, then `total / count` — keyed by the
    /// alias name, every value the accumulation's OWN `sequence_name`
    /// (there is only ever one sequence per accumulation, so the value
    /// is redundant with the key's own binding, but is spelled out
    /// rather than implied, matching `same_length_as`'s own spelling).
    /// `is_len_of` consults this for a bare-name numerator's divisor;
    /// `record_length_alias` is the only writer. Populated by the
    /// caller's own one-hop scan (`is_length_alias_assignment`) — this
    /// module never walks a statement list itself to fill it.
    pub length_aliases: std::collections::HashMap<String, String>,
    /// The sequence's own count set, read once by `element_and_count_sets`
    /// when the accumulation was recognized — the same nonnegative,
    /// integer-sorted window `entry_states`'s slot 2 carries into the
    /// kernel ask. `walk_accumulation` binds a count-alias name
    /// (`count = len(samples)`) to exactly this set: the length is
    /// already known here, from the sequence's own repetition window,
    /// so a count-alias's value is this field re-read, never a fresh
    /// derivation.
    pub count_set: RefinedSet,
}

/// A read of one slot.
pub(super) fn slot(index: i64) -> LoopEffect {
    LoopEffect {
        kind: LoopEffectKind::Var,
        index,
        ..Default::default()
    }
}

/// A plain number state: a set, admitting neither absence nor NaN nor
/// a thrown exit.
pub(super) fn number_state(set: RefinedSet) -> KnownStateWire {
    KnownStateWire {
        top: false,
        set,
        undef: false,
        null: false,
        nan: false,
        thrown: false,
    }
}
