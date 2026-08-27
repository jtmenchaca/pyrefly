//! Recording published node values, collected returns, and collected
//! evaluations for one walk — the opt-in sinks a caller (relational
//! sum, foreign-edge crossing, a hover/position query) installs to read
//! back what the walk computed, without changing any evaluator's own
//! signature.

use std::sync::Arc;
use std::sync::Mutex;

use refined_domain::abstract_value::AbstractValue;
use ruff_text_size::TextRange;

use super::Environment;

impl Environment {
    /// Publishes up to two expression nodes' already-computed values for
    /// the walk of a single statement (see the field's own doc). An
    /// empty `Vec` clears it, which every caller does once that
    /// statement is walked.
    pub fn set_evaluated_node(&mut self, evaluated: Vec<(TextRange, AbstractValue)>) {
        self.evaluated_node = evaluated;
    }

    /// The published value for the node at `range`, if one was set for
    /// this walk. Every other node reads `None` and evaluates normally.
    pub fn evaluated_node(&self, range: TextRange) -> Option<&AbstractValue> {
        self.evaluated_node
            .iter()
            .find(|(published, _)| *published == range)
            .map(|(_, value)| value)
    }

    /// Asks this body's walk to record every value its `return`
    /// statements produce (`returned_values`'s own doc). Called once,
    /// before the body walks; every fork made afterwards shares the one
    /// recorder.
    pub fn collect_returned_values(&mut self) {
        self.returned_values = Some(Arc::new(Mutex::new(Vec::new())));
    }

    /// Records one `return`'s value, when this walk was asked for them.
    /// A no-op otherwise, which is every ordinary walk.
    pub fn record_returned_value(&self, value: AbstractValue) {
        let Some(recorder) = self.returned_values.as_ref() else {
            return;
        };
        recorder
            .lock()
            .expect("returned-values recorder poisoned by an earlier panic")
            .push(value);
    }

    /// Every value this body's `return` statements produced, in walk
    /// order — an empty vector for a walk that recorded none, `None`
    /// for a walk that was never asked to record.
    pub fn returned_values(&self) -> Option<Vec<AbstractValue>> {
        Some(
            self.returned_values
                .as_ref()?
                .lock()
                .expect("returned-values recorder poisoned by an earlier panic")
                .clone(),
        )
    }

    /// Installs `recorder` as this environment's own evaluations sink —
    /// the SAME `Arc` a caller (`check.rs::refined_set_at_position`,
    /// through `WalkContext::evaluations_recorder`) already holds, so
    /// every write this body's walk makes lands in the one `Vec` the
    /// caller reads back once the whole module walk finishes. Unlike
    /// `collect_returned_values` (which mints a FRESH recorder scoped
    /// to one body), this shares an EXISTING one across every body the
    /// module walk reaches — the aggregation `refined_set_at_position`
    /// needs, since the asked-about position may sit inside any nested
    /// `def`'s own body, each of which builds its own fresh
    /// `Environment`.
    pub fn set_evaluations_recorder(&mut self, recorder: Arc<Mutex<Vec<(TextRange, AbstractValue)>>>) {
        self.evaluations = Some(recorder);
    }

    /// Records one expression node's own range and value, when this
    /// walk was asked to collect them. A no-op otherwise, which is
    /// every ordinary check.
    pub fn record_evaluation(&self, range: TextRange, value: AbstractValue) {
        let Some(recorder) = self.evaluations.as_ref() else {
            return;
        };
        recorder
            .lock()
            .expect("evaluations recorder poisoned by an earlier panic")
            .push((range, value));
    }

    /// Installs `collector` as this environment's derivation-trace sink —
    /// the SAME `Arc` the caller holds, shared across every fork and
    /// every nested body's own `Environment` for the fork-blind reason
    /// the field's own doc states. `walk_body_with_self_binding` calls
    /// this the moment it builds a body's environment, exactly where it
    /// already calls `set_evaluations_recorder`.
    pub fn set_trace_collector(&mut self, collector: Arc<Mutex<crate::trace::TraceCollector>>) {
        self.trace = Some(collector);
    }

    /// This environment's trace collector, if a caller installed one.
    /// `None` for every ordinary check.
    pub fn trace_collector(&self) -> Option<&Arc<Mutex<crate::trace::TraceCollector>>> {
        self.trace.as_ref()
    }
}
