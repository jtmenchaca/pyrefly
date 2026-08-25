//! The statement walk that judges values against stated sets. Every
//! body gets its own environment, seeded with the names the body
//! itself binds (so a module-level alias name goes dark inside a body
//! that rebinds it — Python's whole-body scoping rule). The walk
//! dispatches on every statement form; a construct it cannot yet walk
//! is the body's blocker — recorded once, as an RTS7002 finding naming
//! the construct in place — and the walk keeps going conservatively
//! (forgetting names it cannot account for) so later determinable rows
//! still judge. The membership question `x ∈ A` always goes to the
//! proved kernel (memberB_iff), never decided host-side; every
//! value-against-set judgment routes through `assignability::judge` so
//! fire wording and undetermined sentences stay uniform across every
//! sink (AnnAssign, plain Assign, return, aug-assign, class field). A
//! write sink that Fires never binds the refused value — `judge_and_bind`
//! is the refused-write law: the slot keeps its DECLARED SET afterward,
//! so a later read judges the declaration against itself (always
//! silent) rather than firing a second time for the same refusal.

mod body;
mod receivers;
mod scope;
mod statement;

pub(in crate::check) use body::{
    foreign_edge_consumer_position, serve_foreign_edge_at, walk_body, walk_body_with_self_binding,
};
pub(in crate::check) use receivers::{
    forget_mutated_receivers_in_body,
    forget_mutated_receivers_in_stmt, forget_names_bound_by_stmt, forget_names_bound_in_body,
    forget_target_from_provably_unbound, forget_target_names,
};
pub(in crate::check) use scope::{
    bind_walrus_targets, collect_bound_names, collect_bound_names_stmt, collect_parameter_names,
    collect_walrus_names, locally_bound_names, statement_kind_name,
};
pub(in crate::check) use statement::{record_blocker, walk_statement};
