//! One arm's own TAKEN/NOT-TAKEN/UNDECIDABLE verdict against a known
//! subject — `arm_outcome`, its guard application, and the recursive
//! `pattern_outcome` core every pattern shape (including the
//! `Kind::KindUnion` per-arm judge) dispatches through.

use std::sync::Arc;

use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_kernel::kernel_interface::RefinedTSKernel;
use ruff_python_ast::Expr;
use ruff_python_ast::Pattern;
use ruff_python_ast::Singleton;

use crate::env::Environment;
use crate::expressions::evaluate_expression;

use super::value_proof::exact_string_values;
use super::value_proof::single_numeric_value;
use super::values::enumerable_numeric_members;

/// What a match arm's pattern (and, where present, its guard) decided
/// about a known subject.
pub enum ArmOutcome {
    /// The arm is taken: `Environment` is a fork of the arm's incoming
    /// environment with every capture the pattern makes bound.
    Taken(Environment),
    /// The arm is provably not taken — the pattern (or its guard) is
    /// known false for this subject.
    NotTaken,
    /// The walk cannot decide this arm — an unknown subject, an
    /// unmodeled pattern shape (sequence/mapping/class this wave), or a
    /// guard whose truth this file cannot read.
    Undecidable,
}

/// Decide one arm: `pattern` against `subject`, then (if the pattern
/// took) `guard` against the arm's own environment. `environment` is
/// the environment the arm's fork starts from — the caller's job to
/// fork per case, not this function's (so a caller walking several
/// arms in sequence controls its own forking/joining).
pub fn arm_outcome(
    pattern: &Pattern,
    guard: Option<&Expr>,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    match pattern_outcome(pattern, subject, environment, kernel) {
        ArmOutcome::Taken(arm_env) => apply_guard(guard, arm_env, kernel),
        other => other,
    }
}

/// After a pattern is Taken, run its guard (if any) via
/// `evaluate_expression` over the arm's own environment (the pattern's
/// captures are already bound, so the guard reads them). A guard that
/// evaluates to a known boolean decides the arm outright; anything else
/// is Undecidable — the caller (the walk) must then treat every LATER
/// arm as Undecidable too, since CPython only reaches a later case when
/// this guard is known false, and this file cannot prove that. This
/// poisoning rule is `match_taken_environment`'s job to enforce, not
/// this function's; this function reports only its OWN arm's verdict.
fn apply_guard(guard: Option<&Expr>, arm_env: Environment, kernel: &Arc<RefinedTSKernel>) -> ArmOutcome {
    let Some(guard) = guard else {
        return ArmOutcome::Taken(arm_env);
    };
    let guard_value = evaluate_expression(guard, &arm_env, kernel);
    match known_boolean(&guard_value) {
        Some(true) => ArmOutcome::Taken(arm_env),
        Some(false) => ArmOutcome::NotTaken,
        None => ArmOutcome::Undecidable,
    }
}

/// The known boolean a guard's evaluated value carries, if it carries
/// exactly one. A Boolean-tagged single value reads directly; a
/// Number-tagged single value reads through Python's truthiness rule
/// for numbers (nonzero is true) since a guard is `if`-tested, not
/// `==`-compared. Anything else (unknown, a set, no single value) is
/// not a known boolean.
fn known_boolean(value: &AbstractValue) -> Option<bool> {
    if value.kind != Kind::Values || value.values.len() != 1 {
        return None;
    }
    match value.kind_tag {
        Some(PrimitiveKind::Boolean)
        | Some(PrimitiveKind::Number)
        | Some(PrimitiveKind::Integer)
        | Some(PrimitiveKind::Float) => Some(value.values[0] != 0.0),
        _ => None,
    }
}

/// One pattern's own outcome against `subject`, with no guard
/// considered — the recursive core `arm_outcome`, `MatchAs`, and
/// `MatchOr` all share. `kernel` is threaded through only because
/// `evaluate_expression` (read by `match_value_outcome` for a pattern's
/// own literal expression) requires one by contract; no pattern shape
/// this wave decides asks the kernel a question.
pub(super) fn pattern_outcome(
    pattern: &Pattern,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    if subject.kind == Kind::KindUnion {
        return kind_union_pattern_outcome(pattern, subject, environment, kernel);
    }
    match pattern {
        Pattern::MatchValue(value_pattern) => match_value_outcome(value_pattern, subject, environment, kernel),
        Pattern::MatchSingleton(singleton_pattern) => match_singleton_outcome(singleton_pattern, subject, environment),
        Pattern::MatchAs(as_pattern) => match_as_outcome(as_pattern, subject, environment, kernel),
        Pattern::MatchOr(or_pattern) => match_or_outcome(or_pattern, subject, environment, kernel),
        // Sequence/Mapping/Class patterns: this wave carries no
        // container state on AbstractValue (element_set/keys exist for
        // other purposes but this file does not read into them for
        // structural matching), so every one of these declines rather
        // than assume a shape the subject may not have.
        Pattern::MatchSequence(_) => ArmOutcome::Undecidable,
        Pattern::MatchMapping(_) => ArmOutcome::Undecidable,
        Pattern::MatchClass(_) => ArmOutcome::Undecidable,
        // MatchStar only appears nested inside a MatchSequence's own
        // pattern list (`case [*rest]:`), never as a case's top-level
        // pattern per the grammar (closed_pattern excludes it) — reached
        // only if a caller hands this function a sub-pattern directly,
        // which match_taken_environment never does.
        Pattern::MatchStar(_) => ArmOutcome::Undecidable,
    }
}

/// A `Kind::KindUnion` subject (`json.loads`'s own honest return-space
/// union, `expressions.rs::json_loads_value_space`, is the one producer
/// today) — the pattern is judged against EACH arm through this same
/// recursive `pattern_outcome` core, the per-arm reading
/// `assignability.rs`'s KindUnion judge and `narrowing.rs`'s
/// `narrow_isinstance_call` KindUnion filter both already use for their
/// own questions, applied here to a match pattern's TAKEN/NOT-TAKEN
/// question instead of an assignability Fire or an isinstance filter.
/// The union claims the runtime subject is SOME arm, never which one:
/// an arm the pattern proves Undecidable makes the whole union
/// Undecidable (this file cannot rule out that arm being the real
/// runtime value); with no Undecidable arm, ANY arm the pattern takes
/// makes the union Taken (that arm's own environment is a sound fork —
/// CPython runs the arm whenever the runtime value happens to be that
/// arm, so the union subject can genuinely reach this case); every arm
/// NotTaken is the whole union NotTaken (no possible runtime shape ever
/// reaches this pattern).
fn kind_union_pattern_outcome(
    pattern: &Pattern,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    for arm in &subject.arms {
        if let ArmOutcome::Undecidable = pattern_outcome(pattern, arm, environment, kernel) {
            return ArmOutcome::Undecidable;
        }
    }
    for arm in &subject.arms {
        if let ArmOutcome::Taken(arm_env) = pattern_outcome(pattern, arm, environment, kernel) {
            return ArmOutcome::Taken(arm_env);
        }
    }
    ArmOutcome::NotTaken
}

/// `MatchValue` — "`LITERAL` will succeed only if `<subject> ==
/// LITERAL`." Decided when `subject` ENUMERATES its admitted numeric
/// members (`enumerable_numeric_members` — a single known scalar, a
/// multi-valued `Kind::Values` set such as `{1, 2, 4}`, or a `Kind::Set`
/// that enumerates a union-of-singletons form) and the pattern's own
/// evaluated expression (via `evaluate_expression`, which forces the
/// walk's OWN literal rules) is a known single numeric/boolean value —
/// or BOTH sides are known exact strings — the two `==` rows CPython's
/// own equality actually reaches for a `MatchValue` pattern
/// (expressions.rst, "Comparisons": numeric types compare by
/// mathematical value, strings compare by their code-point sequence).
/// Numeric: `1 == True` and `1 == 1.0` both hold, so a Number-tagged
/// subject admitting 1 DOES take `case 1:` and `case True:`'s VALUE
/// reading would too if it appeared as a MatchValue (it never does —
/// `True` always parses as MatchSingleton, never MatchValue). The
/// pattern's own literal being a MEMBER of the subject's admitted set
/// is Taken; not a member is NotTaken (a dead arm no runtime value of
/// this subject can ever reach — the same NotTaken every other
/// unreachable arm answers). String: `case "left":` against a
/// String-tagged subject compares the code-point vectors directly, the
/// same reading `expressions.rs::exact_string_values` gives an ordinary
/// `==` comparison — this is what lets `anchor_of`'s own `match o: case
/// "left": ...` decide its arm for a concrete `Literal["left", ...]`
/// argument instead of falling through to the undecided join over every
/// arm. A subject/pattern pair that is neither both-numeric nor
/// both-string (or one side unknown, or the subject's set does not
/// enumerate) is Undecidable.
fn match_value_outcome(
    value_pattern: &ruff_python_ast::PatternMatchValue,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    let literal_value = evaluate_expression(&value_pattern.value, environment, kernel);
    if let (Some(subject_members), Some(pattern_value)) =
        (enumerable_numeric_members(subject), single_numeric_value(&literal_value))
    {
        return if subject_members.iter().any(|member| *member == pattern_value) {
            ArmOutcome::Taken(environment.fork())
        } else {
            ArmOutcome::NotTaken
        };
    }
    if let (Some(subject_text), Some(pattern_text)) =
        (exact_string_values(subject), exact_string_values(&literal_value))
    {
        return if subject_text == pattern_text {
            ArmOutcome::Taken(environment.fork())
        } else {
            ArmOutcome::NotTaken
        };
    }
    // `None`/a dict/a list is a STRUCTURAL SORT MISMATCH against any
    // scalar (numeric or string) `MatchValue` literal — neither is ever
    // `==` a number or a string, the same "never a member of a scalar
    // set" reading `assignability.rs::judge`'s own Null/Object/List
    // rows give a declared scalar refinement. This is a definite
    // NotTaken, not Undecidable: it is what lets a `Kind::KindUnion`'s
    // own None/list/dict arms (`kind_union_pattern_outcome`) drop out
    // of a numeric pattern's union judgment instead of poisoning it.
    if is_structural_scalar_mismatch(subject) && (single_numeric_value(&literal_value).is_some() || exact_string_values(&literal_value).is_some())
    {
        return ArmOutcome::NotTaken;
    }
    ArmOutcome::Undecidable
}

/// Whether `subject` is a KNOWN kind that can never be `==` a scalar
/// (a number or a string) — `Kind::Null` (`None`), `Kind::Object` (a
/// dict), and `Kind::List` (a list/tuple). The same three kinds
/// `assignability.rs::judge` fires outright against a declared scalar
/// refinement, read here for a `MatchValue` pattern's own `==` question
/// instead of an assignability question.
fn is_structural_scalar_mismatch(subject: &AbstractValue) -> bool {
    matches!(subject.kind, Kind::Null | Kind::Object | Kind::List)
}

/// `MatchSingleton` — "For the singletons `None`, `True` and `False`,
/// the `is` operator is used": identity, not equality. A
/// Boolean-tagged 1.0/0.0 subject IS `True`/`False`; a Number-tagged 1
/// subject is NOT `True` (AGENT-BRIEF.md's pinned fact) because CPython
/// identity distinguishes the `bool` singletons from the `int` they
/// happen to equal. `Kind::Null` (this crate's representation of
/// Python's `None`, per expressions.rs's NoneLiteral reading) IS
/// `None`; nothing else is.
fn match_singleton_outcome(
    singleton_pattern: &ruff_python_ast::PatternMatchSingleton,
    subject: &AbstractValue,
    environment: &Environment,
) -> ArmOutcome {
    match subject_is_singleton(subject, singleton_pattern.value) {
        Some(true) => ArmOutcome::Taken(environment.fork()),
        Some(false) => ArmOutcome::NotTaken,
        None => ArmOutcome::Undecidable,
    }
}

/// Whether `subject` IS `target` under Python's `is` identity — the one
/// place this file decides singleton identity, so every future kind
/// AbstractValue gains (an eventual Integer/Float split) only needs
/// this function taught about it. `None` means "not decidable" (an
/// unknown or otherwise unread subject); `Some(true)`/`Some(false)` is
/// the decided identity.
fn subject_is_singleton(subject: &AbstractValue, target: Singleton) -> Option<bool> {
    match target {
        Singleton::None => Some(subject.kind == Kind::Null),
        Singleton::True => Some(is_exact_boolean(subject, 1.0)),
        Singleton::False => Some(is_exact_boolean(subject, 0.0)),
    }
}

/// Whether `subject` is a known single Boolean-tagged value equal to
/// `want` (1.0 for True, 0.0 for False) — the only shape `is True` /
/// `is False` can affirm under this domain's current representation.
/// Any other kind (Number-tagged, multi-valued, unknown, …) is not this
/// exact singleton, matching CPython's `is` refusing to unify `bool`
/// with a same-valued `int`.
fn is_exact_boolean(subject: &AbstractValue, want: f64) -> bool {
    subject.kind == Kind::Values
        && subject.kind_tag == Some(PrimitiveKind::Boolean)
        && subject.values.len() == 1
        && subject.values[0] == want
}

/// `MatchAs` — a bare capture (`case x:`, `pattern: None`) or a wildcard
/// (`case _:`, `pattern: None, name: None`) "always succeeds"; a
/// capture binds the subject to its name. A subpattern
/// (`case <pattern> as x:`) recurses on the subpattern first, and only
/// on Taken does it also bind the alias name — a wildcard/capture
/// nested under `as` still always succeeds by the same rule, so the
/// only way this arm is NotTaken/Undecidable is through the subpattern.
fn match_as_outcome(
    as_pattern: &ruff_python_ast::PatternMatchAs,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    let Some(inner_pattern) = as_pattern.pattern.as_deref() else {
        // bare `case x:` or wildcard `case _:` — always succeeds
        let mut arm_env = environment.fork();
        if let Some(name) = as_pattern.name.as_ref() {
            arm_env.bind(name.id.as_str(), subject.clone());
        }
        return ArmOutcome::Taken(arm_env);
    };
    match pattern_outcome(inner_pattern, subject, environment, kernel) {
        ArmOutcome::Taken(mut arm_env) => {
            if let Some(name) = as_pattern.name.as_ref() {
                arm_env.bind(name.id.as_str(), subject.clone());
            }
            ArmOutcome::Taken(arm_env)
        }
        other => other,
    }
}

/// `MatchOr` — "matches each of its subpatterns in turn to the subject
/// value, until one succeeds": first Taken wins (left to right); every
/// alternative NotTaken means the whole pattern is NotTaken; any
/// Undecidable alternative encountered BEFORE a Taken one makes the
/// whole pattern Undecidable (this file cannot rule out that an
/// earlier, undecided alternative would have matched first).
fn match_or_outcome(
    or_pattern: &ruff_python_ast::PatternMatchOr,
    subject: &AbstractValue,
    environment: &Environment,
    kernel: &Arc<RefinedTSKernel>,
) -> ArmOutcome {
    for alternative in &or_pattern.patterns {
        match pattern_outcome(alternative, subject, environment, kernel) {
            ArmOutcome::Taken(arm_env) => return ArmOutcome::Taken(arm_env),
            ArmOutcome::Undecidable => return ArmOutcome::Undecidable,
            ArmOutcome::NotTaken => continue,
        }
    }
    ArmOutcome::NotTaken
}
