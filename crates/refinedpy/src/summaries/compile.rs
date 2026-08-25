/*
 * Copyright (c) TypeRefinery.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;
use std::sync::Mutex;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::lattice_operations::set_of_known;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::narrow_questions::KnownStateWire;
use refined_kernel::summary_questions::ask_apply_summary;
use refined_kernel::summary_questions::ask_summarize;
use refined_kernel::summary_questions::SummaryBlob;
use refined_sets::refinement_forms::fold_ray_forms;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::Refinement;
use ruff_python_ast::Expr;
use ruff_python_ast::StmtFunctionDef;

use crate::summary_lowering::lower_function_body;
use crate::summary_lowering::LoweredBody;

/// The identity a compiled summary is stored under: the module the def
/// was parsed from, the def's name, and its own span in that source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) struct SummaryKey {
    module: String,
    name: String,
    start: u32,
    end: u32,
}

pub(super) fn summary_key(def: &StmtFunctionDef, module: &str) -> SummaryKey {
    SummaryKey {
        module: module.to_owned(),
        name: def.name.id.as_str().to_owned(),
        start: def.range.start().to_u32(),
        end: def.range.end().to_u32(),
    }
}

/// One `def`'s compiled answer: the blob the kernel wrote, beside the
/// slot bookkeeping a call site reads its entry states and its result
/// out of. A `None` entry is a REMEMBERED decline — a body that failed
/// to lower, or a compile the kernel refused, answers `None` forever
/// rather than paying the lowering again at every call.
pub(super) struct CompiledSummary {
    blob: SummaryBlob,
    lowered: LoweredBody,
}

/// One entry per `def` asked about: the compiled answer, or `None` for a
/// remembered decline.
type SummaryStore = std::collections::HashMap<SummaryKey, Option<Arc<CompiledSummary>>>;

/// SUMMARY_REGISTRY holds the finished answers; SUMMARY_BUILDING holds
/// the keys whose build is in flight — the cycle guard. A body whose own
/// lowering re-enters itself is a recursive `def`; it answers `None`
/// WITHOUT storing a decline, so the outer build's real answer is what
/// lands in the store.
///
/// (The lowering reaches no callee today — a call declines the body —
/// so the guard fires only on a re-entry through the apply path. It is
/// here because the registry, not its current caller, is where the
/// invariant lives.)
pub(super) static SUMMARY_REGISTRY: Mutex<Option<SummaryStore>> = Mutex::new(None);
static SUMMARY_BUILDING: Mutex<Option<std::collections::HashSet<SummaryKey>>> = Mutex::new(None);

/// `def`'s compiled summary, building it on the first ask and storing
/// the answer — hit or decline — under the key.
pub(super) fn compiled_summary_for(def: &StmtFunctionDef, module: &str) -> Option<Arc<CompiledSummary>> {
    let key = summary_key(def, module);
    {
        let registry = SUMMARY_REGISTRY.lock().expect("summary registry lock poisoned");
        if let Some(held) = registry.as_ref().and_then(|map| map.get(&key)) {
            return held.clone();
        }
    }
    {
        let mut building = SUMMARY_BUILDING.lock().expect("summary build lock poisoned");
        let in_flight = building.get_or_insert_with(Default::default);
        if !in_flight.insert(key.clone()) {
            // a re-entry while this def's own build is running: answer
            // nothing, and store nothing, so the outer build's real
            // answer is the one that lands
            return None;
        }
    }
    // the build runs OUTSIDE the registry lock — a lowering that reached
    // a callee would take the same lock for it
    let built = build_summary(def);
    {
        let mut building = SUMMARY_BUILDING.lock().expect("summary build lock poisoned");
        if let Some(in_flight) = building.as_mut() {
            in_flight.remove(&key);
        }
    }
    let mut registry = SUMMARY_REGISTRY.lock().expect("summary registry lock poisoned");
    registry
        .get_or_insert_with(Default::default)
        .insert(key, built.clone());
    built
}

/// Lowers `def`'s body and hands it to the kernel's compiler exactly
/// once. `None` where the body leaves the lowering's grammar, or where
/// the kernel refuses the compile.
fn build_summary(def: &StmtFunctionDef) -> Option<Arc<CompiledSummary>> {
    let lowered = lower_function_body(def)?;
    // ARITY is the WHOLE slot count, not the parameter count: the
    // compiler numbers one entry state per binding, and the apply side
    // sends one state per slot — the two numberings must agree or every
    // local's read collapses onto entry 0.
    let blob = ask_summarize(lowered.slot_count as i64, &lowered.statements, &[])?;
    Some(Arc::new(CompiledSummary { blob, lowered }))
}

/// The compiled summary applied to one call's own arguments, or `None`
/// wherever this route cannot serve — which is always a fall-through to
/// the interpreter, never a claim.
///
/// The declines, each of them a fall-through:
///
/// - the body does not lower, or the kernel refused the compile;
/// - the call passes a different number of arguments than the def
///   declares parameters (the entry vector has no place to put the
///   difference, and this route reads no defaults);
/// - an argument's value has no state the wire spells
///   (`entry_state_of`);
/// - the kernel refuses the application, or answers a short exit row;
/// - the exit says nothing (a TOP result), which is not an answer;
/// - a path may fall off the end without returning, so the value is
///   sometimes the result and sometimes `None` — a shape this route
///   does not spell, and the interpreter reads exactly.
pub(super) fn kernel_summary_result(
    def: &StmtFunctionDef,
    module: &str,
    arguments: &[AbstractValue],
) -> Option<AbstractValue> {
    let compiled = compiled_summary_for(def, module)?;
    let lowered = &compiled.lowered;
    if arguments.len() != lowered.parameter_count {
        return None;
    }
    let mut entries: Vec<KnownStateWire> = Vec::with_capacity(lowered.slot_count);
    for argument in arguments {
        entries.push(entry_state_of(argument)?);
    }
    // every slot past the parameters is a local, which enters ABSENT —
    // it holds no value until the body writes one
    while entries.len() < lowered.slot_count {
        entries.push(absent_entry_state());
    }
    // the done flag enters exactly "not yet returned"
    entries[lowered.done_index] = flag_down_entry_state();
    let exits = ask_apply_summary(&compiled.blob, &entries)?;
    if lowered.ret_index >= exits.len() || lowered.done_index >= exits.len() {
        return None;
    }
    let done_exit = &exits[lowered.done_index];
    // EVERY PATH RETURNED, or the value is sometimes the result and
    // sometimes `None` — the interpreter's own fall-through join reads
    // that case, and this route declines it rather than answering the
    // returned half alone.
    if done_exit.top || done_exit.undef || done_exit.null || !flag_is_definitely_up(done_exit) {
        return None;
    }
    // the RETURNED half of the result slot: what the runs that COMPLETED
    // left there, which the kernel proves admits every non-thrown outcome
    // (`returned_denotes`)
    let returned = exits[lowered.ret_index].returned();
    let value = value_of_exit_state(&returned)?;
    let sort = declared_return_sort(def).or_else(|| argument_numeric_sort(arguments));
    // A `Kind::Set` answer's `kind_tag` genuinely has no requirement — the
    // existing "unstated sort leaves the answer untagged" reading applies
    // exactly as before, whether or not `sort` found evidence.
    if value.kind != Kind::Values {
        return Some(AbstractValue { kind_tag: sort, ..value });
    }
    // A `Kind::Values` answer is a FRESH Python-sorted read (this route's
    // own exact-scalar folding, just proved by the kernel), and
    // `PrimitiveKind`'s own doc is explicit that such a read "always tags
    // Integer or Float, never bare Number" — `Number` is reserved for a
    // JOINED or otherwise-undetermined sort, neither of which applies to
    // one value this route just derived outright. Lacking real evidence
    // for which of the two this is (the wire carries no int/float
    // distinction of its own — `KnownStateWire` is extended-reals only),
    // this route declines rather than manufacture the placeholder `Number`
    // tag `value_of_exit_state` set as its own internal default; the
    // interpreter's fall-through reads the same literal concretely and
    // tags it correctly from the source.
    let Some(sort) = sort else {
        return None;
    };
    Some(AbstractValue { kind_tag: Some(sort), ..value })
}

/// The one numeric `PrimitiveKind` every ARGUMENT this call passed
/// agrees on, or `None` where they disagree (or there are none to read).
/// Read only when `def` states no return annotation of its own
/// (`declared_return_sort`'s own `None`): the lowering's arithmetic is
/// total-or-decline over the arguments' own entry states (this file's
/// module doc — the compile reaches no callee, no defaulted parameter,
/// nothing that could introduce a DIFFERENT sort mid-body), so a body
/// that compiles at all carries its answer's sort forward from its
/// arguments exactly the way CPython's own `int + int -> int` /
/// `float + anything -> float` arithmetic does — this is a DERIVATION
/// from a concretely-sorted input, never the blind guess
/// `declared_return_sort`'s own doc warns against for an unstated
/// annotation. `Integer` wins only when every argument is Integer-tagged;
/// any Float-tagged argument makes the whole answer Float-tagged
/// (Python's own float-contagion rule); anything else (a bare `Number`
/// tag, a `Boolean` tag, no arguments at all, or disagreement) answers
/// `None`, leaving the result untagged exactly as before this reading
/// existed.
fn argument_numeric_sort(arguments: &[AbstractValue]) -> Option<PrimitiveKind> {
    let mut sort: Option<PrimitiveKind> = None;
    for argument in arguments {
        let tag = argument.kind_tag?;
        match tag {
            PrimitiveKind::Float => return Some(PrimitiveKind::Float),
            PrimitiveKind::Integer => {
                if sort.is_none() {
                    sort = Some(PrimitiveKind::Integer);
                }
            }
            _ => return None,
        }
    }
    sort
}

/// The SORT the `def` declares its return to be, read from its own
/// annotation — `int` and `float` and nothing else, the same two numeric
/// names `return_sort_fallback` reads.
///
/// The compiled summary answers a SET of real numbers and carries no
/// sort of its own: the kernel decides membership on the real line and
/// never holds this checker's int/float tags. A `def` that states its
/// sort supplies it here; one that does not leaves the answer untagged,
/// which the assignability laws read as numeric-sorted and never as
/// float-sorted, so an unstated sort costs a fire that the tag would
/// have caught and never claims one it should not.
fn declared_return_sort(def: &StmtFunctionDef) -> Option<PrimitiveKind> {
    let Expr::Name(sort) = def.returns.as_deref()? else {
        return None;
    };
    match sort.id.as_str() {
        "int" => Some(PrimitiveKind::Integer),
        "float" => Some(PrimitiveKind::Float),
        _ => None,
    }
}

/// One argument's value as the entry state the wire carries, or `None`
/// when this domain's value has no faithful state — the call then falls
/// through to the interpreter rather than entering on a fabricated one.
///
/// What crosses: a scalar value list (`Kind::Values` over a numeric
/// sort), an untagged numeric `Kind::Set` (`set_of_known`'s own reading —
/// the one set reader this file shares with every other kernel question),
/// and the two absent admissions. A STRING-sorted value does not cross:
/// the lowering's arithmetic reads its slots numerically, so a word
/// entering one of them would be reread across sorts.
///
/// Everything else — an object, a list, a collection, a promise, an
/// unknown — has no state this wire spells, and answers `None`.
pub(super) fn entry_state_of(argument: &AbstractValue) -> Option<KnownStateWire> {
    match argument.kind {
        Kind::Values => {
            if !matches!(
                argument.kind_tag,
                Some(PrimitiveKind::Number)
                    | Some(PrimitiveKind::Integer)
                    | Some(PrimitiveKind::Float)
                    | Some(PrimitiveKind::Boolean)
            ) {
                return None;
            }
            // a value LIST is the scalar set of those values —
            // `one_of([a, b])`, never `set_of_known`'s tuple
            // concatenation, which spells a SEQUENCE of them
            Some(KnownStateWire {
                top: false,
                set: make_refined_set(vec![one_of(&argument.values)]),
                undef: false,
                null: false,
                nan: false,
                thrown: false,
            })
        }
        Kind::Set => {
            // a WORN set's members are not doubles, and `set_of_known`
            // already refuses one; a string-tagged set would cross the
            // sort line, so it is refused here
            if argument.set_kind_tag != SetKindTag::None
                || argument.kind_tag == Some(PrimitiveKind::String)
            {
                return None;
            }
            let set = set_of_known(argument)?;
            Some(KnownStateWire {
                top: false,
                set,
                undef: false,
                null: false,
                nan: false,
                thrown: false,
            })
        }
        Kind::Null => Some(KnownStateWire {
            top: false,
            set: make_refined_set(vec![one_of(&[])]),
            undef: false,
            null: true,
            nan: false,
            thrown: false,
        }),
        Kind::Undef => Some(absent_entry_state()),
        _ => None,
    }
}

/// The definitely-absent entry state: no value at all. Every slot past
/// the parameters enters holding this, since a local holds nothing until
/// the body writes it.
fn absent_entry_state() -> KnownStateWire {
    KnownStateWire {
        top: false,
        set: make_refined_set(vec![one_of(&[])]),
        undef: true,
        null: true,
        nan: false,
        thrown: false,
    }
}

/// The done flag's own entry state: exactly `{0}`, "not yet returned."
fn flag_down_entry_state() -> KnownStateWire {
    KnownStateWire {
        top: false,
        set: make_refined_set(vec![one_of(&[0.0])]),
        undef: false,
        null: false,
        nan: false,
        thrown: false,
    }
}

/// Whether the done flag's EXIT admits only the raised value — every
/// path through the body returned. The set is read as an intersection of
/// forms, so a shape this reader cannot judge answers false, which costs
/// this route a serving and never claims one.
fn flag_is_definitely_up(exit: &KnownStateWire) -> bool {
    exit.set
        .forms
        .iter()
        .any(|form| form.form == Form::OneOf && form.w.len() == 1 && form.w[0] == 1.0)
}

/// The folded form list's own exact scalar list, when the fold landed on
/// one finite, non-empty `OneOf` — the one set shape whose canonical
/// spelling is `Kind::Values`, matching `intersect_refinements.rs`'s own
/// `exact_scalar_values` reading of a narrow's folded intersection
/// (private there, so this file — the kernel-summary route's own owner —
/// carries the identical reading rather than reaching across the crate
/// boundary for it).
fn exact_scalar_values(forms: &[Refinement]) -> Option<Vec<f64>> {
    if forms.len() == 1 && forms[0].form == Form::OneOf && !forms[0].w.is_empty() {
        return Some(forms[0].w.clone());
    }
    // The ARITHMETIC transfer's own exact-point spelling: a non-strict
    // `AtLeast(v)` paired with a non-strict `AtMost(v)` at the SAME bound
    // pins the set to exactly `{v}` — real arithmetic composes ray forms
    // rather than folding all the way back down to a `OneOf`, so
    // `double(3) == 6` exits as `[AtLeast(6), AtMost(6), Integer,
    // MultipleOf(2)]`, never a bare `OneOf(6)`. Every OTHER conjunct
    // alongside the pinned pair (`Integer`, `MultipleOf`, …) is a fact
    // about that same single point, already proved consistent by the
    // kernel's own derivation — this reading does not need to re-check
    // them, only find the pair that narrows the ray forms to one value.
    exact_point_of_ray_pair(forms)
}

/// The one point an `AtLeast`/`AtMost` ray pair pins, when both rays
/// share the same finite bound — `None` when no such matching pair
/// exists (an open range, a one-sided ray, two different bounds, or no
/// ray forms at all).
fn exact_point_of_ray_pair(forms: &[Refinement]) -> Option<Vec<f64>> {
    let lower = forms.iter().find(|f| f.form == Form::AtLeast)?;
    let upper = forms.iter().find(|f| f.form == Form::AtMost)?;
    if lower.a.is_infinite() || upper.a.is_infinite() || lower.a != upper.a {
        return None;
    }
    Some(vec![lower.a])
}

/// The result slot's exit state as this domain's value, or `None` where
/// the exit says nothing worth answering. A TOP exit is exactly "the
/// return value is unconstrained," which is what the interpreter would
/// have to derive for itself — so this route declines and lets it, rather
/// than serving a silence that would displace a real reading.
///
/// A folded exit that lands on one finite scalar list crosses as
/// `Kind::Values` at `TrustProved` — the same canonical spelling
/// `interpret_body`'s own concrete arithmetic would answer for `double(3)
/// == 6`, so a caller reading this route's answer never has to tell "the
/// kernel proved exactly 6" apart from "the interpreter computed exactly
/// 6." Every coarser exit (a real range, a union that never folds to one
/// point) crosses as `Kind::Set` at SPEC grade instead: that claim is the
/// kernel's own derivation over the entry states this call supplied, and
/// the entries carried the arguments' own sets rather than their grades,
/// so it can never overclaim PROVED for a fact only the kernel's
/// derivation step established.
fn value_of_exit_state(exit: &KnownStateWire) -> Option<AbstractValue> {
    if exit.top || exit.nan {
        return None;
    }
    if exit.undef || exit.null {
        // the value is sometimes absent: an admission this route's own
        // answer has no arm for, and the interpreter's join reads it
        return None;
    }
    if exit.set.forms.is_empty() {
        return None;
    }
    let folded = fold_ray_forms(&exit.set.forms);
    if let Some(values) = exact_scalar_values(&folded) {
        return Some(known_values(values, PrimitiveKind::Number, TrustProved));
    }
    Some(known_set(exit.set.clone(), None, TrustSpec, SetKindTag::None))
}
