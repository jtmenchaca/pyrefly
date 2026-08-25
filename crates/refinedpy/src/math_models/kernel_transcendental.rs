use std::sync::Arc;

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustSpec;
use refined_kernel::kernel_interface::RefinedTSKernel;
use refined_kernel::transfer_questions::PowOperandKind;
use refined_kernel::transfer_questions::PowOperandWire;
use refined_kernel::transfer_questions::TransferAnswerKind;
use refined_kernel::transfer_questions::TransferQuestion;
use refined_kernel::transfer_questions::TransferQuestionOp;
use refined_sets::refinement_forms::above;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::RefinedSet;

use super::single_numeric_operand;

/// The operand a one-argument float transcendental question can be
/// posed over: a known single numeric value reads as the one-element
/// set `{v}` (the same "known value → singleton set" reading
/// `int_transferable_operand` performs for the `int` theory, widened
/// to every numeric sort since these questions are not integer-only),
/// and an already-numeric-sorted `Kind::Set` reads as its own set —
/// the same operand shape `sqrt_call_over_set`/`rounding_call_over_set`
/// pose, generalized to accept a known SINGLE value too (`transferExp`
/// and its siblings answer a bracketing window even for a singleton
/// operand, since none of this family is exactly computable at an
/// arbitrary interior point — the pins table's own "implementation-
/// approximated interior" note). A boolean-sorted or non-numeric
/// operand declines.
pub(super) fn float_transferable_operand(value: &AbstractValue) -> Option<RefinedSet> {
    if let Some((number, _)) = single_numeric_operand(value) {
        return Some(make_refined_set(vec![one_of(&[number])]));
    }
    if value.kind == Kind::Set
        && matches!(
            value.kind_tag,
            Some(PrimitiveKind::Integer)
                | Some(PrimitiveKind::Float)
                | Some(PrimitiveKind::Boolean)
                | Some(PrimitiveKind::Number)
        )
    {
        return Some(value.set.clone());
    }
    None
}

/// `math.sin`/`math.cos`/`math.tan` on a KNOWN INFINITE operand
/// provably raises `ValueError` rather than answering a value — the
/// SAME module-introduction "invalid operations" clause `sqrt_argument_
/// is_known_negative` cites (library/math.html's impl-detail note: "The
/// current implementation will raise ValueError for invalid operations
/// like sqrt(-1.0) or log(0.0)... following C99 Annex F"), extended to
/// the platform C `sin`/`cos`/`tan`'s own C99 Annex F domain error for
/// an infinite argument (Annex F.10.1.4/.6/.9: "sin/cos/tan(±∞) returns
/// a NaN and raises the 'invalid' floating-point exception" — CPython's
/// `mathmodule.c` turns that platform-signaled invalid operation into
/// `ValueError: math domain error`, the same translation `sqrt(-1.0)`
/// already goes through for its own domain error). Scoped to exactly
/// these three names — `sinh`/`cosh`/`tanh`/`asin`/`acos`/`atan`/
/// `atan2` do not share this domain restriction (the hyperbolic and
/// inverse trig families are each total, or gated by a different
/// clause already read elsewhere in this file), so widening past
/// `sin`/`cos`/`tan` needs its own citation, not an assumed extension.
/// A NaN operand is NOT read here: library/math.html's own note states
/// NaN propagates through as NaN rather than raising ("A NaN will not
/// be returned... unless one or more of the input arguments was a
/// NaN"), a different row this function does not answer for — only a
/// known FINITE-sorted operand that is actually infinite fires.
pub fn trig_argument_is_known_infinite(function: &str, arguments: &[AbstractValue]) -> bool {
    if !matches!(function, "sin" | "cos" | "tan") {
        return false;
    }
    let Some(first) = arguments.first() else {
        return false;
    };
    match single_numeric_operand(first) {
        Some((value, _)) => value.is_infinite(),
        None => false,
    }
}

/// `math.sqrt(x)` on a KNOWN NEGATIVE operand provably raises
/// `ValueError` rather than answering a value — library/math.html's own
/// module-introduction note: "The current implementation will raise
/// `ValueError` for invalid operations like `sqrt(-1.0)`..." A
/// negative operand is `provable_raise`'s own business (expressions.rs
/// calls this row through its own dispatch), not this function's — this
/// helper only reports WHETHER the operand is a known negative,
/// leaving the raise message's own wording to the caller that owns
/// `provable_raise`'s one voice.
pub fn sqrt_argument_is_known_negative(arguments: &[AbstractValue]) -> bool {
    let Some(first) = arguments.first() else {
        return false;
    };
    match single_numeric_operand(first) {
        Some((value, _)) => value < 0.0,
        None => false,
    }
}

/// `math.pow(x, y)` on TWO KNOWN operands provably raises `ValueError`
/// exactly when: both are finite, `x` is negative, and `y` is not an
/// integer — library/math.rst's own `pow(x, y)` clause, quoted verbatim:
/// "If both x and y are finite, x is negative, and y is not an integer
/// then pow(x, y) is undefined, and raises ValueError." (The clause's
/// own preceding sentence pins the two outcomes this predicate does NOT
/// need to gate for: `pow(1.0, x)` and `pow(x, 0.0)` always return
/// `1.0` "even when x is a zero or a NaN" — neither shape reaches a
/// negative, non-integer-exponent judgment at all, since `x == 1.0` or
/// `y == 0.0` are checked here BEFORE the negative-base test, matching
/// the doc's own precedence.) Both arguments must be known single
/// numeric values (`single_numeric_operand`, the same reading
/// `sqrt_argument_is_known_negative` performs) — an unknown or
/// Set-shaped operand answers `false`, the same "known operands only"
/// discipline this whole file keeps; `possible_raise`'s own row is
/// where a SET-shaped straddling operand belongs, not this one.
pub fn pow_arguments_provably_raise(arguments: &[AbstractValue]) -> bool {
    let [base_argument, exponent_argument] = arguments else {
        return false;
    };
    let Some((base, _)) = single_numeric_operand(base_argument) else {
        return false;
    };
    let Some((exponent, _)) = single_numeric_operand(exponent_argument) else {
        return false;
    };
    if !base.is_finite() || !exponent.is_finite() {
        return false;
    }
    if base == 1.0 || exponent == 0.0 {
        return false;
    }
    base < 0.0 && exponent.fract() != 0.0
}

/// The DOMAIN-LIMITED members of the kernel-backed family and the exact
/// window each one raises `ValueError` over in CPython — verified
/// against `tmp/cpython/Modules/mathmodule.c`, not against the kernel's
/// own JavaScript-facing `.nan` corner, because the two do NOT always
/// agree at the boundary:
///
/// - `log`/`log2`/`log10`: `loghelper` routes a float argument through
///   `math_1(arg, func, 0)` (`can_overflow = 0`). `m_log`/`m_log2`/
///   `m_log10` (mathmodule.c) return `-HUGE_VAL` (an INFINITE result)
///   at `x == 0.0` — a finite input — and `math_1`'s own rule ("an
///   infinite result from finite inputs causes... ValueError if
///   can_overflow is 0") fires there, so `math.log(0.0)` RAISES rather
///   than returning `-inf`. The kernel's `logCorners` answers the
///   value `-inf` at that same point (JavaScript's `Math.log(0) ===
///   -Infinity`) — a real JS/Python divergence at exactly one point.
///   The raise domain is therefore `x <= 0` (CLOSED at zero), one wider
///   than the kernel's own open `x < 0` NaN corner. Cited by
///   specifications/python/Doc/library/math.rst:696-698, whose own
///   worked example is `log(0.0)`.
/// - `log1p`: `FUNC1(log1p, m_log1p, 0, ...)` — same `can_overflow = 0`
///   rule. The platform `log1p(-1.0)` returns `-inf` (an infinite
///   result from a finite input), so `math.log1p(-1.0)` RAISES —
///   diverging from the kernel's `jsLog1p`, which serves the exact
///   value `-inf` there (`Eqv d ⟨-1,0⟩`). The raise domain is `x <=
///   -1` (closed), one wider than the kernel's own open `x < -1` NaN
///   corner.
/// - `asin`/`acos`: `FUNC1(asin, asin, 0, ...)` / `FUNC1(acos, acos, 0,
///   ...)`, the platform libm functions directly. `|x| = 1` is finite
///   (`asin(1) = pi/2`, `acos(-1) = pi`) — no infinite-result rule
///   fires there, so the raise domain is `|x| > 1` (OPEN), matching the
///   kernel's own boundary exactly: no divergence.
/// - `atanh`: `FUNC1(atanh, atanh, 0, ...)`. The platform `atanh(±1.0)`
///   returns `±inf` (an infinite result from a finite input), so
///   `math.atanh(±1.0)` RAISES — diverging from the kernel's
///   `jsAtanh`, which serves `±inf` there. The raise domain is `|x| >=
///   1` (closed), matching `atanh_sound.lean`'s own "`1 ± x <= 0`"
///   domain-error comment, one wider than a naive open reading.
/// - `acosh`: `FUNC1(acosh, acosh, 0, ...)`. `x = 1` is finite
///   (`acosh(1) = 0`) — the raise domain is `x < 1` (OPEN), matching
///   the kernel's own boundary exactly: no divergence.
///
/// Each row's SERVED half — the window's complement against the raise
/// domain — is what `served_half_window` intersects the operand
/// against for the straddling case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainLimitedFamily {
    Log,
    Log2,
    Log10,
    Log1p,
    Asin,
    Acos,
    Atanh,
    Acosh,
}

impl DomainLimitedFamily {
    /// The `math.*` attribute name this family answers, or `None` for
    /// every other function — the one place a name string is read into
    /// this enum, so every caller (the value dispatch, `expressions.rs`'s
    /// fire arms) shares one recognition.
    pub fn of_function(function: &str) -> Option<DomainLimitedFamily> {
        match function {
            "log" => Some(DomainLimitedFamily::Log),
            "log2" => Some(DomainLimitedFamily::Log2),
            "log10" => Some(DomainLimitedFamily::Log10),
            "log1p" => Some(DomainLimitedFamily::Log1p),
            "asin" => Some(DomainLimitedFamily::Asin),
            "acos" => Some(DomainLimitedFamily::Acos),
            "atanh" => Some(DomainLimitedFamily::Atanh),
            "acosh" => Some(DomainLimitedFamily::Acosh),
            _ => None,
        }
    }

    fn transfer_op(self) -> TransferQuestionOp {
        match self {
            DomainLimitedFamily::Log => TransferQuestionOp::Log,
            DomainLimitedFamily::Log2 => TransferQuestionOp::Log2,
            DomainLimitedFamily::Log10 => TransferQuestionOp::Log10,
            DomainLimitedFamily::Log1p => TransferQuestionOp::Log1p,
            DomainLimitedFamily::Asin => TransferQuestionOp::Asin,
            DomainLimitedFamily::Acos => TransferQuestionOp::Acos,
            DomainLimitedFamily::Atanh => TransferQuestionOp::Atanh,
            DomainLimitedFamily::Acosh => TransferQuestionOp::Acosh,
        }
    }

    /// The window CPython raises `ValueError` over — this enum's own
    /// doc names the exact `mathmodule.c` clause behind each row.
    fn raise_domain(self) -> RefinedSet {
        match self {
            DomainLimitedFamily::Log | DomainLimitedFamily::Log2 | DomainLimitedFamily::Log10 => {
                make_refined_set(vec![at_most(0.0)])
            }
            DomainLimitedFamily::Log1p => make_refined_set(vec![at_most(-1.0)]),
            DomainLimitedFamily::Asin | DomainLimitedFamily::Acos => {
                make_refined_set(vec![union(make_refined_set(vec![below(-1.0)]), make_refined_set(vec![above(1.0)]))])
            }
            DomainLimitedFamily::Atanh => make_refined_set(vec![union(
                make_refined_set(vec![at_most(-1.0)]),
                make_refined_set(vec![at_least(1.0)]),
            )]),
            DomainLimitedFamily::Acosh => make_refined_set(vec![below(1.0)]),
        }
    }

    /// The window's COMPLEMENT — the served half — spelled directly
    /// rather than through a generic set-difference form, the same way
    /// `split_divisor_transfer`'s own negative/positive halves are
    /// spelled directly rather than built from a `Difference` node.
    fn served_domain(self) -> RefinedSet {
        match self {
            DomainLimitedFamily::Log | DomainLimitedFamily::Log2 | DomainLimitedFamily::Log10 => {
                make_refined_set(vec![above(0.0)])
            }
            DomainLimitedFamily::Log1p => make_refined_set(vec![above(-1.0)]),
            DomainLimitedFamily::Asin | DomainLimitedFamily::Acos => {
                make_refined_set(vec![at_least(-1.0), at_most(1.0)])
            }
            DomainLimitedFamily::Atanh => make_refined_set(vec![above(-1.0), below(1.0)]),
            DomainLimitedFamily::Acosh => make_refined_set(vec![at_least(1.0)]),
        }
    }

    /// CPython's own runtime message for every row in this family —
    /// `is_error` (mathmodule.c): `if (errno == EDOM) PyErr_SetString
    /// (PyExc_ValueError, "math domain error")` — one shared string
    /// across the whole module, not a per-function wording, matching
    /// `expressions.rs`'s existing `math.sqrt` raise arm.
    pub fn raise_message(self) -> &'static str {
        "this expression provably raises ValueError: math domain error"
    }
}

/// Whether a KNOWN operand's window is ENTIRELY inside a family's raise
/// domain, STRADDLES the boundary (admits both raising and non-raising
/// values), or is ENTIRELY inside the served domain — the three-way
/// read `expressions.rs`'s `call_provable_raise` (entirely-raises) and
/// `possible_raise` (straddles) both ask, mirroring
/// `divisor_is_provably_always_zero`/`divisor_provably_excludes_zero`'s
/// own `scalar_subset`/`scalar_disjoint` pair exactly — the same two
/// kernel questions, posed against this family's own `raise_domain()`
/// rather than the fixed `{0.0}` divisor does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainRaiseClassification {
    EntirelyRaises,
    Straddles,
    EntirelyServed,
}

/// Classifies a KNOWN operand (a single value or a bounded set) against
/// `family`'s raise domain. `None` when the operand cannot be read as a
/// transferable window at all (an unknown argument, a non-numeric
/// sort) — the caller declines exactly as every other unread shape in
/// this file does.
pub fn domain_raise_classification(
    family: DomainLimitedFamily,
    argument: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<DomainRaiseClassification> {
    let operand = float_transferable_operand(argument)?;
    let raise_domain = family.raise_domain();
    let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(&operand));
    if matches!(empty, Ok(true)) || empty.is_err() {
        return None;
    }
    let entirely_raises = crate::kernel_ask::ask_kernel(|| (kernel.scalar_subset)(&operand, &raise_domain));
    if matches!(entirely_raises, Ok(true)) {
        return Some(DomainRaiseClassification::EntirelyRaises);
    }
    let entirely_served = crate::kernel_ask::ask_kernel(|| (kernel.scalar_disjoint)(&operand, &raise_domain));
    if matches!(entirely_served, Ok(true)) {
        return Some(DomainRaiseClassification::EntirelyServed);
    }
    Some(DomainRaiseClassification::Straddles)
}

/// The served half's kernel window for a STRADDLING operand — the exact
/// mirror of `split_divisor_transfer`'s own split-and-re-ask pattern,
/// narrowed to one half (this family's `served_domain()`) rather than
/// two, since a domain-limited unary function has one raise-side ray
/// and one served-side ray/interval, not two symmetric halves around a
/// point. Poses the operand's window INTERSECTED with the served
/// domain — never the raw operand window, which would ask
/// `js.log`/`js.asin`/… a question a raising sub-window makes unsound
/// for Python. `None` on a kernel refusal, an empty intersection (the
/// operand does not actually straddle — the caller's own
/// `domain_raise_classification` should have already ruled this out),
/// or a `NaN`/`Unknown` answer on the served half (a decline, never a
/// mis-answer — the same discipline `kernel_backed_unary_family_call`
/// keeps for the non-straddling case).
pub fn domain_raise_served_half_value(
    family: DomainLimitedFamily,
    argument: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let operand = float_transferable_operand(argument)?;
    let served_half = make_refined_set({
        let mut forms = operand.forms.clone();
        forms.extend(family.served_domain().forms.clone());
        forms
    });
    let empty = crate::kernel_ask::ask_kernel(|| (kernel.scalar_empty)(&served_half));
    if matches!(empty, Ok(true)) || empty.is_err() {
        return None;
    }
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: family.transfer_op(),
            a: served_half,
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(argument));
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// Poses one KERNEL-BACKED question for the explog/trig family's
/// one-argument members (`Exp`, `Expm1`, `Log`, `Log1p`, `Log2`,
/// `Log10`, `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh`, `Asin`,
/// `Acos`, `Atan`, `Asinh`, `Acosh`, `Atanh`) and reads the answer back
/// Float-sorted — the exact mirror of `sqrt_call_over_set`'s own
/// construction and refusal discipline, generalized to any unary
/// `TransferQuestionOp` and to a known-single-value operand via
/// `float_transferable_operand`.
///
/// A `TransferAnswerKind::NaN` answer declines to `None` rather than
/// answering a value — the same reasoning `sqrt_argument_is_known_negative`
/// already keeps for `sqrt`, generalized to the rest of the family
/// rather than restated per function.
///
/// For the six DOMAIN-LIMITED members (`DomainLimitedFamily::of_function`),
/// this function additionally gates the VALUE side against CPython's
/// own raise domain — `DomainLimitedFamily::raise_domain`'s own doc —
/// which is WIDER than the kernel's `.nan` corner for `log`/`log2`/
/// `log10`/`log1p`/`atanh` at exactly one boundary point each (the
/// JS-vs-Python divergence that enum documents). Without this gate,
/// `math.log(0.0)` would read the kernel's served `-inf` value as a
/// Python return, when the real call raises there instead. A window
/// that STRADDLES the raise domain (some served values, some raising)
/// still declines HERE — `expressions.rs`'s `possible_raise` sibling
/// asks `domain_raise_served_half_value` directly for that case, since
/// this function's own "one call, one answer" shape has no room to
/// speak the served HALF only.
pub(super) fn kernel_backed_unary_family_call(
    function: &str,
    op: TransferQuestionOp,
    value: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    if let Some(family) = DomainLimitedFamily::of_function(function) {
        match domain_raise_classification(family, value, kernel) {
            Some(DomainRaiseClassification::EntirelyServed) => {}
            // EntirelyRaises: the real call never returns a value here —
            // `call_provable_raise`'s own row, not this function's to
            // answer. Straddles: only the served half determines, and
            // this function answers no partial value —
            // `possible_raise`'s own row reads `domain_raise_served_
            // half_value` directly. A classification refusal (`None`)
            // is the same unread-operand-shape decline every other row
            // in this file already gives.
            _ => return None,
        }
    }
    let operand = float_transferable_operand(value)?;
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op,
            a: operand,
            b: make_refined_set(vec![]),
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        // NaN: the real Python call raises rather than returning a
        // value — this function's own doc. Unknown: the kernel arm
        // itself declines this operand shape (e.g. `jsAtan2`'s
        // non-`x>0` quadrants — see `kernel_backed_atan2_call`).
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// `math.atan2(y, x)` — the one two-argument member of this family
/// (pins row trig.10). Poses `TransferQuestionOp::Atan2` over both
/// known operands; the exact two-operand mirror of
/// `kernel_backed_unary_family_call` above.
///
/// `jsAtan2` (`languages/javascript/trig/atan2.lean`) only serves the
/// `x > 0, y ≠ 0` quadrant today ("the axis and left-half-plane
/// corners wait on π pins," the file's own comment) and answers
/// `Unknown` — never `NaN` — everywhere else, so there is no raise-vs-
/// NaN divergence to gate here the way the log/asin/acos/atanh/acosh
/// family needs: `atan2` is total over the reals in Python
/// (library/math.rst's own clause states no domain restriction), and
/// an `Unknown` kernel answer is this arm's own current serving gap,
/// not a Python raise — it declines the same as every other unread
/// shape in this file.
pub(super) fn kernel_backed_atan2_call(
    y: &AbstractValue,
    x: &AbstractValue,
    kernel: &Arc<RefinedTSKernel>,
) -> Option<AbstractValue> {
    let y_operand = float_transferable_operand(y)?;
    let x_operand = float_transferable_operand(x)?;
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Atan2,
            a: y_operand,
            b: x_operand,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, &[y.clone(), x.clone()]);
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// `math.hypot(a, b)` — the TWO-ARGUMENT form (pow.8's own two-
/// coordinate row; the general N-ary `math.hypot(*coordinates)` stays
/// this file's own named seam below). Poses `TransferQuestionOp::Hypot`
/// (wire `"js.hypot"`, `boundary/javascript.lean`'s shared name-keyed
/// transfer table — the SAME table `kernel_backed_atan2_call` above
/// asks) over both known operands; the exact two-operand mirror of
/// `kernel_backed_atan2_call`.
///
/// `transferHypot` (`languages/python/powers_and_roots/hypot.lean`'s
/// `pyHypot`, proved sound in `hypot_sound.lean`'s three rows: the
/// exact-zero empty-coordinate case, the two-coordinate window shape,
/// and the unbounded fallback) serves EVERY finite operand window —
/// unlike `jsAtan2`, there is no served-quadrant gap here, so an
/// `Unknown` answer from this arm reads as an unread operand SHAPE
/// (non-numeric-sorted, or the kernel's own general refusal), never a
/// missing quadrant. `math.hypot` also never raises on finite operands
/// (library/math.rst, `hypot(*coordinates)`'s own clause states no
/// domain restriction the way `sqrt`/`log` do), so a `NaN` answer is
/// not expected here either — both still decline honestly rather than
/// assume which case produced them.
pub(super) fn kernel_backed_hypot_call(a: &AbstractValue, b: &AbstractValue, kernel: &Arc<RefinedTSKernel>) -> Option<AbstractValue> {
    let a_operand = float_transferable_operand(a)?;
    let b_operand = float_transferable_operand(b)?;
    let nan_operand = PowOperandWire { kind: PowOperandKind::NaN, set: make_refined_set(vec![]) };
    let asked = crate::kernel_ask::ask_kernel(|| {
        (kernel.transfer)(&TransferQuestion {
            op: TransferQuestionOp::Hypot,
            a: a_operand,
            b: b_operand,
            c: 0.0,
            base: nan_operand.clone(),
            exp: nan_operand,
        })
    })
    .ok()?;
    let grade = derived_trust_level(TrustSpec, &[a.clone(), b.clone()]);
    match asked.kind {
        TransferAnswerKind::Values => Some(known_values(asked.values, PrimitiveKind::Float, grade)),
        TransferAnswerKind::Set => Some(AbstractValue {
            kind_tag: Some(PrimitiveKind::Float),
            ..known_set(asked.set, None, grade, SetKindTag::None)
        }),
        TransferAnswerKind::NaN | TransferAnswerKind::Unknown => None,
    }
}

/// The explog/trig pins rows' own `TransferQuestionOp` election, one
/// per one-argument function name — the kernel operation column each
/// pins row (`explog.1`–`explog.6`, `trig.1`–`trig.9`, `trig.11`–
/// `trig.13`) now reads through `boundary/javascript.lean`'s shared
/// name-keyed transfer table (`"js.exp"`, `"js.sin"`, …), the same
/// table Python's own `int.*` arms register into
/// (`boundary/python.lean`'s own header: "Registered into the SAME
/// name-keyed transfer table... every wire op name is a flat string
/// key"). `atan2` (trig.10) is excluded — its own
/// `kernel_backed_atan2_call` above poses the two-operand question
/// directly. `hypot` (pow.8) is excluded the same way: its own
/// TWO-ARGUMENT form poses `TransferQuestionOp::Hypot` directly through
/// `kernel_backed_hypot_call` above — only the general VARIADIC form
/// (`math.hypot(*coordinates)`, three or more arguments) has no kernel
/// election and stays this file's own named seam. `cbrt` (pow.6) is
/// excluded from THIS table for the same reason `sqrt` is — its
/// known-SET operand is posed directly by its own dedicated row
/// (`cbrt_call_over_set`), mirroring `sqrt_call_over_set`, not folded
/// into this shared-shape dispatch table.
pub(super) fn kernel_backed_unary_family_op(function: &str) -> Option<TransferQuestionOp> {
    match function {
        "exp" => Some(TransferQuestionOp::Exp),
        "expm1" => Some(TransferQuestionOp::Expm1),
        "log" => Some(TransferQuestionOp::Log),
        "log1p" => Some(TransferQuestionOp::Log1p),
        "log2" => Some(TransferQuestionOp::Log2),
        "log10" => Some(TransferQuestionOp::Log10),
        "sin" => Some(TransferQuestionOp::Sin),
        "cos" => Some(TransferQuestionOp::Cos),
        "tan" => Some(TransferQuestionOp::Tan),
        "sinh" => Some(TransferQuestionOp::Sinh),
        "cosh" => Some(TransferQuestionOp::Cosh),
        "tanh" => Some(TransferQuestionOp::Tanh),
        "asin" => Some(TransferQuestionOp::Asin),
        "acos" => Some(TransferQuestionOp::Acos),
        "atan" => Some(TransferQuestionOp::Atan),
        "asinh" => Some(TransferQuestionOp::Asinh),
        "acosh" => Some(TransferQuestionOp::Acosh),
        "atanh" => Some(TransferQuestionOp::Atanh),
        _ => None,
    }
}
