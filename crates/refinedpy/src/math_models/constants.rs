use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::nan_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::TrustProved;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::below;
use refined_sets::refinement_forms::make_refined_set;

/// `math.pi` / `math.e` / `math.tau` / `math.inf` / `math.nan` —
/// ATTRIBUTE READS, not calls (library/math.rst, "Constants" section:
/// `data:: pi`/`data:: e`/`data:: tau`/`data:: inf`/`data:: nan`, each
/// "to available precision" or, for `inf`/`nan`, "Equivalent to the
/// output of `float('inf')`" / `float('nan')`. Each answers the EXACT
/// CPython value, not a sort-only approximation:
///
/// - `pi`/`e`/`tau`: `std::f64::consts::PI`/`E`/`TAU` ARE CPython's own
///   values — both are the nearest binary64 double to the mathematical
///   constant, and IEEE 754 binary64 has exactly one nearest
///   representable value for a given real number, so Rust's constant
///   and CPython's libm-derived constant are the same bit pattern.
/// - `inf`: `f64::INFINITY`, library/math.rst's own "Equivalent to the
///   output of `float('inf')`." `+inf` is a legal `RefinedSet` element
///   (`refinement_forms.rs`'s `element` helper panics ONLY on NaN, never
///   on an infinite operand — `one_of`/`at_least`/`above` all route
///   through it unchanged), matching the Lean kernel's own admission of
///   `+-infinity` as elements of R-bar (`refinement_forms.go`'s twin
///   comment) — so `known_values` is the ordinary, unmodified route for
///   this constant, the same route every other exact numeric constant
///   in this file takes.
/// - `nan`: `nan_value()` — the domain's own `Kind::NaN` carrier, NEVER
///   a value inside `known_values`: `element`'s construction-time panic
///   refuses NaN for every refined-set form, so a `one_of`/singleton
///   containing NaN cannot be built at all. This is the same NaN
///   carrier `float_result` reaches for elsewhere in this file.
pub fn math_constant_value(name: &str) -> Option<AbstractValue> {
    match name {
        "pi" => Some(known_values(vec![std::f64::consts::PI], PrimitiveKind::Float, TrustProved)),
        "e" => Some(known_values(vec![std::f64::consts::E], PrimitiveKind::Float, TrustProved)),
        "tau" => Some(known_values(vec![std::f64::consts::TAU], PrimitiveKind::Float, TrustProved)),
        "inf" => Some(known_values(vec![f64::INFINITY], PrimitiveKind::Float, TrustProved)),
        "nan" => Some(nan_value()),
        _ => None,
    }
}

/// `random.random()` — library/random.rst, `function:: random()`:
/// "Return the next random floating-point number in the range `0.0 <=
/// X < 1.0`." A Float-tagged Set bounded to that half-open window
/// (`at_least(0.0)` meets `below(1.0)`, the same ray-intersection shape
/// `float_sorted_unknown()` builds over the unbounded ray) — narrower
/// than the sort-only all-numbers answer other approximated `math`
/// calls carry, since this clause pins the interval exactly, only the
/// specific real drawn within it. Scoped to this one function of the
/// `random` module; no other `random.*` call is modeled here.
pub fn random_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "random" || !arguments.is_empty() {
        return None;
    }
    let window = make_refined_set(vec![at_least(0.0), below(1.0)]);
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Float),
        ..known_set(window, None, TrustSpec, SetKindTag::None)
    })
}
