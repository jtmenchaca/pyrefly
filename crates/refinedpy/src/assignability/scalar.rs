//! Scalar/string shape helpers and fire-message spellings.

use refined_sets::codepoint_sets::is_string_ground;
use refined_sets::refinement_forms::on_one_tuple_layer;

use super::sequence::sequence_shaped;

/// Whether the declared set names scalars or strings — the shapes a
/// dict/list/None/opaque value can NEVER inhabit. Three recognizers:
/// numeric 1-tuple forms (`on_one_tuple_layer`), the full string ground
/// (`is_string_ground`), and SEQUENCE-SHAPED sets (every top-level form
/// is a string/sequence form — EmptyTuple/Concatenation/Star/Repeat/
/// RepeatWord — or a Union/Difference of sequence-shaped operands),
/// which is what a `Literal["a", "b"]` union of string tuples compiles
/// to. A set none of the three recognize declines the structural laws
/// and falls through to the general undetermined answer.
pub(super) fn scalar_or_string_shaped(set: &refined_sets::refinement_forms::RefinedSet) -> bool {
    on_one_tuple_layer(set) || is_string_ground(set) || sequence_shaped(set)
}

/// One admitted scalar that could be a one-character string's own
/// codepoint — ported from refined-ts-go's `CodepointScalar`
/// (walk/sequence_measures.go): a natural number inside the codepoint
/// alphabet (`codepoint_sets::codepoints`'s own surrogate-gap-excluding
/// range), never a negative, fractional, or out-of-range value.
pub(super) fn codepoint_scalar(v: f64) -> bool {
    v == v.trunc() && v >= 0.0 && (v <= 0xD7FF as f64 || (v >= 0xE000 as f64 && v <= 0x10FFFF as f64))
}

/// Whether EVERY value a scalar set admits sits inside the codepoint
/// alphabet — ported from refined-ts-go's `WithinCodepointDoor`
/// (walk/sequence_measures.go): such a set is indistinguishable from a
/// union of one-character strings, so the string-vs-numeric sort laws
/// must not refute a string value against it on shape alone. Two
/// spellings pass: enumerated codepoints (`OneOf`), and
/// INTEGER-constrained windows wholly inside one side of the surrogate
/// gap (`Field(pattern=r"^[\x00-\x7f]$")`'s own shape). Windows without
/// the `Integer` form answer false (they admit non-codepoint reals),
/// and the window test is conservative (`Above`/`Below` widen to their
/// closed bounds) so the door never opens wrongly. `integer_inherited`
/// carries the ancestor's own `Integer` form down through a `Union`
/// (the same recursion refined-ts-go's Go source takes), since a bound
/// form nested under a `Union` reads its sort from the branch that
/// states it, not from its own immediate siblings.
pub(super) fn within_codepoint_door(
    set: &refined_sets::refinement_forms::RefinedSet,
    integer_inherited: bool,
) -> bool {
    use refined_sets::refinement_forms::Form;
    if set.forms.is_empty() {
        return false;
    }
    let mut integer = integer_inherited;
    if !integer {
        integer = set.forms.iter().any(|form| form.form == Form::Integer);
    }
    let mut lo = f64::NEG_INFINITY;
    let mut hi = f64::INFINITY;
    let mut content = false;
    for form in &set.forms {
        match form.form {
            Form::Integer => {}
            Form::OneOf => {
                if !form.w.iter().all(|&w| codepoint_scalar(w)) {
                    return false;
                }
                content = true;
            }
            Form::AtLeast | Form::Above => lo = lo.max(form.a),
            Form::AtMost | Form::Below => hi = hi.min(form.a),
            Form::Union => {
                let a = form.a_.as_deref();
                let b = form.b.as_deref();
                let a_ok = a.map(|s| within_codepoint_door(s, integer)).unwrap_or(false);
                let b_ok = b.map(|s| within_codepoint_door(s, integer)).unwrap_or(false);
                if !a_ok || !b_ok {
                    return false;
                }
                content = true;
            }
            _ => return false,
        }
    }
    if lo != f64::NEG_INFINITY || hi != f64::INFINITY {
        if !integer || lo > hi {
            return false;
        }
        let in_low = lo >= 0.0 && hi <= 0xD7FF as f64;
        let in_high = lo >= 0xE000 as f64 && hi <= 0x10FFFF as f64;
        if !in_low && !in_high {
            return false;
        }
        content = true;
    }
    content
}

/// The readable spelling of a string word for a fire message: the code
/// points decoded back to text and JSON-quoted, the same spelling
/// `format_string_shapes::format_string_literal` gives a set's own
/// literal chain. Falls back to the Python `repr`-style bare digits
/// only if the points sit outside the representable scalar range (an
/// honest label rather than a silent drop) — `from_points` returns
/// `None` there.
pub(super) fn spelled_string_word(points: &[f64]) -> String {
    refined_sets::format_string_shapes::from_points(points)
        .unwrap_or_else(|| format!("{:?}", points))
}

/// The readable spelling of a Boolean-tagged value for a fire message:
/// the Python literal `True`/`False`, never the bare `1`/`0` a numeric
/// spelling would give (`format_py_number` reads the sort tag, not the
/// PRODUCT-LAW distinction this file's Boolean law exists to state).
/// Falls back to the bare digit only for the unreached case of an
/// empty or multi-valued Boolean word (a boolean is always exactly one
/// value, `expressions.rs`'s own `BooleanLiteral` encoding) — an honest
/// label rather than a silent drop.
pub(super) fn spelled_boolean_word(values: &[f64]) -> String {
    match values {
        [v] if *v == 1.0 => "True".to_owned(),
        [v] if *v == 0.0 => "False".to_owned(),
        _ => format!("{:?}", values),
    }
}
