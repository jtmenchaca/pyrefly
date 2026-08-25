use refined_domain::abstract_value::kind_union_of;
use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::known_values;
use refined_domain::abstract_value::null_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::ObjectKey;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::known_constructors::known_object;
use refined_domain::trust_grades::TrustSpec;
use refined_sets::refinement_forms::at_least;
use refined_sets::refinement_forms::at_most;
use refined_sets::refinement_forms::requires_integer;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;
use refined_sets::regex_compiler::format_grammar;

use crate::foreign_edge_artifact::ForeignCase;
use crate::foreign_edge_artifact::ForeignTsArtifact;

use super::crossing::uncarriable_corner_of;

/// A cases LIST lowered to one `AbstractValue` — one case direct, several
/// through `kind_union_of` — the same channel `foreign_return_value_or_
/// undetermined` applies at the top level (after `clip_uncarriable_
/// corners` has run) and a member's own `Vec<ForeignCase>`
/// (`ForeignCase::Object`'s own field) applies once per member, since a
/// member's cases list is the identical "one or several wire cases name
/// one value" shape recursed one layer down.
///
/// `TrustSpec`, mirroring the Go twin's `foreignReturnValue`: the value
/// is not this kernel's own decision about this expression, it is
/// another language's claim carried across a transport whose identity is
/// a CITED PREMISE, not a proved theorem — every arm `foreign_case_value`
/// builds wears that grade.
pub(super) fn foreign_case_list_value(cases: &[ForeignCase], function_name: &str) -> Result<AbstractValue, String> {
    let mut values = Vec::with_capacity(cases.len());
    for case in cases {
        values.push(foreign_case_value(case, function_name)?);
    }
    Ok(kind_union_of(values))
}

/// One case lowered to its own `AbstractValue`: a number/string set
/// case's SORT comes from the case tag itself, never guessed from the
/// set's own forms — the whole point of the wire stating "number" or
/// "string" explicitly is that a crossed return never needs the
/// declared-position sort law's own shape heuristic. `requires_or_
/// reads_integer` still decides Integer vs Float WITHIN a number case
/// (a union-of-integer-literal return like `union_levels.ts`'s derived
/// `{1, 2, 4}` reads Integer; a numeric set stating neither reads
/// Float) — that distinction is orthogonal to the case's own number/
/// string/boolean/null tag.
///
/// `ForeignCase::Object` lowers into the domain's own object vocabulary
/// — the same `Kind::Object` shape `collection_models.rs`'s
/// `dict_literal_value` builds for an ordinary `{...}` display, through
/// the identical `known_object` constructor
/// (`refined_domain::known_constructors::known_object`), never a
/// parallel object representation. Each member's own cases list
/// recurses through `foreign_case_list_value` — the same one-direct/
/// several-union channel this function's own caller applies at the top
/// level — so a member typed as several wire cases (a Result-shape
/// member itself carrying a nested object union) lowers the same way a
/// multi-case return does. `complete` comes straight from the case's own
/// `closed`: a closed case states its member list is the WHOLE key set,
/// which is exactly what `known_object`'s `complete` flag claims.
/// Every member key is a plain string entry (`numeric: false`) — the
/// wire's own member-name vocabulary is always a JSON object key, never
/// a Python int-keyed dict entry. `stated` is `None` (no
/// `ObjectAnnotationRef` for a value this checker derived rather than
/// read off a declared annotation) and `bare_proto` is `false`,
/// matching `dict_literal_value`'s own call.
pub(super) fn foreign_case_value(case: &ForeignCase, function_name: &str) -> Result<AbstractValue, String> {
    Ok(match case {
        ForeignCase::Number(set) => {
            let sort = if requires_or_reads_integer(set) { PrimitiveKind::Integer } else { PrimitiveKind::Float };
            AbstractValue { kind_tag: Some(sort), ..known_set(set.clone(), None, TrustSpec, SetKindTag::None) }
        }
        ForeignCase::String(set) => known_set(set.clone(), None, TrustSpec, SetKindTag::None),
        ForeignCase::Boolean => known_values(vec![0.0, 1.0], PrimitiveKind::Boolean, TrustSpec),
        ForeignCase::Null => null_value(),
        ForeignCase::Object { members, closed } => {
            let mut keys = Vec::with_capacity(members.len());
            for (name, member_cases) in members {
                let value = foreign_case_list_value(member_cases, function_name)?;
                keys.push(ObjectKey { name: name.clone(), numeric: false, value });
            }
            known_object(keys, None, *closed, TrustSpec, false)
        }
    })
}

/// Whether a set's own forms state an integer sort — `requires_integer`
/// (the explicit `Form::Integer` marker, looking through `Union`/
/// `Difference`) OR every value an `OneOf` form admits is a whole,
/// finite number. A crossed return carries no annotation to attach an
/// explicit `Integer` form to (unlike a declared `int`-based alias), so
/// a derived Literal-set return (`union_levels.ts`'s `{1, 2, 4}`) is
/// only ever an all-integer `OneOf` — this is the wider reading the
/// crossed-value case needs beyond the declared-position law it
/// otherwise mirrors.
pub(super) fn requires_or_reads_integer(set: &RefinedSet) -> bool {
    if requires_integer(set) {
        return true;
    }
    for form in &set.forms {
        match form.form {
            Form::OneOf => {
                if !form.w.is_empty() && form.w.iter().all(|&w| w.is_finite() && w == w.trunc()) {
                    return true;
                }
            }
            Form::Union | Form::Difference => {
                if requires_or_reads_integer(form.a_.as_ref().unwrap())
                    || form.b.as_ref().is_some_and(|b| requires_or_reads_integer(b))
                {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

/// The return leg's own fact, DETERMINED at a corner the target's own
/// `JSON.stringify` serializes as the bare token `null` (ECMA-262's
/// `SerializeJSONProperty`, the finiteness check on a Number value — not
/// an RFC 8259 gap, since `1e999` is legal JSON text that parses to
/// Infinity in both runtimes): a NUMBER case admitting +Infinity or
/// -Infinity crosses as EXACTLY the claimed set's finite portion UNION
/// the null case, since a run whose result is non-finite lands a `null`
/// on this leg's own `json.loads` — never a value outside a set this
/// crossing still claims. `clip_uncarriable_corners` performs that
/// transform on the whole cases list before it lowers; the null arm
/// rides the SAME possibly-absent channel the wire's own `{"sort":
/// "null"}` case already lowers through (`foreign_case_value`'s `Null`
/// arm), so an ordinary judge sees a plain `float | None` union at the
/// call site — a declared plain float refuses the None arm (a
/// determined fire), and a declared `Optional`/`| None` return admits
/// it. This is the gate every caller of `foreign_return_value` for a
/// RETURN (never an entry — the outbound leg's own NaN-freedom check is
/// the different, already-landed premise for the value crossing OUT)
/// must pass through first.
pub(super) fn foreign_return_value_or_undetermined(artifact: &ForeignTsArtifact) -> Result<AbstractValue, String> {
    let clipped_cases = clip_uncarriable_corners(&artifact.called.return_cases);
    foreign_case_list_value(&clipped_cases, &artifact.called.name)
}

/// Every NUMBER case's own set clipped to its finite portion at any
/// corner (+Infinity/-Infinity) `uncarriable_corner_of` names, with a
/// `Null` case appended once (never duplicated — a return whose wire
/// already states its own `{"sort":"null"}` case is left with exactly
/// one) when at least one clip fired — a string/boolean/null/object case
/// states no scalar corner this premise is about and passes through
/// unchanged. Clipping intersects the set with the largest representable
/// finite window (`at_most(f64::MAX)`/`at_least(-f64::MAX)`), narrowing
/// the hull to exactly the claimed set's finite portion without touching
/// any finite bound the set already states; `uncarriable_corner_of` is
/// asked again after the first clip so a set admitting BOTH corners
/// (an unbounded-both-ways window) is clipped on each side in turn.
pub(super) fn clip_uncarriable_corners(cases: &[ForeignCase]) -> Vec<ForeignCase> {
    let mut clipped = Vec::with_capacity(cases.len());
    let mut any_clipped = false;
    let mut already_null = false;
    for case in cases {
        match case {
            ForeignCase::Number(set) => {
                let mut set = set.clone();
                let mut this_clipped = false;
                while let Some(corner) = uncarriable_corner_of(&set) {
                    set.forms.push(if corner == "-Infinity" { at_least(-f64::MAX) } else { at_most(f64::MAX) });
                    this_clipped = true;
                }
                any_clipped = any_clipped || this_clipped;
                clipped.push(ForeignCase::Number(set));
            }
            ForeignCase::Null => {
                already_null = true;
                clipped.push(case.clone());
            }
            other => clipped.push(other.clone()),
        }
    }
    if any_clipped && !already_null {
        clipped.push(ForeignCase::Null);
    }
    clipped
}

/* ── the intermediate captured-stdout reading ────────────────────── */

/// The JSON number production (RFC 8259 §6 / json.org's number diagram,
/// the same grammar `json.loads`'s own `NUMBER_RE` cites): an optional
/// sign, an integer part that is either the single digit 0 or a nonzero
/// digit followed by any run of digits (no leading zero —
/// `json.dumps` never writes one), an optional fractional part, an
/// optional exponent. Anchored `^...$` by `format_grammar`'s own
/// convention (this crate's kernel-bridge-facts row: anchor sub-patterns
/// yourself, since an unanchored compile pads both sides with `C*`) and
/// followed by the ONE trailing newline `subprocess.run`'s captured
/// stdout always carries when `text=True` (the harness's own print/
/// stdout.write terminates its line) — the harness never writes a
/// SECOND line for a scalar return, so exactly one `\n`, not a star of
/// them. Mirrors `refined-ts-go/internal/refinedts/walk/foreign_edge.go`'s
/// `jsonNumberGrammarPattern` exactly, the Go twin's own reverse-pair
/// row this derivation ports.
const JSON_NUMBER_GRAMMAR_PATTERN: &str = r"-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?\n";

/// Compiles `JSON_NUMBER_GRAMMAR_PATTERN` through `format_grammar` — the
/// SAME door `surface.rs`'s pydantic `pattern=` compiles a `Field`
/// constraint through — the pattern vocabulary this binding reuses
/// rather than hand-building the concatenation/union forms a regex
/// source already denotes. The pattern's own constructs (character
/// classes, `?`, alternation, `*`/`+`) are all supported forms
/// (`regex_compiler.rs`'s own `test_anchored_classes_and_quantifiers_
/// compile`); a compile failure here is an impossible state — panics
/// rather than silently widening to an unconstrained string set, so a
/// future change to the supported regex subset that actually broke this
/// pattern fails loudly at the first call instead of quietly degrading
/// every stdout reading to residue.
pub(super) fn json_number_grammar_set() -> RefinedSet {
    let compiled = format_grammar(&("^".to_owned() + JSON_NUMBER_GRAMMAR_PATTERN + "$"), "");
    if !compiled.ok {
        panic!("JSON_NUMBER_GRAMMAR_PATTERN does not compile: {}", compiled.unsupported);
    }
    compiled.set
}

/// The SERIALIZED form of a discharged crossing's return cases — the
/// string-sorted set the intermediate captured-stdout reading (`result
/// .stdout` for `ResultRead::StdoutAttribute`, or the bound name itself
/// for `ResultRead::Bare`) actually holds, read structurally off what
/// the target's own JSON encoder can spell for that return.
///
/// The only derivable shape today is a return whose every PRESENT case
/// is number-sorted: the harness writes exactly one JSON number followed
/// by one newline, so the serialized set is `json_number_grammar_set()`
/// — the WINDOWLESS general JSON-number grammar, not a tightened
/// per-window grammar the case's own bounds might suggest (that
/// tightening needs a number-to-JSON-text theory this tree does not
/// have yet). The windowless grammar is still a REAL claim — it
/// excludes every non-numeric text ("abc", an empty stdout, a bare
/// token like "Infinity") — and a weaker true claim beats no claim at
/// all.
///
/// `None` for an empty cases list, or a cases list carrying any
/// NON-NUMBER case (string, boolean, object) — the JSON text a mixed or
/// non-numeric return spells is a different, wider question this
/// derivation does not attempt, so the caller leaves the stdout reading
/// unbound rather than guess. A NULL case riding alongside a number
/// case ALSO answers `None` — `json.dumps(None)` writes the bare token
/// `null`, not a JSON number, and covering that second literal
/// alongside the number grammar is not part of what this derivation
/// states. Mirrors the Go twin's `foreignStdoutSerializedValue` exactly.
pub(super) fn foreign_stdout_serialized_value(cases: &[ForeignCase]) -> Option<AbstractValue> {
    if cases.is_empty() {
        return None;
    }
    if !cases.iter().all(|case| matches!(case, ForeignCase::Number(_))) {
        return None;
    }
    Some(known_set(json_number_grammar_set(), None, TrustSpec, SetKindTag::None))
}

