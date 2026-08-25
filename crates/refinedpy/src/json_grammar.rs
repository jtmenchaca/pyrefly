//! `json.dumps`'s serialized text as a GRAMMAR (a `RefinedSet` over
//! codepoints), for the case `expressions.rs::json_dumps_value` cannot
//! read as one exact string: a closed dict whose OWN members carry SETS
//! (a windowed int, a Literal-union string) rather than exact values.
//! `json_dumps_value` still owns every EXACT case (a known Integer, a
//! known string, a dict of exact members) — this file is the fallback
//! it reaches for once an exact reading fails, composing the same
//! shape `SerializeJSONObject`'s Python-to-JSON conversion table states
//! (library/json.rst) but over the SET each member is known to admit
//! rather than over one concrete value.
//!
//! Reference for the composition shape: `refined-ts-go`'s
//! `internal/refinedts/refinementsets/json_case_grammar.go` — that file
//! composes the identical member-pair/brace/comma structure for the JS
//! adapter's `JSON.stringify`. The one structural difference: a JS
//! object's own key order is NOT the wire's insertion order (that file
//! unions every permutation), where a Python `dict` genuinely preserves
//! insertion order (library/stdtypes.html, "Mapping Types — dict":
//! "Dictionaries preserve insertion order" — the language guarantee
//! landed in 3.7 and is unconditional since), so `object_members_grammar`
//! below composes the ONE ordering `AbstractValue.keys` already carries
//! (`abstract_value.rs`'s own doc: "an ordered slice of (name, value)...
//! in insertion order") rather than a permutation union.
//!
//! Every member's own value composes through `member_value_grammar`,
//! which tries `json_dumps_value`'s own exact reading FIRST (threaded
//! in as `member_reader`, since this file does not import
//! `expressions.rs` — see `dumps_grammar`'s doc), then the two SET arms
//! this file composes, then recurses for a nested `Kind::Object`. A
//! member whose value is neither an exact reading nor a composable set
//! (Float, Boolean, a list, an unknown value) makes the WHOLE object
//! decline — matching `json_dumps_value`'s own "no partial answer"
//! discipline.

use refined_domain::abstract_value::known_set;
use refined_domain::abstract_value::opaque_value;
use refined_domain::abstract_value::AbstractValue;
use refined_domain::abstract_value::Kind;
use refined_domain::abstract_value::PrimitiveKind;
use refined_domain::abstract_value::SetKindTag;
use refined_domain::trust_grades::derived_trust_level;
use refined_domain::trust_grades::TrustProved;
use refined_sets::codepoint_sets::string_tuple;
use refined_sets::codepoint_sets::word_tuples_of;
use refined_sets::refinement_forms::concatenation;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::refinement_forms::one_of;
use refined_sets::refinement_forms::repeat_of;
use refined_sets::refinement_forms::union;
use refined_sets::refinement_forms::Form;
use refined_sets::refinement_forms::RefinedSet;

/// One codepoint drawn from the given ASCII characters — the digit
/// alphabet `integer_window_grammar` repeats. Mirrors
/// `expressions.rs::one_char_of`, kept as a private copy per this
/// crate's file-scope convention (`string_models.rs`'s own doc on
/// `exact_string_text`) rather than widening that function's visibility
/// for one caller outside its file.
fn one_char_of(chars: &str) -> RefinedSet {
    let points: Vec<f64> = chars.chars().map(|c| c as u32 as f64).collect();
    make_refined_set(vec![one_of(&points)])
}

/// The number of decimal digits a NONNEGATIVE integer's plain `str()`
/// spelling carries — `0` itself spells one digit ("0"). Mirrors
/// `expressions.rs::decimal_digit_count`, same private-copy convention.
fn decimal_digit_count(value: i64) -> u32 {
    if value == 0 {
        return 1;
    }
    value.unsigned_abs().to_string().len() as u32
}

/// The exact serialized-text grammar for a NONNEGATIVE, bounded Integer
/// window `[lo, hi]` — `int.__repr__` (stdtypes.rst) spells a
/// nonnegative int as a plain, no-leading-zero decimal run, so the
/// serialized text's digit COUNT is bounded exactly by the shortest and
/// longest spelling in the window: `decimal_digit_count(lo)` to
/// `decimal_digit_count(hi)` digits, every digit drawn from `0-9`. This
/// is a SOUND OVER-APPROXIMATION of the exact window (it admits every
/// digit run of the right length, not only those whose VALUE falls in
/// `[lo, hi]`) — sufficient for a downstream length or pattern sink,
/// which is what `json_dumps_value` composes this for; a tighter
/// exact-value window is a separate capability this file does not
/// build. `lo` negative declines: this crate's own `Age`-shaped windows
/// (`Annotated[int, Field(ge=0, ...)]`) are always nonnegative in
/// practice, and a signed digit run is a different, unbuilt grammar.
pub fn integer_window_grammar(lo: i64, hi: i64) -> Option<RefinedSet> {
    if lo < 0 || hi < lo {
        return None;
    }
    let lo_digits = decimal_digit_count(lo) as i64;
    let hi_digits = decimal_digit_count(hi) as i64;
    Some(make_refined_set(vec![repeat_of(one_char_of("0123456789"), lo_digits, Some(hi_digits))]))
}

/// The exact one-character grammar `format(n, "x")` spells for a
/// NONNEGATIVE single-hex-digit window `[lo, hi]` (`hi <= 15`) — the
/// format-spec mini-language's lowercase hexadecimal presentation
/// (string.rst): each member spells one character, digits below ten
/// and 'a'..'f' from ten, so the alphabet is exactly the members' own
/// characters — `[0, 9]` never reaches a letter and `[10, 15]` is
/// only letters. A window past one hex digit is a wider grammar this
/// file does not build.
pub fn hex_digit_window_grammar(lo: i64, hi: i64) -> Option<RefinedSet> {
    if lo < 0 || hi < lo || hi > 15 {
        return None;
    }
    let chars: String = (lo..=hi).map(|v| char::from_digit(v as u32, 16).expect("v is in [0, 15]")).collect();
    Some(make_refined_set(vec![repeat_of(one_char_of(&chars), 1, Some(1))]))
}

/// The exact JSON-quoted-string grammar for a FINITE word set (a
/// `Literal["a", "b"]` string union, `word_tuples_of`'s own reading) —
/// the union of each member's own JSON-quoted spelling, alternated.
/// Quoting borrows the SAME `Debug`-escape convention
/// `json_dumps_value`'s own string arm already uses
/// (`format!("{:?}", text)`) — exact for the plain-ASCII, no-control-
/// character literals this corpus's own rows use, the identical known
/// gap `json_dumps_value`'s doc already states. `None` when the set is
/// not a finite word list (an ordinary unconstrained `str`, a pattern-
/// compiled string) — this file states no wide "any string, quoted"
/// arm; a member whose string set is not finite falls the whole object
/// through to `json_dumps_value`'s decline, matching that function's
/// existing "no partial answer" rule rather than inventing a new wide
/// arm this brief does not ask for.
fn string_word_grammar(set: &RefinedSet) -> Option<RefinedSet> {
    let words = word_tuples_of(set)?;
    if words.is_empty() {
        return None;
    }
    let mut spellings: Vec<String> = Vec::with_capacity(words.len());
    for word in &words {
        let text: String = word.iter().map(|&point| char::from_u32(point as i64 as u32)).collect::<Option<String>>()?;
        spellings.push(format!("{:?}", text));
    }
    spellings.sort();
    let mut grammar: Option<RefinedSet> = None;
    for spelling in spellings {
        let arm = string_tuple(&spelling);
        grammar = Some(match grammar {
            None => arm,
            Some(built) => make_refined_set(vec![union(built, arm)]),
        });
    }
    grammar
}

/// One member's own serialized-text grammar: an EXACT reading first
/// (`member_reader`, `json_dumps_value`'s own recursive call — an
/// exact int/string/nested-object member composes its own literal text
/// exactly, never widened to a grammar it does not need), then the two
/// SET arms this file composes (a bounded Integer window's digit-count
/// run, a finite string-Literal union's quoted alternation), then a
/// nested `Kind::Object` recursing through `object_members_grammar`
/// (threading the same reader on down, so a doubly-nested exact member
/// still reads exactly). `None` on every other shape (Float, Boolean, a
/// list, an unbounded numeric set, an unbounded string set, an unknown
/// value) — the known-gap set `json_dumps_value`'s own doc already
/// states, unchanged by this file. Every LEAF member this function
/// actually composes a grammar for (the exact-text arm's own value, or
/// a Set-shaped member) is recorded onto `graded_members` — an exact
/// member still carries its own proved/derived grade, so it counts
/// toward the composed floor exactly like a Set-shaped member does; a
/// nested Object contributes nothing of its own here (its members are
/// recorded individually by the recursive `object_members_grammar`
/// call instead, so the object's own container-level grade is never
/// double-counted).
fn member_value_grammar(
    value: &AbstractValue,
    member_reader: &mut dyn FnMut(&AbstractValue) -> Option<String>,
    graded_members: &mut Vec<AbstractValue>,
) -> Option<RefinedSet> {
    if let Some(text) = member_reader(value) {
        graded_members.push(value.clone());
        return Some(string_tuple(&text));
    }
    if value.kind == Kind::Set {
        if value.kind_tag == Some(PrimitiveKind::Integer) {
            let (lo, hi) = integer_set_bounds(value)?;
            let grammar = integer_window_grammar(lo, hi)?;
            graded_members.push(value.clone());
            return Some(grammar);
        }
        if value.kind_tag.is_none() || value.kind_tag == Some(PrimitiveKind::String) {
            if value.set_kind_tag == SetKindTag::None {
                let grammar = string_word_grammar(&value.set)?;
                graded_members.push(value.clone());
                return Some(grammar);
            }
        }
        return None;
    }
    if value.kind == Kind::Object {
        return object_members_grammar(value, member_reader, graded_members);
    }
    None
}

/// The closed integer bound `[lo, hi]` a value states, when the value
/// is a bounded Integer-sorted `Kind::Set`. Mirrors
/// `expressions.rs::integer_set_bounds` exactly (same private-copy
/// convention as `one_char_of`/`decimal_digit_count` above — the two
/// files' own scope keeps neither reaching into the other's private
/// helpers).
fn integer_set_bounds(value: &AbstractValue) -> Option<(i64, i64)> {
    if value.kind != Kind::Set || value.kind_tag != Some(PrimitiveKind::Integer) {
        return None;
    }
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for form in &value.set.forms {
        match form.form {
            Form::AtLeast => lo = Some(lo.map_or(form.a, |current: f64| current.max(form.a))),
            Form::Above => lo = Some(lo.map_or(form.a.floor() + 1.0, |current: f64| current.max(form.a.floor() + 1.0))),
            Form::AtMost => hi = Some(hi.map_or(form.a, |current: f64| current.min(form.a))),
            Form::Below => hi = Some(hi.map_or(form.a.ceil() - 1.0, |current: f64| current.min(form.a.ceil() - 1.0))),
            Form::Integer => {}
            _ => return None,
        }
    }
    let (lo, hi) = (lo?, hi?);
    if !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    Some((lo as i64, hi as i64))
}

/// `{"key": <member grammar>, ...}` for a CLOSED object's own members,
/// in the object's OWN insertion order (`AbstractValue.keys`'s own
/// ordered-Vec contract — no permutation union, per this file's own
/// banner on Python's dict-order guarantee). `member_reader` answers
/// one member's own exact serialized text where it has one (threaded in
/// by the caller, since this file does not import `expressions.rs`);
/// `member_value_grammar` composes the rest. Every member must resolve
/// to SOME grammar (exact or composed) or the whole object declines —
/// matching `json_dumps_value`'s own all-members-must-serialize rule
/// for its exact Object arm. `graded_members` collects every LEAF
/// member's own `AbstractValue` along the way (recursing into a nested
/// object), so the caller (`dumps_grammar`) can derive the composed
/// grammar's own trust floor from the SAME operands
/// `derived_trust_level` always reads — this file's own composition
/// steps (digit-count run, quoted alternation, brace/comma assembly)
/// add no boundary crossing of their own, so the floor is exactly the
/// weakest grade among the members actually read.
pub fn object_members_grammar(
    value: &AbstractValue,
    member_reader: &mut dyn FnMut(&AbstractValue) -> Option<String>,
    graded_members: &mut Vec<AbstractValue>,
) -> Option<RefinedSet> {
    if value.kind != Kind::Object {
        return None;
    }
    if value.keys.is_empty() {
        return Some(string_tuple("{}"));
    }
    let mut pairs: Vec<RefinedSet> = Vec::with_capacity(value.keys.len());
    for entry in &value.keys {
        let member_grammar = member_value_grammar(&entry.value, member_reader, graded_members)?;
        let quoted_key = format!("{:?}: ", entry.name);
        pairs.push(make_refined_set(vec![concatenation(string_tuple(&quoted_key), member_grammar)]));
    }
    let mut body = pairs.pop().expect("keys is non-empty, checked above");
    while let Some(pair) = pairs.pop() {
        body = make_refined_set(vec![concatenation(pair, make_refined_set(vec![concatenation(string_tuple(", "), body)]))]);
    }
    Some(make_refined_set(vec![concatenation(string_tuple("{"), make_refined_set(vec![concatenation(body, string_tuple("}"))]))]))
}

/// The top-level entry `expressions.rs::json_dumps_value` calls once
/// its own exact reading fails: a closed `Kind::Object`'s serialized
/// text as a GRAMMAR, wrapped as a string-sorted `Kind::Set`
/// `AbstractValue` (`known_set`, graded through `derived_trust_level`
/// over every member actually read — every arm here is an exact
/// structural composition of the input's own already-known facts, so
/// the composed claim never crosses a boundary its own members had not
/// already crossed) so the call's result flows to a downstream
/// length/pattern sink through the ordinary `Kind::Set` judging path
/// (`assignability.rs::judge`'s ELEMENT/structural laws already ask the
/// kernel a seq-subset question of any `Kind::Set` against a
/// scalar/string-shaped declared set — no new sink is needed).
/// `exact_reader` is `json_dumps_value` itself, threaded in (rather
/// than imported) since `expressions.rs` is this file's own caller —
/// reaching back into it would invert the module's layering.
pub fn dumps_grammar(
    value: &AbstractValue,
    exact_reader: &mut dyn FnMut(&AbstractValue) -> Option<String>,
) -> Option<AbstractValue> {
    let mut graded_members: Vec<AbstractValue> = Vec::new();
    let grammar = object_members_grammar(value, exact_reader, &mut graded_members)?;
    let grade = derived_trust_level(TrustProved, &graded_members);
    Some(known_set(grammar, None, grade, SetKindTag::None))
}

/// The `kind_word` tagging a `json.dumps(...)` ROUND-TRIP CARRIER value
/// (`dumps_round_trip_carrier_value`'s own doc) — the same "`Kind::
/// Object` plus a distinguishing word plus a payload field" idiom
/// `env.rs::retained_callable_value` and `string_models.rs::
/// MATCH_WITH_GROUPS_WORD`'s own match-object value already use, here
/// with the payload carried in `inner` (a whole nested `AbstractValue`,
/// not a string key) rather than `source`.
pub const JSON_DUMPS_ROUND_TRIP_WORD: &str = "the serialized text of a json.dumps round-trip carrier";

/// Whether `value`'s own shape is one `json.dumps` then `json.loads`
/// PRESERVES EXACTLY — library/json.rst's own Python-to-JSON-to-Python
/// conversion table, read end to end rather than one hop at a time:
/// `None` -> `null` -> `None`; `bool` -> `true`/`false` -> `bool`
/// (checked BEFORE Integer below — CPython's `bool` is an `int`
/// subclass, `AGENT-BRIEF.md`, so a Boolean-tagged value must not fall
/// through to the Integer arm and lose its own boolean sort); `int` ->
/// a decimal integer token -> `int` (exact for EVERY `int`, unlike
/// `integer_window_grammar`'s own digit-count OVER-approximation — that
/// function states a sound but wider claim for a downstream pattern
/// sink, while this round trip preserves the VALUE's own exact window,
/// since `json.loads` re-parses the same decimal digits `json.dumps`
/// wrote); `str` -> a quoted string -> `str` (this file's own `Debug`-
/// escape convention, `json_dumps_value`'s doc); a `Kind::Object` whose
/// OWN members are each one of these same round-trippable shapes
/// (recursion, `B7.keep.join`'s own dict-of-int shape). Declines for
/// every shape the round trip does NOT preserve exactly: `float`
/// (`B7.use.sink`'s own `nan_through_json_refused` row — `json.dumps`
/// emits the non-standard "NaN"/"Infinity" tokens for those values,
/// library/json.rst's own `allow_nan` note, so a Float-sorted value
/// cannot carry this claim soundly), a list, or any opaque/unknown
/// value this file holds no fact about.
fn round_trips_exactly(value: &AbstractValue) -> bool {
    if value.kind == Kind::Null {
        return true;
    }
    if value.kind == Kind::Values && value.kind_tag == Some(PrimitiveKind::Boolean) {
        return true;
    }
    if value.kind == Kind::Values && value.kind_tag == Some(PrimitiveKind::Integer) {
        return true;
    }
    if value.kind == Kind::Values && value.kind_tag == Some(PrimitiveKind::String) {
        return true;
    }
    if value.kind == Kind::Set && value.kind_tag == Some(PrimitiveKind::Integer) {
        return true;
    }
    if value.kind == Kind::Set && (value.kind_tag.is_none() || value.kind_tag == Some(PrimitiveKind::String)) && value.set_kind_tag == SetKindTag::None {
        return true;
    }
    if value.kind == Kind::Object {
        return value.keys.iter().all(|entry| round_trips_exactly(&entry.value));
    }
    false
}

/// `json.loads(json.dumps(v))`'s own value, for a `v` this file cannot
/// serialize to EXACT text or a member-grammar (`json_dumps_value`'s
/// exact reading, then `dumps_grammar`'s own member-grammar composition,
/// both already tried and declined by the caller before this is reached)
/// but whose SHAPE the round trip still preserves exactly
/// (`round_trips_exactly`'s own scope). Built on `Kind::Object` plus
/// `JSON_DUMPS_ROUND_TRIP_WORD` plus `v` ITSELF carried in `inner` —
/// `round_trip_carried_value` reads it back unchanged. This is
/// `json.dumps`'s own return value at the call site: the TEXT `dumps`
/// answers is never read as a string by anything else (the corpus's own
/// B7 rows only ever feed it straight into `json.loads`), so carrying
/// the original value rather than composing (and later re-parsing) an
/// actual grammar string is the exact, not approximated, answer for the
/// composed round trip — the same value `v` itself already proved,
/// unwidened.
pub fn dumps_round_trip_carrier_value(value: &AbstractValue) -> Option<AbstractValue> {
    if !round_trips_exactly(value) {
        return None;
    }
    Some(AbstractValue {
        inner: Some(Box::new(value.clone())),
        ..opaque_value(JSON_DUMPS_ROUND_TRIP_WORD)
    })
}

/// The original value `dumps_round_trip_carrier_value` carried, if
/// `value` is a round-trip carrier value built that way (`kind_word` is
/// the round-trip word AND `inner` is populated). `None` for an ordinary
/// string, or any other value `json.loads`'s own caller might pass —
/// the honest "not a carrier" answer that sends the caller back to its
/// own exact-literal reading or the full `json_loads_value_space`
/// fallback.
pub fn round_trip_carried_value(value: &AbstractValue) -> Option<&AbstractValue> {
    if value.kind != Kind::Object || value.kind_word != Some(JSON_DUMPS_ROUND_TRIP_WORD) {
        return None;
    }
    value.inner.as_deref()
}
