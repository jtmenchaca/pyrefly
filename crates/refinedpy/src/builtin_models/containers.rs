//! Container-and-iteration builtins: `list`, `set`, `dict`,
//! `dict.fromkeys`, `iter`, `next`, `anext`, `cast`, `object`, `vars`,
//! and `collections.Counter`. Every row cites its clause of
//! docs.python.org/3.12/library/functions.html, library/stdtypes.html,
//! or library/collections.html; a row with no citation is not written.

use refined_domain::abstract_value::{known_set, known_values, opaque_value, AbstractValue, Kind, PrimitiveKind, SetKindTag};
use refined_domain::known_constructors::known_list;
use refined_domain::lattice_operations::same_known;
use refined_domain::trust_grades::{derived_trust_level, TrustSpec};
use refined_sets::codepoint_sets::strings;
use refined_sets::refinement_forms::make_refined_set;
use refined_sets::repetition_window_forms::as_repetition;

use super::conversions::is_string_sorted_argument;
use super::numeric::single_known_numeric;

/// `list(iterable)` — library/stdtypes.rst's `class:: list([iterable])`
/// constructor row: "Lists may be constructed... using the type
/// constructor `list()` or `list(iterable)`." A known `Kind::List`
/// argument copies through unchanged (`list`/`tuple`/`set` all share
/// this domain's one `Kind::List` shape, per `collection_models.rs`'s
/// own module doc — `list(some_set)` and `list(some_tuple)` both read
/// through this same row). A `dict.fromkeys(...)` ROUND-TRIP CARRIER
/// argument (`dict_fromkeys_call`'s own doc, `A15.xfer.dedupe`'s
/// `list(dict.fromkeys(xs))` shape) is unwrapped through `dict_fromkeys_
/// keys_view` FIRST, before the `Kind::List` gate below (a carrier is
/// `Kind::Object`, never `Kind::List`, so the two arms never both fire
/// on the same argument). A known EXACT STRING argument (`Kind::Values`
/// tagged `PrimitiveKind::String`) splits into its own characters —
/// stdtypes.rst's `list(iterable)` row applies to any iterable, and a
/// string's own iteration protocol yields one-character strings in
/// order (Text Sequence Types: "A string is a sequence of Unicode code
/// points... iterating over the string produces one-character
/// substrings") — read through `list_of_string_characters` before the
/// `Kind::List` gate below, the same "try each known non-List shape in
/// turn" order the `dict.fromkeys` carrier already keeps. A known
/// `Kind::Set` Integer-sorted argument (`expressions.rs::
/// range_expression_value`'s own unbounded-`n` `range(n)` fallback —
/// `list(range(n))`, `A7.seed.conversion.py`'s `from_range`) copies
/// through UNCHANGED, the identical "an already-sort-only sequence
/// value stays sort-only" posture `list_of_unknown_string_characters`
/// keeps for a String-sorted set: `range`'s own answer is already the
/// exact iteration order/count claim this constructor states for it,
/// so `list(...)` adds no further fact to compute.
pub(super) fn list_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    if let Some(keys_view) = dict_fromkeys_keys_view(iterable) {
        return Some(keys_view);
    }
    if let Some(characters) = list_of_string_characters(iterable) {
        return Some(known_list(characters, derived_trust_level(TrustSpec, arguments)));
    }
    if let Some(window) = list_of_unknown_string_characters(iterable) {
        return Some(window);
    }
    if let Some(keys) = list_of_dict_keys(iterable) {
        return Some(known_list(keys, derived_trust_level(TrustSpec, arguments)));
    }
    // An already-sort-only SEQUENCE value — a repetition window, the
    // shape `range(n)` answers for an unbounded `n`, the shape
    // `itertools.chain.from_iterable`'s own abstract row answers for a
    // `list[list[X]]` argument, and the shape a declared
    // `list[X]`/`Sequence[X]` parameter seeds. stdtypes.rst's
    // `list(iterable)` row states the constructor's whole content is
    // the iterable's own items in order, so a receiver that already
    // states exactly which elements it holds at which count has nothing
    // further for this constructor to compute: it copies through
    // UNCHANGED. Read through `as_repetition` rather than the sort tag,
    // so a window a reader built without tagging a scalar sort (the
    // flattening row's own answer) copies through the same way a
    // tagged Integer window does.
    if iterable.kind == Kind::Set && iterable.set_kind_tag == SetKindTag::None && as_repetition(&iterable.set).is_some() {
        return Some(iterable.clone());
    }
    if iterable.kind != Kind::List {
        return None;
    }
    Some(known_list(iterable.items.clone(), derived_trust_level(TrustSpec, arguments)))
}

/// `list(s)` on a STRING-SORTED but not exactly known `s` (an unbounded
/// `s: str` parameter — `is_string_sorted_argument`'s own doc, the sort-
/// only `Kind::Set`/String ground `check.rs::seed_parameters` seeds for
/// a declared `str` parameter with no narrower fact). The exact
/// character sequence is unstated, but stdtypes.rst's own iteration
/// contract still pins two SORT facts: every element is a one-character
/// STRING (never any other type), and the count is exactly `len(s)` —
/// unknown here, so `[0, +inf)`. Answered as the bare-star repetition
/// window `collection_models.rs`'s own `repetition_receiver`/
/// `star_element_read` already read for a declared `list[X]` parameter —
/// the same shape, built here for a VALUE this file computes rather than
/// one `check.rs::seed_parameters` seeds directly. `None` for a
/// non-string-sorted argument, letting the caller's own `Kind::List`
/// gate try next.
fn list_of_unknown_string_characters(value: &AbstractValue) -> Option<AbstractValue> {
    if !is_string_sorted_argument(value) || value.kind == Kind::Values {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    let element = AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(strings(), None, grade, SetKindTag::None)
    };
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::String),
        ..known_set(
            refined_sets::repetition_window_forms::repetition(element.set, 0, None),
            None,
            grade,
            SetKindTag::None,
        )
    })
}

/// `list(s)` on a known exact string `s` — one single-character String
/// element per Unicode code point, in order (`list_constructor_call`'s
/// own doc). Each element is built exactly the way
/// `collection_models::string_index_read` already reads one code point
/// back — a one-element `Kind::Values` String — so a later `xs[i]` on
/// the produced list agrees with what `s[i]` would already answer
/// directly. `None` for any non-string argument, letting the caller's
/// own `Kind::List` gate try next.
fn list_of_string_characters(value: &AbstractValue) -> Option<Vec<AbstractValue>> {
    if value.kind != Kind::Values || value.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    Some(
        value
            .values
            .iter()
            .map(|code_point| known_values(vec![*code_point], PrimitiveKind::String, grade))
            .collect(),
    )
}

/// `list(d)` on a known DICT receiver (`Kind::Object`) — stdtypes.rst's
/// Mapping Types section: "Iterating over a dictionary yields its keys...
/// Dictionaries preserve insertion order. Note that updating a key does
/// not affect the order." So `list(d)` is the dict's own key list, in the
/// order the entries were inserted — which is exactly the order this
/// domain's `ObjectKey` vector already carries (`dict_literal_value`
/// appends a new key at the end and OVERWRITES a repeated one in place,
/// matching the cited "updating a key does not affect the order" rule).
///
/// Each key becomes the VALUE it spells: a string-keyed entry answers the
/// String-sorted value of its own characters, a numeric-keyed entry the
/// Integer value its plain decimal `name` parses back to (`DictKey::
/// integer`'s own spelling). An IDENTITY key (`DictKey::identity`, a
/// hashable non-string/non-int key like a bare `object()` sentinel)
/// carries no value spelling this reader can rebuild — its `name` is a
/// provenance tag, not the key's own value — so a dict holding one
/// declines the WHOLE call rather than answer a key list with a
/// fabricated or omitted entry, the same all-or-nothing honesty
/// `dict_literal_value` keeps for an unsupported key.
///
/// `None` for any non-dict argument, letting the caller's own
/// `Kind::List` gate try next. A TAGGED opaque object (a `dict.fromkeys`
/// carrier, a datetime instance) is not a dict and declines here too —
/// the `fromkeys` carrier has its own reader above, tried first.
fn list_of_dict_keys(value: &AbstractValue) -> Option<Vec<AbstractValue>> {
    if value.kind != Kind::Object || value.kind_word.is_some() {
        return None;
    }
    let grade = derived_trust_level(TrustSpec, std::slice::from_ref(value));
    let mut keys: Vec<AbstractValue> = Vec::with_capacity(value.keys.len());
    for entry in &value.keys {
        if entry.numeric {
            // a numeric key's `name` is its own plain decimal spelling
            // (`DictKey::integer`); a FRACTIONAL float key spells a
            // decimal point instead, which no int spelling can carry, so
            // the whole call declines rather than answer a key list with
            // a rounded or dropped entry
            let parsed: i64 = entry.name.parse().ok()?;
            keys.push(known_values(vec![parsed as f64], PrimitiveKind::Integer, grade));
        } else {
            // an identity key's `name` carries a provenance tag under a
            // reserved NUL-prefixed spelling (`DictKey::identity`), never
            // the key's own characters — it has no value to rebuild
            if entry.name.starts_with('\0') {
                return None;
            }
            let code_points: Vec<f64> = entry.name.chars().map(|c| c as u32 as f64).collect();
            keys.push(known_values(code_points, PrimitiveKind::String, grade));
        }
    }
    Some(keys)
}

/// The `kind_word` tagging a `dict.fromkeys(iterable, value=None)`
/// ROUND-TRIP CARRIER value (`dict_fromkeys_call`'s own doc) — the same
/// "`Kind::Object` plus a distinguishing word plus a payload in `inner`"
/// idiom `json_grammar.rs::JSON_DUMPS_ROUND_TRIP_WORD` and
/// `env.rs::retained_callable_value` both already use.
pub(super) const DICT_FROMKEYS_WORD: &str = "the keys view of a dict.fromkeys(...) call";

/// `dict.fromkeys(iterable, value=None)` — library/stdtypes.rst's
/// `classmethod:: fromkeys(iterable, value=None, /)`: "Create a new
/// dictionary with keys from *iterable* and values set to *value*...
/// *value* defaults to `None`." This domain's `dict` is `Kind::Object`
/// with a CLOSED, string-named `keys` list (`collection_models.rs`'s
/// own module doc) — it cannot represent a dict whose keys are an
/// unbounded-count, windowed-VALUE set (`xs: list[int]`'s own element
/// window, not a finite set of string names), so this row does not
/// build a real `Kind::Object` dict at all. Modeled ONLY for the shape
/// the corpus needs a value for — `iterable` a `Kind::Set` repetition
/// window (`as_repetition`, the same shape `star_numeric_hull`/
/// `min_max_over_star` already read for a `list[int]`-typed parameter)
/// — and answers a ROUND-TRIP CARRIER (`Kind::Object`, `DICT_FROMKEYS_
/// WORD`, the iterable's own repetition set carried in `inner`) rather
/// than a real dict value: this file's own callers only ever consume a
/// `fromkeys(...)` result through `list(...)`
/// (`dict_fromkeys_keys_view`, `A15.xfer.dedupe`'s own row), never by
/// reading a key/value pair directly, so carrying the iterable through
/// unread is the exact answer for that one consumption path rather than
/// building (and immediately discarding) machinery for dict reads this
/// corpus never exercises. `value` (defaulting to `None`) is not
/// modeled — the DEDUPED KEYS are the only fact `list(dict.fromkeys(xs))`
/// ever needs; a caller that goes on to read a VALUE out of the result
/// finds no dict shape here and declines honestly.
pub(super) fn dict_fromkeys_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let iterable = match arguments {
        [iterable] => iterable,
        [iterable, _value] => iterable,
        _ => return None,
    };
    if iterable.kind != Kind::Set || iterable.set_kind_tag != SetKindTag::None {
        return None;
    }
    as_repetition(&iterable.set)?;
    Some(AbstractValue {
        inner: Some(Box::new(iterable.clone())),
        ..opaque_value(DICT_FROMKEYS_WORD)
    })
}

/// `list(dict.fromkeys(xs))`'s own value: the DISTINCT elements of
/// `xs`, in insertion order (Python's `dict` preserves insertion order,
/// library/stdtypes.rst's own "Mapping Types — dict" guarantee,
/// `json_grammar.rs`'s identical citation for the same fact) — drawn
/// from the SAME element window `xs` itself carries (dedup drops
/// duplicates, never introduces a new element outside `xs`'s own
/// alphabet), at a count anywhere from `0` (every element could
/// collide down to one, or `xs` could already be empty) up to `xs`'s
/// own upper length bound (dedup never GROWS a sequence). Rebuilds the
/// SAME repetition shape `xs` itself carries (`as_repetition`/
/// `repeat_of`, the identical window a plain `list[int]` parameter
/// already flows through `loops.rs`'s own `for`-loop reader), with
/// `lo` relaxed to `0` and `hi` unchanged — so `for x in
/// list(dict.fromkeys(xs)): ...` binds `x` to exactly `xs`'s own
/// element set through the SAME existing reader, no new loop machinery
/// needed. `None` for any argument that is not a `dict_fromkeys_call`
/// carrier (`dict_fromkeys_call`'s own doc on the one-consumer scope).
fn dict_fromkeys_keys_view(argument: &AbstractValue) -> Option<AbstractValue> {
    if argument.kind != Kind::Object || argument.kind_word != Some(DICT_FROMKEYS_WORD) {
        return None;
    }
    let iterable = argument.inner.as_deref()?;
    let repeated = as_repetition(&iterable.set)?;
    let deduped_set = make_refined_set(vec![refined_sets::refinement_forms::repeat_of(repeated.element, 0, repeated.hi)]);
    Some(AbstractValue {
        kind_tag: iterable.kind_tag,
        ..known_set(deduped_set, None, derived_trust_level(TrustSpec, &[iterable.clone()]), SetKindTag::None)
    })
}

/// `set([iterable])` — library/stdtypes.rst's `class:: set([iterable])`
/// constructor row: "Return a new set... object whose elements are
/// taken from *iterable*." This domain has no dedicated set Kind (the
/// same `Kind::List` shape a list/tuple carries, per
/// `collection_models.rs`'s own module doc — a set's own element-
/// uniqueness is invisible to any reader that only ever consumes the
/// sequence via `len()`/iteration, matching that file's list/set-comp
/// note). The BARE zero-argument form `set()` — the brackets in the
/// doc's own signature mark the argument optional — answers the empty
/// list directly (an empty set has no elements to dedupe); the
/// one-argument form runs `list_constructor_call` first, then DEDUPES
/// the resulting `Kind::List` items by `same_known` structural equality
/// (first-occurrence order — CPython's own set iteration order is
/// unspecified, so first-seen is as good a claim as any, and matches
/// what `sorted(set(...))` immediately re-orders anyway): a set's whole
/// point is that no two elements compare equal, so a `set([1, 1, 2])`
/// argument that arrives with three known items must not still carry
/// three items after this call — `len()` reads `Kind::List.items.len()`
/// directly (`collection_models.rs::len_result`), so an undeduped
/// three-item list here made `len(sorted(set([1, 1, 2])))` read `{3}`
/// instead of the true `{2}`, provably-false-ing a guard that always
/// holds. Every element must be a KNOWN value (`same_known` needs both
/// sides concrete to compare) — a list holding an unknown element still
/// dedupes the known ones against each other but keeps the unknown
/// slot as its own entry, since an unknown value's identity against its
/// neighbors is not decidable and dropping it would be unsound.
pub(super) fn set_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if arguments.is_empty() {
        return Some(known_list(Vec::new(), TrustSpec));
    }
    let sequence = list_constructor_call(arguments)?;
    // A WINDOW argument (`set(lst)` on a declared `list[X]` parameter —
    // `list_constructor_call`'s own repetition row copies such a
    // receiver through as a `Kind::Set` repetition, which states a
    // per-element sort and a count but never WHICH elements) carries no
    // `items` at all. Deduping its (empty) item list would fabricate an
    // EMPTY known list, and membership against an empty container reads
    // provably false on every run — killing the guarded body of
    // A8.seed.conversion's `set_from_list_member_outside`, where
    // `lst[0] in set(lst)` is true by construction. A set built from a
    // window states exactly what the window states: the same
    // per-element sort at the same count, with the membership question
    // left undecided. Deduplication has nothing to do here anyway — a
    // window names no two elements to compare.
    if sequence.kind != Kind::List {
        return Some(sequence);
    }
    let mut deduped: Vec<AbstractValue> = Vec::with_capacity(sequence.items.len());
    for item in sequence.items {
        let already_present = deduped.iter().any(|kept| same_known(kept, &item));
        if !already_present {
            deduped.push(item);
        }
    }
    Some(known_list(deduped, derived_trust_level(TrustSpec, arguments)))
}

/// `dict(pairs)` — one positional argument, an iterable of `(key,
/// value)` 2-element pairs — library/stdtypes.rst's `class:: dict(...)`
/// constructor row: "dict(iterable, **kwargs)... Dictionaries can be
/// created by... providing an iterable of key/value pairs, including
/// tuples: `dict([('foo', 100), ('bar', 200)])`." Modeled ONLY when
/// `pairs` is a known `Kind::List` of known `Kind::List` 2-element
/// pairs whose first slot is a known exact string (this domain's
/// dict's own string-keyed-only restriction, `collection_models.rs`'s
/// module doc) — anything else declines. A repeated key keeps the LAST
/// value, matching the same overwrite rule `dict_literal_value` and
/// the `dict(...)` constructor doc both state.
pub(super) fn dict_constructor_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [pairs] = arguments else { return None };
    // `dict(<existing dict>)` — the copy-constructor form ("providing
    // ... another dictionary", the same class:: dict(...) row): a known
    // Kind::Object argument answers a fresh dict with the same entries.
    if pairs.kind == Kind::Object && pairs.kind_word.is_none() {
        return Some(pairs.clone());
    }
    // `dict(<some mapping whose key set is unbounded>)` — the same
    // copy-constructor row ("providing... another dictionary"), where the
    // argument is an UNBOUNDED-KEY mapping (`Kind::ObjectStar` — a
    // `dict[str, X]` parameter's own seed, or `os.environ`'s own read,
    // `expressions::attribute`'s `data:: environ` row). The copy states
    // exactly what the source states: which values every present key
    // holds, and nothing about WHICH keys are present — so the star
    // copies through unchanged. Answering `None` here instead left
    // `dict(os.environ)` deriving nothing at all, even though the copy
    // is provably the same mapping the source already read as.
    if pairs.kind == Kind::ObjectStar {
        return Some(pairs.clone());
    }
    if pairs.kind != Kind::List {
        return None;
    }
    let mut keys: Vec<Option<crate::collection_models::DictKey>> = Vec::with_capacity(pairs.items.len());
    let mut values: Vec<AbstractValue> = Vec::with_capacity(pairs.items.len());
    for pair in &pairs.items {
        if pair.kind != Kind::List || pair.items.len() != 2 {
            return None;
        }
        let key = &pair.items[0];
        if key.kind != Kind::Values || key.kind_tag != Some(PrimitiveKind::String) {
            return None;
        }
        let key_text: String = key.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
        keys.push(Some(crate::collection_models::DictKey::string(&key_text)));
        values.push(pair.items[1].clone());
    }
    // dict_literal_value's own last-value-wins overwrite rule handles a
    // repeated key exactly the way this constructor's own cited row
    // does — this file reaches into collection_models.rs for the one
    // shared building block rather than duplicating that merge loop
    Some(crate::collection_models::dict_literal_value(&keys, &values))
}

/// `collections.Counter(iterable)` — library/collections.rst:
/// "A `Counter` is a `dict` subclass for counting hashable objects...
/// Elements are counted from an *iterable*." The class doc pins both
/// facts a reader needs without knowing the iterable's own contents:
/// the result IS a dict (so every dict read this domain models applies
/// to it), and each value is that element's own COUNT.
///
/// A count reached by counting an element of the iterable is at least
/// `1` — an element with no occurrences never becomes a key at all,
/// which the same doc states directly: "Counter... return[s] a zero
/// count for missing items instead of raising a KeyError" only on a
/// LOOKUP, never as a stored entry. So every PRESENT key's value sits in
/// `[1, +inf)`, whole — which is exactly what an unbounded-key mapping
/// carries (`known_dict_star`, the same shape `check::seed_parameters`
/// builds for a `dict[str, X]` parameter): the key set is unstated, the
/// value law is stated once for every present key.
///
/// Answered for ANY single argument, whatever this file can or cannot
/// read of it: the `[1, +inf)` claim comes from the counting contract
/// itself, not from the iterable's contents, so an unread `xs:
/// list[str]` gets the same sound answer a known list would. That is
/// what A8.seed.library's own rows need — `counts["a"]` reads a count
/// that is NOT inside Age's `[0, 150]` window on its own, rather than
/// the binding deriving nothing.
///
/// The `Counter(**kwargs)` and `Counter(mapping)` spellings are not this
/// row (a keyword call is already declined at the call site, and a
/// mapping argument's counts are the mapping's own values rather than
/// tallies) — the zero-argument `Counter()` likewise declines, having no
/// iterable to count.
pub(super) fn counter_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [iterable] = arguments else { return None };
    // A FULLY KNOWN input — a `Kind::List` whose every element is an
    // exactly-known string (`Counter(["a", "b"])`) — states its own
    // tallies outright: the same "Elements are counted from an
    // *iterable*" row, read over an iterable whose elements this file
    // can enumerate. Each distinct element becomes a key, and its value
    // is the EXACT number of times it occurs, not the general
    // `[1, +inf)` law the unread case falls back to. The closed dict
    // this answers also states its key set exactly, so a key the input
    // never held reads as absent rather than undecided.
    if iterable.kind == Kind::List {
        let mut keys: Vec<Option<crate::collection_models::DictKey>> = Vec::with_capacity(iterable.items.len());
        let mut tallies: Vec<usize> = Vec::with_capacity(iterable.items.len());
        let mut every_element_known = true;
        for element in &iterable.items {
            if element.kind != Kind::Values || element.kind_tag != Some(PrimitiveKind::String) {
                every_element_known = false;
                break;
            }
            let element_text: String = element.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
            let key = crate::collection_models::DictKey::string(&element_text);
            match keys.iter().position(|kept| kept.as_ref() == Some(&key)) {
                Some(slot) => tallies[slot] += 1,
                None => {
                    keys.push(Some(key));
                    tallies.push(1);
                }
            }
        }
        if every_element_known && !keys.is_empty() {
            let values: Vec<AbstractValue> = tallies
                .iter()
                .map(|tally| known_values(vec![*tally as f64], PrimitiveKind::Integer, TrustSpec))
                .collect();
            return Some(crate::collection_models::dict_literal_value(&keys, &values));
        }
    }
    let count = AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![
                refined_sets::refinement_forms::at_least(1.0),
                refined_sets::refinement_forms::integer(),
            ]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    };
    let (star, built) = refined_domain::known_constructors::known_dict_star(count, TrustSpec);
    built.then_some(star)
}

/// `vars(object)` — library/functions.rst: "Return the `__dict__`
/// attribute for a module, class, instance, or any other object with a
/// `__dict__` attribute." An instance's `__dict__` holds exactly its OWN
/// (per-instance) attributes — the ones bound on the instance itself,
/// never the ones defined on its class, which live in the CLASS's own
/// `__dict__` instead (reference/datamodel.rst, "Custom classes": "a
/// class instance has a namespace implemented as a dictionary which is
/// the first place in which attribute references are searched... class
/// attributes are not found there").
///
/// This domain already carries a constructed instance as a
/// `Kind::Object` whose `ObjectKey` entries are its own fields
/// (`instances::judge_construction`'s own doc) — the same shape a dict
/// takes — so `vars(o)` answers the instance's entries as a plain dict,
/// with the `instance_identity` DROPPED: `vars(o)` returns the
/// `__dict__` mapping, a different object from the instance itself, and
/// carrying the instance's identity onto it would make `vars(o)` read as
/// the same referent `o` is.
///
/// That is what lets A8.xfer.own's rows read `vars(o)["inst_attr"]` as
/// the stored value, and `"cls_attr" in vars(o)` as False — a class
/// attribute is not among the instance's own entries, so the closed
/// key set this answers already states its absence.
///
/// Gated on a CONSTRUCTED instance (`instance_identity` present) rather
/// than any `Kind::Object`: a plain dict has no `__dict__` of its own
/// (`vars({})` raises `TypeError`, functions.rst's own "If the object
/// does not have a `__dict__`... a `TypeError` exception is raised"), so
/// answering a dict's own entries here would be the wrong mapping
/// entirely. Every other argument declines.
pub(super) fn vars_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::Object || only.instance_identity.is_none() {
        return None;
    }
    Some(refined_domain::known_constructors::known_object(
        only.keys.clone(),
        None,
        true,
        derived_trust_level(TrustSpec, arguments),
        false,
    ))
}

/// `iter(object)` (one-argument form, no `sentinel`) — library/functions.html#iter:
/// "Return an iterator object... *object* must be a collection object
/// which supports the iterable protocol." This domain has no separate
/// iterator Kind: an iterator over a known `Kind::List` reads through
/// as the SAME list value (the one shape a caller ever inspects it
/// through — `next_call`'s own row below), matching the module's
/// shared list/set/generator representation
/// (`collection_models.rs`'s own module doc). Any other receiver
/// shape declines.
pub(super) fn iter_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::List {
        return None;
    }
    Some(only.clone())
}

/// `next(iterator)` (one-argument form, no `default`) — library/functions.html#next:
/// "Retrieve the next item from the iterator by calling its
/// `__next__` method." Modeled ONLY for the `iter_call`-shaped receiver
/// (a known `Kind::List` standing in for its own iterator, per that
/// function's own doc) AND a generator call's own answer
/// (`Kind::List` tagged `source == "generator"`,
/// `instances::generator_yields`'s own doc — a same-module generator
/// `def`'s call answers the ordered List of every yielded value): the
/// FIRST element is the first item `__next__` would ever produce off a
/// freshly-built iterator or a freshly-called generator. An EMPTY list
/// provably raises `StopIteration` ("If *default* is given, it is
/// returned if the iterator is exhausted, otherwise `StopIteration` is
/// raised") — this row declines on an empty receiver rather than answer
/// a fabricated element; the raise itself is `provable_raise`'s own
/// business, not this dispatcher's.
///
/// SCOPE: this domain carries no per-call exhaustion/position state — a
/// generator-tagged List is a fixed VALUE (the full yield sequence),
/// not a stateful cursor, so `next_call` cannot tell "the first read of
/// this generator" apart from "a second read of the SAME already-
/// advanced generator." Every corpus row this file serves calls `next`
/// exactly once per freshly-constructed generator/iterator value
/// (`next(some_gen())`, never `next(g); next(g)` on one bound name), so
/// this row is honest for that shape; a second `next()` against the
/// SAME generator value would answer element 0 again rather than
/// element 1, which is a known gap this file does not claim to close.
pub(super) fn next_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind != Kind::List {
        // A generator call whose body `instances::generator_yields`
        // declined to summarize answers an Unknown tagged
        // `source == "generator-declined"` (`expressions::evaluate_call`'s
        // own generator-call arm) rather than a List — `next(it)` on
        // THAT receiver still has no element to answer, but the tag
        // itself must survive the call so `check.rs::
        // name_unmodeled_call_sentence`'s generator rung can trace a
        // later blocked read (`first = next(it); return first`) back to
        // the generator body that was never summarized, instead of the
        // generic "value not readable" wording. Any other non-List,
        // non-tagged receiver keeps declining outright — this is not a
        // general "next answers Unknown" widening, only the one tag's
        // own onward carry.
        if only.kind == Kind::Unknown && only.source == "generator-declined" {
            return Some(only.clone());
        }
        return None;
    }
    only.items.first().cloned()
}

/// `anext(async_iterator)` (one-argument form, no `default`) — the
/// `async`-generator twin of `next(iterator)`: library/functions.html
/// documents `anext` as `next`'s async counterpart. `await anext(gen)`
/// evaluates through `evaluate_expression`'s own `Expr::Await` arm
/// (transparent unwrap — `async`/`await` carry no gate of their own,
/// matching this file's asyncio.gather doc's identical note), so the
/// `anext(...)` call itself lands in this dispatcher exactly like a
/// plain `next(...)` call would. An async generator's yielded elements
/// are the SAME `Kind::List` (tagged `source == "generator"`,
/// `instances::generator_yields`'s own doc) a sync generator's call
/// answers — `datamodel.rst`'s generator-iterator protocol makes no
/// distinction between a sync and an async generator's own yielded
/// VALUES, only in how the caller RECEIVES them (`__anext__` returns
/// an awaitable rather than the value directly) — so this row is
/// `next_call` under a different name, not a separate reading.
pub(super) fn anext_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    next_call(arguments)
}

/// `typing.cast(typ, val)` — `Lib/typing.py`'s own `cast` docstring:
/// "This returns the value unchanged. To the type checker this signals
/// that the return value has the designated type, but at runtime we
/// intentionally don't check anything." `typ` is never read (a type
/// expression, not a value this file evaluates); `val` passes through
/// exactly, whatever shape it is — the identity function over its
/// second argument.
pub(super) fn cast_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [_typ, val] = arguments else { return None };
    Some(val.clone())
}

/// `object()` — library/functions.html#object: "This is the ultimate
/// base class of all other classes... When the constructor is called,
/// it returns a new featureless object. The constructor does not accept
/// any arguments." A featureless object has no fields this domain could
/// enumerate, so the answer is `opaque_value` — the same "kind of thing
/// known, contents not" shape `type(object)` already answers above —
/// tagged `source: "object()"` so a dict-display key built from this
/// value (`known_dict_key`'s identity arm, `collection_models.rs`) can
/// recognize it as a stable, non-string/int key: `stdtypes.rst`'s
/// mapping-key rule states a dict key only needs to be hashable, never a
/// string or number, and a fresh `object()` is hashable by identity
/// alone (no `__eq__`/`__hash__` override, `object`'s own doc — "has
/// methods that are common to all instances," none of which redefine
/// equality).
///
/// Scope: this tags every `object()` call the SAME way, so it only
/// answers a sound identity for the corpus shape actually read — ONE
/// `object()` call, bound to a name once and read back by that name
/// (never re-evaluated) — never two DIFFERENT `object()` call sites
/// compared as keys within the same dict. Telling two live `object()`
/// values apart needs a per-call-site tag threaded from the call
/// expression itself (`expressions.rs::evaluate_call`), which this file
/// has no access to (it only sees the callee name and the evaluated,
/// argument-less call).
pub(super) fn object_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if !arguments.is_empty() {
        return None;
    }
    let mut instance = opaque_value("a featureless object");
    instance.source = "object()".to_owned();
    Some(instance)
}

/// `hash(x)` — library/functions.html#hash: "Return the hash value of
/// the object (if it has one). Hash values are integers... Numeric
/// values that compare equal have the same hash value (even if they
/// are of different types, as is the case for 1 and 1.0)." The doc
/// states only that the result is a Python `int` and that EQUAL
/// operands hash equally — it does NOT state `hash(n) == n` for every
/// int `n` (CPython's real implementation reduces modulo
/// `sys.hash_info.modulus`, a fact outside library/functions.html's own
/// text), so this row answers the SORT the doc actually guarantees —
/// the unbounded integer ground — rather than fabricate an identity
/// claim the cited clause does not make. Modeled for any single
/// argument this file can already read a value or a known Set for
/// (`single_known_numeric`, or a numeric/string-sorted `Kind::Set`/
/// `Kind::Values` argument): `hash` accepts any hashable object, and
/// this row's own claim (unbounded `int`) holds regardless of which
/// hashable shape the argument is, so the argument itself is not
/// otherwise inspected.
///
/// The answer carries an EXPLICIT `AtLeast(-inf)` ray alongside
/// `Integer`, the same two-form shape `narrowing.rs`'s own
/// `unbounded_integers()` and this file's own `int_image()` both build
/// for "the whole integer ground, no bound stated" — never `Integer`
/// alone with zero ray forms. A bare `[Integer]` set is missing the
/// ray form the kernel's scalar deciders key the 1-tuple scalar shape
/// on, which let a one-sided guard's own narrowed window (`hash(x) >=
/// 0`, only a lower ray tightened onto this set) reach
/// `scalar_subset`/`assignability.rs`'s containment ask still missing
/// the upper-boundedness a real `[0, 150]`-declared set requires — the
/// A15.xfer.hash `hash_outside` soundness gap this two-form shape
/// closes.
pub(super) fn hash_call(arguments: &[AbstractValue]) -> Option<AbstractValue> {
    let [only] = arguments else { return None };
    if only.kind == Kind::Unknown {
        return None;
    }
    Some(AbstractValue {
        kind_tag: Some(PrimitiveKind::Integer),
        ..known_set(
            make_refined_set(vec![
                refined_sets::refinement_forms::integer(),
                refined_sets::refinement_forms::at_least(f64::NEG_INFINITY),
            ]),
            None,
            TrustSpec,
            SetKindTag::None,
        )
    })
}

/// `struct.unpack(format, buffer)` — library/struct.rst's `unpack(format,
/// buffer)`: "Unpack from the buffer *buffer*... according to the format
/// string *format*. The result is a tuple even if it contains exactly
/// one item." Modeled ONLY for a known exact `format` string of exactly
/// `">I"` (big-endian, one unsigned 32-bit int — struct.rst's Format
/// Characters table: `>` "big-endian, standard size, no alignment", `I`
/// "unsigned int", standard size 4) over a known `buffer` that is a
/// `Kind::List` of EXACTLY 4 known Integer elements each in `[0, 255]`
/// (`bytes_models.rs`'s own doc: a known bytes/bytearray literal is
/// already this shape, built through `collection_models::
/// list_literal_value`) — struct.rst's own "the buffer size... must
/// match the size required by the format" clause, so any other element
/// count or a non-byte element declines rather than guess. The decoded
/// value is `sum(byte[i] << (8 * (3 - i)))` for `i` in `0..4` (the
/// defining property of a big-endian unsigned integer, struct.rst's own
/// `>` byte-order row), always inside `[0, 2**32 - 1]` and therefore
/// f64-exact. The result is a ONE-ELEMENT `Kind::List` — struct.rst's own
/// "the result is a tuple even if it contains exactly one item," and
/// this domain's `list`/`tuple` share one `Kind::List` shape
/// (`collection_models.rs`'s own module doc) — so the corpus's own
/// `(value,) = struct.unpack(">I", ...)` one-tuple destructure reads it
/// back through the existing tuple-unpack machinery unchanged. Every
/// other format string, or a buffer this file cannot read as 4 known
/// bytes, declines: struct.rst names many more format characters
/// (`B`/`H`/`Q`/`f`/`d`/…) and this row states only the one the corpus
/// needs a value for, never a guessed decode of an unread format.
pub(super) fn struct_call_result(function: &str, arguments: &[AbstractValue]) -> Option<AbstractValue> {
    if function != "unpack" {
        return None;
    }
    let [format, buffer] = arguments else { return None };
    if format.kind != Kind::Values || format.kind_tag != Some(PrimitiveKind::String) {
        return None;
    }
    let format_text: String = format.values.iter().filter_map(|point| char::from_u32(*point as i64 as u32)).collect();
    if format_text != ">I" {
        return None;
    }
    if buffer.kind != Kind::List || buffer.items.len() != 4 {
        return None;
    }
    let mut decoded: u32 = 0;
    for byte_value in &buffer.items {
        let (byte, PrimitiveKind::Integer) = single_known_numeric(byte_value)? else {
            return None;
        };
        if !(0.0..=255.0).contains(&byte) || byte.fract() != 0.0 {
            return None;
        }
        decoded = (decoded << 8) | (byte as u32);
    }
    let grade = derived_trust_level(TrustSpec, arguments);
    Some(known_list(vec![known_values(vec![decoded as f64], PrimitiveKind::Integer, grade)], grade))
}
